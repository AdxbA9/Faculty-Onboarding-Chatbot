"""
End-to-end question-answering pipeline.

Flow:
    question
        -> classify_query
        -> greeting? return canned reply
        -> embed + FAISS + lexical -> candidates
        -> cross-encoder rerank
        -> low-confidence? refuse
        -> try deterministic extractor (contact/date/count)
        -> else prompt LLM, parse + verify answer
        -> return structured result

Public entry point: :func:`answer_question`.

Light timing instrumentation is included so the UI's developer-mode
panel can show retrieval / rerank / generation breakdowns. It adds only
a handful of ``time.perf_counter()`` calls and does not change logic.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import (
    EVIDENCE_PREVIEW_CHARS,
    FINAL_K,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MIN_RERANK_SCORE,
)
from .extractors import (
    extract_contact_answer,
    extract_count_answer,
    extract_date_answer,
)
from .retrieval import (
    classify_query,
    deduplicate_by_text,
    gather_candidates,
)
from .text_utils import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    normalize_text,
    tokenize,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class QAResult:
    """Structured answer returned to callers."""
    answer: str
    pages: List[int] = field(default_factory=list)
    best_section: str = ""
    evidence: str = ""
    query_type: str = "policy"
    items: List[Dict] = field(default_factory=list)
    used_llm: bool = False

    # Timing + diagnostics for the developer-mode debug panel.
    timings: Dict[str, float] = field(default_factory=dict)
    num_candidates: int = 0
    num_reranked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "pages": self.pages,
            "best_section": self.best_section,
            "evidence": self.evidence,
            "query_type": self.query_type,
            "items": self.items,
            "used_llm": self.used_llm,
            "timings": self.timings,
            "num_candidates": self.num_candidates,
            "num_reranked": self.num_reranked,
        }


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a cosine-similarity FAISS index (inputs are already normalised)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
_INTENT_GUIDANCE = {
    "contact": "Return the exact phone, fax, or email only if it is clearly tied to the requested office.",
    "date": "For calendar questions, return the exact event/date wording from the source. Do not guess the semester.",
    "count": "For count questions, return all requested categories exactly if they are present.",
    "policy_yesno": "For yes/no questions, answer only if the context directly addresses that exact policy point.",
    "list": "For list questions, enumerate only items explicitly present in the context.",
}


def build_prompt(question: str, items: List[Dict], query_type: str) -> str:
    """Assemble the grounded prompt sent to the LLM."""
    context_parts = []
    for i, item in enumerate(items, 1):
        meta = item["meta"]
        label = f"Source {i} | Page {meta['page']} | Type {meta.get('chunk_type', 'paragraph')}"
        if meta.get("section"):
            label += f" | Section {meta['section']}"
        context_parts.append(f"{label}\n{item['chunk']}")
    context = "\n\n".join(context_parts)

    extra = _INTENT_GUIDANCE.get(query_type, "")

    return f"""You answer questions about the University of Sharjah Faculty Handbook.
Use ONLY the supplied context.

Rules:
1. Give a direct, specific answer in one or two short sentences.
2. If the answer is not clearly supported by the context, reply exactly: I do not have this information.
3. Do not guess. Do not combine unrelated rows or pages.
4. {extra}
5. After the answer, add a new line exactly like this: Pages: page_numbers_only
6. Only cite pages from the supplied context.

Context:
{context}

Question: {question}

Answer:""".strip()


# ---------------------------------------------------------------------------
# LLM call + parsing
# ---------------------------------------------------------------------------
def ask_groq(client, prompt: str) -> str:
    """Call the Groq chat completion endpoint and return the raw text."""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer accurately from supplied context only. "
                    "Be conservative. Never invent facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def parse_answer_and_pages(raw: str, items: List[Dict]) -> Tuple[str, List[int]]:
    """Split the LLM output into (answer_body, cited_pages).

    Only pages that appear in the supplied context are accepted - a cheap
    but effective guard against hallucinated citations.
    """
    allowed = sorted({int(item["meta"]["page"]) for item in items})
    pages: List[int] = []
    m = re.search(r"Pages:\s*([0-9,\-\s]+)", raw, re.I)
    if m:
        for n in re.findall(r"\d+", m.group(1)):
            val = int(n)
            if val in allowed and val not in pages:
                pages.append(val)

    body = re.sub(r"\n?Pages:\s*[0-9,\-\s]+\s*$", "", raw, flags=re.I).strip()
    if not pages:
        pages = allowed[:3]
    return body, pages


# ---------------------------------------------------------------------------
# Answer verification
# ---------------------------------------------------------------------------
_REFUSAL = "I do not have this information."


def verify_answer(answer: str, items: List[Dict], query_type: str) -> bool:
    """Lightweight grounding check - catches obvious hallucinations."""
    if answer.strip() == _REFUSAL:
        return True

    context = " \n ".join(item["chunk"] for item in items)

    if query_type == "contact":
        matches = PHONE_PATTERN.findall(answer) + EMAIL_PATTERN.findall(answer)
        return bool(matches) and any(m in context for m in matches)

    if query_type == "count":
        numbers = re.findall(r"\d{1,4}(?:,\d{3})?", answer)
        return bool(numbers) and all(n in context for n in numbers)

    if query_type == "date":
        tokens = re.findall(
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun|\d{1,2}|"
            r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\b",
            answer, re.I,
        )
        if not tokens:
            return False
        hits = sum(
            1 for t in tokens
            if re.search(rf"\b{re.escape(t)}\b", context, re.I)
        )
        return hits >= max(1, len(tokens) // 2)

    # Generic text answers: at least some meaningful overlap with the top hit.
    best = items[0]["chunk"].lower()
    tokens = [t for t in tokenize(answer) if len(t) > 2]
    if not tokens:
        return False
    overlap = sum(1 for t in tokens[:10] if t in best)
    return overlap >= 2 or any(
        snippet.lower() in best
        for snippet in (answer[:50], answer[:80])
        if len(snippet) > 20
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    *,
    embedder,
    reranker,
    index,
    chunks: List[str],
    metadata: List[Dict],
    pages: List[Dict],
    groq_client=None,
) -> QAResult:
    """Run the full QA pipeline for a single question."""
    t_total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    question = question.strip()
    query_type = classify_query(question)

    if query_type == "greeting":
        timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
        return QAResult(
            answer="Hello! Ask me a question about the University of Sharjah Faculty Handbook.",
            query_type=query_type,
            best_section="Greeting",
            timings=timings,
        )

    # ---- Dense retrieval + lexical fusion ---------------------------------
    t_retrieval_start = time.perf_counter()
    query_embedding = embedder.encode(
        [normalize_text(question)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    candidates = gather_candidates(
        question, query_type, query_embedding, index, chunks, metadata
    )
    timings["retrieval_ms"] = (time.perf_counter() - t_retrieval_start) * 1000.0

    if not candidates:
        timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
        return QAResult(
            answer=_REFUSAL,
            query_type=query_type,
            best_section="No matching evidence",
            timings=timings,
            num_candidates=0,
        )

    # ---- Cross-encoder rerank ---------------------------------------------
    t_rerank_start = time.perf_counter()
    pairs = [[question, cand["chunk"]] for cand in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked: List[Dict] = []
    for cand, score in zip(candidates, rerank_scores):
        item = dict(cand)
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda x: (
            x["rerank_score"],
            x.get("routing_boost", 0.0),
            x.get("lexical_score", 0.0),
            x.get("dense_score", 0.0),
        ),
        reverse=True,
    )
    reranked = deduplicate_by_text(reranked)
    final_items = reranked[:FINAL_K]
    timings["rerank_ms"] = (time.perf_counter() - t_rerank_start) * 1000.0

    if not final_items or final_items[0]["rerank_score"] < MIN_RERANK_SCORE:
        timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
        return QAResult(
            answer=_REFUSAL,
            query_type=query_type,
            best_section="Low confidence retrieval",
            items=final_items,
            timings=timings,
            num_candidates=len(candidates),
            num_reranked=len(final_items),
        )

    # ---- Deterministic extractor fast-path --------------------------------
    deterministic = None
    if query_type == "contact":
        deterministic = extract_contact_answer(question, final_items)
    elif query_type == "count":
        deterministic = extract_count_answer(question, final_items, pages)
    elif query_type == "date":
        deterministic = extract_date_answer(question, final_items)

    if deterministic:
        answer, source_pages, evidence = deterministic
        used_llm = False
        timings["generation_ms"] = 0.0
    else:
        if groq_client is None:
            timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0
            return QAResult(
                answer="GROQ_API_KEY is not set. Please set it, then reload the app.",
                pages=sorted({int(item["meta"]["page"]) for item in final_items[:3]}),
                query_type=query_type,
                items=final_items,
                evidence=final_items[0]["chunk"],
                timings=timings,
                num_candidates=len(candidates),
                num_reranked=len(final_items),
            )
        t_gen_start = time.perf_counter()
        prompt = build_prompt(question, final_items, query_type)
        raw = ask_groq(groq_client, prompt)
        answer, source_pages = parse_answer_and_pages(raw, final_items)
        if not answer or not verify_answer(answer, final_items, query_type):
            answer = _REFUSAL
        evidence = final_items[0]["chunk"]
        used_llm = True
        timings["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000.0

    # Trim evidence for display
    if evidence and len(evidence) > EVIDENCE_PREVIEW_CHARS:
        evidence = evidence[:EVIDENCE_PREVIEW_CHARS].rstrip() + "..."

    best_section = (
        final_items[0]["meta"].get("section")
        or f"Page {final_items[0]['meta']['page']}"
    )

    timings["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0

    return QAResult(
        answer=answer,
        pages=source_pages,
        best_section=best_section,
        evidence=evidence,
        query_type=query_type,
        items=final_items,
        used_llm=used_llm,
        timings=timings,
        num_candidates=len(candidates),
        num_reranked=len(final_items),
    )
