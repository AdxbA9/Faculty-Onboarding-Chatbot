"""
Question-answering public API and the Part 1 building blocks it is made of.

Public entry point: :func:`answer_question`. Its signature and the fields of
:class:`QAResult` are a compatibility contract with ``ui/`` and ``eval/``.

Since Part 2 (Milestone 1) ``answer_question`` is a thin wrapper. The control
flow lives in :mod:`handbook_bot.orchestrator`:

    question
        -> Router agent            (agents/router.py)
        -> greeting? canned reply
        -> embed + FAISS + lexical -> rerank -> low confidence? refuse
        -> deterministic extractor (contact/date/count), else Synthesis agent
        -> Verifier                (agents/verifier.py; Part 1 check until it lands)
        -> at most one retry -> structured result

What stays in this module are the pieces those agents reuse: the prompt
builder, the Groq call, the ``Pages:`` parser and the Part 1 grounding check.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .agents.types import SubQuestion, TraceEntry
from .config import (
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TRANSPORT_RETRIES,
)
from .text_utils import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    tokenize,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class QAResult:
    """Structured answer returned to callers.

    Every Part 1 field is unchanged in name, type and meaning. The Part 2
    fields are appended with defaults, so existing constructors and readers
    keep working.
    """
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

    # Part 2: what each agent decided, and what the question cost.
    agent_trace: List[TraceEntry] = field(default_factory=list)
    llm_calls: int = 0              # logical LLM calls issued for this question, the router's included
    retried: bool = False           # True when the Verifier asked for, and got, a second draft
    sub_questions: List[SubQuestion] = field(default_factory=list)   # filled by the Planner (Milestone 2)

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
            "agent_trace": [dataclasses.asdict(entry) for entry in self.agent_trace],
            "llm_calls": self.llm_calls,
            "retried": self.retried,
            "sub_questions": [dataclasses.asdict(sq) for sq in self.sub_questions],
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

#: Longest reviewer feedback that is copied into a retry prompt.
_MAX_FEEDBACK_CHARS = 400


def build_prompt(
    question: str,
    items: List[Dict],
    query_type: str,
    *,
    feedback: Optional[str] = None,
) -> str:
    """Assemble the grounded prompt sent to the LLM.

    ``feedback`` is only given for a Verifier-requested second draft. It says
    what the first draft got wrong; the grounding rules still apply in full.
    """
    context_parts = []
    for i, item in enumerate(items, 1):
        meta = item["meta"]
        label = f"Source {i} | Page {meta['page']} | Type {meta.get('chunk_type', 'paragraph')}"
        if meta.get("section"):
            label += f" | Section {meta['section']}"
        context_parts.append(f"{label}\n{item['chunk']}")
    context = "\n\n".join(context_parts)

    extra = _INTENT_GUIDANCE.get(query_type, "")

    review = ""
    if feedback:
        note = re.sub(r"\s+", " ", feedback).strip()[:_MAX_FEEDBACK_CHARS]
        review = (
            "\nA reviewer rejected your previous answer: "
            f"{note}\n"
            "Write a corrected answer. Every rule above still applies: use ONLY the context, "
            "and if the context does not support an answer reply exactly: I do not have this information.\n"
        )

    # Rule 1 used to read "one or two short sentences". A three-part question
    # cannot be answered in two sentences, so that rule produced the
    # "incomplete multi-part answers" failure of Part 1 (report 6.7.2).
    return f"""You answer questions about the University of Sharjah Faculty Handbook.
Use ONLY the supplied context.

Rules:
1. Answer concisely, but use enough sentences to fully address every part of the question that the context supports. If the question has several parts, answer each part clearly, roughly one sentence per part. A simple question gets a short answer.
2. If the answer is not clearly supported by the context, reply exactly: I do not have this information.
3. Do not guess. Do not combine unrelated rows or pages.
4. {extra}
5. After the answer, add a new line exactly like this: Pages: page_numbers_only
6. Only cite pages from the supplied context.
{review}
Context:
{context}

Question: {question}

Answer:""".strip()


# ---------------------------------------------------------------------------
# LLM call + parsing
# ---------------------------------------------------------------------------
def ask_groq(client, prompt: str) -> str:
    """Call the Groq chat completion endpoint and return the raw text.

    One call here is ONE logical LLM call, which is what ``QAResult.llm_calls``
    counts. The Groq SDK may repeat the HTTP request on 429, 5xx and timeouts,
    with back-off; ``LLM_TRANSPORT_RETRIES`` sets how often and is applied
    explicitly so that it is a documented setting and not a hidden SDK default.
    """
    caller = client
    if hasattr(client, "with_options"):
        caller = client.with_options(max_retries=max(0, LLM_TRANSPORT_RETRIES))
    resp = caller.chat.completions.create(
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


# En and em dashes as code points, so this source file stays pure ASCII.
_DASHES = "-" + chr(0x2013) + chr(0x2014)
_PAGE_ITEM = r"(?:pages?\s*)?\d+"
_PAGE_SPEC = _PAGE_ITEM + r"(?:\s*(?:,|;|&|\band\b|\bto\b|[" + _DASHES + r"])\s*" + _PAGE_ITEM + r")*"
# A citation that ends the output: on its own line, or closing the last sentence.
_TRAILING_PAGES = re.compile(
    r"(?:^|[\s(\[])(?P<kw>pages?)\s*:\s*(?P<spec>" + _PAGE_SPEC + r")\s*[.)\]]*\s*$", re.I)
# Otherwise: a line that is nothing but a citation (the last such line wins).
_PAGES_LINE = re.compile(r"^[ \t]*pages?\s*:\s*(" + _PAGE_SPEC + r")\s*[.)\]]*[ \t]*$", re.I | re.M)
# "Pages:" with nothing citable after it. The model writes this under a refusal;
# it must still be stripped or the refusal sentence is no longer recognised.
_EMPTY_PAGES_LINE = re.compile(r"\n?[ \t]*pages?\s*:\s*(?:none|n/?a|not applicable|[-,\s])*[ \t.]*$", re.I)
_PAGE_RANGE = re.compile(r"(\d+)(?:\s*(?:\bto\b|[" + _DASHES + r"])\s*(?:pages?\s*)?(\d+))?", re.I)
#: "Pages: 12-14" means 12, 13 and 14. A span wider than this is read as two
#: separate numbers, so a stray "1-272" cannot cite the whole handbook.
_MAX_RANGE_SPAN = 10


def _pages_from_spec(spec: str, allowed: List[int]) -> List[int]:
    pages: List[int] = []
    for m in _PAGE_RANGE.finditer(spec):
        first = int(m.group(1))
        last = int(m.group(2)) if m.group(2) else first
        span = range(first, last + 1) if 0 <= last - first <= _MAX_RANGE_SPAN else (first, last)
        for page in span:
            if page in allowed and page not in pages:
                pages.append(page)
    return pages


def split_answer_and_claimed_pages(raw: str, items: List[Dict]) -> Tuple[str, List[int]]:
    """Split the LLM output into (answer_body, claimed_pages).

    Only pages that appear in the supplied context are accepted - a cheap
    but effective guard against hallucinated citations. When the model wrote
    no usable ``Pages:`` line the list is EMPTY: this function never invents a
    citation. Deciding what to cite then is the Verifier's job.

    The citation is the LAST ``Pages:`` expression, because prompt rule 5 puts
    it after the answer. Part 1 read the FIRST one, so an answer that merely
    contained the words "two pages: ..." lost its real citation. Ranges
    ("12-14"), "and", en dashes and a trailing full stop are understood.
    Prose that mentions pages elsewhere is left in the answer untouched.
    """
    raw = raw or ""
    allowed = sorted({int(item["meta"]["page"]) for item in items})

    m = _TRAILING_PAGES.search(raw)
    if m:
        body = raw[:m.start("kw")].rstrip(" \t\r\n([").strip()
        return body, _pages_from_spec(m.group("spec"), allowed)

    lines = list(_PAGES_LINE.finditer(raw))
    if lines:
        last = lines[-1]
        body = (raw[:last.start()] + raw[last.end():]).strip()
        return body, _pages_from_spec(last.group(1), allowed)

    return _EMPTY_PAGES_LINE.sub("", raw).strip(), []


def legacy_fallback_pages(items: List[Dict]) -> List[int]:
    """Part 1 behaviour when the model cited nothing: the first three context
    pages, whether or not they support the answer.

    Known defect (invented citations). It is kept ONLY so the Part 1 check can
    be reproduced exactly until ``agents/verifier.py`` replaces it; the new
    Verifier selects pages from the items that actually support the answer.
    """
    return sorted({int(item["meta"]["page"]) for item in items})[:3]


def parse_answer_and_pages(raw: str, items: List[Dict]) -> Tuple[str, List[int]]:
    """Part 1 parser, kept for compatibility: claimed pages, else the legacy
    first-three-pages fallback. New code should use
    :func:`split_answer_and_claimed_pages`."""
    body, pages = split_answer_and_claimed_pages(raw, items)
    if not pages:
        pages = legacy_fallback_pages(items)
    return body, pages


# ---------------------------------------------------------------------------
# Answer verification (Part 1 check)
# ---------------------------------------------------------------------------
_REFUSAL = "I do not have this information."
#: Public name for the controlled refusal sentence.
REFUSAL = _REFUSAL


def verify_answer(answer: str, items: List[Dict], query_type: str) -> bool:
    """Lightweight grounding check - catches obvious hallucinations.

    This is the Part 1 (MCBV9) check, unchanged, including its known defect:
    generic answers are compared with the top item only. The orchestrator uses
    it until ``agents/verifier.py`` lands; fixing the defect belongs there.
    """
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
# Public entry point
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
    """Run the full QA pipeline for a single question.

    Thin compatibility wrapper: same signature and same ``QAResult`` as Part 1.
    All control flow is in :func:`handbook_bot.orchestrator.run`. The import is
    local because the orchestrator imports this module for ``QAResult`` and the
    prompt helpers; importing it at module level would be circular.
    """
    from .orchestrator import run

    return run(
        question,
        embedder=embedder,
        reranker=reranker,
        index=index,
        chunks=chunks,
        metadata=metadata,
        pages=pages,
        groq_client=groq_client,
    )
