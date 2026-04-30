"""
Retrieval: embedding cache, FAISS, lexical scoring and hybrid candidate
gathering with query-type routing.

The pipeline is:

    query  -->  [dense FAISS]  +  [lexical overlap]  -->  candidate pool
                         |
                         v
                intent-aware routing boosts
                         |
                         v
                top-N for cross-encoder rerank
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List

import numpy as np

from .config import (
    CACHE_DIR,
    CACHE_VERSION,
    RERANK_CANDIDATES,
    TOP_K_DENSE,
    TOP_K_LEXICAL,
)
from .text_utils import (
    CONTACT_QUESTION_PATTERN,
    COUNT_PATTERN,
    DATE_QUESTION_PATTERN,
    DAY_PATTERN,
    EMAIL_PATTERN,
    GREETING_PATTERN,
    MONTH_PATTERN,
    PHONE_PATTERN,
    TABLE_HINT_PATTERN,
    YEAR_TERM_PATTERN,
    YESNO_PATTERN,
    normalize_text,
    tokenize,
)


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------
def classify_query(question: str) -> str:
    """Return a coarse intent label used to route retrieval and extraction."""
    q = question.strip().lower()
    if GREETING_PATTERN.search(q):
        return "greeting"
    if CONTACT_QUESTION_PATTERN.search(q):
        return "contact"
    if COUNT_PATTERN.search(q):
        return "count"
    if DATE_QUESTION_PATTERN.search(q) or YEAR_TERM_PATTERN.search(q):
        return "date"
    if re.search(
        r"\b(name|list|which are|what are the|standing committee|categories|core values)\b",
        q,
    ):
        return "list"
    if YESNO_PATTERN.search(q):
        return "policy_yesno"
    return "policy"


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
def _cache_paths(pdf_file: str):
    """Return (embeddings, chunks, metadata, pages) cache file paths."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_file))[0]
    prefix = os.path.join(CACHE_DIR, f"{base}_{CACHE_VERSION}")
    return (
        f"{prefix}_embeddings.npy",
        f"{prefix}_chunks.json",
        f"{prefix}_meta.json",
        f"{prefix}_pages.json",
    )


def build_or_load_embeddings(
    pdf_file: str,
    chunks: List[str],
    metadata: List[Dict],
    pages: List[Dict],
    embedder,
) -> np.ndarray:
    """Load cached embeddings if the inputs are unchanged, else rebuild."""
    emb_path, chunks_path, meta_path, pages_path = _cache_paths(pdf_file)

    if all(os.path.exists(p) for p in (emb_path, chunks_path, meta_path, pages_path)):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                cached_chunks = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            with open(pages_path, "r", encoding="utf-8") as f:
                cached_pages = json.load(f)
            if cached_chunks == chunks and cached_meta == metadata and cached_pages == pages:
                return np.load(emb_path)
        except Exception:
            # Any cache error just falls through to a rebuild.
            pass

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    np.save(emb_path, embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(pages_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    return embeddings


# ---------------------------------------------------------------------------
# Lexical scoring
# ---------------------------------------------------------------------------
def lexical_score(query: str, text: str) -> float:
    """Quick-and-useful lexical overlap score used alongside dense retrieval.

    Rewards token overlap, substring matches and partial-phrase matches.
    Chosen to be cheap enough to run over the whole chunk list in Python.
    """
    q = tokenize(query)
    if not q:
        return 0.0
    t = tokenize(text)
    if not t:
        return 0.0
    tset = set(t)

    overlap = sum(1 for token in q if token in tset)

    phrase_bonus = 0.0
    qnorm = normalize_text(query).lower()
    tnorm = normalize_text(text).lower()
    if len(qnorm) > 8 and qnorm in tnorm:
        phrase_bonus += 2.4
    elif len(q) >= 2 and sum(1 for token in q[:4] if token in tset) >= max(2, min(4, len(q)) - 1):
        phrase_bonus += 0.9

    return overlap / math.sqrt(len(tset) + 1) + phrase_bonus


# ---------------------------------------------------------------------------
# Intent-aware routing boosts
# ---------------------------------------------------------------------------
def _routing_boost(query_type: str, question: str, meta: Dict, chunk: str) -> float:
    """Add a small score bump based on chunk type matching the query intent."""
    boost = 0.0
    ctype = meta.get("chunk_type")

    if query_type == "contact":
        if ctype == "row":
            boost += 0.9
        elif ctype == "row_window":
            boost += 0.55
        if PHONE_PATTERN.search(chunk) or EMAIL_PATTERN.search(chunk):
            boost += 0.55

    elif query_type == "date":
        if ctype in {"row", "row_window"}:
            boost += 0.7
        if (
            MONTH_PATTERN.search(chunk)
            or DAY_PATTERN.search(chunk)
            or re.search(r"\b\d{1,2}\s+[A-Z][a-z]{2,}\b", chunk)
        ):
            boost += 0.35

    elif query_type == "count":
        if re.search(r"\b\d{1,4}(?:,\d{3})?\b", chunk):
            boost += 0.35
        if ctype == "paragraph":
            boost += 0.25

    elif query_type == "list":
        if ctype in {"row_window", "paragraph"}:
            boost += 0.25
        if "•" in chunk or " | " in chunk:
            boost += 0.25

    elif query_type == "policy_yesno":
        if ctype == "paragraph":
            boost += 0.4

    else:  # "policy" / default
        if ctype == "paragraph":
            boost += 0.2

    if TABLE_HINT_PATTERN.search(question) and meta.get("table_like"):
        boost += 0.2

    return boost


# ---------------------------------------------------------------------------
# Candidate gathering
# ---------------------------------------------------------------------------
def gather_candidates(
    question: str,
    query_type: str,
    query_embedding: np.ndarray,
    index,
    chunks: List[str],
    metadata: List[Dict],
) -> List[Dict]:
    """Fuse dense (FAISS) and lexical candidates, apply routing boosts and
    return the top-N to feed to the cross-encoder reranker."""
    D, I = index.search(query_embedding, TOP_K_DENSE)
    candidates: Dict[int, Dict] = {}

    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        candidates[int(idx)] = {
            "idx": int(idx),
            "chunk": chunks[idx],
            "meta": metadata[idx],
            "dense_score": float(score),
            "lexical_score": 0.0,
        }

    lexical_ranked = sorted(
        ((i, lexical_score(question, text)) for i, text in enumerate(chunks)),
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_K_LEXICAL]

    for idx, lex_score in lexical_ranked:
        if lex_score <= 0:
            continue
        if idx not in candidates:
            candidates[idx] = {
                "idx": idx,
                "chunk": chunks[idx],
                "meta": metadata[idx],
                "dense_score": 0.0,
                "lexical_score": float(lex_score),
            }
        else:
            candidates[idx]["lexical_score"] = float(lex_score)

    merged = list(candidates.values())
    for cand in merged:
        cand["routing_boost"] = _routing_boost(
            query_type, question, cand["meta"], cand["chunk"]
        )
    merged.sort(
        key=lambda x: (x["dense_score"] + 0.28 * x["lexical_score"] + x["routing_boost"]),
        reverse=True,
    )
    return merged[:RERANK_CANDIDATES]


def deduplicate_by_text(items: List[Dict]) -> List[Dict]:
    """Remove exact-duplicate chunks while preserving order (first wins)."""
    seen: set = set()
    out: List[Dict] = []
    for item in items:
        text = item["chunk"]
        if text not in seen:
            seen.add(text)
            out.append(item)
    return out
