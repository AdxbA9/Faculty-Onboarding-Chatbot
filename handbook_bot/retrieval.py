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
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import faiss
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
# Retrieval scope (optional): restrict a search to part of the corpus
# ---------------------------------------------------------------------------
_SECTION_NO = re.compile(r"^\d+(?:\.\d+)*$")


def _frozen_strings(name: str, value, pattern=None) -> Optional[FrozenSet[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError("%s must be a collection of strings, not a single string" % name)
    out = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("%s must contain non-empty strings, got %r" % (name, item))
        if pattern is not None and not pattern.match(item):
            raise ValueError("%s entry %r must look like 1 or 1.2" % (name, item))
        out.add(item)
    return frozenset(out)


def _section_nos(meta: Dict) -> Sequence[str]:
    present = meta.get("page_section_nos")
    if present:
        return present
    single = meta.get("section_no")
    return (single,) if single else ()


def _eligible_section_nos(meta: Dict) -> List[str]:
    """Printed section numbers on the chunk's page whose leading number equals
    the chunk's resolved chapter (``meta["chapter"]``, from the chapter page
    ranges). A number printed under the wrong chapter is kept in the metadata
    as printed but is never eligible here; a chunk without a resolved chapter
    has no eligible section at all."""
    chapter = meta.get("chapter")
    if isinstance(chapter, bool) or not isinstance(chapter, int):
        return []
    eligible: List[str] = []
    for number in _section_nos(meta):
        head = str(number).split(".", 1)[0]
        if head.isdigit() and int(head) == chapter:
            eligible.append(number)
    return eligible


@dataclass(frozen=True)
class RetrievalScope:
    """Which chunks a search may see. Every constraint given must hold (AND);
    within one constraint any listed value matches (OR). Matching reads the
    chunk metadata written by ``handbook_bot.sources.annotate_metadata``:

    * ``source_ids``: ``meta["source_id"]`` is one of them.
    * ``chapters``: ``meta["chapter"]`` is one of them.
    * ``section_numbers``: a section present on the chunk's page
      (``meta["page_section_nos"]``, else ``meta["section_no"]``) equals one
      of them, and that section is chapter-consistent (see below).
    * ``section_prefixes``: as above, matched on dotted-number prefix:
      ``"1"`` matches ``"1"`` and ``"1.14"``, never ``"12"``; ``"12.1"``
      matches ``"12.1"``, never ``"12.10"``.
    * ``page_ranges``: inclusive ``(first, last)`` pairs on ``meta["page"]``.

    Section constraints are CHAPTER-CONSISTENT. A printed section number is
    eligible only when its leading number equals ``meta["chapter"]``, the
    chapter resolved from the chapter page ranges. The metadata records
    numbers as the document prints them (the handbook prints a chapter-9
    section as "12.3"), but such a number is never admitted by
    ``section_numbers`` or ``section_prefixes`` of another chapter; reach
    that section through ``chapters`` or ``page_ranges``. A chapter
    constraint combined with a section number of a different chapter
    matches nothing, by design.

    Section constraints are PAGE-GRANULAR. They operate on page-level
    section metadata: they restrict pages associated with a section (every
    section present on a page counts, so a shared boundary page is included)
    and do not guarantee text-level isolation on a page shared by several
    sections. Printed section numbers are not unique identifiers either (the
    handbook uses 5.2 and 7.8 twice), so one constraint may admit more than
    one printed section. Prefer ``source_ids`` plus ``chapters``; use
    ``page_ranges`` where exact physical isolation is required; never treat
    ``section_no`` as a primary key.

    A chunk without the metadata a constraint needs never matches. A scope
    with no constraint at all is rejected: pass ``scope=None`` for an
    unrestricted search. ``RetrievalScope`` is immutable and hashable.
    """

    source_ids: Optional[FrozenSet[str]] = None
    chapters: Optional[FrozenSet[int]] = None
    section_numbers: Optional[FrozenSet[str]] = None
    section_prefixes: Optional[FrozenSet[str]] = None
    page_ranges: Optional[Tuple[Tuple[int, int], ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", _frozen_strings("source_ids", self.source_ids))
        object.__setattr__(self, "section_numbers", _frozen_strings("section_numbers", self.section_numbers, _SECTION_NO))
        object.__setattr__(self, "section_prefixes", _frozen_strings("section_prefixes", self.section_prefixes, _SECTION_NO))
        if self.chapters is not None:
            chapters = set()
            for item in self.chapters:
                if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                    raise ValueError("chapters must contain positive integers, got %r" % (item,))
                chapters.add(item)
            object.__setattr__(self, "chapters", frozenset(chapters))
        if self.page_ranges is not None:
            ranges = []
            for item in self.page_ranges:
                try:
                    first, last = item
                except (TypeError, ValueError):
                    raise ValueError("page_ranges must contain (first, last) pairs, got %r" % (item,)) from None
                if any(isinstance(v, bool) or not isinstance(v, int) for v in (first, last)) or first < 1 or last < first:
                    raise ValueError("page range must satisfy 1 <= first <= last, got %r" % (item,))
                ranges.append((first, last))
            object.__setattr__(self, "page_ranges", tuple(sorted(set(ranges))))
        if not any(v for v in (self.source_ids, self.chapters, self.section_numbers,
                               self.section_prefixes, self.page_ranges)):
            raise ValueError("RetrievalScope needs at least one non-empty constraint; use scope=None for no restriction")

    def allows(self, meta: Dict) -> bool:
        """True when the chunk described by ``meta`` is inside the scope."""
        if self.source_ids is not None and meta.get("source_id") not in self.source_ids:
            return False
        if self.chapters is not None and meta.get("chapter") not in self.chapters:
            return False
        if self.section_numbers is not None or self.section_prefixes is not None:
            nos = _eligible_section_nos(meta)
            if self.section_numbers is not None and not any(s in self.section_numbers for s in nos):
                return False
            if self.section_prefixes is not None and not any(
                s == p or s.startswith(p + ".") for p in self.section_prefixes for s in nos
            ):
                return False
        if self.page_ranges is not None:
            page = meta.get("page")
            if isinstance(page, bool) or not isinstance(page, int):
                return False
            if not any(first <= page <= last for first, last in self.page_ranges):
                return False
        return True

    def allowed_indices(self, metadata: Sequence[Dict]) -> List[int]:
        """Positions of the chunks inside the scope, in corpus order."""
        return [i for i, meta in enumerate(metadata) if self.allows(meta)]

    def describe(self) -> Dict:
        return {
            "source_ids": sorted(self.source_ids) if self.source_ids is not None else None,
            "chapters": sorted(self.chapters) if self.chapters is not None else None,
            "section_numbers": sorted(self.section_numbers) if self.section_numbers is not None else None,
            "section_prefixes": sorted(self.section_prefixes) if self.section_prefixes is not None else None,
            "page_ranges": [list(r) for r in self.page_ranges] if self.page_ranges is not None else None,
        }


def _search_within(index, query_embedding: np.ndarray, k: int, allowed: Sequence[int]):
    """Exact nearest-neighbour search over ``allowed`` corpus positions only.

    The search is restricted at the index level (FAISS id selector), so an
    in-scope chunk ranked below the global top-k can never be lost. Only when
    the index cannot take search parameters at all (an older FAISS build or
    a non-FAISS index object) are the allowed vectors scored directly, and
    only for an inner-product index: scoring another metric that way would
    return misleading similarities, so it raises ``TypeError`` instead. Any
    other exception from the index, ``RuntimeError`` included, is a genuine
    search failure and propagates.
    Returns ``(distances, ids)`` shaped like ``index.search``.
    """
    ids = np.asarray(list(allowed), dtype=np.int64)
    try:
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
        return index.search(query_embedding, k, params=params)
    except (TypeError, AttributeError):
        pass                                    # no parameter support: fall back below
    metric = getattr(index, "metric_type", None)
    if metric != faiss.METRIC_INNER_PRODUCT:
        raise TypeError(
            "scoped dense search without FAISS id selectors supports inner-product indexes only; "
            "this index reports metric_type %r" % (metric,)
        )
    vectors = index.reconstruct_batch(ids)
    scores = np.asarray(vectors, dtype=np.float32) @ np.asarray(query_embedding, dtype=np.float32)[0]
    order = np.argsort(-scores, kind="stable")[:k]
    return scores[order][None, :], ids[order][None, :]


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
    scope: Optional[RetrievalScope] = None,
) -> List[Dict]:
    """Fuse dense (FAISS) and lexical candidates, apply routing boosts and
    return the top-N to feed to the cross-encoder reranker.

    With ``scope=None`` (the default) the whole corpus is searched exactly as
    before. With a :class:`RetrievalScope`, both the dense and the lexical
    search see only the chunks inside the scope; a scope that matches no
    chunk returns an empty list and never falls back to the whole corpus.
    """
    if scope is None:
        D, I = index.search(query_embedding, TOP_K_DENSE)
        lexical_pool: Iterable = enumerate(chunks)
    else:
        if not isinstance(scope, RetrievalScope):
            raise TypeError("scope must be a RetrievalScope or None, got %s" % type(scope).__name__)
        allowed = scope.allowed_indices(metadata)
        if not allowed:
            return []
        D, I = _search_within(index, query_embedding, min(TOP_K_DENSE, len(allowed)), allowed)
        lexical_pool = ((i, chunks[i]) for i in allowed)
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
        ((i, lexical_score(question, text)) for i, text in lexical_pool),
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
