"""
Knowledge-base bootstrap.

``build_knowledge_base`` loads models, reads the registered documents (with
optional OCR), builds chunks + embeddings + FAISS index, and returns everything
the QA pipeline needs. Used by the NiceGUI UI so startup behaviour is identical
no matter who calls it.

POST-FREEZE PF-2 (several documents)
    The corpus is driven by the source registry (``knowledge/sources.json``):
    every enabled record is loaded in registry order, the first one (the
    Faculty Handbook) exactly as before, so its chunk texts and chunk ids are
    unchanged. Each further document is loaded on its own (a failure names the
    source), keeps its own 1-based page numbers, passes the deterministic
    page-level language filter its record asks for, is chunked with the
    procedure-aware chunk types when it is a system manual, and is annotated
    with its record's identity. The embedding cache is keyed by the enabled
    source set, the files' sha256 and the cache format, never by a path.
    ``kb.pages`` stays the primary document's pages (the legacy extractors
    read the handbook only); ``kb.source_pages`` holds every document's pages.

    ``pdf_path`` still builds a one-document knowledge base from that file,
    as before; so does a start whose registry cannot be read.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chunking import build_chunks
from .config import (
    CACHE_VERSION,
    EMBED_MODEL,
    ENABLE_OCR,
    PAGE_ARABIC_SHARE_MAX,
    RERANK_MODEL,
    ROW_ARABIC_SHARE_MAX,
)
from .pdf_loader import find_pdf_file, load_pdf, ocr_stats
from .qa import build_faiss_index
from .retrieval import build_or_load_embeddings
from .sources import (
    SectionMap,
    SourceRecord,
    SourceRegistry,
    annotate_metadata,
    check_section_map_pages,
    file_sha256,
    load_source_registry,
    section_map_for,
    stamp_conflicts,
    unregistered_source,
)
from .text_utils import normalize_text

#: Document types chunked with the procedure-aware chunk types (``chunking.build_chunks(..., procedures=True)``).
PROCEDURE_DOCUMENT_TYPES = frozenset({"system_guide", "user_guide", "guideline"})

#: Unicode blocks counted as Arabic script by the page-level language filter.
_ARABIC_BLOCKS = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


class SourceLoadError(RuntimeError):
    """One registered source could not be loaded or indexed; the message names the source."""


@dataclass
class KnowledgeBase:
    """Container for everything the QA pipeline needs at runtime."""
    pdf_file: str
    embedder: Any
    reranker: Any
    groq_client: Optional[Any]
    pages: List[Dict]
    chunks: List[str]
    metadata: List[Dict]
    embeddings: np.ndarray
    index: Any
    stats: Dict[str, Any] = field(default_factory=dict)
    # Identity of the loaded document from the source registry (or a
    # deterministic "unregistered_..." record when it is not listed). With
    # several documents this is the primary one (the handbook).
    source: Optional[SourceRecord] = None
    # Post-freeze PF-2: every loaded document in corpus order (the primary
    # first), the indexed pages of each by source id, and the cache identity
    # of the enabled source set ("" for a one-document build).
    sources: List[SourceRecord] = field(default_factory=list)
    source_pages: Dict[str, List[Dict]] = field(default_factory=dict)
    corpus_key: str = ""


def _load_models():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return embedder, reranker


def _make_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Language control (deterministic, metadata-level; no Arabic support)
# ---------------------------------------------------------------------------
def _is_arabic(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _ARABIC_BLOCKS)


def arabic_share(text: str) -> float:
    """Share of the alphabetic characters of ``text`` written in Arabic script (0.0 when it has no letters)."""
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _is_arabic(c)) / len(letters)


def filter_english_pages(pages: Sequence[Dict], *, page_max: float = PAGE_ARABIC_SHARE_MAX,
                         row_max: float = ROW_ARABIC_SHARE_MAX) -> Tuple[List[Dict], Dict[str, Any]]:
    """The ``english_only`` page filter.

    A page whose Arabic share exceeds ``page_max`` is not indexed. On a kept page, every line or row whose share
    exceeds ``row_max`` is dropped and the page text is rebuilt from what remains (a page left without text and
    without OCR snippets is dropped too); a section hint in Arabic script is cleared. A page that contains no Arabic
    script is returned as the same object, unchanged. Page numbers are never renumbered. Returns the kept pages and
    a report of what was dropped.
    """
    kept: List[Dict] = []
    dropped_pages: List[int] = []
    rows_dropped = 0
    for page in pages:
        if arabic_share(page.get("text", "")) > page_max:
            dropped_pages.append(page["page"])
            continue
        lines = list(page.get("lines") or [])
        rows = list(page.get("rows") or [])
        hint = page.get("section_hint") or ""
        keep_lines = [line for line in lines if arabic_share(line) <= row_max]
        keep_rows = [row for row in rows if arabic_share(row) <= row_max]
        hint_ok = arabic_share(hint) <= row_max
        if len(keep_lines) == len(lines) and len(keep_rows) == len(rows) and hint_ok:
            kept.append(page)
            continue
        rows_dropped += (len(lines) - len(keep_lines)) + (len(rows) - len(keep_rows))
        text = normalize_text(" ".join(keep_lines or keep_rows))
        if not text and not page.get("ocr_snippets"):
            dropped_pages.append(page["page"])
            continue
        cleaned = dict(page)
        cleaned["lines"] = keep_lines
        cleaned["rows"] = keep_rows
        cleaned["text"] = text
        cleaned["section_hint"] = hint if hint_ok else ""
        kept.append(cleaned)
    report = {
        "filter": "english_only", "pages_loaded": len(pages), "pages_indexed": len(kept),
        "pages_dropped": dropped_pages, "rows_dropped": rows_dropped,
        "page_share_max": page_max, "row_share_max": row_max,
    }
    return kept, report


# ---------------------------------------------------------------------------
# Source identity and section maps
# ---------------------------------------------------------------------------
def _apply_section_map(source: SourceRecord, pages: Sequence[Dict], log) -> Optional[SectionMap]:
    """The source's section map when it declares one and it fits ``pages``; otherwise None, with the reason logged.
    A map whose page count does not fit is not applied: the document keeps its source identity and its chunks carry
    no chapter or section labels, rather than labels from another edition."""
    try:
        section_map = section_map_for(source)
    except (OSError, ValueError) as exc:
        log(f"Section map for {source.source_id} failed to load ({exc}); chunks carry no section data.")
        return None
    if section_map is not None:
        problem = check_section_map_pages(section_map, pages)
        if problem:
            log(f"WARNING: section map for {source.source_id} not applied: {problem}. "
                "Chunks keep the source identity but carry no chapter or section labels.")
            return None
    return section_map


def _resolve_source(pdf_file: str, pages: List[Dict], log) -> Tuple[SourceRecord, Optional[SectionMap]]:
    """Identify ``pdf_file`` in the source registry and load its section map (the one-document path).

    Never raises: a missing or malformed registry, an unlisted document or a
    broken section map is reported through ``log`` and the document goes on
    as unregistered (no section data), so the app still starts.
    """
    try:
        registry = load_source_registry()
        source = registry.find_source_for_path(pdf_file)
    except (OSError, ValueError) as exc:
        log(f"Source registry unavailable ({exc}); treating the document as unregistered.")
        source = None
    if source is None:
        source = unregistered_source(pdf_file)
        log(f"Document is not listed in the source registry: source_id={source.source_id}, no section map.")
        return source, None
    return source, _apply_section_map(source, pages, log)


def corpus_identity(records: Sequence[SourceRecord]) -> Tuple[str, List[Dict[str, Any]]]:
    """``(key, entries)`` for the enabled source set: ``key`` is the sha256 (first 16 hex digits) of the ordered
    sources, each as its id, its file's sha256, its page filter, its chunking and its section map name, together
    with the cache format version. No path takes part, so the key is the same on every machine holding the same
    files, and it changes whenever a source is added, removed, disabled, replaced or chunked differently."""
    entries: List[Dict[str, Any]] = []
    for record in records:
        entries.append({
            "source_id": record.source_id,
            "sha256": file_sha256(record.absolute_path()) if record.exists() else "",
            "page_filter": record.page_filter,
            "procedures": record.document_type in PROCEDURE_DOCUMENT_TYPES,
            "section_map": os.path.basename(record.section_map) if record.section_map else "",
        })
    payload = json.dumps({"format": CACHE_VERSION, "sources": entries}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], entries


# ---------------------------------------------------------------------------
# Loading one registered source
# ---------------------------------------------------------------------------
def load_source(record: SourceRecord, log=lambda message: None) -> Tuple[List[Dict], Optional[SectionMap], Dict[str, Any]]:
    """Load, hash-check and language-filter one registered document. Returns ``(pages, section_map, report)``.
    Every failure is a :class:`SourceLoadError` naming the source; another source is never blamed."""
    path = record.absolute_path()
    if not os.path.isfile(path):
        raise SourceLoadError("source %s: file not found at %s" % (record.source_id, record.relative_path))
    if record.sha256:
        actual = file_sha256(path)
        if actual != record.sha256:
            raise SourceLoadError("source %s: file sha256 %s... does not match the registered %s... "
                                  "(another version of the document; update the registry deliberately)"
                                  % (record.source_id, actual[:12], record.sha256[:12]))
    try:
        pages = load_pdf(path)
    except Exception as exc:
        raise SourceLoadError("source %s: %s: %s" % (record.source_id, type(exc).__name__, exc)) from exc
    report: Dict[str, Any] = {"pages_loaded": len(pages), "language_filter": None}
    if record.page_filter == "english_only":
        pages, report["language_filter"] = filter_english_pages(pages)
    elif record.page_filter:
        raise SourceLoadError("source %s: unknown page filter %r" % (record.source_id, record.page_filter))
    if not pages:
        raise SourceLoadError("source %s: no indexable page" % record.source_id)
    report["pages_indexed"] = len(pages)
    return pages, _apply_section_map(record, pages, log), report


@dataclass
class CorpusParts:
    """What ``assemble_corpus`` produced before any model runs: the inputs of the embedding cache and the index."""
    records: List[SourceRecord]
    chunks: List[str]
    metadata: List[Dict]
    cache_pages: List[Dict[str, Any]]            # [{"source_id", "pages"}] in corpus order: the cache's page input
    source_pages: Dict[str, List[Dict]]
    per_source: List[Dict[str, Any]]
    key: str
    entries: List[Dict[str, Any]]
    primary_section_records: int


def assemble_corpus(records: Sequence[SourceRecord], log=lambda message: None) -> CorpusParts:
    """Load and chunk every record in order and annotate the chunks. Pure (no model, no cache): the handbook comes
    first and is chunked exactly as a one-document build chunks it, so its chunk ids and texts are unchanged; every
    later document's chunk ids continue the numbering (the frozen section guard requires ``chunk_id == index``)."""
    if not records:
        raise SourceLoadError("no enabled source to build the corpus from")
    all_chunks: List[str] = []
    all_meta: List[Dict] = []
    cache_pages: List[Dict[str, Any]] = []
    source_pages: Dict[str, List[Dict]] = {}
    per_source: List[Dict[str, Any]] = []
    primary_section_records = 0
    for position, record in enumerate(records):
        log(f"Reading {record.source_id}: {record.relative_path}")
        pages, section_map, report = load_source(record, log)
        procedures = record.document_type in PROCEDURE_DOCUMENT_TYPES
        chunks, metadata = build_chunks(pages, procedures=procedures)
        if not chunks:
            raise SourceLoadError("source %s: no text chunks were created" % record.source_id)
        annotate_metadata(metadata, record, section_map)
        flagged = stamp_conflicts(chunks, metadata, record)
        offset = len(all_chunks)
        if offset:
            for meta in metadata:
                meta["chunk_id"] = offset + int(meta["chunk_id"])
        types = Counter(str(m.get("chunk_type")) for m in metadata)
        per_source.append({
            "source_id": record.source_id, "title": record.title, "citation_label": record.label,
            "document_type": record.document_type, "language": record.language, "enabled": record.enabled,
            "pages_loaded": report["pages_loaded"], "pages_indexed": len(pages), "language_filter": report["language_filter"],
            "procedures": procedures, "chunks": len(chunks), "chunk_types": dict(sorted(types.items())),
            "chunk_id_range": [offset, offset + len(chunks) - 1], "conflict_chunks": flagged,
            "section_records": len(section_map.records) if section_map else 0,
        })
        if position == 0:
            primary_section_records = len(section_map.records) if section_map else 0
        all_chunks.extend(chunks)
        all_meta.extend(metadata)
        cache_pages.append({"source_id": record.source_id, "pages": pages})
        source_pages[record.source_id] = pages
        log(f"  {len(pages)} page(s) indexed of {report['pages_loaded']}, {len(chunks)} chunks"
            + (f", {flagged} under a registered conflict" if flagged else "")
            + (f" (section map: {len(section_map.records)} records)" if section_map else ""))
    key, entries = corpus_identity(records)
    return CorpusParts(list(records), all_chunks, all_meta, cache_pages, source_pages, per_source, key, entries,
                       primary_section_records)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _build_single(pdf_file: str, embedder, reranker, groq_client, log) -> KnowledgeBase:
    """The one-document build, unchanged from the frozen runtime: ``pdf_file`` is chunked, identified in the
    registry when it is listed there, cached under its own file name."""
    log(f"Reading PDF: {pdf_file}")
    pages = load_pdf(pdf_file)
    chunks, metadata = build_chunks(pages)
    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    # Source identity and section labels on every chunk (metadata only; the
    # chunk text is untouched, so ranking is unaffected).
    source, section_map = _resolve_source(pdf_file, pages, log)
    annotate_metadata(metadata, source, section_map)
    flagged = stamp_conflicts(chunks, metadata, source)
    log(f"Source: {source.source_id} ({'section map: %d records' % len(section_map.records) if section_map else 'no section map'}).")

    ocr_chunks = sum(1 for m in metadata if m.get("chunk_type") == "image_ocr")
    log(f"Built {len(chunks)} chunks from {len(pages)} pages "
        f"({ocr_chunks} from OCR).")

    embeddings = build_or_load_embeddings(pdf_file, chunks, metadata, pages, embedder)
    log("Building FAISS index...")
    index = build_faiss_index(embeddings)

    types = Counter(str(m.get("chunk_type")) for m in metadata)
    stats: Dict[str, Any] = {
        "total_chunks": len(chunks),
        "total_pages": len(pages),
        "primary_pages": len(pages),
        "ocr_chunks": ocr_chunks,
        "ocr": ocr_stats(),
        "source": {
            "source_id": source.source_id,
            "title": source.title,
            "registered": source.registered,
            "section_records": len(section_map.records) if section_map else 0,
        },
        "sources": [{
            "source_id": source.source_id, "title": source.title, "citation_label": source.label,
            "document_type": source.document_type, "language": source.language, "enabled": True,
            "pages_loaded": len(pages), "pages_indexed": len(pages), "language_filter": None, "procedures": False,
            "chunks": len(chunks), "chunk_types": dict(sorted(types.items())), "chunk_id_range": [0, len(chunks) - 1],
            "conflict_chunks": flagged, "section_records": len(section_map.records) if section_map else 0,
        }],
        "chunk_types": dict(sorted(types.items())),
        "corpus_key": "",
    }

    return KnowledgeBase(
        pdf_file=pdf_file,
        embedder=embedder,
        reranker=reranker,
        groq_client=groq_client,
        pages=pages,
        chunks=chunks,
        metadata=metadata,
        embeddings=embeddings,
        index=index,
        stats=stats,
        source=source,
        sources=[source],
        source_pages={source.source_id: pages},
        corpus_key="",
    )


def _build_corpus(records: Sequence[SourceRecord], embedder, reranker, groq_client, log) -> KnowledgeBase:
    """The registry-driven multi-document build (post-freeze PF-2)."""
    parts = assemble_corpus(records, log)
    primary = parts.records[0]
    ocr_chunks = sum(1 for m in parts.metadata if m.get("chunk_type") == "image_ocr")
    total_pages = sum(len(p) for p in parts.source_pages.values())
    log(f"Corpus {parts.key}: {len(parts.records)} source(s), {total_pages} pages, {len(parts.chunks)} chunks "
        f"({ocr_chunks} from OCR).")

    embeddings = build_or_load_embeddings("corpus_" + parts.key, parts.chunks, parts.metadata, parts.cache_pages, embedder)
    log("Building FAISS index...")
    index = build_faiss_index(embeddings)

    stats: Dict[str, Any] = {
        "total_chunks": len(parts.chunks),
        "total_pages": total_pages,
        "primary_pages": len(parts.source_pages[primary.source_id]),
        "ocr_chunks": ocr_chunks,
        "ocr": ocr_stats(),
        "source": {
            "source_id": primary.source_id,
            "title": primary.title,
            "registered": primary.registered,
            "section_records": parts.primary_section_records,
        },
        "sources": parts.per_source,
        "chunk_types": dict(sorted(Counter(str(m.get("chunk_type")) for m in parts.metadata).items())),
        "corpus_key": parts.key,
        "corpus": parts.entries,
    }

    return KnowledgeBase(
        pdf_file=primary.absolute_path(),
        embedder=embedder,
        reranker=reranker,
        groq_client=groq_client,
        pages=parts.source_pages[primary.source_id],
        chunks=parts.chunks,
        metadata=parts.metadata,
        embeddings=embeddings,
        index=index,
        stats=stats,
        source=primary,
        sources=parts.records,
        source_pages=parts.source_pages,
        corpus_key=parts.key,
    )


def build_knowledge_base(pdf_path: Optional[str] = None,
                         *,
                         verbose: bool = True,
                         registry: Optional[SourceRegistry] = None) -> KnowledgeBase:
    """Full bootstrap. Without ``pdf_path`` the enabled sources of the registry (``registry`` or the repository's)
    are loaded in order; ``pdf_path`` builds a one-document knowledge base from that file, as before."""
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("Loading embedding model...")
    embedder, reranker = _load_models()

    log("Initialising Groq client...")
    groq_client = _make_groq_client()

    if ENABLE_OCR:
        log("OCR is ON (rapidocr-onnxruntime). First run may take several minutes.")

    if pdf_path is not None:
        return _build_single(pdf_path, embedder, reranker, groq_client, log)

    records: List[SourceRecord] = []
    try:
        reg = registry if registry is not None else load_source_registry()
        records = reg.enabled_sources()
    except (OSError, ValueError) as exc:
        log(f"Source registry unavailable ({exc}); loading the first PDF in data/ as a single document.")
    if not records:
        return _build_single(find_pdf_file(), embedder, reranker, groq_client, log)
    return _build_corpus(records, embedder, reranker, groq_client, log)
