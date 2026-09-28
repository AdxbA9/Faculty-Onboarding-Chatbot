"""
Knowledge-base bootstrap.

``build_knowledge_base`` loads models, reads the PDF (with optional OCR),
builds chunks + embeddings + FAISS index, and returns everything the QA
pipeline needs. Used by the NiceGUI UI so startup behaviour is identical
no matter who calls it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chunking import build_chunks
from .config import EMBED_MODEL, ENABLE_OCR, RERANK_MODEL
from .pdf_loader import find_pdf_file, load_pdf, ocr_stats
from .qa import build_faiss_index
from .retrieval import build_or_load_embeddings
from .sources import (
    SectionMap,
    SourceRecord,
    annotate_metadata,
    check_section_map_pages,
    load_source_registry,
    section_map_for,
    unregistered_source,
)


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
    # deterministic "unregistered_..." record when it is not listed).
    source: Optional[SourceRecord] = None


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


def _resolve_source(pdf_file: str, pages: List[Dict], log) -> Tuple[SourceRecord, Optional[SectionMap]]:
    """Identify ``pdf_file`` in the source registry and load its section map.

    Never raises: a missing or malformed registry, an unlisted document or a
    broken section map is reported through ``log`` and the document goes on
    as unregistered (no section data), so the app still starts. A section
    map whose page count does not fit the loaded ``pages`` is not applied:
    the document keeps its source identity and its chunks carry no chapter
    or section labels, rather than labels from another edition.
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
    try:
        section_map = section_map_for(source)
    except (OSError, ValueError) as exc:
        log(f"Section map for {source.source_id} failed to load ({exc}); chunks carry no section data.")
        section_map = None
    if section_map is not None:
        problem = check_section_map_pages(section_map, pages)
        if problem:
            log(f"WARNING: section map for {source.source_id} not applied: {problem}. "
                "Chunks keep the source identity but carry no chapter or section labels.")
            section_map = None
    return source, section_map


def build_knowledge_base(pdf_path: Optional[str] = None,
                         *,
                         verbose: bool = True) -> KnowledgeBase:
    """Full bootstrap. Pass ``pdf_path`` to override auto-detection."""
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("Loading embedding model...")
    embedder, reranker = _load_models()

    log("Initialising Groq client...")
    groq_client = _make_groq_client()

    pdf_file = pdf_path or find_pdf_file()
    log(f"Reading PDF: {pdf_file}")
    if ENABLE_OCR:
        log("OCR is ON (rapidocr-onnxruntime). First run may take several minutes.")

    pages = load_pdf(pdf_file)
    chunks, metadata = build_chunks(pages)
    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    # Source identity and section labels on every chunk (metadata only; the
    # chunk text is untouched, so ranking is unaffected).
    source, section_map = _resolve_source(pdf_file, pages, log)
    annotate_metadata(metadata, source, section_map)
    log(f"Source: {source.source_id} ({'section map: %d records' % len(section_map.records) if section_map else 'no section map'}).")

    ocr_chunks = sum(1 for m in metadata if m.get("chunk_type") == "image_ocr")
    log(f"Built {len(chunks)} chunks from {len(pages)} pages "
        f"({ocr_chunks} from OCR).")

    embeddings = build_or_load_embeddings(pdf_file, chunks, metadata, pages, embedder)
    log("Building FAISS index...")
    index = build_faiss_index(embeddings)

    stats: Dict[str, Any] = {
        "total_chunks": len(chunks),
        "total_pages": len(pages),
        "ocr_chunks": ocr_chunks,
        "ocr": ocr_stats(),
        "source": {
            "source_id": source.source_id,
            "title": source.title,
            "registered": source.registered,
            "section_records": len(section_map.records) if section_map else 0,
        },
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
    )
