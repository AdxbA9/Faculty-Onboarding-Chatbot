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
from typing import Any, Dict, List, Optional

import numpy as np

from .chunking import build_chunks
from .config import EMBED_MODEL, ENABLE_OCR, RERANK_MODEL
from .pdf_loader import find_pdf_file, load_pdf, ocr_stats
from .qa import build_faiss_index
from .retrieval import build_or_load_embeddings


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
    )
