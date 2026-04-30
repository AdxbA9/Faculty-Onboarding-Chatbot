"""
Chunking strategy for the handbook.

Four complementary chunk types per page:

* ``paragraph`` – overlapping sliding windows of prose.
* ``row``       – individual visual table rows.
* ``row_window`` – small windows over consecutive rows.
* ``image_ocr`` – OCR'd text from embedded images or rasterised pages
                  (only populated when ENABLE_OCR is on).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .config import (
    PARA_CHUNK_OVERLAP_WORDS,
    PARA_CHUNK_SIZE_WORDS,
    ROW_WINDOW_SIZE,
    ROW_WINDOW_STEP,
)
from .pdf_loader import row_has_table_shape
from .text_utils import normalize_text


def build_chunks(pages: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Return ``(chunks, metadata)`` in matching order."""
    chunks: List[str] = []
    metadata: List[Dict] = []

    def add_chunk(text: str, meta: Dict) -> None:
        clean = normalize_text(text)
        if len(clean) < 12:
            return
        meta = dict(meta)
        meta["chunk_id"] = len(chunks)
        meta["text"] = clean
        chunks.append(clean)
        metadata.append(meta)

    for page in pages:
        page_num = page["page"]
        section_hint = page.get("section_hint", "")
        lines = page["lines"]
        rows = page.get("rows") or lines
        text = page["text"]
        ocr_snippets = page.get("ocr_snippets", []) or []

        _emit_paragraph_chunks(text, page_num, section_hint, add_chunk)
        _emit_row_chunks(rows, page_num, section_hint, add_chunk)
        _emit_row_window_chunks(rows, page_num, section_hint, add_chunk)
        _emit_ocr_chunks(ocr_snippets, page_num, section_hint, add_chunk)

    return chunks, metadata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _emit_paragraph_chunks(text, page_num, section_hint, add_chunk) -> None:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    bucket: List[str] = []
    bucket_words = 0
    para_id = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if bucket_words + len(words) > PARA_CHUNK_SIZE_WORDS and bucket:
            chunk_text = " ".join(bucket)
            add_chunk(chunk_text, {
                "page": page_num,
                "section": section_hint,
                "chunk_type": "paragraph",
                "chunk_id_on_page": para_id,
            })
            para_id += 1
            overlap_words = chunk_text.split()[-PARA_CHUNK_OVERLAP_WORDS:] \
                if PARA_CHUNK_OVERLAP_WORDS else []
            bucket = [" ".join(overlap_words)] if overlap_words else []
            bucket_words = len(overlap_words)
        bucket.append(sent)
        bucket_words += len(words)

    if bucket:
        add_chunk(" ".join(bucket), {
            "page": page_num,
            "section": section_hint,
            "chunk_type": "paragraph",
            "chunk_id_on_page": para_id,
        })


def _emit_row_chunks(rows, page_num, section_hint, add_chunk) -> None:
    for i, row in enumerate(rows):
        if len(row) < 6:
            continue
        add_chunk(row, {
            "page": page_num,
            "section": section_hint,
            "chunk_type": "row",
            "row_id": i,
            "table_like": row_has_table_shape(row),
        })


def _emit_row_window_chunks(rows, page_num, section_hint, add_chunk) -> None:
    for start in range(0, max(1, len(rows)), ROW_WINDOW_STEP):
        window = rows[start:start + ROW_WINDOW_SIZE]
        if not window:
            continue
        add_chunk(" | ".join(window), {
            "page": page_num,
            "section": section_hint,
            "chunk_type": "row_window",
            "row_start": start,
            "row_end": min(start + ROW_WINDOW_SIZE - 1, len(rows) - 1),
            "table_like": any(row_has_table_shape(r) for r in window),
        })


def _emit_ocr_chunks(ocr_snippets, page_num, section_hint, add_chunk) -> None:
    """Emit a chunk per OCR snippet. Marked so the UI can show a badge."""
    for idx, snippet in enumerate(ocr_snippets):
        add_chunk(snippet, {
            "page": page_num,
            "section": section_hint,
            "chunk_type": "image_ocr",
            "ocr_id": idx,
        })
