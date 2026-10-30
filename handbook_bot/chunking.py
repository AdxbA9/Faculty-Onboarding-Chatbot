"""
Chunking strategy for the handbook.

Four complementary chunk types per page:

* ``paragraph`` – overlapping sliding windows of prose.
* ``row``       – individual visual table rows.
* ``row_window`` – small windows over consecutive rows.
* ``image_ocr`` – OCR'd text from embedded images or rasterised pages
                  (only populated when ENABLE_OCR is on).

Post-freeze PF-2 adds two chunk types for system manuals only (``procedures=True``;
the handbook keeps exactly the four types above and its chunk list is unchanged):

* ``procedure_step``   – one numbered or lettered step with its continuation
                         lines, prefixed by the procedure's local heading.
* ``procedure_window`` – three consecutive steps of one procedure, so a
                         multi-step answer keeps its order and neighbours.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .config import (
    PARA_CHUNK_OVERLAP_WORDS,
    PARA_CHUNK_SIZE_WORDS,
    ROW_WINDOW_SIZE,
    ROW_WINDOW_STEP,
)
from .pdf_loader import detect_heading, row_has_table_shape
from .text_utils import normalize_text

#: A numbered step ("1.", "2)", "Step 3:") at the start of a visual row; the text may continue on the next row.
_NUMBERED_STEP = re.compile(r"^\s*(?:step\s+)?(\d{1,2})\s*[.):](?:\s+(\S.*))?$", re.I)
#: A lettered step ("a)", "B.") needs its text on the same row, so a figure callout letter is never a step.
_LETTERED_STEP = re.compile(r"^\s*([A-Za-z])\s*[.)]\s+(\S.*)$")
#: A bulleted item: hyphen, dash or bullet glyph (the guides use private-use glyphs for Wingdings bullets).
_BULLET_STEP = re.compile("^\\s*[-–•●▪■]\\s+(\\S.*)$")
_PROCEDURE_WINDOW_SIZE = 3
_PROCEDURE_WINDOW_STEP = 2
_MIN_STEP_WORDS = 2


def _match_step(row: str):
    """``(label, text)`` when ``row`` opens a step, else None. The label is the marker as printed. A short bulleted row
    that ends with ':' or '?' is a bulleted heading (" Your Courses:"), not a step."""
    m = _NUMBERED_STEP.match(row)
    if m:
        return "%s." % m.group(1), (m.group(2) or "")
    m = _LETTERED_STEP.match(row)
    if m:
        return "%s)" % m.group(1), m.group(2)
    m = _BULLET_STEP.match(row)
    if m:
        text = m.group(1).strip()
        if text.endswith((":", "?")) and len(text.split()) <= 8:
            return None
        return "-", text
    return None


def build_chunks(pages: List[Dict], *, procedures: bool = False) -> Tuple[List[str], List[Dict]]:
    """Return ``(chunks, metadata)`` in matching order. ``procedures`` adds the
    procedure chunk types (system manuals); the default output is unchanged."""
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
        if procedures:
            _emit_procedure_chunks(rows, page_num, section_hint, add_chunk)

    return chunks, metadata


# ---------------------------------------------------------------------------
# Procedure chunks (system manuals; post-freeze PF-2)
# ---------------------------------------------------------------------------
_HEADING_WORD = re.compile("[A-Za-z]{3,}")
_BULLET_GLYPHS = "-–•●▪■ \t"


def _clean_heading(row: str) -> str:
    return row.strip().lstrip(_BULLET_GLYPHS).rstrip(":").strip()


def _is_procedure_heading(row: str) -> bool:
    """A row that introduces a procedure or a new list: ends with ':' or '?', or looks like a heading. A row without a
    word of three letters (figure callout letters such as "A E") is never a heading."""
    text = row.strip()
    if not _HEADING_WORD.search(text) or _match_step(text) is not None:
        return False
    return text.endswith((":", "?")) or detect_heading(text)


def _is_list_lead_in(row: str) -> bool:
    """A short row without sentence-ending punctuation ("How to upload a document", "Checking Similarity Report") that
    stands directly above the first step of a list names that list."""
    text = row.strip()
    return (3 <= len(text) <= 80 and len(text.split()) <= 12 and bool(_HEADING_WORD.search(text))
            and not text.endswith((".", "!", ";", ",")))


def read_procedures(rows: List[str], section_hint: str = "") -> List[Dict]:
    """Group the visual rows of one page into procedures: ``{"heading", "steps": [{"label", "text"}]}``.

    A step starts at a numbered, lettered or bulleted row; the rows that follow it and are neither a step nor a
    heading are its continuation; a heading-like row closes the open procedure and becomes the heading of the next
    one, and a numbered step that restarts at 1 opens a new procedure under the same heading. The heading is the
    nearest heading row above the first step, else a short lead-in row standing directly above it, else the page's
    section hint. Deterministic and purely lexical; a step's label is the marker as printed and nothing is inferred
    about its meaning.
    """
    procedures: List[Dict] = []
    heading = _clean_heading(section_hint) if section_hint else ""
    steps: List[Dict] = []
    lead_in: Optional[str] = None                 # the plain row seen last while no list was open
    last_continuation: Optional[str] = None       # the row last appended to the open step

    def close() -> None:
        nonlocal steps
        kept = [s for s in steps if len(s["text"].split()) >= _MIN_STEP_WORDS]
        if kept:
            procedures.append({"heading": heading, "steps": kept})
        steps = []

    for row in rows:
        step = _match_step(row)
        if step is not None:
            label, text = step
            if label == "1." and steps:
                # a list restarts: a lead-in row that was swallowed as the previous step's continuation names the new list
                if last_continuation is not None and _is_list_lead_in(last_continuation):
                    cut = normalize_text(last_continuation)
                    if steps[-1]["text"].endswith(cut):
                        steps[-1]["text"] = steps[-1]["text"][: -len(cut)].rstrip()
                    lead_in = last_continuation.strip()
                close()
            if not steps and lead_in is not None:
                heading = _clean_heading(lead_in)
            lead_in = None
            last_continuation = None
            steps.append({"label": label, "text": normalize_text(text)})
            continue
        if _is_procedure_heading(row):
            close()
            heading = _clean_heading(row)
            lead_in = None
            last_continuation = None
            continue
        if steps and len(row.strip()) >= 3:
            steps[-1]["text"] = normalize_text((steps[-1]["text"] + " " + row.strip()).strip())
            last_continuation = row
        elif not steps:
            lead_in = row.strip() if _is_list_lead_in(row) else None
    close()
    return procedures


def _step_text(step: Dict) -> str:
    return "%s %s" % (step["label"], step["text"])


def _emit_procedure_chunks(rows, page_num, section_hint, add_chunk) -> None:
    for proc_index, procedure in enumerate(read_procedures(rows, section_hint)):
        heading: Optional[str] = procedure["heading"] or None
        steps = procedure["steps"]
        prefix = (heading + ": ") if heading else ""
        for i, step in enumerate(steps):
            add_chunk(prefix + _step_text(step), {
                "page": page_num,
                "section": section_hint,
                "chunk_type": "procedure_step",
                "procedure_id": proc_index,
                "procedure_heading": heading,
                "step_label": step["label"],
                "step_index": i,
                "step_count": len(steps),
            })
        if len(steps) >= 2:
            for start in range(0, len(steps), _PROCEDURE_WINDOW_STEP):
                window = steps[start:start + _PROCEDURE_WINDOW_SIZE]
                if len(window) < 2 and start > 0:
                    break
                add_chunk(prefix + " | ".join(_step_text(s) for s in window), {
                    "page": page_num,
                    "section": section_hint,
                    "chunk_type": "procedure_window",
                    "procedure_id": proc_index,
                    "procedure_heading": heading,
                    "step_start": start,
                    "step_end": start + len(window) - 1,
                    "step_count": len(steps),
                })
                if start + _PROCEDURE_WINDOW_SIZE >= len(steps):
                    break


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
