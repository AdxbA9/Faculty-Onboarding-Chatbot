"""
PDF loading with layout-aware text extraction and optional OCR.

We prefer PyMuPDF (fitz) because it exposes word-level coordinates, which
lets us rebuild table rows from handbook pages that mix tables with
prose. If PyMuPDF is unavailable we fall back to pypdf for plain-text
extraction.

When OCR is enabled (``ENABLE_OCR=1`` in .env), we ALSO:

* OCR every sufficiently-large embedded image on each page
* OCR a full-page raster of pages that are text-sparse (likely scans or
  heavy diagrams/figures)

OCR uses rapidocr-onnxruntime (pip-only, no system dependencies). If the
library isn't installed or fails to load, OCR silently becomes a no-op -
nothing else in the pipeline breaks.
"""
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ENABLE_OCR,
    OCR_MIN_IMAGE_PIXELS,
    OCR_MIN_TEXT_CHARS,
    OCR_PAGE_DPI,
    OCR_PAGE_TEXT_THRESHOLD,
)
from .text_utils import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    normalize_line,
    normalize_text,
)

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# OCR engine (lazy-loaded singleton)
# ---------------------------------------------------------------------------
_OCR_ENGINE: Any = None
_OCR_INIT_TRIED: bool = False
_OCR_AVAILABLE: bool = False


def _get_ocr_engine() -> Optional[Any]:
    """Lazy-initialise the RapidOCR engine. Returns None if unavailable."""
    global _OCR_ENGINE, _OCR_INIT_TRIED, _OCR_AVAILABLE
    if _OCR_INIT_TRIED:
        return _OCR_ENGINE if _OCR_AVAILABLE else None
    _OCR_INIT_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
        _OCR_AVAILABLE = True
        return _OCR_ENGINE
    except Exception as exc:
        print(f"[OCR] rapidocr-onnxruntime unavailable, skipping OCR. ({exc})")
        _OCR_AVAILABLE = False
        return None


def _run_ocr(image_bytes: bytes) -> str:
    """Run OCR on raw image bytes and return joined text, or '' on failure."""
    engine = _get_ocr_engine()
    if engine is None:
        return ""
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        arr = np.array(img)
        result, _elapsed = engine(arr)
        if not result:
            return ""
        # Each result row is (box, text, confidence).
        lines = [
            row[1].strip()
            for row in result
            if len(row) >= 2 and isinstance(row[1], str) and row[1].strip()
        ]
        return normalize_text(" ".join(lines))
    except Exception as exc:
        print(f"[OCR] extraction failed: {exc}")
        return ""


def ocr_stats() -> Dict[str, Any]:
    """Small diagnostic used by the UI's developer mode."""
    return {
        "enabled": ENABLE_OCR,
        "available": _OCR_AVAILABLE,
        "tried": _OCR_INIT_TRIED,
    }


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------
def find_pdf_file(search_dirs: Optional[List[str]] = None) -> str:
    """Return the path to the first PDF found in any of ``search_dirs``."""
    if search_dirs is None:
        search_dirs = ["data", "."]
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        pdfs = sorted(
            f for f in os.listdir(directory) if f.lower().endswith(".pdf")
        )
        if pdfs:
            return os.path.join(directory, pdfs[0])
    raise FileNotFoundError(
        "No PDF file was found. Place the handbook PDF in ./data/ or the "
        "project root and try again."
    )


# ---------------------------------------------------------------------------
# Heading / row heuristics
# ---------------------------------------------------------------------------
def detect_heading(line: str) -> bool:
    """Guess whether a line is a section heading."""
    if not line or len(line) > 115 or PHONE_PATTERN.search(line):
        return False
    letters = re.sub(r"[^A-Za-z]", "", line)
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    numbered = bool(re.match(r"^\d+(?:\.\d+)*\s+", line))
    title_like = line.istitle() and len(line.split()) <= 11
    return numbered or upper_ratio > 0.72 or title_like


def row_has_table_shape(row: str) -> bool:
    """True if a row looks like a table row (pipes, phone or email)."""
    pipes = row.count(" | ")
    return pipes >= 2 or bool(PHONE_PATTERN.search(row)) or bool(EMAIL_PATTERN.search(row))


# ---------------------------------------------------------------------------
# PyMuPDF row reconstruction
# ---------------------------------------------------------------------------
def _extract_rows_from_pymupdf_page(page) -> List[str]:
    """Rebuild visual rows from word-level coordinates."""
    words = page.get_text("words") or []
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))

    rows: List[List[Tuple[float, str]]] = []
    current: List[Tuple[float, str]] = []
    current_y: Optional[float] = None
    tolerance = 2.8

    for w in words:
        x0, y0, txt = w[0], w[1], str(w[4])
        if current_y is None or abs(y0 - current_y) <= tolerance:
            current.append((x0, txt))
            current_y = y0 if current_y is None else (current_y + y0) / 2
        else:
            rows.append(current)
            current = [(x0, txt)]
            current_y = y0
    if current:
        rows.append(current)

    out: List[str] = []
    for row in rows:
        row.sort(key=lambda z: z[0])
        text = normalize_line(" ".join(t for _, t in row))
        if text:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# OCR collection per page
# ---------------------------------------------------------------------------
def _ocr_page(page, page_text_length: int) -> List[str]:
    """Return a list of OCR'd text snippets for the page.

    Two passes:
      1. Every sufficiently-large embedded image on the page.
      2. If the page's normal text is very short, OCR a full-page raster.
    """
    if not ENABLE_OCR:
        return []
    engine = _get_ocr_engine()
    if engine is None:
        return []

    snippets: List[str] = []
    seen: set = set()

    # Pass 1: embedded images
    try:
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                pix_dict = page.parent.extract_image(xref)
            except Exception:
                continue
            if not pix_dict:
                continue
            width = pix_dict.get("width", 0)
            height = pix_dict.get("height", 0)
            if width * height < OCR_MIN_IMAGE_PIXELS:
                continue
            image_bytes = pix_dict.get("image")
            if not image_bytes:
                continue
            text = _run_ocr(image_bytes)
            if len(text) >= OCR_MIN_TEXT_CHARS and text not in seen:
                seen.add(text)
                snippets.append(text)
    except Exception as exc:
        print(f"[OCR] page {page.number + 1} image pass failed: {exc}")

    # Pass 2: full-page raster for text-sparse pages
    if page_text_length < OCR_PAGE_TEXT_THRESHOLD:
        try:
            zoom = OCR_PAGE_DPI / 72.0
            mat = fitz.Matrix(zoom, zoom) if fitz is not None else None
            if mat is not None:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                png_bytes = pix.tobytes("png")
                text = _run_ocr(png_bytes)
                if len(text) >= OCR_MIN_TEXT_CHARS and text not in seen:
                    seen.add(text)
                    snippets.append(text)
        except Exception as exc:
            print(f"[OCR] page {page.number + 1} raster pass failed: {exc}")

    return snippets


# ---------------------------------------------------------------------------
# Header/footer stripping
# ---------------------------------------------------------------------------
def _strip_headers_and_footers(raw_pages: List[Dict]) -> List[Dict]:
    """Drop lines/rows that repeat across many pages (running header/footer)."""
    line_counts: Dict[str, int] = {}
    row_counts: Dict[str, int] = {}
    page_lines: List[List[str]] = []
    page_rows: List[List[str]] = []

    for page in raw_pages:
        lines = [normalize_line(x) for x in page["raw_lines"] if normalize_line(x)]
        rows = [normalize_line(x) for x in page.get("raw_rows", []) if normalize_line(x)]
        page_lines.append(lines)
        page_rows.append(rows)
        for line in set(lines[:3] + lines[-3:]):
            if len(line) >= 4:
                line_counts[line] = line_counts.get(line, 0) + 1
        for row in set(rows[:3] + rows[-3:]):
            if len(row) >= 4:
                row_counts[row] = row_counts.get(row, 0) + 1

    threshold = max(4, int(len(raw_pages) * 0.45))
    repeated = {
        line
        for line, count in {**line_counts, **row_counts}.items()
        if count >= threshold
        and (
            len(line) > 70
            or re.search(r"faculty handbook|university of sharjah|page\s+\d+", line, re.I)
        )
    }

    cleaned: List[Dict] = []
    current_section = ""
    for page, lines, rows in zip(raw_pages, page_lines, page_rows):
        filtered_lines = [line for line in lines if line not in repeated]
        filtered_rows = [row for row in rows if row not in repeated]

        heading_source = filtered_lines[:8] if filtered_lines else filtered_rows[:8]
        for line in heading_source:
            if detect_heading(line):
                current_section = line
                break

        cleaned.append({
            "page": page["page"],
            "lines": filtered_lines,
            "rows": filtered_rows,
            "text": normalize_text(" ".join(filtered_lines or filtered_rows)),
            "section_hint": current_section,
            "ocr_snippets": page.get("ocr_snippets", []),
        })

    return [p for p in cleaned if p["text"] or p.get("ocr_snippets")]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_pdf(path: str) -> List[Dict]:
    """Extract text + visual rows + (optionally) OCR snippets from every page.

    Returns list of dicts with ``page``, ``lines``, ``rows``, ``text``,
    ``section_hint``, ``ocr_snippets``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found.")

    raw_pages: List[Dict] = []

    if fitz is not None:
        doc = fitz.open(path)
        try:
            ocr_pages_done = 0
            total_pages = len(doc)
            for i, page in enumerate(doc):
                txt = page.get_text("text") or ""
                rows = _extract_rows_from_pymupdf_page(page)
                ocr_snippets: List[str] = []
                if ENABLE_OCR:
                    ocr_snippets = _ocr_page(page, len(txt.strip()))
                    if ocr_snippets:
                        ocr_pages_done += 1
                raw_pages.append({
                    "page": i + 1,
                    "raw_lines": txt.splitlines(),
                    "raw_rows": rows,
                    "ocr_snippets": ocr_snippets,
                })
                if ENABLE_OCR and (i + 1) % 25 == 0:
                    print(f"[OCR] scanned {i + 1}/{total_pages} pages "
                          f"({ocr_pages_done} with extracted text)")
        finally:
            doc.close()
    elif PdfReader is not None:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            raw_pages.append({
                "page": i + 1,
                "raw_lines": txt.splitlines(),
                "raw_rows": txt.splitlines(),
                "ocr_snippets": [],
            })
    else:
        raise RuntimeError(
            "Neither PyMuPDF nor pypdf is installed. "
            "Install them via `pip install -r requirements.txt`."
        )

    return _strip_headers_and_footers(raw_pages)
