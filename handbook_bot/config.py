"""
Central configuration for the UOS Faculty Onboarding Chatbot.

Every tunable parameter lives here, so experiments stay cheap.
"""
from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBED_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
PARA_CHUNK_SIZE_WORDS: int = 160
PARA_CHUNK_OVERLAP_WORDS: int = 50
ROW_WINDOW_SIZE: int = 4
ROW_WINDOW_STEP: int = 2

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_DENSE: int = 36
TOP_K_LEXICAL: int = 18
RERANK_CANDIDATES: int = 24
FINAL_K: int = 5
MIN_RERANK_SCORE: float = -1.4

# ---------------------------------------------------------------------------
# OCR / image extraction
# ---------------------------------------------------------------------------
# OCR is OFF by default because it adds minutes to the first-run index build.
# Enable it by setting ENABLE_OCR=1 in your .env file.
# Uses rapidocr-onnxruntime (pure pip, no system Tesseract needed).
ENABLE_OCR: bool = _env_bool("ENABLE_OCR", False)

# Minimum embedded image size (total pixels) worth OCRing - filters icons.
OCR_MIN_IMAGE_PIXELS: int = 40_000  # e.g. ~200x200

# Pages with fewer than this many characters of normal text get their whole
# page rasterised and OCRed (catches fully-scanned or image-heavy pages).
OCR_PAGE_TEXT_THRESHOLD: int = 60

# DPI for full-page rasterisation when running page-level OCR.
OCR_PAGE_DPI: int = 180

# Minimum OCR text length to keep - throws out very noisy extractions.
OCR_MIN_TEXT_CHARS: int = 12

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: str = "data"
CACHE_DIR: str = "cache"

# Bump when ingestion logic changes so old caches invalidate automatically.
CACHE_VERSION: str = "v11"

# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 512

# How many characters of evidence to show under the answer in the UI.
EVIDENCE_PREVIEW_CHARS: int = 520
