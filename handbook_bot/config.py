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


def _env_float(name: str, default: float) -> float:
    """Float from the environment. A malformed value falls back to the default
    instead of crashing the app at import time."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Int from the environment, with the same fall-back-to-default rule."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


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
# Refusal gate. Overridable from the environment so eval sweeps can vary it
# without editing this file; the default is the Part 1 (MCBV9) value.
MIN_RERANK_SCORE: float = _env_float("MIN_RERANK_SCORE", -1.4)

# Post-generation grounding check (qa.verify_answer). True is the Part 1
# behaviour. Set VERIFY_ANSWERS=0 to measure how many refusals it causes.
VERIFY_ANSWERS: bool = _env_bool("VERIFY_ANSWERS", True)

# ---------------------------------------------------------------------------
# Agents and orchestration (Part 2)
# ---------------------------------------------------------------------------
# Router. Deterministic scoring always runs first. One LLM call arbitrates only
# when the best two intents are closer than ROUTER_AMBIGUITY_MARGIN.
ROUTER_LLM_FALLBACK: bool = _env_bool("ROUTER_LLM_FALLBACK", True)
ROUTER_AMBIGUITY_MARGIN: float = _env_float("ROUTER_AMBIGUITY_MARGIN", 0.15)
# The router always has a deterministic answer to fall back on, so its LLM call
# is kept short and is never retried.
ROUTER_LLM_TIMEOUT_S: float = _env_float("ROUTER_LLM_TIMEOUT_S", 8.0)
ROUTER_LLM_MAX_TOKENS: int = 60

# Planner (Milestone 2). agents/planner.py does not exist yet, so this is off.
# The agreed target default is True: flip it when the Planner lands. With it
# off the system is the Option A pipeline: route -> answer -> verify.
PLANNER_ENABLED: bool = _env_bool("PLANNER_ENABLED", False)

# LLM-call budget per question. MAX_LLM_CALLS covers the normal calls (router
# fallback, synthesis; later the planner). A verifier-requested retry draws
# from its own allowance of MAX_VERIFY_RETRIES, so the hard ceiling per
# question is MAX_LLM_CALLS + MAX_VERIFY_RETRIES. The orchestrator refuses to
# issue a call beyond either limit.
MAX_LLM_CALLS: int = _env_int("MAX_LLM_CALLS", 2)
# The orchestrator clamps this to at most 1: a draft is retried once or not at all.
MAX_VERIFY_RETRIES: int = _env_int("MAX_VERIFY_RETRIES", 1)
# HTTP-level repeats of ONE logical answer call by the Groq SDK (429, 5xx,
# timeouts, with back-off). 2 is the SDK default, stated here so it is a visible
# setting. These repeats are not counted in llm_calls; set 0 to forbid them.
# The router's arbitration call never uses them.
LLM_TRANSPORT_RETRIES: int = _env_int("LLM_TRANSPORT_RETRIES", 2)

# ---------------------------------------------------------------------------
# Plan F budgets (INACTIVE until the Plan F orchestration path is wired in)
# ---------------------------------------------------------------------------
# Plan F is the findings-based multi-specialist architecture whose contracts
# and registry live in handbook_bot/agents/specialists/. Nothing in the
# current runtime reads these values: the Milestone 1 pipeline keeps
# MAX_LLM_CALLS and MAX_VERIFY_RETRIES above, unchanged. They are declared
# here so the limits are agreed and visible before the Coordinator exists.
# PLAN_F_ENABLED is the single switch a later phase will consult; it is off.
PLAN_F_ENABLED: bool = _env_bool("PLAN_F_ENABLED", False)
# Specialists the Coordinator may select for one question (first round).
PLAN_F_MAX_SPECIALISTS: int = _env_int("PLAN_F_MAX_SPECIALISTS", 3)
# Subtasks the Coordinator may create for one question.
PLAN_F_MAX_SUBTASKS: int = _env_int("PLAN_F_MAX_SUBTASKS", 3)
# Handoff rounds after the first: specialists request, the orchestrator runs
# at most this many further rounds. 1 means one handoff round, never more.
PLAN_F_MAX_HANDOFF_DEPTH: int = _env_int("PLAN_F_MAX_HANDOFF_DEPTH", 1)
# Verifier-requested retries of the final answer under Plan F.
PLAN_F_MAX_RETRY: int = _env_int("PLAN_F_MAX_RETRY", 1)
# Logical LLM calls per question under Plan F (coordinator, specialists,
# synthesis, retry). Higher than MAX_LLM_CALLS because several specialists
# may each need one call on a cross-domain question.
PLAN_F_MAX_LLM_CALLS: int = _env_int("PLAN_F_MAX_LLM_CALLS", 5)
# Specialist executions per question, handoffs included.
PLAN_F_MAX_AGENT_CALLS: int = _env_int("PLAN_F_MAX_AGENT_CALLS", 4)

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
