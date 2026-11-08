"""
UOS Faculty Onboarding Chatbot - single-process entry point.

Run locally with:

    python app.py

The UI (NiceGUI) and the RAG pipeline live in the same Python process.
No separate frontend/backend servers, no Node, no Docker, no build step.

Open http://localhost:8501 after starting.
"""
from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from ui.chat import start_app


DEFAULT_PORT = 8501
APP_TITLE = "UOS Faculty Onboarding Chatbot"


def _model_warning() -> str | None:
    """Post-freeze PF-8 (D-119): one ``models.retrieve`` request (no inference) so a model this account cannot use is
    reported here instead of surfacing as "the service did not respond" on the first question."""
    from handbook_bot.config import GROQ_MODEL, LLM_REASONING_TOKENS
    from handbook_bot.groq_check import model_problem
    from handbook_bot.knowledge_base import _make_groq_client

    print(f"  Model: {GROQ_MODEL} (reasoning allowance {LLM_REASONING_TOKENS} tokens)")
    try:
        return model_problem(_make_groq_client(), GROQ_MODEL)
    except Exception as exc:                       # the banner must never stop the app
        return "Groq model check failed: %s" % type(exc).__name__


def _print_banner() -> None:
    line = "=" * 70
    print(line)
    print(f"  {APP_TITLE}")
    print(f"  Open http://localhost:{DEFAULT_PORT} in your browser.")
    warnings = []
    if not os.getenv("GROQ_API_KEY"):
        warnings.append(
            "GROQ_API_KEY is not set. Deterministic contact/date/count\n"
            "    answers will still work, but policy questions cannot reach\n"
            "    the LLM. Add your key to .env (see .env.example)."
        )
    else:
        problem = _model_warning()
        if problem:
            warnings.append(problem)
    if os.getenv("ENABLE_OCR", "").strip().lower() in {"1", "true", "yes", "on"}:
        warnings.append(
            "OCR is enabled. The FIRST run will take several extra minutes\n"
            "    to scan images in the PDF. Subsequent runs use the cache."
        )
    for w in warnings:
        print()
        print(f"  NOTE: {w}")
    print(line)


if __name__ == "__main__":
    _print_banner()
    try:
        start_app(port=DEFAULT_PORT)
    except KeyboardInterrupt:
        sys.exit(0)
