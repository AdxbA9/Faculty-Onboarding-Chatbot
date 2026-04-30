"""
Thin wrapper around the handbook_bot pipeline.

Keeps the UI layer decoupled from backend internals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from handbook_bot import KnowledgeBase, QAResult, answer_question, build_knowledge_base


@dataclass
class StartupCheck:
    ok: bool
    message: str = ""


def preflight() -> StartupCheck:
    """Cheap sanity checks before the slow KB build."""
    found = False
    for directory in ("data", "."):
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            if fname.lower().endswith(".pdf"):
                found = True
                break
        if found:
            break
    if not found:
        return StartupCheck(
            ok=False,
            message=(
                "No handbook PDF was found. Place the UOS Faculty Handbook "
                "PDF inside the 'data/' folder and restart the app."
            ),
        )
    return StartupCheck(ok=True)


def load_knowledge_base(verbose: bool = True) -> KnowledgeBase:
    """Build the knowledge base (embeddings + index). Called once at startup."""
    return build_knowledge_base(verbose=verbose)


def ask(kb: KnowledgeBase, question: str) -> QAResult:
    """Answer a single question against the loaded knowledge base."""
    return answer_question(
        question,
        embedder=kb.embedder,
        reranker=kb.reranker,
        index=kb.index,
        chunks=kb.chunks,
        metadata=kb.metadata,
        pages=kb.pages,
        groq_client=kb.groq_client,
    )


def groq_is_configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())
