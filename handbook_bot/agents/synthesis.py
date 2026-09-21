"""
Synthesis agent.

DECISION OWNED
    The wording of a grounded natural-language answer built from the evidence
    it is handed - and only from that evidence. It decides how to say it and
    how many sentences the question needs. It does not decide what evidence is
    relevant (retrieval does), whether the answer is supported, or which pages
    are finally cited (the Verifier does).

WHAT IT REUSES
    ``qa.build_prompt`` and ``qa.ask_groq`` are the Part 1 prompt builder and
    Groq call, unchanged apart from prompt rule 1. This module does not talk to
    Groq in any new way; it gives the existing call a typed result.

GROUNDING
    The prompt tells the model to use only the supplied context, never to
    guess, and to reply with the exact refusal sentence when the context does
    not support an answer. Nothing here can add a date, phone number, email,
    count, policy or page that is not in the evidence: ``claimed_pages`` is
    filtered to pages that were actually in the context, and it is a *claim*.
    The Verifier decides what is cited.

This agent issues exactly one LLM call per invocation and never retries by
itself. Whether a second draft is requested is the orchestrator's decision,
made from the Verifier's feedback.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..qa import ask_groq, build_prompt, split_answer_and_claimed_pages
from .types import SynthesisResult

#: Identifies the prompt that produced an answer, so an evaluation run can be
#: tied to the exact prompt wording. Bump it whenever build_prompt changes.
#: "2": rule 1 no longer caps the answer at "one or two short sentences".
PROMPT_VERSION = "synthesis-2"

#: Used when the deterministic extractor answered and no prompt was involved.
EXTRACTOR_VERSION = "extractor"


def synthesize(
    question: str,
    items: List[Dict],
    route: str,
    client,
    *,
    feedback: Optional[str] = None,
) -> SynthesisResult:
    """Write one grounded draft from ``items``. Makes exactly one LLM call.

    ``feedback`` is the Verifier's ``retry_feedback`` for a second draft, or
    None for a first draft. Any exception from the Groq call propagates: the
    orchestrator owns failure handling and call counting.
    """
    prompt = build_prompt(question, items, route, feedback=feedback)
    raw = ask_groq(client, prompt)
    answer, claimed_pages = split_answer_and_claimed_pages(raw, items)
    version = PROMPT_VERSION + ("+retry" if feedback else "")
    return SynthesisResult(answer=answer, claimed_pages=claimed_pages, prompt_version=version)


def from_extractor(answer: str, pages: List[int]) -> SynthesisResult:
    """Give a deterministic extractor answer the same shape as an LLM draft.

    No LLM is involved and nothing is reworded: extractor answers are built
    directly from handbook rows and are passed through untouched.
    """
    return SynthesisResult(answer=answer, claimed_pages=list(pages), prompt_version=EXTRACTOR_VERSION)
