"""
Shared interface contract between the agents and the orchestrator.

Source of truth: docs/WORK_SPLIT.md section 3 (agreed by the team). The Verifier
is written by a different engineer than the Router/Synthesis/Orchestrator, so
these shapes are the only thing both sides may rely on. Field names and field
order are part of the contract: change them only by a pull request reviewed by
both engineers.

One name differs from the draft contract, on purpose. The draft spells the
yes/no route ``"yes_no"`` while also saying "existing query_type names kept".
The existing name is ``"policy_yesno"``: it is what ``QAResult.query_type`` has
always carried, what ``retrieval._routing_boost`` keys its paragraph boost on,
and what ``eval/golden_set.jsonl`` expects. The codebase name is kept so that
retrieval behaviour and the routing-accuracy metric do not silently change.

This module has no imports from the rest of ``handbook_bot``, so any agent can
import it without risk of an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Route vocabulary
# ---------------------------------------------------------------------------
Route = Literal["greeting", "contact", "count", "date", "list", "policy_yesno", "policy"]

#: Every route, in the legacy first-match order. Used only as a last-resort,
#: deterministic tie-break so results never depend on dict ordering.
ALL_ROUTES: Tuple[str, ...] = (
    "greeting", "contact", "count", "date", "list", "policy_yesno", "policy",
)

#: Routes first offered to a deterministic extractor (a tool, not an agent).
#: If the extractor finds nothing they fall through to the LLM path.
EXTRACTOR_ROUTES: Tuple[str, ...] = ("contact", "count", "date")

#: Routes answered by retrieval + LLM synthesis.
LLM_ROUTES: Tuple[str, ...] = ("list", "policy_yesno", "policy")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@dataclass
class RouteDecision:
    primary_route: Route
    routes: list[Route]            # all intents that scored above threshold, best first
    is_multipart: bool
    confidence: float              # 0-1
    used_llm: bool
    scores: dict[str, float]       # every intent's score, for trace and tests


# ---------------------------------------------------------------------------
# Planner (Milestone 2) - the type exists now so QAResult.sub_questions is stable
# ---------------------------------------------------------------------------
@dataclass
class SubQuestion:
    question: str
    route: Route


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
@dataclass
class EvidenceResult:
    question: str
    route: Route
    items: list[dict]              # existing item dicts: chunk, meta{page, section, chunk_type}, scores
    best_score: Optional[float]
    sufficient: bool
    reason: Optional[str]          # "rerank_score_below_threshold" | "extractor_no_match" | "no_candidates" | None
    used_extractor: bool
    extractor_answer: Optional[str]
    doc_id: Optional[str] = None   # carried, unused while the corpus is a single document


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
@dataclass
class SynthesisResult:
    answer: str
    claimed_pages: list[int]       # pages the writer claimed; the Verifier decides what is cited
    prompt_version: str


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    accepted: bool
    pages: list[int]               # ONLY pages of items whose text supports the answer
    unsupported_claims: list[str]
    uncovered_subquestions: list[int]
    retry_feedback: Optional[str]  # None when accepted, or when refusing after the retry


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
@dataclass
class TraceEntry:
    agent: str                     # "router" | "planner" | "evidence" | "synthesis" | "verifier"
    decision: str
    used_llm: bool
    ms: float
    detail: dict = field(default_factory=dict)
