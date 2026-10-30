"""
Retrieval route of one focus clause (Milestone 4, Step 4.7; D-078).

The Coordinator classifies one intent per task from the joined focus
clauses. A specialist that reused that task intent as the retrieval route of
every clause let one contact clause ("what is the VCRGS phone number") make
the substantive clause beside it retrieve as a contact lookup, whose ranking
favours directory rows over the paragraph that answers it (Step 4.6 MAJOR).
:func:`resolve_clause_retrieval_intent` derives each clause's route from the
clause itself with the Coordinator's deterministic router. Shared by the
Teaching & Learning and Research & Innovation specialists; no LLM, no
network, no state.
"""
from __future__ import annotations

from typing import Optional

from ..router import classify as _classify_route
from ..types import ALL_ROUTES

_DEFAULT_ROUTE = "policy"


def _known_route(intent: Optional[str]) -> str:
    """An existing retrieval route, or the default; a greeting is no route."""
    return intent if intent in ALL_ROUTES and intent != "greeting" else _DEFAULT_ROUTE


def _contact_clause(clause: str, substantive: Optional[bool]) -> bool:
    """Whether ``clause`` is itself a contact or location ask: the Coordinator's
    deterministic router labels it ``contact`` and the caller's own analysis,
    when it has one, found no substantive half beside the contact half."""
    return _classify_route(clause) == "contact" and not substantive


def resolve_clause_retrieval_intent(clause: str, task_intent: Optional[str], *, sole_clause: bool = False,
                                    substantive: Optional[bool] = None) -> str:
    """The retrieval route of ONE focus clause.

    The Coordinator classifies one intent per task from the joined focus
    clauses, so a contact clause ("what is the VCRGS phone number") can make
    the task intent ``contact`` for the substantive clause beside it. The
    contact route ranks directory rows and phone-bearing lines first, and the
    paragraph that answers the substantive clause can fall out of the
    candidate window (Step 4.6 MAJOR). A clause therefore derives its route
    from itself, never from another clause's words:

    * ``sole_clause``: the task intent was classified from this clause alone,
      or stated for it by the caller, so it stays the route; there is no
      other clause to contaminate it;
    * otherwise the clause is classified on its own with the Coordinator's
      deterministic router (the same classifier that produced the task
      intent), so one clause never sets another clause's route;
    * the ``contact`` route is used only by a clause that is itself a contact
      or location ask; ``substantive`` is the caller's own finding of a
      substantive half in the clause (``None`` when it has no such analysis).
      Any other clause given ``contact`` retrieves under the default
      ``policy`` route, as it would on its own;
    * ``greeting`` and unknown labels fall back to ``policy`` as before.

    Procedure and system handling, ownership, handoffs and the evidence
    checks are untouched: this decides retrieval ranking only. Deterministic:
    no LLM, no network, no state.
    """
    route = _known_route(task_intent) if sole_clause else _known_route(_classify_route(clause))
    if route == "contact" and not _contact_clause(clause, substantive):
        return _DEFAULT_ROUTE
    return route


__all__ = ["resolve_clause_retrieval_intent"]
