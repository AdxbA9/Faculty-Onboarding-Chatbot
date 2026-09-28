"""
Coordinator to specialist dispatch (multi-agent architecture, Step 3.10).

DECISION OWNED
    None. This module is the deterministic seam between the Faculty
    Onboarding Coordinator and the domain specialists: it takes a
    ``CoordinatorDecision``, runs the specialists that are executable in the
    current phase on the tasks the Coordinator assigned to them, and returns
    the decision together with their ``SpecialistFindings``. It never
    routes, never answers, never writes prose and never verifies.

WHAT IT DOES IN THIS PHASE
    * Only the Teaching & Learning Specialist is executable
      (``EXECUTABLE_SPECIALISTS``). A task assigned to any other specialist
      is kept, in Coordinator order, as ``pending``: it is never executed,
      never faked and never handed to Teaching. There is no default-to-
      Teaching fallback.
    * Tasks are passed to the specialist exactly as the Coordinator built
      them (task id, full question, intent, level, system and
      ``context["focus_clauses"]``); nothing is reconstructed.
    * Handoff requests a specialist returns are collected and preserved;
      they are not executed (handoff rounds are a later milestone).
    * A specialist that raises is reported as an ``error`` finding for its
      task, with key-shaped secrets and filesystem paths removed; the
      decision and the other tasks are unaffected.
    * No LLM call is made here; ``DispatchResult.llm_calls`` counts the
      specialists' own ``llm_used`` flags (0 for the deterministic Teaching
      specialist).

NOT WIRED INTO THE RUNTIME
    Nothing in ``handbook_bot.orchestrator`` or ``answer_question`` imports
    this module. ``PLAN_F_ENABLED`` stays False. The seam exists so that the
    Coordinator to Teaching path can be tested end to end.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..coordinator import coordinate
from .base import Specialist, check_findings
from .contracts import CoordinatorDecision, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from .registry import SpecialistRegistry

#: Specialists that may be executed in this phase (Step 3.10: Teaching only).
EXECUTABLE_SPECIALISTS: Tuple[SpecialistId, ...] = (SpecialistId.TEACHING,)

STATE_EXECUTED = "executed"
STATE_PENDING = "pending"

_KEY_SHAPED = re.compile(r"gsk_[A-Za-z0-9]+")
_PATH_SHAPED = re.compile(
    r"[A-Za-z]:[\\/][^\s'\"<>|]+"
    r"|(?<![\w:./])/(?:home|users|tmp|var|etc|opt|mnt|root|srv|usr|c|d|data|app|workspace)/[^\s'\"<>|]*"
    r"|(?<![\w:./])/[\w.-]+(?:/[\w.-]+)+",
    re.I)


def redact_error(exc: BaseException) -> str:
    """The exception type and a short message with key-shaped secrets and
    filesystem paths removed (the specialists' own error convention)."""
    text = _KEY_SHAPED.sub("gsk_***", str(exc))
    text = _PATH_SHAPED.sub("<path>", text)
    text = re.sub(r"\s+", " ", text)[:160]
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__


def error_findings(specialist_id: SpecialistId, task: SpecialistTask, exc: BaseException) -> SpecialistFindings:
    """An ``error`` result for ``task`` when the specialist raised instead of
    returning findings. Missing content is never reported this way."""
    return SpecialistFindings(
        specialist_id=specialist_id, task_id=task.task_id, status=FindingStatus.ERROR,
        summary="specialist execution failed", confidence=0.0, llm_used=False,
        metadata={"error": redact_error(exc), "raised_by": specialist_id.value},
    )


@dataclass
class SpecialistRun:
    """One Coordinator task and what happened to it."""

    task: SpecialistTask
    specialist_id: SpecialistId
    state: str                                  # STATE_EXECUTED or STATE_PENDING
    findings: Optional[SpecialistFindings] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "specialist_id": self.specialist_id.value,
            "state": self.state,
            "reason": self.reason,
            "findings": self.findings.to_dict() if self.findings is not None else None,
        }


@dataclass
class DispatchResult:
    """The Coordinator's decision plus the specialists' results, in
    Coordinator order. Pending tasks stay visible; nothing is reordered."""

    question: str
    decision: CoordinatorDecision
    runs: List[SpecialistRun] = field(default_factory=list)

    @property
    def executed_specialists(self) -> List[SpecialistId]:
        return [r.specialist_id for r in self.runs if r.state == STATE_EXECUTED]

    @property
    def pending_specialists(self) -> List[SpecialistId]:
        return [r.specialist_id for r in self.runs if r.state == STATE_PENDING]

    @property
    def results(self) -> List[SpecialistFindings]:
        return [r.findings for r in self.runs if r.findings is not None]

    @property
    def handoffs(self) -> List[HandoffRequest]:
        return [h for f in self.results for h in f.handoff_requests]

    @property
    def llm_calls(self) -> int:
        return sum(1 for f in self.results if f.llm_used)

    def result_for(self, task_id: str) -> Optional[SpecialistFindings]:
        return next((r.findings for r in self.runs if r.task.task_id == task_id), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "decision": self.decision.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "executed_specialists": [s.value for s in self.executed_specialists],
            "pending_specialists": [s.value for s in self.pending_specialists],
            "handoffs": [h.to_dict() for h in self.handoffs],
            "llm_calls": self.llm_calls,
        }


def dispatch(decision: CoordinatorDecision, registry: SpecialistRegistry, *,
             executable: Sequence[SpecialistId] = EXECUTABLE_SPECIALISTS,
             question: Optional[str] = None) -> DispatchResult:
    """Run the executable specialists on the tasks the Coordinator assigned
    to them and keep every other task pending, in Coordinator order.

    The decision is used as given: the specialist ids, the task order, the
    task ids and the focus clauses are never reinterpreted. The number of
    tasks is whatever the Coordinator produced under its own budgets; no
    further limit is applied here.
    """
    if not isinstance(decision, CoordinatorDecision):
        raise TypeError("dispatch() expects a CoordinatorDecision, got %s" % type(decision).__name__)
    if not isinstance(registry, SpecialistRegistry):
        raise TypeError("dispatch() expects a SpecialistRegistry, got %s" % type(registry).__name__)
    allowed = {SpecialistId.parse(s) for s in executable}
    runs: List[SpecialistRun] = []
    seen_ids: set = set()
    for task in decision.subtasks:
        # The decision was validated when it was built; these two checks only
        # catch a decision altered afterwards (a non-task, a repeated id).
        if not isinstance(task, SpecialistTask):
            raise TypeError("Coordinator subtasks must be SpecialistTask objects, got %s" % type(task).__name__)
        if task.task_id in seen_ids:
            raise ValueError("Coordinator task ids are not unique: %s" % task.task_id)
        seen_ids.add(task.task_id)
        sid = task.specialist_id
        if sid is None:                                   # cannot happen for a valid decision; kept as a guard
            raise ValueError("Coordinator task %s has no specialist id" % task.task_id)
        if sid not in allowed:
            runs.append(SpecialistRun(task, sid, STATE_PENDING, None,
                                      "%s is not executable in this phase" % sid.value))
            continue
        if not registry.is_registered(sid):
            runs.append(SpecialistRun(task, sid, STATE_PENDING, None,
                                      "%s is executable but not registered" % sid.value))
            continue
        specialist: Specialist = registry.resolve(sid)
        try:
            findings = check_findings(specialist, task, specialist.run(task))
        except Exception as exc:                          # the specialist's failure never fails the decision
            findings = error_findings(sid, task, exc)
        runs.append(SpecialistRun(task, sid, STATE_EXECUTED, findings, "executed"))
    text = question if question is not None else (decision.subtasks[0].question if decision.subtasks else "")
    return DispatchResult(question=text, decision=decision, runs=runs)


def run_question(question: str, registry: SpecialistRegistry, *,
                 executable: Sequence[SpecialistId] = EXECUTABLE_SPECIALISTS,
                 max_specialists: Optional[int] = None,
                 max_subtasks: Optional[int] = None) -> DispatchResult:
    """Coordinate ``question`` and dispatch the result. ``max_specialists``
    and ``max_subtasks`` are passed to the Coordinator unchanged (tests
    only); the dispatcher adds no limit of its own."""
    decision = coordinate(question, max_specialists=max_specialists, max_subtasks=max_subtasks)
    return dispatch(decision, registry, executable=executable, question=question)


__all__ = ["DispatchResult", "EXECUTABLE_SPECIALISTS", "STATE_EXECUTED", "STATE_PENDING", "SpecialistRun",
           "dispatch", "error_findings", "redact_error", "run_question"]
