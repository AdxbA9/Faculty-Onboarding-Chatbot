"""
Plan F contracts: the typed shapes exchanged between the Coordinator, the
domain specialists, the orchestrator and the Verifier.

Design rules (see docs/AGENT_CONTRACTS.md):

* A specialist receives a :class:`SpecialistTask` and returns a
  :class:`SpecialistFindings`. It never returns free-form prose as its result.
* A :class:`Finding` keeps ``claim`` (what the specialist asserts) and
  ``evidence_quote`` (the text it relies on) in SEPARATE fields, so the
  Verifier can later check one against the other. They are never merged.
* Specialists never call each other. A specialist that needs another domain
  returns a :class:`HandoffRequest`; the orchestrator decides whether to
  execute it.
* Canonical specialist ids and finding statuses are the enums below. Nothing
  else in the codebase should spell them as raw strings.

Every dataclass validates itself in ``__post_init__`` and raises
``ValueError`` on a malformed value, and offers ``to_dict()`` for traces and
run files. This module imports only the standard library, so any part of
``handbook_bot`` can import it without risk of an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Canonical vocabularies
# ---------------------------------------------------------------------------
class SpecialistId(str, Enum):
    """The four Plan F domain specialists. Orientation is a Coordinator journey
    mode and Blackboard/Banner are knowledge sources: none of them is an id."""

    TEACHING = "teaching"
    RESEARCH = "research"
    FACULTY_SERVICES = "faculty_services"
    INSTITUTIONAL = "institutional"

    @property
    def display_name(self) -> str:
        return _DISPLAY_NAMES[self]

    @classmethod
    def parse(cls, value: Union["SpecialistId", str]) -> "SpecialistId":
        """Accept an enum member or its string value; reject anything else."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise ValueError(
                "unknown specialist id %r (expected one of %s)" % (value, ", ".join(ALL_SPECIALIST_IDS))
            ) from None


_DISPLAY_NAMES = {
    SpecialistId.TEACHING: "Teaching & Learning",
    SpecialistId.RESEARCH: "Research & Innovation",
    SpecialistId.FACULTY_SERVICES: "Faculty Services & HR",
    SpecialistId.INSTITUTIONAL: "Institutional Navigation",
}

#: Every canonical id, in declaration order, as plain strings.
ALL_SPECIALIST_IDS = tuple(member.value for member in SpecialistId)


class FindingStatus(str, Enum):
    """Outcome of one specialist run.

    supported     every part of the task is answered by findings with evidence
    partial       some of the task is answered; ``missing`` says what is not
    not_found     the task is in scope but the approved sources do not cover it
    out_of_scope  another specialist owns the task (usually with a handoff request)
    error         the specialist could not run (exception, timeout, invalid output)
    """

    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    OUT_OF_SCOPE = "out_of_scope"
    ERROR = "error"

    @classmethod
    def parse(cls, value: Union["FindingStatus", str]) -> "FindingStatus":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            raise ValueError(
                "unknown finding status %r (expected one of %s)" % (value, ", ".join(ALL_FINDING_STATUSES))
            ) from None


#: Every status, in declaration order, as plain strings.
ALL_FINDING_STATUSES = tuple(member.value for member in FindingStatus)


# ---------------------------------------------------------------------------
# Small validators shared by the dataclasses
# ---------------------------------------------------------------------------
def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % name)
    return value


def _require_confidence(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number between 0 and 1" % name)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("%s must be between 0 and 1, got %r" % (name, value))
    return float(value)


def _require_str_list(name: str, value: Any) -> List[str]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("%s must be a list of strings" % name)
    out = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("%s must contain only strings" % name)
        out.append(item)
    return out


def _require_dict(name: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("%s must be a dict" % name)
    return value


# ---------------------------------------------------------------------------
# SpecialistTask
# ---------------------------------------------------------------------------
@dataclass
class SpecialistTask:
    """One unit of work assigned to one specialist.

    ``source_scope`` lists the source ids the specialist may retrieve from
    (empty means "the specialist's own default scope"). ``requested_by`` is
    ``"coordinator"`` for a first-round task or the id of the specialist whose
    handoff request produced it.
    """

    task_id: str
    question: str
    specialist_id: Optional[SpecialistId] = None
    domain: Optional[str] = None
    intent: Optional[str] = None
    level: Optional[str] = None            # "university" | "college" | "department" | None
    system: Optional[str] = None           # "blackboard" | "banner" | "myuos" | ... | None
    source_scope: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    requested_by: str = "coordinator"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("task_id", self.task_id)
        _require_text("question", self.question)
        if self.specialist_id is not None:
            self.specialist_id = SpecialistId.parse(self.specialist_id)
        for name in ("domain", "intent", "level", "system"):
            value = getattr(self, name)
            if value is not None:
                _require_str(name, value)
        self.source_scope = _require_str_list("source_scope", self.source_scope)
        self.context = _require_dict("context", self.context)
        _require_text("requested_by", self.requested_by)
        self.metadata = _require_dict("metadata", self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "specialist_id": self.specialist_id.value if self.specialist_id else None,
            "domain": self.domain,
            "intent": self.intent,
            "level": self.level,
            "system": self.system,
            "source_scope": list(self.source_scope),
            "context": dict(self.context),
            "requested_by": self.requested_by,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    """One grounded statement made by a specialist.

    ``claim`` is what the specialist asserts in its own words. ``evidence_quote``
    is the passage from the source that the claim rests on. The two are kept
    apart on purpose: the Verifier checks that the quote exists in the cited
    source and that the claim is supported by the quote. ``page`` is the
    1-based page inside ``source_id`` when the source is paginated.
    """

    finding_id: str
    claim: str
    evidence_quote: str
    source_id: str
    source_title: str = ""
    page: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("finding_id", self.finding_id)
        _require_text("claim", self.claim)
        _require_text("evidence_quote", self.evidence_quote)   # a finding without a quote is not grounded
        _require_text("source_id", self.source_id)
        _require_str("source_title", self.source_title)
        if self.page is not None:
            if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
                raise ValueError("page must be a positive integer or None, got %r" % (self.page,))
        self.confidence = _require_confidence("confidence", self.confidence)
        self.metadata = _require_dict("metadata", self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "claim": self.claim,
            "evidence_quote": self.evidence_quote,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "page": self.page,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# HandoffRequest
# ---------------------------------------------------------------------------
@dataclass
class HandoffRequest:
    """A specialist asking the orchestrator to run another specialist.

    Returned inside :class:`SpecialistFindings`; never executed by the
    specialist itself. The orchestrator validates the request against its
    budgets (one handoff round at most) before acting on it.
    """

    from_specialist: SpecialistId
    requested_specialist: SpecialistId
    reason: str
    task: Optional[SpecialistTask] = None
    priority: int = 0                      # higher runs first when several requests compete
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.from_specialist = SpecialistId.parse(self.from_specialist)
        self.requested_specialist = SpecialistId.parse(self.requested_specialist)
        if self.from_specialist == self.requested_specialist:
            raise ValueError("a specialist cannot request a handoff to itself (%s)" % self.from_specialist.value)
        _require_text("reason", self.reason)
        if self.task is not None and not isinstance(self.task, SpecialistTask):
            raise ValueError("task must be a SpecialistTask or None")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("priority must be an integer")
        self.metadata = _require_dict("metadata", self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_specialist": self.from_specialist.value,
            "requested_specialist": self.requested_specialist.value,
            "reason": self.reason,
            "task": self.task.to_dict() if self.task else None,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# SpecialistFindings
# ---------------------------------------------------------------------------
@dataclass
class SpecialistFindings:
    """The structured result of one specialist run.

    Invariants enforced here:

    * ``status`` is a :class:`FindingStatus`.
    * ``supported`` requires at least one finding and an empty ``missing``:
      it means grounded AND complete.
    * ``partial`` requires at least one finding; a specialist that found
      nothing must say ``not_found``.
    * finding ids are unique within the result.
    * handoff requests originate from this specialist and never target it.
    """

    specialist_id: SpecialistId
    task_id: str
    status: FindingStatus
    findings: List[Finding] = field(default_factory=list)
    summary: str = ""
    missing: List[str] = field(default_factory=list)          # what the task asked that no finding covers
    handoff_requests: List[HandoffRequest] = field(default_factory=list)
    confidence: float = 0.0
    limitations: List[str] = field(default_factory=list)      # caveats the answer must carry
    llm_used: bool = False
    ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.specialist_id = SpecialistId.parse(self.specialist_id)
        _require_text("task_id", self.task_id)
        self.status = FindingStatus.parse(self.status)
        if not isinstance(self.findings, list) or not all(isinstance(f, Finding) for f in self.findings):
            raise ValueError("findings must be a list of Finding")
        ids = [f.finding_id for f in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding ids must be unique within a result")
        if self.status is FindingStatus.SUPPORTED and not self.findings:
            raise ValueError("status 'supported' requires at least one finding")
        if self.status is FindingStatus.PARTIAL and not self.findings:
            raise ValueError("status 'partial' requires at least one finding; use 'not_found' when there are none")
        _require_str("summary", self.summary)
        self.missing = _require_str_list("missing", self.missing)
        if self.status is FindingStatus.SUPPORTED and self.missing:
            raise ValueError("status 'supported' means complete: 'missing' must be empty (use 'partial')")
        if not isinstance(self.handoff_requests, list) or not all(
            isinstance(h, HandoffRequest) for h in self.handoff_requests
        ):
            raise ValueError("handoff_requests must be a list of HandoffRequest")
        for request in self.handoff_requests:
            if request.from_specialist != self.specialist_id:
                raise ValueError(
                    "handoff request from %s inside findings of %s"
                    % (request.from_specialist.value, self.specialist_id.value)
                )
        self.confidence = _require_confidence("confidence", self.confidence)
        self.limitations = _require_str_list("limitations", self.limitations)
        if not isinstance(self.llm_used, bool):
            raise ValueError("llm_used must be a bool")
        if isinstance(self.ms, bool) or not isinstance(self.ms, (int, float)) or self.ms < 0:
            raise ValueError("ms must be a non-negative number")
        self.ms = float(self.ms)
        self.metadata = _require_dict("metadata", self.metadata)

    @property
    def requested_agents(self) -> List[SpecialistId]:
        """Ids of the specialists this result asks the orchestrator to run."""
        return [request.requested_specialist for request in self.handoff_requests]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specialist_id": self.specialist_id.value,
            "task_id": self.task_id,
            "status": self.status.value,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "missing": list(self.missing),
            "handoff_requests": [h.to_dict() for h in self.handoff_requests],
            "requested_agents": [s.value for s in self.requested_agents],
            "confidence": self.confidence,
            "limitations": list(self.limitations),
            "llm_used": self.llm_used,
            "ms": self.ms,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# CoordinatorDecision (contract only; no Coordinator logic exists yet)
# ---------------------------------------------------------------------------
@dataclass
class CoordinatorDecision:
    """What the Coordinator decided about a question.

    ``selected_specialists`` are the first-round specialists, best first.
    ``subtasks`` are the tasks handed to them; every subtask's
    ``specialist_id`` must be one of the selected specialists.
    ``requires_synthesis`` is False when one supported specialist can be
    rendered deterministically without an LLM.
    """

    domains: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    level: Optional[str] = None
    systems: List[str] = field(default_factory=list)
    selected_specialists: List[SpecialistId] = field(default_factory=list)
    subtasks: List[SpecialistTask] = field(default_factory=list)
    requires_synthesis: bool = False
    confidence: float = 0.0
    reason: str = ""
    used_llm: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.domains = _require_str_list("domains", self.domains)
        self.intents = _require_str_list("intents", self.intents)
        if self.level is not None:
            _require_str("level", self.level)
        self.systems = _require_str_list("systems", self.systems)
        if isinstance(self.selected_specialists, str) or not isinstance(self.selected_specialists, Sequence):
            raise ValueError("selected_specialists must be a list of specialist ids")
        self.selected_specialists = [SpecialistId.parse(s) for s in self.selected_specialists]
        if len(self.selected_specialists) != len(set(self.selected_specialists)):
            raise ValueError("selected_specialists must not contain duplicates")
        if not isinstance(self.subtasks, list) or not all(isinstance(t, SpecialistTask) for t in self.subtasks):
            raise ValueError("subtasks must be a list of SpecialistTask")
        task_ids = [t.task_id for t in self.subtasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("subtask ids must be unique")
        for task in self.subtasks:
            if task.specialist_id is None:
                raise ValueError("subtask %s has no specialist_id" % task.task_id)
            if task.specialist_id not in self.selected_specialists:
                raise ValueError(
                    "subtask %s is assigned to %s, which is not a selected specialist"
                    % (task.task_id, task.specialist_id.value)
                )
        if not isinstance(self.requires_synthesis, bool):
            raise ValueError("requires_synthesis must be a bool")
        self.confidence = _require_confidence("confidence", self.confidence)
        _require_str("reason", self.reason)
        if not isinstance(self.used_llm, bool):
            raise ValueError("used_llm must be a bool")
        self.metadata = _require_dict("metadata", self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": list(self.domains),
            "intents": list(self.intents),
            "level": self.level,
            "systems": list(self.systems),
            "selected_specialists": [s.value for s in self.selected_specialists],
            "subtasks": [t.to_dict() for t in self.subtasks],
            "requires_synthesis": self.requires_synthesis,
            "confidence": self.confidence,
            "reason": self.reason,
            "used_llm": self.used_llm,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "ALL_FINDING_STATUSES",
    "ALL_SPECIALIST_IDS",
    "CoordinatorDecision",
    "Finding",
    "FindingStatus",
    "HandoffRequest",
    "SpecialistFindings",
    "SpecialistId",
    "SpecialistTask",
]
