"""
Deterministic multi-finding aggregation of one Plan F execution (runtime integration, phase I-5).

DECISION OWNED
    Whether the structured findings that ONE Plan F execution already produced (the Coordinator's decision and the
    dispatched specialists' results: nothing re-run, nothing routed again, no handoff executed) together answer
    EVERY clause of the user's request completely, and in which order their verbatim quotes are rendered. All or
    nothing: a request clause without a verified finding, a specialist result that is not ``supported``, a handoff
    request, a shared or uncovered clause, a value conflict between equal asks or a rejected component finding means
    no aggregate at all, and the pipeline falls back to the legacy path. Nothing here calls a model, writes prose,
    resolves a conflict by score or changes a frozen specialist.

WHAT IT CONSUMES
    * the request clauses as the Coordinator split them (``decision.metadata["analysis"]["clauses"]``, in order);
    * the Coordinator's tasks and their ``focus_clauses``: a scoped task names the request clauses it owns; an
      unscoped task (one specialist given the whole question) carries the whole question as its one clause;
    * the dispatched runs: every task must have executed; each result's clause reports (``metadata["clauses"]``)
      and each finding's ``metadata["clauses"]`` (the report indexes it answers);
    * since phase I-6, the settled execution of one bounded handoff round (``plan_f_handoff``): a source task's
      handed-off clauses are owned by the handoff task that answered them (stamped ``handoff`` metadata, validated
      at depth one and listed in the decision's recorded round); the aggregator itself never executes a handoff.

HOW A CLAUSE IS COVERED (structured; never by word similarity)
    1. Ownership: a request clause is owned by the task whose focus names it exactly. A clause owned by two tasks
       is CROSS_DOMAIN_UNRESOLVED, one owned by none is UNRESOLVED_CLAUSE, and a task that owns no clause makes the
       execution CROSS_DOMAIN_UNRESOLVED. When a task carries the whole question as its one clause, its single
       report is shared by every request clause and is tried one finding at a time.
    2. Run-level safety, on every executed run, before any clause is read: ``supported`` status, no handoff, no
       ``missing`` entry, no model call, a clause report list, no unprocessed clause (``plan_f_verification.run_code``).
    3. Per clause, the phase I-4 findings-aware verifier on the clause's own view (``plan_f_verification.clause_view``):
       the report's outcome, ask shape, requested details and relation (``report_code``); per finding the quote
       guard, the subject binding, the requested detail, entities, population, modifiers, conditions, relation and
       provision boundary, and the provision slice of an excess quote (``verify_findings_sliced``). A shared report
       is tried one finding at a time and the eligible findings are then verified together, so the primary-first
       rule and the value-conflict rule of I-4 still apply.
    4. Conflicts: two request clauses that read as the same ask (equal ``RequestShape``) must not be answered with
       different values (a number or a bound contact value): VALUE_CONFLICT, nothing is chosen.
    5. Deduplication: a quote already rendered for an earlier clause (same chunk, same text) is not rendered again;
       the clause stays covered and the duplicate is recorded. A duplicate never covers a clause it did not verify for.
    6. Ordering: request clause order, then the specialist's finding order inside a clause, then the first
       occurrence of a quote. Nothing depends on set or dict iteration, scores or randomness.

RESULT
    :class:`Aggregation`: an immutable :class:`AggregationResult` (code, clause and coverage counts, per-clause
    coverage with its verification record, ordered evidence references, duplicates, conflicts, pages, specialists,
    completeness, ``safe_to_render``; ``to_dict`` is JSON serialisable, deterministic and free of question and
    answer text) and the ordered parts to render. :func:`render_parts` joins the verbatim quotes with blank lines:
    no label, no connective sentence, no rewording. The adapter (``plan_f_runtime``) adapts the parts into legacy
    evidence items, runs the unchanged legacy Verifier on the rendering and builds the ``QAResult``.

FAILURE POLICY
    :func:`aggregate` never raises: an unexpected exception becomes AGGREGATION_ERROR with the exception category
    only, ``accepted`` False and ``safe_to_render`` False.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import plan_f_verification as _verification
from .agents.coordinator import REQUESTED_BY
from .agents.specialists.base import redact_error
from .agents.specialists.contracts import CoordinatorDecision, Finding, SpecialistFindings, SpecialistTask
from .agents.specialists.dispatch import STATE_EXECUTED, DispatchResult
from .plan_f_runtime import PATH_AGGREGATE, PATH_PASSTHROUGH  # noqa: F401  (the adapter owns the path names; re-exported here)
from .plan_f_verification import FindingsVerification, ReasonCode, RequestKind, RequestShape

#: Integration phase and contract version of this module.
PHASE = "I-5"
VERSION = "plan-f-aggregation-1"

#: Completeness of the coverage of the request's clauses.
COMPLETE = "complete"
PARTIAL = "partial"
NOT_ASSESSED = "not_assessed"


class AggregationCode(str, Enum):
    """Why the aggregator did or did not produce one complete answer from the execution's findings."""

    AGGREGATED_ACCEPTED = "AGGREGATED_ACCEPTED"          # every request clause covered by verified findings, rendered in request order
    NO_AGGREGATION_REQUIRED = "NO_AGGREGATION_REQUIRED"  # one clause for one task: the single-finding pass-through path decides
    MULTIPART_UNSUPPORTED = "MULTIPART_UNSUPPORTED"      # the request's clauses or tasks cannot be read from the decision
    UNRESOLVED_CLAUSE = "UNRESOLVED_CLAUSE"              # a request clause no executed task owns (uncovered, pending or unprocessed)
    CROSS_DOMAIN_UNRESOLVED = "CROSS_DOMAIN_UNRESOLVED"  # a clause assigned to two specialists, or a specialist run that answers no clause
    HANDOFF_REQUIRED = "HANDOFF_REQUIRED"                # a specialist asked for another specialist; handoff rounds belong to phase I-6
    UNSAFE_RESULT = "UNSAFE_RESULT"                      # an error result, a model call, a malformed result or a task not issued by the Coordinator
    MISSING_REQUIRED_CLAUSE = "MISSING_REQUIRED_CLAUSE"  # a specialist did not support a required clause (partial, not found, out of scope, outcome)
    CHILD_FINDING_REJECTED = "CHILD_FINDING_REJECTED"    # the findings-aware verifier rejected a clause's finding (its code is in the detail)
    VALUE_CONFLICT = "VALUE_CONFLICT"                    # two clauses that are the same ask were answered with different values
    AGGREGATION_ERROR = "AGGREGATION_ERROR"              # the aggregator raised; the exception category only


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClauseCoverage:
    """What answered one request clause: the owning task and report, the verification of that view and the finding
    ids it accepted. ``shared_report`` marks a report that answers the whole question (an unscoped task)."""

    index: int
    task_id: Optional[str]
    specialist_id: Optional[str]
    report_index: Optional[int]
    shared_report: bool
    covered: bool
    reason: str
    finding_ids: Tuple[str, ...] = ()
    sliced: Tuple[Tuple[str, str, int, int], ...] = ()
    verification: Optional[FindingsVerification] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "task_id": self.task_id, "specialist_id": self.specialist_id, "report_index": self.report_index,
                "shared_report": self.shared_report, "covered": self.covered, "reason": self.reason[:200], "finding_ids": list(self.finding_ids),
                "sliced": [list(s) for s in self.sliced], "verification": self.verification.to_dict() if self.verification is not None else None}


@dataclass(frozen=True)
class EvidenceRef:
    """One rendered finding: its provenance and the request clauses it covers (several when a duplicate was dropped)."""

    finding_id: str
    task_id: str
    specialist_id: str
    source_id: str
    page: Optional[int]
    chunk_id: Optional[int]
    clause_indexes: Tuple[int, ...]
    derived_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"finding_id": self.finding_id, "task_id": self.task_id, "specialist_id": self.specialist_id, "source_id": self.source_id,
                "page": self.page, "chunk_id": self.chunk_id, "clause_indexes": list(self.clause_indexes), "derived_from": self.derived_from}


@dataclass(frozen=True)
class AggregationResult:
    """The deterministic verdict on one execution. ``safe_to_render`` is True only when ``accepted`` is True and at
    least one finding is rendered. ``to_dict`` leaves out the wall times, so equal inputs give equal records."""

    accepted: bool
    reason: AggregationCode
    detail: str
    clause_count: int
    covered_count: int
    clauses: Tuple[ClauseCoverage, ...]
    evidence: Tuple[EvidenceRef, ...]
    duplicates: Tuple[Tuple[str, str, int], ...]          # (dropped finding id, kept finding id, clause index)
    conflicts: Tuple[Tuple[int, int, str], ...]           # (clause index, clause index, request kind)
    specialists: Tuple[str, ...]
    pages: Tuple[int, ...]
    completeness: str
    safe_to_render: bool
    findings_considered: int
    dispatched: bool
    ms: float = 0.0
    timings: Mapping[str, float] = field(default_factory=dict)
    version: str = VERSION

    @property
    def findings_accepted(self) -> int:
        return len(self.evidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted, "reason": self.reason.value, "detail": self.detail[:200], "clause_count": self.clause_count,
            "covered_count": self.covered_count, "clauses": [c.to_dict() for c in self.clauses], "evidence": [e.to_dict() for e in self.evidence],
            "duplicates": [list(d) for d in self.duplicates], "conflicts": [list(c) for c in self.conflicts], "specialists": list(self.specialists),
            "pages": list(self.pages), "completeness": self.completeness, "safe_to_render": self.safe_to_render,
            "findings_considered": self.findings_considered, "findings_accepted": self.findings_accepted, "dispatched": self.dispatched,
            "version": self.version,
        }


@dataclass(frozen=True)
class AggregatePart:
    """One finding to render, for one request clause (the first clause it covers)."""

    clause_index: int
    clause: str
    task_id: str
    specialist_id: str
    finding: Finding


@dataclass(frozen=True)
class Aggregation:
    """The result and, when accepted, the ordered parts to render."""

    result: AggregationResult
    parts: Tuple[AggregatePart, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.result.accepted and self.result.safe_to_render and bool(self.parts)


# ---------------------------------------------------------------------------
# Reading the decision
# ---------------------------------------------------------------------------
def request_clauses(decision: CoordinatorDecision) -> List[str]:
    """The request clauses as the Coordinator split them, in order; empty when the decision carries none."""
    md = decision.metadata if isinstance(getattr(decision, "metadata", None), dict) else {}
    analysis = md.get("analysis") if isinstance(md.get("analysis"), dict) else {}
    clauses = analysis.get("clauses")
    if not isinstance(clauses, list):
        return []
    return [c for c in clauses if isinstance(c, str) and c.strip()]


def _issued_for(task: SpecialistTask, decision: CoordinatorDecision) -> bool:
    """A task the Coordinator issued, or a handoff task settled into the decision by the phase I-6 handoff round."""
    if task.requested_by == REQUESTED_BY:
        return True
    from . import plan_f_handoff as _handoff                       # imported here: that module builds on the verifier, as this one does
    return _handoff.validated_handoff_task(task, decision)


def _handoff_round_recorded(decision: CoordinatorDecision) -> bool:
    md = decision.metadata if isinstance(getattr(decision, "metadata", None), dict) else {}
    return isinstance(md.get("handoff_round"), dict)


def applicable(decision: CoordinatorDecision) -> bool:
    """Whether the aggregator, not the single-finding path, owns this decision: a Coordinator-issued, non-greeting
    decision with at least one specialist and either several request clauses or several tasks (or a settled handoff round)."""
    md = decision.metadata if isinstance(getattr(decision, "metadata", None), dict) else {}
    if md.get("complexity") == "greeting" or not decision.selected_specialists or not decision.subtasks:
        return False
    if any(not _issued_for(task, decision) for task in decision.subtasks):
        return False
    return len(request_clauses(decision)) >= 2 or len(decision.subtasks) >= 2 or _handoff_round_recorded(decision)


def clause_owners(decision: CoordinatorDecision) -> List[List[Tuple[SpecialistTask, int, bool]]]:
    """For every request clause, the (task, report index, shared) pairs that answer it: the tasks whose focus names
    the clause exactly or, when none does, every task whose one focus clause is the whole question (an unscoped task),
    whose single report is then shared by the clause. A focus that is neither is owned by nothing."""
    clauses = request_clauses(decision)
    owners: List[List[Tuple[SpecialistTask, int, bool]]] = [[] for _ in clauses]
    shared: List[Tuple[SpecialistTask, int]] = []
    for task in decision.subtasks:
        focus = (task.context or {}).get("focus_clauses") if isinstance(task.context, dict) else None
        focus = list(focus) if isinstance(focus, (list, tuple)) else []
        handed = (task.metadata or {}).get("handed_off_clauses") if isinstance(task.metadata, dict) else None
        handed = set(i for i in (handed or []) if isinstance(i, int)) if isinstance(handed, (list, tuple)) else set()
        for report_index, text in enumerate(focus):
            if not isinstance(text, str) or report_index in handed:     # a clause the task handed off is owned by the handoff task (phase I-6)
                continue
            hits = [ci for ci, clause in enumerate(clauses) if clause == text]
            if hits:
                for ci in hits:
                    owners[ci].append((task, report_index, False))
            elif len(focus) == 1 and text.strip() == (task.question or "").strip():
                shared.append((task, report_index))
    for ci in range(len(clauses)):
        if not owners[ci]:
            owners[ci] = [(task, report_index, True) for task, report_index in shared]
    return owners


def request_problem(decision: CoordinatorDecision) -> Optional[Tuple[AggregationCode, str]]:
    """Why the request cannot be aggregated whatever the specialists return, knowable from the decision alone (so the
    adapter need not dispatch): no clauses, a task not issued by the Coordinator, a clause the Coordinator left
    uncovered, a clause owned by two tasks, a task that owns no clause. None when the execution may be aggregated."""
    clauses = request_clauses(decision)
    if not clauses:
        return AggregationCode.MULTIPART_UNSUPPORTED, "the decision carries no request clauses"
    if not decision.subtasks or not decision.selected_specialists:
        return AggregationCode.MULTIPART_UNSUPPORTED, "the decision carries no specialist task"
    for task in decision.subtasks:
        if not _issued_for(task, decision):
            return AggregationCode.UNSAFE_RESULT, "task %s was not issued by the Coordinator or settled by the handoff round" % task.task_id
    md = decision.metadata if isinstance(decision.metadata, dict) else {}
    uncovered = md.get("uncovered_clauses") or []
    if uncovered:
        return AggregationCode.UNRESOLVED_CLAUSE, "%d request clause(s) belong to no selected specialist" % len(uncovered)
    owners = clause_owners(decision)
    owning: List[str] = []
    for ci, candidates in enumerate(owners):
        if not candidates:
            return AggregationCode.UNRESOLVED_CLAUSE, "request clause %d is owned by no task" % ci
        task_ids = list(dict.fromkeys(task.task_id for task, _r, _s in candidates))
        if len(task_ids) > 1:
            names = ", ".join(dict.fromkeys(task.specialist_id.value for task, _r, _s in candidates if task.specialist_id is not None))
            return AggregationCode.CROSS_DOMAIN_UNRESOLVED, "request clause %d is assigned to %s" % (ci, names)
        owning.extend(task_ids)
    for task in decision.subtasks:
        if task.task_id not in owning:
            return (AggregationCode.CROSS_DOMAIN_UNRESOLVED,
                    "task %s (%s) answers no request clause" % (task.task_id, task.specialist_id.value if task.specialist_id else "?"))
    return None


# ---------------------------------------------------------------------------
# Building results
# ---------------------------------------------------------------------------
def _result(accepted: bool, code: AggregationCode, detail: str, *, clause_count: int, coverage: Sequence[ClauseCoverage] = (),
            evidence: Sequence[EvidenceRef] = (), duplicates: Sequence[Tuple[str, str, int]] = (), conflicts: Sequence[Tuple[int, int, str]] = (),
            specialists: Sequence[str] = (), pages: Sequence[int] = (), considered: int = 0, dispatched: bool = False, started: float,
            timings: Optional[Mapping[str, float]] = None) -> AggregationResult:
    covered = sum(1 for c in coverage if c.covered)
    if accepted:
        completeness = COMPLETE
    elif covered:
        completeness = PARTIAL
    else:
        completeness = NOT_ASSESSED
    return AggregationResult(accepted=accepted, reason=code, detail=detail[:200], clause_count=clause_count, covered_count=covered,
                             clauses=tuple(coverage), evidence=tuple(evidence), duplicates=tuple(duplicates), conflicts=tuple(conflicts),
                             specialists=tuple(specialists), pages=tuple(pages), completeness=completeness,
                             safe_to_render=bool(accepted and evidence), findings_considered=considered, dispatched=dispatched,
                             ms=(time.perf_counter() - started) * 1000.0, timings=dict(timings or {}))


def request_rejection(decision: CoordinatorDecision, code: AggregationCode, detail: str) -> AggregationResult:
    """The record of a request rejected before any specialist ran (:func:`request_problem`)."""
    started = time.perf_counter()
    specialists = tuple(s.value for s in getattr(decision, "selected_specialists", []) or [])
    return _result(False, code, detail, clause_count=len(request_clauses(decision)), specialists=specialists, started=started)


def _clause_value(coverage: ClauseCoverage, shape: RequestShape, findings_by_id: Mapping[str, Finding]) -> Optional[Any]:
    """The value a covered clause was answered with: the primary accepted verdict's number, or the bound contact value
    of a contact ask. None when the request kind carries no comparable value."""
    verification = coverage.verification
    if verification is None:
        return None
    primary = next((v for v in verification.verdicts if v.accepted), None)
    if primary is None:
        return None
    if primary.value is not None:
        return primary.value
    if shape.kind is RequestKind.CONTACT and shape.field:
        finding = findings_by_id.get(primary.finding_id)
        values = finding.metadata.get("contact_values") if finding is not None and isinstance(finding.metadata, dict) else None
        value = (values or {}).get(shape.field) if isinstance(values, dict) else None
        return value.strip() if isinstance(value, str) and value.strip() else None
    return None


# ---------------------------------------------------------------------------
# The aggregator
# ---------------------------------------------------------------------------
def aggregate(decision: CoordinatorDecision, dispatched: DispatchResult, chunks: Sequence[str],
              metadata: Sequence[Dict[str, Any]], guide_index: Any = None) -> Aggregation:
    """Combine the findings of one execution into one complete answer, or say exactly why not. Deterministic; no
    model; no specialist call; never raises. ``guide_index`` (post-freeze PF-3C, D-111) lets the findings-aware verifier
    check a guide clause as one answer; without it a guide clause is refused."""
    started = time.perf_counter()
    timings: Dict[str, float] = {}
    clause_count = 0
    coverage: List[ClauseCoverage] = []
    specialists: Tuple[str, ...] = ()
    considered = 0
    ran = False

    def reject(code: AggregationCode, detail: str) -> Aggregation:
        return Aggregation(_result(False, code, detail, clause_count=clause_count, coverage=coverage, specialists=specialists,
                                   considered=considered, dispatched=ran, started=started, timings=timings))

    try:
        if not isinstance(decision, CoordinatorDecision):
            return reject(AggregationCode.MULTIPART_UNSUPPORTED, "not a CoordinatorDecision")
        clauses = request_clauses(decision)
        clause_count = len(clauses)
        specialists = tuple(s.value for s in decision.selected_specialists)
        if len(clauses) <= 1 and len(decision.subtasks) <= 1 and not _handoff_round_recorded(decision):
            return reject(AggregationCode.NO_AGGREGATION_REQUIRED, "one request clause for one specialist: the single-finding path decides")
        problem = request_problem(decision)
        if problem is not None:
            return reject(*problem)
        if not isinstance(dispatched, DispatchResult):
            return reject(AggregationCode.UNSAFE_RESULT, "no dispatch result")
        ran = True

        # ---- run-level safety on every task, before any clause is read -----------------------------------------
        t = time.perf_counter()
        if dispatched.llm_calls != 0:
            return reject(AggregationCode.UNSAFE_RESULT, "the dispatch reported %d LLM call(s)" % dispatched.llm_calls)
        runs = {run.task.task_id: run for run in dispatched.runs}
        results: Dict[str, SpecialistFindings] = {}
        for task in decision.subtasks:
            run = runs.get(task.task_id)
            if run is None or run.state != STATE_EXECUTED or run.findings is None:
                return reject(AggregationCode.UNRESOLVED_CLAUSE, "task %s was not executed" % task.task_id)
            results[task.task_id] = run.findings
        considered = sum(len(r.findings) for r in results.values())
        for task in decision.subtasks:
            findings = results[task.task_id]
            if findings.handoff_requests:
                targets = ", ".join(sorted({h.requested_specialist.value for h in findings.handoff_requests}))
                return reject(AggregationCode.HANDOFF_REQUIRED, "task %s requests a handoff to %s" % (task.task_id, targets))
            code, detail = _verification.run_code(findings)
            if code is not None:
                if code in (ReasonCode.PARTIAL_FINDING, ReasonCode.NOT_FOUND, ReasonCode.OUT_OF_SCOPE):
                    return reject(AggregationCode.MISSING_REQUIRED_CLAUSE, "task %s: %s" % (task.task_id, detail))
                if code is ReasonCode.HANDOFF_REQUIRED:
                    return reject(AggregationCode.HANDOFF_REQUIRED, "task %s: %s" % (task.task_id, detail))
                return reject(AggregationCode.UNSAFE_RESULT, "task %s: %s" % (task.task_id, detail))
            unprocessed = (findings.metadata or {}).get("unprocessed_clauses") or []
            if unprocessed:
                return reject(AggregationCode.UNRESOLVED_CLAUSE, "task %s left %d clause(s) unprocessed" % (task.task_id, len(unprocessed)))
        owners = clause_owners(decision)
        timings["map_ms"] = (time.perf_counter() - t) * 1000.0

        # ---- per clause: the findings-aware verifier on the clause's own view ------------------------------------
        t = time.perf_counter()
        parts: List[AggregatePart] = []
        for ci, clause in enumerate(clauses):
            task, report_index, shared = owners[ci][0]
            findings = results[task.task_id]
            sid = findings.specialist_id.value

            def fail(code: AggregationCode, detail: str, verification: Optional[FindingsVerification] = None,
                     _ci: int = ci, _task: SpecialistTask = task, _ri: int = report_index, _shared: bool = shared, _sid: str = sid) -> Aggregation:
                coverage.append(ClauseCoverage(_ci, _task.task_id, _sid, _ri, _shared, False, detail[:200], (), (), verification))
                timings["verify_ms"] = (time.perf_counter() - t) * 1000.0
                return reject(code, "clause %d: %s" % (_ci, detail))

            reports = findings.metadata["clauses"]
            if not 0 <= report_index < len(reports):
                return fail(AggregationCode.MISSING_REQUIRED_CLAUSE, "task %s has no clause report %d" % (task.task_id, report_index))
            code, detail = _verification.report_code(findings.specialist_id, reports[report_index])
            if code is not None:
                if code is ReasonCode.UNSUPPORTED_STATUS:
                    return fail(AggregationCode.MISSING_REQUIRED_CLAUSE, detail)
                return fail(AggregationCode.CHILD_FINDING_REJECTED, "%s: %s" % (code.value, detail))
            if shared:
                eligible: List[str] = []
                first_failure: Optional[FindingsVerification] = None
                for finding in findings.findings:
                    trial = _verification.verify_findings(clause, _verification.clause_view(findings, report_index, [finding.finding_id]),
                                                          chunks, metadata, guide_index)
                    if trial.accepted and trial.safe_to_render:
                        eligible.append(finding.finding_id)
                    elif first_failure is None:
                        first_failure = trial
                if not eligible:
                    assert first_failure is not None
                    return fail(AggregationCode.CHILD_FINDING_REJECTED, "%s: %s" % (first_failure.reason.value, first_failure.detail), first_failure)
                view = _verification.clause_view(findings, report_index, eligible)
            else:
                try:
                    view = _verification.clause_view(findings, report_index)
                except ValueError:
                    return fail(AggregationCode.MISSING_REQUIRED_CLAUSE, "no finding of task %s answers its clause report" % task.task_id)
            sliced = _verification.verify_findings_sliced(clause, view, chunks, metadata, guide_index)
            verification = sliced.verification
            if not verification.accepted or not verification.safe_to_render:
                return fail(AggregationCode.CHILD_FINDING_REJECTED, "%s: %s" % (verification.reason.value, verification.detail), verification)
            keep = set(verification.accepted_finding_ids)
            accepted = [f for f in sliced.findings.findings if f.finding_id in keep]
            coverage.append(ClauseCoverage(ci, task.task_id, sid, report_index, shared, True, "covered", tuple(f.finding_id for f in accepted),
                                           sliced.sliced, verification))
            parts.extend(AggregatePart(ci, clause, task.task_id, sid, f) for f in accepted)
        timings["verify_ms"] = (time.perf_counter() - t) * 1000.0

        # ---- conflicts between equal asks -------------------------------------------------------------------------
        t = time.perf_counter()
        shapes = [_verification.read_request(clause) for clause in clauses]
        findings_by_id = {part.finding.finding_id: part.finding for part in parts}
        conflicts: List[Tuple[int, int, str]] = []
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                if shapes[i] != shapes[j]:
                    continue
                left, right = _clause_value(coverage[i], shapes[i], findings_by_id), _clause_value(coverage[j], shapes[j], findings_by_id)
                if left is not None and right is not None and left != right:
                    conflicts.append((i, j, shapes[i].kind.value))
        timings["conflict_ms"] = (time.perf_counter() - t) * 1000.0
        if conflicts:
            i, j, kind = conflicts[0]
            return Aggregation(_result(False, AggregationCode.VALUE_CONFLICT,
                                       "clauses %d and %d are the same %s ask but are answered with different values" % (i, j, kind),
                                       clause_count=clause_count, coverage=coverage, conflicts=conflicts, specialists=specialists,
                                       considered=considered, dispatched=True, started=started, timings=timings))

        # ---- deduplication and the rendering order ------------------------------------------------------------------
        t = time.perf_counter()
        seen: Dict[Tuple[Any, str], AggregatePart] = {}
        rendered: List[AggregatePart] = []
        duplicates: List[Tuple[str, str, int]] = []
        covers: Dict[str, List[int]] = {}
        for part in parts:
            key = (part.finding.metadata.get("chunk_id"), part.finding.evidence_quote)
            kept = seen.get(key)
            if kept is not None:
                duplicates.append((part.finding.finding_id, kept.finding.finding_id, part.clause_index))
                if part.clause_index not in covers[kept.finding.finding_id]:
                    covers[kept.finding.finding_id].append(part.clause_index)
                continue
            seen[key] = part
            rendered.append(part)
            covers[part.finding.finding_id] = [part.clause_index]
        evidence = [EvidenceRef(part.finding.finding_id, part.task_id, part.specialist_id, part.finding.source_id, part.finding.page,
                                part.finding.metadata.get("chunk_id"), tuple(covers[part.finding.finding_id]),
                                part.finding.metadata.get("derived_from")) for part in rendered]
        pages: List[int] = []
        for ref in evidence:
            if isinstance(ref.page, int) and not isinstance(ref.page, bool) and ref.page not in pages:
                pages.append(ref.page)
        timings["dedup_ms"] = (time.perf_counter() - t) * 1000.0
        used = tuple(dict.fromkeys(part.specialist_id for part in rendered))
        result = _result(True, AggregationCode.AGGREGATED_ACCEPTED,
                         "%d clause(s) covered by %d verified finding(s) of %s" % (clause_count, len(rendered), "+".join(used)),
                         clause_count=clause_count, coverage=coverage, evidence=evidence, duplicates=duplicates, specialists=specialists,
                         pages=pages, considered=considered, dispatched=True, started=started, timings=timings)
        return Aggregation(result, tuple(rendered))
    except Exception as exc:                                        # a defect here must never become an answer
        category = redact_error(exc).split(":", 1)[0].strip() or type(exc).__name__      # the exception category only, never its text
        return reject(AggregationCode.AGGREGATION_ERROR, "aggregation raised: %s" % category)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_parts(parts: Sequence[AggregatePart]) -> str:
    """The verbatim quotes of the parts, in their order, one paragraph each. No label, no connective, no rewording; the
    first step of a D-111 guide procedure puts its printed heading on its own line (post-freeze PF-3C)."""
    from . import guide_grounding as _guides
    return "\n\n".join(_guides.render_part(part.finding.evidence_quote.strip(), (part.finding.metadata or {}).get(_guides.GUIDE_KEY))
                       for part in parts if part.finding.evidence_quote.strip())


__all__ = [
    "Aggregation", "AggregationCode", "AggregationResult", "AggregatePart", "COMPLETE", "ClauseCoverage", "EvidenceRef", "NOT_ASSESSED",
    "PARTIAL", "PATH_AGGREGATE", "PATH_PASSTHROUGH", "PHASE", "VERSION", "aggregate", "applicable", "clause_owners", "render_parts",
    "request_clauses", "request_problem", "request_rejection",
]
