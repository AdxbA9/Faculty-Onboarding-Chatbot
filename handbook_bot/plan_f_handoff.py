"""
One bounded handoff round for a Plan F execution (runtime integration, phase I-6).

DECISION OWNED
    Whether the handoff requests the dispatched specialists returned may be executed, the execution of each valid
    one exactly once, and the settled execution (decision and dispatch result) the aggregator then reads. This is a
    bounded retry, not an agent conversation: depth one, one execution per specialist per request, no query
    invention, no model call, no routing change. Everything a target returns goes through the same I-1 guards, the
    I-4 findings-aware verifier and the I-5 all-or-nothing aggregation as a first-round result.

WHAT MAKES A HANDOFF EXECUTABLE (every rule must hold, or no handoff of the request runs)
    * the target is one of the four registered specialists and is registered in the runtime's registry;
    * the target is not the source (self loop) and has not executed for this request (A to B to A, a repeated
      target, a target that already ran in the first round): visited-specialist protection, no cycles;
    * the request carries the specialist's own handoff task: the target's id, the source as ``requested_by``,
      and exactly ONE focus clause, which is passed to the target unchanged (no rewriting);
    * that clause is a request clause the source task owned and declined: the source's clause report for it has
      outcome ``handed_off``. A handoff for the whole of a multipart question, for a clause the source answered
      itself (a "shared" ownership note) or for text outside the request binds to no unresolved clause;
    * every first-round result the request still depends on (the source's remaining clauses, every other run) is
      supported, so that executing the handoff can complete the request; otherwise no target runs.
    A target that itself asks for a handoff is the second hop: blocked, the request falls back.

RESULT
    :func:`run_handoffs` returns an immutable :class:`HandoffRound` (code, depth, visited specialists, one record per
    request with its code, the executed targets' status and timing; ``to_dict`` JSON-serialisable, deterministic
    and free of question text) and, when every request executed, the settled decision and dispatch result: the
    source task keeps only the clauses it did not hand off (or is dropped when it handed off every clause), the
    handoff task joins the decision with a validated ``handoff`` record, the source result loses the executed
    requests from its pending list, and the target's run joins the dispatch result. Nothing else is altered.

FAILURE POLICY
    Never raises: a raising target or a defect here becomes ``HANDOFF_EXCEPTION`` with the exception category only,
    and no settled execution is returned, so the adapter falls back.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents.specialists.base import check_findings, redact_error
from .agents.specialists.contracts import CoordinatorDecision, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from .agents.specialists.dispatch import STATE_EXECUTED, DispatchResult, SpecialistRun
from .agents.specialists.registry import SpecialistRegistry
from .plan_f_verification import ReasonCode, run_code

PHASE = "I-6"
VERSION = "plan-f-handoff-1"
#: The one and only handoff depth this phase executes.
MAX_DEPTH = 1
#: The four specialists a handoff may target.
APPROVED_TARGETS = frozenset(s.value for s in SpecialistId)
#: Clause-report outcomes that mean "the source declined this clause for its owner".
DECLINED_OUTCOMES = frozenset({"handed_off"})


class HandoffCode(str, Enum):
    HANDOFF_NONE = "HANDOFF_NONE"                                      # no specialist asked for a handoff
    HANDOFF_EXECUTED = "HANDOFF_EXECUTED"                              # every request was valid and its target ran once
    HANDOFF_TARGET_INVALID = "HANDOFF_TARGET_INVALID"                  # not one of the four registered specialists, or a malformed request or task
    HANDOFF_SELF_LOOP = "HANDOFF_SELF_LOOP"                            # source and target are the same specialist
    HANDOFF_TARGET_ALREADY_VISITED = "HANDOFF_TARGET_ALREADY_VISITED"  # the target already executed for this request (cycle or repeated target)
    HANDOFF_CLAUSE_UNRESOLVED = "HANDOFF_CLAUSE_UNRESOLVED"            # the handoff binds to no request clause the source declined
    HANDOFF_REMAINDER_UNSUPPORTED = "HANDOFF_REMAINDER_UNSUPPORTED"    # a first-round result the request still depends on is not supported: no target runs
    HANDOFF_SECOND_HOP_BLOCKED = "HANDOFF_SECOND_HOP_BLOCKED"          # the target asked for another handoff
    HANDOFF_UNSAFE_RESULT = "HANDOFF_UNSAFE_RESULT"                    # the target reported a model call
    HANDOFF_EXCEPTION = "HANDOFF_EXCEPTION"                            # the target or the round raised; the exception category only


@dataclass(frozen=True)
class HandoffRecord:
    source: str
    target: str
    task_id: Optional[str]
    request_clause: Optional[int]
    report_index: Optional[int]
    kind: Optional[str]
    code: HandoffCode
    detail: str
    executed: bool = False
    status: Optional[str] = None
    findings: int = 0
    second_hop: Tuple[str, ...] = ()
    ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target, "task_id": self.task_id, "request_clause": self.request_clause,
                "report_index": self.report_index, "kind": self.kind, "code": self.code.value, "detail": self.detail[:200], "executed": self.executed,
                "status": self.status, "findings": self.findings, "second_hop": list(self.second_hop)}


@dataclass(frozen=True)
class HandoffRound:
    attempted: bool
    code: HandoffCode
    detail: str
    depth: int
    visited: Tuple[str, ...]
    records: Tuple[HandoffRecord, ...]
    specialist_calls: int
    ms: float = 0.0
    version: str = VERSION

    @property
    def executed(self) -> Tuple[HandoffRecord, ...]:
        return tuple(r for r in self.records if r.executed)

    @property
    def sources(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(r.source for r in self.records))

    @property
    def targets(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(r.target for r in self.records))

    def to_dict(self) -> Dict[str, Any]:
        return {"attempted": self.attempted, "code": self.code.value, "detail": self.detail[:200], "depth": self.depth, "visited": list(self.visited),
                "records": [r.to_dict() for r in self.records], "specialist_calls": self.specialist_calls, "sources": list(self.sources),
                "targets": list(self.targets), "version": self.version}


_Settled = Tuple[HandoffRound, Optional[CoordinatorDecision], Optional[DispatchResult]]


# ---------------------------------------------------------------------------
# Reading the first round
# ---------------------------------------------------------------------------
def _focus(task: SpecialistTask) -> List[str]:
    focus = (task.context or {}).get("focus_clauses") if isinstance(task.context, dict) else None
    return [c for c in (focus if isinstance(focus, (list, tuple)) else []) if isinstance(c, str)]


def _bind_clause(text: str, clauses: Sequence[str], question: str) -> Optional[int]:
    """The request clause ``text`` names: an exact clause, or the whole question of a one-clause request."""
    for ci, clause in enumerate(clauses):
        if clause == text:
            return ci
    if len(clauses) == 1 and text.strip() == (question or "").strip():
        return 0
    return None


def _report_for(findings: SpecialistFindings, report_index: int) -> Optional[Dict[str, Any]]:
    reports = (findings.metadata or {}).get("clauses")
    if not isinstance(reports, list) or not 0 <= report_index < len(reports) or not isinstance(reports[report_index], dict):
        return None
    return reports[report_index]


def _without_requests(findings: SpecialistFindings, executed: Sequence[HandoffRequest]) -> SpecialistFindings:
    """The source result with the executed requests no longer pending; everything else unchanged."""
    keep = [h for h in findings.handoff_requests if not any(h is e for e in executed)]
    return SpecialistFindings(specialist_id=findings.specialist_id, task_id=findings.task_id, status=findings.status, findings=list(findings.findings),
                              summary=findings.summary, missing=list(findings.missing), handoff_requests=keep, confidence=findings.confidence,
                              limitations=list(findings.limitations), llm_used=findings.llm_used, ms=findings.ms, metadata=dict(findings.metadata))


# ---------------------------------------------------------------------------
# The round
# ---------------------------------------------------------------------------
def run_handoffs(decision: CoordinatorDecision, dispatched: DispatchResult, registry: SpecialistRegistry, clauses: Sequence[str]) -> _Settled:
    """Validate every handoff request of the dispatched runs, execute each valid target once, and settle the execution
    for the aggregator. No handoff runs unless every request is valid. Deterministic; no model; never raises."""
    started = time.perf_counter()
    records: List[HandoffRecord] = []
    visited: List[str] = []
    calls = 0

    def done(code: HandoffCode, detail: str, attempted: bool = True) -> HandoffRound:
        return HandoffRound(attempted=attempted, code=code, detail=detail[:200], depth=MAX_DEPTH if any(r.executed for r in records) else 0,
                            visited=tuple(visited), records=tuple(records), specialist_calls=calls, ms=(time.perf_counter() - started) * 1000.0)

    try:
        question = dispatched.question if isinstance(dispatched, DispatchResult) else ""
        runs = [r for r in dispatched.runs if r.state == STATE_EXECUTED and r.findings is not None]
        visited = list(dict.fromkeys(r.specialist_id.value for r in runs))
        pending = [(run, request) for run in runs for request in run.findings.handoff_requests]
        if not pending:
            return done(HandoffCode.HANDOFF_NONE, "no handoff requested", attempted=False), decision, dispatched
        tasks_by_id = {task.task_id: task for task in decision.subtasks}

        # ---- validation of every request, before any execution -----------------------------------------------------
        plan: List[Tuple[SpecialistRun, HandoffRequest, HandoffRecord]] = []
        claimed: List[str] = list(visited)
        for run, request in pending:
            source = run.specialist_id.value
            target_id = getattr(request, "requested_specialist", None)
            target = target_id.value if isinstance(target_id, SpecialistId) else str(target_id)
            task = getattr(request, "task", None)
            meta = request.metadata if isinstance(getattr(request, "metadata", None), dict) else {}
            kind = meta.get("kind") if isinstance(meta.get("kind"), str) else None

            def invalid(code: HandoffCode, detail: str, ci: Optional[int] = None, ri: Optional[int] = None) -> HandoffRecord:
                return HandoffRecord(source, target, getattr(task, "task_id", None), ci, ri, kind, code, detail)

            if not isinstance(target_id, SpecialistId) or target not in APPROVED_TARGETS:
                records.append(invalid(HandoffCode.HANDOFF_TARGET_INVALID, "target is not one of the four specialists"))
                continue
            if not registry.is_registered(target_id):
                records.append(invalid(HandoffCode.HANDOFF_TARGET_INVALID, "target is not registered"))
                continue
            if target == source:
                records.append(invalid(HandoffCode.HANDOFF_SELF_LOOP, "a specialist cannot hand a clause to itself"))
                continue
            if target in claimed:
                records.append(invalid(HandoffCode.HANDOFF_TARGET_ALREADY_VISITED, "target already executed for this request"))
                continue
            focus = _focus(task) if isinstance(task, SpecialistTask) else []
            if (not isinstance(task, SpecialistTask) or task.specialist_id != target_id or task.requested_by != source or len(focus) != 1
                    or not focus[0].strip()):
                records.append(invalid(HandoffCode.HANDOFF_TARGET_INVALID, "the request carries no well-formed handoff task for one clause"))
                continue
            ci = _bind_clause(focus[0], clauses, question)
            if ci is None:
                records.append(invalid(HandoffCode.HANDOFF_CLAUSE_UNRESOLVED, "the handoff names no single request clause"))
                continue
            source_task = tasks_by_id.get(run.task.task_id, run.task)
            source_focus = _focus(source_task)
            ri: Optional[int] = None
            for index, text in enumerate(source_focus):
                if text == clauses[ci] or (len(clauses) == 1 and text.strip() == question.strip()):
                    ri = index
                    break
            if ri is None:
                records.append(invalid(HandoffCode.HANDOFF_CLAUSE_UNRESOLVED, "the source task does not own the handed-off clause", ci))
                continue
            report = _report_for(run.findings, ri)
            if report is None or report.get("outcome") not in DECLINED_OUTCOMES:
                outcome = report.get("outcome") if report else None
                records.append(invalid(HandoffCode.HANDOFF_CLAUSE_UNRESOLVED, "the source did not decline the clause (report outcome %s)" % outcome, ci, ri))
                continue
            claimed.append(target)
            plan.append((run, request, HandoffRecord(source, target, task.task_id, ci, ri, kind, HandoffCode.HANDOFF_EXECUTED, "valid", executed=False)))
        invalid_records = [r for r in records if r.code is not HandoffCode.HANDOFF_EXECUTED]
        if invalid_records:
            for _run, _request, record in plan:                      # the valid ones are recorded too, unexecuted
                records.append(record)
            records.sort(key=lambda r: (r.request_clause if r.request_clause is not None else -1, r.source, r.target))
            first = invalid_records[0]
            return done(first.code, "%s -> %s: %s" % (first.source, first.target, first.detail)), None, None

        # ---- every first-round result the request still depends on must be supported, or no target runs ---------------
        handed: Dict[str, List[int]] = {}
        for run, _request, record in plan:
            handed.setdefault(run.task.task_id, []).append(int(record.report_index))
        for run in runs:
            remaining = [i for i in range(len(_focus(run.task))) if i not in handed.get(run.task.task_id, [])]
            if not remaining:
                continue                                              # every clause of this run was handed off: its own result no longer matters
            settled = _without_requests(run.findings, [request for r, request, _rec in plan if r is run])
            code, detail = run_code(settled)
            if code is not None:
                for _r, _q, record in plan:
                    records.append(record)
                return done(HandoffCode.HANDOFF_REMAINDER_UNSUPPORTED,
                            "%s keeps clause(s) the request depends on and did not support them: %s" % (run.specialist_id.value, detail)), None, None

        # ---- execution: each target once, depth one -------------------------------------------------------------------
        executed_runs: List[SpecialistRun] = []
        new_tasks: List[SpecialistTask] = []
        for run, request, record in plan:
            task = request.task
            t = time.perf_counter()
            calls += 1
            try:
                specialist = registry.resolve(request.requested_specialist)
                findings = check_findings(specialist, task, specialist.run(task))
            except Exception as exc:
                records.append(HandoffRecord(record.source, record.target, record.task_id, record.request_clause, record.report_index, record.kind,
                                             HandoffCode.HANDOFF_EXCEPTION, "target raised: %s" % (redact_error(exc).split(":", 1)[0].strip() or type(exc).__name__),
                                             executed=True, ms=(time.perf_counter() - t) * 1000.0))
                return done(HandoffCode.HANDOFF_EXCEPTION, "%s -> %s: the target raised" % (record.source, record.target)), None, None
            visited.append(record.target)
            second_hop = tuple(h.requested_specialist.value for h in findings.handoff_requests)
            code = HandoffCode.HANDOFF_EXECUTED
            detail = "executed once"
            if findings.llm_used:
                code, detail = HandoffCode.HANDOFF_UNSAFE_RESULT, "the target reported a model call"
            elif second_hop:
                code, detail = HandoffCode.HANDOFF_SECOND_HOP_BLOCKED, "the target asked for a handoff to %s" % ", ".join(second_hop)
            records.append(HandoffRecord(record.source, record.target, record.task_id, record.request_clause, record.report_index, record.kind, code, detail,
                                         executed=True, status=findings.status.value, findings=len(findings.findings), second_hop=second_hop,
                                         ms=(time.perf_counter() - t) * 1000.0))
            if code is not HandoffCode.HANDOFF_EXECUTED:
                return done(code, "%s -> %s: %s" % (record.source, record.target, detail)), None, None
            stamped = SpecialistTask(task_id=task.task_id, question=task.question, specialist_id=task.specialist_id, domain=task.domain, intent=task.intent,
                                     level=task.level, system=task.system, source_scope=list(task.source_scope), context=dict(task.context),
                                     requested_by=task.requested_by,
                                     metadata=dict(task.metadata, handoff={"source": record.source, "target": record.target, "depth": MAX_DEPTH,
                                                                           "request_clause": record.request_clause, "report_index": record.report_index,
                                                                           "validated": True}))
            new_tasks.append(stamped)
            executed_runs.append(SpecialistRun(stamped, request.requested_specialist, STATE_EXECUTED, findings, "handoff round, depth %d" % MAX_DEPTH))

        # ---- settlement for the aggregator -----------------------------------------------------------------------------
        round_ = done(HandoffCode.HANDOFF_EXECUTED, "%d handoff(s) executed at depth %d" % (len(executed_runs), MAX_DEPTH))
        executed_requests = [request for _r, request, _rec in plan]
        kept_tasks: List[SpecialistTask] = []
        settled_runs: List[SpecialistRun] = []
        for run in dispatched.runs:
            task = tasks_by_id.get(run.task.task_id, run.task)
            if run.task.task_id in handed:
                indexes = sorted(set(handed[run.task.task_id]))
                remaining = [i for i in range(len(_focus(task))) if i not in indexes]
                if remaining:
                    kept_tasks.append(SpecialistTask(task_id=task.task_id, question=task.question, specialist_id=task.specialist_id, domain=task.domain,
                                                     intent=task.intent, level=task.level, system=task.system, source_scope=list(task.source_scope),
                                                     context=dict(task.context), requested_by=task.requested_by,
                                                     metadata=dict(task.metadata, handed_off_clauses=indexes)))
                findings = _without_requests(run.findings, executed_requests) if run.findings is not None else None
                settled_runs.append(SpecialistRun(run.task, run.specialist_id, run.state, findings, run.reason))
            else:
                if task.task_id in tasks_by_id:
                    kept_tasks.append(task)
                settled_runs.append(run)
        kept_tasks.extend(new_tasks)
        settled_runs.extend(executed_runs)
        specialists = list(dict.fromkeys(t.specialist_id.value for t in kept_tasks))
        md = dict(decision.metadata or {})
        md["handoff_round"] = round_.to_dict()
        decision2 = CoordinatorDecision(domains=list(decision.domains), intents=list(decision.intents), level=decision.level, systems=list(decision.systems),
                                        selected_specialists=[SpecialistId.parse(s) for s in specialists], subtasks=kept_tasks,
                                        requires_synthesis=len(kept_tasks) >= 2, confidence=decision.confidence, reason=decision.reason,
                                        used_llm=decision.used_llm, metadata=md)
        dispatched2 = DispatchResult(question=question, decision=decision2, runs=settled_runs)
        return round_, decision2, dispatched2
    except Exception as exc:                                        # a defect here must never become an answer
        category = redact_error(exc).split(":", 1)[0].strip() or type(exc).__name__
        return done(HandoffCode.HANDOFF_EXCEPTION, "handoff round raised: %s" % category), None, None


def validated_handoff_task(task: SpecialistTask, decision: CoordinatorDecision) -> bool:
    """Whether ``task`` is a handoff task this module settled into ``decision``: stamped as validated at depth one and
    listed among the executed handoffs of the decision's recorded round."""
    stamp = task.metadata.get("handoff") if isinstance(task.metadata, dict) else None
    if not isinstance(stamp, dict) or stamp.get("validated") is not True or stamp.get("depth") != MAX_DEPTH:
        return False
    if task.specialist_id is None or task.specialist_id.value != stamp.get("target") or task.requested_by != stamp.get("source"):
        return False
    round_ = (decision.metadata or {}).get("handoff_round") if isinstance(decision.metadata, dict) else None
    if not isinstance(round_, dict) or round_.get("code") != HandoffCode.HANDOFF_EXECUTED.value:
        return False
    return any(r.get("task_id") == task.task_id and r.get("executed") and r.get("code") == HandoffCode.HANDOFF_EXECUTED.value
               for r in round_.get("records", []) if isinstance(r, dict))


__all__ = ["APPROVED_TARGETS", "DECLINED_OUTCOMES", "HandoffCode", "HandoffRecord", "HandoffRound", "MAX_DEPTH", "PHASE", "VERSION", "run_handoffs",
           "validated_handoff_task"]
