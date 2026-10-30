"""
Deterministic routing refinement of a Coordinator decision (runtime integration, phase I-6).

DECISION OWNED
    Which ONE specialist owns each request clause when the Coordinator's decision leaves a clause with several
    owners, and the clause-scoped tasks of a single specialist that received a multipart question as one clause.
    It runs after the Coordinator and before any specialist executes, reads only the decision and the request's
    own structure, calls no model and never retrieves. It never invents an owner: a clause the rules below cannot
    settle stays ambiguous and the attempt falls back before any specialist runs.

OWNERSHIP POLICY (derived from the specialists' own boundaries and the Coordinator's rules; see docs/STATUS.md)
    1. A contact or location clause (telephone, fax, e-mail, extension, website, where is / located) belongs to
       Institutional Navigation alone. The Coordinator already sends a bare channel lookup there; Teaching, Research
       and Faculty Services & HR hand such a clause to Institutional Navigation when it reaches them ("a pure place
       or contact ask is Institutional Navigation's alone"); Institutional Navigation keeps every contact or location
       ask "whoever's unit it names".
    2. An authority or responsibility clause about a subject ("who approves X", "who handles X", "who is responsible
       for X") belongs to the ONE subject specialist whose cues claim it; Institutional Navigation is dropped. This is
       Institutional Navigation's own boundary: it answers no approval or authority ask whose subject the shared
       research or HR cues claim and hands such a clause to that owner ("misrouted clause belongs to research").
    3. A "which office / unit / department ... handles X" clause with a subject claim stays ambiguous: Institutional
       Navigation owns the office-selection ask kind while the subject specialist owns the subject, and the two hand
       the clause to each other. No specialist runs for it.
    4. Two subject specialists claiming one clause stay ambiguous. Nothing is chosen by score, by registration order
       or at random.
    5. A single specialist that received a multipart question as one clause gets one focus clause per request
       clause, assigned by the same cue rule the Coordinator applies to several specialists: a clause with the
       specialist's cue, or with no cue at all (it continues the previous clause), is its own clause; a clause whose
       cues belong only to other domains has no owner and the attempt falls back. The clause texts are the
       Coordinator's own split, unchanged; nothing is rewritten.

RESULT
    :func:`refine` returns the decision to execute (the ORIGINAL object when nothing changed) and an immutable
    :class:`RoutingResult` with a machine-testable :class:`RoutingCode`, the owners of every clause before and after,
    the specialists before and after, the dropped tasks and the ambiguous clauses. ``to_dict`` is JSON-serialisable,
    deterministic and free of question text. Any exception becomes ``ROUTE_ERROR`` with the original decision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .agents.coordinator import REQUESTED_BY, cue_scores
from .agents.specialists.base import redact_error
from .agents.specialists.contracts import CoordinatorDecision, SpecialistId, SpecialistTask
from .plan_f_aggregation import request_clauses
from .plan_f_verification import RequestKind, read_request

PHASE = "I-6"
VERSION = "plan-f-routing-1"

#: Request kinds Institutional Navigation owns whoever's unit the clause names (policy rule 1).
NAVIGATION_KINDS = frozenset({RequestKind.CONTACT, RequestKind.LOCATION})
#: Request kinds a subject specialist owns against Institutional Navigation (policy rule 2).
SUBJECT_PARTY_KINDS = frozenset({RequestKind.AUTHORITY, RequestKind.RESPONSIBILITY})
#: Body nouns of an office-selection ask ("which office handles"), which stays ambiguous with a subject claim (rule 3).
OFFICE_BODY_NOUNS = frozenset({"office", "unit", "department", "section", "division", "centre", "center", "directorate", "deanship", "body"})
_INSTITUTIONAL = SpecialistId.INSTITUTIONAL.value


class RoutingCode(str, Enum):
    ROUTE_UNCHANGED = "ROUTE_UNCHANGED"                            # every clause already has one owner; the decision is used as given
    ROUTE_CLAUSES_SCOPED = "ROUTE_CLAUSES_SCOPED"                  # a single specialist's whole-question task became one clause per request clause
    ROUTE_REFINED_BY_REQUEST_KIND = "ROUTE_REFINED_BY_REQUEST_KIND"  # a shared clause settled by its request kind (rules 1 and 2)
    ROUTE_AMBIGUOUS = "ROUTE_AMBIGUOUS"                            # a clause still has two or more owners: no specialist runs
    ROUTE_NO_OWNER = "ROUTE_NO_OWNER"                              # a clause has no owner among the selected specialists
    ROUTE_ERROR = "ROUTE_ERROR"                                    # the refinement raised; the exception category only


@dataclass(frozen=True)
class RoutingResult:
    refined: bool
    reason: RoutingCode
    detail: str
    initial_specialists: Tuple[str, ...]
    final_specialists: Tuple[str, ...]
    owners_before: Tuple[Tuple[str, ...], ...]                     # per request clause, in request order
    owners_after: Tuple[Tuple[str, ...], ...]
    rules_applied: Tuple[Tuple[int, str], ...]                     # (clause index, rule)
    dropped_tasks: Tuple[str, ...]
    ambiguous_clauses: Tuple[int, ...]
    unowned_clauses: Tuple[int, ...]
    scoped_tasks: Tuple[str, ...]
    ms: float = 0.0
    version: str = VERSION

    @property
    def executable(self) -> bool:
        return self.reason in (RoutingCode.ROUTE_UNCHANGED, RoutingCode.ROUTE_CLAUSES_SCOPED, RoutingCode.ROUTE_REFINED_BY_REQUEST_KIND)

    def to_dict(self) -> Dict[str, Any]:
        return {"refined": self.refined, "reason": self.reason.value, "detail": self.detail[:200], "initial_specialists": list(self.initial_specialists),
                "final_specialists": list(self.final_specialists), "owners_before": [list(o) for o in self.owners_before],
                "owners_after": [list(o) for o in self.owners_after], "rules_applied": [list(r) for r in self.rules_applied],
                "dropped_tasks": list(self.dropped_tasks), "ambiguous_clauses": list(self.ambiguous_clauses),
                "unowned_clauses": list(self.unowned_clauses), "scoped_tasks": list(self.scoped_tasks), "version": self.version}


# ---------------------------------------------------------------------------
# Reading the decision
# ---------------------------------------------------------------------------
def _focus(task: SpecialistTask) -> List[str]:
    focus = (task.context or {}).get("focus_clauses") if isinstance(task.context, dict) else None
    return [c for c in (focus if isinstance(focus, (list, tuple)) else []) if isinstance(c, str) and c.strip()]


def _is_whole_question(task: SpecialistTask, focus: Sequence[str]) -> bool:
    return len(focus) == 1 and focus[0].strip() == (task.question or "").strip()


def assign_by_cue(clauses: Sequence[str], domains: Sequence[str]) -> Tuple[Dict[str, List[str]], List[int]]:
    """The Coordinator's clause assignment rule for ``domains``: a clause goes to every domain whose cue fires in it; a
    clause with no cue at all continues the previous clause's owners (a first clause with no cue goes to every
    domain); a clause whose cues belong only to other domains is unowned. Returns (clauses per domain, unowned indexes)."""
    assigned: Dict[str, List[str]] = {d: [] for d in domains}
    unowned: List[int] = []
    previous: List[str] = list(domains)
    for index, clause in enumerate(clauses):
        scores = cue_scores(clause)
        owners = [d for d in domains if scores.get(d, 0.0) > 0.0]
        if not owners and scores:
            unowned.append(index)
            continue
        if not owners:
            owners = previous
        for d in owners:
            assigned[d].append(clause)
        previous = owners
    return assigned, unowned


def _owners_of(clauses: Sequence[str], tasks: Sequence[Tuple[SpecialistTask, List[str]]]) -> List[List[str]]:
    """Per request clause, the specialists whose focus names it (a whole-question focus names every clause)."""
    owners: List[List[str]] = [[] for _ in clauses]
    for task, focus in tasks:
        sid = task.specialist_id.value if task.specialist_id is not None else "?"
        if _is_whole_question(task, focus):
            for ci in range(len(clauses)):
                owners[ci].append(sid)
            continue
        for text in focus:
            for ci, clause in enumerate(clauses):
                if clause == text and sid not in owners[ci]:
                    owners[ci].append(sid)
    return owners


def settle_owner(clause: str, owners: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """(the one owner, the rule) for a clause with several owners, or (None, why) when the policy keeps it ambiguous."""
    distinct = list(dict.fromkeys(owners))
    if len(distinct) <= 1:
        return (distinct[0] if distinct else None), None
    shape = read_request(clause)
    subjects = [o for o in distinct if o != _INSTITUTIONAL]
    if shape.kind in NAVIGATION_KINDS:
        if _INSTITUTIONAL in distinct:
            return _INSTITUTIONAL, "navigation ask: Institutional Navigation owns a contact or location clause"
        return None, "contact or location clause without Institutional Navigation among its owners"
    if shape.kind in SUBJECT_PARTY_KINDS and _INSTITUTIONAL in distinct:
        if shape.body_noun in OFFICE_BODY_NOUNS:
            return None, "office-selection ask with a subject claim: Institutional Navigation and %s hand it to each other" % ", ".join(subjects)
        if len(subjects) == 1:
            return subjects[0], "party ask about a subject: the subject specialist owns it; Institutional Navigation answers no approval or authority ask"
        return None, "party ask claimed by several subject specialists"
    return None, "no ownership rule for a %s clause shared by %s" % (shape.kind.value, ", ".join(distinct))


# ---------------------------------------------------------------------------
# Rebuilding the decision
# ---------------------------------------------------------------------------
def _complexity(md: Mapping[str, Any], specialists: Sequence[str]) -> str:
    current = md.get("complexity")
    if current in ("greeting", "journey"):
        return str(current)
    if not specialists:
        return "unknown"
    if len(specialists) >= 2:
        return "cross_domain"
    if bool((md.get("router") or {}).get("is_multipart")):
        return "multipart"
    return "simple"


def _rebuild(decision: CoordinatorDecision, kept: Sequence[Tuple[SpecialistTask, List[str]]], routing: Dict[str, Any]) -> CoordinatorDecision:
    tasks: List[SpecialistTask] = []
    for task, focus in kept:
        scoped = not _is_whole_question(task, focus)
        tasks.append(SpecialistTask(task_id=task.task_id, question=task.question, specialist_id=task.specialist_id, domain=task.domain,
                                    intent=task.intent, level=task.level, system=task.system, source_scope=list(task.source_scope),
                                    context=dict(task.context, focus_clauses=list(focus)), requested_by=task.requested_by,
                                    metadata=dict(task.metadata, scoped=scoped)))
    specialists = list(dict.fromkeys(t.specialist_id.value for t in tasks))
    md = dict(decision.metadata or {})
    md["routing"] = routing
    md["complexity"] = _complexity(md, specialists)
    multipart = bool((md.get("router") or {}).get("is_multipart"))
    return CoordinatorDecision(domains=list(decision.domains), intents=list(decision.intents), level=decision.level, systems=list(decision.systems),
                               selected_specialists=[SpecialistId.parse(s) for s in specialists], subtasks=tasks,
                               requires_synthesis=len(specialists) >= 2 or (len(specialists) == 1 and multipart),
                               confidence=decision.confidence, reason=decision.reason, used_llm=decision.used_llm, metadata=md)


# ---------------------------------------------------------------------------
# The refinement
# ---------------------------------------------------------------------------
def refine(decision: CoordinatorDecision) -> Tuple[CoordinatorDecision, RoutingResult]:
    """Settle one owner per request clause. Deterministic; no model; never raises."""
    started = time.perf_counter()
    initial = tuple(s.value for s in getattr(decision, "selected_specialists", []) or [])

    def result(reason: RoutingCode, detail: str, *, refined: bool = False, final: Sequence[str] = initial, before: Sequence[Sequence[str]] = (),
               after: Sequence[Sequence[str]] = (), rules: Sequence[Tuple[int, str]] = (), dropped: Sequence[str] = (),
               ambiguous: Sequence[int] = (), unowned: Sequence[int] = (), scoped: Sequence[str] = ()) -> RoutingResult:
        return RoutingResult(refined=refined, reason=reason, detail=detail[:200], initial_specialists=initial, final_specialists=tuple(final),
                             owners_before=tuple(tuple(o) for o in before), owners_after=tuple(tuple(o) for o in after),
                             rules_applied=tuple((int(i), r) for i, r in rules), dropped_tasks=tuple(dropped), ambiguous_clauses=tuple(ambiguous),
                             unowned_clauses=tuple(unowned), scoped_tasks=tuple(scoped), ms=(time.perf_counter() - started) * 1000.0)

    try:
        md = decision.metadata if isinstance(decision.metadata, dict) else {}
        clauses = request_clauses(decision)
        tasks = [(task, _focus(task)) for task in decision.subtasks]
        if not clauses or not tasks or md.get("complexity") == "greeting" or any(t.requested_by != REQUESTED_BY for t, _f in tasks):
            return decision, result(RoutingCode.ROUTE_UNCHANGED, "no request clauses or no Coordinator task to refine")
        before = _owners_of(clauses, tasks)

        # ---- rule 5: a whole-question task of a multipart request becomes one clause per request clause --------------
        scoped: List[str] = []
        unowned: List[int] = []
        if len(clauses) >= 2 and any(_is_whole_question(t, f) for t, f in tasks):
            whole = [t.specialist_id.value for t, f in tasks if _is_whole_question(t, f)]
            assigned, missing = assign_by_cue(clauses, whole)
            unowned = [ci for ci in missing if not any(clauses[ci] in f for t, f in tasks if not _is_whole_question(t, f))]
            rescoped: List[Tuple[SpecialistTask, List[str]]] = []
            for task, focus in tasks:
                if _is_whole_question(task, focus):
                    rescoped.append((task, list(assigned.get(task.specialist_id.value, []))))
                    scoped.append(task.task_id)
                else:
                    rescoped.append((task, focus))
            tasks = rescoped
        owners = _owners_of(clauses, tasks)

        # ---- rules 1 to 4: one owner per shared clause ----------------------------------------------------------------
        rules: List[Tuple[int, str]] = []
        ambiguous: List[int] = []
        settled: List[Optional[str]] = []
        for ci, clause in enumerate(clauses):
            if ci in unowned or not owners[ci]:
                settled.append(None)
                if ci not in unowned:
                    unowned.append(ci)
                continue
            owner, rule = settle_owner(clause, owners[ci])
            if owner is None:
                ambiguous.append(ci)
                settled.append(None)
                if rule:
                    rules.append((ci, "ambiguous: " + rule))
                continue
            if len(dict.fromkeys(owners[ci])) > 1 and rule:
                rules.append((ci, rule))
            settled.append(owner)
        after = [[o] if o is not None else list(dict.fromkeys(owners[ci])) for ci, o in enumerate(settled)]
        if unowned:
            return decision, result(RoutingCode.ROUTE_NO_OWNER, "request clause %d has no owner among the selected specialists" % unowned[0],
                                    before=before, after=after, rules=rules, unowned=sorted(set(unowned)), ambiguous=ambiguous, scoped=scoped)
        if ambiguous:
            ci = ambiguous[0]
            return decision, result(RoutingCode.ROUTE_AMBIGUOUS, "request clause %d is claimed by %s" % (ci, ", ".join(dict.fromkeys(owners[ci]))),
                                    before=before, after=after, rules=rules, ambiguous=ambiguous, scoped=scoped)

        # ---- rebuild: each task keeps the clauses it owns; a task left without a clause is dropped ---------------------
        kept: List[Tuple[SpecialistTask, List[str]]] = []
        dropped: List[str] = []
        changed = bool(scoped)
        for task, focus in tasks:
            sid = task.specialist_id.value
            mine = [clauses[ci] for ci, o in enumerate(settled) if o == sid]
            if _is_whole_question(task, focus):
                focus_after = focus if mine else []
            else:
                focus_after = [c for c in focus if c in mine]
            if focus_after != focus:
                changed = True
            if not focus_after:
                dropped.append(task.task_id)
                continue
            kept.append((task, focus_after))
        final = tuple(dict.fromkeys(t.specialist_id.value for t, _f in kept))
        if not kept:
            return decision, result(RoutingCode.ROUTE_NO_OWNER, "no task left after refinement", before=before, after=after, rules=rules, scoped=scoped)
        if not changed:
            return decision, result(RoutingCode.ROUTE_UNCHANGED, "every request clause already has one owner", before=before, after=after)
        reason = RoutingCode.ROUTE_REFINED_BY_REQUEST_KIND if rules else RoutingCode.ROUTE_CLAUSES_SCOPED
        record = result(reason, "owners settled for %d clause(s); %d task(s) dropped; %d task(s) scoped" % (len(clauses), len(dropped), len(scoped)),
                        refined=True, final=final, before=before, after=after, rules=rules, dropped=dropped, scoped=scoped)
        return _rebuild(decision, kept, record.to_dict()), record
    except Exception as exc:                                        # a defect here must never route a question
        category = redact_error(exc).split(":", 1)[0].strip() or type(exc).__name__
        return decision, result(RoutingCode.ROUTE_ERROR, "routing raised: %s" % category)


__all__ = ["NAVIGATION_KINDS", "OFFICE_BODY_NOUNS", "PHASE", "RoutingCode", "RoutingResult", "SUBJECT_PARTY_KINDS", "VERSION", "assign_by_cue",
           "refine", "settle_owner"]
