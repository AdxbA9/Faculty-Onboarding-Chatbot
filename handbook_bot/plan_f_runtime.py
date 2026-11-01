"""
Plan F runtime adapter (runtime integration, phase I-1).

DECISION OWNED
    None. This module is a thin, deterministic seam between the Plan F path
    (``coordinator.coordinate`` -> ``dispatch.dispatch`` -> the four frozen
    specialists) and the legacy result contract (``QAResult``, checked by the
    unchanged ``agents.verifier.verify``). It never routes, never retrieves,
    never answers from anything but a specialist's own supported findings,
    never calls a language model and never executes a handoff.

WIRING (phases I-2, I-4 and I-5)
    The one approved seam is ``ui/pipeline.py`` (phase I-2): it builds a
    :class:`PlanFRuntime` lazily through :class:`RuntimeCache` and calls
    ``attempt`` only in SHADOW or ACTIVE mode; both switches default to off
    and LEGACY mode never imports this module. The orchestrator, the Router,
    retrieval, Synthesis and the legacy Verifier are unchanged. Since phase
    I-4 a finding passes the findings-aware verification layer
    (``plan_f_verification.verify_findings``) after the I-1 guards and
    before rendering; its rejection is ``PLAN_F_REJECTED_BY_FINDINGS_VERIFIER``.
    Since phase I-5 a decision with several request clauses or several
    specialists takes the aggregation path (``plan_f_aggregation.aggregate``):
    every task of the decision is dispatched once, every request clause must
    be covered by findings that pass the findings-aware verifier on their own
    clause, and the verbatim quotes are rendered in request order; its
    rejection is ``PLAN_F_REJECTED_BY_AGGREGATOR``. A single-clause decision
    keeps the pass-through path of phases I-1 to I-4 unchanged, except that
    the findings-aware layer may now accept the provision slice of a quote it
    would otherwise reject as excess evidence (``verify_findings_sliced``).
    Since phase I-6 the decision passes a deterministic routing refinement
    before any specialist runs (``plan_f_routing.refine``: one owner per
    request clause, or ``PLAN_F_REJECTED_BY_ROUTING`` with no specialist
    call), and a specialist's handoff request is executed ONCE, at depth one,
    by ``plan_f_handoff.run_handoffs`` (visited-specialist protection, no
    second hop, no query invention), after which the settled execution goes
    through the aggregator like any other; a round that cannot run or
    complete is ``PLAN_F_REJECTED_BY_HANDOFF``.

WHAT IT DOES
    1. ``build_specialist_registry(kb)`` builds the four specialists from a
       real ``KnowledgeBase`` with their own ``from_knowledge_base`` builders.
       It either returns a complete registry or no registry at all, with every
       builder failure recorded in redacted form: a partial registry is never
       returned. There is no module-level registry or cache.
    2. ``PlanFRuntime.attempt(question)`` runs the Coordinator and the
       dispatcher on an explicitly supplied registry and corpus and returns an
       :class:`AdapterResult` whose ``outcome`` says exactly why Plan F did or
       did not answer. Only the narrowest safe case is accepted in I-1:
       exactly one selected specialist and one task with one focus clause, no
       Synthesis requirement, no multipart question, no uncovered clause, a
       ``supported`` result with no handoff and no ``missing`` entry, and a
       clause report that shows a specific, fully covered ask (Teaching,
       Research, Faculty Services & HR) or an evidence-tier answer bound to a
       lexicon office or to the University (Institutional Navigation).
    3. Two guards run on every finding before anything is rendered. The quote
       guard requires ``evidence_quote`` to be an exact substring of the chunk
       named by ``metadata["chunk_id"]`` on the cited page of the cited source,
       with no normalisation and no fuzzy rescue. The subject guard requires
       the asked subject to be bound to the finding: every named entity of the
       clause must appear in the quote, the specialist's own requested details
       must all be covered by the finding, and the clause's content words must
       be covered by the quote under the unchanged Verifier's rules; an
       Institutional finding must name the office (or the University) that the
       clause asks about, with a complete directory row for a contact answer.
       The primary finding must pass; a secondary finding that fails the
       subject guard is dropped and traced; any finding that fails the quote
       guard rejects the whole result.
    4. (Phase I-4) The findings-aware verification layer then reads the
       structured result against the clause: request kind, requested detail
       in the answering unit, entity and category-modifier binding, stated
       conditions, the asked relation, provision boundaries and completeness.
       Only a verified result is rendered; a dropped secondary finding is
       traced, never shown.
    5. The answer is rendered extractively: the accepted quotes, verbatim, one
       per paragraph, nothing added. Pages come only from accepted findings.
       The rendered answer is shaped as a ``SynthesisResult`` and checked by
       the unchanged legacy Verifier against the accepted evidence; a
       rejection means fallback.
    6. On acceptance a complete, unchanged-shape ``QAResult`` is built
       (``used_llm=False``, ``llm_calls=0``, ``retried=False``, trace filled).
       Every other outcome asks the caller to fall back to the legacy path.

FAILURE POLICY
    Nothing here raises to the caller. Exceptions from the Coordinator, the
    dispatcher, the guards or the renderer become ``PLAN_F_ERROR_REDACTED``
    with the message passed through ``base.redact_error`` (no secrets, no
    paths). Every decision is recorded as a ``TraceEntry``.
"""
from __future__ import annotations

import dataclasses
import re
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .agents.coordinator import REQUESTED_BY, coordinate
from .agents.specialists import navigation_cues as _nav
from .agents.specialists.base import Specialist, redact_error
from .agents.specialists.contracts import CoordinatorDecision, Finding, FindingStatus, SpecialistFindings, SpecialistId
from .agents.specialists.dispatch import STATE_EXECUTED, DispatchResult, dispatch
from .agents.specialists.faculty_services import FacultyServicesSpecialist
from .agents.specialists.institutional import InstitutionalNavigationSpecialist
from .agents.specialists.registry import SpecialistRegistry
from .agents.specialists.research import ResearchInnovationSpecialist
from .agents.specialists.teaching import TeachingLearningSpecialist
from .agents.types import ALL_ROUTES, EvidenceResult, SubQuestion, SynthesisResult, TraceEntry, VerifyResult
from .agents.verifier import verify
from .config import EVIDENCE_PREVIEW_CHARS
from .qa import QAResult, answer_question
from . import guide_grounding as _guides                         # post-freeze PF-3C (D-111): the one guide evidence path
from .agents.specialists.guide_evidence import GuideCapableSpecialist, GuideEvidence

#: Integration phase this module implements.
PHASE = "I-1"
#: ``SynthesisResult.prompt_version`` of a rendered Plan F answer (no prompt is involved).
PROMPT_VERSION = "plan-f-extractive-1"
#: ``prompt_version`` of the subject check that reuses the Verifier's coverage rules.
SUBJECT_CHECK_VERSION = "plan-f-subject-check-1"

#: The specialists a registry must hold before Plan F may answer anything.
REQUIRED_SPECIALISTS: Tuple[SpecialistId, ...] = (
    SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES, SpecialistId.INSTITUTIONAL,
)

#: The frozen specialists' own builders. Read-only: tests inject their own mapping instead of editing this one.
DEFAULT_BUILDERS: Mapping[SpecialistId, Callable[[Any], Specialist]] = MappingProxyType({
    SpecialistId.TEACHING: TeachingLearningSpecialist.from_knowledge_base,
    SpecialistId.RESEARCH: ResearchInnovationSpecialist.from_knowledge_base,
    SpecialistId.FACULTY_SERVICES: FacultyServicesSpecialist.from_knowledge_base,
    SpecialistId.INSTITUTIONAL: InstitutionalNavigationSpecialist.from_knowledge_base,
})

#: Clause ask shape and relation values the retrieval specialists report (docs/AGENT_CONTRACTS.md).
ASK_SPECIFIC_RECOGNISED = "specific_recognised"
RELATION_STRONG = "strong"
#: Institutional clause outcome and evidence tiers that may be rendered.
INSTITUTIONAL_OUTCOME_READ = "read"
INSTITUTIONAL_TIERS = (_nav.E1, _nav.E2, _nav.STATED_FACT)
#: Institutional statement bindings whose quote names its own subject.
INSTITUTIONAL_BINDINGS = ("named subject", "list item under its lead-in")
#: Institutional ask kind whose office is the answer rather than the asked subject.
INSTITUTIONAL_RESPONSIBLE_OFFICE = "responsible_office"
#: How a Plan F answer was produced (``AdapterResult.path``; phase I-5): one specialist's findings for one request
#: clause, passed through the guards and the findings-aware verifier, or the aggregate of a multi-clause execution.
PATH_PASSTHROUGH = "single_finding_passthrough"
PATH_AGGREGATE = "multi_finding_aggregate"


class PlanFOutcome(str, Enum):
    """Why the adapter did or did not produce a Plan F answer."""

    ACCEPTED = "PLAN_F_ACCEPTED"
    NOT_APPLICABLE = "PLAN_F_NOT_APPLICABLE"                    # greeting, empty or no specialist selected
    REGISTRY_UNAVAILABLE = "PLAN_F_REGISTRY_UNAVAILABLE"        # no complete registry
    UNSAFE_RESULT = "PLAN_F_UNSAFE_RESULT"                      # decision or specialist result outside the I-1 safe case
    REJECTED_BY_QUOTE_GUARD = "PLAN_F_REJECTED_BY_QUOTE_GUARD"
    REJECTED_BY_SUBJECT_GUARD = "PLAN_F_REJECTED_BY_SUBJECT_GUARD"
    REJECTED_BY_SOURCE_CONFLICT = "PLAN_F_REJECTED_BY_SOURCE_CONFLICT"      # post-freeze PF-2: the finding rests on a registered source conflict
    REJECTED_BY_SOURCE_AUTHORITY = "PLAN_F_REJECTED_BY_SOURCE_AUTHORITY"    # post-freeze PF-3B: a guide outside its system, a policy ask, or a partial procedure
    REJECTED_BY_FINDINGS_VERIFIER = "PLAN_F_REJECTED_BY_FINDINGS_VERIFIER"  # phase I-4: the structured finding is not a suitable direct answer
    REJECTED_BY_AGGREGATOR = "PLAN_F_REJECTED_BY_AGGREGATOR"          # phase I-5: the findings of a multi-clause execution do not cover the request
    REJECTED_BY_ROUTING = "PLAN_F_REJECTED_BY_ROUTING"                # phase I-6: a request clause has no single owner; no specialist ran
    REJECTED_BY_HANDOFF = "PLAN_F_REJECTED_BY_HANDOFF"                # phase I-6: a handoff request could not be executed within the bounds
    REJECTED_BY_VERIFIER = "PLAN_F_REJECTED_BY_VERIFIER"
    REQUIRES_FALLBACK = "PLAN_F_REQUIRES_FALLBACK"              # Plan F ran but left the task pending or nothing renderable
    ERROR_REDACTED = "PLAN_F_ERROR_REDACTED"


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegistryBuild:
    """What building the registry produced: a complete registry, or none with the redacted failures."""

    registry: Optional[SpecialistRegistry]
    errors: Dict[str, str]                       # specialist id -> redacted reason
    ms: float = 0.0
    scope_extension: Dict[str, Any] = field(default_factory=dict)   # post-freeze PF-2: the source-scope seam's report

    @property
    def ok(self) -> bool:
        return self.registry is not None and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "specialists": [s.value for s in self.registry.ids()] if self.registry is not None else [],
                "errors": dict(self.errors), "ms": self.ms, "scope_extension": dict(self.scope_extension)}


def _collision_report(kb: Any, registry: SpecialistRegistry, errors: Dict[str, str]) -> Dict[str, Any]:
    """Post-freeze PF-2/PF-3B check, run once the four specialists are built: refuse the registry when a non-handbook
    document's page numbers fall inside a specialist's handbook page ranges (frozen indexes such as the Teaching
    calendar rows and the Institutional directory are keyed by handbook page number). A one-document knowledge base
    (no ``sources``) is left exactly as it was. The report is JSON-serialisable and names no path."""
    records = list(getattr(kb, "sources", None) or ())
    if not records:
        return {}
    try:
        from .source_scopes import page_collisions
        counts = {sid: len(pages) for sid, pages in (getattr(kb, "source_pages", None) or {}).items()}
        collisions = page_collisions(records, counts, registry)
        if collisions:
            errors["scope_extension"] = ("document pages overlap a specialist scope; Plan F is not built: "
                                         + "; ".join(collisions))[:400]
        return {"collisions": collisions}
    except Exception as exc:                                      # the check never raises past the registry factory
        errors["scope_extension"] = redact_error(exc)
        return {"collisions": []}


def build_specialist_registry(kb: Any, *, builders: Optional[Mapping[SpecialistId, Callable[[Any], Specialist]]] = None) -> RegistryBuild:
    """Build the four specialists from ``kb`` with their own ``from_knowledge_base`` builders.

    Either every required specialist is built and registered, or no registry is returned and ``errors`` names each
    specialist that could not be built, with its reason redacted. No model is called; nothing is cached. The default
    builders give every specialist the I-9 handbook-only source boundary; ``guide_builders`` gives the PF-3B one.
    """
    table = DEFAULT_BUILDERS if builders is None else builders
    started = time.perf_counter()
    registry = SpecialistRegistry()
    errors: Dict[str, str] = {}
    for sid in REQUIRED_SPECIALISTS:
        builder = table.get(sid)
        if builder is None:
            errors[sid.value] = "no builder registered for %s" % sid.value
            continue
        try:
            specialist = builder(kb)
            registered = registry.register(specialist)
            if registered is not sid:
                errors[sid.value] = "builder for %s produced the %s specialist" % (sid.value, registered.value)
        except Exception as exc:                                  # a failing builder never raises past this seam
            errors[sid.value] = redact_error(exc)
    scope_report: Dict[str, Any] = {}
    if not errors:
        scope_report = _collision_report(kb, registry, errors)
    ms = (time.perf_counter() - started) * 1000.0
    if errors:
        return RegistryBuild(None, errors, ms, scope_report)
    return RegistryBuild(registry, {}, ms, scope_report)


def guide_builders(policy: Mapping[str, Any], evidence: Optional[GuideEvidence] = None,
                   base_registry: Optional[SpecialistRegistry] = None) -> Dict[SpecialistId, Callable[[Any], Specialist]]:
    """Builders of the guide-enabled registry. Post-freeze PF-3C (D-111): a specialist whose D-110 approved set holds a
    guide (``source_scopes.source_policy``) is the frozen I-9 specialist, with its handbook-only extraction untouched,
    behind the D-111 guide path (``GuideCapableSpecialist``) for exactly its approved guides. Guide evidence reaches Plan F
    through that path only. A specialist approved for the handbook only (Faculty Services & HR) is the I-9 specialist.
    ``base_registry`` lends its already built I-9 specialists (nothing is mutated); ``evidence`` is shared by all."""
    table: Dict[SpecialistId, Callable[[Any], Specialist]] = dict(DEFAULT_BUILDERS)
    if base_registry is not None:
        for sid in REQUIRED_SPECIALISTS:
            table[sid] = (lambda kb, sid=sid: base_registry.resolve(sid))
    frozen = dict(table)
    shared: Dict[str, GuideEvidence] = {}

    def evidence_for(kb: Any) -> GuideEvidence:
        if evidence is not None:
            return evidence
        if "evidence" not in shared:
            shared["evidence"] = GuideEvidence.from_knowledge_base(kb)
        return shared["evidence"]

    for sid in (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.INSTITUTIONAL):
        approved = frozenset(policy.get(sid.value) or ())
        if len(approved) > 1:
            table[sid] = (lambda kb, sid=sid, approved=approved: GuideCapableSpecialist.wrap(frozen[sid](kb), evidence_for(kb), approved))
    return table


def missing_specialists(registry: Optional[SpecialistRegistry]) -> List[str]:
    """Required specialist ids that ``registry`` does not hold."""
    if not isinstance(registry, SpecialistRegistry):
        return [s.value for s in REQUIRED_SPECIALISTS]
    return [s.value for s in REQUIRED_SPECIALISTS if not registry.is_registered(s)]


# ---------------------------------------------------------------------------
# Adapter result
# ---------------------------------------------------------------------------
@dataclass
class AdapterResult:
    """The outcome of one Plan F attempt. ``result`` is a ``QAResult`` only when ``outcome`` is ``ACCEPTED``."""

    outcome: PlanFOutcome
    reason: str
    question: str
    result: Optional[QAResult] = None
    trace: List[TraceEntry] = field(default_factory=list)
    specialist_id: Optional[str] = None
    status: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)
    dispatch: Optional[DispatchResult] = None
    verification: Any = None                     # plan_f_verification.FindingsVerification once the findings-aware layer ran (phase I-4)
    aggregation: Any = None                      # plan_f_aggregation.AggregationResult once the aggregation path ran (phase I-5)
    routing: Any = None                          # plan_f_routing.RoutingResult of the decision (phase I-6)
    handoff: Any = None                          # plan_f_handoff.HandoffRound once a specialist asked for a handoff (phase I-6)
    handoff_outcome: Optional[str] = None        # the round's code, or HANDOFF_ACCEPTED / HANDOFF_TARGET_REJECTED / HANDOFF_CONFLICT after aggregation
    guide_scope: Optional[Dict[str, Any]] = None  # post-freeze PF-3B: {"enabled", "selected", "named"} (source ids only)

    @property
    def requires_fallback(self) -> bool:
        return self.outcome is not PlanFOutcome.ACCEPTED

    @property
    def specialist_calls(self) -> int:
        """Specialist executions of this attempt: the dispatched runs plus the handoff targets (phase I-6)."""
        runs = [r for r in (getattr(self.dispatch, "runs", None) or []) if getattr(r, "state", None) == STATE_EXECUTED]
        seen = {getattr(getattr(r, "task", None), "task_id", None) for r in runs}
        extra = [rec for rec in (getattr(self.handoff, "executed", None) or ()) if getattr(rec, "task_id", None) not in seen]
        return len(runs) + len(extra)

    @property
    def path(self) -> Optional[str]:
        """How the attempt was decided: the single-finding pass-through path (phases I-1 to I-4), the multi-finding
        aggregation path (phase I-5), or None when neither produced a record."""
        if self.aggregation is not None:
            return PATH_AGGREGATE
        if self.verification is not None:
            return PATH_PASSTHROUGH
        return None

    @property
    def verified(self) -> bool:
        """Whether the findings-aware verification layer accepted the result for rendering: on the pass-through path its one
        record (phase I-4); on the aggregation path the aggregate itself AND every clause's own record (phase I-5)."""
        if self.aggregation is not None:
            a = self.aggregation
            clauses = tuple(getattr(a, "clauses", ()) or ())
            return (bool(getattr(a, "accepted", False)) and bool(getattr(a, "safe_to_render", False)) and bool(clauses)
                    and all(accepted_record(getattr(c, "verification", None)) for c in clauses))
        return accepted_record(self.verification)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "requires_fallback": self.requires_fallback,
            "question": self.question,
            "specialist_id": self.specialist_id,
            "status": self.status,
            "timings": dict(self.timings),
            "trace": [{"agent": e.agent, "decision": e.decision, "used_llm": e.used_llm, "ms": e.ms, "detail": e.detail} for e in self.trace],
            "result": self.result.to_dict() if self.result is not None else None,
            "verification": self.verification.to_dict() if self.verification is not None else None,
            "aggregation": self.aggregation.to_dict() if self.aggregation is not None else None,
            "path": self.path,
            "routing": self.routing.to_dict() if self.routing is not None else None,
            "handoff": self.handoff.to_dict() if self.handoff is not None else None,
            "handoff_outcome": self.handoff_outcome,
            "specialist_calls": self.specialist_calls,
            "guide_scope": dict(self.guide_scope) if self.guide_scope is not None else None,
        }


def handoff_outcome(round_: Any, record: Any) -> str:
    """The user-level outcome of an executed handoff round once the aggregator decided: accepted, the target's clause
    rejected, a conflict with the target's evidence, or the aggregate's own code."""
    if bool(getattr(record, "accepted", False)):
        return "HANDOFF_ACCEPTED"
    handed = {getattr(r, "request_clause", None) for r in (getattr(round_, "executed", None) or ())}
    reason = getattr(getattr(record, "reason", None), "value", None)
    if reason == "VALUE_CONFLICT" and any(i in handed or j in handed for i, j, _k in (getattr(record, "conflicts", None) or ())):
        return "HANDOFF_CONFLICT"
    failing = next((c for c in (getattr(record, "clauses", None) or ()) if not getattr(c, "covered", False)), None)
    if failing is not None and getattr(failing, "index", None) in handed:
        return "HANDOFF_TARGET_REJECTED"
    return str(reason or "HANDOFF_EXECUTED")


def accepted_record(verification: Any) -> bool:
    """Whether a findings-aware verification record accepted its result for rendering."""
    return (verification is not None and bool(getattr(verification, "accepted", False))
            and bool(getattr(verification, "safe_to_render", False)))


# ---------------------------------------------------------------------------
# Decision and result gating (the I-1 safe case)
# ---------------------------------------------------------------------------
def decision_problem(decision: CoordinatorDecision) -> Optional[str]:
    """Why a Coordinator decision is outside the I-1 safe case, or None."""
    md = decision.metadata or {}
    if md.get("complexity") == "greeting":
        return "greeting: no specialist needed"
    if not decision.selected_specialists or not decision.subtasks:
        return "no specialist selected: %s" % (decision.reason or "no reason given")
    if len(decision.selected_specialists) != 1 or len(decision.subtasks) != 1:
        return "multiple specialists selected (%s)" % ", ".join(s.value for s in decision.selected_specialists)
    if decision.requires_synthesis:
        return "synthesis required by the Coordinator"
    router = md.get("router") or {}
    if router.get("is_multipart"):
        return "multipart question"
    if md.get("complexity") in ("multipart", "cross_domain"):
        return "complexity %s" % md.get("complexity")
    if md.get("uncovered_clauses"):
        return "uncovered clause(s): %s" % "; ".join(str(c)[:80] for c in md["uncovered_clauses"])
    task = decision.subtasks[0]
    if task.requested_by != REQUESTED_BY:
        return "task %s was not issued by the Coordinator" % task.task_id
    focus = (task.context or {}).get("focus_clauses") or []
    if len(focus) != 1 or not isinstance(focus[0], str) or not focus[0].strip():
        return "task carries %d focus clauses; I-1 renders one" % len(focus)
    return None


def result_problem(findings: SpecialistFindings) -> Optional[str]:
    """Why a specialist result is not eligible for rendering in I-1, or None."""
    if findings.status is not FindingStatus.SUPPORTED:
        extra = ""
        if findings.status is FindingStatus.ERROR:
            extra = ": %s" % str((findings.metadata or {}).get("error") or findings.summary)[:120]
        return "status %s%s" % (findings.status.value, extra)
    if findings.handoff_requests:
        return "handoff pending to %s" % ", ".join(sorted({h.requested_specialist.value for h in findings.handoff_requests}))
    if findings.missing:
        return "result carries missing entries"
    if not findings.findings:
        return "supported result without findings"
    if findings.llm_used:
        return "specialist reported an LLM call"
    clauses = (findings.metadata or {}).get("clauses")
    if not isinstance(clauses, list) or len(clauses) != 1 or not isinstance(clauses[0], dict):
        return "expected exactly one clause report, found %s" % (len(clauses) if isinstance(clauses, list) else "none")
    if (findings.metadata or {}).get("unprocessed_clauses"):
        return "unprocessed clauses"
    report = clauses[0]
    if _guides.has_guide_record(report):                          # post-freeze PF-3C (D-111): a guide clause report
        return _guides.guide_report_problem(report)
    if findings.specialist_id is SpecialistId.INSTITUTIONAL:
        if report.get("outcome") != INSTITUTIONAL_OUTCOME_READ:
            return "institutional clause outcome %s" % report.get("outcome")
        if report.get("evidence_tier_reached") not in INSTITUTIONAL_TIERS:
            return "institutional evidence tier %s is not eligible" % report.get("evidence_tier_reached")
        return None
    if report.get("outcome") != FindingStatus.SUPPORTED.value:
        return "clause outcome %s" % report.get("outcome")
    if report.get("ask_shape") != ASK_SPECIFIC_RECOGNISED:
        return "ask shape %s: only a recognised specific ask is eligible (topical support is not)" % report.get("ask_shape")
    if not report.get("requested_details"):
        return "specific ask without requested details"
    if report.get("uncovered_details"):
        return "uncovered details: %s" % "; ".join(str(d)[:60] for d in report["uncovered_details"])
    if "relation" in report and report.get("relation") != RELATION_STRONG:
        return "relation %s is not strong" % report.get("relation")
    return None


# ---------------------------------------------------------------------------
# Guard 1: the quote must exist in its source chunk
# ---------------------------------------------------------------------------
def quote_problem(finding: Any, chunks: Sequence[str], metadata: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Why ``finding`` is not an exact quote of its own chunk, or None. Exact substring, no normalisation."""
    quote = getattr(finding, "evidence_quote", None)
    if not isinstance(quote, str) or not quote.strip():
        return "empty quote"
    meta = getattr(finding, "metadata", None)
    if not isinstance(meta, dict):
        return "finding has no metadata"
    cid = meta.get("chunk_id")
    if isinstance(cid, bool) or not isinstance(cid, int):
        return "chunk_id missing or not an integer"
    if not 0 <= cid < len(chunks):
        return "chunk_id %d out of range" % cid
    chunk = chunks[cid]
    if not isinstance(chunk, str) or quote not in chunk:
        return "quote not found in chunk %d" % cid
    if cid >= len(metadata) or not isinstance(metadata[cid], dict):
        return "no metadata for chunk %d" % cid
    cmeta = metadata[cid]
    if cmeta.get("chunk_id") not in (None, cid):
        return "metadata chunk id %r does not match %d" % (cmeta.get("chunk_id"), cid)
    page = getattr(finding, "page", None)
    if isinstance(page, bool) or not isinstance(page, int):
        return "finding has no page"
    if cmeta.get("page") != page:
        return "page mismatch: finding %s, chunk %s" % (page, cmeta.get("page"))
    source_id = cmeta.get("source_id")
    if source_id is not None and source_id != getattr(finding, "source_id", None):
        return "source mismatch: finding %r, chunk %r" % (getattr(finding, "source_id", None), source_id)
    return None


def source_conflict_problem(finding: Any, metadata: Sequence[Dict[str, Any]], guide_index: Optional[_guides.GuideIndex] = None) -> Optional[str]:
    """Why ``finding`` must not be served because its chunk carries a registered source conflict (post-freeze PF-2),
    or None. A conflict is a statement another version or authority of the same document contradicts; the registry
    marks the chunks (``meta["source_conflict"]``) and this guard keeps them out of every Plan F answer. Read after
    :func:`quote_problem` passed, so the chunk id is known to be valid; an unreadable id is simply not a conflict.
    Post-freeze PF-5B (D-114): a finding that carries its D-111 guide record is checked at claim level when the guide
    index is given (``GuideIndex.conflict_problem``: the unit that states the disputed claim is refused, another unit
    of the same chunk is not); every other finding keeps the chunk-level rule."""
    meta = getattr(finding, "metadata", None)
    cid = meta.get("chunk_id") if isinstance(meta, dict) else None
    if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < len(metadata):
        return None
    if guide_index is not None and _guides.has_guide_record(finding):
        problem = guide_index.conflict_problem(cid, str(getattr(finding, "evidence_quote", "") or ""))
        return None if problem is None else "%s; the statement is not served" % problem
    cmeta = metadata[cid]
    topic = cmeta.get("source_conflict") if isinstance(cmeta, dict) else None
    if isinstance(topic, str) and topic:
        return "source conflict %r: the statement is contradicted by another version of its document and is not served" % topic
    return None


def guide_evidence_problem(clause: str, quote: str, finding: Any, chunks: Sequence[str],
                           metadata: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Why a finding resting on an official guide must not answer ``clause`` (post-freeze PF-3B, D-110; PF-3C, D-111), or
    None. Read after :func:`quote_problem` passed. Handbook evidence is never refused here: a guide answers only for the
    system it documents and never a policy, entitlement or obligation ask. Since PF-3C guide evidence reaches Plan F only
    through the D-111 guide path: a finding that carries its D-111 record is verified as a set by
    ``guide_grounding.verify_guide_findings``; any other guide finding keeps the PF-3B single-quote procedure rule and is
    then refused."""
    meta = getattr(finding, "metadata", None)
    cid = meta.get("chunk_id") if isinstance(meta, dict) else None
    if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < len(metadata):
        return None
    cmeta = metadata[cid]
    from .source_scopes import is_guide_meta, procedure_problem, source_authority_problem
    if not is_guide_meta(cmeta):
        return None
    from .guide_inference import inferred_sources                 # post-freeze PF-5C (D-115): recomputed from the clause
    problem = source_authority_problem(clause, cmeta, inferred_sources(clause))
    if problem is None and not _guides.has_guide_record(finding):
        problem = procedure_problem(clause, quote or "", chunks[cid] if cid < len(chunks) else "", cmeta)
        if problem is None:
            problem = "guide evidence outside the D-111 guide path is not served"
    return problem


# ---------------------------------------------------------------------------
# Finding -> legacy evidence item
# ---------------------------------------------------------------------------
def adapt_finding(finding: Finding, specialist_id: SpecialistId, focus_clause: str, chunks: Sequence[str],
                  metadata: Sequence[Dict[str, Any]], *, clause_index: Optional[int] = None) -> Dict[str, Any]:
    """The legacy evidence item for one finding that passed :func:`quote_problem`.

    ``chunk`` is the finding's own quote (the exact evidence the answer rests on), ``meta`` is a copy of the source
    chunk's metadata with the finding's provenance under ``meta["plan_f"]``: a provision slice (phase I-5) keeps its
    ``derived_from`` and ``slice`` offsets, and ``clause_index`` names the request clause the item answers on the
    aggregation path. No rerank score is invented: the Verifier reads ``rerank_score`` with a default and only
    orders pages by it.
    """
    problem = quote_problem(finding, chunks, metadata)
    if problem is not None:
        raise ValueError("finding %s cannot be adapted: %s" % (getattr(finding, "finding_id", "?"), problem))
    cid = finding.metadata["chunk_id"]
    meta = dict(metadata[cid])
    plan_f: Dict[str, Any] = {
        "specialist_id": specialist_id.value,
        "finding_id": finding.finding_id,
        "claim": finding.claim,
        "evidence_quote": finding.evidence_quote,
        "source_id": finding.source_id,
        "source_title": finding.source_title,
        "page": finding.page,
        "confidence": finding.confidence,
        "chunk_id": cid,
        "focus_clause": focus_clause,
    }
    for key in ("evidence_tier", "relation", "office_id", "office_name", "contact_values", "details_covered", "derived_from", "slice",
                "slice_provision", "projected_coverage", _guides.GUIDE_KEY):
        if key in finding.metadata:
            plan_f[key] = finding.metadata[key]
    if clause_index is not None:
        plan_f["clause_index"] = int(clause_index)
    meta["plan_f"] = plan_f
    return {"idx": cid, "chunk": finding.evidence_quote, "meta": meta}


# ---------------------------------------------------------------------------
# Guard 2: the asked subject must be bound to the finding
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z][A-Za-z'’&-]*")
_ACRONYM = re.compile(r"^[A-Z][A-Z&]{1,5}s?$")
_CONNECTORS = frozenset({"of", "and", "&", "for", "the", "in"})
#: Capitalised words that open or frame a question and never name an entity.
_FRAME_WORDS = frozenset({
    "what", "who", "whom", "whose", "which", "how", "when", "where", "why", "is", "are", "do", "does", "did", "can",
    "could", "should", "would", "will", "may", "must", "shall", "has", "have", "had", "was", "were", "tell", "please",
    "give", "explain", "describe", "list", "name", "show", "i", "my", "we", "our", "you", "your", "if", "as", "at", "on",
    "to", "by", "with", "from", "about", "there", "this", "that", "these", "those", "it", "its", "be", "any", "all", "a", "an",
})
#: Generic words that are no entity on their own and never need to appear in a quote.
_GENERIC_WORDS = frozenset({"university", "uos", "uae", "sharjah", "faculty", "handbook", "members", "member"})
#: A unit noun that may close a printed name but is often omitted ("Human Resources Department" / "Human Resources").
_UNIT_NOUNS = frozenset({
    "department", "office", "center", "centre", "unit", "section", "division", "directorate", "committee", "council",
    "deanship", "institute", "college", "school", "board", "library", "service", "services", "hospital", "clinic",
})


def _norm(text: str) -> str:
    text = (text or "").replace("’", "'").replace("‘", "'").replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


def _phrase_in(phrase: str, text: str) -> bool:
    p = _norm(phrase)
    return bool(p) and re.search(r"(?<![a-z0-9'])" + re.escape(p) + r"(?![a-z0-9'])", _norm(text)) is not None


def _acronym_in(acronym: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(acronym) + r"(?![A-Za-z0-9])", text or "") is not None


def named_entities(clause: str) -> List[str]:
    """Named entities of a clause, read deterministically: runs of capitalised words (connectors allowed between
    them) and acronyms. Question frame words, the University and other generic words are never entities."""
    tokens = _WORD.findall(clause or "")
    entities: List[str] = []
    run: List[str] = []

    def flush() -> None:
        while run and run[-1].lower() in _CONNECTORS:
            run.pop()
        if run and (len(run) >= 2 or _ACRONYM.match(run[0])):
            entities.append(" ".join(run))
        run.clear()

    for tok in tokens:
        low = tok.lower()
        if _ACRONYM.match(tok) and low not in _GENERIC_WORDS:
            run.append(tok)
        elif tok[0].isupper() and low not in _FRAME_WORDS and low not in _GENERIC_WORDS and low not in _CONNECTORS:
            run.append(tok)
        elif run and low in _CONNECTORS:
            run.append(tok)
        else:
            flush()
    flush()
    return list(dict.fromkeys(entities))


def entity_in_text(entity: str, text: str) -> bool:
    """Whether a named entity appears in ``text``: an acronym case-sensitively, a phrase case-insensitively, also
    without a closing unit noun when at least two words remain ("Human Resources Department" -> "Human Resources")."""
    words = entity.split()
    if len(words) == 1 and _ACRONYM.match(words[0]):
        return _acronym_in(words[0], text)
    if _phrase_in(entity, text):
        return True
    if len(words) >= 3 and words[-1].lower() in _UNIT_NOUNS:
        return _phrase_in(" ".join(words[:-1]), text)
    return False


def clause_covered_by(clause: str, item: Dict[str, Any]) -> bool:
    """Whether the clause's content words are covered by the item's quote under the unchanged Verifier's rules
    (the clause is checked as if it were an answer drafted from that one item)."""
    synthesis = SynthesisResult(answer=clause, claimed_pages=[], prompt_version=SUBJECT_CHECK_VERSION)
    evidence = EvidenceResult(clause, "policy", [item], None, True, None, False, None)
    verdict = verify(clause, synthesis, [evidence], [], is_retry=True)
    return isinstance(verdict, VerifyResult) and bool(verdict.accepted)


def _office_names(office_id: str) -> List[str]:
    office = _nav.OFFICES_BY_ID.get(office_id)
    if office is None:
        return []
    names = [office.name] + [alias for alias, _pages in office.aliases] + [alias for alias, _pages in office.subject_aliases]
    return [n for n in names if n]


def _office_named_in(office_id: str, text: str) -> bool:
    for name in _office_names(office_id):
        if _ACRONYM.match(name):
            if _acronym_in(name, text):
                return True
        elif _phrase_in(name, text):
            return True
    return False


def _institutional_binding(clause: str, report: Dict[str, Any], finding: Finding) -> Tuple[bool, str]:
    meta = finding.metadata
    office_id = meta.get("office_id")
    if not isinstance(office_id, str) or not office_id:
        return False, "finding names no office"
    tier = meta.get("evidence_tier")
    quote = finding.evidence_quote
    if office_id == _nav.UNIVERSITY:
        if tier != _nav.STATED_FACT:
            return False, "university subject with evidence tier %s" % tier
        if not (_phrase_in("university", quote) or _acronym_in("UoS", quote)):
            return False, "the quote does not name the University"
        return True, "bound: the University is the stated subject"
    if not _office_names(office_id):
        return False, "office %s is not in the lexicon" % office_id
    if not _office_named_in(office_id, quote):
        return False, "office %s is not named in the quote" % office_id
    asked = office_id in (report.get("offices") or []) or _office_named_in(office_id, clause)
    if tier == _nav.E1:
        if not (meta.get("row_complete") is True and meta.get("directory_entry") == "complete_row" and meta.get("contact_values")):
            return False, "directory entry is not a complete row"
        if not asked:
            return False, "office %s is not the office the clause asks about" % office_id
        return True, "bound: complete directory row of the asked office"
    if tier in (_nav.E2, _nav.STATED_FACT):
        if meta.get("binding") not in INSTITUTIONAL_BINDINGS:
            return False, "statement binding %r does not name its subject" % meta.get("binding")
        if not asked and report.get("ask_kind") != INSTITUTIONAL_RESPONSIBLE_OFFICE:
            return False, "office %s is not the office the clause asks about" % office_id
        return True, "bound: statement whose subject is the office"
    return False, "evidence tier %s is not eligible" % tier


def subject_binding(clause: str, report: Dict[str, Any], finding: Finding, item: Dict[str, Any],
                    specialist_id: SpecialistId) -> Tuple[bool, str]:
    """Whether the asked subject is bound to ``finding``. Deterministic; no model; uncertain means unbound."""
    if _guides.has_guide_record(finding):                        # post-freeze PF-3C: guide findings are bound as a set only
        return False, "a guide finding is bound only by the D-111 set check (guide_grounding.verify_guide_findings)"
    if specialist_id is SpecialistId.INSTITUTIONAL:
        return _institutional_binding(clause, report, finding)
    for entity in named_entities(clause):
        if not entity_in_text(entity, finding.evidence_quote):
            return False, "entity %r is not in the quote" % entity
    requested = [d for d in (report.get("requested_details") or []) if isinstance(d, str)]
    covered = set(d for d in (finding.metadata.get("details_covered") or []) if isinstance(d, str))
    if not requested:
        return False, "the clause report lists no requested detail"
    uncovered = [d for d in requested if d not in covered]
    if uncovered:
        return False, "requested detail not covered by this finding: %s" % uncovered[0][:80]
    if not clause_covered_by(clause, item):
        return False, "the quote does not cover the clause's content words"
    return True, "bound: entities present, requested details covered, clause covered"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_answer(items: Sequence[Dict[str, Any]]) -> str:
    """The accepted quotes, verbatim, one per paragraph. Nothing is added or reworded; the first step of a D-111 guide
    procedure puts its printed heading on its own line (post-freeze PF-3C, ``guide_grounding.render_part``)."""
    parts = []
    for item in items:
        text = str(item.get("chunk") or "").strip()
        if text:
            record = ((item.get("meta") or {}).get("plan_f") or {}).get(_guides.GUIDE_KEY)
            parts.append(_guides.render_part(text, record))
    return "\n\n".join(parts)


def accepted_pages(items: Sequence[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for item in items:
        page = item["meta"].get("page")
        if isinstance(page, int) and not isinstance(page, bool) and page not in pages:
            pages.append(page)
    return pages


def guide_citation_items(items: Sequence[Dict[str, Any]], index: Optional[_guides.GuideIndex]) -> List[Dict[str, Any]]:
    """Post-freeze PF-3D (D-112): the citation label of each verified guide item is the printed heading of its verified
    procedure or structure (``guide_grounding.citation_heading``), or none (the page alone) for a statement, instead of
    the chunk's carried page hint. Runs after every check, so it changes no decision; any other item is returned as is."""
    out: List[Dict[str, Any]] = []
    for item in items:
        meta = item.get("meta") or {}
        record = (meta.get("plan_f") or {}).get(_guides.GUIDE_KEY)
        if not record:
            out.append(item)
            continue
        meta = {k: v for k, v in meta.items() if k != "section_title"}
        meta["section"] = _guides.citation_heading(record, index)
        out.append(dict(item, meta=meta))
    return out


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
class PlanFRuntime:
    """A complete specialist registry together with the corpus it was built on. Created explicitly by the caller
    (``build_runtime``); there is no module-level instance."""

    def __init__(self, registry: SpecialistRegistry, chunks: Sequence[str], metadata: Sequence[Dict[str, Any]], *,
                 guide_registry: Optional[SpecialistRegistry] = None, guide_records: Sequence[Any] = (),
                 guide_index: Optional[_guides.GuideIndex] = None) -> None:
        if not isinstance(registry, SpecialistRegistry):
            raise TypeError("PlanFRuntime expects a SpecialistRegistry, got %s" % type(registry).__name__)
        if guide_registry is not None and not isinstance(guide_registry, SpecialistRegistry):
            raise TypeError("PlanFRuntime expects a SpecialistRegistry for guides, got %s" % type(guide_registry).__name__)
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata differ in length (%d vs %d)" % (len(chunks), len(metadata)))
        self.registry = registry                              # the I-9 specialists: handbook-only source boundary
        self.guide_registry = guide_registry                  # post-freeze PF-3B (D-110): specialists with approved guides
        self.guide_records = tuple(guide_records or ())
        self.guide_index = guide_index                        # post-freeze PF-3C (D-111): procedures, registry and approved sets
        self.chunks = chunks
        self.metadata = metadata

    def select_registry(self, text: str) -> Tuple[SpecialistRegistry, Dict[str, Any]]:
        """The registry a question runs on (post-freeze PF-3B): the guide-enabled one only when the question names the
        system an approved guide documents, or (PF-5C, D-115) guide inference accepts a candidate guide for it; every other
        question runs on the I-9 specialists, so no guide chunk can reach a handbook-only question. Deterministic; the
        record holds source ids, signal keys and scores only."""
        if self.guide_registry is None:
            return self.registry, {"enabled": False, "selected": False, "named": []}
        from .source_scopes import guide_sources_named, usable_record
        named = guide_sources_named(text, self.guide_records)
        if named:
            return self.guide_registry, {"enabled": True, "selected": True, "named": named}
        # post-freeze PF-5C (D-115): a question that names no system runs on the guide registry only when guide inference
        # accepts a candidate guide for the question or one of its clauses (as the Coordinator splits it); the record holds
        # signal keys, scores and source ids only
        from .agents.coordinator import analyze
        from .guide_inference import infer_question
        inference = infer_question(text, analyze(text).clauses if text else ())
        usable = {r.source_id for r in self.guide_records if usable_record(r)}
        inferred = [s for s in inference.candidates if s in usable]
        record = {"enabled": True, "selected": bool(inferred), "named": [], "inferred": inferred, "inference": inference.to_dict()}
        return (self.guide_registry if inferred else self.registry), record

    @property
    def missing_specialists(self) -> List[str]:
        return missing_specialists(self.registry)

    def corpus_word(self, word: str) -> bool:
        """Post-freeze PF-5B (D-114): whether the knowledge base prints ``word`` (lower case) anywhere; a printed word is never
        a misspelled system name. The word set is built on the first question that needs it."""
        words = self.__dict__.get("_corpus_words")
        if words is None:
            words = frozenset(re.findall(r"[a-z]+", "\n".join(str(c) for c in self.chunks).lower()))
            self.__dict__["_corpus_words"] = words
        return word.lower() in words

    def attempt(self, question: str) -> AdapterResult:
        """Try to answer ``question`` through Plan F: the single-clause case of phases I-1 to I-4 (one specialist, one
        request clause; the pass-through path) or, since phase I-5, the deterministic aggregation of the findings of a
        multi-clause execution. Never raises."""
        state = _Attempt(question)
        state.trace.append(TraceEntry("plan_f_adapter", "start", False, 0.0, {"phase": PHASE, "question_chars": len(state.text)}))
        missing = self.missing_specialists
        if missing:
            state.add("plan_f_registry", "incomplete", detail={"missing": missing})
            return state.finish(PlanFOutcome.REGISTRY_UNAVAILABLE, "registry lacks: %s" % ", ".join(missing))
        state.add("plan_f_registry", "ok", detail={"specialists": [s.value for s in self.registry.ids()]})
        if self.guide_registry is not None and state.text:
            # post-freeze PF-5B (D-114): a misspelled system name is corrected for routing and grounding only; the typed
            # question stays the display, legacy and log text (records hold the canonical names, never the typed word)
            from .name_spelling import correct_with_spans
            state.routing_text, spelling, state.spelling_spans = correct_with_spans(state.text, self.corpus_word)
            if spelling:
                state.add("system_name_spelling", "corrected", detail={"names": [r["name"] for r in spelling],
                                                                         "kinds": [r["kind"] for r in spelling]})
        state.registry, state.guide = self.select_registry(state.routing_text)
        if state.guide.get("enabled"):
            state.add("source_scope", "guides" if state.guide["selected"] else "handbook",
                      detail={k: v for k, v in state.guide.items() if k != "inference"})
            inference = state.guide.get("inference")
            if inference:                                         # post-freeze PF-5C (D-115): candidates, signals, scores, margin
                state.add("guide_inference", "accepted" if inference["accepted"] else "refused:%s" % inference["lead"]["reason"],
                          detail=dict(inference["lead"], clauses=len(inference["parts"])))
        if not state.text:
            state.add("coordinator", "skipped:empty")
            return state.finish(PlanFOutcome.NOT_APPLICABLE, "empty question")

        # ---- Coordinator ------------------------------------------------------
        t = time.perf_counter()
        try:
            decision = coordinate(state.routing_text)
        except Exception as exc:
            state.add("coordinator", "error", detail={"error": redact_error(exc)})
            return state.finish(PlanFOutcome.ERROR_REDACTED, "coordinator raised: %s" % redact_error(exc))
        state.timings["coordinator_ms"] = (time.perf_counter() - t) * 1000.0
        md = decision.metadata or {}
        state.add("coordinator", ",".join(s.value for s in decision.selected_specialists) or "none", detail={
            "selected": [s.value for s in decision.selected_specialists],
            "complexity": md.get("complexity"),
            "requires_synthesis": decision.requires_synthesis,
            "is_multipart": bool((md.get("router") or {}).get("is_multipart")),
            "route": (md.get("router") or {}).get("route"),
            "uncovered_clauses": len(md.get("uncovered_clauses") or []),
            "dropped_domains": list(md.get("dropped_domains") or []),
            "confidence": decision.confidence,
            "reason": (decision.reason or "")[:120],
        })
        from . import plan_f_aggregation as _aggregation          # imported here: that module builds on this one's guards
        from . import plan_f_routing as _routing                  # phase I-6: one owner per request clause before any specialist runs
        decision = self._inferred_owner(state, decision)          # post-freeze PF-5C (D-115)
        t = time.perf_counter()
        decision, routing = _routing.refine(decision)
        state.timings["routing_ms"] = (time.perf_counter() - t) * 1000.0
        state.routing = routing
        if routing.reason is not _routing.RoutingCode.ROUTE_UNCHANGED:
            state.add("routing", routing.reason.value, detail=routing.to_dict())
        if not routing.executable:
            return state.finish(PlanFOutcome.REJECTED_BY_ROUTING, "%s: %s" % (routing.reason.value, routing.detail),
                                specialist=",".join(routing.initial_specialists) or None)
        md = decision.metadata or {}
        problem = decision_problem(decision)
        clauses = _aggregation.request_clauses(decision)
        if problem is None and len(clauses) <= 1:
            decision = self._software_owner(state, decision)        # post-freeze PF-5B (D-114)
            return self._single_clause(state, decision, md)
        if _aggregation.applicable(decision):
            return self._aggregate(state, decision, md, clauses, _aggregation)
        if not decision.selected_specialists or md.get("complexity") == "greeting":
            return state.finish(PlanFOutcome.NOT_APPLICABLE, problem)
        return state.finish(PlanFOutcome.UNSAFE_RESULT, problem)

    # ---- post-freeze PF-5B (D-114): software navigation is the system's specialist's -------------------------------
    def _software_owner(self, state: "_Attempt", decision: CoordinatorDecision) -> CoordinatorDecision:
        """A one-clause decision the Coordinator gave to Institutional Navigation is given to the specialist that owns the guide
        the clause names, when the clause is a software question (an approved guide binds it with a non-contact ask), names no
        office, team, role or contact, and the guide is approved for exactly one other specialist and not for Institutional
        Navigation ("Where do I find the gradebook in Blackboard Ultra?"). Guide registry only; any other decision is returned
        unchanged. The trace records the specialist ids only."""
        if not (state.guide or {}).get("selected") or self.guide_index is None or len(decision.subtasks) != 1:
            return decision
        task = decision.subtasks[0]
        focus = (task.context or {}).get("focus_clauses") or []
        if task.specialist_id is not SpecialistId.INSTITUTIONAL or len(focus) != 1 or not isinstance(focus[0], str):
            return decision
        clause = focus[0]
        if _guides.ORGANIZATIONAL.search(clause):
            return decision
        from .source_scopes import guide_sources_named
        policy = self.guide_index.policy
        owners: set = set()
        sources = guide_sources_named(clause, self.guide_records)
        inferred = False
        if not sources:                                           # post-freeze PF-5C (D-115): an inferred system
            from .guide_inference import inferred_sources
            sources, inferred = list(inferred_sources(clause)), True
        for source_id in sources:
            if source_id in policy.get(SpecialistId.INSTITUTIONAL.value, frozenset()):
                continue
            binding, _problem = self.guide_index.bind(clause, source_id)
            if binding is None or binding.intent is _guides.GuideIntent.CONTACT:
                continue
            owners |= {sid for sid, ids in policy.items() if source_id in ids and sid != SpecialistId.INSTITUTIONAL.value}
        if len(owners) != 1:
            return decision
        owner = SpecialistId.parse(owners.pop())
        moved = dataclasses.replace(decision, selected_specialists=[owner],
                                    subtasks=[dataclasses.replace(task, specialist_id=owner, domain=owner.value)])
        if state.routing is not None and dataclasses.is_dataclass(state.routing):
            # the routing record names the specialist that now owns the clause (the chip and the diagnostics read it)
            state.routing = dataclasses.replace(state.routing, final_specialists=(owner.value,),
                                                owners_after=tuple((owner.value,) for _ in state.routing.owners_after))
        state.inferred_owner = inferred                           # D-115: then only a verified guide answer may be served
        detail = {"from": SpecialistId.INSTITUTIONAL.value, "to": owner.value}
        if inferred:
            detail["inferred"] = True
        state.add("software_navigation", "reassigned", detail=detail)
        return moved

    # ---- post-freeze PF-5C (D-115): a system question the Coordinator gives to no specialist --------------------------
    def _inferred_owner(self, state: "_Attempt", decision: CoordinatorDecision) -> CoordinatorDecision:
        """A one-clause decision with no specialist (no domain cue matched) is given to the specialist that owns the guide
        inferred for the question: the Coordinator's own system-owner rule ("system access question: blackboard guides are
        owned by teaching") applied to an inferred system. Guide inference must accept candidate guides, one of them must
        bind the clause with a non-contact ask, and they must be approved for exactly one specialist other than
        Institutional Navigation. Only a verified guide answer may then be served (``_single_clause``); otherwise Plan F
        declines exactly as before. Any other decision is returned unchanged; the trace records the specialist id only."""
        guide = state.guide or {}
        md = decision.metadata or {}
        if not guide.get("inferred") or self.guide_index is None or decision.selected_specialists or decision.subtasks:
            return decision
        if md.get("complexity") != "unknown" or md.get("journey"):
            return decision
        from .agents import coordinator as _coordinator
        analysis = _coordinator.analyze(state.routing_text)
        if analysis.greeting or analysis.journey or len(analysis.clauses) != 1:
            return decision
        from .guide_inference import inferred_sources
        clause = analysis.question
        policy = self.guide_index.policy
        owners: set = set()
        for source_id in inferred_sources(clause):
            binding, _problem = self.guide_index.bind(clause, source_id)
            if binding is None or binding.intent is _guides.GuideIntent.CONTACT:
                continue
            owners |= {sid for sid, ids in policy.items() if source_id in ids and sid != SpecialistId.INSTITUTIONAL.value}
        if len(owners) != 1:
            return decision
        owner = owners.pop()
        limits = md.get("limits") or {}
        specialist_limit, task_limit = int(limits.get("specialists", 3)), int(limits.get("subtasks", 3))
        tasks, uncovered = _coordinator._decompose(analysis, [owner], min(specialist_limit, task_limit))
        if len(tasks) != 1 or (tasks[0].context or {}).get("focus_clauses") != [clause]:
            return decision
        complexity = _coordinator._complexity(analysis, [owner])
        moved = dataclasses.replace(
            decision, selected_specialists=[SpecialistId.parse(owner)], subtasks=tasks,
            requires_synthesis=bool(analysis.is_multipart), confidence=_coordinator._confidence(analysis, [owner], True),
            reason="inferred system question: the inferred guide is owned by %s (D-115)" % owner,
            metadata=_coordinator._metadata(analysis, complexity, [], uncovered, specialist_limit, task_limit))
        state.inferred_owner = True
        state.add("guide_inference", "owner_assigned", detail={"to": owner})
        return moved

    # ---- the pass-through path (phases I-1 to I-4): one specialist, one request clause ---------------------------
    def _single_clause(self, state: "_Attempt", decision: CoordinatorDecision, md: Dict[str, Any]) -> AdapterResult:
        state.path = PATH_PASSTHROUGH
        task = decision.subtasks[0]
        clause = task.context["focus_clauses"][0]
        sid = task.specialist_id

        # ---- Dispatch ------------------------------------------------------------
        t = time.perf_counter()
        try:
            dispatched = dispatch(decision, state.registry, question=state.routing_text)
        except Exception as exc:
            state.add("dispatcher", "error", detail={"error": redact_error(exc)})
            return state.finish(PlanFOutcome.ERROR_REDACTED, "dispatcher raised: %s" % redact_error(exc), specialist=sid.value)
        state.timings["dispatch_ms"] = (time.perf_counter() - t) * 1000.0
        state.add("dispatcher", "executed:%d,pending:%d" % (len(dispatched.executed_specialists), len(dispatched.pending_specialists)),
                  detail={"executed": [s.value for s in dispatched.executed_specialists],
                          "pending": [s.value for s in dispatched.pending_specialists],
                          "handoffs": len(dispatched.handoffs), "llm_calls": dispatched.llm_calls})
        if dispatched.llm_calls != 0:
            return state.finish(PlanFOutcome.UNSAFE_RESULT, "the dispatch reported %d LLM call(s)" % dispatched.llm_calls,
                                specialist=sid.value, dispatched=dispatched)
        if len(dispatched.runs) != 1:
            return state.finish(PlanFOutcome.UNSAFE_RESULT, "expected one run, the dispatcher produced %d" % len(dispatched.runs),
                                specialist=sid.value, dispatched=dispatched)
        run = dispatched.runs[0]
        if run.state != STATE_EXECUTED or run.findings is None:
            state.add("specialist:%s" % run.specialist_id.value, "pending", detail={"reason": run.reason[:160]})
            return state.finish(PlanFOutcome.REQUIRES_FALLBACK, "task pending: %s" % run.reason, specialist=run.specialist_id.value,
                                dispatched=dispatched)
        findings = run.findings
        sid = findings.specialist_id
        status = findings.status.value
        state.timings["specialist_ms"] = float(findings.ms)
        state.add("specialist:%s" % sid.value, status, bool(findings.llm_used), {
            "findings": len(findings.findings), "handoffs": [h.requested_specialist.value for h in findings.handoff_requests],
            "missing": len(findings.missing), "limitations": len(findings.limitations), "specialist_ms": findings.ms,
            "summary": (findings.summary or "")[:160],
        })
        if state.inferred_owner and not ((findings.metadata or {}).get("guide_path") or {}).get("answered"):
            # post-freeze PF-5C (D-115): a decision guide inference changed serves only a verified guide answer
            state.add("guide_inference", "declined", detail={"specialist": sid.value})
            return state.finish(PlanFOutcome.NOT_APPLICABLE, "D-115: the decision was changed by guide inference and the guide path did not "
                                "answer; Plan F declines as before", specialist=sid.value, status=status, dispatched=dispatched)
        if findings.handoff_requests:                             # phase I-6: one bounded handoff round, then the aggregator decides
            from . import plan_f_aggregation as _aggregation
            return self._handoff_round(state, decision, md, _aggregation.request_clauses(decision), dispatched, specialist=sid.value, status=status)
        problem = result_problem(findings)
        if problem is not None:
            state.add("plan_f_gate", "not_eligible", detail={"reason": problem[:200]})
            return state.finish(PlanFOutcome.UNSAFE_RESULT, problem, specialist=sid.value, status=status, dispatched=dispatched)

        # ---- Guards, rendering, verification -----------------------------------------
        try:
            t = time.perf_counter()
            for finding in findings.findings:
                problem = quote_problem(finding, self.chunks, self.metadata)
                if problem is not None:
                    state.add("quote_guard", "rejected", detail={"finding": finding.finding_id, "reason": problem})
                    return state.finish(PlanFOutcome.REJECTED_BY_QUOTE_GUARD, "%s: %s" % (finding.finding_id, problem),
                                        specialist=sid.value, status=status, dispatched=dispatched)
                conflict = source_conflict_problem(finding, self.metadata, self.guide_index)
                if conflict is not None:                          # post-freeze PF-2
                    state.add("source_conflict_guard", "rejected", detail={"finding": finding.finding_id, "reason": conflict})
                    return state.finish(PlanFOutcome.REJECTED_BY_SOURCE_CONFLICT, "%s: %s" % (finding.finding_id, conflict),
                                        specialist=sid.value, status=status, dispatched=dispatched)
                authority = guide_evidence_problem(clause, finding.evidence_quote, finding, self.chunks, self.metadata)
                if authority is not None:                         # post-freeze PF-3B (D-110)
                    state.add("source_authority_guard", "rejected", detail={"finding": finding.finding_id, "reason": authority})
                    return state.finish(PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY, "%s: %s" % (finding.finding_id, authority),
                                        specialist=sid.value, status=status, dispatched=dispatched)
            state.add("quote_guard", "passed", detail={"findings": len(findings.findings)})
            report = findings.metadata["clauses"][0]
            accepted: List[Dict[str, Any]] = []
            dropped: List[Dict[str, Any]] = []
            guided = _guides.has_guide_record(report) or any(_guides.has_guide_record(f) for f in findings.findings)
            if guided:
                # post-freeze PF-3C (D-111): guide evidence is bound and checked for completeness as ONE answer
                problem = _guides.verify_guide_findings(clause, findings.findings, sid.value, self.guide_index)
                if problem is None and not _guides.has_guide_record(report):
                    problem = "guide findings under a clause report without its D-111 record"
                if problem is not None:
                    authority = _guides.is_authority_problem(problem)
                    state.add("guide_guard", "rejected", detail={"reason": problem[:200]})
                    return state.finish(PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY if authority else PlanFOutcome.REJECTED_BY_SUBJECT_GUARD,
                                        "D-111: %s" % problem, specialist=sid.value, status=status, dispatched=dispatched)
                accepted = [adapt_finding(f, sid, clause, self.chunks, self.metadata) for f in findings.findings]
                state.add("guide_guard", "passed", detail={"form": report[_guides.GUIDE_KEY].get("form"), "findings": len(accepted)})
            for n, finding in enumerate(() if guided else findings.findings):
                item = adapt_finding(finding, sid, clause, self.chunks, self.metadata)
                bound, why = subject_binding(clause, report, finding, item, sid)
                if bound:
                    accepted.append(item)
                elif n == 0:
                    state.add("subject_guard", "rejected", detail={"finding": finding.finding_id, "reason": why})
                    return state.finish(PlanFOutcome.REJECTED_BY_SUBJECT_GUARD, "%s: %s" % (finding.finding_id, why),
                                        specialist=sid.value, status=status, dispatched=dispatched)
                else:
                    dropped.append({"finding": finding.finding_id, "reason": why})
            state.timings["guards_ms"] = (time.perf_counter() - t) * 1000.0
            state.add("subject_guard", "passed", detail={"accepted": [i["meta"]["plan_f"]["finding_id"] for i in accepted], "dropped": dropped})
            if not accepted:
                return state.finish(PlanFOutcome.REQUIRES_FALLBACK, "no finding left to render", specialist=sid.value, status=status,
                                    dispatched=dispatched)

            # ---- Findings-aware verification (phase I-4): the structured result must suit the exact request ----------
            # (since phase I-5 a quote rejected as excess evidence may be replaced by its provision slice, re-verified)
            from . import plan_f_verification as _verification      # imported here: that module builds on this one's guards
            t = time.perf_counter()
            sliced = _verification.verify_findings_sliced(clause, findings, self.chunks, self.metadata, guide_index=self.guide_index)
            verification = sliced.verification
            state.timings["findings_verify_ms"] = (time.perf_counter() - t) * 1000.0
            detail = verification.to_dict()
            if sliced.sliced:
                detail["sliced"] = [list(s) for s in sliced.sliced]
            state.add("findings_verifier", "accepted" if verification.accepted else "rejected", detail=detail)
            if not verification.accepted or not verification.safe_to_render:
                return state.finish(PlanFOutcome.REJECTED_BY_FINDINGS_VERIFIER, "%s: %s" % (verification.reason.value, verification.detail),
                                    specialist=sid.value, status=status, dispatched=dispatched, verification=verification)
            keep = set(verification.accepted_finding_ids)
            kept = [f for f in sliced.findings.findings if f.finding_id in keep]
            for finding in kept:                                  # post-freeze PF-3B: a provision slice is checked again
                authority = guide_evidence_problem(clause, finding.evidence_quote, finding, self.chunks, self.metadata)
                if authority is not None:
                    state.add("source_authority_guard", "rejected", detail={"finding": finding.finding_id, "reason": authority})
                    return state.finish(PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY, "%s: %s" % (finding.finding_id, authority),
                                        specialist=sid.value, status=status, dispatched=dispatched, verification=verification)
            accepted = [adapt_finding(f, sid, clause, self.chunks, self.metadata) for f in kept]
            if not accepted:
                return state.finish(PlanFOutcome.REQUIRES_FALLBACK, "no verified finding left to render", specialist=sid.value, status=status,
                                    dispatched=dispatched, verification=verification)

            answer = render_answer(accepted)
            pages = accepted_pages(accepted)
            if not answer or not pages:
                return state.finish(PlanFOutcome.REQUIRES_FALLBACK, "nothing renderable", specialist=sid.value, status=status,
                                    dispatched=dispatched)
            route = (md.get("router") or {}).get("route") or "policy"
            draft = SynthesisResult(answer=answer, claimed_pages=list(pages), prompt_version=PROMPT_VERSION)
            evidence = EvidenceResult(clause, route, list(accepted), None, True, None, False, None)
            t = time.perf_counter()
            verdict = verify(state.text, draft, [evidence], [], is_retry=False)
            state.timings["verify_ms"] = (time.perf_counter() - t) * 1000.0
            if not isinstance(verdict, VerifyResult):
                raise TypeError("verify() returned %s" % type(verdict).__name__)
            cited = [int(p) for p in verdict.pages if isinstance(p, int) and not isinstance(p, bool) and p in pages]
            state.add("verifier", "accepted" if verdict.accepted else "rejected", detail={
                "impl": "agents.verifier", "pages": cited, "unsupported_claims": [c[:80] for c in verdict.unsupported_claims[:3]]})
            if not verdict.accepted or not cited:
                return state.finish(PlanFOutcome.REJECTED_BY_VERIFIER,
                                    "verifier %s" % ("rejected the rendered answer" if not verdict.accepted else "cited no accepted page"),
                                    specialist=sid.value, status=status, dispatched=dispatched, verification=verification)

            accepted = guide_citation_items(accepted, self.guide_index)    # post-freeze PF-3D: printed headings only
            first = accepted[0]["meta"]
            best_section = first.get("section_title") or first.get("section") or "Page %d" % cited[0]
            preview = accepted[0]["chunk"]
            if len(preview) > EVIDENCE_PREVIEW_CHARS:
                preview = preview[:EVIDENCE_PREVIEW_CHARS].rstrip() + "..."
            state.timings["generation_ms"] = 0.0
            result = QAResult(answer=answer, pages=cited, best_section=str(best_section), evidence=preview, query_type=str(route),
                              items=list(accepted), used_llm=False, num_candidates=len(findings.findings), num_reranked=len(accepted),
                              llm_calls=0, retried=False, sub_questions=[])
            return state.finish(PlanFOutcome.ACCEPTED, "rendered %d supported finding(s) of %s" % (len(accepted), sid.value),
                                result=result, specialist=sid.value, status=status, dispatched=dispatched, verification=verification)
        except Exception as exc:                                  # a defect here must never reach the user as an answer
            state.add("plan_f_adapter", "error", detail={"error": redact_error(exc)})
            return state.finish(PlanFOutcome.ERROR_REDACTED, "adapter raised: %s" % redact_error(exc), specialist=sid.value, status=status,
                                dispatched=dispatched)

    # ---- the aggregation path (phase I-5): several request clauses or several specialists ------------------------
    def _aggregate(self, state: "_Attempt", decision: CoordinatorDecision, md: Dict[str, Any], clauses: List[str],
                   aggregation: Any) -> AdapterResult:
        state.path = PATH_AGGREGATE
        specialists = [s.value for s in decision.selected_specialists]
        joined = ",".join(specialists)
        state.add("aggregator", "shape:%s" % (md.get("complexity") or "unknown"),
                  detail={"phase": aggregation.PHASE, "clauses": len(clauses), "tasks": len(decision.subtasks), "specialists": specialists})
        try:
            problem = aggregation.request_problem(decision)
            if problem is not None:                               # knowable from the decision alone: no specialist runs
                record = aggregation.request_rejection(decision, *problem)
                state.add("aggregator", "rejected", detail=record.to_dict())
                return state.finish(PlanFOutcome.REJECTED_BY_AGGREGATOR, "%s: %s" % (problem[0].value, problem[1]), specialist=joined,
                                    aggregation=record)
        except Exception as exc:                                  # a defect in the request check is a redacted error, never a raise
            state.add("aggregator", "error", detail={"error": redact_error(exc)})
            return state.finish(PlanFOutcome.ERROR_REDACTED, "aggregator raised: %s" % redact_error(exc), specialist=joined)

        # ---- Dispatch: every task of the decision, once; nothing re-routed, no handoff executed ------------------
        t = time.perf_counter()
        try:
            dispatched = dispatch(decision, state.registry, question=state.routing_text)
        except Exception as exc:
            state.add("dispatcher", "error", detail={"error": redact_error(exc)})
            return state.finish(PlanFOutcome.ERROR_REDACTED, "dispatcher raised: %s" % redact_error(exc), specialist=joined)
        state.timings["dispatch_ms"] = (time.perf_counter() - t) * 1000.0
        state.add("dispatcher", "executed:%d,pending:%d" % (len(dispatched.executed_specialists), len(dispatched.pending_specialists)),
                  detail={"executed": [s.value for s in dispatched.executed_specialists],
                          "pending": [s.value for s in dispatched.pending_specialists],
                          "handoffs": len(dispatched.handoffs), "llm_calls": dispatched.llm_calls})
        if dispatched.llm_calls != 0:
            return state.finish(PlanFOutcome.UNSAFE_RESULT, "the dispatch reported %d LLM call(s)" % dispatched.llm_calls,
                                specialist=joined, dispatched=dispatched)
        statuses: List[str] = []
        specialist_ms = 0.0
        for run in dispatched.runs:
            if run.state != STATE_EXECUTED or run.findings is None:
                state.add("specialist:%s" % run.specialist_id.value, "pending", detail={"reason": run.reason[:160]})
                statuses.append("pending")
                continue
            findings = run.findings
            specialist_ms += float(findings.ms)
            statuses.append(findings.status.value)
            state.add("specialist:%s" % findings.specialist_id.value, findings.status.value, bool(findings.llm_used), {
                "task_id": findings.task_id, "findings": len(findings.findings),
                "clauses": len((findings.metadata or {}).get("clauses") or []),
                "handoffs": [h.requested_specialist.value for h in findings.handoff_requests], "missing": len(findings.missing),
                "limitations": len(findings.limitations), "specialist_ms": findings.ms, "summary": (findings.summary or "")[:160],
            })
        state.timings["specialist_ms"] = specialist_ms
        status = "supported" if statuses and all(s == "supported" for s in statuses) else next((s for s in statuses if s != "supported"), None)

        if any(run.findings is not None and run.findings.handoff_requests for run in dispatched.runs):
            return self._handoff_round(state, decision, md, clauses, dispatched, specialist=joined, status=status)
        return self._finish_aggregate(state, decision, md, clauses, dispatched, specialist=joined)

    # ---- the bounded handoff round (phase I-6): every valid request executed once, depth one, then the aggregator --------
    def _handoff_round(self, state: "_Attempt", decision: CoordinatorDecision, md: Dict[str, Any], clauses: List[str],
                       dispatched: DispatchResult, *, specialist: Optional[str], status: Optional[str]) -> AdapterResult:
        from . import plan_f_handoff as _handoff                  # imported here: that module builds on the verifier, as the aggregator does
        state.path = PATH_AGGREGATE
        t = time.perf_counter()
        round_, settled_decision, settled_dispatch = _handoff.run_handoffs(decision, dispatched, state.registry, clauses)
        state.timings["handoff_ms"] = (time.perf_counter() - t) * 1000.0
        state.handoff = round_
        state.add("handoff", round_.code.value, detail=round_.to_dict())
        target_ms = 0.0
        for record in round_.executed:
            target_ms += float(record.ms)
            state.add("specialist:%s" % record.target, record.status or "error", False, {
                "task_id": record.task_id, "handoff": True, "source": record.source, "depth": round_.depth, "request_clause": record.request_clause,
                "findings": record.findings, "second_hop": list(record.second_hop), "specialist_ms": record.ms})
        state.timings["handoff_specialist_ms"] = target_ms
        if round_.code is not _handoff.HandoffCode.HANDOFF_EXECUTED or settled_decision is None or settled_dispatch is None:
            state.handoff_outcome = round_.code.value
            return state.finish(PlanFOutcome.REJECTED_BY_HANDOFF, "%s: %s" % (round_.code.value, round_.detail), specialist=specialist,
                                status=status, dispatched=dispatched)
        joined = ",".join(s.value for s in settled_decision.selected_specialists)
        return self._finish_aggregate(state, settled_decision, md, clauses, settled_dispatch, specialist=joined)

    # ---- the aggregator, the rendering and the unchanged legacy Verifier (phase I-5) -----------------------------------
    def _finish_aggregate(self, state: "_Attempt", decision: CoordinatorDecision, md: Dict[str, Any], clauses: List[str],
                          dispatched: DispatchResult, *, specialist: Optional[str]) -> AdapterResult:
        from . import plan_f_aggregation as aggregation
        joined = specialist
        statuses = [run.findings.status.value if run.findings is not None else "pending" for run in dispatched.runs]
        status = "supported" if statuses and all(s == "supported" for s in statuses) else next((s for s in statuses if s != "supported"), None)
        t = time.perf_counter()
        outcome = aggregation.aggregate(decision, dispatched, self.chunks, self.metadata, guide_index=self.guide_index)
        record = outcome.result
        state.timings["aggregation_ms"] = (time.perf_counter() - t) * 1000.0
        for key, value in record.timings.items():
            state.timings["aggregation_%s" % key] = float(value)
        if state.handoff is not None and getattr(state.handoff, "attempted", False):
            state.handoff_outcome = handoff_outcome(state.handoff, record)
        state.add("aggregator", "accepted" if outcome.accepted else "rejected", detail=record.to_dict())
        if not outcome.accepted:
            return state.finish(PlanFOutcome.REJECTED_BY_AGGREGATOR, "%s: %s" % (record.reason.value, record.detail), specialist=joined,
                                status=status, dispatched=dispatched, aggregation=record)

        # ---- Rendering and the unchanged legacy Verifier ----------------------------------------------------------
        try:
            for part in outcome.parts:                            # post-freeze PF-2: no part may rest on a source conflict
                conflict = source_conflict_problem(part.finding, self.metadata, self.guide_index)
                if conflict is not None:
                    state.add("source_conflict_guard", "rejected", detail={"finding": part.finding.finding_id, "reason": conflict})
                    return state.finish(PlanFOutcome.REJECTED_BY_SOURCE_CONFLICT, "%s: %s" % (part.finding.finding_id, conflict),
                                        specialist=joined, status=status, dispatched=dispatched, aggregation=record)
                authority = guide_evidence_problem(part.clause, part.finding.evidence_quote, part.finding, self.chunks, self.metadata)
                if authority is not None:                         # post-freeze PF-3B (D-110): each clause answered within its source's authority
                    state.add("source_authority_guard", "rejected", detail={"finding": part.finding.finding_id, "reason": authority})
                    return state.finish(PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY, "%s: %s" % (part.finding.finding_id, authority),
                                        specialist=joined, status=status, dispatched=dispatched, aggregation=record)
            items: List[Dict[str, Any]] = []
            for part in outcome.parts:
                items.append(adapt_finding(part.finding, SpecialistId.parse(part.specialist_id), part.clause, self.chunks, self.metadata,
                                           clause_index=part.clause_index))
            answer = aggregation.render_parts(outcome.parts)
            pages = accepted_pages(items)
            if not answer or not pages:
                return state.finish(PlanFOutcome.REQUIRES_FALLBACK, "nothing renderable", specialist=joined, status=status,
                                    dispatched=dispatched, aggregation=record)
            route = (md.get("router") or {}).get("route") or "policy"
            draft = SynthesisResult(answer=answer, claimed_pages=list(pages), prompt_version=PROMPT_VERSION)
            evidence = EvidenceResult(state.text, route, list(items), None, True, None, False, None)
            t = time.perf_counter()
            verdict = verify(state.text, draft, [evidence], [], is_retry=False)
            state.timings["verify_ms"] = (time.perf_counter() - t) * 1000.0
            if not isinstance(verdict, VerifyResult):
                raise TypeError("verify() returned %s" % type(verdict).__name__)
            cited = [int(p) for p in verdict.pages if isinstance(p, int) and not isinstance(p, bool) and p in pages]
            state.add("verifier", "accepted" if verdict.accepted else "rejected", detail={
                "impl": "agents.verifier", "pages": cited, "unsupported_claims": [c[:80] for c in verdict.unsupported_claims[:3]]})
            if not verdict.accepted or not cited:
                return state.finish(PlanFOutcome.REJECTED_BY_VERIFIER,
                                    "verifier %s" % ("rejected the rendered answer" if not verdict.accepted else "cited no accepted page"),
                                    specialist=joined, status=status, dispatched=dispatched, aggregation=record)
            uncited = [p for p in pages if p not in cited]
            if uncited:                                           # every rendered part keeps its page: no collapsed attribution
                return state.finish(PlanFOutcome.REJECTED_BY_VERIFIER,
                                    "verifier did not cite every evidence page (%s)" % ", ".join(str(p) for p in uncited),
                                    specialist=joined, status=status, dispatched=dispatched, aggregation=record)
            intents = {task.task_id: task.intent for task in decision.subtasks}
            sub_questions: List[SubQuestion] = []
            for cov, clause in zip(record.clauses, clauses):
                intent = intents.get(cov.task_id)
                sub_questions.append(SubQuestion(clause, intent if intent in ALL_ROUTES and intent != "greeting" else route))
            items = guide_citation_items(items, self.guide_index)          # post-freeze PF-3D: printed headings only
            first = items[0]["meta"]
            best_section = first.get("section_title") or first.get("section") or "Page %d" % cited[0]
            preview = items[0]["chunk"]
            if len(preview) > EVIDENCE_PREVIEW_CHARS:
                preview = preview[:EVIDENCE_PREVIEW_CHARS].rstrip() + "..."
            state.timings["generation_ms"] = 0.0
            used = "+".join(dict.fromkeys(ref.specialist_id for ref in record.evidence))
            result = QAResult(answer=answer, pages=cited, best_section=str(best_section), evidence=preview, query_type=str(route),
                              items=list(items), used_llm=False, num_candidates=record.findings_considered, num_reranked=len(items),
                              llm_calls=0, retried=False, sub_questions=sub_questions)
            return state.finish(PlanFOutcome.ACCEPTED, "aggregated %d verified finding(s) for %d clause(s) of %s" % (len(items), len(clauses), used),
                                result=result, specialist=joined, status=status, dispatched=dispatched, aggregation=record)
        except Exception as exc:                                  # a defect here must never reach the user as an answer
            state.add("plan_f_adapter", "error", detail={"error": redact_error(exc)})
            return state.finish(PlanFOutcome.ERROR_REDACTED, "adapter raised: %s" % redact_error(exc), specialist=joined, status=status,
                                dispatched=dispatched, aggregation=record)

class _Attempt:
    """The state of one ``attempt``: the question text, the trace, the timings, the path taken and the outcome builder."""

    def __init__(self, question: Any) -> None:
        self.started = time.perf_counter()
        self.trace: List[TraceEntry] = []
        self.timings: Dict[str, float] = {}
        self.text = (question or "").strip() if isinstance(question, str) else ""
        self.routing_text = self.text                             # post-freeze PF-5B (D-114): system names spelled correctly
        self.spelling_spans: List[Any] = []                       # the corrections (typed spans; never logged)
        self.path: Optional[str] = None
        self.routing: Any = None
        self.handoff: Any = None
        self.handoff_outcome: Optional[str] = None
        self.registry: Any = None                                 # post-freeze PF-3B: the registry this question runs on
        self.guide: Optional[Dict[str, Any]] = None              # post-freeze PF-3B: the source selection record
        self.inferred_owner = False                               # post-freeze PF-5C (D-115): the owner was chosen by guide inference

    def ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def add(self, agent: str, decision: str, used_llm: bool = False, detail: Optional[Dict[str, Any]] = None) -> None:
        self.trace.append(TraceEntry(agent, decision, used_llm, self.ms(), dict(detail or {})))

    def finish(self, outcome: PlanFOutcome, reason: str, *, result: Optional[QAResult] = None, specialist: Optional[str] = None,
               status: Optional[str] = None, dispatched: Optional[DispatchResult] = None, verification: Any = None,
               aggregation: Any = None) -> AdapterResult:
        self.trace.append(TraceEntry("plan_f_outcome", outcome.value, False, self.ms(),
                                     {"reason": reason[:200], "requires_fallback": outcome is not PlanFOutcome.ACCEPTED, "path": self.path}))
        self.timings["total_ms"] = self.ms()
        if result is not None:
            result.timings = dict(self.timings)
            result.agent_trace = list(self.trace)
            if self.spelling_spans:
                self._typed_clauses(result)
        return AdapterResult(outcome=outcome, reason=reason, question=self.text, result=result, trace=self.trace, specialist_id=specialist,
                             status=status, timings=dict(self.timings), dispatch=dispatched, verification=verification, aggregation=aggregation,
                             routing=self.routing, handoff=self.handoff, handoff_outcome=self.handoff_outcome,
                             guide_scope=dict(self.guide) if self.guide is not None else None)

    def _typed_clauses(self, result: QAResult) -> None:
        """Post-freeze PF-5B (D-114): the clauses a served answer records (its sub-questions and each item's focus clause) are
        mapped back to the words the user typed; the corrected spelling was for routing and grounding only."""
        from .name_spelling import typed_slice
        result.sub_questions = [SubQuestion(typed_slice(self.text, self.spelling_spans, self.routing_text, s.question), s.route)
                                for s in result.sub_questions]
        for item in result.items:
            plan_f = (item.get("meta") or {}).get("plan_f")
            if isinstance(plan_f, dict) and isinstance(plan_f.get("focus_clause"), str):
                plan_f["focus_clause"] = typed_slice(self.text, self.spelling_spans, self.routing_text, plan_f["focus_clause"])


def build_runtime(kb: Any, *, builders: Optional[Mapping[SpecialistId, Callable[[Any], Specialist]]] = None,
                  guide_scopes: Optional[bool] = None) -> Tuple[Optional[PlanFRuntime], RegistryBuild]:
    """A :class:`PlanFRuntime` on ``kb``, or None with the registry failures. Builds nothing but the specialists.

    Post-freeze PF-3B (D-110): when ``guide_scopes`` is on (``config.PLAN_F_GUIDE_SCOPES`` when None), the default
    builders are used and ``kb`` holds approved guides, a second, guide-enabled registry is built beside the I-9 one
    (``guide_builders``); a question runs on it only when it names a guide's system. A failed guide build leaves the
    runtime on the I-9 registry alone and is reported in ``scope_extension``. Post-freeze PF-3C (D-111): the guide
    registry wraps the SAME I-9 specialist objects behind the D-111 guide path, and the runtime keeps the guide index
    (procedures, registry, approved sets) that the adapter's independent guide check reads."""
    build = build_specialist_registry(kb, builders=builders)
    if not build.ok:
        return None, build
    if guide_scopes is None:
        from . import config as _config
        guide_scopes = bool(getattr(_config, "PLAN_F_GUIDE_SCOPES", False))
    guide_registry: Optional[SpecialistRegistry] = None
    guide_index: Optional[_guides.GuideIndex] = None
    records = tuple(getattr(kb, "sources", None) or ())
    report: Dict[str, Any] = dict(build.scope_extension)
    report["enabled"] = bool(guide_scopes and builders is None and records)
    if report["enabled"]:
        from .source_scopes import source_policy
        try:
            policy = source_policy(records)
            report["policy"] = {sid: sorted(ids) for sid, ids in policy.items()}
            if any(len(ids) > 1 for ids in policy.values()):
                t = time.perf_counter()
                index = _guides.GuideIndex(kb.chunks, kb.metadata, getattr(kb, "source_pages", None) or {}, records, policy)
                evidence = GuideEvidence.from_knowledge_base(kb, index)
                report["guide_index"] = {"procedures": sum(len(p) for p in index.procedures.values()), "skipped_pages": len(index.skipped_pages),
                                         "ms": (time.perf_counter() - t) * 1000.0}
                guide_build = build_specialist_registry(kb, builders=guide_builders(policy, evidence, build.registry))
                report["guide_registry"] = {"ok": guide_build.ok, "errors": dict(guide_build.errors), "ms": guide_build.ms}
                guide_registry = guide_build.registry if guide_build.ok else None
                guide_index = index if guide_registry is not None else None
        except Exception as exc:                                  # a failed guide build never takes the I-9 registry down
            report["guide_registry"] = {"ok": False, "errors": {"guide_registry": redact_error(exc)}, "ms": 0.0}
            guide_registry, guide_index = None, None
    build = RegistryBuild(build.registry, dict(build.errors), build.ms, report)
    try:
        chunks, metadata = kb.chunks, kb.metadata
        runtime = PlanFRuntime(build.registry, chunks, metadata, guide_registry=guide_registry, guide_records=records, guide_index=guide_index)
    except Exception as exc:
        return None, RegistryBuild(None, {"runtime": redact_error(exc)}, build.ms)
    return runtime, build


# ---------------------------------------------------------------------------
# Per-knowledge-base runtime cache (runtime integration phase I-2; owned by the seam ui/pipeline.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheEntry:
    """What the cache holds for one knowledge-base object: the runtime (None when its build failed), the build record
    and whether this call reused an earlier build."""

    runtime: Optional[PlanFRuntime]
    build: RegistryBuild
    reused: bool

    @property
    def ok(self) -> bool:
        return self.runtime is not None


class RuntimeCache:
    """Builds a :class:`PlanFRuntime` once per knowledge-base object and reuses it while that object is alive.

    Entries are keyed by the object's identity and validated through a weak reference, so a different knowledge base
    never inherits another's registry and a collected one releases its specialists. A failed build is remembered for
    that object (runtime None, the redacted errors kept), so the chatbot does not rebuild on every question; another
    object builds afresh. The cache is bounded, thread-safe (a build holds the lock, so concurrent askers wait for one
    build instead of starting their own) and makes no model call. An object that cannot be weakly referenced is built
    but not cached.
    """

    def __init__(self, max_entries: int = 4,
                 builder: Optional[Callable[[Any], Tuple[Optional[PlanFRuntime], RegistryBuild]]] = None) -> None:
        self._entries: Dict[int, Tuple[Any, Optional[PlanFRuntime], RegistryBuild]] = {}
        self._order: List[int] = []
        self._lock = threading.RLock()
        self._max = max(1, int(max_entries))
        self._builder = builder                                    # None: this module's build_runtime, resolved at build time
        self.builds = 0                                            # builds performed, for diagnostics and tests

    def __len__(self) -> int:
        return len(self._entries)

    def _forget(self, key: int) -> None:
        """Drop one entry. Also the weak-reference callback, so it takes no lock."""
        self._entries.pop(key, None)
        try:
            self._order.remove(key)
        except ValueError:
            pass

    def get_or_build(self, kb: Any) -> CacheEntry:
        key = id(kb)
        with self._lock:
            held = self._entries.get(key)
            if held is not None:
                if held[0]() is kb:
                    return CacheEntry(held[1], held[2], True)
                self._forget(key)                                  # the id was reused by another object
            builder = self._builder if self._builder is not None else build_runtime
            try:
                runtime, build = builder(kb)
            except Exception as exc:                               # build_runtime never raises; an injected builder might
                runtime, build = None, RegistryBuild(None, {"runtime": redact_error(exc)}, 0.0)
            self.builds += 1
            try:
                ref = weakref.ref(kb, lambda _dead, key=key: self._forget(key))
            except TypeError:
                return CacheEntry(runtime, build, False)           # not weakly referenceable: built, not cached
            self._entries[key] = (ref, runtime, build)
            self._order.append(key)
            while len(self._order) > self._max:
                oldest = self._order[0] if self._order else None
                if oldest is None:
                    break
                self._forget(oldest)
            return CacheEntry(runtime, build, False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._order.clear()


# ---------------------------------------------------------------------------
# Test-only helper: Plan F first, legacy otherwise (not called by the application)
# ---------------------------------------------------------------------------
def answer_with_fallback(question: str, kb: Any, runtime: Optional[PlanFRuntime]) -> QAResult:
    """Return the Plan F answer when the adapter accepted one, else the legacy ``answer_question`` result with the
    Plan F trace prepended. Used by tests to exercise the fallback contract; the live application does not call it."""
    attempt = runtime.attempt(question) if runtime is not None else None
    if attempt is not None and attempt.outcome is PlanFOutcome.ACCEPTED and attempt.result is not None:
        return attempt.result
    legacy = answer_question(question, embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                             metadata=kb.metadata, pages=kb.pages, groq_client=kb.groq_client)
    if attempt is not None:
        legacy.agent_trace = list(attempt.trace) + list(legacy.agent_trace)
    return legacy


__all__ = [
    "ASK_SPECIFIC_RECOGNISED", "AdapterResult", "CacheEntry", "DEFAULT_BUILDERS", "INSTITUTIONAL_BINDINGS", "INSTITUTIONAL_TIERS",
    "PATH_AGGREGATE", "PATH_PASSTHROUGH", "PHASE", "PROMPT_VERSION", "PlanFOutcome", "PlanFRuntime", "REQUIRED_SPECIALISTS", "RELATION_STRONG",
    "RegistryBuild", "RuntimeCache", "SUBJECT_CHECK_VERSION", "accepted_pages", "accepted_record", "adapt_finding", "answer_with_fallback",
    "build_runtime", "handoff_outcome",
    "build_specialist_registry", "clause_covered_by", "decision_problem", "entity_in_text", "missing_specialists",
    "named_entities", "quote_problem", "render_answer", "result_problem", "source_conflict_problem", "guide_evidence_problem", "guide_builders", "subject_binding",
    "guide_citation_items",
]
