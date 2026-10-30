"""
Plan F runtime adapter (runtime integration, phase I-1).

DECISION OWNED
    None. This module is a thin, deterministic seam between the Plan F path
    (``coordinator.coordinate`` -> ``dispatch.dispatch`` -> the four frozen
    specialists) and the legacy result contract (``QAResult``, checked by the
    unchanged ``agents.verifier.verify``). It never routes, never retrieves,
    never answers from anything but a specialist's own supported findings,
    never calls a language model and never executes a handoff.

WIRING (phases I-2 and I-4)
    The one approved seam is ``ui/pipeline.py`` (phase I-2): it builds a
    :class:`PlanFRuntime` lazily through :class:`RuntimeCache` and calls
    ``attempt`` only in SHADOW or ACTIVE mode; both switches default to off
    and LEGACY mode never imports this module. The orchestrator, the Router,
    retrieval, Synthesis and the legacy Verifier are unchanged. Since phase
    I-4 a finding passes the findings-aware verification layer
    (``plan_f_verification.verify_findings``) after the I-1 guards and
    before rendering; its rejection is ``PLAN_F_REJECTED_BY_FINDINGS_VERIFIER``.

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
from .agents.types import EvidenceResult, SynthesisResult, TraceEntry, VerifyResult
from .agents.verifier import verify
from .config import EVIDENCE_PREVIEW_CHARS
from .qa import QAResult, answer_question

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


class PlanFOutcome(str, Enum):
    """Why the adapter did or did not produce a Plan F answer."""

    ACCEPTED = "PLAN_F_ACCEPTED"
    NOT_APPLICABLE = "PLAN_F_NOT_APPLICABLE"                    # greeting, empty or no specialist selected
    REGISTRY_UNAVAILABLE = "PLAN_F_REGISTRY_UNAVAILABLE"        # no complete registry
    UNSAFE_RESULT = "PLAN_F_UNSAFE_RESULT"                      # decision or specialist result outside the I-1 safe case
    REJECTED_BY_QUOTE_GUARD = "PLAN_F_REJECTED_BY_QUOTE_GUARD"
    REJECTED_BY_SUBJECT_GUARD = "PLAN_F_REJECTED_BY_SUBJECT_GUARD"
    REJECTED_BY_FINDINGS_VERIFIER = "PLAN_F_REJECTED_BY_FINDINGS_VERIFIER"  # phase I-4: the structured finding is not a suitable direct answer
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

    @property
    def ok(self) -> bool:
        return self.registry is not None and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "specialists": [s.value for s in self.registry.ids()] if self.registry is not None else [],
                "errors": dict(self.errors), "ms": self.ms}


def build_specialist_registry(kb: Any, *, builders: Optional[Mapping[SpecialistId, Callable[[Any], Specialist]]] = None) -> RegistryBuild:
    """Build the four specialists from ``kb`` with their own ``from_knowledge_base`` builders.

    Either every required specialist is built and registered, or no registry is returned and ``errors`` names each
    specialist that could not be built, with its reason redacted. No model is called; nothing is cached.
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
    ms = (time.perf_counter() - started) * 1000.0
    if errors:
        return RegistryBuild(None, errors, ms)
    return RegistryBuild(registry, {}, ms)


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

    @property
    def requires_fallback(self) -> bool:
        return self.outcome is not PlanFOutcome.ACCEPTED

    @property
    def verified(self) -> bool:
        """Whether the findings-aware verification layer accepted the result for rendering (phase I-4)."""
        v = self.verification
        return v is not None and bool(getattr(v, "accepted", False)) and bool(getattr(v, "safe_to_render", False))

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
        }


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


# ---------------------------------------------------------------------------
# Finding -> legacy evidence item
# ---------------------------------------------------------------------------
def adapt_finding(finding: Finding, specialist_id: SpecialistId, focus_clause: str, chunks: Sequence[str],
                  metadata: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The legacy evidence item for one finding that passed :func:`quote_problem`.

    ``chunk`` is the finding's own quote (the exact evidence the answer rests on), ``meta`` is a copy of the source
    chunk's metadata with the finding's provenance under ``meta["plan_f"]``. No rerank score is invented: the
    Verifier reads ``rerank_score`` with a default and only orders pages by it.
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
    for key in ("evidence_tier", "relation", "office_id", "office_name", "contact_values", "details_covered"):
        if key in finding.metadata:
            plan_f[key] = finding.metadata[key]
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
    """The accepted quotes, verbatim, one per paragraph. Nothing is added or reworded."""
    return "\n\n".join(str(item["chunk"]).strip() for item in items if str(item.get("chunk") or "").strip())


def accepted_pages(items: Sequence[Dict[str, Any]]) -> List[int]:
    pages: List[int] = []
    for item in items:
        page = item["meta"].get("page")
        if isinstance(page, int) and not isinstance(page, bool) and page not in pages:
            pages.append(page)
    return pages


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
class PlanFRuntime:
    """A complete specialist registry together with the corpus it was built on. Created explicitly by the caller
    (``build_runtime``); there is no module-level instance."""

    def __init__(self, registry: SpecialistRegistry, chunks: Sequence[str], metadata: Sequence[Dict[str, Any]]) -> None:
        if not isinstance(registry, SpecialistRegistry):
            raise TypeError("PlanFRuntime expects a SpecialistRegistry, got %s" % type(registry).__name__)
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata differ in length (%d vs %d)" % (len(chunks), len(metadata)))
        self.registry = registry
        self.chunks = chunks
        self.metadata = metadata

    @property
    def missing_specialists(self) -> List[str]:
        return missing_specialists(self.registry)

    def attempt(self, question: str) -> AdapterResult:
        """Try to answer ``question`` through Plan F under the I-1 safe case. Never raises."""
        started = time.perf_counter()
        trace: List[TraceEntry] = []
        timings: Dict[str, float] = {}
        text = (question or "").strip() if isinstance(question, str) else ""

        def ms() -> float:
            return (time.perf_counter() - started) * 1000.0

        def finish(outcome: PlanFOutcome, reason: str, *, result: Optional[QAResult] = None, specialist: Optional[str] = None,
                   status: Optional[str] = None, dispatched: Optional[DispatchResult] = None, verification: Any = None) -> AdapterResult:
            trace.append(TraceEntry("plan_f_outcome", outcome.value, False, ms(),
                                    {"reason": reason[:200], "requires_fallback": outcome is not PlanFOutcome.ACCEPTED}))
            timings["total_ms"] = ms()
            if result is not None:
                result.timings = dict(timings)
                result.agent_trace = list(trace)
            return AdapterResult(outcome=outcome, reason=reason, question=text, result=result, trace=trace,
                                 specialist_id=specialist, status=status, timings=dict(timings), dispatch=dispatched, verification=verification)

        trace.append(TraceEntry("plan_f_adapter", "start", False, 0.0, {"phase": PHASE, "question_chars": len(text)}))
        missing = self.missing_specialists
        if missing:
            trace.append(TraceEntry("plan_f_registry", "incomplete", False, ms(), {"missing": missing}))
            return finish(PlanFOutcome.REGISTRY_UNAVAILABLE, "registry lacks: %s" % ", ".join(missing))
        trace.append(TraceEntry("plan_f_registry", "ok", False, ms(), {"specialists": [s.value for s in self.registry.ids()]}))
        if not text:
            trace.append(TraceEntry("coordinator", "skipped:empty", False, ms(), {}))
            return finish(PlanFOutcome.NOT_APPLICABLE, "empty question")

        # ---- Coordinator ------------------------------------------------------
        t = time.perf_counter()
        try:
            decision = coordinate(text)
        except Exception as exc:
            trace.append(TraceEntry("coordinator", "error", False, ms(), {"error": redact_error(exc)}))
            return finish(PlanFOutcome.ERROR_REDACTED, "coordinator raised: %s" % redact_error(exc))
        timings["coordinator_ms"] = (time.perf_counter() - t) * 1000.0
        md = decision.metadata or {}
        trace.append(TraceEntry("coordinator", ",".join(s.value for s in decision.selected_specialists) or "none", False, ms(), {
            "selected": [s.value for s in decision.selected_specialists],
            "complexity": md.get("complexity"),
            "requires_synthesis": decision.requires_synthesis,
            "is_multipart": bool((md.get("router") or {}).get("is_multipart")),
            "route": (md.get("router") or {}).get("route"),
            "uncovered_clauses": len(md.get("uncovered_clauses") or []),
            "dropped_domains": list(md.get("dropped_domains") or []),
            "confidence": decision.confidence,
            "reason": (decision.reason or "")[:120],
        }))
        problem = decision_problem(decision)
        if problem is not None:
            if not decision.selected_specialists or md.get("complexity") == "greeting":
                return finish(PlanFOutcome.NOT_APPLICABLE, problem)
            return finish(PlanFOutcome.UNSAFE_RESULT, problem)
        task = decision.subtasks[0]
        clause = task.context["focus_clauses"][0]
        sid = task.specialist_id

        # ---- Dispatch ------------------------------------------------------------
        t = time.perf_counter()
        try:
            dispatched = dispatch(decision, self.registry, question=text)
        except Exception as exc:
            trace.append(TraceEntry("dispatcher", "error", False, ms(), {"error": redact_error(exc)}))
            return finish(PlanFOutcome.ERROR_REDACTED, "dispatcher raised: %s" % redact_error(exc), specialist=sid.value)
        timings["dispatch_ms"] = (time.perf_counter() - t) * 1000.0
        trace.append(TraceEntry("dispatcher", "executed:%d,pending:%d" % (len(dispatched.executed_specialists), len(dispatched.pending_specialists)),
                                False, ms(), {"executed": [s.value for s in dispatched.executed_specialists],
                                              "pending": [s.value for s in dispatched.pending_specialists],
                                              "handoffs": len(dispatched.handoffs), "llm_calls": dispatched.llm_calls}))
        if dispatched.llm_calls != 0:
            return finish(PlanFOutcome.UNSAFE_RESULT, "the dispatch reported %d LLM call(s)" % dispatched.llm_calls,
                          specialist=sid.value, dispatched=dispatched)
        if len(dispatched.runs) != 1:
            return finish(PlanFOutcome.UNSAFE_RESULT, "expected one run, the dispatcher produced %d" % len(dispatched.runs),
                          specialist=sid.value, dispatched=dispatched)
        run = dispatched.runs[0]
        if run.state != STATE_EXECUTED or run.findings is None:
            trace.append(TraceEntry("specialist:%s" % run.specialist_id.value, "pending", False, ms(), {"reason": run.reason[:160]}))
            return finish(PlanFOutcome.REQUIRES_FALLBACK, "task pending: %s" % run.reason, specialist=run.specialist_id.value,
                          dispatched=dispatched)
        findings = run.findings
        sid = findings.specialist_id
        status = findings.status.value
        timings["specialist_ms"] = float(findings.ms)
        trace.append(TraceEntry("specialist:%s" % sid.value, status, bool(findings.llm_used), ms(), {
            "findings": len(findings.findings), "handoffs": [h.requested_specialist.value for h in findings.handoff_requests],
            "missing": len(findings.missing), "limitations": len(findings.limitations), "specialist_ms": findings.ms,
            "summary": (findings.summary or "")[:160],
        }))
        problem = result_problem(findings)
        if problem is not None:
            decision_label = "HANDOFF_PENDING" if findings.handoff_requests else "not_eligible"
            trace.append(TraceEntry("plan_f_gate", decision_label, False, ms(), {"reason": problem[:200]}))
            return finish(PlanFOutcome.UNSAFE_RESULT, problem, specialist=sid.value, status=status, dispatched=dispatched)

        # ---- Guards, rendering, verification -----------------------------------------
        try:
            t = time.perf_counter()
            for finding in findings.findings:
                problem = quote_problem(finding, self.chunks, self.metadata)
                if problem is not None:
                    trace.append(TraceEntry("quote_guard", "rejected", False, ms(), {"finding": finding.finding_id, "reason": problem}))
                    return finish(PlanFOutcome.REJECTED_BY_QUOTE_GUARD, "%s: %s" % (finding.finding_id, problem),
                                  specialist=sid.value, status=status, dispatched=dispatched)
            trace.append(TraceEntry("quote_guard", "passed", False, ms(), {"findings": len(findings.findings)}))
            report = findings.metadata["clauses"][0]
            accepted: List[Dict[str, Any]] = []
            dropped: List[Dict[str, Any]] = []
            for n, finding in enumerate(findings.findings):
                item = adapt_finding(finding, sid, clause, self.chunks, self.metadata)
                bound, why = subject_binding(clause, report, finding, item, sid)
                if bound:
                    accepted.append(item)
                elif n == 0:
                    trace.append(TraceEntry("subject_guard", "rejected", False, ms(), {"finding": finding.finding_id, "reason": why}))
                    return finish(PlanFOutcome.REJECTED_BY_SUBJECT_GUARD, "%s: %s" % (finding.finding_id, why),
                                  specialist=sid.value, status=status, dispatched=dispatched)
                else:
                    dropped.append({"finding": finding.finding_id, "reason": why})
            timings["guards_ms"] = (time.perf_counter() - t) * 1000.0
            trace.append(TraceEntry("subject_guard", "passed", False, ms(),
                                    {"accepted": [i["meta"]["plan_f"]["finding_id"] for i in accepted], "dropped": dropped}))
            if not accepted:
                return finish(PlanFOutcome.REQUIRES_FALLBACK, "no finding left to render", specialist=sid.value, status=status,
                              dispatched=dispatched)

            # ---- Findings-aware verification (phase I-4): the structured result must suit the exact request ----------
            from . import plan_f_verification as _verification      # imported here: that module builds on this one's guards
            t = time.perf_counter()
            verification = _verification.verify_findings(clause, findings, self.chunks, self.metadata)
            timings["findings_verify_ms"] = (time.perf_counter() - t) * 1000.0
            trace.append(TraceEntry("findings_verifier", "accepted" if verification.accepted else "rejected", False, ms(), verification.to_dict()))
            if not verification.accepted or not verification.safe_to_render:
                return finish(PlanFOutcome.REJECTED_BY_FINDINGS_VERIFIER, "%s: %s" % (verification.reason.value, verification.detail),
                              specialist=sid.value, status=status, dispatched=dispatched, verification=verification)
            keep = set(verification.accepted_finding_ids)
            accepted = [item for item in accepted if item["meta"]["plan_f"]["finding_id"] in keep]
            if not accepted:
                return finish(PlanFOutcome.REQUIRES_FALLBACK, "no verified finding left to render", specialist=sid.value, status=status,
                              dispatched=dispatched, verification=verification)

            answer = render_answer(accepted)
            pages = accepted_pages(accepted)
            if not answer or not pages:
                return finish(PlanFOutcome.REQUIRES_FALLBACK, "nothing renderable", specialist=sid.value, status=status,
                              dispatched=dispatched)
            route = (md.get("router") or {}).get("route") or "policy"
            draft = SynthesisResult(answer=answer, claimed_pages=list(pages), prompt_version=PROMPT_VERSION)
            evidence = EvidenceResult(clause, route, list(accepted), None, True, None, False, None)
            t = time.perf_counter()
            verdict = verify(text, draft, [evidence], [], is_retry=False)
            timings["verify_ms"] = (time.perf_counter() - t) * 1000.0
            if not isinstance(verdict, VerifyResult):
                raise TypeError("verify() returned %s" % type(verdict).__name__)
            cited = [int(p) for p in verdict.pages if isinstance(p, int) and not isinstance(p, bool) and p in pages]
            trace.append(TraceEntry("verifier", "accepted" if verdict.accepted else "rejected", False, ms(), {
                "impl": "agents.verifier", "pages": cited, "unsupported_claims": [c[:80] for c in verdict.unsupported_claims[:3]]}))
            if not verdict.accepted or not cited:
                return finish(PlanFOutcome.REJECTED_BY_VERIFIER,
                              "verifier %s" % ("rejected the rendered answer" if not verdict.accepted else "cited no accepted page"),
                              specialist=sid.value, status=status, dispatched=dispatched, verification=verification)

            first = accepted[0]["meta"]
            best_section = first.get("section_title") or first.get("section") or "Page %d" % cited[0]
            preview = accepted[0]["chunk"]
            if len(preview) > EVIDENCE_PREVIEW_CHARS:
                preview = preview[:EVIDENCE_PREVIEW_CHARS].rstrip() + "..."
            timings["generation_ms"] = 0.0
            result = QAResult(answer=answer, pages=cited, best_section=str(best_section), evidence=preview, query_type=str(route),
                              items=list(accepted), used_llm=False, num_candidates=len(findings.findings), num_reranked=len(accepted),
                              llm_calls=0, retried=False, sub_questions=[])
            return finish(PlanFOutcome.ACCEPTED, "rendered %d supported finding(s) of %s" % (len(accepted), sid.value),
                          result=result, specialist=sid.value, status=status, dispatched=dispatched, verification=verification)
        except Exception as exc:                                  # a defect here must never reach the user as an answer
            trace.append(TraceEntry("plan_f_adapter", "error", False, ms(), {"error": redact_error(exc)}))
            return finish(PlanFOutcome.ERROR_REDACTED, "adapter raised: %s" % redact_error(exc), specialist=sid.value, status=status,
                          dispatched=dispatched)


def build_runtime(kb: Any, *, builders: Optional[Mapping[SpecialistId, Callable[[Any], Specialist]]] = None
                  ) -> Tuple[Optional[PlanFRuntime], RegistryBuild]:
    """A :class:`PlanFRuntime` on ``kb``, or None with the registry failures. Builds nothing but the four specialists."""
    build = build_specialist_registry(kb, builders=builders)
    if not build.ok:
        return None, build
    try:
        chunks, metadata = kb.chunks, kb.metadata
        runtime = PlanFRuntime(build.registry, chunks, metadata)
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
    "PHASE", "PROMPT_VERSION", "PlanFOutcome", "PlanFRuntime", "REQUIRED_SPECIALISTS", "RELATION_STRONG", "RegistryBuild",
    "RuntimeCache", "SUBJECT_CHECK_VERSION", "accepted_pages", "adapt_finding", "answer_with_fallback", "build_runtime",
    "build_specialist_registry", "clause_covered_by", "decision_problem", "entity_in_text", "missing_specialists",
    "named_entities", "quote_problem", "render_answer", "result_problem", "subject_binding",
]
