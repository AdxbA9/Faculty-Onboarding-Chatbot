"""
Coordinator (Plan F): decides WHO should work on a faculty question.

DECISION OWNED
    Which domain specialists a question needs, which organizational level
    and university systems it refers to, whether it is simple, multipart,
    cross-domain or an orientation journey, and how it is cut into
    specialist tasks. Nothing else: the Coordinator never answers the
    question, never retrieves evidence, never calls a specialist and never
    verifies anything. It returns a :class:`CoordinatorDecision`; the
    deterministic orchestrator (later phase) acts on it.

HOW IT DECIDES (deterministic, no LLM call)
    1. Technical intent comes from the existing Router, called with its LLM
       arbitration switched off (``route_with_report(..., llm_fallback=False)``).
       The Router's route, multipart flag and scores are reused, not
       re-implemented.
    2. Every domain is scored from regular-expression cues (routing
       vocabulary only, never policy content). A domain is a candidate when
       its score reaches ``SELECT_THRESHOLD``: one strong cue or two hints.
       Hyphen-like separators between words are read as spaces for cue
       matching only ("office-hours", "add/drop", "e-learning"); the question
       and its clauses are stored verbatim. Research cues score a view of the
       text in which the WLAM category phrases ("research-intensive faculty",
       "active research faculty") are blanked, so a teaching-load question
       about a faculty category is not a research question; every other
       domain scores the normal text, and the category phrase is itself a
       Teaching cue.
    3. Level (department > college > university) and systems (blackboard,
       banner, myuos) are detected from their own cue lists. A system is
       never a specialist.
    4. Candidates are ordered by score, then by where their first cue appears,
       then by canonical order, and capped at ``PLAN_F_MAX_SPECIALISTS``.
       Contact questions are conditional: a direct channel lookup (phone,
       fax, e-mail address, extension, "number of the ...") is the
       Institutional specialist's alone; "who do I contact about <subject>"
       keeps the subject specialist(s) and adds Institutional. A question
       that only names Blackboard or Banner goes to Teaching, which owns those
       guides; a bare MyUOS mention selects nobody (MyUOS is a general portal).
    5. One task per selected specialist, capped at ``PLAN_F_MAX_SUBTASKS``.
       Every task's ``question`` is the complete original question. The
       specialist's own clauses (split with the Router's separator, so
       multipart detection and decomposition agree) travel separately in
       ``context["focus_clauses"]``; a clause may be shared; a specialist with
       no clause of its own has the whole question as its focus.
    6. ``requires_synthesis`` is True when two or more specialists are
       selected, or when one specialist receives a multipart question. The
       journey classification alone never forces Synthesis.

WHY THERE IS NO LLM ARBITRATION HERE
    The Router must pick exactly one route, so a tie needs a tie-breaker. The
    Coordinator may select several specialists when domains tie, and the
    orchestrator, Synthesis and the Verifier reconcile their findings. A
    structured arbitration call would only save the cost of an extra
    specialist run; that trade-off is measured at the Teaching Agent gate
    (see docs/DECISIONS.md, D-PLANF-018).

FAILURE POLICY
    A question with no domain cue gets no specialist, confidence 0.0 and
    complexity ``unknown``; the future orchestrator falls back to the
    Milestone 1 handbook path. A greeting gets no specialist and intent
    ``greeting``. Nonsense is never turned into a confident domain decision.

This module is not imported by the production pipeline. ``PLAN_F_ENABLED``
stays False; wiring the Coordinator into the orchestrator is a later phase.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import PLAN_F_MAX_SPECIALISTS, PLAN_F_MAX_SUBTASKS
from . import router as router_agent
from .specialists.contracts import CoordinatorDecision, SpecialistId, SpecialistTask
from .specialists.domain_cues import (DEVELOPMENT_FUNDING, HR_CONTEXT_PATTERNS, HR_OWNERSHIP_PATTERNS,  # noqa: F401
                                      RESEARCH_OWNERSHIP_PATTERNS, WLAM_CATEGORY, research_view)

#: A domain becomes a candidate at this score: one strong cue, or two hints.
SELECT_THRESHOLD = 1.0
#: Weights. STRONG: the word alone names the domain. HINT: needs company.
STRONG = 1.0
HINT = 0.5

#: Complexity values carried in ``CoordinatorDecision.metadata["complexity"]``.
COMPLEXITIES = ("greeting", "unknown", "simple", "multipart", "cross_domain", "journey")

#: Organizational levels, most specific first.
LEVELS = ("department", "college", "university")
#: University systems the Coordinator recognises. Systems are never specialists.
SYSTEMS = ("blackboard", "banner", "myuos")

#: Value of ``SpecialistTask.requested_by`` for every task the Coordinator creates.
REQUESTED_BY = "coordinator"


# ---------------------------------------------------------------------------
# Cue lexicons (routing vocabulary only; no policy content)
# ---------------------------------------------------------------------------
def _rx(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern, re.I)


#: Hyphen-like separators between word characters ("office-hours",
#: "peer-observation", "add/drop", "e-learning") are read as spaces before
#: cue matching. Each separator is one character, so cue positions still
#: refer to the original text. Classification only: nothing stored changes.
_CUE_SEPARATORS = re.compile(r"(?<=\w)[-‐‑‒–—/](?=\w)")


def cue_text(text: str) -> str:
    """``text`` as the cue lexicons read it: hyphen-like separators between
    words replaced by spaces, same length as the original."""
    return _CUE_SEPARATORS.sub(" ", text)


#: WLAM faculty categories and development funding are blanked from the research
#: view (see specialists/domain_cues.py, shared with the specialists).
_WLAM_CATEGORY = WLAM_CATEGORY
_DEVELOPMENT_FUNDING = DEVELOPMENT_FUNDING


def _research_view(text: str) -> str:
    return research_view(text)


#: (domain, weight, pattern). Each pattern counts once per question.
_DOMAIN_CUES: Tuple[Tuple[SpecialistId, float, "re.Pattern[str]"], ...] = (
    # ---- teaching & learning ---------------------------------------------
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:teach(?:ing|es|er)?|taught|instructor)\b")),
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:teaching|instructional)\s+load\b|\bworkload\b|\bwlam\b")),
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:courses?|lectures?|classes|class(?:room)?|syllab(?:us|i)|curricul(?:um|a))\b")),
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:blackboard|bb\s+ultra|lms|learning management|gradebook|grade ?center|safeassign|collaborate|e[- ]?learning|elearning)\b")),
    (SpecialistId.TEACHING, STRONG, _rx(r"\badd[ /-]?drop\b|\badd\s+(?:and|or)\s+drop\b")),         # the academic calendar's add/drop period
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:grades?|grading|marks?|attendance|assignments?|quiz(?:zes)?|exams?|examinations?|midterms?|rubrics?|assessments?)\b")),
    (SpecialistId.TEACHING, STRONG, _rx(r"\boffice hours?\b|\bcredit hours?\b|\bcontact hours?\b")),
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:research[- ]intensive|active[- ]research|research[- ]active|teaching[- ]track)\s+faculty\b")),   # WLAM categories
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:thesis|dissertation)\s+supervis|\bsupervis(?:e|ing|ion)\s+(?:a\s+|the\s+)?(?:thesis|dissertation|students?|theses)\b")),
    (SpecialistId.TEACHING, STRONG, _rx(r"\b(?:professional development|faculty development|pedagog(?:y|ical)|peer observation|mentor(?:s|ing|ship)?|teaching (?:award|excellence|workshop))\b")),
    (SpecialistId.TEACHING, HINT, _rx(r"\b(?:students?|semester|lesson|academic integrity|plagiarism|turnitin|training|workshops?)\b")),
    (SpecialistId.TEACHING, HINT, _rx(r"\bbanner\b")),                       # Banner is mostly grade entry and class lists
    # ---- research & innovation -------------------------------------------
    # Research ownership terms live in specialists/domain_cues.py (Step 4.5A-R), shared with the Research and
    # Teaching specialists so that the three never drift apart
    *((SpecialistId.RESEARCH, STRONG, _rx(p)) for p in RESEARCH_OWNERSHIP_PATTERNS),
    (SpecialistId.RESEARCH, HINT, _rx(r"\b(?:conferences?|innovation|laborator(?:y|ies)|labs?)\b")),
    # ---- faculty services & hr -------------------------------------------
    # HR ownership terms live in specialists/domain_cues.py (Step 4.11, D-085), shared with the Faculty Services & HR
    # specialist so that a question routed here is never rejected there; context terms count only beside another cue
    *((SpecialistId.FACULTY_SERVICES, STRONG, _rx(p)) for p in HR_OWNERSHIP_PATTERNS),
    *((SpecialistId.FACULTY_SERVICES, HINT, _rx(p)) for p in HR_CONTEXT_PATTERNS),
    # ---- institutional navigation ----------------------------------------
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\bwho\b.{0,40}\b(?:approv|authori[sz]|sign(?:s|ed)? off|decid|responsib|in charge|handles?|oversee|report(?:s)? to)")),
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\b(?:who|whom)\b.{0,30}\b(?:contact|ask|talk to|speak to|reach|see|go to|refer)\b")),
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\b(?:which|what)\s+(?:office|unit|department|committee|council|body)\b.{0,30}\b(?:handles?|responsib|approv|deals?|manages?|in charge|do i)")),
    # "who manages / supports / maintains ... <support, help desk, service, system, office, unit>":
    # the object must be a service or unit, so "who manages course grading" stays with the subject domain.
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\bwho\s+(?:manages|supports|maintains|administers|runs|provides)\b.{0,40}\b(?:support|help ?desk|services?|systems?|portal|blackboard|banner|my ?uos|lms|office|unit|department|cent(?:er|re)|desk)\b")),
    # "where is / where can I find <office, unit, named service>"; "office hours" is a teaching phrase, not a place.
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\bwhere\s+(?:do|can|should|must)\s+i\s+(?:go|find|get|submit|apply|report|collect|pick up|obtain|register|hand in)\b|\bwhere\s+(?:is|are)\b.{0,30}\b(?:office(?!\s+hours?)|located|building|department|unit|cent(?:er|re)|desk|library|clinic|hospital|registrar\w*|registration (?:office|department|desk)|admissions? (?:office|department)|finance (?:office|department)|hr|human resources|it (?:services?|support)|reception|security|bookstore|cafeteria|parking)\b")),
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\b(?:phone|telephone|fax|e[- ]?mail address|extension|contact (?:number|details|information)|number (?:of|for) the)\b")),
    # Step 4.5A: "where is X located", "how do I contact / reach X", "who is the contact person", floors, buildings,
    # addresses, finding or getting to a place, opening hours
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\bwhere\b[^.?]{0,80}\blocated\b|\blocations? of the (?:office|department|unit|cent(?:er|re)|institute|building|tto|irb|rfd)\b|"
                                             r"\bhow (?:do|does|can|could|should) (?:i|we|one) (?:contact|reach|get in touch with|e ?mail|call)\b|\bcontact (?:person|persons|people|point)\b|"
                                             r"\bpoint of contact\b|\b(?:someone|somebody|anyone) (?:i|we) can (?:contact|ask)\b|\b(?:what|which) (?:floor|building|room)\b|\bon which floor\b|"
                                             r"\b(?:mailing|postal|office|physical|street) address\b|\baddress of the\b|"
                                             r"\bhow (?:do|can|could|should) (?:i|we|one) (?:find|get to|go to|locate|visit) (?:the |a |an )?[\w' ]{0,40}?"
                                             r"(?:office|department|unit|cent(?:er|re)|building|institute|desk|room|committee|board|lab(?:oratory)?|tto|rfd|irb)\b|"
                                             r"\b(?:office|department|unit|desk|library|tto|rfd) (?:is )?open (?:during|on|at|until)\b|\bopening hours\b")),
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\b(?:approval (?:chain|process|authority)|chain of command|organi[sz]ational (?:chart|structure)|governance|structure of the university|university structure)\b")),
    (SpecialistId.INSTITUTIONAL, STRONG, _rx(r"\b(?:colleges|departments|campuses|degree programs?|programs? (?:are |is )?offered|accreditation|ranking|mission|vision|core values|board of trustees|university council|college council|department council|standing committees?)\b")),
    (SpecialistId.INSTITUTIONAL, HINT, _rx(r"\b(?:dean|deanship|chair|chancellor|vice[- ]chancellor|director|head of)\b")),
    (SpecialistId.INSTITUTIONAL, HINT, _rx(r"\b(?:department|college|university|office|unit)\b")),
)

_LEVEL_CUES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("department", _rx(r"\b(?:departments?|dept\.?|department chair|head of department|chair(?:person)?)\b")),
    ("college", _rx(r"\b(?:colleges?|deans?|deanship|college council)\b")),
    ("university", _rx(r"\b(?:universit(?:y|ies)|uos|chancellor|vice[- ]chancellor|university council|board of trustees|institution(?:al)?)\b")),
)

_SYSTEM_CUES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("blackboard", _rx(r"\b(?:blackboard|bb\s+ultra|blackboard ultra|safeassign|blackboard collaborate)\b")),
    ("banner", _rx(r"\bbanner\b|\bfaculty self[- ]service\b|\bself[- ]service banner\b|\bmy ?udc\b")),
    ("myuos", _rx(r"\bmy ?uos\b|\buos portal\b|\bemployee portal\b|\bmyuos portal\b")),
)

_JOURNEY_CUES = _rx(
    r"\b(?:first (?:week|day|days|month|semester)|new(?:ly)? (?:hired|appointed|joined|faculty member|faculty)|"
    r"just joined|get(?:ting)? started|onboarding|orientation|induction|where do i start|"
    r"set up first|checklist|before (?:i )?arriv(?:e|ing|al)|when i arrive|settle in)\b"
)

#: Contact route: a direct channel lookup goes to Institutional Navigation
#: alone; a contact request about a subject keeps the subject specialist(s).
_CONTACT_ROUTE = "contact"
_CHANNEL_CUES = _rx(
    r"\b(?:phone|telephone|fax|e-?mail(?:\s+address)?|extension|ext\.|"
    r"contact (?:number|details|information)|numbers?\s+(?:of|for)\s+the)\b"
)

#: A question that only names one of these systems, with no other domain cue,
#: goes to the specialist that owns that system's guides. MyUOS is deliberately
#: absent: it is the general university portal (HR, academic, research and
#: student services alike), so a bare mention implies no domain.
_SYSTEM_OWNER: Dict[str, SpecialistId] = {
    "blackboard": SpecialistId.TEACHING,
    "banner": SpecialistId.TEACHING,
}


# ---------------------------------------------------------------------------
# Analysis (the evidence behind a decision)
# ---------------------------------------------------------------------------
@dataclass
class CoordinatorAnalysis:
    """Everything the Coordinator measured before deciding. Kept for the trace
    and for tests; :func:`coordinate` builds the decision from it."""

    question: str
    route: str
    routes: List[str]
    is_multipart: bool
    domain_scores: Dict[str, float]
    domain_cues: Dict[str, List[str]]
    first_cue: Dict[str, int]
    level: Optional[str]
    systems: List[str]
    journey: bool
    greeting: bool
    channel_lookup: bool = False           # phone / fax / e-mail / extension asked for directly
    router_llm_attempted: bool = False
    clauses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "route": self.route,
            "routes": list(self.routes),
            "is_multipart": self.is_multipart,
            "domain_scores": dict(self.domain_scores),
            "domain_cues": {k: list(v) for k, v in self.domain_cues.items()},
            "level": self.level,
            "systems": list(self.systems),
            "journey": self.journey,
            "greeting": self.greeting,
            "channel_lookup": self.channel_lookup,
            "router_llm_attempted": self.router_llm_attempted,
            "clauses": list(self.clauses),
        }


def _score_domains(text: str) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, int]]:
    text = cue_text(text)                     # classification only; positions are unchanged
    research_text = _research_view(text)      # WLAM category phrases blanked, same length
    scores: Dict[str, float] = {}
    cues: Dict[str, List[str]] = {}
    first: Dict[str, int] = {}
    for domain, weight, pattern in _DOMAIN_CUES:
        m = pattern.search(research_text if domain is SpecialistId.RESEARCH else text)
        if not m:
            continue
        key = domain.value
        scores[key] = round(scores.get(key, 0.0) + weight, 2)
        cues.setdefault(key, []).append(m.group(0).strip().lower())
        first[key] = min(first.get(key, m.start()), m.start())
    return scores, cues, first


def cue_scores(text: str) -> Dict[str, float]:
    """The domain cue scores of ``text`` (the same lexicons :func:`coordinate` uses to select and assign clauses).
    Read-only; added for the runtime routing refinement (phase I-6), which applies the Coordinator's own clause
    assignment rule to a single specialist's multipart task."""
    return _score_domains(text)[0]


def _detect_level(text: str) -> Optional[str]:
    """Most specific level mentioned: department, then college, then university."""
    for level, pattern in _LEVEL_CUES:
        if pattern.search(text):
            return level
    return None


def _detect_systems(text: str) -> List[str]:
    found = []
    for system, pattern in _SYSTEM_CUES:
        m = pattern.search(text)
        if m:
            found.append((m.start(), system))
    return [system for _, system in sorted(found)]


#: A piece that opens a question or a request of its own ("... and how many
#: programs are offered", "..., and who approves it"). A piece that does not
#: ("... and conference travel claims", "... local and international
#: conferences") continues the previous clause.
_CLAUSE_HEAD = re.compile(
    r"^(?:what|who|whom|whose|which|how|when|where|why|is|are|was|were|can|could|do|does|did|may|might|must|should|shall|will|would|has|have|"
    r"i|we|my|our|your|tell|explain|describe|list|give|show|please|also)\b", re.I)
_HARD_SEPARATOR = re.compile(r"[?;]|\.\s")


_GREETING_PIECE = re.compile(r"^(?:(?:hi|hello|hey|greetings|good (?:morning|afternoon|evening|day))(?: there| all| everyone| team| again)?|"
                             r"dear \w+|thanks?(?: you)?(?: (?:so|very) much)?(?: in advance)?|"
                             r"please|salam|assalamu? ?alaikum)[\s,!.]*$", re.I)


def _split_clauses(question: str) -> List[str]:
    """Clauses of the original text, using the Router's separator so that
    multipart detection and decomposition agree on where a question splits.
    A piece joined by "and", a comma or "as well as" that does not open a
    question of its own is kept with the previous clause, so the shared
    head of "What is the due date for course syllabi and the conference travel
    form?" is never lost. Every clause is a slice of the question."""
    spans: List[List[int]] = []
    pos, separator = 0, ""
    for m in router_agent._CLAUSE_SEP.finditer(question):
        if m.start() > pos:
            spans.append([pos, m.start(), separator])
        separator, pos = m.group(0), m.end()
    if pos < len(question):
        spans.append([pos, len(question), separator])
    merged: List[List[int]] = []
    for start, end, sep in spans:
        piece = question[start:end].strip(" \t\r\n\"'()")
        if not piece:
            continue
        if merged and not _HARD_SEPARATOR.search(sep) and not _CLAUSE_HEAD.match(piece):
            merged[-1][1] = end                                     # a continuation of the previous clause
        else:
            merged.append([start, end, sep])
    parts = [question[start:end].strip(" \t\r\n\"'()") for start, end, _sep in merged]
    parts = [part for part in parts if part]
    # a greeting or thanks fragment ("Good morning, what is ...") is no clause of its own
    return [part for part in parts if not _GREETING_PIECE.match(part)] or parts


def analyze(question: str) -> CoordinatorAnalysis:
    """Measure a question without deciding anything. Never calls an LLM."""
    text = (question or "").strip()
    report = router_agent.route_with_report(text, llm_fallback=False)
    decision = report.decision
    greeting = decision.primary_route == "greeting"
    scores, cues, first = ({}, {}, {}) if greeting else _score_domains(text)
    if not greeting and decision.primary_route == _CONTACT_ROUTE:
        # A phone, fax or e-mail lookup is institutional navigation even when
        # no navigation word appears ("What is the email of my supervisor?").
        key = SpecialistId.INSTITUTIONAL.value
        scores[key] = max(scores.get(key, 0.0), SELECT_THRESHOLD)
        cues.setdefault(key, []).append("route:contact")
        first.setdefault(key, 0)
    return CoordinatorAnalysis(
        question=text,
        route=decision.primary_route,
        routes=list(decision.routes),
        is_multipart=bool(decision.is_multipart),
        domain_scores=scores,
        domain_cues=cues,
        first_cue=first,
        level=None if greeting else _detect_level(text),
        systems=[] if greeting else _detect_systems(text),
        journey=bool(text) and not greeting and bool(_JOURNEY_CUES.search(text)),
        greeting=greeting,
        channel_lookup=bool(text) and not greeting and bool(_CHANNEL_CUES.search(text)),
        router_llm_attempted=report.llm_attempted,
        clauses=_split_clauses(text) if text else [],
    )


# ---------------------------------------------------------------------------
# Selection and decomposition
# ---------------------------------------------------------------------------
def _rank(analysis: CoordinatorAnalysis) -> List[str]:
    """Domains best first: score, then earliest cue, then canonical order."""
    order = {member.value: i for i, member in enumerate(SpecialistId)}
    return sorted(
        analysis.domain_scores,
        key=lambda d: (-analysis.domain_scores[d], analysis.first_cue.get(d, 10 ** 6), order[d]),
    )


#: A channel clause that also asks a subject question ("What is the TTO's role and contact details?", "What is the
#: per diem for conferences and the email of the research office?") keeps its subject specialist (Step 4.5A-R).
_SUBSTANCE_ASK = _rx(r"\b(?:roles?|responsibilit\w*|dut(?:y|ies)|functions?|purpose|mission|polic(?:y|ies)|rules?|requirements?|process|procedures?|"
                     r"deadlines?|per diem|fees?|budget|percentage|share|eligib\w*|approv\w*|how much|how many)\b|"
                     r"\bwhat\s+(?:does|do)\b[^?]{0,40}\b(?:do|review|handle|manage|cover|offer|provide|approve|fund)\b")


def _select(analysis: CoordinatorAnalysis, limit: int) -> Tuple[List[str], List[str], Optional[str]]:
    """Returns (selected, dropped, override_reason)."""
    ranked = _rank(analysis)
    candidates = [d for d in ranked if analysis.domain_scores[d] >= SELECT_THRESHOLD]
    institutional = SpecialistId.INSTITUTIONAL.value
    if limit <= 0:
        return [], candidates, None
    if analysis.route == _CONTACT_ROUTE:
        subjects = [d for d in candidates if d != institutional]
        # a channel lookup is Institutional's alone unless another clause asks a subject question
        # ("What does the IRB review, and what is its email?" keeps Research for the first clause)
        subject_clause = any((not _CHANNEL_CUES.search(c) or _SUBSTANCE_ASK.search(_CHANNEL_CUES.sub(" ", c)))
                             and any(_score_domains(c)[0].get(d, 0.0) >= SELECT_THRESHOLD for d in subjects)
                             for c in analysis.clauses)
        if (analysis.channel_lookup and not subject_clause) or not subjects:
            return ([institutional], subjects,
                    "contact channel lookup is handled by institutional navigation alone")
        kept = subjects[:limit - 1]                 # institutional always keeps one slot
        return (kept + [institutional], subjects[len(kept):],
                "contact request about %s: subject specialist(s) plus institutional navigation"
                % ", ".join(kept))
    if not candidates and analysis.systems:
        owner = _SYSTEM_OWNER.get(analysis.systems[0])
        if owner is not None:
            return ([owner.value], [],
                    "system access question: %s guides are owned by %s" % (analysis.systems[0], owner.value))
    return candidates[:limit], candidates[limit:], None


def _complexity(analysis: CoordinatorAnalysis, selected: List[str]) -> str:
    if analysis.greeting:
        return "greeting"
    if analysis.journey:
        return "journey"           # classified even when no domain cue fired
    if not selected:
        return "unknown"
    if len(selected) >= 2:
        return "cross_domain"
    if analysis.is_multipart:
        return "multipart"
    return "simple"


def _confidence(analysis: CoordinatorAnalysis, selected: List[str], override: bool) -> float:
    if analysis.greeting:
        return 1.0
    if not selected:
        return 0.0
    if override:
        return 0.75            # route or system evidence, not a domain cue count
    best = max(analysis.domain_scores[d] for d in selected)
    return round(min(1.0, 0.5 + 0.25 * best), 2)


def _assign_clauses(analysis: CoordinatorAnalysis, selected: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Give each selected domain the clauses in which it has a cue. A clause
    whose only cues belong to domains that were NOT selected is reported as
    uncovered rather than glued onto another specialist's task. A clause with
    no cue at all continues the previous clause's domains; a first clause with
    no cue belongs to every selected domain."""
    assigned: Dict[str, List[str]] = {d: [] for d in selected}
    uncovered: List[str] = []
    previous: List[str] = list(selected)
    for clause in analysis.clauses:
        clause_scores, _, _ = _score_domains(clause)
        owners = [d for d in selected if clause_scores.get(d, 0.0) > 0.0]
        if not owners and clause_scores:
            uncovered.append(clause)
            continue
        if not owners:
            owners = previous
        for d in owners:
            assigned[d].append(clause)
        previous = owners
    return assigned, uncovered


def _focus(clauses: List[str], full_question: str) -> Tuple[List[str], bool]:
    """The clauses a specialist should concentrate on, and whether they are a
    real subset of the question. A specialist with no clause of its own, or
    whose clauses add up to the whole question, has the whole question as focus."""
    clauses = [c for c in clauses if c]
    if not clauses or " ".join(clauses).strip().lower() == full_question.strip(" ?").lower():
        return [full_question], False
    return clauses, True


def _decompose(analysis: CoordinatorAnalysis, selected: List[str], limit: int) -> Tuple[List[SpecialistTask], List[str]]:
    """One task per selected specialist. ``question`` is ALWAYS the complete
    original question, so every task stands alone; the specialist's own
    clauses travel in ``context["focus_clauses"]`` and drive its intent and
    system hints."""
    assigned, uncovered = _assign_clauses(analysis, selected) if len(selected) > 1 else ({}, [])
    tasks: List[SpecialistTask] = []
    for n, domain in enumerate(selected[:limit], 1):
        focus, scoped = _focus(assigned.get(domain, []), analysis.question)
        focus_text = " ".join(focus)
        intent = router_agent.classify(focus_text) if scoped else analysis.route
        systems = _detect_systems(focus_text) if scoped else analysis.systems
        tasks.append(SpecialistTask(
            task_id="task-%d" % n,
            question=analysis.question,
            specialist_id=SpecialistId.parse(domain),
            domain=domain,
            intent=intent,
            level=analysis.level,
            system=systems[0] if systems else None,
            context={"full_question": analysis.question, "focus_clauses": list(focus)},
            requested_by=REQUESTED_BY,
            metadata={"scoped": scoped},
        ))
    return tasks, uncovered


def _metadata(analysis: CoordinatorAnalysis, complexity: str, dropped: List[str], uncovered: List[str],
              specialist_limit: int, task_limit: int) -> Dict:
    """The one metadata shape every decision carries, greeting and empty
    questions included. Values are empty when nothing applies; nothing is
    invented."""
    return {
        "complexity": complexity,
        "analysis": analysis.to_dict(),
        "scores": dict(analysis.domain_scores),
        "cues": {k: list(v) for k, v in analysis.domain_cues.items()},
        "dropped_domains": list(dropped),
        "uncovered_clauses": list(uncovered),
        "limits": {"specialists": specialist_limit, "subtasks": task_limit},
        "router": {"route": analysis.route, "is_multipart": analysis.is_multipart,
                   "llm_attempted": analysis.router_llm_attempted},
        "journey": analysis.journey,
    }


#: Keys present in ``CoordinatorDecision.metadata`` for every decision.
METADATA_KEYS = ("complexity", "analysis", "scores", "cues", "dropped_domains", "uncovered_clauses",
                 "limits", "router", "journey")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def coordinate(
    question: str,
    *,
    max_specialists: Optional[int] = None,
    max_subtasks: Optional[int] = None,
) -> CoordinatorDecision:
    """Decide which specialists a question needs and how it splits.

    Deterministic and offline: the same question always yields the same
    decision, and no LLM or network call is made. ``max_specialists`` and
    ``max_subtasks`` override the configured Plan F budgets (tests only).
    """
    specialist_limit = PLAN_F_MAX_SPECIALISTS if max_specialists is None else max(0, int(max_specialists))
    task_limit = PLAN_F_MAX_SUBTASKS if max_subtasks is None else max(0, int(max_subtasks))
    limit = min(specialist_limit, task_limit)     # every selected specialist needs one task

    analysis = analyze(question)
    if analysis.greeting:
        return CoordinatorDecision(
            intents=["greeting"], confidence=1.0, reason="greeting: no specialist needed",
            metadata=_metadata(analysis, "greeting", [], [], specialist_limit, task_limit),
        )
    if not analysis.question:
        return CoordinatorDecision(
            confidence=0.0, reason="empty question",
            metadata=_metadata(analysis, "unknown", [], [], specialist_limit, task_limit),
        )

    selected, dropped, override = _select(analysis, limit)
    complexity = _complexity(analysis, selected)
    tasks, uncovered = _decompose(analysis, selected, limit)
    if not selected and analysis.journey:
        reason = "orientation journey question; journey mode is not implemented yet"
    elif not selected:
        reason = "no domain cue matched; fall back to the general handbook path"
        if analysis.systems:
            reason += " (system %s recorded, it implies no specialist)" % ", ".join(analysis.systems)
    elif override:
        reason = override
    else:
        reason = "%s: %s" % (complexity, ", ".join(selected))
    if analysis.journey and selected:
        reason += " (orientation journey; journey mode not implemented yet)"

    return CoordinatorDecision(
        domains=_rank(analysis),
        intents=analysis.routes,
        level=analysis.level,
        systems=analysis.systems,
        selected_specialists=[SpecialistId.parse(d) for d in selected],
        subtasks=tasks,
        # Synthesis aggregates several specialists' findings, or a single
        # specialist's findings for a multipart question. The journey label
        # by itself never forces it.
        requires_synthesis=len(selected) >= 2 or (len(selected) == 1 and analysis.is_multipart),
        confidence=_confidence(analysis, selected, override is not None),
        reason=reason,
        used_llm=False,
        metadata=_metadata(analysis, complexity, dropped, uncovered, specialist_limit, task_limit),
    )


__all__ = ["COMPLEXITIES", "CoordinatorAnalysis", "LEVELS", "METADATA_KEYS", "REQUESTED_BY", "SYSTEMS",
           "analyze", "coordinate", "cue_scores", "cue_text"]
