"""
Research & Innovation Specialist (canonical id: ``research``).

DECISION OWNED
    Whether the approved, source-scoped evidence answers the research part
    of a faculty question, expressed as structured findings. Nothing else:
    the specialist does not route (the Coordinator did), does not write the
    answer (Synthesis will), does not verify (the Verifier will) and never
    calls another specialist. It imports the shared helpers (scoping,
    evidence, details, base) and never the Teaching specialist.

VERSION 1 IS DETERMINISTIC AND EXTRACTIVE
    No LLM call is made anywhere in this module (``llm_used`` is always
    False). Every finding is extractive: ``claim`` is the quoted handbook
    text itself and ``evidence_quote`` is an exact slice of a retrieved
    chunk. Nothing is paraphrased, so nothing can be invented.

SIX GATES BEFORE A CHUNK SUPPORTS A CLAUSE
    1. source and page scope: the owned section identities (chapter plus
       printed number) plus the chapter 9 preamble page compiled by
       ``scoping.compile_scope``;
    2. section eligibility on shared pages (``scoping.SectionGuard``): text
       of a non-owned section is never quoted; an unresolved page is rejected;
    3. relevance: one shared anchor concept or two shared vocabulary
       concepts from the Research vocabulary (``is_relevant``); generic words
       carry no concept;
    4. the production reranker gate ``MIN_RERANK_SCORE`` on every kept item;
    5. quote quality (``evidence.select_quote``);
    6. completeness: when the clause asks for a detail (amount, deadline,
       lead time, duration, percentage, quantity, frequency, minimum,
       maximum, approving authority, responsible party, eligibility,
       required documents, project type, funding source, role, publication
       type, requirement, ownership, permission, a specific focus, a contact
       channel or a named research category), the clause is fully
       ``supported`` only when ONE local unit of ONE finding contains that
       detail, BOUND to the object the clause asks about (the asked grant,
       advance, per diem, report, recipient or conference locality, never a
       value of another object in the same text); otherwise ``partial`` and
       ``missing`` names the gap. Topic relevance makes a chunk searchable;
       it never makes a clause complete.

PROCEDURAL SAFETY
    A system or click procedure (portal, screen, button, upload field) is
    never ``supported`` from the handbook: at most ``partial`` when the
    evidence names the system or shares an anchor. A policy-process request
    ("how do I apply for a grant") is at most ``partial`` because the
    handbook states review stages, not the complete submission procedure;
    "what happens after submission" may be ``supported`` from the stated
    review chain. Instructions are never generated.

OWNERSHIP DEFENCE
    A clause about a WLAM faculty category and its teaching load, credit
    hours or supervision credit is Teaching's even when it says "research";
    it is returned ``out_of_scope`` with a Teaching handoff. Leave,
    promotion procedure, evaluation process and incentives are Faculty
    Services & HR; navigation and contacts are Institutional: a clause that
    only asks where something is or how to reach someone is ``out_of_scope``
    with an Institutional handoff, and one that also asks something
    substantive stays ``partial`` because research text never supplies a
    contact channel. A shared clause is answered from Research evidence and
    handed off as well, to every other domain it names.
"""
from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from ...config import MIN_RERANK_SCORE, PLAN_F_MAX_SUBTASKS
from ...retrieval import RetrievalScope, deduplicate_by_text, gather_candidates
from ...sources import SectionMap, SourceRecord, section_map_for
from ...text_utils import normalize_text, tokenize
from ..types import ALL_ROUTES
from . import details as _details
from . import evidence as _evidence
from . import scoping as _scoping
from .base import Specialist, redact_error
from .retrieval_intent import resolve_clause_retrieval_intent
from .contracts import Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from .details import (Detail, _AUTHORITY_NOUN, _CURRENCY_AFTER, _CURRENCY_FIRST, _IDENTIFIER_BEFORE, _LIMIT_MAX, _LIMIT_MIN, _NUMBER,
                      _NUMBER_WORD, _OBLIGATION, _PENALTY_EVIDENCE, _TIMING, _frequency_ok, authority_check, responsibility_check)
from .domain_cues import research_owned
from .evidence import _ITEM_START, _LIST_MARKER, _local_units, _spans, _units

#: The only source this specialist may retrieve from.
APPROVED_SOURCE_ID = "uos_faculty_handbook_2025_26"

#: Sections the specialist owns: (chapter, printed number, label). Chapter AND
#: number identify a section because the handbook prints some numbers twice
#: and prints the copyright policy inside chapter 9 as 12.3.
OWNED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (3, "3.5", "Responsibility to Research"),
    (5, "5.4", "Participation in Conferences"),
    (8, "8.1", "Research Institutes"),
    (8, "8.2", "Research Support Units and Departments"),
    (8, "8.3", "Research Priority Areas"),
    (8, "8.4", "Research Groups"),
    (8, "8.5", "University Research Board"),
    (8, "8.6", "Research Committees"),
    (8, "8.7", "Research Grants"),
    (8, "8.8", "Research Regulations"),
    (8, "8.9", "Scientific Publishing at the University of Sharjah Journal"),
    (8, "8.10", "Collaborative Research Agreements"),
    (8, "8.11", "Scientific Research Procedures and Regulations"),
    (9, "9.1", "Intellectual Property Ownership Rights"),
    (9, "9.2", "University Responsibilities"),
    (9, "9.3", "Responsibilities of UoS Inventors"),
    (9, "9.4", "Disclosure of Intellectual Property"),
    (9, "9.5", "Patent Protection"),
    (9, "12.3", "Copyright Protection Policy (printed as 12.3 inside chapter 9)"),
    (9, "9.7", "Tangible Research Property"),
    (9, "9.8", "Commercialization"),
    (9, "9.9", "Distribution of Licensing Revenues"),
    (9, "9.10", "Dispute Resolution"),
    (16, "16.1", "Faculty Professional Consultancy"),
    (16, "16.2", "Conflict of Interest"),
    (16, "16.3", "Animal Use in Research"),
    (16, "16.4", "Human Use in Research"),
)
#: Sections whose research CONTENT the specialist may quote while the
#: evaluation or promotion PROCESS stays with Faculty Services & HR (every
#: clause naming that process is also handed off).
SHARED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (6, "6.4", "Research and Scholarly Activities Evaluation"),
    (7, "7.2", "Research and Scholarship"),
    (7, "7.7", "Criteria for Accrediting Periodicals and Journals for Promotion Purposes"),
)
SCOPE_SECTIONS: Tuple[Tuple[int, str, str], ...] = OWNED_SECTIONS + SHARED_SECTIONS
#: The chapter 9 preamble (TTO, definitions of Inventor, IP, Gross and Net
#: Income) lies outside every section record; it is admitted explicitly.
EXTRA_PAGES: Tuple[Tuple[int, int], ...] = ((175, 175),)

#: Sections deliberately kept out because another specialist owns them.
EXCLUDED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (3, "3.1", "Teaching Responsibilities, including supervision eligibility"),
    (3, "3.4", "Responsibility to Office Hours"),
    (3, "3.6", "Responsibility to Community Service"),
    (3, "3.11", "Workload Allocation Model (WLAM), including the research categories"),
    (3, "3.13", "Professional Ethics and Disciplinary Actions"),
    (4, "4.3", "Leaves and Absences"),
    (4, "4.6", "Annual Incentives for Distinguished Faculty Members"),
    (5, "5.1", "Faculty development goals"),
    (5, "5.2", "Faculty development categories and new faculty training (both printed 5.2)"),
    (5, "5.3", "Faculty Peer Observation of Teaching"),
    (5, "5.5", "Policy on Offering Courses in the Summer Semester"),
    (6, "6.6", "Evaluation Process and Criteria (and the rest of chapter 6)"),
    (7, "7.8", "Promotion Procedures (and the rest of chapter 7)"),
    (13, "13.8", "Laboratory Safety"),
    (15, "15.1", "Central Laboratories"),
    (16, "16.5", "Policy Prohibiting Sexual Harassment"),
)

#: University systems a research question may name. Never specialists.
SYSTEMS: Tuple[str, ...] = ("portal", "myuos", "fis", "banner", "blackboard")

#: At most this many findings per focus clause.
MAX_FINDINGS_PER_CLAUSE = 2
#: At most this many relevant, quotable candidates are examined per clause.
MAX_CANDIDATES_PER_CLAUSE = 10
#: Focus clauses handled in one run; the architecture's subtask limit.
MAX_CLAUSES = PLAN_F_MAX_SUBTASKS
#: A quote extended by its neighbouring unit may not exceed this many words.
MAX_QUOTE_WORDS = 90

#: Hyphen-like separators between words are read as spaces before concept
#: and cue matching ("add/drop", "peer-reviewed", "co-investigator").
_CUE_SEPARATORS = re.compile(r"(?<=\w)[-‐‑‒–—/](?=\w)")
#: WLAM faculty categories are Teaching vocabulary: the research cues never
#: see them (blanked, same length); "research" elsewhere still counts.
_WLAM_CATEGORY = re.compile(r"\b(?:research[- ]intensive|active[- ]research|research[- ]active)\s+faculty\b", re.I)


def cue_text(text: str) -> str:
    """``text`` as the cue and concept patterns read it."""
    return _CUE_SEPARATORS.sub(" ", text)


def _research_view(text: str) -> str:
    return _WLAM_CATEGORY.sub(lambda m: " " * len(m.group(0)), text)


# ---------------------------------------------------------------------------
# Research concept vocabulary (relevance)
# ---------------------------------------------------------------------------
#: (pattern, concept, anchor). Patterns run in order on lower-cased text and
#: each matched span is consumed. An ANCHOR establishes topic relevance on
#: its own; plain vocabulary needs a second match. Nothing here establishes
#: answer completeness (see the detail gate).
_CONCEPT_PATTERNS: Tuple[Tuple[str, str, bool], ...] = (
    (r"research\s+grants?|seed\s+(?:fund\w*|grants?|projects?)|competitive\s+(?:grants?|projects?)|collaborative\s+(?:grants?|projects?|research)|"
     r"targeted\s+(?:grants?|projects?|research)|\bgrants?\b", "grant", True),
    # funding is an anchor only in a research context ("funding for training" is not)
    (r"internal(?:ly)?\s+fund\w*|external(?:ly)?\s+fund\w*|research\s+fund\w*|seed\s+fund\w*|grant\s+fund\w*|"
     r"\bfund(?:s|ed|ing)?\s+(?:research|projects?|proposals?|grants?)|\bsponsor\w*|\bfunders?\b", "funding", True),
    # research ethics, ethics bodies and "ethics governing research" are anchors; bare "ethics" is vocabulary
    (r"research\s+ethics|ethics\s+(?:committee|approval|application|review|clearance)|ethical\s+(?:standards?|approval|review|conduct|compliance)|"
     r"ethics\s+governing\s+research", "research_ethics", True),
    # informed consent is the human-research consent rule (16.4), an IRB topic
    (r"\birb\b|institutional\s+review\s+board|human\s+(?:subjects?|participants?|use\s+in\s+research)|informed\s+consent", "irb", True),
    (r"\biacuc\b|animal\s+(?:care|use|research|welfare)|institutional\s+animal", "iacuc", True),
    # a Materials Transfer Agreement is tangible research property (9.7)
    (r"materials?\s+transfer\s+agreements?|\bmtas?\b", "mta", True),
    (r"intellectual\s+property|\bip\s+(?:rights?|policy|ownership|protection|assets?)|\bip\b", "ip", True),
    (r"patent\w*", "patent", True),
    (r"tech(?:nology)?\s+transfer|\btto\b", "technology_transfer", True),
    (r"commerciali[sz]\w*|licens(?:e|es|ing|ed)\b|royalt(?:y|ies)|net\s+royalty", "commercialisation", True),
    (r"inventors?\b|inventions?\b|invention\s+disclosures?|disclosure\s+of\s+(?:intellectual|inventions?)", "inventor", True),
    (r"copyright\w*", "copyright", True),
    (r"research\s+institutes?|\binstitutes?\b|\brimhs\b|\brise\b|\briham\b", "research_institute", True),
    (r"research\s+cent(?:er|re)s?", "research_centre", True),
    (r"research\s+groups?", "research_group", True),
    (r"university\s+research\s+board|research\s+board|\bvcrgs\b|\bovcrgs\b|vice\s+chancellor\s+for\s+research", "research_board", True),
    (r"research\s+committees?", "research_committee", True),
    (r"research\s+projects?|funded\s+projects?", "research_project", True),
    (r"principal\s+investigators?|co-?\s?investigators?|\bpi\b", "principal_investigator", True),
    # scientific publishing: publications, refereed or indexed journals, editorial bodies; "publish your course
    # materials" and "library journal circulation" are not the topic
    (r"scientific\s+publishing|publications?\b|publish\w*\b(?!\s+(?:your\s+|the\s+|my\s+)?(?:course|courses|lecture|lectures|slides|materials?|grades?))|"
     r"(?<!library )journals?\b(?!\s+(?:circulation|loans?))|editorial\s+(?:board|secretary)|chief\s+editor|editors?\s+in\s+chief|scopus|proceedings",
     "publishing", True),
    # "consulting" is consultancy; "a consultant or research partner" is the community-service role of 3.6
    (r"consultanc(?:y|ies)|consulting\b", "consultancy", True),
    (r"misconduct|plagiaris\w*|fabricat\w*|falsif\w*|redundant\s+publication", "misconduct", True),
    (r"conflicts?\s+of\s+interest", "conflict_of_interest", True),
    (r"research\s+priority\s+areas?|priority\s+areas?", "research_priority", True),
    # the per diem is its own topic: a registration-fee line must not answer a per diem question
    (r"per\s+diem", "per_diem", True),
    (r"(?<!video )conferences?\b", "conference", True),
    (r"research\s+assistants?", "research_assistant", True),
    (r"sabbatical\w*", "sabbatical", True),
    (r"(?:thes[ie]s|dissertations?)[^.?]{0,40}\bsupervis\w*|supervis\w*[^.?]{0,40}\b(?:thes[ie]s|dissertations?)|"
     r"supervis\w*\s+(?:of\s+)?(?:a\s+|the\s+|my\s+)?(?:research\s+(?:projects?|students?|work)|postgraduate|graduate\s+students?|phd)|supervisees?", "supervision", True),
    (r"research\s+funding\s+department|\brfd\b|research\s+support\s+units?|research\s+office|research\s+groups\s+department|research\s+publishing\s+department", "research_support", True),
    (r"visiting\s+researchers?|research\s+visits?", "visiting_researcher", True),
    (r"petty\s+cash|research\s+advance|funding\s+ceiling|research\s+budget", "research_budget", True),
    (r"research\s+(?:regulations?|policy|policies|rules|procedures?)", "research_policy", True),
    # ---- plain vocabulary: needs a second match --------------------------
    (r"\bresearch\w*", "research", False),
    (r"\bfunding\b|\bfunded\b|\bfunds?\b", "fund", False),
    (r"\bethics\b|\bethical\b", "ethics", False),
    (r"registration\b", "registration", False),
    (r"\blocal\b|\bexternal(?:ly)?\b|\binternal(?:ly)?\b|\binternational\b|\bnational\b", "locality", False),
    (r"scholar\w*", "scholarship", False),
    (r"proposals?\b", "proposal", False),
    (r"budgets?\b", "budget", False),
    (r"awards?\b|rewards?\b|incentives?\b", "award", False),
    (r"approv\w*|authori[sz]\w*|endors\w*|permission", "approval", False),
    (r"committees?\b|councils?\b|boards?\b", "committee", False),
    (r"\bdeans?\b|\bchairs?\b|chancellor|president", "authority_person", False),
    (r"offices?\b|departments?\b|\bunits?\b|deanship", "office", False),
    (r"agreements?\b|contracts?\b|memorand\w*", "agreement", False),
    (r"collaborat\w*|cooperat\w*|partners?\w*", "collaboration", False),
    (r"applicants?\b|\bapply\b|applications?\b", "applicant", False),
    (r"eligib\w*|entitled|qualif\w*", "eligibility", False),
    (r"revenue|income|\bshare\b|proceeds", "revenue", False),
    (r"reports?\b|\bforms?\b|documents?\w*|abstracts?\b|fliers?\b|flyers?\b|acceptance|invoices?\b|copies|\bcopy\b|letters?\b", "document", False),
    (r"thes[ie]s\b|dissertations?\b", "thesis", False),
    (r"supervis\w*", "supervise", False),
    (r"students?\b|candidates?\b|postgraduate|graduate", "student", False),
    (r"projects?\b", "project", False),
    (r"travel\w*|\btrips?\b", "travel", False),
    (r"\bdays?\b|\bmonths?\b|\byears?\b|\bweeks?\b", "period", False),
    (r"deadlines?|\bdates?\b|submission|submit\w*", "submission", False),
    (r"teaching\s+loads?|workloads?|\bloads?\b", "teaching_load", False),
    (r"reduc\w*|releas\w*|exempt\w*", "release", False),
    (r"promotion|promoted|\brank\b", "promotion", False),
    (r"evaluat\w*|appraisal|assessment", "evaluation", False),
    (r"\bleaves?\b", "leave", False),
    (r"training|workshops?|capacity\s+building|development", "training", False),
    (r"laborator\w*|\blabs?\b|equipment|facilities", "facility", False),
    (r"payment|\bpaid\b|\bpay\b|compensation|remuneration|\bfees?\b|\bcosts?\b|expenses?", "payment", False),
    (r"percent\w*|%", "percent", False),
    (r"members?\b|membership", "member", False),
    (r"disclos\w*", "disclosure", False),
    (r"protect\w*", "protection", False),
    (r"own(?:s|ed|ership)?\b", "ownership", False),
    (r"disputes?\b|arbit\w*", "dispute", False),
    (r"terms?\s+of\s+office|renewable", "term", False),
    (r"editors?\b", "editor", False),
)
_COMPILED_CONCEPTS = tuple((re.compile(p, re.I), c, a) for p, c, a in _CONCEPT_PATTERNS)
ANCHOR_CONCEPTS: FrozenSet[str] = frozenset(c for _, c, a in _CONCEPT_PATTERNS if a)
VOCABULARY_CONCEPTS: FrozenSet[str] = frozenset(c for _, c, _ in _CONCEPT_PATTERNS)

#: Words that never establish relevance on their own. Absent from the
#: vocabulary above; the list documents the decision and is checked by tests.
GENERIC_TERMS: FrozenSet[str] = frozenset({
    "policy", "policies", "process", "procedure", "rules", "regulations", "guidelines", "university", "uos", "faculty",
    "staff", "academic", "information", "requirement", "requirements", "provide", "use", "activities", "activity",
    "support", "services", "service", "the", "following", "quality", "excellence", "mission", "strategic", "plan",
})

#: Every anchor built on the word "research" also carries "research" as
#: vocabulary, and a text that uses the word is on a research-policy topic.
_ANCHOR_IMPLIES: Dict[str, str] = {
    "research_policy": "research", "research_ethics": "research", "research_institute": "research", "research_centre": "research",
    "research_group": "research", "research_board": "research", "research_committee": "research", "research_project": "research",
    "research_priority": "research", "research_assistant": "research", "research_support": "research", "research_budget": "research",
    "visiting_researcher": "research",
}


@dataclass(frozen=True)
class Concepts:
    """Normalised concepts of a text. ``anchors`` is a subset of ``vocab``."""

    anchors: FrozenSet[str]
    vocab: FrozenSet[str]


def concepts(text: str) -> Concepts:
    """Research concepts found in ``text`` after normalisation."""
    work = cue_text(text.lower())
    found: List[Tuple[str, bool]] = []

    def consume(concept: str, anchor: bool):
        def _sub(match: "re.Match[str]") -> str:
            found.append((concept, anchor))
            return " " * (match.end() - match.start())
        return _sub

    for pattern, concept, anchor in _COMPILED_CONCEPTS:
        work = pattern.sub(consume(concept, anchor), work)
    anchors = frozenset(c for c, a in found if a)
    vocab = {c for c, _ in found}
    vocab.update(_ANCHOR_IMPLIES[a] for a in anchors if a in _ANCHOR_IMPLIES)
    return Concepts(anchors=anchors, vocab=frozenset(vocab))


def relevance(clause: Concepts, text: str) -> Tuple[int, int]:
    """(shared anchor concepts, shared vocabulary concepts) between the clause
    and ``text``. Only the research-policy anchor is satisfied by its base
    word alone (a policy question is broad); the other research anchors need
    their own phrase in the text."""
    other = concepts(text)
    shared = clause.anchors & other.anchors
    implied = {a for a in clause.anchors - other.anchors if a == "research_policy" and _ANCHOR_IMPLIES.get(a) in other.vocab}
    return len(shared) + len(implied), len(clause.vocab & other.vocab)


def is_relevant(clause: Concepts, text: str) -> bool:
    """One shared anchor concept, or two shared vocabulary concepts."""
    anchors, vocab = relevance(clause, text)
    return anchors >= 1 or vocab >= 2


# ---------------------------------------------------------------------------
# Requested details (completeness)
# ---------------------------------------------------------------------------
_RESEARCH_APPROVAL_VERB = re.compile(r"\bapprov\w*|\bauthori[sz]\w*|\bendors\w*|\bsign(?:ed|s)?\s+off\b|\bratif\w*|\breview(?:s|ed)?\b|\brecommend\w*|"
                                     r"\bdetermin\w*|\bgrant(?:s|ed)\b|\bpermit\w*|\bdecid\w*|\bdecisions?\b|\bappoint\w*|\bnominat\w*|\belect(?:s|ed)?\b|"
                                     r"\bdesignat\w*|\bsign(?:s|ed|ing)?\b", re.I)
#: A recommending body is not the approver: "Approving the annual research budget
#: ..., based on the College's recommendation" names no approver.
_RECOMMENDER = re.compile(r"\bbased\s+on\s+(?:the\s+)?[\w\s'’-]{0,50}?recommendations?(?:\s+(?:from|of|by)\s+(?:the\s+)?[\w\s'’-]{0,50}?(?=[,.;]|$))?|"
                          r"\bupon\s+(?:the\s+)?[\w\s'’-]{0,50}?recommendations?|\brecommendations?\s+(?:from|of|by)\s+(?:the\s+)?[\w\s'’-]{0,50}?(?=[,.;]|$)|"
                          r"\bafter\s+consult\w*\s+(?:with\s+)?(?:the\s+)?[\w\s'’-]{0,40}", re.I)
_RESEARCH_AUTHORITY_NOUN = re.compile(
    _AUTHORITY_NOUN.pattern.rstrip(")\\b") +
    r"|vcrgs|ovcrgs|vice\s+chancellor\s+for\s+research(?:\s+and\s+graduate\s+studies)?|vice\s+chancellor\s+for\s+academic\s+affairs|"
    r"university\s+research\s+board|research\s+board|research\s+committees?|\btto\b|technology\s+transfer\s+office|\birb\b|\biacuc\b|"
    r"institutional\s+review\s+board|institutional\s+animal\s+care\s+and\s+use\s+committee|college\s+of\s+graduate\s+studies|deanship|"
    r"editorial\s+board|research\s+funding\s+department|\brfd\b|graduate\s+studies)\b", re.I)
_RESEARCH_PARTY = re.compile(
    r"\b(?:research\s+funding\s+department|\brfd\b|research\s+groups\s+department|research\s+publishing\s+department|research\s+support\s+units?|"
    r"research\s+office|technology\s+transfer\s+office|\btto\b|\birb\b|\biacuc\b|institutional\s+review\s+board|university\s+research\s+board|"
    r"research\s+board|research\s+committees?|vcrgs|ovcrgs|vice\s+chancellor\s+for\s+research(?:\s+and\s+graduate\s+studies)?|deans?|chairs?|"
    r"chairpersons?|chancellor|colleges?|departments?|institutes?|directors?|coordinators?|principal\s+investigators?|editorial\s+board|"
    r"chief\s+editor|deanship|college\s+of\s+graduate\s+studies|health\s+and\s+safety\s+office)\b", re.I)
_authority_ok = authority_check(_RESEARCH_APPROVAL_VERB, _RESEARCH_AUTHORITY_NOUN)
#: Responsibility verb forms: the shared forms, plus a named party set up "to
#: manage / oversee / administer" something and a matter that "falls under the
#: purview of" a party (both are how the IP policy states responsibility).
_RESEARCH_RESPONSIBLE_VERB = re.compile(
    _details._RESPONSIBLE_VERB.pattern +
    r"|\bto\s+(?:manage|oversee|administer|coordinate|handle|operate|run|supervise)\b|\b(?:falls?|is|are)\s+under\s+the\s+(?:purview|responsibility|supervision)\s+of\b",
    re.I)
_responsibility_ok = responsibility_check(_RESEARCH_PARTY, _RESEARCH_RESPONSIBLE_VERB)
#: A consequence stated by the research rules: the shared penalty words plus
#: closing a project, reclaiming or repaying funds, referral, withdrawal of
#: support, rejection, and refusal to reimburse.
_RESEARCH_PENALTY_EVIDENCE = re.compile(
    _PENALTY_EVIDENCE.pattern +
    r"|\breclaim\w*|\brepa(?:y|id)\b|\brefund\w*|\breturn\s+(?:the\s+|any\s+|all\s+)?(?:spent\s+|unused\s+)?funds\b|\bclose\s+the\s+project\b|"
    r"\brefer\s+the\s+(?:issue|matter|case)\b|\brevok\w*|\bwithdraw\w*\s+(?:the\s+)?(?:funding|grant|support)\b|\bterminat\w*|\brejection\b|\brejected\b|"
    r"\bnot\s+(?:be\s+)?(?:obligated|obliged)\s+to\s+reimburse\b|\bno\s+expenses\s+will\s+be\s+covered\b", re.I)

_TIME_UNIT = re.compile(r"^(?:(?:calendar|working|academic|renewable|consecutive|full|additional)\s+)?(?:days?|weeks?|months?|years?|semesters?)\b", re.I)
_ELIGIBILITY_EVIDENCE = re.compile(r"\beligib\w*|\bentitled\b|\b(?:may|can)\s+apply\b|\bapplicants?\s+must\b|\bmust\s+be\s+(?:a|an|the|regular|full|main|joint)\b|\bmust\s+(?:have|hold)\b|"
                                   r"\bqualif\w*|\bconditions?\b|\bcriteria\b|\bprovided\s+(?:that|they|it)\b|\bonly\s+(?:if|once|when)\b|\bare\s+(?:allowed|permitted)\s+to\b", re.I)
_DOCUMENT_EVIDENCE = re.compile(r"\b(?:forms?|reports?|acceptance|abstracts?|fliers?|flyers?|invoices?|receipts?|supporting\s+documents?|copy|copies|letters?|"
                                r"cvs?|curriculum\s+vitae|certificates?|proposals?|itinerary|tickets?|boarding\s+pass(?:es)?)\b", re.I)
_FUNDING_SOURCE_EVIDENCE = re.compile(r"\b(?:internal(?:ly)?|external(?:ly)?|sponsor\w*|funders?|foundation|ministry|industry|donors?|agenc(?:y|ies)|self[- ]funded)\b", re.I)
_ROLE_EVIDENCE = re.compile(r"\b(?:principal\s+investigators?|co-?\s?investigators?|research\s+assistants?|coordinators?|chief\s+editor|editorial\s+board|"
                            r"editorial\s+secretary|editor-?\s?in-?\s?chief|group\s+(?:leaders?|coordinators?)|directors?|\bpi\b|inventors?)\b", re.I)
_PUBLICATION_TYPE_EVIDENCE = re.compile(r"\b(?:scopus|class\s*\(?[ab]\)?|refereed|peer[- ]reviewed|journals?|books?|proceedings|conference\s+papers?|sjr|impact\s+factor|indexed)\b", re.I)
_PROJECT_TYPE_EVIDENCE = re.compile(r"\b(?:seed|competitive|collaborative|targeted|internal(?:ly)?|external(?:ly)?|basic|applied|pilot)\b", re.I)

#: Count nouns a research quantity may be measured in (the shared list plus
#: research nouns), with up to two qualifiers ("62 active research groups").
_RESEARCH_UNIT_NOUN = re.compile(
    r"^(?:\(\d+\)\s*)?(?:(?:credit|contact|working|regular|calendar|academic|research|active|external|internal|local|international|national|regional|"
    r"scientific|university[- ]supported|supported)\s+){0,2}"
    r"(?:hours?|days?|weeks?|months?|semesters?|years?|students?|candidates?|researchers?|courses?|sessions?|times|credits?|points?|percent|%|"
    r"papers?|publications?|articles?|books?|members?|groups?|projects?|thes[ie]s|dissertations?|conferences?|proposals?|patents?|institutes?|"
    r"cent(?:er|re)s?|grants?|awards?|workshops?|participants?|assistants?|editors?)\b", re.I)
#: The counted subject of a quantity question: the noun phrase after "how
#: many", stopping at the verb or pronoun ("how many theses can I supervise"
#: counts theses; the verb phrase is not part of the subject).
_RESEARCH_SUBJECT_AFTER = re.compile(
    r"\b(?:how\s+many|number\s+of|count\s+of)\s+((?:(?!(?:can|may|might|must|should|shall|do|does|did|is|are|was|were|will|would|could|"
    r"i|we|you|they|he|she|it|to|at|per|for|in|on|of|by|with|from)\b)[\w'/-]+\s*){1,4})", re.I)
#: A monetary value is a currency next to a number, or an explicit "amount
#: of N". A "free of charge" or "at no cost" statement is not an amount.
_NUMERIC_MONEY = re.compile(
    r"(?<!\w)" + _CURRENCY_FIRST + r"\s*" + _NUMBER_WORD + r"\b|\b" + _NUMBER_WORD + r"\s*" + _CURRENCY_AFTER + r"(?!\w)|"
    r"\b(?:amount|sum|price|costs?|fees?|charges?|payment)\s+(?:of\s+|is\s+|are\s+|will\s+be\s+|shall\s+be\s+)" + _NUMBER_WORD + r"\b(?!\.\d)", re.I)
#: A percentage value is a number with a percent sign or the word percent;
#: the bare word "percentage" states no value.
_NUMERIC_PERCENT = re.compile(r"(?<![\d.])" + _NUMBER_WORD + r"\s*(?:%|percent\b)", re.I)
#: Questions whose interrogative asks for a party ("who determines", "which
#: committee reviews"): a value noun in the object ("the funding ceiling") is
#: not a requested value unless a value interrogative is present as well.
_PARTY_QUESTION = re.compile(r"^\s*(?:(?:and|but|also)\s+)?(?:who|whom|whose|which\s+(?:office|body|council|committee|board|person|dean|unit|"
                             r"department|authority))\b", re.I)
_VALUE_INTERROGATIVE = re.compile(r"\b(?:how\s+much|how\s+many|how\s+long|how\s+often|how\s+early|how\s+far\s+in\s+advance|what\s+percentage|what\s+share|"
                                  r"what\s+proportion|when|by\s+when|until\s+when|what\s+date|what\s+(?:is|are)\s+the\s+(?:amounts?|maximum|minimum|ceiling|limits?|"
                                  r"rates?|deadlines?|duration|percentages?))\b", re.I)
_VALUE_FAMILIES = frozenset({"amount", "deadline", "quantity", "duration", "frequency", "minimum", "maximum", "percentage", "advance"})


def is_table_row(text: str) -> bool:
    """A printed line that is table data: the shared rule (Research has no domain table shapes)."""
    return _evidence.is_table_row(text, ())


#: A limit phrase attached to a currency amount ("up to $750", "up to AED
#: 700", "a minimum of AED 500"): the shared value rule reads a currency
#: written after the number; research amounts are written currency first.
_MONEY_AFTER_LIMIT_MAX = re.compile(r"\b(?:up\s+to|maximum\s+of|at\s+most|not\s+(?:to\s+)?exceed(?:ing)?|capped\s+at|ceiling\s+of)\s+" + _CURRENCY_FIRST +
                                    r"\s*" + _NUMBER_WORD + r"\b", re.I)
_MONEY_AFTER_LIMIT_MIN = re.compile(r"\b(?:at\s+least|minimum\s+of|no\s+less\s+than|not\s+less\s+than)\s+" + _CURRENCY_FIRST + r"\s*" + _NUMBER_WORD + r"\b", re.I)


def _max_ok(unit: str) -> bool:
    return _details._limit_ok(unit, _LIMIT_MAX, is_table_row) or bool(_MONEY_AFTER_LIMIT_MAX.search(unit))


def _min_ok(unit: str) -> bool:
    return _details._limit_ok(unit, _LIMIT_MIN, is_table_row) or bool(_MONEY_AFTER_LIMIT_MIN.search(unit))


def _duration_ok(unit: str) -> bool:
    """A number followed by a time unit ("two years", "four days", "90 calendar
    days"); a year, a section number or an identifier never counts."""
    for m in _NUMBER.finditer(unit):
        if _IDENTIFIER_BEFORE.search(unit[:m.start()]):
            continue
        if _TIME_UNIT.match(unit[m.end():].lstrip(" -")):
            return True
    return False


def _amount_ok(unit: str) -> bool:
    """A monetary amount: "AED 5000", "$750", "AED 1000-1500", "amount of N"."""
    return bool(_NUMERIC_MONEY.search(unit))


def _percentage_ok(unit: str) -> bool:
    """A percentage value: "50%", "up to 10%", "30 percent"."""
    return bool(_NUMERIC_PERCENT.search(unit))


def _eligibility_ok(unit: str) -> bool:
    return bool(_ELIGIBILITY_EVIDENCE.search(unit))


def _documents_ok(unit: str) -> bool:
    return bool(_DOCUMENT_EVIDENCE.search(unit))


# ---- timing (Research-specific; the shared timing pattern is unchanged) ----
#: Calendar months. "May" is a month only in date context ("15 May", "May
#: 2026", "by May"); the modal verb "may" never is. Abbreviations count only
#: next to a day number.
_MONTH = r"(?:january|february|march|april|june|july|august|september|october|november|december)"
_CALENDAR_MAY = (r"(?-i:May)\s+\d{1,2}(?:st|nd|rd|th)?\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?-i:May)\b|(?-i:May)\s*,?\s+(?:19|20)\d\d\b|"
                 r"\b(?:in|by|until|till|before|from|during|since|through|of|early|mid|late)[\s-]+(?-i:May)\b")
_MONTH_ABBREVIATION = r"\b(?:jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?\s+\d{1,2}\b|\b\d{1,2}\s+(?:jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b"
_ANY_MONTH = r"(?:" + _MONTH + r"|(?-i:May))"
#: Any stated timing: a date, a weekday, a period boundary, a relative limit.
_RESEARCH_TIMING = re.compile(
    r"\b" + _MONTH + r"\b|" + _CALENDAR_MAY + r"|" + _MONTH_ABBREVIATION + r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b|"
    r"\bwithin\s+" + _NUMBER_WORD + r"\s+(?:working\s+|calendar\s+)?(?:hours?|days?|weeks?|months?|years?)\b|"
    r"\b(?:at\s+the\s+)?(?:beginning|start|end)\s+of\s+(?:the\s+|each\s+|every\s+)?(?:first\s+|second\s+|fall\s+|spring\s+|summer\s+)?"
    r"(?:semester|term|course|academic\s+year|first\s+week|year|month)\b|"
    r"\bfirst\s+(?:week|day)\b|\blast\s+day\b|\bno\s+later\s+than\b|\bprior\s+to\b|\bby\s+the\s+(?:end|start|beginning)\b|"
    r"\bbefore\s+(?:the\s+|a\s+|an\s+|its\s+|their\s+)?(?:[\w-]+\s+){0,2}?(?:conferences?|deadline|start|end|beginning|dates?|meetings?|events?|trips?|travel|"
    r"departure|submission|semesters?|terms?|courses?|academic\s+year|announcement|closing|call|expiry|due\s+date|project|visits?|sessions?|periods?|\d{1,2}\b)|"
    r"\bupon\s+(?:the\s+)?(?:project\s+)?(?:completion|return|closure)\b|\b\d{1,2}\s+\w+\s+20\d\d\b", re.I)
#: End-point timing, required when the clause asks for a deadline: an end, a
#: limit or a lead time. A window start alone ("announced at the beginning of
#: each academic year") and the bare word "deadline" are not deadlines.
#: The time-bearing head a relative end point must name ("before the
#: conference date", "until the end of April"); "before the Board" and "three
#: years before the publication of the article" name no time.
_TEMPORAL_HEAD = (r"(?:conferences?|deadline|start|end|beginning|dates?|meetings?|events?|trips?|travel|departure|submission|semesters?|terms?|courses?|"
                  r"academic\s+year|announcement|closing|call|expiry|due\s+date|project|visits?|sessions?|periods?|" + _ANY_MONTH +
                  r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}\b)")
_DEADLINE_END = re.compile(
    r"\buntil\s+(?!further\s+notice)(?:the\s+)?\w+|\btill\s+(?:the\s+)?\w+|\bend\s+of\s+(?:the\s+|each\s+|every\s+|its\s+)?\w+|\blast\s+(?:day|date)\b|"
    r"\bno\s+later\s+than\b|\bprior\s+to\b|\bbefore\s+(?:submission|submitting|applying|travel(?:l?ing)?|departure)\b|"
    r"\b(?:on\s+or\s+)?before\s+(?:the\s+|a\s+|an\s+|its\s+|their\s+)?(?:[\w-]+\s+){0,2}?" + _TEMPORAL_HEAD + r"|"
    r"\bby\s+(?:the\s+(?:end|start|beginning|deadline|due\s+date)|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + _ANY_MONTH + r"|" + _ANY_MONTH +
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\bwithin\s+" + _NUMBER_WORD + r"\s+(?:working\s+|calendar\s+)?(?:hours?|days?|weeks?|months?|years?)\b|"
    r"\b" + _NUMBER_WORD + r"\s+(?:working\s+|calendar\s+)?(?:days?|weeks?|months?)\s+(?:before|prior\s+to|in\s+advance|ahead\s+of)\b|"
    r"\bclos(?:e|es|ed|ing)\s+(?:on|by|at|in)\s+\w+|\bcloses?\s+mid[\s-]+\w+|\bdue\s+(?:on|by|before|within|in)\s+\w+|"
    r"\bupon\s+(?:the\s+)?(?:project\s+)?(?:completion|return|closure)\b|"
    # an explicit date tied to deadline wording ("The deadline is 1 May", "closing date: 15 March 2026")
    r"\b(?:deadline|due\s+date|closing\s+date|last\s+day)\b[^.;]{0,60}?(?:\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + _ANY_MONTH + r"\b|\b" + _ANY_MONTH +
    r"\s+\d{1,2}\b)", re.I)
#: A window START is not an end point: "opens at the end of the first
#: semester", "after the end of the project", "meets at the end of each
#: semester" are blanked before an end point is looked for.
_START_PHRASE = re.compile(r"\b(?:open(?:s|ed|ing)?|start(?:s|ed|ing)?|begin(?:s|ning)?|after|following|from|since|announced|meet(?:s|ing)?)\s+"
                           r"(?:at\s+|on\s+|by\s+|in\s+)?(?:the\s+)?(?:end|beginning|start)\s+of\s+(?:the\s+|each\s+|every\s+)?[\w-]+(?:\s+[\w-]+)?", re.I)
#: A clause that asks for an end point rather than for a time in general.
_EXPLICIT_DEADLINE = re.compile(r"\b(?:deadlines?|due\s+dates?|by\s+when|until\s+when|last\s+(?:day|date)|cut\s?offs?|closing\s+dates?|no\s+later\s+than|"
                                r"latest\s+date|final\s+date|end\s+date|due(?!\s+to))\b", re.I)
#: A lead time: a number of days, weeks or months before something.
_ADVANCE_EVIDENCE = re.compile(r"\b" + _NUMBER_WORD + r"\s+(?:working\s+|calendar\s+)?(?:days?|weeks?|months?)\s+(?:before|prior\s+to|in\s+advance|ahead\s+of)\b", re.I)
#: "from receipt of a complete invention disclosure" names the START of a
#: period, not the thing that is due; it never binds a timing to its object.
_START_REFERENCE = re.compile(r"\b(?:from|after|following|upon)\s+(?:the\s+)?(?:receipt|date|submission|completion|approval|return|end)\s+of\s+"
                              r"(?:a\s+|an\s+|the\s+)?(?:complete\s+)?(?:[\w'-]+\s*){1,4}", re.I)


def _timing_ok(unit: str) -> bool:
    """Any stated timing, never the modal verb "may"."""
    return bool(_RESEARCH_TIMING.search(unit) or _DEADLINE_END.search(unit))


def _deadline_end_ok(unit: str) -> bool:
    """An end point, a limit or a lead time, never a window start alone."""
    return bool(_DEADLINE_END.search(_START_PHRASE.sub(" ", unit)))


# ---- semantic families for yes/no, ownership and permission questions -----
_OWNERSHIP_EVIDENCE = re.compile(
    r"\bown(?:s|ed|er|ers|ership)?\b|\bassign(?:ed|s)?\s+to\b|\bbelong\w*\s+to\b|\bvest(?:s|ed)?\s+in\b|\bretain\w*\s+(?:the\s+)?(?:ownership|rights?|copyright)\b|"
    r"\bproperty\s+of\b|\bclaim\w*\s+(?:the\s+)?(?:ownership|rights?|copyright)\b|\brights?\s+(?:belong|remain|are\s+held|rest)\b", re.I)
_PERMISSION_EVIDENCE = re.compile(
    r"\b(?:allowed|permitted|permissible|may|can(?:not)?|must(?:\s+not)?|shall(?:\s+not)?|prohibit\w*|forbidden|entitled|eligible|only|except|excluded|"
    r"approv\w*|requires?|required|subject\s+to|not\s+(?:be\s+)?(?:allowed|permitted|accepted))\b", re.I)
#: A role's duties: a named role is not enough, a duty or action must be stated.
_ROLE_DUTY = re.compile(r"\b(?:responsib\w*|dut(?:y|ies)|tasks?|functions?|must|shall|should|expected\s+to|required\s+to|coordinat\w+|oversee\w*|"
                        r"manag(?:e|es|ing)|supervis(?:e|es|ing)|leads?|prepar(?:e|es|ing)|submits?|provid(?:e|es|ing)|ensur(?:e|es|ing)|reviews?|"
                        r"conducts?|reports?\s+to|represents?|support(?:s|ing)?\s+the)\b", re.I)


def _role_ok(unit: str) -> bool:
    return bool(_ROLE_EVIDENCE.search(unit) and _ROLE_DUTY.search(unit))


# ---- conference locality ----------------------------------------------------
_CONFERENCE_CONTEXT = re.compile(r"\bconferences?\b|\bper\s+diem\b|\bregistration\s+fees?\b", re.I)
_LOCAL_WORDS = re.compile(r"\blocal(?:ly)?\b|\bdomestic\b|\b(?:within|inside|in)\s+the\s+(?:uae|country|emirates)\b", re.I)
_NONLOCAL_WORDS = re.compile(r"\binternational(?:ly)?\b|\bregional\b|\boverseas\b|\babroad\b|\bforeign\b|\boutside\s+the\s+(?:uae|country|emirates)\b|"
                             r"\bexternal\s+(?:[\w-]+\s+){0,2}conferences?\b", re.I)


def conference_locality(clause: str) -> Optional[str]:
    """"local" or "external" when a conference clause names exactly one locality."""
    text = cue_text(clause)
    if not _CONFERENCE_CONTEXT.search(text):
        return None
    local, other = bool(_LOCAL_WORDS.search(text)), bool(_NONLOCAL_WORDS.search(text))
    if local and not other:
        return "local"
    if other and not local:
        return "external"
    return None


def locality_conflict(locality: Optional[str], text: str) -> bool:
    """True when ``text`` names only the other conference locality. The general
    per diem and registration rules name no locality and stay usable for an
    international question; a local question needs the word "local"."""
    text = cue_text(text)
    if locality == "local":
        return bool(_NONLOCAL_WORDS.search(text)) and not _LOCAL_WORDS.search(text)
    if locality == "external":
        return bool(_LOCAL_WORDS.search(text)) and not _NONLOCAL_WORDS.search(text)
    return False


def _local_ok(unit: str) -> bool:
    return bool(_LOCAL_WORDS.search(cue_text(unit)))


# ---- objects: a value must belong to what the clause asks about ------------
#: (name, clause pattern, evidence pattern, specific, submission). Every
#: SPECIFIC object of the clause segment must be named where the requested
#: detail is stated; when none is, one CONTEXT object must be. Submission
#: objects (requests, applications, reports, proposals) are specific for a
#: timing question ("when must requests be submitted") and context otherwise.
_OBJECTS: Tuple[Tuple[str, str, str, bool, bool], ...] = (
    ("petty_cash", r"petty\s+cash", r"petty\s+cash", True, False),
    # "The advance may be closed ..." inside the petty-cash item is the petty-cash advance, never the research advance
    ("research_advance", r"research\s+advances?|(?<!cash\s)(?<!in\s)\badvances?\b(?!\s+notice)",
                         r"research\s+advances?|^(?![\s\S]*petty\s+cash)[\s\S]*?(?<!in\s)\badvances?\b", True, False),
    ("per_diem", r"per\s+diem|daily\s+allowances?", r"per\s+diem|daily\s+allowances?", True, False),
    ("registration", r"registration\s+fees?|conference\s+registration|register(?:ing)?\s+for\s+(?:a\s+|the\s+)?conference", r"registration", True, False),
    ("consultancy_income", r"consult\w*\s+(?:work\s+|activit(?:y|ies)\s+|services?\s+)?(?:income|revenues?|fees?|earnings|payments?)|"
                           r"(?:income|revenues?|fees?|earnings|payments?)\s+(?:from|of|for)\s+(?:my\s+|the\s+|a\s+|your\s+|our\s+)?consult\w*",
                           r"consult\w*", True, False),
    ("revenue", r"\b(?:income|revenues?|royalt(?:y|ies)|proceeds|earnings)\b|\b(?:university|uos|inventors?|college|department)(?:'s|s'|’s)?\s+share\b|"
                r"\bshares?\s+of\s+(?:the\s+)?(?:licens\w*|royalt\w*|revenues?|income|proceeds)", r"\b(?:income|revenues?|royalt(?:y|ies)|proceeds|earnings|shares?)\b", True, False),
    ("research_assistant", r"research\s+assistants?", r"research\s+assistants?", True, False),
    # student research projects (8.11 item 28) are funded apart from faculty projects
    ("student_project", r"\bstudents?(?:'|’)?s?\s+(?:research\s+)?projects?\b|\bstudent\s+research\b", r"\bstudents?\b", True, False),
    ("university_name", r"(?:university|uos)(?:'s|’s)?\s+name|name\s+of\s+the\s+university|university\s+logo",
                        r"(?:university|uos)(?:'s|’s)?\s+name|name\s+of\s+the\s+university|university\s+logo", True, False),
    ("final_report", r"final\s+(?:research\s+|project\s+)?reports?", r"final\s+(?:research\s+|project\s+)?reports?", True, False),
    ("progress_report", r"progress\s+reports?|annual\s+reports?", r"progress\s+reports?|annual\s+(?:and\s+final\s+)?reports?", True, False),
    ("thesis", r"thes[ie]s|dissertations?", r"thes[ie]s|dissertations?", True, False),
    ("chief_editor", r"chief\s+editors?|editors?\s+in\s+chief", r"chief\s+editors?|editors?\s+in\s+chief", True, False),
    ("editorial_secretary", r"editorial\s+secretar(?:y|ies)", r"editorial\s+secretar(?:y|ies)", True, False),
    ("editorial_board", r"editorial\s+boards?", r"editorial\s+boards?", True, False),
    ("editor", r"(?<!chief\s)\beditors?\b(?!\s+in\s+chief)", r"\beditors?\b|editorial", True, False),
    ("moderator", r"moderators?|arbit(?:er|rator)s?", r"moderators?|arbit(?:er|rator)s?", True, False),
    ("trp", r"tangible\s+research\s+property|\btrp\b|samples?", r"tangible\s+research\s+property|\btrp\b|samples?", True, False),
    ("closure", r"closure|clos(?:e|ing)\s+(?:a\s+|the\s+|my\s+)?(?:research\s+)?project", r"closure|close\s+the\s+project|closed", True, False),
    ("licence", r"licen[cs](?:e|es|ing)\s+agreements?|\bsign\w*[^.?]{0,20}licen[cs]\w*|"
                r"\b(?:grant|issu|negotiat|approv|award)\w*\s+(?:the\s+|a\s+|an\s+)?licen[cs](?:e|es)\b|\blicen[cs](?:e|es)\s+(?:for|to|of)\b",
                r"licen[cs]\w*", True, False),
    ("commercialisation", r"commerciali[sz]\w*", r"commerciali[sz]\w*", True, False),
    ("payment", r"payments?(?:\s+procedures?)?|\b(?:the|my|maximum|minimum|annual|monthly|extra|additional|basic)\s+pay\b|salar(?:y|ies)|"
                r"compensation|remuneration", r"payments?|\bpaid\b|\bpay\b|salar(?:y|ies)|compensat\w*|remunerat\w*", True, False),
    ("journal", r"journals?", r"journals?|periodicals?", True, False),
    ("visitor", r"visiting\s+researchers?|\bvisitors?\b", r"\bvisiting\b|\bvisitors?\b", True, False),
    ("research_group", r"research\s+groups?", r"research\s+groups?|groups?\s+of\s+researchers", True, False),
    ("sabbatical", r"sabbatical\w*", r"sabbatical\w*", True, False),
    ("patent", r"patent\w*", r"patent\w*", True, False),
    ("copyright", r"copyright\w*", r"copyright\w*", True, False),
    ("disclosure", r"disclos\w*", r"disclos\w*", True, False),
    ("proceedings", r"proceedings", r"proceedings", True, False),
    ("mta", r"materials?\s+transfer\s+agreements?|\bmtas?\b", r"materials?\s+transfer\s+agreements?|\bmtas?\b", True, False),
    ("equipment", r"equipment|laborator\w*|\blabs?\b|facilities", r"equipment|laborator\w*|\blabs?\b|facilities", True, False),
    ("time", r"\bhow\s+much\s+(?:of\s+(?:my|their|his|her)\s+)?time\b|\b(?:percentage|share|proportion|portion|fraction)\s+of\s+(?:my\s+|their\s+|faculty\s+)?(?:working\s+)?time\b|"
             r"\bhours?\s+(?:per|a)\s+week\b|\btime\s+(?:allocation|limits?)\b", r"\btime\b|\bhours?\b|\bworking\s+days?\b|\bdays?\s+per\s+week\b", True, False),
    ("request", r"\brequests?\b", r"\brequests?\b", False, True),
    ("application", r"\bapplications?\b|\bapply\b", r"\bapplications?\b|\bappl(?:y|ied|ying)\b", False, True),
    ("report", r"\breports?\b", r"\breports?\b", False, True),
    ("proposal", r"\bproposals?\b", r"\bproposals?\b", False, True),
    ("conference", r"conferences?|travel\w*", r"conferences?|travel\w*", False, False),
    ("grant", r"grants?|\bfund(?:s|ed|ing)?\b|projects?|budgets?", r"grants?|fund\w*|projects?|budgets?", False, False),
    ("principal_investigator", r"principal\s+investigators?|co\s?investigators?|\bpi\b", r"principal\s+investigators?|co\s?investigators?|\bpi\b", False, False),
    ("consultancy", r"consult\w*", r"consult\w*", False, False),
    ("inventor", r"inventors?|inventions?", r"inventors?|inventions?", False, False),
    ("ip", r"intellectual\s+property|\bip\b", r"intellectual\s+property|\bip\b", False, False),
    ("irb", r"\birb\b|institutional\s+review\s+board|human\s+(?:subjects?|participants?)|informed\s+consent",
            r"\birb\b|institutional\s+review\s+board|human\s+(?:subjects?|participants?|research)|informed\s+consent|ethics\s+committee|research\s+ethics", False, False),
    ("iacuc", r"\biacuc\b|animals?", r"\biacuc\b|animals?", False, False),
    ("publishing", r"publications?|publish\w*", r"journals?|publications?|publish\w*", False, False),
)
_COMPILED_OBJECTS = tuple((name, re.compile(c, re.I), re.compile(e, re.I), specific, submission) for name, c, e, specific, submission in _OBJECTS)
#: "goes to the inventor", "distributed to the college": the recipient must be
#: named too. "created by students": so must the creator.
_RECIPIENT = re.compile(r"\b(?:go(?:es)?|paid|distributed|allocated|given|awarded|transferred|allotted)\s+to\s+(?:the\s+|an?\s+|each\s+|its\s+)?"
                        r"((?:[a-z][\w'’]*\s+){0,2}[a-z][\w'’]*)|\bshares?\s+(?:for|of)\s+(?:the\s+)?(inventors?|university|college|department|vcrgs)\b", re.I)
_RECIPIENT_STOP = frozenset({"after", "if", "when", "in", "for", "of", "to", "and", "or", "from", "by", "with", "as", "under", "each", "per"})
_CREATOR = re.compile(r"\b(?:made|created|produced|developed|written|authored|invented|generated)\s+by\s+(?:the\s+|an?\s+|its\s+|their\s+)?([a-z][\w'’]*)", re.I)
_UNBOUND_WORDS = frozenset({"faculty", "member", "staff", "employee", "researcher", "people", "person", "them", "him", "her", "me", "us", "you", "whom",
                            "which", "what", "it", "its", "their", "this", "that", "these", "those"})
_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")


def _stem(word: str) -> str:
    """The word lower-cased without a possessive ending (a matching key, not a linguistic stem)."""
    w = word.lower().replace("’", "'")
    return w[:-2] if w.endswith("'s") else w.rstrip("'")


def _word_forms(word: str) -> FrozenSet[str]:
    """A word and its possible singular forms ("drones" -> drone, "studies" -> study)."""
    w = _stem(word)
    forms = {w}
    if w.endswith("ies") and len(w) > 4:
        forms.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        forms.add(w[:-2])
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        forms.add(w[:-1])
    for british, american in (("isation", "ization"), ("ise", "ize"), ("ised", "ized"), ("licence", "license")):
        if british in w:
            forms.add(w.replace(british, american))
        elif american in w:
            forms.add(w.replace(american, british))
    if w.endswith("ed") and len(w) > 5:                          # "closed" / close, "renewed" / renew
        forms.update((w[:-2], w[:-1]))
    if w.endswith("ing") and len(w) > 5:                         # "leaving" / leave, "closing" / close
        forms.update((w[:-3], w[:-3] + "e"))
    elif len(w) > 3:
        forms.update((w + "ing", w[:-1] + "ing" if w.endswith("e") else w + "ing"))
    return frozenset(forms)


def _has_word(text: str, word: str) -> bool:
    """``text`` contains ``word`` in a singular, plural, possessive or -ing
    form. No looser match: "members" never stands for "membership"."""
    wanted = _word_forms(word)
    return any(_word_forms(token) & wanted for token in _WORD.findall(text.lower()))


def _recipient_named(text: str, word: str) -> bool:
    """``word`` is named as a receiver: "50% to the inventor(s)," or "5% to the
    Inventors' College(s)" for the college, never "to the Inventors' College"
    for the inventor (there the word only modifies the receiver)."""
    forms = "|".join(sorted((re.escape(f) for f in _word_forms(word)), key=len, reverse=True))
    return bool(re.search(r"\bto\s+(?:the\s+|each\s+|an?\s+|its\s+|their\s+)?"
                          r"(?:(?!support|fund|cover|help|assist|promote|enable|finance|develop|sustain|strengthen)[\w'’]+\s+){0,2}?(?:" + forms +
                          r")(?:s|\(s\))?(?:['’]s)?"
                          r"(?=\s*(?:[,.;:)]|and\b|or\b|if\b|in\b|for\b|to\b|$))", text, re.I))


@dataclass(frozen=True)
class _Binding:
    """What a clause segment asks about, and how evidence must name it."""

    specific: Tuple["re.Pattern[str]", ...]
    context: Tuple["re.Pattern[str]", ...]
    recipients: Tuple[str, ...]
    creators: Tuple[str, ...]
    timing: bool

    @property
    def empty(self) -> bool:
        return not (self.specific or self.context or self.recipients or self.creators)

    def names(self, text: str) -> bool:
        text = cue_text(text)
        if self.timing:
            text = _START_REFERENCE.sub(" ", text)
        if self.specific and not all(p.search(text) for p in self.specific):
            return False
        if not self.specific and self.context and not any(p.search(text) for p in self.context):
            return False
        return all(_recipient_named(text, w) for w in self.recipients) and all(_has_word(text, w) for w in self.creators)


def _named_words(pattern: "re.Pattern[str]", text: str) -> Tuple[str, ...]:
    """The receiver or creator a clause names: the head of the noun phrase
    ("the Inventors' College" -> college), not a modifier before it."""
    words: List[str] = []
    for m in pattern.finditer(text):
        phrase = next((g for g in m.groups() if g), "")
        tokens: List[str] = []
        for token in phrase.split():
            if token.lower() in _RECIPIENT_STOP:
                break
            tokens.append(token)
        if not tokens:
            continue
        word = _stem(tokens[-1])
        if not (_word_forms(word) & _UNBOUND_WORDS) and word not in words:
            words.append(word)
    return tuple(words)


def _binding(segment: str, timing: bool = False) -> _Binding:
    text = cue_text(segment)
    specific: List["re.Pattern[str]"] = []
    context: List["re.Pattern[str]"] = []
    for _name, clause_pattern, evidence_pattern, is_specific, submission in _COMPILED_OBJECTS:
        if clause_pattern.search(text):
            (specific if is_specific or (submission and timing) else context).append(evidence_pattern)
    return _Binding(tuple(specific), tuple(context), _named_words(_RECIPIENT, text), _named_words(_CREATOR, text), timing)


#: Approval, review and decision wording: an active verb names its object
#: after it ("The VCAA approves travel requests"); a passive or noun form
#: names the thing right before it ("invoices approved by the Dean") or, after
#: a function word, the subject of the sentence ("a petty cash advance ...,
#: subject to approval by the VCRGS"), plus an "approval of X" phrase.
_ACTIVE_APPROVAL = re.compile(
    r"\b(?:approves|authori[sz]es|endorses|ratifies|reviews|recommends|determines|decides|grants|permits|appoints|nominates|signs\s+off\s+on|signs|"
    r"(?:may|shall|must|will|can|should|to|would)\s+(?:(?:also|first|then|only|formally)\s+)?(?:approve|authori[sz]e|endorse|ratify|review|recommend|"
    r"determine|decide|grant|permit|appoint|nominate|sign)|reviewing|approving|authori[sz]ing|determining|deciding|granting|appointing|endorsing|signing|"
    # the plural present after a plural subject ("research committees REVIEW research grant applications")
    r"(?<=s\s)(?:approve|authori[sz]e|endorse|ratify|review|recommend|determine|decide|appoint|nominate))\b", re.I)
_PASSIVE_APPROVAL = re.compile(
    r"\b(?:approved|approval|pre\s?approval|authori[sz]ed|authori[sz]ation|endorsed|endorsement|ratified|reviewed|recommended|recommendation|"
    r"determined|decided|decisions?|granted|permitted|permission|appointed|consent|nominated|nomination|signed)\b", re.I)
#: "approved research projects", "an appointed Editor-in-Chief": an adjective, not an act of approval.
_ADJECTIVAL_NEXT = re.compile(r"^\s+(?!by\b|in\b|with\b|from\b|for\b|on\b|at\b|upon\b|under\b|through\b|and\b|or\b|if\b|unless\b|to\b|as\b|of\b)[a-z]", re.I)
#: The agent after "by" / "from" is not what is approved.
_AGENT_PHRASE = re.compile(r"\b(?:by|from)\s+(?:the\s+|an?\s+)?(?:[\w'’-]+\s+){0,4}?[\w'’-]+(?=\s*(?:,|\band\b|\bor\b|$))", re.I)
#: A subject's modifiers are not its core: "the use of the University's name in relation to any consultancy activity".
_SUBJECT_MODIFIER = re.compile(r"\s+(?:in\s+relation\s+to|related\s+to|regarding|concerning|with\s+respect\s+to|in\s+connection\s+with)\b.*$", re.I)
_ATTACH_FUNCTION_WORDS = frozenset({
    "is", "are", "be", "been", "being", "was", "were", "has", "have", "had", "unless", "if", "once", "when", "after", "upon", "following", "to", "subject",
    "obtain", "obtains", "obtained", "obtaining", "require", "requires", "required", "requiring", "need", "needs", "needed", "with", "without", "pending",
    "receive", "receives", "received", "receiving", "seek", "seeks", "sought", "get", "gets", "got", "the", "a", "an", "written", "prior", "final", "formal",
    "and", "or", "for", "by", "of", "its", "their", "his", "her", "any", "all", "no", "explicit", "official", "necessary", "relevant", "appropriate",
    "express", "must", "shall", "may", "will", "should", "can", "not", "them", "it", "they", "this", "these", "those", "which", "that", "who", "whose"})
_ATTACH_ADVERBS = frozenset({"explicitly", "formally", "duly", "also", "first", "then", "previously", "officially", "only", "specially", "specifically",
                             "further", "subsequently", "fully", "partially", "properly", "separately"})


def _subject_level(before: str) -> bool:
    stripped = before.rstrip()
    if not stripped or stripped[-1] in ",;:(":
        return True
    words = re.findall(r"[\w'’-]+", stripped.lower())
    while words and words[-1] in _ATTACH_ADVERBS:
        words.pop()
    return not words or words[-1] in _ATTACH_FUNCTION_WORDS


def _sentence_start(text: str, position: int) -> int:
    """Where the clause holding ``position`` starts: a sentence, or a coordinated
    clause after ", and" / ", or" ("..., and the approval of the Chancellor is
    required" approves nothing named before the comma)."""
    ends = [text.rfind(". ", 0, position) + 2, text.rfind("; ", 0, position) + 2, text.rfind(": ", 0, position) + 2,
            text.rfind(", and ", 0, position) + 6, text.rfind(", or ", 0, position) + 5]
    return max([e for e in ends if e > 5] or [0])


def _approval_targets(text: str) -> List[str]:
    """What each approval, review or decision in ``text`` is about, read inside
    its own sentence: the object after an active verb; the noun right before a
    passive form; after a function word, the sentence subject without its
    modifiers and without the agent ("by the VCRGS"). An adjective ("approved
    research projects") is no approval at all."""
    targets: List[str] = []
    for m in _ACTIVE_APPROVAL.finditer(text):
        tail = re.split(r"[.;]", text[m.end():], maxsplit=1)[0]
        targets.append(" ".join(tail.split()[:15]))
    for m in _PASSIVE_APPROVAL.finditer(text):
        if m.group(0).lower() in ("approved", "appointed", "authorized", "authorised", "endorsed", "nominated", "signed") and _ADJECTIVAL_NEXT.match(text[m.end():]):
            continue
        before = text[_sentence_start(text, m.start()):m.start()]
        if _subject_level(before):
            target = _SUBJECT_MODIFIER.sub("", _AGENT_PHRASE.sub(" ", before))
        else:
            target = " ".join(before.split()[-4:])
        of_phrase = re.match(r"\s*(?:of|for|to|on|regarding|concerning|whether)\s+((?:\S+\s*){1,10})", text[m.end():])
        if of_phrase:
            target += " " + of_phrase.group(1)
        targets.append(target)
    return targets


#: Eligibility wording whose object FOLLOWS it ("entitled to a petty cash
#: advance", "may apply for a research advance"); any other eligibility
#: wording (conditions, "must hold a Ph.D.") is about the sentence subject.
_ELIGIBILITY_FOR = re.compile(r"entitled|(?:may|can)\s+apply|eligib\w*(?=\s+(?:for|to)\b)|qualif\w*(?=\s+for\b)", re.I)


def _eligibility_targets(text: str) -> List[str]:
    targets: List[str] = []
    for m in _ELIGIBILITY_EVIDENCE.finditer(text):
        after = " ".join(re.split(r"[.;]", text[m.end():], maxsplit=1)[0].split()[:14])
        if _ELIGIBILITY_FOR.match(text, m.start()):
            targets.append(after)
        else:
            sentence_start = max(text.rfind(". ", 0, m.start()), text.rfind("; ", 0, m.start()))
            targets.append(text[sentence_start + 1:m.start()] + " " + after)
    return targets


# ---- focus: the specific thing a question is about -------------------------
#: A slot that names the focus of a question: "a grant FOR time travel",
#: "research IN Antarctica", "attend WITHOUT presenting a paper".
_FOCUS_SLOT = re.compile(r"\b(?:for|in|on|about|regarding|without|after|towards?|made\s+with|created\s+with|using|to\s+(?:buy|purchase|pay\s+for|cover|fund|rent|hire))\s+"
                         r"((?:[\w'’-]+\s*){1,5})", re.I)
#: In a yes/no question, words of the clause that no research concept covers
#: are the thing asked about ("Is the seed grant RENEWABLE?", "Can I hire my
#: SPOUSE ...?"); these words carry the question and are never required.
_YES_NO_SKIP = frozenset({"need", "needs", "take", "make", "made", "get", "gets", "use", "uses", "have", "has", "had", "do", "does", "did", "go", "goes", "know",
                          "want", "apply", "applies", "allow", "allows", "allowed", "permitted", "require", "requires", "required", "include", "includes",
                          "cover", "covers", "covered", "receive", "receives", "offer", "offers", "provide", "provides", "pay", "pays", "count", "counts",
                          "possible", "able", "there", "any", "some", "someone", "anyone", "also", "still", "ever", "really", "just", "only", "even",
                          "true", "okay", "fine", "normal", "usual", "typically", "necessary", "mandatory", "compulsory", "obliged", "obligated", "necessarily",
                          "university", "uos", "faculty", "members", "member", "staff", "i", "me", "my", "we", "our", "you", "your", "they", "them", "their",
                          "he", "she", "his", "her", "it", "its", "be", "been", "being", "is", "are", "was", "were", "can", "could", "may", "might",
                          "will", "would", "should", "shall", "must", "not", "no", "yes", "the", "a", "an", "to", "of", "for", "in", "on", "at", "by",
                          "with", "from", "as", "and", "or", "if", "when", "whether", "this", "that", "these", "those", "before", "after", "during",
                          "without", "while", "until", "into", "between", "over", "than", "per", "about", "through", "within", "under", "upon", "such",
                          "other", "own", "more", "most", "same", "what", "which", "who", "how", "where", "why", "here", "then", "so", "very", "all",
                          "both", "either", "neither", "each", "every", "approval", "approved", "approve", "permission", "research", "researchers",
                          "researcher", "grant", "grants", "fund", "funds", "funded", "funding"})
#: An agent or status noun built on a research verb ("publishers", "memberships"); never "researchers" or "members".
_AGENT_NOUN = re.compile(r"^(?!research|member|invent)[a-z]{4,}(?:ers|ors|ships?)$")
_PRONOUN_VERB = re.compile(r"\b(?:i|we|you|they)\s+(?:still\s+|also\s+|ever\s+)?[a-z]+\b", re.I)
#: In a yes/no question, the object of "cover / include / license / review /
#: handle ...": "Does copyright protection cover computer software?".
_YES_NO = re.compile(r"^\s*(?:(?:and|also|so|but)\s+)?(?:does|do|did|is|are|was|were|can|could|may|might|will|would|should|shall|has|have)\b", re.I)
_COVER_OBJECT = re.compile(r"\b(?:cover|covers|include|includes|protect|protects|fund|funds|support|supports|pay\s+for|pays\s+for|licen[cs]e|licen[cs]es|"
                           r"handle|handles|review|reviews|accept|accepts|allow|allows|apply\s+to|applies\s+to|extend\s+to|extends\s+to)\s+((?:[\w'’-]+\s*){1,5})", re.I)
#: Named things a question may single out that carry no research concept.
_FOCUS_VOCABULARY = re.compile(r"\b(?:software|computer\s+programs?|cars?|vehicles?|drones?|trademarks?|databases?|datasets?|tuition|course\s+materials?|"
                               r"lecture\s+notes|textbooks?|books?|visitors?|post\s?docs?|post\s?doctoral|part\s+time)\b", re.I)
_FOCUS_BOUNDARY = frozenset({"and", "or", "but", "if", "when", "while", "is", "are", "was", "were", "be", "can", "could", "may", "might", "must", "should",
                             "shall", "will", "would", "do", "does", "did", "to", "that", "which", "who", "what", "how", "per", "at", "by", "from", "with",
                             "as", "than", "because", "so", "into", "under", "over", "during", "before", "within", "i", "we", "you", "they", "it"})
_FOCUS_SKIP = frozenset({"a", "an", "the", "my", "our", "your", "their", "his", "her", "its", "this", "that", "these", "those", "any", "every", "each", "some",
                         "all", "new", "special", "specific", "additional", "other", "extra", "certain", "such", "more", "own", "particular", "same",
                         "different", "various", "of", "use", "using", "used", "get", "getting", "local", "international", "external", "internal",
                         "national", "regional", "overseas", "abroad", "foreign", "annual", "yearly", "whole", "entire", "full", "first", "second",
                         "final", "progress", "current", "previous", "next", "complete", "request", "requests", "participation",
                         "attendance", "details", "detail", "issue", "issues", "question", "questions", "reason", "reasons", "help", "advice",
                         "guidance", "assistance", "advance", "them", "it", "this", "that", "these", "those", "touch", "for", "about", "with",
                         "permitted", "allowed", "related", "regarding", "concerning",
                         "year", "years", "semester", "semesters", "people", "person", "member", "members", "staff", "work", "works", "purpose",
                         "purposes", "case", "cases", "thing", "things", "type", "types", "kind", "kinds", "area", "areas", "field", "fields",
                         "level", "levels", "matter", "matters", "part", "parts", "example", "general", "one", "two", "three"}) | GENERIC_TERMS


def _uncovered_text(text: str, anchors_only: bool = False) -> str:
    """``text`` with its research concept phrases blanked (only the anchor
    phrases when ``anchors_only``)."""
    work = cue_text(text.lower())
    for pattern, _concept, anchor in _COMPILED_CONCEPTS:
        if anchor or not anchors_only:
            work = pattern.sub(lambda m: " " * (m.end() - m.start()), work)
    return work


_TOKEN = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")


def _focus_words(phrase: str) -> List[str]:
    """The content words of a focus slot, when the slot names something no
    research concept covers: "time travel" (time is uncovered, so travel
    counts too); "conference travel" and "international conferences" name
    nothing beyond the research topic."""
    head: List[str] = []
    tokens = _TOKEN.findall(cue_text(phrase.lower()))
    if len(tokens) >= 2 and tokens[0] in ("i", "we", "you", "they", "he", "she"):
        return [tokens[1]] if tokens[1] not in _FOCUS_SKIP and len(tokens[1]) >= 3 else []   # "after I LEAVE", "after they RETIRE"
    for token in tokens:
        if token in _FOCUS_BOUNDARY:
            break
        head.append(token)
    phrase_text = " ".join(head)
    uncovered = set(_TOKEN.findall(_uncovered_text(phrase_text)))
    beyond_anchor = set(_TOKEN.findall(_uncovered_text(phrase_text, anchors_only=True)))
    words = [t for t in head if t in beyond_anchor and t not in _FOCUS_SKIP and not t.isdigit() and len(t) >= 3]
    if not any(w in uncovered for w in words):
        return []
    content = [w for w in words if not w.endswith("ing")]
    return (content or words)[:3]


def focus_terms(clause: str) -> List[str]:
    """The specific things a clause singles out that no research concept
    covers ("time travel", "a new car", "computer software"). The evidence
    must name them; topic relevance alone never answers them."""
    # the navigation half of a clause ("... and who should I contact about them?") names no research focus
    text = " ".join(seg for seg in clause_segments(clause) if not _navigation_segment(seg)) or cue_text(clause)
    phrases = [m.group(1) for m in _FOCUS_SLOT.finditer(text)]
    if _YES_NO.match(text):
        phrases += [m.group(1) for m in _COVER_OBJECT.finditer(text)]
    stems: List[str] = []
    for phrase in phrases:
        for word in _focus_words(phrase):
            stem = _stem(word)
            if stem not in stems:
                stems.append(stem)
    for m in _FOCUS_VOCABULARY.finditer(text):
        words = m.group(0).lower().split()
        stem = " ".join(words) if len(words) > 1 else _stem(words[0])
        if stem not in stems:
            stems.append(stem)
    if _YES_NO.match(text) and not stems:
        # with no slot or object focus, the property or subject a yes/no question asks about ("taxable", "renewable", "spouse", "adjunct")
        # only the topic's anchor phrases are blanked: a vocabulary word ("renewable", "paid", "memberships") can be the property itself;
        # an agent noun an anchor swallows ("PUBLISHERS" in publish\w*) is kept
        # words of a slot phrase the slot rule already judged topic-only ("for conference travel") stay out
        rest = _FOCUS_SLOT.sub(" ", text)
        rest = _PRONOUN_VERB.sub(" ", rest)                              # "Can I HIRE ...": the asked action, not a property
        agents = [t for t in _TOKEN.findall(text.lower()) if _AGENT_NOUN.match(t)]         # the slot rule drops these as covered
        loose = [t for t in _TOKEN.findall(_uncovered_text(rest, anchors_only=True)) + agents
                 if t not in _YES_NO_SKIP and t not in _FOCUS_SKIP and not t.isdigit() and len(t) >= 3]
        content = [t for t in loose if not t.endswith(("ed", "ing"))] or loose
        for word in content:
            stem = _stem(word)
            if stem not in stems and not any(stem in s.split() for s in stems):
                stems.append(stem)
    return stems[:5]


def _focus_present(text: str, stem: str) -> bool:
    if " " in stem:                                       # a multi-word term is matched as a phrase ("part time")
        return bool(re.search(r"\b" + r"[\s-]+".join(map(re.escape, stem.split())) + r"s?\b", text, re.I))
    return _has_word(text, stem)


def _focus_check(stems: Sequence[str]) -> Callable[[str], bool]:
    return lambda unit: all(_focus_present(cue_text(unit), s) for s in stems)


#: (family, label, clause pattern, evidence check). The quantity check is
#: built per clause by the shared ``requested_details``.
_DETAIL_FAMILIES: Tuple[Tuple[str, str, str, Callable[[str], bool]], ...] = (
    # "how much time / notice" asks for a limit or a lead time, not money
    ("amount", "amount or cost", r"\b(?:how\s+much(?!\s+(?:of\s+(?:my|their|his|her)\s+)?(?:time|notice|in\s+advance))|amounts?|fees?|costs?|price|per\s+diem|rates?|allowances?|ceiling|"
     r"reimburs\w*|funding\s+(?:amount|limit|cap))\b", _amount_ok),
    ("penalty", "penalty or consequence", r"\b(?:penalt(?:y|ies)|consequences?|sanctions?|disciplinary|what\s+happens\s+(?:if|when)|punish\w*)\b",
     lambda u: bool(_RESEARCH_PENALTY_EVIDENCE.search(u))),
    # "when ..." asks for any stated timing; explicit deadline wording asks for an end point (see requested_details)
    ("deadline", "deadline or date", r"\b(?:deadlines?|due\s+dates?|by\s+when|when\s+(?:is|are|must|do|does|should|will|can|did|may|could|shall|would|was|were|has|"
     r"have|might)|last\s+(?:day|date)|latest\s+date|final\s+date|end\s+date|closing\s+dates?|cut[\s-]?off|what\s+date|which\s+date|by\s+what\s+date|"
     r"application\s+(?:window|period)|until\s+when|(?:in\s+)?(?:which|what)\s+(?:month|week|semester|term|day|year)|what\s+time\s+of\s+(?:the\s+)?year|"
     r"at\s+what\s+(?:point|time|stage)|how\s+soon|due(?!\s+to))\b", _timing_ok),
    ("advance", "how far in advance", r"\bhow\s+(?:early|far\s+in\s+advance|long\s+before|much\s+in\s+advance|soon\s+before|much\s+notice)\b|"
     r"\bhow\s+many\s+(?:days|weeks|months)\s+(?:before|in\s+advance|ahead|prior)\b|\bin\s+advance\b|\blead\s+time\b",
     lambda u: bool(_ADVANCE_EVIDENCE.search(u))),
    ("quantity", "number or amount", r"\b(?:how\s+many|number\s+of|count\s+of)\b", lambda u: False),
    ("duration", "duration", r"\b(?:how\s+long|duration|for\s+how\s+many\s+(?:years|months|days|weeks)|term\s+of\s+office|validity|"
     r"how\s+many\s+(?:years|months|days|weeks))\b", _duration_ok),
    ("frequency", "frequency", r"\b(?:how\s+often|how\s+frequently|how\s+many\s+times|times\s+(?:a|per)\s+year|per\s+year|each\s+year|every\s+year)\b", _frequency_ok),
    ("minimum", "minimum", r"\b(?:minimum|minimal|at\s+least|fewest|smallest|lowest)\b", _min_ok),
    ("maximum", "maximum or limit", r"\b(?:maximum|maximal|at\s+most|upper\s+limit|limits?|caps?|capped|no\s+more\s+than|largest|up\s+to\s+how|"
     r"how\s+much\s+(?:of\s+(?:my|their|his|her)\s+)?time\s+(?:can|may|could|am|is|are|should))\b", _max_ok),
    # possessives are part of the distributed thing ("How is the University's share of licensing revenue distributed?")
    ("percentage", "percentage or share", r"\b(?:percent(?:age)?s?|%|what\s+share|what\s+proportion|how\s+(?:is|are)\s+[\w\s'’-]{0,60}?(?:distributed|split|shared|divided|allocated)|"
     r"(?:royalty|revenue|income|licensing|inventors?|university)(?:'s|s'|’s)?\s+shares?|shares?\s+(?:for|of|to)\s+(?:the\s+)?(?:inventors?|university|college|department))\b",
     _percentage_ok),
    ("authority", "approving authority", r"^\s*(?:(?:and|also|so|but)\s+)?(?:does|do|can|will|would|must|may)\s+(?:the\s+)?[\w'’-]+(?:\s+[\w'’-]+){0,3}?\s+"
     r"(?:approve|authori[sz]e|sign|decide|endorse)\b|\b(?:is|are)\s+[\w\s'’-]{1,40}?\s+approved\s+by\b|"
     r"\b(?:who\s+(?:approves?|authori[sz]es?|signs?|decides?|determines?|sets?|grants?|must\s+approve|can\s+approve|gives?\s+approval|"
     r"appoints?|nominates?|elects?|selects?|designates?|names?|"
     r"reviews?|recommends?)|approved\s+by\s+whom|whose\s+approval|which\s+(?:office|body|council|committee|board|person|dean)\s+(?:approves?|reviews?|decides?|"
     r"determines?|authori[sz]es?)|"
     r"approval\s+(?:from|of|by)\s+whom|who\s+is\s+the\s+approving|what\s+body\s+approves?)\b", _authority_ok),
    ("responsibility", "responsible party or unit",
     r"\bwho\s+(?:manages|supports|maintains|administers|runs|operates|provides|handles|oversees|coordinates|looks\s+after|is\s+responsible\s+for|is\s+in\s+charge\s+of)\b|"
     r"\bwhich\s+(?:office|unit|department|body)\s+(?:handles|manages|is\s+responsible|coordinates|administers)\b", _responsibility_ok),
    ("eligibility", "eligibility conditions", r"\b(?:who\s+is\s+eligible|am\s+i\s+eligible|eligib\w*|can\s+i\s+apply|who\s+(?:may|can)\s+apply|entitled\s+to|"
     r"qualify\s+for|requirements?\s+to\s+apply|who\s+qualifies)\b", _eligibility_ok),
    ("documents", "required documents", r"\b(?:what\s+documents?|which\s+documents?|attachments?|paperwork|what\s+(?:must|do|should)\s+i\s+(?:submit|attach|include|provide|upload)|"
     r"required\s+documentation|supporting\s+documents?|documentation\s+(?:is\s+)?required|what\s+documentation)\b", _documents_ok),
    ("project_type", "research category or project type", r"\b(?:(?:which|what)\s+(?:categor(?:y|ies)|types?|kinds?)\s+of|categories\s+of|types\s+of\s+(?:grants?|projects?|research|funding))\b",
     lambda u: bool(_PROJECT_TYPE_EVIDENCE.search(u))),
    ("funding_source", "funding source", r"\b(?:who\s+funds|funding\s+sources?|sources?\s+of\s+funding|internal\s+or\s+external|external\s+funders?|which\s+(?:sponsors?|funders?))\b",
     lambda u: bool(_FUNDING_SOURCE_EVIDENCE.search(u))),
    ("role", "the responsible role", r"\b(?:what\s+(?:is|are)\s+the\s+(?:role|duties|responsibilities)\s+of|whose\s+role|which\s+role|"
     r"what\s+does\s+(?:a|the)\s+(?:principal\s+investigator|pi|co-?\s?investigator|research\s+assistant|coordinator|chief\s+editor|inventor)\s+(?:do|have\s+to\s+do))\b",
     _role_ok),
    ("publication_type", "publication type", r"\b(?:which\s+journals?|what\s+(?:publications?|journals?|papers?)\s+(?:count|are\s+(?:recogni[sz]ed|accepted|admissible|accredited|considered))|"
     r"class\s+\(?[ab]\)?\s+journals?|scopus|types?\s+of\s+publications?|what\s+counts\s+as\s+(?:a\s+)?publication)\b", lambda u: bool(_PUBLICATION_TYPE_EVIDENCE.search(u))),
    ("requirement", "an explicit requirement statement about the subject",
     r"^\s*(?:(?:and|also|so|but)\s+)?(?:do|does|will|would)\s+(?:i|we|you|faculty(?:\s+members?)?|researchers?|one)\s+(?:need|have\s+to|require)\b|"
     r"^\s*(?:(?:and|also|so|but)\s+)?(?:is|are)\s+(?:an?\s+|the\s+)?[\w\s'’-]{1,40}?\s+(?:required|mandatory|compulsory|necessary|needed|obligatory)\b|"
     r"^\s*(?:(?:and|also|so|but)\s+)?(?:must|should)\s+(?:i|we|faculty(?:\s+members?)?|researchers?)\b|\bwhat\s+must\b|\bwhat\s+should\s+(?:\w+\s+){0,2}(?:include|contain|cover|provide)\b|\bwhat\s+is\s+required\b|\bwhat\s+are\s+the\s+requirements\b|"
     r"\bwhat\s+are\s+(?:the\s+|my\s+|its\s+|your\s+|our\s+)?(?:[\w'/-]+\s+){1,4}?requirements?\b|\brequirements?\s+for\b|\bmust\s+(?:contain|include)\b|"
     r"\bwhat\s+information\s+(?:must|should)\b|\bwhat\s+(?:do|does)\s+\w+\s+(?:need|have)\s+to\s+(?:include|contain|provide)\b", lambda u: bool(_OBLIGATION.search(u))),
    ("identity", "the named person",
     r"^\s*(?:(?:and|also)\s+)?who\s+(?:is|are)\s+(?:the\s+)?(?:current\s+|present\s+)?(?:chief\s+editors?|editors?\s+in\s+chief|editorial\s+secretar(?:y|ies)|director|head|"
     r"chair(?:person)?|dean|vice\s+chancellor|vcrgs|coordinator|members?\s+of|moderator)\b",
     lambda u: bool(re.search(r"\b(?:dr|prof|professor|mr|ms|mrs|eng)\.?\s+[A-Z][a-z]+", u))),
    # "who owns ...": an ownership or assignment statement about the asked thing, never a topically related line
    ("ownership", "ownership or rights holder",
     r"\bwho\s+owns\b|\bwho\s+(?:holds|has|keeps|retains)\s+(?:the\s+)?(?:rights?|ownership|copyright)\b|\bwhose\s+(?:property|ip|invention|copyright)\b|"
     r"\bownership\s+of\b|\bdoes\s+(?:the\s+)?(?:university|uos|faculty|inventor|author|creator)\s+(?:own|claim|retain|keep)\b|"
     r"\bdo\s+(?:i|we|faculty|students|inventors|authors)\s+(?:own|keep|retain)\b|\bbelongs?\s+to\b|\bclaims?\s+(?:the\s+)?(?:ownership|copyright|rights?)\b|"
     r"\bis\s+(?:it\s+)?owned\s+by\b|\bwho\s+keeps\s+the\s+rights?\b", lambda u: bool(_OWNERSHIP_EVIDENCE.search(u))),
    # "can I / may visitors / is it allowed ...": a statement that permits, conditions or prohibits the asked thing
    ("permission", "a statement permitting or prohibiting it",
     r"^\s*(?:(?:and|also|so|but)\s+)?(?:can|may|could)\s+(?:i|we|faculty(?:\s+members?)?|(?:a|an|the|my|our)\s+[\w-]+|(?:[\w-]+\s+)?[\w-]+s)\b|"
     r"\bam\s+i\s+(?:allowed|permitted)\b|\bis\s+(?:it|\w+)\s+(?:allowed|permitted|possible)\b|\bare\s+[\w\s-]{1,30}?(?:allowed|permitted)\b|"
     r"\bis\s+[\w\s-]{1,30}?(?:allowed|permitted)\b", lambda u: bool(_PERMISSION_EVIDENCE.search(u))),
)
_COMPILED_FAMILIES = tuple((f, label, re.compile(p, re.I), check) for f, label, p, check in _DETAIL_FAMILIES)

#: Research categories a clause may name; evidence must name the same one.
CATEGORY_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("seed projects", r"\bseed\b"),
    ("competitive projects", r"\bcompetitive\b"),
    ("collaborative research", r"\bcollaborative\b"),
    ("targeted research", r"\btargeted\b"),
    ("internal funding", r"\binternal(?:ly)?\b"),
    ("external funding", r"\bexternal(?:ly)?\b"),
    ("visiting researchers", r"\bvisiting\b"),
)
_COMPILED_CATEGORIES = tuple((label, re.compile(p, re.I)) for label, p in CATEGORY_GROUPS)


def requested_details(clause: str) -> List[Detail]:
    """The details a clause explicitly asks for, in a fixed order (shared rule
    with Research families, categories, count nouns and counted subject).

    Two precedence rules follow the interrogative, not the nouns: a party
    question ("Who determines the funding ceiling?") asks for the party, so
    value families triggered only by the object noun are dropped unless a
    value interrogative is present too ("who approves it and how much is
    it?"); and a clause that asks for a percentage does not also ask for a
    money amount ("the maximum research advance as a percentage")."""
    details = _details.requested_details(clause, _COMPILED_FAMILIES, _COMPILED_CATEGORIES, concepts,
                                         category_label="the requested research category (%s)",
                                         subject_after=_RESEARCH_SUBJECT_AFTER, unit_noun=_RESEARCH_UNIT_NOUN)
    if _PARTY_QUESTION.search(clause) and not _VALUE_INTERROGATIVE.search(clause):
        details = [d for d in details if d.family not in _VALUE_FAMILIES]
    if any(d.family == "percentage" for d in details):
        details = [d for d in details if d.family != "amount"]
    if any(d.family == "advance" for d in details):                 # "how early" asks for a lead time, not a date or a count
        details = [d for d in details if d.family not in ("deadline", "duration", "quantity")]
    text = cue_text(clause)
    if _YES_NO.match(text) and not re.search(r"\bhow\s+(?:much|many|long|often)\b", text, re.I):
        # "Is the per diem taxable?" asks whether, not how much: value families triggered by a noun are dropped
        details = [d for d in details if d.family not in ("amount", "minimum", "maximum", "quantity", "duration", "percentage")]
    if _EXPLICIT_DEADLINE.search(text):                             # a deadline is an end point; a window start is not
        details = [Detail(d.family, d.label, _deadline_end_ok) if d.family == "deadline" else d for d in details]
    if not re.search(r"\brecommend", text, re.I):                   # the recommending body is not the approver
        details = [Detail(d.family, d.label, (lambda base: (lambda u: base(_RECOMMENDER.sub(" ", u))))(d.check)) if d.family == "authority" else d
                   for d in details]
    if re.search(r"\b(?:deduct\w*|shares?|portion|proportion|withh\w*)\b", text, re.I):
        details = [Detail(d.family, d.label, lambda u: _amount_ok(u) or _percentage_ok(u)) if d.family == "amount" else d for d in details]
    action = re.search(r"\b(?:responsible\s+for|in\s+charge\s+of)\s+([a-z]{3,})ing\b", text, re.I)
    if action:                                                      # "responsible for APPROVING": the action must be stated too
        stem = action.group(1)
        details = [Detail(d.family, d.label, (lambda base, st: (lambda u: base(u) and bool(re.search(r"\b" + st + _NOT_ADJECTIVAL, u, re.I))))(d.check, stem))
                   if d.family == "responsibility" else d for d in details]
    details = [Detail(d.family, d.label, (lambda base: (lambda u: base(u) or bool(re.search(r"\bannual(?:ly)?\b|\byearly\b", u, re.I))))(d.check))
               if d.family == "frequency" else d for d in details]
    details = [_with_quantity_head(d, clause) if d.family == "quantity" else d for d in details]
    role = re.search(r"\b(?:role|duties|responsibilities|functions?)\s+of\s+(?P<who>[^?.,;]{1,60})", text, re.I)
    if role and not _ROLE_EVIDENCE.search(role.group("who")):
        # "the role of the TTO" asks what a body does (an overview), not for a person's duty (Step 4.5A-R)
        details = [d for d in details if d.family != "role"]
    locality = conference_locality(clause)
    if _CONFERENCE_CONTEXT.search(text):                            # "external conference" is a locality, not external funding
        details = [d for d in details if not (d.family == "category" and d.label.endswith(("(internal funding)", "(external funding)")))]
    if locality == "local":
        details.append(Detail("category", "the requested conference locality (local)", _local_ok))
    focus = focus_terms(clause)
    if focus:
        details.append(Detail("focus", "the specific focus asked about (%s)" % ", ".join(focus), _focus_check(focus)))
    return [_bind(d, clause, locality) for d in details]


def _with_quantity_head(detail: Detail, clause: str) -> Detail:
    """A count must name the counted thing: when the counted subject carries no
    specific research concept ("research drones"), its head noun must appear
    in the unit, so "at least three researchers" never counts drones."""
    m = _RESEARCH_SUBJECT_AFTER.search(clause)
    words = re.findall(r"[a-z][\w'’-]*", m.group(1).lower()) if m else []
    if not words or any(concepts(w).vocab - {"research"} for w in words):
        return detail
    head = next((w for w in words if not concepts(w).vocab and w not in _FOCUS_SKIP), None)
    if head is None:
        return detail
    stem = _stem(head)
    return Detail(detail.family, detail.label, lambda u, base=detail.check: base(u) and _has_word(u, stem))


@dataclass(frozen=True)
class _BoundDetail(Detail):
    """A requested detail bound to what the clause asks about. ``value`` finds
    the detail in a unit; ``bind`` checks that the unit, or the unit with its
    list lead-in, is about the asked object (and never names only the other
    conference locality). ``check`` is both, on the unit alone."""

    value: Optional[Callable[[str], bool]] = None
    bind: Optional[Callable[[str], bool]] = None
    unbound: bool = False                                           # the clause names no specific object, receiver or creator
    anchors: Tuple[str, ...] = ()                                   # then, the clause's own context words ("research grants")


#: Families whose value must belong to the asked object.
_BOUND_FAMILIES = frozenset({"amount", "penalty", "deadline", "advance", "quantity", "duration", "frequency", "minimum", "maximum", "percentage",
                             "authority", "responsibility", "eligibility", "ownership", "permission"})
_TIMING_FAMILIES = frozenset({"deadline", "advance"})
_FAMILY_PATTERN = {f: p for f, _label, p, _check in _COMPILED_FAMILIES}


def _segment_for(family: str, clause: str) -> str:
    """The interrogative segment of ``clause`` that asks for ``family``."""
    pattern = _FAMILY_PATTERN.get(family)
    if pattern is not None:
        for segment in clause_segments(clause):
            if pattern.search(segment):
                return segment
    return clause


#: Research categories that single out one kind of grant; a unit about exactly
#: one of them states that category's rule, not a general one.
_SINGLE_CATEGORY = tuple((label, re.compile(p, re.I)) for label, p in (
    ("seed", r"\bseed\b"), ("competitive", r"\bcompetitive\b"), ("collaborative", r"\bcollaborative\b"),
    ("targeted", r"\btargeted\b"), ("visiting", r"\bvisiting\b|\bvisitors?\b")))
_EXCEPTION = re.compile(r"\b(?:except(?:\s+for|\s+in|\s+when)?|other\s+than|excluding)\b[^.;]*|\bnot\s+(?:a\s+|an\s+)?visitors?\b", re.I)


def _category_specific(segment: str) -> Callable[[str], bool]:
    """For a clause that names no category: True when ``text`` states the rule of
    exactly one category ("grants to host visiting researchers, subject to Board
    or VCRGS approval") and so does not answer the general question. A category
    named only as an exception ("except for seed projects") does not count."""
    if any(p.search(segment) for _label, p in _SINGLE_CATEGORY) or any(p.search(segment) for _label, p in _COMPILED_CATEGORIES):
        return lambda text: False

    def specific(text: str) -> bool:
        stripped = _EXCEPTION.sub(" ", cue_text(text))
        return sum(1 for _label, p in _SINGLE_CATEGORY if p.search(stripped)) == 1

    return specific


def _bind(detail: Detail, clause: str, locality: Optional[str]) -> Detail:
    if detail.family not in _BOUND_FAMILIES:
        return detail
    segment = _segment_for(detail.family, clause)
    binding = _binding(segment, timing=detail.family in _TIMING_FAMILIES)
    category_specific = _category_specific(segment)
    if binding.empty:                                               # nothing named to bind to: the value alone decides
        attached: Callable[[str], bool] = lambda text: True
    elif detail.family == "authority":
        attached = lambda text: any(binding.names(t) for t in _approval_targets(text))
    elif detail.family == "eligibility":
        attached = lambda text: any(binding.names(t) for t in _eligibility_targets(text))
    else:
        attached = binding.names

    value = detail.check
    subject, action = _timing_subject(clause) if detail.family in _TIMING_FAMILIES else ((), None)

    def bind(text: str) -> bool:
        if locality == "local" and not _LOCAL_WORDS.search(cue_text(text)):
            return False                                             # a local value must be stated for local conferences
        if detail.family in _TIMING_FAMILIES:
            # a time binds to the thing that is due in the SAME sentence: "until the end of April" belongs to the
            # applications, not to the seed announcement that follows it in the item
            return any(value(sentence) and attached(sentence) and not category_specific(sentence) and not locality_conflict(locality, sentence)
                       and all(_has_word(cue_text(sentence), w) for w in subject) and _action_ok(sentence, action, value)
                       for sentence in _sentences(text))
        if category_specific(text):
            return False                                             # another category's rule never answers a general question
        return attached(text) and not locality_conflict(locality, text)

    return _BoundDetail(detail.family, detail.label, lambda unit: value(unit) and bind(unit), value=value, bind=bind,
                        unbound=not (binding.specific or binding.recipients or binding.creators), anchors=_context_anchors(segment))


def _context_anchors(segment: str) -> Tuple[str, ...]:
    """The words a clause uses for its context objects, each with the modifier
    right before it ("a faculty RESEARCH PROJECT", "SEED PROJECT"). Generic
    submission objects (request, application, report, proposal) are left out."""
    text = cue_text(segment).lower()
    words: List[str] = []
    modified = False
    for _name, clause_pattern, _evidence, is_specific, submission in _COMPILED_OBJECTS:
        if is_specific or submission:
            continue
        for m in clause_pattern.finditer(text):
            before = _TOKEN.findall(text[:m.start()])
            found = [m.group(0).split()[-1]]
            modifier = before[-1] if before else ""
            if (modifier and modifier not in _FOCUS_SKIP and modifier not in _TIMING_SUBJECT_STOP and len(modifier) >= 3
                    and modifier not in ("what", "which", "how", "who", "is", "are", "much", "many", "does", "do")):
                found.append(modifier)
            words.extend(_stem(w) for w in found if _stem(w) not in words)
            modified = modified or len(found) > 1
    # "a project" names no kind of project; only a modified context ("research grants", "seed project") can anchor an
    # orphan lettered item (Step 4.5A-R)
    return tuple(words) if modified else ()


#: The asked action, not its adjective: "approved research projects" states no approving.
_NOT_ADJECTIVAL = r"(?!ed\s+(?!by\b|and\b|or\b)[a-z])"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9•o(])")
#: The thing a timing question asks about: "the deadline for submitting the
#: conference FLIER", "When must adverse EVENTS be reported", "When must I
#: submit the ABSTRACT for conference funding".
_TIMING_SUBJECT = (
    re.compile(r"\b(?:deadlines?|due\s+date|closing\s+date|last\s+(?:day|date)|latest\s+date|final\s+date|end\s+date|cut\s?off)\s+(?:for|to|of)\s+"
               r"((?:[\w'-]+\s*){1,7})", re.I),
    re.compile(r"\bwhen\s+(?:is|are|must|should|do|does|can|will|shall|may|could|would|was|were)\s+(?:the\s+|a\s+|an\s+|my\s+|our\s+)?((?:[\w'-]+\s+){0,4}?[\w'-]+)\s+"
               r"(?:be\s+)?(submitted|due|sent|filed|paid|approved|announced|made|decided|reported|returned|settled|closed|published|provided|received)\b", re.I),
    re.compile(r"\b(?:when|by\s+when|how\s+early)\s+(?:must|should|do|can|may|could|shall)\s+(?:i|we|faculty|researchers?|inventors?)\s+"
               r"(submit|send|file|return|report|provide|request|apply\s+for|publish|disclose|settle|close)\s+((?:[\w'-]+\s*){1,7})", re.I),
)
#: Words that end a timing subject; "of" is kept ("evidence of the conference QUALITY").
_TIMING_SUBJECT_STOP = frozenset({"for", "to", "in", "on", "at", "with", "from", "by", "about", "under", "after", "before", "be", "is", "are", "and", "or",
                                  "within", "during", "if", "when", "so", "that", "which"})
_TIMING_SUBJECT_SKIP = frozenset({"submitting", "sending", "providing", "filing", "requesting", "reporting", "publishing", "announcing", "approving", "closing",
                                  "returning", "settling", "paying", "disclosing", "the", "a", "an", "my", "our", "of",
                                  "research", "university", "faculty", "project", "projects", "applications", "application", "submission", "submissions"})
#: A generic submission verb names no separate action; a specific one must be
#: the action the stated time governs ("By when are seed projects ANNOUNCED?").
_GENERIC_ACTIONS = frozenset({"submitted", "submit", "due", "sent", "send", "filed", "file", "made", "provided", "provide", "received", "request",
                              "apply for"})
_GERUND_ACTIONS = {"publishing": "publish", "announcing": "announced", "reporting": "report", "approving": "approved", "closing": "close",
                   "returning": "return", "settling": "settle", "paying": "paid", "disclosing": "disclose"}
_ACTION_STEMS = {"paid": "pa(?:id|y)", "approved": "approv", "announced": "announc", "decided": "decid|decision", "reported": "report",
                 "returned": "return", "settled": "settl", "closed": "clos", "published": "publish", "return": "return", "report": "report",
                 "publish": "publish", "disclose": "disclos", "settle": "settl", "close": "clos"}
#: Coordinate parts of a sentence ("... will be published in a Scopus-indexed outlet OR commitment to submit ... within six months").
_COORDINATE_SPLIT = re.compile(r",\s+|;\s+|\s+or\s+|\s+and\s+(?=(?:the\s+|a\s+|an\s+)?[a-z]+\s+(?:is|are|must|shall|will|may|can)\b)", re.I)


def _timing_subject(clause: str) -> Tuple[Tuple[str, ...], Optional[str]]:
    """What a timing question says is due: the plain words no research object
    already binds ("flier", "abstract", "acceptance letter", "evidence ...
    quality"), and the specific action asked about ("announced", "published"),
    each empty when the question names none."""
    text = cue_text(clause)
    for index, pattern in enumerate(_TIMING_SUBJECT):
        m = pattern.search(text)
        if not m:
            continue
        if index == 0:
            phrase = m.group(1)
            first = phrase.split()[0].lower() if phrase.split() else ""
            verb = _GERUND_ACTIONS.get(first)                        # "the deadline for PUBLISHING conference proceedings"
        elif index == 1:
            phrase, verb = m.group(1), m.group(2).lower()
        else:
            verb, phrase = " ".join(m.group(1).lower().split()), m.group(2)
        words: List[str] = []
        for token in phrase.split():
            low = token.lower().strip("'?.,")
            if low in _TIMING_SUBJECT_STOP:
                break
            if (low in _TIMING_SUBJECT_SKIP or (low in _FOCUS_SKIP and low != "quality") or len(low) < 3 or not _binding(low, timing=True).empty
                    or any(pattern.search(low) for _label, pattern in _COMPILED_CATEGORIES)):
                continue                                            # covered or generic words bind through the objects instead
            words.append(_stem(low))
        action = _ACTION_STEMS.get(verb) if verb and verb not in _GENERIC_ACTIONS else None
        return tuple(words[:3]), action
    return (), None


def _action_ok(sentence: str, action: Optional[str], value: Callable[[str], bool]) -> bool:
    """A specific action must sit in the coordinate part that states the time."""
    if action is None:
        return True
    return any(value(part) and re.search(r"\b(?:" + action + r")", part, re.I) for part in _COORDINATE_SPLIT.split(sentence))


def _sentences(text: str) -> List[str]:
    return [p for p in _SENTENCE_SPLIT.split(text) if p.strip()] or [text]


#: A lettered sub-item ("d. ..."): it continues a numbered item that names its subject (a numbered item or a bullet of
#: a section's own list stands on its own).
_SUB_ITEM = re.compile(r"^[a-z]\.\s+")


def _anchored(detail: "_BoundDetail", unit: str) -> bool:
    """An orphan lettered item names the clause's own context words itself."""
    text = cue_text(unit)
    return bool(detail.anchors) and all(_has_word(text, w) for w in detail.anchors)


def _lead_in(units: Sequence[str], index: int) -> Optional[str]:
    """The lead-in ("... including:") of the list that the item ``units[index]`` belongs to."""
    if not _ITEM_START.match(units[index]):
        return None
    j = index - 1
    while j >= 0 and _ITEM_START.match(units[j]):
        j -= 1
    return units[j] if j >= 0 and units[j].rstrip().endswith(":") else None


def unit_coverage(details: Sequence[Detail], clause: Concepts, quote: str) -> FrozenSet[str]:
    """The largest set of requested details that ONE local unit of ``quote``
    states (shared locality rule). A bound detail's value must be in the unit
    itself; the object it belongs to may be named by the unit or by the lead-in
    of the list the unit is an item of ("The University's share ... as
    follows: a. 40% to the VCRGS")."""
    units = _local_units(quote)
    best: FrozenSet[str] = frozenset()
    for index, unit in enumerate(units):
        if relevance(clause, unit)[1] < 1:
            continue
        stripped = _LIST_MARKER.sub("", unit)
        lead = _lead_in(units, index)
        context = stripped if lead is None else lead + " " + stripped
        # a lettered item quoted without its lead-in belongs to a list whose subject is unknown here ("d. The
        # funding ceiling for each project is AED 5000." continues the student-projects list of the previous page): a value
        # the clause cannot bind to a named object is not credited from it
        orphan = lead is None and bool(_SUB_ITEM.match(unit))
        got = frozenset(_details._detail_key(d) for d in details
                        if (d.value(stripped) and d.bind(context) and not (orphan and d.unbound and not _anchored(d, unit)) if isinstance(d, _BoundDetail)
                            else d.check(stripped)))
        if len(got) > len(best):
            best = got
    return best


def detail_covered(detail: Detail, clause: Concepts, quote: str) -> bool:
    return _evidence.detail_covered(detail, lambda text: relevance(clause, text), quote)


def quote_problem(quote: str, unit_kind: str, last_unit: bool = False) -> Optional[str]:
    return _evidence.quote_problem(quote, unit_kind, last_unit, table_row=is_table_row)


def select_quote(text: str, chunk_type: Optional[str], clause: Concepts,
                 allowed: Optional[Sequence[Tuple[int, int]]] = None,
                 details: Sequence[Detail] = ()) -> Optional[Tuple[str, int, int]]:
    """The best quotable unit of ``text`` for the clause (shared rule, Research
    relevance). Research's shared sections (7.2, 7.7, 6.4) and several
    policies (5.4.5, 8.11 item 12) state their content as a lead-in and a
    list, so a list item is quoted with the lead-in that introduces it when
    that makes the extract more relevant (``join_lead_in``)."""
    return _evidence.select_quote(text, chunk_type, lambda t: relevance(clause, t), allowed, details,
                                  max_words=MAX_QUOTE_WORDS, table_row=is_table_row, join_lead_in=True)


# ---------------------------------------------------------------------------
# Ownership and handoff cues (clause classification)
# ---------------------------------------------------------------------------
#: Positive research cues. A clause is Research's work only when one fires
#: on the research view of the clause (WLAM category phrases blanked).
_RESEARCH_CUES = re.compile(
    r"\b(?:research\w*|fund(?:ed|ing|s|ers?)?|seed\s+fund\w*|sponsor\w*|ethics|ethical|irb|iacuc|acuc|institutional\s+review\s+board|human\s+subjects?|"
    # "grant" as a verb ("who grants tenure", "grant a student an extension") and a grant of leave are not research grants
    r"(?<!who\s)(?<!to\s)(?<!\bi\s)(?<!we\s)grants?(?!\s+(?:tenure|leave|access|permission|approval|exemptions?|exceptions?|extensions?|an?\s|me\b|him\b|her\b|"
    r"them\b|students?\b|of\s+(?:leave|absence|tenure)))|"
    r"animal\s+(?:use|care|research)|intellectual\s+property|ip(?!\s+address)|patents?|technology\s+transfer|tto|commerciali[sz]\w*|licen[cs]\w*|royalt\w*|inventors?|"
    r"rfd|vcrgs|ovcrgs|rimhs|rihss|pi|rec(?!\s+(?:cent(?:er|re)|room|hall))|h\s?index|citations?|symposi(?:um|a)|congress(?:es)?|present(?:ing)?\s+a\s+paper|"
    r"open\s+access\s+(?:fees?|charges?|publication|publishing|journals?|articles?)|article\s+processing\s+charges?|page\s+charges?|"
    r"(?:book|author|authors|authored\s+book)\s+awards?|awards?\s+for\s+(?:authored\s+)?books|authored\s+books?|tangible\s+research\s+property|trp|"
    r"disclosure\s+evaluation|project\s+closure|closure\s+of\s+(?:a\s+|the\s+|my\s+)?(?:research\s+)?project|conflicts?\s+of\s+commitment|"
    r"inventions?|copyright\w*|institutes?|publications?|publish\w*|journals?|scientific\s+publishing|editorial|chief\s+editors?|editors?\s+in\s+chief|"
    r"scopus|consultanc\w*|consulting(?!\s+(?:with\s+)?(?:the\s+|my\s+|a\s+|an\s+)?(?:students?|colleagues?|chair|dean|advisors?|supervisors?|parents?|peers?)\b)|"
    r"paid\s+consultants?|misconduct|"
    r"plagiaris\w*|fabricat\w*|falsif\w*|conflicts?\s+of\s+interest|priority\s+areas?|conferences?|per\s+diem|sabbatical\w*|visiting\s+researchers?|"
    r"petty\s+cash|scholar\w*|principal\s+investigators?|co-?\s?investigators?|seed\s+(?:projects?|grants?|funding)|tech\s+transfer|"
    r"materials?\s+transfer\s+agreements?|mtas?|informed\s+consent|"
    r"(?<!student\s)(?<!capstone\s)(?<!course\s)(?:final|progress|annual)\s+(?:research|project)\s+reports?|project\s+(?:final|progress|annual)\s+reports?|"
    r"(?:university|uos)(?:'s|’s)?\s+share|editorial\s+secretary|"
    r"(?:how\s+many|maximum|max|limit|cap|simultaneous\w*|at\s+once|at\s+a\s+time)\b[^.?]{0,40}\b(?:thes[ie]s|dissertations?)\b(?!\s+supervision\s+(?:credits?|load|hours|workload))|"
    r"(?:thes[ie]s|dissertations?)\b(?!\s+supervision\s+(?:credits?|load|hours|workload))[^.?]{0,40}\b(?:how\s+many|maximum|limit|cap|simultaneous\w*|at\s+once|at\s+a\s+time)|"
    r"(?:how\s+many|maximum)\b[^.?]{0,30}\b(?:students?|candidates?|supervisees?)\b[^.?]{0,30}\bsupervis\w*|"
    r"supervis\w*\s+(?:of\s+)?(?:a\s+|the\s+|my\s+)?(?:research\s+(?:projects?|students?|work)|postgraduate|graduate\s+research|phd))\b",
    re.I,
)

#: Other-domain cues, phrase-level, navigation first so that a tie ("where
#: is the HR department") goes to Institutional as it does in Teaching.
#: Teaching: WLAM categories with load or credit words, supervision credit
#: and eligibility, teaching matters. HR: leave, employment, promotion and
#: evaluation process, incentives. Institutional: who to contact, where,
#: which office, portal support.
_OTHER_CUES: Tuple[Tuple[SpecialistId, "re.Pattern[str]"], ...] = (
    (SpecialistId.INSTITUTIONAL, re.compile(
        r"\b(?:who(?:m)?\s+(?:do|should|can|could|must|would)\s+i\s+(?:contact|ask|call|e-?\s?mail|talk\s+to|speak\s+to|reach|see|go\s+to|refer\s+to|approach)|"
        r"who\s+(?:is|are)\s+the\s+(?:contact|person|people)\s+(?:for|to)|contact\s+(?:person|details|information|numbers?)|phone|telephone|fax|e-?\s?mail\s+address|extension\s+number|"
        r"where\s+(?:is|are|can\s+i\s+find|do\s+i\s+find)\s+(?:the\s+)?[\w' ]{0,30}?(?:office|department|unit|cent(?:er|re)|building|desk|help\s*desk|library|clinic|institute|"
        r"registrar\w*|hr|human\s+resources|it\s+(?:services?|support)|reception|security|portal)|"
        r"who\s+(?:manages|supports|maintains|administers|runs|provides)\s+[\w' ]{0,40}?(?:support|help\s*desk|services?|systems?|portal|office|unit|department|cent(?:er|re)|desk)|"
        r"which\s+(?:office|unit|department|committee|council|body)\s+(?:handles?|is\s+responsible|deals|manages|approves|do\s+i)|organi[sz]ational\s+(?:chart|structure)|"
        r"governance|chain\s+of\s+command|board\s+of\s+trustees|university\s+council|help\s*desk|it\s+support|service\s+desk|portal\s+support|"
        r"(?:university|uos)(?:'s|’s)?\s+(?:mission|vision|core\s+values|strategic\s+plan)|mission\s+statement|core\s+values|accreditation|campuses|"
        r"degree\s+programs?|laborator(?:y|ies)\s+safety|lab\s+safety|safety\s+(?:rules|training|regulations|procedures|inspections?|requirements?|guidelines?|measures?|protocols?|standards?)|"
        r"health\s+and\s+safety)\b", re.I)),
    (SpecialistId.TEACHING, re.compile(
        r"\b(?:(?:research[- ]intensive|active[- ]research|research[- ]active|teaching[- ]track)\s+faculty|teaching\s+loads?|credit\s+hours?|contact\s+hours?|"
        r"wlam|workloads?|teaching\s+hours|supervis\w*\s+(?:credits?|load|hours|workload|eligib\w*|qualif\w*)|"
        r"(?:credits?|load|hours|workload|eligib\w*|qualif\w*|calculat\w*)\b[^.?]{0,25}\bsupervis\w*|supervis\w*\b[^.?]{0,25}\b(?:credits?|calculat\w*|eligib\w*|qualif\w*)|"
        r"academic\s+advising|student\s+advising|advis(?:e|ing)\s+students|syllab\w*|office\s+hours?|blackboard|\bexams?\b|grading|course\s+design|"
        r"teaching\s+development|professional\s+development|peer\s+observation|lesson\s+plans?|classroom)\b", re.I)),
    (SpecialistId.FACULTY_SERVICES, re.compile(
        r"\b(?:(?:annual|sick|maternity|paternity|emergency|unpaid|study|compassionate|casual)\s+leaves?|leave\s+(?:of\s+absence|entitlement|balance|request|application|days)|"
        r"days?\s+of\s+leave|vacation|sabbatical\w*|employment|contracts?|probation\w*|payroll|salar(?:y|ies)|pay\s?slips?|end\s+of\s+service|gratuity|pension|"
        r"benefits?|allowances?|housing|insurance|visa|passport|emirates\s+id|resign\w*|termination|promot(?:ion|ed)|rank|evaluation\s+(?:process|procedure|form|weights?|score|rubric)|"
        r"annual\s+evaluation|performance\s+evaluation|appraisal|incentives?|human\s+resources|\bhr\b|tenure|"
        # 16.5 (harassment) is outside the Research scope
        r"(?:sexual\s+)?harassment|discrimination|grievances?)\b", re.I)),
)

#: A clause that asks for a system or click procedure rather than a policy.
_PROCEDURE_CUES = re.compile(
    r"\b(?:step[- ]by[- ]step|clicks?|clicking|buttons?|screens?|menus?|tabs?|icons?|navigat\w*|dropdown|log\s?in|login|upload\s+field|"
    r"where\s+(?:do|can|should|must)\s+i\s+(?:click|find|go|press|select|upload)|which\s+(?:portal|screen|system|page|website)|"
    r"how\s+(?:do|can|should|would)\s+i\s+(?:upload|log|open|access|attach|import|export|download|edit|delete|fill\s+in|complete\s+the\s+form))\b", re.I)
_SYSTEM_CUES = re.compile(r"\b(?:research\s+portal|portal|online\s+system|my\s?uos|\bfis\b|faculty\s+information\s+system|banner|blackboard)\b", re.I)
#: A clause that asks for the policy process ("how do I apply"): the handbook
#: states stages, not the complete submission procedure, so at most partial.
_PROCESS_CUES = re.compile(r"\b(?:how\s+(?:do|can|should|would|to)\s+(?:i\s+)?(?:apply|submit|obtain|get|request|register|claim|file)|what\s+is\s+the\s+(?:process|procedure)|"
                           r"application\s+process|procedure\s+for|steps?\s+(?:to|for|involved)|how\s+is\s+[\w\s]{0,20}?(?:applied|submitted|obtained)|"
                           r"(?:walk|guide|take|talk)\s+(?:me|us)\s+through|"
                           r"(?:fill(?:ing)?\s+(?:in|out)|complet(?:e|ing))\s+(?:the\s+|a\s+|an\s+|my\s+|this\s+)?[\w\s'-]{0,40}?\bforms?|"
                           # click, screen and step requests name a procedure even when no system is named
                           r"step[- ]by[- ]step|(?:list|what\s+are|give\s+me|tell\s+me)\s+(?:the\s+|all\s+the\s+)?steps|steps?\s+(?:do|must|should|would)\s+i|"
                           r"what\s+steps|instructions?\s+(?:for|to|on)|sequence\s+of\s+(?:screens|steps|clicks)|"
                           r"(?:what|where)\s+(?:do\s+i|to|should\s+i)\s+(?:click|press|tap|select)|which\s+(?:button|menu(?:\s+option)?|icon|link|screen|tab|page|field)|"
                           r"what\s+(?:button|menu(?:\s+option)?|icon|link|url|screen|tab)\s+(?:do|should|can|must)\s+i\s+(?:use|click|press|open|select|follow)|"
                           r"click\w*|press\s+(?:the\s+)?(?:button|submit)|tap\s+(?:the\s+)?(?:icon|button))\b", re.I)
_AFTER_SUBMISSION = re.compile(r"\b(?:what\s+happens\s+after|after\s+(?:i\s+|my\s+)?(?:submi\w*|application)|review\s+(?:process|chain|stages?)|who\s+reviews|"
                               r"how\s+(?:is|are)\s+[\w\s]{0,20}?reviewed)\b", re.I)


#: Navigation and contact intent is Institutional Navigation's work: where
#: something is or where to go, how to reach someone, a contact channel,
#: which office handles a matter, portal or system support. A policy action
#: by a named body ("which committee reviews", "who approves", "who handles
#: research grants") is substantive research authority or responsibility,
#: never navigation. A system click procedure ("where do I click in the
#: portal") is a procedure request, not navigation.
_NAVIGATION_CUES = re.compile(
    # where something is (a "where" that asks for a place; "where does the University's share go" is policy)
    r"^\s*(?:(?:and|also|so|but|then|how\s+and)\s+)?where\s+(?:is|are|was|were|can|could|do|should|shall|would|will|to|r|'s)\b|"
    r"\bwhere\s+(?:is|are|can|could|do|should|shall|would|will|to)\s+(?:i|we|one|faculty|the|a|an|my|our|it|they)\b|"
    r"\bwhere\s+must\s+(?:i|we)\s+(?:submit|send|go|file|deliver|hand|take|apply)\b|"
    # common misspellings of "where is" and "contact"
    r"\b(?:were|wehre|whre|wher|hwere)\s+(?:is|are|can|do)\b|\bcon(?:tcat|atct|tatc|tacct|act)\b|"
    # location, building, floor, room, address
    r"\blocated\s*[?.!]?\s*$|\bwhere\b[^.?]{0,80}\blocated\b|\bis\s+(?:it|the\s+[\w\s'-]{1,40}?)\s+located\b|"
    r"(?<!conference\s)(?<!conferences\s)\blocations?\b(?!\s+of\s+(?:the\s+|a\s+|an\s+)?(?:[\w-]+\s+){0,2}conferences?)|"
    r"\b(?:what|which)\s+(?:floor|building|room|campus)\b|\bon\s+which\s+floor\b|\b(?:mailing|postal|office|physical|street|e\s?mail|email|web)\s+address\b|"
    r"\baddress\s+of\s+(?:the\s+)?[\w\s'-]{0,40}?(?:office|department|unit|cent(?:er|re)|institute|committee|board|tto|rfd|irb|iacuc)\b|\bwhat\s+is\s+the\s+address\b|"
    r"\bhow\s+(?:do|does|can|could|should|would)\s+(?:i|we|one)\s+(?:find|get\s+to|go\s+to|locate|visit)\s+(?:the\s+|a\s+|an\s+)?[\w\s'-]{0,40}?"
    r"(?:office|department|unit|cent(?:er|re)|building|institute|desk|room|committee|board|lab(?:oratory)?|tto|rfd|irb|iacuc|portal)\b|"
    r"\b(?:office|department|unit|desk|library|tto|rfd)\s+(?:is\s+)?open\s+(?:during|on|at|until)\b|\bopening\s+hours\b|\bwhereabouts\b|"
    # how to reach someone, and who to contact
    r"\bhow\s+(?:do|does|can|could|should|would|might|to)\s+(?:i\s+|we\s+|one\s+|faculty\s+)?(?:contact|reach|call|e\s?mail|get\s+in\s+touch|write\s+to|talk\s+to|speak\s+to)\b|"
    r"\b(?:best|quickest|fastest)\s+way\s+to\s+(?:contact|reach|get\s+in\s+touch)\b|"
    r"\bwho(?:m)?(?:'s)?\b[^.?]{0,30}\b(?:contact|get\s+in\s+touch|reach\s+out|talk\s+to|speak\s+to|approach)\b|"
    r"\bwho(?:m)?\s+(?:do|should|can|could|must|would)\s+(?:i|we)\s+(?:ask|call|e\s?mail|see|go\s+to|refer\s+to|write\s+to)\b|"
    r"\bto\s+whom\s+(?:should|do|can|must|may)\s+(?:i|we)\s+(?:send|address|submit|write|direct|forward|report)\b|"
    r"\b(?:which|what)\s+person\b[^.?]{0,20}\bcontact\b|\b(?:someone|somebody|anyone|anybody)\s+(?:i|we)\s+can\s+(?:contact|ask|talk\s+to|reach)\b|"
    r"\bcontact(?:s|ing)?\b(?!\s+(?:hours?|time|with\s+(?:human|participants?|subjects?|animals?|students?)))|"
    # contact channels
    r"\b(?:tele)?phone\s+(?:number|no|extension)s?\b|\bphone\s*[?.!]?\s*$|\b(?:e\s?mail|email)\s+(?:address|id)\b|\bfax\b|\bhotline\b|"
    r"\bwhat\s+is\s+(?:the|their|its|his|her)\s+(?:e\s?mail|email|phone|telephone|website|url|extension|hotline|number)\b|"
    r"\b(?:its|their|the\s+office's|the\s+department's)\s+(?:e\s?mail|email|phone|telephone|website|extension|number)\b|"
    r"\bextension\s+(?:numbers?|no)\b|\bextension\s*[?.!]?\s*$|\b(?:website|web\s?site|web\s+page|webpage)\s+(?:of|for)\b|\burls?\b|"
    r"\b(?:call|phone|e\s?mail|email)\s+(?:the|them|him|her|it)\b|"
    # which office handles a matter; portal or system support
    r"\b(?:which|what)\s+(?:office|unit|department|desk)\s+(?:handles?|deals?|is\s+(?:in\s+charge|responsible)|manages?|should|do|does|can|to\s+contact|takes?\s+care|looks?\s+after)\b|"
    r"\bto\s+which\s+(?:office|department|unit)\b|"
    r"\bwho\s+(?:manages|supports|maintains|administers|runs|operates)\s+[\w' ]{0,40}?(?:portal|systems?|websites?|help\s*desk|service\s+desk)\b|"
    r"\bportal\s+support\b|\bhelp\s*desk\b|\bit\s+support\b|\bservice\s+desk\b", re.I)
#: A clause is cut into segments at a conjunction or a question mark followed
#: by a question word ("What does the RFD do and how do I contact it?"), or at
#: "and" followed by a contact or location noun phrase ("... and its phone
#: number", "... and the contact person"), so a navigation half never hides a
#: substantive half.
_SEGMENT_SPLIT = re.compile(
    r"\s*(?:,\s*(?:and\s+|or\s+)?|;\s*|\s+and\s+(?:also\s+)?|\s+or\s+|\?\s*)"
    r"(?=(?:what|who|whom|whose|which|how|when|where|why|is|are|can|could|does|do|did|may|must|should|will|would)\b)|"
    r"\s*,?\s+and\s+(?=(?:(?:its|their|the|his|her|your|our)\s+)?(?:(?:tele)?phone|e\s?mail|email|fax|contact|location|address|office\s+location|"
    r"extension|website|hotline|whereabouts|opening\s+hours)\b)|"
    r"^\s*how\s+and\s+(?=where\b)", re.I)
#: A click, screen or upload procedure; with such a cue a "where" question
#: asks for a step in a system, not for a place.
_CLICK_CUES = re.compile(r"\b(?:click\w*|press|select|tap|buttons?|screens?|menus?|tabs?|icons?|dropdown|log\s?in|login|upload\w*|links?|fields?)\b", re.I)
#: Contact or location evidence: an e-mail address, a telephone number, a
#: room, building or floor, a web address. Research text that merely names a
#: body never supplies it, and it must describe the body the clause names.
_CONTACT_EVIDENCE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|\+\s?\d[\d\s()-]{6,}\d|\b(?:tel|phone|telephone|fax|ext)\.?\s*:?\s*\+?\d|"
    r"\b(?:room|office|building|floor|block)\s+(?:no\.?\s*)?[A-Z]?\d+\b|\bwww\.|https?://", re.I)
#: Bodies a contact or location question can name.
_BODY_ANCHORS = frozenset({"research_support", "technology_transfer", "irb", "iacuc", "research_board", "research_committee",
                           "research_institute", "research_centre", "research_group"})


def clause_segments(clause: str) -> List[str]:
    """The segments of a clause (the clause itself when it has one)."""
    parts = [p.strip(" ,;?") for p in _SEGMENT_SPLIT.split(cue_text(clause))]
    return [p for p in parts if p] or [clause.strip()]


def navigation_intent(clause: str, system_hint: Optional[str] = None) -> Tuple[bool, bool]:
    """(asks for navigation or contact, also asks something substantive). A
    segment that asks for a click, screen or upload step is a procedure
    request, not navigation; a "where can I find X" question is navigation
    whatever system the task names. ``system_hint`` is accepted for the
    caller's convenience and never turns a place question into a procedure."""
    nav = substantive = False
    text = cue_text(clause)
    for segment in clause_segments(clause):
        if _NAVIGATION_CUES.search(segment) and not _CLICK_CUES.search(segment):
            nav = True
        elif len(segment.split()) >= 2 or re.match(r"^\s*how\s*$", segment, re.I):
            substantive = True
    if re.match(r"^\s*how\s+and\s+where\b", text, re.I):
        nav = substantive = True                                    # "How and where do I submit ...": the stages, and the place
    return nav, substantive


def _navigation_segment(segment: str) -> bool:
    return bool(_NAVIGATION_CUES.search(segment) and not _CLICK_CUES.search(segment))


def _contact_check(clause: str) -> Callable[[str], bool]:
    """Contact or location evidence that describes the body the clause names."""
    bodies = concepts(clause).anchors & _BODY_ANCHORS

    def check(unit: str) -> bool:
        # no body the specialist can name: a channel or URL of something else never counts (Step 4.5A-R, NAV-07)
        return bool(bodies) and bool(_CONTACT_EVIDENCE.search(unit)) and bool(concepts(unit).anchors & bodies)

    return check


def _contact_ok(unit: str) -> bool:
    return bool(_CONTACT_EVIDENCE.search(unit))


_DOMAIN_LABEL = {SpecialistId.TEACHING: "Teaching", SpecialistId.FACULTY_SERVICES: "Faculty Services & HR"}
#: Institutional subjects that are not navigation (lab and health and safety rules): the Research evidence must state
#: them too, or they stay missing (Step 4.5A, SAF-2).
_INSTITUTIONAL_SUBJECT = re.compile(r"\b(?:laborator(?:y|ies)\s+safety|labs?\s+safety|safety\s+(?:rules?|training|regulations?|procedures?|manuals?|"
                                    r"requirements?|guidelines?|measures?|protocols?|standards?)|"
                                    r"health\s+and\s+safety)\b", re.I)


def _foreign_subject(clause: str, others: Sequence[SpecialistId]) -> Optional[Tuple[str, List[str]]]:
    """(owner label, phrases) of the Teaching or HR subject a shared clause
    names ("sick leave", "office hours", "promotion"), or None. Navigation is
    handled by the contact detail, never here."""
    text = cue_text(clause)
    for sid, pattern in _OTHER_CUES:
        if sid in others and sid in _DOMAIN_LABEL:
            phrases = []
            for m in pattern.finditer(text):
                phrase = re.sub(r"\s+", " ", m.group(0).strip().lower())
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
            if phrases:
                return _DOMAIN_LABEL[sid], phrases[:3]
    if SpecialistId.INSTITUTIONAL in others:
        phrases = []
        for m in _INSTITUTIONAL_SUBJECT.finditer(text):
            phrase = re.sub(r"\s+", " ", m.group(0).strip().lower())
            if phrase not in phrases:
                phrases.append(phrase)
        if phrases:
            return "Institutional Navigation", phrases[:3]
    return None


def _foreign_check(phrases: Sequence[str]) -> Callable[[str], bool]:
    patterns = [re.compile(r"\b" + r"[\s-]+".join(map(re.escape, p.split())) + r"s?\b", re.I) for p in phrases]
    return lambda unit: all(p.search(cue_text(unit)) for p in patterns)


def other_domains(clause: str) -> List[SpecialistId]:
    """Every other-domain specialist whose cue fires on the clause, strongest
    first (cue count, then Institutional, Teaching, Faculty Services & HR)."""
    clause = cue_text(clause)
    hits = [(len(pattern.findall(clause)), -order, sid) for order, (sid, pattern) in enumerate(_OTHER_CUES)]
    return [sid for _count, _order, sid in sorted((h for h in hits if h[0]), reverse=True)]


def classify_clause(clause: str) -> Tuple[bool, Optional[SpecialistId]]:
    """(carries a positive research cue, strongest other-domain specialist or
    None). The research cue is read on the research view of the clause: a
    WLAM category phrase never counts as research on its own."""
    # the shared ownership lexicon (specialists/domain_cues.py) is also read, so that a clause the Coordinator routes to
    # Research is never rejected here for a phrase this table lacks (Step 4.5A-R)
    research = bool(_RESEARCH_CUES.search(_research_view(cue_text(clause)))) or research_owned(clause)
    others = other_domains(clause)
    return research, (others[0] if others else None)


def system_named(text: str) -> Optional[str]:
    """The canonical system id named in ``text``, or None."""
    named = _SYSTEM_CUES.search(text)
    if not named:
        return None
    system = re.sub(r"[\s-]", "", named.group(0).lower())
    if system in ("researchportal", "portal", "onlinesystem"):
        return "portal"
    if system in ("fis", "facultyinformationsystem"):
        return "fis"
    return system


def is_procedure_request(clause: str, system_hint: Optional[str]) -> Tuple[bool, Optional[str]]:
    """(asks for a system or click procedure, system named). A procedure cue
    without a system named in the clause still counts when the task carries
    a system hint."""
    system = system_named(clause)
    if system is None and system_hint in SYSTEMS:
        system = system_hint
    procedural = bool(_PROCEDURE_CUES.search(clause)) and system is not None
    return procedural, system


def is_process_request(clause: str) -> bool:
    """A policy-process request ("how do I apply for a grant") that the
    handbook answers with stages only; "what happens after submission" is
    not one (the review chain can be stated completely)."""
    return bool(_PROCESS_CUES.search(clause)) and not _AFTER_SUBMISSION.search(clause)


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------
def compile_scope(section_map: SectionMap, source_id: str = APPROVED_SOURCE_ID,
                  owned: Sequence[Tuple[int, str, str]] = SCOPE_SECTIONS,
                  extra_pages: Sequence[Tuple[int, int]] = EXTRA_PAGES) -> RetrievalScope:
    """The Research retrieval scope: owned and shared section identities plus
    the chapter 9 preamble page (shared rule; fails loudly)."""
    return _scoping.compile_scope(section_map, source_id, owned, extra_pages)


# ---------------------------------------------------------------------------
# Resources (dependency injection; nothing is loaded by importing this module)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Ask shape: completeness fails safe (Milestone 4, Step 4.5A-R)
# ---------------------------------------------------------------------------
#: Before this step a clause was complete as soon as every RECOGNISED requested
#: detail was covered: an ask the detail families did not recognise ("What is
#: the timeframe for ...?", "Who stamps ...?") requested nothing, and topic
#: relevance alone made it `supported` (fail-open). Every substantive segment
#: of a clause is now assessed from its question structure:
#:   general              an overview ("What does the TTO do?", "Tell me about
#:                        research grants"): topic evidence may answer it;
#:   specific_recognised  a specific ask whose detail family can answer its
#:                        interrogative ("who" needs a party family, "when" a
#:                        timing family, "how much" a value family ...): the
#:                        family's evidence is required, and so are the words of
#:                        the ask the specialist cannot map, the yes/no subject,
#:                        the approval action and any stated purpose, in the
#:                        sentence that states the answer;
#:   specific_unknown     any other specific ask: a requested detail that no
#:                        evidence can cover is added, so the clause is at most
#:                        `partial` (or `not_found`), never `supported`.
ASK_GENERAL = "general"
ASK_RECOGNISED = "specific_recognised"
ASK_UNKNOWN = "specific_unknown"
ASK_PROCESS = "process"

#: Detail families that answer an ask. Category, locality, focus, foreign subject, contact and the ask details themselves
#: qualify an ask but are not one.
_ASK_FAMILIES = frozenset({"amount", "penalty", "deadline", "advance", "quantity", "duration", "frequency", "minimum", "maximum", "percentage",
                           "authority", "responsibility", "eligibility", "documents", "project_type", "funding_source", "role", "publication_type",
                           "requirement", "identity", "ownership", "permission"})
_PARTY_ASK = frozenset({"authority", "responsibility", "ownership", "eligibility", "identity", "role", "funding_source", "permission"})
#: The families that can answer each interrogative kind; a kind that is absent accepts any ask family.
#: Nouns of an unrecognised ask that point to a place, a time of service or a channel (Institutional Navigation's).
_CONTACT_HEADS = frozenset({"number", "no", "phone", "email", "extension", "office", "days", "day", "hours", "hour", "times", "way", "route",
                            "directions", "direction", "location", "address", "website", "contact", "staff", "entrance", "building", "room", "floor"})
_ASK_REQUIRES: Dict[str, FrozenSet[str]] = {
    "who": _PARTY_ASK,
    "when": frozenset({"deadline", "advance"}),
    "where": frozenset({"publication_type"}),
    "why": frozenset(),
    "how_much": frozenset({"amount", "percentage", "maximum", "minimum", "quantity"}),
    "how_many": frozenset({"quantity", "maximum", "minimum", "amount", "frequency"}),
    "how_long": frozenset({"duration", "deadline", "maximum", "advance"}),
    "how_often": frozenset({"frequency", "quantity", "maximum"}),
    "how_early": frozenset({"advance", "deadline"}),
    "how": frozenset({"percentage", "requirement"}),
    "elliptical": frozenset(),
}
#: Interrogative kinds that may ask for a general overview.
_GENERAL_KINDS = frozenset({"what", "which", "overview", "imperative", "none"})

_ASK_LEAD = re.compile(r"^\s*(?:(?:and|also|so|but|then|please|ok|okay|now|plus)\b[\s,]*)+", re.I)
_EMBEDDED_ASK = re.compile(r"^(?:(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|show|let)\s+me|(?:please\s+)?(?:tell|show|let)\s+me|"
                           r"i\s+(?:want|would\s+like|need)\s+to\s+know|do\s+you\s+know|i\s+(?:wonder|am\s+wondering))\s+"
                           r"(?=(?:who|whom|whose|when|where|why|which|what|how|whether|if)\b)", re.I)
_OVERVIEW_REQUEST = re.compile(r"^(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?(?:tell\s+me\s+(?:more\s+)?about|explain|describe|summari[sz]e|"
                               r"give\s+(?:me\s+)?(?:an?\s+)?(?:overview|summary|description|explanation)\s+of|(?:an?\s+)?overview\s+of|"
                               r"what\s+can\s+you\s+tell\s+me\s+about|i\s+(?:want|would\s+like|need)\s+to\s+know\s+(?:more\s+)?about|"
                               r"(?:more\s+)?information\s+(?:on|about))\b", re.I)
_AUX_START = re.compile(r"^(?:is|are|was|were|am|can|could|may|might|must|should|shall|will|would|does|do|did|has|have|had|isn['’]t|aren['’]t|"
                        r"can['’]t|doesn['’]t|don['’]t)\b", re.I)
_IMPERATIVE = re.compile(r"^(?:list|name|identify|give|show|state|provide|find|enumerate|outline|specify)\b", re.I)
_WH_WORD = re.compile(r"\b(?:(?:by|until|till|from)\s+when|when|who(?:m|se)?|where(?:['’]s|['’]re)?|why|which|what(?:['’]s)?|how)\b", re.I)
_LEADING_CONDITION = re.compile(r"^(?:when|if|after|before|once|while|since|as\s+soon\s+as)\s+[^,?]{1,80},\s*(?P<rest>\S.*)$", re.I)
_TIME_NOUNS = r"(?:time|date|day|month|week|year|semester|term|period|stage|point|timeframe|time\s+frame|timeline|schedule|window)"


def _ask_text(segment: str) -> str:
    text = _ASK_LEAD.sub("", cue_text(segment).strip()).strip()
    return _EMBEDDED_ASK.sub("", text)


def _ask_kind(segment: str) -> str:
    """The interrogative kind of one segment, read from its structure."""
    text = _ask_text(segment)
    condition = _LEADING_CONDITION.match(text)
    if condition and (_AUX_START.match(condition.group("rest")) or _WH_WORD.match(condition.group("rest"))):
        text = condition.group("rest")                             # "When I leave, do I lose my grant?" asks the second part
    if _OVERVIEW_REQUEST.match(text):
        return "overview"
    if re.match(r"^whether\b", text, re.I) or _AUX_START.match(text) or re.search(r"\bor\s+not\b", text, re.I):
        return "yes_no"
    if re.match(r"^if\b", text, re.I) and not _WH_WORD.search(text):
        return "yes_no"
    if _IMPERATIVE.match(text) and not _WH_WORD.search(text):
        return "who" if re.match(r"^(?:name|identify)\b", text, re.I) else "imperative"
    m = _WH_WORD.search(text)
    if not m:
        if re.search(r"\b\w+(?:ed|able|ible)\b|\bin\s+full\b|\b(?:possible|needed|required|available|mandatory|allowed|permitted|ok|okay)\b",
                     text, re.I):
            return "yes_no"                                        # a yes/no question without its auxiliary
        return "none"
    word = m.group(0).lower()
    rest = text[m.end():].lower()
    if "when" in word:
        return "when"
    if word.startswith("who"):
        return "who"
    if word.startswith("where"):
        return "where"
    if word == "why":
        return "why"
    if word == "how":
        following = re.match(r"\s+(\w+)", rest)
        nxt = following.group(1) if following else ""
        if nxt == "much":
            return "how_much"
        if nxt == "many":
            return "how_many"
        if nxt == "long":
            return "how_early" if re.match(r"\s+long\s+(?:before|in\s+advance|ahead)", rest) else "how_long"
        if nxt in ("often", "frequently"):
            return "how_often"
        if nxt in ("early", "soon", "far", "late", "quickly", "fast"):
            return "how_early"
        return "how"
    if re.match(r"\s+(?:exact\s+|specific\s+)?" + _TIME_NOUNS + r"\b", rest) or \
            re.match(r"(?:['’]s|\s+is|\s+are)\s+the\s+(?:exact\s+|specific\s+)?" + _TIME_NOUNS + r"\b", rest):
        return "when"
    if re.match(r"\s+(?:percentage|percent|share|proportion|portion|cut|fraction|amount|sum|rate)\b", rest):
        return "how_much"
    if re.match(r"\s+number\b", rest):
        return "how_many"
    return "which" if word == "which" else "what"


#: Nouns that ask for a description of the topic they govern ("the role of the TTO").
_OVERVIEW_NOUNS = frozenset({
    "role", "roles", "responsibility", "responsibilities", "duty", "duties", "function", "functions", "purpose", "purposes", "mission",
    "vision", "objective", "objectives", "goal", "goals", "aim", "aims", "scope", "definition", "meaning", "type", "types", "kind", "kinds",
    "category", "categories", "component", "components", "principle", "principles", "area", "areas", "priority", "priorities", "service",
    "services", "activity", "activities", "policy", "policies", "guideline", "guidelines", "regulation", "regulations", "rule", "rules",
    "overview", "structure", "composition", "mandate", "remit", "program", "programs", "programme", "programmes", "initiative", "initiatives",
    "opportunity", "opportunities", "work", "support", "summary", "description", "information", "details", "system"})
_GENERAL_FORMS = (
    re.compile(r"^what(?:['’]s|\s+is|\s+are|\s+was|\s+were)\s+(?P<np>.+)$", re.I),
    re.compile(r"^what\s+(?:does|do)\s+(?P<np>.+?)\s+(?:do|mean|cover|involve|include|handle|offer|provide|manage|oversee|entail|support|review|approve|"
               r"fund|administer|coordinate|evaluate|regulate|govern)$", re.I),
    re.compile(r"^(?:what|which)\s+(?:types?|kinds?|categories|sorts?)\s+of\s+(?P<np>.+?)(?:\s+(?:are|is)\s+(?:there|available|offered|supported|"
               r"provided|funded)|\s+(?:does|do|can)\s+.+?\s+(?:offer|provide|support|fund|have|get|receive))?$", re.I),
    re.compile(r"^(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?(?:tell\s+me\s+(?:more\s+)?about|explain|describe|summari[sz]e|"
               r"give\s+(?:me\s+)?(?:an?\s+)?(?:overview|summary|description|explanation)\s+of|(?:an?\s+)?overview\s+of|"
               r"what\s+can\s+you\s+tell\s+me\s+about|i\s+(?:want|would\s+like|need)\s+to\s+know\s+(?:more\s+)?about|(?:more\s+)?information\s+(?:on|about))"
               r"\s+(?P<np>.+)$", re.I),
    re.compile(r"^(?:list|give|show|outline|enumerate)\s+(?:me\s+)?(?P<np>.+)$", re.I),
)
#: A "what / which <topic> counts / is covered ..." question asks which items of the topic qualify.
_SELECTION_FORM = re.compile(
    r"^(?:what|which)\s+(?:kinds?\s+of\s+|types?\s+of\s+)?(?P<np>[\w\s'’-]{2,60}?)\s+(?:counts?|qualif(?:y|ies)|(?:is|are)\s+(?:considered|counted|"
    r"eligible|covered|allowed|permitted|supported|funded|admissible|accepted|recogni[sz]ed|included|reimbursed)|(?:does|do)\s+.{1,40}?\s+"
    r"(?:cover|fund|support|accept|recogni[sz]e|include|reimburse|count))\b", re.I)
#: A yes/no question about whether a thing exists or is offered ("Is there funding for conference travel?").
_EXISTENCE_FORM = re.compile(
    r"^(?:is|are)\s+there\b|^(?:does|do|will|can)\s+(?:the\s+)?(?:university|uos|college|department|it|they)\s+(?:have|offer|provide|fund|support|cover|"
    r"pay\s+for|give|run|host|publish|sponsor)\b|^(?:can|may|could)\s+(?:i|we|faculty(?:\s+members?)?|researchers?)\s+(?:get|receive|obtain|apply\s+for|have)\b",
    re.I)

#: Question scaffolding: words that carry no ask of their own.
_ASK_NEUTRAL = frozenset({
    "attend", "attending", "attended", "apply", "applying", "applied", "applies", "get", "gets", "getting", "got", "receive", "receives",
    "received", "receiving", "submit", "submits", "submitted", "submitting", "send", "sends", "sent", "sending", "take", "takes", "taken",
    "taking", "took", "have", "has", "having", "need", "needs", "needed", "needing", "use", "uses", "used", "using", "make", "makes", "made",
    "making", "give", "gives", "given", "giving", "go", "goes", "going", "went", "obtain", "obtained", "obtaining", "request", "requests",
    "requested", "requesting", "hold", "holds", "held", "spend", "spends", "spent", "keep", "keeps", "kept", "allow", "allows", "allowed",
    "permit", "permits", "permitted", "require", "requires", "required", "include", "includes", "included", "including", "cover", "covers",
    "covered", "covering", "pay", "pays", "paid", "paying", "fund", "funds", "funded", "support", "supports", "supported", "supporting",
    "offer", "offers", "offered", "provide", "provides", "provided", "providing", "count", "counts", "counted", "consider", "considered",
    "involve", "involves", "involved", "work", "works", "working", "serve", "serves", "act", "acts", "participate", "participates",
    "participating", "present", "presenting", "presented", "register", "registering", "registered", "travel", "travels", "travelling",
    "traveling", "travelled", "traveled", "host", "hosting", "join", "joining", "create", "creates", "created",
    "creating", "produce", "produced", "develop", "developed", "developing", "write", "writing", "written", "conduct", "conducted",
    "conducting", "carry", "carried", "happen", "happens", "happened", "qualify", "qualifies", "become", "becomes", "exist", "exists",
    "available", "eligible", "possible", "able", "entitled", "responsible", "valid", "know", "tell", "learn", "want",
    "like", "wish", "plan", "intend", "expect", "hope", "try", "done", "doing", "faculty", "member", "members", "staff", "person", "people",
    "someone", "anyone", "everyone", "university", "uos", "sharjah", "year", "years", "semester", "semesters", "time", "times", "day", "days",
    "month", "months", "week", "weeks", "thing", "things", "case", "cases", "new", "current", "total", "general", "standard",
    "usual", "normal", "typical", "regular", "exact", "exactly", "approximately", "roughly", "also", "still", "ever", "really", "actually",
    "currently", "usually", "normally", "typically", "generally", "already", "yet", "again", "please", "question", "questions", "answer",
    "information", "details", "rules", "rule", "policy", "policies", "number", "numbers", "amount", "amounts", "count", "classified",
    "attach", "attached", "attaching", "enclose", "enclosed", "lead", "leads", "leading", "same",
    "categorised", "categorized", "regarded", "called", "known", "defined", "described", "termed", "listed", "labelled", "labeled",
    # label nouns: "class A", "tier 2", "rank Q1" name a category label, not an ask
    "class", "classes", "tier", "tiers", "rank", "ranks", "level", "levels", "quartile", "quartiles"})
#: Predicate verbs outside the action lexicon: when a question uses one, the answering statement must state it.
_PREDICATE_VERBS = frozenset({
    "take", "taken", "taking", "took", "receive", "receives", "received", "receiving", "obtain", "obtained", "obtaining", "hold", "holds",
    "held", "spend", "spends", "spent", "keep", "keeps", "kept", "include", "includes", "included", "offer", "offers", "offered",
    "provide", "provides", "provided", "count", "counts", "counted", "consider", "considered", "involve", "involves", "involved", "work",
    "works", "worked", "working", "serve", "serves", "served", "act", "acts", "acted", "host", "hosts", "hosted", "hosting", "conduct",
    "conducts", "conducted", "conducting", "happen", "happens", "happened", "qualify", "qualifies", "qualified", "become", "becomes",
    "exist", "exists", "run", "runs", "manage", "manages", "managed", "lead", "leads", "led", "use", "uses", "used", "using", "lose",
    "loses", "lost", "check", "checks", "checked", "stamp", "stamps", "stamped", "handle", "handles", "handled", "oversee", "oversees"})
_ASK_STOP = frozenset({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "whether", "if", "much", "many", "long", "often", "me", "about",
    "there", "here", "any", "some", "it", "its", "this", "that", "these", "those", "do", "does", "did", "is", "are", "was", "were", "be",
    "been", "being", "am", "can", "could", "may", "might", "must", "should", "shall", "will", "would", "has", "have", "had", "not", "no",
    "yes", "or", "and", "but", "so", "to", "of", "for", "in", "on", "at", "by", "with", "from", "as", "into", "onto", "than", "then", "per",
    "via", "within", "without", "under", "over", "after", "before", "during", "between", "against", "towards", "toward", "upon", "across",
    "through", "throughout", "regarding", "concerning", "s", "i", "we", "you", "they", "my", "our", "your", "their", "his", "her", "a", "an",
    "the", "all", "each", "every", "one", "own", "other", "such", "more", "most", "less", "least", "only", "just", "very", "too", "up", "out",
    "off", "down", "because", "while", "until", "till", "since", "once", "whereas", "unless", "though", "although", "else", "etc",
    "explain", "describe", "tell", "overview", "summary", "summarize", "summarise", "give", "list", "show", "what's", "who's", "where's",
    "when's", "how's", "that's", "it's", "what’s", "who’s", "where’s"})
#: An agent or status noun built on a research verb, singular or plural ("publisher", "memberships").
_AGENT_NOUN_ANY = re.compile(r"^(?!research|member|invent|investigator|coordinator|supervisor|editor)[a-z]{4,}(?:ers?|ors?|ships?)$")
_ACRONYM = re.compile(r"\b[A-Z]{2}\b")


def _ask_residue(text: str, focus: Sequence[str] = (), relational: bool = False) -> List[str]:
    """The words of an ask the specialist cannot map: not a research concept, a detail-family cue, an object, a
    category, a locality, another domain's subject, a focus term or question scaffolding."""
    original = cue_text(text)
    work = original.lower()
    agents = [t for t in _TOKEN.findall(work) if _AGENT_NOUN_ANY.match(t)]
    acronyms = [a.lower() for a in _ACRONYM.findall(original) if a.lower() not in ("ip", "pi", "it", "hr", "ok", "uo")]

    def blank(m: "re.Match[str]") -> str:
        return " " * (m.end() - m.start())

    for _family, _label, pattern, _check in _COMPILED_FAMILIES:
        work = pattern.sub(blank, work)
    for _label, pattern in _COMPILED_CATEGORIES:
        work = pattern.sub(blank, work)
    for _name, pattern, _evidence, _specific, _submission in _COMPILED_OBJECTS:
        work = pattern.sub(blank, work)
    for _sid, pattern in _OTHER_CUES:
        work = pattern.sub(blank, work)
    for _role, pattern in _COMPILED_ROLES:                         # roles and actions are proved by the relation rules
        work = pattern.sub(blank, work)
    for _action, pattern in _COMPILED_ACTIONS:
        work = pattern.sub(blank, work)
    for pattern, _concept, anchor in _COMPILED_CONCEPTS:
        if anchor:                                                   # only a research topic is mapped; vocabulary words ("form",
            work = pattern.sub(blank, work)                          # "dates", "office") can carry the ask and must be grounded
    work = _LOCAL_WORDS.sub(blank, work)
    work = _NONLOCAL_WORDS.sub(blank, work)
    focus_words = {w for f in focus for w in f.split()}
    residue: List[str] = []
    for token in _TOKEN.findall(work) + agents + acronyms:
        if token.isdigit() or (len(token) < 3 and token not in acronyms):
            continue
        forms = _word_forms(token)
        predicate = relational and bool(forms & _PREDICATE_VERBS)   # in a relational ask a predicate verb carries the relation
        if not predicate and (token in _ASK_STOP or token in _ASK_NEUTRAL or token in _FOCUS_SKIP or token in _YES_NO_SKIP or
                              token in GENERIC_TERMS or forms & _ASK_NEUTRAL):
            continue
        if any(forms & _word_forms(f) for f in focus_words):
            continue
        stem = _stem(token)
        if stem not in residue:
            residue.append(stem)
    return residue


#: Nouns whose question asks for a description of the topic they govern ("the ROLE of the TTO", "the IP POLICY").
_OVERVIEW_HEADS = frozenset({
    "role", "roles", "responsibility", "responsibilities", "duty", "duties", "function", "functions", "purpose", "mission", "vision",
    "objective", "objectives", "goal", "goals", "aim", "aims", "scope", "definition", "meaning", "type", "types", "kind", "kinds", "category",
    "categories", "overview", "structure", "composition", "mandate", "remit", "policy", "policies", "priority", "priorities", "area", "areas",
    "activity", "activities", "service", "services", "initiative", "initiatives", "program", "programs", "programme", "programmes", "principle",
    "principles", "component", "components"})
_EMBEDDED_WH = re.compile(r"\b(?:when|who|whom|whose|where|why|how|which|whether|what)\b", re.I)
_TOPIC_FILLERS = frozenset({"the", "a", "an", "of", "and", "or", "for", "in", "at", "on", "to", "its", "their", "our", "my", "your", "his", "her",
                            "this", "that", "these", "those", "university", "uos", "sharjah", "faculty", "academic", "scientific", "s", "all",
                            "research", "researches", "researcher", "researchers"})
#: A body named by its modifiers ("the research funding DEPARTMENT", "the research COMMITTEE") is a topic.
_BODY_NOUNS = frozenset({"department", "departments", "office", "offices", "unit", "units", "centre", "centres", "center", "centers", "committee",
                         "committees", "board", "boards", "institute", "institutes", "council", "councils", "division", "section", "deanship"})
_NP_SPLIT = re.compile(r"\s+(?:of|for|on|about|regarding|concerning|in|at|to|from|with|within|under|by)\s+", re.I)


def _strict_topic(text: str, focus: Sequence[str], allowed: FrozenSet[str] = frozenset()) -> bool:
    """Every word names a research topic (an anchor concept), a proper name, a focus term, an allowed word or a filler."""
    original = cue_text(text)
    names = {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z]+", original)}
    focus_words = {w for f in focus for w in f.split()}
    for word in _TOKEN.findall(_uncovered_text(original, anchors_only=True)):
        stem = _stem(word)
        if word.isdigit() or word in _TOPIC_FILLERS or stem in _TOPIC_FILLERS or word in names or stem in names or word in allowed:
            continue
        if any(_word_forms(stem) & _word_forms(f) for f in focus_words):
            continue
        return False
    return True


def _topic_only(np: str, focus: Sequence[str]) -> bool:
    """A general subject: a research topic ("research groups", "the TTO"), or an overview noun of one ("the role of the
    TTO", "the IP policy"). A property of a topic ("the TTO's office NUMBER", "the application DATES", "the PROCEDURE to
    request ...") and an embedded question ("the policy on WHEN ...") are specific asks, never general."""
    text = cue_text(np).strip(" ?.!,")
    if not text or _EMBEDDED_WH.search(text):
        return False
    owner = re.match(r"^(?:the\s+)?(?P<owner>.+?)['’]s\s+(?P<rest>.+)$", text)
    complement = ""
    if owner:
        complement, text = owner.group("owner"), owner.group("rest")
    parts = _NP_SPLIT.split(text, maxsplit=1)
    left = parts[0]
    if len(parts) > 1:
        complement = (complement + " " + parts[1]).strip()
    words = [w for w in _TOKEN.findall(left.lower()) if w not in ("the", "a", "an", "my", "our", "their", "its", "your")]
    if not words:
        return False
    head = _stem(words[-1])
    whole = (left + " " + complement).strip()
    if head in _OVERVIEW_HEADS or words[-1] in _OVERVIEW_HEADS:
        # an overview of a named topic: every other word is a topic, and a research topic is named
        return len(words) + len(_TOKEN.findall(complement)) > 1 and _strict_topic(whole, focus, _OVERVIEW_HEADS) and \
            bool(concepts(whole).anchors or re.search(r"\b[A-Z]{2,}\b|\b[A-Z][a-z]+\s+[A-Z]", cue_text(np)))
    if words[-1] in _BODY_NOUNS and concepts(left).anchors and _strict_topic(whole, focus, _BODY_NOUNS):
        return True                                                 # "the research funding department"
    if not _strict_topic(whole, focus):
        return False
    uncovered_head = head in {_stem(w) for w in _TOKEN.findall(_uncovered_text(left, anchors_only=True))}
    proper = bool(re.search(r"\b" + re.escape(words[-1]) + r"\b", left) and re.search(r"\b[A-Z]", left.split()[-1]))
    domain = head in ("research", "researches")                    # "what research counts ...": the domain itself
    return bool(concepts(whole).anchors or concepts(whole).vocab) and (not uncovered_head or proper or domain)


def _general_form(segment: str, focus: Sequence[str]) -> bool:
    text = _ask_text(segment).strip(" ?.!")
    for form in _GENERAL_FORMS:
        m = form.match(text)
        if m and _topic_only(m.group("np"), focus):
            return True
    return False


def _selection_form(segment: str, focus: Sequence[str]) -> bool:
    m = _SELECTION_FORM.match(_ask_text(segment).strip(" ?.!"))
    return bool(m) and _topic_only(m.group("np"), focus)


#: The subject of a yes/no question ("Can VISITORS be principal investigators?").
_YES_NO_SUBJECT = re.compile(
    r"^(?:can|could|may|might|must|should|shall|will|would|is|are|was|were|does|do|did|has|have)\s+"
    r"(?:(?:a|an|the|my|our|their|his|her|its|all|any|every)\s+)?(?P<s>[a-z][\w'’-]*(?:\s+[a-z][\w'’-]*){0,3}?)\s+"
    r"(?:be|apply|get|receive|act|serve|take|use|hold|have|become|attend|submit|request|work|hire|buy|purchase|publish|travel|join|lead|supervise|"
    r"own|retain|lose|sign|approve|approved|authori[sz]e|extend|combine|keep|transfer|patent|register|host|claim|charge|spend|carry|do|go|"
    r"participate|present|need|remain|stay|continue|exceed|cover|covered|pay|fund|funded|include|count|qualify|allow|allowed|permit|permitted|require|"
    r"required|decide|grant|endorse|review|reviewed|signed|owned)\b", re.I)
_GENERIC_SUBJECTS = frozenset({"i", "we", "you", "they", "it", "he", "she", "one", "someone", "anyone", "faculty", "faculty member", "faculty members",
                               "member", "members", "researcher", "researchers", "staff", "university", "uos", "this", "that", "there", "people",
                               "academic members", "academic member"})


def _subject_pattern(segment: str) -> Optional["re.Pattern[str]"]:
    m = _YES_NO_SUBJECT.match(_ask_text(segment))
    if not m:
        return None
    words = [w for w in m.group("s").lower().split() if w not in ("a", "an", "the", "my", "our", "their")]
    phrase = " ".join(words)
    if not words or phrase in _GENERIC_SUBJECTS or all(w in _GENERIC_SUBJECTS or w in _FOCUS_SKIP for w in words):
        return None
    parts = []
    for word in words:
        base = _stem(word)
        if base.endswith("s") and not base.endswith("ss") and len(base) > 3:
            base = base[:-1]
        parts.append(re.escape(base) + r"(?:s|es)?")
    return re.compile(r"\b" + r"[\s-]+".join(parts) + r"\b", re.I)


#: The approval act a party question names; the answer sentence must state the same kind of act ("reviews and
#: recommends" never answers "who approves").
_ACTION_CLASSES = (
    (re.compile(r"\b(?:approv|authori[sz]|endors|ratif)\w*", re.I),
     re.compile(r"\b(?:approv\w*|authori[sz]\w*|endors\w*|ratif\w*|consent|permission|sign(?:s|ed)?\s+off)", re.I)),
    (re.compile(r"\bsign\w*", re.I), re.compile(r"\bsign\w*", re.I)),
    (re.compile(r"\b(?:review|evaluat|assess)\w*", re.I), re.compile(r"\b(?:review\w*|evaluat\w*|assess\w*)", re.I)),
    (re.compile(r"\b(?:decid|determin)\w*|\bsets?\b", re.I), re.compile(r"\b(?:decid\w*|decision\w*|determin\w*|sets?\b)", re.I)),
    (re.compile(r"\b(?:appoint|nominat|designat|elect|select)\w*|\bnames?\b", re.I),
     re.compile(r"\b(?:appoint\w*|nominat\w*|designat\w*|elect\w*|select\w*|nam(?:e|es|ed)\b)", re.I)),
    (re.compile(r"\b(?:grant|award)(?:s|ed|ing)?\b", re.I), re.compile(r"\b(?:grant\w*|award\w*|approv\w*)", re.I)),
)


def _action_pattern(segment: str, details: Sequence[Detail]) -> Optional["re.Pattern[str]"]:
    if not any(d.family == "authority" for d in details):
        return None
    text = _ask_text(segment)
    hits = [(m.start(), evidence) for question, evidence in _ACTION_CLASSES for m in [question.search(text)] if m]
    return min(hits, key=lambda h: h[0])[1] if hits else None


_PURPOSE = re.compile(r"\bfor\s+(research|commercial|non[\s-]?commercial|educational|teaching|personal|academic|private|profit)\s+"
                      r"(?:purposes?|use|reasons?)\b", re.I)


def _purpose_conflict(purpose: str, text: str) -> bool:
    stated = {m.group(1).lower() for m in _PURPOSE.finditer(text)}
    return bool(stated) and purpose not in stated


def _ask_unknown_check(unit: str) -> bool:
    """The safety guard: an ask the specialist does not recognise is never covered by any evidence."""
    return False


#: A statement: a label and its content ("Class A: journals indexed in Scopus") stay together.
_STATEMENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9•o(])")


#: A statement restricted to one group of beneficiaries ("awards for coordinators of research groups") never answers a
#: question that names none of that group.
_BENEFICIARY = re.compile(r"\bfor\s+(?:the\s+|all\s+)?(?P<b>(?:group\s+)?coordinators?(?:\s+of\s+research\s+groups?)?|students?|undergraduates?|"
                          r"graduate\s+students?|visitors?|visiting\s+researchers?|post\s?docs?|postdoctoral\s+\w+|lecturers?|research\s+assistants?|"
                          r"principal\s+investigators?|inventors?|newly\s+appointed\s+faculty|new\s+faculty)\b", re.I)


def _beneficiary_conflict(clause_words: FrozenSet[str], text: str) -> bool:
    for m in _BENEFICIARY.finditer(text):
        group = {_stem(w) for w in _TOKEN.findall(m.group("b").lower())} - {"of", "the"}
        if not any(_word_forms(w) & clause_words for w in group):
            return True
    return False


#: The object of a party question ("Who approves FIELD TRIPS for my project?"): its head noun must be stated with the
#: answer, whatever research concept it belongs to.
_PARTY_OBJECT = re.compile(
    r"\b(?:approves?|authori[sz]es?|signs?|decides?(?:\s+on)?|determines?|sets?|grants?|appoints?|nominates?|reviews?|endorses?|owns?|"
    r"(?:is|are)\s+responsible\s+for|oversees?|manages?|handles?)\s+(?:the\s+|a\s+|an\s+|my\s+|our\s+|their\s+|any\s+)?"
    r"(?P<np>(?:[\w'’-]+\s+){0,4}?[\w'’-]+?)(?=\s+(?:for|of|in|on|at|by|with|from|to|under|when|if|that|which|who|during|before|after|and|or)\b|\s*[?.!,]|\s*$)",
    re.I)


_SUBMISSION_HEADS = frozenset({"application", "applications", "request", "requests", "form", "forms", "proposal", "proposals", "submission",
                               "submissions", "document", "documents", "paperwork"})


def _object_head(segment: str) -> Optional[str]:
    m = _PARTY_OBJECT.search(_ask_text(segment))
    if not m:
        return None
    words = [w for w in _TOKEN.findall(m.group("np").lower()) if w not in _ASK_STOP]
    head = _stem(words[-1]) if words else None
    if not head or head in _ASK_NEUTRAL or head in _FOCUS_SKIP or head in GENERIC_TERMS or len(head) < 3 or head in _SUBMISSION_HEADS:
        return None                                                 # a generic submission noun binds through the objects instead
    return head


def _grounding_detail(details: Sequence[Detail], residue: Sequence[str], subject: Optional["re.Pattern[str]"],
                      action: Optional["re.Pattern[str]"], purpose: Optional[str], clause_words: FrozenSet[str] = frozenset(),
                      frame: Optional["RelationFrame"] = None) -> Optional[Detail]:
    """The asked specifics stated in the statement that states the answer, with no other purpose or beneficiary."""
    answers = [d for d in details if d.family in _ASK_FAMILIES and d.family != "documents"]
    values = [d.value if isinstance(d, _BoundDetail) and d.value is not None else d.check for d in answers]
    binds = [d.bind for d in answers if isinstance(d, _BoundDetail) and d.bind is not None]

    def value(unit: str) -> bool:
        # the specifics are stated in the same statement as an answer value
        for sentence in _STATEMENT_SPLIT.split(unit):
            text = cue_text(sentence)
            if values and not any(v(sentence) for v in values):
                continue
            if not all(_has_word(text, w) for w in residue):
                continue
            if subject is not None and not subject.search(text):
                continue
            if action is not None and not action.search(text):
                continue
            if purpose and _purpose_conflict(purpose, text):
                continue
            if clause_words and _beneficiary_conflict(clause_words, text):
                continue
            if frame is not None and not _relation_ok(frame, sentence, unit, values):
                continue                                             # the asked relation is not proved by this statement
            return True
        return False

    def bind(context: str) -> bool:
        # and that answer belongs to the asked object (the unit with its list lead-in, as for every bound detail)
        if frame is not None and frame.share_owner and frame.share_owner not in {_stem(m.group(1)) for m in _SHARE_OWNER.finditer(cue_text(context))}:
            return False
        return all(b(context) for b in binds)

    parts = list(residue) or ["answer statement"]
    if subject is not None:
        parts.append("subject")
    if action is not None:
        parts.append("action")
    if purpose:
        parts.append("%s purpose" % purpose)
    if frame is not None:
        parts += ["role %s" % r for r in sorted(frame.roles)] + ["action %s" % a for a, _h in frame.actions]
        if frame.proposition:
            parts.append("proposition %s" % " ".join(frame.proposition))
        if frame.required:
            parts.append("required %s" % frame.required)
    return _BoundDetail("ask_grounding", "the asked specifics stated with the answer (%s)" % ", ".join(parts),
                        lambda unit: value(unit) and bind(unit), value=value, bind=bind)


# ---------------------------------------------------------------------------
# Relation strength: the conservative SUPPORTED contract (Milestone 4, Step 4.5A-D)
# ---------------------------------------------------------------------------
#: A recognised ask is SUPPORTED only when the deterministic rules PROVE that the answering statement states the asked
#: relation: every role the question names is named there, every action it asks about (and its family does not already
#: imply) is predicated there of the asked object, in the same coordinate part as the answer, a stated purpose is the
#: asked one, and a yes/no proposition is contained in the unit. Anything the rules cannot prove is AMBIGUOUS and at most
#: `partial`. Semantic relation verification belongs to the Milestone 5 Verifier; Milestone 4 specialists fail
#: conservatively when deterministic relation evidence is ambiguous.
RELATION_STRONG = "strong"
RELATION_AMBIGUOUS = "ambiguous"

#: Roles a question can name. A rule stated for one role never answers a question about another.
_ROLE_LEXICON: Tuple[Tuple[str, str], ...] = (
    ("co_investigator", r"co\s?(?:principal\s+)?investigators?|co\s?pis?\b|associate\s+investigators?"),
    ("principal_investigator", r"(?<!co\s)(?<!co)principal\s+investigators?|(?<!co\s)\bpis?\b"),
    ("research_assistant", r"research\s+assistants?"),
    ("inventor", r"inventors?"),
    ("tto", r"\btto\b|technology\s+transfer\s+office"),
    ("vcrgs", r"\bo?vcrgs\b|vice\s+chancellor\s+for\s+research"),
    ("vcaa", r"\bvcaa\b|vice\s+chancellor\s+for\s+academic\s+affairs"),
    ("chancellor", r"(?<!vice\s)\bchancellor\b"),
    ("dean", r"\bdeans?\b"),
    ("chair", r"\bchairs?\b|\bchairperson\b"),
    ("chief_editor", r"chief\s+editors?|editors?\s+in\s+chief"),
    ("editorial_board", r"editorial\s+boards?"),
    ("editorial_secretary", r"editorial\s+secretar(?:y|ies)"),
    ("coordinator", r"coordinators?"),
    ("student", r"\bstudents?\b|\bundergraduates?\b"),
    ("staff", r"\bstaff\b(?!\s+development)"),
    ("irb", r"\birb\b|institutional\s+review\s+board"),
    ("iacuc", r"\biacuc\b|institutional\s+animal\s+care"),
    ("ethics_committee", r"ethics\s+committees?|\brec\b"),
    ("library", r"\blibrary\b|\blibraries\b"),
    ("purchasing_department", r"purchasing\s+departments?|purchasing\s+committees?"),
    ("head", r"\bheads?\s+of\b"),
    ("member", r"\bmembers?\s+of\s+(?:a\s+|the\s+)?research\s+groups?|research\s+groups?\s+members?"),
    ("committee", r"\bcommittees?\b"),
    ("board", r"\bboards?\b(?!\s+of\b)"),
    ("visitor", r"visiting\s+researchers?|\bvisitors?\b"),
    ("postdoc", r"post\s?docs?|postdoctoral"),
    ("research_board", r"research\s+board"),
    ("research_committee", r"research\s+committees?"),
    ("consultant", r"\bconsultants?\b"),
    ("referee", r"\breferees?\b|\breviewers?\b"),
    ("moderator", r"\bmoderators?\b|\barbit(?:er|rator)s?\b"),
)
#: Actions a question can ask about, each with the forms that state it.
_ACTION_LEXICON: Tuple[Tuple[str, str], ...] = (
    ("approve", r"approv\w*|authori[sz]\w*|endors\w*|ratif\w*|sign(?:s|ed)?\s+off"),
    ("review", r"review\w*|evaluat\w*|assess\w*|examin\w*"),
    ("recommend", r"recommend\w*"),
    ("submit", r"submi(?:t|ts|tted|tting|ssion|ssions)\b|forward(?:s|ed|ing)?\b"),
    ("apply", r"appl(?:y|ies|ied|ying|ication|ications)\b|request(?:s|ed|ing)?\b"),
    ("register", r"(?:re\s?)?regist(?:er|ers|ered|ering|ration|rations)\b"),
    ("create", r"creat(?:e|es|ed|ing|ion)\b|creative|establish\w*|formation|forming|formed\b|found(?:ed|ing)\b|set(?:ting)?\s+up|"
               r"produc(?:e|es|ed|ing)\b|develop(?:s|ed|ing)?\b|made\b|mak(?:e|es|ing)\b|writ(?:e|es|ing|ten)\b"),
    ("join", r"join(?:s|ed|ing)?\b|become\s+(?:a\s+)?members?|membership"),
    ("extend", r"extend\w*|extension\w*|renew\w*|carr(?:y|ies|ied)\s+(?:it\s+)?forward|prolong\w*"),
    ("close", r"clos(?:e|es|ed|ing|ure)\b|terminat\w*|dissol\w*|settl(?:e|es|ed|ing|ement)\b"),
    ("transfer", r"transfer(?:s|red|ring)?\b"),
    ("publish", r"publish\w*|publication\w*"),
    ("report", r"reported|reporting|reports?\s+to\b|to\s+report\b"),
    ("announce", r"announc\w*"),
    ("decide", r"decid\w*|decision\w*|determin\w*"),
    ("disclose", r"disclos\w*"),
    ("sign", r"sign(?:s|ed|ing)?\b(?!\s+off)|signature\w*"),
    ("appoint", r"appoint\w*|designat\w*|elect(?:s|ed|ion)?\b|select(?:s|ed|ion)?\b"),
    ("nominate", r"nominat\w*"),
    ("pay", r"pa(?:y|ys|id|ying|yment|yments)\b|reimburs\w*|compensat\w*|remunerat\w*|cover(?:s|ed|ing)?\b|fund(?:s|ed|ing)?\b|financ\w*|"
            r"sponsor\w*|support(?:s|ed|ing)?\b"),
    ("deduct", r"deduct\w*|withh[eo]ld\w*"),
    ("distribute", r"distribut\w*|allocat\w*|(?:goes|go|given|allotted)\s+to\b|shar(?:e|ed|es|ing)\s+(?:with|between|among)\b|\d+\s?%\s+to\b"),
    ("attend", r"attend\w*|participat\w*|take\s+part|taking\s+part|present(?:s|ed|ing)?\s+(?:at|a\s+paper|papers)\b"),
    ("travel", r"travel\w*|\btrips?\b"),
    ("supervise", r"supervis\w*"),
    ("own", r"own(?:s|ed|er|ers|ership)?\b|retain\w*"),
    ("purchase", r"purchas\w*|\bbuy(?:s|ing)?\b|bought|acqui(?:re|res|red|ring|sition)\b"),
    ("hire", r"hir(?:e|es|ed|ing)\b|recruit\w*|employ(?:s|ed|ing)?\b"),
    ("exceed", r"exceed\w*"),
    ("combine", r"combin\w*"),
    ("manage", r"manag(?:e|es|ed|ing|ement)\b|oversee\w*|oversaw|administ(?:er|ers|ered|ering|ration)\b|handl(?:e|es|ed|ing)\b|"
               r"coordinat(?:e|es|ed|ing|ion)\b"),
    ("teach", r"teach\w*|taught|lectur(?:e|es|ed|ing)\b"),
)
_COMPILED_ROLES = tuple((role, re.compile(r"\b(?:" + p + r")", re.I)) for role, p in _ROLE_LEXICON)
_COMPILED_ACTIONS = tuple((action, re.compile(r"\b(?:" + p + r")", re.I)) for action, p in _ACTION_LEXICON)
_ACTION_PATTERNS = dict(_COMPILED_ACTIONS)
#: Actions a recognised family already states in its own evidence (an amount is what is paid, a percentage what is
#: distributed, a deadline is for a submission): they are not asked again.
_IMPLIED_ACTIONS: Dict[str, FrozenSet[str]] = {
    "amount": frozenset({"pay", "distribute"}), "percentage": frozenset({"distribute", "pay"}), "maximum": frozenset({"pay"}),
    "minimum": frozenset({"pay"}), "quantity": frozenset({"pay"}), "deadline": frozenset({"submit", "apply"}),
    "advance": frozenset({"submit", "apply"}), "documents": frozenset({"submit", "apply"}), "eligibility": frozenset({"apply"}),
    "ownership": frozenset({"own"}), "funding_source": frozenset({"pay"}), "publication_type": frozenset({"publish"}),
}
_PREPOSITIONS = frozenset({"for", "of", "in", "on", "at", "to", "by", "with", "from", "per", "during", "before", "after", "under", "within",
                           "about", "until", "through", "between", "into", "upon", "regarding", "concerning", "as", "without"})
_TARGET_DROP = frozenset({"the", "a", "an", "my", "our", "their", "his", "her", "its", "your", "this", "that", "these", "those", "who", "whom",
                          "whose", "what", "which", "when", "where", "how", "why", "will", "would", "can", "could", "may", "might", "must",
                          "should", "shall", "do", "does", "did", "is", "are", "was", "were", "be", "been", "being", "i", "we", "you", "they",
                          "it", "he", "she", "any", "each", "every", "all", "some", "and", "or", "not", "also", "get", "have", "has"})
#: Words a yes/no proposition never needs stated (the answer type, light verbs, generic subjects).
_PROPOSITION_EXEMPT = frozenset({"allowed", "permitted", "possible", "able", "eligible", "entitled", "required", "mandatory", "necessary",
                                 "compulsory", "obligatory", "available", "okay", "ok", "fine", "true", "must", "need", "needs", "needed", "require",
                                 "requires", "permit", "permits", "allow", "allows", "there", "get", "gets", "getting", "got", "have", "has", "having",
                                 "had", "go", "goes", "going", "faculty", "member", "members", "staff", "researcher", "researchers", "university",
                                 "uos", "people", "person", "someone", "anyone", "one", "still", "ever", "also", "really", "actually", "just",
                                 "please", "yes", "no", "not"})


@dataclass(frozen=True)
class RelationFrame:
    """The relation a recognised ask names: its roles, its actions (each with the head noun it is predicated of, when the
    question states one), a stated purpose, and a yes/no proposition's content words."""

    roles: FrozenSet[str]
    actions: Tuple[Tuple[str, Optional[str]], ...]
    purpose: Optional[str]
    proposition: Tuple[str, ...]
    timed: bool = False
    recipients: Tuple[str, ...] = ()
    permission: bool = False
    share_owner: Optional[str] = None
    competitors: Tuple[Tuple[str, str], ...] = ()
    percent_base: Tuple[str, ...] = ()
    counted: bool = False
    modifiers: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    required: Optional[str] = None
    words: FrozenSet[str] = frozenset()
    #: position binding (the fresh adversarial check of Step 4.5A-D)
    subject: Tuple[str, ...] = ()
    subject_roles: FrozenSet[str] = frozenset()
    subject_generic: bool = False
    subject_np: Tuple[str, ...] = ()
    subject_text: str = ""
    subject_complement: Tuple[str, ...] = ()
    copular: Tuple[str, ...] = ()
    yes_no: bool = False
    predicate: Optional[str] = None
    passive: bool = False
    asked_object: Tuple[str, ...] = ()
    agents: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    arguments: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    period: Optional[Tuple[str, str]] = None
    negated: Tuple[str, ...] = ()
    restrictions: Tuple[Tuple[str, Tuple[Tuple[str, ...], Tuple[str, ...]]], ...] = ()
    unit: str = ""


def _roles_in(text: str) -> FrozenSet[str]:
    view = cue_text(text)
    return frozenset(role for role, pattern in _COMPILED_ROLES if pattern.search(view))


_BE_AUX = frozenset({"is", "are", "was", "were", "be", "been", "being"})
_MODALS = frozenset({"can", "could", "may", "might", "must", "should", "shall", "will", "would", "do", "does", "did", "to", "not"})
_DETERMINERS = frozenset({"the", "a", "an", "my", "our", "their", "his", "her", "its", "your", "this", "that", "these", "those", "any", "each",
                          "every", "some", "no"})
_NOUN_GERUNDS = frozenset({"funding", "building", "housing", "training", "meeting", "hearing", "setting", "reporting", "processing", "writing",
                           "publishing", "licensing", "accounting", "teaching"})
_NOMINAL = re.compile(r"(?:al|ion|ions|ment|ments|ure|ures|ship|ance|ence|sal|ation|ations)$")
#: Asks whose answer is a relation between a party or a rule and an action (not a value of an object).
_RELATIONAL_FAMILIES = frozenset({"authority", "responsibility", "ownership", "permission", "penalty", "requirement", "deadline", "advance"})


def _occurrence_form(text: str, m: "re.Match[str]") -> str:
    """How an action word is used: "verbal", "nominal", "modifier" (a participle describing a noun: "IP CREATED by faculty",
    "NON-COMPENSATED consultancy") or "noun" (a noun compound: "conference TRAVEL")."""
    word = m.group(0).lower().split()[-1]
    before_words = re.findall(r"[\w'’-]+", text[:m.start()].lower())
    previous = before_words[-1] if before_words else ""
    following = re.match(r"[\s,]*([\w'’-]+)", text[m.end():])
    following_word = following.group(1).lower() if following else ""
    if re.search(r"(?:ed|en)$", word) and word not in ("open", "often", "given"):
        if previous in _BE_AUX or not following_word or following_word in ("to", "and", "or", "?"):
            return "verbal"
        return "modifier"                                         # "created BY faculty", "approved completion period"
    if word.endswith("ing"):
        return "noun" if previous in _DETERMINERS or word in _NOUN_GERUNDS else "verbal"
    if _NOMINAL.search(word):
        if following_word in ("of", "for", "to", "is", "are", "required", "needed", "necessary", "mandatory") and \
                not re.match(r"(?:application|submission|request)", word):
            return "nominal"
        return "noun"
    if previous in _MODALS or previous in ("who", "which", "that", "i", "we", "you", "they", "he", "she", "it") or \
            any(w in _MODALS for w in before_words[-4:]) or (len(before_words) >= 2 and before_words[-2] in ("which", "what")):
        return "verbal"
    return "noun"


def _statement_noun(text: str, m: "re.Match[str]") -> bool:
    before_words = re.findall(r"[\w'’-]+", text[:m.start()].lower())
    word = m.group(0).lower().split()[-1]
    return bool(before_words and before_words[-1] in _DETERMINERS and not _NOMINAL.search(word)) or word in _NOUN_GERUNDS


def _local_target(text: str, m: "re.Match[str]") -> str:
    """The words an action occurrence is predicated of in a statement: the clause subject for a passive ("the advance may
    be CLOSED", "unless explicitly APPROVED by", "..., NOMINATED by"), the noun right before a reduced participle
    ("invoices APPROVED by the Dean"), the noun after an adjectival participle ("APPROVED completion period"), the whole
    statement for a nominal ("prior APPROVAL is required"), and otherwise the words after it."""
    before, after = text[:m.start()], text[m.end():]
    word = m.group(0).lower().split()[-1]
    before_words = re.findall(r"[\w'’-]+", before.lower())
    previous = before_words[-1] if before_words else ""
    following = re.match(r"[\s,]*([\w'’-]+)", after)
    following_word = following.group(1).lower() if following else ""
    start = max(before.rfind(". "), before.rfind("; "), before.rfind(": "), before.rfind("? "))
    clause = before[start + 1:]
    participle = bool(re.search(r"(?:ed|en)$", word)) and word not in ("open", "often", "given")
    if _NOMINAL.search(word) and not participle:
        if re.search(r"\b(?:subject\s+to|upon|with|requires?|requiring|following)\s+(?:the\s+|prior\s+|written\s+)?(?:[\w'’-]+\s+){0,4}$",
                     before, re.I):
            return clause                                          # "X requires the APPROVAL of ...": the approval of X
        complement = " ".join(re.split(r"[.;:?]", after)[0].split()[:10])
        if following_word in ("of", "for", "to") and not _RESEARCH_AUTHORITY_NOUN.search(" ".join(complement.split()[:4])):
            return complement                                      # "the APPROVAL of the budget": its complement
        return text                                                # "prior APPROVAL is required", "approval of the VCRGS": the statement
    if participle:
        if re.search(r",\s*$", before) or previous in ("and", "or"):
            return ""                                              # "report for the project, APPROVED by the Dean": ambiguous attachment
        if previous in _BE_AUX or previous.endswith("ly") or previous in ("unless", "if", "once", "when", "until", "not"):
            return clause                                          # a passive: the subject of its clause
        if following_word in ("by", "with", "until", "under", "through", "at", "in", "for") or not following_word:
            return " ".join(before.split()[-3:])                   # a reduced participle: the noun it follows
        if following_word not in _PREPOSITIONS:
            return " ".join(re.split(r"[.;:]", after)[0].split()[:3])   # an adjective: the noun it precedes
    if re.search(r"\b(?:subject\s+to|upon|with|requires?|requiring|following)\s+(?:the\s+|prior\s+|written\s+)?$", before, re.I):
        return clause
    return " ".join(re.split(r"[.;:?]", after)[0].split()[:10])


_TIME_WORDS = frozenset({"year", "years", "semester", "semesters", "month", "months", "week", "weeks", "day", "days", "time", "times",
                         "once", "twice", "annually", "yearly", "simultaneously", "same"})


def _reduced_relative(token: str, previous: str, following: str) -> bool:
    """A participle that ends a noun phrase: "licenses RELATED to", "research INVOLVING animals" (never "the FUNDING
    ceiling", "the funding CEILING for", "university SUPPORTED conference")."""
    if previous in _DETERMINERS or token in _NOUN_GERUNDS:
        return False
    if re.search(r"(?<!e)ed$", token):
        return following in _PREPOSITIONS or not following
    return token.endswith("ing") and bool(following) and following not in _PREPOSITIONS and following not in ("and", "or")


def _object_parts(target: str) -> Tuple[Tuple[str, ...], FrozenSet[str], FrozenSet[str]]:
    """The noun phrase an action takes or is said of (its words in order), its one of/for complement and its possessors
    ("the PROJECT'S closure")."""
    tokens = _TOKEN.findall(re.sub(r"\([^)]*\)", " ", cue_text(target)).lower())
    if tokens and tokens[0] == "o":
        tokens = tokens[1:]                                        # a printed bullet
    if tokens and tokens[0] in _PREPOSITIONS:
        tokens = tokens[1:]
    phrase: List[str] = []
    complement_words: FrozenSet[str] = frozenset()
    possessors: FrozenSet[str] = frozenset()
    complement = False
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        previous = tokens[i - 1] if i else ""
        if token in _PREPOSITIONS:
            if token in ("of", "for") and phrase and not complement:
                complement = True
                continue
            break
        if token in _SUBORDINATORS:
            break
        if phrase and _reduced_relative(token, previous, following):
            break
        if not phrase and token.endswith("ing") and token not in _NOUN_GERUNDS and any(p.fullmatch(token) for _a, p in _COMPILED_ACTIONS):
            continue                                               # "reviewing AND APPROVING research protocols": a coordinated verb
        if re.search(r"['’]s$|s['’]$", token):
            possessors = possessors | {_stem(token)}
            continue
        if token in _TARGET_DROP or token.isdigit():
            continue
        if complement:
            complement_words = complement_words | {token}
        else:
            phrase.append(token)
    return tuple(phrase), complement_words, possessors


def _object_words(target: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """The words of the (coordinated) noun phrases an action is predicated of, and of their one "of"/"for" complement:
    "approves TRAVEL REQUESTS and ...", "determines the FUNDING CEILING for [research grants] based on ...", "covers COPYRIGHT
    PROTECTION for [books, computer software] ...", "participate in one LOCAL ... CONFERENCE per year". A possessor ("the
    project's closure") is not the object; the phrase ends at another preposition, a subordinator or a reduced relative
    ("licenses RELATED to ...", "protocols INVOLVING human subjects")."""
    tokens = _TOKEN.findall(re.sub(r"\([^)]*\)", " ", cue_text(target)).lower())
    if tokens and tokens[0] in _PREPOSITIONS:
        tokens = tokens[1:]                                        # an intransitive verb's object follows its preposition
    words: FrozenSet[str] = frozenset()
    complement_words: FrozenSet[str] = frozenset()
    complement = False
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        previous = tokens[i - 1] if i else ""
        if token in _PREPOSITIONS:
            if token in ("of", "for") and words and not complement:
                complement = True
                continue
            break
        if token in _SUBORDINATORS:
            break
        if words and _reduced_relative(token, previous, following):
            break                                                  # a reduced relative: "RELATED to", "INVOLVING human subjects"
        if token in _TARGET_DROP or token.isdigit() or re.search(r"['’]s$|s['’]$", token):
            continue
        if complement:
            complement_words = complement_words | {token}
        else:
            words = words | {token}
    return words, complement_words


def _phrase_names(target: str, head: str, asked: FrozenSet[str],
                  restrictions: Tuple[Tuple[str, ...], Tuple[str, ...]] = ((), ()), statement: str = "") -> bool:
    """The asked head heads the noun phrase, or is in the complement of a head the question itself names ("does COPYRIGHT
    PROTECTION cover software?" / "covers copyright protection for ... software"). An asked modifier that is not the topic
    is stated somewhere in the unit; when the question restricts the head with a word the phrase lacks, the phrase must
    not restrict it with a word the question lacks ("PUBLICATION grant applications" is never "RESEARCH grant
    applications", "research load" never "TEACHING loads"; "grant applications" may be "research grant applications");
    an asked "of" complement is the phrase's own complement or possessor ("costs OF THE HOST INSTITUTION" is not "faculty
    members' costs")."""
    phrase, complement, possessors = _object_parts(target)
    wanted = _word_forms(head)
    position = next((i for i, w in enumerate(phrase) if _word_forms(w) & wanted), None)
    if position is None:
        wrapper = all(w in _NON_CONTRAST or w in _QUANTITY_WRAPPERS for w in phrase)   # "a MAXIMUM OF four theses"
        if not (any(_word_forms(w) & wanted for w in complement) and (wrapper or any(_word_forms(w) & _word_forms(a) for w in phrase for a in asked))):
            return False
        position = len(phrase)
    premodifiers, of_words = restrictions[0], restrictions[1]
    for_words = restrictions[2] if len(restrictions) > 2 else ()
    available = set(phrase) | complement | possessors
    for w in premodifiers:
        if w not in _TOPIC_FILLERS and not _anchor_word(w) and not (_word_forms(w) & _forms_of(available)) and \
                not (statement and _has_word(statement, w)):
            return False                                           # an asked modifier the unit never states
    if any(not (_word_forms(w) & _forms_of(available)) for w in premodifiers):
        stated = [w for w in list(phrase[:position]) + list(possessors) if w not in _NON_CONTRAST and not w.isdigit()]
        if any(not (_word_forms(w) & _forms_of(asked)) for w in stated):
            return False                                           # a contrasting restriction
    for w in of_words:
        if w not in _TOPIC_FILLERS and not _anchor_word(w) and not (_word_forms(w) & _forms_of(complement | possessors | set(phrase))):
            return False
    for w in for_words:
        if w not in _TOPIC_FILLERS and not (_word_forms(w) & _forms_of(available)) and not (statement and _anchor_stated(w, statement)):
            return False                                           # "the ceiling for PUBLICATION grants" never stated
    return True


_QUANTITY_WRAPPERS = frozenset({"maximum", "minimum", "total", "number", "amount", "majority", "range", "variety", "series", "set", "list"})
#: Modifiers that never contrast an asked object: quantities, scope and generic adjectives, time words.
_NON_CONTRAST = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "single", "multiple", "several", "various", "many",
    "local", "external", "internal", "international", "regional", "national", "other", "same", "such", "new", "current", "total",
    "general", "standard", "appropriate", "relevant", "specific", "special", "full", "partial", "prior", "written", "original",
    "final", "annual", "regular", "additional", "extra", "different", "similar", "particular", "certain", "own", "main", "joint",
    "first", "second", "last", "next", "previous", "maximum", "minimum", "basic", "complete", "necessary", "proper", "official",
    "formal", "informal", "direct", "indirect", "potential", "possible", "future", "further", "any", "all", "each", "every", "both",
    "supported", "approved", "recognized", "recognised", "designated", "concerned", "relevant"} | _TARGET_DROP | _TIME_WORDS)


def _anchor_stated(word: str, text: str) -> bool:
    """The word, or the topic it names ("IP" / "intellectual property"), is in the text."""
    if _has_word(text, word):
        return True
    return any(anchor and pattern.search(word) and pattern.search(text) for pattern, _concept, anchor in _COMPILED_CONCEPTS)


def _anchor_word(word: str) -> bool:
    """A topic word (an anchor concept: "conference", "IP", "licensing"): retrieval and the gates match it; the relation
    rules do not require it as a modifier."""
    return any(anchor and pattern.search(word) for pattern, _concept, anchor in _COMPILED_CONCEPTS)


def _forms_of(words: "Set[str] | FrozenSet[str] | Sequence[str]") -> FrozenSet[str]:
    out: FrozenSet[str] = frozenset()
    for w in words:
        out = out | _word_forms(w)
    return out


#: A section heading printed before a unit's first sentence ("16.3 Animal Use in Research | All research ...").
_HEADING_LEAD = re.compile(r"^[\s•o]*(?:\d+(?:\.\d+)*\.?\s+)?(?!(?:The|A|An|All|Each|Every|Any|No)\b)"
                           r"(?:(?:[A-Z][a-z][\w'’-]*|of|for|and|in|on|the|to|a|an|&)\s+){2,}?(?=[A-Z][a-z][\w'’-]*\s+([a-z][\w'’-]*))")


def _strip_heading(text: str) -> str:
    """The text without a printed heading glued before its first sentence ("16.3 Animal Use in Research All research
    ..."): Title Case words followed by a capitalised sentence start whose next word is not a verb ("Vice Chancellor for
    Academic Affairs approves ..." is a name, not a heading)."""
    m = _HEADING_LEAD.match(text)
    if not m:
        return text
    following = m.group(1).lower()
    if _is_verb_token(following, "", ""):
        return text
    return text[m.end():]


def _passive_subject(clause: str) -> str:
    """The subject of a passive: its clause without a printed heading, a leading adverb or an introductory adverbial ("Even
    in cases of non-compensated consultancy, PRIOR APPROVAL is ...")."""
    clause = _strip_heading(clause)
    clause = re.sub(r"^\s*(?:even|also|however|additionally|moreover|furthermore|similarly|therefore|thus|hence)\s+", "", clause, flags=re.I)
    tokens = _TOKEN.findall(cue_text(clause).lower())
    if tokens and (tokens[0] in _PREPOSITIONS or tokens[0] in _SUBORDINATORS) and "," in clause:
        return clause.split(",", 1)[1]
    return clause


def _bare_definite(target: str) -> bool:
    """An anaphoric object: a definite determiner and one noun ("reviews THE APPLICATION"); its antecedent is elsewhere."""
    tokens = _TOKEN.findall(re.sub(r"\([^)]*\)", " ", cue_text(target)).lower())
    return len(tokens) >= 2 and tokens[0] in ("the", "this", "that", "such", "these", "those", "said") and \
        (len(tokens) == 2 or tokens[2] in _PREPOSITIONS or tokens[2] in _SUBORDINATORS or tokens[2] in ("and", "or"))


_SUBORDINATORS = frozenset({"if", "unless", "when", "because", "provided", "whereas", "which", "that", "who", "while", "once", "so"})


def _target_matches(text: str, m: "re.Match[str]", head: str, modifiers: Sequence[str] = (),
                    asked: FrozenSet[str] = frozenset(), restrictions: Tuple[Tuple[str, ...], Tuple[str, ...]] = ((), ()),
                    unit: str = "") -> bool:
    """The asked head is what the occurrence is predicated of: anywhere in the statement or clause a nominal is stated of
    ("prior APPROVAL is required"); otherwise the head of the noun phrase it takes or is said of (the object, the passive
    subject, the noun a participle modifies). A bare definite object ("reviews THE application") or a nominal stated of a
    whole clause ("..., subject to APPROVAL by the VCRGS") proves a modified asked target ("my research GRANT application")
    only when the statement states the modifiers."""
    target = _local_target(text, m)
    if not target:
        return False
    word = m.group(0).lower().split()[-1]
    participle = bool(re.search(r"(?:ed|en)$", word)) and word not in ("open", "often", "given")
    before = text[:m.start()]
    if _NOMINAL.search(word) and not participle and target == text:
        # a nominal stated of its whole statement ("prior APPROVAL is required"): the asked target with its restricting words
        return _has_word(target, head) and all(_has_word(text, w) for w in modifiers)
    if _NOMINAL.search(word) and not participle and before.endswith(target):
        # a nominal attached to a clause ("..., upon the Chancellor's APPROVAL"): the asked target heads the clause's object
        return _clause_core_names(target, head) and all(_has_word(text, w) for w in modifiers)
    if participle and before.endswith(target):
        return _phrase_names(_passive_subject(target), head, asked, restrictions, unit or text)
    if not _phrase_names(target, head, asked, restrictions, unit or text):
        return False
    return participle or not _bare_definite(target) or all(_has_word(text, w) for w in modifiers)


def _clause_core_names(clause: str, head: str) -> bool:
    """The asked head is the clause's subject or the object of one of its verbs ("upon the Chancellor's approval" attached
    to "the Board may approve AWARDS for coordinators" is an approval of awards, not of coordinators); a clause with no
    readable verb keeps the word test."""
    clause = _passive_subject(clause)
    tokens = _split_tokens(clause)
    heads: List[str] = []
    for i, t in enumerate(tokens):
        if t in _AUX or t == "be" or t in (",", ";", ":", "(", ")") or (t in _PREPOSITIONS and t not in ("of", "for")) or \
                (i and _is_verb_token(t, tokens[i - 1], tokens[i + 1] if i + 1 < len(tokens) else "")):
            break
        if t not in _DETERMINERS and t != "of" and t != "for":
            heads.append(_stem(t))
    found = False
    active = False
    subject_heads = list(heads)
    heads = []
    text = " ".join(tokens)
    for action, pattern in _COMPILED_ACTIONS:
        for m in pattern.finditer(text):
            if _statement_noun(text, m):
                continue
            voice = _occurrence_voice(text, m)
            found = True
            if voice == "active":
                active = True
                heads.extend(_object_parts(text[m.end():])[0])
            elif voice == "passive":
                heads.extend(_object_parts(_passive_subject(_local_target(text, m)))[0])
    for m in re.finditer(r"\b(?:entitled|eligible|allowed|permitted|subject|expected|required|obligated|obliged|apply|applies|applied)\s+(?:to|for)\s+", text):
        found = True
        heads.extend(_object_parts(text[m.end():])[0])
    for m in re.finditer(r"\b(?:is|are|was|were)\s+(?:the\s+|an?\s+)?(?!not\b)", text):
        found = True
        heads.extend(_object_parts(text[m.end():])[0])
    for i, t in enumerate(tokens):
        if i and t in _QUESTION_VERBS and t not in _PROPOSITION_EXEMPT:
            found = True
            active = True
            heads.extend(_object_parts(" ".join(tokens[i + 1:]))[0])
    if not found:
        return _has_word(clause, head)
    if not active:
        heads.extend(subject_heads)                                # "Research groups are registered ..., subject to Board approval"
    return any(_word_forms(w) & _word_forms(head) for w in heads)


def _target_head(target: str) -> Optional[str]:
    """The head noun of what a question's action is predicated of: the last content word before the first preposition. A
    target that opens with a preposition ("attend PER year") names no object (the object was moved: "how many conferences")."""
    tokens = _TOKEN.findall(cue_text(target).lower())
    if tokens and tokens[0] in _PREPOSITIONS:
        return None
    words: List[str] = []
    for i, token in enumerate(tokens):
        if token in _PREPOSITIONS and words:
            break
        if words and _reduced_relative(token, tokens[i - 1], tokens[i + 1] if i + 1 < len(tokens) else ""):
            break                                                  # "research INVOLVING animals": the head is research
        if token in _TARGET_DROP or token in _PREPOSITIONS or token.isdigit() or token in _PROPOSITION_EXEMPT or token in _TIME_WORDS:
            continue                                               # "attend EACH YEAR": a time adverbial names no object
        words.append(token)
    return _stem(words[-1]) if words else None


def _nominal_head(after: str) -> Optional[str]:
    """The object of an asked nominal: its "of"/"for" complement ("Dean approval FOR every PROJECT"), unless that names the
    agent ("approval of the VCRGS"); otherwise what follows it."""
    m = re.match(r"\s*(?:of|for)\s+(.*)", after, re.I | re.S)
    if m and not _RESEARCH_AUTHORITY_NOUN.search(" ".join(m.group(1).split()[:4])):
        return _target_head(m.group(1))
    return _target_head(after)


def _target_restrictions(target: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """What restricts an asked target's head: its noun modifiers ("my research GRANT application"), its "of" complement
    ("costs OF THE HOST INSTITUTION") and its "for" complement ("the application FOR A RESEARCH GRANT")."""
    tokens = _TOKEN.findall(cue_text(target).lower())
    if tokens and tokens[0] in _PREPOSITIONS:
        return (), (), ()
    words: List[str] = []
    of_words: List[str] = []
    for_words: List[str] = []
    complement: Optional[List[str]] = None
    for i, token in enumerate(tokens):
        if token in _PREPOSITIONS:
            if token in ("of", "for") and words and complement is None:
                complement = of_words if token == "of" else for_words
                continue
            break
        if token in ("and", "or") or token in _SUBORDINATORS:
            break
        if words and _reduced_relative(token, tokens[i - 1], tokens[i + 1] if i + 1 < len(tokens) else ""):
            break
        if token in _TARGET_DROP or token.isdigit() or token in _PROPOSITION_EXEMPT or token in _TIME_WORDS or token in _ASK_NEUTRAL:
            continue
        (words if complement is None else complement).append(token)
    return tuple(_stem(w) for w in words[:-1]), tuple(_stem(w) for w in of_words), tuple(_stem(w) for w in for_words)


def _target_modifiers(target: str) -> Tuple[str, ...]:
    """Every restricting word of an asked target (the anaphor rule: all of them are stated by the statement)."""
    premodifiers, of_words, for_words = _target_restrictions(target)
    return premodifiers + of_words + for_words


#: "the UNIVERSITY'S share": a percentage of one owner's share is stated under that owner's share.
_SHARE_OWNER = re.compile(r"\b(university|uos|inventors?|colleges?|departments?|vcrgs)(?:['’]s|s['’]|['’])?\s+shares?\b", re.I)
#: Objects sharing a generic head ("advance"): a unit about the rival never answers for the asked one unless it names it.
_COMPETING_OBJECTS: Tuple[Tuple[str, str], ...] = ((r"research\s+advances?", r"petty\s+cash"),)
#: The verb a permission governs ("NOT ALLOWED TO combine", "MAY apply"), or the subject of a passive permission.
_PERMISSION_GOVERNS = (re.compile(r"\b(?:may|can|could|shall|should|must)\s+(?:not\s+|only\s+|also\s+)*(?!be\b)([\w-]+)", re.I),
                       re.compile(r"\b(?:allowed|permitted|entitled|eligible|authori[sz]ed)\s+to\s+([\w-]+)", re.I))
_PERMISSION_PASSIVE = re.compile(r"\b(?:may|can|could|shall|should|must|is|are)\s+(?:not\s+|only\s+|also\s+)*(?:be\s+)?(?:permitted|allowed)\b(?!\s+to)", re.I)


def _permission_governs(statement: str, actions: Sequence[Tuple[str, Optional[str]]]) -> bool:
    """A permission answers an action only when it governs that action ("not allowed to COMBINE teaching" never answers
    "can faculty TEACH?")."""
    if not actions:
        return True
    text = cue_text(statement)
    governed = [m.group(1) for pattern in _PERMISSION_GOVERNS for m in pattern.finditer(text)]
    subjects = [text[:m.start()] for m in _PERMISSION_PASSIVE.finditer(text)]
    for action, _head in actions:
        pattern = _ACTION_PATTERNS[action]
        if not (any(pattern.match(g) for g in governed) or any(pattern.search(sub) for sub in subjects)):
            return False
    return True


#: A requirement ask that names the required thing: "Is IRB CLEARANCE necessary ...", "Do I need prior APPROVAL to ...",
#: "Should researchers get Dean APPROVAL for ...".
_REQUIRED_ASK = (
    re.compile(r"^\s*(?:(?:and|also|so|but)\s+)?(?:is|are)\s+((?:[\w'’-]+\s+){1,5}?)(?:required|mandatory|compulsory|necessary|needed|obligatory)\b", re.I),
    re.compile(r"^\s*(?:(?:and|also|so|but)\s+)?(?:do|does)\s+(?:i|we|you|faculty(?:\s+members?)?|researchers?|one)\s+(?:need|require)\s+(?!to\b)"
               r"((?:[\w'’-]+\s*){1,6})", re.I),
    re.compile(r"^\s*(?:(?:and|also|so|but)\s+)?(?:must|should)\s+(?:i|we|faculty(?:\s+members?)?|researchers?)\s+(?:get|obtain|have|hold|secure|seek|receive)\s+"
               r"((?:[\w'’-]+\s*){1,6})", re.I),
)
#: Where a statement states what is required: the object of an obligation ("must OBTAIN IRB APPROVAL", "requires the
#: APPROVAL of"), or the subject of a required-predicate ("prior APPROVAL is required").
_OBLIGATION_OBJECT = re.compile(r"\b(?:must|shall|should|(?:is|are)\s+(?:required|obliged|obligated|expected)\s+to|(?:need|needs|has|have)\s+to|"
                                r"requir(?:e|es|ing))\s+([^.;:]{1,160})", re.I)
_OBLIGATION_SUBJECT = re.compile(r"\b(?:is|are|be|was|were)\s+(?:also\s+|strictly\s+|always\s+|first\s+)?(?:required|mandatory|compulsory|necessary|"
                                 r"needed|obligatory|essential)\b", re.I)


def _required_head(clause: str) -> Optional[str]:
    for segment in clause_segments(clause) or [clause]:
        for pattern in _REQUIRED_ASK:
            m = pattern.search(cue_text(segment))
            if m:
                return _target_head(m.group(1))
    return None


def _head_stated(text: str, head: str) -> bool:
    """The head, or its action class ("APPROVAL" is stated by "must be APPROVED")."""
    if _has_word(text, head):
        return True
    for _action, pattern in _COMPILED_ACTIONS:
        m = pattern.match(head)
        if m and m.end() == len(head) and pattern.search(text):
            return True
    return False


def _requirement_binds(head: Optional[str], statement: str) -> bool:
    """The asked required thing is what the statement requires ("must obtain written informed consent" never states that
    IRB clearance is necessary; "the PI must hold a Ph.D." never states that a PI is mandatory)."""
    if head is None:
        return True
    text = cue_text(statement)
    for m in _OBLIGATION_OBJECT.finditer(text):
        if _head_stated(" ".join(m.group(1).split()[:10]), head):
            return True
    for m in _OBLIGATION_SUBJECT.finditer(text):
        if _head_stated(_COORDINATE_SPLIT.split(text[:m.start()])[-1], head):
            return True
    return False


_PERCENT_BASE = re.compile(r"\b(?:percentage|percent|proportion|fraction)\s+of\s+((?:[\w'’-]+\s+){1,5}?)"
                           r"(?=(?:goes|go|does|do|is|are|will|shall|may|must|can|received|paid|deducted|kept|distributed|allocated|retained)\b)", re.I)


def _percent_base(text: str) -> Tuple[str, ...]:
    """The base of an asked percentage ("what percentage OF TOTAL LICENSING REVENUE does the VCRGS receive")."""
    m = _PERCENT_BASE.search(cue_text(text))
    if not m:
        return ()
    return tuple(_stem(t) for t in _TOKEN.findall(m.group(1).lower()) if t not in _DETERMINERS and t not in _TOPIC_FILLERS)


def _relation_frame(clause: str, details: Sequence[Detail], yes_no: bool) -> RelationFrame:
    """The relation the clause asks about (substantive segments only; navigation is Institutional's)."""
    text = cue_text(" ".join(s for s in clause_segments(clause) if not _navigation_segment(s)) or clause)
    families = {d.family for d in details} & _ASK_FAMILIES
    implied = frozenset(a for f in families for a in _IMPLIED_ACTIONS.get(f, frozenset()))
    if families & {"deadline", "advance"}:
        implied = implied - {"submit", "apply"}                    # a timing ask names the action it times
    relational = yes_no or bool(families & _RELATIONAL_FAMILIES)
    actions: List[Tuple[str, Optional[str]]] = []
    modifiers: List[Tuple[str, Tuple[str, ...]]] = []
    restrictions: List[Tuple[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]] = []
    for action, pattern in _COMPILED_ACTIONS:
        if action in implied or any(a == action for a, _h in actions):
            continue
        for m in pattern.finditer(text):
            form = _occurrence_form(text, m)
            if form == "verbal" or (form == "nominal" and relational):
                # an asked action, with the head noun the question predicates it of (a participle's target is its subject)
                participle = form == "verbal" and bool(re.search(r"(?:ed|en)$", m.group(0).lower()))
                target = _local_target(text, m) if not participle else " ".join(re.findall(r"[\w'’-]+", text[:m.start()])[-4:])
                actions.append((action, _target_head(target) if form == "verbal" else _nominal_head(text[m.end():])))
                if form == "verbal" and not participle:
                    modifiers.append((action, _target_modifiers(target)))
                    premodifiers, of_words, for_words = _target_restrictions(target)
                    restrictions.append((action, (premodifiers, of_words, for_words)))
                break
    purpose = _PURPOSE.search(text)
    owner = _SHARE_OWNER.search(text)
    required = _required_head(clause) if "requirement" in families else None
    base = _percent_base(text) if "percentage" in families and not owner else ()
    competitors = tuple((asked, rival) for asked, rival in _COMPETING_OBJECTS if re.search(asked, text, re.I))
    proposition = tuple(_proposition_words(text)) if yes_no else ()
    structure = _ask_structure(text, yes_no)
    if structure.predicate and structure.asked_object and any(w not in _TIME_WORDS for w in structure.asked_object):
        fronted = " ".join(structure.asked_object)
        for i, (action, head) in enumerate(actions):
            if head is None and _ACTION_PATTERNS[action].fullmatch(structure.predicate):
                actions[i] = (action, _target_head(fronted))     # "how many REGIONAL CONFERENCES can faculty attend"
                premodifiers, of_words, for_words = _target_restrictions(fronted)
                restrictions.append((action, (premodifiers, of_words, for_words)))
                modifiers.append((action, _target_modifiers(fronted)))
    frame = RelationFrame(_roles_in(text), tuple(actions), purpose.group(1).lower() if purpose else None, proposition,
                         bool(families & {"deadline", "advance", "duration", "frequency"}),
                         _binding(text).recipients if "percentage" in families else (), "permission" in families,
                         _stem(owner.group(1)) if owner else None, competitors, base, bool(families & {"quantity", "maximum", "minimum"}),
                         tuple(modifiers), required,
                         frozenset(_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _TARGET_DROP and t not in _PREPOSITIONS),
                         structure.subject, structure.subject_roles, structure.generic, structure.np, structure.text, structure.complement,
                         structure.copular, yes_no, structure.predicate, structure.passive, structure.asked_object, _ask_agents(text),
                         _ask_arguments(text, structure.start, structure.predicate) if (yes_no or structure.predicate) else (), _ask_period(text),
                         tuple(_negated_in(text, proposition)), tuple(restrictions))
    return frame


def _proposition_words(text: str) -> List[str]:
    """The content words of a yes/no proposition that no family, object, category, role, action, other domain or research
    topic already accounts for: they must be stated in the unit ("a SPECIAL grant for the FIRST YEAR")."""
    work = cue_text(text).lower()

    def blank(m: "re.Match[str]") -> str:
        return " " * (m.end() - m.start())

    for _family, _label, pattern, _check in _COMPILED_FAMILIES:
        work = pattern.sub(blank, work)
    for _sid, pattern in _OTHER_CUES:
        work = pattern.sub(blank, work)
    for _role, pattern in _COMPILED_ROLES:
        work = pattern.sub(blank, work)
    actions: List[str] = []
    for action, pattern in _COMPILED_ACTIONS:                       # an action word is required by its class ("attend" / "participation")
        if pattern.search(work):
            actions.append("@" + action)
        work = pattern.sub(blank, work)
    for pattern, _concept, anchor in _COMPILED_CONCEPTS:
        if anchor:
            work = pattern.sub(blank, work)
    words: List[str] = list(actions)
    for token in _TOKEN.findall(work):
        if token not in _PROPOSITION_KEEP and not token.isdigit() and (
                len(token) < 3 or token in _ASK_STOP or token in _PROPOSITION_EXEMPT or token in _TOPIC_FILLERS):
            continue
        stem = _stem(token)
        if stem not in words:
            words.append(stem)
    return words


#: Words a yes/no proposition keeps although they are function words: they change what is asked ("ONLY to the faculty
#: member", "WITHOUT the project budget", "AFTER the study", "30%").
_PROPOSITION_KEEP = frozenset({"only", "without", "before", "after", "prior", "during", "until", "both"})
_NEGATORS = frozenset({"no", "not", "non", "without", "never", "neither", "nor", "except", "excluding", "unlike", "cannot"})


def _negated_in(text: str, words: Sequence[str]) -> List[str]:
    """The proposition words the question itself negates ("NON-compensated consultancy")."""
    tokens = _TOKEN.findall(cue_text(text).lower())
    out: List[str] = []
    for i, token in enumerate(tokens):
        if i and tokens[i - 1] in _NEGATORS:
            stem = _stem(token)
            if stem in words and stem not in out:
                out.append(stem)
            for action, pattern in _COMPILED_ACTIONS:
                if pattern.fullmatch(token) and "@" + action in words and "@" + action not in out:
                    out.append("@" + action)
    return out


def _stated_positively(text: str, word: str, negated: Sequence[str] = ()) -> bool:
    """``word`` occurs in ``text`` outside a negation ("at NO cost", "NON-commercial", "NOT viewed as") unless the question
    negates it too."""
    wanted = _word_forms(word) | frozenset(_TEMPORAL_FORMS.get(word, ()))
    tokens = _TOKEN.findall(text.lower())
    for i, token in enumerate(tokens):
        if _word_forms(token) & wanted and (word in negated or not (i and tokens[i - 1] in _NEGATORS)):
            return True
    return False


_TEMPORAL_FORMS = {"before": ("prior", "pre", "ahead", "advance"), "prior": ("before", "pre"), "after": ("following", "upon", "once"),
                   "during": ("while", "throughout"), "until": ("till",)}


def _class_stated_positively(action: str, text: str, negated: bool = False) -> bool:
    pattern = _ACTION_PATTERNS[action]
    for m in pattern.finditer(text):
        before = _TOKEN.findall(text[:m.start()].lower())
        if negated or not (before and before[-1] in _NEGATORS):
            return True
    return False


def _coordinate_part(text: str, position: int) -> str:
    start, end = 0, len(text)
    for m in _COORDINATE_SPLIT.finditer(text):
        if m.end() <= position:
            start = m.end()
        elif m.start() >= position:
            end = m.start()
            break
    return text[start:end]


def _actions_related(actions: Sequence[Tuple[str, Optional[str]]], statement: str, values: Sequence[Callable[[str], bool]],
                     timed: bool = False, modifiers: Sequence[Tuple[str, Tuple[str, ...]]] = (), asked: FrozenSet[str] = frozenset(),
                     restrictions: Sequence[Tuple[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]] = (), unit: str = "") -> bool:
    """Every asked action is stated of the asked object in the statement; a time value must be in the same coordinate part
    as the action ("published ... OR commitment to submit ... within six months" times the commitment)."""
    text = cue_text(statement)
    asked_modifiers = dict(modifiers)
    asked_restrictions = dict(restrictions)
    for action, head in actions:
        stated = False
        for m in _ACTION_PATTERNS[action].finditer(text):
            if _statement_noun(text, m):
                continue                                           # "the TRAVEL budget" states no travelling
            if head is not None and not _target_matches(text, m, head, asked_modifiers.get(action, ()), asked,
                                                        asked_restrictions.get(action, ((), (), ())), unit):
                continue
            if timed and values and not any(v(_coordinate_part(text, m.start())) for v in values):
                continue
            stated = True
            break
        if not stated:
            return False
    return True


def _purpose_ok(purpose: Optional[str], text: str) -> bool:
    """An asked purpose is the one the statement states ("for commercial purposes" never answers "for research purposes")."""
    return purpose is None or purpose in {m.group(1).lower() for m in _PURPOSE.finditer(text)}


def _proposition_contained(words: Sequence[str], unit: str, negated: Sequence[str] = ()) -> bool:
    text = cue_text(unit)
    return all(_class_stated_positively(w[1:], text, w in negated) if w.startswith("@") else _stated_positively(text, w, negated)
               for w in words)


def _relation_ok(frame: "RelationFrame", statement: str, unit: str, values: Sequence[Callable[[str], bool]]) -> bool:
    """The deterministic proof that the statement states the asked relation."""
    text = cue_text(statement)
    return (_roles_stated(frame.roles, statement, unit) and _recipients_stated(frame.recipients, statement)
            and _actions_related(frame.actions, statement, values, frame.timed, frame.modifiers, frame.words, frame.restrictions, unit)
            and _purpose_ok(frame.purpose, text) and _proposition_contained(frame.proposition, unit, frame.negated)
            and (not frame.permission or _permission_governs(statement, frame.actions))
            and _competitors_ok(frame.competitors, unit) and _share_statement_ok(frame.share_owner, statement, unit, frame.percent_base)
            and _requirement_binds(frame.required, statement) and _position_ok(frame, statement, unit)
            and _conditioned_role_ok(frame.roles, statement) and _period_ok(frame.period, statement))


# ---------------------------------------------------------------------------
# Position binding: a party is proved in the position the question gives it
# ---------------------------------------------------------------------------
#: A statement that names the asked party proves nothing unless the party occupies the asked position: the subject of
#: the asked predicate ("the VCRGS submits" is not "submits ... TO the VCRGS"), the agent of a passive ("determined BY
#: the Head" is not "determined by the VCRGS, upon the recommendation of the Head"), the patient of a passive
#: ("Uncompensated conference travel may be authorized" never answers for COMPENSATED travel), a recipient or source
#: ("report ... TO the IRB" is not "to the IACUC"). Read from the question's structure, proved in the statement's; what
#: the structure cannot settle (a pronoun subject, a parenthetical, a gerund after a preposition) is left unproved.
_AUX = frozenset({"do", "does", "did", "is", "are", "am", "was", "were", "can", "could", "may", "might", "must", "shall", "should", "will",
                  "would"})
_BE_FORMS = frozenset({"is", "are", "am", "was", "were", "be"})
_GENERIC_SUBJECTS = frozenset({"i", "we", "you", "one", "someone", "anyone", "faculty", "member", "members", "researcher", "researchers",
                               "academic", "academics", "university", "uos", "people", "person", "applicant", "applicants", "he", "she",
                               "they", "it", "there", "who", "anybody", "everyone", "everybody", "us"})
_QUESTION_VERBS = frozenset({
    "need", "needs", "have", "has", "get", "gets", "receive", "receives", "obtain", "obtains", "spend", "spends", "host", "hosts",
    "provide", "provides", "deliver", "delivers", "ensure", "ensures", "choose", "chooses", "assume", "assumes", "forward", "forwards",
    "offer", "offers", "hold", "holds", "retain", "retains", "reclaim", "reclaims", "establish", "establishes", "keep", "keeps",
    "conduct", "conducts", "allocate", "allocates", "substitute", "substitutes", "consult", "consults", "require", "requires",
    "include", "includes", "become", "becomes", "remain", "remains", "lose", "loses", "carry", "carries", "assist", "assists", "help",
    "helps", "promote", "promotes", "enable", "enables", "facilitate", "facilitates", "avoid", "avoids", "maintain", "maintains",
    "protect", "protects", "develop", "develops", "enhance", "enhances", "inform", "informs", "notify", "notifies", "give", "gives",
    "entitled", "eligible", "required", "allowed", "permitted", "responsible", "mandatory", "necessary", "available", "refundable",
    "transferable", "renewable", "taxable", "negotiable", "covered", "funded", "liable", "exempt", "able", "obliged", "expected",
    "supposed", "free", "possible", "report", "reports", "disclose", "discloses", "monitor", "monitors", "comply", "complies",
    "adhere", "adheres", "undergo", "undergoes", "follow", "follows", "serve", "serves", "engage", "engages", "settle", "settles",
    "reimburse", "reimburses", "bear", "bears", "earn", "earns", "lead", "leads", "count", "counts", "discretionary", "optional",
    "compulsory", "obligatory", "acceptable", "admissible", "permissible", "applicable", "payable", "deductible", "binding"})
_LIGHT_PREDICATES = frozenset({"receive", "receives", "get", "gets", "obtain", "obtains", "have", "has", "entitled", "eligible", "given",
                               "paid", "awarded", "granted", "offered", "provided", "need", "needs", "require", "requires", "spend",
                               "spends", "spent", "take", "takes", "keep", "keeps", "retain", "retains", "claim", "claims", "lose", "loses",
                               "covered", "funded", "reimbursed"})
_IRREGULAR_PARTICIPLES = frozenset({"set", "held", "given", "made", "taken", "kept", "put", "run", "paid", "sent", "met", "led", "found",
                                    "chosen", "done", "seen", "known", "drawn", "withheld", "borne", "shown", "spent", "lost"})
_RELATIVES = frozenset({"which", "who", "that", "whom", "whose"})
_ADVERBS = frozenset({"also", "then", "only", "not", "already", "still", "first", "directly", "immediately", "formally", "usually",
                      "normally", "always", "never", "ever", "actually", "really", "just", "even", "explicitly", "fully", "partially"})
_AUTHORITY_ROLES = frozenset({"dean", "chair", "vcrgs", "vcaa", "chancellor", "research_board", "research_committee", "tto",
                              "editorial_board", "chief_editor", "irb", "iacuc", "ethics_committee", "library", "purchasing_department",
                              "head", "committee", "board"})
_POPULATION_ROLES = frozenset({"student", "staff", "visitor", "postdoc", "research_assistant"})
_STRICT_ARGUMENTS = frozenset({"to", "from", "by", "among", "between", "without"})
_LENIENT_ARGUMENTS = frozenset({"for", "with", "on", "at"})
_PERIOD = re.compile(r"\b(?:(?:per|each|every|an?)\s+(?:(?:academic|calendar|working|fiscal|full)\s+)?(year|semester|month|week|day|trip|term)s?\b|"
                     r"in\s+(one|two|three|four|five|six|\d+)\s+(years|semesters|months|weeks|days)\b|(annually|yearly|monthly|weekly|daily|semesterly))",
                     re.I)
_PERIOD_WORDS = {"annually": "year", "yearly": "year", "monthly": "month", "weekly": "week", "daily": "day", "semesterly": "semester",
                 "years": "year", "semesters": "semester", "months": "month", "weeks": "week", "days": "day"}


@dataclass(frozen=True)
class _AskStructure:
    subject: Tuple[str, ...] = ()
    subject_roles: FrozenSet[str] = frozenset()
    generic: bool = False
    predicate: Optional[str] = None
    np: Tuple[str, ...] = ()
    asked_object: Tuple[str, ...] = ()
    start: int = 0
    passive: bool = False
    complement: Tuple[str, ...] = ()
    copular: Tuple[str, ...] = ()
    text: str = ""


def _is_verb_token(token: str, previous: str, following: str) -> bool:
    """A finite predicate of a question: a known verb, or an action-class word used as a verb."""
    if following == "be" or following in _AUX:
        return False                                               # "can conference TRAVEL be authorized": a noun
    if token in _QUESTION_VERBS:
        return True
    if following and re.search(r"(?:ed|en)$", following) and following not in _DETERMINERS and following not in ("open", "often") and \
            following not in _QUESTION_VERBS:
        return False                                               # "the publishing COST deducted": the -ed is the predicate
    if token.endswith("ing") or token in _NOUN_GERUNDS or token in _DETERMINERS or re.search(r"(?:or|ors|ist|ists|ant|ants|ee|ees)$", token):
        return False                                               # "the general SUPERVISOR", "the COORDINATOR": agent nouns
    return any(p.fullmatch(token) for _a, p in _COMPILED_ACTIONS) and not _NOMINAL.search(token)


def _np_head(tokens: Sequence[str]) -> Tuple[Tuple[str, ...], Optional[str]]:
    """The content words and head of a noun phrase: up to its first preposition, conjunction or reduced relative."""
    words: List[str] = []
    for i, t in enumerate(tokens):
        if t in _PREPOSITIONS or t in _SUBORDINATORS or t in ("and", "or"):
            break
        if i and (t.endswith("ing") or re.search(r"(?:ed|en)$", t)) and t not in _NOUN_GERUNDS and tokens[i - 1] not in _DETERMINERS and                 t not in ("open", "often"):
            break
        if t not in _DETERMINERS and t not in _AUX and t not in _ADVERBS:
            words.append(_stem(t))
    return tuple(words), (words[-1] if words else None)


def _content(tokens: Sequence[str]) -> Tuple[str, ...]:
    return tuple(_stem(t) for t in tokens if t not in _DETERMINERS and t not in _AUX and t not in _ADVERBS and t != "of" and t != "be")


def _ask_structure(text: str, yes_no: bool) -> _AskStructure:
    """The subject, predicate and voice of an ask with an auxiliary ("Does the VCRGS submit ...", "Is X approved by Y",
    "How much can the PI receive", "What percentage of X is deducted for Y"). A value ask whose auxiliary is a copula
    ("What is the ceiling") asks for a value, not a relation, and binds nothing but its subject's complement."""
    tokens = _TOKEN.findall(cue_text(text).lower())
    positions = [i for i, t in enumerate(tokens) if t in _AUX]
    if not positions:
        return _AskStructure()
    idx = positions[-1]                                            # "What is the number of dissertations I CAN supervise"
    aux = tokens[idx]
    rest = tokens[idx + 1:]
    lead = [t for t in tokens[:idx] if t not in _WH_LEAD]           # "what percentage of external grants" before "is"
    if idx and tokens[idx - 1] in ("who", "what", "which", "where", "when") and rest and \
            _is_verb_token(rest[0], aux, rest[1] if len(rest) > 1 else ""):
        return _AskStructure(start=idx)                            # "Who must APPROVE ...": the answer is the subject
    if len(positions) > 1 and rest and _is_verb_token(rest[0], aux, rest[1] if len(rest) > 1 else ""):
        before: List[str] = []
        for t in reversed(tokens[:idx]):
            if t in _PREPOSITIONS or t in _AUX or t in _WH_LEAD:
                break
            before.insert(0, t)
            if t in _GENERIC_SUBJECTS and t in ("i", "we", "you", "they", "he", "she", "it") or t in _DETERMINERS:
                break                                              # "dissertations [I] can supervise", "[the VCRGS] can"
        lead = [t for t in lead if t not in before]
        rest = before + rest                                       # the subject precedes the auxiliary
    subject: List[str] = []
    predicate: Optional[str] = None
    passive = False
    obj: Tuple[str, ...] = ()
    copular: Tuple[str, ...] = ()
    scope = next((i for i, t in enumerate(rest) if t in _PREPOSITIONS and t != "of"), len(rest))
    finite = next((i for i, t in enumerate(rest[:scope]) if i and _is_verb_token(t, rest[i - 1], rest[i + 1] if i + 1 < len(rest) else "")
                   and not (re.search(r"(?:ed|en)$", t) and t not in _QUESTION_VERBS)), None)
    be = next((i for i, t in enumerate(rest) if t == "be"), None)
    relative = finite is not None and any(t in _RELATIVES for t in rest[:finite])
    if be is not None and be == len(rest) - 1:
        subject = rest[:be]                                        # "How long can X be": a value of X
        finite = None
        be = None
    elif be is not None and (finite is None or be <= finite + 1 or relative):
        subject = rest[:be]
        passive = True
        predicate = rest[be + 1] if be + 1 < len(rest) else None
        if predicate is not None and not re.search(r"(?:ed|en)$", predicate) and predicate not in _IRREGULAR_PARTICIPLES:
            predicate = None                                       # "be full-time employees": a copula
    elif finite is not None:
        subject = rest[:finite]
        predicate = rest[finite]
        passive = aux in _BE_FORMS and bool(re.search(r"(?:ed|en)$", predicate))
        tail = rest[finite + 1:]
        obj = tuple(w for w in _content(tuple(t for t in itertools.takewhile(lambda t: t not in _PREPOSITIONS and t not in _SUBORDINATORS, tail)))
                    if w not in _TIME_WORDS)
        if not obj and lead and not yes_no and rest[finite] not in _LIGHT_PREDICATES:
            cut = max((i for i, t in enumerate(lead) if t in _PREPOSITIONS), default=-1)
            obj = _np_head(lead[cut + 1:])[0]                      # a fronted object: "how many CONFERENCES can I attend", "the number of DISSERTATIONS I can supervise"
    elif lead and rest and aux in _BE_FORMS and (re.search(r"(?:ed|en)$", rest[0]) or rest[0] in _IRREGULAR_PARTICIPLES) and rest[0] not in _DETERMINERS:
        subject = lead                                             # "what percentage of X IS DEDUCTED for Y"
        predicate = rest[0]
        passive = True
    elif subject:
        pass                                                       # a trailing "be": the subject is known
    else:
        participle = next((i for i, t in enumerate(rest) if i and (re.search(r"(?:ed|en)$", t) or t in _IRREGULAR_PARTICIPLES)
                           and t not in _DETERMINERS and (i + 1 >= len(rest) or rest[i + 1] in _PREPOSITIONS or
                                                                                     rest[i + 1].endswith("ly") or re.search(r"(?:ed|en)$", rest[i + 1]))), None)
        trailing = _trailing_predicate(rest, participle) if participle is not None else None
        if participle is not None and trailing is not None and _is_verb_token(rest[trailing], rest[trailing - 1], rest[trailing + 1] if trailing + 1 < len(rest) else ""):
            subject = rest[:trailing]                              # "conferences attended at no cost ... REQUIRE the same process"
            predicate = rest[trailing]
            tail = rest[trailing + 1:]
            obj = tuple(w for w in _content(tuple(itertools.takewhile(lambda t: t not in _PREPOSITIONS and t not in _SUBORDINATORS, tail)))
                        if w not in _TIME_WORDS)
        elif participle is not None and trailing is not None:
            subject = rest[:trailing]                              # "conferences attended at no cost ... DISCRETIONARY": a copula
        elif participle is not None and aux in _BE_FORMS:
            subject = rest[:participle]
            predicate = rest[participle]
            passive = True
        else:
            second = next((i for i, t in enumerate(rest) if i and t in _DETERMINERS and rest[i - 1] not in _PREPOSITIONS), None)
            subject = rest[:second] if second is not None else list(itertools.takewhile(lambda t: t not in _PREPOSITIONS or t == "of", rest))
            if second is not None and yes_no:
                copular = _np_complement(rest[second:])            # "Is X THE GENERAL SUPERVISOR OF THE RESEARCH GROUPS"
    subject_text = " ".join(subject)
    roles = _roles_in(subject_text)
    view = subject_text
    for role, pattern in _COMPILED_ROLES:
        if role in roles:
            view = pattern.sub(" ", view)
    view_tokens = _TOKEN.findall(view)
    np, head = _np_head(view_tokens)
    complement = _np_complement(view_tokens)[1:] if np else ()
    if predicate is None and (not yes_no or passive):
        return _AskStructure(np=np if not passive else (), start=idx, complement=complement if not passive else ())
    generic = not roles and (head is None or head in _GENERIC_SUBJECTS and all(w in _GENERIC_SUBJECTS or w in _TOPIC_FILLERS for w in np))
    words = () if generic else tuple(w for w in _content(view_tokens) if w not in _PREPOSITIONS)
    return _AskStructure(words, roles, generic, predicate, np if not generic else (), obj, idx, passive,
                         complement if not generic else (), copular, subject_text)


_WH_LEAD = frozenset({"what", "which", "how", "much", "many", "long", "often", "early", "who", "when", "where", "whom", "whose", "why",
                      "percentage", "percent", "proportion", "share", "fraction", "number", "amount"})


def _trailing_predicate(rest: Sequence[str], participle: int) -> Optional[int]:
    """The index of a token left after the participle's prepositional phrases ("attended at no cost to the University
    DISCRETIONARY", "... REQUIRE the same approval process"); None when the phrases run to the end."""
    i = participle + 1
    while i < len(rest):
        t = rest[i]
        if t in _PREPOSITIONS or re.search(r"(?:ed|en)$", t) or t.endswith("ly") or t in ("and", "or"):
            i += 1
            while i < len(rest) and rest[i] not in _PREPOSITIONS and not _is_verb_token(rest[i], rest[i - 1], rest[i + 1] if i + 1 < len(rest) else "") \
                    and not re.search(r"(?:ed|en)$", rest[i]):
                i += 1
            continue
        return i
    return None


def _np_complement(tokens: Sequence[str]) -> Tuple[str, ...]:
    """The head of a noun phrase followed by the content words of its of/for complement (through "and"/"or"):
    ("supervisor", "journal") for "the general supervisor of the journal"."""
    np, head = _np_head(tokens)
    if head is None:
        return ()
    rest = list(tokens)
    while rest and rest[0] not in ("of", "for"):
        rest = rest[1:]
    words: List[str] = []
    for i, t in enumerate(rest[1:]):
        if (t in _PREPOSITIONS and t not in ("of", "for")) or t in _SUBORDINATORS or t == "be" or t in _AUX or t in (",", ";", ":", ".", "?", "(", ")"):
            break
        if i and (t.endswith("ing") or re.search(r"(?:ed|en)$", t)) and t not in _NOUN_GERUNDS and rest[i] not in _DETERMINERS and \
                t not in ("open", "often"):
            break
        if t not in _DETERMINERS and t not in ("of", "for", "and", "or") and t not in _ADVERBS:
            words.append(_stem(t))
    return (head,) + tuple(words)


def _phrase_after(tokens: Sequence[str], start: int, stop: Optional[str] = None) -> Tuple[str, ...]:
    """The noun phrase that follows a preposition (through "and"/"or" and one "of"), without its determiners, ending at
    ``stop`` (the question's predicate: "to the University | discretionary")."""
    out: List[str] = []
    i = start
    while i < len(tokens):
        t = tokens[i]
        previous = tokens[i - 1] if i else ""
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if (t in _PREPOSITIONS and t != "of") or t in _SUBORDINATORS or t in _AUX or t == "be" or (stop is not None and t == stop):
            break
        if out and t not in ("and", "or") and (re.search(r"(?<!e)ed$", t) and t not in _DETERMINERS or t in _IRREGULAR_PARTICIPLES) and \
                (following in _PREPOSITIONS or not following or following in ("and", "or")):
            break                                                  # a reduced relative: "invoices APPROVED by the Dean"
        if (t not in _DETERMINERS or t in ("no", "not")) and t not in ("and", "or", "of") and not t.isdigit():
            out.append(_stem(t))
        i += 1
    return tuple(out)


def _split_tokens(text: str) -> List[str]:
    """Tokens with punctuation kept as "," so that phrase boundaries survive."""
    return re.findall(r"[a-z0-9]+(?:['’][a-z]+)?|[,;:.?()]", cue_text(text).lower())


def _ask_agents(text: str) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Each participle the question gives an agent: ("sponsored", ("university",))."""
    tokens = _split_tokens(text)
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for i, t in enumerate(tokens):
        if t == "by" and i and (re.search(r"(?:ed|en)$", tokens[i - 1]) or tokens[i - 1] in _IRREGULAR_PARTICIPLES):
            words = _phrase_after(tokens, i + 1)
            if words:
                out.append((_stem(tokens[i - 1]), words))
    return tuple(out)


def _ask_arguments(text: str, start: int, predicate: Optional[str] = None) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """The prepositional arguments after the auxiliary ("to the IACUC", "from the college budget", "among colleges")."""
    tokens = _split_tokens(text)
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for i, t in enumerate(tokens):
        if i <= start or t not in (_STRICT_ARGUMENTS | _LENIENT_ARGUMENTS):
            continue
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t == "by" and i and (re.search(r"(?:ed|en)$", tokens[i - 1]) or tokens[i - 1] in _IRREGULAR_PARTICIPLES):
            continue                                               # an agent (see _ask_agents)
        if t == "to" and not (following in _DETERMINERS or following in ("all", "both") or _roles_in(following) or following in _GENERIC_SUBJECTS):
            continue                                               # an infinitive
        if t == "for" and _PURPOSE.match(" ".join(tokens[i:i + 4])):
            continue                                               # a purpose (see _purpose_ok)
        words = _phrase_after(tokens, i + 1, predicate)
        if words:
            out.append((t, words))
    return tuple(out)


def _ask_period(text: str) -> Optional[Tuple[str, str]]:
    m = _PERIOD.search(cue_text(text))
    if not m:
        return None
    if m.group(1):
        return (_PERIOD_WORDS.get(m.group(1).lower(), m.group(1).lower()), "")
    if m.group(2):
        return (_PERIOD_WORDS.get(m.group(3).lower(), m.group(3).lower()), m.group(2).lower())
    return (_PERIOD_WORDS[m.group(4).lower()], "")


def _period_ok(period: Optional[Tuple[str, str]], statement: str) -> bool:
    """An asked period ("per month", "in two years") is the stated one."""
    if period is None:
        return True
    for m in _PERIOD.finditer(cue_text(statement)):
        stated = _ask_period(m.group(0))
        if stated == period:
            return True
    return False


def _statement_words(text: str) -> Tuple[str, ...]:
    return _content(_TOKEN.findall(cue_text(text).lower()))


def _subject_span(text: str, position: int) -> Optional[str]:
    """The words a finite verb at ``position`` is predicated of: the last comma-free stretch before it (or the noun a
    relative pronoun refers to), without a printed heading or list label. None when the subject cannot be read (a gerund
    after a preposition, a pronoun)."""
    before = _strip_heading(text[:position])
    before = re.sub(r"^[\s•o]*(?:\d+(?:\.\d+)*\.|[a-z]\.)\s*", "", before)
    if not before.strip() and re.match(r"\w+ing\b", text[position:]):
        return None                                                # "4. ASSESSING the colleges' needs": a duty whose subject is the list's
    tokens = _split_tokens(before)
    while tokens and tokens[-1] == "to":
        tokens = tokens[:-1]                                       # "are permitted TO spend": the governing verb's subject
    if tokens and tokens[-1] in _PREPOSITIONS:
        return None                                                # "by ENSURING": the gerund's subject is elsewhere
    segments: List[List[str]] = [[]]
    for t in tokens:
        if t in (",", ";", ":", "(", ")"):
            segments.append([])
        else:
            segments[-1].append(t)
    segment = segments[-1]
    if segment and segment[0] in _RELATIVES:
        segment = segments[-2] if len(segments) > 1 else []
        cuts = [i for i, t in enumerate(segment) if t in _PREPOSITIONS and t != "of"]
        if cuts:
            segment = segment[cuts[-1] + 1:]                       # "hosted by COLLEGES, which ...": the noun the pronoun refers to
    if any(t in ("they", "it", "he", "she", "them", "these", "those", "such", "this") for t in segment[:2]):
        return None
    return " ".join(segment)


def _agent_after(text: str, end: int, nominal: bool = False) -> Optional[Tuple[str, ...]]:
    """The "by X" agent right after a passive or nominal occurrence ("approved in writing by the VCAA", "the approval of
    the VCRGS"); None when there is none."""
    m = re.match(r"\s*(?:[\w-]+\s+){0,2}?(?:by|from" + ("|of" if nominal else "") + r")\s+", text[end:], re.I)
    if not m:
        return None
    tokens = _split_tokens(text[end + m.end():])
    return _phrase_after(tokens, 0)


def _agent_before(text: str, start: int) -> Optional[Tuple[str, ...]]:
    """The possessor before a nominal ("the CHANCELLOR'S approval")."""
    m = re.search(r"([\w-]+(?:\s+[\w-]+){0,3}?)['’]s?\s+(?:written\s+|prior\s+|final\s+|formal\s+)?$", text[:start], re.I)
    return tuple(_content(_TOKEN.findall(m.group(1).lower()))) if m else None


def _party_in(words: Sequence[str], roles: FrozenSet[str], span: Optional[str]) -> bool:
    """The asked party (its lexicon roles and its other words) is in the span."""
    if span is None:
        return False
    return roles <= _roles_in(span) and all(_has_word(span, w) for w in words)


def _occurrence_voice(text: str, m: "re.Match[str]") -> str:
    """How a predicate occurs in a statement: "active" ("the IACUC ENSURES"), "passive" ("must be APPROVED by"), "nominal"
    ("prior APPROVAL is required", "the Chancellor's APPROVAL") or "modifier" ("an APPROVED project")."""
    word = m.group(0).lower().split()[-1]
    before = _TOKEN.findall(text[:m.start()].lower())
    after = _TOKEN.findall(text[m.end():].lower())[:2]
    previous = before[-1] if before else ""
    following = after[0] if after else ""
    participle = bool(re.search(r"(?:ed|en)$", word)) and word not in ("open", "often") or word in _IRREGULAR_PARTICIPLES
    if _NOMINAL.search(word) and not participle and (
            previous in _DETERMINERS or re.search(r"['’]s?$", text[:m.start()].rstrip()) or following in ("of", "by", "from", "is", "are", "was",
            "were", "for", "to", "required", "needed", "necessary", "mandatory") or previous in _NOMINAL_LEADS):
        return "nominal"
    if participle:
        recent = [t for t in before[-3:] if t not in _ADVERBS]
        return "passive" if (recent and recent[-1] in _BE_AUX) or previous.endswith("ly") or "by" in after or \
            (previous not in _DETERMINERS and word in _IRREGULAR_PARTICIPLES) else "modifier"
    return "active"


_NOMINAL_LEADS = frozenset({"prior", "written", "final", "formal", "subject", "upon", "with", "without", "requires", "require", "requiring",
                            "following", "pending", "obtain", "obtains", "receive", "receives", "seek", "seeks", "need", "needs", "explicit",
                            "official", "necessary", "appropriate", "express", "vcrgs", "board", "dean", "chancellor", "committee"})


_INFINITIVES = frozenset({"support", "fund", "cover", "help", "assist", "promote", "enable", "finance", "develop", "sustain", "strengthen",
                          "ensure", "facilitate", "maintain", "protect", "enhance", "meet", "avoid", "participate", "host", "attend", "obtain",
                          "submit", "apply", "provide", "deliver", "receive", "reach", "achieve", "identify", "address", "conduct", "perform",
                          "make", "take", "use", "keep", "manage", "monitor", "review", "approve", "evaluate", "assess", "verify", "purchase",
                          "acquire", "publish", "disseminate", "encourage", "foster", "recover", "pay", "settle", "transfer", "register",
                          "resolve", "reclaim", "close", "extend", "renew", "join", "form", "establish", "create", "propel", "kickstart",
                          "safeguard", "protect", "better", "commercialize", "commercialise", "exploit", "negotiate", "sign", "file", "seek"})


def _infinitive(token: str, following: str) -> bool:
    return token in _INFINITIVES or _is_verb_token(token, "to", following)


def _predicate_pattern(predicate: str) -> "re.Pattern[str]":
    for action, pattern in _COMPILED_ACTIONS:
        if pattern.fullmatch(predicate):
            return pattern
    forms = "|".join(sorted((re.escape(f) for f in _word_forms(predicate)), key=len, reverse=True))
    return re.compile(r"\b(?:" + forms + r")(?:s|es|d|ed|ing)?\b", re.I)


def _position_ok(frame: "RelationFrame", statement: str, unit: str = "") -> bool:
    text = _strip_heading(cue_text(statement))
    if not _agents_bound(frame.agents, text) or not _arguments_bound(frame.arguments, text, frame.yes_no):
        return False
    if frame.predicate is None:
        if frame.yes_no and (frame.subject_np or frame.subject_roles):
            first = _first_subject(text)
            return _party_in(frame.subject_np, frame.subject_roles, first) and _complement_agrees(frame.subject_np, frame.subject_complement, first) \
                and _copular_agrees(frame.copular, text)
        if frame.subject_np and frame.subject_complement:
            return _complement_agrees(frame.subject_np, frame.subject_complement, _first_subject(text)) and \
                (not frame.counted or _complement_stated(frame.subject_complement, text))
        return True
    pattern = _predicate_pattern(frame.predicate)
    occurrences = [m for m in pattern.finditer(text) if not _statement_noun(text, m) and not re.search(r"\d", m.group(0))]
    if not occurrences:
        if frame.subject_roles and not frame.subject and not frame.yes_no and all(_acts_as(text, r) for r in frame.subject_roles):
            return True                                            # "how many projects can a PI lead" / "may not act as principal investigators for two projects"
        if frame.predicate not in _LIGHT_PREDICATES and frame.predicate not in _PROPOSITION_EXEMPT and \
                not (not frame.yes_no and _lead_in_states(pattern, unit, statement)):
            return False                                           # "when can the advance be CARRIED forward": the act is not stated
        if not frame.yes_no and not frame.subject_roles:
            return True                                            # a value ask about no named party binds nothing more
        return _subject_bound(frame, text, None, unit)             # a light or synonym predicate: the statement's subject
    if any(_subject_bound(frame, text, m, unit) for m in occurrences):
        return True
    return not frame.yes_no and frame.subject_generic              # "30% of research grants DEDUCTED for": a value, no party


def _complement_agrees(np: Sequence[str], complement: Sequence[str], first: Optional[str]) -> bool:
    """When the asked subject restricts its head with an of/for complement ("the sabbatical leave OF A FACULTY MEMBER")
    and the statement's subject restricts the same head with one ("Sabbatical leave FOR VISITING RESEARCHERS"), the two
    share a word; otherwise the statement is about something else."""
    if not np or not complement or not first:
        return True
    tokens = _TOKEN.findall(first.lower())
    stated = _np_complement(tokens)
    if not stated or not (_word_forms(stated[0]) & _word_forms(np[-1])) or len(stated) == 1:
        return True
    asked = [w for w in complement if w not in _TOPIC_FILLERS]
    return not asked or any(_word_forms(a) & _forms_of(stated[1:]) for a in asked)


def _complement_stated(complement: Sequence[str], statement: str) -> bool:
    """The restricting words of a value ask's subject ("the number of EXTERNAL researchers") are in the statement that
    states the value, not only elsewhere in the unit."""
    return all(_anchor_stated(w, statement) for w in complement
               if w not in _TOPIC_FILLERS and w not in _TARGET_DROP and w not in _TIME_WORDS and w not in _GENERIC_SUBJECTS and not w.isdigit())


def _copular_agrees(copular: Sequence[str], text: str) -> bool:
    """The complement of an asked predicate nominal ("the general supervisor OF THE RESEARCH GROUPS") is the stated one
    ("the general supervisor of the journal")."""
    if len(copular) < 2:
        return True
    asked = [w for w in copular[1:] if w not in _TOPIC_FILLERS]
    if not asked:
        return True
    tokens = _split_tokens(text)
    for i, t in enumerate(tokens):
        if _word_forms(t) & _word_forms(copular[0]):
            stated = _np_complement(tokens[i:])
            if len(stated) > 1 and not any(_word_forms(a) & _forms_of(stated[1:]) for a in asked):
                return False
    return True


def _lead_in_states(pattern: "re.Pattern[str]", unit: str, statement: str) -> bool:
    """The list lead-in of the statement's item states the predicate ("The University's share shall be DISTRIBUTED as
    follows: a. 40% to the VCRGS")."""
    if not unit or not statement.strip():
        return False
    position = unit.find(statement.strip()[:40])
    if position <= 0:
        return False
    before = unit[:position]
    m = re.search(r"([^.;]*?(?:as\s+follows|following|include|includes)?[^.;]*:)\s*(?:[a-z]\.|\d+\.|•|o)?\s*$", before, re.I)
    return bool(m and pattern.search(m.group(1)))


def _first_subject(text: str) -> Optional[str]:
    tokens = _split_tokens(_strip_heading(text))
    end = next((i for i, t in enumerate(tokens) if t in _AUX or t == "be" or t in (",", ";", ":") or (t in _PREPOSITIONS and t != "of" and t != "for")
                or (i and _is_verb_token(t, tokens[i - 1], tokens[i + 1] if i + 1 < len(tokens) else ""))), len(tokens))
    span = " ".join(t for t in tokens[:end] if t not in (",", ";", ":", "(", ")"))
    return re.sub(r"^\s*(?:\d+(?:\.\d+)*\.|[a-z]\.)\s*", "", span)


def _subject_bound(frame: "RelationFrame", text: str, m: Optional["re.Match[str]"], unit: str = "") -> bool:
    words, roles = frame.subject, frame.subject_roles
    if m is None:
        span = _first_subject(text)
        if frame.subject_generic:
            return not (_roles_in(span or "") & (_AUTHORITY_ROLES | _POPULATION_ROLES))
        if _party_in(words, roles, span):
            return True
        if roles and not words and all(_acts_as(text, r) for r in roles):
            return True                                            # "Faculty members may not ACT AS principal investigators": a capacity
        if frame.predicate in _LIGHT_PREDICATES or frame.predicate in _PROPOSITION_EXEMPT or not frame.yes_no:
            tokens = _split_tokens(text)
            phrases = [_phrase_after(tokens, i + 1) for i, t in enumerate(tokens) if t in ("for", "to") and not
                       (t == "to" and i + 1 < len(tokens) and _infinitive(tokens[i + 1], tokens[i + 2] if i + 2 < len(tokens) else ""))]
            if any(roles <= _roles_in(" ".join(p)) and all(any(_word_forms(a) & _word_forms(w) for a in p) for w in words) for p in phrases if p):
                return True                                        # "awards FOR coordinators of research groups": the receiver
            return bool(roles and all(_recipient_named(text, r.replace("_", " ")) for r in roles))
        return False
    voice = _occurrence_voice(text, m)
    if voice == "modifier":
        return False
    if frame.passive:
        if voice == "active":
            return _party_in(words, roles, " ".join(_object_parts(text[m.end():])[0]))
        patient = _passive_subject(_local_target(text, m)) if voice == "passive" else _local_target(text, m)
        if not patient:
            return False
        return _patient_bound(frame, patient, unit or text)
    if frame.subject_generic:
        if voice == "active":
            span = _subject_span(text, m.start())
            if span == "" and _only_as_agent(frame.subject_text, text):
                return False                                       # "Reviewing applications submitted BY FACULTY MEMBERS": not the subject
            return span is not None and not (_roles_in(span) & (_AUTHORITY_ROLES | _POPULATION_ROLES)) and _object_bound(frame, text, m)
        return True
    if voice == "active":
        span = _subject_span(text, m.start())
        if _party_in(words, roles, span):
            return _object_bound(frame, text, m)
        if re.match(r"\s*(?:to|from|by|for|with)\b", text[m.end():], re.I):
            return False                                           # "disclose TO the TTO": a recipient, not the object
        phrase, complement, _possessors = _object_parts(text[m.end():])
        obj = " ".join(phrase)
        return bool(obj) and _party_in(words, roles, obj) and bool(frame.asked_object) and \
            all(_word_forms(w) & _forms_of(complement | set(phrase)) for w in frame.asked_object if w not in _TOPIC_FILLERS)
    agent = _agent_after(text, m.end(), voice == "nominal")
    if agent is None and voice == "nominal":
        agent = _agent_before(text, m.start())
    if agent is None:
        return False
    if not (roles <= _roles_in(" ".join(agent)) and all(any(_word_forms(a) & _word_forms(w) for a in agent) for w in words)):
        return False
    if voice == "passive" and frame.asked_object:
        patient = _local_target(text, m)                           # "IP CREATED by inventors" is not "inventors establish PROCEDURES"
        phrase, complement, possessors = _object_parts(_passive_subject(patient) if text[:m.start()].endswith(patient) else patient)
        return bool(_word_forms(frame.asked_object[-1]) & _forms_of(set(phrase) | complement | possessors))
    return True


def _acts_as(text: str, role: str) -> bool:
    """The statement's subject holds the role as a capacity ("act as principal investigators", "serving as coordinator")."""
    pattern = dict(_COMPILED_ROLES).get(role)
    if pattern is None:
        return False
    for m in re.finditer(r"\b(?:act|acts|acting|serve|serves|serving|capacity)\s+as\s+(?:an?\s+|the\s+)?", text, re.I):
        following = pattern.match(text[m.end():])
        if following:
            return True
    return False


def _only_as_agent(subject_text: str, text: str) -> bool:
    """The asked party's words occur in the statement, but only inside a by / to / from phrase."""
    words = [w for w in _content(_TOKEN.findall(subject_text.lower())) if w not in _TOPIC_FILLERS and len(w) > 2]
    if not words or not all(_has_word(text, w) for w in words):
        return False
    tokens = _split_tokens(text)
    outside: List[str] = []
    inside = False
    for t in tokens:
        if t in ("by", "to", "from"):
            inside = True
        elif t in (",", ";", ":", ".", "(", ")") or (t in _PREPOSITIONS and t != "of") or t in _AUX or t == "be":
            inside = False
        if not inside:
            outside.append(_stem(t))
    return not any(_word_forms(w) & _forms_of(outside) for w in words)


def _patient_bound(frame: "RelationFrame", patient: str, unit: str) -> bool:
    """The asked passive subject heads the stated one; a stated restriction the question lacks is a contrast
    ("UNCOMPENSATED conference travel" never answers for "compensated conference travel"); an asked modifier is stated in
    the unit ("the advance" of the petty cash item)."""
    phrase, complement, possessors = _object_parts(patient)
    if frame.subject_roles and not frame.subject_roles <= _roles_in(patient):
        return False
    np = frame.subject_np
    if not np:
        return True
    head = np[-1]
    position = next((i for i, w in enumerate(phrase) if _word_forms(w) & _word_forms(head)), None)
    if position is None:
        return False
    stated = [w for w in list(phrase[:position]) + list(possessors) if w not in _NON_CONTRAST and not w.isdigit()]
    if any(not (_word_forms(w) & _forms_of(np)) for w in stated):
        return False
    return all(_has_word(unit, w) for w in np[:-1] if w not in _TOPIC_FILLERS and not _anchor_word(w))


def _object_bound(frame: "RelationFrame", text: str, m: "re.Match[str]") -> bool:
    """The asked object of the predicate ("offer VISITING RESEARCHERS opportunities", "have their RESEARCH load reduced")
    is the stated one: its head is in the stated phrase and the stated phrase restricts the head with no word the
    question lacks."""
    if not frame.asked_object or any(a for a, _h in frame.actions if _ACTION_PATTERNS[a].fullmatch(frame.predicate or "")):
        return True
    phrase, complement, possessors = _object_parts(text[m.end():])
    if not phrase:
        return True
    head = frame.asked_object[-1]
    position = next((i for i, w in enumerate(phrase) if _word_forms(w) & _word_forms(head)), None)
    if position is None:
        return _has_word(" ".join(phrase) + " " + " ".join(complement), head)
    stated = [w for w in list(phrase[:position]) + list(possessors) if w not in _NON_CONTRAST and not w.isdigit()]
    return not any(not (_word_forms(w) & _forms_of(frame.asked_object)) for w in stated)


def _agents_bound(agents: Sequence[Tuple[str, Tuple[str, ...]]], text: str) -> bool:
    """Each participle the question gives an agent has that agent in the statement ("sponsored by external
    organizations" never answers "sponsored by the University")."""
    tokens = _split_tokens(text)
    for participle, words in agents:
        pattern = _predicate_pattern(participle)
        stated: List[Tuple[str, ...]] = []
        for m in pattern.finditer(text):
            agent = _agent_after(text, m.end())
            if agent is not None:
                stated.append(agent)
        if not stated:
            stated = [_phrase_after(tokens, i + 1) for i, t in enumerate(tokens) if t == "by"]
        if not stated or not any(all(any(_word_forms(a) & _word_forms(w) for a in agent) for w in words) for agent in stated):
            return False
    return True


def _arguments_bound(arguments: Sequence[Tuple[str, Tuple[str, ...]]], text: str, strict: bool = True) -> bool:
    """Each prepositional argument of a yes/no ask is stated with the same preposition ("to the IACUC" is not "to the
    IRB", "for students" is not "for graduate research assistants"); when the statement has no phrase with that
    preposition, or for a value ask, the words may be stated without it."""
    tokens = _split_tokens(text)
    for prep, words in arguments:
        strict_here = strict or prep == "without" or bool(_roles_in(" ".join(words)))   # a party, a negation: always bound
        words = tuple(w for w in words if w not in _TIME_WORDS)      # "at the SAME TIME": a time adverbial, not a party
        if not strict_here:
            words = tuple(w for w in words if w not in _TOPIC_FILLERS and not _anchor_word(w) and
                          not any(p.fullmatch(w) for _a, p in _COMPILED_ACTIONS))
        if not words:
            continue
        stated = [_phrase_after(tokens, i + 1) for i, t in enumerate(tokens) if t == prep and not
                  (prep == "to" and i + 1 < len(tokens) and _infinitive(tokens[i + 1], tokens[i + 2] if i + 2 < len(tokens) else ""))]
        stated = [p for p in stated if p]
        if any(all(any(_word_forms(a) & _word_forms(w) for a in phrase) for w in words) for phrase in stated):
            continue
        anywhere = all(_has_word(text, w) for w in words)
        if not strict_here and anywhere:
            continue                                               # a value ask: the words suffice
        if prep in ("with", "on") and anywhere:
            continue
        if prep in ("for", "at") and not stated and anywhere:
            continue                                               # "Even in cases of non-compensated consultancy, ..."
        return False
    return True


_CONDITIONAL = re.compile(r"(?:^|[,;]\s*(?:and|or|but)?\s*)(?:if|when|whenever|where|unless|provided\s+that|in\s+the\s+event(?:\s+that)?|in\s+case|"
                          r"in\s+cases\s+where)\s+([^,;:]{0,90}),", re.I)


def _conditioned_role_ok(roles: FrozenSet[str], statement: str) -> bool:
    """A rule conditioned on another role ("in the event the DEAN is planning ..., a written approval from the VCAA is
    required") does not answer a question that does not name that role."""
    text = cue_text(statement)
    for m in _CONDITIONAL.finditer(text):
        if _roles_in(m.group(1)) - roles:
            return False
    return True


def _roles_stated(roles: FrozenSet[str], statement: str, unit: str) -> bool:
    """Every asked role is named in the statement, or the statement refers to it with a pronoun ("they", "their") and the
    unit names it."""
    stated = _roles_in(statement)
    missing = roles - stated
    if not missing:
        return True
    return bool(re.search(r"\b(?:they|their|them|he|she|his|her|it|its)\b", cue_text(statement), re.I)) and missing <= _roles_in(unit)


def _recipients_stated(recipients: Sequence[str], statement: str) -> bool:
    """A share's receiver is named as the receiver in the statement that states the share ("5% to the Inventors' COLLEGE"
    never names the inventor as the receiver)."""
    text = cue_text(statement)
    return all(_recipient_named(text, w) for w in recipients)


def _competitors_ok(competitors: Sequence[Tuple[str, str]], unit: str) -> bool:
    text = cue_text(unit)
    return all(re.search(asked, text, re.I) or not re.search(rival, text, re.I) for asked, rival in competitors)


_SCOPE_WORDS = frozenset({"total", "gross", "net", "overall", "entire", "whole", "all", "full"})


def _share_statement_ok(owner: Optional[str], statement: str, unit: str, base: Sequence[str] = ()) -> bool:
    """A percentage of an owner's share is stated with that share, or by a list item under it (checked with its lead-in);
    a percentage of another base ("of TOTAL LICENSING REVENUE") is never an item of an owner's share."""
    statements = _STATEMENT_SPLIT.split(unit.strip())
    position = next((i for i, st in enumerate(statements) if statement.strip() and statement.strip() in st), len(statements))
    named = {_stem(m.group(1)) for m in _SHARE_OWNER.finditer(cue_text(statement))}
    lead_owners: FrozenSet[str] = frozenset()
    for earlier in reversed(statements[:position]):
        owners = {_stem(m.group(1)) for m in _SHARE_OWNER.finditer(cue_text(earlier))}
        if owners:
            lead_owners = frozenset(owners) if re.search(r"as\s+follows|:\s*$", earlier, re.I) else frozenset()
            break
    if owner is None:
        if base and (named or lead_owners) and set(base) & _SCOPE_WORDS and not any(_has_word(" ".join(base), o) for o in named | lead_owners):
            return False                                           # "40% to the VCRGS" is of the University's share, not of TOTAL revenue
        return True
    if named:
        return owner in named
    if len(statements) == 1:
        return True                                                # a single item: its lead-in names the share (see bind)
    return owner in lead_owners


def _relation_strength(shape: Optional[str], grounding: Sequence[Detail], covered: FrozenSet[str]) -> Optional[str]:
    """STRONG when the relation detail of a recognised ask is covered by one unit; AMBIGUOUS otherwise; None for general,
    unknown and process asks (their own rules decide)."""
    if shape != ASK_RECOGNISED:
        return None
    return RELATION_STRONG if grounding and all(_details._detail_key(d) in covered for d in grounding) else RELATION_AMBIGUOUS


def _relation_downgrade(relation: Optional[str]) -> bool:
    """The conservative contract: an ambiguous relation is never SUPPORTED."""
    return relation == RELATION_AMBIGUOUS


#: An elliptical clause ("... and my conference travel request", "what about my final report?") carries the ask of
#: the question head; each template restates it for the clause's own object.
_ELLIPSIS_LEAD = re.compile(r"^\s*(?:(?:and|also|plus|as\s+well\s+as|or)\s+)?(?:(?:what|how)\s+about\s+)?", re.I)
_ELLIPTICAL = re.compile(r"^\s*(?:(?:and|also|plus|as\s+well\s+as|or)\s+)?(?:what|how)\s+about\b", re.I)
_INHERIT_TEMPLATE = {
    "deadline": "What is the deadline for {np}?", "advance": "How far in advance must {np} be submitted?", "amount": "How much is {np}?",
    "percentage": "What percentage is {np}?", "maximum": "What is the maximum for {np}?", "minimum": "What is the minimum for {np}?",
    "authority": "Who approves {np}?", "responsibility": "Who is responsible for {np}?", "eligibility": "Who is eligible for {np}?",
    "ownership": "Who owns {np}?", "documents": "What documents are required for {np}?", "duration": "How long is {np}?",
    "frequency": "How often is {np}?", "quantity": "How many {np} are there?", "penalty": "What is the penalty for {np}?",
}


def _question_head(clause: str, question: Optional[str]) -> Optional[str]:
    if not question:
        return None
    key = cue_text(clause).strip(" ?.!,").lower()[:60]
    pos = cue_text(question).lower().find(key) if key else -1
    if pos <= 0:
        return None
    head = re.sub(r"[\s,;:]*(?:(?:and|or|also|plus|as\s+well\s+as|then)\s*)*$", "", question[:pos], flags=re.I).strip()
    return head or None


def _inherited_ask(segment: str, clause: str, question: Optional[str]) -> Tuple[Optional[str], bool]:
    """(the elliptical segment restated with the question head's ask, whether that head asks something specific)."""
    head = _question_head(clause, question)
    if head is None:
        return None, False
    pieces = [p.strip() for p in re.split(r"[,;]|\band\b|\bor\b|\bas\s+well\s+as\b", head, flags=re.I) if p.strip()]
    asking = [p for p in pieces if _WH_WORD.search(p) or _AUX_START.match(_ask_text(p))]
    if not asking:
        return None, False                                          # no interrogative before it: a bare topic phrase
    if len(asking) > 1:
        return None, True                                           # which question does the phrase continue? ambiguous
    head = asking[0]
    core = re.match(r"^(?P<core>.*\b(?:for|of|about|on|to|submit|send|file|provide|approves?|reviews?|nominates?|signs?|owns?|is|are|was|were|"
                    r"must|should|can|will)\b)\s+(?:the\s+|my\s+|our\s+|a\s+|an\s+|their\s+|your\s+|his\s+|her\s+)?[\w\s'’-]+$", cue_text(head), re.I)
    if _ask_residue(core.group("core") if core else head, focus_terms(head)):
        return None, True                                           # the head asks for its own specifics ("the REFUND for ...")
    head_kind = _ask_kind(head)
    families = [d.family for d in requested_details(head) if d.family in _INHERIT_TEMPLATE]
    specific = head_kind not in _GENERAL_KINDS or bool(families)
    np = _ELLIPSIS_LEAD.sub("", cue_text(segment)).strip(" ?.!,")
    if not families or not np:
        return None, specific
    family = families[0]
    template = _INHERIT_TEMPLATE[family]
    verb = re.match(r"^\s*who\s+(\w+)\b", _ask_text(head), re.I)
    if family in ("authority", "responsibility") and verb:
        template = "Who %s {np}?" % verb.group(1)                  # "Who REVIEWS my course portfolio and my application?"
    if family == "deadline" and not (_EXPLICIT_DEADLINE.search(head) or re.search(r"\bby\s+when\b", head, re.I)):
        template = "When must {np} be submitted?"
    return template.format(np=np), specific


@dataclass(frozen=True)
class AskAssessment:
    """How a clause asks: its shape, the interrogative kind of each substantive segment, the words of the ask the
    specialist cannot map, the requested details to apply, and the restated ask of an elliptical clause."""

    shape: str
    kinds: Tuple[str, ...]
    residue: Tuple[str, ...]
    details: Tuple[Detail, ...]
    inherited: Optional[str] = None


def assess_ask(clause: str, details: Sequence[Detail], question: Optional[str] = None) -> AskAssessment:
    """Classify the clause's ask (general, specific recognised, specific unknown) and return the requested details
    that make completeness fail safe: an unrecognised specific ask adds a detail no evidence covers; a recognised one
    adds the grounding of its unmapped words, subject, approval act and purpose in the answering sentence."""
    details = list(details)
    segments = [s for s in clause_segments(clause) if not _navigation_segment(s)] or [clause]
    kinds: List[str] = []
    shapes: List[str] = []
    residue_texts: List[str] = []
    inherited: Optional[str] = None
    subject: Optional["re.Pattern[str]"] = None
    action: Optional["re.Pattern[str]"] = None
    heads: List[str] = []
    for segment in segments:
        kind = _ask_kind(segment)
        if kind == "none" or _ELLIPTICAL.match(cue_text(segment)):
            restated, specific_head = _inherited_ask(segment, clause, question)
            if restated is not None:
                inherited = restated
                carried = requested_details(restated)
                present = {d.family for d in details}
                details += [d for d in carried if d.family in _ASK_FAMILIES and d.family not in present]
                kinds.append(_ask_kind(restated))
                shapes.append(ASK_RECOGNISED if {d.family for d in carried} & _ASK_FAMILIES else ASK_UNKNOWN)
                residue_texts.append(_ELLIPSIS_LEAD.sub("", cue_text(segment)))
                continue
            if specific_head:                                       # a bare phrase of a specific question whose ask cannot be carried
                kinds.append("elliptical")
                shapes.append(ASK_UNKNOWN)
                residue_texts.append(segment)
                continue
        own = requested_details(segment)
        families = {d.family for d in own} & _ASK_FAMILIES
        focus = focus_terms(segment)
        required = _ASK_REQUIRES.get(kind)
        kinds.append(kind)
        residue_texts.append(segment)
        subject = subject or _subject_pattern(segment)
        action = action or _action_pattern(segment, own)
        if families & {"authority", "responsibility", "ownership"}:
            head = _object_head(segment)
            if head and head not in heads:
                heads.append(head)
        if families and (required is None or families & required):
            shapes.append(ASK_RECOGNISED)
            # invariant: the family that makes the ask recognised is a requested detail of the clause (the segment reads the
            # hyphen-normalised text, so "editor-in-chief" is recognised here even when the raw clause matched nothing)
            present = {d.family for d in details}
            details += [d for d in own if d.family in families and d.family not in present]
        elif kind == "yes_no" and (families or focus or _EXISTENCE_FORM.match(_ask_text(segment))):
            shapes.append(ASK_RECOGNISED)
        elif kind in ("what", "which") and not families and _selection_form(segment, focus):
            shapes.append(ASK_RECOGNISED)
        elif kind in _GENERAL_KINDS and not families and _general_form(segment, focus):
            shapes.append(ASK_GENERAL)
        elif kind == "none" and not families and _topic_only(_ask_text(segment), focus):
            shapes.append(ASK_GENERAL)                             # a bare topic phrase ("Research grants.")
        else:
            shapes.append(ASK_UNKNOWN)
    focus_all = focus_terms(clause)
    relational = "yes_no" in kinds or bool({d.family for d in details} & _RELATIONAL_FAMILIES)
    residue: List[str] = []
    for text in residue_texts:
        for word in _ask_residue(text, focus_all, relational):
            if word not in residue:
                residue.append(word)
    if ASK_UNKNOWN in shapes:
        shape = ASK_UNKNOWN
        unknown_kinds = [k for k, sh in zip(kinds, shapes) if sh == ASK_UNKNOWN]
        details.append(Detail("ask_unknown", "the specific ask of the question (%s)" % ", ".join(unknown_kinds), _ask_unknown_check))
    elif ASK_RECOGNISED in shapes:
        shape = ASK_RECOGNISED
        purpose_match = _PURPOSE.search(cue_text(clause))
        grounded = list(residue) + [h for h in heads if h not in residue]
        clause_words = frozenset(f for t in _TOKEN.findall(cue_text(clause).lower()) for f in _word_forms(t))
        frame = _relation_frame(inherited or clause, details, "yes_no" in kinds)
        grounding = _grounding_detail(details, grounded, subject, action, purpose_match.group(1).lower() if purpose_match else None,
                                      clause_words, frame)
        if grounding is not None:
            details.append(grounding)
    else:
        shape = ASK_GENERAL
    return AskAssessment(shape, tuple(kinds), tuple(residue), tuple(details), inherited)


# Navigation (Step 4.5A-R): contracted "where's", opening and closing times, directions; a contact half phrased as a
# request ("..., and give me its email") is its own segment.
_NAVIGATION_CUES = re.compile(_NAVIGATION_CUES.pattern +
                              r"|\bwhere(?:['’]s|['’]re|\s+s)\b|\bwhat\s+time\s+(?:does|do|is|are)\b[^?]{0,50}\b(?:open|close)\w*|"
                              r"\b(?:opening|closing|office|working)\s+hours\s+of\b|\bdirections?\s+to\b|\bget\s+hold\s+of\b|"
                              r"\b(?:give|send|tell)\s+me\s+(?:its|their|the)\s+(?:(?:tele)?phone|e\s?mail|email|fax|extension|contact|address|location)\b|"
                              r"\bwho\s+(?:do|should|can|must|would)\s+(?:i|we)\s+(?:submit|send|give|hand|address|report|forward)\b[^?]{0,60}\bto\s*[?.!]?\s*$|"
                              r"^\s*where\s+(?:exactly|precisely|specifically)\s+(?:is|are)\b|\bextension\s+(?:number\s+)?(?:of|for)\b|"
                              r"(?:what(?:['’]s|\s+is)\s+(?:the\s+)?)(?:(?:tele)?phone(?:\s+number)?|fax|extension|e\s?mail(?:\s+address)?)\b|"
                              r"\b(?:e\s?mail|email|phone(?:\s+number)?|telephone|fax|extension)\s*[?.!]?\s*$|"
                              r"^\s*(?:(?:and|also)\s+)?(?:the\s+|its\s+|their\s+)?(?:(?:tele)?phone(?:\s+number)?|e\s?mail(?:\s+address)?|fax|extension)"
                              r"\s+(?:of|for)\b", re.I)
_SEGMENT_SPLIT = re.compile(_SEGMENT_SPLIT.pattern + r"|\s*,?\s+(?:and\s+|then\s+)+(?=(?:give|send|tell|show)\s+me\b)", re.I)


@dataclass(frozen=True)
class ResearchResources:
    """What the specialist needs to retrieve evidence. Built from a
    ``KnowledgeBase`` in production and from fakes in tests."""

    embed: Callable[[str], np.ndarray]                       # text -> (1, dim) float32
    rerank: Callable[[Sequence[Sequence[str]]], Sequence[float]]   # [[query, text], ...] -> scores
    index: Any
    chunks: List[str]
    metadata: List[Dict[str, Any]]
    source: SourceRecord
    section_map: SectionMap

    @classmethod
    def from_knowledge_base(cls, kb) -> "ResearchResources":
        source = getattr(kb, "source", None)
        if source is None or not source.registered or source.source_id != APPROVED_SOURCE_ID:
            raise ValueError("the Research specialist requires the registered handbook %r" % APPROVED_SOURCE_ID)
        applied = ((getattr(kb, "stats", None) or {}).get("source") or {}).get("section_records", 0)
        if not applied:
            raise ValueError("the section map was not applied to the loaded document; research scope cannot be compiled")
        embedder, reranker = kb.embedder, kb.reranker

        def embed(text: str) -> np.ndarray:
            return embedder.encode([normalize_text(text)], convert_to_numpy=True,
                                   normalize_embeddings=True).astype(np.float32)

        return cls(embed=embed, rerank=reranker.predict, index=kb.index, chunks=list(kb.chunks),
                   metadata=list(kb.metadata), source=source, section_map=section_map_for(source))


# ---------------------------------------------------------------------------
# The specialist
# ---------------------------------------------------------------------------
_safe_error = redact_error


def _quote_key(quote: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", quote.lower()).strip()


@dataclass
class _Candidate:
    rank: int
    item: Dict[str, Any]
    quote: str
    anchors: int
    vocab: int
    covered: FrozenSet[str]
    list_complete: Optional[bool] = None


#: A required-documents list is quoted whole when it fits this many words.
MAX_LIST_WORDS = 120


def whole_item(text: str, quote: str, allowed: Optional[Sequence[Tuple[int, int]]] = None) -> str:
    """A quote that opens a numbered or lettered item, extended over the rest of
    that item ("15. The principal investigator must hold a Ph.D. ...
    Additionally, they must be full-time employees ... and not visitors."). A
    list item is ONE evidence unit (the shared locality rule), so its later
    sentences belong with its opening; a sibling item is never added, and the
    extract stays within the allowed spans and ``MAX_QUOTE_WORDS``."""
    start = text.find(quote)
    if start < 0:
        return quote
    end = start + len(quote)
    units = _units(text, "paragraph")
    opening = next((item for s, e, item in units if s == start and item == s and _ITEM_START.match(text[s:e])), None)
    if opening is None:
        return quote
    new_end = end
    for s, e, item in units:
        if s < end:
            continue
        if item != opening or item == s:                             # the next item, or text after the list
            break
        if (allowed is not None and not any(a <= s and e <= b for a, b in allowed)) or len(text[start:e].split()) > MAX_QUOTE_WORDS:
            return quote
        new_end = e
    return text[start:new_end].strip()


def complete_list(text: str, quote: str, allowed: Optional[Sequence[Tuple[int, int]]] = None) -> Tuple[str, Optional[bool]]:
    """For a quote that holds items of a list (a required-documents list): the
    quote extended over the remaining items of the same list, and whether the
    whole list is now quoted. ``None`` when the quote holds no list item. The
    list is complete only when its end is visible inside the chunk (a plain
    sentence or a heading follows), every further item is the specialist's to
    quote, and the extract stays within ``MAX_LIST_WORDS``; otherwise the
    original quote is kept and the list is reported incomplete. Nothing is
    ever added that the chunk does not print."""
    start = text.find(quote)
    if start < 0:
        return quote, None
    end = start + len(quote)
    units = _units(text, "paragraph")
    if not any(s >= start and e <= end and item == s and _ITEM_START.match(text[s:e]) for s, e, item in units):
        return quote, None

    def permitted(s: int, e: int) -> bool:
        return allowed is None or any(a <= s and e <= b for a, b in allowed)

    # backwards: earlier items of the same list, up to and including the lead-in that introduces it
    new_start = start
    for s, e, item in reversed([u for u in units if u[1] <= start]):
        piece = text[s:e].strip()
        if (item == s and _ITEM_START.match(piece)) or item < s:
            if not permitted(s, e):
                return quote, False
            new_start = s
            continue
        if piece.endswith(":") and permitted(s, e):
            new_start = s                                            # the lead-in ("... including:")
        break
    # forwards: the remaining items, until a plain sentence or a heading shows where the list ends
    new_end = end
    for s, e, item in units:
        if s < end:
            continue
        opens_item = item == s and bool(_ITEM_START.match(text[s:e]))
        if not (opens_item or item < s):
            extract = text[new_start:new_end].strip()
            return (extract, True) if len(extract.split()) <= MAX_LIST_WORDS else (quote, False)
        if not permitted(s, e):
            return quote, False
        new_end = e
    return quote, False                                             # the list runs to the end of the chunk


class ResearchInnovationSpecialist(Specialist):
    """Research & Innovation Specialist, deterministic version 1."""

    specialist_id = SpecialistId.RESEARCH

    def __init__(self, resources: ResearchResources, *,
                 owned_sections: Sequence[Tuple[int, str, str]] = SCOPE_SECTIONS,
                 extra_pages: Sequence[Tuple[int, int]] = EXTRA_PAGES,
                 approved_sources: Optional[Sequence[str]] = None) -> None:
        if resources.source.source_id != APPROVED_SOURCE_ID or not resources.source.registered:
            raise ValueError("the Research specialist may only use the registered handbook %r" % APPROVED_SOURCE_ID)
        self.resources = resources
        self.owned_sections = tuple(owned_sections)
        self.extra_pages = tuple(extra_pages)
        # post-freeze PF-3B (D-110): the source-acceptance boundary. None keeps the I-9 handbook-only boundary; the runtime's
        # source policy may approve named official guides, admitted beside the unchanged handbook scope
        self.approved_sources = _scoping.approved_source_set(approved_sources, APPROVED_SOURCE_ID)
        self.scope = _scoping.approve_sources(compile_scope(resources.section_map, APPROVED_SOURCE_ID, self.owned_sections, self.extra_pages),
                                              self.approved_sources, APPROVED_SOURCE_ID)
        self.guard = _scoping.SectionGuard(resources.chunks, resources.metadata, resources.section_map, self.owned_sections, self.scope)

    @classmethod
    def from_knowledge_base(cls, kb, *, approved_sources: Optional[Sequence[str]] = None) -> "ResearchInnovationSpecialist":
        return cls(ResearchResources.from_knowledge_base(kb), approved_sources=approved_sources)

    # ---- input checks -------------------------------------------------------
    def _check_task(self, task: SpecialistTask) -> None:
        if not isinstance(task, SpecialistTask):
            raise TypeError("run() expects a SpecialistTask, got %s" % type(task).__name__)
        if task.specialist_id is not None and task.specialist_id != self.specialist_id:
            raise ValueError("task %s is assigned to %s, not to %s"
                             % (task.task_id, task.specialist_id.value, self.specialist_id.value))

    @staticmethod
    def _source_scope_problem(task: SpecialistTask) -> Optional[str]:
        unapproved = [s for s in task.source_scope if s != APPROVED_SOURCE_ID]
        if unapproved:
            return "task source_scope names sources the Research specialist is not approved to use: %s" % ", ".join(unapproved)
        return None

    @staticmethod
    def _focus_clauses(task: SpecialistTask) -> List[str]:
        raw = task.context.get("focus_clauses") if isinstance(task.context, dict) else None
        clauses = [c.strip() for c in raw if isinstance(c, str) and c.strip()] if isinstance(raw, (list, tuple)) else []
        return clauses or [task.question.strip()]

    @staticmethod
    def _route(task: SpecialistTask) -> str:
        return task.intent if task.intent in ALL_ROUTES and task.intent != "greeting" else "policy"

    # ---- evidence -----------------------------------------------------------
    def _evidence(self, clause: str, route: str) -> List[Dict[str, Any]]:
        """Scoped retrieval, rerank, dedupe and gate for one clause, in the
        production order. Every kept item must reach ``MIN_RERANK_SCORE``."""
        embedding = np.asarray(self.resources.embed(clause), dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        candidates = gather_candidates(clause, route, embedding, self.resources.index,
                                       self.resources.chunks, self.resources.metadata, scope=self.scope)
        if not candidates:
            return []
        scores = list(self.resources.rerank([[clause, c["chunk"]] for c in candidates]))
        if len(scores) != len(candidates):
            # a short score list would silently drop evidence and read as "not found"
            raise ValueError("reranker returned %d scores for %d candidates" % (len(scores), len(candidates)))
        items: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            items.append(item)
        items.sort(key=lambda x: (x["rerank_score"], x.get("routing_boost", 0.0),
                                  x.get("lexical_score", 0.0), x.get("dense_score", 0.0)), reverse=True)
        items = deduplicate_by_text(items)
        return [item for item in items if item["rerank_score"] >= MIN_RERANK_SCORE]

    @staticmethod
    def _shape_problem(item: Dict[str, Any]) -> Optional[str]:
        """Printed prose lines are not evidence (the paragraph chunk carries the same text)."""
        meta, chunk = item["meta"], item["chunk"]
        kind = meta.get("chunk_type")
        if kind in ("row", "row_window") and not any(is_table_row(chunk[s:e].strip()) for s, e in _spans(chunk, kind)):
            return "printed prose line, not an evidence unit"
        return None

    @staticmethod
    def _about_request(quote: str, anchors: int, system: Optional[str]) -> bool:
        """For a system procedure request: the evidence must name the system
        asked about or share an anchor concept with the request."""
        if anchors >= 1:
            return True
        named = system_named(quote)
        return named is not None and (system is None or named == system)

    def _boundary_label(self, meta: Dict[str, Any], shared: bool) -> str:
        if shared:
            return "attributed to an owned section"
        nos = list(meta.get("page_section_nos") or [])
        if not any("." in str(n) for n in nos):
            return "chapter preamble: page carries no section record"
        return "page carries owned sections only"

    def _finding(self, task: SpecialistTask, number: int, cand: _Candidate, clause_index: int, route: str,
                 details: Sequence[Detail]) -> Finding:
        item, quote = cand.item, cand.quote
        meta = item["meta"]
        chunk = item["chunk"]
        if quote not in chunk:                                   # cannot happen by construction; kept as a guard
            raise ValueError("quote is not a substring of the retrieved chunk")
        if meta.get("source_id") not in self.approved_sources or not self.scope.allows(meta):   # D-110: approved-source check
            raise ValueError("retrieved chunk %r is outside the approved scope" % meta.get("chunk_id"))
        page = meta.get("page")
        shared = self.guard.is_shared(meta)
        metadata: Dict[str, Any] = {
            "chunk_id": meta.get("chunk_id"),
            "chunk_type": meta.get("chunk_type"),
            "chapter": meta.get("chapter"),
            "section_no": meta.get("section_no"),
            "section_title": meta.get("section_title"),
            "section_label_page_level": True,
            "page_section_nos": list(meta.get("page_section_nos") or []),
            "shared_page": shared,
            "boundary_guard": self._boundary_label(meta, shared),
            "clauses": [clause_index],
            "route": route,
            "rerank_score": item["rerank_score"],
            "score_semantics": "cross-encoder rerank score; heuristic, uncalibrated, not a probability",
            "relevance": {"anchors": cand.anchors, "vocabulary": cand.vocab},
            "details_covered": sorted(d.label for d in details if _details._detail_key(d) in cand.covered),
            "extractive": True,
        }
        return Finding(
            finding_id="%s-f%d" % (task.task_id, number),
            claim=quote,
            evidence_quote=quote,
            source_id=meta["source_id"],
            source_title=str(meta.get("source_title") or self.resources.source.title),
            page=page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None,
            metadata=metadata,
        )

    # ---- results ------------------------------------------------------------
    def _result(self, task: SpecialistTask, status: FindingStatus, *, findings=(), missing=(), handoffs=(),
                limitations=(), summary: str, metadata: Dict[str, Any], started: float) -> SpecialistFindings:
        confidence = {FindingStatus.SUPPORTED: 1.0, FindingStatus.PARTIAL: 0.5}.get(status, 0.0)
        metadata = dict(metadata)
        metadata["confidence_semantics"] = "ordinal by status: 1.0 supported, 0.5 partial, 0.0 otherwise; not a probability"
        metadata["scope"] = self.scope.describe()
        metadata["boundary"] = {"shared_pages": sorted(self.guard.shared_pages), "unresolved_pages": list(self.guard.unresolved_pages),
                                "extra_pages": [list(r) for r in self.extra_pages]}
        return SpecialistFindings(
            specialist_id=self.specialist_id, task_id=task.task_id, status=status, findings=list(findings),
            summary=summary, missing=list(missing), handoff_requests=list(handoffs), confidence=confidence,
            limitations=list(limitations), llm_used=False, ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def _handoff(self, task: SpecialistTask, target: SpecialistId, clause: str, index: int, kind: str, ordinal: int = 0) -> HandoffRequest:
        # one clause may carry several foreign intents: the first handoff keeps the historical id, the others are numbered
        suffix = "%d" % index if ordinal == 0 else "%d-%d" % (index, ordinal)
        return HandoffRequest(
            from_specialist=self.specialist_id, requested_specialist=target,
            reason="%s clause %d belongs to %s: %s" % (kind, index, target.value, clause[:120]),
            task=SpecialistTask(
                task_id="%s-handoff-%s" % (task.task_id, suffix), question=task.question, specialist_id=target,
                domain=target.value, intent=task.intent, level=task.level, system=task.system,
                context={"full_question": task.question, "focus_clauses": [clause]},
                requested_by=self.specialist_id.value,
                metadata={"origin_task_id": task.task_id, "clause_index": index, "kind": kind},
            ),
            metadata={"clause_index": index, "kind": kind},
        )

    # ---- one clause -----------------------------------------------------------
    def _candidates(self, clause: str, route: str, clause_concepts: Concepts, details: Sequence[Detail],
                    procedural: bool, system: Optional[str], rejected: List[Dict[str, Any]],
                    locality: Optional[str] = None) -> List[_Candidate]:
        found: List[_Candidate] = []
        wants_documents = any(d.family == "documents" for d in details)
        for rank, item in enumerate(self._evidence(clause, route)):
            if len(found) >= MAX_CANDIDATES_PER_CLAUSE:
                break
            meta = item["meta"]
            shape = self._shape_problem(item)
            if shape:
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": shape})
                continue
            allowed = self.guard.owned_spans(meta.get("chunk_id", item.get("idx")), item["chunk"], meta)
            if allowed is not None and not allowed:
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": "non-owned section on a shared page"})
                continue
            selected = select_quote(item["chunk"], meta.get("chunk_type"), clause_concepts, allowed, details)
            if selected is None:
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": "no relevant quotable unit"})
                continue
            quote, anchors, vocab = selected
            if procedural and not self._about_request(quote, anchors, system):
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": "tangential to the system procedure asked about"})
                continue
            if locality_conflict(locality, quote):
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": "names a different conference locality"})
                continue
            list_complete: Optional[bool] = None
            if meta.get("chunk_type") == "paragraph":
                quote = whole_item(item["chunk"], quote, allowed)
            if wants_documents and meta.get("chunk_type") == "paragraph":
                quote, list_complete = complete_list(item["chunk"], quote, allowed)
            covered = unit_coverage(details, clause_concepts, quote)
            if list_complete is False:
                covered = covered - {"documents"}                   # one item of a longer list never answers "which documents"
            found.append(_Candidate(rank, item, quote, anchors, vocab, covered, list_complete))
        found.sort(key=lambda c: (-len(c.covered), c.rank))
        return found

    # ---- public API -----------------------------------------------------------
    def run(self, task: SpecialistTask) -> SpecialistFindings:
        started = time.perf_counter()
        self._check_task(task)
        problem = self._source_scope_problem(task)
        if problem:
            return self._result(task, FindingStatus.ERROR, summary="refused: unapproved source scope",
                                metadata={"error": problem}, started=started)

        clauses = self._focus_clauses(task)
        processed, unprocessed = clauses[:MAX_CLAUSES], clauses[MAX_CLAUSES:]
        route = self._route(task)
        findings: List[Finding] = []
        handoffs: List[HandoffRequest] = []
        missing: List[str] = []
        limitations: List[str] = []
        reports: List[Dict[str, Any]] = []
        owned_clauses = 0
        systems_seen: List[str] = []

        try:
            for index, clause in enumerate(processed):
                research, _strongest = classify_clause(clause)
                others = other_domains(clause)                      # every foreign intent of the clause, strongest first
                navigation, substantive = navigation_intent(clause, task.system)
                if navigation and not substantive:
                    # "where is the HR department": a pure navigation clause is Institutional's alone and is never
                    # answered from research text, whatever unit it names
                    others, research = [SpecialistId.INSTITUTIONAL], False
                elif navigation:
                    # the contact or location half is Institutional's; the substantive half keeps its other domains
                    others = [SpecialistId.INSTITUTIONAL] + [o for o in others if o is not SpecialistId.INSTITUTIONAL]
                other = others[0] if others else None
                if not research:
                    if others:
                        handoffs.extend(self._handoff(task, o, clause, index, "misrouted", n) for n, o in enumerate(others))
                        reports.append({"index": index, "clause": clause, "outcome": "handed_off", "target": other.value,
                                        "targets": [o.value for o in others]})
                    else:
                        reports.append({"index": index, "clause": clause, "outcome": "unowned", "target": None})
                        missing.append("clause %d carries no research cue and was not searched: %s" % (index, clause[:120]))
                    continue
                owned_clauses += 1
                # the retrieval route of THIS clause (Step 4.7, D-078): the task intent is classified from the joined focus
                # clauses, so a contact clause beside a substantive one must not make it retrieve as a contact lookup
                clause_route = resolve_clause_retrieval_intent(clause, task.intent, sole_clause=len(processed) == 1,
                                                               substantive=substantive)
                handoffs.extend(self._handoff(task, o, clause, index, "shared", n) for n, o in enumerate(others))
                procedural, system = is_procedure_request(clause, task.system)
                process = not procedural and is_process_request(clause)
                if system and system not in systems_seen:
                    systems_seen.append(system)
                clause_concepts = concepts(clause)
                details = requested_details(clause)
                if navigation:
                    # the substantive half is answered; the contact or location half needs Institutional evidence
                    details = list(details) + [Detail("contact", "contact or location details", _contact_check(clause))]
                foreign = _foreign_subject(clause, others)
                if foreign:
                    # a shared clause whose subject belongs to another domain ("Can I take sick leave while running a funded
                    # research project?") is complete only when the research evidence states that subject too
                    details = list(details) + [Detail("foreign", "the %s subject asked about (%s)" % (foreign[0], ", ".join(foreign[1])),
                                                      _foreign_check(foreign[1]))]
                assessment: Optional[AskAssessment] = None
                if not (procedural or process):
                    # the root completeness rule (Step 4.5A-R): an unrecognised specific ask can never be completed by topic
                    # relevance; a recognised one must be stated with its unmapped specifics in the answering sentence
                    assessment = assess_ask(clause, details, task.question)
                    details = list(assessment.details)
                    if (assessment.shape == ASK_UNKNOWN and ("where" in assessment.kinds or set(assessment.residue) & _CONTACT_HEADS)
                            and SpecialistId.INSTITUTIONAL not in others):
                        # an unrecognised "where" may ask for a place: Institutional Navigation gets it as data
                        handoffs.append(self._handoff(task, SpecialistId.INSTITUTIONAL, clause, index, "shared", len(others)))
                        others = others + [SpecialistId.INSTITUTIONAL]
                        other = others[0]
                locality = conference_locality(clause)
                rejected: List[Dict[str, Any]] = []
                clause_findings: List[str] = []
                best_covered: FrozenSet[str] = frozenset()
                seen_anywhere: set = set()
                partial_list = False
                for cand in self._candidates(clause, clause_route, clause_concepts, details, procedural, system, rejected, locality):
                    partial_list = partial_list or cand.list_complete is False
                    if len(clause_findings) >= MAX_FINDINGS_PER_CLAUSE:
                        break
                    key = _quote_key(cand.quote)
                    page = cand.item["meta"].get("page")
                    duplicate = next((f for f in findings if f.page == page and
                                      (key in _quote_key(f.evidence_quote) or _quote_key(f.evidence_quote) in key)), None)
                    if duplicate is not None:
                        if index not in duplicate.metadata["clauses"]:
                            duplicate.metadata["clauses"].append(index)
                        if duplicate.finding_id not in clause_findings:
                            clause_findings.append(duplicate.finding_id)
                        if len(cand.covered) > len(best_covered):
                            best_covered = cand.covered
                        seen_anywhere |= set(cand.covered)
                        continue
                    finding = self._finding(task, len(findings) + 1, cand, index, clause_route, details)
                    if assessment is not None and assessment.shape == ASK_RECOGNISED:
                        keys = [_details._detail_key(d) for d in details if d.family == "ask_grounding"]
                        finding.metadata["relation"] = RELATION_STRONG if keys and all(k in cand.covered for k in keys) else RELATION_AMBIGUOUS
                    findings.append(finding)
                    clause_findings.append(finding.finding_id)
                    if len(cand.covered) > len(best_covered):
                        best_covered = cand.covered
                    seen_anywhere |= set(cand.covered)
                grounding = [d for d in details if d.family == "ask_grounding"]
                relation = _relation_strength(assessment.shape if assessment is not None else None, grounding, best_covered)
                ambiguous = _relation_downgrade(relation)
                uncovered = [d for d in details if _details._detail_key(d) not in best_covered and d.family != "ask_grounding"]
                if not clause_findings:
                    outcome = "not_found"
                    if assessment is not None and assessment.shape == ASK_UNKNOWN:
                        missing.append("the specific ask of clause %d is not recognised by the Research specialist: %s" % (index, clause[:120]))
                    if navigation:
                        missing.append("contact or location details are not in the Research evidence for clause %d; Institutional "
                                       "Navigation holds them: %s" % (index, clause[:120]))
                    if procedural:
                        missing.append("step-by-step procedure for %s requested by clause %d is not in the indexed sources: %s"
                                       % (system, index, clause[:120]))
                    else:
                        missing.append("no evidence in the Research scope for clause %d: %s" % (index, clause[:120]))
                elif procedural or process or uncovered or ambiguous:
                    outcome = "partial"
                    if ambiguous:
                        missing.append("the asked relation (%s) is not established unambiguously by the evidence for clause %d; semantic "
                                       "relation verification belongs to the Milestone 5 Verifier: %s"
                                       % (", ".join(d.label for d in grounding), index, clause[:120]))
                    if procedural:
                        missing.append("step-by-step procedure for %s requested by clause %d is not in the indexed sources: %s"
                                       % (system, index, clause[:120]))
                    if process:
                        missing.append("the complete submission procedure for clause %d is not stated in the handbook, which gives policy stages only: %s"
                                       % (index, clause[:120]))
                    for d in uncovered:
                        if d.family == "documents" and partial_list:
                            missing.append("the complete list of required documents is not quoted for clause %d (only part of the handbook's list "
                                           "is in the evidence): %s" % (index, clause[:120]))
                        elif d.family == "contact":
                            missing.append("contact or location details are not in the Research evidence for clause %d; Institutional "
                                           "Navigation holds them: %s" % (index, clause[:120]))
                        elif d.family == "ask_unknown":
                            missing.append("the specific ask of clause %d is not recognised by the Research specialist; the quoted evidence is "
                                           "related to the topic but does not confirm the answer: %s" % (index, clause[:120]))
                        elif _details._detail_key(d) in seen_anywhere:
                            missing.append("%s is not stated together with the other requested details in one place for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        else:
                            missing.append("%s not found in the evidence for clause %d: %s" % (d.label, index, clause[:120]))
                else:
                    outcome = "supported"
                reports.append({"index": index, "clause": clause, "outcome": outcome, "procedural": procedural, "process": process,
                                "system": system, "findings": clause_findings, "rejected": rejected,
                                "requested_details": [d.label for d in details], "uncovered_details": [d.label for d in uncovered],
                                "relation": relation, "route": clause_route,
                                "ask_shape": assessment.shape if assessment is not None else ASK_PROCESS,
                                "ask_kinds": list(assessment.kinds) if assessment is not None else [],
                                "ask_residue": list(assessment.residue) if assessment is not None else [],
                                "inherited_ask": assessment.inherited if assessment is not None else None,
                                "shared_with": other.value if other is not None else None})
        except Exception as exc:                          # retrieval, reranking or construction failed
            return self._result(task, FindingStatus.ERROR, summary="execution failed", handoffs=handoffs,
                                metadata={"error": _safe_error(exc), "clauses": reports}, started=started)

        for index, clause in enumerate(unprocessed, start=len(processed)):
            missing.append("clause %d was not processed (limit of %d clauses per task): %s" % (index, MAX_CLAUSES, clause[:120]))
        for system in systems_seen:
            limitations.append("The indexed handbook contains policy-level references to %s only; no procedural guide "
                               "is indexed, so step-by-step instructions cannot be provided." % system)
        handed_off = [r for r in reports if r["outcome"] == "handed_off"]
        if handed_off:
            limitations.append("%d clause(s) belong to another specialist and were handed off, not answered." % len(handed_off))
        unowned = [r for r in reports if r["outcome"] == "unowned"]
        if unowned:
            limitations.append("%d clause(s) carry no research cue and were not searched." % len(unowned))

        metadata = {"clauses": reports, "unprocessed_clauses": unprocessed, "route": route,
                    "owned_clauses": owned_clauses, "gate": MIN_RERANK_SCORE}
        if owned_clauses == 0:
            return self._result(task, FindingStatus.OUT_OF_SCOPE, handoffs=handoffs, limitations=limitations, missing=missing,
                                summary="no assigned clause belongs to Research & Innovation", metadata=metadata, started=started)
        if findings and not missing:
            status, summary = FindingStatus.SUPPORTED, "%d clause(s) supported by %d finding(s)" % (owned_clauses, len(findings))
        elif findings:
            status, summary = FindingStatus.PARTIAL, "%d finding(s); %d gap(s)" % (len(findings), len(missing))
        else:
            status, summary = FindingStatus.NOT_FOUND, "no evidence in the Research scope for the assigned clause(s)"
        return self._result(task, status, findings=findings, missing=missing, handoffs=handoffs,
                            limitations=limitations, summary=summary, metadata=metadata, started=started)


__all__ = [
    "ASK_GENERAL", "ASK_PROCESS", "ASK_RECOGNISED", "ASK_UNKNOWN", "AskAssessment", "assess_ask",
    "RELATION_AMBIGUOUS", "RELATION_STRONG", "RelationFrame",
    "ANCHOR_CONCEPTS", "APPROVED_SOURCE_ID", "CATEGORY_GROUPS", "Concepts", "EXCLUDED_SECTIONS", "EXTRA_PAGES", "GENERIC_TERMS",
    "MAX_CANDIDATES_PER_CLAUSE", "MAX_CLAUSES", "MAX_FINDINGS_PER_CLAUSE", "MAX_LIST_WORDS", "MAX_QUOTE_WORDS", "OWNED_SECTIONS", "SCOPE_SECTIONS",
    "SHARED_SECTIONS", "SYSTEMS", "ResearchInnovationSpecialist", "ResearchResources", "VOCABULARY_CONCEPTS", "classify_clause",
    "clause_segments", "compile_scope", "complete_list", "concepts", "conference_locality", "detail_covered", "focus_terms",
    "is_procedure_request", "is_process_request", "is_relevant", "is_table_row", "locality_conflict", "navigation_intent", "other_domains",
    "quote_problem", "relevance", "requested_details", "select_quote", "system_named", "unit_coverage", "whole_item",
]
