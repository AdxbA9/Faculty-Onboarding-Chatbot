"""
Faculty Services & HR Specialist (canonical id: ``faculty_services``).

DECISION OWNED
    Whether the approved, source-scoped evidence answers the employment,
    benefits, leave, evaluation-process, promotion-process, discipline and
    faculty-services part of a faculty question, expressed as structured
    findings. Nothing else: the specialist does not route the question (the
    Coordinator did), does not write the user-facing answer (Synthesis will),
    does not verify (the Verifier will) and never calls another specialist.

VERSION 1 IS DETERMINISTIC AND EXTRACTIVE
    No LLM call is made anywhere in this module (``llm_used`` is always
    False). Every finding is extractive: ``claim`` is the quoted handbook
    text itself and ``evidence_quote`` is an exact slice of a retrieved
    chunk. Nothing is paraphrased, so nothing can be invented.

THE SAME GATES AS THE FROZEN SPECIALISTS, READ WITH HR VOCABULARY
    1. source and page scope compiled from the section map for the owned and
       shared section identities (never a whole chapter);
    2. section eligibility on a shared page (``SectionGuard``);
    3. relevance by HR concepts (one anchor or two vocabulary concepts);
    4. the production reranker gate;
    5. quote quality (shared rules; the salary table rows are the one HR
       table shape);
    6. completeness: the requested details, the shared ask gate (D-079 to
       D-084: concepts, terms, roles, relation direction, numeric facts) and
       the HR fact gate (D-086, D-087): a leave type, a pay status, a
       faculty category, an administrative role, a rank, a money amount or
       range and its qualifiers (per child, total, annual, monthly, one-time,
       up to) must sit in the SAME local item of the evidence as the asked
       number. Facts of one unit never mix.

PROCEDURES AND SOURCE GAPS
    A "how do I apply / resign / obtain" request is a policy-process request:
    the handbook states stages only, so such a clause is at most ``partial``
    and only the documented stages are quoted. Payroll, leave applications,
    leave balances, visa renewal, ID cards, parking, HR forms and the
    Employee Handbook's contents are not in the indexed source: they are
    recognised as HR topics and fail closed (``not_found`` or ``partial``).

TASKS
    ``task.question`` is the full original question; the assigned work is
    ``task.context["focus_clauses"]``. A clause is handled only when it
    carries an HR cue (the shared lexicon in ``domain_cues.py``); a clause
    with another domain's cue is returned as a ``HandoffRequest``; a pure
    place or contact ask is Institutional Navigation's and is never answered
    from HR text.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ...config import MIN_RERANK_SCORE, PLAN_F_MAX_SUBTASKS
from ...retrieval import RetrievalScope, deduplicate_by_text, gather_candidates
from ...sources import SectionMap, SourceRecord, section_map_for
from ...text_utils import normalize_text
from ..types import ALL_ROUTES
from . import claim_model as _cm
from . import details as _details
from . import evidence as _evidence
from . import grounding as _grounding
from . import scoping as _scoping
from .base import Specialist, redact_error
from .contracts import Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from .details import (Detail, _OBLIGATION, _PENALTY_EVIDENCE, _TIMING, _detail_key, _frequency_ok, _limit_ok, _LIMIT_MAX, _LIMIT_MIN,
                      _money_ok, authority_check, responsibility_check)
from .domain_cues import HR_CONTEXT, HR_OWNERSHIP, cue_text, hr_owned, research_owned
from .evidence import _spans
from .retrieval_intent import resolve_clause_retrieval_intent
from .scoping import SectionGuard

#: The only source this specialist may retrieve from.
APPROVED_SOURCE_ID = "uos_faculty_handbook_2025_26"

#: Handbook sections the specialist owns: (chapter, printed number, label). Section identities only, never a chapter.
#: Both printed 7.8 sections (department and college level promotion procedures) are owned.
OWNED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (2, "2.1", "Faculty Appointment Classes"),
    (2, "2.2", "Recruitment and Hiring"),
    (2, "2.3", "Contract Types"),
    (2, "2.4", "Probation and Resignation Notice"),
    (2, "2.5", "Confirmation"),
    (2, "2.6", "Dual Career"),
    (2, "2.7", "Special Appointments for Academic Administrators"),
    (2, "2.8", "Termination for Cause of a Contract"),
    (2, "2.9", "Hiring and Transitioning of Emirati Faculty Members"),
    (2, "2.10", "Clinical Faculty Appointment and Promotion"),
    (2, "2.11", "Emeritus Faculty Appointment"),
    (2, "2.12", "Administrative Appointment of Regular Faculty"),
    (2, "2.13", "Prohibited Nepotism and Employment of Relatives"),
    (3, "3.9", "Role and Responsibilities of Administrative Faculty"),
    (3, "3.13", "Professional Ethics and Disciplinary Actions"),
    (3, "3.14", "Faculty Discipline and Grievance"),
    (4, "4.1", "Basic Salary"),
    (4, "4.2", "Monetary Benefits"),
    (4, "4.3", "Leaves and Absences"),
    (4, "4.4", "Duplication of Benefits"),
    (4, "4.5", "Compensation for Administrative Positions"),
    (4, "4.6", "Annual Incentives for Distinguished Faculty Members"),
    (4, "4.7", "Demise Policy"),
    (6, "6.1", "General Evaluation Considerations"),
    (6, "6.6", "Evaluation Process and Criteria"),
    (6, "6.7", "Special Issues and Considerations"),
    (6, "6.8", "Weights and Grades for each Criteria"),
    (6, "6.9", "The Faculty Information System"),
    (6, "6.10", "Rubric for Contract Renewal"),
    (6, "6.11", "Evaluation Rubric"),
    (6, "6.12", "Continuous Improvement"),
    (6, "6.13", "Faculty Information Systems User Guide"),
    (7, "7.1", "Promotion Criteria"),
    (7, "7.5", "Promotion to the Rank of Associate Professor"),
    (7, "7.6", "Promotion to the Rank of Full Professor"),
    (7, "7.8", "Promotion Procedures at the Department Level; Promotion Procedures at the College Level"),
    (7, "7.9", "Promotion Procedures at the University Level"),
    (7, "7.10", "Preparing the Promotion Dossier"),
    (7, "7.11", "General Provisions"),
    (7, "7.12", "Appeals"),
    (15, "15.4", "Child Care Center"),
    (15, "15.6", "Medical Clinics"),
    (16, "16.5", "Policy Prohibiting Sexual Harassment"),
)
#: Sections whose text HR may quote for a bounded purpose only: the weight, process and timing of the research and
#: teaching components of evaluation and promotion, and the procedural sentence on journal classification. Research
#: output (publications, journals, Scopus) and teaching evidence (course files, syllabi) are the other specialists'
#: content and are handed off by clause.
SHARED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (6, "6.4", "Research and Scholarly Activities Evaluation (weight and process only)"),
    (7, "7.2", "Research and Scholarship (promotion weight and process only)"),
    (7, "7.3", "Teaching and Academic Advising (evidence requirement and weight only)"),
    (7, "7.7", "Criteria for Accrediting Periodicals and Journals (the classification request only)"),
)
SCOPE_SECTIONS: Tuple[Tuple[int, str, str], ...] = OWNED_SECTIONS + SHARED_SECTIONS
#: The chapter 2 preamble (p56) lies before the first section record of the chapter and carries the appointment
#: definitions (faculty classes, visiting term, ranked positions) and the policy authority; ``compile_scope`` admits it
#: as an extra page because no section record covers it. The chapter 4 lead-in (p106) is covered by 4.1 and needs
#: nothing (verified against the real section map in Step 4.11).
EXTRA_PAGES: Tuple[Tuple[int, int], ...] = ((56, 56),)
#: Sections deliberately kept out because another specialist owns them or no one does. Listed so the decision is visible.
EXCLUDED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (1, "1.16", "Contact Information (directory rows: Institutional Navigation)"),
    (3, "3.1", "Teaching Responsibilities"),
    (3, "3.2", "Responsibilities in Student Advising and Availability"),
    (3, "3.3", "Responsibility to Curriculum Development"),
    (3, "3.4", "Responsibility to Office Hours"),
    (3, "3.5", "Responsibility to Research"),
    (3, "3.6", "Responsibility to Community Service"),
    (3, "3.7", "Professional Development Responsibilities"),
    (3, "3.8", "Responsibility to Quality Assurance and Institutional Effectiveness"),
    (3, "3.10", "Academic Freedom and Responsibility"),
    (3, "3.11", "Workload Allocation Model (WLAM)"),
    (3, "3.12", "Academic Integrity and Student Honor Code"),
    (5, "5.1", "Faculty development goals"),
    (5, "5.2", "Categories of Faculty Development; New Faculty Development Training"),
    (5, "5.3", "Faculty Peer Observation of Teaching"),
    (5, "5.4", "Participation in Conferences"),
    (5, "5.5", "Policy on Offering Courses in the Summer Semester"),
    (6, "6.2", "Teaching and Learning Evaluation"),
    (6, "6.3", "Academic Advising (evaluation)"),
    (6, "6.5", "Community Service Evaluation"),
    (7, "7.4", "University and Community Service (promotion)"),
    (15, "15.1", "Central Laboratories"),
    (15, "15.2", "Business Service Department"),
    (15, "15.3", "Facilities Management"),
    (15, "15.5", "Disability Resource Center"),
    (15, "15.7", "Sport Facilities"),
    (15, "15.8", "Food Courts"),
    (16, "16.1", "Faculty Professional Consultancy"),
    (16, "16.2", "Conflict of Interest"),
    (16, "16.3", "Animal Use in Research"),
    (16, "16.4", "Human Use in Research"),
    (16, "16.6", "Guidelines on Artificial Intelligence and Large Language Models"),
)

#: University systems the handbook mentions in HR text. Never specialists.
SYSTEMS: Tuple[str, ...] = ("fis", "myuos", "hr portal")

MAX_FINDINGS_PER_CLAUSE = 2
MAX_CANDIDATES_PER_CLAUSE = 10
MAX_CLAUSES = PLAN_F_MAX_SUBTASKS
MAX_QUOTE_WORDS = 90


# ---------------------------------------------------------------------------
# HR concept vocabulary (relevance)
# ---------------------------------------------------------------------------
#: (pattern, concept, anchor). Multi-word phrases first; each matched span is consumed. An ANCHOR establishes topic
#: relevance alone; a plain vocabulary concept needs a second match. No concept establishes completeness.
_CONCEPT_PATTERNS: Tuple[Tuple[str, str, bool], ...] = (
    (r"sick\s+leaves?", "sick_leave", True),
    (r"maternity\s+leaves?", "maternity_leave", True),
    (r"paternity\s+leaves?", "paternity_leave", True),
    (r"(?:unpaid\s+)?special\s+leaves?", "special_leave", True),
    (r"hajj(?:\s+leaves?)?", "hajj_leave", True),
    (r"vacation\s+(?:leaves?|days)|annual\s+leaves?|\bvacations?\b", "vacation_leave", True),
    (r"sabbatical\s+leaves?|\bsabbaticals?\b", "sabbatical", True),
    (r"leaves?\s+(?:of\s+absence|and\s+absences?)|\bleaves?\b(?!\s+(?:the|my|our|a|an|early|campus|class|office|for|to)\b)", "leave", True),
    (r"end[- ]of[- ]service(?:\s+(?:salary|compensation|benefits?|gratuity))?|\bgratuity\b", "end_of_service", True),
    (r"clinical\s+allowances?", "clinical_allowance", True),
    (r"furniture\s+allowances?", "furniture_allowance", True),
    (r"shipment\s+allowances?|\bshipment\b", "shipment_allowance", True),
    (r"housing\s+allowances?", "housing_allowance", True),
    (r"(?:children'?s?\s+)?education(?:al)?\s+allowances?|children'?s?\s+education|school\s+fees|textbook\s+fees", "education_allowance", True),
    (r"(?:annual\s+)?(?:return\s+)?air\s?tickets?|ticket\s+allowances?|return\s+tickets?|\btickets?\b|\bairfare\b", "air_ticket", True),
    (r"health\s+insurance|medical\s+insurance|\binsurance\b|medical\s+cover(?:age)?", "insurance", True),
    (r"\bhousing\b|\baccommodation\b|housing\s+compounds?", "housing", True),
    (r"basic\s+(?:monthly\s+)?salar(?:y|ies)|salary\s+scales?|pay\s+scales?|\bsalar(?:y|ies)\b|\bpayroll\b|\bpay\s?slips?\b", "salary", True),
    (r"monetary\s+benefits?|employment\s+package|\bbenefits?\b", "benefit", True),
    (r"duplication\s+of\s+benefits?|\bduplicat\w*", "duplication", True),
    (r"compensation\s+for\s+administrative\s+positions?|administrative\s+compensation|monthly\s+compensation|\bcompensation\b", "admin_compensation", True),
    (r"annual\s+incentives?|bonus\s+awards?|\bincentives?\b|\bbonus(?:es)?\b|outstanding\s+faculty", "incentive", True),
    (r"\bdemise\b", "demise", True),
    (r"probation(?:ary)?(?:\s+(?:period|year))?", "probation", True),
    (r"resign\w*", "resignation", True),
    (r"notice\s+periods?|\bnotice\b", "notice", True),
    (r"terminat\w*|dismiss\w*", "termination", True),
    (r"contract\s+renewals?|renewal\s+rubric|\brenew\w*|non[- ]renewal", "contract_renewal", True),
    (r"\bcontracts?\b|\bcontractual\b|(?:employment|job)\s+offers?|\boffers?\b", "contract", True),
    (r"\bconfirm\w*", "confirmation", True),
    (r"appointment\s+classes|faculty\s+appointments?|\bappoint\w*", "appointment", True),
    (r"recruit\w*|\bhir(?:e|es|ed|ing)\b|background\s+check|security\s+clearance|good\s+conduct\s+certificate", "recruitment", True),
    (r"\bnepotism\b|employment\s+of\s+relatives|\brelatives?\b", "nepotism", True),
    (r"\bemirati\b", "emirati", True),
    (r"clinical\s+(?:faculty|appointments?|professors?)|\bclinical\b", "clinical", True),
    (r"\bemeritus\b", "emeritus", True),
    (r"dual\s+career", "dual_career", True),
    (r"administrative\s+(?:appointments?|positions?|faculty|roles?|duties)|academic\s+administrators?|\badministrators?\b", "administrative", True),
    (r"performance\s+evaluations?|annual\s+evaluations?|faculty\s+evaluations?|evaluation\s+(?:process|criteria|weights?|rubrics?|forms?|periods?|results?)|"
     r"\bevaluat\w*|\bappraisals?\b", "evaluation", True),
    (r"faculty\s+information\s+systems?|\bfis\b", "fis", True),
    (r"\bpromot\w*", "promotion", True),
    (r"\bdossiers?\b", "dossier", True),
    (r"\bappeals?\b", "appeal", True),
    (r"\bgrievances?\b", "grievance", True),
    (r"disciplinary\s+(?:actions?|measures?|procedures?)|\bdisciplin\w*|\bmisconduct\b|\bwarning\s+letters?\b", "discipline", True),
    (r"professional\s+ethics|\bethics\b|\bethical\b", "ethics", True),
    (r"(?:sexual\s+)?harassment", "harassment", True),
    (r"(?:entry|residence|residency)\s+visas?|\bvisas?\b|emirates\s+id|passport\s+unit|\bpassports?\b|\bresidenc[ey]\b", "visa", True),
    (r"employee\s+(?:files?|records?)|personal\s+data|document\s+authentication|attest\w*", "employee_file", True),
    (r"employee\s+handbook", "employee_handbook", True),
    (r"child\s*care(?:\s+cent(?:er|re))?|\bnursery\b", "child_care", True),
    (r"(?:university\s+)?medical\s+clinics?|\bclinics?\b|\bumc\b", "clinic", True),
    (r"human\s+resources?(?:\s+(?:department|office))?|\bhr\b|\bpersonnel\b", "hr", True),
    (r"\bemploy\w*", "employment", True),
    (r"(?:employment|job)\s+offers?|offer\s+(?:letters?|acceptance)", "offer", True),
    (r"mandatory\s+training|\bmentorship\b", "onboarding", True),
    (r"teaching\s+load\s+reductions?|reduction\s+in\s+teaching\s+load|load\s+reductions?", "load_reduction", True),
    (r"academic\s+ranks?|\branks?\b|\btenure\b", "rank", True),
    # academic ranks are anchors so that a salary-table row ("Associate Professor 25,318 - 37,926") is on the topic of an
    # ask that names the rank; the row itself carries no other concept
    (r"associate\s+professors?", "rank_associate_professor", True),
    (r"assistant\s+professors?", "rank_assistant_professor", True),
    (r"(?:full\s+)?professors?\b", "rank_professor", True),
    (r"(?:senior\s+)?lecturers?\b", "rank_lecturer", True),
    (r"(?:senior\s+)?instructors?\b", "rank_instructor", True),
    # ---- plain vocabulary: needs a second match --------------------------
    (r"faculty\s+members?|\bfaculty\b", "faculty", False),
    (r"\bspouses?\b|\bchildren\b|\bchild\b|\bdependents?\b|\bfamily\b", "dependents", False),
    (r"\bdays?\b", "day", False),
    (r"\bmonths?\b|\bmonthly\b", "month", False),
    (r"\byears?\b|\byearly\b|\bannual(?:ly)?\b", "year", False),
    (r"\bweeks?\b|\bweekly\b", "week", False),
    (r"\bpaid\b|\bpay\b|\bunpaid\b", "pay", False),
    (r"approv\w*", "approval", False),
    (r"\bchancellor\w*|vice[- ]chancellor\w*|\bvcaa\b", "chancellor", False),
    (r"\bdeans?\b|\bchairs?\b|chairpersons?\b|\bdirectors?\b|coordinators?\b", "dean_chair", False),
    (r"councils?\b|committees?\b|board\s+of\s+trustees|\bbot\b", "council", False),
    (r"\bdeduct\w*", "deduction", False),
    (r"percent(?:age)?s?\b", "percent", False),
    (r"\baed\b|\bdirhams?\b|\bdhs?\b", "currency", False),
    (r"\bservice\b", "service", False),
    (r"\bentitle\w*|\beligib\w*|\bqualif\w*", "eligibility", False),
    (r"\bextend\w*|\bextension\b", "extension", False),
    (r"\bmaximum\b|\bminimum\b|\blimits?\b|\bup\s+to\b", "limit", False),
    (r"part[- ]time", "part_time", False),
    (r"full[- ]time", "full_time", False),
    (r"\bvisiting\b", "visiting", False),
    (r"\bregular\b", "regular", False),
    (r"\bnew\s+faculty\b|\bnewly\b", "new_faculty", False),
    (r"\btravel\w*", "travel", False),
    (r"\btraining\b", "training", False),
    (r"\bresearch\w*", "research", False),
    (r"\bteaching\b", "teaching", False),
    (r"credit\s+hours?", "credit_hours", False),
    (r"\bsemesters?\b", "semester", False),
    (r"\bdocuments?\b|\bforms?\b", "document", False),
    (r"\bportal\b|my\s?uos\b", "portal", False),
    (r"submi(?:t|ts|tted|tting|ssions?)\b", "submit", False),
    (r"recommend\w*", "recommend", False),
    (r"\bweights?\b|\bgrades?\b", "weight", False),
    (r"\bcommunity\b", "community", False),
    (r"\bstudents?\b", "student", False),
    (r"\bexams?\b|\bexaminations?\b", "exam", False),
    (r"\bworkloads?\b", "workload", False),
)
_COMPILED_CONCEPTS = tuple((re.compile(p, re.I), c, a) for p, c, a in _CONCEPT_PATTERNS)
ANCHOR_CONCEPTS: FrozenSet[str] = frozenset(c for _, c, a in _CONCEPT_PATTERNS if a)
VOCABULARY_CONCEPTS: FrozenSet[str] = frozenset(c for _, c, _ in _CONCEPT_PATTERNS)

#: A leave-type anchor is on the topic of "leave"; a renewal anchor is on the topic of "contract".
_ANCHOR_IMPLIES: Dict[str, str] = {"sick_leave": "leave", "maternity_leave": "leave", "paternity_leave": "leave", "special_leave": "leave",
                                   "hajj_leave": "leave", "vacation_leave": "leave", "contract_renewal": "contract",
                                   "clinical_allowance": "benefit", "furniture_allowance": "benefit", "shipment_allowance": "benefit",
                                   "housing_allowance": "housing", "education_allowance": "benefit"}

#: Words that never establish relevance on their own (absent from the vocabulary above; checked by tests).
GENERIC_TERMS: FrozenSet[str] = frozenset({
    "policy", "policies", "process", "procedure", "procedures", "require", "required", "requirement", "requirements", "information",
    "university", "uos", "member", "members", "academic", "rule", "rules", "guideline", "guidelines", "office", "department", "college",
    "provide", "provided", "use", "apply", "application", "time", "number", "date", "dates", "regulations", "according",
    "level", "levels",                                            # the organisational level is bound by the HR fact gate (D-088)
})


@dataclass(frozen=True)
class Concepts:
    """Normalised concepts of a text. ``anchors`` is a subset of ``vocab``."""

    anchors: FrozenSet[str]
    vocab: FrozenSet[str]


def _consume_concepts(text: str) -> Tuple[List[Tuple[str, bool]], str]:
    work = cue_text(text.lower())
    found: List[Tuple[str, bool]] = []

    def consume(concept: str, anchor: bool):
        def _sub(match: "re.Match[str]") -> str:
            found.append((concept, anchor))
            return " " * (match.end() - match.start())
        return _sub

    for pattern, concept, anchor in _COMPILED_CONCEPTS:
        work = pattern.sub(consume(concept, anchor), work)
    return found, work


def concept_residue(text: str) -> str:
    """``text`` with every HR concept span blanked: what a clause says beyond its topic words."""
    return _consume_concepts(text)[1]


def concepts(text: str) -> Concepts:
    """HR concepts found in ``text`` after normalisation."""
    found, _work = _consume_concepts(text)
    anchors = frozenset(c for c, a in found if a)
    vocab = {c for c, _ in found}
    vocab.update(_ANCHOR_IMPLIES[a] for a in anchors if a in _ANCHOR_IMPLIES)
    return Concepts(anchors=anchors, vocab=frozenset(vocab))


def relevance(clause: Concepts, text: str) -> Tuple[int, int]:
    """(shared anchor concepts, shared vocabulary concepts) between the clause and ``text``."""
    other = concepts(text)
    shared = clause.anchors & other.anchors
    implied = {a for a in clause.anchors - other.anchors if _ANCHOR_IMPLIES.get(a) in other.vocab}
    return len(shared) + len(implied), len(clause.vocab & other.vocab)


def is_relevant(clause: Concepts, text: str) -> bool:
    """One shared anchor concept, or two shared vocabulary concepts."""
    anchors, vocab = relevance(clause, text)
    return anchors >= 1 or vocab >= 2


def concept_spans(text: str) -> List[Tuple[int, int]]:
    """The spans of HR concept phrases in ``text`` (a subject that is no role: "sick leave", "furniture allowance")."""
    lowered = cue_text(text.lower())
    spans: List[Tuple[int, int]] = []
    for pattern, _concept, _anchor in _COMPILED_CONCEPTS:
        for m in pattern.finditer(lowered):
            if not any(not (m.end() <= a or m.start() >= b) for a, b in spans):
                spans.append((m.start(), m.end()))
    return sorted(spans)


# ---------------------------------------------------------------------------
# Categories, ranks, leave types, pay statuses, administrative roles (D-086)
# ---------------------------------------------------------------------------
#: Faculty categories a clause may name; the evidence must name the same category next to the subject (shared rule).
CATEGORY_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("regular faculty", r"\bregular\s+faculty\b|\branked\s+(?:full[- ]time\s+)?(?:regular\s+)?faculty\b|\bassistant\s+professors?\b|\bassociate\s+professors?\b|\bfull\s+professors?\b"),
    ("visiting faculty", r"\bvisiting\b"),
    ("lecturers", r"\b(?:senior\s+)?lecturers?\b"),
    ("instructors", r"\binstructors?\b"),
    ("clinical faculty", r"\bclinical\b"),
    ("part-time", r"\bpart[- ]time\b"),
    ("full-time", r"\bfull[- ]time\b"),
    ("emirati track", r"\bemirati\b"),
    ("emeritus faculty", r"\bemeritus\b"),
    ("administrative role", r"\b(?:associate\s+|assistant\s+)?deans?\b|\bchair(?:persons?|s)?\b|\bdirectors?\b|\bprogram(?:me)?\s+coordinators?\b|\badministrators?\b|"
     r"\badministrative\s+(?:positions?|roles?|faculty|appointments?)\b"),
    ("new faculty", r"\bnew\s+faculty\b|\bnewly\s+(?:hired|appointed|joined)\b|\bduring\s+(?:the\s+)?probation\b|\bfirst\s+year\s+of\s+(?:the\s+)?(?:contract|appointment|service)\b"),
    ("dependents", r"\bspouses?\b|\bchildren\b|\bchild\b|\bdependents?\b"),
)
_COMPILED_CATEGORIES = tuple((label, re.compile(p, re.I)) for label, p in CATEGORY_GROUPS)

#: Category axes for the HR fact gate: a fact stated for one member of an axis settles nothing about another member.
_AXES: Dict[str, Dict[str, "re.Pattern[str]"]] = {
    "time": {"part_time": re.compile(r"\bpart\s*time\b"), "full_time": re.compile(r"\bfull\s*time\b")},
    "track": {"regular": re.compile(r"\bregular\s+faculty\b|\branked\b"), "visiting": re.compile(r"\bvisiting\b"),
              "lecturer": re.compile(r"\blecturers?\b"), "instructor": re.compile(r"\binstructors?\b"), "clinical": re.compile(r"\bclinical\b"),
              "emeritus": re.compile(r"\bemeritus\b"), "emirati": re.compile(r"\bemirati\b"), "part_timer": re.compile(r"\bpart\s*timers?\b|\badjunct\b")},
    "stage": {"new": re.compile(r"\bnew\s+faculty\b|\bnewly\b|\bduring\s+(?:the\s+)?probation\b|\bfirst\s+year\b")},
    "admin": {"dean": re.compile(r"(?<!associate )(?<!assistant )\bdeans?\b"), "associate_dean": re.compile(r"\bassociate\s+deans?\b"),
              "assistant_dean": re.compile(r"\bassistant\s+deans?\b"), "chair": re.compile(r"\bchair(?:persons?|s)?\b|\bhead\s+of\s+department\b"),
              "director": re.compile(r"\bdirectors?\b"), "program_coordinator": re.compile(r"\bprogram(?:me)?\s+coordinators?\b")},
    "rank": {"professor": re.compile(r"(?<!associate )(?<!assistant )(?<!clinical )\b(?:full\s+)?professors?\b"),
             "associate_professor": re.compile(r"\bassociate\s+professors?\b"), "assistant_professor": re.compile(r"\bassistant\s+professors?\b"),
             "lecturer": re.compile(r"\blecturers?\b"), "instructor": re.compile(r"\binstructors?\b")},
}
_LEAVE_TYPES: Dict[str, "re.Pattern[str]"] = {
    "sick": re.compile(r"\bsick\b"), "maternity": re.compile(r"\bmaternity\b"), "paternity": re.compile(r"\bpaternity\b"),
    "special": re.compile(r"\bspecial\s+leave\b"), "hajj": re.compile(r"\bhajj\b"),
    "vacation": re.compile(r"\bvacations?\b|\bannual\s+leave\b|\bholidays?\b"), "sabbatical": re.compile(r"\bsabbatical\b"),
    "study": re.compile(r"\bstudy\s+leave\b"), "compassionate": re.compile(r"\bcompassionate\b"), "emergency": re.compile(r"\bemergency\s+leave\b"),
}
_PAY_STATUS: Dict[str, "re.Pattern[str]"] = {
    "unpaid": re.compile(r"\bwithout\s+pay\b|\bunpaid\b|\bno\s+pay\b|\bnot\s+paid\b"),
    "half": re.compile(r"\bhalf\s+pay\b|\bhalf[- ]paid\b"),
    "full": re.compile(r"\bfull\s+pay\b|\bfully\s+paid\b|\bwith\s+pay\b|\bpaid\s+leave\b|\bpaid\b(?!\s+(?:to|for|by|into|out))"),
}
_MONEY_QUALIFIERS: Dict[str, Dict[str, "re.Pattern[str]"]] = {
    "scope": {"per_child": re.compile(r"\bper\s+child\b|\bfor\s+each\s+child\b|\beach\s+child\b"),
              "total": re.compile(r"\btotal\b|\bfor\s+all\b|\bin\s+total\b|\baltogether\b|\bcombined\b")},
    "period": {"annual": re.compile(r"\bannual(?:ly)?\b|\bper\s+year\b|\byearly\b|\beach\s+year\b|\bevery\s+year\b"),
               "monthly": re.compile(r"\bmonthly\b|\bper\s+month\b|\beach\s+month\b|\bevery\s+month\b"),
               "one_time": re.compile(r"\b1\s+time\b|\bone[- ]time\b|\bonce\b|\blump\s+sum\b")},
}
_LIMIT_UP_TO = re.compile(r"\bup\s+to\b|\bmaximum\s+of\b|\bmaximum\b|\bat\s+most\b|\bnot\s+(?:to\s+)?exceed\w*|\bno\s+more\s+than\b|\bceiling\b|"
                          r"\bnot\s+more\s+than\b|\b(?:no|not)\s+longer\s+than\b|\blongest\b|\bat\s+the\s+most\b|\bcapped\s+at\b|\blimited\s+to\b")
_LIMIT_AT_LEAST = re.compile(r"\bat\s+least\b|\bminimum\s+of\b|\bminimum\b|\bno\s+less\s+than\b|\bnot\s+less\s+than\b|\b(?:no|not)\s+fewer\s+than\b|"
                             r"\bshortest\b|\bat\s+the\s+least\b")
_EXTENSION_WORDS = re.compile(r"\bextend\w*|\bextension\b|\bprolong\w*")
_RENEWAL_WORDS = re.compile(r"\brenew\w*")
_LATER_STAGE = re.compile(r"\bsubsequent(?:ly)?\b|\bsecond\b|\blater\b|\bfurther\b|\bnext\b|\bthereafter\b|\bafter\s+that\b")
_FIRST_STAGE = re.compile(r"\bfirst\b|\binitial(?:ly)?\b")
#: A rule stated for the probation stage ("resign during probation", "on probation"): it settles no fact ask that does not
#: name probation, whatever the general rule says.
_PROBATION_SCOPE = re.compile(r"\b(?:during|on|in|within|throughout)\s+(?:the\s+|their\s+)?probation(?:ary)?(?:\s+period|\s+year)?\b")
#: A list lead-in that states nothing itself, its first marker glued on ("...involve the following steps: A.").
_LEAD_IN = re.compile(r"(?:following|includes?|as\s+follows|comprises?|consists?\s+of|steps?|criteria|conditions|procedures?)\s*:\s*"
                      r"(?:[A-Za-z]\.|\d{1,2}\.|[•▪●])?\s*$", re.I)
#: Words that mark a stated number as a further tranche rather than the base entitlement ("additional maternity leave").
_ADDITIONAL = re.compile(r"\badditional\b|\bextra\b|\bfurther\b|\bsupplementary\b")
#: Tranche markers that block a base ask (D-088): additional wording in the fact's lead, up-to / maximum or minimum / at
#: least wording in the fact's local window, a non-full pay status at the fact. Each is observable on its own.
_BASE_BLOCKERS: Tuple[str, ...] = ("additional", "limit", "minimum", "pay")
#: Words that only name the unit of a number: an ask whose object is nothing but these carries its subject in its anchors.
_GENERIC_OBJECT: FrozenSet[str] = frozenset({"day", "days", "month", "months", "year", "years", "week", "weeks", "hour", "hours",
                                             "percent", "percentage", "calendar", "working", "consecutive", "academic", "full", "half"})
#: Organisational levels of the promotion procedures (D-088): asked by an explicit level phrase, stated by the bodies that
#: act at a level; the first body an item names is the level that governs the item.
_LEVEL_ASK: Dict[str, "re.Pattern[str]"] = {
    "department": re.compile(r"\bdepartment(?:al)?\s+level\b|\bdepartmental\b"),
    "college": re.compile(r"\bcollege\s+level\b"),
    "university": re.compile(r"\buniversity\s+level\b|\bcentral\s+(?:promotions?\s+)?committee\b"),
}
_LEVEL_EVIDENCE: Dict[str, "re.Pattern[str]"] = {
    "department": re.compile(r"\bdepartment(?:al)?\s+level\b|\bdepartmental\b|\bdepartment\s+(?:academic\s+)?promotions?\s+committee\b|"
                             r"\bdepartment\s+council\b|\bdepartment\s+chair(?:person)?s?\b"),
    "college": re.compile(r"\bcollege\s+level\b|\bcollege\s+(?:academic\s+)?promotions?\s+committee\b|\bcollege\s+council\b|\bcollege\s+deans?\b|"
                          r"\bdeans?\b(?!\s+council)"),
    "university": re.compile(r"\buniversity\s+level\b|\buniversity\s+central\s+committee\b|\bcentral\s+(?:academic\s+)?(?:promotions?\s+)?committee\b|"
                             r"\bdeans\s+council\b|\bboard\s+of\s+trustees\b|\bchancellor\b"),
}


_HEADING_ITEM = re.compile(r"^\d+(?:\.\d+)+\s")
_GLUED_MARKER = re.compile(r"\s(?:[a-h]|\d{1,2})\.$")
_VERB_HINT = re.compile(r"\b(?:shall|must|may|can|could|should|will|would|is|are|was|were|be|been|has|have|had|does|do|did|include\w*|involve\w*|"
                        r"require\w*|receive\w*|submit\w*|form\w*|review\w*|forward\w*|provide\w*|appl(?:y|ies|ied)|grant\w*|approve\w*|chair\w*|"
                        r"prepare\w*|select\w*|process\w*|consist\w*|comprise\w*|entitle\w*|pay|paid|offer\w*|make|made|take\w*|serve\w*)\b")


def _heading_item(item: str) -> bool:
    """A section heading or a list lead-in among the items of a unit: it names a topic, it states no step (D-088). A
    heading is short and starts with its section number, or ends with the glued marker of the list it opens ("promotion
    procedures at the college level a."), or has no verb at all."""
    if _LEAD_IN.search(item):
        return True
    if len(item.split()) > 12:
        return False
    return bool(_HEADING_ITEM.match(item)) or bool(_GLUED_MARKER.search(item)) or not _VERB_HINT.search(item)


def _levels_of(text: str) -> FrozenSet[str]:
    """The organisational levels an ask names explicitly."""
    return frozenset(k for k, p in _LEVEL_ASK.items() if p.search(text))


def _levels_stated(item: str) -> List[str]:
    """The organisational levels an item names, in order of first mention."""
    found = []
    for level, pattern in _LEVEL_EVIDENCE.items():
        m = pattern.search(item)
        if m:
            found.append((m.start(), level))
    return [level for _pos, level in sorted(found)]


def _axis_members(text: str, axis: str) -> FrozenSet[str]:
    return frozenset(m for m, p in _AXES[axis].items() if p.search(text))


def leave_types_of(text: str) -> FrozenSet[str]:
    """The leave types ``text`` names (normalised or raw text)."""
    low = cue_text(text.lower())
    return frozenset(k for k, p in _LEAVE_TYPES.items() if p.search(low))


def pay_statuses_of(text: str) -> FrozenSet[str]:
    """The pay statuses ``text`` names: unpaid, half or full."""
    low = cue_text(text.lower())
    found = {k for k, p in _PAY_STATUS.items() if p.search(low)}
    if "unpaid" in found and "full" in found and not re.search(r"\bfull\s+pay\b|\bwith\s+pay\b|\bpaid\s+leave\b", low):
        found.discard("full")                                     # "without pay" alone: the bare "pay" is no full pay
    return frozenset(found)


# ---------------------------------------------------------------------------
# The HR fact gate (D-086 category and leave-type binding, D-087 money, ranges and rows)
# ---------------------------------------------------------------------------
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
#: Items of an evidence unit: sentences, bullets ("o Dean: ...", "• Housing: ...") and lettered or numbered points.
_ITEM_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=\S)|\s*[•▪●�]\s*|\s+o\s+(?=[a-z])|\s+(?=\d{1,2}\.\s+[a-z])|\s+(?=[a-h]\.\s+[a-z])|\s+(?=\([a-h]\)\s)")
_MONEY_RANGE = re.compile(
    r"(?:aed|dirhams?)\s*(\d+(?:\.\d+)?)\s*(?:to|and|-|–|—)\s*(?:aed\s*)?(\d+(?:\.\d+)?)|"
    r"(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(?:aed\s*)?(\d+(?:\.\d+)?)\s*(?:aed|dirhams?)|"
    r"(\d+(?:\.\d+)?)\s*percent\s*(?:to|and|-|–|—)\s*(\d+(?:\.\d+)?)\s*percent|"
    r"(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:\.\d+)?)\s*percent|"
    r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(?:percent|aed|dirhams?)")
_MONEY_SINGLE = re.compile(r"(?:aed|dirhams?)\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:aed|dirhams?)|(\d+(?:\.\d+)?)\s*percent")


def _hr_norm(text: str) -> str:
    """The shared normalisation (lower case, number words as digits, "%" as percent) with thousands separators removed."""
    return _THOUSANDS.sub("", _grounding.normalise(text))


_ITEM_MARKER = re.compile(r"^(?:o|[•▪●�]|[a-h]\.|\([a-h]\)|\d{1,2}\.)\s+")


def hr_items(text: str) -> List[str]:
    """The local items of ``text``: sentences, bullet items and lettered points of the normalised text, without their
    markers."""
    norm = _hr_norm(text)
    items = []
    for part in _ITEM_SPLIT.split(norm):
        part = _ITEM_MARKER.sub("", (part or "").strip()).strip()
        if part:
            items.append(part)
    return items


def money_facts(norm_text: str) -> List[Dict[str, Any]]:
    """Money and percentage facts of normalised text: (kind, low, high, start, end). A range ("aed 2000 to aed 3000",
    "10 percent to 20 percent", "10 20 percent") is one fact; a single amount has ``high`` None."""
    facts: List[Dict[str, Any]] = []
    taken: List[Tuple[int, int]] = []
    for m in _MONEY_RANGE.finditer(norm_text):
        groups = [g for g in m.groups() if g is not None]
        low, high = float(groups[0]), float(groups[1])
        kind = "percent" if "percent" in m.group(0) else "aed"
        facts.append({"kind": kind, "low": low, "high": high, "start": m.start(), "end": m.end()})
        taken.append((m.start(), m.end()))
    for m in _MONEY_SINGLE.finditer(norm_text):
        if any(not (m.end() <= a or m.start() >= b) for a, b in taken):
            continue
        value = float(next(g for g in m.groups() if g is not None))
        kind = "percent" if "percent" in m.group(0) else "aed"
        facts.append({"kind": kind, "low": value, "high": None, "start": m.start(), "end": m.end()})
    facts.sort(key=lambda f: f["start"])
    return facts


_WINDOW_CUT = re.compile(r"[,;:]|\s+(?:and|or|but|while|whereas)\s+")
_NUMBER_RUN = re.compile(r"\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)*")


def _windows(item: str, facts: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(local, lead) windows of each fact of an item. ``local`` runs from the last comma, semicolon, colon or conjunction
    before the fact (never past the previous fact) to the first one after it (never past the next fact): "1 year,
    extendable to a maximum of 2 years" gives "the faculty probation period is 1 year" and "extendable to a maximum of
    2 years". ``lead`` runs from the item start to the same end, so the subject of a copular statement ("the faculty
    probation period is ...", "a definite term contract of up to 2 years") stands in it."""
    out = []
    for i, f in enumerate(facts):
        floor = facts[i - 1]["end"] if i else 0
        limit = facts[i + 1]["start"] if i + 1 < len(facts) else len(item)
        head = item[floor:f["start"]]
        cuts = list(_WINDOW_CUT.finditer(head))
        if cuts:
            head = head[cuts[-1].end():]
        tail = item[f["end"]:limit]
        cut = _WINDOW_CUT.search(tail)
        if cut:
            tail = tail[:cut.start()]
        out.append((head + item[f["start"]:f["end"]] + tail, item[:f["end"]] + tail))
    return out


def _qualifiers(text: str) -> Dict[str, FrozenSet[str]]:
    return {axis: frozenset(k for k, p in members.items() if p.search(text)) for axis, members in _MONEY_QUALIFIERS.items()}


@dataclass(frozen=True)
class HRAsk:
    """What a clause asks at the fact level: its numeric facts, money facts, leave types, pay statuses, categories per axis,
    extension / renewal / stage wording and limit wording."""

    numeric: Tuple[Dict[str, Any], ...]
    money: Tuple[Dict[str, Any], ...]
    leave_types: FrozenSet[str]
    pay: FrozenSet[str]
    axes: Dict[str, FrozenSet[str]]
    qualifiers: Dict[str, FrozenSet[str]]
    extension: bool
    renewal: bool
    later_stage: bool
    first_stage: bool
    up_to: bool
    at_least: bool
    anchors: FrozenSet[str] = frozenset()
    additional: bool = False                                      # "additional", "extra", "further": a further tranche is asked
    all_anchors: FrozenSet[str] = frozenset()                     # every anchor, the value-kind anchors included
    levels: FrozenSet[str] = frozenset()                          # department / college / university level asked


def _ask_numeric(norm: str) -> List[Dict[str, Any]]:
    facts = []
    for f in _grounding.numeric_facts(norm):
        values, obj = f["values"], f["obj"]
        if any(v >= 1900 for v in values) or (not obj and not f["freqs"]):
            continue
        head = obj[0] if obj else ""
        if head and (head in _grounding._LIGHT_VERBS or head in _grounding._FILLERS or (_grounding._known_verb(head) and not head.endswith(("ing", "s")))):
            continue
        if obj and obj[0] in ("percent", "aed", "dirhams", "dirham"):
            continue                                              # money and percentages are money facts
        facts.append(f)
    return facts


def hr_ask(clause: str) -> HRAsk:
    """The fact-level reading of a clause."""
    norm = _hr_norm(clause)
    stated = concepts(norm)
    anchors = frozenset(a for a in stated.anchors if a not in _FACT_FREE_ANCHORS)
    return HRAsk(numeric=tuple(_ask_numeric(norm)), money=tuple(money_facts(norm)), leave_types=leave_types_of(norm), pay=pay_statuses_of(norm),
                 axes={axis: _axis_members(norm, axis) for axis in _AXES}, qualifiers=_qualifiers(norm),
                 extension=bool(_EXTENSION_WORDS.search(norm)), renewal=bool(_RENEWAL_WORDS.search(norm)),
                 later_stage=bool(_LATER_STAGE.search(norm)), first_stage=bool(_FIRST_STAGE.search(norm)),
                 up_to=bool(_LIMIT_UP_TO.search(norm)), at_least=bool(_LIMIT_AT_LEAST.search(norm)), anchors=anchors,
                 additional=bool(_ADDITIONAL.search(norm)), all_anchors=frozenset(stated.anchors), levels=_levels_of(norm))


#: Anchors that name the kind of value rather than the subject of the fact: never demanded inside the fact's lead. Notice,
#: contract and appointment name what a duration measures (a notice period, a contract term, an appointment term) and are
#: demanded like any subject (D-088).
_FACT_FREE_ANCHORS: FrozenSet[str] = frozenset({"rank", "offer", "employment", "hr", "benefit", "evaluation", "contract_renewal", "confirmation",
                                                "recruitment", "administrative", "onboarding", "eligibility", "salary"})


def _generic_object(af: Dict[str, Any]) -> bool:
    """The ask's numeric object is nothing but unit words ("6 months", "30 calendar days"): its subject is its anchors."""
    return all(w in _GENERIC_OBJECT for w in af["obj"])


def _subject_in_lead(ask: HRAsk, lead: str, generic: bool = False) -> bool:
    """The ask's subject stands in the fact's lead (the item start to the fact's local end, never beyond the item): the
    demanded anchors always; when none is demanded and the object is a bare unit word, every anchor of the ask (D-088)."""
    stated = concepts(lead)
    present = stated.anchors | stated.vocab
    if ask.anchors:
        return ask.anchors <= present
    if generic and ask.all_anchors:
        return ask.all_anchors <= present
    return True


def _fact_object_ok(ask: HRAsk, uf: Dict[str, Any]) -> bool:
    """The stated number measures what the ask asks: an anchor named by the fact's own object ("a six-month notice") that
    the ask never names (the probation period) makes it another fact (D-088)."""
    stated = concepts(" ".join(uf["obj"])).anchors
    return not {a for a in stated if a not in ask.all_anchors and _ANCHOR_IMPLIES.get(a) not in ask.all_anchors}


def _tranche_markers(local: str, lead: str) -> FrozenSet[str]:
    """How the stated number is marked: "additional" in its lead, a ceiling or a floor in its local window, a non-full
    pay status at the fact."""
    found = set()
    if _ADDITIONAL.search(lead):
        found.add("additional")
    if _LIMIT_UP_TO.search(local):
        found.add("limit")
    if _LIMIT_AT_LEAST.search(local):
        found.add("minimum")
    if (pay_statuses_of(local) or pay_statuses_of(lead)) & {"unpaid", "half"}:
        found.add("pay")
    return frozenset(found)


def _qualifiers_ok(ask: HRAsk, local: str, lead: str) -> bool:
    """D-088: a base ask (no pay status, no additional, limit or extension wording) is never settled by a number marked
    additional, up to / maximum, minimum / at least or without / half pay ("additional maternity leave without pay, up to a
    maximum of 30 days" states no 30-day maternity leave). An ask that says "additional" needs that word at the fact; a
    fact that says it answers only an ask that says it too or names the tranche by a non-full pay status the fact states."""
    markers = _tranche_markers(local, lead) & frozenset(_BASE_BLOCKERS)
    base = not (ask.pay or ask.additional or ask.up_to or ask.at_least or ask.extension)
    if base:
        return not markers
    if ask.additional and "additional" in _BASE_BLOCKERS and not _ADDITIONAL.search(lead):
        return False
    if "additional" in markers and not ask.additional:
        statuses = pay_statuses_of(local) or pay_statuses_of(lead)
        if not (ask.pay & {"unpaid", "half"} and ask.pay <= statuses):
            return False
    return True


def _fact_unit(fact: Dict[str, Any]) -> str:
    """The unit a stated number counts ("years", "working days" -> "day", "dependent children" -> "children")."""
    words = [w for w in fact["obj"] if w in _GENERIC_OBJECT] or list(fact["obj"][-1:])
    return _stem(words[-1]) if words else ""


def _qualifier_bound(ask: HRAsk, local: str, lead: str, fact: Dict[str, Any], others: Sequence[Tuple[Dict[str, Any], str]]) -> bool:
    """D-096: a value asked with a qualifier is the stated value with that qualifier. A maximum (up to, at most, not
    exceeding, longest) is a fact marked as a limit, or an unmarked fact whose item states no larger limit or extension of
    the same unit ("the probation period is one year, extendable to a maximum of two years" states no one-year maximum); a
    minimum likewise with a smaller floor; a fact marked the other way ("a minimum six-month notice" for a maximum ask)
    never answers."""
    if ask.up_to == ask.at_least:
        return True
    markers = _tranche_markers(local, lead)
    unit, value = _fact_unit(fact), max(fact["values"])
    if ask.up_to:
        if "limit" in markers:
            return True
        return "minimum" not in markers and not any(
            _fact_unit(f) == unit and max(f["values"]) > value and (_LIMIT_UP_TO.search(w) or _EXTENSION_WORDS.search(w)) for f, w in others)
    if "minimum" in markers:
        return True
    return "limit" not in markers and not any(_fact_unit(f) == unit and min(f["values"]) < value and _LIMIT_AT_LEAST.search(w) for f, w in others)


def _context_ok(ask: HRAsk, item: str, local: str, lead: str) -> bool:
    """The matched fact is the asked one: the ask's anchor concepts stand in the fact's lead ("the faculty probation
    period is 1 year", not "a contract of up to 2 years, including a probation year"); the item names the asked leave
    type; the local window names the asked pay status; the item names the asked category, administrative role and rank
    and no other member of the same axis unless the asked one leads the fact; extension wording agrees locally; stage
    wording ("first", "subsequently") agrees in the lead; a rule for the probation stage ("during probation") settles no
    ask that does not name probation; an asked organisational level (department, college, university) is the level the
    item's first-named body acts at (D-088)."""
    if not _subject_in_lead(ask, lead):
        return False
    if ask.levels:
        if _heading_item(item):
            return False                                          # "7.8 promotion procedures at the college level" proves no step
        stated = _levels_stated(item)
        if not ask.levels <= set(stated):
            return False
        if len(set(stated)) > 1 and stated[0] not in ask.levels:
            return False                                          # a college step that names the department chair is college-level
    if ask.leave_types:
        if not ask.leave_types <= leave_types_of(item):
            return False
    if ask.pay:
        statuses = pay_statuses_of(local) or pay_statuses_of(lead) or pay_statuses_of(item)
        if not ask.pay <= statuses:
            return False
    for axis, asked in ask.axes.items():
        if not asked:
            continue
        stated = _axis_members(item, axis)
        if not asked <= stated:
            return False
        if axis in ("time", "admin", "rank") and (stated - asked) and not (_axis_members(lead, axis) & asked):
            return False                                          # "part-time and full-time" in one item: the asked one must lead the fact
    if bool(_EXTENSION_WORDS.search(local)) != ask.extension:
        return False                                              # "extendable to 2 years" states no 2-year probation, and the reverse
    if ask.renewal and not _RENEWAL_WORDS.search(item):
        return False
    if ask.later_stage and not _LATER_STAGE.search(lead):
        return False
    if ask.first_stage and _LATER_STAGE.search(lead):
        return False                                              # "subsequently renewed for 4 years" is no first renewal
    if "probation" not in ask.anchors and _PROBATION_SCOPE.search(item):
        return False                                              # "resign during probation ... six-month notice" settles no unscoped notice ask
    return True


def _numeric_in_item(ask: HRAsk, af: Dict[str, Any], item: str, clause_concepts: Callable[[str], object]) -> bool:
    unit_facts = _grounding.numeric_facts(item)
    spans = []
    for f in unit_facts:
        m = _NUMBER_RUN.match(item, f["pos"])
        spans.append({"start": f["pos"], "end": m.end() if m else f["pos"]})
    windows = _windows(item, spans) if unit_facts else []
    for index, (uf, (local, lead)) in enumerate(zip(unit_facts, windows)):
        if uf["values"] != af["values"]:
            continue
        asked_obj = tuple(w for w in af["obj"] if not _ADDITIONAL.fullmatch(w))      # "30 additional days": the tranche word is wording
        if not (_grounding._numeric_object_ok(asked_obj, uf["obj"], clause_concepts) or
                _grounding._numeric_object_ok(asked_obj, uf["obj"] + uf["before"], clause_concepts)):
            continue
        if any(not _grounding._freq_matches(f, uf["freqs"]) for f in af["freqs"]):
            continue
        if not _fact_object_ok(ask, uf):
            continue                                              # "a six-month notice" measures no probation period
        if not _subject_in_lead(ask, lead, _generic_object(af)):
            continue                                              # "the probation period is 1 year" states no notice period
        if not _qualifiers_ok(ask, local, lead):
            continue                                              # "additional ... without pay, up to a maximum of 30 days" is no base entitlement
        if _QUALIFIER_IDENTITY and not _qualifier_bound(ask, local, lead, uf, [(f, w[0]) for k, (f, w) in enumerate(zip(unit_facts, windows)) if k != index]):
            continue                                              # "is one year, extendable to a maximum of two years" states no one-year maximum (D-096)
        if not _context_ok(ask, item, local, lead):
            continue
        return True
    return False


def _money_in_item(ask: HRAsk, am: Dict[str, Any], item: str) -> bool:
    row = salary_row(item)
    if row is not None:
        # a salary-table row states its rank's range without a currency word: the asked amounts must be the row's bounds
        if am["kind"] != "aed" or not _context_ok(ask, item, item, item):
            return False
        if am["high"] is not None:
            return (am["low"], am["high"]) == (float(row[1]), float(row[2]))
        return am["low"] in (float(row[1]), float(row[2]))
    facts = money_facts(item)
    for uf, (local, lead) in zip(facts, _windows(item, facts)):
        window = local
        if uf["kind"] != am["kind"]:
            continue
        if am["high"] is not None:
            if uf["high"] is None or (uf["low"], uf["high"]) != (am["low"], am["high"]):
                continue
        elif uf["high"] is None:
            if uf["low"] != am["low"]:
                continue
        else:
            if not ((ask.up_to and uf["high"] == am["low"]) or (ask.at_least and uf["low"] == am["low"])):
                continue                                          # a single amount never settles a range
        quals = _qualifiers(window)
        conflict = False
        for axis, asked in ask.qualifiers.items():
            if not asked:
                continue
            if not asked <= quals[axis]:
                conflict = True
            elif quals[axis] - asked:
                conflict = True                                   # "25,000 per child" vs "75,000 total"
        if conflict:
            continue
        if not _context_ok(ask, item, local, lead):
            continue
        return True
    return False


def hr_fact_detail(clause: str, clause_concepts: Callable[[str], object] = None) -> Optional[Detail]:
    """The HR fact detail of a clause, or None when the clause asks no numeric, money, leave-type, pay-status or category
    fact. Its check passes a unit only when every asked fact sits in ONE local item of the unit with its leave type, pay
    status, category, administrative role, rank, range and qualifiers."""
    ask = hr_ask(clause)
    if not (ask.numeric or ask.money or ask.leave_types or ask.pay or any(ask.axes.values()) or ask.levels):
        return None
    fn = clause_concepts or concepts
    what = []
    if ask.numeric or ask.money:
        what.append("the asked value with its object and period")
    if ask.leave_types:
        what.append("the leave type (%s)" % ", ".join(sorted(ask.leave_types)))
    if ask.pay:
        what.append("the pay status (%s)" % ", ".join(sorted(ask.pay)))
    for axis, members in ask.axes.items():
        if members:
            what.append("the %s (%s)" % ({"time": "appointment basis", "track": "faculty category", "stage": "appointment stage",
                                          "admin": "administrative role", "rank": "academic rank"}[axis], ", ".join(sorted(members))))
    if ask.levels:
        what.append("the organisational level (%s)" % ", ".join(sorted(ask.levels)))

    def check(unit: str) -> bool:
        items = hr_items(unit)
        if not items:
            return False
        for af in ask.numeric:
            if not any(_numeric_in_item(ask, af, item, fn) for item in items):
                return False
        for am in ask.money:
            if not any(_money_in_item(ask, am, item) for item in items):
                return False
        if not ask.numeric and not ask.money:
            # a leave type, pay status or category asked without a number: the same item must name all of them
            if not any(_context_ok(ask, item, item, item) for item in items):
                return False
        return True

    return Detail("hr_fact", "the asked fact stated in one place: " + "; ".join(what), check)


# ---------------------------------------------------------------------------
# The salary table (D-087)
# ---------------------------------------------------------------------------
_SALARY_ROW = re.compile(r"^\s*(professor|associate\s+professor|assistant\s+professor|lecturer|instructor)\s+(\d{2},?\d{3})\s*[^\d\s]{1,3}\s*(\d{2},?\d{3})\s*$", re.I)
_SALARY_HEADER = re.compile(r"^\s*academic\s+rank\s+basic\s+monthly\s+salary", re.I)
_RANK_ASK = re.compile(r"\b(?:(associate|assistant|full)\s+)?(professors?|lecturers?|instructors?)\b", re.I)
_SALARY_ASK = re.compile(r"\bsalar(?:y|ies)\b|\bpay\s+scale\b|\bbasic\s+monthly\b|\bmonthly\s+salary\b", re.I)


def is_salary_header(text: str) -> bool:
    """The salary-table header ("Academic Rank Basic Monthly Salary (AED)"): the table's schema, never a fact (D-090)."""
    return bool(_SALARY_HEADER.match(text or ""))


def is_salary_data_row(text: str) -> bool:
    """A salary-table data row ("Associate Professor 25,318 - 37,926"): a rank with its lowest and highest amounts."""
    return bool(_SALARY_ROW.match(text or ""))


def _salary_shape(text: str) -> bool:
    """The HR table shape: a salary data row only; the header names columns and states nothing (D-090)."""
    return is_salary_data_row(text)


def salary_row(text: str) -> Optional[Tuple[str, int, int]]:
    """(rank, low, high) of a salary-table line, else None."""
    m = _SALARY_ROW.match(text)
    if not m:
        return None
    rank = re.sub(r"\s+", " ", m.group(1).lower())
    return rank, int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))


def asked_rank(clause: str) -> Optional[str]:
    """The academic rank a salary clause names, canonical ("professor", "associate professor", ...), else None."""
    m = _RANK_ASK.search(cue_text(clause))
    if not m:
        return None
    noun = m.group(2).lower().rstrip("s")
    prefix = (m.group(1) or "").lower()
    if noun == "professor":
        return "professor" if prefix in ("", "full") else prefix + " professor"
    return noun


#: Words that, before a rank word, name another rank than the salary table's rows (D-091). "full" is listed so that the
#: one accepted alias below is an explicit exception, not an accident of the reader.
_RANK_MODIFIERS: Tuple[str, ...] = ("senior", "junior", "principal", "chief", "head", "adjunct", "affiliate", "honorary", "distinguished",
                                    "emeritus", "visiting", "clinical", "research", "teaching", "acting", "interim", "guest", "part time",
                                    "full time", "contract", "tenured", "full")
#: The accepted rank aliases: "full professor" is the Professor row (D-087, D-091). No other modifier is accepted.
_RANK_ALIASES: Dict[str, str] = {"full professor": "professor"}


def qualified_rank(clause: str) -> Optional[str]:
    """The qualified rank a clause names ("senior lecturer", "research professor", "assistant lecturer"), or None for a
    plain rank or an accepted alias ("full professor"). A qualified rank has no row in the salary table (D-091)."""
    low = re.sub(r"[-\u2010\u2011]", " ", cue_text(clause.lower()))
    if _RANK_MODIFIERS:
        mods = "|".join(re.escape(m).replace("\\ ", "\\s+") for m in _RANK_MODIFIERS)
        m = re.search(r"\b(?:" + mods + r")\s+(?:(?:associate|assistant)\s+)?(?:professors?|lecturers?|instructors?)\b", low)
        if m:
            phrase = re.sub(r"\s+", " ", m.group(0)).rstrip("s")
            if phrase not in _RANK_ALIASES:
                return phrase
    m = re.search(r"\b(?:associate|assistant)\s+(?:lecturers?|instructors?)\b", low)
    return re.sub(r"\s+", " ", m.group(0)).rstrip("s") if m else None


def _qualified_salary_detail(clause: str) -> Optional[Detail]:
    """A salary ask about a qualified rank: no row of the table answers it, and no base-rank row is inherited (D-091)."""
    rank = qualified_rank(clause)
    if rank is None:
        return None
    return Detail("salary_row", "the salary range of the requested rank (%s): the salary table has no row for it" % rank, lambda unit: False)


def salary_detail(clause: str) -> Optional[Detail]:
    """A salary ask that names a rank needs the row of that rank, with the asked amounts when the clause states any; a
    qualified rank ("senior lecturer") is unknown and never inherits the base rank's row (D-091)."""
    if not _SALARY_ASK.search(clause):
        return None
    qualified = _qualified_salary_detail(clause)
    if qualified is not None:
        return qualified
    rank = asked_rank(clause)
    if rank is None:
        return None
    amounts = [int(a.replace(",", "")) for a in re.findall(r"\b\d{2},?\d{3}\b", clause)]

    def check(unit: str) -> bool:
        for line in re.split(r"\s*\|\s*|\n", unit):
            row = salary_row(line)
            if row is None or row[0] != rank:
                continue
            if not amounts:
                return True
            if len(amounts) >= 2:
                return (amounts[0], amounts[1]) == (row[1], row[2])
            return amounts[0] in (row[1], row[2])
        return False

    return Detail("salary_row", "the salary range of the requested rank (%s)" % rank, check)


# ---------------------------------------------------------------------------
# Requested details (completeness)
# ---------------------------------------------------------------------------
_LOCATION_EVIDENCE = re.compile(
    r"\b(?:portal|website|web\s?page|e-?\s?services?|online|electronically|hr\s+portal|faculty\s+information\s+system|\bfis\b|my\s?uos|"
    r"rooms?\s+\w?\d+|buildings?\s+\w?\d+|(?:\d+(?:st|nd|rd|th)|ground|first|second|third)\s+floor|located\s+(?:in|at|on)|"
    r"(?:in|at|to|through)\s+the\s+(?:[\w'-]+\s+){0,3}?(?:office|department|building|unit|desk|cent(?:er|re)|hall|campus|passport\s+unit)\b|main\s+campus)", re.I)
_RESPONSIBLE_PARTY = re.compile(
    r"\b(?:hr|human\s+resources?|personnel|passport\s+unit|services\s+department|chancellor|vice[- ]chancellor|vcaa|deans?|chairs?|chairpersons?|"
    r"department|college|university|councils?|committees?|directors?|coordinators?|administrators?|administration|faculty\s+members?|registrar\w*|"
    r"units?|offices?|cent(?:er|re)s?|nursery|clinics?)\b", re.I)
_RESPONSIBLE_VERB = re.compile(r"\b(?:manag\w*|support\w*|maintain\w*|administer\w*|run|runs|operat\w*|provid\w*|handl\w*|overse\w*|responsib\w*|"
                               r"in\s+charge|coordinat\w*|arrang\w*|assist\w*|compil\w*|inform\w*|process\w*)\b", re.I)
_authority_ok = authority_check()
_responsibility_ok = responsibility_check(_RESPONSIBLE_PARTY)
_PERCENT_STATED = re.compile(r"\d\s*%|\bpercent(?:age)?s?\b", re.I)
_ELIGIBILITY_EVIDENCE = re.compile(r"\b(?:eligib\w*|entitled|qualif\w*|may\s+(?:receive|apply|be\s+granted|take)|is\s+provided\s+for|are\s+provided\s+for|"
                                   r"offered\s+for|for\s+faculty\s+(?:members\s+)?who|applies\s+to|who\s+(?:have|has|hold)|conditions?)\b", re.I)
_DURATION_EVIDENCE = re.compile(r"\b\d+\s*(?:calendar\s+|working\s+|academic\s+)?(?:days?|weeks?|months?|years?)\b|\b(?:one|two|three|four|five|six|"
                                r"seven|eight|nine|ten|fifteen|twenty|thirty|sixty|ninety)[- ](?:calendar\s+|working\s+)?(?:days?|weeks?|months?|years?)\b", re.I)


def _max_ok(unit: str) -> bool:
    return _limit_ok(unit, _LIMIT_MAX, is_table_row)


def _min_ok(unit: str) -> bool:
    return _limit_ok(unit, _LIMIT_MIN, is_table_row)


def _percent_ok(unit: str) -> bool:
    return bool(_PERCENT_STATED.search(unit))


_DETAIL_FAMILIES: Tuple[Tuple[str, str, str, Callable[[str], bool]], ...] = (
    ("fee", "amount or allowance", r"\b(?:how\s+much|what\s+amount|amount\s+of|aed|dirhams?|how\s+(?:large|big)\s+is\s+the\s+(?:allowance|salary|bonus|compensation)|"
     r"what\s+is\s+the\s+(?:basic\s+)?(?:salary|allowance|compensation|bonus)\s+(?:of|for)|cost|fees?)\b",
     lambda u: _money_ok(u) or any(salary_row(line) is not None for line in re.split(r"\s*\|\s*|\n", u))),
    ("penalty", "penalty or consequence", r"\b(?:penalt(?:y|ies)|consequences?|sanctions?|disciplinary\s+(?:actions?|measures?)|what\s+happens\s+(?:if|when)|deduct\w*|"
     r"forfeit\w*|punish\w*)\b", lambda u: bool(_PENALTY_EVIDENCE.search(u)) or bool(re.search(r"\bdeduct\w*|\bforfeit\w*|\bterminat\w*|\bwarning\b", u, re.I))),
    ("deadline", "deadline or date", r"\b(?:deadlines?|due\s+dates?|by\s+when|when\s+(?:is|are|must|do|does|should|will|can)|last\s+day|cut-?off|what\s+date|"
     r"which\s+date|by\s+what\s+date|within\s+how\s+many)\b", lambda u: bool(_TIMING.search(u))),
    ("quantity", "number or amount", r"\b(?:how\s+many|how\s+much|number\s+of|count\s+of|how\s+long)\b", lambda u: False),
    ("duration", "duration or period", r"\b(?:how\s+long|for\s+how\s+many\s+(?:days|weeks|months|years)|duration|length\s+of|what\s+period)\b",
     lambda u: bool(_DURATION_EVIDENCE.search(u))),
    ("frequency", "frequency", r"\b(?:how\s+often|how\s+frequently)\b", _frequency_ok),
    ("minimum", "minimum", r"\b(?:minimum|minimal|at\s+least|fewest|smallest|lowest)\b", _min_ok),
    ("maximum", "maximum or limit", r"\b(?:maximum|maximal|at\s+most|upper\s+limit|limits?|caps?|capped|no\s+more\s+than|largest)\b", _max_ok),
    ("authority", "approving authority", r"\b(?:who\s+(?:approves?|authori[sz]es?|signs?|decides?|grants?|must\s+approve|can\s+approve|gives?\s+approval|recommends?|confirms?)|"
     r"approved\s+by\s+whom|whose\s+approval|which\s+(?:office|body|council|committee|person)\s+approves?|approval\s+(?:from|of|by)\s+whom|who\s+is\s+the\s+approving)\b",
     lambda u: _authority_ok(u) or bool(re.search(r"\b(?:recommend\w*|decid\w*|confirm\w*)\b", u, re.I) and _RESPONSIBLE_PARTY.search(u))),
    ("responsibility", "responsible party or unit",
     r"\bwho\s+(?:manages|supports|maintains|administers|runs|operates|provides|handles|oversees|looks\s+after|is\s+responsible\s+for|is\s+in\s+charge\s+of|arranges|"
     r"assists|compiles|processes|informs|coordinates)\b", lambda u: bool(_RESPONSIBLE_VERB.search(u) and _RESPONSIBLE_PARTY.search(u))),
    ("percentage", "percentage or range", r"\b(?:percent(?:age)?s?|%|what\s+range)\b", _percent_ok),
    ("eligibility", "eligibility condition", r"\b(?:am\s+i\s+eligible|who\s+is\s+eligible|eligib\w*|who\s+qualifies|who\s+can\s+(?:apply|receive|get|take)|"
     r"am\s+i\s+entitled|who\s+is\s+entitled|entitled\s+to)\b", lambda u: bool(_ELIGIBILITY_EVIDENCE.search(u))),
    ("part_time", "part-time distinction", r"\bpart[- ]time\b", lambda u: bool(re.search(r"\bpart[- ]time\b", u, re.I))),
    ("full_time", "full-time distinction", r"\bfull[- ]time\b", lambda u: bool(re.search(r"\bfull[- ]time\b", u, re.I))),
    ("location", "location or system", r"\bwhere\s+(?:do|must|should|can|are|is|to)\b|\bin\s+which\s+(?:system|platform|office|portal)\b|\bwhich\s+(?:system|portal)\b",
     lambda u: bool(_LOCATION_EVIDENCE.search(u))),
    ("requirement", "an explicit requirement statement about the subject",
     r"\bwhat\s+must\b|\bwhat\s+should\s+(?:\w+\s+){0,2}(?:include|contain|cover|provide)\b|\bwhat\s+is\s+required\b|\bwhat\s+are\s+the\s+requirements\b|"
     r"\brequirements?\s+for\b|\bmust\s+(?:contain|include)\b|\bwhat\s+documents?\b", lambda u: bool(_OBLIGATION.search(u))),
)
_COMPILED_FAMILIES = tuple((f, label, re.compile(p, re.I), check) for f, label, p, check in _DETAIL_FAMILIES)


def _quantity_ok(unit: str, subject: Concepts, subject_tokens: FrozenSet[str]) -> bool:
    return _details._quantity_ok(unit, subject, subject_tokens, concepts)


_REGULAR_RANKS: FrozenSet[str] = frozenset({"professor", "associate professor", "assistant professor"})


def requested_details(clause: str) -> List[Detail]:
    """The details a clause explicitly asks for, in a fixed order; a faculty category named in the clause is a detail
    of its own (the evidence must name the same category). A salary ask naming a regular-faculty rank ("full professor",
    the accepted alias of the Professor row) has that category stated by the rank's own salary row (D-091)."""
    details = _details.requested_details(clause, _COMPILED_FAMILIES, _COMPILED_CATEGORIES, concepts)
    rank = asked_rank(clause) if _SALARY_ASK.search(clause) and qualified_rank(clause) is None else None
    if rank not in _REGULAR_RANKS:
        return details
    out = []
    for d in details:
        if d.family == "category" and "regular faculty" in d.label:
            def check(unit: str, _old=d.check, _rank=rank) -> bool:
                return _old(unit) or any((salary_row(line) or ("",))[0] == _rank for line in _table_lines(unit))
            d = Detail(d.family, d.label, check)
        out.append(d)
    return out


_CONTEXT_START = re.compile(r"^\s*(?:such|these|this|those|it|they)\b", re.I)


def unit_coverage(details: Sequence[Detail], clause: Concepts, quote: str) -> FrozenSet[str]:
    """The largest set of requested details that ONE local unit of ``quote`` contains while sharing a concept with the
    clause (shared rule); the grounding detail of a pronoun-initial sentence may be stated with its antecedent. A D-097
    detail whose window the returned quote contradicts is removed (the window veto)."""
    shared = _evidence.unit_coverage(details, lambda text: relevance(clause, text), quote)
    grounding = [d for d in details if d.family == "grounding"]
    if not grounding or "grounding" in shared:
        return _window_vetoed(details, quote, shared)
    units = [_evidence._LIST_MARKER.sub("", u) for u in _evidence._local_units(quote)]
    for index, unit in enumerate(units[1:], start=1):
        if not _CONTEXT_START.match(unit) or relevance(clause, unit)[1] < 1:
            continue
        pair = units[index - 1] + " " + unit
        if all(d.check(pair) for d in grounding):
            others = frozenset(_detail_key(d) for d in details if d.family not in ("grounding", "claim") and d.check(unit))
            need = frozenset(_detail_key(d) for d in details if d.family not in ("grounding", "claim"))
            for d in details:                                     # D-098: the composite with every other detail, grounding on the pair
                if d.family == "claim" and need <= others and getattr(d.check, "ungated", d.check)(unit):
                    others = others | {_detail_key(d)}
            candidate = others | {"grounding"}
            if len(candidate) > len(shared):
                shared = candidate
    return _window_vetoed(details, quote, shared)


def detail_covered(detail: Detail, clause: Concepts, quote: str) -> bool:
    return _evidence.detail_covered(detail, lambda text: relevance(clause, text), quote)


# ---------------------------------------------------------------------------
# Quote units and quote quality (shared rules; HR adds the salary-table row shape)
# ---------------------------------------------------------------------------
def is_table_row(text: str) -> bool:
    """A printed line that is table data rather than prose: a salary-table line (rank and range), or the shared rule."""
    return _evidence.is_table_row(text, (_salary_shape,))


def _header_quote_problem(quote: str) -> Optional[str]:
    """D-090: a salary-table header is never quoted as evidence, whatever the unit kind."""
    core = _evidence._LIST_MARKER.sub("", quote.strip())
    return "salary header" if is_salary_header(core) else None


def quote_problem(quote: str, unit_kind: str, last_unit: bool = False) -> Optional[str]:
    header = _header_quote_problem(quote)
    if header:
        return header
    problem = _evidence.quote_problem(quote, unit_kind, last_unit, table_row=is_table_row)
    if problem is None and _LEAD_IN.search(quote.strip()):
        return "lead-in"                                          # D-088: a heading with "the following steps: A." proves no step
    return problem


# ---------------------------------------------------------------------------
# Ownership and handoff cues (clause classification)
# ---------------------------------------------------------------------------
#: Other-domain cues, phrase-level. Teaching content and research content are handed off even when the clause also
#: carries an HR cue (a shared clause keeps both).
_OTHER_CUES: Tuple[Tuple[SpecialistId, "re.Pattern[str]"], ...] = (
    (SpecialistId.INSTITUTIONAL, re.compile(
        r"\b(?:who(?:m)?\s+(?:do|should|can|could|must|would)\s+i\s+(?:contact|ask|call|e-?mail|talk\s+to|speak\s+to|reach|see|go\s+to|refer\s+to|approach)|"
        r"who\s+(?:is|are)\s+the\s+(?:contact|person|people)\s+(?:for|to)|where\s+(?:do|can|should|must)\s+i\s+(?:submit|send|file|collect|pick\s+up|obtain|get|go|register)\b|"
        r"which\s+(?:office|department|unit)\s+(?:handles|deals\s+with|processes)|"
        r"how\s+(?:do|does|can|could|should)\s+(?:i|we|one)\s+(?:contact|reach|get\s+in\s+touch\s+with)|to\s+whom\s+(?:should|do|can)\s+i\s+(?:send|submit|address)|"
        r"contact\s+(?:person|details|information|numbers?)|phone|telephone|fax|e-?\s?mail\s+address|extension\s+number|"
        r"where\s+(?:is|are|can\s+i\s+find|do\s+i\s+find)\s+(?:the\s+)?[\w' ]{0,30}?(?:office(?!\s+hours?)|department|unit|cent(?:er|re)|building|desk|help\s*desk|library|clinic|"
        r"nursery|registrar\w*|hr|human\s+resources|passport\s+unit|services\s+department|finance\s+(?:office|department)|it\s+(?:services?|support)|reception|security|parking)|"
        r"who\s+(?:manages|supports|maintains|administers|runs|provides)\s+[\w' ]{0,40}?(?:support|help\s*desk|services?|systems?|portal|my\s?uos|office(?!\s+hours?)|unit|department|cent(?:er|re)|desk)|"
        r"which\s+(?:office|unit|department|committee|council|body)\s+(?:handles?|is\s+responsible|deals|manages|approves|do\s+i)|"
        r"organi[sz]ational\s+(?:chart|structure)|governance|chain\s+of\s+command|help\s*desk|it\s+support|service\s+desk|which\s+(?:building|floor|room|campus)|"
        r"portal\s+(?:menu|page|screen)|which\s+menu)\b", re.I)),
    (SpecialistId.TEACHING, re.compile(
        r"\b(?:teaching\s+loads?|office\s+hours?|grad(?:e|es|ing)\s+(?:entry|submission|scale|system)|grade\s+(?:entry|submission)|syllab\w*|course\s+files?|"
        r"student\s+attendance|attendance\s+(?:policy|of\s+students)|exam(?:ination)?s?\s+(?:policy|schedule|rules)|final\s+exams?|midterms?|blackboard|bb\s+ultra|banner|lms|"
        r"class\s+sizes?|credit\s+hours?(?!\s+(?:reduction|release))|contact\s+hours?|wlam|workload\s+allocation|teaching\s+(?:evaluation|responsibilit\w*|assignments?)|"
        r"student\s+evaluations?\s+of\s+teaching|course\s+evaluations?|peer\s+observation|curricul\w*|academic\s+calendar|add[/ -]?drop|incomplete\s+grades?|"
        r"academic\s+integrity|plagiaris\w*|lectures?|classroom|thesis\s+supervision|students'?\s+grades?|workload\s+plans?|"
        r"teach(?:es|ing)?|taught|(?<!appointment\s)(?<!faculty\s)classes|students?|courses?|coursework|exam(?:ination)?s?|grad(?:e|es|ing)|"
        r"office\s+hour|credit\s+hours?(?!\s+(?:reduction|release)))\b", re.I)),
    (SpecialistId.RESEARCH, re.compile(
        r"\b(?:research\s+(?:grants?|projects?|funding|advances?|travel|leave|output|publications?|groups?|institutes?|board|ethics)|grants?\s+(?:applications?|proposals?)|"
        r"publications?|publish\w*|journals?|scopus|h-?\s?index|citations?|patents?|intellectual\s+property|conferences?\s+(?:travel|funding|registration|per\s+diem)|per\s+diem|"
        r"attend(?:ing)?\s+(?:a\s+|the\s+)?conferences?|conference\s+(?:abroad|overseas)|consultanc\w*|seed\s+(?:grants?|projects?)|principal\s+investigator|"
        r"research\s+(?:misconduct|regulations?|committees?))\b", re.I)),
)


def other_domains(clause: str) -> List[SpecialistId]:
    """Every other-domain specialist whose cue fires on the clause: Institutional Navigation first, then by cue count."""
    text = cue_text(clause)
    hits = []
    for order, (sid, pattern) in enumerate(_OTHER_CUES):
        count = len(pattern.findall(text))
        if not count and sid is SpecialistId.RESEARCH and research_owned(clause):
            count = 1                                             # the shared Research ownership lexicon (Step 4.5A-R)
        if count:
            hits.append((count, -order, sid))
    ranked = sorted(hits, key=lambda h: (h[2] is SpecialistId.INSTITUTIONAL, h[0], h[1]), reverse=True)
    return [sid for _count, _order, sid in ranked]


def classify_clause(clause: str) -> Tuple[bool, Optional[SpecialistId]]:
    """(carries a positive HR cue, strongest other-domain specialist or None). The HR cue is the shared lexicon: a
    strong cue, or two context cues; a collision word alone (classes, attendance, travel, evaluation, workload) never
    counts."""
    mine = hr_owned(clause)
    others = other_domains(clause)
    return mine, (others[0] if others else None)


_SYSTEM_CUES = re.compile(r"\b(?:faculty\s+information\s+system|\bfis\b|my\s?uos|hr\s+portal|employee\s+portal|self[- ]service)\b", re.I)
_PROCEDURE_CUES = re.compile(
    r"\b(?:step[- ]by[- ]step|steps?|clicks?|clicking|buttons?|screens?|menus?|tabs?|icons?|navigat\w*|dropdown|log\s?in|login|"
    r"where\s+(?:do|can|should|must)\s+i\s+(?:click|find|go|press|select)|"
    r"how\s+(?:do|can|should|would)\s+i\s+(?:create|upload|enable|add|enter|submit|post|set\s+up|configure|record|open|access|log|attach|import|export|download|edit|delete|update))\b",
    re.I)
#: Named sources the indexed handbook mentions but does not contain (D-089): a question for THEIR contents is at most
#: partial, the mention quoted; a yes / no question about their existence or coverage may be settled by the mention.
_MISSING_SOURCES: Dict[str, "re.Pattern[str]"] = {
    "the Employee Handbook": re.compile(r"\bemployee\s+handbook\b", re.I),
    "the Faculty and Staff Pay Grade and Benefits Chart": re.compile(r"\b(?:pay\s+grade\s+and\s+)?benefits?\s+chart\b", re.I),
}
#: The name of each missing source as a frame may write it, the full Benefits Chart name included (D-090).
_SOURCE_PATTERNS: Dict[str, str] = {
    "the Employee Handbook": r"employee\s+handbook",
    "the Faculty and Staff Pay Grade and Benefits Chart": r"(?:faculty\s+and\s+staff\s+)?(?:pay\s+grade\s+and\s+)?benefits?\s+chart",
}
#: Weak frames: a wh-question or a possessive contents phrase; a coverage question ("what topics does it cover") is a
#: mention ask, not a contents ask.
_CONTENT_FRAME = re.compile(r"^\s*(?:what|which|how|when|where|who|whom)\b|"
                            r"\b(?:employee\s+handbook|benefits?\s+chart)(?:'s)?\s+(?:rules?|provisions?|guidance|requirements?|polic(?:y|ies)|contents?|advice|"
                            r"entitlements?|allowances?|procedures?|steps)\b", re.I)
_COVERAGE_ASK = re.compile(r"\bcover(?:s|ed|age)?\b|\btopics?\b|\bareas?\b|\bsubjects?\b", re.I)
#: Strong frames (D-090): the source is the speaker of the claim ("the Employee Handbook says / states / requires",
#: "provides that") or its frame ("according to / under / in / per the ..."; "as per the ..." inside the asked
#: proposition is not a frame). A strong frame is a contents ask whatever coverage word it carries.
_SOURCE_VERBS: Tuple[str, ...] = ("say", "says", "said", "state", "states", "stated", "require", "requires", "required", "stipulate", "stipulates",
                                  "specify", "specifies", "mandate", "mandates", "prescribe", "prescribes", "provide that", "provides that")
_SOURCE_PREPOSITIONS: Tuple[str, ...] = ("according to", "under", "in", "per")


def _strong_source_frame(clause: str, source: str) -> bool:
    pattern = _SOURCE_PATTERNS.get(source)
    if not pattern:
        return False
    low = cue_text(clause.lower())
    if _SOURCE_VERBS:
        verbs = "|".join(re.escape(v).replace("\\ ", "\\s+") for v in _SOURCE_VERBS)
        if re.search(r"\b" + pattern + r"(?:'s)?\s+(?:" + verbs + r")\b", low):
            return True
    if _SOURCE_PREPOSITIONS:
        preps = "|".join(re.escape(p).replace("\\ ", "\\s+") for p in _SOURCE_PREPOSITIONS)
        if re.search(r"(?<!\bas )\b(?:" + preps + r")\s+(?:the\s+)?" + pattern + r"\b", low):
            return True
    return False


def named_missing_source(clause: str) -> Optional[str]:
    """The named source whose CONTENTS the clause asks for although only its mention is indexed ("What does the Employee
    Handbook say about settling in the UAE?", "Does the Employee Handbook state that ...", "According to the Faculty and
    Staff Pay Grade and Benefits Chart, ..."), else None. A question about whether the source exists or what it covers
    ("Does the Employee Handbook cover living in the UAE?") is answered by the mention and is not a contents ask."""
    for name, pattern in _MISSING_SOURCES.items():
        if not pattern.search(clause):
            continue
        if _strong_source_frame(clause, name):
            return name
        if _CONTENT_FRAME.search(clause) and not _COVERAGE_ASK.search(clause):
            return name
    return None


#: Sameness and independence frames of an ask (D-091): "the same for all", "uniform across", "regardless of", "independent
#: of", "equal pay". Such a claim is settled only by a unit that states sameness or independence OF THE ASKED SUBJECT
#: for the asked comparison (D-092); topic overlap ("based on rank and specialization") never settles it.
_SAMENESS_FRAMES: Tuple[str, ...] = (r"\bthe\s+same\b", r"\bsame\s+(?:for|across|in|to)\b", r"\bidentical\b", r"\buniform(?:ly)?\b",
                                     r"\bequal(?:ly)?\s+(?:for|across|among|between)\b", r"\beveryone\s+(?:receives?|gets?|has)\b",
                                     r"\bequal\s+(?!to\b|opportunit)(?:[\w-]+\s+){0,2}?(?:pay|salary|salaries|leave|benefits?|allowances?|compensation|"
                                     r"treatment|loads?|amounts?|entitlements?|terms|rights|requirements?|criteria)\b")
_INDEPENDENCE_FRAMES: Tuple[str, ...] = (r"\bregardless\s+of\b", r"\birrespective\s+of\b", r"\bindependent(?:ly)?\s+of\b",
                                         r"\b(?:does|do|did)\s*(?:not|n['’]t)\s+depend\b", r"\bnot\s+depend(?:ent)?\s+on\b", r"\bno\s+effect\s+on\b",
                                         r"\bunaffected\s+by\b")
#: Evidence phrases that can state sameness or independence (D-092). Bare "equal" is none: "a rank equal to or higher
#: than the desired rank" is a threshold and "equal opportunity" a principle; "at the same time" is an idiom.
_SAMENESS_EVIDENCE = re.compile(r"\bthe\s+same\b|\bsame\s+(?:for|across|as)\b|\bidentical\b|\buniform(?:ly)?\b|"
                                r"\bequal(?:ly)?\s+(?:for|across|among|between|to\s+all)\b|\b(?:appl(?:y|ies|ied)|distributed|shared|treated|allocated)\s+equally\b|"
                                r"\bregardless\s+of\b|\birrespective\s+of\b|\bapplies\s+to\s+all\b|\bwithout\s+(?:regard|distinction)\b", re.I)
_INDEPENDENCE_EVIDENCE = re.compile(r"\bregardless\s+of\b|\birrespective\s+of\b|\bindependent(?:ly)?\s+of\b|\b(?:does|do)\s+not\s+depend\b|"
                                    r"\bwithout\s+(?:regard|distinction)\b", re.I)
#: Words that name neither the compared subject nor the comparison group of a universality ask.
_UNIVERSAL_STOP: FrozenSet[str] = frozenset(
    "a an the is are was were be been being do does did of for to in at on by as across between among amongst within throughout upon and or "
    "there it its this that these those same identical uniform uniformly equal equally regardless irrespective independent independently "
    "depend depends dependent not no effect unaffected all every each any both everyone everybody receive receives get gets have has had "
    "follow follows apply applies applied applicable available eligible entitled valid fixed determined calculated provided offered paid "
    "given matter one what which how whether must can should will would may shall uos except exception excluding apart other than save".split())
_UNIVERSAL_PREPS: FrozenSet[str] = frozenset("for across in at to between among amongst as within throughout of on by upon".split())
_UNIVERSAL_QUANTIFIERS: FrozenSet[str] = frozenset("all every each any both".split())
#: Words that end the noun phrase governed by "the same" in evidence ("the same field OF specialization").
_SAME_NOUN_STOP: FrozenSet[str] = _UNIVERSAL_PREPS | frozenset(
    "that which who whose being is are was were be been shall should must will may can has have had than with from or "
    "mentioned described outlined stated specified indicated noted listed referred set".split())
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")
#: A sameness stated for a narrower scope than the ask (D-092): a reference back to one earlier procedure ("the same
#: procedure mentioned above"), a sentence under a condition ("If the number ... is less than three, ..."), an exception.
_ANAPHORIC = re.compile(r"^\s*(?:[\w-]+\s+){0,3}?(?:mentioned|described|outlined|stated|specified|indicated|noted|listed|set\s+out|referred\s+to)\s+"
                        r"(?:above|earlier|previously|before|below|herein)\b|^\s*as\s+(?:above|before|previously)\b", re.I)
_CONDITION_OPEN = re.compile(r"^\W*(?:[a-z][.)]\s+)?(?:if|when|whenever|unless|in\s+case|where|should|once|provided)\b", re.I)
_EXCEPTION = re.compile(r"\b(?:with\s+the\s+exception|except|excluding|other\s+than|apart\s+from|save\s+for|unless)\b", re.I)
_ASK_CONDITION = re.compile(r"\b(?:if|when|whenever|unless|in\s+case|provided)\b", re.I)
_ASK_EXCEPTION = re.compile(r"\b(?:except|exception|excluding|other\s+than|apart\s+from)\b", re.I)


@dataclass(frozen=True)
class UniversalAsk:
    """What a universality ask claims (D-092): the compared subject ("promotion requirements"), the comparison group or
    dimension ("every rank", "department and college level", "regardless of gender") and its own condition or exception."""
    kind: str
    subject: FrozenSet[str]
    comparison: FrozenSet[str]
    conditional: bool
    exception: bool


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "ches", "shes", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _raw_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", cue_text(text.lower()).replace("|", " ")) if len(t) > 1]


def _tokens(text: str) -> List[str]:
    return [_stem(t) for t in _raw_tokens(text)]


def universality_frame(clause: str) -> Optional[str]:
    """"sameness" or "independence" when the clause claims one (D-091), else None."""
    low = cue_text(clause.lower())
    if any(re.search(p, low) for p in _SAMENESS_FRAMES):
        return "sameness"
    if any(re.search(p, low) for p in _INDEPENDENCE_FRAMES):
        return "independence"
    return None


def universal_ask(clause: str) -> Optional[UniversalAsk]:
    """The subject and comparison of a sameness or independence ask (D-092), else None. "Are promotion requirements the
    same for every rank?" compares {promotion, requirement} across {rank}; "Do all colleges follow the same promotion
    procedure?" compares {promotion, procedure} across {college}; "regardless of gender" names the dimension {gender}."""
    kind = universality_frame(clause)
    if kind is None:
        return None
    low = cue_text(clause.lower())
    frames = _SAMENESS_FRAMES if kind == "sameness" else _INDEPENDENCE_FRAMES
    first = min((m for m in (re.search(p, low) for p in frames) if m), key=lambda m: m.start())
    before, frame, after = _raw_tokens(low[:first.start()]), _raw_tokens(first.group(0)), _raw_tokens(low[first.end():])
    subject: List[str] = [t for t in frame if t not in _UNIVERSAL_STOP]
    comparison: List[str] = []
    if kind == "sameness" and after and after[0] not in _UNIVERSAL_PREPS and (not frame or frame[-1] not in _UNIVERSAL_PREPS):
        while after and after[0] not in _UNIVERSAL_PREPS:      # "the same promotion procedure (for ...)": the subject
            subject.append(after.pop(0))
    comparison.extend(after)
    index = 0
    while index < len(before):                                   # "all colleges", "every faculty member": comparison groups
        token = before[index]
        if token in _UNIVERSAL_QUANTIFIERS:
            index += 1
            taken = 0
            while index < len(before) and taken < 2 and before[index] not in _UNIVERSAL_STOP:
                comparison.append(before[index])
                index += 1
                taken += 1
            continue
        subject.append(token)
        index += 1
    comparison_set = frozenset(_stem(t) for t in comparison if t not in _UNIVERSAL_STOP)
    subject_set = frozenset(_stem(t) for t in subject if t not in _UNIVERSAL_STOP) - comparison_set
    return UniversalAsk(kind, subject_set, comparison_set, bool(_ASK_CONDITION.search(low)), bool(_ASK_EXCEPTION.search(low)))


def _same_noun(after: str) -> Tuple[str, ...]:
    """The noun phrase "the same" governs in evidence ("the same timeline and process as ..." -> timeline, process); empty
    when "the same" is predicative ("is the same for ...")."""
    out: List[str] = []
    for token in _raw_tokens(after):
        if token in _SAME_NOUN_STOP or len(out) >= 4:
            break
        if token != "and":
            out.append(_stem(token))
    return tuple(out)


def _narrow_scope(ask: UniversalAsk, before: str, after: str) -> bool:
    """The evidence states sameness for a narrower scope than the ask: an anaphoric "the same procedure mentioned above",
    a condition the ask does not carry, or an exception the ask does not carry (D-092)."""
    if _ANAPHORIC.match(after):
        return True
    if not ask.conditional and _CONDITION_OPEN.match(before):
        return True
    return not ask.exception and bool(_EXCEPTION.search(after))


def _same_noun_bound(ask: UniversalAsk, noun: Tuple[str, ...]) -> bool:
    """"the same field / purpose / academic rank" names another property than the asked subject (D-092)."""
    return not noun or bool(set(noun) & ask.subject)


def _subject_bound(ask: UniversalAsk, before: Sequence[str], noun: Tuple[str, ...]) -> bool:
    """Every word of the compared subject is stated with the sameness: before it in its sentence, or as its noun."""
    return bool(ask.subject) and ask.subject <= (set(before) | set(noun))


def _comparison_bound(ask: UniversalAsk, sentence: Sequence[str], after: Sequence[str]) -> bool:
    """The comparison group is in the sentence; an independence dimension follows "regardless of / irrespective of"."""
    return ask.comparison <= set(after if ask.kind == "independence" else sentence)


def universality_bound(unit: str, ask: UniversalAsk) -> bool:
    """A sentence of ``unit`` states the asked sameness or independence of the asked subject for the asked comparison,
    without a narrower scope (D-092)."""
    evidence = _SAMENESS_EVIDENCE if ask.kind == "sameness" else _INDEPENDENCE_EVIDENCE
    for sentence in _SENTENCE_SPLIT.split(cue_text(unit).replace("|", " ")):
        sentence_tokens = _tokens(sentence)
        for m in evidence.finditer(sentence):
            before, after = sentence[:m.start()], sentence[m.end():]
            if re.search(r"\bat\s*$", before, re.I) and re.match(r"\s*time\b", after, re.I):
                continue                                         # "at the same time" is an idiom
            if _narrow_scope(ask, before, after):
                continue
            noun = _same_noun(after) if m.group(0).lower().endswith("same") else ()
            if not _same_noun_bound(ask, noun):
                continue
            if not _subject_bound(ask, _tokens(before), noun):
                continue
            if not _comparison_bound(ask, sentence_tokens, _tokens(after)):
                continue
            return True
    return False


def universality_detail(clause: str) -> Optional[Detail]:
    """The asked sameness or independence, stated by the unit itself for the asked subject and comparison (D-091, D-092)."""
    ask = universal_ask(clause)
    if ask is None:
        return None
    return Detail("universality", "the asked %s (stated as such, not inferred from the topic)" % ask.kind, lambda unit, _a=ask: universality_bound(unit, _a))


#: The asker's own HR record (D-091): leave already used or taken, leave left or remaining, a balance or account. The
#: indexed handbook states policy, never a person's record; a hypothetical ("if I have used all my sick leave") is policy.
_PERSONAL_USAGE: Tuple[str, ...] = (r"\b(?:have|had)\s+i\s+(?:already\s+)?(?:used|taken|consumed|spent|availed)\b",
                                    r"\bi\s+(?:have\s+|'ve\s+|had\s+)?(?:already\s+)?(?:used|took|taken|consumed|spent)\b", r"\bmy\s+used\b",
                                    r"\balready\s+used\b", r"\bdid\s+i\s+(?:use|take)\b")
_PERSONAL_REMAINING: Tuple[str, ...] = (r"\b(?:do|does|did)\s+i\s+(?:still\s+)?have\s+(?:[\w-]+\s+){0,4}?(?:left|remaining|available|outstanding)\b",
                                        r"\b(?:do|does)\s+i\s+still\s+have\b", r"\bi\s+(?:still\s+)?have\s+(?:[\w-]+\s+){0,3}?(?:left|remaining)\b",
                                        r"\bhave\s+i\s+(?:got\s+)?left\b", r"\b(?:remaining|left|available|outstanding)\s+(?:for|to)\s+me\b",
                                        r"\bremain(?:s|ing)?\s+(?:for\s+me|in\s+my)\b", r"\bmy\s+remaining\b", r"\bleft\s+in\s+my\b",
                                        r"\bpersonally\s+have\b")
_PERSONAL_BALANCE: Tuple[str, ...] = (r"\bmy\s+(?:[\w-]+\s+){0,2}?balance\b", r"\bmy\s+current\b", r"\bin\s+my\s+account\b",
                                      r"\bmy\s+(?:[\w-]+\s+)?entitlement\s+remaining\b")
_HYPOTHETICAL = re.compile(r"\b(?:if|once|after|when|whenever|unless)\s+i\b")
#: The asker's own grant, approval, acceptance or decision status (D-092): "have I been granted", "was my promotion
#: approved", "did HR approve my ...", "has my request been accepted". The handbook states who approves and what may be
#: granted, never a decision on a person's case. The mere word "approved / granted / accepted" is not personal.
_GRANT_VERBS = r"(?:granted|given|awarded|allotted|allocated|assigned|offered)"
_APPROVAL_VERBS = (r"(?:approved|confirmed|rejected|denied|declined|processed|signed|renewed|finali[sz]ed|decided|extended|terminated|upheld|"
                   r"overturned|revoked|cancell?ed|reviewed|evaluated)")
_DECISION_ON_ME = r"(?:promoted|appointed|hired|selected|shortlisted|confirmed|terminated|renewed|extended|approved|rejected)"
_PERSONAL_GRANT: Tuple[str, ...] = (
    r"\b(?:have|has|had)\s+i\s+(?:already\s+|ever\s+)?(?:been\s+)?" + _GRANT_VERBS + r"\b",
    r"\b(?:was|were)\s+i\s+(?:already\s+|ever\s+)?" + _GRANT_VERBS + r"\b",
    r"\bi\s+(?:have\s+|'ve\s+|ve\s+|had\s+)?(?:already\s+)?been\s+" + _GRANT_VERBS + r"\b", r"\bi\s+was\s+" + _GRANT_VERBS + r"\b",
    _GRANT_VERBS + r"\s+(?:to|for)\s+me\b",
    r"\b(?:has|have|had|was|were)\s+my\s+(?:[\w'-]+\s+){0,4}?(?:been\s+)?" + _GRANT_VERBS + r"\b",
    r"\b(?:is|are)\s+my\s+(?:[\w'-]+\s+){0,4}?" + _GRANT_VERBS + r"\s*(?:yet|already|now)?\s*[?.!]?\s*$",
    r"\b(?:did|has|have|had|will)\s+(?!to\b)(?:[\w'-]+\s+){1,6}?(?:grant|granted|give|given|award|awarded)\s+me\b")
_PERSONAL_APPROVAL: Tuple[str, ...] = (
    _APPROVAL_VERBS + r"\s+(?:to|for)\s+me\b",
    r"\b(?:has|have|had|was|were)\s+my\s+(?:[\w'-]+\s+){0,4}?(?:been\s+)?" + _APPROVAL_VERBS + r"\b",
    r"\b(?:is|are)\s+my\s+(?:[\w'-]+\s+){0,4}?" + _APPROVAL_VERBS + r"\s*(?:yet|already|now)?\s*[?.!]?\s*$",
    r"\b(?:did|has|have|had|will)\s+(?!to\b)(?:[\w'-]+\s+){1,6}?(?:approve|approved|confirm|confirmed|reject|rejected|deny|denied|decline|declined|"
    r"sign|signed|process|processed|renew|renewed|extend|extended|terminate|terminated|uphold|upheld|overturn|overturned|revoke|revoked|"
    r"cancel|cancell?ed|review|reviewed|evaluate|evaluated)\s+my\b",
    r"\b(?:did|has|have|had|will)\s+(?!to\b)(?:[\w'-]+\s+){1,6}?(?:promote|promoted|appoint|appointed|hire|hired|terminate|terminated)\s+me\b",
    r"\b(?:have|has|had)\s+i\s+(?:already\s+)?been\s+" + _DECISION_ON_ME + r"\b", r"\b(?:was|were)\s+i\s+" + _DECISION_ON_ME + r"\b",
    r"\b(?:is|are)\s+my\s+(?:[\w'-]+\s+){0,4}?(?:still\s+)?(?:pending|under\s+review|in\s+process|being\s+(?:processed|reviewed|considered))\b",
    r"\b(?:outcome|result|results|decision)\s+(?:of|on|about|regarding)\s+my\b",
    r"\bdecision\s+(?:was\s+|has\s+been\s+)?(?:made|taken|reached|issued)\s+(?:on|about|regarding|for|in)\s+my\b",
    r"\b(?:did|have|has)\s+i\s+(?:already\s+)?(?:get|got|gotten|receive|received|obtain|obtained)\s+(?:the\s+|an?\s+|my\s+)?(?:approval|confirmation|decision)\b",
    r"\bmy\s+(?:[\w'-]+\s+){0,3}?(?:approval|decision|application|request|appeal|case|renewal|promotion)\s+status\b", r"\bstatus\s+of\s+my\b")
_PERSONAL_ACCEPTANCE: Tuple[str, ...] = (
    r"\baccepted\s+(?:to|for)\s+me\b",
    r"\b(?:has|have|had|was|were)\s+my\s+(?:[\w'-]+\s+){0,4}?(?:been\s+)?accepted\b",
    r"\b(?:is|are)\s+my\s+(?:[\w'-]+\s+){0,4}?accepted\s*(?:yet|already|now)?\s*[?.!]?\s*$",
    r"\b(?:did|has|have|had|will)\s+(?!to\b)(?:[\w'-]+\s+){1,6}?(?:accept|accepted)\s+my\b",
    r"\b(?:have|has|had)\s+i\s+(?:already\s+)?been\s+accepted\b", r"\b(?:was|were)\s+i\s+accepted\b")
#: Any past action or state of the asker's own case, whatever the verb ("did the committee recommend my promotion", "has
#: my dossier been forwarded", "was I nominated"); a deontic participle ("was I supposed / required / entitled to ...")
#: asks for policy.
_DEONTIC_PARTICIPLE = r"(?!(?:supposed|required|expected|allowed|permitted|entitled|obliged|meant|eligible|needed)\b)"
_PARTICIPLE = r"(?:[a-z]+(?:ed|en)|gone|done|paid|sent|made|held|set|put|cut|got|gotten|kept|left)"
_PERSONAL_DECISION: Tuple[str, ...] = (
    r"\bdid\s+(?!i\b|you\b|we\b)(?:[\w'-]+\s+){1,6}?" + _DEONTIC_PARTICIPLE + r"[a-z]+\s+(?:my|me)\b",
    r"\b(?:has|have|had)\s+(?!i\b|you\b|we\b|to\b)(?:[\w'-]+\s+){1,6}?" + _DEONTIC_PARTICIPLE + _PARTICIPLE + r"\s+(?:off\s+|on\s+|through\s+|up\s+|out\s+)?(?:my|me)\b",
    r"\b(?:has|have|had|was|were)\s+my\s+(?:[\w'-]+\s+){0,4}?(?:been\s+)?" + _DEONTIC_PARTICIPLE + _PARTICIPLE + r"\b",
    r"\b(?:was|were)\s+i\s+(?:ever\s+|already\s+)?" + _DEONTIC_PARTICIPLE + _PARTICIPLE + r"\b",
    r"\b(?:have|has|had)\s+i\s+(?:ever\s+|already\s+)?been\s+" + _DEONTIC_PARTICIPLE + _PARTICIPLE + r"\b",
    r"\bdid\s+i\s+(?:ever\s+|already\s+)?(?:get|receive|obtain|collect)\b",
    r"\b(?:have|has|had)\s+i\s+(?:ever\s+|already\s+)?(?:got|gotten|received|obtained|collected)\b",
    r"\bdid\s+my\b",
    r"\bam\s+i\s+(?:already\s+|now\s+)?(?:approved|confirmed|accepted|granted|promoted|selected|shortlisted|cleared|registered)\b",
    r"\b(?:was|were|has\s+been|have\s+been)\s+(?:decided|determined|concluded|agreed)\s+(?:on|about|regarding|for|in)\s+my\b",
    r"\bwho\s+(?:[a-z]+(?:ed|en)|signed|sent|made|gave|took)\s+my\b")
#: A status ask framed as a condition ("if my leave is approved, ...") or as eligibility ("can unpaid leave be granted to
#: me?") asks for policy, not for a decision on the asker's case.
_STATUS_HYPOTHETICAL = re.compile(r"^\W*(?:if|once|after|whenever|unless|until|in\s+case)\b|\b(?:if|once|after|when|whenever|unless|until)\s+(?:i|my)\b")
_STATUS_MODAL = re.compile(r"\b(?:can|could|may|might|would|should|shall|will)\s+(?:[\w'-]+\s+){0,4}?be\s+(?:granted|given|awarded|allotted|allocated|"
                           r"assigned|offered|approved|accepted|confirmed)\b")


def personal_record(clause: str) -> Optional[str]:
    """"usage", "remaining" or "balance" when the clause asks for the asker's own HR record (D-091), "status" when it asks
    for the asker's own grant, approval, acceptance or decision status (D-092), else None."""
    low = cue_text(clause.lower())
    if _HYPOTHETICAL.search(low):
        return None
    for kind, patterns in (("usage", _PERSONAL_USAGE), ("remaining", _PERSONAL_REMAINING), ("balance", _PERSONAL_BALANCE)):
        if any(re.search(p, low) for p in patterns):
            return kind
    if _STATUS_HYPOTHETICAL.search(low) or _STATUS_MODAL.search(low):
        return None
    if any(re.search(p, low) for p in _PERSONAL_GRANT + _PERSONAL_APPROVAL + _PERSONAL_ACCEPTANCE + _PERSONAL_DECISION):
        return "status"
    return None


#: The detail labels of a personal-record ask: leave usage or balance (D-091), a decision on the asker's case (D-092).
_STATUS_LABEL = "the asker's own grant, approval, acceptance or decision status"


def _personal_label(kind: str) -> str:
    return _STATUS_LABEL if kind == "status" else "the asker's own leave usage or balance (%s)" % kind


#: A condition the ask states as absent ("without Chancellor approval", "with no notice", "no need for approval", "does
#: not require the Chancellor's approval", "the approval is not required", "even if not approved by the Chancellor") is
#: settled only by a sentence that states the same absence; a unit naming the condition as required or given ("with
#: Chancellor approval", "requires the approval of the Chancellor", "must provide a six-month notice") does not, and
#: contradicts it (D-093). "Without pay" is a pay status (D-086) and "regardless of" a universality claim (D-091, D-092):
#: both are bound by their own details.
#: Each absence frame ends at the next frame or a conjunction (D-095): in "additional sick leave without pay be granted
#: without special circumstances" the pay status "without pay" never swallows the later "without special circumstances".
_FRAME_TAIL = r"[^?.,;!]+?(?=\s+(?:without|with\s+no|and|or|but|unless|despite|absent)\b|[?.,;!]|$)"
_NEGATED_CONDITION_FRAMES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("without", (r"\bwithout\s+(?P<what>" + _FRAME_TAIL + r")",)),
    ("with no", (r"\bwith\s+no\s+(?P<what>" + _FRAME_TAIL + r")", r"\bno\s+need\s+for\s+(?P<what>" + _FRAME_TAIL + r")")),
    ("not required", (r"\b(?:not|n['’]t)\s+(?:require|need)\s+(?P<what>" + _FRAME_TAIL + r")",
                      r"\b(?P<what>(?:[\w'-]+\s+){1,4})(?:(?:is|are|be)\s+)?not\s+(?:required|needed|necessary)\b",
                      r"\beven\s+(?:if|when|though)\s+not\s+(?P<what>" + _FRAME_TAIL + r")")),
    ("need no", (r"\b(?:need|needs|needed)\s+no\s+(?P<what>" + _FRAME_TAIL + r")",)),
    ("require no", (r"\b(?:require|requires|required)\s+no\s+(?P<what>" + _FRAME_TAIL + r")",
                    r"\bno\s+(?P<what>(?:[\w'-]+\s+){0,3}?[\w'-]+)\s+(?:is\s+|are\s+)?(?:needed|required|necessary)\b")),
    ("unnecessary", (r"\b(?P<what>(?:[\w'-]+\s+){1,4})(?:(?:is|are|be)\s+)?unnecessary\b",)),
    ("absence", (r"\bin\s+the\s+absence\s+of\s+(?P<what>" + _FRAME_TAIL + r")", r"\bdespite\s+(?:no|the\s+absence\s+of)\s+(?P<what>" + _FRAME_TAIL + r")",
                 r"\babsent\s+(?P<what>(?:[\w'-]+\s+){0,3}?(?:approval|consent|notice|recommendation|permission|authori[sz]ation|agreement|endorsement)s?)\b")),
)
#: Words that end a negated condition ("without notice during probation") and words that name no part of it.
_CONDITION_END = re.compile(r"\s+(?:and|or|but|if|when|while|during|for|to|from|in|at|before|after|until|so|because|is|are|can|may|must)\b.*$", re.I)
_CONDITION_STOP: FrozenSet[str] = _UNIVERSAL_STOP | frozenset("required requires require needed need necessary prior any his her their its "
                                                              "giving providing submitting obtaining receiving getting through via with "
                                                              "under about when if where while whether needing requiring having first".split())
_PAY_CONDITION = re.compile(r"^\s*(?:full\s+|half\s+|any\s+)?pay\b", re.I)
#: A stated absence in the evidence: a negator that governs the condition words directly, or "... is not required" after
#: them (D-093, D-094).
_ABSENCE = re.compile(r"\b(?:without|no|not|never|none\s+of|exempt\s+from|regardless\s+of|irrespective\s+of|in\s+the\s+absence\s+of|absent|"
                      r"(?:need|needs|require|requires)\s+no|[a-z]+n['’]t)\b", re.I)
_ABSENCE_AFTER = re.compile(r"\b(?:(?:is|are|be)\s+)?(?:not\s+(?:required|needed|necessary)|unnecessary)\b", re.I)
#: What a negator governs ends at a clause break or a new verb phrase: in "not exceeding thirty days, may be granted upon
#: approval", "not" governs "exceeding thirty days" only (D-094).
_SCOPE_END = re.compile(r"[,;:()]|\s(?:may|shall|must|will|can|could|should|would|is|are|was|were|be|been|upon|if|when|while|unless|which|who|"
                        r"that|whose|whereas|but)\b", re.I)
_SCOPE_WORDS = 8


def _governed(text: str) -> List[str]:
    """The words a negator governs directly: the text after it up to its clause break, at most eight words."""
    end = _SCOPE_END.search(text)
    return _tokens(text[:end.start()] if end else text)[:_SCOPE_WORDS]


def _governing(text: str) -> List[str]:
    """The words "... is not required" governs: the text before it back to its clause break, at most six words."""
    breaks = list(re.finditer(r"[,;:()]", text))
    return _tokens(text[breaks[-1].end():] if breaks else text)[-6:]
#: The state of the employee's leave in an ask or a unit (D-093, D-094): unused vacation leave exists ("has unused vacation
#: leave", "vacation leave is unused / remains / is left") or does not ("has no unused vacation leave", "no vacation leave
#: remains", "vacation leave is exhausted", "has used all vacation leave"). The negative phrasings are read first.
_VL = r"(?:vacation|annual)\s+leave"
_LEAVE_STATE_NEGATIVE: Tuple[str, ...] = (
    r"\b(?:has|have|having|had|with|holds?)\s+(?:still\s+)?(?:no|not\s+any|zero)\s+(?:unused|remaining|available|outstanding|any)?\s*" + _VL + r"\b",
    r"\bno\s+(?:unused\s+)?" + _VL + r"\s+(?:is\s+|remains\s+)?(?:unused|left|remaining|available|outstanding)\b",
    r"\bno\s+(?:unused\s+)?" + _VL + r"\s+remains\b",
    r"\b" + _VL + r"\s+(?:is|has\s+been|was|are|have\s+been)\s+(?:exhausted|used\s+up|fully\s+used|all\s+used|depleted)\b",
    r"\b(?:has|have|had)\s+(?:already\s+)?used\s+(?:up\s+)?(?:all\s+)?(?:of\s+)?(?:his|her|their|the|its)?\s*" + _VL + r"\b",
    r"\ball\s+(?:of\s+)?(?:the\s+)?" + _VL + r"\s+(?:is|has\s+been|was)\s+used\b")
_LEAVE_STATE_POSITIVE: Tuple[str, ...] = (
    r"\b(?:has|have|having|had|with|holds?)\s+(?:still\s+)?(?:unused|remaining|available|outstanding|any|some)?\s*" + _VL + r"\b",
    r"\bstill\s+(?:has|have)\s+(?:any\s+|some\s+)?(?:unused\s+)?" + _VL + r"\b",
    r"\b(?:unused\s+)?" + _VL + r"\s+(?:is\s+|are\s+)?(?:still\s+)?(?:unused|left|remaining|available|outstanding)\b",
    r"\b(?:unused\s+)?" + _VL + r"\s+(?:still\s+)?remains\b",
    r"\b(?:has|have)\s+(?:some\s+)?leave\s+left\b",
    r"\b" + _VL + r"\s+(?:(?:has|have|had)\s+not\s+(?:yet\s+)?been\s+|(?:is|are|was)\s+not\s+(?:yet\s+)?)(?:fully\s+)?(?:used|taken|exhausted)\b")


@dataclass(frozen=True)
class AskCondition:
    """The conditions an ask states as absent (each a set of words) and the leave states it states (D-093); the conditions it
    states as given and the modifiers it attaches to its population, given or negated (D-096)."""
    absent: Tuple[FrozenSet[str], ...]
    leave_states: Tuple[bool, ...]
    label: str
    present: Tuple[FrozenSet[str], ...] = ()
    absent_clauses: Tuple[FrozenSet[str], ...] = ()               # conditions stated absent as clauses ("if the Chancellor does not approve")
    modifiers: Tuple[FrozenSet[str], ...] = ()
    absent_modifiers: Tuple[FrozenSet[str], ...] = ()
    claim: FrozenSet[str] = frozenset()                           # the roots of the ask outside its clausal conditions


def _negated_conditions(low: str) -> List[Tuple[str, FrozenSet[str]]]:
    out: List[Tuple[str, FrozenSet[str]]] = []
    for _frame, patterns in _NEGATED_CONDITION_FRAMES:
        for pattern in patterns:
            for m in re.finditer(pattern, low):
                what = _CONDITION_END.sub("", m.group("what")).strip()
                if not what or _PAY_CONDITION.match(what):
                    continue                                     # "without pay": the pay-status detail binds it
                words = frozenset(_stem(t) for t in _raw_tokens(what) if t not in _CONDITION_STOP)
                if words:
                    out.append((what, words))
    return out


def _leave_states(text: str) -> List[bool]:
    """The leave states ``text`` states, in order: False for no unused vacation leave, True for some (D-093, D-094)."""
    low = cue_text(text.lower())
    found: List[Tuple[int, int, bool]] = []
    for patterns, state in ((_LEAVE_STATE_NEGATIVE, False), (_LEAVE_STATE_POSITIVE, True)):
        for pattern in patterns:
            for m in re.finditer(pattern, low):
                if not any(not (m.end() <= a or m.start() >= b) for a, b, _s in found):
                    found.append((m.start(), m.end(), state))
    return [state for _a, _b, state in sorted(found)]


def ask_condition(clause: str) -> Optional[AskCondition]:
    """The conditions a clause states as absent and the leave states it states (D-093); the conditions it states as clauses,
    given or absent, and the modifiers of its population, given or negated (D-096); else None."""
    low = cue_text(clause.lower())
    negated = _negated_conditions(low)
    states = tuple(_leave_states(low))
    absent = [w for _what, w in negated]
    parts = ["without " + what for what, _w in negated] + [("has " if s else "has no ") + "unused vacation leave" for s in states]
    present: List[FrozenSet[str]] = []
    absent_clauses: List[FrozenSet[str]] = []
    for text, words, positive in _clausal_conditions(low):
        if positive:
            present.append(words)
            parts.append("if " + text)
        elif not any(words <= a or a <= words for a in absent):    # "even if not approved by the Chancellor" is read once
            absent_clauses.append(words)
            parts.append("unless " + text)
    modifiers = [w for w, positive in _subject_modifiers(low) if positive]
    absent_modifiers = [w for w, positive in _subject_modifiers(low) if not positive]
    parts += ["of the population %s" % " ".join(sorted(w)) for w in modifiers] + ["not of the population %s" % " ".join(sorted(w)) for w in absent_modifiers]
    if not (absent or states or present or absent_clauses or modifiers or absent_modifiers):
        return None
    return AskCondition(tuple(absent), states, "; ".join(parts), tuple(present), tuple(absent_clauses), tuple(modifiers), tuple(absent_modifiers),
                        _roots(_CLAIM_CUT.split(low)[0]))


def _absence_stated(unit: str, words: FrozenSet[str], antecedents: bool = True) -> bool:
    """Some sentence states the absence of the condition: a negator governing every condition word within its own clause,
    or the condition words right before "is not required / is unnecessary" (D-093, D-094); or a negated conditional
    antecedent naming them ("If the case is not referred to the Disciplinary Council ...", "If neither occurs" after the
    sentence it refers to) (D-096). Words are compared by family ("approval" / "approve")."""
    roots = frozenset(_root(w) for w in words)
    previous = ""
    for sentence in _SENTENCE_SPLIT.split(cue_text(unit).replace("|", " ")):
        for m in _ABSENCE.finditer(sentence):
            if _covered(roots, frozenset(_root(t) for t in _governed(sentence[m.end():]))):
                return True
        for m in _ABSENCE_AFTER.finditer(sentence):
            if _covered(roots, frozenset(_root(t) for t in _governing(sentence[:m.start()]))):
                return True
        if antecedents and any(not a.positive and _covered(roots, a.roots) for a in _antecedents(sentence, previous)):
            return True
        previous = sentence
    return False


def _leave_state_stated(unit: str, state: bool) -> bool:
    """Some sentence states the asked leave state with the asked polarity; the opposite polarity contradicts the ask."""
    return any(s == state for s in _leave_states(unit))


def condition_detail(clause: str) -> Optional[Detail]:
    """The asked condition, stated by the unit with the asked polarity (D-093)."""
    cond = ask_condition(clause)
    if cond is None:
        return None

    def check(unit: str, _c=cond) -> bool:
        if not all(_absence_stated(unit, w) for w in _c.absent + _c.absent_modifiers):
            return False                                          # an absent condition or a negated population needs its absence stated
        if not all(_leave_state_stated(unit, s) for s in _c.leave_states):
            return False
        if _FRAME_POLARITY and any(_modifier_negated(unit, w) for w in _c.modifiers):
            return False                                          # "members not meeting UoS standards" is no population "meeting" them (D-096)
        if not all(_condition_matched(unit, w, _c.claim, False) for w in _c.absent_clauses):
            return False                                          # "if the Chancellor does not approve" needs that absence as the claim's condition
        return not _CONDITION_IDENTITY or all(_condition_matched(unit, w, _c.claim) for w in _c.present)

    return Detail("condition", "the asked condition (%s) stated with the asked polarity" % cond.label, check)


#: Predicates whose polarity an affirmative yes/no ask claims (D-094): "Are faculty who resign during probation entitled to
#: annual leave?" is settled only by a clause that states the predicate affirmatively of the asked subject, and a clause
#: that negates it for that subject ("... who resign during probation are not entitled to benefits such as annual leave")
#: contradicts the ask. Modes: "require" - a structural predicate (part of, involved, based on, impact, included) must be
#: stated affirmatively of the subject, topic overlap never stands for it; "stated" - an entitlement (entitled / eligible,
#: allowed / permitted) is bound wherever the unit uses the word, since "may be granted" can state it in other words;
#: "negation" - "required" is bound by its negation only (the D-093 condition binds absent requirements).
#: The negated entitlement in evidence (D-094, D-095): "not entitled / eligible / qualified / given / provided / awarded",
#: "ineligible", "does not receive / get".
_ENTITLEMENT_NEGATED = (r"\b(?:not|never|no\s+longer)\s+(?:be\s+)?(?:entitled|eligible|qualified|given|provided|awarded)\b|"
                        r"\b(?:disentitled|ineligible|disqualified)\b|\b(?:does|do|did|will|shall|can)\s*(?:not|n['’]t)\s+(?:receive|get|qualify)\b")
_POLARITY_PREDICATES: Tuple[Tuple[str, str, str, str], ...] = (
    ("entitled", r"\b(?:entitled|eligible|qualified|qualify|qualifies)\b", _ENTITLEMENT_NEGATED, "stated"),
    ("based on", r"\bbased\s+(?:on|upon)\b", r"\bnot\s+(?:be\s+)?based\s+(?:on|upon)\b", "require"),
    ("impact", r"\b(?:impact|impacts|impacted|affect|affects|affected)\b",
     r"\b(?:not|never)\s+(?:be\s+)?(?:impact|impacts|impacted|affect|affects|affected)\b|\bno\s+(?:impact|effect)\s+on\b", "require"),
    ("part of", r"\bpart\s+of\b", r"\bnot\s+(?:be\s+)?part\s+of\b", "require"),
    ("included", r"\bincluded\b", r"\bnot\s+(?:be\s+)?included\b|\bexcluded\b", "require"),
    ("allowed", r"\b(?:allowed|permitted)\b", r"\bnot\s+(?:be\s+)?(?:allowed|permitted)\b|\bprohibited\b|\bforbidden\b", "stated"),
    ("involved", r"\binvolved\b", r"\bnot\s+(?:be\s+)?involved\b|\buninvolved\b", "require"),
    ("required", r"\b(?:required|requires?)\b", r"\bnot\s+(?:be\s+)?required\b|\b(?:does|do)\s*(?:not|n['’]t)\s+require\b", "negation"),
    # the everyday entitlement verbs (D-095): a negated entitlement of the asked subject contradicts them; their affirmative
    # use of another subject ("a shipment allowance is provided") settles nothing either way
    ("entitled", r"\b(?:receives?|received|gets?|(?:be\s+)?(?:given|provided|awarded)(?!\s+that))\b", _ENTITLEMENT_NEGATED, "negation"),
)
_YES_NO = re.compile(r"^\s*(?:am|is|are|was|were|does|do|did|can|could|will|would|must|should|has|have|may)\b", re.I)
_ASK_NEGATED = re.compile(r"\b(?:not|no|never|without)\b|n['’]t\b", re.I)          # "n't" is a suffix: no word boundary before it
_CLAUSE_SPLIT = re.compile(r",?\s+(?:while|whereas|but|however)\s+|;\s*", re.I)
_RELATIVE_VERB = re.compile(r"\b(?:who|which|that)\s+(?:also\s+|already\s+|still\s+)?$", re.I)
#: A predicate followed by "no" is negated ("get no ticket allowances", "require no notice"): no affirmative claim (D-095).
_NEGATED_AFTER = re.compile(r"\s+(?:no|none)\b", re.I)
#: An ask opened by a negation ("Is not any furniture allowance provided?", "Is no housing provided?", "Isn't any X given?")
#: claims no affirmative predicate (D-096).
_ASK_OPENS_NEGATED = re.compile(r"^\s*(?:(?:am|is|are|was|were|does|do|did|can|could|will|would|must|should|has|have|may|shall)\s+(?:there\s+)?"
                                r"(?:not|no)\b|[a-z]+n['’]t\s+(?:there\s+)?any\b)", re.I)


def _lemma(token: str) -> str:
    t = _stem(token)
    for suffix in ("ing", "ed"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            return t[:-len(suffix)]
    return t


def _lemmas(text: str) -> FrozenSet[str]:
    return frozenset(_lemma(t) for t in _raw_tokens(text))


@dataclass(frozen=True)
class PredicateAsk:
    """The predicate an affirmative yes/no ask claims, its subject words and its object words (D-094)."""
    name: str
    affirmative: "re.Pattern[str]"
    negated: "re.Pattern[str]"
    mode: str
    subject: FrozenSet[str]
    obj: FrozenSet[str]


def predicate_ask(clause: str) -> Optional[PredicateAsk]:
    """The affirmative predicate a yes/no ask claims ("entitled", "based on", "impact", "part of"), else None."""
    low = cue_text(clause.lower())
    if not _YES_NO.match(low) or _ASK_OPENS_NEGATED.match(low):
        return None
    for name, aff, neg, mode in _POLARITY_PREDICATES:
        # a synonym in a relative clause of the subject ("faculty who receive a housing allowance") is no claim (D-095)
        m = next((x for x in re.finditer(aff, low) if not _RELATIVE_VERB.search(low[:x.start()])), None)
        if m is None or _ASK_NEGATED.search(low[max(0, m.start() - 12):m.start()]) or _NEGATED_AFTER.match(low[m.end():]):
            continue
        if _ASK_CONDITION.search(low[:m.start()]):
            continue                                              # "... confirmed if I receive no notification": the condition, not the claim (D-095)
        stop = _CONDITION_STOP | {"i", "me", "we", "you", "who", "whom", "whose", "during", "am", "rather", "than", "over", "instead", "solely"}
        subject = frozenset(_lemma(t) for t in _raw_tokens(low[:m.start()]) if t not in stop)
        obj = frozenset(_lemma(t) for t in _raw_tokens(low[m.end():])[:6] if t not in stop)
        return PredicateAsk(name, re.compile(aff, re.I), re.compile(neg, re.I), mode, subject, obj)
    return None


def _clauses(unit: str) -> List[str]:
    return [c for sentence in _SENTENCE_SPLIT.split(cue_text(unit).replace("|", " ")) for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]


def _predicate_negated_for(clause_text: str, ask: PredicateAsk) -> bool:
    """The clause negates the asked predicate of the asked subject (or of the asked object when the ask names no subject),
    and what it negates shares the asked object ("not based on a single year's performance" negates no "cumulative
    contributions")."""
    who = ask.subject or ask.obj
    m = ask.negated.search(clause_text)
    if m is None or not who <= _lemmas(clause_text):
        return False
    return not (ask.subject and ask.obj) or bool(ask.obj & _lemmas(clause_text[m.end():]))


def _contrast_affirmed(previous: str, clause_text: str, ask: PredicateAsk) -> bool:
    """"not based on a single year's performance but rather on cumulative contributions": the "rather / instead" clause
    after a negated predicate affirms it of its own object (D-094)."""
    if not (ask.negated.search(previous) and re.match(r"^\s*(?:rather|instead)\b", clause_text, re.I)):
        return False
    return (ask.subject or ask.obj) <= _lemmas(previous + " " + clause_text) and (not ask.obj or bool(ask.obj & _lemmas(clause_text)))


def _predicate_affirmed_for(clause_text: str, ask: PredicateAsk) -> bool:
    """The clause states the asked predicate affirmatively of the asked subject (or object)."""
    who = ask.subject or ask.obj
    return bool(ask.affirmative.search(clause_text)) and not ask.negated.search(clause_text) and who <= _lemmas(clause_text)


def predicate_detail(clause: str) -> Optional[Detail]:
    """The asked predicate stated with the asked polarity of the asked subject (D-094)."""
    ask = predicate_ask(clause)
    if ask is None:
        return None

    def check(unit: str, _a=ask) -> bool:
        clauses = _clauses(unit)
        if any(_predicate_negated_for(c, _a) for c in clauses):
            return False
        if _a.mode == "negation":
            return True
        stated = [c for c in clauses if _a.affirmative.search(c) or _a.negated.search(c)]
        if _a.mode == "stated" and not stated:
            return True                                           # "may be granted" may state an entitlement in other words
        return (any(_predicate_affirmed_for(c, _a) for c in stated)
                or any(_contrast_affirmed(clauses[i - 1], clauses[i], _a) for i in range(1, len(clauses))))

    return Detail("predicate", "the asked predicate (%s) stated affirmatively of the asked subject" % ask.name, check)


#: A negated yes/no ask claims the negative proposition (D-095): "Is the basic salary not based on rank?", "Is the probation
#: period not one year?", "Is the College Academic Promotions Committee not formed by the College Council?" are settled only
#: by a clause that negates the same words of the same subject; affirmative evidence ("based on faculty rank", "is one
#: year", "the College Council shall form ...") contradicts them. A rhetorical opener ("Isn't ...?" never matches the
#: auxiliary), "whether / if / or not", a quantity bound ("not more than", "not exceeding"), a negation inside a relative
#: or conditional clause and an absent condition ("does not require X", the D-093 condition) are no negative claims.
_NEGATIVE_ASK = re.compile(r"^\s*(?:am|is|are|was|were|does|do|did|can|could|will|would|must|should|has|have|may|shall)\s+(?P<subject>.+?)\s+"
                           r"(?:not|never|no\s+longer)\s+(?P<rest>.+?)\s*[?.!]?\s*$", re.I)
_NOT_A_NEGATIVE_CLAIM = re.compile(r"\b(?:whether|if|or)\s+not\b|\bnot\s+(?:more|less|exceeding|over|under|fewer|later|earlier|only|just)\b|"
                                   r"\bno\s+(?:more|less|fewer|later|earlier|longer|sooner)\s+than\b|"
                                   r"\b(?:who|which|that|whose)\s+(?:\w+\s+)?(?:do|does|did|have|has|had|is|are|was|were|can|could|will|would)\s+(?:not|no)\b", re.I)
#: "No" as a determiner opens a negative claim too (D-095): an existence ("Is there no furniture allowance?", "Are there not
#: any ...", "Is no housing provided to faculty?") or a light verb ("Do faculty get / receive no return tickets?" claims no
#: entitlement, "Does the award have no impact on the monthly salary?" the negated predicate).
_NEGATIVE_NO_ASK = re.compile(r"^\s*(?:am|is|are|was|were|does|do|did|can|could|will|would|must|should|has|have|may|shall)\s+"
                              r"(?:there\s+(?:no|not\s+any)\s+(?P<exists>.+?)|no\s+(?P<none>.+?)|"
                              r"(?P<subject>.+?)\s+(?P<verb>gets?|receives?|ha(?:s|ve))\s+no\s+(?P<rest>.+?))\s*[?.!]?\s*$", re.I)
_CONDITIONAL_SUBJECT = re.compile(r"\b(?:if|unless|when|whenever|once)\b", re.I)
#: "Is not any X provided?", "Isn't any X given?", "Is there not any X?": a negative existence (D-096).
_NEGATIVE_ANY_ASK = re.compile(r"^\s*(?:(?:is|are|was|were|does|do|did|has|have)\s+(?:there\s+)?not\s+any|"
                               r"(?:is|are|was|were|does|do|did|has|have)n['’]t\s+(?:there\s+)?any)\s+(?P<none>.+?)\s*[?.!]?\s*$", re.I)
_NUMERIC_HEAD = re.compile(r"^\s*(?:aed\s*|dhs?\s*)?(?:\d|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
                           r"forty|fifty|sixty|ninety|half|a\s+(?:year|month|week|day))", re.I)
_RELATION_HEAD = re.compile(r"^\s*(?:be\s+)?(?!(?:entitled|eligible|qualified|required|allowed|permitted|based|involved|included|"
                            r"subject|related|limited|restricted)\b)[a-z]+(?:ed|en)\s+(?:[\w'-]+\s+){0,2}?(?:by|from|to|with|for)\b", re.I)
#: The kinds of negative claim (a value, a relation, a predicate); each is bound the same way, a value right after its
#: negator: "is not one year" negates the value, "if not completed within one year" does not.
_NEGATIVE_KINDS: Tuple[str, ...] = ("value", "relation", "predicate", "existence")
_VALUE_SLACK = 1
#: A copula right after the negator opens the negated phrase ("shall not be one year", "is not being deducted").
_NEGATED_COPULA = re.compile(r"^\s*(?:be|been|being)\b", re.I)
_NEGATOR = re.compile(r"\b(?:not|never|no|none|neither|nor|without|(?P<word>ineligible|disqualified|disentitled))\b|n['’]t\b", re.I)
#: Entitlement words read as one predicate when polarity is compared (D-095): lemmas ("qualified" -> "qualifi") and the
#: participles that only an entitlement reads ("provided", "awarded"; the noun "award" stays itself).
_POLARITY_SYNONYMS: Dict[str, str] = {"eligible": "entitl", "qualifi": "entitl", "qualify": "entitl", "receive": "entitl", "receiv": "entitl",
                                      "get": "entitl", "given": "entitl", "provided": "entitl", "awarded": "entitl"}
_NEGATIVE_STOP: FrozenSet[str] = frozenset(
    "a an the is are was were be been being am do does did has have had can could will would may must should shall i me we you he she they "
    "it its his her their my our your who whom whose which that this these those of to by for from with on upon in at as during any not never "
    "there".split())


def _polarity_lemma(token: str) -> str:
    t = _lemma(token)
    return _POLARITY_SYNONYMS.get(t, _POLARITY_SYNONYMS.get(token, t))


def _polarity_lemmas(text: str) -> FrozenSet[str]:
    tokens = _raw_tokens(text)
    return frozenset(_lemma(t) for t in tokens) | frozenset(_polarity_lemma(t) for t in tokens)


@dataclass(frozen=True)
class NegativeAsk:
    """A negated yes/no ask: its kind, subject words and the words it negates (D-095); the roots and categories of the whole
    ask, against which the scope of the evidence negation is compared (D-096)."""
    kind: str
    subject: FrozenSet[str]
    negated: FrozenSet[str]
    roots: FrozenSet[str] = frozenset()
    members: FrozenSet[str] = frozenset()
    existential: bool = False                                     # "some / certain ... not": a narrower negation settles it (D-097)


def negative_ask(clause: str) -> Optional[NegativeAsk]:
    """The negative claim of a yes/no ask, else None (D-095)."""
    low = cue_text(clause.lower()).strip()
    if _NOT_A_NEGATIVE_CLAIM.search(low) or any(not positive for _w, positive in _subject_modifiers(low)):
        return None                                               # "members not meeting UoS standards": the population's negation (D-096)
    cond = ask_condition(clause)
    if cond is not None and (cond.absent or cond.absent_clauses):
        return None                                               # "does not require X": the D-093 condition binds it
    extra: FrozenSet[str] = frozenset()
    m = _NEGATIVE_ASK.match(low)
    if m is not None:
        subject_text, rest = m.group("subject"), m.group("rest")
        kind = "value" if _NUMERIC_HEAD.match(rest) else "relation" if _RELATION_HEAD.match(rest) else "predicate"
    else:
        m = _NEGATIVE_NO_ASK.match(low)
        if m is not None and m.group("verb"):
            subject_text, rest, kind = m.group("subject"), m.group("rest"), "predicate"
            if m.group("verb").startswith(("get", "receive")):
                extra = frozenset({"entitl"})                     # "get no X": no entitlement to X
        elif m is not None:
            subject_text, rest, kind = "", m.group("exists") or m.group("none"), "existence"
        else:
            m = _NEGATIVE_ANY_ASK.match(low)
            if m is None:
                return None
            subject_text, rest, kind = "", m.group("none"), "existence"
    if _CONDITIONAL_SUBJECT.search(subject_text) or kind not in _NEGATIVE_KINDS:
        return None
    rest = _SUBORDINATOR.split(rest)[0]                           # the claim, not its condition: the condition binds itself (D-096)
    existential = _EXISTENTIAL_SCOPE and any(t in _cm.EXISTENTIAL_WORDS for t in _raw_tokens(subject_text))
    subject = frozenset(_lemma(t) for t in _raw_tokens(subject_text) if t not in _NEGATIVE_STOP and not (existential and t in _cm.EXISTENTIAL_WORDS))
    negated = extra | frozenset(_polarity_lemma(t) for t in [t for t in _raw_tokens(rest) if t not in _NEGATIVE_STOP][:6])
    if not negated - extra:
        return None
    return NegativeAsk(kind, subject, negated, _roots(clause), _scope_members(low), existential)


def _negation_stated(unit: str, ask: NegativeAsk) -> bool:
    """Some clause of the unit names the asked subject and negates every negated word within the negator's own scope; the
    negation is no condition's ("if the employee has no unused vacation leave"), and its population and object are at least
    as broad as the asked ones (D-096)."""
    previous = ""
    for sentence in _unit_sentences(unit):
        for clause_text in [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]:
            if not ask.subject <= _polarity_lemmas(clause_text):
                continue
            antecedents = _antecedents(clause_text, previous)
            for m in _NEGATOR.finditer(clause_text):
                if _CONDITIONAL_FACTS and _in_antecedent(antecedents, m.start()):
                    continue                                      # a condition's negation states no fact (D-096)
                after = _NEGATED_COPULA.sub("", clause_text[m.end():], count=1)
                end = _SCOPE_END.search(after)
                span = after[:end.start()] if end else after
                tokens = [_polarity_lemma(t) for t in _raw_tokens(span) if t not in _NEGATIVE_STOP]
                tokens = tokens[:len(ask.negated) + _VALUE_SLACK] if ask.kind == "value" else tokens[:12]
                words = frozenset(tokens) | (frozenset({"entitl"}) if m.group("word") else frozenset())
                if ask.negated <= words and _negative_scope_ok(clause_text[:m.start()], span, ask):
                    return True
        previous = sentence
    return False


def negative_detail(clause: str) -> Optional[Detail]:
    """The asked negative proposition, stated by the unit as a negation of the same subject (D-095)."""
    ask = negative_ask(clause)
    if ask is None:
        return None

    def check(unit: str, _a=ask) -> bool:
        return _negation_stated(unit, _a)

    return Detail("negative", "the asked negative %s (stated as a negation of the asked subject)" % ask.kind, check)


# ---------------------------------------------------------------------------
# D-096 (Step 4.12-C9): claim compatibility
# ---------------------------------------------------------------------------
#: The HR-local semantic contracts read what a specific yes/no ask claims once (its subject, predicate, object, polarity,
#: conditions, population modifiers, value and qualifier) and what an evidence clause states once (its negations, its
#: conditional antecedents, the restrictions of its subject). A unit settles the ask only when they are compatible:
#:   scope       - a negative claim is settled only by a negation whose population and object are at least as broad;
#:   conditional - what an "if / when / unless" antecedent says is no stated fact: a negation inside one negates nothing,
#:                 and a claim stated only inside one is no fact;
#:   polarity    - an explicit negation of the asked predicate contradicts the affirmative ask, whatever the predicate;
#:   frames      - an asked condition or population modifier keeps its polarity, and an asked condition is the evidence's;
#:   action      - a role subject keeps its explicit verb, whatever the verb, and a label is never the actor;
#:   value       - a value asked with a qualifier is the stated value with that qualifier.
#: Uncertain compatibility leaves the clause partial. The six switches below are the seams of scope, conditional facts,
#: modifier polarity, condition identity, the generic verb fallback and qualifier identity; the other contracts live in one
#: reader each (negative existence, labels, predicate polarity, clausal conditions).
_SCOPE_BINDING = True
_CONDITIONAL_FACTS = True
_FRAME_POLARITY = True
_CONDITION_IDENTITY = True
_GENERIC_VERB_FALLBACK = True
_QUALIFIER_IDENTITY = True
#: Derivational endings stripped (first match) to compare word families: "approval" / "approve" / "approved", "submission" /
#: "submit", "notification" / "notify", "termination" / "terminate"; a root keeps at least four letters.
_ROOT_RULES: Tuple[Tuple[str, str], ...] = (("ification", "ify"), ("ission", "it"), ("ments", ""), ("ment", ""), ("ions", ""), ("ion", ""),
                                            ("als", ""), ("al", ""), ("ings", ""), ("ing", ""), ("ied", "y"), ("ies", "y"), ("ed", ""),
                                            ("es", ""), ("s", ""), ("e", ""))
#: Two roots name one family when equal, or when the shorter, of at least six letters, begins the longer ("recommend" /
#: "recommendat", "exception" / "except").
_ROOT_PREFIX = 6
#: Words that carry no content of a claim, a condition or a modifier.
_CLAIM_STOP: FrozenSet[str] = _CONDITION_STOP | frozenset(
    "i me my mine we us our you your he him his she her hers they them their it its this that these those who whom whose which what "
    "before after during until since within upon from into onto over under by of for with without about against between among through "
    "via per neither either nor not no never none nothing also only just still yet already ever than then so such do does did done "
    "be been being am is are was were will would shall should can could may might must there".split())
#: An antecedent made only of these words refers back ("if neither occurs", "if this happens"): its content is the previous
#: sentence's.
_ANAPHOR_WORDS: FrozenSet[str] = frozenset("neither either occur occurs occurred happen happens happened this that so not otherwise case event "
                                           "these those".split())
#: A negator that negates what follows it in evidence; a quantity bound ("not exceeding", "no more than", "no later than"),
#: "not only" and "not limited to" negate nothing.
_FACT_NEGATOR = re.compile(r"(?:\b(?:not|never|no|none|neither|nor|cannot)\b|n['’]t\b)(?!\s+(?:only|merely|just|simply|(?:more|less|later|"
                           r"earlier|fewer|longer|sooner)\s+than|exceed\w*|limited\s+to)\b)", re.I)
#: The openers of a conditional antecedent.
_SUBORDINATOR = re.compile(r"\b(?:only\s+if|even\s+if|if|unless|whenever|when|once|in\s+case(?:\s+of)?|in\s+the\s+event\s+(?:that|of)|"
                           r"provided\s+that|as\s+long\s+as)\b", re.I)


def _root(token: str) -> str:
    """The family root of a word ("approval", "approve", "approved" -> "approv"; "submission" -> "submit")."""
    word = _stem(token.lower())
    for suffix, repl in _ROOT_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(repl) >= 4:
            word = word[:-len(suffix)] + repl
            break
    if len(word) >= 5 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]                                          # "submitted" -> "submitt" -> "submit"
    return word


def _same_root(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= _ROOT_PREFIX and (a.startswith(b) or b.startswith(a)))


def _roots(text: str, stop: FrozenSet[str] = _CLAIM_STOP) -> FrozenSet[str]:
    return frozenset(_root(t) for t in _raw_tokens(text) if t not in stop)


def _overlap(asked: FrozenSet[str], stated: FrozenSet[str]) -> int:
    return sum(1 for a in asked if any(_same_root(a, s) for s in stated))


def _covered(asked: FrozenSet[str], stated: FrozenSet[str]) -> bool:
    return _overlap(asked, stated) == len(asked)


@dataclass(frozen=True)
class Antecedent:
    """A conditional antecedent of an evidence clause: its character span, its polarity and its content roots (D-096)."""
    start: int
    end: int
    positive: bool
    roots: FrozenSet[str]


#: A main-clause marker: the comma before it ends a sentence-initial antecedent ("If the Central Committee, Deans Council, or
#: Board of Trustees does not approve a promotion, the Chancellor shall notify ...").
_MAIN_MARKER = re.compile(r"\b(?:shall|will|may|must|should|can|could|would|might|is|are|was|were)\b", re.I)
_SENTENCE_OPEN = re.compile(r"^[\s•▪●�\-–(\"'\d.a-z)]{0,6}$", re.I)


def _antecedent_end(text: str, start: int, initial: bool) -> int:
    """The end of an antecedent body starting at ``start``: for a sentence-initial antecedent the comma before the main
    clause, else the next comma, alternative ("... if a full professor or by another professor ...") or subordinator."""
    rest = text[start:]
    if initial:
        commas = [c.start() for c in re.finditer(r",", rest)]
        for i, c in enumerate(commas):
            stop = commas[i + 1] if i + 1 < len(commas) else len(rest)
            if _MAIN_MARKER.search(rest[c + 1:stop]):
                return start + c
    nxt = _SUBORDINATOR.search(rest)
    cut = re.search(r"[,;:]|\.(?!\d)|\s+or\s+", rest)
    ends = [x for x in (cut.start() if cut else None, nxt.start() if nxt else None) if x is not None]
    return start + (min(ends) if ends else len(rest))


def _antecedents(text: str, previous: str = "") -> List[Antecedent]:
    """The conditional antecedents of a clause: "if / when / unless / once / in case / provided that ...". An anaphoric
    antecedent ("if neither occurs", "if this happens") takes its content from the previous sentence; a negator inside it
    negates it; "unless" inverts it."""
    out: List[Antecedent] = []
    for m in _SUBORDINATOR.finditer(text):
        initial = bool(_SENTENCE_OPEN.match(text[:m.start()]))
        end = _antecedent_end(text, m.end(), initial)
        body = text[m.end():end]
        negated = bool(_FACT_NEGATOR.search(body)) or bool(re.search(r"\bwithout\b", body, re.I))
        positive = negated if m.group(0).lower() == "unless" else not negated
        words = [t for t in _raw_tokens(body) if t not in _CLAIM_STOP]
        if all(w in _ANAPHOR_WORDS for w in words):
            roots = _roots(previous)                              # "If neither occurs": what the previous sentence states
        else:
            roots = frozenset(_root(w) for w in words if w not in _ANAPHOR_WORDS)
        out.append(Antecedent(m.start(), end, positive, roots))
    return out


def _consequent(text: str, antecedents: Sequence[Antecedent], a: Antecedent) -> str:
    """What an antecedent conditions: the main clause after a sentence-initial antecedent, else the text before it back to
    the previous antecedent."""
    if _SENTENCE_OPEN.match(text[:a.start]):
        return text[a.end:]
    floor = max([b.end for b in antecedents if b.end <= a.start] or [0])
    return text[floor:a.start]


def _in_antecedent(antecedents: Sequence[Antecedent], position: int) -> bool:
    return any(a.start <= position < a.end for a in antecedents)


def _unit_sentences(unit: str) -> List[str]:
    return [s for s in _SENTENCE_SPLIT.split(cue_text(unit).replace("|", " ")) if s.strip()]


#: A copula or auxiliary chain right after a negator ("shall not be permitted to transfer").
_CHAIN_START = re.compile(r"^\s*(?:(?:be|been|being|have|has|had|be\s+able\s+to|able\s+to)\s+)+", re.I)


def _governed_text(after: str) -> str:
    """The text a negator governs: past a copula or auxiliary chain, up to its clause break or a new verb phrase (the D-094
    scope end)."""
    after = _CHAIN_START.sub("", after, count=1)
    end = _SCOPE_END.search(after)
    return after[:end.start()] if end else after


# ---- frames: conditions and population modifiers of the ask ------------------------------------------------------------
#: A wh-question opened by "when" asks a time, not a condition ("When is the next payroll date?").
_WH_AUX = re.compile(r"^\s*(?:is|are|was|were|do|does|did|can|could|will|would|must|should|shall|may|has|have)\b", re.I)


def _clausal_conditions(low: str) -> List[Tuple[str, FrozenSet[str], bool]]:
    """The conditions an ask states as clauses, with their polarity (D-096): "if the Chancellor does not approve", "when the
    Chancellor has not approved it", "if there are no special circumstances", "unless X" (absent); "if the faculty member
    receives a termination notification" (given). A leave state and a pay status bind themselves (D-086, D-093)."""
    out: List[Tuple[str, FrozenSet[str], bool]] = []
    for m in _SUBORDINATOR.finditer(low):
        if m.start() == 0 and m.group(0).lower() == "when" and _WH_AUX.match(low[m.end():]):
            continue
        tail = low[m.end():]
        cut = re.search(r"[,;?]|\s+(?:and|but|or)\s+(?=(?:can|could|may|must|should|will|would|is|are|does|do)\b)", tail)
        text = (tail[:cut.start()] if cut else tail).strip()
        if not text or _leave_states(text) or _PAY_CONDITION.match(text):
            continue
        negated = bool(_FACT_NEGATOR.search(text)) or bool(re.search(r"\bwithout\b", text))
        positive = negated if m.group(0).lower() == "unless" else not negated
        words = frozenset(_stem(t) for t in _raw_tokens(text) if t not in _CLAIM_STOP)
        if words:
            out.append((text, words, positive))
    return out


#: Words that end a population modifier: the ask's main predicate or a main-clause auxiliary.
_MODIFIER_END = re.compile(r"\b(?:entitled|eligible|qualified|qualify|allowed|permitted|required|based|involved|included|excluded|part\s+of|"
                           r"subject\s+to|responsible|expected|considered|given|provided|awarded|granted|offered|paid|appointed|notified|"
                           r"treated|exempt|get|gets|receive|receives|be|can|could|may|must|should|shall|will|would|is|are)\b", re.I)
_PERSON_NOUN = r"(?:faculty|members?|employees?|staff|applicants?|candidates?|professors?|lecturers?|instructors?|appointees?)"
_RELATIVE_MODIFIER = re.compile(r"(?:\b(?:who|which)|\b" + _PERSON_NOUN + r"\s+that)\s+(?:(?:is|are|was|were|do|does|did|have|has|had|can|could|"
                                r"will|would)\s+)?(?P<neg>(?:not|never|(?:is|are|was|were|do|does|did|have|has|had|can|could|will|would|wo|ca)n['’]t)\s+)?"
                                r"(?:been\s+|being\s+|yet\s+)?(?P<body>[a-z][\w'-]*(?:\s+[\w'-]+){0,6})", re.I)
_PARTICIPLE_MODIFIER = re.compile(r"\b" + _PERSON_NOUN + r"\s+(?P<neg>not\s+)?(?P<body>(?:[a-z]+ing|[a-z]{3,}ed)\b(?:\s+[\w'-]+){1,5})", re.I)
_WITH_MODIFIER = re.compile(r"\b" + _PERSON_NOUN + r"\s+with\s+(?!no\b|out\b)(?P<neg>)(?P<body>[\w'-]+(?:\s+[\w'-]+){0,6})", re.I)


def _subject_modifiers(low: str) -> List[Tuple[FrozenSet[str], bool]]:
    """The modifiers an ask attaches to its population, with their polarity (D-096): a relative clause ("faculty who are not
    terminated during probation"), a participle ("faculty members meeting UoS standards", "faculty not engaged in clinical
    activities"), a "with" phrase ("members with a prior appointment"). A leave state and a pay status bind themselves."""
    out: List[Tuple[FrozenSet[str], bool]] = []
    for pattern in (_RELATIVE_MODIFIER, _PARTICIPLE_MODIFIER, _WITH_MODIFIER):
        for m in pattern.finditer(low):
            body = m.group("body")
            if pattern is _PARTICIPLE_MODIFIER and _MODIFIER_END.match(body):
                continue                                          # "faculty not entitled to ...": the predicate, no modifier
            end = _MODIFIER_END.search(body, 1)
            body = body[:end.start()] if end else body
            if not body.strip() or _leave_states(body) or _PAY_CONDITION.match(body):
                continue
            words = frozenset(_stem(t) for t in _raw_tokens(body) if t not in _CLAIM_STOP)
            if words and not any(words == w for w, _p in out):
                out.append((words, not m.group("neg")))
    return out


def _modifier_negated(unit: str, words: FrozenSet[str]) -> bool:
    """The unit negates a population modifier the ask states as given (D-096): "newly appointed faculty members not meeting
    UoS standards", "with no prior appointment in an academic setting"."""
    roots = frozenset(_root(w) for w in words)
    need = max(1, -(-len(roots) * 3 // 5))
    for sentence in _unit_sentences(unit):
        for m in _FACT_NEGATOR.finditer(sentence):
            if _overlap(roots, _roots(_governed_text(sentence[m.end():]), frozenset())) >= need:
                return True
    return False


def _condition_matched(unit: str, words: FrozenSet[str], claim: FrozenSet[str] = frozenset(), positive: bool = True) -> bool:
    """A condition the ask states, given (``positive``) or absent, is compatible with the unit (D-096). An antecedent that
    conditions the asked claim (its consequent carries the claim) and names the condition must state it with the asked
    polarity: "If neither occurs" after "receive notification of ... termination" is not "if the faculty member receives a
    termination notification", "if a full professor" is not "if the dean is not a full professor". A given condition may
    also be stated outside the antecedents ("upon approval") and is contradicted by a negator governing it; a claim stated
    without conditions holds under any given condition. An absent condition needs its absence stated: by such an antecedent
    or by a negator governing it (D-093)."""
    roots = frozenset(_root(w) for w in words)
    distinct = frozenset(r for r in roots if r not in _SCOPE_GENERIC and not any(_same_root(r, c) for c in claim)) or roots
    need = max(1, -(-len(distinct) // 2))
    claim_need = max(1, -(-len(claim) // 2))
    previous = ""
    conditional, matched, opposed, outside_matched = False, False, False, False
    for sentence in _unit_sentences(unit):
        antecedents = _antecedents(sentence, previous)
        for a in antecedents:
            if claim and _overlap(claim, _roots(_consequent(sentence, antecedents, a), frozenset())) < claim_need:
                continue                                          # this condition governs another statement of the unit
            conditional = True
            if _overlap(distinct, a.roots) >= need:
                if a.positive == positive:
                    matched = True
                else:
                    opposed = True                                # the evidence conditions the claim on the opposite
        outside, last = [], 0
        for a in sorted(antecedents, key=lambda x: x.start):
            outside.append(sentence[last:a.start])
            last = max(last, a.end)
        outside.append(sentence[last:])
        for m in _FACT_NEGATOR.finditer(sentence):
            if (not _in_antecedent(antecedents, m.start())
                    and _overlap(distinct, _roots(_governed_text(sentence[m.end():]), frozenset())) >= max(need, min(2, len(distinct)))):
                if positive:
                    return False                                  # "... is not referred" against "if it is referred"
                outside_matched = True                            # the absence stated outright
        if positive and _overlap(distinct, _roots(" ".join(outside), frozenset())) >= need:
            outside_matched = True
        previous = sentence
    if opposed and not matched:
        return False                                              # only a same-polarity antecedent outweighs an opposed one
    if positive:
        return matched or outside_matched or not conditional
    return matched or outside_matched or _absence_stated(unit, words, antecedents=False)


# ---- scope of a negative claim --------------------------------------------------------------------------------------------
#: Words of a population or a place that restrict nothing in a scope comparison.
_SCOPE_GENERIC: FrozenSet[str] = frozenset(_root(w) for w in "faculty faculties member members staff employee employees personnel individual "
                                                             "individuals uos university sharjah academic".split())
#: Nouns ending in -ing that are no participle.
_ING_NOUNS: FrozenSet[str] = frozenset("housing training teaching funding building hearing marketing advising nursing engineering accounting "
                                       "meeting rating ranking setting".split())
_RESTRICTION_OPENERS: FrozenSet[str] = frozenset("who whom whose which that where in on at with without during within under for from not".split())
#: A noun whose "of" phrase is its complement, not a restriction ("employment of a relative", "termination of contracts").
_COMPLEMENT_HEAD = re.compile(r"(?:ment|tion|sion|ance|ence|al|ure)s?$")
_STAGE_SCOPE = re.compile(r"\bnew\b|\bnewly\b|\bprobation\w*|\bfirst\s+year\b", re.I)
#: Words that end a modifier run before a negated word ("based on | a single year's performance": "on" ends it).
_MODIFIER_BREAK: FrozenSet[str] = frozenset("to of on upon in for by with as such and or from at into than be been being is are was were the an "
                                            "any its their his her this that these those".split())
#: Light verb forms that restrict nothing ("for making a complaint in good faith").
_LIGHT_GERUNDS: FrozenSet[str] = frozenset("making taking doing being having getting giving make take do have get give".split())
#: Prepositions and relative pronouns that open a restriction after a negated object.
_POST_RESTRICTORS: FrozenSet[str] = frozenset("for in on at with within during under from by of who which that where whose".split())


def _scope_members(text: str) -> FrozenSet[str]:
    """The categories a phrase names: appointment basis, track, administrative role, rank, the new-faculty stage."""
    low = text.lower()
    members = {axis + ":" + m for axis in ("time", "track", "admin", "rank") for m in _axis_members(low, axis)}
    if _STAGE_SCOPE.search(low):
        members.add("stage:new")
    return frozenset(members)


def _subject_restrictions(subject_text: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """The categories and the restriction roots of an evidence clause's subject phrase (D-096): what follows its head (a
    relative clause, a participle, a prepositional phrase) narrows the population."""
    tokens = _raw_tokens(subject_text)
    start = None
    for i, t in enumerate(tokens):
        if i and (t in _RESTRICTION_OPENERS or (t == "of" and not _COMPLEMENT_HEAD.search(tokens[i - 1]))
                  or (t.endswith("ing") and len(t) > 5 and t not in _ING_NOUNS)):
            start = i
            break
    head = tokens if start is None else tokens[:start]
    tail = [] if start is None else tokens[start:]
    head_roots = frozenset(_root(t) for t in head)
    restriction = frozenset(r for r in (_root(t) for t in tail if t not in _CLAIM_STOP) if r not in head_roots and r not in _SCOPE_GENERIC)
    return _scope_members(subject_text), restriction


def _span_narrower(span: str, negated: FrozenSet[str], ask_roots: FrozenSet[str]) -> bool:
    """The negation covers a narrower object than the ask (D-096): a modifier right before a negated word ("not based on a
    single year's performance") or a restriction right after the last one ("not involved in reviewing applications for full
    professor rank") that the ask does not name."""
    tokens = _raw_tokens(span)
    hits = [i for i, t in enumerate(tokens) if _polarity_lemma(t) in negated or _lemma(t) in negated]
    if not hits:
        return False
    extra: List[str] = []
    for i in hits:
        k = i - 1
        while k >= 0 and tokens[k] not in _MODIFIER_BREAK:
            if k not in hits:
                extra.append(tokens[k])
            k -= 1
    last = hits[-1]
    if last + 1 < len(tokens) and tokens[last + 1] in _POST_RESTRICTORS:
        for t in tokens[last + 2:]:
            if t in ("such", "and", "or", "including", "like", "as"):
                break
            extra.append(t)
    for t in extra:
        if t in _CLAIM_STOP or t in _LIGHT_GERUNDS:
            continue
        r = _root(t)
        if r not in _SCOPE_GENERIC and not any(_same_root(r, a) for a in ask_roots):
            return True
    return False


def _negative_scope_ok(subject_text: str, span: str, ask: "NegativeAsk") -> bool:
    """D-096 scope: the evidence negation's population and object are at least as broad as the asked ones."""
    if not _SCOPE_BINDING:
        return True
    if ask.existential and _EXISTENTIAL_SCOPE:
        return not _span_narrower(span, ask.negated, ask.roots)  # "some faculty members ... not": a narrower population settles it (D-097)
    members, restriction = _subject_restrictions(subject_text)
    if members and not (members & ask.members):
        return False                                              # "part-time and visiting faculty" settle nothing about all faculty
    if restriction and not any(_same_root(r, a) for r in restriction for a in ask.roots):
        return False                                              # "faculty members who resign during probation" are not all faculty
    return not _span_narrower(span, ask.negated, ask.roots)


# ---- the structure of a role or label ask (explicit verb, actor, label) -----------------------------------------------------
#: Nouns that keep the role before them as the actor ("the HR office", "the Chancellor's office").
_ORG_HEADS: FrozenSet[str] = frozenset("office offices department departments unit units team teams staff personnel division divisions "
                                       "directorate".split())
#: A comma-delimited insertion naming a source, a consultation or a frame, never a subject ("the Dean, under the HR policy,").
_PARENTHETICAL = re.compile(r",\s*(?:under|according\s+to|per|as\s+per|in\s+line\s+with|in\s+accordance\s+with|in\s+consultation\s+with|"
                            r"in\s+coordination\s+with|together\s+with|along\s+with|pursuant\s+to)\b[^,.;?]*,", re.I)
_STRUCTURE_START = re.compile(r"^\s*(?:does|do|did|can|could|may|might|must|should|shall|will|would)\s+", re.I)
#: Words that never stand for the explicit verb of a role subject.
_NOT_A_VERB: FrozenSet[str] = frozenset("the a an this that these those its their his her our your my any each every all some no not never also "
                                        "only and or of for to in on at by with without from under over per as than then there here it they "
                                        "he she we who whom whose which what when where why how be been being is are was were have has had "
                                        "do does did will would shall should can could may might must ever still always including excluding "
                                        "except".split())


#: An auxiliary or copula right after a word makes the word part of the subject ("May grievances against Disciplinary
#: Council decisions be directed ...?").
_AUXILIARIES: FrozenSet[str] = frozenset("be been being is are was were am have has had do does did will would shall should can could may might "
                                         "must".split())


def _bare_verb(word: str) -> bool:
    """After "do" or a modal the explicit verb is bare: never a plural noun ("decisions"); a base form may end in -ss / -us."""
    return not word.endswith("s") or word.endswith(("ss", "us", "is"))


@dataclass(frozen=True)
class AskStructure:
    """An "auxiliary + subject + verb + rest" ask read by its role words (D-096): the subject words, the role the subject
    names (None for a label or a subject whose head is no role), its label noun (None for an actor), the explicit verb and
    the rest of the ask."""
    subject: Tuple[str, ...]
    actor: Optional[str]
    label: Optional[str]
    verb: str
    rest: str


def ask_structure(clause: str) -> Optional[AskStructure]:
    """The structure of an auxiliary-led ask whose subject names a role or a label, else None."""
    low = _PARENTHETICAL.sub(" ", cue_text(clause.lower()).strip()).strip().rstrip("?.!").strip()
    m = _STRUCTURE_START.match(low)
    return _declarative(low[m.end():]) if m else None


def _declarative(body: str, depth: int = 0) -> Optional[AskStructure]:
    """Subject, explicit verb and rest of "subject + verb + rest": the verb is the first word after a phrase ending in a
    role, a label or an office that is a known HR verb, or, with the generic fallback, any other word that is no role, label,
    function word, preposition or adverb and is not followed by a known verb. After "do" or a modal the verb is bare and no
    auxiliary follows it; a "be" makes the ask passive or copular, with no explicit verb ("May grievances against
    Disciplinary Council decisions be directed ...?"). A reporting verb ("say", "state") reports the clause after it (depth 1)."""
    words = [w.strip(",;:'\"()") for w in body.split()]
    words = [w for w in words if w]
    if len(words) < 2:
        return None
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    role_of: List[Optional[str]] = [None] * len(words)
    role_end = [False] * len(words)
    for a, b, role in ASK_GATE.lexicon.spans(" ".join(words)):
        inside = [i for i, st in enumerate(starts) if st >= a and st + len(words[i]) <= b]
        for i in inside:
            role_of[i] = role
        if inside:
            role_end[inside[-1]] = True
    verb_at = None
    for i in range(1, min(len(words), 14)):
        word, prev = words[i], words[i - 1]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if word in ("be", "been", "being"):
            break                                                 # a passive or copular ask names no explicit verb of a role subject
        if not any(role_of[j] is not None or words[j] in _LABEL_NOUNS for j in range(i)):
            continue
        np_end = role_end[i - 1] or prev in _LABEL_NOUNS or prev in _ORG_HEADS
        verbal = nxt not in _ASK_VERBS and nxt not in _AUXILIARIES and (depth > 0 or _bare_verb(word))
        known = word in _ASK_VERBS or bool(re.match(_ACTION_STEMS + r"(?:e|es|s|ed|d|ing)?$", word))
        if known and verbal and (np_end or (role_of[i] is None and word not in _LABEL_NOUNS)):
            verb_at = i
            break
        if (_GENERIC_VERB_FALLBACK and np_end and verbal and role_of[i] is None and word not in _LABEL_NOUNS and word not in _ORG_HEADS
                and word not in _NOT_A_VERB and word not in _PLAIN_PREPS and word.isalpha() and not word.endswith("ly")):
            verb_at = i
            break
    if verb_at is None:
        return None
    subject, verb, rest = tuple(words[:verb_at]), words[verb_at], " ".join(words[verb_at + 1:])
    if verb in _PHRASE_BREAK and depth == 0:
        inner = _declarative(re.sub(r"^\s*that\s+", "", rest), depth + 1)
        if inner is not None:
            return inner                                          # "Does the Faculty Handbook say HR provides ...": HR provides
    label = next((w for w in reversed(subject) if w in _LABEL_NOUNS), None)
    head_is_role = role_end[verb_at - 1] or (subject[-1] in _ORG_HEADS and any(r is not None for r in role_of[:verb_at]))
    actor = ASK_GATE.lexicon.head_role(" ".join(subject)) if label is None and head_is_role else None
    return AskStructure(subject, actor, label, verb, rest)


# ---- the asked predicate outside the D-094 list, and its negation in evidence ---------------------------------------------
#: Light verbs: the asked predicate is their object ("Does UoS have a statute of limitations ...?").
_LIGHT_ASK_VERBS: FrozenSet[str] = frozenset("have has had get gets got do does did make makes made take takes took give gives gave hold holds "
                                             "keep keeps be".split())
#: Copular predicates of an "is X ...?" ask.
_COPULAR_PREDICATE = re.compile(r"^(?:[a-z]{3,}(?:able|ible|ive|ful|ous|ory)|void|valid|invalid|final|open|closed|free|mandatory|optional|"
                                r"available|possible|necessary|payable|liable|exempt|applicable|effective|binding|conclusive|compulsory|"
                                r"prohibited|forbidden|permissible)$")
_NOUN_LIKE = re.compile(r"(?:tion|sion|ment|ance|ence|ity|ness|ship|ism|ist|ary|al|ic|age|ure|er|or)s?$")
#: Participles ending in -en (other -en words, "between", "often", "children", are none).
_EN_PARTICIPLES: FrozenSet[str] = frozenset("given taken written chosen driven forbidden proven shown seen known broken spoken drawn eaten fallen "
                                            "hidden stolen sworn".split())
#: Prepositions that never stand for the predicate of a plain ask.
_PLAIN_PREPS: FrozenSet[str] = _UNIVERSAL_PREPS | frozenset("between among through about against within upon into onto across beyond despite via "
                                                             "toward towards throughout".split())


def _participle(word: str) -> bool:
    return (word.endswith("ed") and len(word) >= 5) or word in _EN_PARTICIPLES


#: The claim of an ask ends where its condition or a concessive clause begins.
_CLAIM_CUT = re.compile(r"\b(?:if|unless|when|whenever|once|while|in\s+case|provided\s+that)\b", re.I)


@dataclass(frozen=True)
class ClaimPredicate:
    """The predicate an affirmative yes/no ask claims outside the D-094 list (D-096): subject, predicate and object roots."""
    subject: FrozenSet[str]
    predicate: FrozenSet[str]
    obj: FrozenSet[str]


def _plain_predicate(main: str) -> Optional[Tuple[List[str], str, str]]:
    """Subject, verb and rest of a yes/no ask whose subject names no role: "Should termination be the first resort?",
    "Is the investigation case rendered void ...?", "Will disciplinary action impede ...?"."""
    m = re.match(r"^\s*(?P<aux>is|are|was|were|am|can|could|may|might|must|should|shall|will|would|does|do|did|has|have)\s+(?P<body>.+)$", main)
    if m is None:
        return None
    words = [w.strip(",;:'\"()") for w in m.group("body").split()]
    words = [w for w in words if w]
    role_words = set()
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    for a, b, _role in ASK_GATE.lexicon.spans(" ".join(words)):
        role_words |= {i for i, st in enumerate(starts) if st >= a and st + len(words[i]) <= b}
    copular = m.group("aux") in ("is", "are", "was", "were", "am")
    k = next((i for i, w in enumerate(words) if i and w in ("be", "been")), None)
    if not copular and k is not None:                             # "can ... be granted", "have all sick leave days been utilized"
        nxt = words[k + 1] if k + 1 < len(words) else ""
        if _participle(nxt) or _COPULAR_PREDICATE.match(nxt):
            return words[:k], nxt, " ".join(words[k + 2:])
        return words[:k], "be", " ".join(words[k + 1:])
    for i in range(1, min(len(words), 9)):
        w = words[i]
        if i in role_words or w in _PLAIN_PREPS or (w in _NOT_A_VERB and w not in _LIGHT_ASK_VERBS):
            continue
        if copular:
            if _participle(w) or _COPULAR_PREDICATE.match(w):
                return words[:i], w, " ".join(words[i + 1:])
        elif w in _LIGHT_ASK_VERBS or w in _ASK_VERBS or (w.isalpha() and len(w) >= 3 and not _NOUN_LIKE.search(w) and w not in _NOT_A_VERB):
            return words[:i], w, " ".join(words[i + 1:])
    return None


def affirmative_claim(clause: str) -> Optional[ClaimPredicate]:
    """The subject, predicate and object an affirmative yes/no ask claims, whatever its predicate (D-096), else None: no
    negative claim, no negated main clause."""
    low = cue_text(clause.lower()).strip().rstrip("?.! ")
    if not _YES_NO.match(low) or negative_ask(clause) is not None:
        return None
    cut = _CLAIM_CUT.search(low)
    main = low[:cut.start()] if cut else low
    if _ASK_NEGATED.search(main):
        return None                                               # a negated main clause is no affirmative claim
    st = ask_structure(main)
    if st is not None:
        subject_words, verb, rest = list(st.subject), st.verb, st.rest
    else:
        parsed = _plain_predicate(main)
        if parsed is None:
            return None
        subject_words, verb, rest = parsed
    rest_words = [t for t in _raw_tokens(rest) if t not in _CLAIM_STOP]
    if verb in _LIGHT_ASK_VERBS:
        predicate, obj = frozenset(_root(t) for t in rest_words[:2]), frozenset(_root(t) for t in rest_words[2:8])
    else:
        predicate, obj = frozenset({_root(verb)}), frozenset(_root(t) for t in rest_words[:6])
    subject = frozenset(_root(t) for t in _raw_tokens(" ".join(subject_words)) if t not in _NOT_A_VERB)
    return ClaimPredicate(subject, predicate, obj) if predicate else None


def claim_predicate(clause: str) -> Optional[ClaimPredicate]:
    """The claim of an affirmative yes/no ask whose predicate no D-094 predicate reads (the generic polarity veto), else None."""
    return None if predicate_ask(clause) is not None else affirmative_claim(clause)


def _claim_negated(unit: str, cp: ClaimPredicate) -> bool:
    """The unit negates the asked predicate for the asked subject: a negator outside a conditional antecedent governing the
    predicate, with the asked object when one is asked (D-096)."""
    previous = ""
    for sentence in _unit_sentences(unit):
        for clause_text in [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]:
            if cp.subject and not _covered(cp.subject, _roots(clause_text, frozenset())):
                continue
            antecedents = _antecedents(clause_text, previous)
            for m in _FACT_NEGATOR.finditer(clause_text):
                if _CONDITIONAL_FACTS and _in_antecedent(antecedents, m.start()):
                    continue
                span = _roots(_governed_text(clause_text[m.end():]), frozenset())
                if not _overlap(cp.predicate, span):
                    continue
                if cp.obj and not _overlap(cp.obj, span | _roots(clause_text[m.end():], frozenset())):
                    continue
                return True
        previous = sentence
    return False


def _claim_conditional_only(unit: str, cp: ClaimPredicate) -> bool:
    """The unit states the asked claim only inside conditional antecedents (D-096): an antecedent states its subject and
    predicate ("If the Deans Council approves a promotion, ...", "If all sick leave days have been utilized, ..."), and no
    sentence states them together outside its antecedents."""
    previous, inside, outside = "", False, False
    for sentence in _unit_sentences(unit):
        antecedents = _antecedents(sentence, previous)
        rest, last = [], 0
        for a in sorted(antecedents, key=lambda x: x.start):
            body = _roots(sentence[a.start:a.end], frozenset())
            if _overlap(cp.predicate, body) and _covered(cp.subject, body):
                inside = True
            rest.append(sentence[last:a.start])
            last = max(last, a.end)
        rest.append(sentence[last:])
        stated = _roots(" ".join(rest), frozenset())
        if _overlap(cp.predicate, stated) and _covered(cp.subject, stated):
            outside = True
        previous = sentence
    return inside and not outside


def polarity_detail(clause: str) -> Optional[Detail]:
    """The asked predicate, outside the D-094 list, not negated for the asked subject (D-096)."""
    cp = claim_predicate(clause)
    if cp is None:
        return None

    def check(unit: str, _cp=cp) -> bool:
        return not _claim_negated(unit, _cp)

    return Detail("polarity", "the asked predicate (%s) stated, not negated, of the asked subject" % ", ".join(sorted(cp.predicate)), check)


def assertion_detail(clause: str) -> Optional[Detail]:
    """The asked claim, whatever its predicate, stated as a fact and not only inside an "if / when / unless" antecedent (D-096)."""
    cp = affirmative_claim(clause)
    if cp is None:
        return None

    def check(unit: str, _cp=cp) -> bool:
        return not (_CONDITIONAL_FACTS and _claim_conditional_only(unit, _cp))

    return Detail("assertion", "the asked claim (%s) stated as a fact, not only inside a condition" % ", ".join(sorted(cp.subject | cp.predicate)), check)


# ---------------------------------------------------------------------------
# D-097 (Step 4.12-C10): the structured claim model
# ---------------------------------------------------------------------------
#: One claim per specific yes/no question and one statement per sentence of the returned quote (``claim_model``). The
#: dimensions the C1-C9 readers never represented are read once and compared per dimension: the modality bound to the
#: asked action, the breadth of the asked population (universal, generic, existential), exclusivity and its focus, and the
#: role of a condition (sufficient, necessary, exception, trigger, concessive). A recognised operator no reader verifies
#: (a proportional quantifier, an ambiguous "just", an unqualified comparative, an event order) fails closed. Support stays
#: local to one unit; a bound contradiction or exception anywhere in the returned quote vetoes it (the window veto).
#: Each enablement has one switch; the binding switches are the seams of their rules.
_C10_MODALITY = True
_C10_SCOPE = True
_C10_EXCLUSIVITY = True
_C10_CONDITION_ROLES = True
_C10_OPERATORS = True
_WINDOW_VETO = True
_MODAL_BINDING = True                 # the modal marker must govern the asked action (off: any marker in the unit counts)
_EXISTENTIAL_SCOPE = True             # "some ... not": a narrower negation settles an existential negative claim
_GENERIC_EXCEPTION_COMPATIBLE = True  # a stated exception leaves a generic rule standing (it vetoes universal claims only)
_QUANTIFIER_SLOT = True               # an object or time quantifier is no population quantifier
_VALUE_EXCLUSIVITY = True             # "only one year" is contradicted by an extension, a renewal or another value
_WINDOW_BINDING = True                # a window veto binds to the asked predicate, object and population
#: Person nouns of one family: a statement about "individuals appointed as Deans" is about the asked "Deans".
_PERSON_FAMILY: FrozenSet[str] = frozenset(_root(w) for w in "faculty faculties member members employee employees staff individual individuals "
                                                            "personnel appointee appointees recruit recruits candidate candidates applicant "
                                                            "applicants people person persons".split())
#: Light verbs of a restriction: "faculty members intending to resign" restrict by "resign", not by "intend"; "female
#: faculty employed on a full-time basis" by "full-time", not by "employed" or "basis".
_LIGHT_RESTRICTION_ROOTS: FrozenSet[str] = frozenset(_root(w) for w in "attending attend intending intend wishing wish planning plan seeking seek "
                                                                       "choosing choose deciding decide wanting want hoping hope appointed "
                                                                       "appoint serving serve employed employ hired hire working work basis".split())
#: Negators and contracted negators are polarity, never a restriction ("faculty who doesn't meet expectations").
_NEGATOR_ROOTS: FrozenSet[str] = frozenset(_root(w) for w in "not no never cannot doesn don didn isn aren wasn weren hasn haven hadn won wouldn "
                                                             "shouldn mustn couldn".split())
_POP_DETERMINERS: FrozenSet[str] = frozenset("the a an this that these those my your his her its our their".split())
#: Function words of a subject phrase: no restriction ("individuals appointed as Deans").
_POP_FUNCTION: FrozenSet[str] = frozenset("as of for to in on at by with from and or than both either neither each per via".split())
#: An inclusive phrase restricts nothing ("all faculty, including those with administrative assignments"); an exclusion
#: narrows the population ("all faculty excluding visiting faculty").
_INCLUSIVE_PHRASE = re.compile(r"\b(?:including|such\s+as|as\s+well\s+as|especially|particularly|in\s+particular)\b", re.I)
_EXCLUSION_PHRASE = re.compile(r"\b(?:excluding|except|other\s+than|apart\s+from|with\s+the\s+exception\s+of)\b", re.I)
#: A comma-enclosed inclusive phrase leaves the question before it is parsed.
_INCLUSIVE_ASIDE = re.compile(r",\s*(?:including|such\s+as|as\s+well\s+as|especially|particularly)\b[^,?]*,", re.I)
#: Words of a verb group before the governed predicate (stripped to find the subject noun phrase).
_VERB_GROUP_WORDS: FrozenSet[str] = frozenset("may might must shall should can could will would is are was were be been being has have had do does did "
                                              "not also only equally strongly highly then to required allowed permitted expected encouraged obliged "
                                              "able eligible entitled never no longer".split())
_SUBJECT_BREAK = re.compile(r"[,;:]|[.!?](?=\s)|\b(?:that|which|who|whom|whose|if|unless|when|whenever|once|while|whereas|although|because|but|however)\b",
                            re.I)
#: The entitlement family: a negation of any of them binds an asked get / receive / entitled / eligible ("not entitled to ...
#: annual leave" answers "do they get annual leave?").
_ENTITLEMENT_ROOTS: FrozenSet[str] = frozenset(_root(w) for w in "entitled eligible qualify qualified get gets receive receives obtain obtains "
                                                                  "given provided awarded".split())
_ENTITLEMENT_ASK = re.compile(r"\b(?:get|gets|receive|receives|obtain|obtains|entitled|eligible|qualif\w*)\b", re.I)
_EXCEPTION_MARKERS = re.compile(r"\b(?:except|excluding|other\s+than|with\s+the\s+exception\s+of|in\s+exceptional\s+cases?|exceptions?\s+(?:may|are|shall|can|is))\b",
                                re.I)
_VALUE_UNIT = re.compile(r"\b(?P<n>\d+(?:\.\d+)?)\s*-?\s*(?P<u>years?|months?|days?|weeks?|semesters?|times?|hours?)\b|\b(?P<w>once|twice)\b", re.I)


class _ClaimCheck:
    """A detail check of the structured claim model: ``__call__`` reads one local unit (support); ``window`` reads the
    whole returned quote (a bound veto). Shared code sees an ordinary callable."""

    def __init__(self, local: Callable[[str], bool], window: Optional[Callable[[str], "_cm.Compatibility"]] = None,
                 explain: Optional[Callable[[Optional[str]], str]] = None, ungated: Optional[Callable[[str], bool]] = None) -> None:
        self.local = local
        self._window = window
        self.windowed = window is not None
        self.explain = explain                                    # D-098: the composite's evidence-specific missing message
        self.ungated = ungated if ungated is not None else local  # D-098: the composite's own reading, without the other details

    def __call__(self, unit: str) -> bool:
        return self.local(unit)

    def window(self, quote: str) -> "_cm.Compatibility":
        return self._window(quote) if self._window is not None else _cm.Compatibility.COMPATIBLE


def _window_vetoed(details: Sequence[Detail], quote: str, covered: FrozenSet[str]) -> FrozenSet[str]:
    """D-097: a detail whose check reads the whole returned quote loses its coverage when the window contradicts it."""
    if not _WINDOW_VETO:
        return covered
    out = covered
    for d in details:
        check = d.check
        if isinstance(check, _ClaimCheck) and check.windowed and _detail_key(d) in out and check.window(quote) is _cm.Compatibility.CONTRADICTED:
            out = out - {_detail_key(d)}
    return out


#: A question led by a phrase and a comma ("Subject to Chancellor approval, can ...?").
_LEADING_PHRASE = re.compile(r"^\s*(?P<phrase>(?:subject\s+to|with|upon|following|after)\b[^,?]{1,80}),\s*(?P<rest>.+)$", re.I)
#: "once" after "only / just / than / at least / at most", or before "a / per / every / for / again / ..." or the end,
#: counts occurrences: no antecedent ("renewable only once", "apply only once for the award", "once a year").
_FREQUENCY_ONCE_BEFORE = re.compile(r"\b(?:only|just|than|least|most)\s+$", re.I)
_FREQUENCY_ONCE_AFTER = re.compile(r"\s*(?:$|[,.;:?!]|(?:a|an|per|every|each|for|again|more|or|and|in|during)\b)", re.I)


def _conditional(text: str, a: "Antecedent") -> bool:
    """An antecedent proper of the C10 readers: content, and no frequency "once"."""
    if not a.roots:
        return False
    if not re.match(r"once\b", text[a.start:a.start + 5], re.I):
        return True
    return not (_FREQUENCY_ONCE_BEFORE.search(text[max(0, a.start - 12):a.start]) or _FREQUENCY_ONCE_AFTER.match(text, a.start + 4))


def _main_question(low: str) -> str:
    """The question without its conditional antecedents: their operators are the conditions', not the claim's. An opener
    with no content or a frequency "once" ("renewable only once") is no antecedent."""
    chars = list(low)
    for a in _antecedents(low):
        if not _conditional(low, a):
            continue
        for i in range(a.start, min(a.end, len(chars))):
            chars[i] = " "
    return re.sub(r"\s+", " ", "".join(chars)).strip(" ,;:")


def _population(words: Sequence[str]) -> Optional["_cm.Population"]:
    """The population a subject phrase names: its head (a person noun), quantifier, categories and restrictions (the roots
    of a modifier before the head and of a restriction after it, light verbs and generic words excluded)."""
    toks = [w.lower().strip(",;:'\"()") for w in words]
    toks = [w for w in toks if w]
    if not toks:
        return None
    joined = " ".join(toks)
    excluded = bool(_EXCLUSION_PHRASE.search(joined))
    inclusive = _INCLUSIVE_PHRASE.search(joined)
    if inclusive is not None:
        toks = [w for w, _s, _e in _cm.tokens(joined[:inclusive.start()])] or toks
    text = " ".join(toks)
    label = any(w in _LABEL_NOUNS for w in toks)
    cut = len(toks)                                               # the head phrase ends where a restriction begins
    for i, t in enumerate(toks):
        if i and (t in _RESTRICTION_OPENERS or (t == "of" and not _COMPLEMENT_HEAD.search(toks[i - 1]))
                  or (t.endswith("ing") and len(t) > 5 and t not in _ING_NOUNS)
                  or (t.endswith("ed") and len(t) > 4 and _cm.POPULATION_HEADS.match(toks[i - 1]) and not _cm.POPULATION_HEADS.match(t)
                      and toks[i + 1:i + 2] != ["as"])):
            cut = i                                               # "faculty members appointed before 2020": a restriction
            break
    end = cut - 1
    while end > 0 and toks[end].endswith("ed") and len(toks[end]) > 4 and not _cm.POPULATION_HEADS.match(toks[end]):
        end -= 1                                                  # "faculty employed on a full-time basis": the head is "faculty"
    if end < 0 or label or not _cm.POPULATION_HEADS.match(toks[end]):
        return _cm.Population(head=frozenset(), quantifier=_cm.Quantifier.DEFINITE, is_label=label, text=text)
    h = end                                                       # "the Faculty Grievance Committee": the head is "committee"
    while h - 1 >= 0 and _cm.POPULATION_HEADS.match(toks[h - 1]):
        h -= 1
    run_start = h                                                 # the modifiers are the run right before the head
    while run_start > 0 and h - run_start < 4:
        t = toks[run_start - 1]
        if t in _POP_FUNCTION - {"and", "or"} or t in _POP_DETERMINERS or t[0].isdigit() or _cm.POPULATION_HEADS.match(t):
            break
        run_start -= 1
    pre = toks[:h]
    modifiers_before = toks[run_start:h]
    quantifier = _cm.Quantifier.GENERIC
    for w in pre:
        q = _cm.quantifier_of(w)
        if q is not _cm.Quantifier.GENERIC:
            quantifier = q
            break
    head_word = toks[end]
    singular = not head_word.endswith("s") and head_word not in ("faculty", "staff", "personnel")
    if quantifier is _cm.Quantifier.GENERIC and pre[:1] and pre[0] in _POP_DETERMINERS - {"a", "an"} and singular:
        quantifier = _cm.Quantifier.DEFINITE
    skip = _POP_DETERMINERS | _cm.UNIVERSAL_WORDS | _cm.EXISTENTIAL_WORDS | _cm.PROPORTIONAL_WORDS | {"only", "and", "or", "both", "just"}
    pre_roots = frozenset(_root(t) for w in modifiers_before if w not in skip and w not in _POP_FUNCTION for t in _raw_tokens(w)
                          if t not in _CLAIM_STOP and t not in _POP_FUNCTION)
    _members, post = _subject_restrictions(text)
    head = frozenset(_root(w) for w in toks[h:end + 1])
    restrictions = frozenset(r for r in (pre_roots | post) if r not in _SCOPE_GENERIC and r not in _LIGHT_RESTRICTION_ROOTS and r not in head
                             and r not in _POP_FUNCTION and r not in _NEGATOR_ROOTS)
    if excluded:
        restrictions = restrictions | {"exclud"}                  # "all faculty excluding visiting faculty" is narrower
    negated = frozenset(_root(w) for words_, positive in _subject_modifiers(text) if not positive for w in words_)
    return _cm.Population(head=head, categories=_scope_members(cue_text(text)), restrictions=restrictions, negated_restrictions=negated,
                          quantifier=quantifier, text=text)


def _statement_population(text: str, position: int) -> Optional["_cm.Population"]:
    """The population a statement's clause names before ``position`` (the asked predicate): its subject noun phrase. A
    relative clause after a person head belongs to it ("new faculty members who resign during probation")."""
    before = text[:position]
    cut = 0
    for m in _SUBJECT_BREAK.finditer(before):
        if m.group(0).lower() in _RELATIVE:
            prev = [w for w, _s, _e in _cm.tokens(before[:m.start()])]
            if prev and _cm.POPULATION_HEADS.match(prev[-1]):
                continue
        cut = m.end()
    words = [w for w, _s, _e in _cm.tokens(before[cut:])]
    while words and words[-1] in _VERB_GROUP_WORDS:
        words.pop()
    return _population(words[-10:]) if words else None


def _breadth(claim: "_cm.Claim", claim_roots: FrozenSet[str], stated: Optional["_cm.Population"]) -> "_cm.Breadth":
    """How the stated population compares with the asked one: an agentless or entity-subject rule narrows nothing; a
    population of another family is unrelated; a restriction or category the ask does not name narrows it."""
    asked = claim.subject
    if stated is None or not stated.head or stated.is_label or asked is None:
        return _cm.Breadth.SAME_OR_BROADER
    if asked.head and not (asked.head & stated.head) and not ((asked.head | stated.head) & _PERSON_FAMILY):
        return _cm.Breadth.UNRELATED
    extra = {r for r in stated.restrictions if not any(_same_root(r, c) for c in claim_roots)}
    if extra or (stated.categories - asked.categories):
        return _cm.Breadth.NARROWER
    return _cm.Breadth.SAME_OR_BROADER


def _anchor_positions(text: str, roots: FrozenSet[str]) -> List[Tuple[int, str]]:
    """The positions of the words of ``text`` in the family of ``roots``."""
    if not roots:
        return []
    out: List[Tuple[int, str]] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z'’-]*", text):
        w = m.group(0).lower()
        r = _root(w)
        if any(_same_root(r, x) for x in roots):
            out.append((m.start(), w))
    return out


#: Word endings of adjectives and nouns: never the bare verb after a modal ("-tive" and "-sive" adjectives, "-al" but
#: not "-eal", "-ary" after two letters; "give", "receive", "appeal" and "vary" are verbs).
_NOT_BARE_VERB = re.compile(r"(?:[st]ive|[^e]al|ous|ful|ic|[a-z]{2}ary|able|ible|ant|ent|less|tion|sion|ment|ance|ence|ity|ness|ship)$")
#: Relative pronouns and the prepositions beyond the plain-ask ones: each opens a phrase of the subject.
_RELATIVE: FrozenSet[str] = frozenset("who whom whose which that".split())
_EXTRA_PREPS: FrozenSet[str] = frozenset("during after before until since without under over above below behind beside besides regarding "
                                         "concerning following pending absent like unlike outside inside".split())
#: Words that never stand for the asked verb: function words, prepositions, coordinators and number words.
_NEVER_A_VERB: FrozenSet[str] = _NOT_A_VERB | _POP_FUNCTION | _PLAIN_PREPS | _EXTRA_PREPS | _cm.NUMBER_WORDS | frozenset({"nor", "none"})
_PRONOUNS: FrozenSet[str] = frozenset("i we you they he she".split())
#: Words that link a following word to the subject ("faculty with administrative assignments", "failure to comply",
#: "the Dean nor the Chancellor", "termination during probation", "a faculty member who resigns").
_SUBJECT_LINKS: FrozenSet[str] = _PLAIN_PREPS | _POP_FUNCTION | _EXTRA_PREPS | _RELATIVE | frozenset({"nor"})
#: A possessive or perfect "have" (or "get") after the subject is the asked verb ("do all categories have the same ...").
_HAVE_VERBS: FrozenSet[str] = frozenset("have has had get gets got".split())


def _known_verb(word: str) -> bool:
    """A known HR verb of the D-096 ask structure ("grant", "present", "waive")."""
    return word in _ASK_VERBS or bool(re.match(_ACTION_STEMS + r"(?:e|es|s|ed|d|ing)?$", word))


def _verb_like(word: str) -> bool:
    if not word or word in _NEVER_A_VERB or (word.endswith("ly") and len(word) > 4 and not word.endswith("ply")):
        return False                                              # "-ly" adverbs: "strictly"; "apply", "comply" stay verbs
    if word.endswith("ing") and len(word) > 5:
        return False                                              # "faculty members intending to resign ... provide"
    return _known_verb(word) or (word.isalpha() and len(word) >= 3 and not _NOT_BARE_VERB.search(word) and _bare_verb(word))


def _named_words(clause: str) -> FrozenSet[str]:
    """Words the ask writes capitalised, and never in lower case, after its first word: parts of a name ("the Faculty
    Grievance Committee", "the Benefits Chart"), never the asked verb."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", clause)[1:]
    upper = {w.lower() for w in words if w[0].isupper() and w != "I"}
    return frozenset(upper - {w for w in words if w[0].islower()})


def _cut_in_name(main: str, words: Sequence[str], named: FrozenSet[str]) -> bool:
    """A parse subject cut inside a capitalised name: the word after it is a name word ("the Faculty | Grievance
    Committee", "the Faculty | Development Center")."""
    toks = [w for w in (x.strip(",;:'\"()") for x in main.split()) if w]
    n = len(words)
    for i in range(len(toks) - n):
        if toks[i:i + n] == list(words):
            return toks[i + n] in named
    return False


def _subject_broken(words: Sequence[str]) -> bool:
    """A parse subject that ends in a preposition or link word, or holds no content word, broke inside the subject."""
    return bool(words) and (words[-1] in _SUBJECT_LINKS or all(w in _NOT_A_VERB or w in _POP_DETERMINERS for w in words))


def _modal_verb(main: str, named: FrozenSet[str] = frozenset()) -> Optional[str]:
    """The bare verb of a modal- or do-led ask: the first word after the subject that can be a bare verb. The subject runs
    over its first content word, role, label, office and population heads, name words and the word right after a link;
    inside a linked phrase a word followed by another candidate belongs to the phrase ("must faculty with administrative
    assignments undergo", "can failure to comply result", "does neither the Dean nor the Chancellor approve", "can
    termination for financial emergency happen"). A possessive or perfect "have" is the verb; a relative clause whose
    verb takes an object leaves the verb unread (None)."""
    m = re.match(r"^\s*(?:must|shall|should|may|can|could|might|will|would|do|does|did)\s+(?P<body>.+)$", main)
    if m is None:
        return None
    words = [w for w in (x.strip(",;:'\"()").lower() for x in m.group("body").split()) if w]

    def subject_word(w: str) -> bool:
        return (w in named or w in _LABEL_NOUNS or w in _ORG_HEADS or bool(_cm.POPULATION_HEADS.match(w))
                or bool(ASK_GATE.lexicon.head_role(w)))

    def candidate(w: str) -> bool:
        return (bool(w) and not subject_word(w) and _verb_like(w) and not (_participle(w) and not w.endswith("eed"))
                and not (w.endswith("ing") and len(w) > 5))

    def verb_follows(w: str) -> bool:                             # what follows a verb: a link, determiner, name or the end
        return (not w or w in _SUBJECT_LINKS or w in _NEVER_A_VERB or w in _POP_DETERMINERS or w[0].isdigit()
                or subject_word(w))

    started = linked = in_phrase = False
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w in _PRONOUNS:
            started = True
            continue
        if w in _SUBJECT_LINKS:
            if started and w in _RELATIVE:
                after = words[i + 2] if i + 2 < len(words) else ""
                if after in _NOT_A_VERB and after not in _SUBJECT_LINKS or after in _POP_DETERMINERS:
                    return None                                   # "who has no unused vacation leave take ...": unread
            linked, in_phrase = (True, True) if started else (linked, in_phrase)
            continue
        if w in _HAVE_VERBS and started and not linked:
            return nxt if w in ("have", "has", "had") and _participle(nxt) else w
        if w in _NOT_A_VERB or w in _POP_DETERMINERS:
            continue
        if not started or linked or subject_word(w):
            started, linked = True, False
            continue
        if candidate(w):
            if in_phrase and candidate(nxt) and verb_follows(words[i + 2] if i + 2 < len(words) else ""):
                continue                                          # a noun of the phrase: "for financial emergency happen"
            return w
    return None


def _claim_verb_roots(claim: "_cm.Claim") -> FrozenSet[str]:
    """The asked action as roots: the parsed verb, else the claim predicate."""
    if claim.action:
        return frozenset({_root(claim.action)})
    return frozenset(claim.predicate)


def _unit_modality(unit: str, verbs: FrozenSet[str], objects: FrozenSet[str]) -> Optional["_cm.Modality"]:
    """The modality the unit states for the asked action: the marker governing an occurrence of the asked verb (the one
    with the asked object first), else the marker before the asked object; None when neither is in the unit."""
    if not _MODAL_BINDING:
        return _cm.unit_modality(unit)
    anchors = _anchor_positions(unit, verbs)
    if anchors and objects:
        bound = [(p, w) for p, w in anchors if _overlap(objects, _roots(unit[p:p + 140], frozenset()))]
        anchors = bound or anchors
    fallback = False
    if not anchors and objects:
        anchors, fallback = _anchor_positions(unit, objects), True
    if not anchors:
        return None
    stated = [_cm.statement_modality(unit, p, w) for p, w in anchors]
    if fallback:
        stated = [m for m in stated if m is not _cm.Modality.NONE] or [_cm.Modality.NONE]   # the object occurrence a modal governs
    if _cm.Modality.PROHIBITION in stated:
        return _cm.Modality.PROHIBITION
    for m in (_cm.Modality.OBLIGATION, _cm.Modality.RECOMMENDATION, _cm.Modality.ENCOURAGEMENT, _cm.Modality.PERMISSION, _cm.Modality.POSSIBILITY):
        if m in stated:
            return m
    return _cm.Modality.NONE


def _antecedent_role(sentence: str, a: "Antecedent") -> "_cm.ConditionRole":
    """The role of an antecedent from its opener ("only if" / "only when": necessary)."""
    m = _SUBORDINATOR.match(sentence, a.start)
    role = _cm.condition_role(m.group(0)) if m else _cm.ConditionRole.UNKNOWN
    if re.search(r"\bonly\s*$", sentence[max(0, a.start - 6):a.start], re.I):
        role = _cm.ConditionRole.NECESSARY
    return role


def _claim_conditions(low: str) -> Tuple["_cm.Condition", ...]:
    conds: List[_cm.Condition] = []
    for a in _antecedents(low):
        if _conditional(low, a):
            conds.append(_cm.Condition(low[a.start:a.end], a.roots, _antecedent_role(low, a), a.positive))
    for phrase, (s, _e) in _cm.approval_phrases(low):
        if re.search(r"\bonly\s*$", low[max(0, s - 6):s]) or not low[:s].strip(" ,"):
            conds.append(_cm.Condition(phrase, _roots(phrase, frozenset()), _cm.ConditionRole.NECESSARY, True))   # "only with", a leading phrase
    return tuple(conds)


def _value_tokens(text: str) -> FrozenSet[Tuple[str, str]]:
    out = set()
    for m in _VALUE_UNIT.finditer(_hr_norm(text)):
        if m.group("w"):
            out.add((m.group("w").lower(), ""))
        else:
            out.add((m.group("n"), _stem(m.group("u").lower())))
    return frozenset(out)


def _claim_value(clause: str, exclusivity: Sequence["_cm.Exclusivity"]) -> Optional["_cm.QualifiedValue"]:
    a = hr_ask(clause)
    if not a.numeric:
        return None
    f = a.numeric[0]
    obj = f.get("obj") or ()
    qualifier = "up_to" if a.up_to and not a.at_least else "at_least" if a.at_least and not a.up_to else "exact"
    exclusive = any(e.focus is _cm.ExclusivityFocus.VALUE for e in exclusivity)
    return _cm.QualifiedValue(tuple(float(v) for v in f["values"]), str(obj[-1]) if obj else "", qualifier, exclusive)


def read_claim(clause: str) -> Optional["_cm.Claim"]:
    """The structured claim of a specific yes/no question (D-097), else None. It composes the C1-C9 readers (negative ask,
    ask structure, affirmative claim, conditions, modifiers, values, named source) with the claim model's operator readers.
    A question led by its condition is read from its main question ("If ..., may ...?")."""
    low = cue_text(clause.lower()).strip()
    main = _INCLUSIVE_ASIDE.sub("", _main_question(low)).rstrip("?.! ")
    if _C11_FRAMES:
        main = strip_leading_frame(main)                                 # D-098: "According to HR policy, does ...?"
    lead = _LEADING_PHRASE.match(main) if not _cm.ASK_LEAD.match(main) else None
    if lead is not None and _cm.approval_phrases(lead.group("phrase")) and _cm.ASK_LEAD.match(lead.group("rest")):
        main = lead.group("rest")                                        # "Subject to Chancellor approval, can ...?"
    if not _cm.ASK_LEAD.match(low) and not (main and _cm.ASK_LEAD.match(main)):
        return None
    named = _named_words(clause)
    parsed = _plain_predicate(main) if main else None
    st = ask_structure(main) if main else None
    if st is not None and (st.actor or st.label):
        subject_words, verb = list(st.subject), st.verb                   # a role or a label subject: the structural parse
    else:
        subject_words = list(parsed[0]) if parsed else (list(st.subject) if st else [])
        verb = parsed[1] if parsed else (st.verb if st else "")
        if verb == "be" and parsed and parsed[2]:
            nxt = next((w for w in parsed[2].split() if not w.endswith("ly") and w != "not"), "")
            verb = nxt if nxt and nxt not in _NEVER_A_VERB and nxt not in _POP_DETERMINERS and not nxt[0].isdigit() else "be"
        tail = [w for w in (parsed[0] if parsed else []) if w != "not"][-2:]
        if (parsed and verb == parsed[1] and re.match(r"^\s*(?:is|are|was|were|am)\b", main) and len(tail) == 2 and tail[0] in _RELATIVE
                and tail[1] in ("is", "are", "was", "were", "be", "been")):     # the parse's predicate is the relative clause's own
            rest = parsed[2].split()
            k = next((i for i, w in enumerate(rest) if _participle(w) or _COPULAR_PREDICATE.match(w)), None)
            if k is not None:                                    # "are new faculty members who are terminated during probation entitled"
                subject_words, verb = list(parsed[0]) + [verb] + rest[:k], rest[k]
    deontic = _cm.ASK_DEONTIC_ADJ.search(main) or re.search(r"\b(?:have|has|need|needs)\s+to\s+(?P<next>[a-z'’-]+)", main)
    if deontic is not None and deontic.group("next") not in _cm.NOT_A_VERB_AFTER_TO:
        verb = deontic.group("next")                                     # "required to hold a hearing": the action is "hold"
        if verb in ("be", "been"):
            after = re.match(r"\s*(?:\w+ly\s+)?(?P<p>[a-z'’-]+)", main[deontic.end():])
            verb = after.group("p") if after else verb                   # "required to be extended": the action is "extended"
    elif verb and verb not in _LIGHT_ASK_VERBS and (_subject_broken(subject_words) or verb in named or not (
            _verb_like(verb) or _participle(verb) or (re.match(r"^\s*(?:is|are|was|were|am)\b", main) and _COPULAR_PREDICATE.match(verb)))):
        found = _modal_verb(main, named)                                 # "must the Faculty Grievance Committee contact", "must
        if found:                                                        # faculty with administrative assignments undergo"
            toks = [w for w in (x.strip(",;:'\"()") for x in main.split()) if w]
            if found in toks[1:]:
                subject_words = toks[1:toks.index(found, 1)]             # the subject runs up to the verb
            verb = found
    if verb in ("have", "has", "had"):
        rest = parsed[2] if parsed and parsed[1] == verb else (st.rest if st is not None and st.verb == verb else "")
        nxt = rest.split()[0] if rest.split() else ""
        verb = nxt if _participle(nxt) else verb                         # "must emeritus candidates have served"
    ac = affirmative_claim(clause)
    modality = _cm.ask_modality(main or low)
    exclusivity, _ambiguous = _cm.read_exclusivity(main or low)
    population = _population(subject_words) if subject_words else None
    if population is not None and population.head and _cut_in_name(main, subject_words, named):
        population = _cm.Population(head=frozenset(), quantifier=_cm.Quantifier.DEFINITE, text=population.text)   # an entity
    slots = _cm.quantifier_phrases(main)
    if population is not None and not _QUANTIFIER_SLOT:
        q = next((_cm.quantifier_of(w) for w, slot, _h, _s in slots if slot in ("object", "time")), None)
        if q is not None:
            population = replace(population, quantifier=q)
    obj_q = next((_cm.quantifier_of(w) for w, slot, _h, _s in slots if slot == "object"), _cm.Quantifier.GENERIC)
    predicate = ac.predicate if ac else frozenset()
    obj = ac.obj if ac else frozenset()
    action = verb if verb and verb not in ("be", "have", "has", "do", "get") else None
    return _cm.Claim(text=clause, polarity=_cm.Polarity.NEGATIVE if negative_ask(clause) is not None else _cm.Polarity.AFFIRMATIVE,
                     subject=population, actor=st.actor if st else None, label=st.label if st else None, action=action, predicate=predicate,
                     obj=obj, obj_quantifier=obj_q, modality=modality, exclusivity=exclusivity, conditions=_claim_conditions(low),
                     modifiers=tuple(_subject_modifiers(low)), value=_claim_value(clause, exclusivity), source=named_missing_source(clause),
                     operators=_cm.find_operators(main or low, modality=modality))


def read_statements(quote: str, claim: "_cm.Claim") -> Tuple["_cm.Statement", ...]:
    """The statements of a returned quote about the asked claim (D-097): per sentence, the population at the asked
    predicate, the negation of it outside any antecedent, the modality governing it, exclusivity, antecedents with roles,
    exception markers and stated values."""
    out: List[_cm.Statement] = []
    verbs = _claim_verb_roots(claim)
    if _ENTITLEMENT_ASK.search(claim.text):
        verbs = verbs | _ENTITLEMENT_ROOTS
    units = [_evidence._LIST_MARKER.sub("", u) for u in _evidence._local_units(quote)]
    previous = ""
    cursor = 0
    for ui, unit in enumerate(units):
        for si, sentence in enumerate(_unit_sentences(unit)):
            start = quote.find(sentence[:40], cursor)
            start = start if start >= 0 else cursor
            cursor = start
            ants = [a for a in _antecedents(sentence, previous) if _conditional(sentence, a)]
            negated_at = None
            for m in _FACT_NEGATOR.finditer(sentence):
                if _in_antecedent(ants, m.start()):
                    continue
                governed = _roots(_governed_text(sentence[m.end():]), frozenset())
                if _WINDOW_BINDING and not _overlap(verbs, governed):
                    continue
                if _WINDOW_BINDING and claim.obj and not _overlap(claim.obj, governed | _roots(sentence[m.end():], frozenset())):
                    continue
                negated_at = m.start()
                break
            anchors = _anchor_positions(sentence, verbs)
            position = negated_at if negated_at is not None else (anchors[0][0] if anchors else None)
            subject = _statement_population(sentence, position) if position is not None else None
            modality = _cm.statement_modality(sentence, anchors[0][0], anchors[0][1]) if anchors else None
            exclusivity, _amb = _cm.read_exclusivity(sentence)
            antecedents = tuple(_cm.Condition(sentence[a.start:a.end], a.roots, _antecedent_role(sentence, a), a.positive) for a in ants)
            exceptions = tuple(m.group(0) for m in _EXCEPTION_MARKERS.finditer(sentence) if not _in_antecedent(ants, m.start()))
            values = tuple(_cm.QualifiedValue((float(n) if n[0].isdigit() else (1.0 if n == "once" else 2.0),), u)
                           for n, u in sorted(_value_tokens(sentence)))
            roots = _roots(sentence, frozenset())
            out.append(_cm.Statement(text=sentence, unit_index=ui, sentence_index=si, span=(start, start + len(sentence)), subject=subject,
                                     actor=None, action_roots=frozenset(_root(w) for _p, w in anchors),
                                     predicate_roots=frozenset(r for r in roots if any(_same_root(r, v) for v in verbs)),
                                     obj=frozenset(r for r in roots if any(_same_root(r, o) for o in claim.obj)),
                                     polarity=_cm.Polarity.NEGATIVE if negated_at is not None else _cm.Polarity.AFFIRMATIVE,
                                     modality=modality, exclusivity=exclusivity, antecedents=antecedents, exceptions=exceptions, values=values))
            previous = sentence
    return tuple(out)


def modality_detail(clause: str, claim: Optional["_cm.Claim"]) -> Optional[Detail]:
    """D-097 modality: the asked modality, bound to the asked action, stated with a compatible force ("must" is never
    answered by "may" or "encouraged"; "should" answers "must" by D-056; an obligation answers "may")."""
    if not _C10_MODALITY or claim is None or claim.polarity is _cm.Polarity.NEGATIVE or claim.modality is _cm.Modality.NONE:
        return None
    asked = claim.modality
    verbs, objects = _claim_verb_roots(claim), claim.obj

    def local(unit: str) -> bool:
        return _cm.modal_compatibility(asked, _unit_modality(unit, verbs, objects)) in (_cm.Compatibility.COMPATIBLE, _cm.Compatibility.NOT_APPLICABLE)

    action = claim.action or ", ".join(sorted(claim.predicate)) or "the asked action"
    return Detail("modality", "the asked %s for the asked action (%s)" % (asked.value, action), _ClaimCheck(local))


def _scope_window(claim: "_cm.Claim", quote: str, claim_roots: FrozenSet[str]) -> "_cm.Compatibility":
    """The window veto of a population claim: a negation of the asked predicate (with the asked object) for the asked
    population - or, for a universal claim, any part of it - and, for a universal claim, a bound exception."""
    asked = claim.subject
    verbs = _claim_verb_roots(claim)
    for s in read_statements(quote, claim):
        if s.polarity is _cm.Polarity.NEGATIVE:
            breadth = _breadth(claim, claim_roots, s.subject)
            if breadth is not _cm.Breadth.UNRELATED and (asked.quantifier is _cm.Quantifier.UNIVERSAL or breadth is _cm.Breadth.SAME_OR_BROADER):
                return _cm.Compatibility.CONTRADICTED
        if s.exceptions and (asked.quantifier is _cm.Quantifier.UNIVERSAL or not _GENERIC_EXCEPTION_COMPATIBLE):
            sroots = _roots(s.text, frozenset())
            if _overlap(asked.head, sroots) and (_overlap(verbs, sroots) or _overlap(claim.obj, sroots)):
                return _cm.Compatibility.CONTRADICTED
    return _cm.Compatibility.COMPATIBLE


def _stated_populations(unit: str, verbs: FrozenSet[str], objects: FrozenSet[str]) -> List["_cm.Population"]:
    """The person populations the unit states the asked predicate of: the subject before every occurrence of the asked verb
    (else of the asked object) that names a population head."""
    anchors = _anchor_positions(unit, verbs) or _anchor_positions(unit, objects)
    return [p for p in (_statement_population(unit, pos) for pos, _w in anchors) if p is not None and p.head]


def scope_detail(clause: str, claim: Optional["_cm.Claim"]) -> Optional[Detail]:
    """D-097 scope: the asked population stated with the asked breadth. A universal claim needs a universal statement at
    least as broad and no bound exception or negation in the quote; a generic claim a statement at least as broad (the
    strict generic rule: "Muslim faculty attending their first Hajj" does not settle "faculty members"); an existential
    claim any statement of the same population."""
    if not _C10_SCOPE or claim is None or claim.polarity is _cm.Polarity.NEGATIVE:
        return None
    asked = claim.subject
    if asked is None or asked.is_label or not asked.head or asked.quantifier in (_cm.Quantifier.DEFINITE, _cm.Quantifier.PROPORTIONAL):
        return None
    claim_roots = _roots(clause, frozenset())
    verbs = _claim_verb_roots(claim)

    ok = (_cm.Compatibility.COMPATIBLE, _cm.Compatibility.NOT_APPLICABLE)

    def local(unit: str) -> bool:
        stated = _stated_populations(unit, verbs, claim.obj)
        if not stated:                                            # a heading or an agentless sentence names no population
            return _cm.scope_compatibility(asked.quantifier, None, _cm.Breadth.SAME_OR_BROADER) in ok
        return any(_cm.scope_compatibility(asked.quantifier, p.quantifier, _breadth(claim, claim_roots, p)) in ok for p in stated)

    def window(quote: str) -> "_cm.Compatibility":
        return _scope_window(claim, quote, claim_roots)

    return Detail("scope", "the asked %s population (%s) stated with the asked breadth" % (asked.quantifier.value, asked.text), _ClaimCheck(local, window))


def _entity_roots(claim: "_cm.Claim") -> FrozenSet[str]:
    """The content of an entity subject ("the probation period" -> probation, period); none for a person population, a label
    or no subject."""
    p = claim.subject
    if p is None or p.head or p.is_label:
        return frozenset()
    return frozenset(r for r in _roots(p.text) if r not in _PERSON_FAMILY)


def _names_subject(sentence: str, subject: FrozenSet[str]) -> bool:
    sroots = _roots(sentence, frozenset())
    return all(any(_same_root(r, x) for x in sroots) for r in subject)


def _about_subject(unit: str, subject: FrozenSet[str]) -> bool:
    """Some sentence of the unit names every content word of the entity subject."""
    return any(_names_subject(sentence, subject) for sentence in _unit_sentences(unit))


def _stated_role(unit: str, roots: FrozenSet[str], subject: FrozenSet[str] = frozenset()) -> Tuple[Optional["_cm.ConditionRole"], bool]:
    """The role the unit gives a condition naming ``roots``: an antecedent's role, an approval gate (necessary), or a
    requirement ("requires", "subject to"); (None, False) when the unit does not state it. With ``subject`` (an entity
    subject), only a sentence naming the subject, or a pronoun-led sentence continuing one, states it."""
    need = max(1, -(-len(roots) // 2))
    previous = ""
    found: Optional[_cm.ConditionRole] = None
    about = False
    for sentence in _unit_sentences(unit):
        if subject:
            about = _names_subject(sentence, subject) or (about and bool(_ANAPHOR.match(sentence)))
            if not about:
                previous = sentence
                continue
        for phrase, _span in _cm.approval_phrases(sentence):
            if _overlap(roots, _roots(phrase, frozenset())) >= 1:
                return _cm.ConditionRole.NECESSARY, True
        for a in _antecedents(sentence, previous):
            if _conditional(sentence, a) and _overlap(roots, a.roots) >= need:
                found = _antecedent_role(sentence, a)
        if re.search(r"\b(?:requires?|subject\s+to)\b", sentence, re.I) and _overlap(roots, _roots(sentence, frozenset())) >= need:
            return _cm.ConditionRole.NECESSARY, False
        previous = sentence
    return found, False


def _target_roots(words: Iterable[str]) -> FrozenSet[str]:
    return frozenset(_root(w) for w in words if w not in _CLAIM_STOP) or frozenset(_root(w) for w in words)


#: A negated extension or renewal leaves a value exclusive ("cannot be extended", "non-renewable", "no extension").
_NEGATED_ALTERNATIVE = re.compile(r"\b(?:not|no|cannot|never|non)\b[\s-]+(?:\w+[\s-]+){0,2}?(?:extend\w*|extension|prolong\w*|renew\w*)|\bnon-?renewable\b",
                                  re.I)
#: A sentence that continues the one before it ("It is extendable to two years ...").
_ANAPHOR = re.compile(r"^\s*(?:it|this|that|these|they|such|the\s+same)\b", re.I)


def _alternative_stated(norm: str, values: FrozenSet[Tuple[str, str]], asked_values: FrozenSet[Tuple[str, str]], asked_roots: FrozenSet[str]) -> bool:
    """An alternative to an exclusive value: an extension or a renewal not negated and not named by the ask itself
    ("renewable only once"), or another value of the asked unit ("twice" for "once")."""
    text = _NEGATED_ALTERNATIVE.sub(" ", norm)
    extension = bool(_EXTENSION_WORDS.search(text)) and not any(r.startswith(("extend", "extens", "prolong")) for r in asked_roots)
    renewal = bool(_RENEWAL_WORDS.search(text)) and not any(r.startswith("renew") for r in asked_roots)
    other = any(n != an and u == au and (u or an in ("once", "twice")) for n, u in values for an, au in asked_values)
    return extension or renewal or other


def _value_bind_roots(clause: str, asked_values: FrozenSet[Tuple[str, str]]) -> FrozenSet[str]:
    """The content of a value claim without its value: "Is the probation period only one year?" -> probation, period."""
    drop = {_root(w) for w in _cm.NUMBER_WORDS} | {_root(u) for _n, u in asked_values if u} | {u for _n, u in asked_values if u}
    return frozenset(r for r in _roots(clause) if r not in drop and not r[:1].isdigit())


def _exclusivity_local(unit: str, e: "_cm.Exclusivity", claim: "_cm.Claim", asked_values: FrozenSet[Tuple[str, str]]) -> "_cm.Compatibility":
    target = _target_roots(e.target)
    stated, _amb = _cm.read_exclusivity(unit)
    stated_bound = any(_overlap(target, _target_roots(x.target)) for x in stated if x.focus is e.focus or e.focus is _cm.ExclusivityFocus.ACTOR)
    if e.focus is _cm.ExclusivityFocus.VALUE:
        if not _VALUE_EXCLUSIVITY:
            return _cm.Compatibility.COMPATIBLE
        values = _value_tokens(unit)
        if asked_values and not (asked_values & values):
            return _cm.Compatibility.UNKNOWN
        norm = _hr_norm(unit)
        exclusive = any(re.search(r"\b(?:only|solely|just)\s+" + re.escape(n) + r"\b", norm, re.I) for n, _u in asked_values)
        alternative = _alternative_stated(norm, values, asked_values, _roots(claim.text))
        if exclusive:
            return _cm.Compatibility.COMPATIBLE
        return _cm.exclusivity_compatibility(e.focus, False, alternative, bool(asked_values & values))
    if e.focus is _cm.ExclusivityFocus.TIME:
        if stated_bound:
            return _cm.Compatibility.COMPATIBLE
        hits = _anchor_positions(unit, target)
        if hits and _overlap(target, _roots(unit, frozenset())) >= max(1, len(target) // 2):
            if _cm.statement_modality(unit, hits[0][0], "") is _cm.Modality.OBLIGATION:
                return _cm.Compatibility.COMPATIBLE                   # an obligation on the time slot ("must coincide with the end of ...")
        return _cm.Compatibility.UNKNOWN
    if e.focus is _cm.ExclusivityFocus.CONDITION:
        if stated_bound:
            return _cm.Compatibility.COMPATIBLE
        role, _gate = _stated_role(unit, target, _entity_roots(claim))
        return _cm.Compatibility.COMPATIBLE if role is _cm.ConditionRole.NECESSARY else _cm.Compatibility.UNKNOWN
    return _cm.exclusivity_compatibility(e.focus, stated_bound, False, False)


def _exclusivity_window(quote: str, e: "_cm.Exclusivity", asked_values: FrozenSet[Tuple[str, str]], claim_text: str) -> "_cm.Compatibility":
    """The window veto of a value exclusivity: an alternative (an extension, a renewal, another value of the asked unit) in
    a sentence of the quote bound to the asked claim - one that states the asked value, names the content of the claim,
    or continues a bound sentence with a pronoun ("It is extendable to two years")."""
    if e.focus is not _cm.ExclusivityFocus.VALUE or not _VALUE_EXCLUSIVITY:
        return _cm.Compatibility.COMPATIBLE
    asked_roots = _roots(claim_text)
    bind = _value_bind_roots(claim_text, asked_values)
    need = max(1, (len(bind) + 1) // 2)
    bound_before = False
    for sentence in _unit_sentences(quote):
        values = _value_tokens(sentence)
        bound = (not asked_values or bool(asked_values & values) or (bool(bind) and _overlap(bind, _roots(sentence, frozenset())) >= need)
                 or (bound_before and bool(_ANAPHOR.match(sentence))))
        bound_before = bound
        if not bound:
            continue
        norm = _hr_norm(sentence)
        if any(re.search(r"\b(?:only|solely|just)\s+" + re.escape(n) + r"\b", norm, re.I) for n, _u in asked_values):
            continue
        if _alternative_stated(norm, values, asked_values, asked_roots):
            return _cm.Compatibility.CONTRADICTED
    return _cm.Compatibility.COMPATIBLE


def exclusivity_detail(clause: str, claim: Optional["_cm.Claim"]) -> Optional[Detail]:
    """D-097 exclusivity: an "only / solely / exclusively / just N" claim stated as exclusive on the same focus (actor,
    population, object, condition, value, time); a value exclusivity also by the exact value with no extension, renewal or
    other value in the quote; a time or condition exclusivity also by an obligation or a necessary condition on the slot."""
    if not _C10_EXCLUSIVITY or claim is None or not claim.exclusivity:
        return None
    e = claim.exclusivity[0]
    asked_values = _value_tokens(claim.text) if e.focus is _cm.ExclusivityFocus.VALUE else frozenset()

    def local(unit: str) -> bool:
        return _exclusivity_local(unit, e, claim, asked_values) is _cm.Compatibility.COMPATIBLE

    def window(quote: str) -> "_cm.Compatibility":
        return _exclusivity_window(quote, e, asked_values, claim.text)

    label = "the asked exclusivity (%s, %s: %s)" % (e.marker, e.focus.value, " ".join(sorted(e.target)))
    return Detail("exclusivity", label, _ClaimCheck(local, window))


def _condition_window(quote: str, claim: "_cm.Claim", contents: Sequence[FrozenSet[str]]) -> "_cm.Compatibility":
    """A negation of the asked predicate stated for the population an asked sufficient condition identifies ("new faculty
    members who resign during probation are not entitled to ... annual leave" against "If I resign during probation, am I
    entitled to annual leave?")."""
    for st in read_statements(quote, claim):
        if st.polarity is not _cm.Polarity.NEGATIVE:
            continue
        sroots = _roots(st.subject.text, frozenset()) if st.subject is not None else frozenset()
        for a in st.antecedents:
            sroots = sroots | a.roots
        if any(content and all(any(_same_root(r, x) for x in sroots) for r in content) for content in contents):
            return _cm.Compatibility.CONTRADICTED
    return _cm.Compatibility.COMPATIBLE


def condition_role_detail(clause: str, claim: Optional["_cm.Claim"]) -> Optional[Detail]:
    """D-097 condition roles: a necessary condition asked ("only if", "only with approval", "provided that") needs a
    necessary statement (an approval gate, "only if", "requires", "subject to"); a sufficient one is not answered by a bare
    necessary statement ("X only if Y" states no sufficiency). The C9 identity contract binds the condition's content."""
    if not _C10_CONDITION_ROLES or claim is None:
        return None
    necessary = [c for c in claim.conditions if c.role is _cm.ConditionRole.NECESSARY]
    sufficient = [c for c in claim.conditions if c.role in (_cm.ConditionRole.SUFFICIENT, _cm.ConditionRole.TRIGGER)]
    if not necessary and not sufficient:
        return None

    subject = _entity_roots(claim)

    def local(unit: str) -> bool:
        if subject and not _about_subject(unit, subject):
            return False                                          # the unit states nothing of the asked entity subject
        for c in necessary:
            role, gate = _stated_role(unit, c.roots, subject)
            if _cm.condition_role_compatibility(c.role, role, gate) is not _cm.Compatibility.COMPATIBLE:
                return False
        for c in sufficient:
            role, gate = _stated_role(unit, c.roots, subject)
            if _cm.condition_role_compatibility(c.role, role, gate) is _cm.Compatibility.UNKNOWN:
                return False
        return True

    contents = [frozenset(r for r in c.roots if r not in _PERSON_FAMILY) for c in sufficient]

    def window(quote: str) -> "_cm.Compatibility":
        return _condition_window(quote, claim, contents)

    shown = necessary[0] if necessary else sufficient[0]
    return Detail("condition_role", "the asked %s condition (%s)" % (shown.role.value, shown.text.strip()), _ClaimCheck(local, window))


def operator_detail(clause: str, claim: Optional["_cm.Claim"]) -> Optional[Detail]:
    """D-097 fail-closed: a recognised operator no reader verifies (most / many / few / several, an ambiguous "just", an
    unqualified comparative, an event order) never settles the clause."""
    if not _C10_OPERATORS or claim is None:
        return None
    bad = _cm.unhandled(claim.operators)
    if not bad:
        return None
    return Detail("operator", "the asked %s (%s)" % (bad[0].kind.value.replace("_", " "), bad[0].text), lambda unit: False)


def structured_details(clause: str, claim: Optional["_cm.Claim"]) -> List[Detail]:
    """The D-097 details of a clause: modality, scope, exclusivity, condition role and the fail-closed operator detail."""
    out: List[Detail] = []
    for build in (modality_detail, scope_detail, exclusivity_detail, condition_role_detail, operator_detail):
        d = build(clause, claim)
        if d is not None:
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# D-098 (Step 4.12-C11): statement-level composite claim
# ---------------------------------------------------------------------------
#: One predicate statement per finite clause (with its secondary predications), read from the RAW unit text; ONE joint
#: comparison of the claim with one statement on every dimension (``claim_model.compare_claim_to_statement``); one operator
#: ledger; the rest of the returned quote can only veto. The D-097 per-dimension details stay as explainers. Each rule has
#: one switch (a mutation seam).
_C11_JOINT = True
_C11_LEDGER = True
_C11_FREQUENCY = True
_C11_SLOT_UNIVERSAL = True
_C11_ENUMERATION = True
_C11_ALTERNATIVES = True
_C11_UNCONDITIONAL = True
_C11_EXCEPTION_WINDOW = True
_C11_FRAMES = True
#: A finite verb group: a clause's subject runs up to its first word.
_C11_VERB_GROUP = re.compile(r"\b(?:may|must|shall|should|can|could|will|would|might|cannot|is|are|was|were|has|have|had|does|do|did|needs?)\b", re.I)
_C11_COORDINATOR = re.compile(r",?\s+(?:and|or|but|yet|while|whereas|whilst)\s+", re.I)
_C11_ALWAYS_SPLIT: FrozenSet[str] = frozenset({"while", "whereas", "whilst"})
#: A secondary predication after the verb group: an -able / -ible adjective, a past participle with a preposition, a present
#: participle with a determiner ("Nursing, and Law" and ", including ..." are none).
_C11_SECONDARY = re.compile(r",\s+(?=(?:[a-z]{4,}(?:able|ible)\b|(?:[a-z]{3,}ed|based|made|given|held|paid)\s+(?:by|to|for|with|until|in|on|at|from|upon)\b|"
                            r"(?!including|excluding|following|according|regarding|concerning|pending|during)[a-z]{4,}ing\s+"
                            r"(?:the|a|an|their|its|his|her|this|that|these|those)\b))", re.I)
#: A safe leading frame of a question (claim side only): a source or level frame, or "In practice". "In general" and
#: "Typically" are hedges (operators) and stay.
_C11_FRAME = re.compile(r"^\s*(?:(?:according\s+to|as\s+per|per|under|based\s+on|in\s+accordance\s+with|as\s+stated\s+in|in\s+line\s+with)\s+(?:the\s+)?"
                        r"(?:[\w'’-]+\s+){0,4}?(?:policy|policies|handbook|rules|regulations|bylaws|guidelines|procedures?)|"
                        r"at\s+the\s+(?:department|college|university|institutional|faculty)\s+level|in\s+practice)\s*,\s*", re.I)
_C11_FRAME_OPERATOR = re.compile(r"\b(?:only|all|every|each|any|some|no|not|never|always|if|unless|when|except|without|\d+)\b", re.I)
_C11_SUBJECT_TO = re.compile(r"\bsubject\s+to\b(?P<body>[^,.;]{0,80})", re.I)
_C11_APPROVAL_NOUN = re.compile(r"\b(?:approval|decision|recommendation|endorsement|consent|authori[sz]ation|permission|review)s?\b", re.I)
_C11_PAREN = re.compile(r"\([^()]*\)")
#: Irregular participles of a copular ask with no parsed action ("Is the allowance paid ...?").
_C11_IRREGULAR_PARTICIPLES: FrozenSet[str] = frozenset("paid made given held taken sent set kept met won led told known shown seen done chosen drawn "
                                                       "written spent lent sought bought brought taught thought caught built run".split())
_C11_ROLE_KIND = {"sufficient": "SUFFICIENT", "trigger": "TRIGGER", "necessary": "NECESSARY", "exception": "EXCEPTION", "concessive": "CONCESSIVE"}


def strip_leading_frame(text: str) -> str:
    """The question without a safe leading frame ("According to HR policy, ...", "At the department level, ...", "In practice,
    ..."), when the rest is a yes/no question and the frame holds no operator; else the text unchanged. Only the claim reader
    uses it: the evidence, ``ask_text`` and the C9 content readers keep the full question (D-098)."""
    m = _C11_FRAME.match(text)
    if m is None or _C11_FRAME_OPERATOR.search(m.group(0)):
        return text
    rest = text[m.end():]
    return rest if _cm.ASK_LEAD.match(rest) else text


def _raw_sentences(unit: str) -> List[str]:
    """The sentences of a unit in raw text: never cue-normalised, as the C10 local readers read units (D-098)."""
    return [s for s in _SENTENCE_SPLIT.split(unit.replace("|", " ")) if s.strip()]


def _clause_spans(sentence: str) -> List[Tuple[int, int, bool]]:
    """The finite clauses of a sentence as (start, end, elided): hard breaks at ";" and ":"; a coordinated clause after "while /
    whereas / whilst", or after "and / or / but / yet" when both sides carry a verb group (an enumeration is never split). A
    clause that opens with its verb group has an elided subject ("must submit X and must coincide with Y")."""
    out: List[Tuple[int, int, bool]] = []
    seg_start = 0
    for h in [m.start() for m in re.finditer(r"[;:]", sentence)] + [len(sentence)]:
        seg = sentence[seg_start:h]
        accepted, last = [], 0
        for m in _C11_COORDINATOR.finditer(seg):
            after = seg[m.end():]
            stop = re.search(r"[,;]", after)
            piece = after[:stop.start()] if stop else after
            if m.group(0).strip(" ,").lower() in _C11_ALWAYS_SPLIT or (
                    _C11_VERB_GROUP.search(" ".join(piece.split()[:12])) and _C11_VERB_GROUP.search(seg[last:m.start()])):
                accepted.append(m)
                last = m.end()
        starts = [0] + [m.end() for m in accepted]
        ends = [m.start() for m in accepted] + [len(seg)]
        for i, (a, b) in enumerate(zip(starts, ends)):
            if seg[a:b].strip(" ,"):
                first = (seg[a:b].split() or [""])[0].lower()
                out.append((seg_start + a, seg_start + b, i > 0 and bool(_C11_VERB_GROUP.fullmatch(first))))
        seg_start = h + 1
    return out


def _secondary_spans(sentence: str, a: int, b: int, threshold: int) -> List[Tuple[int, int]]:
    """The predicate spans of a finite clause: its main part, then each secondary predication opened after the verb group
    (``threshold``); before the verb group the same shapes are appositive restrictions of the subject."""
    cuts = [m.start() for m in _C11_SECONDARY.finditer(sentence, max(a, threshold), b)]
    bounds = [a] + cuts + [b]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _c11_non_finite(sentence: str, p: int, word: str) -> bool:
    """An anchor inside the subject span that is no main verb: an infinitive ("intending to resign"), an -ing or -ed form, or a
    noun-shaped form ("the appointment of the Dean is ...")."""
    return (bool(re.search(r"\bto\s+$", sentence[max(0, p - 4):p], re.I)) or word.endswith(("ing", "ed")) or bool(_NOT_BARE_VERB.search(word)))


def _c11_anchor_kind(claim: "_cm.Claim", word: str) -> "_cm.AnchorKind":
    r = _root(word)
    if claim.action and _same_root(r, _root(claim.action)):
        return _cm.AnchorKind.VERB
    if any(_same_root(r, e) for e in _ENTITLEMENT_ROOTS):
        return _cm.AnchorKind.ENTITLEMENT
    return _cm.AnchorKind.VERB if claim.action else _cm.AnchorKind.PREDICATE


def _c11_np_stop(claim: "_cm.Claim") -> Callable[[str], bool]:
    """The noun-phrase stop of both sides: a verb-group word or a word of the asked verb (one break set, D-098)."""
    verbs = _claim_verb_roots(claim)
    return lambda w: bool(_C11_VERB_GROUP.fullmatch(w)) or any(_same_root(_root(w), v) for v in verbs)


def _c11_noun_phrases(text: str, stop: Callable[[str], bool], slot: str = "") -> Tuple["_cm.NounPhrase", ...]:
    """The quantified noun phrases of a text, read by the shared reader (``claim_model.noun_phrase_after``)."""
    words = [w for w, _s, _e in _cm.tokens(text)]
    out = []
    for i, w in enumerate(words):
        q = _cm.quantifier_of(w)
        if q not in (_cm.Quantifier.UNIVERSAL, _cm.Quantifier.EXISTENTIAL):
            continue
        head, mods = _cm.noun_phrase_after(words, i, stop)
        if head:
            capped = any(m[0].isdigit() or m in _cm.NUMBER_WORDS or m in ("up", "maximum", "minimum") for m in mods)
            out.append(_cm.NounPhrase(text=" ".join((w,) + mods + (head,)), head=_root(head),
                                      modifiers=frozenset(_root(m) for m in mods if m not in _cm.ITEM_DROP), quantifier=q, capped=capped, slot=slot))
    return tuple(out)


def _c11_negated(sentence: str, pa: int, pb: int, ants: Sequence[Antecedent], verbs: FrozenSet[str], obj: FrozenSet[str]) -> bool:
    """The anchored predicate is negated: a negator of the predicate span, outside any antecedent, governing the asked verb (and
    the asked object when one is asked) - the read_statements binding on one predicate."""
    for m in _FACT_NEGATOR.finditer(sentence, pa, pb):
        if _in_antecedent(ants, m.start()):
            continue
        governed = _roots(_governed_text(sentence[m.end():pb]), frozenset())
        if not _overlap(verbs, governed):
            continue
        if obj and not _overlap(obj, governed | _roots(sentence[m.end():pb], frozenset())):
            continue
        return True
    return False


def _c11_restrictions(sentence: str, pa: int, pb: int, rel: int, ants: Sequence[Antecedent], modality: "_cm.Modality"
                      ) -> Tuple[Tuple["_cm.Restriction", ...], Tuple[str, ...]]:
    """The restrictions bound to one predicate (its antecedents with their roles, approval gates, "subject to", special
    circumstances, occasion exceptions, bound hedges, a may / can modality) and its bound hedges. A parenthetical hedge and a
    hedge on a noun-phrase modifier ("preferably holding ...") are not bound."""
    pred = sentence[pa:pb]
    out: List[_cm.Restriction] = []
    for x in ants:
        role = _antecedent_role(sentence, x).value
        kind = getattr(_cm.RestrictionKind, _C11_ROLE_KIND.get(role, "SUFFICIENT"))
        out.append(_cm.Restriction(kind, sentence[x.start:x.end], x.roots))
    for m in _C11_SUBJECT_TO.finditer(pred):
        kind = _cm.RestrictionKind.GATE if _C11_APPROVAL_NOUN.search(m.group("body")) else _cm.RestrictionKind.NECESSARY
        out.append(_cm.Restriction(kind, m.group(0), _roots(m.group(0), frozenset())))
    for phrase, _span in _cm.approval_phrases(pred):
        if not phrase.lower().startswith("subject"):
            out.append(_cm.Restriction(_cm.RestrictionKind.GATE, phrase, _roots(phrase, frozenset())))
    for m in _cm.SPECIAL_CIRCUMSTANCES.finditer(pred):
        out.append(_cm.Restriction(_cm.RestrictionKind.SPECIAL, m.group(0)))
    for m in _cm.OCCASION_EXCEPTION.finditer(pred):
        out.append(_cm.Restriction(_cm.RestrictionKind.EXCEPTION, m.group(0)))
    no_paren = _C11_PAREN.sub(lambda m: " " * len(m.group(0)), pred)
    hedges: List[str] = []
    for m in _cm.HEDGES.finditer(no_paren):
        after = no_paren[m.end():]
        if re.match(r"\s+\w+ing\b", after):
            continue
        if m.start() < rel or abs(m.start() - rel) < 40 or re.match(r"\s*(?:only|\d|one|two|three|once|twice|not)\b", after, re.I):
            hedges.append(m.group(0).lower())
            out.append(_cm.Restriction(_cm.RestrictionKind.HEDGE, m.group(0)))
    if modality in (_cm.Modality.PERMISSION, _cm.Modality.POSSIBILITY):
        out.append(_cm.Restriction(_cm.RestrictionKind.POSSIBILITY, modality.value))
    return tuple(out), tuple(hedges)


def _c11_statement(sentence: str, a: int, b: int, pa: int, pb: int, ui: int, si: int, ci: int, kind: "_cm.AnchorKind", p: int, w: str,
                   secondary: bool, subject: Optional["_cm.Population"], subject_text: str, inherited: bool, ants: Sequence[Antecedent],
                   frame: "_cm.ClaimFrame") -> "_cm.PredicateStatement":
    text, pred = sentence[a:b], sentence[pa:pb]
    rel = p - pa
    if kind is _cm.AnchorKind.FALLBACK:
        modality = _cm.unit_modality(pred)
    else:
        modality = _cm.statement_modality(pred, max(0, rel), w)
        if secondary and modality is _cm.Modality.NONE:
            modality = _cm.statement_modality(sentence[a:pb], p - a, w)     # a secondary predication shares the main verb group
    initial = [x for x in ants if _SENTENCE_OPEN.match(sentence[:x.start])]
    trailing = [x for x in ants if (pa <= x.start < pb) or (not secondary and a <= x.start < b)]
    bound_ants = initial + [x for x in trailing if x not in initial]
    restrictions, hedges = _c11_restrictions(sentence, pa, pb, rel, bound_ants, modality)
    stop = _c11_np_stop(frame.claim)
    nps = _c11_noun_phrases(text, stop)
    items = tuple(_cm.enumeration_items(text, m.end(), _root) for m in _cm.EXCLUSIVE_LIST_MARKERS.finditer(text))
    times = []
    for m in _cm.TIME_FILLER.finditer(text):
        around = text[max(0, m.start() - 30):m.end() + 30]
        times.append(_cm.TimeFiller(m.group(0), _roots(m.group(0), frozenset()), bool(_cm.RELATIVE_OFFSET.search(around))))
    numbers = frozenset(re.findall(r"\d+(?:\.\d+)?", _hr_norm(text)))
    negative = kind is not _cm.AnchorKind.FALLBACK and _c11_negated(sentence, pa, pb, ants, frame.anchor_roots, frame.claim.obj)
    return _cm.PredicateStatement(
        text=text, predicate=pred, sentence=sentence, unit_index=ui, sentence_index=si, clause_index=ci,
        anchor=_cm.Anchor(kind, w, p, secondary), subject=subject, subject_text=subject_text, subject_inherited=inherited, modality=modality,
        negative=negative, restrictions=restrictions, member_exceptions=tuple(text[m.start():m.start() + 60] for m in _cm.MEMBER_EXCEPTION.finditer(text)),
        hedges=hedges, frequency=_cm.frequency_of(pred)[0], noun_phrases=nps, exclusive_items=items, times=tuple(times), numbers=numbers)


def read_clause_statements(unit: str, frame: "_cm.ClaimFrame", unit_index: int = 0, fallback: bool = True) -> Tuple["_cm.PredicateStatement", ...]:
    """The predicate statements of one local unit for a claim (D-098). Per finite clause: anchors of the asked verb, the
    entitlement family or the copular predicate (outside antecedents; a non-finite or noun-shaped occurrence inside the subject
    span is none), else of the asked object after the subject; at most three. The subject is read from the clause start (an
    elided clause inherits it; a secondary predication has none). With no anchor in the unit, the finite clause sharing the
    most claim content (at least two roots) is a FALLBACK statement; with none, there is no statement (never support). D-099:
    for a specific claim a coordinated clause with a new explicit subject and finite verb is its own clause, and when a statement
    states the asked verb, a clause that only mentions the asked object under its own subject is no statement of the claim."""
    claim = frame.claim
    verbs, objects = frame.anchor_roots, claim.obj
    specific = is_specific_claim(claim.text)
    spans_of = _c12_clause_spans if specific else _clause_spans
    out: List[_cm.PredicateStatement] = []
    previous = ""
    sentences = _raw_sentences(unit)
    for si, sentence in enumerate(sentences):
        ants = [x for x in _antecedents(sentence, previous) if _conditional(sentence, x)]
        prev_subject: Optional[_cm.Population] = None
        prev_text = ""
        for ci, (a, b, elided) in enumerate(spans_of(sentence)):
            vg = next((m for m in _C11_VERB_GROUP.finditer(sentence, a, b) if not _in_antecedent(ants, m.start())), None)
            subject_end = vg.start() if (vg is not None and not elided) else a
            found = []
            for p, w in _anchor_positions(sentence[a:b], verbs):
                p += a
                if _in_antecedent(ants, p) or (p < subject_end and _c11_non_finite(sentence, p, w)):
                    continue
                found.append((p, w, _c11_anchor_kind(claim, w)))
            if found and objects:
                bound = [x for x in found if _overlap(objects, _roots(sentence[x[0]:x[0] + 140], frozenset()))]
                found = bound or found
            if not found and objects:
                found = [(p + a, w, _cm.AnchorKind.OBJECT) for p, w in _anchor_positions(sentence[a:b], objects)
                         if not _in_antecedent(ants, p + a) and p + a >= subject_end]
            marks = [x for x in ((vg.end() if vg is not None else None),
                                 next((p for p, _w, k in found if k is not _cm.AnchorKind.OBJECT), None)) if x is not None]
            spans = _secondary_spans(sentence, a, b, min(marks) if marks else a)
            if elided:
                subject, subject_text, inherited = prev_subject, prev_text, True
            else:
                subject = _statement_population(sentence[a:subject_end], subject_end - a) if subject_end > a else None
                subject_text, inherited = sentence[a:subject_end], False
                prev_subject, prev_text = subject, subject_text
            for p, w, kind in found[:3]:
                k = next((i for i, (x, y) in enumerate(spans) if x <= p < y), 0)
                pa, pb = spans[k]
                subj, subj_text = subject, subject_text
                if k > 0:
                    subj = None                                   # a secondary predication: its subject is not read
                elif not elided and kind is not _cm.AnchorKind.OBJECT and (vg is None or vg.start() > p) and not _NOT_BARE_VERB.search(w):
                    subj, subj_text = _statement_population(sentence[a:p], p - a), sentence[a:p]   # an active verb with no auxiliary
                out.append(_c11_statement(sentence, a, b, pa, pb, unit_index, si, ci, kind, p, w, k > 0, subj, subj_text, inherited, ants, frame))
        previous = sentence
    if fallback and not out:
        claim_roots = _roots(claim.text)
        best = None
        previous = ""
        for si, sentence in enumerate(sentences):
            ants = [x for x in _antecedents(sentence, previous) if _conditional(sentence, x)]
            for ci, (a, b, _elided) in enumerate(spans_of(sentence)):
                n = _overlap(claim_roots, _roots(sentence[a:b], frozenset()))
                if n >= 2 and (best is None or n > best[0]):
                    vg = _C11_VERB_GROUP.search(sentence, a, b)
                    cut = vg.start() if vg is not None else a
                    subject = _statement_population(sentence[a:cut], cut - a) if cut > a else None
                    pa, pb = _secondary_spans(sentence, a, b, vg.end() if vg is not None else a)[0]   # the main predicate only
                    best = (n, _c11_statement(sentence, a, b, pa, pb, unit_index, si, ci, _cm.AnchorKind.FALLBACK, cut, "", False, subject,
                                              sentence[a:cut], False, ants, frame))
            previous = sentence
        if best is not None:
            out.append(best[1])
    if specific and _C12_OBJECT_ANCHOR and any(s.anchor.kind in _C12_BOUND_ANCHORS for s in out):
        out = [s for s in out if s.anchor.kind is not _cm.AnchorKind.OBJECT or s.subject_inherited]
    return tuple(out)


def _asked_population(claim: "_cm.Claim", main: str) -> Optional["_cm.Population"]:
    """The asked population read from the whole subject phrase (the auxiliary to the asked predicate, the entitlement word
    first), operator words removed - read as statements are read ("female faculty employed full time"); else the C10 one."""
    if claim.subject is None or not claim.subject.head or claim.subject.is_label:
        return claim.subject
    words = [w for w, _s, _e in _cm.tokens(main)]
    ent = [i for i, t in enumerate(words[1:], start=1) if _ENTITLEMENT_ASK.fullmatch(t)]
    verbs = _claim_verb_roots(claim)
    end = ent[0] if ent else next((i for i, t in enumerate(words[1:], start=1) if any(_same_root(_root(t), v) for v in verbs)), None)
    if end is None or end < 2:
        return claim.subject
    keep = [t for t in words[1:end] if t not in ("not", "be", "been", "only") and not any(p.fullmatch(t) for _c, p in _cm.FREQUENCY_LEXICON)]
    pop = _population(keep) if keep else None
    return pop if (pop is not None and pop.head) else claim.subject


def _asked_slots(claim: "_cm.Claim", main: str, skip: Sequence[Tuple[int, int]], anchor_roots: FrozenSet[str]) -> Tuple["_cm.NounPhrase", ...]:
    """The asked universal noun phrases read by the shared reader: every one but a person-headed subject (the C10 Population
    reads that one) - entity subjects ("all senior administrative appointments", "all faculty appointments"), objects and
    complements; a universal inside an exclusive enumeration or an occasion phrase is read by those dimensions instead."""
    stop = _c11_np_stop(claim)
    toks = _cm.tokens(main)
    words = [w for w, _s, _e in toks]
    starts = [s for _w, s, _e in toks]
    pred_at = next((s for w, s, _e in toks[1:] if any(_same_root(_root(w), v) for v in anchor_roots)), len(main))
    out = []
    for w, slot, _h, span in _cm.quantifier_phrases(main):
        if _cm.quantifier_of(w) is not _cm.Quantifier.UNIVERSAL or slot in ("time", "proportional") or any(x <= span[0] < y for x, y in skip):
            continue
        if span[0] not in starts:
            continue
        head, mods = _cm.noun_phrase_after(words, starts.index(span[0]), stop)
        position = "subject" if span[0] < pred_at else "object"
        if not head or (position == "subject" and _cm.POPULATION_HEADS.match(head)):
            continue                                              # a person subject: the C10 Population scope reads it
        out.append(_cm.NounPhrase(text=" ".join((w,) + mods + (head,)), head=_root(head),
                                  modifiers=frozenset(_root(m) for m in mods if m not in _cm.ITEM_DROP), quantifier=_cm.Quantifier.UNIVERSAL,
                                  slot=position, span=span))
    return tuple(out)


def _claim_structured(claim: "_cm.Claim") -> bool:
    return claim.modality is not _cm.Modality.NONE or bool(claim.exclusivity) or bool(claim.conditions) or (
        claim.subject is not None and bool(claim.subject.head) and claim.subject.quantifier is not _cm.Quantifier.DEFINITE)


def _claim_ledger(claim: "_cm.Claim", main: str, freq_span: Tuple[int, int], excl_spans: Sequence[Tuple[int, int]],
                  slots: Sequence["_cm.NounPhrase"]) -> Tuple["_cm.LedgerEntry", ...]:
    """The operators of the asked claim, longest match first, in the D-098 precedence (conditionality, frequency, exclusivity,
    scope, quantity, comparison, ordering, residue), each with the dimension that must consume it on the bound statement;
    "static" names an accepted C9 / C10 reader that verifies (or fails closed) outside the composite."""
    E = _cm.LedgerEntry
    out: List[_cm.LedgerEntry] = []
    taken: List[Tuple[int, int]] = []

    def free(span: Tuple[int, int]) -> bool:
        return not any(span[0] < y and x < span[1] for x, y in taken)

    for c in claim.conditions:
        out.append(E("conditionality", c.text.strip(), (0, 0), "condition", "condition"))
    freq, words, _span = _cm.frequency_of(main)
    if freq is not None:
        out.append(E("frequency", words, freq_span, "occasion", "frequency"))
        taken.append(freq_span)
    for e in claim.exclusivity:
        out.append(E("exclusivity", e.marker, (0, 0), e.focus.value, "exclusivity"))
    slot_at = {s.span[0]: s for s in slots}
    for w, slot, _head, span in _cm.quantifier_phrases(main):
        if not free(span):
            continue
        taken.append(span)
        q = _cm.quantifier_of(w)
        if slot in ("time", "proportional") or (claim.polarity is _cm.Polarity.NEGATIVE and q is _cm.Quantifier.EXISTENTIAL):
            handler = "static"                                    # a time distributive, a proportional (C10 fails it), a negative existential
        elif any(x <= span[0] for x, _y in excl_spans):
            handler = "exclusivity"
        elif span[0] in slot_at:
            handler = ("scope" if slot_at[span[0]].slot == "subject" else "object") if _C11_SLOT_UNIVERSAL else "static"
        elif slot == "population":
            handler = "scope"
        else:
            handler = "static"
        out.append(E("scope", w, span, slot, handler))
    structured = _claim_structured(claim)
    for m in list(_cm.NUMERIC_QUALIFIERS.finditer(main)) + list(re.finditer(r"\b\d+(?:\.\d+)?\b", _hr_norm(main))):
        out.append(E("quantity", m.group(0), m.span(), "value", "value_time" if structured else "static"))
    for m in list(_cm.COMPARATIVE.finditer(main)):
        out.append(E("comparison", m.group(0), m.span(), "value", "static"))
    for m in list(_cm.EVENT_ORDER.finditer(main)):
        out.append(E("ordering", m.group(0), m.span(), "predicate", "static"))
    for w in _cm.residue_words(main):
        out.append(E("residue", w, (0, 0), "predicate", "residue"))
    return tuple(out)


def read_claim_frame(clause: str, claim: Optional["_cm.Claim"] = None) -> Optional["_cm.ClaimFrame"]:
    """The D-098 claim frame of a specific yes/no question (None when no claim is read): the unchanged C10 claim with the main
    question (a safe leading frame stripped), its force, frequency, the asked population, the universal slots, the asked
    enumeration, the ledger and the anchor roots (the asked verb, plus the entitlement family for an entitlement ask). D-099: a
    specific claim's leading hedge ("In general, ...") is read as its frequency operator (two frequency operators: no frame)."""
    claim = claim if claim is not None else read_claim(clause)
    if claim is None:
        return None
    if _C12_HEDGE and is_specific_claim(clause):
        hedge, rest = _c12_split_hedge(clause)
        if hedge is not None:
            return _c12_hedged_frame(hedge, read_claim_frame(rest, claim))
    low = cue_text(clause.lower()).strip()
    main = _INCLUSIVE_ASIDE.sub("", _main_question(low)).rstrip("?.! ")
    frame_text = None
    if _C11_FRAMES:
        stripped = strip_leading_frame(main)
        if stripped != main:
            frame_text, main = main[:len(main) - len(stripped)].strip(" ,"), stripped
    freq, freq_words, freq_span = _cm.frequency_of(main)
    excl_spans = [m.span() for m in _cm.EXCLUSIVE_LIST_MARKERS.finditer(main)]
    anchor_roots = _claim_verb_roots(claim) | (_ENTITLEMENT_ROOTS if _ENTITLEMENT_ASK.search(claim.text) else frozenset())
    if not anchor_roots:                                          # "Is the allowance paid only with Dean approval?": the participle
        anchor_roots = frozenset(_root(w) for w, _s, _e in _cm.tokens(main)[1:] if _participle(w) or w in _C11_IRREGULAR_PARTICIPLES)
    slots = _asked_slots(claim, main, [(x, len(main)) for x, _y in excl_spans] + ([freq_span] if freq is not None else []), anchor_roots)
    items: Tuple[_cm.EnumItem, ...] = ()
    if excl_spans and any(e.focus in (_cm.ExclusivityFocus.OBJECT, _cm.ExclusivityFocus.POPULATION) for e in claim.exclusivity):
        items = _cm.enumeration_items(main, excl_spans[0][1], _root)
    subject = _asked_population(claim, main)
    return _cm.ClaimFrame(claim=claim, main=main, frame=frame_text,
                          force=_cm.force_of(claim.modality, claim.subject.quantifier if claim.subject is not None else None, main),
                          frequency=freq, frequency_text=freq_words, subject=subject, slots=slots, items=items,
                          ledger=_c12_ledger(claim, main, freq_span, excl_spans, slots), anchor_roots=anchor_roots)


def _c11_own(text: str, roots: FrozenSet[str], claim_roots: FrozenSet[str], main_roots: FrozenSet[str] = frozenset()) -> bool:
    """A restriction that is part of the claim's own content: at least half of its content words are the claim's. D-101: given
    ``main_roots`` (the roots of the statement's main clause) the same test must also hold on the words that distinguish the
    restriction from that clause - the asked entity restated inside a condition ("... the housing allowance if they receive
    university housing") is no evidence that the question asked the condition - and a restriction with no distinguishing word
    is not the claim's own. D-101 can only withdraw the D-098 exemption, never grant one."""
    content = {r for r in (roots or _roots(text, frozenset())) if r not in _PERSON_FAMILY}
    own = bool(content) and 2 * len({r for r in content if any(_same_root(r, c) for c in claim_roots)}) >= len(content)
    if own and _C12_OWN_CONTENT and main_roots:
        distinct = {r for r in content if not any(_same_root(r, m) for m in main_roots)}
        own = bool(distinct) and 2 * len({r for r in distinct if any(_same_root(r, c) for c in claim_roots)}) >= len(distinct)
    return own


def _c11_main_roots(statement: "_cm.PredicateStatement") -> FrozenSet[str]:
    """D-101: the roots of a statement's main clause - its text with the texts of its restrictions blanked. A restriction the
    text does not contain is not blanked, so its words stay main-clause words and it fails closed as unasked."""
    main = statement.text
    for r in statement.restrictions:
        if r.text and r.text in main:
            main = main.replace(r.text, " ")
    return _roots(main, frozenset())


def _c11_unasked(statement: "_cm.PredicateStatement", claim_roots: FrozenSet[str], low_q: str, claim: Optional["_cm.Claim"] = None
                 ) -> FrozenSet["_cm.RestrictionKind"]:
    """The restriction kinds of a statement the question does not ask itself (self-content excluded: "Is renewal subject to
    successful evaluations?", "Is approval required for ...?", "... unless the Dean objects?"). D-101: own content is also judged
    on the words that distinguish a restriction from the statement's main clause; a restriction the D-098 rule exempted and that
    restates one of the claim's own asked conditions keeps its exemption (the condition dimension still verifies its role)."""
    RK = _cm.RestrictionKind
    out = set()
    main = _c11_main_roots(statement) if _C12_OWN_CONTENT else frozenset()
    for r in statement.restrictions:
        if r.kind in (RK.NECESSARY, RK.EXCEPTION, RK.SUFFICIENT, RK.TRIGGER, RK.GATE):
            if _c11_own(r.text, r.roots, claim_roots, main):
                continue
            if _C12_OWN_CONTENT and claim is not None and _c12_asked_condition(r, claim) and _c11_own(r.text, r.roots, claim_roots):
                continue                                          # D-101 keeps the D-098 exemption of a condition the question asks
        if r.kind is RK.NECESSARY and "subject to" in r.text.lower() and "subject to" in low_q:
            continue
        if r.kind is RK.SPECIAL and _cm.SPECIAL_CIRCUMSTANCES.search(low_q):
            continue
        if r.kind is RK.EXCEPTION and re.search(r"\bunless\b|\bexcept\b", low_q):
            continue
        if r.kind is RK.HEDGE and r.text.lower() in low_q:
            continue
        out.add(r.kind)
    return frozenset(out)


def _c11_copular(claim: "_cm.Claim") -> bool:
    """A copular attribute claim ("Is the probation period one year?", "Is the leave up to 30 days?"): no asked action, so only a
    hedge bound to the attribute restricts it (an occasion restriction of another predicate does not)."""
    return claim.action is None


#: The head phrase of a subject ends at its first preposition or relative word ("Termination of faculty contracts for cause":
#: "faculty" modifies the object of "of"; the subject names no role).
_SUBJECT_HEAD_CUT = re.compile(r"\b(?:of|for|to|in|on|at|with|from|by|during|under|over|including|who|whom|whose|which|that)\b", re.I)


def _head_roles(subject_text: str) -> FrozenSet[str]:
    """The roles a subject names in its head phrase: the role spans read on the whole text (a role name with a preposition,
    "Office of Human Resources", stays one span) that start before the head cut."""
    text = subject_text.strip()
    cut = _SUBJECT_HEAD_CUT.search(text, 1)
    end = cut.start() if cut is not None else len(text)
    return frozenset(role for a, _b, role in ASK_GATE.lexicon.spans(text) if a < end)


def _c11_value_stated(hr: Optional[Detail], numbers: FrozenSet[str], claim: "_cm.Claim", statement: "_cm.PredicateStatement") -> bool:
    """The asked fact on the statement: the C9 HR fact reader on its sentence ("sixty calendar days") and every asked number
    inside its span."""
    return (hr is None or hr.check(statement.sentence)) and numbers <= statement.numbers


def _composite_checks(frame: "_cm.ClaimFrame", clause: str) -> Dict[str, Callable[["_cm.ClaimFrame", "_cm.PredicateStatement"], Optional["_cm.DimensionResult"]]]:
    """The HR readers of the joint comparison, one per dimension (``claim_model.DIMENSIONS``); each returns None when the
    dimension does not apply to the claim (or its switch is off) and evaluates the ONE statement it is given. The C9 polarity
    contracts, the C9 HR fact reader, the C10 Population breadth, exclusivity and condition-role readers are reused as they are."""
    claim = frame.claim
    C, R, D = _cm.Compatibility, _cm.RestrictionKind, _cm.DimensionResult
    text = strip_leading_frame(clause) if frame.frame else clause
    low_q = cue_text(text.lower())
    claim_roots = _roots(text, frozenset())
    affirmative = claim.polarity is _cm.Polarity.AFFIRMATIVE
    polarity_family = [d for d in (predicate_detail(text), negative_detail(text), polarity_detail(text), assertion_detail(text)) if d is not None]
    hr = hr_fact_detail(text, concepts)
    numbers = frozenset(re.findall(r"\d+(?:\.\d+)?", _hr_norm(claim.text)))
    structured = _claim_structured(claim)
    entity = _entity_roots(claim)
    asked = frame.subject
    breadth_claim = replace(claim, subject=asked) if asked is not None else claim
    # The asked actor binds only as the claim's own subject: an actor the ask parse found elsewhere ("Does termination require
    # Chancellor approval?": the Chancellor is in the object) is no subject to compare.
    actor = claim.actor if claim.actor and claim.subject is not None and claim.actor in _head_roles(claim.subject.text) else None

    def polarity(f, s):
        if not polarity_family or s.anchor.kind is _cm.AnchorKind.FALLBACK:
            return None
        if all(d.check(s.sentence) for d in polarity_family):
            return D("polarity", C.COMPATIBLE)
        return D("polarity", C.UNKNOWN, "the statement negates the asked predicate or states it only inside a condition")

    def subject(f, s):
        # An asked actor binds by contradiction: a statement whose own subject names ANOTHER role in its head phrase (the ask
        # reader's role lexicon: "HR" = "Human Resources Department") and the asked one nowhere states the predicate of someone
        # else and cannot support. The subject text runs from the clause start, so it may hold a leading antecedent or an
        # unsplit clause before the subject ("If the Council does not approve ..., the Chancellor"): an asked actor named in it
        # is tolerated. An agentless, pronoun or entity subject stays unknown and is tolerated; entity identity is the C9
        # content contracts' (unit level).
        if not actor:
            return None
        text = s.subject_text or ""
        roles = _head_roles(text)
        if roles and actor not in roles and actor not in ASK_GATE.lexicon.roles_in(text):
            return D("subject", C.UNKNOWN, "the statement states it of another actor (%s), not of the asked %s"
                     % (", ".join(sorted(r.replace("_", " ") for r in roles)), actor.replace("_", " ")))
        return D("subject", C.COMPATIBLE)

    def predicate(f, s):
        if s.anchor.kind in (_cm.AnchorKind.OBJECT, _cm.AnchorKind.FALLBACK):
            return None
        return D("predicate", C.COMPATIBLE)

    f_slots = frame.slots if _C11_SLOT_UNIVERSAL else ()

    def slot_results(kind: str, s) -> Optional[_cm.DimensionResult]:
        wanted = [x for x in f_slots if x.slot == kind]
        if not wanted:
            return None
        for x in wanted:
            if _cm.slot_universal_compatibility(x, s.noun_phrases, _same_root) is not C.COMPATIBLE:
                return D(kind if kind == "object" else "scope", C.UNKNOWN, "'%s' is not stated as an explicit, unlimited universal" % x.text)
            if s.member_exceptions:
                return D(kind if kind == "object" else "scope", C.UNKNOWN, "the statement excepts part of '%s'" % x.text)
        return D(kind if kind == "object" else "scope", C.COMPATIBLE)

    def obj(f, s):
        return slot_results("object", s)

    def modality(f, s):
        if not affirmative or claim.modality is _cm.Modality.NONE:
            return None
        comp = _cm.modal_compatibility(claim.modality, s.modality)
        if comp in (C.COMPATIBLE, C.NOT_APPLICABLE):
            return D("modality", C.COMPATIBLE)
        return D("modality", C.UNKNOWN, "%s asked, %s stated for the asked predicate" % (claim.modality.value, s.modality.value))

    def scope(f, s):
        results = []
        if affirmative and asked is not None and asked.head and not asked.is_label and asked.quantifier in (
                _cm.Quantifier.UNIVERSAL, _cm.Quantifier.GENERIC, _cm.Quantifier.EXISTENTIAL):
            sub = s.subject
            if sub is None or not sub.head:
                comp = _cm.scope_compatibility(asked.quantifier, None, _cm.Breadth.SAME_OR_BROADER)
            else:
                comp = _cm.scope_compatibility(asked.quantifier, sub.quantifier, _breadth(breadth_claim, claim_roots, sub))
            if comp not in (C.COMPATIBLE, C.NOT_APPLICABLE):
                return D("scope", C.UNKNOWN, "the asked %s population (%s) is not stated with its breadth (statement subject: %s)"
                         % (asked.quantifier.value, asked.text, sub.text if sub is not None else "none"))
            results.append(D("scope", C.COMPATIBLE))
        r = slot_results("subject", s)
        if r is not None:
            if r.compatibility is not C.COMPATIBLE:
                return r
            results.append(r)
        return results[0] if results else None

    def frequency(f, s):
        if frame.frequency is None or not _C11_FREQUENCY:
            return None
        kinds = _c11_unasked(s, claim_roots, low_q, claim)
        comp = _cm.frequency_compatibility(frame.frequency, s.predicate, kinds, s.negative)
        if comp is C.COMPATIBLE:
            return D("frequency", C.COMPATIBLE)
        return D("frequency", C.UNKNOWN, "'%s' asked; the statement states no %s frequency or restricts it (%s)"
                 % (frame.frequency_text, "explicit universal" if frame.frequency is _cm.FrequencyClass.UNIVERSAL else "matching",
                    ", ".join(sorted(k.value for k in kinds)) or "none"))

    def condition(f, s):
        results = []
        for c in claim.conditions:
            role, gate = _stated_role(s.sentence, c.roots, entity)
            comp = _cm.condition_role_compatibility(c.role, role, gate)
            if (c.role is _cm.ConditionRole.NECESSARY and comp is not C.COMPATIBLE) or comp is C.UNKNOWN:
                return D("condition", C.UNKNOWN, "the asked %s condition (%s) is not stated in that role" % (c.role.value, c.text.strip()))
            results.append(D("condition", C.COMPATIBLE))
        if _C11_UNCONDITIONAL and affirmative:
            kinds = _c11_unasked(s, claim_roots, low_q, claim)
            if _c11_copular(claim):
                kinds = kinds & {R.HEDGE}                         # a copular attribute ("Is the probation period one year?")
            if _cm.restriction_compatibility(frame.force, kinds) is not C.COMPATIBLE:
                bad = sorted(k.value for k in kinds & _cm.FORCE_RESTRICTIONS[frame.force])
                return D("condition", C.UNKNOWN, "the %s claim is stated only under a restriction (%s)" % (frame.force.value.replace("_", " "), ", ".join(bad)))
            results.append(D("condition", C.COMPATIBLE))
        return results[0] if results else None

    def exclusivity(f, s):
        if not claim.exclusivity:
            return None
        for e in claim.exclusivity:
            av = _value_tokens(claim.text) if e.focus is _cm.ExclusivityFocus.VALUE else frozenset()
            if _exclusivity_local(s.text, e, claim, av) is not C.COMPATIBLE:
                return D("exclusivity", C.UNKNOWN, "the asked exclusivity (%s, %s) is not stated in the statement" % (e.marker, e.focus.value))
            if _C11_ENUMERATION and e.focus in (_cm.ExclusivityFocus.OBJECT, _cm.ExclusivityFocus.POPULATION) and frame.items:
                if _cm.enumeration_compatibility(frame.items, s.exclusive_items) is not C.COMPATIBLE:
                    return D("exclusivity", C.UNKNOWN, "the asked list (%s) is not the stated exclusive list" % "; ".join(i.text for i in frame.items))
        if _C11_EXCEPTION_WINDOW and s.hedges:
            return D("exclusivity", C.UNKNOWN, "the exclusivity is stated with a hedge (%s)" % ", ".join(s.hedges))
        return D("exclusivity", C.COMPATIBLE)

    def value_time(f, s):
        if not structured or (hr is None and not numbers):
            return None
        if not _c11_value_stated(hr, numbers, claim, s):
            return D("value_time", C.UNKNOWN, "the asked fact or value (%s) is not stated in the statement" % (", ".join(sorted(numbers)) or "fact"))
        return D("value_time", C.COMPATIBLE)

    checks = {"polarity": polarity, "subject": subject, "predicate": predicate, "object": obj, "modality": modality, "scope": scope,
              "frequency": frequency, "condition": condition, "exclusivity": exclusivity, "value_time": value_time}
    if is_specific_claim(clause):
        checks.update(_c12_dimension_checks(frame, clause, checks))    # D-099
    return checks


def _bound_to_predicate(s: "_cm.PredicateStatement") -> bool:
    return s.anchor.kind in (_cm.AnchorKind.VERB, _cm.AnchorKind.ENTITLEMENT, _cm.AnchorKind.PREDICATE)


def _claim_window(quote: str, frame: "_cm.ClaimFrame") -> Tuple["_cm.Compatibility", str]:
    """The window vetoes of the composite (D-098), read from every statement of the returned quote bound to the asked
    predicate: alternatives of an exclusive claim (exclusivity owns them: a repetition of an exclusive value - by the C10
    value binding -, another time for the same predicate and object - never a relative offset -, an explicit waiver of an
    exclusive condition, another actor, population or object), and the exceptions, negations and hedges that defeat a
    universal claim. A veto removes coverage; it never adds support."""
    claim = frame.claim
    C = _cm.Compatibility
    low_q = cue_text(claim.text.lower())
    statements: List[_cm.PredicateStatement] = []
    for ui, u in enumerate(_evidence._local_units(quote)):
        statements.extend(read_clause_statements(_evidence._LIST_MARKER.sub("", u), frame, ui, fallback=False))
    bound = [s for s in statements if _bound_to_predicate(s)]
    if _C11_ALTERNATIVES:
        for e in claim.exclusivity:
            if e.focus is _cm.ExclusivityFocus.VALUE:
                asked_values = _value_tokens(claim.text)
                bind = _value_bind_roots(claim.text, asked_values)
                need = max(1, (len(bind) + 1) // 2)
                before = False
                for sentence in _unit_sentences(quote):
                    on = (not asked_values or bool(asked_values & _value_tokens(sentence)) or (bool(bind) and _overlap(bind, _roots(sentence, frozenset())) >= need)
                          or (before and bool(_ANAPHOR.match(sentence))))
                    before = on
                    m = _cm.REPETITION.search(sentence) if on else None
                    if m and not _cm.REPETITION_NEGATED.search(sentence) and m.group(0).lower() not in low_q:
                        return C.CONTRADICTED, "an alternative to the exclusive value is stated (%s)" % m.group(0)
            elif e.focus is _cm.ExclusivityFocus.TIME:
                asked = _roots(" ".join(e.target), frozenset())
                obj = frozenset(r for r in claim.obj if r not in asked)
                for s in bound:
                    if s.anchor.kind is not _cm.AnchorKind.VERB or (obj and not _overlap(obj, _roots(s.text, frozenset()))):
                        continue
                    states_asked = any(_overlap(asked, t.roots) for t in s.times)
                    for t in s.times:
                        if t.offset or _overlap(asked, t.roots):
                            continue
                        if not states_asked or re.search(r"\bor\b", s.text[max(0, s.text.find(t.text) - 6):s.text.find(t.text)], re.I):
                            return C.CONTRADICTED, "another time is stated for the asked predicate (%s)" % t.text
            elif e.focus is _cm.ExclusivityFocus.CONDITION:
                for s in bound:
                    m = _cm.WAIVER.search(s.text)
                    if m and m.group(0).lower() not in low_q:
                        return C.CONTRADICTED, "the exclusive condition is waived (%s)" % m.group(0)
            else:
                target = _roots(" ".join(e.target), frozenset())
                for s in bound:
                    m = _cm.ADDITIVE.search(s.text)
                    filler = frozenset(r for r in _roots(s.subject_text, frozenset()) if r not in _PERSON_FAMILY)
                    if m and not s.negative and filler and not _overlap(filler, target):
                        return C.CONTRADICTED, "another valid filler is stated for the asked predicate (%s)" % m.group(0)
    if _C11_EXCEPTION_WINDOW:
        universal = frame.force in (_cm.Force.UNIVERSAL_QUANTIFIED, _cm.Force.UNIVERSAL_FREQUENCY)
        for s in bound:
            if frame.force is _cm.Force.UNIVERSAL_QUANTIFIED and frame.slots:
                if s.member_exceptions:
                    return C.CONTRADICTED, "the evidence excepts part of the universal (%s)" % s.member_exceptions[0].strip()
                if s.negative and any(_overlap(frozenset({x.head}), _roots(s.text, frozenset())) for x in frame.slots):
                    return C.CONTRADICTED, "the evidence negates the asked predicate for part of the universal"
            if frame.force is _cm.Force.UNIVERSAL_FREQUENCY and any(r.kind in (_cm.RestrictionKind.EXCEPTION, _cm.RestrictionKind.SPECIAL) for r in s.restrictions):
                return C.CONTRADICTED, "the evidence states an exception to the asked frequency"
            if (universal or claim.exclusivity) and s.hedges and not _cm.HEDGES.search(low_q):
                return C.CONTRADICTED, "the evidence hedges the asked universal or exclusive claim (%s)" % ", ".join(s.hedges)
    if _C12_ALTERNATIVE_FILLER and is_specific_claim(claim.text):
        filler = _c12_alternative_filler(quote, frame)
        if filler:
            return C.CONTRADICTED, filler
    return C.COMPATIBLE, ""


def _c11_best(frame: "_cm.ClaimFrame", checks, quote: str) -> Tuple[Optional["_cm.PredicateStatement"], Tuple["_cm.DimensionResult", ...]]:
    """The statement of the returned quote that comes closest to the claim: the most compatible dimensions, then the latest
    first failure in dimension order, then the earliest clause."""
    best, best_key, best_results = None, None, ()
    for ui, u in enumerate(_evidence._local_units(quote)):
        for s in read_clause_statements(_evidence._LIST_MARKER.sub("", u), frame, ui):
            results = _cm.compare_claim_to_statement(frame, s, checks, ledger=_C11_LEDGER)
            ok = sum(1 for r in results if r.compatibility is _cm.Compatibility.COMPATIBLE)
            fail = _cm.first_failure(results)
            pos = _cm.DIMENSIONS.index(fail.dimension) if fail is not None else len(_cm.DIMENSIONS)
            key = (ok, pos, -ui, -s.sentence_index, -s.clause_index)
            if best_key is None or key > best_key:
                best, best_key, best_results = s, key, results
    return best, best_results


def explain_claim(frame: "_cm.ClaimFrame", checks, quote: Optional[str]) -> str:
    """The missing message of the composite: the best statement's first failed dimension (fixed order) with its reason and
    clause, or the window veto, or that no statement states the asked predicate."""
    if not quote:
        return "the asked claim is not stated in the returned evidence"
    s, results = _c11_best(frame, checks, quote)
    if s is None:
        return "no statement of the returned evidence states the asked predicate for the asked subject"
    fail = _cm.first_failure(results)
    if fail is None:
        comp, why = _claim_window(quote, frame)
        if comp is _cm.Compatibility.CONTRADICTED:
            return "the returned evidence vetoes the asked claim: %s" % why
        return "the asked claim is not stated in one statement of the returned evidence"
    return "the asked claim is not stated in one statement: %s - %s (statement: \"%s\")" % (fail.dimension.replace("_", "/"), fail.reason,
                                                                                         " ".join(s.text.split())[:110])


def _coverage_key(covered: FrozenSet[str], keys: FrozenSet[str]) -> Tuple[int, bool]:
    """How a candidate ranks: by the C1-C10 details it covers; the D-098 composite only breaks the tie between candidates that
    cover every C1-C10 detail (there it decides support). With the composite gated on every other detail of its unit
    (``_supporting_only``), a partial answer keeps the C10 units, quotes and ranking."""
    c10 = covered - {"claim"}
    return len(c10), "claim" in covered and (keys - {"claim"}) <= c10


def claim_detail(clause: str, claim: Optional["_cm.Claim"]) -> Optional[Detail]:
    """D-098 composite: ONE detail for every claim read (``_C11_JOINT``); a unit supports it only when one of its predicate
    statements is compatible on every dimension at once (``claim_model.compare_claim_to_statement``), and the returned quote
    can veto it (``_claim_window``). The D-097 per-dimension details stay as explainers."""
    if not _C11_JOINT or claim is None:
        return None
    frame = read_claim_frame(clause, claim)
    if frame is None:
        return None
    checks = _composite_checks(frame, clause)
    local_cache: Dict[str, bool] = {}
    window_cache: Dict[str, _cm.Compatibility] = {}

    def local(unit: str) -> bool:
        if unit not in local_cache:
            ok = (_cm.Compatibility.COMPATIBLE, _cm.Compatibility.NOT_APPLICABLE)
            local_cache[unit] = any(_cm.combine(_cm.compare_claim_to_statement(frame, s, checks, ledger=_C11_LEDGER)) in ok
                                    for s in read_clause_statements(unit, frame))
        return local_cache[unit]

    def window(quote: str) -> "_cm.Compatibility":
        if quote not in window_cache:
            window_cache[quote] = _claim_window(quote, frame)[0]
        return window_cache[quote]

    return Detail("claim", "the asked claim stated in one statement", _ClaimCheck(local, window, lambda quote: explain_claim(frame, checks, quote)))


def _supporting_only(composite: Detail, others: Sequence[Detail]) -> Detail:
    """The composite as a clause uses it (D-098): it counts on a unit only together with every other detail of the clause on
    that same unit, so it decides support and never which unit or quote a partial answer reports (the shared quote selection
    and unit coverage rank units by the number of details they cover). ``ungated`` keeps its own reading for the pronoun pair
    of ``unit_coverage``, whose grounding is read with the sentence before."""
    inner = composite.check
    rest = tuple(others)
    memo: Dict[str, bool] = {}

    def gated(unit: str) -> bool:
        if unit not in memo:
            memo[unit] = inner(unit) and all(d.check(unit) for d in rest)
        return memo[unit]

    return Detail(composite.family, composite.label, _ClaimCheck(gated, inner.window if inner.windowed else None, inner.explain, ungated=inner.local))


# ----------------------------------------------------------------------------------------------------------------------
# D-099: specific HR claims require structured proof. A specific yes/no claim (direct, led by a phrase or a condition, or
# wrapped in a conversational frame) is SUPPORTED only through the D-098 composite of a read claim and frame, with every
# operator consumed by a concrete check on ONE statement; with no claim or frame it carries a detail no unit can cover. A
# general question (overview, what / how / where, procedure, open request) keeps the C10 / C11 contract unchanged. Each rule
# has a switch for the mutation harness.
# ----------------------------------------------------------------------------------------------------------------------
_C12_GATE = True                 # a specific claim with no structured proof never supports
_C12_WRAPPERS = True             # conversational wrappers are normalised before the claim is read
_C12_HEDGE = True                # a leading semantic hedge ("In general,") is kept as the claim's frequency operator
_C12_LEDGER = True               # no "static" operator: each is consumed by a concrete check on the statement
_C12_CLAUSE = True               # a coordinated clause with a new explicit subject and a finite lexical verb is its own statement
_C12_OBJECT_ANCHOR = True        # an object mention with its own subject is no statement of a claim whose verb is stated
_C12_COMPARE = True              # comparisons with a number are evaluated (restated, or one unit-bound value), else unknown
_C12_CONDITION = True            # an if / unless restriction governing the asked predicate leaves an unconditional claim unknown
_C12_COPULAR = True              # no copular bypass of the force x restriction table
_C12_NEGATIVE_CONDITION = True   # the unconditional check binds a negative claim too
_C12_MODIFIER = True             # an asked important modifier is stated; another of its family vetoes
_C12_OBJECT_BINDING = True       # in a sentence of several clauses the asked object is stated in the chosen clause
_C12_ALTERNATIVE_FILLER = True   # an exclusive value is vetoed by another filler of the same slot
_C12_LEADS = True                # a lead focus clause split off by the coordinator ("In general") rejoins its yes/no clause
_C12_FALLBACK_POLARITY = True    # a FALLBACK statement (no anchor in its unit) meets the claim's polarity
_C12_PAREN_NUMERAL = True        # D-100: a bracketed numeral repeating its number words is dropped before the readers
_C12_ENTITY_SUBJECT = True       # D-100: the asked entity of a claim with no actor and no object is stated in the supporting clause
_C12_OWN_CONTENT = True          # D-101: a restriction is the claim's own content only through the words that distinguish it from its clause
_C12_WHEN_CONDITION = True       # D-102: a "when" / "whenever" restriction of the asked predicate is a material condition, as "if" is

_C12_GREETING = re.compile(r"^\s*(?:hi|hello|hey|dear\s+(?:sir|madam|hr|team)|good\s+(?:morning|afternoon|evening)|greetings)\b[\s,!.:;-]*", re.I)
_C12_POLITE = re.compile(r"^\s*(?:please|kindly|quick\s+question|one\s+(?:more\s+)?question|a\s+question|question|sorry|"
                         r"just\s+(?:to\s+)?(?:check|confirm|clarify|ask|double[\s-]check)|just\s+checking|quick\s+check|by\s+the\s+way|"
                         r"(?:also|so|ok(?:ay)?|well|actually)(?=\s*,))\b[\s,!.:;-]*", re.I)
_C12_CONFIRM = re.compile(r"^\s*(?:please\s+)?(?:confirm|clarify|verify|check)\s*[:,\-]\s*", re.I)
_C12_TRAIL_POLITE = re.compile(r"[\s,]+(?:please|thanks|thank\s+you)\s*([?.!]*)\s*$", re.I)
#: An embedded yes/no question: a conversational lead (a person word or a softener) with a verb of knowing or asking, then
#: "whether" / "if" and the embedded clause ("Can you tell me whether ...", "I'd like to know if ...", "Any idea whether ...").
_C12_EMBED = re.compile(
    r"^(?P<lead>[^?]{0,80}?\b(?:know|tell|telling|confirm|check|checking|ask|asking|wonder|wondering|clarify|verify|sure|unsure|idea|"
    r"find\s+out|inform|advise|curious|understand|enquire|inquire|explain)\b(?:(?!\b(?:what|how|where|who|whom|which|when|why)\b)[^?]){0,24}?)"
    r"\s+(?:whether(?:\s+or\s+not)?|if)\s+(?P<body>.+)$", re.I)
_C12_EMBED_LEAD = re.compile(r"\b(?:i|i['’]d|i['’]m|me|you|we|us|my|our|anyone|anybody|someone|just|any|wondering|curious|please|kindly)\b", re.I)
#: An embedded declarative put up for confirmation ("Is it true that ...", "Am I right that ...", "Can you confirm that ...").
_C12_TRUE_THAT = re.compile(
    r"^(?P<lead>[^?]{0,60}?)\b(?:(?:is|was)\s+it\s+(?:true|correct|right|accurate|the\s+case|so)|am\s+i\s+(?:right|correct)(?:\s+in\s+thinking)?|"
    r"would\s+it\s+be\s+(?:true|correct|right|accurate)(?:\s+to\s+say)?|do\s+i\s+understand(?:\s+correctly)?|"
    r"(?:can|could|would|will)\s+you\s+(?:please\s+)?confirm|(?:please\s+)?confirm|i\s+(?:heard|read|was\s+told|understand|believe|think|assume))"
    r"\s+that\s+(?P<body>.+)$", re.I)
#: A declarative with a confirmation tag ("Termination ... doesn't require ..., right?", "I heard X. Is that right?").
_C12_HEARD = re.compile(r"^\s*(?:i\s+(?:heard|read|was\s+told|understand|think|believe|assume)\s+(?:that\s+)?)?(?P<body>[^?]+?)[\s.,;:!-]*"
                        r"(?:is\s+(?:that|this|it)\s+(?:right|correct|true|so)|right|correct|true)\s*\?\s*$", re.I)
_C12_TAG = re.compile(r"[\s.,;:!-]*(?:is\s+(?:that|this|it)\s+(?:right|correct|true|so)|right|correct|true)\s*\?\s*$", re.I)
_C12_WH_START = re.compile(r"^\s*(?:what|how|where|who|whom|whose|which|when|why)\b", re.I)
_C12_ANY_WH = re.compile(r"\b(?:what|how|where|whom|whose|which|when|why)\b", re.I)
#: An open request worded as a polite yes/no question ("Can you tell me about maternity leave?") or a wh- embedding.
_C12_OPEN_REQUEST = re.compile(
    r"^\s*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+)?(?:tell|explain|describe|give|show|list|walk|summari[sz]e|outline|share|"
    r"send|provide)\b(?!\s+(?:me\s+|us\s+)?(?:whether|if)\b)|"
    r"^\s*(?:(?:do|does)\s+(?:you|anyone)\s+know|(?:can|could|would)\s+you\s+(?:please\s+)?tell\s+me|i\s+(?:want|would\s+like|['’]d\s+like|need)\s+"
    r"to\s+know)\s+(?:what|how|where|who|whom|which|when|why|about)\b|"
    r"^\s*(?:is|are)\s+there\s+(?:any\s+)?(?:information|details|guidance|policy|policies|rules?|regulations?|provisions?|procedures?)\s+"
    r"(?:about|on|regarding|for|concerning)\b", re.I)
_C12_AUX = re.compile(r"(?:is|are|was|were|am|can|could|may|might|must|shall|should|will|would|does|do|did|has|have|had|need|needs)$", re.I)
_C12_LEAD = re.compile(r"^\s*(?P<lead>[^,?]{1,70}),\s*(?P<rest>.+)$")
#: A question led by a negative contraction ("Aren't ...", "Cannot ..."): specific, never read (its polarity is ambiguous).
_C12_NEG_LEAD = re.compile(r"^\s*(?:aren['’]t|isn['’]t|wasn['’]t|weren['’]t|doesn['’]t|don['’]t|didn['’]t|can['’]t|cannot|couldn['’]t|won['’]t|"
                           r"wouldn['’]t|shouldn['’]t|mustn['’]t|needn['’]t|hasn['’]t|haven['’]t|hadn['’]t|mayn['’]t)\b", re.I)
#: Negative contractions of an embedded clause, read as the auxiliary and "not".
_C12_NEG_CONTRACTION = {"doesn't": "does", "don't": "do", "didn't": "did", "isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were",
                        "can't": "can", "cannot": "can", "won't": "will", "wouldn't": "would", "mustn't": "must", "shouldn't": "should",
                        "couldn't": "could", "hasn't": "has", "haven't": "have", "hadn't": "had", "needn't": "need", "mayn't": "may"}
#: Reporting verbs: a declarative that reports another clause is not read as one yes/no question.
_C12_REPORTING = re.compile(r"(?:says|said|states|stated|mentions|mentioned|notes|noted|indicates|indicated|specifies|specified|stipulates|"
                            r"stipulated|shows|showed|claims|claimed|suggests|suggested|implies|implied)", re.I)
#: Words that may stand between a subject and its verb, or that end in -s without being one.
_C12_TRANSPARENT = frozenset("also only still even just always sometimes perhaps besides unless thus towards afterwards whereas nevertheless "
                             "regardless nowadays across".split())
#: A leading semantic hedge: an operator, kept as the claim's frequency.
_C12_HEDGE_LEAD = re.compile(r"^\s*(?P<hedge>in\s+general|generally(?:\s+speaking)?|typically|usually|normally|as\s+a\s+rule|by\s+default|"
                             r"ordinarily|customarily|in\s+principle|commonly|mostly|often|frequently|rarely|seldom|sometimes|occasionally|"
                             r"always|never)\s*,\s*(?P<rest>.+)$", re.I)
_C12_SPECIFIC_CACHE: Dict[str, bool] = {}


def _c12_peel(text: str) -> Tuple[str, bool]:
    """The question without its greeting / courtesy openers and a trailing "please" / "thanks", and whether any was removed."""
    t, changed = text, False
    for _ in range(4):
        m = _C12_GREETING.match(t) or _C12_POLITE.match(t) or _C12_CONFIRM.match(t)
        if m and m.end() < len(t) and t[m.end():].strip():
            t, changed = t[m.end():], True
            continue
        break
    t2 = _C12_TRAIL_POLITE.sub(lambda m: m.group(1) or "?", t)
    return t2, changed or t2 != t


#: D-100 (Step 4.14, P2): a numeral in brackets right after up to four words ("two (2) years", "thirty (30) days",
#: "five thousand (5,000) Dirhams"). Number words and their values; "thousand" added to the shared table.
_C12_PAREN_NUMERAL_RX = re.compile(r"(?P<words>\b[a-z]+(?:[\s-]+(?:and\s+)?[a-z]+){0,3})\s*\(\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*\)", re.I)
_C12_NUMBER_VALUES: Dict[str, int] = dict({w: int(d) for w, d in _grounding._NUMBER_WORDS.items()}, thousand=1000)


def _c12_number_words_value(words: Sequence[str]) -> Optional[int]:
    """The value of a run of number words ("two" -> 2, "thirty thousand" -> 30000, "one thousand five hundred" -> 1500,
    "one hundred and twenty" -> 120); None for an empty run or a word that is no number word."""
    total = current = 0
    seen = False
    for w in words:
        if w == "and":
            continue
        v = _C12_NUMBER_VALUES.get(w)
        if v is None:
            return None
        seen = True
        if v == 1000:
            total, current = total + (current or 1) * v, 0
        elif v == 100:
            current = (current or 1) * v
        else:
            current += v
    return total + current if seen else None


def _c12_drop_parenthesised_numeral(text: str) -> str:
    """D-100: "two (2) years" -> "two years". A bracketed numeral that only repeats the number words right before it is
    dropped from the text the readers read, so the C9 HR fact reader binds the number to its period or unit instead of the
    bare digit matching any "2" of a statement; a numeral that says something else ("Article (3)", "one (2)") is kept."""
    def repl(m: "re.Match[str]") -> str:
        words = m.group("words")
        try:
            n = float(m.group("num").replace(",", ""))
        except ValueError:
            return m.group(0)
        toks = re.findall(r"[a-z]+", words.lower())
        k = len(toks)
        while k > 0 and (toks[k - 1] in _C12_NUMBER_VALUES or toks[k - 1] == "and"):
            k -= 1
        value = _c12_number_words_value(toks[k:]) if k < len(toks) else None
        return words if (value is not None and float(value) == n) else m.group(0)
    return _C12_PAREN_NUMERAL_RX.sub(repl, text)


def _c12_lemma(word: str) -> Optional[str]:
    """The bare form of a third-person verb ("drafts" -> draft, "applies" -> apply), when one reads as a verb."""
    w = word.lower()
    for cut in ("ies", "es", "s"):
        if w.endswith(cut) and len(w) > len(cut) + 2:
            stem = w[:-len(cut)] + ("y" if cut == "ies" else "")
            if _verb_like(stem) or _known_verb(stem):
                return stem
            if cut == "es" and (_verb_like(w[:-1]) or _known_verb(w[:-1])):
                return w[:-1]
    return None


def _c12_transparent(word: str) -> bool:
    return word in _C12_TRANSPARENT or _cm.frequency_of(word)[0] is not None


def _c12_plural(phrase: str) -> bool:
    last = phrase.lower().strip(" ,;:.").split()[-1] if phrase.strip() else ""
    return (last.endswith("s") and not last.endswith(("ss", "us", "is"))) or last in ("faculty", "staff", "people")


def _c12_finite(word: str, plural: bool) -> bool:
    """The word right after an explicit subject is its finite verb, agreeing with it: an auxiliary or copula, a known HR verb,
    a third-person form after a singular subject, a bare verb after a plural one. A bare word after a singular or nameless
    noun phrase is no verb ("Dental Medicine and Veterinary Medicine")."""
    w = word.lower().strip(",;:.")
    if not w:
        return False
    if _C12_AUX.fullmatch(w):
        return True
    if w in _NEVER_A_VERB or (w.endswith("ly") and len(w) > 4) or w.endswith(("ing", "ed")) or "'" in w or "’" in w:
        return False
    if _known_verb(w) and (plural or w.endswith("s")):
        return True
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and not plural:
        stem = _c12_lemma(w)
        return bool(stem) and not _NOT_BARE_VERB.search(stem) and (_known_verb(stem) or _verb_like(stem))
    return plural and _verb_like(w) and not _NOT_BARE_VERB.search(w)


def _c12_interrogative(body: str) -> Optional[str]:
    """A declarative embedded clause as the yes/no question it asks: its first auxiliary or copula (outside a relative clause)
    fronted (a negative contraction as its auxiliary with "not" kept), else do-support for a finite lexical verb agreeing with
    its subject; None when no such verb is found, or the clause reports another one ("The Handbook says X is Y") - unreadable."""
    text = body.strip().rstrip("?.! ").replace("’", "'")
    words = text.split()
    if len(words) < 2 or _C12_WH_START.match(text):
        return None
    for i in range(1, min(len(words), 14)):
        w = words[i].lower().strip(",;:")
        prev = words[i - 1].lower().strip(",;:")
        if _C12_REPORTING.fullmatch(w):
            return None
        if prev in ("who", "which", "that", "whom", "whose", "to"):
            continue
        if w in _C12_NEG_CONTRACTION:
            return "%s %s not %s?" % (_C12_NEG_CONTRACTION[w], " ".join(words[:i]), " ".join(words[i + 1:]))
        if _C12_AUX.fullmatch(w):
            return "%s %s %s?" % (w, " ".join(words[:i]), " ".join(words[i + 1:]))
    for i in range(1, min(len(words), 14)):
        w = words[i].lower().strip(",;:")
        if _c12_transparent(w):
            continue
        subject = [x for x in words[:i] if not _c12_transparent(x.lower().strip(",;:"))]
        prev = subject[-1].lower().strip(",;:") if subject else ""
        if not prev or prev in _NEVER_A_VERB or prev in ("who", "which", "that", "whom", "whose", "to"):
            continue
        if _root(w) in _PERSON_FAMILY or _cm.POPULATION_HEADS.match(w):
            continue                                              # a person noun is never the verb ("faculty members receive")
        plural = _c12_plural(" ".join(subject))
        if not _c12_finite(w, plural):
            continue
        if w.endswith("s") and not w.endswith(("ss", "us", "is")) and not plural:
            stem = _c12_lemma(w)
            if stem:
                return "does %s %s %s?" % (" ".join(words[:i]), stem, " ".join(words[i + 1:]))
            continue
        return "do %s %s %s?" % (" ".join(words[:i]), w, " ".join(words[i + 1:]))
    return None


def _c12_embedded(text: str) -> Optional[Tuple[str, str]]:
    """(the embedded yes/no clause of a wrapped question, the leading phrase before its wrapper), or None."""
    m = _C12_EMBED.match(text)
    if m is not None and _C12_EMBED_LEAD.search(m.group("lead")):
        lead = m.group("lead")
        return m.group("body"), (lead[:lead.rindex(",") + 1].strip() if "," in lead else "")
    m = _C12_TRUE_THAT.match(text)
    if m is not None:
        lead = m.group("lead")
        return m.group("body"), (lead[:lead.rindex(",") + 1].strip() if "," in lead else "")
    if "?" in text and not _cm.ASK_LEAD.match(text):
        m = _C12_HEARD.match(text)
        if m is not None:
            return m.group("body"), ""
    return None


def _c12_declarative(text: str) -> Tuple[bool, Optional[str]]:
    """(specific, reading) of a declarative put as a question ("Faculty members must apply for the award?"): no auxiliary lead,
    no wh- word, no leading phrase, ends with "?". It is specific when a finite verb is read or seen; its reading is None when
    it is not one yes/no question. A verbless topic ("Sick leave?") is general."""
    if (_cm.ASK_LEAD.match(text) or _C12_ANY_WH.search(text) or _C12_OPEN_REQUEST.match(text) or _C12_LEAD.match(text)
            or _C12_NEG_LEAD.match(text) or not text.rstrip().endswith("?")):
        return False, None
    q = _c12_interrogative(text)
    words = [w.lower().strip(",;:?.!") for w in text.split()[1:]]
    seen = any(_C12_AUX.fullmatch(w) or _C12_REPORTING.fullmatch(w) or w in _C12_NEG_CONTRACTION for w in words)
    return (q is not None or seen), q


def _c12_led(text: str) -> Optional["re.Match[str]"]:
    """A question led by a phrase or a condition and a comma whose rest is itself a yes/no question ("When the department has
    fewer than three ..., is ...?", "For new faculty, can you tell me whether ...?")."""
    m = _C12_LEAD.match(text)
    if m is not None and (_cm.ASK_LEAD.match(m.group("rest")) or _C12_NEG_LEAD.match(m.group("rest"))):
        return m
    return None


def is_specific_claim(clause: str, _depth: int = 0) -> bool:
    """D-099: the clause asks a specific yes/no claim - direct ("Must ...?"), led by a phrase or a condition and a comma, embedded
    in a conversational wrapper ("Hi, can you tell me whether ...", "Is it true that ...", "..., right?"), led by a negative
    contraction, or a declarative put as a question. An overview, a what / how / where / who question and an open request
    ("Can you tell me about ...?") are general and keep the C10 / C11 contract."""
    if _depth == 0 and clause in _C12_SPECIFIC_CACHE:
        return _C12_SPECIFIC_CACHE[clause]
    t, _changed = _c12_peel(clause)
    m = _c12_led(t)
    if m is not None and _depth < 3:
        out = is_specific_claim(m.group("rest"), _depth + 1)
    elif _C12_WH_START.match(t):
        out = False
    elif _C12_NEG_LEAD.match(t) or _c12_embedded(t) is not None:
        out = True
    elif _C12_OPEN_REQUEST.match(t):
        out = False
    elif _cm.ASK_LEAD.match(t) or _c12_declarative(t)[0]:
        out = True
    else:
        m = _C12_LEAD.match(t)
        out = bool(m) and _depth < 3 and is_specific_claim(m.group("rest"), _depth + 1)
    if _depth == 0:
        if len(_C12_SPECIFIC_CACHE) > 20000:
            _C12_SPECIFIC_CACHE.clear()
        _C12_SPECIFIC_CACHE[clause] = out
    return out


def specific_claim_text(clause: str, _depth: int = 0) -> Optional[str]:
    """D-099: the text the claim reader reads for a specific claim - greeting and courtesy openers removed, an embedded or
    declarative yes/no clause turned into the yes/no question it asks (a leading phrase before a wrapper kept, every semantic
    operator kept); None when the claim cannot be read (no readable finite verb, a negative-contraction lead). A clause with
    no wrapper is returned unchanged. D-100: a bracketed numeral that repeats its number words ("two (2) years") is dropped
    from every text returned (never from a None)."""
    text = _c12_claim_text(clause, _depth)
    return _c12_drop_parenthesised_numeral(text) if (_C12_PAREN_NUMERAL and text is not None) else text


def _c12_claim_text(clause: str, _depth: int = 0) -> Optional[str]:
    """The D-099 reading of ``specific_claim_text`` before the D-100 numeral normalisation."""
    if not _C12_WRAPPERS:
        return clause
    t, changed = _c12_peel(clause)
    unchanged = t.strip() if changed else clause
    if _C12_NEG_LEAD.match(t):
        return None
    m = _c12_led(t)
    if m is not None and _depth < 3:
        rest = _c12_claim_text(m.group("rest"), _depth + 1)
        if rest is None:
            return None
        return "%s, %s" % (m.group("lead").strip(), rest) if rest != m.group("rest") else unchanged
    if _C12_WH_START.match(t):
        return unchanged
    found = _c12_embedded(t)
    if found is not None:
        q = _c12_interrogative(_C12_TAG.sub("", found[0]))
        return ("%s %s" % (found[1], q)) if (q is not None and found[1]) else q
    specific, q = _c12_declarative(t)
    return q if specific else unchanged


def _c12_attach_leads(clauses: Sequence[str]) -> List[str]:
    """Class F: the coordinator splits "In general, must Deans ...?" at its comma into the focus clauses "In general" and "must
    Deans ..."; a focus clause that is only such a lead - at most four words, no verb, no question mark - is re-attached to the
    yes/no clause right after it, so a semantic prefix stays the claim's operator and an unknown lead keeps it unreadable."""
    if not _C12_LEADS:
        return list(clauses)
    out: List[str] = []
    i = 0
    while i < len(clauses):
        c = clauses[i]
        words = c.split()
        nxt = clauses[i + 1] if i + 1 < len(clauses) else ""
        if (nxt and len(words) <= 4 and "?" not in c and not _C12_WH_START.match(c) and not _C11_VERB_GROUP.search(c)
                and not any(_C12_AUX.fullmatch(w.lower().strip(",;:")) for w in words) and (_cm.ASK_LEAD.match(nxt) or _C12_NEG_LEAD.match(nxt))):
            out.append("%s, %s" % (c.strip(" ,;:"), nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return out


def _c12_split_hedge(text: str) -> Tuple[Optional[str], str]:
    """(a leading semantic hedge, the yes/no question after it) - "In general, must ...?" -> ("In general", "must ...?")."""
    if not _C12_HEDGE:
        return None, text
    m = _C12_HEDGE_LEAD.match(text)
    if m is not None and _cm.ASK_LEAD.match(m.group("rest")):
        return m.group("hedge"), m.group("rest")
    return None, text


def _c12_read_claim(text: str) -> Optional["_cm.Claim"]:
    """The structured claim of a normalised specific question: read after a leading hedge (the hedge stays in the claim text
    and becomes the frame's frequency operator)."""
    hedge, rest = _c12_split_hedge(text)
    claim = read_claim(rest)
    return replace(claim, text=text) if (claim is not None and hedge is not None) else claim


def _c12_hedged_frame(hedge: str, frame: Optional["_cm.ClaimFrame"]) -> Optional["_cm.ClaimFrame"]:
    """The frame of a hedged claim: the hedge is its frequency operator (consumed by the frequency dimension); an unknown hedge
    or a second frequency operator leaves no usable frame."""
    if frame is None:
        return None
    cls, words, _span = _cm.frequency_of(hedge.lower())
    if cls is None or frame.frequency is not None:
        return None
    force = _cm.Force.UNIVERSAL_FREQUENCY if cls in (_cm.FrequencyClass.UNIVERSAL, _cm.FrequencyClass.NEGATIVE_UNIVERSAL) else frame.force
    ledger = frame.ledger + (_cm.LedgerEntry("frequency", words, (0, 0), "occasion", "frequency"),)
    return replace(frame, frequency=cls, frequency_text=words, force=force, ledger=ledger)


_C12_UNREAD_LABEL = "the specific claim read as a structured claim"


def _c12_unread(_quote: Optional[str]) -> str:
    return "the specific claim could not be read as a structured claim (no complete structured proof)"


def _specific_claim_support_gate(clause: str) -> Tuple[Optional["_cm.Claim"], List[Detail], Optional[Detail]]:
    """D-099, the single owner of the support contract of a clause's claim: (claim, D-097 details, the claim detail). A general
    question keeps the C10 / C11 reading unchanged. A specific claim is read from its wrapper-normalised text (a leading hedge
    kept as its frequency operator); its claim detail is the D-098 composite of that claim, or - with no claim or no frame -
    a detail no unit can cover: SPECIFIC CLAIM + NO COMPLETE STRUCTURED PROOF = NEVER SUPPORTED."""
    if not (_C12_GATE and is_specific_claim(clause)):
        claim = read_claim(clause)
        return claim, structured_details(clause, claim), claim_detail(clause, claim)
    text = specific_claim_text(clause)
    claim = _c12_read_claim(text) if text is not None else None
    details = structured_details(text if text is not None else clause, claim)
    composite = claim_detail(text, claim) if claim is not None else None
    if composite is None:
        composite = Detail("claim", _C12_UNREAD_LABEL, _ClaimCheck(lambda unit: False, None, _c12_unread))
    return claim, details, composite


# ---- the ledger: no static operator --------------------------------------------------------------------------------------
_C12_TIME_UNITS = frozenset(_cm.TIME_NOUNS) - {"academic"}
_C12_TIME_UNIT_ROOTS = frozenset(_root(w) for w in "day days week weeks month months year years semester semesters hour hours".split())
_C12_TIME_EQUIVALENTS = {
    "year": r"(?:annual(?:ly)?|yearly|per\s+(?:academic\s+)?year|(?:each|every|a)\s+(?:academic\s+)?year|per\s+annum)",
    "month": r"(?:monthly|per\s+month|(?:each|every|a)\s+month)", "week": r"(?:weekly|per\s+week|(?:each|every|a)\s+week)",
    "semester": r"(?:per\s+semester|(?:each|every|a)\s+(?:regular\s+)?semester)", "day": r"(?:daily|per\s+day|(?:each|every|a)\s+day)"}
_C12_MONTHS = frozenset("january february march april may june july august september october november december".split())


def _c12_time_head(main: str, span: Tuple[int, int]) -> Optional[str]:
    """The real time noun of a quantifier phrase read as a time slot ("each academic year" -> year); None when its time word is
    only a modifier ("all University-level academic administrators")."""
    for w, _s, _e in _cm.tokens(main[span[1]:])[:4]:
        if w in _cm.NP_BREAK or w in _cm.UNIVERSAL_WORDS or w in _cm.EXISTENTIAL_WORDS or w in _cm.PROPORTIONAL_WORDS:
            return None
        if w in _C12_TIME_UNITS:
            return w
    return None


def _c12_time_distributives(main: str) -> List[Tuple[str, str]]:
    return [(w, head) for w, slot, _h, span in _cm.quantifier_phrases(main) if slot == "time"
            for head in [_c12_time_head(main, span)] if head]


def _c12_time_restated(head: str, text: str) -> bool:
    """An asked time distributive restated on a statement ("each year": annually, per year, every academic year; "each
    September": in September of each year)."""
    low = text.lower()
    if head in _C12_MONTHS and re.search(r"\b" + head + r"\b", low) and re.search(r"\b(?:(?:each|every)\s+(?:academic\s+)?year|annually|yearly)\b", low):
        return True
    key = next((k for k in _C12_TIME_EQUIVALENTS if head.startswith(k)), None)
    if key and re.search(r"\b" + _C12_TIME_EQUIVALENTS[key] + r"\b", low):
        return True
    return bool(re.search(r"\b(?:each|every)\s+(?:\w+\s+)?" + re.escape(head[:5]), low))


def _c12_modifier_universals(frame: "_cm.ClaimFrame") -> List["_cm.NounPhrase"]:
    """Universal noun phrases read as time slots although they name no time ("all University-level academic administrators"),
    read by the shared noun-phrase reader; the scope dimension checks them as explicit, unlimited universals."""
    main = frame.main
    toks = _cm.tokens(main)
    words = [w for w, _s, _e in toks]
    starts = [s for _w, s, _e in toks]
    stop = _c11_np_stop(frame.claim)
    out = []
    for w, slot, _h, span in _cm.quantifier_phrases(main):
        if slot != "time" or _cm.quantifier_of(w) is not _cm.Quantifier.UNIVERSAL or _c12_time_head(main, span) or span[0] not in starts:
            continue
        head, mods = _cm.noun_phrase_after(words, starts.index(span[0]), stop)
        if head:
            out.append(_cm.NounPhrase(text=" ".join((w,) + mods + (head,)), head=_root(head),
                                      modifiers=frozenset(_root(m) for m in mods if m not in _cm.ITEM_DROP), quantifier=_cm.Quantifier.UNIVERSAL,
                                      slot="subject", span=span))
    return out


def _c12_ledger(claim: "_cm.Claim", main: str, freq_span: Tuple[int, int], excl_spans: Sequence[Tuple[int, int]],
                slots: Sequence["_cm.NounPhrase"]) -> Tuple["_cm.LedgerEntry", ...]:
    """The D-098 ledger hardened for a specific claim (D-099): no "static" entry - an operator is consumed only when a concrete
    dimension evaluates it on the chosen statement (a time distributive and a quantity by the value dimension, a universal named
    by a modifier by the scope dimension, an event order by its literal restatement, a negative existential by the polarity
    dimension); any other is "unverified" and never consumed. Every asked comparison with a number is an entry of its own."""
    ledger = _claim_ledger(claim, main, freq_span, excl_spans, slots)
    if not (_C12_LEDGER and is_specific_claim(claim.text)):
        return ledger
    out = []
    for e in ledger:
        if e.handler != "static":
            out.append(e)
            continue
        if e.cls == "scope":
            q = _cm.quantifier_of(e.text)
            if e.slot == "time":
                handler = "value_time" if _c12_time_head(main, e.span) else ("scope" if q is _cm.Quantifier.UNIVERSAL else "unverified")
            elif claim.polarity is _cm.Polarity.NEGATIVE and q is _cm.Quantifier.EXISTENTIAL and e.slot != "proportional":
                handler = "polarity"
            else:
                handler = "unverified"
        elif e.cls in ("quantity", "comparison"):
            handler = "value_time"
        elif e.cls == "ordering":
            handler = "residue"
        else:
            handler = "unverified"
        out.append(replace(e, handler=handler))
    if _C12_COMPARE:
        out.extend(_cm.LedgerEntry("comparison", c.text, c.span, "value", "value_time") for c in _c12_comparisons(main))
    return tuple(out)


# ---- class B: comparisons ------------------------------------------------------------------------------------------------
#: Comparison operators, longest first, with their relation.
_C12_COMPARISON_OPERATORS: Tuple[Tuple[str, str], ...] = (
    ("no more than", "<="), ("not more than", "<="), ("no longer than", "<="), ("not longer than", "<="), ("not exceeding", "<="),
    ("not to exceed", "<="), ("not exceed", "<="), ("does not exceed", "<="), ("a maximum of", "<="), ("maximum of", "<="), ("at most", "<="),
    ("up to", "<="), ("capped at", "<="),
    ("no less than", ">="), ("not less than", ">="), ("no fewer than", ">="), ("not fewer than", ">="), ("a minimum of", ">="),
    ("minimum of", ">="), ("at least", ">="),
    ("in excess of", ">"), ("more than", ">"), ("greater than", ">"), ("higher than", ">"), ("longer than", ">"), ("larger than", ">"),
    ("exceeding", ">"), ("exceeds", ">"), ("exceeded", ">"), ("exceed", ">"), ("over", ">"), ("above", ">"),
    ("less than", "<"), ("fewer than", "<"), ("lower than", "<"), ("shorter than", "<"), ("smaller than", "<"), ("under", "<"),
    ("below", "<"), ("beneath", "<"),
    ("equal to", "=="), ("exactly", "=="),
)


def _c12_operator_pattern() -> str:
    return "|".join(re.escape(p).replace(r"\ ", r"\s+") for p, _r in _C12_COMPARISON_OPERATORS)


_C12_COMPARISON = re.compile(r"\b(?P<op>" + _c12_operator_pattern() + r")\s+(?P<pre>(?:the\s+age\s+of\s+|aed\s+|dhs?\.?\s+|dirhams?\s+))?"
                             r"(?P<num>\d[\d,]*(?:\.\d+)?)\b", re.I)
#: The comparison operator a stated number carries ("up to an additional 30 days", "a maximum of thirty days").
_C12_STATED_OPERATOR = re.compile(r"\b(?P<op>" + _c12_operator_pattern() + r")(?:\s+(?:an?|the|additional|further|extra|aed|dhs?\.?|dirhams?|"
                                  r"age\s+of))*\s*$", re.I)
_C12_CURRENCY = frozenset({"aed", "dhs", "dh", "dirham", "dirhams"})
_C12_RANGE_UNIT = r"(?P<ua>\s*(?:percent|%|days?|weeks?|months?|years?|hours?))?"
#: A stated range of one unit ("AED 2,000 to AED 3,000", "10% to 20%", "between 10 and 20 days"); "and" separates a range only
#: after "between" ("2 years and 3 months" is no range).
_C12_RANGES = (re.compile(r"\b(?:from\s+)?(?:aed\s+|dhs?\.?\s+)?(?P<a>\d+(?:\.\d+)?)" + _C12_RANGE_UNIT + r"\s*(?:to|-|–)\s*(?P<pb>aed\s+|dhs?\.?\s+)?"
                          r"(?P<b>\d+(?:\.\d+)?)\b", re.I),
               re.compile(r"\bbetween\s+(?:aed\s+|dhs?\.?\s+)?(?P<a>\d+(?:\.\d+)?)" + _C12_RANGE_UNIT + r"\s+and\s+(?P<pb>aed\s+|dhs?\.?\s+)?"
                          r"(?P<b>\d+(?:\.\d+)?)\b", re.I))
_C12_UNIT_SKIP = frozenset(("calendar", "working", "consecutive", "full", "academic", "credit", "additional", "extra", "further"))
_C12_POPULATION_WORDS = frozenset("aged age children child dependents dependent".split())
#: The word before a comparison that makes it the asked value itself ("is over", "composed of over", "receive more than").
_C12_VALUE_PREVIOUS = frozenset("is are was were be been being of for by to can could may might must shall should will would do does did "
                                "receive receives get gets earn earns total totals reach reaches last lasts cost costs exceed exceeds".split())


@dataclass(frozen=True)
class _C12Comparison:
    """An asked comparison: its relation, number (and its normalised text), unit (a root, "currency" or "age"), span and text;
    ``exact_only`` when it restricts a population rather than stating the asked value (only its exact restatement proves it)."""
    relation: str
    number: float
    number_text: str
    unit: Optional[str]
    span: Tuple[int, int]
    text: str
    exact_only: bool


def _c12_unit_after(text: str, end: int, pre: Optional[str] = None) -> Optional[str]:
    """The unit a number is bound to: a currency word or "the age of" before it, else the first noun after it ("calendar",
    "working", ... skipped); None when a verb or a function word follows."""
    if pre:
        first = pre.strip().split()[0].rstrip(".").lower()
        if first in _C12_CURRENCY:
            return "currency"
        if "age" in pre.lower().split():
            return "age"
    for w in re.findall(r"[a-z][a-z'-]*", text[end:end + 40].lower())[:3]:
        if w in _C12_CURRENCY:
            return "currency"
        if w in _C12_UNIT_SKIP:
            continue
        if w in _cm.NP_BREAK or w in _cm.NP_STOP_WORDS or _C11_VERB_GROUP.fullmatch(w) or w in ("be", "been", "being", "and", "or"):
            return None
        return _root(w)
    return None


def _c12_unit_of(norm: str, start: int, end: int) -> Optional[str]:
    before = re.findall(r"[a-z]+", norm[max(0, start - 12):start].lower())
    if before and before[-1] in _C12_CURRENCY:
        return "currency"
    if before[-2:] == ["age", "of"]:
        return "age"
    return _c12_unit_after(norm, end)


def _c12_same_unit(stated: Optional[str], asked: Optional[str]) -> bool:
    return asked is None or stated == asked or bool(stated and asked and _same_root(stated, asked))


def _c12_population_comparison(norm: str, m: "re.Match[str]", unit: Optional[str], verbs: FrozenSet[str]) -> bool:
    """A comparison that limits a population or an item ("faculty members over the age of 65 ... offered", "children under 18",
    "faculty with more than ten years of service receive"): right after a person word with an age, a year or no unit; not one
    right after its predicate ("is over", "receive more than"); else one followed by a verb."""
    prev = (re.findall(r"[a-z]+", norm[max(0, m.start() - 25):m.start()]) or [""])[-1]
    person = prev in _C12_POPULATION_WORDS or _root(prev) in _PERSON_FAMILY or bool(_cm.POPULATION_HEADS.match(prev))
    if person and (unit is None or unit == "age" or unit in _C12_TIME_UNIT_ROOTS):
        return True
    if prev in _C12_VALUE_PREVIOUS or any(_same_root(_root(prev), v) for v in verbs):
        return False
    return any(_C11_VERB_GROUP.fullmatch(w) or w.endswith("ed") or any(_same_root(_root(w), v) for v in verbs)
               for w in re.findall(r"[a-z]+", norm[m.end():]))


def _c12_comparisons(main: str, verbs: FrozenSet[str] = frozenset()) -> List[_C12Comparison]:
    """Every comparison operator of the asked claim followed by a number (class B: under, over, below, above, less / more than,
    at least / most, no more / less than, equal to, higher / lower than, up to, ...)."""
    norm = _hr_norm(main)
    out = []
    for m in _C12_COMPARISON.finditer(norm):
        relation = dict(_C12_COMPARISON_OPERATORS).get(" ".join(m.group("op").lower().split()))
        if relation is not None:
            unit = _c12_unit_after(norm, m.end(), m.group("pre"))
            number = m.group("num").replace(",", "")
            out.append(_C12Comparison(relation, float(number), number, unit, m.span(), m.group(0), _c12_population_comparison(norm, m, unit, verbs)))
    return out


def _c12_stated_values(text: str, unit: Optional[str]) -> List[Tuple[str, float, float, Optional[str]]]:
    """The values a statement states in the asked unit: ("range", low, high, None) or ("value", v, v, the relation of the
    comparison operator the number carries, None when plain)."""
    norm = _hr_norm(text)
    items: List[Tuple[str, float, float, Optional[str]]] = []
    used: List[Tuple[int, int]] = []
    for rx in _C12_RANGES:
        for m in rx.finditer(norm):
            stated = "currency" if m.group("pb") else _c12_unit_of(norm, m.start("b"), m.end("b"))
            if m.group("ua") and not _c12_same_unit(_root(m.group("ua").strip().replace("%", "percent")), stated):
                continue
            if _c12_same_unit(stated, unit) and not any(x <= m.start("a") < y for x, y in used):
                items.append(("range", float(m.group("a")), float(m.group("b")), None))
                used.append((m.start("a"), m.end("b")))
    relations = dict(_C12_COMPARISON_OPERATORS)
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", norm):
        if any(x <= m.start() < y for x, y in used) or not _c12_same_unit(_c12_unit_of(norm, m.start(), m.end()), unit):
            continue
        op = _C12_STATED_OPERATOR.search(norm[max(0, m.start() - 40):m.start()])
        items.append(("value", float(m.group(1)), float(m.group(1)), relations.get(" ".join(op.group("op").lower().split())) if op else None))
    return items


def _c12_comparison_holds(c: _C12Comparison, statement: "_cm.PredicateStatement") -> bool:
    """An asked comparison on one statement (safe deterministic comparison): the same comparison restated (same relation and
    number), or - for an asked value - exactly one stated value of the same unit that entails it (a plain value, a range, a
    value with its own bound); a statement that also states a number of another time unit, or anything else, is unknown."""
    items = _c12_stated_values(statement.text, c.unit)
    n = c.number
    if any(kind == "value" and rel == c.relation and a == n for kind, a, _b, rel in items):
        return True
    if c.exact_only or len(items) != 1:
        return False
    if c.unit in _C12_TIME_UNIT_ROOTS:
        norm = _hr_norm(statement.text)
        if any(u in _C12_TIME_UNIT_ROOTS and not _c12_same_unit(u, c.unit)
               for u in (_c12_unit_of(norm, m.start(), m.end()) for m in re.finditer(r"\b\d+(?:\.\d+)?\b", norm))):
            return False
    kind, a, b, rel = items[0]
    if kind == "range":
        return {"<": b < n, ">": a > n, "<=": b <= n, ">=": a >= n}.get(c.relation, False)
    if rel is None or rel == "==":
        return {"<": a < n, ">": a > n, "<=": a <= n, ">=": a >= n, "==": a == n}[c.relation]
    if rel in ("<", "<="):
        return {"<": a < n or (rel == "<" and a <= n), "<=": a <= n}.get(c.relation, False)
    return {">": a > n or (rel == ">" and a >= n), ">=": a >= n}.get(c.relation, False)


# ---- class A: lexical-verb clause boundary, object binding --------------------------------------------------------------
_C12_COORDINATOR = re.compile(r",?\s+(?:and|or|but|yet)\s+", re.I)
_C12_SUBJECT_DETERMINERS = frozenset("the a an this that these those its their his her our each every all some".split())


def _c12_subject_end(text: str) -> Optional[int]:
    """Where the explicit subject at the start of ``text`` ends when its finite verb follows: a pronoun, a role name, or a
    determiner noun phrase of up to four words with no comma; None when ``text`` opens with no subject and verb (a bare noun
    phrase is an enumeration, never a clause)."""
    words = text.split()
    if len(words) < 2:
        return None
    w0 = words[0].lower().strip(",;:")
    if w0 in _PRONOUNS or w0 in ("it", "he", "she"):
        w1 = words[1].lower().strip(",;:.")
        ok = bool(_C12_AUX.fullmatch(w1)) or (w1 not in _NEVER_A_VERB and (_known_verb(w1) or _verb_like(w1) or bool(_c12_lemma(w1))))
        return len(words[0]) if ok else None
    spans = ASK_GATE.lexicon.spans(text[:90])
    if spans and spans[0][0] <= 4:
        rest = text[spans[0][1]:]
        after = rest.split()
        ok = bool(after) and not rest.lstrip().startswith(",") and _c12_finite(after[0], _c12_plural(text[:spans[0][1]]))
        return spans[0][1] if ok else None
    if w0 not in _C12_SUBJECT_DETERMINERS:
        return None
    for k in range(2, min(len(words), 5)):
        if any(c in words[k - 1] for c in ",;:") or words[k - 1].lower().strip(",") in _NEVER_A_VERB:
            return None
        if _c12_finite(words[k], _c12_plural(" ".join(words[1:k]))):
            return len(" ".join(words[:k]))
    return None


def _c12_left_finite(text: str) -> bool:
    """The left side of a coordinator is a finite clause: a verb group, or an explicit subject and its verb at its start or
    after a leading comma phrase."""
    if _C11_VERB_GROUP.search(text):
        return True
    starts = [0] + [m.end() for m in re.finditer(r",\s*", text)][:2]
    return any(_c12_subject_end(text[s:]) is not None for s in starts)


def _c12_clause_spans(sentence: str) -> List[Tuple[int, int, bool]]:
    """The finite clauses of a sentence for a specific claim (class A): the D-098 clauses, each also split after "and / or / but
    / yet" when the left side is finite and the right side opens with a NEW explicit subject and its finite lexical verb
    ("Professors receive ..., and lecturers receive ...", "... or they email ..."); an enumeration and an elided subject never
    split (no general parser: an ambiguous coordinator keeps the clause whole)."""
    base = _clause_spans(sentence)
    if not _C12_CLAUSE:
        return base
    out: List[Tuple[int, int, bool]] = []
    for a, b, elided in base:
        seg = sentence[a:b]
        cuts, last = [], 0
        for m in _C12_COORDINATOR.finditer(seg):
            if _c12_left_finite(seg[last:m.start()]) and _c12_subject_end(seg[m.end():]) is not None:
                cuts.append(m)
                last = m.end()
        if not cuts:
            out.append((a, b, elided))
            continue
        starts = [0] + [m.end() for m in cuts]
        ends = [m.start() for m in cuts] + [len(seg)]
        out.extend((a + x, a + y, elided if i == 0 else False) for i, (x, y) in enumerate(zip(starts, ends)) if seg[x:y].strip(" ,"))
    return out


def _c12_label(s: "_cm.PredicateStatement") -> str:
    """A short verbless list label the statement's clause follows after a colon ("Visiting Appointment: Involves ...", "o Dean:
    Monthly compensation ..."): the topic the clause states; else ""."""
    i = s.sentence.find(s.text)
    before = s.sentence[:i].rstrip() if i > 0 else ""
    if not before.endswith(":"):
        return ""
    seg = before[:-1]
    label = seg[max(seg.rfind(";"), seg.rfind(":")) + 1:].strip()
    return label if label and len(label.split()) <= 6 and not _C11_VERB_GROUP.search(label) else ""


def _c12_pronoun_antecedent(s: "_cm.PredicateStatement") -> str:
    """The subject of the sentence's previous clause when the statement's own object is a pronoun right after its anchor verb
    ("The probation period lasts one year, yet the Chancellor may shorten it"); else ""."""
    if not s.anchor.word or not re.search(r"\b" + re.escape(s.anchor.word) + r"\s+(?:it|them)\b", s.text, re.I):
        return ""
    i = s.sentence.find(s.text)
    previous = [(a, b) for a, b, _e in _c12_clause_spans(s.sentence) if b <= i + 1 and (a, b) != (i, i + len(s.text))]
    if not previous:
        return ""
    text = s.sentence[previous[-1][0]:previous[-1][1]].strip(" ,")
    end = _c12_subject_end(text)
    return text[:end] if end else ""


def _c12_object_roots(claim: "_cm.Claim", frame: "_cm.ClaimFrame") -> FrozenSet[str]:
    """The asked object's roots: no number, no person word, not the asked verb and not a subject word (the C9 object reader may
    keep them: "Does the visiting appointment cover a probation year?" -> appoint, cover, probat, year)."""
    verbs = frame.anchor_roots | _claim_verb_roots(claim)
    subject = _roots(claim.subject.text, frozenset()) if claim.subject is not None and claim.subject.text else frozenset()
    return frozenset(r for r in claim.obj if not r[0].isdigit() and r not in _cm.NUMBER_WORDS and r not in _PERSON_FAMILY
                     and not any(_same_root(r, v) for v in verbs) and r not in subject)


#: D-100 (Step 4.14, P1): words that never name the asked entity - determiners, possessives, the possessive "s" the tokeniser
#: splits off, prepositions, coordinators and currency words - and the words that end the entity phrase: a comparison
#: operator's words (not its articles), an exclusivity or frequency word and a universal, existential or proportional quantifier.
_C12_ENTITY_STOP: FrozenSet[str] = frozenset("the a an this that these those its their his her our my your s of for to in on at by with from per "
                                             "under over during within after before and or".split()) | _C12_CURRENCY
_C12_ENTITY_BREAK: FrozenSet[str] = ((frozenset(w for p, _r in _C12_COMPARISON_OPERATORS for w in p.split()) - frozenset({"a", "an", "the", "of", "to", "in", "than"}))
                                     | frozenset({"only", "also", "always", "never", "still"}) | _cm.UNIVERSAL_WORDS | _cm.EXISTENTIAL_WORDS
                                     | _cm.PROPORTIONAL_WORDS)


def _c12_asked_entity(frame: "_cm.ClaimFrame") -> Tuple[str, FrozenSet[str]]:
    """D-100: the asked entity of a specific claim that names no role actor and no object ("Is the housing allowance paid
    once?" -> "the housing allowance" and its roots): the words of the claim's main clause after its leading auxiliary, up to
    the first anchor root, number, number word, comparison operator, quantifier, verb or exclusivity word; its content roots
    without determiners, prepositions, currency and person words (an empty phrase or no root leaves the claim unbound). When
    the claim reader did not recognise the asked verb (its anchor fell to an object noun) the phrase may still carry that
    verb; the subject reader therefore vetoes only on a root named elsewhere in the statement's sentence."""
    toks = [w for w, _s, _e in _cm.tokens(frame.main)]
    if toks and _C12_AUX.fullmatch(toks[0]):
        toks = toks[1:]
    out: List[str] = []
    for w in toks:
        if w[0].isdigit() or w in _cm.NUMBER_WORDS or w in _C12_NUMBER_VALUES or w in _C12_ENTITY_BREAK:
            break
        if any(_same_root(_root(w), v) for v in frame.anchor_roots) or _C11_VERB_GROUP.fullmatch(w) or _C12_AUX.fullmatch(w):
            break
        out.append(w)
    phrase = " ".join(out).strip()
    roots = frozenset(r for r in _roots(phrase, frozenset())
                      if r not in _PERSON_FAMILY and not r[0].isdigit() and r not in _C12_ENTITY_STOP and len(r) > 1)
    return phrase, roots


# ---- class D: important modifiers ---------------------------------------------------------------------------------------
_C12_MODIFIERS = {"department": "level", "departmental": "level", "college": "level", "university": "level", "clinical": "appointment",
                  "visiting": "appointment", "part-time": "appointment", "full-time": "appointment", "new": "new", "senior": "senior",
                  "dependent": "dependent", "academic": "academic"}
_C12_MODIFIER_FAMILIES = {"level": frozenset({"department", "college", "university", "institutional"}),
                          "appointment": frozenset({"clinical", "visiting", "part-time", "full-time", "regular", "adjunct", "joint"}),
                          "new": frozenset({"new", "existing", "current", "continuing"}), "senior": frozenset({"senior", "junior"}),
                          "dependent": frozenset({"dependent"}), "academic": frozenset({"academic", "administrative", "non-academic"})}
_C12_MODIFIER_CANON = {"departmental": "department", "parttime": "part-time", "fulltime": "full-time", "nonacademic": "non-academic"}
_C12_HEAD_STOP = (_cm.NP_BREAK | _cm.NP_STOP_WORDS | _cm.UNIVERSAL_WORDS | _cm.EXISTENTIAL_WORDS | _cm.PROPORTIONAL_WORDS
                  | frozenset("under over above below up per within during after before until than upon".split()))


@dataclass(frozen=True)
class _C12Modifier:
    """An asked important modifier: the modifier, its family, the head it modifies, whether it sits in the subject, and the
    acronyms of the capitalised name it belongs to ("University Medical Clinic" -> umc)."""
    word: str
    family: str
    head: str
    in_subject: bool
    acronyms: FrozenSet[str]


def _c12_words(text: str) -> List[str]:
    """Lower-case words with "part-time", "full-time", "non-academic" whole and other hyphenated compounds split."""
    t = text.lower()
    t = re.sub(r"\bpart[\s-]+time\b", "parttime", t)
    t = re.sub(r"\bfull[\s-]+time\b", "fulltime", t)
    t = re.sub(r"\bnon[\s-]+academic\b", "nonacademic", t)
    t = t.replace("-", " ").replace("–", " ")
    return [_C12_MODIFIER_CANON.get(w, w) for w in re.findall(r"[a-z0-9][a-z0-9'’]*", t)]


def _c12_acronyms(text: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for m in re.finditer(r"\b(?:[A-Z][\w'’-]*\s+(?:(?:of|and|for|the)\s+)?){1,5}[A-Z][\w'’-]*", text):
        words = m.group(0).split()
        while words and (_C12_AUX.fullmatch(words[0].lower()) or words[0].lower() in ("of", "and", "for", "the")):
            words = words[1:]
        if len(words) < 2:
            continue
        acronym = "".join(w[0] for w in words if w.lower() not in ("of", "and", "for", "the")).lower()
        for w in words:
            out.setdefault(w.lower(), set()).add(acronym)
    return out


def _c12_asked_modifiers(frame: "_cm.ClaimFrame") -> List[_C12Modifier]:
    """The asked important modifiers with their head - on the subject too: a person population narrowed by one ("visiting
    faculty") is not proved by a statement of the unmodified population ("Faculty members may apply ...": the handbook may
    exclude it in the next sentence), so the modifier must be stated (the C10 Population breadth alone would accept it)."""
    claim = frame.claim
    words = _c12_words(frame.main)
    verbs = frame.anchor_roots
    acronyms = _c12_acronyms(claim.text)

    def stop(x: str) -> bool:
        return x in _C12_HEAD_STOP or x[0].isdigit() or bool(_C11_VERB_GROUP.fullmatch(x)) or any(_same_root(_root(x), v) for v in verbs)

    predicate_at = next((i for i, w in enumerate(words) if i > 0 and any(_same_root(_root(w), v) for v in verbs)), len(words))
    out = []
    for i, w in enumerate(words):
        if w not in _C12_MODIFIERS:
            continue
        in_subject = 0 < i < predicate_at
        k = i + 1
        while k < len(words) and (words[k] in _C12_MODIFIERS or words[k] in _cm.ITEM_DROP):
            k += 1
        if k >= len(words) or stop(words[k]):
            continue
        head = words[k]
        for x in words[k + 1:k + 3]:                              # a compound noun: its head is its last word
            if stop(x) or x in _C12_MODIFIERS:
                break
            head = x
        out.append(_C12Modifier(w, _C12_MODIFIERS[w], head, in_subject, frozenset(acronyms.get(w, ()))))
    return out


def _c12_modifier_ok(m: _C12Modifier, context: str) -> Tuple[bool, str]:
    """An asked modifier against one statement's context (its list label, its subject and its clause): verified by the modifier
    attached to an occurrence of the head; else vetoed by another member of its family attached to the head; else (on the
    subject) verified by the modifier, or the acronym of the name it belongs to, stated in the context; else unknown."""
    words = _c12_words(context)
    head_root = _root(m.head)
    attached, clash = False, None
    for k, w in enumerate(words):
        if not _same_root(_root(w), head_root):
            continue
        before = words[max(0, k - 3):k]
        attached = attached or m.word in before
        other = [x for x in before if x in _C12_MODIFIER_FAMILIES[m.family] and x != m.word]
        if other and clash is None:
            clash = other[-1]
    if attached:
        return True, ""
    if clash is not None:
        return False, "the statement states '%s %s', not the asked '%s %s'" % (clash, m.head, m.word, m.head)
    if m.in_subject and (m.word in words or any(a in words for a in m.acronyms)):
        return True, ""
    return False, "the asked '%s %s' is not stated" % (m.word, m.head)


# ---- class C: material conditions of the asked predicate ----------------------------------------------------------------
_C12_COMPLEMENT_VERB = re.compile(
    r"\b(?:determine[sd]?|determining|check(?:s|ed|ing)?|verif(?:y|ies|ied|ying)|decide[sd]?|deciding|assess(?:es|ed|ing)?|see|sees|seeing|"
    r"know|knows|ask(?:s|ed|ing)?|examine[sd]?|examining|evaluate[sd]?|evaluating|review(?:s|ed|ing)?|confirm(?:s|ed|ing)?|"
    r"establish(?:es|ed|ing)?|ascertain(?:s|ed|ing)?|consider(?:s|ed|ing)?|investigate[sd]?|investigating|inquire[sd]?|enquire[sd]?|"
    r"judge[sd]?|judging|whether)\s*$", re.I)


def _c12_asked_condition(r: "_cm.Restriction", claim: "_cm.Claim") -> bool:
    """The restriction is the claim's own asked condition (at least half of an asked condition's content words)."""
    stated = {x for x in (r.roots or _roots(r.text, frozenset())) if x not in _PERSON_FAMILY}
    for c in claim.conditions:
        asked = {x for x in c.roots if x not in _PERSON_FAMILY}
        if asked and stated and 2 * sum(1 for x in asked if any(_same_root(x, y) for y in stated)) >= len(asked):
            return True
    return False


#: D-102: the reader's trigger restrictions introduced by "when" / "whenever" (not "once", "upon", "after"), and the concessive
#: "even" that turns such a clause into no condition at all ("remains their faculty role, even when holding an administrative title").
_C12_WHEN_LEAD = re.compile(r"\s*when(?:ever)?\b", re.I)
_C12_CONCESSIVE_BEFORE = re.compile(r"\beven\s*$", re.I)


def _c12_when_condition(r: "_cm.Restriction") -> bool:
    """D-102: a trigger restriction the claim reader introduced by "when" or "whenever"; with ``_C12_WHEN_CONDITION`` on it is
    a material condition of a generic, deontic or universal claim, exactly as an "if" restriction is."""
    return bool(_C12_WHEN_CONDITION and r.kind is _cm.RestrictionKind.TRIGGER and r.text and _C12_WHEN_LEAD.match(r.text))


def _c12_material_conditions(s: "_cm.PredicateStatement", claim: "_cm.Claim", claim_roots: FrozenSet[str], low_q: str
                             ) -> FrozenSet["_cm.RestrictionKind"]:
    """Class C: the if / unless restrictions of a statement that govern its asked predicate - fronted before its anchor, or
    inside its main predicate span (a restriction of a secondary predication, ", chaired by the dean if a full professor", does
    not); an embedded "to determine if" is no condition; the asked condition, an asked "unless" and the claim's own content are
    exempt. D-102: a "when" / "whenever" restriction is read the same way (same exemptions, same position test), except a
    concessive "even when"; other triggers ("once", "upon", "after") and approval gates stay accepted (D-098)."""
    out = set()
    sentence = s.sentence
    pa = sentence.find(s.predicate) if s.predicate else -1
    main = _c11_main_roots(s) if _C12_OWN_CONTENT else frozenset()
    for r in s.restrictions:
        when = _c12_when_condition(r)
        if r.kind not in (_cm.RestrictionKind.SUFFICIENT, _cm.RestrictionKind.EXCEPTION) and not when:
            continue
        if _c11_own(r.text, r.roots, claim_roots, main) or _c12_asked_condition(r, claim):
            continue
        if r.kind is _cm.RestrictionKind.EXCEPTION and re.search(r"\bunless\b|\bexcept\b", low_q):
            continue
        pos = sentence.find(r.text)
        if pos < 0:
            out.add(r.kind)                                       # not located: fails closed
            continue
        if when and _C12_CONCESSIVE_BEFORE.search(sentence[max(0, pos - 12):pos]):
            continue                                              # D-102: "even when" concedes, it does not condition
        if re.match(r"\s*if\b", r.text, re.I) and _C12_COMPLEMENT_VERB.search(sentence[max(0, pos - 40):pos]):
            continue
        if pos < s.anchor.position or (pa >= 0 and pa <= pos < pa + len(s.predicate)):
            out.add(r.kind)
    return frozenset(out)


# ---- the C12 dimension checks of a specific claim ------------------------------------------------------------------------
_C12_BOUND_ANCHORS = (_cm.AnchorKind.VERB, _cm.AnchorKind.ENTITLEMENT, _cm.AnchorKind.PREDICATE)
#: The negators of a FALLBACK statement's predicate: the C10 fact negator ("not only", "not more than" excluded) and a negative
#: entitlement or prohibition word used predicatively ("are ineligible", "is prohibited"; a heading "Prohibited Nepotism" is none).
_C12_FALLBACK_NEGATOR = re.compile(_FACT_NEGATOR.pattern + r"|\b(?:is|are|was|were|be|been|being|remains?)\s+(?:\w+\s+)?(?:ineligible|disqualified|"
                                   r"disentitled|excluded|prohibited|forbidden|barred)\b", re.I)


def _c12_fallback_negated(s: "_cm.PredicateStatement") -> bool:
    """A FALLBACK statement's main predicate span (its conditions and restrictions blanked) carries a negator."""
    span = s.predicate or s.text
    for r in s.restrictions:
        if r.text and r.text in span:
            span = span.replace(r.text, " ")
    return bool(_C12_FALLBACK_NEGATOR.search(span))


def _c12_dimension_checks(frame: "_cm.ClaimFrame", clause: str, checks: Dict[str, Callable]) -> Dict[str, Callable]:
    """The polarity, value, scope, object and condition readers of a specific claim (D-099), replacing the C11 ones in the joint
    comparison: a FALLBACK statement's polarity against the claim's; every asked number, fact, comparison and time distributive
    on the statement; universals named by a modifier as explicit unlimited universals; the asked object in the statement's own
    clause and its important modifiers; the unconditional check without the copular bypass, for a negative claim too, with the
    material if / unless conditions (D-102: and when / whenever). D-100: for a claim that names no role actor and no object (a copular or passive ask about
    an entity, "Is the housing allowance paid once?") the subject reader also needs the asked entity stated in the statement's
    own clause, its list label or its pronoun antecedent when the sentence has several clauses and names the entity in another
    clause, so a clause about another entity ("... and the furniture allowance is paid once") never supports it; an asked word
    the sentence states nowhere is left to the predicate and value readers. The
    C11 exclusivity dimension stays as it is: its hedge veto also holds when the question asks a hedge (exempting an asked
    hedge would let a hedged statement about another entity support the claim)."""
    C, D = _cm.Compatibility, _cm.DimensionResult
    claim = frame.claim
    text = strip_leading_frame(clause) if frame.frame else clause
    low_q = cue_text(text.lower())
    claim_roots = _roots(text, frozenset())
    affirmative = claim.polarity is _cm.Polarity.AFFIRMATIVE
    entity = _entity_roots(claim)
    hr = hr_fact_detail(text, concepts)
    numbers = frozenset(re.findall(r"\d+(?:\.\d+)?", _hr_norm(claim.text)))
    comparisons = _c12_comparisons(frame.main, frame.anchor_roots) if _C12_COMPARE else []
    times = _c12_time_distributives(frame.main) if _C12_LEDGER else []
    universals = _c12_modifier_universals(frame) if _C12_LEDGER else []
    plain = numbers - frozenset(c.number_text for c in comparisons)
    modifiers = _c12_asked_modifiers(frame) if _C12_MODIFIER else []
    object_roots = _c12_object_roots(claim, frame)
    c11_object, c11_scope, c11_polarity, c11_subject = checks.get("object"), checks.get("scope"), checks.get("polarity"), checks.get("subject")
    actor_bound = bool(claim.actor and claim.subject is not None and claim.actor in _head_roles(claim.subject.text))
    entity_phrase, entity_roots = "", frozenset()                           # D-100: the asked entity of an actorless, objectless claim
    if _C12_ENTITY_SUBJECT and not actor_bound and not object_roots and (claim.subject is None or not claim.subject.head):
        entity_phrase, entity_roots = _c12_asked_entity(frame)

    def polarity(f, s):
        if s.anchor.kind is not _cm.AnchorKind.FALLBACK or not _C12_FALLBACK_POLARITY:
            return c11_polarity(f, s) if c11_polarity else None
        if _c12_fallback_negated(s) is not (claim.polarity is _cm.Polarity.NEGATIVE):
            return D("polarity", C.UNKNOWN, "the statement %s the predicate (no anchor of the asked predicate binds it)"
                     % ("negates" if claim.polarity is _cm.Polarity.AFFIRMATIVE else "does not negate"))
        return D("polarity", C.COMPATIBLE)

    def subject(f, s):
        r = c11_subject(f, s) if c11_subject else None
        if (r is not None and r.compatibility is not C.COMPATIBLE) or len(_c12_clause_spans(s.sentence)) <= 1:
            return r                                                        # a single clause: the unit-level C9 check binds the entity
        have = _roots(" ".join((_c12_label(s), s.subject_text or "", s.text, _c12_pronoun_antecedent(s))), frozenset())
        missing = [x for x in entity_roots if not any(_same_root(x, h) for h in have)]
        if not missing:
            return D("subject", C.COMPATIBLE)
        elsewhere = _roots(s.sentence, frozenset())
        if not any(any(_same_root(x, h) for h in elsewhere) for x in missing):
            return r                                                        # named nowhere in the sentence: the predicate / value readers decide
        return D("subject", C.UNKNOWN, "the statement is about another subject (%s), not the asked '%s'"
                 % ((s.subject_text or s.text).strip()[:50], entity_phrase))

    def value_time(f, s):
        if hr is None and not plain and not comparisons and not times:
            return None
        if not _c11_value_stated(hr, plain, claim, s):                     # the C9 HR fact reader and every plain asked number
            return D("value_time", C.UNKNOWN, "the asked fact or value (%s) is not stated in the statement" % (", ".join(sorted(plain)) or "fact"))
        for c in comparisons:
            if not _c12_comparison_holds(c, s):
                return D("value_time", C.UNKNOWN, "the asked comparison '%s' is not established by the statement" % c.text)
        for w, head in times:
            if not _c12_time_restated(head, s.text):
                return D("value_time", C.UNKNOWN, "the asked '%s %s' is not stated" % (w, head))
        return D("value_time", C.COMPATIBLE)

    def scope(f, s):
        r = c11_scope(f, s) if c11_scope else None
        if r is not None and r.compatibility is not C.COMPATIBLE:
            return r
        for np in universals:
            if _cm.slot_universal_compatibility(np, s.noun_phrases, _same_root) is not C.COMPATIBLE:
                return D("scope", C.UNKNOWN, "'%s' is not stated as an explicit, unlimited universal" % np.text)
        return D("scope", C.COMPATIBLE) if universals else r

    def obj(f, s):
        r = c11_object(f, s) if c11_object else None
        if r is not None and r.compatibility is not C.COMPATIBLE:
            return r
        evaluated = r is not None
        label = _c12_label(s)
        if _C12_OBJECT_BINDING and object_roots and len(_c12_clause_spans(s.sentence)) > 1:
            have = _roots(s.text + " " + label, frozenset())
            missing = [x for x in object_roots if not any(_same_root(x, h) for h in have)]
            if missing:
                antecedent = _roots(_c12_pronoun_antecedent(s), frozenset())
                if antecedent and all(any(_same_root(x, h) for h in antecedent) for x in missing):
                    missing = []
            if missing:
                return D("object", C.UNKNOWN, "the asked object (%s) is not stated in the same clause" % ", ".join(sorted(missing)))
            evaluated = True
        context = " ".join((label, s.subject_text or "", s.text))
        for m in modifiers:
            ok, why = _c12_modifier_ok(m, context)
            if not ok:
                return D("object", C.UNKNOWN, why)
            evaluated = True
        return D("object", C.COMPATIBLE) if evaluated else None

    def condition(f, s):
        results = []
        for c in claim.conditions:
            role, gate = _stated_role(s.sentence, c.roots, entity)
            comp = _cm.condition_role_compatibility(c.role, role, gate)
            if (c.role is _cm.ConditionRole.NECESSARY and comp is not C.COMPATIBLE) or comp is C.UNKNOWN:
                return D("condition", C.UNKNOWN, "the asked %s condition (%s) is not stated in that role" % (c.role.value, c.text.strip()))
            results.append(D("condition", C.COMPATIBLE))
        if _C11_UNCONDITIONAL and (affirmative or _C12_NEGATIVE_CONDITION):
            kinds = _c11_unasked(s, claim_roots, low_q, claim)
            if _c11_copular(claim) and not _C12_COPULAR:
                kinds = kinds & {_cm.RestrictionKind.HEDGE}
            bad = set(kinds & _cm.FORCE_RESTRICTIONS[frame.force])
            if _C12_CONDITION and frame.force in (_cm.Force.GENERIC, _cm.Force.DEONTIC, _cm.Force.UNIVERSAL_QUANTIFIED):
                bad |= _c12_material_conditions(s, claim, claim_roots, low_q)
            if bad:
                return D("condition", C.UNKNOWN, "the %s claim is stated only under a restriction (%s)"
                         % (frame.force.value.replace("_", " "), ", ".join(sorted(k.value for k in bad))))
            results.append(D("condition", C.COMPATIBLE))
        return results[0] if results else None

    out = {"polarity": polarity, "value_time": value_time, "scope": scope, "object": obj, "condition": condition}
    if entity_roots:
        out["subject"] = subject
    return out


# ---- class E: another filler of an exclusive slot ------------------------------------------------------------------------
_C12_FILLER_AFTER = re.compile(r"\b(?:additional|another|further|extra|second|third)\b", re.I)
_C12_FILLER_EITHER = re.compile(r"\b(?:again|renewed|extended|repeated)\b", re.I)
_C12_UNIT_ROOTS = frozenset(_root(w) for w in "day days week weeks month months year years time times hour hours semester semesters".split())


def _c12_alternative_filler(quote: str, frame: "_cm.ClaimFrame") -> str:
    """Class E: the window veto of an exclusive value / object claim by another filler of the SAME slot stated in the returned
    quote ("Do faculty receive only ten days of training leave?" - "... and may request additional training leave"): an
    additional / another / further / extra / second / third followed by every content word of the asked slot, or by the asked
    unit in a sentence that states the slot; again / renewed / extended / repeated on either side. An unrelated "additional"
    never vetoes. The reason, or ""."""
    claim = frame.claim
    if not any(e.focus in (_cm.ExclusivityFocus.VALUE, _cm.ExclusivityFocus.OBJECT) for e in claim.exclusivity):
        return ""
    slot = frozenset(r for r in _c12_object_roots(claim, frame) if r not in _C12_UNIT_ROOTS)
    units = frozenset(r for r in claim.obj if r in _C12_UNIT_ROOTS)
    if not slot:
        return ""
    low_q = claim.text.lower()

    def covers(words: Sequence[str]) -> bool:
        have = _roots(" ".join(words), frozenset())
        return all(any(_same_root(x, h) for h in have) for x in slot)

    for sentence in _unit_sentences(quote):
        states_slot = covers(sentence.split())
        for rx, after_only in ((_C12_FILLER_AFTER, True), (_C12_FILLER_EITHER, False)):
            for m in rx.finditer(sentence):
                if m.group(0).lower() in low_q:
                    continue
                after = sentence[m.end():].split()[:4]
                near = after if after_only else after + sentence[:m.start()].split()[-4:]
                first = _root(after[0].strip(".,;:")) if after else ""
                if covers(near) or (first in units and states_slot):
                    return "another filler of the exclusive slot is stated (%s)" % " ".join(sentence[m.start():].split()[:4])
    return ""


#: A policy-process request: the handbook states stages, not a complete procedure, so at most partial (as Research).
_PROCESS_CUES = re.compile(
    r"\b(?:how\s+(?:do|can|should|would|to)\s+(?:i\s+|we\s+)?(?:apply|submit|obtain|get|request|register|claim|file|resign|renew|extend|collect|cancel|book|reserve)|"
    r"what\s+(?:is|are)\s+the\s+(?:[\w-]+\s+){0,3}?(?:process|procedures?|steps)\b|application\s+process|procedure\s+for|steps?\s+(?:to|for|involved)|how\s+is\s+[\w\s]{0,20}?(?:applied|submitted|obtained|renewed|processed)|"
    r"(?:walk|guide|take|talk)\s+(?:me|us)\s+through|(?:fill(?:ing)?\s+(?:in|out)|complet(?:e|ing))\s+(?:the\s+|a\s+|an\s+|my\s+|this\s+)?[\w\s'-]{0,40}?\bforms?|"
    r"step[- ]by[- ]step|(?:list|what\s+are|give\s+me|tell\s+me)\s+(?:the\s+|all\s+the\s+)?steps|what\s+steps|instructions?\s+(?:for|to|on)|which\s+forms?|what\s+forms?)\b", re.I)
_AFTER_SUBMISSION = re.compile(r"\b(?:what\s+happens\s+after|after\s+(?:i\s+|my\s+)?(?:submi\w*|application|resignation)|review\s+(?:process|chain|stages?)|who\s+reviews|"
                               r"who\s+evaluates|who\s+decides|what\s+are\s+the\s+stages)\b", re.I)


def system_named(text: str) -> Optional[str]:
    named = _SYSTEM_CUES.search(text)
    if not named:
        return None
    system = re.sub(r"[\s-]", "", named.group(0).lower())
    if system in ("fis", "facultyinformationsystem"):
        return "fis"
    if system in ("hrportal", "employeeportal", "selfservice"):
        return "hr portal"
    return "myuos"


def is_procedure_request(clause: str, system_hint: Optional[str]) -> Tuple[bool, Optional[str]]:
    """(asks for a system or click procedure, system named)."""
    system = system_named(clause)
    if system is None and system_hint in SYSTEMS:
        system = system_hint
    procedural = bool(_PROCEDURE_CUES.search(clause)) and system is not None
    return procedural, system


def is_process_request(clause: str) -> bool:
    """A policy-process request ("how do I apply for leave", "how do I resign") that the handbook answers with stages
    only; "what happens after submission" and "who approves" are not (the chain can be stated completely)."""
    return bool(_PROCESS_CUES.search(clause)) and not _AFTER_SUBMISSION.search(clause)


_NAVIGATION_CUES = re.compile(
    r"\b(?:where(?:'s|\s+(?:is|are|was|were))\b(?![^.?]{0,50}\b(?:stated|written|documented|defined|described|listed|specified|set\s+out|published)\b)|"
    r"where\s+(?:can|could|do|does|should|shall|would|will|must|to)\s+(?:i|we|one|faculty|employees?|the|a|an|my|our)\s+"
    r"(?:find|get|go|locate|visit|contact|reach|collect|pick|meet|obtain|see|submit|hand|register|apply|report)\b|"
    r"\blocated\b|\blocations?\b|\bwhereabouts\b|\b(?:what|which)\s+(?:floor|building|room|campus|menu|screen|page)\b|\baddress\s+of\b|"
    r"\b(?:mailing|postal|office|physical|street|e\s?mail|email|web)\s+address\b|"
    r"\bhow\s+(?:do|does|can|could|should|would|to)\s+(?:i\s+|we\s+|one\s+)?(?:contact|reach|call|e\s?mail|get\s+in\s+touch|find|get\s+to|go\s+to|locate|visit)\b|"
    r"\bwho(?:m)?\s+(?:do|should|can|could|must|would)\s+(?:i|we)\s+(?:contact|ask|call|e\s?mail|see|go\s+to|refer\s+to|approach|reach|talk\s+to|speak\s+to)\b|"
    r"\bto\s+whom\s+(?:should|do|can|must|may)\s+(?:i|we)\b|\bcontact\s+(?:person|details|information|numbers?)\b|\b(?:tele)?phone\b|\bfax\b|\bhotline\b|"
    r"\bextension\s+(?:numbers?|no)\b|\b(?:e\s?mail|email)\s+(?:address|id)\b|\bwhat\s+is\s+(?:the|their|its|his|her)\s+(?:e\s?mail|email|phone|telephone|website|url|extension|hotline|number)\b|"
    r"\b(?:its|their|the\s+office's|the\s+department's|hr's)\s+(?:e\s?mail|email|phone|telephone|website|extension|number)\b|"
    r"\b[\w' ]{0,30}?(?:'s|s')\s+(?:e\s?mail|email|phone|telephone|website|extension|fax|number)\b|"
    r"\b(?:which|what)\s+(?:office|unit|department|desk)\s+(?:handles?|deals?|is\s+(?:in\s+charge|responsible)|manages?|should|do|does|can|to\s+contact|takes?\s+care|looks?\s+after)\b|"
    r"\bto\s+which\s+(?:office|department|unit)\b|\bhelp\s*desk\b|\bservice\s+desk\b|\bopening\s+hours\b)", re.I)
_CLICK_CUES = re.compile(r"\b(?:click\w*|press|select|tap|buttons?|screens?|menus?|tabs?|icons?|dropdown|log\s?in|login|upload\w*|links?|fields?)\b", re.I)
_SEGMENT_SPLIT = re.compile(
    r"\s*(?:,\s*(?:and\s+|or\s+)?|;\s*|\s+and\s+(?:also\s+)?|\s+or\s+|\?\s*)"
    r"(?=(?:what|who|whom|whose|which|how|when|where|why|is|are|can|could|does|do|did|may|must|should|will|would)\b)|"
    r"\s*,?\s+and\s+(?=(?:(?:its|their|the|his|her|your|our)\s+)?(?:(?:tele)?phone|e\s?mail|email|fax|contact|location|address|extension|website|hotline|whereabouts|opening\s+hours)\b)|"
    r"^\s*how\s+and\s+(?=where\b)", re.I)
_CONTACT_EVIDENCE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|\+\s?\d[\d\s()-]{6,}\d|\b(?:tel|phone|telephone|fax|ext)\.?\s*:?\s*\+?\d|"
    r"\b(?:room|office|building|floor|block)\s+(?:no\.?\s*)?[A-Z]?\d+\b|\bwww\.|https?://|\blocated\s+(?:in|at|on)\b", re.I)


def clause_segments(clause: str) -> List[str]:
    parts = [p.strip(" ,;?") for p in _SEGMENT_SPLIT.split(cue_text(clause))]
    return [p for p in parts if p] or [clause.strip()]


def navigation_intent(clause: str) -> Tuple[bool, bool]:
    """(asks for a place or a contact channel, also asks something substantive). A pure navigation clause is
    Institutional Navigation's alone and is never answered from HR text."""
    nav = substantive = False
    for segment in clause_segments(clause):
        if _NAVIGATION_CUES.search(segment) and not _CLICK_CUES.search(segment):
            nav = True
        elif len(segment.split()) >= 2:
            substantive = True
    if re.match(r"^\s*how\s+and\s+where\b", cue_text(clause), re.I):
        nav = substantive = True
    return nav, substantive


def contact_evidence(unit: str) -> bool:
    return bool(_CONTACT_EVIDENCE.search(unit))


# ---------------------------------------------------------------------------
# The ask gate (shared D-079 to D-084 machinery with the HR role lexicon)
# ---------------------------------------------------------------------------
_LOOSE_CONCEPTS: FrozenSet[str] = frozenset({"limit", "day", "month", "year", "week", "pay", "deduction", "percent", "extension", "eligibility", "currency"})
#: Role words, in priority order: composite bodies first; the VCAA is its own role, distinct from the Vice Chancellor,
#: and the Chancellor from both; a dean is never a chair; the faculty categories that carry their own policies are
#: their own roles. The person asking is faculty.
GROUNDING_ROLES: Tuple[Tuple[str, str], ...] = (
    (r"\bboard\s+of\s+trustees\b|\bbot\b", "board_of_trustees"),
    (r"\buniversity\s+councils?(?!\w)", "university_council"),
    (r"\bcollege\s+councils?(?!\w)", "college_council"),
    (r"\bdepartment(?:al)?\s+councils?(?!\w)", "department_council"),
    (r"\b(?:academic|deans'?)\s+councils?(?!\w)|\bcouncils?(?!\w)", "council"),
    (r"\b(?:central\s+)?(?:hiring|hiring\s+and\s+evaluation|recruitment|search)\s+committees?(?!\w)", "hiring_committee"),
    (r"\b(?:department|college|university)?\s*(?:academic\s+)?promotions?\s+committees?(?!\w)", "promotion_committee"),
    (r"\bcommittees?(?!\w)", "committee"),
    (r"\b(?:department|departmental|program(?:me)?|college)\s+chair(?:s|persons?|person|man|men)?(?:'s)?(?!\w)|\bchair(?:s|persons?|person|man|men)?(?:'s)?(?!\w)|"
     r"\bhead\s+of\s+(?:the\s+)?department(?!\w)|\bhods?(?!\w)", "chair"),
    (r"\bvice[- ]chancellor\s+for\s+academic\s+affairs(?:'s)?(?!\w)|\bvcaa(?:'s)?\b", "vcaa"),
    (r"\bvice[- ]chancellor\w*|\bvcrgs\b|\bvice[- ]president\w*", "vice_chancellor"),
    (r"\bchancellor(?:'s)?(?!\w)", "chancellor"),
    (r"\bpresident(?:'s)?(?!\w)", "president"),
    (r"\b(?:college\s+)?(?:associate\s+|assistant\s+)?deans?(?:'s|')?(?!\w)|\bdeans?\s+of\s+(?:the\s+)?college(?!\w)", "dean"),
    (r"\bhuman\s+resources?(?:\s+(?:department|office|unit))?(?:'s)?(?!\w)|\bhr(?:'s)?\b|\bpersonnel\s+(?:department|office)(?!\w)", "hr"),
    (r"\bpassport\s+unit(?!\w)", "passport_unit"),
    (r"\b(?:business\s+)?services?\s+department(?!\w)", "services_department"),
    (r"\buniversity\s+administration(?!\w)|\b(?:academic\s+)?administrators?(?!\w)", "administration"),
    (r"\bregistrar(?:'s|')?(?!\w)", "registrar"),
    (r"\bdirectors?(?!\w)", "director"),
    (r"\bcoordinators?(?!\w)", "coordinator"),
    (r"\bclinical\s+(?:faculty(?:\s+members?)?|professors?)(?!\w)", "clinical_faculty"),
    (r"\bemeritus\s+(?:faculty(?:\s+members?)?|professors?)(?!\w)", "emeritus_faculty"),
    (r"\bemirati\s+(?:faculty(?:\s+members?)?|track\s+faculty|talent)(?!\w)", "emirati_faculty"),
    (r"\bvisiting\s+(?:faculty(?:\s+members?)?|professors?|lecturers?)(?!\w)", "visiting_faculty"),
    (r"\bpart[- ]time\s+(?:faculty(?:\s+members?)?|instructors?|lecturers?)(?!\w)|\bpart[- ]timers?(?!\w)|\badjunct\s+faculty(?!\w)", "part_time_faculty"),
    (r"\b(?:senior\s+)?lecturers?(?:'s|')?(?!\w)", "lecturer"),
    (r"\bstudents?(?:'s|')?(?!\w)", "student"),
    (r"\b(?:full[- ]time|regular|new|female|ranked|academic)?\s*(?:faculty(?:\s+members?(?:'s|')?)?|instructors?(?:'s|')?|professors?|employees?|academic\s+staff|"
     r"staff\s+members?|candidates?|applicants?|recruits?)(?!\w)", "faculty"),
    (r"\b(?:the\s+)?university\s+of\s+sharjah\b(?:\s*\(\s*uos\s*\))?|\bthe\s+university\b|\buos\b", "university"),
    (r"\bdepartments?\b", "department"),
    (r"\bcolleges?\b", "college"),
)
_ROLE_CONCEPTS: FrozenSet[str] = frozenset({"dean_chair", "council", "chancellor", "hr", "administrative"})

ASK_GATE = _grounding.AskGate(concepts, concept_residue, generic=GENERIC_TERMS, party=_RESPONSIBLE_PARTY, implies=_ANCHOR_IMPLIES,
                              loose=_LOOSE_CONCEPTS, table_row=is_table_row, roles=GROUNDING_ROLES, self_role="faculty",
                              role_concepts=_ROLE_CONCEPTS, phrases=concept_spans)
ASK_GENERAL, ASK_RECOGNISED, ASK_UNKNOWN = _grounding.GENERAL, _grounding.SPECIFIC_RECOGNISED, _grounding.SPECIFIC_UNKNOWN


_NUMBER_WORD = re.compile(r"\b(" + "|".join(sorted(_grounding._NUMBER_WORDS, key=len, reverse=True)) + r")\b(?![- ]time\b)", re.I)


_CURRENCY_AFTER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(AED|aed|Aed|dirhams?|Dirhams?)\b")
_TERM_CONTRACT = re.compile(r"\b(definite|fixed|limited)[- ]term\b", re.I)


def ask_text(clause: str) -> str:
    """The clause as the gate reads it: number words as digits ("sixty calendar days" -> "60 calendar days", "one-year"
    -> "1-year"), so that a number is never read as the verb of the ask, and a currency written after its amount dropped
    ("43,860 AED" -> "43,860"), so that the shared reader builds no fact with the currency as its object (the HR money
    gate reads the clause itself and keeps the currency); nothing else changes."""
    text = _NUMBER_WORD.sub(lambda m: _grounding._NUMBER_WORDS[m.group(1).lower()], clause)
    text = _CURRENCY_AFTER.sub(lambda m: m.group(1), text)
    return _TERM_CONTRACT.sub(lambda m: m.group(1), text)       # "definite term contract": the term is no period


#: Verbs of HR asks that the shared inventory lacks ("chaired by the department chair", "the College Council shall form"):
#: with them the relation survives and its actor is checked instead of being dropped (D-088).
_HR_VERBS: FrozenSet[str] = frozenset({"chair", "chairs", "chaired", "chairing", "head", "heads", "headed", "heading", "form", "forms", "formed",
                                       "nominate", "nominates", "nominated", "endorse", "endorses", "endorsed", "convene", "convenes", "convened",
                                       "constitute", "constitutes", "constituted", "deduct", "deducts", "deducted"})


def _real_verb(action: Optional[str]) -> bool:
    return action is not None and (action in _HR_VERBS or _grounding._known_verb(action) or action in _grounding._LIGHT_VERBS
                                   or action in _grounding._RELATION_LIGHT)


#: Verbs an HR ask may carry as its action when the shared parser builds no relation for it (D-089): the HR verbs plus
#: the actions of the appointment, promotion and approval chains.
_ASK_VERBS: FrozenSet[str] = _HR_VERBS | frozenset({
    "approve", "approves", "approved", "recommend", "recommends", "recommended", "forward", "forwards", "forwarded", "select", "selects", "selected",
    "refer", "refers", "referred", "appoint", "appoints", "appointed", "renew", "renews", "renewed", "terminate", "terminates", "terminated",
    "review", "reviews", "reviewed", "submit", "submits", "submitted", "present", "presents", "presented", "suggest", "suggests", "suggested",
    "issue", "issues", "issued", "notify", "notifies", "notified", "grant", "grants", "granted", "conduct", "conducts", "conducted",
    "evaluate", "evaluates", "evaluated", "sign", "signs", "signed", "prepare", "prepares", "prepared", "decide", "decides", "decided",
    "determine", "determines", "determined", "supervise", "supervises", "supervised", "assist", "assists", "assisted", "inform", "informs", "informed",
    "provide", "provides", "provided", "arrange", "arranges", "arranged", "lead", "leads", "led", "oversee", "oversees", "monitor", "monitors"})
#: HR actions a role may be asked to perform beyond the approval chain (D-095): with them the relation keeps its verb
#: ("Does the Chancellor extend probation?") instead of dropping it and settling on the role's mere presence.
_ROLE_ACTION_VERBS: FrozenSet[str] = frozenset({
    "extend", "extends", "extended", "confirm", "confirms", "confirmed", "waive", "waives", "waived", "cancel", "cancels", "cancelled", "canceled",
    "reduce", "reduces", "reduced", "increase", "increases", "increased", "revoke", "revokes", "revoked", "suspend", "suspends", "suspended",
    "award", "awards", "awarded", "deduct", "deducts", "deducted", "endorse", "endorses", "endorsed"})
_ASK_VERBS = _ASK_VERBS | _ROLE_ACTION_VERBS
#: Nouns that make a role a label ("the Chancellor policy", "the HR section", "UoS HR policies"): a label names no actor.
_LABEL_NOUNS: FrozenSet[str] = frozenset("policy policies section sections chapter chapters procedure procedures guideline guidelines rule rules "
                                         "regulation regulations handbook manual document documents page pages form forms portal".split())
_ASK_START = re.compile(r"^(?:does|do|did|can|may|must|should|shall|will|would)\s+", re.I)
#: Container roles that identify no actor on their own: never demanded as named roles.
_STRUCTURAL_ROLES: FrozenSet[str] = frozenset({"faculty", "university", "department", "college", "student"})


def _named_roles(text: str) -> FrozenSet[str]:
    """The roles and bodies the policy segments of an ask name, the structural ones aside (D-089): they stay demanded in
    the evidence unit whatever the shared parser made of the ask. A navigation segment ("where is the HR office") names
    a place, not an actor, and is Institutional Navigation's."""
    segments = [seg for seg in clause_segments(text) if not (_NAVIGATION_CUES.search(seg) and not _CLICK_CUES.search(seg))] or [text]
    return frozenset(r for seg in segments for r in ASK_GATE.lexicon.roles_in(seg) if r not in _STRUCTURAL_ROLES)


def _degenerate_relation(rel: Optional[_grounding.Relation]) -> bool:
    """A relation whose "action" is no verb at all ("department chair chair the committee" read as chair -> the)."""
    return rel is not None and rel.action is not None and not _real_verb(rel.action)


#: Reporting verbs and complementisers: "Does the Faculty Handbook say HR provides ..." reports the clause after the verb.
_PHRASE_BREAK: FrozenSet[str] = frozenset("say says said state states stated mention mentions indicate indicates that whether if when where "
                                          "does do did can may must should shall will would".split())


def _hr_relation(text: str) -> Optional[_grounding.Relation]:
    """The actor / action / object an HR ask states when the shared parser builds no relation for its verb (D-089): the
    explicit verb after a role subject, whether or not the HR inventory knows it ("Does HR draft the Security Clearance?",
    "Does the Chancellor exercise authority ...?"), the rest split as the shared parser splits an object and a target. A
    label subject ("the Chancellor policy", "the HR rules and regulations") names no actor (D-095, D-096)."""
    st = ask_structure(text)
    if st is None or st.label is not None or st.actor is None or st.actor in _STRUCTURAL_ROLES:
        return None
    obj, target, prep = _grounding._split_rest(st.rest)
    target_role = ASK_GATE.lexicon.head_role(" ".join(target)) if target else None
    if target and (target_role is None or target_role in _STRUCTURAL_ROLES) and st.verb not in _ASK_VERBS:
        target, target_role, prep = (), None, None                # a generic verb binds actor, action and object ("for the Employee File" is no direction)
    if not obj and not target:
        return None
    return _grounding.Relation(actor=st.actor, party=False, recipient=False, action=_grounding._verb_base(st.verb), modifiers=(), obj=obj,
                               target=target, target_role=target_role, passive=False, actor_words=(), target_prep=prep, source=(), source_role=None)


def _hr_relation_stated(rel: _grounding.Relation, unit_text: str) -> bool:
    """The HR-built relation is stated by an occurrence of its action whose stated subject (active) or agent (passive)
    IS the asked actor, with the asked object and target: a role merely present elsewhere in the unit settles nothing,
    and an occurrence that names no subject never counts (D-089)."""
    text = _PARENTHETICAL.sub(" ", _grounding.strip_exclusions(unit_text))    # ", in consultation with the Chancellor," is no subject (D-096)
    unit = _grounding._Unit(text, ASK_GATE.lexicon)
    for occ in ASK_GATE.relations.occurrences(rel, unit):
        if occ["voice"] in ("active", "passive") and occ["subject"] == rel.actor and occ["object_ok"] and occ["target_ok"] and occ["modifiers_ok"]:
            return True
    return False


#: The three promotion committees are distinct bodies (D-089): a committee an ask names is never settled by the text of
#: another one ("the Department Academic Promotions Committee" is not "the College Academic Promotions Committee").
_PROMOTION_BODIES: Dict[str, "re.Pattern[str]"] = {
    "department": re.compile(r"\bdepartment(?:al)?\s+(?:academic\s+)?promotions?\s+committees?\b", re.I),
    "college": re.compile(r"\bcollege\s+(?:academic\s+)?promotions?\s+committees?\b", re.I),
    "university": re.compile(r"\b(?:university\s+)?central\s+(?:academic\s+)?(?:promotions?\s+)?committees?\b|"
                             r"\buniversity\s+(?:academic\s+)?promotions?\s+committees?\b", re.I),
}


def _bodies_of(text: str) -> FrozenSet[str]:
    """The promotion committees ``text`` names."""
    return frozenset(k for k, p in _PROMOTION_BODIES.items() if p.search(text or ""))


def _body_relation_stated(rel: _grounding.Relation, bodies: FrozenSet[str], item: str) -> bool:
    """The relation is stated in ``item`` with each asked committee in the place the ask gives it: the object (the
    patient of a passive occurrence, the words after an active one), the target or source phrase, or the actor (the
    agent after "by", the subject before an active verb); the actor, when asked, is the occurrence's subject."""
    unit = _grounding._Unit(_grounding.strip_exclusions(item), ASK_GATE.lexicon)
    obj_bodies = _bodies_of(" ".join(rel.obj)) & bodies
    tgt_bodies = _bodies_of(" ".join(rel.target)) & bodies
    src_bodies = _bodies_of(" ".join(rel.source)) & bodies
    act_bodies = bodies - obj_bodies - tgt_bodies - src_bodies
    to_class = _grounding._PREP_CLASSES.get(rel.target_prep or "to", _grounding._PREP_CLASSES["to"])
    for occ in ASK_GATE.relations.occurrences(rel, unit):
        if not (occ["object_ok"] and occ["target_ok"] and occ["modifiers_ok"]):
            continue
        if rel.actor is not None and occ["subject"] != rel.actor:
            continue
        i = occ["index"]
        if obj_bodies:
            text = occ.get("patient", "") if occ["voice"] == "passive" else unit.window(i, 12, stop_at_preps=False)[0]
            if not obj_bodies <= _bodies_of(text):
                continue
        if tgt_bodies and not tgt_bodies <= _bodies_of(unit.phrase_after_prep(i, to_class) or ""):
            continue
        if src_bodies and not src_bodies <= _bodies_of(unit.phrase_after_prep(i, frozenset({"from"})) or ""):
            continue
        if act_bodies:
            agent = unit.phrase_after_prep(i, frozenset({"by"})) if occ["voice"] == "passive" else unit.subject_text_before(i)
            if not act_bodies <= _bodies_of(agent or ""):
                continue
        return True
    return False


def _bodies_bound(rel: Optional[_grounding.Relation], bodies: FrozenSet[str], unit_text: str, base: Callable[[str], bool]) -> bool:
    """Some sentence of the unit names every asked committee and states the ask there (D-089)."""
    for unit in _evidence._local_units(unit_text):
        item = _evidence._LIST_MARKER.sub("", unit)
        if not bodies <= _bodies_of(item):
            continue
        if (rel is None and base(item)) or (rel is not None and _body_relation_stated(rel, bodies, item)):
            return True
    return False


def _with_bodies(a: _grounding.AskAssessment, text: str) -> _grounding.AskAssessment:
    """The assessment with the asked promotion committees bound into its grounding detail."""
    bodies = _bodies_of(text)
    if not bodies:
        return a
    old = next((d for d in a.details if d.family == "grounding"), None)
    base = old.check if old is not None else (lambda unit: True)
    label = (old.label + "; " if old is not None else "") + "the %s promotion committee" % " / ".join(sorted(bodies))

    def check(unit: str, _base=base, _rel=a.relation, _bodies=bodies) -> bool:
        return _base(unit) and _bodies_bound(_rel, _bodies, unit, _base)

    details = tuple(d for d in a.details if d.family != "grounding") + (Detail("grounding", label, check),)
    return replace(a, details=details)


#: Words a salary-value ask may carry beside the rank and the amounts: a data row states them by being a row (D-090).
_SALARY_VALUE_WORDS: FrozenSet[str] = frozenset({
    "basic", "monthly", "month", "salary", "salaries", "scale", "range", "ranges", "pay", "aed", "dirham", "dirhams", "rank", "ranks", "academic",
    "minimum", "maximum", "lowest", "highest", "starting", "start", "top", "amount", "amounts", "between", "to", "from", "band", "grade"})


def _table_lines(unit: str) -> List[str]:
    return [line.strip() for line in re.split(r"\s*\|\s*|\n", unit or "") if line.strip()]


def _salary_header_unit(unit: str) -> bool:
    """The unit is the salary header alone (or headers only): it states no proposition (D-090)."""
    lines = _table_lines(unit)
    return bool(lines) and all(is_salary_header(_evidence._LIST_MARKER.sub("", line)) for line in lines)


def _salary_row_terms_ok(a: _grounding.AskAssessment, unit: str) -> bool:
    """A unit made of salary data rows (and headers) settles only a rank's amounts and currency: every term of the ask
    beyond the salary-value words and numbers is stated in a row ("Is a professor's salary based on years of service?"
    is never settled by "Professor 30,576 - 43,860") (D-090)."""
    lines = _table_lines(unit)
    rows = [line for line in lines if is_salary_data_row(line)]
    if not rows or len(rows) + sum(1 for line in lines if is_salary_header(line)) != len(lines):
        return True                                               # not a salary-table unit: the other gates decide
    text = _grounding.normalise(" ".join(rows))
    words = _grounding._WORD.findall(text)
    alias_words = {w for alias in _RANK_ALIASES for w in alias.split()}
    for term in a.terms:
        t = str(term).lower()
        if t in _SALARY_VALUE_WORDS or t in alias_words or re.fullmatch(r"[\d,.]+", t):
            continue
        if not _grounding.term_stated(term, text, words):
            return False
    return not a.deontic and not a.party


def _with_salary_table(a: _grounding.AskAssessment) -> _grounding.AskAssessment:
    """The assessment with the salary-table contract bound into its grounding detail (D-090)."""
    old = next((d for d in a.details if d.family == "grounding"), None)
    if old is None:
        return a                                                  # no grounding detail to bind (an unknown or general ask)
    base = old.check

    def check(unit: str, _base=base, _a=a) -> bool:
        if _salary_header_unit(unit):
            return False                                          # the header is schema, never evidence
        return _salary_row_terms_ok(_a, unit) and _base(unit)

    details = tuple(d for d in a.details if d.family != "grounding") + (Detail("grounding", old.label, check),)
    return replace(a, details=details)


def _dropped_relation_roles(assessment: _grounding.AskAssessment, rel: Optional[_grounding.Relation],
                            named: FrozenSet[str] = frozenset()) -> FrozenSet[str]:
    """The roles an ask names stay demanded when its relation is dropped as bogus (D-088): "Does the Chancellor approve
    additional maternity leave without pay of up to thirty days" still needs the Chancellor in the unit."""
    roles = set(assessment.roles) | set(named)
    if rel is not None:
        roles.update(r for r in (rel.actor, rel.target_role, rel.source_role) if r)
    roles.discard("faculty")
    return frozenset(roles)


_RANGE_TO = re.compile(r"\b(?:up|of up)\s+to\s+(?:a\s+(?:maximum|minimum)\s+of\s+)?\d|\b\d[\d,.]*\s*(?:%|percent|aed|dirhams?)?\s+to\s+(?:aed\s*)?\d", re.I)
#: A nominalised HR concept read as the verb of the ask ("the first appointment ... for three years"): the subject, no action.
_NOMINAL_ACTION = re.compile(r"(?:ment|tion|sion|ance|ence|ship)s?$")
#: "pay" as a noun in a pay-status phrase ("maternity leave without pay"): no action either.
_PAY_NOUN = re.compile(r"\b(?:without|with|full|half|no)\s+pay\b", re.I)


def _bogus_relation(rel: Optional[_grounding.Relation], text: str) -> bool:
    """A relation the shared parser built from a number phrase: its "action" is no verb, or its target is the number
    after "up to" / "N to M" ("Can additional maternity leave without pay of up to 30 days be approved" read as pay -> to
    30 days)."""
    if rel is None or rel.action is None or rel.party:
        return False
    if not _real_verb(rel.action):
        return True
    if _NOMINAL_ACTION.search(rel.action) and concepts(rel.action).anchors:
        return True                                               # "appointment", "termination": the ask's subject, read as a verb
    if rel.action == "pay" and _PAY_NOUN.search(text):
        return True                                               # "without pay", "full pay": a pay status, not an action
    if rel.target and rel.target_prep == "to" and _RANGE_TO.search(text):
        if rel.target[0].replace(",", "").replace(".", "").isdigit():
            return True
        low = cue_text(text.lower())
        if re.search(r"\b(?:up\s+to|to)\s+(?:a\s+(?:maximum|minimum)\s+of\s+)?\d[\d,.]*\s+(?:\w+\s+){0,2}?" + re.escape(rel.target[0].lower()) + r"\b", low):
            return True                                           # "offered for faculty ... up to 3 dependent children": no target
    return False


#: A passive ask without an actor ("Can the probation period be extended?", "May unpaid special leave be granted?"): the
#: shared parser builds no relation for it, or one from a noun ("employment confirmation includes the probation period"
#: read as confirm -> probation period), so any sentence naming the subject could settle it. The verb is bound to the
#: asked subject in some sentence (D-092, D-093): a verbal form with the subject as the nearest HR anchor before it ("the
#: visiting contract may be extended") or as its object right after it ("UoS may terminate faculty contracts"); a noun
#: form only through its "of" phrase ("termination of faculty contracts"); a relational verb ("includes", "involves")
#: states no action on the subject.
_PASSIVE_ASK = re.compile(r"^\s*(?:can|could|may|might|must|should|shall|will|would)\s+(?P<subject>(?:[\w'-]+\s+){1,6}?)be\s+"
                          r"(?P<verb>[a-z]+(?:ed|en))\b(?P<rest>.*)$", re.I)
#: The HR actions an "is X <verb>ed (by Y)" ask or an active ask with a non-role subject states (D-094); relational verbs
#: ("include", "involve", "cover") are no action and are never in this list.
_ACTION_STEMS = r"(?:extend|renew|terminat|confirm|approv|grant|reduc|increas|deduct|review|forward|endors|cancel|revok|replac|adjust|waiv|suspend|appoint|promot|evaluat)"
_COPULAR_PASSIVE_ASK = re.compile(r"^\s*(?:is|are|was|were)\s+(?P<subject>(?:[\w'-]+\s+){1,6}?)(?P<verb>" + _ACTION_STEMS + r"(?:ed|led|d))\b(?P<rest>.*)$", re.I)
def _bindable_subject(subject: str) -> bool:
    """An HR concept that is no role ("the probation period", "employment confirmation"): its action is bound here; a role
    ("a dean", "the Chancellor") binds through the shared or the HR-built relation (D-089, D-094)."""
    role = ASK_GATE.lexicon.head_role(subject)
    return (role is None or role in _STRUCTURAL_ROLES) and bool(concepts(subject).anchors)


#: The participle of an "is X <verb>ed" ask is its main predicate: its subject holds no relative clause ("who are
#: terminated during probation") and it ends the question or opens a prepositional phrase ("terminated by the Dean"), never a
#: noun phrase ("a terminated faculty member") (D-094).
_CLAUSE_WORDS = re.compile(r"\b(?:who|whom|which|that|whose|is|are|was|were|be|been)\b", re.I)
_COPULAR_REST = re.compile(r"^\s*(?:[?.!]?\s*$|(?:by|with|for|upon|after|before|during|in|at|on|under|without|within|every|each|annually|"
                           r"automatically|once|twice|when|if|until|from|to|beyond|again|only|yearly)\b)", re.I)
#: A predicate after the participle makes it a reduced relative clause ("Are new faculty terminated during probation
#: entitled to return tickets?"), not the main predicate.
_LATER_PREDICATE = re.compile(r"\b(?:entitled|eligible|allowed|permitted|required|based|included|paid|subject|responsible|expected|considered|"
                              r"exempt|liable|obliged)\b", re.I)
_ACTIVE_ASK = re.compile(r"^\s*(?:does|do|did|can|could|will|would|may|might|must|should|shall)\s+(?P<subject>(?:[\w'-]+\s+){1,5}?)"
                         r"(?P<verb>" + _ACTION_STEMS + r"(?:e|es|s)?|affects?|impacts?)\s+(?P<rest>.+)$", re.I)


@dataclass(frozen=True)
class PassiveAsk:
    """The subject words, the subject's HR anchors and the verb (any form of its stem) of a passive ask (D-092), with the
    agent an "is X <verb>ed by Y" ask names (D-094)."""
    subject: FrozenSet[str]
    anchors: FrozenSet[str]
    verb: "re.Pattern[str]"
    agent: FrozenSet[str] = frozenset()


def _verb_forms(stem: str) -> "re.Pattern[str]":
    """Every form of a verb: the forms that start with its stem; no other verb stands for it (D-094)."""
    return re.compile(r"\b" + re.escape(stem) + r"\w*", re.I)


def _verb_stem(participle: str) -> str:
    """"extended" -> "extend", "approved" -> "approv", "given" -> "give": every form of the verb starts with it."""
    p = participle.lower()
    if p.endswith("ied"):
        return p[:-3] + "i"
    if p.endswith(("ed", "en")):
        return p[:-2] if len(p) - 2 >= 4 else p[:-1]
    return p


def passive_ask(clause: str) -> Optional[PassiveAsk]:
    """The passive ask without an agent ("Can the probation period be extended (with Chancellor approval)?"), else None;
    an ask naming its agent ("be approved by the Vice Chancellor") binds the agent through the shared relation."""
    text = cue_text(ask_text(clause)).strip()
    m = _PASSIVE_ASK.match(text)
    agent: FrozenSet[str] = frozenset()
    if m is not None and re.search(r"\bby\b", m.group("rest"), re.I):
        return None
    if m is None:
        m = _COPULAR_PASSIVE_ASK.match(text)                      # "Is the probation period terminated by the Dean?" (D-094)
        if (m is None or _CLAUSE_WORDS.search(m.group("subject")) or not _COPULAR_REST.match(m.group("rest"))
                or _LATER_PREDICATE.search(m.group("rest"))):
            return None                                           # "Is a terminated faculty member entitled ...": no passive
        if not _bindable_subject(m.group("subject")):
            return None                                           # "Is a dean appointed ...": a role binds through its relation
        by = re.search(r"\bby\s+(?P<agent>[^?.,;!]+)", m.group("rest"), re.I)
        if by:
            agent = frozenset(_stem(t) for t in _raw_tokens(_CONDITION_END.sub("", by.group("agent"))) if t not in _CONDITION_STOP)
    subject = m.group("subject")
    words = frozenset(_stem(t) for t in _raw_tokens(subject) if t not in _UNIVERSAL_STOP and t not in ("my", "our", "their", "your"))
    return PassiveAsk(words, concepts(subject).anchors, _verb_forms(_verb_stem(m.group("verb"))), agent)


#: A noun form of the verb ("termination", "confirmation", "extension", "renewal", "approval").
_NOMINAL_FORM = re.compile(r"(?:tion|sion|ment|al|ance|ence|ure)s?$", re.I)
#: Relational verbs: "the contract includes a probation year" states no action on the probation year (D-093).
_RELATIONAL_STEMS: FrozenSet[str] = frozenset({"includ", "include", "involv", "involve", "contain", "cover", "compris", "comprise", "consist"})


def _subject_phrase_bound(phrase: str, ask: PassiveAsk) -> bool:
    """The phrase names the asked subject: its first HR anchor is the subject's and every subject word is in it."""
    low = cue_text(phrase.lower())
    first: FrozenSet[str] = frozenset()
    for a, b in concept_spans(low):
        found = concepts(low[a:b]).anchors
        if found:
            first = found
            break
    return bool(first & ask.anchors) and ask.subject <= set(_tokens(low))


def _nominal_bound(after: str, ask: PassiveAsk) -> bool:
    """"termination of faculty contracts": the noun form's "of" phrase names the asked subject (D-093)."""
    words = after.split()
    return bool(words) and words[0].lower() == "of" and _subject_phrase_bound(" ".join(words[1:6]), ask)


#: Words that make a verbal form passive ("who are terminated") or open a phrase that is no object ("during probation").
_PASSIVE_AUXILIARY = re.compile(r"\b(?:is|are|was|were|be|been|being|get|gets|got)\s*$", re.I)
_NOT_AN_OBJECT: FrozenSet[str] = frozenset("during for in at on by with from to after before upon under within of into through until without "
                                           "as than while when if".split())


def _object_bound(before: str, after: str, ask: PassiveAsk) -> bool:
    """"UoS may terminate faculty contracts": the asked subject is the object right after an ACTIVE verbal form (D-093);
    "who are terminated during probation" is passive and its following phrase is no object."""
    words = after.split()
    if not words or _PASSIVE_AUXILIARY.search(before) or words[0].lower().strip(",") in _NOT_AN_OBJECT:
        return False
    return _subject_phrase_bound(" ".join(words[:4]), ask)


def _nearest_anchor(prefix: str) -> FrozenSet[str]:
    low = cue_text(prefix.lower())
    nearest: FrozenSet[str] = frozenset()
    for a, b in concept_spans(low):
        found = concepts(low[a:b]).anchors
        if found:
            nearest = found
    return nearest


def _passive_bound(unit: str, ask: PassiveAsk) -> bool:
    """Some sentence states a form of the asked verb of the asked subject (D-092, D-093), with the asked agent (D-094)."""
    if ask.verb.pattern.replace("\\b", "").replace("\\w*", "") in _RELATIONAL_STEMS:
        return False
    for sentence in _SENTENCE_SPLIT.split(cue_text(unit).replace("|", " ")):
        for m in ask.verb.finditer(sentence):
            before, after = sentence[:m.start()], sentence[m.end():]
            if _NOMINAL_FORM.search(m.group(0)):
                if _nominal_bound(after, ask) and _agent_bound(before, after, ask, active=False):
                    return True
                continue
            if ask.subject <= set(_tokens(before)) and _nearest_anchor(before) & ask.anchors and _agent_bound(before, after, ask, active=False):
                return True
            if _object_bound(before, after, ask) and _agent_bound(before, after, ask, active=True):
                return True
    return False


def _agent_bound(before: str, after: str, ask: PassiveAsk, active: bool) -> bool:
    """The asked agent ("terminated by the Dean") is the actor of the occurrence (D-094): the "by" phrase after a passive or
    noun form, or the subject right before an active form; a role named elsewhere in the sentence ("UoS HR policies") is
    no agent."""
    if not ask.agent:
        return True
    by = re.match(r"\s+(?:[\w'/-]+\s+){0,6}?by\s+((?:[\w'/-]+\s*){1,6})", after, re.I)
    if by and ask.agent <= set(_tokens(by.group(1))):
        return True
    if active:
        return ask.agent <= set(_tokens(re.split(r"[,;:]", before)[-1])[-6:])
    return False


@dataclass(frozen=True)
class ActiveAsk:
    """An active ask whose subject is an HR concept, not a role ("Does employment confirmation extend the probation
    period?"): subject words and anchors, the verb and the object's words and anchors (D-094)."""
    subject: FrozenSet[str]
    anchors: FrozenSet[str]
    verb: "re.Pattern[str]"
    obj: FrozenSet[str]
    obj_anchors: FrozenSet[str]


def active_ask(clause: str) -> Optional[ActiveAsk]:
    """The active ask with a non-role HR subject and an HR action verb, else None (a role subject binds through the
    HR-built relation, D-089)."""
    m = _ACTIVE_ASK.match(cue_text(ask_text(clause)).strip())
    if m is None:
        return None
    subject = m.group("subject")
    if not _bindable_subject(subject):
        return None
    anchors = concepts(subject).anchors
    verb = m.group("verb").lower()
    stem = verb[:-2] if verb.endswith("es") and len(verb) > 5 else verb[:-1] if verb.endswith(("s", "e")) and len(verb) > 5 else verb
    rest = _CONDITION_END.sub("", m.group("rest").rstrip("?"))
    words = frozenset(_stem(t) for t in _raw_tokens(subject) if t not in _UNIVERSAL_STOP)
    obj = frozenset(_stem(t) for t in _raw_tokens(rest) if t not in _UNIVERSAL_STOP)
    return ActiveAsk(words, anchors, _verb_forms(stem), obj, concepts(rest).anchors)


def _active_bound(unit: str, ask: ActiveAsk) -> bool:
    """Some sentence states a verbal form of the asked verb with the asked subject as the nearest HR anchor before it and
    the asked object after it (D-094); a noun form or another verb ("includes") never stands for it."""
    for sentence in _SENTENCE_SPLIT.split(cue_text(unit).replace("|", " ")):
        for m in ask.verb.finditer(sentence):
            if _NOMINAL_FORM.search(m.group(0)):
                continue
            prefix = sentence[:m.start()]
            if not (ask.subject <= set(_tokens(prefix)) and _nearest_anchor(prefix) & ask.anchors):
                continue
            after = " ".join(sentence[m.end():].split()[:8])
            if (ask.obj_anchors and concepts(after).anchors & ask.obj_anchors) or (not ask.obj_anchors and ask.obj <= set(_tokens(after))):
                return True
    return False


def _with_active_action(a: _grounding.AskAssessment, text: str) -> _grounding.AskAssessment:
    """The assessment with the action of an active non-role ask bound to its subject and object when the parser built no
    relation (D-094)."""
    if a.relation is not None:
        return a
    ask = active_ask(text)
    old = next((d for d in a.details if d.family == "grounding"), None)
    if ask is None or old is None:
        return a

    def check(unit: str, _base=old.check, _ask=ask) -> bool:
        return _base(unit) and _active_bound(unit, _ask)

    details = tuple(d for d in a.details if d.family != "grounding") + (Detail("grounding", old.label + "; the asked action stated of the asked subject and object", check),)
    return replace(a, details=details)


def _with_passive_action(a: _grounding.AskAssessment, text: str) -> _grounding.AskAssessment:
    """The assessment with the passive verb bound to the asked subject, whether or not the parser built a relation: a
    relation read from a noun ("confirmation" -> probation period) binds nothing on its own (D-092, D-093)."""
    ask = passive_ask(text)
    old = next((d for d in a.details if d.family == "grounding"), None)
    if ask is None or old is None:
        return a

    def check(unit: str, _base=old.check, _ask=ask) -> bool:
        return _base(unit) and _passive_bound(unit, _ask)

    details = tuple(d for d in a.details if d.family != "grounding") + (Detail("grounding", old.label + "; the asked action stated of the asked subject", check),)
    return replace(a, details=details)


@dataclass(frozen=True)
class LabelAsk:
    """An ask whose subject is a label ("Does the Chancellor policy extend probation?"): the label words and the verb (D-095)."""
    subject: FrozenSet[str]
    label: str
    verb: "re.Pattern[str]"


#: Verbs of content: a document is their natural subject ("Does the Employee Handbook cover HR policies?"); the named-source
#: contract (D-090) binds those asks, not the label-as-actor contract.
_CONTENT_VERBS: FrozenSet[str] = frozenset("cover covers include includes contain contains list lists describe describes outline outlines specify "
                                           "specifies address addresses explain explains mention mentions say says state states detail details "
                                           "define defines set sets summarise summarises summarize summarizes".split())


def label_ask(clause: str) -> Optional[LabelAsk]:
    """The label subject of an action ask, else None (D-095): a subject phrase holding a label noun ("the Chancellor policy",
    "the HR rules and regulations", "HR guidelines", "the HR policy manual"), whatever the action verb after it (D-096); a
    content verb ("cover", "state") makes no label ask."""
    st = ask_structure(clause)
    if st is None or st.label is None or st.verb in _CONTENT_VERBS:
        return None
    stem = re.sub(r"(?:ed|es|s|e)$", "", st.verb) if len(st.verb) > 5 else st.verb
    subject = frozenset(t for t in _raw_tokens(" ".join(st.subject)) if t not in _UNIVERSAL_STOP | _UNIVERSAL_PREPS)
    return LabelAsk(subject, st.label, _verb_forms(stem))


def _label_bound(unit: str, ask: LabelAsk) -> bool:
    """The label itself is the subject of the verb: in some clause the label words stand right before a form of the verb."""
    for sentence in _SENTENCE_SPLIT.split(cue_text(unit).replace("|", " ")):
        for m in ask.verb.finditer(sentence):
            if _NOMINAL_FORM.search(m.group(0)):
                continue
            before = re.split(r"[,;:]", sentence[:m.start()])[-1]
            tokens = _raw_tokens(before)
            if ask.label in tokens[-4:] and ask.subject <= set(tokens):
                return True
    return False


def _with_label_subject(a: _grounding.AskAssessment, text: str) -> _grounding.AskAssessment:
    """The assessment with a label subject bound as the subject of its verb (D-095): "UoS HR policies" or a sentence that
    only mentions the Chancellor never settles what "the Chancellor policy" does."""
    ask = label_ask(text)
    old = next((d for d in a.details if d.family == "grounding"), None)
    if ask is None or old is None:
        return a

    def check(unit: str, _base=old.check, _ask=ask) -> bool:
        return _base(unit) and _label_bound(unit, _ask)

    details = tuple(d for d in a.details if d.family != "grounding") + (Detail("grounding", old.label + "; the asked label as the subject of the asked action", check),)
    return replace(a, details=details)


def assess_ask(clause: str) -> _grounding.AskAssessment:
    """The shape of a clause and the grounding detail it adds (shared rule with the HR vocabulary). A relation whose
    "action" is no verb ("Is the allowance up to 25,000 Dirhams per child" read as children -> to 25,000) is a copular
    ask: the relation is dropped and the grounding detail rebuilt without it; concepts, terms, the roles the ask names
    (D-088) and the numeric facts still bind. When the parser builds no relation or a degenerate one for an ask that
    names a role, the role stays demanded and an HR-built relation binds the actor to the action (D-089)."""
    text = ask_text(clause).strip()
    a = ASK_GATE.assess(text)
    rel = a.relation
    named = _named_roles(text)
    if rel is None and not a.party and named:
        # the parser built no relation ("form" is no verb to it): the named roles stay demanded, and the HR-built relation
        # binds the actor to the action when the ask states one (D-089)
        a = _rebuilt(a, text, named, _hr_relation(text))
    elif _bogus_relation(rel, text):
        a = _rebuilt(a, text, _dropped_relation_roles(a, rel, named), _hr_relation(text) if _degenerate_relation(rel) else None)
    return _with_salary_table(_with_bodies(_with_label_subject(_with_active_action(_with_passive_action(a, text), text), text), text))


def _rebuilt(a: _grounding.AskAssessment, text: str, roles: FrozenSet[str], hr_rel: Optional[_grounding.Relation]) -> _grounding.AskAssessment:
    """The assessment with its grounding detail rebuilt: concepts, terms and the given roles through the shared check,
    plus the HR-built relation when there is one."""
    contents = bool(_grounding._CONTENTS_ASK.search(text))
    base = ASK_GATE._grounding_check(a.concepts, a.terms, a.deontic, a.party, a.action, a.value, contents, roles=roles, relation=None, clause_text=text)
    if hr_rel is not None:
        def check(unit: str, _base=base, _rel=hr_rel) -> bool:
            return _base(unit) and _hr_relation_stated(_rel, unit)
    else:
        check = base
    what = ", ".join(list(a.terms) + sorted(a.concepts) + sorted(roles)) or "the topic"
    if hr_rel is not None:
        what += "; %s %s %s" % (hr_rel.actor, hr_rel.action, " ".join(hr_rel.obj + hr_rel.target))
    details = tuple(d for d in a.details if d.family != "grounding") + (Detail("grounding", "the asked specifics stated with the answer (%s)" % what, check),)
    return replace(a, details=details, roles=roles, relation=hr_rel)


# ---------------------------------------------------------------------------
# Scope, quote selection, resources
# ---------------------------------------------------------------------------
def compile_scope(section_map: SectionMap, source_id: str = APPROVED_SOURCE_ID,
                  owned: Sequence[Tuple[int, str, str]] = SCOPE_SECTIONS,
                  extra_pages: Sequence[Tuple[int, int]] = EXTRA_PAGES) -> RetrievalScope:
    """The retrieval scope for the owned and shared sections (shared rule; never silently smaller or wider)."""
    return _scoping.compile_scope(section_map, source_id, owned, extra_pages)


def select_quote(text: str, chunk_type: Optional[str], clause: Concepts, allowed: Optional[Sequence[Tuple[int, int]]] = None,
                 details: Sequence["Detail"] = ()) -> Optional[Tuple[str, int, int]]:
    return _evidence.select_quote(text, chunk_type, lambda t: relevance(clause, t), allowed, details, max_words=MAX_QUOTE_WORDS,
                                  table_row=is_table_row)


@dataclass(frozen=True)
class FacultyServicesResources:
    """What the specialist needs to retrieve evidence. Built from a ``KnowledgeBase`` in production and from fakes in tests."""

    embed: Callable[[str], np.ndarray]
    rerank: Callable[[Sequence[Sequence[str]]], Sequence[float]]
    index: Any
    chunks: List[str]
    metadata: List[Dict[str, Any]]
    source: SourceRecord
    section_map: SectionMap

    @classmethod
    def from_knowledge_base(cls, kb) -> "FacultyServicesResources":
        source = getattr(kb, "source", None)
        if source is None or not source.registered or source.source_id != APPROVED_SOURCE_ID:
            raise ValueError("the Faculty Services & HR specialist requires the registered handbook %r" % APPROVED_SOURCE_ID)
        applied = ((getattr(kb, "stats", None) or {}).get("source") or {}).get("section_records", 0)
        if not applied:
            raise ValueError("the section map was not applied to the loaded document; HR scope cannot be compiled")
        embedder, reranker = kb.embedder, kb.reranker

        def embed(text: str) -> np.ndarray:
            return embedder.encode([normalize_text(text)], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        return cls(embed=embed, rerank=reranker.predict, index=kb.index, chunks=list(kb.chunks), metadata=list(kb.metadata), source=source,
                   section_map=section_map_for(source))


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


class FacultyServicesSpecialist(Specialist):
    """Faculty Services & HR Specialist, deterministic version 1."""

    specialist_id = SpecialistId.FACULTY_SERVICES

    def __init__(self, resources: FacultyServicesResources, *, owned_sections: Sequence[Tuple[int, str, str]] = SCOPE_SECTIONS,
                 extra_pages: Sequence[Tuple[int, int]] = EXTRA_PAGES) -> None:
        if resources.source.source_id != APPROVED_SOURCE_ID or not resources.source.registered:
            raise ValueError("the Faculty Services & HR specialist may only use the registered handbook %r" % APPROVED_SOURCE_ID)
        self.resources = resources
        self.owned_sections = tuple(owned_sections)
        self.scope = compile_scope(resources.section_map, APPROVED_SOURCE_ID, self.owned_sections, extra_pages)
        self.guard = SectionGuard(resources.chunks, resources.metadata, resources.section_map, self.owned_sections, self.scope)

    @classmethod
    def from_knowledge_base(cls, kb) -> "FacultyServicesSpecialist":
        return cls(FacultyServicesResources.from_knowledge_base(kb))

    # ---- input checks -------------------------------------------------------
    def _check_task(self, task: SpecialistTask) -> None:
        if not isinstance(task, SpecialistTask):
            raise TypeError("run() expects a SpecialistTask, got %s" % type(task).__name__)
        if task.specialist_id is not None and task.specialist_id != self.specialist_id:
            raise ValueError("task %s is assigned to %s, not to %s" % (task.task_id, task.specialist_id.value, self.specialist_id.value))

    @staticmethod
    def _source_scope_problem(task: SpecialistTask) -> Optional[str]:
        unapproved = [s for s in task.source_scope if s != APPROVED_SOURCE_ID]
        if unapproved:
            return "task source_scope names sources the Faculty Services & HR specialist is not approved to use: %s" % ", ".join(unapproved)
        return None

    @staticmethod
    def _focus_clauses(task: SpecialistTask) -> List[str]:
        raw = task.context.get("focus_clauses") if isinstance(task.context, dict) else None
        clauses = [c.strip() for c in raw if isinstance(c, str) and c.strip()] if isinstance(raw, (list, tuple)) else []
        return _c12_attach_leads(clauses) or [task.question.strip()]                # D-099: a split-off lead rejoins its clause

    @staticmethod
    def _route(task: SpecialistTask) -> str:
        return task.intent if task.intent in ALL_ROUTES and task.intent != "greeting" else "policy"

    # ---- evidence -----------------------------------------------------------
    def _evidence(self, clause: str, route: str) -> List[Dict[str, Any]]:
        embedding = np.asarray(self.resources.embed(clause), dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        candidates = gather_candidates(clause, route, embedding, self.resources.index, self.resources.chunks, self.resources.metadata, scope=self.scope)
        if not candidates:
            return []
        scores = self.resources.rerank([[clause, c["chunk"]] for c in candidates])
        items: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            items.append(item)
        items.sort(key=lambda x: (x["rerank_score"], x.get("routing_boost", 0.0), x.get("lexical_score", 0.0), x.get("dense_score", 0.0)), reverse=True)
        items = deduplicate_by_text(items)
        return [item for item in items if item["rerank_score"] >= MIN_RERANK_SCORE]

    @staticmethod
    def _shape_problem(item: Dict[str, Any]) -> Optional[str]:
        """Printed prose lines are not evidence (the paragraph chunk carries the same text as sentences); a salary-table
        line is."""
        meta, chunk = item["meta"], item["chunk"]
        kind = meta.get("chunk_type")
        if kind in ("row", "row_window") and not any(is_table_row(chunk[s:e].strip()) for s, e in _spans(chunk, kind)):
            return "printed prose line, not an evidence unit"
        return None

    def _finding(self, task: SpecialistTask, number: int, cand: _Candidate, clause_index: int, route: str, details: Sequence[Detail]) -> Finding:
        item, quote = cand.item, cand.quote
        meta = item["meta"]
        if quote not in item["chunk"]:
            raise ValueError("quote is not a substring of the retrieved chunk")
        if meta.get("source_id") != APPROVED_SOURCE_ID or not self.scope.allows(meta):
            raise ValueError("retrieved chunk %r is outside the approved scope" % meta.get("chunk_id"))
        page = meta.get("page")
        shared = self.guard.is_shared(meta)
        metadata: Dict[str, Any] = {
            "chunk_id": meta.get("chunk_id"), "chunk_type": meta.get("chunk_type"), "chapter": meta.get("chapter"),
            "section_no": meta.get("section_no"), "section_title": meta.get("section_title"), "section_label_page_level": True,
            "page_section_nos": list(meta.get("page_section_nos") or []), "shared_page": shared,
            "boundary_guard": "attributed to an owned section" if shared else "page carries owned sections only",
            "clauses": [clause_index], "route": route, "rerank_score": item["rerank_score"],
            "score_semantics": "cross-encoder rerank score; heuristic, uncalibrated, not a probability",
            "relevance": {"anchors": cand.anchors, "vocabulary": cand.vocab},
            "details_covered": sorted(d.label for d in details if d.family in cand.covered or ("category:" + d.label) in cand.covered),
            "extractive": True,
        }
        return Finding(finding_id="%s-f%d" % (task.task_id, number), claim=quote, evidence_quote=quote, source_id=meta["source_id"],
                       source_title=str(meta.get("source_title") or self.resources.source.title),
                       page=page if isinstance(page, int) and not isinstance(page, bool) and page >= 1 else None, metadata=metadata)

    # ---- results ------------------------------------------------------------
    def _result(self, task: SpecialistTask, status: FindingStatus, *, findings=(), missing=(), handoffs=(), limitations=(), summary: str,
                metadata: Dict[str, Any], started: float) -> SpecialistFindings:
        confidence = {FindingStatus.SUPPORTED: 1.0, FindingStatus.PARTIAL: 0.5}.get(status, 0.0)
        metadata = dict(metadata)
        metadata["confidence_semantics"] = "ordinal by status: 1.0 supported, 0.5 partial, 0.0 otherwise; not a probability"
        metadata["scope"] = self.scope.describe()
        metadata["boundary"] = {"shared_pages": sorted(self.guard.shared_pages), "unresolved_pages": list(self.guard.unresolved_pages)}
        return SpecialistFindings(specialist_id=self.specialist_id, task_id=task.task_id, status=status, findings=list(findings), summary=summary,
                                  missing=list(missing), handoff_requests=list(handoffs), confidence=confidence, limitations=list(limitations),
                                  llm_used=False, ms=(time.perf_counter() - started) * 1000.0, metadata=metadata)

    def _handoff(self, task: SpecialistTask, target: SpecialistId, clause: str, index: int, kind: str, ordinal: int = 0) -> HandoffRequest:
        suffix = "%d" % index if ordinal == 0 else "%d-%d" % (index, ordinal)
        return HandoffRequest(
            from_specialist=self.specialist_id, requested_specialist=target,
            reason="%s clause %d belongs to %s: %s" % (kind, index, target.value, clause[:120]),
            task=SpecialistTask(task_id="%s-handoff-%s" % (task.task_id, suffix), question=task.question, specialist_id=target, domain=target.value,
                                intent=task.intent, level=task.level, system=task.system,
                                context={"full_question": task.question, "focus_clauses": [clause]}, requested_by=self.specialist_id.value,
                                metadata={"origin_task_id": task.task_id, "clause_index": index, "kind": kind}),
            metadata={"clause_index": index, "kind": kind})

    # ---- one clause -----------------------------------------------------------
    def _candidates(self, clause: str, route: str, clause_concepts: Concepts, details: Sequence[Detail], rejected: List[Dict[str, Any]]) -> List[_Candidate]:
        found: List[_Candidate] = []
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
            if quote_problem(quote, "prose") == "lead-in":
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": "list lead-in, not an evidence unit"})
                continue
            covered = unit_coverage(details, clause_concepts, quote)
            found.append(_Candidate(rank, item, quote, anchors, vocab, covered))
        keys = frozenset(_detail_key(d) for d in details)
        found.sort(key=lambda c: (tuple(-x for x in _coverage_key(c.covered, keys)), c.rank))
        return found

    # ---- public API -----------------------------------------------------------
    def run(self, task: SpecialistTask) -> SpecialistFindings:
        started = time.perf_counter()
        self._check_task(task)
        problem = self._source_scope_problem(task)
        if problem:
            return self._result(task, FindingStatus.ERROR, summary="refused: unapproved source scope", metadata={"error": problem}, started=started)

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
        processes_seen = 0
        missing_sources_seen: List[str] = []
        personal_seen = 0
        status_seen = 0

        try:
            for index, clause in enumerate(processed):
                mine, other = classify_clause(clause)
                navigation, substantive = navigation_intent(clause)
                if navigation and not substantive:
                    mine, other = False, SpecialistId.INSTITUTIONAL     # a pure place or contact ask is never answered from HR text
                if not mine:
                    if other is not None:
                        handoffs.append(self._handoff(task, other, clause, index, "misrouted"))
                        reports.append({"index": index, "clause": clause, "outcome": "handed_off", "target": other.value})
                    else:
                        reports.append({"index": index, "clause": clause, "outcome": "unowned", "target": None})
                        missing.append("clause %d carries no faculty services or HR cue and was not searched: %s" % (index, clause[:120]))
                    continue
                owned_clauses += 1
                clause_route = resolve_clause_retrieval_intent(clause, task.intent, sole_clause=len(processed) == 1, substantive=substantive)
                handoffs.extend(self._handoff(task, o, clause, index, "shared", n) for n, o in enumerate(other_domains(clause)))
                procedural, system = is_procedure_request(clause, task.system)
                process = not procedural and is_process_request(clause)
                missing_source = named_missing_source(clause)
                personal = personal_record(clause)
                universal = universality_detail(clause)
                condition = condition_detail(clause)
                predicate = predicate_detail(clause)
                negative = negative_detail(clause)
                polarity = polarity_detail(clause)
                assertion = assertion_detail(clause)
                claim, claim_details, composite = _specific_claim_support_gate(clause)    # D-099
                if system and system not in systems_seen:
                    systems_seen.append(system)
                clause_concepts = concepts(clause)
                details = requested_details(ask_text(clause))
                ask = assess_ask(clause)
                if not procedural:
                    details = list(details) + list(ask.details)
                    hr = hr_fact_detail(clause, concepts)
                    if hr is not None:
                        details.append(hr)
                    salary = salary_detail(clause)
                    if salary is not None:
                        details.append(salary)
                if navigation:
                    details.append(Detail("contact", "contact or location details", contact_evidence))
                if process:
                    processes_seen += 1
                    details.append(Detail("process", "the complete procedure (the handbook states stages only)", lambda unit: False))
                if missing_source:
                    missing_sources_seen.append(missing_source)
                    details.append(Detail("named_source", "the contents of %s (not in the indexed sources)" % missing_source, lambda unit: False))
                if personal:
                    if personal == "status":
                        status_seen += 1
                    else:
                        personal_seen += 1
                    details.append(Detail("personal_record", _personal_label(personal), lambda unit: False))
                if universal is not None:
                    details.append(universal)
                if condition is not None:
                    details.append(condition)
                if predicate is not None:
                    details.append(predicate)
                if negative is not None:
                    details.append(negative)
                if polarity is not None:
                    details.append(polarity)
                if assertion is not None:
                    details.append(assertion)
                details.extend(claim_details)
                if composite is not None:
                    details.append(_supporting_only(composite, details))
                rejected: List[Dict[str, Any]] = []
                clause_findings: List[str] = []
                best_covered: FrozenSet[str] = frozenset()
                best_quote: Optional[str] = None
                detail_keys = frozenset(_detail_key(d) for d in details)
                seen_anywhere: set = set()
                for cand in self._candidates(clause, clause_route, clause_concepts, details, rejected):
                    if len(clause_findings) >= MAX_FINDINGS_PER_CLAUSE:
                        break
                    key = _quote_key(cand.quote)
                    page = cand.item["meta"].get("page")
                    duplicate = next((f for f in findings if f.page == page and (key in _quote_key(f.evidence_quote) or _quote_key(f.evidence_quote) in key)), None)
                    if duplicate is not None:
                        if index not in duplicate.metadata["clauses"]:
                            duplicate.metadata["clauses"].append(index)
                        if duplicate.finding_id not in clause_findings:
                            clause_findings.append(duplicate.finding_id)
                        if _coverage_key(cand.covered, detail_keys) > _coverage_key(best_covered, detail_keys):
                            best_covered, best_quote = cand.covered, cand.quote
                        elif best_quote is None:
                            best_quote = cand.quote
                        seen_anywhere |= set(cand.covered)
                        continue
                    finding = self._finding(task, len(findings) + 1, cand, index, clause_route, details)
                    findings.append(finding)
                    clause_findings.append(finding.finding_id)
                    if _coverage_key(cand.covered, detail_keys) > _coverage_key(best_covered, detail_keys):
                        best_covered, best_quote = cand.covered, cand.quote
                    elif best_quote is None:
                        best_quote = cand.quote
                    seen_anywhere |= set(cand.covered)
                uncovered = [d for d in details if _detail_key(d) not in best_covered]
                if not clause_findings:
                    outcome = "not_found"
                    if procedural:
                        missing.append("step-by-step procedure for %s requested by clause %d is not in the indexed sources: %s" % (system, index, clause[:120]))
                    else:
                        missing.append("no evidence in the Faculty Services & HR scope for clause %d: %s" % (index, clause[:120]))
                elif procedural or uncovered:
                    outcome = "partial"
                    if procedural:
                        missing.append("step-by-step procedure for %s requested by clause %d is not in the indexed sources: %s" % (system, index, clause[:120]))
                    for d in uncovered:
                        if d.family == "ask_unknown":
                            missing.append("the specific ask of clause %d is not recognised by the Faculty Services & HR specialist; the quoted evidence is "
                                           "related to the topic but does not confirm the answer: %s" % (index, clause[:120]))
                        elif d.family == "grounding":
                            missing.append("%s: the quoted evidence is on the topic but does not settle the question for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "hr_fact":
                            missing.append("%s: the quoted evidence states the topic but not this fact in one place for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "salary_row":
                            missing.append("%s is not the row quoted for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "process":
                            missing.append("the complete procedure for clause %d is not in the indexed sources; the quoted evidence states the documented "
                                           "stages only: %s" % (index, clause[:120]))
                        elif d.family == "named_source":
                            missing.append("%s: the indexed handbook states only that it exists and what it covers, not its contents, for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "personal_record" and d.label == _STATUS_LABEL:
                            missing.append("the asker's own grant, approval, acceptance or decision status is not available in the indexed handbook (it "
                                           "holds policy, not a person's case or workflow state) for clause %d: %s" % (index, clause[:120]))
                        elif d.family == "personal_record":
                            missing.append("personal leave usage or balance is not available in the indexed handbook (it holds policy, not a person's record) "
                                           "for clause %d: %s" % (index, clause[:120]))
                        elif d.family == "universality":
                            missing.append("%s is not stated: the evidence states the topic, not that it is the same or independent, for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "polarity":
                            missing.append("%s is not stated: the evidence negates it for the asked subject, for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "assertion":
                            missing.append("%s is not stated: the evidence states it only inside a condition (if / when / unless), for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "modality":
                            missing.append("%s is not stated: the evidence states a weaker or another modality for the asked action, or "
                                           "none, for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "scope":
                            missing.append("%s is not stated: the evidence states it for a narrower or an unquantified population, or the "
                                           "returned quote excepts part of it, for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "exclusivity":
                            missing.append("%s is not stated: the evidence does not state it as exclusive, or states an extension or an "
                                           "alternative, for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "condition_role":
                            missing.append("%s is not stated: the evidence does not state the condition in the asked role, or the returned "
                                           "quote negates the asked predicate for the population the condition names, for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "operator":
                            missing.append("%s is not verified: the operator is recognised but not verified by the Faculty Services & HR "
                                           "specialist, for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "claim":
                            explain = getattr(d.check, "explain", None)
                            missing.append("%s, for clause %d: %s" % (explain(best_quote) if explain is not None else d.label, index, clause[:120]))
                        elif d.family == "negative":
                            missing.append("%s is not stated: the evidence states the affirmative or another subject, for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "predicate":
                            missing.append("%s is not stated: the evidence negates it or states it of another subject, for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "condition":
                            missing.append("%s is not stated: the evidence states the condition as required or given, or does not address it, "
                                           "for clause %d: %s" % (d.label, index, clause[:120]))
                        elif d.family == "contact":
                            missing.append("contact or location details are not in the Faculty Services & HR evidence for clause %d; Institutional "
                                           "Navigation holds them: %s" % (index, clause[:120]))
                        elif _detail_key(d) in seen_anywhere:
                            missing.append("%s is not stated together with the other requested details in one place for clause %d: %s" % (d.label, index, clause[:120]))
                        else:
                            missing.append("%s not found in the evidence for clause %d: %s" % (d.label, index, clause[:120]))
                else:
                    outcome = "supported"
                reports.append({"index": index, "clause": clause, "outcome": outcome, "procedural": procedural, "process": process, "system": system,
                                "findings": clause_findings, "rejected": rejected, "requested_details": [d.label for d in details],
                                "uncovered_details": [d.label for d in uncovered], "ask_shape": ask.shape, "route": clause_route,
                                "shared_with": other.value if other is not None else None})
        except Exception as exc:
            return self._result(task, FindingStatus.ERROR, summary="execution failed", handoffs=handoffs,
                                metadata={"error": _safe_error(exc), "clauses": reports}, started=started)

        for index, clause in enumerate(unprocessed, start=len(processed)):
            missing.append("clause %d was not processed (limit of %d clauses per task): %s" % (index, MAX_CLAUSES, clause[:120]))
        for system in systems_seen:
            limitations.append("The indexed handbook contains policy-level references to %s only; no procedural guide is indexed, so step-by-step "
                               "instructions cannot be provided." % system)
        if processes_seen:
            limitations.append("%d clause(s) asked for a procedure; the handbook states the documented stages only, not forms, portal steps or timelines."
                               % processes_seen)
        if personal_seen:
            limitations.append("%d clause(s) asked for the asker's own leave usage or balance; the indexed handbook states policy only and holds no "
                               "personal record." % personal_seen)
        if status_seen:
            limitations.append("%d clause(s) asked for a decision on the asker's own case (granted, approved or accepted); the indexed handbook "
                               "states policy only and holds no personal case or workflow state." % status_seen)
        for name in sorted(set(missing_sources_seen)):
            limitations.append("%s is not in the indexed sources; the handbook mentions it and lists its topics, so its contents cannot be quoted." % name)
        handed_off = [r for r in reports if r["outcome"] == "handed_off"]
        if handed_off:
            limitations.append("%d clause(s) belong to another specialist and were handed off, not answered." % len(handed_off))
        unowned = [r for r in reports if r["outcome"] == "unowned"]
        if unowned:
            limitations.append("%d clause(s) carry no faculty services or HR cue and were not searched." % len(unowned))

        metadata = {"clauses": reports, "unprocessed_clauses": unprocessed, "route": route, "owned_clauses": owned_clauses, "gate": MIN_RERANK_SCORE}
        if owned_clauses == 0:
            return self._result(task, FindingStatus.OUT_OF_SCOPE, handoffs=handoffs, limitations=limitations, missing=missing,
                                summary="no assigned clause belongs to Faculty Services & HR", metadata=metadata, started=started)
        if findings and not missing:
            status, summary = FindingStatus.SUPPORTED, "%d clause(s) supported by %d finding(s)" % (owned_clauses, len(findings))
        elif findings:
            status, summary = FindingStatus.PARTIAL, "%d finding(s); %d gap(s)" % (len(findings), len(missing))
        else:
            status, summary = FindingStatus.NOT_FOUND, "no evidence in the Faculty Services & HR scope for the assigned clause(s)"
        return self._result(task, status, findings=findings, missing=missing, handoffs=handoffs, limitations=limitations, summary=summary,
                            metadata=metadata, started=started)


__all__ = [
    "ANCHOR_CONCEPTS", "APPROVED_SOURCE_ID", "ASK_GATE", "ASK_GENERAL", "ASK_RECOGNISED", "ASK_UNKNOWN", "CATEGORY_GROUPS", "Concepts", "EXCLUDED_SECTIONS",
    "EXTRA_PAGES", "FacultyServicesResources", "FacultyServicesSpecialist", "GENERIC_TERMS", "GROUNDING_ROLES", "HRAsk", "MAX_CANDIDATES_PER_CLAUSE",
    "MAX_CLAUSES", "MAX_FINDINGS_PER_CLAUSE", "MAX_QUOTE_WORDS", "OWNED_SECTIONS", "SCOPE_SECTIONS", "SHARED_SECTIONS", "SYSTEMS", "SectionGuard",
    "VOCABULARY_CONCEPTS", "asked_rank", "assess_ask", "classify_clause", "clause_segments", "compile_scope", "concept_residue", "concept_spans",
    "ask_text", "concepts", "contact_evidence", "detail_covered", "hr_ask", "hr_fact_detail", "hr_items", "is_procedure_request", "is_process_request",
    "named_missing_source", "personal_record", "qualified_rank", "universal_ask", "universality_bound", "universality_detail", "universality_frame",
    "UniversalAsk", "PassiveAsk", "passive_ask", "AskCondition", "ask_condition", "condition_detail",
    "ActiveAsk", "active_ask", "PredicateAsk", "predicate_ask", "predicate_detail", "NegativeAsk", "negative_ask", "negative_detail",
    "LabelAsk", "label_ask", "AskStructure", "ask_structure", "ClaimPredicate", "claim_predicate", "polarity_detail", "Antecedent", "affirmative_claim", "assertion_detail",
    "read_claim", "read_statements", "structured_details", "modality_detail", "scope_detail", "exclusivity_detail", "condition_role_detail",
    "operator_detail", "strip_leading_frame", "read_clause_statements", "read_claim_frame", "claim_detail", "explain_claim",
    "is_specific_claim", "specific_claim_text",
    "is_relevant", "is_salary_data_row", "is_salary_header", "is_table_row", "leave_types_of", "money_facts", "navigation_intent", "other_domains", "pay_statuses_of", "quote_problem",
    "relevance", "requested_details", "salary_detail", "salary_row", "select_quote", "system_named", "unit_coverage",
]
