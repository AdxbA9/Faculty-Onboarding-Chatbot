"""
Teaching & Learning Specialist (canonical id: ``teaching``).

DECISION OWNED
    Whether the approved, source-scoped evidence answers the teaching part of
    a faculty question, expressed as structured findings. Nothing else: the
    specialist does not route the question (the Coordinator did), does not
    write the user-facing answer (Synthesis will), does not verify (the
    Verifier will) and never calls another specialist.

VERSION 1 IS DETERMINISTIC AND EXTRACTIVE
    No LLM call is made anywhere in this module (``llm_used`` is always
    False). Every finding is extractive: ``claim`` is the quoted handbook
    text itself and ``evidence_quote`` is an exact slice of a retrieved
    chunk. Nothing is paraphrased, so nothing can be invented.

SIX GATES BEFORE A CHUNK SUPPORTS A CLAUSE
    1. source and page scope: the shared ``RetrievalScope`` compiled from the
       section map for ``OWNED_SECTIONS`` (page-granular, first boundary);
    2. section eligibility: on a page that also carries a non-owned section,
       the chunk's own text is attributed to a section from the section
       headings printed on that page (``SectionGuard``); text of a non-owned
       section, or text that cannot be attributed, is rejected;
    3. relevance: the quoted unit must share one anchor concept or two
       vocabulary concepts with the focus clause (``is_relevant``); generic
       words never count. Relevance establishes the TOPIC only;
    4. the production reranker gate ``MIN_RERANK_SCORE`` on every kept item;
    5. quote quality: headings, lead-ins, dangling line fragments, truncated
       sentences and antecedent-less sentences are not evidence
       (``quote_problem``, ``select_quote``);
    6. completeness: when the clause explicitly asks for a detail (fee,
       penalty, deadline, quantity, minimum, maximum, approving authority,
       responsible party, faculty category, percentage, part- or full-time,
       location or system, an explicit requirement), the clause is fully
       ``supported`` only when the evidence contains that detail next to the
       subject (``requested_details``, ``detail_covered``); otherwise the
       clause is ``partial`` and ``missing`` names the detail. A number
       counts as a value only when it is tied to a unit, a percent sign or
       a limit phrase, never a year, a section number or an identifier.
    Scope membership alone never implies ownership or support.

SYSTEMS
    Blackboard, Banner and MyUOS are systems, not agents. The handbook holds
    policy-level statements about them and no procedural guide. A clause
    that asks for a procedure about a system is at most ``partial``, and only
    when the retained evidence names the system or shares an anchor concept
    with the request; otherwise ``not_found``. Instructions are never
    generated.

TASKS
    ``task.question`` is the full original question; the assigned work is
    ``task.context["focus_clauses"]`` (the whole question when absent). A
    clause is handled only when it carries a positive teaching cue; a clause
    with another domain's cue is returned as a ``HandoffRequest``; a clause
    with neither is reported as unowned and not searched. The Coordinator is
    never called from here. One task carries one intent for all its clauses;
    per-clause intents are a Coordinator concern (deferred, see docs).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ...config import MIN_RERANK_SCORE, PLAN_F_MAX_SUBTASKS
from ...retrieval import RetrievalScope, deduplicate_by_text, gather_candidates
from ...sources import SectionMap, SectionRecord, SourceRecord, section_map_for
from ...text_utils import normalize_text, tokenize
from ..types import ALL_ROUTES
from .base import Specialist
from .contracts import Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask

#: The only source this specialist may retrieve from.
APPROVED_SOURCE_ID = "uos_faculty_handbook_2025_26"

#: Handbook sections the specialist owns: (chapter, printed number, label).
#: Chapter AND number identify a section because the handbook prints some
#: numbers twice; both printed 5.2 sections are teaching development and are
#: included. Everything not listed is outside the default scope.
OWNED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (1, "1.14", "Academic Calendar"),
    (3, "3.1", "Teaching Responsibilities"),
    (3, "3.2", "Responsibilities in Student Advising and Availability"),
    (3, "3.3", "Responsibility to Curriculum Development"),
    (3, "3.4", "Responsibility to Office Hours"),
    (3, "3.7", "Professional Development Responsibilities"),
    (3, "3.10", "Academic Freedom and Responsibility"),
    (3, "3.11", "Workload Allocation Model (WLAM)"),
    (3, "3.12", "Academic Integrity and Student Honor Code"),
    (5, "5.1", "Goals and Objectives"),
    (5, "5.2", "Categories of Faculty Development; New Faculty Development Training"),
    (5, "5.3", "Faculty Peer Observation of Teaching"),
    (5, "5.5", "Policy on Offering Courses in the Summer Semester"),
    (10, "10.6", "Learning Management System (LMS) Policy"),
    (12, "12.7", "Curricula Approval and Revision Policy"),
    (12, "12.12", "Examinations Policy"),
    (12, "12.13", "Student Assessment and Grading System"),
    (12, "12.14", "Grading System"),
    (12, "12.15", "Student Attendance and Assessment Policies"),
    (12, "12.16", "Penalties for Academic Misconduct During Exams"),
    (12, "12.17", "Instructor Responsibilities"),
    (12, "12.18", "Class Size Policy"),
    (16, "16.6", "Guidelines on Artificial Intelligence and Large Language Models"),
)

#: Sections deliberately kept out of the default scope because their
#: ownership is shared or unclear. Listed so the decision is visible.
EXCLUDED_SECTIONS: Tuple[Tuple[int, str, str], ...] = (
    (2, "2.9", "Emirati-track appointment, including the 2.9.1.3 load reduction"),
    (3, "3.5", "Responsibility to Research"),
    (3, "3.6", "Responsibility to Community Service"),
    (5, "5.4", "Participation in Conferences"),
    (6, "6.2", "Teaching and Learning Evaluation"),
    (6, "6.3", "Academic Advising (evaluation)"),
    (12, "12.8", "Internship Policy"),
    (12, "12.9", "Undergraduate Completion Policy"),
    (12, "12.10", "Graduate Completion Requirements Policy"),
    (12, "12.11", "Academic Progress Policy"),
    (12, "12.19", "Student Code of Honor and Disciplinary Policy"),
    (12, "12.20", "Student Rights and Responsibilities Policy"),
)

#: University systems the handbook mentions. Never specialists.
SYSTEMS: Tuple[str, ...] = ("blackboard", "banner", "myuos")

#: At most this many findings per focus clause.
MAX_FINDINGS_PER_CLAUSE = 2
#: At most this many relevant, quotable candidates are examined per clause
#: before the findings are chosen (coverage of requested details first).
MAX_CANDIDATES_PER_CLAUSE = 10
#: Focus clauses handled in one run; the architecture's subtask limit.
MAX_CLAUSES = PLAN_F_MAX_SUBTASKS
#: A quote extended by its neighbouring unit may not exceed this many words.
MAX_QUOTE_WORDS = 90

_KEY_SHAPED = re.compile(r"gsk_[A-Za-z0-9]+")

#: Hyphen-like separators between words are read as spaces before concept
#: and cue matching ("office-hours", "add/drop", "e-learning", "part-time"),
#: the rule the Coordinator applies to its own cues. Same length, so nothing
#: shifts; quotes and evidence text are never rewritten.
_CUE_SEPARATORS = re.compile(r"(?<=\w)[-‐‑‒–—/](?=\w)")


def cue_text(text: str) -> str:
    """``text`` as the cue and concept patterns read it."""
    return _CUE_SEPARATORS.sub(" ", text)


# ---------------------------------------------------------------------------
# Teaching concept vocabulary (relevance)
# ---------------------------------------------------------------------------
#: (pattern, concept, anchor). Patterns run in order on lower-cased text and
#: each matched span is consumed, so a word maps to one concept. Multi-word
#: phrases come first. An ANCHOR concept is specific enough to establish
#: topic relevance on its own; a plain vocabulary concept needs a second
#: match. No concept establishes answer completeness (see the detail gate).
_CONCEPT_PATTERNS: Tuple[Tuple[str, str, bool], ...] = (
    (r"office\s+hours?", "office_hours", True),
    (r"credit\s+hours?", "credit_hours", True),
    (r"contact\s+hours?", "contact_hours", True),
    (r"(?:teaching|instructional)\s+loads?|workload\s+allocation|workloads?|\bwlam\b", "teaching_load", True),
    (r"\bcrns?\b", "crn", True),
    (r"add\s*[/-]?\s*drop|add\s+(?:and|or)\s+drop|drop(?:ping)?\s+(?:a\s+|the\s+)?courses?", "add_drop", True),
    (r"final\s+exam(?:ination)?s?\b", "final_exam", True),
    (r"make[- ]?up\s+exam(?:ination)?s?\b", "makeup_exam", True),
    (r"midterms?\b", "midterm", True),
    (r"(?:exam|examination)\s+(?:schedule|period|dates?|timetable|week)", "exam_schedule", True),
    # "exam rules", "examination regulations": the topic anchor; bare "policy" or "rules" is generic
    (r"exam(?:ination)?s?\s+(?:policy|policies|rules|regulations|procedures?|guidelines|conduct|instructions)\b", "exam_policy", True),
    (r"re-?\s?marking|re-?\s?grad(?:e|es|ing)\b|grade\s+appeals?", "regrade", True),
    (r"peer\s+observation", "peer_observation", True),
    (r"academic\s+integrity", "academic_integrity", True),
    (r"hono?u?r\s+code", "honor_code", True),
    (r"class\s+sizes?", "class_size", True),
    (r"academic\s+freedom", "academic_freedom", True),
    (r"academic\s+calendar", "calendar", True),
    (r"learning\s+management\s+system|\blms\b|\be-?\s?learning\b|\belearning\b", "lms", True),      # e-learning: the LMS policy's topic
    (r"blackboard|bb\s+ultra", "blackboard", True),
    (r"\bbanner\b", "banner", True),
    (r"my\s?uos\b", "myuos", True),
    (r"grading\s+(?:system|scale|scheme|policy|policies|rules|criteria|regulations|procedures?|guidelines)|grade\s+(?:scale|points?)\b", "grading_system", True),
    (r"incomplete\s+grades?|\bincompletes?\b", "incomplete_grade", True),
    (r"e-?\s?course\s+files?|course\s+files?", "course_file", True),
    (r"(?:classes|semester|term|courses)\s+(?:begin|begins|start|starts|commence|commences)|"
     r"(?:beginning|start)\s+of\s+(?:classes|the\s+semester|the\s+term)|first\s+day\s+of\s+classes", "term_start", True),
    (r"(?:classes|semester|term)\s+ends?\b|last\s+day\s+of\s+classes", "term_end", True),
    (r"summer\s+(?:semester|courses?|teaching|classes|session)", "summer_teaching", True),
    (r"artificial\s+intelligence|large\s+language\s+models?|generative\s+ai|\bllms?\b|\bai\b", "ai", True),
    (r"plagiaris\w*", "plagiarism", True),
    (r"cheat\w*", "cheating", True),
    # classroom attendance: the noun, absence, or "attend" applied to classes, lectures, sessions or exams
    (r"attendance|absences?|\babsent\b|\battend(?:s|ed|ing)?\s+(?:all\s+)?(?:the\s+|their\s+|scheduled\s+)?(?:theoretical\s+|practical\s+|online\s+)?"
     r"(?:lectures?|classes|class|sessions?|labs?|laborator(?:y|ies)|examinations?|exams?|courses?|tutorials?|seminars?)\b", "attendance", True),
    (r"syllab(?:us|i|uses)\b", "syllabus", True),
    (r"curricul(?:um|a|ar)\b", "curriculum", True),
    (r"\bc?gpa\b", "gpa", True),
    (r"withdraw\w*", "withdrawal", True),
    (r"proctor\w*|invigilat\w*", "proctoring", True),
    (r"rubrics?\b", "rubric", True),
    (r"(?:thesis|dissertation)\s+supervis\w*|supervis\w*\s+(?:of\s+)?(?:thesis|theses|dissertations?)", "thesis_supervision", True),
    (r"teaching\s+assistants?\b", "teaching_assistant", True),
    (r"consultation\s+(?:hours?|times?)|student\s+consultations?", "consultation_hours", True),
    (r"professional\s+development|faculty\s+development|teaching\s+development|instructional\s+development|development\s+training|"
     r"training\s+(?:modules?|workshops?|programs?|programmes?|sessions?)\b|(?:pedagogical|teaching)\s+training|"
     r"training\s+(?:must|should|do|does|for|of)\s+(?:new\s+)?faculty|(?:new\s+)?faculty\s+training", "faculty_development", True),
    (r"new\s+faculty\b", "new_faculty", True),
    # ---- plain vocabulary: needs a second match --------------------------
    (r"\battend(?:s|ed|ing)?\b", "attend", False),
    (r"fall\s+semester|\bfall\b|\bautumn\b", "fall", False),
    (r"spring\s+semester|\bspring\b", "spring", False),
    (r"summer\s+semester|\bsummer\b", "summer", False),
    (r"academic\s+year", "academic_year", False),
    (r"part[- ]time", "part_time", False),
    (r"full[- ]time", "full_time", False),
    (r"exam(?:ination)?s?\b", "exam", False),
    (r"\bgrad(?:e|es|ed|ing)\b|\bmarks?\b|\bmarking\b", "grade", False),
    (r"teach\w*|\btaught\b", "teach", False),
    (r"instructors?\b", "instructor", False),
    (r"lecturers?\b", "lecturer", False),
    (r"lectures?\b|lecturing", "lecture", False),
    (r"assess\w*", "assessment", False),
    (r"advis(?:e|es|ed|ing|ers?|ors?|ees?)\b", "advise", False),
    (r"class(?:es|rooms?)?\b", "class", False),
    (r"courses?\b|coursework", "course", False),
    (r"responsib\w*", "responsibility", False),
    (r"upload\w*", "upload", False),
    (r"submi(?:t|ts|tted|tting|ssions?)\b", "submit", False),
    (r"evaluat\w*", "evaluation", False),
    (r"observ\w*", "observation", False),
    (r"\bloads?\b", "load", False),
    (r"credits?\b", "credit", False),
    (r"enrol\w*", "enrollment", False),
    (r"regist\w*", "registration", False),
    (r"schedul\w*|timetables?\b", "schedule", False),
    (r"\bhours?\b", "hour", False),
    (r"\bweek(?:s|ly)?\b", "week", False),
    (r"quiz(?:zes)?\b", "quiz", False),
    (r"deadlines?\b", "deadline", False),
    (r"penalt(?:y|ies)\b", "penalty", False),
    (r"laborator(?:y|ies)\b|\blabs?\b", "lab", False),
    (r"tutorials?\b", "tutorial", False),
    (r"seminars?\b", "seminar", False),
    (r"\bthes[ie]s\b|dissertations?\b", "thesis", False),
    (r"modules?\b", "module", False),
    (r"\btraining\b", "training", False),
    (r"professional\s+development|\bdevelopment\b", "development", False),
    (r"holidays?\b|\bbreak\b", "holiday", False),
    (r"\bfeedback\b", "feedback", False),
    (r"learning\s+outcomes?|\boutcomes?\b", "outcome", False),
    (r"objectives?\b", "objective", False),
    (r"recordings?\b|\brecorded\b", "recording", False),
    (r"\bonline\b", "online", False),
    (r"\bevening\b", "evening", False),
    (r"approv\w*", "approval", False),
    (r"councils?\b|committees?\b", "council", False),
    (r"\bdeans?\b|\bchairs?\b|chairpersons?\b", "dean_chair", False),
    (r"\bmaximum\b|\bminimum\b|\blimits?\b", "limit", False),
    (r"materials?\b|textbooks?\b", "material", False),
    (r"modif(?:y|ies|ied|ication|ications)\b|revis(?:e|es|ed|ion|ions)\b|changes?\b", "modification", False),
    (r"\bnew\s+courses?\b|course\s+(?:proposal|design|development)", "course_design", False),
    (r"innovat\w*", "innovation", False),
    (r"mentor\w*", "mentoring", False),
    (r"sections?\b", "section", False),
    (r"workshops?\b|lessons?\b", "workshop", False),
)
_COMPILED_CONCEPTS = tuple((re.compile(p, re.I), c, a) for p, c, a in _CONCEPT_PATTERNS)
ANCHOR_CONCEPTS: FrozenSet[str] = frozenset(c for _, c, a in _CONCEPT_PATTERNS if a)
VOCABULARY_CONCEPTS: FrozenSet[str] = frozenset(c for _, c, _ in _CONCEPT_PATTERNS)

#: A topic anchor built on a base word ("grading policy" on grade, "exam
#: rules" on exam) also carries that base as plain vocabulary, and a text
#: that uses the base word is on the anchor's topic: "What is the grading
#: policy?" is answered by sentences about grades, not only by sentences
#: that repeat the phrase "grading policy". Nothing generic is implied.
_ANCHOR_IMPLIES: Dict[str, str] = {"grading_system": "grade", "exam_policy": "exam"}

#: Words that never establish relevance on their own. They carry no concept
#: (they are absent from the vocabulary above); the list documents the
#: decision and is checked by tests.
GENERIC_TERMS: FrozenSet[str] = frozenset({
    "policy", "policies", "process", "procedure", "require", "required", "requirement", "requirements",
    "contact", "fee", "fees", "assignment", "assignments", "information", "faculty", "student", "students",
    "university", "member", "members", "academic", "rule", "rules", "guideline", "guidelines", "semester",
    "semesters", "office", "department", "college", "provide", "use", "apply", "application", "late",
    "day", "days", "date", "dates", "time", "number", "year",
})


@dataclass(frozen=True)
class Concepts:
    """Normalised concepts of a text. ``anchors`` is a subset of ``vocab``."""

    anchors: FrozenSet[str]
    vocab: FrozenSet[str]


def concepts(text: str) -> Concepts:
    """Teaching concepts found in ``text`` after normalisation (aliases such
    as syllabi/syllabus, grading/grade, examinations/exam)."""
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


#: Calendar events that exclude each other: a "classes end" line is not
#: evidence for a "classes begin" question even when both name the term.
_EXCLUSIVE_EVENTS = frozenset({"term_start", "term_end"})


def relevance(clause: Concepts, text: str) -> Tuple[int, int]:
    """(shared anchor concepts, shared vocabulary concepts) between the clause
    and ``text``; (0, 0) when the two name different exclusive events."""
    other = concepts(text)
    mine, theirs = clause.anchors & _EXCLUSIVE_EVENTS, other.anchors & _EXCLUSIVE_EVENTS
    if mine and theirs and mine != theirs:
        return 0, 0
    shared = clause.anchors & other.anchors
    # a topic anchor of the clause is also shared when the text uses its base word
    implied = {a for a in clause.anchors - other.anchors if _ANCHOR_IMPLIES.get(a) in other.vocab}
    return len(shared) + len(implied), len(clause.vocab & other.vocab)


def is_relevant(clause: Concepts, text: str) -> bool:
    """The relevance rule: one shared anchor concept, or two shared
    vocabulary concepts. Generic words carry no concept and never count.
    Relevance means the text is on the clause's topic, nothing more."""
    anchors, vocab = relevance(clause, text)
    return anchors >= 1 or vocab >= 2


# ---------------------------------------------------------------------------
# Requested details (completeness)
# ---------------------------------------------------------------------------
_NUMBER_WORD = (r"(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
                r"(?:-(?:one|two|three|four|five|six|seven|eight|nine))?")
_NUMBER = re.compile(r"(?<![\d.])\b" + _NUMBER_WORD + r"\b(?!\.\d)", re.I)     # not a section number such as 12.18.1
_FREQUENCY = re.compile(r"\bper\s+(?:week|semester|day|month|year|course|section)\b|\b(?:weekly|daily|monthly|twice|once)\b|"
                        r"\b(?:each|every)\s+(?:week|semester|day|month|year)\b", re.I)
_TIMING = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b|"
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)\b|\bwithin\s+" + _NUMBER_WORD + r"\s+(?:working\s+)?(?:hours?|days?|weeks?|months?)\b|"
    r"\b(?:at\s+the\s+)?(?:beginning|start|end)\s+of\s+(?:the\s+|each\s+|every\s+)?(?:semester|term|course|academic\s+year|first\s+week|year)\b|"
    r"\bfirst\s+(?:week|day)\b|\blast\s+day\b|\bno\s+later\s+than\b|\bprior\s+to\b|\bbefore\s+the\b|\bby\s+the\s+(?:end|start|beginning)\b|"
    r"\b\d{1,2}\s+\w+\s+20\d\d\b", re.I)
_APPROVAL_VERB = re.compile(r"\bapprov\w*|\bauthori[sz]\w*|\bendors\w*|\bsign(?:ed|s)?\s+off\b|\bratif\w*", re.I)
_AUTHORITY_NOUN = re.compile(r"\b(?:deans?|chairs?|chairpersons?|councils?|committees?|vice[- ]chancellor|chancellor|president|registrar\w*|"
                             r"department|college|head|directors?|provost|vcaa|board|coordinator)\b", re.I)
_PERCENT = re.compile(r"%|\bpercent(?:age)?\b|\b\d{1,3}\s*-\s*\d{1,3}\b|\b(?:below|above|under|over)\s+\d{1,3}\b", re.I)
_LIMIT_MAX = re.compile(r"\b(?:maximum|maximal|at\s+most|exceed\w*|no\s+more\s+than|up\s+to|limited\s+to|not\s+more\s+than|caps?|ceiling)\b", re.I)
_LIMIT_MIN = re.compile(r"\b(?:minimum|minimal|at\s+least|no\s+fewer\s+than|not\s+less\s+than|not\s+fewer\s+than)\b", re.I)
_FEE_WORD = re.compile(r"\b(?:fees?|costs?|charges?|payment|paid|pay|price)\b", re.I)
#: A monetary value: a currency next to a number, an explicit amount, or a
#: "free of charge" statement. A bare number ("examination 101") is not money.
_CURRENCY_FIRST = r"(?:aed|dhs\.?|dirhams?|usd|us\$|\$|€|£|eur|gbp)"
_CURRENCY_AFTER = r"(?:aed|dhs\.?|dirhams?|usd|dollars?|euros?|pounds?|\$)"
_MONEY = re.compile(
    r"(?<!\w)" + _CURRENCY_FIRST + r"\s*" + _NUMBER_WORD + r"\b|"
    r"\b" + _NUMBER_WORD + r"\s*" + _CURRENCY_AFTER + r"(?!\w)|"
    r"\b(?:free\s+of\s+charge|no\s+fee|no\s+charge|without\s+charge|at\s+no\s+cost|waived|free)\b|"
    r"\b(?:amount|sum|price|costs?|fees?|charges?|payment)\s+(?:of\s+|is\s+|are\s+|will\s+be\s+|shall\s+be\s+)" + _NUMBER_WORD + r"\b(?!\.\d)", re.I)
#: A number is a value when a count noun, a percent sign or a limit phrase
#: is tied to it; a year, a section number and an identifier are not values.
_YEAR_SHAPED = re.compile(r"^(?:19|20)\d\d$")
_IDENTIFIER_BEFORE = re.compile(r"\b(?:room|building|course|section|chapter|page|exam|examination|article|clause|form|item|level|floor|"
                                r"extension|ext\.?|code|id|phone|tel\.?|box|unit|no\.?|number|#)\s*$", re.I)
_UNIT_AFTER = re.compile(r"^\s*(?:%|percent(?:age)?\b|aed\b|dhs\b|dirhams?\b|usd\b|dollars?\b)", re.I)
_LIMIT_ATTACHED = re.compile(r"\b(?:maximum|maximal|minimum|minimal|at\s+least|at\s+most|up\s+to|not\s+(?:to\s+)?exceed(?:ing)?|exceed(?:s|ing)?|"
                             r"no\s+(?:more|fewer|less)\s+than|not\s+(?:more|fewer|less)\s+than|limited\s+to|capped\s+at|ceiling\s+of|caps?\s+(?:at|of))"
                             r"\s+(?:of\s+|is\s+|are\s+|to\s+|the\s+|a\s+|an\s+)?$", re.I)
_OF_SUBJECT = re.compile(r"^\s+of\s+((?:[\w'/-]+\s*){1,4})", re.I)
_PENALTY_EVIDENCE = re.compile(r"\bpenalt\w*|\bsanction\w*|\bdisciplinary\b|\bwarning\b|\bbarred\b|\bforfeit\w*|\bdeduct\w*|\bdismiss\w*|\bsuspend\w*|"
                               r"\bconsequence\w*|\bheld\s+accountable\b|\bfail(?:ing)?\s+grade\b|\b\"?f\"?\s+grade\b|\breferred\s+to\b|\bnot\s+(?:be\s+)?allowed\b", re.I)
_LOCATION_EVIDENCE = re.compile(r"\b(?:blackboard|banner|my\s?uos|lms|portal|course\s+files?|department\w*|registrar\w*|office|college|online|system|website|"
                                r"campus|room|building|library)\b", re.I)

#: Faculty categories and ranks. A clause naming one asks for that category;
#: evidence must name the same category (any alias) next to the subject.
CATEGORY_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("regular faculty", r"\bregular\s+faculty\b|[\"“”]a[\"“”]|\bassistant\s+professors?\b|\bassociate\s+professors?\b|\bfull\s+professors?\b|\branked\s+(?:full[- ]time\s+)?(?:regular\s+)?faculty\b"),
    ("active research faculty", r"\bactive\s+research\s+faculty\b|[\"“”]b[\"“”]"),
    ("research intensive faculty", r"\bresearch[- ]intensive\b|[\"“”]c[\"“”]"),
    ("teaching track faculty", r"\bteaching[- ]track\b|[\"“”]d[\"“”]"),
    ("lecturers", r"\b(?:senior\s+)?lecturers?\b|[\"“”]e[\"“”]"),
    ("instructors", r"\binstructors?\b|[\"“”]f[\"“”]"),
    ("visiting faculty", r"\bvisiting\b"),
    ("adjunct faculty", r"\badjunct\b"),
    ("teaching assistants", r"\bteaching\s+assistants?\b"),
)
_COMPILED_CATEGORIES = tuple((label, re.compile(p, re.I)) for label, p in CATEGORY_GROUPS)


@dataclass(frozen=True)
class Detail:
    """A detail the clause explicitly asks for."""

    family: str
    label: str
    check: Callable[[str], bool]


_UNIT_NOUN = re.compile(
    r"^(?:\(\d+\)\s*)?(?:credit\s+|contact\s+|office\s+|class\s+|working\s+|regular\s+|theoretical\s+|teaching\s+|instructional\s+|consultation\s+|"
    r"academic\s+|calendar\s+|business\s+)?"
    r"(?:hours?|days?|weeks?|months?|semesters?|years?|students?|learners?|courses?|classes|sections?|lectures?|sessions?|times|credits?|points?|"
    r"percent|%|assignments?|exams?|examinations?|quizzes|tutorials?|labs?|laboratories|participants?|members?|minutes?|papers?|copies|meetings?)\b", re.I)
_SUBJECT_AFTER = re.compile(r"\b(?:how\s+many|how\s+much|number\s+of|count\s+of|how\s+long)\s+((?:[\w'/-]+\s*){1,5})", re.I)
_OBLIGATION = re.compile(r"\b(?:must|shall|required|requirements?|obligat\w*|mandatory|expected\s+to|responsible\s+for|responsibility\s+to|"
                         r"needs?\s+to|should|are\s+to\s+be|is\s+to\s+be)\b", re.I)
_LIST_MARKER = re.compile(r"^(?:[•�▪-]\s+|o\s+(?=[\"“A-Z])|\d{1,2}\.\s+|[a-z]\.\s+|[A-Z]\.\s+(?=[A-Z])|\d+(?:\.\d+){1,3}\s+)")


def _fee_ok(unit: str) -> bool:
    """A fee word with a monetary value ("fee of AED 100", "cost is 100
    AED", "free of charge"); "fee applies to examination 101" has none."""
    return bool(_FEE_WORD.search(unit) and _MONEY.search(unit))


def _is_value(unit: str, m: "re.Match[str]") -> bool:
    """Whether the number matched by ``m`` denotes a value: a count noun or
    a percent sign follows it, or it is attached to a limit phrase ("not
    exceed 40"). A year ("in 2024"), a dotted section number ("section
    3.4") or an identifier ("room 204", "examination 101") is not a value."""
    token, before, tail = m.group(0), unit[:m.start()], unit[m.end():].lstrip(" -")
    if _IDENTIFIER_BEFORE.search(before):
        return False
    if _UNIT_AFTER.match(tail) or _UNIT_NOUN.match(tail):
        return True
    if _YEAR_SHAPED.match(token) or "." in token or "," in token:
        return False
    return bool(_LIMIT_ATTACHED.search(before))


def _limit_ok(unit: str, limit: "re.Pattern[str]") -> bool:
    """A limit word with a value-shaped number in the same unit. A table row's
    numeric cells are its values ("Laboratory 20 10")."""
    if not limit.search(unit):
        return False
    if is_table_row(unit):
        return bool(_NUMBER.search(unit))
    return any(_is_value(unit, m) for m in _NUMBER.finditer(unit))


def _frequency_ok(unit: str) -> bool:
    return bool(_FREQUENCY.search(unit) or re.search(r"\b" + _NUMBER_WORD + r"\s+times\b", unit, re.I))


def _quantity_ok(unit: str, subject: Concepts, subject_tokens: FrozenSet[str]) -> bool:
    """A number tied to a count noun ("five office hours", "12 credit hours",
    "20 students"), whose noun phrase names the counted subject of the
    question. Bare numbers (rooms, pages, sections, dates, list markers) and
    counts of something else ("3 credit hours" for an office-hours question)
    do not count."""
    def names_subject(text: str) -> bool:
        if concepts(text).vocab & subject.vocab:
            return True
        return not subject.vocab and bool(set(tokenize(text)) & subject_tokens)

    for m in _NUMBER.finditer(unit):
        tail = unit[m.end():].lstrip()
        noun = _UNIT_NOUN.match(tail)
        if not noun:
            continue
        phrase = tail[:noun.end()]
        if not subject.vocab and not subject_tokens:
            return True
        if names_subject(phrase):
            return True
        # "48 hours of office hours": the counted subject follows the unit noun
        of_phrase = _OF_SUBJECT.match(tail[noun.end():])
        if of_phrase and names_subject(of_phrase.group(1)):
            return True
    return False


def _max_ok(unit: str) -> bool:
    return _limit_ok(unit, _LIMIT_MAX)


def _min_ok(unit: str) -> bool:
    return _limit_ok(unit, _LIMIT_MIN)


#: Evidence for "who manages / supports / is responsible for ...": a
#: responsibility verb and a named party or unit in the same unit.
#: Verb forms only: "Management System" and the noun "support" name nobody.
_RESPONSIBLE_VERB = re.compile(r"\bmanag(?:es|ed|ing)\b|\bsupport(?:s|ed)\b|\bmaintain(?:s|ed)\b|\badminister(?:s|ed)\b|\bresponsible\s+for\b|"
                               r"\bresponsibility\s+(?:for|of|as|to)\b|\bin\s+charge\s+of\b|\bhandl(?:es|ed)\b|\boversee(?:s|n)\b|\boperat(?:es|ed)\b|"
                               r"\bprovid(?:es|ed)\b|\brun\s+by\b|\bcoordinat(?:es|ed)\b|\bassist(?:s|ed)\b", re.I)
_RESPONSIBLE_PARTY = re.compile(r"\b(?:it\s+(?:department|services?|support|unit|team)|help\s*desk|registrar\w*|registration\s+department|deans?|chairs?|"
                                r"chairpersons?|department|college|councils?|committees?|directors?|coordinators?|administrators?|faculty\s+members?|"
                                r"instructors?|lecturers?|units?|offices?(?!\s+hours?)|teams?|cent(?:er|re)s?|division|services)\b", re.I)


def _authority_ok(unit: str) -> bool:
    return bool(_APPROVAL_VERB.search(unit) and _AUTHORITY_NOUN.search(unit))


#: (family, label, clause pattern, evidence check). The clause pattern says
#: when the detail is requested; the check says when a local evidence unit
#: contains it. The quantity check is built per clause (it needs the counted
#: subject); its entry here is a placeholder.
_DETAIL_FAMILIES: Tuple[Tuple[str, str, str, Callable[[str], bool]], ...] = (
    ("fee", "fee or cost", r"\b(?:fees?|costs?|charges?|price|how\s+much\s+(?:do|does|will|must)\b.*\b(?:cost|pay)|payment)\b", _fee_ok),
    ("penalty", "penalty or consequence", r"\b(?:penalt(?:y|ies)|consequences?|sanctions?|disciplinary|what\s+happens\s+(?:if|when)|punish\w*|fined?)\b",
     lambda u: bool(_PENALTY_EVIDENCE.search(u))),
    ("deadline", "deadline or date", r"\b(?:deadlines?|due\s+dates?|by\s+when|when\s+(?:is|are|must|do|does|should|will|can|did)|last\s+day|cut-?off|"
     r"what\s+date|which\s+date|on\s+what\s+day|by\s+what\s+date)\b", lambda u: bool(_TIMING.search(u))),
    ("quantity", "number or amount", r"\b(?:how\s+many|how\s+much|number\s+of|count\s+of|how\s+long)\b", lambda u: False),
    ("frequency", "frequency", r"\b(?:how\s+often|how\s+frequently)\b", _frequency_ok),
    ("minimum", "minimum", r"\b(?:minimum|minimal|at\s+least|fewest|smallest|lowest)\b", _min_ok),
    ("maximum", "maximum or limit", r"\b(?:maximum|maximal|at\s+most|upper\s+limit|limits?|caps?|capped|no\s+more\s+than|largest)\b", _max_ok),
    ("authority", "approving authority", r"\b(?:who\s+(?:approves?|authori[sz]es?|signs?|decides?|grants?|must\s+approve|can\s+approve|gives?\s+approval)|"
     r"approved\s+by\s+whom|whose\s+approval|which\s+(?:office|body|council|committee|person)\s+approves?|approval\s+(?:from|of|by)\s+whom|"
     r"who\s+is\s+the\s+approving)\b", _authority_ok),
    ("responsibility", "responsible party or unit",
     r"\bwho\s+(?:manages|supports|maintains|administers|runs|operates|provides|handles|oversees|looks\s+after|is\s+responsible\s+for|is\s+in\s+charge\s+of)\b",
     lambda u: bool(_RESPONSIBLE_VERB.search(u) and _RESPONSIBLE_PARTY.search(u))),
    ("percentage", "percentage or range", r"\b(?:percent(?:age)?s?|%|what\s+range|score\s+range)\b", lambda u: bool(_PERCENT.search(u))),
    ("part_time", "part-time distinction", r"\bpart[- ]time\b", lambda u: bool(re.search(r"\bpart[- ]time\b", u, re.I))),
    ("full_time", "full-time distinction", r"\bfull[- ]time\b", lambda u: bool(re.search(r"\bfull[- ]time\b", u, re.I))),
    ("location", "location or system", r"\bwhere\s+(?:do|must|should|can|are|is|to)\b|\bin\s+which\s+(?:system|platform|office)\b|\bwhich\s+system\b",
     lambda u: bool(_LOCATION_EVIDENCE.search(u))),
    ("requirement", "an explicit requirement statement about the subject",
     r"\bwhat\s+must\b|\bwhat\s+should\s+(?:\w+\s+){0,2}(?:include|contain|cover|provide)\b|\bwhat\s+is\s+required\b|\bwhat\s+are\s+the\s+requirements\b|"
     r"\bwhat\s+are\s+(?:the\s+|my\s+|its\s+|your\s+|our\s+)?(?:[\w'/-]+\s+){1,4}?requirements?\b|"        # "what are the office hours requirements"
     r"\brequirements?\s+for\b|\bmust\s+(?:contain|include)\b|\bwhat\s+information\s+(?:must|should)\b|"
     r"\bwhat\s+(?:do|does)\s+\w+\s+(?:need|have)\s+to\s+(?:include|contain|provide)\b", lambda u: bool(_OBLIGATION.search(u))),
)
_COMPILED_FAMILIES = tuple((f, label, re.compile(p, re.I), check) for f, label, p, check in _DETAIL_FAMILIES)


def _detail_key(detail: "Detail") -> str:
    return "category:" + detail.label if detail.family == "category" else detail.family


def requested_details(clause: str) -> List[Detail]:
    """The details a clause explicitly asks for, in a fixed order. A quantity
    detail carries the counted subject ("office hours" in "how many office
    hours"); a faculty category named in the clause is a detail of its own
    (the evidence must name the same category)."""
    details: List[Detail] = []
    for family, label, pattern, check in _COMPILED_FAMILIES:
        if not pattern.search(clause):
            continue
        if family == "quantity":
            m = _SUBJECT_AFTER.search(clause)
            subject_text = m.group(1) if m else ""
            subject, tokens = concepts(subject_text), frozenset(tokenize(subject_text))
            details.append(Detail(family, label, (lambda u, s=subject, k=tokens: _quantity_ok(u, s, k))))
        else:
            details.append(Detail(family, label, check))
    for label, pattern in _COMPILED_CATEGORIES:
        if pattern.search(clause):
            details.append(Detail("category", "the requested faculty category (%s)" % label, (lambda p: (lambda u: bool(p.search(u))))(pattern)))
    return details


def _local_units(quote: str) -> List[str]:
    """Evidence units for the completeness check: a list item whole, else a
    sentence. Every requested detail must sit in one such unit together
    with the subject."""
    units: List[str] = []
    for start, end in _split(quote, _ITEM_BREAK):
        piece = quote[start:end]
        if _ITEM_START.match(piece):
            units.append(piece)
        else:
            units.extend(piece[a:b] for a, b in _split(piece, _SENTENCE_BREAK))
    return [u.strip() for u in units if u.strip()] or [quote]


def unit_coverage(details: Sequence[Detail], clause: Concepts, quote: str) -> FrozenSet[str]:
    """The largest set of requested details that ONE local unit of ``quote``
    contains while sharing a concept with the clause. Details found in
    different units are never combined: "part-time" in one sentence and
    "five hours" in another do not answer a part-time quantity."""
    best: FrozenSet[str] = frozenset()
    for unit in _local_units(quote):
        if relevance(clause, unit)[1] < 1:
            continue
        stripped = _LIST_MARKER.sub("", unit)
        got = frozenset(_detail_key(d) for d in details if d.check(stripped))
        if len(got) > len(best):
            best = got
    return best


def detail_covered(detail: Detail, clause: Concepts, quote: str) -> bool:
    """True when some local unit of ``quote`` contains the detail and shares
    a concept with the clause. Use ``unit_coverage`` for completeness: this
    looks at one detail at a time."""
    return _detail_key(detail) in frozenset().union(*[unit_coverage([detail], clause, u) for u in _local_units(quote)]) if _local_units(quote) else False


# ---------------------------------------------------------------------------
# Quote units and quote quality
# ---------------------------------------------------------------------------
_ROW_BREAK = re.compile(r"\s+\|\s+")
#: Items: bullets, "o " items, numbered items ("2. "), lettered items ("b. ",
#: "B. ") after punctuation, and section numbers. A lettered marker opens an
#: item only when a capital follows it.
_ITEM_BREAK = re.compile(
    r"\s+(?=[•�▪]\s)|\s+(?=o\s+[\"“A-Z])|(?<=[.:;!?])\s+(?=\d{1,2}\.\s+[A-Z])|(?<=[.:;!?])\s+(?=[A-Za-z]\.\s+[A-Z])|"
    r"\s+(?=\d+(?:\.\d+){1,3}\s+[A-Z])|\s+(?=Chapter\s+\d+\.)|"
    r"(?<=^Page \d)\s+|(?<=^Page \d\d)\s+|(?<=^Page \d\d\d)\s+")
_ITEM_START = re.compile(r"^(?:[•�▪]\s+|o\s+(?=[\"“A-Z])|\d{1,2}\.\s+(?=[A-Z])|[A-Za-z]\.\s+(?=[A-Z]))")
#: A sentence ends at .!? before a capital, except after a list marker
#: ("2.", "b.", or "B." opening the piece) so the marker stays with its item.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b[a-z]\.)(?<!^[A-Z]\.)\s+(?=[A-Z\"“(\[\d])")
_BULLET = re.compile(r"^(?:[•�▪-]\s+|o\s+(?=[\"“A-Z]))")
_LETTER_MARKER = re.compile(r"^[A-Za-z]\.\s+(?=[A-Z])")
_HEADING = re.compile(r"^(?:chapter\s+\d+\.?|\d+(?:\.\d+){1,3}\.?)\s+[A-Za-z(][^.!?]{0,110}:?$", re.I)
_LIST_TITLE = re.compile(r"^\d{1,2}\.\s+[A-Z][^.!?]{0,60}:?$")
_WEEKDAY_ROW = re.compile(r"^(?:mon|tue|wed|thu|fri|sat|sun)\b", re.I)
#: A grade-table line: a grade letter, then a numeric or textual range ("90-100", "Below 60") and grade points.
_GRADE_ROW = re.compile(r"^[A-F][+-]?\s+(?:\d{1,3}\s*-\s*\d{1,3}|(?:below|above|under|over|less\s+than|more\s+than)\s+\d{1,3})\b", re.I)
_TERMINAL = re.compile(r"[.!?:;]['\")\]]?$")
_ENDS_WITH_CELL = re.compile(r"\d\S*$")
_DIGIT = re.compile(r"\d")
_PRONOUN_START = re.compile(r"^(?:such|they|these|this|those|it|their|its|both)\b", re.I)
_SEMESTER_HEADER = re.compile(r"\b(fall|spring|summer|winter)\s+semester\s+(20\d\d)\s*[/\-–]\s*(20\d\d)", re.I)
_TERM_IN_TEXT = re.compile(r"\b(fall|spring|summer|winter)\b(?:\s+(?:semester\s+)?(20\d\d)\s*[/\-–]\s*(20\d\d))?", re.I)
_YEAR_SPAN = re.compile(r"(20\d\d)\s*[/\-–]\s*(?:20)?(\d\d)\b")


def _split(text: str, breaker: "re.Pattern[str]") -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in breaker.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _units(text: str, chunk_type: Optional[str]) -> List[Tuple[int, int, int]]:
    """Quotable units of a chunk as ``(start, end, item_start)``: the whole
    chunk for a row; the rows of a row window; the sentences of a paragraph,
    each knowing where its bullet or numbered item begins (``item_start`` is
    the unit's own start outside a list item)."""
    if chunk_type == "row":
        return [(0, len(text), 0)]
    if chunk_type == "row_window":
        return [(s, e, s) for s, e in _split(text, _ROW_BREAK)] or [(0, len(text), 0)]
    units: List[Tuple[int, int, int]] = []
    for start, end in _split(text, _ITEM_BREAK):
        piece = text[start:end]
        is_item = bool(_ITEM_START.match(piece))
        for a, b in _split(piece, _SENTENCE_BREAK):
            units.append((start + a, start + b, start if is_item else start + a))
    return units or [(0, len(text), 0)]


def _spans(text: str, chunk_type: Optional[str]) -> List[Tuple[int, int]]:
    """Character spans of the quotable units of a chunk (see ``_units``)."""
    return [(s, e) for s, e, _ in _units(text, chunk_type)]


def is_table_row(text: str) -> bool:
    """A printed line that is table data rather than prose: a weekday-led
    calendar line, a grade-table line, or at least three digit-bearing cells
    with a label, within a table-like length."""
    words = text.split()
    if not 2 <= len(words) <= 24:
        return False
    if _WEEKDAY_ROW.match(text) and any(_DIGIT.search(w) for w in words):
        return True
    if _GRADE_ROW.match(text):
        return True
    digit_cells = sum(1 for w in words if _DIGIT.search(w))
    alpha_words = sum(1 for w in words if len(w) >= 3 and w.isalpha())
    return digit_cells >= 3 and alpha_words >= 1


def quote_problem(quote: str, unit_kind: str, last_unit: bool = False) -> Optional[str]:
    """Why ``quote`` is not usable evidence, or None. ``unit_kind`` is
    "prose" (a unit of a paragraph) or "line" (a printed row)."""
    q = quote.strip()
    words = q.split()
    if not words:
        return "empty"
    core = _BULLET.sub("", q).strip()
    if not core:
        return "empty"
    if _HEADING.match(core) or _LIST_TITLE.match(core):
        return "heading"
    if core.endswith(":"):
        return "lead-in"                                   # "...include the following:" states nothing itself
    if _LETTER_MARKER.sub("", core)[0].islower():           # "b. These ..." is an item, not a fragment
        return "dangling"
    if unit_kind == "line":
        return None if is_table_row(core) else "line fragment"
    if len(words) < 5:
        return "too short"
    if last_unit and not _TERMINAL.search(q) and not _ENDS_WITH_CELL.search(q):
        return "truncated"                                 # a window cut mid-sentence; table cells may end a unit
    return None


def calendar_context(rows_before: Sequence[str], row_text: str) -> Optional[Dict[str, Optional[str]]]:
    """Term and academic year of a calendar line: from the line itself when it
    names them ("Classes begin for Fall 2026-2027"), else from the nearest
    preceding semester header ("Fall Semester 2025/ 2026")."""
    m = _TERM_IN_TEXT.search(row_text)
    if m and m.group(2):
        return {"term": m.group(1).lower(), "academic_year": "%s/%s" % (m.group(2), m.group(3)), "from": "line"}
    for prior in reversed(list(rows_before)):
        h = _SEMESTER_HEADER.search(prior)
        if h:
            ctx: Dict[str, Optional[str]] = {"term": h.group(1).lower(), "academic_year": "%s/%s" % (h.group(2), h.group(3)), "from": "header"}
            if m:                                   # the line names a term without a year
                ctx["term"] = m.group(1).lower()
            return ctx
    if m:
        return {"term": m.group(1).lower(), "academic_year": None, "from": "line"}
    return None


def requested_term(clause: str) -> Tuple[Optional[str], Optional[str]]:
    """(term, academic year) named in a clause, e.g. ("fall", "2026/2027")."""
    m = _TERM_IN_TEXT.search(clause)
    term = m.group(1).lower() if m else None
    y = _YEAR_SPAN.search(clause)
    year = None
    if y:
        first, second = y.group(1), y.group(2)
        year = "%s/%s" % (first, first[:2] + second)
    return term, year


# ---------------------------------------------------------------------------
# Ownership and handoff cues (clause classification)
# ---------------------------------------------------------------------------
#: Positive teaching cues. A clause is Teaching's work only when one fires.
#: A safe superset of the Coordinator's teaching cues plus natural phrasings
#: (consultation time, class section, first week of classes, timetable).
_TEACHING_CUES = re.compile(
    r"\b(?:teach\w*|taught|instructors?|instructional|lectur\w*|courses?|coursework|class(?:es|room|rooms)?|syllab\w*|curricul\w*|"
    r"exam\w*|midterms?|quiz(?:zes)?|assessments?|assignments?|grad(?:e|es|ed|ing)|gradebook|grade\s+cent(?:er|re)|marks?|marking|attendance|absences?|"
    r"office\s+hours?|workloads?|wlam|credit\s+hours?|contact\s+hours?|blackboard|bb\s+ultra|lms|learning\s+management|collaborate|"
    r"safeassign|turnitin|banner|academic\s+calendar|semester\s+(?:begin|start|end)s?|classes\s+(?:begin|start|end)s?|"
    r"add\s*[/-]?\s*drop|withdraw\w*|academic\s+integrity|plagiaris\w*|cheat\w*|hono?u?r\s+code|e-?\s?learning|elearning|"
    r"professional\s+development|faculty\s+development|peer\s+observation|pedagog\w*|mentor\w*|academic\s+freedom|advis(?:e|ing|ors?|ees?)|"
    r"students?|learners?|thesis|dissertation|supervis(?:e|ing|ion)\s+(?:of\s+)?(?:a\s+|the\s+|my\s+)?(?:students?|theses|thesis|dissertations?)|"
    r"artificial\s+intelligence|ai|llms?|generative\s+(?:ai|tools?)|summer\s+(?:semester|courses?|teaching)|"
    r"class\s+sizes?|course\s+files?|learning\s+outcomes?|rubrics?|laborator(?:y|ies)|labs?|tutorials?|seminars?|lessons?|training|workshops?|"
    r"teaching\s+(?:award|excellence|workshop|schedule|assistants?)|make[- ]?up|incomplete|c?gpa|timetables?|academic\s+year|"
    r"consultation\s+(?:hours?|times?)|student\s+consultations?|consultations?|class\s+sections?|sections?\s+(?:is\s+|are\s+|be\s+|get\s+|was\s+)?(?:cancel\w*|closed|open\w*|size|enrol\w*|merged)|"
    r"first\s+(?:week|day)\s+of\s+(?:classes|the\s+semester|the\s+term|term))\b",
    re.I,
)

#: Other-domain cues, phrase-level. Bare "contract", "appointment",
#: "benefits", "funding", "extension" and "who approves" are NOT cues.
_OTHER_CUES: Tuple[Tuple[SpecialistId, "re.Pattern[str]"], ...] = (
    (SpecialistId.INSTITUTIONAL, re.compile(
        r"\b(?:who(?:m)?\s+(?:do|should|can|could|must|would)\s+i\s+(?:contact|ask|call|e-?mail|talk\s+to|speak\s+to|reach|see|go\s+to|refer\s+to|approach)|"
        r"who\s+(?:is|are)\s+the\s+(?:contact|person|people)\s+(?:for|to)|"
        r"contact\s+(?:person|details|information|numbers?)|phone|telephone|fax|e-?\s?mail\s+address|extension\s+number|"
        # "where is / where can I find <office, unit, named service>"; "office hours" is a teaching phrase, not a place
        r"where\s+(?:is|are|can\s+i\s+find|do\s+i\s+find)\s+(?:the\s+)?[\w' ]{0,30}?(?:office(?!\s+hours?)|department|unit|cent(?:er|re)|building|desk|help\s*desk|library|clinic|"
        r"registrar\w*|registration\s+(?:office|department|desk)|admissions?\s+(?:office|department)|finance\s+(?:office|department)|hr|human\s+resources|"
        r"it\s+(?:services?|support)|reception|security|bookstore|cafeteria|parking)|"
        # "who manages / supports / maintains ... <support, help desk, service, system, office, unit>"
        r"who\s+(?:manages|supports|maintains|administers|runs|provides)\s+[\w' ]{0,40}?(?:support|help\s*desk|services?|systems?|portal|blackboard|banner|my\s?uos|lms|"
        r"office(?!\s+hours?)|unit|department|cent(?:er|re)|desk)|"
        r"which\s+(?:office|unit|department|committee|council|body)\s+(?:handles?|is\s+responsible|deals|manages|approves|do\s+i)|"
        r"organi[sz]ational\s+(?:chart|structure)|governance|chain\s+of\s+command|board\s+of\s+trustees|university\s+council|"
        r"help\s*desk|it\s+support|service\s+desk)\b", re.I)),
    (SpecialistId.RESEARCH, re.compile(
        r"\b(?:research\w*|grants?|seed\s+fund\w*|external\s+fund\w*|principal\s+investigator|"
        r"ethics\s+(?:committee|approval|application|review|clearance)|irb|publications?|publish\w*|journals?|"
        r"patents?|intellectual\s+property|scopus|citations?|h-?\s?index)\b", re.I)),
    (SpecialistId.FACULTY_SERVICES, re.compile(
        r"\b(?:(?:annual|sick|maternity|paternity|emergency|unpaid|study|compassionate|casual)\s+leaves?|leave\s+(?:of\s+absence|entitlement|balance|request|application|days)|"
        r"days?\s+of\s+leave|vacation|sabbatical|employment\s+contracts?|contract\s+(?:renewal|termination|period|expiry|duration)|probation\w*|"
        r"payroll|salar(?:y|ies)|pay\s?slips?|end\s+of\s+service|gratuity|pension|employment\s+benefits|(?:housing|transport|education|travel)\s+allowances?|allowances|"
        r"housing|accommodation|(?:health|medical)\s+insurance|visa|residence\s+permit|passport|emirates\s+id|resign\w*|termination|"
        r"promotion|tenure|recruitment|hiring|human\s+resources|hr)\b", re.I)),
)

#: A clause that asks for a system procedure rather than a policy or fact.
_PROCEDURE_CUES = re.compile(
    r"\b(?:step[- ]by[- ]step|steps?|clicks?|clicking|buttons?|screens?|menus?|tabs?|icons?|navigat\w*|"
    r"dropdown|log\s?in|login|where\s+(?:do|can|should|must)\s+i\s+(?:click|find|go|press|select)|"
    r"how\s+(?:do|can|should|would)\s+i\s+(?:create|upload|enable|add|enter|submit|post|set\s+up|configure|record|"
    r"mark|open|access|log|attach|publish|import|export|download|edit|delete))\b",
    re.I,
)
_SYSTEM_CUES = re.compile(r"\b(?:blackboard|bb\s+ultra|banner|my\s?uos|lms|learning\s+management\s+system|self[- ]service)\b", re.I)


def classify_clause(clause: str) -> Tuple[bool, Optional[SpecialistId]]:
    """(carries a positive teaching cue, strongest other-domain specialist or None)."""
    clause = cue_text(clause)                                   # "office-hours" reads as "office hours"
    teaching = bool(_TEACHING_CUES.search(clause))
    hits = [(len(pattern.findall(clause)), -order, sid) for order, (sid, pattern) in enumerate(_OTHER_CUES)]
    hits = [h for h in hits if h[0]]
    other = max(hits)[2] if hits else None
    return teaching, other


def system_named(text: str) -> Optional[str]:
    """The canonical system id named in ``text`` (blackboard, banner, myuos), or None."""
    named = _SYSTEM_CUES.search(text)
    if not named:
        return None
    system = re.sub(r"[\s-]", "", named.group(0).lower())
    if system in ("bbultra", "lms", "learningmanagementsystem"):
        return "blackboard"
    if system == "selfservice":
        return "banner"
    return system


def is_procedure_request(clause: str, system_hint: Optional[str]) -> Tuple[bool, Optional[str]]:
    """(asks for a system procedure, system named). A procedure cue without a
    system named in the clause still counts when the task carries a system hint."""
    system = system_named(clause)
    if system is None and system_hint in SYSTEMS:
        system = system_hint
    procedural = bool(_PROCEDURE_CUES.search(clause)) and system is not None
    return procedural, system


# ---------------------------------------------------------------------------
# Scope and the section guard
# ---------------------------------------------------------------------------
def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def compile_scope(section_map: SectionMap, source_id: str = APPROVED_SOURCE_ID,
                  owned: Sequence[Tuple[int, str, str]] = OWNED_SECTIONS) -> RetrievalScope:
    """The retrieval scope for the owned sections, as page ranges read from the
    section map. Raises ``ValueError`` when the map is not the approved
    source's or lacks an owned section: the scope is never silently smaller."""
    if section_map.source_id != source_id:
        raise ValueError("section map belongs to %s, not %s" % (section_map.source_id, source_id))
    ranges: List[Tuple[int, int]] = []
    absent: List[str] = []
    for chapter, number, _label in owned:
        records = [r for r in section_map.sections() if r.chapter == chapter and r.section_no == number]
        if not records:
            absent.append("chapter %d section %s" % (chapter, number))
        ranges.extend((r.start_page, r.end_page) for r in records)
    if absent:
        raise ValueError("owned sections absent from the section map: " + ", ".join(absent))
    return RetrievalScope(source_ids=[source_id], page_ranges=_merge_ranges(ranges))


def heading_marker(record: SectionRecord) -> "re.Pattern[str]":
    """Regex for the printed heading of a section: its number followed by the
    first two title words, e.g. "3.5 Responsibility to"."""
    words = [re.sub(r"[^\w()/-]", "", w) for w in re.split(r"\s+", record.section_title or "") if w][:2]
    parts = [re.escape(record.section_no or "")] + [re.escape(w) for w in words if w]
    return re.compile(r"(?<![\d.])" + r"\W+".join(parts) + r"(?![\w])", re.I)


@dataclass
class _PageSegments:
    """Sections printed on one shared page, in document order, with the row
    index at which each begins (-1 for the section already in progress)."""

    page: int
    segments: List[Tuple[Tuple[int, str], int, bool]]          # ((chapter, number), first row id, owned)
    markers: List[Tuple["re.Pattern[str]", int]]               # heading regex -> segment index (starting sections)
    para_spans: Dict[int, List[Tuple[int, int, bool]]]         # chunk id -> [(start, end, owned)]

    def row_owned(self, row_id: int) -> bool:
        owned = self.segments[0][2]
        for _key, first_row, is_owned in self.segments:
            if first_row <= row_id:
                owned = is_owned
        return owned


class SectionGuard:
    """Chunk-level section eligibility on pages that carry a non-owned section.

    Built once from the chunks, their metadata and the section map. On a
    shared page the printed section headings are located: as row chunks (row
    ids) and inside paragraph chunks (character offsets). A row belongs to the
    last section whose heading precedes it; a row window is checked row by
    row; a paragraph window is split at the headings it contains. Inside a
    window, text after a heading belongs to that heading's section until the
    next heading, and text before the first heading belongs to the section
    that precedes that heading in the page's section order, never to a state
    remembered from an overlapping window. A window without any heading lies
    inside the section that the previous window ended in (the windows overlap
    by more than a heading marker, so a heading is never missed). Text of a
    non-owned section is not Teaching evidence. A shared page whose headings
    cannot be located is rejected whole: false ``not_found`` is preferred to
    a supported claim from an excluded section.
    """

    def __init__(self, chunks: Sequence[str], metadata: Sequence[Dict[str, Any]], section_map: SectionMap,
                 owned: Sequence[Tuple[int, str, str]], scope: RetrievalScope) -> None:
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata differ in length (%d vs %d)" % (len(chunks), len(metadata)))
        for i, meta in enumerate(metadata):                         # paragraph spans are keyed by list index
            cid = meta.get("chunk_id")
            if cid is not None and cid != i:
                raise ValueError("chunk metadata is out of order: metadata[%d]['chunk_id'] == %r" % (i, cid))
        self.owned = {(c, n) for c, n, _ in owned}
        self.shared_pages: Dict[int, _PageSegments] = {}
        self.unresolved_pages: List[int] = []
        pages = sorted({p for first, last in (scope.page_ranges or ()) for p in range(first, last + 1)})
        sections = section_map.sections()
        by_page: Dict[int, List[int]] = {}
        for i, meta in enumerate(metadata):
            page = meta.get("page")
            if isinstance(page, int) and page in pages:
                by_page.setdefault(page, []).append(i)
        for page in pages:
            on_page = [r for r in sections if r.start_page <= page <= r.end_page]
            if not on_page or all((r.chapter, r.section_no) in self.owned for r in on_page):
                continue                                            # nothing non-owned printed here
            if not by_page.get(page):
                continue                                            # no chunk carries this page: nothing to attribute
            built = self._build(page, on_page, by_page[page], chunks, metadata)
            if built is None:
                self.unresolved_pages.append(page)
            else:
                self.shared_pages[page] = built

    def _build(self, page: int, on_page: List[SectionRecord], ids: List[int], chunks: Sequence[str],
               metadata: Sequence[Dict[str, Any]]) -> Optional[_PageSegments]:
        segments: List[Tuple[Tuple[int, str], int, bool]] = []
        markers: List[Tuple["re.Pattern[str]", int]] = []
        rows = [(metadata[i].get("row_id"), chunks[i]) for i in ids if metadata[i].get("chunk_type") == "row"]
        for record in on_page:
            key = (record.chapter, record.section_no)
            owned = key in self.owned
            if record.start_page < page:
                segments.append((key, -1, owned))
                continue
            marker = heading_marker(record)
            hits = [rid for rid, text in rows if isinstance(rid, int) and marker.match(text)]
            if not hits:
                return None                                         # heading not printed as a row: unresolved
            segments.append((key, min(hits), owned))
            markers.append((marker, len(segments) - 1))
        starts = [first for _k, first, _o in segments]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            return None                                             # headings out of document order
        para_spans: Dict[int, List[Tuple[int, int, bool]]] = {}
        paragraphs = sorted(
            (i for i in ids if metadata[i].get("chunk_type") == "paragraph"),
            key=lambda i: (metadata[i]["chunk_id_on_page"] if isinstance(metadata[i].get("chunk_id_on_page"), int) else i, i))
        current = 0                                                 # section a window without headings lies in
        for i in paragraphs:                                        # paragraph windows are emitted in document order
            text = chunks[i]
            cuts: List[Tuple[int, int]] = []
            for marker, seg_index in markers:
                m = marker.search(text)
                if m:
                    cuts.append((m.start(), seg_index))
            cuts.sort()
            spans: List[Tuple[int, int, bool]] = []
            if not cuts:
                spans.append((0, len(text), segments[current][2]))
            else:
                first_pos, first_seg = cuts[0]
                if first_pos > 0:                                   # text before the first heading: the preceding section
                    spans.append((0, first_pos, segments[max(first_seg - 1, 0)][2]))
                for k, (pos, seg_index) in enumerate(cuts):
                    end = cuts[k + 1][0] if k + 1 < len(cuts) else len(text)
                    if end > pos:
                        spans.append((pos, end, segments[seg_index][2]))
                current = max(current, cuts[-1][1])
            para_spans[i] = spans
        return _PageSegments(page=page, segments=segments, markers=markers, para_spans=para_spans)

    def is_shared(self, meta: Dict[str, Any]) -> bool:
        return meta.get("page") in self.shared_pages or meta.get("page") in self.unresolved_pages

    def owned_spans(self, chunk_id: Any, chunk: str, meta: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        """Character spans of ``chunk`` that are Teaching's, or None when the
        whole chunk is (page without a non-owned section). An empty list
        means nothing in the chunk may be quoted."""
        page = meta.get("page")
        if page in self.unresolved_pages:
            return []
        segs = self.shared_pages.get(page)
        if segs is None:
            return None
        kind = meta.get("chunk_type")
        if kind == "row":
            rid = meta.get("row_id")
            return [(0, len(chunk))] if isinstance(rid, int) and segs.row_owned(rid) else []
        if kind == "row_window":
            first, last = meta.get("row_start"), meta.get("row_end")
            units = _spans(chunk, "row_window")
            if not isinstance(first, int) or not isinstance(last, int) or len(units) != last - first + 1:
                return []                                           # misaligned window: reject
            return [(s, e) for (s, e), rid in zip(units, range(first, last + 1)) if segs.row_owned(rid)]
        if kind == "paragraph":
            spans = segs.para_spans.get(chunk_id)
            if spans is None:
                return []
            return [(s, e) for s, e, owned in spans if owned]
        return []


# ---------------------------------------------------------------------------
# Quote selection
# ---------------------------------------------------------------------------
def select_quote(text: str, chunk_type: Optional[str], clause: Concepts,
                 allowed: Optional[Sequence[Tuple[int, int]]] = None,
                 details: Sequence["Detail"] = ()) -> Optional[Tuple[str, int, int]]:
    """The best quotable unit of ``text`` for the clause as ``(quote, shared
    anchors, shared vocabulary)``, or None when no unit is both relevant and
    quotable. ``allowed`` restricts units to character spans attributed to
    Teaching (None = whole chunk). A sentence inside a list item is quoted
    with the item's opening; a sentence that opens with a context-dependent
    word (Such, They, These, This, Those, It) is quoted with the sentence
    before it when that sentence is allowed and usable, and is rejected
    otherwise; a prose unit is extended by the next unit when that one is
    allowed, quotable, shares a concept and does not open a new item. The
    quote is an exact substring of ``text``. Among relevant units, one that
    contains more of the requested ``details`` wins, then relevance, then
    document order."""
    units = _units(text, chunk_type)
    kind = "line" if chunk_type in ("row", "row_window") else "prose"

    def permitted(span: Tuple[int, int]) -> bool:
        return allowed is None or any(s <= span[0] and span[1] <= e for s, e in allowed)

    def bad(index: int) -> Optional[str]:
        s, e, _ = units[index]
        last = kind == "prose" and chunk_type == "paragraph" and index == len(units) - 1
        return quote_problem(text[s:e].strip(), kind, last_unit=last)

    def is_item_head(index: int) -> bool:
        s, e, item_start = units[index]
        return item_start == s and bool(_ITEM_START.match(text[s:e]))

    best: Optional[Tuple[Tuple[int, int, int, int], str]] = None
    for index, (start, end, item_start) in enumerate(units):
        if not permitted((start, end)) or bad(index) or relevance(clause, text[start:end])[1] < 1:
            continue                                                # a unit must share a concept itself
        core = _LIST_MARKER.sub("", text[start:end].strip())
        if kind == "prose" and _PRONOUN_START.match(core):
            # The sentence depends on its antecedent: quote it with the unit
            # before it, or not at all. A list item's antecedent is the list's
            # lead-in or a plain sentence, never a sibling item; a sentence
            # inside an item takes the item's own earlier sentence.
            if index == 0:
                continue
            p_start, p_end, p_item = units[index - 1]
            prev = text[p_start:p_end].strip()
            prev_is_item = p_item == p_start and bool(_ITEM_START.match(prev))
            if item_start == start:
                usable = p_item == p_start and not prev_is_item
            else:
                usable = p_item == item_start
            prev_core = _LIST_MARKER.sub("", prev)
            if not (usable and permitted((p_start, end)) and prev_core and not prev_core[0].islower()
                    and not _HEADING.match(prev) and len(text[p_start:end].split()) <= MAX_QUOTE_WORDS):
                continue                                            # a heading is never an antecedent
            start = p_start
        # A sentence inside a bullet or numbered item is quoted with the item's
        # opening, so "The load for this category is 12 hours" keeps its category.
        if item_start < start and permitted((item_start, end)) and len(text[item_start:end].split()) <= MAX_QUOTE_WORDS:
            start = item_start
        # A prose unit is extended by the next unit when that one is allowed,
        # quotable and shares a concept. Printed lines are never joined.
        if kind == "prose" and index + 1 < len(units) and not is_item_head(index + 1) and permitted(units[index + 1][:2]) and not bad(index + 1):
            nxt_end = units[index + 1][1]
            if relevance(clause, text[units[index + 1][0]:nxt_end])[1] >= 1 and len(text[start:nxt_end].split()) <= MAX_QUOTE_WORDS:
                end = nxt_end
        quote = text[start:end].strip()
        if quote_problem(quote, kind, last_unit=(kind == "prose" and chunk_type == "paragraph" and end == units[-1][1])):
            continue
        anchors, vocab = relevance(clause, quote)
        if anchors >= 1 or vocab >= 2:
            covered = len(unit_coverage(details, clause, quote))
            score = (covered, anchors, vocab, -index)
            if best is None or score > best[0]:
                best = (score, quote)
    if best is None:
        return None
    (_, anchors, vocab, _), quote = best
    return quote, anchors, vocab


# ---------------------------------------------------------------------------
# Resources (dependency injection; nothing is loaded by importing this module)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TeachingResources:
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
    def from_knowledge_base(cls, kb) -> "TeachingResources":
        source = getattr(kb, "source", None)
        if source is None or not source.registered or source.source_id != APPROVED_SOURCE_ID:
            raise ValueError("the Teaching specialist requires the registered handbook %r" % APPROVED_SOURCE_ID)
        applied = ((getattr(kb, "stats", None) or {}).get("source") or {}).get("section_records", 0)
        if not applied:
            raise ValueError("the section map was not applied to the loaded document; teaching scope cannot be compiled")
        embedder, reranker = kb.embedder, kb.reranker

        def embed(text: str) -> np.ndarray:
            return embedder.encode([normalize_text(text)], convert_to_numpy=True,
                                   normalize_embeddings=True).astype(np.float32)

        return cls(embed=embed, rerank=reranker.predict, index=kb.index, chunks=list(kb.chunks),
                   metadata=list(kb.metadata), source=source, section_map=section_map_for(source))


# ---------------------------------------------------------------------------
# The specialist
# ---------------------------------------------------------------------------
_PATH_SHAPED = re.compile(
    r"[A-Za-z]:[\\/][^\s'\"<>|]+"                                       # Windows drive paths
    r"|(?<![\w:./])/(?:home|users|tmp|var|etc|opt|mnt|root|srv|usr|c|d|data|app|workspace)/[^\s'\"<>|]*"   # Unix homes and system trees
    r"|(?<![\w:./])/[\w.-]+(?:/[\w.-]+)+",                                # any other absolute path
    re.I)


def _safe_error(exc: BaseException) -> str:
    """The exception type and a short message with key-shaped secrets and
    filesystem paths removed."""
    text = _KEY_SHAPED.sub("gsk_***", str(exc))
    text = _PATH_SHAPED.sub("<path>", text)
    text = re.sub(r"\s+", " ", text)[:160]
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__


def _quote_key(quote: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", quote.lower()).strip()


@dataclass
class _Candidate:
    rank: int
    item: Dict[str, Any]
    quote: str
    anchors: int
    vocab: int
    context: Optional[Dict[str, Any]]
    covered: FrozenSet[str]


class TeachingLearningSpecialist(Specialist):
    """Teaching & Learning Specialist, deterministic version 1."""

    specialist_id = SpecialistId.TEACHING

    def __init__(self, resources: TeachingResources, *,
                 owned_sections: Sequence[Tuple[int, str, str]] = OWNED_SECTIONS) -> None:
        if resources.source.source_id != APPROVED_SOURCE_ID or not resources.source.registered:
            raise ValueError("the Teaching specialist may only use the registered handbook %r" % APPROVED_SOURCE_ID)
        self.resources = resources
        self.owned_sections = tuple(owned_sections)
        self.scope = compile_scope(resources.section_map, APPROVED_SOURCE_ID, self.owned_sections)
        self.guard = SectionGuard(resources.chunks, resources.metadata, resources.section_map, self.owned_sections, self.scope)
        calendar = [r for r in resources.section_map.sections() if (r.chapter, r.section_no) == (1, "1.14")]
        self.calendar_pages = {p for r in calendar for p in range(r.start_page, r.end_page + 1)}
        self._rows_by_page: Dict[int, List[Tuple[int, str]]] = {}
        for i, meta in enumerate(resources.metadata):
            if meta.get("chunk_type") == "row" and meta.get("page") in self.calendar_pages and isinstance(meta.get("row_id"), int):
                self._rows_by_page.setdefault(meta["page"], []).append((meta["row_id"], resources.chunks[i]))
        for rows in self._rows_by_page.values():
            rows.sort()
        self.handbook_year = self._version_year(resources.source.version)

    @staticmethod
    def _version_year(version: str) -> Optional[str]:
        m = _YEAR_SPAN.search(version or "")
        return "%s/%s" % (m.group(1), m.group(1)[:2] + m.group(2)) if m else None

    @classmethod
    def from_knowledge_base(cls, kb) -> "TeachingLearningSpecialist":
        return cls(TeachingResources.from_knowledge_base(kb))

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
            return "task source_scope names sources the Teaching specialist is not approved to use: %s" % ", ".join(unapproved)
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
        production order. Every kept item, not only the best, must reach
        ``MIN_RERANK_SCORE``. The production ``FINAL_K`` cut is not applied:
        it limits answer context, not evidence eligibility, and most chunks
        are printed lines that the quote rules never accept; the per-clause
        finding cap bounds the output instead."""
        embedding = np.asarray(self.resources.embed(clause), dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        candidates = gather_candidates(clause, route, embedding, self.resources.index,
                                       self.resources.chunks, self.resources.metadata, scope=self.scope)
        if not candidates:
            return []
        scores = self.resources.rerank([[clause, c["chunk"]] for c in candidates])
        items: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            items.append(item)
        items.sort(key=lambda x: (x["rerank_score"], x.get("routing_boost", 0.0),
                                  x.get("lexical_score", 0.0), x.get("dense_score", 0.0)), reverse=True)
        items = deduplicate_by_text(items)
        return [item for item in items if item["rerank_score"] >= MIN_RERANK_SCORE]

    def _shape_problem(self, item: Dict[str, Any]) -> Optional[str]:
        """Clause-independent eligibility: printed prose lines are not
        evidence (the paragraph chunk carries the same text as sentences) and
        the paragraph chunk of a calendar page is not either (its tables are
        quoted from rows, with term context)."""
        meta, chunk = item["meta"], item["chunk"]
        kind = meta.get("chunk_type")
        if kind in ("row", "row_window") and not any(is_table_row(chunk[s:e].strip()) for s, e in _spans(chunk, kind)):
            return "printed prose line, not an evidence unit"
        if kind == "paragraph" and meta.get("page") in self.calendar_pages:
            return "calendar page: tables are quoted from rows"
        return None

    def _calendar_check(self, item: Dict[str, Any], quote: str, clause: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Term and academic-year safety for calendar lines. A line from
        another academic year, or another term than the one asked for, is not
        evidence. Returns (eligible, context)."""
        meta = item["meta"]
        if meta.get("page") not in self.calendar_pages or meta.get("chunk_type") not in ("row", "row_window"):
            return True, None
        rows = self._rows_by_page.get(meta["page"], [])
        if meta.get("chunk_type") == "row":
            first = meta.get("row_id", 0)
        else:
            head = quote.split(" | ")[0].strip()
            units = _spans(item["chunk"], "row_window")
            offset = next((k for k, (s, e) in enumerate(units) if item["chunk"][s:e].strip() == head), 0)
            first = (meta.get("row_start") or 0) + offset
        before = [text for rid, text in rows if rid < first]
        if not any(_SEMESTER_HEADER.search(t) for t in before):    # look back to earlier calendar pages
            for page in sorted((p for p in self._rows_by_page if p < meta["page"]), reverse=True):
                before = [text for _rid, text in self._rows_by_page[page]] + before
                if any(_SEMESTER_HEADER.search(t) for t in before):
                    break
        context = calendar_context(before, quote)
        if context is None:
            return False, None
        want_term, want_year = requested_term(clause)
        expected_year = want_year or self.handbook_year
        year_ok = context["academic_year"] == expected_year if expected_year else True
        term_ok = context["term"] == want_term if want_term else True
        return year_ok and term_ok, context

    @staticmethod
    def _about_request(quote: str, anchors: int, system: Optional[str]) -> bool:
        """For a procedural request: the evidence must name the system asked
        about or share an anchor concept with the request. Tangential
        teaching sentences do not make a procedure question ``partial``."""
        if anchors >= 1:
            return True
        named = system_named(quote)
        return named is not None and (system is None or named == system)

    def _finding(self, task: SpecialistTask, number: int, cand: _Candidate, clause_index: int, route: str,
                 details: Sequence[Detail]) -> Finding:
        item, quote = cand.item, cand.quote
        meta = item["meta"]
        chunk = item["chunk"]
        if quote not in chunk:                                   # cannot happen by construction; kept as a guard
            raise ValueError("quote is not a substring of the retrieved chunk")
        if meta.get("source_id") != APPROVED_SOURCE_ID or not self.scope.allows(meta):
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
            "boundary_guard": "attributed to an owned section" if shared else "page carries owned sections only",
            "clauses": [clause_index],
            "route": route,
            "rerank_score": item["rerank_score"],
            "score_semantics": "cross-encoder rerank score; heuristic, uncalibrated, not a probability",
            "relevance": {"anchors": cand.anchors, "vocabulary": cand.vocab},
            "details_covered": sorted(d.label for d in details if d.family in cand.covered or ("category:" + d.label) in cand.covered),
            "extractive": True,
        }
        if cand.context:
            metadata["calendar_context"] = cand.context
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
        metadata["boundary"] = {"shared_pages": sorted(self.guard.shared_pages), "unresolved_pages": list(self.guard.unresolved_pages)}
        return SpecialistFindings(
            specialist_id=self.specialist_id, task_id=task.task_id, status=status, findings=list(findings),
            summary=summary, missing=list(missing), handoff_requests=list(handoffs), confidence=confidence,
            limitations=list(limitations), llm_used=False, ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def _handoff(self, task: SpecialistTask, target: SpecialistId, clause: str, index: int, kind: str) -> HandoffRequest:
        return HandoffRequest(
            from_specialist=self.specialist_id, requested_specialist=target,
            reason="%s clause %d belongs to %s: %s" % (kind, index, target.value, clause[:120]),
            task=SpecialistTask(
                task_id="%s-handoff-%d" % (task.task_id, index), question=task.question, specialist_id=target,
                domain=target.value, intent=task.intent, level=task.level, system=task.system,
                context={"full_question": task.question, "focus_clauses": [clause]},
                requested_by=self.specialist_id.value,
                metadata={"origin_task_id": task.task_id, "clause_index": index, "kind": kind},
            ),
            metadata={"clause_index": index, "kind": kind},
        )

    # ---- one clause -----------------------------------------------------------
    def _candidates(self, clause: str, route: str, clause_concepts: Concepts, details: Sequence[Detail],
                    procedural: bool, system: Optional[str], rejected: List[Dict[str, Any]]) -> List[_Candidate]:
        """Relevant, quotable, eligible candidates for one clause, ordered so
        that evidence covering more of the requested details comes first and
        the reranker order breaks ties."""
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
            ok, context = self._calendar_check(item, quote, clause)
            if not ok:
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": "calendar term or year mismatch"})
                continue
            if procedural and not self._about_request(quote, anchors, system):
                rejected.append({"chunk_id": meta.get("chunk_id"), "page": meta.get("page"), "why": "tangential to the system procedure asked about"})
                continue
            covered = unit_coverage(details, clause_concepts, quote)
            found.append(_Candidate(rank, item, quote, anchors, vocab, context, covered))
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
                teaching, other = classify_clause(clause)
                if not teaching:
                    if other is not None:
                        handoffs.append(self._handoff(task, other, clause, index, "misrouted"))
                        reports.append({"index": index, "clause": clause, "outcome": "handed_off", "target": other.value})
                    else:
                        reports.append({"index": index, "clause": clause, "outcome": "unowned", "target": None})
                        missing.append("clause %d carries no teaching cue and was not searched: %s" % (index, clause[:120]))
                    continue
                owned_clauses += 1
                if other is not None:
                    handoffs.append(self._handoff(task, other, clause, index, "shared"))
                procedural, system = is_procedure_request(clause, task.system)
                if system and system not in systems_seen:
                    systems_seen.append(system)
                clause_concepts = concepts(clause)
                details = requested_details(clause)
                rejected: List[Dict[str, Any]] = []
                clause_findings: List[str] = []
                best_covered: FrozenSet[str] = frozenset()      # the most complete single evidence unit
                seen_anywhere: set = set()                       # details present somewhere, for the message
                for cand in self._candidates(clause, route, clause_concepts, details, procedural, system, rejected):
                    if len(clause_findings) >= MAX_FINDINGS_PER_CLAUSE:
                        break
                    key = _quote_key(cand.quote)
                    page = cand.item["meta"].get("page")
                    duplicate = next((f for f in findings if f.page == page and
                                      (key in _quote_key(f.evidence_quote) or _quote_key(f.evidence_quote) in key)), None)
                    if duplicate is not None:                        # the same evidence never takes two slots
                        if index not in duplicate.metadata["clauses"]:
                            duplicate.metadata["clauses"].append(index)
                        if duplicate.finding_id not in clause_findings:
                            clause_findings.append(duplicate.finding_id)
                        if len(cand.covered) > len(best_covered):
                            best_covered = cand.covered
                        seen_anywhere |= set(cand.covered)
                        continue
                    finding = self._finding(task, len(findings) + 1, cand, index, route, details)
                    findings.append(finding)
                    clause_findings.append(finding.finding_id)
                    if len(cand.covered) > len(best_covered):
                        best_covered = cand.covered
                    seen_anywhere |= set(cand.covered)
                uncovered = [d for d in details if _detail_key(d) not in best_covered]
                if not clause_findings:
                    outcome = "not_found"
                    if procedural:
                        missing.append("step-by-step procedure for %s requested by clause %d is not in the indexed sources: %s"
                                       % (system, index, clause[:120]))
                    else:
                        missing.append("no evidence in the Teaching scope for clause %d: %s" % (index, clause[:120]))
                elif procedural or uncovered:
                    outcome = "partial"
                    if procedural:
                        missing.append("step-by-step procedure for %s requested by clause %d is not in the indexed sources: %s"
                                       % (system, index, clause[:120]))
                    for d in uncovered:
                        if _detail_key(d) in seen_anywhere:
                            missing.append("%s is not stated together with the other requested details in one place for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        else:
                            missing.append("%s not found in the evidence for clause %d: %s" % (d.label, index, clause[:120]))
                else:
                    outcome = "supported"
                reports.append({"index": index, "clause": clause, "outcome": outcome, "procedural": procedural,
                                "system": system, "findings": clause_findings, "rejected": rejected,
                                "requested_details": [d.label for d in details], "uncovered_details": [d.label for d in uncovered],
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
            limitations.append("%d clause(s) carry no teaching cue and were not searched." % len(unowned))

        metadata = {"clauses": reports, "unprocessed_clauses": unprocessed, "route": route,
                    "owned_clauses": owned_clauses, "gate": MIN_RERANK_SCORE}
        if owned_clauses == 0:
            return self._result(task, FindingStatus.OUT_OF_SCOPE, handoffs=handoffs, limitations=limitations, missing=missing,
                                summary="no assigned clause belongs to Teaching & Learning", metadata=metadata, started=started)
        if findings and not missing:
            status, summary = FindingStatus.SUPPORTED, "%d clause(s) supported by %d finding(s)" % (owned_clauses, len(findings))
        elif findings:
            status, summary = FindingStatus.PARTIAL, "%d finding(s); %d gap(s)" % (len(findings), len(missing))
        else:
            status, summary = FindingStatus.NOT_FOUND, "no evidence in the Teaching scope for the assigned clause(s)"
        return self._result(task, status, findings=findings, missing=missing, handoffs=handoffs,
                            limitations=limitations, summary=summary, metadata=metadata, started=started)


__all__ = [
    "ANCHOR_CONCEPTS", "APPROVED_SOURCE_ID", "CATEGORY_GROUPS", "Concepts", "Detail", "EXCLUDED_SECTIONS", "GENERIC_TERMS",
    "MAX_CANDIDATES_PER_CLAUSE", "MAX_CLAUSES", "MAX_FINDINGS_PER_CLAUSE", "MAX_QUOTE_WORDS", "OWNED_SECTIONS", "SYSTEMS",
    "SectionGuard", "TeachingLearningSpecialist", "TeachingResources", "VOCABULARY_CONCEPTS", "calendar_context",
    "classify_clause", "compile_scope", "concepts", "detail_covered", "heading_marker", "is_procedure_request", "is_relevant",
    "is_table_row", "quote_problem", "relevance", "requested_details", "requested_term", "select_quote", "system_named", "unit_coverage",
]
