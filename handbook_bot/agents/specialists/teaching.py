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
from . import details as _details
from . import evidence as _evidence
from . import grounding as _grounding
from . import scoping as _scoping
from .base import Specialist, _KEY_SHAPED, _PATH_SHAPED, redact_error
from .retrieval_intent import resolve_clause_retrieval_intent
from .details import (  # noqa: F401  (domain-neutral primitives; the names stay available here)
    Detail, _APPROVAL_VERB, _AUTHORITY_NOUN, _FEE_WORD, _FREQUENCY, _IDENTIFIER_BEFORE, _LIMIT_ATTACHED, _LIMIT_MAX, _LIMIT_MIN, _MONEY,
    _NUMBER, _NUMBER_WORD, _OBLIGATION, _OF_SUBJECT, _PENALTY_EVIDENCE, _PERCENT, _RESPONSIBLE_VERB, _SUBJECT_AFTER, _TIMING, _UNIT_AFTER,
    _UNIT_NOUN, _YEAR_SHAPED, _detail_key, _fee_ok, _frequency_ok, _is_value, _limit_ok, authority_check, responsibility_check,
)
from .evidence import (  # noqa: F401
    _BULLET, _DIGIT, _ENDS_WITH_CELL, _HEADING, _ITEM_BREAK, _ITEM_START, _LETTER_MARKER, _LIST_MARKER, _LIST_TITLE, _PRONOUN_START,
    _ROW_BREAK, _SENTENCE_BREAK, _TERMINAL, _local_units, _spans, _split, _units,
)
from .scoping import SectionGuard, _PageSegments, _merge_ranges, heading_marker  # noqa: F401
from .contracts import Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId, SpecialistTask
from .domain_cues import research_owned

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
    (r"\bre-?\s?marking|\bre-?\s?grad(?:e|es|ing)\b|grade\s+appeals?", "regrade", True),
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
    # "absence" is attendance only in a class or student context ("absences reaching 20%", "student absences"); "the
    # absence of material errors" is not (Step 4.8T-C2, D-080)
    (r"attendance|(?:(?<=\bstudent\s)|(?<=\bstudents'\s)|(?<=\bstudents\s)|(?<=\bunexcused\s)|(?<=\bexcused\s)|(?<=\bclass\s))absences?\b|"
     r"absences?\b(?=\s+(?:from|rates?|percentages?|reaching|exceed\w*|due\s+to|without|will|shall|may|must|is|are|record\w*|"
     r"polic\w*|limit\w*|in\s+(?:excess|class|lectures?|courses?)|for\s+(?:more|one|two|three|a|the\s+(?:course|class|semester))|"
     r"during|before|after|of\s+(?:more|students?|up|over|between|\d))|\s*[,.;:)]|\s*$)|"
     r"\babsent\b|\battend(?:s|ed|ing)?\s+(?:all\s+)?(?:the\s+|their\s+|scheduled\s+)?(?:theoretical\s+|practical\s+|online\s+)?"
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


def _consume_concepts(text: str) -> Tuple[List[Tuple[str, bool]], str]:
    """(concepts found in order, the text with every concept span blanked)."""
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
    """``text`` with every Teaching concept span blanked: what a clause says
    beyond its topic words (the specifics the answering unit must state)."""
    return _consume_concepts(text)[1]


def concepts(text: str) -> Concepts:
    """Teaching concepts found in ``text`` after normalisation (aliases such
    as syllabi/syllabus, grading/grade, examinations/exam)."""
    found, _work = _consume_concepts(text)
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
#: A location or system stated as such: a named system, an electronic place, or a locative phrase ("in the department
#: office", "room 204", "located at"). A bare "department", "office" or "college" inside a sentence about something
#: else names no place (Step 4.8T-C, D-079).
_LOCATION_EVIDENCE = re.compile(
    r"\b(?:blackboard|bb\s+ultra|banner|my\s?uos|\blms\b|portal|website|web\s?page|e-?\s?services?|e-?\s?course\s+files?|online|electronically|"
    r"rooms?\s+\w?\d+|room\s+(?:no\.?\s*)?\d+|buildings?\s+\w?\d+|(?:\d+(?:st|nd|rd|th)|ground|first|second|third)\s+floor|"
    r"located\s+(?:in|at|on)|(?:in|at|to)\s+the\s+(?:[\w'-]+\s+){0,3}?(?:office|department|building|library|desk|cent(?:er|re)|hall|campus|"
    r"registrar\w*)\b(?!\s+hours?)|main\s+campus|campus\b)", re.I)

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


def _quantity_ok(unit: str, subject: Concepts, subject_tokens: FrozenSet[str]) -> bool:
    """The shared count rule, read with Teaching's concept vocabulary."""
    return _details._quantity_ok(unit, subject, subject_tokens, concepts)


def _max_ok(unit: str) -> bool:
    return _details._limit_ok(unit, _LIMIT_MAX, is_table_row)


def _min_ok(unit: str) -> bool:
    return _details._limit_ok(unit, _LIMIT_MIN, is_table_row)


#: Approval verb and authority noun in one unit (the shared rule with the default bodies).
_authority_ok = authority_check()

#: A percentage stated as one: a percent sign or the word, a grade-table line (its ranges are percentages), never a
#: calendar line, whose day and date ranges are not values (Step 4.8T-C2, D-080).
_PERCENT_STATED = re.compile(r"\d\s*%|\bpercent(?:age)?s?\b", re.I)


def _percent_ok(unit: str) -> bool:
    if _weekday_shape(unit) or _CALENDAR_LINE.search(unit):
        return False
    return bool(_PERCENT_STATED.search(unit)) or _grade_shape(unit)


_CALENDAR_LINE = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
                            r"nov(?:ember)?|dec(?:ember)?)\b\s*\d|\b\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
                            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", re.I)

#: Parties the handbook names for "who manages / supports / is responsible for ..." clauses.
_RESPONSIBLE_PARTY = re.compile(r"\b(?:it\s+(?:department|services?|support|unit|team)|help\s*desk|registrar\w*|registration\s+department|deans?|chairs?|"
                                r"chairpersons?|department|college|councils?|committees?|directors?|coordinators?|administrators?|faculty\s+members?|"
                                r"instructors?|lecturers?|units?|offices?(?!\s+hours?)|teams?|cent(?:er|re)s?|division|services)\b", re.I)
_responsibility_ok = responsibility_check(_RESPONSIBLE_PARTY)


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
    ("percentage", "percentage or range", r"\b(?:percent(?:age)?s?|%|what\s+range|score\s+range)\b", lambda u: _percent_ok(u)),
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


def requested_details(clause: str) -> List[Detail]:
    """The details a clause explicitly asks for, in a fixed order. A quantity
    detail carries the counted subject ("office hours" in "how many office
    hours"); a faculty category named in the clause is a detail of its own
    (the evidence must name the same category)."""
    return _details.requested_details(clause, _COMPILED_FAMILIES, _COMPILED_CATEGORIES, concepts)


#: A sentence that opens with a context-dependent word is one statement with the sentence before it (select_quote
#: joins them): the grounding detail reads the pair, the detail families stay per unit.
_CONTEXT_START = re.compile(r"^\s*(?:such|these|this|those|it|they)\b", re.I)


def unit_coverage(details: Sequence[Detail], clause: Concepts, quote: str) -> FrozenSet[str]:
    """The largest set of requested details that ONE local unit of ``quote``
    contains while sharing a concept with the clause (shared rule). The
    grounding detail of a pronoun-initial sentence may also be stated by
    that sentence together with its antecedent (Step 4.8T-C, D-079); no
    other detail is ever combined across units."""
    shared = _evidence.unit_coverage(details, lambda text: relevance(clause, text), quote)
    grounding = [d for d in details if d.family == "grounding"]
    if not grounding or "grounding" in shared:
        return shared
    units = [_evidence._LIST_MARKER.sub("", u) for u in _evidence._local_units(quote)]
    for index, unit in enumerate(units[1:], start=1):
        if not _CONTEXT_START.match(unit) or relevance(clause, unit)[1] < 1:
            continue
        pair = units[index - 1] + " " + unit
        if all(d.check(pair) for d in grounding):
            others = frozenset(_detail_key(d) for d in details if d.family != "grounding" and d.check(unit))
            candidate = others | {"grounding"}
            if len(candidate) > len(shared):
                shared = candidate
    return shared


def detail_covered(detail: Detail, clause: Concepts, quote: str) -> bool:
    """True when some local unit of ``quote`` contains the detail and shares
    a concept with the clause (shared rule)."""
    return _evidence.detail_covered(detail, lambda text: relevance(clause, text), quote)


# ---------------------------------------------------------------------------
# Quote units and quote quality (shared rules in evidence.py; Teaching adds its table-row shapes)
# ---------------------------------------------------------------------------
_WEEKDAY_ROW = re.compile(r"^(?:mon|tue|wed|thu|fri|sat|sun)\b", re.I)
#: A grade-table line: a grade letter, then a numeric or textual range ("90-100", "Below 60") and grade points.
_GRADE_ROW = re.compile(r"^[A-F][+-]?\s+(?:\d{1,3}\s*-\s*\d{1,3}|(?:below|above|under|over|less\s+than|more\s+than)\s+\d{1,3})\b", re.I)
_SEMESTER_HEADER = re.compile(r"\b(fall|spring|summer|winter)\s+semester\s+(20\d\d)\s*[/\-–]\s*(20\d\d)", re.I)
_TERM_IN_TEXT = re.compile(r"\b(fall|spring|summer|winter)\b(?:\s+(?:semester\s+)?(20\d\d)\s*[/\-–]\s*(20\d\d))?", re.I)
_YEAR_SPAN = re.compile(r"(20\d\d)\s*[/\-–]\s*(?:20)?(\d\d)\b")


def _weekday_shape(text: str) -> bool:
    return bool(_WEEKDAY_ROW.match(text)) and any(_DIGIT.search(w) for w in text.split())


def _grade_shape(text: str) -> bool:
    return bool(_GRADE_ROW.match(text))


def is_table_row(text: str) -> bool:
    """A printed line that is table data rather than prose: a weekday-led
    calendar line, a grade-table line, or at least three digit-bearing cells
    with a label, within a table-like length."""
    return _evidence.is_table_row(text, (_weekday_shape, _grade_shape))


def quote_problem(quote: str, unit_kind: str, last_unit: bool = False) -> Optional[str]:
    """Why ``quote`` is not usable evidence, or None. ``unit_kind`` is
    "prose" (a unit of a paragraph) or "line" (a printed row)."""
    return _evidence.quote_problem(quote, unit_kind, last_unit, table_row=is_table_row)


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
        # Step 4.5A (NAV-12): where to submit a research form, which office handles it: navigation, handed to Institutional
        r"where\s+(?:do|can|should|must)\s+i\s+(?:submit|send|file)\s+(?:my\s+|the\s+|a\s+)?(?:research|grant|ethics|irb|conference|ip|patent|disclosure)|"
        r"which\s+(?:office|department|unit)\s+(?:handles|deals\s+with|processes)|"
        r"how\s+(?:do|does|can|could|should)\s+(?:i|we|one)\s+(?:contact|reach|get\s+in\s+touch\s+with)|to\s+whom\s+(?:should|do|can)\s+i\s+(?:send|submit|address)|"
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
        r"\b(?:research\w*|(?<!who\s)(?<!to\s)(?<!\bi\s)grants?(?!\s+(?:tenure|leave|access|permission|an?\s|students?\b|them\b|extensions?))|"
        r"seed\s+fund\w*|external\s+fund\w*|principal\s+investigator|co-?\s?investigators?|"
        r"ethics\s+(?:committee|approval|application|review|clearance)|irb|publications?|"
        r"publish\w*(?!\s+(?:the\s+|my\s+|a\s+|an\s+)?(?:grades?|marks?|course|announcements?|syllab\w*|results?|lecture))|"
        r"(?<!library\s)journals?(?!\s+(?:in\s+the\s+library|for\s+(?:my\s+)?assignments?))|"
        r"patents?|intellectual\s+property|scopus|citations?|h-?\s?index|"
        r"materials?\s+transfer\s+agreements?|mtas?|copyright\w*|conflicts?\s+of\s+(?:interest|commitment)|"
        r"(?:final|progress|annual)\s+(?:research|project)\s+reports?|licen[cs](?:e|es|ing)\s+agreements?|"
        r"approv\w*\s+(?:my\s+|the\s+|a\s+)?conferences?|conferences?\b[^.?]{0,40}\b(?:registration|documents?|deadline|funding|reimburs\w*|approv\w*)|"
        r"(?:registration\s+fees?|documents?)\b[^.?]{0,40}\bconferences?|"
        # Milestone 4 (Step 4.5A): conference policy (5.4) and research-only terms, so that a compound clause sent here
        # whole ("the per diem for conferences and my teaching load") keeps its research half as a handoff
        r"per\s+diem|conferences?\s+(?:registration|fees?|funding|funds?|travel|trips?|support|attendance|allowances?|expenses?|costs?|requests?|"
        r"documents?|documentation|deadlines?|budget|participation|papers?|proceedings)|"
        r"(?:local|international|external|overseas|scientific|academic|university\s+supported)\s+conferences?(?!\s+(?:calls?|rooms?|halls?))|"
        r"conferences?\s+(?:abroad|overseas)|"
        r"(?:attend(?:ing)?|participat\w*\s+in|present(?:ing)?\s+(?:a\s+paper\s+)?at|travel(?:l?ing)?\s+to)\s+(?:an?\s+|the\s+|my\s+)?conferences?"
        r"(?!\s+(?:calls?|rooms?|halls?|cent(?:er|re)s?))|"
        r"how\s+many\s+conferences|licens(?:e|es|ing)\s+(?:revenues?|income|agreements?)|licensing\s+(?:of\s+)?(?:inventions?|patents?|technolog\w*|ip|intellectual)|"
        r"petty\s+cash|royalt\w*|inventions?|iacuc|informed\s+consent|seed\s+(?:projects?|grants?)|tech(?:nology)?\s+transfer|tto|consultanc\w*|"
        r"consulting\s+(?:work|services|activities|contracts?|income|fees?|projects?|engagements?)|paid\s+consultants?)\b", re.I)),
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


#: WLAM faculty categories name a teaching-load category, never research
#: (the Coordinator and the Research specialist read them the same way).
_WLAM_CATEGORY = re.compile(r"\b(?:research[- ]intensive|active[- ]research|research[- ]active)\s+faculty\b", re.I)


def other_domains(clause: str) -> List[SpecialistId]:
    """Every other-domain specialist whose cue fires on the clause: Institutional
    Navigation first when its cue fires, then by cue count (ties: Research,
    Faculty Services & HR)."""
    clause = cue_text(clause)
    hits = []
    for order, (sid, pattern) in enumerate(_OTHER_CUES):
        text = _WLAM_CATEGORY.sub(lambda m: " " * len(m.group(0)), clause) if sid is SpecialistId.RESEARCH else clause
        count = len(pattern.findall(text))
        if not count and sid is SpecialistId.RESEARCH and research_owned(clause):
            count = 1                                           # the shared Research ownership lexicon (Step 4.5A-R)
        if count:
            hits.append((count, -order, sid))
    # navigation first (Step 4.5A, NAV-12): a where / who-to-contact / which-office clause goes to Institutional Navigation
    # before any subject domain, as the Research specialist does
    ranked = sorted(hits, key=lambda h: (h[2] is SpecialistId.INSTITUTIONAL, h[0], h[1]), reverse=True)
    return [sid for _count, _order, sid in ranked]


def classify_clause(clause: str) -> Tuple[bool, Optional[SpecialistId]]:
    """(carries a positive teaching cue, strongest other-domain specialist or None)."""
    teaching = bool(_TEACHING_CUES.search(cue_text(clause)))    # "office-hours" reads as "office hours"
    others = other_domains(clause)
    return teaching, (others[0] if others else None)


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


#: A place, contact channel or "who do I contact" ask (the Institutional Navigation cue read on one segment), a click,
#: menu or screen step excepted (that is a procedure request).
_NAVIGATION_CUES = re.compile(
    # "where is the curriculum committee office", "where's the registrar": a place; "where is the syllabus posted / kept /
    # uploaded" asks where a teaching artefact goes, a policy fact answered from teaching text
    r"\b(?:where(?:'s|\s+(?:is|are|was|were))\b(?![^.?]{0,50}\b(?:posted|uploaded|stored|kept|published|entered|recorded|submitted|"
    r"available|listed|shown|displayed|announced|distributed|documented|archived|saved|filed)\b)|"
    r"where\s+(?:can|could|do|does|should|shall|would|will|must|to)\s+(?:i|we|one|faculty|students?|the|a|an|my|our)\s+"
    r"(?:find|get|go|locate|visit|contact|reach|collect|pick|meet|obtain|see)\b|"
    r"\blocated\b|\blocations?\b|\bwhereabouts\b|\b(?:what|which)\s+(?:floor|building|room|campus)\b|\baddress\s+of\b|"
    r"\b(?:mailing|postal|office|physical|street|e\s?mail|email|web)\s+address\b|"
    r"\bhow\s+(?:do|does|can|could|should|would|to)\s+(?:i\s+|we\s+|one\s+)?(?:contact|reach|call|e\s?mail|get\s+in\s+touch|find|get\s+to|"
    r"go\s+to|locate|visit)\b|\bwho(?:m)?\s+(?:do|should|can|could|must|would)\s+(?:i|we)\s+(?:contact|ask|call|e\s?mail|see|go\s+to|refer\s+to|"
    r"approach|reach|talk\s+to|speak\s+to)\b|\bto\s+whom\s+(?:should|do|can|must|may)\s+(?:i|we)\b|"
    r"\bcontact\s+(?:person|details|information|numbers?)\b|\b(?:tele)?phone\b|\bfax\b|\bhotline\b|\bextension\s+(?:numbers?|no)\b|"
    r"\b(?:e\s?mail|email)\s+(?:address|id)\b|\bwhat\s+is\s+(?:the|their|its|his|her)\s+(?:e\s?mail|email|phone|telephone|website|url|extension|"
    r"hotline|number)\b|\b(?:its|their|the\s+office's|the\s+department's)\s+(?:e\s?mail|email|phone|telephone|website|extension|number)\b|"
    r"\b(?:which|what)\s+(?:office|unit|department|desk)\s+(?:handles?|deals?|is\s+(?:in\s+charge|responsible)|manages?|should|do|does|can|"
    r"to\s+contact|takes?\s+care|looks?\s+after)\b|\bto\s+which\s+(?:office|department|unit)\b|\bhelp\s*desk\b|\bservice\s+desk\b|"
    r"\bopening\s+hours\b)", re.I)
_CLICK_CUES = re.compile(r"\b(?:click\w*|press|select|tap|buttons?|screens?|menus?|tabs?|icons?|dropdown|log\s?in|login|upload\w*|links?|fields?)\b",
                         re.I)
_SEGMENT_SPLIT = re.compile(
    r"\s*(?:,\s*(?:and\s+|or\s+)?|;\s*|\s+and\s+(?:also\s+)?|\s+or\s+|\?\s*)"
    r"(?=(?:what|who|whom|whose|which|how|when|where|why|is|are|can|could|does|do|did|may|must|should|will|would)\b)|"
    r"\s*,?\s+and\s+(?=(?:(?:its|their|the|his|her|your|our)\s+)?(?:(?:tele)?phone|e\s?mail|email|fax|contact|location|address|"
    r"extension|website|hotline|whereabouts|opening\s+hours)\b)|^\s*how\s+and\s+(?=where\b)", re.I)
#: Contact or location evidence for a mixed clause: an e-mail address, a telephone number, a room, building or floor,
#: a web address, or a stated location.
_CONTACT_EVIDENCE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+|\+\s?\d[\d\s()-]{6,}\d|\b(?:tel|phone|telephone|fax|ext)\.?\s*:?\s*\+?\d|"
    r"\b(?:room|office|building|floor|block)\s+(?:no\.?\s*)?[A-Z]?\d+\b|\bwww\.|https?://|\blocated\s+(?:in|at|on)\b", re.I)


def clause_segments(clause: str) -> List[str]:
    """The segments of a clause (the clause itself when it has one)."""
    parts = [p.strip(" ,;?") for p in _SEGMENT_SPLIT.split(cue_text(clause))]
    return [p for p in parts if p] or [clause.strip()]


def navigation_intent(clause: str) -> Tuple[bool, bool]:
    """(asks for a place or a contact channel, also asks something
    substantive). A click, screen or upload step is a procedure request, not
    navigation. A pure navigation clause is Institutional Navigation's alone
    and is never answered from teaching text (Step 4.8T-C, D-079)."""
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
    """A contact channel or a location stated in the unit."""
    return bool(_CONTACT_EVIDENCE.search(unit))


# ---------------------------------------------------------------------------
# The ask gate: general / specific recognised / specific unknown (D-079)
# ---------------------------------------------------------------------------
#: Concepts verified elsewhere and not demanded of the answering unit: a calendar term (the calendar rule checks it)
#: and the words of a detail family (the family's own check verifies them).
_LOOSE_CONCEPTS: FrozenSet[str] = frozenset({"fall", "spring", "summer", "academic_year", "deadline", "penalty", "limit"})
#: Role words for grounding (Step 4.8T-C2, D-080), in priority order: a department chair is a chair, the registration
#: department is not the registrar, the Deans' Council is a council. Instructors, professors and faculty members are one
#: role, as the handbook uses the words; lecturers, a workload category, are their own. The person asking is faculty.
GROUNDING_ROLES: Tuple[Tuple[str, str], ...] = (
    # composite bodies first, so that "Deans' Council" is a council and "department chair" a chair, not a department
    (r"\b(?:department|college|university|academic|deans'?)\s+councils?(?!\w)|\bcouncils?(?!\w)", "council"),
    (r"\b(?:curricul\w+|study\s+plans?\s+and\s+curricula|teaching\s+and\s+learning|accreditation)\s+committees?(?!\w)|\bcommittees?(?!\w)", "committee"),
    (r"\b(?:department|departmental|program(?:me)?|college)\s+chair(?:s|persons?|person|man|men)?(?:'s)?(?!\w)|\bchair(?:s|persons?|person|man|men)?(?:'s)?(?!\w)|"
     r"\bhead\s+of\s+(?:the\s+)?department(?!\w)|\bhods?(?!\w)", "chair"),
    (r"\bvice[- ]chancellor\w*|\bvcaa\b|\bvcrgs\b|\bchancellor(?:'s)?(?!\w)", "vice_chancellor"),
    (r"\bdeans?(?:'s|')?(?!\w)", "dean"),
    (r"\bregistrar(?:'s|')?(?!\w)", "registrar"),
    (r"\bregistration\s+(?:department|office|system|unit)(?!\w)", "registration"),
    (r"\bcoordinators?\b", "coordinator"),
    (r"\b(?:it|information\s+technology)\s+(?:department|services?|support|unit|team)\b|\bhelp\s*desk\b", "it"),
    (r"\bblackboard\b|\bbb\s+ultra\b|\blms\b|\blearning\s+management\s+system\b", "blackboard"),
    (r"\bbanner\b", "banner"),
    (r"\bmy\s?uos\b", "myuos"),
    (r"\bteaching\s+assistants?\b", "teaching_assistant"),
    (r"\bstudents?(?:'s|')?(?!\w)|\blearners?(?!\w)|\badvisees?(?!\w)", "student"),
    (r"\b(?:part[- ]time|full[- ]time|regular|visiting|adjunct|new)?\s*(?:instructors?(?:'s|')?|teachers?|professors?|faculty(?:\s+members?(?:'s|')?)?|"
     r"academic\s+staff|teaching\s+staff)(?!\w)", "faculty"),
    (r"\b(?:senior\s+)?lecturers?(?:'s|')?(?!\w)", "lecturer"),
    (r"\b(?:the\s+)?university\s+of\s+sharjah\b(?:\s*\(\s*uos\s*\))?|\bthe\s+university\b|\buos\b", "university"),
    (r"\bdepartments?\b", "department"),
    (r"\bcolleges?\b", "college"),
)
#: Concepts that stand for roles: checked by role, never by the shared concept.
_ROLE_CONCEPTS: FrozenSet[str] = frozenset({"dean_chair", "council", "registration", "instructor", "lecturer", "teaching_assistant"})


def concept_spans(text: str) -> List[Tuple[int, int]]:
    """The spans of Teaching concept phrases in ``text`` (a subject that is no role: "course files", "office hours")."""
    lowered = cue_text(text.lower())
    spans: List[Tuple[int, int]] = []
    for pattern, _concept, _anchor in _COMPILED_CONCEPTS:
        for m in pattern.finditer(lowered):
            if not any(not (m.end() <= a or m.start() >= b) for a, b in spans):
                spans.append((m.start(), m.end()))
    return sorted(spans)


ASK_GATE = _grounding.AskGate(concepts, concept_residue, generic=GENERIC_TERMS, party=_RESPONSIBLE_PARTY, implies=_ANCHOR_IMPLIES,
                              loose=_LOOSE_CONCEPTS, table_row=is_table_row, roles=GROUNDING_ROLES, self_role="faculty",
                              role_concepts=_ROLE_CONCEPTS, calendar_row=lambda unit: _weekday_shape(unit) or bool(_CALENDAR_LINE.search(unit)),
                              phrases=concept_spans)
ASK_GENERAL, ASK_RECOGNISED, ASK_UNKNOWN = _grounding.GENERAL, _grounding.SPECIFIC_RECOGNISED, _grounding.SPECIFIC_UNKNOWN


def assess_ask(clause: str) -> _grounding.AskAssessment:
    """The shape of a clause and the grounding detail it adds: a general
    overview is grounded by its concepts, a specific recognised ask by its
    concepts, its remaining content words, a normative marker for a deontic
    proposition, a party with the asked action and a value where asked; a
    specific unknown ask receives a detail no evidence covers."""
    return ASK_GATE.assess(clause)


# ---------------------------------------------------------------------------
# Scope and the section guard
# ---------------------------------------------------------------------------
def compile_scope(section_map: SectionMap, source_id: str = APPROVED_SOURCE_ID,
                  owned: Sequence[Tuple[int, str, str]] = OWNED_SECTIONS) -> RetrievalScope:
    """The retrieval scope for the owned sections, as page ranges read from the
    section map (shared rule). Raises ``ValueError`` when the map is not the
    approved source's or lacks an owned section: the scope is never silently
    smaller."""
    return _scoping.compile_scope(section_map, source_id, owned)


# ---------------------------------------------------------------------------
# Quote selection (shared rule, read with Teaching's relevance and table-row shapes)
# ---------------------------------------------------------------------------
def select_quote(text: str, chunk_type: Optional[str], clause: Concepts,
                 allowed: Optional[Sequence[Tuple[int, int]]] = None,
                 details: Sequence["Detail"] = ()) -> Optional[Tuple[str, int, int]]:
    """The best quotable unit of ``text`` for the clause as ``(quote, shared
    anchors, shared vocabulary)``, or None when no unit is both relevant and
    quotable (see ``evidence.select_quote``; relevance is Teaching's)."""
    return _evidence.select_quote(text, chunk_type, lambda t: relevance(clause, t), allowed, details,
                                  max_words=MAX_QUOTE_WORDS, table_row=is_table_row)


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
    context: Optional[Dict[str, Any]]
    covered: FrozenSet[str]


class TeachingLearningSpecialist(Specialist):
    """Teaching & Learning Specialist, deterministic version 1."""

    specialist_id = SpecialistId.TEACHING

    def __init__(self, resources: TeachingResources, *,
                 owned_sections: Sequence[Tuple[int, str, str]] = OWNED_SECTIONS,
                 approved_sources: Optional[Sequence[str]] = None) -> None:
        if resources.source.source_id != APPROVED_SOURCE_ID or not resources.source.registered:
            raise ValueError("the Teaching specialist may only use the registered handbook %r" % APPROVED_SOURCE_ID)
        self.resources = resources
        self.owned_sections = tuple(owned_sections)
        # post-freeze PF-3B (D-110): the source-acceptance boundary. None keeps the I-9 handbook-only boundary; the runtime's
        # source policy may approve named official guides, admitted beside the unchanged handbook scope
        self.approved_sources = _scoping.approved_source_set(approved_sources, APPROVED_SOURCE_ID)
        self.scope = _scoping.approve_sources(compile_scope(resources.section_map, APPROVED_SOURCE_ID, self.owned_sections),
                                              self.approved_sources, APPROVED_SOURCE_ID)
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
    def from_knowledge_base(cls, kb, *, approved_sources: Optional[Sequence[str]] = None) -> "TeachingLearningSpecialist":
        return cls(TeachingResources.from_knowledge_base(kb), approved_sources=approved_sources)

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

    def _handoff(self, task: SpecialistTask, target: SpecialistId, clause: str, index: int, kind: str, ordinal: int = 0) -> HandoffRequest:
        suffix = "%d" % index if ordinal == 0 else "%d-%d" % (index, ordinal)   # the first handoff keeps the historical id
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
                navigation, substantive = navigation_intent(clause)
                if navigation and not substantive:
                    # "where is the curriculum committee office": a pure place or contact ask is Institutional Navigation's
                    # alone and is never answered from teaching text, whatever unit it names (Step 4.8T-C, D-079)
                    teaching, other = False, SpecialistId.INSTITUTIONAL
                if not teaching:
                    if other is not None:
                        handoffs.append(self._handoff(task, other, clause, index, "misrouted"))
                        reports.append({"index": index, "clause": clause, "outcome": "handed_off", "target": other.value})
                    else:
                        reports.append({"index": index, "clause": clause, "outcome": "unowned", "target": None})
                        missing.append("clause %d carries no teaching cue and was not searched: %s" % (index, clause[:120]))
                    continue
                owned_clauses += 1
                # the retrieval route of THIS clause (Step 4.7, D-078): a contact clause beside a substantive one must not
                # make it retrieve as a contact lookup
                clause_route = resolve_clause_retrieval_intent(clause, task.intent, sole_clause=len(processed) == 1,
                                                               substantive=substantive)
                # a shared clause keeps every foreign intent it names as a handoff (Milestone 4, Step 4.5A)
                handoffs.extend(self._handoff(task, o, clause, index, "shared", n) for n, o in enumerate(other_domains(clause)))
                procedural, system = is_procedure_request(clause, task.system)
                if system and system not in systems_seen:
                    systems_seen.append(system)
                clause_concepts = concepts(clause)
                details = requested_details(clause)
                # the ask gate (Step 4.8T-C, D-079): topic relevance alone never supports a specific ask; the answering
                # unit must state the asked specifics, and an unrecognised ask is never supported
                ask = assess_ask(clause)
                if not procedural:
                    details = list(details) + list(ask.details)      # a procedure request is never supported: its own rule
                if navigation:
                    # the substantive half is answered; the contact or location half needs contact evidence
                    details.append(Detail("contact", "contact or location details", contact_evidence))
                rejected: List[Dict[str, Any]] = []
                clause_findings: List[str] = []
                best_covered: FrozenSet[str] = frozenset()      # the most complete single evidence unit
                seen_anywhere: set = set()                       # details present somewhere, for the message
                for cand in self._candidates(clause, clause_route, clause_concepts, details, procedural, system, rejected):
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
                    finding = self._finding(task, len(findings) + 1, cand, index, clause_route, details)
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
                        if d.family == "ask_unknown":
                            missing.append("the specific ask of clause %d is not recognised by the Teaching specialist; the quoted evidence is "
                                           "related to the topic but does not confirm the answer: %s" % (index, clause[:120]))
                        elif d.family == "grounding":
                            missing.append("%s: the quoted evidence is on the topic but does not settle the question for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        elif d.family == "contact":
                            missing.append("contact or location details are not in the Teaching evidence for clause %d; Institutional "
                                           "Navigation holds them: %s" % (index, clause[:120]))
                        elif _detail_key(d) in seen_anywhere:
                            missing.append("%s is not stated together with the other requested details in one place for clause %d: %s"
                                           % (d.label, index, clause[:120]))
                        else:
                            missing.append("%s not found in the evidence for clause %d: %s" % (d.label, index, clause[:120]))
                else:
                    outcome = "supported"
                reports.append({"index": index, "clause": clause, "outcome": outcome, "procedural": procedural,
                                "system": system, "findings": clause_findings, "rejected": rejected,
                                "requested_details": [d.label for d in details], "uncovered_details": [d.label for d in uncovered],
                                "ask_shape": ask.shape, "route": clause_route,
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
    "ANCHOR_CONCEPTS", "APPROVED_SOURCE_ID", "ASK_GATE", "ASK_GENERAL", "ASK_RECOGNISED", "ASK_UNKNOWN", "CATEGORY_GROUPS", "Concepts",
    "GROUNDING_ROLES",
    "Detail", "EXCLUDED_SECTIONS", "GENERIC_TERMS", "assess_ask", "concept_residue", "contact_evidence", "navigation_intent",
    "MAX_CANDIDATES_PER_CLAUSE", "MAX_CLAUSES", "MAX_FINDINGS_PER_CLAUSE", "MAX_QUOTE_WORDS", "OWNED_SECTIONS", "SYSTEMS",
    "SectionGuard", "TeachingLearningSpecialist", "TeachingResources", "VOCABULARY_CONCEPTS", "calendar_context",
    "classify_clause", "compile_scope", "concepts", "detail_covered", "heading_marker", "is_procedure_request", "is_relevant",
    "is_table_row", "quote_problem", "relevance", "requested_details", "requested_term", "select_quote", "system_named", "unit_coverage",
]
