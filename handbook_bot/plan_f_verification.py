"""
Findings-aware verification of a Plan F specialist result (runtime integration, phase I-4).

DECISION OWNED
    Whether ONE structured specialist result (``SpecialistFindings``) is suitable as the direct answer to the
    user's exact request, before the adapter renders it as a ``QAResult``. The principle of the phase: a result
    may be supported by a source and still be unsuitable as the final answer. Nothing here retrieves, routes,
    rewrites a quote, executes a handoff or calls a language model; the frozen specialists are read, never
    changed, and the legacy Verifier (``agents.verifier``) is untouched.

WHAT IS VERIFIED (deterministic, structured, in this order)
    1. Status and structure: only a ``supported`` result with one clause report, no handoff, no ``missing`` entry,
       no model call and a recognised specific ask (or an Institutional ``read`` outcome at an eligible tier) may
       proceed. Partial, not found, out of scope, error, handoff, multi-clause and general asks never render.
    2. The request shape: the clause is read deterministically into a request kind (contact field, location,
       quantity, amount, percentage, duration, frequency, timing, value, definition, authority, responsibility,
       selection, eligibility, obligation, yes/no, enumeration, general), with its named entities, its content
       noun phrases, its condition phrases, the counted unit of a quantity ask and the asked action.
    3. Per finding: evidence integrity (the I-1 quote guard, reused), the I-1 subject binding (reused), then the
       request-aware checks on the ONE local evidence unit that answers: the requested detail is present in that
       unit (a telephone value, a location entry, a number tied to the asked unit noun, a money amount, a
       percentage, a copular definition, the asked action by a named party in the asked position, a normative
       statement, the asked predicate with the asked polarity); the clause's named entities are in that unit;
       no asked category modifier is contradicted by the unit's modifier of the same noun (annual versus
       maternity leave, part-time versus full-time); every condition phrase of the clause is stated (per week,
       with full pay, at once); the quote does not span two provisions (a sentence of another numbered item or
       section before or after the answering unit is excess evidence).
    4. Completeness: an enumerative ask (what does X provide, what are the X, what does X do, requirements)
       cannot be proven complete by one finding and is not rendered directly (I-5 aggregation may). A quote whose
       other sentences belong to the same provision is an elaboration and is kept whole; one that spans another
       provision is rejected (no trimming: the specialist's quote is never rewritten in this phase).
    5. Several findings: the primary finding must pass; a secondary finding that fails any check, or that states
       a different value for the asked quantity, is dropped and recorded, never rendered.

RESULT
    :class:`FindingsVerification`, an immutable record with a machine-testable :class:`ReasonCode`, the finding
    status, the request kind, per-dimension verdicts (subject, relation, requested detail, evidence integrity),
    a completeness class, ``safe_to_render`` and the ids of the findings that may be rendered. ``to_dict`` is JSON
    serialisable and carries no question or answer text.

FAILURE POLICY
    :func:`verify_findings` never raises: an unexpected exception becomes ``VERIFICATION_ERROR`` with the message
    passed through ``base.redact_error``, ``accepted`` False and ``safe_to_render`` False, so the caller falls back.

PHASE I-5 (aggregation support; no verdict above changes)
    ``status_code`` is split into :func:`run_code` (the result as a whole) and :func:`report_code` (one clause
    report) so that the deterministic aggregator (``plan_f_aggregation``) can apply the same rules to a result that
    holds one report per request clause. :func:`clause_view` projects such a result onto ONE of its reports and the
    findings that answer it; the verifier then runs on the view exactly as on a single-clause result.
    :func:`provision_slice` derives, from a quote the verifier rejected as EXCESS_EVIDENCE, the exact substring that
    holds the answering unit's own provision (and its lead-in), with explicit provenance back to the original
    finding; :func:`verify_findings_sliced` accepts that slice only when it passes every check on its own. Nothing
    is reworded and no frozen specialist changes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents.specialists import navigation_cues as _nav
from .agents.specialists.base import redact_error
from .agents.specialists.contracts import Finding, FindingStatus, SpecialistFindings, SpecialistId
from .plan_f_runtime import (ASK_SPECIFIC_RECOGNISED, INSTITUTIONAL_OUTCOME_READ, INSTITUTIONAL_TIERS, RELATION_STRONG, adapt_finding,
                             entity_in_text, named_entities, quote_problem, subject_binding)

#: Integration phase and contract version of this module.
PHASE = "I-4"
VERSION = "plan-f-findings-verification-1"


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
class ReasonCode(str, Enum):
    """Why a specialist result may or may not be rendered as the direct answer."""

    ACCEPTED = "ACCEPTED"
    UNSUPPORTED_STATUS = "UNSUPPORTED_STATUS"              # error, model call, malformed or ineligible report
    PARTIAL_FINDING = "PARTIAL_FINDING"
    NOT_FOUND = "NOT_FOUND"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    HANDOFF_REQUIRED = "HANDOFF_REQUIRED"
    MULTIPART_UNSUPPORTED = "MULTIPART_UNSUPPORTED"        # more than one clause report, or clauses left unprocessed
    GENERAL_ASK_MISMATCH = "GENERAL_ASK_MISMATCH"
    UNRECOGNISED_ASK = "UNRECOGNISED_ASK"                  # the specialist could not recognise the specific ask
    SUBJECT_MISMATCH = "SUBJECT_MISMATCH"
    RELATION_MISMATCH = "RELATION_MISMATCH"
    REQUESTED_DETAIL_MISSING = "REQUESTED_DETAIL_MISSING"
    CONDITION_MISMATCH = "CONDITION_MISMATCH"
    MODIFIER_MISMATCH = "MODIFIER_MISMATCH"
    ENUMERATION_INCOMPLETE = "ENUMERATION_INCOMPLETE"
    EXCESS_EVIDENCE = "EXCESS_EVIDENCE"
    VALUE_CONFLICT = "VALUE_CONFLICT"                      # a secondary finding states another value for the asked quantity
    INVALID_EVIDENCE_REFERENCE = "INVALID_EVIDENCE_REFERENCE"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"


class Completeness(str, Enum):
    """How complete the rendered evidence is for the request."""

    COMPLETE = "complete"                      # the answering unit is the quote (or the quote's other units only frame it)
    ELABORATED = "elaborated"                  # the quote adds sentences of the same provision
    CONTRACT = "specialist_contract"           # the verifier does not read this request kind; the specialist's contract decided
    PARTIAL = "partial"
    ENUMERATION_UNPROVEN = "enumeration_unproven"
    EXCESS = "excess"
    NOT_ASSESSED = "not_assessed"


class RequestKind(str, Enum):
    CONTACT = "contact"
    LOCATION = "location"
    QUANTITY = "quantity"
    AMOUNT = "amount"
    PERCENTAGE = "percentage"
    DURATION = "duration"
    FREQUENCY = "frequency"
    TIMING = "timing"
    VALUE = "value"
    DEFINITION = "definition"
    AUTHORITY = "authority"
    RESPONSIBILITY = "responsibility"
    SELECTION = "selection"
    ELIGIBILITY = "eligibility"
    OBLIGATION = "obligation"
    YES_NO = "yes_no"
    ENUMERATION = "enumeration"
    GENERAL = "general"
    UNRECOGNISED = "unrecognised"


#: Request kinds whose answer is a set that one finding cannot prove complete.
ENUMERATIVE_KINDS = frozenset({RequestKind.ENUMERATION})
#: Request kinds answered by a number of some unit.
VALUE_KINDS = frozenset({RequestKind.QUANTITY, RequestKind.AMOUNT, RequestKind.PERCENTAGE, RequestKind.DURATION, RequestKind.VALUE})


# ---------------------------------------------------------------------------
# Tokens, stems and small lexicons
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’]*(?:-[A-Za-z]+)*|\d+(?:[.,]\d+)*")
_NUMBER_WORDS: Dict[str, float] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}
_STOP = frozenset("""
a an the this that these those it its their my our your his her any all each every such other another some no
of for to from by with without on in at during after before under per and or if when while unless within into about than as through via
is are was were be been being do does did can could may might must shall should will would has have had
what which who whom whose how when where why many much often long
""".split())
_GENERIC = frozenset({"university", "uos", "uae", "sharjah", "handbook", "please", "tell", "me", "us", "i", "we", "you"})
_DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those", "its", "their", "my", "our", "your", "his", "her", "any", "all", "each", "every", "such"})
#: Words that split the clause into phrases: prepositions, conjunctions and auxiliaries.
_SPLIT = frozenset("""
of for to from by with without on in at during after before under per and or if when while unless within into about than as through via
is are was were be been being do does did can could may might must shall should will would has have had get gets got
""".split())
#: Condition introducers: the phrase they open must be stated in the answering unit.
_CONDITION_PREPS = frozenset({"per", "with", "without", "during", "after", "before", "if", "when", "while", "unless", "within"})
#: Condition idioms and the wordings that state them (stem tuples).
_IDIOMS: Dict[Tuple[str, ...], Tuple[Tuple[str, ...], ...]] = {
    ("at", "once"): (("simultaneously",), ("at", "once"), ("at", "the", "same", "time"), ("concurrently",), ("at", "any", "one", "time")),
    ("at", "the", "same", "time"): (("simultaneously",), ("at", "once"), ("at", "the", "same", "time"), ("concurrently",)),
    ("simultaneously",): (("simultaneously",), ("at", "once"), ("at", "the", "same", "time"), ("concurrently",)),
}
_PERIODS = ("week", "semester", "month", "year", "day", "term", "course", "section")
#: Contrast sets: an asked modifier of a noun is contradicted when the evidence modifies the same noun with another member.
_CONTRASTS: Tuple[Tuple[str, ...], ...] = (
    ("annual", "vacation", "sick", "maternity", "paternity", "hajj", "study", "sabbatical", "unpaid", "special", "emergency",
     "compassionate", "bereavement", "casual", "marriage", "examination", "research", "medical"),
    ("full", "half", "unpaid", "without"),                                     # pay status
    ("full-time", "part-time", "fulltime", "parttime", "female", "male", "women", "men", "new", "existing", "current", "former",
     "retired", "visiting", "regular", "adjunct", "emeritus", "ranked", "senior", "junior", "clinical", "undergraduate", "graduate",
     "postgraduate", "doctoral"),                                               # population category
    ("working", "calendar", "academic", "business"),                            # days
    ("final", "midterm", "mid-term", "make-up", "makeup", "entrance", "placement"),
    ("regular", "summer", "fall", "spring"),                                    # semester
    ("credit", "contact", "office", "teaching", "working", "instructional", "class", "lab", "laboratory"),   # hours
    ("entry", "residence", "residency", "exit", "visit", "transit"),             # visas
    ("telephone", "phone", "fax", "mobile", "email", "e-mail"),
    ("petty", "travel", "salary", "housing", "furniture", "relocation"),         # advances and allowances
    ("main", "medical", "law", "dental", "women's", "men's", "branch"),           # libraries and campuses
)
_CONTRAST_INDEX: Dict[str, Tuple[int, ...]] = {}
for _i, _set in enumerate(_CONTRASTS):
    for _w in _set:
        _CONTRAST_INDEX.setdefault(_w, tuple())
        _CONTRAST_INDEX[_w] = _CONTRAST_INDEX[_w] + (_i,)

_MEASURE_HEADS = frozenset("""
load workload period notice number fee fees rate amount allowance salary limit maximum minimum percentage share deadline duration length size
cost price ceiling cap threshold hours days weeks months years credit credits age quota stipend bonus compensation budget
""".split())
_PLURAL_SET_HEADS = frozenset("""
requirements criteria conditions documents steps procedures stages responsibilities duties functions services benefits areas priorities
topics rules regulations policies guidelines types kinds categories options forms components elements objectives goals
""".split())
_BODY_NOUNS = frozenset({"office", "department", "committee", "unit", "body", "council", "board", "centre", "center", "authority", "division",
                         "deanship", "directorate", "section", "person", "party"})
_ROLE_WORDS = frozenset("""
dean deans chair chairs chairperson director directors registrar council committee board office department unit centre center chancellor
vice-chancellor president provost coordinator investigator supervisor hr vcaa vcrgs vc deanship college university faculty member members staff
administration management institute library hospital clinic
""".split())
_POPULATION_HEADS = frozenset({"faculty", "instructor", "instructors", "student", "students", "member", "members", "staff", "lecturer", "lecturers",
                               "professor", "professors", "employee", "employees", "candidate", "candidates", "dean", "deans", "chair", "chairs"})
_VERBS = frozenset("""
arrange arranges arranged arranging assist assists assisted assisting provide provides provided providing offer offers offered allow allows
allowed permit permits permitted require requires required pay pays paid cover covers covered include includes included entitle entitles
entitled grant grants granted approve approves approved apply applies applied need needs needed receive receives received hold holds held
teach teaches taught supervise supervises supervised change changes changed modify modifies modified use uses used take takes took submit
submits submitted attend attends attended reimburse reimburses reimbursed sponsor sponsors sponsored process processes processed issue issues
issued help helps helped support supports supported handle handles handled manage manages managed oversee oversees determine determines
determined decide decides decided draft drafts drafted review reviews reviewed sign signs signed chair chairs chaired own owns owned
authorise authorises authorized authorize authorizes recommend recommends nominate nominates appoint appoints evaluate evaluates assess
assesses administer administers coordinate coordinates maintain maintains operate operates run runs deal deals respond responds report
reports post posts enter enters spend spends claim claims qualify qualifies get gets obtain obtains do does make makes conduct conducts
liable responsible mandatory compulsory eligible accept accepts bar bars prohibit prohibits prevent prevents correct corrects
""".split())
#: Verb forms that are verbs when they close a clause phrase ("faculty members get"); nouns that double as verbs
#: ("full pay", "research grants", "employment offers", "annual reports") are deliberately absent.
_BASE_VERBS = frozenset("""
get gets receive receives hold holds spend spends supervise supervises change modify modifies use uses take takes submit submits attend attends
obtain obtains apply applies qualify qualifies make makes have has need needs include includes provide provides arrange arranges assist assists
handle handles manage manages oversee oversees determine determines decide decides draft drafts approve approves sign signs chair chairs own owns
enter enters teach teaches conduct conducts correct corrects prevent prevents prohibit prohibits paid held taken given made done allow allows permit
permits require requires entitle entitles cover covers deliver delivers earn earns do does did
""".split())
_NORMATIVE = re.compile(r"\b(?:must|shall|should|required|require|obliged|obligated|mandatory|compulsory|expected\s+to|responsible\s+for|"
                        r"has\s+to|have\s+to|needs?\s+to|are\s+to|is\s+to|will)\b", re.I)
_RESPONSIBILITY = re.compile(r"\b(?:responsible\s+for|responsibility\s+(?:for|of|to)|in\s+charge\s+of|handl(?:es|ed|ing)|manag(?:es|ed|ing)|"
                             r"overse(?:es|en|eing)|administer(?:s|ed|ing)|coordinat(?:es|ed|ing)|maintain(?:s|ed|ing)|operat(?:es|ed|ing)|"
                             r"deals?\s+with|takes?\s+care\s+of|entrusted\s+with|tasked\s+with|serves?\s+as)\b", re.I)
_ELIGIBLE = re.compile(r"\b(?:eligible|entitled|qualif(?:y|ies|ied)|may\s+apply|can\s+apply|open\s+to|eligibility)\b", re.I)
_NEGATION = frozenset({"not", "no", "never", "cannot", "neither", "nor", "n't", "without"})
#: Phase I-8: the predicate words that let a statement answer a yes/no ask whose own action could not be read ("Is the
#: teaching load for lecturers 12 credit hours?", "Can a faculty member be PI for two projects?"): a permission, prohibition,
#: entitlement or requirement about the asked subject. A plain value statement never confirms or refutes such an ask.
_YES_NO_PREDICATE = re.compile(r"\b(?:may|can|cannot|could|allowed|permitted|prohibited|forbidden|entitled|eligible|not\s+(?:allowed|permitted))\b", re.I)
_COPULA = re.compile(r"^\s*(?:,\s*[^,]{0,80},)?\s*(?:is|are|was|were|means|refers\s+to|stands\s+for|comprises|consists\s+of|describes|denotes|"
                     r"represents|provides|aims|serves|shall\s+mean|constitutes|:)\b", re.I)
_PAREN_ACRONYM = re.compile(r"^\s*\(\s*[A-Za-z&\.\s-]{1,12}\s*\)")
_MONEY = re.compile(r"(?<![A-Za-z])(?:aed|dhs\.?|dirhams?|usd|us\$|\$|€|£|eur|gbp)\s*\d[\d,]*(?:\.\d+)?|"
                    r"\b\d[\d,]*(?:\.\d+)?\s*(?:aed|dhs\.?|dirhams?|usd|dollars?|euros?|pounds?)\b|\b(?:free\s+of\s+charge|no\s+fee|no\s+charge|at\s+no\s+cost)\b", re.I)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent\b|per\s+cent\b)", re.I)
_FREQUENCY = re.compile(r"\b(?:per|each|every|a)\s+(?:week|semester|month|year|day|term|course|section)\b|\b(?:weekly|daily|monthly|annually|"
                        r"yearly|once|twice|\d+\s+times)\b", re.I)
_TIMING = re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|"
                     r"sept?|oct|nov|dec|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b|"
                     r"\bwithin\s+(?:\w+(?:-\w+)?\s+)?(?:\w+\s+)?(?:hours?|days?|weeks?|months?|years?)\b|\b(?:no\s+later\s+than|prior\s+to|"
                     r"before\s+the|after\s+the|by\s+the\s+(?:end|start|beginning)|at\s+the\s+(?:end|start|beginning)|first\s+(?:week|day)|last\s+day|"
                     r"deadline)\b|\b\d{1,2}\s+\w+\s+20\d\d\b", re.I)
_LOCATION_WORDS = re.compile(r"\b(?:located|location|building|campus|floor|room|wing|block|hall|premises)\b|\b[MW]\d{1,2}\b", re.I)
_TIME_UNITS = frozenset({"hour", "day", "week", "month", "year", "semester", "term", "minute"})
_MEASURE_UNITS = _TIME_UNITS | frozenset({"credit", "percent", "aed", "dirham", "dh", "student", "course", "section", "thes", "colleg",
                                           "member", "time", "page", "word", "copy", "copi", "project", "paper", "publication", "class", "lecture"})

#: Provision starts inside a quote: bullets, "o " items, numbered and lettered items, dotted section numbers followed by a title word.
_PROVISION_START = re.compile(r"(?:(?<=\s)|^)(?:[•▪�]\s+|o\s+(?=[\"“A-Z])|\d{1,2}\.\s+(?=[A-Z])|\d+(?:\.\d+){1,3}\s*(?=[A-Z])|"
                              r"(?<=[.:;!?]\s)[a-z]\.\s+(?=[A-Z])|(?<=[.:;!?]\s)[A-Z]\.\s+(?=[A-Z]))")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(\[\d•])")
_ABBREVIATIONS = ("no.", "nos.", "dr.", "prof.", "mr.", "mrs.", "ms.", "e.g.", "i.e.", "vs.", "art.", "sec.", "para.", "p.", "pp.", "fig.", "vol.",
                  "ext.", "tel.", "st.", "etc.")
_LIST_MARKER = re.compile(r"^(?:[•▪�-]\s+|o\s+(?=[\"“A-Z])|\d{1,2}\.\s+|[a-z]\.\s+|[A-Z]\.\s+(?=[A-Z])|\d+(?:\.\d+){1,3}\s*)")


_IRREGULAR = {"paid": "pay", "held": "hold", "taught": "teach", "taken": "take", "took": "take", "given": "give", "gave": "give", "made": "make",
              "done": "do", "got": "get", "gotten": "get", "drawn": "draw", "drew": "draw", "sent": "send", "spent": "spend", "met": "meet",
              "chosen": "choose", "chose": "choose", "written": "write", "wrote": "write", "sought": "seek", "brought": "bring", "kept": "keep",
              "left": "leave", "lent": "lend", "built": "build", "found": "find", "run": "run", "ran": "run", "began": "begin", "begun": "begin"}


def _stem(word: str) -> str:
    w = word.lower().replace("’", "'")
    if w.isdigit() or w[:1].isdigit():
        return w
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 4 and w.endswith("es") and not w.endswith("ses") and not w.endswith("ees"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def _tokens(text: str) -> List[str]:
    return [t.lower().replace("’", "'") for t in _TOKEN.findall(text or "")]


def _stems(text: str) -> List[str]:
    return [_stem(t) for t in _tokens(text)]


def _number_value(token: str) -> Optional[float]:
    """The numeric value of a token: digits (commas allowed) or a number word, including "forty-eight"."""
    t = token.lower().replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    if t in _NUMBER_WORDS:
        return _NUMBER_WORDS[t]
    if "-" in t:
        parts = t.split("-")
        if all(p in _NUMBER_WORDS for p in parts):
            return float(sum(_NUMBER_WORDS[p] for p in parts))
    return None


def _contrast_sets(word: str) -> Tuple[int, ...]:
    return _CONTRAST_INDEX.get(word, ()) + _CONTRAST_INDEX.get(_stem(word), ())


# ---------------------------------------------------------------------------
# Reading the request
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RequestShape:
    """What the clause asks, as this verifier reads it (deterministic, domain-neutral)."""

    kind: RequestKind
    entities: Tuple[str, ...]                                  # named entities (the I-1 reader)
    phrases: Tuple[Tuple[Tuple[str, ...], str], ...]           # content noun phrases: (modifier words, head stem)
    conditions: Tuple[Tuple[str, ...], ...]                    # condition phrases, as stem tuples
    unit: Optional[str] = None                                 # counted noun (stem) of a quantity ask
    unit_modifiers: Tuple[str, ...] = ()                       # its modifiers ("credit" of "credit hours")
    field: Optional[str] = None                                # contact field of a contact ask
    verb: Optional[str] = None                                 # asked action (stem) of a party, yes/no or selection ask
    object_head: Optional[str] = None                          # asked object class (stem) of a selection ask
    body_noun: Optional[str] = None                            # "committee" of "which committee"
    negated: bool = False                                      # the clause itself is negative ("Is X not ...")
    actor: Optional[str] = None                                # population or role head (stem) that acts, if named

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "entities": len(self.entities), "phrases": len(self.phrases), "conditions": len(self.conditions),
                "unit": self.unit, "field": self.field, "verb": self.verb, "object_head": self.object_head, "body_noun": self.body_noun,
                "negated": self.negated}


_CONTACT_FIELDS = (
    (_nav.FAX, re.compile(r"\bfax(?:es)?\b", re.I)),
    (_nav.EMAIL, re.compile(r"\be-?mail\b", re.I)),
    (_nav.EXTENSION, re.compile(r"\bextensions?\b|\bext\b", re.I)),
    (_nav.WEBSITE, re.compile(r"\bweb\s?sites?\b|\bweb\s?pages?\b|\burls?\b|\bhome\s?pages?\b", re.I)),
    (_nav.TELEPHONE, re.compile(r"\b(?:tele)?phones?\b|\bhotlines?\b|\bcontact\s+(?:number|details|information)\b|\bcall\b", re.I)),
)
_LOCATION_ASK = re.compile(r"^\s*where\b|\b(?:location|located|address|which\s+(?:building|campus|floor))\b", re.I)
_HOW_MANY = re.compile(r"\b(?:how\s+many|number\s+of|count\s+of|total\s+number\s+of)\s+((?:[\w'’-]+\s*){1,4})", re.I)
_HOW_MUCH = re.compile(r"\bhow\s+much\b", re.I)
_PERCENT_ASK = re.compile(r"\b(?:what|which)\s+(?:percentage|percent|share|proportion)\b|\bhow\s+much\s+of\b", re.I)
_HOW_LONG = re.compile(r"\bhow\s+long\b|\bwhat\s+is\s+the\s+(?:length|duration)\b", re.I)
_HOW_OFTEN = re.compile(r"\bhow\s+often\b|\bhow\s+frequently\b", re.I)
_WHEN = re.compile(r"^\s*(?:when|by\s+when)\b|\b(?:what\s+(?:is\s+the\s+)?(?:deadline|date|day|time)|deadline\s+for)\b", re.I)
_WHO = re.compile(r"^\s*(?:who|whom|which\s+(office|department|committee|unit|body|council|board|centre|center|authority|division|deanship|"
                  r"directorate|section|person|party)|what\s+(office|department|committee|unit|body|council|board))\b", re.I)
_YESNO = re.compile(r"^\s*(?:does|do|did|is|are|was|were|can|could|may|might|must|should|shall|will|would|has|have|had|am)\b", re.I)
_OBLIGATION_ASK = re.compile(r"^\s*what\s+(?:must|should|shall|do(?:es)?\s+\w+\s+(?:have|need)\s+to|is\s+required\s+(?:of|from)|happens\s+if)\b|"
                             r"^\s*what\s+(?:is|are)\s+(?:my|the|a|an)?\s*(?:obligation|duty|duties|obligations)\b", re.I)
_REQUIREMENT_ASK = re.compile(r"^\s*what\s+(?:is|are)\s+(?:the\s+)?(?:requirements?|criteria|conditions|documents|steps|procedures?|process|stages)\b|"
                              r"^\s*what\s+documents?\b|^\s*which\s+documents?\b", re.I)
_FUNCTION_ASK = re.compile(r"^\s*what\s+(?:does|do)\s+(.+?)\s+do(?:\s+(?:for|to|with|in|on|about|regarding|during)\b.*)?\s*\??$|"
                           r"^\s*what\s+(?:is|are)\s+(.+?)\s+responsible\s+for\s*\??$|"
                           r"^\s*what\s+(?:is|are)\s+the\s+(?:role|roles|purpose|function|functions|responsibilit(?:y|ies)|duties|services|tasks)\s+of\b", re.I)
_OPEN_OBJECT_ASK = re.compile(r"^\s*what\s+(?:does|do|can|will|is|are)\s+(.+?)\s+(provide[sd]?|offer[sd]?|include[sd]?|cover(?:s|ed)?|handle[sd]?|"
                              r"support(?:s|ed)?|manage[sd]?|arrange[sd]?|deliver(?:s|ed)?|give[sn]?|gave|entail(?:s|ed)?|involve[sd]?|require[sd]?|provided|offered)\b", re.I)
_WH_OBJECT_ASK = re.compile(r"^\s*(?:what|which)\s+([a-z][\w'’-]*(?:\s+[a-z][\w'’-]*)?)\s+(?:does|do|did|is|are|can|will|must|should|may)\b", re.I)
_WHAT_IS = re.compile(r"^\s*what\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s*\??$", re.I)
_POLICY_ASK = re.compile(r"\b(?:policy|policies|rules?|regulations?|guidelines?|procedures?)\s+(?:on|for|about|regarding|governing|of)\b|"
                         r"^\s*(?:tell\s+me\s+about|explain|describe|give\s+me\s+an\s+overview|what\s+about|information\s+(?:on|about))\b", re.I)
_ELIGIBLE_ASK = re.compile(r"\b(?:eligible|eligibility|qualif(?:y|ies)|entitled\s+to\s+apply)\b", re.I)
_RESPONSIBLE_VERB = re.compile(r"\b(?:responsible|handles?|manages?|oversees?|in\s+charge|deals?\s+with|administers?|coordinates?|takes?\s+care|"
                               r"maintains?|operates?|runs?|supports?|provides?)\b", re.I)
_LEADING_FRAME = re.compile(r"^\s*(?:how\s+(?:many|much|long|often|far)|what\s+(?:percentage|percent|share|proportion)\s+of|what\s+(?:is|are|was|were|does|do|did|"
                            r"must|should|shall|can|will|happens)|which|who|whom|whose|when|where|why|does|do|did|is|are|was|were|can|could|may|might|must|"
                            r"should|shall|will|would|has|have|had|am|tell\s+me|explain|describe)\b\s*", re.I)


def _phrases_of(clause: str) -> Tuple[List[Tuple[Tuple[str, ...], str]], List[Tuple[str, ...]], Optional[str], bool]:
    """Content noun phrases of the clause (modifier words, head stem), its condition phrases (stem tuples), the acting
    population or role head and whether the clause is negated."""
    text = _LEADING_FRAME.sub("", clause.strip().rstrip("?.! "))
    toks = _tokens(text)
    phrases: List[Tuple[Tuple[str, ...], str]] = []
    conditions: List[Tuple[str, ...]] = []
    actor: Optional[str] = None
    negated = any(t in ("not", "n't", "never") for t in _tokens(clause))
    run: List[str] = []
    pending_condition = False
    first_run = True

    def flush() -> None:
        nonlocal run, pending_condition, actor, first_run
        words = [w for w in run if w not in _DETERMINERS and w not in _GENERIC and w not in _STOP]
        if first_run and len(words) > 1 and words[0] in _VERBS:
            words = words[1:]                                        # "Who determines the funding ceiling": the asked action opens the phrase
        while len(words) > 1 and (words[-1] in _BASE_VERBS or (words[-1] in _VERBS and words[-1].endswith(("ed", "ing")))):
            words.pop()                                              # "female faculty entitled", "faculty members get": the verb closes it
        if words:
            first_run = False
            head = _stem(words[-1])
            mods = tuple(w for w in words[:-1])
            if head not in {_stem(v) for v in _VERBS} or len(words) > 1:
                phrases.append((mods, head))
            if actor is None and words[-1] in _POPULATION_HEADS:
                actor = head
            if pending_condition:
                conditions.append(tuple(_stem(w) for w in words))
        run = []
        pending_condition = False

    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok in _SPLIT:                                            # prepositions, conjunctions and auxiliaries bound a phrase; a verb may be a noun
            flush()
            if tok in _CONDITION_PREPS:
                pending_condition = True
                if tok == "at" and i + 1 < len(toks) and toks[i + 1] == "once":
                    conditions.append(("at", "once"))
                    pending_condition = False
                    i += 2
                    continue
            elif tok == "at" and i + 1 < len(toks) and toks[i + 1] == "once":
                conditions.append(("at", "once"))
                i += 2
                continue
            i += 1
            continue
        run.append(tok)
        i += 1
    flush()
    if actor is None:
        for mods, head in phrases:
            if head in {_stem(p) for p in _POPULATION_HEADS}:
                actor = head
                break
    return phrases, conditions, actor, negated


def _unit_of(clause: str) -> Tuple[Optional[str], Tuple[str, ...]]:
    """The counted noun of a "how many" ask with its modifiers: ("hour", ("credit",)) for "how many credit hours"."""
    m = _HOW_MANY.search(clause)
    if not m:
        return None, ()
    words = [w for w in _tokens(m.group(1)) if w not in _DETERMINERS]
    kept: List[str] = []
    for w in words:
        if w in _SPLIT or (w in _VERBS and kept):
            break
        kept.append(w)
    if not kept:
        return None, ()
    return _stem(kept[-1]), tuple(kept[:-1])


def read_request(clause: str) -> RequestShape:
    """Read the clause into a :class:`RequestShape`. Deterministic; a clause outside every pattern is ``UNRECOGNISED``."""
    text = (clause or "").strip()
    low = text.lower()
    entities = tuple(named_entities(text))
    phrases, conditions, actor, negated = _phrases_of(text)
    base = dict(entities=entities, phrases=tuple(phrases), conditions=tuple(conditions), negated=negated, actor=actor)
    for field_name, pattern in _CONTACT_FIELDS:
        if pattern.search(text):
            return RequestShape(RequestKind.CONTACT, field=field_name, **base)
    if _LOCATION_ASK.search(text):
        return RequestShape(RequestKind.LOCATION, **base)
    if _PERCENT_ASK.search(text):
        return RequestShape(RequestKind.PERCENTAGE, **base)
    if _HOW_MANY.search(text):
        unit, mods = _unit_of(text)
        return RequestShape(RequestKind.QUANTITY, unit=unit, unit_modifiers=mods, **base)
    if _HOW_LONG.search(text):
        return RequestShape(RequestKind.DURATION, **base)
    if _HOW_OFTEN.search(text):
        return RequestShape(RequestKind.FREQUENCY, **base)
    if _HOW_MUCH.search(text):
        return RequestShape(RequestKind.AMOUNT, **base)
    if _WHEN.search(text):
        return RequestShape(RequestKind.TIMING, **base)
    if _ELIGIBLE_ASK.search(text):
        return RequestShape(RequestKind.ELIGIBILITY, **base)
    who = _WHO.match(text)
    if who:
        body = (who.group(1) or who.group(2) or "").lower() or None
        verb = _party_verb(text)
        kind = RequestKind.RESPONSIBILITY if _RESPONSIBLE_VERB.search(text) and not (verb and verb in {"draft", "determin", "approv", "review", "sign", "chair", "own", "decid"}) else RequestKind.AUTHORITY
        if body and verb and kind is RequestKind.AUTHORITY:
            kind = RequestKind.SELECTION
        return RequestShape(kind, verb=verb, body_noun=body, **base)
    if _OBLIGATION_ASK.search(text):
        return RequestShape(RequestKind.OBLIGATION, verb=_main_verb(text), **base)
    if _YESNO.match(text):
        return RequestShape(RequestKind.YES_NO, verb=_main_verb(text), **base)
    if _REQUIREMENT_ASK.search(text) or _FUNCTION_ASK.search(text) or _OPEN_OBJECT_ASK.search(text):
        return RequestShape(RequestKind.ENUMERATION, verb=_main_verb(text), **base)
    wh = _WH_OBJECT_ASK.match(text)
    if wh:
        obj_words = [w for w in _tokens(wh.group(1)) if w not in _DETERMINERS]
        head = obj_words[-1] if obj_words else ""
        if head in _PLURAL_SET_HEADS or (head.endswith("s") and not head.endswith("ss") and head not in ("information", "status", "basis")):
            return RequestShape(RequestKind.ENUMERATION, verb=_main_verb(text), object_head=_stem(head), **base)
        return RequestShape(RequestKind.SELECTION, verb=_main_verb(text), object_head=_stem(head) if head else None, **base)
    if _POLICY_ASK.search(text):
        return RequestShape(RequestKind.GENERAL, **base)
    what = _WHAT_IS.match(text)
    if what:
        np_words = [w for w in _tokens(what.group(1)) if w not in _DETERMINERS]
        head_raw = np_words[0] if np_words else ""
        for w in np_words:                                           # the head of "the teaching load for lecturers" is "load"
            if w in _SPLIT:
                break
            head_raw = w
        if head_raw in _MEASURE_HEADS or _stem(head_raw) in {_stem(h) for h in _MEASURE_HEADS}:
            return RequestShape(RequestKind.VALUE, **base)
        if head_raw in _PLURAL_SET_HEADS or (head_raw.endswith("s") and not head_raw.endswith("ss") and len(np_words) <= 3
                                             and not entities):
            return RequestShape(RequestKind.ENUMERATION, **base)
        return RequestShape(RequestKind.DEFINITION, **base)
    return RequestShape(RequestKind.UNRECOGNISED, **base)


_PARTY_HEAD = re.compile(r"^\s*(?:who|whom|which\s+\w+|what\s+\w+)\s+(?:(?:is|are|do|does|did|can|could|should|must|will|would|may|shall)\s+)?"
                         r"(?:(?:i|we|the|a|an)\s+)?([a-z][a-z'’-]*)", re.I)


def _party_verb(text: str) -> Optional[str]:
    """The asked action of a party ask: the word that follows the interrogative (and its auxiliary), when it is a verb
    form. Nothing later in the clause is ever taken for it ("research grants" names no action)."""
    m = _PARTY_HEAD.match(text)
    if not m:
        return None
    cand = m.group(1).lower()
    if cand in _VERBS or (len(cand) > 3 and cand.endswith(("s", "ed", "ing")) and cand not in _STOP and cand not in _DETERMINERS
                          and cand not in _GENERIC):
        return _stem(cand)
    return None


def _main_verb(text: str) -> Optional[str]:
    """The asked action of a yes/no, obligation or selection clause: the first lexical verb after the frame."""
    toks = _tokens(text)
    for i, tok in enumerate(toks):
        if i == 0 and tok in _SPLIT:
            continue
        if tok in _VERBS and tok not in _SPLIT and tok not in ("responsible", "liable", "mandatory", "compulsory", "eligible"):
            if tok in ("do", "does", "did") and i + 1 < len(toks):
                continue
            return _stem(tok)
    for tok in toks:
        if tok in ("responsible", "liable", "mandatory", "compulsory", "eligible", "entitled", "allowed", "permitted", "required"):
            return _stem(tok)
    return None


# ---------------------------------------------------------------------------
# Units of a quote (sentences inside provisions)
# ---------------------------------------------------------------------------
def _provision_spans(quote: str) -> List[Tuple[int, int]]:
    starts = [0]
    for m in _PROVISION_START.finditer(quote):
        if m.start() > 0 and m.start() not in starts:
            starts.append(m.start())
    starts = sorted(set(starts))
    return [(s, starts[i + 1] if i + 1 < len(starts) else len(quote)) for i, s in enumerate(starts)]


def _sentences(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        before = text[start:m.start()].rstrip()
        last = before.rsplit(None, 1)[-1].lower() if before else ""
        if last in _ABBREVIATIONS or re.fullmatch(r"\(?[a-z0-9]{1,2}\.", last):
            continue                                                 # "No. (3)", "e.g.", "12. " are not sentence ends
        parts.append(text[start:m.start()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def quote_units(quote: str) -> List[Tuple[str, int]]:
    """The quotable units of a quote as ``(text, provision_index)``: a provision is a list item, a numbered or lettered
    item or a dotted-section passage; a unit is one sentence of it."""
    units: List[Tuple[str, int]] = []
    for p, (s, e) in enumerate(_provision_spans(quote)):
        for sentence in _sentences(quote[s:e]):
            units.append((sentence, p))
    return units or [(quote.strip(), 0)]


def _is_lead_in(text: str) -> bool:
    return text.rstrip().endswith(":")


# ---------------------------------------------------------------------------
# Checks on one evidence unit
# ---------------------------------------------------------------------------
def _value_in_unit(unit: str, shape: RequestShape) -> Optional[float]:
    """The asked value stated in ``unit``, or None: a number tied to the asked unit noun (quantity), a measured number
    (value, duration), a money amount or a percentage."""
    toks = _tokens(unit)
    stems = [_stem(t) for t in toks]
    kind = shape.kind
    if kind is RequestKind.AMOUNT:
        m = _MONEY.search(unit)
        if not m:
            return None
        digits = re.search(r"\d[\d,]*(?:\.\d+)?", m.group(0))
        return float(digits.group(0).replace(",", "")) if digits else 0.0
    if kind is RequestKind.PERCENTAGE:
        m = _PERCENT.search(unit)
        return float(re.match(r"\d+(?:\.\d+)?", m.group(0)).group(0)) if m else None
    wanted = {shape.unit} if (kind is RequestKind.QUANTITY and shape.unit) else None
    for i, tok in enumerate(toks):
        value = _number_value(tok)
        if value is None:
            continue
        window = stems[i + 1:i + 4]
        if wanted is not None:
            if any(w in wanted for w in window):
                return value
            continue
        if kind is RequestKind.DURATION and any(w in _TIME_UNITS for w in window):
            return value
        if kind is RequestKind.VALUE and (any(w in _MEASURE_UNITS for w in window) or (i > 0 and stems[i - 1] in ("aed", "dhs", "usd"))):
            return value
    if kind is RequestKind.VALUE:
        m = _MONEY.search(unit) or _PERCENT.search(unit)
        if m:
            digits = re.search(r"\d[\d,]*(?:\.\d+)?", m.group(0))
            return float(digits.group(0).replace(",", "")) if digits else 0.0
    return None


def _premodifiers(head: str, toks: Sequence[str], stems: Sequence[str]) -> List[str]:
    """The words that modify each occurrence of ``head`` in the unit: the contiguous run before it that holds no
    stopword, number or verb ("sixty calendar days of maternity leave" -> ["maternity"] for "leave")."""
    out: List[str] = []
    for pos, s in enumerate(stems):
        if s != head:
            continue
        j = pos - 1
        while j >= 0 and toks[j] not in _STOP and _number_value(toks[j]) is None and toks[j] not in _VERBS:
            out.append(toks[j])
            j -= 1
    return out


def _modifier_conflict(unit: str, shape: RequestShape) -> Optional[str]:
    """An asked category modifier that the unit contradicts, or None. The modifier is absent from the unit while a
    contrasting member of its closed set modifies the SAME noun ("annual leave" asked, "maternity leave" stated;
    "credit hours" asked, "office hours" stated; "calendar days" asked, "working days" stated). For a population
    noun the category is often stated predicatively ("Female faculty employed on a full-time basis"), so a
    contrasting category anywhere in the unit counts ("part-time faculty" asked). A modifier outside every set, or
    a noun the unit does not use, is never a conflict: the I-1 coverage rule decides those."""
    toks = _tokens(unit)
    stems = [_stem(t) for t in toks]
    present = set(toks) | set(stems)
    checks: List[Tuple[Tuple[str, ...], str]] = list(shape.phrases)
    if shape.unit:
        checks.append((shape.unit_modifiers, shape.unit))
    for mods, head in checks:
        for m in mods:
            sets = set(_contrast_sets(m))
            if not sets or m in present or _stem(m) in present:
                continue
            candidates = toks if head in _POPULATION_STEMS else _premodifiers(head, toks, stems)
            for w in candidates:
                if (w != m and _stem(w) != _stem(m)) and sets & set(_contrast_sets(w)):
                    return m
    return None


_POPULATION_STEMS = frozenset(_stem(w) for w in _POPULATION_HEADS | {"investigator", "assistant", "supervisor", "applicant", "researcher",
                                                                       "director", "coordinator", "chairperson", "registrar", "provost"})


def _population_missing(unit: str, shape: RequestShape) -> Optional[str]:
    """The asked population or role (the head of a clause phrase) that the unit does not name, or None."""
    stems = set(_stems(unit))
    for _mods, head in shape.phrases:
        if head in _POPULATION_STEMS and head not in stems:
            return head
    return None


def _condition_missing(unit: str, shape: RequestShape) -> Optional[Tuple[str, ...]]:
    """The first condition phrase of the clause not stated in the unit, or None."""
    stems = _stems(unit)
    joined = " " + " ".join(stems) + " "
    for cond in shape.conditions:
        words = tuple(w for w in cond if w not in _STOP)
        alternatives = _IDIOMS.get(cond) or _IDIOMS.get(tuple(_stem(w) for w in cond))
        if alternatives:
            if any((" " + " ".join(_stem(w) for w in alt) + " ") in joined or all(_stem(w) in stems for w in alt) for alt in alternatives):
                continue
            return cond
        if not words:
            continue
        if cond and cond[0] == "per" or (len(words) == 1 and words[0] in _PERIODS):
            period = words[-1]
            if re.search(r"\b(?:per|each|every|a|an)\s+" + re.escape(period), " ".join(stems)) or (period + "ly") in stems or period in stems:
                continue
            return cond
        if all(w in stems for w in words):
            continue
        return cond
    return None


def _asked_offices(clause: str) -> Tuple[str, ...]:
    """The lexicon offices the clause names (the Institutional cues' own reader); empty when none or unreadable."""
    try:
        return tuple(str(o) for o in _nav.resolve_offices(clause))
    except Exception:
        return ()


def _entities_missing(unit: str, shape: RequestShape) -> Optional[str]:
    for entity in shape.entities:
        if not entity_in_text(entity, unit):
            return entity
    return None


def _party_positions(unit: str) -> List[int]:
    """Token indexes at which a named party is stated: a capitalised run (not the sentence opener alone), an acronym
    or a role word."""
    raw = _TOKEN.findall(unit)
    positions: List[int] = []
    for i, tok in enumerate(raw):
        low = tok.lower()
        if low in _ROLE_WORDS and low not in ("university", "faculty", "member", "members", "staff"):
            positions.append(i)
        elif re.fullmatch(r"[A-Z][A-Z&]{1,6}s?", tok):
            positions.append(i)                                      # an acronym names a party wherever it stands
        elif tok[:1].isupper() and i > 0 and low not in _STOP and low not in _DETERMINERS:
            positions.append(i)                                      # a capitalised word inside the sentence (the opener alone is not a name)
    return positions


def _verb_positions(unit: str, verb: str) -> List[int]:
    stems = _stems(unit)
    return [i for i, s in enumerate(stems) if s == verb or s.startswith(verb) and len(verb) >= 4]


def _relation_stated(unit: str, shape: RequestShape) -> Tuple[Optional[bool], str]:
    """Whether the unit states the asked relation. (None, reason) when the kind carries no relation to check."""
    kind = shape.kind
    toks = _tokens(unit)
    stems = [_stem(t) for t in toks]
    if kind is RequestKind.DEFINITION:
        subjects: List[str] = list(shape.entities)
        if shape.phrases:
            mods, head = shape.phrases[0]
            subjects.append(" ".join(list(mods) + [head]))
        for subject in subjects:
            words = [re.escape(w) for w in _tokens(subject)]
            if not words:
                continue
            pattern = re.compile(r"\b" + r"\s+".join(w + r"[a-z’']*" for w in words), re.I)
            for m in pattern.finditer(unit):
                rest = unit[m.end():]
                rest = _PAREN_ACRONYM.sub("", rest, count=1)
                if _COPULA.match(rest):
                    return True, "definition stated"
            acronyms = [e for e in shape.entities if re.fullmatch(r"[A-Z][A-Z&]{1,6}", e)]
            for acr in acronyms:
                for m in re.finditer(r"\b" + re.escape(acr) + r"\b", unit):
                    if _COPULA.match(unit[m.end():]):
                        return True, "definition stated"
        return False, "the asked subject is not the subject of a defining statement in the answering unit"
    if kind in (RequestKind.AUTHORITY, RequestKind.RESPONSIBILITY) or (kind is RequestKind.SELECTION and shape.body_noun):
        parties = _party_positions(unit)
        if not parties:
            return False, "no named party in the answering unit"
        if shape.body_noun and shape.kind is not RequestKind.RESPONSIBILITY:
            if _stem(shape.body_noun) not in stems:
                return False, "the asked kind of body is not named in the answering unit"
        if kind is RequestKind.RESPONSIBILITY and not shape.verb:
            if not _RESPONSIBILITY.search(unit):
                return False, "no responsibility statement in the answering unit"
            return True, "responsibility stated by a named party"
        if kind is RequestKind.RESPONSIBILITY and shape.verb and _RESPONSIBILITY.search(unit):
            return True, "responsibility stated by a named party"
        verb = shape.verb
        if not verb:
            return None, "no asked action read"
        verbs = _verb_positions(unit, verb)
        if not verbs:
            nominal = verb[:5]
            for n, s in enumerate(stems):
                if s.startswith(nominal):                          # "the Dean's approval", "approval of the Department Council"
                    if any(n - 2 <= p < n for p in parties) or (n + 1 < len(toks) and toks[n + 1] in ("of", "by", "from")
                                                                 and any(n + 1 < p <= n + 4 for p in parties)):
                        return True, "the asked action is stated as a noun bound to a party"
            return False, "the asked action is not stated in the answering unit"
        for v in verbs:
            if any(p < v and v - p <= 8 for p in parties):
                return True, "a named party precedes the asked action"
            after = toks[v + 1:v + 3]
            if "by" in after:
                by_index = v + 1 + after.index("by")
                if any(by_index < p <= by_index + 4 for p in parties):
                    return True, "the asked action is stated with its agent"
        return False, "the asked action is stated, but not by a named party in the asked position"
    if kind is RequestKind.OBLIGATION:
        if not _NORMATIVE.search(unit):
            return False, "no requirement statement in the answering unit"
        if shape.actor and shape.actor not in stems and not any(_stem(e.split()[-1]) in stems for e in shape.entities):
            return False, "the asked actor is not named in the requirement statement"
        return True, "requirement stated for the asked actor"
    if kind is RequestKind.ELIGIBILITY:
        if not _ELIGIBLE.search(unit):
            return False, "no eligibility statement in the answering unit"
        return True, "eligibility stated"
    if kind is RequestKind.YES_NO or kind is RequestKind.SELECTION:
        verb = shape.verb
        if not verb:
            if kind is RequestKind.YES_NO:                         # phase I-8: a copular or modal yes/no ask with no readable action
                if _YES_NO_PREDICATE.search(unit) or _NORMATIVE.search(unit):
                    return True, "a permission, prohibition, entitlement or requirement about the asked subject answers the yes/no ask"
                return False, "no asked action read: a value statement cannot confirm or refute a yes/no ask"
            return None, "no asked action read"
        verbs = _verb_positions(unit, verb)
        if not verbs:
            nominal = verb[:5]
            if any(s.startswith(nominal) for s in stems):
                return True, "the asked action is stated as a noun"
            return False, "the asked action is not stated in the answering unit"
        if shape.object_head and shape.object_head not in stems:
            return False, "the asked object is not stated with the action"
        if not shape.negated:
            negated_all = True
            for v in verbs:
                before = toks[max(0, v - 3):v]
                if not any(w in _NEGATION for w in before):
                    negated_all = False
                    break
            if negated_all:
                return False, "the answering unit negates the asked predicate"
        return True, "the asked predicate is stated with the asked polarity"
    return None, "no relation check for this request kind"


def _detail_present(unit: str, shape: RequestShape, finding: Finding) -> Tuple[Optional[bool], str, Optional[float]]:
    """(present, reason, value) for the requested detail of the request kind in ``unit``."""
    kind = shape.kind
    meta = finding.metadata if isinstance(finding.metadata, dict) else {}
    if kind is RequestKind.CONTACT:
        values = meta.get("contact_values") if isinstance(meta.get("contact_values"), dict) else {}
        value = values.get(shape.field)
        if isinstance(value, str) and value and value in unit:
            return True, "the asked contact field is bound in the directory row", None
        if value is None and shape.field in (_nav.TELEPHONE, _nav.FAX) and not values:
            return False, "no contact value bound for the asked field", None
        return False, "the asked contact field (%s) is not the bound field" % shape.field, None
    if kind is RequestKind.LOCATION:
        if meta.get("relation") == "location" and (str(meta.get("place") or "") in unit or str(meta.get("building_code") or "") in unit):
            return True, "location entry bound", None
        if _LOCATION_WORDS.search(unit):
            return True, "location stated", None
        return False, "no location stated in the answering unit", None
    if kind in VALUE_KINDS:
        value = _value_in_unit(unit, shape)
        if value is None:
            what = {"quantity": "a number of the asked unit", "amount": "a money amount", "percentage": "a percentage",
                    "duration": "a duration", "value": "a measured value"}[kind.value]
            return False, "%s is not stated in the answering unit" % what, None
        return True, "the asked value is stated", value
    if kind is RequestKind.FREQUENCY:
        return (True, "frequency stated", None) if _FREQUENCY.search(unit) else (False, "no frequency stated in the answering unit", None)
    if kind is RequestKind.TIMING:
        return (True, "timing stated", None) if _TIMING.search(unit) else (False, "no timing stated in the answering unit", None)
    return None, "no requested-detail check for this request kind", None


# ---------------------------------------------------------------------------
# Status and structure
# ---------------------------------------------------------------------------
def run_code(findings: SpecialistFindings) -> Tuple[Optional[ReasonCode], str]:
    """Why the result AS A WHOLE is non-renderable (its status, a handoff request, a ``missing`` entry, no finding, a
    model call, no clause report), as a reason code, or (None, "") when its clause reports may be examined. Shared
    by :func:`status_code` (one clause) and the phase I-5 aggregator (one report per request clause)."""
    status = findings.status
    if status is FindingStatus.PARTIAL:
        return ReasonCode.PARTIAL_FINDING, "status partial"
    if status is FindingStatus.NOT_FOUND:
        return ReasonCode.NOT_FOUND, "status not_found"
    if status is FindingStatus.OUT_OF_SCOPE:
        return ReasonCode.OUT_OF_SCOPE, "status out_of_scope"
    if status is not FindingStatus.SUPPORTED:
        return ReasonCode.UNSUPPORTED_STATUS, "status %s" % status.value
    if findings.handoff_requests:
        return ReasonCode.HANDOFF_REQUIRED, "handoff pending"
    if findings.missing:
        return ReasonCode.PARTIAL_FINDING, "result carries missing entries"
    if not findings.findings:
        return ReasonCode.UNSUPPORTED_STATUS, "supported result without findings"
    if findings.llm_used:
        return ReasonCode.UNSUPPORTED_STATUS, "specialist reported an LLM call"
    md = findings.metadata if isinstance(findings.metadata, dict) else {}
    clauses = md.get("clauses")
    if not isinstance(clauses, list) or not clauses or not all(isinstance(c, dict) for c in clauses):
        return ReasonCode.UNSUPPORTED_STATUS, "no clause report"
    return None, ""


def report_code(specialist_id: SpecialistId, report: Dict[str, Any]) -> Tuple[Optional[ReasonCode], str]:
    """Why ONE clause report makes its clause non-renderable (the outcome, the ask shape, the requested details and
    the relation of a retrieval specialist; the outcome and evidence tier of Institutional Navigation), as a reason
    code, or (None, "")."""
    if specialist_id is SpecialistId.INSTITUTIONAL:
        if report.get("outcome") != INSTITUTIONAL_OUTCOME_READ:
            return ReasonCode.UNSUPPORTED_STATUS, "institutional clause outcome %s" % report.get("outcome")
        if report.get("evidence_tier_reached") not in INSTITUTIONAL_TIERS:
            return ReasonCode.UNSUPPORTED_STATUS, "institutional evidence tier %s" % report.get("evidence_tier_reached")
        return None, ""
    if report.get("outcome") != FindingStatus.SUPPORTED.value:
        return ReasonCode.UNSUPPORTED_STATUS, "clause outcome %s" % report.get("outcome")
    shape = report.get("ask_shape")
    if shape == "general":
        return ReasonCode.GENERAL_ASK_MISMATCH, "general ask: topical support is not an answer"
    if shape != ASK_SPECIFIC_RECOGNISED:
        return ReasonCode.UNRECOGNISED_ASK, "ask shape %s" % shape
    if not report.get("requested_details"):
        return ReasonCode.REQUESTED_DETAIL_MISSING, "specific ask without requested details"
    if report.get("uncovered_details"):
        return ReasonCode.REQUESTED_DETAIL_MISSING, "uncovered details"
    if "relation" in report and report.get("relation") != RELATION_STRONG:
        return ReasonCode.RELATION_MISMATCH, "relation %s" % report.get("relation")
    return None, ""


def status_code(findings: SpecialistFindings) -> Tuple[Optional[ReasonCode], str]:
    """Why the result's status or clause report makes it non-renderable, as a reason code, or (None, "") when it may
    proceed to the finding-level checks. The same conditions as the I-1 ``result_problem`` gate, as codes:
    :func:`run_code`, then exactly one processed clause report, then :func:`report_code` on it."""
    code, detail = run_code(findings)
    if code is not None:
        return code, detail
    md = findings.metadata if isinstance(findings.metadata, dict) else {}
    clauses = md["clauses"]
    if len(clauses) != 1:
        return ReasonCode.MULTIPART_UNSUPPORTED, "%d clause reports" % len(clauses)
    if md.get("unprocessed_clauses"):
        return ReasonCode.MULTIPART_UNSUPPORTED, "unprocessed clauses"
    return report_code(findings.specialist_id, clauses[0])


# ---------------------------------------------------------------------------
# The verification result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FindingVerdict:
    finding_id: str
    accepted: bool
    reason: ReasonCode
    detail: str
    completeness: Completeness
    value: Optional[float] = None
    unit_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"finding_id": self.finding_id, "accepted": self.accepted, "reason": self.reason.value, "detail": self.detail[:160],
                "completeness": self.completeness.value, "value": self.value, "unit_index": self.unit_index}


@dataclass(frozen=True)
class FindingsVerification:
    """The deterministic verdict on one specialist result. ``safe_to_render`` is True only when ``accepted`` is True and
    at least one finding may be rendered."""

    accepted: bool
    reason: ReasonCode
    detail: str
    status: Optional[str]
    specialist_id: Optional[str]
    request_kind: RequestKind
    subject_match: Optional[bool]
    relation_match: Optional[bool]
    detail_coverage: Optional[bool]
    evidence_integrity: Optional[bool]
    completeness: Completeness
    safe_to_render: bool
    accepted_finding_ids: Tuple[str, ...]
    verdicts: Tuple[FindingVerdict, ...]
    ms: float
    version: str = VERSION

    @property
    def dropped(self) -> Tuple[Tuple[str, str], ...]:
        return tuple((v.finding_id, v.reason.value) for v in self.verdicts if not v.accepted)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable and deterministic for equal inputs (the wall time stays on the ``ms`` attribute, read by the
        pipeline diagnostics); carries no question or answer text."""
        return {
            "accepted": self.accepted, "reason": self.reason.value, "detail": self.detail[:200], "status": self.status,
            "specialist_id": self.specialist_id, "request_kind": self.request_kind.value, "subject_match": self.subject_match,
            "relation_match": self.relation_match, "detail_coverage": self.detail_coverage, "evidence_integrity": self.evidence_integrity,
            "completeness": self.completeness.value, "safe_to_render": self.safe_to_render,
            "accepted_finding_ids": list(self.accepted_finding_ids), "dropped": [list(d) for d in self.dropped],
            "verdicts": [v.to_dict() for v in self.verdicts], "version": self.version,
        }


def _rejected(reason: ReasonCode, detail: str, *, status: Optional[str], specialist: Optional[str], kind: RequestKind, started: float,
              subject: Optional[bool] = None, relation: Optional[bool] = None, coverage: Optional[bool] = None, integrity: Optional[bool] = None,
              completeness: Completeness = Completeness.NOT_ASSESSED, verdicts: Sequence[FindingVerdict] = ()) -> FindingsVerification:
    return FindingsVerification(accepted=False, reason=reason, detail=detail[:200], status=status, specialist_id=specialist, request_kind=kind,
                                subject_match=subject, relation_match=relation, detail_coverage=coverage, evidence_integrity=integrity,
                                completeness=completeness, safe_to_render=False, accepted_finding_ids=(), verdicts=tuple(verdicts),
                                ms=(time.perf_counter() - started) * 1000.0)


# ---------------------------------------------------------------------------
# One finding
# ---------------------------------------------------------------------------
def verify_finding(clause: str, shape: RequestShape, findings: SpecialistFindings, finding: Finding, chunks: Sequence[str],
                   metadata: Sequence[Dict[str, Any]]) -> Tuple[FindingVerdict, Dict[str, Optional[bool]]]:
    """The verdict on one finding for the request, with the per-dimension flags it established."""
    flags: Dict[str, Optional[bool]] = {"integrity": None, "subject": None, "detail": None, "relation": None}
    fid = finding.finding_id
    problem = quote_problem(finding, chunks, metadata)
    flags["integrity"] = problem is None
    if problem is not None:
        return FindingVerdict(fid, False, ReasonCode.INVALID_EVIDENCE_REFERENCE, problem, Completeness.NOT_ASSESSED), flags
    report = findings.metadata["clauses"][0]
    item = adapt_finding(finding, findings.specialist_id, clause, chunks, metadata)
    bound, why = subject_binding(clause, report, finding, item, findings.specialist_id)
    flags["subject"] = bool(bound)
    if not bound:
        code = ReasonCode.REQUESTED_DETAIL_MISSING if "requested detail" in why else ReasonCode.SUBJECT_MISMATCH
        return FindingVerdict(fid, False, code, why, Completeness.NOT_ASSESSED), flags
    if shape.kind is RequestKind.GENERAL:
        return FindingVerdict(fid, False, ReasonCode.GENERAL_ASK_MISMATCH, "a general ask is not answered by one statement", Completeness.NOT_ASSESSED), flags
    if shape.kind in ENUMERATIVE_KINDS:
        return FindingVerdict(fid, False, ReasonCode.ENUMERATION_INCOMPLETE, "an enumerative ask cannot be proven complete by one finding",
                              Completeness.ENUMERATION_UNPROVEN), flags
    if shape.kind is RequestKind.UNRECOGNISED:
        units = quote_units(finding.evidence_quote)
        provisions = {p for _t, p in units}
        if len(provisions) > 1:
            return FindingVerdict(fid, False, ReasonCode.EXCESS_EVIDENCE, "the quote spans %d provisions" % len(provisions), Completeness.EXCESS), flags
        return FindingVerdict(fid, True, ReasonCode.ACCEPTED, "request kind not read by the verifier; the specialist's contract and the I-1 guards decided",
                              Completeness.CONTRACT), flags

    units = quote_units(finding.evidence_quote)
    institutional = findings.specialist_id is SpecialistId.INSTITUTIONAL
    if institutional:
        # independent of the clause report: the finding's office must be one the clause itself names (lexicon read)
        asked_offices = _asked_offices(clause)
        office_id = finding.metadata.get("office_id")
        if asked_offices and office_id != _nav.UNIVERSITY and office_id not in asked_offices:
            flags["subject"] = False
            return FindingVerdict(fid, False, ReasonCode.SUBJECT_MISMATCH, "the finding belongs to an office the clause does not name",
                                  Completeness.NOT_ASSESSED), flags
    failures: List[Tuple[ReasonCode, str]] = []
    for index, (text, provision) in enumerate(units):
        present, why_detail, value = _detail_present(text, shape, finding)
        if present is False:
            failures.append((ReasonCode.REQUESTED_DETAIL_MISSING, why_detail))
            continue
        if not institutional:
            missing_entity = _entities_missing(text, shape)
            if missing_entity is not None:
                failures.append((ReasonCode.SUBJECT_MISMATCH, "a named entity of the clause is not in the answering unit"))
                continue
            missing_population = _population_missing(text, shape)
            if missing_population is not None:
                failures.append((ReasonCode.SUBJECT_MISMATCH, "the asked population or role is not named in the answering unit"))
                continue
        conflict = _modifier_conflict(text, shape)
        if conflict is not None:
            failures.append((ReasonCode.MODIFIER_MISMATCH, "the asked modifier %r is contradicted on the same noun" % conflict))
            continue
        cond = _condition_missing(text, shape)
        if cond is not None:
            failures.append((ReasonCode.CONDITION_MISMATCH, "the asked condition %r is not stated with the answer" % " ".join(cond)))
            continue
        stated, why_rel = _relation_stated(text, shape)
        if stated is False:
            failures.append((ReasonCode.RELATION_MISMATCH, why_rel))
            continue
        flags["detail"] = present
        flags["relation"] = stated
        # ---- provision boundary: the quote must not span another provision ----------------------------------------
        earlier = [t for t, p in units if p < provision]
        later = [t for t, p in units if p > provision]
        if earlier and not all(_is_lead_in(t) for t in earlier):
            return FindingVerdict(fid, False, ReasonCode.EXCESS_EVIDENCE, "the quote opens with a sentence of another provision",
                                  Completeness.EXCESS, value, index), flags
        if later:
            return FindingVerdict(fid, False, ReasonCode.EXCESS_EVIDENCE, "the quote continues into another provision",
                                  Completeness.EXCESS, value, index), flags
        siblings = sum(1 for _t, p in units if p == provision) - 1
        completeness = Completeness.ELABORATED if siblings > 0 else Completeness.COMPLETE
        return FindingVerdict(fid, True, ReasonCode.ACCEPTED, why_rel if stated else why_detail, completeness, value, index), flags
    # no unit answers: report the most specific failure (detail missing < subject < modifier < condition < relation)
    order = [ReasonCode.RELATION_MISMATCH, ReasonCode.CONDITION_MISMATCH, ReasonCode.MODIFIER_MISMATCH, ReasonCode.SUBJECT_MISMATCH,
             ReasonCode.REQUESTED_DETAIL_MISSING]
    for code in order:
        for got, why in failures:
            if got is code:
                flags["detail"] = False if code is ReasonCode.REQUESTED_DETAIL_MISSING else flags["detail"]
                flags["relation"] = False if code is ReasonCode.RELATION_MISMATCH else flags["relation"]
                if code is ReasonCode.SUBJECT_MISMATCH:
                    flags["subject"] = False
                return FindingVerdict(fid, False, code, why, Completeness.NOT_ASSESSED), flags
    return FindingVerdict(fid, False, ReasonCode.REQUESTED_DETAIL_MISSING, "no unit of the quote answers the request", Completeness.NOT_ASSESSED), flags


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------
def verify_findings(clause: str, findings: SpecialistFindings, chunks: Sequence[str], metadata: Sequence[Dict[str, Any]]) -> FindingsVerification:
    """Verify one specialist result against the clause it answers. Deterministic; no model; never raises."""
    started = time.perf_counter()
    kind = RequestKind.UNRECOGNISED
    status: Optional[str] = None
    specialist: Optional[str] = None
    try:
        if not isinstance(findings, SpecialistFindings):
            return _rejected(ReasonCode.UNSUPPORTED_STATUS, "not a SpecialistFindings", status=None, specialist=None, kind=kind, started=started)
        status, specialist = findings.status.value, findings.specialist_id.value
        code, detail = status_code(findings)
        if code is not None:
            completeness = Completeness.PARTIAL if code is ReasonCode.PARTIAL_FINDING else Completeness.NOT_ASSESSED
            return _rejected(code, detail, status=status, specialist=specialist, kind=kind, started=started, completeness=completeness)
        text = (clause or "").strip()
        if not text:
            return _rejected(ReasonCode.UNSUPPORTED_STATUS, "empty clause", status=status, specialist=specialist, kind=kind, started=started)
        shape = read_request(text)
        kind = shape.kind
        verdicts: List[FindingVerdict] = []
        flags_all: List[Dict[str, Optional[bool]]] = []
        for finding in findings.findings:
            verdict, flags = verify_finding(text, shape, findings, finding, chunks, metadata)
            verdicts.append(verdict)
            flags_all.append(flags)
        primary, primary_flags = verdicts[0], flags_all[0]
        if not primary.accepted:
            return _rejected(primary.reason, primary.detail, status=status, specialist=specialist, kind=kind, started=started,
                             subject=primary_flags["subject"], relation=primary_flags["relation"], coverage=primary_flags["detail"],
                             integrity=primary_flags["integrity"], completeness=primary.completeness, verdicts=verdicts)
        # ---- secondary findings: must pass on their own and must not state another value for the asked quantity -------
        kept: List[FindingVerdict] = [primary]
        final: List[FindingVerdict] = [primary]
        for verdict in verdicts[1:]:
            if verdict.accepted and kind in VALUE_KINDS and primary.value is not None and verdict.value is not None and verdict.value != primary.value:
                verdict = FindingVerdict(verdict.finding_id, False, ReasonCode.VALUE_CONFLICT,
                                         "states another value for the asked quantity than the primary finding", Completeness.NOT_ASSESSED,
                                         verdict.value, verdict.unit_index)
            final.append(verdict)
            if verdict.accepted:
                kept.append(verdict)
        completeness = primary.completeness
        return FindingsVerification(
            accepted=True, reason=ReasonCode.ACCEPTED, detail=primary.detail[:200], status=status, specialist_id=specialist, request_kind=kind,
            subject_match=True, relation_match=primary_flags["relation"], detail_coverage=primary_flags["detail"],
            evidence_integrity=True, completeness=completeness, safe_to_render=bool(kept),
            accepted_finding_ids=tuple(v.finding_id for v in kept), verdicts=tuple(final), ms=(time.perf_counter() - started) * 1000.0)
    except Exception as exc:                                        # a defect here must never become an answer
        category = redact_error(exc).split(":", 1)[0].strip() or type(exc).__name__      # the exception category only, never its text
        return _rejected(ReasonCode.VERIFICATION_ERROR, "verification raised: %s" % category, status=status, specialist=specialist,
                         kind=kind, started=started)


# ---------------------------------------------------------------------------
# Phase I-5 helpers: one clause report of a multi-clause result, and the provision slice of an excess quote
# ---------------------------------------------------------------------------
#: Suffix of the id of a finding derived by :func:`provision_slice`.
SLICE_SUFFIX = "/slice"


def clause_view(findings: SpecialistFindings, index: int, finding_ids: Optional[Sequence[str]] = None) -> SpecialistFindings:
    """The result restricted to ONE of its clause reports: the report at ``index`` and, by default, the findings whose
    ``metadata["clauses"]`` name that report's index (or exactly ``finding_ids``, kept in the result's own order).
    Status, ``missing``, handoff requests and the model flag are carried over unchanged, so :func:`verify_findings`
    applies the same run-level rules to the view as to the result; only the single-report rule is replaced by the
    aggregator's per-clause coverage. Raises ``ValueError`` when the report does not exist or no finding answers it."""
    md = findings.metadata if isinstance(findings.metadata, dict) else {}
    reports = md.get("clauses")
    if (not isinstance(reports, list) or not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(reports)
            or not isinstance(reports[index], dict)):
        raise ValueError("result %s has no clause report %r" % (findings.task_id, index))
    report = reports[index]
    report_index = report.get("index", index)
    if finding_ids is None:
        chosen = [f for f in findings.findings if report_index in (f.metadata.get("clauses") or [])]
    else:
        wanted = set(finding_ids)
        chosen = [f for f in findings.findings if f.finding_id in wanted]
    if not chosen:
        raise ValueError("no finding of %s answers clause report %d" % (findings.task_id, index))
    chosen = [_project_shared_coverage(f, report) for f in chosen]
    metadata = {"clauses": [report], "unprocessed_clauses": [], "view_of": findings.task_id, "view_index": index}
    return SpecialistFindings(specialist_id=findings.specialist_id, task_id=findings.task_id, status=findings.status, findings=chosen,
                              summary=findings.summary, missing=list(findings.missing), handoff_requests=list(findings.handoff_requests),
                              confidence=findings.confidence, limitations=list(findings.limitations), llm_used=findings.llm_used,
                              ms=findings.ms, metadata=metadata)


def _project_shared_coverage(finding: Finding, report: Dict[str, Any]) -> Finding:
    """A finding the specialist attached to several clause reports keeps the ``details_covered`` of the clause it was first
    found for. When ``report`` (another of its clauses) is ``supported`` with no uncovered detail and lists the finding
    FIRST, the specialist's own report asserts that this finding covers every requested detail of that clause (the three
    retrieval specialists order a clause's findings by coverage and compute the clause's uncovered details from the
    best one). The clause view carries that coverage, marked ``projected_coverage``; the original finding is untouched
    and every text check of the verifier still applies to the view (phase I-6)."""
    md = finding.metadata if isinstance(finding.metadata, dict) else {}
    shared = isinstance(md.get("clauses"), list) and len(md["clauses"]) > 1
    listed = report.get("findings") if isinstance(report.get("findings"), list) else []
    requested = report.get("requested_details") if isinstance(report.get("requested_details"), list) else None
    if not (shared and requested and report.get("outcome") == FindingStatus.SUPPORTED.value and not report.get("uncovered_details")
            and listed and listed[0] == finding.finding_id):
        return finding
    covered = list(dict.fromkeys([d for d in (md.get("details_covered") or []) if isinstance(d, str)] + [d for d in requested if isinstance(d, str)]))
    if covered == list(md.get("details_covered") or []):
        return finding
    projected = dict(md, details_covered=covered, projected_coverage=True)
    return Finding(finding_id=finding.finding_id, claim=finding.claim, evidence_quote=finding.evidence_quote, source_id=finding.source_id,
                   source_title=finding.source_title, page=finding.page, confidence=finding.confidence, metadata=projected)


def provision_slice(finding: Finding, unit_index: int) -> Optional[Finding]:
    """A derived finding whose quote is the exact substring of ``finding.evidence_quote`` that holds the answering
    unit's own provision (every sentence of that list item, numbered item or section passage that the quote
    contains) together with the lead-in sentences immediately before it; sentences of OTHER provisions before or
    after it are left out. Nothing is reworded: the derived finding keeps the source, page, chunk and metadata of the
    original and records ``derived_from``, the ``slice`` offsets and the provision. None when the unit is unknown,
    the units cannot be located in the quote or nothing would be cut."""
    quote = finding.evidence_quote
    units = quote_units(quote)
    if not isinstance(unit_index, int) or isinstance(unit_index, bool) or not 0 <= unit_index < len(units):
        return None
    provision = units[unit_index][1]
    spans: List[Tuple[int, int]] = []
    position = 0
    for text, _p in units:
        start = quote.find(text, position)
        if start < 0:
            return None
        spans.append((start, start + len(text)))
        position = start + len(text)
    members = [i for i, (_t, p) in enumerate(units) if p == provision]
    first, last = members[0], members[-1]
    while first > 0 and _is_lead_in(units[first - 1][0]):
        first -= 1
    start, end = spans[first][0], spans[last][1]
    sliced = quote[start:end]
    if not sliced.strip() or sliced.strip() == quote.strip():
        return None
    metadata = dict(finding.metadata)
    metadata.update({"derived_from": finding.finding_id, "slice": [start, end], "slice_provision": provision})
    return Finding(finding_id=finding.finding_id + SLICE_SUFFIX, claim=sliced, evidence_quote=sliced, source_id=finding.source_id,
                   source_title=finding.source_title, page=finding.page, confidence=finding.confidence, metadata=metadata)


@dataclass(frozen=True)
class SlicedVerification:
    """A verification together with the result it applies to: the original result, or a copy whose primary finding is
    the provision slice of an excess quote. ``sliced`` lists (derived id, original id, start, end) for each slice."""

    verification: FindingsVerification
    findings: SpecialistFindings
    sliced: Tuple[Tuple[str, str, int, int], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"verification": self.verification.to_dict(), "sliced": [list(s) for s in self.sliced]}


def verify_findings_sliced(clause: str, findings: SpecialistFindings, chunks: Sequence[str],
                           metadata: Sequence[Dict[str, Any]]) -> SlicedVerification:
    """:func:`verify_findings` and, when its only objection to the primary finding is EXCESS_EVIDENCE, the same
    verification of a copy whose primary finding is the provision slice of that quote (:func:`provision_slice`). The
    slice is used only when it passes every check on its own, the quote guard included; otherwise the original
    rejection stands, with the slice's own reason appended to its detail. Deterministic; never raises."""
    first = verify_findings(clause, findings, chunks, metadata)
    try:
        if first.accepted or first.reason is not ReasonCode.EXCESS_EVIDENCE or not first.verdicts or not findings.findings:
            return SlicedVerification(first, findings)
        primary = first.verdicts[0]
        original = findings.findings[0]
        if primary.finding_id != original.finding_id or primary.unit_index is None:
            return SlicedVerification(first, findings)
        derived = provision_slice(original, primary.unit_index)
        if derived is None:
            return SlicedVerification(first, findings)
        replaced = SpecialistFindings(specialist_id=findings.specialist_id, task_id=findings.task_id, status=findings.status,
                                      findings=[derived] + list(findings.findings[1:]), summary=findings.summary,
                                      missing=list(findings.missing), handoff_requests=list(findings.handoff_requests),
                                      confidence=findings.confidence, limitations=list(findings.limitations), llm_used=findings.llm_used,
                                      ms=findings.ms, metadata=dict(findings.metadata))
        second = verify_findings(clause, replaced, chunks, metadata)
        if second.accepted and second.safe_to_render and derived.finding_id in second.accepted_finding_ids:
            start, end = derived.metadata["slice"]
            return SlicedVerification(second, replaced, ((derived.finding_id, original.finding_id, int(start), int(end)),))
        detail = ("%s; provision slice rejected: %s" % (first.detail, second.reason.value))[:200]
        return SlicedVerification(replace(first, detail=detail), findings)
    except Exception as exc:                                        # the slice is an extra; its failure never changes the verdict
        category = redact_error(exc).split(":", 1)[0].strip() or type(exc).__name__
        return SlicedVerification(replace(first, detail=("%s; provision slice raised: %s" % (first.detail, category))[:200]), findings)


__all__ = [
    "Completeness", "ENUMERATIVE_KINDS", "FindingVerdict", "FindingsVerification", "PHASE", "ReasonCode", "RequestKind", "RequestShape",
    "SLICE_SUFFIX", "SlicedVerification", "VALUE_KINDS", "VERSION", "clause_view", "provision_slice", "quote_units", "read_request",
    "report_code", "run_code", "status_code", "verify_finding", "verify_findings", "verify_findings_sliced",
]
