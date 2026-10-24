"""
Requested-detail primitives shared by the domain specialists.

DECISION OWNED
    None. This module holds the domain-neutral vocabulary of details a
    clause may ask for (a number, a money amount, a percentage, a timing, a
    frequency, a minimum or maximum, an approving authority, a responsible
    party, an obligation) and the checks that decide whether ONE evidence
    unit contains such a detail. Which families a specialist recognises,
    with which clause cues, is that specialist's decision: it assembles its
    own family tuple from these primitives and passes it to
    ``requested_details``.

VALUE-SHAPED NUMBERS
    A number counts as a value only when a count noun, a percent sign or a
    currency follows it, or when it is attached to a limit phrase ("not
    exceed 40"). A year ("in 2024"), a dotted section number ("section 3.4")
    and an identifier ("room 204", "examination 101", "page 12", "item 20")
    are never values. The same principle protects every family that reads a
    number.

The code was extracted unchanged from the Teaching & Learning Specialist
(Milestone 3); the factories ``authority_check`` and ``responsibility_check``
let a specialist extend the named bodies and parties without copying the
rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, FrozenSet, List, Optional, Sequence, Tuple

from ...text_utils import tokenize

_NUMBER_WORD = (r"(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
                r"(?:-(?:one|two|three|four|five|six|seven|eight|nine))?")
_NUMBER = re.compile(r"(?<![\d.])\b" + _NUMBER_WORD + r"\b(?!\.\d)", re.I)     # not a section number such as 12.18.1
_FREQUENCY = re.compile(r"\bper\s+(?:week|semester|day|month|year|course|section)\b|\b(?:weekly|daily|monthly|annually|twice|once)\b|"
                        r"\b(?:each|every)\s+(?:week|semester|day|month|year)\b", re.I)
_TIMING = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\b|"
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)\b|\bwithin\s+" + _NUMBER_WORD + r"\s+(?:working\s+|calendar\s+)?(?:hours?|days?|weeks?|months?)\b|"
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
_UNIT_NOUN = re.compile(
    r"^(?:\(\d+\)\s*)?(?:credit\s+|contact\s+|office\s+|class\s+|working\s+|regular\s+|theoretical\s+|teaching\s+|instructional\s+|consultation\s+|"
    r"academic\s+|calendar\s+|business\s+)?"
    r"(?:hours?|days?|weeks?|months?|semesters?|years?|students?|learners?|courses?|classes|sections?|lectures?|sessions?|times|credits?|points?|"
    r"percent|%|assignments?|exams?|examinations?|quizzes|tutorials?|labs?|laboratories|participants?|members?|minutes?|papers?|copies|meetings?)\b", re.I)
_SUBJECT_AFTER = re.compile(r"\b(?:how\s+many|how\s+much|number\s+of|count\s+of|how\s+long)\s+((?:[\w'/-]+\s*){1,5})", re.I)
_OBLIGATION = re.compile(r"\b(?:must|shall|required|requirements?|obligat\w*|mandatory|expected\s+to|responsible\s+for|responsibility\s+to|"
                         r"needs?\s+to|should|are\s+to\s+be|is\s+to\s+be)\b", re.I)
#: Verb forms only: "Management System" and the noun "support" name nobody.
_RESPONSIBLE_VERB = re.compile(r"\bmanag(?:es|ed|ing)\b|\bsupport(?:s|ed)\b|\bmaintain(?:s|ed)\b|\badminister(?:s|ed)\b|\bresponsible\s+for\b|"
                               r"\bresponsibility\s+(?:for|of|as|to)\b|\bin\s+charge\s+of\b|\bhandl(?:es|ed)\b|\boversee(?:s|n)\b|\boperat(?:es|ed)\b|"
                               r"\bprovid(?:es|ed)\b|\brun\s+by\b|\bcoordinat(?:es|ed)\b|\bassist(?:s|ed)\b", re.I)


@dataclass(frozen=True)
class Detail:
    """A detail the clause explicitly asks for."""

    family: str
    label: str
    check: Callable[[str], bool]


def _detail_key(detail: "Detail") -> str:
    return "category:" + detail.label if detail.family == "category" else detail.family


def _fee_ok(unit: str) -> bool:
    """A fee word with a monetary value ("fee of AED 100", "cost is 100
    AED", "free of charge"); "fee applies to examination 101" has none."""
    return bool(_FEE_WORD.search(unit) and _MONEY.search(unit))


def _money_ok(unit: str) -> bool:
    """A monetary value in the unit, whatever the surrounding word."""
    return bool(_MONEY.search(unit))


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


def _limit_ok(unit: str, limit: "re.Pattern[str]", table_row: Optional[Callable[[str], bool]] = None) -> bool:
    """A limit word with a value-shaped number in the same unit. A table row's
    numeric cells are its values ("Laboratory 20 10"); ``table_row`` decides
    what a table row is (default: never)."""
    if not limit.search(unit):
        return False
    if table_row is not None and table_row(unit):
        return bool(_NUMBER.search(unit))
    return any(_is_value(unit, m) for m in _NUMBER.finditer(unit))


def _max_ok(unit: str, table_row: Optional[Callable[[str], bool]] = None) -> bool:
    return _limit_ok(unit, _LIMIT_MAX, table_row)


def _min_ok(unit: str, table_row: Optional[Callable[[str], bool]] = None) -> bool:
    return _limit_ok(unit, _LIMIT_MIN, table_row)


def _frequency_ok(unit: str) -> bool:
    return bool(_FREQUENCY.search(unit) or re.search(r"\b" + _NUMBER_WORD + r"\s+times\b", unit, re.I))


def _quantity_ok(unit: str, subject, subject_tokens: FrozenSet[str], concepts_fn: Callable[[str], object],
                 unit_noun: Optional["re.Pattern[str]"] = None) -> bool:
    """A number tied to a count noun ("five office hours", "12 credit hours",
    "20 students"), whose noun phrase names the counted subject of the
    question, or whose "of" phrase does ("48 hours of office hours"). Bare
    numbers (rooms, pages, sections, dates, list markers) and counts of
    something else ("3 credit hours" for an office-hours question) do not
    count. ``subject`` is the specialist's concepts of the counted subject
    (an object with a ``vocab`` set); ``concepts_fn`` maps text to such an
    object."""
    noun_pattern = unit_noun or _UNIT_NOUN

    def names_subject(text: str) -> bool:
        if concepts_fn(text).vocab & subject.vocab:
            return True
        return not subject.vocab and bool(set(tokenize(text)) & subject_tokens)

    for m in _NUMBER.finditer(unit):
        tail = unit[m.end():].lstrip()
        noun = noun_pattern.match(tail)
        if not noun:
            continue
        phrase = tail[:noun.end()]
        if not subject.vocab and not subject_tokens:
            return True
        if names_subject(phrase):
            return True
        of_phrase = _OF_SUBJECT.match(tail[noun.end():])
        if of_phrase and names_subject(of_phrase.group(1)):
            return True
    return False


def authority_check(verbs: Optional["re.Pattern[str]"] = None, nouns: Optional["re.Pattern[str]"] = None) -> Callable[[str], bool]:
    """A check for "who approves" details: an approval verb AND a named
    authority in the same unit. A specialist extends ``nouns`` with its own
    bodies; "approval is required" without a body never satisfies it."""
    verb_pattern, noun_pattern = verbs or _APPROVAL_VERB, nouns or _AUTHORITY_NOUN

    def check(unit: str) -> bool:
        return bool(verb_pattern.search(unit) and noun_pattern.search(unit))

    return check


def responsibility_check(parties: "re.Pattern[str]", verbs: Optional["re.Pattern[str]"] = None) -> Callable[[str], bool]:
    """A check for "who manages / is responsible for" details: a
    responsibility verb form AND a named party in the same unit."""
    verb_pattern = verbs or _RESPONSIBLE_VERB

    def check(unit: str) -> bool:
        return bool(verb_pattern.search(unit) and parties.search(unit))

    return check


#: One requested-detail family: (family, label, clause pattern, evidence check).
Family = Tuple[str, str, "re.Pattern[str]", Callable[[str], bool]]


def requested_details(clause: str, families: Sequence[Family], categories: Sequence[Tuple[str, "re.Pattern[str]"]],
                      concepts_fn: Callable[[str], object], *, category_label: str = "the requested faculty category (%s)",
                      subject_after: Optional["re.Pattern[str]"] = None, unit_noun: Optional["re.Pattern[str]"] = None) -> List[Detail]:
    """The details a clause explicitly asks for, in the specialist's family
    order. The family named ``quantity`` is built per clause: it carries the
    counted subject ("office hours" in "how many office hours") read with
    ``subject_after`` and ``concepts_fn``. A category named in the clause is
    a detail of its own (the evidence must name the same category)."""
    details: List[Detail] = []
    for family, label, pattern, check in families:
        if not pattern.search(clause):
            continue
        if family == "quantity":
            m = (subject_after or _SUBJECT_AFTER).search(clause)
            subject_text = m.group(1) if m else ""
            subject, tokens = concepts_fn(subject_text), frozenset(tokenize(subject_text))
            details.append(Detail(family, label, (lambda u, s=subject, k=tokens: _quantity_ok(u, s, k, concepts_fn, unit_noun))))
        else:
            details.append(Detail(family, label, check))
    for label, pattern in categories:
        if pattern.search(clause):
            details.append(Detail("category", category_label % label, (lambda p: (lambda u: bool(p.search(u))))(pattern)))
    return details


__all__ = ["Detail", "Family", "authority_check", "requested_details", "responsibility_check"]
