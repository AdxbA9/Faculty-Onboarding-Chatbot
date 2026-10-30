"""
Conservative grounding of a specific ask (Milestone 4, Step 4.8T-C; D-079).

A specialist's relevance rule establishes the TOPIC of an evidence unit and
its detail families establish the details a clause explicitly asks for.
Neither settles a specific ask whose specifics map to no family: "Can I hold
my office hours ONLINE?" was `supported` by "faculty must hold office hours".
This module reads the ask's shape and, for a specific ask, demands that ONE
local evidence unit states the ask's key terms together with the answer:

* every concept of the clause (the domain's vocabulary, aliases included);
* every remaining content word of the clause, stem-matched, with a small
  table of verb classes ("reviews" is stated by "assessing");
* a normative marker (must, may, allowed, prohibited, optional ...) when the
  clause is a deontic yes/no proposition, so a descriptive sentence never
  settles a permission or obligation;
* a named party stated with the asked action when the clause asks "who";
* a number when the clause asks for a value ("what is the teaching load").

A table row (a calendar line, a grade line, a class-size line) is terse: it
must name at least one of the ask's specifics; its cells are its values.
An ask whose shape is not recognised is never `supported`: it receives a
detail no evidence can cover. A general overview ask ("What are the office
hour responsibilities?") is grounded by its concepts alone. The domain
vocabulary is injected; nothing here names a domain. Word-bag grounding
does not bind grammatical positions: that remains the Milestone 5
Verifier's work. No LLM, no network, no state.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .details import Detail

GENERAL = "general"
SPECIFIC_RECOGNISED = "specific_recognised"
SPECIFIC_UNKNOWN = "specific_unknown"

_WORD = re.compile(r"[a-z][a-z'-]*|\d+(?:\.\d+)?")
_NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}
#: Words that carry no specific of their own: interrogatives, auxiliaries, pronouns, determiners, prepositions,
#: conjunctions, adverbs of degree, the wording of a request, and the nouns that name a rule rather than a specific
#: ("obligations", "expectations", "rules"): the answer to "what are my office hour obligations" is the rule itself.
_FILLERS: FrozenSet[str] = frozenset("""
a an the this that these those it its they them their there here some any all every each one ones other another such same own
per of in on at to for from by with about into onto over under between among as than then and or but if whether so nor
i me my mine we us our you your he she his her him
what which who whom whose when where why how many much often long far
can could may might must shall should will would ought
is are am was were be been being do does did done has have had having
need needs needed want wants wanted wish like able unable
still also too very only just even ever always usually generally normally typically please currently now
tell describe explain say says said mean means meant know knows find see look show give
specify specifies specified state states stated mention mentions mentioned cover covers covered address addresses indicate
indicates define defines defined correspond corresponds equal equals equivalent represent represents refer refers
handbook uos policy policies rule rules regulation regulations guideline guidelines information details detail
obligations obligation duties duty expectations expectation expected responsibilities responsibility roles role
main key basic general important specific particular official current standard usual normal relevant applicable exact
typical common appropriate proper correct right certain actual overall major minor various different
yes no not never
""".split())
#: Light verbs and request verbs: they name no specific ("hold", "get", "take", "complete").
_LIGHT_VERBS: FrozenSet[str] = frozenset("""
hold holds held have has had get gets got make makes made take takes took give gives gave use uses used
go goes went come comes came put puts let lets become becomes became carry carries carried count counts counted
assume assumes apply applies applied allow allows allowed permit permits permitted require requires required
complete completes completed undertake undergo fulfil fulfill perform performs accomplish
do does did done be been being is are am was were
""".split())
#: The verbs of a contents ask ("what must a syllabus contain"): the contents check verifies them; in any other ask
#: ("does the load include research") they are the asked relation and must be stated.
_CONTENT_WORDS: FrozenSet[str] = frozenset("""
contain contains contained include includes included including consist consists comprise comprises cover covers
""".split())
#: A phrase that excludes what follows it ("four hours per week, excluding academic advising"): what it names is not
#: stated by the unit.
_EXCLUSION = re.compile(r"\b(?:excluding|except(?:ing)?(?:\s+for)?|other\s+than|not\s+including|apart\s+from|exclusive\s+of|"
                        r"with\s+the\s+exception\s+of|but\s+not|rather\s+than|instead\s+of)\b[^.;:]*", re.I)
#: A term that is a noun ("appointment", "evaluation") is stated by itself, never by the verbs of its class.
_NOUN_LIKE = re.compile(r"(?:ment|tion|sion|ance|ence|ity|ness|ship|ism)s?$")
#: Polarity words of a deontic proposition: read from the clause, matched in the unit as a normative marker.
_POLARITY: FrozenSet[str] = frozenset("""
allowed permitted permissible required mandatory compulsory optional voluntary obliged obligatory prohibited forbidden
possible necessary entitled eligible acceptable
""".split())
#: The words that trigger a detail family: the family's own check verifies them, the unit need not repeat the word.
_LOOSE_TERMS: FrozenSet[str] = frozenset("""
deadline deadlines due date dates maximum maximal minimum minimal limit limits cap caps penalty penalties consequence
consequences sanction sanctions fee fees cost costs charge charges price prices percentage percentages percent number
numbers amount amounts count frequency often range requirement requirements required timing
""".split())
#: Specifics kept even when the domain lists them as generic: a period asked about ("every year") must be stated.
_KEEP: FrozenSet[str] = frozenset("""
year years yearly annual annually month months monthly daily weekly quarterly day days week weeks periodically periodic
""".split())
# ---- Step 4.8T-C3 (D-081): explicit frequency units ----------------------------------------------------------------
#: The bounded frequency vocabulary: "per week", "each academic year", "annually" normalise to a unit; "periodically" is
#: no unit and satisfies no exact unit. No temporal reasoning beyond this.
_FREQ_UNIT_WORDS: Tuple[Tuple[str, str], ...] = (("day", "day"), ("days", "day"), ("week", "week"), ("weeks", "week"), ("month", "month"),
                                                 ("months", "month"), ("semester", "semester"), ("semesters", "semester"), ("term", "term"),
                                                 ("terms", "term"), ("year", "year"), ("years", "year"))
_FREQ_WORDS: Dict[str, str] = {"daily": "day", "weekly": "week", "monthly": "month", "annually": "year", "annual": "year", "yearly": "year",
                               "periodically": "periodic", "periodic": "periodic", "regularly": "periodic"}


def _compile_freq(units: Sequence[Tuple[str, str]]) -> "re.Pattern[str]":
    return re.compile(r"\b(?:per|each|every)\s+(?:(?!(?:" + "|".join(u for u, _ in units) + r")\b)\w+\s+)?(" + "|".join(u for u, _ in units) + r")\b", re.I)


_FREQ_RE = _compile_freq(_FREQ_UNIT_WORDS)


#: Step 4.8T-C6 (D-084): the frequency reader normalises its text first, so "per-semester" and "each-year" are read
#: like "per semester" and "each year". False restores the raw lowercased text (mutation only).
_FREQ_NORMALISED = True


def frequencies_of(text: str) -> FrozenSet[str]:
    """The frequency units ``text`` states: "per week" and "weekly" are week, "each academic year", "each-year" and
    "annually" are year, "periodically" is periodic. The text is normalised first (D-084): a hyphenated period is the
    spaced one."""
    lowered = normalise(text) if _FREQ_NORMALISED else text.lower()
    units = dict(_FREQ_UNIT_WORDS)
    found = {units[m.group(1).lower()] for m in _FREQ_RE.finditer(lowered) if m.group(1).lower() in units}
    for word in _WORD.findall(lowered):
        if word in _FREQ_WORDS:
            found.add(_FREQ_WORDS[word])
    return frozenset(found)


def _freq_term_unit(term: str) -> Optional[str]:
    """The frequency unit a kept term stands for ("annually" -> year, "semester" -> semester), else None."""
    return _FREQ_WORDS.get(term) or dict(_FREQ_UNIT_WORDS).get(term)


# ---- Step 4.8T-C5 (D-083): local numeric facts ----------------------------------------------------------------------
#: A specific numeric ask ("two office hours per week", "24 credit hours per semester") is one fact: value, the object it
#: qualifies and its period. The evidence must state that fact in one clause; a unit with several facts ("12 credit hours
#: per regular semester or 24 credit hours per academic year", "five office hours per week, spread over two days") never
#: lends the value of one fact to the object or period of another. Mutation switches restore the pre-D-083 behaviour.
_NUMERIC_BINDING = True                  # False: a number anywhere in the unit
_NUMERIC_OBJECT_BINDING = True           # False: the number needs no object
_NUMERIC_FREQ_BINDING = True             # False: the fact's period is not compared
_NUMERIC_SEGMENTS = True                 # False: the whole unit is one clause (cross product)
_NUMERIC_TIME_OBJECTS_MERGED = False     # True: "days" satisfies "hours"
_FREQ_UNITS_MERGED = False               # True: semester equals year
_FREQ_ACTION_LOCAL = True                # False: an action's frequency may sit in another clause
_FREQ_WINDOW_COMMA = True                # False: a fact's period is read past a comma
_FACT_CLAUSE_LOCAL = True                # False: a fact of another independent clause counts for the asked action
_STRONG_BOUNDARY = re.compile(r"[;:.?!]")
_PATIENT_PRONOUNS: FrozenSet[str] = frozenset("they these those it this such them".split())
_SUBORDINATE_OPENERS: FrozenSet[str] = frozenset("if when while because although unless until whereas where before after since provided".split())
_FREQ_WINDOW_SPLIT = re.compile(r"[,;:]|\.(?!\d)|\b(?:and|or|but)\b", re.I)
_SEGMENT_SPLIT = re.compile(r"[;:]|\.(?!\d)|\b(?:and|or|but)\b", re.I)
_FINITE_CUES: FrozenSet[str] = frozenset("must shall should may might can could will would have has had do does did".split())
_PERIOD_ADJECTIVE_UNITS: Dict[str, str] = {"day": "day", "week": "week", "month": "month", "semester": "semester", "term": "term", "year": "year"}
_NUM_TOKEN = re.compile(r"^\d+(?:\.\d+)?$")
_FACT_TOKEN = re.compile(r"[a-z][a-z']*|\d+(?:\.\d+)?|[.,;:()?!]")


def _segment_index(text: str, position: int) -> int:
    """The clause of ``text`` (split at ";", ":", "and", "or", "but") that contains ``position``."""
    if not _NUMERIC_SEGMENTS:
        return 0
    return len(_SEGMENT_SPLIT.findall(text[:position]))


def _segment_end(text: str, position: int) -> int:
    if not _NUMERIC_SEGMENTS:
        return len(text)
    m = _SEGMENT_SPLIT.search(text, position)
    return m.start() if m else len(text)


def numeric_facts(norm_text: str) -> List[Dict[str, object]]:
    """The numeric facts of normalised text: each run of numbers with the content words it qualifies (up to three, stopping
    at a preposition, a period word or punctuation), the frequency units stated in the rest of its clause, its clause
    index and its character position. "a minimum of 5 office hours per week, spread over a minimum of 2 days" gives
    (5,) office hours {week} and (2,) days {}."""
    tokens = list(_FACT_TOKEN.finditer(norm_text))
    facts: List[Dict[str, object]] = []
    i = 0
    while i < len(tokens):
        if not _NUM_TOKEN.match(tokens[i].group(0)):
            i += 1
            continue
        j = i
        values: List[float] = []
        while j < len(tokens) and _NUM_TOKEN.match(tokens[j].group(0)):
            values.append(float(tokens[j].group(0)))
            j += 1
        obj: List[str] = []
        k = j
        while k < len(tokens) and len(obj) < 3:
            w = tokens[k].group(0)
            nxt = tokens[k + 1].group(0) if k + 1 < len(tokens) else None
            if (not w[0].isalpha() or w in _PREPS or w in ("per", "each", "every") or w in _CLAUSE_CUT or w in ("and", "or", "but")
                    or w in _FREQ_WORDS or w in _AGENT_MARKERS or w in _CHAIN_TOKENS
                    or (_verb_candidate(w, nxt) and not w.endswith(("ing", "s")))):
                break                                             # "5 office hours weekly", "20 percent prohibit the student": the object ends
            if w not in _DETS:
                obj.append(w)
            k += 1
        before: List[str] = []
        k = i - 1
        while k >= 0 and len(before) < 3:
            w = tokens[k].group(0)
            if not w[0].isalpha() or w in _PREPS or w in _CLAUSE_CUT or w in ("and", "or", "but") or w in _AGENT_MARKERS or w in _CHAIN_TOKENS:
                break
            if w not in _DETS:
                before.insert(0, w)                               # "absences reaching 20 percent": the object may precede the number
            k -= 1
        end = _segment_end(norm_text, tokens[j - 1].end())
        if _FREQ_WINDOW_COMMA and _NUMERIC_SEGMENTS:
            m = _FREQ_WINDOW_SPLIT.search(norm_text, tokens[j - 1].end())
            end = min(end, m.start() if m else len(norm_text))    # "12 credit hours, hold 5 office hours per week": no period for 12
        facts.append({"values": tuple(values), "obj": tuple(obj), "before": tuple(before), "freqs": frequencies_of(norm_text[tokens[j - 1].end():end]),
                      "segment": _segment_index(norm_text, tokens[i].start()), "pos": tokens[i].start(), "token": i})
        i = j
    return facts


def _period_adjectives(norm_clause: str) -> FrozenSet[str]:
    """The period a clause states as a modifier of the asked object ("the semester workload", "the annual load"): a unit
    word that a content word follows and no number or "per" precedes."""
    words = _FACT_TOKEN.findall(norm_clause)
    found = set()
    for k, w in enumerate(words):
        if w in _PERIOD_ADJECTIVE_UNITS and k + 1 < len(words):
            nxt, prev = words[k + 1], words[k - 1] if k else ""
            if (nxt[0].isalpha() and nxt not in _PREPS and nxt not in _DETS and nxt not in ("and", "or", "but") and nxt not in _FREQ_WORDS
                    and not _NUM_TOKEN.match(prev) and prev not in ("per", "each", "every")):
                found.add(_PERIOD_ADJECTIVE_UNITS[w])
    return frozenset(found)


def _ask_numeric_facts(clause: str) -> List[Dict[str, object]]:
    """The facts a clause asks about: a number with an object or a period; a year span, a bare number and a number whose
    "object" is a verb are no facts (the calendar rule and the term check keep them). A period stated before the number
    as a modifier ("the semester workload ... 24 credit hours") is the fact's period."""
    facts = []
    norm = normalise(clause)
    adjectives = _period_adjectives(norm) if _NUMERIC_FREQ_BINDING else frozenset()
    for f in numeric_facts(norm):
        values, obj = f["values"], f["obj"]
        if any(v >= 1900 for v in values) or (not obj and not f["freqs"]):
            continue
        head = obj[0] if obj else ""
        if head and (head in _LIGHT_VERBS or head in _FILLERS or (_known_verb(head) and not head.endswith(("ing", "s")))):
            continue                                              # "semester 2 begin": a verb after the number; "15 teaching hours" is a fact
        if not f["freqs"] and adjectives:
            f["freqs"] = adjectives
        facts.append(f)
    return facts


def _exact_verb(word: str) -> bool:
    """``word`` is a verb form as written (a light verb or a known verb by its base), never by a stem ("examination")."""
    return word in _LIGHT_VERBS or word in _RELATION_LIGHT or word in _KNOWN_VERBS or _verb_base(word) in _KNOWN_VERBS


def _independent_clause(unit: "_Unit", start: int, end: int, phrases: Sequence[Tuple[int, int]]) -> bool:
    """The clause between ``start`` and ``end`` stands on its own: it does not open with a subordinator, and content words
    stand before its first auxiliary, copula or finite verb ("re-marking requests are filed", "the dean reviews them",
    "full-time faculty hold 5 office hours"). "and submit the results", "or booklets", "and percentages, to the
    department chair within 48 hours" and "if a student's absences exceed 10%" do not."""
    content, first = 0, True
    for k, (_w, low, s, _e) in enumerate(unit.tokens):
        if s < start or s >= end:
            continue
        if first:
            first = False
            if low in _SUBORDINATE_OPENERS:
                return False
        finite = low in _BE_FORMS or low in _FINITE_CUES or (
            unit.role[k] is None and _exact_verb(low) and not low.endswith("ing") and not _is_participle(low)
            and (k == 0 or unit.low(k - 1) not in _DETS)
            and not any(a <= s < b and len(_TOKEN.findall(unit.text[a:b])) > 1 for a, b in phrases))   # "course files": a noun
        if finite:
            return content > 0
        if low.isalpha() and low not in _DETS and low not in _PREPS and low not in _CHAIN_TOKENS and low not in _BOUNDARIES:
            content += 1
    return False


def _excluded_segments(relations: "RelationChecker", relation: Relation, unit: "_Unit", text: str) -> Set[int]:
    """The clauses whose numbers state nothing about the asked action: the independent clauses of the unit that hold no
    complete occurrence of the action by the asked actor ("re-marking requests are filed within 7 days" states no
    48-hour submission). A subordinate or coordinated clause of the occurrence keeps its numbers ("if absences exceed
    10% ..., the instructor will issue a warning"; "submit the results ... and percentages ... within 48 hours")."""
    if not (_FACT_CLAUSE_LOCAL and _NUMERIC_SEGMENTS):
        return set()
    occupied: Set[int] = set()
    for o in relations.occurrences(relation, unit):
        if not (o["object_ok"] and o["target_ok"] and o["modifiers_ok"]):
            continue                                              # a synonym with another object ("filed" for "submit")
        if o["voice"] == "passive" and relation.obj:
            local = _STRONG_BOUNDARY.split(str(o.get("patient", "")))[-1]
            first = _words(local)[:1]
            if first and first[0] not in _PATIENT_PRONOUNS and not relations._object_stated(relation.obj, local):
                continue                                          # "results within 48 hours; re-marking requests are filed": another patient
        if relation.actor is not None and o["subject"] not in (None, relation.actor):
            continue
        if relation.actor is None and relation.actor_words and o["subject"] is not None:
            continue
        occupied.add(_segment_index(text, unit.tokens[o["index"]][2]))
    bounds = list(_SEGMENT_SPLIT.finditer(text))
    phrases = relations.lexicon.phrases(unit.text)
    excluded: Set[int] = set()
    for seg in range(len(bounds) + 1):
        if seg in occupied:
            continue
        start = bounds[seg - 1].end() if seg else 0
        end = bounds[seg].start() if seg < len(bounds) else len(text)
        if _independent_clause(unit, start, end, phrases):
            excluded.add(seg)
    return excluded


def _numeric_object_ok(ask_obj: Sequence[str], unit_obj: Sequence[str], concepts_fn: Callable[[str], object]) -> bool:
    """The unit's number qualifies the asked object: the same concept ("office hours", "credit hours") or the same head
    word ("days", "percent", "credits")."""
    if not ask_obj or not _NUMERIC_OBJECT_BINDING:
        return True
    if _NUMERIC_TIME_OBJECTS_MERGED and any(w in ("days", "weeks", "hours") for w in unit_obj) and any(w.endswith("hours") for w in ask_obj):
        return True                                               # mutation only
    own = frozenset(getattr(concepts_fn(" ".join(ask_obj)), "vocab", frozenset()))
    have = frozenset(getattr(concepts_fn(" ".join(unit_obj)), "vocab", frozenset()))
    if own:
        return own <= have
    return any(_same(ask_obj[-1], u) for u in unit_obj)


_CATEGORY_WORDS: FrozenSet[str] = frozenset("part full regular visiting adjunct senior new".split())


def _category_of(clause: str, lexicon: RoleLexicon) -> Optional[str]:
    """The category word that opens the clause's role phrase ("part-time instructors" -> part, "senior lecturer" -> senior)."""
    for a, b, _role in lexicon.spans(normalise(clause)):
        first = normalise(clause)[a:b].split()[0]
        if first in _CATEGORY_WORDS:
            return first
    return None


def _fact_subject_ok(unit: "_Unit", actor: str, token: int, ask_category: Optional[str] = None) -> bool:
    """The clause of the number names no subject other than the asked actor: the subject read before the number is the
    actor, or none, or the actor stands with it in one coordinated subject ("regular faculty, visiting faculty or
    senior lecturer workload equal to 12 credit hours"); a category prefix of the subject ("full-time faculty") must be
    the asked one ("part-time instructors")."""
    subject = unit.subject_before(token)
    if subject == actor and ask_category is not None:
        k = next((m for m in range(token - 1, -1, -1) if unit.role[m] == actor), None)
        while k is not None and k > 0 and unit.role[k - 1] == actor:
            k -= 1
        if k is not None and unit.low(k) in _CATEGORY_WORDS and unit.low(k) != ask_category:
            return False                                          # "full-time faculty hold 5 office hours": not part-time
    if subject is None or subject == _OTHER_SUBJECT or subject == actor:
        return True
    subject_index = next((k for k in range(token - 1, -1, -1) if unit.role[k] == subject), None)
    actor_index = next((k for k in range(token - 1, -1, -1) if unit.role[k] == actor), None)
    if subject_index is None or actor_index is None or actor_index > subject_index:
        return False
    between = [unit.low(m) for m in range(actor_index + 1, subject_index) if unit.role[m] is None]
    return all(w in _DETS or w in (",", "and", "or") for w in between)


def _freq_matches(asked: str, stated: FrozenSet[str]) -> bool:
    if asked == "periodic":
        return bool(stated)
    if _FREQ_UNITS_MERGED and asked in ("semester", "year") and stated & {"semester", "year"}:
        return True                                               # mutation only
    return asked in stated

#: An ask about the contents of something ("what must a syllabus contain"): the unit must state contents, not merely
#: oblige someone to have the thing.
_CONTENTS_ASK = re.compile(r"\b(?:what|which)\s+(?:must|should|does|do|has|have|needs?|is|are)\b[^.?]{0,40}\b(?:contain|include|cover|consist|"
                           r"comprise|have\s+in|list|specify)\w*\b|\b(?:contents?|components?|elements?|items?|sections?)\s+of\b", re.I)
_CONTENT_VERB = re.compile(r"\b(?:outlin\w*|includ\w*|contain\w*|compris\w*|consist\w*|cover\w*|list\w*|specif\w*|provid\w*|encompass\w*|"
                           r"stat(?:e|es|ed|ing)|address\w*|detail\w*|describ\w*|present\w*|set\s+out|indicat\w*|shows?|identif\w*|"
                           r"consists?\s+of|made\s+up\s+of)\b", re.I)
_INFLECTIONS = ("", "e", "s", "es", "ed", "d", "ing")
#: The noun of an action bound to a party ("approval of the department council", "review by the chair"): states the
#: party's action as well as the verb does; a bare noun elsewhere ("teaching assignments") does not.
_NOUN_SUFFIXES = ("al", "ion", "tion", "ation", "ment", "ance", "ence", "ing")
#: A unit that states a rule rather than describing a state of affairs.
_NORMATIVE = re.compile(
    r"\b(?:must|shall|should|required?|requires|requirement|mandatory|compulsory|obligat\w*|(?<!\d\s)(?<!\d)may(?!\s*\d)|can(?:not)?|could|allowed|permitted|"
    r"permissible|prohibited|forbidden|not\s+allowed|optional|voluntary|expected\s+to|responsible\s+for|responsibility\s+to|"
    r"needs?\s+to|entitled|eligible|has\s+to|have\s+to|will\s+not|won't|no\s+longer|at\s+least|minimum|maximum|only)\b", re.I)
#: Verb classes: the asked action is stated by any member (prefixes of stems; a member with a space is a phrase).
_VERB_CLASSES: Tuple[Tuple[str, ...], ...] = (
    ("review", "assess", "evaluat", "examin", "monitor", "check", "inspect", "audit", "scrutin"),
    ("approv", "authori", "endors", "sign", "sanction", "ratif", "clear", "grant"),
    ("assign", "appoint", "designat", "distribut"),
    ("schedul", "timetabl"),
    ("decid", "determin", "establish", "fix", "defin", "set"),
    ("submit", "enter", "post", "upload", "hand", "send", "deliver", "file", "turn in"),
    ("reduc", "lower", "decreas", "cut", "releas", "exempt", "reliev", "waiv", "lighten"),
    ("chang", "modif", "revis", "amend", "replac", "updat", "alter", "switch", "substitut"),
    ("keep", "kept", "maintain", "stor", "retain", "preserv", "archiv"),
    ("receiv", "get", "gets", "got", "getting", "obtain"),
    ("deliver", "offer", "teach", "conduct", "run", "provid", "taught"),
    ("record", "track", "register", "log", "document", "note", "taken", "take attendance"),
    ("cancel", "call off", "suspend", "postpon", "resched"),
    ("bar", "prohibit", "prevent", "exclud", "debar", "deny", "denied", "forbid", "not allowed", "not permitted", "disqualif"),
    ("manag", "administer", "operat", "oversee", "supervis", "handl", "responsib", "in charge", "coordinat"),
    ("support", "assist", "help"),
    ("pay", "paid", "compensat", "remunerat", "salar", "stipend"),
    ("combin", "merg", "join", "consolidat", "cross list", "cross-list"),
    ("inform", "notif", "tell", "advis", "report", "announc"),
    ("count", "includ", "consider", "recogni", "contribut", "counted towards", "credited towards"),
    ("observ", "visit", "attend"),
    ("post", "publish", "display", "announc", "upload"),      # D-081: "enter" (grades) is not "publish"
    ("meet", "consult", "see", "advis"),
    ("miss", "absen", "absent", "fail to attend", "not attend"),
    ("attend", "presen", "participat"),
    ("complet", "finish", "fulfil", "satisf"),
    ("pass", "succeed", "clear"),
    ("start", "begin", "commenc", "first day"),
    ("end", "finish", "last day", "conclud"),
    ("reach", "forward", "transfer", "transmit", "send", "sent", "flow", "pass", "export", "feed"),
    ("caught", "catch", "detect", "discover", "found", "identif"),
    ("enter", "input", "key in", "record", "post", "submit"),
)
_VALUE_ASK = re.compile(
    r"^\s*(?:(?:and|also|but|so)\s+)?what\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+|my\s+|our\s+|your\s+|its\s+)?(?:[\w'-]+\s+){0,5}?"
    r"(?:loads?|fees?|costs?|charges?|deadlines?|numbers?|percentages?|maximum|minimum|limits?|caps?|rates?|duration|length|"
    r"amounts?|sizes?|weights?|weighting|shares?|ratios?|thresholds?|prices?|salar(?:y|ies)|pay|"
    r"compensation|proportion|quota|allowance)\b|\bhow\s+(?:many|much|often|long)\b|\bwhat\s+(?:percentage|proportion|share|date|time|day)\b", re.I)
_LEAD = r"^\s*(?:(?:and|also|but|so|then|please)\s+)?"
_PARTY_ASK = re.compile(_LEAD + r"(?:who|whom|whose|which\s+(?:office|body|council|committee|department|person|unit|authority|dean|chair|"
                        r"officer|staff|team|division|section))\b", re.I)
_PARTY_ACTION = re.compile(_LEAD + r"(?:who|whom|which\s+[\w-]+)\s+(?:(?:is|are|must|should|can|could|will|would|shall|may|has|have|do|does|did)\s+)?"
                           r"(?:(?:the|a|an|my|our|their|be|been|to|i|we)\s+)*([a-z][a-z-]*)", re.I)
_YESNO_ASK = re.compile(_LEAD + r"(?:can|could|may|might|must|should|shall|am|is|are|was|were|do|does|did|have|has|had|will|would|need)\b", re.I)
_DEONTIC_ASK = re.compile(
    _LEAD + r"(?:can|could|may|might|must|should|shall|am\s+i|are\s+(?:we|faculty|instructors|lecturers|students|members)|"
    r"do\s+(?:i|we)\s+(?:have|need)\s+to|does\s+[\w\s'-]{1,25}?\s+(?:have|need)\s+to|is\s+it\s+(?:mandatory|required|allowed|permitted|"
    r"possible|optional|compulsory|necessary|ok|okay|acceptable))\b|"
    r"\b(?:allowed|permitted|permissible|required|mandatory|compulsory|optional|voluntary|prohibited|forbidden|obliged|obligatory|"
    r"have\s+to|need\s+to|entitled|eligible)\b", re.I)
_OVERVIEW = re.compile(
    _LEAD + r"(?:tell\s+me\s+(?:more\s+)?(?:about|on)|describe|explain|summari[sz]e|(?:give\s+me\s+)?(?:an\s+)?(?:overview|summary)\s+(?:of|on)|"
    r"what\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+|my\s+|our\s+|your\s+)?|what\s+about|what\s+(?:does|do)\s+(?:the\s+)?(?:handbook|policy|"
    r"policies|rules?|regulations?|university|uos)\s+say\s+(?:about|on)|information\s+(?:on|about)|anything\s+(?:on|about)|"
    r"what\s+(?:is|are)\s+there\s+(?:on|about))\b", re.I)
_WHAT_ASK = re.compile(_LEAD + r"what\b", re.I)
_SPECIFIC_WH = re.compile(
    _LEAD + r"(?:when|where|whereabouts|by\s+when|until\s+when|how\s+(?:many|much|often|long|early|late|far|soon)|which|what\s+(?:percentage|"
    r"proportion|share|date|time|day|must|should|is\s+required|are\s+the\s+requirements|happens|information|documents?)|"
    r"how\s+(?:is|are|was|were|do|does|did|can|could|should|would|will|to)\b|what\s+(?:do|does)\s+[\w\s'-]{1,30}?\s+(?:need|have)\s+to\b)", re.I)
_UNKNOWN_WH = re.compile(_LEAD + r"(?:why|how\s+come|what\s+(?:kind|type|sort)\s+of|what\s+if|what\s+else|whatever|whoever|whenever)\b", re.I)
_QUESTIONISH = re.compile(_LEAD + r"(?:who|whom|whose|what|which|when|where|why|how|can|could|may|might|must|should|shall|am|is|are|was|"
                          r"were|do|does|did|have|has|had|will|would|need|tell|describe|explain|summari[sz]e|give|information|anything|"
                          r"overview)\b", re.I)
_GRADE_LETTER = re.compile(r"\b(?:grade|mark|letter)\s+(?:of\s+)?([A-F][+-]?)(?![\w-])")
_MIN_STEM = 4


def _stem(word: str) -> str:
    w = word.lower()
    for suffix in ("ization", "isation", "ities", "ations", "ation", "ments", "ment", "ingly", "ing", "edly", "ness", "ies", "ed",
                   "es", "ly", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:-len(suffix)]
            break
    return w


def _same(a: str, b: str) -> bool:
    """Two words name one word: equal, or (by their stems or as written) one is a prefix of the other and both are
    long enough."""
    if a == b:
        return True
    forms_a, forms_b = {a, _stem(a)}, {b, _stem(b)}
    for x in forms_a:
        for y in forms_b:
            if x == y:
                return True
            if len(x) >= _MIN_STEM and len(y) >= _MIN_STEM and (x.startswith(y) or y.startswith(x)):
                return True
    return False


def normalise(text: str) -> str:
    """Lower-cased text with number words as digits, "%" as "percent" and separators as spaces."""
    text = text.lower().replace("%", " percent ").replace("&", " and ")
    text = re.sub(r"(?<=\w)[-‐‑‒–—/](?=\w)", " ", text)
    return " ".join(_NUMBER_WORDS.get(w, w) for w in text.split())


def _words(text: str) -> List[str]:
    return _WORD.findall(normalise(text))


def _member_same(member: str, word: str) -> bool:
    """``word`` (as written or by its stem) is the class member ``member``, a stem written without its ending."""
    for form in (word.lower(), _stem(word)):
        if form == member or (len(member) >= _MIN_STEM and form.startswith(member)):
            return True
    return False


def _classes_of(term: str) -> List[Tuple[str, ...]]:
    if _NOUN_LIKE.search(term.lower()):
        return []
    return [cls for cls in _VERB_CLASSES if any(_member_same(m, term) for m in cls if " " not in m)]


def strip_exclusions(text: str) -> str:
    """``text`` without what its exclusion phrases name."""
    return _EXCLUSION.sub(" ", text)


def _verb_form(member: str, word: str) -> bool:
    """``word`` is an inflection of the verb ``member`` (assign, assigns, assigned, assigning; allocate, allocated),
    never a derived noun (assignment, allocation)."""
    if not word.startswith(member) and not (member.endswith("e") and word.startswith(member[:-1])):
        return False
    rest = word[len(member):] if word.startswith(member) else word[len(member) - 1:]
    if rest in _INFLECTIONS:
        return True
    # a doubled final consonant before -ed / -ing ("transferred", "submitting")
    return len(member) >= 3 and member[-1] not in "aeiouy" and rest[:1] == member[-1] and rest[1:] in ("ed", "ing")


def verb_stated(action: str, unit_text: str, unit_words: Sequence[str]) -> bool:
    """The asked action is stated as a verb form in the unit: the action itself or a member of its verb class, as a
    verb (not as a derived noun), or a class phrase."""
    base = action.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if base.endswith(suffix) and len(base) - len(suffix) >= 3:
            base = base[:-len(suffix)]
            break
    if any(_verb_form(base, w) for w in unit_words):
        return True
    for cls in _classes_of(action):
        for member in cls:
            if " " in member:
                if member in unit_text:
                    return True
            elif any(_verb_form(member, w) for w in unit_words):
                return True
    return False


def term_stated(term: str, unit_text: str, unit_words: Sequence[str]) -> bool:
    """``term`` (a content word, a grade letter or a number) is stated in the unit: the number or letter itself, the
    word by stem, or a member of one of its verb classes (by stem, or as a phrase)."""
    if term.isdigit() or term.replace(".", "", 1).isdigit() or len(term) <= 2:
        return term in unit_words
    if any(_same(term, w) for w in unit_words):
        return True
    for cls in _classes_of(term):
        for member in cls:
            if " " in member:
                if member in unit_text:
                    return True
            elif any(_member_same(member, w) for w in unit_words):
                return True
    return False



# ---------------------------------------------------------------------------
# Relation binding (Step 4.8T-C2, D-080): the asked actor performs the asked action on the asked object
# ---------------------------------------------------------------------------
#: Verbs that carry no action of their own; the action is the noun after them ("give final approval").
_RELATION_LIGHT: FrozenSet[str] = frozenset("""
give gives gave given giving grant grants granted granting provide provides provided providing get gets got getting obtain obtains
obtained receive receives received make makes made making take takes took taken taking hold holds held holding have has had do does did
be is are was were been being need needs use uses used
""".split())
_RECIPIENT_VERBS: FrozenSet[str] = frozenset("get gets got obtain obtains obtained receive receives received".split())
#: Step 4.8T-C3 (D-081): with an ordinary object these are actions of their own (recipient, object, source), never a
#: light verb satisfied by the object's words.
_RECEIVE_VERBS: FrozenSet[str] = frozenset("receive receives received receiving get gets got getting obtain obtains obtained obtaining".split())
#: Nouns that name an action, and the verb they stand for.
_ACTION_NOUNS: Dict[str, str] = {
    "approval": "approve", "approvals": "approve", "review": "review", "reviews": "review", "permission": "permit", "consent": "consent",
    "decision": "decide", "decisions": "decide", "authorization": "authorize", "authorisation": "authorise", "recommendation": "recommend",
    "recommendations": "recommend", "notification": "notify", "submission": "submit", "assessment": "assess", "evaluation": "evaluate",
    "distribution": "distribute", "appointment": "appoint", "assignment": "assign", "supervision": "supervise", "observation": "observe",
    "endorsement": "endorse", "clearance": "clear", "signature": "sign", "enrolment": "enrol", "enrollment": "enroll", "registration": "register",
    "confirmation": "confirm", "payment": "pay", "correction": "correct", "verification": "verify", "preparation": "prepare",
}
_ACTION_MODIFIERS: FrozenSet[str] = frozenset("final initial prior written formal official preliminary annual periodic direct".split())
_IRREGULAR: Dict[str, str] = {"held": "hold", "kept": "keep", "taught": "teach", "made": "make", "given": "give", "taken": "take",
                              "written": "write", "done": "do", "paid": "pay", "met": "meet", "led": "lead", "sent": "send",
                              "built": "build", "set": "set", "put": "put", "gave": "give", "took": "take", "wrote": "write",
                              "shown": "show", "found": "find"}
#: Past participles a passive ask may carry that end in neither -ed nor -en (Step 4.8T-C4, D-082); the parser already
#: knows their bases.
_IRREGULAR_PARTICIPLES: Tuple[str, ...] = ("sent", "kept", "held", "made", "taught", "built", "given", "taken", "written", "done", "paid",
                                           "met", "led", "set", "put", "shown", "found")
_AUX = r"(?:must|should|shall|do|does|did|can|could|may|might|will|would|is|are|was|were|need|needs|has|have|am)"
_CHAIN = (r"(?:(?:be\s+)?(?:required|allowed|permitted|expected|obliged|supposed|entitled|encouraged|able|responsible|eligible)\s+(?:to|for)\s+|"
          r"have\s+to\s+|need\s+to\s+|to\s+)?")
_DET = r"(?:(?:the|a|an|my|our|your|their|its|his|her|all|any|each|every|this|these|those|such)\s+)*"
_NP = r"[\w'-]+(?:\s+[\w'-]+){0,5}?"
_NOT_CHAIN = r"(?!(?:required|allowed|permitted|expected|obliged|supposed|entitled|encouraged|eligible|able|responsible|meant)\b)"
#: What may follow the agent phrase of a passive ask without ending it early: "by faculty members of the department",
#: "by instructors whose courses ...", "by the instructor to the Registrar" (Step 4.8T-C3, D-081).
_AGENT_TAIL: Tuple[str, ...] = ("of", "whose", "who", "which", "that", "for", "in", "at", "on", "before", "after", "within", "during", "each",
                                "every", "to", "from", "with", "per", "via", "through", "about", "under", "between", "when", "if")


def _participle_group(parts: Sequence[str]) -> str:
    return r"(?P<verb>[a-z]+(?:ed|en|t)" + "".join("|" + p for p in parts) + r")"


def _compile_passive_ask(tail: Sequence[str]) -> "re.Pattern[str]":
    return re.compile(r"^\s*" + _AUX + r"\s+" + _DET + r"(?P<obj>" + _NP + r")\s+(?:be\s+)?" + _NOT_CHAIN +
                      _participle_group(_IRREGULAR_PARTICIPLES) + r"(?:\s+\w+ly)?\s+by\s+" + _DET + r"(?P<agent>" + _NP + r")"
                      r"(?P<rest>\s*(?:\?|$|[,.].*|\s+(?:" + "|".join(tail) + r")\b.*))", re.I | re.S)


_PASSIVE_ASK = _compile_passive_ask(_AGENT_TAIL)
def _compile_short_passive(parts: Sequence[str]) -> "re.Pattern[str]":
    """"Are final grades sent to the dean?": an agentless passive ask with a regular or a listed irregular participle."""
    return re.compile(r"^\s*(?:is|are|was|were)\s+" + _DET + r"(?P<obj>" + _NP + r")\s+" + _NOT_CHAIN +
                      r"(?P<verb>[a-z]+(?:ed|en)" + "".join("|" + p for p in parts) + r")\b(?P<rest>[^?]*)", re.I)


_PASSIVE_SHORT_ASK = _compile_short_passive(_IRREGULAR_PARTICIPLES)
_TRAILING_AGENT = re.compile(r"\bby\s+" + _DET + r"(?P<agent>" + _NP + r")\s*$", re.I)
_PASSIVE_NOAGENT_ASK = re.compile(r"^\s*" + _AUX + r"\s+" + _DET + r"(?P<obj>" + _NP + r")\s+be\s+" + _NOT_CHAIN + _participle_group(_IRREGULAR_PARTICIPLES) +
                                  r"\b(?P<rest>[^?]*)", re.I)
_ACTIVE_ASK = re.compile(r"\b(?P<aux>" + _AUX + r")\s+" + _DET + r"(?P<actor>" + _NP + r")\s+(?P<chain>" + _CHAIN + r")(?P<verb>[a-z]+)\b(?P<rest>[^?]*)", re.I)
_PARTY_REL = re.compile(_LEAD + r"(?:who|whom|which\s+[\w-]+)\s+(?:(?:is|are|must|should|can|could|will|would|shall|may)\s+)?" + _DET +
                        r"(?:(?:be|been)\s+)?(?:(?:required|allowed|permitted|expected|obliged|supposed|entitled|responsible)\s+(?:to|for)\s+)?"
                        r"(?P<verb>[a-z]+)\b(?P<rest>[^?]*)", re.I)
_PREPS: FrozenSet[str] = frozenset("""
of by for with from to in on at into onto under through during before after within without between among about via per against upon
towards toward regarding concerning across along versus
""".split())
_DEST_PREPS: FrozenSet[str] = frozenset("to into onto towards".split())
_CLAUSE_CUT: FrozenSet[str] = frozenset("if when while because unless whether although whereas once since each every".split())
#: The evidence may phrase a destination with these; a source ("from") and a beneficiary ("for") are their own classes.
_PREP_CLASSES: Dict[str, FrozenSet[str]] = {
    "to": frozenset("to into onto towards on in via through at".split()), "into": frozenset("to into onto towards on in via through at".split()),
    "onto": frozenset("to into onto towards on in via through at".split()), "from": frozenset({"from"}), "for": frozenset({"for"}),
}
_CAUSATIVES: FrozenSet[str] = frozenset("have has had having get gets got getting".split())
_OTHER_SUBJECT = "__other__"                                  # a sentinel: some party other than the asked one
_BOUNDARIES: FrozenSet[str] = frozenset(". , ; : ( ) ? ! while if when because where unless although whereas".split())
_RELATIVES: FrozenSet[str] = frozenset("which that who whom".split())
_CHAIN_TOKENS: FrozenSet[str] = frozenset("""
to be being been am is are was were required allowed permitted expected obliged supposed entitled encouraged responsible able eligible
for must shall should may can could will would might also directly automatically equally then not and or
""".split())
_BE_FORMS: FrozenSet[str] = frozenset("is are was were be been being am get gets got".split())
_DETS: FrozenSet[str] = frozenset("the a an my our your their its his her all any each every this these those such own".split())
_NORMATIVE_TOKENS: FrozenSet[str] = frozenset("""
must shall should may can cannot required allowed permitted expected obliged entitled encouraged responsible responsibility
responsibilities obligation obligations duty duties need needs
""".split())
_PRONOUN_OBJECTS: FrozenSet[str] = frozenset("it them these those this that".split())
_PERSON_PRONOUNS: FrozenSet[str] = frozenset("he she they".split())
_SENTENCE_ENDS: FrozenSet[str] = frozenset(". ; : ! ?".split())
_CHAIN_PARTICIPLES: FrozenSet[str] = frozenset("required allowed permitted expected obliged supposed entitled encouraged meant".split())
#: Tokens a finite clause's subject sits before: auxiliaries, modals, copulas and normative markers.
_AGENT_MARKERS: FrozenSet[str] = frozenset("""
is are was were be been being am has have had do does did will would shall should must may might can could
""".split()) | frozenset("""
must shall should may can cannot required allowed permitted expected obliged entitled encouraged responsible responsibility
responsibilities obligation obligations duty duties need needs
""".split())
#: Step 4.8T-C3 (D-081): the person asking is a role like any other; the unit must name it (mutation switch).
_SELF_ROLE_EXEMPT = False
_SELF_PRONOUNS: FrozenSet[str] = frozenset("i me my mine we us our".split())
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+(?:\.\d+)?|[.,;:()?!]")


def _verb_base(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    for suffix in ("ied", "ies", "ing", "ed", "es", "s"):
        if suffix == "s" and (w.endswith("ss") or w.endswith("us") or w.endswith("is")):
            break                                                 # assess, focus, basis
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:-len(suffix)] + ("y" if suffix in ("ied", "ies") else "")
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiou" and w[-1] not in "sl":
        w = w[:-1]                                                # submitt -> submit
    return w


@dataclass(frozen=True)
class Relation:
    """What a clause asks about an actor, an action and an object."""

    actor: Optional[str]            # the canonical role asked as the agent (None: not named, or asked)
    party: bool                     # the agent itself is asked ("who ...")
    recipient: bool                 # the asked actor receives the action ("does the chair receive approval")
    action: Optional[str]           # the action (verb base) or None when the verb carries no action
    modifiers: Tuple[str, ...]      # adjectives of the action ("final")
    obj: Tuple[str, ...]            # content words of the object
    target: Tuple[str, ...]         # content words of the destination or beneficiary
    target_role: Optional[str]
    passive: bool
    actor_words: Tuple[str, ...] = ()   # a subject that is no role ("course files", "office hours"): the evidence subject must be it
    target_prep: Optional[str] = None   # the preposition of the target ("to", "from", "for"): its class fixes the direction
    source: Tuple[str, ...] = ()        # a "from" source beside a destination ("from Banner into Blackboard"), D-081
    source_role: Optional[str] = None


class RoleLexicon:
    """Role words of one domain, as (pattern, role id) in priority order, with aliases to canonical ids and the id of
    the person asking ("I", "we")."""

    def __init__(self, roles: Sequence[Tuple[str, str]], aliases: Optional[Dict[str, str]] = None, self_role: Optional[str] = None,
                 phrases: Optional[Callable[[str], List[Tuple[int, int]]]] = None) -> None:
        self.patterns = [(re.compile(p, re.I), r) for p, r in roles]
        self.aliases = dict(aliases or {})
        self.self_role = self_role
        self.phrases = phrases or (lambda text: [])            # the domain's concept phrases: a subject that is no role

    def canonical(self, role: str) -> str:
        return self.aliases.get(role, role)

    def spans(self, text: str) -> List[Tuple[int, int, str]]:
        taken: List[Tuple[int, int, str]] = []
        for pattern, role in self.patterns:
            for m in pattern.finditer(text):
                if any(not (m.end() <= a or m.start() >= b) for a, b, _ in taken):
                    continue
                taken.append((m.start(), m.end(), self.canonical(role)))
        return sorted(taken)

    def roles_in(self, text: str) -> FrozenSet[str]:
        return frozenset(r for _a, _b, r in self.spans(text))

    def head_role(self, text: str) -> Optional[str]:
        """The role of a noun phrase: its last role word ("Deans' Council" is a council)."""
        spans = self.spans(text)
        return spans[-1][2] if spans else None


def _content(words: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for w in words:
        w = w.lower()
        if w.endswith("'s"):
            w = w[:-2]
        if w in _FILLERS or w in _LIGHT_VERBS or w in _POLARITY or w in _DETS or w in _PREPS or len(w) < 2 or w.endswith("ly"):
            continue
        if w not in out:
            out.append(w)
    return tuple(out)


def _split_source(rest: str) -> Tuple[str, ...]:
    """The "from" phrase of ``rest`` when a destination is also named ("from Banner into Blackboard"), else ()."""
    words = _WORD.findall(normalise(rest))
    cut = next((i for i, w in enumerate(words) if w in _CLAUSE_CUT), len(words))
    words = words[:cut]
    if not any(w in _DEST_PREPS for w in words) or "from" not in words:
        return ()
    i = words.index("from") + 1
    phrase: List[str] = []
    while i < len(words) and words[i] not in _PREPS:
        phrase.append(words[i])
        i += 1
    return _content(phrase)


def _split_rest(rest: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], Optional[str]]:
    """(object words, target words, target preposition) of what follows the verb: the object runs to the first
    preposition other than "of"; a destination or beneficiary phrase follows to / into / onto / for / from."""
    words = _WORD.findall(normalise(rest))
    cut = next((i for i, w in enumerate(words) if w in _CLAUSE_CUT), len(words))
    words = words[:cut]                                           # "... if I am unable to attend": a condition is no target
    obj: List[str] = []
    target: List[str] = []
    prep: Optional[str] = None
    i = 0
    while i < len(words) and (words[i] not in _PREPS or words[i] == "of"):
        obj.append(words[i])
        i += 1
    phrases: List[Tuple[str, List[str]]] = []
    while i < len(words):
        if words[i] in _PREPS and words[i] != "of" and words[i] in _DEST_PREPS | {"for", "from"}:
            prep, i, phrase = words[i], i + 1, []
            while i < len(words) and words[i] not in _PREPS:
                phrase.append(words[i])
                i += 1
            phrases.append((prep, phrase))
            continue
        i += 1
    # "from Blackboard to Banner": the destination is the target; a source or a beneficiary only when no destination
    chosen = next((p for p in phrases if p[0] in _DEST_PREPS), phrases[0] if phrases else None)
    if chosen is not None:
        prep, target = chosen[0], chosen[1]
    return _content(obj), _content(target), prep


_ASK_AUX = re.compile(r"\b(?:must|should|shall|do|does|did|can|could|may|might|will|would|is|are|was|were|need|needs|has|have|am)\s+", re.I)
_ASK_CHAIN_WORDS: FrozenSet[str] = frozenset("be required allowed permitted expected obliged supposed entitled encouraged able responsible eligible to for have need".split())
_COPULAR: FrozenSet[str] = frozenset("be is are was were been being am have has had do does did get gets got become becomes became".split())
#: Step 4.8T-C4 (D-082): a subject outside the role lexicon and the concept vocabulary ("the quality office", "the
#: academic advisor") is kept as a generic subject phrase; the evidence clause must have that phrase as its subject.
_GENERIC_SUBJECTS = True
_GENERIC_WORD_BAG = False                                     # mutation switch: True drops a generic-subject relation
_TABLE_ROW_ACTOR_ALLOWED = False                              # mutation switch: True lets a table row settle an actor ask
_PATIENT_PRONOUN_GUARD = True                                 # "They are transferred": the patient is no agent


_AGENTIVE_SUFFIXES = ("or", "ors", "er", "ers", "ant", "ants", "ist", "ists", "ian", "ians")


def _verb_candidate(word: str, nxt: Optional[str]) -> bool:
    """``word`` can be the verb after a generic subject: a known or light verb, a chain word, or a word a determiner
    follows ("remove the"). An agentive noun ("advisor", "coordinator") that a verb follows is part of the subject."""
    if not word.isalpha() or word in _DETS or word in _PREPS or word in _POLARITY or word in _FILLERS and word not in _LIGHT_VERBS:
        return False
    exact_verb = word in _LIGHT_VERBS or word in _RELATION_LIGHT or _verb_base(word) in _KNOWN_VERBS or word in _KNOWN_VERBS
    if (nxt is not None and word.endswith(_AGENTIVE_SUFFIXES) and not exact_verb and nxt.isalpha()
            and nxt not in _DETS and nxt not in _PREPS and nxt not in _POLARITY):
        return False                                              # "advisor remove", "coordinator upload": the noun is the subject
    return (_known_verb(word) or word in _LIGHT_VERBS or word in _RELATION_LIGHT or word in _ASK_CHAIN_WORDS
            or (nxt is not None and nxt in _DETS))


def _actor_and_verb(clause: str, lexicon: RoleLexicon) -> Optional[Tuple[str, str, str]]:
    """(actor role, verb, the rest) of an ask "AUX <actor> [chain] <verb> ...": the actor is the role phrase after the
    auxiliary (up to three modifiers before it, "evening course instructors"), or the person asking; None when no
    role follows the auxiliary."""
    lead = len(clause) - len(clause.lstrip())
    for aux in _ASK_AUX.finditer(clause):
        parsed = _actor_and_verb_after(clause[aux.end():], lexicon, at_start=aux.start() == lead)
        if parsed is not None:
            return parsed
    return None


_SUBORDINATORS: FrozenSet[str] = frozenset("when if while because that which who whom where before after until unless and or once".split())
_KNOWN_VERBS: FrozenSet[str] = frozenset(_ACTION_NOUNS.values()) | frozenset(m for cls in _VERB_CLASSES for m in cls if " " not in m)


def _known_verb(word: str) -> bool:
    base = _verb_base(word)
    return word in _RELATION_LIGHT or base in _KNOWN_VERBS or any(_member_same(m, base) for m in _KNOWN_VERBS)


def _imperative_verb(word: str) -> bool:
    """A bare verb that can open an imperative sentence ("Complete", "Submit", "Use"): a light verb or a known verb in
    its base form (no inflection)."""
    if word in _FILLERS and word not in _LIGHT_VERBS:
        return False
    if word.endswith("s") or word.endswith("ed") or word.endswith("ing"):
        return False
    return word in _LIGHT_VERBS or _verb_base(word) in _KNOWN_VERBS or any(m == word or _member_same(m, word) for m in _KNOWN_VERBS)


def _actor_and_verb_after(after: str, lexicon: RoleLexicon, at_start: bool = False) -> Optional[Tuple[Optional[str], Tuple[str, ...], str, str]]:
    """(actor role, actor words, verb, rest): the actor is the role phrase after the auxiliary, the person asking, or a
    concept phrase that is no role ("course files", "office hours"), whose evidence subject must then be that phrase."""
    spans = lexicon.spans(after)
    phrases = lexicon.phrases(after)
    tokens = list(_TOKEN.finditer(after))

    def role_at(t: "re.Match[str]") -> Optional[str]:
        word = t.group(0).lower()
        if word in _SELF_PRONOUNS:
            return lexicon.self_role
        return next((r for a, b, r in spans if t.start() >= a and t.end() <= b), None)

    def in_phrase(t: "re.Match[str]") -> bool:
        return any(t.start() >= a and t.end() <= b for a, b in phrases)

    i = 0
    while i < len(tokens) and tokens[i].group(0).lower() in _DETS:
        i += 1
    if i >= len(tokens):
        return None
    phrase_then_role = False
    if at_start and role_at(tokens[i]) is None and in_phrase(tokens[i]):
        # "the course coordinator upload", "evening instructors schedule": a role right after the phrase is the actor
        # and the phrase its modifier (D-083); the role path reads such an ask
        span = next((a, b) for a, b in phrases if tokens[i].start() >= a and tokens[i].end() <= b)
        k = i
        while k < len(tokens) and tokens[k].start() >= span[0] and tokens[k].end() <= span[1]:
            k += 1
        phrase_then_role = k < len(tokens) and role_at(tokens[k]) is not None
    if at_start and role_at(tokens[i]) is None and in_phrase(tokens[i]) and not phrase_then_role:
        # "Do course files assess department chairs?" (a yes/no ask, the auxiliary first): the subject is a concept
        # phrase, not a role; a following token that a known verb follows belongs to the subject ("the workload plan approve")
        span = next((a, b) for a, b in phrases if tokens[i].start() >= a and tokens[i].end() <= b)
        start = i
        while i < len(tokens) and tokens[i].start() >= span[0] and tokens[i].end() <= span[1]:
            i += 1
        # the phrase runs on to the first verb candidate ("the academic advisor remove", "the course coordinator upload");
        # a role after the phrase ("the teaching load for lecturers") is never the verb: the role path reads such an ask
        extra = 0
        while (i + 1 < len(tokens) and extra < 3 and tokens[i].group(0).isalpha() and tokens[i].group(0).lower() not in _ASK_CHAIN_WORDS
               and role_at(tokens[i]) is None and role_at(tokens[i + 1]) is None
               and not _verb_candidate(tokens[i].group(0).lower(), tokens[i + 1].group(0).lower())
               and _verb_candidate(tokens[i + 1].group(0).lower(), tokens[i + 2].group(0).lower() if i + 2 < len(tokens) else None)):
            i += 1
            extra += 1
        actor_words = _content(t.group(0) for t in tokens[start:i])
        while i < len(tokens) and tokens[i].group(0).lower() in _ASK_CHAIN_WORDS:
            i += 1
        if i < len(tokens) and role_at(tokens[i]) is not None:
            return None                                           # "the teaching load for lecturers ...": a role is never the verb
        if i >= len(tokens) or not tokens[i].group(0).isalpha() or not actor_words or tokens[i].group(0).lower() in _COPULAR                 or tokens[i].group(0).lower() in _PREPS or tokens[i].group(0).lower() in _DETS or tokens[i].group(0).lower() in _POLARITY:
            return None                                           # a copula carries no action ("are office hours ...")
        return None, actor_words, tokens[i].group(0).lower(), after[tokens[i].end():]
    if not any(role_at(t) is not None for t in tokens[i:i + 4]):
        if not (at_start and _GENERIC_SUBJECTS):
            return None
        # "Does the quality office assess course files?": a subject that is neither a role nor a concept phrase is kept
        # as a generic subject phrase, up to the first verb candidate (D-082); the evidence clause must have it as subject
        j = i + 1
        while j < len(tokens) and j <= i + 6 and not _verb_candidate(tokens[j].group(0).lower(), tokens[j + 1].group(0).lower() if j + 1 < len(tokens) else None):
            if not tokens[j].group(0).isalpha() or tokens[j].group(0).lower() in _SUBORDINATORS:
                return None
            j += 1
        if j >= len(tokens) or j > i + 6:
            return None
        actor_words = _content(t.group(0) for t in tokens[i:j])
        while j < len(tokens) and tokens[j].group(0).lower() in _ASK_CHAIN_WORDS:
            j += 1
        if j < len(tokens) and role_at(tokens[j]) is not None:
            return None                                           # a role is never the verb
        if j < len(tokens) and tokens[j].group(0).lower().endswith("ing") and any(
                tokens[j].start() >= a and tokens[j].end() <= b and len(_TOKEN.findall(after[a:b])) > 1 for a, b in phrases):
            return None                                           # "the baseline teaching load": a gerund heading a concept phrase
        if j >= len(tokens) or not actor_words or not tokens[j].group(0).isalpha() or tokens[j].group(0).lower() in _COPULAR \
                or tokens[j].group(0).lower() in _PREPS or tokens[j].group(0).lower() in _DETS or tokens[j].group(0).lower() in _POLARITY:
            return None
        return None, actor_words, tokens[j].group(0).lower(), after[tokens[j].end():]
    while i < len(tokens) and role_at(tokens[i]) is None:
        if tokens[i].group(0).lower() in _SUBORDINATORS or not tokens[i].group(0).isalpha():
            return None                                           # "the penalty when a student ...": no actor after this auxiliary
        i += 1
    actor = role_at(tokens[i])
    while i < len(tokens) and role_at(tokens[i]) == actor:
        i += 1
    while i < len(tokens) and tokens[i].group(0).lower() in _ASK_CHAIN_WORDS:
        i += 1
    if i >= len(tokens) or actor is None:
        return None
    verb = tokens[i].group(0).lower()
    if not verb.isalpha():
        return None
    for a, b in phrases:
        # "the faculty teaching load", "the regular faculty workload": a gerund heading a multi-word concept phrase, or a
        # concept word that is no verb, is a noun, not the verb (D-082, D-083)
        if tokens[i].start() >= a and tokens[i].end() <= b and (
                (verb.endswith("ing") and len(_TOKEN.findall(after[a:b])) > 1) or not (_known_verb(verb) or verb in _LIGHT_VERBS)):
            return None
    return actor, (), verb, after[tokens[i].end():]


def parse_relation(clause: str, lexicon: RoleLexicon) -> Optional[Relation]:
    """The actor / action / object relation a clause asks about, or None when it asks none (no actor, no "who", no
    passive object): "Must faculty attend all theoretical lectures?", "Does Banner forward grades to Blackboard?",
    "Is the workload plan approved by the dean?", "Who reviews course files?", "Does the chair give final approval
    of the workload plan?"."""
    text = clause.strip()
    party = bool(_PARTY_ASK.search(text))
    m = None if party else _PASSIVE_ASK.match(text)
    if m:
        obj, _t, _p = _split_rest(m.group("obj"))
        actor = lexicon.head_role(m.group("agent"))
        actor_words = () if actor is not None else _content(_WORD.findall(normalise(m.group("agent"))))
        rest = m.group("rest")
        # "by the instructor to the Registrar": a destination, beneficiary or source after the agent is the target;
        # "by faculty members of the department" carries none
        _o, target, prep = _split_rest("" if re.match(r"\s*of\b", rest) else rest)
        verb = m.group("verb").lower()
        return Relation(actor=actor, party=False, recipient=False, action="receive" if verb in _RECEIVE_VERBS else _verb_base(verb),
                        modifiers=(), obj=obj, target=target, target_role=lexicon.head_role(" ".join(target)) if target else None,
                        passive=True, actor_words=actor_words, target_prep=prep)
    m = None if party else (_PASSIVE_NOAGENT_ASK.match(text) or _PASSIVE_SHORT_ASK.match(text))
    if m:
        obj, _t, _p = _split_rest(m.group("obj"))
        rest = m.group("rest")
        # "sent to Blackboard by Banner": an agent after the destination is the actor (D-082)
        agent = _TRAILING_AGENT.search(rest)
        actor = lexicon.head_role(agent.group("agent")) if agent else None
        actor_words = _content(_WORD.findall(normalise(agent.group("agent")))) if agent and actor is None else ()
        if agent:
            rest = rest[:agent.start()]
        _o, target, prep = _split_rest(rest)
        if not target and not obj:
            return None
        verb = m.group("verb").lower()
        source = _split_source(rest)
        return Relation(actor=actor, party=False, recipient=False, action="receive" if verb in _RECEIVE_VERBS else _verb_base(verb), modifiers=(),
                        obj=obj, target=target, target_role=lexicon.head_role(" ".join(target)) if target else None, passive=True, target_prep=prep,
                        actor_words=actor_words, source=source, source_role=lexicon.head_role(" ".join(source)) if source else None)
    actor: Optional[str] = None
    actor_words: Tuple[str, ...] = ()
    if party:
        m = _PARTY_REL.match(text)
        if not m:
            return None
        verb, rest = m.group("verb").lower(), m.group("rest")
    else:
        parsed = _actor_and_verb(text, lexicon)
        if parsed is None:
            return None
        actor, actor_words, verb, rest = parsed
    if verb in _FILLERS and verb not in _LIGHT_VERBS:
        # "correct", "cover": a filler is the action only in verb position with an object after it
        rest_words = [w for w in _WORD.findall(normalise(rest)) if w not in _DETS]
        if not rest_words or rest_words[0] in _PREPS or rest_words[0] in _FILLERS or rest_words[0] in _POLARITY or not rest_words[0].isalpha():
            return None
    action: Optional[str] = None
    modifiers: Tuple[str, ...] = ()
    recipient = False
    if verb in _RELATION_LIGHT:
        words = _WORD.findall(normalise(rest))
        head = next((i for i, w in enumerate(words) if w not in _DETS), None)
        noun_index = next((i for i, w in enumerate(words[:5]) if w in _ACTION_NOUNS), None) if head is not None else None
        if noun_index is not None:
            action = _ACTION_NOUNS[words[noun_index]]
            modifiers = tuple(w for w in words[:noun_index] if w in _ACTION_MODIFIERS)
            recipient = verb in _RECIPIENT_VERBS
            rest = " ".join(words[noun_index + 1:])
            if rest.startswith("of ") or rest.startswith("for ") or rest.startswith("on "):
                rest = rest.split(" ", 1)[1] if " " in rest else ""
        elif verb in _RECEIVE_VERBS:
            action = "receive"                                    # "receive the grade report from the Registrar": a real relation
    else:
        action = _verb_base(verb)
    obj, target, prep = _split_rest(rest)
    source = _split_source(rest)
    return Relation(actor=actor, party=party, recipient=recipient, action=action, modifiers=modifiers, obj=obj, target=target,
                    target_role=lexicon.head_role(" ".join(target)) if target else None, passive=False, actor_words=actor_words,
                    target_prep=prep, source=source, source_role=lexicon.head_role(" ".join(source)) if source else None)


class _Unit:
    """A tokenised evidence unit with the role of every token."""

    def __init__(self, text: str, lexicon: RoleLexicon) -> None:
        self.text = text
        self.self_role = lexicon.self_role
        self.tokens = [(m.group(0), m.group(0).lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        self.role = [None] * len(self.tokens)
        for a, b, role in lexicon.spans(text):
            for i, (_w, _l, s0, e0) in enumerate(self.tokens):
                if s0 >= a and e0 <= b:
                    self.role[i] = role

    def low(self, i: int) -> str:
        return self.tokens[i][1]

    def subject_before(self, index: int) -> Optional[str]:
        """The role that is the grammatical subject of the verb at ``index``: scanning back over the modal chain, past a
        relative pronoun to its antecedent, a role word that is not the object of a preposition; None when the clause
        names no role before the verb."""
        j = index - 1
        while j >= 0 and (self.low(j) in _CHAIN_TOKENS or self.low(j) in _BE_FORMS or self.low(j).endswith("ly")):
            j -= 1
        if j >= 0 and self.low(j) in _RELATIVES:
            # "the blackboard which will forward it": the antecedent is the phrase right before the relative pronoun,
            # whatever preposition introduced that phrase
            for k in range(j - 1, max(-1, j - 4), -1):
                if self.role[k] is not None:
                    return self.role[k]
                if self.low(k) in _BOUNDARIES and not (self.low(k) == "," and k == j - 1):
                    break                                         # ", which forwards them": the comma before the pronoun is transparent
            return None
        if j >= 0 and self.low(j) in _PERSON_PRONOUNS:
            return self._antecedent(j)
        steps = 0
        while j >= 0 and steps < 14:
            word = self.low(j)
            if word in "()":
                j -= 1
                continue
            if word in _BOUNDARIES:
                return None
            if (word in _CAUSATIVES and j < index - 1 and _is_participle(self.low(index)) and self.low(j + 1) not in _CHAIN_TOKENS
                    and not self.low(j + 1).endswith("ly")):
                return _OTHER_SUBJECT                             # "have their answer sheets corrected": someone else acts
            if word in _PERSON_PRONOUNS:
                return self._antecedent(j)
            if self.role[j] is not None:
                k = j - 1
                while k >= 0 and (self.role[k] == self.role[j] or self.low(k) in _DETS or self.low(k).endswith("'s") or self.low(k) in "()"):
                    k -= 1
                while k >= 0 and self.low(k).endswith("ly"):
                    k -= 1
                if k >= 0 and self.low(k) in _PREPS:
                    j = k - 1                                     # a role inside a prepositional phrase is not the subject
                    steps += 1
                    continue
                return self.role[j]
            j -= 1
            steps += 1
        return None

    def _antecedent(self, index: int) -> Optional[str]:
        """"if an instructor is unable ..., he/she should inform students": the nearest role before the pronoun at
        ``index``, across clause boundaries."""
        for k in range(index - 1, max(-1, index - 24), -1):
            if self.role[k] is not None:
                return self.role[k]
        return None

    def subject_text_before(self, index: int) -> str:
        """The text of the clause before the verb at ``index`` (its subject phrase, up to a boundary)."""
        j = index - 1
        while j >= 0 and (self.low(j) in _CHAIN_TOKENS or self.low(j) in _BE_FORMS or self.low(j).endswith("ly")):
            j -= 1
        end = j
        while j >= 0 and self.low(j) not in _BOUNDARIES and end - j < 12:
            j -= 1
        if end < 0 or end < j + 1:
            return ""
        return self.text[self.tokens[j + 1][2]:self.tokens[end][3]]

    def is_imperative(self) -> bool:
        """The unit opens with a bare verb addressed to the reader ("Complete training modules ...")."""
        for i, (_w, low, _s, _e) in enumerate(self.tokens):
            if low in _DETS or low.isdigit() or len(low) == 1 or low in ("o", "•") or low == ".":
                continue
            return (self.role[i] is None and _imperative_verb(low) and (i + 1 >= len(self.tokens) or (
                self.low(i + 1) not in _AGENT_MARKERS and self.low(i + 1) not in _PREPS and not self.low(i + 1).endswith("ly"))))
        return False

    def sentence_range(self, index: int) -> Tuple[int, int]:
        """The token range [start, end) of the sentence that contains ``index`` (D-082): actor evidence is local to it."""
        start = index
        while start > 0 and self.low(start - 1) not in _SENTENCE_ENDS:
            start -= 1
        end = index
        while end < len(self.tokens) and self.low(end) not in _SENTENCE_ENDS:
            end += 1
        return start, end

    def _patient_marker(self, i: int) -> bool:
        """"They are transferred", "Students are informed by": the subject of a copula followed by a participle is the
        patient, not an agent, unless the participle is a chain word ("are required to attend")."""
        if not _PATIENT_PRONOUN_GUARD or self.low(i) not in _BE_FORMS:
            return False
        k = i + 1
        while k < len(self.tokens) and self.low(k).endswith("ly"):
            k += 1
        return k < len(self.tokens) and _is_participle(self.low(k)) and self.low(k) not in _CHAIN_PARTICIPLES

    def agent_roles(self, start: int = 0, end: Optional[int] = None) -> FrozenSet[str]:
        """The roles that are grammatical subjects in the token range (the whole unit by default): the subject read
        before an auxiliary, modal, copula or normative marker, a passive agent after "by", and a role that opens a
        sentence. A role named only after a preposition or as an object ("distributed to students", "arranged with the
        instructor"), or the patient of a passive ("they are transferred"), is not among them (D-081, D-082)."""
        end = len(self.tokens) if end is None else end
        found = set()
        for i in range(start, end):
            low = self.low(i)
            if low in _AGENT_MARKERS:
                if self._patient_marker(i):
                    continue
                subject = self.subject_before(i)
                if subject is not None and subject != _OTHER_SUBJECT:
                    found.add(subject)
            elif low == "by":
                agent = self.role_after(i - 1, frozenset({"by"}), limit=1)   # "prepared by faculty members": a passive agent
                if agent is not None:
                    found.add(agent)
        first = True
        for i in range(start, end):
            word, low = self.tokens[i][0], self.low(i)
            if low in ".;:!?" or (low.isdigit() and i + 1 < len(self.tokens) and self.low(i + 1) == "."):
                first = True
                continue
            if not first:
                continue
            if low in _DETS or low.isdigit() or len(low) == 1 or low in ("o", "•"):
                continue
            if self.role[i] is not None:
                k = i
                while k < len(self.tokens) and self.role[k] == self.role[i]:
                    k += 1
                if not (k < len(self.tokens) and self._patient_marker(k)):
                    found.add(self.role[i])                       # "Students are informed by ...": a patient opens no agent
            elif low in _PERSON_PRONOUNS and i + 1 < len(self.tokens) and not self._patient_marker(i + 1):
                antecedent = self._antecedent(i)                  # "They submit them to the chair": an actor pronoun
                if antecedent is not None:
                    found.add(antecedent)
            elif self.self_role is not None and _imperative_verb(low) and (i + 1 >= len(self.tokens) or (
                    self.low(i + 1) not in _AGENT_MARKERS and self.low(i + 1) not in _PREPS and not self.low(i + 1).endswith("ly"))):
                found.add(self.self_role)                         # "Complete training modules ...": an imperative addresses the reader
            first = False
        return frozenset(found)

    def role_after(self, index: int, preps: FrozenSet[str], limit: int = 7) -> Optional[str]:
        """The head role of the phrase after one of ``preps`` within ``limit`` tokens after ``index``."""
        for j in range(index + 1, min(len(self.tokens), index + 1 + limit)):
            if self.low(j) in _BOUNDARIES:
                return None
            if self.low(j) in preps:
                last = None
                for k in range(j + 1, min(len(self.tokens), j + 8)):
                    if self.low(k) in _BOUNDARIES or (self.low(k) in _PREPS and self.low(k) != "of"):
                        break
                    if self.role[k] is not None:
                        last = self.role[k]
                return last
        return None

    def window(self, start: int, limit: int, stop_at_preps: bool = True) -> Tuple[str, Tuple[int, int]]:
        """The text of up to ``limit`` tokens after ``start`` inside the clause (stopping at a preposition other than
        "of" when asked; a parenthesis is transparent: "the registration system (Banner)"), and the token range."""
        j = start + 1
        end = j
        while end < len(self.tokens) and end - j < limit:
            word = self.low(end)
            if (word in _BOUNDARIES and word not in "()") or (stop_at_preps and word in _PREPS and word != "of"):
                break
            end += 1
        if end <= j:
            return "", (j, j)
        return self.text[self.tokens[j][2]:self.tokens[end - 1][3]], (j, end)

    def phrase_after_prep(self, start: int, preps: FrozenSet[str], limit: int = 12) -> Optional[str]:
        for j in range(start + 1, min(len(self.tokens), start + 1 + limit)):
            if self.low(j) in _BOUNDARIES:
                return None
            if self.low(j) in preps:
                text, _span = self.window(j, 7, stop_at_preps=True)
                return text
        return None


def _is_participle(word: str) -> bool:
    return word.endswith("ed") or word.endswith("en") or word in _IRREGULAR


_VERB_CUES: FrozenSet[str] = frozenset("will would shall should must may can could might to not and or also then".split())
_NOUN_CUES_AFTER: FrozenSet[str] = frozenset("of by from with process rests lies is was are were".split())


def _matches_action(word: str, action: str, prev: str = "", nxt: str = "", prev_role: bool = False) -> Optional[str]:
    """"verb" when ``word`` is a verb form of ``action`` or of a member of its class, "noun" when it is a noun naming
    that action, else None. A word that is both ("review", "approval") is read from its neighbours: after a modal, "to"
    or a coordination it is the verb; after a determiner or an adjective, or before "of", "by" or "process", the noun."""
    if word in _IRREGULAR:
        base = _IRREGULAR[word]                                   # "held" is a form of "hold", "sent" of "send" (D-082)
        if _member_same(base, action) or _member_same(action, base) or any(_member_same(m, base) for cls in _classes_of(action) for m in cls if " " not in m):
            return "verb"
    if word in _ACTION_NOUNS:
        named = _ACTION_NOUNS[word]
        same_action = (_member_same(_stem(named), action) or _member_same(_stem(action), named) or any(
            _member_same(m, named) and _member_same(m, action) for cls in _VERB_CLASSES for m in cls if " " not in m))
        if not same_action:
            return None
        verb_like = _verb_form(action, word) or (action.endswith("e") and _verb_form(action[:-1], word))
        if verb_like and (prev in _VERB_CUES or prev.endswith("ly") or prev_role) and nxt not in _NOUN_CUES_AFTER:
            return "verb"                                         # "the chair reviews teaching loads": a role before it makes it the verb
        return "noun"
    if _verb_form(action, word) or (action.endswith("e") and _verb_form(action[:-1], word)):
        return "verb"
    for cls in _classes_of(action):
        for member in cls:
            if " " not in member and _verb_form(member, word):
                return "verb"
    return None


class RelationChecker:
    """Checks a :class:`Relation` against one evidence unit with the domain's roles and concepts."""

    def __init__(self, lexicon: RoleLexicon, concepts_fn: Callable[[str], object]) -> None:
        self.lexicon = lexicon
        self.concepts_fn = concepts_fn

    def _object_stated(self, obj: Sequence[str], text: str) -> bool:
        """The asked object is stated in ``text``: as its concept when the phrase names one ("workload allocation" is
        the teaching load), else by its head word."""
        if not obj:
            return True
        own = frozenset(getattr(self.concepts_fn(" ".join(obj)), "vocab", frozenset()))
        if own:
            have = frozenset(getattr(self.concepts_fn(text), "vocab", frozenset()))
            return own <= have
        return self._stated(obj[-1:], text)

    def _stated(self, words: Sequence[str], text: str) -> bool:
        if not words:
            return True
        have = frozenset(getattr(self.concepts_fn(text), "vocab", frozenset()))
        unit_words = _WORD.findall(normalise(text))
        for w in words:
            own = frozenset(getattr(self.concepts_fn(w), "vocab", frozenset()))
            if own and own <= have:
                continue
            if any(_same(w, u) for u in unit_words):
                continue
            return False
        return True

    def occurrences(self, rel: Relation, unit: _Unit) -> List[Dict[str, object]]:
        found: List[Dict[str, object]] = []
        if rel.action is None:
            return found
        for i, (_w, low, _s, _e) in enumerate(unit.tokens):
            kind = _matches_action(low, rel.action, unit.low(i - 1) if i > 0 else "", unit.low(i + 1) if i + 1 < len(unit.tokens) else "",
                                   prev_role=i > 0 and unit.role[i - 1] is not None)
            if kind is None:
                continue
            occ: Dict[str, object] = {"index": i, "kind": kind, "subject": None, "voice": "active", "object_ok": True, "target_ok": True,
                                      "modifiers_ok": True}
            if rel.modifiers:
                before = [unit.low(k) for k in range(max(0, i - 3), i)]
                occ["modifiers_ok"] = all(mod in before for mod in rel.modifiers)
            if kind == "noun":
                occ["voice"] = "nominal"
                agent = unit.role_after(i, frozenset({"of", "by", "from", "with"}), limit=3)
                if agent is None and i >= 1 and unit.low(i - 1).endswith("'s") and unit.role[i - 1] is not None:
                    agent = unit.role[i - 1]
                if agent is None and i >= 2 and unit.role[i - 2] is not None and unit.low(i - 1).endswith("'s"):
                    agent = unit.role[i - 2]
                occ["subject"] = agent
                occ["object_ok"] = self._object_stated(rel.obj, unit.text)
                occ["target_ok"] = not rel.target or self._stated(rel.target, unit.text)
                found.append(occ)
                continue
            passive = False
            be_index = None
            if _is_participle(low):
                k = i - 1
                while k >= 0 and unit.low(k).endswith("ly"):
                    k -= 1
                if k >= 0 and unit.low(k) in _BE_FORMS:
                    passive, be_index = True, k
                elif i + 1 < len(unit.tokens) and unit.low(i + 1) == "by":
                    passive = True
                else:
                    # "shall be distributed to students ... and uploaded on Blackboard": a participle coordinated with an
                    # earlier passive of the same clause shares its patient
                    k = i - 1
                    while k >= 0 and unit.low(k) not in _BOUNDARIES and i - k <= 14:
                        if unit.low(k) in _BE_FORMS and k + 1 < i and _is_participle(unit.low(k + 1)):
                            passive, be_index = True, k
                            break
                        k -= 1
            if passive:
                occ["voice"] = "passive"
                occ["subject"] = unit.role_after(i, frozenset({"by"}), limit=5)
                anchor = be_index if be_index is not None else i
                patient_text = unit.text[unit.tokens[max(0, anchor - 9)][2]:unit.tokens[anchor][2]]
                occ["patient"] = patient_text
                occ["object_ok"] = self._object_stated(rel.obj, patient_text)   # the patient only (D-081): never the whole unit
                if occ["subject"] is None and rel.actor is not None and unit.role_after(i, frozenset({"to", "for"}), limit=4) == rel.actor:
                    occ["subject"] = _OTHER_SUBJECT                # "distributed to students": the asked actor receives, so someone else acts
                if rel.actor_words and occ["subject"] is None and not self._stated(rel.actor_words[-1:], patient_text):
                    occ["subject"] = _OTHER_SUBJECT                # the patient is not the asked subject phrase
            else:
                occ["subject"] = unit.subject_before(i)
                if occ["subject"] is None and low.endswith("ing"):
                    # "Number of students receiving incompletes or forfeiting the right": the role before an active
                    # participle is its logical subject, whatever preposition introduced that role (D-081)
                    for k in range(i - 1, max(-1, i - 8), -1):
                        if unit.low(k) in _BOUNDARIES:
                            break
                        if unit.role[k] is not None:
                            occ["subject"] = unit.role[k]
                            break
                window_text, span = unit.window(i, 12, stop_at_preps=True)
                first = unit.low(span[0]) if span[0] < len(unit.tokens) else ""
                occ["object_ok"] = (first in _PRONOUN_OBJECTS) or self._object_stated(rel.obj, window_text)
                if rel.actor_words and occ["subject"] is None and not self._stated(rel.actor_words[-1:], unit.subject_text_before(i)):
                    occ["subject"] = _OTHER_SUBJECT                # "course files assess ...": the clause's subject is another phrase
            if rel.target:
                # the evidence may place a destination with "into", "on", "in" or "via" ("uploaded on Blackboard"); a source
                # ("from") and a beneficiary ("for") are their own classes, so direction is kept
                phrase = unit.phrase_after_prep(i, _PREP_CLASSES.get(rel.target_prep or "to", _PREP_CLASSES["to"]))
                if rel.target_role is not None:
                    # a role target is stated by that role in the destination phrase ("to the registration system (Banner)"
                    # states Banner and the registration system, not the Registrar)
                    occ["target_ok"] = bool(phrase) and rel.target_role in self.lexicon.roles_in(phrase)
                else:
                    occ["target_ok"] = bool(phrase) and self._stated(rel.target, phrase)
            if rel.source and occ["target_ok"]:
                # "from Banner into Blackboard": the source is stated after "from", or it is the agent of the occurrence
                # ("Blackboard forwards it to Banner" states "from Blackboard to Banner"); never elsewhere (D-081)
                from_phrase = unit.phrase_after_prep(i, frozenset({"from"}))
                if rel.source_role is not None:
                    occ["target_ok"] = (bool(from_phrase) and rel.source_role in self.lexicon.roles_in(from_phrase)) or occ["subject"] == rel.source_role
                else:
                    occ["target_ok"] = bool(from_phrase) and self._stated(rel.source, from_phrase)
            found.append(occ)
        return found

    def stated(self, rel: Relation, unit_text: str) -> bool:
        """The relation is stated in the unit: an occurrence of the action whose subject is the asked actor (any named
        party for a "who" ask), with the asked object, direction and modifiers; an occurrence that names no subject
        counts only when no occurrence names a different one."""
        unit = _Unit(unit_text, self.lexicon)
        if rel.recipient:
            return self._recipient_stated(rel, unit)
        occs = self.occurrences(rel, unit)
        if not occs:
            return False
        complete = [o for o in occs if o["object_ok"] and o["target_ok"] and o["modifiers_ok"]]
        if rel.party:
            return any(o["subject"] is not None and o["subject"] != _OTHER_SUBJECT for o in complete)
        if rel.actor is None:
            # no role asked: a concept-phrase subject must be the clause's own subject; otherwise any complete occurrence
            if any(o["subject"] is not None for o in occs) and rel.actor_words:
                return False
            return bool(complete)
        if any(o["subject"] == rel.actor for o in complete):
            return True
        if any(o["subject"] is not None and o["subject"] != rel.actor for o in occs):
            return False                                          # a different named subject
        # a subject-less occurrence counts only for a role that is a grammatical subject in the SENTENCE of the occurrence
        # (D-082): a role named only as a recipient or object ("distributed to students ... and uploaded") never becomes
        # the agent, and an actor of another sentence never licenses this action ("Faculty enter grades. They are
        # transferred to Banner.")
        return any(o["subject"] is None and rel.actor in unit.agent_roles(*unit.sentence_range(o["index"])) for o in complete)

    def actor_is_agent(self, actor: Optional[str], unit: _Unit) -> bool:
        """The asked actor is a grammatical subject somewhere in the unit (D-081)."""
        return actor is not None and actor in unit.agent_roles()

    def actor_is_agent_near(self, actor: Optional[str], unit: _Unit, words: Sequence[str]) -> bool:
        """The asked actor is a grammatical subject in a sentence that states the object words (D-082); with no object
        words, anywhere in the unit."""
        if actor is None:
            return False
        if not words:
            return actor in unit.agent_roles()
        head = words[-1]
        for i, (_w, low, _s, _e) in enumerate(unit.tokens):
            if _same(head, low) and actor in unit.agent_roles(*unit.sentence_range(i)):
                return True
        return False

    def _recipient_stated(self, rel: Relation, unit: _Unit) -> bool:
        """"Does the chair receive approval": the actor as the recipient of the action noun ("approval is given to the
        chair", "the chair receives approval"), never as its agent."""
        noun_indices = [i for i, (_w, low, _s, _e) in enumerate(unit.tokens) if low in _ACTION_NOUNS and _ACTION_NOUNS[low] == rel.action]
        for i in noun_indices:
            recipient = unit.role_after(i, frozenset({"to", "for"}), limit=4)
            if recipient is not None and (rel.actor is None or recipient == rel.actor):
                return True
            for k in range(max(0, i - 4), i):
                if unit.low(k) in _RECIPIENT_VERBS:
                    subject = unit.subject_before(k)
                    if subject is not None and (rel.actor is None or subject == rel.actor):
                        return True
        return False

    def agent_roles(self, unit_text: str) -> FrozenSet[str]:
        return _Unit(unit_text, self.lexicon).agent_roles()

    def normative_subjects(self, unit_text: str) -> List[Optional[str]]:
        """The subject role of every normative marker in the unit (None for an impersonal rule)."""
        unit = _Unit(unit_text, self.lexicon)
        return [unit.subject_before(i) for i, (_w, low, _s, _e) in enumerate(unit.tokens) if low in _NORMATIVE_TOKENS]


@dataclass(frozen=True)
class AskAssessment:
    """What a clause asks, as the gate reads it."""

    shape: str                                   # GENERAL, SPECIFIC_RECOGNISED or SPECIFIC_UNKNOWN
    concepts: FrozenSet[str]                     # domain concepts of the clause that the answering unit must share
    terms: Tuple[str, ...]                       # remaining content words, letters and numbers the unit must state
    yes_no: bool                                 # a yes/no proposition
    deontic: bool                                # ... about a permission, obligation or prohibition
    party: bool                                  # asks who (a named party stated with the asked action)
    action: Optional[str]                        # the asked action of a party ask
    value: bool                                  # asks for a value: the unit must carry a number
    details: Tuple[Detail, ...]                  # the details to add to the clause's requested details
    roles: FrozenSet[str] = frozenset()          # roles the clause names (the unit must name them too)
    relation: Optional[Relation] = None          # the actor / action / object relation asked, if any

    @property
    def label(self) -> str:
        return ", ".join(list(self.terms) + sorted(self.concepts))


class AskGate:
    """The gate for one domain. ``concepts_fn(text)`` returns an object with ``anchors`` and ``vocab`` frozensets;
    ``residue_fn(text)`` returns the text with every concept span blanked; ``generic`` are the domain's generic words;
    ``party`` matches a named party; ``implies`` maps an anchor to the base word that also states it; ``loose`` are
    concepts verified elsewhere (a calendar term, a detail family's own word) and not demanded of the unit;
    ``table_row`` tells a terse table line, which must name at least one of the ask's specifics."""

    def __init__(self, concepts_fn: Callable[[str], object], residue_fn: Callable[[str], str], *, generic: FrozenSet[str],
                 party: "re.Pattern[str]", implies: Optional[Dict[str, str]] = None, loose: FrozenSet[str] = frozenset(),
                 table_row: Optional[Callable[[str], bool]] = None, roles: Sequence[Tuple[str, str]] = (),
                 role_aliases: Optional[Dict[str, str]] = None, self_role: Optional[str] = None,
                 role_concepts: FrozenSet[str] = frozenset(), calendar_row: Optional[Callable[[str], bool]] = None,
                 phrases: Optional[Callable[[str], List[Tuple[int, int]]]] = None) -> None:
        self.concepts_fn = concepts_fn
        self.residue_fn = residue_fn
        self.generic = frozenset(w.lower() for w in generic)
        self.party = party
        self.implies = dict(implies or {})
        self.loose = frozenset(loose)
        self.table_row = table_row or (lambda unit: False)
        self.calendar_row = calendar_row or (lambda unit: False)
        # Step 4.8T-C2 (D-080): the domain's role words, the role of the person asking, and the concepts that stand for
        # roles (checked by role, never by a shared concept: a dean is not a chair, a registrar is not a registration)
        self.lexicon = RoleLexicon(roles, role_aliases, self_role, phrases)
        self.relations = RelationChecker(self.lexicon, concepts_fn)
        self.role_concepts = frozenset(role_concepts)

    # ---- reading the clause ----------------------------------------------------------------------------------------
    def key_terms(self, clause: str) -> Tuple[str, ...]:
        """Content words, grade letters and numbers of the clause outside its concepts, fillers, light verbs,
        polarity words, detail-family words and the domain's generic words."""
        residue = self.residue_fn(clause)
        contents = bool(_CONTENTS_ASK.search(clause))
        seen: List[str] = []
        for word in _words(residue):
            if word.endswith("'s"):
                word = word[:-2]
            if word in _KEEP:
                pass
            elif (word in _FILLERS or word in _LIGHT_VERBS or word in _POLARITY or word in _LOOSE_TERMS or word in self.generic
                    or (contents and word in _CONTENT_WORDS) or (len(word) <= 2 and not word.isdigit())):
                continue                                          # a two-letter leftover of a blanked concept is no specific
            if word not in seen:
                seen.append(word)
        for letter in _GRADE_LETTER.findall(clause):
            if letter.lower() not in seen:
                seen.append(letter.lower())
        return tuple(seen)

    def clause_concepts(self, clause: str) -> FrozenSet[str]:
        c = self.concepts_fn(clause)
        return frozenset(getattr(c, "vocab", frozenset())) - self.loose - self.role_concepts

    def clause_roles(self, clause: str) -> FrozenSet[str]:
        """The roles a clause names, other than the person asking."""
        return self.lexicon.roles_in(clause)

    def assess(self, clause: str) -> AskAssessment:
        text = clause.strip()
        concepts = self.clause_concepts(text)
        terms = self.key_terms(text)
        party = bool(_PARTY_ASK.search(text))
        action = None
        if party:
            m = _PARTY_ACTION.search(text)
            if m and m.group(1).lower() not in _FILLERS and m.group(1).lower() not in _LIGHT_VERBS:
                action = m.group(1).lower()
            terms = tuple(t for t in terms if action is None or not _same(_stem(t), _stem(action)))
        yes_no = bool(_YESNO_ASK.search(text)) and not party
        deontic = yes_no and bool(_DEONTIC_ASK.search(text))
        value = bool(_VALUE_ASK.search(text)) and not party and not yes_no
        contents = bool(_CONTENTS_ASK.search(text))
        specific_wh = bool(_SPECIFIC_WH.search(text))
        unknown_wh = bool(_UNKNOWN_WH.search(text))
        overview = (bool(_OVERVIEW.search(text)) or bool(_WHAT_ASK.search(text))) and not value
        questionish = bool(_QUESTIONISH.search(text))
        if unknown_wh or len(_words(text)) < 2:
            shape = SPECIFIC_UNKNOWN
        elif party or yes_no or value or specific_wh:
            shape = SPECIFIC_RECOGNISED if (terms or concepts or action) else SPECIFIC_UNKNOWN
        elif overview or not questionish:
            shape = GENERAL if not terms else SPECIFIC_RECOGNISED
        else:
            shape = SPECIFIC_UNKNOWN
        relation = parse_relation(text, self.lexicon) if shape == SPECIFIC_RECOGNISED else None
        # the roles the unit must name by role: the actor and the target of the asked relation (a dean is not a chair);
        # a role in a condition ("the penalty when a student is caught cheating") is not demanded
        roles = frozenset(r for r in ((relation.actor, relation.target_role) if relation else ())
                          if r and (not _SELF_ROLE_EXEMPT or r != self.lexicon.self_role))
        details: List[Detail] = []
        if shape == SPECIFIC_UNKNOWN:
            details.append(Detail("ask_unknown", "the specific ask stated in the evidence (the ask is not recognised)", lambda unit: False))
        else:
            check = self._grounding_check(concepts, terms, deontic, party, action, value, contents, roles=roles, relation=relation,
                                          clause_text=text)
            what = ", ".join(list(terms) + sorted(concepts) + sorted(roles)) or "the topic"
            if contents:
                what += "; the contents stated"
            if relation is not None and (relation.actor or relation.party):
                what += "; %s %s%s" % ("the party that" if relation.party else relation.actor, relation.action or "acts",
                                       (" " + " ".join(relation.obj)) if relation.obj else "")
            details.append(Detail("grounding", "the asked specifics stated with the answer (%s)" % what, check))
        return AskAssessment(shape=shape, concepts=concepts, terms=terms, yes_no=yes_no, deontic=deontic, party=party, action=action,
                             value=value, details=tuple(details), roles=roles, relation=relation)

    # ---- checking a unit -------------------------------------------------------------------------------------------
    def action_stated(self, action: str, unit: str, text: str, words: Sequence[str]) -> bool:
        """The asked action of a party ask is stated: as a verb form (``verb_stated``), or as its noun bound to a
        party ("approval of the department council", "the chair's review")."""
        if verb_stated(action, text, words):
            return True
        base = action.lower()
        for suffix in ("ing", "ed", "es", "s"):
            if base.endswith(suffix) and len(base) - len(suffix) >= 3:
                base = base[:-len(suffix)]
                break
        stem = base[:-1] if base.endswith("e") else base
        noun = re.compile(r"\b(?:the\s+|a\s+|an\s+)?" + re.escape(stem) + r"\w{0,2}(?:" + "|".join(_NOUN_SUFFIXES) + r")s?\b\s+(?:of|by|from)\s+"
                          r"(?:the\s+|a\s+|an\s+|its\s+|their\s+)?", re.I)
        for m in noun.finditer(unit):
            if self.party.match(unit, m.end()):
                return True
        owner = re.compile(r"\b(?:" + self.party.pattern + r")(?:'s|s')?\s+(?:own\s+)?" + re.escape(stem) + r"\w{0,2}(?:" + "|".join(_NOUN_SUFFIXES) + r")s?\b",
                           re.I)
        return bool(owner.search(unit))

    def concepts_stated(self, concepts: FrozenSet[str], unit: str) -> FrozenSet[str]:
        """The clause concepts the unit states (an anchor also by its base word)."""
        have = frozenset(getattr(self.concepts_fn(unit), "vocab", frozenset()))
        return frozenset(c for c in concepts if c in have or self.implies.get(c) in have)

    def _grounding_check(self, concepts: FrozenSet[str], terms: Tuple[str, ...], deontic: bool, party: bool, action: Optional[str],
                         value: bool, contents: bool = False, roles: FrozenSet[str] = frozenset(),
                         relation: Optional[Relation] = None, clause_text: str = "") -> Callable[[str], bool]:
        ask_freqs = frequencies_of(clause_text) if clause_text else frozenset()
        ask_facts = _ask_numeric_facts(clause_text) if clause_text else []

        def check(unit: str, relation: Optional[Relation] = relation) -> bool:
            stated_in = strip_exclusions(unit)                   # what an exclusion phrase names is not stated
            text = normalise(stated_in)
            words = _WORD.findall(text)
            stated_concepts = self.concepts_stated(concepts, stated_in)
            unit_freqs = frequencies_of(stated_in)
            # a kept frequency word is stated by its unit ("annually" by "each year"); "periodically" states no unit
            stated_terms = [t for t in terms if term_stated(t, text, words) or (_freq_term_unit(t) not in (None, "periodic") and
                                                                                 _freq_term_unit(t) in unit_freqs)]
            table_row = self.table_row(unit)
            if _GENERIC_WORD_BAG and relation is not None and relation.actor is None and relation.actor_words:
                relation = None                                   # mutation only: the pre-D-082 word bag
            if table_row and relation is not None and (relation.actor or relation.actor_words or relation.party) and not _TABLE_ROW_ACTOR_ALLOWED:
                return False                                      # a table row names no actor: it settles no ask that names one (D-082)
            if ask_facts and _NUMERIC_BINDING:
                # D-083: every asked numeric fact is stated as ONE local fact: the same value qualifying the same object,
                # with the asked period in the same clause, and a named subject of that clause that is the asked actor
                unit_facts = numeric_facts(text)
                nunit = _Unit(text, self.lexicon)
                actor = relation.actor if relation is not None else None
                ask_category = _category_of(clause_text, self.lexicon)
                excluded: Set[int] = set()
                if relation is not None and relation.action is not None:
                    # a fact of another independent clause states nothing about the asked action ("re-marking requests
                    # are filed within 7 days" states no 48-hour submission)
                    excluded = _excluded_segments(self.relations, relation, nunit, text)
                if actor is None and (relation is None or not relation.actor_words):
                    # "Does the chair have 15 teaching hours?": no relation, but the clause names one role
                    named = self.lexicon.roles_in(clause_text)
                    if len(named) == 1:
                        actor = next(iter(named))
                    elif not named and self.lexicon.self_role and any(w in _SELF_PRONOUNS for w in _words(clause_text)):
                        actor = self.lexicon.self_role
                for af in ask_facts:
                    found = False
                    asked_freqs = af["freqs"] or (ask_freqs if _NUMERIC_FREQ_BINDING else frozenset())
                    for uf in unit_facts:
                        if uf["values"] != af["values"] or not (_numeric_object_ok(af["obj"], uf["obj"], self.concepts_fn) or
                                                                _numeric_object_ok(af["obj"], uf["obj"] + uf["before"], self.concepts_fn)):
                            continue
                        if _NUMERIC_FREQ_BINDING and any(not _freq_matches(f, uf["freqs"]) for f in asked_freqs):
                            continue
                        if uf["segment"] in excluded:
                            continue                              # the number belongs to another clause's action
                        if (relation is not None and relation.actor is None and relation.actor_words and uf["token"] < len(nunit.tokens)
                                and not self.relations._stated(relation.actor_words, nunit.subject_text_before(uf["token"]))):
                            continue                              # "a lecture course meets 3 contact hours": not the laboratory course
                        if actor is not None and uf["token"] < len(nunit.tokens) and not _fact_subject_ok(nunit, actor, uf["token"], ask_category):
                            continue                              # "the lecturer has 15 hours" states nothing about the chair
                        found = True
                        break
                    if not found:
                        return False
            if ask_freqs and not table_row:
                # an exact unit needs the same unit ("per week" is not "per semester" or "per day"); a periodic ask needs
                # some stated frequency; "periodically" satisfies no exact unit (D-081)
                for f in ask_freqs:
                    if f == "periodic":
                        if not unit_freqs:
                            return False
                    elif f not in unit_freqs:
                        return False
                if _FREQ_ACTION_LOCAL and relation is not None and relation.action is not None and not ask_facts:
                    # D-083: the period of an action is stated in the clause of that action ("update the course page
                    # weekly and check the discussion board daily" states no daily update)
                    runit = _Unit(stated_in, self.lexicon)
                    clauses = []
                    for o in self.relations.occurrences(relation, runit):
                        if o["subject"] not in (None, relation.actor):
                            continue                              # "the dean reviews them each year" states no chair review
                        start = runit.tokens[o["index"]][2]
                        seg_start = max([m.end() for m in _SEGMENT_SPLIT.finditer(stated_in[:start])] or [0]) if _NUMERIC_SEGMENTS else 0
                        clauses.append(stated_in[seg_start:_segment_end(stated_in, start)])
                    if clauses and not any(all(_freq_matches(f, frequencies_of(c)) for f in ask_freqs) for c in clauses):
                        return False
            if table_row:
                # a terse table line names at least one of the ask's specifics; its cells are its values; it states no
                # rule, so it never settles a permission, obligation or prohibition; a calendar line states no value
                if deontic or not (stated_concepts or stated_terms) or (value and self.calendar_row(unit)):
                    return False
            else:
                if len(stated_concepts) < len(concepts) or len(stated_terms) < len(terms):
                    return False
            # D-080: every role the clause names is named in the unit, by role (a dean is not a chair); D-081: the person
            # asking is a grammatical subject of the unit (or the unit is an imperative addressed to the reader)
            if roles:
                named = self.lexicon.roles_in(stated_in)
                for role in roles:
                    if role == self.lexicon.self_role and relation is not None and relation.actor == role:
                        if not self.relations.actor_is_agent(role, _Unit(stated_in, self.lexicon)):
                            return False
                    elif role not in named and not (role == self.lexicon.self_role and
                                                    self.relations.actor_is_agent(role, _Unit(stated_in, self.lexicon))):
                        return False
            if deontic and not _NORMATIVE.search(unit) and not _Unit(stated_in, self.lexicon).is_imperative():
                return False                                      # an imperative ("Enter final grades ...") states the obligation
            if relation is not None and not table_row:
                if relation.action is not None or relation.party or relation.recipient:
                    if not self.relations.stated(relation, stated_in):
                        return False
                elif relation.actor is not None:
                    # no action to anchor on ("can I hold office hours online", "must I use the University grading
                    # scale"): the object's words are stated, the asked actor is a grammatical subject of the sentence
                    # that states them, and for a rule the subject of the rule is the asked actor or none
                    if relation.obj and not self.relations._stated(relation.obj, stated_in):
                        return False
                    if not self.relations.actor_is_agent_near(relation.actor, _Unit(stated_in, self.lexicon), relation.obj):
                        return False
                    if deontic:
                        subjects = self.relations.normative_subjects(stated_in)
                        if relation.actor not in subjects and any(sub is not None for sub in subjects):
                            return False
                elif relation.actor_words:
                    return False                                  # a generic subject with no action to bind: never the word bag (D-082)
            elif party:
                if not self.party.search(unit):
                    return False
                if action is not None and not self.action_stated(action, unit, text, words):
                    return False
            if value and not re.search(r"\d", text):
                return False
            if contents and not _CONTENT_VERB.search(unit):
                return False
            return True
        return check


__all__ = ["AskAssessment", "AskGate", "GENERAL", "Relation", "RelationChecker", "RoleLexicon", "SPECIFIC_RECOGNISED", "SPECIFIC_UNKNOWN",
           "frequencies_of", "normalise", "numeric_facts", "parse_relation", "term_stated"]
