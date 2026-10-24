"""
Structured claim model (Milestone 4, Steps 4.12-C10 and 4.12-C11, D-097 and D-098).

A specific yes/no question makes one claim: a subject or population, an actor, an action or predicate, an object, a
polarity, a modality, quantifiers, exclusivity, conditions with their roles and a qualified value. An evidence window makes
statements of the same shape. This module reads the operators that bound a claim (modality, quantifier, exclusivity,
condition roles and the operators no reader verifies) with a small domain-neutral English lexicon, and returns the
compatibility of a claim and a statement per dimension: COMPATIBLE, CONTRADICTED, UNKNOWN or NOT_APPLICABLE.

Standard library only. Nothing here imports the specialists package, retrieves, routes or calls a model: a specialist
injects its own vocabulary (population heads) and reads its own evidence units. Unknown operators fail closed: a
recognised operator without a handler is reported, never dropped. Module-level tables and patterns are the mutation seams.

D-098 adds the statement-level composite: one predicate statement per finite clause, one joint comparison of a claim with
ONE statement on every dimension (the HR readers are injected as checks), the force x restriction table, the closed
frequency lexicon, explicit universal noun-phrase slots, exclusive enumerations as item sets, the alternative fillers of
an exclusivity and one operator ledger in which every operator must be consumed or the claim fails closed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Polarity(Enum):
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"


class Quantifier(Enum):
    DEFINITE = "definite"            # an entity: "the Dean", "HR", "the probation period"
    GENERIC = "generic"              # a bare plural or an indefinite singular: "faculty members", "a college dean"
    UNIVERSAL = "universal"          # all / every / each / any (free choice) over a population
    EXISTENTIAL = "existential"      # some / certain
    PROPORTIONAL = "proportional"    # most / many / few / several: recognised, never verified


class Modality(Enum):
    NONE = "none"                    # descriptive: no modal governs the action
    POSSIBILITY = "possibility"      # epistemic: "may vary", "can result in"
    PERMISSION = "permission"        # may / can / allowed to / permitted to / -able
    ENCOURAGEMENT = "encouragement"  # encouraged / recommended / advised to
    RECOMMENDATION = "recommendation"  # should / expected to (D-056: the handbook's requirement wording)
    OBLIGATION = "obligation"        # must / shall / required to / has to
    PROHIBITION = "prohibition"      # must not / may not / cannot / not permitted / prohibited


class ConditionRole(Enum):
    SUFFICIENT = "sufficient"        # if, in case
    NECESSARY = "necessary"          # only if, provided that, as long as, subject to, an approval gate
    EXCEPTION = "exception"          # unless
    TRIGGER = "trigger"              # when, once, upon, after
    CONCESSIVE = "concessive"        # even if, regardless of, irrespective of
    UNKNOWN = "unknown"


class ExclusivityFocus(Enum):
    ACTOR = "actor"
    POPULATION = "population"
    OBJECT = "object"
    CONDITION = "condition"
    VALUE = "value"
    TIME = "time"


class Compatibility(Enum):
    COMPATIBLE = "compatible"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class OperatorKind(Enum):
    NEGATION = "negation"
    MODALITY = "modality"
    QUANTIFIER = "quantifier"
    PROPORTIONAL = "proportional"
    EXCLUSIVITY = "exclusivity"
    CONDITION = "condition"
    COMPARATIVE = "comparative"
    NUMERIC_QUALIFIER = "numeric_qualifier"
    TIME_DISTRIBUTIVE = "time_distributive"
    EVENT_ORDER = "event_order"
    AMBIGUOUS = "ambiguous"


class Breadth(Enum):
    SAME_OR_BROADER = "same_or_broader"
    NARROWER = "narrower"
    UNRELATED = "unrelated"


# ---------------------------------------------------------------------------
# Data model (frozen)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Population:
    """A population the claim or statement is about: its head nouns (as roots or words), category axes, restrictions
    (modifier roots that narrow it), negated restrictions, its quantifier and whether the subject is a document label."""
    head: FrozenSet[str]
    categories: FrozenSet[str] = frozenset()
    restrictions: FrozenSet[str] = frozenset()
    negated_restrictions: FrozenSet[str] = frozenset()
    quantifier: Quantifier = Quantifier.GENERIC
    is_label: bool = False
    text: str = ""


@dataclass(frozen=True)
class Condition:
    """A condition with its role: given (``given``) or stated absent."""
    text: str
    roots: FrozenSet[str]
    role: ConditionRole
    given: bool = True


@dataclass(frozen=True)
class Exclusivity:
    """An exclusivity marker ("only", "solely", "exclusively", "limited to", "restricted to", "just" + number) and its focus."""
    focus: ExclusivityFocus
    target: FrozenSet[str]
    marker: str


@dataclass(frozen=True)
class QualifiedValue:
    """A value with its qualifier: "exact", "up_to" or "at_least"; ``exclusive`` for "only / just N"."""
    values: Tuple[float, ...]
    unit: str
    qualifier: str = "exact"
    exclusive: bool = False


@dataclass(frozen=True)
class Operator:
    """One recognised operator occurrence; ``handled_by`` names the dimension that verifies it, None fails closed."""
    kind: OperatorKind
    text: str
    span: Tuple[int, int]
    handled_by: Optional[str]


@dataclass(frozen=True)
class Claim:
    """The claim a specific yes/no question makes."""
    text: str
    polarity: Polarity
    subject: Optional[Population]
    actor: Optional[str]
    label: Optional[str]
    action: Optional[str]
    predicate: FrozenSet[str]
    obj: FrozenSet[str]
    obj_quantifier: Quantifier
    modality: Modality
    exclusivity: Tuple[Exclusivity, ...]
    conditions: Tuple[Condition, ...]
    modifiers: Tuple[Tuple[FrozenSet[str], bool], ...]
    value: Optional[QualifiedValue]
    source: Optional[str]
    operators: Tuple[Operator, ...]


@dataclass(frozen=True)
class Statement:
    """What one local unit of an evidence window states about the asked claim. Support is local to one statement; a
    contradiction or exception may come from any statement of the window."""
    text: str
    unit_index: int
    sentence_index: int
    span: Tuple[int, int]
    subject: Optional[Population]
    actor: Optional[str]
    action_roots: FrozenSet[str]
    predicate_roots: FrozenSet[str]
    obj: FrozenSet[str]
    polarity: Polarity
    modality: Optional[Modality]
    exclusivity: Tuple[Exclusivity, ...]
    antecedents: Tuple[Condition, ...]
    exceptions: Tuple[str, ...]
    values: Tuple[QualifiedValue, ...]


@dataclass(frozen=True)
class DimensionResult:
    """The compatibility of a claim and a statement (or a window) on one dimension."""
    dimension: str
    compatibility: Compatibility
    reason: str = ""
    unit_index: Optional[int] = None


# ---------------------------------------------------------------------------
# Operator inventory (domain-neutral English; every pattern is a mutation seam)
# ---------------------------------------------------------------------------
#: Modal markers of an evidence verb group, read immediately before the verb they govern. The prohibition forms are
#: separate patterns so that "shall not be permitted to" is read up to "to" (a single alternation would stop at "not").
PROHIBITION_MARKERS = re.compile(r"\b(?:must|shall|may|should|can)\s+not\b|\bcannot\b|\b(?:must|shall|may|should|ca)n['’]t\b", re.I)
PROHIBITION_PREDICATES = re.compile(
    r"\bnot\s+(?:be\s+)?(?:permitted|allowed)\s+to\b|\b(?:is|are|be)\s+(?:prohibited|forbidden|barred)\s+(?:from|to)\b|\bunder\s+no\s+circumstances?\b", re.I)
#: A negated obligation states no obligation and no prohibition ("is not required to", "need not").
NO_OBLIGATION_MARKERS = re.compile(r"\b(?:is|are|be)\s+not\s+(?:required|obliged|obligated)\s+to\b|\bneed\s+not\b|\b(?:does|do)\s+not\s+(?:have|need)\s+to\b|"
                                   r"\b(?:does|do|did)\s+not\s+require\b|\brequires?\s+no\b", re.I)
#: "requires" states an obligation of what it requires ("Termination ... requires the approval of the Chancellor").
OBLIGATION_MARKERS = re.compile(
    r"\b(?:must|shall)\b|\b(?:is|are|be|being|was|were)\s+(?:required|obliged|obligated|compelled)\s+to\b|\b(?:has|have|had|needs?)\s+to\b|"
    r"\b(?:is|are)\s+to\s+be\b|\brequires?\b", re.I)
RECOMMENDATION_MARKERS = re.compile(r"\bshould\b|\b(?:is|are|be)\s+expected\s+to\b|\bought\s+to\b", re.I)
ENCOURAGEMENT_MARKERS = re.compile(
    r"\b(?:is|are|be)\s+(?:(?:strongly|equally|also|highly)\s+)*(?:encouraged|recommended|advised|invited|urged)\s+to\b|\bpreferably\b", re.I)
PERMISSION_MARKERS = re.compile(
    r"\b(?:may|can|could|might)\b|\b(?:is|are|be)\s+(?:allowed|permitted|able|authori[sz]ed)\s+to\b|"
    r"\b(?:holds?|has|have)\s+(?:the\s+)?(?:jurisdiction|authority|right|option)\s+to\b", re.I)
#: The closest marker before the verb wins; on a tie the earlier entry wins (a prohibition contains the permission word it
#: negates, a negated obligation contains the obligation word).
MODAL_MARKERS: Tuple[Tuple["re.Pattern[str]", Modality], ...] = (
    (PROHIBITION_PREDICATES, Modality.PROHIBITION), (PROHIBITION_MARKERS, Modality.PROHIBITION), (NO_OBLIGATION_MARKERS, Modality.NONE),
    (OBLIGATION_MARKERS, Modality.OBLIGATION), (RECOMMENDATION_MARKERS, Modality.RECOMMENDATION),
    (ENCOURAGEMENT_MARKERS, Modality.ENCOURAGEMENT), (PERMISSION_MARKERS, Modality.PERMISSION))
#: Verbs that make may / might / could / can epistemic: a possibility, not a permission ("office hours may vary").
EPISTEMIC_VERBS: FrozenSet[str] = frozenset(
    "vary varies differ differs change changes result results lead leads cause causes affect affects impact impacts influence "
    "influences occur occurs happen happens arise arises depend depends increase increases decrease decreases".split())
#: How far (characters) a modal marker may stand before the verb it governs.
MODAL_REACH = 70
#: Where a verb group's clause begins (a modal before it governs another clause).
CLAUSE_START = re.compile(r"[;:.!?]|\b(?:if|unless|when|whenever|once|while|whereas|although|though|because|since|but|however|which|who|whom|whose)\b",
                          re.I)

#: Question-side modality.
ASK_LEAD = re.compile(r"^\s*(?P<aux>must|shall|should|ought|may|can|could|might|will|would|is|are|am|was|were|do|does|did|has|have)\b", re.I)
ASK_DEONTIC_ADJ = re.compile(
    r"\b(?:is|are|am|was|were|be)\b[^?]*?\b(?P<adj>required|obliged|obligated|expected|allowed|permitted|encouraged|recommended|prohibited|forbidden)\s+"
    r"(?P<prep>to|from)\s+(?P<next>[a-z'’-]+)", re.I)
ASK_HAVE_TO = re.compile(r"^\s*(?:do|does|did)\s+[\w\s'’-]{1,40}?\s+(?:have|need)\s+to\b", re.I)
ASK_IT_IS = re.compile(r"^\s*is\s+it\s+(?P<adj>mandatory|compulsory|required|necessary|optional|possible|allowed|permitted|prohibited|forbidden)\b",
                       re.I)
#: Nouns of a requirement predicate: "Is Chancellor approval required to terminate ...?" asks whether the approval is required,
#: not an obligation of the approval.
REQUIREMENT_NOUNS: FrozenSet[str] = frozenset("approval approvals permission consent notice notification recommendation authorization authorisation "
                                              "clearance documentation documents signature endorsement decision evidence certificate letter".split())
#: Words that never follow "to" as a verb in "required to <verb>" (a noun phrase follows instead).
NOT_A_VERB_AFTER_TO: FrozenSet[str] = frozenset("a an the this that these those their his her its our your my any all some each every one two".split())

#: Quantifiers.
UNIVERSAL_WORDS: FrozenSet[str] = frozenset({"all", "every", "each", "any"})
EXISTENTIAL_WORDS: FrozenSet[str] = frozenset({"some", "certain"})
PROPORTIONAL_WORDS: FrozenSet[str] = frozenset({"most", "many", "few", "several", "majority", "minority"})
#: Time nouns: "each academic year", "every two years", "each September" are time distributives, no population.
TIME_NOUNS: FrozenSet[str] = frozenset(
    "year years academic semester semesters term terms month months week weeks day days cycle cycles period periods session sessions "
    "quarter quarters time times occasion occasions january february march april may june july august september october november "
    "december spring summer fall autumn winter morning evening".split())
#: Default population heads (a specialist injects its own).
POPULATION_HEADS = re.compile(
    r"^(?:faculty|faculties|members?|employees?|staff|candidates?|applicants?|appointees?|professors?|lecturers?|instructors?|individuals?|"
    r"recruits?|personnel|researchers?|deans?|chairs?|chairpersons?|people|persons?)$", re.I)
#: Words that end a noun phrase after a quantifier.
NP_BREAK: FrozenSet[str] = frozenset(
    "of for to in on at with by from including who whom whose which that where is are was were be been being can could may might must "
    "shall should will would do does did has have had not and or than as".split())

#: Exclusivity markers.
EXCLUSIVITY_MARKERS = re.compile(r"\b(?:only|solely|exclusively|just)\b|\blimited\s+to\b|\brestricted\s+to\b", re.I)
NUMBER_WORDS: FrozenSet[str] = frozenset(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand once twice thrice single half".split())
TIME_PREPOSITIONS: FrozenSet[str] = frozenset("at during before by on within in from until".split())
CONDITION_AFTER_ONLY = re.compile(
    r"^\s*(?:if|when|whenever|upon|once|after|provided|in\s+case|in\s+(?:special|exceptional|certain|specific)\s+(?:circumstances?|cases?)|"
    r"under\s+(?:special|exceptional|certain|specific)\s+(?:circumstances?|conditions?)|with\b)", re.I)

#: Condition openers and their roles.
CONDITION_ROLES: Dict[str, ConditionRole] = {
    "if": ConditionRole.SUFFICIENT, "in case": ConditionRole.SUFFICIENT, "in case of": ConditionRole.SUFFICIENT,
    "in the event": ConditionRole.SUFFICIENT, "in the event that": ConditionRole.SUFFICIENT, "in the event of": ConditionRole.SUFFICIENT,
    "when": ConditionRole.TRIGGER, "whenever": ConditionRole.TRIGGER, "once": ConditionRole.TRIGGER, "upon": ConditionRole.TRIGGER,
    "after": ConditionRole.TRIGGER,
    "only if": ConditionRole.NECESSARY, "only when": ConditionRole.NECESSARY, "only upon": ConditionRole.NECESSARY,
    "only after": ConditionRole.NECESSARY, "only with": ConditionRole.NECESSARY, "provided that": ConditionRole.NECESSARY,
    "as long as": ConditionRole.NECESSARY, "subject to": ConditionRole.NECESSARY,
    "unless": ConditionRole.EXCEPTION, "except if": ConditionRole.EXCEPTION, "except when": ConditionRole.EXCEPTION,
    "even if": ConditionRole.CONCESSIVE, "regardless of": ConditionRole.CONCESSIVE, "irrespective of": ConditionRole.CONCESSIVE,
}
#: An approval gate: an approval-type noun phrase attached to the predicate ("with Chancellor approval", "upon approval
#: from appropriate University authorities", "subject to the Dean's approval"). "with full pay" is no condition.
APPROVAL_PHRASE = re.compile(
    r"\b(?:with|upon|subject\s+to|following|on|after)\s+(?:the\s+|an?\s+|prior\s+|written\s+|formal\s+|appropriate\s+|final\s+)*"
    r"(?:[\w'’-]+\s+){0,4}?(?:approval|consent|authori[sz]ation|endorsement|recommendation|permission)s?\b", re.I)

#: Event order: before / after + a clause whose verb is a process action ("before the Deans Council approves it"). A state
#: ("after sick leave days are exhausted") is a condition, read by the condition contracts.
EVENT_ORDER = re.compile(
    r"\b(?:before|after|prior\s+to|following)\s+(?:the\s+|a\s+|an\s+|their\s+|its\s+|his\s+|her\s+)?(?:[\w'’-]+\s+){0,4}?"
    r"(?:approves|approved|reviews|reviewed|decides|decided|submits|submitted|notifies|notified|recommends|recommended|forwards|forwarded|"
    r"signs|signed|receives|received|completes|completed|issues|issued)\b", re.I)
#: Comparatives (a negated one is a numeric qualifier, read by the value contract).
COMPARATIVE = re.compile(
    r"\b(?P<neg>not\s+|no\s+)?(?:more|less|fewer|longer|shorter|higher|lower|greater|larger|smaller|earlier|later)\s+than\b|"
    r"\b(?P<neg2>not\s+(?:to\s+)?|no\s+)?exceed(?:s|ed|ing)?\b", re.I)
NUMERIC_QUALIFIERS = re.compile(
    r"\bup\s+to\b|\bat\s+(?:most|least)\b|\bmaximum\b|\bminimum\b|\bno\s+(?:more|less|fewer|longer)\s+than\b|\bnot\s+(?:more|less|fewer|longer)\s+than\b|"
    r"\bnot\s+(?:to\s+)?exceed\w*|\bcapped\s+at\b", re.I)
NEGATION_WORDS = re.compile(r"\b(?:not|never|no|none|neither|nor|without|cannot)\b|n['’]t\b", re.I)
CONDITION_OPENERS = re.compile(r"\b(?:only\s+if|even\s+if|if|unless|whenever|when|once|in\s+case|provided\s+that|as\s+long\s+as)\b", re.I)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d+(?:[.,]\d+)*")


def tokens(text: str) -> List[Tuple[str, int, int]]:
    """Lower-cased word tokens with their spans."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def ask_modality(text: str) -> Modality:
    """The modality a yes/no question asks: its opening modal, a deontic adjective before "to <verb>" ("Is X required to
    submit ...?"), "Do I have / need to ...?" or "Is it mandatory / optional ...?". A requirement predicate ("Is approval
    required for X?") and an entitlement ("eligible / entitled") ask no modality: their own contracts bind them."""
    low = text.lower().strip()
    m = ASK_LEAD.match(low)
    if m is None:
        return Modality.NONE
    aux = m.group("aux")
    if aux in ("must", "shall"):
        return Modality.OBLIGATION
    if aux in ("should", "ought"):
        return Modality.RECOMMENDATION
    if aux in ("may", "can", "could", "might"):
        words = [w for w, _s, _e in tokens(low[m.end():])][:10]
        return Modality.POSSIBILITY if any(w in EPISTEMIC_VERBS for w in words) else Modality.PERMISSION
    d = ASK_DEONTIC_ADJ.search(low)
    if d is not None and (d.group("prep") == "from" or d.group("next") not in NOT_A_VERB_AFTER_TO):
        adj = d.group("adj")
        subject = [w for w, _s, _e in tokens(low[m.end():d.start("adj")]) if w not in ("is", "are", "am", "was", "were", "be")]
        if subject and subject[-1] in REQUIREMENT_NOUNS:
            return Modality.NONE                                  # a requirement predicate: "is approval required to terminate"
        if adj in ("required", "obliged", "obligated"):
            return Modality.OBLIGATION
        if adj == "expected":
            return Modality.RECOMMENDATION
        if adj in ("encouraged", "recommended"):
            return Modality.ENCOURAGEMENT
        if adj in ("allowed", "permitted"):
            return Modality.PERMISSION
        return Modality.PROHIBITION
    if ASK_HAVE_TO.match(low):
        return Modality.OBLIGATION
    i = ASK_IT_IS.match(low)
    if i is not None:
        adj = i.group("adj")
        if adj in ("mandatory", "compulsory", "required", "necessary"):
            return Modality.OBLIGATION
        if adj in ("prohibited", "forbidden"):
            return Modality.PROHIBITION
        return Modality.PERMISSION
    return Modality.NONE


#: Openers of a subordinate clause: after one, a comma ends the antecedent and the main clause begins.
SUBORDINATE_OPENERS = re.compile(r"(?:if|unless|when|whenever|once|while|whereas|although|though|because|since)", re.I)


def clause_start(text: str, position: int) -> int:
    """Where the clause holding ``position`` begins: after the last clause boundary, or after the comma that ends a leading
    antecedent ("If faculty may apply, the committee notifies ...": the main clause begins at the comma)."""
    start, opener = 0, False
    for m in CLAUSE_START.finditer(text, 0, position):
        start, opener = m.end(), bool(SUBORDINATE_OPENERS.fullmatch(m.group(0)))
    if opener:
        comma = text.rfind(",", start, position)
        if comma >= 0:
            start = comma + 1
    return start


def statement_modality(text: str, anchor: int, anchor_word: str = "", reach: Optional[int] = None) -> Modality:
    """The modality of the verb group governing the word at ``anchor``: the closest modal marker before it within its
    clause and ``reach`` characters. "may / can / could / might" before an epistemic verb is a possibility. An "-able"
    form of the anchor ("extendable") is a permission. No marker: NONE (descriptive)."""
    word = anchor_word.lower()
    if re.search(r"(?:able|ible)$", word) and len(word) > 6:
        return Modality.PERMISSION
    reach = MODAL_REACH if reach is None else reach
    lo = max(clause_start(text, anchor), anchor - reach)
    window = text[lo:anchor]
    best: Optional[Tuple[int, Modality]] = None
    for pattern, modality in MODAL_MARKERS:
        for m in pattern.finditer(window):
            if best is None or m.end() > best[0]:
                best = (m.end(), modality)
    if best is None:
        return Modality.NONE
    if best[1] is Modality.PERMISSION and word in EPISTEMIC_VERBS:
        return Modality.POSSIBILITY
    return best[1]


def unit_modality(text: str) -> Modality:
    """The modality of the first modal marker anywhere in ``text``, bound to no verb (the shared-gate reading; kept for the
    mutation that unbinds modality)."""
    first: Optional[Tuple[int, Modality]] = None
    for pattern, modality in MODAL_MARKERS:
        m = pattern.search(text)
        if m is not None and (first is None or m.start() < first[0]):
            first = (m.start(), modality)
    return first[1] if first else Modality.NONE


def quantifier_phrases(text: str, heads: "re.Pattern[str]" = POPULATION_HEADS) -> List[Tuple[str, str, str, Tuple[int, int]]]:
    """Every quantifier word of ``text`` with the slot it binds: (word, slot, head, span), slot one of "population",
    "object", "time"; a proportional word is reported with slot "proportional"."""
    out: List[Tuple[str, str, str, Tuple[int, int]]] = []
    toks = tokens(text)
    for i, (w, s, e) in enumerate(toks):
        if w not in UNIVERSAL_WORDS and w not in EXISTENTIAL_WORDS and w not in PROPORTIONAL_WORDS:
            continue
        if w == "many" and i and toks[i - 1][0] == "how":
            continue                                              # "how many" asks a number
        if w == "most" and i and toks[i - 1][0] in ("at", "the"):
            continue                                              # "at most", "at the most": a numeric qualifier
        if w == "any" and i and toks[i - 1][0] in ("there", "not", "no"):
            continue                                              # "is there any", "is not any": existence (M3)
        if w == "few" and i and toks[i - 1][0] != "a" and (i + 1 >= len(toks) or toks[i + 1][0] in NP_BREAK):
            continue
        phrase: List[str] = []
        for w2, _s2, _e2 in toks[i + 1:i + 5]:
            if w2 in NP_BREAK or w2 in UNIVERSAL_WORDS or w2 in EXISTENTIAL_WORDS or w2 in PROPORTIONAL_WORDS:
                break
            phrase.append(w2)
        if not phrase:
            continue
        if w == "certain" and phrase[0] in ("that", "whether"):
            continue
        slot, head = "object", phrase[-1]
        for p in phrase:                                          # the first population head or time noun decides the slot
            if heads.match(p):
                slot, head = "population", p
                break
            if p in TIME_NOUNS:
                slot, head = "time", p
                break
        if w in PROPORTIONAL_WORDS:
            slot = "proportional"
        out.append((w, slot, head, (s, e)))
    return out


def quantifier_of(word: str) -> Quantifier:
    if word in UNIVERSAL_WORDS:
        return Quantifier.UNIVERSAL
    if word in EXISTENTIAL_WORDS:
        return Quantifier.EXISTENTIAL
    if word in PROPORTIONAL_WORDS:
        return Quantifier.PROPORTIONAL
    return Quantifier.GENERIC


def condition_role(opener: str) -> ConditionRole:
    """The role of a condition opener ("if", "only if", "unless", "upon", "provided that", ...)."""
    key = re.sub(r"\s+", " ", opener.strip().lower())
    return CONDITION_ROLES.get(key, ConditionRole.UNKNOWN)


def approval_phrases(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """The approval gates of ``text`` (an approval-type noun phrase introduced by with / upon / subject to / following)."""
    return [(m.group(0), (m.start(), m.end())) for m in APPROVAL_PHRASE.finditer(text)]


#: Function words never stand for an exclusivity target.
TARGET_STOP: FrozenSet[str] = frozenset("the a an at of in on by from to during before within until for with and or that which who is are be".split())


def _content(words: Iterable[str]) -> FrozenSet[str]:
    return frozenset(w for w in words if w not in TARGET_STOP)


def _phrase_after(toks: Sequence[Tuple[str, int, int]], i: int, limit: int = 8) -> List[str]:
    out: List[str] = []
    for w, _s, _e in toks[i:i + limit]:
        if w in ("?", ".", ",", ";"):
            break
        out.append(w)
    return out


def read_exclusivity(text: str, heads: "re.Pattern[str]" = POPULATION_HEADS) -> Tuple[Tuple[Exclusivity, ...], Tuple[Operator, ...]]:
    """The exclusivity markers of ``text`` with their focus, and the ambiguous ones (an operator no reader verifies):
    "Is <X> the only <noun> ..." (ACTOR), "only / just" + a number (VALUE), "only if / when / upon / with / in special
    circumstances" (CONDITION), "only at / during / before / in the first week" + time (TIME), "only" + a population head
    (POPULATION), "may only be <verb> by <agent>" (ACTOR), else "only" + a noun phrase (OBJECT). Any other "just" is
    ambiguous."""
    found: List[Exclusivity] = []
    ambiguous: List[Operator] = []
    toks = tokens(text)
    low = text.lower()
    lead = re.match(r"^\s*(?:is|are|was|were)\s+(?P<x>.+?)\s+the\s+(?:only|sole)\b", low)
    for m in EXCLUSIVITY_MARKERS.finditer(text):
        marker = m.group(0).lower()
        idx = next((k for k, (_w, s, _e) in enumerate(toks) if s >= m.end()), len(toks))
        after = [w for w, _s, _e in toks[idx:idx + 8]]
        prev = [w for w, _s, _e in toks[max(0, idx - 4):idx - 1]] if idx else []
        rest = text[m.end():]
        if marker == "just":
            if after and (after[0] in NUMBER_WORDS or after[0][0].isdigit()):
                found.append(Exclusivity(ExclusivityFocus.VALUE, _content(after[:2]), marker))
            else:
                ambiguous.append(Operator(OperatorKind.AMBIGUOUS, marker, (m.start(), m.end()), None))
            continue
        if lead is not None and m.start() >= lead.start("x") and m.start() <= lead.end() + 1:
            focus_words = _content(w for w, _s, _e in tokens(lead.group("x")))
            pop = any(heads.match(w) for w in focus_words)
            found.append(Exclusivity(ExclusivityFocus.POPULATION if pop else ExclusivityFocus.ACTOR, focus_words, marker))
            continue
        if after and (after[0] in NUMBER_WORDS or after[0][0].isdigit()):
            found.append(Exclusivity(ExclusivityFocus.VALUE, _content(after[:3]), marker))
            continue
        verb_form = re.match(r"^\s+(?:(?:be|been)\s+)?[a-z'’-]+(?:ed|en|able|ible)(?=\s)", rest, re.I)
        if CONDITION_AFTER_ONLY.match(rest) or (verb_form is not None and CONDITION_AFTER_ONLY.match(rest[verb_form.end():])):
            found.append(Exclusivity(ExclusivityFocus.CONDITION, _content(_phrase_after(toks, idx)), marker))
            continue                                              # "only if ...", "only extendable with Chancellor approval"
        if after and after[0] in TIME_PREPOSITIONS and any(w in TIME_NOUNS or w in ("end", "start", "beginning", "first", "last") for w in after[1:6]):
            end = 6
            k = next((k for k in range(1, len(after)) if after[k] in TIME_NOUNS), None)
            if k is not None:
                while k + 1 < len(after) and after[k + 1] in TIME_NOUNS:
                    k += 1
                end = max(end, k + 1)                             # through the time nouns: "the end of an academic semester"
            found.append(Exclusivity(ExclusivityFocus.TIME, _content(after[:end]), marker))
            continue
        agent = re.match(r"^\s+(?:be\s+)?(?:[\w'’-]+\s+){0,2}?by\s+(?P<agent>[^,.;?]+)", rest, re.I)
        if agent is not None and (prev[-1:] in (["may"], ["can"], ["shall"], ["must"], ["be"]) or re.match(r"^\s+be\b", rest, re.I)):
            found.append(Exclusivity(ExclusivityFocus.ACTOR, _content(w for w, _s, _e in tokens(agent.group("agent"))), marker))
            continue
        np = []
        for w in after:
            if w in NP_BREAK:
                break
            np.append(w)
            if heads.match(w):
                break                                             # a population ends at its head ("only Muslim faculty")
        if np and any(heads.match(w) for w in np):
            found.append(Exclusivity(ExclusivityFocus.POPULATION, _content(np), marker))
        elif np:
            found.append(Exclusivity(ExclusivityFocus.OBJECT, _content(np), marker))
        else:
            before = [w for w, _s, _e in toks[max(0, idx - 4):idx - 1]]
            found.append(Exclusivity(ExclusivityFocus.OBJECT, _content(before), marker))
    return tuple(found), tuple(ambiguous)


def find_operators(text: str, heads: "re.Pattern[str]" = POPULATION_HEADS, modality: Optional[Modality] = None) -> Tuple[Operator, ...]:
    """Every operator of a question with the dimension that verifies it; ``handled_by`` None fails closed."""
    ops: List[Operator] = []
    if modality is None:
        modality = ask_modality(text)
    if modality is not Modality.NONE:
        ops.append(Operator(OperatorKind.MODALITY, modality.value, (0, 0), "modality"))
    for m in NEGATION_WORDS.finditer(text):
        ops.append(Operator(OperatorKind.NEGATION, m.group(0), (m.start(), m.end()), "negation"))
    for w, slot, _head, span in quantifier_phrases(text, heads):
        if slot == "proportional":
            ops.append(Operator(OperatorKind.PROPORTIONAL, w, span, None))
        elif slot == "time":
            ops.append(Operator(OperatorKind.TIME_DISTRIBUTIVE, w, span, "period"))
        else:
            ops.append(Operator(OperatorKind.QUANTIFIER, w, span, "scope"))
    excl, ambiguous = read_exclusivity(text, heads)
    for e in excl:
        ops.append(Operator(OperatorKind.EXCLUSIVITY, e.marker + ":" + e.focus.value, (0, 0), "exclusivity"))
    ops.extend(ambiguous)
    for m in CONDITION_OPENERS.finditer(text):
        ops.append(Operator(OperatorKind.CONDITION, m.group(0), (m.start(), m.end()), "condition"))
    for m in NUMERIC_QUALIFIERS.finditer(text):
        ops.append(Operator(OperatorKind.NUMERIC_QUALIFIER, m.group(0), (m.start(), m.end()), "value"))
    for m in COMPARATIVE.finditer(text):
        if m.group("neg") or m.group("neg2"):
            continue                                              # "not more than", "not exceeding": a numeric qualifier
        ops.append(Operator(OperatorKind.COMPARATIVE, m.group(0), (m.start(), m.end()), None))
    for m in EVENT_ORDER.finditer(text):
        ops.append(Operator(OperatorKind.EVENT_ORDER, m.group(0), (m.start(), m.end()), None))
    return tuple(ops)


def unhandled(operators: Iterable[Operator]) -> Tuple[Operator, ...]:
    """The operators no reader verifies."""
    return tuple(o for o in operators if o.handled_by is None)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------
_C, _X, _U = Compatibility.COMPATIBLE, Compatibility.CONTRADICTED, Compatibility.UNKNOWN
_M = Modality
#: (asked, stated) -> compatibility. "should" satisfies "must" (D-056); "may" never does; an obligation entails a permission.
MODAL_TABLE: Dict[Tuple[Modality, Modality], Compatibility] = {
    (_M.OBLIGATION, _M.OBLIGATION): _C, (_M.OBLIGATION, _M.RECOMMENDATION): _C, (_M.OBLIGATION, _M.ENCOURAGEMENT): _U,
    (_M.OBLIGATION, _M.PERMISSION): _U, (_M.OBLIGATION, _M.POSSIBILITY): _U, (_M.OBLIGATION, _M.NONE): _U,
    (_M.OBLIGATION, _M.PROHIBITION): _X,
    (_M.RECOMMENDATION, _M.OBLIGATION): _C, (_M.RECOMMENDATION, _M.RECOMMENDATION): _C, (_M.RECOMMENDATION, _M.ENCOURAGEMENT): _C,
    (_M.RECOMMENDATION, _M.PERMISSION): _U, (_M.RECOMMENDATION, _M.POSSIBILITY): _U, (_M.RECOMMENDATION, _M.NONE): _U,
    (_M.RECOMMENDATION, _M.PROHIBITION): _X,
    (_M.ENCOURAGEMENT, _M.OBLIGATION): _C, (_M.ENCOURAGEMENT, _M.RECOMMENDATION): _C, (_M.ENCOURAGEMENT, _M.ENCOURAGEMENT): _C,
    (_M.ENCOURAGEMENT, _M.PERMISSION): _U, (_M.ENCOURAGEMENT, _M.POSSIBILITY): _U, (_M.ENCOURAGEMENT, _M.NONE): _U,
    (_M.ENCOURAGEMENT, _M.PROHIBITION): _X,
    (_M.PERMISSION, _M.OBLIGATION): _C, (_M.PERMISSION, _M.RECOMMENDATION): _C, (_M.PERMISSION, _M.ENCOURAGEMENT): _C,
    (_M.PERMISSION, _M.PERMISSION): _C, (_M.PERMISSION, _M.POSSIBILITY): _U, (_M.PERMISSION, _M.NONE): _C,
    (_M.PERMISSION, _M.PROHIBITION): _X,
    (_M.POSSIBILITY, _M.OBLIGATION): _C, (_M.POSSIBILITY, _M.RECOMMENDATION): _C, (_M.POSSIBILITY, _M.ENCOURAGEMENT): _C,
    (_M.POSSIBILITY, _M.PERMISSION): _C, (_M.POSSIBILITY, _M.POSSIBILITY): _C, (_M.POSSIBILITY, _M.NONE): _C,
    (_M.POSSIBILITY, _M.PROHIBITION): _U,
    (_M.PROHIBITION, _M.OBLIGATION): _X, (_M.PROHIBITION, _M.RECOMMENDATION): _X, (_M.PROHIBITION, _M.ENCOURAGEMENT): _X,
    (_M.PROHIBITION, _M.PERMISSION): _X, (_M.PROHIBITION, _M.POSSIBILITY): _U, (_M.PROHIBITION, _M.NONE): _U,
    (_M.PROHIBITION, _M.PROHIBITION): _C,
}


def modal_compatibility(asked: Modality, stated: Optional[Modality]) -> Compatibility:
    """The asked modality against the stated one (None: the asked action is not found in the unit)."""
    if asked is Modality.NONE:
        return Compatibility.NOT_APPLICABLE
    if stated is None:
        return Compatibility.COMPATIBLE if asked in (Modality.PERMISSION, Modality.POSSIBILITY) else Compatibility.UNKNOWN
    return MODAL_TABLE.get((asked, stated), Compatibility.UNKNOWN)


#: A universal claim needs an explicitly universal statement: a generic rule states a default, not "all" or "every".
UNIVERSAL_NEEDS_EXPLICIT = True
#: An existential claim is satisfied by a narrower population ("some faculty members" by "part-time faculty members").
EXISTENTIAL_ACCEPTS_NARROWER = True


def scope_compatibility(claimed: Quantifier, stated: Optional[Quantifier], breadth: Breadth) -> Compatibility:
    """The asked population against the stated one: a universal claim needs a universal statement at least as broad, a
    generic claim a statement at least as broad, an existential claim any statement of the same population (narrower
    included). Definite subjects and proportional quantifiers are not scope claims (the latter fail closed elsewhere)."""
    if claimed in (Quantifier.DEFINITE, Quantifier.PROPORTIONAL):
        return Compatibility.NOT_APPLICABLE
    if breadth is Breadth.UNRELATED:
        return Compatibility.UNKNOWN
    if claimed is Quantifier.EXISTENTIAL:
        return Compatibility.COMPATIBLE if (EXISTENTIAL_ACCEPTS_NARROWER or breadth is Breadth.SAME_OR_BROADER) else Compatibility.UNKNOWN
    if breadth is Breadth.NARROWER:
        return Compatibility.UNKNOWN
    if claimed is Quantifier.UNIVERSAL and UNIVERSAL_NEEDS_EXPLICIT and stated is not Quantifier.UNIVERSAL:
        return Compatibility.UNKNOWN
    return Compatibility.COMPATIBLE


def exclusivity_compatibility(focus: ExclusivityFocus, stated_exclusive: bool, alternative: bool, exact_value: bool) -> Compatibility:
    """An exclusivity claim: an alternative on the same focus contradicts it; an exclusivity marker on the same focus (or an
    obligation on a time or condition slot, which the caller reports as ``stated_exclusive``) states it; an exact value
    with no extension, limit or alternative states a value exclusivity; otherwise it is unknown."""
    if alternative:
        return Compatibility.CONTRADICTED
    if stated_exclusive:
        return Compatibility.COMPATIBLE
    if focus is ExclusivityFocus.VALUE and exact_value:
        return Compatibility.COMPATIBLE
    return Compatibility.UNKNOWN


def condition_role_compatibility(asked: ConditionRole, stated: Optional[ConditionRole], approval_gate: bool = False) -> Compatibility:
    """An asked condition role against the stated one: a necessary condition needs a necessary statement (an approval gate
    included); a sufficient or triggering condition is answered by a sufficient, triggering or approval-gate statement, not
    by a bare necessary one ("only if Y" states no sufficiency)."""
    if asked is ConditionRole.NECESSARY:
        return Compatibility.COMPATIBLE if (stated is ConditionRole.NECESSARY or approval_gate) else Compatibility.UNKNOWN
    if asked in (ConditionRole.SUFFICIENT, ConditionRole.TRIGGER):
        if stated is ConditionRole.NECESSARY and not approval_gate:
            return Compatibility.UNKNOWN
        return Compatibility.COMPATIBLE
    return Compatibility.NOT_APPLICABLE


def combine(results: Iterable[DimensionResult]) -> Compatibility:
    """Conservative combination: any contradiction contradicts, else any unknown is unknown, else compatible."""
    seen = [r.compatibility for r in results]
    if Compatibility.CONTRADICTED in seen:
        return Compatibility.CONTRADICTED
    if Compatibility.UNKNOWN in seen:
        return Compatibility.UNKNOWN
    if Compatibility.COMPATIBLE in seen:
        return Compatibility.COMPATIBLE
    return Compatibility.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# D-098 (Step 4.12-C11): one statement per predicate clause, one joint comparison, one operator ledger
# ---------------------------------------------------------------------------
#: A claim is compared with ONE predicate statement (a finite clause with its secondary predications) on every dimension at
#: once; other statements of the returned quote may only veto. Every operator of the question is recorded in a ledger and
#: must be consumed by a dimension that actually evaluated it on that statement, else the claim fails closed.
class AnchorKind(Enum):
    VERB = "verb"
    ENTITLEMENT = "entitlement"
    PREDICATE = "predicate"
    OBJECT = "object"
    FALLBACK = "fallback"


class RestrictionKind(Enum):
    NECESSARY = "necessary"
    EXCEPTION = "exception"
    SPECIAL = "special"
    HEDGE = "hedge"
    GATE = "gate"
    SUFFICIENT = "sufficient"
    TRIGGER = "trigger"
    CONCESSIVE = "concessive"
    POSSIBILITY = "possibility"


class FrequencyClass(Enum):
    UNIVERSAL = "universal"
    NEGATIVE_UNIVERSAL = "negative_universal"
    EXISTENTIAL = "existential"
    DEFAULT = "default"
    PROPORTIONAL = "proportional"


class Force(Enum):
    EXISTENTIAL = "existential"
    GENERIC = "generic"
    DEONTIC = "deontic"
    UNIVERSAL_QUANTIFIED = "universal_quantified"
    UNIVERSAL_FREQUENCY = "universal_frequency"


@dataclass(frozen=True)
class NounPhrase:
    """A quantified noun-phrase slot (no scope engine): the quantifier, the head and the modifiers between them (as roots),
    whether a number or a cap sits inside it, and the slot it fills in its clause ("subject", "object")."""
    text: str
    head: str
    modifiers: FrozenSet[str] = frozenset()
    quantifier: Quantifier = Quantifier.GENERIC
    capped: bool = False
    slot: str = "object"
    span: Tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class Anchor:
    """Where a statement states the asked predicate: its kind, the word and its offset in the sentence; ``secondary`` when it
    sits in a secondary predication (", extendable to ...")."""
    kind: AnchorKind
    word: str
    position: int
    secondary: bool = False


@dataclass(frozen=True)
class Restriction:
    """A restriction bound to a statement's predicate: its kind, its text and its content roots."""
    kind: RestrictionKind
    text: str
    roots: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class EnumItem:
    """One item of an enumeration ("Dental Medicine"), compared by its content roots."""
    text: str
    roots: FrozenSet[str]


@dataclass(frozen=True)
class TimeFiller:
    """A time expression of a statement; ``offset`` marks a relative offset ("six months before ..."), never an alternative."""
    text: str
    roots: FrozenSet[str]
    offset: bool = False


@dataclass(frozen=True)
class PredicateStatement:
    """One predicate statement of a returned quote (D-098): a finite clause together with its secondary predications. The
    subject, entity, value, enumeration and slot fields read the whole statement; the modality, the restrictions and the
    frequency read the anchored predicate only. Support is decided on one statement; others can only veto."""
    text: str
    predicate: str
    sentence: str
    unit_index: int
    sentence_index: int
    clause_index: int
    anchor: Anchor
    subject: Optional[Population]
    subject_text: str
    subject_inherited: bool
    modality: Modality
    negative: bool
    restrictions: Tuple[Restriction, ...]
    member_exceptions: Tuple[str, ...]
    hedges: Tuple[str, ...]
    frequency: Optional[FrequencyClass]
    noun_phrases: Tuple[NounPhrase, ...]
    exclusive_items: Tuple[Tuple[EnumItem, ...], ...]
    times: Tuple[TimeFiller, ...]
    numbers: FrozenSet[str]


@dataclass(frozen=True)
class LedgerEntry:
    """One operator occurrence of the asked claim: its class (frequency, scope, quantity, restriction, exclusivity, ordering,
    comparison, conditionality, residue), text, span, target slot and the dimension that must consume it ("static": an
    accepted C9/C10 reader verifies it outside the composite)."""
    cls: str
    text: str
    span: Tuple[int, int]
    slot: str
    handler: str


@dataclass(frozen=True)
class ClaimFrame:
    """The D-098 reading of a specific yes/no question: the unchanged C10 claim with its force, the asked frequency, the asked
    population read from the whole subject phrase, the universal noun-phrase slots, the enumeration and the operator ledger."""
    claim: Claim
    main: str
    frame: Optional[str]
    force: Force
    frequency: Optional[FrequencyClass]
    frequency_text: str
    subject: Optional[Population]
    slots: Tuple[NounPhrase, ...]
    items: Tuple[EnumItem, ...]
    ledger: Tuple[LedgerEntry, ...]
    anchor_roots: FrozenSet[str]


#: The closed occasion lexicon, longest phrases first within a class; the first class that matches is the asked frequency.
FREQUENCY_LEXICON: Tuple[Tuple[FrequencyClass, "re.Pattern[str]"], ...] = (
    (FrequencyClass.UNIVERSAL, re.compile(
        r"\b(?:in\s+(?:all|every|each)\s+cases?|in\s+all\s+circumstances|under\s+all\s+circumstances|at\s+all\s+times|every\s+time|each\s+time|"
        r"without\s+exception|in\s+(?:all|every)\s+instances?|always|invariably|automatically)\b", re.I)),
    (FrequencyClass.NEGATIVE_UNIVERSAL, re.compile(r"\b(?:under\s+no\s+circumstances|in\s+no\s+case|at\s+no\s+time|never)\b", re.I)),
    (FrequencyClass.EXISTENTIAL, re.compile(r"\b(?:in\s+some\s+cases|at\s+times|sometimes|occasionally|(?<!hardly\s)ever)\b", re.I)),
    (FrequencyClass.DEFAULT, re.compile(r"\b(?:as\s+a\s+rule|in\s+general|in\s+principle|by\s+default|usually|normally|generally|typically|ordinarily|"
                                        r"customarily)\b", re.I)),
    (FrequencyClass.PROPORTIONAL, re.compile(r"\b(?:hardly\s+ever|often|frequently|mostly|commonly|rarely|seldom|infrequently)\b", re.I)),
)
#: Predicate adjectives of possibility: "Is the contract extendable?" asks whether it can be extended (existential force).
POSSIBILITY_ADJECTIVES = re.compile(r"\b(?:extendable|extendible|renewable|transferable|refundable|negotiable|reducible|waivable)\b", re.I)
#: Restriction lexicons of a statement's predicate.
SPECIAL_CIRCUMSTANCES = re.compile(r"\b(?:in|under)\s+(?:special|exceptional|certain|specific|particular|extraordinary)\s+(?:circumstances?|cases?|situations?|"
                                   r"conditions?)\b|\bexceptionally\b|\bon\s+an\s+exceptional\s+basis\b", re.I)
HEDGES = re.compile(r"\b(?:as\s+a\s+rule|in\s+general|in\s+principle|normally|usually|generally|typically|ordinarily|customarily|often|frequently|mostly|"
                    r"preferably)\b", re.I)
OCCASION_EXCEPTION = re.compile(r"\bunless\b|\bexcept\s+(?:when|if|where)\b", re.I)
MEMBER_EXCEPTION = re.compile(r"\b(?:except(?:\s+for)?|excluding|other\s+than|with\s+the\s+exception\s+of)\b(?!\s+(?:when|if|where)\b)", re.I)
#: Alternative fillers of an exclusive value: a repetition (owned by the exclusivity window, D-098).
REPETITION = re.compile(
    r"\bagain\b|\b(?:a\s+)?(?:second|third|another)\s+time\b|\bonce\s+more\b|\bmore\s+than\s+once\b|\brepeat(?:ed|edly|s)?\b|\bmultiple\s+times\b|"
    r"\btwice\b|\bthrice\b|\b(?:second|third|another|additional|further)\s+(?:payment|term|period|contract|renewal|appointment|application|extension|"
    r"instal(?:l)?ments?|advance|allowance|time)\b|\bre-?(?:apply|applies|applied|appoint(?:ed|s|ment)?|submit(?:s|ted)?|elect(?:ed|s)?)\b", re.I)
REPETITION_NEGATED = re.compile(r"\b(?:not|no|cannot|never)\b[\s-]+(?:\w+[\s-]+){0,3}?(?:again|repeat\w*|second|another|re-?\w+)\b", re.I)
#: An explicit waiver of an exclusive condition ("without approval", "regardless of", "no approval is required").
WAIVER = re.compile(r"\bwithout\s+(?:\w+\s+){0,3}?(?:approval|consent|permission)\b|\bregardless\s+of\b|\bno\s+(?:\w+\s+)?(?:approval|permission)\s+is\s+"
                    r"(?:required|needed)\b|\bexempt(?:ed|s)?\b", re.I)
#: Another valid actor, population or object for the same predicate ("may also approve", "as well").
ADDITIVE = re.compile(r"\b(?:also|as\s+well|in\s+addition|alternatively)\b", re.I)
#: Markers of an exclusive enumeration ("limited to Medicine, Dental Medicine, ... and Law").
EXCLUSIVE_LIST_MARKERS = re.compile(r"\b(?:only|solely|exclusively|limited\s+to|restricted\s+to|no\s+other\s+than)\b", re.I)
#: Where an enumeration ends, and words dropped from an item.
ITEM_STOP = re.compile(r"[.;:?()]|\b(?:who|which|that|whose|whom|if|unless|when|where|provided|except|for\s+a\s+period|in\s+accordance|per\s+(?:contract|year))\b",
                       re.I)
ITEM_WRAPPER = re.compile(r"^(?:a|the)?\s*(?:decision|approval|recommendation|consent|endorsement|authori[sz]ation)\s+(?:of|from|by)\s+", re.I)
ITEM_DROP: FrozenSet[str] = frozenset("the a an of for to in on at by with from their its his her".split())
#: Time expressions and relative offsets.
TIME_FILLER = re.compile(r"\b(?:in|on|at|during|by|before|within|until|prior\s+to)\s+(?:the\s+|a\s+|an\s+|each\s+|every\s+|their\s+|its\s+)?(?:[\w'’-]+\s+){0,4}?"
                         r"(?:" + "|".join(sorted(TIME_NOUNS, key=len, reverse=True)) + r"|joining|appointment|\d+)\b", re.I)
RELATIVE_OFFSET = re.compile(r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|thirty|sixty|ninety)\s+(?:\w+\s+)?(?:days?|weeks?|months?|"
                             r"years?)\s+(?:before|after|prior\s+to)\b|\bat\s+least\b", re.I)
#: The closed residue vocabulary: operator-like words no dimension reads, consumed only by a literal restatement.
RESIDUE_WORDS: FrozenSet[str] = frozenset(("merely simply strictly fully entirely completely partially partly wholly necessarily directly immediately "
                                           "promptly still already even first last final same equal equivalent identical mandatory compulsory optional").split())
NEUTRAL_LY: FrozenSet[str] = frozenset("only early family daily weekly monthly yearly annually quarterly newly likely apply reply supply comply rely july italy "
                                       "holy ugly".split())
EQUIVALENT_WORDS: Tuple[FrozenSet[str], ...] = (frozenset({"equal", "equivalent", "same", "identical"}),)
#: The force x restriction table (D-098): the restriction kinds that leave an unconditional claim of that force unknown.
FORCE_RESTRICTIONS: Dict[Force, FrozenSet[RestrictionKind]] = {
    Force.EXISTENTIAL: frozenset(),
    Force.GENERIC: frozenset({RestrictionKind.NECESSARY, RestrictionKind.SPECIAL}),
    Force.DEONTIC: frozenset({RestrictionKind.NECESSARY, RestrictionKind.SPECIAL}),
    Force.UNIVERSAL_QUANTIFIED: frozenset({RestrictionKind.NECESSARY, RestrictionKind.EXCEPTION, RestrictionKind.SPECIAL, RestrictionKind.HEDGE}),
    Force.UNIVERSAL_FREQUENCY: frozenset({RestrictionKind.NECESSARY, RestrictionKind.EXCEPTION, RestrictionKind.SPECIAL, RestrictionKind.HEDGE,
                                          RestrictionKind.GATE, RestrictionKind.SUFFICIENT, RestrictionKind.TRIGGER, RestrictionKind.POSSIBILITY}),
}
#: The joint comparison's dimensions, in the order of the first-failure message.
DIMENSIONS: Tuple[str, ...] = ("polarity", "subject", "predicate", "object", "modality", "scope", "frequency", "condition", "exclusivity", "value_time",
                               "operator")


def frequency_of(text: str) -> Tuple[Optional[FrequencyClass], str, Tuple[int, int]]:
    """The occasion operator of ``text``: its class, its words and its span (the first class of the lexicon that matches)."""
    for cls, pattern in FREQUENCY_LEXICON:
        m = pattern.search(text)
        if m is not None:
            return cls, m.group(0).lower(), m.span()
    return None, "", (0, 0)


def force_of(modality: Modality, subject_quantifier: Optional[Quantifier], main: str) -> Force:
    """The force of a claim, first match wins: a universal or negative-universal frequency; a universal word on a population
    or noun-phrase slot (a time distributive is not one); an existential modality, quantifier, frequency or possibility
    adjective; an obligation, prohibition, recommendation or encouragement; else generic ("eligible" and "entitled" stay generic)."""
    freq, _w, _s = frequency_of(main)
    if freq in (FrequencyClass.UNIVERSAL, FrequencyClass.NEGATIVE_UNIVERSAL):
        return Force.UNIVERSAL_FREQUENCY
    if subject_quantifier is Quantifier.UNIVERSAL or any(quantifier_of(w) is Quantifier.UNIVERSAL for w, slot, _h, _sp in quantifier_phrases(main)
                                                          if slot in ("population", "object")):
        return Force.UNIVERSAL_QUANTIFIED
    if (freq is FrequencyClass.EXISTENTIAL or modality in (Modality.PERMISSION, Modality.POSSIBILITY) or subject_quantifier is Quantifier.EXISTENTIAL
            or POSSIBILITY_ADJECTIVES.search(main)):
        return Force.EXISTENTIAL
    if modality in (Modality.OBLIGATION, Modality.PROHIBITION, Modality.RECOMMENDATION, Modality.ENCOURAGEMENT):
        return Force.DEONTIC
    return Force.GENERIC


def restriction_compatibility(force: Force, kinds: Iterable[RestrictionKind]) -> Compatibility:
    """An unconditional claim of ``force`` against the restrictions bound to the statement's predicate (the D-098 table)."""
    return Compatibility.UNKNOWN if set(kinds) & FORCE_RESTRICTIONS[force] else Compatibility.COMPATIBLE


def frequency_compatibility(asked: FrequencyClass, predicate: str, kinds: Iterable[RestrictionKind], negative: bool) -> Compatibility:
    """An asked occasion operator against a statement's predicate: a universal one needs an explicit universal frequency and no
    restriction of any kind but a concessive one; a negative universal ("never" = not + ever) the predicate negated with no
    restriction; an existential one an affirmative statement; a default or proportional one the same class restated."""
    kinds = set(kinds)
    stated, _w, _s = frequency_of(predicate)
    if asked is FrequencyClass.UNIVERSAL:
        ok = stated is FrequencyClass.UNIVERSAL and not (kinds & FORCE_RESTRICTIONS[Force.UNIVERSAL_FREQUENCY])
        return Compatibility.COMPATIBLE if ok else Compatibility.UNKNOWN
    if asked is FrequencyClass.NEGATIVE_UNIVERSAL:
        ok = negative and not (kinds & (FORCE_RESTRICTIONS[Force.UNIVERSAL_FREQUENCY] - {RestrictionKind.POSSIBILITY}))
        return Compatibility.COMPATIBLE if ok else Compatibility.UNKNOWN
    if asked is FrequencyClass.EXISTENTIAL:
        return Compatibility.UNKNOWN if negative else Compatibility.COMPATIBLE
    same = any(cls is asked and pattern.search(predicate) for cls, pattern in FREQUENCY_LEXICON)
    return Compatibility.COMPATIBLE if same else Compatibility.UNKNOWN


def noun_phrase_after(words: Sequence[str], i: int, stop: Callable[[str], bool]) -> Tuple[str, Tuple[str, ...]]:
    """The noun phrase after a quantifier at ``words[i]`` (the shared reader of both sides): up to five words, ending before a
    break word, a determiner, "only" / "also", a verb-group word or a word ``stop`` names (the asked verb). Returns the head
    and the words before it ("" when there is none)."""
    out: List[str] = []
    for w in words[i + 1:i + 6]:
        if w in NP_BREAK or w in NP_STOP_WORDS or stop(w):
            break
        out.append(w)
    if not out:
        return "", ()
    return out[-1], tuple(out[:-1])


#: Words that end a quantified noun phrase besides NP_BREAK.
NP_STOP_WORDS: FrozenSet[str] = frozenset("a an the their its his her this these those only also cannot might".split())


def slot_universal_compatibility(asked: NounPhrase, stated: Sequence[NounPhrase], same_root: Callable[[str, str], bool]) -> Compatibility:
    """An asked universal noun phrase against a statement's quantified noun phrases: an explicit universal over the same head,
    modifiers no narrower than the ask, and no number or cap inside it; otherwise unknown."""
    for s in stated:
        if s.quantifier is not Quantifier.UNIVERSAL or not same_root(s.head, asked.head) or s.capped:
            continue
        if all(any(same_root(m, a) for a in asked.modifiers) for m in s.modifiers):
            return Compatibility.COMPATIBLE
    return Compatibility.UNKNOWN


def enumeration_items(text: str, start: int, root: Callable[[str], str]) -> Tuple[EnumItem, ...]:
    """The items of an enumeration starting at ``start``: up to the first stop (punctuation or a clause opener), split at commas,
    "and", "or", "as well as" and "&"; each item keeps its modifiers and drops leading determiners and procedural wrappers
    ("a decision from the Disciplinary Council" -> Disciplinary Council)."""
    rest = text[start:]
    m = ITEM_STOP.search(rest)
    region = (rest[:m.start()] if m else rest)[:240]
    out: List[EnumItem] = []
    for part in re.split(r",\s*(?:and|or)\s+|\s*,\s*|\s+(?:and|or|as\s+well\s+as|&)\s+", region):
        part = ITEM_WRAPPER.sub("", re.sub(r"^\s*(?:the|a|an|their|its)\s+", "", part.strip(), flags=re.I))
        roots = frozenset(root(w) for w, _s, _e in tokens(part) if w not in ITEM_DROP)
        if roots:
            out.append(EnumItem(part.strip(), roots))
    return tuple(out)


def enumeration_compatibility(asked: Sequence[EnumItem], stated: Sequence[Sequence[EnumItem]]) -> Compatibility:
    """An exclusive enumeration is stated only by the same item set: an order-free one-to-one match of item roots. A subset, a
    superset or an overlap is unknown (never contradicted)."""
    want = sorted(sorted(i.roots) for i in asked)
    for items in stated:
        if sorted(sorted(i.roots) for i in items) == want:
            return Compatibility.COMPATIBLE
    return Compatibility.UNKNOWN


def residue_words(main: str) -> Tuple[str, ...]:
    """The operator-like words of a question no dimension reads: the closed residue vocabulary and the "-ly" adverbs (four
    letters and more, the neutral ones excepted), frequency words excluded (the frequency dimension reads them)."""
    out = []
    for w, _s, _e in tokens(main):
        if w in RESIDUE_WORDS or (w.endswith("ly") and len(w) > 4 and w not in NEUTRAL_LY):
            if not any(pattern.fullmatch(w) for _c, pattern in FREQUENCY_LEXICON) and w not in out:
                out.append(w)
    return tuple(out)


def restated(word: str, text: str) -> bool:
    """A residue word is consumed by a literal restatement, or by a word of its closed equivalence class."""
    group = next((g for g in EQUIVALENT_WORDS if word in g), frozenset({word}))
    return any(re.search(r"\b" + re.escape(g) + r"\b", text, re.I) for g in group)


def ledger_compatibility(ledger: Sequence[LedgerEntry], evaluated: FrozenSet[str], text: str) -> DimensionResult:
    """Every operator of the claim must be consumed on the bound statement: by the dimension that evaluated it, by a literal
    restatement (residue), or by an accepted reader outside the composite ("static"); otherwise the claim fails closed
    (unknown, never contradicted)."""
    for e in ledger:
        if e.handler == "static":
            continue
        if e.handler == "residue":
            if not restated(e.text, text):
                return DimensionResult("operator", Compatibility.UNKNOWN, "the asked %s '%s' is not restated" % (e.cls, e.text))
            continue
        if e.handler not in evaluated:
            return DimensionResult("operator", Compatibility.UNKNOWN, "the asked %s '%s' is not verified" % (e.cls, e.text))
    return DimensionResult("operator", Compatibility.NOT_APPLICABLE if not ledger else Compatibility.COMPATIBLE)


def compare_claim_to_statement(frame: ClaimFrame, statement: PredicateStatement,
                               checks: Dict[str, Callable[[ClaimFrame, PredicateStatement], Optional[DimensionResult]]],
                               ledger: bool = True) -> Tuple[DimensionResult, ...]:
    """The joint comparison (D-098): every dimension evaluated on the SAME statement, in the fixed order; a dimension check
    returns None when it does not apply to the claim. The ledger then requires every operator to have been consumed by a
    dimension that evaluated it here."""
    results: List[DimensionResult] = []
    evaluated = set()
    for name in DIMENSIONS[:-1]:
        check = checks.get(name)
        r = check(frame, statement) if check is not None else None
        if r is not None:
            evaluated.add(name)
            results.append(r)
    if ledger:
        results.append(ledger_compatibility(frame.ledger, frozenset(evaluated), statement.text))
    return tuple(results)


def first_failure(results: Iterable[DimensionResult]) -> Optional[DimensionResult]:
    """The first result, in dimension order, that is not compatible (the missing message)."""
    for r in results:
        if r.compatibility in (Compatibility.CONTRADICTED, Compatibility.UNKNOWN):
            return r
    return None


__all__ = [
    "Breadth", "Claim", "Compatibility", "Condition", "ConditionRole", "DimensionResult", "Exclusivity", "ExclusivityFocus", "Modality",
    "Operator", "OperatorKind", "Polarity", "Population", "QualifiedValue", "Quantifier", "Statement", "approval_phrases", "ask_modality",
    "clause_start", "combine", "condition_role", "condition_role_compatibility", "exclusivity_compatibility", "find_operators",
    "modal_compatibility", "quantifier_of", "quantifier_phrases", "read_exclusivity", "scope_compatibility", "statement_modality",
    "tokens", "unhandled", "unit_modality",
    "Anchor", "AnchorKind", "ClaimFrame", "DIMENSIONS", "EnumItem", "FORCE_RESTRICTIONS", "Force", "FrequencyClass", "LedgerEntry", "NounPhrase",
    "PredicateStatement", "Restriction", "RestrictionKind", "TimeFiller", "compare_claim_to_statement", "enumeration_compatibility",
    "enumeration_items", "first_failure", "force_of", "frequency_compatibility", "frequency_of", "ledger_compatibility", "noun_phrase_after",
    "residue_words", "restated", "restriction_compatibility", "slot_universal_compatibility",
]
