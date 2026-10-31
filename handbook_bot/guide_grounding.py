"""
Guide grounding, evidence units and verified procedure assembly (post-freeze PF-3C, decision record D-111).

WHY
    PF-3B (D-110) let approved official guides reach the specialists but served no guide answer. The frozen specialists
    read a system manual with handbook rules (the asked system and the asking actor must be printed in the answering
    sentence; a procedure request is never supported), and the PF-3C measurement showed that their quote selection,
    not the retrieval order, chose the off-target evidence: guide-scoped retrieval ranks an on-target chunk first for
    30 of 37 guide questions, but those chunks are page-wide windows, and the sentence chosen from them was the one
    naming the system, usually the guide's introduction. This module is the one place where guide evidence is bound
    to a question, selected and verified. Handbook evidence never passes through it.

WHAT (deterministic; no model; no file read)
    * ``read_intent``: the request kind of a guide clause, read with the findings verifier's request reader (I-4)
      plus three guide readings (a "how do I" procedure ask, a "what can I ..." capability ask, "what are the N ...").
      Policy, authority, eligibility, obligation, timing and frequency asks are never answered from a guide.
    * ``bind``: the D-111 binding of a clause to ONE guide. Metadata establishes context, text establishes the claim:
        - the source must be usable (registered, enabled, not internal use, not held), the clause must name the
          system the guide documents (the D-110 vocabulary) and must not be a policy ask;
        - the guide's DECLARED system names found in the clause (``source_scopes.SYSTEM_NAMES``) are bound by the
          registry and need not be repeated by the evidence;
        - only a document the registry records for a faculty or instructor audience answers at all, and in it a
          first-person or faculty actor ("I", "my", "faculty", "instructor") is bound by that audience;
        - every other content word (actions, objects, conditions, other roles, numbers) is a TERM the evidence must
          state, and every named entity left after binding must appear in the evidence.
    * ``statement_units``: exact sub-quotes of one chunk: a procedure step, a sentence, or two adjacent sentences.
    * ``GuideIndex``: per guide page, the procedures read with the PF-2 chunker's own rules (checked against the
      ``procedure_step`` chunks) with the PROVENANCE of each heading (a heading row, a lead-in row, or only carried
      from the page hint or the previous list); a numbered list that continues across a page break is merged.
    * ``procedure_problem``: why a procedure is not provably complete. ``select_answer`` chooses the evidence for a
      clause (a complete procedure, one item of an unordered list, or one statement); ``verify_answer`` is the
      independent re-check the adapter applies to whatever a specialist returns.
    * Post-freeze PF-5B (D-114), natural faculty wording: discourse words are not terms; the default modifier "final"
      in a procedure ask is not a term when the guide's registry record declares final grades and the guide text names
      neither final nor midterm; a term is also stated by a written-together compound ("log in" / "login") or by the
      other member of two noun synonym pairs; a clause about a physical place is never a guide's; a registered VALUE
      conflict blocks the units that state the disputed value, not every unit of its chunk; a unit cut mid-sentence by
      the extraction, or a statement that announces content it does not contain, is never served.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .chunking import _clean_heading, _is_list_lead_in, _is_procedure_heading, _match_step
from .source_scopes import (GUIDE_DOCUMENT_TYPES, SYSTEM_NAMES, SYSTEM_TERMS, _HOWTO_ASK, _POLICY_ASK, faculty_audience,
                            usable_record)
from .text_utils import normalize_text

#: Contract version of this module (carried in every guide clause report and finding).
VERSION = "guide-grounding-1"
PHASE = "PF-3C"
#: Key of the D-111 record in a clause report and in a finding's metadata.
GUIDE_KEY = "guide"

FORM_PROCEDURE = "procedure"        # every step of one complete procedure, in source order
FORM_LIST_ITEM = "list_item"        # one item of an unordered (bulleted) list that is not a procedure
FORM_STATEMENT = "statement"        # one statement (a step, a sentence or two adjacent sentences) of one chunk
FORM_SECTION = "section"            # post-freeze PF-3D (D-112): a recovered section or list, every unit in source order
FORMS = (FORM_PROCEDURE, FORM_LIST_ITEM, FORM_STATEMENT, FORM_SECTION)


class GuideIntent(str, Enum):
    PROCEDURE = "procedure"
    LOCATION = "location"
    CONTACT = "contact"
    VALUE = "value"
    YES_NO = "yes_no"
    DEFINITION = "definition"
    ENUMERATION = "enumeration"
    SELECTION = "selection"
    AUTHORITY = "authority"          # who approves / is responsible / is eligible / must: the handbook's question
    POLICY = "policy"                # policy, entitlement, obligation, deadline or timing: the handbook's question
    UNRECOGNISED = "unrecognised"


#: Intents a guide may answer. Every other intent is left to the frozen handbook path.
ANSWERABLE_INTENTS = frozenset({GuideIntent.PROCEDURE, GuideIntent.LOCATION, GuideIntent.CONTACT, GuideIntent.VALUE, GuideIntent.YES_NO,
                                GuideIntent.DEFINITION, GuideIntent.ENUMERATION, GuideIntent.SELECTION})

_CAPABILITY_ASK = re.compile(r"^\s*what\s+(?:can|could|may)\s+(?:i|we|you|faculty|instructors?)\s+\w+", re.I)
_PLURAL_WHAT = re.compile(r"^\s*what\s+are\s+(?:the\s+)?(?:[a-z0-9]+\s+){0,2}[a-z]+s\b", re.I)


def read_intent(clause: str) -> Tuple[GuideIntent, Dict[str, Any]]:
    """The guide intent of ``clause`` and what the request reader saw (kind, contact field, counted unit, object head)."""
    text = (clause or "").strip()
    if not text:
        return GuideIntent.UNRECOGNISED, {}
    if _POLICY_ASK.search(text) or _HANDBOOK_SUBJECT.search(text):
        return GuideIntent.POLICY, {"request_kind": "policy"}
    from .plan_f_verification import RequestKind, read_request      # imported here: that module imports the adapter
    shape = read_request(text)
    kind = shape.kind
    info = {"request_kind": kind.value, "field": shape.field, "unit": shape.unit, "object_head": shape.object_head}
    if kind in (RequestKind.AUTHORITY, RequestKind.RESPONSIBILITY, RequestKind.ELIGIBILITY, RequestKind.OBLIGATION):
        return GuideIntent.AUTHORITY, info
    if kind in (RequestKind.TIMING, RequestKind.FREQUENCY):
        return GuideIntent.POLICY, info
    if kind is RequestKind.CONTACT:
        return GuideIntent.CONTACT, info
    if kind is RequestKind.LOCATION:
        return GuideIntent.LOCATION, info
    if kind in (RequestKind.QUANTITY, RequestKind.AMOUNT, RequestKind.PERCENTAGE, RequestKind.DURATION, RequestKind.VALUE):
        return GuideIntent.VALUE, info
    if kind is RequestKind.YES_NO:
        return GuideIntent.YES_NO, info
    if _HOWTO_ASK.search(text):
        return GuideIntent.PROCEDURE, info
    if kind is RequestKind.ENUMERATION or _CAPABILITY_ASK.search(text) or _PLURAL_WHAT.search(text):
        return GuideIntent.ENUMERATION, info
    if kind is RequestKind.SELECTION:
        return GuideIntent.SELECTION, info
    if kind is RequestKind.DEFINITION:
        return GuideIntent.DEFINITION, info
    return GuideIntent.UNRECOGNISED, info


# ---------------------------------------------------------------------------
# Terms, stems and the D-111 binding
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
_STOP = frozenset("""
a an the this that these those it its it's their there here they them his her he she him one ones
of for to from by with without on in at during after before under per and or nor if when while unless within into onto about than as
through via over across up down out off upon also just then so such
is are was were be been being am do does did done doing can could may might must shall should will would has have had having
what which who whom whose how when where why whether
any all each every some no not other own same more most very
please kindly tell explain describe
""".split())
#: Post-freeze PF-5B (D-114): greetings and discourse words that never carry what a guide must state ("ok so", "hey",
#: "honestly"). Scope words ("specific", "particular", "current", "whole", "already") restrict the ask and stay terms;
#: "quick" stays a term (a guide prints "Quick Links"); determiners such as "all", "this" and "those" were already stop words.
_NON_CRITICAL = frozenset("""
ok okay hey hi hello thanks thank honestly
""".split())
#: Post-freeze PF-5B (D-114): DEFAULT modifiers and the marked members they contrast with. Only the unmarked default may be
#: left unstated by the evidence, and only (see ``bind``) as an attributive modifier in a procedure ask ("send my final
#: grades"), when the guide's registry record declares it ("transferring final grades") and the guide text states neither it
#: nor a marked partner. A marked member ("midterm", "group", "optional", "required", "available") is always a term.
_DEFAULT_MODIFIERS: Mapping[str, FrozenSet[str]] = {"final": frozenset({"midterm", "mid", "interim", "partial"})}
#: Post-freeze PF-5B (D-114): the only synonym pairs, nouns and adjectives with one meaning in a system manual; applied to term
#: coverage only. Verb synonyms ("remove" for "delete") are not used: a bag of words cannot tell what they act on.
_SYNONYM_GROUPS: Tuple[FrozenSet[str], ...] = tuple(frozenset(g) for g in (("online", "virtual"), ("student", "learner")))
#: "before" and "after" stay function words (D-111); a statement that states only the opposite one is refused (D-114).
_POLARITY = {"before": "after", "after": "before"}
#: Question frame verbs that never carry the asked content ("Where do I find ...", "How do I make Blackboard assign ...").
_FRAME = frozenset({"find", "get", "use", "make", "see", "go", "need", "want", "know", "able", "possible", "way", "help", "let", "try"})
#: Words a request kind supplies itself; the evidence answers them with its value, place, channel or set.
_INTENT_WORDS: Mapping[GuideIntent, FrozenSet[str]] = {
    GuideIntent.VALUE: frozenset({"minimum", "maximum", "min", "max", "size", "length", "limit", "limits", "number", "many", "much", "long",
                                  "large", "big", "small", "largest", "smallest", "longest", "shortest", "biggest", "least", "amount", "total"}),
    GuideIntent.LOCATION: frozenset({"located", "location", "locate"}),
    GuideIntent.CONTACT: frozenset({"email", "e", "mail", "contact", "phone", "telephone", "number", "reach", "address", "call", "write"}),
    GuideIntent.PROCEDURE: frozenset({"step", "steps", "procedure", "process", "instructions", "instruction", "walk", "guide"}),
    GuideIntent.DEFINITION: frozenset({"mean", "means", "meaning", "definition"}),
}
#: A first-person or faculty actor: bound by the registry audience of a faculty or instructor document (D-111 B).
_ACTOR = re.compile(r"(?<![A-Za-z0-9])(?:i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|yourself|"
                    r"faculty(?:\s+members?)?|instructors?|teachers?|professors?|lecturers?)(?:'s|’s|')?(?![A-Za-z0-9])", re.I)
#: A preposition or conjunction that closes the core (action and object) of a "how do I ..." clause.
_CORE_END = frozenset("""
to from in on at with for of into onto via by without while when after before so if as through during within under over using
""".split())
_HOWTO_FRAME = re.compile(r"^\s*(?:how|what\s+are\s+the\s+steps\s+to|what\s+steps\s+do)\s+(?:does|do|can|could|should|would|to)?\b\s*"
                          r"(?:(?:i|we|you|one)\s+)?(?:go\s+about\s+)?", re.I)
#: Handbook-only subjects named inside a question that otherwise reads as a system question ("What is the required teaching
#: load for faculty who teach on Blackboard?"): workload, hours and entitlements are policy, never a manual's.
_HANDBOOK_SUBJECT = re.compile(r"\brequired\s+(?:teaching|work|number|hours?|load)\b|\bteaching\s+loads?\b|\bwork\s?loads?\b|"
                               r"\boffice\s+hours\b|\bcredit\s+hours?\b|\bentitle\w*|\beligib\w*|\ballowances?\b|\bsalar(?:y|ies)\b", re.I)
#: Post-freeze PF-5B (D-114): a physical or organizational place. A clause about one (an office, a building, a department)
#: is Institutional Navigation's, never a system manual's; a place inside the software (a menu, a page, a Collaborate room)
#: is the manual's. The list over-refuses rare in-software uses ("lab grades", a "Help Center" page): a fallback, never a
#: wrong answer.
_PHYSICAL_PLACE = re.compile(
    r"\b(?:offices?|buildings?|campus(?:es)?|help\s?desks?|desks?|departments?|deanships?|centers?|centres?|halls?|"
    r"floors?|librar(?:y|ies)|labs?|laborator(?:y|ies)|room\s+(?:number|no\b|\d))\b", re.I)
#: Post-freeze PF-5B (D-114): a clause that names an organizational unit, place or role (an office, a team, staff, an
#: administrator, the registrar, ITC, HR, a dean or chair, a contact channel) is never treated as a pure software question:
#: the specialist's handoff of it to Institutional Navigation is kept even when a guide binds it.
ORGANIZATIONAL = re.compile(
    _PHYSICAL_PLACE.pattern + r"|\b(?:rooms?|spaces?|teams?|staff|admins?|administrators?|coordinators?|trainers?|units?|colleges?|registrar|"
    r"registration|itc|it\s+(?:department|center|centre|services?)|human\s+resources|hr|deans?|chairs?|chairpersons?|heads?|"
    r"support|helpdesk|contacts?|e-?mails?|phones?|extensions?|who|whom)\b", re.I)


def _core_text(blanked: str) -> str:
    """The action and object of a "how do I ..." clause, up to its first preposition (D-114: "How do I find the support
    office?" asks for a place; "How do I log in from outside the campus?" does not)."""
    rest = _HOWTO_FRAME.sub("", blanked, count=1)
    words: List[str] = []
    for tok in _TOKEN.findall(rest):
        if tok.lower() in _CORE_END:
            if words:
                break
            continue
        words.append(tok)
    return " ".join(words)


def stem(word: str) -> str:
    """A deterministic light stemmer, applied identically to clause and evidence (a matching key, not linguistics)."""
    w = word.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    w = w.strip("'")
    if not w or w[0].isdigit():
        return w
    if len(w) > 5 and w.endswith("ically"):
        w = w[:-4]
    elif len(w) > 4 and w.endswith("ly"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suffix in ("ing", "ed"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            w = w[: -len(suffix)]
            if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


_JOINABLE = re.compile(r"^[\s-]{1,2}$")


def stems(text: str) -> FrozenSet[str]:
    """Stems of every word of ``text`` (stop words included: coverage asks whether a term is stated, nothing more). Post-freeze
    PF-5B (D-114): also the stem of every two adjacent words written together ("log in" -> "login", "grade book" ->
    "gradebook", "e-mail" -> "email"), so a compound is stated however it is spaced."""
    text = text or ""
    found = [m for m in _TOKEN.finditer(text)]
    out = {stem(m.group(0)) for m in found}
    for a, b in zip(found, found[1:]):
        if _JOINABLE.match(text[a.end():b.start()]) and a.group(0).isalpha() and b.group(0).isalpha():
            out.add(stem(a.group(0) + b.group(0)))
    return frozenset(out)


_SYNONYMS: Dict[str, FrozenSet[str]] = {}
for _group in _SYNONYM_GROUPS:
    _stems = frozenset(stem(w) for w in _group)
    for _s in _stems:
        _SYNONYMS[_s] = _SYNONYMS.get(_s, frozenset()) | _stems


def _name_pattern(name: str) -> "re.Pattern[str]":
    return re.compile(r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(w) for w in name.split()) + r"(?![A-Za-z0-9])", re.I)


def _blank(text: str, spans: Sequence[Tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


@dataclass(frozen=True)
class Binding:
    """What ONE guide source must state to answer ONE clause (D-111)."""

    source_id: str
    intent: GuideIntent
    names_bound: Tuple[str, ...]          # declared system names found in the clause (bound by the registry)
    actor_bound: bool                     # a first-person or faculty actor bound by the registry audience
    terms: Tuple[str, ...]                # stems the evidence must state
    core: Tuple[str, ...]                 # stems of the asked action and object (procedure asks)
    entities: Tuple[str, ...]             # named entities left after binding: printed in the evidence
    minimum: bool = False                 # a value ask for a lower bound ("minimum", "at least")
    maximum: bool = False                 # a value ask for an upper bound ("maximum", "largest", "limit")
    info: Mapping[str, Any] = field(default_factory=dict)
    # post-freeze PF-5B (D-114)
    alternatives: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()   # term -> the compound stems it forms with a neighbour
    tolerated: Tuple[str, ...] = ()       # contrast modifiers the guide never contrasts (not terms)
    polarity: Tuple[str, ...] = ()        # "before" / "after" as asked

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable; the registry's system names and counts only (no question text, no word of it)."""
        return {"source_id": self.source_id, "intent": self.intent.value, "names_bound": list(self.names_bound), "actor_bound": self.actor_bound,
                "terms": len(self.terms), "core": len(self.core), "entities": len(self.entities), "minimum": self.minimum, "maximum": self.maximum,
                "tolerated": len(self.tolerated)}

    def equivalents(self, term: str) -> FrozenSet[str]:
        """The stems that state ``term`` (D-114): itself, its synonym group and the compounds it forms in the clause."""
        alts = dict(self.alternatives).get(term, ())
        return frozenset((term,) + tuple(alts)) | _SYNONYMS.get(term, frozenset())


_MIN_WORDS = re.compile(r"\b(?:minimum|min|least|smallest|shortest)\b", re.I)
_MAX_WORDS = re.compile(r"\b(?:maximum|max|most|largest|longest|biggest|limit|limits|limited)\b", re.I)


def bind(clause: str, record: Any, vocabulary: Optional[FrozenSet[str]] = None) -> Tuple[Optional[Binding], Optional[str]]:
    """``(binding, None)`` when ``record`` may answer ``clause`` under D-111, else ``(None, reason)``. Never raises for
    ordinary input; the reason is a fixed phrase plus ids (no question text). ``vocabulary`` (post-freeze PF-5B, D-114) is
    the stem set of the whole guide (``GuideIndex.vocabulary``): without it every contrast modifier is a term (strict)."""
    text = (clause or "").strip()
    if not text:
        return None, "empty clause"
    if not usable_record(record):
        return None, "source is not usable (unregistered, disabled, internal use or held)"
    source_id = record.source_id
    if record.document_type not in GUIDE_DOCUMENT_TYPES:
        return None, "source %s is not a system guide" % source_id
    if not faculty_audience(record):
        # every Plan F question comes from a faculty member: a student-only, staff-only or unknown-audience guide never answers
        return None, "%s is not registered for a faculty or instructor audience" % source_id
    pattern = SYSTEM_TERMS.get(source_id)
    if pattern is None or not pattern.search(text):
        return None, "the clause does not name the system documented by %s" % source_id
    intent, info = read_intent(text)
    if intent not in ANSWERABLE_INTENTS:
        return None, "a %s ask is not answered from a system guide" % intent.value
    spans: List[Tuple[int, int]] = []
    names: List[str] = []
    for name in sorted(SYSTEM_NAMES.get(source_id, ()), key=len, reverse=True):
        for m in _name_pattern(name).finditer(text):
            if any(not (m.end() <= a or m.start() >= b) for a, b in spans):
                continue
            spans.append((m.start(), m.end()))
            names.append(name)
    actors = [(m.start(), m.end()) for m in _ACTOR.finditer(text) if not any(not (m.end() <= a or m.start() >= b) for a, b in spans)]
    blanked = _blank(text, spans + actors)                         # the faculty actor is bound by the audience checked above
    if _PHYSICAL_PLACE.search(blanked) and (intent is not GuideIntent.PROCEDURE or _PHYSICAL_PLACE.search(_core_text(blanked))):
        return None, "a physical or organizational place is not answered from a system guide"
    from .plan_f_runtime import named_entities                    # imported here: the adapter imports this module
    entities = tuple(named_entities(blanked))
    drop = {stem(w) for w in _INTENT_WORDS.get(intent, frozenset())}
    head = info.get("object_head")
    if intent in (GuideIntent.SELECTION, GuideIntent.ENUMERATION) and isinstance(head, str) and head:
        drop.add(stem(head))                                       # the asked control or set is what the evidence supplies

    tolerated: List[str] = []
    if intent is GuideIntent.PROCEDURE and vocabulary is not None:
        declared = stems(" ".join(str(getattr(record, k, "") or "") for k in ("title", "notes", "system")))
        tokens = list(_TOKEN.finditer(blanked))
        for i, m in enumerate(tokens):
            low = m.group(0).lower()
            partners = _DEFAULT_MODIFIERS.get(low)
            if partners is None or i + 1 >= len(tokens):
                continue
            following = tokens[i + 1].group(0).lower()
            attributive = following not in _STOP and following not in _FRAME and not blanked[m.end():tokens[i + 1].start()].strip()
            silent = not ({stem(low)} | {stem(p) for p in partners}) & vocabulary
            if attributive and silent and stem(low) in declared:
                tolerated.append(stem(low))                        # D-114: the registry declares it and the text is silent
    drop |= set(tolerated)

    def content(tok: str) -> bool:
        low = tok.lower()
        return low not in _STOP and low not in _FRAME and low not in _NON_CRITICAL and stem(tok) not in drop

    terms = tuple(dict.fromkeys(stem(t) for t in _TOKEN.findall(blanked) if content(t)))
    alternatives: Dict[str, List[str]] = {}
    found = list(_TOKEN.finditer(blanked))
    for a, b in zip(found, found[1:]):
        if _JOINABLE.match(blanked[a.end():b.start()]) and a.group(0).isalpha() and b.group(0).isalpha():
            joined = stem(a.group(0) + b.group(0))
            for tok in (a.group(0), b.group(0)):
                if content(tok):
                    alternatives.setdefault(stem(tok), [])
                    if joined not in alternatives[stem(tok)]:
                        alternatives[stem(tok)].append(joined)
    core: Tuple[str, ...] = ()
    if intent is GuideIntent.PROCEDURE:
        rest = _HOWTO_FRAME.sub("", blanked, count=1)
        words: List[str] = []
        for tok in _TOKEN.findall(rest):
            if tok.lower() in _CORE_END:
                if words:
                    break
                continue
            if content(tok):
                words.append(stem(tok))
        core = tuple(dict.fromkeys(words))
    if not terms and not entities:
        return None, "the clause asks nothing of %s beyond its system name" % source_id
    polarity = tuple(w for w in ("before", "after") if re.search(r"\b%s\b" % w, blanked, re.I))
    binding = Binding(source_id, intent, tuple(names), bool(actors), terms, core, entities,
                      minimum=intent is GuideIntent.VALUE and bool(_MIN_WORDS.search(blanked)),
                      maximum=intent is GuideIntent.VALUE and bool(_MAX_WORDS.search(blanked)) and not _MIN_WORDS.search(blanked),
                      info=dict(info), alternatives=tuple((k, tuple(v)) for k, v in alternatives.items() if k in terms),
                      tolerated=tuple(tolerated), polarity=polarity)
    return binding, None


def stated_terms(binding: Binding, terms: Iterable[str], text: str) -> FrozenSet[str]:
    """The ``terms`` of ``binding`` that ``text`` states (post-freeze PF-5B, D-114): the stem itself, a compound it forms in
    the clause, or the other member of its synonym pair."""
    present = stems(text)
    return frozenset(term for term in terms if binding.equivalents(term) & present)


def missing_terms(binding: Binding, text: str) -> List[str]:
    """Terms of ``binding`` that ``text`` does not state."""
    stated = stated_terms(binding, binding.terms, text)
    return [t for t in binding.terms if t not in stated]


def missing_entities(binding: Binding, text: str) -> List[str]:
    from .plan_f_runtime import entity_in_text
    return [e for e in binding.entities if not entity_in_text(e, text)]


# ---------------------------------------------------------------------------
# What a statement must show for each intent
# ---------------------------------------------------------------------------
#: A stated quantity: a number with its unit (a list label such as "1." is never a value).
_QUANTITY = re.compile(r"(?<![\w.])\d+(?:[,.]\d+)*\s*(?:x\s*\d+(?:[,.]\d+)*\s*)?(?:%|percent\b|characters?\b|pixels?\b|px\b|pages?\b|mb\b|kb\b|gb\b|"
                       r"words?\b|files?\b|documents?\b|docs?\b|times?\b|minutes?\b|hours?\b|days?\b|weeks?\b|months?\b|years?\b|points?\b|"
                       r"items?\b|students?\b|columns?\b|rows?\b|submissions?\b)", re.I)
_LEADING_LABEL = re.compile(r"^\s*(?:\d{1,2}[.)]|[A-Za-z][.)]|[-–•])\s+")
_MIN_EVIDENCE = re.compile(r"\bat\s+least\b|\bminimum\b|\bmin\b|\bno\s+less\s+than\b|\bnot\s+less\s+than\b", re.I)
_MAX_EVIDENCE = re.compile(r"\bat\s+most\b|\bmaximum\b|\bmax\b|\bless\s+than\b|\bup\s+to\b|\blimited\s+to\b|\bno\s+more\s+than\b|"
                           r"\b(?:must\s+)?not\s+(?:to\s+)?exceed\b|\bcan(?:not|'t|’t)\s+be\s+larger\b|\bno\s+larger\b", re.I)
_YES_NO_EVIDENCE = re.compile(r"\b(?:can|cannot|can't|can’t|could|may|must|not|allowed|permitted|only|always|never|enabled|disabled|"
                              r"by\s+default|automatically)\b", re.I)
_DEFINITION_EVIDENCE = re.compile(r"\b(?:is|are|means|refers\s+to|lets|allows|enables|displays|shows|contains|provides|helps)\b", re.I)
_SELECTION_ACTION = re.compile(r"\b(?:select|click|choose|use|press|tap|open)\b", re.I)
_SELECTION_TARGET = re.compile(r"\b(?:option|button|icon|link|menu|tab|check\s?box|field|panel)\b|[“\"][^”\"]{2,60}[”\"]", re.I)
_LOCATION_EVIDENCE = re.compile(
    r"\b(?:on|in|under|from|at|inside|within|via|through)\s+(?:the\s+)?(?:[\w&’'-]+\s+){0,4}(?:bar|menu|page|tab|panel|pane|corner|section|"
    r"announcement|list|header|icon|screen|view|area|toolbar|home\s+screen)\b|https?://\S+|\bwww\.\S+|"
    r"\b(?:under|in|on|from)\s+(?:the\s+)?[“\"][^”\"]{2,60}[”\"]", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_TRAILING_LABEL = re.compile(r"(?:^|[\s:])\d{1,2}[.)]\s*$")
#: Post-freeze PF-5B (D-114): a statement that opens as the consequence or continuation of an earlier step ("You will be
#: redirected to ...", "Then ...", "After that ...", also under a printed heading): one step of a sequence, not a self-contained
#: answer.
_DEPENDENT = re.compile(r"^\s*(?:[^:.!?]{1,120}:\s+)?(?:\d{1,2}[.)]\s+|[-–•]\s+)?(?:you\s+will\s+(?:then\s+)?be\s+(?:redirected|taken|sent|moved|returned|brought)|"
                        r"then\b|next\b|after\s+that\b|afterwards\b)", re.I)
_PHONE = re.compile(r"\+?\d[\d\s()-]{6,}\d")
_URL = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
_INLINE_LIST = re.compile(r"(?:\binclude[sd]?\b|\bincluding\b|\bare\b|:)\s+[^.;]*,[^.;]*,[^.;]*", re.I)


def statement_problem(binding: Binding, quote: str) -> Optional[str]:
    """Why ``quote`` (one statement) does not answer the clause of ``binding``, or None."""
    intent = binding.intent
    if intent is GuideIntent.PROCEDURE:
        return "a procedure ask is answered by a complete procedure or one item of an unordered list, never by a single statement"
    missing = missing_terms(binding, quote)
    if missing:
        return "the evidence does not state %d asked term(s)" % len(missing)
    if missing_entities(binding, quote):
        return "a named entity of the clause is not in the evidence"
    if _TRAILING_LABEL.search(quote or ""):                        # post-freeze PF-5B (D-114)
        return "the unit ends with the label of the next list item, so it is cut"
    if dangling_lead_in(quote):                                    # post-freeze PF-5B (D-114)
        return "the statement announces content it does not contain"
    if _DEPENDENT.match(quote or ""):                              # post-freeze PF-5B (D-114)
        return "the statement is the consequence of a step it does not include"
    for word in binding.polarity:                                  # post-freeze PF-5B (D-114)
        if re.search(r"\b%s\b" % _POLARITY[word], quote, re.I) and not re.search(r"\b%s\b" % word, quote, re.I):
            return "the evidence states the opposite condition (%s, not %s)" % (_POLARITY[word], word)
    if intent is GuideIntent.VALUE:
        if not _QUANTITY.search(_LEADING_LABEL.sub("", quote, count=1)):
            return "a value ask needs a stated number with its unit"
        if binding.minimum and not _MIN_EVIDENCE.search(quote):
            return "a minimum is asked; the evidence states no lower bound"
        if binding.maximum and not _MAX_EVIDENCE.search(quote):
            return "a maximum is asked; the evidence states no upper bound"
    elif intent is GuideIntent.YES_NO:
        if not _YES_NO_EVIDENCE.search(quote):
            return "a yes/no ask needs a stated permission, prohibition or behaviour"
    elif intent is GuideIntent.DEFINITION:
        if not _DEFINITION_EVIDENCE.search(quote):
            return "a definition ask needs a defining statement"
    elif intent is GuideIntent.SELECTION:
        if not (_SELECTION_ACTION.search(quote) and _SELECTION_TARGET.search(quote)):
            return "a selection ask needs the control to select"
    elif intent is GuideIntent.LOCATION:
        if not _LOCATION_EVIDENCE.search(quote):
            return "a location ask needs a stated place in the interface or an address"
    elif intent is GuideIntent.CONTACT:
        fieldname = str((binding.info or {}).get("field") or "")
        if "mail" in fieldname:
            ok = bool(_EMAIL.search(quote))
        elif "web" in fieldname:
            ok = bool(_URL.search(quote))
        elif fieldname:
            ok = bool(_PHONE.search(quote))
        else:
            ok = bool(_EMAIL.search(quote) or _PHONE.search(quote))
        if not ok:
            return "a contact ask needs the asked channel"
    elif intent is GuideIntent.ENUMERATION:
        if not _INLINE_LIST.search(quote):
            return "an enumeration ask is answered by a complete list; one statement must itself list the set"
    return None


# ---------------------------------------------------------------------------
# Statement units: exact sub-quotes of one chunk
# ---------------------------------------------------------------------------
_GLYPH = "•▪■●‣⁃"
#: A sentence ends at ". ", "! " or "? ", never after a list label ("4. Next step") and never before one (that is an item start).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?<!^\d\.)(?<!\s\d\.)(?<!^\d\d\.)(?<!\s\d\d\.)\s+(?=\S)(?!\d{1,2}[.)]\s)")
#: A list item opens: a glyph bullet, an "o " item, or a numbered or lettered item label inside running text.
_ITEM_BOUNDARY = re.compile(r"\s*[" + _GLYPH + r"]\s*|\s+(?=o\s+[A-Z])|\s+(?=\d{1,2}[.)]\s+[A-Z])|\s+(?=[A-Za-z]\)\s+[A-Z])")
_ACTION_VERB = re.compile(r"\b(?:select|click|open|choose|add|create|upload|download|enter|type|press|tab|drag|search|post|log\s?in|"
                          r"sign\s+in|navigate|submit|transfer|approve|assign|set|edit|delete|copy|import|export|use|go\s+to)\b", re.I)


@dataclass(frozen=True)
class Unit:
    """One candidate quote: an exact substring of chunk ``chunk_id``."""

    chunk_id: int
    source_id: str
    page: int
    quote: str
    kind: str                      # "step" | "sentence" | "sentences"
    rank: int                      # retrieval rank of the chunk (0 is best)
    intro: bool = False            # a generic introduction (deterministic features)


def _spans(text: str) -> List[Tuple[int, int, str]]:
    """``(start, end, boundary-before)`` of every unit of a prose chunk; the boundary is "item" after a list glyph."""
    cuts: List[Tuple[int, int, str]] = []
    for m in _SENTENCE_BOUNDARY.finditer(text):
        cuts.append((m.start(), m.end(), "sentence"))
    for m in _ITEM_BOUNDARY.finditer(text):
        if m.end() > m.start():
            cuts.append((m.start(), m.end(), "item"))
    cuts.sort()
    merged: List[Tuple[int, int, str]] = []
    for c in cuts:
        if merged and c[0] < merged[-1][1]:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], c[1]), "item" if "item" in (prev[2], c[2]) else "sentence")
        else:
            merged.append(c)
    out: List[Tuple[int, int, str]] = []
    start, before = 0, "start"
    for cut_start, cut_end, kind in merged:
        if cut_start > start:
            out.append((start, cut_start, before))
        start, before = cut_end, kind
    if start < len(text):
        out.append((start, len(text), before))
    return [(a, b, k) for a, b, k in out if text[a:b].strip(_GLYPH + " ")]


def is_generic_introduction(text: str, meta: Mapping[str, Any], binding: Optional[Binding]) -> bool:
    """A broad system description (D-111 evidence selection): prose, no action verb, no asked term beyond the bound
    system names, and it names the documented system. Never applied to a definition ask."""
    if meta.get("chunk_type") in ("procedure_step", "procedure_window"):
        return False
    if _ACTION_VERB.search(text or ""):
        return False
    names = SYSTEM_NAMES.get(str(meta.get("source_id")), ())
    if not any(_name_pattern(n).search(text or "") for n in names):
        return False
    if binding is not None and binding.terms and stated_terms(binding, binding.terms, text):
        return False
    return True


def statement_units(chunk_id: int, chunk: str, meta: Mapping[str, Any], rank: int, binding: Optional[Binding] = None) -> List[Unit]:
    """The candidate quotes of one chunk: a procedure step (with and without its heading) or the sentences and adjacent
    sentence pairs of a prose chunk. Row and window chunks are printed fragments or duplicates and give no unit."""
    source_id, page = str(meta.get("source_id")), meta.get("page")
    if not isinstance(page, int) or isinstance(page, bool):
        return []
    kind = meta.get("chunk_type")
    units: List[Unit] = []
    if kind == "procedure_step":
        heading = meta.get("procedure_heading")
        prefix = (heading + ": ") if isinstance(heading, str) and heading and chunk.startswith(heading + ": ") else ""
        units.append(Unit(chunk_id, source_id, page, chunk[len(prefix):].strip(), "step", rank))
        if prefix and not prefix[:1].islower():                    # D-114: a heading opening in lower case is a cut row
            units.append(Unit(chunk_id, source_id, page, chunk.strip(), "step", rank))
        return [u for u in units if u.quote]
    if kind != "paragraph":
        return []
    spans = _spans(chunk)
    singles: List[Unit] = []
    pairs: List[Unit] = []
    for i, (a, b, before) in enumerate(spans):
        quote = chunk[a:b].strip(_GLYPH + " ")
        opening = _LEADING_LABEL.sub("", quote, count=1)[:1]
        preceding = chunk[:a].rstrip(_GLYPH + " ")
        if opening.islower() and not preceding.endswith((".", "!", "?", ";", ":")):
            continue                                               # D-114: the unit opens in the middle of a sentence
        if quote:
            singles.append(Unit(chunk_id, source_id, page, quote, "sentence", rank, is_generic_introduction(quote, meta, binding)))
        if i + 1 < len(spans) and spans[i + 1][2] == "sentence":
            pair = chunk[a:spans[i + 1][1]].strip(_GLYPH + " ")
            if pair:
                pairs.append(Unit(chunk_id, source_id, page, pair, "sentences", rank, is_generic_introduction(pair, meta, binding)))
    return singles + pairs                                         # one sentence is always preferred to two


# ---------------------------------------------------------------------------
# Procedures, with heading provenance and page continuations
# ---------------------------------------------------------------------------
PROVENANCE_ROW = "heading_row"
PROVENANCE_LEAD_IN = "lead_in"
PROVENANCE_HINT = "page_hint"
PROVENANCE_CARRIED = "carried"
RELIABLE_HEADINGS = frozenset({PROVENANCE_ROW, PROVENANCE_LEAD_IN})
_NUMBERED_LABEL = re.compile(r"^(\d{1,2})\.$")
_LETTER_LABEL = re.compile(r"^([A-Za-z])\)$")
_HOWTO_HEADING = re.compile(r"^\s*(?:how\s+(?:to|do|can)\b|to\s+\w+.*\bfollow\b|.*\bsteps?\b|.*\bprocedure\b)", re.I)


@dataclass(frozen=True)
class Step:
    chunk_id: int
    page: int
    label: str
    quote: str                      # the step as printed in its chunk ("1. Open ..."), without the heading


@dataclass(frozen=True)
class Procedure:
    source_id: str
    key: str                        # "<source>:<page>:<procedure id>[+<page>:<procedure id>]"
    pages: Tuple[int, ...]
    heading: Optional[str]
    provenance: str                 # how the heading was found (heading row, lead-in, page hint, carried)
    steps: Tuple[Step, ...]
    problems: Tuple[str, ...] = ()  # structural problems found while reading (missing steps, a possible continuation)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(s.label for s in self.steps)

    @property
    def sequential(self) -> bool:
        """Ordered steps: numbered or lettered labels, or a "how to / steps" heading over a bulleted list, or (post-freeze
        PF-4, D-113) a bulleted list under an action heading with at least two instruction steps, which D-112 reads as an
        ordered procedure ("Checking Session Attendance": open the tool, select overview, export): one of its items never
        answers alone."""
        if any(label != "-" for label in self.labels):
            return True
        if self.heading and _HOWTO_HEADING.match(self.heading):
            return True
        from . import guide_structure as _structure
        return bool(self.heading and is_action_heading(self.heading)
                    and sum(1 for step in self.steps if _structure.is_instruction(step.quote)) >= 2)

    def text(self) -> str:
        return " ".join([self.heading or ""] + [s.quote for s in self.steps])

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "source_id": self.source_id, "pages": list(self.pages), "heading": self.heading, "provenance": self.provenance,
                "labels": list(self.labels), "chunk_ids": [s.chunk_id for s in self.steps], "problems": list(self.problems)}


def _read_page(rows: Sequence[str], section_hint: str) -> List[Dict[str, Any]]:
    """The procedures of one page exactly as ``chunking.read_procedures`` groups them, with the provenance of each
    heading. Mirrors that function line for line; ``GuideIndex`` checks the result against the indexed chunks."""
    procedures: List[Dict[str, Any]] = []
    heading = _clean_heading(section_hint) if section_hint else ""
    provenance = PROVENANCE_HINT
    steps: List[Dict[str, Any]] = []
    lead_in: Optional[str] = None
    last_continuation: Optional[str] = None
    started_with: Optional[str] = None

    continuation: Optional[str] = None

    def close() -> None:
        nonlocal steps, started_with, continuation
        kept = [s for s in steps if len(s["text"].split()) >= 2]
        if kept:
            procedures.append({"heading": heading, "provenance": started_with or provenance, "steps": kept, "continuation": continuation})
        steps = []
        started_with = None
        continuation = None

    for row in rows:
        step = _match_step(row)
        if step is not None:
            label, text = step
            if label == "1." and steps:
                if last_continuation is not None and _is_list_lead_in(last_continuation):
                    cut = normalize_text(last_continuation)
                    if steps[-1]["text"].endswith(cut):
                        steps[-1]["text"] = steps[-1]["text"][: -len(cut)].rstrip()
                    lead_in = last_continuation.strip()
                close()
                provenance = PROVENANCE_CARRIED
            if not steps and lead_in is not None:
                heading = _clean_heading(lead_in)
                provenance = PROVENANCE_LEAD_IN
            if not steps:
                started_with = provenance
            lead_in = None
            last_continuation = None
            steps.append({"label": label, "text": normalize_text(text)})
            continue
        if _is_procedure_heading(row):
            # a row that opens in lower case right after a step continues that step's sentence ("... after the" /
            # "approval from Chair and Dean as below:"); the chunker read it as a heading, so the step is cut there
            fragment = bool(steps) and row.strip()[:1].islower()
            if fragment:
                continuation = row.strip()
            close()
            heading = _clean_heading(row)
            provenance = PROVENANCE_CARRIED if fragment else PROVENANCE_ROW
            lead_in = None
            last_continuation = None
            continue
        if steps and len(row.strip()) >= 3:
            steps[-1]["text"] = normalize_text((steps[-1]["text"] + " " + row.strip()).strip())
            last_continuation = row
        elif not steps:
            lead_in = row.strip() if _is_list_lead_in(row) else None
    close()
    return procedures


def label_problem(labels: Sequence[str]) -> Optional[str]:
    """Why a label sequence is not one ordered list, or None. Accepted: bullets only; numbers 1..n with lettered
    sub-items a), b), ... directly under a number; letters A..n or a..n only."""
    if not labels:
        return "no steps"
    if all(label == "-" for label in labels):
        return None
    if all(_LETTER_LABEL.match(label) for label in labels):
        letters = [label[0] for label in labels]
        start = "A" if letters[0].isupper() else "a"
        expected = [chr(ord(start) + i) for i in range(len(letters))]
        return None if letters == expected else "lettered steps are not A, B, C ... from the first letter"
    if not _NUMBERED_LABEL.match(labels[0]):
        return "the list mixes bullets, letters and numbers"
    number = 0
    letter: Optional[str] = None
    for label in labels:
        m = _NUMBERED_LABEL.match(label)
        if m:
            if int(m.group(1)) != number + 1:
                return "the numbering is broken (%s after %d)" % (label, number)
            number += 1
            letter = "a"
            continue
        m = _LETTER_LABEL.match(label)
        if m and letter is not None and m.group(1).lower() == letter:
            letter = chr(ord(letter) + 1)
            continue
        return "the list mixes bullets, letters and numbers"
    if number < 2:
        return "a numbered list with one step is a list cut after its first item"
    return None


def _top_number(labels: Sequence[str]) -> Optional[int]:
    numbers = [int(m.group(1)) for m in (_NUMBERED_LABEL.match(label) for label in labels) if m]
    return numbers[-1] if numbers else None


def _first_number(labels: Sequence[str]) -> Optional[int]:
    m = _NUMBERED_LABEL.match(labels[0]) if labels else None
    return int(m.group(1)) if m else None


class GuideIndex:
    """The procedures of every guide page and the registry records, built once per knowledge base. Deterministic;
    built from the cached page rows and the chunk metadata; no model."""

    def __init__(self, chunks: Sequence[str], metadata: Sequence[Mapping[str, Any]], pages_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
                 records: Iterable[Any], policy: Optional[Mapping[str, Iterable[str]]] = None) -> None:
        self.chunks = chunks
        self.metadata = metadata
        self.records: Dict[str, Any] = {r.source_id: r for r in records}
        if policy is None:
            from .source_scopes import source_policy
            policy = source_policy(list(self.records.values())) if self.records else {}
        #: the D-110 approved source set of every specialist (the adapter re-checks a guide answer against it)
        self.policy: Dict[str, FrozenSet[str]] = {sid: frozenset(ids) for sid, ids in policy.items()}
        self.procedures: Dict[str, List[Procedure]] = {}
        self.by_chunk: Dict[int, Procedure] = {}
        self.skipped_pages: List[Tuple[str, int, str]] = []
        step_chunks: Dict[Tuple[str, int, int, int], int] = {}
        for i, meta in enumerate(metadata):
            if meta.get("chunk_type") == "procedure_step":
                step_chunks[(str(meta.get("source_id")), meta.get("page"), meta.get("procedure_id"), meta.get("step_index"))] = i
        self.step_chunks = step_chunks
        for source_id, pages in pages_by_source.items():
            record = self.records.get(source_id)
            if record is None or record.document_type not in GUIDE_DOCUMENT_TYPES:
                continue
            per_page: List[Tuple[int, List[Procedure]]] = []
            for page in sorted(pages, key=lambda p: p.get("page", 0)):
                number = page.get("page")
                rows = page.get("rows") or page.get("lines") or []
                read = _read_page(rows, page.get("section_hint", "") or "")
                built: List[Procedure] = []
                ok = True
                for pid, proc in enumerate(read):
                    steps: List[Step] = []
                    raw_heading = proc["heading"] or ""
                    heading = normalize_text(raw_heading) or None
                    prefix = (heading + ": ") if heading else ""
                    for si, s in enumerate(proc["steps"]):
                        cid = step_chunks.get((source_id, number, pid, si))
                        printed = "%s %s" % (s["label"], s["text"])
                        expected = normalize_text(((raw_heading + ": ") if raw_heading else "") + printed)
                        if cid is None or chunks[cid] != expected or not expected.startswith(prefix):
                            ok = False
                            break
                        steps.append(Step(cid, number, s["label"], chunks[cid][len(prefix):]))
                    if not ok:
                        break
                    problems: Tuple[str, ...] = ()
                    if proc.get("continuation") and steps:
                        repaired = self._complete_step(source_id, number, steps[-1], proc["continuation"])
                        if repaired is None:
                            problems = ("the last step continues on a row read as a heading and no indexed text holds the whole step",)
                        else:
                            steps[-1] = repaired
                    built.append(Procedure(source_id, "%s:%d:%d" % (source_id, number, pid), (number,), heading, proc["provenance"], tuple(steps),
                                           problems))
                indexed = sum(1 for k in step_chunks if k[0] == source_id and k[1] == number)
                if not ok or indexed != sum(len(p.steps) for p in built):
                    self.skipped_pages.append((source_id, number, "procedures differ from the indexed procedure chunks"))
                    built = []
                per_page.append((number, built))
            self.procedures[source_id] = self._link(source_id, per_page)
            for proc in self.procedures[source_id]:
                for step in proc.steps:
                    self.by_chunk[step.chunk_id] = proc
        # ---- post-freeze PF-3D (D-112): sections and lists recovered from the printed formatting -----------------------
        from . import guide_structure as _structure
        guide_ids = [sid for sid in pages_by_source if sid in self.records and self.records[sid].document_type in GUIDE_DOCUMENT_TYPES]
        page_chunks = _structure.PageChunks(chunks, metadata, guide_ids)
        self.structures: Dict[str, List[Any]] = {}
        self.structure_by_key: Dict[str, Any] = {}
        self.structures_by_chunk: Dict[int, List[str]] = {}
        for source_id in guide_ids:
            found = _structure.recover(source_id, pages_by_source[source_id], page_chunks, is_action_heading)
            self.structures[source_id] = found
            for structure in found:
                self.structure_by_key[structure.key] = structure
                for part in structure.parts:
                    keys = self.structures_by_chunk.setdefault(part.chunk_id, [])
                    if structure.key not in keys:
                        keys.append(structure.key)

    def structures_of(self, source_id: str) -> List[Any]:
        return list(self.structures.get(source_id, ()))

    def vocabulary(self, source_id: str) -> FrozenSet[str]:
        """Post-freeze PF-5B (D-114): the stems of every indexed chunk of one guide (with written-together compounds), built
        once per source. The binding reads it to decide whether the guide draws a contrast ("final" versus "midterm")."""
        cache = self.__dict__.setdefault("_vocabulary", {})
        if source_id not in cache:
            words: set = set()
            for i, meta in enumerate(self.metadata):
                if meta.get("source_id") == source_id:
                    words |= stems(self.chunks[i])
            cache[source_id] = frozenset(words)
        return cache[source_id]

    def bind(self, clause: str, source_id: str) -> Tuple[Optional[Binding], Optional[str]]:
        """``bind`` against the registry record of ``source_id`` with that guide's vocabulary (D-114)."""
        record = self.record(source_id)
        return bind(clause, record, self.vocabulary(source_id) if record is not None else None)

    def structure(self, key: str) -> Any:
        return self.structure_by_key.get(key)

    def _complete_step(self, source_id: str, page: int, step: Step, continuation: str) -> Optional[Step]:
        """The cut last step completed with its continuation row, quoted from the page's prose chunk that holds the whole
        step verbatim (the chunker's paragraph text), or None when no indexed chunk holds it."""
        whole = normalize_text("%s %s" % (step.quote, continuation))
        for i, meta in enumerate(self.metadata):
            if meta.get("source_id") == source_id and meta.get("page") == page and meta.get("chunk_type") == "paragraph" and whole in self.chunks[i]:
                return Step(i, page, step.label, whole)
        return None

    @staticmethod
    def _link(source_id: str, per_page: List[Tuple[int, List[Procedure]]]) -> List[Procedure]:
        """Merge a numbered list continued at the top of the next page; flag a list that may continue or may have
        started on an earlier page."""
        out: List[Procedure] = []
        consumed: set = set()
        for i, (number, procs) in enumerate(per_page):
            nxt = per_page[i + 1][1] if i + 1 < len(per_page) and per_page[i + 1][0] == number + 1 else []
            prev = per_page[i - 1][1] if i > 0 and per_page[i - 1][0] == number - 1 else []
            for j, proc in enumerate(procs):
                if proc.key in consumed:
                    continue                                       # merged into the previous page's list
                problems = list(proc.problems)
                if j == len(procs) - 1 and nxt and nxt[0].provenance == PROVENANCE_HINT:
                    head = nxt[0]
                    last, first = _top_number(proc.labels), _first_number(head.labels)
                    if last is not None and first == last + 1 and not head.problems:
                        consumed.add(head.key)
                        out.append(Procedure(source_id, "%s+%d:0" % (proc.key, number + 1), proc.pages + (number + 1,), proc.heading,
                                             proc.provenance, proc.steps + head.steps, tuple(problems)))
                        continue
                    problems.append("the list may continue on page %d (steps open that page under no heading)" % (number + 1))
                if j == 0 and proc.provenance == PROVENANCE_HINT and prev:
                    problems.append("the list opens the page under no heading and may continue a list of page %d" % (number - 1))
                out.append(Procedure(source_id, proc.key, proc.pages, proc.heading, proc.provenance, proc.steps, tuple(problems)))
        return out

    # ---- lookups -----------------------------------------------------------------------------------------------
    def record(self, source_id: str) -> Any:
        return self.records.get(source_id)

    def procedures_of(self, source_id: str) -> List[Procedure]:
        return list(self.procedures.get(source_id, ()))

    def procedure_of_chunk(self, chunk_id: int) -> Optional[Procedure]:
        """The procedure a procedure_step or procedure_window chunk belongs to, or None."""
        if chunk_id in self.by_chunk:
            return self.by_chunk[chunk_id]
        meta = self.metadata[chunk_id] if isinstance(chunk_id, int) and 0 <= chunk_id < len(self.metadata) else {}
        if meta.get("chunk_type") == "procedure_window":
            first = self.step_chunks.get((str(meta.get("source_id")), meta.get("page"), meta.get("procedure_id"), meta.get("step_start")))
            if first is not None:
                return self.by_chunk.get(first)
        return None

    def procedure(self, key: str) -> Optional[Procedure]:
        for procs in self.procedures.values():
            for proc in procs:
                if proc.key == key:
                    return proc
        return None

    def conflict_problem(self, chunk_id: int, quote: str) -> Optional[str]:
        """A registered source conflict in the quoted text (the registry's phrases), or a unit of a conflict-stamped chunk
        (PF-2 stamp). Post-freeze PF-5B (D-114), claim-level isolation for a VALUE conflict (a phrase of the conflict states a
        number: "maximum 30"): a stamped chunk's unit is blocked when it states a phrase of the conflict, a count or limit
        word (``CLAIM_WORDS``), or a number with one of the conflict topic's words; any other unit of that chunk (an upload
        step on the same page as the disputed quota) is not the disputed claim. Any other conflict (a recommendation whose
        note covers the text around it) and a stamp not registered for the chunk's own source keep blocking the whole
        chunk."""
        meta = self.metadata[chunk_id] if 0 <= chunk_id < len(self.metadata) else {}
        record = self.records.get(str(meta.get("source_id")))
        conflicts = tuple(getattr(record, "conflicts", ()) or ())
        for conflict in conflicts:
            if conflict.matches(quote):
                return "source conflict %r in the quoted text" % conflict.topic
        topic = meta.get("source_conflict")
        if isinstance(topic, str) and topic:
            own = next((c for c in conflicts if c.topic == topic), None)
            if own is None or not value_conflict(own):
                return "source conflict %r on the evidence chunk" % topic
            if disputed_claim(own, quote):
                return "source conflict %r: the unit states the disputed claim" % topic
        return None


#: Post-freeze PF-5B (D-114): words that make a unit of a conflict-stamped chunk a claim about a count or a limit.
CLAIM_WORDS = re.compile(r"\b(?:quotas?|count(?:s|ed|ing)?|limit(?:s|ed|ation)?|allowances?|yearly|annual(?:ly)?|per\s+(?:year|month|semester|term))\b",
                         re.I)
_NUMBER = re.compile(r"(?<![\w.])\d[\d,.]*")


def value_conflict(conflict: Any) -> bool:
    """True when a registered conflict disputes a value (one of its phrases states a number); only such a conflict is
    isolated at claim level (D-114)."""
    return any(re.search(r"\d", str(p)) for p in (getattr(conflict, "phrases", ()) or ()))


def disputed_claim(conflict: Any, quote: str) -> bool:
    """True when ``quote`` states the claim a registered value conflict disputes (D-114): one of its phrases, a count or limit
    word, or a number (a list label is not one) together with a content word of one of its registered phrases."""
    if conflict.matches(quote) or CLAIM_WORDS.search(quote or ""):
        return True
    body = _LEADING_LABEL.sub("", quote or "", count=1)
    if not _NUMBER.search(body):
        return False
    phrase_words = {stem(w) for p in (getattr(conflict, "phrases", ()) or ()) for w in _TOKEN.findall(str(p))
                    if not w[:1].isdigit() and w.lower() not in _STOP}
    return bool(phrase_words & stems(body))


#: Action verbs (stems) a procedure heading or a "how do I" clause may open with.
ACTION_VERBS = frozenset(stem(w) for w in """
access add adding approve assign attach change check checking choose click copy create delete download edit enable enter export filter
find import log login manage mark move navigate open pin post publish release remove reset review search select send set sign submit
transfer update upload verify view grade hide show keep schedule allow
""".split())
#: The only verbs treated as the same action (D-111; each pair is documented by the guide itself: the grade transfer guide
#: calls "Send Grades to Banner" the process of "transferring grades"; "access" is the guides' word for logging in).
SAME_ACTION: Tuple[FrozenSet[str], ...] = (frozenset({stem("transfer"), stem("send")}),
                                           frozenset({stem("access"), stem("log"), stem("login"), stem("sign")}))
_ROLE_STEMS = frozenset(stem(w) for w in "dean chair chairperson head student students instructor faculty member staff ta".split())
_HEADING_LEAD = re.compile(r"^\s*(?:(?P<howto>how\s+(?:to|do\s+i|can\s+i)\s+|to\s+))?(?P<word>[A-Za-z]+)", re.I)
#: The imperative or gerund words that make a heading without "How to" name an action ("Checking Similarity Report",
#: "Download and upload grades"). A noun heading such as "Grade Columns" or "Course Image" names no action.
_HEADING_ACTION_WORDS = frozenset("""
access accessing add adding approve approving assign assigning change changing check checking create creating delete deleting download
downloading edit editing export exporting filter filtering import importing manage managing move moving open opening post posting
publish publishing remove removing reset resetting review reviewing search searching select selecting send sending submit submitting
transfer transferring upload uploading verify verifying
""".split())
_CAPABILITY_HEADING = re.compile(r"\byou\s+can\s*:?\s*$", re.I)
_PROPER_HEADING = re.compile(r"^[^A-Za-z0-9]*[A-Z0-9]")
_FIGURE_LEAD = re.compile(r"\b(?:below|figure|screenshot|image|picture)\s*:\s*$", re.I)


def dangling_lead_in(text: Optional[str]) -> Optional[str]:
    """Post-freeze PF-3D (D-112): a last step that ends with ":" announces content the answer does not contain (a missing
    last step), unless it points at a figure the text extraction cannot carry ("... after the approval as below:")."""
    text = (text or "").rstrip()
    if text.endswith(":") and not _FIGURE_LEAD.search(text):
        return "the last step announces content that does not follow it, so a step is missing"
    return None


#: Post-freeze PF-5B (D-114): words a printed unit ends with only when the extraction cut its sentence ("... and how much",
#: "... before they are"). Verb particles ("log in", "sign out", "set up") are not among them.
_DANGLING = frozenset("""
a an the and or but nor of to for with by at from as than that which who whose whom how much many when where if whether
is are was were be been being will can could should would may might must not their your its our his her this these those
such very more less into onto about because while unless
""".split())
_TERMINAL_END = (".", "!", "?", ";", ":", ")", "]", "\"", "”", "'", "’")
_PRINTED_LABEL = re.compile(r"^\s*(?:\d{1,2}|[A-Za-z])\s*[.)-]\s+")


def cut_unit_problem(texts: Sequence[str]) -> Optional[str]:
    """Post-freeze PF-5B (D-114): why a sequence of printed units (steps, items, prose) contains a unit the extraction cut,
    or None: a unit that ends on a function word with no closing punctuation ("... and how much"), or a unit opening in
    lower case after a unit with no closing punctuation (the rest of the previous unit read as a new item). A printed number
    or letter label proves a new item ("3. view the gradebook categories"), so a labelled unit is never read as a
    continuation."""
    previous: Optional[str] = None
    for raw in texts:
        text = (raw or "").strip()
        if not text:
            continue
        body = _LEADING_LABEL.sub("", text, count=1)
        labelled = bool(_PRINTED_LABEL.match(text))
        if previous is not None and not labelled and body[:1].islower() and not previous.endswith(_TERMINAL_END):
            return "a unit opens in the middle of the previous unit's sentence, so the text is cut"
        words = re.findall(r"[A-Za-z’']+", text)
        if not text.endswith(_TERMINAL_END) and words and words[-1].lower() in _DANGLING:
            return "a unit ends in the middle of a sentence, so the text is cut"
        previous = text
    return None


def heading_verb(heading: Optional[str]) -> Optional[str]:
    """The action a procedure heading names ("How to access ..." -> access, "Checking Similarity Report" -> check), or None
    for a noun heading ("Your Courses", "File Requirements")."""
    m = _HEADING_LEAD.match(heading or "")
    if not m:
        return None
    word = m.group("word")
    if m.group("howto"):
        return stem(word) if stem(word) in ACTION_VERBS else None
    return stem(word) if word.lower() in _HEADING_ACTION_WORDS else None


def clause_verb(binding: Binding) -> Optional[str]:
    """The asked action of a procedure clause: its first core term that is an action verb (a role such as "Dean" is not)."""
    for term in binding.core:
        if term in _ROLE_STEMS:
            continue
        return term if term in ACTION_VERBS else None
    return None


def same_action(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b) and (a == b or any(a in group and b in group for group in SAME_ACTION))


def is_action_heading(text: Optional[str]) -> bool:
    """A heading that names an action ("Send an Announcement", "How to access ...") (D-112 list classification)."""
    title = re.sub(r"^\s*(?:\d{1,2}|[A-Za-z])\s*[.)-]\s+", "", text or "")     # "1.", "a)", "D-" (post-freeze PF-4)
    return heading_verb(title) is not None or bool(_HOWTO_HEADING.match(title))


# ---------------------------------------------------------------------------
# Post-freeze PF-3D (D-112): may a recovered section or list answer the clause?
# ---------------------------------------------------------------------------
_LABEL_DIGIT = re.compile(r"^(\d{1,2})\s*[.)]$")
_LABEL_LETTER_ANY = re.compile(r"^([A-Za-z])\s*[.)]$")


def _normal_label(label: str) -> str:
    """"1)" and "1." -> "1."; "a." and "a)" -> "a)" (the shapes ``label_problem`` reads); a bullet -> "-"."""
    m = _LABEL_DIGIT.match(label or "")
    if m:
        return "%s." % m.group(1)
    m = _LABEL_LETTER_ANY.match(label or "")
    if m:
        return "%s)" % m.group(1)
    return "-"


def _labelled_runs(blocks: Sequence[Any]) -> List[List[Any]]:
    runs: List[List[Any]] = []
    current: List[Any] = []
    for block in blocks:
        if block.kind == "item" and block.marker in ("numbered", "lettered"):
            current.append(block)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _states_action(verb: str, objects: Sequence[str], text: str) -> bool:
    """The text states the asked action on its object ("You can set accommodations ..."): the verb's stem followed by an
    object stem within three words (D-112 heading-verb override)."""
    words = [stem(t) for t in _TOKEN.findall(text or "")]
    for i, word in enumerate(words):
        if word == verb and any(o in words[i + 1:i + 4] for o in objects):
            return True
    return False


def structure_problem(structure: Any, binding: Binding, index: GuideIndex) -> Optional[str]:
    """Why a D-112 recovered section or list is not a provably complete answer to the procedure or list ask of ``binding``,
    or None. Every D-111 rule still applies (one approved source, conflicts, every term and entity stated, a heading that
    names the asked action or object); D-112 adds the completeness rules of recovered structure."""
    if structure is None:
        return "the claimed structure is not in the structure index"
    if structure.source_id != binding.source_id:
        return "the structure belongs to another source"
    if structure.problems:
        return structure.problems[0]
    if not structure.parts:
        return "the structure has no exact quotes"
    problem = dangling_lead_in(structure.parts[-1].quote)
    if problem:
        return problem
    problem = cut_unit_problem([b.text for b in structure.blocks if b.kind in ("item", "prose")])    # D-114
    if problem:
        return problem
    for part in structure.parts:
        conflict = index.conflict_problem(part.chunk_id, part.quote)
        if conflict:
            return conflict
    text = structure.text()
    if missing_terms(binding, text):
        return "the structure does not state %d asked term(s)" % len(missing_terms(binding, text))
    if missing_entities(binding, text):
        return "a named entity of the clause is not in the structure"
    context = " ".join(filter(None, [structure.heading, structure.section_title]))
    if not _PROPER_HEADING.match(structure.heading or ""):
        return "the structure's heading or lead-in is a sentence fragment"
    intent = binding.intent
    asked = set(binding.core) if intent is GuideIntent.PROCEDURE and binding.core else set(binding.terms)
    # the declared system names are bound by the registry: they never also count as the heading naming the asked action
    # ("How to use the Grade Transfer Dashboard" does not name "transfer grades")
    named_context = context
    for name in sorted(SYSTEM_NAMES.get(structure.source_id, ()), key=len, reverse=True):
        named_context = _name_pattern(name).sub(" ", named_context)
    if not stated_terms(binding, asked, named_context):
        return "the structure's heading names neither the asked action nor its object"
    head = (binding.info or {}).get("object_head")
    if intent is GuideIntent.ENUMERATION and isinstance(head, str) and head and stem(head) not in stems(named_context):
        return "the list's heading does not name the asked set"
    for run in _labelled_runs(structure.blocks):
        problem = label_problem([_normal_label(b.label) for b in run])
        if problem:
            return problem
    if structure.list_type == MIXED_PROCEDURE:                      # post-freeze PF-5B (D-114)
        problem = mixed_problem(structure)
        if problem:
            return problem
    if intent is GuideIntent.PROCEDURE:
        if any(_CAPABILITY_HEADING.search(t or "") for t in (structure.heading, structure.section_title)):
            return "a list of what can be done is not a procedure"
        # post-freeze PF-4 (D-113): a numbered or lettered sub-section titled after its object ("1. Session Details" under
        # "A. Create New Scheduled Sessions") is one part of a larger workflow, never the whole procedure of the asked action
        head = structure.context[0] if structure.kind == "section" and structure.context else None
        if head is not None and head.kind == "heading" and head.label:
            asked_verb, named_verb = clause_verb(binding), heading_verb(structure.heading)
            if named_verb is None or (asked_verb and not same_action(asked_verb, named_verb)):
                return "a labelled sub-section that does not name the asked action is one part of a larger procedure"
        verb, named = clause_verb(binding), heading_verb(re.sub(r"^\s*(?:\d{1,2}|[A-Za-z])\s*[.)]\s+", "", structure.section_title or ""))
        if verb and named and not same_action(verb, named):
            objects = [t for t in binding.core if t != verb and t not in _ROLE_STEMS]
            if not _states_action(verb, objects, " ".join(b.text for b in structure.blocks)):
                return "the structure's heading names another action than the one asked"
        if structure.kind == "list":
            if structure.list_type != "ordered_procedure":
                return "a %s list is not an ordered procedure" % structure.list_type.replace("_", " ")
            if any(b.marker == "bullet" for b in structure.blocks):
                outside = [b for b in structure.section_blocks if b not in structure.blocks and b.kind in ("item", "prose") and b.instruction]
                if outside:
                    return "the section gives instructions outside this bulleted list, so the list alone is not the procedure"
        elif not any(b.instruction for b in structure.blocks if b.kind in ("item", "prose")):
            return "a section without an instruction is not a procedure"
        return None
    if intent is GuideIntent.ENUMERATION:
        if structure.kind != "list":
            return "an enumeration is answered by a complete list"
        return None
    return "a %s ask is not answered by a recovered structure" % intent.value


#: Post-freeze PF-5B (D-114): the list type of a lettered section whose parts are numbered sub-sections (``guide_structure``).
MIXED_PROCEDURE = "mixed_procedure"


def mixed_problem(structure: Any) -> Optional[str]:
    """Why a recovered lettered section with numbered sub-sections ("A. Create New Scheduled Sessions:" over "1. Session
    Details:", "2. Session Settings:" ...) is not one complete procedure, or None (D-114). The sub-sections are numbered
    1..n in order (n >= 2), each has at least one unit, and the section ends at the next letter or at the end of the
    document; the cut-unit, conflict, term and heading rules apply as to any structure."""
    subs = [b for b in structure.blocks if b.kind == "heading"]
    if len(subs) < 2:
        return "a lettered section with fewer than two numbered sub-sections is not a mixed procedure"
    if any(b.marker != "numbered" for b in subs):
        return "a lettered section's sub-sections are not all numbered"
    problem = label_problem([_normal_label(b.label) for b in subs])
    if problem:
        return problem
    blocks = list(structure.blocks)
    for i, block in enumerate(blocks):
        if block.kind == "heading" and (i + 1 == len(blocks) or blocks[i + 1].kind == "heading"):
            return "a numbered sub-section of the lettered section has no content"
    if not structure.end.startswith(("next letter", "end of document")):
        return "the lettered section's end is not proven (%s)" % structure.end
    return None


def procedure_problem(proc: Procedure, binding: Binding, index: GuideIndex) -> Optional[str]:
    """Why ``proc`` is not a provably complete answer to the procedure (or list) ask of ``binding``, or None."""
    if proc.source_id != binding.source_id:
        return "the procedure belongs to another source"
    if proc.problems:
        return proc.problems[0]
    problem = label_problem(proc.labels) or dangling_lead_in(proc.steps[-1].quote if proc.steps else None)
    if problem:
        return problem
    problem = cut_unit_problem([s.quote for s in proc.steps])        # post-freeze PF-5B (D-114)
    if problem:
        return problem
    if proc.provenance not in RELIABLE_HEADINGS or not proc.heading:
        return "the procedure heading is only carried from the page or an earlier list, so the procedure is not identified"
    if not _PROPER_HEADING.match(proc.heading):
        return "the procedure heading is a sentence fragment, so the procedure is not identified"
    if binding.intent is GuideIntent.PROCEDURE:
        if _CAPABILITY_HEADING.search(proc.heading):
            return "a list of what can be done is not a procedure"
        asked, named = clause_verb(binding), heading_verb(proc.heading)
        if asked and named and not same_action(asked, named):
            return "the procedure heading names another action than the one asked"
    for step in proc.steps:
        conflict = index.conflict_problem(step.chunk_id, step.quote)
        if conflict:
            return conflict
    text = proc.text()
    if missing_terms(binding, text):
        return "the procedure does not state %d asked term(s)" % len(missing_terms(binding, text))
    if missing_entities(binding, text):
        return "a named entity of the clause is not in the procedure"
    asked = set(binding.core) if binding.intent is GuideIntent.PROCEDURE and binding.core else set(binding.terms)
    if not stated_terms(binding, asked, proc.heading or ""):
        return "the procedure heading names neither the asked action nor its object"
    return None


# ---------------------------------------------------------------------------
# Selection and verification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Part:
    chunk_id: int
    source_id: str
    page: int
    quote: str


@dataclass(frozen=True)
class GuideAnswer:
    """The evidence chosen for one clause from one guide."""

    binding: Binding
    form: str
    parts: Tuple[Part, ...]
    procedure_key: Optional[str] = None
    heading: Optional[str] = None
    markers: Tuple[str, ...] = ()            # D-112: how each part is printed (heading, lead_in, bullet, numbered, prose)
    list_type: Optional[str] = None          # D-112: the recovered list type (or "section")

    @property
    def pages(self) -> Tuple[int, ...]:
        return tuple(dict.fromkeys(p.page for p in self.parts))

    def record(self, part_index: int) -> Dict[str, Any]:
        """The D-111 record a finding carries (ids, stems and labels only; no question text)."""
        record = {"version": VERSION, "phase": PHASE, "source_id": self.binding.source_id, "form": self.form, "intent": self.binding.intent.value,
                  "procedure_key": self.procedure_key, "part": part_index, "parts": len(self.parts), "binding": self.binding.to_dict()}
        if self.form == FORM_SECTION:
            record["marker"] = self.markers[part_index] if part_index < len(self.markers) else "prose"
            record["list_type"] = self.list_type
        return record


def _procedure_parts(proc: Procedure, index: GuideIndex) -> Tuple[Part, ...]:
    """Every step of ``proc`` in source order; the first keeps its heading ("heading: 1. ..."), which is exactly its chunk,
    so the rendered procedure is introduced by its own printed heading and nothing else."""
    parts = [Part(step.chunk_id, proc.source_id, step.page, step.quote) for step in proc.steps]
    if proc.heading and parts:
        first = parts[0]
        headed = "%s: %s" % (proc.heading, first.quote)
        if index.chunks[first.chunk_id].startswith(headed):
            parts[0] = Part(first.chunk_id, first.source_id, first.page, headed)
    return tuple(parts)


_HEADED_FIRST_STEP = re.compile(r"^(?P<heading>[^\n]{2,160}?):\s+(?P<step>(?:\d{1,2}\.|[A-Za-z]\)|-)\s.*)$", re.S)


def render_part(quote: str, record: Optional[Mapping[str, Any]]) -> str:
    """Minimal deterministic formatting of one rendered guide part (D-111 rendering): the first part of a procedure puts
    its printed heading on its own line above step 1; a bulleted unit of a D-112 structure is shown as a bullet ("- ").
    Every word stays the source's; nothing is numbered that the source did not number; any other quote is returned as is."""
    if isinstance(record, Mapping) and record.get("form") == FORM_SECTION:
        return "- " + quote if record.get("marker") == "bullet" else quote
    if not isinstance(record, Mapping) or record.get("form") != FORM_PROCEDURE or record.get("part") != 0:
        return quote
    m = _HEADED_FIRST_STEP.match(quote or "")
    return "%s:\n\n%s" % (m.group("heading"), m.group("step")) if m else quote


def structure_answer(structure: Any, binding: Binding) -> GuideAnswer:
    """The D-112 answer of a recovered structure: every exact unit in source order, its printed marker kept."""
    parts = tuple(Part(p.chunk_id, structure.source_id, p.page, p.quote) for p in structure.parts)
    return GuideAnswer(binding, FORM_SECTION, parts, structure.key, structure.heading, tuple(p.marker for p in structure.parts),
                       structure.list_type)


def cut_step_problem(index: GuideIndex, chunk_id: int) -> Optional[str]:
    """Post-freeze PF-5B (D-114): why a statement taken from a procedure step chunk is cut, or None: the step is the last step
    of a procedure the index records as continuing on a row read as a heading (its text stops mid-sentence)."""
    proc = index.procedure_of_chunk(chunk_id) if isinstance(chunk_id, int) else None
    if proc is None or not proc.steps or proc.steps[-1].chunk_id != chunk_id:
        return None
    if any("continues" in str(p) for p in proc.problems):
        return "the statement is the last step of a procedure that continues on a row read as a heading, so it is cut"
    return None


def select_answer(binding: Binding, units: Sequence[Unit], index: GuideIndex, procedure_ranks: Mapping[str, int],
                  structure_ranks: Optional[Mapping[str, int]] = None) -> Tuple[Optional[GuideAnswer], List[Dict[str, Any]]]:
    """The evidence for the clause of ``binding`` from its guide, or None, with the candidates considered (ids, kinds
    and reasons for the diagnostics). ``units`` are the statement units of the retrieved chunks of this source in
    retrieval order; ``procedure_ranks`` maps the key of every procedure retrieval reached (one of its step or window
    chunks was retrieved) to its best rank. Only a retrieved procedure is considered, procedure-shaped evidence first
    for a procedure or list ask (D-111 evidence selection): heading overlap with the asked action and object, then
    retrieval rank, then the shorter list."""
    considered: List[Dict[str, Any]] = []
    intent = binding.intent
    if intent in (GuideIntent.PROCEDURE, GuideIntent.ENUMERATION):
        procs = [p for p in index.procedures_of(binding.source_id) if p.key in procedure_ranks]
        asked = set(binding.core) if intent is GuideIntent.PROCEDURE and binding.core else set(binding.terms)

        def order(proc: Procedure) -> Tuple[int, int, int]:
            return (-len(stated_terms(binding, asked, proc.heading or "")), procedure_ranks.get(proc.key, 10 ** 6), len(proc.steps))

        for proc in sorted(procs, key=order):
            problem = procedure_problem(proc, binding, index)
            if problem is None and intent is GuideIntent.PROCEDURE and not proc.sequential:
                single = [s for s in proc.steps if not missing_terms(binding, s.quote) and not missing_entities(binding, s.quote)]
                if len(single) == 1:
                    step = single[0]
                    considered.append({"procedure": proc.key, "form": FORM_LIST_ITEM, "chunk_id": step.chunk_id, "accepted": True})
                    return GuideAnswer(binding, FORM_LIST_ITEM, (Part(step.chunk_id, proc.source_id, step.page, step.quote),), proc.key, proc.heading), considered
            considered.append({"procedure": proc.key, "form": FORM_PROCEDURE, "accepted": problem is None, "reason": problem})
            if problem is None:
                return GuideAnswer(binding, FORM_PROCEDURE, _procedure_parts(proc, index), proc.key, proc.heading), considered
        # ---- post-freeze PF-3D (D-112): recovered sections and lists, after every D-111 procedure was refused ----------
        ranks = structure_ranks or {}
        candidates = [s for s in index.structures_of(binding.source_id) if s.key in ranks]

        def structure_order(s: Any) -> Tuple[int, int, int, int]:
            context = " ".join(filter(None, [s.heading, s.section_title]))
            return (-len(stated_terms(binding, asked, context)), ranks.get(s.key, 10 ** 6), 0 if s.kind == "list" else 1, len(s.parts))

        for structure in sorted(candidates, key=structure_order):
            problem = structure_problem(structure, binding, index)
            considered.append({"structure": structure.key, "form": FORM_SECTION, "list_type": structure.list_type, "accepted": problem is None,
                               "reason": problem})
            if problem is None:
                return structure_answer(structure, binding), considered
        if intent is GuideIntent.PROCEDURE:
            return None, considered
    for unit in units:
        if unit.source_id != binding.source_id:
            continue
        if unit.intro and intent is not GuideIntent.DEFINITION:
            considered.append({"chunk_id": unit.chunk_id, "kind": unit.kind, "accepted": False, "reason": "generic introduction"})
            continue
        conflict = index.conflict_problem(unit.chunk_id, unit.quote)
        if conflict:
            considered.append({"chunk_id": unit.chunk_id, "kind": unit.kind, "accepted": False, "reason": "source conflict"})
            continue
        problem = cut_step_problem(index, unit.chunk_id) or statement_problem(binding, unit.quote)
        considered.append({"chunk_id": unit.chunk_id, "kind": unit.kind, "accepted": problem is None, "reason": problem})
        if problem is None:
            return GuideAnswer(binding, FORM_STATEMENT, (Part(unit.chunk_id, unit.source_id, unit.page, unit.quote),)), considered
    return None, considered


_LABEL_TRIM = re.compile(r"^[^\w\"“(]+|[\s:]+$")


def citation_heading(record: Optional[Mapping[str, Any]], index: Optional[GuideIndex]) -> str:
    """Post-freeze PF-3D (D-112): the printed heading a VERIFIED guide answer is cited under, read from the guide index for
    the record's procedure or structure (never from the specialist's report, and never the chunk's ``section`` metadata,
    which is the PF-2 chunker's page hint and is often carried from an earlier page). A D-111 procedure or unordered list:
    its heading; a D-112 list: its section's title, else its lead-in; a D-112 section: its title; a statement: none ("")."""
    if not isinstance(record, Mapping) or index is None:
        return ""
    key = str(record.get("procedure_key") or "")
    heading: Optional[str] = None
    if record.get("form") == FORM_SECTION:
        structure = index.structure(key)
        if structure is not None:
            heading = structure.section_title if structure.kind == "list" and structure.section_title else structure.heading
    elif record.get("form") in (FORM_PROCEDURE, FORM_LIST_ITEM):
        proc = index.procedure(key)
        if proc is not None:
            heading = proc.heading
    return _LABEL_TRIM.sub("", heading or "").strip()


def verify_answer(clause: str, form: str, parts: Sequence[Tuple[int, str, Optional[int]]], procedure_key: Optional[str], index: GuideIndex,
                  specialist_sources: Optional[Iterable[str]] = None) -> Optional[str]:
    """The adapter's independent D-111 check of a guide answer: why it must not be served, or None. ``parts`` are
    ``(chunk_id, quote, page)`` in rendering order. Everything is recomputed from the clause, the registry, the chunk
    metadata and the procedure index; nothing the specialist reported is trusted except which form it claims."""
    if form not in FORMS:
        return "unknown guide answer form %r" % (form,)
    if not parts:
        return "a guide answer without evidence"
    chunks, metadata = index.chunks, index.metadata
    sources = set()
    for chunk_id, quote, page in parts:
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or not 0 <= chunk_id < len(chunks):
            return "evidence chunk id out of range"
        if not isinstance(quote, str) or not quote.strip() or quote not in chunks[chunk_id]:
            return "the quote is not in its chunk"
        meta = metadata[chunk_id]
        if page is not None and meta.get("page") != page:
            return "page mismatch"
        sources.add(meta.get("source_id"))
        conflict = index.conflict_problem(chunk_id, quote)
        if conflict:
            return conflict
    if len(sources) != 1:
        return "a guide answer must rest on one source"
    source_id = sources.pop()
    if specialist_sources is not None and source_id not in set(specialist_sources):
        return "source %s is not approved for the answering specialist" % source_id
    record = index.record(str(source_id))
    if record is None:
        return "source %s is not in the registry" % source_id
    binding, problem = index.bind(clause, str(source_id))            # D-114: the same vocabulary the specialist bound with
    if binding is None:
        return problem
    if form == FORM_STATEMENT:
        if len(parts) != 1:
            return "a statement answer has exactly one quote"
        return cut_step_problem(index, parts[0][0]) or statement_problem(binding, parts[0][1])
    if form == FORM_SECTION:                                   # post-freeze PF-3D (D-112)
        structure = index.structure(procedure_key or "")
        if structure is None or structure.source_id != source_id:
            return "the claimed structure is not in the structure index"
        problem = structure_problem(structure, binding, index)
        if problem:
            return problem
        if tuple((p[0], p[1]) for p in parts) != tuple((p.chunk_id, p.quote) for p in structure.parts):
            return "the units are not exactly the claimed structure's units in source order"
        return None
    proc = index.procedure(procedure_key or "")
    if proc is None or proc.source_id != source_id:
        return "the claimed procedure is not in the procedure index"
    problem = procedure_problem(proc, binding, index)
    if problem:
        return problem
    if form == FORM_LIST_ITEM:
        if binding.intent is not GuideIntent.PROCEDURE or proc.sequential or len(parts) != 1:
            return "only one item of an unordered list may answer alone"
        chunk_id, quote, _page = parts[0]
        step = next((s for s in proc.steps if s.chunk_id == chunk_id), None)
        if step is None or quote != step.quote:
            return "the list item is not a step of the claimed list"
        if missing_terms(binding, quote) or missing_entities(binding, quote):
            return "the list item does not state the asked terms"
        others = [s for s in proc.steps if s.chunk_id != chunk_id and not missing_terms(binding, s.quote) and not missing_entities(binding, s.quote)]
        if others:
            return "several items of the list answer the clause"
        return None
    if binding.intent not in (GuideIntent.PROCEDURE, GuideIntent.ENUMERATION):
        return "a %s ask is not answered with a procedure" % binding.intent.value
    expected = _procedure_parts(proc, index)
    got = tuple((p[0], p[1]) for p in parts)
    if got != tuple((p.chunk_id, p.quote) for p in expected):
        return "the steps are not exactly the claimed procedure's steps in source order"
    return None


#: Reasons of ``verify_answer`` / ``bind`` that concern the source's authority rather than the evidence's content.
_AUTHORITY_REASONS = ("not approved", "audience", "does not name the system", "is not answered from a system guide", "not usable",
                      "not a system guide", "not in the registry", "no approved source set")


def is_authority_problem(reason: Optional[str]) -> bool:
    """True when a D-111 refusal is about which source may answer (approval, audience, system, policy) rather than about
    what the evidence states."""
    return bool(reason) and any(marker in str(reason) for marker in _AUTHORITY_REASONS)


def guide_report_problem(report: Mapping[str, Any]) -> Optional[str]:
    """Why a clause report that claims the D-111 guide path is not eligible, or None."""
    record = report.get(GUIDE_KEY)
    if not isinstance(record, Mapping):
        return "no D-111 record"
    if record.get("version") != VERSION:
        return "D-111 record version %r is not %r" % (record.get("version"), VERSION)
    if record.get("form") not in FORMS:
        return "unknown guide answer form %r" % (record.get("form"),)
    if report.get("outcome") != "supported":
        return "guide clause outcome %s" % report.get("outcome")
    if not report.get("findings"):
        return "a guide clause report without findings"
    return None


def verify_guide_findings(clause: str, findings: Sequence[Any], specialist_id: str, index: Optional[GuideIndex]) -> Optional[str]:
    """The adapter-side D-111 check of the findings that answer ONE clause, or None. Every finding must carry a D-111
    record of the same form, source and procedure with part numbers 0..n-1 in order; the evidence is then re-verified by
    :func:`verify_answer` against the D-110 approved set of ``specialist_id``."""
    if index is None:
        return "no guide index: a guide answer cannot be verified"
    if not findings:
        return "a guide answer without evidence"
    records = [(getattr(f, "metadata", None) or {}).get(GUIDE_KEY) for f in findings]
    if not all(isinstance(r, Mapping) for r in records):
        return "a finding without a D-111 record answers a guide clause"
    first = records[0]
    for n, record in enumerate(records):
        if record.get("version") != VERSION:
            return "D-111 record version mismatch"
        if (record.get("form"), record.get("source_id"), record.get("procedure_key")) != (first.get("form"), first.get("source_id"), first.get("procedure_key")):
            return "the findings of one guide clause disagree on form, source or procedure"
        if record.get("part") != n or record.get("parts") != len(records):
            return "the guide findings are not the claimed parts in order"
    sources = index.policy.get(str(specialist_id))
    if sources is None:
        return "no approved source set for %s" % specialist_id
    parts = []
    for f in findings:
        if getattr(f, "source_id", None) != first.get("source_id"):
            return "a guide finding names another source than its record"
        parts.append(((f.metadata or {}).get("chunk_id"), getattr(f, "evidence_quote", None), getattr(f, "page", None)))
    return verify_answer(clause, str(first.get("form")), parts, first.get("procedure_key"), index, specialist_sources=sources)


def has_guide_record(item: Any) -> bool:
    """True for a finding (or clause report) that claims the D-111 guide path."""
    meta = getattr(item, "metadata", item)
    return isinstance(meta, Mapping) and isinstance(meta.get(GUIDE_KEY), Mapping)


__all__ = [
    "ANSWERABLE_INTENTS", "Binding", "FORMS", "FORM_LIST_ITEM", "FORM_PROCEDURE", "FORM_STATEMENT", "GUIDE_KEY", "GuideAnswer", "GuideIndex",
    "GuideIntent", "PHASE", "Part", "Procedure", "RELIABLE_HEADINGS", "Step", "Unit", "VERSION", "bind", "guide_report_problem",
    "has_guide_record", "is_authority_problem", "is_generic_introduction", "label_problem", "missing_entities", "missing_terms", "render_part",
    "procedure_problem", "read_intent", "select_answer", "statement_problem", "statement_units", "stem", "stems", "verify_answer",
    "verify_guide_findings",
    # post-freeze PF-5B (D-114)
    "CLAIM_WORDS", "MIXED_PROCEDURE", "ORGANIZATIONAL", "cut_step_problem", "cut_unit_problem", "disputed_claim", "mixed_problem", "stated_terms",
    "value_conflict",
]
