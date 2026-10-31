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
FORMS = (FORM_PROCEDURE, FORM_LIST_ITEM, FORM_STATEMENT)


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


def stems(text: str) -> FrozenSet[str]:
    """Stems of every word of ``text`` (stop words included: coverage asks whether a term is stated, nothing more)."""
    return frozenset(stem(t) for t in _TOKEN.findall(text or ""))


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

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable; the registry's system names and counts only (no question text, no word of it)."""
        return {"source_id": self.source_id, "intent": self.intent.value, "names_bound": list(self.names_bound), "actor_bound": self.actor_bound,
                "terms": len(self.terms), "core": len(self.core), "entities": len(self.entities), "minimum": self.minimum, "maximum": self.maximum}


_MIN_WORDS = re.compile(r"\b(?:minimum|min|least|smallest|shortest)\b", re.I)
_MAX_WORDS = re.compile(r"\b(?:maximum|max|most|largest|longest|biggest|limit|limits|limited)\b", re.I)


def bind(clause: str, record: Any) -> Tuple[Optional[Binding], Optional[str]]:
    """``(binding, None)`` when ``record`` may answer ``clause`` under D-111, else ``(None, reason)``. Never raises for
    ordinary input; the reason is a fixed phrase plus ids (no question text)."""
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
    from .plan_f_runtime import named_entities                    # imported here: the adapter imports this module
    entities = tuple(named_entities(blanked))
    drop = {stem(w) for w in _INTENT_WORDS.get(intent, frozenset())}
    head = info.get("object_head")
    if intent in (GuideIntent.SELECTION, GuideIntent.ENUMERATION) and isinstance(head, str) and head:
        drop.add(stem(head))                                       # the asked control or set is what the evidence supplies

    def content(tok: str) -> bool:
        low = tok.lower()
        return low not in _STOP and low not in _FRAME and stem(tok) not in drop

    terms = tuple(dict.fromkeys(stem(t) for t in _TOKEN.findall(blanked) if content(t)))
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
    binding = Binding(source_id, intent, tuple(names), bool(actors), terms, core, entities,
                      minimum=intent is GuideIntent.VALUE and bool(_MIN_WORDS.search(blanked)),
                      maximum=intent is GuideIntent.VALUE and bool(_MAX_WORDS.search(blanked)) and not _MIN_WORDS.search(blanked),
                      info=dict(info))
    return binding, None


def missing_terms(binding: Binding, text: str) -> List[str]:
    """Terms of ``binding`` that ``text`` does not state."""
    present = stems(text)
    return [t for t in binding.terms if t not in present]


def missing_entities(binding: Binding, text: str) -> List[str]:
    from .plan_f_runtime import entity_in_text
    return [e for e in binding.entities if not entity_in_text(e, text)]


# ---------------------------------------------------------------------------
# What a statement must show for each intent
# ---------------------------------------------------------------------------
#: A stated quantity: a number with its unit (a list label such as "1." is never a value).
_QUANTITY = re.compile(r"(?<![\w.])\d[\d,.]*\s*(?:x\s*\d[\d,.]*\s*)?(?:%|percent\b|characters?\b|pixels?\b|px\b|pages?\b|mb\b|kb\b|gb\b|"
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
    if binding is not None and binding.terms and any(t in stems(text) for t in binding.terms):
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
        if prefix:
            units.append(Unit(chunk_id, source_id, page, chunk.strip(), "step", rank))
        return [u for u in units if u.quote]
    if kind != "paragraph":
        return []
    spans = _spans(chunk)
    singles: List[Unit] = []
    pairs: List[Unit] = []
    for i, (a, b, _before) in enumerate(spans):
        quote = chunk[a:b].strip(_GLYPH + " ")
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
        """Ordered steps: numbered or lettered labels, or a "how to / steps" heading over a bulleted list."""
        if any(label != "-" for label in self.labels):
            return True
        return bool(self.heading and _HOWTO_HEADING.match(self.heading))

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
        """A registered source conflict on the chunk (PF-2 stamp) or in the quoted text (the registry's phrases)."""
        meta = self.metadata[chunk_id] if 0 <= chunk_id < len(self.metadata) else {}
        topic = meta.get("source_conflict")
        if isinstance(topic, str) and topic:
            return "source conflict %r on the evidence chunk" % topic
        record = self.records.get(str(meta.get("source_id")))
        for conflict in getattr(record, "conflicts", ()) or ():
            if conflict.matches(quote):
                return "source conflict %r in the quoted text" % conflict.topic
        return None


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


def procedure_problem(proc: Procedure, binding: Binding, index: GuideIndex) -> Optional[str]:
    """Why ``proc`` is not a provably complete answer to the procedure (or list) ask of ``binding``, or None."""
    if proc.source_id != binding.source_id:
        return "the procedure belongs to another source"
    if proc.problems:
        return proc.problems[0]
    problem = label_problem(proc.labels)
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
    if not asked & stems(proc.heading or ""):
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

    @property
    def pages(self) -> Tuple[int, ...]:
        return tuple(dict.fromkeys(p.page for p in self.parts))

    def record(self, part_index: int) -> Dict[str, Any]:
        """The D-111 record a finding carries (ids, stems and labels only; no question text)."""
        return {"version": VERSION, "phase": PHASE, "source_id": self.binding.source_id, "form": self.form, "intent": self.binding.intent.value,
                "procedure_key": self.procedure_key, "part": part_index, "parts": len(self.parts), "binding": self.binding.to_dict()}


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
    its printed heading on its own line above step 1. Every word stays the source's; any other quote is returned as is."""
    if not isinstance(record, Mapping) or record.get("form") != FORM_PROCEDURE or record.get("part") != 0:
        return quote
    m = _HEADED_FIRST_STEP.match(quote or "")
    return "%s:\n\n%s" % (m.group("heading"), m.group("step")) if m else quote


def select_answer(binding: Binding, units: Sequence[Unit], index: GuideIndex,
                  procedure_ranks: Mapping[str, int]) -> Tuple[Optional[GuideAnswer], List[Dict[str, Any]]]:
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
            return (-len(asked & stems(proc.heading or "")), procedure_ranks.get(proc.key, 10 ** 6), len(proc.steps))

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
        problem = statement_problem(binding, unit.quote)
        considered.append({"chunk_id": unit.chunk_id, "kind": unit.kind, "accepted": problem is None, "reason": problem})
        if problem is None:
            return GuideAnswer(binding, FORM_STATEMENT, (Part(unit.chunk_id, unit.source_id, unit.page, unit.quote),)), considered
    return None, considered


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
    binding, problem = bind(clause, record)
    if binding is None:
        return problem
    if form == FORM_STATEMENT:
        if len(parts) != 1:
            return "a statement answer has exactly one quote"
        return statement_problem(binding, parts[0][1])
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
]
