"""
Router agent.

DECISION OWNED
    Which query route should handle the user's question, and whether
    deterministic routing is ambiguous. Nothing else: the Router does not
    retrieve, does not answer and does not verify.

HOW IT DECIDES
    1. Every intent is scored from regular-expression cues. All intents are
       always scored. The Part 1 classifier was a first-match if-chain, so the
       first pattern that matched won regardless of what else was in the
       question; that was the cause of the documented routing failure.
    2. If the best intent beats the runner-up by at least
       ``ROUTER_AMBIGUITY_MARGIN`` the decision is final and no LLM is used.
    3. Otherwise the question is ambiguous and ONE bounded LLM call arbitrates
       between the tied candidates. It returns JSON validated by Pydantic.
    4. If that call is disabled, not permitted by the call budget, or fails for
       any reason (network, timeout, quota, malformed JSON, schema violation,
       an intent outside the candidates) the best deterministic route is
       returned. The Router never raises on a question.

THE ``number`` FIX
    Part 1's contact pattern contained the bare word ``number`` and contact was
    tested first, so "What is the number of degree programs?" was routed to
    ``contact``. Here ``number`` is never contact evidence by itself. Contact
    needs a channel word (phone, telephone, fax, email, extension), the verb
    ``contact``, or a phrase such as "phone number" or "number for <office>".
    "number of" is count evidence.

MULTI-PART DETECTION (Milestone 1: detected and reported, not acted on)
    ``is_multipart`` is true when the question holds two independently
    answerable questions: two question marks, two clauses that each open with
    an interrogative, or two strong structured intents in different clauses.
    A plain "and" between nouns ("rules and regulations") is not a split.
    The detector is deliberately conservative because a false positive will
    cost a Planner LLM call in Milestone 2.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..config import (
    GROQ_MODEL,
    ROUTER_AMBIGUITY_MARGIN,
    ROUTER_LLM_FALLBACK,
    ROUTER_LLM_MAX_TOKENS,
    ROUTER_LLM_TIMEOUT_S,
)
from ..text_utils import GREETING_PATTERN, MONTH_PATTERN
from .types import ALL_ROUTES, RouteDecision

#: Intents with at least this score are reported in ``RouteDecision.routes``.
RELEVANT_THRESHOLD = 0.5
#: Two structured intents this strong, in different clauses, mean two questions.
STRONG_THRESHOLD = 0.8
#: Every additional, independent kind of cue adds this much to an intent.
EXTRA_CUE_BONUS = 0.1
#: Ceiling for an intent supported only by body keywords. Only an intent whose
#: cue is a clause's own interrogative ("how many", "when") can reach 1.0. The
#: 0.2 gap is wider than ROUTER_AMBIGUITY_MARGIN on purpose: however many
#: keywords pile up, they never make a clear interrogative look ambiguous.
BODY_ONLY_CEILING = 0.8

#: Intents scored from cues. ``policy`` is the complement (see score_intents)
#: and ``greeting`` is matched on the whole string.
_SCORED_INTENTS: Tuple[str, ...] = ("contact", "count", "date", "list", "policy_yesno")
_STRUCTURED: Tuple[str, ...] = ("contact", "count", "date")

#: Auxiliary verbs. An interrogative is recognised by inversion ("when DO classes
#: begin"), which is what separates it from a conjunction ("when I resign").
_AUX = r"(?:is|are|was|were|am|do|does|did|can|could|may|might|should|shall|will|would|must|has|have|had)"


# ---------------------------------------------------------------------------
# Cues
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Cue:
    intent: str
    family: str          # cues of one family do not stack with each other
    weight: float        # 0-1: how strongly this cue alone indicates the intent
    pattern: "re.Pattern[str]"
    head: bool = False   # True: must open a clause (it is the clause's interrogative)


def _cue(intent: str, family: str, weight: float, regex: str, head: bool = False) -> _Cue:
    return _Cue(intent, family, weight, re.compile(regex, re.I), head)


# Weights. 1.0 = the clause's own interrogative names the answer type ("how
# many ...", "when ..."). 0.8 = an unambiguous keyword in the body. 0.4-0.6 = a
# hint that needs company. The 0.2 gap between an interrogative and a body
# keyword is deliberate and must stay above ROUTER_AMBIGUITY_MARGIN: in "how
# many departments have a phone number?" the interrogative decides, so the
# question is a clear count and no LLM call is spent on it.
_CUES: Tuple[_Cue, ...] = (
    # ---- contact ---------------------------------------------------------
    # "who should I contact ..." - the interrogative itself asks for a contact.
    # Only the verb "contact". "Who can call a council meeting?" and "who should
    # I ask about remote work?" expect a role or a rule, not a phone number, and
    # the contact tooling (extractor, Part 1 check) can only serve the latter:
    # routing them to contact turned answerable questions into refusals.
    _cue("contact", "who-contact", 1.0,
         r"^(?:who|whom)\b.{0,60}?\b(?:contact|get in touch)\b", head=True),
    # One family on purpose: "phone" and "phone number" are the same evidence
    # and must not stack into a score that rivals an interrogative.
    # "contactnumber" is what _prepare() leaves behind for "<channel> number".
    # "extension", "mobile" and "hotline" are NOT cues by themselves: "extension
    # of my contract" and "mobile devices in class" are policy questions.
    _cue("contact", "channel", 0.8, r"\b(?:phone|telephone|fax|e-?mail|contactnumber)\b"),
    _cue("contact", "contact-verb", 0.8, r"\b(?:contact|get in touch)\b"),
    # "the number for the IT help desk": a number FOR something is a contact
    # number; a number OF something is a count.
    _cue("contact", "number-for", 0.6, r"\bnumbers?\s+(?:for|to call|to reach)\b"),
    # ---- count -----------------------------------------------------------
    _cue("count", "how-many", 1.0, r"^how many\b", head=True),
    _cue("count", "how-many", 0.8, r"\bhow many\b"),
    _cue("count", "number-of", 0.8, r"\b(?:total number|number of|count of)\b"),
    _cue("count", "total", 0.5, r"\btotal\b"),
    # ---- date ------------------------------------------------------------
    # "when" is an interrogative only when the verb is inverted ("when do classes
    # begin"). "When I am sick, who do I contact?" opens with a conjunction.
    _cue("date", "when", 1.0, r"^when\s+" + _AUX + r"\b", head=True),
    _cue("date", "what-date", 1.0,
         r"^(?:what|which)(?:'s|\s+is|\s+are|\s+was|\s+were)?\s+(?:the\s+)?(?:dates?|day|deadlines?)\b", head=True),
    # The Part 1 calendar vocabulary plus plurals, nothing more. A question sent
    # to `date` is checked by the Part 1 date rule, which refuses any answer
    # without a date in it, so "what is the policy on midterm exams?" must not
    # land here.
    _cue("date", "calendar-noun", 0.8, r"\b(?:dates?|deadlines?|last day|add/drop|final exams?)\b"),
    _cue("date", "begin", 0.6, r"\b(?:begins?|starts?|held)\b"),
    _cue("date", "term", 0.4, r"\b(?:fall|spring|summer)\b"),
    # A "when" that does not open a clause is usually a conjunction
    # ("what should faculty do when ..."), so it is only a hint.
    _cue("date", "when", 0.3, r"\bwhen\b"),
    # ---- list (the Part 1 vocabulary, unchanged) --------------------------
    # 0.6: "what are the ..." and "name ..." are weak, structural hints. They
    # must lose clearly to a real keyword ("what are the ... dates", "name the
    # fax number of ...") yet still beat plain policy.
    _cue("list", "list", 0.6,
         r"\b(?:name|list|which are|what are the|standing committee|categories|core values)\b"),
    # ---- yes/no -----------------------------------------------------------
    # 0.6, not higher: "is there a deadline ...?" is a date question that happens
    # to be phrased as yes/no, so a body keyword (0.8) must clearly outrank this.
    _cue("policy_yesno", "aux-head", 0.6,
         r"^(?:does|do|did|can|could|is|are|was|were|may|should|will|would|must|has|have)\b", head=True),
)

# Handbook vocabulary that merely *contains* an intent word. It is neutralised
# before scoring so the word inside it is not counted as a cue.
_NEUTRALISE: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"\bcontact\s+hours?\b", re.I), "contacthours"),   # a unit of teaching load
    (re.compile(r"\bpoints?\s+of\s+contact\b", re.I), "contact"),   # keep as ONE contact cue
    # "<channel> number" is one contact expression. Rewriting it stops the count
    # cue "number of" from firing on "the phone number of the IT Center", which
    # made the most common contact question a contact/count tie.
    (re.compile(r"\b(phone|telephone|tel|fax|mobile|cell|contact|extension|hotline|office|whatsapp)"
                r"\s+(?:numbers?|no\.?)(?=\W|$)", re.I), r"\1 contactnumber"),
    # "the number of the <office>" asks for a phone number, not for a count.
    (re.compile(r"\bnumbers?\s+of\s+the\s+((?:[\w'&-]+\s+){0,6}?"
                r"(?:office|center|centre|desk|department|college|unit|clinic|hospital|library|deanship)\b)",
                re.I), r"contactnumber for the \1"),
)

# Politeness in front of the real question: "Can you tell me when ..." is a
# date question, not a yes/no question.
_POLITE_PREFIX = re.compile(
    r"^(?:please\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:tell|show|give|explain\s+to|let)\s+(?:me|us)\s*(?:know\s*)?"
    r"|(?:tell|show|give)\s+(?:me|us)\s*"
    r"|i\s+(?:want|need|would\s+like)\s+to\s+know\s*"
    r"|please\s+)",
    re.I,
)

_CLAUSE_SEP = re.compile(r"\s*(?:\?|;|&|,\s*and\b|\band\b|,|\bas well as\b|\.\s+(?=[a-z]))\s*", re.I)
# The FIRST clause is the main question, so any interrogative opening counts.
_WH_HEAD = re.compile(r"^(?:what|who|whom|whose|why|how|which)\b|^(?:when|where)\s+" + _AUX + r"\b", re.I)
_AUX_HEAD = re.compile(
    r"^(?:does|do|did|can|could|is|are|was|were|may|should|will|would|must|has|have)\b", re.I)
# A LATER clause must really be shaped like a question, because "and" joins far
# more noun phrases and predicates than questions:
#   wh-word + an auxiliary within a few words   "what ... are", "how many programs are"
#   "who" + a verb                               "who approves it"
#   "when"/"where" + inverted auxiliary          "when is notice required"
# which rejects noun clauses ("... and what the dean decides"), conjunctions
# ("... when I resign") and relative clauses ("..., which is chaired by the VC").
_LATER_WH_QUESTION = re.compile(
    r"^(?:when|where)\s+" + _AUX + r"\b"
    r"|^(?:what|whom|whose|why|how)\b(?:\s+\S+){0,5}?\s+" + _AUX + r"\b"
    r"|^who\s+(?:" + _AUX + r"|[a-z]+s)\b"
    r"|^which\s+(?!(?:is|are|was|were|has|have|can|will|may)\b)\S+(?:\s+\S+){0,4}?\s+" + _AUX + r"\b",
    re.I)
# A later yes/no clause needs an explicit subject straight after the auxiliary:
# "... and do I need approval?". Without one it is normally a second predicate
# of the SAME question ("... and have a PhD", "... and are required to sign in").
_LATER_AUX_QUESTION = re.compile(
    r"^" + _AUX + r"\s+(?:i|we|you|they|he|she|it|there|faculty|staff|students)\b", re.I)


# ---------------------------------------------------------------------------
# Diagnostics returned next to the contract type
# ---------------------------------------------------------------------------
@dataclass
class RouterReport:
    """A RouteDecision plus the evidence behind it, for the trace and for tests."""
    decision: RouteDecision
    margin: float
    ambiguous: bool
    candidates: List[str]                       # intents within the ambiguity margin of the best
    cues: Dict[str, List[str]] = field(default_factory=dict)   # intent -> cue families that fired
    llm_attempted: bool = False
    llm_error: Optional[str] = None             # why the LLM result was not used
    llm_skipped: Optional[str] = None           # "disabled" | "no_client" | "budget" | "multipart"
    ms: float = 0.0


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------
# Written as code points so this source file stays pure ASCII: an invisible
# literal non-breaking space in a regex module is a bug waiting to happen.
_NBSP = chr(0xA0)
_CURLY_APOSTROPHE = chr(0x2019)


def _prepare(question: str) -> str:
    q = (question or "").replace(_NBSP, " ").replace(_CURLY_APOSTROPHE, "'")
    q = re.sub(r"[\r\n]+", " ; ", q)          # a line break separates two questions
    q = re.sub(r"\s+", " ", q).strip(" ;").lower()
    for pattern, replacement in _NEUTRALISE:
        q = pattern.sub(replacement, q)
    return q


def _clauses(q: str) -> List[Tuple[int, str]]:
    """Split into clauses on ? ; , and 'and'. Returns (start offset, text)."""
    out: List[Tuple[int, str]] = []
    pos = 0
    for m in _CLAUSE_SEP.finditer(q):
        if m.start() > pos:
            out.append((pos, q[pos:m.start()]))
        pos = m.end()
    if pos < len(q):
        out.append((pos, q[pos:]))
    cleaned: List[Tuple[int, str]] = []
    for i, (start, text) in enumerate(out):
        if i == 0:
            stripped = _POLITE_PREFIX.sub("", text, count=1)
            start += len(text) - len(stripped)
            text = stripped
        lead = len(text) - len(text.lstrip(" \"'("))
        text = text.strip(" \"'()")
        if text:
            cleaned.append((start + lead, text))
    return cleaned


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_intents(question: str) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, int]]:
    """Score every intent. Returns (scores, cue families fired, first cue offset).

    An intent's score is its strongest cue plus ``EXTRA_CUE_BONUS`` for every
    further *independent* kind of cue, capped at 1.0 when one of its cues is a
    clause's interrogative and at ``BODY_ONLY_CEILING`` otherwise. ``policy`` is the
    complement of the strongest other intent: the less any specific cue fires,
    the more confident the Router is that this is a general handbook question.
    """
    q = _prepare(question)
    clauses = _clauses(q)

    fired: Dict[str, Dict[str, Tuple[float, int]]] = {i: {} for i in _SCORED_INTENTS}
    has_head = {i: False for i in _SCORED_INTENTS}

    def _note(cue: _Cue, pos: int) -> None:
        if cue.head:
            has_head[cue.intent] = True
        best = fired[cue.intent].get(cue.family)
        if best is None or cue.weight > best[0]:
            fired[cue.intent][cue.family] = (cue.weight, pos if best is None else min(pos, best[1]))

    for cue in _CUES:
        if cue.head:
            for start, text in clauses:
                if cue.pattern.search(text):
                    _note(cue, start)
        else:
            m = cue.pattern.search(q)
            if m:
                _note(cue, m.start())

    # A month name supports a date reading only when something else already does.
    if fired["date"] and MONTH_PATTERN.search(q):
        fired["date"].setdefault("month", (0.0, len(q)))

    scores: Dict[str, float] = {}
    cues: Dict[str, List[str]] = {}
    first: Dict[str, int] = {}
    for intent in _SCORED_INTENTS:
        families = fired[intent]
        if not families:
            scores[intent] = 0.0
            continue
        strongest = max(w for w, _ in families.values())
        ceiling = 1.0 if has_head[intent] else BODY_ONLY_CEILING
        scores[intent] = round(min(ceiling, strongest + EXTRA_CUE_BONUS * (len(families) - 1)), 2)
        cues[intent] = sorted(families)
        first[intent] = min(p for _, p in families.values())

    scores["policy"] = round(1.0 - max(scores.values()), 2)
    scores["greeting"] = 0.0
    return scores, cues, first


def detect_multipart(question: str,
                     scores: Optional[Dict[str, float]] = None,
                     first: Optional[Dict[str, int]] = None) -> bool:
    """True when the question contains two independently answerable questions."""
    q = _prepare(question)
    if not q:
        return False

    # 1. Two question marks, each closing at least two words. Counting the marks
    #    (not the text around them) keeps "What is the policy? Thanks in advance."
    #    a single question.
    asked = [part for part in q.split("?")[:-1] if len(part.split()) >= 2]
    if len(asked) >= 2:
        return True

    clauses = _clauses(q)
    if len(clauses) < 2:
        return False

    # 2. The first clause is a question and a LATER clause is shaped like one too
    #    (see _LATER_WH_QUESTION / _LATER_AUX_QUESTION). "and" joins far more
    #    nouns and predicates than questions, so the later clause carries the
    #    burden of proof. This misses some real splits on purpose: a false
    #    positive costs a Planner LLM call, a miss costs nothing in Milestone 1.
    first_text = clauses[0][1]
    if _WH_HEAD.match(first_text) or _AUX_HEAD.match(first_text):
        for _, text in clauses[1:]:
            if len(text.split()) < 3:
                continue
            if _LATER_WH_QUESTION.match(text) or _LATER_AUX_QUESTION.match(text):
                return True

    # 3. Two strong structured intents whose cues sit in different clauses.
    #    In the same clause it is ONE mixed-intent question ("how many
    #    departments have a phone number?"), which scoring resolves.
    if scores is None or first is None:
        scores, _, first = score_intents(question)
    strong = [i for i in _STRUCTURED if scores.get(i, 0.0) >= STRONG_THRESHOLD]
    if len(strong) >= 2:
        def _clause_of(offset: int) -> int:
            index = 0
            for n, (start, _) in enumerate(clauses):
                if offset >= start:
                    index = n
            return index
        if len({_clause_of(first[i]) for i in strong}) >= 2:
            return True
    return False


def _rank(scores: Dict[str, float], first: Dict[str, int]) -> List[str]:
    """Intents best-first. Ties break on where the cue appears (the earlier
    clause is the main question), then on the Part 1 order, so the result never
    depends on dict ordering."""
    far = 10 ** 6
    return sorted(
        (i for i in scores if i != "greeting"),
        key=lambda i: (-scores[i], first.get(i, far), ALL_ROUTES.index(i)),
    )


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------
_LLM_SYSTEM = (
    "You are the intent router of a university faculty-handbook assistant. "
    "Classify the user's question into exactly ONE intent and reply with a "
    "single JSON object and nothing else. Schema: "
    '{"intent": <one of the allowed intents>, "confidence": <number 0 to 1>, '
    '"is_multipart": <true if the text contains two independent questions, else false>}. '
    "Intent meanings: contact = wants a phone, fax, email or whom to contact; "
    "count = wants how many of something; date = wants a calendar date or deadline; "
    "list = wants several named items enumerated; policy_yesno = a yes/no question "
    "about a rule; policy = any other question about rules, procedures or descriptions."
)


def _safe_error(exc: BaseException) -> str:
    """Exception class plus a short message with anything key-shaped removed."""
    text = re.sub(r"gsk_[A-Za-z0-9]+", "gsk_***", str(exc))
    text = re.sub(r"\s+", " ", text)[:120]
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__


def _validate_llm_payload(raw: str, allowed: List[str]) -> Tuple[str, float, bool]:
    """Parse and validate the router LLM reply. Raises ValueError on any defect."""
    try:
        from pydantic import BaseModel, ConfigDict, Field, ValidationError
    except ImportError as exc:                       # pydantic ships with the groq SDK
        raise ValueError("pydantic unavailable") from exc

    class _Choice(BaseModel):
        model_config = ConfigDict(extra="ignore", strict=True)
        intent: str
        confidence: float = Field(ge=0.0, le=1.0)
        is_multipart: bool = False

    text = (raw or "").strip()
    try:
        payload = json.loads(text)
    except ValueError:
        block = re.search(r"\{.*\}", text, re.S)     # tolerate a code fence around the object
        if not block:
            raise ValueError("reply is not JSON")
        try:
            payload = json.loads(block.group(0))
        except ValueError as exc:
            raise ValueError("reply is not JSON") from exc
    if isinstance(payload, dict) and isinstance(payload.get("confidence"), int) \
            and not isinstance(payload.get("confidence"), bool):
        payload["confidence"] = float(payload["confidence"])   # 1 is a valid confidence
    try:
        choice = _Choice.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("schema violation (%d error(s))" % exc.error_count()) from exc
    intent = choice.intent.strip().lower()
    if intent not in allowed:
        raise ValueError("intent outside candidates: %s" % intent[:24])
    return intent, float(choice.confidence), bool(choice.is_multipart)


def _ask_llm(client, question: str, allowed: List[str]) -> Tuple[str, float, bool]:
    caller = client
    if hasattr(client, "with_options"):
        # Short timeout, no transport retries: a deterministic route is always
        # available, so a slow or failing arbitration is simply abandoned.
        caller = client.with_options(timeout=ROUTER_LLM_TIMEOUT_S, max_retries=0)
    response = caller.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=ROUTER_LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": "Allowed intents: %s\nQuestion: %s\nJSON:"
                                        % (", ".join(allowed), question.strip()[:600])},
        ],
    )
    return _validate_llm_payload(response.choices[0].message.content or "", allowed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def route_with_report(
    question: str,
    *,
    groq_client=None,
    llm_gate: Optional[Callable[[], bool]] = None,
    llm_fallback: Optional[bool] = None,
    margin: Optional[float] = None,
) -> RouterReport:
    """Route a question and explain the decision.

    ``llm_gate`` is called immediately before the LLM would be used. It must
    return True to permit the call and is expected to count it; the
    orchestrator passes its call budget here so the Router cannot spend a call
    the budget does not allow. ``llm_fallback`` and ``margin`` override the
    config values (used by tests).
    """
    started = time.perf_counter()
    use_llm = ROUTER_LLM_FALLBACK if llm_fallback is None else llm_fallback
    min_margin = ROUTER_AMBIGUITY_MARGIN if margin is None else margin

    text = (question or "").strip()
    if GREETING_PATTERN.search(text.lower()):
        scores = {i: 0.0 for i in ALL_ROUTES}
        scores["greeting"] = 1.0
        decision = RouteDecision("greeting", ["greeting"], False, 1.0, False, scores)
        return RouterReport(decision, 1.0, False, ["greeting"],
                            ms=(time.perf_counter() - started) * 1000.0)

    scores, cues, first = score_intents(text)
    ranked = _rank(scores, first)
    best, runner_up = ranked[0], ranked[1]
    gap = round(scores[best] - scores[runner_up], 2)
    ambiguous = gap < min_margin
    candidates = [i for i in ranked if scores[best] - scores[i] < min_margin]
    multipart = detect_multipart(text, scores, first)

    # Deterministic choice. Among tied candidates the general RAG route is the
    # safe one: it can answer any question, an extractor only its own kind.
    primary = "policy" if (ambiguous and "policy" in candidates) else best
    confidence = max(0.0, min(1.0, gap))
    used_llm = False
    report = RouterReport(
        decision=RouteDecision(primary, [], multipart, confidence, False, scores),
        margin=gap, ambiguous=ambiguous, candidates=candidates, cues=cues,
    )

    if ambiguous:
        if multipart:
            # Two questions explain the tie; the Planner will split them
            # (Milestone 2). Spending an LLM call to pick one would be waste.
            report.llm_skipped = "multipart"
        elif not use_llm:
            report.llm_skipped = "disabled"
        elif groq_client is None:
            report.llm_skipped = "no_client"
        elif llm_gate is not None and not llm_gate():
            report.llm_skipped = "budget"
        else:
            report.llm_attempted = True
            allowed = list(dict.fromkeys(candidates + ["policy"]))
            try:
                intent, llm_conf, llm_multi = _ask_llm(groq_client, text, allowed)
                primary, confidence, used_llm = intent, llm_conf, True
                multipart = multipart or llm_multi
            except Exception as exc:     # never let arbitration break a question
                report.llm_error = _safe_error(exc)

    relevant = [i for i in ranked if scores[i] >= RELEVANT_THRESHOLD and i != primary]
    report.decision = RouteDecision(
        primary_route=primary,
        routes=[primary] + relevant,
        is_multipart=multipart,
        confidence=round(confidence, 2),
        used_llm=used_llm,
        scores=scores,
    )
    report.ms = (time.perf_counter() - started) * 1000.0
    return report


def route(question: str, *, groq_client=None,
          llm_gate: Optional[Callable[[], bool]] = None) -> RouteDecision:
    """Route a question. See :func:`route_with_report` for the diagnostics."""
    return route_with_report(question, groq_client=groq_client, llm_gate=llm_gate).decision


def classify(question: str) -> str:
    """Deterministic route only - never calls an LLM."""
    return route_with_report(question, llm_fallback=False).decision.primary_route
