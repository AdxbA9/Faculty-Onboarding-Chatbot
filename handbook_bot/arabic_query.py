"""
Arabic question to English retrieval query (post-freeze PF-11, approach B: controlled query transformation).

The approved documents are English and every retrieval, routing and verification layer of the chatbot reads English
(English-only embedder and reranker, regular-expression cues, English request shapes). An Arabic question is therefore
rewritten, deterministically and auditably, into the English question the documents would answer:

    normalise -> tokenise -> greedy longest-phrase lookup in the bilingual lexicon -> render in English order

The rendering keeps the clause structure (``و`` and ``؟`` stay ``and`` and ``?``), renders stripped particles
("for", "in", "and"), inserts "of the" after a head noun ("phone number OF THE Human Resources Department"), moves the
subject into a verb-first frame ("does {subj} provide"), turns "what is the" into "what are the" before a plural and
"how" into "how do I" before a verb. Latin words, acronyms and numbers pass through unchanged, so code-switched
questions ("كيف أستخدم Blackboard لإرسال Announcement؟") keep their English terms. Unknown Arabic words are dropped from
the query and reported; ``coverage`` says how much of the question's content the lexicon understood.

Model fallback (bounded, optional): when the coverage is below ``ARABIC_QUERY_MIN_COVERAGE`` and a Groq client is given,
ONE translation request may replace the lexicon query. Its output is validated (Latin text, sane length, every digit of
the question preserved) and recorded next to the lexicon query, so the transformation stays auditable. Without a client
(tests, offline evaluation, no key) the lexicon query is used as it is.

Nothing here answers a question: the English query goes into the unchanged pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import arabic_lexicon
from .arabic_text import (PARTICLE_MEANING, LanguageProfile, detect_language, has_arabic, lookup_forms, normalize_arabic,
                          tokenize)

#: Longest phrase (in tokens) the lexicon lookup tries.
MAX_PHRASE = 6
#: English function words: never a subject, never a head's complement, never counted as content.
_FUNCTION = frozenset({"in", "on", "to", "about", "with", "from", "through", "until", "after", "before", "during", "between", "or", "and",
                       "then", "but", "this", "same", "per", "at", "according to", "that", "not", "without", "except", "cannot",
                       "not allowed", "true", "all", "some", "who", "what", "which", "when", "where", "why", "how", "is", "can", "must",
                       "state", "ignore", "summarize", "tell me everything about"})
_ARABIC_TOKEN = re.compile(r"^[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]+$")
_LATIN_TOKEN = re.compile(r"^[A-Za-z]")
_NUMBER_TOKEN = re.compile(r"^\d")
_PUNCT = {"?", "!", ",", ";", ":", "."}
_KEY_SHAPED = re.compile(r"gsk_[A-Za-z0-9]+")

_INDEX = arabic_lexicon.build_index()
_FIRST: Dict[str, List[Tuple[str, ...]]] = {}
for _key in _INDEX:
    _FIRST.setdefault(_key[0], []).append(_key)
for _keys in _FIRST.values():
    _keys.sort(key=len, reverse=True)
_GREETINGS = frozenset(normalize_arabic(g) for g in arabic_lexicon.GREETINGS)


@dataclass
class Piece:
    """One rendered unit of the English query."""
    kind: str                      # term | latin | number | punct | unknown
    english: str
    flags: str = ""
    particle: Optional[str] = None
    surface: str = ""              # the Arabic tokens this piece came from (normalised)
    consumed: bool = False         # moved into a {subj} slot

    @property
    def is_noun(self) -> bool:
        """A noun or modifier that can start a noun phrase (fills a {subj} slot, takes "of the" after a head)."""
        if self.kind in ("latin", "number"):
            return True
        if self.kind != "term" or not self.english:
            return False
        if any(f in self.flags.split() for f in ("q", "v", "x", "neg")):
            return False
        return self.english not in _FUNCTION

    @property
    def is_verb(self) -> bool:
        return self.kind == "term" and "v" in self.flags.split()


@dataclass
class QueryTransform:
    """What happened to one question on its way to the English pipeline (JSON-serialisable through ``to_dict``)."""
    original: str
    language: LanguageProfile
    english_query: str
    lexicon_query: str
    method: str                                   # passthrough | lexicon | model | greeting | empty
    mapped: List[Dict[str, str]] = field(default_factory=list)
    unmapped: List[str] = field(default_factory=list)
    coverage: float = 1.0
    llm_calls: int = 0
    model_error: Optional[str] = None
    is_greeting: bool = False
    meaning_loss: List[str] = field(default_factory=list)       # PF-16: why the lexicon query is unsafe (see meaning_loss())

    def to_dict(self) -> Dict[str, Any]:
        return {"language": self.language.to_dict(), "english_query": self.english_query, "lexicon_query": self.lexicon_query,
                "method": self.method, "mapped": list(self.mapped), "unmapped": list(self.unmapped), "coverage": round(self.coverage, 3),
                "llm_calls": self.llm_calls, "model_error": self.model_error, "is_greeting": self.is_greeting,
                "meaning_loss": list(self.meaning_loss)}


def is_arabic_greeting(text: str) -> bool:
    """True when the whole (normalised) text is an Arabic greeting, punctuation aside."""
    norm = normalize_arabic(text).strip(" ?!.,;:")
    return bool(norm) and norm in _GREETINGS


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
def _match_at(tokens: List[str], forms: List[List[Tuple[str, Optional[str]]]], i: int) -> Optional[Tuple[int, Tuple[str, str], Optional[str], Tuple[str, ...]]]:
    """Longest lexicon phrase starting at token ``i``: ``(length, (english, flags), particle, key)`` or None."""
    best: Optional[Tuple[int, Tuple[str, str], Optional[str], Tuple[str, ...]]] = None
    best_cost = 0
    for rank, (form, particle) in enumerate(forms[i]):
        for key in _FIRST.get(form, ()):
            length = len(key)
            if length > MAX_PHRASE or i + length > len(tokens):
                continue
            if best is not None and length < best[0]:
                continue
            cost = rank                                   # lookup forms are ordered exact form first; inflected forms cost more
            ok = True
            for j in range(1, length):
                hit = next((r for r, (f, _) in enumerate(forms[i + j]) if f == key[j]), None)
                if hit is None:
                    ok = False
                    break
                cost += hit
            if ok and (best is None or length > best[0] or cost < best_cost):
                best, best_cost = (length, _INDEX[key], particle, key), cost
    return best


def _pieces(tokens: List[str]) -> List[Piece]:
    forms = [lookup_forms(t) if _ARABIC_TOKEN.match(t) else [] for t in tokens]
    out: List[Piece] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _PUNCT:
            out.append(Piece("punct", tok, surface=tok))
            i += 1
            continue
        if _LATIN_TOKEN.match(tok):
            out.append(Piece("latin", tok, surface=tok))
            i += 1
            continue
        if _NUMBER_TOKEN.match(tok):
            out.append(Piece("number", tok, surface=tok))
            i += 1
            continue
        match = _match_at(tokens, forms, i)
        if match is None:
            out.append(Piece("unknown", "", surface=tok))
            i += 1
            continue
        length, (english, flags), particle, _ = match
        out.append(Piece("term", english, flags, particle, " ".join(tokens[i:i + length])))
        i += length
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _gerund(verb: str) -> str:
    words = verb.split()
    w = words[0]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]                                         # third-person form -> base form
    if w.endswith("ie"):
        g = w[:-2] + "ying"
    elif w.endswith("e") and not w.endswith("ee"):
        g = w[:-1] + "ing"
    elif len(w) >= 3 and not w.endswith("ed") and w[-1] not in "aeiouwxy" and w[-2] in "aeiou" and w[-3] not in "aeiou" and w not in ("visit", "open", "offer", "cover", "travel"):
        g = w + w[-1] + "ing"
    else:
        g = w + "ing"
    return " ".join([g] + words[1:])


#: Arabic adjectives follow their noun; these English renderings are moved in front of the noun they qualify.
_ADJECTIVES = frozenset({"new", "additional", "final", "full", "direct", "calendar", "required", "first", "unlimited", "mandatory", "paid", "full-time", "part-time", "evening", "monthly", "minimum", "maximum",
                         "reduced", "international", "local", "university-supported", "non-administrative", "direct teaching", "Muslim",
                         "annual", "total", "other", "same", "true", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"})
#: Post-nominal phrases that stay after the noun and never take "of the" in front of them.
_TAILS = frozenset({"an unlimited number of", "two projects", "with full pay", "at half pay", "without pay", "per week", "per semester", "per year", "per", "at once", "AED", "first year",
                    "including", "according to", "fall", "spring", "summer", "January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December"})
_PREPOSITIONS = frozenset({"in", "on", "about", "with", "through", "after", "before", "during", "without", "for", "from", "of", "at"})
_QUANTITY_FRAMES = ("how many", "how much", "what percentage", "how long", "number of", "duration", "percentage", "amount")


def _is_adjective(p: Piece) -> bool:
    return p.kind == "term" and p.english in _ADJECTIVES


def _is_plain_noun(p: Piece) -> bool:
    return p.is_noun and not _is_adjective(p) and p.english not in _TAILS


def _next_noun(pieces: List[Piece], start: int) -> Optional[int]:
    """Index of the first noun after ``start`` in the same clause (skipping dropped words), or None."""
    for j in range(start, len(pieces)):
        p = pieces[j]
        if p.kind == "punct" or (p.kind == "term" and p.english in ("and", "or", "then", "but")):
            return None
        if p.consumed or p.kind == "unknown" or (p.kind == "term" and "x" in p.flags.split()):
            continue
        if p.is_noun:
            return j
        return None
    return None


def _noun_phrase(pieces: List[Piece], j: int) -> str:
    """ONE noun term at ``j`` with the adjectives that follow it in Arabic, rendered in English order; all consumed."""
    head = pieces[j]
    head.consumed = True
    adjectives: List[str] = []
    k = j + 1
    while k < len(pieces) and _is_adjective(pieces[k]) and not pieces[k].consumed and pieces[k].particle is None:
        pieces[k].consumed = True
        adjectives.append(pieces[k].english)
        k += 1
    return " ".join(adjectives + [head.english])


def render(pieces: List[Piece]) -> str:
    words: List[str] = []
    prev: Optional[Piece] = None               # the last piece that produced words
    last_noun_at: Optional[int] = None         # index in ``words`` of the last plain noun (adjectives go in front of it)
    for idx, p in enumerate(pieces):
        if p.consumed or p.kind == "unknown" or (p.kind == "term" and "x" in p.flags.split()):
            continue
        if p.kind == "punct":
            if words:
                words[-1] = words[-1] + p.english
            prev = p
            last_noun_at = None
            continue
        english = p.english
        prev_flags = prev.flags.split() if (prev is not None and prev.kind == "term") else []
        prev_english = prev.english if prev is not None else ""
        if p.kind == "term" and english == "from":
            if prev is None or prev.kind == "punct" or prev_english in ("and", "then", "but"):
                english = "who"
            elif ("q" in prev_flags and prev_english.startswith(_QUANTITY_FRAMES)) or prev_english in _QUANTITY_FRAMES \
                    or prev_english in ("days", "hours", "weeks", "months", "years", "times", "working days", "calendar days"):
                english = "of"
        # ---- stripped particle ("for", "in", "and", "to <verb>") -------------------------------------------------------------
        if p.particle and p.kind == "term":
            meaning = PARTICLE_MEANING.get(p.particle, "")
            if p.particle == "ل" and p.is_verb:
                meaning = "to"
            elif p.particle == "ب" and prev_english.endswith(("do", "contact", "work")):
                meaning = "with"
            elif p.particle == "ل" and prev_english.endswith("what is the"):
                meaning = "of the"
            if p.particle == "ب" and prev_english.endswith(("what is", "what is the")):
                meaning = ""                                      # "what is meant BY X" -> "what is X"
            if p.particle == "\u0644" and prev_english.endswith(("minimum", "maximum")):
                meaning = ""                                      # "the minimum for X" -> "minimum X"
            if meaning and not (words and words[-1].endswith(meaning)):
                words.append(meaning)
                prev_flags, prev_english = [], meaning
        # ---- verb-first frame: "does {subj} provide" takes the next noun phrase -------------------------------------------------
        if "{subj}" in english:
            verb_after = ""
            k = idx + 1
            while k < len(pieces) and pieces[k].kind == "term" and ("x" in pieces[k].flags.split() or pieces[k].consumed):
                k += 1
            if k < len(pieces) and pieces[k].is_verb and "{subj}" not in pieces[k].english and _next_noun(pieces, k + 1) is not None:
                pieces[k].consumed = True
                verb_after = pieces[k].english
                j = _next_noun(pieces, k + 1)
            else:
                j = _next_noun(pieces, idx + 1)
            subject = _noun_phrase(pieces, j) if j is not None else ""
            english = re.sub(r"\s{2,}", " ", (english.replace("{subj}", subject) + (" " + verb_after if verb_after else ""))).strip()
        # ---- adjustments that depend on the previous word -------------------------------------------------------------------------
        if words:
            if "pl" in p.flags.split() and words[-1].endswith("what is the"):
                words[-1] = words[-1][:-len("what is the")] + "what are the"
            if "h" in prev_flags and p.particle is None and prev is not None and not prev.consumed:
                if _is_plain_noun(p):
                    words.append("of the")
                elif p.is_verb and "{subj}" not in p.english:
                    words.append("for")
                    english = _gerund(english)
            elif prev_english in _PREPOSITIONS and p.is_verb and "{subj}" not in p.english:
                english = _gerund(english)
            if words[-1].endswith("how do I") and p.kind == "term" and p.english.startswith("can I"):
                words[-1] = words[-1][:-len("how do I")] + "how"
        if _is_adjective(p) and p.particle is None and prev is not None and prev.kind == "term" and "q" in prev_flags and words \
                and prev_english.split()[-1] in ("days", "hours", "times") and words[-1].endswith(prev_english):
            head, last = words[-1].rsplit(" ", 1)
            words[-1] = "%s %s %s" % (head, english, last)
            prev = p
            continue
        # ---- Arabic adjective after its noun -> English adjective before it ---------------------------------------------------------
        if (_is_adjective(p) and p.particle is None and last_noun_at is not None and prev is not None and _is_plain_noun(prev)
                and not (last_noun_at >= 1 and words[last_noun_at - 1].split()[-1] in ("is", "are", "does", "do", "can", "must", "may"))):
            words.insert(last_noun_at, english)
            prev = p
            continue
        if english:
            words.append(english)
            last_noun_at = len(words) - 1 if _is_plain_noun(p) else last_noun_at
        prev = p
    if words and not any(p.kind == "term" and "q" in p.flags.split() for p in pieces) \
            and not words[0].split()[0].lower() in ("who", "what", "when", "where", "why", "how", "which", "is", "can", "must", "does", "do", "are", "may", "summarize", "ignore", "tell"):
        words.insert(0, "what is the")
    text = " ".join(w for w in words if w).strip()
    text = re.sub(r"\s+([?,;:.!])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\b(the|of the|for|in|and|to|of) \1\b", r"\1", text)
    text = re.sub(r"\bof the (in|on|for|to|about|with|from|through)\b", r"\1", text)
    if text and not text.endswith(("?", ".", "!")):
        text += "?"
    return text[:1].upper() + text[1:] if text else text


# ---------------------------------------------------------------------------
# Model fallback (one bounded request)
# ---------------------------------------------------------------------------
_QUERY_SYSTEM = ("You translate Arabic questions from university faculty members into faithful English questions for searching "
                 "official university documents. Keep every number, name, acronym, condition and negation. Do not answer the "
                 "question and do not add information. Reply with the English question only.")


def _redact(exc: BaseException) -> str:
    text = _KEY_SHAPED.sub("gsk_***", str(exc))
    return ("%s: %s" % (type(exc).__name__, re.sub(r"\s+", " ", text)[:120])).strip(": ")


def translate_question(client, question: str) -> str:
    """ONE translation request; the caller counts it. Raises on failure or on an unusable reply."""
    from .config import GROQ_MODEL, LLM_REASONING_TOKENS
    caller = client.with_options(max_retries=0, timeout=20.0) if hasattr(client, "with_options") else client
    resp = caller.chat.completions.create(
        model=GROQ_MODEL, temperature=0.0, max_tokens=160 + LLM_REASONING_TOKENS,
        messages=[{"role": "system", "content": _QUERY_SYSTEM}, {"role": "user", "content": question.strip()[:600]}])
    text = (resp.choices[0].message.content or "").strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    if not text or len(text) > 400 or not re.search(r"[A-Za-z]{2}", text) or has_arabic(text):
        raise ValueError("unusable translation")
    wanted = set(re.findall(r"\d+", normalize_arabic(question)))
    if not wanted.issubset(set(re.findall(r"\d+", text))):
        raise ValueError("translation lost a number")
    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
#: Post-freeze PF-16 (D-127) meaning-loss guard. A negation, refusal or prohibition word that the lexicon drops or renders
#: without any English negation inverts or erases what the question asks; an unknown word after a contact keyword means the
#: unit whose number is asked for is incomplete ("phone number of the center" for the Central Laboratories). Either makes
#: the lexicon query unsafe however high its coverage is, so the validated model translation replaces it when a client is
#: configured. Multi-word lexicon phrases are trusted (``غير المتفرغ`` is "part-time", not a negation).
_NEGATION_WORDS = frozenset(normalize_arabic(w) for w in (
    "لا", "لم", "لن", "ليس", "ليست", "عدم", "غير", "دون", "بدون", "بلا", "رفض", "يرفض", "أرفض", "نرفض", "ترفض",
    "ممنوع", "يمنع", "تمنع", "يحظر", "محظور"))
_CONTACT_WORDS = frozenset(normalize_arabic(w) for w in (
    "رقم", "أرقام", "هاتف", "هاتفيا", "تلفون", "تليفون", "فاكس", "بريد", "ايميل", "إيميل", "اتصل", "أتصل", "اتصال", "تواصل", "أتواصل"))
_ENGLISH_NEGATION = re.compile(r"\b(?:not|no|without|cannot|never|refus\w*|prohibit\w*|forbid\w*|declin\w*)\b", re.I)


def _bare(token: str) -> str:
    """A token without one leading conjunction or preposition letter and without the definite article."""
    t = token
    if len(t) > 3 and t[0] in "وفبل":
        t = t[1:]
    if len(t) > 3 and t.startswith("ال"):
        t = t[2:]
    return t


def meaning_loss(pieces: List[Piece]) -> List[str]:
    """Reasons why the lexicon query does not say what the Arabic question asks (empty when none)."""
    reasons: List[str] = []
    for p in pieces:
        words = p.surface.split()
        if len(words) != 1:
            continue                                          # a multi-word lexicon phrase carries its own meaning
        w = words[0]
        if w not in _NEGATION_WORDS and (len(w) <= 3 or w[0] not in "وف" or w[1:] not in _NEGATION_WORDS):
            continue
        if p.kind == "unknown":
            reasons.append("negation_dropped")
        elif p.kind == "term" and "neg" not in p.flags.split() and not _ENGLISH_NEGATION.search(p.english or ""):
            reasons.append("negation_mistranslated")
    head = next((i for i, p in enumerate(pieces) if any(w in _CONTACT_WORDS or _bare(w) in _CONTACT_WORDS for w in p.surface.split())), None)
    if head is not None and any(p.kind == "unknown" and len(p.surface) >= 2 for p in pieces[head + 1:]):
        reasons.append("contact_target_incomplete")
    return sorted(set(reasons))


def transform_query(question: str, *, client=None, allow_model: Optional[bool] = None, min_coverage: Optional[float] = None) -> QueryTransform:
    """The English retrieval query for ``question`` and the record of how it was made. Never raises."""
    from . import config
    text = (question or "").strip()
    profile = detect_language(text)
    if not has_arabic(text):
        return QueryTransform(text, profile, text, text, "passthrough")
    if is_arabic_greeting(text):
        return QueryTransform(text, profile, "Hello", "Hello", "greeting", is_greeting=True)
    norm = normalize_arabic(text)
    tokens = tokenize(norm)
    pieces = _pieces(tokens)
    lexicon_query = render(pieces)
    mapped = [{"ar": p.surface, "en": p.english} for p in pieces if p.kind == "term" and p.english and p.english not in _FUNCTION
              and "x" not in p.flags.split()]
    unmapped = [p.surface for p in pieces if p.kind == "unknown" and len(p.surface) >= 2]
    content = len(mapped) + len(unmapped) + sum(1 for p in pieces if p.kind == "latin")
    coverage = 1.0 if content == 0 else (len(mapped) + sum(1 for p in pieces if p.kind == "latin")) / content
    result = QueryTransform(text, profile, lexicon_query, lexicon_query, "lexicon" if lexicon_query.strip("? ") else "empty",
                            mapped=mapped, unmapped=unmapped, coverage=coverage, meaning_loss=meaning_loss(pieces))
    use_model = config.ARABIC_QUERY_MODEL_FALLBACK if allow_model is None else bool(allow_model)
    threshold = config.ARABIC_QUERY_MIN_COVERAGE if min_coverage is None else float(min_coverage)
    if client is not None and use_model and (coverage < threshold or result.method == "empty" or result.meaning_loss):
        result.llm_calls = 1
        try:
            result.english_query = translate_question(client, text)
            result.method = "model"
        except Exception as exc:                                  # the lexicon query stands; the reason is recorded
            result.model_error = _redact(exc)
    return result


__all__ = ["MAX_PHRASE", "Piece", "QueryTransform", "is_arabic_greeting", "render", "transform_query", "translate_question"]
