"""
Arabic text layer (post-freeze PF-11): normalisation, script detection and light stemming.

Everything here is deterministic and dependency-free. It never changes what the user typed: the page keeps and shows the
original question; these helpers produce internal forms for language detection and for the bilingual lexicon lookup
(``handbook_bot.arabic_lexicon`` / ``handbook_bot.arabic_query``).

Normalisation (``normalize_arabic``)
    Unicode NFKC, Arabic-Indic and Extended Arabic-Indic digits to ASCII digits, Arabic punctuation to its ASCII
    counterpart (``؟`` ``،`` ``؛``), diacritics (tashkeel) and tatweel removed, alef variants folded to a bare alef,
    alef maqsura to ya, hamza carriers folded (``ؤ`` to ``و``, ``ئ`` to ``ي``). Ta marbuta (``ة``) is KEPT: folding it to
    ``ه`` would merge distinct words (library ``مكتبة`` and office ``مكتب``).

Language detection (``detect_language``)
    Counts Arabic-script letters against Latin letters. ``ar``: Arabic dominates; ``en``: Latin dominates; ``mixed``: both
    scripts carry a real share (code-switching such as "كيف أستخدم Blackboard؟"); ``unknown``: no letters at all. The
    ``primary`` language of a mixed text is the language with more words (ties go to the opening script): the question
    frame ("how do I ...") is what decides how the question should be answered.

Light stemming (``strip_article``, ``inflection_forms``)
    Lookup forms of a token: the token as written, then the token without its definite article or clitic particle (``ال`` ``لل`` ``وال`` ``بال``
    ``كال`` ``فال``, then a lone ``و`` ``ف`` ``ب`` ``ل`` ``ك``), and the same without a common inflectional suffix (``ات``
    ``ون`` ``ين`` ``ان`` ``ها`` ``هم`` ``كم`` ``نا`` ``ني`` ``ية`` ``ي`` ``ه`` ``ك``). The lexicon stores article-free surface
    forms, so a query token matches when ANY of its forms is a lexicon form. The stripped particle is reported so the
    query builder can render it ("for", "in", "and").
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

#: Unicode blocks counted as Arabic script (the same blocks the knowledge base uses for its page filter).
ARABIC_BLOCKS = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_TATWEEL = "ـ"
_ALEF_VARIANTS = re.compile("[آأإٱ]")          # آ أ إ ٱ -> ا
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_PUNCT_MAP = {ord("؟"): "?", ord("،"): ",", ord("؛"): ";", ord("٪"): "%", ord("٫"): ".", ord("٬"): ","}
_FOLD_MAP = {ord("ى"): "ي", ord("ؤ"): "و", ord("ئ"): "ي"}   # ى->ي ؤ->و ئ->ي

#: Clitic prefixes, longest first (``لل`` is ``ل`` + ``ال`` with the alef dropped).
ARTICLE_PREFIXES: Tuple[str, ...] = ("وال", "فال", "بال", "كال", "لل", "ال")
#: Lone clitic particles that may precede a word without an article.
PARTICLE_PREFIXES: Tuple[str, ...] = ("و", "ف", "ب", "ل", "ك")
#: Inflectional suffixes, longest first.
SUFFIXES: Tuple[str, ...] = ("تين", "تان", "ات", "ون", "ين", "ان", "ها", "هم",
                             "كم", "نا", "ني", "ية", "يه", "ي", "ه", "ك", "ا")
#: What a stripped particle means in the English rendering.
PARTICLE_MEANING = {"و": "and", "ف": "and", "ب": "in", "ل": "for", "ك": "as"}

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'&./-]*|\d+(?:[.,:/-]\d+)*%?|[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]+|[?!,;:.]")


def is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in ARABIC_BLOCKS)


def has_arabic(text: str) -> bool:
    return any(is_arabic_char(c) for c in (text or ""))


def normalize_arabic(text: str) -> str:
    """The internal matching form of ``text`` (see the module docstring). Never shown to the user."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(_DIGIT_MAP).translate(_PUNCT_MAP).translate(_FOLD_MAP)
    out = out.replace(_TATWEEL, "")
    out = _DIACRITICS.sub("", out)
    out = _ALEF_VARIANTS.sub("ا", out)
    return _WS.sub(" ", out).strip()


def western_digits(text: str) -> str:
    """Only the digit folding of ``normalize_arabic`` (used on answers, where nothing else may change)."""
    return (text or "").translate(_DIGIT_MAP)


def tokenize(text: str) -> List[str]:
    """Tokens of a normalised text: Latin words, numbers, Arabic words and sentence punctuation."""
    return _TOKEN.findall(text or "")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LanguageProfile:
    code: str                 # "ar" | "en" | "mixed" | "unknown"
    primary: str              # "ar" | "en" | "unknown": the language the question is framed in
    arabic_letters: int
    latin_letters: int

    @property
    def arabic_share(self) -> float:
        total = self.arabic_letters + self.latin_letters
        return self.arabic_letters / total if total else 0.0

    @property
    def is_arabic(self) -> bool:
        """True when the question should be treated as an Arabic question (pure Arabic or Arabic-framed code-switching)."""
        return self.primary == "ar"

    def to_dict(self) -> dict:
        return {"code": self.code, "primary": self.primary, "arabic_letters": self.arabic_letters, "latin_letters": self.latin_letters,
                "arabic_share": round(self.arabic_share, 3)}


def detect_language(text: str) -> LanguageProfile:
    """Script-based language profile of ``text`` (tolerant of short questions and code-switching)."""
    arabic = latin = 0
    arabic_words = latin_words = 0
    first: Optional[str] = None
    for word in re.split(r"\s+", text or ""):
        a = sum(1 for ch in word if is_arabic_char(ch) and ch.isalpha())
        l = sum(1 for ch in word if ch.isascii() and ch.isalpha())
        arabic += a
        latin += l
        if a and a >= l:
            arabic_words += 1
            first = first or "ar"
        elif l:
            latin_words += 1
            first = first or "en"
    total = arabic + latin
    if total == 0:
        return LanguageProfile("unknown", "unknown", 0, 0)
    share = arabic / total
    if share >= 0.85:
        code = "ar"
    elif share <= 0.15:
        code = "en"
    else:
        code = "mixed"
    if code == "ar":
        primary = "ar"
    elif code == "en":
        primary = "en"
    else:                                   # code-switching: the language with more words frames the question; ties go to the opening script
        primary = "ar" if arabic_words > latin_words else ("en" if latin_words > arabic_words else (first or "en"))
    return LanguageProfile(code, primary, arabic, latin)


# ---------------------------------------------------------------------------
# Light stemming for lexicon lookup
# ---------------------------------------------------------------------------
def strip_article(word: str) -> Tuple[str, Optional[str]]:
    """``(stem, particle)``: the word without its clitic prefix, and the lone particle that was in front of it (None when
    there was none or when only the article was removed)."""
    for prefix in ARTICLE_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 2:
            particle = prefix[0] if prefix not in ("ال",) else None
            if prefix == "لل":
                particle = "ل"
            return word[len(prefix):], particle
    return word, None


def particle_forms(word: str) -> List[Tuple[str, Optional[str]]]:
    """Candidate ``(form, particle)`` pairs for a word that may carry a lone particle: the word itself and, when it opens
    with ``و`` ``ف`` ``ب`` ``ل`` ``ك`` and is long enough, the rest with that particle."""
    out: List[Tuple[str, Optional[str]]] = [(word, None)]
    if (len(word) >= 4 and word[0] in PARTICLE_PREFIXES) or (len(word) == 3 and word[0] in ("و", "ف")):
        rest = word[1:]
        stem, inner = strip_article(rest)
        out.append((stem if inner is None else rest, word[0]))
        if stem != rest and inner is None:
            out.append((stem, word[0]))
    return out


def inflection_forms(word: str) -> List[str]:
    """Lookup forms of an article-free word: the word, then the word without one inflectional suffix (and the ``ة``
    singular when a feminine plural ``ات`` was removed)."""
    forms = [word]
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[:-len(suffix)]
            if base not in forms:
                forms.append(base)
            if suffix == "ات" and base + "ة" not in forms:
                forms.append(base + "ة")
            if suffix in ("ه", "ك", "ي") and base + "ة" not in forms:   # possessive on a feminine noun
                forms.append(base + "ة")
    return forms


def lookup_forms(token: str) -> List[Tuple[str, Optional[str]]]:
    """Every ``(form, particle)`` a normalised Arabic token may stand for, best first: article-free exact form, then
    inflection-stripped forms, then the same for the token read as particle + word."""
    seen: List[Tuple[str, Optional[str]]] = []
    stem, particle = strip_article(token)
    candidates: List[Tuple[str, Optional[str]]] = [(token, None)] + [(f, particle) for f in inflection_forms(stem)]
    for form, part in particle_forms(token):
        inner, inner_particle = strip_article(form)
        for f in inflection_forms(inner):
            candidates.append((f, part or inner_particle))
    for cand in candidates:
        if cand not in seen:
            seen.append(cand)
    return seen


__all__ = ["ARABIC_BLOCKS", "LanguageProfile", "PARTICLE_MEANING", "detect_language", "has_arabic", "inflection_forms", "is_arabic_char",
           "lookup_forms", "normalize_arabic", "particle_forms", "strip_article", "tokenize", "western_digits"]
