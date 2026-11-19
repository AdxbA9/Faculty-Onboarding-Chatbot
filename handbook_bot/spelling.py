"""
Conservative spelling correction for English questions ("who aproves confrence travel" -> "who approves conference
travel").

A word is corrected only when ALL of these hold, so real words and names are never changed:

* it is a Latin word of 4+ letters, not an acronym, not containing digits;
* it is NOT a known English word (pyspellchecker's dictionary) and NOT a word of the loaded documents;
* a candidate within two edits exists that IS a word of the loaded documents (the closest one wins; among equally
  close ones the word the documents use most, so "leve" becomes "leave", not "love").

"paternity" is a real English word, so it is never "corrected" to "maternity" even though only the latter is in the
handbook. Arabic text is left to the bilingual layer. Without pyspellchecker installed, nothing is corrected.
"""
from __future__ import annotations

import difflib
import re
import threading
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]")
_LOCK = threading.Lock()
_CACHE: Dict[int, "Speller"] = {}

try:                                                    # optional dependency (pure Python, ships an English dictionary)
    from spellchecker import SpellChecker as _SpellChecker
except Exception:                                       # pragma: no cover - exercised only where it is not installed
    _SpellChecker = None


class Speller:
    """Corrects unknown words towards the vocabulary of the loaded documents."""

    def __init__(self, texts: Iterable[str]):
        counts: Counter = Counter()
        for text in texts:
            if isinstance(text, str):
                counts.update(w.lower() for w in _WORD.findall(text))
        self.vocab = counts
        self._by_initial: Dict[str, list] = {}
        for word in counts:
            if len(word) >= 4:
                self._by_initial.setdefault(word[0], []).append(word)
        self._dict = _SpellChecker(distance=2) if _SpellChecker is not None else None

    def known(self, word: str) -> bool:
        w = word.lower()
        return w in self.vocab or (self._dict is not None and w in self._dict)

    def suggest(self, word: str) -> Optional[str]:
        """The corrected form of ``word`` (lower case), or None when it should stay as written."""
        w = word.lower()
        if self._dict is None or len(w) < 4 or not w.isalpha() or self.known(w):
            return None
        if word.isupper() and len(word) <= 6:            # acronyms: UOS, LMS, ILHE
            return None
        candidates = set(c for c in (self._dict.candidates(w) or ()) if c in self.vocab)
        candidates.update(difflib.get_close_matches(w, self._by_initial.get(w[0], ()), n=5, cutoff=0.8))
        candidates = [c for c in candidates if _edits(w, c) <= 2]
        if not candidates:
            return None
        return max(candidates, key=lambda c: (-_edits(w, c), self.vocab.get(c, 0)))

    def correct(self, text: str) -> Tuple[str, Dict[str, str]]:
        """``(corrected text, {original word: replacement})``. Case of the first letter is kept."""
        changes: Dict[str, str] = {}

        def repl(m: "re.Match[str]") -> str:
            word = m.group(0)
            fix = self.suggest(word)
            if not fix:
                return word
            fixed = fix.capitalize() if word[:1].isupper() else fix
            changes[word] = fixed
            return fixed

        return _WORD.sub(repl, text or ""), changes


def _edits(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (small strings only)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def speller_for(kb: Any) -> Optional[Speller]:
    """The (cached) speller built from a knowledge base's chunks, or None when there is nothing to build from."""
    chunks = getattr(kb, "chunks", None)
    if not chunks or _SpellChecker is None:
        return None
    key = id(chunks)
    with _LOCK:
        speller = _CACHE.get(key)
        if speller is None:
            speller = _CACHE[key] = Speller(chunks)
        return speller


def available() -> bool:
    return _SpellChecker is not None


__all__ = ["Speller", "available", "speller_for"]
