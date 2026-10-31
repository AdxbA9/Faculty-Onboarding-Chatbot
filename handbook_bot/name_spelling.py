"""
Bounded correction of misspelled system names in a Plan F question (post-freeze PF-5B, decision record D-114).

WHY
    The PF-4 held-out set showed faculty writing "blackbord", "ithentcate", "gradbook" or "colab": the guide path is bound
    to a guide only by the system the question names, so a misspelled name sent a system question to the handbook path
    (or to no specialist). Correcting ordinary words is not safe ("polcy" must never become "policy" by guesswork), so only
    the names of the documented systems are ever corrected, and only when the correction is unambiguous.

WHAT (deterministic; no model; standard library only)
    ``correct_system_names(text, is_word)`` returns the text with each misspelled system name replaced by its canonical
    spelling, and one record per correction (the canonical name and the kind of correction; never the typed word):
      * a token is a candidate only when it is at least the name's minimum length, starts with the name's first letter,
        is not the name itself, is not an inflection of it ("Blackboards"), is not a known English word or neighbouring
        product within reach ("backboard", "authenticate", "collaborative", "Blackbaud", "Collabora") and is not a word the
        knowledge base itself prints (``is_word``);
      * it is corrected when exactly ONE system name lies within the bound: one edit of any kind, or (Blackboard,
        Collaborate, iThenticate) two edits that are insertions, deletions or adjacent transpositions only (two edits with a
        substitution reach other words and products: "Blackbaud"); Banner, whose one-letter substitutions are English words
        ("banker", "manner"), is corrected only by one inserted or deleted letter; two names in reach is ambiguous and
        nothing is corrected;
      * Collaborate is also an English verb, so "colab" (the abbreviation of the product) and a misspelled Collaborate are
        corrected only in PRODUCT position, the positions the guide's system vocabulary names it in (right before Ultra,
        session(s), room(s) or recording(s), or right after Blackboard or BB), and only when the question also names
        Blackboard, BB or Ultra; "colab with ...", "students colab in Blackboard", "the faculty colab room" and "Google
        Colab" are never the product.
    Policy, HR and handbook words are never corrected: they are not system names.
    ``typed_slice`` maps a slice of the corrected text (a request clause) back to the words the user typed.

USE
    ``plan_f_runtime.PlanFRuntime.attempt`` routes and grounds with the corrected text; the question the user typed stays
    the display, legacy and log text, and a served answer's clauses are mapped back to it. The LEGACY mode never imports
    this module.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

VERSION = "name-spelling-1"

#: (canonical name, maximum edit distance, minimum token length)
NAMES: Tuple[Tuple[str, int, int], ...] = (("Blackboard", 2, 7), ("Collaborate", 2, 8), ("iThenticate", 2, 8), ("Gradebook", 1, 8),
                                           ("Banner", 1, 5))
#: Names corrected only by one inserted or deleted letter (a substitution makes an English word).
INSERT_DELETE_ONLY = frozenset({"Banner"})
#: Abbreviations of a product name.
ALIASES: Dict[str, str] = {"colab": "Collaborate"}
#: Names that are also English words: corrected only in product position and with product context.
PRODUCT_POSITION_ONLY = frozenset({"Collaborate"})
_FOLLOWING_OBJECT = re.compile(r"\s+(?:ultra|sessions?|rooms?|recordings?)\b", re.I)
_PRECEDING_PRODUCT = re.compile(r"\b(?:blackboard|bb)\s+$", re.I)
_PRODUCT_CONTEXT = re.compile(r"\b(?:blackboard|bb|ultra)\b", re.I)
_OTHER_PRODUCT = re.compile(r"\bgoogle\s+$", re.I)
#: English words and neighbouring products that the other rules would still correct (each is reachable: a test checks it).
#: Every other near-miss ("blackbird", "Blackbaud", "authenticate", "banker", "collaborated") is already refused by the
#: first-letter, inflection, substitution or Banner rules.
NOT_TYPOS = frozenset({"backboard", "blackbeard", "collabora", "collaborative"})
_INFLECTIONS = ("s", "es", "d", "ed", "ing", "'s", "’s")
_TOKEN = re.compile(r"[A-Za-z]+")
Span = Tuple[int, int, str, str]


def osa_distance(a: str, b: str, substitutions: bool = True) -> int:
    """Optimal string alignment distance (insertions, deletions, adjacent transpositions and, unless ``substitutions`` is
    False, substitutions)."""
    big = len(a) + len(b) + 1
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            same = a[i - 1] == b[j - 1]
            replace = rows[i - 1][j - 1] + (0 if same else (1 if substitutions else big))
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1, replace)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[len(a)][len(b)]


def _within(low: str, name: str, bound: int) -> bool:
    canon = name.lower()
    if name in INSERT_DELETE_ONLY:
        return abs(len(low) - len(canon)) == 1 and osa_distance(low, canon, substitutions=False) == 1
    if osa_distance(low, canon) <= 1:
        return True
    return bound >= 2 and osa_distance(low, canon, substitutions=False) <= 2


def _candidate(low: str, is_word: Optional[Callable[[str], bool]]) -> Optional[str]:
    if low in NOT_TYPOS:
        return None
    hits: List[str] = []
    for name, bound, minimum in NAMES:
        canon = name.lower()
        if low == canon or len(low) < minimum or low[:1] != canon[:1]:
            continue
        if any(low == canon + suffix for suffix in _INFLECTIONS):
            continue                                              # "Blackboards" names the system, it is not a typo
        if _within(low, name, bound):
            hits.append(name)
    if len(hits) != 1:
        return None                                               # none, or ambiguous: never guess
    if is_word is not None and is_word(low):
        return None                                               # a word the knowledge base prints is not a typo
    return hits[0]


def _apply(text: str, chosen: Sequence[Span]) -> str:
    out, pos = [], 0
    for start, end, name, _kind in sorted(chosen):
        out.append(text[pos:start])
        out.append(name)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _product_position(text: str, start: int, end: int, others: Sequence[Span]) -> bool:
    """True when the token at ``start:end`` stands where the guides' system vocabulary names the product (right before
    Ultra or a Collaborate object, or right after Blackboard or BB) and the question, with the other names corrected, also
    names Blackboard, BB or Ultra outside the token."""
    if _OTHER_PRODUCT.search(text[:start]):
        return False                                              # "Google Colab" is another product
    before = _apply(text[:start], [o for o in others if o[1] <= start])
    if not (_PRECEDING_PRODUCT.search(before) or _FOLLOWING_OBJECT.match(text[end:])):
        return False
    removed = end - start - 1
    moved = [(s - removed, e - removed, n, k) if s >= end else (s, e, n, k) for s, e, n, k in others]
    return bool(_PRODUCT_CONTEXT.search(_apply(text[:start] + " " + text[end:], moved)))


def correct_with_spans(text: str, is_word: Optional[Callable[[str], bool]] = None) -> Tuple[str, List[Dict[str, str]], List[Span]]:
    """``correct_system_names`` plus the corrections kept, as ``(start, end, name, kind)`` spans of the typed text (for
    ``typed_slice``; never logged)."""
    text = text or ""
    found: List[Span] = []
    for m in _TOKEN.finditer(text):
        low = m.group(0).lower()
        if low in ALIASES:
            found.append((m.start(), m.end(), ALIASES[low], "alias"))
            continue
        name = _candidate(low, is_word)
        if name is not None:
            found.append((m.start(), m.end(), name, "edit"))
    others = [f for f in found if f[2] not in PRODUCT_POSITION_ONLY]
    kept = [f for f in found if f[2] not in PRODUCT_POSITION_ONLY or _product_position(text, f[0], f[1], others)]
    if not kept:
        return text, [], []
    return _apply(text, kept), [{"name": name, "kind": kind} for _s, _e, name, kind in kept], kept


def correct_system_names(text: str, is_word: Optional[Callable[[str], bool]] = None) -> Tuple[str, List[Dict[str, str]]]:
    """``(text with misspelled system names corrected, one record per correction)``. Records carry the canonical name and
    the kind ("edit" or "alias"), never the typed word. Deterministic; never raises for a string."""
    corrected, records, _spans = correct_with_spans(text, is_word)
    return corrected, records


def typed_slice(typed: str, spans: Sequence[Span], corrected: str, piece: str) -> str:
    """The words the user typed for ``piece``, a slice of ``corrected`` (a request clause): every corrected name inside it
    is mapped back to its typed token. ``piece`` itself when nothing was corrected or it is not a slice of ``corrected``."""
    if not spans or not piece:
        return piece
    at = corrected.find(piece)
    if at < 0:
        return piece
    end = at + len(piece)
    # (corrected start, corrected end, typed start, typed end) of every segment, in order
    segments: List[Tuple[int, int, int, int]] = []
    typed_pos = corr_pos = 0
    for s, e, name, _kind in sorted(spans):
        if s > typed_pos:
            segments.append((corr_pos, corr_pos + s - typed_pos, typed_pos, s))
            corr_pos += s - typed_pos
        segments.append((corr_pos, corr_pos + len(name), s, e))
        corr_pos += len(name)
        typed_pos = e
    segments.append((corr_pos, corr_pos + len(typed) - typed_pos, typed_pos, len(typed)))

    def position(offset: int, closing: bool) -> int:
        for cs, ce, ts, te in segments:
            if cs <= offset < ce or (closing and offset == ce):
                if ce - cs == te - ts:
                    return ts + (offset - cs)                     # an unchanged stretch
                return te if closing else ts                      # a corrected name maps to its whole token
        return len(typed)

    return typed[position(at, False):position(end, True)]


__all__ = ["ALIASES", "INSERT_DELETE_ONLY", "NAMES", "NOT_TYPOS", "PRODUCT_POSITION_ONLY", "VERSION", "correct_system_names",
           "correct_with_spans", "osa_distance", "typed_slice"]
