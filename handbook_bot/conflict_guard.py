"""
Registered source conflicts on the legacy answer path (post-freeze PF-7, decision record D-117).

WHY
    Plan F keeps every registered conflict out of its answers (``plan_f_runtime.source_conflict_problem``; claim level for
    a value conflict since D-114). The legacy path that every mode falls back to (retrieval, synthesis, Verifier) never
    read the conflict stamp: with a working language model a stamped chunk reached the prompt and its disputed value
    was served. This module applies the same rule to the legacy evidence and to the legacy answer, from the registry
    alone. It names no topic, document, question or value.

WHAT
    * :func:`registered_conflicts`: source id -> topic -> ``SourceConflict`` of every registry record (cached per file).
    * :func:`screen_items`: before retrieved chunks are used as evidence, a chunk stamped with a VALUE conflict (one of
      its phrases states a number) loses every unit that states the disputed claim; any other stamp (a recommendation
      conflict, or a topic not registered for the chunk's own source) removes the chunk.
    * :func:`answer_problem`: a final answer is refused when one of its units states the disputed claim of a value
      conflict whose stamped chunk was among the evidence retrieved for the question (a phrase, a number that counts
      the subject of the phrases, or a count or limit word with that subject; :func:`answer_states_disputed_claim`).

    A unit states the disputed claim (the D-114 rule of ``guide_grounding.disputed_claim``) when it contains one of the
    conflict's phrases, a count or limit word, or a number (a list label is not one) together with a content word of
    one of the phrases. Nothing is merged or chosen between the conflicting values.

This module imports only the standard library and ``sources``, so LEGACY never imports a Plan F module.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .sources import SourceConflict, default_registry_path, load_source_registry

#: Words that make a unit a claim about a count or a limit (the D-114 list).
CLAIM_WORDS = re.compile(r"\b(?:quotas?|count(?:s|ed|ing)?|limit(?:s|ed|ation)?|allowances?|yearly|annual(?:ly)?|"
                         r"per\s+(?:year|month|semester|term))\b", re.I)
_NUMBER = re.compile(r"(?<![\w.])\d[\d,.]*")
_LEADING_LABEL = re.compile(r"^\s*(?:\d{1,2}[.)]|[A-Za-z][.)]|[-–•])\s+")
_WORD = re.compile(r"[A-Za-z]+")
_STOP = frozenset("a an the of to in on at by for per and or with is are be this that each every".split())
#: Unit boundaries inside a chunk: line breaks, row separators, bullet glyphs, sentence ends, numbered labels "1)" and
#: dash-led steps ("File Requirements: - Each ...").
_UNIT_BREAK = re.compile(r"\s*(?:\n+|\s\|\s|[•▪●]|(?<=[.!?;])\s+(?=[A-Z0-9“\"(])|"
                         r"\s(?=\d{1,2}\)\s)|\s-\s(?=[A-Z]))\s*")

_CACHE: Dict[str, Tuple[float, Dict[str, Dict[str, SourceConflict]]]] = {}


def _stem(word: str) -> str:
    word = word.lower()
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _content_words(text: str) -> set:
    return {_stem(w) for w in _WORD.findall(text or "") if w.lower() not in _STOP}


def split_units(text: str) -> List[str]:
    """The statements of a chunk or an answer, in order (empty pieces dropped)."""
    return [u.strip() for u in _UNIT_BREAK.split(text or "") if u and u.strip()]


def is_value_conflict(conflict: Any) -> bool:
    """True when the conflict disputes a value (one of its phrases states a number); only such a conflict is isolated
    at claim level, every other conflict blocks its whole chunk (D-114)."""
    return any(re.search(r"\d", str(p)) for p in (getattr(conflict, "phrases", ()) or ()))


def states_disputed_claim(conflict: Any, text: str) -> bool:
    """True when ``text`` states the claim the value conflict disputes: one of its phrases, a count or limit word, or a
    number (a list label is not one) together with a content word of one of its phrases."""
    text = text or ""
    if conflict.matches(text) or CLAIM_WORDS.search(text):
        return True
    body = _LEADING_LABEL.sub("", text, count=1)
    if not _NUMBER.search(body):
        return False
    phrase_words = set().union(*(_content_words(str(p)) for p in (getattr(conflict, "phrases", ()) or ())))
    return bool(phrase_words & _content_words(body))


#: Words of a conflict phrase that say how much or how often, not what about; they never make an answer about the
#: conflict's subject ("thirty days of annual leave each year" is not about a "documents per year" limit).
_GENERIC = frozenset(_stem(w) for w in """year years month months week weeks day days semester semesters term terms time times
maximum minimum number amount specific total limit per""".split())


_NUMBER_WORDS = frozenset("""one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred""".split())
_TOKEN = re.compile(r"\d[\d,.]*|[A-Za-z]+")
#: A rate in an answer ("every year", "a semester", "monthly"): with the subject of a value conflict it states a limit.
_RATE = re.compile(r"\b(?:per|a|an|each|every)\s+(?:academic\s+)?(?:year|month|semester|term)\b|\b(?:annually|monthly)\b", re.I)


def answer_states_disputed_claim(conflict: Any, text: str) -> bool:
    """True when an ANSWER unit states the disputed claim: one of the conflict's phrases; a number (in digits or words)
    that counts the subject of the phrases within the next three words ("30 documents", "fifteen submissions"); or a
    count, limit or rate word in the same unit as that subject ("the yearly limit on submissions", "submit ... every
    year"). The subject is a phrase word or a word sharing its first five letters ("submit"). A number that measures
    something else in a sentence that merely names the subject ("the document title ... under 100 MB") is not the claim.
    Stricter than :func:`states_disputed_claim`, which judges the units of a chunk already stamped with the conflict."""
    text = text or ""
    if conflict.matches(text):
        return True
    body = _LEADING_LABEL.sub("", text, count=1)
    subject = set().union(*(_content_words(str(p)) for p in (getattr(conflict, "phrases", ()) or ()))) - _GENERIC

    def about_subject(word: str) -> bool:           # the same word, or one sharing its first five letters ("submit")
        word = _stem(word)
        return any(word == s or (len(word) >= 5 and len(s) >= 5 and word[:5] == s[:5]) for s in subject)

    if not subject or not any(about_subject(w) for w in _WORD.findall(body)):
        return False
    if CLAIM_WORDS.search(body) or _RATE.search(body):
        return True
    tokens = [t.lower() for t in _TOKEN.findall(body)]
    for i, token in enumerate(tokens):
        if token[:1].isdigit() or token in _NUMBER_WORDS:
            if any(about_subject(t) for t in tokens[i + 1:i + 4] if t.isalpha()):
                return True
    return False


def registered_conflicts(path: Optional[str] = None) -> Dict[str, Dict[str, SourceConflict]]:
    """source id -> topic -> conflict, for every record of the registry (enabled or not). An unreadable registry gives
    no conflicts, so every stamp is then treated as unknown and blocks its whole chunk."""
    path = path or default_registry_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        registry = load_source_registry(path)
        conflicts = {r.source_id: {c.topic: c for c in (r.conflicts or ())} for r in registry.list_sources()}
    except Exception:                       # a broken registry must not take the answer path down
        conflicts = {}
    _CACHE[path] = (mtime, conflicts)
    return conflicts


def _stamp(item: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    meta = item.get("meta") if isinstance(item.get("meta"), Mapping) else {}
    topic = meta.get("source_conflict")
    return (str(meta.get("source_id")) if meta.get("source_id") is not None else None,
            topic if isinstance(topic, str) and topic else None)


def stamped_topics(items: Iterable[Mapping[str, Any]]) -> List[Tuple[Optional[str], str]]:
    """(source id, topic) of every conflict stamp on ``items``, first occurrence order."""
    seen: List[Tuple[Optional[str], str]] = []
    for item in items:
        sid, topic = _stamp(item)
        if topic is not None and (sid, topic) not in seen:
            seen.append((sid, topic))
    return seen


def _lookup(conflicts: Mapping[str, Mapping[str, SourceConflict]], sid: Optional[str], topic: str) -> Optional[SourceConflict]:
    return (conflicts.get(sid) or {}).get(topic) if sid is not None else None


def screen_items(items: Sequence[Dict[str, Any]],
                 conflicts: Optional[Mapping[str, Mapping[str, SourceConflict]]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(kept items, report). A stamped chunk of a value conflict keeps only its units that do not state the disputed
    claim (a copy of the item, ``meta["conflict_screened"] = topic``); it is removed when nothing is left or the kept
    text still carries a conflict phrase. A chunk with any other stamp is removed. Unstamped items are kept as they are."""
    conflicts = registered_conflicts() if conflicts is None else conflicts
    kept: List[Dict[str, Any]] = []
    report: List[Dict[str, Any]] = []
    for item in items:
        sid, topic = _stamp(item)
        if topic is None:
            kept.append(item)
            continue
        meta = item.get("meta") if isinstance(item.get("meta"), Mapping) else {}
        entry: Dict[str, Any] = {"topic": topic, "source_id": sid, "page": meta.get("page")}
        conflict = _lookup(conflicts, sid, topic)
        if conflict is None or not is_value_conflict(conflict):
            entry["action"] = "chunk_removed"
            report.append(entry)
            continue
        units = split_units(str(item.get("chunk") or ""))
        keep = [u for u in units if not states_disputed_claim(conflict, u)]
        text = "\n".join(keep)
        if not keep or not _WORD.search(text) or conflict.matches(text):
            entry["action"] = "chunk_removed"
            report.append(entry)
            continue
        if len(keep) < len(units):
            screened = dict(item)
            screened["chunk"] = text
            screened["meta"] = dict(meta, conflict_screened=topic)
            kept.append(screened)
            entry.update(action="units_removed", units_removed=len(units) - len(keep))
        else:
            kept.append(item)
            entry["action"] = "kept"
        report.append(entry)
    return kept, report


def answer_problem(answer: str, stamped: Iterable[Tuple[Optional[str], str]],
                   conflicts: Optional[Mapping[str, Mapping[str, SourceConflict]]] = None) -> Optional[str]:
    """Why ``answer`` must not be served (it states the disputed claim of a value conflict stamped on the retrieved
    evidence), or None."""
    conflicts = registered_conflicts() if conflicts is None else conflicts
    units = split_units(answer)
    for sid, topic in stamped:
        conflict = _lookup(conflicts, sid, topic)
        if conflict is None or not is_value_conflict(conflict):
            continue                        # the whole chunk was removed from the evidence
        if any(answer_states_disputed_claim(conflict, unit) for unit in units):
            return "the answer states the disputed claim of source conflict %r" % topic
    return None


__all__ = ["CLAIM_WORDS", "answer_problem", "answer_states_disputed_claim", "is_value_conflict", "registered_conflicts", "screen_items", "split_units",
           "stamped_topics", "states_disputed_claim"]
