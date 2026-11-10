"""
Conditional exceptions a model answer leaves out (post-freeze PF-16.6, decision D-130).

The PF-16.5 live run answered "Can an assistant professor apply for promotion before completing five years in rank?" with
a flat "No" from p146 (five years of experience) although the SAME evidence held p152 B: "An assistant or associate
professor may apply for promotion to a higher rank four months before completing the term at the lower rank if other
conditions required for promotion to the higher rank have been fulfilled." The Verifier checks the claims an answer makes,
never the conditions it leaves out, so a conditional permission became an unconditional answer.

This guard is deterministic and reads only the evidence the answer was written from:

    exception sentence  a sentence of an evidence item that states an exception or a conditional permission: "except",
                        "unless", "provided that", "exceptional circumstances/cases", "special circumstances",
                        "extendable", "may be extended", "may apply ... before/after/earlier/prior to"
    applicable          the question asks yes/no or about eligibility or timing (a clause that starts with a modal or an
                        auxiliary, or a word such as "before", "early", "longer", "exceed", "allowed", "without")
    whole sentence      it ends with sentence punctuation and starts with a capital, a digit or a quote: a printed row or a
                        window that cuts a sentence is a fragment and is never used
    relevant            the sentence shares at least three content words with the question (or two, when they are at least
                        40 % of the question's content words)
    left out            fewer than half of the content words of the EXCEPTION SPAN (from the exception keyword to the end of
                        the sentence, e.g. "except for seed projects."), not counting words already in the question, are in
                        the answer - restating the rule before the exception does not count as stating the exception

``uncovered`` returns the relevant exception sentences an answer leaves out (at most two, most relevant first). The
orchestrator gives such an answer the existing single retry, quoting the sentences; an answer that still leaves one out
gets the sentence appended verbatim with its page - printed text, never a rewording. Nothing is decided by a model and no
threshold of the evidence gate or the Verifier is touched.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .agents.verifier import content_tokens

_EXCEPTION = re.compile(
    r"\b(?:except|unless|provided\s+that|exceptional\s+(?:circumstances|cases)|special\s+circumstances|extendable|"
    r"may\s+(?:also\s+)?be\s+extended|may\s+apply\b[^.;]{0,120}?\b(?:before|after|earlier|prior\s+to))\b", re.I)
_YES_NO = re.compile(r"(?:^|[,;?.:]\s*|\band\s+|\bor\s+)(?:can|could|may|is|are|am|do|does|did|must|should|will|would|has|have)\b", re.I)
_ELIGIBILITY = re.compile(r"\b(?:eligible|allowed|permitted|possible|before|early|earlier|longer|more\s+than|exceed\w*|extend\w*|"
                          r"without|exception\w*|exempt\w*)\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z•(\"“])|\s*•\s*")
_LEAD = re.compile(r"^(?:Page\s+\d+\s+)?(?:[A-Za-z]|\d{1,2}|[ivx]{1,4})[.)]\s+|^(?:Page\s+\d+\s+)|^[o•\-]\s+")
#: Words that say nothing about WHICH exception a sentence states.
_GENERIC = {"faculty", "member", "university", "uos", "sharjah", "may", "shall", "must", "can", "also", "approval", "approved"}
_COMPLETE = re.compile(r"[.!?;][\"”')]?$")
_STARTS = re.compile(r"[A-Z0-9\"“(]")
#: Two shared content words make a sentence relevant when they are at least this share of the question's content words.
RELEVANT_SHARE = 0.4
MAX_NOTES = 2
MIN_SENTENCE_CHARS = 30


def applicable(question: str) -> bool:
    """A yes/no, eligibility or timing question - where leaving out an exception turns a condition into a rule."""
    q = question or ""
    return bool(_YES_NO.search(q) or _ELIGIBILITY.search(q))


def _sentences(text: str) -> List[str]:
    flat = re.sub(r"\s*\|\s*", " ", text or "")
    flat = re.sub(r"\s+", " ", flat).strip()
    out = []
    for part in _SENTENCE_END.split(flat):
        part = _LEAD.sub("", (part or "").strip()).strip()
        # a whole sentence only: a printed row that stops mid-sentence ("... four months before") or a window that starts
        # inside one ("publications for promotion) may apply ...") is a fragment, and a fragment is never quoted
        if len(part) >= MIN_SENTENCE_CHARS and _COMPLETE.search(part) and _STARTS.match(part):
            out.append(part)
    return out


def exception_sentences(items: Sequence[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    """(sentence, page) of every exception sentence of the evidence items, first occurrence only."""
    seen: Dict[str, Tuple[str, Any]] = {}
    for item in items or []:
        page = (item.get("meta") or {}).get("page")
        for sentence in _sentences(str(item.get("chunk") or "")):
            if _EXCEPTION.search(sentence):
                key = " ".join(content_tokens(sentence))
                if key and key not in seen:
                    seen[key] = (sentence, page)
    return list(seen.values())


def _relevance(question_tokens: set, sentence: str) -> int:
    shared = question_tokens & set(content_tokens(sentence))
    if len(shared) >= 3 or (len(shared) >= 2 and question_tokens and len(shared) / len(question_tokens) >= RELEVANT_SHARE):
        return len(shared)
    return 0


def exception_span(sentence: str) -> str:
    """The exception itself: from its keyword to the end of the sentence ("except for seed projects."). The rule before it
    ("may not act as principal investigators for two projects simultaneously, ...") is not the exception."""
    m = _EXCEPTION.search(sentence or "")
    return sentence[m.start():] if m else (sentence or "")


def covered(question_tokens: set, sentence: str, answer: str) -> bool:
    """Whether the answer states the exception's own content: at least half of the words of its exception span that are
    not already in the question."""
    own = [t for t in content_tokens(exception_span(sentence)) if t not in question_tokens and t not in _GENERIC]
    if not own:
        return True
    answer_tokens = set(content_tokens(answer))
    return sum(1 for t in own if t in answer_tokens) / len(own) >= 0.5


def uncovered(question: str, answer: str, items: Sequence[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    """The relevant exception sentences of the evidence that ``answer`` leaves out, most relevant first (at most two)."""
    if not applicable(question) or not (answer or "").strip():
        return []
    q_tokens = set(content_tokens(question))
    ranked = []
    for sentence, page in exception_sentences(items):
        score = _relevance(q_tokens, sentence)
        if score and not covered(q_tokens, sentence, answer):
            ranked.append((score, sentence, page))
    ranked.sort(key=lambda r: -r[0])
    return [(s, p) for _score, s, p in ranked[:MAX_NOTES]]


def feedback(missed: Sequence[Tuple[str, Any]]) -> str:
    quoted = "; ".join("“%s” (page %s)" % (s, p) for s, p in missed)
    return ("The context also states: %s. If this condition or exception applies to the question, the answer must state it "
            "with its conditions; do not answer yes or no without it." % quoted)


def note(missed: Sequence[Tuple[str, Any]]) -> str:
    """The verbatim exception sentences with their pages, appended to an answer that still leaves them out."""
    return "\n\n".join("The approved document also states (page %s): “%s”" % (p, s) for s, p in missed)


__all__ = ["MAX_NOTES", "applicable", "covered", "exception_sentences", "feedback", "note", "uncovered"]
