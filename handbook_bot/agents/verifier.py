"""
Verifier agent.

DECISION OWNED
    Whether a drafted answer is sufficiently supported by the retrieved
    evidence, and which pages may be cited for it. Nothing else: the Verifier
    does not retrieve, does not rewrite the answer and does not decide whether
    a retry happens (the orchestrator does, from ``retry_feedback``).

HOW IT DECIDES (deterministic; no LLM call)
    1. The answer is split into claims, one per sentence or bullet.
    2. Every claim is checked against EVERY evidence item. The Part 1 check
       compared a generic answer with ``items[0]`` only, so an answer that was
       supported by the second or third retrieved chunk was rejected. Here the
       item that supports a claim can be any of them.
    3. A claim is supported by an item when the item contains the claim's
       structured tokens (phone numbers, e-mail addresses, numbers, month and
       weekday names) and enough of its content words. See ``_support``.
    4. The answer is accepted when every checkable claim is supported. A
       structured token that appears in no item makes its claim unsupported
       whatever else matches: a number the context does not contain is the
       most common hallucination in this system.
    5. Cited pages are the pages of the items that support at least one
       claim, strongest support first. Pages the writer claimed are kept only
       when they are among those pages. If no item supports the answer, no
       page is cited. Nothing here can cite a page the evidence does not
       contain, and there is no "first three pages" fallback.

RETRY
    A rejected first draft gets targeted ``retry_feedback``: which statement
    is unsupported, which value is missing from the context, or which cited
    page does not support the answer. The orchestrator issues at most one
    retry; on the retry this module returns ``retry_feedback=None``.

MULTI-PART QUESTIONS
    ``sub_questions`` is empty in Milestone 1. When the Planner (Milestone 2)
    supplies them, ``uncovered_subquestions`` lists the indexes whose content
    words do not appear in the answer at all. Nothing else changes.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set, Tuple

from ..text_utils import DAY_PATTERN, EMAIL_PATTERN, MONTH_PATTERN, PHONE_PATTERN, tokenize
from .types import EvidenceResult, SubQuestion, SynthesisResult, VerifyResult

#: The controlled refusal sentence (kept in sync with qa.REFUSAL without
#: importing qa, which would create an import cycle through the orchestrator).
REFUSAL = "I do not have this information."

#: A claim is supported by an item when at least this share of its content
#: words appears in the item ...
MIN_COVERAGE = 0.5
#: ... and at least this many distinct content words match. Two words is the
#: Part 1 bar; a single generic word in common is never enough.
MIN_MATCHED = 2
#: A longer claim may match a smaller share if the absolute overlap is large.
LONG_CLAIM_MATCHED = 4
LONG_CLAIM_COVERAGE = 0.35
#: Two matching word pairs (bigrams) are strong phrase evidence on their own.
MIN_BIGRAMS = 2
#: A claim whose numbers or dates all appear in the item needs less prose overlap.
STRUCTURED_COVERAGE = 0.25
#: A claim spread over several chunks: coverage against the union of all items.
UNION_COVERAGE = 0.75
UNION_MATCHED = 3
#: Contribution needed for an item to be cited for a union-supported claim.
UNION_ITEM_COVERAGE = 0.2
#: Never cite more pages than this; ranked by how many claims each supports.
MAX_CITED_PAGES = 3
#: retry feedback is copied into the retry prompt; keep it short and specific.
MAX_FEEDBACK_CHARS = 380
MAX_QUOTED_CLAIM = 90

# Words that carry no checkable content on top of text_utils.STOPWORDS.
_EXTRA_STOPWORDS = {
    "yes", "no", "not", "may", "must", "will", "shall", "also", "only", "all",
    "any", "each", "per", "such", "than", "then", "there", "these", "those",
    "was", "were", "been", "being", "has", "have", "had", "its", "his", "her",
    "other", "more", "most", "some", "both", "either", "neither", "within",
    "without", "upon", "via", "etc", "however", "therefore", "according",
    "information", "provided", "context", "based", "states", "mentioned",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*[;]\s+")
#: A full stop after one of these does not end a sentence ("Decision No. (3)").
_ABBREVIATIONS = {"no", "nos", "dr", "prof", "mr", "mrs", "ms", "e.g", "i.e", "vs",
                  "art", "sec", "para", "p", "pp", "fig", "vol", "ext", "tel", "st"}
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\(?\d{1,2}[.)]|[a-z][.)])\s+", re.I)
_NUMBER = re.compile(r"(?<![\w.])\d{1,4}(?:,\d{3})?(?![\w,])")
_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_DIGITS = re.compile(r"\D+")
#: "page 65", "pp. 3-4", "pages 3, 4 and 12": a page reference is a citation,
#: handled by the page rules, not a fact to look for in the chunk text.
_PAGE_REF_BEFORE = re.compile(r"(?:\bpages?|\bpp?\.)\s*(?:\d+\s*(?:,|-|to|and)\s*)*$", re.I)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _stem(token: str) -> str:
    """A deliberately small stemmer: plurals and common verb endings only, so
    'programs' matches 'program' and 'approved' matches 'approve'."""
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> List[str]:
    """Stemmed content words of ``text``, in order, without duplicates."""
    seen: List[str] = []
    for tok in tokenize(text):
        if tok in _EXTRA_STOPWORDS or tok.isdigit():
            continue
        stem = _stem(tok)
        if stem not in seen:
            seen.append(stem)
    return seen


def _bigrams(text: str) -> Set[Tuple[str, str]]:
    toks = [_stem(t) for t in tokenize(text) if t not in _EXTRA_STOPWORDS]
    return {(a, b) for a, b in zip(toks, toks[1:])}


def _phones(text: str) -> List[str]:
    """Digit strings of the phone numbers in ``text``. PHONE_PATTERN also
    matches date fragments such as "2025 (02", so a match counts as a phone
    only with at least 7 digits and a run of 3+ digits that is not a year."""
    out = []
    for raw in PHONE_PATTERN.findall(text):
        digits = _DIGITS.sub("", raw)
        runs = re.findall(r"\d{3,}", raw)
        if len(digits) < 7 or not any(not _YEAR.match(r) for r in runs):
            continue
        if digits not in out:
            out.append(digits)
    return out


def _numbers(text: str, phones: Sequence[str]) -> List[str]:
    """Standalone numbers of the text (not part of a phone number, not a
    year), commas removed."""
    joined_phones = " ".join(phones)
    out = []
    for m in _NUMBER.finditer(text):
        value = m.group(0).replace(",", "")
        if value in joined_phones and len(value) >= 4:
            continue
        if _YEAR.match(value):
            continue          # a year is checked as prose, not as a hard token
        if _PAGE_REF_BEFORE.search(text[:m.start()]):
            continue          # "(page 65)" is a citation, not a claim
        if value not in out:
            out.append(value)
    return out


def _near_digit(text: str, m: "re.Match[str]") -> bool:
    window = text[max(0, m.start() - 6):m.end() + 6]
    return bool(re.search(r"\d", window))


def _date_words(text: str, *, strict: bool = True) -> List[str]:
    """Month and weekday names as 3-letter keys. "may", "mar", "sat", "sun",
    "mon" and the other abbreviations are ordinary words too, so in a claim
    (``strict``) they count only when a digit is close by ("12 May", "Mon 25
    Aug"). On the evidence side every match is kept: an extra date word there
    can only make the evidence more permissive, never reject an answer."""
    words = []
    for pattern in (MONTH_PATTERN, DAY_PATTERN):
        for m in pattern.finditer(text):
            word = m.group(0).lower()
            if strict and (len(word) <= 3 or word == "may") and not _near_digit(text, m):
                continue
            words.append(word[:3])
    return list(dict.fromkeys(words))


def split_claims(answer: str) -> List[str]:
    """Sentences and bullets of the answer, list markers removed. A full stop
    that follows an abbreviation such as "No." or "Dr." is not a boundary."""
    claims: List[str] = []
    for line in re.split(r"\n+", answer or ""):
        line = _LIST_MARKER.sub("", line.strip())
        parts: List[str] = []
        for part in _SENTENCE_SPLIT.split(line):
            part = (part or "").strip()
            if not part:
                continue
            if parts and parts[-1].endswith("."):
                last_word = parts[-1].rstrip(".").rsplit(None, 1)[-1].lower().strip("(")
                if last_word in _ABBREVIATIONS:
                    parts[-1] = parts[-1] + " " + part
                    continue
            parts.append(part)
        for part in parts:
            part = part.strip(" \t\"'()")
            if len(part) >= 2:
                claims.append(part)
    return claims


def is_refusal(answer: str) -> bool:
    return (answer or "").strip().rstrip(".").lower() == REFUSAL.rstrip(".").lower()


# ---------------------------------------------------------------------------
# Evidence index
# ---------------------------------------------------------------------------
class _Item:
    """One evidence item with everything the support check needs precomputed."""

    __slots__ = ("index", "page", "text", "lower", "tokens", "bigrams", "phones",
                 "emails", "numbers", "dates", "score")

    def __init__(self, index: int, item: Dict) -> None:
        self.index = index
        self.page = int(item["meta"]["page"])
        self.text = str(item.get("chunk") or "")
        self.lower = self.text.lower()
        self.tokens: Set[str] = set(content_tokens(self.text))
        self.bigrams = _bigrams(self.text)
        self.phones = set(_phones(self.text))
        self.emails = {e.lower() for e in EMAIL_PATTERN.findall(self.text)}
        self.numbers = set(_numbers(self.text, list(self.phones)))
        self.dates = set(_date_words(self.text, strict=False))
        self.score = float(item.get("rerank_score", 0.0) or 0.0)


class _Claim:
    __slots__ = ("text", "tokens", "bigrams", "phones", "emails", "numbers", "dates")

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = content_tokens(text)
        self.bigrams = _bigrams(text)
        self.phones = _phones(text)
        self.emails = [e.lower() for e in EMAIL_PATTERN.findall(text)]
        self.numbers = _numbers(text, self.phones)
        self.dates = _date_words(text)

    @property
    def checkable(self) -> bool:
        return bool(self.tokens or self.phones or self.emails or self.numbers or self.dates)

    @property
    def structured(self) -> bool:
        return bool(self.phones or self.emails or self.numbers or self.dates)


# ---------------------------------------------------------------------------
# Support check
# ---------------------------------------------------------------------------
def _missing_structured(claim: _Claim, item: _Item, all_items: Sequence[_Item]) -> List[str]:
    """Structured tokens of the claim that the item does not contain."""
    missing: List[str] = []
    for phone in claim.phones:
        if not any(phone in p or p in phone for p in item.phones):
            missing.append(phone)
    for email in claim.emails:
        if email not in item.emails:
            missing.append(email)
    for number in claim.numbers:
        if number not in item.numbers:
            missing.append(number)
    for word in claim.dates:
        if word not in item.dates:
            missing.append(word)
    return missing


def _support(claim: _Claim, item: _Item, all_items: Sequence[_Item]) -> Tuple[bool, float]:
    """(supported, coverage) of one claim by one item."""
    if _missing_structured(claim, item, all_items):
        return False, 0.0
    n = len(claim.tokens)
    matched = sum(1 for t in claim.tokens if t in item.tokens)
    coverage = matched / n if n else 0.0
    if n == 0:
        # Nothing but structured tokens (e.g. "+971 6 5050000"): they all matched.
        return claim.structured, 1.0
    if claim.phones or claim.emails:
        # The contact detail is the claim; "The phone number is X" needs no
        # further prose overlap once X is in the item.
        return True, max(coverage, 1.0)
    if n <= 2 and matched == n:
        return True, coverage
    if matched >= MIN_MATCHED and coverage >= MIN_COVERAGE:
        return True, coverage
    if matched >= LONG_CLAIM_MATCHED and coverage >= LONG_CLAIM_COVERAGE:
        return True, coverage
    if len(claim.bigrams & item.bigrams) >= MIN_BIGRAMS:
        return True, max(coverage, MIN_COVERAGE)
    if claim.structured and coverage >= STRUCTURED_COVERAGE:
        return True, coverage
    return False, coverage


def _union_support(claim: _Claim, items: Sequence[_Item]) -> List[_Item]:
    """Items that together support a claim spread over several chunks."""
    if not items or not claim.tokens:
        return []
    union = set().union(*(item.tokens for item in items))
    matched = sum(1 for t in claim.tokens if t in union)
    if matched < UNION_MATCHED or matched / len(claim.tokens) < UNION_COVERAGE:
        return []
    # Structured tokens must still sit in a contributing item.
    contributing = []
    for item in items:
        cov = sum(1 for t in claim.tokens if t in item.tokens) / len(claim.tokens)
        if cov >= UNION_ITEM_COVERAGE and not _missing_structured(claim, item, items):
            contributing.append(item)
    if claim.structured and not contributing:
        return []
    return contributing


def _rank_pages(support_count: Dict[int, int], items: Sequence[_Item]) -> List[int]:
    best_score = {}
    for item in items:
        best_score[item.page] = max(best_score.get(item.page, float("-inf")), item.score)
    return sorted(support_count, key=lambda p: (-support_count[p], -best_score.get(p, 0.0), p))


def _quote(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= MAX_QUOTED_CLAIM else text[:MAX_QUOTED_CLAIM - 3].rstrip() + "..."


def _feedback(unsupported: List[Tuple[_Claim, List[str]]], dropped_pages: List[int],
              uncheckable_only: bool) -> str:
    parts: List[str] = []
    if uncheckable_only:
        parts.append("The answer states a conclusion without the rule that supports it. "
                     "Quote or restate the specific statement from the context that answers "
                     "the question.")
    for claim, missing in unsupported[:2]:
        if missing:
            parts.append('The statement "%s" uses the value %s, which does not appear in the '
                         'context.' % (_quote(claim.text), ", ".join(missing[:3])))
        else:
            parts.append('The statement "%s" is not supported by the retrieved context.'
                         % _quote(claim.text))
    if len(unsupported) > 2:
        parts.append("%d further statement(s) are unsupported." % (len(unsupported) - 2))
    if dropped_pages:
        parts.append("Page %s was cited but does not support the answer; cite only pages "
                     "whose text supports it." % ", ".join(str(p) for p in dropped_pages[:3]))
    if unsupported or uncheckable_only:
        parts.append("Use only facts and wording present in the context, or reply exactly: "
                     + REFUSAL)
    text = " ".join(parts)
    return text if len(text) <= MAX_FEEDBACK_CHARS else text[:MAX_FEEDBACK_CHARS - 3].rstrip() + "..."


def _uncovered(sub_questions: Sequence[SubQuestion], answer: str) -> List[int]:
    answer_tokens = set(content_tokens(answer))
    uncovered: List[int] = []
    for index, sq in enumerate(sub_questions):
        toks = content_tokens(getattr(sq, "question", "") or "")
        if not toks:
            continue
        needed = max(1, (len(toks) + 1) // 2)
        if sum(1 for t in toks if t in answer_tokens) < needed:
            uncovered.append(index)
    return uncovered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def verify(
    question: str,
    synthesis: SynthesisResult,
    evidence: List[EvidenceResult],
    sub_questions: List[SubQuestion],
    *,
    is_retry: bool = False,
) -> VerifyResult:
    """Check ``synthesis.answer`` against every retrieved item. See the module docstring.

    Called by the orchestrator exactly as ``verify(question, synthesis,
    evidence, sub_questions, is_retry=...)``. ``evidence`` holds one entry in
    Milestone 1; items of every entry are checked.
    """
    answer = (synthesis.answer or "").strip()
    raw_items = [item for ev in evidence for item in (ev.items or [])]
    items = [_Item(i, item) for i, item in enumerate(raw_items)]
    claimed = [int(p) for p in (synthesis.claimed_pages or [])]
    uncovered = _uncovered(sub_questions or [], answer)

    if not answer:
        return VerifyResult(False, [], ["(empty answer)"], uncovered,
                            None if is_retry else "The answer was empty. Answer from the "
                            "context, or reply exactly: " + REFUSAL)
    if is_refusal(answer):
        # A refusal makes no claim, so it cites nothing and is always acceptable.
        return VerifyResult(True, [], [], uncovered, None)
    if not items:
        return VerifyResult(False, [], [_quote(answer)], uncovered,
                            None if is_retry else "No evidence was retrieved for this "
                            "question. Reply exactly: " + REFUSAL)

    claims = [_Claim(text) for text in split_claims(answer)]
    checkable = [c for c in claims if c.checkable]
    support_count: Dict[int, int] = {}
    unsupported: List[Tuple[_Claim, List[str]]] = []

    for claim in checkable:
        supporters: List[_Item] = []
        for item in items:
            ok, _ = _support(claim, item, items)
            if ok:
                supporters.append(item)
        if not supporters:
            supporters = _union_support(claim, items)
        if supporters:
            for item in supporters:
                support_count[item.page] = support_count.get(item.page, 0) + 1
        else:
            # Report the value that no item contains, if that is the reason.
            missing_everywhere = [
                m for m in _missing_structured(claim, items[0], items)
                if all(m in _missing_structured(claim, item, items) for item in items)
            ]
            unsupported.append((claim, missing_everywhere))

    uncheckable_only = bool(claims) and not checkable
    accepted = bool(checkable) and not unsupported

    supporting_pages = _rank_pages(support_count, items)
    dropped_pages = [p for p in claimed if p not in supporting_pages]
    if accepted:
        # Claimed pages that really support the answer come first, then the
        # other supporting pages, strongest first.
        pages = [p for p in claimed if p in supporting_pages]
        pages += [p for p in supporting_pages if p not in pages]
        pages = pages[:MAX_CITED_PAGES]
    else:
        pages = []

    feedback = None
    if not accepted and not is_retry:
        feedback = _feedback(unsupported, dropped_pages, uncheckable_only)

    return VerifyResult(
        accepted=accepted,
        pages=pages,
        unsupported_claims=[_quote(c.text) for c, _ in unsupported] or
                           (["(no checkable statement)"] if uncheckable_only else []),
        uncovered_subquestions=uncovered,
        retry_feedback=feedback,
    )
