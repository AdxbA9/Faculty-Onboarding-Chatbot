"""
Evidence units and quote quality shared by the domain specialists.

DECISION OWNED
    None. This module cuts a retrieved chunk into quotable units (printed
    rows, sentences, list items), decides whether a unit is usable evidence
    (no headings, lead-ins, dangling fragments, truncated windows), selects
    the best quotable unit for a clause, and measures which requested details
    ONE local unit of a quote covers. What counts as relevant, and which
    details a clause asks for, are the calling specialist's decisions and
    arrive as callables and ``Detail`` objects.

RULES CARRIED OVER UNCHANGED FROM THE TEACHING SPECIALIST (Milestone 3)
    * a list item is one unit, else a sentence; lettered items ("a.", "B.")
      after punctuation are items like numbered ones;
    * a sentence that opens with a context-dependent word is quoted with its
      antecedent (the list's lead-in or the preceding plain sentence, the
      item's own earlier sentence inside an item), never with a sibling item
      or a heading, and not at all without one;
    * a prose unit is extended by the next unit only when that unit is
      allowed, quotable, relevant and does not open a new item;
    * requested details are never combined across units.

KNOWN BOUNDED LIMITATION (Milestone 3 finding R-1)
    A paragraph window joins the printed rows of a page with spaces and keeps
    no row boundaries. A bullet whose last row ends without terminal
    punctuation, followed by a row that starts a new capitalised sentence,
    therefore forms one unit with that sentence. No safe automatic row break
    exists: the same shape is also an ordinary wrapped line. The limitation is
    documented here so that every specialist inherits it knowingly; the
    one-unit locality rule still applies inside the joined unit.
"""
from __future__ import annotations

import re
from typing import Callable, FrozenSet, List, Optional, Sequence, Tuple

from .details import Detail, _detail_key

#: A quote extended by its neighbouring unit may not exceed this many words.
MAX_QUOTE_WORDS = 90

#: Relevance of a text to the clause being answered: (shared anchor
#: concepts, shared vocabulary concepts). Supplied by the specialist.
Relevance = Callable[[str], Tuple[int, int]]
#: Whether a printed line is table data rather than prose. Supplied by the
#: specialist when it knows domain table shapes (see ``is_table_row``).
TableRow = Callable[[str], bool]

_ROW_BREAK = re.compile(r"\s+\|\s+")
#: Items: bullets, "o " items, numbered items ("2. "), lettered items ("b. ",
#: "B. ") after punctuation, and section numbers. A lettered marker opens an
#: item only when a capital follows it.
_ITEM_BREAK = re.compile(
    r"\s+(?=[•�▪]\s)|\s+(?=o\s+[\"“A-Z])|(?<=[.:;!?])\s+(?=\d{1,2}\.\s+[A-Z])|(?<=[.:;!?])\s+(?=[A-Za-z]\.\s+[A-Z])|"
    r"\s+(?=\d+(?:\.\d+){1,3}\s+[A-Z])|\s+(?=Chapter\s+\d+\.)|"
    r"(?<=^Page \d)\s+|(?<=^Page \d\d)\s+|(?<=^Page \d\d\d)\s+")
_ITEM_START = re.compile(r"^(?:[•�▪]\s+|o\s+(?=[\"“A-Z])|\d{1,2}\.\s+(?=[A-Z])|[A-Za-z]\.\s+(?=[A-Z]))")
#: A sentence ends at .!? before a capital, except after a list marker
#: ("2.", "b.", or "B." opening the piece) so the marker stays with its item.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b[a-z]\.)(?<!^[A-Z]\.)\s+(?=[A-Z\"“(\[\d])")
_BULLET = re.compile(r"^(?:[•�▪-]\s+|o\s+(?=[\"“A-Z]))")
_LETTER_MARKER = re.compile(r"^[A-Za-z]\.\s+(?=[A-Z])")
_LIST_MARKER = re.compile(r"^(?:[•�▪-]\s+|o\s+(?=[\"“A-Z])|\d{1,2}\.\s+|[a-z]\.\s+|[A-Z]\.\s+(?=[A-Z])|\d+(?:\.\d+){1,3}\s+)")
_HEADING = re.compile(r"^(?:chapter\s+\d+[.:]?|\d+(?:\.\d+){1,3}\.?)\s+[A-Za-z(][^.!?]{0,110}:?$", re.I)
_LIST_TITLE = re.compile(r"^\d{1,2}\.\s+[A-Z][^.!?]{0,60}:?$")
_TERMINAL = re.compile(r"[.!?:;]['\")\]]?$")
_ENDS_WITH_CELL = re.compile(r"\d\S*$")
_DIGIT = re.compile(r"\d")
_PRONOUN_START = re.compile(r"^(?:such|they|these|this|those|it|their|its|both)\b", re.I)
#: A unit that introduces a list and says so ("... admissible includes:", "the following conditions:").
_LEAD_IN = re.compile(r"\b(?:includ\w*|following|as\s+follows|compris\w*|consist\w*|encompass\w*|conditions|criteria|such\s+as)\b[^.!?]*:\s*$", re.I)


def _split(text: str, breaker: "re.Pattern[str]") -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in breaker.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _units(text: str, chunk_type: Optional[str]) -> List[Tuple[int, int, int]]:
    """Quotable units of a chunk as ``(start, end, item_start)``: the whole
    chunk for a row; the rows of a row window; the sentences of a paragraph,
    each knowing where its bullet or numbered item begins (``item_start`` is
    the unit's own start outside a list item)."""
    if chunk_type == "row":
        return [(0, len(text), 0)]
    if chunk_type == "row_window":
        return [(s, e, s) for s, e in _split(text, _ROW_BREAK)] or [(0, len(text), 0)]
    units: List[Tuple[int, int, int]] = []
    for start, end in _split(text, _ITEM_BREAK):
        piece = text[start:end]
        is_item = bool(_ITEM_START.match(piece))
        for a, b in _split(piece, _SENTENCE_BREAK):
            units.append((start + a, start + b, start if is_item else start + a))
    return units or [(0, len(text), 0)]


def _spans(text: str, chunk_type: Optional[str]) -> List[Tuple[int, int]]:
    """Character spans of the quotable units of a chunk (see ``_units``)."""
    return [(s, e) for s, e, _ in _units(text, chunk_type)]


def _local_units(quote: str) -> List[str]:
    """Evidence units for the completeness check: a list item whole, else a
    sentence. Every requested detail must sit in one such unit together
    with the subject."""
    units: List[str] = []
    for start, end in _split(quote, _ITEM_BREAK):
        piece = quote[start:end]
        if _ITEM_START.match(piece):
            units.append(piece)
        else:
            units.extend(piece[a:b] for a, b in _split(piece, _SENTENCE_BREAK))
    return [u.strip() for u in units if u.strip()] or [quote]


def is_table_row(text: str, shapes: Sequence[Callable[[str], bool]] = ()) -> bool:
    """A printed line that is table data rather than prose: one of the
    specialist's ``shapes`` (each a callable on the line), or at least three
    digit-bearing cells with a label, within a table-like length."""
    words = text.split()
    if not 2 <= len(words) <= 24:
        return False
    for shape in shapes:
        if shape(text):
            return True
    digit_cells = sum(1 for w in words if _DIGIT.search(w))
    alpha_words = sum(1 for w in words if len(w) >= 3 and w.isalpha())
    return digit_cells >= 3 and alpha_words >= 1


def quote_problem(quote: str, unit_kind: str, last_unit: bool = False,
                  table_row: Optional[TableRow] = None) -> Optional[str]:
    """Why ``quote`` is not usable evidence, or None. ``unit_kind`` is
    "prose" (a unit of a paragraph) or "line" (a printed row). ``table_row``
    decides whether a printed line is table data (default: the generic rule)."""
    q = quote.strip()
    words = q.split()
    if not words:
        return "empty"
    core = _BULLET.sub("", q).strip()
    if not core:
        return "empty"
    if _HEADING.match(core) or _LIST_TITLE.match(core):
        return "heading"
    if core.endswith(":"):
        return "lead-in"                                   # "...include the following:" states nothing itself
    if _LETTER_MARKER.sub("", core)[0].islower():           # "b. These ..." is an item, not a fragment
        return "dangling"
    if unit_kind == "line":
        return None if (table_row or is_table_row)(core) else "line fragment"
    if len(words) < 5:
        return "too short"
    if last_unit and not _TERMINAL.search(q) and not _ENDS_WITH_CELL.search(q):
        return "truncated"                                 # a window cut mid-sentence; table cells may end a unit
    return None


def unit_coverage(details: Sequence[Detail], relevance: Relevance, quote: str) -> FrozenSet[str]:
    """The largest set of requested details that ONE local unit of ``quote``
    contains while sharing a concept with the clause. Details found in
    different units are never combined: "part-time" in one sentence and
    "five hours" in another do not answer a part-time quantity."""
    best: FrozenSet[str] = frozenset()
    for unit in _local_units(quote):
        if relevance(unit)[1] < 1:
            continue
        stripped = _LIST_MARKER.sub("", unit)
        got = frozenset(_detail_key(d) for d in details if d.check(stripped))
        if len(got) > len(best):
            best = got
    return best


def detail_covered(detail: Detail, relevance: Relevance, quote: str) -> bool:
    """True when some local unit of ``quote`` contains the detail and shares
    a concept with the clause. Use ``unit_coverage`` for completeness: this
    looks at one detail at a time."""
    units = _local_units(quote)
    return _detail_key(detail) in frozenset().union(*[unit_coverage([detail], relevance, u) for u in units]) if units else False


def select_quote(text: str, chunk_type: Optional[str], relevance: Relevance,
                 allowed: Optional[Sequence[Tuple[int, int]]] = None,
                 details: Sequence[Detail] = (), *, max_words: int = MAX_QUOTE_WORDS,
                 table_row: Optional[TableRow] = None, join_lead_in: bool = False) -> Optional[Tuple[str, int, int]]:
    """The best quotable unit of ``text`` for the clause as ``(quote, shared
    anchors, shared vocabulary)``, or None when no unit is both relevant and
    quotable. ``allowed`` restricts units to character spans attributed to
    the specialist (None = whole chunk). A sentence inside a list item is
    quoted with the item's opening; a sentence that opens with a
    context-dependent word (Such, They, These, This, Those, It) is quoted
    with the sentence before it when that sentence is allowed and usable,
    and is rejected otherwise; a prose unit is extended by the next unit
    when that one is allowed, quotable, shares a concept and does not open a
    new item. The quote is an exact substring of ``text``. Among relevant
    units, one that contains more of the requested ``details`` wins, then
    relevance, then document order.

    ``join_lead_in`` (opt-in, default off): a list item that directly
    follows the unit introducing its list ("For promotion purposes, research
    considered admissible includes:") is quoted together with that lead-in
    when the lead-in makes the quote more relevant, so the extract carries
    its own context. Only the first item is ever joined (the quote stays one
    contiguous slice), a sibling item is never joined, and a bare heading
    ending in a colon is not a lead-in. Details are still counted per local
    unit: the lead-in and the item are separate units of the quote."""
    units = _units(text, chunk_type)
    kind = "line" if chunk_type in ("row", "row_window") else "prose"

    def permitted(span: Tuple[int, int]) -> bool:
        return allowed is None or any(s <= span[0] and span[1] <= e for s, e in allowed)

    def bad(index: int) -> Optional[str]:
        s, e, _ = units[index]
        last = kind == "prose" and chunk_type == "paragraph" and index == len(units) - 1
        return quote_problem(text[s:e].strip(), kind, last_unit=last, table_row=table_row)

    def is_item_head(index: int) -> bool:
        s, e, item_start = units[index]
        return item_start == s and bool(_ITEM_START.match(text[s:e]))

    best: Optional[Tuple[Tuple[int, int, int, int], str]] = None
    for index, (start, end, item_start) in enumerate(units):
        if not permitted((start, end)) or bad(index) or relevance(text[start:end])[1] < 1:
            continue                                                # a unit must share a concept itself
        core = _LIST_MARKER.sub("", text[start:end].strip())
        if kind == "prose" and _PRONOUN_START.match(core):
            # The sentence depends on its antecedent: quote it with the unit
            # before it, or not at all. A list item's antecedent is the list's
            # lead-in or a plain sentence, never a sibling item; a sentence
            # inside an item takes the item's own earlier sentence.
            if index == 0:
                continue
            p_start, p_end, p_item = units[index - 1]
            prev = text[p_start:p_end].strip()
            prev_is_item = p_item == p_start and bool(_ITEM_START.match(prev))
            if item_start == start:
                usable = p_item == p_start and not prev_is_item
            else:
                usable = p_item == item_start
            prev_core = _LIST_MARKER.sub("", prev)
            if not (usable and permitted((p_start, end)) and prev_core and not prev_core[0].islower()
                    and not _HEADING.match(prev) and len(text[p_start:end].split()) <= max_words):
                continue                                            # a heading is never an antecedent
            start = p_start
        elif join_lead_in and kind == "prose" and index > 0 and is_item_head(index):
            p_start, p_end, _p_item = units[index - 1]
            prev = text[p_start:p_end].strip()
            is_lead_in = prev.endswith(":") and (not (_HEADING.match(prev) or _LIST_TITLE.match(prev)) or bool(_LEAD_IN.search(prev)))
            if (is_lead_in and permitted((p_start, end)) and len(text[p_start:end].split()) <= max_words
                    and relevance(text[p_start:end]) > relevance(text[start:end])):
                start = p_start
        # A sentence inside a bullet or numbered item is quoted with the item's
        # opening, so "The load for this category is 12 hours" keeps its category.
        if item_start < start and permitted((item_start, end)) and len(text[item_start:end].split()) <= max_words:
            start = item_start
        # A prose unit is extended by the next unit when that one is allowed,
        # quotable and shares a concept. Printed lines are never joined.
        if kind == "prose" and index + 1 < len(units) and not is_item_head(index + 1) and permitted(units[index + 1][:2]) and not bad(index + 1):
            nxt_end = units[index + 1][1]
            if relevance(text[units[index + 1][0]:nxt_end])[1] >= 1 and len(text[start:nxt_end].split()) <= max_words:
                end = nxt_end
        quote = text[start:end].strip()
        if quote_problem(quote, kind, last_unit=(kind == "prose" and chunk_type == "paragraph" and end == units[-1][1]), table_row=table_row):
            continue
        anchors, vocab = relevance(quote)
        if anchors >= 1 or vocab >= 2:
            covered = len(unit_coverage(details, relevance, quote))
            score = (covered, anchors, vocab, -index)
            if best is None or score > best[0]:
                best = (score, quote)
    if best is None:
        return None
    (_, anchors, vocab, _), quote = best
    return quote, anchors, vocab


__all__ = ["MAX_QUOTE_WORDS", "Relevance", "TableRow", "detail_covered", "is_table_row", "quote_problem", "select_quote", "unit_coverage"]
