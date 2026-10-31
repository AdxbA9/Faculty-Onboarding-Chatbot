"""
Source scoping shared by the domain specialists (multi-agent architecture).

DECISION OWNED
    None. This module turns a specialist's owned section identities into a
    ``RetrievalScope`` and attributes the text of a shared page to the
    section it belongs to. Which sections a specialist owns is that
    specialist's decision and lives in its own module.

WHAT IS HERE
    * ``compile_scope``: owned (chapter, printed number, label) identities,
      plus optional explicit extra page ranges (a chapter preamble that no
      section record covers), to page ranges. It fails loudly when an owned
      section is absent from the map, when the map belongs to another
      source, or when an extra page would widen the scope into a section the
      specialist does not own.
    * ``heading_marker``: the regex for a section's printed heading.
    * ``SectionGuard``: chunk-level section eligibility on pages that carry a
      non-owned section (see the class docstring).
    * ``approved_source_set`` / ``approve_sources`` (post-freeze PF-3B,
      D-110): the source-acceptance boundary of a specialist. The section-
      mapped primary source (the handbook) is always approved; further
      sources are approved only when the runtime's source policy names them
      explicitly. ``approve_sources`` adds them to the scope as
      ``also_source_ids``: the handbook constraints stay exactly as compiled.

The code was extracted unchanged from the Teaching & Learning Specialist
(Milestone 3) so that the remaining specialists share one implementation.
No specialist imports another specialist; each imports this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from ...retrieval import RetrievalScope
from ...sources import SectionMap, SectionRecord
from .evidence import _spans

#: One owned section identity: (chapter, printed section number, label).
SectionIdentity = Tuple[int, str, str]


def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def compile_scope(section_map: SectionMap, source_id: str, owned: Sequence[SectionIdentity],
                  extra_pages: Sequence[Tuple[int, int]] = ()) -> RetrievalScope:
    """The retrieval scope for ``owned`` sections, as page ranges read from
    the section map, plus ``extra_pages`` (inclusive ``(first, last)``
    ranges for pages no section record covers, such as a chapter preamble).

    Raises ``ValueError`` when the map is not the source's, when an owned
    section is absent from the map (the scope is never silently smaller),
    when an extra page lies outside the document, or when an extra page
    carries a section record outside ``owned`` (the scope is never silently
    wider). A page inside a chapter but before its first section, or a
    front-matter page, is an admissible extra page.
    """
    if section_map.source_id != source_id:
        raise ValueError("section map belongs to %s, not %s" % (section_map.source_id, source_id))
    owned_keys = {(chapter, number) for chapter, number, _label in owned}
    ranges: List[Tuple[int, int]] = []
    absent: List[str] = []
    sections = section_map.sections()
    for chapter, number, _label in owned:
        records = [r for r in sections if r.chapter == chapter and r.section_no == number]
        if not records:
            absent.append("chapter %d section %s" % (chapter, number))
        ranges.extend((r.start_page, r.end_page) for r in records)
    if absent:
        raise ValueError("owned sections absent from the section map: " + ", ".join(absent))
    for first, last in extra_pages:
        if not (isinstance(first, int) and isinstance(last, int) and 1 <= first <= last <= section_map.page_count):
            raise ValueError("extra page range %r is not inside the document (1-%d)" % ((first, last), section_map.page_count))
        for page in range(first, last + 1):
            foreign = [r for r in sections if r.start_page <= page <= r.end_page and (r.chapter, r.section_no) not in owned_keys]
            if foreign:
                raise ValueError("extra page %d carries section %s, which is not owned"
                                 % (page, ", ".join("%d/%s" % (r.chapter, r.section_no) for r in foreign)))
        ranges.append((first, last))
    return RetrievalScope(source_ids=[source_id], page_ranges=_merge_ranges(ranges))


def approved_source_set(approved: Optional[Iterable[str]], primary_source_id: str) -> FrozenSet[str]:
    """The source ids a specialist accepts evidence from (post-freeze PF-3B, D-110). ``None`` means the primary
    (section-mapped) source only, exactly as at the I-9 freeze. Any other value must be a collection of non-empty
    source ids that includes the primary source; nothing is ever implied beyond the ids given (no wildcard)."""
    if approved is None:
        return frozenset({primary_source_id})
    if isinstance(approved, str):
        raise ValueError("approved_sources must be a collection of source ids, not a single string")
    ids = set()
    for source_id in approved:
        if not isinstance(source_id, str) or not source_id.strip() or source_id.strip() != source_id or "*" in source_id:
            raise ValueError("approved_sources must contain plain source ids, got %r" % (source_id,))
        ids.add(source_id)
    if primary_source_id not in ids:
        raise ValueError("approved_sources must include the primary source %r" % primary_source_id)
    return frozenset(ids)


def approve_sources(scope: RetrievalScope, approved: FrozenSet[str], primary_source_id: str) -> RetrievalScope:
    """``scope`` with every approved source other than the primary one admitted as ``also_source_ids`` (their chunks
    are not subject to the primary source's page ranges). The scope is returned unchanged when nothing beyond the
    primary source is approved, so the I-9 scope object is reproduced exactly."""
    extra = frozenset(approved) - {primary_source_id}
    if not extra:
        return scope
    return replace(scope, also_source_ids=frozenset(set(scope.also_source_ids or ()) | extra))


def heading_marker(record: SectionRecord) -> "re.Pattern[str]":
    """Regex for the printed heading of a section: its number followed by the
    first two title words, e.g. "3.5 Responsibility to"."""
    words = [re.sub(r"[^\w()/-]", "", w) for w in re.split(r"\s+", record.section_title or "") if w][:2]
    parts = [re.escape(record.section_no or "")] + [re.escape(w) for w in words if w]
    return re.compile(r"(?<![\d.])" + r"\W+".join(parts) + r"(?![\w])", re.I)


@dataclass
class _PageSegments:
    """Sections printed on one shared page, in document order, with the row
    index at which each begins (-1 for the section already in progress)."""

    page: int
    segments: List[Tuple[Tuple[int, str], int, bool]]          # ((chapter, number), first row id, owned)
    markers: List[Tuple["re.Pattern[str]", int]]               # heading regex -> segment index (starting sections)
    para_spans: Dict[int, List[Tuple[int, int, bool]]]         # chunk id -> [(start, end, owned)]

    def row_owned(self, row_id: int) -> bool:
        owned = self.segments[0][2]
        for _key, first_row, is_owned in self.segments:
            if first_row <= row_id:
                owned = is_owned
        return owned


class SectionGuard:
    """Chunk-level section eligibility on pages that carry a non-owned section.

    Built once from the chunks, their metadata and the section map. On a
    shared page the printed section headings are located: as row chunks (row
    ids) and inside paragraph chunks (character offsets). A row belongs to the
    last section whose heading precedes it; a row window is checked row by
    row; a paragraph window is split at the headings it contains. Inside a
    window, text after a heading belongs to that heading's section until the
    next heading, and text before the first heading belongs to the section
    that precedes that heading in the page's section order, never to a state
    remembered from an overlapping window. A window without any heading lies
    inside the section that the previous window ended in (the windows overlap
    by more than a heading marker, so a heading is never missed). Text of a
    non-owned section is not the specialist's evidence. A shared page whose
    headings cannot be located is rejected whole: false ``not_found`` is
    preferred to a supported claim from an excluded section. A page that no
    section record covers (a chapter preamble admitted as an extra page) is
    never shared and is admitted whole.

    Source-aware since post-freeze PF-3B (D-110): page and section
    attribution applies to the section map's own source only. A chunk of
    another approved source (a system guide, which has no section map) is
    never counted on a handbook page and is never shared: its page numbers
    are its own document's, so handbook page ranges and headings say nothing
    about it. For the handbook every result is exactly as before.
    """

    def __init__(self, chunks: Sequence[str], metadata: Sequence[Dict[str, Any]], section_map: SectionMap,
                 owned: Sequence[SectionIdentity], scope: RetrievalScope) -> None:
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata differ in length (%d vs %d)" % (len(chunks), len(metadata)))
        for i, meta in enumerate(metadata):                         # paragraph spans are keyed by list index
            cid = meta.get("chunk_id")
            if cid is not None and cid != i:
                raise ValueError("chunk metadata is out of order: metadata[%d]['chunk_id'] == %r" % (i, cid))
        self.owned = {(c, n) for c, n, _ in owned}
        self.source_id = section_map.source_id
        self.shared_pages: Dict[int, _PageSegments] = {}
        self.unresolved_pages: List[int] = []
        pages = sorted({p for first, last in (scope.page_ranges or ()) for p in range(first, last + 1)})
        sections = section_map.sections()
        by_page: Dict[int, List[int]] = {}
        for i, meta in enumerate(metadata):
            page = meta.get("page")
            if isinstance(page, int) and page in pages and self._mapped(meta):
                by_page.setdefault(page, []).append(i)
        for page in pages:
            on_page = [r for r in sections if r.start_page <= page <= r.end_page]
            if not on_page or all((r.chapter, r.section_no) in self.owned for r in on_page):
                continue                                            # nothing non-owned printed here
            if not by_page.get(page):
                continue                                            # no chunk carries this page: nothing to attribute
            built = self._build(page, on_page, by_page[page], chunks, metadata)
            if built is None:
                self.unresolved_pages.append(page)
            else:
                self.shared_pages[page] = built

    def _build(self, page: int, on_page: List[SectionRecord], ids: List[int], chunks: Sequence[str],
               metadata: Sequence[Dict[str, Any]]) -> Optional[_PageSegments]:
        segments: List[Tuple[Tuple[int, str], int, bool]] = []
        markers: List[Tuple["re.Pattern[str]", int]] = []
        rows = [(metadata[i].get("row_id"), chunks[i]) for i in ids if metadata[i].get("chunk_type") == "row"]
        for record in on_page:
            key = (record.chapter, record.section_no)
            owned = key in self.owned
            if record.start_page < page:
                segments.append((key, -1, owned))
                continue
            marker = heading_marker(record)
            hits = [rid for rid, text in rows if isinstance(rid, int) and marker.match(text)]
            if not hits:
                return None                                         # heading not printed as a row: unresolved
            segments.append((key, min(hits), owned))
            markers.append((marker, len(segments) - 1))
        starts = [first for _k, first, _o in segments]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            return None                                             # headings out of document order
        para_spans: Dict[int, List[Tuple[int, int, bool]]] = {}
        paragraphs = sorted(
            (i for i in ids if metadata[i].get("chunk_type") == "paragraph"),
            key=lambda i: (metadata[i]["chunk_id_on_page"] if isinstance(metadata[i].get("chunk_id_on_page"), int) else i, i))
        current = 0                                                 # section a window without headings lies in
        for i in paragraphs:                                        # paragraph windows are emitted in document order
            text = chunks[i]
            cuts: List[Tuple[int, int]] = []
            for marker, seg_index in markers:
                m = marker.search(text)
                if m:
                    cuts.append((m.start(), seg_index))
            cuts.sort()
            spans: List[Tuple[int, int, bool]] = []
            if not cuts:
                spans.append((0, len(text), segments[current][2]))
            else:
                first_pos, first_seg = cuts[0]
                if first_pos > 0:                                   # text before the first heading: the preceding section
                    spans.append((0, first_pos, segments[max(first_seg - 1, 0)][2]))
                for k, (pos, seg_index) in enumerate(cuts):
                    end = cuts[k + 1][0] if k + 1 < len(cuts) else len(text)
                    if end > pos:
                        spans.append((pos, end, segments[seg_index][2]))
                current = max(current, cuts[-1][1])
            para_spans[i] = spans
        return _PageSegments(page=page, segments=segments, markers=markers, para_spans=para_spans)

    def _mapped(self, meta: Dict[str, Any]) -> bool:
        """True when ``meta`` belongs to the section map's source (metadata without a source id, as in fixtures built
        before source annotation, is read as the mapped source, the I-9 behaviour)."""
        source_id = meta.get("source_id")
        return source_id is None or source_id == self.source_id

    def is_shared(self, meta: Dict[str, Any]) -> bool:
        if not self._mapped(meta):
            return False
        return meta.get("page") in self.shared_pages or meta.get("page") in self.unresolved_pages

    def owned_spans(self, chunk_id: Any, chunk: str, meta: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        """Character spans of ``chunk`` that are the specialist's, or None
        when the whole chunk is (page without a non-owned section, or a chunk
        of another approved source, which has no sections). An empty list
        means nothing in the chunk may be quoted."""
        if not self._mapped(meta):
            return None
        page = meta.get("page")
        if page in self.unresolved_pages:
            return []
        segs = self.shared_pages.get(page)
        if segs is None:
            return None
        kind = meta.get("chunk_type")
        if kind == "row":
            rid = meta.get("row_id")
            return [(0, len(chunk))] if isinstance(rid, int) and segs.row_owned(rid) else []
        if kind == "row_window":
            first, last = meta.get("row_start"), meta.get("row_end")
            units = _spans(chunk, "row_window")
            if not isinstance(first, int) or not isinstance(last, int) or len(units) != last - first + 1:
                return []                                           # misaligned window: reject
            return [(s, e) for (s, e), rid in zip(units, range(first, last + 1)) if segs.row_owned(rid)]
        if kind == "paragraph":
            spans = segs.para_spans.get(chunk_id)
            if spans is None:
                return []
            return [(s, e) for s, e, owned in spans if owned]
        return []


__all__ = ["SectionGuard", "SectionIdentity", "approve_sources", "approved_source_set", "compile_scope", "heading_marker"]
