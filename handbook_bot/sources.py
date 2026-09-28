"""
Source registry and section maps for the retrieval corpus.

WHY
    Retrieval was built for one document and its chunks carried a page
    number but no source identity and no usable section: the heading
    heuristic labelled almost every page "Page N". Future work needs to know,
    for every retrieved chunk, which document it came from, which chapter and
    section of that document, and to restrict a search to an approved subset
    of the corpus. This module adds that metadata without changing how text
    is loaded, chunked, embedded, ranked or cited.

WHAT
    * ``SourceRecord`` / ``SourceRegistry``: the documents the corpus may be
      built from, read from ``knowledge/sources.json`` (repository-relative
      paths only; every ``source_id`` is unique; unknown ids fail loudly).
    * ``SectionMap``: chapter and section page ranges of one document, read
      from a JSON file named by the source record, validated on load and
      resolved per page deterministically.
    * ``annotate_metadata``: attaches source and section keys to chunk
      metadata (see ``METADATA_KEYS``). Existing keys are never removed.

    A document that is not in the registry still gets a deterministic
    identity (``unregistered_...``) and no section data, so it can never be
    mistaken for an approved source.

PAGE CONVENTION
    Pages are physical, 1-based PDF page indexes, the same convention as
    ``meta["page"]`` produced by ``pdf_loader`` and ``chunking``.

This module imports only the standard library and ``config``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .config import KNOWLEDGE_DIR, SOURCES_FILE

#: Repository root: the parent of the ``handbook_bot`` package. Registry files
#: and source paths are resolved against it, whatever the working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Keys ``annotate_metadata`` adds to every chunk's metadata.
METADATA_KEYS: Tuple[str, ...] = (
    "source_id", "source_title", "source_type", "source_version",
    "chapter", "chapter_title", "section_no", "section_title", "page_section_nos",
)

RECORD_KINDS = ("front_matter", "chapter", "section")

_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9_]{2,63}$")
_SECTION_NO = re.compile(r"^\d+(?:\.\d+)*$")
_DRIVE = re.compile(r"^[A-Za-z]:")


class SourceRegistryError(ValueError):
    """The registry or a section map file is malformed."""


class UnknownSourceError(KeyError):
    """No source is registered under the requested id."""

    def __str__(self) -> str:                      # KeyError quotes its argument otherwise
        return str(self.args[0]) if self.args else ""


# ---------------------------------------------------------------------------
# Small validators
# ---------------------------------------------------------------------------
def _text(name: str, value: Any, *, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise SourceRegistryError("%s must be a non-empty string, got %r" % (name, value))
    return value


def _relative_path(name: str, value: Any) -> str:
    """A repository-relative path with forward slashes and no escape upwards."""
    path = _text(name, value)
    if os.path.isabs(path) or _DRIVE.match(path) or path.startswith(("/", "\\")):
        raise SourceRegistryError("%s must be repository-relative, got %r" % (name, path))
    if "\\" in path:
        raise SourceRegistryError("%s must use forward slashes, got %r" % (name, path))
    if any(part in ("..", "") for part in path.split("/")):
        raise SourceRegistryError("%s must not contain '..' or empty segments, got %r" % (name, path))
    return path


def _page(name: str, value: Any, page_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= page_count:
        raise SourceRegistryError("%s must be an integer page between 1 and %d, got %r" % (name, page_count, value))
    return value


# ---------------------------------------------------------------------------
# Source records and registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceRecord:
    """One document the corpus may be built from. Only ``relative_path`` is a
    location; it is resolved against ``REPO_ROOT``."""

    source_id: str
    title: str
    relative_path: str
    source_type: str
    version: str = ""
    authority: str = ""
    verification_status: str = ""
    section_map: Optional[str] = None      # repository-relative JSON file, or None
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _SOURCE_ID.match(self.source_id):
            raise SourceRegistryError(
                "source_id must be 3-64 lowercase letters, digits or underscores, got %r" % (self.source_id,))
        _text("title", self.title)
        _relative_path("relative_path", self.relative_path)
        _text("source_type", self.source_type)
        for name in ("version", "authority", "verification_status", "notes"):
            _text(name, getattr(self, name), required=False)
        if self.section_map is not None:
            _relative_path("section_map", self.section_map)

    @property
    def registered(self) -> bool:
        return not self.source_id.startswith("unregistered_")

    def absolute_path(self) -> str:
        return os.path.normpath(os.path.join(REPO_ROOT, self.relative_path))

    def exists(self) -> bool:
        return os.path.isfile(self.absolute_path())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id, "title": self.title, "relative_path": self.relative_path,
            "source_type": self.source_type, "version": self.version, "authority": self.authority,
            "verification_status": self.verification_status, "section_map": self.section_map,
            "notes": self.notes,
        }


def _same_file(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` name the same file. Relative paths are taken
    from the current working directory, as ``open`` would take them. When
    both exist the operating system decides (``os.path.samefile``), which
    covers Windows letter case and either separator; otherwise the two
    normalised absolute paths must be equal."""
    try:
        if os.path.exists(a) and os.path.exists(b):
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normcase(os.path.normpath(os.path.abspath(a))) == os.path.normcase(os.path.normpath(os.path.abspath(b)))


class SourceRegistry:
    """An explicit, ordered map from ``source_id`` to :class:`SourceRecord`."""

    def __init__(self, records: Iterable[SourceRecord], *, path: Optional[str] = None) -> None:
        self._records: List[SourceRecord] = []
        self.path = path
        seen = set()
        for record in records:
            if not isinstance(record, SourceRecord):
                raise SourceRegistryError("registry entries must be SourceRecord, got %s" % type(record).__name__)
            if record.source_id in seen:
                raise SourceRegistryError("duplicate source_id %r" % record.source_id)
            seen.add(record.source_id)
            self._records.append(record)

    def get_source(self, source_id: str) -> SourceRecord:
        for record in self._records:
            if record.source_id == source_id:
                return record
        raise UnknownSourceError("unknown source_id %r (registered: %s)"
                                 % (source_id, ", ".join(r.source_id for r in self._records) or "none"))

    def find_source_for_path(self, path: str) -> Optional[SourceRecord]:
        """The record whose registered file is ``path``, else None.

        Identity is strict: ``path`` must name the very file at the record's
        repository-relative location. Any spelling of that file matches
        (relative to the working directory or absolute, either separator,
        any letter case on Windows, redundant segments). A different file
        elsewhere with the same name never matches; it is unregistered.
        Identity is by location, not content.
        """
        if not isinstance(path, str) or not path:
            return None
        for record in self._records:
            if _same_file(path, record.absolute_path()):
                return record
        return None

    def list_sources(self) -> List[SourceRecord]:
        return list(self._records)

    def validate_source_registry(self) -> List[str]:
        """Problems with the registry as a list of messages; empty means valid.
        Checks that every file exists and every section map loads."""
        problems: List[str] = []
        for record in self._records:
            if not record.exists():
                problems.append("%s: file not found at %s" % (record.source_id, record.relative_path))
            if record.section_map:
                try:
                    section_map = load_section_map(os.path.join(REPO_ROOT, record.section_map))
                except (OSError, ValueError) as exc:
                    problems.append("%s: section map %s failed to load: %s" % (record.source_id, record.section_map, exc))
                else:
                    if section_map.source_id != record.source_id:
                        problems.append("%s: section map belongs to %s" % (record.source_id, section_map.source_id))
        return problems

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": 1, "sources": [r.to_dict() for r in self._records]}

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[SourceRecord]:
        return iter(self._records)

    def __contains__(self, source_id: object) -> bool:
        return any(r.source_id == source_id for r in self._records)


def default_registry_path() -> str:
    return os.path.join(REPO_ROOT, KNOWLEDGE_DIR, SOURCES_FILE)


def load_source_registry(path: Optional[str] = None) -> SourceRegistry:
    """Read ``knowledge/sources.json`` (or ``path``). Raises
    :class:`SourceRegistryError` on a malformed file and ``OSError`` when
    the file cannot be read."""
    path = path or default_registry_path()
    with open(path, "r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SourceRegistryError("%s is not valid JSON: %s" % (path, exc)) from None
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise SourceRegistryError("%s must be an object with a 'sources' list" % path)
    records = []
    for entry in payload["sources"]:
        if not isinstance(entry, dict):
            raise SourceRegistryError("every source entry must be an object")
        unknown = set(entry) - set(SourceRecord.__dataclass_fields__)
        if unknown:
            raise SourceRegistryError("unknown source field(s): %s" % ", ".join(sorted(unknown)))
        try:
            records.append(SourceRecord(**entry))
        except TypeError as exc:                   # a required field is missing
            raise SourceRegistryError("source entry %r: %s" % (entry.get("source_id"), exc)) from None
    return SourceRegistry(records, path=path)


def unregistered_source(path: str) -> SourceRecord:
    """A deterministic identity for a document that is not in the registry.
    It carries no section map and its id starts with ``unregistered_``, so a
    scope restricted to approved sources never admits it."""
    name = os.path.basename(path or "") or "document"
    slug = re.sub(r"[^a-z0-9]+", "_", os.path.splitext(name)[0].lower()).strip("_")[:48] or "document"
    return SourceRecord(
        source_id="unregistered_" + slug,
        title=name,
        relative_path=name,
        source_type=(os.path.splitext(name)[1].lstrip(".").lower() or "unknown"),
        verification_status="unregistered",
        notes="document not listed in the source registry",
    )


# ---------------------------------------------------------------------------
# Section map
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionRecord:
    """One page range of a document: a front-matter part, a chapter or a
    section. ``end_page`` is inclusive. A section that starts on the page
    where the previous one ends shares that boundary page."""

    kind: str
    chapter: Optional[int]
    section_no: Optional[str]
    section_title: Optional[str]
    start_page: int
    end_page: int
    note: str = ""

    def contains(self, page: int) -> bool:
        return self.start_page <= page <= self.end_page

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind, "chapter": self.chapter, "section_no": self.section_no,
            "section_title": self.section_title, "start_page": self.start_page, "end_page": self.end_page,
        }
        if self.note:
            out["note"] = self.note
        return out


@dataclass(frozen=True)
class PageSections:
    """What a section map says about one page."""

    page: int
    kind: str                              # "section" | "chapter" | "front_matter" | "unmapped"
    chapter: Optional[int]
    chapter_title: Optional[str]
    section_no: Optional[str]              # the page's primary section (see SectionMap.resolve)
    section_title: Optional[str]
    section_nos: Tuple[str, ...] = ()      # every section present on the page, document order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page, "kind": self.kind, "chapter": self.chapter, "chapter_title": self.chapter_title,
            "section_no": self.section_no, "section_title": self.section_title, "section_nos": list(self.section_nos),
        }


class SectionMap:
    """Chapter and section page ranges of one document, validated on creation.

    Resolution rule for a page (deterministic):
      * ``chapter`` is the chapter whose range contains the page.
      * ``section_nos`` are all sections whose range contains the page.
      * the primary section is the first section (document order) that
        STARTS on the page; if none starts there, the last section that was
        already in progress. A page inside a chapter but before its first
        section is labelled with the chapter itself; a front-matter page
        with its front-matter part; anything else is ``unmapped``.
    """

    def __init__(self, source_id: str, page_count: int, records: Iterable[SectionRecord],
                 *, page_numbering: str = "", derived_from: str = "") -> None:
        self.source_id = source_id
        self.page_count = page_count
        self.records: Tuple[SectionRecord, ...] = tuple(records)
        self.page_numbering = page_numbering
        self.derived_from = derived_from
        self.validate()

    # ---- validation --------------------------------------------------------
    def validate(self) -> None:
        if not isinstance(self.source_id, str) or not _SOURCE_ID.match(self.source_id):
            raise SourceRegistryError("section map source_id %r is not a valid source id" % (self.source_id,))
        if isinstance(self.page_count, bool) or not isinstance(self.page_count, int) or self.page_count < 1:
            raise SourceRegistryError("page_count must be a positive integer")
        if not self.records:
            raise SourceRegistryError("section map has no records")
        for rec in self.records:
            if rec.kind not in RECORD_KINDS:
                raise SourceRegistryError("record kind must be one of %s, got %r" % (RECORD_KINDS, rec.kind))
            _page("start_page", rec.start_page, self.page_count)
            _page("end_page", rec.end_page, self.page_count)
            if rec.end_page < rec.start_page:
                raise SourceRegistryError("record %r ends before it starts" % (rec.section_no or rec.section_title,))
            if rec.kind == "front_matter":
                if rec.chapter is not None or rec.section_no is not None:
                    raise SourceRegistryError("front matter carries no chapter or section number")
            else:
                if isinstance(rec.chapter, bool) or not isinstance(rec.chapter, int) or rec.chapter < 1:
                    raise SourceRegistryError("record %r needs a positive chapter number" % (rec.section_no,))
                if not isinstance(rec.section_no, str) or not _SECTION_NO.match(rec.section_no):
                    raise SourceRegistryError("section_no must look like 1 or 1.2, got %r" % (rec.section_no,))
                if not isinstance(rec.section_title, str) or not rec.section_title.strip():
                    raise SourceRegistryError("record %r needs a title" % (rec.section_no,))
        chapters = self.chapters()
        for a, b in zip(chapters, chapters[1:]):
            if b.start_page <= a.end_page:
                raise SourceRegistryError("chapters %s and %s overlap" % (a.chapter, b.chapter))
            if b.chapter <= a.chapter:
                raise SourceRegistryError("chapters must be in ascending order (%s before %s)" % (a.chapter, b.chapter))
        by_chapter = {c.chapter: c for c in chapters}
        for fm in self.front_matter():
            if any(c.start_page <= fm.end_page and fm.start_page <= c.end_page for c in chapters):
                raise SourceRegistryError("front matter %r overlaps a chapter" % (fm.section_title,))
        sections = self.sections()
        for rec in sections:
            chapter = by_chapter.get(rec.chapter)
            if chapter is None:
                raise SourceRegistryError("section %s refers to chapter %s, which has no record" % (rec.section_no, rec.chapter))
            if not (chapter.start_page <= rec.start_page and rec.end_page <= chapter.end_page):
                raise SourceRegistryError("section %s lies outside chapter %s" % (rec.section_no, rec.chapter))
        for a, b in zip(sections, sections[1:]):
            if b.start_page < a.start_page:
                raise SourceRegistryError("sections must be in page order (%s before %s)" % (a.section_no, b.section_no))
        # Two sections may share pages only when the later one starts on the
        # page where the earlier one ends: a shared boundary page.
        for i, a in enumerate(sections):
            for b in sections[i + 1:]:
                if b.start_page > a.end_page:
                    break
                if a.end_page != b.start_page:
                    raise SourceRegistryError("sections %s and %s overlap beyond a shared boundary page"
                                              % (a.section_no, b.section_no))

    # ---- views -------------------------------------------------------------
    def chapters(self) -> List[SectionRecord]:
        return sorted((r for r in self.records if r.kind == "chapter"), key=lambda r: r.start_page)

    def sections(self) -> List[SectionRecord]:
        return [r for r in self.records if r.kind == "section"]

    def front_matter(self) -> List[SectionRecord]:
        return sorted((r for r in self.records if r.kind == "front_matter"), key=lambda r: r.start_page)

    def unmapped_pages(self) -> List[int]:
        return [p for p in range(1, self.page_count + 1) if self.resolve(p).kind == "unmapped"]

    # ---- resolution --------------------------------------------------------
    def resolve(self, page: int) -> PageSections:
        if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= self.page_count:
            raise ValueError("page must be an integer between 1 and %d, got %r" % (self.page_count, page))
        chapter = next((c for c in self.chapters() if c.contains(page)), None)
        present = [s for s in self.sections() if s.contains(page)]
        if present:
            starting = [s for s in present if s.start_page == page]
            primary = starting[0] if starting else present[-1]
            return PageSections(page, "section", chapter.chapter if chapter else primary.chapter,
                                chapter.section_title if chapter else None,
                                primary.section_no, primary.section_title,
                                tuple(s.section_no for s in present))
        if chapter is not None:
            return PageSections(page, "chapter", chapter.chapter, chapter.section_title,
                                chapter.section_no, chapter.section_title, (chapter.section_no,))
        front = next((f for f in self.front_matter() if f.contains(page)), None)
        if front is not None:
            return PageSections(page, "front_matter", None, None, None, front.section_title, ())
        return PageSections(page, "unmapped", None, None, None, None, ())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id, "page_count": self.page_count, "page_numbering": self.page_numbering,
            "derived_from": self.derived_from, "records": [r.to_dict() for r in self.records],
        }


def load_section_map(path: str) -> SectionMap:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SourceRegistryError("%s is not valid JSON: %s" % (path, exc)) from None
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise SourceRegistryError("%s must be an object with a 'records' list" % path)
    records = []
    for entry in payload["records"]:
        if not isinstance(entry, dict):
            raise SourceRegistryError("every section record must be an object")
        unknown = set(entry) - set(SectionRecord.__dataclass_fields__)
        if unknown:
            raise SourceRegistryError("unknown section record field(s): %s" % ", ".join(sorted(unknown)))
        try:
            records.append(SectionRecord(**entry))
        except TypeError as exc:
            raise SourceRegistryError("section record %r: %s" % (entry.get("section_no"), exc)) from None
    return SectionMap(
        payload.get("source_id", ""), payload.get("page_count", 0), records,
        page_numbering=str(payload.get("page_numbering", "")), derived_from=str(payload.get("derived_from", "")),
    )


def check_section_map_pages(section_map: SectionMap, pages: Sequence[Dict[str, Any]]) -> Optional[str]:
    """None when ``pages`` (as returned by ``pdf_loader.load_pdf``) fit
    ``section_map``; otherwise a message saying why the map must not be
    applied. Both the number of loaded pages and the highest page index must
    equal the map's ``page_count``: a document of another length is another
    edition, and its chunks would receive wrong chapter and section labels.
    Length is a cheap edition check, not a content check.
    """
    numbers = [page.get("page") for page in pages]
    if not numbers or any(isinstance(n, bool) or not isinstance(n, int) for n in numbers):
        return "the loaded document carries no usable page numbers"
    loaded, highest = len(numbers), max(numbers)
    if loaded != section_map.page_count or highest != section_map.page_count:
        return ("the loaded document has %d pages (highest page index %d) but the section map for %s "
                "describes %d pages" % (loaded, highest, section_map.source_id, section_map.page_count))
    return None


def section_map_for(source: SourceRecord) -> Optional[SectionMap]:
    """The source's section map, or None when it declares none."""
    if not source.section_map:
        return None
    section_map = load_section_map(os.path.join(REPO_ROOT, source.section_map))
    if section_map.source_id != source.source_id:
        raise SourceRegistryError("section map %s belongs to %s, not %s"
                                  % (source.section_map, section_map.source_id, source.source_id))
    return section_map


# ---------------------------------------------------------------------------
# Chunk metadata annotation
# ---------------------------------------------------------------------------
def annotate_metadata(metadata: List[Dict[str, Any]], source: SourceRecord,
                      section_map: Optional[SectionMap] = None) -> List[Dict[str, Any]]:
    """Add the ``METADATA_KEYS`` to every chunk's metadata, in place.

    Existing keys (``page``, ``section``, ``chunk_type``, ``chunk_id``, ...)
    are left untouched. Section keys are None, and ``page_section_nos`` is
    an empty list, for a page the map does not cover or when there is no map.
    Returns the same list for convenience.
    """
    for meta in metadata:
        meta["source_id"] = source.source_id
        meta["source_title"] = source.title
        meta["source_type"] = source.source_type
        meta["source_version"] = source.version
        resolved: Optional[PageSections] = None
        page = meta.get("page")
        if section_map is not None and isinstance(page, int) and not isinstance(page, bool) \
                and 1 <= page <= section_map.page_count:
            resolved = section_map.resolve(page)
        meta["chapter"] = resolved.chapter if resolved else None
        meta["chapter_title"] = resolved.chapter_title if resolved else None
        meta["section_no"] = resolved.section_no if resolved else None
        meta["section_title"] = resolved.section_title if resolved else None
        meta["page_section_nos"] = list(resolved.section_nos) if resolved else []
    return metadata


__all__ = [
    "METADATA_KEYS", "REPO_ROOT", "PageSections", "SectionMap", "SectionRecord", "SourceRecord",
    "SourceRegistry", "SourceRegistryError", "UnknownSourceError", "annotate_metadata",
    "check_section_map_pages", "default_registry_path", "load_section_map", "load_source_registry",
    "section_map_for", "unregistered_source",
]
