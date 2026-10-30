"""Source registry, section map and chunk metadata for the retrieval corpus.

Everything here runs offline. The tests that read the real handbook load and
chunk the local PDF (no embedding model, no network) and validate the section
map against the document's own table of contents and headings, not against
the JSON file itself.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from handbook_bot import sources
from handbook_bot.retrieval import RetrievalScope
from handbook_bot.sources import (
    METADATA_KEYS,
    SectionMap,
    SectionRecord,
    SourceRecord,
    SourceRegistry,
    SourceRegistryError,
    UnknownSourceError,
    annotate_metadata,
    check_section_map_pages,
    default_registry_path,
    load_section_map,
    load_source_registry,
    section_map_for,
    unregistered_source,
)

REPO = Path(__file__).resolve().parents[1]
HANDBOOK_ID = "uos_faculty_handbook_2025_26"
PAGE_COUNT = 272


# ---------------------------------------------------------------------------
# Fixtures (module scope: the PDF is read once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry():
    return load_source_registry(str(REPO / "knowledge" / "sources.json"))


@pytest.fixture(scope="module")
def handbook(registry):
    return registry.get_source(HANDBOOK_ID)


@pytest.fixture(scope="module")
def section_map(handbook):
    return section_map_for(handbook)


@pytest.fixture(scope="module")
def handbook_pages(handbook):
    from handbook_bot.pdf_loader import load_pdf

    return load_pdf(handbook.absolute_path())


@pytest.fixture(scope="module")
def page_text(handbook_pages):
    return {p["page"]: " ".join(p["lines"]) for p in handbook_pages}


@pytest.fixture(scope="module")
def handbook_chunks(handbook_pages, handbook, section_map):
    from handbook_bot.chunking import build_chunks

    chunks, metadata = build_chunks(handbook_pages)
    annotate_metadata(metadata, handbook, section_map)
    return chunks, metadata


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
def test_registry_loads_from_the_default_location(registry):
    assert os.path.isfile(default_registry_path())
    assert load_source_registry().to_dict() == registry.to_dict()


def test_handbook_source_exists_with_a_stable_id(registry, handbook):
    assert HANDBOOK_ID in registry
    assert handbook.source_id == HANDBOOK_ID
    assert handbook.title == "UOS Faculty Handbook 2025/26"
    assert handbook.source_type == "pdf"
    assert handbook.version == "2025/2026"
    assert handbook.registered


def test_handbook_path_is_repository_relative_and_the_file_exists(handbook):
    assert not os.path.isabs(handbook.relative_path)
    assert not re.match(r"^[A-Za-z]:", handbook.relative_path)
    assert "\\" not in handbook.relative_path and ".." not in handbook.relative_path
    assert handbook.relative_path == "data/UOS Faculty Handbook 25-26.pdf"
    assert handbook.exists()
    assert handbook.absolute_path() == os.path.normpath(str(REPO / "data" / "UOS Faculty Handbook 25-26.pdf"))


def test_find_source_for_path_accepts_any_spelling_of_the_same_file(registry, monkeypatch):
    absolute = str(REPO / "data" / "UOS Faculty Handbook 25-26.pdf")
    assert registry.find_source_for_path(absolute).source_id == HANDBOOK_ID
    assert registry.find_source_for_path(absolute.replace("/", os.sep)).source_id == HANDBOOK_ID
    monkeypatch.chdir(REPO)
    assert registry.find_source_for_path(os.path.join("data", "UOS Faculty Handbook 25-26.pdf")).source_id == HANDBOOK_ID
    assert registry.find_source_for_path("data/other.pdf") is None
    assert registry.find_source_for_path("") is None


def test_unknown_source_id_is_rejected_clearly(registry):
    with pytest.raises(UnknownSourceError) as exc:
        registry.get_source("blackboard_guide")
    assert "blackboard_guide" in str(exc.value) and HANDBOOK_ID in str(exc.value)
    assert "blackboard_guide" not in registry


def test_duplicate_source_ids_are_rejected(tmp_path):
    record = {"source_id": "dup_source", "title": "A", "relative_path": "data/a.pdf", "source_type": "pdf"}
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": [record, dict(record, title="B")]}), encoding="utf-8")
    with pytest.raises(SourceRegistryError, match="duplicate"):
        load_source_registry(str(path))
    with pytest.raises(SourceRegistryError, match="duplicate"):
        SourceRegistry([SourceRecord(**record), SourceRecord(**record)])


@pytest.mark.parametrize("bad", [
    {"relative_path": "C:/Users/someone/handbook.pdf"},
    {"relative_path": "/srv/handbook.pdf"},
    {"relative_path": "data\\handbook.pdf"},
    {"relative_path": "../outside.pdf"},
    {"source_id": "Bad Id"},
    {"title": ""},
])
def test_malformed_source_records_are_rejected(bad):
    record = {"source_id": "good_source", "title": "Good", "relative_path": "data/good.pdf", "source_type": "pdf"}
    record.update(bad)
    with pytest.raises(SourceRegistryError):
        SourceRecord(**record)


def test_missing_and_unknown_fields_are_rejected(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": [{"source_id": "x_source", "title": "X"}]}), encoding="utf-8")
    with pytest.raises(SourceRegistryError):
        load_source_registry(str(path))
    path.write_text(json.dumps({"sources": [{"source_id": "x_source", "title": "X", "relative_path": "data/x.pdf",
                                             "source_type": "pdf", "url": "http://example"}]}), encoding="utf-8")
    with pytest.raises(SourceRegistryError, match="unknown source field"):
        load_source_registry(str(path))


def test_registry_loading_is_deterministic(registry):
    again = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    assert [r.to_dict() for r in again.list_sources()] == [r.to_dict() for r in registry.list_sources()]
    assert list(again) == list(registry)
    assert len(registry) == 1


def test_registry_validation_finds_no_problem_locally(registry):
    assert registry.validate_source_registry() == []


def test_registry_validation_reports_a_missing_file(tmp_path):
    reg = SourceRegistry([SourceRecord("ghost_source", "Ghost", "data/ghost.pdf", "pdf")])
    problems = reg.validate_source_registry()
    assert len(problems) == 1 and "ghost_source" in problems[0] and "not found" in problems[0]


def test_unregistered_source_identity_is_deterministic_and_never_registered():
    a = unregistered_source(r"C:\somewhere\My Other Guide (v2).pdf")
    b = unregistered_source("/tmp/My Other Guide (v2).pdf")
    assert a == b
    assert a.source_id == "unregistered_my_other_guide_v2"
    assert not a.registered and a.section_map is None and a.source_type == "pdf"
    assert a.relative_path == "My Other Guide (v2).pdf"      # never a machine path


# ---------------------------------------------------------------------------
# Section map: structure
# ---------------------------------------------------------------------------
def test_section_map_loads_and_belongs_to_the_handbook(section_map, handbook):
    assert section_map.source_id == HANDBOOK_ID
    assert section_map.page_count == PAGE_COUNT
    assert handbook.section_map == "knowledge/handbook_sections.json"
    assert len(section_map.chapters()) == 16
    assert len(section_map.sections()) == 186
    assert len(section_map.front_matter()) == 4
    assert len(section_map.records) == 206


def test_every_page_resolves_and_none_is_unmapped(section_map):
    for page in range(1, PAGE_COUNT + 1):
        resolved = section_map.resolve(page)
        assert resolved.page == page
        assert resolved.kind in ("section", "chapter", "front_matter")
    assert section_map.unmapped_pages() == []


def test_front_matter_is_pages_1_to_13_without_chapter_or_section(section_map):
    for page in range(1, 14):
        resolved = section_map.resolve(page)
        assert resolved.kind == "front_matter"
        assert resolved.chapter is None and resolved.section_no is None and resolved.section_nos == ()
    assert section_map.resolve(2).section_title == "Preface"
    assert section_map.resolve(3).section_title == "Important Notice"
    assert section_map.resolve(1).section_title is None            # the cover has no document heading
    assert section_map.resolve(4).section_title is None            # the contents pages carry no heading


def test_chapters_are_contiguous_ascending_and_cover_pages_14_to_272(section_map):
    chapters = section_map.chapters()
    assert [c.chapter for c in chapters] == list(range(1, 17))
    assert chapters[0].start_page == 14 and chapters[-1].end_page == PAGE_COUNT
    for a, b in zip(chapters, chapters[1:]):
        assert b.start_page == a.end_page + 1


def test_sections_stay_inside_their_chapter_and_share_only_boundary_pages(section_map):
    by_chapter = {c.chapter: c for c in section_map.chapters()}
    sections = section_map.sections()
    for rec in sections:
        chapter = by_chapter[rec.chapter]
        assert chapter.start_page <= rec.start_page <= rec.end_page <= chapter.end_page
        assert 1 <= rec.start_page <= PAGE_COUNT and 1 <= rec.end_page <= PAGE_COUNT
    for a, b in zip(sections, sections[1:]):
        assert a.start_page <= b.start_page
    for i, a in enumerate(sections):
        for b in sections[i + 1:]:
            if b.start_page > a.end_page:
                break
            assert a.end_page == b.start_page, (a.section_no, b.section_no)


def test_page_resolution_is_deterministic(section_map):
    first = [section_map.resolve(p).to_dict() for p in range(1, PAGE_COUNT + 1)]
    again = load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))
    second = [again.resolve(p).to_dict() for p in range(1, PAGE_COUNT + 1)]
    assert first == second


def test_page_out_of_range_is_rejected(section_map):
    for bad in (0, PAGE_COUNT + 1, -3, True, "49"):
        with pytest.raises(ValueError):
            section_map.resolve(bad)


@pytest.mark.parametrize("records,message", [
    ([SectionRecord("chapter", 1, "1", "A", 1, 10), SectionRecord("chapter", 2, "2", "B", 10, 20)], "overlap"),
    ([SectionRecord("chapter", 2, "2", "B", 1, 10), SectionRecord("chapter", 1, "1", "A", 11, 20)], "ascending"),
    ([SectionRecord("chapter", 1, "1", "A", 1, 10), SectionRecord("section", 1, "1.1", "S", 5, 12)], "outside chapter"),
    ([SectionRecord("chapter", 1, "1", "A", 1, 10), SectionRecord("section", 2, "2.1", "S", 2, 3)], "no record"),
    ([SectionRecord("chapter", 1, "1", "A", 1, 10), SectionRecord("section", 1, "1.1", "S", 2, 6),
      SectionRecord("section", 1, "1.2", "T", 4, 8)], "beyond a shared boundary"),
    ([SectionRecord("chapter", 1, "1", "A", 1, 30)], "between 1 and 20"),
    ([SectionRecord("front_matter", 1, None, "Preface", 1, 1)], "front matter"),
    ([SectionRecord("section", 1, "one", "S", 2, 3), SectionRecord("chapter", 1, "1", "A", 1, 10)], "look like"),
])
def test_invalid_section_maps_are_rejected(records, message):
    with pytest.raises(SourceRegistryError, match=message):
        SectionMap("some_source", 20, records)


def test_shared_boundary_page_carries_both_sections_and_a_deterministic_primary():
    records = [
        SectionRecord("chapter", 1, "1", "A", 1, 10),
        SectionRecord("section", 1, "1.1", "First", 2, 5),
        SectionRecord("section", 1, "1.2", "Second", 5, 5),
        SectionRecord("section", 1, "1.3", "Third", 5, 8),
        SectionRecord("section", 1, "1.4", "Fourth", 8, 10),
    ]
    section_map = SectionMap("some_source", 10, records)
    assert section_map.resolve(1).kind == "chapter" and section_map.resolve(1).section_nos == ("1",)
    assert section_map.resolve(3).section_no == "1.1" and section_map.resolve(3).section_nos == ("1.1",)
    boundary = section_map.resolve(5)
    assert boundary.section_no == "1.2"                          # first section that starts on the page
    assert boundary.section_nos == ("1.1", "1.2", "1.3")
    assert section_map.resolve(6).section_no == "1.3"            # the section in progress
    assert section_map.resolve(8).section_nos == ("1.3", "1.4") and section_map.resolve(8).section_no == "1.4"


# ---------------------------------------------------------------------------
# Section map: agreement with the real handbook
# ---------------------------------------------------------------------------
def _toc_entries(page_text):
    toc = " ".join(page_text[p] for p in range(4, 14))
    pattern = re.compile(r"(Chapter\s+\d+[:.]?|(?:\d{1,2}\.){1,4}\d{0,2})\s*([^.]{3,120}?)\s*\.{3,}\s*(\d{1,3})")
    return [(m.group(1).strip().rstrip("."), m.group(2).strip().rstrip(": ").strip(), int(m.group(3)))
            for m in pattern.finditer(toc)]


def test_every_section_record_appears_in_the_handbook_table_of_contents(section_map, page_text):
    """Each record's number, title and start page is an entry of the document's
    own contents pages. Keyed by the full triple: the handbook lists 5.2 twice
    on page 113 with different titles, and both records must be present."""
    toc = {(num, title.lower(), page) for num, title, page in _toc_entries(page_text)}
    for rec in section_map.sections():
        assert (rec.section_no, rec.section_title.lower(), rec.start_page) in toc, rec
    listed = {(num, page) for num, _, page in _toc_entries(page_text) if re.fullmatch(r"\d{1,2}\.\d{1,2}", num)}
    assert {(r.section_no, r.start_page) for r in section_map.sections()} == listed


def test_every_chapter_heading_is_on_its_start_page(section_map, page_text):
    for chapter in section_map.chapters():
        head = page_text[chapter.start_page][:200]
        assert re.search(r"Chapter\s+%d\b" % chapter.chapter, head), (chapter.chapter, head)
        assert chapter.section_title.lower() in head.lower(), (chapter.chapter, head)


@pytest.mark.parametrize("section_no,start_page,title_start", [
    ("1.14", 49, "Academic Calendar"),
    ("1.16", 52, "Contact Information"),
    ("2.4", 63, "Probation and Resignation"),
    ("3.11", 89, "Workload Allocation Model"),
    ("4.3", 107, "Leaves and Absences"),
    ("5.2", 113, "Categories of Faculty Development"),
    ("6.9", 128, "The Faculty Information System"),
    ("7.8", 148, "Promotion Procedures at the Department"),
    ("8.7", 160, "Research Grants"),
    ("9.7", 180, "Tangible Research Property"),
    ("10.6", 192, "Learning Management System"),
    ("11.7", 199, "Copyright Policy"),
    ("12.12", 222, "Examinations Policy"),
    ("13.9", 241, "Health Services Policy"),
    ("14.6", 245, "Disability Resource Center"),
    ("15.3", 254, "Facilities Management"),
    ("16.6", 268, "University of Sharjah Guidelines on Artificial"),
])
def test_spot_checked_headings_are_on_their_listed_pages(section_map, page_text, section_no, start_page, title_start):
    rec = next(r for r in section_map.sections() if r.section_no == section_no and r.start_page == start_page)
    assert rec.section_title.lower().startswith(title_start.lower())
    assert re.search(r"(?<![\d.])" + re.escape(section_no) + r"\s+" + re.escape(title_start[:18]),
                     page_text[start_page], re.I), (section_no, start_page)
    assert section_no in section_map.resolve(start_page).section_nos


def test_academic_calendar_1_14_spans_pages_49_to_51(section_map, page_text):
    assert section_map.resolve(49).section_no == "1.14"
    assert section_map.resolve(50).section_no == "1.14" and section_map.resolve(50).section_nos == ("1.14",)
    boundary = section_map.resolve(51)
    assert "1.14" in boundary.section_nos and boundary.section_no == "1.15"     # 1.15 starts on page 51
    assert "Classes begin" in page_text[49]
    assert section_map.resolve(49).chapter == 1 and section_map.resolve(49).chapter_title == "The University"


def test_examinations_policy_12_12_starts_on_page_222_beside_its_neighbours(section_map, page_text):
    resolved = section_map.resolve(222)
    assert resolved.chapter == 12
    assert resolved.section_no == "12.10"                          # first section that starts on the page
    assert ("12.10", "12.11", "12.12") == resolved.section_nos[-3:]
    assert "12.12 Examinations Policy" in page_text[222]


def test_document_numbering_quirks_are_recorded_not_repaired(section_map, page_text):
    """The handbook prints 5.2 and 7.8 twice and numbers the copyright policy
    in chapter 9 as 12.3. The map records what the document says and notes it;
    the chapter comes from the page range, so chapter scoping stays correct."""
    twice = [r for r in section_map.sections() if r.section_no in ("5.2", "7.8", "12.3")]
    assert [(r.section_no, r.start_page) for r in twice] == [("5.2", 113), ("5.2", 113), ("7.8", 148), ("7.8", 149),
                                                             ("12.3", 180), ("12.3", 212)]
    assert all("used twice" in r.note for r in twice)
    copyright_in_ip = next(r for r in twice if r.section_no == "12.3" and r.start_page == 180)
    assert copyright_in_ip.chapter == 9 and "located in chapter 9" in copyright_in_ip.note
    assert "12.3 Copyright" in page_text[180]
    assert section_map.resolve(180).chapter == 9


# ---------------------------------------------------------------------------
# Chunk metadata on the real handbook
# ---------------------------------------------------------------------------
def test_handbook_chunk_count_is_unchanged_by_annotation(handbook_chunks):
    chunks, metadata = handbook_chunks
    assert len(chunks) == len(metadata) == 13688


def test_every_handbook_chunk_carries_source_and_section_metadata(handbook_chunks):
    chunks, metadata = handbook_chunks
    for i, meta in enumerate(metadata):
        for key in METADATA_KEYS:
            assert key in meta, (i, key)
        assert meta["source_id"] == HANDBOOK_ID
        assert meta["source_title"] == "UOS Faculty Handbook 2025/26"
        assert meta["source_type"] == "pdf" and meta["source_version"] == "2025/2026"
        assert meta["chunk_id"] == i and 1 <= meta["page"] <= PAGE_COUNT
        assert isinstance(meta["page_section_nos"], list)
        if meta["chapter"] is None:
            assert meta["page"] <= 13 and meta["section_no"] is None and meta["page_section_nos"] == []
        else:
            assert meta["section_no"] is not None and meta["section_no"] in meta["page_section_nos"]


def test_old_chunk_metadata_keys_are_preserved(handbook_chunks):
    chunks, metadata = handbook_chunks
    for meta in metadata:
        for key in ("page", "section", "chunk_type", "chunk_id", "text"):
            assert key in meta
        if meta["chunk_type"] == "row":
            assert "row_id" in meta and "table_like" in meta
        if meta["chunk_type"] == "row_window":
            assert "row_start" in meta and "row_end" in meta
        assert meta["text"] == chunks[meta["chunk_id"]]


def test_representative_chunks_carry_the_expected_sections(handbook_chunks):
    chunks, metadata = handbook_chunks
    calendar = next(m for m in metadata if m["page"] == 49 and m["chunk_type"] == "paragraph")
    assert (calendar["chapter"], calendar["section_no"], calendar["section_title"]) == (1, "1.14", "Academic Calendar")
    assert calendar["chapter_title"] == "The University"
    exams = next(m for m in metadata if m["page"] == 222)
    assert exams["chapter"] == 12 and "12.12" in exams["page_section_nos"]
    preface = next(m for m in metadata if m["page"] == 2)
    assert preface["chapter"] is None and preface["section_title"] == "Preface" and preface["section_no"] is None
    intro = next(m for m in metadata if m["page"] == 14)
    assert intro["section_no"] == "1" and intro["section_title"] == "The University"


def test_annotation_without_a_registered_source_gives_no_section_data():
    metadata = [{"page": 49, "section": "Page 49", "chunk_type": "paragraph", "chunk_id": 0, "text": "x"},
                {"page": 500, "section": "", "chunk_type": "row", "chunk_id": 1, "text": "y", "row_id": 0}]
    annotate_metadata(metadata, unregistered_source("data/other.pdf"), None)
    for meta in metadata:
        assert meta["source_id"] == "unregistered_other"
        assert meta["chapter"] is None and meta["section_no"] is None and meta["page_section_nos"] == []
    assert metadata[0]["section"] == "Page 49" and metadata[1]["row_id"] == 0       # untouched


def test_annotation_leaves_pages_outside_the_map_unmapped(section_map, handbook):
    metadata = [{"page": 999, "section": "", "chunk_type": "paragraph", "chunk_id": 0, "text": "x"}]
    annotate_metadata(metadata, handbook, section_map)
    assert metadata[0]["source_id"] == HANDBOOK_ID and metadata[0]["chapter"] is None
    assert metadata[0]["page_section_nos"] == []


def test_sources_module_has_no_runtime_side_effects():
    assert not hasattr(sources, "_DEFAULT_REGISTRY")
    assert sources.REPO_ROOT == str(REPO)


# ---------------------------------------------------------------------------
# Strict source identity (review finding F-3)
# ---------------------------------------------------------------------------
HANDBOOK_REL = "data/UOS Faculty Handbook 25-26.pdf"


def test_registered_file_matches_in_every_spelling(registry, monkeypatch):
    monkeypatch.chdir(REPO)
    absolute = str(REPO / "data" / "UOS Faculty Handbook 25-26.pdf")
    spellings = [HANDBOOK_REL, absolute, HANDBOOK_REL.replace("/", "\\"), absolute.replace("/", "\\"),
                 "data/./UOS Faculty Handbook 25-26.pdf", "data/../data/UOS Faculty Handbook 25-26.pdf"]
    if os.name == "nt":
        spellings.append("DATA/uos faculty handbook 25-26.PDF")
    for spelling in spellings:
        assert registry.find_source_for_path(spelling).source_id == HANDBOOK_ID, spelling


def test_same_file_name_elsewhere_is_not_the_registered_source(registry, tmp_path):
    assert registry.find_source_for_path("C:/elsewhere/UOS Faculty Handbook 25-26.pdf") is None
    assert registry.find_source_for_path("/srv/handbooks/UOS Faculty Handbook 25-26.pdf") is None
    twin = tmp_path / "UOS Faculty Handbook 25-26.pdf"
    twin.write_bytes(b"%PDF-1.4 not the handbook")
    assert registry.find_source_for_path(str(twin)) is None                # exists, same name, different file
    assert registry.find_source_for_path("data/other.pdf") is None
    assert registry.find_source_for_path("") is None and registry.find_source_for_path(None) is None


def test_unregistered_twin_gets_an_unregistered_identity_through_the_knowledge_base(tmp_path):
    from handbook_bot.knowledge_base import _resolve_source

    twin = tmp_path / "UOS Faculty Handbook 25-26.pdf"
    twin.write_bytes(b"x")
    source, section_map = _resolve_source(str(twin), _fake_pages(272), lambda message: None)
    assert source.source_id == "unregistered_uos_faculty_handbook_25_26" and not source.registered
    assert section_map is None


# ---------------------------------------------------------------------------
# Section map page-count compatibility (review finding F-3)
# ---------------------------------------------------------------------------
def _fake_pages(n):
    return [{"page": i, "lines": [], "rows": [], "text": "p%d" % i, "section_hint": "", "ocr_snippets": []}
            for i in range(1, n + 1)]


def test_page_count_check_accepts_the_handbook_length_and_rejects_others(section_map):
    assert check_section_map_pages(section_map, _fake_pages(272)) is None
    for n in (200, 271, 273, 300):
        message = check_section_map_pages(section_map, _fake_pages(n))
        assert message and "272" in message and str(n) in message
    assert check_section_map_pages(section_map, []) is not None
    assert check_section_map_pages(section_map, _fake_pages(271) + [{"page": 300}]) is not None   # highest page off


def test_resolve_source_applies_the_map_only_when_page_counts_match(monkeypatch):
    from handbook_bot.knowledge_base import _resolve_source

    monkeypatch.chdir(REPO)
    messages = []
    source, section_map = _resolve_source(HANDBOOK_REL, _fake_pages(272), messages.append)
    assert source.source_id == HANDBOOK_ID and section_map is not None and len(section_map.records) == 206
    messages.clear()
    source, section_map = _resolve_source(HANDBOOK_REL, _fake_pages(200), messages.append)
    assert source.source_id == HANDBOOK_ID and source.registered              # identity kept
    assert section_map is None                                                # incompatible map not applied
    assert any("section map" in m.lower() and "200" in m and "not applied" in m for m in messages)


def test_incompatible_map_leaves_section_fields_null_not_fabricated(handbook):
    metadata = [{"page": 49, "section": "Page 49", "chunk_type": "paragraph", "chunk_id": 0, "text": "x"}]
    annotate_metadata(metadata, handbook, None)
    meta = metadata[0]
    assert meta["source_id"] == HANDBOOK_ID and meta["source_title"] == "UOS Faculty Handbook 2025/26"
    assert meta["chapter"] is None and meta["section_no"] is None and meta["section_title"] is None
    assert meta["page_section_nos"] == [] and meta["page"] == 49 and meta["section"] == "Page 49"


# ---------------------------------------------------------------------------
# Scope semantics against the real handbook (review findings F-1, F-2, F-5)
# ---------------------------------------------------------------------------
def _pages_allowed(metadata, scope):
    return sorted({m["page"] for m in metadata if scope.allows(m)})


def test_chapter_consistent_section_matching_keeps_the_printed_12_3_out_of_chapter_12(handbook_chunks):
    chunks, metadata = handbook_chunks
    page_180 = [m for m in metadata if m["page"] == 180]
    assert page_180 and all(m["chapter"] == 9 and "12.3" in m["page_section_nos"] for m in page_180)
    assert not any(RetrievalScope(section_prefixes=["12"]).allows(m) for m in page_180)              # A
    assert not any(RetrievalScope(section_numbers=["12.3"]).allows(m) for m in page_180)             # B
    assert _pages_allowed(metadata, RetrievalScope(section_numbers=["12.3"])) == [212, 213, 214]      # C
    assert all(RetrievalScope(chapters=[9]).allows(m) for m in page_180)                              # D
    assert _pages_allowed(metadata, RetrievalScope(chapters=[9], section_numbers=["12.3"])) == []      # E
    assert _pages_allowed(metadata, RetrievalScope(chapters=[12], section_numbers=["12.3"])) == [212, 213, 214]   # F
    prefix_12 = RetrievalScope(section_prefixes=["12"])
    assert {m["chapter"] for m in metadata if prefix_12.allows(m)} == {12}
    assert min(_pages_allowed(metadata, prefix_12)) == 203


def test_section_number_as_printed_is_kept_even_though_matching_is_chapter_consistent(handbook_chunks):
    chunks, metadata = handbook_chunks
    meta = next(m for m in metadata if m["page"] == 180 and m["section_no"] == "12.3")
    assert meta["section_title"] == "Copyright Protection Policy" and meta["chapter"] == 9


def test_prefix_scope_never_crosses_chapters(handbook_chunks):
    chunks, metadata = handbook_chunks
    for prefix, chapter in (("1", 1), ("10", 10), ("12", 12), ("16", 16)):
        assert {m["chapter"] for m in metadata if RetrievalScope(section_prefixes=[prefix]).allows(m)} == {chapter}
    assert _pages_allowed(metadata, RetrievalScope(section_prefixes=["12.1"])) == [203, 204, 205, 206, 207, 208]
    matched = {s for m in metadata if RetrievalScope(section_prefixes=["12.1"]).allows(m)
               for s in m["page_section_nos"] if s.startswith("12.1")}
    assert matched == {"12.1"}                                    # "12.10", "12.11" and "12.12" are not under "12.1"


def test_section_scopes_are_page_granular_on_shared_pages(handbook_chunks, section_map):
    """Documented contract (F-2): a section constraint admits every chunk of a
    page the section is present on, including chunks whose text belongs to a
    neighbouring section that shares the page. It is not text-level isolation."""
    chunks, metadata = handbook_chunks
    page_51 = [m for m in metadata if m["page"] == 51]
    assert page_51[0]["page_section_nos"] == ["1.14", "1.15"]
    assert all(RetrievalScope(section_numbers=["1.14"]).allows(m) for m in page_51)
    assert any("Diversity" in chunks[m["chunk_id"]] for m in page_51)               # 1.15 text is on the page
    page_222 = [m for m in metadata if m["page"] == 222]
    assert page_222[0]["page_section_nos"] == ["12.9", "12.10", "12.11", "12.12"]
    assert all(RetrievalScope(section_numbers=["12.12"]).allows(m) for m in page_222)
    assert any("12.10 Graduate Completion" in chunks[m["chunk_id"]] for m in page_222)
    shared = sum(1 for page in range(1, PAGE_COUNT + 1) if len(section_map.resolve(page).section_nos) > 1)
    assert shared == 115


def test_printed_section_numbers_are_not_unique_identifiers(handbook_chunks):
    """Documented contract (F-5): 5.2 and 7.8 are printed twice, so a section
    constraint on them admits both printed sections. A caller wanting one of
    them must add a page range."""
    chunks, metadata = handbook_chunks
    assert _pages_allowed(metadata, RetrievalScope(section_numbers=["5.2"])) == [113, 114, 115]
    assert _pages_allowed(metadata, RetrievalScope(section_numbers=["7.8"])) == [148, 149, 150]
    assert {m["section_title"] for m in metadata if m["section_no"] == "7.8"} == {
        "Promotion Procedures at the Department Level", "Promotion Procedures at the College Level"}
    assert _pages_allowed(metadata, RetrievalScope(section_numbers=["7.8"], page_ranges=[(149, 149)])) == [149]
