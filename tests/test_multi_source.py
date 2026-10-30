"""
Post-freeze PF-2: controlled multi-source guide ingestion.

Everything here runs offline: the registry, the page-level language filter, the procedure-aware chunking, the
multi-document assembly on the real files (handbook plus the four approved pilot guides, no embedding model), the
cache identity, the retrieval scope-extension seam, the source-conflict guard of the Plan F adapter, the per-document
citation chips and the diagnostics. The frozen specialists are built from the assembled corpus with the fake embedder,
exactly as ``tests/test_plan_f_runtime.py`` builds them from the handbook alone.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import faiss
import pytest

from conftest import FakeEmbedder
from faculty_services_support import RawOverlapReranker
from handbook_bot import plan_f_runtime as PF
from handbook_bot import sources as S
from handbook_bot import source_scopes as SS
from handbook_bot.chunking import build_chunks, read_procedures
from handbook_bot.knowledge_base import (
    PROCEDURE_DOCUMENT_TYPES,
    KnowledgeBase,
    SourceLoadError,
    arabic_share,
    assemble_corpus,
    corpus_identity,
    filter_english_pages,
    load_source,
)
from handbook_bot.qa import QAResult
from handbook_bot.retrieval import RetrievalScope
from handbook_bot.sources import METADATA_KEYS, SourceConflict, SourceRecord, SourceRegistryError, load_source_registry
from ui import pipeline as UP
from ui import presentation as P

REPO = Path(__file__).resolve().parents[1]
HANDBOOK = "uos_faculty_handbook_2025_26"
SHORT_GUIDE = "uos_blackboard_faculty_short_guide"
GRADE_TRANSFER = "uos_grade_transfer_dashboard_guide"
GRADEBOOK = "uos_blackboard_gradebook_guide"
ITHENTICATE = "uos_ithenticate_guidelines"
APPRAISAL = "uos_performance_appraisal_phase2_guide"
GUIDES = (SHORT_GUIDE, GRADE_TRANSFER, GRADEBOOK, ITHENTICATE)
HANDBOOK_CHUNKS = 13688
ARABIC = "كيفية إرسال الدرجات"   # "how to send the grades"


# ---------------------------------------------------------------------------
# Fixtures (module scope: the five PDFs are read once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry():
    return load_source_registry(str(REPO / "knowledge" / "sources.json"))


@pytest.fixture(scope="module")
def parts(registry):
    return assemble_corpus(registry.enabled_sources())


@pytest.fixture(scope="module")
def by_source(parts):
    out = {}
    for text, meta in zip(parts.chunks, parts.metadata):
        out.setdefault(meta["source_id"], []).append((text, meta))
    return out


@pytest.fixture(scope="module")
def kb(parts):
    embedder = FakeEmbedder()
    embeddings = embedder.encode(parts.chunks)
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embeddings)
    primary = parts.records[0]
    stats = {"total_chunks": len(parts.chunks), "total_pages": sum(len(p) for p in parts.source_pages.values()),
             "source": {"source_id": primary.source_id, "registered": True, "section_records": parts.primary_section_records},
             "sources": parts.per_source}
    return KnowledgeBase(pdf_file="corpus", embedder=embedder, reranker=RawOverlapReranker(), groq_client=None,
                         pages=parts.source_pages[primary.source_id], chunks=parts.chunks, metadata=parts.metadata,
                         embeddings=embeddings, index=index, stats=stats, source=primary, sources=parts.records,
                         source_pages=parts.source_pages, corpus_key=parts.key)


def _result(items, pages, answer="x"):
    return QAResult(answer=answer, pages=list(pages), best_section="s", evidence="e", query_type="policy", items=list(items),
                    used_llm=False, num_candidates=len(items), num_reranked=len(items), llm_calls=0, retried=False, sub_questions=[])


def _item(source_id, title, page, section=None, chunk_type="row"):
    meta = {"source_id": source_id, "source_title": title, "page": page, "chunk_type": chunk_type, "section": "Page %d" % page}
    if section:
        meta["section_title"] = section
    return {"idx": 0, "chunk": "quote", "meta": meta}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_registry_holds_the_handbook_four_enabled_guides_and_one_held_record(registry):
    ids = [r.source_id for r in registry]
    assert ids == [HANDBOOK, SHORT_GUIDE, GRADE_TRANSFER, GRADEBOOK, ITHENTICATE, APPRAISAL]
    assert [r.source_id for r in registry.enabled_sources()] == [HANDBOOK, *GUIDES]
    held = registry.get_source(APPRAISAL)
    assert not held.enabled and held.access == "internal_use" and "HOLD" in held.verification_status
    assert not held.exists()                                      # the Internal Use document is not in the repository
    assert registry.validate_source_registry() == []            # a disabled record needs no file; the enabled hashes match


def test_every_enabled_source_is_a_repository_relative_english_pdf_with_a_hash_and_a_label(registry):
    for record in registry.enabled_sources():
        assert record.exists() and not os.path.isabs(record.relative_path) and "\\" not in record.relative_path
        assert record.language == "en" and record.source_type == "pdf" and len(record.sha256) == 64
        assert record.label and record.citation_label == record.title
        assert S.file_sha256(record.absolute_path()) == record.sha256
    assert registry.get_source(HANDBOOK).page_filter == "" and all(registry.get_source(g).page_filter == "english_only" for g in GUIDES)


def test_domains_follow_the_pf1_mapping(registry):
    assert registry.get_source(HANDBOOK).domains == ("teaching", "research", "faculty_services", "institutional")
    assert registry.get_source(SHORT_GUIDE).domains == ("teaching", "institutional")
    assert registry.get_source(GRADE_TRANSFER).domains == ("teaching", "institutional")
    assert registry.get_source(GRADEBOOK).domains == ("teaching",)
    assert registry.get_source(ITHENTICATE).domains == ("research",)
    assert registry.get_source(APPRAISAL).domains == ("faculty_services",)


def test_the_ithenticate_conflict_is_registered_unresolved_and_matched_not_merged(registry):
    (conflict,) = registry.get_source(ITHENTICATE).conflicts
    assert conflict.topic == "ithenticate_yearly_submission_limit" and conflict.status == "unresolved"
    assert conflict.matches("... (Maximum 30 | Documents per year)") and conflict.matches("documents  per year)")
    assert not conflict.matches("Files must be less than 100MB.")
    assert "15" in conflict.note and "30" in conflict.note       # both values recorded, neither chosen


def test_a_frozen_runtime_registry_with_only_the_old_fields_still_loads(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": [{
        "source_id": "old_handbook", "title": "Old", "relative_path": "data/old.pdf", "source_type": "pdf",
        "version": "1", "authority": "a", "verification_status": "v", "section_map": None, "notes": ""}]}), encoding="utf-8")
    (record,) = load_source_registry(str(path)).list_sources()
    assert record.enabled and record.language == "en" and record.label == "Old" and record.domains == () and record.conflicts == ()
    assert record.page_filter == "" and record.sha256 == "" and record.access == ""


@pytest.mark.parametrize("field, value, message", [
    ("language", "arabic", "two-letter"),
    ("language", "AR", "two-letter"),
    ("enabled", "yes", "true or false"),
    ("sha256", "abc", "64 lower-case hex"),
    ("page_filter", "arabic_only", "page_filter"),
    ("access", "secret", "access"),
    ("domains", "teaching", "list of specialist ids"),
    ("conflicts", [{"topic": "t", "phrases": ["p"], "severity": "high"}], "unknown conflict field"),
    ("conflicts", [{"topic": "t", "phrases": []}], "non-empty list of phrases"),
])
def test_malformed_pf2_fields_are_rejected(field, value, message):
    base = {"source_id": "x_source", "title": "X", "relative_path": "data/x.pdf", "source_type": "pdf"}
    base[field] = value
    with pytest.raises(SourceRegistryError, match=message):
        SourceRecord(**base)


def test_record_to_dict_round_trips_through_the_loader(registry, tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(registry.to_dict()), encoding="utf-8")
    again = load_source_registry(str(path))
    assert again.to_dict() == registry.to_dict() and again.to_dict()["schema_version"] == 2


# ---------------------------------------------------------------------------
# Language control (deterministic, no Arabic support)
# ---------------------------------------------------------------------------
def test_arabic_share_counts_letters_only():
    assert arabic_share("") == 0.0 and arabic_share("1234 | ") == 0.0 and arabic_share("Grade Transfer") == 0.0
    assert arabic_share(ARABIC) == 1.0
    assert 0.4 < arabic_share("Banner " + ARABIC[:6]) < 0.6


def _page(number, rows, hint=""):
    text = " ".join(rows)
    return {"page": number, "lines": list(rows), "rows": list(rows), "text": text, "section_hint": hint, "ocr_snippets": []}


def test_english_only_filter_drops_arabic_pages_and_rows_and_keeps_english_pages_untouched():
    english = _page(1, ["How to transfer grades:", "1. Log in to Blackboard."])
    rows = ["Search for the course by using the search bar at the top of the dashboard.", "You can search by Course ID and Term.", ARABIC + " Banner"]
    mixed = _page(2, rows, hint=ARABIC)
    arabic = _page(4, [ARABIC, ARABIC + " Banner", ARABIC])
    assert arabic_share(mixed["text"]) < 0.3 < arabic_share(arabic["text"])
    kept, report = filter_english_pages([english, mixed, arabic])
    assert [p["page"] for p in kept] == [1, 2]                  # numbers are never renumbered
    assert kept[0] is english                                      # a page without Arabic script is the same object
    assert kept[1]["rows"] == rows[:2] and kept[1]["text"] == " ".join(rows[:2])
    assert kept[1]["section_hint"] == "" and mixed["rows"][2].startswith(ARABIC)   # the input page is not mutated
    assert report["pages_dropped"] == [4] and report["rows_dropped"] == 2 and report["pages_indexed"] == 2


def test_a_page_left_without_text_by_the_filter_is_dropped():
    only_arabic_rows = _page(3, [ARABIC])
    only_arabic_rows["text"] = "x"                                 # below the page threshold by construction
    kept, report = filter_english_pages([only_arabic_rows])
    assert kept == [] and report["pages_dropped"] == [3]


# ---------------------------------------------------------------------------
# Procedure-aware chunking (system manuals only)
# ---------------------------------------------------------------------------
def test_read_procedures_groups_numbered_lettered_and_bulleted_steps_under_their_heading():
    rows = ["Grade Transfer Dashboard Guide", "How to Send Grades to Banner as instructor?",
            "1. Search for the course by using the search bar.", "You can search by Course ID and Term.",
            "2. Click on View Details on the dashboard", "3. Click on Transfer:",
            "How to approve Grades as Chair/Dean?", "1. Click on the link to access the portal", "2. Use your credentials to log in",
            "A", "a) Gradable Items list.", "b) Student list view.",
            "File Requirements:", "- Files must be less than 100MB.", " The maximum document length is 400 pages."]
    procedures = read_procedures(rows, "Page 1")
    assert [p["heading"] for p in procedures] == ["How to Send Grades to Banner as instructor?",
                                                  "How to approve Grades as Chair/Dean?", "File Requirements"]
    first = procedures[0]["steps"]
    assert [s["label"] for s in first] == ["1.", "2.", "3."]
    assert first[0]["text"] == "Search for the course by using the search bar. You can search by Course ID and Term."
    assert [s["label"] for s in procedures[1]["steps"]] == ["1.", "2.", "a)", "b)"]   # the bare callout "A" is neither a step nor a heading
    assert [s["label"] for s in procedures[2]["steps"]] == ["-", "-"]


def test_a_short_lead_in_row_names_the_list_below_it():
    rows = ["If you are unable to access iThenticate software email: servicedesk@sharjah.ac.ae", "How to upload a document",
            "1) Select the Upload a File link under the Submit a Document section", "2) Enter in the author name and the document title",
            "Note: The analysis may take a few hours", "Checking Similarity Report",
            " The similarity index and originality report can be found under the Documents",
            " You need to click on report percentage shown to be able to view the similarity report"]
    procedures = read_procedures(rows, "")
    assert [p["heading"] for p in procedures] == ["How to upload a document", "Checking Similarity Report"]
    assert procedures[0]["steps"][1]["text"].endswith("Note: The analysis may take a few hours")
    swallowed = ["How to access IThenticate", "1) Open www.ithenticate.com", "2) Click on login", "3) Type your university email and your password",
                 "If you are unable to access iThenticate software email: servicedesk@sharjah.ac.ae", "How to upload a document",
                 "1) Select the Upload a File link under the Submit a Document section", "2) Enter in the author name and the document title"]
    procedures = read_procedures(swallowed, "")
    assert [p["heading"] for p in procedures] == ["How to access IThenticate", "How to upload a document"]
    assert [s["text"] for s in procedures[0]["steps"]][:2] == ["Open www.ithenticate.com", "Click on login"]   # two-word steps are kept
    assert procedures[0]["steps"][2]["text"].endswith("servicedesk@sharjah.ac.ae")          # the lead-in is given back to its list
    bulleted = read_procedures([" Customize Your Course:", "- Select Edit display settings under Course Image"], " Course View:")
    assert [p["heading"] for p in bulleted] == ["Customize Your Course"]                   # a bulleted heading is not a step
    assert read_procedures(["- Select Edit display settings under Course Image"], " Course View:")[0]["heading"] == "Course View"
    long_sentence = ["This paragraph explains at length, in far more than twelve words, what the following numbered list is going to describe in detail.",
                     "1) First step of the list here", "2) Second step of the list here"]
    assert read_procedures(long_sentence, "Page 3")[0]["heading"] == "Page 3"       # a long sentence is not a lead-in


def test_a_numbered_step_that_restarts_opens_a_new_procedure():
    rows = ["Steps:", "1. First thing to do", "2. Second thing to do", "1. Another first thing", "2. Another second thing"]
    procedures = read_procedures(rows)
    assert len(procedures) == 2 and [len(p["steps"]) for p in procedures] == [2, 2]


def test_default_chunking_emits_no_procedure_chunks_and_procedures_add_step_and_window_chunks():
    page = _page(1, ["How to access iThenticate", "1) Open www.ithenticate.com now", "2) Click on login button", "3) Type your university email"])
    plain, plain_meta = build_chunks([page])
    assert {m["chunk_type"] for m in plain_meta} <= {"paragraph", "row", "row_window"}
    with_procedures, meta = build_chunks([page], procedures=True)
    assert with_procedures[:len(plain)] == plain                   # the four classic types come first, unchanged
    steps = [m for m in meta if m["chunk_type"] == "procedure_step"]
    windows = [m for m in meta if m["chunk_type"] == "procedure_window"]
    assert [s["step_label"] for s in steps] == ["1.", "2.", "3."] and all(s["procedure_heading"] == "How to access iThenticate" for s in steps)
    assert len(windows) == 1 and windows[0]["step_start"] == 0 and windows[0]["step_end"] == 2
    assert with_procedures[steps[0]["chunk_id"]] == "How to access iThenticate: 1. Open www.ithenticate.com now"
    assert " | " in with_procedures[windows[0]["chunk_id"]]
    assert all(m["chunk_id"] == i for i, m in enumerate(meta))


# ---------------------------------------------------------------------------
# The assembled corpus on the real files
# ---------------------------------------------------------------------------
def test_handbook_comes_first_and_is_chunked_exactly_as_before(parts, registry):
    from handbook_bot.pdf_loader import load_pdf

    handbook = parts.per_source[0]
    assert handbook["source_id"] == HANDBOOK and handbook["chunks"] == HANDBOOK_CHUNKS and handbook["chunk_id_range"] == [0, HANDBOOK_CHUNKS - 1]
    assert handbook["chunk_types"] == {"paragraph": 847, "row": 8412, "row_window": 4429} and handbook["procedures"] is False
    alone, _ = build_chunks(load_pdf(registry.get_source(HANDBOOK).absolute_path()))
    assert parts.chunks[:HANDBOOK_CHUNKS] == alone
    assert handbook["section_records"] == 206 and parts.primary_section_records == 206


def test_chunk_ids_are_global_positions_and_every_chunk_names_its_source(parts):
    assert all(m["chunk_id"] == i for i, m in enumerate(parts.metadata))
    assert all(m["text"] == c for c, m in zip(parts.chunks, parts.metadata))
    for meta in parts.metadata:
        for key in METADATA_KEYS:
            assert key in meta
        assert meta["source_page"] == meta["page"] and meta["language"] == "en"


def test_guides_keep_their_own_page_numbers_and_the_arabic_pages_are_not_indexed(parts):
    pages = {k: [p["page"] for p in v] for k, v in parts.source_pages.items()}
    assert pages[SHORT_GUIDE] == list(range(1, 12)) and pages[GRADEBOOK] == list(range(1, 11))
    assert pages[GRADE_TRANSFER] == [1, 2, 3] and pages[ITHENTICATE] == [1, 3]
    reports = {s["source_id"]: s["language_filter"] for s in parts.per_source}
    assert reports[HANDBOOK] is None
    assert reports[GRADE_TRANSFER]["pages_dropped"] == [4, 5, 6] and reports[ITHENTICATE]["pages_dropped"] == [2]
    assert reports[SHORT_GUIDE]["pages_dropped"] == [] and reports[GRADEBOOK]["pages_dropped"] == []


def test_no_guide_chunk_carries_arabic_script(by_source):
    for guide in GUIDES:
        assert by_source[guide] and all(arabic_share(text) == 0.0 for text, _ in by_source[guide]), guide


def test_guide_chunks_carry_identity_and_content_type_but_no_fabricated_section(by_source, registry):
    for guide in GUIDES:
        record = registry.get_source(guide)
        for text, meta in by_source[guide]:
            assert meta["source_title"] == record.title and meta["citation_label"] == record.label
            assert meta["document_type"] == record.document_type and meta["system"] == record.system
            assert meta["authority_level"] == "system_guide" and meta["department"] == "ITC Academic Computing"
            assert meta["chapter"] is None and meta["section_no"] is None and meta["section_title"] is None and meta["page_section_nos"] == []
            assert meta["content_type"] == S.CONTENT_TYPES[meta["chunk_type"]]


def test_procedure_chunks_exist_only_in_the_guides_and_classic_chunks_are_kept(parts, by_source):
    counts = {s["source_id"]: s["chunk_types"] for s in parts.per_source}
    assert "procedure_step" not in counts[HANDBOOK] and "procedure_window" not in counts[HANDBOOK]
    for guide in GUIDES:
        assert counts[guide]["procedure_step"] > 0 and counts[guide]["procedure_window"] > 0
        assert counts[guide]["row"] > 0 and counts[guide]["row_window"] > 0 and counts[guide]["paragraph"] > 0
    step = next(text for text, meta in by_source[GRADE_TRANSFER] if meta["chunk_type"] == "procedure_step" and meta["page"] == 2)
    assert step.startswith("How to Send Grades to Banner as instructor?: 1. Search for the course")


def test_the_conflict_phrases_mark_only_ithenticate_chunks_that_carry_them(parts, by_source, registry):
    (conflict,) = registry.get_source(ITHENTICATE).conflicts
    flagged = [(text, meta) for text, meta in zip(parts.chunks, parts.metadata) if meta.get("source_conflict")]
    assert flagged and all(meta["source_id"] == ITHENTICATE for _, meta in flagged)
    assert all(meta["source_conflict"] == conflict.topic and conflict.matches(text) for text, meta in flagged)
    assert all(meta["source_conflict"] is None for text, meta in by_source[ITHENTICATE] if not conflict.matches(text))
    assert next(s for s in parts.per_source if s["source_id"] == ITHENTICATE)["conflict_chunks"] == len(flagged)


def test_corpus_identity_is_deterministic_path_free_and_sensitive_to_the_enabled_set(registry, parts):
    key, entries = corpus_identity(registry.enabled_sources())
    assert key == parts.key and len(key) == 16
    assert all(set(e) == {"source_id", "sha256", "page_filter", "procedures", "section_map"} for e in entries)
    assert json.dumps(entries).count("/") == 0 and ":\\" not in json.dumps(entries)
    fewer, _ = corpus_identity(registry.enabled_sources()[:-1])
    assert fewer != key
    assert corpus_identity(registry.enabled_sources())[0] == key


def test_load_source_errors_name_the_source(tmp_path, monkeypatch, registry):
    monkeypatch.setattr(S, "REPO_ROOT", str(tmp_path))
    (tmp_path / "data").mkdir()
    missing = SourceRecord(source_id="ghost_guide", title="Ghost", relative_path="data/ghost.pdf", source_type="pdf")
    with pytest.raises(SourceLoadError, match="ghost_guide"):
        load_source(missing)
    shutil.copy(REPO / "data" / "guides" / "uos_ithenticate_guidelines.pdf", tmp_path / "data" / "copy.pdf")
    wrong = SourceRecord(source_id="other_edition", title="Other", relative_path="data/copy.pdf", source_type="pdf", sha256="0" * 64)
    with pytest.raises(SourceLoadError, match="other_edition.*sha256"):
        load_source(wrong)
    right = SourceRecord(source_id="good_copy", title="Good", relative_path="data/copy.pdf", source_type="pdf",
                         sha256=registry.get_source(ITHENTICATE).sha256, page_filter="english_only")
    pages, section_map, report = load_source(right)
    assert [p["page"] for p in pages] == [1, 3] and section_map is None and report["language_filter"]["pages_dropped"] == [2]


def test_assemble_corpus_refuses_an_empty_record_list():
    with pytest.raises(SourceLoadError):
        assemble_corpus([])


# ---------------------------------------------------------------------------
# The scope-extension seam
# ---------------------------------------------------------------------------
def test_extended_scope_admits_the_extra_source_and_keeps_the_handbook_constraints():
    scope = RetrievalScope(source_ids=[HANDBOOK], page_ranges=[(56, 74)])
    extended = SS.extended_scope(scope, [GRADEBOOK, HANDBOOK])
    assert extended.also_source_ids == frozenset({GRADEBOOK})
    assert extended.allows({"source_id": GRADEBOOK, "page": 6}) and not scope.allows({"source_id": GRADEBOOK, "page": 6})
    assert extended.allows({"source_id": HANDBOOK, "page": 60}) and not extended.allows({"source_id": HANDBOOK, "page": 10})
    assert not extended.allows({"source_id": SHORT_GUIDE, "page": 6})
    assert "also_source_ids" not in scope.describe() and extended.describe()["also_source_ids"] == [GRADEBOOK]
    assert SS.extended_scope(scope, [HANDBOOK]) is scope and SS.extended_scope(extended, [GRADEBOOK]) is extended


def test_domain_policy_is_read_from_the_records_and_ignores_disabled_ones(registry):
    policy = SS.domain_sources(registry.list_sources())
    assert policy["teaching"] == frozenset({HANDBOOK, SHORT_GUIDE, GRADE_TRANSFER, GRADEBOOK})
    assert policy["research"] == frozenset({HANDBOOK, ITHENTICATE})
    assert policy["faculty_services"] == frozenset({HANDBOOK})          # the appraisal guide is on hold
    assert policy["institutional"] == frozenset({HANDBOOK, SHORT_GUIDE, GRADE_TRANSFER})


def test_extend_specialist_scopes_applies_the_policy_per_specialist(registry):
    scope = RetrievalScope(source_ids=[HANDBOOK], page_ranges=[(14, 60)])
    fake = {sid: SimpleNamespace(scope=scope) for sid in SS.SPECIALIST_IDS}
    report = SS.extend_specialist_scopes(fake, registry.list_sources())
    assert report["teaching"]["added"] == sorted([SHORT_GUIDE, GRADE_TRANSFER, GRADEBOOK]) and report["teaching"]["applied"]
    assert report["research"]["added"] == [ITHENTICATE] and report["institutional"]["added"] == sorted([SHORT_GUIDE, GRADE_TRANSFER])
    assert report["faculty_services"] == {"applied": False, "added": [], "allowed_sources": [HANDBOOK]}
    assert fake["faculty_services"].scope is scope and fake["research"].scope.also_source_ids == frozenset({ITHENTICATE})
    assert SS.extend_specialist_scopes({}, registry.list_sources())["teaching"] == {"applied": False, "reason": "not registered"}


def test_page_collisions_flag_a_guide_whose_pages_reach_into_a_specialist_scope(registry):
    scope = RetrievalScope(source_ids=[HANDBOOK], page_ranges=[(14, 60)])
    fake = {sid: SimpleNamespace(scope=scope) for sid in SS.SPECIALIST_IDS}
    records = registry.enabled_sources()
    assert SS.page_collisions(records, {GRADEBOOK: 10, SHORT_GUIDE: 11, GRADE_TRANSFER: 3, ITHENTICATE: 2, HANDBOOK: 272}, fake) == []
    hits = SS.page_collisions(records, {GRADEBOOK: 20}, fake)
    assert len(hits) == 4 and all("pages 14-20 of %s" % GRADEBOOK in h for h in hits)


def test_specialists_build_from_the_corpus_with_the_seam_off_by_default_and_on_when_asked(kb):
    off = PF.build_specialist_registry(kb)
    assert off.ok, off.errors
    assert off.scope_extension == {"enabled": False, "applied": {}, "collisions": []}
    assert "scope_extension" in off.to_dict() and json.dumps(off.to_dict())
    teaching = off.registry.resolve("teaching")
    assert not teaching.scope.allows({"source_id": GRADEBOOK, "page": 6})
    on = PF.build_specialist_registry(kb, guide_scopes=True)
    assert on.ok and on.scope_extension["enabled"] and on.scope_extension["collisions"] == []
    assert on.scope_extension["applied"]["teaching"]["added"] == sorted([SHORT_GUIDE, GRADE_TRANSFER, GRADEBOOK])
    assert on.registry.resolve("teaching").scope.allows({"source_id": GRADEBOOK, "page": 6})
    assert on.registry.resolve("research").scope.allows({"source_id": ITHENTICATE, "page": 1})
    assert not on.registry.resolve("research").scope.allows({"source_id": GRADEBOOK, "page": 1})
    assert not on.registry.resolve("faculty_services").scope.also_source_ids
    assert PF.build_specialist_registry(kb).scope_extension["enabled"] is False     # each build reads the switch afresh


def test_a_guide_whose_pages_overlap_a_specialist_scope_refuses_the_whole_registry(kb):
    fake_pages = dict(kb.source_pages)
    fake_pages[GRADEBOOK] = [{"page": n} for n in range(1, 21)]
    shadow = SimpleNamespace(**{**kb.__dict__, "source_pages": fake_pages})
    build = PF.build_specialist_registry(shadow)
    assert not build.ok and build.registry is None and "scope_extension" in build.errors
    assert "overlap" in build.errors["scope_extension"] and build.scope_extension["collisions"]


def test_a_one_document_knowledge_base_is_left_alone_by_the_seam(kb):
    single = SimpleNamespace(**{**kb.__dict__, "sources": [], "source_pages": {}})
    build = PF.build_specialist_registry(single, guide_scopes=True)
    assert build.ok and build.scope_extension == {}


# ---------------------------------------------------------------------------
# The source-conflict guard
# ---------------------------------------------------------------------------
def test_source_conflict_problem_reads_the_stamped_chunk(parts):
    flagged = next(m for m in parts.metadata if m.get("source_conflict"))
    clean = next(m for m in parts.metadata if m["source_id"] == ITHENTICATE and not m.get("source_conflict"))
    finding = SimpleNamespace(metadata={"chunk_id": flagged["chunk_id"]}, finding_id="f-1")
    problem = PF.source_conflict_problem(finding, parts.metadata)
    assert problem and "ithenticate_yearly_submission_limit" in problem and "not served" in problem
    assert PF.source_conflict_problem(SimpleNamespace(metadata={"chunk_id": clean["chunk_id"]}), parts.metadata) is None
    assert PF.source_conflict_problem(SimpleNamespace(metadata={"chunk_id": 10 ** 9}), parts.metadata) is None
    assert PF.source_conflict_problem(SimpleNamespace(metadata=None), parts.metadata) is None
    assert PF.PlanFOutcome.REJECTED_BY_SOURCE_CONFLICT.value == "PLAN_F_REJECTED_BY_SOURCE_CONFLICT"


def test_the_adapter_rejects_a_passthrough_finding_under_a_conflict(kb, parts, monkeypatch):
    """A specialist whose finding quotes the conflicted iThenticate limit exactly is stopped by the conflict guard,
    after the quote guard passed and before anything is rendered."""
    from handbook_bot.agents.specialists.contracts import Finding, FindingStatus, SpecialistFindings, SpecialistId

    flagged_id = next(m["chunk_id"] for m in parts.metadata if m.get("source_conflict") and m["chunk_type"] == "row")
    chunk = parts.chunks[flagged_id]
    meta = parts.metadata[flagged_id]
    runtime, build = PF.build_runtime(kb)
    assert build.ok
    research = runtime.registry.resolve("research")

    def fake_run(task):
        finding = Finding(finding_id="f-conflict", claim=chunk, evidence_quote=chunk, source_id=meta["source_id"],
                          source_title=meta["source_title"], page=meta["page"], confidence=0.9,
                          metadata={"chunk_id": flagged_id, "clauses": [0], "evidence_tier": "E1"})
        return SpecialistFindings(specialist_id=SpecialistId.RESEARCH, task_id=task.task_id, status=FindingStatus.SUPPORTED,
                                  findings=[finding], summary="limit", metadata={"clauses": [{"index": 0, "clause": task.question,
                                                                                               "status": "supported", "subject": "iThenticate"}]})

    monkeypatch.setattr(research, "run", fake_run)
    attempt = runtime.attempt("How many documents can I submit to iThenticate per year?")
    assert attempt.outcome is not PF.PlanFOutcome.ACCEPTED
    if attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SOURCE_CONFLICT:
        assert "ithenticate_yearly_submission_limit" in attempt.reason
        assert any(e.agent == "source_conflict_guard" for e in attempt.trace)
    else:                                                          # routed elsewhere or stopped earlier: still never served
        assert attempt.result is None


# ---------------------------------------------------------------------------
# Citations, diagnostics and the legacy path
# ---------------------------------------------------------------------------
def test_single_document_chips_are_unchanged():
    items = [_item(HANDBOOK, "UOS Faculty Handbook 2025/26", 224, "Examinations Policy"), _item(HANDBOOK, "UOS Faculty Handbook 2025/26", 225)]
    assert P.source_chips(_result(items, [224, 225])) == [("doc", "UOS Faculty Handbook 2025/26"), ("page", "Page 224 · Examinations Policy"), ("page", "Page 225")]
    assert not any(cls == "doc" for cls, _ in P.source_chips(_result([], [])))    # nothing cited: no document chip


def test_multi_document_chips_group_each_document_with_its_own_pages_without_raw_ids():
    items = [_item(HANDBOOK, "UOS Faculty Handbook 2025/26", 224, "Examinations Policy"),
             _item(GRADE_TRANSFER, "Grade Transfer Dashboard Guide", 2), _item(GRADE_TRANSFER, "Grade Transfer Dashboard Guide", 1)]
    chips = P.source_chips(_result(items, [224, 2, 1]))
    assert chips == [("doc", "UOS Faculty Handbook 2025/26"), ("page", "Page 224 · Examinations Policy"),
                     ("doc", "Grade Transfer Dashboard Guide"), ("page", "Page 2"), ("page", "Page 1")]
    assert not any("uos_" in label for _, label in chips)
    rows = dict(next(rows for title, rows in P.dev_sections(_result(items, [224, 2, 1]), {"mode": "legacy", "served": "legacy", "attempted": False}) if title == "Evidence"))
    assert rows["Document"] == "UOS Faculty Handbook 2025/26; Grade Transfer Dashboard Guide"
    assert rows["Source ids"] == "%s, %s" % (HANDBOOK, GRADE_TRANSFER)


def test_two_documents_citing_the_same_page_number_keep_both_chips():
    items = [_item(GRADEBOOK, "Blackboard Ultra Guide: Gradebook", 2), _item(GRADE_TRANSFER, "Grade Transfer Dashboard Guide", 2)]
    chips = P.source_chips(_result(items, [2]))
    assert chips == [("doc", "Blackboard Ultra Guide: Gradebook"), ("page", "Page 2"), ("doc", "Grade Transfer Dashboard Guide"), ("page", "Page 2")]


def test_diagnostics_carry_source_ids_only(kb):
    legacy = _result([_item(GRADEBOOK, "Blackboard Ultra Guide: Gradebook", 6)], [6], answer="x")
    d = UP._diagnose("legacy", "legacy", None, None, None, legacy=legacy)
    assert d["legacy_sources"] == [GRADEBOOK] and d["plan_f_sources"] == [] and d["legacy_pages"] == [6]
    assert "Gradebook" not in json.dumps(d)                        # ids, never titles or text
    assert json.dumps(d)


def test_legacy_retrieval_sees_the_guides_and_scoped_specialist_retrieval_does_not(kb):
    """The unscoped legacy search is the only path that reaches guide chunks today; a frozen specialist scope admits none."""
    from handbook_bot.retrieval import classify_query, gather_candidates

    question = "How do I transfer grades from Blackboard to Banner using the Grade Transfer Dashboard?"
    embedding = kb.embedder.encode([question])
    unscoped = gather_candidates(question, classify_query(question), embedding, kb.index, kb.chunks, kb.metadata)
    assert any(c["meta"]["source_id"] == GRADE_TRANSFER for c in unscoped)
    teaching = PF.build_specialist_registry(kb).registry.resolve("teaching")
    scoped = gather_candidates(question, classify_query(question), embedding, kb.index, kb.chunks, kb.metadata, scope=teaching.scope)
    assert scoped and all(c["meta"]["source_id"] == HANDBOOK for c in scoped)


# ---------------------------------------------------------------------------
# Source-specific retrieval (brief section 30) and wrong-source adversarial retrieval (section 31), offline:
# the fake embedder plus the production lexical scoring over the real assembled corpus, unscoped as the legacy path
# searches. The real-model counterpart is qa/post_freeze_pf2/wrong_source_retrieval.py.
# ---------------------------------------------------------------------------
SOURCE_RETRIEVAL_CASES = [
    (SHORT_GUIDE, "How do I add a Learning Module or Folder to my Blackboard course content?", 8),
    (GRADEBOOK, "How do I assign automatic zeros for past due work in the Blackboard gradebook settings?", 8),
    (GRADE_TRANSFER, "How do I send grades to Banner as instructor with View Details and Transfer on the dashboard?", 2),
    (ITHENTICATE, "How do I upload a document to iThenticate with the Upload a File link?", 1),
]


def _top(kb, question, n=5):
    from handbook_bot.retrieval import classify_query, gather_candidates

    embedding = kb.embedder.encode([question])
    return gather_candidates(question, classify_query(question), embedding, kb.index, kb.chunks, kb.metadata)[:n]


@pytest.mark.parametrize("source_id, question, page", SOURCE_RETRIEVAL_CASES, ids=[c[0] for c in SOURCE_RETRIEVAL_CASES])
def test_each_pilot_document_is_retrieved_with_its_page_identity_and_procedure_context(kb, registry, source_id, question, page):
    top = _top(kb, question)
    record = registry.get_source(source_id)
    assert top[0]["meta"]["source_id"] == source_id                                   # correct source first
    assert sum(1 for c in top if c["meta"]["source_id"] == source_id) >= 3          # unrelated sources not preferred
    assert any(c["meta"]["source_id"] == source_id and c["meta"]["page"] == page for c in top)
    for cand in top:
        meta = kb.metadata[cand["idx"]]
        assert cand["meta"] is meta and meta["source_id"] == cand["meta"]["source_id"]   # source id preserved end to end
        if meta["source_id"] == source_id:
            assert meta["source_title"] == record.title and meta["citation_label"] == record.label
    procedure = [c for c in _top(kb, question, 24) if c["meta"]["source_id"] == source_id and c["meta"]["chunk_type"] in ("procedure_step", "procedure_window")]
    assert procedure and all(c["meta"]["procedure_heading"] for c in procedure)    # step context: each step names its procedure


WRONG_SOURCE_CASES = [
    ("Who approves grades to Banner as Chair or Dean in the grade transfer portal?", GRADE_TRANSFER, 3),
    ("How do I add an attendance row in the Blackboard gradebook?", GRADEBOOK, 3),
    ("How do I log in to iThenticate with my university email?", ITHENTICATE, 1),
    ("Whom do I email for support if I cannot access iThenticate?", ITHENTICATE, 1),
    # the hashed bag-of-words test embedder is not semantic ("sign in" also matches page 2's "Sign Out"), so the offline
    # case names the mechanism; the real-model check asks it the vague way
    ("How do I verify my identity with multifactor authentication when I sign in to Blackboard?", SHORT_GUIDE, 1),
]


@pytest.mark.parametrize("question, source_id, page", WRONG_SOURCE_CASES, ids=["approval", "attendance", "login", "support", "mfa"])
def test_terms_shared_across_documents_bind_the_right_source(kb, question, source_id, page):
    top = _top(kb, question)
    assert top[0]["meta"]["source_id"] == source_id
    assert any(c["meta"]["source_id"] == source_id and c["meta"]["page"] == page for c in top)


def test_a_grading_policy_question_stays_with_the_handbook(kb):
    """System/policy boundary: a policy question is not answered from a system manual."""
    top = _top(kb, "Within how many hours must instructors submit the final examination results to the Department Chair?")
    assert top[0]["meta"]["source_id"] == HANDBOOK and any(c["meta"]["page"] == 224 for c in top if c["meta"]["source_id"] == HANDBOOK)
