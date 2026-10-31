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
# post-freeze PF-4 (D-113): the first P1 batch, approved for Teaching & Learning only
ATTENDANCE = "uos_blackboard_attendance_guide"
COLLABORATE = "uos_blackboard_collaborate_guide"
COMMUNICATION = "uos_blackboard_communication_guide"
P1_GUIDES = (ATTENDANCE, COLLABORATE, COMMUNICATION)
GUIDES = (SHORT_GUIDE, GRADE_TRANSFER, GRADEBOOK, ITHENTICATE, *P1_GUIDES)
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
def test_registry_holds_the_handbook_the_enabled_guides_and_one_held_record(registry):
    ids = [r.source_id for r in registry]
    assert ids == [HANDBOOK, SHORT_GUIDE, GRADE_TRANSFER, GRADEBOOK, ITHENTICATE, *P1_GUIDES, APPRAISAL]
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
    # post-freeze PF-4: the P1 guides are English throughout (the attendance guide's image-only page 3 has no text)
    assert pages[ATTENDANCE] == [1, 2] and pages[COLLABORATE] == [1, 2, 3, 4] and pages[COMMUNICATION] == [1, 2, 3, 4, 5]
    assert all(reports[g]["pages_dropped"] == [] for g in P1_GUIDES)


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


def test_the_conflict_phrases_mark_only_the_chunks_of_their_own_source_that_carry_them(parts, by_source, registry):
    """The iThenticate quota (PF-2) and, post-freeze PF-4, the communication guide's 2022 journal recommendation: each
    registered conflict stamps exactly the chunks of its own source that carry one of its phrases, nothing else."""
    conflicts = {ITHENTICATE: registry.get_source(ITHENTICATE).conflicts, COMMUNICATION: registry.get_source(COMMUNICATION).conflicts}
    assert len(conflicts[ITHENTICATE]) == 1 and len(conflicts[COMMUNICATION]) == 1
    flagged = [(text, meta) for text, meta in zip(parts.chunks, parts.metadata) if meta.get("source_conflict")]
    assert {meta["source_id"] for _, meta in flagged} == {ITHENTICATE, COMMUNICATION}
    for source_id, (conflict,) in conflicts.items():
        own = [(text, meta) for text, meta in flagged if meta["source_id"] == source_id]
        assert own and all(meta["source_conflict"] == conflict.topic and conflict.matches(text) for text, meta in own)
        assert all(meta["source_conflict"] is None for text, meta in by_source[source_id] if not conflict.matches(text))
    journal = [meta["page"] for text, meta in flagged if meta["source_id"] == COMMUNICATION]
    assert set(journal) == {5}                                      # the journal section, and only it
    for source_id in conflicts:
        own = sum(1 for _, meta in flagged if meta["source_id"] == source_id)
        assert next(s for s in parts.per_source if s["source_id"] == source_id)["conflict_chunks"] == own


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
def test_approve_sources_admits_only_the_named_extra_sources_and_keeps_the_handbook_constraints():
    from handbook_bot.agents.specialists.scoping import approve_sources, approved_source_set

    scope = RetrievalScope(source_ids=[HANDBOOK], page_ranges=[(56, 74)])
    approved = approved_source_set([HANDBOOK, GRADEBOOK], HANDBOOK)
    extended = approve_sources(scope, approved, HANDBOOK)
    assert extended.also_source_ids == frozenset({GRADEBOOK})
    assert extended.allows({"source_id": GRADEBOOK, "page": 6}) and not scope.allows({"source_id": GRADEBOOK, "page": 6})
    assert extended.allows({"source_id": HANDBOOK, "page": 60}) and not extended.allows({"source_id": HANDBOOK, "page": 10})
    assert not extended.allows({"source_id": SHORT_GUIDE, "page": 6})
    assert "also_source_ids" not in scope.describe() and extended.describe()["also_source_ids"] == [GRADEBOOK]
    assert approve_sources(scope, approved_source_set(None, HANDBOOK), HANDBOOK) is scope      # I-9 scope object reproduced


@pytest.mark.parametrize("bad, message", [
    ("uos_blackboard_gradebook_guide", "not a single string"),
    ([HANDBOOK, "*"], "plain source ids"),
    ([HANDBOOK, ""], "plain source ids"),
    ([HANDBOOK, " " + GRADEBOOK], "plain source ids"),
    ([GRADEBOOK], "must include the primary source"),
])
def test_approved_source_set_has_no_wildcard_and_always_keeps_the_handbook(bad, message):
    from handbook_bot.agents.specialists.scoping import approved_source_set

    with pytest.raises(ValueError, match=message):
        approved_source_set(bad, HANDBOOK)
    assert approved_source_set(None, HANDBOOK) == frozenset({HANDBOOK})


def test_domain_policy_is_read_from_the_records_and_ignores_disabled_ones(registry):
    policy = SS.domain_sources(registry.list_sources())
    assert policy["teaching"] == frozenset({HANDBOOK, SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER, *P1_GUIDES})
    assert policy["research"] == frozenset({HANDBOOK, ITHENTICATE})
    assert policy["faculty_services"] == frozenset({HANDBOOK})          # the appraisal guide is on hold
    assert policy["institutional"] == frozenset({HANDBOOK, SHORT_GUIDE, GRADE_TRANSFER})


def test_source_policy_is_the_d110_map_and_needs_both_the_allowlist_and_the_registry(registry):
    policy = SS.source_policy(registry.list_sources())
    assert policy == {"teaching": frozenset({HANDBOOK, SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER, *P1_GUIDES}),
                      "research": frozenset({HANDBOOK, ITHENTICATE}),
                      "faculty_services": frozenset({HANDBOOK}),
                      "institutional": frozenset({HANDBOOK, SHORT_GUIDE, GRADE_TRANSFER})}
    assert all(not ids & {APPRAISAL} for ids in policy.values())                  # the held document is never approved
    with pytest.raises(ValueError, match="unknown specialist"):
        SS.approved_sources_for(registry.list_sources(), "everyone")


def _record(source_id, **kw):
    base = dict(source_id=source_id, title="T", relative_path="data/guides/x.pdf", source_type="pdf", document_type="system_guide",
                domains=("teaching",))
    base.update(kw)
    return SourceRecord(**base)


@pytest.mark.parametrize("record, why", [
    (_record("uos_unlisted_vendor_guide"), "not in the code allow-list"),
    (_record(GRADEBOOK, enabled=False), "disabled"),
    (_record(GRADEBOOK, access="internal_use"), "internal use"),
    (_record(GRADEBOOK, verification_status="INGESTION HOLD - REVIEW"), "on hold"),
    (_record(GRADEBOOK, domains=("research",)), "registry does not list the specialist"),
])
def test_unknown_disabled_held_or_unlisted_sources_are_never_approved(record, why):
    assert SS.approved_sources_for([record], "teaching") == frozenset({HANDBOOK}), why
    assert SS.approved_sources_for([_record(GRADEBOOK)], "teaching") == frozenset({HANDBOOK, GRADEBOOK})
    assert SS.approved_sources_for([_record(GRADEBOOK)], "research") == frozenset({HANDBOOK})      # allow-list narrows


def test_page_collisions_flag_a_guide_whose_pages_reach_into_a_specialist_scope(registry):
    scope = RetrievalScope(source_ids=[HANDBOOK], page_ranges=[(14, 60)])
    fake = {sid: SimpleNamespace(scope=scope) for sid in SS.SPECIALIST_IDS}
    records = registry.enabled_sources()
    assert SS.page_collisions(records, {GRADEBOOK: 10, SHORT_GUIDE: 11, GRADE_TRANSFER: 3, ITHENTICATE: 2, HANDBOOK: 272}, fake) == []
    hits = SS.page_collisions(records, {GRADEBOOK: 20}, fake)
    assert len(hits) == 4 and all("pages 14-20 of %s" % GRADEBOOK in h for h in hits)


def test_the_i9_registry_stays_handbook_only_and_the_guide_registry_is_built_beside_it(kb):
    base = PF.build_specialist_registry(kb)
    assert base.ok, base.errors
    assert base.scope_extension == {"collisions": []} and json.dumps(base.to_dict())
    for sid in SS.SPECIALIST_IDS:
        specialist = base.registry.resolve(sid)
        assert not specialist.scope.also_source_ids
        assert getattr(specialist, "approved_sources", frozenset({HANDBOOK})) == frozenset({HANDBOOK})
    off, off_build = PF.build_runtime(kb, guide_scopes=False)
    assert off.guide_registry is None and off_build.scope_extension["enabled"] is False
    on, on_build = PF.build_runtime(kb, guide_scopes=True)
    assert on_build.ok and on.guide_registry is not None and on.registry is not on.guide_registry
    report = on_build.scope_extension
    assert report["enabled"] and report["collisions"] == [] and report["guide_registry"]["ok"]
    assert report["policy"]["research"] == sorted([HANDBOOK, ITHENTICATE]) and report["policy"]["faculty_services"] == [HANDBOOK]
    guides = on.guide_registry
    assert guides.resolve("teaching").approved_sources == frozenset({HANDBOOK, SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER, *P1_GUIDES})
    assert guides.resolve("research").approved_sources == frozenset({HANDBOOK, ITHENTICATE})
    assert guides.resolve("institutional").approved_sources == frozenset({HANDBOOK, SHORT_GUIDE, GRADE_TRANSFER})
    assert not hasattr(guides.resolve("faculty_services"), "approved_sources")                  # Faculty Services & HR untouched
    # PF-3C (D-111): the guide registry wraps the SAME I-9 specialist objects; their extraction stays handbook-only and guide
    # evidence enters only through the D-111 guide path, for exactly the D-110 approved guides
    from handbook_bot.agents.specialists.guide_evidence import GuideCapableSpecialist
    assert guides.resolve("teaching").approved_guides == frozenset({SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER, *P1_GUIDES})
    assert guides.resolve("research").approved_guides == frozenset({ITHENTICATE})
    assert guides.resolve("faculty_services") is on.registry.resolve("faculty_services")
    for sid in ("teaching", "research", "institutional"):
        wrapper = guides.resolve(sid)
        assert isinstance(wrapper, GuideCapableSpecialist) and wrapper.base is on.registry.resolve(sid), sid
        assert not wrapper.scope.allows({"source_id": GRADEBOOK, "page": 6}) and not wrapper.scope.also_source_ids, sid
        assert wrapper.scope.page_ranges == base.registry.resolve(sid).scope.page_ranges, sid      # handbook constraints unchanged
        assert getattr(wrapper.base, "approved_sources", frozenset({HANDBOOK})) == frozenset({HANDBOOK}), sid
    assert on.guide_index is not None and on.guide_index.policy["research"] == frozenset({HANDBOOK, ITHENTICATE})
    assert off.guide_index is None


def test_the_guide_registry_is_used_only_for_a_question_that_names_a_guide_system(kb):
    runtime, _ = PF.build_runtime(kb, guide_scopes=True)
    for question, named in [
        ("How do I post grades in the Blackboard gradebook?", [SHORT_GUIDE, GRADEBOOK, *P1_GUIDES]),
        ("Where are the recordings of my Collaborate sessions?", [COLLABORATE]),                # post-freeze PF-4
        ("How do I collaborate with colleagues on a research grant?", []),                    # the verb names no system
        ("How do I transfer grades to Banner?", [GRADE_TRANSFER]),
        ("How do I access iThenticate?", [ITHENTICATE]),
        ("What is the minimum size of a Blackboard course banner image?", [SHORT_GUIDE, GRADEBOOK, *P1_GUIDES]),   # not Banner
        ("Can I upload a banner image for my course?", []),
        ("How many credit hours is a full teaching load?", []),
        ("How do I start the Phase 2 progress review in the HR Portal?", []),          # the held appraisal guide has no scope
        ("What is the grading policy for final examinations?", []),
    ]:
        registry, record = runtime.select_registry(question)
        assert record == {"enabled": True, "selected": bool(named), "named": named}, question
        assert registry is (runtime.guide_registry if named else runtime.registry), question
    plain, _ = PF.build_runtime(kb, guide_scopes=False)
    assert plain.select_registry("How do I access iThenticate?") == (plain.registry, {"enabled": False, "selected": False, "named": []})


def test_the_section_guard_is_unchanged_for_the_handbook_and_never_attributes_a_guide_chunk(kb):
    base = PF.build_specialist_registry(kb).registry
    _, built = PF.build_runtime(kb, guide_scopes=True)
    guides, _ = PF.build_runtime(kb, guide_scopes=True)
    for sid in ("teaching", "research", "institutional"):
        old, new = base.resolve(sid).guard, guides.guide_registry.resolve(sid).guard
        assert sorted(old.shared_pages) == sorted(new.shared_pages) and old.unresolved_pages == new.unresolved_pages, sid
        if new.shared_pages:
            page = sorted(new.shared_pages)[0]
            fake_guide = {"source_id": GRADEBOOK, "page": page, "chunk_type": "row", "row_id": 0}
            assert new.is_shared(fake_guide) is False and new.owned_spans(0, "text", fake_guide) is None
            handbook_meta = {"source_id": HANDBOOK, "page": page}
            assert new.is_shared(handbook_meta) is True


def _cand(module, kb, index, quote=None):
    meta = kb.metadata[index]
    item = {"meta": meta, "chunk": kb.chunks[index], "rerank_score": 1.0, "idx": index}
    quote = quote or kb.chunks[index][:60]
    if module == "teaching":
        from handbook_bot.agents.specialists.teaching import _Candidate
        return _Candidate(0, item, quote, 1, 1, None, frozenset())
    from handbook_bot.agents.specialists.research import _Candidate
    return _Candidate(0, item, quote, 1, 1, frozenset())


def _first(kb, source_id, chunk_type="paragraph"):
    return next(i for i, m in enumerate(kb.metadata) if m["source_id"] == source_id and m["chunk_type"] == chunk_type)


def _reopened(kb, sid):
    """A specialist built with its D-110 approved source set through the constructor option PF-3B added. Since PF-3C (D-111)
    the runtime no longer builds specialists this way (the guide registry puts the I-9 specialists behind the D-111 guide
    path); the option and its source check stay in the specialists and are tested here as the D-110 boundary itself."""
    from handbook_bot.agents.specialists.institutional import InstitutionalNavigationSpecialist
    from handbook_bot.agents.specialists.research import ResearchInnovationSpecialist
    from handbook_bot.agents.specialists.teaching import TeachingLearningSpecialist

    cls = {"teaching": TeachingLearningSpecialist, "research": ResearchInnovationSpecialist, "institutional": InstitutionalNavigationSpecialist}[sid]
    return cls.from_knowledge_base(kb, approved_sources=SS.source_policy(kb.sources)[sid])


@pytest.mark.parametrize("sid, accepted, refused", [
    ("teaching", [SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER], [ITHENTICATE]),
    ("research", [ITHENTICATE], [SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER]),
])
def test_the_specialist_guard_accepts_exactly_its_approved_sources(kb, sid, accepted, refused):
    from handbook_bot.agents.specialists.contracts import SpecialistTask

    base = PF.build_specialist_registry(kb).registry.resolve(sid)
    reopened = _reopened(kb, sid)
    task = SpecialistTask(task_id="t-1", question="q", specialist_id=base.specialist_id, domain=sid, context={"focus_clauses": ["q"]})
    for source_id in accepted:
        cand = _cand(sid, kb, _first(kb, source_id))
        finding = reopened._finding(task, 1, cand, 0, "policy", [])
        assert finding.source_id == source_id and finding.source_title == kb.metadata[_first(kb, source_id)]["source_title"]
        assert finding.metadata["shared_page"] is False                         # no handbook section attribution
        with pytest.raises(ValueError, match="outside the approved scope"):
            base._finding(task, 1, cand, 0, "policy", [])                      # the I-9 boundary is untouched
    for source_id in refused + ["uos_unregistered_source"]:
        cand = _cand(sid, kb, _first(kb, source_id) if source_id != "uos_unregistered_source" else 0)
        if source_id == "uos_unregistered_source":
            cand.item["meta"] = dict(cand.item["meta"], source_id=source_id)
        with pytest.raises(ValueError, match="outside the approved scope"):
            reopened._finding(task, 1, cand, 0, "policy", [])


def test_institutional_statements_come_only_from_its_approved_sources(kb):
    base = PF.build_specialist_registry(kb).registry.resolve("institutional")
    reopened = _reopened(kb, "institutional")
    old = {(s.index, s.quote_start) for s in base.statements}
    new = {(s.index, s.quote_start) for s in reopened.statements}
    assert old <= new
    sources = {kb.metadata[i]["source_id"] for i, _ in new - old}
    assert sources <= {SHORT_GUIDE, GRADE_TRANSFER}
    assert {kb.metadata[r.index]["source_id"] for rows in reopened.directory.values() for r in rows} == {HANDBOOK}   # directory: handbook only


def test_a_guide_whose_pages_overlap_a_specialist_scope_refuses_the_whole_registry(kb):
    fake_pages = dict(kb.source_pages)
    fake_pages[GRADEBOOK] = [{"page": n} for n in range(1, 21)]
    shadow = SimpleNamespace(**{**kb.__dict__, "source_pages": fake_pages})
    build = PF.build_specialist_registry(shadow)
    assert not build.ok and build.registry is None and "scope_extension" in build.errors
    assert "overlap" in build.errors["scope_extension"] and build.scope_extension["collisions"]


def test_a_one_document_knowledge_base_is_left_alone_by_the_seam(kb):
    single = SimpleNamespace(**{**kb.__dict__, "sources": [], "source_pages": {}})
    runtime, build = PF.build_runtime(single, guide_scopes=True)
    assert build.ok and runtime.guide_registry is None and build.scope_extension == {"enabled": False}


# ---------------------------------------------------------------------------
# Source authority, the policy/system boundary and procedure completeness (PF-3B, D-110)
# ---------------------------------------------------------------------------
def _meta(source_id, **kw):
    doc = "handbook" if source_id == HANDBOOK else ("guideline" if source_id == ITHENTICATE else "system_guide")
    base = {"source_id": source_id, "document_type": doc, "authority_level": "university_policy" if source_id == HANDBOOK else "system_guide",
            "chunk_type": "paragraph"}
    base.update(kw)
    return base


@pytest.mark.parametrize("clause, meta, expected", [
    ("What is the minimum size of a Blackboard course banner image?", _meta(SHORT_GUIDE), None),
    ("What is the maximum file size for an iThenticate upload?", _meta(ITHENTICATE), None),
    ("Who approves the grade transfer to Banner?", _meta(GRADE_TRANSFER), None),
    ("What is the minimum size of a course banner image?", _meta(SHORT_GUIDE), "does not name the system"),
    ("What is the minimum size of a Blackboard course banner image?", _meta(GRADE_TRANSFER), "does not name the system"),
    ("What is the minimum size of a Blackboard course banner image?", _meta(ITHENTICATE), "does not name the system"),
    ("What grading policy applies to final examinations in Blackboard?", _meta(GRADEBOOK), "policy, entitlement or obligation"),
    ("Must instructors post grades in Blackboard within 48 hours?", _meta(GRADEBOOK), "policy, entitlement or obligation"),
    ("Am I required to use iThenticate for my thesis?", _meta(ITHENTICATE), "policy, entitlement or obligation"),
    ("What is the grading policy?", _meta(HANDBOOK), None),                              # handbook evidence is never refused here
])
def test_a_guide_answers_only_within_its_own_system_and_never_a_policy_ask(clause, meta, expected):
    problem = SS.source_authority_problem(clause, meta)
    assert (problem is None) if expected is None else (expected in problem), (clause, problem)


def test_a_procedure_ask_is_answered_by_the_whole_procedure_or_not_at_all(parts):
    windows = [(c, m) for c, m in zip(parts.chunks, parts.metadata) if m["chunk_type"] == "procedure_window"]
    full = next((c, m) for c, m in windows if m["step_start"] == 0 and m["step_end"] == m["step_count"] - 1)
    partial = next((c, m) for c, m in windows if m["step_end"] < m["step_count"] - 1)
    step = next((c, m) for c, m in zip(parts.chunks, parts.metadata) if m["chunk_type"] == "procedure_step" and m["step_count"] > 1)
    para = next((c, m) for c, m in zip(parts.chunks, parts.metadata) if m["chunk_type"] == "paragraph" and m["source_id"] == SHORT_GUIDE)
    ask = "How do I access iThenticate?"
    assert SS.procedure_problem(ask, full[0], full[0], full[1]) is None
    assert "leaves out" in SS.procedure_problem(ask, full[0].split(" | ")[0], full[0], full[1])
    assert "not the whole procedure" in SS.procedure_problem(ask, partial[0], partial[0], partial[1])
    assert "one step of a" in SS.procedure_problem(ask, step[0], step[0], step[1])
    assert "complete procedure is not established" in SS.procedure_problem(ask, para[0], para[0], para[1])
    assert SS.procedure_problem("What is the minimum size of a Blackboard image?", para[0], para[0], para[1]) is None   # not a procedure ask
    assert SS.procedure_problem(ask, "x", "x", _meta(HANDBOOK)) is None                                               # handbook untouched


def _stub_runtime(kb, sid, cid, quote):
    """The guide-enabled runtime with ``sid`` replaced by a stub that returns one supported finding on chunk ``cid``."""
    from handbook_bot.agents.specialists.contracts import Finding, FindingStatus, SpecialistFindings, SpecialistId

    runtime, build = PF.build_runtime(kb, guide_scopes=True)
    meta = kb.metadata[cid]

    def run(task):
        finding = Finding(finding_id="%s-f1" % task.task_id, claim=quote, evidence_quote=quote, source_id=meta["source_id"],
                          source_title=meta["source_title"], page=meta["page"], confidence=0.9,
                          metadata={"chunk_id": cid, "clauses": [0], "details_covered": ["number or amount"]})
        clause = task.context["focus_clauses"][0]
        return SpecialistFindings(specialist_id=SpecialistId.parse(sid), task_id=task.task_id, status=FindingStatus.SUPPORTED,
                                  findings=[finding], summary="stub", metadata={"clauses": [{"index": 0, "clause": clause, "outcome": "supported",
                                                                                          "ask_shape": "specific_recognised",
                                                                                          "requested_details": ["number or amount"],
                                                                                          "uncovered_details": [], "route": "policy"}]})

    runtime.guide_registry.resolve(sid).run = run
    return runtime


def test_the_adapter_refuses_a_guide_finding_outside_its_system_or_on_a_policy_ask(kb):
    cid = _first(kb, GRADE_TRANSFER)
    quote = kb.chunks[cid][:80]
    runtime = _stub_runtime(kb, "teaching", cid, quote)
    attempt = runtime.attempt("What is the minimum size of a Blackboard course banner image?")
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY and "does not name the system" in attempt.reason
    assert attempt.guide_scope == {"enabled": True, "selected": True, "named": [SHORT_GUIDE, GRADEBOOK, *P1_GUIDES]}
    cid = _first(kb, GRADEBOOK)
    runtime = _stub_runtime(kb, "teaching", cid, kb.chunks[cid][:80])
    attempt = runtime.attempt("What grading policy applies to the Blackboard gradebook?")
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY and "policy" in attempt.reason
    assert attempt.result is None and any(e.agent == "source_authority_guard" for e in attempt.trace)


def test_the_adapter_refuses_one_step_as_a_procedure(kb):
    cid = next(i for i, m in enumerate(kb.metadata) if m["source_id"] == SHORT_GUIDE and m["chunk_type"] == "procedure_step" and m["step_count"] > 2)
    runtime = _stub_runtime(kb, "teaching", cid, kb.chunks[cid])
    attempt = runtime.attempt("How do I access Blackboard from outside the campus?")
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY and "one step of a" in attempt.reason


@pytest.mark.parametrize("question", [
    "How many documents can I submit to iThenticate per year?",
    "What is the yearly iThenticate submission quota?",
    "Is the iThenticate limit 30 documents per year?",
    "Is the iThenticate limit 15 documents per year?",
    "How do I upload a document to iThenticate and how many documents can I submit per year?",
])
def test_the_disputed_ithenticate_quota_is_never_served_through_the_guide_registry(kb, question):
    cid = next(i for i, m in enumerate(kb.metadata) if m.get("source_conflict") and m["chunk_type"] == "row")
    runtime = _stub_runtime(kb, "research", cid, kb.chunks[cid])
    attempt = runtime.attempt(question)
    assert attempt.outcome is not PF.PlanFOutcome.ACCEPTED and attempt.result is None, question
    if attempt.dispatch is not None and any(r.findings is not None and r.findings.findings for r in attempt.dispatch.runs):
        assert attempt.outcome in (PF.PlanFOutcome.REJECTED_BY_SOURCE_CONFLICT, PF.PlanFOutcome.REJECTED_BY_AGGREGATOR), (question, attempt.outcome)


@pytest.mark.parametrize("question, owner", [
    ("How do I access iThenticate?", "research"),
    ("How do I upload a document to iThenticate?", "research"),
    ("What is the maximum file size for an iThenticate upload?", "research"),
    ("How do I check my research manuscript for plagiarism?", "research"),
    ("How do I check a student's assignment for plagiarism with SafeAssign in Blackboard?", "teaching"),
    ("Can students check their course assignments with Turnitin?", "teaching"),
])
def test_ithenticate_routes_to_research_and_course_originality_stays_with_teaching(question, owner):
    from handbook_bot import plan_f_routing as RT
    from handbook_bot.agents.coordinator import coordinate

    refined, _ = RT.refine(coordinate(question))
    assert [s.value for s in refined.selected_specialists] == [owner], question


def test_source_scope_diagnostics_carry_ids_and_flags_only():
    attempt = SimpleNamespace(outcome=SimpleNamespace(value="PLAN_F_UNSAFE_RESULT"), reason="r", specialist_id="teaching", status="partial",
                              result=None, timings={}, dispatch=None, trace=[], path=None, verification=None, aggregation=None,
                              routing=None, handoff=None, handoff_outcome=None, specialist_calls=1,
                              guide_scope={"enabled": True, "selected": True, "named": [GRADEBOOK]})
    d = UP._diagnose("active", "legacy", attempt, None, None, legacy=None)
    assert d["guide_scope_enabled"] is True and d["guide_scope_selected"] is True and d["guide_sources_named"] == [GRADEBOOK]
    assert d["plan_f_guide_served"] is False and json.dumps(d)
    rows = dict(next(rows for title, rows in P.dev_sections(_result([], []), d) if title == "Routing"))
    assert rows["Source scope"] == "handbook + guides: %s" % GRADEBOOK
    d.update(guide_scope_selected=False, guide_sources_named=[])
    assert dict(next(r for t, r in P.dev_sections(_result([], []), d) if t == "Routing"))["Source scope"] == "handbook only (no guide system named)"


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
