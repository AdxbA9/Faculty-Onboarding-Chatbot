"""
Post-freeze PF-4 (decision record D-113): the first P1 guide batch and the held-out evaluation set.

Offline: the real eight-source corpus without a model. The three activated P1 guides (attendance tool, Collaborate Ultra,
communication tools) are registered completely, approved for Teaching & Learning only, English throughout, below the
frozen page-keyed indexes, and pass through the same D-111/D-112 rules as the P0 guides. Two P1 candidates stay on hold
and are never registered. The held-out question set is frozen: its wording is pinned by the hash recorded when it was
written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from handbook_bot import guide_grounding as GG
from handbook_bot import guide_structure as GS
from handbook_bot import source_scopes as SS
from handbook_bot.knowledge_base import arabic_share, assemble_corpus
from handbook_bot.sources import file_sha256, load_source_registry

REPO = Path(__file__).resolve().parents[1]
HANDBOOK = "uos_faculty_handbook_2025_26"
ATTENDANCE = "uos_blackboard_attendance_guide"
COLLABORATE = "uos_blackboard_collaborate_guide"
COMMUNICATION = "uos_blackboard_communication_guide"
P1 = (ATTENDANCE, COLLABORATE, COMMUNICATION)
P0 = ("uos_blackboard_faculty_short_guide", "uos_grade_transfer_dashboard_guide", "uos_blackboard_gradebook_guide", "uos_ithenticate_guidelines")
PF1_SHA = {ATTENDANCE: "Attendance_En.pdf", COLLABORATE: "bb_collab_en.pdf", COMMUNICATION: "bb_commun_en.pdf"}


@pytest.fixture(scope="module")
def registry():
    return load_source_registry(str(REPO / "knowledge" / "sources.json"))


@pytest.fixture(scope="module")
def parts(registry):
    return assemble_corpus(registry.enabled_sources())


@pytest.fixture(scope="module")
def index(parts):
    return GG.GuideIndex(parts.chunks, parts.metadata, parts.source_pages, parts.records)


def _worst_case(index, parts, clause, source_id):
    """Every chunk of the source counts as retrieved: what the selection could ever accept."""
    binding, problem = GG.bind(clause, index.record(source_id))
    if binding is None:
        return None, [{"reason": problem}]
    units = []
    for cid, meta in enumerate(parts.metadata):
        if meta.get("source_id") == source_id:
            units.extend(GG.statement_units(cid, parts.chunks[cid], meta, 0, binding))
    return GG.select_answer(binding, units, index, {p.key: 0 for p in index.procedures_of(source_id)},
                            {s.key: 0 for s in index.structures_of(source_id)})


# ---------------------------------------------------------------------------
# Registry, files, pages, scopes
# ---------------------------------------------------------------------------
def test_the_three_p1_records_are_complete_official_guides_for_teaching_only(registry):
    inventory = {row["filename"]: row["sha256"] for row in __import__("csv").DictReader(
        open(REPO / "qa" / "post_freeze_pf1" / "document_inventory.csv", encoding="utf-8"))}
    for source_id in P1:
        record = registry.get_source(source_id)
        assert record.enabled and record.access == "public" and record.verification_status == "pf4_approved_p1"
        assert record.document_type == "system_guide" and record.authority_level == "system_guide" and record.authority == "official_university_source"
        assert record.department == "ITC Academic Computing" and record.language == "en" and record.page_filter == "english_only"
        assert record.audience == ("faculty", "instructor") and record.domains == ("teaching",) and record.citation_label == record.title
        assert record.relative_path == "data/guides/%s.pdf" % source_id and record.publication_date and record.version.startswith("file dated")
        assert file_sha256(record.absolute_path()) == record.sha256 == inventory[PF1_SHA[source_id]]      # the PF-1 file, unchanged


def test_held_p1_candidates_are_never_registered_and_the_appraisal_guide_stays_held(registry):
    text = (REPO / "knowledge" / "sources.json").read_text(encoding="utf-8").lower()
    for held in ("bb_assignment_en", "assignments and test", "grade_category", "grade categories"):
        assert held not in text
    assert not any("assignment" in p.name or "categor" in p.name for p in (REPO / "data" / "guides").iterdir())
    appraisal = registry.get_source("uos_performance_appraisal_phase2_guide")
    assert not appraisal.enabled and appraisal.access == "internal_use" and appraisal.audience == ()
    assert len(registry.enabled_sources()) == 8
    assert all(appraisal.source_id not in ids for ids in SS.source_policy(registry.list_sources()).values())


def test_the_p1_guides_are_english_and_below_the_frozen_page_keyed_indexes(parts):
    pages = {k: [p["page"] for p in v] for k, v in parts.source_pages.items()}
    assert pages[ATTENDANCE] == [1, 2] and pages[COLLABORATE] == [1, 2, 3, 4] and pages[COMMUNICATION] == [1, 2, 3, 4, 5]
    for source_id in P1:
        texts = [c for c, m in zip(parts.chunks, parts.metadata) if m["source_id"] == source_id]
        assert texts and all(arabic_share(t) == 0.0 for t in texts)
    # the lowest handbook page any specialist scope covers is 14: a P1 guide must stay below it (D-113 hold reason)
    scope = SimpleNamespace(scope=SimpleNamespace(page_ranges=((14, 272),)))
    stub = SimpleNamespace(get=lambda sid: scope)
    records = [r for r in parts.records if r.source_id in P1]
    assert SS.page_collisions(records, {sid: len(pages[sid]) for sid in P1}, stub) == []
    longer = SS.page_collisions(records, {ATTENDANCE: 25}, stub)            # a 25-page guide (the held Assignments guide) collides
    assert longer and "pages 14-25" in longer[0]


def test_the_p1_guides_are_approved_for_teaching_only(registry):
    policy = SS.source_policy(registry.list_sources())
    assert set(P1) <= policy["teaching"]
    for specialist in ("research", "faculty_services", "institutional"):
        assert not set(P1) & policy[specialist], specialist


@pytest.mark.parametrize("question, named", [
    ("How do I schedule a session in Blackboard Collaborate?", None),          # Blackboard names every Blackboard guide
    ("Where are the recordings of my Collaborate sessions?", [COLLABORATE]),
    ("How do I open the Collaborate room?", [COLLABORATE]),
    ("How do I collaborate with colleagues on a research grant?", []),       # the verb names no system
    ("What is the student attendance policy?", []),                           # policy wording names no guide
    ("Which journals count for promotion?", []),
])
def test_system_terms_name_the_p1_guides_only_by_their_systems(registry, question, named):
    found = SS.guide_sources_named(question, registry.list_sources())
    if named is None:
        assert set(P1) <= set(found)
    else:
        assert found == named


# ---------------------------------------------------------------------------
# D-111 / D-112 on the P1 guides
# ---------------------------------------------------------------------------
def test_the_journal_recommendation_conflict_blocks_every_journal_answer(index, parts):
    for clause in ("How do I create a journal in Blackboard Ultra?", "Can I grade a journal in Blackboard Ultra?"):
        answer, considered = _worst_case(index, parts, clause, COMMUNICATION)
        assert answer is None
        assert any("source conflict" in str(c.get("reason")) for c in considered), clause
    stamped = {m["page"] for m in parts.metadata if m["source_id"] == COMMUNICATION and m.get("source_conflict")}
    assert stamped == {5}


def test_a_labelled_sub_section_named_after_its_object_never_answers_alone(index, parts):
    details = next(s for s in index.structures_of(COLLABORATE) if s.kind == "section" and s.heading == "Session Details")
    binding, _ = GG.bind("How do I create a scheduled session in Blackboard Collaborate?", index.record(COLLABORATE))
    assert "labelled sub-section" in GG.structure_problem(details, binding, index)
    for clause in ("How do I create a scheduled session in Blackboard Collaborate?", "How do I make a Blackboard Collaborate session repeat every week?"):
        answer, _ = _worst_case(index, parts, clause, COLLABORATE)
        assert answer is None, clause


def test_an_action_headed_bulleted_list_is_one_ordered_procedure(index, parts):
    check = index.procedure("%s:4:0" % COLLABORATE)
    assert check.heading == "D- Checking Session Attendance" and set(check.labels) == {"-"} and check.sequential
    answer, tried = _worst_case(index, parts, "How do I export the attendance sheet for my Blackboard Collaborate sessions?", COLLABORATE)
    # post-freeze PF-5C (D-115): the D-111 procedure is refused because its labelled heading names another action ("Checking"),
    # and the D-112 section that states the asked action on its object ("export the attendance sheet") serves the same words,
    # each printed bullet on its own line
    assert any(t.get("procedure") == "%s:4:0" % COLLABORATE and t.get("reason") == "the procedure heading names another action than the one asked"
               for t in tried)
    assert answer is not None and answer.form == GG.FORM_SECTION and answer.pages == (4,)
    assert answer.parts[1].quote.endswith("Click on attendance tool.") and "export icon" in answer.parts[-1].quote
    assert GG.citation_heading(answer.record(0), index) == "D- Checking Session Attendance"


def test_instruction_detection_skips_printed_markers_and_d_labels_name_an_action():
    assert GS.is_instruction("- From the Details & Actions panel, Click on attendance tool.")
    assert GS.is_instruction("1. Click on Post.") and not GS.is_instruction("- You will have an overview for all sessions.")
    assert GG.is_action_heading("D- Checking Session Attendance") and not GG.is_action_heading("Session Details")


@pytest.mark.parametrize("clause", [
    "According to the Blackboard attendance tool, how many absences are allowed before a student fails?",
    "When are students marked late in Blackboard Collaborate attendance?",
    "Who approves the attendance records in Blackboard?",
])
def test_policy_timing_and_authority_asks_never_bind_a_p1_guide(index, clause):
    for source_id in P1:
        binding, problem = GG.bind(clause, index.record(source_id))
        assert binding is None and "not answered from a system guide" in problem, (source_id, problem)


def test_every_p1_structure_and_procedure_quotes_its_own_page_exactly(index):
    for source_id in P1:
        for structure in index.structures_of(source_id):
            for part in structure.parts:
                assert part.quote in index.chunks[part.chunk_id]
                assert index.metadata[part.chunk_id]["source_id"] == source_id and index.metadata[part.chunk_id]["page"] == part.page
        for proc in index.procedures_of(source_id):
            for step in proc.steps:
                assert step.quote in index.chunks[step.chunk_id] and index.metadata[step.chunk_id]["source_id"] == source_id


def test_p0_answers_are_unchanged_by_the_p1_rules(index, parts):
    """The PF-4 rules (ordered bullets in the D-111 index, labelled sub-sections) leave the P0 answers they served."""
    cases = [("How do I download the gradebook from Blackboard?", "uos_blackboard_gradebook_guide", GG.FORM_SECTION, (10,)),
             ("How do I send an announcement to my students in Blackboard?", "uos_blackboard_faculty_short_guide", GG.FORM_SECTION, (9,)),
             ("How do I transfer grades from Blackboard to Banner?", "uos_grade_transfer_dashboard_guide", GG.FORM_PROCEDURE, (2,))]
    for clause, source_id, form, pages in cases:
        binding, _ = GG.bind(clause, index.record(source_id))
        answer, _ = GG.select_answer(binding, [], index, {p.key: 0 for p in index.procedures_of(source_id)},
                                     {s.key: 0 for s in index.structures_of(source_id)})
        assert answer is not None and answer.form == form and answer.pages == pages, clause


# ---------------------------------------------------------------------------
# The held-out set
# ---------------------------------------------------------------------------
def test_the_held_out_questions_are_frozen_and_fully_labelled():
    path = REPO / "qa" / "post_freeze_pf4" / "held_out_questions.jsonl"
    labels = json.loads((REPO / "qa" / "post_freeze_pf4" / "held_out_labels.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == labels["frozen"]["sha256"]      # wording never edited after freezing
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(questions) == labels["frozen"]["count"] == 50 and {q["id"] for q in questions} == set(labels["labels"])
    assert all(l["answerable"] in ("yes", "partial", "no") for l in labels["labels"].values())
    held = [l for l in labels["labels"].values() if "uos_performance_appraisal_phase2_guide" in l["expected_sources"]]
    assert not held                                                                           # the held guide is never expected
