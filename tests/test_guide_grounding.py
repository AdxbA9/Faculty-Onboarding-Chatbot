"""
Post-freeze PF-3C (decision record D-111): guide grounding, evidence selection and verified procedure assembly.

Offline. Pure tests of ``handbook_bot.guide_grounding`` on synthetic guide pages (built with the PF-2 chunker) and on the
real five-document corpus (no embedding model), plus end-to-end runs of the Plan F adapter on the real corpus with the
fake embedder and the raw-overlap reranker, exactly as ``tests/test_multi_source.py`` builds the runtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import pytest

from conftest import FakeEmbedder
from faculty_services_support import RawOverlapReranker
from handbook_bot import guide_grounding as GG
from handbook_bot import plan_f_runtime as PF
from handbook_bot import source_scopes as SS
from handbook_bot.agents.specialists.contracts import Finding, FindingStatus, SpecialistFindings, SpecialistId
from handbook_bot.chunking import build_chunks
from handbook_bot.knowledge_base import KnowledgeBase, assemble_corpus
from handbook_bot.sources import SourceConflict, SourceRecord, SourceRegistryError, load_source_registry
from ui import pipeline as UP
from ui import presentation as P

REPO = Path(__file__).resolve().parents[1]
HANDBOOK = "uos_faculty_handbook_2025_26"
SHORT_GUIDE = "uos_blackboard_faculty_short_guide"
GRADE_TRANSFER = "uos_grade_transfer_dashboard_guide"
GRADEBOOK = "uos_blackboard_gradebook_guide"
ITHENTICATE = "uos_ithenticate_guidelines"
APPRAISAL = "uos_performance_appraisal_phase2_guide"


# ---------------------------------------------------------------------------
# Synthetic guides
# ---------------------------------------------------------------------------
def _record(source_id=SHORT_GUIDE, **kw):
    base = dict(source_id=source_id, title="Test Guide", relative_path="data/guides/test.pdf", source_type="pdf", document_type="system_guide",
                authority_level="system_guide", system="Blackboard Ultra", access="public", enabled=True, domains=("teaching",),
                audience=("faculty", "instructor"), verification_status="verified_local_file")
    base.update(kw)
    return SourceRecord(**base)


def _index(pages, record=None, policy=None):
    """A GuideIndex over synthetic pages: ``pages`` is {page: (section hint, [rows])}."""
    record = record or _record()
    page_dicts = [{"page": n, "section_hint": hint, "rows": rows, "lines": rows, "text": "\n".join(rows)} for n, (hint, rows) in sorted(pages.items())]
    chunks, metadata = build_chunks(page_dicts, procedures=True)
    for meta in metadata:
        meta.update(source_id=record.source_id, document_type=record.document_type, authority_level="system_guide",
                    source_title=record.title)
    policy = policy or {"teaching": frozenset({HANDBOOK, record.source_id}), "research": frozenset({HANDBOOK}),
                        "faculty_services": frozenset({HANDBOOK}), "institutional": frozenset({HANDBOOK})}
    return GG.GuideIndex(chunks, metadata, {record.source_id: page_dicts}, [record], policy)


PAGES = {
    1: ("Course Tools Guide", [
        "Course Tools Guide",
        "How to publish a course:",
        "1. Open the course in Blackboard.",
        "2. Select Course Availability in the menu.",
        "3. Choose Open to publish the course to students.",
        "Your Courses:",
        "- You can mark a course as a favorite to pin it to the top of the course list.",
        "- You can change the course list view from the menu.",
        "From the Settings panel, you can:",
        "1. Change the grading schema.",
        "2. Turn on automatic zeros for missing work.",
        "Grade Columns:",
        "1. Select the plus sign to add a column.",
        "2. Name the column with a short title.",
    ]),
    2: ("Grade Columns:", [
        "3. Choose the grade unit for the column.",
        "4. Select Save to add the column to the gradebook.",
        "To export grades follow these steps:",
        "1. Open the gradebook and select Export.",
        "2. Choose the file type for the",
        "download format:",
    ]),
}


@pytest.fixture(scope="module")
def index():
    return _index(PAGES)


def _proc(index, heading):
    return next(p for p in index.procedures_of(SHORT_GUIDE) if p.heading == heading)


def _bind(clause, record=None):
    binding, problem = GG.bind(clause, record or _record())
    assert binding is not None, problem
    return binding


# ---------------------------------------------------------------------------
# Intent: the existing request reader, extended minimally
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause, intent", [
    ("How do I publish a course in Blackboard?", GG.GuideIntent.PROCEDURE),
    ("How can I pin a course in Blackboard?", GG.GuideIntent.PROCEDURE),
    ("What are the steps to add a column in Blackboard?", GG.GuideIntent.PROCEDURE),
    ("Where do I find the course gradebook in Blackboard?", GG.GuideIntent.LOCATION),
    ("Whom do I e-mail if I cannot access iThenticate?", GG.GuideIntent.CONTACT),
    ("What is the minimum size of a Blackboard banner image?", GG.GuideIntent.VALUE),
    ("How long can a Blackboard column title be?", GG.GuideIntent.VALUE),
    ("Can I add a time limit to a group assignment in Blackboard?", GG.GuideIntent.YES_NO),
    ("What can I change from the Settings panel in Blackboard?", GG.GuideIntent.ENUMERATION),
    ("Which file types can I upload to Blackboard?", GG.GuideIntent.ENUMERATION),
    ("Which button do I use in Blackboard to post grades?", GG.GuideIntent.SELECTION),
    ("Who approves the grade transfer to Banner?", GG.GuideIntent.AUTHORITY),
    ("When must final grades be transferred to Banner?", GG.GuideIntent.POLICY),
    ("What is the grading policy for Blackboard courses?", GG.GuideIntent.POLICY),
    ("What is the required teaching load for faculty who teach on Blackboard?", GG.GuideIntent.POLICY),
    ("Am I required to use iThenticate for my thesis?", GG.GuideIntent.POLICY),
])
def test_guide_intent_extends_the_request_reader(clause, intent):
    assert GG.read_intent(clause)[0] is intent
    assert (intent in GG.ANSWERABLE_INTENTS) is (intent not in (GG.GuideIntent.AUTHORITY, GG.GuideIntent.POLICY))


# ---------------------------------------------------------------------------
# D-111 A/B: system and faculty-actor binding
# ---------------------------------------------------------------------------
def test_the_declared_system_and_the_faculty_actor_are_bound_by_the_registry():
    b = _bind("How do I transfer grades from Blackboard to Banner?", _record(GRADE_TRANSFER, system="Grade Transfer Dashboard"))
    assert set(b.names_bound) == {"Blackboard", "Banner"} and b.actor_bound
    assert b.terms == ("transfer", GG.stem("grades")) and b.core == ("transfer", GG.stem("grades"))
    assert not GG.missing_terms(b, "Click on Transfer to send the grades") and GG.missing_terms(b, "Click on Transfer")
    b = _bind("How do I add a column in Blackboard Ultra?")
    assert b.names_bound == ("Blackboard Ultra",) and "blackboard" not in b.terms and "ultra" not in b.terms


def test_the_system_must_be_named_and_metadata_alone_is_never_the_answer():
    assert GG.bind("How do I add a column?", _record())[0] is None                                       # no system named
    _, why = GG.bind("How do I add a column in Blackboard?", _record(ITHENTICATE, system="iThenticate"))
    assert "does not name the system" in why                                                              # another guide's system
    b = _bind("How do I add a column in Blackboard?")
    assert GG.missing_terms(b, "Blackboard Ultra Faculty Short Guide")                                    # the system name states nothing


def test_an_undeclared_name_is_never_bound():
    b = _bind("How do I log in to Banner Self Service?", _record(GRADE_TRANSFER, system="Grade Transfer Dashboard"))
    assert b.entities == ("Self Service",) and GG.missing_entities(b, "Use your credentials to log in")


@pytest.mark.parametrize("audience, ok", [(("faculty",), True), (("instructor",), True), (("faculty", "researcher"), True),
                                          (("student",), False), (("staff",), False), ((), False)])
def test_only_a_faculty_or_instructor_document_may_answer(audience, ok):
    binding, why = GG.bind("How do I publish a course in Blackboard?", _record(audience=audience))
    assert (binding is not None) is ok
    if not ok:
        assert "faculty or instructor audience" in why and GG.is_authority_problem(why)


def test_the_registry_rejects_an_unknown_audience():
    with pytest.raises(SourceRegistryError, match="audience"):
        _record(audience=("everyone",))
    with pytest.raises(SourceRegistryError, match="audience"):
        _record(audience="faculty")


@pytest.mark.parametrize("record", [
    _record(enabled=False), _record(access="internal_use"), _record(verification_status="INGESTION HOLD"),
    _record(source_id="unregistered_x"), _record(document_type="handbook"),
])
def test_an_unusable_or_policy_document_never_binds(record):
    assert GG.bind("How do I publish a course in Blackboard?", record)[0] is None


def test_policy_authority_and_handbook_subjects_never_bind():
    transfer = _record(GRADE_TRANSFER, system="Grade Transfer Dashboard")
    for clause, record in (("Who approves the grade transfer to Banner?", transfer), ("What grading policy applies to Blackboard courses?", _record()),
                           ("How many office hours must faculty hold on Blackboard?", _record()),
                           ("When must final grades be posted in Blackboard?", _record())):
        binding, why = GG.bind(clause, record)
        assert binding is None and "not answered from a system guide" in why, clause


def test_the_real_registry_declares_a_faculty_audience_for_every_enabled_guide():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    for record in registry.enabled_sources():
        assert SS.faculty_audience(record), record.source_id
    held = next(r for r in registry.records() if r.source_id == APPRAISAL) if hasattr(registry, "records") else None
    if held is not None:
        assert not held.enabled and held.audience == ()


# ---------------------------------------------------------------------------
# What one statement must show
# ---------------------------------------------------------------------------
def test_a_value_needs_a_number_with_its_unit_never_a_list_label():
    b = _bind("How long can a Blackboard column title be?")
    assert GG.statement_problem(b, "1. Name the column with a short title.") is not None                 # the label is no value
    assert GG.statement_problem(b, "The column title is limited to 255 characters.") is None
    b = _bind("What is the minimum size of a Blackboard banner image?")
    assert GG.statement_problem(b, "The banner image must be at least 1200 x 240 pixels.") is None
    assert "lower bound" in GG.statement_problem(b, "The banner image can't be larger than 25 KB.")
    b = _bind("What is the maximum size of a Blackboard banner image?")
    assert GG.statement_problem(b, "The banner image can't be larger than 25 KB.") is None


@pytest.mark.parametrize("clause, good, bad", [
    ("Can I add a time limit to a group assignment in Blackboard?", "You cannot add a time limit to group assignments.",
     "A time limit keeps every group assignment on track."),
    ("Where do I find the course gradebook in Blackboard?", "Access the course gradebook on the navigation bar.",
     "The course gradebook lists every grade."),
    ("Whom do I e-mail if I cannot access Blackboard?", "If you cannot access Blackboard, e-mail servicedesk@sharjah.ac.ae",
     "If you cannot access Blackboard, contact the service desk."),
    ("Which button do I use in Blackboard to post grades?", "To post grades, select the Post option in the column header.",
     "Grades are posted to students."),
    ("Which file types can I upload to Blackboard?", "Supported file types to upload include DOC, PDF, PPT and TXT.",
     "You can upload file types from your computer."),
])
def test_each_intent_needs_its_own_kind_of_statement(clause, good, bad):
    b = _bind(clause)
    assert GG.statement_problem(b, good) is None, clause
    assert GG.statement_problem(b, bad) is not None, clause


def test_a_procedure_ask_is_never_answered_by_one_statement():
    b = _bind("How do I publish a course in Blackboard?")
    assert "never by a single statement" in GG.statement_problem(b, "Choose Open to publish the course to students in Blackboard.")


def test_statement_units_are_exact_and_one_sentence_comes_before_two():
    chunk = "Add a time limit. A limit keeps students on track. At this time, you cannot add a time limit to group assignments. 4. Next step"
    units = GG.statement_units(7, chunk, {"source_id": SHORT_GUIDE, "page": 3, "chunk_type": "paragraph"}, 0)
    assert all(u.quote in chunk for u in units)
    kinds = [u.kind for u in units]
    assert kinds.index("sentences") > max(i for i, k in enumerate(kinds) if k == "sentence")
    assert "4. Next step" in [u.quote for u in units]                                                    # a list label opens its own unit
    assert not any(u.quote.endswith("4.") for u in units)
    b = _bind("Can I add a time limit to a group assignment in Blackboard?")
    assert next(u for u in units if GG.statement_problem(b, u.quote) is None).quote == \
        "At this time, you cannot add a time limit to group assignments."


def test_a_generic_introduction_is_flagged_by_deterministic_features():
    meta = {"source_id": SHORT_GUIDE, "chunk_type": "paragraph"}
    b = _bind("How do I publish a course in Blackboard?")
    assert GG.is_generic_introduction("Blackboard Ultra provides instructors with a modern teaching environment.", meta, b)
    assert not GG.is_generic_introduction("Select Open to publish the course in Blackboard.", meta, b)      # an action verb
    assert not GG.is_generic_introduction("Blackboard Ultra courses can be published when ready.", meta, b)  # an asked term
    assert not GG.is_generic_introduction("Blackboard intro", {"source_id": SHORT_GUIDE, "chunk_type": "procedure_step"}, b)


# ---------------------------------------------------------------------------
# Procedures: labels, heading provenance, pages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("labels, ok", [
    (("-", "-", "-"), True), (("1.", "2.", "3."), True), (("1.", "a)", "b)", "2."), True), (("A)", "B)", "C)"), True),
    (("a)", "b)"), True), (("1.", "3."), False), (("1.", "2.", "D)"), False), (("-", "1.", "2."), False), (("1.",), False),
    (("C)", "D)"), False), (("2.", "3."), False), (("1.", "a)", "c)"), False),
])
def test_only_one_ordered_list_shape_is_accepted(labels, ok):
    assert (GG.label_problem(labels) is None) is ok, labels


def test_the_index_reads_procedures_as_the_chunker_does_and_records_heading_provenance(index):
    assert index.skipped_pages == []
    publish = _proc(index, "How to publish a course")
    assert publish.provenance == GG.PROVENANCE_ROW and publish.labels == ("1.", "2.", "3.") and publish.sequential
    assert [index.metadata[s.chunk_id]["chunk_type"] for s in publish.steps] == ["procedure_step"] * 3
    settings = _proc(index, "Your Courses")
    assert not settings.sequential and settings.labels == ("-", "-")
    for proc in index.procedures_of(SHORT_GUIDE):
        for step in proc.steps:
            assert step.quote in index.chunks[step.chunk_id]


def test_a_numbered_list_continued_on_the_next_page_is_one_procedure_citing_both_pages(index):
    columns = _proc(index, "Grade Columns")
    assert columns.pages == (1, 2) and columns.labels == ("1.", "2.", "3.", "4.") and not columns.problems
    assert {s.page for s in columns.steps} == {1, 2}
    b = _bind("How do I add a column in Blackboard?")
    answer, _ = GG.select_answer(b, [], index, {columns.key: 0})
    assert answer is not None and answer.form == GG.FORM_PROCEDURE and answer.pages == (1, 2)
    parts = [(p.chunk_id, p.quote, p.page) for p in answer.parts]
    assert GG.verify_answer("How do I add a column in Blackboard?", answer.form, parts, answer.procedure_key, index, [SHORT_GUIDE]) is None


def test_a_list_that_may_continue_or_may_have_started_earlier_is_refused():
    pages = {1: ("Guide", ["How to export grades:", "1. Open the gradebook.", "2. Select Export."]),
             2: ("Guide", ["1. Choose a file type for the export.", "2. Select Download."])}
    idx = _index(pages)
    first, second = idx.procedures_of(SHORT_GUIDE)
    assert "may continue on page 2" in first.problems[0] and "may continue a list of page 1" in second.problems[0]
    b = _bind("How do I export grades in Blackboard?")
    assert GG.procedure_problem(first, b, idx) is not None and GG.select_answer(b, [], idx, {first.key: 0, second.key: 1})[0] is None


def test_a_step_cut_by_a_row_read_as_a_heading_is_completed_from_the_page_prose(index):
    export = _proc(index, "To export grades follow these steps")
    last = export.steps[-1]
    assert last.quote == "2. Choose the file type for the download format:" and index.metadata[last.chunk_id]["chunk_type"] == "paragraph"
    assert not export.problems


def test_a_cut_step_without_indexed_whole_text_is_refused():
    pages = {1: ("Guide", ["To export grades follow these steps:", "1. Open the gradebook and select Export.", "2. Choose the file type for the",
                           "download format:"])}
    idx = _index(pages)
    for i, meta in enumerate(idx.metadata):                         # remove the prose that holds the whole step
        if meta["chunk_type"] == "paragraph":
            idx.chunks[i] = "unrelated prose"
    rebuilt = GG.GuideIndex(idx.chunks, idx.metadata, {SHORT_GUIDE: [{"page": 1, "section_hint": "Guide", "rows": pages[1][1]}]}, [_record()])
    export = rebuilt.procedures_of(SHORT_GUIDE)[0]
    assert export.problems and "row read as a heading" in export.problems[0]


def test_a_page_whose_procedures_differ_from_the_indexed_chunks_gives_no_procedure():
    idx = _index(PAGES)
    cid = _proc(idx, "How to publish a course").steps[1].chunk_id
    idx.chunks[cid] = idx.chunks[cid] + " (edited)"
    rebuilt = GG.GuideIndex(idx.chunks, idx.metadata, {SHORT_GUIDE: [{"page": n, "section_hint": h, "rows": r} for n, (h, r) in PAGES.items()]},
                            [_record()])
    assert (SHORT_GUIDE, 1, "procedures differ from the indexed procedure chunks") in rebuilt.skipped_pages
    assert all(1 not in p.pages for p in rebuilt.procedures_of(SHORT_GUIDE))


# ---------------------------------------------------------------------------
# Selection and the independent verification
# ---------------------------------------------------------------------------
def test_a_complete_procedure_is_selected_before_any_statement_and_verified(index):
    clause = "How do I publish a course in Blackboard?"
    b = _bind(clause)
    publish = _proc(index, "How to publish a course")
    units = [GG.Unit(publish.steps[2].chunk_id, SHORT_GUIDE, 1, publish.steps[2].quote, "step", 0)]
    answer, considered = GG.select_answer(b, units, index, {publish.key: 3})
    assert answer.form == GG.FORM_PROCEDURE and answer.procedure_key == publish.key and len(answer.parts) == 3
    assert answer.parts[0].quote.startswith("How to publish a course: 1. Open")
    parts = [(p.chunk_id, p.quote, p.page) for p in answer.parts]
    assert GG.verify_answer(clause, answer.form, parts, answer.procedure_key, index, [HANDBOOK, SHORT_GUIDE]) is None
    rendered = GG.render_part(parts[0][1], {"form": GG.FORM_PROCEDURE, "part": 0})
    assert rendered == "How to publish a course:\n\n1. Open the course in Blackboard."


def test_an_unretrieved_procedure_is_never_selected(index):
    b = _bind("How do I publish a course in Blackboard?")
    assert GG.select_answer(b, [], index, {})[0] is None


@pytest.mark.parametrize("mutate, expected", [
    (lambda parts: parts[:-1], "not exactly the claimed procedure"),                     # a missing step
    (lambda parts: [parts[1], parts[0]] + parts[2:], "not exactly the claimed procedure"),   # reordered
    (lambda parts: parts + parts[-1:], "not exactly the claimed procedure"),              # a repeated step
    (lambda parts: [(parts[0][0], parts[0][1] + " (and more)", 1)] + parts[1:], "not in its chunk"),
    (lambda parts: [(parts[0][0], parts[0][1], 2)] + parts[1:], "page mismatch"),
])
def test_the_verification_refuses_any_procedure_that_is_not_exactly_complete(index, mutate, expected):
    clause = "How do I publish a course in Blackboard?"
    b = _bind(clause)
    publish = _proc(index, "How to publish a course")
    answer, _ = GG.select_answer(b, [], index, {publish.key: 0})
    parts = [(p.chunk_id, p.quote, p.page) for p in answer.parts]
    assert expected in GG.verify_answer(clause, GG.FORM_PROCEDURE, mutate(parts), publish.key, index, [SHORT_GUIDE])


def test_one_step_is_never_served_as_the_procedure(index):
    clause = "How do I publish a course in Blackboard?"
    publish = _proc(index, "How to publish a course")
    step = publish.steps[2]
    assert GG.verify_answer(clause, GG.FORM_STATEMENT, [(step.chunk_id, step.quote, 1)], None, index, [SHORT_GUIDE]) is not None
    assert GG.verify_answer(clause, GG.FORM_LIST_ITEM, [(step.chunk_id, step.quote, 1)], publish.key, index, [SHORT_GUIDE]) is not None


def test_one_item_of_an_unordered_list_answers_alone_only_when_it_alone_states_the_ask(index):
    clause = "How can I pin a course to the top of my course list in Blackboard?"
    b = _bind(clause)
    settings = _proc(index, "Your Courses")
    answer, _ = GG.select_answer(b, [], index, {settings.key: 0})
    assert answer.form == GG.FORM_LIST_ITEM and len(answer.parts) == 1 and "favorite" in answer.parts[0].quote
    part = answer.parts[0]
    assert GG.verify_answer(clause, GG.FORM_LIST_ITEM, [(part.chunk_id, part.quote, part.page)], settings.key, index, [SHORT_GUIDE]) is None
    other = settings.steps[1]
    assert GG.verify_answer(clause, GG.FORM_LIST_ITEM, [(other.chunk_id, other.quote, 1)], settings.key, index, [SHORT_GUIDE]) is not None


def test_a_heading_that_names_another_action_or_a_capability_list_is_no_procedure(index):
    b = _bind("How do I export a course in Blackboard?")
    assert "another action" in GG.procedure_problem(_proc(index, "How to publish a course"), b, index)
    b = _bind("How do I change the grading schema in Blackboard?")
    assert "list of what can be done" in GG.procedure_problem(_proc(index, "From the Settings panel, you can"), b, index)
    b = _bind("What can I change from the Settings panel in Blackboard?")                       # but it is the complete list asked for
    assert GG.procedure_problem(_proc(index, "From the Settings panel, you can"), b, index) is None
    assert GG.same_action("transfer", GG.stem("send")) and GG.same_action("access", "log") and not GG.same_action("export", "publish")


def test_a_fragment_heading_does_not_identify_a_procedure():
    pages = {1: ("Guide", ["You can grade coursework, manage", "items, and post grades from two views:", "1. List view.", "2. Grid view."])}
    idx = _index(pages)
    proc = idx.procedures_of(SHORT_GUIDE)[0]
    assert proc.heading.startswith("items,")
    assert "sentence fragment" in GG.procedure_problem(proc, _bind("How do I post grades in Blackboard?"), idx)


def test_a_registered_conflict_in_any_step_refuses_the_whole_procedure():
    record = _record(conflicts=[SourceConflict(topic="column_limit", phrases=("short title",), status="unresolved")])
    idx = _index(PAGES, record)
    b = _bind("How do I add a column in Blackboard?", record)
    columns = next(p for p in idx.procedures_of(SHORT_GUIDE) if p.heading == "Grade Columns")
    assert "source conflict" in GG.procedure_problem(columns, b, idx)
    parts = [(s.chunk_id, s.quote, s.page) for s in columns.steps]
    assert "source conflict" in GG.verify_answer("How do I add a column in Blackboard?", GG.FORM_PROCEDURE, parts, columns.key, idx, [SHORT_GUIDE])


def test_the_verification_refuses_an_unapproved_specialist_source_and_mixed_sources(index):
    clause = "How do I publish a course in Blackboard?"
    publish = _proc(index, "How to publish a course")
    answer, _ = GG.select_answer(_bind(clause), [], index, {publish.key: 0})
    parts = [(p.chunk_id, p.quote, p.page) for p in answer.parts]
    assert "not approved" in GG.verify_answer(clause, GG.FORM_PROCEDURE, parts, publish.key, index, [HANDBOOK])
    assert GG.is_authority_problem(GG.verify_answer(clause, GG.FORM_PROCEDURE, parts, publish.key, index, [HANDBOOK]))


# ---------------------------------------------------------------------------
# The real corpus (no model)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def registry():
    return load_source_registry(str(REPO / "knowledge" / "sources.json"))


@pytest.fixture(scope="module")
def parts(registry):
    return assemble_corpus(registry.enabled_sources())


@pytest.fixture(scope="module")
def real_index(parts):
    return GG.GuideIndex(parts.chunks, parts.metadata, parts.source_pages, parts.records)


def test_the_real_guides_index_without_skipped_pages_and_with_their_procedures(real_index):
    assert real_index.skipped_pages == [] and HANDBOOK not in real_index.procedures
    keys = {p.key: p for procs in real_index.procedures.values() for p in procs}
    access = keys["%s:1:0" % ITHENTICATE]
    assert access.heading == "How to access IThenticate" and access.labels == ("1.", "2.", "3.") and access.provenance == GG.PROVENANCE_LEAD_IN
    send = keys["%s:2:0" % GRADE_TRANSFER]
    assert send.steps[-1].quote.endswith("after the approval from Chair and Dean as below:") and not send.problems
    merged = next(p for p in real_index.procedures_of(GRADEBOOK) if p.pages == (2, 3))
    assert merged.labels[:6] == ("1.", "2.", "3.", "4.", "5.", "6.") and GG.label_problem(merged.labels) is None
    assert real_index.policy["teaching"] == frozenset({HANDBOOK, SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER})
    assert APPRAISAL not in real_index.records


def test_no_step_of_the_disputed_quota_can_be_quoted(real_index):
    quota = next(s for p in real_index.procedures_of(ITHENTICATE) for s in p.steps if "Maximum 30" in s.quote)
    assert real_index.conflict_problem(quota.chunk_id, quota.quote)
    requirements = real_index.procedure_of_chunk(quota.chunk_id)
    b = _bind("What are the file requirements in iThenticate?", real_index.record(ITHENTICATE))
    assert "source conflict" in GG.procedure_problem(requirements, b, real_index)


# ---------------------------------------------------------------------------
# The adapter, end to end on the real corpus (fake embedder, raw-overlap reranker)
# ---------------------------------------------------------------------------
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


@pytest.fixture(scope="module")
def runtime(kb):
    runtime, build = PF.build_runtime(kb, guide_scopes=True)
    assert build.ok and runtime.guide_registry is not None and runtime.guide_index is not None
    return runtime


def test_a_complete_guide_procedure_is_served_with_its_own_pages_and_records(runtime):
    attempt = runtime.attempt("How do I access iThenticate?")
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED, attempt.reason
    items = attempt.result.items
    assert [i["meta"]["source_id"] for i in items] == [ITHENTICATE] * 3 and attempt.result.pages == [1]
    records = [i["meta"]["plan_f"]["guide"] for i in items]
    assert [r["part"] for r in records] == [0, 1, 2] and {r["form"] for r in records} == {GG.FORM_PROCEDURE}
    assert attempt.result.answer.startswith("How to access IThenticate:\n\n1. Open www.ithenticate.com")
    assert attempt.result.llm_calls == 0 and not attempt.result.used_llm
    assert any(e.agent == "guide_guard" and e.decision == "passed" for e in attempt.trace)
    assert "access" not in json.dumps(records)                                                          # no word of the question


@pytest.mark.parametrize("question", [
    "How many documents can I submit to iThenticate per year?",
    "What is the yearly iThenticate submission quota?",
    "Is the iThenticate limit 30 documents per year?",
    "Is the iThenticate limit 15 documents per year?",
    "How do I upload a document to iThenticate and how many documents can I submit per year?",
])
def test_the_disputed_quota_is_never_served_after_the_d111_changes(runtime, question):
    attempt = runtime.attempt(question)
    if attempt.outcome is PF.PlanFOutcome.ACCEPTED:                                                     # only the upload clause may be answered
        assert "30" not in attempt.result.answer and "15" not in attempt.result.answer, question
        assert all(not i["meta"].get("source_conflict") for i in attempt.result.items)
    assert not attempt.result or "Maximum 30" not in attempt.result.answer


def _forged(runtime, sid, findings_spec, form, key=None, source=None):
    """Replace the guide-registry ``sid`` with a stub that returns D-111-looking findings (bypass attempts)."""
    meta_of = runtime.metadata

    def run(task):
        clause = task.context["focus_clauses"][0]
        findings = []
        for n, (cid, quote) in enumerate(findings_spec):
            record = {"version": GG.VERSION, "phase": GG.PHASE, "source_id": source or meta_of[cid]["source_id"], "form": form, "intent": "procedure",
                      "procedure_key": key, "part": n, "parts": len(findings_spec), "binding": {}}
            findings.append(Finding(finding_id="forged-%d" % n, claim=quote, evidence_quote=quote, source_id=meta_of[cid]["source_id"],
                                    source_title="t", page=meta_of[cid]["page"], metadata={"chunk_id": cid, "clauses": [0], GG.GUIDE_KEY: record}))
        report = {"index": 0, "clause": clause, "outcome": "supported", "findings": [f.finding_id for f in findings], "ask_shape": "specific_recognised",
                  "requested_details": ["x"], "uncovered_details": [], "relation": "strong", GG.GUIDE_KEY: {k: v for k, v in findings[0].metadata[GG.GUIDE_KEY].items() if k != "part"}}
        return SpecialistFindings(specialist_id=SpecialistId.parse(sid), task_id=task.task_id, status=FindingStatus.SUPPORTED, findings=findings,
                                  summary="forged", metadata={"clauses": [report]})

    runtime.guide_registry.resolve(sid).run = run


def _steps(runtime, key):
    proc = runtime.guide_index.procedure(key)
    first = "%s: %s" % (proc.heading, proc.steps[0].quote)
    return [(proc.steps[0].chunk_id, first)] + [(s.chunk_id, s.quote) for s in proc.steps[1:]]


def test_forged_guide_answers_are_refused_by_the_adapter(kb):
    access = "%s:1:0" % ITHENTICATE
    cases = [
        ("research", lambda s: s[:-1], GG.FORM_PROCEDURE, access, PF.PlanFOutcome.REJECTED_BY_SUBJECT_GUARD),        # a step missing
        ("research", lambda s: [s[1], s[0], s[2]], GG.FORM_PROCEDURE, access, PF.PlanFOutcome.REJECTED_BY_SUBJECT_GUARD),
        ("research", lambda s: s[2:], GG.FORM_STATEMENT, None, PF.PlanFOutcome.REJECTED_BY_SUBJECT_GUARD),      # one step as the answer
        ("teaching", lambda s: s, GG.FORM_PROCEDURE, access, PF.PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY),     # an unapproved specialist
    ]
    for sid, mutate, form, key, expected in cases:
        runtime, _ = PF.build_runtime(kb, guide_scopes=True)
        _forged(runtime, sid, mutate(_steps(runtime, access)), form, key)
        question = "How do I access iThenticate?" if sid == "research" else "How do I access iThenticate in Blackboard?"
        attempt = runtime.attempt(question)
        if sid == "teaching" and attempt.specialist_id != "teaching":
            continue                                                                                        # routed away: never served by it
        assert attempt.outcome is expected and attempt.result is None, (sid, form, attempt.outcome, attempt.reason)


def test_a_forged_record_on_the_disputed_quota_is_stopped_by_the_conflict_guard(kb):
    runtime, _ = PF.build_runtime(kb, guide_scopes=True)
    quota = next(i for i, m in enumerate(runtime.metadata) if m.get("source_conflict") and m["chunk_type"] == "procedure_step")
    _forged(runtime, "research", [(quota, runtime.chunks[quota])], GG.FORM_STATEMENT)
    attempt = runtime.attempt("How many documents can I submit to iThenticate per year?")
    assert attempt.result is None
    if attempt.specialist_id == "research":
        assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SOURCE_CONFLICT


def test_guide_findings_without_the_d111_path_are_never_served(kb):
    runtime, _ = PF.build_runtime(kb, guide_scopes=True)
    cid = next(i for i, m in enumerate(runtime.metadata) if m["source_id"] == ITHENTICATE and m["chunk_type"] == "procedure_window"
               and m["step_start"] == 0 and m["step_end"] == m["step_count"] - 1 and not m.get("source_conflict"))
    quote = runtime.chunks[cid]

    def run(task):
        meta = runtime.metadata[cid]
        finding = Finding(finding_id="plain", claim=quote, evidence_quote=quote, source_id=ITHENTICATE, source_title="t", page=meta["page"],
                          metadata={"chunk_id": cid, "clauses": [0], "details_covered": ["x"]})
        return SpecialistFindings(specialist_id=SpecialistId.RESEARCH, task_id=task.task_id, status=FindingStatus.SUPPORTED, findings=[finding],
                                  summary="plain", metadata={"clauses": [{"index": 0, "clause": task.question, "outcome": "supported",
                                                                          "ask_shape": "specific_recognised", "requested_details": ["x"],
                                                                          "uncovered_details": [], "relation": "strong"}]})

    runtime.guide_registry.resolve("research").run = run
    attempt = runtime.attempt("How do I access iThenticate?")
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SOURCE_AUTHORITY and "outside the D-111 guide path" in attempt.reason


def test_handbook_answers_are_identical_with_and_without_the_guide_path(kb, runtime):
    plain, _ = PF.build_runtime(kb, guide_scopes=False)
    for question in ("How many credit hours is a full teaching load?", "How many theses can a faculty member supervise at once?",
                     "What is the phone number of the Human Resources Department?"):
        a, b = plain.attempt(question), runtime.attempt(question)
        assert a.outcome is b.outcome, question
        assert (a.result.answer if a.result else None) == (b.result.answer if b.result else None), question


def test_a_policy_question_naming_a_system_is_never_answered_by_a_guide(runtime):
    for question in ("What is the required teaching load for faculty who teach on Blackboard?",
                     "Who must approve final grades according to university policy before they go to Banner?",
                     "Within how many hours must instructors submit final examination results after Blackboard grading?"):
        attempt = runtime.attempt(question)
        sources = {i["meta"]["source_id"] for i in (attempt.result.items if attempt.result else [])}
        assert sources <= {HANDBOOK}, question


def test_the_guide_path_leaves_the_frozen_result_when_it_finds_nothing(runtime):
    attempt = runtime.attempt("How do I reset my Blackboard password?")
    assert attempt.outcome is not PF.PlanFOutcome.ACCEPTED
    run = next(r for r in attempt.dispatch.runs if r.findings is not None)
    path = run.findings.metadata.get("guide_path")
    assert path is not None and path["answered"] == [] and "reset" not in json.dumps(path)


def test_diagnostics_and_the_developer_panel_carry_the_guide_form_only(runtime):
    attempt = runtime.attempt("How do I access iThenticate?")
    d = UP._diagnose("active", "plan_f", attempt, None, None, legacy=None)
    assert d["plan_f_guide_served"] and d["guide_answer_forms"] == [GG.FORM_PROCEDURE] and d["guide_answer_parts"] == 3
    assert "IThenticate" not in json.dumps(d) and "access" not in json.dumps({k: v for k, v in d.items() if k not in ("reason",)})
    rows = dict(next(rows for title, rows in P.dev_sections(attempt.result, d) if title == "Routing"))
    assert rows["Guide answer"] == "procedure (3 parts)"
