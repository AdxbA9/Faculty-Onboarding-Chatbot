"""
Post-freeze PF-3D (decision record D-112): deterministic guide structure recovery.

Offline. Synthetic guide pages built with the PF-2 chunker (so every quote is checked against real chunk text), the real
five-document corpus without a model, and the Plan F adapter end to end with the fake embedder and the raw-overlap
reranker. Structure may be recovered only from printed formatting and text: glyph or printed bullets, labels, heading
rows, short title lines and page breaks. Nothing is invented, guessed or renumbered.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import pytest

from conftest import FakeEmbedder
from faculty_services_support import RawOverlapReranker
from handbook_bot import guide_grounding as GG
from handbook_bot import guide_structure as GS
from handbook_bot import plan_f_runtime as PF
from handbook_bot.chunking import build_chunks
from handbook_bot.knowledge_base import KnowledgeBase, assemble_corpus
from handbook_bot.sources import SourceConflict, SourceRecord, load_source_registry

REPO = Path(__file__).resolve().parents[1]
HANDBOOK = "uos_faculty_handbook_2025_26"
SHORT_GUIDE = "uos_blackboard_faculty_short_guide"
GRADEBOOK = "uos_blackboard_gradebook_guide"
GRADE_TRANSFER = "uos_grade_transfer_dashboard_guide"
ITHENTICATE = "uos_ithenticate_guidelines"
G = ""                                           # the private-use bullet glyph the guides print on its own row


def _record(source_id=SHORT_GUIDE, **kw):
    base = dict(source_id=source_id, title="Test Guide", relative_path="data/guides/test.pdf", source_type="pdf", document_type="system_guide",
                authority_level="system_guide", system="Blackboard Ultra", access="public", enabled=True, domains=("teaching",),
                audience=("faculty", "instructor"), verification_status="verified_local_file")
    base.update(kw)
    return SourceRecord(**base)


def _index(pages, record=None):
    """A GuideIndex over synthetic pages ({page: [rows]}); chunks are built by the PF-2 chunker from the same rows."""
    record = record or _record()
    page_dicts = [{"page": n, "section_hint": "", "rows": rows, "lines": rows, "text": "\n".join(rows)} for n, rows in sorted(pages.items())]
    chunks, metadata = build_chunks(page_dicts, procedures=True)
    for meta in metadata:
        meta.update(source_id=record.source_id, document_type=record.document_type, authority_level="system_guide", source_title=record.title)
    policy = {"teaching": frozenset({HANDBOOK, record.source_id}), "research": frozenset({HANDBOOK}),
              "faculty_services": frozenset({HANDBOOK}), "institutional": frozenset({HANDBOOK})}
    return GG.GuideIndex(chunks, metadata, {record.source_id: page_dicts}, [record], policy)


def _bind(clause, record=None):
    binding, problem = GG.bind(clause, record or _record())
    assert binding is not None, problem
    return binding


def _answer(index, clause, record=None):
    binding = _bind(clause, record)
    ranks = {s.key: 0 for s in index.structures_of(binding.source_id)}
    answer, considered = GG.select_answer(binding, [], index, {}, ranks)
    return answer, considered


def _structure(index, kind, heading):
    return next(s for s in index.structures_of(SHORT_GUIDE) if s.kind == kind and s.heading == heading)


ANNOUNCE = {9: [G, "Send an Announcement:", G, "From details and actions click on create Announcements.", G,
                "Click on the Plus sign in the top right of the page.", G, "Add the title and the message of the",
                "announcement.", G, "Click on Post so the announcement will reach students."]}


# ---------------------------------------------------------------------------
# Reading: blocks, markers, headings, continuations
# ---------------------------------------------------------------------------
def test_glyph_rows_mark_items_and_a_bulleted_colon_row_is_a_heading():
    blocks = GS.read_blocks(ANNOUNCE[9], 9)
    assert [b.kind for b in blocks] == ["heading", "item", "item", "item", "item"]
    assert blocks[0].text == "Send an Announcement:" and blocks[0].rank == 2
    assert all(b.marker == "bullet" for b in blocks[1:])
    assert blocks[3].text == "Add the title and the message of the announcement."          # the wrapped row continues its item


@pytest.mark.parametrize("rows, kinds", [
    (["How to upload a document", "1) Select Upload.", "2) Click Upload."], ["heading", "item", "item"]),                 # "How to ..."
    (["The setting is on.", "Filter by child courses", "Select Filter to open the panel."], ["prose", "heading", "prose"]),  # short title
    (["If you are unable to access it email the service desk at servicedesk@example.org", "Checking Similarity Report",
      "Click the percentage to open it."], ["prose", "heading", "prose"]),                                             # Title Case
    (["1. Download Workflow", "In the list view, select the icon."], ["heading", "prose"]),                               # numbered title
    (["C. Reorder content:", "Point to an item to reveal its functions."], ["heading", "prose"]),                           # lettered title
    (["1. Click on View Details", "2. Click on Transfer:", "3. Approve."], ["item", "item", "item"]),                       # steps stay steps
    (["You can filter your gradebook view based on merged", "courses:", "In the grid, select Filter."], ["prose"]),         # lower case
    (["This is a full explanatory sentence that runs well past any title length and keeps going:", "Select it."], ["prose"]),
    (["A E", "C", "Select the plus sign to add content."], ["prose"]),                                                   # callouts
])
def test_heading_recovery_and_false_heading_rejection(rows, kinds):
    assert [b.kind for b in GS.read_blocks(rows, 1)] == kinds, rows


@pytest.mark.parametrize("sentence, instruction", [
    ("Click on Post so the announcement will reach students.", True),
    ("In the menu, select Create to open the panel.", True),
    ("From details and actions click on create Announcements.", True),
    ("Next, select Assign Zeros.", True),
    ("When you create an assignment, a gradebook item is created automatically.", False),
    ("Filter only appears if you have merged courses.", False),
    ("Students cannot see private courses.", False),
])
def test_an_instruction_opens_its_sentence(sentence, instruction):
    assert GS.Block("item", "bullet", "", sentence, 1).instruction is instruction


# ---------------------------------------------------------------------------
# Sections, lists and their boundaries
# ---------------------------------------------------------------------------
def test_an_unnumbered_bulleted_workflow_is_one_ordered_procedure_with_exact_quotes():
    index = _index(ANNOUNCE)
    lst = _structure(index, "list", "Send an Announcement")
    assert lst.list_type == "ordered_procedure" and lst.pages == (9,) and not lst.problems
    assert lst.start == "section heading" and [p.marker for p in lst.parts] == ["heading", "bullet", "bullet", "bullet", "bullet"]
    for part in lst.parts:
        assert part.quote in index.chunks[part.chunk_id] and index.metadata[part.chunk_id]["page"] == 9
    answer, _ = _answer(index, "How do I send an announcement to my students in Blackboard?")
    assert answer.form == GG.FORM_SECTION and answer.procedure_key == lst.key and answer.pages == (9,)
    rendered = [GG.render_part(p.quote, answer.record(i)) for i, p in enumerate(answer.parts)]
    assert rendered[0] == "Send an Announcement:" and rendered[1].startswith("- From details") and not any(r[:2].isdigit() for r in rendered)


def test_a_section_ends_at_the_next_heading_of_its_rank_and_a_lead_in_heading_does_not_end_it():
    pages = {10: ["Download and upload grades", "1. Download Workflow", "In the list or grid view, select the icon with the down",
                  "arrow to open the Download panel.", "Make your selections:", G + " Download the full gradebook or select columns.",
                  G + " Select the file type for the downloaded file.", "2. Upload a file to the gradebook", G,
                  "You can upload a grade file you worked on offline."]}
    index = _index(pages)
    section = _structure(index, "section", "Download Workflow")
    assert [b.kind for b in section.blocks] == ["prose", "heading", "item", "item"] and section.end == "next heading on page 10"
    answer, _ = _answer(index, "How do I download the gradebook in Blackboard?")
    assert answer.procedure_key == section.key                                    # not the options list alone
    lst = _structure(index, "list", "Make your selections")
    problem = GG.structure_problem(lst, _bind("How do I download the gradebook in Blackboard?"), index)
    assert problem is not None                                                    # the instruction above the bullets is outside it


def test_a_section_continued_on_the_next_page_cites_both_pages_and_a_heading_on_the_next_page_ends_it():
    pages = {10: [G, "Create an Assignment:", G, "On the Course Content page, select the plus sign to add an assignment.", G,
                  "In the menu, select Create and select Assignment."],
             11: [G, "Add a time limit. A time limit keeps students on track and focused on their work.", G,
                  "Provide a due date. Due dates appear on the calendar and in the activity stream."],
             12: ["Grading Assignments:", "Select a submission to grade it."]}
    index = _index(pages)
    section = _structure(index, "section", "Create an Assignment")
    assert section.pages == (10, 11) and section.end == "page 12 opens with a heading" and not section.problems
    answer, _ = _answer(index, "How do I create an assignment in Blackboard?")
    assert answer is not None and answer.pages == (10, 11)
    assert {p.page for p in answer.parts} == {10, 11}


def test_a_page_continuation_that_is_not_indexed_leaves_the_end_unproven():
    pages = {1: ["How to export grades:", "1. Open the gradebook and select Export.", "2. Choose the file type."],
             3: ["Other topic:", "Something else entirely."]}
    index = _index(pages)
    lst = next(s for s in index.structures_of(SHORT_GUIDE) if s.kind == "list")
    assert lst.problems and "not indexed" in lst.problems[0]
    answer, _ = _answer(index, "How do I export grades in Blackboard?")
    assert answer is None or answer.form != GG.FORM_SECTION


@pytest.mark.parametrize("rows, why", [
    (["How to export grades:", "2. Choose the file type.", "3. Select Download."], "numbering"),                     # first step missing
    (["How to export grades:", "1. Open the gradebook and select Export.", "3. Select Download."], "numbering"),      # middle step missing
    (["How to export grades:", "1. Open the gradebook and select Export.", "2. Select Download.", "1. Open the roster.",
      "2. Select Export."], "numbering"),                                                                               # two procedures mixed
])
def test_incomplete_or_mixed_numbered_procedures_are_refused(rows, why):
    index = _index({1: rows, 2: ["Next topic:", "Select something."]})
    answer, considered = _answer(index, "How do I export grades in Blackboard?")
    assert answer is None, considered
    assert any(why in str(c.get("reason")) for c in considered)


@pytest.mark.parametrize("rows", [
    ["How to export grades:", "1. Open the gradebook and select Export.", "2. Make your selections:"],      # numbered: the D-111 path
    ["Export grades:", G, "Open the gradebook and select Export.", G, "Make your selections:"],             # bulleted: the D-112 path
])
def test_a_last_step_that_announces_missing_content_is_refused(rows):
    index = _index({1: rows, 2: ["Next topic:", "Select something."]})
    binding = _bind("How do I export grades in Blackboard?")
    procedure_ranks = {p.key: 0 for p in index.procedures_of(SHORT_GUIDE)}
    ranks = {s.key: 0 for s in index.structures_of(SHORT_GUIDE)}
    answer, considered = GG.select_answer(binding, [], index, procedure_ranks, ranks)
    assert answer is None, considered
    assert any("a step is missing" in str(c.get("reason")) for c in considered)


def test_a_last_step_that_points_at_a_figure_is_not_a_missing_step():
    assert GG.dangling_lead_in("You can see the grade under the Banner Column after the approval from Chair and Dean as below:") is None
    assert GG.dangling_lead_in("Make your selections:") is not None
    assert GG.dangling_lead_in("Select Download.") is None


def test_a_heading_that_belongs_to_another_procedure_is_refused():
    index = _index({1: ["How to publish a course:", "1. Open the course and check it.", "2. Select Open.", "How to archive a course:",
                        "1. Open the course menu.", "2. Select Archive."], 2: ["Next topic:", "Select something."]})
    answer, considered = _answer(index, "How do I export a course in Blackboard?")
    assert answer is None                                                         # no heading names the asked action
    publish = next(s for s in index.structures_of(SHORT_GUIDE) if s.kind == "list" and s.heading.startswith("How to publish"))
    archive = next(s for s in index.structures_of(SHORT_GUIDE) if s.kind == "list" and s.heading.startswith("How to archive"))
    parts = [(p.chunk_id, p.quote, p.page) for p in publish.parts]
    clause = "How do I publish a course in Blackboard?"
    assert GG.verify_answer(clause, GG.FORM_SECTION, parts, publish.key, index, [SHORT_GUIDE]) is None
    assert GG.verify_answer(clause, GG.FORM_SECTION, parts, archive.key, index, [SHORT_GUIDE]) is not None   # steps under another heading
    archive_parts = [(p.chunk_id, p.quote, p.page) for p in archive.parts]
    assert GG.verify_answer(clause, GG.FORM_SECTION, archive_parts, archive.key, index, [SHORT_GUIDE]) is not None


@pytest.mark.parametrize("mutate", [lambda p: p[:-1], lambda p: [p[0], p[2], p[1]] + p[3:], lambda p: p + p[-1:]])
def test_the_verification_refuses_a_structure_that_is_not_exactly_complete(mutate):
    index = _index(ANNOUNCE)
    answer, _ = _answer(index, "How do I send an announcement to my students in Blackboard?")
    parts = [(p.chunk_id, p.quote, p.page) for p in answer.parts]
    clause = "How do I send an announcement to my students in Blackboard?"
    assert GG.verify_answer(clause, GG.FORM_SECTION, parts, answer.procedure_key, index, [SHORT_GUIDE]) is None
    assert "not exactly the claimed structure" in GG.verify_answer(clause, GG.FORM_SECTION, mutate(parts), answer.procedure_key, index, [SHORT_GUIDE])
    assert GG.is_authority_problem(GG.verify_answer(clause, GG.FORM_SECTION, parts, answer.procedure_key, index, [HANDBOOK]))


def test_a_block_that_no_chunk_holds_verbatim_makes_the_structure_unusable():
    index = _index(ANNOUNCE)
    lst = _structure(index, "list", "Send an Announcement")
    for part in lst.parts:
        index.chunks[part.chunk_id] = "rewritten text"
    rebuilt = GG.GuideIndex(index.chunks, index.metadata, {SHORT_GUIDE: [{"page": 9, "rows": ANNOUNCE[9]}]}, [_record()])
    assert all(s.problems or not s.parts for s in rebuilt.structures_of(SHORT_GUIDE))


def test_a_registered_conflict_inside_a_structure_refuses_it():
    record = _record(conflicts=[SourceConflict(topic="post_rule", phrases=("will reach students",), status="unresolved")])
    index = _index(ANNOUNCE, record)
    answer, considered = _answer(index, "How do I send an announcement to my students in Blackboard?", record)
    assert answer is None and any("source conflict" in str(c.get("reason")) for c in considered)


# ---------------------------------------------------------------------------
# Unordered lists, list types, mixed hierarchy
# ---------------------------------------------------------------------------
FEATURES = {4: ["Course Tools:", G, "Roster: View your class list and send messages to students.", G,
                "Groups: Create group sets for graded work and activities.", G,
                "Attendance: Mark attendance grades and access detailed records."], 5: ["Next topic:", "Select something."]}


def test_an_unordered_list_is_never_presented_as_steps():
    index = _index(FEATURES)
    lst = _structure(index, "list", "Course Tools")
    assert lst.list_type == "unordered_options"
    assert GG.structure_problem(lst, _bind("How do I mark attendance grades in Blackboard?"), index) is not None
    answer, _ = _answer(index, "What tools are in Course Tools in Blackboard?")
    if answer is not None:
        rendered = [GG.render_part(p.quote, answer.record(i)) for i, p in enumerate(answer.parts)]
        assert all(not r.lstrip()[:2].rstrip(".)").isdigit() for r in rendered)       # bullets are never numbered
        assert [r[:2] for r in rendered[1:]] == ["- "] * (len(rendered) - 1)


def test_lettered_callouts_with_titles_are_options_not_a_sequence():
    items = [GS.Block("item", "lettered", "%s." % L, "%s. %s" % (L, t), 4) for L, t in
             (("A", "Change your view: Navigate between the tabs to see your tasks."), ("B", "Set your view. Use the Items per page menu to choose rows."),
              ("C", "Search for a student. Limit your search to a few letters."))]
    assert GS.classify("You can also view the grades", items, False) == "unordered_options"


@pytest.mark.parametrize("context, texts, markers, action, expected", [
    ("From the Settings panel, you can:", ["Change the schema.", "Add categories."], "numbered", False, "features"),
    ("File Requirements", ["Files must be less than 100MB.", "Files must contain 20 words."], "bullet", False, "requirements"),
    ("Gradebook settings", ["Schema", "Categories here"], "bullet", False, "settings"),
    ("Support", ["Mail servicedesk@sharjah.ac.ae today.", "Call +971 6 505 0000 now."], "bullet", False, "contacts"),
    ("Limits", ["Up to 200MB per file.", "At most 400 pages in total."], "bullet", False, "numeric_limits"),
    ("Approval stages", ["The Chair reviews it.", "The Dean signs it."], "bullet", False, "approval_stages"),
    ("Export grades", ["Open the gradebook.", "Select Export."], "numbered", False, "ordered_procedure"),
    ("Two views", ["List View.", "Grid View."], "numbered", False, "ordered_list"),
    ("Send an Announcement", ["Click on the Plus sign.", "Click on Post."], "bullet", True, "ordered_procedure"),
    ("About Navigation", ["Click on the Plus sign.", "Click on Post."], "bullet", False, "unordered_options"),
])
def test_list_type_classification(context, texts, markers, action, expected):
    items = [GS.Block("item", markers, "%d." % (i + 1) if markers == "numbered" else "", ("%d. %s" % (i + 1, t)) if markers == "numbered" else t, 1)
             for i, t in enumerate(texts)]
    assert GS.classify(context, items, action) == expected


def test_steps_with_lettered_substeps_keep_their_hierarchy():
    index = _index({1: ["How to export grades:", "1. Open the gradebook and select Export.", "a) Select the full gradebook.",
                        "b) Select the file type.", "2. Select Download to save the file."], 2: ["Next topic:", "Select something."]})
    answer, _ = _answer(index, "How do I export grades in Blackboard?")
    labels = [p.quote.split()[0] for p in answer.parts[1:]]
    assert labels == ["1.", "a)", "b)", "2."]                                     # printed labels, never flattened or renumbered


def test_a_list_lead_in_split_across_rows_is_recovered_and_answers_the_set():
    index = _index({1: ["Gradebook views:", "The gradebook is populated with students when they are", "enrolled. You can grade coursework, manage",
                        "items, and post grades from two views:", "1. List View.", "a) Gradable Items list.", "b) Student list.", "2. Student grid View."],
                    2: ["Next topic:", "Select something."]})
    lst = next(s for s in index.structures_of(SHORT_GUIDE) if s.kind == "list")
    assert lst.lead_in == "You can grade coursework, manage items, and post grades from two views:" and lst.list_type == "ordered_list"
    answer, _ = _answer(index, "What are the two views of the Blackboard gradebook?")
    assert answer is not None and answer.parts[0].quote == lst.lead_in


def test_a_list_must_name_the_asked_set_to_answer_an_enumeration():
    index = _index({6: ["Managing Content:", G + " Create Content: Select the plus sign to add content.", G + " Upload: Browse for files. Supported file types include DOC, PDF and TXT.",
                        G + " Copy Content: Copy an entire course or items."], 7: ["Next topic:", "Select something."]})
    answer, considered = _answer(index, "Which file types can I upload to Blackboard course content?")
    assert answer is None or answer.form != GG.FORM_SECTION


# ---------------------------------------------------------------------------
# The real corpus (no model) and the adapter end to end
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parts():
    return assemble_corpus(load_source_registry(str(REPO / "knowledge" / "sources.json")).enabled_sources())


@pytest.fixture(scope="module")
def real_index(parts):
    return GG.GuideIndex(parts.chunks, parts.metadata, parts.source_pages, parts.records)


def _real(real_index, clause, source_id):
    binding, problem = GG.bind(clause, real_index.record(source_id))
    assert binding is not None, problem
    ranks = {s.key: 0 for s in real_index.structures_of(source_id)}
    procedure_ranks = {p.key: 0 for p in real_index.procedures_of(source_id)}
    return GG.select_answer(binding, [], real_index, procedure_ranks, ranks)


@pytest.mark.parametrize("clause, source_id, form, pages, first", [
    ("How do I move a content item with the keyboard in Blackboard Ultra?", SHORT_GUIDE, "section", (7,), "With your keyboard, you can move"),
    ("How do I send an announcement to my students in Blackboard?", SHORT_GUIDE, "section", (9,), "Send an Announcement:"),
    ("How do I create an assignment in Blackboard Ultra?", SHORT_GUIDE, "section", (10, 11), "Create an Assignment:"),
    ("What are the two views of the Blackboard Ultra gradebook?", GRADEBOOK, "section", (1,), "You can grade coursework"),
    ("How do I make Blackboard assign zeros to missing work automatically?", GRADEBOOK, "section", (8,), "Assign automatic zeros:"),
    ("How do I download the gradebook from Blackboard?", GRADEBOOK, "section", (10,), "1. Download Worflow"),
    ("How do I set accommodations for a student in the Blackboard gradebook?", GRADEBOOK, "section", (5,), "Add accommodations"),
    ("How do I filter the Blackboard gradebook by merged child courses?", GRADEBOOK, "section", (8,), "Filter by child courses"),
    ("How do I transfer grades from Blackboard to Banner?", GRADE_TRANSFER, "procedure", (2,), "How to Send Grades"),     # D-111 first
    ("How do I access iThenticate?", ITHENTICATE, "procedure", (1,), "How to access IThenticate"),
])
def test_the_real_guides_recover_the_measured_structures(real_index, clause, source_id, form, pages, first):
    answer, considered = _real(real_index, clause, source_id)
    assert answer is not None, considered[:4]
    assert answer.form == form and answer.pages == pages and answer.parts[0].quote.startswith(first), (answer.form, answer.pages, answer.parts[0].quote[:60])


@pytest.mark.parametrize("clause, source_id", [
    ("Which file types can I upload to Blackboard course content?", SHORT_GUIDE),             # the options list is not the asked set
    ("How do I post grades to students from the Blackboard gradebook?", GRADEBOOK),          # one item of a numbered list of actions
    ("How do I add an attendance row to the Blackboard gradebook?", GRADEBOOK),               # an option inside a capability list
    ("How do I verify my identity when signing in to Blackboard?", SHORT_GUIDE),             # a sub-task of another procedure
])
def test_structures_never_serve_what_they_do_not_prove(real_index, clause, source_id):
    answer, _ = _real(real_index, clause, source_id)
    assert answer is None or answer.form != GG.FORM_SECTION


def test_a_system_name_in_a_heading_never_names_the_asked_action(real_index):
    access = next(s for s in real_index.structures_of(GRADE_TRANSFER) if s.kind == "list" and s.section_title.startswith("How to use"))
    binding, _ = GG.bind("How do I transfer grades from Blackboard to Banner?", real_index.record(GRADE_TRANSFER))
    assert "names neither" in GG.structure_problem(access, binding, real_index)


def test_every_real_structure_quotes_its_own_page_exactly(real_index):
    count = 0
    for source_id in (SHORT_GUIDE, GRADEBOOK, GRADE_TRANSFER, ITHENTICATE):
        for structure in real_index.structures_of(source_id):
            for part in structure.parts:
                assert part.quote in real_index.chunks[part.chunk_id]
                assert real_index.metadata[part.chunk_id]["source_id"] == source_id and real_index.metadata[part.chunk_id]["page"] == part.page
                count += 1
    assert count > 100


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


def test_a_recovered_procedure_is_served_end_to_end_with_its_markers_and_pages(kb):
    runtime, build = PF.build_runtime(kb, guide_scopes=True)
    assert build.ok and runtime.guide_index.structures_of(SHORT_GUIDE)
    attempt = runtime.attempt("How do I send an announcement to my students in Blackboard?")
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED, attempt.reason
    records = [i["meta"]["plan_f"]["guide"] for i in attempt.result.items]
    assert {r["form"] for r in records} == {GG.FORM_SECTION} and records[0]["marker"] == "heading"
    assert attempt.result.pages == [9] and {i["meta"]["source_id"] for i in attempt.result.items} == {SHORT_GUIDE}
    assert attempt.result.answer.startswith("Send an Announcement:\n\n- From details and actions click on create Announcements.")
    assert attempt.result.llm_calls == 0 and "announcement" not in json.dumps(records).lower()
    # cited under the printed heading, not under the chunk's carried page hint ("B. Content Visibility:" from page 7)
    assert {i["meta"]["section"] for i in attempt.result.items} == {"Send an Announcement"}
    assert attempt.result.best_section == "Send an Announcement"
    assert not any("section_title" in i["meta"] for i in attempt.result.items)


def test_citation_headings_come_from_the_verified_procedure_or_structure_only(real_index):
    transfer = next(p for p in real_index.procedures_of(GRADE_TRANSFER) if (p.heading or "").startswith("How to Send Grades"))
    keyboard = next(s for s in real_index.structures_of(SHORT_GUIDE) if s.kind == "list" and s.heading.startswith("With your keyboard"))
    assert GG.citation_heading({"form": GG.FORM_PROCEDURE, "procedure_key": transfer.key}, real_index) == "How to Send Grades to Banner as instructor?"
    assert GG.citation_heading({"form": GG.FORM_SECTION, "procedure_key": keyboard.key}, real_index) == "Reorder content"
    assert GG.citation_heading({"form": GG.FORM_STATEMENT, "procedure_key": None}, real_index) == ""        # a statement: the page alone
    assert GG.citation_heading({"form": GG.FORM_SECTION, "procedure_key": "no-such-key"}, real_index) == ""
    assert GG.citation_heading(None, real_index) == "" and GG.citation_heading({"form": GG.FORM_PROCEDURE}, None) == ""


def test_only_guide_items_are_relabelled(real_index):
    keyboard = next(s for s in real_index.structures_of(SHORT_GUIDE) if s.kind == "list" and s.heading.startswith("With your keyboard"))
    guide = {"idx": 1, "chunk": "q", "meta": {"page": 7, "section": "B. Content Visibility:", "section_title": "B. Content Visibility:",
                                             "plan_f": {GG.GUIDE_KEY: {"form": GG.FORM_SECTION, "procedure_key": keyboard.key}}}}
    handbook = {"idx": 2, "chunk": "q", "meta": {"page": 79, "section": "Responsibility to Office Hours", "plan_f": {"finding_id": "f"}}}
    out = PF.guide_citation_items([guide, handbook], real_index)
    assert out[0]["meta"]["section"] == "Reorder content" and "section_title" not in out[0]["meta"]
    assert out[1] is handbook and guide["meta"]["section"] == "B. Content Visibility:"                    # inputs are not mutated


def test_forged_section_answers_are_refused_by_the_adapter(kb):
    runtime, _ = PF.build_runtime(kb, guide_scopes=True)
    structure = next(s for s in runtime.guide_index.structures_of(SHORT_GUIDE) if s.kind == "list" and s.heading == "Send an Announcement")
    from handbook_bot.agents.specialists.contracts import Finding, FindingStatus, SpecialistFindings, SpecialistId

    def run(task):
        parts = structure.parts[:-1]                                               # the last step left out
        findings = []
        for n, part in enumerate(parts):
            record = {"version": GG.VERSION, "phase": GG.PHASE, "source_id": SHORT_GUIDE, "form": GG.FORM_SECTION, "intent": "procedure",
                      "procedure_key": structure.key, "part": n, "parts": len(parts), "binding": {}, "marker": part.marker}
            findings.append(Finding(finding_id="forged-%d" % n, claim=part.quote, evidence_quote=part.quote, source_id=SHORT_GUIDE, source_title="t",
                                    page=part.page, metadata={"chunk_id": part.chunk_id, "clauses": [0], GG.GUIDE_KEY: record}))
        report = {"index": 0, "clause": task.context["focus_clauses"][0], "outcome": "supported", "findings": [f.finding_id for f in findings],
                  GG.GUIDE_KEY: {k: v for k, v in findings[0].metadata[GG.GUIDE_KEY].items() if k != "part"}}
        return SpecialistFindings(specialist_id=SpecialistId.TEACHING, task_id=task.task_id, status=FindingStatus.SUPPORTED, findings=findings,
                                  summary="forged", metadata={"clauses": [report]})

    runtime.guide_registry.resolve("teaching").run = run
    attempt = runtime.attempt("How do I send an announcement to my students in Blackboard?")
    assert attempt.outcome is PF.PlanFOutcome.REJECTED_BY_SUBJECT_GUARD and attempt.result is None and "not exactly" in attempt.reason
