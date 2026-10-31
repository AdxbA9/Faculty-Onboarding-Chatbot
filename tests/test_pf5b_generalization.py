"""
Post-freeze PF-5B (decision record D-114): natural-language guide generalization.

Offline. Synthetic guide pages built with the PF-2 chunker, the real eight-source corpus without a model, and the Plan F
adapter end to end with the fake embedder and the raw-overlap reranker. Every rule is generic: none names a question of
an evaluation set. Each fix area has positive and negative cases:
    A  wording: non-critical modifiers, contrast modifiers, container phrases, compounds, synonyms, derivations, polarity
    B  routing: software navigation stays with the system's specialist; physical and organizational asks do not
    C  misspelled system names (never policy words; ambiguous or unsupported corrections refused)
    D  multi-part: all clauses or nothing (the aggregation contract has no partial answer)
    E  conflict granularity: the unit stating the disputed claim is blocked, its neighbours are not
    F  mixed-label procedures: lettered sections over numbered sub-sections, continuation rows, cut units
and the held sources and the frozen fresh held-out set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import pytest

from conftest import FakeEmbedder
from faculty_services_support import RawOverlapReranker
from handbook_bot import guide_grounding as GG
from handbook_bot import guide_structure as GS
from handbook_bot import name_spelling as NS
from handbook_bot import plan_f_runtime as PF
from handbook_bot import source_scopes as SS
from handbook_bot.agents.specialists.contracts import (Finding, FindingStatus, HandoffRequest, SpecialistFindings, SpecialistId,
                                                       SpecialistTask)
from handbook_bot.agents.specialists.guide_evidence import SOFTWARE_NAVIGATION, GuideCapableSpecialist
from handbook_bot.chunking import build_chunks
from handbook_bot.knowledge_base import KnowledgeBase, assemble_corpus
from handbook_bot.sources import SourceConflict, SourceRecord, load_source_registry

REPO = Path(__file__).resolve().parents[1]
HANDBOOK = "uos_faculty_handbook_2025_26"
SHORT_GUIDE = "uos_blackboard_faculty_short_guide"
GRADEBOOK = "uos_blackboard_gradebook_guide"
GRADE_TRANSFER = "uos_grade_transfer_dashboard_guide"
ITHENTICATE = "uos_ithenticate_guidelines"
COLLABORATE = "uos_blackboard_collaborate_guide"
G = ""                                     # the private-use bullet glyph the guides print on its own row


def _record(source_id=SHORT_GUIDE, **kw):
    base = dict(source_id=source_id, title="Test Guide", relative_path="data/guides/test.pdf", source_type="pdf", document_type="system_guide",
                authority_level="system_guide", system="Blackboard Ultra", access="public", enabled=True, domains=("teaching",),
                audience=("faculty", "instructor"), verification_status="verified_local_file")
    base.update(kw)
    return SourceRecord(**base)


def _index(pages, record=None):
    record = record or _record()
    page_dicts = [{"page": n, "section_hint": "", "rows": rows, "lines": rows, "text": "\n".join(rows)} for n, rows in sorted(pages.items())]
    chunks, metadata = build_chunks(page_dicts, procedures=True)
    for meta in metadata:
        meta.update(source_id=record.source_id, document_type=record.document_type, authority_level="system_guide", source_title=record.title)
    policy = {"teaching": frozenset({HANDBOOK, record.source_id}), "research": frozenset({HANDBOOK}),
              "faculty_services": frozenset({HANDBOOK}), "institutional": frozenset({HANDBOOK})}
    return GG.GuideIndex(chunks, metadata, {record.source_id: page_dicts}, [record], policy)


def _answer(index, clause, source_id=SHORT_GUIDE):
    binding, problem = index.bind(clause, source_id)
    assert binding is not None, problem
    ranks = {s.key: 0 for s in index.structures_of(source_id)}
    procs = {p.key: 0 for p in index.procedures_of(source_id)}
    units = []
    for cid, meta in enumerate(index.metadata):
        if meta.get("source_id") == source_id:
            units.extend(GG.statement_units(cid, index.chunks[cid], meta, 0, binding))
    return GG.select_answer(binding, units, index, procs, ranks)


def _terms(clause, record=None, vocabulary=None):
    binding, problem = GG.bind(clause, record or _record(), vocabulary)
    assert binding is not None, problem
    return binding


ANNOUNCE = {9: [G, "Send an Announcement:", G, "From details and actions click on create Announcements.", G,
                "Click on the Plus sign in the top right of the page.", G, "Add the title and the message of the",
                "announcement.", G, "Click on Post so the announcement will reach students."]}


# ---------------------------------------------------------------------------
# A. Wording
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause, absent", [
    ("ok so how do I honestly send an announcement in Blackboard?", "ok"),
    ("hey quick q, how do I send an announcement in Blackboard?", "hey"),
    ("Hi, thanks - how do I send an announcement in Blackboard?", "thanks"),
    ("How do I send an announcement to all the students in Blackboard?", "all"),
])
def test_discourse_words_are_not_terms(clause, absent):
    assert GG.stem(absent) not in _terms(clause).terms


@pytest.mark.parametrize("clause, kept", [
    ("How do I export the gradebook in Blackboard?", "export"),                        # actions stay terms
    ("How do I send an announcement to students in Blackboard?", "student"),
    ("How do I hide an assignment from specific students in Blackboard?", "specific"),  # scope words restrict the ask
    ("How do I view the current grade in Blackboard?", "current"),
    ("How do I copy the whole course in Blackboard?", "whole"),
    ("Can I edit a test once students submit it in Blackboard?", "once"),
    ("Can I add a time limit to group assignments in Blackboard?", "group"),            # marked members are always terms
    ("How do I send midterm grades in Blackboard?", "midterm"),
    ("Is recording required for Blackboard sessions?", "required"),
    ("How do I make my course available in Blackboard?", "available"),
])
def test_content_and_scope_words_stay_terms(clause, kept):
    final_guide = _record(notes="Guide to transferring final grades from Blackboard to Banner.")
    assert GG.stem(kept) in _terms(clause, final_guide, vocabulary=frozenset({"send", "grad"})).terms


FINAL_GUIDE = dict(notes="Guide to transferring final grades from Blackboard to Banner.")


@pytest.mark.parametrize("clause, record_kw, vocabulary, tolerated", [
    ("How do I send my final grades in Blackboard?", FINAL_GUIDE, {"send", "grad"}, True),
    ("How do I send my final course grades in Blackboard?", FINAL_GUIDE, {"send", "grad", "cours"}, True),
    ("How do I send my final grades in Blackboard?", {}, {"send", "grad"}, False),                   # the registry does not declare it
    ("How do I send my final grades in Blackboard?", FINAL_GUIDE, {"send", "grad", "midterm"}, False),  # the text draws the contrast
    ("How do I send my final grades in Blackboard?", FINAL_GUIDE, {"send", "grad", "final"}, False),    # the text states it itself
    ("How do I send my final grades in Blackboard?", FINAL_GUIDE, None, False),                         # no vocabulary: strict
    ("Are my final grades sent automatically in Blackboard?", FINAL_GUIDE, {"send", "grad"}, False),    # not a procedure ask
    ("How do I make the grade final in Blackboard?", FINAL_GUIDE, {"grad"}, False),                     # a predicate, not a modifier
])
def test_only_the_default_final_is_tolerated_and_only_when_declared_and_silent(clause, record_kw, vocabulary, tolerated):
    binding = _terms(clause, _record(**record_kw), frozenset(vocabulary) if vocabulary is not None else None)
    assert (GG.stem("final") not in binding.terms) is tolerated
    assert binding.tolerated == ((GG.stem("final"),) if tolerated else ())
    # "mid-term" written with a hyphen is the same contrast partner
    if tolerated:
        assert GG.stem("final") in _terms(clause, _record(**record_kw), GG.stems("Send the mid-term grades")).terms


def test_a_guide_that_distinguishes_final_from_midterm_grades_never_answers_the_other_one():
    pages = {1: ["How to send final grades", "1. Open the dashboard.", "2. Select the course.", "3. Click Send Final Grades."],
             2: ["How to send midterm grades", "1. Open the midterm page.", "2. Select the section.", "3. Click Send Midterm Grades."]}
    index = _index(pages, _record(**FINAL_GUIDE))
    answer, _ = _answer(index, "How do I send my final grades in Blackboard?")
    assert answer is not None and "final" in " ".join(p.quote for p in answer.parts).lower()
    assert "midterm" not in " ".join(p.quote for p in answer.parts).lower()
    answer, _ = _answer(index, "How do I send midterm grades in Blackboard?")
    assert answer is not None and "final" not in " ".join(p.quote for p in answer.parts).lower()


@pytest.mark.parametrize("clause", [
    "How do I post an announcement to the students in my Blackboard course?",
    "How do I send a message to all the students in my Blackboard course?",
    "What is the character limit for the title of my Blackboard course?",
    "How do I copy a Blackboard course?",
])
def test_the_course_a_question_names_is_always_a_term(clause):
    # D-114 tried binding "in my Blackboard course" with the system name; the differential check against PF-4 showed it let
    # announcements answer "send a message", attendance reports answer "mark attendance" and a gradebook item title answer a
    # course title, so the course stays a term the evidence must state
    assert GG.stem("course") in _terms(clause).terms


@pytest.mark.parametrize("clause, evidence, stated", [
    ("How do I log in to Blackboard?", "Click Login on the home page.", True),
    ("How do I login to Blackboard?", "Select Log in on the home page.", True),
    ("Where is the grade book in Blackboard?", "Open the Gradebook page from the course menu.", True),
    ("How do I add a virtual session in Blackboard?", "Select Add online session.", True),
    ("How do I add a learner in Blackboard?", "Select Add student.", True),
    ("How do I log in to Blackboard?", "Click Logout on the home page.", False),
    ("How do I add a class to Blackboard?", "Select Add course.", False),                  # class/course is not a synonym pair
    ("How do I remove a column in Blackboard?", "Select Delete column from the menu.", False),   # no verb synonyms
    ("How do I create an announcement in Blackboard?", "Select Announce to create it.", False),  # no derivations
    ("Can I enable discussions for content items in Blackboard?", "Students can discuss the content items.", False),
    ("Can students see their assignments in Blackboard?", "You can assign students to groups.", False),
])
def test_a_term_is_stated_only_by_itself_a_compound_or_its_noun_synonym(clause, evidence, stated):
    binding = _terms(clause)
    assert (not GG.missing_terms(binding, evidence)) is stated


def test_a_statement_that_states_only_the_opposite_condition_is_refused():
    binding = _terms("Can I edit a test before students submit it in Blackboard?")
    assert binding.polarity == ("before",)
    assert "opposite condition" in GG.statement_problem(binding, "You can edit a test after students submit it.")
    assert GG.statement_problem(binding, "You can edit a test before students submit it.") is None


def test_a_statement_cut_at_a_list_label_or_at_its_chunk_start_or_announcing_missing_content_is_refused():
    binding = _terms("Can I post grades in Blackboard?")
    assert "label of the next list item" in GG.statement_problem(binding, "You can post grades from two views: 1.")
    assert "announces content" in GG.statement_problem(binding, "You can post grades from two views:")
    units = GG.statement_units(0, "Manage the course. You can post grades from the grid. Then select manage items, and post grades from",
                               {"source_id": SHORT_GUIDE, "page": 1, "chunk_type": "paragraph"}, 0, binding)
    assert units and all(not u.quote[:1].islower() for u in units)
    units = GG.statement_units(0, "items, and post grades from two views. You can post grades from the grid.",
                               {"source_id": SHORT_GUIDE, "page": 1, "chunk_type": "paragraph"}, 0, binding)
    assert units and not any(u.quote.startswith("items") for u in units)


@pytest.mark.parametrize("quote, refused", [
    ("You will be redirected to the Collaborate administrator panel, in this panel you can manage the sessions.", True),
    ("Blackboard Collaborate Ultra Guide: - You will be redirected to the panel where you manage the sessions.", True),
    ("Then select the sessions panel on the course page.", True),
    ("Inside your course, you can access the sessions panel on the navigation bar.", False),
])
def test_a_statement_that_continues_an_unstated_step_is_refused(quote, refused):
    binding = _terms("Where do I find the sessions panel in Blackboard?")
    problem = GG.statement_problem(binding, quote)
    assert (problem is not None and "consequence of a step" in problem) is refused, problem


def test_the_cut_last_step_of_a_procedure_flagged_as_continuing_is_never_a_statement(real_index):
    flagged = [p for p in real_index.procedures_of(GRADEBOOK) if any("continues" in x for x in p.problems)]
    assert flagged
    for proc in flagged:
        assert "cut" in GG.cut_step_problem(real_index, proc.steps[-1].chunk_id)
        for step in proc.steps[:-1]:
            assert GG.cut_step_problem(real_index, step.chunk_id) is None


# ---------------------------------------------------------------------------
# B. Routing: software navigation versus physical and organizational navigation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause, binds", [
    ("Where do I find the gradebook in Blackboard?", True),
    ("Where can I see the course settings in Blackboard?", True),
    ("Where is the Grade Center in Blackboard?", False),                     # a centre: over-refused, a fallback
    ("Where is the Blackboard support office?", False),
    ("Which building is the Blackboard help desk in?", False),
    ("Where is the department that runs Blackboard?", False),
    ("How do I find the Blackboard support office?", False),                 # a procedure ask whose object is a place
    ("Is the Blackboard help desk on campus?", False),                        # a yes/no ask about a place
    ("How do I log in to Blackboard from outside the campus?", True),         # the place is a condition, not the object
])
def test_a_physical_or_organizational_place_never_binds_to_a_guide(clause, binds):
    binding, problem = GG.bind(clause, _record())
    assert (binding is not None) is binds, problem
    if not binds:
        assert GG.is_authority_problem(problem)


class _Base:
    """A frozen specialist stand-in that hands its only clause to Institutional Navigation."""

    def __init__(self, sid):
        self.specialist_id = sid

    def run(self, task):
        clause = task.context["focus_clauses"][0]
        handoff = HandoffRequest(from_specialist=self.specialist_id, requested_specialist=SpecialistId.INSTITUTIONAL, reason="location clause 0",
                                 task=None, metadata={"clause_index": 0, "kind": "location"})
        return SpecialistFindings(specialist_id=self.specialist_id, task_id=task.task_id, status=FindingStatus.OUT_OF_SCOPE, findings=[],
                                  summary="handed off", handoff_requests=[handoff],
                                  metadata={"clauses": [{"index": 0, "clause": clause, "outcome": "handed_off", "target": "institutional"}]})


class _NoGuide:
    """Guide evidence that binds (or not) but never answers."""

    def __init__(self, intent):
        self.intent = intent

    def answer(self, clause, route, approved):
        if self.intent is None:
            return None, {"applies": False, "refused": [], "bindings": []}
        return None, {"applies": True, "refused": [], "bindings": [{"source_id": SHORT_GUIDE, "intent": self.intent}]}


def _wrapped(intent):
    from handbook_bot.agents.specialists.base import Specialist
    base = _Base(SpecialistId.TEACHING)
    wrapper = GuideCapableSpecialist.__new__(GuideCapableSpecialist.wrap.__globals__["GuideCapableTeaching"])
    wrapper.base, wrapper.evidence = base, _NoGuide(intent)
    wrapper.approved_sources = frozenset({HANDBOOK, SHORT_GUIDE})
    wrapper.approved_guides = frozenset({SHORT_GUIDE})
    assert issubclass(type(wrapper), Specialist)
    return wrapper


def _task(clause):
    return SpecialistTask(task_id="t-1", question=clause, specialist_id=SpecialistId.TEACHING, domain="teaching", intent="policy",
                          context={"full_question": clause, "focus_clauses": [clause]})


def test_a_software_navigation_clause_stays_with_its_systems_specialist():
    result = _wrapped("location").run(_task("Where do I find the gradebook in Blackboard?"))
    assert not result.handoff_requests and result.status is FindingStatus.NOT_FOUND
    report = result.metadata["clauses"][0]
    assert report["outcome"] == "not_found" and report["kept"] == SOFTWARE_NAVIGATION
    assert result.missing and "clause 0" in result.missing[0] and result.metadata["software_navigation_kept"] == [0]


@pytest.mark.parametrize("intent, clause", [(None, "Where is the Blackboard support office?"), ("contact", "Where do I email about Blackboard?"),
                                           ("location", "Where is the Blackboard support team?"), ("location", "Where is the ITC for Blackboard?")])
def test_physical_organizational_and_contact_clauses_keep_their_handoff(intent, clause):
    result = _wrapped(intent).run(_task(clause))
    assert [h.requested_specialist for h in result.handoff_requests] == [SpecialistId.INSTITUTIONAL]
    assert result.status is FindingStatus.OUT_OF_SCOPE and result.metadata["clauses"][0]["outcome"] == "handed_off"


# ---------------------------------------------------------------------------
# C. Misspelled system names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, corrected, names", [
    ("how do i start a colab ultra sesion from my blackbord course", "how do i start a Collaborate ultra sesion from my Blackboard course",
     ["Collaborate", "Blackboard"]),
    ("how to upload my paper to ithentcate", "how to upload my paper to iThenticate", ["iThenticate"]),
    ("add a column in the gradbook on BB ultra", "add a column in the Gradebook on BB ultra", ["Gradebook"]),
    ("send grades to baner", "send grades to Banner", ["Banner"]),
    ("open the blakboard course", "open the Blackboard course", ["Blackboard"]),
    ("where are colab recordings kept in Blackboard", "where are Collaborate recordings kept in Blackboard", ["Collaborate"]),
    ("colaborate session settings in Blackboard", "Collaborate session settings in Blackboard", ["Collaborate"]),
    ("Blackboard colab is down", "Blackboard Collaborate is down", ["Collaborate"]),
    ("open the blackbaord course", "open the Blackboard course", ["Blackboard"]),                  # a transposition
])
def test_misspelled_system_names_are_corrected(text, corrected, names):
    out, records = NS.correct_system_names(text)
    assert out == corrected and sorted(r["name"] for r in records) == sorted(names)
    assert all(set(r) == {"name", "kind"} for r in records)                       # never the typed word


@pytest.mark.parametrize("text", [
    "can I colab with a colleague on a paper",           # the verb, not the product
    "I want to colaborate with my department",
    "is there a colab space for faculty",                # no product context
    "send grades to banker",                             # substitution into an English word
    "the backboard is broken",
    "how do I authenticate my account",                  # starts with another letter
    "Blackboards are used in every room",                # an inflection of the name
    "what is the attendance polcy",                      # policy words are never corrected
    "whats the max teching load for asst prof",
    "is the gradebook visible",                           # already right
    "Can my students colab in Blackboard?",               # the verb in the position of a verb
    "Where is the faculty colab room?",                   # a physical room: no Blackboard, BB or Ultra
    "where are colab recordings kept",                    # a product object but no product context
    "Blackboard collab recordings",                       # only the abbreviation the brief names ("colab") is an alias
    "I use Google Colab with Blackboard",                 # another product
    "How do I access Blackbaud?",                         # another product two edits away
    "how do I upload to ithentikat",                      # two edits with a substitution
    "Is the blackbird on campus?",                        # English words two edits away
    "Blackbeard was a pirate",
    "the paper was ithenticated last week",               # an inflection
])
def test_unsupported_or_ambiguous_corrections_are_refused(text):
    assert NS.correct_system_names(text) == (text, [])


def test_every_listed_near_miss_is_one_the_other_rules_would_correct(monkeypatch):
    listed = sorted(NS.NOT_TYPOS)
    monkeypatch.setattr(NS, "NOT_TYPOS", frozenset())
    for word in listed:
        assert any(NS.correct_system_names(ctx % word)[1] for ctx in ("%s", "Blackboard %s session")), word


def test_a_word_the_knowledge_base_prints_is_never_a_typo():
    assert NS.correct_system_names("blackbord", is_word=lambda w: w == "blackbord") == ("blackbord", [])
    assert NS.correct_system_names("blackbord", is_word=lambda w: False)[0] == "Blackboard"


def test_the_typed_words_of_a_clause_are_recovered_from_the_corrected_text():
    typed = "send grades to baner, and how do I post an anouncement in blackbord?"
    corrected, records, spans = NS.correct_with_spans(typed)
    assert corrected == "send grades to Banner, and how do I post an anouncement in Blackboard?" and len(records) == 2
    assert NS.typed_slice(typed, spans, corrected, "send grades to Banner") == "send grades to baner"
    assert NS.typed_slice(typed, spans, corrected, "how do I post an anouncement in Blackboard?") == "how do I post an anouncement in blackbord?"
    assert NS.typed_slice(typed, spans, corrected, corrected) == typed
    assert NS.typed_slice(typed, [], corrected, "anything") == "anything"


def test_two_names_in_reach_are_ambiguous(monkeypatch):
    monkeypatch.setattr(NS, "NAMES", NS.NAMES + (("Blackbored", 2, 7),))
    assert NS.correct_system_names("blackbord") == ("blackbord", [])


# ---------------------------------------------------------------------------
# E. Conflict granularity (real corpus) and F. mixed procedures (real corpus)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parts():
    return assemble_corpus(load_source_registry(str(REPO / "knowledge" / "sources.json")).enabled_sources())


@pytest.fixture(scope="module")
def real_index(parts):
    return GG.GuideIndex(parts.chunks, parts.metadata, parts.source_pages, parts.records)


def _chunk_with(parts, source_id, text, kind=None):
    return next(i for i, m in enumerate(parts.metadata) if m.get("source_id") == source_id and text in parts.chunks[i]
                and (kind is None or m.get("chunk_type") == kind))


def _stamped_chunks_with(parts, quote):
    found = [i for i, m in enumerate(parts.metadata) if m.get("source_id") == ITHENTICATE and quote in parts.chunks[i]
             and m.get("source_conflict") == "ithenticate_yearly_submission_limit"]
    assert found, quote
    return found


def test_the_quota_claim_is_blocked_and_its_page_neighbours_are_not(parts, real_index):
    blocked = ["Each instructor is limited to a specific amount of submission (Maximum 30 Documents per year)",
               "Use “Upload a New Revision” instead of “upload file” for every time you update the same research paper (this will not be counted as extra doc.)"]
    for quote in blocked:
        assert all(real_index.conflict_problem(cid, quote) for cid in _stamped_chunks_with(parts, quote)), quote
    served = ["Select the “Upload a File” link under the “Submit a Document” section on the home screen", "Click on login",
              "Files must be less than 100MB."]
    for quote in served:
        assert all(real_index.conflict_problem(cid, quote) is None for cid in _stamped_chunks_with(parts, quote)), quote


def test_the_upload_and_login_procedures_are_no_longer_blocked_but_the_requirements_list_is(real_index):
    structures = {s.heading: s for s in real_index.structures_of(ITHENTICATE) if s.kind == "list"}
    for heading in ("How to upload a document", "How to access IThenticate"):
        assert all(real_index.conflict_problem(p.chunk_id, p.quote) is None for p in structures[heading].parts), heading
    requirements = structures["File Requirements"]
    assert any(real_index.conflict_problem(p.chunk_id, p.quote) for p in requirements.parts)


@pytest.mark.parametrize("clause, served", [
    ("How do I upload a document to iThenticate?", True),
    ("How do I access iThenticate?", True),
    ("How many documents can I submit to iThenticate per year?", False),
    ("What is the yearly submission limit in iThenticate?", False),
    ("Is there a maximum number of submissions per year in iThenticate?", False),
])
def test_quota_questions_stay_blocked_and_procedures_are_served(real_index, parts, clause, served):
    binding, problem = real_index.bind(clause, ITHENTICATE)
    if binding is None:
        assert not served, problem
        return
    units = []
    for cid, meta in enumerate(parts.metadata):
        if meta.get("source_id") == ITHENTICATE:
            units.extend(GG.statement_units(cid, parts.chunks[cid], meta, 0, binding))
    answer, _ = GG.select_answer(binding, units, real_index, {p.key: 0 for p in real_index.procedures_of(ITHENTICATE)},
                                 {s.key: 0 for s in real_index.structures_of(ITHENTICATE)})
    assert (answer is not None) is served
    if answer is not None:
        text = " ".join(p.quote for p in answer.parts)
        assert "30" not in text and "per year" not in text.lower()


def test_an_unregistered_stamp_still_blocks_the_whole_chunk(real_index, parts):
    cid = _chunk_with(parts, ITHENTICATE, "Click on login", "paragraph")
    meta = dict(parts.metadata[cid], source_conflict="some_other_topic")
    index = GG.GuideIndex.__new__(GG.GuideIndex)
    index.metadata = {cid: meta}.get
    index.metadata = [{}] * cid + [meta]
    index.records = real_index.records
    assert "on the evidence chunk" in index.conflict_problem(cid, "2) Click on login")


def test_the_adapter_conflict_guard_is_claim_level_only_for_guide_findings(real_index, parts):
    page = _chunk_with(parts, ITHENTICATE, "Maximum 30 Documents per year", "paragraph")
    record = {"version": GG.VERSION, "form": GG.FORM_STATEMENT}
    neighbour = Finding("f1", "q", "2) Click on login", ITHENTICATE, "t", 1, metadata={"chunk_id": page, GG.GUIDE_KEY: record})
    quota = Finding("f2", "q", "Maximum 30 Documents per year", ITHENTICATE, "t", 1, metadata={"chunk_id": page, GG.GUIDE_KEY: record})
    plain = Finding("f3", "q", "2) Click on login", ITHENTICATE, "t", 1, metadata={"chunk_id": page})
    assert PF.source_conflict_problem(neighbour, parts.metadata, real_index) is None
    assert "not served" in PF.source_conflict_problem(quota, parts.metadata, real_index)
    assert PF.source_conflict_problem(plain, parts.metadata, real_index)                 # no D-111 record: the chunk rule
    assert PF.source_conflict_problem(neighbour, parts.metadata)                        # no guide index: the chunk rule


def test_the_journal_conflict_still_blocks_every_journal_unit_and_its_whole_chunk(real_index, parts):
    for structure in real_index.structures_of("uos_blackboard_communication_guide"):
        for part in structure.parts:
            if "journal" in part.quote.lower():
                assert real_index.conflict_problem(part.chunk_id, part.quote), part.quote
    stamped = [i for i, m in enumerate(parts.metadata) if m.get("source_conflict") == "blackboard_journals_usage_recommendation_2022"]
    assert stamped
    for cid in stamped:                                      # a recommendation conflict covers its chunk: no claim-level isolation
        sentence = parts.chunks[cid].split(". ")[0]
        assert real_index.conflict_problem(cid, sentence), sentence
    conflicts = {c.topic: c for r in parts.records for c in r.conflicts}
    assert GG.value_conflict(conflicts["ithenticate_yearly_submission_limit"])
    assert not GG.value_conflict(conflicts["blackboard_journals_usage_recommendation_2022"])


def test_the_real_collaborate_lettered_section_is_recovered_and_refused_because_its_text_is_cut(real_index):
    mixed = [s for s in real_index.structures_of(COLLABORATE) if s.list_type == GG.MIXED_PROCEDURE]
    assert len(mixed) == 1 and mixed[0].heading == "Create New Scheduled Sessions" and mixed[0].pages == (1, 2)
    assert [b.label for b in mixed[0].blocks if b.kind == "heading"] == ["1.", "2.", "3."]
    binding, _ = real_index.bind("How do I create new scheduled sessions in Blackboard Collaborate?", COLLABORATE)
    assert GG.structure_problem(mixed[0], binding, real_index) is not None


# ---------------------------------------------------------------------------
# F. Mixed-label procedures (synthetic)
# ---------------------------------------------------------------------------
MIXED = {1: ["A. Create New Sessions:", "1. Session Details:", G, "Click on Create Session.", G, "Name the session and set the start date.",
             "2. Session Settings:", G, "Select Allow recording downloads.", G, "Click Save to keep the settings.",
             "B. How to join the session:", G, "Click on join session.", G, "Select the session name."]}


def test_a_lettered_section_over_numbered_sub_sections_is_one_procedure_in_order():
    index = _index(MIXED)
    mixed = [s for s in index.structures_of(SHORT_GUIDE) if s.list_type == GG.MIXED_PROCEDURE]
    assert len(mixed) == 1 and mixed[0].heading == "Create New Sessions" and mixed[0].end.startswith("next letter B")
    assert [p.marker for p in mixed[0].parts] == ["heading", "numbered", "bullet", "bullet", "numbered", "bullet", "bullet"]
    for part in mixed[0].parts:
        assert part.quote in index.chunks[part.chunk_id]
    answer, _ = _answer(index, "How do I create new sessions in Blackboard?")
    assert answer is not None and answer.form == GG.FORM_SECTION and answer.list_type == GG.MIXED_PROCEDURE
    rendered = [GG.render_part(p.quote, answer.record(i)) for i, p in enumerate(answer.parts)]
    assert rendered == ["A. Create New Sessions:", "1. Session Details:", "- Click on Create Session.", "- Name the session and set the start date.",
                        "2. Session Settings:", "- Select Allow recording downloads.", "- Click Save to keep the settings."]
    assert GG.verify_answer("How do I create new sessions in Blackboard?", GG.FORM_SECTION, [(p.chunk_id, p.quote, p.page) for p in answer.parts],
                            answer.procedure_key, index) is None


@pytest.mark.parametrize("rows, why", [
    (["A. Create New Sessions:", "1. Session Details:", G, "Click on Create Session.", "3. Session Settings:", G, "Select Save.",
      "B. Join:", G, "Click join."], "numbering"),
    (["A. Create New Sessions:", "1. Session Details:", G, "Click on Create Session.", G, "If you want to control how late they can join and how much",
      "2. Session Settings:", G, "Select Save.", "B. Join:", G, "Click join."], "mid"),
    (["A. Create New Sessions:", "1. Session Details:", G, "Click on Create Session.", "2. Session Settings:", G, "Select Save.",
      "D. Join:", G, "Click join."], "end"),
    (["A. Create New Sessions:", "1. Session Details:", G, "Click on Create Session.", "2. Session Settings:", "B. Join:", G, "Click join."],
     "content"),
    (["A. Create New Sessions:", "1. Session Details:", G, "Click on Create Session.", "2. Session Settings:", G, "Select Save.",
      "B- To join the session you open the course and click on the session name.", G, "Click on join session."], "dash letter"),
])
def test_an_incomplete_mixed_procedure_is_refused(rows, why):
    index = _index({1: rows})
    answer, considered = _answer(index, "How do I create new sessions in Blackboard?")
    assert answer is None or answer.list_type != GG.MIXED_PROCEDURE
    reasons = " ".join(str(c.get("reason")) for c in considered if "M:" in str(c.get("structure")))
    assert reasons, considered


def test_a_wrapped_row_printed_with_its_own_bullet_is_joined_to_its_item():
    blocks = GS.merge_continuations(GS.read_blocks([G, "Activate the option to connect blackboard", G, "attendance service to the sessions.",
                                                    G, "Students stay in the session."], 1))
    assert [b.text for b in blocks] == ["Activate the option to connect blackboard attendance service to the sessions.",
                                        "Students stay in the session."]
    closed = GS.merge_continuations(GS.read_blocks([G, "Click Save.", G, "then select Done."], 1))
    assert len(closed) == 2                                                    # a closed item is never extended


@pytest.mark.parametrize("units, cut", [
    (["Click Save.", "Select Done."], False),
    (["Click Log in", "Select Sign out"], False),                              # verb particles are not cut words
    (["Change late after and absent after and how much"], True),
    (["If you want to control how late the attendees can join before they are"], True),
    (["Activate the option to connect blackboard", "attendance service to the sessions."], True),
    (["2. Automatically assign zeros to missing work", "3. view the gradebook categories"], False),    # a printed label is a new item
    (["- Automatically assign zeros to missing work", "- view the gradebook categories"], True),        # a bullet is not proof
])
def test_cut_units_are_detected(units, cut):
    assert (GG.cut_unit_problem(units) is not None) is cut


@pytest.mark.parametrize("quote, value", [
    ("Files must be less than 100MB.", True), ("Zip files may contain up to 200MB or 1,000 files.", True),
    ("Your title is limited to 255 characters.", True),
    ("You can grade from two views: 2. Student grid View.", False),          # a list label is not a quantity
])
def test_a_value_needs_a_number_with_its_unit_and_a_list_label_is_not_one(quote, value):
    binding = _terms("How many students can I add in Blackboard?")
    assert (GG._QUANTITY.search(GG._LEADING_LABEL.sub("", quote, count=1)) is not None) is value


# ---------------------------------------------------------------------------
# D. Multi-part (end to end with the fake embedder): all clauses or nothing
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
    rt, build = PF.build_runtime(kb, guide_scopes=True)
    assert build.ok
    return rt


SUPPORTED = "How do I send an announcement to my students in Blackboard?"


def test_the_supported_guide_clause_is_served(runtime):
    attempt = runtime.attempt(SUPPORTED)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED, attempt.reason


@pytest.mark.parametrize("question", [
    "What is the capital of France, and how do I send an announcement to my students in Blackboard?",          # first half unsupported
    "How do I send an announcement to my students in Blackboard? How many documents can I submit to iThenticate per year?",
    "How do I send an announcement to my students in Blackboard, and what is the attendance policy for students?",
])
def test_a_multi_part_question_with_an_unsupported_clause_falls_back_whole(runtime, question):
    attempt = runtime.attempt(question)
    assert attempt.outcome is not PF.PlanFOutcome.ACCEPTED and attempt.result is None, attempt.reason


def test_a_typo_routes_to_the_guide_registry_and_the_trace_keeps_no_typed_word(runtime):
    attempt = runtime.attempt("How do I send an announcement to my students in Blackbord?")
    entries = [e for e in attempt.trace if e.agent == "system_name_spelling"]
    assert entries and entries[0].detail == {"names": ["Blackboard"], "kinds": ["edit"]}
    assert "blackbord" not in json.dumps([e.detail for e in attempt.trace], default=str).lower()
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED, attempt.reason


def test_a_served_answer_records_the_typed_clause_not_the_corrected_one(runtime):
    attempt = runtime.attempt("How do I send an announcement to my students in Blackbord?")
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED
    clauses = {i["meta"]["plan_f"]["focus_clause"] for i in attempt.result.items}
    assert clauses == {"How do I send an announcement to my students in Blackbord?"}
    assert attempt.question == "How do I send an announcement to my students in Blackbord?"


@pytest.mark.parametrize("question, owner", [
    ("Where do I find the gradebook in Blackboard Ultra?", "teaching"),          # a guide only Teaching is approved for
    ("Where do I find the similarity report in iThenticate?", "research"),
    ("Where is the Blackboard support office?", None),                          # a physical office stays with Institutional
    ("Where is the ITC office for Blackboard help?", None),
    ("Where can I find the Grade Transfer Dashboard link?", None),              # a guide Institutional is approved for
])
def test_a_software_where_clause_routed_to_institutional_goes_to_the_guides_owner(runtime, question, owner):
    attempt = runtime.attempt(question)
    moved = [e.detail for e in attempt.trace if e.agent == "software_navigation"]
    assert moved == ([{"from": "institutional", "to": owner}] if owner else []), moved
    if owner:                                                     # the routing record (chip, diagnostics) names the new owner
        assert attempt.routing.final_specialists == (owner,)


def test_spelling_is_never_applied_without_the_guide_registry(kb):
    rt, build = PF.build_runtime(kb, guide_scopes=False)
    attempt = rt.attempt("How do I send an announcement to my students in Blackbord?")
    assert not any(e.agent == "system_name_spelling" for e in attempt.trace)
    attempt = rt.attempt("Where do I find the gradebook in Blackboard Ultra?")
    assert not any(e.agent == "software_navigation" for e in attempt.trace)          # the I-9 routing is untouched


# ---------------------------------------------------------------------------
# Held sources and the frozen fresh held-out set
# ---------------------------------------------------------------------------
def test_the_appraisal_guide_and_the_held_p1_guides_stay_out(parts):
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    appraisal = registry.get_source("uos_performance_appraisal_phase2_guide")
    assert not appraisal.enabled and appraisal.access == "internal_use"
    assert len(registry.enabled_sources()) == 8 and len(parts.chunks) == 15000
    named = SS.guide_sources_named("How do I start the Phase 2 progress review in the HR Portal appraisal?", registry.list_sources())
    assert named == []
    assert not any("assignment" in p.name or "categor" in p.name for p in (REPO / "data" / "guides").iterdir())


def test_the_fresh_held_out_set_is_frozen_and_never_edited():
    path = REPO / "qa" / "post_freeze_pf5b" / "fresh_held_out_questions.jsonl"
    labels = json.loads((REPO / "qa" / "post_freeze_pf5b" / "fresh_held_out_labels.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == labels["frozen"]["sha256"]
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(questions) == labels["frozen"]["questions"] == len(labels["labels"]) == 56
    assert [q["id"] for q in questions] == sorted(labels["labels"])
