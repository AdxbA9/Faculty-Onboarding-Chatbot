"""Teaching conservative SUPPORTED contract (Milestone 4, Step 4.8T-C; D-079).

Step 4.8T confirmed that Teaching marked a specific ask `supported` on evidence that was on the topic but did not settle
the question: "Can I hold my office hours online?" from "faculty must hold office hours". The ask gate
(`specialists/grounding.py`) now demands that ONE local unit state the ask's specifics with the answer, a normative
marker for a deontic yes/no proposition, a party with the asked action for a "who" ask, a number for a value ask and the
contents for a contents ask; an unrecognised ask is never `supported`; a pure place or contact ask is Institutional
Navigation's alone. General overview asks stay answerable and correct specific answers stay `supported`. The fixture
lines mirror the handbook. No model, no LLM, no network; handoffs are data, never executed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import grounding as G
from handbook_bot.agents.specialists import teaching as T
from handbook_bot.agents.specialists.contracts import FindingStatus, HandoffRequest, SpecialistId
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.agents.specialists.research import APPROVED_SOURCE_ID
from handbook_bot.agents.specialists.retrieval_intent import resolve_clause_retrieval_intent
from handbook_bot.sources import load_section_map, load_source_registry

import test_teaching_specialist as TS

REPO = Path(__file__).resolve().parents[1]
SUPPORTED, PARTIAL, NOT_FOUND, OUT_OF_SCOPE = (FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND,
                                               FindingStatus.OUT_OF_SCOPE)


@pytest.fixture(scope="module")
def registry_and_map():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    return registry.get_source(APPROVED_SOURCE_ID), load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))


@pytest.fixture(scope="module")
def teaching(registry_and_map):
    """The Teaching fixture pages of test_teaching_specialist (real handbook lines)."""
    return TS.make_specialist(TS.corpus_from(registry_and_map, TS.PAGES))


@pytest.fixture(scope="module")
def mini(registry_and_map):
    """A specialist over a few handbook lines chosen per test."""
    cache = {}

    def build(*pages):
        key = tuple((p["page"], tuple(p["rows"])) for p in pages)
        if key not in cache:
            cache[key] = TS.make_specialist(TS.corpus_from(registry_and_map, list(pages)))
        return cache[key]

    return build


def report(result, index=0):
    return result.metadata["clauses"][index]


def quotes(result):
    return [f.evidence_quote for f in result.findings]


def status(sp, question, **kw):
    return TS.run(sp, question, **kw).status


# ---- handbook lines used by the mini fixtures --------------------------------------------------------------------------
DIRECTED_STUDY = ("12.18.5 Direct Study/Independent Study: Directed/Independent Study refers to courses where students receive individual "
                  "supervision from faculty. Such courses must have an appropriate syllabus, learning outcomes, teaching/learning methods, and "
                  "assessment tools.")
COURSE_FILES = "12.18.3 Course Files Policy: Faculty members are responsible for maintaining and updating an E-Course File for each course they teach."
SYLLABUS_CONTENTS = ("12.13.1 Course Outline Instructors are responsible for preparing comprehensive syllabi that outline the course's objectives, "
                     "outcomes, content, teaching methods, evaluation criteria, references, and additional readings.")
INCOMPLETE_WEEK = ("• Any student receiving an Incomplete must remove this mark from their record within the first week after the commencement of "
                   "the next regular semester by sitting for a special examination, arranged with the instructor and department.")
ANNUAL_PLAN = ("• Annual Workload Plan: Each faculty member will develop an annual workload plan in consultation with their department chair, "
               "outlining their teaching, research, service, and administrative commitments for the upcoming academic year.")
WORKLOAD_REVIEW = "• The workload shall be reviewed periodically by the Deans' Council."
SERVICE_HOURS = ("• University and community service-related activities' expectations are four (4) hours per week, on average, excluding regular "
                 "office hours and academic advising.")
ABSENCE_RULE = "3. Absences reaching 20% will prohibit the student from sitting for the final examination(s)."
FINAL_EXAM_SHARE = ("12.13.3.1 Purely or mainly theoretical courses: 1. 40-50% of the final grade shall be allocated to the final examination, a "
                    "comprehensive written test covering the entire course content and fulfilling its objectives and outcomes.")
CHAIRS_ASSESS = ("8. Summary of student feedback on the course evaluation. Department Chairs are responsible for assessing the contents and quality of "
                 "course files prepared by faculty members in their departments at the end of each term.")
ONLINE_DESCRIPTIVE = "Online Courses: Courses offered fully online use an LMS or web presence to enhance the course."
ONLINE_NORMATIVE = "Courses may be offered fully online with the approval of the department council."
LECTURER_NO_NUMBER = "The teaching load for lecturers is set by the department at the start of each semester."
APPROVAL_NOUN = "Course modifications require the approval of the department council before submission to the college council."
ATTENDANCE_DEPARTMENT = "Attendance is recorded by the department for each course and reviewed at the end of the semester."
GRADES_INTO_BLACKBOARD = ("Faculty members shall directly enter the grades electronically into the blackboard which will forward it automatically "
                          "to the registration system (Banner).")
TWO_SENTENCES = "Faculty members hold regular office hours during the semester. Consultations may also take place online."


# ---------------------------------------------------------------------------
# The ask gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,shape", [
    ("What are faculty office-hour responsibilities?", G.GENERAL),
    ("Tell me about course syllabi.", G.GENERAL),
    ("What is the curriculum committee?", G.GENERAL),
    ("What does the LMS policy require?", G.GENERAL),
    ("Can my office hours be held online?", G.SPECIFIC_RECOGNISED),
    ("Must my syllabus include assessment weights?", G.SPECIFIC_RECOGNISED),
    ("How many office hours must I hold?", G.SPECIFIC_RECOGNISED),
    ("Who approves a course modification?", G.SPECIFIC_RECOGNISED),
    ("When is the deadline to submit final grades?", G.SPECIFIC_RECOGNISED),
    ("What is the teaching load for lecturers?", G.SPECIFIC_RECOGNISED),
    ("How is the teaching load reduced for department chairs?", G.SPECIFIC_RECOGNISED),
    ("Why must I hold office hours?", G.SPECIFIC_UNKNOWN),
    ("What kind of syllabus is expected?", G.SPECIFIC_UNKNOWN),
])
def test_ask_shapes(clause, shape):
    assert T.assess_ask(clause).shape == shape


def test_specific_terms_polarity_party_and_value_are_read_from_the_clause():
    online = T.assess_ask("Can my office hours be held online?")
    assert online.yes_no and online.deontic and "office_hours" in online.concepts and "online" in online.concepts
    library = T.assess_ask("May I hold office hours in the library instead of my office?")
    assert library.deontic and set(library.terms) >= {"library", "instead"}
    who = T.assess_ask("Who assigns office hours to part-time faculty?")
    assert who.party and who.action == "assigns" and "assign" not in who.terms
    value = T.assess_ask("What is the teaching load for lecturers?")
    assert value.value and "lecturer" not in value.concepts           # a role is a role, not a concept (D-080)
    assert T.ASK_GATE.lexicon.roles_in("What is the teaching load for lecturers?") == {"lecturer"}
    year = T.assess_ask("Is peer observation of teaching mandatory every year?")
    assert year.deontic and "year" in year.terms                     # a period asked about is a specific, generic or not


def test_an_unrecognised_ask_receives_a_detail_no_evidence_covers():
    ask = T.assess_ask("Why must I hold office hours?")
    assert ask.shape == G.SPECIFIC_UNKNOWN and [d.family for d in ask.details] == ["ask_unknown"]
    assert not ask.details[0].check("Faculty members must hold regular office hours because students need them.")


def test_grounding_check_reads_the_unit():
    check = T.assess_ask("Can my office hours be held online?").details[0].check
    assert not check("Faculty members must commit to maintaining regular office hours.")     # topic only
    assert not check("Office hours are held online by many faculty members.")                # no rule stated
    assert check("Office hours may be held online with the chair's approval.")
    who = T.assess_ask("Who approves a course modification?").details[0].check
    assert not who("Course modifications follow the curricula approval process.")           # no party with the action
    assert who("The department council must approve any course modification.")
    assert who(APPROVAL_NOUN)                                                              # the noun bound to the party
    assert not who("Course modifications are listed in the teaching assignments of the council.")


def test_verb_forms_not_derived_nouns_state_a_party_action():
    words = ["office", "hours", "are", "prorated", "based", "on", "teaching", "assignments"]
    assert not G.verb_stated("assigns", " ".join(words), words)
    assert G.verb_stated("assigns", "the chair assigns office hours", ["the", "chair", "assigns", "office", "hours"])
    assert G.verb_stated("reviews", "files are assessed by chairs", ["files", "are", "assessed", "by", "chairs"])


# ---------------------------------------------------------------------------
# The five Step 4.8 observations
# ---------------------------------------------------------------------------
def test_a_office_hours_online_is_not_settled_by_the_office_hours_rule(teaching):
    result = TS.run(teaching, "Can I hold my office hours online?")
    assert result.status is not SUPPORTED and result.findings                # evidence kept, status honest
    assert any("does not settle the question" in m for m in result.missing)
    assert report(result)["ask_shape"] == G.SPECIFIC_RECOGNISED


def test_b_changing_a_textbook_is_not_settled_by_course_file_text(teaching):
    assert status(teaching, "Can I change the textbook of a course I teach?") is not SUPPORTED


def test_c_syllabus_contents_need_a_contents_statement(mini, teaching):
    unrelated = mini(TS.page(229, DIRECTED_STUDY), TS.page(231, COURSE_FILES))      # owned, unshared pages
    result = TS.run(unrelated, "What must a course syllabus contain?")
    assert result.status is not SUPPORTED and result.findings                   # the obligation sentence is quoted, not settled
    assert any("Such courses must have an appropriate syllabus" in q for q in quotes(result))
    stated = mini(TS.page(226, SYLLABUS_CONTENTS))
    good = TS.run(stated, "What must a course syllabus contain?")
    assert good.status is SUPPORTED and "outline" in quotes(good)[0]
    assert status(teaching, "What must the syllabus contain?") is SUPPORTED     # the fixture states the contents


def test_d_chair_load_reduction_is_not_settled_by_general_workload_text(teaching):
    result = TS.run(teaching, "How is the teaching load reduced for department chairs?")
    assert result.status is not SUPPORTED


def test_e_the_committee_office_is_institutionals_and_the_handoff_stays(teaching):
    result = TS.run(teaching, "Where is the curriculum committee office?")
    assert result.status is OUT_OF_SCOPE and not result.findings
    assert report(result)["outcome"] == "handed_off" and report(result)["target"] == "institutional"
    assert [h.requested_specialist for h in result.handoff_requests] == [SpecialistId.INSTITUTIONAL]
    assert all(isinstance(h, HandoffRequest) for h in result.handoff_requests)


def test_e_through_the_coordinator_and_dispatcher(teaching):
    registry = SpecialistRegistry()
    registry.register(teaching)
    result = D.run_question("Where is the curriculum committee office?", registry)
    run = next(r for r in result.runs if r.specialist_id is SpecialistId.TEACHING)
    assert run.findings.status is OUT_OF_SCOPE and report(run.findings)["outcome"] == "handed_off"
    assert SpecialistId.INSTITUTIONAL in result.pending_specialists


# ---------------------------------------------------------------------------
# Fresh confirmed classes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "May I hold office hours in the library instead of my office?",
    "Am I allowed to teach more than 15 credit hours a semester?",
    "Is a mid-semester grade report required for every course?",
    "Do I have to submit my course file before the final exam?",
    "Is peer observation of teaching mandatory every year?",
    "Can students see attendance records in Blackboard?",
    "Is LMS training mandatory for new faculty?",
])
def test_yes_no_propositions_are_not_settled_by_topic_evidence(teaching, question):
    assert status(teaching, question) is not SUPPORTED


def test_who_assigns_part_time_office_hours_is_not_the_proration_rule(teaching):
    result = TS.run(teaching, "Who assigns office hours to part-time faculty?")
    assert result.status is not SUPPORTED


def test_grade_submission_deadline_is_not_the_incomplete_removal_week(mini, teaching):
    only_incomplete = mini(TS.page(225, INCOMPLETE_WEEK))
    assert status(only_incomplete, "When must final grades be submitted after the exam?") is not SUPPORTED
    assert status(teaching, "When is the deadline to submit final grades?") is SUPPORTED     # the 48-hour rule


def test_who_reviews_workload_allocation_needs_the_review_stated(mini):
    only_plan = mini(TS.page(96, ANNUAL_PLAN))
    assert status(only_plan, "Who reviews the workload allocation?") is not SUPPORTED
    reviewed = mini(TS.page(92, WORKLOAD_REVIEW))
    good = TS.run(reviewed, "Who reviews the workload allocation?")
    assert good.status is SUPPORTED and "Deans' Council" in quotes(good)[0]


def test_student_work_hours_are_not_the_service_hours(mini):
    service = mini(TS.page(92, SERVICE_HOURS))
    assert status(service, "How many hours of student work per week does a three-credit course assume?") is not SUPPORTED


def test_the_course_file_portal_is_navigation(teaching):
    result = TS.run(teaching, "Where do I find the course file portal?")
    assert report(result)["outcome"] == "handed_off" and report(result)["target"] == "institutional"


# ---------------------------------------------------------------------------
# Deontic marker, value, contents, location, table rows, antecedents
# ---------------------------------------------------------------------------
def test_a_deontic_proposition_needs_a_rule_not_a_description(mini):
    descriptive = mini(TS.page(193, ONLINE_DESCRIPTIVE))
    assert status(descriptive, "Can a course be delivered fully online?") is not SUPPORTED
    normative = mini(TS.page(193, ONLINE_NORMATIVE))
    assert status(normative, "Can a course be delivered fully online?") is SUPPORTED


def test_a_calendar_row_never_settles_a_deontic_proposition(teaching):
    # "02 - 12 May ... Final exams" is a calendar line: the month is not the modal "may", and a table row states no rule
    assert G._NORMATIVE.search("Sat - Tue 02 - 12 May 15 -25 Dhi Al- Final exams") is None
    assert G._NORMATIVE.search("Office hours may be held online.")
    result = TS.run(teaching, "Do I have to submit my course file before the final exam?")
    assert result.status is not SUPPORTED


def test_a_value_ask_needs_a_number(mini, teaching):
    assert status(mini(TS.page(96, LECTURER_NO_NUMBER)), "What is the teaching load for lecturers?") is not SUPPORTED
    good = TS.run(teaching, "What is the teaching load for lecturers?")
    assert good.status is SUPPORTED and "15 credit hours" in quotes(good)[0]


def test_a_party_action_may_be_stated_as_a_noun_bound_to_the_party(mini, teaching):
    assert status(mini(TS.page(229, APPROVAL_NOUN)), "Who approves a course modification?") is SUPPORTED
    assert status(teaching, "Who approves a course modification?") is SUPPORTED          # "to approve"


def test_location_evidence_is_a_system_or_a_place_not_a_bare_department(mini, teaching):
    result = TS.run(mini(TS.page(228, ATTENDANCE_DEPARTMENT)), "Where is attendance recorded?")
    assert result.status is not SUPPORTED and result.findings                 # on the topic, but "the department" names no place
    good = TS.run(teaching, "Where must syllabi be uploaded?")
    assert good.status is SUPPORTED and any("uploaded on Blackboard" in q for q in quotes(good))


def test_a_table_row_names_at_least_one_specific(teaching):
    good = TS.run(teaching, "What percentage corresponds to grade A in the grading system?")
    assert good.status is SUPPORTED and any("A 90-100" in q for q in quotes(good))


def test_a_pronoun_initial_sentence_is_grounded_with_its_antecedent_only(teaching, mini):
    good = TS.run(teaching, "What syllabus requirements apply to directed study courses?")
    assert good.status is SUPPORTED and any("Such courses must have an appropriate syllabus" in q for q in quotes(good))
    split = mini(TS.page(79, TWO_SENTENCES))
    assert status(split, "Can office hours be held online?") is not SUPPORTED          # two plain sentences never combine


def test_an_unknown_ask_is_never_supported(teaching):
    result = TS.run(teaching, "Why must I hold office hours?")
    assert result.status is not SUPPORTED and report(result)["ask_shape"] == G.SPECIFIC_UNKNOWN
    assert any("not recognised" in m for m in result.missing)


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,needle", [
    ("How many office hours must full-time faculty hold?", "five office hours"),
    ("What is the minimum number of office hours per week?", "minimum of five"),
    ("Is attendance required in laboratory sessions?", "required to attend"),
    ("Must I enter grades into Blackboard?", "enter the grades"),
    ("What is the teaching load for Regular Faculty?", '"A" Regular Faculty'),
    ("What are my office hour obligations?", "office hours"),
    ("Tell me about the LMS policy.", "Learning Management System"),
    ("What does the LMS policy require?", "LMS"),
    ("Must syllabi be uploaded on Blackboard?", "uploaded on Blackboard"),
    ("How do grades entered in Blackboard reach Banner?", "Banner"),
])
def test_correct_specific_and_general_answers_stay_supported(teaching, question, needle):
    result = TS.run(teaching, question)
    assert result.status is SUPPORTED, (question, result.missing)
    assert any(needle in q for q in quotes(result)), quotes(result)


def test_absence_percentage_final_exam_share_and_chairs_stay_supported(mini):
    absences = TS.run(mini(TS.page(228, ABSENCE_RULE)), "What percentage of absences bars a student from the final exam?")
    assert absences.status is SUPPORTED and "20%" in quotes(absences)[0]
    share = TS.run(mini(TS.page(227, FINAL_EXAM_SHARE)), "What percentage of the final grade may the final exam carry?")
    assert share.status is SUPPORTED and "40-50%" in quotes(share)[0]
    chairs = TS.run(mini(TS.page(231, CHAIRS_ASSESS)), "Who reviews course files at the end of the semester?")
    assert chairs.status is SUPPORTED and "Department Chairs" in quotes(chairs)[0]


# ---------------------------------------------------------------------------
# Navigation, contact halves and D-078
# ---------------------------------------------------------------------------
def test_a_mixed_clause_keeps_its_substantive_half_and_misses_the_contact_half(teaching):
    result = TS.run(teaching, "How many office hours must I hold and what is the Registrar's phone number?")
    assert result.status is PARTIAL and any("five office hours" in q for q in quotes(result))
    assert any("Institutional Navigation holds them" in m for m in result.missing)
    assert SpecialistId.INSTITUTIONAL in {h.requested_specialist for h in result.handoff_requests}


def test_d078_routes_are_unchanged_by_the_gate(teaching):
    clauses = ["How many office hours must I hold", "what is the Registrar's phone number"]
    result = TS.run(teaching, " and ".join(clauses) + "?", clauses=clauses, intent="contact")
    reports = {r["clause"]: r for r in result.metadata["clauses"]}
    assert reports[clauses[0]]["route"] == "count" and reports[clauses[0]]["outcome"] == "supported"
    assert reports[clauses[1]]["outcome"] == "handed_off" and reports[clauses[1]]["target"] == "institutional"
    assert resolve_clause_retrieval_intent(clauses[1], "contact") == "contact"
    dated = TS.run(teaching, "When do Fall classes begin?", intent="date")
    assert report(dated)["route"] == "date" and dated.status is SUPPORTED


def test_navigation_intent_reads_places_and_channels_not_postings():
    assert T.navigation_intent("Where is the curriculum committee office?") == (True, False)
    assert T.navigation_intent("What is the phone number of the Blackboard support team?") == (True, False)
    assert T.navigation_intent("Where do I find the course file portal?") == (True, False)
    assert T.navigation_intent("Where is the syllabus posted?") == (False, True)
    assert T.navigation_intent("Where must the course file be kept?") == (False, True)
    assert T.navigation_intent("Which Blackboard menu shows the attendance screen?") == (False, True)
    assert T.navigation_intent("How many office hours must I hold and what is the Registrar's phone number?") == (True, True)
