"""Teaching residual grounding (Milestone 4, Step 4.8T-C3; D-081).

Step 4.8T-R2 found five residual classes after D-080: the person asking was exempt from role grounding (a student rule
answered "Must I attend the final examination?"); a subject-less occurrence counted for a role named only as a
recipient ("distributed to students ... and uploaded"); a passive ask lost its agent before "of" or "whose"; the light
verbs receive / get / obtain bound neither recipient nor source; and "semester" or "day" vanished from frequency asks.
The fixture lines mirror the handbook and sit on owned, unshared Teaching pages. No model, no LLM, no network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from handbook_bot.agents.specialists import grounding as G
from handbook_bot.agents.specialists import teaching as T
from handbook_bot.agents.specialists.contracts import FindingStatus
from handbook_bot.agents.specialists.research import APPROVED_SOURCE_ID
from handbook_bot.sources import load_section_map, load_source_registry

import test_teaching_specialist as TS

REPO = Path(__file__).resolve().parents[1]
SUPPORTED, PARTIAL = FindingStatus.SUPPORTED, FindingStatus.PARTIAL

FAIL_ATTEND = "Failure to attend the final examination without an acceptable excuse may result in a failing grade for the course."
STUDENTS_ATTEND = "1. Students are required to attend all theoretical lectures, laboratory hours, training sessions, research sessions, and examinations."
GRADES_FLOW = ("Faculty members shall directly enter the grades electronically into the blackboard which will forward it automatically to "
               "the registration system (Banner).")
IMPERATIVE = "Enter final grades in Blackboard within forty-eight hours of the examination date."
INCOMPLETE = ("• Any student receiving an Incomplete must remove this mark from their record within the first week after the commencement "
              "of the next regular semester by sitting for a special examination, arranged with the instructor and department.")
REPORT_PAIR = "Faculty submit the grade report to the Registrar. The Registrar receives the grade report from the instructor after the examination."
REPORT_DEAN = "The Registrar receives the grade report from the instructor and forwards a copy to the dean."
OBTAIN = "Students obtain the syllabus from Blackboard at the start of the semester."
SYLLABI_DISTRIBUTED = ("These syllabi shall be distributed to students at the beginning of the semester and uploaded on Blackboard and kept in "
                       "the course files within the college.")
COORDINATED = "Faculty prepare, upload and keep the syllabus in the course files of the college."
DISTRIBUTED_BY = "The syllabus is distributed to students, uploaded by faculty and kept in the course files."
HOLD_FOR = "Faculty members hold office hours for students every week."
CHAIRS_ASSESS = ("Department Chairs are responsible for assessing the contents and quality of course files prepared by faculty members in "
                 "their departments at the end of each term.")
SENT = "The grade report is sent to the Registrar by the instructor after the examination."
INFORMED = "Students are informed by the instructor of any change to the office hours."
INFORMED_MUST = "Students must be informed by the instructor of any change to the office hours."
FULL_TIME = ("1. Full-Time Faculty: Full-time faculty members in academic and career/technical programs are required to schedule a minimum of "
             "five office hours per week, spread over a minimum of two days.")
WORKLOAD_REVIEW = "• The workload shall be reviewed periodically by the Deans' Council."
ANNUAL = "• Annual Evaluation: At the end of each academic year, the department chair evaluates faculty performance against the workload plan."
CHAIRS_SEMESTER = "Department chairs assess course files at the end of each semester."
LECTURERS = "Lecturers teach 15 credit hours per semester."
PUBLISH = "The University publishes the grading scale; grades are entered in Banner by the instructor."


@pytest.fixture(scope="module")
def registry_and_map():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    return registry.get_source(APPROVED_SOURCE_ID), load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))


@pytest.fixture(scope="module")
def mini(registry_and_map):
    cache = {}

    def build(page_no, *lines):
        key = (page_no, lines)
        if key not in cache:
            cache[key] = TS.make_specialist(TS.corpus_from(registry_and_map, [TS.page(page_no, *lines)]))
        return cache[key]

    return build


def status(sp, question):
    return TS.run(sp, question).status


def relation(question):
    return G.parse_relation(question, T.ASK_GATE.lexicon)


# ---------------------------------------------------------------------------
# 1. The person asking is a role like any other
# ---------------------------------------------------------------------------
def test_1_the_self_role_cannot_inherit_a_student_rule(mini):
    student_rule = mini(225, FAIL_ATTEND)
    assert status(student_rule, "Must I attend the final examination?") is PARTIAL
    assert status(student_rule, "Am I required to attend the final examination?") is PARTIAL
    assert status(student_rule, "Must instructors attend the final examination?") is PARTIAL
    assert status(mini(228, STUDENTS_ATTEND), "Must I attend all theoretical lectures?") is PARTIAL
    assert T.ASK_GATE.relations.agent_roles(FAIL_ATTEND) == frozenset()


def test_1_the_self_role_is_supported_where_faculty_is_the_subject(mini):
    assert status(mini(224, GRADES_FLOW), "Must I enter grades into Blackboard?") is SUPPORTED
    assert "faculty" in T.ASK_GATE.relations.agent_roles(GRADES_FLOW)
    imperative = mini(224, IMPERATIVE)
    assert status(imperative, "Must I enter final grades in Blackboard within forty-eight hours?") is SUPPORTED   # an imperative addresses the reader
    assert status(imperative, "Must students enter final grades in Blackboard within forty-eight hours?") is PARTIAL
    assert T.ASK_GATE.relations.agent_roles(IMPERATIVE) == frozenset({"faculty"})
    assert T.ASK_GATE.assess("Must I attend the final examination?").roles == frozenset({"faculty"})


# ---------------------------------------------------------------------------
# 2. A subject-less occurrence never counts for a role named only as recipient or object
# ---------------------------------------------------------------------------
def test_2_recipients_do_not_become_agents(mini):
    sp = mini(229, SYLLABI_DISTRIBUTED)
    assert status(sp, "Do students upload the syllabi on Blackboard?") is PARTIAL
    assert status(sp, "Must students keep the syllabi in the course files?") is PARTIAL
    assert status(sp, "Are syllabi kept by students in the course files?") is PARTIAL
    assert status(sp, "Are syllabi uploaded on Blackboard?") is SUPPORTED                       # no actor asked
    assert status(sp, "Are the syllabi distributed to students at the beginning of the semester?") is SUPPORTED
    assert T.ASK_GATE.relations.agent_roles(SYLLABI_DISTRIBUTED) == frozenset()


def test_2_a_governing_subject_binds_coordinated_verbs(mini):
    coordinated = mini(229, COORDINATED)
    assert status(coordinated, "Do faculty upload the syllabus?") is SUPPORTED
    assert status(coordinated, "Do faculty keep the syllabus in the course files?") is SUPPORTED
    distributed = mini(229, DISTRIBUTED_BY)
    assert status(distributed, "Do students upload the syllabus?") is PARTIAL
    assert status(distributed, "Do students keep the syllabus in the course files?") is PARTIAL
    assert status(distributed, "Is the syllabus uploaded by faculty?") is SUPPORTED
    held = mini(79, HOLD_FOR)
    assert status(held, "Do students hold office hours?") is PARTIAL                            # a light verb: the actor must be a subject
    assert status(held, "Do faculty members hold office hours?") is SUPPORTED


# ---------------------------------------------------------------------------
# 3. The passive agent phrase survives "of", "whose", "to", "for" and "from"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,actor,target_role,prep", [
    ("Are course files assessed by faculty members of the department?", "faculty", None, None),
    ("Is the syllabus prepared by instructors whose courses are online?", "faculty", None, None),
    ("Is the workload plan approved by the Dean of the College?", "dean", None, None),
    ("Are course files reviewed by chairs of departments?", "chair", None, None),
    ("Is the grade report sent by the instructor to the Registrar?", "faculty", "registrar", "to"),
    ("Is the syllabus prepared by the instructor for each course?", "faculty", None, "for"),
    ("Is the grade report received by the Registrar from the instructor?", "registrar", "faculty", "from"),
])
def test_3_the_agent_phrase_is_kept(clause, actor, target_role, prep):
    rel = relation(clause)
    assert rel is not None and rel.passive and rel.actor == actor
    assert rel.target_role == target_role and rel.target_prep == prep


def test_3_a_lost_agent_cannot_borrow_another_agents_action(mini):
    chairs = mini(231, CHAIRS_ASSESS)
    assert status(chairs, "Are course files assessed by faculty members of the department?") is PARTIAL
    assert status(chairs, "Are course files assessed by the department chair?") is SUPPORTED
    assert status(chairs, "Are course files prepared by faculty members of the department?") is SUPPORTED
    sent = mini(224, SENT)
    assert status(sent, "Is the grade report sent by the instructor to the Registrar?") is SUPPORTED
    assert status(sent, "Is the grade report sent by the Registrar to the instructor?") is PARTIAL
    informed = mini(79, INFORMED)
    assert status(informed, "Is the instructor informed by students of changes to office hours?") is PARTIAL
    assert status(informed, "Does the instructor inform students about office hours?") is SUPPORTED
    must = mini(79, INFORMED_MUST)
    assert status(must, "Are instructors informed of changes to office hours?") is PARTIAL        # the patient is students, not the unit
    assert status(must, "Are students informed of changes to office hours?") is SUPPORTED


# ---------------------------------------------------------------------------
# 4. receive / get / obtain bind recipient, object and source
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", [
    "Does the instructor receive the grade report from the Registrar?", "Do students receive an Incomplete grade?",
    "Did the chair get the workload plan from the dean?", "Do faculty members obtain the syllabus from Blackboard?",
    "Has the Registrar received the grade report from the instructor?",
])
def test_4_receive_is_an_action_with_a_source(clause):
    rel = relation(clause)
    assert rel is not None and rel.action == "receive" and rel.obj
    if " from " in clause:
        assert rel.target_prep == "from" and rel.target_role is not None


def test_4_direction_of_receiving_is_kept(mini):
    pair = mini(224, REPORT_PAIR)
    assert status(pair, "Does the instructor receive the grade report from the Registrar?") is PARTIAL
    assert status(pair, "Does the Registrar receive the grade report from the instructor?") is SUPPORTED
    dean = mini(224, REPORT_DEAN)
    assert status(dean, "Does the Registrar receive the grade report from the dean?") is PARTIAL    # the dean is a recipient, not the source
    assert status(dean, "Does the Registrar receive the grade report from the instructor?") is SUPPORTED
    incomplete = mini(225, INCOMPLETE)
    assert status(incomplete, "Do instructors receive an Incomplete grade?") is PARTIAL
    assert status(incomplete, "Do I receive an Incomplete grade?") is PARTIAL
    assert status(incomplete, "Do students receive an Incomplete grade?") is SUPPORTED
    obtain = mini(229, OBTAIN)
    assert status(obtain, "Do students get the syllabus from Blackboard?") is SUPPORTED
    assert status(obtain, "Does Blackboard obtain the syllabus from students?") is PARTIAL


# ---------------------------------------------------------------------------
# 5. Frequency units cannot migrate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,units", [
    ("five office hours per week", {"week"}), ("reviewed periodically", {"periodic"}), ("At the end of each academic year", {"year"}),
    ("Annual Evaluation", {"year"}), ("15 credit hours per semester", {"semester"}), ("weekly meetings", {"week"}),
    ("at the beginning of the semester", set()), ("every day", {"day"}), ("each term", {"term"}),
])
def test_5_frequencies_are_read(text, units):
    assert G.frequencies_of(text) == frozenset(units)


def test_5_per_week_is_neither_per_semester_nor_per_day(mini):
    sp = mini(79, FULL_TIME)
    assert status(sp, "Must full-time faculty schedule five office hours per semester?") is PARTIAL
    assert status(sp, "Must full-time faculty schedule five office hours per day?") is PARTIAL
    assert status(sp, "Must full-time faculty schedule five office hours per week?") is SUPPORTED
    assert status(sp, "Must full-time faculty schedule five office hours weekly?") is SUPPORTED


def test_5_periodically_proves_no_exact_frequency(mini):
    sp = mini(92, WORKLOAD_REVIEW)
    assert status(sp, "Is the workload reviewed every semester by the Deans' Council?") is PARTIAL
    assert status(sp, "Is the workload reviewed each semester?") is PARTIAL
    assert status(sp, "Is the workload reviewed annually by the Deans' Council?") is PARTIAL
    assert status(sp, "Is the workload reviewed periodically by the Deans' Council?") is SUPPORTED


def test_5_exact_units_and_their_synonyms(mini):
    annual = mini(96, ANNUAL)
    assert status(annual, "Does the chair evaluate performance against the workload plan annually?") is SUPPORTED
    assert status(annual, "Does the chair evaluate performance against the workload plan every year?") is SUPPORTED
    assert status(annual, "Does the chair evaluate performance against the workload plan each semester?") is PARTIAL
    chairs = mini(231, CHAIRS_SEMESTER)
    assert status(chairs, "Do department chairs assess course files at the end of each month?") is PARTIAL
    assert status(chairs, "Do department chairs assess course files at the end of each semester?") is SUPPORTED
    lecturers = mini(96, LECTURERS)
    assert status(lecturers, "Do lecturers teach 15 credit hours per year?") is PARTIAL
    assert status(lecturers, "Do lecturers teach 15 credit hours per semester?") is SUPPORTED


# ---------------------------------------------------------------------------
# The constructed MINOR of Step 4.8T-R2 and the earlier classes
# ---------------------------------------------------------------------------
def test_publish_is_not_enter(mini):
    sp = mini(224, PUBLISH)
    assert status(sp, "Does the University enter grades in Banner?") is PARTIAL
    assert status(sp, "Does the instructor enter grades in Banner?") is SUPPORTED


def test_4_a_source_beside_a_destination_is_bound_for_every_action(mini):
    flow = mini(224, GRADES_FLOW)
    assert status(flow, "Are grades entered from Banner into Blackboard?") is PARTIAL              # Banner is the destination of the flow
    assert status(flow, "Are grades forwarded from Blackboard to Banner?") is SUPPORTED            # Blackboard is the agent: the source
    assert status(flow, "Are grades forwarded from the Registrar to Banner?") is PARTIAL
    rel = relation("Are grades entered from Banner into Blackboard?")
    assert rel.source == ("banner",) and rel.source_role == "banner" and rel.target_role == "blackboard"


def test_earlier_position_classes_still_hold(mini):
    assert status(mini(228, STUDENTS_ATTEND), "Must faculty attend all theoretical lectures?") is PARTIAL
    assert status(mini(228, STUDENTS_ATTEND), "Must students attend all theoretical lectures?") is SUPPORTED
    assert status(mini(224, GRADES_FLOW), "Does Banner forward grades to Blackboard?") is PARTIAL
    assert status(mini(224, GRADES_FLOW), "Does the Registrar enter grades into Blackboard?") is PARTIAL
    assert status(mini(231, CHAIRS_ASSESS), "Does the dean review course files?") is PARTIAL
    assert status(mini(231, CHAIRS_ASSESS), "Who reviews course files?") is SUPPORTED
