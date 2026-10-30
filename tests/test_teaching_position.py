"""Teaching subject / position grounding (Milestone 4, Step 4.8T-C2; D-080).

Step 4.8T-R showed that the asked words could all appear in a unit in the wrong roles: "Must faculty attend all
theoretical lectures?" was `supported` by the rule that students must attend them. The grounding gate now binds the
asked actor to the asked action on the asked object in one evidence clause: a different named subject disqualifies,
direction is kept ("Blackboard forwards grades to Banner" does not state the reverse), roles are matched by an injected
role lexicon (a dean is not a chair, a registrar is not a registration system), an antecedent cannot lend its actor to
the next sentence's action, "absence of errors" is not attendance, and a calendar line is no percentage. The fixture
lines mirror the handbook and sit on owned, unshared Teaching pages. No model, no LLM, no network.
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
SUPPORTED, PARTIAL, NOT_FOUND = FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND

STUDENTS_ATTEND = ("1. Students are required to attend all theoretical lectures, laboratory hours, training sessions, research sessions, and "
                   "examinations for the courses they are enrolled in.")
CHAIRS_ASSESS = ("Department Chairs are responsible for assessing the contents and quality of course files prepared by faculty members in "
                 "their departments at the end of each term.")
GRADES_FLOW = ("Faculty members shall directly enter the grades electronically into the blackboard which will forward it automatically to "
               "the registration system (Banner).")
APPROVAL_FLOW = ("• Approval Process: The department chair will review and approve the workload plan, ensuring it aligns with departmental "
                 "goals and university policies. Final approval rests with the dean.")
INCOMPLETE = ("• Any student receiving an Incomplete must remove this mark from their record within the first week after the commencement "
              "of the next regular semester by sitting for a special examination, arranged with the instructor and department.")
SYLLABUS_PAIR = "Instructors prepare the course syllabus before the semester begins. These syllabi are approved by the department council."
ERRORS = "The review shall verify the absence of material errors in grading, question components, or partial/complete question corrections."
ABSENCES = "3. Absences reaching 20% will prohibit the student from sitting for the final examination(s)."
CALENDAR = "Sat - Tue 02 - 12 May 15 -25 Dhi Al- Final exams"
FINAL_SHARE = ("12.13.3.1 Purely or mainly theoretical courses: 1. 40-50% of the final grade shall be allocated to the final examination, a "
               "comprehensive written test covering the entire course content.")
GRADE_ROWS = ("12.14 Grading System:", "Grades Percentages Points", "A 90-100 4.00", "B+ 85-89 3.50", "F Below 60 0.00")
WORKLOAD_REVIEW = "• The workload shall be reviewed periodically by the Deans' Council."


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


def quotes(result):
    return [f.evidence_quote for f in result.findings]


# ---------------------------------------------------------------------------
# The role lexicon and the relation parser
# ---------------------------------------------------------------------------
def test_role_lexicon_separates_roles_and_reads_composites():
    roles = T.ASK_GATE.lexicon.roles_in
    assert roles("the dean") == {"dean"} and roles("the department chair") == {"chair"}
    assert roles("the Deans' Council") == {"council"}
    assert roles("the Registrar") == {"registrar"} and roles("the registration system (Banner)") == {"registration", "banner"}
    assert roles("faculty members and instructors") == {"faculty"} and roles("lecturers") == {"lecturer"} and roles("students") == {"student"}
    assert T.ASK_GATE.lexicon.self_role == "faculty"


@pytest.mark.parametrize("clause,actor,action,obj,target_role", [
    ("Must faculty attend all theoretical lectures?", "faculty", "attend", ("theoretical", "lectures"), None),
    ("Does the dean review course files?", "dean", "review", ("course", "files"), None),
    ("Does the Registrar enter grades into Blackboard?", "registrar", "enter", ("grades",), "blackboard"),
    ("Does Banner forward grades to Blackboard?", "banner", "forward", ("grades",), "blackboard"),
    ("Does the chair give final approval of the workload plan?", "chair", "approve", ("workload", "plan"), None),
    ("Must instructors remove an Incomplete within the first week?", "faculty", "remove", ("incomplete",), None),
    ("Can I hold my office hours online?", "faculty", None, ("office", "hours", "online"), None),
])
def test_parse_relation_reads_actor_action_object_and_target(clause, actor, action, obj, target_role):
    rel = G.parse_relation(clause, T.ASK_GATE.lexicon)
    assert rel is not None and rel.actor == actor and rel.action == action and rel.obj == obj and rel.target_role == target_role


def test_parse_relation_reads_passive_and_party_asks_and_skips_the_rest():
    passive = G.parse_relation("Is the workload plan approved by the dean?", T.ASK_GATE.lexicon)
    assert passive.passive and passive.actor == "dean" and passive.action == "approv" and passive.obj == ("workload", "plan")
    party = G.parse_relation("Who reviews course files?", T.ASK_GATE.lexicon)
    assert party.party and party.actor is None and party.action == "review"
    final = G.parse_relation("Does the chair give final approval of the workload plan?", T.ASK_GATE.lexicon)
    assert final.modifiers == ("final",)
    assert G.parse_relation("What percentage of the final grade may the final exam carry?", T.ASK_GATE.lexicon) is None
    assert G.parse_relation("What is the penalty when a student is caught cheating late?", T.ASK_GATE.lexicon) is None   # a condition, not an actor


def test_a_word_that_is_verb_and_noun_is_read_from_its_neighbours():
    assert G._matches_action("review", "review", prev="will", nxt="and") == "verb"
    assert G._matches_action("approval", "approve", prev="final", nxt="rests") == "noun"
    assert G._matches_action("approval", "approve", prev="the", nxt="of") == "noun"
    assert G._matches_action("assessing", "assess", prev="for", nxt="the") == "verb"
    assert G._matches_action("assignments", "assign", prev="teaching", nxt=".") is None


# ---------------------------------------------------------------------------
# The seven confirmed wrong-actor cases, each with its positive control
# ---------------------------------------------------------------------------
def test_1_students_attendance_rule_does_not_bind_faculty(mini):
    sp = mini(228, STUDENTS_ATTEND)
    assert status(sp, "Must faculty attend all theoretical lectures?") is PARTIAL
    assert status(sp, "Must I attend the laboratory sessions?") is PARTIAL
    assert status(sp, "Are instructors required to attend all examinations?") is PARTIAL
    assert status(sp, "Must students attend all theoretical lectures?") is SUPPORTED           # 10. the right actor


def test_2_chairs_assess_course_files_not_the_dean(mini):
    sp = mini(231, CHAIRS_ASSESS)
    assert status(sp, "Does the dean review course files?") is PARTIAL
    assert status(sp, "Does the college council review course files?") is PARTIAL
    assert status(sp, "Do faculty members assess course files?") is PARTIAL                      # faculty prepare them
    assert status(sp, "Does the department chair assess course files?") is SUPPORTED
    assert status(sp, "Who reviews course files?") is SUPPORTED


def test_3_faculty_enter_grades_not_the_registrar(mini):
    sp = mini(224, GRADES_FLOW)
    assert status(sp, "Does the Registrar enter grades into Blackboard?") is PARTIAL
    assert status(sp, "Does the chair enter grades into Blackboard?") is PARTIAL
    assert status(sp, "Do faculty members enter grades into Blackboard?") is SUPPORTED
    assert status(sp, "Must I enter grades into Blackboard?") is SUPPORTED


def test_4_direction_blackboard_to_banner_is_kept(mini):
    sp = mini(224, GRADES_FLOW)
    assert status(sp, "Does Banner forward grades to Blackboard?") is PARTIAL
    assert status(sp, "Does Blackboard forward grades to Banner automatically?") is SUPPORTED     # 12. the right direction
    assert status(sp, "Who forwards grades to the Registration Department?") is SUPPORTED
    assert status(sp, "Who forwards grades to the Registrar?") is PARTIAL                       # a registration system is not the registrar


def test_5_final_approval_rests_with_the_dean_not_the_chair(mini):
    sp = mini(96, APPROVAL_FLOW)
    assert status(sp, "Does the chair give final approval of the workload plan?") is PARTIAL
    assert status(sp, "Does the dean review the workload plan?") is PARTIAL
    assert status(sp, "Does the dean give final approval of the workload plan?") is SUPPORTED    # 11. the right actor
    assert status(sp, "Is the workload plan approved by the dean?") is SUPPORTED
    assert status(sp, "Who reviews the workload plan?") is SUPPORTED
    assert status(sp, "Does the chair receive approval from the dean for the workload plan?") is PARTIAL   # not stated as such


def test_6_the_student_removes_an_incomplete_not_the_instructor(mini):
    sp = mini(225, INCOMPLETE)
    assert status(sp, "Must instructors remove an Incomplete within the first week?") is PARTIAL
    assert status(sp, "Must a student remove an Incomplete within the first week?") is SUPPORTED


def test_7_an_antecedent_lends_no_actor_to_the_next_sentence(mini):
    sp = mini(229, SYLLABUS_PAIR)
    assert status(sp, "Does the department council prepare the syllabus?") is PARTIAL
    assert status(sp, "Do instructors approve the course syllabus?") is PARTIAL
    assert status(sp, "Who prepares the course syllabus?") is SUPPORTED
    assert status(sp, "Who approves the course syllabus?") is SUPPORTED


# ---------------------------------------------------------------------------
# Sense and type collisions
# ---------------------------------------------------------------------------
def test_8_absence_of_errors_is_not_attendance(mini):
    assert "attendance" not in T.concepts(ERRORS).anchors
    assert "attendance" in T.concepts(ABSENCES).anchors
    assert "attendance" in T.concepts("student absences exceed 10%").anchors
    assert "attendance" in T.concepts("the student will receive an F for absence.").anchors
    sp = mini(225, ERRORS)
    assert status(sp, "Must students attend tutorials?") is not SUPPORTED


def test_9_a_calendar_range_is_no_percentage(mini):
    assert not T._percent_ok(CALENDAR)
    assert T._percent_ok(FINAL_SHARE) and T._percent_ok("A 90-100 4.00")
    sp = mini(50, "Spring Semester 2025/2026", "Day Description", CALENDAR)
    assert status(sp, "What percentage of the grade is the final exam?") is not SUPPORTED
    assert status(sp, "What percentage of the final grade may the final exam carry?") is not SUPPORTED


def test_13_percentage_rows_and_sentences_stay_supported(mini, registry_and_map):
    share = TS.run(mini(227, FINAL_SHARE), "What percentage of the final grade may the final exam carry?")
    assert share.status is SUPPORTED and "40-50%" in quotes(share)[0]
    rows = TS.run(TS.make_specialist(TS.corpus_from(registry_and_map, TS.PAGES)), "What percentage corresponds to grade A in the grading system?")
    assert rows.status is SUPPORTED and any("A 90-100" in q for q in quotes(rows))         # a grade-table row keeps its ranges
    absences = TS.run(mini(228, ABSENCES), "What percentage of absences bars a student from the final exam?")
    assert absences.status is SUPPORTED and "20%" in quotes(absences)[0]


# ---------------------------------------------------------------------------
# The second cut: subjects that are no role, passive agents, recipients, pronouns, direction by preposition
# ---------------------------------------------------------------------------
CORRECT_SHEETS = ("Instructors must correct final examination answer sheets or booklets and submit the results, documented in letter grades "
                  "and percentages, to the Department Chair within forty-eight hours of the examination date.")
SYLLABI_DISTRIBUTED = ("These syllabi shall be distributed to students at the beginning of the semester and uploaded on Blackboard and kept in "
                       "the course files within the college.")
ABSENCE_NOTICE = ("6. Communication of Absence: If an instructor is unable to be present during their designated office hours, he/she should "
                  "make reasonable efforts to inform students in advance.")
FULL_TIME = ("1. Full-Time Faculty: Full-time faculty members in academic and career/technical programs are required to schedule a minimum of "
             "five office hours per week, spread over a minimum of two days.")
ANNUAL = ("• Annual Evaluation: At the end of each academic year, the department chair will evaluate the faculty member's performance against "
          "the workload plan, providing feedback and recommendations for the next year.")
EMBRACE = ("3.4 Responsibility to Office Hours Recognizing the significance of being available for student consultation, faculty members at "
           "the University of Sharjah (UoS) embrace the responsibility to hold regular office hours outside of their scheduled instructional time.")


def test_14_a_concept_subject_cannot_swap_with_the_object(mini):
    chairs = mini(231, CHAIRS_ASSESS)
    assert status(chairs, "Do course files assess department chairs?") is PARTIAL
    assert status(chairs, "Do department chairs assess course files?") is SUPPORTED
    approval = mini(96, APPROVAL_FLOW)
    assert status(approval, "Does the workload plan approve the dean?") is PARTIAL
    assert status(approval, "Is the department chair approved by the workload plan?") is PARTIAL
    assert status(mini(229, SYLLABUS_PAIR), "Do syllabi prepare instructors?") is PARTIAL
    rel = G.parse_relation("Does the workload plan approve the dean?", T.ASK_GATE.lexicon)
    assert rel.actor is None and rel.actor_words == ("workload", "plan") and rel.action == "approve" and rel.obj == ("dean",)


def test_15_a_passive_ask_binds_its_agent(mini):
    sp = mini(92, WORKLOAD_REVIEW)
    assert status(sp, "Is the workload reviewed periodically by the department chair?") is PARTIAL
    assert status(sp, "Is the workload reviewed periodically by the Deans' Council?") is SUPPORTED
    assert status(sp, "Is the workload reviewed by the dean?") is PARTIAL                       # the Deans' Council is a council


def test_16_the_recipient_of_a_passive_is_not_its_agent(mini):
    sp = mini(229, SYLLABI_DISTRIBUTED)
    assert status(sp, "Do students distribute the syllabi at the beginning of the semester?") is PARTIAL
    assert status(sp, "Are the syllabi distributed to students at the beginning of the semester?") is SUPPORTED
    assert status(sp, "Are syllabi uploaded on Blackboard?") is SUPPORTED


def test_17_a_filler_in_verb_position_is_the_action(mini):
    sp = mini(224, CORRECT_SHEETS)
    assert status(sp, "Do students correct final examination answer sheets?") is PARTIAL
    assert status(sp, "Must instructors correct final examination answer sheets?") is SUPPORTED
    assert G.parse_relation("Do students correct final examination answer sheets?", T.ASK_GATE.lexicon).action == "correct"
    assert G.parse_relation("Are faculty correct in assuming that grades are final?", T.ASK_GATE.lexicon) is None


def test_18_a_pronoun_subject_takes_its_antecedent(mini):
    notice = mini(79, ABSENCE_NOTICE)
    assert status(notice, "Are students required to hold office hours?") is PARTIAL
    assert status(notice, "Should I inform students in advance if I am unable to be present during my office hours?") is SUPPORTED
    assert status(notice, "Must students inform the instructor in advance about office hours?") is PARTIAL
    assert T.ASK_GATE.relations.normative_subjects(ABSENCE_NOTICE) == ["faculty"]
    embrace = mini(79, EMBRACE)
    assert status(embrace, "Are students required to hold office hours?") is PARTIAL
    assert status(embrace, "Must faculty members hold regular office hours?") is SUPPORTED
    assert T.ASK_GATE.relations.normative_subjects(EMBRACE) == [None, "faculty"]


def test_19_direction_is_read_from_the_preposition(mini):
    sp = mini(224, GRADES_FLOW)
    assert status(sp, "Does Blackboard forward grades to faculty members?") is PARTIAL          # both roles named, wrong destination
    assert status(sp, "Are grades forwarded from Blackboard to Banner?") is SUPPORTED
    assert status(sp, "Are grades forwarded from Banner to Blackboard?") is PARTIAL
    assert status(sp, "Do the grades forward Blackboard to Banner?") is PARTIAL
    assert G.parse_relation("Are grades forwarded from Blackboard to Banner?", T.ASK_GATE.lexicon).target_prep == "to"


def test_21_a_light_verb_after_a_concept_subject_still_binds_the_subject(mini):
    sp = mini(79, FULL_TIME)
    assert status(sp, "Do office hours require five faculty members per week?") is PARTIAL
    assert status(sp, "Are full-time faculty members required to schedule five office hours per week?") is SUPPORTED
    assert status(sp, "How many office hours must full-time faculty schedule per week?") is SUPPORTED


def test_22_the_object_of_evaluate_is_read_before_against(mini):
    sp = mini(96, ANNUAL)
    assert status(sp, "Who reviews the workload allocation each year?") is PARTIAL              # performance is evaluated, not the allocation
    assert status(sp, "Who evaluates performance against the workload plan each year?") is SUPPORTED


def test_20_a_party_ask_needs_a_named_party(mini):
    sp = mini(231, "Course files are assessed at the end of each term.")
    assert status(sp, "Who assesses course files?") is PARTIAL
    assert status(mini(231, CHAIRS_ASSESS), "Who assesses course files?") is SUPPORTED


# ---------------------------------------------------------------------------
# Other positive controls and the earlier rules
# ---------------------------------------------------------------------------
def test_party_asks_read_passive_agents_and_nominal_actions(mini):
    assert status(mini(92, WORKLOAD_REVIEW), "Who reviews the workload allocation?") is SUPPORTED
    assert status(mini(92, WORKLOAD_REVIEW), "Who reviews the workload each year?") is PARTIAL              # periodically is not every year
    assert status(mini(92, "• The workload shall be reviewed every year by the Deans' Council."), "Who reviews the workload each year?") is SUPPORTED


def test_light_verb_asks_bind_the_normative_subject(mini):
    normative = mini(79, "Faculty members may hold their office hours online through the learning management system with the chair's approval.")
    assert status(normative, "Can I hold my office hours online?") is SUPPORTED
    assert status(normative, "Can students hold their office hours online?") is PARTIAL
    impersonal = mini(79, "Office hours may be held online through the learning management system with the chair's approval.")
    assert status(impersonal, "Can I hold my office hours online?") is PARTIAL                        # D-081: no faculty subject, no support


def test_a_condition_does_not_name_an_actor(mini):
    late = mini(228, "Late Detection of Cheating: If cheating is detected at a later stage, the offender will still be held accountable, and the case "
                     "will be referred to the appropriate committee for investigation and determination of a suitable penalty.")
    assert status(late, "What is the penalty when a student is caught cheating late?") is SUPPORTED
