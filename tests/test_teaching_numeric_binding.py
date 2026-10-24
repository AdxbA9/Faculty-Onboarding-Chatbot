"""Teaching numeric binding (Milestone 4, Step 4.8T-C5; D-083).

Step 4.8T-R4 found that a specific numeric ask was checked word by word: the number, its object and its period each
anywhere in the evidence unit. The real workload row "12 credit hours per regular semester or 24 credit hours per
academic year" then supported "24 credit hours per semester", and "a minimum of five office hours per week, spread over
a minimum of two days" supported "two office hours per week". A numeric ask is now one local fact (value, object,
period) stated in one clause, an action's period sits in the clause of the action, and facts of one unit never mix.
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

WORKLOAD_ROW = "o Ranked full time regular faculty, visiting faculty or senior lecturer workload equal to 12 credit hours per regular semester or 24 credit hours per academic year."
OFFICE_HOURS = ("1. Full-Time Faculty: Full-time faculty members in academic and career/technical programs are required to schedule a minimum of "
                "five office hours per week, spread over a minimum of two days.")
TWO_FREQ = "Instructors update the course page weekly and check the course discussion board daily."
TEACH_AND_HOLD = "Faculty teach 12 credit hours and hold 5 office hours per week."
MISS_AND_SUBMIT = "Absences of 20 percent of the lectures result in a warning, and make-up work must be submitted within 5 days."
CHAIR_LECTURER = "The department chair has 6 teaching hours per week and the lecturer has 15 teaching hours per week."
CHAIR_DEAN = "The department chair reviews teaching loads each semester and the dean reviews them each year."
PART_FULL = "Part-time instructors hold at least 2 office hours per week on 1 day, and full-time faculty hold 5 office hours per week on 2 days."
SEMICOLONS = "Examination results are submitted within 48 hours; re-marking requests are filed within 7 days; grade appeals are decided within 14 days."
COMMA_PREDICATES = "Regular faculty members teach 12 credit hours, hold five office hours per week and supervise four theses each year."
ACTIVE_THEN_PASSIVE = "Instructors submit examination results within 48 hours; re-marking requests are filed within 7 days."
WARNING = ("2. If a student's absences exceed 10% of the total designated hours for one or more courses without an acceptable excuse, the "
           "instructor will issue a warning. Absences reaching 20% will prohibit the student from sitting for the final examination(s).")
TWO_COURSES = "A lecture course meets three contact hours per week; a laboratory course meets four contact hours per week."
LOAD_ROWS = ("Faculty category Teaching load per semester", "Regular faculty 12 credit hours per semester", "Lecturer 15 credit hours per semester")
GRADE_ROWS = ("12.14 Grading System:", "Grades Percentages Points", "A 90-100 4.00", "B+ 85-89 3.50")
FINAL_SHARE = "12.13.3.1 Purely or mainly theoretical courses: 1. 40-50% of the final grade shall be allocated to the final examination."
ABSENCES = "3. Absences reaching 20% will prohibit the student from sitting for the final examination(s)."
SUBMIT_48 = ("Instructors must correct final examination answer sheets or booklets and submit the results, documented in letter grades and "
             "percentages, to the Department Chair within forty-eight hours of the examination date.")


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


def facts(text):
    return [(f["values"], f["obj"], frozenset(f["freqs"]), f["segment"]) for f in G.numeric_facts(G.normalise(text))]


# ---------------------------------------------------------------------------
# The local fact reader
# ---------------------------------------------------------------------------
def test_facts_are_read_per_clause():
    assert [f[:3] for f in facts(OFFICE_HOURS)[-2:]] == [((5.0,), ("office", "hours"), frozenset({"week"})), ((2.0,), ("days",), frozenset())]
    row = facts(WORKLOAD_ROW)
    assert [f[:3] for f in row] == [((12.0,), ("credit", "hours"), frozenset({"semester"})), ((24.0,), ("credit", "hours"), frozenset({"year"}))]
    assert row[0][3] != row[1][3]                                                              # two clauses
    assert facts(CHAIR_LECTURER) == [((6.0,), ("teaching", "hours"), frozenset({"week"}), 0), ((15.0,), ("teaching", "hours"), frozenset({"week"}), 1)]
    assert [f["values"] for f in G._ask_numeric_facts("Does the department chair have 15 teaching hours per week?")] == [(15.0,)]
    asked = G._ask_numeric_facts("Must full-time faculty schedule two office hours per week?")
    assert [(f["values"], f["obj"], frozenset(f["freqs"])) for f in asked] == [((2.0,), ("office", "hours"), frozenset({"week"}))]
    assert G._ask_numeric_facts("When does the spring semester 2025/2026 begin?") == []           # a year span is no fact
    assert G._ask_numeric_facts("What is the teaching load for course 0104 210?") == []            # a bare code is no fact


# ---------------------------------------------------------------------------
# 1. The real workload row: four pairings, two of them wrong
# ---------------------------------------------------------------------------
def test_1_workload_row_pairs_do_not_cross(mini):
    row = mini(91, WORKLOAD_ROW)
    assert status(row, "Is the workload of regular faculty 24 credit hours per semester?") is PARTIAL
    assert status(row, "Is the workload of regular faculty 12 credit hours per academic year?") is PARTIAL
    assert status(row, "Is the workload of regular faculty 12 credit hours per regular semester?") is SUPPORTED
    assert status(row, "Is the workload of regular faculty 24 credit hours per academic year?") is SUPPORTED


# ---------------------------------------------------------------------------
# 2. The real office-hours rule: five office hours, two days
# ---------------------------------------------------------------------------
def test_2_office_hours_value_is_not_the_days_value(mini):
    rule = mini(79, OFFICE_HOURS)
    assert status(rule, "Must full-time faculty schedule two office hours per week?") is PARTIAL
    assert status(rule, "Must full-time faculty schedule five office hours per week?") is SUPPORTED
    assert status(rule, "Must full-time faculty schedule five office hours weekly?") is SUPPORTED
    assert status(rule, "Must office hours be spread over a minimum of five days?") is PARTIAL


# ---------------------------------------------------------------------------
# 3. Two actions, two periods
# ---------------------------------------------------------------------------
def test_3_action_periods_do_not_cross(mini):
    sp = mini(79, TWO_FREQ)
    assert status(sp, "Do instructors update the course page daily?") is PARTIAL
    assert status(sp, "Do instructors check the course discussion board weekly?") is PARTIAL
    assert status(sp, "Do instructors update the course page weekly?") is SUPPORTED
    assert status(sp, "Do instructors check the course discussion board daily?") is SUPPORTED


# ---------------------------------------------------------------------------
# 4. Other pairs in one unit
# ---------------------------------------------------------------------------
def test_4_values_stay_with_their_objects(mini):
    both = mini(96, TEACH_AND_HOLD)
    assert status(both, "Do faculty hold 12 office hours per week?") is PARTIAL
    assert status(both, "Do faculty teach 5 credit hours?") is PARTIAL
    assert status(both, "Do faculty teach 12 credit hours?") is SUPPORTED
    assert status(both, "Do faculty hold 5 office hours per week?") is SUPPORTED
    miss = mini(228, MISS_AND_SUBMIT)
    assert status(miss, "Do absences of 5 percent of the lectures result in a warning?") is PARTIAL
    assert status(miss, "Do absences of 20 percent of the lectures result in a warning?") is SUPPORTED


def test_4_an_actor_keeps_its_own_value(mini):
    sp = mini(96, CHAIR_LECTURER)
    assert status(sp, "Does the department chair have 15 teaching hours per week?") is PARTIAL
    assert status(sp, "Does the lecturer have 6 teaching hours per week?") is PARTIAL
    assert status(sp, "Does the department chair have 6 teaching hours per week?") is SUPPORTED
    assert status(sp, "Does the lecturer have 15 teaching hours per week?") is SUPPORTED


def test_4_an_actor_keeps_its_own_period(mini):
    sp = mini(92, CHAIR_DEAN)
    assert status(sp, "Does the department chair review teaching loads each year?") is PARTIAL
    assert status(sp, "Does the dean review teaching loads each semester?") is PARTIAL
    assert status(sp, "Does the department chair review teaching loads each semester?") is SUPPORTED
    assert status(sp, "Does the dean review teaching loads each year?") is SUPPORTED


def test_4_a_category_keeps_its_own_value(mini):
    sp = mini(79, PART_FULL)
    assert status(sp, "Do part-time instructors hold 5 office hours per week?") is PARTIAL
    assert status(sp, "Do full-time faculty hold 2 office hours per week?") is PARTIAL
    assert status(sp, "Do part-time instructors hold 2 office hours per week?") is SUPPORTED
    assert status(sp, "Do full-time faculty hold 5 office hours per week?") is SUPPORTED


def test_4_a_value_stays_in_the_clause_of_its_action(mini):
    sp = mini(224, SEMICOLONS)
    assert status(sp, "Are re-marking requests filed within 48 hours?") is PARTIAL
    assert status(sp, "Are examination results submitted within 7 days?") is PARTIAL
    assert status(sp, "Are examination results submitted within 48 hours?") is SUPPORTED
    assert status(sp, "Are grade appeals decided within 14 days?") is SUPPORTED


def test_4_a_role_heading_a_concept_phrase_is_the_actor():
    rel = G.parse_relation("Must the course coordinator upload the syllabus on Blackboard?", T.ASK_GATE.lexicon)
    assert rel is not None and rel.actor == "coordinator" and rel.action == "upload"
    rel = G.parse_relation("Must evening instructors schedule at least one office hour during the evening?", T.ASK_GATE.lexicon)
    assert rel is not None and rel.actor == "faculty" and rel.action == "schedule"
    assert G.parse_relation("Is the regular faculty workload 12 credit hours per regular semester?", T.ASK_GATE.lexicon) is None
    assert G.parse_relation("Is the baseline teaching load for regular faculty 12 credit hours per semester?", T.ASK_GATE.lexicon) is None


def test_4_a_coordinated_predicate_keeps_its_period(mini):
    sp = mini(224, SUBMIT_48)
    assert status(sp, "Must I submit the results within forty-eight hours of the examination date?") is SUPPORTED
    assert status(sp, "Must I correct final examination answer sheets within forty-eight hours?") is SUPPORTED
    assert status(sp, "Must I submit the results within forty-eight days of the examination date?") is PARTIAL


def test_4_a_period_does_not_cross_a_comma(mini):
    sp = mini(96, COMMA_PREDICATES)
    assert status(sp, "Do regular faculty teach 12 credit hours per week?") is PARTIAL
    assert status(sp, "Do regular faculty hold 5 office hours each year?") is PARTIAL
    assert status(sp, "Do regular faculty teach 12 credit hours?") is SUPPORTED
    assert status(sp, "Do regular faculty hold 5 office hours per week?") is SUPPORTED


def test_4_a_value_of_another_independent_clause_never_counts(mini):
    sp = mini(224, ACTIVE_THEN_PASSIVE)
    assert status(sp, "Are examination results submitted within 7 days?") is PARTIAL                  # the only "submit" names a subject
    assert status(sp, "Are examination results submitted within 48 hours?") is SUPPORTED


def test_4_a_value_of_a_subordinate_clause_keeps_counting(mini):
    sp = mini(228, WARNING)
    assert status(sp, "Is a warning issued when absences exceed 10 percent?") is SUPPORTED
    assert status(sp, "Is a warning issued when absences exceed 20 percent?") is PARTIAL


def test_4_a_generic_subject_keeps_its_own_value(mini):
    sp = mini(93, TWO_COURSES)
    assert status(sp, "Does a laboratory course meet 3 contact hours per week?") is PARTIAL
    assert status(sp, "Does a laboratory course meet 4 contact hours per week?") is SUPPORTED


def test_4_an_asked_period_applies_to_the_asked_number(mini):
    row = mini(91, WORKLOAD_ROW)
    assert status(row, "Is the annual workload of regular faculty 12 credit hours?") is PARTIAL
    assert status(row, "Is the semester workload of senior lecturers 24 credit hours?") is PARTIAL
    assert status(row, "Is the annual workload of regular faculty 24 credit hours?") is SUPPORTED


# ---------------------------------------------------------------------------
# 5. Table rows keep their values, only the right pairing
# ---------------------------------------------------------------------------
def test_5_table_rows_answer_their_own_values(mini, registry_and_map):
    pages = TS.make_specialist(TS.corpus_from(registry_and_map, TS.PAGES))
    assert TS.run(pages, "Is the teaching load for lecturers 15 credit hours per semester?").status is SUPPORTED
    assert TS.run(pages, "Is the teaching load for lecturers 12 credit hours per semester?").status is PARTIAL
    assert TS.run(pages, "Is the teaching load for lecturers 15 credit hours per year?").status is PARTIAL
    grade = TS.run(pages, "What percentage corresponds to grade A in the grading system?")
    assert grade.status is SUPPORTED and any("A 90-100" in f.evidence_quote for f in grade.findings)
    assert TS.run(pages, "What is the teaching load for lecturers?").status is SUPPORTED


# ---------------------------------------------------------------------------
# 6. Positive numeric facts still answer
# ---------------------------------------------------------------------------
def test_6_correct_facts_stay_supported(mini):
    assert status(mini(227, FINAL_SHARE), "Is 40-50 percent of the final grade allocated to the final examination?") is SUPPORTED
    assert status(mini(227, FINAL_SHARE), "Is 50-60 percent of the final grade allocated to the final examination?") is PARTIAL
    assert status(mini(228, ABSENCES), "Do absences reaching 20 percent prohibit the student from sitting for the final examination?") is SUPPORTED
    assert status(mini(228, ABSENCES), "Do absences reaching 10 percent prohibit the student from sitting for the final examination?") is PARTIAL
    assert status(mini(224, SUBMIT_48), "Must I correct final examination answer sheets within forty-eight hours?") is SUPPORTED
    assert status(mini(224, SUBMIT_48), "Must I correct final examination answer sheets within forty-eight days?") is PARTIAL


# ---------------------------------------------------------------------------
# 7. A hyphenated period is the spaced one (Step 4.8T-C6, D-084)
# ---------------------------------------------------------------------------
UNITS = ("day", "week", "month", "semester", "term", "year")


def test_7_hyphenated_periods_equal_spaced_periods():
    for lead in ("per", "each", "every"):
        for unit in UNITS:
            spaced, hyphenated = "%s %s" % (lead, unit), "%s-%s" % (lead, unit)
            assert G.frequencies_of(hyphenated) == G.frequencies_of(spaced) == frozenset({unit}), hyphenated
    assert G.frequencies_of("per-regular-semester") == G.frequencies_of("per regular semester") == frozenset({"semester"})
    assert G.frequencies_of("each-academic-year") == G.frequencies_of("each academic year") == frozenset({"year"})
    assert G.frequencies_of("annually") == G.frequencies_of("per-year") == G.frequencies_of("each year") == frozenset({"year"})
    assert G.frequencies_of("weekly") == G.frequencies_of("per-week") == frozenset({"week"})
    assert G.frequencies_of("monthly") == G.frequencies_of("per-month") == frozenset({"month"})
    assert G.frequencies_of("daily") == G.frequencies_of("every-day") == frozenset({"day"})


def test_7_hyphen_normalisation_keeps_units_distinct():
    sets = {unit: G.frequencies_of("per-%s" % unit) for unit in UNITS}
    for a in UNITS:
        for b in UNITS:
            assert (sets[a] == sets[b]) == (a == b), (a, b)
    assert G.frequencies_of("periodically") == frozenset({"periodic"})
    assert G.frequencies_of("periodically") != G.frequencies_of("per-semester")
    assert G.frequencies_of("periodically") != G.frequencies_of("annually")
    assert G.frequencies_of("per-semester") != G.frequencies_of("per-term")


def test_7_a_hyphenated_period_binds_like_a_spaced_one(mini):
    row = mini(91, WORKLOAD_ROW)
    assert status(row, "Is the per-semester workload of visiting faculty 24 credit hours?") is PARTIAL
    assert status(row, "Is the per-year workload of regular faculty 12 credit hours?") is PARTIAL
    assert status(row, "Is the per-semester workload of visiting faculty 12 credit hours?") is SUPPORTED
    assert status(row, "Is the per semester workload of visiting faculty 12 credit hours?") is SUPPORTED
    assert status(row, "Is the per-year workload of regular faculty 24 credit hours?") is SUPPORTED
    assert status(row, "Is the per-academic-year workload of regular faculty 12 credit hours?") is PARTIAL
    hours = mini(79, OFFICE_HOURS)
    assert status(hours, "Must full-time faculty schedule two office hours per-week?") is PARTIAL
    assert status(hours, "Must full-time faculty schedule five office hours per-week?") is SUPPORTED


def test_7_an_actor_keeps_its_own_hyphenated_period(mini):
    sp = mini(92, CHAIR_DEAN)
    assert status(sp, "Does the department chair review teaching loads per-year?") is PARTIAL
    assert status(sp, "Does the dean review teaching loads per-semester?") is PARTIAL
    assert status(sp, "Does the chair review teaching loads per-semester?") is SUPPORTED
    assert status(sp, "Does the dean review teaching loads per-year?") is SUPPORTED
    assert status(sp, "Does the dean review teaching loads each-year?") is SUPPORTED
    assert status(sp, "Does the dean review teaching loads every-semester?") is PARTIAL
