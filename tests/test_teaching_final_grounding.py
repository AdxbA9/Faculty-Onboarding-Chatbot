"""Teaching final grounding (Milestone 4, Step 4.8T-C4; D-082).

Step 4.8T-R3 found three residual classes after D-081: a subject outside the role lexicon ("the quality office", "the
academic advisor") received no relation and fell to the word bag (a calendar row then answered an actor ask); an
agentless passive ask with an irregular participle ("sent", "kept", "held", "made") received no relation; and the
agent rule looked for the actor anywhere in the unit, so "Faculty enter grades. They are transferred to Banner." made
faculty the transfer agent. The fixture lines sit on owned, unshared Teaching pages. No model, no LLM, no network.
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

CHAIRS_ASSESS = ("Department Chairs are responsible for assessing the contents and quality of course files prepared by faculty members in "
                 "their departments at the end of each term.")
QUALITY_OFFICE = "The Quality Office assesses the contents and quality of course files at the end of each term."
INCOMPLETE_ROW = ("Summer Semester 2025/2026", "Day Description", "Mon-Tue 27-28 July 13-14 Safar Last day to remove Incomplete grade")
ADVISOR_REMOVES = "The academic advisor removes the Incomplete mark once the special examination has been passed."
GRADE_ROWS = ("12.14 Grading System:", "Grades Percentages Points", "A 90-100 4.00", "F Below 60 0.00")
LOAD_ROWS = ("Faculty category Teaching load per semester", "Lecturer 15 credit hours per semester")
SENT_REGISTRAR = "Final grades are sent to the Registrar and approved by the dean before release."
SENT_DEAN = "Final grades are sent to the dean before release."
KEPT_PAIR = "The marked final examination answer sheets are kept by the department for one semester, and the course files are sent to the dean."
BANNER_FLOW = "Faculty members enter the grades into Blackboard, which forwards them automatically to Banner."
KEPT = "The marked final examination answer sheets are kept by the department for one semester."
HELD = "Office hours are held by faculty in their offices."
MADE = "The final decision on a grade appeal is made by the college council."
TRANSFER_PAIR = "Faculty enter grades in Blackboard. They are transferred to Banner overnight."
COORDINATED = "Faculty enter and transfer grades to Banner at the end of the semester."
ACTOR_PRONOUN = "Faculty prepare course files. They submit them to the chair at the end of the term."
OBJECT_PRONOUN = "Faculty prepare course files. They are submitted to the chair at the end of the term."
FORWARDS = "Faculty members enter grades into Blackboard, which forwards them to Banner."


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
# 1. A subject outside the role lexicon is bound, never dropped
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,words,action", [
    ("Does the quality office assess course files?", ("quality", "office"), "assess"),
    ("Does the academic advisor remove the Incomplete mark?", ("academic", "advisor"), "remove"),
    ("Does the assessment team review the course files?", ("team",), "review"),
    ("Must the course team upload the syllabus on Blackboard?", ("team",), "upload"),
])
def test_1_a_generic_subject_is_kept(clause, words, action):
    rel = relation(clause)
    assert rel is not None and rel.actor is None and rel.actor_words[-1:] == words[-1:] and rel.action == action


def test_1_the_quality_office_does_not_inherit_the_chairs_duty(mini):
    chairs = mini(231, CHAIRS_ASSESS)
    assert status(chairs, "Does the quality office assess course files?") is PARTIAL
    assert status(chairs, "Does the department chair assess course files?") is SUPPORTED
    office = mini(231, QUALITY_OFFICE)
    assert status(office, "Does the Quality Office assess course files?") is SUPPORTED                # the same subject: bound, not rejected
    assert status(office, "Does the department chair assess course files?") is PARTIAL
    assert status(office, "Do faculty members assess course files?") is PARTIAL


def test_1_a_generic_subject_with_no_action_is_never_the_word_bag(mini):
    assert status(mini(231, CHAIRS_ASSESS), "Does the quality office have course files?") is not SUPPORTED


# ---------------------------------------------------------------------------
# 1b. A table row settles no ask that names an actor
# ---------------------------------------------------------------------------
def test_1_a_calendar_row_answers_no_actor_ask(mini):
    row = mini(50, *INCOMPLETE_ROW)
    assert status(row, "Does the academic advisor remove the Incomplete mark?") is PARTIAL
    assert status(row, "Must students remove the Incomplete grade?") is not SUPPORTED
    assert status(row, "When is the last day to remove an Incomplete grade?") is SUPPORTED          # a date: still a table answer
    assert status(mini(225, ADVISOR_REMOVES), "Does the academic advisor remove the Incomplete mark?") is SUPPORTED


def test_1_table_values_still_answer_value_asks(mini, registry_and_map):
    rows = TS.run(TS.make_specialist(TS.corpus_from(registry_and_map, TS.PAGES)), "What percentage corresponds to grade A in the grading system?")
    assert rows.status is SUPPORTED and any("A 90-100" in f.evidence_quote for f in rows.findings)
    assert status(mini(227, *GRADE_ROWS), "Do instructors award an A for 90 percent?") is not SUPPORTED
    pages = TS.make_specialist(TS.corpus_from(registry_and_map, TS.PAGES))
    assert TS.run(pages, "What is the teaching load for lecturers?").status is SUPPORTED
    assert status(mini(96, *LOAD_ROWS), "Do lecturers teach 15 credit hours per semester?") is not SUPPORTED   # an actor ask needs prose


# ---------------------------------------------------------------------------
# 2. Irregular participles in passive asks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,action,target_role", [
    ("Are final grades sent to the dean?", "send", "dean"), ("Are the examination answer sheets kept by the department?", "keep", None),
    ("Are office hours held by faculty?", "hold", None), ("Is the final decision made by the college council?", "make", None),
    ("Must the syllabus be given to students?", "give", "student"), ("Is attendance taken in every lecture?", "take", None),
])
def test_2_irregular_participles_are_read(clause, action, target_role):
    rel = relation(clause)
    assert rel is not None and rel.passive and rel.action == action
    if target_role:
        assert rel.target_role == target_role


def test_2_sent_to_the_registrar_is_not_sent_to_the_dean(mini):
    registrar = mini(224, SENT_REGISTRAR)
    assert status(registrar, "Are final grades sent to the dean?") is PARTIAL
    assert status(registrar, "Are final grades sent to the Registrar?") is SUPPORTED
    assert status(registrar, "Are final grades approved by the dean?") is SUPPORTED
    assert status(mini(224, SENT_DEAN), "Are final grades sent to the dean?") is SUPPORTED


def test_2_an_agentless_irregular_ask_binds_its_patient(mini):
    sp = mini(224, KEPT_PAIR)
    assert status(sp, "Are the course files kept for one semester?") is PARTIAL                 # the sheets are kept, the course files sent
    assert status(sp, "Are the final examination answer sheets sent to the dean?") is PARTIAL
    assert status(sp, "Are the marked final examination answer sheets kept for one semester?") is SUPPORTED
    assert status(sp, "Are the course files sent to the dean?") is SUPPORTED


def test_2_an_agent_after_the_destination_is_the_actor(mini):
    flow = mini(224, BANNER_FLOW)
    assert status(flow, "Are grades sent to Blackboard by Banner?") is PARTIAL
    assert status(flow, "Are grades entered into Blackboard by faculty members?") is SUPPORTED
    rel = relation("Are grades sent to Blackboard by Banner?")
    assert rel.actor == "banner" and rel.target_role == "blackboard"


def test_2_kept_held_and_made_bind_their_agents(mini):
    kept = mini(224, KEPT)
    assert status(kept, "Are the marked final examination answer sheets kept by faculty?") is PARTIAL
    assert status(kept, "Are the marked final examination answer sheets kept by the department?") is SUPPORTED
    held = mini(79, HELD)
    assert status(held, "Are office hours held by students?") is PARTIAL
    assert status(held, "Are office hours held by faculty?") is SUPPORTED
    made = mini(225, MADE)
    assert status(made, "Is the final decision on a grade appeal made by the dean?") is PARTIAL
    assert status(made, "Is the final decision on a grade appeal made by the college council?") is SUPPORTED


# ---------------------------------------------------------------------------
# 3. Actor evidence is local to the sentence of the action
# ---------------------------------------------------------------------------
def test_3_an_actor_of_another_sentence_licenses_no_action(mini):
    pair = mini(224, TRANSFER_PAIR)
    assert status(pair, "Do faculty transfer grades to Banner?") is PARTIAL
    assert status(pair, "Are grades transferred to Banner?") is SUPPORTED
    assert status(pair, "Do faculty enter grades in Blackboard?") is SUPPORTED
    unit = G._Unit(TRANSFER_PAIR, T.ASK_GATE.lexicon)
    second = next(i for i, t in enumerate(unit.tokens) if t[1] == "transferred")
    assert "faculty" not in unit.agent_roles(*unit.sentence_range(second)) and "faculty" in unit.agent_roles()


def test_3_coordinated_verbs_and_actor_pronouns_still_bind(mini):
    assert status(mini(224, COORDINATED), "Do faculty transfer grades to Banner?") is SUPPORTED
    assert status(mini(231, ACTOR_PRONOUN), "Do faculty submit course files to the chair?") is SUPPORTED
    assert status(mini(231, OBJECT_PRONOUN), "Do faculty submit course files to the chair?") is PARTIAL
    assert status(mini(231, OBJECT_PRONOUN), "Are course files submitted to the chair?") is SUPPORTED
    forwards = mini(224, FORWARDS)
    assert status(forwards, "Do faculty members forward grades to Banner?") is PARTIAL
    assert status(forwards, "Does Blackboard forward grades to Banner?") is SUPPORTED


def test_3_an_object_pronoun_is_no_agent():
    unit = G._Unit(OBJECT_PRONOUN, T.ASK_GATE.lexicon)
    submitted = next(i for i, t in enumerate(unit.tokens) if t[1] == "submitted")
    assert "faculty" not in unit.agent_roles(*unit.sentence_range(submitted))
    actor = G._Unit(ACTOR_PRONOUN, T.ASK_GATE.lexicon)
    submit = next(i for i, t in enumerate(actor.tokens) if t[1] == "submit")
    assert "faculty" in actor.agent_roles(*actor.sentence_range(submit))
    assert "student" not in G._Unit("Students are informed by the instructor of any change.", T.ASK_GATE.lexicon).agent_roles()
    assert "student" in G._Unit("Students are required to attend all lectures.", T.ASK_GATE.lexicon).agent_roles()


# ---------------------------------------------------------------------------
# The earlier classes still hold
# ---------------------------------------------------------------------------
def test_earlier_classes_still_hold(mini):
    assert relation("Does the research office enter grades in Blackboard?").actor_words == ("research", "office")
    assert relation("Does the final examination allocate 40-50 percent of the course?").action == "allocate"
    assert relation("Does the department chair evaluate faculty performance each academic year?").obj == ("faculty", "performance")
    assert status(mini(231, CHAIRS_ASSESS), "Are course files assessed by faculty members of the department?") is PARTIAL
    assert status(mini(231, CHAIRS_ASSESS), "Does the dean review course files?") is PARTIAL
    assert status(mini(231, CHAIRS_ASSESS), "Who assesses course files?") is SUPPORTED
