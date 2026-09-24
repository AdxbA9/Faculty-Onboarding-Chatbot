"""Plan F Coordinator: deterministic domain, level, system and complexity
classification, specialist selection and task decomposition.

Offline by construction: the Coordinator never calls an LLM, and a test
makes the Router's arbitration raise if it were ever reached."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from handbook_bot import config
from handbook_bot.agents import coordinator, router
from handbook_bot.agents.coordinator import METADATA_KEYS, analyze, coordinate
from handbook_bot.agents.specialists.contracts import (
    ALL_SPECIALIST_IDS,
    CoordinatorDecision,
    SpecialistId,
    SpecialistTask,
)

REPO = Path(__file__).resolve().parents[1]

FUNDED = "Can I reduce my teaching load if I have a funded research project, and who approves it?"
FOUR_DOMAINS = ("I have a research grant, I need annual leave, my teaching load must change, "
                "and who approves all of this?")
TWO_PARTS = "When do classes begin? And how many programs are offered?"


def selected(decision: CoordinatorDecision):
    return [s.value for s in decision.selected_specialists]


def complexity(decision: CoordinatorDecision) -> str:
    return decision.metadata["complexity"]


@pytest.fixture(autouse=True)
def router_llm_must_not_be_called(monkeypatch):
    """Y: the Coordinator must never reach the Router's LLM arbitration."""
    def _boom(*args, **kwargs):
        raise AssertionError("Router LLM arbitration was invoked by the Coordinator")
    monkeypatch.setattr(router, "_ask_llm", _boom)


# ---------------------------------------------------------------------------
# A-H: single-domain selection
# ---------------------------------------------------------------------------
def test_simple_teaching_question():
    d = coordinate("What are my teaching responsibilities?")
    assert selected(d) == ["teaching"] and complexity(d) == "simple"
    assert len(d.subtasks) == 1 and d.requires_synthesis is False


def test_blackboard_question_is_teaching_with_blackboard_system():
    d = coordinate("How do I upload my first lecture to Blackboard?")
    assert selected(d) == ["teaching"]
    assert d.systems == ["blackboard"] and d.subtasks[0].system == "blackboard"


def test_banner_teaching_question_is_teaching_with_banner_system():
    d = coordinate("How do I enter final grades in Banner?")
    assert selected(d) == ["teaching"]
    assert d.systems == ["banner"] and d.subtasks[0].system == "banner"


def test_research_funding_question():
    d = coordinate("How do I apply for an internal research grant?")
    assert selected(d) == ["research"] and complexity(d) == "simple"


def test_research_ethics_question():
    d = coordinate("Do I need ethics approval for a survey of students?")
    assert selected(d) == ["research"]
    assert d.intents[0] == "policy_yesno"


@pytest.mark.parametrize("question", [
    "How many days of annual leave do I get?",
    "How do I apply for sabbatical leave?",
])
def test_hr_leave_question(question):
    assert selected(coordinate(question)) == ["faculty_services"]


@pytest.mark.parametrize("question", [
    "Does my health insurance cover my family?",
    "What allowances am I entitled to?",
])
def test_benefits_question(question):
    assert selected(coordinate(question)) == ["faculty_services"]


@pytest.mark.parametrize("question", [
    "Which office handles parking permits?",
    "Where do I go to get my ID card?",
    "Who sits on the University Council?",
])
def test_institutional_navigation_question(question):
    assert selected(coordinate(question)) == ["institutional"]


# ---------------------------------------------------------------------------
# I-K: organizational level
# ---------------------------------------------------------------------------
def test_department_level_detection():
    d = coordinate("What is my department's policy on office hours?")
    assert d.level == "department" and selected(d) == ["teaching"]
    assert d.subtasks[0].level == "department"


def test_college_level_detection():
    assert coordinate("Does the college dean approve sabbatical leave?").level == "college"


def test_university_level_detection():
    assert coordinate("What is the university's mission?").level == "university"


def test_most_specific_level_wins_and_none_when_absent():
    assert coordinate("My department in the College of Engineering asks about exams.").level == "department"
    assert coordinate("How do I mark attendance in Blackboard?").level is None


# ---------------------------------------------------------------------------
# L-N: decomposition
# ---------------------------------------------------------------------------
def test_simple_question_produces_one_unscoped_task():
    question = "How do I mark attendance in Blackboard?"
    d = coordinate(question)
    task = d.subtasks[0]
    assert len(d.subtasks) == 1
    assert task.question == question and task.metadata["scoped"] is False
    assert task.context["full_question"] == question and task.context["focus_clauses"] == [question]
    assert task.specialist_id is SpecialistId.TEACHING and task.domain == "teaching"
    assert task.intent == d.intents[0]


def test_multi_domain_question_keeps_the_full_question_and_scopes_by_focus():
    d = coordinate(TWO_PARTS)
    assert selected(d) == ["teaching", "institutional"] and complexity(d) == "cross_domain"
    assert [t.question for t in d.subtasks] == [TWO_PARTS, TWO_PARTS]
    assert [t.context["focus_clauses"] for t in d.subtasks] == [["When do classes begin"],
                                                                ["how many programs are offered"]]
    assert [t.intent for t in d.subtasks] == ["date", "count"]          # derived from the focus text
    assert all(t.metadata["scoped"] and t.context["full_question"] == TWO_PARTS for t in d.subtasks)


def test_funded_research_teaching_load_approver_case():
    d = coordinate(FUNDED)
    assert selected(d) == ["research", "teaching", "institutional"]
    assert complexity(d) == "cross_domain" and d.requires_synthesis is True
    by_specialist = {t.specialist_id.value: t for t in d.subtasks}
    assert set(by_specialist) == {"research", "teaching", "institutional"}
    assert all(t.question == FUNDED for t in d.subtasks)               # every task stands alone
    focus = {k: " ".join(t.context["focus_clauses"]) for k, t in by_specialist.items()}
    assert "approves" in focus["institutional"]
    assert "funded research project" in focus["research"]
    assert "teaching load" in focus["teaching"]
    assert d.confidence == 1.0


@pytest.mark.parametrize("question", [FUNDED, TWO_PARTS, FOUR_DOMAINS, "How do I enter final grades in Banner?"])
def test_every_task_question_is_the_complete_original_question(question):
    d = coordinate(question)
    assert d.subtasks
    for task in d.subtasks:
        assert task.question == question
        assert task.context["full_question"] == question
        focus = task.context["focus_clauses"]
        assert isinstance(focus, list) and focus and all(isinstance(c, str) and c for c in focus)
        assert all(clause in question for clause in focus)              # focus is never rewritten text


# ---------------------------------------------------------------------------
# O-R: budgets and task hygiene
# ---------------------------------------------------------------------------
def test_max_specialists_is_enforced_from_config():
    d = coordinate(FOUR_DOMAINS)
    assert len(d.selected_specialists) == config.PLAN_F_MAX_SPECIALISTS == 3
    assert d.metadata["dropped_domains"] == ["institutional"]
    assert d.metadata["limits"] == {"specialists": 3, "subtasks": 3}
    # The dropped domain's clause is reported, not glued onto another task.
    assert d.metadata["uncovered_clauses"] == ["who approves all of this"]


def test_max_specialists_override_drops_lowest_ranked_deterministically():
    d = coordinate(FOUR_DOMAINS, max_specialists=2)
    assert selected(d) == ["research", "teaching"]
    assert d.metadata["dropped_domains"] == ["faculty_services", "institutional"]
    assert len(d.subtasks) == 2


def test_max_subtasks_is_enforced():
    assert len(coordinate(FOUR_DOMAINS).subtasks) <= config.PLAN_F_MAX_SUBTASKS
    d = coordinate(FUNDED, max_subtasks=1)
    assert len(d.subtasks) == 1 and len(d.selected_specialists) == 1      # one task per specialist


def test_task_ids_are_unique_and_stable():
    d = coordinate(FUNDED)
    ids = [t.task_id for t in d.subtasks]
    assert ids == ["task-1", "task-2", "task-3"] and len(set(ids)) == 3


@pytest.mark.parametrize("question", [FUNDED, TWO_PARTS, "How do I enter final grades in Banner?"])
def test_every_task_is_requested_by_the_coordinator(question):
    for task in coordinate(question).subtasks:
        assert isinstance(task, SpecialistTask)
        assert task.requested_by == coordinator.REQUESTED_BY == "coordinator"
        assert task.specialist_id is not None and task.domain == task.specialist_id.value


# ---------------------------------------------------------------------------
# S-T: synthesis rule
# ---------------------------------------------------------------------------
def test_single_specialist_simple_question_needs_no_synthesis():
    assert coordinate("How do I create an assignment in Blackboard?").requires_synthesis is False


def test_multi_specialist_question_requires_synthesis():
    assert coordinate(FUNDED).requires_synthesis is True
    assert coordinate(TWO_PARTS).requires_synthesis is True


# ---------------------------------------------------------------------------
# U-V: systems and orientation are not specialists
# ---------------------------------------------------------------------------
def test_blackboard_and_banner_are_systems_not_specialists():
    for name in ("blackboard", "banner", "myuos"):
        assert name not in ALL_SPECIALIST_IDS
        with pytest.raises(ValueError):
            SpecialistId.parse(name)
    d = coordinate("How do I record grades in Blackboard and submit final grades in Banner?")
    assert selected(d) == ["teaching"] and d.systems == ["blackboard", "banner"]


def test_bare_blackboard_or_banner_question_goes_to_teaching():
    banner = coordinate("How do I use Banner?")           # Banner alone is only a hint: owner rule applies
    assert selected(banner) == ["teaching"] and banner.confidence == 0.75
    assert "guides are owned by teaching" in banner.reason
    blackboard = coordinate("How do I log in to Blackboard?")   # Blackboard is a strong teaching cue
    assert selected(blackboard) == ["teaching"] and blackboard.systems == ["blackboard"]


def test_bare_myuos_question_selects_no_specialist():
    for question in ("How do I use MyUOS?", "How do I reset my MyUOS password?"):
        d = coordinate(question)
        assert selected(d) == [] and d.subtasks == []
        assert d.systems == ["myuos"] and complexity(d) == "unknown" and d.confidence == 0.0
        assert "myuos" in d.reason and "fall back" in d.reason


def test_myuos_with_hr_context_is_faculty_services():
    for question in ("How do I check my payslip in MyUOS?", "Where is my employee information in MyUOS?"):
        d = coordinate(question)
        assert selected(d) == ["faculty_services"] and d.systems == ["myuos"]
        assert d.subtasks[0].system == "myuos"


def test_myuos_with_teaching_context_is_teaching():
    for question in ("Where can I see my teaching schedule in MyUOS?",
                     "How do I access my course information in MyUOS?"):
        d = coordinate(question)
        assert selected(d) == ["teaching"] and d.systems == ["myuos"]


def test_orientation_is_a_journey_mode_not_a_specialist():
    assert "orientation" not in ALL_SPECIALIST_IDS
    with pytest.raises(ValueError):
        SpecialistId.parse("orientation")
    for question in ("What should I do during my first week?",
                     "What systems should a new faculty member set up first?"):
        d = coordinate(question)
        assert complexity(d) == "journey" and d.metadata["journey"] is True
        assert selected(d) == [] and d.subtasks == [] and d.requires_synthesis is False
        assert "journey mode is not implemented" in d.reason


def test_get_started_is_a_journey_cue():
    assert complexity(coordinate("How do I get started?")) == "journey"


def test_single_specialist_journey_question_does_not_force_synthesis():
    d = coordinate("As a new faculty member, how do I upload my syllabus to Blackboard?")
    assert selected(d) == ["teaching"] and complexity(d) == "journey"
    assert d.metadata["journey"] is True and d.requires_synthesis is False
    assert len(d.subtasks) == 1


def test_multi_specialist_journey_question_requires_synthesis():
    d = coordinate("As a new faculty member, who approves my research grant and how do I upload my syllabus to Blackboard?")
    assert complexity(d) == "journey" and len(d.selected_specialists) >= 2
    assert d.requires_synthesis is True


# ---------------------------------------------------------------------------
# W: safe behaviour on unknown input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", ["purple monkey dishwasher", "asdfgh qwerty", "???", "   "])
def test_unknown_or_nonsense_selects_nobody(question):
    d = coordinate(question)
    assert isinstance(d, CoordinatorDecision)
    assert selected(d) == [] and d.subtasks == [] and d.confidence == 0.0
    assert complexity(d) == "unknown" and d.requires_synthesis is False
    json.dumps(d.to_dict())


def test_policy_question_without_domain_cue_falls_back_rather_than_guessing():
    d = coordinate("What is the policy on academic freedom?")
    assert selected(d) == [] and complexity(d) == "unknown"
    assert "fall back" in d.reason


def test_greeting_needs_no_specialist():
    d = coordinate("Hello")
    assert d.intents == ["greeting"] and selected(d) == [] and complexity(d) == "greeting"
    assert d.confidence == 1.0 and d.subtasks == []


# ---------------------------------------------------------------------------
# Contact and count adversarial pair; contact override
# ---------------------------------------------------------------------------
def test_hr_phone_number_is_institutional_not_faculty_services():
    d = coordinate("What is the HR phone number?")
    assert selected(d) == ["institutional"] and d.intents[0] == "contact"
    assert d.metadata["dropped_domains"] == ["faculty_services"]
    assert "channel lookup" in d.reason


def test_number_of_degree_programs_is_institutional_count():
    d = coordinate("What is the number of degree programs?")
    assert selected(d) == ["institutional"] and d.intents[0] == "count"


def test_contact_lookup_is_institutional_even_without_navigation_words():
    d = coordinate("What is the email of my research supervisor?")
    assert selected(d) == ["institutional"] and d.metadata["dropped_domains"] == ["research"]


def test_contact_request_about_hr_subject_keeps_faculty_services():
    d = coordinate("Who do I contact about annual leave?")
    assert selected(d) == ["faculty_services", "institutional"]
    assert d.intents[0] == "contact" and d.requires_synthesis is True
    assert "subject specialist" in d.reason


def test_contact_request_about_research_subject_keeps_research():
    d = coordinate("Who should I contact about research funding?")
    assert selected(d) == ["research", "institutional"]


def test_who_do_i_ask_about_blackboard_keeps_teaching_and_institutional():
    d = coordinate("Who do I ask about Blackboard?")
    assert set(selected(d)) == {"teaching", "institutional"} and d.systems == ["blackboard"]


def test_contact_request_with_many_subjects_keeps_institutional_within_budget():
    d = coordinate("Who do I contact about my teaching load and my research grant and my visa?")
    assert "institutional" in selected(d)
    assert len(d.selected_specialists) <= config.PLAN_F_MAX_SPECIALISTS
    assert d.metadata["dropped_domains"]                                # one subject had to be dropped


def test_where_is_a_unit_is_institutional_navigation():
    for question in ("Where is the HR department?", "Where is the research unit?"):
        d = coordinate(question)
        assert selected(d)[0] == "institutional"


# ---------------------------------------------------------------------------
# X-Y: determinism and no network
# ---------------------------------------------------------------------------
def test_decisions_are_deterministic_and_repeatable():
    for question in (FUNDED, TWO_PARTS, "purple monkey dishwasher", "How do I enter final grades in Banner?"):
        first = coordinate(question).to_dict()
        assert all(coordinate(question).to_dict() == first for _ in range(3))
        assert analyze(question).to_dict() == analyze(question).to_dict()


def test_coordinator_never_uses_an_llm():
    for question in (FUNDED, "Name the total programs", "How do I log in to MyUOS?"):
        d = coordinate(question)
        assert d.used_llm is False
        assert d.metadata["router"]["llm_attempted"] is False
    source = (REPO / "handbook_bot" / "agents" / "coordinator.py").read_text(encoding="utf-8")
    for forbidden in ("groq", "Groq", "requests", "urllib", "httpx", "socket"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Z: existing Router behaviour unchanged, Plan F still inactive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,expected", [
    ("Hello", "greeting"),
    ("Who should I contact about medical insurance?", "contact"),
    ("What is the number of degree programs?", "count"),
    ("How many degree programs are offered?", "count"),
    ("When do Fall classes begin?", "date"),
    ("List the relevant policies.", "list"),
    ("Can faculty members work remotely?", "policy_yesno"),
    ("What is the policy on academic freedom?", "policy"),
])
def test_router_matrix_is_unchanged(question, expected):
    assert router.classify(question) == expected


def test_coordinator_is_not_wired_into_the_runtime():
    assert config.PLAN_F_ENABLED is False
    for name in ("orchestrator.py", "qa.py", "retrieval.py", "agents/router.py",
                 "agents/synthesis.py", "agents/verifier.py"):
        text = (REPO / "handbook_bot" / name).read_text(encoding="utf-8")
        assert "coordinator" not in text.lower(), name
    for name in ("ui/pipeline.py", "ui/chat.py", "eval/run_eval.py"):
        assert "coordinator" not in (REPO / name).read_text(encoding="utf-8").lower(), name


def test_every_decision_is_a_valid_serialisable_contract():
    for question in (FUNDED, TWO_PARTS, "Hello", "", "What is the HR phone number?",
                     "What should I do during my first week?"):
        d = coordinate(question)
        assert isinstance(d, CoordinatorDecision)
        payload = json.loads(json.dumps(d.to_dict()))
        assert payload["used_llm"] is False
        assert set(payload["selected_specialists"]) <= set(ALL_SPECIALIST_IDS)


@pytest.mark.parametrize("question,expected_complexity", [
    ("Hello", "greeting"),
    ("", "unknown"),
    ("purple monkey dishwasher", "unknown"),
    ("How do I enter final grades in Banner?", "simple"),
    (FUNDED, "cross_domain"),
])
def test_metadata_has_the_same_shape_for_every_decision(question, expected_complexity):
    d = coordinate(question)
    assert set(d.metadata) == set(METADATA_KEYS)
    assert d.metadata["complexity"] == expected_complexity
    assert isinstance(d.metadata["scores"], dict) and isinstance(d.metadata["cues"], dict)
    assert isinstance(d.metadata["dropped_domains"], list) and isinstance(d.metadata["uncovered_clauses"], list)
    assert set(d.metadata["router"]) == {"route", "is_multipart", "llm_attempted"}
    assert d.metadata["limits"] == {"specialists": config.PLAN_F_MAX_SPECIALISTS,
                                    "subtasks": config.PLAN_F_MAX_SUBTASKS}
    assert isinstance(d.metadata["journey"], bool) and isinstance(d.metadata["analysis"], dict)
    json.dumps(d.metadata)


def test_greeting_and_empty_metadata_carry_no_invented_measurements():
    for question in ("Hello", ""):
        m = coordinate(question).metadata
        assert m["scores"] == {} and m["cues"] == {} and m["dropped_domains"] == []
        assert m["uncovered_clauses"] == [] and m["journey"] is False
