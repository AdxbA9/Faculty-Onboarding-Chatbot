"""
Faculty Services & HR alignment with the Coordinator: the shared HR cue lexicon routes a question to Faculty Services &
HR exactly when the specialist accepts it, collision words never route alone, and the frozen specialists still hand HR
clauses off (Milestone 4, Step 4.11; D-085).
"""
from __future__ import annotations

import re

import pytest

from handbook_bot.agents import coordinator
from handbook_bot.agents.specialists import domain_cues as D
from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists import research as R
from handbook_bot.agents.specialists import teaching as T
from handbook_bot.agents.specialists.contracts import SpecialistId


def selected(decision):
    return [t.specialist_id.value for t in decision.subtasks]


# ---------------------------------------------------------------------------
# 1. One lexicon
# ---------------------------------------------------------------------------
def test_coordinator_reads_the_shared_hr_lexicon():
    strong = [p.pattern for sid, weight, p in coordinator._DOMAIN_CUES if sid is SpecialistId.FACULTY_SERVICES and weight == coordinator.STRONG]
    hints = [p.pattern for sid, weight, p in coordinator._DOMAIN_CUES if sid is SpecialistId.FACULTY_SERVICES and weight == coordinator.HINT]
    assert strong == list(D.HR_OWNERSHIP_PATTERNS) and hints == list(D.HR_CONTEXT_PATTERNS)


def test_domain_cues_import_only_the_standard_library():
    import handbook_bot.agents.specialists.domain_cues as module
    source = open(module.__file__, encoding="utf-8").read()
    assert not re.search(r"^from (?:handbook_bot|\.)", source, re.M) and not re.search(r"^import (?:handbook_bot|numpy|faiss)", source, re.M)


@pytest.mark.parametrize("question", [
    "What is the probation period for new faculty?", "How many days of sick leave am I entitled to?", "Is the furniture allowance AED 30,000?",
    "Does the Chancellor approve termination for cause?", "What is the demise policy?", "How are Emirati faculty members hired?",
    "What compensation do administrative positions receive?", "Can I register my child at the University nursery?", "Is my employee ID card issued by HR?",
    "What is the basic salary of an associate professor?", "Who approves additional maternity leave without pay?", "Does the dean receive AED 10,000 monthly?",
])
def test_hr_questions_select_faculty_services_and_the_specialist_accepts_them(question):
    assert "faculty_services" in selected(coordinator.coordinate(question)), question
    assert F.classify_clause(question)[0] is True, question


@pytest.mark.parametrize("question", [
    "What classes do I teach this semester?", "Is student attendance mandatory?", "What is my teaching load?", "What is the policy on research misconduct?",
    "Can I travel abroad during the semester?", "When does the semester start?", "Who is the dean of the college?",
])
def test_collision_words_do_not_route_to_hr_alone(question):
    assert "faculty_services" not in selected(coordinator.coordinate(question)), question
    assert F.classify_clause(question)[0] is False, question


def test_hr_owned_rule_needs_a_strong_cue_or_two_context_cues():
    assert D.hr_owned("What is the probation period?")
    assert not D.hr_owned("Is the evaluation annual?")              # one context cue
    assert D.hr_owned("Is the parking evaluation annual?")             # two distinct context cues
    assert not D.hr_owned("Leave the campus early.")
    assert D.hr_owned("Does the dean receive AED 10,000 monthly?")


# ---------------------------------------------------------------------------
# 2. Collisions stay multi-domain where both halves exist
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,expected", [
    ("What are the faculty appointment classes?", {"faculty_services"}),
    ("Does the annual evaluation process cover my teaching evaluation?", {"faculty_services", "teaching"}),
    ("Does promotion require published research and who approves it?", {"faculty_services", "research", "institutional"}),
    ("Is my teaching load reduced by the administrative compensation of a dean?", {"faculty_services", "teaching"}),
    ("Can I take leave to attend a conference?", {"faculty_services", "research"}),
    ("Is free housing provided and where is the housing office?", {"faculty_services", "institutional"}),
])
def test_collisions_select_every_domain_with_a_cue(question, expected):
    assert set(selected(coordinator.coordinate(question))) >= expected, question


# ---------------------------------------------------------------------------
# 3. The frozen specialists still hand HR clauses off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", ["How do I apply for annual leave?", "What is the probation period?", "Is housing provided?", "What is my salary?"])
def test_teaching_and_research_hand_hr_clauses_off(clause):
    assert SpecialistId.FACULTY_SERVICES in T.other_domains(clause)
    assert SpecialistId.FACULTY_SERVICES in R.other_domains(clause)


def test_hr_hands_teaching_research_and_navigation_off():
    assert F.other_domains("What is my teaching load?") == [SpecialistId.TEACHING]
    assert F.other_domains("How many theses may I supervise?") == [SpecialistId.RESEARCH]
    assert F.other_domains("Where is the HR office?")[0] is SpecialistId.INSTITUTIONAL
    assert F.other_domains("What is the probation period?") == []


def test_routing_is_deterministic():
    for q in ("What is the probation period?", "How many days of sick leave do I get and what is my teaching load?"):
        assert coordinator.coordinate(q).to_dict() == coordinator.coordinate(q).to_dict()
