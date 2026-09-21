"""Router agent: routing matrix, the ``number`` regression, ambiguity, multi-part
detection and every failure mode of the optional LLM arbitration."""
from __future__ import annotations

import json
import time

import pytest

from handbook_bot.agents import router
from handbook_bot.agents.types import ALL_ROUTES, RouteDecision
from handbook_bot.retrieval import classify_query

from conftest import FakeGroqClient

ROUTE_MATRIX = [
    ("Hello", "greeting"),
    ("Who should I contact about medical insurance?", "contact"),
    ("What is the phone number of the Information Technology Center?", "contact"),
    ("What is the fax number for the Registrar's office?", "contact"),
    ("What is the number of degree programs?", "count"),
    ("How many degree programs are offered?", "count"),
    ("When do Fall classes begin?", "date"),
    ("What is the last day for add/drop?", "date"),
    ("List the relevant policies.", "list"),
    ("Name the standing committees of the University Council.", "list"),
    ("Can faculty members work remotely?", "policy_yesno"),
    ("Does UoS allow faculty to work remotely?", "policy_yesno"),
    ("What is the policy on academic freedom?", "policy"),
]


@pytest.mark.parametrize("question,expected", ROUTE_MATRIX)
def test_route_matrix(question, expected):
    assert router.classify(question) == expected


def test_number_regression_against_part1():
    """Part 1 sent "the number of degree programs" to contact; the Router must not."""
    question = "What is the number of degree programs?"
    assert classify_query(question) == "contact"
    assert router.classify(question) == "count"


def test_number_for_office_is_still_contact():
    assert router.classify("What is the number for the IT help desk?") == "contact"
    assert router.classify("What is the phone number of the Registrar?") == "contact"


def test_every_intent_is_scored_and_confidence_bounded():
    report = router.route_with_report("How many departments have a phone number?", llm_fallback=False)
    assert set(report.decision.scores) == set(ALL_ROUTES)
    assert 0.0 <= report.decision.confidence <= 1.0
    # The interrogative decides; a body keyword must not make it ambiguous.
    assert report.decision.primary_route == "count"
    assert not report.ambiguous


def test_greeting_is_deterministic_and_final():
    report = router.route_with_report("hello!", groq_client=FakeGroqClient("x"))
    assert report.decision.primary_route == "greeting"
    assert report.decision.confidence == 1.0
    assert not report.llm_attempted


@pytest.mark.parametrize("question,expected", [
    ("When do classes begin? And how many programs are offered?", True),
    ("What is the phone number of the IT Center and when do classes begin?", True),
    ("What are the rules and regulations for promotion?", False),
    ("What is the policy? Thanks in advance.", False),
    ("How many departments have a phone number?", False),
])
def test_multipart_detection(question, expected):
    decision = router.route_with_report(question, llm_fallback=False).decision
    assert decision.is_multipart is expected


# ---------------------------------------------------------------------------
# Ambiguity and the LLM fallback
# ---------------------------------------------------------------------------
AMBIGUOUS = "Name the total programs"      # list vs count tie in deterministic scoring


def test_ambiguous_question_is_detected():
    report = router.route_with_report(AMBIGUOUS, llm_fallback=False)
    assert report.ambiguous
    assert len(report.candidates) >= 2
    # Deterministic choice among tied candidates is stable and never an LLM.
    assert not report.decision.used_llm


def test_no_client_skips_the_llm():
    report = router.route_with_report(AMBIGUOUS, groq_client=None)
    assert report.llm_skipped == "no_client"
    assert not report.llm_attempted


def test_disabled_fallback_skips_the_llm():
    client = FakeGroqClient('{"intent": "list", "confidence": 0.9}')
    report = router.route_with_report(AMBIGUOUS, groq_client=client, llm_fallback=False)
    assert report.llm_skipped == "disabled"
    assert client.calls == 0


def test_budget_gate_can_refuse_the_llm():
    client = FakeGroqClient('{"intent": "list", "confidence": 0.9}')
    report = router.route_with_report(AMBIGUOUS, groq_client=client, llm_gate=lambda: False)
    assert report.llm_skipped == "budget"
    assert client.calls == 0


def test_llm_failure_falls_back_and_redacts_keys():
    client = FakeGroqClient(RuntimeError("boom gsk_fakekey123"))
    report = router.route_with_report(AMBIGUOUS, groq_client=client)
    assert report.llm_attempted
    assert report.llm_error and "fakekey123" not in report.llm_error
    assert not report.decision.used_llm
    assert report.decision.primary_route in report.candidates + ["policy"]


@pytest.mark.parametrize("reply", [
    "not json at all",
    '{"intent": "greeting", "confidence": 0.9}',          # outside the candidates
    '{"intent": "list", "confidence": 7}',                # schema violation
    '{"confidence": 0.5}',                                # missing intent
])
def test_invalid_structured_output_falls_back(reply):
    report = router.route_with_report(AMBIGUOUS, groq_client=FakeGroqClient(reply))
    assert report.llm_error
    assert not report.decision.used_llm


def test_valid_llm_reply_is_used():
    report = router.route_with_report(AMBIGUOUS, llm_fallback=False)
    chosen = report.candidates[0]
    client = FakeGroqClient(json.dumps({"intent": chosen, "confidence": 0.77, "is_multipart": False}))
    report = router.route_with_report(AMBIGUOUS, groq_client=client)
    assert report.decision.used_llm
    assert report.decision.primary_route == chosen
    assert report.decision.confidence == 0.77
    assert client.calls == 1


def test_clear_question_never_calls_the_llm():
    client = FakeGroqClient("x")
    report = router.route_with_report("How many programs does UoS offer?", groq_client=client)
    assert client.calls == 0 and not report.llm_attempted


@pytest.mark.parametrize("question", ["", None, "???", "   ", "a" * 5000, "and " * 3000,
                                      "?" * 3000, "number of the " * 2000])
def test_never_raises_and_no_regex_blowup(question):
    started = time.perf_counter()
    decision = router.route(question)
    assert isinstance(decision, RouteDecision)
    assert decision.primary_route in ALL_ROUTES
    assert time.perf_counter() - started < 2.0
