"""Multi-part questions on the legacy path (post-freeze PF-16.5, D-128): evidence per clause under the unchanged gate, the
clauses handed to the Verifier, one completeness retry within the existing budget, and an unsupported clause named instead
of guessed. Fake models and the six-chunk corpus of conftest; no network."""
from __future__ import annotations

import pytest

from conftest import FakeGroqClient, FakeReranker
from handbook_bot import orchestrator
from handbook_bot.qa import REFUSAL

TWO_PART = "How much annual leave are faculty members entitled to, and how many years in rank are required for promotion to associate professor?"
BOTH = ("Faculty members are entitled to thirty days of annual leave each academic year. Promotion to associate professor "
        "requires a minimum of five years in rank.\nPages: 3, 40")
LEAVE_ONLY = "Faculty members are entitled to thirty days of annual leave each academic year.\nPages: 3"


@pytest.mark.parametrize("question, clauses", [
    ("I joined this semester. How many office hours do I need each week, and how many fully paid sick leave days do I get per contract year?",
     ["How many office hours do I need each week", "how many fully paid sick leave days do I get per contract year"]),
    ("How do I reach the HR department by phone? And do they have a fax number?",
     ["How do I reach the HR department by phone", "How do I reach the HR department by phone do they have a fax number"]),
    ("What is the phone number of the College of Graduate Studies and Research?", []),     # "and" inside a name
    ("Who approves leave and promotion requests?", []),                                     # one question
    ("Dr. Sara asks: what is the teaching load for lecturers?", []),
    ("What is the phone number of HR? Thanks.", []),
])
def test_only_two_or_three_real_question_clauses_are_split(question, clauses):
    assert orchestrator.answerable_clauses(question) == clauses


def test_each_clause_brings_its_own_evidence_into_one_synthesis(run):
    client = FakeGroqClient(BOTH)
    result = run(TWO_PART, client=client)
    assert client.calls == 1
    prompt = client.prompts[0]
    assert "thirty days of annual leave" in prompt and "five years in rank" in prompt
    assert "thirty days" in result.answer and "five years" in result.answer
    assert any(e.agent == "clauses" and e.detail["supported"] == 2 for e in result.agent_trace)


def test_an_answer_that_leaves_out_a_clause_gets_the_completeness_retry(run):
    client = FakeGroqClient(LEAVE_ONLY, BOTH)
    result = run(TWO_PART, client=client)
    assert client.calls == 2 and "five years" in result.answer
    assert "leaves out part of the question" in client.prompts[1]


def test_the_retry_is_kept_only_when_it_covers_more(run):
    client = FakeGroqClient(LEAVE_ONLY, LEAVE_ONLY)
    result = run(TWO_PART, client=client)
    assert client.calls == 2 and "thirty days" in result.answer and result.answer != REFUSAL


class _NoParkingEvidence(FakeReranker):
    def predict(self, pairs):
        return [-5.0 if "parking" in q.lower() else s for (q, _c), s in zip(pairs, super().predict(pairs))]


def test_a_clause_without_evidence_is_named_never_guessed(run):
    question = "How much annual leave are faculty members entitled to, and what is the parking fee for faculty members?"
    client = FakeGroqClient(LEAVE_ONLY)
    result = run(question, client=client, reranker=_NoParkingEvidence())
    assert "thirty days" in result.answer
    assert "do not answer this part of the question" in result.answer and "parking fee" in result.answer
    assert "parking" not in client.prompts[0].split("Question:")[0]          # no parking evidence reached the model


def test_merging_adds_nothing_from_a_query_that_failed_its_gate(run):
    # the whole question fails the gate (it names parking, scored -5.0 for every chunk); only the leave clause, which
    # passed its own gate, may contribute items
    question = "How much annual leave are faculty members entitled to, and what is the parking fee for faculty members?"
    result = run(question, client=FakeGroqClient(LEAVE_ONLY), reranker=_NoParkingEvidence())
    assert result.items and all(i["rerank_score"] != -5.0 for i in result.items)


def test_when_no_clause_has_evidence_the_answer_is_the_refusal(run):
    client = FakeGroqClient(LEAVE_ONLY)
    result = run(TWO_PART, client=client, reranker=FakeReranker(fixed=-9.0))
    assert result.answer == REFUSAL and client.calls == 0


def test_a_single_question_takes_the_unchanged_path(run):
    client = FakeGroqClient(LEAVE_ONLY)
    result = run("How much annual leave are faculty members entitled to?", client=client)
    assert not any(e.agent == "clauses" for e in result.agent_trace) and client.calls == 1
