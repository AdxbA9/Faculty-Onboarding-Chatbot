"""Orchestrator: every path through ``run`` with fake models and a scripted
LLM client. The real Verifier is used unless a test injects its own."""
from __future__ import annotations

import dataclasses
import json

from handbook_bot import orchestrator as orch
from handbook_bot.agents.types import TraceEntry, VerifyResult
from handbook_bot.qa import REFUSAL

from conftest import FABRICATED_LEAVE, GROUNDED_LEAVE, LEAVE_QUESTION, FakeGroqClient, FakeReranker


def agents(result):
    return [(entry.agent, entry.decision) for entry in result.agent_trace]


def verifier_entries(result):
    return [entry for entry in result.agent_trace if entry.agent == "verifier"]


# ---------------------------------------------------------------------------
# Paths that never touch the LLM
# ---------------------------------------------------------------------------
def test_greeting_path(ask):
    result = ask("hello")
    assert result.answer == orch.GREETING_REPLY
    assert result.query_type == "greeting"
    assert result.llm_calls == 0 and not result.used_llm
    assert agents(result) == [("router", "greeting")]


def test_missing_api_key_path(ask):
    result = ask(LEAVE_QUESTION, client=None)
    assert result.answer == orch.NO_KEY_REPLY
    assert not result.used_llm and result.llm_calls == 0
    assert ("synthesis", "skipped:no_client") in agents(result)


def test_contact_extractor_path_makes_no_llm_call(ask):
    client = FakeGroqClient(GROUNDED_LEAVE)
    result = ask("What is the phone number of the Information Technology Center?", client)
    assert result.query_type == "contact"
    assert "+971 6 5050000" in result.answer
    assert result.pages == [14]
    assert client.calls == 0 and result.llm_calls == 0 and not result.used_llm
    assert ("synthesis", "extractor") in agents(result)
    assert verifier_entries(result)[-1].decision == "accepted"


def test_date_extractor_path(ask):
    client = FakeGroqClient(GROUNDED_LEAVE)
    result = ask("When do classes begin?", client)
    assert result.query_type == "date"
    assert "Classes begin" in result.answer and result.pages == [49]
    assert client.calls == 0


def test_count_extractor_path(ask):
    client = FakeGroqClient(GROUNDED_LEAVE)
    result = ask("How many degree programs does UoS offer?", client)
    assert result.query_type == "count"
    assert "149" in result.answer and result.pages == [15]
    assert client.calls == 0


def test_count_outside_tool_scope_goes_to_the_llm(ask):
    client = FakeGroqClient("The teaching load is twelve credit hours.\nPages: 40")
    result = ask("What is the total number of teaching hours?", client)
    evidence = [e for e in result.agent_trace if e.agent == "evidence"][0]
    assert result.query_type == "count"
    assert evidence.detail["reason"] == "extractor_not_applicable"
    assert client.calls >= 1


def test_insufficient_evidence_refuses_without_llm(ask):
    client = FakeGroqClient(GROUNDED_LEAVE)
    result = ask(LEAVE_QUESTION, client, reranker=FakeReranker(fixed=-5.0))
    assert result.answer == REFUSAL
    assert result.best_section == "Low confidence retrieval"
    assert client.calls == 0 and result.llm_calls == 0


# ---------------------------------------------------------------------------
# Synthesis + real Verifier
# ---------------------------------------------------------------------------
def test_normal_synthesis_path_accepted_first_time(ask):
    client = FakeGroqClient(GROUNDED_LEAVE)
    result = ask(LEAVE_QUESTION, client)
    assert "thirty days" in result.answer
    assert result.pages == [3]
    assert result.used_llm and result.llm_calls == 1 and not result.retried
    assert [a for a, _ in agents(result)] == ["router", "evidence", "synthesis", "verifier"]
    assert verifier_entries(result)[-1].detail["impl"] == "agents.verifier"


def test_real_verifier_rejects_fabrication_then_retry_succeeds(ask):
    client = FakeGroqClient(FABRICATED_LEAVE, GROUNDED_LEAVE)
    result = ask(LEAVE_QUESTION, client)
    assert result.retried and result.llm_calls == 2 and client.calls == 2
    assert "thirty days" in result.answer and result.pages == [3]
    assert [e.decision for e in verifier_entries(result)] == ["retry_requested", "accepted"]
    assert "5000" in client.prompts[-1]           # the retry prompt carries the targeted feedback


def test_real_verifier_retry_still_failing_refuses(ask):
    client = FakeGroqClient(FABRICATED_LEAVE)
    result = ask(LEAVE_QUESTION, client)
    assert result.answer == REFUSAL
    assert result.pages == []
    assert result.retried and client.calls == 2 and result.llm_calls == 2
    assert [e.decision for e in verifier_entries(result)] == ["retry_requested", "rejected"]


def test_wrong_claimed_page_is_dropped_by_the_verifier(ask):
    client = FakeGroqClient("Faculty members are entitled to thirty days of annual leave each year.\nPages: 7")
    result = ask(LEAVE_QUESTION, client)
    assert result.pages == [3]
    assert 7 not in result.pages


def test_model_without_pages_line_gets_evidence_pages_not_a_fallback(ask):
    client = FakeGroqClient("Faculty members are entitled to thirty days of annual leave each year.")
    result = ask(LEAVE_QUESTION, client)
    assert result.answer.startswith("Faculty members")
    assert result.pages == [3]                    # not "the first three retrieved pages"


# ---------------------------------------------------------------------------
# Retry cap and LLM budget
# ---------------------------------------------------------------------------
def test_maximum_one_retry_even_if_config_is_higher(run, monkeypatch):
    monkeypatch.setattr(orch, "MAX_VERIFY_RETRIES", 5)

    def always_reject(question, synthesis, evidence, sub_questions, *, is_retry):
        return VerifyResult(False, [], [], [], "still wrong")

    client = FakeGroqClient(GROUNDED_LEAVE)
    result = run(LEAVE_QUESTION, client, verifier=always_reject)
    assert result.answer == REFUSAL
    assert client.calls == 2 and result.llm_calls == 2 and result.retried


def test_injected_verifier_retry_then_accept(run):
    def retry_then_accept(question, synthesis, evidence, sub_questions, *, is_retry):
        if not is_retry:
            return VerifyResult(False, [], ["claim"], [], "Cite the leave duration.")
        return VerifyResult(True, [3], [], [], None)

    client = FakeGroqClient("first draft.\nPages: 3", GROUNDED_LEAVE)
    result = run(LEAVE_QUESTION, client, verifier=retry_then_accept)
    assert result.retried and result.llm_calls == 2 and result.pages == [3]
    assert "reviewer rejected" in client.prompts[-1].lower()


def test_llm_budget_zero_makes_no_call(ask, monkeypatch):
    monkeypatch.setattr(orch, "MAX_LLM_CALLS", 0)
    client = FakeGroqClient(GROUNDED_LEAVE)
    result = ask(LEAVE_QUESTION, client)
    assert result.answer == orch.LLM_BUDGET_REPLY
    assert client.calls == 0 and result.llm_calls == 0


def test_llm_budget_hard_ceiling_is_max_calls_plus_one_retry(ask, monkeypatch):
    monkeypatch.setattr(orch, "MAX_LLM_CALLS", 2)
    monkeypatch.setattr(orch, "MAX_VERIFY_RETRIES", 1)
    client = FakeGroqClient(FABRICATED_LEAVE)
    result = ask(LEAVE_QUESTION, client)
    assert client.calls <= orch.MAX_LLM_CALLS + 1
    assert result.llm_calls == client.calls


def test_router_fallback_shares_the_budget_and_reserves_the_answer_call(ask, monkeypatch):
    ambiguous = "Name the total programs"
    grounded = "UoS offers a total of 149 degree programs.\nPages: 15"      # accepted first time
    client = FakeGroqClient('{"intent": "policy", "confidence": 0.6, "is_multipart": false}', grounded)
    result = ask(ambiguous, client)
    router_entry = [e for e in result.agent_trace if e.agent == "router"][0]
    assert router_entry.used_llm and client.calls == 2 and result.llm_calls == 2
    assert not result.retried

    # With one call in the budget the router gives it up so the answer can be written.
    monkeypatch.setattr(orch, "MAX_LLM_CALLS", 1)
    client = FakeGroqClient(grounded)
    result = ask(ambiguous, client)
    router_entry = [e for e in result.agent_trace if e.agent == "router"][0]
    assert router_entry.detail.get("llm_skipped") == "budget"
    assert client.calls == 1 and result.llm_calls == 1 and not result.retried


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_external_api_error_is_reported_not_disguised_as_refusal(ask):
    # A key-shaped token in the error text must never reach the trace.
    client = FakeGroqClient(RuntimeError("503 gsk_fakekey123"))
    result = ask(LEAVE_QUESTION, client)
    assert result.answer == orch.LLM_UNAVAILABLE_REPLY
    assert result.answer != REFUSAL
    assert result.llm_calls == 1 and result.used_llm
    assert "fakekey123" not in json.dumps(result.to_dict())


def test_verifier_exception_falls_back_to_the_part1_check(run):
    def broken(question, synthesis, evidence, sub_questions, *, is_retry):
        raise ValueError("verifier bug")

    result = run(LEAVE_QUESTION, FakeGroqClient(GROUNDED_LEAVE), verifier=broken)
    assert result.answer != REFUSAL
    entry = verifier_entries(result)[-1]
    assert "error" in entry.detail and "after error" in entry.detail["impl"]


def test_verifier_wrong_return_type_falls_back(run):
    def wrong(question, synthesis, evidence, sub_questions, *, is_retry):
        return "yes"

    result = run(LEAVE_QUESTION, FakeGroqClient(GROUNDED_LEAVE), verifier=wrong)
    assert result.answer != REFUSAL
    assert "after error" in verifier_entries(result)[-1].detail["impl"]


def test_verifier_pages_outside_evidence_are_dropped(run):
    def bad_pages(question, synthesis, evidence, sub_questions, *, is_retry):
        return VerifyResult(True, [999, "3", True], [], [], None)

    result = run(LEAVE_QUESTION, FakeGroqClient(GROUNDED_LEAVE), verifier=bad_pages)
    assert result.pages == [3]
    assert verifier_entries(result)[-1].detail["dropped_pages"]


def test_verify_answers_off_skips_the_verifier(ask, monkeypatch):
    monkeypatch.setattr(orch, "VERIFY_ANSWERS", False)
    result = ask(LEAVE_QUESTION, FakeGroqClient("Unrelated text.\nPages: 3"))
    assert verifier_entries(result)[-1].decision == "skipped"
    assert result.answer.startswith("Unrelated")


def test_trace_entries_are_typed_and_serialisable(ask):
    result = ask(LEAVE_QUESTION, FakeGroqClient(GROUNDED_LEAVE))
    assert all(isinstance(e, TraceEntry) for e in result.agent_trace)
    assert all(e.ms >= 0 for e in result.agent_trace)
    json.dumps([dataclasses.asdict(e) for e in result.agent_trace])
