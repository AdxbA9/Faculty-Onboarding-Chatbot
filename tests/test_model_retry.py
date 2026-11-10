"""Bounded, counted retry of a transient model failure (post-freeze PF-16.5, D-128). The SDK never repeats a request on
its own; a connection error, timeout, HTTP 429 or 5xx is repeated by ``qa.ask_groq`` at most LLM_TRANSPORT_RETRIES
times, each attempt an ordinary counted request; authentication, unknown model and the process cap are never repeated."""
from __future__ import annotations

from conftest import GROUNDED_LEAVE, LEAVE_QUESTION, FakeGroqClient
from handbook_bot import groq_check as G
from handbook_bot import orchestrator as orch
from handbook_bot import qa


class _Status(Exception):
    def __init__(self, status):
        super().__init__("HTTP %d" % status)
        self.status_code = status


class APIConnectionError(Exception):
    pass


class _Recorder(FakeGroqClient):
    def __init__(self, *replies):
        super().__init__(*replies)
        self.options = []

    def with_options(self, **options):
        self.options.append(options)
        return self


def test_a_transient_failure_is_repeated_and_the_answer_is_served(ask):
    client = _Recorder(APIConnectionError("reset"), GROUNDED_LEAVE)
    result = ask(LEAVE_QUESTION, client)
    assert client.calls == 2 and "thirty days" in result.answer
    assert {"max_retries": 0} in client.options                     # the SDK itself never repeats a request


def test_server_errors_are_repeated_at_most_the_configured_number_of_times(ask):
    client = FakeGroqClient(_Status(503))                            # every attempt fails
    result = ask(LEAVE_QUESTION, client)
    assert client.calls == 1 + qa.LLM_TRANSPORT_RETRIES and result.answer == orch.LLM_UNAVAILABLE_REPLY


def test_authentication_and_unknown_model_errors_are_never_repeated(ask):
    client = FakeGroqClient(_Status(401))
    result = ask(LEAVE_QUESTION, client)
    assert client.calls == 1 and result.answer == orch.LLM_UNAVAILABLE_REPLY
    assert any(e.decision == "llm_error" and e.detail["kind"] == G.AUTHENTICATION for e in result.agent_trace)
    client = FakeGroqClient(_Status(404))
    assert ask(LEAVE_QUESTION, client).answer == orch.MODEL_UNAVAILABLE_REPLY and client.calls == 1


def test_every_attempt_is_counted_by_a_capped_client_and_the_cap_is_never_exceeded(ask):
    inner = FakeGroqClient(_Status(503), GROUNDED_LEAVE)
    capped = G.CappedClient(inner, 1)
    result = ask(LEAVE_QUESTION, capped)
    assert capped.calls_issued == 1 and capped.calls_refused == 1 and inner.calls == 1   # the repeat was refused, not sent
    assert result.answer == orch.LLM_UNAVAILABLE_REPLY


def test_the_transient_classification():
    assert qa.transient_error(APIConnectionError("x")) and qa.transient_error(_Status(429)) and qa.transient_error(_Status(502))
    assert not qa.transient_error(_Status(400)) and not qa.transient_error(_Status(401)) and not qa.transient_error(_Status(404))
    assert not qa.transient_error(G.LLMCallCapExceeded("spent"))
