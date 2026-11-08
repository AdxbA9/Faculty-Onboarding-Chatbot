"""Post-freeze PF-8 (decision record D-119): model configuration is visible and a model this account cannot use is named.

* ``GROQ_MODEL`` and ``LLM_REASONING_TOKENS`` come from the environment (``.env`` through ``app.py``), with the documented
  defaults; ``.env.example`` carries the verified values.
* ``groq_check`` classifies SDK failures and describes the configured model's problem from ONE ``models.retrieve`` request.
* The legacy path answers a 404 with the model-configuration notice, not the outage notice and never the refusal.
* The web app listens on this computer only unless ``APP_HOST`` says otherwise.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from handbook_bot import groq_check as G
from handbook_bot import orchestrator as orch
from handbook_bot.qa import REFUSAL

from conftest import FakeGroqClient, LEAVE_QUESTION

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Environment-driven settings (read in a fresh interpreter, so no module reload)
# ---------------------------------------------------------------------------
def _config_in(env):
    code = ("from handbook_bot import config; import json; "
            "print(json.dumps({'model': config.GROQ_MODEL, 'reasoning': config.LLM_REASONING_TOKENS}))")
    clean = {k: v for k, v in os.environ.items() if k not in ("GROQ_MODEL", "LLM_REASONING_TOKENS")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), env={**clean, **env})
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_documented_defaults_apply_without_environment():
    assert _config_in({}) == {"model": "llama-3.3-70b-versatile", "reasoning": 0}


def test_env_selects_the_model_and_the_reasoning_allowance():
    assert _config_in({"GROQ_MODEL": "openai/gpt-oss-120b", "LLM_REASONING_TOKENS": "2048"}) == {"model": "openai/gpt-oss-120b", "reasoning": 2048}


def test_a_negative_allowance_is_clamped_to_zero():
    assert _config_in({"LLM_REASONING_TOKENS": "-5"})["reasoning"] == 0


def test_env_example_documents_the_verified_model_settings():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GROQ_MODEL=openai/gpt-oss-120b" in text
    assert "LLM_REASONING_TOKENS=2048" in text
    assert "APP_HOST" in text
    assert "gsk_" not in text.replace("your_groq_api_key_here", "")


# ---------------------------------------------------------------------------
# Failure classification and the startup check
# ---------------------------------------------------------------------------
class _Status(Exception):
    def __init__(self, status):
        super().__init__("status %d" % status)
        self.status_code = status


class APIConnectionError(Exception):
    pass


@pytest.mark.parametrize("exc,kind", [
    (_Status(404), G.MODEL_UNAVAILABLE), (_Status(401), G.AUTHENTICATION), (_Status(403), G.AUTHENTICATION),
    (_Status(429), G.RATE_LIMIT), (_Status(500), G.OTHER), (APIConnectionError("x"), G.CONNECTION), (ValueError("x"), G.OTHER),
])
def test_error_kind(exc, kind):
    assert G.error_kind(exc) == kind


class _Models:
    def __init__(self, exc):
        self.exc, self.asked = exc, []

    def retrieve(self, model):
        self.asked.append(model)
        if self.exc:
            raise self.exc
        return {"id": model}


class _Client:
    def __init__(self, exc=None):
        self.models = _Models(exc)
        self.options = None

    def with_options(self, **kw):
        self.options = kw
        return self


def test_model_problem_names_the_model_and_the_setting_on_a_404():
    client = _Client(_Status(404))
    problem = G.model_problem(client, "some-model")
    assert "some-model" in problem and "GROQ_MODEL" in problem and "404" in problem
    assert client.models.asked == ["some-model"] and client.options == {"max_retries": 0, "timeout": 10.0}


def test_model_problem_is_none_when_the_model_is_listed_or_there_is_no_client():
    assert G.model_problem(_Client(), "m") is None
    assert G.model_problem(None, "m") is None


def test_model_problem_distinguishes_key_and_network_failures():
    assert "key" in G.model_problem(_Client(_Status(401)), "m").lower()
    assert "inconclusive" in G.model_problem(_Client(APIConnectionError("down")), "m")
    assert "inconclusive" in G.model_problem(_Client(_Status(429)), "m")


# ---------------------------------------------------------------------------
# The legacy answer path
# ---------------------------------------------------------------------------
def test_a_404_from_the_model_gives_the_configuration_notice_not_a_refusal(ask):
    client = FakeGroqClient(_Status(404))
    result = ask(LEAVE_QUESTION, client)
    assert result.answer == orch.MODEL_UNAVAILABLE_REPLY
    assert result.answer not in (REFUSAL, orch.LLM_UNAVAILABLE_REPLY) and result.pages == []
    entry = [e for e in result.agent_trace if e.agent == "synthesis"][-1]
    assert entry.decision == "llm_error" and entry.detail["kind"] == G.MODEL_UNAVAILABLE
    assert "GROQ_MODEL" in result.answer


def test_other_failures_keep_the_outage_notice(ask):
    assert ask(LEAVE_QUESTION, FakeGroqClient(_Status(500))).answer == orch.LLM_UNAVAILABLE_REPLY
    assert ask(LEAVE_QUESTION, FakeGroqClient(APIConnectionError("down"))).answer == orch.LLM_UNAVAILABLE_REPLY


def test_a_404_on_the_retry_also_gives_the_configuration_notice(ask):
    from conftest import FABRICATED_LEAVE
    result = ask(LEAVE_QUESTION, FakeGroqClient(FABRICATED_LEAVE, _Status(404)))
    assert result.answer == orch.MODEL_UNAVAILABLE_REPLY


# ---------------------------------------------------------------------------
# Network exposure of the web app
# ---------------------------------------------------------------------------
def _run_kwargs(monkeypatch, **env):
    from ui import chat
    captured = {}
    monkeypatch.setattr(chat.ui, "run", lambda **kw: captured.update(kw))
    monkeypatch.setattr(chat.ui, "page", lambda path: (lambda fn: fn))
    monkeypatch.setattr(chat.app, "add_static_files", lambda *a, **k: None)
    monkeypatch.setattr(chat.app, "on_startup", lambda *a, **k: None)
    monkeypatch.delenv("APP_HOST", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    chat.start_app(port=8501)
    return captured


def test_the_app_listens_on_this_computer_only_by_default(monkeypatch):
    kwargs = _run_kwargs(monkeypatch)
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8501 and kwargs["reload"] is False


def test_app_host_opens_the_app_to_the_network_on_request(monkeypatch):
    assert _run_kwargs(monkeypatch, APP_HOST="0.0.0.0")["host"] == "0.0.0.0"
