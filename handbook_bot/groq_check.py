"""
Groq model-availability diagnostics (post-freeze PF-8, decision record D-119).

WHY
    The code default ``GROQ_MODEL`` (``llama-3.3-70b-versatile``) stopped being available to the project's Groq account
    on 2026-11-01 (HTTP 404). Nothing told the operator: the sidebar still read "Ready", and every answer that needed the
    model came back as the generic "service did not respond" reply, which reads like an outage. A misconfigured model
    must be visible at startup and named in the reply.

WHAT
    * :func:`error_kind`: a coarse class for an exception raised by the Groq SDK, read from its ``status_code`` or its
      class name, so the orchestrator can tell "the model does not exist for this account" from a transient failure.
    * :func:`model_problem`: ONE ``models.list`` request (never an inference) that says why the configured model
      cannot answer, or None. Used by ``app.py`` at startup when a key is configured.

    Nothing here changes which model is used, how it is prompted, or any retrieval or verification setting. The key is
    never read or printed; the SDK is imported only through the client the caller passes in.
"""
from __future__ import annotations

from typing import Any, Optional

MODEL_UNAVAILABLE = "model_unavailable"
AUTHENTICATION = "authentication"
RATE_LIMIT = "rate_limit"
CONNECTION = "connection"
OTHER = "other"


def error_kind(exc: BaseException) -> str:
    """The kind of failure ``exc`` is: a 404 is a model this account cannot use, 401 and 403 a key problem, 429 a rate
    limit, a connection or timeout error a network problem; anything else is ``other``."""
    status = getattr(exc, "status_code", None)
    if status == 404:
        return MODEL_UNAVAILABLE
    if status in (401, 403):
        return AUTHENTICATION
    if status == 429:
        return RATE_LIMIT
    name = type(exc).__name__
    if status is None and ("Connection" in name or "Timeout" in name):
        return CONNECTION
    return OTHER


def describe(kind: str, model: str) -> str:
    """One sentence an operator can act on."""
    if kind == MODEL_UNAVAILABLE:
        return ("Groq model %r does not exist or is not available to this account (HTTP 404). Set GROQ_MODEL in .env to a "
                "model the account can use and restart the app." % model)
    if kind == AUTHENTICATION:
        return "Groq rejected the API key (HTTP 401/403). Check GROQ_API_KEY in .env."
    if kind == RATE_LIMIT:
        return "Groq rate limit reached (HTTP 429); the model check is inconclusive. Try again in a minute."
    if kind == CONNECTION:
        return "Groq could not be reached (network or timeout); the model check is inconclusive."
    return "Groq model %r could not be checked." % model


def model_problem(client: Any, model: str, *, timeout: float = 10.0) -> Optional[str]:
    """Why ``model`` cannot answer through ``client``, or None when Groq lists it for this account. ``client`` None (no
    key) gives None: the missing key is reported elsewhere.

    The check reads ``models.list`` and looks for the id. ``models.retrieve`` is not used: for an id with a slash
    (``openai/gpt-oss-120b``) it answers 404 even when the model is listed and answers, so it would report a working
    configuration as broken (seen on 2026-11-08)."""
    if client is None:
        return None
    caller = client.with_options(max_retries=0, timeout=timeout) if hasattr(client, "with_options") else client
    try:
        listed = {getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None) for m in caller.models.list().data}
    except Exception as exc:                 # the SDK raises many classes; the kind is read from the exception itself
        return describe(error_kind(exc), model)
    return None if model in listed else describe(MODEL_UNAVAILABLE, model)


class LLMCallCapExceeded(RuntimeError):
    """A request beyond ``LLM_PROCESS_CALL_CAP`` was refused (post-freeze PF-11); reported as a service problem, never dropped."""


class _Counter:
    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.issued = 0
        self.refused = 0


class CappedClient:
    """A Groq client whose chat completions are counted per process and refused beyond a cap. ``with_options`` keeps the
    same counter, so the router's and the translators' short-timeout callers share the budget. Every other attribute is
    the wrapped client's."""

    def __init__(self, inner: Any, cap: int, counter: Optional[_Counter] = None) -> None:
        self._inner = inner
        self._counter = counter or _Counter(cap)
        self.chat = _CappedChat(inner.chat, self._counter)

    @property
    def calls_issued(self) -> int:
        return self._counter.issued

    @property
    def calls_refused(self) -> int:
        return self._counter.refused

    @property
    def cap(self) -> int:
        return self._counter.cap

    def with_options(self, **options: Any) -> "CappedClient":
        inner = self._inner.with_options(**options) if hasattr(self._inner, "with_options") else self._inner
        return CappedClient(inner, self._counter.cap, self._counter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CappedChat:
    def __init__(self, chat: Any, counter: _Counter) -> None:
        self._chat = chat
        self.completions = _CappedCompletions(chat.completions, counter)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _CappedCompletions:
    def __init__(self, completions: Any, counter: _Counter) -> None:
        self._completions = completions
        self._counter = counter

    def create(self, **kwargs: Any) -> Any:
        if self._counter.issued >= self._counter.cap:
            self._counter.refused += 1
            raise LLMCallCapExceeded("the process cap of %d model request(s) is spent" % self._counter.cap)
        self._counter.issued += 1
        return self._completions.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


__all__ = ["AUTHENTICATION", "CONNECTION", "CappedClient", "LLMCallCapExceeded", "MODEL_UNAVAILABLE", "OTHER", "RATE_LIMIT", "describe",
           "error_kind", "model_problem"]
