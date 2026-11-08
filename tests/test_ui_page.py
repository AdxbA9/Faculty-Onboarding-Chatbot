"""The redesigned chat page driven by NiceGUI's simulated user (post-freeze PF-10).

No browser, no model, no corpus: the knowledge base is a stand-in object and the pipeline call is replaced by a scripted
function, so these tests exercise the page's own behaviour: first-use screen, sending (button and the keyboard handler),
processing state, answer rendering with sources and evidence, notices for refusals and service problems, empty-message
help, New chat, conversation list, and the Developer toggle that only ``?dev=1`` shows.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest
from nicegui.testing import User

from handbook_bot.orchestrator import LLM_UNAVAILABLE_REPLY
from handbook_bot.qa import QAResult, REFUSAL
from ui import chat

HR_QUESTION = "What is the phone number of the Human Resources Department?"
HR_ANSWER = "The phone number is +971 6 5050023."
HR_ITEM = {"chunk": "Human Resources +971 6 5050023 +971 6 5585200",
           "meta": {"page": 54, "source_title": "UOS Faculty Handbook 2025/26", "section_title": "Contact Information",
                    "source_id": "uos_faculty_handbook_2025_26", "chunk_type": "row"}}


class Scripted:
    """Stands in for ``chat._call_pipeline``: returns the queued results in order and records the questions."""

    def __init__(self, *results: QAResult):
        self.results: List[QAResult] = list(results)
        self.questions: List[str] = []
        self.diagnostics: Dict[str, Any] = {"mode": "active", "served": "plan_f", "attempted": True, "specialist_id": "institutional",
                                            "plan_f_path": "single_finding_passthrough", "plan_f_pages": [54]}

    def __call__(self, kb, question: str):
        self.questions.append(question)
        result = self.results.pop(0) if self.results else QAResult(answer=HR_ANSWER, pages=[54], items=[HR_ITEM])
        served = "plan_f" if result.answer == HR_ANSWER else "legacy"
        return result, {**self.diagnostics, "served": served}, 12.5


@pytest.fixture
def scripted(monkeypatch) -> Scripted:
    script = Scripted()
    monkeypatch.setattr(chat, "_call_pipeline", script)
    monkeypatch.setitem(chat._KB_STATE, "kb", object())
    monkeypatch.setitem(chat._KB_STATE, "error", None)
    monkeypatch.setattr(chat, "_CLIENTS", {})
    monkeypatch.setenv("GROQ_API_KEY", "test-key-for-the-page-tests")
    chat.register_pages()
    return script


async def _settle(user: User, text: str, timeout: float = 5.0) -> None:
    await user.should_see(text, retries=int(timeout * 10))


# ---------------------------------------------------------------------------
# First-use screen
# ---------------------------------------------------------------------------
async def test_first_use_screen_explains_the_scope_and_offers_working_examples(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await user.should_see("Welcome to the Faculty Onboarding Assistant")
    await user.should_see("What you can ask")
    await user.should_see("Grounded in approved documents")
    await user.should_see(chat.SESSION_NOTE)
    for question in chat.EXAMPLE_QUESTIONS:
        await user.should_see(question)
    await user.should_not_see("Developer")
    await user.should_not_see("gpt-oss")                      # no model id for ordinary users
    assert len(user.find(marker="example").elements) == len(chat.EXAMPLE_QUESTIONS)
    assert len(user.find(marker="session").elements) == 1


async def test_an_example_shortcut_sends_its_question(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(chat.EXAMPLE_QUESTIONS[0]).click()
    await _settle(user, HR_ANSWER)
    assert scripted.questions == [chat.EXAMPLE_QUESTIONS[0]]


# ---------------------------------------------------------------------------
# Sending, processing and the answer
# ---------------------------------------------------------------------------
async def test_the_send_button_renders_the_turn_with_sources_and_evidence(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="composer").type(HR_QUESTION)
    user.find(marker="send").click()
    await _settle(user, HR_ANSWER)
    await user.should_see(HR_QUESTION)                        # the user's own bubble
    await user.should_see("Sources")
    await user.should_see("Page 54")
    await user.should_see("Supporting evidence (1)")
    await user.should_see("Copy answer")
    await user.should_see("Verified · Institutional Navigation")
    assert scripted.questions == [HR_QUESTION]
    assert user.find(marker="composer").elements.pop().value == ""   # the box is emptied on send
    assert len(user.find(marker="session").elements) == 1


async def test_the_keyboard_handler_sends_like_the_button(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="composer").type("Who approves conference travel requests?")
    user.find(marker="composer").trigger("keydown")           # the browser-side filter (D-09) already decided it was a plain Enter
    await _settle(user, HR_ANSWER)
    assert scripted.questions == ["Who approves conference travel requests?"]


async def test_the_processing_state_is_announced_while_the_pipeline_runs(user: User, scripted: Scripted, monkeypatch) -> None:
    gate = asyncio.Event()
    import time as _time

    def slow(kb, question):
        while not gate.is_set():
            _time.sleep(0.02)
        return QAResult(answer=HR_ANSWER, pages=[54], items=[HR_ITEM]), scripted.diagnostics, 1.0

    monkeypatch.setattr(chat, "_call_pipeline", slow)
    await user.open("/")
    user.find(marker="composer").type(HR_QUESTION)
    user.find(marker="send").click()
    await user.should_see(chat.presentation.SEARCHING_TEXT)
    gate.set()
    await _settle(user, HR_ANSWER)
    await user.should_not_see(chat.presentation.SEARCHING_TEXT)


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------
async def test_a_refusal_is_a_labelled_notice_without_sources_or_evidence(user: User, scripted: Scripted) -> None:
    scripted.results.append(QAResult(answer=REFUSAL, pages=[], items=[HR_ITEM] * 3))
    await user.open("/")
    user.find(marker="composer").type("Something the documents do not cover")
    user.find(marker="send").click()
    await _settle(user, "No supporting evidence found")
    await user.should_see(REFUSAL)
    await user.should_not_see("Supporting evidence (")
    await user.should_not_see("Sources")
    await user.should_not_see("Copy answer")


async def test_a_service_problem_keeps_the_backend_sentence_under_its_label(user: User, scripted: Scripted) -> None:
    scripted.results.append(QAResult(answer=LLM_UNAVAILABLE_REPLY))
    await user.open("/")
    user.find(marker="composer").type("What is the purpose of the Faculty Handbook?")
    user.find(marker="send").click()
    await _settle(user, "Language-model service unavailable")
    await user.should_see(LLM_UNAVAILABLE_REPLY)
    await user.should_see("Please send the question again in a moment.")


async def test_an_exception_in_the_pipeline_becomes_the_safe_error_notice(user: User, scripted: Scripted, monkeypatch) -> None:
    def boom(kb, question):
        raise RuntimeError("secret path C:/private/key gsk_ABCD...")

    monkeypatch.setattr(chat, "_call_pipeline", boom)
    await user.open("/")
    user.find(marker="composer").type("anything")
    user.find(marker="send").click()
    await _settle(user, "Something went wrong")
    await user.should_see(chat.presentation.ERROR_TEXT)
    await user.should_not_see("gsk_ABCD")
    await user.should_not_see("C:/private")


# ---------------------------------------------------------------------------
# Empty message, New chat, Developer toggle
# ---------------------------------------------------------------------------
async def test_an_empty_message_shows_help_and_sends_nothing(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="send").click()
    await user.should_see("Type a question to send.")
    user.find(marker="composer").type("   ")
    user.find(marker="composer").trigger("keydown")
    await asyncio.sleep(0.2)
    assert scripted.questions == []
    await user.should_see("Welcome to the Faculty Onboarding Assistant")


async def test_new_chat_starts_a_fresh_conversation_and_keeps_the_previous_one_listed(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="composer").type(HR_QUESTION)
    user.find(marker="send").click()
    await _settle(user, HR_ANSWER)
    user.find(marker="new-chat").click()
    await user.should_see("Welcome to the Faculty Onboarding Assistant")
    await user.should_not_see(HR_ANSWER)
    sessions = user.find(marker="session").elements
    assert len(sessions) == 2
    user.find(marker="new-chat").click()                      # a second New chat on an empty conversation adds nothing
    await asyncio.sleep(0.1)
    assert len(user.find(marker="session").elements) == 2


async def test_the_phone_width_new_chat_button_exists_too(user: User, scripted: Scripted) -> None:
    await user.open("/")
    assert len(user.find(marker="new-chat-phone").elements) == 1


async def test_developer_diagnostics_only_with_the_dev_query(user: User, scripted: Scripted) -> None:
    await user.open("/?dev=1")
    await user.should_see("Developer")
    await user.open("/")
    await user.should_not_see("Developer")


async def test_the_theme_starts_light_and_the_toggle_switches_it(user: User, scripted: Scripted) -> None:
    await user.open("/")
    state = next(iter(chat._CLIENTS.values()))
    assert state["theme"] == "light"
    user.find(marker="theme").click()
    await asyncio.sleep(0.1)
    assert state["theme"] == "dark"
