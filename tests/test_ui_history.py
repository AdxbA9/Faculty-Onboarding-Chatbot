"""Saved conversations on the chat page, driven by NiceGUI's simulated user (post-freeze PF-13).

A temporary database per test (the ``scripted`` fixture sets ``APP_DB_PATH``); the simulated browser keeps its session
cookie between ``user.open`` calls, so reopening the page is a refresh. What is checked: a saved conversation survives a
refresh and is replayed from the database (never re-asked), the sidebar lists, reopens, renames (with a dialog) and deletes
(with confirmation), the settings dialog persists the interface language, the answer language, the theme and the voice
language, Arabic turns keep their translation state, two browsers never see each other's conversations, and a failing or
missing database is reported instead of hidden.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
from nicegui.testing import User
from nicegui.testing.user_interaction import UserInteraction

from handbook_bot.qa import QAResult
from persistence import StoreUnavailable, get_store
from test_ui_page import HR_ANSWER, HR_ITEM, HR_QUESTION, Scripted, _settle, scripted  # noqa: F401  (fixture re-used)
from ui import chat, i18n

TRAVEL_QUESTION = "Who approves conference travel requests?"
TRAVEL_ANSWER = "The Dean approves conference travel requests."
ARABIC_QUESTION = "ما هي سياسة الحضور؟"
ARABIC_ANSWER = "تنص السياسة على حضور 75% من المحاضرات (ص. 12)."
ENGLISH_SOURCE = "The policy requires attending 75% of classes (p. 12)."


def _state() -> dict:
    return list(chat._CLIENTS.values())[-1]


def _owner_id() -> int:
    return _state()["owner_id"]


async def _ask(user: User, question: str, answer: str) -> None:
    user.find(marker="composer").type(question)
    user.find(marker="send").click()
    await _settle(user, answer)


async def _tick(seconds: float = 0.15) -> None:
    await asyncio.sleep(seconds)


def _session_buttons(user: User, title_fragment: str) -> set:
    """The sidebar (or history dialog) buttons whose title contains the fragment."""
    found = set()
    for button in user.find(marker="session").elements:
        for child in button.default_slot.children:
            if title_fragment in (getattr(child, "content", "") or ""):
                found.add(button)
    return found


def _click_session(user: User, title_fragment: str, newest: bool = False) -> None:
    buttons = sorted(_session_buttons(user, title_fragment), key=lambda e: e.id)
    assert buttons, "no listed conversation contains %r" % title_fragment
    UserInteraction(user, {buttons[-1] if newest else buttons[0]}, None).click()


# ---------------------------------------------------------------------------
# Persistence across a refresh
# ---------------------------------------------------------------------------
async def test_a_conversation_survives_a_refresh_and_is_replayed_from_the_database(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await _tick()
    owner = _owner_id()
    store = get_store()
    saved = store.list_conversations(owner)
    assert len(saved) == 1, saved
    assert saved[0]["title"] == chat._title_for(HR_QUESTION) and saved[0]["message_count"] == 2, saved

    await user.open("/")                                                   # the refresh: same browser, new page
    await user.should_see(HR_ANSWER)
    await user.should_see(HR_QUESTION)
    await user.should_see("Saved answer")                                  # a replayed answer says it is saved
    await user.should_see("Page 54")                                       # with its citations
    await user.should_see("Contact Information")                           # and its evidence
    assert scripted.questions == [HR_QUESTION]                             # displayed from the record, never asked again
    assert _owner_id() == owner
    turn = _state()["sessions"][0]["messages"][1]
    assert turn["stored"] is True and turn["result"].answer == HR_ANSWER and turn["result"].pages == [54]
    assert turn["result"].items[0]["chunk"] == HR_ITEM["chunk"]


async def test_the_first_use_screen_says_conversations_are_saved_and_lists_none_yet(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await user.should_see(chat.SESSION_NOTE)
    await user.should_see("No saved conversations yet.")
    assert len(user.find(marker="session").elements) == 1                  # the unsaved draft
    with pytest.raises(AssertionError):
        user.find(marker="session-menu")                                    # a draft has nothing to rename or delete
    assert get_store().counts()["owners"] == 0                              # a visit alone leaves no row behind
    assert _state()["owner_id"] is None and _state()["owner_token"]
    user.find(marker="new-chat").click()
    await _tick()
    assert get_store().counts()["owners"] == 0                              # nor does New chat on an empty page
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await _tick()
    assert get_store().counts()["owners"] == 1                              # the first stored turn creates the owner


# ---------------------------------------------------------------------------
# The sidebar: list, reopen, continue
# ---------------------------------------------------------------------------
async def test_saved_conversations_are_listed_newest_first_and_reopen_without_a_new_request(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    user.find(marker="new-chat").click()
    await user.should_see("Welcome to the Faculty Onboarding Assistant")
    scripted.results.append(QAResult(answer=TRAVEL_ANSWER, pages=[70], items=[HR_ITEM]))
    await _ask(user, TRAVEL_QUESTION, TRAVEL_ANSWER)
    await _tick()
    titles = [s["title"] for s in _state()["sessions"]]
    assert titles == [TRAVEL_QUESTION, chat._title_for(HR_QUESTION)]
    assert len(user.find(marker="session").elements) == 2
    _click_session(user, "phone number")                                   # the older conversation
    await user.should_see(HR_ANSWER)
    await user.should_not_see(TRAVEL_ANSWER)
    assert scripted.questions == [HR_QUESTION, TRAVEL_QUESTION]
    await _tick()
    assert get_store().get_preferences(_owner_id())["extra"]["active_conversation"] == _state()["active_id"]


async def test_continuing_a_reopened_conversation_appends_to_the_same_record(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await _tick()
    await user.open("/")
    await user.should_see(HR_ANSWER)
    scripted.results.append(QAResult(answer=TRAVEL_ANSWER, pages=[70], items=[HR_ITEM]))
    await _ask(user, TRAVEL_QUESTION, TRAVEL_ANSWER)
    await _tick()
    owner = _owner_id()
    saved = get_store().list_conversations(owner)
    assert len(saved) == 1 and saved[0]["message_count"] == 4
    turns = get_store().load_messages(owner, saved[0]["id"])
    assert [m["content"] for m in turns] == [HR_QUESTION, HR_ANSWER, TRAVEL_QUESTION, TRAVEL_ANSWER]


# ---------------------------------------------------------------------------
# Rename and delete
# ---------------------------------------------------------------------------
async def test_rename_uses_a_dialog_and_the_new_title_persists(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await _tick()
    user.find(marker="session-menu").click()
    user.find(marker="rename").click()
    await user.should_see("Rename conversation")
    field = user.find(marker="rename-input").elements.pop()
    field.set_value("HR phone numbers")
    user.find(marker="rename-ok").click()
    await user.should_see("Conversation renamed.")
    await user.should_not_see("Rename conversation")                        # the dialog is gone
    await user.should_see("HR phone numbers")
    await _tick()
    assert get_store().list_conversations(_owner_id())[0]["title"] == "HR phone numbers"
    await user.open("/")
    await user.should_see("HR phone numbers")


async def test_an_empty_title_is_refused_and_cancel_keeps_the_old_one(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await _tick()
    user.find(marker="session-menu").click()
    user.find(marker="rename").click()
    user.find(marker="rename-input").elements.pop().set_value("   ")
    user.find(marker="rename-ok").click()
    await user.should_see("Enter a title.")
    await user.should_see("Rename conversation")                            # still open
    user.find(marker="dialog-cancel").click()
    await user.should_not_see("Rename conversation")
    assert get_store().list_conversations(_owner_id())[0]["title"] == chat._title_for(HR_QUESTION)


async def test_delete_asks_for_confirmation_and_removes_the_conversation_for_good(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await _tick()
    user.find(marker="session-menu").click()
    user.find(marker="delete").click()
    await user.should_see("Delete this conversation?")
    user.find(marker="dialog-cancel").click()
    await user.should_not_see("Delete this conversation?")
    await user.should_see(HR_ANSWER)                                        # cancel changed nothing
    user.find(marker="session-menu").click()
    user.find(marker="delete").click()
    user.find(marker="delete-ok").click()
    await user.should_see("Conversation deleted.")
    await user.should_see("Welcome to the Faculty Onboarding Assistant")
    await user.should_not_see(HR_ANSWER)
    await _tick()
    assert get_store().list_conversations(_owner_id()) == []
    assert get_store().counts()["messages"] == 0
    await user.open("/")
    await user.should_see("No saved conversations yet.")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
async def test_settings_persist_theme_answer_language_and_voice_language_across_a_refresh(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="settings").click()
    await user.should_see("Your choices are saved for this browser")
    user.find(marker="set-theme").elements.pop().set_value("dark")
    user.find(marker="set-answer-lang").elements.pop().set_value("ar")
    user.find(marker="set-voice-lang").elements.pop().set_value("en")
    await _tick(0.3)
    prefs = get_store().get_preferences(_owner_id())
    assert prefs["theme"] == "dark" and prefs["answer_lang"] == "ar" and prefs["voice_lang"] == "en"
    user.find(marker="settings-close").click()
    await user.should_not_see("Your choices are saved for this browser")
    await user.open("/")
    state = _state()
    assert state["theme"] == "dark" and state["answer_lang"] == "ar" and state["voice_lang"] == "en"
    await _ask(user, HR_QUESTION, HR_ANSWER)
    assert scripted.answer_languages == ["ar"]                               # the saved answer language reaches the pipeline


async def test_the_interface_language_chosen_in_settings_persists_and_the_query_parameter_still_wins(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="settings").click()
    user.find(marker="set-lang").elements.pop().set_value("ar")
    await user.should_see("مساعد تهيئة أعضاء هيئة التدريس")
    await user.should_see("الإعدادات")                                      # the dialog reopened in Arabic
    await _tick(0.3)
    assert get_store().get_preferences(_owner_id())["interface_lang"] == "ar"
    await user.open("/")
    assert _state()["lang"] == "ar"
    await user.should_see("مساعد تهيئة أعضاء هيئة التدريس")
    await user.open("/?lang=en")
    assert _state()["lang"] == "en"


async def test_the_header_toggles_persist_too(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="theme").click()
    user.find(marker="lang").click()
    await user.should_see("مساعد تهيئة أعضاء هيئة التدريس")
    await _tick(0.3)
    prefs = get_store().get_preferences(_owner_id())
    assert prefs["theme"] == "dark" and prefs["interface_lang"] == "ar"


async def test_reset_returns_the_defaults_and_clears_the_saved_preferences(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="settings").click()
    user.find(marker="set-theme").elements.pop().set_value("dark")
    await _tick(0.2)
    user.find(marker="settings-reset").click()
    await user.should_see("Settings reset to their defaults.")
    await _tick(0.2)
    assert get_store().get_preferences(_owner_id()) == {}
    assert _state()["theme"] == "light"


async def test_delete_all_conversations_asks_first_and_clears_only_this_browser(create_user, scripted: Scripted) -> None:
    alice: User = create_user()
    bob: User = create_user()
    await alice.open("/")
    await _ask(alice, HR_QUESTION, HR_ANSWER)
    await bob.open("/")
    scripted.results.append(QAResult(answer=TRAVEL_ANSWER, pages=[70], items=[HR_ITEM]))
    await _ask(bob, TRAVEL_QUESTION, TRAVEL_ANSWER)
    await _tick()
    alice.find(marker="settings").click()
    alice.find(marker="clear-all").click()
    await alice.should_see("Delete all conversations?")
    alice.find(marker="clear-all-ok").click()
    await alice.should_see("All conversations were deleted.")
    await _tick()
    store = get_store()
    owners = {s["owner_id"] for s in chat._CLIENTS.values()}
    assert len(owners) == 2
    counts = sorted(len(store.list_conversations(o)) for o in owners)
    assert counts == [0, 1]                                                 # Bob's conversation is untouched


# ---------------------------------------------------------------------------
# Arabic turns
# ---------------------------------------------------------------------------
async def test_an_arabic_turn_is_saved_and_replayed_with_its_translation_state_and_english_source(user: User, scripted: Scripted) -> None:
    await user.open("/?lang=ar")
    scripted.results.append(QAResult(answer=ARABIC_ANSWER, pages=[12], items=[HR_ITEM], answer_language="ar", original_answer=ENGLISH_SOURCE,
                                     translation={"status": "translated", "reasons": []}, query_transform={"english": "What is the attendance policy?"}))
    await _ask(user, ARABIC_QUESTION, ARABIC_ANSWER)
    await _tick()
    await user.open("/?lang=ar")
    await user.should_see(ARABIC_ANSWER)
    await user.should_see(i18n.STRINGS["ar"]["translation_checked"])
    await user.should_see("إجابة محفوظة")
    turn = _state()["sessions"][0]["messages"][1]
    assert turn["result"].original_answer == ENGLISH_SOURCE and turn["result"].translation["status"] == "translated"
    assert turn["result"].query_transform == {"english": "What is the attendance policy?"}
    assert scripted.questions == [ARABIC_QUESTION]
    saved = get_store().list_conversations(_owner_id())[0]
    assert saved["title"] == ARABIC_QUESTION and saved["interface_lang"] == "ar"


# ---------------------------------------------------------------------------
# Isolation between browsers
# ---------------------------------------------------------------------------
async def test_two_browsers_never_see_each_others_conversations(create_user, scripted: Scripted) -> None:
    alice: User = create_user()
    bob: User = create_user()
    await alice.open("/")
    await _ask(alice, HR_QUESTION, HR_ANSWER)
    await _tick()
    alice_state = _state()
    alice_conversation = alice_state["active_id"]
    await bob.open("/")
    await bob.should_see("No saved conversations yet.")
    await bob.should_not_see(HR_ANSWER)
    bob_state = _state()
    assert bob_state["owner_id"] != alice_state["owner_id"]
    store = get_store()
    assert store.get_conversation(bob_state["owner_id"], alice_conversation) is None
    assert store.list_conversations(bob_state["owner_id"]) == []
    with pytest.raises(Exception):
        store.load_messages(bob_state["owner_id"], alice_conversation)


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
async def test_a_failed_save_is_reported_and_the_answer_is_still_shown(user: User, scripted: Scripted, monkeypatch, caplog) -> None:
    caplog.set_level(logging.CRITICAL, logger="ui.chat")                   # the failure IS logged; here it is the expected outcome
    await user.open("/")

    class Failing:
        def __getattr__(self, name):
            def _fail(*args, **kwargs):
                raise StoreUnavailable("disk full")
            return _fail
    monkeypatch.setattr(chat, "_store", lambda: Failing())
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await user.should_see("The conversation could not be saved.")
    await user.should_see("Not saved: this answer will be lost when the page closes.")
    assert _state()["sessions"][0]["messages"][1]["unsaved"] is True
    assert get_store().counts()["messages"] == 0                             # nothing was pretended


async def test_an_unavailable_database_shows_a_notice_and_the_page_still_answers(user: User, scripted: Scripted, monkeypatch) -> None:
    monkeypatch.setattr(chat, "_store", lambda: None)
    await user.open("/")
    await user.should_see("Saving is unavailable.")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await user.should_see("Not saved: this answer will be lost when the page closes.")
    assert _state()["owner_id"] is None


# ---------------------------------------------------------------------------
# Phone-width history
# ---------------------------------------------------------------------------
async def test_the_phone_width_history_dialog_lists_and_opens_conversations(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    user.find(marker="new-chat").click()
    await user.should_see("Welcome to the Faculty Onboarding Assistant")
    user.find(marker="history-phone").click()
    await user.should_see("Conversation history")
    assert len(user.find(marker="session").elements) == 4                   # two in the sidebar, two in the dialog
    _click_session(user, "phone number", newest=True)                       # the dialog's copy was created last
    await user.should_see(HR_ANSWER)
    await user.should_not_see("Conversation history")                       # picking a conversation closes the dialog
