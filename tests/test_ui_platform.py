"""The PF-15 web platform driven by NiceGUI's simulated user: the shared shell and navigation on every page, the Settings
page (persistence, validation, reset, data management), the Guest profile (display name, isolation), the Conversation
history page (grouping, search, open, rename, delete), the Guided procedures index and the chat additions (``?new=1``,
message times, the menu button). Temporary database per test; the pipeline is the scripted stand-in of the page tests."""
from __future__ import annotations

import asyncio
import datetime as dt

import pytest
from nicegui.elements.query import QueryElement
from nicegui.testing import User

from onboarding import catalog as C
from persistence import get_store, reset_store
from test_ui_page import HR_ANSWER, HR_ITEM, HR_QUESTION, Scripted, _settle, scripted  # noqa: F401  (fixture re-used)
from ui import chat, guest, i18n, shell

ROUTES = {"chat": "/", "onboarding": "/onboarding", "procedures": "/procedures", "resources": "/resources", "history": "/history", "profile": "/profile", "settings": "/settings"}


@pytest.fixture
def ready(scripted: Scripted) -> Scripted:
    chat._sync_onboarding_catalog()
    return scripted


async def _tick(seconds: float = 0.2) -> None:
    await asyncio.sleep(seconds)


def _owner_ids() -> set:
    conn = __import__("persistence").db.connect(get_store().path)
    try:
        return {r[0] for r in conn.execute("SELECT id FROM owners")}
    finally:
        conn.close()


def _owner() -> int:
    owners = _owner_ids()
    assert len(owners) == 1, owners
    return next(iter(owners))


def _html_props(user: User) -> dict:
    """The attributes the page put on <html> (theme, language, direction, text size, motion)."""
    merged: dict = {}
    for element in user.find(kind=QueryElement).elements:
        if element._props.get("selector") == "html":
            merged.update(element._props.get("props") or {})
    return merged


def _row_button(user: User, fragment: str, marker: str):
    """The button with ``marker`` inside the history row whose title contains ``fragment``."""
    for row in user.find(marker="hs-row").elements:
        texts = " ".join(str(getattr(el, "content", "") or "") for el in row.descendants())
        if fragment in texts:
            for el in row.descendants():
                if marker in getattr(el, "_markers", []):
                    return el
    raise AssertionError("no row with %r" % fragment)


async def _ask(user: User, question: str, answer: str) -> None:
    user.find(marker="composer").type(question)
    user.find(marker="send").click()
    await _settle(user, answer)


# ---------------------------------------------------------------------------
# Shell and navigation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(ROUTES))
async def test_every_page_has_the_shared_navigation_with_its_own_item_marked(user: User, ready: Scripted, key: str) -> None:
    await user.open(ROUTES[key])
    for item in shell.NAV_MAIN + shell.NAV_YOU:
        await user.should_see(i18n.STRINGS["en"][item[2]].replace("&", "&amp;"))
    await user.should_see('href="%s" aria-current="page"' % ROUTES[key])
    await user.should_see('aria-controls="app-nav" aria-expanded="false"')         # the drawer's menu button
    await user.should_see("Guest session · this browser")                          # the guest chip
    assert _owner_ids() == set()                                                    # a visit creates no owner


async def test_every_page_carries_the_preferences_on_the_html_element(user: User, ready: Scripted) -> None:
    await user.open("/settings")
    props = _html_props(user)
    assert props["data-theme"] == "light" and props["data-text"] == "standard" and props["data-motion"] == "full" and props["dir"] == "ltr"
    user.find(marker="pref-text-size").elements.pop().set_value("large")
    user.find(marker="pref-motion").elements.pop().set_value("on")
    await _tick(0.4)
    for route in ROUTES.values():
        await user.open(route)
        props = _html_props(user)
        assert props["data-text"] == "large" and props["data-motion"] == "reduced", route


async def test_the_navigation_is_arabic_and_right_to_left_in_the_arabic_interface(user: User, ready: Scripted) -> None:
    for route in ("/history?lang=ar", "/profile?lang=ar", "/settings?lang=ar", "/procedures?lang=ar"):
        await user.open(route)
        assert _html_props(user)["dir"] == "rtl"
        for item in shell.NAV_MAIN + shell.NAV_YOU:
            await user.should_see(i18n.STRINGS["ar"][item[2]])


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
async def test_settings_are_stored_applied_and_restored_after_a_restart(user: User, ready: Scripted) -> None:
    await user.open("/settings")
    await user.should_see("Accessibility")
    user.find(marker="pref-answer-lang").elements.pop().set_value("ar")
    user.find(marker="pref-voice-lang").elements.pop().set_value("en")
    user.find(marker="pref-theme").elements.pop().set_value("dark")
    user.find(marker="pref-text-size").elements.pop().set_value("large")
    user.find(marker="pref-motion").elements.pop().set_value("on")
    await user.should_see("Saved.")
    await _tick(0.4)
    prefs = get_store().get_preferences(_owner())
    assert (prefs["answer_lang"], prefs["voice_lang"], prefs["theme"], prefs["text_size"], prefs["reduced_motion"]) == ("ar", "en", "dark", "large", "on")
    assert _html_props(user)["data-theme"] == "dark"
    reset_store()                                                                     # the process restarts: the file is the truth
    await user.open("/settings")
    assert user.find(marker="pref-theme").elements.pop().value == "dark"
    assert user.find(marker="pref-text-size").elements.pop().value == "large"
    await user.open("/")
    state = list(chat._CLIENTS.values())[-1]
    assert state["answer_lang"] == "ar" and state["voice_lang"] == "en" and state["theme"] == "dark" and state["text_size"] == "large"
    await _ask(user, HR_QUESTION, HR_ANSWER)
    assert ready.answer_languages == ["ar"]                                           # the answer language reaches the pipeline


async def test_the_interface_language_switch_reloads_the_settings_in_arabic(user: User, ready: Scripted) -> None:
    await user.open("/settings")
    user.find(marker="pref-lang").elements.pop().set_value("ar")
    await user.should_see(i18n.STRINGS["ar"]["st_title"])
    await _tick(0.3)
    assert get_store().get_preferences(_owner())["interface_lang"] == "ar"


async def test_reset_settings_asks_first_keeps_the_display_name_and_returns_the_defaults(user: User, ready: Scripted) -> None:
    await user.open("/profile")
    user.find(marker="profile-name").elements.pop().set_value("Dr. Sara")
    user.find(marker="profile-save").click()
    await user.should_see("Display name saved.")
    await user.open("/settings")
    user.find(marker="pref-theme").elements.pop().set_value("dark")
    await _tick(0.3)
    user.find(marker="settings-page-reset").click()
    await user.should_see("Reset all settings?")
    user.find(marker="dialog-cancel").click()
    await _tick()
    assert get_store().get_preferences(_owner())["theme"] == "dark"                  # cancel changes nothing
    user.find(marker="settings-page-reset").click()
    user.find(marker="reset-ok").click()
    await _tick(0.4)
    prefs = get_store().get_preferences(_owner())
    assert "theme" not in prefs and prefs["display_name"] == "Dr. Sara"


async def test_delete_all_from_settings_asks_first_and_removes_only_this_browsers_conversations(create_user, ready: Scripted) -> None:
    alice: User = create_user()
    bob: User = create_user()
    await alice.open("/")
    await _ask(alice, HR_QUESTION, HR_ANSWER)
    await bob.open("/")
    await _ask(bob, HR_QUESTION, HR_ANSWER)
    await _tick()
    await alice.open("/settings")
    await alice.should_see("Saved conversations: 1")
    alice.find(marker="settings-page-clear").click()
    await alice.should_see("Delete all conversations?")
    alice.find(marker="clear-all-ok").click()
    await alice.should_see("All conversations were deleted.")
    await alice.should_see("Saved conversations: 0")
    store = get_store()
    assert sorted(len(store.list_conversations(o)) for o in _owner_ids()) == [0, 1]


# ---------------------------------------------------------------------------
# Guest profile
# ---------------------------------------------------------------------------
async def test_the_display_name_is_validated_stored_shown_and_restored(user: User, ready: Scripted) -> None:
    await user.open("/profile")
    await user.should_see("Nothing has been saved for this browser yet.")
    await user.should_see("not a University of Sharjah login")
    field = user.find(marker="profile-name").elements.pop()
    for bad in ("someone@sharjah.ac.ae", "ID 784123456", "<script>x</script>"):
        field.set_value(bad)
        user.find(marker="profile-save").click()
        await user.should_see("Email addresses, long numbers")
    assert _owner_ids() == set()                                                      # nothing invalid was stored (no row either)
    field.set_value("  Dr.   Sara  ")
    user.find(marker="profile-save").click()
    await user.should_see("Display name saved.")
    await _tick()
    assert get_store().get_preferences(_owner())["display_name"] == "Dr. Sara"
    await user.should_see("Guest profile: Dr. Sara")                                  # the sidebar chip follows
    reset_store()
    await user.open("/onboarding")
    await user.should_see("Dr. Sara")                                                 # restored after a restart, on every page
    await user.open("/profile")
    assert user.find(marker="profile-name").elements.pop().value == "Dr. Sara"
    user.find(marker="profile-remove").click()
    await user.should_see("Display name removed.")
    await _tick()
    assert "display_name" not in get_store().get_preferences(_owner())


async def test_the_profile_shows_this_guests_progress_and_activity_only(create_user, ready: Scripted) -> None:
    alice: User = create_user()
    bob: User = create_user()
    await alice.open("/onboarding")
    alice.find(marker="complete-hr_documents").click()
    await alice.should_see("Saved:")
    await alice.open("/profile")
    alice.find(marker="profile-name").elements.pop().set_value("Alice")
    alice.find(marker="profile-save").click()
    await alice.should_see("Display name saved.")
    await alice.open("/profile")
    await alice.should_see("Complete the pre-joining document formalities with HR: Completed")
    await alice.should_see("Saving since")
    await bob.open("/profile")
    await bob.should_not_see("Alice")
    await bob.should_see("No activity yet.")
    await bob.should_see("Nothing has been saved for this browser yet.")


# ---------------------------------------------------------------------------
# Conversation history page
# ---------------------------------------------------------------------------
async def test_the_history_page_lists_searches_renames_deletes_and_opens(user: User, ready: Scripted) -> None:
    await user.open("/history")
    await user.should_see("No saved conversations yet.")
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    user.find(marker="new-chat").click()
    await _tick()
    ready.results.append(chat.QAResult(answer="The Dean approves conference travel requests.", pages=[70], items=[HR_ITEM]))
    await _ask(user, "Who approves conference travel requests?", "The Dean approves conference travel requests.")
    await _tick()
    await user.open("/history")
    await user.should_see("Last 24 hours")
    await user.should_see("2 saved")
    assert len(user.find(marker="hs-row").elements) == 2
    await user.should_see('data-local datetime="')                                   # times rendered in the reader's local time
    user.find(marker="history-search").elements.pop().set_value("travel")
    await _tick(0.4)
    assert len(user.find(marker="hs-row").elements) == 1
    user.find(marker="history-search").elements.pop().set_value("parking")
    await _tick(0.4)
    await user.should_see("No conversation title matches")
    user.find(marker="history-search").elements.pop().set_value("")
    await _tick(0.4)
    from nicegui.testing.user_interaction import UserInteraction
    UserInteraction(user, {_row_button(user, "phone number", "hs-rename")}, None).click()
    await user.should_see("Rename conversation")
    user.find(marker="rename-input").elements.pop().set_value("HR phone numbers")
    user.find(marker="rename-ok").click()
    await user.should_see("Conversation renamed.")
    await user.should_see("HR phone numbers")
    owner = _owner()
    assert {c["title"] for c in get_store().list_conversations(owner)} == {"HR phone numbers", "Who approves conference travel requests?"}
    UserInteraction(user, {_row_button(user, "travel", "hs-delete")}, None).click()
    await user.should_see("Delete this conversation?")
    user.find(marker="delete-ok").click()
    await user.should_see("Conversation deleted.")
    assert len(get_store().list_conversations(owner)) == 1
    asked = list(ready.questions)
    user.find(marker="hs-open").click()                                               # open the remaining one
    await user.should_see(HR_ANSWER)                                                  # the chat replays it
    await user.should_see("Saved answer")
    assert ready.questions == asked                                                   # nothing asked again
    assert get_store().get_preferences(owner)["extra"]["active_conversation"] == get_store().list_conversations(owner)[0]["id"]


def test_history_groups_by_elapsed_time() -> None:
    now = dt.datetime(2026, 11, 9, 12, 0, 0)
    assert guest.age_group("2026-11-09T01:00:00.000Z", now) == "day"
    assert guest.age_group("2026-11-05T12:00:00Z", now) == "week"
    assert guest.age_group("2026-10-20T12:00:00Z", now) == "month"
    assert guest.age_group("2025-01-01T00:00:00Z", now) == "older"
    assert guest.age_group("garbage", now) == "older"


# ---------------------------------------------------------------------------
# Guided procedures index
# ---------------------------------------------------------------------------
async def test_the_procedures_index_lists_every_procedure_with_task_and_position(user: User, ready: Scripted) -> None:
    await user.open("/procedures")
    assert len(user.find(marker="procedure-card").elements) == len(C.PROCEDURES)
    await user.should_see('href="/onboarding/procedure/bb_announcement?task=bb_announcement"')
    await user.should_see('href="/onboarding/procedure/grade_transfer_access"')      # reachable now, though no task links to it
    await user.should_see("Not linked to a checklist task")
    await user.should_see("8 steps")
    await user.open("/onboarding/procedure/bb_announcement?task=bb_announcement")
    user.find(marker="next").click()
    user.find(marker="next").click()
    await _tick(0.4)
    await user.open("/procedures")
    await user.should_see("Continue at step 3 of 8")


# ---------------------------------------------------------------------------
# Chat additions
# ---------------------------------------------------------------------------
async def test_new_chat_link_starts_an_empty_conversation_and_messages_carry_times(user: User, ready: Scripted) -> None:
    await user.open("/")
    await _ask(user, HR_QUESTION, HR_ANSWER)
    await user.should_see('class="msg-time" data-local datetime="')                   # the user's message and the answer
    await _tick()
    await user.open("/?new=1")
    await user.should_see("Welcome to the Faculty Onboarding Assistant")
    await user.should_not_see(HR_ANSWER)
    await user.should_see("View all conversations")
    await user.should_see('aria-controls="app-nav"')


async def test_the_quick_settings_dialog_links_to_the_settings_page(user: User, ready: Scripted) -> None:
    await user.open("/")
    user.find(marker="settings").click()
    await user.should_see("All settings")
    await user.should_see('href="/settings"')


def test_phone_numbers_in_answers_are_kept_whole_in_the_browser_without_changing_the_server_html() -> None:
    # The answer HTML is left exactly as PF-14 rendered it (tests find answer text in it); the shell script wraps a printed
    # number in a left-to-right, non-breaking span in the browser (checked in Chrome by the platform harness, step A3b).
    from ui import rendering, styles
    out = rendering.answer_html("Human Resources +971 6 5050023 +971 6 5585200")
    assert "+971 6 5050023 +971 6 5585200" in out and 'class="tel"' not in out
    assert "window.fobTel" in shell.SHELL_JS and "'.answer, .original-body'" in shell.SHELL_JS and "closest('code, pre, .tel')" in shell.SHELL_JS
    assert ".answer .tel, .original-body .tel { white-space: nowrap; unicode-bidi: isolate;" in styles.GLOBAL_CSS
