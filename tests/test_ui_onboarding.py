"""The onboarding pages driven by NiceGUI's simulated user (post-freeze PF-14): the checklist dashboard, the guided
procedure navigator, the resource finder and their integration with the chat page and the PF-13 store. Temporary
database per test; the pipeline is the scripted stand-in of the page tests."""
from __future__ import annotations

import asyncio

import pytest
from nicegui.testing import User

from onboarding import catalog as C
from persistence import get_store, reset_store
from test_ui_page import HR_ANSWER, HR_ITEM, HR_QUESTION, Scripted, _settle, scripted  # noqa: F401  (fixture re-used)
from ui import chat, i18n

ANNOUNCEMENT = C.PROCEDURES["bb_announcement"]


@pytest.fixture
def ready(scripted: Scripted) -> Scripted:
    chat._sync_onboarding_catalog()
    return scripted


def _state() -> dict:
    return list(chat._CLIENTS.values())[-1]


async def _tick(seconds: float = 0.2) -> None:
    await asyncio.sleep(seconds)


def _owner_ids() -> set:
    conn = __import__("persistence").db.connect(get_store().path)
    try:
        return {r[0] for r in conn.execute("SELECT id FROM owners")}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
async def test_the_dashboard_lists_every_catalog_task_with_its_evidence_and_zero_progress(user: User, ready: Scripted) -> None:
    await user.open("/onboarding")
    await user.should_see(i18n.STRINGS["en"]["ob_title"])
    cards = user.find(marker="task").elements
    assert len(cards) == len(C.TASKS)
    await user.should_see("Required by the handbook")
    await user.should_see(C.TASKS["ilhe_training"].evidence[0].quote)                 # the exact quote is on the page
    await user.should_see("UOS Faculty Handbook 2025/26")
    await user.should_see("0%")
    assert _owner_ids() == set()                                                      # a visit creates no owner


async def test_the_category_filter_narrows_the_cards(user: User, ready: Scripted) -> None:
    await user.open("/onboarding")
    user.find(marker="filter-research").click()
    await _tick()
    assert len(user.find(marker="task").elements) == len(C.tasks_by_category("research"))
    user.find(marker="filter-all").click()
    await _tick()
    assert len(user.find(marker="task").elements) == len(C.TASKS)


async def test_marking_a_task_complete_persists_updates_the_summary_and_is_idempotent(user: User, ready: Scripted) -> None:
    await user.open("/onboarding")
    user.find(marker="complete-hr_documents").click()
    await user.should_see("Saved: Complete the pre-joining document formalities with HR is now Completed.")
    await _tick()
    store = get_store()
    owner = next(iter(_owner_ids()))
    assert store.task_progress(owner)["hr_documents"]["status"] == "completed"
    assert len(store.progress_events(owner)) == 1
    await user.should_see("1")                                                        # completed count
    with pytest.raises(AssertionError):
        user.find(marker="complete-hr_documents")                                     # the button is gone once completed
    select = user.find(marker="status-hr_documents").elements.pop()
    assert select.value == "completed"
    select.set_value("completed")                                                     # a repeated identical update
    await _tick()
    assert len(store.progress_events(owner)) == 1
    await user.open("/onboarding")                                                    # reload: the status comes back
    assert user.find(marker="status-hr_documents").elements.pop().value == "completed"
    reset_store()                                                                     # the process restarts: the file is the truth
    await user.open("/onboarding")
    assert user.find(marker="status-hr_documents").elements.pop().value == "completed"


async def test_not_applicable_tasks_leave_the_applicable_count_and_zero_applicable_is_handled(user: User, ready: Scripted) -> None:
    await user.open("/onboarding")
    user.find(marker="status-grade_transfer").elements.pop().set_value("not_applicable")
    await _tick()
    owner = next(iter(_owner_ids()))
    store = get_store()
    assert store.task_progress(owner)["grade_transfer"]["status"] == "not_applicable"
    for t in C.TASKS.values():
        store.set_task_status(owner, t.id, "not_applicable")
    await user.open("/onboarding")
    await user.should_see("No applicable tasks: every task is marked not applicable.")
    await user.should_see("0%")


async def test_progress_is_isolated_between_two_browsers(create_user, ready: Scripted) -> None:
    alice: User = create_user()
    bob: User = create_user()
    await alice.open("/onboarding")
    alice.find(marker="complete-bb_access").click()
    await alice.should_see("Saved:")
    await bob.open("/onboarding")
    assert bob.find(marker="status-bb_access").elements.pop().value == "not_started"
    await bob.should_see("0%")
    store = get_store()
    owners = _owner_ids()
    assert len(owners) == 1                                                           # Bob never wrote, so Bob has no row
    assert store.task_progress(next(iter(owners)))["bb_access"]["status"] == "completed"


async def test_the_dashboard_in_arabic_is_rtl_with_arabic_titles_and_the_same_statuses(user: User, ready: Scripted) -> None:
    await user.open("/onboarding")
    user.find(marker="complete-ilhe_training").click()
    await user.should_see("Saved:")
    await user.open("/onboarding?lang=ar")
    await user.should_see(i18n.STRINGS["ar"]["ob_title"])
    await user.should_see(C.TASKS["ilhe_training"].title_ar)
    assert user.find(marker="status-ilhe_training").elements.pop().value == "completed"   # the same stored status
    await user.should_see("مطلوب بحسب الدليل")


async def test_the_language_toggle_on_the_dashboard_saves_the_preference(user: User, ready: Scripted) -> None:
    await user.open("/onboarding")
    user.find(marker="lang").click()
    await user.should_see(i18n.STRINGS["ar"]["ob_title"])
    await _tick()
    owner = next(iter(_owner_ids()))
    assert get_store().get_preferences(owner)["interface_lang"] == "ar"
    await user.open("/")                                                              # the chat page follows the saved language
    assert _state()["lang"] == "ar"


# ---------------------------------------------------------------------------
# Procedure navigator
# ---------------------------------------------------------------------------
async def test_the_navigator_shows_the_steps_in_order_with_their_flags(user: User, ready: Scripted) -> None:
    await user.open("/onboarding/procedure/bb_announcement?task=bb_announcement")
    await user.should_see("Step 1 of 8")
    await user.should_see(ANNOUNCEMENT.steps[0].text)
    await user.should_see("Blackboard Ultra Faculty Short Guide")
    assert _owner_ids() == set()                                                      # opening the guide creates no row
    for n in range(1, 4):
        user.find(marker="next").click()
        await user.should_see("Step %d of 8" % (n + 1))
    await user.should_see(ANNOUNCEMENT.steps[3].text)
    await user.should_see("Conditional / optional")
    user.find(marker="prev").click()
    await user.should_see("Step 3 of 8")
    await user.should_not_see("Conditional / optional")
    for _ in range(5):
        user.find(marker="next").click()
        await _tick(0.05)
    await user.should_see("Step 8 of 8")
    await user.should_see("After the procedure")
    await _tick(0.3)
    owner = next(iter(_owner_ids()))                                                  # moving through the steps is an interaction: remembered
    assert get_store().procedure_position(owner, "bb_announcement") == 7
    assert get_store().task_progress(owner) == {}                                     # but it completes nothing


async def test_the_position_is_remembered_once_the_browser_has_an_owner_and_completion_is_explicit(user: User, ready: Scripted) -> None:
    await user.open("/onboarding/procedure/bb_access?task=bb_access")
    user.find(marker="next").click()
    user.find(marker="next").click()
    await user.should_see("Step 3 of 7")
    await _tick(0.3)
    owner = next(iter(_owner_ids()))
    store = get_store()
    assert store.procedure_position(owner, "bb_access") == 2
    assert store.task_progress(owner).get("bb_access") is None                        # moving through steps completed nothing
    await user.open("/onboarding/procedure/bb_access?task=bb_access")
    await user.should_see("Step 3 of 7")                                              # the place is kept
    user.find(marker="mark-done").click()
    await user.should_see("Task marked completed")
    await _tick()
    assert store.task_progress(owner)["bb_access"]["status"] == "completed"
    assert len(store.progress_events(owner)) == 1


async def test_a_forged_task_pairing_and_an_unknown_procedure_are_handled(user: User, ready: Scripted) -> None:
    await user.open("/onboarding/procedure/bb_access?task=hr_documents")
    await user.should_see("Step 1 of 7")
    await user.should_not_see("Checklist task")                                       # the task is not linked to this guide
    with pytest.raises(AssertionError):
        user.find(marker="mark-done")
    await user.open("/onboarding/procedure/no_such_guide")
    await user.should_see("This guide does not exist.")


async def test_ask_about_this_step_opens_the_chat_with_the_question_prefilled_and_the_task_as_context(user: User, ready: Scripted) -> None:
    await user.open("/onboarding/procedure/bb_announcement?task=bb_announcement")
    user.find(marker="ask-step").click()
    await user.should_see("Asking about:")
    await user.should_see(C.TASKS["bb_announcement"].title_en)
    question = user.find(marker="composer").elements.pop().value
    assert question.startswith('In the "Send an Announcement" guide, what does this step mean:')
    assert ANNOUNCEMENT.steps[0].text in question
    assert ready.questions == []                                                      # pre-filled, not sent
    user.find(marker="send").click()                                                  # the user decides to send it
    await _settle(user, HR_ANSWER)
    await _tick()
    assert ready.questions == [question]
    owner = next(iter(_owner_ids()))
    saved = get_store().list_conversations(owner)
    assert saved[0]["context_task"] == "bb_announcement"
    assert get_store().task_progress(owner) == {}                                    # the answer changed no progress
    await user.open("/")                                                              # reopened: the chip comes back from the row
    await user.should_see("Asking about:")


# ---------------------------------------------------------------------------
# Finder
# ---------------------------------------------------------------------------
async def test_the_finder_lists_the_catalog_and_searches_in_english(user: User, ready: Scripted) -> None:
    await user.open("/resources")
    assert len(user.find(marker="resource").elements) == len(C.RESOURCES)
    await user.should_see("%d results" % len(C.RESOURCES))
    user.find(marker="finder-search").elements.pop().set_value("Human Resources")
    await _tick()
    cards = user.find(marker="resource").elements
    assert cards and "+971 6 5050023" in str(sorted(cards, key=lambda e: e.id)[0].default_slot.children[3].content)
    await user.should_see("As printed in the Faculty Handbook, p. 54")
    user.find(marker="finder-search").elements.pop().set_value("parking permit")
    await _tick()
    await user.should_see('No resource matches')
    user.find(marker="ask-instead")                                                   # the empty state offers the assistant


async def test_the_finder_searches_in_arabic_and_shows_arabic_titles(user: User, ready: Scripted) -> None:
    await user.open("/resources?lang=ar")
    await user.should_see(i18n.STRINGS["ar"]["rf_title"])
    user.find(marker="finder-search").elements.pop().set_value("دليل أعضاء هيئة التدريس")
    await _tick()
    cards = sorted(user.find(marker="resource").elements, key=lambda e: e.id)
    assert cards and "دليل أعضاء هيئة التدريس 2025/2026" in str(cards[0].default_slot.children[1].content)


async def test_the_finder_shows_task_context_first_and_only_safe_verified_links(user: User, ready: Scripted) -> None:
    await user.open("/resources?task=bb_access")
    await user.should_see("Resources for:")
    await user.should_see(C.TASKS["bb_access"].title_en)
    cards = sorted(user.find(marker="resource").elements, key=lambda e: e.id)
    assert "contextual" in cards[0]._classes
    await user.should_see("https://www.sharjah.ac.ae/en/")
    await user.should_see('rel="noopener noreferrer"')
    await user.should_not_see("Performance Appraisal")                                # the held source never appears
    user.find(marker="clear-context").click()
    await _tick()
    await user.should_not_see("Resources for:")


async def test_the_finder_category_filter_and_the_department_without_a_printed_contact(user: User, ready: Scripted) -> None:
    await user.open("/resources")
    user.find(marker="rfilter-research").click()
    await _tick()
    assert len(user.find(marker="resource").elements) == len([r for r in C.RESOURCES.values() if r.category == "research"])
    await user.should_see("The approved documents print no telephone number or email for this department.")


# ---------------------------------------------------------------------------
# Chat page integration
# ---------------------------------------------------------------------------
async def test_the_chat_page_accepts_a_task_question_and_links_back_to_the_task(user: User, ready: Scripted) -> None:
    await user.open("/?task=ilhe_training&ask=Who runs the new faculty training?")
    await user.should_see("Asking about:")
    await user.should_see("Open the task")
    assert user.find(marker="composer").elements.pop().value == "Who runs the new faculty training?"
    await user.open("/?task=not_a_task&ask=hello")
    await user.should_not_see("Asking about:")                                        # an unknown task is ignored
    assert user.find(marker="composer").elements.pop().value == "hello"


async def test_the_chat_sidebar_links_to_the_onboarding_pages(user: User, ready: Scripted) -> None:
    await user.open("/")
    await user.should_see("Onboarding checklist")
    await user.should_see("Services &amp; documents")
