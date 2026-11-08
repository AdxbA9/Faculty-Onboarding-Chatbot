"""The microphone control on the chat page, driven by NiceGUI's simulated user (post-freeze PF-12).

The simulated user runs no browser JavaScript, so the recogniser is replaced by the page's own event entry point
(``state["voice_apply"]``), fed with the exact event shapes the browser controller emits. What these tests establish is the
page's behaviour (states, labels, transcript handling, cleanup, submission); what they cannot establish is speech
recognition itself, which the browser harness (mocked recogniser) and the human acceptance test cover.
"""
from __future__ import annotations

import pytest
from nicegui.testing import User

from handbook_bot.qa import QAResult
from test_ui_page import HR_ANSWER, HR_ITEM, HR_QUESTION, Scripted, _settle, scripted  # noqa: F401  (fixture re-used)
from ui import chat, i18n, voice

AR_QUESTION = "ما هو رقم هاتف قسم الموارد البشرية؟"


def _state() -> dict:
    assert len(chat._CLIENTS) == 1
    return next(iter(chat._CLIENTS.values()))


def _mic(user: User):
    return user.find(marker="mic").elements.pop()


async def _speak(user: User, text: str, lang: str = "en-US") -> None:
    apply = _state()["voice_apply"]
    apply({"kind": "starting", "lang": lang})
    apply({"kind": "listening", "lang": lang})
    apply({"kind": "interim", "text": text[:8], "lang": lang})
    apply({"kind": "final", "text": text, "lang": lang})
    apply({"kind": "end", "lang": lang})


async def test_the_microphone_is_present_labelled_and_idle(user: User, scripted: Scripted) -> None:
    await user.open("/")
    mic = _mic(user)
    assert "Ask by voice" in mic._props.get("aria-label", "") and mic._props.get("aria-pressed") == "false"
    assert "voice-idle" in mic._classes
    assert user.find(marker="voice-lang").elements
    assert _state()["voice_lang"] == "auto" and _state()["voice"].state == voice.IDLE


async def test_listening_shows_a_stop_control_and_the_live_strip(user: User, scripted: Scripted) -> None:
    await user.open("/")
    apply = _state()["voice_apply"]
    apply({"kind": "starting", "lang": "en-US"})
    apply({"kind": "listening", "lang": "en-US"})
    mic = _mic(user)
    assert mic._props.get("aria-pressed") == "true" and mic._props.get("icon") == "stop" and "voice-listening" in mic._classes
    assert "Stop listening" in mic._props.get("aria-label", "")
    await user.should_see("Listening")
    apply({"kind": "interim", "text": "what is the", "lang": "en-US"})
    await user.should_see("what is the")
    assert "hidden" not in user.find(marker="voice-cancel").elements.pop()._classes


async def test_a_final_transcript_lands_in_the_composer_and_is_not_sent(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _speak(user, HR_QUESTION)
    field = user.find(marker="composer").elements.pop()
    assert field.value == HR_QUESTION
    await user.should_see("Transcript added to the message box")
    assert scripted.questions == []                                       # nothing was sent by itself
    mic = _mic(user)
    assert "voice-ready" in mic._classes and mic._props.get("aria-pressed") == "false" and mic._props.get("icon") == "mic"
    user.find(marker="send").click()
    await _settle(user, HR_ANSWER)
    assert scripted.questions == [HR_QUESTION]                            # the same handler as a typed question


async def test_a_transcript_is_appended_to_an_existing_draft(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="composer").type("Please tell me:")
    await _speak(user, "what is the teaching load for lecturers?")
    assert user.find(marker="composer").elements.pop().value == "Please tell me: what is the teaching load for lecturers?"


async def test_the_user_can_edit_the_transcript_before_sending(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _speak(user, "What is the phone number of the HR")
    user.find(marker="composer").type(" Department?")
    user.find(marker="send").click()
    await _settle(user, HR_ANSWER)
    assert scripted.questions == ["What is the phone number of the HR Department?"]


async def test_an_arabic_transcript_in_the_arabic_interface(user: User, scripted: Scripted) -> None:
    scripted.results.append(QAResult(answer=HR_ANSWER, pages=[54], items=[HR_ITEM], answer_language="ar", original_answer=HR_ANSWER,
                                     translation={"status": "framed", "llm_calls": 0, "problems": [], "reason": "no_client"}))
    await user.open("/?lang=ar")
    mic = _mic(user)
    assert "اسأل بالصوت" in mic._props.get("aria-label", "") and "العربية" in mic._props.get("aria-label", "")
    await _speak(user, AR_QUESTION, lang="ar-AE")
    assert user.find(marker="composer").elements.pop().value == AR_QUESTION
    await user.should_see("أُضيف النص المنطوق إلى مربع الرسالة")
    user.find(marker="send").click()
    await _settle(user, HR_ANSWER)
    assert scripted.questions == [AR_QUESTION] and scripted.answer_languages == ["auto"]


async def test_a_mixed_language_transcript_is_accepted_as_text(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await _speak(user, "كيف أستخدم Blackboard لإرسال Announcement", lang="ar-AE")
    assert user.find(marker="composer").elements.pop().value == "كيف أستخدم Blackboard لإرسال Announcement"


@pytest.mark.parametrize("code, fragment_en", [
    ("not-allowed", "Microphone access was not allowed"), ("audio-capture", "No microphone was found"), ("no-speech", "No speech was heard"),
    ("network", "speech service could not be reached"), ("timeout", "within the time limit"), ("language-not-supported", "not available in the browser"),
])
async def test_recogniser_errors_become_plain_messages_and_leave_typing_available(user: User, scripted: Scripted, code: str, fragment_en: str) -> None:
    await user.open("/")
    apply = _state()["voice_apply"]
    apply({"kind": "starting"})
    apply({"kind": "listening"})
    apply({"kind": "error", "code": code})
    apply({"kind": "end"})
    await user.should_see(fragment_en)
    mic = _mic(user)
    assert "voice-error" in mic._classes and mic._props.get("aria-pressed") == "false"
    user.find(marker="composer").type(HR_QUESTION)
    user.find(marker="send").click()
    await _settle(user, HR_ANSWER)                                        # typing always works


async def test_an_unsupported_browser_gets_an_honest_state(user: User, scripted: Scripted) -> None:
    await user.open("/?lang=ar")
    _state()["voice_apply"]({"kind": "unsupported"})
    await user.should_see("الإدخال الصوتي غير متاح في هذا المتصفح")
    mic = _mic(user)
    assert "voice-unsupported" in mic._classes and mic._props.get("icon") == "mic_off"


async def test_no_speech_end_without_results_and_cancellation(user: User, scripted: Scripted) -> None:
    await user.open("/")
    apply = _state()["voice_apply"]
    apply({"kind": "starting"})
    apply({"kind": "listening"})
    apply({"kind": "end"})
    await user.should_see("No speech was heard")
    apply({"kind": "starting"})
    apply({"kind": "listening"})
    apply({"kind": "interim", "text": "partial words"})
    apply({"kind": "cancelled"})
    await user.should_see("Voice input cancelled")
    assert user.find(marker="composer").elements.pop().value in ("", None)    # a cancelled recognition sends and keeps nothing
    assert scripted.questions == []


async def test_a_duplicate_start_changes_nothing(user: User, scripted: Scripted) -> None:
    await user.open("/")
    apply = _state()["voice_apply"]
    apply({"kind": "starting"})
    apply({"kind": "listening"})
    apply({"kind": "busy"})
    assert _state()["voice"].state == voice.LISTENING and _mic(user)._props.get("aria-pressed") == "true"


async def test_new_chat_and_a_language_switch_cancel_a_running_recognition(user: User, scripted: Scripted, monkeypatch) -> None:
    await user.open("/")
    calls = []
    client = _state()
    monkeypatch.setattr(chat, "_safe", chat._safe)
    apply = client["voice_apply"]
    apply({"kind": "starting"})
    apply({"kind": "listening"})
    assert client["voice"].active
    user.find(marker="composer").type(HR_QUESTION)
    user.find(marker="send").click()
    await _settle(user, HR_ANSWER)
    user.find(marker="new-chat").click()                                   # asks the browser to cancel; the browser answers with 'cancelled'
    apply({"kind": "cancelled"})
    assert not client["voice"].active and client["voice"].state == voice.IDLE
    apply({"kind": "starting"})
    apply({"kind": "listening"})
    user.find(marker="lang").click()                                       # the shell is rebuilt; the new mic starts from the machine's state
    apply({"kind": "cancelled"})
    await user.should_see("تم إلغاء الإدخال الصوتي")
    assert "voice-idle" in _mic(user)._classes


async def test_the_voice_language_choice_changes_the_label_and_the_locale(user: User, scripted: Scripted) -> None:
    await user.open("/")
    user.find(marker="voice-lang").elements.pop().set_value("ar")
    assert _state()["voice_lang"] == "ar"
    assert "Arabic" in _mic(user)._props.get("aria-label", "")
    assert voice.locale_for(_state()["voice_lang"], "en") == "ar-AE"


async def test_the_privacy_note_appears_on_first_use_only(user: User, scripted: Scripted) -> None:
    await user.open("/")
    await user.should_not_see("speech service")
    _state()["voice_start"]()
    await user.should_see("your browser's speech service")
    assert _state()["voice_privacy_shown"] is True


async def test_the_browser_controller_is_part_of_the_page(user: User, scripted: Scripted) -> None:
    await user.open("/")
    assert "window.fobVoice" in voice.VOICE_JS and "fob_voice" in voice.VOICE_JS
    assert i18n.STRINGS["en"]["voice_privacy"] in i18n.t("en", "voice_privacy")
