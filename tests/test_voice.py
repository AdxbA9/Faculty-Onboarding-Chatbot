"""The voice-input state machine, transcript policy and browser controller source (post-freeze PF-12). Pure Python."""
from __future__ import annotations

import pytest

from ui import i18n, voice
from ui.voice import VoiceMachine, locale_for, merge_draft


def _drive(*events):
    vm = VoiceMachine()
    keys = [vm.apply(e) for e in events]
    return vm, keys


def test_locales_follow_the_choice_then_the_interface_language():
    assert locale_for("ar", "en") == "ar-AE" and locale_for("en", "ar") == "en-US"
    assert locale_for("auto", "ar") == "ar-AE" and locale_for(None, "en") == "en-US" and locale_for("xx", "xx") == "en-US"


def test_a_transcript_is_appended_to_the_draft_never_replacing_it():
    assert merge_draft("", "ما هو العبء التدريسي") == "ما هو العبء التدريسي"
    assert merge_draft("Please tell me", "the teaching load") == "Please tell me the teaching load"
    assert merge_draft("Please tell me ", "  the   teaching load ") == "Please tell me the teaching load"
    assert merge_draft("draft text", "") == "draft text" and merge_draft("", "   ") == ""


def test_idle_to_listening_to_ready_with_interim_and_final_results():
    vm, keys = _drive({"kind": "starting", "lang": "en-US"}, {"kind": "listening"}, {"kind": "interim", "text": "what is"},
                      {"kind": "final", "text": "What is the teaching load for lecturers?"}, {"kind": "end"})
    assert keys == ["voice_starting", "voice_listening", None, None, "voice_ready"]
    assert vm.state == voice.READY and vm.final == "What is the teaching load for lecturers?" and vm.interim == "" and not vm.active


def test_listening_to_cancelled_and_aborted():
    vm, keys = _drive({"kind": "starting"}, {"kind": "listening"}, {"kind": "interim", "text": "half"}, {"kind": "cancelled"})
    assert keys[-1] == "voice_cancelled" and vm.state == voice.IDLE and vm.interim == ""
    vm, keys = _drive({"kind": "starting"}, {"kind": "listening"}, {"kind": "error", "code": "aborted"})
    assert keys[-1] == "voice_cancelled" and vm.state == voice.IDLE


@pytest.mark.parametrize("code, key", [
    ("not-allowed", "voice_err_permission"), ("service-not-allowed", "voice_err_permission"), ("audio-capture", "voice_err_no_mic"),
    ("no-speech", "voice_err_no_speech"), ("network", "voice_err_network"), ("timeout", "voice_err_timeout"),
    ("language-not-supported", "voice_err_language"), ("bad-grammar", "voice_err_unrecognised"), ("something-new", "voice_err_unrecognised"),
])
def test_every_recogniser_error_has_a_localised_message_in_both_languages(code, key):
    vm, keys = _drive({"kind": "starting"}, {"kind": "listening"}, {"kind": "error", "code": code}, {"kind": "end"})
    assert keys[2] == key and keys[3] is None and vm.state == voice.ERROR and vm.error == code
    assert i18n.STRINGS["en"][key] and i18n.STRINGS["ar"][key]


def test_an_end_without_any_result_is_reported_as_no_speech():
    vm, keys = _drive({"kind": "starting"}, {"kind": "listening"}, {"kind": "end"})
    assert keys[-1] == "voice_err_no_speech" and vm.state == voice.ERROR and not vm.had_final


def test_unsupported_browser_and_duplicate_start():
    vm, keys = _drive({"kind": "unsupported"})
    assert keys == ["voice_unsupported"] and vm.state == voice.UNSUPPORTED
    vm, keys = _drive({"kind": "starting"}, {"kind": "listening"}, {"kind": "busy"})
    assert keys[-1] is None and vm.state == voice.LISTENING and vm.active


def test_a_new_recognition_resets_the_previous_transcript_and_error():
    vm, _ = _drive({"kind": "starting"}, {"kind": "error", "code": "no-speech"}, {"kind": "end"}, {"kind": "starting", "lang": "ar-AE"})
    assert vm.state == voice.STARTING and vm.error is None and vm.final == "" and vm.lang == "ar-AE"


def test_interim_results_after_an_error_are_ignored():
    vm, _ = _drive({"kind": "starting"}, {"kind": "error", "code": "network"}, {"kind": "interim", "text": "late"})
    assert vm.state == voice.ERROR and vm.interim == ""


def test_two_final_segments_are_joined_in_order():
    vm, _ = _drive({"kind": "starting"}, {"kind": "listening"}, {"kind": "final", "text": "ما هو"}, {"kind": "final", "text": "رقم الهاتف؟"}, {"kind": "end"})
    assert vm.final == "ما هو رقم الهاتف؟" and vm.state == voice.READY


def test_the_browser_controller_guards_the_recogniser():
    js = voice.VOICE_JS
    assert "window.SpeechRecognition || window.webkitSpeechRecognition" in js
    assert "kind: 'unsupported'" in js and "kind: 'busy'" in js                      # honest fallback and the duplicate-start guard
    assert "interimResults = true" in js and "continuous = false" in js and "maxAlternatives = 1" in js
    assert "pagehide" in js and "beforeunload" in js and "rec.abort()" in js            # no recording outlives the page
    assert "'timeout'" in js and "noResultMs" in js and "capMs" in js                   # no-result timeout and the hard cap
    assert "emitEvent('fob_voice'" in js
    assert "localStorage" not in js and "fetch(" not in js and "XMLHttpRequest" not in js   # nothing stored, nothing sent by the page itself


def test_the_voice_strings_exist_in_both_languages():
    keys = [k for k in i18n.STRINGS["en"] if k.startswith("voice_")]
    assert len(keys) >= 20 and all(k in i18n.STRINGS["ar"] for k in keys)
    assert "{voice}" in i18n.STRINGS["en"]["voice_start_aria"] and "{voice}" in i18n.STRINGS["ar"]["voice_start_aria"]
    assert "servers" in i18n.STRINGS["en"]["voice_privacy"].lower()                       # the privacy limitation is stated
