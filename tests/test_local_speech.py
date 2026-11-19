"""Local speech-to-text for the microphone button (faster-whisper, ui.local_speech) and its browser controller.

The transcription tests synthesise a spoken question with the Windows speech synthesiser when it is available, encode it
the way Chrome's MediaRecorder does (WebM / Opus) and send it through the same function the web endpoint calls. They are
skipped where faster-whisper, the base.en model or a speech synthesiser is missing.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from ui import chat, i18n, local_speech, voice


# ---------------------------------------------------------------- configuration and selection -----------------------------
def test_engine_is_local_when_installed_and_browser_when_switched_off(monkeypatch):
    monkeypatch.setattr(local_speech, "installed", lambda: True)
    monkeypatch.delenv("VOICE_ENGINE", raising=False)
    assert local_speech.engine() == "local"
    monkeypatch.setenv("VOICE_ENGINE", "browser")
    assert local_speech.engine() == "browser"
    monkeypatch.setenv("VOICE_ENGINE", "local")
    monkeypatch.setattr(local_speech, "installed", lambda: False)
    assert local_speech.engine() == "browser"              # not installed: the Web Speech controller is used


def test_language_and_model_follow_the_recogniser_locale(monkeypatch):
    for name in ("VOICE_WHISPER_MODEL_EN", "VOICE_WHISPER_MODEL_AR"):
        monkeypatch.delenv(name, raising=False)
    assert local_speech.lang_code("en-US") == "en" and local_speech.lang_code("ar-AE") == "ar" and local_speech.lang_code(None) == "en"
    assert local_speech.model_name("en") == "base.en" and local_speech.model_name("ar") == "small"


def test_controller_choice():
    assert voice.controller_js("local") is voice.LOCAL_VOICE_JS
    assert voice.controller_js("browser") is voice.VOICE_JS


def test_local_controller_keeps_the_page_contract():
    js = voice.LOCAL_VOICE_JS
    assert "window.fobVoice = V" in js and "emitEvent('fob_voice'" in js
    for kind in ("unsupported", "busy", "starting", "listening", "final", "end", "error", "cancelled"):
        assert "kind: '%s'" % kind in js
    assert "/api/voice/transcribe" in js and "'X-FOB-Voice': '1'" in js
    assert "getUserMedia" in js and "MediaRecorder" in js
    # a cancelled recording is never uploaded; a recording the level meter did not hear is still transcribed
    assert "outcome === 'cancel'" in js and "outcome === 'timeout'" not in js
    assert "V.ctx.resume()" in js and "X-FOB-Voice-Peak" in js
    # never sends the question by itself: the controller only reports text, the page appends it to the draft
    assert "send" not in js.lower().replace("x-fob-voice", "")


def test_privacy_notice_names_the_local_engine_in_both_languages():
    assert "on this computer" in i18n.t("en", "voice_privacy_local")
    assert i18n.t("ar", "voice_privacy_local") != i18n.t("en", "voice_privacy_local")


def test_voice_machine_fills_the_draft_from_a_local_transcript():
    vm = voice.VoiceMachine()
    for event in ({"kind": "starting", "lang": "en-US"}, {"kind": "listening"}, {"kind": "interim", "text": "…"},
                  {"kind": "final", "text": "What is the phone number of Human Resources?"}):
        vm.apply(event)
    assert vm.apply({"kind": "end"}) == "voice_ready" and vm.state == voice.READY
    assert voice.merge_draft("", vm.final) == "What is the phone number of Human Resources?"


def test_a_recording_with_no_words_ends_as_no_speech():
    vm = voice.VoiceMachine()
    for event in ({"kind": "starting"}, {"kind": "listening"}, {"kind": "interim", "text": "…"}):
        vm.apply(event)
    assert vm.apply({"kind": "end"}) == "voice_err_no_speech"


# ---------------------------------------------------------------- input limits (no model needed) --------------------------
def test_empty_and_oversized_recordings_are_rejected_without_loading_a_model(monkeypatch):
    def boom(_code):
        raise AssertionError("the model must not be loaded")
    monkeypatch.setattr(local_speech, "_load", boom)
    assert local_speech.transcribe(b"", "en-US") == {"error": "no-speech"}
    assert local_speech.transcribe(b"x" * (local_speech.MAX_AUDIO_BYTES + 1), "en-US") == {"error": "too-large"}


def test_engine_switched_off_reports_unavailable(monkeypatch):
    monkeypatch.setenv("VOICE_ENGINE", "browser")
    assert local_speech.transcribe(b"abc", "en-US") == {"error": "unavailable"}


# ---------------------------------------------------------------- the HTTP endpoint ----------------------------------------
@pytest.fixture()
def endpoint(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import nicegui.run as nrun

    async def io_bound(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    monkeypatch.setattr(nrun, "io_bound", io_bound)
    seen = []
    monkeypatch.setattr(local_speech, "transcribe", lambda body, lang: (seen.append((body, lang)), {"text": "hello"})[1])
    app = FastAPI()
    local_speech.register(app)
    return TestClient(app), seen


def test_endpoint_requires_the_page_header(endpoint):
    client, seen = endpoint
    r = client.post(local_speech.TRANSCRIBE_URL + "?lang=en-US", content=b"audio")
    assert r.status_code == 403 and not seen


def test_endpoint_passes_audio_and_locale(endpoint):
    client, seen = endpoint
    r = client.post(local_speech.TRANSCRIBE_URL + "?lang=ar-AE", content=b"audio", headers={"X-FOB-Voice": "1"})
    assert r.status_code == 200 and r.json() == {"text": "hello"} and seen == [(b"audio", "ar-AE")]
    assert r.headers.get("cache-control") == "no-store"


def test_endpoint_refuses_a_declared_oversized_body(endpoint):
    client, seen = endpoint
    r = client.post(local_speech.TRANSCRIBE_URL, content=b"a", headers={"X-FOB-Voice": "1", "Content-Length": str(local_speech.MAX_AUDIO_BYTES + 1)})
    assert r.status_code == 413 and not seen


def test_start_app_registers_the_endpoint_and_page_uses_the_selected_controller():
    import inspect
    src = inspect.getsource(chat.start_app)
    assert "local_speech.register(app)" in src and "local_speech.warm_up" in src
    assert "voice.controller_js(local_speech.engine())" in inspect.getsource(chat)


# ---------------------------------------------------------------- real transcription ---------------------------------------
def _model_cached(name: str) -> bool:
    root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--Systran--faster-whisper-" + name)
    return os.path.isdir(root)


def _speak_to_wav(text: str, path: str) -> bool:
    if sys.platform != "win32" or not shutil.which("powershell"):
        return False
    script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$s.SetOutputToWaveFile('%s'); $s.Speak('%s'); $s.Dispose()" % (path.replace("'", "''"), text.replace("'", "''")))
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True, timeout=60)
    except Exception:
        return False
    return os.path.getsize(path) > 1000


def _wav_to_webm_opus(wav_path: str) -> bytes:
    import av
    out = io.BytesIO()
    with av.open(wav_path) as src, av.open(out, "w", format="webm") as dst:
        stream = dst.add_stream("libopus", rate=48000)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        for frame in src.decode(audio=0):
            for f in resampler.resample(frame):
                for packet in stream.encode(f):
                    dst.mux(packet)
        for packet in stream.encode(None):
            dst.mux(packet)
    return out.getvalue()


needs_model = pytest.mark.skipif(not (local_speech.installed() and _model_cached("base.en")), reason="faster-whisper or the base.en model is not installed")


@needs_model
def test_a_spoken_english_question_recorded_as_webm_is_transcribed(monkeypatch):
    monkeypatch.delenv("VOICE_ENGINE", raising=False)
    monkeypatch.delenv("VOICE_WHISPER_MODEL_EN", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "q.wav")
        if not _speak_to_wav("What is the phone number of the Human Resources Department?", wav):
            pytest.skip("no speech synthesiser")
        result = local_speech.transcribe(_wav_to_webm_opus(wav), "en-US")
    assert "error" not in result, result
    text = result["text"].lower()
    assert "phone number" in text and "human resources" in text
    assert result["language"] == "en" and result["model"] == "base.en"


@needs_model
def test_institution_names_are_recognised(monkeypatch):
    monkeypatch.delenv("VOICE_ENGINE", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "q.wav")
        if not _speak_to_wav("How do I check a paper with iThenticate on Blackboard?", wav):
            pytest.skip("no speech synthesiser")
        result = local_speech.transcribe(_wav_to_webm_opus(wav), "en-US")
    assert "ithenticate" in result.get("text", "").lower().replace(" ", "") and "blackboard" in result["text"].lower()


@needs_model
def test_garbage_audio_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("VOICE_ENGINE", raising=False)
    assert local_speech.transcribe(b"this is not audio at all" * 50, "en-US") == {"error": "bad-audio"}
