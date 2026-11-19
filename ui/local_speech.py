"""
Local speech-to-text for the microphone button (faster-whisper, https://github.com/SYSTRAN/faster-whisper).

Why: the browser's Web Speech API depends on the browser vendor's online service and did not work on the team's PC on
19 Nov 2026. With this module the page records the question with the browser's MediaRecorder, sends the recording to
THIS application, and faster-whisper turns it into text on the same computer. The flow the user sees is unchanged:

    microphone button -> speak -> the text appears in the message box -> the user reviews / edits -> Send or Enter

Nothing is sent to a third party, nothing is stored: the recording is held in memory only for the transcription.
The answer pipeline is untouched; this module only produces the text that is put into the message box.

Configuration (environment, all optional):
    VOICE_ENGINE              ``local`` (default when faster-whisper is installed) or ``browser`` (the Web Speech API)
    VOICE_WHISPER_MODEL_EN    default ``base.en``  (English; about 0.35 s per sentence on a desktop CPU)
    VOICE_WHISPER_MODEL_AR    default ``small``    (multilingual model, used for Arabic)
    VOICE_WHISPER_DEVICE      default ``cpu``;  ``cuda`` needs the NVIDIA cuBLAS/cuDNN libraries and falls back to CPU
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

#: Largest recording accepted (bytes) and longest audio transcribed (seconds); a question is a few seconds of speech.
MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
#: Words the English model is biased towards: institution and system names it would otherwise mishear ("iThenticate").
HOTWORDS_EN = ("University of Sharjah, UoS, MyUOS, Blackboard, Blackboard Ultra, Collaborate, iThenticate, Grade Transfer Dashboard, "
               "faculty handbook, Human Resources, Chancellor, VCAA, VCRGS, sabbatical, promotion, probation")
TRANSCRIBE_URL = "/api/voice/transcribe"
#: A request must carry this header: an ordinary cross-site form cannot set it, so other web pages cannot use the endpoint.
REQUIRED_HEADER = ("x-fob-voice", "1")

_models: Dict[str, Any] = {}
_lock = threading.Lock()


def installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def engine() -> str:
    """``local`` when faster-whisper is installed and not switched off, otherwise ``browser``."""
    wanted = (os.getenv("VOICE_ENGINE") or "local").strip().lower()
    return "local" if wanted != "browser" and installed() else "browser"


def lang_code(locale: Optional[str]) -> str:
    return "ar" if (locale or "").lower().startswith("ar") else "en"


def model_name(code: str) -> str:
    if code == "ar":
        return os.getenv("VOICE_WHISPER_MODEL_AR") or "small"
    return os.getenv("VOICE_WHISPER_MODEL_EN") or "base.en"


def _load(code: str):
    name = model_name(code)
    if name in _models:
        return _models[name]
    from faster_whisper import WhisperModel
    device = (os.getenv("VOICE_WHISPER_DEVICE") or "cpu").strip().lower()
    model = None
    if device == "cuda":
        try:
            model = WhisperModel(name, device="cuda", compute_type="float16")
        except Exception as exc:                          # missing NVIDIA libraries: the CPU is fast enough for one question
            log.warning("faster-whisper on CUDA unavailable (%s); using the CPU", type(exc).__name__)
    if model is None:
        model = WhisperModel(name, device="cpu", compute_type="int8")
    _models[name] = model
    return model


def warm_up(code: str = "en") -> None:
    """Load the model in advance (called in a background thread at start-up) so the first question is not slow."""
    if engine() != "local":
        return
    try:
        with _lock:
            _load(code)
    except Exception as exc:
        log.warning("faster-whisper could not be loaded (%s: %s)", type(exc).__name__, exc)


def transcribe(audio: bytes, locale: Optional[str]) -> Dict[str, Any]:
    """Text of one recorded question. Returns ``{"text": ...}`` or ``{"error": code}`` with the page's error codes:
    ``no-speech`` (nothing recognised), ``too-large``, ``bad-audio`` (could not be decoded), ``unavailable``."""
    if not audio:
        return {"error": "no-speech"}
    if len(audio) > MAX_AUDIO_BYTES:
        return {"error": "too-large"}
    if engine() != "local":
        return {"error": "unavailable"}
    code = lang_code(locale)
    started = time.perf_counter()
    try:
        with _lock:
            model = _load(code)
            options: Dict[str, Any] = dict(language=code, task="transcribe", beam_size=5, vad_filter=True, condition_on_previous_text=False)
            if code == "en":
                options["hotwords"] = HOTWORDS_EN
            segments, info = model.transcribe(io.BytesIO(audio), **options)
            parts = []
            for seg in segments:
                if seg.start > MAX_AUDIO_SECONDS:
                    break
                parts.append(seg.text.strip())
    except ImportError:
        return {"error": "unavailable"}
    except Exception as exc:
        log.info("voice transcription failed: %s", type(exc).__name__)
        return {"error": "bad-audio"}
    text = " ".join(p for p in parts if p).strip()
    if not text:
        return {"error": "no-speech"}
    return {"text": text, "language": code, "model": model_name(code), "audio_seconds": round(float(getattr(info, "duration", 0.0)), 2),
            "seconds": round(time.perf_counter() - started, 2)}


def register(app: Any) -> None:
    """Add ``POST /api/voice/transcribe?lang=<locale>`` (body: the recorded audio) to the NiceGUI / FastAPI app."""
    from nicegui import run

    @app.post(TRANSCRIBE_URL, include_in_schema=False)
    async def _transcribe(request: Request) -> JSONResponse:
        if request.headers.get(REQUIRED_HEADER[0]) != REQUIRED_HEADER[1]:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_AUDIO_BYTES:
            return JSONResponse({"error": "too-large"}, status_code=413)
        body = await request.body()
        result = await run.io_bound(transcribe, body, request.query_params.get("lang", "en-US"))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})


__all__ = ["HOTWORDS_EN", "MAX_AUDIO_BYTES", "MAX_AUDIO_SECONDS", "REQUIRED_HEADER", "TRANSCRIBE_URL", "engine", "installed", "lang_code",
           "model_name", "register", "transcribe", "warm_up"]
