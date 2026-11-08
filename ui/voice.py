"""
Voice input (post-freeze PF-12): the microphone state machine, the transcript policy and the browser-side recogniser.

Technology (D-123): the browser's Web Speech API (``SpeechRecognition`` / ``webkitSpeechRecognition``). Recognition runs in
the browser the faculty member already uses, needs no new dependency, no new server and no new credential; the audio
goes to the browser vendor's speech service (Google for Chrome, Microsoft for Edge, Apple for Safari), which the page
says plainly. Browsers without the API (Firefox, some embedded browsers) get an honest "not available" state; typing
always works.

Flow (the transcript is never sent by itself):

    microphone button -> browser recogniser (one utterance, interim results shown) -> FINAL transcript appended to the
    composer draft -> the user edits and sends with Send or Enter -> the unchanged chat handler -> the unchanged pipeline

This module is pure Python (no NiceGUI import) so the state machine is unit-tested without a browser:
    :class:`VoiceMachine`  states and transitions driven by the browser events
    :func:`merge_draft`    how a transcript joins text already in the composer (never replaces it)
    :func:`locale_for`     the recogniser locale for a voice language
    :data:`VOICE_JS`       the browser controller that drives the recogniser and reports events with ``emitEvent``

Browser events (``fob_voice``): ``{"kind": ..., "text": ..., "code": ..., "lang": ...}`` with kinds
``unsupported`` ``busy`` ``starting`` ``listening`` ``interim`` ``final`` ``end`` ``error`` ``cancelled``.
Error codes follow the Web Speech API (``not-allowed`` ``service-not-allowed`` ``audio-capture`` ``no-speech``
``network`` ``aborted`` ``language-not-supported`` ``bad-grammar``) plus the controller's own ``timeout``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: Recogniser locales. ``ar-AE`` is the Gulf Arabic locale Chrome's recogniser lists; ``en-US`` the English one.
LOCALES: Dict[str, str] = {"en": "en-US", "ar": "ar-AE"}
#: Silence after which the controller stops a recognition that produced nothing (ms), and the hard cap of one utterance.
NO_RESULT_TIMEOUT_MS = 12000
MAX_UTTERANCE_MS = 60000

IDLE, STARTING, LISTENING, READY, ERROR, UNSUPPORTED = "idle", "starting", "listening", "ready", "error", "unsupported"
STATES = (IDLE, STARTING, LISTENING, READY, ERROR, UNSUPPORTED)

#: Message key (``ui.i18n``) for each error code; unknown codes map to the generic one.
ERROR_KEYS: Dict[str, str] = {
    "not-allowed": "voice_err_permission", "service-not-allowed": "voice_err_permission", "audio-capture": "voice_err_no_mic",
    "no-speech": "voice_err_no_speech", "network": "voice_err_network", "aborted": "voice_cancelled", "timeout": "voice_err_timeout",
    "language-not-supported": "voice_err_language", "bad-grammar": "voice_err_unrecognised", "unsupported": "voice_unsupported",
}


def locale_for(voice_lang: Optional[str], interface_lang: str) -> str:
    """The recogniser locale: the chosen voice language, or the interface language when the choice is ``auto``."""
    code = (voice_lang or "auto").lower()
    if code not in LOCALES:
        code = interface_lang if interface_lang in LOCALES else "en"
    return LOCALES[code]


def merge_draft(draft: str, transcript: str) -> str:
    """Join a final transcript to the text already in the composer: the draft is never replaced. A draft that ends
    without a space gets one; an empty draft is simply the transcript."""
    text = " ".join((transcript or "").split())
    base = (draft or "").rstrip()
    if not text:
        return draft or ""
    if not base:
        return text
    return base + " " + text


@dataclass
class VoiceMachine:
    """The microphone's state as the page shows it. ``apply`` returns the message key to show (or None)."""
    state: str = IDLE
    interim: str = ""
    final: str = ""
    error: Optional[str] = None
    lang: str = ""
    had_final: bool = False
    history: list = field(default_factory=list)

    def apply(self, event: Dict[str, Any]) -> Optional[str]:
        kind = str(event.get("kind") or "")
        self.history.append(kind if kind != "error" else "error:%s" % event.get("code"))
        if kind == "unsupported":
            self.state, self.error = UNSUPPORTED, "unsupported"
            return "voice_unsupported"
        if kind == "busy":                                     # a second start while listening: ignored, nothing changes
            return None
        if kind == "starting":
            self.state, self.interim, self.final, self.error, self.had_final = STARTING, "", "", None, False
            self.lang = str(event.get("lang") or "")
            return "voice_starting"
        if kind == "listening":
            self.state = LISTENING
            return "voice_listening"
        if kind == "interim":
            if self.state in (STARTING, LISTENING):
                self.state = LISTENING
                self.interim = str(event.get("text") or "")
            return None
        if kind == "final":
            text = " ".join(str(event.get("text") or "").split())
            if text:
                self.final = (self.final + " " + text).strip()
                self.had_final = True
            self.interim = ""
            return None
        if kind == "cancelled":
            self.state, self.interim, self.error = IDLE, "", None
            return "voice_cancelled"
        if kind == "error":
            code = str(event.get("code") or "")
            if code == "aborted":
                self.state, self.interim, self.error = IDLE, "", None
                return "voice_cancelled"
            self.state, self.interim, self.error = ERROR, "", code or "unknown"
            return ERROR_KEYS.get(code, "voice_err_unrecognised")
        if kind == "end":
            if self.state in (ERROR, UNSUPPORTED, IDLE):
                return None
            if self.had_final:
                self.state = READY
                return "voice_ready"
            self.state, self.error = ERROR, "no-speech"
            return "voice_err_no_speech"
        return None

    @property
    def active(self) -> bool:
        return self.state in (STARTING, LISTENING)

    def to_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "interim": self.interim, "final": self.final, "error": self.error, "lang": self.lang, "had_final": self.had_final}


#: The browser controller. One recogniser at a time; interim results; a no-result timeout and a hard cap; abort on page
#: hide; every outcome reported to the page with ``emitEvent('fob_voice', {...})``. Nothing is stored; no audio reaches
#: the application server, only the recogniser's text.
VOICE_JS = r"""
(function () {
  if (window.fobVoice) return;
  const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition || null;
  const emit = (payload) => { try { emitEvent('fob_voice', payload); } catch (e) { /* page gone */ } };
  const V = { rec: null, active: false, lang: '', noResultTimer: null, capTimer: null, hadResult: false, cancelling: false };
  V.supported = () => !!Ctor;
  const clearTimers = () => { if (V.noResultTimer) { clearTimeout(V.noResultTimer); V.noResultTimer = null; } if (V.capTimer) { clearTimeout(V.capTimer); V.capTimer = null; } };
  const finish = () => { clearTimers(); V.active = false; V.rec = null; };
  V.start = (lang, noResultMs, capMs) => {
    if (!Ctor) { emit({kind: 'unsupported', lang: lang}); return 'unsupported'; }
    if (V.active) { emit({kind: 'busy', lang: lang}); return 'busy'; }
    let rec;
    try { rec = new Ctor(); } catch (e) { emit({kind: 'unsupported', lang: lang}); return 'unsupported'; }
    V.rec = rec; V.active = true; V.lang = lang; V.hadResult = false; V.cancelling = false;
    rec.lang = lang; rec.continuous = false; rec.interimResults = true; rec.maxAlternatives = 1;
    emit({kind: 'starting', lang: lang});
    rec.onstart = () => { emit({kind: 'listening', lang: lang}); };
    rec.onaudiostart = () => { if (!V.noResultTimer) V.noResultTimer = setTimeout(() => { if (V.active && !V.hadResult) { V.cancelling = 'timeout'; try { rec.abort(); } catch (e) {} } }, noResultMs || 12000); };
    rec.onresult = (ev) => {
      let interim = '', finalText = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (r.isFinal) finalText += r[0].transcript; else interim += r[0].transcript;
      }
      if (finalText) { V.hadResult = true; clearTimers(); emit({kind: 'final', text: finalText, lang: lang}); }
      else if (interim) { V.hadResult = true; emit({kind: 'interim', text: interim, lang: lang}); }
    };
    rec.onerror = (ev) => {
      const code = V.cancelling === 'timeout' ? 'timeout' : (V.cancelling ? 'aborted' : (ev && ev.error) || 'unknown');
      emit({kind: 'error', code: code, lang: lang});
      finish();
    };
    rec.onend = () => { if (V.active) { emit({kind: V.cancelling === true ? 'cancelled' : 'end', lang: lang}); } finish(); };
    V.capTimer = setTimeout(() => { if (V.active) { try { rec.stop(); } catch (e) {} } }, capMs || 60000);
    try { rec.start(); } catch (e) { emit({kind: 'error', code: 'not-allowed', lang: lang}); finish(); return 'error'; }
    return 'started';
  };
  V.stop = () => { if (V.active && V.rec) { try { V.rec.stop(); } catch (e) {} return true; } return false; };
  V.cancel = () => { if (V.active && V.rec) { V.cancelling = true; try { V.rec.abort(); } catch (e) {} return true; } return false; };
  V.isActive = () => V.active;
  window.addEventListener('pagehide', () => { V.cancel(); });
  window.addEventListener('beforeunload', () => { V.cancel(); });
  window.fobVoice = V;
})();
"""

__all__ = ["ERROR", "ERROR_KEYS", "IDLE", "LISTENING", "LOCALES", "MAX_UTTERANCE_MS", "NO_RESULT_TIMEOUT_MS", "READY", "STARTING", "STATES",
           "UNSUPPORTED", "VOICE_JS", "VoiceMachine", "locale_for", "merge_draft"]
