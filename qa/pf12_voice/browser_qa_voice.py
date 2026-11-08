"""Real-browser QA of the microphone control with a MOCKED recogniser (post-freeze PF-12).

    python -X utf8 qa/pf12_voice/browser_qa_voice.py <base url> <output dir> [--browser chrome|edge] [--label after]

Headless Chrome over the DevTools protocol against the RUNNING application. Before every navigation a script replaces
``window.SpeechRecognition`` / ``webkitSpeechRecognition`` with a fake that follows the Web Speech API's event contract
and plays a scenario (``window.__voiceScenario``): a final English transcript, a final Arabic transcript, a mixed
transcript, ``no-speech``, ``not-allowed``, ``network``, a cancellation, a silent recogniser (the page's timeout), or no
API at all (unsupported browser). What this run establishes: the page's integration with the recogniser (states, labels,
live strip, transcript into the composer, draft protection, editing, sending through the normal handler to the REAL
pipeline, cleanup on New chat), at 320, 375, 768, 1024 and 1440 px, in both interface languages and both themes. What it
does NOT establish: recognition of real speech (the recogniser is a fake). Launch the server with the model translation
switched off (``launch_bilingual_qa.py``) so no inference request is made.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pf10_ui"))
from browser_qa import CDP, JS_CHECKS, VIEWPORTS, find_browser, launch, shot   # noqa: E402

INTERACT_AT = {1440, 375}
EN_TEXT = "What is the phone number of the Human Resources Department?"
AR_TEXT = "ما هو رقم هاتف قسم الموارد البشرية؟"
MIXED_TEXT = "كيف أستخدم Blackboard لإرسال Announcement؟"

FAKE_RECOGNISER = r"""
(() => {
  window.__voiceScenario = window.__voiceScenario || {kind: 'final', en: %s, ar: %s, mixed: %s, delayMs: 350};
  window.__voiceLog = [];
  class FakeRecognition {
    constructor() { this.lang = 'en-US'; this.continuous = false; this.interimResults = false; this.maxAlternatives = 1; this._stopped = false; this._timers = []; }
    _emit(name, ev) { window.__voiceLog.push({t: Date.now(), name: name, lang: this.lang}); const h = this['on' + name]; if (typeof h === 'function') h.call(this, ev || {}); }
    _later(ms, fn) { this._timers.push(setTimeout(() => { if (!this._stopped) fn(); }, ms)); }
    _result(text, isFinal) { const alt = {transcript: text, confidence: 0.9}; const res = {0: alt, length: 1, isFinal: isFinal}; const list = {0: res, length: 1}; this._emit('result', {resultIndex: 0, results: list}); }
    start() {
      const sc = window.__voiceScenario; const d = sc.delayMs || 350;
      window.__voiceLog.push({t: Date.now(), name: 'start-called', lang: this.lang});
      if (sc.kind === 'not-allowed') { this._later(d, () => { this._emit('error', {error: 'not-allowed'}); this._emit('end'); }); return; }
      this._later(d, () => this._emit('start'));
      this._later(d + 50, () => this._emit('audiostart'));
      if (sc.kind === 'silent') return;                                   // never speaks: the page's own timeout must end it
      if (sc.kind === 'no-speech') { this._later(d + 600, () => { this._emit('error', {error: 'no-speech'}); this._emit('end'); }); return; }
      if (sc.kind === 'network') { this._later(d + 600, () => { this._emit('error', {error: 'network'}); this._emit('end'); }); return; }
      const text = this.lang.startsWith('ar') ? (sc.kind === 'mixed' ? sc.mixed : sc.ar) : sc.en;
      const words = text.split(' ');
      this._later(d + 400, () => this._result(words.slice(0, 2).join(' '), false));
      this._later(d + 800, () => this._result(words.slice(0, 4).join(' '), false));
      this._later(d + 1300, () => { this._result(text, true); });
      this._later(d + 1500, () => { this._emit('end'); });
    }
    stop() { window.__voiceLog.push({t: Date.now(), name: 'stop-called'}); this._stopped = true; this._timers.forEach(clearTimeout); this._emit('end'); }
    abort() { window.__voiceLog.push({t: Date.now(), name: 'abort-called'}); this._stopped = true; this._timers.forEach(clearTimeout); this._emit('error', {error: 'aborted'}); this._emit('end'); }
  }
  if (window.__voiceScenario.kind === 'unsupported') { try { delete window.SpeechRecognition; delete window.webkitSpeechRecognition; } catch (e) {} window.SpeechRecognition = undefined; window.webkitSpeechRecognition = undefined; }
  else { window.SpeechRecognition = FakeRecognition; window.webkitSpeechRecognition = FakeRecognition; }
})();
""" % (json.dumps(EN_TEXT), json.dumps(AR_TEXT), json.dumps(MIXED_TEXT))

JS_VOICE = r"""
(() => {
  const q = s => document.querySelector(s);
  const mic = q('.mic-btn'), cancel = q('.voice-cancel-btn'), strip = q('.voice-strip'), ta = q('.composer textarea'), help = q('.composer-help'), send = q('.send-btn');
  const rect = el => { if (!el) return null; const r = el.getBoundingClientRect(); return {x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height), inside: r.left >= -1 && r.right <= innerWidth + 1 && r.top >= -1 && r.bottom <= innerHeight + 1}; };
  const overlap = (a, b) => { if (!a || !b) return null; const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(); return !(ra.right <= rb.left || rb.right <= ra.left || ra.bottom <= rb.top || rb.bottom <= ra.top); };
  return {
    micPresent: !!mic, micState: mic ? [...mic.classList].filter(c => c.startsWith('voice-')).join(',') : null, micLabel: mic ? mic.getAttribute('aria-label') : null,
    micPressed: mic ? mic.getAttribute('aria-pressed') : null, micIcon: mic ? (mic.querySelector('.q-icon') || {}).textContent : null, micRect: rect(mic),
    micOverlapsSend: overlap(mic, send), micOverlapsTextarea: overlap(mic, ta),
    cancelVisible: !!(cancel && getComputedStyle(cancel).display !== 'none' && cancel.getBoundingClientRect().width > 0),
    stripVisible: !!(strip && getComputedStyle(strip).display !== 'none' && strip.innerText.trim().length > 0), stripText: strip ? strip.innerText.trim().slice(0, 160) : null,
    textarea: ta ? ta.value : null, textareaDir: ta ? getComputedStyle(ta).direction : null, textareaScrollClipped: ta ? ta.scrollWidth > ta.clientWidth + 1 : null,
    help: help ? help.innerText.trim() : null, privacy: (() => { const p = q('.voice-privacy'); return p ? p.innerText.trim().slice(0, 80) : null; })(),
    activeTag: document.activeElement ? document.activeElement.tagName + '.' + String(document.activeElement.className).slice(0, 30) : null,
    focusOutline: document.activeElement ? getComputedStyle(document.activeElement).outlineStyle : null,
    voiceLog: (window.__voiceLog || []).map(e => e.name), userBubbles: document.querySelectorAll('.msg-row.user').length, chips: document.querySelectorAll('.chip').length,
    lang: document.documentElement.getAttribute('lang'), dir: getComputedStyle(document.documentElement).direction, theme: document.documentElement.getAttribute('data-theme'),
  };
})()
"""


async def set_scenario(cdp: CDP, kind: str) -> None:
    await cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": "window.__voiceScenario = {kind: %s, en: %s, ar: %s, mixed: %s, delayMs: 350};" % (
        json.dumps(kind), json.dumps(EN_TEXT), json.dumps(AR_TEXT), json.dumps(MIXED_TEXT)) + FAKE_RECOGNISER})


async def click_mic(cdp: CDP) -> bool:
    return await cdp.evaluate("(() => { const b = document.querySelector('.mic-btn'); if (!b) return false; b.click(); return true; })()")


async def wait_state(cdp: CDP, state: str, timeout: float = 10) -> bool:
    return await cdp.wait_for("(() => { const b = document.querySelector('.mic-btn'); return !!b && b.classList.contains('voice-%s'); })()" % state, timeout=timeout)


async def open_page(cdp: CDP, base_url: str, lang: str, width: int, height: int) -> bool:
    await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 768})
    await cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": width < 768})
    await cdp.send("Page.navigate", {"url": base_url + "/?lang=" + lang})
    ok = await cdp.wait_for("!!document.querySelector('.composer textarea') && !!document.querySelector('.mic-btn')", timeout=60)
    await asyncio.sleep(1.0)
    return ok


async def send_enter(cdp: CDP) -> None:
    await cdp.evaluate("document.querySelector('.composer textarea').focus()")
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})


async def run(base_url: str, out_dir: str, browser: str, label: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    exe = find_browser(browser)
    profile = tempfile.mkdtemp(prefix="pf12-voice-")
    proc, ws_url = launch(exe, profile)
    record = {"label": label, "browser": exe, "base_url": base_url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "recogniser": "MOCKED (fake SpeechRecognition injected before navigation)",
              "viewports": [], "scenarios": []}
    try:
        async with CDP(ws_url) as cdp:
            targets = (await cdp.send("Target.getTargets"))["targetInfos"]
            page = next(t for t in targets if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send("Browser.grantPermissions", {"origin": base_url, "permissions": ["audioCapture"]})
            # ---- layout at five widths, both languages, idle and listening ---------------------------------------------------
            await set_scenario(cdp, "final")
            for width, height in VIEWPORTS:
                vp = {"width": width, "height": height, "shots": [], "checks": {}}
                record["viewports"].append(vp)
                for lang in ("en", "ar"):
                    ok = await open_page(cdp, base_url, lang, width, height)
                    vp["checks"]["%s_idle" % lang] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_VOICE)), "ready": ok}
                    vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_%s_idle" % (label, width, height, lang)))
                    await click_mic(cdp)
                    await wait_state(cdp, "listening", 8)
                    await asyncio.sleep(0.9)                                   # interim results visible
                    vp["checks"]["%s_listening" % lang] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_VOICE))}
                    vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_%s_listening" % (label, width, height, lang)))
                    await wait_state(cdp, "ready", 8)
                    await asyncio.sleep(0.5)
                    vp["checks"]["%s_ready" % lang] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_VOICE))}
                    vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_%s_ready" % (label, width, height, lang)))
                    if width in INTERACT_AT:
                        # the transcript is in the box and NOT sent; sending it with Enter reaches the real pipeline
                        before = vp["checks"]["%s_ready" % lang]["userBubbles"]
                        await send_enter(cdp)
                        answered = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > 0", timeout=120)
                        await asyncio.sleep(0.8)
                        c = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_VOICE))}
                        c["answered"] = answered
                        c["sent_exactly_once"] = (c["userBubbles"] - before) == 1
                        vp["checks"]["%s_sent" % lang] = c
                        vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_%s_sent" % (label, width, height, lang)))
                        if lang == "ar":                                        # dark theme in Arabic while listening
                            await cdp.evaluate("(() => { const b = [...document.querySelectorAll('button')].find(b => /dark|الداكن/i.test(b.getAttribute('aria-label') || '')); if (b) b.click(); })()")
                            await click_mic(cdp)
                            await wait_state(cdp, "listening", 8)
                            await asyncio.sleep(0.9)
                            vp["checks"]["ar_dark_listening"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_VOICE))}
                            vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_ar_dark_listening" % (label, width, height)))
                            await cdp.evaluate("(() => { const b = document.querySelector('.voice-cancel-btn'); if (b) b.click(); })()")
                            await wait_state(cdp, "idle", 8)
                            vp["checks"]["ar_dark_cancelled"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_VOICE))}
            # ---- scenarios at 1440 px, English interface unless stated ---------------------------------------------------------
            async def scenario(name: str, kind: str, lang: str, steps):
                await set_scenario(cdp, kind)
                await open_page(cdp, base_url, lang, 1440, 900)
                rec = {"name": name, "kind": kind, "lang": lang, "steps": []}
                for step in steps:
                    rec["steps"].append(await step())
                rec["final"] = await cdp.evaluate(JS_VOICE)
                rec["shot"] = await shot(cdp, out_dir, "%s_scenario_%s" % (label, name))
                record["scenarios"].append(rec)

            async def s_click_and_wait(state, timeout=10):
                await click_mic(cdp)
                ok = await wait_state(cdp, state, timeout)
                await asyncio.sleep(0.4)
                return {"waited_for": state, "reached": ok, **{k: v for k, v in (await cdp.evaluate(JS_VOICE)).items() if k in ("micState", "help", "textarea", "micPressed", "cancelVisible", "stripVisible")}}

            async def s_type(text):
                await cdp.evaluate("document.querySelector('.composer textarea').focus()")
                await cdp.send("Input.insertText", {"text": text})
                await asyncio.sleep(0.3)
                return {"typed": text, "textarea": await cdp.evaluate("document.querySelector('.composer textarea').value")}

            async def s_cancel_button():
                await cdp.evaluate("(() => { const b = document.querySelector('.voice-cancel-btn'); if (b) b.click(); })()")
                ok = await wait_state(cdp, "idle", 8)
                return {"cancel_clicked": True, "idle": ok, "help": await cdp.evaluate("document.querySelector('.composer-help').innerText")}

            async def s_escape():
                await cdp.evaluate("document.querySelector('.composer textarea').focus()")
                for kind in ("keyDown", "keyUp"):
                    await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Escape", "code": "Escape", "windowsVirtualKeyCode": 27})
                ok = await wait_state(cdp, "idle", 8)
                return {"escape": True, "idle": ok}

            async def s_double_click():
                await click_mic(cdp)
                await asyncio.sleep(0.15)
                await click_mic(cdp)                                            # second click while starting/listening = Stop
                await asyncio.sleep(1.2)
                return {"double_click": True, **{k: v for k, v in (await cdp.evaluate(JS_VOICE)).items() if k in ("micState", "voiceLog", "help")}}

            async def s_new_chat_while_listening():
                await click_mic(cdp)
                await wait_state(cdp, "listening", 8)
                await cdp.evaluate("(() => { const b = [...document.querySelectorAll('button')].find(b => /new chat|محادثة جديدة/i.test(b.getAttribute('aria-label') || b.textContent || '')); if (b) b.click(); })()")
                ok = await wait_state(cdp, "idle", 8)
                return {"new_chat_during_listening": True, "idle_after": ok, "voiceLog": await cdp.evaluate("(window.__voiceLog || []).map(e => e.name)")}

            async def s_wait_timeout():
                await click_mic(cdp)
                await wait_state(cdp, "listening", 8)
                t0 = time.time()
                ok = await wait_state(cdp, "error", 20)
                return {"silent_recogniser_timeout": True, "error_state_after_s": round(time.time() - t0, 1), "reached_error": ok,
                        "help": await cdp.evaluate("document.querySelector('.composer-help').innerText")}

            async def s_reload_while_listening():
                await click_mic(cdp)
                await wait_state(cdp, "listening", 8)
                await cdp.send("Page.reload")
                await cdp.wait_for("!!document.querySelector('.mic-btn')", timeout=60)
                await asyncio.sleep(1.0)
                return {"reload_during_listening": True, "state_after_reload": await cdp.evaluate("[...document.querySelector('.mic-btn').classList].filter(c => c.startsWith('voice-')).join(',')")}

            await scenario("draft_protected", "final", "en", [lambda: s_type("Please tell me:"), lambda: s_click_and_wait("ready")])
            await scenario("edit_then_send", "final", "en", [lambda: s_click_and_wait("ready"), lambda: s_type(" please"), send_enter_and_wait(cdp)])
            await scenario("arabic_transcript", "final", "ar", [lambda: s_click_and_wait("ready")])
            await scenario("mixed_transcript", "mixed", "ar", [lambda: s_click_and_wait("ready")])
            await scenario("no_speech", "no-speech", "en", [lambda: s_click_and_wait("error")])
            await scenario("permission_denied", "not-allowed", "en", [lambda: s_click_and_wait("error")])
            await scenario("permission_denied_ar", "not-allowed", "ar", [lambda: s_click_and_wait("error")])
            await scenario("network", "network", "en", [lambda: s_click_and_wait("error")])
            await scenario("cancel_button", "silent", "en", [lambda: s_click_and_wait("listening"), s_cancel_button])
            await scenario("escape_cancels", "silent", "en", [lambda: s_click_and_wait("listening"), s_escape])
            await scenario("double_click_stops", "silent", "en", [s_double_click])
            await scenario("new_chat_cancels", "silent", "en", [s_new_chat_while_listening])
            await scenario("timeout", "silent", "en", [s_wait_timeout])
            await scenario("reload_releases", "silent", "en", [s_reload_while_listening])
            await scenario("unsupported", "unsupported", "en", [lambda: s_click_and_wait("unsupported")])
            await scenario("unsupported_ar", "unsupported", "ar", [lambda: s_click_and_wait("unsupported")])
    finally:
        proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "%s_voice_browser_qa.json" % label), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return record


def send_enter_and_wait(cdp: CDP):
    async def _step():
        before = await cdp.evaluate("document.querySelectorAll('.msg-row.user').length")
        await send_enter(cdp)
        answered = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > 0", timeout=120)
        await asyncio.sleep(0.8)
        after = await cdp.evaluate("document.querySelectorAll('.msg-row.user').length")
        return {"sent": after - before, "answered": answered, "chips": await cdp.evaluate("document.querySelectorAll('.chip').length"),
                "question_shown": await cdp.evaluate("(() => { const b = document.querySelector('.bubble.user'); return b ? b.innerText : null; })()")}
    return _step


def summarize(record: dict) -> int:
    failures = 0
    for vp in record["viewports"]:
        w, h = vp["width"], vp["height"]
        for state, c in vp["checks"].items():
            problems = []
            if c.get("horizontalOverflow"):
                problems.append("horizontal overflow")
            if not c.get("micPresent") or not (c.get("micRect") or {}).get("inside"):
                problems.append("microphone missing or outside the viewport")
            if c.get("micOverlapsSend") or c.get("micOverlapsTextarea"):
                problems.append("microphone overlaps another control")
            if not (c.get("textarea") is None) and c.get("textareaScrollClipped"):
                problems.append("transcript clipped in the composer")
            if state.endswith("_listening") and (c.get("micPressed") != "true" or not c.get("stripVisible") or not c.get("cancelVisible")):
                problems.append("listening state not shown (pressed=%s strip=%s cancel=%s)" % (c.get("micPressed"), c.get("stripVisible"), c.get("cancelVisible")))
            if state.endswith("_ready") and (not c.get("textarea") or c.get("userBubbles")):
                problems.append("transcript missing from the composer or sent by itself")
            if state.endswith("_sent") and (not c.get("answered") or not c.get("sent_exactly_once") or not c.get("chips")):
                problems.append("sending the transcript did not produce exactly one answered turn with sources")
            if state.startswith("ar_") and c.get("dir") != "rtl":
                problems.append("Arabic page not rtl")
            failures += len(problems)
            print("%4dx%-4d %-18s %s" % (w, h, state, "OK" if not problems else "; ".join(problems)))
    for s in record["scenarios"]:
        f = s["final"]
        ok = True
        if s["name"] in ("draft_protected",):
            ok = f["textarea"] == "Please tell me: " + EN_TEXT and f["userBubbles"] == 0
        elif s["name"] == "edit_then_send":
            ok = s["steps"][-1]["sent"] == 1 and s["steps"][-1]["answered"] and (s["steps"][-1]["question_shown"] or "").endswith("please") and s["steps"][-1]["chips"] > 0
        elif s["name"] == "arabic_transcript":
            ok = f["textarea"] == AR_TEXT and f["userBubbles"] == 0 and f["textareaDir"] == "rtl"
        elif s["name"] == "mixed_transcript":
            ok = f["textarea"] == MIXED_TEXT and f["userBubbles"] == 0
        elif s["name"] in ("no_speech", "permission_denied", "permission_denied_ar", "network"):
            ok = f["micState"] == "voice-error" and bool(f["help"]) and f["userBubbles"] == 0
        elif s["name"] in ("cancel_button", "escape_cancels", "new_chat_cancels"):
            ok = f["micState"] == "voice-idle" and "abort-called" in f["voiceLog"] and not f["textarea"]
        elif s["name"] == "double_click_stops":
            ok = "stop-called" in f["voiceLog"] and f["micState"] in ("voice-error", "voice-idle", "voice-ready")
        elif s["name"] == "timeout":
            ok = f["micState"] == "voice-error" and s["steps"][0]["reached_error"] and s["steps"][0]["error_state_after_s"] <= 16
        elif s["name"] == "reload_releases":
            ok = s["steps"][0]["state_after_reload"] == "voice-idle"
        elif s["name"].startswith("unsupported"):
            ok = f["micState"] == "voice-unsupported" and bool(f["help"]) and f["userBubbles"] == 0
        failures += 0 if ok else 1
        print("scenario %-22s %s | state=%s help=%s" % (s["name"], "OK" if ok else "PROBLEM", f["micState"], (f["help"] or "")[:70]))
    print("problems:", failures)
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_dir")
    ap.add_argument("--browser", choices=("chrome", "edge"), default="chrome")
    ap.add_argument("--label", default="after")
    a = ap.parse_args()
    rec = asyncio.run(run(a.base_url.rstrip("/"), a.out_dir, a.browser, a.label))
    sys.exit(1 if summarize(rec) else 0)
