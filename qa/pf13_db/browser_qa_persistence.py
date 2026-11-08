"""Real-browser QA of saved conversations (post-freeze PF-13).

Drives the installed Chrome headless over the DevTools protocol against a RUNNING application (nothing mocked) with two
PERSISTENT browser profiles (A and B: separate cookie jars, so two independent visitors). Two phases around a real server
restart:

    python -X utf8 qa/pf13_db/browser_qa_persistence.py <base url> <output dir> --phase create    (server run 1)
    ... stop the server, start it again ...
    python -X utf8 qa/pf13_db/browser_qa_persistence.py <base url> <output dir> --phase restart   (server run 2)

Phase ``create``: visitor A asks a question (the deterministic contact lookup, no model request), the answer and its
citations are recorded, the page is reloaded and the replayed answer compared with the record, the conversation is
renamed through its dialog, a second conversation is held in the Arabic interface, the theme is set to dark in Settings,
and every state is checked at 320/375/768/1024/1440 px in both languages (no overflow, composer visible, a reachable
history control, the phone-width history dialog). Visitor B opens the page and must see none of it. The session cookie's
flags are read from the browser. Phase ``restart``: visitor A's interface language, theme and active conversation come
back, the renamed conversation reopens with exactly the recorded answer and citations without a processing state (no new
request), delete asks for confirmation and removes it, Settings reset returns the defaults, and visitor B still sees
nothing. A JSON record and the screenshots go to the output directory; ``summarize`` prints the problems.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pf10_ui"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from browser_qa import CDP, JS_CHECKS, VIEWPORTS, find_browser, launch, shot   # noqa: E402
from ui import i18n                                                           # noqa: E402

EN_QUESTION = "What is the phone number of the Human Resources Department?"
AR_QUESTION = "ما هو رقم هاتف قسم الموارد البشرية؟"
NEW_TITLE = "HR phone numbers"
S = i18n.STRINGS

JS_STATE = r"""
(() => {
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  const text = el => el ? el.innerText.replace(/\s+/g, ' ').trim() : null;
  const visible = el => { if (!el) return false; const cs = getComputedStyle(el); const r = el.getBoundingClientRect(); return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0 && r.height > 0; };
  const bubbles = qa('.bubble.assistant');
  const sidebarVisible = visible(q('.sidebar'));
  const sessions = qa('.sidebar .session-row');
  return {
    lang: document.documentElement.getAttribute('lang'), dir: getComputedStyle(document.documentElement).direction, theme: document.documentElement.getAttribute('data-theme'),
    sidebarVisible, historyButtonVisible: visible(q('.topbar-history')), settingsButtonVisible: visible(q('.topbar-settings')),
    sessionTitles: sessions.map(r => text(r.querySelector('.session-title'))), activeTitle: text(q('.sidebar .session-row.active .session-title')),
    menuButtons: sessions.filter(r => r.querySelector('.session-menu-btn')).length,
    emptyNote: text(q('.sidebar .sb-empty')), landing: !!q('.landing'),
    userBubbles: qa('.msg-row.user').length, assistantBubbles: bubbles.length, typing: !!q('.typing'),
    lastAnswer: bubbles.length ? text(bubbles[bubbles.length - 1]) : null,
    chips: qa('.chip').map(c => text(c)), evidenceRows: qa('.evidence-row, .evidence details, .evidence-item').length,
    savedNotes: qa('.saved-note').map(n => text(n)), unsavedNotes: qa('.saved-note.warn').length,
    translationNote: text(q('.translation-note')), help: text(q('.composer-help')),
    dialogOpen: !!q('.q-dialog .fob-dialog'), dialogTitle: text(q('.q-dialog .dialog-title')), dialogText: text(q('.q-dialog .dialog-text')),
    dialogSessions: qa('.q-dialog .session-row .session-title').map(text),
    storageBanner: text(q('.storage-banner')), loadError: text(q('.load-error')),
    horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1 || document.body.scrollWidth > innerWidth + 1,
    dialogOverflow: (() => { const d = q('.q-dialog .fob-dialog'); if (!d) return null; const r = d.getBoundingClientRect(); return r.left < -1 || r.right > innerWidth + 1; })(),
  };
})()
"""


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


async def open_page(cdp: CDP, url: str, width: int = 1440, height: int = 900) -> bool:
    await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 768})
    await cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": width < 768})
    await cdp.send("Page.navigate", {"url": url})
    ok = await cdp.wait_for("!!document.querySelector('.composer textarea') && !!document.querySelector('.sidebar .session-list')", timeout=60)
    await asyncio.sleep(1.2)
    return ok


async def send_text(cdp: CDP, text: str) -> None:
    await cdp.evaluate("document.querySelector('.composer textarea').focus()")
    await cdp.send("Input.insertText", {"text": text})
    await asyncio.sleep(0.3)
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})


async def wait_answer(cdp: CDP, count: int, timeout: float = 120) -> bool:
    ok = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length >= %d" % count, timeout=timeout)
    await asyncio.sleep(0.8)
    return ok


async def click(cdp: CDP, selector: str, text: str | None = None) -> bool:
    """Click the first visible element matching the selector (and containing the text when given)."""
    return await cdp.evaluate(
        "(() => { const els = [...document.querySelectorAll(%s)].filter(e => { const cs = getComputedStyle(e); const r = e.getBoundingClientRect(); return cs.display !== 'none' && r.width > 0 && (%s === null || e.innerText.includes(%s)); });"
        " if (!els.length) return false; els[0].click(); return true; })()" % (json.dumps(selector), json.dumps(text), json.dumps(text)))


async def open_menu_item(cdp: CDP, title_fragment: str, item_text: str) -> bool:
    """Open the conversation's menu in the sidebar and choose an item by its label."""
    opened = await cdp.evaluate(
        "(() => { const row = [...document.querySelectorAll('.sidebar .session-row')].find(r => (r.querySelector('.session-title') || {}).innerText.includes(%s));"
        " if (!row) return false; const b = row.querySelector('.session-menu-btn'); if (!b) return false; b.click(); return true; })()" % json.dumps(title_fragment))
    if not opened:
        return False
    await asyncio.sleep(0.6)
    chosen = await click(cdp, ".q-menu .q-item", item_text)
    await asyncio.sleep(0.6)
    return chosen


async def set_input(cdp: CDP, selector: str, value: str) -> None:
    await cdp.evaluate(
        "(() => { const el = document.querySelector(%s); el.focus(); const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; s.call(el, '');"
        " el.dispatchEvent(new Event('input', {bubbles: true})); })()" % json.dumps(selector))
    await cdp.send("Input.insertText", {"text": value})
    await asyncio.sleep(0.4)


async def pick_option(cdp: CDP, row_label: str, option_text: str) -> bool:
    """Open the Quasar select of the settings row with the given label (a real mouse click: the control opens on pointer
    events, not on a synthetic ``click()``) and choose the option with the given text."""
    center = await cdp.evaluate(
        "(() => { const row = [...document.querySelectorAll('.q-dialog .settings-row')].find(r => (r.querySelector('.settings-lbl') || {}).innerText.trim() === %s);"
        " if (!row) return null; const c = row.querySelector('.q-field__control') || row.querySelector('.q-select'); if (!c) return null; const r = c.getBoundingClientRect();"
        " return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()" % json.dumps(row_label))
    if not center:
        return False
    for kind in ("mousePressed", "mouseReleased"):
        await cdp.send("Input.dispatchMouseEvent", {"type": kind, "x": center["x"], "y": center["y"], "button": "left", "clickCount": 1})
    await asyncio.sleep(0.8)
    chosen = await click(cdp, ".q-menu .q-item", option_text)
    await asyncio.sleep(0.8)
    return chosen


async def cookies(cdp: CDP, base_url: str) -> list:
    result = await cdp.send("Network.getCookies", {"urls": [base_url]})
    return [{"name": c["name"], "httpOnly": c.get("httpOnly"), "sameSite": c.get("sameSite"), "secure": c.get("secure"), "session": c.get("session"),
             "expires_days": round((c.get("expires", 0) - time.time()) / 86400, 1) if c.get("expires", -1) > 0 else None, "value_length": len(c.get("value", ""))}
            for c in result.get("cookies", [])]


class Visitor:
    """One persistent headless browser profile (its cookies survive the server restart because the profile folder does)."""

    def __init__(self, exe: str, profile_dir: str, name: str):
        self.exe, self.profile_dir, self.name = exe, profile_dir, name
        self.proc = None
        self.cdp: CDP | None = None

    async def __aenter__(self):
        os.makedirs(self.profile_dir, exist_ok=True)
        self.proc, ws_url = launch(self.exe, self.profile_dir)
        self.cdp = await CDP(ws_url).__aenter__()
        targets = (await self.cdp.send("Target.getTargets"))["targetInfos"]
        page = next(t for t in targets if t["type"] == "page")
        self.cdp.session = (await self.cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
        await self.cdp.send("Page.enable")
        await self.cdp.send("Runtime.enable")
        await self.cdp.send("Network.enable")
        return self

    async def __aexit__(self, *exc):
        try:
            await self.cdp.__aexit__(*exc)
        finally:
            self.proc.kill()
            self.proc.wait(timeout=10)
            await asyncio.sleep(0.5)                                     # let Chrome release the profile before the next phase


def step(record: dict, name: str, ok: bool, **details) -> None:
    record["steps"].append({"name": name, "ok": bool(ok), **details})
    print("%-52s %s" % (name, "OK" if ok else "PROBLEM"), {k: (v if not isinstance(v, str) or len(v) < 80 else v[:77] + "...") for k, v in details.items() if k in ("detail",)} or "")


# ---------------------------------------------------------------------------
# Phase 1: create
# ---------------------------------------------------------------------------
async def phase_create(base_url: str, out_dir: str, exe: str, record: dict) -> dict:
    shots = os.path.join(out_dir, "screenshots")
    os.makedirs(shots, exist_ok=True)
    state = {"base_url": base_url}
    async with Visitor(exe, os.path.join(out_dir, "profiles", "A"), "A") as a:
        cdp = a.cdp
        ok = await open_page(cdp, base_url + "/?lang=en")
        s0 = await cdp.evaluate(JS_STATE)
        step(record, "A: first visit shows the first-use screen and an empty history", ok and s0["landing"] and s0["emptyNote"] == S["en"]["no_conversations"] and s0["sessionTitles"] == [S["en"]["new_conversation"]], detail=s0["emptyNote"])
        await shot(cdp, shots, "create_A_1440_first_visit")

        # --- ask, record, reload, compare ---------------------------------------------------------------------------------
        await send_text(cdp, EN_QUESTION)
        saw_typing = await cdp.wait_for("!!document.querySelector('.typing')", timeout=10)
        answered = await wait_answer(cdp, 1)
        s1 = await cdp.evaluate(JS_STATE)
        step(record, "A: the question is answered with citations", answered and s1["userBubbles"] == 1 and s1["assistantBubbles"] == 1 and len(s1["chips"]) > 0 and not s1["savedNotes"],
             detail="processing=%s chips=%s" % (saw_typing, s1["chips"]))
        state["en_answer"], state["en_chips"] = s1["lastAnswer"], s1["chips"]
        state["en_title"] = s1["sessionTitles"][0] if s1["sessionTitles"] else None
        await shot(cdp, shots, "create_A_1440_fresh_answer")
        step(record, "A: the conversation is listed with a title made from the question", bool(state["en_title"]) and state["en_title"].startswith("What is the phone number"), detail=str(state["en_title"]))

        ok = await open_page(cdp, base_url + "/")
        s2 = await cdp.evaluate(JS_STATE)
        step(record, "A: after a reload the same conversation is open, replayed from the database",
             ok and s2["lastAnswer"] == state["en_answer"] and s2["chips"] == state["en_chips"] and s2["userBubbles"] == 1 and not s2["typing"],
             detail="same answer=%s same chips=%s" % (s2["lastAnswer"] == state["en_answer"], s2["chips"] == state["en_chips"]))
        step(record, "A: the replayed answer is labelled as saved", any(n.startswith(S["en"]["saved_answer"]) for n in s2["savedNotes"]), detail=str(s2["savedNotes"]))
        await shot(cdp, shots, "create_A_1440_after_reload")

        # --- rename through the dialog ------------------------------------------------------------------------------------
        opened = await open_menu_item(cdp, "phone number", S["en"]["rename"])
        s3 = await cdp.evaluate(JS_STATE)
        step(record, "A: Rename opens its dialog", opened and s3["dialogOpen"] and s3["dialogTitle"] == S["en"]["rename_title"], detail=str(s3["dialogTitle"]))
        await shot(cdp, shots, "create_A_1440_rename_dialog")
        await set_input(cdp, ".q-dialog input", NEW_TITLE)
        await click(cdp, ".q-dialog button", S["en"]["rename_ok"])
        await asyncio.sleep(1.0)
        s4 = await cdp.evaluate(JS_STATE)
        step(record, "A: the new title shows in the sidebar and the dialog closed", not s4["dialogOpen"] and NEW_TITLE in s4["sessionTitles"] and s4["help"] == S["en"]["renamed"], detail=str(s4["sessionTitles"]))
        await open_page(cdp, base_url + "/")
        s5 = await cdp.evaluate(JS_STATE)
        step(record, "A: the new title survives a reload", NEW_TITLE in s5["sessionTitles"], detail=str(s5["sessionTitles"]))

        # --- a second conversation in the Arabic interface -----------------------------------------------------------------
        await click(cdp, ".topbar-lang")
        await asyncio.sleep(1.2)
        await click(cdp, ".new-chat-btn")
        await asyncio.sleep(0.8)
        s6 = await cdp.evaluate(JS_STATE)
        step(record, "A: Arabic interface, New chat shows the Arabic first-use screen with the history kept", s6["lang"] == "ar" and s6["dir"] == "rtl" and s6["landing"] and NEW_TITLE in s6["sessionTitles"], detail=str(s6["sessionTitles"]))
        await send_text(cdp, AR_QUESTION)
        answered = await wait_answer(cdp, 1)
        s7 = await cdp.evaluate(JS_STATE)
        state["ar_answer"], state["ar_chips"], state["ar_translation"] = s7["lastAnswer"], s7["chips"], s7["translationNote"]
        state["ar_title"] = s7["activeTitle"]
        step(record, "A: the Arabic question is answered and listed under an Arabic title", answered and s7["assistantBubbles"] == 1 and bool(s7["chips"]) and (s7["activeTitle"] or "").startswith("ما هو"),
             detail="title=%s chips=%d translation=%s" % (s7["activeTitle"], len(s7["chips"]), (s7["translationNote"] or "")[:40]))
        await shot(cdp, shots, "create_A_1440_ar_fresh_answer")
        ok = await open_page(cdp, base_url + "/")
        s8 = await cdp.evaluate(JS_STATE)
        step(record, "A: a plain reload comes back in Arabic (saved interface language) on the Arabic conversation",
             ok and s8["lang"] == "ar" and s8["lastAnswer"] == state["ar_answer"] and s8["chips"] == state["ar_chips"] and s8["translationNote"] == state["ar_translation"],
             detail="lang=%s same answer=%s" % (s8["lang"], s8["lastAnswer"] == state["ar_answer"]))
        step(record, "A: the replayed Arabic answer is labelled as saved in Arabic", any(n.startswith(S["ar"]["saved_answer"]) for n in s8["savedNotes"]), detail=str(s8["savedNotes"]))
        await shot(cdp, shots, "create_A_1440_ar_after_reload")

        # --- settings: theme dark, persisted -------------------------------------------------------------------------------
        await click(cdp, ".topbar-settings")
        await asyncio.sleep(0.8)
        s9 = await cdp.evaluate(JS_STATE)
        step(record, "A: Settings opens (Arabic)", s9["dialogOpen"] and s9["dialogTitle"] == S["ar"]["settings_title"], detail=str(s9["dialogTitle"]))
        await shot(cdp, shots, "create_A_1440_ar_settings")
        picked = await pick_option(cdp, S["ar"]["setting_theme"], S["ar"]["theme_dark"])
        s10 = await cdp.evaluate(JS_STATE)
        step(record, "A: choosing the dark theme applies at once", picked and s10["theme"] == "dark", detail="theme=%s" % s10["theme"])
        await shot(cdp, shots, "create_A_1440_ar_settings_dark")
        await click(cdp, ".q-dialog button", S["ar"]["close"])
        await asyncio.sleep(0.6)
        await open_page(cdp, base_url + "/")
        s11 = await cdp.evaluate(JS_STATE)
        step(record, "A: the dark theme survives a reload", s11["theme"] == "dark" and s11["lang"] == "ar", detail="theme=%s lang=%s" % (s11["theme"], s11["lang"]))
        await shot(cdp, shots, "create_A_1440_ar_dark_after_reload")

        # --- the session cookie ----------------------------------------------------------------------------------------------
        jar = await cookies(cdp, base_url)
        session_cookie = next((c for c in jar if c["name"] == "session"), None)
        record["cookie"] = session_cookie
        step(record, "A: the session cookie is HttpOnly and SameSite=Lax with a bounded lifetime",
             bool(session_cookie) and session_cookie["httpOnly"] is True and str(session_cookie["sameSite"]).lower() == "lax" and (session_cookie["expires_days"] or 0) <= 14.5,
             detail=str(session_cookie))

        # --- widths, both languages ---------------------------------------------------------------------------------------
        record["viewports"] = []
        for lang in ("en", "ar"):
            for width, height in VIEWPORTS:
                vp = {"lang": lang, "width": width, "height": height, "checks": {}}
                record["viewports"].append(vp)
                ok = await open_page(cdp, base_url + "/?lang=" + lang, width, height)
                base = await cdp.evaluate(JS_CHECKS)
                st = await cdp.evaluate(JS_STATE)
                vp["checks"]["conversation"] = {**base, **st, "ready": ok}
                await shot(cdp, shots, "create_A_%s_%dx%d_conversation" % (lang, width, height))
                if not st["sidebarVisible"]:
                    await click(cdp, ".topbar-history")
                    await asyncio.sleep(0.9)
                    hist = await cdp.evaluate(JS_STATE)
                    vp["checks"]["history_dialog"] = hist
                    await shot(cdp, shots, "create_A_%s_%dx%d_history_dialog" % (lang, width, height))
                    await click(cdp, ".q-dialog button", S[lang]["close"])
                    await asyncio.sleep(0.5)
                await click(cdp, ".topbar-settings")
                await asyncio.sleep(0.9)
                vp["checks"]["settings_dialog"] = await cdp.evaluate(JS_STATE)
                await shot(cdp, shots, "create_A_%s_%dx%d_settings" % (lang, width, height))
                await click(cdp, ".q-dialog button", S[lang]["close"])
                await asyncio.sleep(0.4)
        await open_page(cdp, base_url + "/?lang=ar")                        # leave A on the Arabic conversation, dark theme

    # --- visitor B: an independent browser ----------------------------------------------------------------------------------
    async with Visitor(exe, os.path.join(out_dir, "profiles", "B"), "B") as b:
        cdp = b.cdp
        ok = await open_page(cdp, base_url + "/")
        sb = await cdp.evaluate(JS_STATE)
        leaked = [t for t in sb["sessionTitles"] if t in (NEW_TITLE, state.get("ar_title"), state.get("en_title"))]
        step(record, "B: a second browser sees no conversation of A (English defaults, empty history)",
             ok and sb["landing"] and sb["lang"] == "en" and sb["theme"] == "light" and sb["emptyNote"] == S["en"]["no_conversations"] and not leaked and sb["lastAnswer"] is None,
             detail="titles=%s theme=%s" % (sb["sessionTitles"], sb["theme"]))
        await shot(cdp, shots, "create_B_1440_first_visit")
        jar_b = await cookies(cdp, base_url)
        cookie_b = next((c for c in jar_b if c["name"] == "session"), None)
        record["cookie_b_differs"] = bool(cookie_b) and bool(record.get("cookie")) and cookie_b["value_length"] > 0
        step(record, "B: has its own session cookie", bool(cookie_b), detail=str(cookie_b))
    with open(os.path.join(out_dir, "evidence", "create_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)
    return state


# ---------------------------------------------------------------------------
# Phase 2: after the server restart
# ---------------------------------------------------------------------------
async def phase_restart(base_url: str, out_dir: str, exe: str, record: dict) -> None:
    shots = os.path.join(out_dir, "screenshots")
    with open(os.path.join(out_dir, "evidence", "create_state.json"), encoding="utf-8") as f:
        state = json.load(f)
    async with Visitor(exe, os.path.join(out_dir, "profiles", "A"), "A") as a:
        cdp = a.cdp
        ok = await open_page(cdp, base_url + "/")
        s0 = await cdp.evaluate(JS_STATE)
        step(record, "A: after the server restart the interface language, theme and active conversation come back",
             ok and s0["lang"] == "ar" and s0["theme"] == "dark" and s0["lastAnswer"] == state["ar_answer"] and set(s0["sessionTitles"]) >= {NEW_TITLE, state["ar_title"]},
             detail="lang=%s theme=%s titles=%s" % (s0["lang"], s0["theme"], s0["sessionTitles"]))
        await shot(cdp, shots, "restart_A_1440_ar_restored")

        # --- reopen the renamed English conversation: exact answer and citations, no processing state ----------------------
        clicked = await cdp.evaluate(
            "(() => { const row = [...document.querySelectorAll('.sidebar .session-row')].find(r => (r.querySelector('.session-title') || {}).innerText.includes(%s));"
            " if (!row) return false; row.querySelector('.session-item').click(); return true; })()" % json.dumps(NEW_TITLE))
        saw_typing = await cdp.wait_for("!!document.querySelector('.typing')", timeout=1.5)
        await cdp.wait_for("document.querySelectorAll('.bubble.assistant').length > 0", timeout=10)
        await asyncio.sleep(0.8)
        s1 = await cdp.evaluate(JS_STATE)
        digits = lambda chips: sorted(re.findall(r"\d+", " ".join(chips)))          # the cited pages, whatever the interface language calls them
        step(record, "A: the renamed conversation reopens with exactly the recorded answer and the same cited pages (Arabic interface)",
             clicked and s1["lastAnswer"] == state["en_answer"] and digits(s1["chips"]) == digits(state["en_chips"]) and s1["activeTitle"] == NEW_TITLE,
             detail="same answer=%s pages=%s chips(ar)=%s" % (s1["lastAnswer"] == state["en_answer"], digits(s1["chips"]), s1["chips"]))
        step(record, "A: reopening shows no processing state (nothing was asked again)", not saw_typing and any(n.startswith(S["ar"]["saved_answer"]) for n in s1["savedNotes"]), detail=str(s1["savedNotes"]))
        await shot(cdp, shots, "restart_A_1440_reopened_renamed_ar")
        await click(cdp, ".topbar-lang")                                        # the same stored turn in the English interface
        await asyncio.sleep(1.2)
        s1b = await cdp.evaluate(JS_STATE)
        step(record, "A: in the English interface the replayed citations equal the recorded ones exactly",
             s1b["lang"] == "en" and s1b["lastAnswer"] == state["en_answer"] and s1b["chips"] == state["en_chips"],
             detail="same chips=%s" % (s1b["chips"] == state["en_chips"]))
        await shot(cdp, shots, "restart_A_1440_reopened_renamed_en")
        await click(cdp, ".topbar-lang")                                        # back to Arabic for the rest of the phase
        await asyncio.sleep(1.2)

        # --- delete with confirmation ---------------------------------------------------------------------------------------
        opened = await open_menu_item(cdp, NEW_TITLE, S["ar"]["delete"])
        s2 = await cdp.evaluate(JS_STATE)
        step(record, "A: Delete asks for confirmation and names the conversation", opened and s2["dialogOpen"] and s2["dialogTitle"] == S["ar"]["delete_title"] and NEW_TITLE in (s2["dialogText"] or ""), detail=str(s2["dialogText"]))
        await shot(cdp, shots, "restart_A_1440_delete_dialog")
        await click(cdp, ".q-dialog button", S["ar"]["cancel"])
        await asyncio.sleep(0.8)
        s3 = await cdp.evaluate(JS_STATE)
        step(record, "A: Cancel keeps the conversation", not s3["dialogOpen"] and NEW_TITLE in s3["sessionTitles"] and s3["lastAnswer"] == state["en_answer"], detail=str(s3["sessionTitles"]))
        await open_menu_item(cdp, NEW_TITLE, S["ar"]["delete"])
        await click(cdp, ".q-dialog button", S["ar"]["delete_ok"])
        await asyncio.sleep(1.0)
        s4 = await cdp.evaluate(JS_STATE)
        step(record, "A: confirming removes it from the list and the page returns to the first-use screen",
             not s4["dialogOpen"] and NEW_TITLE not in s4["sessionTitles"] and s4["landing"] and s4["help"] == S["ar"]["deleted"], detail=str(s4["sessionTitles"]))
        await shot(cdp, shots, "restart_A_1440_after_delete")
        await open_page(cdp, base_url + "/")
        s5 = await cdp.evaluate(JS_STATE)
        step(record, "A: the deletion holds after a reload and the Arabic conversation remains", NEW_TITLE not in s5["sessionTitles"] and state["ar_title"] in s5["sessionTitles"], detail=str(s5["sessionTitles"]))

        # --- settings reset -------------------------------------------------------------------------------------------------
        await click(cdp, ".topbar-settings")
        await asyncio.sleep(0.8)
        await click(cdp, ".q-dialog button", S["ar"]["settings_reset"])
        await asyncio.sleep(1.2)
        s6 = await cdp.evaluate(JS_STATE)
        step(record, "A: Reset to defaults returns the light theme", s6["theme"] == "light" and not s6["dialogOpen"], detail="theme=%s help=%s" % (s6["theme"], s6["help"]))
        await open_page(cdp, base_url + "/")
        s7 = await cdp.evaluate(JS_STATE)
        step(record, "A: after the reset a reload follows the browser language again (English) with the light theme", s7["lang"] == "en" and s7["theme"] == "light", detail="lang=%s theme=%s" % (s7["lang"], s7["theme"]))
        await shot(cdp, shots, "restart_A_1440_en_after_reset")
        await open_page(cdp, base_url + "/?lang=ar")
        await shot(cdp, shots, "restart_A_1440_ar_final")

    async with Visitor(exe, os.path.join(out_dir, "profiles", "B"), "B") as b:
        cdp = b.cdp
        ok = await open_page(cdp, base_url + "/")
        sb = await cdp.evaluate(JS_STATE)
        step(record, "B: still sees nothing of A after the restart", ok and sb["landing"] and sb["emptyNote"] == S["en"]["no_conversations"] and state["ar_title"] not in sb["sessionTitles"], detail=str(sb["sessionTitles"]))
        await shot(cdp, shots, "restart_B_1440_still_empty")


def summarize(record: dict) -> int:
    failures = sum(1 for s in record["steps"] if not s["ok"])
    for vp in record.get("viewports", []):
        lang, w, h = vp["lang"], vp["width"], vp["height"]
        for name, c in vp["checks"].items():
            problems = []
            if c.get("horizontalOverflow"):
                problems.append("horizontal overflow")
            if name == "conversation":
                if not (c.get("textarea") or {}).get("inside") or not (c.get("send") or {}).get("inside"):
                    problems.append("composer not fully visible")
                if not (c.get("sidebarVisible") or c.get("historyButtonVisible")):
                    problems.append("no reachable history control")
                if not c.get("settingsButtonVisible"):
                    problems.append("settings control not visible")
                if c.get("wideElements"):
                    problems.append("elements wider than the viewport: %s" % c["wideElements"][:3])
                if lang == "ar" and c.get("dir") != "rtl":
                    problems.append("Arabic page not rtl")
            if name == "history_dialog" and (not c.get("dialogOpen") or not c.get("dialogSessions") or c.get("dialogOverflow")):
                problems.append("history dialog missing, empty or clipped")
            if name == "settings_dialog" and (not c.get("dialogOpen") or c.get("dialogOverflow")):
                problems.append("settings dialog missing or clipped")
            failures += len(problems)
            print("%s %4dx%-4d %-16s %s" % (lang, w, h, name, "OK" if not problems else "; ".join(problems)))
    print("problems:", failures)
    return failures


async def main(args) -> int:
    exe = find_browser(args.browser)
    os.makedirs(os.path.join(args.out_dir, "evidence"), exist_ok=True)
    record = {"phase": args.phase, "browser": exe, "base_url": args.base_url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}
    if args.phase == "create":
        await phase_create(args.base_url, args.out_dir, exe, record)
    else:
        await phase_restart(args.base_url, args.out_dir, exe, record)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.out_dir, "evidence", "%s_persistence_browser_qa.json" % args.phase), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return summarize(record)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_dir")
    ap.add_argument("--phase", choices=("create", "restart"), required=True)
    ap.add_argument("--browser", choices=("chrome", "edge"), default="chrome")
    a = ap.parse_args()
    a.base_url = a.base_url.rstrip("/")
    sys.exit(1 if asyncio.run(main(a)) else 0)
