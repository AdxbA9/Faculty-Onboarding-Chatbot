"""Real-browser QA of the PF-15 web platform: workflows A-H, responsive and accessibility checks, a real restart.

Headless Chrome 157 over the DevTools protocol against the RUNNING application (viewport emulation; no physical phone),
two persistent profiles (visitor A, visitor B), two phases around a real server stop/start:

    python -X utf8 qa/pf15_web/browser_qa_platform.py <base url> <output dir> --phase create
    ... stop the server, start it again ...
    python -X utf8 qa/pf15_web/browser_qa_platform.py <base url> <output dir> --phase restart

create:  A chat (ask, sources, local times, reload, history, reopen) · B settings (theme, text size, motion, answer language;
         applied and measured) · C profile (display name: invalid refused, valid saved and shown in the navigation) ·
         D onboarding (checklist -> procedure -> finder -> back -> saved progress; procedures index position) · E history
         (rename and delete through the dialogs) · F language (Arabic across all seven pages, state kept) · G responsive
         (7 pages x 5 widths x 2 languages: overflow, drawer open/close, touch targets, composer) · accessibility (skip
         link, one h1, named controls, keyboard focus ring, drawer focus and Escape) · H visitor B sees none of A's data.
restart: A's settings, name, progress, procedure position and conversation survive; B still sees nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pf10_ui"))
sys.path.insert(0, os.path.join(HERE, "..", "pf13_db"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from browser_qa import VIEWPORTS, find_browser, shot                                    # noqa: E402
from browser_qa_persistence import Visitor, click, step                                 # noqa: E402
from ui import i18n                                                                     # noqa: E402

S = i18n.STRINGS
HR_QUESTION = "What is the phone number of the Human Resources Department?"
TRAVEL_QUESTION = "Who approves conference travel requests?"
PAGES = {"chat": "/", "onboarding": "/onboarding", "procedures": "/procedures", "resources": "/resources", "history": "/history", "profile": "/profile", "settings": "/settings"}
READY = "!!document.querySelector('.composer textarea') || !!document.querySelector('.page-main')"

JS_STATE = r"""
(() => {
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  const text = el => el ? el.innerText.replace(/\s+/g, ' ').trim() : null;
  const visible = el => { if (!el) return false; const cs = getComputedStyle(el); const r = el.getBoundingClientRect(); return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0 && r.height > 0; };
  const vw = document.documentElement.clientWidth;
  const wide = qa('.main *, .sidebar *').filter(el => { if (!visible(el)) return false; const b = el.getBoundingClientRect(); const cs = getComputedStyle(el);
      return (b.right > vw + 1 || b.left < -1) && cs.position !== 'fixed' && !el.closest('.sidebar') || (el.closest('.app-shell.nav-open .sidebar') && b.right > vw + 1); })
      .map(el => String(el.className).slice(0, 40) + ' ' + Math.round(el.getBoundingClientRect().left) + '..' + Math.round(el.getBoundingClientRect().right));
  const html = document.documentElement;
  const unnamed = qa('button, a[href], [role=button], input, textarea').filter(el => visible(el) && !(el.getAttribute('aria-label') || text(el) || el.getAttribute('title') || el.getAttribute('placeholder') || (el.labels && el.labels.length))).map(el => el.outerHTML.slice(0, 80));
  const toggle = q('.topbar .nav-toggle');
  const small = qa('.topbar button, .composer button, .nav-item, .hs-actions button, .seg button').filter(visible).filter(el => { const r = el.getBoundingClientRect(); return r.width < 32 || r.height < 32; }).map(el => (el.getAttribute('aria-label') || text(el) || '').slice(0, 30));
  return {
    path: location.pathname + location.search, lang: html.lang, dir: getComputedStyle(html).direction, theme: html.getAttribute('data-theme'),
    text: html.getAttribute('data-text'), motion: html.getAttribute('data-motion'), clientWidth: vw, scrollX: window.scrollX,
    overflow: document.documentElement.scrollWidth > vw + 1 || Math.abs(window.scrollX) > 1, wide: wide.slice(0, 6),
    h1: qa('h1').filter(visible).map(text), sidebarVisible: visible(q('.sidebar')), drawerOpen: !!q('.app-shell.nav-open'),
    toggleVisible: visible(toggle), toggleExpanded: toggle ? toggle.getAttribute('aria-expanded') : null,
    navItems: qa('#app-nav .nav-item').map(a => ({href: a.getAttribute('href'), current: a.getAttribute('aria-current'), label: text(a)})),
    activeNav: (q('#app-nav .nav-item[aria-current="page"]') || {}).getAttribute ? q('#app-nav .nav-item[aria-current="page"]').getAttribute('href') : null,
    skipLink: !!q('.skip-link'), unnamed: unnamed.slice(0, 5), smallTargets: small.slice(0, 6),
    guestChip: text(q('.guest-chip .guest-name')), composer: visible(q('.composer textarea')), send: visible(q('.send-btn')), mic: visible(q('.mic-btn')),
    sendClipped: (() => { const b = q('.send-btn'); if (!b) return null; const r = b.getBoundingClientRect(); return [...b.querySelectorAll('*')].some(c => { const cr = c.getBoundingClientRect(); return cr.height > 0 && (cr.bottom > r.bottom + 2 || cr.top < r.top - 2); }); })(),
    titleShown: (() => { const t = q('.topbar .topbar-title'); return t ? {text: text(t), width: Math.round(t.getBoundingClientRect().width)} : null; })(),
    sessionOverflow: qa('.sidebar .session-row').some(r => { const b = r.getBoundingClientRect(); const s = q('.sidebar').getBoundingClientRect(); return visible(r) && (b.right > s.right + 1 || b.left < s.left - 1); }),
    lastAnswer: (() => { const b = qa('.bubble.assistant'); return b.length ? text(b[b.length - 1]) : null; })(), typing: !!q('.typing'),
    tels: qa('.answer .tel').map(t => ({text: text(t), dir: t.getAttribute('dir'), ws: getComputedStyle(t).whiteSpace})),
    chips: qa('.sources .chip').map(text), times: qa('time[data-local]').map(t => ({done: t.getAttribute('data-done'), text: text(t)})),
    savedNote: text(q('.saved-note')), note: text(q('.page-note')), hsRows: qa('.hs-row').length, hsTitles: qa('.hs-title-text').map(text),
    profileName: text(q('.profile-name')), stats: Object.fromEntries(qa('[data-stat]').map(d => [d.getAttribute('data-stat'), Number(d.innerText)])),
    taskStatus: (() => { const c = document.getElementById('task-hr_documents'); return c ? text(c.querySelector('.ob-status-pill')) : null; })(),
    pxPositions: qa('.px-position').map(text).filter(Boolean), stepCounter: text(q('.pr-counter')),
    dialogTitle: text(q('.q-dialog .dialog-title')), sizes: {pageSub: q('.page-sub') ? getComputedStyle(q('.page-sub')).fontSize : null, nav: q('.nav-item') ? getComputedStyle(q('.nav-item')).fontSize : null,
      bubble: q('.bubble') ? getComputedStyle(q('.bubble')).fontSize : null},
    navTransition: q('.nav-item') ? getComputedStyle(q('.nav-item')).transitionDuration : null,
  };
})()
"""


async def open_page(cdp, url: str, width: int = 1440, height: int = 900) -> bool:
    await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 768})
    await cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": width < 768})
    await cdp.send("Page.navigate", {"url": url})
    ok = await cdp.wait_for(READY, timeout=60)
    await asyncio.sleep(1.1)
    return ok


async def ask(cdp, question: str) -> bool:
    await cdp.evaluate("document.querySelector('.composer textarea').focus()")
    await cdp.send("Input.insertText", {"text": question})
    await asyncio.sleep(0.3)
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
    ok = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > 0", timeout=120)
    await asyncio.sleep(1.0)
    return ok


async def toggle_option(cdp, label: str, option: str) -> bool:
    ok = await cdp.evaluate(
        "(() => { const g = [...document.querySelectorAll('.seg')].find(g => g.getAttribute('aria-label') === %s); if (!g) return false;"
        " const b = [...g.querySelectorAll('button')].find(b => b.innerText.trim() === %s); if (!b) return false; b.click(); return true; })()" % (json.dumps(label), json.dumps(option)))
    await asyncio.sleep(0.9)
    return ok


async def type_into(cdp, selector: str, text: str) -> None:
    await cdp.evaluate("(() => { const el = document.querySelector(%s); el.focus(); const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; s.call(el, '');"
                       " el.dispatchEvent(new Event('input', {bubbles: true})); })()" % json.dumps(selector))
    await cdp.send("Input.insertText", {"text": text})
    await asyncio.sleep(0.5)


async def key(cdp, name: str, code: str, vk: int, modifiers: int = 0) -> None:
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": name, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk, "modifiers": modifiers})
    await asyncio.sleep(0.25)


async def state(cdp) -> dict:
    return await cdp.evaluate(JS_STATE)


# ---------------------------------------------------------------------------
# Phase create
# ---------------------------------------------------------------------------
async def phase_create(base: str, out: str, exe: str, record: dict) -> None:
    shots = os.path.join(out, "screenshots", "workflows")
    os.makedirs(shots, exist_ok=True)
    keep: dict = {}
    async with Visitor(exe, os.path.join(out, "profiles", "A"), "A") as a:
        cdp = a.cdp
        # ---- A: chat --------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/?lang=en")
        s0 = await state(cdp)
        step(record, "A1 the chat is the default page, with the navigation and the chat item current", s0["activeNav"] == "/" and len(s0["navItems"]) == 7 and s0["composer"], detail="nav=%s" % [n["href"] for n in s0["navItems"]])
        ok = await ask(cdp, HR_QUESTION)
        s1 = await state(cdp)
        keep["hr_answer"], keep["hr_chips"] = s1["lastAnswer"], s1["chips"]
        step(record, "A2 a supported question is answered with its sources and the specialist chip", ok and bool(s1["lastAnswer"]) and any("Verified" in c for c in s1["chips"]) and any("Page 54" in c for c in s1["chips"]),
             detail="chips=%s" % s1["chips"])
        step(record, "A3 message times are rendered in local time by the browser", bool(s1["times"]) and all(t["done"] == "1" and "UTC" not in (t["text"] or "") for t in s1["times"]), detail=str(s1["times"][:2]))
        step(record, "A3b phone numbers in the answer are kept whole and left to right; the visible answer text is unchanged",
             [t["text"] for t in s1["tels"]] == ["+971 6 5050023", "+971 6 5585200"] and all(t["dir"] == "ltr" and t["ws"] == "nowrap" for t in s1["tels"])
             and "Human Resources +971 6 5050023 +971 6 5585200" in (s1["lastAnswer"] or ""), detail=str(s1["tels"]))
        await shot(cdp, shots, "A_chat_answer_en_1440")
        await open_page(cdp, base + "/")
        s2 = await state(cdp)
        step(record, "A4 after a reload the saved answer is replayed with the same sources and the Saved label", s2["lastAnswer"] == keep["hr_answer"] and s2["chips"] == keep["hr_chips"] and (s2["savedNote"] or "").startswith("Saved answer") and not s2["typing"],
             detail="saved=%s" % s2["savedNote"])
        await click(cdp, ".new-chat-btn")
        await asyncio.sleep(0.8)
        await ask(cdp, TRAVEL_QUESTION)
        await open_page(cdp, base + "/history")
        s3 = await state(cdp)
        step(record, "A5 the history page lists both conversations, newest first, in a time group", s3["hsRows"] == 2 and (s3["hsTitles"][0] or "").startswith("Who approves") and s3["activeNav"] == "/history",
             detail=str(s3["hsTitles"]))
        await shot(cdp, shots, "A_history_en_1440")
        await cdp.evaluate("(() => { const b = [...document.querySelectorAll('.hs-title')].find(b => b.innerText.includes('phone number')); if (b) b.click(); })()")
        await cdp.wait_for("!!document.querySelector('.composer textarea') && document.querySelectorAll('.bubble.assistant').length > 0", timeout=30)
        await asyncio.sleep(1.0)
        s4 = await state(cdp)
        step(record, "A6 opening a conversation from the history returns to the chat on that conversation, replayed, not re-asked", s4["lastAnswer"] == keep["hr_answer"] and not s4["typing"] and s4["path"] == "/",
             detail="path=%s" % s4["path"])
        # ---- B: settings ------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/settings")
        sb0 = await state(cdp)
        keep["sub_size_standard"] = sb0["sizes"]["pageSub"]
        r1 = await toggle_option(cdp, S["en"]["st_theme"], S["en"]["theme_dark"])
        r2 = await toggle_option(cdp, S["en"]["st_text_size"], S["en"]["st_text_large"])
        r3 = await toggle_option(cdp, S["en"]["st_motion"], S["en"]["st_motion_on"])
        r4 = await toggle_option(cdp, S["en"]["st_answer_lang"], "العربية")
        sb1 = await state(cdp)
        step(record, "B1 theme, text size, motion and answer language change through the Settings controls", all((r1, r2, r3, r4)) and sb1["theme"] == "dark" and sb1["text"] == "large" and sb1["motion"] == "reduced",
             detail="theme=%s text=%s motion=%s note=%s" % (sb1["theme"], sb1["text"], sb1["motion"], sb1["note"]))
        step(record, "B2 large text measurably enlarges the text; reduced motion removes transitions", sb1["sizes"]["pageSub"] != keep["sub_size_standard"] and sb1["navTransition"] in ("0s", "0s, 0s"),
             detail="page-sub %s -> %s, nav transition %s" % (keep["sub_size_standard"], sb1["sizes"]["pageSub"], sb1["navTransition"]))
        await shot(cdp, shots, "B_settings_dark_large_en_1440")
        await open_page(cdp, base + "/settings")
        sb2 = await state(cdp)
        step(record, "B3 after a reload the settings are still applied", sb2["theme"] == "dark" and sb2["text"] == "large" and sb2["motion"] == "reduced", detail="theme=%s text=%s" % (sb2["theme"], sb2["text"]))
        await open_page(cdp, base + "/")
        sb3 = await state(cdp)
        step(record, "B4 the chat page applies the same settings (large text in the bubbles)", sb3["theme"] == "dark" and sb3["text"] == "large" and sb3["sizes"]["bubble"] == "17.5px", detail="bubble=%s" % sb3["sizes"]["bubble"])
        await shot(cdp, shots, "B_chat_dark_large_en_1440")
        # ---- C: profile ---------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/profile")
        await type_into(cdp, "input[aria-label='%s']" % S["en"]["pf_name"], "sara@sharjah.ac.ae")
        await click(cdp, ".name-form button", S["en"]["pf_save_name"])
        await asyncio.sleep(0.8)
        sc0 = await state(cdp)
        step(record, "C1 an email address as display name is refused with a message", (sc0["note"] or "").startswith("Use up to 40 characters") and sc0["guestChip"] == S["en"]["guest_label"], detail=str(sc0["note"]))
        await type_into(cdp, "input[aria-label='%s']" % S["en"]["pf_name"], "Dr. Sara")
        await click(cdp, ".name-form button", S["en"]["pf_save_name"])
        await asyncio.sleep(0.9)
        sc1 = await state(cdp)
        step(record, "C2 a valid display name is saved and shown in the profile and the navigation", sc1["profileName"] == "Dr. Sara" and sc1["guestChip"] == "Dr. Sara" and (sc1["note"] or "") == S["en"]["pf_name_saved"],
             detail="hero=%s chip=%s" % (sc1["profileName"], sc1["guestChip"]))
        await shot(cdp, shots, "C_profile_en_1440")
        # ---- D: onboarding --------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/onboarding")
        await cdp.evaluate("(() => { const c = document.getElementById('task-hr_documents'); const b = [...c.querySelectorAll('button')].find(b => b.innerText.includes('Mark completed')); b.click(); })()")
        await asyncio.sleep(1.2)
        sd0 = await state(cdp)
        step(record, "D1 marking a task complete updates the card and the summary", sd0["taskStatus"] == S["en"]["status_completed"] and sd0["stats"].get("completed") == 1, detail=str(sd0["stats"]))
        await cdp.evaluate("(() => { const c = document.getElementById('task-bb_announcement'); const b = [...c.querySelectorAll('button')].find(b => b.innerText.includes('Open guide')); b.click(); })()")
        await cdp.wait_for("!!document.querySelector('.pr-card')", timeout=30)
        await asyncio.sleep(0.8)
        for _ in range(2):
            await click(cdp, ".pr-controls button", S["en"]["pr_next"])
            await asyncio.sleep(0.5)
        sd1 = await state(cdp)
        step(record, "D2 the guided procedure opens from the task and moves to step 3 (Guided procedures is the current item)", (sd1["stepCounter"] or "").lower() == "step 3 of 8" and sd1["activeNav"] == "/procedures", detail=str(sd1["stepCounter"]))
        await click(cdp, ".pr-evidence a.ob-link")
        await cdp.wait_for("!!document.querySelector('.rf-results')", timeout=30)
        await asyncio.sleep(0.8)
        sd2 = await state(cdp)
        step(record, "D3 the evidence link opens the finder (Resources & documents current)", sd2["activeNav"] == "/resources", detail=sd2["path"])
        await cdp.evaluate("history.back()")
        await cdp.wait_for("!!document.querySelector('.pr-card')", timeout=30)
        await asyncio.sleep(0.8)
        sd3 = await state(cdp)
        step(record, "D4 the browser back button returns to the procedure at the same step", (sd3["stepCounter"] or "").lower() == "step 3 of 8", detail=str(sd3["stepCounter"]))
        await open_page(cdp, base + "/onboarding")
        sd4 = await state(cdp)
        step(record, "D5 back on the checklist the saved progress is shown", sd4["taskStatus"] == S["en"]["status_completed"] and sd4["stats"].get("completed") == 1, detail=str(sd4["stats"]))
        await open_page(cdp, base + "/procedures")
        sd5 = await state(cdp)
        step(record, "D6 the procedures index shows where the guest left the announcement guide", "Continue at step 3 of 8" in sd5["pxPositions"], detail=str(sd5["pxPositions"]))
        await shot(cdp, shots, "D_procedures_index_en_1440")
        # ---- E: history management ----------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/history")
        await cdp.evaluate("(() => { const row = [...document.querySelectorAll('.hs-row')].find(r => r.innerText.includes('phone number')); row.querySelector('.hs-actions button').click(); })()")
        await asyncio.sleep(0.9)
        se0 = await state(cdp)
        await type_into(cdp, ".q-dialog input", "HR phone numbers")
        await click(cdp, ".q-dialog button", S["en"]["rename_ok"])
        await asyncio.sleep(1.0)
        se1 = await state(cdp)
        step(record, "E1 rename through the dialog updates the history", se0["dialogTitle"] == S["en"]["rename_title"] and "HR phone numbers" in se1["hsTitles"] and se1["note"] == S["en"]["renamed"], detail=str(se1["hsTitles"]))
        await cdp.evaluate("(() => { const row = [...document.querySelectorAll('.hs-row')].find(r => r.innerText.includes('Who approves')); row.querySelectorAll('.hs-actions button')[1].click(); })()")
        await asyncio.sleep(0.9)
        se2 = await state(cdp)
        await shot(cdp, shots, "E_history_delete_confirm_en_1440")
        await click(cdp, ".q-dialog button", S["en"]["delete_ok"])
        await asyncio.sleep(1.0)
        se3 = await state(cdp)
        step(record, "E2 delete asks for confirmation and removes the conversation", se2["dialogTitle"] == S["en"]["delete_title"] and se3["hsRows"] == 1 and se3["note"] == S["en"]["deleted"], detail=str(se3["hsTitles"]))
        # ---- F: Arabic across the platform ---------------------------------------------------------------------------------------
        await open_page(cdp, base + "/settings")
        await click(cdp, ".topbar-lang")
        await cdp.wait_for("document.documentElement.lang === 'ar'", timeout=20)
        await asyncio.sleep(1.0)
        problems = []
        for page_key, path in PAGES.items():
            await open_page(cdp, base + path)
            sf = await state(cdp)
            labels = [n["label"] for n in sf["navItems"]]
            if sf["dir"] != "rtl" or sf["lang"] != "ar" or labels[0] != S["ar"]["nav_chat"] or sf["guestChip"] != "Dr. Sara":
                problems.append((page_key, sf["dir"], labels[:1], sf["guestChip"]))
            await shot(cdp, shots, "F_%s_ar_1440" % page_key)
        sf_onb = None
        await open_page(cdp, base + "/onboarding")
        sf_onb = await state(cdp)
        step(record, "F1 every page switches to Arabic right-to-left with the same data (name, progress)", not problems and sf_onb["stats"].get("completed") == 1, detail=str(problems or sf_onb["stats"]))
        await click(cdp, ".topbar-lang")
        await cdp.wait_for("document.documentElement.lang === 'en'", timeout=20)
        await asyncio.sleep(0.8)
        # ---- accessibility -------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/settings")
        await key(cdp, "Tab", "Tab", 9)
        focus0 = await cdp.evaluate("({cls: document.activeElement.className, text: document.activeElement.innerText.trim().slice(0, 40)})")
        await key(cdp, "Tab", "Tab", 9)
        await key(cdp, "Tab", "Tab", 9)
        focus1 = await cdp.evaluate("(() => { const el = document.activeElement; const cs = getComputedStyle(el); return {tag: el.tagName, cls: String(el.className).slice(0, 40), outline: cs.outlineStyle, width: cs.outlineWidth}; })()")
        step(record, "K1 the first Tab reaches the skip link; keyboard focus shows a visible ring", "skip-link" in str(focus0["cls"]) and focus1["outline"] not in ("none", None), detail="first=%s focus=%s" % (focus0, focus1))
        a11y = []
        for key_name, path in PAGES.items():
            await open_page(cdp, base + path)
            sk = await state(cdp)
            if len(sk["h1"]) != 1 or sk["unnamed"] or not sk["skipLink"]:
                a11y.append((key_name, sk["h1"], sk["unnamed"][:2], sk["skipLink"]))
        step(record, "K2 every page has a skip link, exactly one visible h1 and no unnamed control", not a11y, detail=str(a11y))
        # ---- G: responsive, both languages, drawer -----------------------------------------------------------------------------
        record["viewports"] = []
        for lang in ("en", "ar"):
            for width, height in VIEWPORTS:
                for key_name, path in PAGES.items():
                    sep = "&" if "?" in path else "?"
                    await open_page(cdp, base + path + sep + "lang=" + lang, width, height)
                    c = await state(cdp)
                    entry = {"lang": lang, "width": width, "height": height, "page": key_name, "checks": c}
                    if width < 900:
                        await cdp.evaluate("document.querySelector('.topbar .nav-toggle').click()")
                        await asyncio.sleep(0.6)
                        opened = await state(cdp)
                        entry["drawer"] = {"open": opened["drawerOpen"], "sidebar": opened["sidebarVisible"], "expanded": opened["toggleExpanded"], "overflow": opened["overflow"],
                                           "focus": await cdp.evaluate("document.activeElement && document.activeElement.className")}
                        if key_name in ("chat", "settings") and width in (375, 768):
                            await shot(cdp, os.path.join(out, "screenshots", "workflows"), "G_drawer_%s_%dx%d_%s" % (lang, width, height, key_name))
                        await key(cdp, "Escape", "Escape", 27)
                        closed = await state(cdp)
                        entry["drawer"]["closedByEscape"] = not closed["drawerOpen"]
                        entry["drawer"]["focusReturned"] = await cdp.evaluate("document.activeElement && document.activeElement.classList.contains('nav-toggle')")
                    record["viewports"].append(entry)
                    if width in (320, 375, 1440) and key_name in ("chat", "settings", "profile", "history", "procedures"):
                        await shot(cdp, os.path.join(out, "screenshots", "workflows"), "G_%s_%dx%d_%s" % (lang, width, height, key_name))
        await open_page(cdp, base + "/?lang=en")
    # ---- H: a second, independent browser ----------------------------------------------------------------------------------------
    async with Visitor(exe, os.path.join(out, "profiles", "B"), "B") as b:
        cdp = b.cdp
        await open_page(cdp, base + "/history?lang=en")
        h0 = await state(cdp)
        await open_page(cdp, base + "/profile?lang=en")
        h1 = await state(cdp)
        await open_page(cdp, base + "/onboarding?lang=en")
        h2 = await state(cdp)
        step(record, "H1 visitor B sees no conversation, no name, no progress and the default settings of A",
             h0["hsRows"] == 0 and h1["profileName"] == S["en"]["guest_label"] and h1["guestChip"] == S["en"]["guest_label"] and h2["stats"].get("completed") == 0 and h2["theme"] == "light" and h2["text"] == "standard",
             detail="rows=%s name=%s completed=%s theme=%s" % (h0["hsRows"], h1["profileName"], h2["stats"].get("completed"), h2["theme"]))
        await shot(cdp, shots, "H_visitor_B_profile_en_1440")
    with open(os.path.join(out, "evidence", "create_keep.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Phase restart
# ---------------------------------------------------------------------------
async def phase_restart(base: str, out: str, exe: str, record: dict) -> None:
    shots = os.path.join(out, "screenshots", "workflows")
    keep = json.load(open(os.path.join(out, "evidence", "create_keep.json"), encoding="utf-8"))
    async with Visitor(exe, os.path.join(out, "profiles", "A"), "A") as a:
        cdp = a.cdp
        await open_page(cdp, base + "/")
        r0 = await state(cdp)
        step(record, "R1 after the restart the chat reopens the conversation with the same answer and sources, with the settings applied",
             r0["lastAnswer"] == keep["hr_answer"] and r0["chips"] == keep["hr_chips"] and r0["theme"] == "dark" and r0["text"] == "large" and r0["motion"] == "reduced" and not r0["typing"],
             detail="theme=%s text=%s motion=%s" % (r0["theme"], r0["text"], r0["motion"]))
        await shot(cdp, shots, "R_chat_restored_en_1440")
        await open_page(cdp, base + "/profile")
        r1 = await state(cdp)
        step(record, "R2 the display name survives the restart", r1["profileName"] == "Dr. Sara" and r1["guestChip"] == "Dr. Sara", detail=str(r1["profileName"]))
        await open_page(cdp, base + "/onboarding")
        r2 = await state(cdp)
        await open_page(cdp, base + "/procedures")
        r3 = await state(cdp)
        step(record, "R3 progress and the procedure position survive the restart", r2["stats"].get("completed") == 1 and "Continue at step 3 of 8" in r3["pxPositions"], detail="%s %s" % (r2["stats"], r3["pxPositions"]))
        await open_page(cdp, base + "/history")
        r4 = await state(cdp)
        step(record, "R4 the renamed conversation is the only one left", r4["hsTitles"] == ["HR phone numbers"], detail=str(r4["hsTitles"]))
        await open_page(cdp, base + "/settings")
        await toggle_option(cdp, S["en"]["st_theme"], S["en"]["theme_light"])
        await toggle_option(cdp, S["en"]["st_text_size"], S["en"]["st_text_standard"])
        await toggle_option(cdp, S["en"]["st_motion"], S["en"]["st_motion_off"])
        r5 = await state(cdp)
        step(record, "R5 the settings can be set back to light, standard text and full motion", r5["theme"] == "light" and r5["text"] == "standard" and r5["motion"] == "full", detail=str((r5["theme"], r5["text"], r5["motion"])))
    async with Visitor(exe, os.path.join(out, "profiles", "B"), "B") as b:
        cdp = b.cdp
        await open_page(cdp, base + "/profile?lang=en")
        rb = await state(cdp)
        await open_page(cdp, base + "/history?lang=en")
        rb2 = await state(cdp)
        step(record, "R6 visitor B still sees nothing of A after the restart", rb["profileName"] == S["en"]["guest_label"] and rb2["hsRows"] == 0, detail="%s %s" % (rb["profileName"], rb2["hsRows"]))


def summarize(record: dict) -> int:
    failures = sum(1 for s in record["steps"] if not s["ok"])
    for vp in record.get("viewports", []):
        c = vp["checks"]
        problems = []
        if c["overflow"] or c["wide"]:
            problems.append("overflow %s" % c["wide"][:2])
        if vp["width"] >= 900 and not c["sidebarVisible"]:
            problems.append("sidebar hidden on desktop")
        if vp["width"] < 900 and (c["sidebarVisible"] or not c["toggleVisible"]):
            problems.append("menu button / drawer state wrong")
        if vp["lang"] == "ar" and c["dir"] != "rtl":
            problems.append("not rtl")
        if c["activeNav"] != {"chat": "/", "onboarding": "/onboarding", "procedures": "/procedures", "resources": "/resources", "history": "/history", "profile": "/profile", "settings": "/settings"}[vp["page"]]:
            problems.append("active item %s" % c["activeNav"])
        if c["smallTargets"]:
            problems.append("small targets %s" % c["smallTargets"][:3])
        if c["unnamed"]:
            problems.append("unnamed %s" % c["unnamed"][:2])
        if vp["page"] == "chat" and (not c["composer"] or not c["send"] or not c["mic"] or c["sendClipped"] or c["sessionOverflow"]):
            problems.append("composer/send/mic/session problem")
        if vp["page"] == "chat" and c["titleShown"] and vp["width"] >= 375 and c["titleShown"]["width"] < 90:
            problems.append("title squeezed to %spx" % c["titleShown"]["width"])
        d = vp.get("drawer")
        if d is not None and not (d["open"] and d["sidebar"] and d["expanded"] == "true" and not d["overflow"] and d["closedByEscape"] and d["focusReturned"]):
            problems.append("drawer %s" % d)
        failures += len(problems)
        print("%s %4dx%-4d %-10s %s" % (vp["lang"], vp["width"], vp["height"], vp["page"], "OK" if not problems else "; ".join(problems)))
    print("problems:", failures)
    return failures


async def main(args) -> int:
    exe = find_browser(args.browser)
    os.makedirs(os.path.join(args.out_dir, "evidence"), exist_ok=True)
    record = {"phase": args.phase, "browser": exe, "base_url": args.base_url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": [],
              "environment": "headless Chrome 157 via the DevTools protocol; phone and tablet sizes by viewport emulation (mobile flag and touch emulation), no physical device"}
    if args.phase == "create":
        await phase_create(args.base_url, args.out_dir, exe, record)
    else:
        await phase_restart(args.base_url, args.out_dir, exe, record)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.out_dir, "evidence", "%s_platform_browser_qa.json" % args.phase), "w", encoding="utf-8") as f:
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
