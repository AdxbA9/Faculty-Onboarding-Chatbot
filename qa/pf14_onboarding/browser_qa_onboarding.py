"""Real-browser QA of the three onboarding features (post-freeze PF-14).

Headless Chrome over the DevTools protocol against the RUNNING application, two persistent profiles (visitor A, visitor
B), two phases around a real server restart (same pattern as PF-13):

    python -X utf8 qa/pf14_onboarding/browser_qa_onboarding.py <base url> <output dir> --phase create
    ... stop the server, start it again ...
    python -X utf8 qa/pf14_onboarding/browser_qa_onboarding.py <base url> <output dir> --phase restart

Phase ``create`` runs three workflows with visitor A (Teaching: dashboard → guided announcement procedure → "Ask about this
step" → chat answer → back to the task → finder with task context → mark completed; Faculty services / HR: filter →
task → finder shows the printed HR contact → ask the assistant → status in progress; Institutional: finder search →
ILHE → related task → mark completed), then checks the dashboard, a procedure and the finder at 320/375/768/1024/1440 px
in English and Arabic (no horizontal overflow, navigation and controls reachable), and opens the pages as visitor B (no
progress of A). Phase ``restart`` verifies with visitor A that statuses, the procedure position and the saved chat
context survive the restart, and that visitor B still sees nothing. Screenshots and a JSON record go to the output dir.
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
sys.path.insert(0, os.path.join(HERE, "..", "pf13_db"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from browser_qa import VIEWPORTS, find_browser, shot                                   # noqa: E402
from browser_qa_persistence import Visitor, click, step                                # noqa: E402
from onboarding import catalog as C                                                    # noqa: E402
from ui import i18n                                                                    # noqa: E402

S = i18n.STRINGS

JS_PAGE = r"""
(() => {
  const q = s => document.querySelector(s), qa = s => [...document.querySelectorAll(s)];
  const text = el => el ? el.innerText.replace(/\s+/g, ' ').trim() : null;
  const visible = el => { if (!el) return false; const cs = getComputedStyle(el); const r = el.getBoundingClientRect(); return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0 && r.height > 0; };
  const vw = document.documentElement.clientWidth;                                   // the layout viewport; innerWidth grows with overflowing content under mobile emulation
  const wide = qa('.ob-main *, .ob-topbar *').filter(el => { const b = el.getBoundingClientRect(); return (b.width > vw + 1 || b.right > vw + 1 || b.left < -1) && getComputedStyle(el).position !== 'fixed'; }).map(el => String(el.className).slice(0, 50) + ' ' + Math.round(el.getBoundingClientRect().left) + '..' + Math.round(el.getBoundingClientRect().right));
  const stat = name => { const el = q('[data-stat="' + name + '"]'); return el ? Number(el.innerText) : null; };
  return {
    lang: document.documentElement.getAttribute('lang'), dir: getComputedStyle(document.documentElement).direction, theme: document.documentElement.getAttribute('data-theme'),
    title: text(q('h1')), path: location.pathname + location.search, horizontalOverflow: document.documentElement.scrollWidth > vw + 1 || innerWidth > vw + 1 || Math.abs(window.scrollX) > 1,
    viewport: {innerWidth, clientWidth: vw, scrollWidth: document.documentElement.scrollWidth, scrollX: window.scrollX},
    wideElements: wide.slice(0, 5), navLinks: qa('.ob-nav-link').filter(visible).length, navCurrent: text(q('.ob-nav-link[aria-current]')),
    percent: text(q('.ob-ring-value')), stats: {applicable: stat('applicable'), completed: stat('completed'), in_progress: stat('in_progress'), remaining: stat('remaining')},
    taskCards: qa('.ob-task').length, statuses: Object.fromEntries(qa('.ob-task').map(c => [c.id.replace('task-', ''), text(c.querySelector('.ob-status-pill'))])),
    note: text(q('.page-note')), nextItems: qa('.ob-next li').map(text),
    stepCounter: text(q('.pr-counter')), stepText: text(q('.pr-step-text')), stepFlag: text(q('.pr-flag')), prevDisabled: !!(q('.pr-controls .q-btn.disabled') && /Prev|السابق/.test(text(q('.pr-controls .q-btn.disabled')))),
    markDone: text(qa('.ob-btn.complete').find(b => visible(b))), taskBox: text(q('.pr-task-box h2')),
    resourceCards: qa('.rf-card').length, firstResource: text(q('.rf-card .rf-title')), firstContextual: !!(q('.rf-card') && q('.rf-card').classList.contains('contextual')),
    contacts: qa('.rf-contact-value').map(text), links: qa('a.res-link').map(a => ({href: a.getAttribute('href'), rel: a.getAttribute('rel'), target: a.getAttribute('target')})),
    firstContacts: q('.rf-card') ? [...q('.rf-card').querySelectorAll('.rf-contact-value')].map(text) : [], firstContactNote: q('.rf-card') ? text(q('.rf-card .rf-contact-note')) : null,
    resourceTitles: qa('.rf-card .rf-title').map(text),
    contextBox: text(q('.rf-context-box')), resultCount: text(q('.rf-count')), emptyState: text(q('.ob-empty')),
    composer: (() => { const t = q('.composer textarea'); return t ? t.value : null; })(), chatContext: text(q('.chat-context')),
    assistantBubbles: qa('.bubble.assistant').length, lastAnswer: (() => { const b = qa('.bubble.assistant'); return b.length ? text(b[b.length - 1]) : null; })(),
    chips: qa('.chip').map(text), typing: !!q('.typing'), userBubbles: qa('.msg-row.user').length,
    heldSourceMentioned: /Performance Appraisal/i.test(document.body.innerText),
  };
})()
"""


async def open_page(cdp, url: str, width: int = 1440, height: int = 900, wait: str = "!!document.querySelector('.ob-main') || !!document.querySelector('.composer textarea')") -> bool:
    await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 768})
    await cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": width < 768})
    await cdp.send("Page.navigate", {"url": url})
    ok = await cdp.wait_for(wait, timeout=60)
    await asyncio.sleep(1.0)
    return ok


async def click_in_task(cdp, task_id: str, marker_text: str) -> bool:
    return await cdp.evaluate(
        "(() => { const card = document.getElementById('task-%s'); if (!card) return false; const b = [...card.querySelectorAll('button')].find(b => b.innerText.includes(%s));"
        " if (!b) return false; b.click(); return true; })()" % (task_id, json.dumps(marker_text)))


async def pick_status(cdp, task_id: str, option_text: str) -> bool:
    center = await cdp.evaluate(
        "(() => { const card = document.getElementById('task-%s'); if (!card) return null; const c = card.querySelector('.q-field__control'); if (!c) return null;"
        " const r = c.getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()" % task_id)
    if not center:
        return False
    await cdp.evaluate("document.getElementById('task-%s').scrollIntoView({block: 'center'})" % task_id)
    await asyncio.sleep(0.3)
    center = await cdp.evaluate(
        "(() => { const c = document.getElementById('task-%s').querySelector('.q-field__control'); const r = c.getBoundingClientRect(); return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()" % task_id)
    for kind in ("mousePressed", "mouseReleased"):
        await cdp.send("Input.dispatchMouseEvent", {"type": kind, "x": center["x"], "y": center["y"], "button": "left", "clickCount": 1})
    await asyncio.sleep(0.8)
    chosen = await click(cdp, ".q-menu .q-item", option_text)
    await asyncio.sleep(0.8)
    return chosen


async def send_enter(cdp) -> None:
    await cdp.evaluate("document.querySelector('.composer textarea').focus()")
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})


async def wait_answer(cdp, timeout: float = 150) -> bool:
    ok = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > 0", timeout=timeout)
    await asyncio.sleep(0.8)
    return ok


async def type_search(cdp, text: str) -> None:
    await cdp.evaluate("(() => { const el = document.querySelector('.rf-input input'); el.focus(); const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; s.call(el, ''); el.dispatchEvent(new Event('input', {bubbles: true})); })()")
    await cdp.send("Input.insertText", {"text": text})
    await asyncio.sleep(1.2)


# ---------------------------------------------------------------------------
# Phase create
# ---------------------------------------------------------------------------
async def phase_create(base_url: str, out_dir: str, exe: str, record: dict) -> None:
    shots = os.path.join(out_dir, "screenshots")
    os.makedirs(shots, exist_ok=True)
    state = {}
    async with Visitor(exe, os.path.join(out_dir, "profiles", "A"), "A") as a:
        cdp = a.cdp
        # ---- Workflow A: Teaching & Learning ------------------------------------------------------------------------------
        ok = await open_page(cdp, base_url + "/onboarding?lang=en")
        s0 = await cdp.evaluate(JS_PAGE)
        step(record, "A1 dashboard: every catalog task listed, 0% and no progress yet", ok and s0["taskCards"] == len(C.TASKS) and s0["percent"] == "0%" and s0["stats"]["completed"] == 0 and s0["navCurrent"] == S["en"]["nav_onboarding"],
             detail="cards=%s percent=%s stats=%s" % (s0["taskCards"], s0["percent"], s0["stats"]))
        await shot(cdp, shots, "A_01_dashboard_en")
        clicked = await click_in_task(cdp, "bb_announcement", S["en"]["ob_open_guide"])
        await cdp.wait_for("!!document.querySelector('.pr-card')", timeout=20)
        await asyncio.sleep(0.8)
        s1 = await cdp.evaluate(JS_PAGE)
        step(record, "A2 guided procedure opens at step 1 of 8 with the exact guide text", clicked and (s1["stepCounter"] or "").lower() == "step 1 of 8" and s1["stepText"] == C.PROCEDURES["bb_announcement"].steps[0].text and s1["taskBox"] == S["en"]["pr_task"],
             detail="counter=%s" % s1["stepCounter"])
        await shot(cdp, shots, "A_02_procedure_step1")
        for _ in range(3):
            await click(cdp, ".pr-controls button", S["en"]["pr_next"])
            await asyncio.sleep(0.5)
        s2 = await cdp.evaluate(JS_PAGE)
        step(record, "A3 Next ×3 reaches step 4, flagged conditional", (s2["stepCounter"] or "").lower() == "step 4 of 8" and s2["stepFlag"] == S["en"]["pr_conditional"] and s2["stepText"].startswith("If you want to send a copy"),
             detail="counter=%s flag=%s" % (s2["stepCounter"], s2["stepFlag"]))
        await click(cdp, ".pr-controls button", S["en"]["pr_prev"])
        await asyncio.sleep(0.5)
        s2b = await cdp.evaluate(JS_PAGE)
        step(record, "A4 Previous returns to step 3", (s2b["stepCounter"] or "").lower() == "step 3 of 8", detail=s2b["stepCounter"])
        await shot(cdp, shots, "A_03_procedure_step3")
        state["step_text"] = s2b["stepText"]
        await click(cdp, ".pr-controls button", S["en"]["pr_ask_step"])
        await cdp.wait_for("!!document.querySelector('.composer textarea')", timeout=30)
        await asyncio.sleep(1.2)
        s3 = await cdp.evaluate(JS_PAGE)
        step(record, "A5 Ask about this step opens the chat with the question pre-filled (not sent) and the task as context",
             (s3["composer"] or "").startswith('In the "Send an Announcement" guide') and state["step_text"] in (s3["composer"] or "") and s3["userBubbles"] == 0 and (s3["chatContext"] or "").startswith(S["en"]["chat_context"].strip()),
             detail="context=%s composer=%s" % (s3["chatContext"], (s3["composer"] or "")[:70]))
        await shot(cdp, shots, "A_04_chat_prefilled")
        await send_enter(cdp)
        answered = await wait_answer(cdp)
        s4 = await cdp.evaluate(JS_PAGE)
        state["ask_step_answer"], state["ask_step_chips"] = s4["lastAnswer"], s4["chips"]
        step(record, "A6 the question is answered by the existing pipeline (one user turn, one answer)", answered and s4["userBubbles"] == 1 and s4["assistantBubbles"] == 1,
             detail="answer=%s chips=%s" % ((s4["lastAnswer"] or "")[:90], s4["chips"][:3]))
        await shot(cdp, shots, "A_05_chat_answer")
        await click(cdp, ".chat-context a")
        await cdp.wait_for("!!document.querySelector('.ob-tasks')", timeout=30)
        await asyncio.sleep(0.8)
        s5 = await cdp.evaluate(JS_PAGE)
        step(record, "A7 Open the task returns to the dashboard with the task highlighted and still not started", "task=bb_announcement" in s5["path"] and s5["statuses"].get("bb_announcement") == S["en"]["status_not_started"],
             detail="path=%s status=%s" % (s5["path"], s5["statuses"].get("bb_announcement")))
        await click_in_task(cdp, "bb_announcement", S["en"]["ob_resources"])
        await cdp.wait_for("!!document.querySelector('.rf-results')", timeout=30)
        await asyncio.sleep(1.0)
        s6 = await cdp.evaluate(JS_PAGE)
        step(record, "A8 the finder opens with the task context and the Blackboard guide first", s6["firstContextual"] and (s6["firstResource"] or "").startswith("Blackboard") and (s6["contextBox"] or "").startswith(S["en"]["rf_context"].strip()),
             detail="first=%s context=%s" % (s6["firstResource"], s6["contextBox"]))
        await shot(cdp, shots, "A_06_finder_context")
        await open_page(cdp, base_url + "/onboarding?task=bb_announcement&lang=en")
        await click_in_task(cdp, "bb_announcement", S["en"]["ob_mark_complete"])
        await asyncio.sleep(1.2)
        s7 = await cdp.evaluate(JS_PAGE)
        step(record, "A9 Mark completed saves and the summary shows one completed task", s7["statuses"].get("bb_announcement") == S["en"]["status_completed"] and s7["stats"]["completed"] == 1 and (s7["note"] or "").startswith("Saved:"),
             detail="note=%s stats=%s" % (s7["note"], s7["stats"]))
        await shot(cdp, shots, "A_07_dashboard_completed")

        # ---- Workflow B: Faculty services / HR ---------------------------------------------------------------------------
        await click(cdp, ".ob-filter", i18n.domain_name("en", "faculty_services"))
        await asyncio.sleep(0.6)
        s8 = await cdp.evaluate(JS_PAGE)
        step(record, "B1 the Faculty Services filter shows the HR tasks only", s8["taskCards"] == len(C.tasks_by_category("faculty_services")) and "hr_documents" in s8["statuses"], detail=str(list(s8["statuses"])))
        await shot(cdp, shots, "B_01_dashboard_hr_filter")
        await click_in_task(cdp, "hr_documents", S["en"]["ob_resources"])
        await cdp.wait_for("!!document.querySelector('.rf-results')", timeout=30)
        await asyncio.sleep(1.0)
        s9 = await cdp.evaluate(JS_PAGE)
        step(record, "B2 the finder shows Human Resources first with the contact exactly as printed", s9["firstResource"] == "Human Resources" and s9["firstContacts"] == ["+971 6 5050023 +971 6 5585200"], detail="first=%s contacts=%s" % (s9["firstResource"], s9["firstContacts"]))
        await shot(cdp, shots, "B_02_finder_hr")
        await open_page(cdp, base_url + "/onboarding?task=hr_documents&lang=en")
        await click_in_task(cdp, "hr_documents", S["en"]["ob_ask"])
        await cdp.wait_for("!!document.querySelector('.composer textarea')", timeout=30)
        await asyncio.sleep(1.2)
        s10 = await cdp.evaluate(JS_PAGE)
        step(record, "B3 Ask the assistant pre-fills the task question with the HR task as context", (s10["composer"] or "").endswith("Complete the pre-joining document formalities with HR?") and S["en"]["chat_context"].strip() in (s10["chatContext"] or ""), detail=str(s10["composer"])[:80])
        await send_enter(cdp)
        answered = await wait_answer(cdp)
        s11 = await cdp.evaluate(JS_PAGE)
        state["hr_answer"] = s11["lastAnswer"]
        step(record, "B4 the HR question is answered through the chat", answered and s11["assistantBubbles"] == 1, detail=(s11["lastAnswer"] or "")[:90])
        await shot(cdp, shots, "B_03_chat_hr_answer")
        await open_page(cdp, base_url + "/onboarding?task=hr_documents&lang=en")
        picked = await pick_status(cdp, "hr_documents", S["en"]["status_in_progress"])
        await asyncio.sleep(0.8)
        s12 = await cdp.evaluate(JS_PAGE)
        step(record, "B5 the status select sets the HR task to In progress", picked and s12["statuses"].get("hr_documents") == S["en"]["status_in_progress"] and s12["stats"]["in_progress"] == 1, detail=str(s12["stats"]))
        await shot(cdp, shots, "B_04_dashboard_hr_in_progress")

        # ---- Workflow C: Institutional (ILHE) ----------------------------------------------------------------------------
        await open_page(cdp, base_url + "/resources?lang=en")
        await type_search(cdp, "Orientation information")
        s13 = await cdp.evaluate(JS_PAGE)
        step(record, "C1 searching 'Orientation information' puts ILHE first, with its no-printed-contact note", (s13["firstResource"] or "").startswith("Institute of Leadership") and not s13["firstContacts"] and bool(s13["firstContactNote"]), detail="first=%s note=%s count=%s" % (s13["firstResource"], s13["firstContactNote"], s13["resultCount"]))
        await shot(cdp, shots, "C_01_finder_ilhe")
        await click(cdp, ".rf-card .chip.task", C.TASKS["ilhe_training"].title_en)
        await cdp.wait_for("!!document.querySelector('.ob-tasks')", timeout=30)
        await asyncio.sleep(0.8)
        s14 = await cdp.evaluate(JS_PAGE)
        step(record, "C2 the related-task chip opens the dashboard on the ILHE task", "task=ilhe_training" in s14["path"], detail=s14["path"])
        await click_in_task(cdp, "ilhe_training", S["en"]["ob_mark_complete"])
        await asyncio.sleep(1.0)
        s15 = await cdp.evaluate(JS_PAGE)
        step(record, "C3 marking ILHE complete gives two completed, one in progress", s15["stats"]["completed"] == 2 and s15["stats"]["in_progress"] == 1, detail=str(s15["stats"]))
        await shot(cdp, shots, "C_02_dashboard_two_completed")
        # no held source anywhere, Arabic dashboard shows the same statuses
        await open_page(cdp, base_url + "/onboarding?lang=ar")
        s16 = await cdp.evaluate(JS_PAGE)
        step(record, "D1 the Arabic dashboard is RTL and shows the same stored statuses", s16["dir"] == "rtl" and s16["stats"]["completed"] == 2 and s16["statuses"].get("bb_announcement") == S["ar"]["status_completed"] and not s16["heldSourceMentioned"],
             detail="dir=%s stats=%s" % (s16["dir"], s16["stats"]))
        await shot(cdp, shots, "D_01_dashboard_ar")
        await open_page(cdp, base_url + "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=ar")
        s17 = await cdp.evaluate(JS_PAGE)
        step(record, "D2 the Arabic navigator keeps the English guide text and remembers the position (step 3)", s17["dir"] == "rtl" and (s17["stepCounter"] or "").lower() == S["ar"]["pr_step"].format(n=3, total=8).lower() and s17["stepText"] == state["step_text"],
             detail="counter=%s" % s17["stepCounter"])
        await shot(cdp, shots, "D_02_procedure_ar")
        await open_page(cdp, base_url + "/resources?lang=ar")
        s18_all = await cdp.evaluate(JS_PAGE)
        await type_search(cdp, "الموارد البشرية")
        s18 = await cdp.evaluate(JS_PAGE)
        step(record, "D3 the Arabic finder finds Human Resources with the printed contact", s18["firstResource"] == "الموارد البشرية" and "+971 6 5050023 +971 6 5585200" in s18["contacts"], detail=str(s18["contacts"][:1]))
        await shot(cdp, shots, "D_03_finder_ar")
        state["links"] = s18_all["links"]
        step(record, "D4 every external link on the listing is https, noopener and opens in a new tab", bool(s18_all["links"]) and all(l["href"].startswith("https://") and "noopener" in (l["rel"] or "") and l["target"] == "_blank" for l in s18_all["links"]), detail="%d links, e.g. %s" % (len(s18_all["links"]), s18_all["links"][:1]))

        # ---- dark theme ------------------------------------------------------------------------------------------------------
        await open_page(cdp, base_url + "/onboarding?lang=en")
        await click(cdp, ".ob-topbar-right .topbar-btn:not(.topbar-lang)")
        await asyncio.sleep(0.8)
        sd = await cdp.evaluate(JS_PAGE)
        await shot(cdp, shots, "D_05_dashboard_dark")
        await open_page(cdp, base_url + "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=en")
        await shot(cdp, shots, "D_06_procedure_dark")
        await open_page(cdp, base_url + "/resources?task=bb_access&lang=en")
        sd2 = await cdp.evaluate(JS_PAGE)
        await shot(cdp, shots, "D_07_finder_dark")
        step(record, "D5 the theme toggle switches the pages to dark and the choice persists across pages", sd["theme"] == "dark" and sd2["theme"] == "dark", detail="theme=%s/%s" % (sd["theme"], sd2["theme"]))
        await click(cdp, ".ob-topbar-right .topbar-btn:not(.topbar-lang)")
        await asyncio.sleep(0.8)

        # ---- widths ----------------------------------------------------------------------------------------------------------
        record["viewports"] = []
        for lang in ("en", "ar"):
            for width, height in VIEWPORTS:
                for name, path in (("dashboard", "/onboarding?lang=%s" % lang), ("procedure", "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=%s" % lang), ("finder", "/resources?task=bb_access&lang=%s" % lang)):
                    ok = await open_page(cdp, base_url + path, width, height)
                    c = await cdp.evaluate(JS_PAGE)
                    c["ready"] = ok
                    record["viewports"].append({"lang": lang, "width": width, "height": height, "page": name, "checks": c})
                    await shot(cdp, shots, "W_%s_%dx%d_%s" % (lang, width, height, name))
        await open_page(cdp, base_url + "/onboarding?lang=en")

    async with Visitor(exe, os.path.join(out_dir, "profiles", "B"), "B") as b:
        cdp = b.cdp
        ok = await open_page(cdp, base_url + "/onboarding?lang=en")
        sb = await cdp.evaluate(JS_PAGE)
        step(record, "E1 visitor B sees the checklist with no progress of A", ok and sb["stats"]["completed"] == 0 and sb["stats"]["in_progress"] == 0 and sb["percent"] == "0%", detail=str(sb["stats"]))
        await shot(cdp, shots, "E_01_visitor_B_dashboard")
    with open(os.path.join(out_dir, "evidence", "create_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Phase restart
# ---------------------------------------------------------------------------
async def phase_restart(base_url: str, out_dir: str, exe: str, record: dict) -> None:
    shots = os.path.join(out_dir, "screenshots")
    with open(os.path.join(out_dir, "evidence", "create_state.json"), encoding="utf-8") as f:
        state = json.load(f)
    async with Visitor(exe, os.path.join(out_dir, "profiles", "A"), "A") as a:
        cdp = a.cdp
        ok = await open_page(cdp, base_url + "/onboarding?lang=en")
        s0 = await cdp.evaluate(JS_PAGE)
        step(record, "R1 after the server restart the statuses are back: two completed, HR in progress",
             ok and s0["stats"]["completed"] == 2 and s0["stats"]["in_progress"] == 1 and s0["statuses"].get("bb_announcement") == S["en"]["status_completed"] and s0["statuses"].get("ilhe_training") == S["en"]["status_completed"] and s0["statuses"].get("hr_documents") == S["en"]["status_in_progress"],
             detail=str(s0["stats"]))
        await shot(cdp, shots, "R_01_dashboard_restored")
        await open_page(cdp, base_url + "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=en")
        s1 = await cdp.evaluate(JS_PAGE)
        step(record, "R2 the procedure reopens at the remembered step with the task already marked completed", (s1["stepCounter"] or "").lower() == "step 3 of 8" and (s1["markDone"] is None or (s1["markDone"] or "").endswith(S["en"]["pr_marked_done"])), detail="counter=%s button=%s" % (s1["stepCounter"], s1["markDone"]))
        await shot(cdp, shots, "R_02_procedure_restored")
        await open_page(cdp, base_url + "/?lang=en")
        s2 = await cdp.evaluate(JS_PAGE)
        step(record, "R3 the chat reopens the last conversation with its task context and the saved answer (no new request)",
             (s2["chatContext"] or "").startswith(S["en"]["chat_context"].strip()) and s2["lastAnswer"] == state["hr_answer"] and not s2["typing"], detail="context=%s" % s2["chatContext"])
        await shot(cdp, shots, "R_03_chat_restored")
        await open_page(cdp, base_url + "/onboarding?lang=en")
        await click_in_task(cdp, "bb_announcement", S["en"]["ob_mark_complete"])         # the button no longer exists: nothing happens
        picked = await pick_status(cdp, "bb_announcement", S["en"]["status_completed"])    # the same value again: no second event
        await asyncio.sleep(0.8)
        s3 = await cdp.evaluate(JS_PAGE)
        step(record, "R4 repeating the completion changes nothing (still two completed)", s3["stats"]["completed"] == 2, detail="picked=%s stats=%s" % (picked, s3["stats"]))
    async with Visitor(exe, os.path.join(out_dir, "profiles", "B"), "B") as b:
        cdp = b.cdp
        await open_page(cdp, base_url + "/onboarding?lang=en")
        sb = await cdp.evaluate(JS_PAGE)
        step(record, "R5 visitor B still has no progress of A after the restart", sb["stats"]["completed"] == 0 and sb["stats"]["in_progress"] == 0, detail=str(sb["stats"]))


async def phase_widths(base_url: str, out_dir: str, exe: str, record: dict) -> None:
    """Re-capture the responsive checks and screenshots with visitor A's existing profile (state as left by the workflows)."""
    shots = os.path.join(out_dir, "screenshots")
    record["viewports"] = []
    async with Visitor(exe, os.path.join(out_dir, "profiles", "A"), "A") as a:
        cdp = a.cdp
        for lang in ("en", "ar"):
            for width, height in VIEWPORTS:
                for name, path in (("dashboard", "/onboarding?lang=%s" % lang), ("procedure", "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=%s" % lang), ("finder", "/resources?task=bb_access&lang=%s" % lang)):
                    ok = await open_page(cdp, base_url + path, width, height)
                    c = await cdp.evaluate(JS_PAGE)
                    c["ready"] = ok
                    c["geometry"] = await cdp.evaluate("(() => { const r = s => { const e = document.querySelector(s); if (!e) return null; const b = e.getBoundingClientRect(); return [Math.round(b.left), Math.round(b.right)]; }; return {innerWidth, scrollWidth: document.documentElement.scrollWidth, root: r('.ob-root'), main: r('.ob-main'), topbar: r('.ob-topbar'), nav: r('.ob-nav'), h1: r('.ob-h1')}; })()")
                    record["viewports"].append({"lang": lang, "width": width, "height": height, "page": name, "checks": c})
                    await shot(cdp, shots, "W_%s_%dx%d_%s" % (lang, width, height, name))
        await open_page(cdp, base_url + "/onboarding?lang=en")


def summarize(record: dict) -> int:
    failures = sum(1 for s in record["steps"] if not s["ok"])
    for vp in record.get("viewports", []):
        c = vp["checks"]
        problems = []
        if not c.get("ready"):
            problems.append("page not ready")
        if c.get("horizontalOverflow"):
            problems.append("horizontal overflow")
        g = c.get("geometry") or {}
        if g and any(rect and (rect[0] < -1 or rect[1] > g["innerWidth"] + 1) for rect in (g.get("main"), g.get("h1"), g.get("nav"))):
            problems.append("content outside the viewport: %s" % g)
        if c.get("wideElements"):
            problems.append("elements wider than the viewport: %s" % c["wideElements"][:3])
        if c.get("navLinks", 0) < 3:
            problems.append("navigation links missing")
        if vp["lang"] == "ar" and c.get("dir") != "rtl":
            problems.append("Arabic page not rtl")
        if vp["page"] == "dashboard" and c.get("taskCards") != len(C.TASKS):
            problems.append("task cards missing")
        if vp["page"] == "procedure" and not c.get("stepText"):
            problems.append("step text missing")
        if vp["page"] == "finder" and not c.get("resourceCards"):
            problems.append("resource cards missing")
        if c.get("heldSourceMentioned"):
            problems.append("held source mentioned")
        failures += len(problems)
        print("%s %4dx%-4d %-10s %s" % (vp["lang"], vp["width"], vp["height"], vp["page"], "OK" if not problems else "; ".join(problems)))
    print("problems:", failures)
    return failures


async def main(args) -> int:
    exe = find_browser(args.browser)
    os.makedirs(os.path.join(args.out_dir, "evidence"), exist_ok=True)
    record = {"phase": args.phase, "browser": exe, "base_url": args.base_url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}
    if args.phase == "create":
        await phase_create(args.base_url, args.out_dir, exe, record)
    elif args.phase == "widths":
        await phase_widths(args.base_url, args.out_dir, exe, record)
    else:
        await phase_restart(args.base_url, args.out_dir, exe, record)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.out_dir, "evidence", "%s_onboarding_browser_qa.json" % args.phase), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return summarize(record)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_dir")
    ap.add_argument("--phase", choices=("create", "restart", "widths"), required=True)
    ap.add_argument("--browser", choices=("chrome", "edge"), default="chrome")
    a = ap.parse_args()
    a.base_url = a.base_url.rstrip("/")
    sys.exit(1 if asyncio.run(main(a)) else 0)
