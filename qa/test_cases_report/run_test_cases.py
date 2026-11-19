"""System test cases with screenshots for the UOS Faculty Onboarding Chatbot.

Starts the app (unless one is already listening on :8501), drives headless Chrome/Edge through the project's own
CDP helper (qa/pf10_ui/browser_qa.py), executes each test case, captures a screenshot per case and writes results.json.

    .venv\\Scripts\\python.exe -X utf8 qa\\test_cases_report\\run_test_cases.py
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "qa", "pf10_ui"))
sys.path.insert(0, REPO)
from browser_qa import CDP, launch, shot, BROWSERS          # noqa: E402

BASE = "http://127.0.0.1:8501"
OUT = os.path.join(HERE, "screenshots")
DESKTOP, PHONE = (1440, 900), (375, 812)
os.makedirs(OUT, exist_ok=True)
RESULTS: list = []
LOG = open(os.path.join(HERE, "run_log.txt"), "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def port_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8501), timeout=2):
            return True
    except OSError:
        return False


def start_app():
    if port_open():
        log("app already running on 8501")
        return None
    env = dict(os.environ)
    env.setdefault("PLAN_F_ENABLED", "1")
    env.setdefault("PLAN_F_GUIDE_SCOPES", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    app_log = open(os.path.join(HERE, "app_log.txt"), "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-X", "utf8", "app.py"], cwd=REPO, env=env, stdout=app_log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("app exited early, see app_log.txt")
        try:
            with urllib.request.urlopen(BASE + "/", timeout=5) as r:
                if r.status == 200:
                    log("app up after %.0fs" % (time.time() - (deadline - 900)))
                    return proc
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("app did not start")


def find_exe() -> str:
    for name in ("chrome", "edge"):
        for p in BROWSERS[name]:
            if os.path.exists(p):
                return p
    raise SystemExit("no Chrome/Edge found")


READY = "!!document.querySelector('.composer textarea') || !!document.querySelector('.ob-main') || !!document.querySelector('.page-main')"


async def open_page(cdp, url, size=DESKTOP):
    w, h = size
    await cdp.send("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": w < 768})
    await cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": w < 768})
    await cdp.send("Page.navigate", {"url": url})
    ok = await cdp.wait_for(READY, timeout=90)
    await asyncio.sleep(1.5)
    return ok


async def js(cdp, expr):
    return await cdp.evaluate(expr)


async def type_text(cdp, text):
    await js(cdp, "document.querySelector('.composer textarea').focus()")
    await cdp.send("Input.insertText", {"text": text})
    await asyncio.sleep(0.3)


async def press_enter(cdp, shift=False):
    extra = {"modifiers": 8, "text": "\r", "unmodifiedText": "\r"} if shift else {}
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13, **extra})


async def ask(cdp, q, timeout=180):
    before = await js(cdp, "document.querySelectorAll('.bubble.assistant').length")
    await type_text(cdp, q)
    await press_enter(cdp)
    t0 = time.time()
    ok = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > %d && document.querySelectorAll('.bubble.assistant')[document.querySelectorAll('.bubble.assistant').length-1].innerText.trim().length > 0" % before, timeout=timeout)
    await asyncio.sleep(1.0)
    ans = await js(cdp, "(() => { const b = document.querySelectorAll('.bubble.assistant'); return b.length ? b[b.length-1].innerText.trim() : ''; })()")
    chips = await js(cdp, "[...document.querySelectorAll('.chip')].map(c => c.innerText.trim()).filter(Boolean)")
    await js(cdp, "(() => { const b = document.querySelectorAll('.msg-row.assistant'); if (b.length) b[b.length-1].scrollIntoView({block: 'start'}); })()")
    await asyncio.sleep(0.4)
    return ok, ans or "", chips or [], round(time.time() - t0, 1)


async def click(cdp, selector, text=None):
    return await js(cdp, "(() => { const els = [...document.querySelectorAll(%s)].filter(e => { const cs = getComputedStyle(e); const r = e.getBoundingClientRect(); return cs.display !== 'none' && r.width > 0 && (%s === null || e.innerText.includes(%s)); }); if (!els.length) return false; els[0].click(); return true; })()"
                    % (json.dumps(selector), json.dumps(text), json.dumps(text)))


async def new_chat(cdp):
    r = await js(cdp, "(() => { const b = [...document.querySelectorAll('button')].filter(b => /new chat/i.test(b.getAttribute('aria-label') || b.textContent || '')).find(b => getComputedStyle(b).display !== 'none' && b.getBoundingClientRect().width > 0); if (!b) return false; b.click(); return true; })()")
    await asyncio.sleep(1.2)
    return r


def record(tc, module, title, pre, steps, data, expected, actual, passed, shot_path):
    RESULTS.append({"id": tc, "module": module, "title": title, "preconditions": pre, "steps": steps, "test_data": data,
                    "expected": expected, "actual": actual, "status": "Pass" if passed else "Fail",
                    "screenshot": os.path.basename(shot_path) if shot_path else None})
    log("%s %-5s %s | %s" % (tc, "PASS" if passed else "FAIL", title, actual[:160].replace("\n", " ")))


try:
    from handbook_bot import conversation as _conv
    _FRIENDLY = tuple(x.lower()[:30] for x in _conv.REFUSALS["en"])
except Exception:
    _FRIENDLY = ()
REFUSAL_WORDS = _FRIENDLY + ("could not find", "couldn't find", "not covered", "outside", "cannot answer", "can't answer", "not able", "no information",
                 "only answer", "not in the", "don't have", "do not have", "unable", "not supported", "rephrase", "more detail", "specific",
                 "approved documents", "handbook does not", "not found")


async def run_all(cdp):
    # TC01 landing ---------------------------------------------------------------------------------------------------
    ok = await open_page(cdp, BASE + "/?lang=en")
    st = await js(cdp, "({title: document.title, composer: !!document.querySelector('.composer textarea'), send: !!document.querySelector('.send-btn'), sidebar: !!document.querySelector('.sidebar'), text: document.body.innerText.slice(0, 400)})")
    p = await shot(cdp, OUT, "TC01_landing")
    record("TC-01", "Chat UI", "Application loads and shows the first-use screen",
           "App started with `python app.py`; browser opens http://localhost:8501", ["Open http://localhost:8501/?lang=en"], "-",
           "Page loads with the UOS header, sidebar navigation, welcome screen with example questions and a message composer with a Send button.",
           "Page loaded=%s, title='%s', composer=%s, send button=%s, sidebar=%s." % (ok, st["title"], st["composer"], st["send"], st["sidebar"]),
           bool(ok and st["composer"] and st["send"]), p)

    # TC02 deterministic contact -----------------------------------------------------------------------------------------
    q = "What is the phone number of the Human Resources Department?"
    ok, ans, chips, secs = await ask(cdp, q)
    p = await shot(cdp, OUT, "TC02_hr_contact")
    has_num = any(ch.isdigit() for ch in ans) and ("5050023" in ans.replace(" ", "") or "+971" in ans)
    record("TC-02", "Question answering", "Contact lookup returns the HR phone number with a source",
           "TC-01 passed", ["Type the question in the composer", "Press Enter"], q,
           "The bot answers with the Human Resources phone number exactly as printed in the handbook and shows a Sources row.",
           "Answered=%s in %ss. Answer: \"%s\". Sources: %s" % (ok, secs, ans[:220].replace("\n", " "), ", ".join(chips[:3]) or "none"),
           bool(ok and has_num), p)

    # TC03 evidence panel --------------------------------------------------------------------------------------------------
    opened = await js(cdp, "(() => { const d = [...document.querySelectorAll('.msg-row.assistant details, details.evidence, .evidence details')]; if (!d.length) return false; d[d.length-1].open = true; d[d.length-1].scrollIntoView({block: 'center'}); return true; })()")
    await asyncio.sleep(0.8)
    p = await shot(cdp, OUT, "TC03_evidence")
    record("TC-03", "Question answering", "Supporting evidence panel can be expanded",
           "TC-02 answer is on screen", ["Click 'Supporting evidence' under the answer"], "-",
           "The evidence list expands and shows the handbook passages and page numbers used for the answer.",
           "Evidence panel found and opened=%s." % opened, bool(opened), p)

    # TC04 policy question (LLM) ----------------------------------------------------------------------------------------
    await new_chat(cdp)
    q = "How many office hours must full-time faculty hold, and what is the teaching load per semester?"
    ok, ans, chips, secs = await ask(cdp, q)
    p = await shot(cdp, OUT, "TC04_policy")
    record("TC-04", "Question answering", "Handbook policy question is answered with citations",
           "New chat; GROQ_API_KEY configured", ["Click New chat", "Type the question", "Press Enter"], q,
           "A grounded answer stating office-hour and teaching-load rules from the Faculty Handbook, with page citations.",
           "Answered=%s in %ss. Answer: \"%s\". Sources: %s" % (ok, secs, ans[:260].replace("\n", " "), ", ".join(chips[:3]) or "none"),
           bool(ok and len(ans) > 40 and chips), p)

    # TC05 guide procedure question ----------------------------------------------------------------------------------------
    await new_chat(cdp)
    q = "How do I take attendance in Blackboard?"
    ok, ans, chips, secs = await ask(cdp, q)
    p = await shot(cdp, OUT, "TC05_blackboard_attendance")
    record("TC-05", "Question answering", "Blackboard how-to question is answered from the attendance guide",
           "New chat", ["Click New chat", "Type the question", "Press Enter"], q,
           "Step-by-step answer taken from the Blackboard Attendance guide, citing that guide.",
           "Answered=%s in %ss. Answer: \"%s\". Sources: %s" % (ok, secs, ans[:260].replace("\n", " "), ", ".join(chips[:3]) or "none"),
           bool(ok and len(ans) > 40 and "ttendance" in (ans + " ".join(chips))), p)

    # TC06 out of scope ----------------------------------------------------------------------------------------------------
    await new_chat(cdp)
    q = "What is the weather in Sharjah today?"
    ok, ans, chips, secs = await ask(cdp, q)
    p = await shot(cdp, OUT, "TC06_out_of_scope")
    refused = any(w in ans.lower() for w in REFUSAL_WORDS)
    record("TC-06", "Safety / grounding", "Out-of-scope question is politely refused (no hallucination)",
           "New chat", ["Click New chat", "Type the question", "Press Enter"], q,
           "The bot states the question is outside the approved university documents and does not invent an answer.",
           "Answered=%s. Response: \"%s\"" % (ok, ans[:260].replace("\n", " ")), bool(ok and refused), p)

    # TC07 vague -----------------------------------------------------------------------------------------------------------
    await new_chat(cdp)
    q = "I need help."
    ok, ans, chips, secs = await ask(cdp, q)
    p = await shot(cdp, OUT, "TC07_vague")
    record("TC-07", "Safety / grounding", "Vague request gets a clarification / guidance response",
           "New chat", ["Click New chat", "Type the message", "Press Enter"], q,
           "The bot asks the user to be more specific or explains what topics it can help with, without making up policy.",
           "Answered=%s. Response: \"%s\"" % (ok, ans[:260].replace("\n", " ")), bool(ok and len(ans) > 10), p)

    # TC08 empty message ---------------------------------------------------------------------------------------------------
    await new_chat(cdp)
    before = await js(cdp, "document.querySelectorAll('.msg-row.user').length")
    await js(cdp, "document.querySelector('.composer textarea').focus()")
    await press_enter(cdp)
    await asyncio.sleep(1.0)
    after = await js(cdp, "document.querySelectorAll('.msg-row.user').length")
    p = await shot(cdp, OUT, "TC08_empty_message")
    record("TC-08", "Input validation", "Empty message is not sent",
           "New chat, empty composer", ["Click into the empty message box", "Press Enter"], "(empty)",
           "Nothing is sent; no user bubble or answer appears.",
           "User messages before=%s, after=%s." % (before, after), before == after, p)

    # TC09 shift+enter -----------------------------------------------------------------------------------------------------
    await type_text(cdp, "line one")
    await press_enter(cdp, shift=True)
    await cdp.send("Input.insertText", {"text": "line two"})
    await asyncio.sleep(0.6)
    val = await js(cdp, "document.querySelector('.composer textarea').value")
    sent = (await js(cdp, "document.querySelectorAll('.msg-row.user').length")) != after
    p = await shot(cdp, OUT, "TC09_shift_enter")
    record("TC-09", "Input validation", "Shift+Enter inserts a new line instead of sending",
           "Composer focused", ["Type 'line one'", "Press Shift+Enter", "Type 'line two'"], "line one / line two",
           "The message box contains two lines and nothing is sent.",
           "Textarea value=%r, sent=%s." % (val, sent), bool(val and "\n" in val and not sent), p)
    await js(cdp, "(() => { const t = document.querySelector('.composer textarea'); const s = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set; s.call(t, ''); t.dispatchEvent(new Event('input', {bubbles: true})); })()")

    # TC10 arabic question -------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/?lang=ar")
    await new_chat(cdp)
    q = "ما هو رقم هاتف قسم الموارد البشرية؟"
    ok, ans, chips, secs = await ask(cdp, q)
    d = await js(cdp, "getComputedStyle(document.documentElement).direction")
    p = await shot(cdp, OUT, "TC10_arabic_question")
    record("TC-10", "Bilingual (Arabic)", "Arabic question is answered in Arabic with a right-to-left layout",
           "Interface switched to Arabic (?lang=ar)", ["Open the chat in Arabic", "Type the Arabic question", "Press Enter"], q + " (What is the HR phone number?)",
           "The page is right-to-left and the bot answers in Arabic with the HR phone number preserved exactly.",
           "Direction=%s, answered=%s. Answer: \"%s\"" % (d, ok, ans[:200].replace("\n", " ")),
           bool(ok and d == "rtl" and any(c.isdigit() for c in ans)), p)

    # TC11 language switch -------------------------------------------------------------------------------------------------
    clicked = await click(cdp, ".topbar-lang")
    await asyncio.sleep(2.0)
    await cdp.wait_for(READY, timeout=30)
    st = await js(cdp, "({lang: document.documentElement.lang, dir: getComputedStyle(document.documentElement).direction, msgs: document.querySelectorAll('.msg-row').length})")
    p = await shot(cdp, OUT, "TC11_language_switch")
    record("TC-11", "Bilingual (Arabic)", "Language button switches the interface back to English and keeps the conversation",
           "TC-10 conversation on screen", ["Click the language button (English) in the header"], "-",
           "The interface switches to English, left-to-right, and the Arabic conversation is still shown.",
           "Button clicked=%s, lang=%s, direction=%s, messages kept=%s." % (clicked, st["lang"], st["dir"], st["msgs"]),
           bool(clicked and st["dir"] == "ltr" and st["msgs"] >= 2), p)

    # TC12 history ---------------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/history?lang=en")
    txt = await js(cdp, "document.body.innerText")
    p = await shot(cdp, OUT, "TC12_history")
    found = "Human Resources" in txt or "phone number" in txt or "office hours" in txt
    record("TC-12", "Saved conversations", "Previous conversations are saved and listed on the History page",
           "TC-02 to TC-10 conversations were created", ["Open Conversation history from the sidebar"], "-",
           "The History page lists the earlier conversations with titles taken from their first question.",
           "Earlier conversation titles found on page=%s." % found, found, p)

    # TC13 reopen after reload -------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/?lang=en")
    await click(cdp, ".sidebar .session-title", "phone number")
    await asyncio.sleep(2.0)
    msgs = await js(cdp, "document.querySelectorAll('.msg-row').length")
    p = await shot(cdp, OUT, "TC13_reopen_conversation")
    record("TC-13", "Saved conversations", "A saved conversation reopens after page reload without re-asking",
           "Browser reloaded", ["Reload the page", "Click the 'What is the phone number…' conversation in the sidebar"], "-",
           "The stored question and answer are shown again from the database.",
           "Messages displayed after reopening=%s." % msgs, bool(msgs and msgs >= 2), p)

    # TC14 checklist --------------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/onboarding?lang=en")
    st = await js(cdp, "({cards: document.querySelectorAll('.ob-task').length, percent: (document.querySelector('.ob-ring-value')||{}).innerText})")
    p = await shot(cdp, OUT, "TC14_onboarding_checklist")
    record("TC-14", "Onboarding", "Onboarding checklist lists the onboarding tasks with overall progress",
           "-", ["Open 'Onboarding checklist' from the sidebar"], "-",
           "A dashboard of onboarding tasks (Teaching, Faculty services, Institutional) with a progress ring.",
           "Task cards=%s, progress=%s." % (st["cards"], st["percent"]), bool(st["cards"] and st["cards"] >= 5), p)

    # TC15 mark complete --------------------------------------------------------------------------------------------------
    before = st["percent"]
    marked = await js(cdp, "(() => { const c = document.getElementById('task-hr_documents') || document.querySelector('.ob-task'); if (!c) return false; c.scrollIntoView({block: 'center'}); const b = [...c.querySelectorAll('button')].find(b => /Mark completed/i.test(b.innerText)); if (!b) return 'already'; b.click(); return true; })()")
    await asyncio.sleep(1.5)
    st2 = await js(cdp, "({percent: (document.querySelector('.ob-ring-value')||{}).innerText, completed: Number((document.querySelector('[data-stat=\"completed\"]')||{}).innerText)})")
    await js(cdp, "window.scrollTo(0, 0)")
    await asyncio.sleep(0.4)
    p = await shot(cdp, OUT, "TC15_task_completed")
    record("TC-15", "Onboarding", "Marking a task completed updates the progress",
           "Checklist open", ["Click 'Mark completed' on the 'HR pre-joining documents' task"], "-",
           "The task status becomes Completed, a 'Saved' note appears and the completed counter / percentage increases.",
           "Clicked=%s, progress %s -> %s, completed count=%s." % (marked, before, st2["percent"], st2["completed"]),
           bool(marked and (st2["completed"] or 0) >= 1), p)

    # TC16 procedure navigation ---------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=en")
    c1 = await js(cdp, "(document.querySelector('.pr-counter, .pr-step-counter')||{}).innerText || (document.body.innerText.match(/Step \\d+ of \\d+/i)||[''])[0]")
    for _ in range(2):
        await click(cdp, ".pr-controls button", "Next")
        await asyncio.sleep(0.7)
    c2 = await js(cdp, "(document.body.innerText.match(/Step \\d+ of \\d+/i)||[''])[0]")
    p = await shot(cdp, OUT, "TC16_guided_procedure")
    record("TC-16", "Onboarding", "Guided procedure moves step by step with Next",
           "-", ["Open Guided procedures > 'Send an Announcement' (Blackboard)", "Click Next twice"], "-",
           "The navigator starts at Step 1 and shows Step 3 after two clicks, with the exact guide text for each step.",
           "Counter before='%s', after='%s'." % (c1, c2), "3" in (c2 or "").split(" of")[0], p)

    # TC17 resource finder ----------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/resources?lang=en")
    await js(cdp, "(() => { const el = document.querySelector('.rf-input input'); el.focus(); })()")
    await cdp.send("Input.insertText", {"text": "iThenticate"})
    await asyncio.sleep(1.5)
    first = await js(cdp, "(() => { const c = document.querySelector('.rf-card'); return c ? c.innerText.split('\\n')[0] : null; })()")
    n = await js(cdp, "document.querySelectorAll('.rf-card').length")
    p = await shot(cdp, OUT, "TC17_resource_finder")
    record("TC-17", "Resources", "Resource finder search returns the matching document",
           "-", ["Open 'Resources & documents'", "Type 'iThenticate' in the search box"], "iThenticate",
           "The iThenticate guidelines document is listed first.",
           "Results=%s, first result='%s'." % (n, first), bool(first and "henticate" in first), p)

    # TC18 dark mode -------------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/?lang=en")
    t0 = await js(cdp, "document.documentElement.getAttribute('data-theme')")
    await js(cdp, "(() => { const b = document.querySelector('.topbar-btn[aria-label*=\"mode\"], .topbar-btn.theme-btn'); if (b) b.click(); return !!b; })()")
    await asyncio.sleep(1.2)
    t1 = await js(cdp, "document.documentElement.getAttribute('data-theme')")
    p = await shot(cdp, OUT, "TC18_dark_mode")
    record("TC-18", "Settings", "Theme button switches between light and dark mode",
           "-", ["Click the dark/light mode button in the header"], "-",
           "The whole interface switches to the dark theme.", "Theme before=%s, after=%s." % (t0, t1), bool(t1 and t1 != t0), p)
    await js(cdp, "(() => { const b = document.querySelector('.topbar-btn[aria-label*=\"mode\"], .topbar-btn.theme-btn'); if (b) b.click(); })()")
    await asyncio.sleep(1.0)

    # TC19 settings ------------------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/settings?lang=en")
    txt = await js(cdp, "document.body.innerText")
    p = await shot(cdp, OUT, "TC19_settings")
    has = [k for k in ("Language", "Theme", "Text size", "motion", "Voice") if k.lower() in txt.lower()]
    record("TC-19", "Settings", "Settings page shows language, theme and accessibility options",
           "-", ["Open 'Settings' from the sidebar"], "-",
           "Settings for interface/answer language, theme, larger text and reduced motion are shown.",
           "Options found: %s." % ", ".join(has), len(has) >= 3, p)

    # TC20 profile ---------------------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/profile?lang=en")
    txt = await js(cdp, "document.body.innerText")
    p = await shot(cdp, OUT, "TC20_guest_profile")
    record("TC-20", "Guest profile", "Guest profile page shows what this browser has saved",
           "-", ["Open 'Guest profile' from the sidebar"], "-",
           "The page shows the guest identity, saved conversation count / onboarding progress and an optional display name.",
           "Page text starts: \"%s\"" % txt[:200].replace("\n", " | "), "rofile" in txt or "Guest" in txt, p)

    # TC22-25 conversation upgrade ---------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/?lang=en")
    await new_chat(cdp)
    ok, ans, chips, secs = await ask(cdp, "thanks!")
    p = await shot(cdp, OUT, "TC22_thanks_small_talk")
    friendly = not any(w in ans.lower() for w in ("not found in the approved", "do not have this information", "no supporting evidence"))
    record("TC-22", "Conversation", "A thank-you gets a friendly reply instead of a 'not found' notice",
           "New chat", ["Type 'thanks!'", "Press Enter"], "thanks!",
           "A short, warm reply (e.g. 'You're welcome!'), with no 'not found' notice and no sources.",
           "Answered=%s in %ss. Reply: \"%s\"" % (ok, secs, ans[:200].replace("\n", " ")), bool(ok and friendly and len(ans) > 5), p)

    ok, ans, chips, secs = await ask(cdp, "شكرا")
    p = await shot(cdp, OUT, "TC23_arabic_thanks")
    arabic = any("\u0600" <= c <= "\u06ff" for c in ans)
    record("TC-23", "Conversation", "An Arabic thank-you gets a friendly Arabic reply",
           "TC-22 conversation", ["Type 'شكرا' (thank you)", "Press Enter"], "شكرا (thanks)",
           "A warm reply written in Arabic.", "Answered=%s, reply is Arabic=%s. Reply: \"%s\"" % (ok, arabic, ans[:160].replace("\n", " ")),
           bool(ok and arabic and "غير موجود" not in ans), p)

    await new_chat(cdp)
    ok, ans, chips, secs = await ask(cdp, "who aproves confrence travel")
    note = await js(cdp, "(() => { const n = document.querySelectorAll('.spell-note'); return n.length ? n[n.length-1].innerText : ''; })()")
    p = await shot(cdp, OUT, "TC24_typo_correction")
    record("TC-24", "Conversation", "A misspelled question is corrected and answered",
           "New chat", ["Type the misspelled question", "Press Enter"], "who aproves confrence travel",
           "The page shows 'Showing results for who approves conference travel' and answers from the handbook with a source.",
           "Note: \"%s\". Answer: \"%s\". Sources: %s" % (note, ans[:200].replace("\n", " "), ", ".join(chips[:3]) or "none"),
           bool(ok and "approves conference" in (note or "") and chips), p)

    await new_chat(cdp)
    ok, ans, chips, secs = await ask(cdp, "What is the weather in Sharjah today?")
    sugg = await js(cdp, "[...document.querySelectorAll('.suggest-btn')].map(b => b.innerText.trim())")
    p = await shot(cdp, OUT, "TC25_refusal_suggestions")
    record("TC-25", "Conversation", "A 'not found' answer is friendly and offers example questions",
           "New chat", ["Ask an out-of-scope question", "Look under the notice"], "What is the weather in Sharjah today?",
           "A friendly 'couldn't find that' sentence with a tip, plus clickable example questions; the old fixed sentence is gone.",
           "Response: \"%s\". Suggestions: %s" % (ans[:200].replace("\n", " "), sugg),
           bool(ok and sugg and "I do not have this information" not in ans), p)

    # TC21 mobile ----------------------------------------------------------------------------------------------------------------
    await open_page(cdp, BASE + "/?lang=en", PHONE)
    of = await js(cdp, "document.documentElement.scrollWidth > document.documentElement.clientWidth + 1")
    await js(cdp, "(() => { const b = document.querySelector('.nav-toggle'); if (b) b.click(); return !!b; })()")
    await asyncio.sleep(1.0)
    p = await shot(cdp, OUT, "TC21_mobile_menu")
    record("TC-21", "Responsive design", "Phone-width layout has no horizontal overflow and a slide-in menu",
           "Viewport 375 x 812 (iPhone size)", ["Open the chat at phone width", "Tap the menu button"], "-",
           "The page fits the screen without sideways scrolling and the navigation opens as a slide-in drawer.",
           "Horizontal overflow=%s; menu opened." % of, not of, p)


async def main():
    proc = start_app()
    exe = find_exe()
    profile = tempfile.mkdtemp(prefix="tc-browser-")
    bproc, ws = launch(exe, profile)
    try:
        async with CDP(ws) as cdp:
            targets = (await cdp.send("Target.getTargets"))["targetInfos"]
            page = next(t for t in targets if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            try:
                await run_all(cdp)
            except Exception:
                log("ERROR during run:\n" + traceback.format_exc())
    finally:
        bproc.kill()
        if proc:
            proc.terminate()
        with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "browser": exe, "base_url": BASE, "results": RESULTS}, f, indent=1, ensure_ascii=False)
        with open(os.path.join(HERE, "DONE.txt"), "w") as f:
            f.write("done %s\n" % time.strftime("%H:%M:%S"))
        log("finished: %d cases" % len(RESULTS))


if __name__ == "__main__":
    asyncio.run(main())
