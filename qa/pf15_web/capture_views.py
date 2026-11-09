"""Matched before/after screenshots of the website (post-freeze PF-15).

Runs ONE scenario against a running application with a fresh persistent Chrome profile and captures the same views at the
same viewport sizes, so a PF-14 ("before") run and a PF-15 ("after") run can be compared one to one:

    python -X utf8 qa/pf15_web/capture_views.py <base url> <output dir> --label before|after

Scenario (identical for both labels): open the chat, ask the deterministic HR contact question (no model request), mark
one checklist task completed and move one procedure two steps, then capture: chat (landing, answer, evidence open),
history, settings, profile, checklist, procedure, finder, Arabic chat, checklist and finder (after: also Arabic settings,
profile and history), phone-width chat / checklist / finder
/ history, dark chat and checklist. Where a page did not exist before PF-15 (Settings page, Profile page, History page,
Procedures index) the "before" capture shows what existed then (the settings dialog, the history dialog) or records
"not previously implemented"; nothing is mocked. A JSON index of every capture goes next to the screenshots.
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
from browser_qa import find_browser, shot                                         # noqa: E402
from browser_qa_persistence import Visitor, click                                 # noqa: E402

HR_QUESTION = "What is the phone number of the Human Resources Department?"
DESKTOP = (1440, 900)
PHONE = (375, 812)
READY = "!!document.querySelector('.composer textarea') || !!document.querySelector('.ob-main') || !!document.querySelector('.page-main')"


async def open_page(cdp, url: str, size=DESKTOP, wait: str = READY) -> bool:
    width, height = size
    await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 768})
    await cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": width < 768})
    await cdp.send("Page.navigate", {"url": url})
    ok = await cdp.wait_for(wait, timeout=60)
    await asyncio.sleep(1.2)
    return ok


async def capture(cdp, out_dir: str, index: list, name: str, area: str, note: str = "") -> None:
    path = await shot(cdp, out_dir, name)
    metrics = await cdp.evaluate("({w: document.documentElement.clientWidth, h: innerHeight, overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,"
                                 " lang: document.documentElement.lang, theme: document.documentElement.getAttribute('data-theme'), path: location.pathname + location.search})")
    index.append({"file": path, "area": area, "note": note, **metrics})
    print("captured", path, "|", area, "|", metrics)


async def set_theme(cdp, base: str, label: str, theme: str) -> None:
    """Switch theme through the real UI control of the chat page."""
    await open_page(cdp, base + "/?lang=en")
    current = await cdp.evaluate("document.documentElement.getAttribute('data-theme')")
    if current != theme:
        await cdp.evaluate("(() => { const b = document.querySelector('.topbar-btn[aria-label*=\"mode\"], .topbar-btn.theme-btn'); if (b) b.click(); return !!b; })()")
        await asyncio.sleep(1.0)


async def ask(cdp, question: str) -> bool:
    await cdp.evaluate("document.querySelector('.composer textarea').focus()")
    await cdp.send("Input.insertText", {"text": question})
    await asyncio.sleep(0.3)
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
    ok = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > 0", timeout=120)
    await asyncio.sleep(1.0)
    return ok


async def run(base: str, out_dir: str, label: str) -> list:
    shots_dir = os.path.join(out_dir, "screenshots", label)
    os.makedirs(shots_dir, exist_ok=True)
    index: list = []
    profile = os.path.join(out_dir, "profiles", "capture_" + label)
    async with Visitor(find_browser("chrome"), profile, "C") as v:
        cdp = v.cdp
        # ---- chat ---------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/?lang=en")
        await capture(cdp, shots_dir, index, "01_chat_landing_en_1440", "Agentic AI chat (first use)")
        answered = await ask(cdp, HR_QUESTION)
        await capture(cdp, shots_dir, index, "02_chat_answer_en_1440", "Agentic AI chat (answer with sources)", "answered=%s" % answered)
        await cdp.evaluate("(() => { const d = document.querySelector('.evidence details, details.evidence, .msg-row.assistant details'); if (d) d.open = true; })()")
        await asyncio.sleep(0.5)
        await capture(cdp, shots_dir, index, "03_chat_evidence_en_1440", "Evidence panel opened")
        # ---- onboarding state: one task completed, one procedure moved ---------------------------------------------------
        await open_page(cdp, base + "/onboarding?lang=en")
        await cdp.evaluate("(() => { const c = document.getElementById('task-hr_documents'); const b = c && [...c.querySelectorAll('button')].find(b => /Mark completed/.test(b.innerText)); if (b) b.click(); })()")
        await asyncio.sleep(1.2)
        await open_page(cdp, base + "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=en")
        for _ in range(2):
            await click(cdp, ".pr-controls button", "Next")
            await asyncio.sleep(0.6)
        # ---- history ------------------------------------------------------------------------------------------------------
        if label == "before":
            index.append({"file": None, "area": "Conversation history page", "note": "Not previously implemented: PF-14 lists conversations in the chat sidebar and, at phone width, in a dialog."})
            await open_page(cdp, base + "/?lang=en", PHONE)
            await click(cdp, ".topbar-history")
            await asyncio.sleep(1.0)
            await capture(cdp, shots_dir, index, "04_history_dialog_en_375", "Conversation history (phone dialog)")
        else:
            await open_page(cdp, base + "/history?lang=en")
            await capture(cdp, shots_dir, index, "04_history_en_1440", "Conversation history page")
            await open_page(cdp, base + "/history?lang=en", PHONE)
            await capture(cdp, shots_dir, index, "04_history_en_375", "Conversation history page (phone)")
        # ---- settings -----------------------------------------------------------------------------------------------------
        if label == "before":
            index.append({"file": None, "area": "Settings page", "note": "Not previously implemented: PF-13/PF-14 offer a settings dialog on the chat page."})
            await open_page(cdp, base + "/?lang=en")
            await click(cdp, ".topbar-settings")
            await asyncio.sleep(1.0)
            await capture(cdp, shots_dir, index, "05_settings_dialog_en_1440", "Settings (dialog)")
        else:
            await open_page(cdp, base + "/settings?lang=en")
            await capture(cdp, shots_dir, index, "05_settings_en_1440", "Settings page")
            await open_page(cdp, base + "/settings?lang=en", PHONE)
            await capture(cdp, shots_dir, index, "05_settings_en_375", "Settings page (phone)")
        # ---- profile ------------------------------------------------------------------------------------------------------
        if label == "before":
            index.append({"file": None, "area": "Guest profile page", "note": "Not previously implemented."})
        else:
            await open_page(cdp, base + "/profile?lang=en")
            await capture(cdp, shots_dir, index, "06_profile_en_1440", "Guest profile page")
            await open_page(cdp, base + "/profile?lang=en", PHONE)
            await capture(cdp, shots_dir, index, "06_profile_en_375", "Guest profile page (phone)")
        # ---- onboarding pages ---------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/onboarding?lang=en")
        await capture(cdp, shots_dir, index, "07_checklist_en_1440", "Onboarding checklist")
        if label == "after":
            await open_page(cdp, base + "/procedures?lang=en")
            await capture(cdp, shots_dir, index, "08a_procedures_index_en_1440", "Guided procedures index")
        else:
            index.append({"file": None, "area": "Guided procedures index", "note": "Not previously implemented: procedures were reachable only from a task card."})
        await open_page(cdp, base + "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=en")
        await capture(cdp, shots_dir, index, "08_procedure_en_1440", "Guided procedure navigator")
        await open_page(cdp, base + "/resources?task=bb_access&lang=en")
        await capture(cdp, shots_dir, index, "09_finder_en_1440", "Resources finder (task context)")
        # ---- Arabic ---------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/?lang=ar")
        await capture(cdp, shots_dir, index, "10_chat_ar_1440", "Arabic chat (saved conversation)")
        await open_page(cdp, base + "/onboarding?lang=ar")
        await capture(cdp, shots_dir, index, "11_checklist_ar_1440", "Arabic checklist")
        await open_page(cdp, base + "/resources?lang=ar")
        await capture(cdp, shots_dir, index, "12_finder_ar_1440", "Arabic finder")
        if label == "after":
            await open_page(cdp, base + "/settings?lang=ar")
            await capture(cdp, shots_dir, index, "12b_settings_ar_1440", "Arabic settings page")
            await open_page(cdp, base + "/profile?lang=ar", PHONE)
            await capture(cdp, shots_dir, index, "12c_profile_ar_375", "Arabic guest profile (phone)")
            await open_page(cdp, base + "/history?lang=ar", PHONE)
            await capture(cdp, shots_dir, index, "12d_history_ar_375", "Arabic conversation history (phone)")
        # ---- phone ----------------------------------------------------------------------------------------------------------
        await open_page(cdp, base + "/?lang=en", PHONE)
        await capture(cdp, shots_dir, index, "13_chat_en_375", "Chat (phone)")
        if label == "after":
            await cdp.evaluate("(() => { const b = document.querySelector('.nav-toggle'); if (b) b.click(); })()")
            await asyncio.sleep(0.8)
            await capture(cdp, shots_dir, index, "13b_nav_drawer_en_375", "Navigation drawer (phone)")
            await cdp.evaluate("(() => { if (window.fobNav) window.fobNav(false); })()")
        await open_page(cdp, base + "/onboarding?lang=en", PHONE)
        await capture(cdp, shots_dir, index, "14_checklist_en_375", "Checklist (phone)")
        await open_page(cdp, base + "/resources?lang=en", PHONE)
        await capture(cdp, shots_dir, index, "15_finder_en_375", "Finder (phone)")
        await open_page(cdp, base + "/onboarding/procedure/bb_announcement?task=bb_announcement&lang=en", PHONE)
        await capture(cdp, shots_dir, index, "16_procedure_en_375", "Procedure (phone)")
        await open_page(cdp, base + "/?lang=ar", PHONE)
        await capture(cdp, shots_dir, index, "17_chat_ar_375", "Arabic chat (phone)")
        # ---- dark -----------------------------------------------------------------------------------------------------------
        await set_theme(cdp, base, label, "dark")
        await open_page(cdp, base + "/?lang=en")
        await capture(cdp, shots_dir, index, "18_chat_dark_en_1440", "Chat, dark theme")
        await open_page(cdp, base + "/onboarding?lang=en")
        await capture(cdp, shots_dir, index, "19_checklist_dark_en_1440", "Checklist, dark theme")
        await set_theme(cdp, base, label, "light")
    with open(os.path.join(shots_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "base_url": base, "captured": time.strftime("%Y-%m-%d %H:%M:%S"), "views": index}, f, indent=1, ensure_ascii=False)
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_dir")
    ap.add_argument("--label", choices=("before", "after"), required=True)
    a = ap.parse_args()
    views = asyncio.run(run(a.base_url.rstrip("/"), a.out_dir, a.label))
    print("views:", len(views), "captured files:", len([v for v in views if v.get("file")]), "overflow:", [v["file"] for v in views if v.get("overflow")])
