"""Real-browser QA of the bilingual chat page (post-freeze PF-11), without new dependencies.

Drives the installed Chrome or Edge headless over the DevTools protocol (``websockets`` is a NiceGUI dependency) against a
RUNNING application, in both interface languages, at the viewports the phase requires, with REAL interactions, and
saves a screenshot of every state plus a JSON record of every check.

    python -X utf8 qa/pf11_arabic/browser_qa_bilingual.py <base url> <output dir> [--browser chrome|edge] [--label after]

Per viewport (320, 375, 768, 1024, 1440): the English page (no horizontal overflow, composer, Send, New chat, language
button visible), then the language switch to Arabic (``<html lang="ar" dir="rtl">``, Arabic h1, Arabic composer
placeholder, Arabic Send label, no overflow), then at 1440 and 375 px: an Arabic question typed and sent with Enter, the
processing state, the real answer (Arabic lead-in or translation, RTL bubble direction, sources, evidence), a mixed
Arabic/English message, the language switch BACK to English with the conversation intact, the dark theme, keyboard
focus on the controls. Launch the server with the model translation switched off (``ARABIC_ANSWER_TRANSLATION=0``,
``ARABIC_QUERY_MODEL_FALLBACK=0``) so this run makes no inference request; the live smoke evaluation covers translation.
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
ARABIC_QUESTION = "ما هو رقم هاتف قسم الموارد البشرية؟"
MIXED_QUESTION = "كيف أرسل إعلانًا للطلبة عبر Blackboard؟"

JS_LANG = r"""
(() => {
  const html = document.documentElement;
  const ta = document.querySelector('.composer textarea');
  const h1 = document.querySelector('h1');
  const send = document.querySelector('.send-btn');
  const lang = document.querySelector('.topbar-lang');
  const bubbles = [...document.querySelectorAll('.bubble')];
  const last = bubbles.length ? bubbles[bubbles.length - 1] : null;
  const ar = /[؀-ۿ]/;
  const shell = document.querySelector('.app-shell');
  const shellRect = shell ? shell.getBoundingClientRect() : null;
  const newChatAny = [...document.querySelectorAll('button, [role=button]')].filter(b => /new chat|محادثة جديدة/i.test(b.getAttribute('aria-label') || b.textContent || ''))
    .filter(b => { const cs = getComputedStyle(b); const r = b.getBoundingClientRect(); return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0; });
  const userBubble = document.querySelector('.bubble.user');
  const ub = userBubble ? userBubble.getBoundingClientRect() : null;
  return {
    lang: html.getAttribute('lang'), dir: html.getAttribute('dir'), computedDir: getComputedStyle(html).direction,
    shellWidth: shellRect ? Math.round(shellRect.width) : null, shellFillsViewport: shellRect ? Math.abs(shellRect.width - innerWidth) <= 1 && shellRect.left >= -1 : null,
    newChatVisibleAnyLanguage: newChatAny.length > 0,
    userBubbleInside: ub ? (ub.left >= -1 && ub.right <= innerWidth + 1) : null,
    lastBubbleRtlResolved: (() => {
      // dir="auto" is not visible to :dir() or to the computed direction in every engine: measure where the first
      // character of the first paragraph is laid out (near the right edge = right-to-left)
      if (!last) return null;
      const p = last.querySelector('.answer p, .notice-text') || last;
      const find = n => { for (const c of n.childNodes) { if (c.nodeType === 3 && c.textContent.trim()) return c; if (c.nodeType === 1) { const f = find(c); if (f) return f; } } return null; };
      const tn = find(p); if (!tn) return null;
      const r = document.createRange(); r.setStart(tn, 0); r.setEnd(tn, 1);
      const cr = r.getBoundingClientRect(), pr = p.getBoundingClientRect();
      return (pr.right - cr.right) < (cr.left - pr.left);
    })(),
    h1: h1 ? h1.textContent.trim() : null, h1Arabic: h1 ? ar.test(h1.textContent) : null,
    placeholder: ta ? ta.getAttribute('placeholder') : null, placeholderArabic: ta ? ar.test(ta.getAttribute('placeholder') || '') : null,
    sendLabel: send ? send.innerText.trim() : null, sendArabic: send ? ar.test(send.innerText) : null,
    langButton: lang ? lang.innerText.trim() : null, langButtonVisible: !!(lang && lang.getBoundingClientRect().width > 0),
    bubbleCount: bubbles.length,
    lastBubbleDir: last ? getComputedStyle(last).direction : null,
    lastBubbleArabic: last ? ar.test(last.innerText) : null,
    lastBubbleText: last ? last.innerText.trim().slice(0, 300) : null,
    sourcesLabel: (() => { const s = document.querySelector('.sources-label'); return s ? s.textContent.trim() : null; })(),
    evidenceSummary: (() => { const s = document.querySelector('details.evidence summary'); return s ? s.innerText.trim() : null; })(),
    translationNote: (() => { const s = document.querySelector('.translation-note'); return s ? s.innerText.trim() : null; })(),
    answerLangControl: !!document.querySelector('.answer-lang'),
    userBubbles: document.querySelectorAll('.msg-row.user').length,
    sessionTitles: [...document.querySelectorAll('.session-title')].map(e => e.textContent.trim()),
    theme: html.getAttribute('data-theme'),
    activeTag: document.activeElement ? document.activeElement.tagName + '.' + (document.activeElement.className || '').toString().slice(0, 40) : null,
    focusOutline: document.activeElement ? getComputedStyle(document.activeElement).outlineStyle : null,
  };
})()
"""


async def click_lang(cdp: CDP) -> bool:
    return await cdp.evaluate("(() => { const b = document.querySelector('.topbar-lang'); if (!b) return false; b.click(); return true; })()")


async def send_text(cdp: CDP, text: str) -> None:
    await cdp.evaluate("document.querySelector('.composer textarea').focus()")
    await cdp.send("Input.insertText", {"text": text})
    await asyncio.sleep(0.3)
    for kind in ("keyDown", "keyUp"):
        await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})


async def wait_answer(cdp: CDP, count: int) -> bool:
    return await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length >= %d && "
                              "document.querySelectorAll('.bubble.assistant')[document.querySelectorAll('.bubble.assistant').length-1].innerText.trim().length > 0" % count, timeout=120)


async def run(base_url: str, out_dir: str, browser: str, label: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    exe = find_browser(browser)
    profile = tempfile.mkdtemp(prefix="pf11-browser-")
    proc, ws_url = launch(exe, profile)
    record = {"label": label, "browser": exe, "base_url": base_url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "viewports": []}
    try:
        async with CDP(ws_url) as cdp:
            targets = (await cdp.send("Target.getTargets"))["targetInfos"]
            page = next(t for t in targets if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            for width, height in VIEWPORTS:
                vp = {"width": width, "height": height, "shots": [], "checks": {}, "interactions": {}}
                record["viewports"].append(vp)
                await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 768})
                await cdp.send("Emulation.setTouchEmulationEnabled", {"enabled": width < 768})
                await cdp.send("Page.navigate", {"url": base_url})
                ready = await cdp.wait_for("!!document.querySelector('.composer textarea')", timeout=60)
                await asyncio.sleep(1.2)
                vp["ready"] = ready
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_en_landing" % (label, width, height)))
                vp["checks"]["en_landing"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_LANG))}
                # ---- switch to Arabic ----------------------------------------------------------------------------------
                vp["interactions"]["lang_switch_clicked"] = await click_lang(cdp)
                await cdp.wait_for("document.documentElement.getAttribute('lang') === 'ar'", timeout=10)
                await asyncio.sleep(1.0)
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_ar_landing" % (label, width, height)))
                vp["checks"]["ar_landing"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_LANG))}
                if width not in INTERACT_AT or not ready:
                    continue
                # ---- Arabic question with Enter ---------------------------------------------------------------------------
                await send_text(cdp, ARABIC_QUESTION)
                vp["interactions"]["processing_state_seen"] = await cdp.wait_for("!!document.querySelector('.typing') || document.querySelectorAll('.msg-row.assistant').length > 0", timeout=10)
                vp["interactions"]["ar_answered"] = await wait_answer(cdp, 1)
                await asyncio.sleep(0.8)
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_ar_answer" % (label, width, height)))
                vp["checks"]["ar_answer"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_LANG))}
                vp["checks"]["ar_answer"]["answerClipped"] = await cdp.evaluate(
                    "(() => { const b = document.querySelectorAll('.bubble.assistant'); if (!b.length) return null; const el = b[b.length-1]; return el.scrollWidth > el.clientWidth + 1; })()")
                vp["checks"]["ar_answer"]["evidenceOpens"] = await cdp.evaluate(
                    "(() => { const d = document.querySelector('details.evidence'); if (!d) return null; d.querySelector('summary').click(); return d.open; })()")
                await asyncio.sleep(0.4)
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_ar_evidence" % (label, width, height)))
                # ---- mixed-direction message ---------------------------------------------------------------------------------
                await send_text(cdp, MIXED_QUESTION)
                vp["interactions"]["mixed_answered"] = await wait_answer(cdp, 2)
                await asyncio.sleep(0.8)
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_ar_mixed" % (label, width, height)))
                vp["checks"]["ar_mixed"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_LANG))}
                # ---- switch back to English: the conversation must survive ------------------------------------------------------
                before = await cdp.evaluate("document.querySelectorAll('.msg-row').length")
                await click_lang(cdp)
                await cdp.wait_for("document.documentElement.getAttribute('lang') === 'en'", timeout=10)
                await asyncio.sleep(1.0)
                after = await cdp.evaluate("document.querySelectorAll('.msg-row').length")
                vp["interactions"]["switch_back_keeps_messages"] = {"before": before, "after": after, "kept": before == after and before > 0}
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_en_after_switch" % (label, width, height)))
                vp["checks"]["en_after_switch"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_LANG))}
                # ---- dark theme in Arabic ---------------------------------------------------------------------------------------
                await click_lang(cdp)
                await cdp.wait_for("document.documentElement.getAttribute('lang') === 'ar'", timeout=10)
                await cdp.evaluate("(() => { const b = [...document.querySelectorAll('button')].find(b => /dark|الداكن/i.test(b.getAttribute('aria-label') || '')); if (b) b.click(); })()")
                await asyncio.sleep(0.8)
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_ar_dark" % (label, width, height)))
                vp["checks"]["ar_dark"] = {**(await cdp.evaluate(JS_CHECKS)), **(await cdp.evaluate(JS_LANG))}
                # ---- keyboard focus ---------------------------------------------------------------------------------------------
                await cdp.evaluate("document.querySelector('.composer textarea').focus()")
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Tab", "code": "Tab", "windowsVirtualKeyCode": 9})
                await asyncio.sleep(0.3)
                vp["checks"]["keyboard"] = await cdp.evaluate(JS_LANG)
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_ar_focus" % (label, width, height)))
    finally:
        proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "%s_bilingual_browser_qa.json" % label), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return record


def summarize(record: dict) -> int:
    failures = 0
    for vp in record["viewports"]:
        w, h = vp["width"], vp["height"]
        for state, c in vp["checks"].items():
            problems = []
            if c.get("horizontalOverflow"):
                problems.append("horizontal overflow (%s/%s px)" % (c.get("docScrollWidth"), c.get("innerWidth")))
            if "textarea" in c and not (c.get("textarea") or {}).get("inside"):
                problems.append("textarea not fully visible")
            if "send" in c and not (c.get("send") or {}).get("inside"):
                problems.append("Send button not fully visible")
            if "newChatVisibleAnyLanguage" in c and not c.get("newChatVisibleAnyLanguage"):
                problems.append("no visible New chat control")
            if c.get("shellFillsViewport") is False:
                problems.append("shell does not fill the viewport (%s/%s px)" % (c.get("shellWidth"), c.get("innerWidth")))
            if c.get("userBubbleInside") is False:
                problems.append("user bubble outside the viewport")
            if c.get("wideElements"):
                problems.append("elements wider than viewport: %s" % c["wideElements"][:3])
            if state.startswith("ar_") and state != "keyboard":
                if c.get("lang") != "ar" or c.get("dir") != "rtl" or c.get("computedDir") != "rtl":
                    problems.append("html lang/dir not ar/rtl (%s/%s, computed %s)" % (c.get("lang"), c.get("dir"), c.get("computedDir")))
                if not c.get("h1Arabic") or not c.get("placeholderArabic"):
                    problems.append("English text left in the Arabic shell")
            if state == "en_after_switch" and (c.get("lang") != "en" or c.get("dir") != "ltr" or c.get("computedDir") != "ltr"):
                problems.append("html lang/dir not en/ltr after switching back")
            if state == "ar_answer":
                if c.get("lastBubbleRtlResolved") is False or not c.get("lastBubbleArabic"):
                    problems.append("Arabic answer bubble is not RTL Arabic")
                if c.get("answerClipped"):
                    problems.append("answer bubble clipped")
                if not c.get("chips"):
                    problems.append("no source chips on the answer")
            if state == "keyboard" and c.get("focusOutline") in (None, "none"):
                problems.append("no visible focus outline on the focused control")
            failures += len(problems)
            print("%4dx%-4d %-16s %s" % (w, h, state, "OK" if not problems else "; ".join(problems)))
        for k, v in vp["interactions"].items():
            print("%4dx%-4d   %s: %s" % (w, h, k, v))
            if isinstance(v, dict) and "kept" in v and not v["kept"]:
                failures += 1
    print("problems:", failures)
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_dir")
    ap.add_argument("--browser", choices=("chrome", "edge"), default="chrome")
    ap.add_argument("--label", default="after")
    a = ap.parse_args()
    rec = asyncio.run(run(a.base_url, a.out_dir, a.browser, a.label))
    sys.exit(1 if summarize(rec) else 0)
