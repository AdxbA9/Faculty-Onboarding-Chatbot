"""Real-browser QA of the chat page without new dependencies (post-freeze PF-10).

Drives the Chrome or Edge already installed on the computer, headless, over the DevTools protocol (``websockets`` is a
NiceGUI dependency), at the viewports the phase requires, performs REAL interactions on a RUNNING application (a question
typed and sent with Enter, Shift+Enter, New chat) and saves a screenshot of every state plus a JSON record of every check.
Nothing is mocked: the page answers from the real pipeline of the server it is pointed at.

    python -X utf8 qa/pf10_ui/browser_qa.py <base url> <output dir> [--browser chrome|edge] [--label before|after]
                                           [--question "..."] [--no-interactions]

Checks per viewport: no horizontal page overflow, composer textarea and Send button visible and inside the viewport, a
New chat control visible (sidebar or topbar), no element wider than the viewport inside the chat area. Interactions at
1440 px and 375 px: send a question, wait for the real answer, verify the answer bubble and its source chips exist and are
not clipped, Shift+Enter inserts a newline without sending, New chat empties the conversation.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import websockets

VIEWPORTS = [(320, 568), (375, 812), (768, 1024), (1024, 768), (1440, 900)]
INTERACT_AT = {1440, 375}
BROWSERS = {
    "chrome": [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
    "edge": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"],
}
DEFAULT_QUESTION = "What is the phone number of the Human Resources Department?"

JS_CHECKS = r"""
(() => {
  const vis = (el) => {
    if (!el) return null;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return {visible: false};
    const r = el.getBoundingClientRect();
    const inside = r.width > 0 && r.height > 0 && r.left >= -1 && r.top >= -1 && r.right <= innerWidth + 1 && r.bottom <= innerHeight + 1;
    return {visible: true, inside, w: Math.round(r.width), h: Math.round(r.height), x: Math.round(r.left), y: Math.round(r.top)};
  };
  const wide = [];
  document.querySelectorAll('.main *').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width > innerWidth + 1 && getComputedStyle(el).position !== 'fixed') wide.push(el.className ? String(el.className).slice(0, 60) : el.tagName);
  });
  const newChat = [...document.querySelectorAll('button, [role=button]')].filter(b => /new chat/i.test(b.getAttribute('aria-label') || b.textContent || '')).map(vis).filter(v => v && v.visible);
  return {
    innerWidth, innerHeight,
    docScrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
    horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1 || document.body.scrollWidth > innerWidth + 1,
    textarea: vis(document.querySelector('.composer textarea')),
    send: vis(document.querySelector('.send-btn')),
    newChatVisible: newChat.length > 0,
    newChatControls: newChat,
    sidebarVisible: !!(document.querySelector('.sidebar') && getComputedStyle(document.querySelector('.sidebar')).display !== 'none'),
    assistantBubbles: document.querySelectorAll('.msg-row.assistant').length,
    userBubbles: document.querySelectorAll('.msg-row.user').length,
    typing: !!document.querySelector('.typing'),
    chips: document.querySelectorAll('.chip').length,
    wideElements: wide.slice(0, 8),
    title: document.title,
    h1: document.querySelector('h1') ? document.querySelector('h1').textContent.trim() : null,
    lastAnswer: (() => { const b = document.querySelectorAll('.bubble.assistant'); return b.length ? b[b.length - 1].innerText.trim().slice(0, 400) : null; })(),
    focusVisibleRules: [...document.styleSheets].some(s => { try { return [...s.cssRules].some(r => /focus-visible/.test(r.selectorText || '')); } catch (e) { return false; } }),
  };
})()
"""


class CDP:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.ws = None
        self.next_id = 0
        self.pending = {}
        self.events = asyncio.Queue()
        self.session = None

    async def __aenter__(self):
        self.ws = await websockets.connect(self.ws_url, max_size=50_000_000)
        self.reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc):
        self.reader.cancel()
        await self.ws.close()

    async def _read(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if "id" in msg and msg["id"] in self.pending:
                self.pending.pop(msg["id"]).set_result(msg)
            elif "method" in msg:
                await self.events.put(msg)

    async def send(self, method: str, params: dict | None = None, session: str | None = None, timeout: float = 30):
        self.next_id += 1
        msg = {"id": self.next_id, "method": method, "params": params or {}}
        if session or self.session:
            msg["sessionId"] = session or self.session
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.next_id] = fut
        await self.ws.send(json.dumps(msg))
        reply = await asyncio.wait_for(fut, timeout)
        if "error" in reply:
            raise RuntimeError("%s: %s" % (method, reply["error"]))
        return reply.get("result", {})

    async def evaluate(self, expression: str, timeout: float = 30):
        r = await self.send("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True}, timeout=timeout)
        if "exceptionDetails" in r:
            raise RuntimeError("JS: %s" % r["exceptionDetails"].get("text"))
        return r.get("result", {}).get("value")

    async def wait_for(self, expression: str, timeout: float = 60, interval: float = 0.25):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if await self.evaluate(expression):
                return True
            await asyncio.sleep(interval)
        return False


def find_browser(name: str) -> str:
    for p in BROWSERS[name]:
        if os.path.exists(p):
            return p
    raise SystemExit("browser not found: %s" % name)


def launch(exe: str, profile: str) -> tuple[subprocess.Popen, str]:
    args = [exe, "--headless=new", "--remote-debugging-port=0", "--user-data-dir=" + profile, "--no-first-run", "--no-default-browser-check",
            "--disable-gpu", "--hide-scrollbars", "--window-size=1440,900", "about:blank"]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    ws_url = None
    deadline = time.time() + 30
    while time.time() < deadline and ws_url is None:
        line = proc.stderr.readline()
        m = re.search(r"DevTools listening on (ws://\S+)", line or "")
        if m:
            ws_url = m.group(1)
    if ws_url is None:
        proc.kill()
        raise SystemExit("DevTools endpoint not announced")
    return proc, ws_url


async def run(base_url: str, out_dir: str, browser: str, label: str, question: str, interactions: bool) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    exe = find_browser(browser)
    profile = tempfile.mkdtemp(prefix="pf10-browser-")
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
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_landing" % (label, width, height)))
                vp["checks"]["landing"] = await cdp.evaluate(JS_CHECKS)
                if not interactions or width not in INTERACT_AT or not ready:
                    continue
                # --- send a real question with Enter ---------------------------------------------------------------
                await cdp.evaluate("document.querySelector('.composer textarea').focus()")
                await cdp.send("Input.insertText", {"text": question})
                await asyncio.sleep(0.3)
                vp["interactions"]["typed"] = await cdp.evaluate("document.querySelector('.composer textarea').value")
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
                saw_typing = await cdp.wait_for("!!document.querySelector('.typing') || document.querySelectorAll('.msg-row.assistant').length > 0", timeout=10)
                vp["interactions"]["processing_state_seen"] = saw_typing
                if saw_typing:
                    vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_processing" % (label, width, height)))
                answered = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > 0 && document.querySelectorAll('.bubble.assistant')[document.querySelectorAll('.bubble.assistant').length-1].innerText.trim().length > 0", timeout=120)
                await asyncio.sleep(0.8)
                vp["interactions"]["answered"] = answered
                vp["interactions"]["textarea_after_send"] = await cdp.evaluate("document.querySelector('.composer textarea').value")
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_answer" % (label, width, height)))
                vp["checks"]["answer"] = await cdp.evaluate(JS_CHECKS)
                vp["checks"]["answer"]["chipTexts"] = await cdp.evaluate("[...document.querySelectorAll('.chip')].map(c => c.innerText.trim())")
                vp["checks"]["answer"]["answerClipped"] = await cdp.evaluate(
                    "(() => { const b = document.querySelectorAll('.bubble.assistant'); if (!b.length) return null; const el = b[b.length-1]; return el.scrollWidth > el.clientWidth + 1; })()")
                # --- Shift+Enter inserts a newline and does not send ---------------------------------------------------
                before = await cdp.evaluate("document.querySelectorAll('.msg-row.user').length")
                await cdp.evaluate("document.querySelector('.composer textarea').focus()")
                await cdp.send("Input.insertText", {"text": "line one"})
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "modifiers": 8, "text": "\r", "unmodifiedText": "\r"})
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "modifiers": 8})
                await asyncio.sleep(0.6)
                value = await cdp.evaluate("document.querySelector('.composer textarea').value")
                after = await cdp.evaluate("document.querySelectorAll('.msg-row.user').length")
                vp["interactions"]["shift_enter"] = {"value_repr": repr(value), "newline_inserted": "\n" in (value or ""), "sent": after != before}
                await cdp.evaluate("(() => { const t = document.querySelector('.composer textarea'); const s = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set; s.call(t, ''); t.dispatchEvent(new Event('input', {bubbles: true})); })()")
                # --- empty Enter sends nothing -------------------------------------------------------------------------
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
                await cdp.send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13})
                await asyncio.sleep(0.8)
                vp["interactions"]["empty_enter_sent"] = (await cdp.evaluate("document.querySelectorAll('.msg-row.user').length")) != before
                # --- New chat ------------------------------------------------------------------------------------------
                clicked = await cdp.evaluate(
                    "(() => { const b = [...document.querySelectorAll('button')].filter(b => /new chat/i.test(b.getAttribute('aria-label') || b.textContent || ''))"
                    ".find(b => { const cs = getComputedStyle(b); const r = b.getBoundingClientRect(); return cs.display !== 'none' && r.width > 0; }); if (!b) return false; b.click(); return true; })()")
                await asyncio.sleep(1.0)
                vp["interactions"]["new_chat"] = {"control_found": clicked, "conversation_cleared": (await cdp.evaluate("document.querySelectorAll('.msg-row').length")) == 0}
                vp["shots"].append(await shot(cdp, out_dir, "%s_%dx%d_after_new_chat" % (label, width, height)))
    finally:
        proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "%s_browser_qa.json" % label), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return record


async def shot(cdp: CDP, out_dir: str, name: str) -> str:
    data = (await cdp.send("Page.captureScreenshot", {"format": "png"}))["data"]
    path = os.path.join(out_dir, name + ".png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(data))
    return os.path.basename(path)


def summarize(record: dict) -> int:
    failures = 0
    for vp in record["viewports"]:
        w, h = vp["width"], vp["height"]
        for state, c in vp["checks"].items():
            problems = []
            if c.get("horizontalOverflow"):
                problems.append("horizontal overflow (%s/%s px)" % (c.get("docScrollWidth"), c.get("innerWidth")))
            if not (c.get("textarea") or {}).get("inside"):
                problems.append("textarea not fully visible")
            if not (c.get("send") or {}).get("inside"):
                problems.append("Send button not fully visible")
            if not c.get("newChatVisible"):
                problems.append("no visible New chat control")
            if c.get("wideElements"):
                problems.append("elements wider than viewport: %s" % c["wideElements"][:3])
            if state == "answer" and c.get("answerClipped"):
                problems.append("answer bubble clipped horizontally")
            if state == "answer" and not c.get("chips"):
                problems.append("no source chips on the answer")
            failures += len(problems)
            print("%4dx%-4d %-8s %s" % (w, h, state, "OK" if not problems else "; ".join(problems)))
        for k, v in vp["interactions"].items():
            print("%4dx%-4d   %s: %s" % (w, h, k, v))
    print("problems:", failures)
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_dir")
    ap.add_argument("--browser", choices=sorted(BROWSERS), default="chrome")
    ap.add_argument("--label", default="run")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--no-interactions", action="store_true")
    a = ap.parse_args()
    rec = asyncio.run(run(a.base_url, a.out_dir, a.browser, a.label, a.question, not a.no_interactions))
    sys.exit(1 if summarize(rec) else 0)
