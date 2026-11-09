"""PF-16: a screenshot of one region of a page of the RUNNING application (headless Chrome), for before/after evidence.

    python -X utf8 qa/pf16_audit/element_shot.py <url> <css selector> <out.png> [--width 1440 --height 900] [--pad 24]
        [--type "text" --into "<input selector>"] [--send] [--wait "<js condition>"]

``--type`` types text into an input first (real key input); ``--send`` presses Enter; ``--wait`` waits for a condition.
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pf10_ui"))
from browser_qa import CDP, find_browser, launch   # noqa: E402


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


async def main():
    url, selector, out = sys.argv[1], sys.argv[2], sys.argv[3]
    width, height, pad = int(arg("--width", 1440)), int(arg("--height", 900)), int(arg("--pad", 24))
    proc, ws = launch(find_browser("chrome"), tempfile.mkdtemp(prefix="pf16-shot-"))
    try:
        async with CDP(ws) as cdp:
            page = next(t for t in (await cdp.send("Target.getTargets"))["targetInfos"] if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 768})
            await cdp.send("Page.navigate", {"url": url})
            await cdp.wait_for("!!document.querySelector(%r)" % selector, timeout=60)
            await asyncio.sleep(1.2)
            if "--type" in sys.argv:
                await cdp.evaluate("document.querySelector(%r).focus()" % arg("--into"))
                await cdp.send("Input.insertText", {"text": arg("--type")})
                await asyncio.sleep(0.4)
            if "--send" in sys.argv:
                for kind in ("keyDown", "keyUp"):
                    await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
            if "--wait" in sys.argv:
                await cdp.wait_for(arg("--wait"), timeout=60)
                await asyncio.sleep(1.0)
            rect = await cdp.evaluate("(() => { const r = document.querySelector(%r).getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height}; })()" % selector)
            clip = {"x": max(0, rect["x"] - pad), "y": max(0, rect["y"] - pad), "width": rect["w"] + 2 * pad, "height": rect["h"] + 2 * pad, "scale": 1}
            img = await cdp.send("Page.captureScreenshot", {"format": "png", "clip": clip})
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "wb") as f:
                f.write(base64.b64decode(img["data"]))
            print("saved", out, clip)
    finally:
        proc.kill()


asyncio.run(main())
