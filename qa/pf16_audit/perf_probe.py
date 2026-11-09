"""PF-16 measured responsiveness of the RUNNING application (headless Chrome over CDP, plus plain HTTP), no model request.

    python -X utf8 qa/pf16_audit/perf_probe.py <base url> <out dir>

1. page load: each of the 7 routes 3 times (navigation start -> the page's main region present, and the browser's
   domContentLoaded / load events);
2. deterministic question: the HR contact question 3 times in a fresh conversation (Enter -> answer bubble with sources);
3. conversation switching: open a saved conversation from the History page 5 times (click -> replayed answer visible);
4. concurrency: 10 simultaneous HTTP GET / requests (first byte and total);
5. finder search in-process: ``onboarding.finder`` over the whole catalog, 200 queries.
Writes ``perf_probe.json``. Figures are wall-clock on the evaluation PC with the server and the browser on the same machine.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "pf10_ui"))
sys.path.insert(0, REPO)
from browser_qa import CDP, find_browser, launch   # noqa: E402

ROUTES = ["/", "/onboarding", "/procedures", "/resources", "/history", "/profile", "/settings"]
READY = "!!document.querySelector('.page-main, .composer textarea, .ob-main')"
HR = "What is the phone number of the Human Resources Department?"


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 1), "max_ms": round(xs[-1], 1), "min_ms": round(xs[0], 1)} if xs else {}


async def browser_part(base):
    exe = find_browser("chrome")
    proc, ws = launch(exe, tempfile.mkdtemp(prefix="pf16-perf-"))
    out = {"pages": {}, "question_ms": [], "switch_ms": []}
    try:
        async with CDP(ws) as cdp:
            page = next(t for t in (await cdp.send("Target.getTargets"))["targetInfos"] if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False})
            await cdp.send("Page.navigate", {"url": base + "/?lang=en"})          # warm-up (first page builds the session)
            await cdp.wait_for(READY, timeout=90)
            for route in ROUTES:
                rows = []
                for _ in range(3):
                    t = time.perf_counter()
                    await cdp.send("Page.navigate", {"url": base + route + "?lang=en"})
                    await cdp.wait_for(READY, timeout=60)
                    ready = (time.perf_counter() - t) * 1000
                    nav = await cdp.evaluate("(() => { const n = performance.getEntriesByType('navigation')[0]; return n ? {dcl: n.domContentLoadedEventEnd, load: n.loadEventEnd} : null; })()")
                    rows.append({"ready_ms": round(ready, 1), "dcl_ms": round((nav or {}).get("dcl") or 0, 1), "load_ms": round((nav or {}).get("load") or 0, 1)})
                out["pages"][route] = {"ready": stats([r["ready_ms"] for r in rows]), "domContentLoaded": stats([r["dcl_ms"] for r in rows]), "load": stats([r["load_ms"] for r in rows])}
                print(route, out["pages"][route]["ready"], flush=True)
            for _ in range(3):
                await cdp.send("Page.navigate", {"url": base + "/?new=1&lang=en"})
                await cdp.wait_for("!!document.querySelector('.composer textarea')", timeout=60)
                await asyncio.sleep(0.5)
                await cdp.evaluate("document.querySelector('.composer textarea').focus()")
                await cdp.send("Input.insertText", {"text": HR})
                n0 = await cdp.evaluate("document.querySelectorAll('.bubble.assistant').length")
                t = time.perf_counter()
                for kind in ("keyDown", "keyUp"):
                    await cdp.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13})
                await cdp.wait_for("document.querySelectorAll('.bubble.assistant').length > %d && !document.querySelector('.typing') && document.querySelectorAll('.sources .chip').length > 0" % n0, timeout=60)
                out["question_ms"].append(round((time.perf_counter() - t) * 1000, 1))
            print("question", out["question_ms"], flush=True)
            for _ in range(5):
                await cdp.send("Page.navigate", {"url": base + "/history?lang=en"})
                await cdp.wait_for("!!document.querySelector('.hs-row')", timeout=60)
                await asyncio.sleep(0.4)
                t = time.perf_counter()
                await cdp.evaluate("(() => { const b = document.querySelector('.hs-row .hs-title, .hs-row button'); if (b) b.click(); })()")
                await cdp.wait_for("location.pathname === '/' && document.querySelectorAll('.bubble.assistant').length > 0", timeout=60)
                out["switch_ms"].append(round((time.perf_counter() - t) * 1000, 1))
            print("switch", out["switch_ms"], flush=True)
    finally:
        proc.kill()
    out["question"] = stats(out["question_ms"])
    out["switch"] = stats(out["switch_ms"])
    return out


def http_part(base):
    res, lock = [], threading.Lock()

    def get():
        t = time.perf_counter()
        with urllib.request.urlopen(base + "/?lang=en", timeout=30) as r:
            r.read(1)
            first = (time.perf_counter() - t) * 1000
            r.read()
        with lock:
            res.append((first, (time.perf_counter() - t) * 1000, r.status))
    threads = [threading.Thread(target=get) for _ in range(10)]
    t0 = time.perf_counter()
    [t.start() for t in threads]
    [t.join() for t in threads]
    return {"requests": len(res), "status": sorted({s for _f, _t, s in res}), "first_byte": stats([f for f, _t, _s in res]),
            "total": stats([t for _f, t, _s in res]), "wall_ms": round((time.perf_counter() - t0) * 1000, 1)}


def finder_part():
    from onboarding import finder
    queries = ["library", "blackboard", "human resources", "grade transfer", "ithenticate", "conference", "المكتبة", "research", "phone", "attendance"]
    fn = getattr(finder, "search", None)
    if fn is None:
        return {"skipped": "onboarding.finder.search not found"}
    xs = []
    for i in range(200):
        t = time.perf_counter()
        fn(queries[i % len(queries)])
        xs.append((time.perf_counter() - t) * 1000)
    return stats(xs)


if __name__ == "__main__":
    base, out_dir = sys.argv[1].rstrip("/"), sys.argv[2]
    record = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "browser": asyncio.run(browser_part(base)), "http_concurrent": http_part(base)}
    try:
        record["finder_search"] = finder_part()
    except Exception as exc:                       # recorded, not raised
        record["finder_search"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "perf_probe.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(json.dumps({k: v for k, v in record.items() if k != "browser"}, indent=1))
