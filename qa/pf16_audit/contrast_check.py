"""PF-16 measured text contrast of the running application, light and dark theme, every page (headless Chrome or Edge).

    python -X utf8 qa/pf16_audit/contrast_check.py <base url> <out dir> [--browser chrome|edge]

For every visible element that directly holds text, the computed text colour is compared with the first non-transparent
background colour among its ancestors (WCAG 2 relative-luminance contrast ratio). Thresholds: 4.5:1, or 3:1 for large
text (at least 24 px, or at least 18.66 px and bold). Elements inside images, SVG and hidden subtrees are skipped. This is a
measurement of computed colours, not a WCAG conformance audit (no manual review of states, focus, images or overlays).
Writes ``contrast.json``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pf10_ui"))
from browser_qa import CDP, find_browser, launch   # noqa: E402

PAGES = ["/", "/onboarding", "/procedures", "/onboarding/procedure/bb_announcement?task=bb_announcement", "/resources", "/history", "/profile", "/settings"]
JS = r"""
(() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(',').map(x => parseFloat(x));
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1}; };
  const lum = c => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const blend = (fg, bg) => ({r: fg.r * fg.a + bg.r * (1 - fg.a), g: fg.g * fg.a + bg.g * (1 - fg.a), b: fg.b * fg.a + bg.b * (1 - fg.a), a: 1});
  // every ancestor's background, composited from the page down (a translucent tint sits on what is under it, not on white)
  const bgOf = el => { const stack = []; let e = el; while (e && e.nodeType === 1) { const c = parse(getComputedStyle(e).backgroundColor); if (c && c.a > 0) stack.push(c); e = e.parentElement; }
    let acc = {r: 255, g: 255, b: 255, a: 1}; for (let i = stack.length - 1; i >= 0; i--) acc = blend(stack[i], acc); return acc; };
  const visible = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && parseFloat(s.opacity) > 0.05; };
  const out = [];
  document.querySelectorAll('body *').forEach(el => {
    if (el.closest('svg, img, script, style, [aria-hidden="true"], .q-dialog--minimized')) return;
    const own = [...el.childNodes].filter(n => n.nodeType === 3 && n.textContent.trim()).map(n => n.textContent.trim()).join(' ');
    if (!own || !visible(el)) return;
    const s = getComputedStyle(el); const fg = parse(s.color); if (!fg) return;
    let bg = bgOf(el); if (bg.a < 1) bg = blend(bg, {r: 255, g: 255, b: 255, a: 1});
    const f = blend(fg, bg); const L1 = lum(f), L2 = lum(bg);
    const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
    const size = parseFloat(s.fontSize); const bold = parseInt(s.fontWeight) >= 700;
    const large = size >= 24 || (size >= 18.66 && bold);
    out.push({text: own.slice(0, 50), cls: (el.className && el.className.baseVal === undefined ? String(el.className) : '').slice(0, 60), tag: el.tagName,
              ratio: Math.round(ratio * 100) / 100, need: large ? 3 : 4.5, size: size});
  });
  const fails = out.filter(o => o.ratio < o.need).sort((a, b) => a.ratio - b.ratio);
  return {elements: out.length, below: fails.length, min: out.length ? Math.min(...out.map(o => o.ratio)) : null, worst: fails.slice(0, 8),
          theme: document.documentElement.getAttribute('data-theme')};
})()
"""


async def run(base: str, out_dir: str, browser: str) -> dict:
    exe = find_browser(browser)
    profile = tempfile.mkdtemp(prefix="pf16-contrast-")
    proc, ws = launch(exe, profile)
    record = {"browser": exe, "base": base, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "pages": []}
    try:
        async with CDP(ws) as cdp:
            targets = (await cdp.send("Target.getTargets"))["targetInfos"]
            page = next(t for t in targets if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            for theme in ("light", "dark"):
                await cdp.send("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False})
                await cdp.send("Page.navigate", {"url": base + "/?lang=en"})
                await cdp.wait_for("!!document.querySelector('.composer textarea')", timeout=60)
                await asyncio.sleep(1.0)
                current = await cdp.evaluate("document.documentElement.getAttribute('data-theme')")
                if current != theme:
                    await cdp.evaluate("(() => { const b = document.querySelector('.topbar-btn.theme-btn, .theme-btn'); if (b) b.click(); })()")
                    await asyncio.sleep(1.2)
                for lang in ("en", "ar"):
                    for path in PAGES:
                        sep = "&" if "?" in path else "?"
                        await cdp.send("Page.navigate", {"url": base + path + sep + "lang=" + lang})
                        await cdp.wait_for("!!document.querySelector('.page-main, .composer textarea, .ob-main')", timeout=60)
                        await asyncio.sleep(1.0)
                        res = await cdp.evaluate(JS)
                        res.update({"path": path, "lang": lang, "requested_theme": theme})
                        record["pages"].append(res)
                        print("%-5s %-2s %-60s elements=%-4s below=%-3s min=%s" % (theme, lang, path, res["elements"], res["below"], res["min"]), flush=True)
    finally:
        proc.kill()
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "contrast_%s.json" % browser), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return record


if __name__ == "__main__":
    base, out = sys.argv[1].rstrip("/"), sys.argv[2]
    browser = sys.argv[sys.argv.index("--browser") + 1] if "--browser" in sys.argv else "chrome"
    rec = asyncio.run(run(base, out, browser))
    print("pages:", len(rec["pages"]), "with text below threshold:", sum(1 for p in rec["pages"] if p["below"]))
