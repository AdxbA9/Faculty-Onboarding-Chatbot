"""Does the installed browser's Web Speech API recognise REAL audio in this environment? (post-freeze PF-12, phase A/I)

    python -X utf8 qa/pf12_voice/real_recognition_probe.py <wav file> <lang> <out json> [--browser chrome|edge] [--headed]

Launches the browser with Chromium's fake audio device fed from the WAV file (``--use-fake-device-for-media-stream``,
``--use-file-for-fake-audio-capture``), serves a minimal page from a local HTTP server (localhost is a secure context),
grants the microphone permission through DevTools, and runs ``webkitSpeechRecognition`` with the given locale on the
GENUINE recogniser (no mock). Records every event with timestamps. The audio is synthetic (a system text-to-speech
voice), so a success shows that real recognition works end to end for that utterance; it says nothing about a human
voice or a real microphone, which stay a human acceptance test."""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pf10_ui"))
from browser_qa import BROWSERS, CDP   # noqa: E402

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>speech probe</title></head><body>
<script>
window.__events = [];
window.__log = (kind, extra) => { window.__events.push(Object.assign({t: Date.now(), kind: kind}, extra || {})); };
window.__run = (lang) => {
  const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Ctor) { window.__log('unsupported'); return 'unsupported'; }
  const r = new Ctor(); r.lang = lang; r.continuous = true; r.interimResults = true; r.maxAlternatives = 1;
  r.onstart = () => window.__log('start'); r.onaudiostart = () => window.__log('audiostart'); r.onsoundstart = () => window.__log('soundstart');
  r.onspeechstart = () => window.__log('speechstart'); r.onspeechend = () => window.__log('speechend'); r.onaudioend = () => window.__log('audioend');
  r.onresult = (ev) => { for (let i = ev.resultIndex; i < ev.results.length; i++) { window.__log('result', {final: ev.results[i].isFinal, text: ev.results[i][0].transcript, confidence: ev.results[i][0].confidence}); } };
  r.onerror = (ev) => window.__log('error', {code: ev.error, message: ev.message});
  r.onend = () => window.__log('end');
  r.onnomatch = () => window.__log('nomatch');
  window.__rec = r; r.start(); return 'started';
};
</script><p>speech probe</p></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def launch(exe, profile, wav, headed):
    args = [exe, "--remote-debugging-port=0", "--user-data-dir=" + profile, "--no-first-run", "--no-default-browser-check", "--disable-gpu",
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--use-file-for-fake-audio-capture=" + wav + "%noloop",
            "--autoplay-policy=no-user-gesture-required", "--window-size=900,600", "about:blank"]
    if not headed:
        args.insert(1, "--headless=new")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    ws = None
    deadline = time.time() + 30
    import re
    while time.time() < deadline and ws is None:
        line = proc.stdout.readline()
        m = re.search(r"DevTools listening on (ws://\S+)", line or "")
        if m:
            ws = m.group(1)
    if ws is None:
        proc.kill()
        raise SystemExit("DevTools endpoint not announced")
    return proc, ws


async def run(wav, lang, out, browser, headed, wait_s):
    exe = next(p for p in BROWSERS[browser] if os.path.exists(p))
    srv = serve()
    url = "http://127.0.0.1:%d/" % srv.server_port
    profile = tempfile.mkdtemp(prefix="pf12-speech-")
    proc, ws = launch(exe, profile, wav, headed)
    record = {"browser": exe, "headed": headed, "wav": wav, "lang": lang, "url": url, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        async with CDP(ws) as cdp:
            targets = (await cdp.send("Target.getTargets"))["targetInfos"]
            page = next(t for t in targets if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            try:
                await cdp.send("Browser.grantPermissions", {"origin": url.rstrip("/"), "permissions": ["audioCapture"]})
                record["permission_granted_via_devtools"] = True
            except Exception as exc:
                record["permission_granted_via_devtools"] = "failed: %s" % exc
            await cdp.send("Page.navigate", {"url": url})
            await cdp.wait_for("!!window.__run", timeout=20)
            record["supported"] = await cdp.evaluate("!!(window.SpeechRecognition || window.webkitSpeechRecognition)")
            record["user_agent"] = await cdp.evaluate("navigator.userAgent")
            record["start_result"] = await cdp.evaluate("window.__run(%s)" % json.dumps(lang))
            t0 = time.time()
            while time.time() - t0 < wait_s:
                events = await cdp.evaluate("window.__events")
                if any(e["kind"] in ("end", "error") for e in events) and (time.time() - t0) > 3:
                    break
                await asyncio.sleep(0.5)
            try:
                await cdp.evaluate("window.__rec && window.__rec.stop()")
                await asyncio.sleep(1.0)
            except Exception:
                pass
            record["events"] = await cdp.evaluate("window.__events")
    finally:
        proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
        srv.shutdown()
    finals = [e["text"] for e in record.get("events", []) if e["kind"] == "result" and e.get("final")]
    record["final_transcripts"] = finals
    record["errors"] = [e.get("code") for e in record.get("events", []) if e["kind"] == "error"]
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: record[k] for k in ("supported", "start_result", "final_transcripts", "errors", "permission_granted_via_devtools")}, ensure_ascii=False))
    for e in record.get("events", []):
        print("  ", e)
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("lang")
    ap.add_argument("out")
    ap.add_argument("--browser", choices=("chrome", "edge"), default="chrome")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--wait", type=float, default=25.0)
    a = ap.parse_args()
    asyncio.run(run(os.path.abspath(a.wav), a.lang, a.out, a.browser, a.headed, a.wait))
