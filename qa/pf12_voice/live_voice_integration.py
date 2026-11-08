"""Voice-to-answer integration with REAL model answers under a request cap (post-freeze PF-12, phase K).

    python -X utf8 qa/pf12_voice/live_voice_integration.py <base url> <out dir>

The page runs in real Chrome with the MOCKED recogniser of ``browser_qa_voice.py`` (the recogniser's transcript is a
fixture, so this is NOT a speech-recognition test); the transcript goes into the composer exactly as the real
recogniser would deliver it, the harness presses Enter, and the application answers through its real pipeline with the
model translation ON (server launched with ``launch_bilingual_qa.py --translate --cap N``). Recorded per case: the fixture
transcript, the question the page sent (the user bubble), the answer shown, the translation note, the chips (sources),
and the server's request counter read before and after from the Developer panel is NOT available here, so the cap log
of the server (``launch_voice_live.log``) and the answer presentation are the request evidence."""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pf10_ui"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from browser_qa import CDP, find_browser, launch, shot                      # noqa: E402
from browser_qa_voice import FAKE_RECOGNISER, JS_VOICE, click_mic, open_page, send_enter, wait_state   # noqa: E402

CASES = [
    {"id": "V01", "lang": "ar", "transcript": "ما هو رقم هاتف قسم الموارد البشرية؟", "expect_path": "specialist", "expect_requests": 1,
     "note": "Arabic voice transcript -> PF-11 transformation -> directory row -> translated answer"},
    {"id": "V02", "lang": "ar", "transcript": "كيف أرسل إعلانًا للطلبة عبر Blackboard؟", "expect_path": "specialist", "expect_requests": 1,
     "note": "Arabic mixed transcript -> Blackboard guide steps -> translated answer"},
    {"id": "V03", "lang": "en", "transcript": "What is the policy on plagiarism at the University of Sharjah?", "expect_path": "model", "expect_requests": 1,
     "note": "English voice transcript -> legacy model path -> English answer (no translation)"},
]


async def run(base_url: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    exe = find_browser("chrome")
    profile = tempfile.mkdtemp(prefix="pf12-live-")
    proc, ws_url = launch(exe, profile)
    record = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "base_url": base_url, "recogniser": "MOCKED fixture transcript (no speech recognised)", "cases": []}
    try:
        async with CDP(ws_url) as cdp:
            targets = (await cdp.send("Target.getTargets"))["targetInfos"]
            page = next(t for t in targets if t["type"] == "page")
            cdp.session = (await cdp.send("Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}))["sessionId"]
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            for case in CASES:
                scenario = "window.__voiceScenario = {kind: 'final', en: %s, ar: %s, mixed: %s, delayMs: 300};" % (
                    json.dumps(case["transcript"]), json.dumps(case["transcript"]), json.dumps(case["transcript"]))
                await cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": scenario + FAKE_RECOGNISER})
                await open_page(cdp, base_url, case["lang"], 1440, 900)
                t0 = time.perf_counter()
                await click_mic(cdp)
                ready = await wait_state(cdp, "ready", 10)
                box = await cdp.evaluate("document.querySelector('.composer textarea').value")
                before = await cdp.evaluate("document.querySelectorAll('.msg-row.user').length")
                await send_enter(cdp)
                answered = await cdp.wait_for("!document.querySelector('.typing') && document.querySelectorAll('.bubble.assistant').length > 0", timeout=180)
                await asyncio.sleep(1.0)
                ms = (time.perf_counter() - t0) * 1000.0
                info = await cdp.evaluate(JS_VOICE)
                answer = await cdp.evaluate("(() => { const b = document.querySelectorAll('.bubble.assistant'); return b.length ? b[b.length-1].innerText : null; })()")
                question = await cdp.evaluate("(() => { const b = document.querySelector('.bubble.user'); return b ? b.innerText : null; })()")
                chips = await cdp.evaluate("[...document.querySelectorAll('.chip')].map(c => c.innerText.trim())")
                note = await cdp.evaluate("(() => { const n = document.querySelector('.translation-note'); return n ? n.innerText.trim() : null; })()")
                original = await cdp.evaluate("(() => { const d = document.querySelector('details.original .original-body'); return d ? d.innerText.trim() : null; })()")
                shot_name = await shot(cdp, out_dir, "live_%s_%s" % (case["id"], case["lang"]))
                record["cases"].append(dict(case, transcript_ready=ready, composer_before_send=box, sent=(await cdp.evaluate("document.querySelectorAll('.msg-row.user').length")) - before,
                                            answered=answered, question_sent=question, answer_shown=answer, chips=chips, translation_note=note,
                                            english_source=original, help=info.get("help"), ms=round(ms), shot=shot_name))
                print("%s %s ready=%s sent=%s answered=%s chips=%d note=%s\n   question: %s\n   answer: %s" % (
                    case["id"], case["lang"], ready, record["cases"][-1]["sent"], answered, len(chips), note, question, (answer or "").replace("\n", " / ")[:300]), flush=True)
    finally:
        proc.kill()
        shutil.rmtree(profile, ignore_errors=True)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "live_voice_integration.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    return record


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1].rstrip("/"), sys.argv[2]))
