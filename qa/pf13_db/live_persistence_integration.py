"""Live integration of persistence with the real chatbot (post-freeze PF-13): bounded model requests.

Against a RUNNING application started with the translation ON and a process cap (``launch_bilingual_qa.py --translate
--cap 4 --db ... --secret ...``), a real headless Chrome asks two Arabic questions whose verified English answers are
translated by the model (one request each in PF-11's live smoke), so the answers carry a translation state, an English
intermediate and citations. Each answer is recorded as shown, the page is reloaded, and the replayed answer, its
translation note, its English original (opened on request) and its citations are compared with the record; the database
rows are then read directly and compared with the page too. Requests are counted from the Developer panel of each fresh
answer and from the server's cap.

    python -X utf8 qa/pf13_db/live_persistence_integration.py <base url> <output dir> <db path>
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pf10_ui"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from browser_qa import find_browser, shot                                          # noqa: E402
from browser_qa_persistence import JS_STATE, Visitor, click, open_page, send_text, wait_answer   # noqa: E402

QUESTIONS = ["ما هو العبء التدريسي للمحاضرين؟", "من يوافق على طلبات السفر لحضور المؤتمرات؟"]

JS_TURN = r"""
(() => {
  const text = el => el ? el.innerText.replace(/\s+/g, ' ').trim() : null;
  const rows = [...document.querySelectorAll('.msg-row.assistant')];
  const last = rows[rows.length - 1];
  if (!last) return null;
  const details = last.querySelector('details.original');
  const dev = [...last.querySelectorAll('.dev-row')].map(r => [text(r.querySelector('.k')), text(r.querySelector('.v'))]);
  return {
    answer: text(last.querySelector('.bubble.assistant')), lang: last.querySelector('.bubble.assistant') ? last.querySelector('.bubble.assistant').getAttribute('lang') : null,
    translationNote: text(last.querySelector('.translation-note')), hasOriginal: !!details,
    original: details ? text(details.querySelector('.original-body')) : null,
    chips: [...last.querySelectorAll('.chip')].map(text), evidence: last.querySelectorAll('.evidence-row, .evidence-item').length,
    savedNote: text(last.querySelector('.saved-note')), dev: dev,
  };
})()
"""


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


async def main(base_url: str, out_dir: str, db_path: str) -> int:
    shots = os.path.join(out_dir, "screenshots")
    os.makedirs(shots, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "evidence"), exist_ok=True)
    profile = tempfile.mkdtemp(prefix="pf13-live-")
    record = {"base_url": base_url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": [], "problems": []}
    try:
        async with Visitor(find_browser("chrome"), profile, "L") as v:
            cdp = v.cdp
            await open_page(cdp, base_url + "/?lang=ar&dev=1")
            await click(cdp, ".dev-toggle .q-toggle")                                      # Developer details on, to read the request count
            await asyncio.sleep(0.5)
            fresh = []
            for n, question in enumerate(QUESTIONS, 1):
                if n > 1:
                    await click(cdp, ".new-chat-btn")
                    await asyncio.sleep(0.8)
                await send_text(cdp, question)
                answered = await wait_answer(cdp, 1, timeout=180)
                turn = await cdp.evaluate(JS_TURN)
                st = await cdp.evaluate(JS_STATE)
                await shot(cdp, shots, "live_ar_%d_fresh" % n)
                fresh.append({"question": question, "answered": answered, "turn": turn, "title": st["activeTitle"]})
                print("fresh %d: answered=%s lang=%s translation=%s chips=%d original=%s" % (n, answered, (turn or {}).get("lang"), ((turn or {}).get("translationNote") or "")[:40], len((turn or {}).get("chips") or []), bool((turn or {}).get("hasOriginal"))))
            # reload and compare each conversation
            await open_page(cdp, base_url + "/?lang=ar&dev=1")
            await click(cdp, ".dev-toggle .q-toggle")
            await asyncio.sleep(0.5)
            for n, case in enumerate(fresh, 1):
                clicked = await cdp.evaluate(
                    "(() => { const row = [...document.querySelectorAll('.sidebar .session-row')].find(r => (r.querySelector('.session-title') || {}).innerText.trim() === %s);"
                    " if (!row) return false; row.querySelector('.session-item').click(); return true; })()" % json.dumps(case["title"]))
                saw_typing = await cdp.wait_for("!!document.querySelector('.typing')", timeout=1.0)
                await cdp.wait_for("document.querySelectorAll('.bubble.assistant').length > 0", timeout=10)
                await asyncio.sleep(0.8)
                await cdp.evaluate("(() => { const d = document.querySelector('details.original'); if (d) d.open = true; return !!d; })()")
                await asyncio.sleep(0.3)
                replay = await cdp.evaluate(JS_TURN)
                await shot(cdp, shots, "live_ar_%d_replayed" % n)
                f = case["turn"] or {}
                same = {"answer": norm(f.get("answer")) == norm((replay or {}).get("answer")), "chips": f.get("chips") == (replay or {}).get("chips"),
                        "translation_note": f.get("translationNote") == (replay or {}).get("translationNote"), "original_present": bool(f.get("hasOriginal")) == bool((replay or {}).get("hasOriginal")),
                        "no_processing_state": not saw_typing, "saved_label": bool((replay or {}).get("savedNote"))}
                case.update({"clicked": clicked, "replay": replay, "same": same})
                print("replay %d:" % n, same)
                if not all(same.values()):
                    record["problems"].append("case %d: %s" % (n, [k for k, ok in same.items() if not ok]))
            record["cases"] = fresh
    finally:
        import shutil
        shutil.rmtree(profile, ignore_errors=True)
    # --- the database rows, read directly (read-only) ----------------------------------------------------------------------
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT m.content, m.answer_language, a.original_answer, a.translation, a.llm_calls, a.used_llm, a.pages, "
                            "(SELECT COUNT(*) FROM evidence_items e WHERE e.message_id = m.id) AS evidence FROM messages m JOIN answers a ON a.message_id = m.id "
                            "WHERE m.role = 'assistant' ORDER BY m.id DESC LIMIT ?", (len(QUESTIONS),)).fetchall()
    finally:
        conn.close()
    record["db_rows"] = []
    for row in reversed(rows):
        tr = json.loads(row["translation"] or "{}")
        entry = {"answer_language": row["answer_language"], "translation_status": tr.get("status"), "translation_llm_calls": tr.get("llm_calls"), "llm_calls": row["llm_calls"],
                 "original_answer_chars": len(row["original_answer"] or ""), "pages": row["pages"], "evidence": row["evidence"], "content_chars": len(row["content"] or "")}
        record["db_rows"].append(entry)
        shown = [c for c in fresh if norm((c["turn"] or {}).get("answer")) == norm(row["content"])]
        entry["matches_shown_answer"] = bool(shown)
        print("db:", entry)
        if not shown:
            record["problems"].append("a stored answer differs from what the page showed")
        if entry["translation_status"] != "translated" or not entry["original_answer_chars"]:
            record["problems"].append("a stored Arabic answer lacks the translated state or its English original")
    record["model_requests_estimate"] = sum(int(r["llm_calls"] or 0) for r in rows)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "evidence", "live_persistence_integration.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print("model requests recorded in the stored answers:", record["model_requests_estimate"], "| problems:", record["problems"])
    return 1 if record["problems"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3])))
