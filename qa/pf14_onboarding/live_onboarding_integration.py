"""Bounded live integration of the onboarding features with the real chatbot (post-freeze PF-14).

Against a RUNNING application started with ``launch_bilingual_qa.py --translate --cap 6 --db ... --secret ...``, a real
headless Chrome asks three contextual questions the way the pages ask them (the chat page opened with ``?ask=&task=``,
Developer details on), one per feature entry point:

    1. from the procedure navigator: "Ask about this step" on the announcement guide (Teaching & Learning expected)
    2. from the checklist: "Ask the assistant" on the ILHE training task (Institutional / Faculty development expected)
    3. the same ILHE task question in Arabic (the PF-11 layer: transformation, specialist answer, translation)

For each: the question sent, the task context stored with the conversation, the routing read from the Developer panel
(served path, specialist), the citations, whether the answer is a refusal or notice, and the model requests the stored
answer records. The database is read directly afterwards to confirm no progress row was created or changed by any
answer. Budget: at most six requests in the whole run (the server's process cap enforces it).

    python -X utf8 qa/pf14_onboarding/live_onboarding_integration.py <base url> <output dir> <db path>
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from urllib.parse import quote, urlencode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pf10_ui"))
sys.path.insert(0, os.path.join(HERE, "..", "pf13_db"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from browser_qa import find_browser, shot                                              # noqa: E402
from browser_qa_persistence import Visitor, click                                     # noqa: E402
from browser_qa_onboarding import JS_PAGE, open_page, send_enter, wait_answer         # noqa: E402
from onboarding import catalog as C                                                    # noqa: E402
from ui import i18n                                                                    # noqa: E402

STEP = C.PROCEDURES["bb_announcement"].steps[6]                                         # "Click on Post so the announcement will reach students."
CASES = [
    {"name": "navigator_ask_step", "task": "bb_announcement", "lang": "en", "expected_specialist": "teaching",
     "question": i18n.t("en", "pr_question", guide=C.PROCEDURES["bb_announcement"].heading, step=STEP.text)},
    {"name": "checklist_ask_task_en", "task": "ilhe_training", "lang": "en", "expected_specialist": "institutional|faculty_services",
     "question": i18n.t("en", "ob_task_question", task=C.TASKS["ilhe_training"].title_en)},
    {"name": "checklist_ask_task_ar", "task": "ilhe_training", "lang": "ar", "expected_specialist": "institutional|faculty_services",
     "question": i18n.t("ar", "ob_task_question", task=C.TASKS["ilhe_training"].title_en)},
]

JS_DEV = r"""
(() => {
  const rows = [...document.querySelectorAll('.msg-row.assistant')];
  const last = rows[rows.length - 1];
  if (!last) return null;
  const dev = {};
  last.querySelectorAll('.dev-row').forEach(r => { const k = r.querySelector('.k'), v = r.querySelector('.v'); if (k && v) dev[k.innerText.trim()] = v.innerText.trim(); });
  const notice = last.querySelector('.bubble.notice');
  return {dev, notice: notice ? notice.innerText.trim().slice(0, 120) : null, translation: (last.querySelector('.translation-note') || {}).innerText || null};
})()
"""


async def main(base_url: str, out_dir: str, db_path: str, only: str = "") -> int:
    cases = [c for c in CASES if not only or c["name"] == only]
    shots = os.path.join(out_dir, "screenshots")
    os.makedirs(shots, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "evidence"), exist_ok=True)
    profile = tempfile.mkdtemp(prefix="pf14-live-")
    record = {"base_url": base_url, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": [], "problems": []}
    before = _progress_rows(db_path)
    try:
        async with Visitor(find_browser("chrome"), profile, "L") as v:
            cdp = v.cdp
            for n, case in enumerate(cases, 1):
                url = base_url + "/?" + urlencode({"ask": case["question"], "task": case["task"], "lang": case["lang"], "dev": "1"}, quote_via=quote)
                await open_page(cdp, url, wait="!!document.querySelector('.composer textarea')")
                await click(cdp, ".dev-toggle .q-toggle")
                await asyncio.sleep(0.5)
                s0 = await cdp.evaluate(JS_PAGE)
                prefilled = (s0["composer"] or "") == case["question"]
                await send_enter(cdp)
                answered = await wait_answer(cdp, timeout=180)
                s1 = await cdp.evaluate(JS_PAGE)
                dev = await cdp.evaluate(JS_DEV) or {}
                await shot(cdp, shots, "L_%d_%s" % (n, case["name"]))
                entry = {**case, "prefilled": prefilled, "context_shown": bool(s1["chatContext"]), "answered": answered, "answer": s1["lastAnswer"], "chips": s1["chips"],
                         "notice": dev.get("notice"), "translation_note": dev.get("translation"), "developer": dev.get("dev", {})}
                record["cases"].append(entry)
                print("case %d %s: prefilled=%s answered=%s notice=%s chips=%s" % (n, case["name"], prefilled, answered, entry["notice"], s1["chips"][:3]))
                print("   developer:", {k: v for k, v in (dev.get("dev") or {}).items() if k.lower() in ("served", "served by", "path", "specialist", "specialists", "routing", "model calls", "llm calls", "language", "outcome", "pages")})
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    after = _progress_rows(db_path)
    record["progress_rows_before"], record["progress_rows_after"] = before, after
    if before != after:
        record["problems"].append("progress rows changed during the live run")
    # the stored conversations: context, answer path, model usage
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT c.context_task, m.content, m.answer_language, a.served, a.specialist, a.plan_f_path, a.pages, a.llm_calls, a.translation, a.diagnostics "
                            "FROM conversations c JOIN messages m ON m.conversation_id = c.id JOIN answers a ON a.message_id = m.id WHERE m.role = 'assistant' "
                            "ORDER BY m.id DESC LIMIT ?", (len(cases),)).fetchall()
    finally:
        conn.close()
    record["db_rows"] = []
    total_calls = 0
    for row in reversed(rows):
        diag = json.loads(row["diagnostics"] or "{}")
        tr = json.loads(row["translation"] or "{}")
        calls = int(row["llm_calls"] or 0) + int(tr.get("llm_calls") or 0)
        total_calls += calls
        entry = {"context_task": row["context_task"], "answer_language": row["answer_language"], "served": row["served"], "specialist": row["specialist"] or diag.get("specialist_id"),
                 "routing_final": diag.get("routing_final"), "plan_f_path": row["plan_f_path"], "pages": row["pages"], "answer_llm_calls": row["llm_calls"], "translation_status": tr.get("status"),
                 "translation_llm_calls": tr.get("llm_calls"), "content_chars": len(row["content"] or ""),
                 "outcome": diag.get("outcome"), "fallback_reason": diag.get("fallback_reason"), "findings_reason": diag.get("findings_reason"), "language": diag.get("language"),
                 "plan_f_pages": diag.get("plan_f_pages"), "legacy_pages": diag.get("legacy_pages")}
        record["db_rows"].append(entry)
        print("db:", entry)
    record["model_requests_recorded"] = total_calls
    if total_calls > 6:
        record["problems"].append("more than six model requests recorded")
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    name = "live_onboarding_integration%s.json" % ("_" + only if only else "")
    with open(os.path.join(out_dir, "evidence", name), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)
    print("model requests recorded:", total_calls, "| problems:", record["problems"])
    return 1 if record["problems"] else 0


def _progress_rows(db_path: str):
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    try:
        return [tuple(r) for r in conn.execute("SELECT owner_id, task_id, status, updated_at FROM task_progress ORDER BY owner_id, task_id")]
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")))
