"""PF-16 bounded live evaluation: real Groq inference for a small, chosen subset, with the budget enforced in code.

    python -X utf8 qa/pf16_audit/live_eval.py <repo> <out dir> --budget 20 --items ITEMS.json [--pause SECONDS]

ITEMS.json is a list of {"id", "question", "language", "expected_facts", "evidence": [[source_id, page], ...],
"forbidden_facts", "why"} in priority order.

Budget enforcement (three layers, none of which a retry can bypass):
  1. a ledger file ``<out>/groq_ledger.json`` keeps the number of requests already issued by every earlier run; this run's
     cap is ``budget - used``, and the run stops before it starts when nothing is left;
  2. ``LLM_PROCESS_CALL_CAP`` is set to that remainder before the configuration is imported, so the application's own
     ``CappedClient`` refuses every completion request beyond it (a refused request never reaches the network);
  3. the Groq SDK never repeats an HTTP request on its own (every caller uses ``max_retries=0`` since PF-16.5), so one
     counted call is exactly one request; ``LLM_TRANSPORT_RETRIES=1`` lets the application repeat a TRANSIENT failure
     (connection, timeout, 429, 5xx) once, and that repeat is an ordinary request through the same capped, counted client.
The ledger is written to disk BEFORE every request is sent (PF-16.5): a run that is killed mid-question has still counted
every request it sent, and the next run's cap is computed from that count. It is rewritten again after every question. The key is read from ``.env`` by python-dotenv, exactly as ``app.py`` does,
and is never printed, logged or written: the records carry only whether a client was configured.

Configuration: ACTIVE, guide scopes on, Arabic answer translation on, Arabic query model fallback on (the configuration of
the owner's manual acceptance plus the PF-11 translation layer), so a live Arabic answer exercises the real translation and
its deterministic check.
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
ARGS = sys.argv[3:]
BUDGET = int(ARGS[ARGS.index("--budget") + 1]) if "--budget" in ARGS else 20
ITEMS = ARGS[ARGS.index("--items") + 1]
#: PF-16.6: seconds to wait after a question that sent requests (the PF-16.5 run hit the per-minute rate limit at 30 requests in 102 s)
PAUSE = float(ARGS[ARGS.index("--pause") + 1]) if "--pause" in ARGS else 0.0
os.chdir(REPO)
sys.path.insert(0, REPO)
os.makedirs(OUT, exist_ok=True)
LEDGER = os.path.join(OUT, "groq_ledger.json")
ledger = {"authorised": BUDGET, "used": 0, "runs": []}
if os.path.exists(LEDGER):
    with open(LEDGER, encoding="utf-8") as f:
        ledger = json.load(f)
remaining = int(ledger["authorised"]) - int(ledger["used"])
if remaining <= 0:
    print("budget exhausted: %d of %d used" % (ledger["used"], ledger["authorised"]))
    sys.exit(0)

from dotenv import load_dotenv   # noqa: E402  (the same loader app.py uses; nothing is printed)

load_dotenv(os.path.join(REPO, ".env"))
for name in ("PLAN_F_SHADOW",):
    os.environ.pop(name, None)
os.environ.update({"PLAN_F_ENABLED": "1", "PLAN_F_GUIDE_SCOPES": "1", "ARABIC_SUPPORT": "1", "ARABIC_ANSWER_TRANSLATION": "1",
                   "ARABIC_QUERY_MODEL_FALLBACK": "1", "LLM_PROCESS_CALL_CAP": str(remaining), "LLM_TRANSPORT_RETRIES": "1"})

from handbook_bot.groq_check import CappedClient          # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base  # noqa: E402
from ui import pipeline                                    # noqa: E402

with open(ITEMS, encoding="utf-8") as f:
    items = json.load(f)
kb = build_knowledge_base(verbose=False)
client = kb.groq_client
if not isinstance(client, CappedClient):
    print("no capped client (key missing or cap not applied); nothing is run")
    sys.exit(1)
print("mode=%s cap this run=%d (ledger %d of %d used) model=%s" % (pipeline.plan_f_mode(), client.cap, ledger["used"], ledger["authorised"],
                                                                     os.environ.get("GROQ_MODEL", "")), flush=True)
run = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "cap": client.cap, "model": os.environ.get("GROQ_MODEL", ""), "records": []}
ledger["runs"].append(run)
BASE_USED = int(ledger["used"])


def save():
    ledger["used"] = int(ledger["used"])
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=1, default=str)


from handbook_bot import groq_check as _gc                 # noqa: E402
_capped_create = _gc._CappedCompletions.create


def _create_counted_on_disk(self, **kwargs):
    # every client made by with_options shares this counter; a request the cap allows is on disk before it is sent
    if self._counter.issued < self._counter.cap:
        ledger["used"] = BASE_USED + self._counter.issued + 1
        save()
    return _capped_create(self, **kwargs)


_gc._CappedCompletions.create = _create_counted_on_disk


for item in items:
    if client.calls_issued >= client.cap:
        run["records"].append({"id": item["id"], "status": "not_run_budget"})
        print("%-14s not run (budget)" % item["id"], flush=True)
        continue
    before_issued, before_refused = client.calls_issued, client.calls_refused
    t = time.perf_counter()
    try:
        result, diag = pipeline.ask_with_diagnostics(kb, item["question"])
        error = None
    except Exception as exc:                                       # recorded, never re-raised
        result, diag, error = None, {}, "%s: %s" % (type(exc).__name__, str(exc)[:200])
    issued = client.calls_issued - before_issued
    ledger["used"] = BASE_USED + client.calls_issued
    rec = {"id": item["id"], "question": item["question"], "language": item.get("language"), "why": item.get("why"),
           "requests_issued": issued, "requests_refused_by_cap": client.calls_refused - before_refused, "error": error,
           "ms": round((time.perf_counter() - t) * 1000, 1)}
    if result is not None:
        meta_items = [(i.get("meta") or {}) for i in (result.items or []) if isinstance(i, dict)]
        rec.update({"answer": result.answer, "original_answer": result.original_answer, "answer_language": result.answer_language,
                    "pages": result.pages, "item_pages": [[m.get("source_id"), m.get("page")] for m in meta_items],
                    "served": diag.get("served"), "outcome": diag.get("outcome"), "routing_final": diag.get("routing_final"),
                    "fallback_reason": diag.get("fallback_reason") or diag.get("reason"), "llm_calls": result.llm_calls,
                    "translation": result.translation, "transform": result.query_transform, "used_llm": result.used_llm,
                    "expected_facts": item.get("expected_facts"), "expected_evidence": item.get("evidence"), "forbidden_facts": item.get("forbidden_facts")})
        rec["trace"] = [[getattr(e, "agent", ""), getattr(e, "decision", ""),
                         {k: v for k, v in (getattr(e, "detail", None) or {}).items() if k in ("reason", "error", "kind", "error_kind", "problem")}]
                        for e in (result.agent_trace or [])]
        text = (result.original_answer or result.answer or "").lower()
        rec["facts_found"] = [x for x in item.get("expected_facts") or [] if x.lower() in text]
        rec["forbidden_present"] = [x for x in item.get("forbidden_facts") or [] if x.lower() in (result.answer or "").lower() + " " + text]
    run["records"].append(rec)
    save()
    if PAUSE and issued:
        time.sleep(PAUSE)
    print("%-14s requests=%d served=%s pages=%s facts=%s/%s" % (item["id"], issued, rec.get("served"), rec.get("pages"), len(rec.get("facts_found") or []),
                                                                 len(item.get("expected_facts") or [])), flush=True)
run["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
run["issued_this_run"] = sum(r.get("requests_issued", 0) for r in run["records"])
save()
print("issued this run: %d; ledger: %d of %d" % (run["issued_this_run"], ledger["used"], ledger["authorised"]))
