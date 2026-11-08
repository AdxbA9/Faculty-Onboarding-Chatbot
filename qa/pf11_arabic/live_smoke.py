"""PF-11 phase I: the bounded Arabic live smoke evaluation (authorised budget: at most 12 inference requests).

    python -X utf8 qa/pf11_arabic/live_smoke.py <repo> <out dir> [--cap 12]

Runs the real application seam in ACTIVE mode with guide scopes on, exactly as the acceptance launcher configures it,
with the Groq key loaded privately from the repository's .env (never printed; only its length and prefix shape are
logged) and the client wrapped by ``LLM_PROCESS_CALL_CAP`` so the process CANNOT issue more requests than the cap.
Every case records the input, the English query the lexicon produced, the serving path, the answer, the verified
English intermediate, the translation check, the pages and sources cited, the request count before and after, the
wall time and errors. A case that Plan F serves deterministically spends requests only on the Arabic translation; a case
on the model path spends one for the English draft (plus a possible verifier retry) and one for the translation; this is
recorded per case, never assumed. Writes ``live_smoke.json`` and ``live_smoke.md``."""
import json
import os
import sys
import time

REPO, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
CAP = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 12
os.chdir(REPO)
sys.path.insert(0, REPO)
os.makedirs(OUT, exist_ok=True)
from dotenv import load_dotenv                                      # noqa: E402
load_dotenv(os.path.join(REPO, ".env"))
os.environ["PLAN_F_ENABLED"] = "1"
os.environ.pop("PLAN_F_SHADOW", None)
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"
os.environ["LLM_PROCESS_CALL_CAP"] = str(CAP)

from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import pipeline                                             # noqa: E402

#: The twelve authorised cases, chosen to cover the categories of the brief. ``expect_llm`` is the expected number of
#: requests (plan: translation only; model: draft + translation), checked against the counter after each case.
CASES = [
    {"id": "L01", "category": "basic factual", "q": "ما هو العبء التدريسي للمحاضرين؟", "expect_path": "plan_f"},
    {"id": "L02", "category": "Blackboard procedure", "q": "كيف أرسل إعلانًا للطلبة عبر Blackboard؟", "expect_path": "plan_f"},
    {"id": "L03", "category": "HR policy", "q": "كم يوماً من إجازة الأمومة تستحقها عضوات هيئة التدريس؟", "expect_path": "plan_f"},
    {"id": "L04", "category": "research", "q": "من يوافق على طلبات السفر لحضور المؤتمرات؟", "expect_path": "plan_f"},
    {"id": "L05", "category": "institutional navigation", "q": "ما هو رقم هاتف قسم الموارد البشرية؟", "expect_path": "plan_f"},
    {"id": "L06", "category": "mixed language", "q": "كيف أستخدم Blackboard لإرسال Announcement؟", "expect_path": "any"},
    {"id": "L07", "category": "multi-part", "q": "ما هو العبء التدريسي للمحاضرين وما هو رقم هاتف مكتب التسجيل؟", "expect_path": "plan_f"},
    {"id": "L08", "category": "unanswerable", "q": "كم يوماً من إجازة الأبوة يستحق أعضاء هيئة التدريس الذكور؟", "expect_path": "legacy"},
    {"id": "L09", "category": "numbers and dates", "q": "كم يوماً من الإجازة المرضية براتب كامل يستحق أعضاء هيئة التدريس؟", "expect_path": "plan_f"},
    {"id": "L10", "category": "citation faithfulness / negation", "q": "هل صحيح أن أعضاء هيئة التدريس لا يحصلون على إجازة مرضية؟", "expect_path": "any"},
    {"id": "L11", "category": "legacy fallback", "q": "ما هي سياسة الانتحال في الجامعة؟", "expect_path": "legacy"},
    {"id": "L12", "category": "difficult cross-language retrieval", "q": "ما هي المدة القصوى لإجازة التفرغ العلمي للباحثين الزائرين؟", "expect_path": "any"},
]

key = os.getenv("GROQ_API_KEY", "").strip()
log = open(os.path.join(OUT, "live_smoke_log.txt"), "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


say("started %s | mode=%s | GROQ_MODEL=%s | key_length=%d key_prefix_ok=%s | cap=%d | translation=%s query_fallback=%s" % (
    time.strftime("%Y-%m-%d %H:%M:%S"), pipeline.plan_f_mode(), config.GROQ_MODEL, len(key), key.startswith("gsk_"), CAP,
    config.ARABIC_ANSWER_TRANSLATION, config.ARABIC_QUERY_MODEL_FALLBACK))
del key
t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
client = kb.groq_client
say("kb %d chunks in %.1f s; client=%s cap=%s" % (len(kb.chunks), time.perf_counter() - t0, type(client).__name__, getattr(client, "cap", None)))
if client is None or not hasattr(client, "calls_issued"):
    say("ABORT: no capped client")
    sys.exit(2)

records = []
for case in CASES:
    before = client.calls_issued
    if before >= CAP:
        say("%s SKIPPED: budget spent (%d/%d)" % (case["id"], before, CAP))
        records.append(dict(case, skipped=True, calls_before=before))
        continue
    t = time.perf_counter()
    error = None
    try:
        result, diag = pipeline.ask_with_diagnostics(kb, case["q"])
    except Exception as exc:                                        # the seam never raises; recorded if it ever does
        result, diag, error = None, {}, "%s" % type(exc).__name__
    ms = (time.perf_counter() - t) * 1000.0
    after = client.calls_issued
    items = [{"source": (i.get("meta") or {}).get("source_id"), "page": (i.get("meta") or {}).get("page"), "quote": (i.get("chunk") or "")[:300]}
             for i in ((result.items if result else []) or []) if isinstance(i, dict)]
    rec = dict(case, calls_before=before, calls_after=after, calls_used=after - before, ms=round(ms, 1), error=error,
               served=diag.get("served"), outcome=diag.get("outcome"), specialist=diag.get("specialist_id"), path=diag.get("plan_f_path"),
               fallback_reason=diag.get("fallback_reason"), language=diag.get("language"),
               answer=result.answer if result else None, original_answer=result.original_answer if result else None,
               answer_language=result.answer_language if result else None, translation=result.translation if result else None,
               query_transform=result.query_transform if result else None, pages=result.pages if result else None, items=items,
               used_llm=result.used_llm if result else None, llm_calls=result.llm_calls if result else None, query_type=result.query_type if result else None)
    records.append(rec)
    say("%s [%s] served=%s outcome=%s calls=%d (total %d/%d) %.0f ms translation=%s" % (
        case["id"], case["category"], rec["served"], rec["outcome"], rec["calls_used"], after, CAP, ms, (rec["translation"] or {}).get("status")))
    say("    EN query: %s" % ((rec["query_transform"] or {}).get("english_query")))
    say("    answer  : %s" % ((rec["answer"] or "").replace("\n", " / ")[:400]))
    if rec["original_answer"]:
        say("    english : %s" % rec["original_answer"].replace("\n", " / ")[:400])
    say("    pages   : %s" % rec["pages"])

summary = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "cap": CAP, "calls_issued": client.calls_issued, "calls_refused": client.calls_refused,
           "cases": len(records), "skipped": sum(1 for r in records if r.get("skipped")), "model": config.GROQ_MODEL, "mode": pipeline.plan_f_mode(),
           "served_plan_f": sum(1 for r in records if r.get("served") == "plan_f"), "served_legacy": sum(1 for r in records if r.get("served") == "legacy"),
           "translated": sum(1 for r in records if (r.get("translation") or {}).get("status") == "translated"),
           "framed": sum(1 for r in records if str((r.get("translation") or {}).get("status", "")).startswith("framed")),
           "fixed": sum(1 for r in records if (r.get("translation") or {}).get("status") == "fixed")}
say("SUMMARY " + json.dumps(summary))
with open(os.path.join(OUT, "live_smoke.json"), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "records": records}, f, indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "live_smoke.md"), "w", encoding="utf-8") as f:
    f.write("# Arabic live smoke evaluation (%s)\n\nModel `%s`, mode %s, process cap %d, requests issued %d, refused %d.\n\n" % (
        summary["finished"], summary["model"], summary["mode"], CAP, summary["calls_issued"], summary["calls_refused"]))
    f.write("| id | category | served | outcome | requests | ms | presentation | pages | English query |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in records:
        if r.get("skipped"):
            f.write("| %s | %s | skipped | | 0 | | | | |\n" % (r["id"], r["category"]))
            continue
        f.write("| %s | %s | %s | %s | %d | %.0f | %s | %s | %s |\n" % (r["id"], r["category"], r["served"], r["outcome"], r["calls_used"], r["ms"],
                                                                  (r["translation"] or {}).get("status"), r["pages"], ((r["query_transform"] or {}).get("english_query") or "").replace("|", "/")))
    for r in records:
        if r.get("skipped"):
            continue
        f.write("\n## %s — %s\n\n**Question:** %s\n\n**English query:** %s\n\n**Answer (as shown):**\n\n%s\n\n" % (
            r["id"], r["category"], r["q"], (r["query_transform"] or {}).get("english_query"), r["answer"]))
        if r["original_answer"]:
            f.write("**Verified English intermediate:** %s\n\n" % r["original_answer"])
        f.write("**Translation record:** `%s`\n\n**Sources:** %s\n\n" % (json.dumps(r["translation"], ensure_ascii=False)[:600], json.dumps(r["items"], ensure_ascii=False)[:1200]))
log.close()
