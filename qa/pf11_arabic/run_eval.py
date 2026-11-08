"""PF-11 multilingual evaluation on the REAL knowledge base through the real application seam, offline (no model request).

    python -X utf8 qa/pf11_arabic/run_eval.py <repo> <out dir> [--split dev|holdout|all]

ACTIVE mode with guide scopes on (the acceptance configuration), no Groq key: Plan F's deterministic specialists answer
what they can; a question that needs the model comes back as the "not configured" notice and is recorded as
``model_path`` (it is evaluated live, within the authorised budget, by ``live_smoke.py``). For every paired question the
Arabic and the English version go through ``pipeline.ask_with_diagnostics`` and the records are compared:

    language detection, transformation method and coverage, the English query
    served path / outcome / specialist for AR and EN, and whether they agree
    pages cited for AR and EN, expected-page hit, source id
    the Arabic presentation (fixed reply / framed / translated) and whether the framed English text equals the EN answer
    behaviour against ``expect``: a ``fallback`` question must not be served; a ``serve`` question served by EN should be served by AR

Automated engineering checks only: nothing here judges Arabic fluency; the live smoke test and the human review do that.
Writes ``eval_<split>.json`` and ``eval_<split>.md``."""
import json
import os
import re
import sys
import time

REPO, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
SPLIT = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "qa", "pf11_arabic"))
for name in ("GROQ_API_KEY", "PLAN_F_SHADOW"):
    os.environ.pop(name, None)
os.environ["PLAN_F_ENABLED"] = "1"
os.environ["PLAN_F_GUIDE_SCOPES"] = "1"
os.makedirs(OUT, exist_ok=True)

from question_set import ALL                                         # noqa: E402
from handbook_bot.arabic_answers import REPLIES_AR, localize_reply   # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base        # noqa: E402
from handbook_bot.orchestrator import NO_KEY_REPLY                   # noqa: E402
from handbook_bot.qa import REFUSAL                                  # noqa: E402
from ui import pipeline                                              # noqa: E402

t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
print("mode=%s kb=%d chunks (%.1f s)" % (pipeline.plan_f_mode(), len(kb.chunks), time.perf_counter() - t0), flush=True)


def kind(answer):
    if answer == NO_KEY_REPLY:
        return "model_path"
    if answer == REFUSAL:
        return "refusal"
    if localize_reply(answer) is not None:
        return "notice"
    return "answer"


def run(question, lang):
    t = time.perf_counter()
    result, diag = pipeline.ask_with_diagnostics(kb, question, answer_language=lang)
    ms = (time.perf_counter() - t) * 1000.0
    pages = [[(i.get("meta") or {}).get("source_id"), int((i.get("meta") or {}).get("page"))] for i in (result.items or []) if isinstance(i, dict) and (i.get("meta") or {}).get("page") is not None]
    cited = [[(i.get("meta") or {}).get("source_id"), p] for p in (result.pages or []) for i in (result.items or [])[:1]] if result.items else [[None, p] for p in (result.pages or [])]
    return {"answer": result.answer, "original_answer": result.original_answer, "answer_language": result.answer_language, "kind": kind(result.original_answer or result.answer),
            "served": diag.get("served"), "outcome": diag.get("outcome"), "specialist": diag.get("specialist_id"), "path": diag.get("plan_f_path"),
            "routing_final": diag.get("routing_final"), "pages": result.pages, "item_pages": pages, "query_type": result.query_type,
            "llm_calls": result.llm_calls, "translation": result.translation, "transform": result.query_transform, "language": diag.get("language"), "ms": round(ms, 1)}


rows = []
for q in ALL:
    if SPLIT != "all" and q["split"] != SPLIT:
        continue
    ar = run(q["ar"], "auto")
    en = run(q["en"], "auto")
    expected = [[q["source"], p] for p in (q["pages"] or [])]
    ar_hit = any(e in ar["item_pages"] for e in expected) if expected else None
    en_hit = any(e in en["item_pages"] for e in expected) if expected else None
    ar_served = ar["served"] == "plan_f"
    en_served = en["served"] == "plan_f"
    framed_matches = (ar["translation"].get("status") in ("framed", "framed_after_check", "framed_after_error") and ar["original_answer"] == en["answer"]) if ar["kind"] == "answer" else None
    if q["expect"] == "fallback":
        behaviour_ok = not ar_served and ar["kind"] in ("refusal", "model_path", "notice")
    elif q["expect"] == "serve":
        behaviour_ok = ar_served == en_served and (not en_served or ar_hit is not False)
    else:
        behaviour_ok = (ar_served == en_served) or (ar["kind"] in ("refusal", "model_path"))
    rows.append({"id": q["id"], "pair": q["pair"], "domain": q["domain"], "category": q["category"], "expect": q["expect"], "split": q["split"], "mixed": q["mixed"],
                 "ar_question": q["ar"], "en_question": q["en"], "expected_pages": expected, "fact": q["fact"],
                 "language_detected": (ar["language"] or {}).get("detected"), "query_method": (ar["language"] or {}).get("query_method"),
                 "coverage": (ar["language"] or {}).get("query_coverage"), "english_query": (ar["transform"] or {}).get("english_query"),
                 "unmapped": (ar["transform"] or {}).get("unmapped"),
                 "ar": ar, "en": en, "ar_hit": ar_hit, "en_hit": en_hit, "served_agree": ar_served == en_served,
                 "pages_agree": (ar["pages"] == en["pages"]), "specialist_agree": ar["specialist"] == en["specialist"],
                 "framed_english_equals_en_answer": framed_matches, "behaviour_ok": behaviour_ok, "arabic_presentation": ar["translation"].get("status")})
    print("%-7s AR=%-10s EN=%-10s served %s/%s pages %s | hit AR=%s EN=%s | ok=%s | %s" % (
        q["id"], ar["kind"], en["kind"], ar["served"], en["served"], "=" if rows[-1]["pages_agree"] else "x", ar_hit, en_hit, behaviour_ok, rows[-1]["english_query"]), flush=True)

n = len(rows)
summary = {"split": SPLIT, "questions": n, "mode": pipeline.plan_f_mode(), "kb_chunks": len(kb.chunks),
           "ar_served_plan_f": sum(1 for r in rows if r["ar"]["served"] == "plan_f"), "en_served_plan_f": sum(1 for r in rows if r["en"]["served"] == "plan_f"),
           "served_agree": sum(1 for r in rows if r["served_agree"]), "pages_agree": sum(1 for r in rows if r["pages_agree"]),
           "specialist_agree": sum(1 for r in rows if r["specialist_agree"]),
           "ar_hit": sum(1 for r in rows if r["ar_hit"]), "en_hit": sum(1 for r in rows if r["en_hit"]), "with_expected_pages": sum(1 for r in rows if r["expected_pages"]),
           "behaviour_ok": sum(1 for r in rows if r["behaviour_ok"]),
           "ar_kinds": {k: sum(1 for r in rows if r["ar"]["kind"] == k) for k in ("answer", "refusal", "model_path", "notice")},
           "arabic_presentation": {k: sum(1 for r in rows if r["arabic_presentation"] == k) for k in ("fixed", "framed", "translated", "framed_after_check", "framed_after_error")},
           "framed_english_equals_en": sum(1 for r in rows if r["framed_english_equals_en_answer"]),
           "answers_total_llm_calls": sum(int(r["ar"]["llm_calls"] or 0) for r in rows),
           "mean_coverage": round(sum(float(r["coverage"] or 0) for r in rows) / n, 3) if n else None,
           "mean_ar_ms": round(sum(r["ar"]["ms"] for r in rows) / n, 1) if n else None, "mean_en_ms": round(sum(r["en"]["ms"] for r in rows) / n, 1) if n else None,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(os.path.join(OUT, "eval_%s.json" % SPLIT), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "eval_%s.md" % SPLIT), "w", encoding="utf-8") as f:
    f.write("# Offline multilingual evaluation (%s split, %d paired questions, %s mode, no model request)\n\n" % (SPLIT, n, summary["mode"]))
    f.write("| Measure | Value |\n|---|---|\n")
    for k in ("ar_served_plan_f", "en_served_plan_f", "served_agree", "pages_agree", "specialist_agree", "ar_hit", "en_hit", "with_expected_pages", "behaviour_ok", "framed_english_equals_en", "answers_total_llm_calls", "mean_coverage", "mean_ar_ms", "mean_en_ms"):
        f.write("| %s | %s |\n" % (k, summary[k]))
    f.write("| ar_kinds | %s |\n| arabic_presentation | %s |\n\n" % (json.dumps(summary["ar_kinds"]), json.dumps(summary["arabic_presentation"])))
    f.write("| id | expect | AR kind | EN kind | AR served | EN served | pages = | AR hit | EN hit | behaviour | English query |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (r["id"], r["expect"], r["ar"]["kind"], r["en"]["kind"], r["ar"]["served"], r["en"]["served"],
                                                                     "yes" if r["pages_agree"] else "no", r["ar_hit"], r["en_hit"], "ok" if r["behaviour_ok"] else "CHECK", (r["english_query"] or "").replace("|", "/")))
print(json.dumps(summary, indent=1, ensure_ascii=False))
