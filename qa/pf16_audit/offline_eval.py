"""PF-16 offline evaluation on the REAL eight-source knowledge base through the real application seam (``ui.pipeline``).

No model request is possible: the Groq key is removed from this process before anything is imported (only ``app.py`` reads
``.env``), so ``kb.groq_client`` is None. What a question gets offline is therefore exactly what the deterministic parts of
the system produce: a Plan F specialist answer, a legacy extractor answer (contacts, counts, dates), a controlled refusal
from the evidence gate, or the "not configured" notice where the legacy path would call the model (recorded as
``model_path`` with the retrieved evidence, so retrieval is still measured).

    python -X utf8 qa/pf16_audit/offline_eval.py <repo> <out dir> --set frozen192 [--scopes on|off]
    python -X utf8 qa/pf16_audit/offline_eval.py <repo> <out dir> --set cases --cases a.jsonl b.jsonl [--scopes on|off]
        [--mode active|shadow|legacy] [--tag NAME]

``--scopes on`` (default) is the configuration the owner's manual acceptance and every browser QA used (ACTIVE, guide scopes
on); ``off`` is the frozen production default of Plan F. Writes ``<tag>.json`` (every record) and ``<tag>.md`` (summary).

The automatic verdicts are deliberately conservative and are labelled as automatic: a served answer passes only when every
expected fact string occurs in it and its citation includes an expected (source, page); anything else that was served is
``REVIEW`` for a person to judge against the quoted evidence. A ``model_path`` record is ``NOT_EXECUTED`` for the answer
(retrieval is still judged), never a pass.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata

REPO, OUT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
ARGS = sys.argv[3:]


def _arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


SET = _arg("--set", "cases")
SCOPES = _arg("--scopes", "on")
MODE = _arg("--mode", "active")
TAG = _arg("--tag", "%s_%s_scopes_%s" % (SET, MODE, SCOPES))
CASE_FILES = []
if "--cases" in ARGS:
    for a in ARGS[ARGS.index("--cases") + 1:]:
        if a.startswith("--"):
            break
        CASE_FILES.append(a)

os.chdir(REPO)
sys.path.insert(0, REPO)
for name in ("GROQ_API_KEY", "PLAN_F_ENABLED", "PLAN_F_SHADOW", "PLAN_F_GUIDE_SCOPES"):
    os.environ.pop(name, None)
if MODE == "active":
    os.environ["PLAN_F_ENABLED"] = "1"
elif MODE == "shadow":
    os.environ["PLAN_F_SHADOW"] = "1"
if SCOPES == "on":
    os.environ["PLAN_F_GUIDE_SCOPES"] = "1"
os.makedirs(OUT, exist_ok=True)

from handbook_bot.arabic_answers import localize_reply              # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base        # noqa: E402
from handbook_bot.orchestrator import NO_KEY_REPLY                  # noqa: E402
from handbook_bot.qa import REFUSAL                                 # noqa: E402
from ui import pipeline                                             # noqa: E402

t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
assert kb.groq_client is None, "offline evaluation must not have a model client"
BUILD_S = round(time.perf_counter() - t0, 1)
print("mode=%s scopes=%s kb=%d chunks, %d pages (%.1f s)" % (pipeline.plan_f_mode(), SCOPES, len(kb.chunks), len(kb.pages), BUILD_S), flush=True)


def kind(answer: str) -> str:
    if answer == NO_KEY_REPLY:
        return "model_path"
    if answer == REFUSAL:
        return "refusal"
    if localize_reply(answer) is not None:
        return "notice"
    return "answer"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("’", "'").replace("­", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def item_pages(result):
    out = []
    for i in result.items or []:
        meta = i.get("meta") if isinstance(i, dict) else None
        if meta and meta.get("page") is not None:
            out.append([meta.get("source_id"), int(meta.get("page"))])
    return out


def cited_pairs(result, diag):
    """(source, page) the user sees cited: Plan F pages carry their sources in the diagnostics; the legacy path cites pages
    of its own items."""
    pairs = []
    srcs = (diag.get("plan_f_sources") if diag.get("served") == "plan_f" else diag.get("legacy_sources")) or []
    pages = result.pages or []
    items = item_pages(result)
    for p in pages:
        owners = sorted({s for s, q in items if q == p}) or (srcs[:1] if len(srcs) == 1 else [])
        for s in owners or [None]:
            pairs.append([s, p])
    return pairs


def run(question: str):
    t = time.perf_counter()
    try:
        result, diag = pipeline.ask_with_diagnostics(kb, question)
    except Exception as exc:                                        # an exception is a finding, never a crash of the run
        return {"exception": "%s: %s" % (type(exc).__name__, str(exc)[:200]), "kind": "exception", "ms": round((time.perf_counter() - t) * 1000, 1)}
    ms = (time.perf_counter() - t) * 1000.0
    base_answer = result.original_answer or result.answer
    return {"answer": result.answer, "original_answer": result.original_answer, "answer_language": result.answer_language,
            "kind": kind(base_answer), "served": diag.get("served"), "outcome": diag.get("outcome"),
            "fallback_reason": diag.get("fallback_reason") or diag.get("reason"), "specialist": diag.get("specialist_id"),
            "routing_initial": diag.get("routing_initial"), "routing_final": diag.get("routing_final"), "path": diag.get("plan_f_path"),
            "aggregation_specialists": diag.get("aggregation_specialists"), "handoff_target": diag.get("handoff_target"),
            "pages": result.pages, "item_pages": item_pages(result), "cited": cited_pairs(result, diag), "query_type": result.query_type,
            "best_section": result.best_section, "llm_calls": result.llm_calls, "translation": result.translation,
            "transform": result.query_transform, "language": diag.get("language"), "ms": round(ms, 1), "exception": None,
            "diagnostics": diag}


def facts_found(answer: str, facts):
    a = norm(answer)
    return [f for f in facts if norm(f) in a]


def verdict(case, rec):
    """Automatic, conservative verdict (see the module docstring)."""
    exp = case["expected_behavior"]
    k = rec["kind"]
    answer = rec.get("original_answer") or rec.get("answer") or ""
    forbidden = [f for f in case.get("forbidden_facts") or [] if norm(f) and norm(f) in norm(rec.get("answer") or "") + " " + norm(answer)]
    rec["forbidden_present"] = forbidden
    expected_pairs = [[e["source_id"], e["page"]] for e in case.get("evidence") or []]
    rec["retrieval_hit"] = any(p in rec.get("item_pages", []) for p in expected_pairs) if expected_pairs else None
    rec["retrieval_hit_top3"] = any(p in rec.get("item_pages", [])[:3] for p in expected_pairs) if expected_pairs else None
    rec["citation_hit"] = any(p in rec.get("cited", []) for p in expected_pairs) if expected_pairs else None
    exp_specs = sorted(case.get("expected_specialists") or [])
    got = rec.get("routing_final") or ([rec["specialist"]] if rec.get("specialist") else [])
    rec["specialist_match"] = (sorted(set(got)) == exp_specs) if exp_specs else (not got if k == "refusal" else None)
    rec["specialist_overlap"] = bool(set(got) & set(exp_specs)) if exp_specs else None
    if k == "exception":
        return "FAIL", "exception"
    if forbidden:
        return "FAIL", "forbidden fact present: %s" % forbidden
    if exp == "refuse":
        if k == "refusal":
            return "PASS", "refused"
        if k == "model_path":
            return "NOT_EXECUTED", "would reach the model (judge in a live run)"
        if k == "notice":
            return "PASS", "service notice, no claim"
        return "REVIEW", "served an answer where a refusal was expected"
    facts = case.get("expected_facts") or []
    found = facts_found(answer, facts)
    rec["facts_found"] = found
    if k == "answer":
        if len(found) == len(facts) and rec["citation_hit"]:
            return "PASS", "all expected facts, expected page cited"
        if found and rec["citation_hit"]:
            return "REVIEW", "partial facts (%d of %d), expected page cited" % (len(found), len(facts))
        return "REVIEW", "facts %d of %d, expected page cited: %s" % (len(found), len(facts), rec["citation_hit"])
    if k == "refusal":
        return "FALSE_REFUSAL", "refused although the evidence exists (retrieval hit: %s)" % rec["retrieval_hit"]
    if k == "model_path":
        return "NOT_EXECUTED", "model path (retrieval hit: %s)" % rec["retrieval_hit"]
    return "REVIEW", "notice"


records = []
if SET == "frozen192":
    sys.path.insert(0, os.path.join(REPO, "qa", "plan_f_i8"))
    from question_set import QUESTIONS                                  # noqa: E402
    with open(os.path.join(REPO, "qa", "plan_f_i8", "results_2026-10-30.json"), encoding="utf-8") as f:
        i9 = {r["id"]: r for r in json.load(f)["results"] if r["mode"] == "active"}
    with open(os.path.join(REPO, "qa", "plan_f_i8", "review.json"), encoding="utf-8") as f:
        review = json.load(f)
    for q in QUESTIONS:
        rec = run(q["q"])
        ref = i9.get(q["id"]) or {}
        rec.update({"id": q["id"], "question": q["q"], "domain": q["domain"], "category": q["category"], "expect": q["expect"],
                    "i9_served": (ref.get("diagnostics") or {}).get("served"), "i9_answer_equal": rec.get("answer") == ref.get("answer"),
                    "i9_pages_equal": rec.get("pages") == ref.get("pages"), "review": (review.get(q["id"]) or {}).get("verdict")})
        records.append(rec)
        print("%-5s %-7s %-10s i9=%-7s eq=%s rev=%s" % (q["id"], rec.get("served"), rec["kind"], rec["i9_served"], rec["i9_answer_equal"], rec["review"]), flush=True)
    served = [r for r in records if r.get("served") == "plan_f"]
    summary = {"set": "frozen192", "mode": pipeline.plan_f_mode(), "scopes": SCOPES, "questions": len(records), "served_plan_f": len(served),
               "served_ids": [r["id"] for r in served],
               "served_reviewed_correct": sum(1 for r in served if r["review"] == "CORRECT"),
               "served_reviewed_other": [(r["id"], r["review"]) for r in served if r["review"] != "CORRECT"],
               "served_answer_equal_to_i9": sum(1 for r in served if r["i9_answer_equal"]),
               "served_but_not_served_in_i9": [r["id"] for r in served if r["i9_served"] != "plan_f"],
               "i9_served_not_served_now": sorted(k for k, v in i9.items() if (v.get("diagnostics") or {}).get("served") == "plan_f" and k not in {r["id"] for r in served}),
               "kinds": {k: sum(1 for r in records if r["kind"] == k) for k in ("answer", "refusal", "model_path", "notice", "exception")},
               "exceptions": sum(1 for r in records if r["kind"] == "exception"),
               "model_calls": sum(int(r.get("llm_calls") or 0) for r in records)}
else:
    cases = []
    for path in CASE_FILES:
        with open(path, encoding="utf-8") as f:
            cases += [json.loads(line) for line in f if line.strip()]
    for case in cases:
        rec = run(case["question"])
        rec.update({k: case.get(k) for k in ("id", "question", "language", "domain", "category", "expected_behavior", "expected_specialists", "pair")})
        v, why = verdict(case, rec)
        rec["verdict"], rec["why"] = v, why
        records.append(rec)
        print("%-14s %-6s %-11s %-8s spec=%-28s hit=%-5s %s" % (rec["id"], rec.get("language"), rec["kind"], rec.get("served"),
              ",".join(rec.get("routing_final") or []) or "-", rec.get("retrieval_hit"), v), flush=True)
    vc = {}
    for r in records:
        vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
    summary = {"set": "cases", "files": [os.path.basename(p) for p in CASE_FILES], "mode": pipeline.plan_f_mode(), "scopes": SCOPES,
               "questions": len(records), "verdicts": vc,
               "kinds": {k: sum(1 for r in records if r["kind"] == k) for k in ("answer", "refusal", "model_path", "notice", "exception")},
               "served_plan_f": sum(1 for r in records if r.get("served") == "plan_f"),
               "retrieval_hit": [sum(1 for r in records if r.get("retrieval_hit")), sum(1 for r in records if r.get("retrieval_hit") is not None)],
               "retrieval_hit_top3": [sum(1 for r in records if r.get("retrieval_hit_top3")), sum(1 for r in records if r.get("retrieval_hit_top3") is not None)],
               "model_calls": sum(int(r.get("llm_calls") or 0) for r in records),
               "exceptions": sum(1 for r in records if r["kind"] == "exception")}
summary.update({"kb_chunks": len(kb.chunks), "kb_pages": len(kb.pages), "kb_build_s": BUILD_S, "corpus_key": getattr(kb, "corpus_key", ""),
                "total_s": round(time.perf_counter() - t0, 1), "mean_ms": round(sum(r.get("ms", 0) for r in records) / max(1, len(records)), 1)})
with open(os.path.join(OUT, TAG + ".json"), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps(summary, ensure_ascii=False, default=str)[:3000])
