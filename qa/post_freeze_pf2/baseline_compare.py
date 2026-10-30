"""Compare the frozen I-9 master run (192 questions) with the rerun on the multi-source corpus and write
``baseline_comparison.json``: served and fallback counts, every per-question field the I-8 comparison checks (served path,
Plan F outcome, routing reason, handoff outcome, specialist calls, path, findings reason, answer text, pages), SHADOW
invariance, the frozen metrics beside the rerun metrics, and the latency of both runs.

    python -X utf8 qa/post_freeze_pf2/baseline_compare.py <frozen i8_results.json> <rerun i8_results.json> <rerun metrics.json> <out json>
"""
import json
import statistics
import sys

FROZEN = json.load(open(sys.argv[1], encoding="utf-8"))
RERUN = json.load(open(sys.argv[2], encoding="utf-8"))
METRICS = json.load(open(sys.argv[3], encoding="utf-8")) if sys.argv[3] != "-" else {}
OUT = sys.argv[4]
FIELDS = ("served", "outcome", "routing_reason", "handoff_outcome", "specialist_calls", "plan_f_path", "findings_reason")


def view(rec):
    d = rec.get("diagnostics") or {}
    out = {f: d.get(f) for f in FIELDS}
    out["answer"] = rec.get("answer")
    out["pages"] = list(rec.get("pages") or [])
    out["exception"] = rec.get("exception")
    return out


def quantiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return {"n": len(values), "p50_ms": round(statistics.median(values), 1), "p95_ms": round(values[min(len(values) - 1, int(0.95 * len(values)))], 1),
            "max_ms": round(values[-1], 1), "mean_ms": round(statistics.fmean(values), 1)}


modes = {}
for mode in ("active", "shadow"):
    a = {r["id"]: r for r in FROZEN["results"] if r["mode"] == mode}
    b = {r["id"]: r for r in RERUN["results"] if r["mode"] == mode}
    changes = []
    for qid in a:
        if qid not in b:
            changes.append({"id": qid, "fields": ["missing in the rerun"]})
            continue
        va, vb = view(a[qid]), view(b[qid])
        diff = [k for k in va if va[k] != vb[k]]
        if diff:
            changes.append({"id": qid, "question": a[qid]["question"], "fields": diff,
                            "frozen": {k: va[k] for k in diff}, "rerun": {k: vb[k] for k in diff}})
    served_a = sorted(q for q, r in a.items() if (r.get("diagnostics") or {}).get("served") == "plan_f")
    served_b = sorted(q for q, r in b.items() if (r.get("diagnostics") or {}).get("served") == "plan_f")
    entry = {"questions_common": len(set(a) & set(b)), "served_frozen": len(served_a), "served_rerun": len(served_b),
             "served_ids_equal": served_a == served_b, "questions_changed": len(changes), "changes": changes,
             "latency_frozen": quantiles([r["wall_ms"] for r in a.values()]), "latency_rerun": quantiles([r["wall_ms"] for r in b.values()]),
             "plan_f_ms_frozen": quantiles([(r.get("diagnostics") or {}).get("plan_f_ms") for r in a.values()]),
             "plan_f_ms_rerun": quantiles([(r.get("diagnostics") or {}).get("plan_f_ms") for r in b.values()])}
    if mode == "shadow":
        entry["identical_to_direct_legacy_frozen"] = sum(1 for r in a.values() if r.get("identical_to_direct") is True)
        entry["identical_to_direct_legacy_rerun"] = sum(1 for r in b.values() if r.get("identical_to_direct") is True)
    modes[mode] = entry

# determinism repeats and the legacy sample of the rerun
rep_diff = sorted({r["id"] for r in RERUN.get("repeats", []) if view(r) != view(next(x for x in RERUN["results"] if x["mode"] == "active" and x["id"] == r["id"]))})
legacy = RERUN.get("legacy", [])
summary = {
    "frozen_run": {"file": "qa/plan_f_i8/results_2026-10-30.json", "kb": FROZEN.get("kb"), "total_script_s": FROZEN.get("total_script_s")},
    "rerun": {"kb": RERUN.get("kb"), "total_script_s": RERUN.get("total_script_s"), "cold": RERUN.get("cold")},
    "frozen_metrics_i9": {"questions": 192, "served": 55, "fallback": 137, "coverage_pct": 28.6, "correct": 55, "limited": 0, "wrong": 0, "precision_pct": 100.0,
                          "false_acceptances": 0, "false_fallbacks": 11, "shadow_mismatches": 0, "plan_f_llm_calls": 0, "tests": 6674},
    "rerun_metrics": METRICS,
    "modes": modes,
    "determinism_repeat_differences": rep_diff,
    "legacy_sample": {"questions": len(legacy), "minimal_diagnostics": sum(1 for r in legacy if r.get("minimal_diagnostics")),
                      "cache_untouched": RERUN.get("legacy_cache_untouched")},
    "verdict": "IDENTICAL" if all(m["questions_changed"] == 0 for m in modes.values()) and not rep_diff else "CHANGED",
}
json.dump(summary, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
print(json.dumps({k: (v if k != "rerun_metrics" else "...") for k, v in summary.items() if k != "modes"}, indent=1, default=str))
for mode, m in modes.items():
    print(mode, {k: v for k, v in m.items() if k != "changes"})
