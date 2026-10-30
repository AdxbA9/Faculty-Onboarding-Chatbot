"""Metrics of the PF-2 guide evaluation: what the frozen Plan F runtime served on guide questions, what the unscoped legacy
retrieval cites (source and page precision), the conflict and wrong-source controls, the baseline questions against the
I-9 run, SHADOW invariance, the scope-extension seam pass, determinism and latency. Writes ``metrics.json``.

    python -X utf8 qa/post_freeze_pf2/analyze_guide_eval.py <results.json> <review.json> <metrics.json> [cold build json] [rebuild json] [cached startup json]
"""
import json
import statistics
import sys
from collections import Counter

R = json.load(open(sys.argv[1], encoding="utf-8"))
REVIEW = json.load(open(sys.argv[2], encoding="utf-8")) if len(sys.argv) > 2 and sys.argv[2] != "-" else {}
OUT = sys.argv[3]
EXTRA = {}
for key, pos in (("cold_build", 4), ("cache_rebuild", 5), ("cached_startup", 6)):
    if len(sys.argv) > pos and sys.argv[pos] != "-":
        EXTRA[key] = json.load(open(sys.argv[pos], encoding="utf-8"))

QUESTIONS = {q["id"]: q for q in R["questions"]}
by_mode = {}
for rec in R["results"]:
    by_mode.setdefault(rec["mode"], {})[rec["id"]] = rec
ACTIVE, SHADOW, SEAM, LEGACY = (by_mode.get(m, {}) for m in ("active", "shadow", "active_seam", "legacy"))
GUIDE_IDS = [qid for qid, q in QUESTIONS.items() if q["category"] in ("guide", "cross_document")]
CONTROL_IDS = [qid for qid, q in QUESTIONS.items() if q["category"] in ("wrong_source", "ambiguous")]
BASELINE_IDS = [qid for qid, q in QUESTIONS.items() if q.get("baseline_question")]


def served(rec):
    return ((rec or {}).get("diagnostics") or {}).get("served")


def outcome(rec):
    return ((rec or {}).get("diagnostics") or {}).get("outcome")


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def quantiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return {"n": len(values), "p50_ms": round(statistics.median(values), 1), "p95_ms": round(values[min(len(values) - 1, int(0.95 * len(values)))], 1),
            "max_ms": round(values[-1], 1), "mean_ms": round(statistics.fmean(values), 1)}


# ---- what Plan F did on the guide questions ---------------------------------------------------------------------------
guide_rows = []
for qid in GUIDE_IDS:
    a, q = ACTIVE.get(qid), QUESTIONS[qid]
    leg = LEGACY.get(qid) or {}
    items = leg.get("items") or []
    top1 = items[0]["source_id"] if items else None
    hit_source = any(i["source_id"] == q["source"] for i in items)
    hit_page = any(i["source_id"] == q["source"] and q.get("pages") and i["page"] in q["pages"] for i in items)
    secondary = q.get("secondary_source")
    hit_secondary = any(i["source_id"] == secondary for i in items) if secondary else None
    pool = leg.get("candidate_pool_by_source") or {}
    guide_rows.append({
        "id": qid, "category": q["category"], "expected_source": q["source"], "expected_pages": q.get("pages"), "conflict": q.get("conflict", False),
        "active_served": served(a), "active_outcome": outcome(a), "active_pages": a.get("pages") if a else None,
        "routing_final": ((a or {}).get("diagnostics") or {}).get("routing_final"), "specialist_calls": ((a or {}).get("diagnostics") or {}).get("specialist_calls"),
        "handoff_outcome": ((a or {}).get("diagnostics") or {}).get("handoff_outcome"), "plan_f_sources": ((a or {}).get("diagnostics") or {}).get("plan_f_sources"),
        "legacy_top1_source": top1, "legacy_top1_page": items[0]["page"] if items else None,
        "legacy_top5_sources": [(i["source_id"], i["page"]) for i in items], "legacy_hits_expected_source": hit_source,
        "legacy_hits_expected_page": hit_page, "legacy_hits_secondary_source": hit_secondary,
        "legacy_conflict_items": sum(1 for i in items if i.get("source_conflict")),
        "candidate_pool": pool, "candidate_pool_guide_share": pct(sum(v for k, v in pool.items() if k != "uos_faculty_handbook_2025_26"), sum(pool.values())) if pool else None,
        "review": REVIEW.get(qid),
    })

n_guide = len(GUIDE_IDS)
plan_f_served_guide = [r for r in guide_rows if r["active_served"] == "plan_f"]
guide_metrics = {
    "questions": n_guide,
    "plan_f_served": len(plan_f_served_guide),
    "plan_f_served_ids": [r["id"] for r in plan_f_served_guide],
    "plan_f_served_with_guide_evidence": sum(1 for r in plan_f_served_guide if any(s != "uos_faculty_handbook_2025_26" for s in (r["plan_f_sources"] or []))),
    "fallback": sum(1 for r in guide_rows if r["active_served"] == "legacy"),
    "exceptions": sum(1 for qid in GUIDE_IDS if (ACTIVE.get(qid) or {}).get("exception")),
    "outcomes": dict(Counter(r["active_outcome"] for r in guide_rows)),
    "routing_final": dict(Counter("+".join(r["routing_final"] or []) or "none" for r in guide_rows)),
    "review": dict(Counter((REVIEW.get(r["id"]) or {}).get("verdict", "not_reviewed") for r in plan_f_served_guide)),
    "legacy_retrieval": {
        "top1_expected_source": sum(1 for r in guide_rows if r["legacy_top1_source"] == r["expected_source"]),
        "top1_expected_source_pct": pct(sum(1 for r in guide_rows if r["legacy_top1_source"] == r["expected_source"]), n_guide),
        "top5_expected_source": sum(1 for r in guide_rows if r["legacy_hits_expected_source"]),
        "top5_expected_source_pct": pct(sum(1 for r in guide_rows if r["legacy_hits_expected_source"]), n_guide),
        "top5_expected_page": sum(1 for r in guide_rows if r["legacy_hits_expected_page"]),
        "top5_expected_page_pct": pct(sum(1 for r in guide_rows if r["legacy_hits_expected_page"]), n_guide),
        "top1_handbook_instead_of_guide": sum(1 for r in guide_rows if r["legacy_top1_source"] == "uos_faculty_handbook_2025_26"),
        "top1_other_guide": sum(1 for r in guide_rows if r["legacy_top1_source"] not in (None, r["expected_source"], "uos_faculty_handbook_2025_26")),
        "cross_document_both_sources_in_top5": sum(1 for r in guide_rows if r["category"] == "cross_document" and r["legacy_hits_expected_source"] and r["legacy_hits_secondary_source"]),
        "cross_document_questions": sum(1 for r in guide_rows if r["category"] == "cross_document"),
        "mean_candidate_pool_guide_share_pct": round(statistics.fmean([r["candidate_pool_guide_share"] for r in guide_rows if r["candidate_pool_guide_share"] is not None]), 1) if guide_rows else None,
    },
    "per_source": {},
}
for src in sorted({r["expected_source"] for r in guide_rows if r["expected_source"]}):
    rows = [r for r in guide_rows if r["expected_source"] == src]
    guide_metrics["per_source"][src] = {"questions": len(rows), "plan_f_served": sum(1 for r in rows if r["active_served"] == "plan_f"),
                                        "legacy_top1_source": sum(1 for r in rows if r["legacy_top1_source"] == src),
                                        "legacy_top5_page": sum(1 for r in rows if r["legacy_hits_expected_page"])}

# ---- the conflict question --------------------------------------------------------------------------------------------
conflict = {}
for qid, q in QUESTIONS.items():
    if q.get("conflict"):
        a, s, sm, leg = ACTIVE.get(qid), SHADOW.get(qid), SEAM.get(qid), LEGACY.get(qid)
        conflict[qid] = {"active_served": served(a), "active_outcome": outcome(a), "seam_served": served(sm), "seam_outcome": outcome(sm),
                         "shadow_outcome": outcome(s), "plan_f_served_the_limit": served(a) == "plan_f",
                         "legacy_items_under_conflict": sum(1 for i in (leg or {}).get("items") or [] if i.get("source_conflict")),
                         "legacy_answer_is_model_reply": bool((leg or {}).get("answer")) and "GROQ_API_KEY" in ((leg or {}).get("answer") or "")}

# ---- controls: wrong source, ambiguous, held document ----------------------------------------------------------------
controls = {}
for qid in CONTROL_IDS:
    a, leg, q = ACTIVE.get(qid), LEGACY.get(qid) or {}, QUESTIONS[qid]
    items = leg.get("items") or []
    controls[qid] = {"category": q["category"], "active_served": served(a), "active_outcome": outcome(a), "active_pages": (a or {}).get("pages"),
                     "legacy_top5_sources": [(i["source_id"], i["page"]) for i in items],
                     "appraisal_content_present": any("appraisal" in (i["source_id"] or "") for i in items)}
controls_summary = {"questions": len(CONTROL_IDS), "plan_f_served": sum(1 for c in controls.values() if c["active_served"] == "plan_f"),
                    "appraisal_content_present": sum(1 for c in controls.values() if c["appraisal_content_present"])}

# ---- baseline questions against the I-9 run ----------------------------------------------------------------------------
baseline = {}
for qid in BASELINE_IDS:
    a = ACTIVE.get(qid) or {}
    b = a.get("baseline")
    baseline[qid] = {"found": b is not None, **({k: v for k, v in b.items()} if b else {}), "active_served": served(a), "active_outcome": outcome(a),
                     "served_expected": QUESTIONS[qid]["served_expected"], "served_as_expected": (served(a) == "plan_f") == QUESTIONS[qid]["served_expected"]}
baseline_summary = {"questions": len(BASELINE_IDS), "found_in_i9_run": sum(1 for b in baseline.values() if b["found"]),
                    "unchanged": sum(1 for b in baseline.values() if b["found"] and b.get("answer_equal") and b.get("pages_equal") and b.get("served_equal") and b.get("outcome_equal")),
                    "served_as_expected": sum(1 for b in baseline.values() if b["served_as_expected"])}

# ---- SHADOW invariance and the seam pass -------------------------------------------------------------------------------
shadow = {"questions": len(SHADOW), "identical_to_direct_legacy": sum(1 for r in SHADOW.values() if r.get("identical_to_direct") is True),
          "legacy_served": sum(1 for r in SHADOW.values() if served(r) == "legacy"),
          "same_plan_f_outcome_as_active": sum(1 for qid, r in SHADOW.items() if outcome(r) == outcome(ACTIVE.get(qid)))}
seam_diffs = []
for qid, r in SEAM.items():
    a = ACTIVE.get(qid) or {}
    diff = [k for k in ("served", "outcome") if ((r.get("diagnostics") or {}).get(k)) != ((a.get("diagnostics") or {}).get(k))]
    if r.get("answer") != a.get("answer"):
        diff.append("answer")
    if list(r.get("pages") or []) != list(a.get("pages") or []):
        diff.append("pages")
    if diff:
        seam_diffs.append({"id": qid, "fields": diff, "active": (served(a), outcome(a), a.get("pages")), "seam": (served(r), outcome(r), r.get("pages"))})
seam = {"questions": len(SEAM), "registry": R.get("cold", {}).get("active_seam"), "plan_f_served": sum(1 for r in SEAM.values() if served(r) == "plan_f"),
        "questions_changed_by_the_seam": len(seam_diffs), "changes": seam_diffs}

# ---- determinism ------------------------------------------------------------------------------------------------------
rep_diff = []
for rec in R.get("repeats", []):
    a = ACTIVE.get(rec["id"]) or {}
    same = (rec.get("answer") == a.get("answer") and list(rec.get("pages") or []) == list(a.get("pages") or []) and served(rec) == served(a) and outcome(rec) == outcome(a))
    if not same:
        rep_diff.append(rec["id"])
determinism = {"repeat_records": len(R.get("repeats", [])), "differences": sorted(set(rep_diff))}

# ---- latency ----------------------------------------------------------------------------------------------------------
latency = {}
for mode, table in (("active", ACTIVE), ("shadow", SHADOW), ("active_seam", SEAM), ("legacy", LEGACY)):
    latency[mode] = {"all": quantiles([r["wall_ms"] for r in table.values()]),
                     "guide_questions": quantiles([r["wall_ms"] for qid, r in table.items() if qid in GUIDE_IDS]),
                     "plan_f_ms": quantiles([(r.get("diagnostics") or {}).get("plan_f_ms") for r in table.values()])}
latency["cold"] = R.get("cold")

metrics = {
    "kb": R.get("kb"), "questions": len(QUESTIONS), "categories": dict(Counter(q["category"] for q in QUESTIONS.values())),
    "guide": guide_metrics, "guide_rows": guide_rows, "conflict": conflict, "controls": controls, "controls_summary": controls_summary,
    "baseline": baseline, "baseline_summary": baseline_summary, "shadow": shadow, "seam": seam, "determinism": determinism,
    "latency": latency, "legacy_cache_untouched": R.get("legacy_cache_untouched"),
    "legacy_diagnostics_minimal": all(r.get("minimal_diagnostics") for r in LEGACY.values()) if LEGACY else None,
    "exceptions_total": sum(1 for r in R["results"] if r.get("exception")), "performance_inputs": EXTRA, "total_script_s": R.get("total_script_s"),
}
json.dump(metrics, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({"guide": {k: v for k, v in guide_metrics.items() if k not in ("per_source",)}, "conflict": conflict, "controls": controls_summary,
                  "baseline": baseline_summary, "shadow": shadow, "seam": {k: v for k, v in seam.items() if k != "changes"}, "seam_changes": seam_diffs[:10],
                  "determinism": determinism, "latency_active": latency["active"], "exceptions": metrics["exceptions_total"]}, indent=1))
