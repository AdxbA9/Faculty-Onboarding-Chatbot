"""Metrics of the PF-3B evaluation: guide coverage and precision (with the manual review), per source, per specialist,
the layer that stopped each guide question, conflict variants, contamination both ways, cross-document cases, wrong-source
collisions, SHADOW invariance, the guide-scopes-off control, determinism, latency, model calls. Writes ``metrics.json``.

    python -X utf8 qa/post_freeze_pf3b/analyze.py <results.json> <review.json or -> <metrics.json> [extra json: perf] ...
"""
import json
import statistics
import sys
from collections import Counter, defaultdict

R = json.load(open(sys.argv[1], encoding="utf-8"))
REVIEW = (json.load(open(sys.argv[2], encoding="utf-8")).get("questions") or {}) if sys.argv[2] != "-" else {}
OUT = sys.argv[3]
EXTRA = [json.load(open(p, encoding="utf-8")) for p in sys.argv[4:]]
HANDBOOK = "uos_faculty_handbook_2025_26"

Q = {q["id"]: q for q in R["questions"]}
by_mode = defaultdict(dict)
for rec in R["results"]:
    by_mode[rec["mode"]][rec["id"]] = rec
ACTIVE, SHADOW, OFF, LEGACY = by_mode["active"], by_mode["shadow"], by_mode["active_off"], by_mode["legacy"]
DIRECT = {r["id"]: r for r in R["direct"]}
GUIDE = [qid for qid, q in Q.items() if q["category"] in ("guide", "cross_document") and qid.startswith("G")]


def diag(rec):
    return (rec or {}).get("diagnostics") or {}


def served(rec):
    return diag(rec).get("served") == "plan_f"


def guide_served(rec):
    return bool(diag(rec).get("plan_f_guide_served"))


def pct(a, b):
    return round(100.0 * a / b, 1) if b else None


def quantiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    return {"n": len(values), "p50_ms": round(statistics.median(values), 1), "p95_ms": round(values[min(len(values) - 1, int(0.95 * len(values)))], 1),
            "max_ms": round(values[-1], 1)}


def stop_layer(d):
    """Which layer stopped the question (from the direct pass)."""
    outcome = d.get("outcome") or ""
    if outcome == "PLAN_F_ACCEPTED":
        return "served"
    if outcome in ("PLAN_F_NOT_APPLICABLE",):
        return "no specialist selected (routing)"
    if outcome == "PLAN_F_REJECTED_BY_ROUTING":
        return "routing ambiguity"
    if outcome == "PLAN_F_REJECTED_BY_SOURCE_CONFLICT":
        return "source conflict guard"
    if outcome == "PLAN_F_REJECTED_BY_SOURCE_AUTHORITY":
        return "source authority guard (system, policy or procedure)"
    if outcome == "PLAN_F_REJECTED_BY_SUBJECT_GUARD":
        return "I-1 subject guard (entity not in the quote)"
    if outcome in ("PLAN_F_REJECTED_BY_FINDINGS_VERIFIER", "PLAN_F_REJECTED_BY_VERIFIER"):
        return "verification"
    reports = d.get("reports") or []
    if outcome == "PLAN_F_REJECTED_BY_AGGREGATOR":
        return "aggregator (a clause unsupported)"
    if any(r.get("procedural") or r.get("process") for r in reports):
        return "specialist procedure rule (a procedure request is never supported)"
    if any(r.get("ask_shape") == "specific_unknown" for r in reports):
        return "specialist ask gate (unrecognised ask)"
    if any(any("asked specifics stated with the answer" in str(u) for u in (r.get("uncovered") or [])) for r in reports):
        return "specialist ask gate (asked specifics, such as the system name or actor, not stated in the answering sentence)"
    if any(r.get("outcome") == "not_found" for r in reports):
        return "specialist found no quotable guide evidence"
    if any(r.get("uncovered") for r in reports):
        return "specialist completeness (a requested detail not stated)"
    if any(r.get("state") for r in reports):
        return "specialist did not run"
    return "other (%s)" % outcome


rows = []
for qid in Q:
    a, d, q = ACTIVE.get(qid) or {}, DIRECT.get(qid) or {}, Q[qid]
    guide_findings = sorted({f[0] for r in (d.get("reports") or []) for f in (r.get("findings") or []) if f[0] != HANDBOOK})
    rows.append({"id": qid, "category": q["category"], "expected_source": q.get("source"), "conflict": q.get("conflict", False),
                 "served": served(a), "guide_served": guide_served(a), "outcome": diag(a).get("outcome"), "scope_selected": diag(a).get("guide_scope_selected"),
                 "sources_named": diag(a).get("guide_sources_named"), "routing_final": diag(a).get("routing_final"), "plan_f_sources": diag(a).get("plan_f_sources"),
                 "stop_layer": stop_layer(d), "guide_findings_from": guide_findings, "guards": d.get("guards"),
                 "review": (REVIEW.get(qid) or {}).get("verdict")})

guide_rows = [r for r in rows if r["id"] in GUIDE]
served_rows = [r for r in guide_rows if r["guide_served"]]
verdicts = Counter((REVIEW.get(r["id"]) or {}).get("verdict", "not_reviewed") for r in served_rows)
guide = {"questions": len(guide_rows), "plan_f_served": len(served_rows), "fallback": len(guide_rows) - len(served_rows),
         "coverage_pct": pct(len(served_rows), len(guide_rows)), "correct": verdicts.get("CORRECT", 0),
         "limited": verdicts.get("ACCEPTABLE BUT LIMITED", 0), "wrong": verdicts.get("WRONG", 0),
         "precision_pct": pct(verdicts.get("CORRECT", 0) + verdicts.get("ACCEPTABLE BUT LIMITED", 0), len(served_rows)) if served_rows else None,
         "pf2_coverage_pct": 0.0, "scope_selected": sum(1 for r in guide_rows if r["scope_selected"]),
         "guide_findings_returned": sum(1 for r in guide_rows if r["guide_findings_from"]),
         "stop_layers": dict(Counter(r["stop_layer"] for r in guide_rows))}

per_source = {}
for src in sorted({r["expected_source"] for r in guide_rows if r["expected_source"]}):
    rs = [r for r in guide_rows if r["expected_source"] == src]
    per_source[src] = {"questions": len(rs), "served": sum(r["guide_served"] for r in rs), "fallback": sum(not r["guide_served"] for r in rs),
                       "correct": sum(1 for r in rs if r["guide_served"] and r["review"] == "CORRECT"),
                       "limited": sum(1 for r in rs if r["guide_served"] and r["review"] == "ACCEPTABLE BUT LIMITED"),
                       "wrong": sum(1 for r in rs if r["guide_served"] and r["review"] == "WRONG"),
                       "guide_findings_returned": sum(1 for r in rs if r["guide_findings_from"]),
                       "stop_layers": dict(Counter(r["stop_layer"] for r in rs))}

per_specialist = defaultdict(lambda: Counter())
for r in guide_rows:
    for sid in (r["routing_final"] or ["none"]):
        c = per_specialist[sid]
        c["routed"] += 1
        c["served"] += int(r["guide_served"])
        c["fallback"] += int(not r["guide_served"])
        c["correct"] += int(r["guide_served"] and r["review"] == "CORRECT")
        c["wrong"] += int(r["guide_served"] and r["review"] == "WRONG")
        c["guide_findings_returned"] += int(bool(r["guide_findings_from"]))

groups = {}
for cat in ("conflict", "handbook_contamination", "guide_contamination", "cross_document", "wrong_source", "ambiguous", "handbook_only", "known_hr", "annual_leave"):
    rs = [r for r in rows if r["category"] == cat]
    groups[cat] = {"questions": len(rs), "served": sum(r["served"] for r in rs), "guide_served": sum(r["guide_served"] for r in rs),
                   "outcomes": dict(Counter(r["outcome"] for r in rs)), "stop_layers": dict(Counter(r["stop_layer"] for r in rs)),
                   "ids_served": [r["id"] for r in rs if r["served"]]}

shadow = {"questions": len(SHADOW), "identical_to_direct_legacy": sum(1 for r in SHADOW.values() if r.get("identical_to_direct") is True),
          "visible_mismatches": sum(1 for r in SHADOW.values() if r.get("identical_to_direct") is False),
          "legacy_served": sum(1 for r in SHADOW.values() if diag(r).get("served") == "legacy")}
control = []
for qid, r in ACTIVE.items():
    o = OFF.get(qid) or {}
    if (diag(r).get("served"), diag(r).get("outcome"), r.get("answer"), r.get("pages")) != (diag(o).get("served"), diag(o).get("outcome"), o.get("answer"), o.get("pages")):
        control.append({"id": qid, "on": (diag(r).get("served"), diag(r).get("outcome"), r.get("pages")), "off": (diag(o).get("served"), diag(o).get("outcome"), o.get("pages"))})
rep_diff = sorted({rec["id"] for rec in R.get("repeats", []) if (rec.get("answer"), rec.get("pages"), diag(rec).get("outcome")) !=
                   ((ACTIVE.get(rec["id"]) or {}).get("answer"), (ACTIVE.get(rec["id"]) or {}).get("pages"), diag(ACTIVE.get(rec["id"])).get("outcome"))})
latency = {mode: {"all": quantiles([r["wall_ms"] for r in table.values()]), "plan_f_ms": quantiles([diag(r).get("plan_f_ms") for r in table.values()]),
                  "guide_scope_selected": quantiles([r["wall_ms"] for r in table.values() if diag(r).get("guide_scope_selected")])}
           for mode, table in (("active", ACTIVE), ("shadow", SHADOW), ("active_off", OFF), ("legacy", LEGACY))}
llm = {"plan_f_llm_calls_max": max([diag(r).get("plan_f_llm_calls") or 0 for t in (ACTIVE, SHADOW, OFF) for r in t.values()] or [0]),
       "direct_llm_calls_max": max([d.get("llm_calls") or 0 for d in DIRECT.values()] or [0]),
       "direct_used_llm": sum(1 for d in DIRECT.values() if d.get("used_llm"))}

metrics = {"kb": R.get("kb"), "registry_report": R.get("registry_report"), "cold": R.get("cold"), "questions": len(Q),
           "categories": dict(Counter(q["category"] for q in Q.values())), "guide": guide, "per_source": per_source,
           "per_specialist": {k: dict(v) for k, v in per_specialist.items()}, "groups": groups, "shadow": shadow,
           "control_scopes_off_vs_on": {"changed": len(control), "changes": control}, "determinism": {"repeat_records": len(R.get("repeats", [])), "differences": rep_diff},
           "latency": latency, "llm": llm, "exceptions": sum(1 for r in R["results"] if r.get("exception")) + sum(1 for d in DIRECT.values() if d.get("exception")),
           "legacy_minimal": all(r.get("minimal_diagnostics") for r in LEGACY.values()), "legacy_cache_untouched": R.get("legacy_cache_untouched"),
           "rows": rows, "performance": EXTRA, "total_script_s": R.get("total_script_s")}
json.dump(metrics, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: metrics[k] for k in ("guide", "per_source", "per_specialist", "groups", "shadow", "control_scopes_off_vs_on", "determinism", "llm", "exceptions",
                                          "legacy_minimal")}, indent=1, default=str)[:9000])
