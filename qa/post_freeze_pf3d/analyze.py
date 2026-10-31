"""Metrics of the PF-3D evaluation (D-111 + D-112): guide coverage and precision on the 37 core guide questions (with the manual
review), the extended PF-3C guide questions, per source, per specialist and per answer form, the layer that stopped each
question, safety groups (conflict, contamination both ways, cross-document, wrong-source collisions, appraisal,
incomplete procedures), SHADOW invariance, the guide-scopes-off control (every difference must be a reviewed guide
answer), determinism, latency and model calls. Writes ``metrics.json``.

    python -X utf8 qa/post_freeze_pf3d/analyze.py <results.json> <newly_served_review.json> <metrics.json>
"""
import json
import statistics
import sys
from collections import Counter, defaultdict

R = json.load(open(sys.argv[1], encoding="utf-8"))
REVIEW = json.load(open(sys.argv[2], encoding="utf-8"))
OUT = sys.argv[3]
VERDICTS = REVIEW.get("served") or {}
HANDBOOK = "uos_faculty_handbook_2025_26"

Q = {q["id"]: q for q in R["questions"]}
by_mode = defaultdict(dict)
for rec in R["results"]:
    by_mode[rec["mode"]][rec["id"]] = rec
ACTIVE, SHADOW, OFF, LEGACY = by_mode["active"], by_mode["shadow"], by_mode["active_off"], by_mode["legacy"]
DIRECT = {r["id"]: r for r in R["direct"]}
CORE = [qid for qid, q in Q.items() if qid.startswith("G") and q["category"] in ("guide", "cross_document")]
EXTENDED = [qid for qid, q in Q.items() if qid.startswith("C") and q["category"] == "guide"]


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


def guide_path(d):
    for r in d.get("reports") or []:
        if r.get("guide_path"):
            return r["guide_path"]
    return None


def stop_layer(d):
    outcome = d.get("outcome") or ""
    gp = guide_path(d)
    applied = bool(gp and any(c.get("applies") for c in (gp.get("clauses") or {}).values()))
    if outcome == "PLAN_F_ACCEPTED":
        return "served"
    if outcome == "PLAN_F_NOT_APPLICABLE":
        return "no specialist selected (routing)"
    if outcome == "PLAN_F_REJECTED_BY_ROUTING":
        return "routing ambiguity"
    if outcome == "PLAN_F_REJECTED_BY_SOURCE_CONFLICT":
        return "source conflict guard"
    if outcome == "PLAN_F_REJECTED_BY_SOURCE_AUTHORITY":
        return "source authority guard"
    if outcome == "PLAN_F_REJECTED_BY_SUBJECT_GUARD":
        return "subject / D-111 guide guard"
    if outcome in ("PLAN_F_REJECTED_BY_FINDINGS_VERIFIER", "PLAN_F_REJECTED_BY_VERIFIER"):
        return "verification"
    if outcome == "PLAN_F_REJECTED_BY_AGGREGATOR":
        return "aggregator (a clause unsupported)"
    if outcome == "PLAN_F_REJECTED_BY_HANDOFF":
        return "handoff"
    if gp is None or not applied:
        refused = [r for c in ((gp or {}).get("clauses") or {}).values() for r in (c.get("refused") or [])]
        if refused:
            return "D-111 binding refused (%s)" % refused[0].get("reason", "")[:60]
        return "no approved guide applies (handbook path decided)"
    return "D-111 guide path found no provably complete, fully covered evidence"


rows = []
for qid, q in Q.items():
    a, d = ACTIVE.get(qid) or {}, DIRECT.get(qid) or {}
    gp = guide_path(d) or {}
    clauses = gp.get("clauses") or {}
    rows.append({"id": qid, "category": q["category"], "expected_source": q.get("source"), "served": served(a), "guide_served": guide_served(a),
                 "outcome": diag(a).get("outcome"), "routing_final": diag(a).get("routing_final"), "plan_f_sources": diag(a).get("plan_f_sources"),
                 "forms": diag(a).get("guide_answer_forms") or [], "parts": diag(a).get("guide_answer_parts") or 0, "pages": a.get("pages"),
                 "stop_layer": stop_layer(d), "guide_applies": any(c.get("applies") for c in clauses.values()),
                 "verdict": (VERDICTS.get(qid) or {}).get("verdict")})
ROW = {r["id"]: r for r in rows}


def summary(ids):
    rs = [ROW[i] for i in ids]
    srv = [r for r in rs if r["guide_served"]]
    v = Counter(r["verdict"] or "not_reviewed" for r in srv)
    return {"questions": len(rs), "plan_f_served": len(srv), "fallback": len(rs) - len(srv), "coverage_pct": pct(len(srv), len(rs)),
            "correct": v.get("CORRECT", 0), "limited": v.get("ACCEPTABLE BUT LIMITED", 0), "wrong": v.get("WRONG", 0), "not_reviewed": v.get("not_reviewed", 0),
            "precision_pct": pct(v.get("CORRECT", 0) + v.get("ACCEPTABLE BUT LIMITED", 0), len(srv)) if srv else None,
            "strict_precision_pct": pct(v.get("CORRECT", 0), len(srv)) if srv else None,
            "forms": dict(Counter(f for r in srv for f in r["forms"])), "served_ids": [r["id"] for r in srv],
            "stop_layers": dict(Counter(r["stop_layer"] for r in rs))}


core = summary(CORE)
extended = summary(EXTENDED)
all_served = [r for r in rows if r["guide_served"]]
newly = {"served_guide_answers": len(all_served), "ids": [r["id"] for r in all_served],
         "verdicts": dict(Counter(r["verdict"] or "not_reviewed" for r in all_served))}

per_source = {}
for src in sorted({r["expected_source"] for r in rows if r["expected_source"] and r["expected_source"] != HANDBOOK}):
    rs = [r for r in rows if r["expected_source"] == src and r["id"] in CORE]
    per_source[src] = {"questions": len(rs), "served": sum(r["guide_served"] for r in rs),
                       "correct": sum(1 for r in rs if r["guide_served"] and r["verdict"] == "CORRECT"),
                       "limited": sum(1 for r in rs if r["guide_served"] and r["verdict"] == "ACCEPTABLE BUT LIMITED"),
                       "wrong": sum(1 for r in rs if r["guide_served"] and r["verdict"] == "WRONG"),
                       "served_ids": [r["id"] for r in rs if r["guide_served"]]}

per_specialist = defaultdict(Counter)
for r in rows:
    if r["id"] not in CORE:
        continue
    for sid in (r["routing_final"] or ["none"]):
        c = per_specialist[sid]
        c["routed"] += 1
        c["served"] += int(r["guide_served"])
        c["correct"] += int(r["guide_served"] and r["verdict"] == "CORRECT")
        c["wrong"] += int(r["guide_served"] and r["verdict"] == "WRONG")

groups = {}
for cat in sorted({q["category"] for q in Q.values()}):
    rs = [r for r in rows if r["category"] == cat]
    groups[cat] = {"questions": len(rs), "served": sum(r["served"] for r in rs), "guide_served": sum(r["guide_served"] for r in rs),
                   "outcomes": dict(Counter(r["outcome"] for r in rs)), "ids_served": [r["id"] for r in rs if r["served"]],
                   "ids_guide_served": [r["id"] for r in rs if r["guide_served"]]}

shadow = {"questions": len(SHADOW), "identical_to_direct_legacy": sum(1 for r in SHADOW.values() if r.get("identical_to_direct") is True),
          "visible_mismatches": sum(1 for r in SHADOW.values() if r.get("identical_to_direct") is False),
          "legacy_served": sum(1 for r in SHADOW.values() if diag(r).get("served") == "legacy"),
          "plan_f_accepted_internally": sum(1 for r in SHADOW.values() if diag(r).get("accepted"))}
control = []
for qid, r in ACTIVE.items():
    o = OFF.get(qid) or {}
    if (diag(r).get("served"), diag(r).get("outcome"), r.get("answer"), r.get("pages")) != (diag(o).get("served"), diag(o).get("outcome"), o.get("answer"), o.get("pages")):
        control.append({"id": qid, "on": (diag(r).get("served"), diag(r).get("outcome"), r.get("pages")), "off": (diag(o).get("served"), diag(o).get("outcome"), o.get("pages")),
                        "guide_served_on": guide_served(r), "reviewed": (VERDICTS.get(qid) or {}).get("verdict")})
unexplained = [c["id"] for c in control if not (c["guide_served_on"] and c["reviewed"] in ("CORRECT", "ACCEPTABLE BUT LIMITED"))]
rep_diff = sorted({rec["id"] for rec in R.get("repeats", []) if (rec.get("answer"), rec.get("pages"), diag(rec).get("outcome")) !=
                   ((ACTIVE.get(rec["id"]) or {}).get("answer"), (ACTIVE.get(rec["id"]) or {}).get("pages"), diag(ACTIVE.get(rec["id"])).get("outcome"))})


def guide_ms(d, key):
    vals = []
    for r in d.get("reports") or []:
        for c in ((r.get("guide_path") or {}).get("clauses") or {}).values():
            if isinstance(c.get(key), (int, float)):
                vals.append(c[key])
    return vals


latency = {mode: {"all": quantiles([r["wall_ms"] for r in table.values()]), "plan_f_ms": quantiles([diag(r).get("plan_f_ms") for r in table.values()]),
                  "guide_scope_selected": quantiles([r["wall_ms"] for r in table.values() if diag(r).get("guide_scope_selected")]),
                  "guide_served": quantiles([r["wall_ms"] for r in table.values() if guide_served(r)]),
                  "guide_fallback": quantiles([r["wall_ms"] for r in table.values() if diag(r).get("guide_scope_selected") and not guide_served(r)])}
           for mode, table in (("active", ACTIVE), ("shadow", SHADOW), ("active_off", OFF), ("legacy", LEGACY))}
latency["guide_path_ms_per_clause"] = quantiles([v for d in DIRECT.values() for v in guide_ms(d, "ms")])
latency["guide_retrieval_ms_per_clause"] = quantiles([v for d in DIRECT.values() for v in guide_ms(d, "retrieval_ms")])
latency["guide_selection_ms_per_clause"] = quantiles([v for d in DIRECT.values() for v in guide_ms(d, "selection_ms")])
latency["direct_verification_ms"] = {k: quantiles([(d.get("timings") or {}).get(k) for d in DIRECT.values() if guide_path(d)])
                                     for k in ("guards_ms", "findings_verify_ms", "verify_ms", "dispatch_ms", "total_ms")}
llm = {"plan_f_llm_calls_max": max([diag(r).get("plan_f_llm_calls") or 0 for t in (ACTIVE, SHADOW, OFF) for r in t.values()] or [0]),
       "direct_llm_calls_max": max([d.get("llm_calls") or 0 for d in DIRECT.values()] or [0]),
       "direct_used_llm": sum(1 for d in DIRECT.values() if d.get("used_llm")),
       "served_used_llm": sum(1 for r in ACTIVE.values() if served(r) and r.get("used_llm"))}

metrics = {"kb": R.get("kb"), "registry_report": R.get("registry_report"), "cold": R.get("cold"), "questions": len(Q),
           "categories": dict(Counter(q["category"] for q in Q.values())), "core_guide": core, "extended_guide": extended, "newly_served": newly,
           "per_source": per_source, "per_specialist": {k: dict(v) for k, v in per_specialist.items()}, "groups": groups, "shadow": shadow,
           "control_scopes_off_vs_on": {"changed": len(control), "unexplained": unexplained, "changes": control},
           "determinism": {"repeat_records": len(R.get("repeats", [])), "differences": rep_diff}, "latency": latency, "llm": llm,
           "exceptions": sum(1 for r in R["results"] if r.get("exception")) + sum(1 for d in DIRECT.values() if d.get("exception")),
           "legacy_minimal": all(r.get("minimal_diagnostics") for r in LEGACY.values()), "legacy_cache_untouched": R.get("legacy_cache_untouched"),
           "rows": rows, "total_script_s": R.get("total_script_s")}
json.dump(metrics, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: metrics[k] for k in ("core_guide", "extended_guide", "newly_served", "per_source", "per_specialist", "shadow",
                                          "control_scopes_off_vs_on", "determinism", "llm", "exceptions", "legacy_minimal")}, indent=1, default=str)[:12000])
