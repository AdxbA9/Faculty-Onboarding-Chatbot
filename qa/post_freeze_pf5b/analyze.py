"""Metrics of the PF-5B evaluation (D-111 + D-112 + D-113 + D-114), four question sets reported separately (never one headline):

* P0 regression: the PF-3D guide evaluation (37 core guide questions G01-G37 and the 100-question set);
* P1: the source-specific questions of the three activated P1 guides (plus negatives, conflicts, collisions, cross-document);
* HELD-OUT (PF-4, diagnostic only since PF-5B): the 50 frozen PF-4 questions, before (PF-4 results) and after;
* FRESH (PF-5B): the 56 questions frozen before any PF-5B rule was written, with coverage over all questions and over the
  questions an enabled source answers.

Every Plan F-served answer (guide or handbook) of the P1, held-out and fresh sets, and every P0 answer that differs from
PF-3D, must carry a hand verdict in the review file (CORRECT / ACCEPTABLE BUT LIMITED / WRONG); a served answer without one is
counted "not_reviewed" and fails the evaluation. Writes ``metrics.json``.

    python -X utf8 qa/post_freeze_pf5b/analyze.py <results.json> <review.json> <metrics.json> [<PF-4 results.json>]
"""
import json
import statistics
import sys
from collections import Counter, defaultdict

R = json.load(open(sys.argv[1], encoding="utf-8"))
REVIEW = json.load(open(sys.argv[2], encoding="utf-8"))
OUT = sys.argv[3]
VERDICTS = REVIEW.get("served") or {}
Q = {q["id"]: q for q in R["questions"]}
by_mode = defaultdict(dict)
for rec in R["results"]:
    by_mode[rec["mode"]][rec["id"]] = rec
ACTIVE, SHADOW, OFF, LEGACY = by_mode["active"], by_mode["shadow"], by_mode["active_off"], by_mode["legacy"]
DIRECT = {r["id"]: r for r in R["direct"]}
SETS = {name: [qid for qid, q in Q.items() if q.get("set") == name] for name in ("p0", "p1", "held_out", "fresh")}
BEFORE = {}
if len(sys.argv) > 4:                                              # the PF-4 results: the old held-out set before PF-5B
    BEFORE = {r["id"]: r for r in json.load(open(sys.argv[4], encoding="utf-8"))["results"] if r["mode"] == "active"}
CORE = [qid for qid in SETS["p0"] if qid.startswith("G") and Q[qid]["category"] in ("guide", "cross_document")]


def diag(rec):
    return (rec or {}).get("diagnostics") or {}


def served(rec):
    return diag(rec).get("served") == "plan_f"


def guide_served(rec):
    return bool(diag(rec).get("plan_f_guide_served"))


def verdict(qid):
    return (VERDICTS.get(qid) or {}).get("verdict")


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


def stop_layer(qid):
    d = DIRECT.get(qid) or {}
    outcome = d.get("outcome") or diag(ACTIVE.get(qid)).get("outcome") or ""
    gp = guide_path(d)
    applied = bool(gp and any(c.get("applies") for c in (gp.get("clauses") or {}).values()))
    named = diag(ACTIVE.get(qid)).get("guide_sources_named") or []
    table = {"PLAN_F_ACCEPTED": "served", "PLAN_F_NOT_APPLICABLE": "routing (no specialist selected)", "PLAN_F_REJECTED_BY_ROUTING": "routing ambiguity",
             "PLAN_F_REJECTED_BY_SOURCE_CONFLICT": "conflict guard", "PLAN_F_REJECTED_BY_SOURCE_AUTHORITY": "source authority / scope",
             "PLAN_F_REJECTED_BY_SUBJECT_GUARD": "grounding (subject / D-111 guard)", "PLAN_F_REJECTED_BY_FINDINGS_VERIFIER": "verification",
             "PLAN_F_REJECTED_BY_VERIFIER": "verification", "PLAN_F_REJECTED_BY_AGGREGATOR": "aggregation (a clause unsupported)",
             "PLAN_F_REJECTED_BY_HANDOFF": "handoff"}
    if outcome in table:
        return table[outcome]
    if gp is None or not applied:
        refused = [r for c in ((gp or {}).get("clauses") or {}).values() for r in (c.get("refused") or [])]
        if refused:
            return "binding refused (%s)" % refused[0].get("reason", "")[:60]
        if not named:
            return "no guide system named (handbook path decided)"
        return "no approved guide for the owning specialist"
    tried = [t for c in (gp.get("clauses") or {}).values() for t in (c.get("tried") or [])]
    reasons = [str(t.get("reason") or "") for t in tried]
    if not tried:
        return "retrieval / selection (no guide candidate reached the selection)"
    first = reasons[0] if reasons else ""
    if "conflict" in first:
        return "conflict"
    if "does not state" in first or "named entity" in first:
        return "grounding (an asked word the evidence does not print)"
    if "another action" in first or "names neither" in first:
        return "relation (heading names another action or object)"
    if any(k in first for k in ("step", "list", "section", "structure", "sub-section", "continues", "fragment")):
        return "structure recovery / completeness"
    return "evidence selection (%s)" % first[:60]


rows = {}
for qid, q in Q.items():
    a = ACTIVE.get(qid) or {}
    rows[qid] = {"id": qid, "set": q.get("set"), "category": q["category"], "expected_source": q.get("source"), "answerable": q.get("answerable"),
                 "plan_f_should_serve": q.get("plan_f_should_serve", q.get("served_expected")), "served": served(a), "guide_served": guide_served(a),
                 "outcome": diag(a).get("outcome"), "routing_final": diag(a).get("routing_final"), "plan_f_sources": diag(a).get("plan_f_sources"),
                 "forms": diag(a).get("guide_answer_forms") or [], "parts": diag(a).get("guide_answer_parts") or 0, "pages": a.get("pages"),
                 "stop_layer": stop_layer(qid), "verdict": verdict(qid) if served(a) else None}


def summarize(ids, *, count_handbook=True):
    rs = [rows[i] for i in ids]
    srv = [r for r in rs if (r["served"] if count_handbook else r["guide_served"])]
    v = Counter(r["verdict"] or "not_reviewed" for r in srv)
    good = v.get("CORRECT", 0) + v.get("ACCEPTABLE BUT LIMITED", 0)
    return {"questions": len(rs), "plan_f_served": len(srv), "guide_served": sum(r["guide_served"] for r in rs), "fallback": len(rs) - len(srv),
            "coverage_pct": pct(len(srv), len(rs)), "correct": v.get("CORRECT", 0), "limited": v.get("ACCEPTABLE BUT LIMITED", 0),
            "wrong": v.get("WRONG", 0), "not_reviewed": v.get("not_reviewed", 0), "precision_pct": pct(good, len(srv)) if srv else None,
            "served_ids": [r["id"] for r in srv], "stop_layers": dict(Counter(r["stop_layer"] for r in rs if not (r["served"] if count_handbook else r["guide_served"])))}


metrics = {"kb": R.get("kb"), "questions": len(Q), "sets": {k: len(v) for k, v in SETS.items()}}
# ---- P0 regression ----
metrics["p0"] = {"core_guide": summarize(CORE, count_handbook=False), "all_guide_served": summarize(SETS["p0"], count_handbook=False),
                 "per_source": {}}
for src in sorted({Q[i].get("source") for i in CORE if Q[i].get("source")}):
    ids = [i for i in CORE if Q[i].get("source") == src]
    metrics["p0"]["per_source"][src] = "%d of %d" % (sum(rows[i]["guide_served"] for i in ids), len(ids))
# ---- P1 ----
p1_guide = [i for i in SETS["p1"] if Q[i]["category"] == "p1_guide"]
metrics["p1"] = {"all": summarize(SETS["p1"]), "p1_guide_questions": summarize(p1_guide),
                 "per_source": {src: summarize([i for i in p1_guide if Q[i].get("source") == src]) for src in sorted({Q[i]["source"] for i in p1_guide})},
                 "must_not_serve_guide": {i: rows[i]["guide_served"] for i in SETS["p1"] if Q[i].get("served_expected") is False},
                 "by_category": {c: summarize([i for i in SETS["p1"] if Q[i]["category"] == c]) for c in sorted({Q[i]["category"] for i in SETS["p1"]})}}
# ---- held-out ----
ho = SETS["held_out"]
answerable = [i for i in ho if Q[i].get("answerable") == "yes"]
ho_all = summarize(ho)
metrics["held_out"] = {
    "all": ho_all, "answerable_yes": summarize(answerable), "partial": summarize([i for i in ho if Q[i].get("answerable") == "partial"]),
    "not_answerable": summarize([i for i in ho if Q[i].get("answerable") == "no"]),
    "coverage_all_pct": pct(ho_all["plan_f_served"], len(ho)),
    "coverage_answerable_pct": pct(sum(rows[i]["served"] for i in answerable), len(answerable)),
    "should_serve_true": {i: rows[i]["served"] for i in ho if Q[i].get("plan_f_should_serve") is True},
    "should_serve_false_but_served": [i for i in ho if Q[i].get("plan_f_should_serve") is False and rows[i]["served"]],
    "false_fallbacks_answerable": {i: rows[i]["stop_layer"] for i in answerable if not rows[i]["served"]},
    "by_category": {c: summarize([i for i in ho if Q[i]["category"] == c]) for c in sorted({Q[i]["category"] for i in ho})}}
# ---- safety, invariance, determinism, latency, model calls ----
safety_cats = ("conflict", "guide_contamination", "handbook_contamination", "wrong_source", "appraisal", "known_hr", "annual_leave", "authority",
               "incomplete_procedure", "policy_boundary", "unsupported")
metrics["safety_groups"] = {c: {"questions": len([i for i in Q if Q[i]["category"] == c]),
                                "served": [i for i in Q if Q[i]["category"] == c and rows[i]["served"]],
                                "guide_served": [i for i in Q if Q[i]["category"] == c and rows[i]["guide_served"]]} for c in safety_cats}
# ---- the old held-out set before and after; the fresh held-out set ----
metrics["held_out"]["before_after"] = {
    "served_before": [i for i in ho if diag(BEFORE.get(i)).get("served") == "plan_f"],
    "served_after": [i for i in ho if rows[i]["served"]],
    "answerable_served_before": sum(1 for i in answerable if diag(BEFORE.get(i)).get("served") == "plan_f"),
    "answerable_served_after": sum(1 for i in answerable if rows[i]["served"]),
    "changed": [i for i in ho if (diag(BEFORE.get(i)).get("served"), (BEFORE.get(i) or {}).get("answer"))
                != (diag(ACTIVE.get(i)).get("served"), (ACTIVE.get(i) or {}).get("answer"))]}
fr = SETS["fresh"]
fr_yes = [i for i in fr if Q[i].get("answerable") == "yes"]
fr_all = summarize(fr)
metrics["fresh"] = {
    "all": fr_all, "answerable_yes": summarize(fr_yes), "partial": summarize([i for i in fr if Q[i].get("answerable") == "partial"]),
    "not_answerable": summarize([i for i in fr if Q[i].get("answerable") == "no"]),
    "counts": {"questions": len(fr), "yes": len(fr_yes), "partial": sum(1 for i in fr if Q[i].get("answerable") == "partial"),
               "no": sum(1 for i in fr if Q[i].get("answerable") == "no")},
    "coverage_all_pct": pct(fr_all["plan_f_served"], len(fr)),
    "coverage_answerable_pct": pct(sum(rows[i]["served"] for i in fr_yes), len(fr_yes)),
    "should_serve_true": {i: rows[i]["served"] for i in fr if Q[i].get("plan_f_should_serve") is True},
    "should_serve_false_but_served": [i for i in fr if Q[i].get("plan_f_should_serve") is False and rows[i]["served"]],
    "false_fallbacks_answerable": {i: rows[i]["stop_layer"] for i in fr_yes if not rows[i]["served"]},
    "by_category": {c: summarize([i for i in fr if Q[i]["category"] == c]) for c in sorted({Q[i]["category"] for i in fr})}}
metrics["held_out_traps"] = {c: {"served": [i for i in ho if Q[i]["category"] == c and rows[i]["served"]],
                                 "guide_served": [i for i in ho if Q[i]["category"] == c and rows[i]["guide_served"]]}
                             for c in ("held_out_policy_vs_system_trap", "held_out_wrong_source_collision", "held_out_unsupported", "held_out_yes_no_false_premise")}
metrics["shadow"] = {"questions": len(SHADOW), "identical_to_direct_legacy": sum(1 for r in SHADOW.values() if r.get("identical_to_direct") is True),
                     "visible_mismatches": sum(1 for r in SHADOW.values() if r.get("identical_to_direct") is False),
                     "per_set_mismatches": {k: sum(1 for i in v if (SHADOW.get(i) or {}).get("identical_to_direct") is False) for k, v in SETS.items()}}
control = []
for qid, r in ACTIVE.items():
    o = OFF.get(qid) or {}
    if (diag(r).get("served"), diag(r).get("outcome"), r.get("answer"), r.get("pages")) != (diag(o).get("served"), diag(o).get("outcome"), o.get("answer"), o.get("pages")):
        control.append({"id": qid, "guide_served_on": guide_served(r), "verdict": verdict(qid)})
metrics["control_scopes_off_vs_on"] = {"changed": len(control), "unexplained": [c["id"] for c in control if not (c["guide_served_on"] and c["verdict"] in ("CORRECT", "ACCEPTABLE BUT LIMITED"))]}
metrics["determinism"] = {"repeat_records": len(R.get("repeats", [])), "differences": sorted({rec["id"] for rec in R.get("repeats", []) if
                          (rec.get("answer"), rec.get("pages"), diag(rec).get("outcome")) != ((ACTIVE.get(rec["id"]) or {}).get("answer"),
                          (ACTIVE.get(rec["id"]) or {}).get("pages"), diag(ACTIVE.get(rec["id"])).get("outcome"))})}


def guide_ms(d, key):
    return [c[key] for r in d.get("reports") or [] for c in ((r.get("guide_path") or {}).get("clauses") or {}).values() if isinstance(c.get(key), (int, float))]


P1_SOURCES = {"uos_blackboard_attendance_guide", "uos_blackboard_collaborate_guide", "uos_blackboard_communication_guide"}
metrics["latency"] = {
    mode: {"all": quantiles([r["wall_ms"] for r in t.values()]), "guide_served": quantiles([r["wall_ms"] for r in t.values() if guide_served(r)]),
           "guide_scope_fallback": quantiles([r["wall_ms"] for r in t.values() if diag(r).get("guide_scope_selected") and not guide_served(r)]),
           "p1_served": quantiles([r["wall_ms"] for r in t.values() if guide_served(r) and set(diag(r).get("plan_f_sources") or ()) & P1_SOURCES])}
    for mode, t in (("active", ACTIVE), ("shadow", SHADOW), ("active_off", OFF), ("legacy", LEGACY))}
metrics["latency"]["guide_retrieval_ms_per_clause"] = quantiles([v for d in DIRECT.values() for v in guide_ms(d, "retrieval_ms")])
metrics["latency"]["guide_selection_ms_per_clause"] = quantiles([v for d in DIRECT.values() for v in guide_ms(d, "selection_ms")])
metrics["latency"]["direct_verification_ms"] = {k: quantiles([(d.get("timings") or {}).get(k) for d in DIRECT.values() if guide_path(d)])
                                                for k in ("guards_ms", "findings_verify_ms", "verify_ms")}
metrics["llm"] = {"plan_f_llm_calls_max": max([diag(r).get("plan_f_llm_calls") or 0 for t in (ACTIVE, SHADOW, OFF) for r in t.values()] or [0]),
                  "direct_llm_calls_max": max([d.get("llm_calls") or 0 for d in DIRECT.values()] or [0])}
metrics["exceptions"] = sum(1 for r in R["results"] if r.get("exception")) + sum(1 for d in DIRECT.values() if d.get("exception"))
metrics["legacy_minimal"] = all(r.get("minimal_diagnostics") for r in LEGACY.values())
metrics["legacy_cache_untouched"] = R.get("legacy_cache_untouched")
metrics["rows"] = list(rows.values())
metrics["total_script_s"] = R.get("total_script_s")
json.dump(metrics, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: metrics[k] for k in ("sets", "p0", "p1", "held_out", "fresh", "safety_groups", "held_out_traps", "shadow", "control_scopes_off_vs_on",
                                          "determinism", "llm", "exceptions", "legacy_minimal")}, indent=1, default=str)[:20000])
