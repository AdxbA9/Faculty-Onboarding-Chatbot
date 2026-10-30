"""Analysis of a recorded I-8 master run (``master_eval.py``) joined with the manual evidence review (``review.json``):
coverage and precision separately, false acceptances and false fallbacks, per-domain metrics, routing classification,
aggregation and handoff results, fallback-reason classification, SHADOW invariance, determinism, latency, specialist-call
efficiency, model calls, diagnostics hygiene, citation checks, paraphrase and typo robustness and the known regression cases.
Prints the report and writes ``metrics.json`` next to the results file.

    python -X utf8 qa/plan_f_i8/analyze.py <i8_results.json> [review.json]
"""
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from question_set import owners_of   # noqa: E402

DATA = json.load(open(sys.argv[1], encoding="utf-8"))
REVIEW = json.load(open(sys.argv[2], encoding="utf-8")) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else {}
Q = {q["id"]: q for q in DATA["questions"]}
ORDER = [q["id"] for q in DATA["questions"]]
ACTIVE = {r["id"]: r for r in DATA["results"] if r["mode"] == "active"}
SHADOW = {r["id"]: r for r in DATA["results"] if r["mode"] == "shadow"}
DIRECT = {r["id"]: r for r in DATA["direct"]}
METRICS = {}


def diag(qid):
    return (ACTIVE.get(qid) or {}).get("diagnostics") or {}


def served(qid):
    return diag(qid).get("served") == "plan_f"


def verdict(qid):
    return (REVIEW.get(qid) or {}).get("verdict", "UNREVIEWED")


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "n/a"


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    values.sort()
    p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    return {"n": len(values), "median": statistics.median(values), "p90": p90, "max": values[-1], "min": values[0]}


def fmt(s):
    return "n/a" if not s else "median %.0f ms, p90 %.0f ms, max %.0f ms (n=%d)" % (s["median"], s["p90"], s["max"], s["n"])


def domain_group(qid):
    q = Q[qid]
    if q["category"] == "adversarial" or q["domain"] == "none" or q["category"] in ("unsupported", "general", "ambiguous"):
        return "adversarial/unsupported"
    return {"teaching": "teaching", "research": "research", "hr": "hr", "institutional": "institutional", "cross": "cross-domain"}[q["domain"]]


def is_adversarial(qid):
    q = Q[qid]
    return q["category"] == "adversarial" or any(t.startswith("adversarial:") for t in q["tags"])


line = "=" * 120
print(line)
print("1. COVERAGE VS PRECISION (ACTIVE through the seam; verdicts from the manual evidence review)")
total = len(ORDER)
served_ids = [i for i in ORDER if served(i)]
fallback_ids = [i for i in ORDER if not served(i)]
v = Counter(verdict(i) for i in served_ids)
false_acc = [i for i in served_ids if Q[i]["expect"] == "fallback" or verdict(i) == "WRONG"]
false_fb = [i for i in fallback_ids if Q[i]["expect"] == "serve"]
exceptions = [r for r in DATA["results"] + DATA["repeats"] + DATA["legacy"] if r.get("exception")] + [r for r in DATA["direct"] if r.get("exception")]
print("  questions %d | served %d | fallback %d | coverage %s" % (total, len(served_ids), len(fallback_ids), pct(len(served_ids), total)))
print("  served: CORRECT %d | LIMITED %d | WRONG %d | UNREVIEWED %d | precision %s" % (
    v["CORRECT"], v["LIMITED"], v["WRONG"], v["UNREVIEWED"], pct(v["CORRECT"], len(served_ids))))
print("  false acceptances (served although the set requires a fallback, or WRONG on review): %d %s" % (len(false_acc), false_acc))
print("  false fallbacks (fallback although the handbook states the fact specifically): %d %s" % (len(false_fb), false_fb))
print("  exceptions in any pass: %d" % len(exceptions))
METRICS.update({"questions": total, "served": len(served_ids), "fallback": len(fallback_ids), "coverage_pct": round(100.0 * len(served_ids) / total, 1),
                "correct": v["CORRECT"], "limited": v["LIMITED"], "wrong": v["WRONG"], "unreviewed": v["UNREVIEWED"],
                "precision_pct": round(100.0 * v["CORRECT"] / len(served_ids), 1) if served_ids else None,
                "false_acceptances": false_acc, "false_fallbacks": false_fb, "exceptions": len(exceptions),
                "adversarial_questions": sum(1 for i in ORDER if is_adversarial(i)),
                "adversarial_category_questions": sum(1 for i in ORDER if Q[i]["category"] == "adversarial")})

print(line)
print("2. PER-DOMAIN METRICS")
rows = {}
for i in ORDER:
    g = domain_group(i)
    r = rows.setdefault(g, Counter())
    r["questions"] += 1
    r["served" if served(i) else "fallback"] += 1
    if served(i):
        r[verdict(i)] += 1
print("  %-24s %9s %6s %8s %7s %7s %5s" % ("group", "questions", "served", "fallback", "correct", "limited", "wrong"))
for g in ("teaching", "research", "hr", "institutional", "cross-domain", "adversarial/unsupported"):
    r = rows.get(g, Counter())
    print("  %-24s %9d %6d %8d %7d %7d %5d" % (g, r["questions"], r["served"], r["fallback"], r["CORRECT"], r["LIMITED"], r["WRONG"]))
METRICS["domains"] = {g: dict(rows.get(g, Counter())) for g in rows}
adv = [i for i in ORDER if is_adversarial(i)]
adv_served = [i for i in adv if served(i)]
print("  adversarial-tagged questions %d: served %d (verdicts %s), fallback %d" % (len(adv), len(adv_served), dict(Counter(verdict(i) for i in adv_served)), len(adv) - len(adv_served)))

print(line)
print("3. ROUTING CLASSIFICATION (expected owners vs the refined route; CORRECT / ACCEPTABLE / WRONG / AMBIGUOUS-SAFE-FALLBACK)")
routing_class = {}
for i in ORDER:
    d, owners = diag(i), set(owners_of(Q[i]))
    final, initial = set(d.get("routing_final") or []), set(d.get("routing_initial") or [])
    reason, outcome = d.get("routing_reason"), d.get("outcome")
    if reason == "ROUTE_AMBIGUOUS" or (outcome == "PLAN_F_NOT_APPLICABLE" and not owners):
        c = "AMBIGUOUS-SAFE-FALLBACK" if not served(i) else "WRONG"
    elif served(i):
        c = "CORRECT" if final and final <= owners and verdict(i) in ("CORRECT", "LIMITED", "UNREVIEWED") else "WRONG"
    elif outcome == "PLAN_F_NOT_APPLICABLE":
        c = "WRONG" if owners else "CORRECT"          # a question with an owner that no specialist was selected for (safe: legacy answered)
    elif not final:
        c = "ACCEPTABLE" if not owners else "WRONG"
    elif final == owners:
        c = "CORRECT"
    elif final & owners:
        c = "ACCEPTABLE"
    else:
        c = "WRONG"
    routing_class[i] = c
rc = Counter(routing_class.values())
print("  %s" % dict(rc))
wrong_served = [i for i in ORDER if routing_class[i] == "WRONG" and served(i)]
print("  WRONG routings: %s (served among them: %s)" % ([i for i in ORDER if routing_class[i] == "WRONG"], wrong_served))
for i in ORDER:
    if routing_class[i] in ("WRONG", "ACCEPTABLE"):
        d = diag(i)
        print("    %-24s %-4s owners=%-36s initial=%-30s final=%-30s %s outcome=%s served=%s" % (
            routing_class[i], i, "+".join(sorted(owners_of(Q[i]))) or "-", "+".join(d.get("routing_initial") or []) or "-",
            "+".join(d.get("routing_final") or []) or "-", d.get("routing_reason"), d.get("outcome"), served(i)))
codes = Counter(diag(i).get("routing_reason") for i in ORDER)
print("  routing codes: %s" % dict(codes))
METRICS["routing"] = dict(rc)
METRICS["routing_codes"] = {str(k): n for k, n in codes.items()}
METRICS["routing_wrong_served"] = wrong_served

print(line)
print("4. AGGREGATION (multi-finding path)")
aggs = [i for i in ORDER if diag(i).get("plan_f_path") == "multi_finding_aggregate"]
agg_served = [i for i in aggs if served(i)]
order_ok = dup_ok = 0
for i in agg_served:
    items = (ACTIVE[i].get("items") or [])
    clause_idx = [((it.get("plan_f") or {}).get("clause_index")) for it in items]
    numeric = [c for c in clause_idx if isinstance(c, int)]
    order_ok += numeric == sorted(numeric)
    parts = (ACTIVE[i].get("answer") or "").split("\n\n")
    dup_ok += len(parts) == len(set(parts))
print("  aggregation path taken %d times; served %d (verdicts %s); rejected %d" % (
    len(aggs), len(agg_served), dict(Counter(verdict(i) for i in agg_served)), len(aggs) - len(agg_served)))
print("  served aggregates with answer parts in request order: %d/%d; without duplicated parts: %d/%d" % (order_ok, len(agg_served), dup_ok, len(agg_served)))
for i in agg_served:
    d = diag(i)
    print("    %-4s clauses %s/%s specialists=%s considered=%s accepted=%s duplicates=%s conflict=%s pages=%s parts=%d %s" % (
        i, d.get("aggregation_covered"), d.get("aggregation_clauses"), "+".join(d.get("aggregation_specialists") or []), d.get("aggregation_findings_considered"),
        d.get("aggregation_findings_accepted"), d.get("aggregation_duplicates"), d.get("aggregation_conflict"), ACTIVE[i].get("pages"),
        len((ACTIVE[i].get("answer") or "").split("\n\n")), verdict(i)))
rej = Counter(diag(i).get("aggregation_reason") for i in aggs if not served(i))
print("  aggregate rejections by reason: %s" % dict(rej))
METRICS["aggregates_served"] = len(agg_served)
METRICS["aggregates_attempted"] = len(aggs)

print(line)
print("5. HANDOFFS (real-KB rounds)")
rounds = [i for i in ORDER if diag(i).get("handoff_attempted")]
executed = [i for i in rounds if (diag(i).get("handoff_depth") or 0) >= 1
            or any(r.get("executed") for r in (((DIRECT.get(i) or {}).get("attempt") or {}).get("handoff") or {}).get("records") or [])]
accepted = [i for i in rounds if diag(i).get("handoff_outcome") == "HANDOFF_ACCEPTED"]
depths = [diag(i).get("handoff_depth") for i in rounds]
print("  rounds attempted %d; executed %d; accepted (served through a handoff) %d; max depth %s; codes %s" % (
    len(rounds), len(executed), len(accepted), max(depths) if depths else 0, dict(Counter(diag(i).get("handoff_outcome") for i in rounds))))
for i in rounds:
    d, a = diag(i), (DIRECT.get(i) or {}).get("attempt") or {}
    recs = (a.get("handoff") or {}).get("records") or []
    print("    %-4s %s -> %s depth=%s visited=%s outcome=%-28s served=%s | %s" % (
        i, "+".join(d.get("handoff_source") or []), "+".join(d.get("handoff_target") or []), d.get("handoff_depth"), d.get("handoff_visited"),
        d.get("handoff_outcome"), served(i), Q[i]["q"][:60]))
    for r in recs:
        print("         %s -> %s clause=%s code=%s executed=%s status=%s" % (r["source"], r["target"], r["request_clause"], r["code"], r["executed"], r["status"]))
METRICS.update({"handoffs_attempted": len(rounds), "handoffs_executed": len(executed), "handoffs_accepted": len(accepted), "handoff_max_depth": max(depths) if depths else 0})

print(line)
print("6. FALLBACK REASONS (section 17) and false-fallback causes (section 19)")


def fallback_class(i):
    d = diag(i)
    o, reason = d.get("outcome"), (d.get("fallback_reason") or d.get("reason") or "")
    if o == "PLAN_F_NOT_APPLICABLE":
        return "general question / no domain cue (legacy-only behaviour)" if "greeting" not in reason else "greeting (legacy-only behaviour)"
    if o == "PLAN_F_REJECTED_BY_ROUTING":
        return "routing ambiguity"
    if o == "PLAN_F_REJECTED_BY_HANDOFF" or "handoff" in reason:
        return "handoff unresolved"
    if "ENUMERATION_INCOMPLETE" in reason or "enumerat" in reason.lower():
        return "enumeration incomplete"
    if "status partial" in reason:
        return "partial support"
    if "status not_found" in reason or "out_of_scope" in reason:
        return "unsupported capability (nothing found)"
    if "ask shape general" in reason:
        return "general question"
    if o in ("PLAN_F_REJECTED_BY_SUBJECT_GUARD", "PLAN_F_REJECTED_BY_FINDINGS_VERIFIER", "PLAN_F_UNSAFE_RESULT", "PLAN_F_REJECTED_BY_AGGREGATOR"):
        return "expected safety fallback (%s)" % o.replace("PLAN_F_", "").lower()
    if o in ("PLAN_F_REJECTED_BY_VERIFIER", "PLAN_F_ERROR_REDACTED", "PLAN_F_REQUIRES_FALLBACK", "PLAN_F_REJECTED_BY_QUOTE_GUARD"):
        return "unexpected Plan F rejection (%s)" % o
    return "other (%s)" % o


fc = Counter(fallback_class(i) for i in fallback_ids)
for k, n in fc.most_common():
    print("  %3d  %s" % (n, k))
print("  false-fallback causes:")
for i in false_fb:
    d = diag(i)
    print("    %-4s %-40s outcome=%-34s %s | %s" % (i, fallback_class(i), d.get("outcome"), (d.get("fallback_reason") or "")[:90], Q[i]["q"][:60]))
METRICS["fallback_reasons"] = dict(fc)

print(line)
print("7. SHADOW MASTER CHECK")
sh = list(SHADOW.values())
identical = [r for r in sh if r.get("identical_to_direct") is True]
mism = [r for r in sh if r.get("identical_to_direct") is not True]
leak = [r for r in sh if (r.get("diagnostics") or {}).get("served") != "legacy"]
sh_exc = [r for r in sh if r.get("exception")]
print("  records %d; visible output identical to the direct legacy call %d; mismatches %d; served != legacy %d; exceptions %d" % (
    len(sh), len(identical), len(mism), len(leak), len(sh_exc)))
for r in mism:
    print("    MISMATCH %s fields=%s exc=%s" % (r["id"], r.get("direct_mismatch_fields"), r.get("direct_legacy_exception")))
METRICS.update({"shadow_records": len(sh), "shadow_mismatches": len(mism), "shadow_leaks": len(leak)})

print(line)
print("8. DETERMINISM (main ACTIVE pass + 3 repetitions)")
by_id = defaultdict(list)
for r in DATA["repeats"]:
    by_id[r["id"]].append(r)
unstable = []
for i, reps in by_id.items():
    obs = [ACTIVE[i]] + reps
    keys = [((o.get("diagnostics") or {}).get("served"), (o.get("diagnostics") or {}).get("outcome"), o.get("answer"), tuple(o.get("pages") or []),
             (o.get("diagnostics") or {}).get("specialist_calls"), (o.get("diagnostics") or {}).get("routing_reason"), (o.get("diagnostics") or {}).get("handoff_outcome"))
            for o in obs]
    if len(set(keys)) != 1:
        unstable.append((i, keys))
print("  questions %d x %d observations: stable %d, unstable %d" % (len(by_id), 4, len(by_id) - len(unstable), len(unstable)))
for i, keys in unstable:
    print("    UNSTABLE %s: %s" % (i, [k[:2] + (k[3],) for k in keys]))
METRICS.update({"determinism_questions": len(by_id), "determinism_unstable": len(unstable)})

print(line)
print("9. LATENCY (measured values only)")
cold = DATA.get("cold") or {}
print("  knowledge base build (v12 cache): %.1f s; cold first ACTIVE request wall %.0f ms (registry build %.0f ms); SHADOW first request wall %.0f ms (registry %.0f ms)" % (
    DATA["kb"]["build_s"], (cold.get("active") or {}).get("first_request_wall_ms") or 0, (cold.get("active") or {}).get("registry_build_ms") or 0,
    (cold.get("shadow") or {}).get("first_request_wall_ms") or 0, (cold.get("shadow") or {}).get("registry_build_ms") or 0))
warm = [i for i in ORDER[1:]]
single = [i for i in warm if served(i) and diag(i).get("plan_f_path") == "single_finding_passthrough"]
aggr = [i for i in warm if served(i) and diag(i).get("plan_f_path") == "multi_finding_aggregate"]
fb = [i for i in warm if not served(i)]
routing_rej = [i for i in warm if diag(i).get("outcome") == "PLAN_F_REJECTED_BY_ROUTING"]
lat = {
    "served single Plan F (ACTIVE wall)": stats([ACTIVE[i]["wall_ms"] for i in single]),
    "served aggregate (ACTIVE wall)": stats([ACTIVE[i]["wall_ms"] for i in aggr]),
    "routing rejection attempt (plan_f_ms)": stats([diag(i).get("plan_f_ms") for i in routing_rej]),
    "Plan F attempt before a legacy fallback (plan_f_ms)": stats([diag(i).get("plan_f_ms") for i in fb if diag(i).get("attempted")]),
    "legacy fallback (ACTIVE wall, attempt + legacy)": stats([ACTIVE[i]["wall_ms"] for i in fb]),
    "total ACTIVE response (all warm questions)": stats([ACTIVE[i]["wall_ms"] for i in warm]),
    "direct legacy call (SHADOW pass)": stats([SHADOW[i].get("direct_legacy_wall_ms") for i in ORDER if i in SHADOW]),
    "SHADOW response (legacy + Plan F attempt)": stats([SHADOW[i]["wall_ms"] for i in ORDER[1:] if i in SHADOW]),
    "LEGACY sample (in-process)": stats([r["wall_ms"] for r in DATA["legacy"]]),
}
for k, s in lat.items():
    print("  %-52s %s" % (k, fmt(s)))
regs = [diag(i).get("registry_reused") for i in ORDER]
print("  registry reused on %d of %d ACTIVE questions (built on the first)" % (sum(1 for r in regs if r), len(regs)))
METRICS["latency"] = {k: s for k, s in lat.items()}
METRICS["cold"] = cold
METRICS["kb_build_s"] = DATA["kb"]["build_s"]

print(line)
print("10. SPECIALIST-CALL EFFICIENCY")
calls = Counter()
for i in ORDER:
    c = diag(i).get("specialist_calls")
    calls["0" if not c else "1" if c == 1 else "2" if c == 2 else "3+"] += 1
print("  specialist calls per attempt: %s" % dict(calls))
fan = [(i, diag(i).get("specialist_calls"), served(i), ACTIVE[i]["wall_ms"]) for i in ORDER if (diag(i).get("specialist_calls") or 0) >= 2]
print("  attempts with 2+ calls: %d (served %d); unnecessary fan-out = 2+ calls and fallback: %d" % (len(fan), sum(1 for f in fan if f[2]), sum(1 for f in fan if not f[2])))
for i, c, s, w in fan:
    if not s:
        print("    %-4s calls=%s wall=%5.0f ms outcome=%-34s | %s" % (i, c, w, diag(i).get("outcome"), Q[i]["q"][:60]))
METRICS["specialist_calls"] = dict(calls)

print(line)
print("11. MODEL CALLS AND DIAGNOSTICS HYGIENE")
all_recs = DATA["results"] + DATA["repeats"] + DATA["legacy"]
pf_llm = max([(r.get("diagnostics") or {}).get("plan_f_llm_calls") or 0 for r in all_recs] + [((d.get("attempt") or {}).get("llm_calls") or 0) for d in DATA["direct"]] +
             [((d.get("attempt") or {}).get("dispatch_llm_calls") or 0) for d in DATA["direct"]] + [int(bool((d.get("attempt") or {}).get("used_llm"))) for d in DATA["direct"]])
legacy_llm = sum((r.get("llm_calls") or 0) for r in all_recs)
bad_q = bad_a = 0
for r in all_recs:
    text = json.dumps(r.get("diagnostics") or {})
    q, a = r["question"].strip(), (r.get("answer") or "")
    if q and len(q) > 3 and q in text:
        bad_q += 1
    if len(a) > 20 and a in text:
        bad_a += 1
print("  Plan F model calls (max over every record): %d; legacy model calls (sum): %d; groq client configured: %s" % (pf_llm, legacy_llm, DATA["kb"]["groq_client_configured"]))
print("  diagnostic records %d: containing the raw question %d, containing the raw answer %d" % (len(all_recs), bad_q, bad_a))
METRICS.update({"plan_f_llm_calls": pf_llm, "legacy_llm_calls": legacy_llm, "diag_records": len(all_recs), "diag_with_question": bad_q, "diag_with_answer": bad_a})

print(line)
print("12. CITATION CHECKS on every served answer's evidence (quote in chunk, chunk page is the cited page, quote in page text)")
cit_items = cit_bad = 0
for i in served_ids:
    for it in ACTIVE[i].get("items") or []:
        cit_items += 1
        if not (it["quote_in_chunk"] and it["meta_page_is_cited_page"] and it["quote_in_page_normalized"]):
            cit_bad += 1
            print("    CITATION DEFECT %s page=%s idx=%s in_chunk=%s meta_page=%s in_page_norm=%s" % (i, it["page"], it["idx"], it["quote_in_chunk"], it["meta_page_is_cited_page"], it["quote_in_page_normalized"]))
pages_ok = sum(1 for i in served_ids if set(ACTIVE[i].get("pages") or []) == {it["page"] for it in ACTIVE[i].get("items") or []})
rev_cit = Counter((REVIEW.get(i) or {}).get("citation", "UNREVIEWED") for i in served_ids)
print("  evidence items %d, programmatic defects %d; served answers whose page list equals the evidence pages %d/%d; manual citation review %s" % (
    cit_items, cit_bad, pages_ok, len(served_ids), dict(rev_cit)))
METRICS.update({"citation_items": cit_items, "citation_defects": cit_bad, "citation_review": dict(rev_cit)})

print(line)
print("13. PARAPHRASE ROBUSTNESS (served/fallback, pages and answer per variant)")
groups = defaultdict(list)
for i in ORDER:
    for t in Q[i]["tags"]:
        if t.startswith("paraphrase:"):
            groups[t].append(i)
para_stable = 0
for t, members in sorted(groups.items()):
    answers = {ACTIVE[i].get("answer") for i in members if served(i)}
    pages = {tuple(ACTIVE[i].get("pages") or []) for i in members if served(i)}
    s = [i for i in members if served(i)]
    print("  %-30s %d variants: served %d (%s), distinct answers %d, distinct page sets %d" % (t, len(members), len(s), ",".join(s), len(answers), len(pages)))
    for i in members:
        print("      %-4s %-8s pages=%-12s outcome=%-34s %s" % (i, "SERVED" if served(i) else "fallback", ACTIVE[i].get("pages"), diag(i).get("outcome"), Q[i]["q"][:60]))
    para_stable += len(pages) <= 1
METRICS["paraphrase_groups"] = len(groups)
METRICS["paraphrase_groups_single_page_set"] = para_stable

print(line)
print("14. TYPO / IMPERFECT INPUT")
for i in ORDER:
    if "typo" in Q[i]["tags"] or "fragment" in Q[i]["tags"] or "casual" in Q[i]["tags"]:
        print("  %-4s %-8s route=%-10s pages=%-10s outcome=%-30s %s | %r" % (i, "SERVED" if served(i) else "fallback", ACTIVE[i].get("query_type"), ACTIVE[i].get("pages"),
                                                                           diag(i).get("outcome"), verdict(i) if served(i) else "", Q[i]["q"]))

print(line)
print("15. KNOWN REGRESSION CASES")
for i in ORDER:
    known = [t for t in Q[i]["tags"] if t.startswith("known:")]
    if known:
        d = diag(i)
        print("  %-30s %-4s %-8s outcome=%-34s route=%-30s handoff=%-26s %s | %s" % (
            ",".join(k[6:] for k in known), i, "SERVED" if served(i) else "fallback", d.get("outcome"), d.get("routing_reason"), d.get("handoff_outcome"),
            verdict(i) if served(i) else "", Q[i]["q"][:60]))

print(line)
print("16. LEGACY SAMPLE (in-process): %d questions, minimal diagnostics %d, exceptions %d, runtime cache untouched %s" % (
    len(DATA["legacy"]), sum(1 for r in DATA["legacy"] if r.get("minimal_diagnostics")), sum(1 for r in DATA["legacy"] if r.get("exception")), DATA.get("legacy_cache_untouched")))

print(line)
print("17. SERVED ANSWERS (ACTIVE) with verdicts")
for i in served_ids:
    print("  %-10s %-4s %-26s pages=%-16s calls=%s %s" % (verdict(i), i, diag(i).get("plan_f_path"), ACTIVE[i].get("pages"), diag(i).get("specialist_calls"), Q[i]["q"]))
json.dump(METRICS, open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "metrics.json"), "w", encoding="utf-8"), indent=1, default=str)
