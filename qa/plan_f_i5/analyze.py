"""Analysis of a recorded I-5 run: the I-4 served set regression, the former limited cases, the known cases, the multipart
classification (A to F), acceptance before and after, aggregation latency, SHADOW invariance, LLM calls and diagnostics hygiene.

    python -X utf8 qa/plan_f_i5/analyze.py <i5_results.json> <i4_results.json>
"""
import json
import statistics
import sys

I5 = json.load(open(sys.argv[1], encoding="utf-8"))
I4 = json.load(open(sys.argv[2], encoding="utf-8"))
direct = {r["question"]: r for r in I5["direct"]}
active = {r["question"]: r for r in I5["results"] if r["mode"] == "active"}
shadow = {r["question"]: r for r in I5["results"] if r["mode"] == "shadow"}
i4_active = {r["question"]: r for r in I4["results"] if r["mode"] == "active"}
i4_served = {q: r for q, r in i4_active.items() if (r.get("diagnostics") or {}).get("served") == "plan_f"}
corpus = [r for r in I5["direct"] if r["group"] == "corpus"]
probes = [r for r in I5["direct"] if r["group"] == "probe"]
agg_set = [r for r in I5["direct"] if r["group"] == "aggregation_set"]
WLAM = "What is the Workload Allocation Model?"
CONSULTANCY = "How many days per week can faculty spend on consultancy?"
HR_PROVIDES = "What does HR provide to new faculty members?"
KNOWN = ["What does Human Resources do?", "What does HR do?", "How many days of annual leave do faculty members get?"]


def med(values):
    values = [v for v in values if v is not None]
    return (statistics.median(values), min(values), max(values), len(values)) if values else (None, None, None, 0)


def fmt(stats):
    m, lo, hi, n = stats
    return "n/a" if m is None else "median %.1f ms (%.1f to %.1f, n=%d)" % (m, lo, hi, n)


def served(q):
    d = (active.get(q) or {}).get("diagnostics") or {}
    return d.get("served") == "plan_f"


print("=" * 118)
print("1. I-4 SERVED SET (%d questions): still served, same answer and pages?" % len(i4_served))
preserved = 0
for q in i4_served:
    rec = direct[q]
    a = rec["attempt"]
    ok = served(q) and rec.get("i4_answer_equal") is True
    preserved += ok
    print("  %-5s path=%-26s %-34s pages=%-10s %s" % ("SAME" if ok else "DIFF", a["path"], a["outcome"], a["pages"], q[:60]))
print("  preserved: %d of %d" % (preserved, len(i4_served)))

print("=" * 118)
print("2. FORMER LIMITED CASES")
for q in (WLAM, CONSULTANCY, HR_PROVIDES):
    a = direct[q]["attempt"]
    print("  %-34s path=%-26s served=%-5s | %s" % (a["outcome"], a["path"], served(q), q))
    if a["answer"]:
        print("      answer: %s" % a["answer"][:200].replace("\n", " / "))
        for it in a["items"]:
            pf = it.get("plan_f") or {}
            print("      item p%s chunk %s finding %s derived_from=%s slice=%s" % (it["page"], it["idx"], pf.get("finding_id"), pf.get("derived_from"), pf.get("slice")))
    else:
        print("      reason: %s" % a["reason"][:160])

print("=" * 118)
print("3. KNOWN CASES (must stay blocked)")
for q in KNOWN:
    a = direct[q]["attempt"]
    print("  %-34s served=%-5s | %s | %s" % (a["outcome"], served(q), q, a["reason"][:90]))

print("=" * 118)
print("4. MULTIPART / MULTI-SPECIALIST CLASSIFICATION (A: aggregated; B: clause missing or rejected; C: cross-domain unresolved; "
      "D: handoff required; E: synthesis or general ask; F: unsupported)")
categories = {}
for rec in I5["direct"]:
    dec = rec["decision"]
    if len(dec["clauses"]) < 2 and len(dec["tasks"]) < 2:
        continue
    a = rec["attempt"]
    agg = a.get("aggregation") or {}
    reason = agg.get("reason")
    runs = rec.get("runs") or []
    handoffs = any(r["handoffs"] for r in runs)
    shapes = [rep.get("ask_shape") for r in runs for rep in r["reports"]]
    if a["outcome"] == "PLAN_F_ACCEPTED" and a["path"] == "multi_finding_aggregate":
        cat = "A"
    elif reason == "HANDOFF_REQUIRED" or handoffs:
        cat = "D"
    elif reason in ("CROSS_DOMAIN_UNRESOLVED", "UNRESOLVED_CLAUSE"):
        cat = "C"
    elif any(s in ("general", "specific_unknown") for s in shapes) or (reason == "CHILD_FINDING_REJECTED" and ("GENERAL_ASK_MISMATCH" in a["reason"] or "ENUMERATION_INCOMPLETE" in a["reason"])):
        cat = "E"
    elif reason in ("MISSING_REQUIRED_CLAUSE", "CHILD_FINDING_REJECTED", "VALUE_CONFLICT"):
        cat = "B"
    else:
        cat = "F"
    categories.setdefault(cat, []).append(rec["n"])
    statuses = ",".join("%s:%s" % (r["specialist"], r["status"]) for r in runs) or "(not dispatched)"
    print("  [%3d] %s %-15s clauses=%d tasks=%d %-55s agg=%-24s | %s" % (rec["n"], cat, rec["group"], len(dec["clauses"]), len(dec["tasks"]), statuses[:55], reason, rec["question"][:62]))
    if cat == "A":
        print("        pages=%s specialists=%s duplicates=%s evidence=%s" % (agg.get("pages"), agg.get("specialists"), agg.get("duplicates"), [(e["finding_id"], e["clause_indexes"]) for e in agg.get("evidence", [])]))
print("  totals: " + ", ".join("%s=%d" % (c, len(ns)) for c, ns in sorted(categories.items())) + " (examined %d)" % sum(len(v) for v in categories.values()))

print("=" * 118)
print("5. ACCEPTANCE BEFORE / AFTER (ACTIVE served by Plan F)")
i4_corpus = sum(1 for r in corpus if r["question"] in i4_served)
i5_corpus = sum(1 for r in corpus if served(r["question"]))
print("  I-3/I-4 corpus (62): I-4 %d (%.0f%%) -> I-5 %d (%.0f%%); fallback %d -> %d" % (i4_corpus, 100.0 * i4_corpus / len(corpus), i5_corpus, 100.0 * i5_corpus / len(corpus), len(corpus) - i4_corpus, len(corpus) - i5_corpus))
i4_probe = sum(1 for r in probes if r["question"] in i4_served)
i5_probe = sum(1 for r in probes if served(r["question"]))
print("  I-4 probes (14): I-4 %d -> I-5 %d" % (i4_probe, i5_probe))
i5_agg = sum(1 for r in agg_set if served(r["question"]))
print("  aggregation set (%d): I-5 %d served (%d aggregates, %d single-finding)" % (
    len(agg_set), i5_agg, sum(1 for r in agg_set if served(r["question"]) and r["attempt"]["path"] == "multi_finding_aggregate"),
    sum(1 for r in agg_set if served(r["question"]) and r["attempt"]["path"] == "single_finding_passthrough")))
by_group = {}
for r in agg_set:
    by_group.setdefault(r["domain"], [0, 0])
    by_group[r["domain"]][1] += 1
    by_group[r["domain"]][0] += served(r["question"])
print("  aggregation set by tag: " + ", ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in by_group.items()))
aggregates = [r for r in I5["direct"] if served(r["question"]) and r["attempt"]["path"] == "multi_finding_aggregate"]
print("  REAL-KB AGGREGATES SERVED: %d" % len(aggregates))
for r in aggregates:
    agg = r["attempt"]["aggregation"]
    print("    [%3d] pages=%s specialists=%s clauses=%d evidence=%d dup=%d | %s" % (r["n"], agg["pages"], "+".join(agg["specialists"]), agg["clause_count"], agg["findings_accepted"], len(agg["duplicates"]), r["question"][:70]))
sliced = [r for r in I5["direct"] if served(r["question"]) and any((it.get("plan_f") or {}).get("derived_from") for it in r["attempt"]["items"])]
print("  served through a provision slice: %d %s" % (len(sliced), [r["n"] for r in sliced]))
total_served = sum(1 for r in I5["direct"] if served(r["question"]))
print("  all %d questions: served %d, fallback %d" % (len(I5["direct"]), total_served, len(I5["direct"]) - total_served))

print("=" * 118)
print("6. AGGREGATION LATENCY (adapter timings of dispatched aggregation attempts) versus I-4 verification and specialist retrieval")
disp = [r["attempt"]["timings"] for r in I5["direct"] if (r["attempt"].get("aggregation") or {}).get("dispatched")]
print("  aggregation total:      %s" % fmt(med([t.get("aggregation_ms") for t in disp])))
for key in ("map", "verify", "conflict", "dedup"):
    print("  aggregation %-10s  %s" % (key + ":", fmt(med([t.get("aggregation_%s_ms" % key) for t in disp]))))
print("  rendering + legacy verify (verify_ms, accepted aggregates): %s" % fmt(med([r["attempt"]["timings"].get("verify_ms") for r in aggregates])))
print("  specialist retrieval and rerank (specialist_ms, dispatched aggregation attempts): %s" % fmt(med([t.get("specialist_ms") for t in disp])))
print("  I-4 findings verification on the pass-through path (findings_verify_ms): %s" % fmt(med([r["attempt"]["timings"].get("findings_verify_ms") for r in I5["direct"] if r["attempt"]["path"] == "single_finding_passthrough"])))
pre = [r["attempt"]["timings"].get("total_ms") for r in I5["direct"] if r["attempt"].get("aggregation") and not r["attempt"]["aggregation"]["dispatched"]]
print("  pre-dispatch rejections (no specialist run): total %s" % fmt(med(pre)))
print("  ACTIVE wall, served by Plan F: %s" % fmt(med([active[q]["wall_ms"] for q in active if served(q)])))
print("  ACTIVE wall, aggregates:       %s" % fmt(med([active[r["question"]]["wall_ms"] for r in aggregates])))
print("  ACTIVE wall, legacy fallback:  %s" % fmt(med([active[q]["wall_ms"] for q in active if not served(q)])))

print("=" * 118)
print("7. SHADOW INVARIANCE: %d of %d identical to the direct legacy call; exceptions: %d" % (
    sum(1 for r in shadow.values() if r.get("identical_to_direct") is True), len(shadow), sum(1 for r in I5["results"] if r.get("exception"))))

print("=" * 118)
llm = max([r["attempt"]["llm_calls"] for r in I5["direct"]] + [r["attempt"]["dispatch_llm_calls"] for r in I5["direct"]] +
          [int(bool(r["attempt"]["used_llm"])) for r in I5["direct"]] + [(r.get("diagnostics") or {}).get("plan_f_llm_calls", 0) for r in I5["results"]])
print("8. PLAN F LLM CALLS (max over every attempt, dispatch and diagnostic): %d" % llm)

print("=" * 118)
bad = 0
for r in I5["results"]:
    text = json.dumps(r.get("diagnostics") or {})
    answer = r.get("answer") or ""
    if r["question"] in text or (len(answer) > 20 and answer in text):
        bad += 1
print("9. DIAGNOSTICS HYGIENE: %d of %d records carry the question or the answer text" % (bad, len(I5["results"])))
keys = set()
for r in I5["results"]:
    keys |= set((r.get("diagnostics") or {}).keys())
print("   aggregation keys present: %s" % sorted(k for k in keys if k.startswith("aggregation_") or k == "plan_f_path"))

print("=" * 118)
print("10. SERVED ANSWERS FOR REVIEW (ACTIVE)")
for r in I5["direct"]:
    if not served(r["question"]):
        continue
    a = r["attempt"]
    print("  [%3d] %-26s pages=%-12s %s" % (r["n"], a["path"], a["pages"], r["question"]))
    print("        %s" % a["answer"].replace("\n\n", " || ")[:400])
