"""Analyse i4_results.json: the 23-case re-evaluation table against I-3, outcome distribution, verification codes, latency, shadow
invariance, the known cases, diagnostics hygiene. Usage: python -X utf8 qa/plan_f_i4/analyze.py <i4_results.json> <i3_results.json>"""
import json
import statistics
import sys
from collections import Counter, defaultdict

R = json.load(open(sys.argv[1], encoding="utf-8"))
I3 = json.load(open(sys.argv[2], encoding="utf-8"))
direct = R["direct"]
results = R["results"]
by_mode = defaultdict(list)
for r in results:
    by_mode[r["mode"]].append(r)
i3_active = {r["question"]: r for r in I3["results"] if r["mode"] == "active"}
i3_accepted = {q for q, r in i3_active.items() if (r.get("diagnostics") or {}).get("served") == "plan_f"}
I3_REVIEW = {                                                        # the I-3 source review (docs/STATUS.md, entry of 2026-10-30)
    "What is the Workload Allocation Model?": "ACCEPTABLE BUT LIMITED",
    "How many days per week can faculty spend on consultancy?": "ACCEPTABLE BUT LIMITED",
    "What does HR provide to new faculty members?": "ACCEPTABLE BUT LIMITED",
}


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    return "n=%d median=%.1f min=%.1f max=%.1f ms" % (len(values), statistics.median(values), min(values), max(values))


print("=== KB ===", R["kb"])
active = {r["question"]: r for r in by_mode["active"]}
shadow = {r["question"]: r for r in by_mode["shadow"]}

print("\n=== I-3 ACCEPTED CASES RE-EVALUATED (%d) ===" % len(i3_accepted))
still, newly = [], []
for rec in direct:
    q = rec["question"]
    if q not in i3_accepted:
        continue
    a = active[q]
    d = a.get("diagnostics") or {}
    served = d.get("served")
    verdict = rec.get("direct") or {}
    old = I3_REVIEW.get(q, "CORRECT")
    row = (rec["domain"], rec.get("specialist"), verdict.get("request_kind"), old, verdict.get("reason"), verdict.get("completeness"),
           a.get("pages"), "PLAN F" if served == "plan_f" else "FALLBACK", q)
    (still if served == "plan_f" else newly).append(row)
    print("  %-13s %-16s %-12s I-3=%-23s I-4=%-24s %-20s pages=%-10s ACTIVE=%-8s | %s" % row)
print("still accepted: %d | newly fallback: %d" % (len(still), len(newly)))
for row in newly:
    print("  NEWLY FALLBACK: %s -> %s (%s)" % (row[-1], row[4], row[5]))
print("previously CORRECT still accepted: %d of %d" % (sum(1 for r in still if r[3] == "CORRECT"), sum(1 for r in still + newly if r[3] == "CORRECT")))
print("previously LIMITED now fallback: %d of %d" % (sum(1 for r in newly if r[3] != "CORRECT"), len(I3_REVIEW)))

print("\n=== DIRECT VERIFIER VERDICTS (every dispatched result, I-3 set only) ===")
base = [r for r in direct if r["domain"] != "probe"]
print("kinds:", dict(Counter((r.get("direct") or {}).get("request_kind") for r in base if r.get("direct"))))
print("reasons:", dict(Counter((r.get("direct") or {}).get("reason") for r in base if r.get("direct"))))
print("completeness:", dict(Counter((r.get("direct") or {}).get("completeness") for r in base if r.get("direct"))))
print("adapter outcomes:", dict(Counter(r["attempt"]["outcome"] for r in base)))
print("direct verify ms:", stats([r.get("direct_verify_ms") for r in base]))
print("adapter findings_verify_ms:", stats([(r["attempt"].get("timings") or {}).get("findings_verify_ms") for r in base]))
print("adapter specialist_ms:", stats([(r["attempt"].get("timings") or {}).get("specialist_ms") for r in base]))
print("adapter total_ms:", stats([(r["attempt"].get("timings") or {}).get("total_ms") for r in base]))
print("adapter LLM: used_llm", Counter(r["attempt"]["used_llm"] for r in base), "llm_calls", Counter(r["attempt"]["llm_calls"] for r in base))

print("\n=== SHADOW ===")
sh = [r for r in by_mode["shadow"] if r["domain"] != "probe"]
print("identical_to_direct:", Counter(r.get("identical_to_direct") for r in sh), "exceptions:", [r["question"] for r in sh if r.get("exception")])
print("served:", Counter((r.get("diagnostics") or {}).get("served") for r in sh))
print("plan_f_llm_calls:", Counter((r.get("diagnostics") or {}).get("plan_f_llm_calls") for r in sh))
print("findings_reason:", dict(Counter((r.get("diagnostics") or {}).get("findings_reason") for r in sh)))
print("shadow wall:", stats([r["wall_ms"] for r in sh]), "| findings_ms:", stats([(r.get("diagnostics") or {}).get("findings_ms") for r in sh]))

print("\n=== ACTIVE ===")
ac = [r for r in by_mode["active"] if r["domain"] != "probe"]
print("served:", Counter((r.get("diagnostics") or {}).get("served") for r in ac), "exceptions:", [r["question"] for r in ac if r.get("exception")])
print("outcomes:", dict(Counter((r.get("diagnostics") or {}).get("outcome") for r in ac)))
acc = [r for r in ac if (r.get("diagnostics") or {}).get("served") == "plan_f"]
fb = [r for r in ac if (r.get("diagnostics") or {}).get("served") == "legacy"]
print("accepted %d (%.0f%%) fallback %d (%.0f%%)" % (len(acc), 100.0 * len(acc) / len(ac), len(fb), 100.0 * len(fb) / len(ac)))
print("accepted wall:", stats([r["wall_ms"] for r in acc]), "| fallback wall:", stats([r["wall_ms"] for r in fb]))
print("accepted used_llm/llm_calls:", Counter((r.get("used_llm"), r.get("llm_calls")) for r in acc))
print("accepted findings_accepted:", Counter((r.get("diagnostics") or {}).get("findings_accepted") for r in acc))
for dom in ("teaching", "research", "hr", "institutional", "special"):
    rows = [r for r in ac if r["domain"] == dom]
    print("  %-13s n=%2d accepted=%2d" % (dom, len(rows), sum(1 for r in rows if (r.get("diagnostics") or {}).get("served") == "plan_f")))

print("\n=== KNOWN CASES ===")
for q in ("What does Human Resources do?", "What does HR do?", "How many days of annual leave do faculty members get?"):
    rec = next(r for r in direct if r["question"] == q)
    for mode in ("shadow", "active"):
        r = (shadow if mode == "shadow" else active)[q]
        d = r.get("diagnostics") or {}
        print("  %-6s %-52s served=%-7s outcome=%-26s direct=%s | answer starts: %r" % (
            mode, q, d.get("served"), d.get("outcome"), (rec.get("direct") or {}).get("reason"), str(r.get("answer"))[:40]))

print("\n=== PROBES (mutations and controls) ===")
for rec in [r for r in direct if r["domain"] == "probe"]:
    a = active[rec["question"]]
    d = rec.get("direct") or {}
    print("  %-38s direct=%-24s kind=%-12s served=%-7s | %s" % (rec["attempt"]["outcome"], d.get("reason"), d.get("request_kind"),
                                                                 (a.get("diagnostics") or {}).get("served"), rec["question"]))

print("\n=== DIAGNOSTICS HYGIENE ===")
bad = 0
for r in by_mode["shadow"] + by_mode["active"]:
    text = json.dumps(r.get("diagnostics") or {})
    if r["question"] in text or (r.get("answer") and len(r["answer"]) > 20 and r["answer"] in text):
        bad += 1
print("records containing the question or answer text:", bad, "of", len(by_mode["shadow"]) + len(by_mode["active"]))
print("total script s:", R.get("total_script_s"))
