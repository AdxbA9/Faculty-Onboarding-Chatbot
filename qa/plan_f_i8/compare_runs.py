"""Compare two recorded I-8 master runs question by question (served path, outcome, answer, pages, specialist calls, routing
and handoff codes) and report what changed. Used to confirm that a fix changed only the questions it was meant to change.

    python -X utf8 qa/plan_f_i8/compare_runs.py <before i8_results.json> <after i8_results.json>
"""
import json
import sys
from collections import Counter

A = json.load(open(sys.argv[1], encoding="utf-8"))
B = json.load(open(sys.argv[2], encoding="utf-8"))
FIELDS = ("served", "outcome", "routing_reason", "handoff_outcome", "specialist_calls", "plan_f_path", "findings_reason")


def view(rec):
    d = rec.get("diagnostics") or {}
    out = {f: d.get(f) for f in FIELDS}
    out["answer"] = rec.get("answer")
    out["pages"] = tuple(rec.get("pages") or [])
    out["exception"] = rec.get("exception")
    return out


for mode in ("active", "shadow"):
    a = {r["id"]: r for r in A["results"] if r["mode"] == mode}
    b = {r["id"]: r for r in B["results"] if r["mode"] == mode}
    changed = Counter()
    details = []
    for qid in a:
        if qid not in b:
            details.append((qid, "missing in the second run"))
            continue
        va, vb = view(a[qid]), view(b[qid])
        diff = [k for k in va if va[k] != vb[k]]
        if diff:
            changed[tuple(diff)] += 1
            details.append((qid, {k: (str(va[k])[:70], str(vb[k])[:70]) for k in diff}))
    served_a = sum(1 for r in a.values() if (r.get("diagnostics") or {}).get("served") == "plan_f")
    served_b = sum(1 for r in b.values() if (r.get("diagnostics") or {}).get("served") == "plan_f")
    print("%s: %d common questions; served %d -> %d; questions with any change: %d" % (mode.upper(), len(set(a) & set(b)), served_a, served_b, len(details)))
    for qid, d in details:
        print("  %s %s" % (qid, d))
    if mode == "shadow":
        same = sum(1 for r in b.values() if r.get("identical_to_direct") is True)
        print("  shadow identical to the direct legacy call: %d of %d" % (same, len(b)))
