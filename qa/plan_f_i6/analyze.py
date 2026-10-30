"""Analysis of a recorded I-6 run: the I-5 served set regression (single findings, aggregates, slices), the Category C and D
re-evaluation against the I-5 behaviour, newly served answers, acceptance before and after, routing and handoff latency,
SHADOW invariance, model calls, diagnostics hygiene and the served answers for review.

    python -X utf8 qa/plan_f_i6/analyze.py <i6_results.json>
"""
import json
import statistics
import sys

I6 = json.load(open(sys.argv[1], encoding="utf-8"))
direct = I6["direct"]
by_q = {r["question"]: r for r in direct}
active = {r["question"]: r for r in I6["results"] if r["mode"] == "active"}
shadow = {r["question"]: r for r in I6["results"] if r["mode"] == "shadow"}
KNOWN = ["What does Human Resources do?", "What does HR do?", "How many days of annual leave do faculty members get?"]
CATEGORY_C = ["Who approves research leave?", "Who approves Hajj leave?", "Who handles faculty visas and where is that office?",
              "Can I reduce my teaching load if I have a funded research project, and who approves it?", "Which office handles faculty housing?",
              "Who approves conference travel?", "Who approves the funding ceiling for research grants?",
              "How many theses can a faculty member supervise at once and what is the phone number of the Human Resources Department?"]
CATEGORY_D = ["Where is the Research Institute and what research funding is available?"]


def served(q):
    return ((active.get(q) or {}).get("diagnostics") or {}).get("served") == "plan_f"


def med(values):
    values = [v for v in values if v is not None]
    return (statistics.median(values), min(values), max(values), len(values)) if values else (None, None, None, 0)


def fmt(stats):
    m, lo, hi, n = stats
    return "n/a" if m is None else "median %.1f ms (%.1f to %.1f, n=%d)" % (m, lo, hi, n)


print("=" * 120)
print("1. I-5 SERVED SET: every question Plan F served in I-5 (ACTIVE), still served with the same answer and pages?")
prev_served = [r for r in direct if (r.get("i5") or {}).get("served")]
seen = set()
preserved = kept_aggregates = kept_slices = 0
for r in prev_served:
    if r["question"] in seen:
        continue
    seen.add(r["question"])
    same = served(r["question"]) and r.get("i5_answer_equal") is True
    preserved += same
    kind = r["i5"]["path"] or "?"
    if r["i5"]["path"] == "multi_finding_aggregate":
        kept_aggregates += same
    if any((i.get("plan_f") or {}).get("derived_from") for i in r["attempt"]["items"]):
        kept_slices += same
    print("  %-5s i5=%-26s i6=%-26s route=%-30s %s" % ("SAME" if same else "DIFF", kind, r["attempt"]["path"], r["attempt"]["routing"]["reason"], r["question"][:58]))
print("  preserved: %d of %d unique (aggregates kept %d, slices kept %d)" % (preserved, len(seen), kept_aggregates, kept_slices))

print("=" * 120)
print("2. CATEGORY C (I-5 cross-domain or shared-routing unresolved): old behaviour -> I-6 routing, specialist runs, outcome")
for q in CATEGORY_C:
    r = by_q[q]
    a, prev = r["attempt"], r.get("i5") or {}
    print("  %s" % q)
    print("      I-5: %-34s %-24s | I-6: route=%-30s final=%-28s calls=%d %-34s served=%s" % (
        prev.get("outcome"), prev.get("aggregation_reason"), a["routing"]["reason"], "+".join(a["routing"]["final_specialists"]), a["specialist_calls"],
        a["outcome"], served(q)))
    print("      runs: %s | reason: %s" % (a["runs"], a["reason"][:110]))
    if a["answer"]:
        print("      answer: %s | pages %s" % (a["answer"][:150].replace("\n", " / "), a["pages"]))
print("=" * 120)
print("3. CATEGORY D (I-5 handoff required)")
for q in CATEGORY_D:
    r = by_q[q]
    a, prev = r["attempt"], r.get("i5") or {}
    h = a.get("handoff") or {}
    print("  %s" % q)
    print("      I-5: %s %s | I-6: route=%s tasks=%s calls=%d outcome=%s handoff=%s" % (
        prev.get("outcome"), prev.get("aggregation_reason"), a["routing"]["reason"], r["refined_tasks"], a["specialist_calls"], a["outcome"], a.get("handoff_outcome")))
    for rec in h.get("records", []):
        print("      handoff %s -> %s clause=%s code=%s executed=%s status=%s detail=%s" % (rec["source"], rec["target"], rec["request_clause"], rec["code"], rec["executed"], rec["status"], rec["detail"][:80]))
    print("      reason: %s" % a["reason"][:140])

print("=" * 120)
print("4. HANDOFF ROUNDS ATTEMPTED IN THE WHOLE RUN")
for r in direct:
    h = r["attempt"].get("handoff")
    if not h or not h.get("attempted"):
        continue
    print("  [%3d] %-32s calls=%d served=%-5s %s" % (r["n"], h["code"], r["attempt"]["specialist_calls"], served(r["question"]), r["question"][:60]))
    for rec in h["records"]:
        print("        %s -> %s clause=%s code=%s executed=%s status=%s" % (rec["source"], rec["target"], rec["request_clause"], rec["code"], rec["executed"], rec["status"]))

print("=" * 120)
print("5. KNOWN CASES (must stay blocked)")
for q in KNOWN:
    a = by_q[q]["attempt"]
    print("  %-34s served=%-5s route=%-20s | %s | %s" % (a["outcome"], served(q), a["routing"]["reason"], q, a["reason"][:80]))

print("=" * 120)
print("6. NEWLY SERVED (served now, not served in I-5 or not in I-5)")
new = []
seen = set()
for r in direct:
    q = r["question"]
    if q in seen or not served(q):
        continue
    seen.add(q)
    prev = r.get("i5")
    if prev is None or not prev.get("served"):
        new.append(r)
for r in new:
    a = r["attempt"]
    print("  [%3d] %-16s path=%-26s route=%-30s handoff=%-18s calls=%d pages=%s | %s" % (
        r["n"], r["group"], a["path"], a["routing"]["reason"], a.get("handoff_outcome"), a["specialist_calls"], a["pages"], r["question"]))
    print("        %s" % a["answer"].replace("\n\n", " || ")[:420])
print("  newly served: %d (new real-KB answers to review)" % len(new))

print("=" * 120)
print("7. ACCEPTANCE BEFORE / AFTER (ACTIVE served, unique questions)")
groups = {}
for r in direct:
    q = r["question"]
    g = r["group"]
    groups.setdefault(g, {"n": 0, "i5": 0, "i6": 0, "seen": set()})
    if q in groups[g]["seen"]:
        continue
    groups[g]["seen"].add(q)
    groups[g]["n"] += 1
    groups[g]["i5"] += bool((r.get("i5") or {}).get("served"))
    groups[g]["i6"] += served(q)
for g, v in groups.items():
    print("  %-16s %3d questions: I-5 served %2d -> I-6 served %2d" % (g, v["n"], v["i5"], v["i6"]))
uniq = {r["question"] for r in direct}
print("  all %d unique questions: I-6 served %d, fallback %d" % (len(uniq), sum(1 for q in uniq if served(q)), sum(1 for q in uniq if not served(q))))
routes = {}
for r in direct:
    routes[r["attempt"]["routing"]["reason"]] = routes.get(r["attempt"]["routing"]["reason"], 0) + 1
print("  routing codes over %d runs: %s" % (len(direct), routes))

print("=" * 120)
print("8. LATENCY (adapter timings)")
tm = [r["attempt"]["timings"] for r in direct]
print("  routing refinement (routing_ms, all attempts):          %s" % fmt(med([t.get("routing_ms") for t in tm])))
print("  routing refinement, refined decisions only:             %s" % fmt(med([r["attempt"]["timings"].get("routing_ms") for r in direct if r["attempt"]["routing"]["refined"]])))
print("  pre-dispatch routing rejections, whole attempt:         %s" % fmt(med([r["attempt"]["timings"].get("total_ms") for r in direct if r["attempt"]["outcome"] == "PLAN_F_REJECTED_BY_ROUTING"])))
ho = [r for r in direct if (r["attempt"].get("handoff") or {}).get("attempted")]
print("  handoff round (handoff_ms, rounds attempted):           %s" % fmt(med([r["attempt"]["timings"].get("handoff_ms") for r in ho])))
ex = [r for r in ho if r["attempt"]["handoff"]["code"] == "HANDOFF_EXECUTED"]
print("  handoff target specialist (executed rounds):            %s" % fmt(med([r["attempt"]["timings"].get("handoff_specialist_ms") for r in ex])))
print("  first-round specialists (specialist_ms, dispatched):    %s" % fmt(med([t.get("specialist_ms") for t in tm if t.get("specialist_ms") is not None])))
print("  I-4 verification, pass-through accepted:                %s" % fmt(med([r["attempt"]["timings"].get("findings_verify_ms") for r in direct if r["attempt"]["path"] == "single_finding_passthrough"])))
print("  aggregation (accepted aggregates):                      %s" % fmt(med([r["attempt"]["timings"].get("aggregation_ms") for r in direct if (r["attempt"].get("aggregation") or {}).get("accepted")])))
print("  total Plan F attempt, served:                           %s" % fmt(med([r["attempt"]["timings"].get("total_ms") for r in direct if served(r["question"])])))
print("  ACTIVE wall, served by Plan F:                          %s" % fmt(med([active[q]["wall_ms"] for q in uniq if served(q)])))
print("  ACTIVE wall, served with a handoff executed:            %s" % fmt(med([active[r["question"]]["wall_ms"] for r in ex if served(r["question"])])))
print("  ACTIVE wall, legacy fallback:                           %s" % fmt(med([active[q]["wall_ms"] for q in uniq if not served(q)])))
calls = {}
for r in direct:
    calls[r["attempt"]["specialist_calls"]] = calls.get(r["attempt"]["specialist_calls"], 0) + 1
print("  specialist calls per attempt: %s" % dict(sorted(calls.items())))

print("=" * 120)
print("9. SHADOW INVARIANCE: %d of %d identical to the direct legacy call; exceptions: %d" % (
    sum(1 for r in shadow.values() if r.get("identical_to_direct") is True), len(shadow), sum(1 for r in I6["results"] if r.get("exception"))))
llm = max([r["attempt"]["llm_calls"] for r in direct] + [r["attempt"]["dispatch_llm_calls"] for r in direct] + [int(bool(r["attempt"]["used_llm"])) for r in direct] +
          [(r.get("diagnostics") or {}).get("plan_f_llm_calls", 0) for r in I6["results"]])
print("10. PLAN F LLM CALLS (max over every attempt, dispatch and diagnostic): %d" % llm)
bad = 0
for r in I6["results"]:
    text = json.dumps(r.get("diagnostics") or {})
    answer = r.get("answer") or ""
    if r["question"] in text or (len(answer) > 20 and answer in text):
        bad += 1
keys = set()
for r in I6["results"]:
    keys |= set((r.get("diagnostics") or {}).keys())
print("11. DIAGNOSTICS HYGIENE: %d of %d records carry the question or the answer text; routing/handoff keys: %s" % (
    bad, len(I6["results"]), sorted(k for k in keys if k.startswith(("routing_", "handoff_")) or k == "specialist_calls")))

print("=" * 120)
print("12. ALL SERVED ANSWERS (ACTIVE), newly served marked NEW")
seen = set()
for r in direct:
    q = r["question"]
    if q in seen or not served(q):
        continue
    seen.add(q)
    a = r["attempt"]
    flag = "NEW" if not (r.get("i5") or {}).get("served") else "   "
    print("  %s [%3d] %-26s pages=%-14s calls=%d %s" % (flag, r["n"], a["path"], a["pages"], a["specialist_calls"], q))
