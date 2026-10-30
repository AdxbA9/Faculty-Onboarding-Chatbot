"""Analyse i3_results.json: outcome distribution, rates, latency statistics, shadow invariance, accepted answers for review."""
import json, statistics, sys
from collections import Counter, defaultdict
R = json.load(open(sys.argv[1], encoding="utf-8"))
res = R["results"]
kb = R["kb"]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    return "n=%d median=%.0f min=%.0f max=%.0f ms" % (len(values), statistics.median(values), min(values), max(values))


print("=== KB ===")
print({k: kb[k] for k in ("source_id", "registered", "pdf_name", "pages", "chunks", "embedding_shape", "embedding_dtype", "index_type", "index_ntotal",
                          "cache_version", "build1_s", "build2_s", "second_build_embeddings_identical", "groq_client_configured", "model_download_load_s", "chunk_types")})
print("section records:", kb["stats"].get("source", {}).get("section_records"), "| cache files before/after:", kb["cache_files_before_first_build"], kb["cache_files_after_first_build"])
print("\n=== builders ===")
for k, v in R["builders"].items():
    print("  %-16s ok=%s ms=%.0f shared_pages=%s unresolved=%s" % (k, v.get("ok"), v.get("ms", 0), v.get("shared_pages"), v.get("unresolved_pages")))
print("registry build_runtime ms: %.0f | %s" % (R["registry_ms"], R["registry_build"]))
print("\n=== cache ===")
print(json.dumps(R["cache"], indent=1))

by_mode = defaultdict(list)
for r in res:
    by_mode[r["mode"]].append(r)
legacy_wall = {r["question"]: r["wall_ms"] for r in by_mode["legacy"]}

print("\n=== LEGACY: identity with direct call ===")
print("identical_to_direct:", Counter(r.get("identical_to_direct") for r in by_mode["legacy"]), "| trace equal:", Counter(r.get("direct_trace_equal") for r in by_mode["legacy"]))
print("exceptions:", [r["question"] for r in by_mode["legacy"] if r.get("exception")])
print("legacy wall:", stats([r["wall_ms"] for r in by_mode["legacy"]]))
print("legacy llm_calls:", Counter(r.get("llm_calls") for r in by_mode["legacy"]), "answers starting with the no-key reply:",
      sum(1 for r in by_mode["legacy"] if str(r.get("answer", "")).startswith("GROQ_API_KEY is not set")))
for r in by_mode["legacy"]:
    if r.get("identical_to_direct") is False:
        print("  DIFF:", r["question"])

print("\n=== SHADOW: invariance and outcomes ===")
sh = by_mode["shadow"]
print("identical_to_direct:", Counter(r.get("identical_to_direct") for r in sh), "| trace equal:", Counter(r.get("direct_trace_equal") for r in sh))
print("exceptions:", [r["question"] for r in sh if r.get("exception")])
print("served:", Counter((r.get("diagnostics") or {}).get("served") for r in sh))
print("outcomes:", Counter((r.get("diagnostics") or {}).get("outcome") for r in sh))
print("plan_f_llm_calls:", Counter((r.get("diagnostics") or {}).get("plan_f_llm_calls") for r in sh), "plan_f_used_llm:", Counter((r.get("diagnostics") or {}).get("plan_f_used_llm") for r in sh))
print("registry reused:", Counter((r.get("diagnostics") or {}).get("registry_reused") for r in sh))
print("shadow wall:", stats([r["wall_ms"] for r in sh]))
overhead = [r["wall_ms"] - legacy_wall.get(r["question"], 0) for r in sh if r["question"] in legacy_wall]
print("shadow overhead vs legacy wall (same question):", stats(overhead))
print("plan_f total_ms:", stats([(r.get("diagnostics") or {}).get("plan_f_ms") for r in sh]))
for key in ("coordinator_ms", "dispatch_ms", "specialist_ms", "guards_ms", "verify_ms"):
    print("  plan_f %-14s %s" % (key, stats([((r.get("diagnostics") or {}).get("plan_f_timings") or {}).get(key) for r in sh])))
print("first-question registry build ms:", [(r.get("diagnostics") or {}).get("registry_build_ms") for r in sh[:1]])

print("\n=== ACTIVE ===")
ac = by_mode["active"]
print("served:", Counter((r.get("diagnostics") or {}).get("served") for r in ac))
print("outcomes:", Counter((r.get("diagnostics") or {}).get("outcome") for r in ac))
print("exceptions:", [r["question"] for r in ac if r.get("exception")])
acc = [r for r in ac if (r.get("diagnostics") or {}).get("served") == "plan_f"]
fb = [r for r in ac if (r.get("diagnostics") or {}).get("served") == "legacy"]
print("active accepted wall:", stats([r["wall_ms"] for r in acc]))
print("active fallback wall:", stats([r["wall_ms"] for r in fb]))
print("accepted used_llm/llm_calls:", Counter((r.get("used_llm"), r.get("llm_calls")) for r in acc))

print("\n=== OUTCOME DISTRIBUTION (shadow run, 1 attempt per question) ===")
d = [r.get("diagnostics") or {} for r in sh]
total = len(sh)
attempted = sum(1 for x in d if x.get("attempted"))
accepted = sum(1 for x in d if x.get("accepted"))
outcomes = Counter(x.get("outcome") for x in d)
reasons = Counter()
for x in d:
    o = x.get("outcome"); fr = x.get("fallback_reason") or ""
    if o == "PLAN_F_UNSAFE_RESULT":
        if fr.startswith("status partial"):
            reasons["partial"] += 1
        elif fr.startswith("multiple specialists") or fr.startswith("synthesis required") or fr.startswith("multipart"):
            reasons["multipart/cross-domain"] += 1
        elif fr.startswith("handoff pending"):
            reasons["handoff"] += 1
        elif fr.startswith("ask shape general"):
            reasons["general ask"] += 1
        elif fr.startswith("status not_found") or fr.startswith("status out_of_scope"):
            reasons["not found / out of scope"] += 1
        else:
            reasons["other unsafe: " + fr[:40]] += 1
print("TOTAL %d | attempted %d | accepted %d (%.0f%%) | fallback %d (%.0f%%)" % (total, attempted, accepted, 100.0 * accepted / total, total - accepted, 100.0 * (total - accepted) / total))
print("outcomes:", dict(outcomes))
print("unsafe breakdown:", dict(reasons))
print("guard rejections: quote %d subject %d | verifier %d | errors %d | registry unavailable %d | requires_fallback %d" % (
    outcomes.get("PLAN_F_REJECTED_BY_QUOTE_GUARD", 0), outcomes.get("PLAN_F_REJECTED_BY_SUBJECT_GUARD", 0), outcomes.get("PLAN_F_REJECTED_BY_VERIFIER", 0),
    outcomes.get("PLAN_F_ERROR_REDACTED", 0), outcomes.get("PLAN_F_REGISTRY_UNAVAILABLE", 0), outcomes.get("PLAN_F_REQUIRES_FALLBACK", 0)))
print("\nper domain (shadow):")
for dom in ("teaching", "research", "hr", "institutional", "special"):
    rows = [r for r in sh if r["domain"] == dom]
    c = Counter((r.get("diagnostics") or {}).get("outcome") for r in rows)
    a = sum(1 for r in rows if (r.get("diagnostics") or {}).get("accepted"))
    print("  %-13s n=%2d accepted=%2d  %s" % (dom, len(rows), a, dict(c)))

print("\n=== ACCEPTED ANSWERS (active run) FOR REVIEW ===")
for r in acc:
    dg = r["diagnostics"]
    print("\n[%s] %s" % (r["domain"], r["question"]))
    print("   specialist=%s pages=%s best_section=%r query_type=%s wall=%.0f ms plan_f_ms=%.0f" % (dg.get("specialist_id"), r["pages"], r["best_section"], r["query_type"], r["wall_ms"], dg.get("plan_f_ms") or 0))
    print("   answer: %r" % r["answer"])
    for it in r["items"]:
        print("   item idx=%s page=%s" % (it["idx"], it["page"]))

print("\n=== KNOWN CASES ===")
for q in ("What does Human Resources do?", "What does HR do?", "How many days of annual leave do faculty members get?"):
    for mode in ("shadow", "active"):
        r = next((x for x in by_mode[mode] if x["question"] == q), None)
        if r:
            dg = r.get("diagnostics") or {}
            print("  %-6s %-55s served=%s outcome=%s specialist=%s status=%s reason=%s | answer starts: %r" % (
                mode, q, dg.get("served"), dg.get("outcome"), dg.get("specialist_id"), dg.get("status"), (dg.get("fallback_reason") or "")[:60], str(r.get("answer"))[:50]))

print("\n=== SPECIAL CASES (shadow) ===")
for r in [x for x in sh if x["domain"] == "special"]:
    dg = r.get("diagnostics") or {}
    print("  %-75s outcome=%-28s reason=%s" % (r["question"][:75], dg.get("outcome"), (dg.get("fallback_reason") or "")[:70]))

print("\n=== DIAGNOSTICS HYGIENE ===")
bad = 0
for r in sh + ac:
    dg = r.get("diagnostics") or {}
    text = json.dumps(dg)
    if r["question"] in text or (r.get("answer") and len(r["answer"]) > 20 and r["answer"] in text):
        bad += 1
print("records containing the question or answer text:", bad, "of", len(sh) + len(ac))
print("total script s:", R.get("total_script_s"))
