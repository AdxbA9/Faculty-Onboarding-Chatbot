"""For every answer the ACTIVE run served through Plan F, print the question, the Coordinator's clauses and tasks, the routing
refinement, the handoff round (if any), the rendered parts and, for each part, the exact production chunk and the printed page
text around it, for the manual correctness review of every newly served answer (every clause answered, specialist assignment
and handoff justified, entity, policy, value, source and page correct, no omitted clause, no connective text).

    python -X utf8 qa/plan_f_i6/review_accepted.py <repo> <i6_results.json> [--new-only]
"""
import json
import os
import sys

REPO, RESULTS = sys.argv[1], sys.argv[2]
NEW_ONLY = "--new-only" in sys.argv[3:]
sys.path.insert(0, REPO)
os.chdir(REPO)
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402

kb = build_knowledge_base(verbose=False)
data = json.load(open(RESULTS, encoding="utf-8"))
active = {r["question"]: r for r in data["results"] if r["mode"] == "active"}
page_text = {p["page"]: p["text"] for p in kb.pages}
seen = set()
for rec in data["direct"]:
    q = rec["question"]
    d = (active.get(q) or {}).get("diagnostics") or {}
    if d.get("served") != "plan_f" or q in seen:
        continue
    seen.add(q)
    new = not ((rec.get("i5") or {}).get("served"))
    if NEW_ONLY and not new:
        continue
    a = rec["attempt"]
    print("=" * 118)
    print("[%3d] %s | %s | %s | path=%s | pages=%s | specialist calls=%d" % (rec["n"], "NEW" if new else "kept", rec["group"], q, a["path"], a["pages"], a["specialist_calls"]))
    print("  clauses: %s" % rec["decision"]["clauses"])
    print("  coordinator tasks: %s" % rec["decision"]["tasks"])
    rt = a["routing"]
    print("  routing: %s refined=%s %s -> %s rules=%s" % (rt["reason"], rt["refined"], rt["initial_specialists"], rt["final_specialists"], rt["rules_applied"]))
    if a.get("handoff"):
        h = a["handoff"]
        print("  handoff: %s depth=%d visited=%s outcome=%s" % (h["code"], h["depth"], h["visited"], a.get("handoff_outcome")))
        for r in h["records"]:
            print("    %s -> %s clause=%s code=%s executed=%s status=%s" % (r["source"], r["target"], r["request_clause"], r["code"], r["executed"], r["status"]))
    if a.get("aggregation"):
        agg = a["aggregation"]
        print("  aggregation: %s; clauses %d/%d; specialists %s; duplicates %s" % (agg["reason"], agg["covered_count"], agg["clause_count"], agg["specialists"], agg["duplicates"]))
        for c in agg["clauses"]:
            v = c.get("verification") or {}
            print("    clause %d <- %s task %s report %d findings=%s kind=%s completeness=%s" % (
                c["index"], c["specialist_id"], c["task_id"], c["report_index"], c["finding_ids"], v.get("request_kind"), v.get("completeness")))
    elif a.get("verification"):
        v = a["verification"]
        print("  verification: %s kind=%s completeness=%s %s" % (v["reason"], v["request_kind"], v["completeness"], v["detail"][:80]))
    print("  ANSWER AS SERVED:")
    for part in a["answer"].split("\n\n"):
        print("    | " + part)
    for item in a["items"]:
        pf = item.get("plan_f") or {}
        cid = item["idx"]
        chunk = kb.chunks[cid]
        quote = item["chunk"]
        print("  -- item: page %s, chunk %s, specialist=%s, clause_index=%s, finding %s, derived_from=%s" % (
            item["page"], cid, pf.get("specialist_id"), pf.get("clause_index"), pf.get("finding_id"), pf.get("derived_from")))
        print("     quote is an exact substring of the chunk: %s" % (quote in chunk))
        print("     CHUNK: " + chunk[:900].replace("\n", " / "))
        text = page_text.get(item["page"], "")
        pos = text.find(quote[:40])
        if pos >= 0:
            print("     PAGE %d TEXT: ...%s..." % (item["page"], text[max(0, pos - 250):pos + len(quote) + 250].replace("\n", " / ")))
        else:
            print("     PAGE %d TEXT: (quote start not found verbatim in the page text; see the chunk)" % item["page"])
