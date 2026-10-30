"""For every answer the ACTIVE run served through Plan F (aggregate or single finding), print the question, the request clauses,
the rendered parts and, for each part, the exact production chunk it was taken from and the printed page text around it, for the
manual correctness review of every served aggregate (every clause answered, every part supported, subject, relation, value,
condition, source and page).

    python -X utf8 qa/plan_f_i5/review_accepted.py <repo> <i5_results.json>
"""
import json
import os
import sys

REPO, RESULTS = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
os.chdir(REPO)
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402

kb = build_knowledge_base(verbose=False)
data = json.load(open(RESULTS, encoding="utf-8"))
active = {r["question"]: r for r in data["results"] if r["mode"] == "active"}
page_text = {p["page"]: p["text"] for p in kb.pages}

for rec in data["direct"]:
    run = active.get(rec["question"]) or {}
    d = run.get("diagnostics") or {}
    if d.get("served") != "plan_f":
        continue
    a = rec["attempt"]
    print("=" * 118)
    print("[%3d] %s | %s | path=%s | pages=%s" % (rec["n"], rec["group"], rec["question"], a["path"], a["pages"]))
    print("  request clauses: %s" % rec["decision"]["clauses"])
    if a.get("aggregation"):
        agg = a["aggregation"]
        print("  aggregation: %s; clauses %d/%d covered; specialists %s; duplicates %s; evidence %s" % (
            agg["reason"], agg["covered_count"], agg["clause_count"], agg["specialists"], agg["duplicates"],
            [(e["finding_id"], e["page"], e["clause_indexes"]) for e in agg["evidence"]]))
        for c in agg["clauses"]:
            v = c.get("verification") or {}
            print("    clause %d <- %s report %d shared=%s findings=%s kind=%s completeness=%s %s" % (
                c["index"], c["specialist_id"], c["report_index"], c["shared_report"], c["finding_ids"], v.get("request_kind"), v.get("completeness"), v.get("detail", "")[:70]))
    print("  ANSWER AS SERVED:")
    for part in a["answer"].split("\n\n"):
        print("    | " + part)
    for item in a["items"]:
        pf = item.get("plan_f") or {}
        cid = item["idx"]
        chunk = kb.chunks[cid]
        quote = item["chunk"]
        print("  -- item: page %s, chunk %s, clause_index=%s, finding %s, derived_from=%s, slice=%s" % (
            item["page"], cid, pf.get("clause_index"), pf.get("finding_id"), pf.get("derived_from"), pf.get("slice")))
        print("     quote is an exact substring of the chunk: %s" % (quote in chunk))
        print("     CHUNK: " + chunk[:900].replace("\n", " / "))
        text = page_text.get(item["page"], "")
        needle = quote[:40]
        pos = text.find(needle)
        if pos >= 0:
            print("     PAGE %d TEXT: ...%s..." % (item["page"], text[max(0, pos - 250):pos + len(quote) + 250].replace("\n", " / ")))
        else:
            print("     PAGE %d TEXT: (quote start not found verbatim in the page text; see the chunk)" % item["page"])
