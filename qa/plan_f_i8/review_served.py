"""Evidence review sheet for every answer the ACTIVE run served through Plan F (sections 14 and 15 of the brief): the
question, what a correct runtime may do, the handbook fact, the Coordinator's clauses and tasks, the routing refinement,
the handoff round, the per-clause aggregation coverage, the answer exactly as served and, for every evidence item, the
citation checks (quote in chunk, chunk page, quote in page text), the production chunk and the printed page around the
quote. The manual verdicts go to ``review.json`` (id -> {"verdict": CORRECT | LIMITED | WRONG, "citation": OK | DEFECT, "note"}).

    python -X utf8 qa/plan_f_i8/review_served.py <repo> <i8_results.json> [ids...]
"""
import json
import os
import sys

REPO, RESULTS = sys.argv[1], sys.argv[2]
ONLY = set(sys.argv[3:])
sys.path.insert(0, REPO)
os.chdir(REPO)
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402

kb = build_knowledge_base(verbose=False)
data = json.load(open(RESULTS, encoding="utf-8"))
questions = {q["id"]: q for q in data["questions"]}
active = {r["id"]: r for r in data["results"] if r["mode"] == "active"}
direct = {r["id"]: r for r in data["direct"]}
page_text = {p["page"]: p["text"] for p in kb.pages}
for qid, rec in active.items():
    d = rec.get("diagnostics") or {}
    if d.get("served") != "plan_f" or (ONLY and qid not in ONLY):
        continue
    entry, drec = questions[qid], direct.get(qid) or {}
    a = drec.get("attempt") or {}
    print("=" * 118)
    print("[%s] %s | expect=%s | %s | path=%s | pages=%s | calls=%s | wall=%.0f ms" % (
        qid, entry["q"], entry["expect"], entry["category"], d.get("plan_f_path"), rec.get("pages"), d.get("specialist_calls"), rec["wall_ms"]))
    print("  fact: %s" % entry["fact"])
    print("  tags: %s" % entry["tags"])
    if drec.get("decision"):
        print("  clauses: %s" % drec["decision"]["clauses"])
        print("  coordinator tasks: %s" % drec["decision"]["tasks"])
    rt = a.get("routing") or {}
    print("  routing: %s refined=%s %s -> %s rules=%s" % (rt.get("reason"), rt.get("refined"), rt.get("initial_specialists"), rt.get("final_specialists"), rt.get("rules_applied")))
    if a.get("handoff"):
        h = a["handoff"]
        print("  handoff: %s depth=%s visited=%s outcome=%s" % (h.get("code"), h.get("depth"), h.get("visited"), a.get("handoff_outcome")))
        for r in h.get("records", []):
            print("    %s -> %s clause=%s code=%s executed=%s status=%s" % (r["source"], r["target"], r["request_clause"], r["code"], r["executed"], r["status"]))
    if a.get("aggregation"):
        agg = a["aggregation"]
        print("  aggregation: %s; clauses %s/%s; specialists %s; duplicates %s" % (agg.get("reason"), agg.get("covered_count"), agg.get("clause_count"), agg.get("specialists"), agg.get("duplicates")))
        for c in agg.get("clauses", []):
            v = c.get("verification") or {}
            print("    clause %s <- %s findings=%s kind=%s completeness=%s" % (c.get("index"), c.get("specialist_id"), c.get("finding_ids"), v.get("request_kind"), v.get("completeness")))
    elif a.get("verification"):
        v = a["verification"]
        print("  verification: %s kind=%s completeness=%s %s" % (v.get("reason"), v.get("request_kind"), v.get("completeness"), (v.get("detail") or "")[:80]))
    if a.get("answer") is not None and a.get("answer") != rec.get("answer"):
        print("  NOTE: the direct attempt's answer differs from the seam's served answer")
    print("  ANSWER AS SERVED:")
    for part in (rec.get("answer") or "").split("\n\n"):
        print("    | " + part)
    for item in rec.get("items") or []:
        pf = item.get("plan_f") or {}
        print("  -- item: page %s (%s), chunk %s, specialist=%s, clause_index=%s, finding %s, derived_from=%s" % (
            item["page"], item.get("section"), item["idx"], pf.get("specialist_id"), pf.get("clause_index"), pf.get("finding_id"), pf.get("derived_from")))
        print("     checks: quote_in_chunk=%s meta_page_is_cited_page=%s quote_in_page=%s quote_in_page_normalized=%s source=%s" % (
            item["quote_in_chunk"], item["meta_page_is_cited_page"], item["quote_in_page"], item["quote_in_page_normalized"], item.get("source")))
        chunk = item.get("chunk") or ""
        print("     CHUNK: " + chunk[:700].replace("\n", " / "))
        text = page_text.get(item["page"], "")
        pos = text.find((item.get("quote") or "")[:40])
        if pos >= 0:
            print("     PAGE %s TEXT: ...%s..." % (item["page"], text[max(0, pos - 200):pos + len(item.get("quote") or "") + 200].replace("\n", " / ")))
        else:
            print("     PAGE %s TEXT: (quote start not found verbatim in the page text)" % item["page"])
