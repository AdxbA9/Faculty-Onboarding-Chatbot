"""PF-5B derived QA records from the evaluation results and the review table (no answer text beyond 120 characters):

* ``cross_document_results.json``: every cross-document question of the four sets (P0 ``cross_document``, P1
  ``cross_document``, held-out ``cross_source``, fresh ``stress`` questions served with more than one source): served or not,
  path, sources and pages per clause, verdict;
* ``procedure_qa.json``: for every served P1, held-out or fresh procedure, section or list, its pages, part count, first and
  last unit, verbatim check and the hand verdict.

    python -X utf8 qa/post_freeze_pf5b/derive.py <results.json> <review table> <qa dir>
"""
import json
import os
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
REVIEW = json.load(open(sys.argv[2], encoding="utf-8"))["served"]
OUT = sys.argv[3]
Q = {q["id"]: q for q in R["questions"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}


def diag(rec):
    return (rec or {}).get("diagnostics") or {}


cross = []
for qid, q in Q.items():
    multi = q.get("set") == "fresh" and len(set(diag(ACTIVE.get(qid)).get("plan_f_sources") or ())) > 1
    if q["category"] not in ("cross_document", "held_out_cross_source") and not multi:
        continue
    rec, d = ACTIVE[qid], diag(ACTIVE[qid])
    items = rec.get("items") or []
    clauses = {}
    for item in items:
        clauses.setdefault(str((item.get("plan_f") or {}).get("clause_index")), []).append((item.get("source_id"), item.get("page")))
    cross.append({"id": qid, "set": q.get("set"), "question": q["question"], "served": d.get("served"), "outcome": d.get("outcome"),
                  "path": d.get("plan_f_path"), "sources": d.get("plan_f_sources"), "pages": rec.get("pages"),
                  "clauses": {k: sorted(set(v)) for k, v in clauses.items()} if d.get("served") == "plan_f" else None,
                  "verdict": (REVIEW.get(qid) or {}).get("verdict"), "note": (REVIEW.get(qid) or {}).get("note")})
summary = {"questions": len(cross), "served": sum(1 for c in cross if c["served"] == "plan_f"),
           "served_ids": [c["id"] for c in cross if c["served"] == "plan_f"],
           "wrong": [c["id"] for c in cross if c["verdict"] == "WRONG"],
           "per_set": {s: sum(1 for c in cross if c["set"] == s) for s in ("p0", "p1", "held_out", "fresh")}}
json.dump({"method": __doc__.strip(), "summary": summary, "questions": cross}, open(os.path.join(OUT, "cross_document_results.json"), "w",
          encoding="utf-8"), indent=1, ensure_ascii=False)

procedures = []
for qid, entry in REVIEW.items():
    if Q[qid].get("set") not in ("p1", "held_out", "fresh"):
        continue
    forms = entry.get("forms") or []
    if not set(forms) & {"procedure", "section", "list_item"}:
        continue
    items = ACTIVE[qid].get("items") or []
    quotes = [str(i.get("quote") or "") for i in items]
    procedures.append({"id": qid, "forms": forms, "pages": entry.get("pages"), "parts": len(items),
                       "all_quotes_verbatim_in_their_chunk": all(i.get("quote_in_chunk") for i in items),
                       "first_unit": quotes[0][:120] if quotes else None, "last_unit": quotes[-1][-120:] if quotes else None,
                       "verdict": entry.get("verdict"), "note": entry.get("note")})
json.dump({"method": "Every served P1, held-out or fresh procedure, section or list: complete start and end, no missing middle step, "
                     "printed order and hierarchy (numbered sub-sections, bullets), sub-step labels, cross-page continuity, no "
                     "screenshot-only action, no cut unit (hand review in the verdict note).", "answers": procedures},
          open(os.path.join(OUT, "procedure_qa.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({"cross_document": summary, "procedure_qa": [(p["id"], p["forms"], p["verdict"]) for p in procedures]}, indent=1))
