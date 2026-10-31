"""PF-5B manual review: every Plan F-served answer of the P1, old held-out and fresh held-out sets (guide or handbook) and every
P0 answer.

A hand verdict (CORRECT / ACCEPTABLE BUT LIMITED / WRONG) is bound to the SHA-256 (first 16 hex digits) of the exact answer
text that was read against the official page, so a verdict never carries over to a changed answer: a served answer whose
hash differs, or that has no verdict, gets none and fails ``analyze.py``. P0 answers keep the PF-3D verdict, and P1 answers
the PF-4 verdict, only when the rendered answer is byte-identical.

Writes ``p1_regression.json``, ``fresh_held_out_review.json``, ``fresh_held_out_results.json``, ``p0_regression.json`` and the
review table that ``analyze.py`` reads.

    python -X utf8 qa/post_freeze_pf5b/make_review.py <results.json> <out dir> <review table out>
"""
import hashlib
import json
import os
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
TABLE = sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
PF3D = json.load(open(os.path.join(os.path.dirname(HERE), "post_freeze_pf3d", "newly_served_review.json"), encoding="utf-8"))["served"]
PF4 = json.load(open(os.path.join(os.path.dirname(HERE), "post_freeze_pf4", "p1_results.json"), encoding="utf-8"))["served"]
Q = {q["id"]: q for q in R["questions"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
DIRECT = {d["id"]: d for d in R["direct"]}

# id: (answer sha256[:16], verdict, note)
HAND = {
    "H03": ("00b6da2c249b445a", "CORRECT", "grade transfer guide p2 'How to Send Grades to Banner as instructor?', all four printed steps "
                                           "verbatim and in order (step 4 points at a screenshot); the guide never distinguishes final from "
                                           "midterm grades and its registry record declares final grades (D-114 default modifier)"),
    "G14": ("4cc1e150ed7b83d8", "ACCEPTABLE BUT LIMITED", "gradebook guide p1 'Course gradebook', verbatim: the gradebook is on the navigation bar "
                                                          "on the top right corner of the course; the next sentence (select the Gradebook icon) "
                                                          "is not included. Routed by the Coordinator to Institutional Navigation; D-114 gives it "
                                                          "to Teaching & Learning, which owns the gradebook guide"),
    "G35": ("277c678f012e3587", "ACCEPTABLE BUT LIMITED", "iThenticate guide p1 'Checking Similarity Report', verbatim: the similarity and "
                                                          "originality report is under 'Documents'; the next bullet (click the report percentage) "
                                                          "is not included. Routed by the Coordinator to Institutional Navigation; D-114 gives it "
                                                          "to Research & Innovation, which owns the iThenticate guide"),
}


def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def diag(rec):
    return (rec or {}).get("diagnostics") or {}


table = {}
for qid, rec in ACTIVE.items():
    d = diag(rec)
    if d.get("served") != "plan_f":
        continue
    answer, digest = rec.get("answer"), sha(rec.get("answer"))
    entry = {"set": Q[qid].get("set"), "question": Q[qid]["question"], "category": Q[qid]["category"], "answer_sha": digest, "pages": rec.get("pages"),
             "sources": d.get("plan_f_sources"), "forms": d.get("guide_answer_forms"), "specialist": d.get("routing_final"), "verdict": None,
             "note": "served but NOT REVIEWED", "basis": "-"}
    if qid in HAND and HAND[qid][0] == digest:
        entry.update(verdict=HAND[qid][1], note=HAND[qid][2], basis="PF-5B review")
    elif qid in PF3D and PF3D[qid].get("answer") == answer:
        entry.update(verdict=PF3D[qid]["verdict"], note=PF3D[qid]["note"], basis="PF-3D review (answer unchanged)")
    elif qid in PF4 and PF4[qid].get("answer") == answer:
        entry.update(verdict=PF4[qid]["verdict"], note=PF4[qid]["note"], basis="PF-4 review (answer unchanged)")
    elif Q[qid].get("set") == "p0" and not d.get("plan_f_guide_served"):
        entry.update(verdict="NOT IN SCOPE", note="handbook answer of the P0 set (I-9 path, unchanged; not a guide answer)", basis="-")
    entry["answer"] = answer
    table[qid] = entry
os.makedirs(OUT, exist_ok=True)
json.dump({"served": {k: v for k, v in table.items() if v["verdict"] != "NOT IN SCOPE"}}, open(TABLE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def row(qid):
    rec, d = ACTIVE.get(qid) or {}, diag(ACTIVE.get(qid))
    return {"id": qid, "question": Q[qid]["question"], "category": Q[qid]["category"], "expected_source": Q[qid].get("source"),
            "answerable": Q[qid].get("answerable"), "plan_f_should_serve": Q[qid].get("plan_f_should_serve", Q[qid].get("served_expected")),
            "served": d.get("served"), "outcome": d.get("outcome"), "guide_served": bool(d.get("plan_f_guide_served")), "pages": rec.get("pages"),
            "sources": d.get("plan_f_sources"), "forms": d.get("guide_answer_forms"), "verdict": (table.get(qid) or {}).get("verdict")}


sets = {name: [qid for qid in Q if Q[qid].get("set") == name] for name in ("p0", "p1", "held_out", "fresh")}
p1_changed = [q for q in sets["p1"] if q in table and table[q]["basis"] != "PF-4 review (answer unchanged)"]
p1_lost = [q for q in PF4 if q not in table]
json.dump({"method": "P1 regression: the PF-4 P1 evaluation rerun with D-114. A served answer keeps its PF-4 verdict only when it is "
                     "byte-identical; every other served answer is reviewed by hand (verdict basis 'PF-5B review').",
           "served": len([q for q in sets["p1"] if q in table]), "changed_or_new": p1_changed, "lost": p1_lost,
           "questions": [row(q) for q in sets["p1"]], "answers": {q: table[q] for q in sets["p1"] if q in table}},
          open(os.path.join(OUT, "p1_regression.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump({"method": __doc__.strip(), "served": {q: table[q] for q in sets["fresh"] if q in table}},
          open(os.path.join(OUT, "fresh_held_out_review.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump({"method": "Per fresh held-out question: what the ACTIVE runtime did (guide scopes on), with the label's answerability and Plan F "
                     "expectation, the hand verdict of a served answer and, for a fallback, the layer that stopped it (metrics.json rows).",
           "questions": [row(q) for q in sets["fresh"]]}, open(os.path.join(OUT, "fresh_held_out_results.json"), "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)
p0_changed = []
p0_guide = [q for q in sets["p0"] if diag(ACTIVE.get(q)).get("plan_f_guide_served")]
for q in sets["p0"]:
    entry = table.get(q)
    if entry and entry["set"] == "p0" and entry["basis"] not in ("PF-3D review (answer unchanged)", "-"):
        p0_changed.append(q)
json.dump({"method": "P0 regression: the PF-3D guide evaluation rerun on the eight-source corpus. A guide answer keeps its PF-3D verdict only "
                     "when it is byte-identical; anything else is listed in 'changed_guide_answers' and must be reviewed.",
           "guide_served": len(p0_guide), "guide_served_ids": p0_guide,
           "carried_verdicts": sum(1 for q in p0_guide if (table.get(q) or {}).get("basis") == "PF-3D review (answer unchanged)"),
           "changed_guide_answers": p0_changed,
           "not_reviewed": [q for q in p0_guide if (table.get(q) or {}).get("verdict") is None]},
          open(os.path.join(OUT, "p0_regression.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({"served_total": len(table), "per_set": {s: sum(1 for q in ids if q in table) for s, ids in sets.items()},
                  "not_reviewed": sorted(q for q, v in table.items() if v["verdict"] is None)}, indent=1))
