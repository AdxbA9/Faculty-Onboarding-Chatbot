"""PF-4 manual review: every Plan F-served answer of the P1 and held-out sets (guide or handbook) and every P0 answer.

A hand verdict (CORRECT / ACCEPTABLE BUT LIMITED / WRONG) is bound to the SHA-256 (first 16 hex digits) of the exact answer
text that was read against the official page, so a verdict never carries over to a changed answer: a served answer whose
hash differs, or that has no verdict, gets none and fails ``analyze.py``. P0 answers keep the PF-3D verdict only when the
rendered answer is byte-identical to the PF-3D one.

Writes ``p1_results.json``, ``held_out_review.json``, ``held_out_results.json``, ``p0_regression.json`` and the review
table that ``analyze.py`` reads.

    python -X utf8 qa/post_freeze_pf4/make_review.py <results.json> <out dir> <review table out>
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
Q = {q["id"]: q for q in R["questions"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
DIRECT = {d["id"]: d for d in R["direct"]}

# id: (answer sha256[:16], verdict, note)
HAND = {
    "A09": ("9cd8ef8e7b0be133", "CORRECT", "attendance guide p2 tip, verbatim: an absent student with an acceptable excuse may be marked excused "
                                           "(whether an excuse is acceptable stays handbook policy)"),
    "B05": ("93c077a32be84440", "CORRECT", "Collaborate guide p2: the default attendee role for students is participants"),
    "B10": ("730a6cc093494083", "CORRECT", "Collaborate guide p4 'D- Checking Session Attendance', all three bullets in order: open the attendance tool "
                                           "from Details & Actions, select overview, export icon downloads an Excel sheet"),
    "M01": ("48d740b3eea979b2", "ACCEPTABLE BUT LIMITED", "communication guide p1: the two places to create a discussion and the plus sign/settings; "
                                                          "the New Discussion page fields (p2: title, guidelines, show/hide) are not included"),
    "M05": ("5c1a1a824a644004", "ACCEPTABLE BUT LIMITED", "the Discussion Settings options of p2 contain the Post first sentence, but the list's last "
                                                          "item is cut at the page end ('... use goals to measure'); D-113 now refuses such a list"),
    "M06": ("f026f953b9cd26c0", "CORRECT", "communication guide p3: students cannot edit their discussion titles after they create discussions"),
    "M08": ("f9c1cc39bc3d72b7", "CORRECT", "communication guide p4 'Conversations': enable (item Settings icon > Allow class conversations) and access"),
    "X03": ("b90280713536d790", "ACCEPTABLE BUT LIMITED", "two clauses, each verified on its own: the complete short-guide announcement section (p9, "
                                                          "correct) and the two discussion locations (p1, limited as in M01); no bridge text"),
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
        entry.update(verdict=HAND[qid][1], note=HAND[qid][2], basis="PF-4 review")
    elif qid in PF3D and PF3D[qid].get("answer") == answer:
        entry.update(verdict=PF3D[qid]["verdict"], note=PF3D[qid]["note"], basis="PF-3D review (answer unchanged)")
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


sets = {name: [qid for qid in Q if Q[qid].get("set") == name] for name in ("p0", "p1", "held_out")}
json.dump({"method": __doc__.strip(), "questions": [row(q) for q in sets["p1"]],
           "served": {q: table[q] for q in sets["p1"] if q in table}}, open(os.path.join(OUT, "p1_results.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
json.dump({"method": __doc__.strip(), "served": {q: table[q] for q in sets["held_out"] if q in table}},
          open(os.path.join(OUT, "held_out_review.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump({"method": "Per held-out question: what the ACTIVE runtime did (guide scopes on), with the label's answerability.",
           "questions": [row(q) for q in sets["held_out"]]}, open(os.path.join(OUT, "held_out_results.json"), "w", encoding="utf-8"), indent=1,
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
