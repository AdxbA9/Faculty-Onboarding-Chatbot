"""Write ``review.json``: the manual review of the PF-3B evaluation.

No guide answer was served by Plan F, so the "newly served" review is empty. For every question the review records the
Plan F outcome, the layer that stopped it, and, where the reopened specialists returned guide evidence, a hand-written
assessment of that evidence (ON_TARGET: the quote states the asked fact or procedure; PARTIAL: right page or heading, not
the answer; OFF_TARGET: a different topic, typically the gradebook guide's page-1 introduction chosen because it names
"Blackboard Ultra"). The assessment answers the question the next decision record must face: would relaxing the frozen
grounding rules have served correct answers, or wrong ones?

    python -X utf8 qa/post_freeze_pf3b/make_review.py <results.json> <review.json>
"""
import json
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
Q = {q["id"]: q for q in R["questions"]}
DIRECT = {d["id"]: d for d in R["direct"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
HANDBOOK = "uos_faculty_handbook_2025_26"

# hand-written: the first guide finding the reopened specialist returned, read against the guide page it should answer from
EVIDENCE = {
    "G01": ("ON_TARGET", "page 1 access procedure window (website, MyUOS, Login, off-campus login)"),
    "G02": ("ON_TARGET", "page 1 multi-factor step (SMS or Microsoft Authenticator)"),
    "G03": ("OFF_TARGET", "gradebook page-1 introduction; the favourite/pin sentence (short guide page 3) not chosen"),
    "G04": ("OFF_TARGET", "gradebook page-1 introduction first; the Private availability sentence second; the Details & Actions list not quoted"),
    "G05": ("OFF_TARGET", "page-4 groups sentence and page-5 student preview sentence; the 1200 x 240 / 25 KB sentence not chosen"),
    "G06": ("ON_TARGET", "page 4 Private availability sentence"),
    "G07": ("ON_TARGET", "page 6 procedure window with the Upload item and the supported file types"),
    "G09": ("OFF_TARGET", "gradebook introduction and the access procedure; the learning-module steps (page 8) not chosen"),
    "G10": ("OFF_TARGET", "the short guide's title paragraph; the announcement steps (page 9) not chosen"),
    "G11": ("OFF_TARGET", "gradebook introduction and the access procedure; the assignment steps (page 10) not chosen"),
    "G13": ("OFF_TARGET", "gradebook introduction; the student preview sentence (page 5) not chosen"),
    "G15": ("PARTIAL", "gradebook page-1 paragraph; the two views are on that page but the quote opens with the introduction"),
    "G16": ("OFF_TARGET", "gradebook introduction and the grade transfer introduction; the Post step (page 6) not chosen"),
    "G18": ("OFF_TARGET", "the grade transfer login step; the download workflow (page 10) not chosen"),
    "G19": ("OFF_TARGET", "gradebook introduction; the accommodations steps (pages 4-5) not chosen"),
    "G20": ("ON_TARGET", "page 3 Add Attendance item"),
    "G21": ("OFF_TARGET", "gradebook introduction; the 255-character limit (pages 3 and 9) not chosen"),
    "G22": ("OFF_TARGET", "gradebook introduction; the child-course filter (page 8) not chosen"),
    "G23": ("OFF_TARGET", "gradebook introduction; the settings panel list (page 7) not chosen"),
    "G24": ("PARTIAL", "the guide introduction and the page-2 heading 'How to Send Grades to Banner as instructor?' without its steps"),
    "G26": ("PARTIAL", "the page-3 heading 'How to approve Grades to Banner as Chair/Dean?' (names Chair and Dean, no statement)"),
    "G28": ("PARTIAL", "the page-3 heading without the approval steps"),
    "G31": ("OFF_TARGET", "the access procedure; the disputed quota row is conflict-stamped and would be blocked in any case"),
    "G36": ("PARTIAL", "grade transfer headings for the procedure clause; the handbook deadline clause unsupported"),
}

review = {}
for qid, q in Q.items():
    d, a = DIRECT.get(qid) or {}, ACTIVE.get(qid) or {}
    diag = a.get("diagnostics") or {}
    served = diag.get("served") == "plan_f"
    guide_findings = [f for r in (d.get("reports") or []) for f in (r.get("findings") or []) if f[0] != HANDBOOK]
    if served:
        verdict = "CORRECT" if q["category"] in ("handbook_only",) else "REVIEW"
        note = "handbook answer identical to the I-9 reviewed answer" if verdict == "CORRECT" else "served: review by hand"
    elif q["category"] in ("guide", "cross_document"):
        verdict, note = "NOT SERVED", "safe fallback; no Plan F guide answer"
    else:
        verdict, note = "SAFE FALLBACK", "control question: not served, as required"
    entry = {"question": q["question"], "category": q["category"], "expected_source": q.get("source"), "expected_pages": q.get("pages"),
             "plan_f_outcome": d.get("outcome"), "routing_final": d.get("routing_final"), "guide_scope": d.get("guide_scope"),
             "served": served, "answer": a.get("answer") if served else None, "pages": a.get("pages") if served else None,
             "served_sources": diag.get("plan_f_sources") if served else None, "verdict": verdict, "note": note,
             "guide_findings_returned": [{"source_id": f[0], "page": f[1], "quote": f[2]} for f in guide_findings[:3]]}
    if qid in EVIDENCE:
        entry["guide_evidence_assessment"], entry["guide_evidence_note"] = EVIDENCE[qid]
    review[qid] = entry

guide_ids = [qid for qid, q in Q.items() if qid.startswith("G") and q["category"] in ("guide", "cross_document")]
assessed = [EVIDENCE[qid][0] for qid in guide_ids if qid in EVIDENCE]
summary = {"newly_served_guide_answers": [qid for qid in guide_ids if review[qid]["served"]],
           "verdicts": {v: sum(1 for e in review.values() if e["verdict"] == v) for v in sorted({e["verdict"] for e in review.values()})},
           "guide_evidence_returned": len(assessed),
           "guide_evidence_assessment": {k: assessed.count(k) for k in ("ON_TARGET", "PARTIAL", "OFF_TARGET")}}
json.dump({"method": __doc__.strip(), "summary": summary, "questions": review}, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
