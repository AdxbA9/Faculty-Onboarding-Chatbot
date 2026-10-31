"""Write ``review.json``: the PF-3C manual reviews.

1. NEWLY SERVED: every answer Plan F served from a guide in ACTIVE (any question category), read against the guide page:
   correct source, page, system, actor, detail, complete procedure and step order, no omitted step, no conflicted fact,
   correct specialist. Verdict CORRECT / ACCEPTABLE BUT LIMITED / WRONG (hand-written below; an answer served but not
   listed here is reported as NOT REVIEWED and fails the analysis).
2. EVIDENCE: the PF-3B top-evidence review repeated on the PF-3C selection (``evidence_selection.py --pf3c``): for each
   guide question on which the D-111 path selected evidence, ON_TARGET / PARTIAL / OFF_TARGET by hand, next to the PF-3B
   review of the frozen specialists' first guide quote.

    python -X utf8 qa/post_freeze_pf3c/make_review.py <results.json> <evidence_after.json> <review.json>
"""
import json
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
EV = json.load(open(sys.argv[2], encoding="utf-8"))
OUT = sys.argv[3]
Q = {q["id"]: q for q in R["questions"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
HANDBOOK = "uos_faculty_handbook_2025_26"

# ---- 1. hand verdicts of every served guide answer ---------------------------------------------------------------------
SERVED = {
    "G01": ("CORRECT", "short guide p1 'How to access Blackboard': all seven bullets in order, including the outside-campus Login step and "
                       "multi-factor verification; Teaching; the faculty actor bound by the guide's audience"),
    "G03": ("CORRECT", "short guide p3 'Your Courses', the one bullet that answers: mark the course as a favourite to pin it on top of the list "
                       "(an unordered list item answering alone)"),
    "G05": ("CORRECT", "short guide p4 Course Image item: the banner image must be at least 1200 x 240 pixels (the asked minimum); p5 also "
                       "states a 25 KB maximum, not asked"),
    "G12": ("CORRECT", "short guide p11: 'At this time, you cannot add a time limit to group assignments.' (a direct no)"),
    "G23": ("CORRECT", "gradebook p7 'From the Gradebook Settings panel, you can:' with all four items (grading schema, automatic zeros, "
                       "categories, rubrics): the complete list"),
    "G24": ("CORRECT", "grade transfer p2 'How to Send Grades to Banner as instructor?': steps 1-4 in order; step 4 completed from the page "
                       "prose ('... after the approval from Chair and Dean as below:'); send = transfer as the guide itself says"),
    "G25": ("CORRECT", "grade transfer p1 note: the direct link https://op.sharjah.ac.ae/GradeTransfer (the location asked); the "
                       "announcement route is in the steps, not needed for 'where is the link'"),
    "G28": ("CORRECT", "grade transfer p3 'How to approve Grades to Banner as Chair/Dean?': steps 1-4 in order; the Dean is named by the "
                       "heading; this is the dashboard workflow, not a statement of who has authority"),
    "G29": ("CORRECT", "iThenticate p1 'How to access IThenticate': steps 1-3 with the service-desk note; Research"),
    "G30": ("CORRECT", "iThenticate p1 'How to upload a document': steps 1-4 with the revision step and the note that the analysis may "
                       "take a few hours; Research"),
    "P11": ("CORRECT", "gradebook p6 student grid item 4: 'When you're ready to release grades to students, select the Post option in "
                       "the column header.' (the control asked); a selection ask may quote one item"),
    "P16": ("CORRECT", "two clauses each verified on its own: handbook p81 (four theses) and the iThenticate p1 access procedure; no "
                       "bridge text; Research"),
    "C01": ("CORRECT", "the same complete p2 send-grades procedure as G24 (the dashboard name bound by the registry)"),
    "C04": ("ACCEPTABLE BUT LIMITED", "grade transfer p1 note: the direct link is a correct place to access the dashboard; the other route "
                                      "(the announcement on the Blackboard Institution Page) is not shown"),
    "C05": ("CORRECT", "grade transfer p1 'To access the Grade Transfer link follow these steps': all five steps in order"),
    "C06": ("CORRECT", "the complete p1 Blackboard access procedure"),
    "C07": ("CORRECT", "gradebook list 'In the gradable items list, you can:' items 1-5 (page 2) and item 6 with a), b), c) (page 3): the "
                       "numbered list continued across the page break, merged and cited on both pages"),
    "C09": ("CORRECT", "the p3 favourite bullet (an unordered list item answering alone)"),
    "C11": ("CORRECT", "iThenticate p1 'Checking Similarity Report': the whole item (under Documents, click the report percentage, view "
                       "and exclude resources); printed without sentence breaks in the guide, quoted verbatim"),
    "C15": ("CORRECT", "two clauses each verified on its own: handbook p79 (full-time faculty: at least five office hours per week over "
                       "two days) and the complete p1 Blackboard access procedure; no bridge text"),
    "C16": ("CORRECT", "two clauses each verified on its own: handbook p96 and p91 (12 credit hours per semester; the two statements agree) "
                       "and the complete p2 send-grades procedure; the guide does not answer the workload clause"),
    "C17": ("CORRECT", "two clauses each verified on its own: handbook p81 (four theses) and the complete iThenticate p1 upload procedure"),
}

# ---- 2. hand review of the evidence the D-111 path selected -------------------------------------------------------------
EVIDENCE = {
    "G01": ("ON_TARGET", "the whole p1 access procedure, outside-campus step included"),
    "G03": ("ON_TARGET", "the p3 favourite / pin bullet"),
    "G05": ("ON_TARGET", "the p4 Course Image item with 'at least 1200 x 240 pixels'"),
    "G12": ("ON_TARGET", "the p11 sentence 'you cannot add a time limit to group assignments'"),
    "G14": ("ON_TARGET", "gradebook p1 step 2: the course gradebook on the navigation bar, top right (not served at runtime: the question "
                         "routes to Institutional Navigation, which D-110 does not approve for the gradebook guide)"),
    "G23": ("ON_TARGET", "the complete p7 Gradebook Settings list"),
    "G24": ("ON_TARGET", "the p2 send-grades procedure, step 4 completed"),
    "G25": ("ON_TARGET", "the p1 direct-link note"),
    "G28": ("ON_TARGET", "the p3 Chair/Dean approval procedure"),
    "G29": ("ON_TARGET", "the p1 access procedure"),
    "G30": ("ON_TARGET", "the p1 upload procedure"),
    "G35": ("ON_TARGET", "the p1 Checking Similarity Report item: under Documents, click the percentage (not served at runtime: routed to "
                         "Institutional Navigation, not approved for iThenticate)"),
}

PF3B_EVIDENCE = json.load(open("qa/post_freeze_pf3b/review.json", encoding="utf-8"))["summary"]["guide_evidence_assessment"]

served = {}
for qid, rec in ACTIVE.items():
    d = rec.get("diagnostics") or {}
    if not d.get("plan_f_guide_served"):
        continue
    verdict, note = SERVED.get(qid, ("NOT REVIEWED", "served but not reviewed"))
    served[qid] = {"question": Q[qid]["question"], "category": Q[qid]["category"], "expected_source": Q[qid].get("source"),
                   "expected_pages": Q[qid].get("pages"), "answer": rec.get("answer"), "pages": rec.get("pages"),
                   "sources": d.get("plan_f_sources"), "forms": d.get("guide_answer_forms"), "parts": d.get("guide_answer_parts"),
                   "specialist": d.get("routing_final"), "verdict": verdict, "note": note}

evidence = {}
for row in EV["rows"]:
    sel = row.get("pf3c") or {}
    if not sel.get("evidence"):
        continue
    assessment, note = EVIDENCE.get(row["id"], ("NOT REVIEWED", ""))
    evidence[row["id"]] = {"form": sel.get("form"), "intent": sel.get("intent"), "automatic": row.get("pf3c_top1"),
                           "pages": sorted({e["page"] for e in sel["evidence"]}), "chunk_ids": [e["chunk_id"] for e in sel["evidence"]],
                           "assessment": assessment, "note": note}

counts = {k: sum(1 for e in evidence.values() if e["assessment"] == k) for k in ("ON_TARGET", "PARTIAL", "OFF_TARGET")}
summary = {"served_guide_answers": len(served),
           "served_verdicts": {v: sum(1 for s in served.values() if s["verdict"] == v) for v in sorted({s["verdict"] for s in served.values()})},
           "evidence_before_pf3b": {"questions_with_guide_evidence": sum(PF3B_EVIDENCE.values()), **PF3B_EVIDENCE},
           "evidence_after_pf3c": {"questions_with_guide_evidence": len(evidence), **counts,
                                   "no_evidence_selected": EV["summary"]["questions"] - len(evidence)}}
json.dump({"method": __doc__.strip(), "summary": summary, "served": served, "evidence": evidence}, open(OUT, "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)
print(json.dumps(summary, indent=1))
