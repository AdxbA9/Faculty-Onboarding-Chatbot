"""Write ``review.json``: the manual review of every question of the PF-2 guide evaluation.

For the questions Plan F served (handbook-only controls) the verdict is CORRECT / LIMITED / WRONG on the evidence; for the
guide questions, which the frozen specialists cannot serve, the review records that no Plan F answer was served and grades
what the unscoped legacy retrieval put in its top five (the evidence a model answer would rest on): RETRIEVAL_CORRECT when the
expected document and page carry the asked statement, RETRIEVAL_PARTIAL when the right document is there but the statement is
cut or a second document is missing, RETRIEVAL_MISS when the asked page is absent. The programmatic checks come from the
results file; the verdicts and notes were written by hand after reading the quotes.

    python -X utf8 qa/post_freeze_pf2/make_review.py <results.json> <review.json>
"""
import json
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
Q = {q["id"]: q for q in R["questions"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
LEGACY = {r["id"]: r for r in R["results"] if r["mode"] == "legacy"}

# ---- hand-written verdicts (read against the top-five legacy evidence quotes) -----------------------------------------
MANUAL = {
    "G01": ("RETRIEVAL_CORRECT", "page 1 procedure window and paragraph carry the full access procedure including the off-campus Login step"),
    "G02": ("RETRIEVAL_CORRECT", "the multi-factor step (SMS or Microsoft Authenticator) is the first item"),
    "G03": ("RETRIEVAL_CORRECT", "the favourite / pin sentence is the fifth item (page 3 procedure step); a gradebook page-1 paragraph intrudes at rank 2"),
    "G04": ("RETRIEVAL_CORRECT", "page 4 paragraph and the Details & Actions row; the list items (roster, groups, image, availability, Collaborate, attendance, announcements) are on page 4"),
    "G05": ("RETRIEVAL_CORRECT", "page 4 row states 1200 x 240; the 25 KB limit is on page 5 (page-5 paragraph is in the top five)"),
    "G06": ("RETRIEVAL_CORRECT", "Course Availability: Private is the first item; a handbook page-222 row about course files intrudes at rank 3"),
    "G07": ("RETRIEVAL_CORRECT", "the Upload item with the supported file types is in the page-6 procedure window"),
    "G08": ("RETRIEVAL_CORRECT", "page 7 reorder content with the four keyboard steps (the procedure heading reads 'B. Content Visibility' because C) was parsed as a lettered step)"),
    "G09": ("RETRIEVAL_CORRECT", "page 8 learning module steps"),
    "G10": ("RETRIEVAL_CORRECT", "page 9 announcement steps including Save then Post"),
    "G11": ("RETRIEVAL_CORRECT", "page 10 Create an Assignment paragraph first; gradebook page-1 items intrude at ranks 2 and 3"),
    "G12": ("RETRIEVAL_CORRECT", "the exact sentence 'you cannot add a time limit to group assignments' is the first item"),
    "G13": ("RETRIEVAL_CORRECT", "page 5 student preview sentence first"),
    "G14": ("RETRIEVAL_CORRECT", "page 1 course gradebook location (navigation bar, top right, Gradebook icon) first"),
    "G15": ("RETRIEVAL_CORRECT", "page 1 paragraph covers the two views (list view with gradable items and students, student grid view)"),
    "G16": ("RETRIEVAL_PARTIAL", "page 6 is reached only at rank 4 and that row window shows steps 1-3; the 'Post option in the column header' step is not inside the top five quotes"),
    "G17": ("RETRIEVAL_CORRECT", "page 8 automatic zeros paragraph first, page 7 settings row third"),
    "G18": ("RETRIEVAL_CORRECT", "page 10 download workflow first; grade transfer page-1 login steps intrude at ranks 2 and 3"),
    "G19": ("RETRIEVAL_CORRECT", "gradebook page 5 accommodations at ranks 2 and 3; the short guide's page-10 accommodations paragraph is first (also a correct document for the topic)"),
    "G20": ("RETRIEVAL_CORRECT", "page 3 Add a row with Add Attendance"),
    "G21": ("RETRIEVAL_MISS", "every top-five item is a page-1 overview chunk; the 255-character limit (pages 3 and 9) is not retrieved"),
    "G22": ("RETRIEVAL_CORRECT", "page 8 filter by child courses"),
    "G23": ("RETRIEVAL_CORRECT", "page 7 Gradebook Settings panel list"),
    "G24": ("RETRIEVAL_CORRECT", "page 1 introduction first; the page-2 transfer steps (search by Course ID and Term, View Details, Transfer) are at ranks 4 and 5"),
    "G25": ("RETRIEVAL_CORRECT", "page 1 steps including the announcement and the direct link note"),
    "G26": ("RETRIEVAL_CORRECT", "page 3 Chair/Dean approval procedure"),
    "G27": ("RETRIEVAL_CORRECT", "page 2 'search by Course ID and Term' at rank 2"),
    "G28": ("RETRIEVAL_CORRECT", "page 3 Chair/Dean procedure first"),
    "G29": ("RETRIEVAL_CORRECT", "page 1 access steps and the service-desk e-mail"),
    "G30": ("RETRIEVAL_CORRECT", "page 1 upload steps"),
    "G31": ("RETRIEVAL_CONFLICT", "Plan F not served (no specialist owns the ask; the conflict guard would reject a finding on these chunks); the legacy evidence shows the English 'Maximum 30 Documents per year' rows, which carry source_conflict; the legacy path is unchanged by design and would need the same guard before a model answer is served"),
    "G32": ("RETRIEVAL_CORRECT", "'Files must be less than 100MB' at rank 4 (file requirements list)"),
    "G33": ("RETRIEVAL_CORRECT", "the service-desk e-mail row is first"),
    "G34": ("RETRIEVAL_CORRECT", "'Upload a New Revision ... not counted as extra doc' at ranks 2 and 3"),
    "G35": ("RETRIEVAL_CORRECT", "Checking Similarity Report procedure at rank 2"),
    "G36": ("RETRIEVAL_PARTIAL", "the grade transfer steps are retrieved (pages 1-3) but the handbook page-224 submission deadline is not in the top five: a cross-document answer would be one-sided"),
    "G37": ("RETRIEVAL_PARTIAL", "only iThenticate page 1 is retrieved; no handbook research-integrity statement in the top five"),
    "G38": ("RETRIEVAL_CORRECT", "ambiguous ask: the handbook page-224 grading policy is retrieved (the policy reading); the dashboard procedure is not in the top five"),
    "G39": ("WRONG_SOURCE_RISK", "no reset procedure exists; the top five are the Blackboard sign-in steps of the short guide, which a model answer could misread as a reset procedure; Plan F fell back (UNSAFE_RESULT)"),
    "G40": ("WRONG_SOURCE_RISK", "no direct Banner entry procedure exists; the top five are the dashboard transfer steps; Plan F fell back (UNSAFE_RESULT)"),
    "G41": ("RETRIEVAL_CORRECT_NO_ANSWER", "mixed handbook and iThenticate rows, none stating a Turnitin threshold; Plan F NOT_APPLICABLE"),
    "G42": ("RETRIEVAL_CORRECT_NO_ANSWER", "no appraisal content exists in the corpus (held document): handbook HR rows only; Plan F fell back"),
    "G43": ("CORRECT", "served by Teaching, page 91, identical to the I-9 reviewed answer (T01)"),
    "G44": ("CORRECT", "served by Research, page 81, identical to the I-9 reviewed answer (R01)"),
    "G45": ("CORRECT", "served by Institutional Navigation, page 54, identical to the I-9 reviewed answer (I01)"),
    "G46": ("CORRECT_FALLBACK", "the known wrong HR answer is not served (UNSAFE_RESULT), as in I-9"),
    "G47": ("CORRECT_FALLBACK", "the misleading annual-leave partial is not served (UNSAFE_RESULT), as in I-9"),
    "G48": ("CORRECT", "served by Faculty Services, page 61, identical to the I-9 reviewed answer (H01)"),
}

review = {}
for qid, q in Q.items():
    a, leg = ACTIVE.get(qid) or {}, LEGACY.get(qid) or {}
    d = a.get("diagnostics") or {}
    items = leg.get("items") or []
    verdict, note = MANUAL[qid]
    review[qid] = {
        "question": q["question"], "category": q["category"], "expected_source": q.get("source"), "expected_pages": q.get("pages"),
        "plan_f_served": d.get("served") == "plan_f", "plan_f_outcome": d.get("outcome"), "plan_f_pages": a.get("pages") if d.get("served") == "plan_f" else None,
        "plan_f_answer": a.get("answer") if d.get("served") == "plan_f" else None,
        "legacy_top5": [(i["source_id"], i["page"], i["chunk_type"]) for i in items],
        "legacy_top1_is_expected_source": bool(items) and items[0]["source_id"] == q.get("source"),
        "legacy_top5_has_expected_page": any(i["source_id"] == q.get("source") and q.get("pages") and i["page"] in q["pages"] for i in items),
        "verdict": verdict, "note": note,
    }
summary = {}
for entry in review.values():
    summary[entry["verdict"]] = summary.get(entry["verdict"], 0) + 1
json.dump({"method": __doc__.strip(), "summary": summary, "questions": review}, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
