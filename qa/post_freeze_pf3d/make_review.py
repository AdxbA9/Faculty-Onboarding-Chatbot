"""Write the PF-3D manual reviews:

* ``newly_served_review.json``: every answer Plan F served from a guide in ACTIVE (any category), with a hand verdict
  (CORRECT / ACCEPTABLE BUT LIMITED / WRONG) read against the guide page: source, page, system, actor, detail, complete
  procedure and step order, no omitted step, no conflicted fact, specialist. Answers already reviewed in PF-3C keep that
  verdict only when the rendered answer is byte-identical; anything else is reviewed again here. An answer served but not
  listed gets no verdict (NOT REVIEWED) and fails the analysis.
* ``structure_recovery_review.json``: for each of the 10 STRUCTURAL false fallbacks of ``fallback_classification.json``,
  whether PF-3D recovered it, with the structure used (kind, list type, start, end, pages, units) and the hand check.
``newly_served_review.json`` is also the review input of ``analyze.py`` (its ``served`` table).

    python -X utf8 qa/post_freeze_pf3d/make_review.py <results.json> <out dir>
"""
import json
import os
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
Q = {q["id"]: q for q in R["questions"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
DIRECT = {d["id"]: d for d in R["direct"]}
PF3C = json.load(open(os.path.join(REPO, "qa", "post_freeze_pf3c", "review.json"), encoding="utf-8"))["served"]
PF3C_RESULTS = {r["id"]: r for r in json.load(open(os.path.join(REPO, "qa", "post_freeze_pf3c", "results.json"), encoding="utf-8"))["results"]
                if r["mode"] == "active"}
CLASSES = json.load(open(os.path.join(HERE, "fallback_classification.json"), encoding="utf-8"))["questions"]

# ---- hand verdicts of the answers PF-3D serves for the first time, or serves differently ----------------------------------
NEW = {
    "G08": ("CORRECT", "short guide p7: the lead-in 'With your keyboard, you can move an item ...' and steps 1-4 in order; the next "
                       "section label 'D. Edit and delete content:' is a heading, not a fifth step (D-112 lettered title)"),
    "G09": ("CORRECT", "short guide p8 'Add a Learning Module or Folder:' with every glyph-bulleted unit in order (the module "
                       "description, plus sign, Learning Module or Folder, name it, visibility note); bounded by the next heading"),
    "G10": ("CORRECT", "short guide p9 'Send an Announcement:' with all eight glyph-bulleted units in order (create Announcements, plus "
                       "sign, title, recipients and message, e-mail copy, schedule, save, Post, edit/copy/delete); page 10 opens with a heading"),
    "G11": ("CORRECT", "short guide 'Create an Assignment:' continued from page 10 onto page 11 (which opens under no heading): the "
                       "description, the two creation steps and every assignment setting, cited on both pages; complete and verbatim, long"),
    "G15": ("CORRECT", "gradebook p1: the lead-in 'You can grade coursework, manage items, and post grades from two views:' recovered "
                       "across two rows, then 1. List View (a, b) and 2. Student grid View: the two views"),
    "G17": ("CORRECT", "gradebook p8 'Assign automatic zeros:' (settings icon, the check box, Next select Assign Zeros, the result) "
                       "bounded by the short heading 'Filter by child courses'"),
    "G18": ("CORRECT", "gradebook p10 sub-section '1. Download Worflow': select the down-arrow icon, then 'Make your selections:' with "
                       "the three selections; bounded by '2. Upload a file to the gradebook'"),
    "G19": ("CORRECT", "gradebook p5 'Add accommodations from the Student list view:': set accommodations, three-dot menu, "
                       "Accommodations option, the panel opens; the section's own first sentence states 'set accommodations'"),
    "G22": ("CORRECT", "gradebook p8 'Filter by child courses': the lead-in and the three bulleted lines (select Filter, filter only "
                       "for merged courses, all child-course students appear)"),
    "P14": ("CORRECT", "two clauses each verified on its own: handbook p79 (full-time faculty: a minimum of five office hours per week over at least two days) "
                       "and the complete 'Create an Assignment' section (pages 10-11); no bridge text"),
    "C29": ("CORRECT", "the same complete keyboard procedure as G08 (PF-3C labelled this question an incomplete procedure because the "
                       "PF-2 reading took the section label 'D.' for a fifth step; D-112 reads it as the next heading)"),
}

served = {}
for qid, rec in ACTIVE.items():
    d = rec.get("diagnostics") or {}
    if not d.get("plan_f_guide_served"):
        continue
    previous = PF3C_RESULTS.get(qid) or {}
    if qid in NEW:
        verdict, note = NEW[qid]
        basis = "PF-3D review"
    elif qid in PF3C and previous.get("answer") == rec.get("answer"):
        verdict, note = PF3C[qid]["verdict"], PF3C[qid]["note"]
        basis = "PF-3C review (answer unchanged)"
    else:
        verdict, note, basis = None, "served but NOT REVIEWED", "-"
    served[qid] = {"question": Q[qid]["question"], "category": Q[qid]["category"], "expected_source": Q[qid].get("source"),
                   "answer": rec.get("answer"), "pages": rec.get("pages"), "sources": d.get("plan_f_sources"), "forms": d.get("guide_answer_forms"),
                   "parts": d.get("guide_answer_parts"), "specialist": d.get("routing_final"), "verdict": verdict, "note": note, "basis": basis,
                   "new_in_pf3d": qid not in PF3C}


def guide_path(qid):
    for rep in (DIRECT.get(qid) or {}).get("reports") or []:
        if rep.get("guide_path"):
            return rep["guide_path"]
    return {}


recovery = {}
for qid, cls in CLASSES.items():
    if cls["class"] != "STRUCTURAL":
        continue
    gp = guide_path(qid)
    clause = next(iter((gp.get("clauses") or {}).values()), {})
    entry = {"question": Q[qid]["question"], "source": cls["source"], "primary": cls["primary"], "pf3c_note": cls["note"],
             "recovered": qid in served, "form": clause.get("form"), "structure": clause.get("procedure_key"), "chunk_ids": clause.get("chunk_ids"),
             "pages": (served.get(qid) or {}).get("pages"), "verdict": (served.get(qid) or {}).get("verdict"),
             "reason_if_not": None if qid in served else [t.get("reason") for t in (clause.get("tried") or []) if t.get("reason")][:3]}
    recovery[qid] = entry

summary = {"served_guide_answers": len(served), "new_in_pf3d": sorted(q for q, s in served.items() if s["new_in_pf3d"]),
           "verdicts": {v: sum(1 for s in served.values() if (s["verdict"] or "NOT REVIEWED") == v)
                        for v in sorted({s["verdict"] or "NOT REVIEWED" for s in served.values()})},
           "structural_false_fallbacks": len(recovery), "recovered": sorted(q for q, e in recovery.items() if e["recovered"]),
           "remaining": sorted(q for q, e in recovery.items() if not e["recovered"])}
os.makedirs(OUT, exist_ok=True)
json.dump({"method": __doc__.strip(), "summary": summary, "served": served}, open(os.path.join(OUT, "newly_served_review.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
json.dump({"method": "For each STRUCTURAL false fallback of fallback_classification.json: recovered by PF-3D or not, the D-112 structure "
                     "used (key, chunk ids, pages) and the hand verdict; when not recovered, the first reasons the D-111/D-112 selection gave.",
           "summary": {k: summary[k] for k in ("structural_false_fallbacks", "recovered", "remaining")}, "questions": recovery},
          open(os.path.join(OUT, "structure_recovery_review.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
