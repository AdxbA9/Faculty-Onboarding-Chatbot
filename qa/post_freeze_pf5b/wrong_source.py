"""PF-5B wrong-source adversarial check with the REAL models on the eight-source knowledge base (derived from
``qa/post_freeze_pf4/wrong_source.py``).

The 16 PF-2 cases and the 20 PF-4 cases (unchanged) plus 24 PF-5B cases on what D-114 changes: misspelled system names
(typo collisions), software versus office "where", the verb "collaborate" versus the product, the attendance policy versus
the tool, a grade versus the Gradebook, a submission versus the Grade Transfer Dashboard, a research journal versus the
Blackboard journal tool. For each case:
  * RETRIEVAL: the unscoped legacy ranking (dense + lexical candidates, production cross-encoder): is the first hit from an
    expected source, is an expected page in the top five;
  * PLAN F: the guide-enabled runtime (``build_runtime(kb, guide_scopes=True)``) attempts the question; a served answer
    must rest only on expected sources (a wrong-source acceptance fails the check). ``must_not_serve`` cases (a conflict
    or a policy limit) must not be served from a guide at all.
Writes ``wrong_source_results.json``.

    python -X utf8 qa/post_freeze_pf5b/wrong_source.py <repo> <out json>
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
os.chdir(REPO)
sys.path.insert(0, REPO)

from handbook_bot.knowledge_base import build_knowledge_base           # noqa: E402
from handbook_bot.retrieval import classify_query, gather_candidates   # noqa: E402
from handbook_bot import plan_f_runtime as PF                          # noqa: E402

HB, SG, GB, GT, IT = ("uos_faculty_handbook_2025_26", "uos_blackboard_faculty_short_guide", "uos_blackboard_gradebook_guide",
                      "uos_grade_transfer_dashboard_guide", "uos_ithenticate_guidelines")
AT, CO, CM = "uos_blackboard_attendance_guide", "uos_blackboard_collaborate_guide", "uos_blackboard_communication_guide"
GUIDES = {SG, GB, GT, IT, AT, CO, CM}
# id, term, question, expected sources, expected pages (of any expected source) or None, must_not_serve_from_guide
CASES = [
    ("W01", "approval", "Who approves the grades before they reach Banner?", {GT}, [2, 3], False),
    ("W02", "approval", "Who approves conference travel?", {HB}, [117], False),
    ("W03", "submission", "How do I submit a document for a similarity check?", {IT}, [1], False),
    ("W04", "submission", "Where can I see which submissions still need grading?", {GB}, [1, 2, 4], False),
    ("W05", "submission", "Within how many hours must instructors submit final examination results?", {HB}, [224], False),
    ("W06", "review", "How do I review my course as a student would see it?", {SG}, [5], False),
    ("W07", "review", "How do I review the similarity report of my paper?", {IT}, [1], False),
    ("W08", "attendance", "How do I add attendance to the gradebook?", {GB, AT}, [3, 1], False),
    ("W09", "attendance", "What percentage of absences bars a student from the final examination?", {HB}, None, True),
    ("W10", "support", "Whom do I contact if I cannot access iThenticate?", {IT}, [1], False),
    ("W11", "login", "How do I log in to Blackboard from outside the campus?", {SG}, [1], False),
    ("W12", "login", "How do I log in to iThenticate?", {IT}, [1], False),
    ("W13", "login", "How do I log in to the Grade Transfer Dashboard?", {GT}, [1], False),
    ("W14", "login", "How do I verify my identity when I sign in to Blackboard from outside the campus?", {SG}, [1], False),
    ("W15", "policy_boundary", "What grading policy applies to final examinations?", {HB}, [222, 223, 224], True),
    ("W16", "policy_boundary", "How do I post grades to students in Blackboard?", {GB}, [3, 6], False),
    # post-freeze PF-4: the vocabulary of the P1 guides
    ("P01", "attendance", "How do I set up attendance reporting for a Blackboard Collaborate session?", {AT, CO}, [2], False),
    ("P02", "attendance", "How many absences are allowed before a student fails a course?", {HB}, [228], True),
    ("P03", "attendance", "How do I export the attendance sheet of my Blackboard Collaborate sessions?", {CO}, [4], False),
    ("P04", "session", "How do I schedule a recurring Blackboard Collaborate session?", {CO}, [1], False),
    ("P05", "discussion", "How do I make a Blackboard discussion count for a grade?", {CM}, [2], False),
    ("P06", "discussion", "Who may be invited to the discussions of the College Council?", {HB}, [25], False),
    ("P07", "journal", "Which journals count for promotion purposes?", {HB}, [147, 148], False),
    ("P08", "journal", "How do I create a journal in Blackboard Ultra?", {CM}, [5], True),
    ("P09", "conversation", "How do I allow class conversations on an assignment in Blackboard?", {SG, CM}, [10, 4], False),
    ("P10", "recording", "How do I record my Blackboard Collaborate session?", {CO}, [2], False),
    ("P11", "recording", "Who records the minutes of College Council meetings?", {HB}, [25], False),
    ("P12", "announcement", "How do I send an announcement to my students in Blackboard?", {SG}, [9], False),
    ("P13", "report", "Where do I see the similarity report for my manuscript in iThenticate?", {IT}, [1], False),
    ("P14", "report", "How do I see the attendance overview for my Blackboard Collaborate sessions?", {CO}, [4], False),
    ("P15", "grade", "How do I change the attendance grade schema in Blackboard?", {AT}, [1], False),
    ("P16", "assessment", "Who reviews the final examination results of a course?", {HB}, [224, 225], False),
    ("P17", "upload", "How do I upload a document to iThenticate?", {IT}, [1], False),
    ("P18", "access", "How do I access the attendance tool in Blackboard Ultra?", {AT}, [1], False),
    ("P19", "support", "Who do I contact if I cannot log in to iThenticate?", {IT}, [1], False),
    ("P20", "collaborate", "How do I collaborate with the Technology Transfer Office on a patent?", {HB}, [178], False),
    # post-freeze PF-5B (D-114)
    ("T01", "typo", "how do i send an anouncement to my students in blackbord", {SG}, [9], False),
    ("T02", "typo", "how do I upload a document to ithentcate", {IT}, [1], False),
    ("T03", "typo", "how many documents can I submit to ithentcate per year", {IT}, None, True),
    ("T04", "typo", "can I colab with a colleague from another university on a grant", {HB}, None, True),
    ("T05", "typo", "where are colab recordings kept in blackbord", {CO}, None, False),
    ("T06", "typo", "how do I add a column in the gradbook", {GB}, None, False),
    ("T07", "typo", "how do I send grades to baner", {GT}, [2], False),
    ("T08", "typo", "is the attendance polcy 15% in blackbord", {HB}, None, True),
    ("S01", "where", "Where do I find the gradebook in Blackboard Ultra?", {GB}, None, False),
    ("S02", "where", "Where is the Blackboard support office?", {HB}, None, True),
    ("S03", "where", "Where is the iThenticate support office on campus?", {HB}, None, True),
    ("S04", "where", "Where in Blackboard do I see the attendance overview of my Collaborate sessions?", {CO, AT}, None, False),
    ("C01", "collaborate", "How do I collaborate with industry partners on a research project?", {HB}, None, True),
    ("C02", "collaborate", "How do I start a Blackboard Collaborate session?", {CO}, None, False),
    ("A01", "attendance", "Does the Blackboard attendance tool decide when a student is barred for absences?", {HB}, None, True),
    ("A02", "attendance", "How do I mark a student excused in the Blackboard attendance tool?", {AT}, None, False),
    ("G01", "grade", "What grade does a student need to pass a course?", {HB}, None, True),
    ("G02", "grade", "How do I change a grade in the Blackboard gradebook?", {GB}, None, False),
    ("U01", "submission", "How do I submit final grades to Banner through the Grade Transfer Dashboard?", {GT}, [2], False),
    ("U02", "submission", "What is the deadline to submit final grades?", {HB}, None, True),
    ("J01", "journal", "Which research journals are recognised for promotion?", {HB}, None, True),
    ("J02", "journal", "How do I set up a Blackboard journal for my research group?", {CM}, None, True),
    ("J03", "journal", "How do I submit my journal manuscript to iThenticate?", {IT}, None, False),
    ("J04", "journal", "Can students see each other's journal entries in Blackboard?", {CM}, None, True),
]

t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
runtime, build = PF.build_runtime(kb, guide_scopes=True)
assert build.ok, build.to_dict()
rows = []
for cid, term, question, sources, pages, must_not in CASES:
    emb = kb.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    cands = gather_candidates(question, classify_query(question), emb, kb.index, kb.chunks, kb.metadata)
    scores = kb.reranker.predict([(question, c["chunk"]) for c in cands]) if cands else []
    ranked = [c for _, c in sorted(zip(scores, cands), key=lambda x: -float(x[0]))][:5]
    top = [(c["meta"]["source_id"], c["meta"]["page"], c["meta"]["chunk_type"]) for c in ranked]
    top1_ok = bool(top) and top[0][0] in sources
    page_ok = pages is None or any(s in sources and p in pages for s, p, _ in top)
    attempt = runtime.attempt(question)
    accepted = attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.result is not None
    used = sorted({(i.get("meta") or {}).get("source_id") for i in (attempt.result.items if accepted else [])})
    wrong_source = accepted and not set(used) <= sources
    guide_used = accepted and bool(set(used) & GUIDES)
    violation = wrong_source or (must_not and guide_used)
    rows.append({"id": cid, "term": term, "question": question, "expected_sources": sorted(sources), "expected_pages": pages,
                 "must_not_serve_from_guide": must_not, "top5": top, "top1_expected_source": top1_ok, "expected_page_in_top5": page_ok,
                 "plan_f_outcome": attempt.outcome.value, "plan_f_served_sources": used, "wrong_source_acceptance": wrong_source,
                 "violation": violation})
    print("%s %-15s top1=%-5s page=%-5s planf=%-36s used=%s %s| %s" % (cid, term, top1_ok, page_ok, attempt.outcome.value, [u[4:20] for u in used],
                                                                  "VIOLATION " if violation else "", question[:55]))

summary = {"cases": len(rows), "pf2_cases": 16, "pf4_cases": 20, "pf5b_cases": len(rows) - 36, "top1_expected_source": sum(r["top1_expected_source"] for r in rows),
           "expected_page_in_top5": sum(r["expected_page_in_top5"] for r in rows), "plan_f_served": sum(1 for r in rows if r["plan_f_served_sources"]),
           "wrong_source_acceptances": [r["id"] for r in rows if r["wrong_source_acceptance"]], "violations": [r["id"] for r in rows if r["violation"]],
           "retrieval_wrong_source_first": [r["id"] for r in rows if not r["top1_expected_source"]], "seconds": round(time.perf_counter() - t0, 1)}
json.dump({"method": __doc__.strip(), "summary": summary, "cases": rows}, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary))
