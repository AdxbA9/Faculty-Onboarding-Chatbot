"""PF-6 targeted final checks with the REAL models on the eight-source knowledge base: registered conflicts, the policy/system
boundary and the held sources, each question attempted by the guide-enabled runtime (guide scopes ON, the strictest
setting) and by the production default runtime (guide scopes OFF), plus the unscoped legacy ranking (what LEGACY cites).

    python -X utf8 qa/post_freeze_pf6/final_checks.py <repo> <out dir>

Writes ``conflict_final.json`` and ``boundary_heldsource_final.json`` (no model call; Plan F is deterministic).
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)

from handbook_bot.knowledge_base import build_knowledge_base           # noqa: E402
from handbook_bot.retrieval import classify_query, gather_candidates   # noqa: E402
from handbook_bot import plan_f_runtime as PF                          # noqa: E402

HB = "uos_faculty_handbook_2025_26"
HELD_IDS = ("uos_performance_appraisal_phase2_guide",)
HELD_WORDS = re.compile(r"apprais|assignments?\s*(?:&|and)\s*tests?\s+guide|grade\s+categor", re.I)
QUOTA = re.compile(r"\b30\s+documents?\b|\bmaximum\s+30\b|\b15\s+documents?\b|limited\s+to\s+a\s+specific\s+amount", re.I)
JOURNAL = re.compile(r"\bjournals?\b", re.I)

CONFLICT = [
    # (id, group, question, must_not_regex or None, expect_served_guide)
    ("Q01", "ithenticate_quota", "How many documents can I submit to iThenticate per year?", QUOTA, False),
    ("Q02", "ithenticate_quota", "What is the yearly iThenticate submission limit?", QUOTA, False),
    ("Q03", "ithenticate_quota", "Is the iThenticate quota 30 or 15 documents per year?", QUOTA, False),
    ("Q04", "ithenticate_quota", "What is the maximum number of documents per year in iThenticate?", QUOTA, False),
    ("Q05", "ithenticate_quota", "how many documents can I submit to ithentcate per year", QUOTA, False),
    ("Q06", "ithenticate_quota", "How many documents can I check for similarity per year?", QUOTA, False),
    ("Q07", "ithenticate_quota", "Does a new revision count toward my iThenticate document limit?", QUOTA, False),
    ("Q08", "ithenticate_not_overblocked", "How do I upload a document to iThenticate?", None, True),
    ("Q09", "ithenticate_not_overblocked", "How do I log in to iThenticate?", None, True),
    ("J01", "journal_2022", "How do I create a journal in Blackboard Ultra?", JOURNAL, False),
    ("J02", "journal_2022", "Should I use journals in my Blackboard course?", JOURNAL, False),
    ("J03", "journal_2022", "How do I grade a journal in Blackboard?", JOURNAL, False),
    ("J04", "journal_2022", "Can students see each other's journal entries in Blackboard?", JOURNAL, False),
    ("J05", "journal_not_overblocked", "How do I allow class conversations on an assignment in Blackboard?", None, True),
    ("J06", "journal_not_overblocked", "How do I delete a discussion topic in Blackboard?", None, None),
]
BOUNDARY = [
    # (id, group, question): a policy, authority or organizational ask that no procedural guide may answer
    ("B01", "attendance_policy_vs_tool", "How many absences are allowed before a student fails?"),
    ("B02", "attendance_policy_vs_tool", "According to the Blackboard attendance tool, how many absences are allowed before a student fails?"),
    ("B03", "attendance_policy_vs_tool", "What percentage of absences bars a student from the final examination?"),
    ("B04", "attendance_policy_vs_tool", "Is attendance compulsory for students?"),
    ("B05", "procedure_vs_policy", "What is the grading policy for final examinations?"),
    ("B06", "procedure_vs_policy", "Is it mandatory to use Blackboard for every course?"),
    ("B07", "procedure_vs_policy", "Am I required to record my online classes in Blackboard Collaborate?"),
    ("B08", "gradebook_vs_grade_transfer_policy", "What is the deadline to submit final grades?"),
    ("B09", "gradebook_vs_grade_transfer_policy", "Within how many hours must instructors submit final examination results?"),
    ("B10", "gradebook_vs_grade_transfer_policy", "Can I change a grade after it has been transferred to Banner?"),
    ("B11", "software_vs_organizational", "Where is the Blackboard support office?"),
    ("B12", "software_vs_organizational", "Who is responsible for Blackboard support at the university?"),
    ("B13", "software_vs_organizational", "Which office handles iThenticate accounts?"),
]
BOUNDARY_TOOL = [   # tool questions on the same subjects: a guide answer is allowed here (and expected for the first two)
    ("T01", "attendance_tool", "How do I share attendance information with the LMS in Blackboard?"),
    ("T02", "grade_transfer_tool", "How do I transfer grades to Banner?"),
    ("T03", "software_navigation", "Where do I find the gradebook in Blackboard Ultra?"),
]
HELD = [
    ("H01", "appraisal", "How do I start the Phase 2 progress review in the HR Portal?"),
    ("H02", "appraisal", "What is the performance appraisal Phase 2 process for faculty?"),
    ("H03", "assignments_and_tests", "How do I create a test with a question pool in Blackboard Ultra?"),
    ("H04", "assignments_and_tests", "How do I set up a timed test in Blackboard?"),
    ("H05", "grade_categories", "How do I create grade categories in the Blackboard gradebook?"),
    ("H06", "grade_categories", "How do I weight grade categories in Blackboard Ultra?"),
]

kb = build_knowledge_base(verbose=False)
on, b1 = PF.build_runtime(kb, guide_scopes=True)
off, b2 = PF.build_runtime(kb, guide_scopes=False)
assert b1.ok and b2.ok
ENABLED = {r.source_id for r in kb.sources}


def attempt(rt, q):
    a = rt.attempt(q)
    ok = a.outcome is PF.PlanFOutcome.ACCEPTED and a.result is not None
    items = a.result.items if ok else []
    return {"outcome": a.outcome.value, "served": ok, "sources": sorted({(i.get("meta") or {}).get("source_id") for i in items}),
            "pages": sorted({(i.get("meta") or {}).get("page") for i in items if isinstance((i.get("meta") or {}).get("page"), int)}),
            "answer": a.result.answer if ok else None, "llm_calls": a.dispatch.llm_calls if a.dispatch is not None else 0}


def legacy_top(q):
    emb = kb.embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    cands = gather_candidates(q, classify_query(q), emb, kb.index, kb.chunks, kb.metadata)
    scores = kb.reranker.predict([(q, c["chunk"]) for c in cands]) if cands else []
    ranked = [c for _, c in sorted(zip(scores, cands), key=lambda x: -float(x[0]))][:5]
    return [(c["meta"].get("source_id"), c["meta"].get("page"), c["meta"].get("source_title")) for c in ranked]


conflict_rows = []
for cid, group, q, bad, expect in CONFLICT:
    row = {"id": cid, "group": group, "question": q}
    for name, rt in (("scopes_on", on), ("scopes_off", off)):
        r = attempt(rt, q)
        guide = [s for s in r["sources"] if s != HB]
        leak = bool(bad and r["answer"] and bad.search(r["answer"]) and guide)
        row[name] = {k: r[k] for k in ("outcome", "served", "sources", "pages", "llm_calls")} | {"leak": leak}
    row["pass"] = not row["scopes_on"]["leak"] and not row["scopes_off"]["leak"] and (
        expect is not True or bool([s for s in row["scopes_on"]["sources"] if s != HB]))
    conflict_rows.append(row)
    print(cid, group, row["scopes_on"]["outcome"], row["scopes_on"]["sources"], "LEAK" if row["scopes_on"]["leak"] else "", row["pass"], "|", q)

boundary_rows = []
for cid, group, q in BOUNDARY + BOUNDARY_TOOL:
    row = {"id": cid, "group": group, "question": q, "policy_ask": cid.startswith("B")}
    for name, rt in (("scopes_on", on), ("scopes_off", off)):
        r = attempt(rt, q)
        row[name] = {k: r[k] for k in ("outcome", "served", "sources", "pages", "llm_calls")}
    guide_on = [s for s in row["scopes_on"]["sources"] if s != HB]
    guide_off = [s for s in row["scopes_off"]["sources"] if s != HB]
    row["pass"] = (not guide_on and not guide_off) if row["policy_ask"] else not guide_off
    boundary_rows.append(row)
    print(cid, group, row["scopes_on"]["outcome"], row["scopes_on"]["sources"], row["pass"], "|", q)

held_rows = []
for cid, group, q in HELD:
    row = {"id": cid, "group": group, "question": q}
    for name, rt in (("scopes_on", on), ("scopes_off", off)):
        r = attempt(rt, q)
        row[name] = {k: r[k] for k in ("outcome", "served", "sources", "pages", "llm_calls")}
    top = legacy_top(q)
    row["legacy_top5"] = [(s, p) for s, p, _t in top]
    row["held_in_retrieval"] = [s for s, _p, t in top if s in HELD_IDS or s not in ENABLED or HELD_WORDS.search(str(t or ""))]
    row["held_in_plan_f"] = [s for name in ("scopes_on", "scopes_off") for s in row[name]["sources"] if s in HELD_IDS or s not in ENABLED]
    row["pass"] = not row["held_in_retrieval"] and not row["held_in_plan_f"]
    held_rows.append(row)
    print(cid, group, row["scopes_on"]["outcome"], row["scopes_on"]["sources"], row["legacy_top5"][:2], row["pass"], "|", q)

summary_c = {"cases": len(conflict_rows), "passed": sum(r["pass"] for r in conflict_rows), "failed": [r["id"] for r in conflict_rows if not r["pass"]],
             "quota_leaks": [r["id"] for r in conflict_rows if r["group"] == "ithenticate_quota" and (r["scopes_on"]["leak"] or r["scopes_off"]["leak"])],
             "journal_leaks": [r["id"] for r in conflict_rows if r["group"] == "journal_2022" and (r["scopes_on"]["leak"] or r["scopes_off"]["leak"])],
             "not_overblocked_served": [r["id"] for r in conflict_rows if r["group"].endswith("not_overblocked") and r["scopes_on"]["served"]],
             "llm_calls": sum(r[n]["llm_calls"] for r in conflict_rows for n in ("scopes_on", "scopes_off"))}
json.dump({"method": __doc__.strip(), "summary": summary_c, "cases": conflict_rows}, open(os.path.join(OUT, "conflict_final.json"), "w",
          encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
summary_b = {"boundary_cases": len(boundary_rows), "boundary_passed": sum(r["pass"] for r in boundary_rows),
             "policy_asks_served_from_a_guide": [r["id"] for r in boundary_rows if r["policy_ask"] and not r["pass"]],
             "held_cases": len(held_rows), "held_passed": sum(r["pass"] for r in held_rows),
             "held_failures": [r["id"] for r in held_rows if not r["pass"]],
             "llm_calls": sum(r[n]["llm_calls"] for r in boundary_rows + held_rows for n in ("scopes_on", "scopes_off"))}
json.dump({"method": __doc__.strip(), "summary": summary_b, "boundary": boundary_rows, "held_sources": held_rows},
          open(os.path.join(OUT, "boundary_heldsource_final.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print(json.dumps({"conflict": summary_c, "boundary_held": summary_b}))
