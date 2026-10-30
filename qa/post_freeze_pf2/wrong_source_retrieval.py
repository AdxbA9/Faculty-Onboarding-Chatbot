"""Wrong-source adversarial retrieval with the REAL models (brief sections 30, 31 and 33): questions whose key terms
("approval", "submission", "review", "attendance", "support", "login") occur in several documents, plus system-versus-
policy pairs. For each question the unscoped legacy retrieval (dense + lexical candidates, then the production cross-
encoder rerank, exactly as ``qa.answer_question`` ranks them) is read and the top five are checked against the source and
page the question belongs to. Writes ``wrong_source_retrieval.json``.

    python -X utf8 qa/post_freeze_pf2/wrong_source_retrieval.py <repo> <out json>
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

from handbook_bot.knowledge_base import build_knowledge_base     # noqa: E402
from handbook_bot.retrieval import classify_query, gather_candidates   # noqa: E402

HB, SG, GB, GT, IT = ("uos_faculty_handbook_2025_26", "uos_blackboard_faculty_short_guide", "uos_blackboard_gradebook_guide",
                      "uos_grade_transfer_dashboard_guide", "uos_ithenticate_guidelines")
CASES = [
    # term "approval" / "approve"
    ("W01", "approval", "Who approves the grades before they reach Banner?", GT, [2, 3]),
    ("W02", "approval", "Who approves conference travel?", HB, [117]),
    # term "submission" / "submit"
    ("W03", "submission", "How do I submit a document for a similarity check?", IT, [1]),
    ("W04", "submission", "Where can I see which submissions still need grading?", GB, [1, 2, 4]),
    ("W05", "submission", "Within how many hours must instructors submit final examination results?", HB, [224]),
    # term "review"
    ("W06", "review", "How do I review my course as a student would see it?", SG, [5]),
    ("W07", "review", "How do I review the similarity report of my paper?", IT, [1]),
    # term "attendance"
    ("W08", "attendance", "How do I add attendance to the gradebook?", GB, [3]),
    ("W09", "attendance", "What percentage of absences bars a student from the final examination?", HB, None),
    # term "support" / contact
    ("W10", "support", "Whom do I contact if I cannot access iThenticate?", IT, [1]),
    # term "login" / "log in"
    ("W11", "login", "How do I log in to Blackboard from outside the campus?", SG, [1]),
    ("W12", "login", "How do I log in to iThenticate?", IT, [1]),
    ("W13", "login", "How do I log in to the Grade Transfer Dashboard?", GT, [1]),
    ("W14", "login", "How do I verify my identity when I sign in to Blackboard from outside the campus?", SG, [1]),
    # system versus policy (section 33)
    ("W15", "policy_boundary", "What grading policy applies to final examinations?", HB, [222, 223, 224]),
    ("W16", "policy_boundary", "How do I post grades to students in Blackboard?", GB, [3, 6]),
]

t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
rows = []
for cid, term, question, source, pages in CASES:
    emb = kb.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    cands = gather_candidates(question, classify_query(question), emb, kb.index, kb.chunks, kb.metadata)
    scores = kb.reranker.predict([(question, c["chunk"]) for c in cands]) if cands else []
    ranked = [c for _, c in sorted(zip(scores, cands), key=lambda x: -float(x[0]))][:5]
    top = [(c["meta"]["source_id"], c["meta"]["page"], c["meta"]["chunk_type"]) for c in ranked]
    top1_ok = bool(top) and top[0][0] == source
    page_ok = pages is None or any(s == source and p in pages for s, p, _ in top)
    wrong_first = top[0][0] if top and not top1_ok else None
    rows.append({"id": cid, "term": term, "question": question, "expected_source": source, "expected_pages": pages, "top5": top,
                 "top1_expected_source": top1_ok, "expected_page_in_top5": page_ok, "wrong_source_first": wrong_first})
    print("%s %-16s top1=%-5s page=%-5s %s | %s" % (cid, term, top1_ok, page_ok, [(s[4:18], p) for s, p, _ in top[:3]], question[:60]))

summary = {"cases": len(rows), "top1_expected_source": sum(r["top1_expected_source"] for r in rows),
           "expected_page_in_top5": sum(r["expected_page_in_top5"] for r in rows),
           "wrong_source_first": [r["id"] for r in rows if r["wrong_source_first"]], "seconds": round(time.perf_counter() - t0, 1)}
json.dump({"method": __doc__.strip(), "summary": summary, "cases": rows}, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(summary))
