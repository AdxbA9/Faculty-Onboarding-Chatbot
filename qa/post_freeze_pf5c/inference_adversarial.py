"""PF-5C inference adversarial check (D-115) with the REAL models on the eight-source knowledge base.

Questions that must NOT be inferred into a software guide: the six the project owner listed (single ambiguous words and
pronoun asks), and more of the same kinds (weak single words, policy and authority asks, organizational and contact asks,
other platforms, collisions). For each question the guide-enabled runtime (``build_runtime(kb, guide_scopes=True)``)
attempts it; the check passes when guide inference refuses it (question and every clause) AND no guide answer is served.
Writes ``inference_adversarial.json``.

    python -X utf8 qa/post_freeze_pf5c/inference_adversarial.py <repo> <out json>
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
from handbook_bot import plan_f_runtime as PF                          # noqa: E402

HANDBOOK = "uos_faculty_handbook_2025_26"
# id, kind, question
CASES = [
    ("X01", "owner_listed", "I need help with grades."),
    ("X02", "owner_listed", "How do I use the system?"),
    ("X03", "owner_listed", "Where is the report?"),
    ("X04", "owner_listed", "How do I submit it?"),
    ("X05", "owner_listed", "Can I collaborate with someone?"),
    ("X06", "owner_listed", "Where do I record this?"),
    ("X07", "weak_word", "How do I submit grades?"),
    ("X08", "weak_word", "Where are my submissions?"),
    ("X09", "weak_word", "How do I check my messages?"),
    ("X10", "weak_word", "How do I create a session?"),
    ("X11", "weak_word", "How do I teach online?"),
    ("X12", "weak_word", "Where can I find the reports?"),
    ("X13", "weak_word", "How do I upload a document?"),
    ("X14", "weak_word", "Is attendance shown to students?"),
    ("X15", "policy", "What is the attendance policy?"),
    ("X16", "policy", "Do I have to take attendance in every lecture?"),
    ("X17", "policy", "What is the deadline to submit final grades?"),
    ("X18", "policy", "What is the penalty for plagiarism?"),
    ("X19", "policy", "Am I required to use the discussion board in my course?"),
    ("X20", "authority", "Who approves final grades?"),
    ("X21", "organizational", "Where is the e-learning support office?"),
    ("X22", "organizational", "Which team helps faculty with online classes?"),
    ("X23", "organizational", "Who do I contact about the similarity report?"),
    ("X24", "organizational", "How do I approve final grades as chair?"),
    ("X25", "other_platform", "How do I schedule a Zoom meeting for my online class?"),
    ("X26", "other_platform", "How do I check the similarity report in Turnitin?"),
    ("X27", "collision", "How do I check my students' assignments for plagiarism?"),
    ("X28", "collision", "How do I add a journal to my research publications?"),
    ("X29", "collision", "How do I send final grades to my students?"),
    ("X30", "collision", "How do I upload final grades?"),
]

t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
runtime, build = PF.build_runtime(kb, guide_scopes=True)
assert build.ok, build.to_dict()
rows = []
for cid, kind, question in CASES:
    _registry, record = runtime.select_registry(question)
    inference = record.get("inference") or {}
    attempt = runtime.attempt(question)
    accepted = attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.result is not None
    used = sorted({(i.get("meta") or {}).get("source_id") for i in (attempt.result.items if accepted else [])})
    guide_served = bool(set(used) - {HANDBOOK})
    refused = not inference.get("accepted") and not record.get("inferred")
    rows.append({"id": cid, "kind": kind, "question": question, "inference_refused": refused,
                 "inference_reason": (inference.get("lead") or {}).get("reason"), "inference_parts": inference.get("parts"),
                 "plan_f_outcome": attempt.outcome.value, "plan_f_served_sources": used, "guide_served": guide_served,
                 "pass": refused and not guide_served})
    print("%s %-14s refused=%-5s reason=%-16s outcome=%-34s guide=%-5s | %s" % (cid, kind, refused, rows[-1]["inference_reason"],
                                                                              attempt.outcome.value, guide_served, question))
summary = {"cases": len(rows), "passed": sum(r["pass"] for r in rows), "failed": [r["id"] for r in rows if not r["pass"]],
           "inference_accepted": [r["id"] for r in rows if not r["inference_refused"]], "guide_served": [r["id"] for r in rows if r["guide_served"]],
           "plan_f_served_handbook": [r["id"] for r in rows if r["plan_f_served_sources"] and not r["guide_served"]],
           "seconds": round(time.perf_counter() - t0, 1)}
json.dump({"method": __doc__.strip(), "summary": summary, "cases": rows}, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary))
