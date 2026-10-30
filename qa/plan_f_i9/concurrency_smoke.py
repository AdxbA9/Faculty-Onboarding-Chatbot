"""Small concurrency smoke of the seam in ACTIVE mode (runtime integration, phase I-9): several threads ask the pipeline at
once on a fresh runtime state, so the first requests race for the registry build. Checks: the registry is built exactly once
(no duplicate build, the I-2 cache holds), every concurrent answer equals the sequential answer of the same question (served
path, outcome, answer text, pages, specialist calls: no shared-state corruption), the recorded diagnostics are one per call
with no cross-contamination (each record's pages match its own question), no exception, 0 model calls. Prints a JSON summary.

    python -X utf8 qa/plan_f_i9/concurrency_smoke.py <repo> [threads=8]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = sys.argv[1]
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.pop("PLAN_F_ENABLED", None)
os.environ.pop("PLAN_F_SHADOW", None)
from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import pipeline                                             # noqa: E402

QUESTIONS = [
    "What is the phone number of the Human Resources Department?",                                   # served, directory row
    "How many office hours must faculty hold?",                                                     # served, single finding
    "Who approves conference travel?",                                                              # served, routing refined
    "How many credit hours is a full teaching load and how many days of sick leave with full pay are faculty entitled to?",  # aggregate
    "What does HR do?",                                                                             # known safety fallback
    "Which office handles faculty housing?",                                                        # ambiguous routing, 0 calls
    "Where is the Research Institute?",                                                             # handoff round, fallback
    "What is the capital of France?",                                                               # not applicable, refusal
]
kb = build_knowledge_base(verbose=False)
config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = True, False


def view(res, d):
    return {"served": d.get("served"), "outcome": d.get("outcome"), "answer": res.answer, "pages": list(res.pages),
            "specialist_calls": d.get("specialist_calls"), "handoff_outcome": d.get("handoff_outcome"), "llm": d.get("plan_f_llm_calls")}


# sequential reference on a fresh state
pipeline.reset_plan_f_state()
reference = {}
for q in QUESTIONS:
    res, d = pipeline.ask_with_diagnostics(kb, q)
    reference[q] = view(res, d)
seq_builds = pipeline._plan_f_cache().builds

# concurrent run on a fresh state: THREADS workers, each question twice, interleaved
pipeline.reset_plan_f_state()
work = [q for q in QUESTIONS for _ in range(2)]
errors = []
results = []


def ask(q):
    t = time.perf_counter()
    try:
        res, d = pipeline.ask_with_diagnostics(kb, q)
        return q, view(res, d), (time.perf_counter() - t) * 1000, None
    except Exception as exc:
        return q, None, (time.perf_counter() - t) * 1000, type(exc).__name__


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=THREADS) as pool:
    for q, v, ms, err in pool.map(ask, work):
        results.append((q, v, ms))
        if err:
            errors.append((q, err))
wall = (time.perf_counter() - t0) * 1000
builds = pipeline._plan_f_cache().builds
mismatch = [(q, v, reference[q]) for q, v, _ms in results if v != reference[q]]
recent = pipeline.recent_plan_f_diagnostics(limit=200)
page_sets = {q: set(reference[q]["pages"]) for q in QUESTIONS}
# cross-contamination: every recorded diagnostics record must carry the pages of exactly one of the asked questions' results
expected_page_sets = [set(reference[q]["pages"]) for q in QUESTIONS]
cross = 0
for rec in recent:
    pages = set(rec.get("plan_f_pages") or []) if rec.get("served") == "plan_f" else set(rec.get("legacy_pages") or [])
    if pages not in expected_page_sets:
        cross += 1
print(json.dumps({"threads": THREADS, "calls": len(work), "wall_ms": round(wall), "registry_builds_sequential": seq_builds, "registry_builds_concurrent": builds,
                  "exceptions": errors, "mismatches_vs_sequential": len(mismatch), "mismatch_examples": [(q, v and v["outcome"], r["outcome"]) for q, v, r in mismatch[:3]],
                  "diagnostic_records": len(recent), "records_with_foreign_pages": cross, "plan_f_llm_calls": max((r.get("plan_f_llm_calls") or 0) for r in recent),
                  "served_concurrent": sum(1 for _q, v, _ms in results if v and v["served"] == "plan_f"),
                  "per_question": {q: reference[q]["served"] + "/" + str(reference[q]["outcome"]) for q in QUESTIONS}}, indent=1))
