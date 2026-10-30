"""LEGACY isolation check in a FRESH process (section 23 of the brief): with both Plan F switches off, answer the
representative sample through the real seam and confirm that no Plan F module was imported, no runtime cache was created,
every diagnostics record is the minimal legacy record and no exception escaped. Prints a JSON summary.

    python -X utf8 qa/plan_f_i8/legacy_check.py <repo>
"""
import json
import os
import sys
import time

REPO = sys.argv[1]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "qa", "plan_f_i8"))
os.chdir(REPO)
os.environ.pop("PLAN_F_ENABLED", None)
os.environ.pop("PLAN_F_SHADOW", None)
from question_set import LEGACY_SAMPLE_IDS, by_id                   # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import pipeline                                             # noqa: E402

PLAN_F_MODULES = ("handbook_bot.plan_f_runtime", "handbook_bot.plan_f_routing", "handbook_bot.plan_f_handoff",
                  "handbook_bot.plan_f_aggregation", "handbook_bot.plan_f_verification", "handbook_bot.agents.specialists.dispatch",
                  "handbook_bot.agents.specialists.registry", "handbook_bot.agents.specialists.teaching", "handbook_bot.agents.specialists.research",
                  "handbook_bot.agents.specialists.faculty_services", "handbook_bot.agents.specialists.institutional")
kb = build_knowledge_base(verbose=False)
ids = by_id()
rows, exceptions, minimal = [], 0, 0
for entry_id in LEGACY_SAMPLE_IDS:
    q = ids[entry_id]["q"]
    t = time.perf_counter()
    try:
        res, d = pipeline.ask_with_diagnostics(kb, q)
        ok = d == {"mode": "legacy", "served": "legacy", "attempted": False}
        minimal += ok
        rows.append({"id": entry_id, "route": res.query_type, "pages": list(res.pages), "wall_ms": (time.perf_counter() - t) * 1000, "minimal": ok,
                     "answer_chars": len(res.answer)})
    except Exception as exc:
        exceptions += 1
        rows.append({"id": entry_id, "exception": type(exc).__name__})
imported = [m for m in PLAN_F_MODULES if m in sys.modules]
print(json.dumps({"mode": pipeline.plan_f_mode(), "sample": len(rows), "minimal_diagnostics": minimal, "exceptions": exceptions,
                  "plan_f_modules_imported": imported, "runtime_cache_created": pipeline._PLAN_F_CACHE is not None,
                  "recent_diagnostics": len(pipeline.recent_plan_f_diagnostics()), "rows": rows}, indent=1))
