"""LEGACY isolation with the PF-4 modules (eight sources), in a FRESH process: with every Plan F switch off (guide scopes included), the
representative questions go through the real seam; no Plan F module, no D-111 or D-112 module (``guide_grounding``, ``guide_structure``,
``guide_evidence``) and no source policy module may be imported, no runtime cache may exist, every diagnostics record must
be the minimal legacy one. Prints a JSON summary.

    python -X utf8 qa/post_freeze_pf4/legacy_isolation.py <repo>
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
os.chdir(REPO)
sys.path.insert(0, REPO)
for name in ("PLAN_F_ENABLED", "PLAN_F_SHADOW", "PLAN_F_GUIDE_SCOPES"):
    os.environ.pop(name, None)

from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from ui import pipeline                                             # noqa: E402

WATCHED = ("handbook_bot.plan_f_runtime", "handbook_bot.plan_f_routing", "handbook_bot.plan_f_handoff", "handbook_bot.plan_f_aggregation",
           "handbook_bot.plan_f_verification", "handbook_bot.guide_grounding", "handbook_bot.guide_structure", "handbook_bot.source_scopes",
           "handbook_bot.agents.specialists.guide_evidence", "handbook_bot.agents.specialists.dispatch", "handbook_bot.agents.specialists.teaching",
           "handbook_bot.agents.specialists.research", "handbook_bot.agents.specialists.institutional", "handbook_bot.agents.specialists.faculty_services")
QUESTIONS = ["How do I access iThenticate?", "How do I transfer grades from Blackboard to Banner?", "What can I change from the Gradebook Settings panel in Blackboard Ultra?",
             "How many documents can I submit to iThenticate per year?", "How many credit hours is a full teaching load?", "What does Human Resources do?",
             "How many days of annual leave do faculty members get?", "How do I start the Phase 2 progress review in the HR Portal?",
             "How do I send an announcement in Blackboard Ultra?", "How do I assign automatic zeros in the Blackboard Ultra gradebook?",
             "How do I export the attendance sheet for my Blackboard Collaborate sessions?", "How do I create a discussion in Blackboard Ultra?"]
kb = build_knowledge_base(verbose=False)
minimal = exceptions = 0
for q in QUESTIONS:
    try:
        _res, d = pipeline.ask_with_diagnostics(kb, q)
        minimal += int(d == {"mode": "legacy", "served": "legacy", "attempted": False})
    except Exception:
        exceptions += 1
print(json.dumps({"mode": pipeline.plan_f_mode(), "questions": len(QUESTIONS), "minimal_diagnostics": minimal, "exceptions": exceptions,
                  "watched_modules_imported": [m for m in WATCHED if m in sys.modules], "runtime_cache_created": pipeline._PLAN_F_CACHE is not None},
                 indent=1))
