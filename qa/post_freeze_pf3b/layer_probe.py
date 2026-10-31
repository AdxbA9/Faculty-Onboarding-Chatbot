"""PF-3B layer probe: if a specialist DID return a supported guide finding (stubbed), which adapter layer would decide?
Real KB; the guide-enabled registry's specialist is replaced for one question by a stub that returns exactly one
supported finding quoting a guide chunk. Shows whether the findings verifier, the subject guard, the authority guard or the
legacy Verifier would accept or refuse, so the next decision record can name every layer that guide answers face."""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = sys.argv[1]
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
from handbook_bot.knowledge_base import build_knowledge_base    # noqa: E402
from handbook_bot import plan_f_runtime as PF                    # noqa: E402
import test_plan_f_runtime as T                                  # noqa: E402

kb = build_knowledge_base(verbose=False)
runtime, build = PF.build_runtime(kb, guide_scopes=True)
assert runtime.guide_registry is not None


def locate(source, needle):
    for i, (c, m) in enumerate(zip(kb.chunks, kb.metadata)):
        if m["source_id"] == source and m["chunk_type"] in ("paragraph", "procedure_step", "procedure_window") and needle in c:
            return i, c
    raise LookupError(needle)


CASES = [
    ("teaching", "What is the minimum size of a Blackboard course banner image?", "uos_blackboard_faculty_short_guide",
     "Your image must be at least 1200 x 240 pixels and it can't be larger than 25 KB.", ("number or amount",)),
    ("teaching", "Can I add a time limit to a group assignment in Blackboard Ultra?", "uos_blackboard_faculty_short_guide",
     "At this time, you cannot add a time limit to group assignments.", ("maximum or limit",)),
    ("teaching", "How long can the title of a gradebook item be in Blackboard Ultra?", "uos_blackboard_gradebook_guide",
     "Your title is limited to 255 characters.", ("number or amount",)),
    ("research", "What is the maximum file size for an iThenticate upload?", "uos_ithenticate_guidelines",
     "Files must be less than 100MB.", ("maximum or limit",)),
    ("research", "How many documents can I submit to iThenticate per year?", "uos_ithenticate_guidelines",
     "Each instructor is limited to a specific amount of submission (Maximum 30 Documents per year)", ("number or amount",)),
    ("teaching", "What is the required teaching load for faculty in Blackboard?", "uos_blackboard_faculty_short_guide",
     "Your image must be at least 1200 x 240 pixels and it can't be larger than 25 KB.", ("number or amount",)),
    ("teaching", "How do I add a learning module in Blackboard Ultra?", "uos_blackboard_faculty_short_guide",
     "Click on Learning Module or Folder.", ("location or system",)),
]
rows = []
for sid, question, source, needle, covered in CASES:
    try:
        cid, chunk = locate(source, needle)
    except LookupError:
        cid, chunk = locate(source, needle.split(" (")[0][:40])
    start = chunk.find(needle)
    quote = needle if start >= 0 else chunk[:120]
    specialist = runtime.guide_registry.resolve(sid)
    original = specialist.run
    specialist.run = lambda task, cid=cid, quote=quote, covered=covered: T.retrieval_result(kb, task, cid, quote, requested=covered, covered=covered)
    try:
        a = runtime.attempt(question)
    finally:
        specialist.run = original
    trace = [(e.agent, e.decision) for e in a.trace if e.agent not in ("plan_f_adapter", "plan_f_registry")]
    rows.append({"question": question, "source": source, "quote": quote, "outcome": a.outcome.value, "reason": a.reason[:220],
                 "answer": a.result.answer if a.result else None, "trace": trace})
    print("%-36s | %s\n      %s\n      %s" % (a.outcome.value[7:], question, a.reason[:200], trace))
json.dump(rows, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
