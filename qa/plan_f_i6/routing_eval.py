"""I-6 evaluation on the REAL knowledge base (production build path, real models, real retrieval). Writes JSON to OUT.

For every question of the I-3 set (62), the I-4 probes (14), the I-5 aggregation set (26) and the I-6 routing and handoff set
(24, below) the script records: (1) the Coordinator's decision and the routing refinement (owners before and after, code),
(2) the adapter attempt with its routing, handoff, verification and aggregation records, timings, answer, pages and items,
(3) the I-5 behaviour of the same question from ``qa/plan_f_i5/results_2026-10-30.json`` for the before-and-after comparison,
(4) the pipeline answer in SHADOW mode (compared with the direct ``answer_question`` call) and in ACTIVE mode with the diagnostics.
No .env is touched; the Plan F switches are set on the config module in this process only. Plan F makes no model call."""
import json
import os
import sys
import time

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.pop("PLAN_F_ENABLED", None)
os.environ.pop("PLAN_F_SHADOW", None)
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "i6_log.txt"), "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


#: The routing and handoff set (phase I-6). Every fact asked is one the Faculty Handbook states; the tag names the shape the
#: question was written to exercise, not a prediction of its outcome. Control-flow edges (self loop, cycle, second hop,
#: invalid target) are covered by synthetic fixtures in tests/test_plan_f_routing.py, not by real questions.
ROUTING_SET = [
    ("contact_named_hr", "What is the fax number of Human Resources?"),
    ("location_named_hr", "Where is the Human Resources Department located?"),
    ("contact_person_subject", "Who do I contact about research grants?"),
    ("authority_hr", "Who approves sabbatical leave?"),
    ("authority_hr", "Who approves additional maternity leave without pay?"),
    ("authority_research", "Who approves the petty cash advance for a principal investigator?"),
    ("responsibility_hr", "Who handles faculty visas?"),
    ("selection_research", "Which committee approves research proposals?"),
    ("contact_research_unit", "What is the email address of the Research Funding Department?"),
    ("location_research_unit", "Where is the Technology Transfer Office?"),
    ("location_handoff_candidate", "Where is the Research Institute?"),
    ("location_other", "How do I get to the Main Library?"),
    ("cross_teaching_research", "How many office hours must faculty hold and how much petty cash advance can a principal investigator receive?"),
    ("cross_teaching_institutional", "What is the teaching load for lecturers and what is the phone number of the Registration office?"),
    ("cross_hr_research", "How many days of sick leave with full pay are faculty entitled to and who determines the funding ceiling for research grants?"),
    ("cross_teaching_hr", "What percentage of absences bars a student from the final examination, and how many days of maternity leave are female faculty entitled to?"),
    ("cross_hr_institutional", "Does HR assist new faculty members in obtaining entry visas, and what is the phone number of the Passport Unit?"),
    ("same_domain_research", "How many theses can a faculty member supervise at once, and who approves conference travel?"),
    ("same_domain_institutional", "What is the phone number of the Dental Hospital Call Center and how many colleges does the University of Sharjah have?"),
    ("mixed_partial", "Who approves research leave, and how many days of sick leave with full pay are faculty entitled to?"),
    ("responsibility_institutional", "Who is responsible for health and safety on campus?"),
    ("general_institutional", "What is the mission of the University of Sharjah?"),
    ("unsupported_no_specialist", "What is the purpose of the Faculty Handbook?"),
    ("general_hr", "What does the Human Resources Department do for new faculty?"),
]

T0 = time.perf_counter()
from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot import plan_f_runtime as PF                        # noqa: E402
from handbook_bot import plan_f_routing as RT                        # noqa: E402
from handbook_bot.agents.coordinator import coordinate              # noqa: E402
from ui import pipeline                                             # noqa: E402

t = time.perf_counter()
kb = build_knowledge_base(verbose=False)
build_s = time.perf_counter() - t
runtime, build = PF.build_runtime(kb)
say("imports+kb %.1f s (kb %.1f s): %d pages, %d chunks; registry ok=%s %.0f ms; groq client=%s" % (
    time.perf_counter() - T0, build_s, len(kb.pages), len(kb.chunks), build.ok, build.ms, kb.groq_client is not None))

src = open(os.path.join(REPO, "qa", "plan_f_i3", "runtime_measure.py"), encoding="utf-8").read()
QS = eval(src[src.index("QS = [") + 5:src.index("]\nsay(\"questions:\"") + 1])
src4 = open(os.path.join(REPO, "qa", "plan_f_i4", "reverify.py"), encoding="utf-8").read()
PROBES = eval(src4[src4.index("PROBES = [") + 9:src4.index(")]\nI3 =") + 2])
src5 = open(os.path.join(REPO, "qa", "plan_f_i5", "aggregate_eval.py"), encoding="utf-8").read()
AGG = eval(src5[src5.index("AGGREGATION_SET = [") + 18:src5.index("]\n\nT0 =") + 1])
I5 = json.load(open(os.path.join(REPO, "qa", "plan_f_i5", "results_2026-10-30.json"), encoding="utf-8"))
i5_direct = {r["question"]: r for r in I5["direct"]}
i5_active = {r["question"]: r for r in I5["results"] if r["mode"] == "active"}
ALL = [("corpus", d, q) for d, q in QS] + [("probe", d, q) for d, q in PROBES] + [("aggregation_set", d, q) for d, q in AGG] + [("routing_set", d, q) for d, q in ROUTING_SET]
say("questions: %d corpus + %d probes + %d aggregation set + %d routing set" % (len(QS), len(PROBES), len(AGG), len(ROUTING_SET)))


def items_of(result):
    return [{"idx": i.get("idx"), "page": i["meta"].get("page"), "chunk": i.get("chunk"), "plan_f": i["meta"].get("plan_f")} for i in (result.items if result else [])]


def i5_view(q):
    prev, act = i5_direct.get(q), i5_active.get(q)
    if prev is None:
        return None
    a = prev["attempt"]
    return {"outcome": a["outcome"], "path": a.get("path"), "aggregation_reason": (a.get("aggregation") or {}).get("reason"), "reason": a["reason"][:120],
            "served": (act or {}).get("diagnostics", {}).get("served") == "plan_f", "answer": a.get("answer"), "pages": a.get("pages")}


# ---- 1. decision, routing, adapter attempt ------------------------------------------------------------------------------
direct = []
for n, (group, domain, q) in enumerate(ALL, 1):
    rec = {"n": n, "group": group, "domain": domain, "question": q, "i5": i5_view(q)}
    decision = coordinate(q)
    refined, routing = RT.refine(decision)
    rec["decision"] = {"clauses": [c for c in decision.metadata["analysis"]["clauses"]], "tasks": [(t.specialist_id.value, t.context["focus_clauses"]) for t in decision.subtasks],
                       "complexity": decision.metadata.get("complexity"), "is_multipart": decision.metadata["router"]["is_multipart"],
                       "decision_problem": PF.decision_problem(decision)}
    rec["routing"] = routing.to_dict()
    rec["routing"]["ms"] = routing.ms
    rec["refined_tasks"] = [(t.specialist_id.value, t.context["focus_clauses"]) for t in refined.subtasks]
    a = runtime.attempt(q)
    rec["attempt"] = {"outcome": a.outcome.value, "path": a.path, "reason": a.reason, "specialist": a.specialist_id, "status": a.status, "timings": a.timings,
                      "verified": a.verified, "specialist_calls": a.specialist_calls, "handoff_outcome": a.handoff_outcome,
                      "routing": a.routing.to_dict() if a.routing is not None else None, "handoff": a.handoff.to_dict() if a.handoff is not None else None,
                      "verification": a.verification.to_dict() if a.verification is not None else None,
                      "aggregation": a.aggregation.to_dict() if a.aggregation is not None else None,
                      "answer": a.result.answer if a.result else None, "pages": a.result.pages if a.result else None, "items": items_of(a.result),
                      "sub_questions": [(s.question, s.route) for s in a.result.sub_questions] if a.result else [],
                      "llm_calls": a.result.llm_calls if a.result else 0, "used_llm": any(e.used_llm for e in a.trace),
                      "dispatch_llm_calls": a.dispatch.llm_calls if a.dispatch is not None else 0,
                      "runs": [(r.specialist_id.value, r.state, r.findings.status.value if r.findings else None, r.reason[:40]) for r in (a.dispatch.runs if a.dispatch else [])],
                      "trace": [(e.agent, e.decision) for e in a.trace]}
    prev = rec["i5"]
    if prev and prev.get("served"):
        rec["i5_answer_equal"] = (a.result is not None and a.result.answer == prev["answer"] and list(a.result.pages) == list(prev["pages"]))
    direct.append(rec)
    say("[%3d] %-16s %-34s path=%-26s route=%-30s handoff=%-30s calls=%d | %s" % (
        n, group, a.outcome.value, a.path, routing.reason.value, a.handoff_outcome, a.specialist_calls, q[:52]))

# ---- 2. pipeline in SHADOW and ACTIVE --------------------------------------------------------------------------------------
from handbook_bot.qa import answer_question                         # noqa: E402


def visible(r):
    return {"answer": r.answer, "pages": list(r.pages), "best_section": r.best_section, "query_type": r.query_type, "evidence": r.evidence,
            "items": [(i.get("idx"), i["meta"].get("page"), i.get("chunk")[:60]) for i in r.items], "used_llm": r.used_llm, "llm_calls": r.llm_calls,
            "retried": r.retried, "sub_questions": [(s.question, s.route) for s in r.sub_questions], "num_candidates": r.num_candidates,
            "num_reranked": r.num_reranked}


results = []
for mode in ("shadow", "active"):
    config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = (mode == "active"), (mode == "shadow")
    pipeline.reset_plan_f_state()
    say("\n===== MODE %s (resolved %s) =====" % (mode, pipeline.plan_f_mode()))
    for n, (group, domain, q) in enumerate(ALL, 1):
        t = time.perf_counter()
        try:
            res, d = pipeline.ask_with_diagnostics(kb, q)
            exc = None
        except Exception as e:
            res, d, exc = None, None, PF.redact_error(e)
        wall = (time.perf_counter() - t) * 1000
        rec = {"mode": mode, "n": n, "group": group, "domain": domain, "question": q, "wall_ms": wall, "exception": exc}
        if res is not None:
            rec.update({"answer": res.answer, "pages": list(res.pages), "best_section": res.best_section, "query_type": res.query_type,
                        "used_llm": res.used_llm, "llm_calls": res.llm_calls, "retried": res.retried, "timings": res.timings,
                        "sub_questions": [(s.question, s.route) for s in res.sub_questions],
                        "trace": [(e.agent, e.decision) for e in res.agent_trace], "diagnostics": d, "items": items_of(res)})
            if mode == "shadow":
                direct_res = answer_question(q, embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata,
                                             pages=kb.pages, groq_client=kb.groq_client)
                rec["identical_to_direct"] = visible(res) == visible(direct_res)
        results.append(rec)
        say("[%3d] %-16s wall=%5.0f served=%-7s path=%-26s outcome=%-34s route=%-30s handoff=%-26s %s" % (
            n, group, wall, (d or {}).get("served"), (d or {}).get("plan_f_path"), (d or {}).get("outcome"), (d or {}).get("routing_reason"),
            (d or {}).get("handoff_outcome"), q[:40]))

json.dump({"kb": {"pages": len(kb.pages), "chunks": len(kb.chunks), "build_s": build_s, "registry_ms": build.ms, "groq_client_configured": kb.groq_client is not None},
           "routing_set": ROUTING_SET, "direct": direct, "results": results, "total_script_s": time.perf_counter() - T0},
          open(os.path.join(OUT, "i6_results.json"), "w", encoding="utf-8"), indent=1, default=str)
say("done %.1f s" % (time.perf_counter() - T0))
