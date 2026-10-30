"""I-5 evaluation on the REAL knowledge base (production build path, real models, real retrieval). Writes JSON to OUT.

For every question of the I-3 set (62), the I-4 probes (14) and the I-5 aggregation set (26, below) the script records:
(1) the decision shape (request clauses, tasks, complexity, the I-1 decision problem, the aggregator's pre-dispatch request problem),
(2) every dispatched run (status, clause reports, handoffs) and the aggregator's own verdict on the execution, called directly,
(3) the adapter attempt (outcome, path, verification and aggregation records, timings, answer, pages, items),
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
log = open(os.path.join(OUT, "i5_log.txt"), "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


#: The aggregation-focused set (phase I-5). Every fact asked is one the Faculty Handbook states; the tag is the shape the question
#: was written to exercise, not a prediction of the outcome.
AGGREGATION_SET = [
    ("same_domain", "How many credit hours is a full teaching load, and how many office hours must faculty hold?"),
    ("same_domain", "How many days of maternity leave are female faculty entitled to, and how many days of sick leave with full pay are faculty entitled to?"),
    ("same_domain", "Does HR arrange travel and hotel accommodations for new faculty members, and who drafts employment offers?"),
    ("same_domain", "How many theses can a faculty member supervise at once, and how much petty cash advance can a principal investigator receive?"),
    ("same_domain", "What is the phone number of the Human Resources Department and what is the fax number of the Passport Unit?"),
    ("same_domain", "Who determines the funding ceiling for research grants, and which committee reviews research on human subjects?"),
    ("same_domain", "What is the teaching load for lecturers and what percentage of absences bars a student from the final examination?"),
    ("same_domain", "Does HR arrange travel and hotel accommodations for new faculty members, and does HR assist new faculty members in obtaining entry visas?"),
    ("same_domain", "How many credit hours is a full teaching load, how many office hours must faculty hold, and what percentage of absences bars a student from the final examination?"),
    ("same_domain", "How many colleges does the University of Sharjah have, and what is the phone number of the Finance Department?"),
    ("same_domain", "What is the phone number of the Human Resources Department and what is the fax number of the Human Resources Department?"),
    ("same_domain", "How many office hours must faculty hold, and how many office hours per week must faculty hold?"),
    ("cross_domain", "How many credit hours is a full teaching load and how many days of sick leave with full pay are faculty entitled to?"),
    ("cross_domain", "How many theses can a faculty member supervise at once and what is the phone number of the Human Resources Department?"),
    ("cross_domain", "Does HR assist new faculty members in obtaining entry visas, and how many colleges does the University of Sharjah have?"),
    ("cross_domain", "How many credit hours is a full teaching load and how many theses can a faculty member supervise at once?"),
    ("enumerative", "What does HR provide to new faculty members?"),
    ("enumerative", "What information does HR provide to new faculty members?"),
    ("missing_clause", "How many credit hours is a full teaching load, and what is the policy on make-up examinations?"),
    ("missing_clause", "What is the phone number of the Human Resources Department and where is the Main Library?"),
    ("mixed", "How many days of maternity leave are female faculty entitled to, and how many days of annual leave do faculty members get?"),
    ("mixed", "How many days of sick leave with full pay are faculty entitled to, and how many days of sick leave are faculty entitled to?"),
    ("cross_unresolved", "Who approves research leave?"),
    ("cross_unresolved", "Where is the Research Institute and what research funding is available?"),
    ("handoff", "Who handles faculty visas and where is that office?"),
    ("handoff", "Can I reduce my teaching load if I have a funded research project, and who approves it?"),
]

T0 = time.perf_counter()
from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot import plan_f_runtime as PF                        # noqa: E402
from handbook_bot import plan_f_aggregation as AG                    # noqa: E402
from handbook_bot.agents.coordinator import coordinate              # noqa: E402
from handbook_bot.agents.specialists.dispatch import dispatch       # noqa: E402
from ui import pipeline                                             # noqa: E402

t = time.perf_counter()
kb = build_knowledge_base(verbose=False)
build_s = time.perf_counter() - t
runtime, build = PF.build_runtime(kb)
say("imports+kb %.1f s (kb %.1f s): %d pages, %d chunks; registry ok=%s %.0f ms; groq client=%s" % (
    time.perf_counter() - T0, build_s, len(kb.pages), len(kb.chunks), build.ok, build.ms, kb.groq_client is not None))

# the I-3 question set and the I-4 probes, read from their runners so the phases share them exactly
src = open(os.path.join(REPO, "qa", "plan_f_i3", "runtime_measure.py"), encoding="utf-8").read()
QS = eval(src[src.index("QS = [") + 5:src.index("]\nsay(\"questions:\"") + 1])
src4 = open(os.path.join(REPO, "qa", "plan_f_i4", "reverify.py"), encoding="utf-8").read()
PROBES = eval(src4[src4.index("PROBES = [") + 9:src4.index(")]\nI3 =") + 2])
I4 = json.load(open(os.path.join(REPO, "qa", "plan_f_i4", "results_2026-10-30.json"), encoding="utf-8"))
i4_served = {r["question"]: r for r in I4["results"] if r["mode"] == "active" and (r.get("diagnostics") or {}).get("served") == "plan_f"}
i4_direct = {r["question"]: r for r in I4["direct"]}
ALL = [("corpus", d, q) for d, q in QS] + [("probe", d, q) for d, q in PROBES] + [("aggregation_set", d, q) for d, q in AGGREGATION_SET]
say("questions: %d corpus + %d probes + %d aggregation set; I-4 served: %d" % (len(QS), len(PROBES), len(AGGREGATION_SET), len(i4_served)))


def items_of(result):
    return [{"idx": i.get("idx"), "page": i["meta"].get("page"), "chunk": i.get("chunk"), "plan_f": i["meta"].get("plan_f")} for i in (result.items if result else [])]


# ---- 1. decision shape, every dispatched run, the aggregator's direct verdict, the adapter attempt -----------------------
direct = []
for n, (group, domain, q) in enumerate(ALL, 1):
    rec = {"n": n, "group": group, "domain": domain, "question": q, "i4_served": q in i4_served,
           "i4_outcome": ((i4_direct.get(q) or {}).get("attempt") or {}).get("outcome")}
    decision = coordinate(q)
    md = decision.metadata
    rec["decision"] = {"clauses": AG.request_clauses(decision), "tasks": [(t.specialist_id.value, t.context["focus_clauses"], t.metadata.get("scoped")) for t in decision.subtasks],
                       "complexity": md.get("complexity"), "is_multipart": md["router"]["is_multipart"], "requires_synthesis": decision.requires_synthesis,
                       "uncovered": md.get("uncovered_clauses"), "dropped": md.get("dropped_domains"), "decision_problem": PF.decision_problem(decision),
                       "applicable": AG.applicable(decision)}
    problem = AG.request_problem(decision) if AG.applicable(decision) else None
    rec["decision"]["request_problem"] = [problem[0].value, problem[1]] if problem else None
    multi = len(rec["decision"]["clauses"]) >= 2 or len(decision.subtasks) >= 2
    if multi and problem is None:
        dispatched = dispatch(decision, runtime.registry, question=q)
        rec["runs"] = [{"task_id": r.task.task_id, "specialist": r.specialist_id.value, "state": r.state,
                        "status": r.findings.status.value if r.findings else None, "findings": len(r.findings.findings) if r.findings else 0,
                        "handoffs": [h.requested_specialist.value for h in r.findings.handoff_requests] if r.findings else [],
                        "reports": [{k: rep.get(k) for k in ("index", "outcome", "ask_shape", "ask_kind", "relation", "evidence_tier_reached", "uncovered_details")}
                                    for rep in ((r.findings.metadata or {}).get("clauses") or [])] if r.findings else [],
                        "result_problem": PF.result_problem(r.findings) if r.findings else None, "ms": r.findings.ms if r.findings else None}
                       for r in dispatched.runs]
        t = time.perf_counter()
        outcome = AG.aggregate(decision, dispatched, kb.chunks, kb.metadata)
        rec["direct_aggregate_ms"] = (time.perf_counter() - t) * 1000
        rec["aggregate"] = outcome.result.to_dict()
        rec["aggregate"]["timings"] = dict(outcome.result.timings)
        rec["aggregate"]["parts"] = [{"clause_index": p.clause_index, "finding_id": p.finding.finding_id, "page": p.finding.page,
                                      "chunk_id": p.finding.metadata.get("chunk_id"), "quote": p.finding.evidence_quote} for p in outcome.parts]
    a = runtime.attempt(q)
    rec["attempt"] = {"outcome": a.outcome.value, "path": a.path, "reason": a.reason, "specialist": a.specialist_id, "status": a.status, "timings": a.timings,
                      "verified": a.verified, "verification": a.verification.to_dict() if a.verification is not None else None,
                      "aggregation": a.aggregation.to_dict() if a.aggregation is not None else None,
                      "answer": a.result.answer if a.result else None, "pages": a.result.pages if a.result else None, "items": items_of(a.result),
                      "sub_questions": [(s.question, s.route) for s in a.result.sub_questions] if a.result else [],
                      "llm_calls": a.result.llm_calls if a.result else 0, "used_llm": any(e.used_llm for e in a.trace),
                      "dispatch_llm_calls": a.dispatch.llm_calls if a.dispatch is not None else 0,
                      "trace": [(e.agent, e.decision) for e in a.trace]}
    if q in i4_served:
        prev = i4_served[q]
        rec["i4_answer_equal"] = (a.result is not None and a.result.answer == prev["answer"] and list(a.result.pages) == list(prev["pages"]))
    direct.append(rec)
    agg = rec.get("aggregate") or {}
    say("[%3d] %-15s %-13s %-34s path=%-26s agg=%-24s | %s" % (n, group, domain, a.outcome.value, a.path, agg.get("reason") or rec["decision"].get("request_problem", [None])[0] if rec["decision"].get("request_problem") else agg.get("reason"), q[:58]))

# ---- 2. pipeline in SHADOW and ACTIVE (legacy answer compared with the direct call in SHADOW) ------------------------------
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
        say("[%3d] %-15s wall=%5.0f served=%-7s path=%-26s outcome=%-34s agg=%-24s %s" % (
            n, group, wall, (d or {}).get("served"), (d or {}).get("plan_f_path"), (d or {}).get("outcome"), (d or {}).get("aggregation_reason"), q[:50]))

json.dump({"kb": {"pages": len(kb.pages), "chunks": len(kb.chunks), "build_s": build_s, "registry_ms": build.ms,
                  "groq_client_configured": kb.groq_client is not None},
           "aggregation_set": AGGREGATION_SET, "direct": direct, "results": results, "total_script_s": time.perf_counter() - T0},
          open(os.path.join(OUT, "i5_results.json"), "w", encoding="utf-8"), indent=1, default=str)
say("done %.1f s" % (time.perf_counter() - T0))
