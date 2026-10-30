"""I-4 re-evaluation on the REAL knowledge base (production build path, real models, real retrieval). Writes JSON to OUT.

For every question of the I-3 set (plus the I-4 probes) the script records: the findings-aware verifier's own verdict on the
dispatched specialist result (called directly, so even results the I-1 gate stops are classified), the adapter outcome with
its verification record, and the pipeline answer in SHADOW and ACTIVE mode with the I-2/I-4 diagnostics. No .env is touched; the
Plan F switches are set on the config module in this process only. Plan F makes no model call."""
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
log = open(os.path.join(OUT, "i4_log.txt"), "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


T0 = time.perf_counter()
from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot import plan_f_runtime as PF                        # noqa: E402
from handbook_bot import plan_f_verification as V                    # noqa: E402
from handbook_bot.agents.coordinator import coordinate              # noqa: E402
from handbook_bot.agents.specialists.dispatch import dispatch       # noqa: E402
from ui import pipeline                                             # noqa: E402

t = time.perf_counter()
kb = build_knowledge_base(verbose=False)
build_s = time.perf_counter() - t
runtime, build = PF.build_runtime(kb)
say("imports+kb %.1f s (kb %.1f s): %d pages, %d chunks; registry ok=%s %.0f ms; groq client=%s" % (
    time.perf_counter() - T0, build_s, len(kb.pages), len(kb.chunks), build.ok, build.ms, kb.groq_client is not None))

# the I-3 question set, read from the I-3 runner so the two phases share it exactly
src = open(os.path.join(REPO, "qa", "plan_f_i3", "runtime_measure.py"), encoding="utf-8").read()
QS = eval(src[src.index("QS = [") + 5:src.index("]\nsay(\"questions:\"") + 1])
PROBES = [("probe", q) for q in (
    "What information does HR provide to new faculty members?",
    "How many days of maternity leave with full pay are female faculty entitled to?",
    "What is the fax number of the Human Resources Department?",
    "What is the phone number of the Passport Unit?",
    "How many credit hours per semester is the teaching load for lecturers?",
    "How many theses can a faculty member supervise?",
    "What percentage of their time can faculty spend on consultancy?",
    "What is the Technology Transfer Office?",
    "what is the phone number of human resources",
    "How many working days of sick leave with full pay are faculty entitled to?",
    "How many calendar days of sick leave are faculty entitled to?",
    "How many days of maternity leave are part-time female faculty entitled to?",
    "Does HR arrange travel for visiting faculty members?",
    "Who approves the funding ceiling for research grants?",
)]
I3 = json.load(open(os.path.join(REPO, "qa", "plan_f_i3", "results_2026-10-30.json"), encoding="utf-8"))
i3_accepted = {r["question"]: r for r in I3["results"] if r["mode"] == "active" and (r.get("diagnostics") or {}).get("served") == "plan_f"}
say("questions: %d + %d probes; I-3 accepted: %d" % (len(QS), len(PROBES), len(i3_accepted)))

# ---- 1. direct verifier verdict on every dispatched result + adapter outcome -------------------------------------
direct = []
for n, (domain, q) in enumerate(QS + PROBES, 1):
    rec = {"n": n, "domain": domain, "question": q, "i3_accepted": q in i3_accepted}
    decision = coordinate(q)
    rec["decision_problem"] = PF.decision_problem(decision)
    if rec["decision_problem"] is None:
        clause = decision.subtasks[0].context["focus_clauses"][0]
        run = dispatch(decision, runtime.registry, question=q).runs[0]
        if run.findings is not None:
            t = time.perf_counter()
            v = V.verify_findings(clause, run.findings, kb.chunks, kb.metadata)
            rec["direct_verify_ms"] = (time.perf_counter() - t) * 1000
            rec["direct"] = v.to_dict()
            rec["specialist"] = run.findings.specialist_id.value
            rec["status"] = run.findings.status.value
            rec["i1_result_problem"] = PF.result_problem(run.findings)
    a = runtime.attempt(q)
    rec["attempt"] = {"outcome": a.outcome.value, "reason": a.reason, "specialist": a.specialist_id, "status": a.status, "timings": a.timings,
                      "verification": a.verification.to_dict() if a.verification is not None else None,
                      "answer": a.result.answer if a.result else None, "pages": a.result.pages if a.result else None,
                      "items": [{"idx": i.get("idx"), "page": i["meta"].get("page"), "chunk": i.get("chunk")} for i in (a.result.items if a.result else [])],
                      "llm_calls": a.result.llm_calls if a.result else 0, "used_llm": any(e.used_llm for e in a.trace)}
    direct.append(rec)
    d = rec.get("direct") or {}
    say("[%2d] %-13s %-38s direct=%-24s kind=%-12s comp=%-20s | %s" % (n, domain, a.outcome.value, d.get("reason"), d.get("request_kind"),
                                                                       d.get("completeness"), q[:60]))

# ---- 2. pipeline in SHADOW and ACTIVE (legacy answer compared with the direct call in SHADOW) ----------------------
from handbook_bot.qa import answer_question                         # noqa: E402


def visible(r):
    return {"answer": r.answer, "pages": list(r.pages), "best_section": r.best_section, "query_type": r.query_type, "evidence": r.evidence,
            "items": [(i.get("idx"), i["meta"].get("page"), i.get("chunk")[:60]) for i in r.items], "used_llm": r.used_llm, "llm_calls": r.llm_calls,
            "retried": r.retried, "sub_questions": list(r.sub_questions), "num_candidates": r.num_candidates, "num_reranked": r.num_reranked}


results = []
for mode in ("shadow", "active"):
    config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = (mode == "active"), (mode == "shadow")
    pipeline.reset_plan_f_state()
    say("\n===== MODE %s (resolved %s) =====" % (mode, pipeline.plan_f_mode()))
    for n, (domain, q) in enumerate(QS + PROBES, 1):
        t = time.perf_counter()
        try:
            res, d = pipeline.ask_with_diagnostics(kb, q)
            exc = None
        except Exception as e:
            res, d, exc = None, None, PF.redact_error(e)
        wall = (time.perf_counter() - t) * 1000
        rec = {"mode": mode, "n": n, "domain": domain, "question": q, "wall_ms": wall, "exception": exc}
        if res is not None:
            rec.update({"answer": res.answer, "pages": list(res.pages), "best_section": res.best_section, "query_type": res.query_type,
                        "used_llm": res.used_llm, "llm_calls": res.llm_calls, "retried": res.retried, "timings": res.timings,
                        "trace": [(e.agent, e.decision) for e in res.agent_trace], "diagnostics": d,
                        "items": [{"idx": i.get("idx"), "page": i["meta"].get("page"), "chunk": i.get("chunk")} for i in res.items]})
            if mode == "shadow":
                direct_res = answer_question(q, embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata,
                                             pages=kb.pages, groq_client=kb.groq_client)
                rec["identical_to_direct"] = visible(res) == visible(direct_res)
        results.append(rec)
        say("[%2d] %-13s wall=%5.0f served=%-7s outcome=%-36s fv=%-24s %s" % (n, domain, wall, (d or {}).get("served"), (d or {}).get("outcome"),
                                                                            (d or {}).get("findings_reason"), q[:55]))

json.dump({"kb": {"pages": len(kb.pages), "chunks": len(kb.chunks), "build_s": build_s, "registry_ms": build.ms,
                  "groq_client_configured": kb.groq_client is not None},
           "direct": direct, "results": results, "total_script_s": time.perf_counter() - T0},
          open(os.path.join(OUT, "i4_results.json"), "w", encoding="utf-8"), indent=1, default=str)
say("done %.1f s" % (time.perf_counter() - T0))
