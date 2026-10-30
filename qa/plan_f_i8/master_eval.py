"""I-8 master evaluation on the REAL knowledge base through the real application seam (``ui.pipeline``): the integrated
runtime (Coordinator, routing refinement, specialists, retrieval, reranking, guards, findings verification, aggregation,
handoffs, legacy Verifier, fallback) in every operating mode. Writes ``i8_results.json`` and ``i8_log.txt`` to OUT.

    python -X utf8 qa/plan_f_i8/master_eval.py <repo> <output dir> [--direct-only]

Passes, in order (the registry is built lazily by the seam, so the first ACTIVE question is the cold request):
    1. ACTIVE   every question of the master set through ``pipeline.ask_with_diagnostics`` (served path, outcome, answer,
                pages, evidence items with citation checks, diagnostics, wall time)
    2. DIRECT   the Coordinator decision, the routing refinement and the adapter attempt of every question on the SAME
                runtime object the seam built (full routing, handoff, verification and aggregation records, specialist runs)
    3. SHADOW   every question through the seam in SHADOW mode, compared field by field with the direct legacy call
    4. REPEAT   the determinism subset three more times in ACTIVE mode
    5. LEGACY   the representative sample in LEGACY mode in this process (minimal diagnostics, no runtime cache); the
                import isolation of a fresh LEGACY process is checked by ``legacy_check.py``
The switches are set on the config module in this process only; no .env is read or written; Plan F makes no model call.
Diagnostics never carry the question or the answer text (checked by analyze.py)."""
import json
import os
import re
import sys
import time

REPO, OUT = sys.argv[1], sys.argv[2]
DIRECT_ONLY = "--direct-only" in sys.argv[3:]                   # rerun only pass 2 and merge it into OUT/i8_results.json
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "qa", "plan_f_i8"))
os.chdir(REPO)
os.environ.pop("PLAN_F_ENABLED", None)
os.environ.pop("PLAN_F_SHADOW", None)
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "i8_log.txt"), "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


from question_set import QUESTIONS, DETERMINISM_IDS, LEGACY_SAMPLE_IDS, by_id   # noqa: E402

T0 = time.perf_counter()
from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot.qa import answer_question                         # noqa: E402
from ui import pipeline                                             # noqa: E402

t = time.perf_counter()
kb = build_knowledge_base(verbose=False)
KB_BUILD_S = time.perf_counter() - t
say("imports+kb %.1f s (kb %.1f s): %d pages, %d chunks; groq client=%s; questions=%d" % (
    time.perf_counter() - T0, KB_BUILD_S, len(kb.pages), len(kb.chunks), kb.groq_client is not None, len(QUESTIONS)))
PAGE_TEXT = {p["page"]: p["text"] for p in kb.pages}
_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub(" ", s or "").strip()


def items_of(result):
    """Evidence items with the citation checks of section 15: the quote is an exact substring of the production chunk it
    names, that chunk's metadata page is the cited page, and the quote occurs in the printed page's text."""
    out = []
    for i in (result.items if result else []):
        idx, quote, meta = i.get("idx"), i.get("chunk") or "", i.get("meta") or {}
        page = meta.get("page")
        chunk = kb.chunks[idx] if isinstance(idx, int) and 0 <= idx < len(kb.chunks) else None
        meta_page = kb.metadata[idx].get("page") if isinstance(idx, int) and 0 <= idx < len(kb.metadata) else None
        text = PAGE_TEXT.get(page, "")
        out.append({"idx": idx, "page": page, "section": meta.get("section_title") or meta.get("section"), "source": meta.get("source_title"),
                    "quote": quote, "chunk": chunk, "plan_f": meta.get("plan_f"),
                    "quote_in_chunk": chunk is not None and quote in chunk, "meta_page_is_cited_page": meta_page == page,
                    "quote_in_page": bool(quote) and quote in text, "quote_in_page_normalized": bool(quote) and norm(quote) in norm(text)})
    return out


def visible(r):
    return {"answer": r.answer, "pages": list(r.pages), "best_section": r.best_section, "query_type": r.query_type, "evidence": r.evidence,
            "items": [(i.get("idx"), (i.get("meta") or {}).get("page"), (i.get("chunk") or "")[:80]) for i in r.items], "used_llm": r.used_llm,
            "llm_calls": r.llm_calls, "retried": r.retried, "sub_questions": [(s.question, s.route) for s in r.sub_questions],
            "num_candidates": r.num_candidates, "num_reranked": r.num_reranked}


def redact(exc):
    try:
        from handbook_bot.agents.specialists.base import redact_error
        return redact_error(exc)
    except Exception:
        return type(exc).__name__


def ask(q):
    """One seam call: (result or None, diagnostics or None, redacted exception or None, wall ms)."""
    t = time.perf_counter()
    try:
        res, d = pipeline.ask_with_diagnostics(kb, q)
        exc = None
    except Exception as e:                                          # the seam must never raise; recorded if it does
        res, d, exc = None, None, redact(e)
    return res, d, exc, (time.perf_counter() - t) * 1000


def record(mode, entry, res, d, exc, wall, n):
    rec = {"mode": mode, "id": entry["id"], "n": n, "domain": entry["domain"], "category": entry["category"], "question": entry["q"],
           "wall_ms": wall, "exception": exc}
    if res is not None:
        rec.update({"answer": res.answer, "pages": list(res.pages), "best_section": res.best_section, "query_type": res.query_type,
                    "used_llm": res.used_llm, "llm_calls": res.llm_calls, "retried": res.retried, "timings": res.timings,
                    "sub_questions": [(s.question, s.route) for s in res.sub_questions], "trace": [(e.agent, e.decision) for e in res.agent_trace],
                    "diagnostics": d, "items": items_of(res), "num_candidates": res.num_candidates, "num_reranked": res.num_reranked})
    return rec


def set_mode(mode):
    config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = (mode == "active"), (mode == "shadow")
    pipeline.reset_plan_f_state()
    say("\n===== MODE %s (resolved %s) =====" % (mode, pipeline.plan_f_mode()))


results = []
PREVIOUS = json.load(open(os.path.join(OUT, "i8_results.json"), encoding="utf-8")) if DIRECT_ONLY else None
# ---- 1. ACTIVE -----------------------------------------------------------------------------------------------------------
set_mode("active")
cold = None
for n, entry in enumerate(QUESTIONS if not DIRECT_ONLY else QUESTIONS[:1], 1):
    res, d, exc, wall = ask(entry["q"])
    rec = record("active", entry, res, d, exc, wall, n)
    if cold is None:
        cold = {"first_request_wall_ms": wall, "registry_build_ms": (d or {}).get("registry_build_ms"), "registry_reused": (d or {}).get("registry_reused"),
                "plan_f_ms": (d or {}).get("plan_f_ms")}
    results.append(rec)
    say("[%3d] %-4s wall=%6.0f served=%-7s path=%-26s outcome=%-34s route=%-30s calls=%s handoff=%-26s exc=%s | %s" % (
        n, entry["id"], wall, (d or {}).get("served"), (d or {}).get("plan_f_path"), (d or {}).get("outcome"), (d or {}).get("routing_reason"),
        (d or {}).get("specialist_calls"), (d or {}).get("handoff_outcome"), exc, entry["q"][:48]))

# ---- 2. DIRECT (same runtime object the seam built; warm) -----------------------------------------------------------------
from handbook_bot import plan_f_runtime as PF                        # noqa: E402
from handbook_bot import plan_f_routing as RT                        # noqa: E402
from handbook_bot.agents.coordinator import coordinate              # noqa: E402

entry_cache = pipeline._plan_f_cache().get_or_build(kb)
runtime = entry_cache.runtime
say("\n===== DIRECT (runtime reused=%s, cache builds=%d) =====" % (entry_cache.reused, pipeline._plan_f_cache().builds))
direct = []
for n, entry in enumerate(QUESTIONS, 1):
    q = entry["q"]
    rec = {"n": n, "id": entry["id"], "domain": entry["domain"], "category": entry["category"], "question": q, "exception": None}
    try:
        decision = coordinate(q)
        refined, routing = RT.refine(decision)
        rec["decision"] = {"clauses": [c for c in decision.metadata["analysis"]["clauses"]],
                           "tasks": [(t.specialist_id.value, t.context["focus_clauses"]) for t in decision.subtasks],
                           "complexity": decision.metadata.get("complexity"), "is_multipart": decision.metadata["router"]["is_multipart"],
                           "decision_problem": PF.decision_problem(decision)}
        rec["routing"] = routing.to_dict()
        rec["routing"]["ms"] = routing.ms
        rec["refined_tasks"] = [(t.specialist_id.value, t.context["focus_clauses"]) for t in refined.subtasks]
        t = time.perf_counter()
        a = runtime.attempt(q)
        rec["attempt_wall_ms"] = (time.perf_counter() - t) * 1000
        rec["attempt"] = {"outcome": a.outcome.value, "path": a.path, "reason": a.reason, "specialist": a.specialist_id, "status": a.status, "timings": a.timings,
                          "verified": a.verified, "specialist_calls": a.specialist_calls, "handoff_outcome": a.handoff_outcome,
                          "routing": a.routing.to_dict() if a.routing is not None else None, "handoff": a.handoff.to_dict() if a.handoff is not None else None,
                          "verification": a.verification.to_dict() if a.verification is not None else None,
                          "aggregation": a.aggregation.to_dict() if a.aggregation is not None else None,
                          "answer": a.result.answer if a.result else None, "pages": a.result.pages if a.result else None, "items": items_of(a.result),
                          "llm_calls": a.result.llm_calls if a.result else 0, "used_llm": any(e.used_llm for e in a.trace),
                          "dispatch_llm_calls": a.dispatch.llm_calls if a.dispatch is not None else 0,
                          "runs": [(r.specialist_id.value, r.state, r.findings.status.value if r.findings else None, r.reason[:60]) for r in (a.dispatch.runs if a.dispatch else [])],
                          "trace": [(e.agent, e.decision) for e in a.trace]}
    except Exception as e:
        rec["exception"] = redact(e)
    direct.append(rec)
    a = rec.get("attempt") or {}
    say("[%3d] %-4s %-34s path=%-26s route=%-30s final=%-28s handoff=%-30s calls=%s exc=%s | %s" % (
        n, entry["id"], a.get("outcome"), a.get("path"), (rec.get("routing") or {}).get("reason"), "+".join((a.get("routing") or {}).get("final_specialists", [])),
        a.get("handoff_outcome"), a.get("specialist_calls"), rec["exception"], q[:40]))

if DIRECT_ONLY:
    PREVIOUS["direct"] = direct
    json.dump(PREVIOUS, open(os.path.join(OUT, "i8_results.json"), "w", encoding="utf-8"), indent=1, default=str)
    say("direct pass merged into the previous results; done %.1f s" % (time.perf_counter() - T0))
    sys.exit(0)

# ---- 3. SHADOW -----------------------------------------------------------------------------------------------------------
set_mode("shadow")
shadow_cold = None
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["q"])
    rec = record("shadow", entry, res, d, exc, wall, n)
    if shadow_cold is None:
        shadow_cold = {"first_request_wall_ms": wall, "registry_build_ms": (d or {}).get("registry_build_ms"), "registry_reused": (d or {}).get("registry_reused")}
    if res is not None:
        try:
            t = time.perf_counter()
            direct_res = answer_question(entry["q"], embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata,
                                         pages=kb.pages, groq_client=kb.groq_client)
            rec["direct_legacy_wall_ms"] = (time.perf_counter() - t) * 1000
            va, vb = visible(res), visible(direct_res)
            rec["identical_to_direct"] = va == vb
            rec["direct_mismatch_fields"] = [k for k in va if va[k] != vb[k]]
        except Exception as e:
            rec["identical_to_direct"] = None
            rec["direct_legacy_exception"] = redact(e)
    results.append(rec)
    say("[%3d] %-4s wall=%6.0f served=%-7s identical=%-5s outcome=%-34s exc=%s | %s" % (
        n, entry["id"], wall, (d or {}).get("served"), rec.get("identical_to_direct"), (d or {}).get("outcome"), exc, entry["q"][:48]))

# ---- 4. REPEAT (determinism, ACTIVE) ---------------------------------------------------------------------------------------
set_mode("active")
repeats = []
ids = by_id()
for rep in (1, 2, 3):
    for entry_id in DETERMINISM_IDS:
        entry = ids[entry_id]
        res, d, exc, wall = ask(entry["q"])
        rec = record("active", entry, res, d, exc, wall, 0)
        rec["rep"] = rep
        repeats.append(rec)
    say("determinism rep %d done" % rep)

# ---- 5. LEGACY sample (in-process) -----------------------------------------------------------------------------------------
set_mode("legacy")
legacy = []
for entry_id in LEGACY_SAMPLE_IDS:
    entry = ids[entry_id]
    res, d, exc, wall = ask(entry["q"])
    rec = record("legacy", entry, res, d, exc, wall, 0)
    rec["diagnostics_keys"] = sorted((d or {}).keys())
    rec["minimal_diagnostics"] = d == {"mode": "legacy", "served": "legacy", "attempted": False}
    legacy.append(rec)
    say("[LEGACY] %-4s wall=%6.0f route=%-10s diag=%s exc=%s | %s" % (entry["id"], wall, (res.query_type if res else None), rec["minimal_diagnostics"], exc, entry["q"][:48]))
legacy_cache_untouched = pipeline._PLAN_F_CACHE is None

json.dump({"kb": {"pages": len(kb.pages), "chunks": len(kb.chunks), "build_s": KB_BUILD_S, "groq_client_configured": kb.groq_client is not None},
           "cold": {"active": cold, "shadow": shadow_cold}, "questions": QUESTIONS, "direct": direct, "results": results, "repeats": repeats,
           "legacy": legacy, "legacy_cache_untouched": legacy_cache_untouched, "total_script_s": time.perf_counter() - T0},
          open(os.path.join(OUT, "i8_results.json"), "w", encoding="utf-8"), indent=1, default=str)
say("done %.1f s; legacy cache untouched=%s" % (time.perf_counter() - T0, legacy_cache_untouched))
