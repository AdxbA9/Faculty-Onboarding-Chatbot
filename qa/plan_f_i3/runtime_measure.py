"""I-3 measurement on the REAL knowledge base (production build path, real models, real retrieval). Writes JSON to OUT.
No .env is touched; the Plan F switches are set on the config module in this process only. Plan F makes no model call."""
import dataclasses, json, os, statistics, sys, time
REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO); os.chdir(REPO)
os.environ.pop("PLAN_F_ENABLED", None); os.environ.pop("PLAN_F_SHADOW", None)
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "i3_log.txt"), "w", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True); log.write(msg + "\n"); log.flush()


T0 = time.perf_counter()
from handbook_bot import config                                     # noqa: E402
from handbook_bot.config import CACHE_DIR, CACHE_VERSION, EMBED_MODEL, RERANK_MODEL  # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot.qa import answer_question                         # noqa: E402
from ui import pipeline                                             # noqa: E402

say("imports %.1f s; defaults PLAN_F_ENABLED=%s PLAN_F_SHADOW=%s mode=%s; plan_f loaded=%s" % (
    time.perf_counter() - T0, config.PLAN_F_ENABLED, config.PLAN_F_SHADOW, pipeline.plan_f_mode(), "handbook_bot.plan_f_runtime" in sys.modules))

# ---- models: download/load timing, separately from the build ------------------------------------------------------
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402
t = time.perf_counter(); SentenceTransformer(EMBED_MODEL); t_embed_model = time.perf_counter() - t
t = time.perf_counter(); CrossEncoder(RERANK_MODEL); t_rerank_model = time.perf_counter() - t
say("model download+load: embedder %.1f s, reranker %.1f s" % (t_embed_model, t_rerank_model))

# ---- real knowledge base: first build (embedding), then cache reuse ----------------------------------------------
cache_files_before = sorted(f for f in os.listdir(CACHE_DIR) if CACHE_VERSION in f) if os.path.isdir(CACHE_DIR) else []
t = time.perf_counter(); kb = build_knowledge_base(verbose=True); build1 = time.perf_counter() - t
cache_files_after = sorted(f for f in os.listdir(CACHE_DIR) if CACHE_VERSION in f)
t = time.perf_counter(); kb2 = build_knowledge_base(verbose=False); build2 = time.perf_counter() - t
import numpy as np                                                   # noqa: E402
kb_info = {
    "source_id": kb.source.source_id, "registered": kb.source.registered, "pdf_name": os.path.basename(kb.pdf_file),
    "pages": len(kb.pages), "chunks": len(kb.chunks), "stats": kb.stats, "embedding_shape": list(kb.embeddings.shape),
    "embedding_dtype": str(kb.embeddings.dtype), "index_type": type(kb.index).__name__, "index_ntotal": int(kb.index.ntotal),
    "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL, "cache_version": CACHE_VERSION,
    "cache_files_before_first_build": cache_files_before, "cache_files_after_first_build": cache_files_after,
    "build1_s": build1, "build2_s": build2, "second_build_embeddings_identical": bool(np.array_equal(kb.embeddings, kb2.embeddings)),
    "groq_client_configured": kb.groq_client is not None, "model_download_load_s": {"embedder": t_embed_model, "reranker": t_rerank_model},
    "chunk_types": {}, "metadata_keys": sorted(kb.metadata[0].keys()) if kb.metadata else [],
}
from collections import Counter                                      # noqa: E402
kb_info["chunk_types"] = dict(Counter(m.get("chunk_type") for m in kb.metadata))
say("KB: %d pages, %d chunks, section records %s, build1 %.1f s, build2 (cache) %.1f s, identical embeddings %s, groq client %s" % (
    kb_info["pages"], kb_info["chunks"], kb.stats.get("source", {}).get("section_records"), build1, build2,
    kb_info["second_build_embeddings_identical"], kb_info["groq_client_configured"]))

# ---- four real specialist builders, then the registry ---------------------------------------------------------------
from handbook_bot import plan_f_runtime as PF                        # noqa: E402
builders = {}
for sid, builder in PF.DEFAULT_BUILDERS.items():
    t = time.perf_counter()
    try:
        sp = builder(kb)
        ms = (time.perf_counter() - t) * 1000
        guard = getattr(sp, "guard", None)
        builders[sid.value] = {"ok": True, "ms": ms, "class": type(sp).__name__, "scope_page_ranges": sp.scope.page_ranges,
                               "shared_pages": len(getattr(guard, "shared_pages", {}) or {}), "unresolved_pages": list(getattr(guard, "unresolved_pages", []) or [])}
    except Exception as exc:
        builders[sid.value] = {"ok": False, "ms": (time.perf_counter() - t) * 1000, "error": PF.redact_error(exc)}
    say("builder %-16s %s" % (sid.value, json.dumps(builders[sid.value], default=str)[:200]))
t = time.perf_counter(); runtime, build = PF.build_runtime(kb); registry_ms = (time.perf_counter() - t) * 1000
say("registry via build_runtime: ok=%s ms=%.0f errors=%s" % (build.ok, registry_ms, build.errors))

# ---- question set ---------------------------------------------------------------------------------------------------
QS = [
    ("teaching", "How many credit hours is a full teaching load?"), ("teaching", "How many office hours must faculty hold?"),
    ("teaching", "What must instructors do with final examination answer sheets?"), ("teaching", "What is the Workload Allocation Model?"),
    ("teaching", "Is attendance mandatory for students?"), ("teaching", "Where must instructors post the syllabus?"),
    ("teaching", "How are final grades entered into Blackboard?"), ("teaching", "What is the teaching load for lecturers?"),
    ("teaching", "When must final grades be submitted after the examination?"), ("teaching", "Can faculty change a grade after it has been submitted?"),
    ("teaching", "What is the policy on make-up examinations?"), ("teaching", "What percentage of absences bars a student from the final examination?"),
    ("research", "How many theses can a faculty member supervise at once?"), ("research", "Who determines the funding ceiling for research grants?"),
    ("research", "Who reviews research grant applications?"), ("research", "How much petty cash advance can a principal investigator receive?"),
    ("research", "Which committee reviews research on human subjects?"), ("research", "What does the Technology Transfer Office do?"),
    ("research", "What does the Research Funding Department do?"), ("research", "How many days per week can faculty spend on consultancy?"),
    ("research", "What are the research priority areas?"), ("research", "Does the University provide seed funding for research?"),
    ("research", "Who approves research leave?"), ("research", "Who owns intellectual property created by faculty?"),
    ("hr", "Does HR arrange travel and hotel accommodations for new faculty members?"), ("hr", "Who drafts employment offers?"),
    ("hr", "How many days of maternity leave are female faculty entitled to?"), ("hr", "What does HR provide to new faculty members?"),
    ("hr", "Does HR assist new faculty members in obtaining entry visas?"), ("hr", "How many days of sick leave with full pay are faculty entitled to?"),
    ("hr", "How many days of annual leave do faculty members get?"), ("hr", "What does Human Resources do?"), ("hr", "What does HR do?"),
    ("hr", "Is maternity leave paid?"), ("hr", "How much is the furniture allowance?"), ("hr", "Does the University provide health insurance for faculty?"),
    ("hr", "What is the notice period for resignation?"), ("hr", "Who approves Hajj leave?"),
    ("institutional", "What is the phone number of the Human Resources Department?"), ("institutional", "What is the fax number of the Passport Unit?"),
    ("institutional", "What is the telephone number of the Registration office?"), ("institutional", "What is the phone number of the Finance Department?"),
    ("institutional", "What is the phone number of the Information Technology Center?"), ("institutional", "How many colleges does the University of Sharjah have?"),
    ("institutional", "Where is the Main Library?"), ("institutional", "Who is responsible for health and safety on campus?"),
    ("institutional", "What is the Health and Safety Office responsible for?"), ("institutional", "Who chairs the University Council?"),
    ("institutional", "What is the mission of the University of Sharjah?"), ("institutional", "Where is the Men's Main Library?"),
    ("institutional", "What is the email address of the IT Center?"), ("institutional", "What is the fax number of the Dental Hospital Call Center?"),
    ("special", "Hello"), ("special", "What is the purpose of the Faculty Handbook?"), ("special", "asdfgh qwerty zxcvb"),
    ("special", "Is there a deadline for add/drop?"), ("special", "Who handles faculty visas and where is that office?"),
    ("special", "Where is the Research Institute and what research funding is available?"),
    ("special", "Can I reduce my teaching load if I have a funded research project, and who approves it?"),
    ("special", "Which office handles faculty housing?"), ("special", "Who approves conference travel?"), ("special", "What is the capital of France?"),
]
say("questions:", len(QS))


def visible(r):
    return {"answer": r.answer, "pages": list(r.pages), "best_section": r.best_section, "query_type": r.query_type, "evidence": r.evidence,
            "items": [(i.get("idx"), i["meta"].get("page"), i.get("chunk")[:60]) for i in r.items], "used_llm": r.used_llm,
            "llm_calls": r.llm_calls, "retried": r.retried, "sub_questions": list(r.sub_questions), "num_candidates": r.num_candidates,
            "num_reranked": r.num_reranked}


def legacy_direct(q):
    return answer_question(q, embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks, metadata=kb.metadata, pages=kb.pages,
                           groq_client=kb.groq_client)


results = []
for mode in ("legacy", "shadow", "active"):
    config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = (mode == "active"), (mode == "shadow")
    pipeline.reset_plan_f_state()
    say("\n===== MODE %s (resolved %s) =====" % (mode, pipeline.plan_f_mode()))
    for n, (domain, q) in enumerate(QS, 1):
        t = time.perf_counter()
        try:
            res, d = pipeline.ask_with_diagnostics(kb, q); exc = None
        except Exception as e:
            res, d, exc = None, None, PF.redact_error(e)
        wall = (time.perf_counter() - t) * 1000
        rec = {"mode": mode, "n": n, "domain": domain, "question": q, "wall_ms": wall, "exception": exc}
        if res is not None:
            rec.update({"answer": res.answer, "pages": list(res.pages), "best_section": res.best_section, "query_type": res.query_type,
                        "used_llm": res.used_llm, "llm_calls": res.llm_calls, "retried": res.retried, "timings": res.timings,
                        "trace": [(e.agent, e.decision) for e in res.agent_trace], "diagnostics": d,
                        "items": [{"idx": i.get("idx"), "page": i["meta"].get("page"), "chunk": i.get("chunk")} for i in res.items]})
            if mode in ("legacy", "shadow"):
                t2 = time.perf_counter(); direct = legacy_direct(q); rec["direct_ms"] = (time.perf_counter() - t2) * 1000
                rec["identical_to_direct"] = visible(res) == visible(direct)
                rec["direct_trace_equal"] = [(e.agent, e.decision) for e in res.agent_trace] == [(e.agent, e.decision) for e in direct.agent_trace]
        results.append(rec)
        say("%2d %-13s %-70s served=%-7s outcome=%-30s spec=%-16s wall=%6.0f ms%s" % (
            n, domain, q[:70], (d or {}).get("served"), (d or {}).get("outcome"), (d or {}).get("specialist_id"), wall,
            "" if exc is None else " EXC " + exc))
    cache = pipeline._PLAN_F_CACHE
    say("cache builds in this mode: %s" % (None if cache is None else cache.builds))

# ---- cache on the real KB: a second knowledge-base object ------------------------------------------------------------
config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = False, True
pipeline.reset_plan_f_state()
q = "What is the phone number of the Human Resources Department?"
t = time.perf_counter(); _, d1 = pipeline.ask_with_diagnostics(kb, q); w1 = (time.perf_counter() - t) * 1000
t = time.perf_counter(); _, d2 = pipeline.ask_with_diagnostics(kb, q); w2 = (time.perf_counter() - t) * 1000
kb_other = dataclasses.replace(kb)                                  # another object, same contents
t = time.perf_counter(); _, d3 = pipeline.ask_with_diagnostics(kb_other, q); w3 = (time.perf_counter() - t) * 1000
t = time.perf_counter(); _, d4 = pipeline.ask_with_diagnostics(kb2, q); w4 = (time.perf_counter() - t) * 1000
cache = pipeline._PLAN_F_CACHE
cache_info = {"first": {"wall_ms": w1, "reused": d1["registry_reused"], "build_ms": d1["registry_build_ms"], "plan_f_ms": d1["plan_f_ms"]},
              "second": {"wall_ms": w2, "reused": d2["registry_reused"], "plan_f_ms": d2["plan_f_ms"]},
              "other_object": {"wall_ms": w3, "reused": d3["registry_reused"], "build_ms": d3["registry_build_ms"]},
              "second_kb_from_cache": {"wall_ms": w4, "reused": d4["registry_reused"], "build_ms": d4["registry_build_ms"]},
              "builds": cache.builds, "entries": len(cache),
              "runtimes_distinct": cache.get_or_build(kb).runtime is not cache.get_or_build(kb_other).runtime,
              "registries_distinct": cache.get_or_build(kb).runtime.registry is not cache.get_or_build(kb_other).runtime.registry}
say("cache:", json.dumps(cache_info, default=str))
pipeline.reset_plan_f_state(); config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = False, False

json.dump({"kb": kb_info, "builders": builders, "registry_ms": registry_ms, "registry_build": build.to_dict(), "results": results, "cache": cache_info,
           "total_script_s": time.perf_counter() - T0}, open(os.path.join(OUT, "i3_results.json"), "w", encoding="utf-8"), indent=1, default=str)
say("done in %.0f s" % (time.perf_counter() - T0))
