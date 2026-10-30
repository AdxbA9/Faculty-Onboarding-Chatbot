"""PF-2 guide evaluation on the REAL multi-source knowledge base through the real application seam (``ui.pipeline``).

    python -X utf8 qa/post_freeze_pf2/run_guide_eval.py <repo> <output dir>

Reads ``guide_questions.jsonl`` (next to this script) and writes ``results.json`` and ``log.txt`` to the output dir.
Passes, in order:
    1. ACTIVE        every question through ``pipeline.ask_with_diagnostics``: served path, Plan F outcome, answer, pages,
                     evidence items with their source ids and citation checks, diagnostics, wall time
    2. SHADOW        every question in SHADOW mode (legacy answer visible, Plan F diagnostics), compared field by field with
                     the direct legacy call
    3. ACTIVE+SEAM   every question in ACTIVE mode with ``PLAN_F_GUIDE_SCOPES`` on (the registry is rebuilt), to show what the
                     scope-extension seam changes on its own
    4. LEGACY        every question in LEGACY mode (what the unscoped legacy retrieval cites), plus the candidate pool by source
    5. REPEAT        the guide subset twice more in ACTIVE mode (determinism)
Baseline questions (``baseline_question``) are compared with the I-9 frozen run (``qa/plan_f_i8/results_2026-10-30.json``).
The switches are set on the config module in this process only; no .env is read or written; no model is called.
"""
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ.pop("PLAN_F_ENABLED", None)
os.environ.pop("PLAN_F_SHADOW", None)
os.environ.pop("PLAN_F_GUIDE_SCOPES", None)
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8")
T0 = time.perf_counter()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    LOG.write(line + "\n")
    LOG.flush()


from handbook_bot import config                                     # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot.qa import answer_question                         # noqa: E402
from handbook_bot.retrieval import classify_query, gather_candidates   # noqa: E402
from ui import pipeline                                             # noqa: E402

QUESTIONS = [json.loads(line) for line in open(os.path.join(HERE, "guide_questions.jsonl"), encoding="utf-8") if line.strip()]
BASELINE_PATH = os.path.join(REPO, "qa", "plan_f_i8", "results_2026-10-30.json")
BASELINE = {}
if os.path.exists(BASELINE_PATH):
    for rec in json.load(open(BASELINE_PATH, encoding="utf-8"))["results"]:
        if rec["mode"] == "active":
            BASELINE[rec["question"]] = rec

t = time.perf_counter()
kb = build_knowledge_base(verbose=False)
KB_BUILD_S = time.perf_counter() - t
say("imports+kb %.1f s (kb %.1f s): %d primary pages, %d chunks, %d sources, corpus %s; groq client=%s; questions=%d" % (
    time.perf_counter() - T0, KB_BUILD_S, len(kb.pages), len(kb.chunks), len(kb.sources), kb.corpus_key, kb.groq_client is not None, len(QUESTIONS)))
PAGE_TEXT = {(sid, p["page"]): p["text"] for sid, pages in kb.source_pages.items() for p in pages}
_WS = re.compile(r"\s+")


def norm(s):
    return _WS.sub(" ", s or "").strip()


def items_of(result):
    out = []
    for i in (result.items if result else []):
        idx, quote, meta = i.get("idx"), i.get("chunk") or "", i.get("meta") or {}
        page, sid = meta.get("page"), meta.get("source_id")
        chunk = kb.chunks[idx] if isinstance(idx, int) and 0 <= idx < len(kb.chunks) else None
        kmeta = kb.metadata[idx] if isinstance(idx, int) and 0 <= idx < len(kb.metadata) else {}
        text = PAGE_TEXT.get((sid, page), "")
        out.append({"idx": idx, "source_id": sid, "source_title": meta.get("source_title"), "page": page, "chunk_type": meta.get("chunk_type"),
                    "content_type": meta.get("content_type"), "source_conflict": meta.get("source_conflict"),
                    "section": meta.get("section_title") or meta.get("section"), "quote": quote[:400], "plan_f": meta.get("plan_f"),
                    "quote_in_chunk": chunk is not None and quote in chunk, "meta_matches": kmeta.get("page") == page and kmeta.get("source_id") == sid,
                    "quote_in_page_normalized": bool(quote) and norm(quote) in norm(text)})
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
    t = time.perf_counter()
    try:
        res, d = pipeline.ask_with_diagnostics(kb, q)
        exc = None
    except Exception as e:
        res, d, exc = None, None, redact(e)
    return res, d, exc, (time.perf_counter() - t) * 1000


DIAG_KEYS = ("mode", "served", "attempted", "outcome", "reason", "specialist_id", "status", "accepted", "fallback_reason", "plan_f_pages",
             "legacy_pages", "plan_f_sources", "legacy_sources", "plan_f_ms", "registry_ok", "registry_reused", "registry_build_ms", "registry_errors",
             "plan_f_llm_calls", "plan_f_used_llm", "legacy_llm_calls", "findings_verified", "findings_accepted", "findings_reason",
             "findings_completeness", "findings_request_kind", "plan_f_path", "aggregation_attempted", "aggregation_accepted", "aggregation_reason",
             "routing_refined", "routing_reason", "routing_initial", "routing_final", "handoff_attempted", "handoff_outcome", "specialist_calls", "error")


def record(mode, entry, res, d, exc, wall, n):
    rec = {"mode": mode, "id": entry["id"], "n": n, "category": entry["category"], "expected_source": entry.get("source"),
           "expected_pages": entry.get("pages"), "domain": entry["domain"], "question": entry["question"], "wall_ms": wall, "exception": exc}
    if res is not None:
        rec.update({"answer": res.answer, "pages": list(res.pages), "best_section": res.best_section, "query_type": res.query_type,
                    "used_llm": res.used_llm, "llm_calls": res.llm_calls, "retried": res.retried, "timings": res.timings,
                    "sub_questions": [(s.question, s.route) for s in res.sub_questions], "trace": [(e.agent, e.decision) for e in res.agent_trace],
                    "diagnostics": {k: (d or {}).get(k) for k in DIAG_KEYS} if d is not None else None, "items": items_of(res),
                    "num_candidates": res.num_candidates, "num_reranked": res.num_reranked})
        rec["served_sources"] = sorted({i["source_id"] for i in rec["items"] if i["source_id"]})
    return rec


def set_mode(mode, guide_scopes=False):
    config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = (mode == "active"), (mode == "shadow")
    config.PLAN_F_GUIDE_SCOPES = bool(guide_scopes)
    pipeline.reset_plan_f_state()
    say("\n===== MODE %s (resolved %s, guide scopes %s) =====" % (mode, pipeline.plan_f_mode(), guide_scopes))


def show(n, entry, rec, d, exc, extra=""):
    say("[%3d] %-4s wall=%6.0f served=%-7s outcome=%-36s route=%-26s pages=%-14s sources=%-62s exc=%s %s| %s" % (
        n, entry["id"], rec["wall_ms"], (d or {}).get("served"), (d or {}).get("outcome"), (d or {}).get("routing_reason"),
        str(rec.get("pages"))[:14], ",".join(s.replace("uos_", "") for s in rec.get("served_sources", [])), exc, extra, entry["question"][:44]))


results = []
# ---- 1. ACTIVE -------------------------------------------------------------------------------------------------------
set_mode("active")
cold = None
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["question"])
    rec = record("active", entry, res, d, exc, wall, n)
    if cold is None:
        cold = {"first_request_wall_ms": wall, "registry_build_ms": (d or {}).get("registry_build_ms"), "registry_reused": (d or {}).get("registry_reused"),
                "registry_ok": (d or {}).get("registry_ok"), "registry_errors": (d or {}).get("registry_errors")}
    if entry.get("baseline_question") and entry["question"] in BASELINE:
        b = BASELINE[entry["question"]]
        bd = b.get("diagnostics") or {}
        rec["baseline"] = {"id": b["id"], "served": bd.get("served"), "outcome": bd.get("outcome"), "answer_equal": b.get("answer") == rec.get("answer"),
                           "pages_equal": list(b.get("pages") or []) == list(rec.get("pages") or []), "served_equal": bd.get("served") == (d or {}).get("served"),
                           "outcome_equal": bd.get("outcome") == (d or {}).get("outcome")}
    results.append(rec)
    show(n, entry, rec, d, exc, extra=("baseline=%s " % all(v for k, v in rec["baseline"].items() if k.endswith("_equal"))) if "baseline" in rec else "")

# ---- 2. SHADOW -------------------------------------------------------------------------------------------------------
set_mode("shadow")
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["question"])
    rec = record("shadow", entry, res, d, exc, wall, n)
    if res is not None:
        try:
            direct_res = answer_question(entry["question"], embedder=kb.embedder, reranker=kb.reranker, index=kb.index, chunks=kb.chunks,
                                         metadata=kb.metadata, pages=kb.pages, groq_client=kb.groq_client)
            va, vb = visible(res), visible(direct_res)
            rec["identical_to_direct"] = va == vb
            rec["direct_mismatch_fields"] = [k for k in va if va[k] != vb[k]]
        except Exception as e:
            rec["identical_to_direct"] = None
            rec["direct_legacy_exception"] = redact(e)
    results.append(rec)
    show(n, entry, rec, d, exc, extra="identical=%s " % rec.get("identical_to_direct"))

# ---- 3. ACTIVE with the scope-extension seam on ----------------------------------------------------------------------
set_mode("active", guide_scopes=True)
seam_cold = None
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["question"])
    rec = record("active_seam", entry, res, d, exc, wall, n)
    if seam_cold is None:
        seam_cold = {"registry_ok": (d or {}).get("registry_ok"), "registry_errors": (d or {}).get("registry_errors"), "registry_build_ms": (d or {}).get("registry_build_ms")}
        try:
            build = pipeline._plan_f_cache().get_or_build(kb).build
            seam_cold["scope_extension"] = build.scope_extension
        except Exception as e:
            seam_cold["scope_extension_error"] = redact(e)
    results.append(rec)
    show(n, entry, rec, d, exc)

# ---- 4. LEGACY and the candidate pool --------------------------------------------------------------------------------
set_mode("legacy")
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["question"])
    rec = record("legacy", entry, res, d, exc, wall, n)
    rec["minimal_diagnostics"] = d == {"mode": "legacy", "served": "legacy", "attempted": False}
    try:
        q = entry["question"]
        emb = kb.embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        cands = gather_candidates(q, classify_query(q), emb, kb.index, kb.chunks, kb.metadata)
        pool = {}
        for c in cands:                                          # gather_candidates returns dicts: idx, chunk, meta, scores
            sid = (c.get("meta") or {}).get("source_id") if isinstance(c, dict) else kb.metadata[c[0]]["source_id"]
            pool[sid] = pool.get(sid, 0) + 1
        rec["candidate_pool_by_source"] = pool
        rec["candidate_pool_size"] = len(cands)
    except Exception as e:
        rec["candidate_pool_error"] = redact(e)
    results.append(rec)
    show(n, entry, rec, d, exc, extra="pool=%s " % {k.replace("uos_", ""): v for k, v in rec.get("candidate_pool_by_source", {}).items()})
legacy_cache_untouched = pipeline._PLAN_F_CACHE is None

# ---- 5. REPEAT (determinism, ACTIVE, guide subset) --------------------------------------------------------------------
set_mode("active")
repeats = []
subset = [e for e in QUESTIONS if e["category"] in ("guide", "cross_document", "wrong_source")]
for rep in (1, 2):
    for entry in subset:
        res, d, exc, wall = ask(entry["question"])
        rec = record("active", entry, res, d, exc, wall, 0)
        rec["rep"] = rep
        repeats.append(rec)
    say("determinism rep %d done (%d questions)" % (rep, len(subset)))

json.dump({"kb": {"primary_pages": len(kb.pages), "chunks": len(kb.chunks), "build_s": KB_BUILD_S, "corpus_key": kb.corpus_key,
                  "sources": kb.stats.get("sources"), "chunk_types": kb.stats.get("chunk_types"), "groq_client_configured": kb.groq_client is not None},
           "cold": {"active": cold, "active_seam": seam_cold}, "questions": QUESTIONS, "results": results, "repeats": repeats,
           "legacy_cache_untouched": legacy_cache_untouched, "baseline_available": bool(BASELINE), "total_script_s": time.perf_counter() - T0},
          open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, default=str)
say("done %.1f s; legacy cache untouched=%s" % (time.perf_counter() - T0, legacy_cache_untouched))
