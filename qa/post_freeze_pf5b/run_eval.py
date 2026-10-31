"""PF-5B evaluation (D-111 + D-112 + D-113 + D-114) on the REAL eight-source knowledge base through the real application seam
(``ui.pipeline``). Derived from ``qa/post_freeze_pf4/run_eval.py`` (same passes); one more set.

    python -X utf8 qa/post_freeze_pf5b/run_eval.py <repo> <output dir>

Four question sets, kept apart by the ``set`` field and reported separately:
    * P0 regression: the 48 PF-2, 22 PF-3B and 30 PF-3C questions (the PF-3D guide evaluation, unchanged);
    * P1: ``qa/post_freeze_pf4/p1_questions.jsonl`` (the three activated P1 guides, negatives, collisions, cross-document);
    * HELD-OUT (PF-4, now diagnostic only): ``qa/post_freeze_pf4/held_out_questions.jsonl`` with ``held_out_labels.json``;
    * FRESH (PF-5B): ``qa/post_freeze_pf5b/fresh_held_out_questions.jsonl`` (frozen before any PF-5B rule was written) with the
      labels of ``fresh_held_out_labels.json``; the wording never changes.
Passes, in order:
    1. ACTIVE      guide scopes ON: every question through ``pipeline.ask_with_diagnostics``
    2. DIRECT      the same questions on the SAME runtime object: outcome, the guard or rule that decided, and every
                   specialist clause report (outcome, procedure flag, ask shape, uncovered details, finding sources)
    3. SHADOW      guide scopes ON, compared field by field with the direct legacy call
    4. ACTIVE_OFF  guide scopes OFF (the I-9 specialists for every question): the control
    5. LEGACY      every question (what the unscoped legacy retrieval cites)
    6. REPEAT      the guide and PF-3B questions twice more in ACTIVE (determinism)
Writes ``results.json`` and ``log.txt``. Switches are set on the config module in this process only; no .env; no model call.
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
for name in ("PLAN_F_ENABLED", "PLAN_F_SHADOW", "PLAN_F_GUIDE_SCOPES"):
    os.environ.pop(name, None)
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
from ui import pipeline                                             # noqa: E402

PF2 = [json.loads(l) for l in open(os.path.join(REPO, "qa", "post_freeze_pf2", "guide_questions.jsonl"), encoding="utf-8") if l.strip()]
PF3 = [json.loads(l) for l in open(os.path.join(REPO, "qa", "post_freeze_pf3b", "pf3b_questions.jsonl"), encoding="utf-8") if l.strip()]
PF3C = [json.loads(l) for l in open(os.path.join(REPO, "qa", "post_freeze_pf3c", "pf3c_questions.jsonl"), encoding="utf-8") if l.strip()]
for entry in PF2 + PF3 + PF3C:
    entry["set"] = "p0"
P1 = [json.loads(l) for l in open(os.path.join(REPO, "qa", "post_freeze_pf4", "p1_questions.jsonl"), encoding="utf-8") if l.strip()]
for entry in P1:
    entry["set"] = "p1"
    entry.setdefault("domain", "teaching")
LABELS = json.load(open(os.path.join(REPO, "qa", "post_freeze_pf4", "held_out_labels.json"), encoding="utf-8"))["labels"]
HELD = []
for line in open(os.path.join(REPO, "qa", "post_freeze_pf4", "held_out_questions.jsonl"), encoding="utf-8"):
    if not line.strip():
        continue
    q = json.loads(line)
    label = LABELS[q["id"]]
    source = label["expected_sources"][0] if label["expected_sources"] else None
    HELD.append({"id": q["id"], "question": q["question"], "category": "held_out_" + q["category"], "set": "held_out", "source": source,
                 "pages": label["expected_pages"].get(source) if source else None, "domain": label["expected_specialist"] or "none",
                 "answerable": label["answerable"], "plan_f_should_serve": label["plan_f_should_serve"]})
FRESH_LABELS = json.load(open(os.path.join(REPO, "qa", "post_freeze_pf5b", "fresh_held_out_labels.json"), encoding="utf-8"))["labels"]
FRESH = []
for line in open(os.path.join(REPO, "qa", "post_freeze_pf5b", "fresh_held_out_questions.jsonl"), encoding="utf-8"):
    if not line.strip():
        continue
    q = json.loads(line)
    label = FRESH_LABELS[q["id"]]
    source = label["expected_sources"][0] if label["expected_sources"] else None
    FRESH.append({"id": q["id"], "question": q["question"], "category": "fresh_" + q["category"], "set": "fresh", "source": source,
                  "pages": label["expected_pages"].get(source) if source else None, "domain": label["expected_specialist"] or "none",
                  "answerable": label["answerable"], "plan_f_should_serve": label["plan_f_should_serve"]})
QUESTIONS = PF2 + PF3 + PF3C + P1 + HELD + FRESH
assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS), "question ids collide"

t = time.perf_counter()
kb = build_knowledge_base(verbose=False)
KB_BUILD_S = time.perf_counter() - t
say("kb %.1f s: %d chunks, %d sources; questions %d (PF-2 %d + PF-3B %d + PF-3C %d + P1 %d + held-out %d + fresh %d)"
    % (KB_BUILD_S, len(kb.chunks), len(kb.sources), len(QUESTIONS), len(PF2), len(PF3), len(PF3C), len(P1), len(HELD), len(FRESH)))
_WS = re.compile(r"\s+")
PAGE_TEXT = {(sid, p["page"]): p["text"] for sid, pages in kb.source_pages.items() for p in pages}


def norm(s):
    return _WS.sub(" ", s or "").strip()


def items_of(result):
    out = []
    for i in (result.items if result else []):
        idx, quote, meta = i.get("idx"), i.get("chunk") or "", i.get("meta") or {}
        page, sid = meta.get("page"), meta.get("source_id")
        chunk = kb.chunks[idx] if isinstance(idx, int) and 0 <= idx < len(kb.chunks) else None
        out.append({"idx": idx, "source_id": sid, "source_title": meta.get("source_title"), "page": page, "chunk_type": meta.get("chunk_type"),
                    "source_conflict": meta.get("source_conflict"), "quote": quote[:400], "plan_f": meta.get("plan_f"),
                    "quote_in_chunk": chunk is not None and quote in chunk,
                    "quote_in_page_normalized": bool(quote) and norm(quote) in norm(PAGE_TEXT.get((sid, page), ""))})
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
             "legacy_pages", "plan_f_sources", "legacy_sources", "guide_scope_enabled", "guide_scope_selected", "guide_sources_named",
             "plan_f_guide_served", "guide_answer_forms", "guide_answer_parts", "plan_f_ms", "registry_ok", "registry_reused", "registry_build_ms", "registry_errors", "plan_f_llm_calls",
             "plan_f_used_llm", "legacy_llm_calls", "findings_verified", "findings_accepted", "findings_reason", "findings_request_kind",
             "plan_f_path", "aggregation_attempted", "aggregation_accepted", "aggregation_reason", "routing_refined", "routing_reason",
             "routing_initial", "routing_final", "handoff_attempted", "handoff_outcome", "specialist_calls", "error")


def record(mode, entry, res, d, exc, wall, n):
    rec = {"mode": mode, "id": entry["id"], "set": entry["set"], "n": n, "category": entry["category"], "expected_source": entry.get("source"),
           "expected_pages": entry.get("pages"), "domain": entry["domain"], "question": entry["question"], "wall_ms": wall, "exception": exc}
    if res is not None:
        rec.update({"answer": res.answer, "pages": list(res.pages), "query_type": res.query_type, "used_llm": res.used_llm, "llm_calls": res.llm_calls,
                    "timings": res.timings, "diagnostics": {k: (d or {}).get(k) for k in DIAG_KEYS} if d is not None else None,
                    "items": items_of(res)})
        rec["served_sources"] = sorted({i["source_id"] for i in rec["items"] if i["source_id"]})
    return rec


def set_mode(mode, guide_scopes):
    config.PLAN_F_ENABLED, config.PLAN_F_SHADOW = (mode == "active"), (mode == "shadow")
    config.PLAN_F_GUIDE_SCOPES = bool(guide_scopes)
    pipeline.reset_plan_f_state()
    say("\n===== MODE %s (resolved %s, guide scopes %s) =====" % (mode, pipeline.plan_f_mode(), guide_scopes))


def show(n, entry, rec, d, exc, extra=""):
    say("[%3d] %-4s wall=%6.0f served=%-7s outcome=%-38s scope=%-5s pages=%-12s sources=%-50s exc=%s %s| %s" % (
        n, entry["id"], rec["wall_ms"], (d or {}).get("served"), (d or {}).get("outcome"), (d or {}).get("guide_scope_selected"),
        str(rec.get("pages"))[:12], ",".join(s.replace("uos_", "")[:14] for s in rec.get("served_sources", [])), exc, extra, entry["question"][:44]))


results = []
# ---- 1. ACTIVE, guide scopes on ------------------------------------------------------------------------------------------
set_mode("active", True)
cold = None
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["question"])
    rec = record("active", entry, res, d, exc, wall, n)
    if cold is None:
        cold = {"first_request_wall_ms": wall, "registry_build_ms": (d or {}).get("registry_build_ms"), "registry_ok": (d or {}).get("registry_ok"),
                "registry_errors": (d or {}).get("registry_errors")}
    results.append(rec)
    show(n, entry, rec, d, exc)

# ---- 2. DIRECT on the same runtime -----------------------------------------------------------------------------------------
cache_entry = pipeline._plan_f_cache().get_or_build(kb)
runtime = cache_entry.runtime
build_report = cache_entry.build.scope_extension
say("\n===== DIRECT (runtime reused=%s; guide registry %s) =====" % (cache_entry.reused, runtime.guide_registry is not None))
direct = []
for n, entry in enumerate(QUESTIONS, 1):
    rec = {"id": entry["id"], "question": entry["question"], "category": entry["category"], "exception": None}
    try:
        a = runtime.attempt(entry["question"])
        reports = []
        for run in (a.dispatch.runs if a.dispatch else []):
            f = run.findings
            if f is None:
                reports.append({"specialist": run.specialist_id.value, "state": run.state, "reason": run.reason[:120]})
                continue
            gp = (f.metadata or {}).get("guide_path")
            if gp is not None:
                reports.append({"specialist": run.specialist_id.value, "guide_path": {"answered": gp.get("answered"), "clauses": {
                    k: {"applies": c.get("applies"), "refused": c.get("refused"), "form": c.get("form"), "source_id": c.get("source_id"),
                        "procedure_key": c.get("procedure_key"), "chunk_ids": c.get("chunk_ids"), "ms": c.get("ms"),
                        "retrieval_ms": c.get("retrieval_ms"), "selection_ms": c.get("selection_ms"),
                        "tried": [t for cons in (c.get("considered") or []) for t in cons.get("tried", [])][:8]} for k, c in (gp.get("clauses") or {}).items()}}})
            for rep in (f.metadata or {}).get("clauses") or []:
                reports.append({"specialist": run.specialist_id.value, "status": f.status.value, "outcome": rep.get("outcome"),
                                "procedural": rep.get("procedural"), "process": rep.get("process"), "ask_shape": rep.get("ask_shape"),
                                "uncovered": rep.get("uncovered_details"),
                                "findings": [(x.source_id, x.page, x.evidence_quote[:120]) for x in f.findings]})
        guards = [(e.agent, e.decision) for e in a.trace if e.agent.endswith("guard") or e.agent in ("findings_verifier", "verifier", "aggregator")]
        rec.update({"outcome": a.outcome.value, "reason": a.reason[:240], "guide_scope": a.guide_scope, "guards": guards, "reports": reports,
                    "timings": dict(a.timings),
                    "answer": a.result.answer if a.result else None, "pages": a.result.pages if a.result else None,
                    "sources": sorted({(i["meta"] or {}).get("source_id") for i in a.result.items}) if a.result else None,
                    "routing_final": list(a.routing.final_specialists) if a.routing is not None else None,
                    "specialist_calls": a.specialist_calls, "llm_calls": a.dispatch.llm_calls if a.dispatch is not None else 0,
                    "used_llm": any(e.used_llm for e in a.trace)})
    except Exception as e:
        rec["exception"] = redact(e)
    direct.append(rec)
    say("[%3d] %-4s %-38s guards=%s | %s" % (n, entry["id"], rec.get("outcome"), rec.get("guards"), entry["question"][:50]))

# ---- 3. SHADOW, guide scopes on -------------------------------------------------------------------------------------------
set_mode("shadow", True)
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

# ---- 4. ACTIVE with guide scopes off (control) ----------------------------------------------------------------------------
set_mode("active", False)
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["question"])
    rec = record("active_off", entry, res, d, exc, wall, n)
    results.append(rec)
    show(n, entry, rec, d, exc)

# ---- 5. LEGACY -------------------------------------------------------------------------------------------------------------
set_mode("legacy", False)
for n, entry in enumerate(QUESTIONS, 1):
    res, d, exc, wall = ask(entry["question"])
    rec = record("legacy", entry, res, d, exc, wall, n)
    rec["minimal_diagnostics"] = d == {"mode": "legacy", "served": "legacy", "attempted": False}
    results.append(rec)
    show(n, entry, rec, d, exc)
legacy_cache_untouched = pipeline._PLAN_F_CACHE is None

# ---- 6. REPEAT -------------------------------------------------------------------------------------------------------------
set_mode("active", True)
repeats = []
subset = [e for e in QUESTIONS if e["category"] not in ("handbook_only", "known_hr", "annual_leave")]
for rep in (1, 2):
    for entry in subset:
        res, d, exc, wall = ask(entry["question"])
        rec = record("active", entry, res, d, exc, wall, 0)
        rec["rep"] = rep
        repeats.append(rec)
    say("determinism rep %d done (%d questions)" % (rep, len(subset)))

json.dump({"kb": {"primary_pages": len(kb.pages), "chunks": len(kb.chunks), "build_s": KB_BUILD_S, "corpus_key": kb.corpus_key,
                  "groq_client_configured": kb.groq_client is not None},
           "registry_report": build_report, "cold": cold, "questions": QUESTIONS, "results": results, "direct": direct, "repeats": repeats,
           "legacy_cache_untouched": legacy_cache_untouched, "total_script_s": time.perf_counter() - T0},
          open(os.path.join(OUT, "results.json"), "w", encoding="utf-8"), indent=1, default=str)
say("done %.1f s; legacy cache untouched=%s" % (time.perf_counter() - T0, legacy_cache_untouched))
