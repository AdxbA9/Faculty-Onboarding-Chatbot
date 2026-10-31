"""PF-3C evidence-selection measurement on the REAL knowledge base (no model call besides the local embedder and reranker).

    python -X utf8 qa/post_freeze_pf3c/evidence_selection.py <repo> <out.json> [--pf3c]

For each of the 37 guide questions (``qa/post_freeze_pf2/guide_questions.jsonl`` G01-G37) and its hand-labelled target
passage (``guide_targets.json``) it records:
  * UNSCOPED: the whole-corpus hybrid retrieval plus cross-encoder order (what the legacy path sees): expected source
    top-1 and top-5.
  * GUIDE-SCOPED (B0): the same retrieval restricted to the guide sources the question names (the PF-3B candidate pool
    for guide evidence): top-1 category (ON_TARGET / PARTIAL / OFF_TARGET), rank of the first on-target chunk, whether
    the top hit is a generic guide introduction, whether a procedure chunk is in the top five.
  * PF-3C (with ``--pf3c``): the evidence the D-111 guide path selects for the question (whether or not it is served),
    with the same categories.
A generic introduction is a chunk on a guide's first page carrying the guide's opening description (listed below).
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
WITH_PF3C = "--pf3c" in sys.argv[3:]
os.chdir(REPO)
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np                                                 # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base       # noqa: E402
from handbook_bot.retrieval import RetrievalScope, gather_candidates, deduplicate_by_text   # noqa: E402
from handbook_bot.agents.coordinator import coordinate             # noqa: E402
from handbook_bot.source_scopes import guide_sources_named         # noqa: E402
from handbook_bot.text_utils import normalize_text                 # noqa: E402

INTRO_PHRASES = {
    "uos_blackboard_faculty_short_guide": "blackboard ultra faculty short guide",
    "uos_blackboard_gradebook_guide": "in blackboard ultra grading, you can see what you need to grade",
    "uos_grade_transfer_dashboard_guide": "this guide provides simple steps for transferring grades",
    "uos_ithenticate_guidelines": "ithenticate is a text-matching tool",
}
QUESTIONS = [json.loads(l) for l in open(os.path.join(REPO, "qa", "post_freeze_pf2", "guide_questions.jsonl"), encoding="utf-8") if l.strip()]
QUESTIONS = [q for q in QUESTIONS if q["id"] <= "G37"]
TARGETS = json.load(open(os.path.join(HERE, "guide_targets.json"), encoding="utf-8"))["targets"]
_WS = re.compile(r"\s+")


def fold(s):
    return _WS.sub(" ", (s or "").replace("’", "'").lower()).strip()


def category(qid, source_id, page, text):
    t = TARGETS[qid]
    if source_id != t["source"] or page not in t["pages"]:
        return "OFF_TARGET"
    return "ON_TARGET" if any(fold(p) in fold(text) for p in t["phrases"]) else "PARTIAL"


def is_intro(meta, text):
    phrase = INTRO_PHRASES.get(meta.get("source_id"))
    return bool(phrase) and meta.get("page") == 1 and phrase in fold(text)


t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
load_s = time.perf_counter() - t0
embedder, reranker = kb.embedder, kb.reranker


def ranked(question, route, scope):
    emb = embedder.encode([normalize_text(question)], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    cands = gather_candidates(question, route, emb, kb.index, kb.chunks, kb.metadata, scope=scope)
    if not cands:
        return []
    scores = reranker.predict([[question, c["chunk"]] for c in cands])
    items = []
    for c, s in zip(cands, scores):
        item = dict(c)
        item["rerank_score"] = float(s)
        items.append(item)
    items.sort(key=lambda x: (x["rerank_score"], x.get("routing_boost", 0.0), x.get("lexical_score", 0.0), x.get("dense_score", 0.0)), reverse=True)
    return deduplicate_by_text(items)


selector = None
if WITH_PF3C:
    from handbook_bot.agents.specialists import guide_evidence as _ge       # noqa: E402
    selector = _ge.GuideEvidence.from_knowledge_base(kb)

rows = []
for q in QUESTIONS:
    qid, text = q["id"], q["question"]
    decision = coordinate(text)
    route = ((decision.metadata or {}).get("router") or {}).get("route") or "policy"
    t = time.perf_counter()
    unscoped = ranked(text, route, None)
    named = guide_sources_named(text, kb.sources)
    scoped = ranked(text, route, RetrievalScope(source_ids=frozenset(named))) if named else []
    ms = (time.perf_counter() - t) * 1000.0
    expected = TARGETS[qid]["source"]
    row = {"id": qid, "route": route, "named": named, "ms": round(ms, 1),
           "unscoped_top1_source": unscoped[0]["meta"].get("source_id") if unscoped else None,
           "unscoped_expected_top5": any(i["meta"].get("source_id") == expected for i in unscoped[:5])}
    if scoped:
        top = scoped[0]
        cats = [category(qid, i["meta"].get("source_id"), i["meta"].get("page"), i["chunk"]) for i in scoped]
        row.update({"b0_top1": cats[0], "b0_top1_chunk_type": top["meta"].get("chunk_type"), "b0_top1_page": top["meta"].get("page"),
                    "b0_top1_source": top["meta"].get("source_id"), "b0_top1_intro": is_intro(top["meta"], top["chunk"]),
                    "b0_first_on_target_rank": (cats.index("ON_TARGET") + 1) if "ON_TARGET" in cats else None,
                    "b0_procedure_in_top5": any(i["meta"].get("chunk_type") in ("procedure_step", "procedure_window") for i in scoped[:5]),
                    "b0_expected_top1": top["meta"].get("source_id") == expected,
                    "b0_expected_top5": any(i["meta"].get("source_id") == expected for i in scoped[:5]),
                    "b0_pool": len(scoped)})
    if selector is not None:
        t = time.perf_counter()
        sel = selector.inspect(text)
        row["pf3c_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
        row["pf3c"] = sel
        ev = sel.get("evidence") or []
        if ev:
            first = ev[0]
            row["pf3c_top1"] = category(qid, first["source_id"], first["page"], first["text"])
            row["pf3c_top1_intro"] = is_intro({"source_id": first["source_id"], "page": first["page"]}, first["text"])
            row["pf3c_top1_kind"] = first.get("kind")
    rows.append(row)
    print(qid, row.get("b0_top1"), row.get("b0_first_on_target_rank"), row.get("pf3c_top1", ""), flush=True)


def count(key, value):
    return sum(1 for r in rows if r.get(key) == value)


summary = {"questions": len(rows), "kb_load_s": round(load_s, 1),
           "unscoped_expected_top1": sum(1 for r in rows if r["unscoped_top1_source"] == TARGETS[r["id"]]["source"]),
           "unscoped_expected_top5": count("unscoped_expected_top5", True),
           "b0": {"on_target_top1": count("b0_top1", "ON_TARGET"), "partial_top1": count("b0_top1", "PARTIAL"),
                  "off_target_top1": count("b0_top1", "OFF_TARGET"), "intro_top1": count("b0_top1_intro", True),
                  "procedure_in_top5": count("b0_procedure_in_top5", True), "expected_source_top1": count("b0_expected_top1", True),
                  "expected_source_top5": count("b0_expected_top5", True),
                  "on_target_in_top5": sum(1 for r in rows if (r.get("b0_first_on_target_rank") or 99) <= 5),
                  "median_ms": float(np.median([r["ms"] for r in rows]))}}
if selector is not None:
    with_ev = [r for r in rows if r.get("pf3c_top1")]
    summary["pf3c"] = {"with_evidence": len(with_ev), "on_target_top1": count("pf3c_top1", "ON_TARGET"), "partial_top1": count("pf3c_top1", "PARTIAL"),
                       "off_target_top1": count("pf3c_top1", "OFF_TARGET"), "intro_top1": count("pf3c_top1_intro", True),
                       "median_ms": float(np.median([r["pf3c_ms"] for r in rows]))}
json.dump({"method": __doc__.strip(), "summary": summary, "rows": rows}, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
