"""PF-11 phase B: measure the cross-language retrieval approaches on the REAL corpus, offline (no model request).

    python -X utf8 qa/pf11_arabic/retrieval_probe.py <repo> <out dir> [--split dev|holdout|all]

For every paired question with expected pages, four retrieval queries go through the unchanged retrieval stack
(``retrieval.gather_candidates`` -> cross-encoder rerank -> top FINAL_K):

    A  the Arabic question as typed (direct cross-lingual retrieval with the English-only embedder and reranker)
    B  the lexicon query of ``arabic_query.transform_query`` (deterministic Arabic-to-English transformation)
    C  bilingual expansion: the candidate pools of A and B merged, reranked with the English query of B
    O  the paired English question (the oracle an ideal transformation would reach)

Recorded per query: the route the legacy Router gives it, the Coordinator's specialists, the pages of the reranked
top-5, the best rerank score (the refusal gate is MIN_RERANK_SCORE), whether an expected page is in the top-5 and in the
candidate pool. Also recorded: how the embedder's tokenizer reads Arabic (share of unknown tokens). Writes
``retrieval_probe_<split>.json`` and ``retrieval_probe_<split>.md``."""
import json
import os
import sys
import time

REPO, OUT = os.path.abspath(sys.argv[1]), sys.argv[2]
SPLIT = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "dev"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "qa", "pf11_arabic"))
for name in ("GROQ_API_KEY", "PLAN_F_SHADOW", "PLAN_F_ENABLED"):
    os.environ.pop(name, None)
os.makedirs(OUT, exist_ok=True)

import numpy as np                                                   # noqa: E402

from question_set import ALL                                         # noqa: E402
from handbook_bot import config                                      # noqa: E402
from handbook_bot.agents import router as router_agent               # noqa: E402
from handbook_bot.agents.coordinator import coordinate              # noqa: E402
from handbook_bot.arabic_query import transform_query               # noqa: E402
from handbook_bot.knowledge_base import build_knowledge_base        # noqa: E402
from handbook_bot.retrieval import deduplicate_by_text, gather_candidates   # noqa: E402
from handbook_bot.text_utils import normalize_text                   # noqa: E402

t0 = time.perf_counter()
kb = build_knowledge_base(verbose=False)
print("kb: %d chunks, %d pages, %.1f s" % (len(kb.chunks), len(kb.pages), time.perf_counter() - t0), flush=True)


def retrieve(query, pool=None):
    route = router_agent.classify(query)
    emb = kb.embedder.encode([normalize_text(query)], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    cands = gather_candidates(query, route, emb, kb.index, kb.chunks, kb.metadata)
    if pool is not None:
        seen = {c["idx"] for c in cands}
        cands = cands + [c for c in pool if c["idx"] not in seen]
    if not cands:
        return {"route": route, "top": [], "pool": [], "best": None, "cands": []}
    scores = kb.reranker.predict([[query, c["chunk"]] for c in cands])
    ranked = sorted((dict(c, rerank_score=float(s)) for c, s in zip(cands, scores)),
                    key=lambda x: (x["rerank_score"], x.get("routing_boost", 0.0), x.get("lexical_score", 0.0), x.get("dense_score", 0.0)), reverse=True)
    top = deduplicate_by_text(ranked)[:config.FINAL_K]
    key = lambda c: [c["meta"].get("source_id"), int(c["meta"]["page"])]
    return {"route": route, "top": [key(c) for c in top], "pool": [key(c) for c in cands], "best": top[0]["rerank_score"] if top else None, "cands": cands}


def specialists(query):
    try:
        return [s.value for s in coordinate(query).selected_specialists]
    except Exception as exc:
        return ["error:%s" % type(exc).__name__]


def hit(expected, pages):
    return any(e in pages for e in expected)


def unk_share(text):
    tok = getattr(kb.embedder, "tokenizer", None)
    if tok is None:
        return None
    pieces = tok.tokenize(text)
    return (sum(1 for p in pieces if p == tok.unk_token) / len(pieces)) if pieces else None


rows = []
for q in ALL:
    if SPLIT != "all" and q["split"] != SPLIT:
        continue
    if not q["pages"]:
        continue
    expected = [[q["source"], p] for p in q["pages"]]
    tr = transform_query(q["ar"], client=None, allow_model=False)
    a = retrieve(q["ar"])
    b = retrieve(tr.english_query)
    c = retrieve(tr.english_query, pool=a["cands"])
    o = retrieve(q["en"])
    row = {"id": q["id"], "pair": q["pair"], "domain": q["domain"], "expect": q["expect"], "ar": q["ar"], "en": q["en"], "expected": expected,
           "english_query": tr.english_query, "coverage": round(tr.coverage, 3), "unmapped": tr.unmapped,
           "arabic_unk_share": unk_share(q["ar"]), "english_unk_share": unk_share(q["en"]),
           "approaches": {}}
    for label, r in (("A", a), ("B", b), ("C", c), ("O", o)):
        row["approaches"][label] = {"route": r["route"], "top5": r["top"], "best": r["best"], "hit5": hit(expected, r["top"]), "hit_pool": hit(expected, r["pool"]),
                                    "gate_pass": (r["best"] is not None and r["best"] >= config.MIN_RERANK_SCORE)}
    row["specialists"] = {"B": specialists(tr.english_query), "O": specialists(q["en"]), "expected": q["specialists"]}
    row["route_agrees"] = row["approaches"]["B"]["route"] == row["approaches"]["O"]["route"]
    row["specialists_agree"] = row["specialists"]["B"] == row["specialists"]["O"]
    rows.append(row)
    print("%-7s A=%s B=%s C=%s O=%s route=%s spec=%s | %s" % (q["id"], *(("Y" if row["approaches"][k]["hit5"] else "-") for k in "ABCO"),
                                                             "=" if row["route_agrees"] else "x", "=" if row["specialists_agree"] else "x", tr.english_query), flush=True)

n = len(rows)
summary = {"split": SPLIT, "questions": n, "kb_chunks": len(kb.chunks), "embed_model": config.EMBED_MODEL, "rerank_model": config.RERANK_MODEL,
           "hit5": {k: sum(1 for r in rows if r["approaches"][k]["hit5"]) for k in "ABCO"},
           "hit_pool": {k: sum(1 for r in rows if r["approaches"][k]["hit_pool"]) for k in "ABCO"},
           "gate_pass": {k: sum(1 for r in rows if r["approaches"][k]["gate_pass"]) for k in "ABCO"},
           "route_agrees": sum(1 for r in rows if r["route_agrees"]), "specialists_agree": sum(1 for r in rows if r["specialists_agree"]),
           "mean_coverage": round(sum(r["coverage"] for r in rows) / n, 3) if n else None,
           "mean_arabic_unk_share": round(sum(r["arabic_unk_share"] or 0 for r in rows) / n, 3) if n else None,
           "mean_english_unk_share": round(sum(r["english_unk_share"] or 0 for r in rows) / n, 3) if n else None,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
with open(os.path.join(OUT, "retrieval_probe_%s.json" % SPLIT), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)
with open(os.path.join(OUT, "retrieval_probe_%s.md" % SPLIT), "w", encoding="utf-8") as f:
    f.write("# Retrieval probe (%s split, %d questions with expected pages)\n\n" % (SPLIT, n))
    f.write("Embedder `%s`, reranker `%s`, %d chunks. Hit = an expected (source, page) among the reranked top-%d.\n\n" % (config.EMBED_MODEL, config.RERANK_MODEL, len(kb.chunks), config.FINAL_K))
    f.write("| Approach | hit@5 | in candidate pool | passes the rerank gate (%.1f) |\n|---|---|---|---|\n" % config.MIN_RERANK_SCORE)
    names = {"A": "A direct Arabic", "B": "B lexicon transformation", "C": "C bilingual expansion", "O": "O English oracle"}
    for k in "ABCO":
        f.write("| %s | %d / %d | %d / %d | %d / %d |\n" % (names[k], summary["hit5"][k], n, summary["hit_pool"][k], n, summary["gate_pass"][k], n))
    f.write("\nRouter route B = O: %d / %d. Coordinator specialists B = O: %d / %d. Mean lexicon coverage %.3f. Mean unknown-token share in the embedder tokenizer: Arabic %.3f, English %.3f.\n\n"
            % (summary["route_agrees"], n, summary["specialists_agree"], n, summary["mean_coverage"], summary["mean_arabic_unk_share"], summary["mean_english_unk_share"]))
    f.write("| id | A | B | C | O | route B/O | specialists B/O | English query (B) |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        ap = r["approaches"]
        f.write("| %s | %s | %s | %s | %s | %s/%s | %s/%s | %s |\n" % (r["id"], *("Y" if ap[k]["hit5"] else "-" for k in "ABCO"), ap["B"]["route"], ap["O"]["route"],
                                                                 ",".join(r["specialists"]["B"]) or "-", ",".join(r["specialists"]["O"]) or "-", r["english_query"].replace("|", "/")))
print(json.dumps(summary, indent=1))
