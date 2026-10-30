"""Cached-startup performance of the multi-source knowledge base: model load, cache load, index build, totals, cache size.
Run alone (nothing else on the CPU) after the corpus cache exists. Writes a JSON record; no path is recorded.

    python -X utf8 qa/post_freeze_pf2/perf_snapshot.py <repo> <out json> [runs]
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
os.chdir(REPO)
sys.path.insert(0, REPO)

import handbook_bot.knowledge_base as K   # noqa: E402

runs = []
for n in range(RUNS):
    marks = {}
    real_load, real_cache, real_assemble = K._load_models, K.build_or_load_embeddings, K.assemble_corpus

    def timed_load():
        t = time.perf_counter()
        r = real_load()
        marks["models_ms"] = (time.perf_counter() - t) * 1000.0
        return r

    def timed_assemble(records, log=lambda m: None):
        t = time.perf_counter()
        r = real_assemble(records, log)
        marks["assemble_ms"] = (time.perf_counter() - t) * 1000.0
        return r

    def timed_cache(*a, **kw):
        t = time.perf_counter()
        r = real_cache(*a, **kw)
        marks["embeddings_cache_ms"] = (time.perf_counter() - t) * 1000.0
        return r

    K._load_models, K.build_or_load_embeddings, K.assemble_corpus = timed_load, timed_cache, timed_assemble
    t0 = time.perf_counter()
    kb = K.build_knowledge_base(verbose=False)
    marks["total_ms"] = (time.perf_counter() - t0) * 1000.0
    K._load_models, K.build_or_load_embeddings, K.assemble_corpus = real_load, real_cache, real_assemble
    runs.append({k: round(v, 1) for k, v in marks.items()})
    print("run %d: %s" % (n + 1, runs[-1]))

cache_files = sorted(f for f in os.listdir("cache") if kb.corpus_key in f)
record = {
    "runs": runs, "chunks": len(kb.chunks), "embedding_shape": list(kb.embeddings.shape), "embedding_bytes": int(kb.embeddings.nbytes),
    "index_ntotal": int(kb.index.ntotal), "corpus_key": kb.corpus_key, "cache_files": cache_files,
    "cache_bytes": {f: os.path.getsize(os.path.join("cache", f)) for f in cache_files},
    "cache_bytes_total": sum(os.path.getsize(os.path.join("cache", f)) for f in cache_files),
    "sources": [(s["source_id"], s["pages_indexed"], s["chunks"]) for s in kb.stats["sources"]],
}
json.dump(record, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in record.items() if k not in ("cache_bytes",)}))
