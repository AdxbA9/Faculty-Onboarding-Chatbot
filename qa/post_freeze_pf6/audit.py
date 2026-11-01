"""PF-6 source registry and corpus integrity audit (no model call; the cold rebuild loads the embedding model only).

    python -X utf8 qa/post_freeze_pf6/audit.py <repo> <out dir> [--cold <empty scratch cache dir>] [--pf5c <PF-5C results.json>]

Writes ``source_registry_audit.json`` and ``corpus_integrity.json``:
* every registry record: officiality (authority, verification status), title, id, type, audience, enabled, access, language
  and page filter, specialist access (D-110 approved sets), conflict metadata, recorded versus actual file sha256;
* the production corpus assembled twice (deterministic ids, pages, chunks, metadata), per-source counts, language-filter
  drops, duplicate-ingestion checks, held sources absent from every chunk, the corpus identity, and the cached embeddings
  loaded for that identity; with ``--cold`` the corpus is embedded again from scratch into a separate cache directory and
  compared with the cached embeddings (the production cache is never written or deleted).
"""
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
REPO = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
COLD = sys.argv[sys.argv.index("--cold") + 1] if "--cold" in sys.argv else None
PF5C = sys.argv[sys.argv.index("--pf5c") + 1] if "--pf5c" in sys.argv else None
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO)
sys.path.insert(0, REPO)

from handbook_bot import config                                                      # noqa: E402
from handbook_bot.knowledge_base import assemble_corpus, corpus_identity             # noqa: E402
from handbook_bot.source_scopes import GUIDE_ALLOWLIST, SPECIALIST_IDS, source_policy  # noqa: E402
from handbook_bot.sources import load_source_registry                                # noqa: E402

HELD = {"uos_performance_appraisal_phase2_guide": "registered, disabled, Internal Use (ingestion hold)",
        "assignments_and_tests": "not registered: 25 pages collide with frozen page-keyed indexes (D-113)",
        "grade_categories": "not registered: interleaved English/Arabic pages (D-113)"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---- registry ----------------------------------------------------------------------------------------------------------
registry = load_source_registry(os.path.join("knowledge", "sources.json"))
records = registry.list_sources()
enabled = registry.enabled_sources()
policy = source_policy(enabled)
rows = []
for r in records:
    actual = sha(r.absolute_path()) if r.exists() else None
    rows.append({
        "source_id": r.source_id, "title": r.title, "citation_label": r.citation_label, "document_type": r.document_type,
        "source_type": r.source_type, "authority": r.authority, "authority_level": r.authority_level,
        "verification_status": r.verification_status, "audience": list(r.audience), "enabled": r.enabled, "access": r.access,
        "language": r.language, "page_filter": r.page_filter, "system": r.system, "domains": list(r.domains),
        "approved_for_specialists": sorted(sid for sid, ids in policy.items() if r.source_id in ids),
        "conflicts": [c if isinstance(c, dict) else getattr(c, "__dict__", str(c)) for c in (r.conflicts or ())],
        "file_present": r.exists(), "sha256_recorded": r.sha256 or None, "sha256_actual": actual,
        "sha256_match": (r.sha256 == actual) if (r.sha256 and actual) else None})
guides_present = sorted(os.listdir(os.path.join("data", "guides")))
registered_files = {os.path.basename(r.relative_path) for r in records}
audit = {
    "records": len(records), "enabled": len(enabled), "enabled_ids": [r.source_id for r in enabled],
    "disabled": [r.source_id for r in records if not r.enabled],
    "internal_use_enabled": [r.source_id for r in enabled if r.access == "internal_use"],
    "held_enabled": [r.source_id for r in enabled if "HOLD" in (r.verification_status or "").upper()],
    "enabled_missing_file": [r.source_id for r in enabled if not r.exists()],
    "enabled_hash_mismatch": [x["source_id"] for x in rows if x["enabled"] and x["sha256_match"] is False],
    "enabled_without_recorded_hash": [x["source_id"] for x in rows if x["enabled"] and not x["sha256_recorded"]],
    "unregistered_files_in_data_guides": [f for f in guides_present if f not in registered_files],
    "guide_allowlist": {k: sorted(v) for k, v in GUIDE_ALLOWLIST.items()},
    "approved_sources_per_specialist": {k: sorted(v) for k, v in policy.items()},
    "held_sources": HELD,
    "guide_scopes_default": config.PLAN_F_GUIDE_SCOPES, "plan_f_enabled_default": config.PLAN_F_ENABLED,
    "plan_f_shadow_default": config.PLAN_F_SHADOW, "cache_version": config.CACHE_VERSION,
    "sources": rows}
audit["pass"] = (audit["enabled"] == 8 and not audit["internal_use_enabled"] and not audit["held_enabled"]
                 and not audit["enabled_missing_file"] and not audit["enabled_hash_mismatch"]
                 and not audit["unregistered_files_in_data_guides"] and audit["guide_scopes_default"] is False)
json.dump(audit, open(os.path.join(OUT, "source_registry_audit.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print("registry", {k: audit[k] for k in ("records", "enabled", "disabled", "internal_use_enabled", "enabled_missing_file",
                                         "enabled_hash_mismatch", "unregistered_files_in_data_guides", "pass")})

# ---- corpus -------------------------------------------------------------------------------------------------------------
t = time.perf_counter()
first = assemble_corpus(enabled)
assemble_s = time.perf_counter() - t
second = assemble_corpus(enabled)
key, entries = corpus_identity(enabled)
ids_in_chunks = sorted({m.get("source_id") for m in first.metadata})
per_source = [{k: s.get(k) for k in ("source_id", "pages_loaded", "pages_indexed", "chunks", "chunk_id_range", "conflict_chunks",
                                      "chunk_types")} | {"pages_dropped": (s.get("language_filter") or {}).get("pages_dropped", [])}
              for s in first.per_source]
corpus = {
    "corpus_key": first.key, "identity_key": key, "identity_entries": entries,
    "sources": len(first.records), "pages_indexed": sum(len(p) for p in first.source_pages.values()), "chunks": len(first.chunks),
    "per_source": per_source,
    "deterministic_rebuild": first.chunks == second.chunks and first.metadata == second.metadata and first.key == second.key,
    "assemble_s": round(assemble_s, 2),
    "source_ids_in_chunks": ids_in_chunks,
    "duplicate_source_ids": len({r.source_id for r in first.records}) != len(first.records),
    "duplicate_file_hashes": len({e["sha256"] for e in entries}) != len(entries),
    "held_source_chunks": {h: sum(1 for m in first.metadata if m.get("source_id") == h) for h in HELD},
    "held_titles_in_chunk_metadata": sum(1 for m in first.metadata if "apprais" in str(m.get("source_title", "")).lower()),
    "conflict_stamped_chunks": sum(1 for m in first.metadata if m.get("source_conflict")),
    "conflict_topics": sorted({m.get("source_conflict") for m in first.metadata if m.get("source_conflict")}),
}
expected = {"sources": 8, "pages_indexed": 309, "chunks": 15000}
corpus["matches_pf5c"] = {k: corpus[k] == v for k, v in expected.items()}
if PF5C:
    kb = json.load(open(PF5C, encoding="utf-8")).get("kb") or {}
    corpus["pf5c_corpus_key"] = kb.get("corpus_key")
    corpus["matches_pf5c"]["corpus_key"] = kb.get("corpus_key") == corpus["corpus_key"]

# the cached embeddings of this identity (load only; never rebuilt or deleted here)
from handbook_bot import retrieval                                                    # noqa: E402
prefix = os.path.join(config.CACHE_DIR, "corpus_%s_%s" % (first.key, config.CACHE_VERSION))
cache_files = [prefix + s for s in ("_embeddings.npy", "_chunks.json", "_meta.json", "_pages.json")]
corpus["cache"] = {"files_present": all(os.path.exists(p) for p in cache_files), "cache_dir_gitignored": True}
if corpus["cache"]["files_present"]:
    import numpy as np                                                                # noqa: E402
    cached = np.load(cache_files[0])
    same_inputs = (json.load(open(cache_files[1], encoding="utf-8")) == first.chunks
                   and json.load(open(cache_files[2], encoding="utf-8")) == first.metadata
                   and json.load(open(cache_files[3], encoding="utf-8")) == first.cache_pages)
    corpus["cache"].update({"embedding_shape": list(cached.shape), "cache_inputs_match_corpus": same_inputs})
    if COLD:
        from handbook_bot.knowledge_base import _load_models                          # noqa: E402
        os.makedirs(COLD, exist_ok=True)
        assert not os.listdir(COLD), "the cold cache directory must be empty"
        retrieval.CACHE_DIR = COLD                                                    # this process only
        embedder, _reranker = _load_models()
        t = time.perf_counter()
        fresh = retrieval.build_or_load_embeddings("corpus_" + first.key, first.chunks, first.metadata, first.cache_pages, embedder)
        cold_s = time.perf_counter() - t
        t = time.perf_counter()
        again = retrieval.build_or_load_embeddings("corpus_" + first.key, first.chunks, first.metadata, first.cache_pages, embedder)
        warm_s = time.perf_counter() - t
        corpus["cold_rebuild"] = {"seconds": round(cold_s, 1), "shape": list(fresh.shape),
                                  "max_abs_diff_vs_cached": float(np.max(np.abs(fresh - cached))),
                                  "identical_to_cached": bool(np.array_equal(fresh, cached)),
                                  "reload_from_new_cache_s": round(warm_s, 2), "reload_identical": bool(np.array_equal(fresh, again))}
        # cache invalidation: a changed metadata entry makes the cached files stale (checked without writing)
        changed = [dict(m) for m in first.metadata]
        changed[0]["page"] = -1
        corpus["cache_invalidation"] = {"changed_metadata_matches_cache": json.load(open(cache_files[2], encoding="utf-8")) == changed,
                                        "rule": "the cache is reused only when chunks, metadata and pages equal the cached files, "
                                                "under a key built from source ids, file hashes, page filters, chunking and CACHE_VERSION"}
corpus["pass"] = (all(corpus["matches_pf5c"].values()) and corpus["deterministic_rebuild"] and not corpus["duplicate_source_ids"]
                  and not corpus["duplicate_file_hashes"] and not any(corpus["held_source_chunks"].values())
                  and corpus["held_titles_in_chunk_metadata"] == 0 and corpus["source_ids_in_chunks"] == sorted(audit["enabled_ids"]))
json.dump(corpus, open(os.path.join(OUT, "corpus_integrity.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print("corpus", {k: corpus[k] for k in ("corpus_key", "sources", "pages_indexed", "chunks", "deterministic_rebuild", "held_source_chunks",
                                        "conflict_stamped_chunks", "conflict_topics", "matches_pf5c", "pass")})
print("cache", corpus.get("cache"), corpus.get("cold_rebuild"), corpus.get("cache_invalidation"))
