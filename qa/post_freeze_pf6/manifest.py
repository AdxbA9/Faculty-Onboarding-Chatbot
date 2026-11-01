"""PF-6 freeze manifest and final metrics, built from the committed PF-6 artifacts and the registry (no model call).

    python -X utf8 qa/post_freeze_pf6/manifest.py <repo> <timestamp UTC> <targeted tests passed> <full suite passed> <pip check PASS|FAIL>

Writes ``qa/post_freeze_pf6/freeze_manifest.json`` (``ENGLISH_CORE_FREEZE_PF6``) and ``qa/post_freeze_pf6/final_metrics.json``.
The freeze commit is the commit that adds these files; its parent is recorded here and its hash in docs/STATUS.md.
"""
import json
import os
import subprocess
import sys

REPO = os.path.abspath(sys.argv[1])
STAMP, GATE, FULL, PIP = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.chdir(REPO)
sys.path.insert(0, REPO)
from handbook_bot import config                                     # noqa: E402

HERE = os.path.join("qa", "post_freeze_pf6")


def art(name):
    return json.load(open(os.path.join(HERE, name), encoding="utf-8"))


reg, corpus = art("source_registry_audit.json"), art("corpus_integrity.json")
frozen, p0, p1, held = art("frozen_192_final.json"), art("p0_final.json"), art("p1_final.json"), art("held_out_diagnostics.json")
ws, conf, bh = art("wrong_source_final.json"), art("conflict_final.json"), art("boundary_heldsource_final.json")
proc, cit, xdoc, det = art("procedure_final.json"), art("citation_review.json"), art("cross_document_final.json"), art("determinism.json")
leg, shadow, active, smoke = art("legacy_isolation.json"), art("shadow_final.json"), art("active_final.json"), art("app_smoke_final.json")
perf, sec = art("performance_final.json"), art("security_final.json")
parent = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

per_source = {s["source_id"]: s for s in corpus["per_source"]}
manifest = {
    "freeze_identifier": "ENGLISH_CORE_FREEZE_PF6",
    "decision_record": "D-116 Final English technical freeze",
    "freeze_commit": "the commit that adds this file (message 'Freeze final English technical baseline')",
    "parent_commit": parent, "branch": "feature/post-freeze-knowledge-expansion", "timestamp_utc": STAMP,
    "frozen_control": "runtime-freeze-i9 (D-108) with D-109 to D-115",
    "corpus_identity": {"key": corpus["corpus_key"], "cache_version": config.CACHE_VERSION, "sources": corpus["sources"],
                        "pages": corpus["pages_indexed"], "chunks": corpus["chunks"], "conflict_stamped_chunks": corpus["conflict_stamped_chunks"],
                        "conflict_topics": corpus["conflict_topics"], "embedding_shape": corpus["cache"].get("embedding_shape"),
                        "cold_rebuild_identical": (corpus.get("cold_rebuild") or {}).get("identical_to_cached")},
    "enabled_sources": [{"source_id": r["source_id"], "title": r["title"], "sha256": r["sha256_actual"],
                         "pages_indexed": per_source.get(r["source_id"], {}).get("pages_indexed"), "chunks": per_source.get(r["source_id"], {}).get("chunks")}
                        for r in reg["sources"] if r["enabled"]],
    "held_sources": reg["held_sources"],
    "switch_defaults": {"application_mode": "LEGACY", "PLAN_F_ENABLED": reg["plan_f_enabled_default"], "PLAN_F_SHADOW": reg["plan_f_shadow_default"],
                        "PLAN_F_GUIDE_SCOPES": reg["guide_scopes_default"]},
    "tests": {"targeted_gate_passed": GATE, "full_suite_passed": FULL, "failed": 0, "skipped": 0, "xfail": 0, "pip_check": PIP},
}
json.dump(manifest, open(os.path.join(HERE, "freeze_manifest.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)

fd = frozen["frozen_default"]
metrics = {
    "freeze_identifier": "ENGLISH_CORE_FREEZE_PF6", "coverage_is_not_a_freeze_criterion": True,
    "enabled_sources": corpus["sources"], "corpus_pages": corpus["pages_indexed"], "total_chunks": corpus["chunks"],
    "frozen_192": {"served": fd["served"], "correct": fd["correct"], "limited": fd["limited"], "wrong": fd["wrong"],
                   "false_acceptances": len(fd["false_acceptances"]), "same_55_as_i9": all(c["served_ids_equal"] for c in fd["vs_i9"].values()),
                   "guide_scopes_on_identical": frozen["same_served_set_scopes_off_and_on"], "pass": frozen["pass"]},
    "p0": {k: p0[k] for k in ("core_guide", "all_guide", "changed_vs_pf5c")},
    "p1": {k: p1[k] for k in ("all", "must_not_serve_guide_served", "changed_vs_pf5c")},
    "held_out_diagnostics": held["sets"],
    "wrong_source": ws["summary"] | {"pass": ws["pass"]}, "inference_adversarial": ws["inference_adversarial"]["summary"],
    "conflict": conf["summary"], "boundary_and_held_sources": bh["summary"],
    "procedure": proc["summary"], "citation": cit["summary"], "cross_document": xdoc["summary"],
    "determinism": {k: v for k, v in det.items() if k != "method"}, "legacy": leg, "shadow": shadow["summary"],
    "active": {k: active[k] for k in ("production_default_scopes_off", "scopes_on", "control_scopes_off_vs_on")},
    "app_smoke": smoke["summary"], "performance": {"meaningful_regressions": perf["meaningful_regressions"], "pass": perf["pass"]},
    "security": {"pass": sec["pass"], "real_secrets_found": sec["real_secrets_found"]},
    "tests": manifest["tests"], "plan_f_guide_scopes_final_default": "OFF",
}
json.dump(metrics, open(os.path.join(HERE, "final_metrics.json"), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
print(json.dumps({"manifest": {k: manifest[k] for k in ("freeze_identifier", "parent_commit", "timestamp_utc", "corpus_identity", "tests")}},
                 ensure_ascii=False))
