"""Write the runtime freeze manifest (runtime integration, phase I-9): the identifiers a later comparison needs, with no local
absolute path. SHA-256 of the integration modules, the seam and the interface layer, the six frozen specialist files and
their shared helpers, the evaluation set and the key QA artifacts, the knowledge-base cache files, plus the git HEAD, the
Python and key package versions and the test counts given on the command line.

    python -X utf8 qa/plan_f_i9/make_manifest.py <repo> <out json> [--tests "<gate>,<full>"]

Re-run after post-freeze work and diff the JSON against the committed one: a changed hash names what moved.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

REPO, OUT = sys.argv[1], sys.argv[2]
TESTS = sys.argv[sys.argv.index("--tests") + 1] if "--tests" in sys.argv else ""
os.chdir(REPO)

FREEZE_ID = "runtime-freeze-i9"
INTEGRATION = ["handbook_bot/plan_f_runtime.py", "handbook_bot/plan_f_routing.py", "handbook_bot/plan_f_handoff.py", "handbook_bot/plan_f_aggregation.py",
               "handbook_bot/plan_f_verification.py", "handbook_bot/agents/coordinator.py", "handbook_bot/agents/specialists/dispatch.py",
               "handbook_bot/agents/specialists/registry.py", "handbook_bot/agents/specialists/contracts.py", "handbook_bot/agents/specialists/base.py",
               "handbook_bot/agents/verifier.py", "handbook_bot/agents/router.py", "handbook_bot/orchestrator.py", "handbook_bot/qa.py",
               "handbook_bot/config.py", "ui/pipeline.py", "ui/presentation.py", "ui/chat.py", "ui/styles.py", "app.py"]
FROZEN = ["handbook_bot/agents/specialists/teaching.py", "handbook_bot/agents/specialists/research.py", "handbook_bot/agents/specialists/faculty_services.py",
          "handbook_bot/agents/specialists/claim_model.py", "handbook_bot/agents/specialists/navigation_cues.py", "handbook_bot/agents/specialists/institutional.py"]
SPECIALIST_HELPERS = ["handbook_bot/agents/specialists/details.py", "handbook_bot/agents/specialists/domain_cues.py", "handbook_bot/agents/specialists/evidence.py",
                      "handbook_bot/agents/specialists/grounding.py", "handbook_bot/agents/specialists/retrieval_intent.py", "handbook_bot/agents/specialists/scoping.py"]
QA = ["qa/plan_f_i8/question_set.py", "qa/plan_f_i8/master_eval.py", "qa/plan_f_i8/analyze.py", "qa/plan_f_i8/review.json", "qa/plan_f_i8/metrics.json",
      "qa/plan_f_i8/results_2026-10-30.json", "qa/plan_f_i8/README.md", "qa/plan_f_i9/concurrency_smoke.py", "qa/plan_f_i9/make_manifest.py",
      "docs/RUNTIME_FREEZE_I9.md", "docs/STATUS.md", "docs/DECISIONS.md"]
CACHE_DIR = "cache"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def hashes(paths):
    return {p: (sha(p) if os.path.exists(p) else None) for p in paths}


cache = {}
if os.path.isdir(CACHE_DIR):
    for name in sorted(os.listdir(CACHE_DIR)):
        if "_v12_" in name:
            cache[name] = {"sha256": sha(os.path.join(CACHE_DIR, name)), "bytes": os.path.getsize(os.path.join(CACHE_DIR, name))}


def version(dist):
    try:
        from importlib.metadata import version as v
        return v(dist)
    except Exception:
        return None


manifest = {
    "freeze_id": FREEZE_ID,
    "phase": "I-9",
    "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git": {"base_head_at_manifest_time": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "base_subject": git("log", "-1", "--format=%s"),
            "freeze_commit": "the commit that adds this file (subject: Freeze Plan F runtime integration); a commit cannot record its own hash"},
    "environment": {"python": platform.python_version(), "platform": platform.system() + " " + platform.release(),
                    "packages": {d: version(d) for d in ("nicegui", "torch", "transformers", "sentence-transformers", "numpy", "pymupdf", "groq", "pytest", "faiss-cpu")}},
    "plan_f_versions": {"routing": "plan-f-routing-1", "handoff": "plan-f-handoff-1", "aggregation": "plan-f-aggregation-1", "verification_phase": "I-4 (+ I-8 rules)",
                        "handbook_bot.__version__": None},
    "integration_modules": hashes(INTEGRATION),
    "frozen_specialists": hashes(FROZEN),
    "specialist_helpers": hashes(SPECIALIST_HELPERS),
    "qa_artifacts": hashes(QA),
    "knowledge_base_cache_v12": cache,
    "evaluation_set": {"file": "qa/plan_f_i8/question_set.py", "questions": None, "sha256": sha("qa/plan_f_i8/question_set.py")},
    "tests": {"targeted_gate": TESTS.split(",")[0] if TESTS else None, "full_suite": TESTS.split(",")[1] if TESTS and "," in TESTS else None},
    "frozen_metrics_i8_final": {"questions": 192, "served": 55, "fallback": 137, "coverage_pct": 28.6, "correct": 55, "limited": 0, "wrong": 0,
                                "precision_pct": 100.0, "false_acceptances": 0, "false_fallbacks": 11, "shadow_mismatches": 0, "plan_f_llm_calls": 0},
}
try:
    sys.path.insert(0, REPO)
    import handbook_bot
    manifest["plan_f_versions"]["handbook_bot.__version__"] = getattr(handbook_bot, "__version__", None)
    sys.path.insert(0, os.path.join(REPO, "qa", "plan_f_i8"))
    from question_set import QUESTIONS
    manifest["evaluation_set"]["questions"] = len(QUESTIONS)
except Exception as exc:                                           # the manifest must still be written
    manifest["note"] = "metadata import skipped: %s" % type(exc).__name__
json.dump(manifest, open(OUT, "w", encoding="utf-8"), indent=1)
print("manifest written:", OUT, "| head", manifest["git"]["head"][:12], "| integration files", sum(1 for v in manifest["integration_modules"].values() if v),
      "| frozen", sum(1 for v in manifest["frozen_specialists"].values() if v), "| cache files", len(cache))
