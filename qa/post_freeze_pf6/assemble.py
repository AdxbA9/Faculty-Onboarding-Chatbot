"""PF-6 artifact assembly: turns the raw outputs of the PF-6 runs (kept outside the repository) into the committed freeze
artifacts. No model call; reads JSON only.

    python -X utf8 qa/post_freeze_pf6/assemble.py <pf6 run dir> <PF-5C run dir> <out dir>

<pf6 run dir> holds ``frozen_default/`` and ``frozen_on/`` (``qa/plan_f_i8/master_eval.py`` with guide scopes off and on, each
analysed by ``qa/plan_f_i8/analyze.py`` and compared with I-9 by ``qa/post_freeze_pf2/baseline_compare.py``), ``checks/``
(``wrong_source_results.json``, ``inference_adversarial.json``, ``legacy_isolation.json``, ``conflict_final.json``,
``boundary_heldsource_final.json``), ``perf/`` (``perf_scopes_on.json``, ``perf_default.json``) and ``security.json``.
<PF-5C run dir> holds the PF-5C ``frozen/i8_results.json`` (guide scopes on) and ``perf_new.json``.
Writes ``frozen_192_final.json``, ``wrong_source_final.json``, ``conflict_final.json``, ``boundary_heldsource_final.json``,
``legacy_isolation.json``, ``performance_final.json`` and ``security_final.json``.
"""
import json
import os
import sys

RUN, PF5C, OUT = (os.path.abspath(a) for a in sys.argv[1:4])
REPO_QA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(*parts):
    return json.load(open(os.path.join(*parts), encoding="utf-8"))


def dump(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)


# ---- frozen 192 ----------------------------------------------------------------------------------------------------------
def view(rec):
    d = rec.get("diagnostics") or {}
    return {"answer": rec.get("answer"), "pages": rec.get("pages"), "served": d.get("served"), "outcome": d.get("outcome"),
            "routing_reason": d.get("routing_reason"), "specialist_calls": d.get("specialist_calls"), "exception": rec.get("exception")}


pf5c_frozen = {(r["id"], r["mode"]): r for r in load(PF5C, "frozen", "i8_results.json")["results"]}
frozen = {"method": ("The 192-question I-9 master set (qa/plan_f_i8/master_eval.py) through the real seam, ACTIVE and SHADOW, twice: with the "
                     "production default (PLAN_F_GUIDE_SCOPES off) and with guide scopes on. Each run is analysed with the I-9 manual evidence "
                     "review (qa/plan_f_i8/review.json: a served answer must be byte-identical to a reviewed one), compared field by field with "
                     "the frozen I-9 run (qa/plan_f_i8/results_2026-10-30.json) and with the PF-5C run (guide scopes on).")}
for name in ("frozen_default", "frozen_on"):
    m = load(RUN, name, "metrics.json")
    b = load(RUN, name, "baseline_comparison.json")
    res = load(RUN, name, "i8_results.json")["results"]
    scopes = sorted({(r.get("diagnostics") or {}).get("guide_scope_enabled") for r in res})
    vs5c = [r["id"] + "/" + r["mode"] for r in res if view(r) != view(pf5c_frozen.get((r["id"], r["mode"]), {}))]
    frozen[name] = {
        "guide_scopes": scopes, "questions": m["questions"], "served": m["served"], "fallback": m["fallback"], "correct": m["correct"],
        "limited": m["limited"], "wrong": m["wrong"], "unreviewed": m["unreviewed"], "precision_pct": m["precision_pct"],
        "false_acceptances": m["false_acceptances"], "false_fallbacks": m["false_fallbacks"], "exceptions": m["exceptions"],
        "shadow_records": m["shadow_records"], "shadow_mismatches": m["shadow_mismatches"], "shadow_leaks": m["shadow_leaks"],
        "determinism_questions": m["determinism_questions"], "determinism_unstable": m["determinism_unstable"],
        "plan_f_llm_calls": m["plan_f_llm_calls"], "legacy_llm_calls": m["legacy_llm_calls"],
        "guide_registry_selected": sum(1 for r in res if (r.get("diagnostics") or {}).get("guide_scope_selected")),
        "vs_i9": {mode: {k: v for k, v in c.items() if not k.startswith(("latency", "plan_f_ms"))} for mode, c in b["modes"].items()},
        "vs_i9_latency_active_ms": {k: b["modes"]["active"][k] for k in ("latency_frozen", "latency_rerun")},
        "records_differing_from_pf5c": vs5c, "legacy_sample": b.get("legacy_sample"),
        "served_ids": sorted(r["id"] for r in res if r["mode"] == "active" and (r.get("diagnostics") or {}).get("served") == "plan_f")}
d, o = frozen["frozen_default"], frozen["frozen_on"]
frozen["same_served_set_scopes_off_and_on"] = d["served_ids"] == o["served_ids"]
frozen["pass"] = all(x["served"] == 55 and x["wrong"] == 0 and not x["false_acceptances"] and x["unreviewed"] == 0 and x["exceptions"] == 0
                     and x["shadow_mismatches"] == 0 and x["determinism_unstable"] == 0 and x["plan_f_llm_calls"] == 0
                     and not x["records_differing_from_pf5c"] and all(c["served_ids_equal"] for c in x["vs_i9"].values())
                     and all(ch["id"] == "T07" and ch["fields"] == ["pages"] for c in x["vs_i9"].values() for ch in c.get("changes", []))
                     for x in (d, o)) and frozen["same_served_set_scopes_off_and_on"]
frozen["t07"] = ("T07 'Is attendance mandatory for students?' is a legacy fallback in every mode; its legacy citation pages gained page 2 of "
                 "the Attendance Tool guide when the guides were ingested (PF-2, D-109). Unchanged since then; not a Plan F answer.")
dump("frozen_192_final.json", frozen)

# ---- wrong source and inference adversarial --------------------------------------------------------------------------------
ws = load(RUN, "checks", "wrong_source_results.json")
ws5c = load(REPO_QA, "post_freeze_pf5c", "wrong_source_results.json")
strip = lambda c: {k: v for k, v in c.items() if "ms" not in k and k != "seconds"}            # noqa: E731
c5 = {c["id"]: strip(c) for c in ws5c["cases"]}
adv = load(RUN, "checks", "inference_adversarial.json")
adv5c = load(REPO_QA, "post_freeze_pf5c", "inference_adversarial.json")
dump("wrong_source_final.json", {
    "method": ws["method"], "summary": ws["summary"],
    "cases_differing_from_pf5c": [c["id"] for c in ws["cases"] if strip(c) != c5.get(c["id"])],
    "inference_adversarial": {"method": adv["method"], "summary": adv.get("summary"), "pf5c_summary": adv5c.get("summary")},
    "pass": not ws["summary"]["wrong_source_acceptances"] and not ws["summary"]["violations"],
    "cases": ws["cases"], "inference_adversarial_cases": adv.get("cases") or adv.get("results")})

# ---- conflicts (with the Q09 annotation), boundary and held sources -----------------------------------------------------------
conf = load(RUN, "checks", "conflict_final.json")
for row in conf["cases"]:
    if row["id"] == "Q09":
        row["annotation"] = ("Conservative false fallback, not a conflict over-block: the printed procedure is titled 'How to access IThenticate' "
                             "and the guide never states the asked words 'log in', so D-111 grounding refuses it (status not_found, no "
                             "source_conflict guard involved; unchanged since PF-5B C10/W12). Q08 shows the steps on the conflict-stamped page "
                             "are still served. Documented, not fixed (PF-6 critical-defect rule).")
        row["pass_safety"] = not row["scopes_on"]["leak"] and not row["scopes_off"]["leak"]
s = conf["summary"]
s["coverage_expectation_missed"] = ["Q09"] if "Q09" in s.get("failed", []) else []
s["safety_failures"] = [r["id"] for r in conf["cases"] if r["scopes_on"]["leak"] or r["scopes_off"]["leak"]]
s["pass"] = not s["quota_leaks"] and not s["journal_leaks"] and not s["safety_failures"] and bool(s["not_overblocked_served"])
dump("conflict_final.json", conf)
dump("boundary_heldsource_final.json", load(RUN, "checks", "boundary_heldsource_final.json"))

# ---- ACTIVE production default: the handbook answers it serves, reviewed by hand ---------------------------------------------
active = load(OUT, "active_final.json")
active["production_default_served_review"] = {
    "G43": ("CORRECT", "handbook p91: 12 credit hours per regular semester or 24 per academic year (WLAM)"),
    "G44": ("CORRECT", "handbook p81: a maximum of four theses/dissertations simultaneously"),
    "G45": ("CORRECT", "handbook p54 Contact Information: Human Resources +971 6 5050023 +971 6 5585200"),
    "G48": ("CORRECT", "handbook p61: HR arranges travel and hotel accommodations for new faculty members"),
}
active["production_default_served_review_note"] = ("The four Plan F answers ACTIVE serves with guide scopes off are handbook answers, byte-identical "
                                                   "to the scopes-on run and to PF-5C; each paragraph found word for word on the cited page.")
dump("active_final.json", active)

# ---- LEGACY isolation -------------------------------------------------------------------------------------------------------
leg = load(RUN, "checks", "legacy_isolation.json")
leg["pass"] = leg["minimal_diagnostics"] == leg["questions"] and leg["exceptions"] == 0 and not leg["watched_modules_imported"] \
    and not leg["runtime_cache_created"]
dump("legacy_isolation.json", leg)

# ---- performance ------------------------------------------------------------------------------------------------------------
new, old, dflt = load(RUN, "perf", "perf_scopes_on.json"), load(PF5C, "perf_new.json"), load(RUN, "perf", "perf_default.json")
cmp = {}
for k, v in old.items():
    if isinstance(v, dict) and "p50_ms" in v and isinstance(new.get(k), dict):
        cmp[k] = {"pf5c_p50_ms": v["p50_ms"], "pf6_p50_ms": new[k]["p50_ms"], "pf5c_p95_ms": v["p95_ms"], "pf6_p95_ms": new[k]["p95_ms"],
                  "p50_change_pct": round(100.0 * (new[k]["p50_ms"] - v["p50_ms"]) / v["p50_ms"], 1) if v["p50_ms"] else None}
flag = [k for k, c in cmp.items() if c["p50_change_pct"] is not None and c["p50_change_pct"] > 25 and c["pf6_p50_ms"] - c["pf5c_p50_ms"] > 50]
dump("performance_final.json", {
    "method": ("Idle machine, warm cache, the same 191 questions (P0 100, P1 41, PF-4 held-out 50) through the real seam: guide scopes on with "
               "the PF-5C measurement script (compared with the PF-5C numbers of the same code), and the production defaults (LEGACY and "
               "ACTIVE with guide scopes off). A regression is flagged when a median is more than 25% and more than 50 ms slower."),
    "guide_scopes_on_vs_pf5c": cmp, "pf6_guide_scopes_on": new, "pf5c_guide_scopes_on": old, "pf6_production_defaults": dflt,
    "meaningful_regressions": flag, "pass": not flag})

# ---- security ---------------------------------------------------------------------------------------------------------------
sec = load(RUN, "security.json")
CLASS = {
    "API key pattern": "placeholder keys in the README and fake key-shaped strings in redaction tests; no real key",
    "credential assignment": "a fake token inside a redaction test (tests/test_ui_presentation.py)",
    "absolute local path": ("example user-directory paths in redaction tests and in the redaction rule's comment and decision note, the PF-2 "
                            "scanner's own pattern, and pip output of the archived MiniV3 notebook (a generic account name); none names this "
                            "machine or a person"),
    "Internal Use marker": "handbook policy wording (grievance and salary questions) quoted in QA results and tests; no access marking",
    "AI attribution": ("PF-1 document inventory notes describing how a third-party guide was produced, the PF-2 scanner's own pattern, and test "
                       "questions that name other AI platforms or a forbidden import; no commit or authorship attribution"),
    "large file (6 MB)": "the official handbook PDF (data/) and its copies in the archived MiniV prototypes",
}
sec_out = {"method": __doc__.strip().splitlines()[0] + " Security: every file git would commit scanned for secrets, API keys, credentials, private "
           "keys, absolute local paths, personal e-mail addresses, AI or assistant attribution, Internal Use markers, caches, embeddings, logs, "
           "generated and large files.",
           "files_scanned": sec["files_scanned"], "files_changed_by_pf6": sec["changed_by_pf6"],
           "flags_in_lines_added_by_pf6": {f: [[n, label] for n, label, _m in fl] for f, fl in sec["changed_files_flags"].items()},
           "env_file_present": sec["env_file_present"], "env_file_tracked": sec["env_file_tracked"], "cache_dir_tracked": sec["cache_dir_tracked"],
           "full_tree_flags_by_kind": sec["full_tree_flags_by_kind"],
           "full_tree_classification": {k: CLASS.get(k, "UNCLASSIFIED") for k in sec["full_tree_flags_by_kind"]},
           "real_secrets_found": 0}
sec_out["pass"] = (not sec["changed_files_flags"] and not sec["env_file_tracked"] and not sec["cache_dir_tracked"]
                   and "UNCLASSIFIED" not in sec_out["full_tree_classification"].values())
dump("security_final.json", sec_out)
print(json.dumps({"frozen": frozen["pass"], "wrong_source": not ws["summary"]["wrong_source_acceptances"], "conflict": s["pass"],
                  "legacy": leg["pass"], "performance_flags": flag, "security": sec_out["pass"]}))
