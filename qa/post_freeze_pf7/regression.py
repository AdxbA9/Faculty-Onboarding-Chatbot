"""PF-7 (D-117) regression against the PF-6 freeze: the PF-6 verification rerun on the D-117 code without a model key
(deterministic), compared record by record with the PF-6 runs (kept outside the repository).

    python -X utf8 qa/post_freeze_pf7/regression.py <PF-7 run dir> <PF-6 run dir> <out json>

<PF-7 run dir>: ``frozen_default/`` and ``frozen_on/`` (``qa/plan_f_i8/master_eval.py`` with guide scopes off and on, analysed
by ``qa/plan_f_i8/analyze.py`` and ``qa/post_freeze_pf2/baseline_compare.py``), ``derived/`` (``qa/post_freeze_pf6/derive.py``
run on the five-set evaluation with the PF-6 results as the baseline), ``checks/`` (wrong source, inference adversarial,
conflict, boundary and held sources, LEGACY isolation) and ``tests/`` (gate and full-suite logs).
<PF-6 run dir>: ``frozen_default/i8_results.json``, ``frozen_on/i8_results.json`` and ``eval/results.json``.
"""
import json
import os
import re
import sys

NEW, OLD, OUT = (os.path.abspath(a) for a in sys.argv[1:4])
FIELDS = ("answer", "pages", "served", "outcome")


def load(*parts):
    return json.load(open(os.path.join(*parts), encoding="utf-8"))


def view(rec):
    d = rec.get("diagnostics") or {}
    return {"answer": rec.get("answer"), "pages": rec.get("pages"), "served": d.get("served"), "outcome": d.get("outcome")}


def frozen(name):
    new = load(NEW, name, "i8_results.json")["results"]
    old = {(r["id"], r["mode"]): r for r in load(OLD, name, "i8_results.json")["results"]}
    diffs = []
    for r in new:
        a, b = view(r), view(old.get((r["id"], r["mode"]), {}))
        if a != b:
            diffs.append({"id": r["id"], "mode": r["mode"], "fields": [k for k in FIELDS if a[k] != b[k]],
                          "pf6": {k: b[k] for k in FIELDS if a[k] != b[k] and k != "answer"},
                          "pf7": {k: a[k] for k in FIELDS if a[k] != b[k] and k != "answer"}})
    m = load(NEW, name, "metrics.json")
    b = load(NEW, name, "baseline_comparison.json")
    return {"served": m["served"], "correct": m["correct"], "limited": m["limited"], "wrong": m["wrong"], "unreviewed": m["unreviewed"],
            "false_acceptances": m["false_acceptances"], "exceptions": m["exceptions"], "shadow_mismatches": m["shadow_mismatches"],
            "determinism_unstable": m["determinism_unstable"], "plan_f_llm_calls": m["plan_f_llm_calls"],
            "same_served_as_i9": all(c["served_ids_equal"] for c in b["modes"].values()),
            "changes_vs_i9": {mode: c.get("changes", []) for mode, c in b["modes"].items()},
            "records_differing_from_pf6": diffs}


def tail(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read().strip().splitlines()[-1]
    except OSError:
        return None


def five_set_records():
    """Every five-set record (any mode) whose served path, outcome, answer or pages differ from PF-6."""
    new, old = load(NEW, "eval", "results.json"), load(OLD, "eval", "results.json")
    questions = {q["id"]: q["question"] for q in new["questions"]}
    before = {(r["id"], r["mode"]): r for r in old["results"]}
    rows = []
    for r in new["results"]:
        a, b = view(r), view(before.get((r["id"], r["mode"]), {}))
        if a != b:
            rows.append({"id": r["id"], "mode": r["mode"], "question": questions.get(r["id"]), "pf6": b, "pf7": a})
    return rows


derived = os.path.join(NEW, "derived")
p0, p1, held = load(derived, "p0_final.json"), load(derived, "p1_final.json"), load(derived, "held_out_diagnostics.json")
det, shadow, active = load(derived, "determinism.json"), load(derived, "shadow_final.json"), load(derived, "active_final.json")
xdoc = load(derived, "cross_document_final.json")
checks = os.path.join(NEW, "checks")
gate, full = tail(os.path.join(NEW, "tests", "gate.log")), tail(os.path.join(NEW, "tests", "full.log"))
out = {
    "method": __doc__.strip(),
    "frozen_192_scopes_off": frozen("frozen_default"),
    "frozen_192_scopes_on": frozen("frozen_on"),
    "five_sets": {
        "p0_core": {k: v for k, v in p0["core_guide"].items() if k != "served_ids"},
        "p0_all": {k: v for k, v in p0["all_guide"].items() if k != "served_ids"}, "p0_changed_vs_pf6": p0["changed_vs_pf5c"],
        "p1": {k: v for k, v in p1["all"].items() if k != "served_ids"}, "p1_changed_vs_pf6": p1["changed_vs_pf5c"],
        "held_out_diagnostics": {k: {kk: v[kk] for kk in ("served", "wrong", "changed_vs_pf5c")} for k, v in held["sets"].items()},
        "cross_document": xdoc["summary"], "shadow": shadow["summary"],
        "determinism": {k: v for k, v in det.items() if k != "method"},
        "active_scopes_off": active["production_default_scopes_off"], "active_scopes_on": active["scopes_on"],
        "records_differing_from_pf6": five_set_records(),
        "note": "'changed_vs_pf5c' fields of the derive step compare with the baseline given to it, here the PF-6 results."},
    "wrong_source": load(checks, "wrong_source_results.json")["summary"],
    "inference_adversarial": load(checks, "inference_adversarial.json")["summary"],
    "conflict": load(checks, "conflict_final.json")["summary"],
    "boundary_and_held_sources": load(checks, "boundary_heldsource_final.json")["summary"],
    "legacy_isolation": load(checks, "legacy_isolation.json"),
    "tests": {"targeted_gate": gate, "full_suite": full},
}
json.dump(out, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({"frozen_off_diffs": len(out["frozen_192_scopes_off"]["records_differing_from_pf6"]),
                  "frozen_on_diffs": len(out["frozen_192_scopes_on"]["records_differing_from_pf6"]),
                  "p0": out["five_sets"]["p0_changed_vs_pf6"], "p1": out["five_sets"]["p1_changed_vs_pf6"], "tests": out["tests"]}))
