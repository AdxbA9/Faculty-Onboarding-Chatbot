"""PF-3C versus PF-3D, per question, from the two ``metrics.json`` files (no answer text): writes ``guide_results.json`` and
adds the after-PF-3D state to ``fallback_classification.json`` (its before-change measurement is left as captured).

    python -X utf8 qa/post_freeze_pf3d/compare.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PF3C = os.path.join(os.path.dirname(HERE), "post_freeze_pf3c")
BEFORE = json.load(open(os.path.join(PF3C, "metrics.json"), encoding="utf-8"))
AFTER = json.load(open(os.path.join(HERE, "metrics.json"), encoding="utf-8"))
CLASS_PATH = os.path.join(HERE, "fallback_classification.json")
CLASSES = json.load(open(CLASS_PATH, encoding="utf-8"))
B = {r["id"]: r for r in BEFORE["rows"]}
A = {r["id"]: r for r in AFTER["rows"]}
KEEP = ("guide_served", "outcome", "forms", "parts", "pages", "stop_layer", "verdict")

rows = []
for qid, a in A.items():
    b = B.get(qid) or {}
    rows.append({"id": qid, "category": a["category"], "expected_source": a["expected_source"], "pf3c": {k: b.get(k) for k in KEEP},
                 "pf3d": {k: a.get(k) for k in KEEP}, "changed": any(b.get(k) != a.get(k) for k in ("guide_served", "outcome", "pages"))})


def core(metrics):
    c = metrics["core_guide"]
    return {k: c[k] for k in ("questions", "plan_f_served", "fallback", "coverage_pct", "correct", "limited", "wrong", "precision_pct", "forms")}


summary = {"core_guide": {"pf3c": core(BEFORE), "pf3d": core(AFTER)},
           "per_source": {src: {"pf3c": "%d of %d" % (BEFORE["per_source"][src]["served"], BEFORE["per_source"][src]["questions"]),
                                "pf3d": "%d of %d" % (v["served"], v["questions"]), "pf3d_wrong": v["wrong"]} for src, v in AFTER["per_source"].items()},
           "all_served_guide_answers": {"pf3c": BEFORE["newly_served"]["verdicts"], "pf3d": AFTER["newly_served"]["verdicts"]},
           "newly_served": sorted(r["id"] for r in rows if r["pf3d"]["guide_served"] and not r["pf3c"]["guide_served"]),
           "no_longer_served": sorted(r["id"] for r in rows if r["pf3c"]["guide_served"] and not r["pf3d"]["guide_served"]),
           "other_changes": sorted(r["id"] for r in rows if r["changed"] and r["pf3d"]["guide_served"] == r["pf3c"]["guide_served"])}
json.dump({"method": __doc__.strip(), "summary": summary, "questions": rows}, open(os.path.join(HERE, "guide_results.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)

after = {}
for qid, entry in CLASSES["questions"].items():
    served = bool((A.get(qid) or {}).get("guide_served"))
    after[qid] = {"class_before": entry["class"], "served_after": served, "verdict_after": (A.get(qid) or {}).get("verdict"),
                  "stop_layer_after": None if served else (A.get(qid) or {}).get("stop_layer")}
counts = Counter(v["class_before"] for v in after.values())
CLASSES["after_pf3d"] = {
    "note": "Added after the PF-3D evaluation; the classification above was captured before any PF-3D code change.",
    "summary": {"fallbacks_before": len(after), "legitimate": counts["LEGITIMATE"], "structural_identified": counts["STRUCTURAL"],
                "structural_recovered": sorted(q for q, v in after.items() if v["class_before"] == "STRUCTURAL" and v["served_after"]),
                "structural_remaining": sorted(q for q, v in after.items() if v["class_before"] == "STRUCTURAL" and not v["served_after"]),
                "legitimate_served": sorted(q for q, v in after.items() if v["class_before"] == "LEGITIMATE" and v["served_after"]),
                "non_structural_remaining": sorted(q for q, v in after.items() if v["class_before"] == "NON_STRUCTURAL" and not v["served_after"]),
                "fallbacks_after": sum(1 for v in after.values() if not v["served_after"])},
    "questions": after}
json.dump(CLASSES, open(CLASS_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
print(json.dumps(CLASSES["after_pf3d"]["summary"], indent=1))
