"""PF-3D evidence-selection review: reads the output of ``qa/post_freeze_pf3c/evidence_selection.py --pf3c`` run on the
PF-3D code and the committed PF-3C ``evidence_after.json``, and writes a compact comparison (no chunk text beyond the first
80 characters of the first unit).

Two measures per question for which the guide path selects evidence:
  * TOP-1 (the PF-3C measure): the first selected unit against the hand-labelled target passage (``guide_targets.json``):
    ON_TARGET when it is on the target source and page and holds a target phrase, PARTIAL when it is on the target page
    without one, OFF_TARGET otherwise. A D-112 section starts with its printed heading row, so its first unit is the
    heading and this measure reads PARTIAL even when the section is the target passage.
  * SET: the same categories over the whole selected evidence: every unit on the target source, at least one on a target
    page, and the units together holding a target phrase. Units on a page outside the hand-labelled target pages (a
    section continued on the next page) are listed in ``units_off_target_pages`` for the hand review.

    python -X utf8 qa/post_freeze_pf3d/evidence_review.py <pf3d evidence.json> <out.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PF3C = os.path.join(os.path.dirname(HERE), "post_freeze_pf3c")
EV = json.load(open(sys.argv[1], encoding="utf-8"))
BEFORE = {r["id"]: r for r in json.load(open(os.path.join(PF3C, "evidence_after.json"), encoding="utf-8"))["rows"]}
TARGETS = json.load(open(os.path.join(PF3C, "guide_targets.json"), encoding="utf-8"))
TARGETS = TARGETS.get("targets", TARGETS)


def fold(text):
    return " ".join((text or "").lower().split())


def set_category(qid, evidence):
    t = TARGETS[qid]
    if any(e["source_id"] != t["source"] for e in evidence) or not any(e["page"] in t["pages"] for e in evidence):
        return "OFF_TARGET"
    joined = fold(" ".join(e["text"] for e in evidence))
    return "ON_TARGET" if any(fold(p) in joined for p in t["phrases"]) else "PARTIAL"


rows = []
for r in EV["rows"]:
    evidence = (r.get("pf3c") or {}).get("evidence") or []
    before = BEFORE.get(r["id"]) or {}
    rows.append({"id": r["id"], "pf3c_top1": before.get("pf3c_top1"), "pf3d_form": (r.get("pf3c") or {}).get("form"),
                 "pf3d_top1": r.get("pf3c_top1"), "pf3d_set": set_category(r["id"], evidence) if evidence else None,
                 "units": len(evidence), "pages": sorted({e["page"] for e in evidence}),
                 "units_off_target_pages": sum(1 for e in evidence if e["page"] not in TARGETS[r["id"]]["pages"]),
                 "first_unit": evidence[0]["text"][:80] if evidence else None})


def count(key, value):
    return sum(1 for row in rows if row[key] == value)


summary = {"questions": len(rows),
           "pf3c": {"with_evidence": sum(1 for row in rows if row["pf3c_top1"]), "on_target": count("pf3c_top1", "ON_TARGET"),
                    "partial": count("pf3c_top1", "PARTIAL"), "off_target": count("pf3c_top1", "OFF_TARGET")},
           "pf3d_top1": {"with_evidence": sum(1 for row in rows if row["pf3d_top1"]), "on_target": count("pf3d_top1", "ON_TARGET"),
                         "partial": count("pf3d_top1", "PARTIAL"), "off_target": count("pf3d_top1", "OFF_TARGET")},
           "pf3d_set": {"with_evidence": sum(1 for row in rows if row["pf3d_set"]), "on_target": count("pf3d_set", "ON_TARGET"),
                        "partial": count("pf3d_set", "PARTIAL"), "off_target": count("pf3d_set", "OFF_TARGET")},
           "lost_evidence": [row["id"] for row in rows if row["pf3c_top1"] and not row["pf3d_top1"]],
           "changed_pf3c_category": [row["id"] for row in rows if row["pf3c_top1"] and row["pf3d_top1"] and row["pf3c_top1"] != row["pf3d_top1"]],
           "top1_partial_but_set_on_target": [row["id"] for row in rows if row["pf3d_top1"] == "PARTIAL" and row["pf3d_set"] == "ON_TARGET"],
           "median_ms": (EV.get("summary") or {}).get("pf3c", {}).get("median_ms")}
json.dump({"method": __doc__.strip(), "summary": summary, "rows": rows}, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1))
