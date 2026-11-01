"""PF-5C derived QA records from the evaluation results, the review table and the PF-5B results (answer text at most 160
characters; the inference records hold signal keys, families and scores only):

* ``inference_analysis.json``: per set, how often guide inference was accepted and why it refused; for every PF-5C fresh
  question its label (answerable, system named or implicit) and what inference and Plan F did;
* ``pf4_diagnostic_comparison.json`` / ``pf5b_diagnostic_comparison.json``: the PF-4 held-out and PF-5B fresh sets before (PF-5B
  final code) and after (PF-5C), served or not and the answer hash, with every change listed;
* ``procedure_qa.json``: every served procedure, section or list whose verdict was given in the PF-5C review: pages, part count,
  first and last unit, verbatim check and the hand verdict.

    python -X utf8 qa/post_freeze_pf5c/derive.py <results.json> <review table> <PF-5B results.json> <labels.json> <qa dir>
"""
import collections
import hashlib
import json
import os
import sys

R = json.load(open(sys.argv[1], encoding="utf-8"))
REVIEW = json.load(open(sys.argv[2], encoding="utf-8"))["served"]
BEFORE = {r["id"]: r for r in json.load(open(sys.argv[3], encoding="utf-8"))["results"] if r["mode"] == "active"}
LABELS = json.load(open(sys.argv[4], encoding="utf-8"))["labels"]
OUT = sys.argv[5]
Q = {q["id"]: q for q in R["questions"]}
ACTIVE = {r["id"]: r for r in R["results"] if r["mode"] == "active"}
DIRECT = {d["id"]: d for d in R["direct"]}


def diag(rec):
    return (rec or {}).get("diagnostics") or {}


def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16] if text else None


def inference_of(qid):
    return diag(ACTIVE.get(qid)).get("guide_inference") or {}


# ---- inference analysis -------------------------------------------------------------------------------------------------
per_set = {}
for name in ("p0", "p1", "held_out", "fresh5b", "fresh5c"):
    ids = [i for i in Q if Q[i].get("set") == name]
    reasons = collections.Counter((inference_of(i).get("reason") or ("explicit" if diag(ACTIVE.get(i)).get("guide_sources_named") else "n/a"))
                                  for i in ids)
    accepted = [i for i in ids if diag(ACTIVE.get(i)).get("guide_sources_inferred")]
    per_set[name] = {"questions": len(ids), "inference_accepted": len(accepted), "accepted_ids": accepted,
                     "served_after_inference": [i for i in accepted if diag(ACTIVE.get(i)).get("plan_f_guide_served")],
                     "reasons": dict(reasons)}
fresh = []
for qid in [i for i in Q if Q[i].get("set") == "fresh5c"]:
    label, d, inf = LABELS[qid.replace("F5C_", "")], diag(ACTIVE.get(qid)), inference_of(qid)
    fresh.append({"id": qid, "category": Q[qid]["category"], "answerable": label["answerable"], "system": label["system_named_in_question"],
                  "system_inference_required": label["system_inference_required"], "expected_sources": label["expected_sources"],
                  "named": d.get("guide_sources_named"), "inferred": d.get("guide_sources_inferred"),
                  "inference": {k: inf.get(k) for k in ("accepted", "reason", "family", "candidates", "score", "runner_up", "margin", "signals")}
                  if inf else None,
                  "inferred_family_matches_label": (bool(set(d.get("guide_sources_inferred") or ()) & set(label["expected_sources"]))
                                                    if d.get("guide_sources_inferred") else None),
                  "served": d.get("served"), "guide_served": d.get("plan_f_guide_served"), "outcome": d.get("outcome"),
                  "verdict": (REVIEW.get(qid) or {}).get("verdict")})
implicit = [f for f in fresh if f["system_inference_required"]]
summary = {"per_set": per_set,
           "fresh5c_implicit": {"questions": len(implicit), "inference_accepted": sum(1 for f in implicit if f["inferred"]),
                                "inferred_matches_label": sum(1 for f in implicit if f["inferred_family_matches_label"]),
                                "inferred_wrong_guide": [f["id"] for f in implicit if f["inferred"] and f["inferred_family_matches_label"] is False],
                                "refusal_reasons": dict(collections.Counter((f["inference"] or {}).get("reason") or "n/a"
                                                                            for f in implicit if not f["inferred"]))},
           "fresh5c_not_implicit_inferred": [f["id"] for f in fresh if f["inferred"] and not f["system_inference_required"]]}
json.dump({"method": __doc__.strip(), "summary": summary, "fresh5c": fresh}, open(os.path.join(OUT, "inference_analysis.json"), "w",
          encoding="utf-8"), indent=1, ensure_ascii=False)


# ---- diagnostic sets before and after -----------------------------------------------------------------------------------
def before_after(name, out_name, title):
    rows = []
    for qid in [i for i in Q if Q[i].get("set") == name]:
        b, a = BEFORE.get(qid) or {}, ACTIVE.get(qid) or {}
        rows.append({"id": qid, "question": Q[qid]["question"], "answerable": Q[qid].get("answerable"),
                     "served_before": diag(b).get("served"), "served_after": diag(a).get("served"),
                     "guide_before": bool(diag(b).get("plan_f_guide_served")), "guide_after": bool(diag(a).get("plan_f_guide_served")),
                     "answer_sha_before": sha(b.get("answer")), "answer_sha_after": sha(a.get("answer")),
                     "changed": (diag(b).get("served"), b.get("answer")) != (diag(a).get("served"), a.get("answer")),
                     "inference": inference_of(qid).get("reason"), "verdict_after": (REVIEW.get(qid) or {}).get("verdict")})
    yes = [r for r in rows if r["answerable"] == "yes"]
    summ = {"questions": len(rows), "served_before": sum(r["served_before"] == "plan_f" for r in rows),
            "served_after": sum(r["served_after"] == "plan_f" for r in rows),
            "answerable": len(yes), "answerable_served_before": sum(r["served_before"] == "plan_f" for r in yes),
            "answerable_served_after": sum(r["served_after"] == "plan_f" for r in yes),
            "changed": [r["id"] for r in rows if r["changed"]],
            "wrong_after": [r["id"] for r in rows if r["verdict_after"] == "WRONG"]}
    json.dump({"method": title, "summary": summ, "questions": rows}, open(os.path.join(OUT, out_name), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    return summ


pf4 = before_after("held_out", "pf4_diagnostic_comparison.json",
                   "PF-4 held-out set (50, diagnostic only, exposed since PF-5B): PF-5B final code (before) versus PF-5C (after), ACTIVE "
                   "with guide scopes on. Not independent evidence of generalization.")
pf5b = before_after("fresh5b", "pf5b_diagnostic_comparison.json",
                    "PF-5B fresh set (56, diagnostic only; not used to design or tune any D-115 rule): PF-5B final code (before) versus "
                    "PF-5C (after), ACTIVE with guide scopes on. Not independent evidence of generalization.")

# ---- procedure QA ---------------------------------------------------------------------------------------------------------
procedures = []
for qid, entry in REVIEW.items():
    forms = entry.get("forms") or []
    if entry.get("basis") != "PF-5C review" or not set(forms) & {"procedure", "section", "list_item"}:
        continue
    items = ACTIVE[qid].get("items") or []
    quotes = [str(i.get("quote") or "") for i in items]
    procedures.append({"id": qid, "set": Q[qid].get("set"), "forms": forms, "pages": entry.get("pages"), "parts": len(items),
                       "all_quotes_verbatim_in_their_chunk": all(i.get("quote_in_chunk") for i in items),
                       "first_unit": quotes[0][:160] if quotes else None, "last_unit": quotes[-1][-160:] if quotes else None,
                       "verdict": entry.get("verdict"), "note": entry.get("note")})
json.dump({"method": "Every procedure, section or list newly served in PF-5C (any set): complete start and end, no missing middle step, "
                     "printed order and hierarchy, no screenshot-only action, no cut unit (hand review in the verdict note).",
           "answers": procedures}, open(os.path.join(OUT, "procedure_qa.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps({"inference": summary, "pf4_diagnostic": pf4, "pf5b_diagnostic": pf5b,
                  "procedure_qa": [(p["id"], p["forms"], p["verdict"]) for p in procedures]}, indent=1))
