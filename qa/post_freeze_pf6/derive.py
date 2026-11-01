"""PF-6 final artifacts derived from the PF-6 evaluation (qa/post_freeze_pf6/run_eval.py), the PF-5C evaluation (the comparison
baseline), the review table (qa/post_freeze_pf5c/make_review.py run on the PF-6 results: verdicts carried only for
byte-identical answers) and the metrics (qa/post_freeze_pf5c/analyze.py run on the PF-6 results). Answer text is not copied
beyond 160 characters; answers are compared by SHA-256.

    python -X utf8 qa/post_freeze_pf6/derive.py <pf6 results.json> <pf5c results.json> <review table> <pf6 metrics.json> <out dir>

Writes p0_final.json, p1_final.json, held_out_diagnostics.json, cross_document_final.json, determinism.json, shadow_final.json,
active_final.json.
"""
import hashlib
import json
import os
import sys
from collections import Counter

R = json.load(open(sys.argv[1], encoding="utf-8"))
BASE = json.load(open(sys.argv[2], encoding="utf-8"))
REVIEW = json.load(open(sys.argv[3], encoding="utf-8"))["served"]
M = json.load(open(sys.argv[4], encoding="utf-8"))
OUT = sys.argv[5]
Q = {q["id"]: q for q in R["questions"]}


def by_mode(results, mode):
    return {r["id"]: r for r in results["results"] if r["mode"] == mode}


ACTIVE, SHADOW, OFF, LEGACY = (by_mode(R, m) for m in ("active", "shadow", "active_off", "legacy"))
B_ACTIVE, B_OFF = by_mode(BASE, "active"), by_mode(BASE, "active_off")
DIRECT = {d["id"]: d for d in R["direct"]}


def diag(rec):
    return (rec or {}).get("diagnostics") or {}


def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16] if text else None


def ids(name):
    return [i for i in Q if Q[i].get("set") == name]


def compare(qids, current, before):
    rows = []
    for qid in qids:
        a, b = current.get(qid) or {}, before.get(qid) or {}
        rows.append({"id": qid, "served": diag(a).get("served"), "served_pf5c": diag(b).get("served"),
                     "guide_served": bool(diag(a).get("plan_f_guide_served")), "answer_sha": sha(a.get("answer")),
                     "identical_to_pf5c": (diag(a).get("served"), a.get("answer"), a.get("pages")) == (diag(b).get("served"), b.get("answer"), b.get("pages")),
                     "verdict": (REVIEW.get(qid) or {}).get("verdict"), "basis": (REVIEW.get(qid) or {}).get("basis")})
    return rows


def write(name, doc):
    json.dump(doc, open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)


def slim(s):
    return {k: s.get(k) for k in ("questions", "plan_f_served", "guide_served", "fallback", "correct", "limited", "wrong", "not_reviewed",
                                  "precision_pct", "coverage_pct", "served_ids")}


# ---- P0, P1 -------------------------------------------------------------------------------------------------------------
p0 = compare(ids("p0"), ACTIVE, B_ACTIVE)
write("p0_final.json", {
    "method": "P0 (the PF-3D guide evaluation, 100 questions, 37 core guide questions) with guide scopes ON, compared with PF-5C; a served "
              "answer keeps its reviewed verdict only when it is byte-identical.",
    "core_guide": slim(M["p0"]["core_guide"]), "all_guide": slim(M["p0"]["all_guide_served"]), "per_source": M["p0"]["per_source"],
    "changed_vs_pf5c": [r["id"] for r in p0 if not r["identical_to_pf5c"]], "questions": p0})
p1 = compare(ids("p1"), ACTIVE, B_ACTIVE)
write("p1_final.json", {
    "method": "P1 (41 questions of the three P1 guides, negatives, collisions, cross-document) with guide scopes ON, compared with PF-5C.",
    "all": slim(M["p1"]["all"]), "must_not_serve_guide_served": [k for k, v in M["p1"]["must_not_serve_guide"].items() if v],
    "changed_vs_pf5c": [r["id"] for r in p1 if not r["identical_to_pf5c"]], "questions": p1})

# ---- diagnostic held-out sets -----------------------------------------------------------------------------------------------
diag_sets = {}
for name, label in (("held_out", "PF-4 held-out (50)"), ("fresh5b", "PF-5B fresh (56)"), ("fresh5c", "PF-5C fresh (60)")):
    rows = compare(ids(name), ACTIVE, B_ACTIVE)
    key = {"held_out": "held_out"}.get(name, name)
    s = M[key]["all"] if key == "held_out" else M[key]["all"]
    diag_sets[name] = {"label": label, "served": s["plan_f_served"], "fallback": s["fallback"], "correct": s["correct"],
                       "limited": s["limited"], "wrong": s["wrong"], "served_ids": s["served_ids"],
                       "changed_vs_pf5c": [r["id"] for r in rows if not r["identical_to_pf5c"]]}
write("held_out_diagnostics.json", {"method": "Diagnostic only (never tuned against): the three held-out sets with guide scopes ON, compared "
                                              "with PF-5C.", "sets": diag_sets})

# ---- cross-document ---------------------------------------------------------------------------------------------------------
cross = []
for qid, q in Q.items():
    rec, d = ACTIVE.get(qid) or {}, diag(ACTIVE.get(qid))
    multi = len(set(d.get("plan_f_sources") or ())) > 1
    if q["category"] not in ("cross_document", "held_out_cross_source") and not multi:
        continue
    clauses = {}
    for item in rec.get("items") or []:
        pf = item.get("plan_f") or {}
        clauses.setdefault(str(pf.get("clause_index")), []).append({"source_id": item.get("source_id"), "page": item.get("page"),
                                                                    "chunk_type": item.get("chunk_type"),
                                                                    "quote_in_chunk": item.get("quote_in_chunk"),
                                                                    "quote_on_page_strict": item.get("quote_in_page_normalized")})
    served = d.get("served") == "plan_f"
    # every item is verbatim in its own chunk with its own page and source; a procedure step chunk prints "<heading>: N. <step>",
    # so its strict page match may fail on the label and the heading join only (checked word by word in citation_review.json)
    ok = (not served) or all(all(i["quote_in_chunk"] and i["page"] is not None and i["source_id"]
                                 and (i["quote_on_page_strict"] or i["chunk_type"] == "procedure_step") for i in its)
                             for its in clauses.values())
    cross.append({"id": qid, "set": q.get("set"), "category": q["category"], "served": d.get("served"), "outcome": d.get("outcome"),
                  "path": d.get("plan_f_path"), "sources": d.get("plan_f_sources"), "pages": rec.get("pages"),
                  "clauses": clauses if served else None, "every_clause_grounded": ok,
                  "verdict": (REVIEW.get(qid) or {}).get("verdict"),
                  "identical_to_pf5c": (d.get("served"), rec.get("answer")) == (diag(B_ACTIVE.get(qid)).get("served"), (B_ACTIVE.get(qid) or {}).get("answer"))})
write("cross_document_final.json", {
    "method": "Every cross-document question (P0 and P1 cross_document, PF-4 cross_source) and every answer resting on more than one source: "
              "each served clause with its own sources and pages (items carry their clause index), every quote verbatim in its chunk and "
              "page, the hand verdict.",
    "summary": {"questions": len(cross), "served": [c["id"] for c in cross if c["served"] == "plan_f"],
                "wrong": [c["id"] for c in cross if c["verdict"] == "WRONG"], "ungrounded": [c["id"] for c in cross if not c["every_clause_grounded"]],
                "changed_vs_pf5c": [c["id"] for c in cross if not c["identical_to_pf5c"]]},
    "questions": cross})

# ---- determinism ------------------------------------------------------------------------------------------------------------
reps = R.get("repeats", [])
diff_answer, diff_route = set(), set()
for rec in reps:
    a = ACTIVE.get(rec["id"]) or {}
    if (rec.get("answer"), rec.get("pages"), diag(rec).get("outcome")) != (a.get("answer"), a.get("pages"), diag(a).get("outcome")):
        diff_answer.add(rec["id"])
    if (diag(rec).get("routing_final"), diag(rec).get("specialist_id"), diag(rec).get("served")) != (diag(a).get("routing_final"), diag(a).get("specialist_id"), diag(a).get("served")):
        diff_route.add(rec["id"])
direct_vs_active = [qid for qid, d in DIRECT.items() if d.get("outcome") != diag(ACTIVE.get(qid)).get("outcome")]
write("determinism.json", {
    "method": "The ACTIVE pass (guide scopes ON) and three complete repeats of the determinism subset (every question except the "
              "handbook-only, known-HR and annual-leave categories) on the same process; the DIRECT pass on the same runtime; and the "
              "PF-5C answers of the same questions.",
    "repeats": len({r.get("rep") for r in reps}), "repeat_records": len(reps), "answer_or_outcome_differences": sorted(diff_answer),
    "routing_differences": sorted(diff_route), "direct_vs_active_outcome_differences": sorted(direct_vs_active),
    "active_vs_pf5c_changed": sorted(qid for qid in ACTIVE if (diag(ACTIVE[qid]).get("served"), ACTIVE[qid].get("answer")) !=
                                     (diag(B_ACTIVE.get(qid)).get("served"), (B_ACTIVE.get(qid) or {}).get("answer"))),
    "exceptions": sum(1 for r in R["results"] if r.get("exception")) + sum(1 for d in DIRECT.values() if d.get("exception"))
                  + sum(1 for r in reps if r.get("exception"))})

# ---- SHADOW and ACTIVE -----------------------------------------------------------------------------------------------------
write("shadow_final.json", {"method": "SHADOW with guide scopes ON on the five sets (307 questions): the visible answer must be the direct "
                                      "legacy answer, field by field. The frozen 192-question SHADOW pass is in frozen_192_final.json.",
                            "summary": M["shadow"]})
off_rows = []
for qid, r in OFF.items():
    d = diag(r)
    off_rows.append({"id": qid, "served": d.get("served"), "outcome": d.get("outcome"), "guide_served": bool(d.get("plan_f_guide_served")),
                     "guide_scope_selected": d.get("guide_scope_selected"), "llm": d.get("plan_f_llm_calls"), "exception": r.get("exception"),
                     "identical_to_pf5c_off": (d.get("served"), r.get("answer")) == (diag(B_OFF.get(qid)).get("served"), (B_OFF.get(qid) or {}).get("answer"))})
serve_counts = Counter(r["served"] for r in off_rows)
control = dict(M["control_scopes_off_vs_on"])
PF5C_M = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "post_freeze_pf5c", "metrics.json"),
                        encoding="utf-8"))["control_scopes_off_vs_on"]
if control["unexplained"] == PF5C_M["unexplained"]:              # the same five as PF-5C, explained by its hand review
    control["explanation"] = PF5C_M["explanation"] + " (PF-5C review; the same five in PF-6)"
    control["unexplained_after_review"] = []
else:
    control["unexplained_after_review"] = sorted(set(control["unexplained"]) - set(PF5C_M["unexplained"]))
write("active_final.json", {
    "method": "ACTIVE with the production defaults (guide scopes OFF) on the 307 questions, and ACTIVE with guide scopes ON (the evaluation "
              "configuration of P0/P1): served path, outcome, guide use, model calls, exceptions, compared with PF-5C.",
    "production_default_scopes_off": {"questions": len(off_rows), "served": dict(serve_counts),
                                      "guide_answers_served": [r["id"] for r in off_rows if r["guide_served"]],
                                      "guide_registry_selected": [r["id"] for r in off_rows if r["guide_scope_selected"]],
                                      "exceptions": [r["id"] for r in off_rows if r["exception"]],
                                      "llm_calls": sum(r["llm"] or 0 for r in off_rows),
                                      "changed_vs_pf5c": [r["id"] for r in off_rows if not r["identical_to_pf5c_off"]]},
    "scopes_on": {"questions": len(ACTIVE), "plan_f_served": sum(1 for r in ACTIVE.values() if diag(r).get("served") == "plan_f"),
                  "guide_served": sum(1 for r in ACTIVE.values() if diag(r).get("plan_f_guide_served")),
                  "exceptions": [qid for qid, r in ACTIVE.items() if r.get("exception")],
                  "llm_calls": sum(diag(r).get("plan_f_llm_calls") or 0 for r in ACTIVE.values())},
    "control_scopes_off_vs_on": control, "latency": M["latency"], "llm": M["llm"], "exceptions": M["exceptions"],
    "legacy_pass_minimal": M["legacy_minimal"], "legacy_cache_untouched": M["legacy_cache_untouched"]})
print(json.dumps({"p0_changed": [r["id"] for r in p0 if not r["identical_to_pf5c"]], "p1_changed": [r["id"] for r in p1 if not r["identical_to_pf5c"]],
                  "diag": {k: (v["served"], v["wrong"], v["changed_vs_pf5c"]) for k, v in diag_sets.items()},
                  "cross": {"n": len(cross), "served": [c["id"] for c in cross if c["served"] == "plan_f"],
                            "ungrounded": [c["id"] for c in cross if not c["every_clause_grounded"]]},
                  "determinism": {"answer_diff": sorted(diff_answer), "route_diff": sorted(diff_route), "direct": direct_vs_active},
                  "off": {"served": dict(serve_counts), "changed": [r["id"] for r in off_rows if not r["identical_to_pf5c_off"]]}}, indent=1))
