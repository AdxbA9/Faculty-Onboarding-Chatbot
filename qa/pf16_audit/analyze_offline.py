"""Measures from a PF-16 offline run (``offline_eval.py --set cases``): every figure with its denominator.

    python -X utf8 qa/pf16_audit/analyze_offline.py <run.json> <out.json>

Routing: the Coordinator's final specialist set (``routing_final``) against the case's expected specialists, for answerable
cases with a single expected specialist (exact) and for every case with expected specialists (overlap); refusal-type cases
where any specialist was selected are listed. Arabic: every paired case (``pair``) compared with its English pair on kind,
serving path, retrieval hit, top evidence page and specialists. Retrieval: hit anywhere / top 3 by language and category.
"""
import collections
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as f:
    run = json.load(f)
R = run["records"]
by_id = {r["id"]: r for r in R}


def frac(n, d):
    return [n, d]


out = {"run": run["summary"]}
# ---- routing ------------------------------------------------------------------------------------------------------------------
single = [r for r in R if r.get("expected_behavior") == "answer" and len(r.get("expected_specialists") or []) == 1]
multi = [r for r in R if len(r.get("expected_specialists") or []) >= 1]
got = lambda r: sorted(set(r.get("routing_final") or []))
out["routing"] = {
    "single_expected_exact": frac(sum(1 for r in single if got(r) == r["expected_specialists"]), len(single)),
    "single_expected_included": frac(sum(1 for r in single if r["expected_specialists"][0] in got(r)), len(single)),
    "single_expected_none_selected": frac(sum(1 for r in single if not got(r)), len(single)),
    "any_expected_overlap": frac(sum(1 for r in multi if set(got(r)) & set(r["expected_specialists"])), len(multi)),
    "cross_expected_all_selected": frac(sum(1 for r in R if len(r.get("expected_specialists") or []) >= 2 and set(r["expected_specialists"]) <= set(got(r))),
                                        sum(1 for r in R if len(r.get("expected_specialists") or []) >= 2)),
    "by_expected": {},
    "wrong_specialist_examples": [],
}
for spec in ("teaching", "research", "faculty_services", "institutional"):
    rows = [r for r in single if r["expected_specialists"] == [spec]]
    out["routing"]["by_expected"][spec] = {"cases": len(rows), "exact": sum(1 for r in rows if got(r) == [spec]),
                                           "included": sum(1 for r in rows if spec in got(r)), "none": sum(1 for r in rows if not got(r)),
                                           "selected_instead": dict(collections.Counter(",".join(got(r)) for r in rows if spec not in got(r)))}
for r in single:
    if got(r) and r["expected_specialists"][0] not in got(r):
        out["routing"]["wrong_specialist_examples"].append([r["id"], r["question"][:90], r["expected_specialists"], got(r)])
refuse_cases = [r for r in R if r.get("expected_behavior") == "refuse"]
out["routing"]["refuse_cases_with_a_specialist"] = frac(sum(1 for r in refuse_cases if got(r)), len(refuse_cases))

# ---- outcomes by language and category -------------------------------------------------------------------------------------------
def tally(rows):
    c = collections.Counter(r["verdict"] for r in rows)
    k = collections.Counter(r["kind"] for r in rows)
    hits = [r for r in rows if r.get("retrieval_hit") is not None]
    return {"cases": len(rows), "verdicts": dict(c), "kinds": dict(k), "served_plan_f": sum(1 for r in rows if r.get("served") == "plan_f"),
            "retrieval_hit": frac(sum(1 for r in hits if r["retrieval_hit"]), len(hits)),
            "retrieval_hit_top3": frac(sum(1 for r in hits if r["retrieval_hit_top3"]), len(hits))}


out["by_language"] = {lang: tally([r for r in R if r.get("language") == lang]) for lang in ("en", "ar", "mixed")}
out["by_category"] = {cat: tally([r for r in R if r.get("category") == cat]) for cat in sorted({r.get("category") for r in R})}
out["by_domain"] = {dom: tally([r for r in R if r.get("domain") == dom]) for dom in sorted({r.get("domain") for r in R})}
out["refusal_safety"] = {
    "expected_refuse": len(refuse_cases),
    "refused_or_notice": sum(1 for r in refuse_cases if r["kind"] in ("refusal", "notice")),
    "model_path_not_executed": sum(1 for r in refuse_cases if r["kind"] == "model_path"),
    "served_an_answer": [[r["id"], r["question"][:80]] for r in refuse_cases if r["kind"] == "answer"],
    "forbidden_fact_present_any_case": [[r["id"], r.get("forbidden_present")] for r in R if r.get("forbidden_present")],
}

# ---- Arabic / English pairs ---------------------------------------------------------------------------------------------------------
pairs = collections.defaultdict(dict)
for r in R:
    if r.get("pair"):
        pairs[r["pair"]]["en" if r["id"].startswith("PF16-PE") else ("ar" if r["id"].startswith("PF16-PA") else "stress")] = r
rows = []
for key, p in sorted(pairs.items()):
    if "en" not in p or "ar" not in p:
        continue
    en, ar = p["en"], p["ar"]
    top = lambda r: (r.get("item_pages") or [[None, None]])[0]
    rows.append({"pair": key, "en_id": en["id"], "ar_id": ar["id"], "same_kind": en["kind"] == ar["kind"], "same_served": en.get("served") == ar.get("served"),
                 "en_hit": en.get("retrieval_hit"), "ar_hit": ar.get("retrieval_hit"), "same_top_page": top(en) == top(ar),
                 "same_specialists": got(en) == got(ar), "en_kind": en["kind"], "ar_kind": ar["kind"],
                 "english_query": (ar.get("transform") or {}).get("english_query"), "coverage": ((ar.get("diagnostics") or {}).get("language") or {}).get("query_coverage"),
                 "query_method": ((ar.get("diagnostics") or {}).get("language") or {}).get("query_method"), "unmapped": (ar.get("transform") or {}).get("unmapped")})
n = len(rows)
hit_rows = [x for x in rows if x["en_hit"] is not None]
out["arabic_pairs"] = {
    "pairs": n,
    "same_kind": frac(sum(1 for x in rows if x["same_kind"]), n),
    "same_serving_path": frac(sum(1 for x in rows if x["same_served"]), n),
    "same_specialists": frac(sum(1 for x in rows if x["same_specialists"]), n),
    "same_top_evidence_page": frac(sum(1 for x in rows if x["same_top_page"]), n),
    "expected_page_hit_en": frac(sum(1 for x in hit_rows if x["en_hit"]), len(hit_rows)),
    "expected_page_hit_ar": frac(sum(1 for x in hit_rows if x["ar_hit"]), len(hit_rows)),
    "hit_en_not_ar": [x["pair"] for x in hit_rows if x["en_hit"] and not x["ar_hit"]],
    "hit_ar_not_en": [x["pair"] for x in hit_rows if x["ar_hit"] and not x["en_hit"]],
    "kind_differences": [[x["pair"], x["en_kind"], x["ar_kind"], x["english_query"]] for x in rows if not x["same_kind"]],
    "mean_coverage": round(sum(float(x["coverage"] or 0) for x in rows) / n, 3) if n else None,
    "query_methods": dict(collections.Counter(x["query_method"] for x in rows)),
    "rows": rows,
}
stress = [r for r in R if r["id"].startswith("PF16-AS")]
out["arabic_stress"] = [{"id": r["id"], "category": r.get("category"), "kind": r["kind"], "served": r.get("served"), "hit": r.get("retrieval_hit"),
                         "verdict": r["verdict"], "english_query": (r.get("transform") or {}).get("english_query"),
                         "coverage": ((r.get("diagnostics") or {}).get("language") or {}).get("query_coverage")} for r in stress]
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
show = {k: v for k, v in out.items() if k not in ("run",)}
show["arabic_pairs"] = {k: v for k, v in out["arabic_pairs"].items() if k != "rows"}
print(json.dumps(show, ensure_ascii=False, indent=1)[:9000])
