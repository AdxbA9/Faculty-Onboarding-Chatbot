"""Step 4.14-A deterministic replay runner for Faculty Services & HR specific-claim cases.

Runs every case of a JSONL file through the HR specialist on the shared test fixtures (the verbatim handbook pages of
tests/faculty_services_support.py, C12 corpus) under both stand-in rankings ("raw" token overlap, "high" scripted 9.0),
exactly as tests/test_faculty_services_support_gate.py does. No model is loaded, no network, no production file is touched.

Usage (from anywhere; the repository is read-only):
  python -X utf8 hr_case_runner.py --repo "<repo root>" --cases hr_4_14_a_cases.jsonl --out results.jsonl

Case fields: case_id, failure_class (A|B|C|D|E|F|CONTROL), question, expected (NOT_SUPPORTED | SUPPORTED_OK | PARTIAL_PREFERRED),
evidence_pages, evidence_quote, rationale, origin, optional crafted_lines (extra fixture page lines, page 107 by default),
optional clauses (focus clauses as the Coordinator would split them).

Outcome labels written per ranking:
  materially_wrong_supported  expected NOT_SUPPORTED but the specialist answered SUPPORTED   (the defect class this step hunts)
  conservative_miss           expected SUPPORTED_OK but the specialist answered partial / not_found
  over_support_vs_preference  expected PARTIAL_PREFERRED but SUPPORTED (acceptable only when the condition is asked; reviewed by hand)
  as_expected                 everything else
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated case ids")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    args.cases, args.out = os.path.abspath(args.cases), os.path.abspath(args.out)   # resolved before the chdir below
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "tests"))
    sys.path.insert(0, os.path.join(repo, ".venv", "Lib", "site-packages"))
    os.chdir(repo)                                                      # the registry path is repository-relative
    import faculty_services_support as S                                # noqa: E402  (the project's own fixtures)
    from handbook_bot.agents.specialists.contracts import FindingStatus  # noqa: E402

    rm = S.registry_and_map()
    base_pages = list(S.C12_CORPUS_PAGES)
    cache = {}

    def specialists_for(crafted_lines):
        key = tuple(crafted_lines or ())
        if key not in cache:
            pages = base_pages + ([S.page(107, *crafted_lines)] if crafted_lines else [])
            corpus = S.corpus_from(rm, pages)
            cache[key] = {"raw": S.make_specialist(corpus), "high": S.make_specialist(corpus, S.ScriptedReranker([], default=9.0))}
        return cache[key]

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip() and not l.startswith("#")]
    results, counts = [], {}
    t0 = time.time()
    for c in cases:
        if only and c["case_id"] not in only:
            continue
        sps = specialists_for(c.get("crafted_lines"))
        row = dict(c)
        row["runs"] = {}
        for ranking, sp in sps.items():
            kw = {"clauses": c["clauses"]} if c.get("clauses") else {}
            r = S.run(sp, c["question"], **kw)
            status = r.status.value if hasattr(r.status, "value") else str(r.status)
            supported = r.status is FindingStatus.SUPPORTED
            exp = c["expected"]
            if exp == "NOT_SUPPORTED" and supported:
                outcome = "materially_wrong_supported"
            elif exp == "SUPPORTED_OK" and not supported:
                outcome = "conservative_miss"
            elif exp == "PARTIAL_PREFERRED" and supported:
                outcome = "over_support_vs_preference"
            else:
                outcome = "as_expected"
            counts[outcome] = counts.get(outcome, 0) + 1
            row["runs"][ranking] = {
                "status": status, "outcome": outcome,
                "quotes": [{"page": f.page, "quote": f.evidence_quote[:300], "details_covered": sorted(f.metadata.get("details_covered", []))} for f in r.findings][:3],
                "missing": list(r.missing)[:6],
            }
        results.append(row)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("cases %d | runs %d | %.1fs" % (len(results), 2 * len(results), time.time() - t0))
    for k in ("materially_wrong_supported", "over_support_vs_preference", "conservative_miss", "as_expected"):
        print("  %-28s %d" % (k, counts.get(k, 0)))
    for row in results:
        flags = {k: v["outcome"] for k, v in row["runs"].items() if v["outcome"] != "as_expected"}
        if flags:
            print("  [%s %s] %s -> %s" % (row["case_id"], row["failure_class"], row["question"][:90], flags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
