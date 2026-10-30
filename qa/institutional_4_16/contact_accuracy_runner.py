"""Step 4.16-B2 contact-accuracy evaluation of the Institutional Navigation directory lookups.

Replays every case of ``contact_cases.jsonl`` through the specialist on the shared test fixtures (real handbook pages
through the project's PDF loader and chunker, fake embedder) under both stand-in rankings ("raw" token overlap, "high"
scripted 9.0), judges each run with ``institutional_support.evaluate_contact_case`` (expected values, the printed
directory rebuilt from the PDF's geometry, no number outside a bound finding) and checks that both rankings, a second
run and a second instance give the same result. ``--mutate`` then switches each row-binding protection off in turn
(in-process, restored afterwards), replays the cases on the raw ranking and reports which cases catch it.

No model, no network, no index rebuild, no paid API; nothing is written inside the repository.

Usage:
  python -X utf8 contact_accuracy_runner.py --repo "<repo root>" --out <scratch dir> [--mutate]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

SWITCHES = ("_B2_ROW_START", "_B2_ROW_COMPLETE", "_B2_NEIGHBOR_CHECK", "_B2_COLUMN_HEADER", "_B2_SINGLE_ENTRY", "_B2_REFERENT_CHECK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutate", action="store_true")
    args = ap.parse_args()
    repo, out_dir = os.path.abspath(args.repo), os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    for p in (os.path.join(repo, ".venv", "Lib", "site-packages"), os.path.join(repo, "tests"), repo):
        sys.path.insert(0, p)
    os.chdir(repo)
    import institutional_support as S  # noqa: E402
    from handbook_bot.agents.specialists import institutional as I  # noqa: E402

    cases = S.contact_cases()
    rankings = {"raw": lambda: S.RawOverlapReranker(), "high": lambda: S.ScriptedReranker([], default=9.0)}

    def comparable(result):
        out = result.to_dict()
        out.pop("ms", None)
        return out

    def replay(ranking_names, check_determinism):
        cache = {}
        rows = []
        for case in cases:
            per_ranking = {}
            for name in ranking_names:
                key = (name, json.dumps(case.get("replace_lines") or {}, sort_keys=True))
                if key not in cache:
                    cache[key] = S.specialist_for_case(case, reranker=rankings[name]())
                sp = cache[key]
                result = S.run_case(sp, case)
                verdict = S.evaluate_contact_case(case, result, sp)
                verdict["ranking"] = name
                verdict["summary"] = result.summary
                verdict["missing"] = list(result.missing)
                verdict["findings"] = [{"claim": f.claim, "page": f.page, "chunk_id": f.metadata["chunk_id"],
                                        "entry": f.metadata.get("directory_entry"), "values": f.metadata.get("contact_values")} for f in result.findings]
                per_ranking[name] = comparable(result)
                if check_determinism:
                    again = comparable(S.run_case(sp, case))
                    fresh = comparable(S.run_case(S.specialist_for_case(case, reranker=rankings[name]()), case))
                    verdict["deterministic"] = again == per_ranking[name] and fresh == per_ranking[name]
                rows.append(verdict)
            if check_determinism and len(per_ranking) > 1:
                same = len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in per_ranking.values()}) == 1
                for row in rows[-len(per_ranking):]:
                    row["same_across_rankings"] = same
        return rows

    started = time.time()
    rows = replay(tuple(rankings), True)
    with open(os.path.join(out_dir, "contact_accuracy_results.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "cases": len(cases), "runs": len(rows), "seconds": round(time.time() - started, 1),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "by_category": {cat: dict(Counter(r["outcome"] for r in rows if r["category"] == cat)) for cat in sorted({r["category"] for r in rows})},
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "blockers": [r for r in rows if r["outcome"] == "blocker"],
        "mismatches": [r for r in rows if r["outcome"] == "mismatch"],
        "nondeterministic": [r["case_id"] for r in rows if not r.get("deterministic", True) or not r.get("same_across_rankings", True)],
        "numbers_given": sum(len(v) for r in rows for v in r["produced"].values()),
        "protections": I.binding_protections(),
    }
    if args.mutate:
        mutation = {}
        for switch in SWITCHES:
            assert getattr(I, switch) is True
            setattr(I, switch, False)
            try:
                mutated = replay(("raw",), False)
            finally:
                setattr(I, switch, True)
            failing = [r for r in mutated if r["outcome"] != "as_expected"]
            mutation[switch] = {"killed": bool(failing), "failing_cases": [r["case_id"] for r in failing],
                                "blocker_cases": [r["case_id"] for r in failing if r["outcome"] == "blocker"],
                                "examples": [{"case_id": r["case_id"], "outcome": r["outcome"], "blockers": r["blockers"][:2], "problems": r["problems"][:2]}
                                             for r in failing[:4]]}
        assert I.binding_protections() == {k: True for k in I.binding_protections()}
        after = replay(("raw",), False)
        summary["mutation"] = mutation
        summary["restored_replay_outcomes"] = dict(Counter(r["outcome"] for r in after))
    with open(os.path.join(out_dir, "contact_accuracy_summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("cases", "runs", "seconds", "outcomes", "status_counts", "numbers_given")}, ensure_ascii=False))
    print("blockers:", len(summary["blockers"]), "mismatches:", len(summary["mismatches"]), "nondeterministic:", summary["nondeterministic"])
    for r in summary["mismatches"][:20] + summary["blockers"][:20]:
        print("  ", r["case_id"], r["ranking"], r["outcome"], r["blockers"][:2], r["problems"][:2])
    if args.mutate:
        for switch, info in summary["mutation"].items():
            print("  %-22s killed=%s blockers=%s failing=%s" % (switch, info["killed"], info["blocker_cases"][:6], info["failing_cases"][:8]))
        print("  restored:", summary["restored_replay_outcomes"])
    return 0 if not summary["blockers"] and not summary["mismatches"] and not summary["nondeterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
