"""Step 4.14-A controlled, reversible mutation harness for the D-099 rules.

The production module exposes one boolean switch per D-099 rule (``_C12_*``, "each rule has a switch for the mutation
harness"). Turning a switch off at runtime is a reversible mutant: no file is edited, the process restores the switch after
each run. For every mutant the harness replays the NOT_SUPPORTED cases of the reconstructed case files (raw ranking) and
records which cases turn SUPPORTED - the mutant is "killed" by those cases. A mutant no case kills is reported as
"survived" (the reconstructed set does not exercise that rule; the project's own test_faculty_services_support_gate.py
switch tests still do).

  python -X utf8 mutation_switch_harness.py --repo <repo> --cases a.jsonl b.jsonl --out mutation_results.json
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
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    cases_paths = [os.path.abspath(p) for p in args.cases]
    out_path = os.path.abspath(args.out)
    for p in (os.path.join(repo, ".venv", "Lib", "site-packages"), os.path.join(repo, "tests"), repo):
        sys.path.insert(0, p)
    os.chdir(repo)
    import faculty_services_support as S                                   # noqa: E402
    from handbook_bot.agents.specialists import faculty_services as F      # noqa: E402
    from handbook_bot.agents.specialists.contracts import FindingStatus   # noqa: E402

    switches = sorted(n for n in dir(F) if n.startswith("_C12_") and isinstance(getattr(F, n), bool))
    cases = []
    for cp in cases_paths:
        for l in open(cp, encoding="utf-8"):
            if l.strip() and not l.startswith("#"):
                c = json.loads(l)
                if c["expected"] == "NOT_SUPPORTED":
                    cases.append(c)
    rm = S.registry_and_map()
    base_pages = list(S.C12_CORPUS_PAGES)
    cache = {}

    def specialist_for(crafted_lines):
        key = tuple(crafted_lines or ())
        if key not in cache:
            pages = base_pages + ([S.page(107, *crafted_lines)] if crafted_lines else [])
            cache[key] = S.make_specialist(S.corpus_from(rm, pages))
        return cache[key]

    def replay():
        out = {}
        for c in cases:
            kw = {"clauses": c["clauses"]} if c.get("clauses") else {}
            r = S.run(specialist_for(c.get("crafted_lines")), c["question"], **kw)
            out[c["case_id"]] = r.status is FindingStatus.SUPPORTED
        return out

    t0 = time.time()
    F._C12_SPECIFIC_CACHE.clear()
    baseline = replay()
    already = sorted(k for k, v in baseline.items() if v)
    print("baseline: %d NOT_SUPPORTED cases, %d already supported (known defects): %s (%.0fs)" % (len(cases), len(already), already, time.time() - t0))
    results = {"switches": {}, "cases": len(cases), "baseline_supported": already}
    for sw in switches:
        t1 = time.time()
        setattr(F, sw, False)
        F._C12_SPECIFIC_CACHE.clear()
        try:
            mutant = replay()
        finally:
            setattr(F, sw, True)                                        # restore: the mutation is reversible and process-local
            F._C12_SPECIFIC_CACHE.clear()
        killers = sorted(k for k, v in mutant.items() if v and not baseline[k])
        results["switches"][sw] = {"killed_by": killers, "survived": not killers, "seconds": round(time.time() - t1, 1)}
        print("%-28s %s (%d killers, %.0fs)" % (sw, "SURVIVED" if not killers else "killed", len(killers), time.time() - t1), killers[:8])
    # every switch back on: confirm the baseline is reproduced exactly (determinism)
    F._C12_SPECIFIC_CACHE.clear()
    again = replay()
    results["deterministic_replay"] = again == baseline
    results["seconds"] = round(time.time() - t0, 1)
    print("deterministic replay identical:", again == baseline, "| total %.0fs" % (time.time() - t0))
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
