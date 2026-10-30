"""Compare two replays of the reconstructed HR cases (hr_case_runner.py output) run by run: status, outcome label and quoted
evidence (page + quote) under both rankings, per round, with the expected-status check.

  python -X utf8 compare_replay.py --before <dir> --after <dir> --out <table.md> --json <report.json>
         [--before-name results_round{n}.jsonl] [--after-name results_round{n}_after.jsonl]

A run "regresses" when a SUPPORTED_OK case that was supported before is no longer supported, or a NOT_SUPPORTED case becomes
supported. Every other changed run (a correction, a conservative downgrade, a changed quote) is listed, never hidden.
"""
import argparse
import collections
import json
import os

OUTCOMES = ("materially_wrong_supported", "over_support_vs_preference", "conservative_miss", "as_expected")


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            rows[c["case_id"]] = c
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--before-name", default="results_round{n}.jsonl")
    ap.add_argument("--after-name", default="results_round{n}_after.jsonl")
    ap.add_argument("--rounds", default="1,2,3")
    a = ap.parse_args()
    report = {"rounds": {}, "changed_runs": [], "expected_preserved": True, "violations": []}
    md = ["| Round | Cases | Runs | Outcome | Before | After |", "|---|---|---|---|---|---|"]
    for n in (int(x) for x in a.rounds.split(",")):
        before = load(os.path.join(a.before, a.before_name.format(n=n)))
        after = load(os.path.join(a.after, a.after_name.format(n=n)))
        assert set(before) == set(after), ("case sets differ", n, sorted(set(before) ^ set(after)))
        cb, ca = collections.Counter(), collections.Counter()
        for cid in sorted(before):
            for rk in ("raw", "high"):
                b, f = before[cid]["runs"][rk], after[cid]["runs"][rk]
                cb[b["outcome"]] += 1
                ca[f["outcome"]] += 1
                qb = [(q["page"], q["quote"]) for q in b.get("quotes", [])]
                qa = [(q["page"], q["quote"]) for q in f.get("quotes", [])]
                if b["status"] != f["status"] or b["outcome"] != f["outcome"] or qb != qa:
                    report["changed_runs"].append({"round": n, "case_id": cid, "ranking": rk, "question": before[cid]["question"],
                                                   "expected": before[cid]["expected"],
                                                   "before": {"status": b["status"], "outcome": b["outcome"], "quotes": qb},
                                                   "after": {"status": f["status"], "outcome": f["outcome"], "quotes": qa}})
                exp = after[cid]["expected"]
                if exp == "SUPPORTED_OK" and b["status"] == "supported" and f["status"] != "supported":
                    report["expected_preserved"] = False
                    report["violations"].append({"round": n, "case_id": cid, "ranking": rk, "why": "a SUPPORTED_OK case supported before is no longer supported"})
                if exp == "NOT_SUPPORTED" and f["status"] == "supported":
                    report["expected_preserved"] = False
                    report["violations"].append({"round": n, "case_id": cid, "ranking": rk, "why": "a NOT_SUPPORTED case is supported"})
        report["rounds"][n] = {"cases": len(before), "runs": 2 * len(before), "before": dict(cb), "after": dict(ca)}
        for i, o in enumerate(OUTCOMES):
            md.append("| %s | %s | %s | %s | %d | %d |" % (n if i == 0 else "", len(before) if i == 0 else "", 2 * len(before) if i == 0 else "",
                                                       o, cb.get(o, 0), ca.get(o, 0)))
    report["changed_run_count"] = len(report["changed_runs"])
    md += ["", "Changed runs (status, outcome or quoted evidence): **%d**" % report["changed_run_count"], "",
           "| Round | Case | Ranking | Question | Expected | Before | After |", "|---|---|---|---|---|---|---|"]
    for c in report["changed_runs"]:
        md.append("| %d | %s | %s | %s | %s | %s / %s | %s / %s |" % (c["round"], c["case_id"], c["ranking"], c["question"].replace("|", "/"),
                                                                  c["expected"], c["before"]["status"], c["before"]["outcome"],
                                                                  c["after"]["status"], c["after"]["outcome"]))
    md += ["", "Expected statuses preserved: **%s**%s" % (report["expected_preserved"],
                                                         "" if report["expected_preserved"] else " - violations: %s" % report["violations"])]
    open(a.out, "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(report, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
