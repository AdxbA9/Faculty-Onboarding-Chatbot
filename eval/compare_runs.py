"""
Compare two runs case by case.

Two uses:

1. Determinism check - run the same golden set twice at temperature 0.0 and
   confirm the answers are identical. If they are not, every later "before vs
   after" comparison has noise in it that must be reported.

2. Flip analysis - compare a baseline run with an experiment run and report
   which cases changed verdict, and in which direction.

Usage:
    python eval/compare_runs.py RUN_A RUN_B
    python eval/compare_runs.py RUN_A RUN_B --golden eval/golden_set.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import DEFAULT_GOLDEN, grade_answer, is_refusal, load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two eval runs.")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    args = ap.parse_args()

    cases = {c["id"]: c for c in load_jsonl(args.golden)}
    a = {r["id"]: r for r in load_jsonl(args.run_a)}
    b = {r["id"]: r for r in load_jsonl(args.run_b)}
    ids = [i for i in cases if i in a and i in b]
    missing = [i for i in cases if i not in a or i not in b]

    print("Run A : " + args.run_a)
    print("Run B : " + args.run_b)
    print("Cases in both: %d   missing from one: %d" % (len(ids), len(missing)))
    ca = next(iter(a.values())).get("config", {})
    cb = next(iter(b.values())).get("config", {})
    if ca or cb:
        print("Config A: %s" % ca)
        print("Config B: %s" % cb)

    # ---- identity ---------------------------------------------------------
    same_answer = [i for i in ids if (a[i].get("answer") or "") == (b[i].get("answer") or "")]
    same_pages = [i for i in ids if list(a[i].get("pages") or []) == list(b[i].get("pages") or [])]
    same_route = [i for i in ids if a[i].get("query_type") == b[i].get("query_type")]
    print()
    print("=" * 62)
    print("  IDENTITY")
    print("=" * 62)
    print("  identical answers   %3d/%d" % (len(same_answer), len(ids)))
    print("  identical pages     %3d/%d" % (len(same_pages), len(ids)))
    print("  identical routes    %3d/%d" % (len(same_route), len(ids)))
    verdict = "YES" if len(same_answer) == len(ids) and len(same_pages) == len(ids) else "NO"
    print("  runs identical:     " + verdict)
    for i in ids:
        if i not in same_answer:
            print("    differs: %s" % i)
            print("      A: %s" % (a[i].get("answer") or "")[:160].replace("\n", " "))
            print("      B: %s" % (b[i].get("answer") or "")[:160].replace("\n", " "))

    # ---- flips ------------------------------------------------------------
    print()
    print("=" * 62)
    print("  FLIPS  (A -> B)")
    print("=" * 62)
    ref_to_ans, ans_to_ref, verdict_changes = [], [], []
    for i in ids:
        ra, rb = is_refusal(a[i].get("answer") or ""), is_refusal(b[i].get("answer") or "")
        va, _ = grade_answer(cases[i], a[i])
        vb, why = grade_answer(cases[i], b[i])
        if ra and not rb:
            ref_to_ans.append((i, vb, why))
        elif rb and not ra:
            ans_to_ref.append((i, va, vb))
        if va != vb:
            verdict_changes.append((i, va, vb))
    print("  refusal -> answer   %d" % len(ref_to_ans))
    for i, vb, why in ref_to_ans:
        print("    %-16s now %-9s %s" % (i, vb, why))
    print("  answer -> refusal   %d" % len(ans_to_ref))
    for i, va, vb in ans_to_ref:
        print("    %-16s was %-9s now %s" % (i, va, vb))
    print("  verdict changed     %d" % len(verdict_changes))
    for i, va, vb in verdict_changes:
        print("    %-16s %s -> %s" % (i, va, vb))
    if ref_to_ans:
        n_ok = sum(1 for _, vb, _ in ref_to_ans if vb == "correct")
        n_part = sum(1 for _, vb, _ in ref_to_ans if vb == "partial")
        n_bad = sum(1 for _, vb, _ in ref_to_ans if vb == "incorrect")
        n_ung = len(ref_to_ans) - n_ok - n_part - n_bad
        print()
        print("  of the %d refusal->answer flips: correct %d, partial %d, incorrect %d, ungraded %d"
              % (len(ref_to_ans), n_ok, n_part, n_bad, n_ung))
    print()


if __name__ == "__main__":
    main()
