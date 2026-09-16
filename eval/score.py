"""
Golden-set scorer.

Reads a run produced by ``run_eval.py`` plus the golden set, grades every
answer with deterministic string rules, and prints three separate metric
tables:

    1. Answer accuracy   - the Part 1 metric table (strict + weighted)
    2. Route accuracy    - expected_route vs QAResult.query_type  (bug 4)
    3. Citation accuracy - expected_pages vs QAResult.pages       (bug 3)

Answer accuracy is reported over the deterministically gradable cases only.
Cases marked ``"grading": "judge_only"`` are counted and listed but never
guessed at - they are handed to eval/judge.py. A number that mixed graded and
ungraded cases would be a fabricated metric.

Usage:
    python eval/score.py                              # newest run
    python eval/score.py eval/runs/20260905_120000.jsonl
    python eval/score.py --verbose                    # per-case detail
    python eval/score.py --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLDEN = os.path.join(EVAL_DIR, "golden_set.jsonl")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")

REFUSAL = "i do not have this information"

CORRECT, PARTIAL, INCORRECT, UNGRADED = "correct", "partial", "incorrect", "ungraded"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def norm(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace and dash variants.

    Deliberately does NOT strip digits or letters, so number matching stays
    honest.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _contains_one(needle: str, haystack: str) -> bool:
    """Test one literal term against an answer.

    Three behaviours, chosen by the shape of the term:

    * Short terms (<= 4 characters, e.g. "25", "not", "4") are matched on word
      boundaries. Plain substring matching would score "25" as present in
      "2025" and "not" as present in "note", which silently turns wrong
      answers into passes.
    * Longer terms are matched as plain substrings, so ordinary prose is not
      punished for inflection or surrounding words.
    * Anything with 6+ digits also gets a digits-only comparison, so
      "5050000" matches "+971 6 5050000" and "(06) 505-0000" alike.
    """
    n, h = norm(needle), norm(haystack)
    if not n:
        return False
    if len(n) <= 4:
        if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", h):
            return True
    elif n in h:
        return True
    dn = digits_only(needle)
    if len(dn) >= 6 and dn in digits_only(haystack):
        return True
    return False


def contains(term, haystack: str) -> bool:
    """Test a golden-set term, which may be a list of accepted alternatives.

    A plain string must appear. A list means "any of these is acceptable" -
    needed because the handbook writes dates as "25 Aug" while a correct model
    answer may write "25 August 2025". Requiring one exact spelling would score
    a correct answer as wrong.
    """
    if isinstance(term, (list, tuple)):
        return any(_contains_one(str(t), haystack) for t in term)
    return _contains_one(str(term), haystack)


def term_label(term) -> str:
    if isinstance(term, (list, tuple)):
        return "/".join(str(t) for t in term)
    return str(term)


def is_refusal(answer: str) -> bool:
    return norm(answer).rstrip(".").startswith(REFUSAL.rstrip("."))


# ---------------------------------------------------------------------------
# Grading one case
# ---------------------------------------------------------------------------
def grade_answer(case: Dict, result: Dict) -> Tuple[str, str]:
    """Return (verdict, reason)."""
    answer = result.get("answer", "") or ""
    if result.get("error"):
        return INCORRECT, "pipeline error: " + str(result["error"])

    mode = case.get("grading", "must_include")

    if mode == "refusal":
        if is_refusal(answer):
            return CORRECT, "refused as expected"
        return INCORRECT, "answered a question it should have refused"

    # Any non-refusal case that got refused is a false refusal (failure mode 1).
    if is_refusal(answer):
        return INCORRECT, "false refusal"

    banned = [b for b in case.get("must_not_include", []) if contains(b, answer)]
    if banned:
        return INCORRECT, "contains banned term(s): " + ", ".join(
            term_label(b) for b in banned)

    if mode == "judge_only":
        return UNGRADED, "needs eval/judge.py"

    if mode == "exact":
        if norm(case.get("expected_answer", "")) == norm(answer):
            return CORRECT, "exact match"
        return INCORRECT, "not an exact match"

    required = case.get("must_include", [])
    if not required:
        return UNGRADED, "no must_include terms defined"
    hits = [t for t in required if contains(t, answer)]
    misses = [term_label(t) for t in required if not contains(t, answer)]

    # "any_of": naming one valid item out of many is the whole task, e.g.
    # "Name one standing committee". Requiring all of them would be wrong.
    if mode == "any_of":
        if hits:
            return CORRECT, "named a valid item"
        return INCORRECT, "named no valid item"

    # default: must_include - every term required, some terms = partial credit
    if len(hits) == len(required):
        return CORRECT, "all %d required terms present" % len(required)
    if hits:
        return PARTIAL, "%d/%d required terms present; missing: %s" % (
            len(hits), len(required), ", ".join(misses))
    return INCORRECT, "no required terms present; missing: " + ", ".join(misses)


def grade_route(case: Dict, result: Dict) -> Optional[bool]:
    expected = case.get("expected_route")
    if not expected:
        return None
    return result.get("query_type", "") == expected


def grade_citation(case: Dict, result: Dict) -> Optional[bool]:
    expected = case.get("expected_pages") or []
    if not expected:
        return None
    return bool(set(expected) & set(result.get("pages") or []))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(n: float, d: float) -> str:
    return ("%5.1f%%" % (100.0 * n / d)) if d else "    -"


def _row(label: str, n: float, d: float) -> str:
    return "  %-34s %3s/%-3s  %s" % (label, n, d, _pct(n, d))


def report(cases: List[Dict], results: Dict[str, Dict], verbose: bool = False) -> List[Dict]:
    by_id = {c["id"]: c for c in cases}
    rows: List[Dict] = []
    for case in cases:
        res = results.get(case["id"])
        if res is None:
            continue
        verdict, reason = grade_answer(case, res)
        rows.append({
            "id": case["id"],
            "category": case.get("category", ""),
            "source": case.get("source", ""),
            "question": case["question"],
            "verdict": verdict,
            "reason": reason,
            "part1_verdict": case.get("part1_verdict", ""),
            "route_ok": grade_route(case, res),
            "citation_ok": grade_citation(case, res),
            "expected_route": case.get("expected_route", ""),
            "actual_route": res.get("query_type", ""),
            "expected_pages": case.get("expected_pages", []),
            "actual_pages": res.get("pages", []),
            "retrieved_pages": res.get("retrieved_pages", []),
            "used_llm": res.get("used_llm", False),
            "answer": res.get("answer", ""),
        })

    graded = [r for r in rows if r["verdict"] != UNGRADED]
    ungraded = [r for r in rows if r["verdict"] == UNGRADED]
    refusal_rows = [
        r for r in graded if by_id[r["id"]].get("grading") == "refusal"
    ]

    n = len(graded)
    full = sum(1 for r in graded if r["verdict"] == CORRECT)
    part = sum(1 for r in graded if r["verdict"] == PARTIAL)
    wrong = sum(1 for r in graded if r["verdict"] == INCORRECT)

    print()
    print("=" * 62)
    print("  ANSWER ACCURACY")
    print("=" * 62)
    print(_row("Fully correct", full, n))
    print(_row("Partially correct", part, n))
    print(_row("Incorrect", wrong, n))
    if refusal_rows:
        ok = sum(1 for r in refusal_rows if r["verdict"] == CORRECT)
        print(_row("Correct refusal (out-of-scope)", ok, len(refusal_rows)))
    print("  " + "-" * 50)
    print(_row("Strict success (full only)", full, n))
    weighted = full + 0.5 * part
    print("  %-34s %5.1f/%-3s %s" % (
        "Weighted success (partial = 0.5)", weighted, n, _pct(weighted, n)))
    if ungraded:
        print()
        print("  %d case(s) marked judge_only and NOT included above." % len(ungraded))
        print("  Run eval/judge.py to grade them. Do not quote a combined number.")

    # ---- Route -----------------------------------------------------------
    routed = [r for r in rows if r["route_ok"] is not None]
    if routed:
        ok = sum(1 for r in routed if r["route_ok"])
        print()
        print("=" * 62)
        print("  ROUTE ACCURACY  (classify_query - bug 4)")
        print("=" * 62)
        print(_row("Correct intent", ok, len(routed)))
        for r in [r for r in routed if not r["route_ok"]]:
            print("    %-16s expected %-12s got %s" % (
                r["id"], r["expected_route"], r["actual_route"]))

    # ---- Citation --------------------------------------------------------
    cited = [r for r in rows if r["citation_ok"] is not None]
    if cited:
        ok = sum(1 for r in cited if r["citation_ok"])
        print()
        print("=" * 62)
        print("  CITATION ACCURACY  (cited pages - bug 3)")
        print("=" * 62)
        print(_row("Cited an expected page", ok, len(cited)))
        missed_but_retrieved = [
            r for r in cited
            if not r["citation_ok"] and set(r["expected_pages"]) & set(r["retrieved_pages"])
        ]
        if missed_but_retrieved:
            print("    of the misses, %d DID retrieve the right page but cited"
                  % len(missed_but_retrieved))
            print("    a different one (citation bug, not a retrieval bug)")

    # ---- Manual (report) vs harness comparison ------------------------------
    comparable = [r for r in rows if r["part1_verdict"] and r["verdict"] != UNGRADED]
    if comparable:
        print()
        print("=" * 62)
        print("  PART 1 MANUAL (report) vs MCBV9 HARNESS  (%d comparable cases)" % len(comparable))
        print("  NOTE: handbook_bot/ IS the Part 1 code (MCBV9). This compares the")
        print("  report's hand scoring with this harness's string scoring of the same")
        print("  system; manual and automated scoring are not directly comparable.")
        print("=" * 62)
        improved = [r for r in comparable
                    if r["part1_verdict"] != "correct" and r["verdict"] == CORRECT]
        regressed = [r for r in comparable
                     if r["part1_verdict"] == "correct" and r["verdict"] != CORRECT]
        p1 = sum(1 for r in comparable if r["part1_verdict"] == "correct")
        p2 = sum(1 for r in comparable if r["verdict"] == CORRECT)
        print(_row("Manual (report) fully correct", p1, len(comparable)))
        print(_row("MCBV9 harness fully correct", p2, len(comparable)))
        print("  %-34s %3d" % ("Manual not-correct -> harness OK", len(improved)))
        print("  %-34s %3d" % ("Manual correct -> harness not OK", len(regressed)))
        for r in regressed:
            print("    DISAGREEMENT %s: %s" % (r["id"], r["reason"]))

    if verbose:
        print()
        print("=" * 62)
        print("  PER-CASE DETAIL")
        print("=" * 62)
        marks = {CORRECT: "OK  ", PARTIAL: "PART", INCORRECT: "FAIL", UNGRADED: "----"}
        for r in rows:
            print()
            print("[%s] %s  (%s)" % (marks[r["verdict"]], r["id"], r["category"]))
            print("  Q: " + r["question"])
            print("  A: " + r["answer"][:300].replace("\n", " "))
            print("  route: %s (want %s)   pages: %s (want %s)   llm: %s" % (
                r["actual_route"], r["expected_route"] or "-",
                r["actual_pages"], r["expected_pages"] or "-", r["used_llm"]))
            print("  why: " + r["reason"])

    print()
    return rows


def newest_run() -> str:
    runs = sorted(glob.glob(os.path.join(RUNS_DIR, "*.jsonl")))
    if not runs:
        raise SystemExit("No runs found. Run eval/run_eval.py first.")
    return runs[-1]


def load_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a golden-set run.")
    ap.add_argument("run", nargs="?", default="", help="run file (default: newest)")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--verbose", action="store_true", help="print every case")
    ap.add_argument("--csv", default="", help="also write per-case rows to this CSV")
    args = ap.parse_args()

    run_path = args.run or newest_run()
    print("Golden set : " + args.golden)
    print("Run        : " + run_path)

    cases = load_jsonl(args.golden)
    results = {rec["id"]: rec for rec in load_jsonl(run_path)}

    missing = [c["id"] for c in cases if c["id"] not in results]
    if missing:
        print("NOTE: %d golden case(s) absent from this run: %s%s" % (
            len(missing), ", ".join(missing[:5]), " ..." if len(missing) > 5 else ""))

    rows = report(cases, results, verbose=args.verbose)

    if args.csv and rows:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("Wrote " + args.csv)


if __name__ == "__main__":
    main()
