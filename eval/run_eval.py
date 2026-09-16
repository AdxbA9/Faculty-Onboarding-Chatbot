"""
Golden-set runner.

Loads the knowledge base once, asks every question in the golden set through
``answer_question`` (the same entry point the UI uses), and writes one JSON
object per question to ``eval/runs/<timestamp>.jsonl``.

This file does no grading. Running and scoring are separated on purpose: a run
is expensive (model load + LLM calls) while scoring is free, so a single run can
be re-scored many times as the rubric evolves.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --golden eval/golden_set.jsonl --tag baseline
    python eval/run_eval.py --limit 5            # smoke test
    python eval/run_eval.py --category contact   # one category only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional

# Allow "python eval/run_eval.py" from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from handbook_bot import build_knowledge_base
from handbook_bot.qa import answer_question

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GOLDEN = os.path.join(EVAL_DIR, "golden_set.jsonl")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")


def load_golden(path: str) -> List[Dict]:
    """Read golden_set.jsonl, skipping blank lines and # comments."""
    cases: List[Dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON - {exc}") from exc
    return cases


def run(cases: List[Dict], out_path: str, *, verbose: bool = True) -> str:
    kb = build_knowledge_base(verbose=verbose)

    if kb.groq_client is None:
        print(
            "\n  WARNING: GROQ_API_KEY is not set or invalid.\n"
            "  Only the deterministic contact/count/date paths will produce\n"
            "  answers; every policy question will report the missing-key\n"
            "  message and score as incorrect. Results are NOT comparable to\n"
            "  Part 1 in this state.\n"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    started = time.time()

    with open(out_path, "w", encoding="utf-8") as out:
        for i, case in enumerate(cases, 1):
            question = case["question"]
            if verbose:
                print(f"[{i}/{len(cases)}] {case.get('id', '?')}: {question[:70]}")
            t0 = time.perf_counter()
            try:
                result = answer_question(
                    question,
                    embedder=kb.embedder,
                    reranker=kb.reranker,
                    index=kb.index,
                    chunks=kb.chunks,
                    metadata=kb.metadata,
                    pages=kb.pages,
                    groq_client=kb.groq_client,
                )
                payload = result.to_dict()
                error = ""
            except Exception as exc:  # keep the run alive; record the failure
                payload = {}
                error = f"{type(exc).__name__}: {exc}"
                print(f"    ERROR: {error}")

            record = {
                "id": case.get("id", f"case-{i:03d}"),
                "question": question,
                "category": case.get("category", ""),
                "error": error,
                "answer": payload.get("answer", ""),
                "pages": payload.get("pages", []),
                "query_type": payload.get("query_type", ""),
                "used_llm": payload.get("used_llm", False),
                "best_section": payload.get("best_section", ""),
                "evidence": payload.get("evidence", ""),
                "num_candidates": payload.get("num_candidates", 0),
                "num_reranked": payload.get("num_reranked", 0),
                "timings": payload.get("timings", {}),
                # Retrieved page list, for diagnosing "right page never retrieved"
                # separately from "right page retrieved but not cited".
                "retrieved_pages": sorted(
                    {int(it["meta"]["page"]) for it in payload.get("items", [])}
                ),
                "wall_ms": (time.perf_counter() - t0) * 1000.0,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    if verbose:
        print(f"\nWrote {len(cases)} results to {out_path} "
              f"in {time.time() - started:.1f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the golden set through the RAG pipeline.")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN, help="path to golden_set.jsonl")
    ap.add_argument("--out", default="", help="output path (default: runs/<timestamp>.jsonl)")
    ap.add_argument("--tag", default="", help="label appended to the output filename")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    ap.add_argument("--category", default="", help="run only this category")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cases = load_golden(args.golden)
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected.")

    out_path = args.out
    if not out_path:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{stamp}_{args.tag}.jsonl" if args.tag else f"{stamp}.jsonl"
        out_path = os.path.join(RUNS_DIR, name)

    run(cases, out_path, verbose=not args.quiet)


if __name__ == "__main__":
    main()
