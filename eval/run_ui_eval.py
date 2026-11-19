"""
Golden-set runner through the SAME entry point the web page uses (``ui.pipeline.ask_with_diagnostics``).

``run_eval.py`` calls the legacy ``answer_question`` directly. The page does more: small talk, spelling correction, the
Arabic layer and, in ACTIVE mode, the specialists with their verification. This runner measures what a user actually
gets. Records have the fields ``score.py`` reads, plus ``served`` (plan_f / legacy / smalltalk) and ``mode``.

    set PLAN_F_ENABLED=1
    set PLAN_F_GUIDE_SCOPES=1
    python eval/run_ui_eval.py --tag final
    python eval/score.py eval/runs/<file>.jsonl

    python eval/run_ui_eval.py --golden eval/extra_set.jsonl --tag extra    # Arabic, typos, small talk, refusals, guides
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from eval.run_eval import DEFAULT_GOLDEN, RUNS_DIR, load_golden  # noqa: E402
from handbook_bot import config as _cfg                           # noqa: E402
from ui import pipeline                                           # noqa: E402


def run(cases: List[Dict], out_path: str, *, verbose: bool = True, pause: float = 0.0) -> str:
    kb = pipeline.load_knowledge_base(verbose=verbose)
    mode = pipeline.plan_f_mode()
    if verbose:
        print(f"mode={mode} model={_cfg.GROQ_MODEL} groq={'yes' if kb.groq_client is not None else 'NO KEY'}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    started = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for i, case in enumerate(cases, 1):
            question = case["question"]
            t0 = time.perf_counter()
            payload: Dict = {}
            diagnostics: Dict = {}
            error = ""
            try:
                result, diagnostics = pipeline.ask_with_diagnostics(kb, question, answer_language=case.get("answer_language", "auto"))
                payload = result.to_dict()
            except Exception as exc:                               # keep the run alive; record the failure
                error = f"{type(exc).__name__}: {exc}"
            wall = (time.perf_counter() - t0) * 1000.0
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
                "retrieved_pages": sorted({int(it["meta"]["page"]) for it in payload.get("items", []) if "page" in it.get("meta", {})}),
                "wall_ms": wall,
                "llm_calls": payload.get("llm_calls", 0),
                "retried": payload.get("retried", False),
                "sub_questions": payload.get("sub_questions", []),
                "agent_trace": payload.get("agent_trace", []),
                "mode": (diagnostics or {}).get("mode", mode),
                "served": (diagnostics or {}).get("served", ""),
                "fallback_reason": (diagnostics or {}).get("fallback_reason", ""),
                "config": {"groq_model": _cfg.GROQ_MODEL, "cache_version": _cfg.CACHE_VERSION, "entry": "ui.pipeline"},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if verbose:
                flag = "ERROR " + error if error else record["served"]
                print(f"[{i}/{len(cases)}] {record['id']} {wall / 1000:.1f}s {flag}")
            if pause:
                time.sleep(pause)
    if verbose:
        print(f"\nWrote {len(cases)} results to {out_path} in {time.time() - started:.1f}s")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a question set through the page's own pipeline entry point.")
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="ui")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cases = load_golden(args.golden)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases selected.")
    out_path = args.out or os.path.join(RUNS_DIR, f"{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}.jsonl")
    run(cases, out_path, verbose=not args.quiet, pause=args.pause)


if __name__ == "__main__":
    main()
