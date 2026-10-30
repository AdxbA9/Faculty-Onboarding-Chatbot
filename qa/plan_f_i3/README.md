# Plan F on the real knowledge base (runtime integration, phase I-3)

Measurement only. Nothing in this folder changes behaviour; the four frozen
specialists, the Coordinator, the dispatcher, the adapter and the pipeline
seam were not modified for it.

| File | Purpose |
|---|---|
| `runtime_measure.py` | builds the real knowledge base through `build_knowledge_base` (real embedder, real cross-encoder reranker, the production chunking, metadata, source registry and section map), times the model load, the first build and the cached rebuild, builds the four specialists with their own `from_knowledge_base` builders, then runs the 62-question set through `ui.pipeline.ask_with_diagnostics` in LEGACY, SHADOW and ACTIVE mode with a direct `answer_question` comparison, and checks the runtime cache with a second knowledge-base object; the switches are set on the config module in-process, `.env` is never read or written |
| `analyze.py` | outcome distribution, acceptance and fallback rates, per-domain counts, latency statistics (median and range), shadow invariance, the accepted answers for review, the known HR cases, diagnostics hygiene |
| `review_accepted.py` | for every accepted answer, the exact production chunk and the surrounding printed page text, for the manual correctness review |
| `results_2026-10-30.json` | the recorded run of 2026-10-30 (no model client configured, so the legacy model path returned its no-key reply; Plan F made 0 model calls) |

Run from the repository root with the project interpreter:

    python -X utf8 qa/plan_f_i3/runtime_measure.py . <output dir>
    python -X utf8 qa/plan_f_i3/analyze.py <output dir>/i3_results.json
    python -X utf8 qa/plan_f_i3/review_accepted.py . <output dir>/i3_results.json

Recorded on 2026-10-30 (CPU, Windows): model download and load 32.6 s and
22.2 s; first knowledge-base build 331.9 s (13,688 chunks from 272 pages,
206 section records), cached rebuild 14.8 s; builders 9, 10, 8 and 777 ms
(the first Institutional construction in a process; later registry builds
88 to 117 ms); 62 questions, 23 accepted (37%), 39 fallback (63%), 0 wrong
accepted answers on review (20 correct, 3 correct but limited); legacy and
shadow identical to the direct call for all 62; Plan F median 569 ms per
attempt, shadow overhead median 389 ms, accepted ACTIVE answer median 634 ms,
ACTIVE fallback median 1,314 ms. See `docs/STATUS.md`, entry of 2026-10-30
(phase I-3), for the full record.
