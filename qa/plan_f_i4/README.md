# Findings-aware verification on the real knowledge base (runtime integration, phase I-4)

Re-evaluation of the I-3 question set after the findings-aware verification
layer (`handbook_bot/plan_f_verification.py`) was connected between the I-1
guards and rendering. Nothing in this folder changes behaviour; the four
frozen specialists, the Coordinator, the dispatcher and the legacy Verifier
were not modified in I-4. The historical I-3 results stay in `qa/plan_f_i3/`.

| File | Purpose |
|---|---|
| `reverify.py` | builds the real knowledge base through `build_knowledge_base` (real embedder and cross-encoder reranker, production chunking, metadata, source registry and section map, the existing v12 cache), builds the four specialists, then for the 62 I-3 questions plus 14 probes records (1) the verifier's own verdict on the dispatched specialist result, called directly so that even results the I-1 gate stops are classified, (2) the adapter outcome with its verification record and timings, (3) the pipeline answer in SHADOW mode (compared with the direct `answer_question` call) and in ACTIVE mode with the I-2/I-4 diagnostics; the switches are set on the config module in-process, `.env` is never read or written |
| `analyze.py` | the 23-case table against I-3 (previous review verdict, new reason code, completeness, pages, ACTIVE served by Plan F or fallback), verifier code and request-kind distributions, verification latency separately from retrieval and reranking, shadow invariance, outcome and acceptance counts, the known cases, the probes, diagnostics hygiene |
| `review_accepted.py` | for every answer served by Plan F in the ACTIVE run, the exact production chunk and the surrounding printed page text, for the manual correctness review |
| `results_2026-10-30.json` | the recorded run of 2026-10-30 (no model client configured, so the legacy model path returned its no-key reply; Plan F made 0 model calls) |

Run from the repository root with the project interpreter:

    python -X utf8 qa/plan_f_i4/reverify.py . <output dir>
    python -X utf8 qa/plan_f_i4/analyze.py <output dir>/i4_results.json qa/plan_f_i3/results_2026-10-30.json
    python -X utf8 qa/plan_f_i4/review_accepted.py . <output dir>/i4_results.json

Recorded on 2026-10-30 (CPU, Windows): knowledge base from the v12 cache in 22.6 s (272 pages, 13,688 chunks); 62 questions plus 14 probes; of the 23 I-3 accepted cases 20 are still served by Plan F (all 20 previously CORRECT) and 3 fall back (the 3 previously ACCEPTABLE BUT LIMITED: a definition whose subject is only named, a quote spanning two provisions, an enumerative ask answered by one list item); ACTIVE 20 accepted (32%), 42 fallback (68%); SHADOW identical to the direct legacy call for all 62; verification overhead median 0.6 ms in the adapter (max 1.2 ms) against a specialist retrieval-and-rerank median of 590 ms; 0 wrong served answers on review (20 correct, 0 limited); Plan F 0 model calls; the known HR and annual-leave cases blocked. See `docs/STATUS.md`, entry of
2026-10-30 (phase I-4), for the full record.
