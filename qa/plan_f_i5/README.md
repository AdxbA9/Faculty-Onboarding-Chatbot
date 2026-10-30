# Deterministic aggregation on the real knowledge base (runtime integration, phase I-5)

Evaluation of the multi-finding aggregation layer (`handbook_bot/plan_f_aggregation.py`)
and the provision slice (`plan_f_verification.verify_findings_sliced`) on the real
knowledge base. Nothing in this folder changes behaviour; the four frozen specialists,
the Coordinator, the dispatcher and the legacy Verifier were not modified in I-5. The
historical I-3 and I-4 results stay in `qa/plan_f_i3/` and `qa/plan_f_i4/`.

| File | Purpose |
|---|---|
| `aggregate_eval.py` | builds the real knowledge base through `build_knowledge_base` (real embedder and cross-encoder reranker, production chunking, metadata, source registry and section map, the existing v12 cache), builds the four specialists, then for the 62 I-3 questions, the 14 I-4 probes and the 26-question aggregation set defined in the script records (1) the decision shape (request clauses, tasks, the I-1 decision problem, the aggregator's pre-dispatch request problem), (2) every dispatched run and the aggregator's direct verdict on the execution, (3) the adapter attempt with its verification and aggregation records, timings, answer, pages and items, (4) the pipeline answer in SHADOW mode (compared with the direct `answer_question` call) and in ACTIVE mode with the diagnostics; the switches are set on the config module in-process, `.env` is never read or written |
| `analyze.py` | the I-4 served set regression (same answer and pages), the former limited cases, the known HR and annual-leave cases, the multipart classification A to F of every multi-clause or multi-specialist question, acceptance before and after, aggregation latency separately from retrieval and reranking, SHADOW invariance, Plan F model calls, diagnostics hygiene, the served answers for review |
| `review_accepted.py` | for every answer served by Plan F in the ACTIVE run (aggregate or single finding), the request clauses, the per-clause coverage, the rendered parts, and for each part the exact production chunk and the surrounding printed page text, for the manual correctness review |
| `results_2026-10-30.json` | the recorded run of 2026-10-30 (no model client configured, so the legacy model path returned its no-key reply; Plan F made 0 model calls) |

Run from the repository root with the project interpreter:

    python -X utf8 qa/plan_f_i5/aggregate_eval.py . <output dir>
    python -X utf8 qa/plan_f_i5/analyze.py <output dir>/i5_results.json qa/plan_f_i4/results_2026-10-30.json
    python -X utf8 qa/plan_f_i5/review_accepted.py . <output dir>/i5_results.json

The aggregation set (26 questions, tagged by the shape each was written to exercise,
not by its expected outcome): same-domain multipart (12, including same-page,
multi-page, three-clause, two-office contact and duplicate asks), cross-domain (4),
enumerative (2), missing clause (2), mixed supported and unsupported clauses (2),
cross-domain unresolved (2), handoff required (2). Every fact asked is one the Faculty
Handbook states.

Recorded on 2026-10-30 (CPU, Windows): knowledge base from the v12 cache in 22.2 s
(272 pages, 13,688 chunks); 96 unique questions, 102 runs per mode, 434 s in all.
All 28 answers the I-4 ACTIVE run served are served again with byte-identical text
and pages through the single-finding pass-through path. Five real aggregates served
(three cross-domain: Teaching + HR on pages 96, 91 and 107; Teaching + Research on
pages 96, 91 and 81; HR + Institutional on pages 61 and 24; two same-domain questions
whose two clauses are answered by one verified sentence, pages 79 and 107), all
CORRECT on review, 0 limited, 0 wrong. The former consultancy case and its percentage
probe are served as the 16.1.7 provision slice; WLAM and the HR enumeration stay
rejected; the known HR and annual-leave cases stay blocked. Multipart classification
of the 32 multi-clause or multi-specialist runs: A 5, B 14, C 11, D 2, E 0, F 0.
Acceptance: corpus 20 of 62 (I-4) to 21 of 62; probes 8 of 14 to 9 of 14;
aggregation set 6 of 26 (5 aggregates, 1 single finding). Aggregation overhead for
the accepted aggregates median 2.6 ms (1.5 to 5.3 ms; clause mapping 0.01, per-clause
verification 1.9, conflict check 0.6, deduplication 0.01) against a specialist
retrieval-and-rerank median of 783 ms; pre-dispatch rejections 0.5 ms with no
specialist run. SHADOW identical to the direct legacy call for all 96 unique
questions; Plan F 0 model calls; 0 of 204 diagnostic records carry question or answer
text. See `docs/STATUS.md`, entry of 2026-10-30 (phase I-5), for the full record.
