# Routing refinement and bounded handoffs on the real knowledge base (runtime integration, phase I-6)

Evaluation of the deterministic routing refinement (`handbook_bot/plan_f_routing.py`) and
the one-hop handoff round (`handbook_bot/plan_f_handoff.py`) on the real knowledge base.
Nothing in this folder changes behaviour; the four frozen specialists, the dispatcher and
the legacy Verifier were not modified in I-6. The historical I-3, I-4 and I-5 results stay
in `qa/plan_f_i3/`, `qa/plan_f_i4/` and `qa/plan_f_i5/`.

| File | Purpose |
|---|---|
| `routing_eval.py` | builds the real knowledge base through `build_knowledge_base` (real embedder and cross-encoder reranker, production chunking, metadata, source registry and section map, the existing v12 cache), builds the four specialists, then for the 62 I-3 questions, the 14 I-4 probes, the 26-question I-5 aggregation set and the 24-question routing and handoff set defined in the script records (1) the Coordinator's decision and the routing refinement (owners before and after, code), (2) the adapter attempt with its routing, handoff, verification and aggregation records, timings, answer, pages and items, (3) the I-5 behaviour of the same question for the before-and-after comparison, (4) the pipeline answer in SHADOW mode (compared with the direct `answer_question` call) and in ACTIVE mode with the diagnostics; the switches are set on the config module in-process, `.env` is never read or written |
| `analyze.py` | the I-5 served set regression (single findings, aggregates, slices), the Category C and D re-evaluation against the I-5 behaviour, every handoff round attempted, the known HR and annual-leave cases, newly served answers, acceptance before and after by group, routing and handoff latency separately from the specialists, SHADOW invariance, Plan F model calls, diagnostics hygiene, the served answers for review |
| `review_accepted.py` | for every answer served by Plan F in the ACTIVE run (`--new-only` for the newly served ones), the Coordinator's clauses and tasks, the routing refinement, the handoff round, the per-clause coverage, the rendered parts, and for each part the exact production chunk and the surrounding printed page text, for the manual correctness review |
| `results_2026-10-30.json` | the recorded run of 2026-10-30 (no model client configured, so the legacy model path returned its no-key reply; Plan F made 0 model calls) |

Run from the repository root with the project interpreter:

    python -X utf8 qa/plan_f_i6/routing_eval.py . <output dir>
    python -X utf8 qa/plan_f_i6/analyze.py <output dir>/i6_results.json
    python -X utf8 qa/plan_f_i6/review_accepted.py . <output dir>/i6_results.json [--new-only]

The routing and handoff set (24 questions, tagged by the shape each was written to
exercise, not by its expected outcome): contact and location asks that name Human
Resources or a research unit, a contact-person ask about a subject, authority asks about
HR and Research policy, a responsibility ask, a committee selection, a location ask that
is a handoff candidate, five cross-domain two-clause questions, two same-domain multipart
questions, a mixed partial question, a responsibility ask owned by Institutional
Navigation, a general institutional ask, a question no specialist owns and a general HR
ask. Every fact asked is one the Faculty Handbook states. Control-flow edges (self loop,
cycle, second hop, invalid or unregistered target, malformed task) are exercised by the
synthetic fixtures in `tests/test_plan_f_routing.py`, not by real questions.

Recorded on 2026-10-30 (CPU, Windows): knowledge base from the v12 cache in 22.5 s
(272 pages, 13,688 chunks); 117 unique questions, 126 runs per mode, 527 s in all. All
35 answers the I-5 ACTIVE run served are served again with byte-identical text and
pages (28 single findings, the 5 aggregates, the 2 provision slices). Twenty answers
newly served, all CORRECT on review, 0 limited, 0 wrong: three single findings and
three aggregates by routing refinement (approval asks owned by Research; a contact
clause owned by Institutional Navigation), nine aggregates by clause scoping of a single
specialist's multipart task, and five questions of the new set the I-5 machinery already
handled. Category C: 3 of 8 unique cases newly resolved, 5 fall back for a named reason
(two partial, one not found, two ambiguous with no specialist run). Category D: the
Research Institute question stops before the handoff because Research did not support
its other clause; two real handoffs executed at depth one (Research to Institutional
Navigation) and fell back on the target's result. Routing codes over 126 runs: unchanged
85, refined by request kind 17, clauses scoped 19, ambiguous 5. Acceptance: corpus 21 to
22 of 62, probes 9 to 10 of 14, aggregation set 6 to 16 of 26, routing set 8 of 24; all
117 unique questions 55 served, 62 fallback. Routing refinement median 0.3 ms when a
decision changed, a pre-dispatch rejection 1.0 ms for the whole attempt, the handoff
round 0.1 ms plus the target's own time; first-round specialists median 617 ms;
specialist calls per attempt 0 for 12, 1 for 101, 2 for 13. SHADOW identical to the
direct legacy call for all 117 unique questions; Plan F 0 model calls; 0 of 252
diagnostic records carry question or answer text. See `docs/STATUS.md`, entry of
2026-10-30 (phase I-6), for the full record.
