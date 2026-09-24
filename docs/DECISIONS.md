# Architecture Decisions

Decisions that shape Plan F, recorded so they are not re-argued by accident.
Each entry: what was decided, why, and when to revisit. THIS PHASE DOES NOT
ACTIVATE PLAN F: entries D-PLANF-001 to D-PLANF-012 describe the target
design; only the contracts, base class, registry and inactive budgets exist
in code today.

## D-PLANF-001 Plan F is a findings-based two-layer architecture
- Date: 2026-09-24
- Decision: a Coordinator selects domain specialists; each specialist returns structured findings (claim, evidence quote, source, page, confidence) rather than a final answer; the orchestrator renders or aggregates the findings and the Verifier checks them.
- Why: a specialist that writes the final answer cannot be verified against its evidence, and its prose cannot be combined with another specialist's without losing source boundaries. Findings keep every statement traceable to a source.
- Revisit: if the Teaching Agent gate shows findings add latency without improving verifiability.

## D-PLANF-002 The Coordinator owns routing and decomposition
- Decision: one component decides domains, level, systems, selected specialists and subtasks. It reuses the existing Router for technical intent and adds domain scoring; one structured LLM call is allowed only for ambiguous, multipart or cross-domain questions.
- Why: two competing classifiers would disagree; a deterministic-first Coordinator keeps simple questions at zero LLM calls, as the Router does today.
- Revisit: after domain-accuracy measurement on the labelled set.

## D-PLANF-003 Four specialists, no more
- Decision: `teaching`, `research`, `faculty_services`, `institutional`, centralised in `SpecialistId`.
- Why: they match the handbook's chapter structure and the sources the team can obtain. Every candidate for a fifth (Orientation, Blackboard, Banner, College, Department) either owns no unique decision or is a knowledge source.
- Revisit: only with evidence that a candidate owns decisions none of the four can make.

## D-PLANF-004 Specialists never call each other; the orchestrator executes handoffs
- Decision: a specialist that needs another domain returns a `HandoffRequest`; the deterministic orchestrator validates and executes it.
- Why: direct calls make budgets unenforceable, hide loops, and make traces incomplete.
- Revisit: never; this is a safety property.

## D-PLANF-005 Maximum handoff depth is one round
- Decision: `PLAN_F_MAX_HANDOFF_DEPTH = 1`. Requests produced by the handoff round are recorded, not executed.
- Why: bounded latency and cost; every collaboration case in the candidate set is satisfied by one round.
- Revisit: if reviewed cases show a required second round.

## D-PLANF-006 The orchestrator stays deterministic Python
- Decision: no LLM decides control flow, parallelism, budgets or retries.
- Why: testable, reproducible, and the same reason Milestone 1 rejected an LLM manager.
- Revisit: never.

## D-PLANF-007 One shared, scope-aware evidence retrieval
- Decision: one FAISS index over all approved sources; a hard source-scope filter per specialist; soft system and domain boosts.
- Why: four indexes would quadruple build time and cache size for no retrieval benefit; scope is a filter, not a separate store.
- Revisit: only if benchmarking shows scoped retrieval quality suffers in a shared index.

## D-PLANF-008 Plan F budgets are declared now and inactive
- Decision: `PLAN_F_ENABLED = False` and `PLAN_F_MAX_*` constants exist in `config.py`; the Milestone 1 pipeline keeps `MAX_LLM_CALLS = 2` and `MAX_VERIFY_RETRIES = 1`.
- Why: raising the live budget before the Coordinator exists would change Milestone 1 behaviour and its measured cost for no benefit. Namespacing avoids two meanings of one name.
- Revisit: when the Plan F orchestration path is wired in, `PLAN_F_ENABLED` becomes the switch.

## D-PLANF-009 The Verifier stays independent and gains a findings mode later
- Decision: the existing sentence-level Verifier remains; a later phase adds quote-in-source, claim-supported-by-quote, subtask coverage, citation origin and conflict checks. It trusts no specialist.
- Why: the Milestone 1 rules already prevent invented citations; findings mode extends them without replacing them.
- Revisit: at the Teaching Agent gate.

## D-PLANF-010 Synthesis is used conditionally
- Decision: one supported specialist on a simple question is rendered deterministically without an LLM; several specialists, multipart or partial results go through Synthesis in an aggregation mode that receives structured findings only.
- Why: the common case stays at one LLM call; aggregation is needed only when findings must be combined.
- Revisit: if deterministic rendering reads poorly in acceptance testing.

## D-PLANF-011 Blackboard and Banner are systems and sources, not agents
- Decision: the Teaching & Learning specialist owns them; their guides are ingested as sources with a `system` tag.
- Why: they own no decision; a "Blackboard agent" would be retrieval with a label.
- Revisit: never.

## D-PLANF-012 Orientation is a Coordinator journey mode
- Decision: an orientation question makes the Coordinator select specialists in stage order and Synthesis render an ordered checklist. No Orientation specialist.
- Why: orientation is a sequence over the other domains, not a domain.
- Revisit: only with evidence that orientation owns decisions the Coordinator cannot express.

## D-PLANF-013 Contracts are plain dataclasses with self-validation
- Decision: `contracts.py` uses `@dataclass` with `__post_init__` validation and `to_dict()`, and `str` enums for ids and statuses; no Pydantic, no new dependency.
- Why: matches `agents/types.py`; Pydantic is only a transitive dependency of the Groq SDK and the project does not use it.
- Revisit: if a later phase needs schema export for the Coordinator's structured LLM call.

## D-PLANF-014 QAResult is not extended in the foundation phase
- Decision: no Plan F fields are added to `QAResult` until a runtime component produces them.
- Why: fields with no producer cannot be tested for meaning, and every addition touches `to_dict()`, the UI and the run files.
- Revisit: in the first phase that returns a `CoordinatorDecision` or `SpecialistFindings` through `answer_question()`.
