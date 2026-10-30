# Plan F Agent Contracts

Module: `handbook_bot/agents/specialists/contracts.py`
Base class and registry: `handbook_bot/agents/specialists/base.py`, `registry.py`

THIS PHASE DOES NOT ACTIVATE PLAN F. The contracts, base class and registry
exist so the Coordinator, the specialists and the findings-aware Verifier
can be built against stable shapes. Nothing in the current chatbot runtime
imports them; the Milestone 1 pipeline (Router, evidence, extractors,
Synthesis, Verifier, orchestrator) behaves exactly as before.

## Overview

```
Coordinator  --CoordinatorDecision-->  orchestrator
orchestrator --SpecialistTask------->  specialist (via SpecialistRegistry)
specialist   --SpecialistFindings--->  orchestrator   (may contain HandoffRequest)
orchestrator --findings + answer---->  Verifier
```

All contracts are plain dataclasses (the project's existing convention in
`agents/types.py`). Each validates itself when constructed and raises
`ValueError` on a malformed value, and each has `to_dict()` for traces and
evaluation run files. The module imports only the standard library.

## Canonical specialist ids

`SpecialistId` (a `str` enum) is the only place the four ids are spelled:

| Id | Display name | Owns |
|---|---|---|
| `teaching` | Teaching & Learning | teaching policy, Blackboard, Banner, MyUOS access |
| `research` | Research & Innovation | funding and grants, research ethics (IRB, IACUC), IP, patents, commercialisation, institutes, groups and boards, scientific publishing, conferences, consultancy, research policy; shared: the thesis supervision cap, research release, research content of promotion and evaluation (see `docs/RESEARCH_SPECIALIST.md`) |
| `faculty_services` | Faculty Services & HR | appointment, contracts, benefits, leave, services |
| `institutional` | Institutional Navigation | who, where, which office, approval chains, contacts |

`SpecialistId.parse()` accepts a member or its string value and rejects
anything else. Orientation is a Coordinator journey mode and Blackboard and
Banner are knowledge sources, so none of them is an id.
`ALL_SPECIALIST_IDS` lists the values as plain strings.

## Statuses

`FindingStatus` (a `str` enum), the outcome of one specialist run:

| Status | Meaning | Findings |
|---|---|---|
| `supported` | grounded and complete: every part of the task is answered with evidence; `missing` must be empty | at least one required |
| `partial` | at least one grounded finding, and `missing` names the parts no finding covers | at least one required |
| `not_found` | in scope, but the approved sources do not cover it | normally none |
| `out_of_scope` | another specialist owns the task; usually with a handoff request | normally none |
| `error` | the specialist could not run (exception, timeout, invalid output) | none |

## SpecialistTask

One unit of work for one specialist.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `task_id` | str | required | unique within a question |
| `question` | str | required | the (sub)question in the user's terms |
| `specialist_id` | SpecialistId or None | None | assignee; set on every Coordinator subtask |
| `domain`, `intent`, `level`, `system` | str or None | None | hints: e.g. `teaching`, `procedure`, `university`, `blackboard` |
| `source_scope` | list[str] | [] | source ids the specialist may retrieve from; empty means its default scope |
| `context` | dict | {} | prior findings or conversation context the orchestrator chooses to pass |
| `requested_by` | str | `"coordinator"` | `"coordinator"`, or the id of the specialist whose handoff produced the task |
| `metadata` | dict | {} | free-form |

## Finding

One grounded statement. `claim` and `evidence_quote` are separate fields and
must stay separate: the Verifier checks that the quote exists in the cited
source and that the claim is supported by the quote. A finding whose claim
merely restates its quote is fine; a finding that merges the two is not.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `finding_id` | str | required | unique within a `SpecialistFindings` |
| `claim` | str | required (non-empty) | what the specialist asserts |
| `evidence_quote` | str | required (non-empty) | the source passage relied on; a finding without one is not grounded and is rejected |
| `source_id` | str | required | registry id of the source |
| `source_title` | str | "" | human-readable title for citations |
| `page` | int or None | None | 1-based page inside the source |
| `confidence` | float 0..1 | 1.0 | the specialist's own confidence |
| `metadata` | dict | {} | e.g. section number, chunk id |

Since Milestone 4 Step 4.5A-D a finding for a recognised specific ask also
carries `metadata["relation"]`: `strong` when deterministic rules prove that
its statement states the asked relation, `ambiguous` otherwise (see
"Relation confidence and the future Verifier boundary").

## SpecialistFindings

The structured result of one specialist run. Free-form prose is never a
result; `summary` is a short label for the trace, not the answer.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `specialist_id` | SpecialistId | required | the specialist that produced it |
| `task_id` | str | required | the task it answers |
| `status` | FindingStatus | required | see statuses |
| `findings` | list[Finding] | [] | unique ids; `supported` and `partial` need at least one |
| `summary` | str | "" | one-line description for the trace |
| `missing` | list[str] | [] | parts of the task no finding covers; must be empty for `supported` |
| `handoff_requests` | list[HandoffRequest] | [] | all must originate from this specialist |
| `confidence` | float 0..1 | 0.0 | overall confidence |
| `limitations` | list[str] | [] | caveats the final answer must carry |
| `llm_used` | bool | False | whether the run issued an LLM call |
| `ms` | float | 0.0 | wall time |
| `metadata` | dict | {} | free-form (error text, source scope used, ...) |

`requested_agents` is a read-only property listing the ids of the requested
specialists, derived from `handoff_requests`.

## HandoffRequest

A specialist asking the orchestrator to run another specialist. Specialists
never execute each other. The orchestrator validates every request against
`PLAN_F_MAX_HANDOFF_DEPTH` (1: one handoff round, never more),
`PLAN_F_MAX_AGENT_CALLS` and the registry before acting.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `from_specialist` | SpecialistId | required | the requester |
| `requested_specialist` | SpecialistId | required | the requested specialist; never the requester itself |
| `reason` | str | required (non-empty) | why the handoff is needed |
| `task` | SpecialistTask or None | None | a prepared task for the requested specialist |
| `priority` | int | 0 | higher runs first when requests compete for the budget |
| `metadata` | dict | {} | free-form |

## CoordinatorDecision

The Coordinator's output. Contract only; no Coordinator logic exists yet.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `domains` | list[str] | [] | domains detected, best first |
| `intents` | list[str] | [] | technical intents (the Router's routes) |
| `level` | str or None | None | organizational level |
| `systems` | list[str] | [] | systems named or implied |
| `selected_specialists` | list[SpecialistId] | [] | first-round specialists, unique, best first |
| `subtasks` | list[SpecialistTask] | [] | unique ids; each assigned to a selected specialist |
| `requires_synthesis` | bool | False | False when one supported specialist can be rendered deterministically |
| `confidence` | float 0..1 | 0.0 | |
| `reason` | str | "" | why these specialists |
| `used_llm` | bool | False | whether the Coordinator used its structured LLM call |
| `metadata` | dict | {} | scores and other trace detail |

## Specialist base class and registry

`Specialist` (`base.py`) is an abstract base class with one method,
`run(task: SpecialistTask) -> SpecialistFindings`, and a class attribute
`specialist_id` validated when the subclass is created. `check_findings()`
is the boundary check the orchestrator will apply: the result must be a
`SpecialistFindings` for the same task under the specialist's own id.

`SpecialistRegistry` (`registry.py`) maps ids to implementations. It
rejects non-`Specialist` objects, rejects duplicate registration unless
`replace=True` is passed explicitly, and raises `UnknownSpecialistError`
for unknown or unregistered ids. It never substitutes another specialist.
There is no module-level instance; the orchestrator of a later phase will
create one at start-up.

`base.redact_error(exc)` is the one error-redaction function: the dispatcher
uses it for a raising specialist and every specialist uses it for its own
`error` results (key-shaped secrets and Windows and Unix paths replaced,
including directory names with spaces and UNC shares since Step 4.5A).

The dispatcher executes only tasks issued by the Faculty Onboarding
Coordinator (`requested_by == "coordinator"`); a task produced by a
specialist's handoff and placed in a decision is kept `pending`, so no second
round runs before handoff depth accounting exists (Step 4.5A). Since Step
4.11 the Faculty Services & HR Specialist is executable beside Teaching and
Research (`EXECUTABLE_SPECIALISTS`); since Step 4.16-B6 (D-105) Institutional
Navigation is executable too, so all four specialists run when registered.
Handoff requests remain data and are never executed by the dispatcher.

Shared evidence helpers (Milestone 4). Domain-neutral mechanisms extracted
unchanged from the Teaching & Learning Specialist so that no specialist
imports another: `scoping.py` (`compile_scope` with optional `extra_pages`,
`heading_marker`, `SectionGuard`), `evidence.py` (quote units, lettered
items, quote quality, `select_quote` with an opt-in lead-in join,
`unit_coverage`, `detail_covered`) and `details.py` (value-shaped numbers,
money, timing, frequency, limits, obligation, configurable authority and
responsibility checks, `requested_details`). Ownership cues, vocabularies,
section lists, detail-family tuples and the run logic stay in each
specialist's own module; there is no specialist base class beyond
`Specialist`.

## Future handoff model

1. The Coordinator selects up to `PLAN_F_MAX_SPECIALISTS` specialists and up to `PLAN_F_MAX_SUBTASKS` subtasks.
2. The orchestrator runs the first-round specialists (independent ones in parallel) and collects their findings.
3. Handoff requests are validated: the requested specialist must be registered, must not already have run for that task, and the agent-call budget must remain.
4. At most one further round runs. Requests produced by that round are recorded in the trace and marked as missing coverage, never executed.
5. Findings go to deterministic rendering (one supported specialist, simple question) or to Synthesis aggregation, then to the Verifier.

## Relation confidence and the future Verifier boundary (Milestone 5, design only)

Semantic relation verification belongs to the Milestone 5 Verifier. Milestone 4 specialists fail conservatively when deterministic relation evidence is ambiguous. Nothing in this section is
implemented in Milestone 4; it fixes what the specialists report so that the
Verifier can be built against it.

A specialist reports, per clause, its ask shape (`general`,
`specific_recognised`, `specific_unknown`), its requested details and which
of them are uncovered, and its relation (`strong`, `ambiguous`, or null for
general, unknown and process asks); per finding, `metadata["relation"]`. A
clause whose relation is `ambiguous` is at most `partial`, and its `missing`
entry names the relation that was not established.

Verifier input (per claim):

| Input | From |
|---|---|
| question and clause | the task and the clause report |
| requested specialist | `SpecialistFindings.specialist_id`, the task's `specialist_id` |
| specialist findings | `SpecialistFindings` (status, findings, missing, limitations) |
| claim | `Finding.claim` |
| evidence quote | `Finding.evidence_quote` |
| source and page metadata | `source_id`, `source_title`, `page`, section and chunk metadata |
| requested details and coverage | the clause's `requested_details`, `uncovered_details`, each finding's `details_covered` |
| relation confidence | the clause `relation` and the finding's `metadata["relation"]` |

Possible outcomes:

| Outcome | Meaning |
|---|---|
| verified | the quote states the claimed relation; the claim may be used as stated |
| downgraded | the quote is related but does not state the relation; the claim is used as partial evidence with the reason |
| rejected | the quote does not support the claim; the claim is not used |
| retry requested | a narrower task to the same or another specialist, within the call budget and the handoff rules |

Only the Verifier may treat an `ambiguous` relation as established, after its
own check; a specialist never upgrades one. Specialists add no LLM call for
this.

## Specialist design rules (Faculty Services & HR, Institutional Navigation)

The next specialists are built from the start on the Research contract
(D-074, D-076, D-077):

1. Every substantive clause is `general`, `specific_recognised` or
   `specific_unknown` from its question structure.
2. An unknown specific ask receives a requested detail no evidence can
   cover: it is never `supported`.
3. A recognised specific ask is `supported` only with a STRONG relation
   proved in the answering statement (roles in their asked position,
   actions and their objects, purpose, receiver, required thing, permission
   scope, period, yes/no proposition with its negation).
4. Semantic uncertainty is `partial`, with the evidence kept and the reason
   in `missing`; the relation is reported per clause and per finding.
5. Ownership cues live in one standard-library lexicon module shared with
   the Coordinator, as `specialists/domain_cues.py` does for Research.
6. The retrieval route is resolved per focus clause with the shared
   `specialists/retrieval_intent.py` helper (D-078): the task intent stays
   one per task and never makes a substantive clause retrieve as a contact
   lookup because another clause is a contact ask.
7. Topic relevance alone never supports a specific ask (D-079). The shared
   ask gate (`specialists/grounding.py`, injected with the domain's
   vocabulary) is available to any specialist that does not carry the
   Research relation code: Teaching uses it; the next specialists inherit
   the Research contract as planned and may use the gate for the same
   guarantee.
8. Position is part of the relation (D-080): the gate binds the asked
   actor, action, object and direction in one evidence clause with an
   injected role lexicon; the right words in the wrong roles never support.
9. Applicability is part of the relation (D-081): the person asking is a
   role like any other, a subject-less action never falls to a role named
   only as recipient or patient, a passive agent keeps its whole phrase,
   receiving binds its source, and a frequency unit is satisfied only by
   the same unit.
10. An unknown subject is still a subject (D-082): a subject outside the
    role lexicon is bound as a phrase, never dropped to the word bag; a
    table row settles no ask that names an actor; irregular participles
    are read in passive asks; actor evidence is local to the sentence of
    the action.
11. A number is one fact (D-083): value, object and period are bound
    within one clause of the evidence, an action's period sits in the
    clause of that action by the asked actor, and facts of one unit never
    combine.

## Compatibility guarantees

- `agents/types.py` (`RouteDecision`, `SubQuestion`, `EvidenceResult`, `SynthesisResult`, `VerifyResult`, `TraceEntry`) is unchanged.
- `QAResult` is unchanged; Plan F fields will be added, with defaults, in the phase that first produces them.
- `answer_question()`, the UI and the evaluation harness are unaffected.
- The `PLAN_F_*` settings in `config.py` are declared but read by nothing; `MAX_LLM_CALLS` and `MAX_VERIFY_RETRIES` keep their Milestone 1 values.
