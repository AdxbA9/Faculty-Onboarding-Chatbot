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
| `research` | Research & Innovation | funding, ethics, IP, institutes, research policy |
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

## Future handoff model

1. The Coordinator selects up to `PLAN_F_MAX_SPECIALISTS` specialists and up to `PLAN_F_MAX_SUBTASKS` subtasks.
2. The orchestrator runs the first-round specialists (independent ones in parallel) and collects their findings.
3. Handoff requests are validated: the requested specialist must be registered, must not already have run for that task, and the agent-call budget must remain.
4. At most one further round runs. Requests produced by that round are recorded in the trace and marked as missing coverage, never executed.
5. Findings go to deterministic rendering (one supported specialist, simple question) or to Synthesis aggregation, then to the Verifier.

## Compatibility guarantees

- `agents/types.py` (`RouteDecision`, `SubQuestion`, `EvidenceResult`, `SynthesisResult`, `VerifyResult`, `TraceEntry`) is unchanged.
- `QAResult` is unchanged; Plan F fields will be added, with defaults, in the phase that first produces them.
- `answer_question()`, the UI and the evaluation harness are unaffected.
- The `PLAN_F_*` settings in `config.py` are declared but read by nothing; `MAX_LLM_CALLS` and `MAX_VERIFY_RETRIES` keep their Milestone 1 values.
