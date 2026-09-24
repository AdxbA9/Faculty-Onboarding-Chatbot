# Development Status

Handoff log for the Part 2 implementation. Newest entry first. Update this
file at the end of every meaningful work session, before the final push, so
the next developer can start from the branch and commit named here.

Rules: one entry per session; commit hashes are the short hashes shown by
`git log --oneline`; wording stays factual and neutral.

---

## 2026-09-24 - Plan F core foundation: contracts, specialist framework, registry

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/plan-f-core` (created from `feature/verifier-tests`) |
| Starting commit | `78cd26d` |
| Ending state | committed on `feature/plan-f-core`; see Git history for the commit |

### Completed

- `handbook_bot/agents/specialists/contracts.py`: `SpecialistId` and
  `FindingStatus` enums; `SpecialistTask`, `Finding`, `SpecialistFindings`,
  `HandoffRequest`, `CoordinatorDecision` dataclasses with validation and
  `to_dict()`. `claim` and `evidence_quote` are separate fields by design.
- `handbook_bot/agents/specialists/base.py`: `Specialist` abstract base
  (`run(task) -> SpecialistFindings`) and `check_findings()`.
- `handbook_bot/agents/specialists/registry.py`: `SpecialistRegistry`;
  unknown ids and duplicates rejected explicitly; no global instance.
- `handbook_bot/config.py`: inactive `PLAN_F_ENABLED` and `PLAN_F_MAX_*`
  budgets (3 specialists, 3 subtasks, handoff depth 1, 1 retry, 5 LLM calls,
  4 agent calls). `MAX_LLM_CALLS` and `MAX_VERIFY_RETRIES` unchanged.
- Tests: `tests/test_planf_contracts.py`, `tests/test_planf_registry.py`,
  `tests/test_planf_imports.py` (fresh-interpreter imports, static import
  guard, budgets, `QAResult` unchanged).
- Docs: `docs/AGENT_CONTRACTS.md`, `docs/DECISIONS.md` (D-PLANF-001..014).

### Not done on purpose (later phases)

No Coordinator logic, no specialist implementation, no retrieval, Synthesis
or Verifier change, no `QAResult` change, no UI or evaluation change. Plan F
is not active; the Milestone 1 pipeline is unchanged.

### Tests performed

See the phase report: targeted Plan F tests, full `pytest -q`, `pip check`
and fresh-interpreter imports, all with `.\.venv\Scripts\python.exe` on
Windows.

### Next task

Phase 2, knowledge layer: source registry (`knowledge/sources.yaml` or JSON),
section map from the handbook table of contents, chunk metadata (doc id,
section, domains, systems, level, scope), cache version bump, scope filter
and boosts in retrieval. Blackboard and Banner guides must be obtained and
reviewed by the team before the Teaching Agent phase.

---

## 2026-09-21 - Verifier agent, committed tests, evaluation metadata

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/verifier-tests` (created from `feature/router-orchestrator`) |
| Starting commit | `6b92261` |
| Ending commit | `dd3103c` (code); this file is committed on top of it |

### Completed

- `handbook_bot/agents/verifier.py`: deterministic Verifier agent. Checks
  every sentence of an answer against every retrieved chunk (the Part 1
  check used the top chunk only). Numbers, phone numbers, e-mail addresses
  and date words must appear in the supporting chunk. Cited pages come only
  from supporting chunks; unsupported claimed pages are dropped; no page is
  ever invented. Rejections carry targeted retry feedback. No LLM call.
- `tests/` (115 tests): router matrix and `number` regression, ambiguity
  and LLM-fallback failure modes, verifier support and citation rules,
  orchestrator paths (greeting, extractors, synthesis, retry, one-retry cap,
  LLM budget, API error, missing key, verifier fallback), compatibility of
  `QAResult`, `answer_question()`, the UI fields and the Pages parser.
  Runs offline: fake models and a scripted LLM client.
- `eval/run_eval.py` records `llm_calls`, `retried`, `sub_questions`,
  `agent_trace` and the router, budget and planner settings.
  `eval/score.py` prints an LLM-call and verifier-decision summary for runs
  that carry these fields; Part 1 run files still score unchanged.
- `pytest.ini`, `pytest` in `requirements.txt`, `.pytest_cache/` ignored.

### Tests performed

```
python -m compileall handbook_bot ui eval app.py tests   -> OK
pytest -q                                                 -> 115 passed
```

Both in the system interpreter and in a fresh virtual environment built
from `requirements.txt`.

Verifier calibration against the golden set (no API): of the 33 cases with
an evidence quote, 30 reference answers were accepted against their own
evidence placed after four distractor chunks, with the expected page cited
in all 30; all 33 were rejected when only foreign evidence was supplied.
The three rejections are reference answers that contain facts absent from
their quote.

### Known issues

- Real-model evaluation is not yet run: the development sandbox blocks
  huggingface.co and api.groq.com (HTTP 403 from the network policy), so
  `python eval/run_eval.py --limit 3` fails at model download. Run it on a
  machine with network access and a `GROQ_API_KEY`, then the full golden
  set, then `python eval/score.py`. No accuracy figure for the Milestone 1
  pipeline exists until then.
- Pre-existing tool issues, not changed here: the count extractor takes the
  first regex match in whichever chunks were retrieved; the contact
  extractor accepts loosely spaced digit runs as phone numbers; the date
  extractor still calls the Part 1 classifier internally. Scope of
  `feature/tool-fixes`.
- `docs/IMPLEMENTATION_SPEC.md` and `docs/WORK_SPLIT.md` are cited by the
  code but are not in the repository yet.
- Milestone 2 (Planner, Evidence agent) is not started; `PLANNER_ENABLED`
  stays off.

### Files changed

```
handbook_bot/agents/verifier.py   new
tests/conftest.py                 new
tests/test_router.py              new
tests/test_verifier.py            new
tests/test_orchestrator.py        new
tests/test_compat.py              new
pytest.ini                        new
eval/run_eval.py                  metadata fields
eval/score.py                     cost summary section
requirements.txt                  pytest
.gitignore                        .pytest_cache/
docs/STATUS.md                    new
```

### Do not redo

Router scoring rules, the LLM-call budget and retry clamp in the
orchestrator, the verifier auto-discovery hook, the Pages parser.

### Next task

1. On a machine with network access: `pip install -r requirements.txt`,
   set `GROQ_API_KEY` in `.env`, run `python eval/run_eval.py --limit 3`,
   then the full set, then `python eval/score.py`. Commit the run file
   under `eval/runs/` and record the numbers here.
2. Open pull requests towards an integration branch: `part2-eval`, then
   `feature/router-orchestrator`, then `feature/verifier-tests`.
3. Commit `docs/IMPLEMENTATION_SPEC.md` and `docs/WORK_SPLIT.md`.

### Blockers

Network access to the model host and the LLM API from the development
environment.

---

## 2026-09-21 - Router, Synthesis, Orchestrator (Milestone 1 foundation)

| Field | Value |
|---|---|
| Developer | Mohammad Abdoljalil |
| Branch | `feature/router-orchestrator` (created from `part2-eval`) |
| Starting commit | `3ab4d58` |
| Ending commit | `6b92261` |

### Completed

- `handbook_bot/agents/types.py`: shared interface contract
  (`RouteDecision`, `SubQuestion`, `EvidenceResult`, `SynthesisResult`,
  `VerifyResult`, `TraceEntry`). Route name `policy_yesno` kept.
- `handbook_bot/agents/router.py`: hybrid Router; all intents scored,
  ambiguity margin, one bounded LLM arbitration call only on a tie, never
  raises. Fixes the Part 1 `number` mis-route (count vs contact).
- `handbook_bot/agents/synthesis.py`: typed wrapper over the Part 1 prompt
  and Groq call; prompt rule 1 no longer caps answers at two sentences.
- `handbook_bot/orchestrator.py`: deterministic control flow, LLM-call
  budget, retry clamped to one, agent trace, verifier auto-discovery with
  fallback to the Part 1 check.
- `handbook_bot/config.py`: agent and orchestration flags.
- `handbook_bot/qa.py`: `answer_question()` delegates to the orchestrator;
  `QAResult` gains `agent_trace`, `llm_calls`, `retried`, `sub_questions`.

### Tests performed

Local test scripts and review rounds; not committed to the repository.

### Known issues at handoff

Verifier and committed tests outstanding (done in the entry above); final
parity run on the real index not recorded.

### Files changed

`handbook_bot/agents/__init__.py`, `agents/types.py`, `agents/router.py`,
`agents/synthesis.py`, `handbook_bot/orchestrator.py`,
`handbook_bot/config.py`, `handbook_bot/qa.py`.
