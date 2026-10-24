# qa/hr_4_14_a — reconstructed HR QA campaign (Milestone 4, Steps 4.14-A, 4.14, 4.14-N1 and 4.14-W1)

Durable, deterministic QA assets for the Faculty Services & HR specialist's specific-claim support contract (D-097 to D-102). They replace the Step 4.12-R12 / 4.13 audit artefacts, which were not recoverable, with 194 labelled cases built on the fixture pages of `tests/faculty_services_support.py`, a one-command replay runner and a reversible switch-mutation harness. The pytest regression tests that came out of the campaign live in `tests/test_faculty_services_corrections.py`.

## Files

| File | Content |
|---|---|
| `hr_4_14_a_cases.jsonl`, `hr_4_14_a_cases_round2.jsonl`, `hr_4_14_a_cases_round3.jsonl` | the reconstructed cases: round 1 (105, the six failure classes and controls), round 2 (69, reader-limit probes, wrappers, alternatives), round 3 (20, variants sizing the two defects). Fields: `case_id`, `failure_class`, `question`, `expected` (`NOT_SUPPORTED`, `SUPPORTED_OK`, `PARTIAL_PREFERRED`), `evidence_pages`, `evidence_quote`, `rationale`, `origin`, optional `crafted_lines` (extra lines placed on fixture page 107) and `clauses` (focus clauses as the Coordinator would split them) |
| `hr_4_14_a_cases_r11_switchkillers.jsonl` | 35 per-rule cases from the project's own R11 tables, one or more per D-099 switch |
| `hr_4_14_cases_switchkillers_addendum.jsonl` | SK-A1 (Step 4.14): a negative claim stated only under an `if` condition, the killer of `_C12_NEGATIVE_CONDITION`; SK-A2 to SK-A4 (Step 4.14-N1): claims stated only under a condition that restates their entity (HR-414-N1), the killers of `_C12_OWN_CONTENT`; SK-A5 to SK-A7 (Step 4.14-W1): claims stated only under a "when" / "whenever" condition (HR-R2-W1), the killers of `_C12_WHEN_CONDITION` |
| `hr_case_runner.py` | replays a case file under both stand-in rankings ("raw" token overlap, "high" scripted 9.0) and labels every run `as_expected`, `conservative_miss`, `over_support_vs_preference` or `materially_wrong_supported` |
| `mutation_switch_harness.py` | turns each `_C12_*` switch off in-process, replays the `NOT_SUPPORTED` cases (raw ranking), restores the switch, reports killers / survivors and checks deterministic replay |
| `before_4_14/` | results and mutation tables of Step 4.14-A on the unchanged code (HEAD `bb8432d`): the two defects visible as 4 `materially_wrong_supported` runs in round 2 |
| `after_4_14/` | results of the final Step 4.14 code, the before / after comparison (`replay_comparison.json`, `replay_comparison_table.md`) and the mutation tables |
| `after_4_14_n1/` | results of the Step 4.14-N1 code (D-101), its comparison with `after_4_14/` and with `before_4_14/`, and the mutation tables |
| `after_4_14_w1/` | results of the Step 4.14-W1 code (D-102), its comparison with `after_4_14_n1/` and with `before_4_14/`, and the mutation tables |
| `compare_replay.py` | compares two replays run by run (status, outcome, quoted evidence) and checks the expected statuses |

## Commands (from the repository root; the scripts put `.venv/Lib/site-packages`, `tests` and the repo on `sys.path`)

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/hr_4_14_a/hr_case_runner.py --repo . --cases qa/hr_4_14_a/hr_4_14_a_cases_round2.jsonl --out <scratch>/r2.jsonl
```

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/hr_4_14_a/mutation_switch_harness.py --repo . --cases qa/hr_4_14_a/hr_4_14_a_cases_r11_switchkillers.jsonl qa/hr_4_14_a/hr_4_14_cases_switchkillers_addendum.jsonl --out <scratch>/m.json
```

Use the system interpreter (the repository `.venv` is broken) and `-X utf8`. Nothing is written inside the repository; write outputs to a scratch folder. No real model, no model download, no FAISS rebuild, no paid API is involved: the specialist runs on the fake embedder and the stand-in rerankers of the test fixtures.

## Expected counts (Step 4.14 final code, 2026-10-24)

Replay, both rankings (388 runs):

| Round | Cases | materially_wrong_supported | over_support_vs_preference | conservative_miss | as_expected |
|---|---|---|---|---|---|
| 1 | 105 | 0 | 10 | 28 | 172 |
| 2 | 69 | 0 (was 4 before Step 4.14) | 0 | 20 | 118 (was 114) |
| 3 | 20 | 0 | 0 | 6 | 34 |

The 27 conservative misses and 5 defensible over-supports are documented behaviour (`docs/FACULTY_SERVICES_SPECIALIST.md`, "Known conservative misses"); a second run must be byte-identical.

Mutation, R11 set + addendum (36 cases): 13 of 17 switches killed. Survivors and why: `_C12_ENTITY_SUBJECT` and `_C12_PAREN_NUMERAL` have no triggering case in this set and are killed by R2-07a and R2-12j of the reconstructed rounds; `_C12_HEDGE` is equivalent at the status level (with it off the hedged claim is unreadable and the gate fails closed); `_C12_LEADS` only shows at clause level (the task stays partial because the split-off lead is unsearched). Both are asserted by focused tests in `tests/test_faculty_services_corrections.py`. Mutation on the reconstructed rounds (136 cases): `_C12_CLAUSE` and `_C12_ENTITY_SUBJECT` killed by R2-07a, `_C12_GATE` and `_C12_PAREN_NUMERAL` by R2-12j; the other switches survive there because the reconstructed false claims are blocked by several layers at once (defence in depth). `deterministic replay identical: True` in every run.

## Expected counts after Step 4.14-N1 (D-101, 2026-10-24)

Replay: identical to the Step 4.14 table above (0 changed runs against `after_4_14/`, byte-identical to the Step 4.14-R prototype
replay); HR-414-N1 is not among the 194 cases and is covered by `tests/test_faculty_services_corrections.py` and SK-A2 to SK-A4.

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/hr_4_14_a/compare_replay.py --before qa/hr_4_14_a/after_4_14 --before-name "results_round{n}_after.jsonl" --after <scratch dir with results_round{n}_after.jsonl> --out <scratch>/cmp.md --json <scratch>/cmp.json
```

Mutation, R11 set + addendum (39 cases): 14 of 18 switches killed; `_C12_OWN_CONTENT` by SK-A2, SK-A3 and SK-A4; survivors
`_C12_ENTITY_SUBJECT`, `_C12_PAREN_NUMERAL` (killed on the reconstructed rounds), `_C12_HEDGE`, `_C12_LEADS` (status-equivalent).
Reconstructed rounds (136 cases): 4 of 18 killed, as before. 16 distinct switches killed over both sets.

## Expected counts after Step 4.14-W1 (D-102, 2026-10-24)

Replay: identical to Step 4.14-N1 (0 changed runs against `after_4_14_n1/`); HR-R2-W1 is not among the 194 cases and is covered by
`tests/test_faculty_services_corrections.py` and SK-A5 to SK-A7. Mutation, R11 set + addendum (42 cases): 15 of 19 switches killed;
`_C12_WHEN_CONDITION` by SK-A5, SK-A6 and SK-A7. Reconstructed rounds (136 cases): 4 of 19 killed, as before. 17 distinct switches killed
over both sets; `_C12_HEDGE` and `_C12_LEADS` never (status-equivalent / clause-level, documented).

## Records

Decision D-100 (`docs/DECISIONS.md`), the 2026-10-24 Step 4.14 entry of `docs/STATUS.md`, and the Step 4.14 reports in the project handover folder (`SENIOR_PROJECT_HANDOVER/HR_STEP_4_14/`).
