# qa/institutional_4_16 — Institutional Navigation evaluations (Milestone 4, Steps 4.16-B2 to 4.16-C1)

Deterministic QA assets of the Institutional Navigation specialist. Phase B2: telephone and fax numbers from the
handbook contact directory (section 1.16, pages 52 to 55), bound to one complete directory line (contact-accuracy
cases, tests in `tests/test_institutional_directory.py`). Phase B3: office responsibilities, organisational structure and
documented library locations, bound to handbook statements (evidence-binding cases, tests in
`tests/test_institutional_offices.py`). The shared fixtures (real handbook pages, the printed-directory oracle, both case
evaluators) are in `tests/institutional_support.py`. Phase B4 (ownership checks and handoffs) is tested in
`tests/test_institutional_integration.py` and `tests/test_institutional_alignment.py`; here it changes the expected status
of the cases whose subject another specialist owns to `out_of_scope` (`expected_handoffs` names the specialists the case
must be handed to; absent means none), and both evaluators check every handoff (`institutional_support.handoff_problems`).
Phase B5 (acceptance) adds an independent evaluation: `coverage_cases.jsonl` and `coverage_runner.py`, tested in
`tests/test_institutional_acceptance.py`.

## Files

| File | Content |
|---|---|
| `contact_cases.jsonl` | 113 labelled cases. Fields: `case_id`, `category`, `question`, `expected_status` (`supported`, `partial`, `not_found`), `expected_values` (`{office_id: {field: number}}`, the only numbers the specialist may give), `rationale`, optional `clauses` (focus clauses as the Coordinator would split them), `expected_entry` (`unbound_row`, `labelled_row`), `expected_limitation` (a substring) and `replace_lines` (`{chunk_id: text}`, directory lines replaced in a copy of the fixture corpus, for the negative controls) |
| `build_contact_cases.py` | writes `contact_cases.jsonl`. The expected numbers come from the printed directory rebuilt from the PDF's geometry (`institutional_support.printed_directory`: the ruled cells of the office column and the words inside the telephone and fax columns), never from the extracted lines the specialist reads; the lists of clean, one-number, fragmented, split and absent entries are the reviewed decision |
| `contact_accuracy_runner.py` | replays every case under both stand-in rankings ("raw" token overlap, "high" scripted 9.0), judges each run (`blocker`, `mismatch`, `as_expected`), checks determinism across runs, instances and rankings; `--mutate` switches each `_B2_*` protection off in turn, replays, restores and reports the killing cases |
| `evidence_cases.jsonl` | phase B3: 345 labelled cases in 15 categories (A correct responsibilities, B wrong-office prevention, C similar names, D structure, E locations, F missing information, G procedures, H other specialists' subjects, I B2 regression, M mixed asks, R / T / V / U the confirmed defects of the second, third, fourth and fifth adversarial reviews, S synthetic negative controls). Fields: `case_id`, `category`, `question`, `expected_status` (list), `allowed_offices` (lexicon ids, "university", or null for a unit outside the lexicon), `required_offices`, `required_quotes`, `forbidden_quotes`, `evidence` (page and printed wording), `kills` (the B3 switches the case must catch), optional `expected_locations` ([label, code, place] as printed), `expected_count`, `expected_contacts`, `clauses`, `append_lines` (`{chunk_id: text}` appended to a copy of the corpus) |
| `build_evidence_cases.py` | writes `evidence_cases.jsonl` from the reviewed case table (`OVERRIDES` records the expectations the fifth review changed to fail closed) |
| `coverage_cases.jsonl` | Step 4.16-B5: 62 faculty-style questions written by blind authors from the handbook pages alone (no access to the specialist, its tests or its cases); each with its expected owner(s), whether the handbook answers it, the expected office, value and pages, and a verbatim evidence quote checked on its page before any run. Not generated from the specialist's output |
| `coverage_runner.py` | runs every coverage question through the unchanged Coordinator and the dispatcher with all four specialists (system path) and through Institutional Navigation alone (direct path); judges each result with fixed rules (correct, correct_partial, incomplete, wrong, unexpected_answer, abstained, not_selected); reports answer correctness on answered questions, unsafe supported answers, coverage of source-answerable questions, abstention, Coordinator routing and citation accuracy |
| `coverage_baseline_b5.jsonl` | Step 4.16-B6: a copy of the Step 4.16-B5 `coverage_runner.py` output (SHA-256 `bc6407a4...`), the per-question baseline that `tests/test_four_specialist_dispatch.py` reproduces through the default dispatcher; rerun the runner with `--default-dispatch` to compare |
| `holdout_cases.jsonl` | Step 4.16-C1: 36 natural faculty questions (12 contacts, 12 responsibilities, 12 structure and locations; 24 answerable, 14 safety probes) written by a blind author from the handbook pages alone, with no access to the specialist, its tests, its cases or the benchmark; same fields as `coverage_cases.jsonl` plus `area`, `safety_probe` and `notes`. Frozen (SHA-256 `f6367ccf...`) before any C1 code change and before it was run against the corrected specialist; never edited after |
| `coverage_baseline_c1.jsonl` | Step 4.16-C1: the `coverage_runner.py --default-dispatch` output on the unchanged 62-question benchmark after the natural-phrasing correction (final code; SHA-256 `5426a56d...`); `tests/test_four_specialist_dispatch.py` reproduces it and checks that it differs from `coverage_baseline_b5.jsonl` only by newly correct answers |
| `evidence_binding_runner.py` | replays every B3 case under both rankings, judges each run with `institutional_support.evaluate_evidence_case`, checks determinism; `--mutate` switches each `_B3_*` protection off in turn and reports the cases that catch it |

## Outcome labels

| Label | Meaning |
|---|---|
| `blocker` | contact cases: a number for an office or field the case does not expect, a number that differs from the printed cell, a number outside a bound finding, a quote not verbatim in its line, or numbers without the as-printed limitation. Evidence cases: a quote not verbatim in its passage, a statement quoted from a directory page, a finding bound to an office the case does not allow, a bound quote that does not print its office, a forbidden quote, a supported answer the case does not allow, a location entry the case does not list, any finding for a procedure or another specialist's subject, a contact value the case does not expect, a supported count with another number: a wrong-office, wrong-location, wrong-number or invented answer. Both (since Step 4.16-B4): a handoff to a specialist the case does not expect, a handoff task that does not carry exactly one of the case's own clauses and the full question, the same clause handed twice to one specialist, an `out_of_scope` result with a finding |
| `mismatch` | a status, expected value, entry kind or limitation not met, with no blocker |
| `as_expected` | everything else |

## Commands (from the repository root; the scripts put `.venv/Lib/site-packages`, `tests` and the repo on `sys.path`)

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/institutional_4_16/contact_accuracy_runner.py --repo . --out <scratch dir> --mutate
```

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/institutional_4_16/build_contact_cases.py --repo . --out <scratch dir>/contact_cases.jsonl
```

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/institutional_4_16/evidence_binding_runner.py --repo . --out <scratch dir> --mutate
```

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/institutional_4_16/build_evidence_cases.py --out <scratch dir>/evidence_cases.jsonl
```

Use the system interpreter (the repository `.venv` is broken) and `-X utf8`. The runner writes only to `--out`. No real
model, no model download, no FAISS rebuild, no paid API: the specialist runs on the fake embedder and the stand-in
rerankers of the test fixtures, and its directory lookups do not depend on ranking.

## Expected counts (Step 4.16-B2 code, 2026-10-24)

226 runs (113 cases, two rankings): 226 `as_expected`, 0 `blocker`, 0 `mismatch`, all deterministic; statuses 100
supported, 52 partial, 74 not_found; 118 numbers given, each equal to its printed cell. Mutation: every one of the six
protections, switched off alone, produces at least one `blocker` (`_B2_ROW_START`: SYN-03; `_B2_ROW_COMPLETE`: C6-07,
C6-09, C7-05; `_B2_NEIGHBOR_CHECK`: SYN-01, SYN-02; `_B2_COLUMN_HEADER`: SYN-04, SYN-05; `_B2_SINGLE_ENTRY`: SYN-06;
`_B2_REFERENT_CHECK`: C12-01, C12-02, C12-04 to C12-07); after restoring, 113 of 113 `as_expected`.

## Expected counts (Step 4.16-B3 code, 2026-10-25)

Contact accuracy: unchanged, 226 runs (113 cases) as expected, the same statuses and the same 118 numbers; every `_B2_*`
protection still killed. Case C15-01's rationale was updated and C15-02 became "What is the grading policy?" (its
former question, on health and safety, is answered since phase B3). Evidence binding: 690 runs (345 cases, two
rankings): 690 `as_expected`, 0 `blocker`, 0 `mismatch`, all deterministic; statuses 110 supported, 60 partial, 520
not_found; 258 statements and 4 location entries read. Mutation: every one of the eleven `_B3_*` protections, switched
off alone, produces at least one `blocker` on the cases it is meant to catch (`_B3_SUBJECT_BINDING`: B20, B23;
`_B3_RESPONSIBILITY_VERB`: F05; `_B3_ANAPHORA_STRICT`: B23; `_B3_LIST_EXTENT`: A16; `_B3_DUTY_ALL_TERMS`: A07, B13,
C09; `_B3_AMBIGUITY_CHECK`: S01; `_B3_OWNERSHIP_GUARD`: H08, H16; `_B3_PROCEDURE_GUARD`: G08; `_B3_LOCATION_ENTRY`: S02;
`_B3_SAME_NAME_CHECK`: E01; `_B3_OFFICE_WORDS_EXCLUDED`: B22); after restoring, 345 of 345 `as_expected`.

## Expected counts (Step 4.16-B4 code, 2026-10-25)

Contact accuracy: 226 runs (113 cases) `as_expected`, 0 `blocker`, 0 `mismatch`, deterministic; statuses 100 supported,
52 partial, 72 not_found, 2 out_of_scope (C15-02, "What is the grading policy?", a Teaching subject: out_of_scope with no
handoff since phase B4); the same 118 numbers; every `_B2_*` protection still killed. Evidence binding: 690 runs (345
cases) `as_expected`, 0 `blocker`, 0 `mismatch`, deterministic; statuses 110 supported, 60 partial, 474 not_found, 46
out_of_scope. The 23 cases whose subject another specialist owns moved from not_found to out_of_scope with no finding:
9 handed off because a shared cue names the owner (B01, B02, H01, H09, H10, T63 to Faculty Services & HR; D21, H02, H06
to Research & Innovation), 4 that only the local HR word list claims, with no handoff (H13, R75, R76, R77), and 10 Teaching
subjects with no handoff (H03, H04, H05, H07, H08, H12, H14, H15, H16, H17); every supported and partial result is
unchanged. Every `_B3_*` protection is still killed by the same cases; after restoring, 345 of 345
`as_expected`.

```bash
"C:/Program Files/Python311/python.exe" -X utf8 qa/institutional_4_16/coverage_runner.py --repo . --out <scratch dir>
```

## Expected counts (Step 4.16-B5, 2026-10-26; production code unchanged since B4)

Coverage evaluation (62 questions, 48 answerable from the handbook and Institutional's): system path 3 correct, 3 correct
partial, 34 abstained, 22 not selected by the Coordinator; direct path 3 correct, 1 correct partial, 58 abstained. Answer
correctness on answered questions 6/6 (system) and 4/4 (direct); 0 unsafe supported and 0 wrong answers; 14/14 correct
refusals of unanswerable or foreign questions; citations 6/6 and 4/4 on their expected pages. Coverage of
source-answerable questions 6/48 (system) and 4/48 (direct). Coordinator: exact selection 34/62, Institutional inclusion
38/62. A second run is byte-identical. Evidence binding and contact accuracy are unchanged from Step 4.16-B4.

## Step 4.16-B6 (default dispatch, D-105)

`coverage_runner.py --default-dispatch` runs the system path without the `executable=` override. Its output is
byte-identical to the Step 4.16-B5 output (and to `coverage_baseline_b5.jsonl`): routing 34/62 exact, coverage 6/48, 0
wrong answers. Evidence binding (690/690) and contact accuracy (226/226) are unchanged, every switch still killed.

## Step 4.16-C1 (natural phrasing, 2026-10-26)

`coverage_runner.py --cases holdout_cases.jsonl` runs the frozen holdout instead of the benchmark (default
`coverage_cases.jsonl`). The benchmark questions, expectations, judge and metrics are unchanged.

- Benchmark, default dispatch (final code): system path 8 correct, 4 correct partial, 28 abstained, 22 not selected;
  direct path 9 correct, 1 correct partial, 52 abstained. Coverage of source-answerable questions 12/48 (system; was
  6/48) and 10/48 (direct; was 4/48). 0 wrong or unsafe answers, 14/14 refusals kept, citations 12/12 and 10/10, no
  answer of B5 lost, Coordinator selections identical (routing 34/62 exact, inclusion 38/62). A second run is
  byte-identical.
- Holdout (final code): coverage 5/24 system (was 2/24) and 5/24 direct (was 0/24); 0 wrong answers, 12/12 refusals
  kept, citations 5/5 and 5/5; Coordinator selections identical (11/36 exact, 18/36 inclusion). A second run is
  byte-identical.
- The first C1 version measured 14/48 and 15/48 (benchmark) and 7/24 and 11/24 (holdout); the adversarial review showed
  that its open context handling was unsafe, and the closed allowlist gives those readings up.
- Evidence binding: 690 runs (345 cases) `as_expected`, deterministic; statuses 112 supported, 62 partial, 470 not_found,
  46 out_of_scope; 257 statements (unbound sentences of the eight added units are now bound to them) and 4 location
  entries. Three expectations rested on a unit being outside the lexicon and changed with it (`C1_LEXICON_UNITS` in
  `build_evidence_cases.py`): D09 supported (p24 "The Deans Council is chaired by the Chancellor"), U17 partial (the HR
  telephone; the CLD is not in the directory, as for every chapter unit), S01 allows the Mail Office candidate (still
  partial). Every `_B3_*` protection is still killed; `_B3_SUBJECT_BINDING` is now caught by B20 (B23 is caught by
  `_B3_ANAPHORA_STRICT`), and `tests/test_institutional_natural_phrasing.py` adds a negative control for it on the
  added Deans Council.
- Contact accuracy: unchanged, 226 runs `as_expected`, the same 118 numbers; every `_B2_*` protection still killed.
