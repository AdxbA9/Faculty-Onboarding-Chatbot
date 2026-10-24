# Development Status

Handoff log for the Part 2 implementation. Newest entry first. Update this
file at the end of every meaningful work session, before the final push, so
the next developer can start from the branch and commit named here.

Rules: one entry per session; commit hashes are the short hashes shown by
`git log --oneline`; wording stays factual and neutral.

---

## 2026-10-24 - Faculty Services & HR specialist frozen (Milestone 4, step 4.15)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | HR frozen for Milestone 4 (D-103); not yet committed; architecture inactive; Teaching, Research and HR frozen; Institutional Navigation (Step 4.16) not started |

### Completed

- Step 4.15: the freeze evidence of Step 4.14-W1 verified against the
  current files (the recorded results postdate the last code edit; the code
  hashes match the Step 4.14-W1 record; nothing re-run because nothing was
  missing or inconsistent); the freeze recorded as D-103 with the baseline
  hashes, the accepted test counts, the accepted limitations, the component
  boundary and the change-control rule; a freeze manifest and the Git
  checkpoint instructions prepared for the team in
  `SENIOR_PROJECT_HANDOVER/HR_STEP_4_15/`.

### Tests performed

None re-run in this step. Accepted results (Step 4.14-W1, on the frozen code):
focused 1,167 passed; HR suite 1,309 passed; full suite 3,793 passed; 0
failures, 0 errors, 0 skipped, 0 xfailed; replay 0 changed runs; mutation 15 /
19 (targeted), 4 / 19 (reconstructed), 17 distinct.

### Known issues at handoff

The accepted limitations of D-103 and `docs/FACULTY_SERVICES_SPECIALIST.md`
("Freeze status"). The Milestone 4 work since `bb8432d` (Teaching, Research,
HR, the shared modules, the Coordinator changes, their tests, `qa/` and the
docs) is uncommitted; the team's Git checkpoint is the next action;
`.venv_broken_backup/` is not ignored by `.gitignore` and must never be
staged.

### Files changed

`docs/DECISIONS.md` (D-103), `docs/STATUS.md`,
`docs/FACULTY_SERVICES_SPECIALIST.md` (freeze status). No code, test or QA file
changed.

---

## 2026-10-24 - HR conditional-language correction (Milestone 4, steps 4.14-R2 and 4.14-W1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; HR ready for the Step 4.15 freeze decision; Step 4.15 not started |

### Completed

- Step 4.14-R2 (final freeze-readiness review, read-only): D-101 verified
  correct and monotonic (in code and over 466 instrumented runs: 0 looser
  decisions), D-100 still holding, every recorded gate genuine; one
  freeze-blocking finding, HR-R2-W1: the N1 evidence with "when" instead of
  "if" was supported (also "whenever" and a deontic variant), because D-098 /
  D-099 accepted every "when" trigger.
- Step 4.14-W1 (D-102, Option A chosen by the team): the D-099 class-C rule
  reads a "when" / "whenever" restriction of the asked predicate exactly as an
  "if" restriction (same exemptions; a concessive "even when" exempt; "once",
  "upon", "after", approval gates and existential claims unchanged); switch
  `_C12_WHEN_CONDITION` (nineteen `_C12_*` switches, all on). Two expectations
  moved, each with a companion that asks the condition: the "when promoted"
  salary row of the class-C table and the page-63 "Does confirmation of
  employment include the probation period?" (now partial; "... when
  calculating the End of Service Benefit?" supported).

### Tests performed

The W1 cases reproduced as SUPPORTED on the unchanged code ("when", "whenever",
deontic; the "if" control and HR-414-N1 partial). Focused gate (corrections,
support gate, composite, claims, statements, grounding): 1,167 passed. HR suite
(eleven files): 1,309 passed, 0 failed, 0 skipped, 0 xfailed (was 1,271; +34 in
`tests/test_faculty_services_corrections.py`, +4 in
`tests/test_faculty_services_support_gate.py`). Full suite: 3,755 to 3,793
passed, 0 failed, 0 skipped, 0 xfailed; Teaching 670 and Research 1,221 tests
unchanged and green. The 194 reconstructed cases under both rankings: 0 changed
runs against Step 4.14-N1 (0 supported to partial, 0 partial to supported), D1
and D2 still corrected against the Step 4.14-A baseline, a second replay
byte-identical; the real page-71 "even when" answer stays supported. Switch
mutations: R11 set plus the addendum (42 cases) 15 of 19 killed
(`_C12_WHEN_CONDITION` by SK-A5, SK-A6, SK-A7; no earlier kill lost; survivors
`_C12_ENTITY_SUBJECT`, `_C12_PAREN_NUMERAL`, `_C12_HEDGE`, `_C12_LEADS`);
reconstructed rounds (136 cases) 4 of 19, the same four as before; 17 distinct
switches killed over both sets; deterministic replay in every run. No
real-model run, no model download, no FAISS rebuild, no paid API.
`PLAN_F_ENABLED=False`. teaching.py, research.py, claim_model.py and every
shared module, the Coordinator, the dispatcher, the orchestrator, retrieval,
the knowledge base and the UI byte-identical to the 2026-10-24 snapshot.

### Known issues at handoff

No known materially wrong supported HR answer in the agreed regression scope
(the fixture pages, the 194 reconstructed cases, the R11 and adversarial sets,
the crafted N1 and W1 sets). Accepted conservative downgrades: an omitted
"when" event condition now gives partial (the two moved expectations; asking
with the condition is supported). Unchanged limits: a negative question that
asks its own "when" condition stays partial through the asked-role check;
"once", "upon" and "after" triggers are still accepted; the 27 conservative
misses of the reconstructed set and the F1 / F2 wrapper backlog; crafted-only
evidence for the N1 / W1 shapes; stand-in rankings; real-model validation and
runtime integration outstanding. The Step 4.15 freeze decision and the Git
checkpoint for steps 4.14-A to 4.14-W1 are pending the team (nothing staged or
committed).

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_corrections.py`,
`tests/test_faculty_services_support_gate.py`,
`tests/test_faculty_services_grounding.py` (one expectation moved, one
companion added), `qa/hr_4_14_a/` (SK-A5 to SK-A7, `after_4_14_w1/`, README),
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-102),
`docs/STATUS.md`.

---

## 2026-10-24 - HR condition-ownership correction (Milestone 4, steps 4.14-R and 4.14-N1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; final HR freeze review pending; Step 4.15 not started |

### Completed

- Step 4.14-R (independent review, read-only): every Step 4.14 number
  confirmed from its artefact; P1 and P2 verified against their bounded
  designs; HR-414-N1 confirmed as a reproducible, materially wrong supported
  answer with its root cause in the HR-local own-content exemption
  `_c11_own`; the smallest correction prototyped (no change over the 194
  reconstructed cases); verdict: bounded defect, targeted correction
  required.
- Step 4.14-N1 (D-101): `_c11_own` also judges a restriction on the words
  that distinguish it from its statement's main clause (`_c11_main_roots`),
  so a condition that restates the asked entity ("... the housing allowance
  if they receive university housing") stays a condition; the rule is
  monotonic (it only withdraws a D-098 exemption, never grants one); a
  condition the question asks itself keeps its exemption through
  `_c12_asked_condition`; the three HR call sites pass the claim; switch
  `_C12_OWN_CONTENT` (eighteen `_C12_*` switches, all on). The strict xfail
  that recorded HR-414-N1 is now an ordinary passing test.

### Tests performed

The N1 strict xfail reproduced on the unchanged code and turned XPASS when
the correction landed. Focused gate (corrections, support gate, composite,
claims, statements): 974 passed. HR suite (eleven files): 1,271 passed, 0
failed, 0 xfailed (1,271 collected; was 1,243 passed + 1 xfailed; +27 tests in
`tests/test_faculty_services_corrections.py`, +1 switch row in
`tests/test_faculty_services_support_gate.py`). Full suite: 3,755 passed, 0
failed, 0 xfailed (3,755 collected; was 3,727 + 1 xfailed); Teaching 670 and
Research 1,221 tests unchanged in count and green. The 194 reconstructed cases
under both rankings: 0 changed runs against the final Step 4.14 results,
byte-identical to the Step 4.14-R prototype replay, D1 and D2 still corrected
against the Step 4.14-A baseline, a second replay byte-identical. Seven
crafted N1-shaped claims (negative, affirmative, another entity, heavy
vocabulary overlap, the same word on both sides, the entity restated whole,
eligibility) are partial under both rankings and supported again with the
switch off; asked conditions, the "subject to" case, unconditional claims, D1,
D2 and the true-claim guards are unchanged. Switch mutations: R11 set plus the
addendum (39 cases) 14 of 18 killed (`_C12_OWN_CONTENT` by SK-A2, SK-A3 and
SK-A4; every earlier kill reproduced; survivors `_C12_ENTITY_SUBJECT`,
`_C12_PAREN_NUMERAL`, `_C12_HEDGE`, `_C12_LEADS`, as before); reconstructed
rounds (136 cases) 4 of 18 killed, the same four as in Step 4.14; 16 distinct
switches killed over both sets; deterministic replay in every run. No
real-model run, no model download, no FAISS rebuild, no paid API.
`PLAN_F_ENABLED=False`. teaching.py, research.py, claim_model.py and every
shared module, the Coordinator, the dispatcher, the orchestrator, retrieval,
the knowledge base and the UI byte-identical to the 2026-10-24 snapshot.

### Known issues at handoff

No known materially wrong supported HR answer on the fixture pages or the
crafted sets. HR-414-N1 was only ever shown on crafted sentences; its
incidence on the real handbook is unknown and the full handbook was not
scanned. "Does the furniture allowance cover furniture?" stays supported from
page 106 ("is provided, which can be utilized over four years"): the
documented D-098 / D-099 reader limit that a unit mentioning the asked object
answers a verb it never states; unchanged by D-101. The 27 conservative
misses of the reconstructed set and the F1 / F2 wrapper backlog are
unchanged. Both rankings are stand-ins; real-model validation outstanding.
The final independent HR freeze review and the Git checkpoint for steps
4.14-A to 4.14-N1 are pending the team (nothing staged or committed).

### Files changed

`handbook_bot/agents/specialists/faculty_services.py` (+36 / -11),
`tests/test_faculty_services_corrections.py`,
`tests/test_faculty_services_support_gate.py` (one row),
`qa/hr_4_14_a/` (`hr_4_14_cases_switchkillers_addendum.jsonl` SK-A2 to SK-A4,
`compare_replay.py`, `after_4_14_n1/`, README), `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-101), `docs/STATUS.md`.

---

## 2026-10-24 - HR bounded defect correction (Milestone 4, steps 4.14-A and 4.14)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; Step 4.15 not started |

### Completed

- Step 4.14-A (evidence recovery and QA, repository read-only): the Step
  4.12-R12 / 4.13 audit artefacts (35 reported wrong supported answers) were
  not recoverable from any location; 194 labelled cases were reconstructed on
  the fixture pages and replayed under both stand-in rankings: exactly two
  materially wrong supported answers (a statement about another entity in the
  same sentence, "Is the housing allowance paid once?"; a parenthesised
  numeral, "Is the probation period two (2) years?"), 27 conservative misses
  and 5 defensible over-supports; both corrections prototyped as
  process-local patches with no other change over the 194 cases and the 1,147
  HR tests green. The assets now live in `qa/hr_4_14_a/`.
- Step 4.14 (D-100): P2, `_C12_PAREN_NUMERAL` (a bracketed numeral that
  repeats its number words is dropped from the text the readers read), then
  the refined P1, `_C12_ENTITY_SUBJECT` (a claim with no role actor and no
  object binds its asked entity in the supporting clause when the sentence
  names that entity in another clause); both in the D-099 block of
  `faculty_services.py` (+105 / -5 lines), each behind its own switch; the two
  defect tests ported as ordinary passing tests. The switch-mutation
  investigation found and fixed an over-veto of the ported P1 on a true claim
  ("Does health insurance cover the spouse of faculty members?": a verb the
  claim reader does not anchor had leaked into the entity phrase), classified
  the three former mutation survivors (`_C12_HEDGE` equivalent at the status
  level, `_C12_LEADS` visible at clause level only, `_C12_NEGATIVE_CONDITION`
  without a triggering fixture, now killed by a crafted case) and recorded one
  new, out-of-scope reader gap (HR-414-N1) as a strict xfail.

### Tests performed

Full suite 3,727 passed, 1 xfailed (3,728 tests, 0 failures, 0 errors) in
334 s against the 3,631 baseline (+97 new tests); HR 1,243 passed, 1 xfailed
(1,244) in 253 s against 1,147 (new: `tests/test_faculty_services_corrections.py`,
91 tests, and six rows in `tests/test_faculty_services_support_gate.py`);
the one xfail is the documented HR-414-N1 gap. The two Step 4.14-A defects
reproduced as strict xfails on the unchanged code before any edit, flipped to
XPASS when each correction landed, and now pass as ordinary tests. Replay of
the 194 reconstructed cases under both rankings (388 runs): the only changed
runs are the two defects (supported to partial under both rankings); no
supported true claim regressed; no quoted evidence changed; a second replay
was byte-identical. Switch mutations: the R11 set plus the addendum (36
cases) kill 13 of 17 switches, the survivors being `_C12_ENTITY_SUBJECT` and
`_C12_PAREN_NUMERAL` (killed by R2-07a and R2-12j on the reconstructed set),
`_C12_HEDGE` and `_C12_LEADS` (status-equivalent; focused tests assert their
reading and clause-level effect); the reconstructed set (136 cases) kills 4 of
17 (defence in depth, as in Step 4.14-A); `_C12_LEDGER` is killed by SK-12
again after the P1 refinement; deterministic replay in every run. No
real-model run, no model download, no FAISS rebuild, no paid API.
`PLAN_F_ENABLED=False`. Research files, teaching.py, claim_model.py,
grounding.py, evidence.py, details.py, base.py, contracts.py, registry.py,
scoping.py, domain_cues.py, retrieval_intent.py, coordinator.py, dispatch.py,
orchestrator.py, retrieval.py, sources.py, config.py, the knowledge base and
the UI byte-identical to the 2026-10-24 snapshot.

### Known issues at handoff

HR-414-N1: a claim whose only evidence is conditional is supported when the
condition shares a content word with the claim ("Are faculty members not
entitled to the housing allowance?" against "... not entitled to the housing
allowance if they receive university housing"; `_c11_own` in `_c11_unasked`
treats the condition as the claim's own content). Crafted reproduction, both
rankings; incidence on the real pages unknown; strict xfail in the
corrections file; recommended as a bounded item for the next step with its
own replay. The 27 conservative misses of the reconstructed set (ten wrapped
true claims, four out-of-scope routings, thirteen strict-rule partials) are
unchanged and documented in `docs/FACULTY_SERVICES_SPECIALIST.md`; F1 / F2
(recall through wrappers) are backlog pending a design confirmation.
Real-model validation outstanding. The Git checkpoint for Steps 4.14-A and
4.14 is pending the team's review (nothing staged or committed).

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_support_gate.py` (six rows),
`tests/test_faculty_services_corrections.py` (new), `qa/hr_4_14_a/` (new:
case files, runner, mutation harness, before / after results, README),
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-100),
`docs/STATUS.md`.

---

## 2026-10-05 - HR specific-claim support gate (Milestone 4, steps 4.12-R11 and 4.12-C12)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; the combined Step 4.12-R12 / 4.13 final audit pending |

### Completed

- Step 4.12-R11 (read-only confirmation of D-098, verdict: bounded
  correction required): 26 materially wrong supported answers (17 real, 9 on
  crafted pages) in six classes, the largest one a specific claim that the
  claim reader could not read (a conversational wrapper or a semantic
  prefix) and that the earlier details then supported.
- Step 4.12-C12-A (validation, scratchpad only): the fail-closed contract
  simulated as an in-memory patch over every replay set under both
  rankings; the 43 supported-to-partial changes classified by four judges
  and four adversarial verifiers (28 correct catches, 15 conservative
  losses; the two changes flagged as over-blocks were examined: a hedge
  exemption that would have fixed one let a hedged statement about another
  appointment support a claim and was not kept, the other is a reader limit
  on a nominalised predicate); no stop condition.
- Step 4.12-C12-B (D-099): one HR-local support-gate owner,
  `_specific_claim_support_gate`, in `faculty_services.py`: the specific /
  general boundary, wrapper normalisation that keeps every semantic
  operator, the gate (no claim, no frame or no statement never supports), a
  ledger with no static operator, the clause, comparison, condition (no
  copular bypass), modifier and alternative-filler rules, a lead split off
  by the Coordinator joined back, and the polarity of a fallback statement;
  fifteen module switches. The implementation reproduces the validated
  prototype byte for byte over every replay set before the three additions
  the C12 adversarial transformations and cross-domain gate called for.

### Tests performed

Full suite 3631 passed (0 failures, 0 errors), `pip check` clean, HR 1147
(the 922 earlier tests with five expectations moved to D-099 and one
asked-condition case added, 923 in all, and 224 new in
`tests/test_faculty_services_support_gate.py`), Teaching focused gate 1009,
Research regressions 1221, Teaching mutations 56 / 56, Research mutations 116 / 116,
historical 34 / 34, HR mutations 304 / 304 (the C11 280 plus 24 D-099
ones; 0 survivors, 0 errors). The 26 R11 wrong answers are partial under both
rankings. The 2,672 HR baseline questions, the C10 and R10 campaigns (143
and 150), the 16 R10 analogs, the 35 DR2 probes, the 279 R11 real and 307
R11 synthetic questions and 128 C11 synthetic questions were replayed under
both stand-in rankings: 43 answers changed, all supported to partial (28
correct catches, 15 conservative losses), none partial to supported; no
general question changed its status, quotes or message; no answer changed
its quoted evidence. The 3,290 distinct real-fixture questions split into
3,011 specific and 279 general ones; no specific question without a claim
or frame and no specific answer not covered by the composite is supported.
Adversarial transformations (38 real-fixture questions combining wrappers
and prefixes with must, may, all, every, always, only, under, over, if,
unless and additional, both rankings): no wrong support. Cross-domain:
`llm_calls` 0, handoffs as data, Institutional pending, no foreign-section
quote. In one process, alternating rounds over 800 baseline questions, the
D-099 rules cost 0.0% (96.69 s against 96.70 s with every D-099 switch
off); the replay wall-clock ranged from 131 s to 180 s across runs against
165 s for C11 at the start of the step (machine load); the HR focused gate
(the nine earlier files) took 72.80 s against 73.09 s.
`PLAN_F_ENABLED=False`; executable specialists Teaching, Research and
Faculty Services & HR. Research files, teaching.py, claim_model.py,
grounding.py, evidence.py, details.py, base.py, coordinator.py and
dispatch.py byte-identical.

### Known issues at handoff

Fifteen true or defensible answers are partial under D-099 (leading hedges,
conditions the question does not ask, a time inferred across clauses, an
anaphoric condition, a stated hedge on an exclusive claim, a nominalised
predicate; see `docs/FACULTY_SERVICES_SPECIALIST.md`); the Milestone 5
Verifier may recover them. Reader limits: a clause that only mentions the
asked object still answers a claim whose verb the returned unit never
states; a wrapper's words are part of the retrieval query. Real-model
validation outstanding. The three-specialist cap of a four-domain question
is unchanged. The combined Step 4.12-R12 / 4.13 final audit not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_support_gate.py` (new),
`tests/test_faculty_services_composite.py` (four expectations moved to
D-099), `tests/test_faculty_services_grounding.py` (one expectation moved
to D-099), `tests/faculty_services_support.py` (the verbatim page-104
committee composition), `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-099), `docs/STATUS.md`.

---

## 2026-10-04 - HR statement-level composite claims (Milestone 4, steps 4.12-R10, 4.12-DR2, 4.12-C11-DESIGN and 4.12-C11)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; Step 4.12-R11 confirmation pending |

### Completed

- Step 4.12-R10 (read-only confirmation of D-097, verdict: HR-local design
  review required): four wrong supported answers whose universal, exclusive
  or frequency operator no D-097 dimension evaluated on the sentence that
  answered them.
- Step 4.12-DR2 (design review, read-only) and Step 4.12-C11-DESIGN (design
  only): one statement-level comparison (one predicate statement, one joint
  comparison, one operator ledger, vetoes only from the rest of the returned
  quote) instead of further per-dimension rules; six more probes of the same
  shape.
- Step 4.12-C11 (D-098): `claim_model.py` gains the statement-level schema
  and tables (standard library only; the D-097 claim and statement types
  unchanged); `faculty_services.py` reads each returned unit clause by clause
  into predicate statements and supports a specific question only when one
  statement states the whole claim (one joint comparison of eleven
  dimensions, one operator ledger, vetoes only from the rest of the returned
  quote), as one more detail that decides support and never changes the
  evidence an answer reports; nine module switches, enabled phase by phase
  against an exact replay of the C10 outputs under both stand-in rankings.
- The final replay of every phase found three defects in the new subject
  reading and evidence ranking (an actor read from outside the question's
  subject or from a modifier of the evidence's subject, an actor named after
  a leading antecedent, partial answers reporting other evidence); each was
  corrected within D-098 and every phase was replayed again from the start.
- The ten R10 and DR2 failures: partial with the failed dimension named under
  both rankings.

### Tests performed

Full suite 3406 passed (0 failures, 0 errors), `pip check` clean, HR 922
(the 658 earlier tests and 264 new: 49 in
`tests/test_faculty_services_statements.py`, 215 in
`tests/test_faculty_services_composite.py`), Teaching focused gate 1009,
Research regressions 1221, Teaching mutations 56 / 56, Research mutations
116 / 116, historical 34 / 34, HR mutations 280 / 280 (the D-097 223 plus 57
D-098 ones; 0 survivors, 0 errors; four that stricter rules masked received
direct tests). The 2,672 HR baseline questions, the C10 and R10 campaigns
(143 and 150 questions), the 16 R10 analogs, the 35 DR2 probes and 128
synthetic questions were replayed under both stand-in rankings after each
enablement (the joint comparison; the operator ledger, frequency and safe
leading frames; universal slots; enumerations; alternatives; unconditional
claims; window vetoes): with every switch off the outputs equal C10 byte for
byte; each enablement changed exactly the statuses the design listed for it;
the final state differs from C10 in 15 statuses of real questions, all
supported to partial under both rankings (the four R10 failures, the six
DR2 probes and five true or defensible answers, see below), and in 11
synthetic classes. No answer changed its quoted evidence (the 15 changed
answers keep their findings), and every earlier missing message is kept
with the composite's message added. The replay with every switch on took
124 s against 117 s with every switch off (+6%); the HR focused gate took
64.25 s against 73.49 s at the start of the step. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, evidence.py, details.py, base.py, domain_cues.py, scoping.py,
retrieval_intent.py, coordinator.py, dispatch.py and router.py
byte-identical.

### Known issues at handoff

The statement-level composite (D-098) is strict: five true or defensible
answers are now partial ("Is the end-of-service gratuity always one month's
salary per year of service?", "Is the maximum children's education
allowance 75,000 Dirhams for all children?", "Is the visiting contract
extended with Chancellor approval?", "Is the department chair's appointment
renewable only once?" and a department-level promotions committee question
with "equal to or higher than"). Reader limits: a clause whose verb has no
auxiliary is not split from the clause before it; a role named in a leading
antecedent of the evidence is tolerated rather than read as the actor; a
stated universal with a number in it never answers a universal question,
even one asking the same number. Real-model validation outstanding. Step
4.12-R11 (confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/claim_model.py`,
`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_statements.py` (new),
`tests/test_faculty_services_composite.py` (new),
`tests/test_faculty_services_grounding.py` (one expectation moved to D-098
universal slots), `tests/faculty_services_support.py` (C11 fixture pages),
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-098),
`docs/STATUS.md`.

---

## 2026-10-04 - HR structured claims (Milestone 4, steps 4.12-DR1, 4.12-C10-DESIGN and 4.12-C10)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; Step 4.12-R10 confirmation pending |

### Completed

- Step 4.12-DR1 (HR-local design review, read-only, verdict A) and Step
  4.12-C10-DESIGN (design only, verdict A): a bounded HR-local structured
  claim model for the three claim dimensions the D-096 contracts do not read.
- Step 4.12-C10 (D-097): new `handbook_bot/agents/specialists/claim_model.py`
  (standard library only; imported only by `faculty_services.py`) with the
  claim and statement schema, the operator readers and the compatibility
  tables; HR adapters in `faculty_services.py` reading modality bound to the
  asked action, quantifier scope (universal, generic, existential,
  proportional), exclusivity with its focus, condition roles, fail-closed
  operators and a window veto, each behind a module switch. Enabled phase by
  phase against an exact 2,672-question replay of the C9 outputs under both
  stand-in rankings.
- The 11 known failures: partial with the dimension named under both
  rankings; the 8 paired controls supported.
- Fresh campaign: 143 new questions (modality, scope, exclusivity, condition
  roles, unknown operators, combination attacks, controls) under both
  rankings, every supported answer read (33 questions): 0 materially wrong.
  Against the exact C9 module, 30 C9 supported answers became partial: 25
  were wrong or over-claimed at C9 (four of them wrong in ways this campaign
  first exposed, closed within the D-097 dimensions) and 5 true universal
  answers fell to the strict universal rule; one more C9 answer kept its
  status with a corrected quote (a visiting-contract sentence no longer
  answers a probation question).

### Tests performed

Full suite 3142 passed (0 failures, 0 errors), `pip check` clean, HR 658
(the 297 earlier tests and 361 in `tests/test_faculty_services_claims.py`),
Teaching focused gate 1009, Research regressions 1221, Teaching mutations
56 / 56, Research mutations 116 / 116, historical 34 / 34, HR mutations
223 / 223 (the C9 179 plus 44 D-097 ones; 0 survivors, 0 errors). The 2,672
HR baseline questions replayed under both stand-in rankings after each
enablement (modality; scope; exclusivity; condition roles; operators; the
window veto): with every switch off the outputs equal C9 on all 2,672; the
final state differs from C9 in 18 statuses only (the 11 known failures, "Can
sick leave without pay be granted only in special circumstances?", two
existential negatives now supported, the generic "Are faculty members
entitled to Hajj leave?" and three strict generic narrowings) and, among
answers supported in both, in one primary quote (more focused) and six or
seven secondary findings. The 80-operator set: 65 / 65 kept supported,
10 / 10 now partial, five read by hand. The 23 baseline questions with an
unverified operator are all partial. Cross-domain runs (HR alone and with
Teaching, Research and Institutional, both rankings): 0 flags; handoffs as
data only, no LLM call, Institutional pending, no HR quote from a foreign
section. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists
Teaching, Research and Faculty Services & HR; Institutional pending. Research
files, teaching.py, grounding.py, evidence.py, details.py, domain_cues.py,
scoping.py, retrieval_intent.py, coordinator.py, dispatch.py and router.py
byte-identical.

### Known issues at handoff

Strict universal and generic scope (D-097) leave some true answers partial:
"May all emeritus faculty teach courses as needed?" (the handbook states the
rule generically), "Are female faculty eligible for sixty days of maternity
leave with full pay?" (stated for full-time female faculty). Reader limits:
a relative clause with an object in a question's subject, a compound noun
subject whose second noun can be a verb, and "-ent" / "-ment" verbs outside
the HR verb list leave the asked action unread (the clause then fails closed
or keeps its C9 result). The Teaching modal issue stays documented only.
Real-model validation outstanding. Step 4.12-R10 (confirmation) and 4.13
(adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/claim_model.py` (new),
`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_claims.py` (new),
`tests/test_faculty_services_grounding.py` (one R7 expectation moved to
D-097 strict generic scope), `tests/faculty_services_support.py` (C10
fixture pages), `docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md`
(D-097), `docs/STATUS.md`.

---

## 2026-10-04 - HR semantic consolidation (Milestone 4, steps 4.12-R8 and 4.12-C9)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen; HR-local design review recommended |

### Completed

- Step 4.12-R8 (read-only confirmation, verdict C): 43 materially wrong
  supported answers in ten classes (scope narrowing, conditional evidence
  read as fact, "Is not any X ...?", compound or coordinated labels read as
  actors, role verbs outside the HR inventory dropped, predicate polarity
  outside the D-094 list, relative-clause polarity, clausal absent
  conditions, condition identity, maximum / minimum qualifier binding);
  42 of them present before C8.
- Step 4.12-C9 (D-096, `faculty_services.py` only): the HR-local
  safeguards consolidated into one claim-compatibility layer with ten
  contracts (scope, conditional fact, negative existence, labels, explicit
  verbs, generic predicate polarity, population modifiers, clausal
  conditions, condition identity, qualifier identity), six of them behind
  a module switch; contracted negation read with either apostrophe; shared
  grounding unchanged.
- Tests: 297 HR tests across the six files (11 new D-096 invariant tests
  with metamorphic pairs; two C8 assertions that let a narrower negation
  settle a broader question were updated).
- The 43 R8 cases: 43 / 43 now `partial` under both stand-in rankings, and
  the HR sub-task of both dispatch repeats `partial`. Every earlier campaign
  replayed against the C8 outputs with the intended changes only. A fresh
  campaign of 137 new questions and six dispatch runs, every SUPPORTED read:
  0 materially wrong; against the exact pre-C9 module (reconstructed from
  the session record, md5 matching the recorded pre-C9 hash) ten C8-wrong
  answers became `partial` and one true negative was narrowed by scope.
- Breadth sets of 56 and 34 questions outside the ten classes: 9 wrong
  supported answers in three claim dimensions the contracts do not read,
  identical at C8 (modal strength, universal quantification / affirmative
  scope, "only" exclusivity). Not patched: an HR-local design review is
  recommended.

### Tests performed

Full suite 2781 passed (0 failures, 0 errors), `pip check` clean, HR 297,
Teaching focused gate 1009, Research regressions 1221, Teaching mutations
56 / 56, Research mutations 116 / 116, historical 34 / 34, HR mutations
179 / 179 (the accepted 150, one of them re-pointed because the generic
verb fallback now keeps an explicit role verb, plus 29 new; 0 survivors,
0 errors). The six Teaching corpora replayed with 0 changed rows; the 45
earlier HR probe sets replayed against the C8 outputs with the 43 R8 cases
now `partial`, two true negative questions narrowed by scope and no other
change. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists
Teaching, Research and Faculty Services & HR; Institutional pending.
Research files, teaching.py, grounding.py, domain_cues.py, coordinator.py
and dispatch.py byte-identical.

### Known issues at handoff

HR-local design review recommended (D-096): modal strength ("Must ...?"
settled by "may"; the shared deontic gate accepts any normative marker),
universal quantification and affirmative scope ("Are all faculty members
eligible / entitled ...?"), "only" exclusivity. Real-model validation
outstanding. No personal HR data source or workflow system exists.
Conservative misses as before, plus D-096 scope narrowings ("Does the award
not impact salary?" against "the monthly salary"). Step 4.12-R9
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/faculty_services_support.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-096),
`docs/STATUS.md`.

---

## 2026-10-02 - HR negative-ask polarity, entitlement-synonym, multi-negation and role-verb correction (Milestone 4, steps 4.12-R7 and 4.12-C8)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R7 (read-only confirmation, verdict C): every C1 to C7
  protection held; four classes surfaced (negative asks settled by the
  affirmative rule, entitlement synonyms read without polarity, a second
  "without" frame swallowed by the pay status, a role-subject verb dropped
  and a policy label read as an actor).
- Step 4.12-C8 (D-095, `faculty_services.py` only): a negative ask needs
  the same negation of the same subject (value, relation, predicate and
  "there is no" existence asks); qualified / receive / get / given / provided / awarded join the
  entitlement family with polarity; each absence frame is read on its own;
  role-subject action verbs are kept; policy, section, chapter and
  guideline labels are not actors.
- Tests: 286 HR tests across the six files.
- Every R7 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign of 100 questions and two
  breadth rounds of 44 and 27 on the real fixture under both stand-in
  rankings, every SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2770 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 150 / 150 (the accepted 125 plus 25 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before, plus "Does the Chancellor
extend the visiting contract?" and two paraphrased true negative questions
now `partial`. Step 4.12-R8 (confirmation) and 4.13 (adversarial review) not
yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/test_faculty_services_specialist.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-095),
`docs/STATUS.md`.

---

## 2026-10-02 - HR negation-scope, evidence-polarity, leave-state and verb-binding correction (Milestone 4, steps 4.12-R6 and 4.12-C7)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R6 (read-only confirmation, verdict C): every C1 to C6
  protection held; five classes surfaced (an absence window wider than its
  clause, missing "need no / require no" frames, affirmative asks settled
  by negated evidence, an unread leave-state phrasing, verbs dropped for
  non-role and "is X verbed" asks).
- Step 4.12-C7 (D-094, `faculty_services.py` only): negation governs its
  own clause; new absence frames; an affirmative predicate is bound against
  negated evidence (a negative rule now gives `partial` with the negation
  named, superseding the 4.11 reading); leave states read in more words;
  explicit verbs kept for "is X verbed (by Y)" and active non-role asks,
  with the agent bound to the actor; relational verbs never stand for an
  action.
- Tests: 278 HR tests across the six files (one 4.11 specialist test
  rewritten to the D-094 contract).
- Every R6 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign and a breadth round on the
  real fixture under both stand-in rankings, every SUPPORTED read, 0
  materially wrong.

### Tests performed

Full suite 2762 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 125 / 125 (the accepted 101 plus 24 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before (the two D-093 losses, the
"Does termination for cause require the Chancellor's approval?" retrieval
miss), plus "Is the visiting contract extended by the Chancellor?" now
`partial` (the handbook says "with Chancellor approval"). Step 4.12-R7
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/test_faculty_services_specialist.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-094),
`docs/STATUS.md`.

---

## 2026-10-02 - HR condition-polarity and residual passive-binding correction (Milestone 4, steps 4.12-R5 and 4.12-C6)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R5 (read-only confirmation, verdict C): every C1 to C5
  protection held with no overblocking; a condition stated as absent was
  settled by quotes stating it as required (six questions), and an
  agentless passive ask was settled through a relation the shared parser
  read from a noun ("Can the probation period be confirmed?").
- Step 4.12-C6 (D-093, `faculty_services.py` only): a negated condition
  needs a stated absence; a leave state keeps its polarity; "without pay"
  stays a pay status; the passive binding runs also through noun relations,
  with a noun form bound only through its "of" phrase, an active object
  never read after a passive auxiliary or a preposition, and relational
  verbs ("includes") stating no action.
- Tests: 266 HR tests across the six files, with a verbatim p63
  confirmation fixture kept out of the shared page list.
- Every R5 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign with breadth rounds on the
  real fixture under both stand-in rankings, every SUPPORTED read, 0
  materially wrong.

### Tests performed

Full suite 2750 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 101 / 101 (the accepted 87 plus 14 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before, plus "Does termination for
cause require the Chancellor's approval?" (`not_found` under both stand-in
rankings: the candidate chunks are printed prose lines; "Is Chancellor
approval required for termination for cause?" is supported). Step 4.12-R6
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`, `tests/faculty_services_support.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-093),
`docs/STATUS.md`.

---

## 2026-10-02 - HR sameness-binding, personal-status and passive-action correction (Milestone 4, steps 4.12-R4 and 4.12-C5)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R4 (read-only confirmation, verdict C): every C1 to C4
  protection held; sameness evidence was not bound to the asked comparison
  (four promotion questions) and a decision on the asker's own case was
  answered from policy (five questions).
- Step 4.12-C5 (D-092, `faculty_services.py` only): sameness evidence bound
  to the asked subject, comparison and scope, bare "equal" and "at the same
  time" no evidence; grant, approval, acceptance and other decision status of
  the asker's own case fail closed while policy and eligibility questions
  stay answerable; the campaign's passive class ("Can the probation period
  be extended?" quoting the visiting-contract sentence) closed by binding an
  agentless passive verb to the asked subject.
- Tests: 258 HR tests across the six files, with verbatim fixtures for
  p69, p148 B, p149 D, p150 D and p152 kept out of the shared page list.
- Every R4 wrong case replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign and breadth rounds on the real
  fixture under both stand-in rankings, every SUPPORTED read, 0 materially
  wrong.

### Tests performed

Full suite 2742 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 87 / 87 (the accepted 69 plus 18 new), the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source or workflow
system exists. Conservative misses as before (D-088 base asks, the "earn"
and single-bound salary phrasings, the department committee chair sentence),
plus explicit sameness statements that use other words than the question
("Is the minimum residence period for promotion irrespective of prior years
of service?" stays partial). Step 4.12-R5 (confirmation) and 4.13
(adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_specialist.py`, `tests/faculty_services_support.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-092),
`docs/STATUS.md`.

---

## 2026-10-02 - HR universality, qualified-rank and personal-record correction (Milestone 4, steps 4.12-R3 and 4.12-C4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R3 (read-only confirmation, verdict C): every C1 to C3
  protection held; three older classes surfaced under new phrasings:
  sameness claims settled by topic overlap, "senior" ranks inheriting the
  base salary row, and the asker's own leave usage answered from policy.
- Step 4.12-C4 (D-091, `faculty_services.py` only): sameness and
  independence claims need stated sameness or independence; a qualified
  rank has no salary row ("full professor" the one alias, its category
  stated by the Professor row); personal-record asks fail closed with a
  limitation; hypothetical and policy asks stay answerable.
- Tests: 242 HR tests across the six files (sameness and independence
  frames and evidence, qualified ranks and the alias, personal usage,
  remaining and balance asks against policy asks).
- Every R3 reproduction replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh campaign with breadth probes on the
  real fixture under both stand-in rankings, every SUPPORTED read, 0
  materially wrong.

### Tests performed

Full suite 2726 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 69 / 69, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. No personal HR data source exists.
Conservative misses as before (D-088 base asks, the "earn" and single-bound
salary phrasings, the department committee chair sentence, "Are senior
lecturers appointed on a definite contract for two years?"). Step 4.12-R4
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_tables.py`, `tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_specialist.py`, `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-091), `docs/STATUS.md`.

---

## 2026-10-02 - HR salary-header and named-source frame correction (Milestone 4, steps 4.12-R2 and 4.12-C3)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R2 (read-only confirmation, verdict C): every C1 and C2
  protection held; the salary-table header settled unrelated salary
  propositions, and "Does the Employee Handbook say that ..." and the full
  Benefits Chart name in prepositional frames escaped the named-source guard.
- Step 4.12-C3 (D-090, `faculty_services.py` only): the salary header is
  schema, never a row, a quote or evidence; a salary data row settles only
  its rank's amounts and currency; strong source frames (say / state /
  require / specify / provide that; according to / under / in / per; the
  full chart name) are contents asks; "as per the ..." inside the asked
  proposition stays a mention.
- Tests: 231 HR tests across the six files (header and row predicates,
  header never quoted, header and row grounding, thirteen unsupported salary
  propositions, the strong source frames, mention versus content). The 4.11
  assertion that the header is a table row now states the D-090 contract.
- Every R2 reproduction replayed safe; every earlier campaign replayed with
  the intended changes only; a fresh targeted campaign on the real fixture
  under both stand-in rankings, every SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2715 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 60 / 60, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Conservative misses as before (D-088
base asks, the department committee chair sentence, the Faculty Handbook
"provides an Employee Handbook" phrasing). Step 4.12-R3 (confirmation) and
4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/test_faculty_services_tables.py`, `tests/test_faculty_services_specialist.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-090),
`docs/STATUS.md`.

---

## 2026-10-01 - HR actor preservation and named-source correction (Milestone 4, steps 4.12-R1 and 4.12-C2)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12-R1 (read-only confirmation, verdict C): the Step 4.12
  numeric and leave classes stayed fixed; two bounded classes remained: a
  named actor lost when the shared parser builds no or a degenerate
  relation ("Does the Chancellor form the College Academic Promotions
  Committee?"), and a mentioned-only source answering for its contents
  ("What does the Employee Handbook say about settling in the UAE?").
- Step 4.12-C2 (D-089, `faculty_services.py` only): named roles stay
  demanded; an HR-built relation binds the named actor to the stated
  action and is stated only by an occurrence whose subject or agent is
  that actor; navigation segments demand no actor; the department,
  college and university promotion committees are bound as distinct
  bodies; Employee Handbook and Benefits Chart contents asks fail closed
  with the mention quoted.
- Tests: 219 HR tests across the six files (actor matrix on the
  promotion pages, form / chair collisions, verb-less role asks, committee
  identity, the administrative-role axis, named source detection, Employee
  Handbook and Benefits Chart contents versus mention).
- Every R1 reproduction and breadth case replayed safe; every earlier
  campaign replayed with the intended changes only; a fresh targeted
  campaign on the real fixture under both stand-in rankings, every
  SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2703 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 52 / 52, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py,
grounding.py, domain_cues.py, coordinator.py and dispatch.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Conservative misses: "Does the Faculty
Handbook say HR provides an Employee Handbook?" is partial (the parser
reads "provides" oddly); the D-088 base-ask misses; the department
committee chair sentence with "The Committee" as subject. Step 4.12-R2
(confirmation) and 4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/faculty_services_support.py`, `tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_specialist.py`, `tests/test_faculty_services_tables.py`,
`docs/FACULTY_SERVICES_SPECIALIST.md`, `docs/DECISIONS.md` (D-089), `docs/STATUS.md`.

---

## 2026-10-01 - HR isolation QA and fact-grounding correction (Milestone 4, steps 4.12 and 4.12-C1)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.12 (read-only isolation QA, verdict C): ownership, shared pages
  and shared sections, Teaching / Research / Institutional boundaries,
  source gaps, Coordinator cap and routing all held (0 foreign quote leaks,
  0 silent domain loss); five materially wrong SUPPORTED results were found
  in the HR fact gate on owned pages (probation "six months" from the
  resignation notice, maternity "thirty days" and sick leave "30 days" from
  the additional unpaid tranches, notice "one year" from the probation
  period, the department-level procedure from the college-level heading).
- Step 4.12-C1 (D-088, `faculty_services.py` only): the fact's object must
  measure the asked subject; notice, contract and appointment are demanded
  anchors; a base ask is never settled by a number marked additional, up to
  / maximum, minimum or without / half pay; promotion-procedure asks bind
  the department / college / university level to the item's first-named
  body; headings and list lead-ins carry no level and are never quotes;
  "what is the ... procedure" is a process request; a nominalised HR concept
  or "pay" in a pay-status phrase builds no relation; a dropped relation
  keeps the roles the ask names and HR verbs (chair, form, nominate...)
  keep theirs; "level" is wording.
- Tests: 212 HR tests across the six files (new matrices for
  probation / notice, maternity, sick leave, the demanded anchors, the
  tranche rules, the organisational levels and the heading rule; fixtures
  `P64_APPT`, `P149_HEAD`, `P150`).
- Every Step 4.12 reproduction replayed safe; the 4.12 campaigns and the
  4.11 smoke replayed with the intended changes only; a fresh targeted
  campaign on the real fixture under both stand-in rankings, every
  SUPPORTED read, 0 materially wrong.

### Tests performed

Full suite 2696 passed (0 failures, 0 errors), `pip check` clean, Teaching
focused gate 1009, Research regressions 1221, Teaching mutations 56 / 56,
Research mutations 116 / 116, historical 34 / 34, HR mutations 43 / 43, the
R2 to R6 Teaching corpora replayed with 0 changed rows. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching, Research and Faculty
Services & HR; Institutional pending. Research files, teaching.py and
grounding.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Conservative misses added by D-088: a
base ask against an "up to" entitlement ("Is Hajj leave twenty days?", "Is
sick leave 15 working days?") is partial. Step 4.12 MINORs left for later
steps: general-ask quote order under the stand-in rankings, page-level
section labels on shared pages, the Coordinator cap attaching a dropped
domain's clause through a weak hint, actor-only probation scope, harmless HR
co-selection on "rank" and "pay". Step 4.12-R1 (isolation confirmation) and
4.13 (adversarial review) not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py`,
`tests/faculty_services_support.py`, `tests/test_faculty_services_grounding.py`,
`tests/test_faculty_services_tables.py`, `docs/FACULTY_SERVICES_SPECIALIST.md`,
`docs/DECISIONS.md` (D-088), `docs/STATUS.md`.

---

## 2026-10-01 - Faculty Services & HR Specialist implemented (Milestone 4, steps 4.9 to 4.11)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive; Research and Teaching untouched and frozen |

### Completed

- Step 4.9 (read-only audit, passed with source gaps) and Step 4.10 (design,
  passed with minor open questions): HR ownership, boundaries, source
  coverage, routing collisions, reuse, grounding needs, test and mutation
  architecture.
- Step 4.11 (implementation, D-085 to D-087): the Faculty Services & HR
  Specialist (`faculty_services.py`) built on the shared helpers and the
  shared HR cue lexicon added to `domain_cues.py`; section-level ownership
  with explicit exclusions and bounded shared sections; the HR fact gate
  (leave type, pay status, category, administrative role, rank, money
  ranges and qualifiers, extension and stage wording, one local item); the
  salary-table row shape; the compensation list split by role; procedure
  requests at most partial; source gaps fail closed; navigation handed to
  Institutional; the Coordinator's inline HR cue block replaced by the
  shared lexicon; `EXECUTABLE_SPECIALISTS` extended to Faculty Services &
  HR; the package docstring updated. Both open questions resolved: the VCAA
  and the Vice Chancellor stay distinct roles; the chapter 2 preamble (p56)
  is the one extra page (verified with `compile_scope` against the real
  section map), the chapter 4 lead-in needs none.
- Tests: `tests/test_faculty_services_scope.py`, `_specialist.py`,
  `_grounding.py`, `_tables.py`, `_alignment.py`, `_integration.py` and
  the shared fixtures `tests/faculty_services_support.py` (201 tests
  across the six files).
- Implementation smoke on the real fixture (two stand-in rankings, mixed
  questions through the dispatcher): every SUPPORTED read, 0 materially
  wrong SUPPORTED; the R2 to R6 Teaching corpora replayed unchanged.

### Tests performed

Full suite 2685 passed (0 failures, 0 errors; 2484 before this step),
`pip check` clean, Teaching focused gate 1009, Research regressions 1221,
Teaching mutations 56 / 56, Research mutations 116 / 116, historical
34 / 34, HR mutations 31 / 31, 4 shared-page guard fixtures 0 leaks, 0 D-078
route violations. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable
specialists Teaching, Research and Faculty Services & HR; Institutional
pending; no Verifier, synthesis, orchestrator or inter-specialist execution.
Research files byte-identical; teaching.py and grounding.py byte-identical.

### Known issues at handoff

Real-model validation outstanding. Source gaps (Milestone 4.5): the
Employee Handbook, the Benefits Chart (vacation days), leave procedures,
payroll, visa and passport renewal, ID cards, parking, HR forms. Safe false
negatives: approval asks whose object stands far before the passive verb,
asks whose verb the text lacks, deontic asks against informational text,
a possessive role before a number, hyphenated percentage ranges, the
unscoped resignation-notice ask (the probation sentence is scoped). Formal
isolation (4.12) and adversarial (4.13) reviews not yet run.

### Files changed

`handbook_bot/agents/specialists/faculty_services.py` (new),
`handbook_bot/agents/specialists/domain_cues.py`,
`handbook_bot/agents/coordinator.py` (HR cue block replaced by the shared
lexicon), `handbook_bot/agents/specialists/dispatch.py`
(`EXECUTABLE_SPECIALISTS`), `handbook_bot/agents/specialists/__init__.py`
(docstring), six `tests/test_faculty_services_*.py` files and
`tests/faculty_services_support.py` (new), `docs/FACULTY_SERVICES_SPECIALIST.md`
(new), `docs/DECISIONS.md` (D-085 to D-087), `docs/STATUS.md`,
`docs/AGENT_CONTRACTS.md`, `docs/COORDINATOR.md`.

---

## 2026-10-01 - Teaching hyphenated period normalisation (Milestone 4, temporary steps 4.8T-R5 and 4.8T-C6)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R5 (final independent read-only freeze review of D-083): every
  numeric, actor, period, table and historical gate held, except one class:
  a hyphenated period in the question ("per-semester", "per-year") was read
  as no period, so "Is the per-semester workload of visiting faculty 24
  credit hours?" was SUPPORTED by the p91 row and "Does the department chair
  review teaching loads per-year?" by a sentence stating each semester.
- Step 4.8T-C6 (correction, D-084): `frequencies_of` in `grounding.py`
  normalises its text before matching (one call to the existing
  `normalise`), so hyphenated and spaced periods are the same; units stay
  distinct. No change to numeric facts, the relation parser, the
  Coordinator, retrieval, the dispatcher, the contracts or Research.
- Tests: four tests added to `tests/test_teaching_numeric_binding.py`
  (hyphen/spaced equivalence of the reader for per/each/every and every
  unit, unit distinctness, the p91 row and p79 rule with hyphenated
  periods, the chair/dean periods with hyphenated periods).
- Replays: the R5 reproductions PARTIAL; the D-083 cases (p91 pairs, p79
  rule, update/check periods, chair/dean periods) unchanged in spaced and
  hyphenated wording; representative historical replay unchanged; a fresh
  bounded campaign weighted to hyphenated periods, every SUPPORTED read; the
  4 shared-page guard fixtures (0 leaks).

### Tests performed

Focused suites 1009 passed (1005 before this step), full suite 2484
passed (0 failures, 0 errors; 2480 before), `pip check` clean, Teaching
mutations 56 / 56 (the 55 of Step 4.8T-C5 plus the hyphenated ask period
dropped), Research mutations 116 / 116, historical 34 / 34, Research
regressions 1221 passed. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`;
executable specialists Teaching and Research; Faculty Services & HR and
Institutional pending; no Verifier, synthesis, orchestrator or
inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives as documented at
Step 4.8T-C5.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_numeric_binding.py`, `docs/DECISIONS.md` (D-084),
`docs/TEACHING_SPECIALIST.md`, `docs/STATUS.md`.

---

## 2026-09-30 - Teaching numeric / value-frequency binding (Milestone 4, temporary steps 4.8T-R4 and 4.8T-C5)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R4 (final independent read-only freeze review of D-082): every
  earlier class stayed fixed, but a numeric ask was checked word by word
  across the unit: the real workload row "12 credit hours per regular
  semester or 24 credit hours per academic year" supported "24 credit hours
  per semester", and "five office hours per week, spread over a minimum of
  two days" supported "two office hours per week".
- Step 4.8T-C5 (correction, D-083) in the shared grounding module
  (`grounding.py`): `numeric_facts` reads each run of numbers with the
  object it qualifies and the period of its clause (up to the next comma);
  a specific numeric ask must find one local fact with the same values, the
  same object and the asked period (stated after or before the number),
  whose clause names no other subject or category than the asked actor
  (a generic asked subject must stand in that clause), and which does not
  sit in another independent clause without the asked action; an action's
  period must sit in the clause of that action by the asked actor; clauses
  split at ";", ":", a sentence end, "and", "or", "but".
  Ask parser: a role heading a concept phrase is the actor, a concept noun
  after a role is no verb. No change to research.py, the Research tests or
  docs, the contracts, the Coordinator, the dispatcher, the section guard,
  the source scope or the evidence primitives.
- Tests: `tests/test_teaching_numeric_binding.py` (18 tests: the
  fact reader, the four workload-row pairings, the office-hours rule, two
  actions with two periods, values with their objects, an actor with its
  own value and period, a category with its own value, a value in the
  clause of its action, a value of another independent clause, a value of
  a subordinate clause, a generic subject with its own value, a
  coordinated predicate, a period before the number, a period stopping at
  a comma, the parser corrections, table rows, correct facts). One parse case of
  `tests/test_teaching_final_grounding.py` reworded ("course team": the
  coordinator is a lexicon role and now the actor).
- Replays after the correction (real fixture, two rankings): the Step
  4.8T-R4 cases PARTIAL; every earlier set replayed and read; a fresh check
  weighted to several numbers, periods and pairs in one unit; the 4
  shared-page guard fixtures (0 leaks).

### Tests performed

Focused suites, full suite 2480 passed (0 failures, 0 errors; 2462 before
this step), `pip check` clean, Teaching mutations 55 / 55 (the 44 of
Step 4.8T-C4 plus value, object and frequency matched anywhere,
value-object and value-frequency binding disabled, multi-pair cross
product, office-hours value from the days number, semester and year pairs
crossing, action frequency crossing clauses, a period read past a comma, a
fact taken from another independent clause),
Research mutations 116 / 116, historical 34 / 34. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching and Research; Faculty
Services & HR and Institutional pending; no Verifier, synthesis,
orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: an ask with a
relative clause before its verb, a constructed prose line with three or
more digit cells (read as a table row), and the earlier documented
misses.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_numeric_binding.py` (new),
`tests/test_teaching_final_grounding.py` (one parse case),
`docs/DECISIONS.md` (D-083), `docs/TEACHING_SPECIALIST.md`,
`docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-30 - Teaching generic-subject / irregular-passive / sentence-scope grounding (Milestone 4, temporary steps 4.8T-R3 and 4.8T-C4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R3 (final independent read-only freeze re-review of D-081):
  every earlier class stayed fixed, but three residual classes reproduced:
  a subject outside the role lexicon fell to the word bag ("Does the
  quality office assess course files?" from the chairs' duty, "Does the
  academic advisor remove the Incomplete mark?" from a calendar row), an
  agentless passive ask with an irregular participle received no relation
  ("Are final grades sent to the dean?"), and the agent rule read the actor
  from another sentence ("Do faculty transfer grades to Banner?").
- Step 4.8T-C4 (correction, D-082) in the shared grounding module
  (`grounding.py`): generic subject phrases bound as the evidence clause's
  subject (never the word bag); a table row settles no ask that names an
  actor; irregular participles in the passive-ask patterns; the agent rule
  scoped to the sentence of the occurrence, with actor pronouns kept and
  patients ("They are transferred", "Students are informed by") excluded;
  ", which" keeps its antecedent. No change to research.py, the Research
  tests or docs, the contracts, the Coordinator, the dispatcher, the
  section guard, the source scope or the evidence primitives.
- Tests: `tests/test_teaching_final_grounding.py` (22 tests: the
  three classes each with negatives and positives, the generic positive
  control, the table-row actor rule with table values kept, the irregular
  participles with their agents, coordinated verbs, actor and object
  pronouns, ", which").
- Replays after the correction (real fixture, two rankings): every Step
  4.8T-R3 case PARTIAL; the Step 4.8T, 4.8T-R, 4.8T-R2 and 4.8T-R3 sets
  replayed, every `supported` result read; a fresh check weighted to
  unknown subjects, table rows with actors, irregular passives, sentence
  scope, pronouns, coordination, direction, receiving, frequency and mixed
  questions; the 4 shared-page guard fixtures (0 leaks).

### Tests performed

Focused suites, full suite 2462 passed (0 failures, 0 errors; 2440 before
this step), `pip check` clean, Teaching mutations 44 / 44 (the 36 of
Step 4.8T-C3 plus generic subject ignored, generic actor word bag, table
row satisfies an actor ask, irregular "sent" / "kept" / "held" and "made"
removed, actor lookup widened to the unit, object pronoun inherits the
actor), Research mutations 116 / 116, historical 34 / 34.
`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists Teaching
and Research; Faculty Services & HR and Institutional pending; no Verifier,
synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: an agentless
action in the sentence after an instructor duty ("Must I upload the
syllabus on Blackboard?"), an actor ask answered only by a table row, a
generic subject with a light verb, and the earlier documented misses.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_final_grounding.py` (new), `docs/DECISIONS.md`
(D-082), `docs/TEACHING_SPECIALIST.md`, `docs/AGENT_CONTRACTS.md`,
`docs/STATUS.md`.

---

## 2026-09-30 - Teaching residual actor / passive / light-verb / frequency grounding (Milestone 4, temporary steps 4.8T-R2 and 4.8T-C3)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R2 (final independent read-only re-review of D-080): the
  positional classes stayed fixed, but five residual classes reproduced on
  real handbook sentences: the person asking exempt from role grounding
  ("Must I attend the final examination?" from the student rule), a
  subject-less action given to a role named only as recipient ("Do students
  upload the syllabi on Blackboard?"), a passive ask losing its agent before
  "of" ("Are course files assessed by faculty members of the department?"),
  receive / get / obtain bound to nothing ("Do instructors receive an
  Incomplete grade?"), and frequency units dropped as generic ("five office
  hours per semester" from the per-week rule).
- Step 4.8T-C3 (correction, D-081) in the shared grounding module
  (`grounding.py`): the self role demanded as a grammatical subject (or an
  imperative addressed to the reader); `agent_roles` of a unit; the
  passive-ask agent tail; no object-anywhere fallback; receive / get /
  obtain as actions with recipient, object and source; typed frequency
  units (`frequencies_of`) with day / week / month / year words kept as
  terms; "enter" separated from the publish verb class. No change to
  research.py, the Research tests or docs, the contracts, the Coordinator,
  the dispatcher, the section guard, the source scope or the evidence
  primitives.
- Tests: `tests/test_teaching_residual_grounding.py` (33 tests: the
  five classes each with negatives and positives, the agent-phrase tails,
  receive / get / obtain forms, the frequency pairs, the publish / enter
  separation, the earlier position classes). One position test moved its
  impersonal light-verb positive to PARTIAL by D-081.
- Replays after the correction (real fixture, two rankings): every Step
  4.8T-R2 real-fixture MAJOR case PARTIAL; the constructed reversed
  receive / informed-by cases PARTIAL; the 204-question constructed set,
  the 101-question re-review campaign and a fresh position-weighted check
  (self role, scope, subject-less verbs, agents, receiving, frequency,
  mixed), every `supported` result read; the 4 shared-page guard fixtures
  (0 leaks).

### Tests performed

Focused suites, full suite 2440 passed (0 failures, 0 errors; 2407 before
this step), `pip check` clean, Teaching mutations 36 / 36 (the 25 of
Step 4.8T-C2 plus self-role exemption restored, subject-less action for a
recipient allowed, passive agent truncated before "of" and before "whose",
passive object-anywhere restored, receive recipient and source binding
bypassed, semester and day dropped from frequency units, periodically read
as semester), Research mutations 116 / 116, historical 34 / 34.
`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists Teaching
and Research; Faculty Services & HR and Institutional pending; no Verifier,
synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: an impersonal rule
with no faculty subject ("Office hours may be held online") now PARTIAL for
a faculty ask; "each semester" against "each term"; a source asked with
"from" but stated only as a destination; the earlier safe misses ("Who
supervises laboratory sessions?", "How many service hours are expected per
week?", "Does the chair receive approval from the dean?", "each year"
against "periodically", the annual evaluation of performance).

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`tests/test_teaching_residual_grounding.py` (new),
`tests/test_teaching_position.py`, `docs/DECISIONS.md` (D-081),
`docs/TEACHING_SPECIALIST.md`, `docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-30 - Teaching subject / position grounding (Milestone 4, temporary steps 4.8T-R and 4.8T-C2)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T-R (independent read-only re-review of D-079): the original
  MAJOR class stayed fixed, but deliberate grammatical-position attacks
  reproduced wrong-actor, wrong-object, reversed-direction and sense
  collisions marked `supported` ("Must faculty attend all theoretical
  lectures?" from the students' rule; "Does the dean review course files?"
  from the chairs' duty; "Does Banner forward grades to Blackboard?"; the
  calendar line "02 - 12 May 15 -25 Final exams" as a grade percentage;
  "absence of material errors" as attendance).
- Step 4.8T-C2 (correction, D-080): relation binding in the shared
  grounding module (`grounding.py`: `RoleLexicon`, `parse_relation`,
  `RelationChecker`), vocabulary-injected and domain-neutral: the asked
  actor, action, adjective, object and destination must be stated in one
  evidence clause; a different named subject disqualifies; direction is
  read from the preposition class; action nouns bind their agent; the
  recipient of an agentless passive and the object of a causative are not
  its agent; a personal pronoun takes its antecedent; a subject that is no
  role ("course files", "office hours") must be the evidence clause's
  subject; the antecedent pair cannot migrate an actor. Teaching side:
  `GROUNDING_ROLES` (chair, council, dean, registrar, registration,
  faculty, lecturer, student, systems, the full institution name ...),
  role concepts checked by role, the attendance alias restricted to class
  or student context, the percentage family typed (a calendar line is no
  percentage, a grade-table line is), the regrade pattern anchored at a
  word boundary ("Are grades ..." is no regrade). No change to
  research.py, the Research tests or docs, the contracts, the Coordinator,
  the dispatcher, the section guard, the source scope or the evidence
  primitives.
- Tests: `tests/test_teaching_position.py` (32 tests: the role lexicon and
  the parser, the seven confirmed wrong-actor cases each with its positive
  control, the attendance sense, the calendar and percentage typing,
  subject / object swaps, passive agents, recipients, a filler in verb
  position, pronoun antecedents, direction by preposition, party asks
  without a party, a light verb after a concept subject, the object read
  before "against"). One alignment test moved its role assertion from
  concepts to the lexicon.
- Replays after the correction (real fixture, two rankings): the 28 original
  phrasings (0 of 56 runs `supported`); the 23 confirmed cases of Step
  4.8T-C (0 `supported`); the 34 targeted limit attacks (all wrong-actor
  cases `partial`; two safe misses unchanged); the 4 shared-page guard
  fixtures (0 leaks); the 85-question Step 4.8T-R campaign and a fresh
  91-question position-weighted check, every `supported` result read.

### Tests performed

Focused suites, full suite 2407 passed (0 failures, 0 errors; 2375 before
this step), `pip check` clean, Teaching mutations 25 / 25 (the sixteen of
Step 4.8T-C plus subject check bypassed, different subject ignored,
direction reversed, dean and chair merged, registrar and registration
merged, antecedent actor migration allowed, attendance alias widened,
percentage accepts calendar rows, relation parse removed), Research
mutations 116 / 116, historical 34 / 34. `PLAN_F_ENABLED=False`,
`MAX_LLM_CALLS=2`; executable specialists Teaching and Research; Faculty
Services & HR and Institutional pending; no Verifier, synthesis,
orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Safe false negatives: a party ask on a
one-vocabulary clause ("Who supervises laboratory sessions?"), a clause
without a teaching cue ("How many service hours are expected per week?"),
a recipient ask not stated as such ("Does the chair receive approval from
the dean?"), "each year" against "periodically", "Who reviews the workload
allocation each year?" against the annual evaluation of performance. An
impersonal rule ("Office hours may be held online") still answers any
actor, by design.

### Files changed

`handbook_bot/agents/specialists/grounding.py`,
`handbook_bot/agents/specialists/teaching.py`,
`tests/test_teaching_position.py` (new), `tests/test_teaching_alignment.py`,
`docs/DECISIONS.md` (D-080), `docs/TEACHING_SPECIALIST.md`,
`docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-29 - Teaching conservative-supported alignment (Milestone 4, temporary steps 4.8T and 4.8T-C)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; Research untouched and frozen |

### Completed

- Step 4.8T (read-only audit): confirmed one MAJOR Teaching class, 23
  concrete cases: a specific ask marked `supported` on evidence that was on
  the topic but did not establish the asked proposition, relation, deadline,
  quantity, authority or location. Teaching's Milestone 3 contract required
  relevance plus the explicitly requested detail families only.
- Step 4.8T-C (correction, D-079): a shared, domain-neutral ask gate
  (`handbook_bot/agents/specialists/grounding.py`) injected with the Teaching
  vocabulary; general / specific recognised / specific unknown shapes;
  grounding of the ask's specifics in one local unit (concepts, content
  words by stem with verb classes, a normative marker for a deontic
  proposition, a party with the asked action, a number for a value ask, a
  contents statement for a contents ask; table rows name a specific and
  settle no rule; exclusion phrases state nothing); location evidence
  narrowed to systems, electronic places and locative phrases; a pure place
  or contact clause handed to Institutional Navigation and never answered
  from teaching text, a mixed clause given a strict contact detail; the
  substantive hint passed to the D-078 route helper; the grounding detail of
  a pronoun-initial sentence read with its antecedent. No change to
  research.py, the Research tests or docs, the contracts, the Coordinator,
  the dispatcher, the section guard, the source scope or the evidence
  primitives; base.py still imports only the contracts.
- Tests: `tests/test_teaching_alignment.py` (57 tests: the gate's shapes and
  checks, the five Step 4.8 cases, the fresh yes/no, relation and navigation
  classes, deontic, value, party, location, table-row and antecedent rules,
  positive controls, the mixed contact clause, D-078 unchanged, navigation
  intent). Seven existing Teaching tests moved from the permissive contract
  to the conservative one (documented in place); the D-078 Teaching tests
  now expect a pure contact clause to be handed off.
- Replays after the correction (real fixture, two rankings): 22 of the 23
  confirmed cases are `partial` or handed off; the 23rd ("Who reviews the
  workload allocation each year?") is now `supported` by the annual
  evaluation sentence in which the department chair evaluates performance
  against the workload plan each year, a correct answer. The 66-question
  Step 4.8T audit and a fresh 96-question check were rerun and every
  `supported` result read.

### Tests performed

Focused suites, full suite 2375 passed (0 failures, 0 errors; 2318 before
this step), `pip check` clean, Teaching gate mutations 16 / 16 (a scratchpad
harness: gate bypassed, unknown ask allowed, concept, term, deontic, party,
contents, value, location, navigation, antecedent, table-row, verb-form,
yes/no shape, calendar "May", contact detail), Research mutations
116 / 116, historical 34 / 34.
`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists Teaching
and Research; Faculty Services & HR and Institutional pending; no Verifier,
synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding. Word-bag grounding does not bind
grammatical positions (a student attendance rule states an instructor
attendance question); a permission asked of a descriptive sentence stays
`partial` ("Can a course be delivered fully online?" from "Courses offered
fully online use an LMS"); "each year" is not stated by "periodically".
These are safe false negatives or Verifier work.

### Files changed

`handbook_bot/agents/specialists/grounding.py` (new),
`handbook_bot/agents/specialists/teaching.py`,
`tests/test_teaching_alignment.py` (new), `tests/test_teaching_specialist.py`,
`tests/test_teaching_integration.py`, `tests/test_clause_retrieval_intent.py`,
`docs/DECISIONS.md` (D-079), `docs/TEACHING_SPECIALIST.md`,
`docs/AGENT_CONTRACTS.md`, `docs/STATUS.md`.

---

## 2026-09-29 - Research integration review and freeze (Milestone 4, step 4.8)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | reviewed; no production change; Research frozen for Milestone 4; not yet committed |

### Completed

- Read-only integration review of the Research & Innovation Specialist with
  the Coordinator, the registry, the dispatcher, the Teaching specialist,
  the handoff contracts and the source scope. 21 representative Research
  cues route to Research; 6 negative controls, 4 compounds (tasks, focus
  clauses, full question preserved, nothing executed by the Coordinator).
  Registry: deterministic lookup by id or string, duplicates and non
  specialists rejected, unknown ids raise, no aliases, no fallback.
  Dispatch: Teaching and Research execute, Faculty Services & HR and
  Institutional stay pending, an unregistered or non-executable Research
  stays pending, a raising specialist yields a redacted `error`, unknown
  specialist ids fail before dispatch, handoffs stay data.
- 51 fresh integration questions through the real Coordinator and
  dispatcher on the real fixture, two Research rankings: 0 route-rule
  violations (D-078 holds in both clause orders, three-clause and
  three-way shapes), no clause lost, no specialist executed unexpectedly,
  0 LLM calls. Output contract inspected on supported, partial, not_found,
  out_of_scope and error results: every quote traceable to its chunk and
  page, routes and relations reported per clause and per finding.
- Isolation rerun identical to Steps 4.5A-D and 4.7 (0 leaks, 66 / 66
  shared-page checks, 107 ok lines). Import hygiene: the shared route
  helper imports only the router and the route names; no specialist
  module loads the Coordinator, orchestrator, synthesis or Verifier.
- Decision: the Research specialist is frozen for Milestone 4. D-076
  (conservative `supported`) and D-078 (retrieval intent per focus clause)
  remain; only regression fixes if later cross-specialist QA exposes a
  genuine issue; its contracts are the reference pattern for Faculty
  Services & HR and Institutional Navigation (D-077).

### Tests performed

Focused suites, full suite 2318 passed (0 failures, 0 errors), `pip check`
clean. Mutation evidence unchanged (Research 116 / 116, historical 34 / 34;
the four D-078 mutations remain in the harness). Production flags unchanged
(`PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`).

### Known issues at handoff

Real-model validation outstanding (no embedding model or reranker locally;
no download attempted; no accuracy claimed). MINOR, unchanged: safe
Research false negatives on the hashed-embedder fixture, some same-domain
compounds not split, "extension" alone is not a contact word, a clause
without cap vocabulary ("Can I supervise five theses") reaches no
specialist and is reported unowned, and "How is the University's share of
licensing income divided?" selects no specialist (no cue matches "licensing
income"). Observed on the Teaching side, outside
the Research freeze: Teaching still uses its Milestone 3 contract, so a
yes/no or open ask can be `supported` on topic-level evidence and a shared
navigation clause ("where is the curriculum committee office") can be
`supported` from policy text while its handoff to Institutional Navigation
is kept; a bounded Teaching alignment with D-076 is a candidate for a later
step or for the Milestone 5 Verifier.

### Files changed

`docs/STATUS.md` only.

---

## 2026-09-29 - Research MAJOR correction: retrieval intent per focus clause (Milestone 4, step 4.7)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; ready for Step 4.8 |

### Completed

- Step 4.6 (formal adversarial QA, read-only, 134 fresh questions, three
  layers) found one MAJOR: a mixed question such as "Who approves the petty
  cash advance and what is the VCRGS phone number?" gives the Research task
  the Coordinator's joined intent `contact`; the specialist retrieved every
  focus clause under that route, whose boost fills the candidate window
  with printed prose lines it rejects, and the answered substantive clause
  came back `not_found`. Reproduced on one clause (intent `policy`:
  `supported`; intent `contact`: `not_found`), 8 of 18 targeted probes,
  1 of 134 campaign questions, 1 of 66 questions of the Step 4.5B review.
  Teaching had the same path (3 of 4 probes with a Blackboard or office
  hours contact clause).
- D-078: the retrieval route is resolved per focus clause with one shared
  deterministic helper, `specialists/retrieval_intent.py`
  (`resolve_clause_retrieval_intent`): a sole focus clause keeps the task
  intent; a clause of a multi-clause task is classified on its own with the
  Coordinator's router; `contact` is kept only by a clause that is itself a
  contact ask; a clause with a substantive half retrieves as `policy`.
  Research and Teaching call it per clause; the clause report and each
  finding carry `route`. No change to the Coordinator, `SpecialistTask`,
  `CoordinatorDecision`, `SpecialistFindings`, handoffs, ownership, the ask
  shape, the relation contract, the section guard or the source scope.
  `base.py` still imports only the contracts.
- Regression tests (`tests/test_clause_retrieval_intent.py`, 38 tests): the
  exact minimised Step 4.6 reproducer on a crowd-out fixture (policy lines
  beside a thirty-line phone directory in scope) plus a guard that the
  fixture still fails under the retired rule; mixed tasks in both clause
  orders, the TTO case, two substantive clauses beside one contact clause,
  the real Coordinator and dispatcher (Research executed, Institutional
  pending, no clause lost, handoffs as data); pure-contact and pure-policy
  controls; the same shapes for Teaching. Import hygiene covers the new
  module (2 tests).
- Mutations: four new (`clause_route_from_task_intent`,
  `teaching_clause_route_from_task_intent`, `contact_route_self_check_removed`,
  `per_clause_classification_bypassed`), all caught; Research harness
  116 / 116; historical 34 / 34.
- Replays after the correction: the minimised MAJOR and 18 mixed probes
  (17 research halves `supported` through dispatch; the remaining one is
  the unsplit compound "What is the maximum research advance and whom do I
  call about it?", `partial` with its evidence, a known MINOR); the 134
  campaign questions (only change: the research-group-awards mixed question
  is now `supported` in both rankings; 966 findings, 0 traceability
  violations; misleading and excluded rankings unchanged); the 66-question
  Step 4.5B review (only change: the VCRGS contact + petty cash pair is now
  `supported`); source isolation and shared-page checks identical to Step
  4.5A-D (0 leaks; page 175, section guard, source bounds unchanged).

### Tests performed

Focused suites, full suite 2318 passed (0 failures, 0 errors; 2278 before
this step), `pip check` clean, Research mutations 116 / 116, historical
34 / 34. `PLAN_F_ENABLED=False`, `MAX_LLM_CALLS=2`; executable specialists
Teaching and Research; Faculty Services & HR and Institutional pending; no
Verifier, synthesis, orchestrator or inter-specialist execution.

### Known issues at handoff

Real-model validation outstanding (embedding model and reranker not
available locally; no download attempted; no accuracy claimed). Remaining
MINOR findings from Step 4.6: two `supported` results whose condition or
recipient sits in another sentence (conference eligibility; "TTO royalty
share?"), six questions the Coordinator does not route to Research, two
same-domain compounds not split, fixture recall losses under the hashed
embedder ("Who owns my invention?", "How many theses can I supervise at
once?").

### Files changed

`handbook_bot/agents/specialists/retrieval_intent.py` (new),
`handbook_bot/agents/specialists/research.py`,
`handbook_bot/agents/specialists/teaching.py`,
`tests/test_clause_retrieval_intent.py` (new), `tests/test_planf_imports.py`,
`docs/DECISIONS.md` (D-078), `docs/RESEARCH_SPECIALIST.md`,
`docs/TEACHING_SPECIALIST.md`, `docs/AGENT_CONTRACTS.md`,
`docs/COORDINATOR.md`, `docs/STATUS.md`.

---

## 2026-09-29 - Research semantic boundary decision and conservative SUPPORTED contract (Milestone 4, step 4.5A-D)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; ready for Step 4.5B |

### Completed

- Architectural decision (D-076): domain specialists stay deterministic
  evidence specialists; a recognised specific ask is `supported` only with
  a STRONG deterministic relation, an ambiguous relation is `partial`, false
  negatives are preferred to false-positive `supported` results, semantic
  relation verification belongs to the Milestone 5 Verifier and Milestone 4
  does not implement it.
- Relation contract in the Research specialist: a `RelationFrame` read from
  the question (roles, action classes with their objects, purpose,
  receiver, share owner, required thing, permission scope, proposition,
  period, subject / agent / patient positions, prepositional arguments,
  object restrictions, negation); `_relation_ok` proves it in the statement
  that states the answer; the clause report carries `relation` and each
  finding of a recognised ask carries `metadata["relation"]`; an ambiguous
  relation is `partial` with the reason in `missing`.
- Position binding (after the fresh adversarial relation check): a party is
  proved in the position the question gives it; a negated or contrasting
  occurrence states nothing; arguments are stated with their preposition;
  named bodies (IRB, IACUC, Library, Head of ...) are roles; nominate is not
  appoint.
- Milestone 5 Verifier boundary (design only, `docs/AGENT_CONTRACTS.md`):
  inputs, outcomes (verified, downgraded, rejected, retry requested) and the
  rule that only the Verifier may treat an ambiguous relation as
  established. HR and Institutional reuse rule (D-077).
- Tests: `tests/test_research_relation_boundary.py` (new, 271): strong,
  ambiguous, wrong action, wrong role, wrong purpose or beneficiary, general
  controls, position binding, and the rule helpers.

### Adversarial results

- Replay of the 178 Step 4.5A-R failure questions: 4 `supported` under some
  ranking, all correct answers (research committees review grant
  applications; the Board approves major equipment funding).
- Fresh read-only relation check (1069 fresh questions, seven probe
  dimensions, two rankings, real handbook fixture): 29 `supported`
  results, each inspected; 0 MAJOR relation false positives; 4 MINOR
  (a stated relation of looser scope: the outside-authors item for faculty
  book funding, the Dean-approved invoices for the invoice deadline, the
  per-day local per diem for a whole trip); the check was run before and
  after each correction cut, the last run on the final code.
- Recorded false negatives (all `partial` with the evidence quoted) are
  listed in `docs/RESEARCH_SPECIALIST.md`.

### Not done on purpose

No Verifier, no LLM or semantic model inside a specialist, no change to the
Router, retrieval, knowledge base, sources registry, section map,
Synthesis, orchestrator, UI or evaluation; no handoff execution;
`PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2; only Teaching and Research
executable; no model downloaded, no real-model claim.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows:
`tests/test_research_relation_boundary.py` 271,
`tests/test_research_ask_shape.py` 202, `tests/test_research_specialist.py`
310, `tests/test_research_corrections.py` 415,
`tests/test_research_integration.py` 23, `tests/test_planf_coordinator.py`
143, `tests/test_teaching_specialist.py` 467,
`tests/test_teaching_integration.py` 37, helpers 65,
`tests/test_planf_imports.py` and `tests/test_planf_contracts.py` 91; full
`pytest -q` 2278 passed, 0 failures, 0 errors; `pip check` clean. Scratch
mutation checks: 34 historical and 112 Research mutations caught (100%),
including the relation and position mutations (ambiguous treated as strong,
downgrade removed, action / role / purpose / subject / agent / argument /
patient / object / period / negation / conditioned-role bindings removed,
named bodies removed, heading strip removed, nominate merged). Real-fixture
isolation: unchanged from Step 4.5A-R (no excluded-text leak, shared pages
intact, page 175 handling unchanged).

### Next task

Step 4.5B (not started).

---

## 2026-09-29 - Research completeness root-cause correction (Milestone 4, step 4.5A-R)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; architectural decision required before Step 4.5B |

### Completed

- Root cause located: `ResearchInnovationSpecialist.run()` computed the
  uncovered details only over the details the specialist recognised, so an
  unrecognised ask requested nothing and topic relevance made it
  `supported` (fail-open).
- Root rule (D-074): each substantive segment is `general`,
  `specific_recognised` or `specific_unknown` from its question structure;
  an unknown specific ask gets a requested detail no evidence covers; a
  recognised one is grounded (unmapped words, object head, yes/no subject,
  approval act, purpose, beneficiary) in the answering statement; an
  elliptical clause inherits an unambiguous head's ask; overview questions
  stay `supported`. Contact evidence needs a named body; orphan lettered
  items need a modified context.
- Shared Research ownership lexicon (D-075): `specialists/domain_cues.py`,
  read by the Coordinator, the Research and the Teaching specialists; a
  channel clause that asks a subject question keeps its subject specialist.
- Tests: `tests/test_research_ask_shape.py` (new, 202).

### Open (architectural decision required)

A fresh read-only adversarial check (8 dimensions, 1,136 fresh questions)
reported 88 fail-open, 47 misbinding and 14 other cases; after one
structural tightening, a replay of those 178 questions still found 72
`supported` under at least one ranking (some of them arguably correct). The survivors fall into two classes:
permissive classifier paths (telegraphic yes/no without an auxiliary,
existence forms with scaffolding specifics) and relation misbinding of
recognised asks (the answer statement contains the asked words but states
another relation). Lexical grounding checks co-occurrence, not relations.

### Not done on purpose

No change to the Router, retrieval, knowledge base, sources registry,
section map, Synthesis, Verifier, orchestrator, UI or evaluation; no handoff
execution; no model; `PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows:
`tests/test_research_specialist.py` 310, `tests/test_research_corrections.py`
415, `tests/test_research_integration.py` 23, `tests/test_research_ask_shape.py`
202, `tests/test_planf_coordinator.py` 143, `tests/test_teaching_specialist.py`
467, `tests/test_teaching_integration.py` 37, helpers 65,
`tests/test_planf_imports.py` and `tests/test_planf_contracts.py` 91; full
`pytest -q` 2007 passed, 0 failures, 0 errors; `pip check` clean. Scratch
mutation checks: 34 of 34 historical and 71 of 71 Research mutations caught
(15 of them for the root rule). Real-fixture isolation: 30 of 30
excluded-text attacks and 66 of 66 shared-page checks without a leak.

### Next task

Architectural decision on relation-level answer checking (semantic check in
the Verifier, a semantic ask classifier, or a narrower `supported`
contract); Step 4.5B only after that decision.

---

## 2026-09-29 - Research & Innovation Specialist correction pass (Milestone 4, step 4.5A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | corrected locally; not yet committed; architecture inactive; design review required before Step 4.5B |

### Completed

- Step 4.5 findings corrected in the Research & Innovation Specialist, the
  Faculty Onboarding Coordinator and the Teaching & Learning Specialist's
  handoff cues: navigation and contact (M4-F6), the modal "may" and
  deadline end points (M4-F7), value, approver, eligibility and timing
  binding (M4-F8), topic-only support (M4-F9), compound questions (M4-F10),
  reachability (M4-F11), vocabulary (M4-F12), detail false negatives and
  list completeness (M4-F13), the reranker length contract (M4-F14);
  decisions D-063 to D-068. M4-F15 (quote-level section label) deferred;
  M4-F1 kept as a synthetic-only regression; M4-F5 tracked.
- Second round after an adversarial verification of the first: per-sentence
  timing binding with the asked subject and action, end points with a
  time-bearing head, recommenders and adjectives never approvers, clause-level
  approval targets, new specific objects, orphan lettered items, yes/no
  property focus, no prefix word matching, broader navigation, clause
  continuation and greeting handling in the Coordinator, navigation-first and
  multi-domain handoffs in Teaching, foreign subjects of a shared clause,
  redaction of spaced, UNC and forward-slash UNC paths, and dispatch keeping
  non-Coordinator tasks pending; decisions D-069 to D-073.
- Tests: `tests/test_research_corrections.py` (new), redaction cases in
  `tests/test_specialist_helpers.py`, updated Research, Coordinator and
  Teaching tests.

### Re-check result (design review required)

A read-only re-check of the second round (six agents, 726 variants): 40 of
45 confirmed repros fixed and 5 partly reproducing (TIM-03, NAV-05, NAV-07,
SAF-3, BIND-07), with 56 new MAJOR paraphrase variants of the same classes
(timing wordings with no recognised detail, approvals of another action,
yes/no properties beside a slot focus, navigation phrasings such as
"where's", click questions outside the cue list, lexicon gaps between the
Coordinator, Teaching and Research). Each round closed its enumerated
repros, but fresh paraphrases keep producing wrong `supported` results: the
specialist decides completeness by recognising the requested detail
lexically, and an unrecognised ask is `supported` on topic relevance
(fail-open). Choosing a fail-closed default, moving detail-coverage checking
to the Verifier, or accepting a documented residual rate is a design
decision outside this bounded pass.

### Not done on purpose

No change to the Router, retrieval, knowledge base, sources registry,
section map, Synthesis, Verifier, orchestrator, UI or evaluation. No handoff
execution. No source ingested, no index rebuilt, no model downloaded.
`PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2. Real-model validation
remains outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows:
`tests/test_research_specialist.py` 310 passed;
`tests/test_research_corrections.py` 415 passed;
`tests/test_research_integration.py` 23 passed;
`tests/test_planf_coordinator.py` 143 passed;
`tests/test_teaching_specialist.py` 467 passed;
`tests/test_teaching_integration.py` 37 passed; helpers 65 passed;
`tests/test_planf_imports.py` 15 passed; `tests/test_planf_contracts.py`
76 passed; full `pytest -q` 1805 passed, 0 failures; `pip check` clean.
Scratch mutation checks (not committed): 34 of 34 historical and 56 of 56 Research
mutations caught. Real-fixture isolation re-run: 30 of 30 excluded-text
attacks and 66 of 66 shared-page checks without a leak.

### Next task

Design review of the fail-open completeness rule (see the Step 4.5A
report); Step 4.5B independent re-review only after that decision.

---

## 2026-09-28 - Research & Innovation Specialist implementation (Milestone 4, step 4.4)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/remaining-specialists` |
| Starting commit | `bb8432d` |
| Ending state | implemented locally; not yet committed; architecture inactive |

### Completed

- Shared helpers extracted unchanged from the Teaching & Learning
  Specialist: `specialists/scoping.py` (scope compilation with optional
  extra pages, section guard), `specialists/evidence.py` (quote units and
  quality, quote selection with an opt-in lead-in join, per-unit detail
  coverage), `specialists/details.py` (value-shaped numbers, money, timing,
  limits, configurable authority); one `base.redact_error`. Teaching keeps
  its names and behaviour (467 tests, guard oracle and 34 historical
  mutations unchanged).
- Faculty Onboarding Coordinator: bounded research cues (IACUC, IRB,
  royalties, inventors, commercialisation, licensing, TTO, research office,
  board, units, consultancy and others), the WLAM demotion for
  "research-intensive faculty", and the thesis supervision split by detail
  cue.
- `specialists/research.py`: the Research & Innovation Specialist,
  deterministic and extractive, scope 3.5, 5.4, 8.1 to 8.11, chapter 9
  (with the printed 12.3 and the page-175 preamble), 16.1 to 16.4, shared
  6.4, 7.2, 7.7; section guard on the eleven shared pages; research
  vocabulary; detail families including eligibility, duration, required
  documents, funding source, role and publication type; procedural
  safety; handoffs to Teaching, Faculty Services & HR and Institutional
  Navigation, never executed.
- Dispatch: Research is executable with Teaching; Faculty Services & HR and
  Institutional Navigation stay pending. Teaching-phase integration tests
  pin the Teaching-only executable set.
- Documentation: `docs/RESEARCH_SPECIALIST.md`; Coordinator, contracts,
  Teaching and decisions documents (D-058 to D-062).

### Not done on purpose

No change to the Router, retrieval, knowledge base, sources registry,
section map, Synthesis, Verifier, orchestrator, UI or evaluation. No
handoff execution. No source ingested, no index rebuilt, no model
downloaded. `PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2. Real-model
validation remains outstanding. Known limitations recorded for Step 4.6:
the row-join analogue of Milestone 3 R-1, start-of-window timing counted as
a deadline, unbound limit values, and Coordinator gaps for "chief editor"
and "petty cash" questions.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: helpers 49 passed;
`tests/test_teaching_specialist.py` 467 passed;
`tests/test_teaching_integration.py` 37 passed;
`tests/test_planf_coordinator.py` 143 passed;
`tests/test_research_specialist.py` 310 passed;
`tests/test_research_integration.py` 23 passed; the focused multi-agent
set 1259 passed; `tests/test_planf_*.py` 249 passed; full `pytest -q` 1374
passed, 0 failures; `pip check` clean. Scratch mutation checks (not
committed): 34 of 34 historical and 18 of 18 Research mutations caught.

### Next task

Step 4.5: Research isolation and unit QA.

---

## 2026-09-28 - Final bounded correction pass after the major adversarial QA gate (Milestone 3B, step 3.12A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | corrections implemented locally; not yet committed; architecture inactive |

### Completed

Step 3.11 (major adversarial QA) passed with no blocker and no major
finding; its minor and cosmetic findings are corrected or deferred here.

- F-1 compound institutional wording: the Faculty Onboarding Coordinator
  recognises "who manages / supports / maintains ... a support, service,
  system, office or unit" and "where is / where can I find" a named
  service (Registrar, admissions, finance, HR, IT services ...); bare
  "who" and "where" are not cues and "office hours" is never a place. The
  Teaching & Learning Specialist hands the same phrasings off and treats a
  "who manages / supports / is responsible for" clause as fully supported
  only when one evidence unit names the responsible party.
- F-2 grading and exam topic anchors ("grading policy / rules / scheme",
  "exam rules / regulations / policy") that imply their base word; bare
  "policy" and "rules" stay generic; research, HR, parking, travel and
  conference wording carries no concept; "e-learning" is the LMS topic.
- F-3 hyphen-like separators are read as spaces for cue and concept
  matching only (Coordinator and specialist); add/drop and e-learning
  are Coordinator teaching cues; questions and quotes stay verbatim.
- F-4 minimum, maximum and fee details need value-shaped numbers (unit,
  percent, currency or an attached limit phrase); years, section numbers
  and identifiers never count.
- F-5 "should" satisfies a requirement question by design, documented and
  tested; quotes keep the original wording.
- F-6 "What are the <subject> requirements?" is a requirement question;
  ownership is unchanged.
- F-7 lettered list items ("a.", "b.", "A.", "B.") are list items.
- F-8 "48 hours of office hours" satisfies an office-hours count; "48
  hours of training" does not.
- F-9 deferred as cosmetic (row boundary inside a paragraph window needs
  the chunker); F-10 two cheap dispatch checks for a decision altered
  after validation; F-11 documented only.
- Documentation: Coordinator and specialist documents, D-052 to D-057.

### Not done on purpose

No change to Router, retrieval, knowledge base, sources registry, section
map, Synthesis, Verifier, orchestrator, UI, evaluation or production
activation. `PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2; the dispatch
seam stays inactive. No source ingested, no index rebuilt. Real-model
validation remains outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_planf_coordinator.py`
98 passed; `tests/test_teaching_specialist.py` 467 passed;
`tests/test_teaching_integration.py` 37 passed; sources, retrieval scope,
contracts, registry, Coordinator, Teaching and integration together 817
passed; the multi-agent modules 204 passed; full `pytest -q` 947 passed, 0
failures; `pip check` clean. Mutation checks (scratch, not committed):
every correction reverted in isolation makes at least one test fail; the
historical section-guard oracle and mutation set still pass.

### Next task

Step 3.12B: independent freeze re-review of the first specialist pattern.

---

## 2026-09-28 - Coordinator to Teaching & Learning Specialist integration (Milestone 3B, step 3.10)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | integration seam implemented locally; not yet committed; architecture inactive |

### Completed

- `handbook_bot/agents/specialists/dispatch.py`: `dispatch(decision,
  registry)` and `run_question(question, registry)` run the Teaching &
  Learning Specialist on the tasks the Faculty Onboarding Coordinator
  assigns to it and return the decision plus findings (`DispatchResult`
  with per-task `SpecialistRun` records in Coordinator order). Other
  specialists' tasks stay pending and visible; handoff requests are
  preserved, not executed; a raising specialist becomes an `error` finding
  with redacted text; no LLM call, no Synthesis, no Verifier.
- `tests/test_teaching_integration.py`: Teaching-only, single and
  multi-clause focus, Teaching with research, faculty services and
  institutional tasks, non-Teaching questions with a call-counting
  specialist, procedure, not_found, partial, supported, execution error and
  raising specialist, handoff preserved but not executed, Coordinator
  specialist limit, Teaching subtask limit, unregistered specialist,
  no-default-to-Teaching, task identity and order, determinism, and
  architecture inactivity.
- Documentation: Coordinator and specialist documents, package docstring,
  D-051.

### Not done on purpose

No change to the Coordinator, the Teaching specialist, Router, retrieval,
orchestrator, QA, Synthesis, Verifier, contracts, base, registry, knowledge
files or UI. No handoff execution, no Synthesis, no Verifier collaboration.
No new source ingested; `knowledge/sources.json` and
`knowledge/handbook_sections.json` unchanged. `PLAN_F_ENABLED` stays False,
`MAX_LLM_CALLS` 2. Step 3.9F MINOR findings (minimum, maximum and fee
bare-number semantics; "should" versus "must"; "what are the X
requirements"; lettered list items; a quantity false negative) are deferred
to Step 3.11. Real-model validation remains outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_integration.py`
25 passed; `tests/test_teaching_specialist.py` 382 passed;
`tests/test_planf_coordinator.py` 73 passed; sources, scope, contracts,
registry, Coordinator, Teaching and integration together 695 passed;
the multi-agent modules 179 passed; full `pytest -q` 825 passed,
0 failures; `pip check` clean.

### Next task

Step 3.11: major QA of the Coordinator to Teaching path, including the
deferred Step 3.9F MINOR findings and, once the models are available
locally, real-model validation.

---

## 2026-09-28 - Teaching specialist final correction pass after the final independent re-test (Milestone 3B, step 3.9E)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | final corrections implemented locally; not yet committed; awaiting Step 3.9F |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: completeness judged
  within one local evidence unit of one finding, never unioned across
  findings, sentences or list items (F-1); quantity requires a number tied
  to a count noun naming the counted subject, with a separate frequency
  family (F-2); professional, faculty and teaching development, training
  modules and workshops, training for faculty and "new faculty" are anchor
  concepts (F-3); list markers are stripped before the antecedent check
  and sibling items or headings are never antecedents (F-4); a requirement
  family for what must be contained, included, provided or required (F-5);
  filesystem paths redacted from error text (F-6). F-7 unchanged and
  deferred as documented. Still deterministic, extractive, no LLM.
- `tests/test_teaching_specialist.py`: the real page-79 layout regression,
  a same-quote union attack, eight cross-finding union attacks with wrong
  pieces scored highest, same-unit positive controls, number-semantics
  negatives, development and training anchors with unrelated-training
  controls, numbered-pronoun antecedent tests, requirement-family tests and
  path-redaction tests. Twenty-four in-process mutations are each caught.
- `docs/TEACHING_SPECIALIST.md` updated; `docs/DECISIONS.md` D-047 to
  D-050.

### Not done on purpose

No frozen module changed (Router, Coordinator, retrieval, orchestrator,
QA, Synthesis, Verifier, contracts, base, registry, knowledge files, UI,
evaluation). No production wiring: `PLAN_F_ENABLED` stays False,
`MAX_LLM_CALLS` 2. No model download: real-model validation is still
outstanding. F-7 (construction-time invariant) left as designed.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
382 passed; sources, scope, contracts, registry, Coordinator and
Teaching together 670 passed; the multi-agent modules 179
passed; full `pytest -q` 800 passed, 0 failures; `pip check` clean.
Real-text probes with stand-in ranking: no excluded-section quote, no
fragment or antecedent-less quote, the page-79 part-time question `partial`
with the gap named.

### Next task

Step 3.9F: independent re-test of the final corrections, then real-model
validation once the two models are available locally, then Coordinator to
Teaching integration testing (Step 3.10).

---

## 2026-09-28 - Teaching specialist second correction pass after the independent re-test (Milestone 3B, step 3.9C)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | second corrections implemented locally; not yet committed; awaiting the final independent Teaching re-test (step 3.9D) |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: window-independent
  paragraph attribution in `SectionGuard` with a construction-time id
  invariant (F-A, F-H); a completeness gate separating relevance from
  answerability with eleven requested-detail families and faculty
  categories, plain-language missing entries and detail-first candidate
  ordering (F-B); ownership cues made a superset of the Coordinator's
  teaching cues plus natural phrasings (F-C); grade-table lines with
  textual ranges (F-D); bare "attend" demoted to plain vocabulary (F-E);
  antecedent-dependent sentences quoted with their antecedent or not at
  all (F-F); procedural requests `partial` only with evidence about the
  system or the request (F-G). Still deterministic, extractive, no LLM.
- `tests/test_teaching_specialist.py`: overlap-layout fixtures at several
  heading positions with a character-level consistency invariant; a
  ten-case adversarial completeness matrix where the right evidence is
  absent and wrong-but-related evidence ranks highest, plus positive
  controls; ownership, grade-row, attendance, antecedent, procedure and
  invariant tests. Seventeen in-process mutations (including the old
  overlap rule, a removed detail gate, ignored part-time, authority and
  category checks, narrowed ownership, the old grade-row rule, bare attend
  as an anchor, removed antecedent and procedural gates) are each caught.
- `docs/TEACHING_SPECIALIST.md` rewritten; `docs/DECISIONS.md` D-042 to
  D-046 (D-046 defers the one-intent-per-task concern, F-I).

### Not done on purpose

No frozen module changed (Router, Coordinator, retrieval, orchestrator,
QA, Synthesis, Verifier, contracts, registry, knowledge files, UI,
evaluation). No production wiring: `PLAN_F_ENABLED` stays False,
`MAX_LLM_CALLS` 2. No model download: the embedding model and cross-encoder
remain unavailable locally, so real-model validation is still outstanding.
F-I (one intent per task) is deferred, not fixed.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
326 passed; sources, scope, contracts, registry, Coordinator and Teaching
together 614 passed; the multi-agent modules 179 passed; full `pytest -q`
744 passed, 0 failures; `pip check` clean. Real-text probes with stand-in
ranking: no excluded-section quote on any shared page, zero quotable
false allows or rejects against an independent page-text oracle, no leak
at any synthetic overlap layout.

### Next task

Step 3.9D: final independent Teaching re-test, then real-model validation
once the two models are available locally, then Coordinator to Teaching
integration testing (Step 3.10).

---

## 2026-09-28 - Teaching specialist correction pass after the dedicated test gate (Milestone 3B, step 3.9A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | corrections implemented locally; not yet committed; awaiting the Step 3.9 re-test |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: five evidence gates. New
  `SectionGuard` attributes chunks on shared pages to printed sections and
  rejects excluded-section text (F-1); concept vocabulary with aliases,
  anchor-or-two-concepts relevance rule and a generic-term list (F-2, F-6);
  quote quality rules, sentence units with list-item context, table-row
  detection, quote deduplication (F-3); phrase-level handoff cues,
  positive-cue ownership, unowned clauses not searched (F-4); calendar
  term and year context and filtering (F-5); page-level section-label
  caveat in finding metadata (F-7). `FINAL_K` is no longer applied inside
  the specialist (D-038). Still deterministic, extractive, no LLM.
- `tests/test_teaching_specialist.py`: rewritten around synthetic handbook
  pages chunked by the project's chunker and annotated with the real
  section map; a raw-overlap reranker independent of the specialist's
  normalisation and a scripted reranker for adversarial cases (high-scoring
  wrong candidate, lower-ranked relevant evidence, heading chunks, duplicate
  row and row window, excluded-section text on shared pages, generic-word
  distractors, wrong-year calendar line, morphological variants, ambiguous
  cue words). Mutation checks (guard removed, one-word relevance, heading
  filter removed, dedupe removed, "who approves" cue restored, calendar
  year check removed, table-row rule removed, ownership default removed)
  are each caught by at least one test.
- `docs/TEACHING_SPECIALIST.md` rewritten; `docs/DECISIONS.md` D-035 to
  D-041.

### Not done on purpose

No frozen module changed (retrieval, sources, knowledge base, config,
Router, Coordinator, orchestrator, QA, Synthesis, Verifier, contracts, base,
registry, knowledge files, UI, evaluation). No other specialist. No
production wiring: `PLAN_F_ENABLED` stays False, `MAX_LLM_CALLS` 2. No
model download: the embedding model and cross-encoder remain unavailable
locally, so real-model validation is still outstanding.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
236 passed; sources, scope, contracts, registry, Coordinator and Teaching
together 524 passed; the multi-agent modules 179 passed; full `pytest -q`
654 passed, 0 failures; `pip check` clean; a fresh interpreter with sockets
blocked imports the specialist without loading a model.

### Next task

Step 3.9 re-test of the corrected specialist, then real-model validation
once the two models are available locally, then Coordinator to Teaching
integration testing (Step 3.10).

---

## 2026-09-28 - Teaching & Learning Specialist, deterministic version 1 (Milestone 3B, step 3.8)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` |
| Starting commit | `2d137cb` |
| Ending state | implemented locally on the branch; not yet committed; awaiting dedicated testing and review (step 3.9) |

### Completed

- `handbook_bot/agents/specialists/teaching.py`: `TeachingLearningSpecialist`
  (id `teaching`) implementing `Specialist.run(task) -> SpecialistFindings`
  with no LLM call. Scope compiled from the section map for the owned
  sections (1.14; 3.1 to 3.4, 3.7, 3.10 to 3.12; 5.1, 5.2, 5.3, 5.5; 10.6;
  12.7, 12.12 to 12.18; 16.6) into `RetrievalScope(source_ids, page_ranges)`.
  Per-clause scoped retrieval through the shared `gather_candidates`,
  production rerank and gate, extractive findings with exact quotes,
  deterministic status aggregation, handoffs for misrouted or shared
  clauses, procedure-request detection for Blackboard, Banner and MyUOS,
  unapproved-source refusal, redacted error results, dependency injection
  through `TeachingResources` (`from_knowledge_base` available, unused by
  the runtime).
- `tests/test_teaching_specialist.py`: import safety, contract and registry,
  scope compiled from the real map (included and excluded pages, chapter 5
  handled section by section), supported facts, procedure safety, ownership
  boundaries and handoffs, no Coordinator call, boundary-page safety, zero
  evidence, unapproved source, retrieval failure, multi-part tasks,
  deduplication, clause limit, Coordinator compatibility, quote selection.
- `docs/TEACHING_SPECIALIST.md`; `docs/DECISIONS.md` D-030 to D-034; one
  docstring paragraph in `handbook_bot/agents/specialists/__init__.py`.

### Not done on purpose

No orchestrator, QA, retrieval, sources, knowledge base, config, Router,
Coordinator, Synthesis, Verifier, contract, registry, UI or evaluation
change. No other specialist. No production wiring: `PLAN_F_ENABLED` stays
False and `MAX_LLM_CALLS` 2. No date-extractor hook (the calendar row is
retrieved as evidence directly). No new source.

### Source reality

The handbook remains the only indexed document. Blackboard, Banner and
MyUOS procedures are not in it; the specialist returns partial or not found
for them and never generates instructions.

### Tests performed

All with `.\.venv\Scripts\python.exe` on Windows: `tests/test_teaching_specialist.py`
105 passed; the multi-agent, source and scope modules together 303 passed;
full `pytest -q` 523 passed, 0 failures; `pip check` clean; a fresh
interpreter imports the specialist, registry, Coordinator, retrieval,
orchestrator, Verifier and UI modules without loading a model, with
`PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2 and `CACHE_VERSION` v12.

### Next task

Step 3.9: dedicated adversarial testing and independent review of the
Teaching specialist, then the QA gate that decides whether the specialist
pattern is frozen for the remaining specialists.

---

## 2026-09-28 - Source metadata and scoped retrieval foundation (Milestone 3A)

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/teaching-specialist` (created from `feature/faculty-coordinator`) |
| Starting commit | `1a583f8` |
| Ending state | source-scoped retrieval foundation completed, independently reviewed, corrected and committed on `feature/teaching-specialist`; see Git history for the commit |

### Completed

- `knowledge/sources.json` and `handbook_bot/sources.py`: source registry
  (`SourceRecord`, `SourceRegistry`, `load_source_registry`,
  `find_source_for_path`, `validate_source_registry`, `unregistered_source`).
  One record: `uos_faculty_handbook_2025_26`, repository-relative path.
- `knowledge/handbook_sections.json`: 206-record page-range map of the
  handbook (4 front-matter parts, 16 chapters, 186 level-2 sections),
  derived from the document's table of contents on pages 4 to 13 and
  checked against every chapter and section heading on its listed page.
  `SectionMap.resolve(page)` is deterministic; no page is unmapped; document
  numbering quirks (5.2, 7.8, "12.3" in chapter 9) recorded with notes.
- `annotate_metadata`: every chunk gains `source_id`, `source_title`,
  `source_type`, `source_version`, `chapter`, `chapter_title`, `section_no`,
  `section_title`, `page_section_nos`; old keys and chunk text unchanged
  (13,688 chunks before and after). Called from `build_knowledge_base`;
  an unlisted document gets an `unregistered_...` identity and no sections.
- `handbook_bot/retrieval.py`: optional `RetrievalScope` on
  `gather_candidates` (source ids, chapters, section numbers, section
  prefixes, page ranges). Dense search restricted at the index level with a
  FAISS id selector, lexical search restricted to allowed chunks, empty
  scope returns nothing, `scope=None` unchanged.
- `CACHE_VERSION` v11 to v12 (cache rebuilds once per machine).
- Tests: `tests/test_sources.py`, `tests/test_retrieval_scope.py`.
- Docs: `docs/SOURCE_INVENTORY.md`, `docs/DECISIONS.md` D-021 to D-025.

### Correction pass (same day, after independent review)

- F-1: section and prefix constraints are chapter-consistent; the section
  printed as 12.3 on page 180 (chapter 9) is no longer admitted by
  `section_prefixes={"12"}` or `section_numbers={"12.3"}`; the metadata
  still records it as printed.
- F-3: `find_source_for_path` no longer falls back to file-name matching;
  a same-named file elsewhere is unregistered. A section map is applied
  only when the loaded page count matches its `page_count`; otherwise the
  source identity is kept and section labels are left null with a warning.
- F-4: the scoped dense fallback requires an inner-product index and no
  longer catches `RuntimeError`.
- F-2, F-5, F-7: page-granular section scopes, non-unique printed section
  numbers, strict identity and the single-document runtime are documented
  in `docs/SOURCE_INVENTORY.md`; decisions D-026 to D-029.
- F-8: tests added for all of the above, a global scope invariant across
  scope types, multi-seed brute-force checks, L2 rejection, error
  propagation, empty corpus and single-chunk cases.
- Deferred: F-6 (authority and status vocabulary), content fingerprinting,
  text-level section segmentation, a unique section-record identifier,
  multi-document ingestion.

### Not done on purpose

No Teaching specialist, no other specialist, no Coordinator, orchestrator,
Synthesis, Verifier, UI or evaluation change; `EvidenceResult.doc_id` not
populated; no new source added; no web access. The multi-agent runtime
stays inactive (`PLAN_F_ENABLED` False, `MAX_LLM_CALLS` 2).

### Source reality

The handbook is the only indexed document. It references Blackboard and
Banner at policy level only; no standalone procedural guide for either
system exists locally. Detailed Blackboard or Banner workflow questions must
return not found or partial until verified guides are added.

### Tests performed

See the phase report: targeted tests, all multi-agent tests, full
`pytest -q`, `pip check`, fresh-interpreter imports and a real-handbook
metadata probe, all with `.\.venv\Scripts\python.exe` on Windows.

### Next task

The Teaching specialist on the
handbook's teaching sections with an explicit `RetrievalScope`, returning
findings whose `evidence_quote` is verbatim chunk text and whose
`source_id` and `page` come from chunk metadata.

---

## 2026-09-24 - Plan F Coordinator foundation

| Field | Value |
|---|---|
| Developer | AdxbA9 (project account) |
| Branch | `feature/plan-f-coordinator` (created from `feature/plan-f-core`) |
| Starting commit | `db8f63e` |
| Ending state | Milestone 2 Coordinator foundation completed, independently reviewed and committed on `feature/plan-f-coordinator`; see Git history for the commit |

### Completed

- `handbook_bot/agents/coordinator.py`: `coordinate(question) ->
  CoordinatorDecision` and `analyze(question)`. Deterministic-first on top of
  the existing Router (arbitration off): domain scoring from cue lexicons,
  level (department, college, university), systems (blackboard, banner,
  myuos), complexity (greeting, unknown, simple, multipart, cross_domain,
  journey), specialist selection capped by the Plan F budgets, one task per
  specialist carrying the full question plus `context["focus_clauses"]`,
  `requires_synthesis` by rule. Contact questions are conditional (channel
  lookup: Institutional alone; contact request about a subject: subject
  specialist plus Institutional). Bare Blackboard or Banner questions go to
  Teaching; bare MyUOS selects nobody. Confidence is a heuristic ordinal.
  Uniform metadata shape for every decision. No LLM call; arbitration
  deferred (D-PLANF-018).
- Review corrections applied (F-1 MyUOS owner removed, F-2 conditional
  contact rule, F-3 full question in every task, F-4 uniform metadata,
  F-5 journey no longer forces Synthesis, F-8 wording).
- `tests/test_planf_coordinator.py`: the A to Z behaviour list plus the
  review cases (MyUOS with and without context, contact with a subject,
  task contextual completeness, metadata shape, journey and Synthesis).
- `docs/COORDINATOR.md` (rules as implemented), `docs/DECISIONS.md`
  D-PLANF-015 to 020, one docstring line in `handbook_bot/agents/__init__.py`.

### Not done on purpose (later phases)

No specialist implementation, no retrieval, Synthesis, Verifier or `QAResult`
change, no handoff execution, no orchestration, no UI or evaluation change.
The Coordinator is not imported by the production pipeline; `PLAN_F_ENABLED`
stays False; the Milestone 1 pipeline is unchanged.

### Tests performed

See the phase report: targeted Coordinator tests, all Plan F tests, full
`pytest -q`, `pip check` and fresh-interpreter imports, all with
`.\.venv\Scripts\python.exe` on Windows.

### Next task

Phase 2, knowledge layer (source registry, section map from the handbook
table of contents, chunk metadata, cache version bump, scope filter and
boosts), then the Teaching Agent once the Blackboard and Banner guides are
obtained and reviewed.

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
