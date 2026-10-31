# Post-freeze phase PF-3B: controlled specialist source reopening and guide activation

Date: 2026-10-30/31. Branch `feature/post-freeze-knowledge-expansion`, on top of the PF-2 commit `fd14af8`.
Frozen control: `runtime-freeze-i9` (commit `44ada15`, D-108). Decision record of this phase: D-110.

## Outcome in one paragraph

The specialist source-acceptance boundary was reopened under control and is safe: approved guides now reach the
Teaching, Research and Institutional Navigation specialists, every safety property holds, and the frozen 192-question set
is unchanged. **Plan F still serves 0 of 37 guide questions.** Two further frozen rules, which the brief forbids
redesigning, refuse every guide answer: the specialists' ask gate (the answering sentence must state each asked
specific, including the system name and the asking actor, which second-person manuals never do) and their procedure rule
("a procedure request is never supported"); a layer probe shows the adapter's I-1 subject guard refuses such answers
too. The manual review also shows why relaxing them alone would be unsafe: of the 24 guide questions on which the
reopened specialists returned guide evidence, the top guide quote was on target for 5, partial for 5 and off target for
14 (mostly the gradebook guide's page-1 introduction). PF-3B is therefore **incomplete on its coverage criterion** and
the change is left uncommitted for the project owner's decision; `PLAN_F_GUIDE_SCOPES` stays off by default.

## What changed

| Area | Change |
|---|---|
| `teaching.py`, `research.py`, `institutional.py` | constructor and `from_knowledge_base` take `approved_sources`; the source check reads `not in self.approved_sources` instead of `!= APPROVED_SOURCE_ID` (Institutional: statements only; the contact directory stays handbook-only). With no set the I-9 behaviour is exact. |
| `scoping.py` | `approved_source_set` (no wildcard, handbook always included), `approve_sources` (guides added as `also_source_ids` beside the unchanged handbook page ranges); `SectionGuard` source-aware (a guide chunk is never attributed to a handbook page or section) |
| `domain_cues.py` | the iThenticate cue in the shared research ownership list (routes iThenticate questions to Research; SafeAssign and Turnitin course questions stay with Teaching) |
| `source_scopes.py` | the D-110 source policy (code allow-list intersected with the registry), the guide system vocabulary, the source-authority, policy-boundary and procedure-completeness checks |
| `plan_f_runtime.py` | a guide-enabled registry built beside the I-9 one when `PLAN_F_GUIDE_SCOPES` is on; a question uses it only when it names a guide's system; `PLAN_F_REJECTED_BY_SOURCE_AUTHORITY`; the guards on every guide-backed finding (pass-through, provision slices, aggregation parts) |
| `ui/pipeline.py`, `ui/presentation.py` | diagnostics `guide_scope_enabled`, `guide_scope_selected`, `guide_sources_named`, `plan_f_guide_served` (ids and flags only); Developer panel row "Source scope" |
| Faculty Services & HR | untouched (handbook only; the appraisal guide is on hold) |

Approved source sets (D-110): Teaching = handbook, Blackboard short guide, gradebook guide, grade transfer guide;
Research = handbook, iThenticate; Faculty Services & HR = handbook; Institutional Navigation = handbook, Blackboard short
guide, grade transfer guide. Unknown, disabled, held and unlisted sources are never approved (tested).

## Results

### Frozen 192-question set with guide scopes ON (`baseline_comparison.json`)

| | I-9 | PF-2 | PF-3B |
|---|---|---|---|
| served / fallback / coverage | 55 / 137 / 28.6% | 55 / 137 / 28.6% | 55 / 137 / 28.6% |
| correct / limited / wrong | 55 / 0 / 0 | 55 / 0 / 0 | 55 / 0 / 0 |
| false acceptances / false fallbacks | 0 / 11 | 0 / 11 | 0 / 11 |
| questions changed versus I-9 | - | 0 | 0 (ACTIVE and SHADOW) |
| SHADOW identical to legacy | 192 | 192 | 192 |
| Plan F model calls | 0 | 0 | 0 |

No master question names a guide system, so all 192 ran on the I-9 specialists; contamination is tested by the 22 PF-3B
questions below.

### Guide evaluation (`results.json`, `review.json`, `metrics.json`; 48 PF-2 + 22 PF-3B questions)

| Metric | PF-2 | PF-3B |
|---|---|---|
| guide questions | 37 | 37 |
| Plan F served / fallback | 0 / 37 | 0 / 37 |
| coverage | 0% | 0% |
| correct / limited / wrong | 0 / 0 / 0 | 0 / 0 / 0 |
| precision | n/a | n/a |
| guide questions on which the specialists returned guide evidence | 0 (blocked at retrieval) | 24 |

Per source (served / questions): Blackboard short guide 0/13, gradebook 0/10, grade transfer 0/6, iThenticate 0/8.
Per specialist (guide questions routed / served): Teaching 28/0, Research 6/0, Institutional Navigation 4/0.

What stopped the 37 guide questions: the specialists' ask gate (asked system name or actor not in the answering
sentence) 20, the procedure rule 6, an unrecognised "how" ask in Research 3, Institutional Navigation without an
approved source or statement for the ask 3, a requested detail not stated 1, no quotable guide evidence 1, the
aggregator (a clause unsupported) 2, routing ambiguity 1.

Layer probe (`layer_probe.json`): when a specialist is stubbed to return one supported guide finding, the adapter's
frozen I-1 subject guard refuses it ("entity 'Blackboard Ultra' is not in the quote", "the quote does not cover the
clause's content words"); the conflict guard refuses the quota; the procedure guard refuses one step as a procedure.

Safety groups (all not served): conflict variants 5 (direct, paraphrase, yes/no with 30, yes/no with 15, multipart),
handbook contamination 5, guide contamination 3, cross-document 5, wrong-source collisions 10 (approval, submission,
review, login, grade, originality, support), held appraisal 1. Known HR and annual-leave regressions not served.
SHADOW identical to the direct legacy call 70 of 70. Guide scopes off versus on: 0 differences. Determinism: 128 repeat
records, 0 differences. Exceptions 0. Plan F model calls 0. Real-model wrong-source retrieval (`wrong_source_retrieval.json`):
16 of 16 expected document first and expected page in the top five, unchanged.

### Application smoke (real chat page, guide scopes on)

ACTIVE, 12 turns: Blackboard navigation, gradebook, grade transfer workflow, grade transfer approval, non-conflicted
iThenticate, disputed iThenticate quota, handbook teaching load, known HR, annual leave, cross-document, wrong-source
Banner login, held appraisal. Every guide turn falls back to the legacy path with the guide's title and page chips and the
Developer row "Source scope: handbook + guides: ..."; the teaching load is served by Plan F (page 91, "handbook only (no
guide system named)"); HR, annual leave and the appraisal question are not served. SHADOW, 3 turns: legacy visible, Plan F
in shadow with the same source scope. LEGACY: fresh-process isolation check, 30 minimal records, no Plan F import.

### Performance

| | value |
|---|---|
| guide-enabled registry build | +95 to +98 ms beside the 680-740 ms I-9 registry |
| cold first ACTIVE request | 2.76 s (PF-2 guide run 2.50 s) |
| knowledge base load (cached) | 23.1 s, unchanged corpus |
| master set ACTIVE p50 / p95 | 1,166 / 2,391 ms (I-9 1,252 / 2,590) |
| guide set Plan F attempt p50 / p95, scopes on versus off | 671 / 1,439 ms versus 611 / 1,161 ms |

### Tests

Targeted gate (I-8 list, multi-source, retrieval scope, Teaching specialist, Faculty Services scope): 2,164 passed, 0 failed,
0 skipped. Full suite: **6,770 passed, 0 failed, 0 skipped** (6,730 at the start). `pip check`: no broken requirements.
Security scan of the changed and new files: 0 problems. Frozen files: Faculty Services & HR, claim model and navigation
cues identical to `44ada15`; Teaching, Research and Institutional changed only at the D-110 boundary.

## Recommendation: PF-3C

Coverage stays at zero for structural reasons, not because of a defect in the reopening. Before any guide answer can be
served, a separate decision record (proposed D-111) must decide three things together:
1. **Source-context binding.** A guide's declared system may bind the system named in the question, and the second-person
   "you" may bind the faculty actor. This affects the specialists' ask gate and the I-1 subject guard.
2. **Procedure-aware evidence selection for guides.** Prefer procedure chunks and on-topic paragraphs over generic
   introductions. 14 of 24 top guide quotes were off target.
3. **Procedure answers.** Allow a complete procedure, aggregated from verified steps, to be served.

Each part needs its own evaluation with every served answer reviewed. Expanding to P1/P2 guides (PF-3A/PF-4) is not
recommended until then.

## Files

| File | Purpose |
|---|---|
| `pf3b_questions.jsonl` | 22 new labelled questions: conflict variants, contamination both ways, cross-document, wrong-source collisions |
| `run_eval.py`, `results.json` | the real-KB run through `ui.pipeline`: ACTIVE and SHADOW with guide scopes on, a DIRECT pass with the stop layer and specialist reports, ACTIVE with scopes off, LEGACY, repeats |
| `make_review.py`, `review.json` | the manual review, including the evidence assessment of every returned guide quote |
| `analyze.py`, `metrics.json` | guide, per-source, per-specialist, stop-layer, safety-group, SHADOW, control, determinism, latency and model-call metrics |
| `baseline_comparison.json` | the 192-question rerun with guide scopes on against the frozen I-9 run |
| `layer_probe.py`, `layer_probe.json` | which adapter layer decides a stubbed supported guide finding |
| `wrong_source_retrieval.json` | the PF-2 real-model wrong-source retrieval check, rerun |

Reproduce:

    python -X utf8 qa/post_freeze_pf3b/run_eval.py . <out dir>
    python -X utf8 qa/post_freeze_pf3b/make_review.py <out dir>/results.json qa/post_freeze_pf3b/review.json
    python -X utf8 qa/post_freeze_pf3b/analyze.py <out dir>/results.json qa/post_freeze_pf3b/review.json <metrics.json>
    PLAN_F_GUIDE_SCOPES=1 python -X utf8 qa/plan_f_i8/master_eval.py . <rerun dir>
    python -X utf8 qa/post_freeze_pf2/baseline_compare.py qa/plan_f_i8/results_2026-10-30.json <rerun dir>/i8_results.json <rerun dir>/metrics.json <out json>
