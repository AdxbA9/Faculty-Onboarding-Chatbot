# Post-freeze phase PF-6: final English technical freeze

Date: 2026-11-01. Branch `feature/post-freeze-knowledge-expansion`, starting commit `720e9aa` (PF-5C). Freeze identifier:
**`ENGLISH_CORE_FREEZE_PF6`** (the commit that adds this folder). Decision record: D-116. Frozen control: `runtime-freeze-i9`
(D-108) with D-109 to D-115. PF-6 is a verification phase: no feature, coverage, wording, document, source or architecture
change; code may change only for a critical defect, and none was found, so no runtime, data, registry or test file changed.

## Outcome in one paragraph

The English system was verified end to end on the frozen code and is frozen. Corpus `c8987b6079b11f39` (8 enabled sources, 309
pages, 15,000 chunks) is reproduced exactly, including a cold rebuild of every embedding that is bit-identical to the cache.
Every safety count is zero: 0 wrong served answers in every set, 0 false acceptances, 0 wrong-source acceptances, 0 conflict
leaks, 0 held-source citations, 0 policy answers from a guide, 0 citation or page mismatches, 0 determinism differences, 0 SHADOW
mismatches, 0 exceptions, 0 model calls in Plan F, 0 crashes in the real application. Every measured set is identical to PF-5C.
Coverage was not a criterion and did not change: guide scopes stay OFF by default, and the nine known limitations below are
documented, not fixed.

## Results (sets reported separately; coverage is not a freeze criterion)

| Set | Questions | Plan F served | correct / limited / wrong | precision | against PF-5C |
|---|---|---|---|---|---|
| frozen 192, production default (scopes off) | 192 | 55 (the I-9 55) | 55 / 0 / 0 | 100% | every record identical |
| frozen 192, guide scopes on | 192 | 55 (the I-9 55) | 55 / 0 / 0 | 100% | every record identical; registry never selected |
| P0 core guide (scopes on) | 37 | 21 | 19 / 2 / 0 | 100% | byte-identical |
| P0 all guide answers | 100 | 35 | 32 / 3 / 0 | 100% | byte-identical |
| P1 | 41 | 7 | 5 / 2 / 0 | 100% | byte-identical |
| PF-4 held-out (diagnostic) | 50 | 1 | 1 / 0 / 0 | 100% | identical |
| PF-5B fresh (diagnostic) | 56 | 0 | 0 / 0 / 0 | n/a | identical |
| PF-5C fresh (diagnostic) | 60 | 0 | 0 / 0 / 0 | n/a | identical |
| five sets, production default (scopes off) | 307 | 4 handbook answers | 4 / 0 / 0 | 100% | identical |

## Checks

| Check | Result | Artifact |
|---|---|---|
| source registry | 9 records, 8 enabled, every enabled SHA-256 matching, appraisal disabled, no unregistered guide file, defaults off: PASS | `source_registry_audit.json` |
| corpus integrity | key `c8987b6079b11f39`, 8 / 309 / 15,000, deterministic, 0 held-source chunks, 46 conflict-stamped chunks: PASS | `corpus_integrity.json` |
| cache and rebuild | cold rebuild of 15,000 embeddings in 444 s, bit-identical to the cache (max abs diff 0.0); reuse only on identical inputs | `corpus_integrity.json` |
| wrong source | 88 cases, 0 wrong-source acceptances, 0 violations, 11 served; inference adversarial 30 of 30 refused | `wrong_source_final.json` |
| conflicts | 7 quota and 4 journal questions, 0 leaks (scopes on and off); upload steps and two Communication procedures still served | `conflict_final.json` |
| policy/system boundary, held sources | 16 of 16 (no policy, authority or organizational answer from a guide); 6 of 6 (no retrieval, citation, chip or answer from a held source) | `boundary_heldsource_final.json` |
| procedure completeness | 31 of 31 served procedures and sections complete, in printed order | `procedure_final.json` |
| citation integrity | 58 answers, 204 items, every enabled source: quote on its page, page and source the chunk's own, registry titles, recomputed headings, re-verified guide answers, approved specialist, chips: 0 problems | `citation_review.json` |
| cross-document | 14 questions, 6 served, every clause grounded in its own source, 0 wrong | `cross_document_final.json` |
| determinism | 3 complete repeats, 900 records, 0 differences, DIRECT equal to ACTIVE, 0 exceptions | `determinism.json` |
| SHADOW | 307 of 307 and 192 of 192 (twice) identical to the direct legacy call | `shadow_final.json` |
| ACTIVE | production default and scopes on: 0 exceptions, 0 model calls; scopes off versus on 48 differences, the same as PF-5C | `active_final.json` |
| LEGACY | fresh process: 20 of 20 minimal diagnostics, no Plan F or guide module imported, no runtime cache | `legacy_isolation.json` |
| application smoke | 38 turns (ACTIVE defaults 23, scopes on 12, LEGACY 3): 10 served, all correct, 28 expected fallbacks, 0 crashes, 0 internal ids, developer details only in Developer mode | `app_smoke_final.json` |
| performance | idle, 191 questions: 0 to 4.3% from PF-5C at the median; no meaningful regression | `performance_final.json` |
| Plan F model calls | 0 everywhere | all |
| tests | targeted gate 2,827 passed; full suite 7,279 passed, 0 failed, 0 skipped, 0 xfail; `pip check` clean | `freeze_manifest.json` |
| security | 0 flags in the PF-6 files; every full-tree flag classified (placeholders, test fixtures, quoted policy text, archived prototype output) | `security_final.json` |

## Known English limitations (non-critical)

| # | Limitation | Consequence |
|---|---|---|
| 1 | `PLAN_F_GUIDE_SCOPES` is OFF by default | production Plan F answers from the handbook only; guide questions take the legacy path |
| 2 | conservative implicit-system coverage (D-115: procedure asks only, many refusals) | PF-5C fresh 0 of 25 implicit answerable |
| 3 | conversational and multi-clause questions fall back whole (I-5 all-or-nothing aggregation) | 19 of 31 answerable PF-5C fresh questions stop here |
| 4 | Assignments & Tests guide held (25 pages collide with the frozen page-keyed indexes, D-113) | not registered; never retrieved or cited |
| 5 | Grade Categories guide held (interleaved English and Arabic pages, D-113) | not registered; never retrieved or cited |
| 6 | Performance Appraisal Phase 2 guide held (Internal Use) | registered, disabled; 0 chunks |
| 7 | iThenticate yearly quota conflict (30 versus 15) unresolved | every quota claim blocked; the upload steps still served |
| 8 | Arabic not implemented | English only |
| 9 | voice not implemented | text only |

Also observed and documented, not fixed: "How do I log in to iThenticate?" falls back (the procedure is printed as "How to
access", so "log in" is not stated; not a conflict over-block); without a model key a legacy fallback shows the key notice or
"I do not have this information." with retrieval chips.

## Files

| File | Content |
|---|---|
| `freeze_manifest.json`, `final_metrics.json` | the freeze identifier, parent commit, timestamp, corpus identity, per-source hashes and counts, switch defaults, tests; every final number |
| `source_registry_audit.json`, `corpus_integrity.json` | registry audit; corpus determinism, held-source absence, conflict stamps, cache check, cold rebuild |
| `frozen_192_final.json` | the 192-question master set with scopes off and on against I-9 and PF-5C |
| `p0_final.json`, `p1_final.json`, `held_out_diagnostics.json`, `cross_document_final.json` | the five sets (answers compared by SHA-256 with PF-5C) |
| `wrong_source_final.json`, `conflict_final.json`, `boundary_heldsource_final.json` | wrong-source and inference adversarial; conflicts; policy/system boundary and held sources |
| `citation_review.json`, `procedure_final.json` | every served guide answer and a frozen-192 sample re-verified item by item; procedure completeness |
| `determinism.json`, `shadow_final.json`, `active_final.json`, `legacy_isolation.json` | repeats, SHADOW, ACTIVE (production default and scopes on), LEGACY fresh process |
| `app_smoke_final.json` | 38 turns on the real NiceGUI application with the manual evidence review |
| `performance_final.json`, `security_final.json` | idle performance against PF-5C; the final security scan |
| `audit.py`, `run_eval.py`, `final_checks.py`, `citation_check.py`, `derive.py`, `assemble.py`, `manifest.py` | the PF-6 scripts (the PF-5C `wrong_source.py`, `inference_adversarial.py`, `legacy_isolation.py`, `make_review.py`, `analyze.py` and the I-8 `master_eval.py`, `analyze.py`, PF-2 `baseline_compare.py` are reused unchanged) |

## Reproduce

```
python -X utf8 qa/post_freeze_pf6/audit.py . <out> --cold <empty dir> --pf5c <PF-5C results.json>
python -X utf8 qa/post_freeze_pf6/run_eval.py . <out>
python -X utf8 qa/plan_f_i8/master_eval.py . <out>   (192 questions; again with PLAN_F_GUIDE_SCOPES=1 in the environment)
python -X utf8 qa/post_freeze_pf6/final_checks.py . <out>
python -X utf8 qa/post_freeze_pf6/citation_check.py . <results.json> <i8_results.json> <out>
python -X utf8 qa/post_freeze_pf6/derive.py <results.json> <PF-5C results.json> <review table> <metrics.json> <out>
python -X utf8 qa/post_freeze_pf6/assemble.py <run dir> <PF-5C run dir> <out>
python -X utf8 qa/post_freeze_pf6/manifest.py . <timestamp> <gate passed> <full passed> <pip>
```

## Next phase

ENGLISH CORE = FROZEN. Recommended order (none started): independent manual acceptance testing; Arabic; Arabic regression
against this freeze; voice input and output; final bilingual and security regression; optional guided onboarding if time remains.
