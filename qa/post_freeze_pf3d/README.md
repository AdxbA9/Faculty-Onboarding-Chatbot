# Post-freeze phase PF-3D: deterministic guide structure recovery and coverage refinement

Date: 2026-10-31. Branch `feature/post-freeze-knowledge-expansion`, starting commit `b1d4bd1` (PF-3C). Frozen control:
`runtime-freeze-i9` (commit `44ada15`, D-108). Decision record: D-112 (this phase). No document was added: the four
approved guides only; the Performance Appraisal guide stays on INGESTION HOLD.

## Outcome in one paragraph

Plan F now reads the printed structure of the guides (glyph bullets, headings printed as colon rows or short title lines,
lead-ins split across rows, sections continued on the next page) and serves a recovered section or list only when it is
provably complete. With `PLAN_F_GUIDE_SCOPES` on it serves **19 of the 37 core guide questions (51.4%; PF-3C 10), all
19 correct** on hand review, with 0 wrong. Of the 10 structural false fallbacks measured before any change, 9 are
recovered; the tenth (an option inside a capability list) is refused correctly. Over all 100 evaluation questions 33
guide answers are served (32 correct, 1 acceptable but limited, 0 wrong), and every PF-3C answer is unchanged. Every
conflict, contamination, wrong-source, appraisal, authority, known-HR and annual-leave case falls back; incomplete
procedures and unordered lists are refused by construction and by test. The frozen 192-question set is identical to I-9,
SHADOW is unchanged (100 of 100), and Plan F makes 0 model calls. `PLAN_F_GUIDE_SCOPES` stays off by default.

## Measured first: why 27 core guide questions fell back in PF-3C

`fallback_classification.json` was written from the PF-3C run and the guide pages before any PF-3D code changed (its
`after_pf3d` block was added after the evaluation):

| class | questions | PF-3D |
|---|---|---|
| legitimate (no answer in the guide, specialist not approved for it, policy/system boundary, the quota conflict, ambiguous) | 8 | all still fall back |
| STRUCTURAL false fallback (the answer is printed as one complete unit the D-111 reading could not see) | 10 | 9 recovered, 1 remaining (G20) |
| other false fallback (an asked word the guide does not print, evidence below the reranker gate) | 9 | unchanged; not a structure problem |

Structural causes: steps printed with glyph bullets and no numbers (6), mixed or cut lists (3: a lettered section label
read as a step, a numbered title read as a step, an option inside a capability list), a heading printed as a short title
line (1).

## What changed

| Area | Change |
|---|---|
| `handbook_bot/guide_structure.py` (new) | D-112 reading: printed blocks (glyph, dash, "o", numbered and lettered items, headings, continuation rows), sections (heading to the next heading of the same or higher rank, continued across a page only when the next page opens under no heading), lists with their lead-in, list types, exact quote mapping of every block to an indexed chunk of its own page |
| `handbook_bot/guide_grounding.py` | the guide index builds the structures; `structure_problem` (D-112 completeness and heading rules on top of every D-111 rule); answer form `section`; selection after D-111 procedures; adapter verification of a section answer against the index; the missing-last-step rule (also for D-111 procedures); `citation_heading` |
| `handbook_bot/agents/specialists/guide_evidence.py` | retrieval ranks of the recovered structures passed to the selection |
| `handbook_bot/plan_f_runtime.py` | `guide_citation_items`: after every check, a verified guide item is cited under the printed heading of its verified procedure or structure (a statement: its page alone) instead of the chunk's carried page hint |
| `tests/test_guide_structure.py` (new) | 68 tests: reading, headings, unnumbered and bulleted procedures, unordered lists, mixed lists, continuations, cross-page sections, boundaries, list types, adversarial incomplete and forged answers, the real corpus, the adapter end to end, citation labels |
| frozen specialists, handbook path, D-110, D-111 binding/audience/allow-list/policy boundary/conflict guard | unchanged |

## Results

### Guide evaluation (`guide_results.json`, `newly_served_review.json`, `metrics.json`; 48 PF-2 + 22 PF-3B + 30 PF-3C questions)

| Core guide questions (G01-G37) | PF-3C | PF-3D |
|---|---|---|
| served by Plan F / fallback | 10 / 27 | 19 / 18 |
| coverage | 27.0% | 51.4% |
| correct / limited / wrong | 10 / 0 / 0 | 19 / 0 / 0 |
| precision | 100% | 100% |
| answer forms | 6 procedures, 1 list item, 3 statements | the same 10 plus 9 recovered sections or lists |

| Source (core questions) | PF-3C | PF-3D |
|---|---|---|
| Blackboard Ultra Faculty Short Guide | 4 of 13 | 8 of 13 |
| Blackboard Ultra Gradebook guide | 1 of 10 | 6 of 10 |
| Grade Transfer Dashboard Guide | 3 of 6 | 3 of 6 |
| iThenticate Guidelines | 2 of 8 | 2 of 8 (the yearly quota stays blocked) |

Newly served, each reviewed against the printed page (`newly_served_review.json`): G08 keyboard reorder (lead-in and
steps 1-4; "D. Edit and delete content:" is the next section, not a fifth step), G09 learning module, G10 announcement
(eight bulleted units), G11 create an assignment (pages 10 and 11; page 11 opens under no heading and ends the guide),
G15 the two gradebook views (lead-in recovered across two rows; labels 1., a), b), 2. kept), G17 automatic zeros, G18
gradebook download ("1. Download Worflow" to "2. Upload a file ..."), G19 accommodations, G22 child-course filter, the
cross-document P14 (handbook p79 office hours plus the assignment section, each verified on its own) and C29.

C29 ("How do I reorder content with my keyboard in Blackboard?") was labelled an incomplete procedure in PF-3C because
the PF-2 reading took the section label "D." for a fifth step. The printed page shows sections B, C and D, with the
four keyboard steps under "C. Reorder content:"; the served answer is that complete list. C30 still falls back (the
section does not print the asked word "grades").

Remaining core fallbacks (18): the 8 legitimate ones, G20 (structural, refused: an attendance option inside a "you can"
list is not a procedure) and the 9 non-structural ones (G02, G04, G06, G07, G21, G32, G34, G36, G37).

All served guide answers over the 100 questions: 33 (PF-3C 22), 32 CORRECT, 1 ACCEPTABLE BUT LIMITED (C04, unchanged
from PF-3C), 0 WRONG. Guide scopes off versus on differ in exactly these 33 reviewed answers. SHADOW identical to the
direct legacy call 100 of 100. Determinism: 186 repeat records, 0 differences. Exceptions 0. Plan F model calls 0.

### Evidence selection (`evidence_review.json`)

The guide path selects evidence for 21 of the 37 core questions (PF-3C 12): all 21 on target as a set (every unit on
the target guide, a unit on a target page, a target phrase in the units), 0 partial, 0 off target; the 12 PF-3C
selections keep their categories. By the PF-3C first-unit measure 14 are on target and 7 "partial": for those 7 the first
unit is the section's printed heading row, followed by the target passage. Nothing is selected for the other 16.

### Safety and adversarial (`adversarial_results.json`)

Evaluation groups, none served: conflict 6, handbook contamination 6, wrong-source collisions 18, appraisal, authority,
known HR, annual leave, routing, ambiguous. Synthetic and real-corpus adversarial tests, all refused: first, middle and
last step missing; page continuation not indexed; a heading belonging to another procedure; two procedures mixed;
a dropped, reordered or repeated unit; a forged section answer through the adapter; a block no chunk holds verbatim; a
registered conflict inside a structure; unordered lists (never rendered as steps, bullets never numbered); lettered
callouts with titles; options lists for an enumeration that names another set. Real-model wrong-source retrieval
(`wrong_source_retrieval.json`): 16 of 16 unchanged.

A missing last step was found during PF-3D's own adversarial work: a final step that announces content ("2. Make your
selections:") with nothing after it was served by both the D-111 and the D-112 path. Such a step now refuses the answer
unless it points at a figure ("... after the approval from Chair and Dean as below:", which keeps the reviewed PF-3C
grade transfer answer).

### Frozen baseline (`baseline_comparison.json`)

The 192-question master set with guide scopes on, final PF-3D code: identical to I-9 and to the PF-3C rerun question
by question (55 served, 55 correct, 0 wrong, 0 false acceptances, 11 false fallbacks, 0 changed questions in ACTIVE and
SHADOW, SHADOW 192 of 192 identical, 0 model calls). No master question selects the guide registry, so the D-111 and
D-112 code never runs on it.

### Application (`app_smoke.json`, `legacy_isolation.json`)

ACTIVE, 13 turns on the real chat page: the recovered announcement (section, 9 parts, page 9) and automatic-zeros
(section, page 8) answers, grade transfer, iThenticate access, the quota fallback, handbook teaching load, known HR,
annual leave, G20 fallback, Banner Self Service fallback, the cross-document aggregate (pages 79, 10, 11), the held
appraisal fallback and a guide statement. SHADOW 3 turns: the visible answer stays legacy while Plan F accepts
internally. LEGACY in a fresh process: 10 of 10 minimal records; no Plan F, D-111 or D-112 module imported
(`guide_structure` included); no runtime cache.

Found during the smoke: Plan F guide answers were cited under the chunk's `section` metadata, the PF-2 chunker's page
hint, often carried from an earlier page ("Page 9 · B. Content Visibility:" beside the announcement steps; PF-3C
answers already showed it). They are now cited under the printed heading of the verified procedure or structure
("Page 9 · Send an Announcement"), or under the page alone for a statement. Handbook items and the legacy path are
unchanged.

### Performance

Structure recovery: about 5 ms for the four guides, once per runtime build (the whole guide index about 12 ms). Per
guide clause: scoped retrieval p50 about 550 ms, selection p50 about 1 ms (p95 about 4 ms), adapter guide and findings
verification about 1 ms each, legacy Verifier under 1 ms. A served guide answer p50 about 580 ms end to end; a guide
question that still falls back p50 about 1.97 s (the extra scoped retrieval before the legacy path; the same questions
with scopes off: 1.45 s p50 over all 100).
Plan F model calls 0.

### Tests

Targeted gate (the 27 PF-3C gate modules plus `test_guide_structure.py`): **2,320 passed**, 0 failed, 0 skipped. Full
suite: **6,926 passed, 0 failed, 0 skipped, 0 xfail** (6,858 at the start; 68 new D-112 tests). `pip check`: no broken
requirements. Security scan of the changed and new files: 0 problems.

## Recommendation

`PLAN_F_GUIDE_SCOPES` default: **OFF**. PF-3D meets its criteria, but its rules were developed and measured on the same
47 guide questions; the evidence still missing before a default change is a held-out set of guide questions written
without reference to the guides' layout, at 0 wrong. Enabling it remains the project owner's decision.

Next phase: controlled P1 guide expansion under D-109, D-111 and D-112 (the structure reading is now stable on the four
pilot guides), together with the held-out guide question set; the Performance Appraisal guide stays on HOLD.

## Files

| File | Content |
|---|---|
| `fallback_classification.json` | the 27 PF-3C core fallbacks classified before any change, plus the after-PF-3D state |
| `guide_results.json` | per question PF-3C versus PF-3D (served, outcome, form, parts, pages, stop layer, verdict); no answer text |
| `newly_served_review.json` | every served guide answer with its hand verdict, note and basis (PF-3D review, or the PF-3C review when the answer is byte-identical) |
| `structure_recovery_review.json` | the 10 structural cases: recovered or not, the structure used, the reason if not |
| `evidence_review.json` | evidence selection PF-3C versus PF-3D, first-unit and set measures |
| `adversarial_results.json` | the adversarial tests and the evaluation's safety groups |
| `baseline_comparison.json` | the frozen 192-question rerun against I-9 |
| `metrics.json` | the PF-3D evaluation metrics |
| `app_smoke.json`, `legacy_isolation.json`, `wrong_source_retrieval.json` | application, LEGACY and wrong-source checks |
| `run_eval.py`, `analyze.py`, `make_review.py`, `compare.py`, `evidence_review.py`, `adversarial.py`, `legacy_isolation.py` | the scripts that produce them |
