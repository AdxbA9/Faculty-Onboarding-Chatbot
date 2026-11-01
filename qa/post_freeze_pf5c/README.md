# Post-freeze phase PF-5C: safe guide-intent inference

Date: 2026-11-01. Branch `feature/post-freeze-knowledge-expansion`, starting commit `a394887` (PF-5B). Frozen control:
`runtime-freeze-i9` (D-108). Decision record: D-115. No new document, no Arabic, no voice, no UI feature, no new specialist,
no model call; the corpus stays 8 enabled sources, 309 pages, 15,000 chunks. `PLAN_F_GUIDE_SCOPES` stays off by default.

## Outcome in one paragraph

PF-5C adds deterministic guide-intent inference: when a question names no system, a fixed table of 79 signals printed in the
official guides may make one guide family a candidate (score at least 2.0, a lead of at least 1.5, a distinctive object). The
inference never becomes evidence: it only lifts the system-name check of the D-111 binding, every asked word must still be
stated by the guide, and an inferred clause is held to a stricter profile (a complete procedure or section whose printed
heading names the asked action and every asked object; never a single statement). Two adversarial rounds (3,069 probes, 53
confirmed findings) hardened it, always by refusing more. On the frozen code nothing served is wrong in any set; P0, P1, the
two diagnostic held-out sets and the frozen 192-question set are unchanged. The new blind fresh set, frozen before any code
and run once, serves **0 of 60 (0 of 31 answerable, 0 of 25 implicit-system answerable)**: inference fires on 4 of the 36
implicit questions, every time to the labelled guide, but most answerable questions are conversational (a context sentence
plus the ask) and stop at the all-or-nothing multi-part contract that PF-5C was not allowed to change.

## The inference (`handbook_bot/guide_inference.py`, D-115)

| Step | Rule |
|---|---|
| refused before scoring | a named system (the D-111 path applies); another platform; the D-114 organizational vocabulary; help, support, troubleshooting and information requests; policy, obligation, permission, advice, misconduct and appeal asks; another actor (students, a TA, a secretary, a colleague); the asker's or a colleague's own attendance or accommodation; a task done for a colleague or scoped to one student or group; any ask but a procedure |
| scoring | printed objects (2.0 multi-word, 1.0 distinctive word, 0.5 context) plus 1.0 once when a task verb governs one within three words; phrases match their printed words, adjacent, never across a possessive or another inflection |
| acceptance | best family at least 2.0, at least 1.5 ahead of the runner-up, at least one distinctive object; candidates are the guides whose own signals matched |
| collision blocks | research journal, events (meeting, committee, training, orientation, exam), course administration, housing, grades sent to a person, students' work for iThenticate, teaching assignment or a student's submission |
| evidence | `bind(inferred=True)` lifts only the system-name check; strict profile: a procedure, section or labelled item headed by the asked action and every asked object; no statement |
| runtime | registry selection by inference; a no-specialist one-clause question goes to the inferred guide's owner; the D-114 Institutional-to-owner reassignment also applies; a decision changed by inference serves only a verified guide answer; inferred bindings never drop a handoff |
| diagnostics | trace and Developer mode: outcome or refusal code, family, candidates, score, runner-up, margin, signal keys; no question text |

Named-system grounding gained refusals only (`guide_grounding.py`): labelled and lead-in headings naming another action, a
destructive heading never overridden by a passing mention, see/find asks need an access heading, a heading naming the action
on another object, role-specific procedures ("as Chair/Dean"), negated asks, a location stated in another sentence than its
object or as the place of something else, yes/no statements only about students, bounds in another unit, recipient faculty
groups kept as terms.

## Fresh held-out methodology (`fresh_held_out_questions.jsonl`, `fresh_held_out_labels.json`)

Written before any PF-5C code by four independent writer agents (15 questions each) given only task families, personas and a
required composition (24 implicit-system, 8 explicit-system, 4 each policy trap, organizational, unsupported, ambiguous,
multi-part, false premise, typo or very short); 0 near-duplicates; frozen at 2026-10-31T22:18:30Z, SHA-256
`3c8325a7...d95bcde4ad` (pinned by a test). Labelled afterwards by two independent labelers per question from the source text
(53 agreed, 7 adjudicated by a third agent): 31 answerable, 23 partially, 6 not at all; 36 need system inference (25 of them
answerable), 9 name their system. The implementing session never read the questions or labels before the final run; the run
prefixes the ids (`F5C_C01`) because C01-C60 collide with PF-3C P0 ids.

## Results (`metrics.json`; sets reported separately)

| | questions | Plan F served | correct / limited / wrong | precision | coverage |
|---|---|---|---|---|---|
| P0 core guide | 37 | 21 (PF-5B 21) | 19 / 2 / 0 | 100% | 56.8%, byte-identical |
| P0 all guide answers | 100 | 35 (PF-5B 35) | 32 / 3 / 0 | 100% | byte-identical |
| P1 | 41 | 7 (PF-5B 7) | 5 / 2 / 0 | 100% | B10 the same words in another layout |
| PF-4 held-out (diagnostic) | 50 (15 answerable) | 1 (unchanged) | 1 / 0 / 0 | 100% | 6.7% of answerable |
| PF-5B fresh (diagnostic) | 56 (20 answerable) | 0 (unchanged) | 0 / 0 / 0 | n/a | 0% |
| **PF-5C fresh** | 60 (31 yes, 23 partial, 6 no) | **0** | 0 / 0 / 0 | n/a | 0% of all, 0% of answerable, 0% of 25 implicit |

Fresh false fallbacks (`fresh_held_out_results.json`, `inference_analysis.json`): of the 31 answerable questions, 19 stop at
aggregation (a conversational context clause no source supports), 5 find no specialist, 3 are routing-ambiguous, 1 each is
refused by the organizational vocabulary, by the policy frame, by the owning specialist's approved sources and by grounding.
Inference refused 32 of the 36 implicit questions before scoring or for weakness: organizational wording 9 ("lab", "room",
"building", "who", "registrar"), no printed signal 6 (typos in task words), unrecognised ask 5, weak 3, other 9. Every "must
not serve" fresh question falls back.

## Safety

Frozen 192 (`frozen_baseline_comparison.json`): the same 55 answers as I-9, 0 wrong, 0 false acceptances, SHADOW 192 of 192,
guide registry never selected (inference accepts none of the 192); T07's legacy citation as in PF-4. Wrong-source
(`wrong_source_results.json`): 88 cases (28 new inference collisions: attendance policy and tool, Gradebook and Grade
Transfer, "collaborate", research and Blackboard journals, similarity and other reports, support office and software, online
teaching and other platforms, assignment and grade submission, messages, sessions, course administration): 0 wrong-source
acceptances, 0 violations, 11 served (all correct). Inference adversarial (`inference_adversarial.json`): 30 of 30 refused, the
six the owner listed included, nothing served. iThenticate quota conflict blocked; appraisal hold enforced; incomplete
procedures refused. SHADOW 307 of 307; 600 repeat records, 0 differences; 0 model calls; 0 exceptions; LEGACY fresh process
(`legacy_isolation.json`) 20 of 20 minimal, no D-115 module imported. Scopes off versus on: 48 differences, all reviewed guide
answers or the same visible legacy fallback (H09, H11, H12, H27, F5C_C30).

## Adversarial review and differential

Round one (four lenses, 1,185 probes): 22 findings, 19 confirmed. Round two on the hardened code (three lenses, 1,884 new
probes): 36 findings, 34 confirmed. Every confirmed finding was fixed by refusing more; round two led to procedure-only
inference with the strict evidence profile. HEAD-versus-final worst-case differential over 2,368 probe questions (every chunk
retrieved, fresh sets excluded): 8 newly accepted, all correct; 56 HEAD answers refused (wrong or weak; two conservative
losses); 10 changed (6 identical words, 2 better, 2 equally weak). Known remaining weakness: worst-case location, selection and
yes/no statements for named-system questions (PF-5B-era grounding), not reached by any evaluated set.

## Application, performance, tests

Application smoke (`app_smoke.json`, ACTIVE, 19 turns on the real chat page): 9 served, all correct (named and inferred
Blackboard, Collaborate, Grade Transfer, Gradebook, Attendance, iThenticate, a misspelled system name), 10 fallbacks (online
class cut in the extraction, policy trap, quota, support office, software "where", ambiguous, two-typo, unsupported,
appraisal), 0 model calls; the inference row appears only in Developer mode. Performance (idle, 191 questions, PF-5B versus
PF-5C code): startup 25.0 s and 23.9 s, retrieval p50 606 and 607 ms, selection 2.7 and 2.7 ms, served answer 621 and 618 ms,
fallback 2,137 and 2,156 ms, inference 0.4 ms p50. Tests: `tests/test_pf5c_inference.py` (197); targeted gate (33 modules)
2,827 passed; full suite 7,279 passed, 0 failed, 0 skipped; `pip check` clean.

## Recommendation

`PLAN_F_GUIDE_SCOPES` default: **OFF**. Safe but not useful enough: fresh coverage is 0% (criterion: at least 50% of the
answerable and of the implicit-system answerable questions at 100% precision). Next: PF-6 Final English Technical Freeze.

## Files

| File | Content |
|---|---|
| `fresh_held_out_questions.jsonl`, `fresh_held_out_labels.json` | the frozen blind set (hash pinned) and its two-labeler labels |
| `fresh_held_out_results.json`, `fresh_held_out_review.json` | per fresh question what ACTIVE did and why; served fresh answers (none) |
| `inference_analysis.json`, `inference_adversarial.json` | inference per set and per fresh question; the must-refuse set |
| `pf4_diagnostic_comparison.json`, `pf5b_diagnostic_comparison.json` | the two older held-out sets before and after |
| `p0_regression.json`, `p1_regression.json`, `procedure_qa.json` | P0 and P1 answers carried or reviewed; newly served procedures |
| `frozen_baseline_comparison.json`, `wrong_source_results.json`, `app_smoke.json`, `legacy_isolation.json` | baseline, adversarial, application and LEGACY checks |
| `metrics.json`, `hand_verdicts.json` | compact metrics; PF-5C hand verdicts bound to answer hashes |
| `run_eval.py`, `make_review.py`, `analyze.py`, `derive.py`, `wrong_source.py`, `inference_adversarial.py`, `legacy_isolation.py` | the scripts |
