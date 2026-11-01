# Post-freeze phase PF-7: post-acceptance English safety fixes

Date: 2026-11-01. Branch `feature/post-freeze-knowledge-expansion`, starting commit `962976e` (`ENGLISH_CORE_FREEZE_PF6`).
Decision record: D-117. Trigger: the project owner's 60-question manual ACTIVE acceptance test (guide scopes on, Developer
mode off, the Groq model set by environment to `openai/gpt-oss-120b`). The 60-question results are the owner's frozen
evidence; they are not in this folder and were not changed. Every question used here was written for PF-7.

## What was wrong (reproduced with the real model; `reproduction_before.json`)

| Class | Generic root cause |
|---|---|
| A. iThenticate quota served ("30 documents per year") | Registered conflicts were enforced only inside Plan F. When Plan F declines, every mode falls back to the legacy path (retrieval, synthesis, legacy Verifier), which never read the `source_conflict` stamp: all five retrieved chunks were stamped, the model stated the value, the Verifier accepted it because the value is printed in the evidence. PF-6 ran without a model key, where this path stops at the key notice. |
| B. "I do not have this information." with evidence | The 512-token answer cap was sized for a non-reasoning model; `openai/gpt-oss-120b` spends hidden reasoning inside it (55 to 510 tokens measured). A broad question needing a long answer came back empty at the cap twice and became the refusal sentence. Other refusals are kept safety behaviour: the reranker confidence gate on meta phrasings, the Verifier rejecting an unsupported claim, evidence that no document contains ("course tools"); Plan F still declines broad, conversational and multi-clause questions by design. |
| C. extra or weakly related citations | Legacy citations were bare page numbers: the Verifier cited every page with any individually supporting item, and the UI attached a cited page number to every retrieved item with that number from any document (every guide starts at page 1). A refusal showed a section chip; the model wrote "[Source 3 \| Page 109]" marks into answers. |

## What changed (D-117)

| File | Change |
|---|---|
| `handbook_bot/conflict_guard.py` (new) | legacy-path conflict enforcement from the registry alone: value conflicts lose the units that state the disputed claim (the D-114 rule), other stamps lose the chunk, an answer that states the disputed claim is refused; no Plan F import |
| `handbook_bot/orchestrator.py` | screening before the evidence is used; the answer check; an accepted model answer carries only the items its claims rest on |
| `handbook_bot/agents/verifier.py` | `citing_items`: the one best supporting item per claim (union support keeps every contributor); a number written with a data-size unit ("100MB") is the same number as "100 MB" |
| `handbook_bot/qa.py` | reasoning allowance on the cap; a reply cut off before any text is a generation failure, not a refusal; inline citation marks removed from the answer text |
| `handbook_bot/config.py` | `LLM_REASONING_TOKENS` (environment, default 0 = PF-6 behaviour) |
| `handbook_bot/agents/router.py` | the same allowance on the router's cap |
| `ui/presentation.py` | a refusal shows no source chip |
| `tests/test_pf7_legacy_safety.py` (new) | 43 generic tests (synthetic topics, documents and values; the real registry is only read) |

Not changed: Plan F, retrieval, thresholds, the Verifier's acceptance rule, the corpus and cache, the registry, documents,
held sources, `PLAN_F_GUIDE_SCOPES` (OFF), the default model setting.

## Validation with the real model (fresh wording)

| Set | Result |
|---|---|
| conflicts (13; `conflict_validation.json`) | original yearly-limit question and 5 fresh paraphrases refused; **0 disputed values served**; answer check fired once (revision counting); 5 ordinary iThenticate requests answered (upload, log in, file types, file-size limit, support e-mail); journal recommendation blocked (2) |
| natural language (20; `natural_language_validation.json`) | 14 of 14 fresh broad or conversational questions answered with the approved source cited; **0 drafts cut off at the cap** (the empty one before now answered, 717 tokens); weather, held Grade Categories, held Appraisal, policy trap, "I need help." refused; a question-pool test question answered from the enabled Short Guide (limited; the held guide never retrieved) |
| citations (10, 8 legacy model answers; `citation_validation.json`) | document chips 11 to 8, page chips 15 to 9, documents supporting no claim 1 to **0**; every displayed document supports a claim; Plan F cross-document answers unchanged |

## Regression against PF-6 (no model key, deterministic)

| Check | Result |
|---|---|
| frozen 192, guide scopes off and on | every record identical to PF-6: 55 served (the I-9 55), 55 correct, 0 wrong, 0 false acceptances, SHADOW 192 of 192; T07 the only change against I-9 |
| P0 / P1 / diagnostics | P0 21 of 37 core (19 / 2 / 0), 35 of 100; P1 7 of 41 (5 / 2 / 0); held-out 1 of 50, 0 of 56, 0 of 60; every served answer byte-identical |
| five-set records differing from PF-6 | 16 = 4 questions x 4 modes, all legacy fallbacks without a model key whose retrieval carried conflict stamps (P02 quota, M09/M10 Blackboard journal: other pages behind the key notice; F42 research-group journal: now a refusal) |
| SHADOW / determinism | 307 of 307; 900 repeat records, 0 differences; 0 exceptions; 0 model calls |
| wrong source / inference adversarial | 88 cases identical to PF-6, 0 acceptances, 0 violations; 30 of 30 refused |
| conflicts / boundary / held sources | 0 quota and 0 journal leaks (Q09 the PF-6 conservative fallback); 16 of 16; 6 of 6 |
| LEGACY isolation | fresh process, 20 of 20 minimal, no Plan F module imported |
| tests | targeted gate 2,870 passed; full suite 7,322 passed, 0 failed, 0 skipped, 0 xfail; `pip check` clean |

## Remaining known limitations

The reranker gate still refuses some meta phrasings ("which official source should I use ..."); the legacy Verifier rejects
some correct drafts (a value written "100 MB" against the printed "100MB"); model answers vary between runs; Plan F keeps
its D-114/D-115 limits; a conflict is reported with the refusal sentence, not a dedicated message.

## Retest configuration

`launch_acceptance.py <repo> <log>` starts the application with ACTIVE mode, guide scopes on, `GROQ_MODEL=openai/gpt-oss-120b`
and `LLM_REASONING_TOKENS=2048` in its own process, the key loaded privately from `.env`; nothing in the repository's
defaults changes.

## Files

| File | Content |
|---|---|
| `questions_reproduction.txt`, `reproduction_before.json` | the PF-6 behaviour reproduced (`probe_before.py`) |
| `questions_conflict.txt`, `conflict_validation.json` | quota question, fresh paraphrases, normal iThenticate requests, the journal conflict |
| `questions_natural_language.txt`, `natural_language_validation.json` | fresh broad and conversational paraphrases, unsupported, held, policy-trap and vague questions |
| `questions_citation.txt`, `citation_validation.json` | PF-6 chips and D-117 chips for the same answers, with the support check |
| `regression.json` | the PF-6 verification rerun on the D-117 code |
| `validate_live.py`, `citation_live.py`, `probe_before.py`, `summarize.py`, `regression.py`, `launch_acceptance.py` | the scripts |
