# Plan F runtime integration freeze (runtime integration, phase I-9)

Freeze identifier: `runtime-freeze-i9` (documentation and repository metadata only; `handbook_bot.__version__` stays `1.0.0`).
Date: 2026-10-30. Branch: `feature/remaining-specialists`. Base HEAD: `eeb516f4c90f4e8566491137345fda433a96a3a1` (the
I-8 commit); the frozen HEAD is the commit that adds this document and the manifest, subject "Freeze Plan F runtime
integration" (its hash is recorded in the I-9 STATUS entry's successor and in `git log`, since a commit cannot contain
its own hash). Runtime Integration I-1 through I-9 COMPLETE. The machine-readable manifest is
`qa/plan_f_i9/freeze_manifest.json`, produced by `qa/plan_f_i9/make_manifest.py`; re-run that script after any later
work and diff the JSON to see what moved (file hashes are the comparison; the recorded base HEAD is informational).

## 1. Safety statement

The system does not guarantee perfect answers. The frozen runtime prioritises grounded evidence, deterministic
verification and safe fallback over coverage. The supportable claim is: Plan F achieved 100% precision on the 55 answers
it served in the final 192-question I-8 evaluation, with 0 wrong served answers; overall coverage was 28.6% and
unsupported cases fell back to the legacy path. No claim of "zero hallucinations", "100% accuracy" or "production perfect"
is made or supported. The legacy model path was not evaluated with a real model client in I-3 to I-9 (none is configured);
on that path the application returns its no-key reply.

## 2. Architecture summary

    user question
      -> Coordinator (handbook_bot/agents/coordinator.py): clauses, domain cues, one or more specialist tasks, or none
      -> deterministic routing refinement (plan_f_routing.py, plan-f-routing-1): one owner per request clause or ROUTE_AMBIGUOUS
      -> specialist execution (dispatch.py over the four frozen specialists; real retrieval and reranking inside each)
      -> I-1 guards (quote guard: exact substring of the named chunk; subject guard: entities, requested details, content words)
      -> findings-aware verification (plan_f_verification.py, I-4 + I-8 rules) on every finding against its clause
      -> optional bounded handoff round (plan_f_handoff.py, plan-f-handoff-1): one hop, depth 1, visited-target protection
      -> deterministic aggregation (plan_f_aggregation.py, plan-f-aggregation-1): every clause covered or no aggregate
      -> adapter safety (plan_f_runtime.py): extractive rendering, unchanged legacy Verifier, complete QAResult
      -> Plan F answer, or the legacy path (orchestrator / qa.py: router, retrieval, extractor or model synthesis, Verifier)
      -> seam (ui/pipeline.py: LEGACY, SHADOW, ACTIVE; per-turn diagnostics without question or answer text)
      -> interface (ui/chat.py + ui/presentation.py: answer, handbook and page chips, a domain chip only for a
         verified specialist answer, Developer panel from the diagnostics only)

Modes (`handbook_bot.config`, both switches default to off): LEGACY (`PLAN_F_ENABLED` off, `PLAN_F_SHADOW` off) calls
the legacy path exactly as before and never imports or builds Plan F; SHADOW (`PLAN_F_SHADOW` on) returns the legacy
result unchanged and runs Plan F beside it for diagnostics only; ACTIVE (`PLAN_F_ENABLED` on, shadow off) serves a Plan F
answer only when the adapter accepted it and the findings-aware layer accepted it (on the aggregation path: the aggregate
and every clause's own record), else the legacy result with the fallback reason in the diagnostics. No Plan F failure can
raise through the seam.

## 3. Frozen metrics (I-8 final, 192-question master set, real knowledge base, no model client)

| Metric | Frozen value |
|---|---|
| Questions | 192 (40 adversarial by category, 98 by tag, 75 regression) |
| Plan F served / fallback | 55 / 137 |
| Coverage | 28.6% |
| Correct / limited / wrong among served | 55 / 0 / 0 |
| Precision | 100% |
| False acceptances | 0 |
| False fallbacks | 11 (coverage items, see section 13) |
| SHADOW visible mismatches | 0 of 192 |
| Handoffs executed / accepted | 3 (depth 1) / 0 |
| Plan F model calls | 0 |

I-9 reproducibility (the same harness re-run on the frozen tree with the cached knowledge base, compared question by
question with `qa/plan_f_i8/results_2026-10-30.json`): 0 differences in 192 ACTIVE and 192 SHADOW records on served or
fallback decision, outcome, answer text, cited pages, routing code and final specialists, specialist calls, handoff
outcome, aggregation outcome and coverage, findings reason, the served evidence items (chunk, page, quote), the direct
attempts (outcome, handoff, specialist runs) and the LEGACY sample; SHADOW identical to the direct legacy call 192 of
192; 0 exceptions; the metrics above reproduced exactly (55 served, 55 correct, 0 false acceptances, 11 false fallbacks,
0 model calls). The frozen metrics are therefore unchanged between the I-8 final run and the I-9 freeze.

## 4. Mode contract (verified in I-9)

| Mode | Contract | I-9 result |
|---|---|---|
| LEGACY | the legacy result exactly; no Plan F import, registry build, runtime cache, label, chip or diagnostics row; the minimal `{mode, served, attempted}` record | PASS: fresh-process sample of 30 (minimal record 30 of 30, no Plan F module imported, no cache, 0 exceptions); in-process sample after ACTIVE and SHADOW the same; app smoke with no Plan F row |
| SHADOW | Plan F runs internally; the visible answer, pages, section, query type, evidence, items, model flags and sub-questions are the legacy ones; Plan F visible only in Developer diagnostics | PASS: 192 of 192 visible outputs identical to the direct legacy call in the I-9 rerun (0 leaks, 0 exceptions); app smoke with Developer mode off showed the legacy extractor row and the legacy no-key reply with legacy chips (pages 54; 94, 96, 107) and no Plan F text, with Developer mode on the panels read SHADOW, "Legacy (Plan F in shadow)", the internal outcome (accepted single finding, accepted aggregate), "built (728 ms)" then "reused"; switching the mode off removed every panel |
| ACTIVE | only an adapter-accepted and verifier-accepted result is served; every other case falls back to the legacy result with the reason recorded; evidence pages come only from accepted findings; interface labels follow the diagnostics | PASS: in the I-9 rerun the 55 served answers are the adapter- and verifier-accepted results of I-8 with identical text, pages and evidence items, the other 137 questions fell back with their reason codes and 0 exceptions; in the app smoke the six served answers carried "Verified · domain" chips and page chips from the evidence, the six fallbacks carried legacy chips and the fallback reason in the panel only |

## 5. Routing freeze (`plan_f_routing.py`, version `plan-f-routing-1`)

Deterministic and free of question text; calls no model and never retrieves; never invents an owner. Policy: (1)
Institutional Navigation owns a contact or location clause whoever's unit it names; (2) a subject specialist owns an
approval, authority or responsibility clause about its subject and Institutional Navigation answers no such clause; (3)
an office-selection ask ("which office handles") stays ambiguous against a subject claim; (4) two subject specialists
claiming one clause stay ambiguous: nothing is chosen by score, by registration order or by tie-break; (5) a single
specialist's whole-question task over a multipart question becomes one clause-scoped task per request clause. Codes that
form the freeze contract: `ROUTE_UNCHANGED`, `ROUTE_CLAUSES_SCOPED`, `ROUTE_REFINED_BY_REQUEST_KIND`, `ROUTE_AMBIGUOUS`
(no specialist runs; adapter outcome `PLAN_F_REJECTED_BY_ROUTING`), `ROUTE_NO_OWNER`, `ROUTE_ERROR` (exception category
only). I-8 distribution over 192 attempts: unchanged 142, refined 22, scoped 15, ambiguous 12 (the greeting has no record).

## 6. Handoff freeze (`plan_f_handoff.py`, version `plan-f-handoff-1`)

`MAX_DEPTH = 1`; approved targets are the four registered specialists only; a target runs once, with the source's own
declined clause, never an invented query. Safeguards, each with a synthetic fixture in `tests/test_plan_f_routing.py`
and the real-knowledge-base cases of I-6 and I-8: self-loop (`HANDOFF_SELF_LOOP`), visited or repeated target
(`HANDOFF_TARGET_ALREADY_VISITED`), unknown or malformed target (`HANDOFF_TARGET_INVALID`), second hop
(`HANDOFF_SECOND_HOP_BLOCKED`), unresolved clause (`HANDOFF_CLAUSE_UNRESOLVED`), unsupported remainder
(`HANDOFF_REMAINDER_UNSUPPORTED`), a target that reports a model call (`HANDOFF_UNSAFE_RESULT`), an exception
(`HANDOFF_EXCEPTION`, category only); `HANDOFF_NONE` and `HANDOFF_EXECUTED` complete the round codes, and after the
aggregator decides the user-level outcome is `HANDOFF_ACCEPTED`, `HANDOFF_TARGET_REJECTED` or `HANDOFF_CONFLICT`. No
code path permits a chain: the round runs once per attempt and a stamped handoff task is never re-dispatched. Real
knowledge base: six rounds attempted in I-8, three executed at depth 1 (Research to Institutional Navigation), none
accepted; one blocked by the visited-target protection.

## 7. Aggregation freeze (`plan_f_aggregation.py`, version `plan-f-aggregation-1`)

Every task of the decision is dispatched once; every request clause must be covered by findings that pass the
findings-aware verifier on their own clause; parts are rendered verbatim in request order, one quote per paragraph,
nothing added and no generative rewrite; each part keeps its specialist, finding, chunk and page (provenance in the
item metadata); a duplicate finding is collapsed; a value conflict rejects; a missing clause rejects (all or nothing).
Codes: `AGGREGATED_ACCEPTED`, `NO_AGGREGATION_REQUIRED`, `MULTIPART_UNSUPPORTED`, `UNRESOLVED_CLAUSE`,
`CROSS_DOMAIN_UNRESOLVED`, `HANDOFF_REQUIRED`, `UNSAFE_RESULT`, `MISSING_REQUIRED_CLAUSE`, `CHILD_FINDING_REJECTED`,
`VALUE_CONFLICT`, `AGGREGATION_ERROR`. The rendered aggregate then passes the unchanged legacy Verifier, which cites at
most three pages; an aggregate whose evidence spans more pages is rejected by the adapter's "every rendered part keeps its
page" rule (`PLAN_F_REJECTED_BY_VERIFIER`, see limitation L4).

## 8. Verification freeze (`plan_f_verification.py`, phase I-4 rules plus the two I-8 rules)

Guarantees on every rendered finding: evidence integrity (the quote is an exact substring of the named chunk on the
cited page of the cited source, no normalisation); subject and entity binding (named entities, population or role,
category modifiers, the office a clause names for Institutional Navigation); request kind read deterministically from
the clause (`contact`, `location`, `quantity`, `amount`, `percentage`, `duration`, `frequency`, `timing`, `value`,
`definition`, `authority`, `responsibility`, `selection`, `eligibility`, `obligation`, `yes_no`, `enumeration`, `general`,
`unrecognised`); requested detail present in the answering unit (a value of the asked unit, a contact field bound in the
directory row, a location, a named party before the asked action); condition and modifier preservation (an asked
condition stated with the answer, no contradicted modifier on the same noun); the asked relation stated (definition
subject, party and action, obligation for the asked actor, eligibility); completeness classes `complete`, `elaborated`,
`specialist_contract`, `partial`, `enumeration_unproven`, `excess`, `not_assessed`; provision boundary (a quote that spans
another provision is excess evidence, a provision slice may be accepted). Open-function asks ("what does X do", with or
without a trailing for/to/with phrase since I-8) and other enumerative asks are never answered by one finding
(`ENUMERATION_INCOMPLETE`); a general ask is never answered by topical support (`GENERAL_ASK_MISMATCH`). Yes/no guard
(I-8): a yes/no ask with no readable action is answered only by a statement that itself grants, forbids, entitles or
requires; a plain value statement never confirms or refutes it (`RELATION_MISMATCH`); the verb path keeps the polarity
check. Reason-code vocabulary: `ACCEPTED`, `UNSUPPORTED_STATUS`, `PARTIAL_FINDING`, `NOT_FOUND`, `OUT_OF_SCOPE`,
`HANDOFF_REQUIRED`, `MULTIPART_UNSUPPORTED`, `GENERAL_ASK_MISMATCH`, `UNRECOGNISED_ASK`, `SUBJECT_MISMATCH`,
`RELATION_MISMATCH`, `REQUESTED_DETAIL_MISSING`, `CONDITION_MISMATCH`, `MODIFIER_MISMATCH`, `ENUMERATION_INCOMPLETE`,
`EXCESS_EVIDENCE`, `VALUE_CONFLICT`, `INVALID_EVIDENCE_REFERENCE`, `VERIFICATION_ERROR`. These codes are technical
documentation; the interface shows them in Developer mode only.

## 9. Adapter freeze (`plan_f_runtime.py`)

Accepted: a supported result of the dispatched specialists whose findings pass the quote guard, the subject guard and
the findings-aware verifier (single finding) or the aggregator with every clause's own record (aggregate), rendered
extractively and accepted by the unchanged legacy Verifier with every evidence page cited. Blocked: partial or
not-found or out-of-scope status, a pending handoff, a `missing` entry, a model call, a missing or malformed clause
report, a general or unrecognised ask shape, uncovered requested details, a weak relation (`PLAN_F_UNSAFE_RESULT`);
a quote that is not an exact substring (`PLAN_F_REJECTED_BY_QUOTE_GUARD`); a finding that does not bind the asked
subject (`PLAN_F_REJECTED_BY_SUBJECT_GUARD`); the verifier codes above (`PLAN_F_REJECTED_BY_FINDINGS_VERIFIER`);
aggregator rejections (`PLAN_F_REJECTED_BY_AGGREGATOR`); ambiguous routing (`PLAN_F_REJECTED_BY_ROUTING`, no specialist
runs); a handoff round outside the bounds (`PLAN_F_REJECTED_BY_HANDOFF`); the legacy Verifier (`PLAN_F_REJECTED_BY_VERIFIER`);
nothing renderable (`PLAN_F_REQUIRES_FALLBACK`); any exception (`PLAN_F_ERROR_REDACTED`, category only); greetings,
empty input and questions without a domain cue (`PLAN_F_NOT_APPLICABLE`); no complete registry
(`PLAN_F_REGISTRY_UNAVAILABLE`). Every blocked outcome is a seamless legacy fallback. Known HR gate: "What does HR do?" and
"What does Human Resources do?" stop at the I-1 ask-shape gate (general ask), "What does HR provide to new faculty
members?" and "What does HR do for new faculty?" at the enumeration rule. Annual-leave protection: "How many days of annual
leave do faculty members get?" is a partial result (the handbook gives no number) and is never served, alone or inside an
aggregate, so the maternity sentence cannot stand in for it. Registry cache (I-2): one `RuntimeCache` per process, keyed by
knowledge-base object identity through a weak reference, bounded to four entries, a build holds the lock so concurrent
first requests wait for one build, a failed build is remembered for that object; LEGACY never creates it.

## 10. Interface freeze (`ui/presentation.py`, `ui/chat.py`, `ui/styles.py`; I-7)

Labels come from the turn's diagnostics (verified specialist answer, verified multi-part answer, routing refined,
handoff completed, legacy extractor, legacy model path, refusal, legacy fallback after a Plan F attempt, shadow run);
timings are recorded values only (retrieval and rerank N/A for Plan F answers, "<0.1 ms" for tiny recorded values,
the Plan F attempt apart from the legacy stages on a fallback, the end-to-end wall time measured around the call);
one chip per cited page with its section and the handbook once; a domain chip only for a served Plan F answer; technical
identifiers escape Markdown emphasis; aggregate parts render one per paragraph; each message owns a copy of its
diagnostics; a pipeline exception yields a fixed safe reply with the exception class in Developer mode only; the loading
state reads "Searching official university sources…" with `role="status"`; controls carry accessible names; 375 px
layout without horizontal overflow. `ui/chat.py` contains no runtime module name, Coordinator name or switch name
(enforced by `tests/test_ui_presentation.py`).

## 11. I-9 verification record

- Starting state: HEAD `eeb516f4c90f4e8566491137345fda433a96a3a1`, clean tree, tracking ref equal (I-8 pushed by the
  developer), Python 3.11.9 on Windows, `pip check` clean, knowledge-base cache v12 present, six frozen hashes matched.
- Critical regression cases in the I-9 rerun, all as in the final I-8 run: "What does HR do?" blocked at the I-1 gate
  (`PLAN_F_UNSAFE_RESULT`, general ask shape); annual leave blocked (partial), alone and inside the maternity aggregate;
  "What does HR do for new faculty?" blocked (`ENUMERATION_INCOMPLETE`); the false-premise "Is the teaching load for
  lecturers 12 credit hours per semester?" blocked (`RELATION_MISMATCH`, yes/no guard); the WLAM definition blocked
  (`RELATION_MISMATCH`); the consultancy provision slices served (pages 260); conference-travel authority served by
  request-kind refinement (page 117); research-grant authority served (page 165); housing `ROUTE_AMBIGUOUS` with 0
  specialist calls; "Where is the Research Institute?" handoff executed at depth 1 and fallen back; the same-domain
  multipart, cross-domain and multi-page aggregates served with identical parts and pages; the SHADOW internally
  accepted cases returned the legacy answer. Determinism: 24 questions (single findings, aggregates, routing-refined,
  ambiguous, handoff, known safety fallbacks, unsupported) observed four times each: route, specialist count, served or
  fallback decision, answer, page list and handoff outcome identical in 24 of 24.
- Concurrency and cache: `qa/plan_f_i9/concurrency_smoke.py` with eight threads and sixteen calls on a fresh runtime
  state built the registry exactly once (as the sequential reference did), every concurrent answer equalled the
  sequential answer of its question (served path, outcome, text, pages, specialist calls), sixteen diagnostics records
  with no foreign pages, 0 exceptions, 0 model calls; the existing cache tests (`test_cache_is_safe_under_concurrent_first_use`
  and the `RuntimeCache` suite) are part of the targeted gate. Knowledge-base cache v12 reused at every start (about 23 s,
  no embedding rebuild); registry built once per process and reused on every later request in the harness and in the
  app; LEGACY never builds it.
- App smoke (built-in browser, the real cached knowledge base, no model key, switches in the process environment
  only): ACTIVE with Developer mode, twelve questions: greeting (not applicable), lecturers' load (Teaching single
  finding, page 96), petty cash advance (Research, page 166), sick leave (HR, page 107), HR telephone (Institutional
  Navigation row, 2 ms), HR telephone plus Passport Unit fax (two-part aggregate, pages 54 and 53), conference travel
  (routing refined, page 117), "What does HR do?" (known HR fallback, general ask shape), annual leave (partial,
  fallback), housing (ambiguous routing, 0 specialist calls), "Where is the Research Institute?" (Handoff section:
  Research to Institutional Navigation, depth 1, both visited, fallback), capital of France (refusal, no chips). No
  crash, no stale loading bubble, one Developer panel per answer with its own served path and pages, "Verified ·
  domain" chips on the six served answers only, the registry built on the first request and reused on the eleven
  others, no exception, 0 model calls, process memory 1,293 MB; the launcher log carries 12 records without question
  or answer text. SHADOW, two questions: legacy extractor row and legacy no-key reply with legacy chips, no Plan F text
  with Developer mode off, the internal accepted attempts in the panel only. LEGACY, three questions: Mode LEGACY,
  Served path Legacy, "Legacy answer" / "Legacy deterministic extractor" / "Legacy model path (no model client
  configured)", no Plan F, routing, verification, aggregation or handoff row or section, no Plan F record in the
  launcher log.
- Performance snapshot (I-9 rerun vs I-8 final, median / p90 / max, CPU, Windows): cached knowledge-base startup
  23.0 s (I-8 22.9 s); cold first ACTIVE request 2,069 ms with an 851 ms registry build (I-8 1,755 ms, 697 ms); served
  single Plan F answer 626 / 826 / 996 ms (I-8 660 / 744 / 852); served aggregate 1,179 / 1,305 / 1,326 ms (I-8 1,307 /
  1,408 / 1,456); routing rejection attempt 1 / 2 / 2 ms (same); legacy fallback end to end 1,397 / 2,426 / 3,285 ms
  (I-8 1,487 / 2,560 / 3,489); total ACTIVE response 1,151 / 2,336 / 3,285 ms (I-8 1,239 / 2,516 / 3,489); direct legacy
  call 760 / 1,657 / 1,758 ms (I-8 834 / 1,860 / 2,048). Ordinary run-to-run variance, no material regression; the
  registry was reused on 191 of 192 questions.
- Diagnostics hygiene: 486 seam records of the rerun (ACTIVE, SHADOW, repetitions, LEGACY) with 0 raw questions, 0 raw
  answers, 0 tracebacks, 0 filesystem paths, 0 credentials; the app launcher logs of the smoke likewise; the bounded
  records of the seam carry codes, counts, pages, timings and redacted error categories only.
- Tests on the frozen tree: targeted gate (22 files: adapter, routing and handoff, aggregation, findings verifier,
  pipeline, imports and isolation, dispatcher, Coordinator, contracts, registry, legacy verifier, compatibility,
  orchestrator, router, sources, the frozen specialists' integration and acceptance suites, UI presentation) 1,534
  collected, 1,534 passed, 0 failed, 0 skipped, 0 xfail (91 s); full suite 6,674 collected, 6,674 passed, 0 failed, 0
  skipped, 0 xfail (220 s; the I-8 baseline); `pip check`: no broken requirements (no known warning to document).
- Frozen specialist hashes: 6 of 6 MATCH before and after I-9 (Teaching, Research, Faculty Services & HR, HR claim
  model, Institutional Navigation, navigation cues).
- Security: the freeze change set and the QA artifacts hold no key, token, password, `.env` content, personal path,
  model cache, embeddings, large generated file, screenshot, AI attribution, co-author trailer or generated-by notice.

## 12. Post-I-9 change policy

The runtime baseline is frozen at `runtime-freeze-i9`. Later work (additional official document ingestion, selective
specialist reopening based on measured gaps, Arabic, an improved source viewer, onboarding and product features,
feedback, demo polish) is compared against this baseline: re-run `qa/plan_f_i8/master_eval.py` on the cached knowledge
base, `qa/plan_f_i8/compare_runs.py` against `qa/plan_f_i8/results_2026-10-30.json`, the targeted gate and the full
suite, and `qa/plan_f_i9/make_manifest.py` against `qa/plan_f_i9/freeze_manifest.json`. Frozen behaviour is not changed
silently: a change to a frozen specialist needs a new decision record (D-103, D-107, D-108); a change to the routing,
handoff, aggregation, verification or adapter contract needs a STATUS entry that reports the before-and-after metrics
of the master set, and a safe conservative behaviour is not weakened merely to raise acceptance. Both Plan F switches
keep defaulting to off until a product decision sets a default mode.

## 13. Known limitations register (non-blocking)

| # | Limitation | Class | Note |
|---|---|---|---|
| L1 | Plan F coverage is conservative: 55 of 192 master-set questions served; 11 false fallbacks where the handbook states the fact | COVERAGE | by design; the legacy path answers the rest |
| L2 | WLAM profile loads for named categories (Active Research, Research Intensive, Teaching Track) fall back because the value sentence names the category only in the preceding sentence of the bullet | COVERAGE | verifier entity binding; candidate for a later verifier refinement |
| L3 | Misspelled or unusual phrasings ("ofice", "aproves confernce", "consultation-hour") reach the legacy path because no domain cue is recognised | COVERAGE | safe; a Coordinator cue item |
| L4 | Three-clause aggregates whose evidence spans more than three pages are rejected by the unchanged legacy Verifier's citation cap | COVERAGE | two questions; a post-freeze decision (per-part verification in the adapter, or a verifier page-cap decision) |
| L5 | An approval clause settled in single-clause form stays ambiguous inside a two-clause question | COVERAGE | two questions; routing rule scope |
| L6 | The College of Graduate Studies and Research directory row is not found by Institutional Navigation | COVERAGE | frozen lexicon; reopen by decision record only |
| L7 | Handoff rounds execute but no real handoff has yet been accepted on the real knowledge base (the targets found nothing or were out of scope) | COVERAGE | the "handoff completed" interface label is covered by unit tests only |
| L8 | The legacy model path has not been evaluated with a real model client; the application shows its no-key reply on that path | ENVIRONMENT | configure `GROQ_API_KEY` outside the repository to evaluate it |
| L9 | The evidence section label is the page's dominant section, so a chunk at the end of one section can carry the next section's title (page right) | DATA | ingestion metadata; a cache rebuild item |
| L10 | The legacy contact extractor answers only the contact clause of a question that pairs it with an off-topic clause | UX | legacy behaviour; the number is right |
| L11 | Aggregate answers quote the handbook verbatim, including its list glyphs ("o", "•") | UX | the interface does not alter answer text |

None of these is a wrong supported answer, a safety bypass, a mode-semantic failure or a security issue; a conservative
fallback is not an incorrect answer.
