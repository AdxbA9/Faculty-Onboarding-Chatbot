# Plan F runtime integration, phase I-7: interface compatibility review and defect log

Date: 2026-10-30. Branch `feature/remaining-specialists`, starting commit `8331475`.

Scope: the NiceGUI chat page (`ui/chat.py`, `ui/styles.py`) now presents every kind of
answer the pipeline can return (legacy extractor, legacy model path, refusal, single
verified specialist finding, verified multi-finding aggregate, routing-refined answer,
handoff-derived answer, legacy fallback after a Plan F attempt, shadow run) from the
per-turn diagnostics of `ui.pipeline.ask_with_diagnostics`, through one pure helper
module, `ui/presentation.py` (no NiceGUI import, no Plan F module import, no decision
of its own). Nothing in the pipeline, the adapter, the aggregator, the routing, the
handoff round, the Coordinator or the four frozen specialists changed.

What a normal user sees: the answer, rendered as Markdown with technical identifiers kept
intact; one chip for the handbook, one chip per cited page with the page's section, an
OCR marker when image text was used; and, only when the verified specialist path produced
the answer, a green "Verified · <domain>" chip naming the handbook domain(s). Reason codes,
specialist ids, routing and handoff records, timings and error categories appear only in
Developer mode.

## Defect log

Severity: BLOCKING (wrong or unsafe for every user), MAJOR (misleading to a user or a
developer), MINOR (incorrect but contained), COSMETIC (appearance only).

| # | Issue (before) | Severity | Fix | Verification |
|---|---|---|---|---|
| 1 | The Developer panel labelled every answer without a model call "Deterministic extractor", so a Plan F answer (verified specialist finding or aggregate) was attributed to the legacy extractor. | MAJOR | "Answered by" is derived from the turn's own diagnostics (served path, Plan F path, routing and handoff outcome) and from the legacy result's route: "Verified specialist answer", "Verified multi-part answer", suffixes "(routing refined)" and "(handoff completed)", "Legacy deterministic extractor", "Legacy model path (no model client configured)", "Legacy model synthesis (model)", "Refusal (no supporting evidence)", "Legacy fallback after Plan F attempt: ...", "... · Plan F ran in shadow". The panel also shows Mode, Served path, Plan F path and Plan F outcome. | ACTIVE browser run: HR telephone "Verified specialist answer"; teaching load plus sick leave "Verified multi-part answer"; conference travel "Verified specialist answer (routing refined)"; "What does HR do?" "Legacy fallback after Plan F attempt: Legacy model path (no model client configured)". LEGACY and SHADOW runs never show a specialist label. Unit tests `test_answer_kind_never_claims_a_specialist_answer_the_user_did_not_get`, `test_legacy_kinds_are_read_from_the_result_itself`. |
| 2 | For a Plan F answer the Timing section printed "Retrieval 0.0 ms", "Rerank 0.0 ms" (keys the specialist path never records, defaulted to zero) and the Retrieval section printed "Candidates retrieved 1 / Chunks reranked 1" (the adapter's evidence count, not a candidate count). | MAJOR | Performance rows come from the recorded timings only. A Plan F answer shows "Specialist execution" (the specialists' own run, retrieval and rerank included), "Retrieval N/A", "Rerank N/A", routing refinement, findings verification, aggregation and handoff round when recorded, legacy Verifier, generation, "Plan F total", "Legacy total N/A". Candidate and rerank counts are shown for legacy answers only; Plan F answers show "Evidence items" and "Answer parts". Anything unrecorded is N/A; a recorded value below 0.05 ms is "<0.1 ms", never a bare zero. | ACTIVE browser run (final code): conference travel "Specialist execution 716.7 ms, Retrieval N/A, Rerank N/A, Routing refinement 0.3 ms, Findings verification 0.7 ms, Legacy Verifier 0.1 ms, Plan F total 753.6 ms, Legacy total N/A"; aggregate "Routing refinement <0.1 ms, Aggregation 4.4 ms". Unit test `test_plan_f_timings_come_from_the_runtime_and_unrecorded_stages_are_na`. |
| 3 | On a legacy fallback after a Plan F attempt the panel showed the legacy timings only; the cost of the attempt and the real wall time were invisible. | MAJOR | Fallback rows: "Plan F attempt" (the adapter's `plan_f_ms`), legacy Retrieval, Rerank, Generation, "Legacy total", "End-to-end" (wall time measured in the UI around the whole call, every mode), "Specialist registry built (ms) / reused". | ACTIVE "What does HR do?": Plan F attempt 450.2 ms, Legacy total 768.5 ms, End-to-end 1219.1 ms, registry reused. First question of the ACTIVE run: End-to-end 1091.0 ms with "Specialist registry built (702.4 ms)". SHADOW: "Plan F attempt (shadow)" row. Unit test `test_fallback_timing_separates_the_plan_f_attempt_from_the_legacy_time`. |
| 4 | Chips duplicated or looked duplicated: "Page 54" next to a section chip reading "Page 54" (the legacy `best_section` placeholder), and one section chip for an answer that cited several pages with different sections. | MINOR | One chip per distinct cited page in the answer's own order, carrying that page's section from the evidence ("Page 54 · Contact Information"); placeholder sections ("Page N", "Greeting", ...) are never shown; two evidence items on the same page with the same section give one chip, different sections on one page give one chip each; the handbook title once in front. | LEGACY HR telephone: chips "UOS Faculty Handbook 2025/26", "Page 54 · Contact Information" (before: "Page 54", "Page 54"). ACTIVE aggregate: "Page 96 · Workload Allocation Model (WLAM)", "Page 91 · Workload Allocation Model (WLAM)", "Page 107 · Leaves and Absences". Unit tests `test_a_page_placeholder_section_never_becomes_a_duplicate_looking_chip`, `test_same_page_evidence_with_the_same_section_gives_one_chip_and_different_sections_two`, `test_multi_page_evidence_keeps_every_page_in_the_answers_order`. |
| 5 | The no-key reply "GROQ_API_KEY is not set..." rendered through Markdown as "GROQ<em>API</em>KEY". | MINOR | `presentation.escape_identifiers` escapes underscores inside words (outside code spans) before `ui.markdown`. | LEGACY, SHADOW and ACTIVE runs: rendered HTML of the bubble is `<p>GROQ_API_KEY is not set. Please set it, then reload the app.</p>`, no `<em>`. Unit test `test_technical_identifiers_keep_their_underscores`. |
| 6 | A pipeline exception reached the user verbatim ("Sorry, I ran into an error: {exc}"); a knowledge-base load failure put the exception text in the status banner. | MAJOR | Fixed safe reply ("Sorry, something went wrong while answering that. Please try again."); the Developer panel shows the exception class only ("Error category"); a load failure shows "Could not load the handbook (<class>). Check the server log, then reload the app." and logs the traceback server-side. | No exception occurred in any browser run (none reproducible). Unit test `test_fallback_and_error_panels` (an exception carrying a local path and a key-shaped token leaves neither in the panel); `test_chat_module_consumes_the_turns_diagnostics_and_stays_free_of_runtime_names` checks the source. |
| 7 | While waiting, the assistant bubble showed three dots only. | MINOR | "Searching official university sources…" next to the dots, `role="status"`, `aria-live="polite"`. | ACTIVE annual-leave question: a DOM probe during the wait read the text with `role=status`, `aria-live=polite`. |
| 8 | The Developer mode switch, the message box, the send button and the theme button had no accessible names. | MINOR | `aria-label` on each (`Developer mode`, `Message the Onboarding Chatbot`, `Send message`, `Toggle light or dark mode`). | Accessibility tree of the running page: `switch "Developer mode"`, `textbox "Message the Onboarding Chatbot"`, `button "Send message"`, `button "Toggle light or dark mode"`. |
| 9 | Narrow screens (at and below 820 px): a short user message wrapped mid-word ("Hell / o") because the width cap sat on the inline-block bubble inside a shrink-to-fit flex wrapper. | MINOR | The cap moved to the wrapper (86% at 820 px, 92% at 540 px); the bubble keeps `max-width: 100%`. | 375 px viewport before the fix: "Hell / o"; after the fix: "Hello" on one line (ACTIVE run). |
| 10 | Developer rows at narrow widths clipped long values with an ellipsis (nowrap, right-aligned). | COSMETIC | At 540 px and below the rows stack (label above value, wrapping allowed); chips may take the full width. | 375 px viewport: every value readable (fallback reason, answered-by label, section names). |
| 11 | A refusal ("I do not have this information.") that retrieved candidates but cited nothing showed the handbook chip, presenting the handbook as the source of an answer it did not give. | MINOR | The handbook chip appears only when a page or a section is cited. | ACTIVE "What is the capital of France?": no chips; Developer panel "Pages cited N/A, Document N/A". Unit test `test_chips_without_pages_fall_back_to_a_real_section_and_mark_ocr_evidence`. |
| 12 | The Evidence "Sections" row repeated one section per page ("WLAM; WLAM; Leaves and Absences"). | COSMETIC | Distinct sections only. | Final ACTIVE aggregate: "Workload Allocation Model (WLAM); Leaves and Absences". |

Observed and left as is (recorded for I-8): aggregate answers quote the handbook verbatim,
so the handbook's own list glyphs ("o", "•") open the parts; the interface does not alter
answer text. The greeting's "Legacy total 0.0 ms" is the pipeline's recorded value. Reason
codes and specialist ids in the Developer panel are intended there and nowhere else.

## Browser matrix

`app.py` launched locally with the switch in the process environment only (never `.env`),
no model key configured, the pipeline's debug diagnostics logged by the launcher, driven in
the desktop app's browser at `http://localhost:8501` with Developer mode on; the real cached
knowledge base (272 pages, 13,688 chunks). Every answer below rendered without an
exception, with the chips and Developer sections stated; no developer vocabulary (reason
codes, specialist ids, stack traces, cache internals, handoff structures) appears outside the
Developer panel.

| Mode | Question | Visible result | Developer panel (selected rows) |
|---|---|---|---|
| LEGACY | Hello | greeting, no chips | Mode LEGACY, Served path Legacy, Answered by Legacy answer; no Plan F row anywhere |
| LEGACY | What is the phone number of the Human Resources Department? | "The phone number is +971 6 5050023."; chips handbook, Page 54 · Contact Information | Legacy deterministic extractor; Candidates 24, reranked 5; Retrieval 308.8 ms, Rerank 240.7 ms, Legacy total 555.4 ms, End-to-end 555.5 ms |
| LEGACY | What is the annual leave entitlement for faculty? | no-key reply with intact underscores; chips handbook, Page 107 · Leaves and Absences | Legacy model path (no model client configured); Generation N/A; Legacy LLM calls 0 |
| LEGACY | What is the capital of France? | "I do not have this information."; chips handbook only (fixed afterwards: none, see defect 11) | Refusal (no supporting evidence); Pages cited N/A |
| SHADOW | HR telephone | identical to LEGACY (same text, same chips, no domain chip) | Served path Legacy (Plan F in shadow); Plan F outcome PLAN_F_ACCEPTED, single finding; Plan F attempt (shadow) 29.5 ms, registry built (712.2 ms); Plan F LLM calls 0 |
| SHADOW | teaching load plus sick leave | legacy no-key reply; legacy chips (Pages 94, 96, 107) | Plan F path multi-finding aggregate, PLAN_F_ACCEPTED, Aggregation 2 of 2, 4 considered 3 rendered; attempt 1528.5 ms |
| SHADOW | What does HR do? | legacy no-key reply; Page 61 · Recruitment and Hiring | PLAN_F_UNSAFE_RESULT with the fallback reason; Verification rows N/A; attempt 460.5 ms |
| ACTIVE | HR telephone (single finding) | directory row "Human Resources +971 6 5050023 +971 6 5585200"; chips Verified · Institutional Navigation, handbook, Page 54 · Contact Information | Verified specialist answer; Routing ROUTE_UNCHANGED, 1 call; Verification accepted, complete, contact; Plan F total 28.7 ms; End-to-end 1091.0 ms (registry built 702.4 ms) |
| ACTIVE | teaching load plus sick leave (aggregate) | three verbatim parts; chips Verified · Teaching & Learning, Faculty Services & HR, handbook, Pages 96, 91, 107 with sections | Verified multi-part answer; Aggregation 2 of 2, accepted, 4 considered 3 rendered; Answer parts 3; Specialist execution 1514.6 ms; Plan F LLM calls 0 |
| ACTIVE | Who approves conference travel? (routing refined) | page-117 rule; chips Verified · Research & Innovation, handbook, Page 117 · Participation in Conferences | Verified specialist answer (routing refined); Coordinator route Research & Innovation, Institutional Navigation; Final Research & Innovation; ROUTE_REFINED_BY_REQUEST_KIND; 1 call |
| ACTIVE | What does HR do? (known HR wrong-support case) | legacy no-key reply; no domain chip | Legacy fallback after Plan F attempt; PLAN_F_UNSAFE_RESULT; Fallback reason shown; Plan F attempt 450.2 ms, Legacy total 768.5 ms, End-to-end 1219.1 ms |
| ACTIVE | What is the annual leave entitlement for faculty? (partial) | legacy no-key reply; Page 107 · Leaves and Absences | PLAN_F_UNSAFE_RESULT, Fallback reason "status partial"; Plan F attempt 749.0 ms |
| ACTIVE | Which office handles faculty housing? (ambiguous routing) | legacy no-key reply; Pages 29, 253 | PLAN_F_REJECTED_BY_ROUTING, ROUTE_AMBIGUOUS, Specialist calls 0; Plan F attempt 0.7 ms |
| ACTIVE | HR telephone plus Passport Unit fax (multi-page) | two directory rows; chips Verified · Institutional Navigation, handbook, Page 54 · Contact Information, Page 53 · Contact Information | Verified multi-part answer (routing refined); ROUTE_CLAUSES_SCOPED; Aggregation 2 of 2; Plan F total 3.7 ms |
| ACTIVE | Where is the Research Institute? (real handoff) | legacy no-key reply; Pages 153, 157, 182 | Handoff section: Attempted yes, Source Research & Innovation, Target Institutional Navigation, Depth 1, Visited both, outcome as recorded, Timing 0.6 to 0.8 ms; PLAN_F_REJECTED_BY_AGGREGATOR (MISSING_REQUIRED_CLAUSE: the target found nothing); 2 calls |
| ACTIVE | What is the capital of France? | "I do not have this information."; no chips | PLAN_F_NOT_APPLICABLE; Coordinator route N/A; Pages cited N/A |

Sequential-turn isolation: after the fallbacks, the first answer's panel still read "Served
path Plan F, Verified specialist answer" and the fallback panels read "Legacy (fallback after
Plan F attempt)"; each message owns a copy of its own diagnostics
(`presentation.assistant_message`; unit test `test_each_turn_owns_a_copy_of_its_diagnostics`).

Responsive review: desktop (1024 px pane) in dark and light themes; 375 x 812 px (mobile
preset): sidebar hidden, topbar reduced to title, switch and theme button, chips wrap,
Developer rows stack, composer full width, no horizontal scroll; the user-bubble wrap defect
(9) found and fixed there.

Loading state: "Searching official university sources…" observed in the DOM during a
question (role status, aria-live polite).

Diagnostics hygiene: the launcher's logs of the three runs carry no question or answer text
(0 matches for the question phrases, the telephone number and the no-key reply).

Screenshots were taken in the desktop app's browser pane and kept out of the repository.

## Automated checks

`tests/test_ui_presentation.py` (25 tests): identifier escaping, answer-kind labels for
every served path, served and path labels, domain labels only for a served specialist
answer, chip construction (document once, page with section, placeholder sections,
same-page duplicates, multi-page order, refusal without chips, OCR marker, truncation),
N/A timing rules and the fallback split, Developer sections for LEGACY, ACTIVE single,
aggregate with handoff, fallback, error and SHADOW turns, per-turn diagnostic isolation,
and a source check that `ui/chat.py` consumes `ask_with_diagnostics`, uses the helpers,
contains no exception text and none of the runtime module names, while
`ui/presentation.py` imports neither NiceGUI nor any Plan F module.

Targeted gate (the I-6 gate plus this file, 20 files): 1,416 collected, 1,416 passed, 0 failed, 0 skipped, 0 xfail (82 s).
Full suite: 6,670 collected, 6,670 passed, 0 failed, 0 skipped, 0 xfail (221 s; 6,645 before this phase plus the 25 new tests). `pip check`: no broken requirements.

## Backend regression confirmation

The I-6 real-knowledge-base evaluation (`qa/plan_f_i6/routing_eval.py`: 126 questions
in SHADOW and 126 in ACTIVE, the I-3/I-4 corpus, the I-4 probes, the I-5 aggregation set
and the I-6 routing set) was re-run on this phase's code and compared record by record
with the committed I-6 results (`qa/plan_f_i6/results_2026-10-30.json`): 252 of 252
records equal in served path, outcome, answer text, pages, routing reason, handoff outcome
and specialist calls; ACTIVE served by Plan F 56 records (55 unique questions), every
answer and page list byte-identical to I-6; SHADOW 126 of 126 identical to the direct
legacy call; the known HR wrong-support case still falls back (`PLAN_F_UNSAFE_RESULT`, 1
call) and the ambiguous housing question still runs no specialist
(`PLAN_F_REJECTED_BY_ROUTING`); model calls 0 on every record; exceptions 0. Expected,
since no file under `handbook_bot/` or `ui/pipeline.py` changed; the re-run output stays
outside the repository.

## Files

New: `ui/presentation.py`, `tests/test_ui_presentation.py`, `qa/plan_f_i7/README.md`.
Changed: `ui/chat.py` (diagnostics per turn, helper calls, safe errors, loading text,
accessibility attributes), `ui/styles.py` (chip variants, loading text, narrow-screen
rules), `docs/STATUS.md`. No change under `handbook_bot/`.
