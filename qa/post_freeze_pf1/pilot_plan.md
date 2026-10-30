# PF-2 ingestion plan (drafted in PF-1; nothing here is executed)

## Pilot documents (first controlled ingestion, five documents, 43 pages)

| # | Document | Pages used | Specialist | Why this one |
|---|---|---|---|---|
| 1 | Blackboard Ultra Faculty Short Guide (`guide_bb_faculty_short`) | 11 | Teaching, secondary Institutional | official ITC guide, clean extraction, the access and navigation knowledge the handbook lacks (login, MFA, course view, content, visibility) |
| 2 | Grade Transfer Dashboard Guide (`guide_grade_transfer_dashboard`) | English pages 1 to 3 of 6 | Teaching, secondary Institutional | the Blackboard-to-Banner workflow that the handbook's page 224 names without steps; the first cross-document case; exercises the bilingual page filter and the precedence rule (C1) |
| 3 | Blackboard Ultra Guide: Gradebook (`guide_bb_gradebook`) | 10 | Teaching | the grading workflow with the most faculty questions; textual, 28 numbered steps; exercises procedure chunking |
| 4 | Performance Appraisal Phase 2 User Guide, PDF (`guide_hr_appraisal_phase2`) | 13 | Faculty Services & HR | the only HR-portal procedure among the approved set; tables, validation rules and a troubleshooting table; confirm the Internal Use marking first |
| 5 | iThenticate Guidelines (`guide_ithenticate`) | English page 1 of 3 | Research | the only Research document; small; carries the one unresolved internal conflict (C2), which PF-2 must handle explicitly |

Diversity: four specialists, two bilingual documents, one HR portal, one dashboard, two Blackboard tools; sizes from 1 to 13
pages; every pilot document is official UOS material with GOOD or minor-cleanup extraction.

## Second batch (P1, 48 pages)

Blackboard Assignments and Tests (25), Communication Tools (5), Collaborate Ultra instructor guide (4), Attendance tool (3),
Grade categories (1 bilingual page, split first), plus the remaining English pages of the pilot documents if PF-2 chose
page subsets. Microsoft Forms (DOCX, 3 pages) and the MyUOS announcement guide (7) are P2.

## Held for review (not ingested until the owner decides)

Akindi guide (issuing unit unverified), Business Mission guide (unverified, DOCX, faculty-relevant if confirmed), DocuSign
note (four lines, unattributed), the 2026 Blackboard compilation (not UOS-authored), the McGraw Hill Connect sync guide
and the Coursera learner guide (vendor material), the Turnitin instructor guide (Arabic), the two student guides (audience).
Rejected: the Coursera administrator guide (vendor, administrator audience), the Connect student exam support sheet
(vendor, students), the Turnitin student guide (Arabic, students, misnamed), the appraisal DOCX (duplicate of the PDF).

## What PF-2 must build before the pilot can be measured (smallest safe set)

1. Multi-document loading: iterate the registry's records instead of the first PDF in `data/`; one `pages` list per
   document with `source_id` stamped before chunking; a corpus-level cache key (hash of the ordered source ids, versions
   and file hashes) replacing the single-file key.
2. Page-language filter: skip or tag pages whose Arabic-script share exceeds a threshold, so bilingual guides contribute
   their English pages only (Grade Transfer pages 4 to 6, iThenticate page 2, the grade-category page needs an in-page split).
3. Metadata: the additions listed in `gap_mapping.json` (canonical_title, document_type, authority_level, system,
   department, language, publication_date or DATE UNKNOWN, source_hash); handbook records unchanged.
4. Procedure chunking: `procedure_step` and `procedure_window` chunk types beside the existing ones; no change to the
   handbook's chunks (verify by comparing the handbook's chunk list before and after).
5. Citations: one document chip per distinct source (`canonical_title · Page n`); the Developer panel shows source ids.
6. Specialist reach: decide, by a decision record, how the frozen specialists see the new sources. Options: (a) a
   scope-extension seam in non-frozen code (`retrieval.py` gains an OR-of-scopes or a second scope per specialist applied by
   `plan_f_runtime`), or (b) a formal reopening of each specialist's `compile_scope` call. Until one of these exists the
   new documents are reachable by the legacy path only, which is still a valid first measurement.
7. Evaluation: ingest the pilot, rebuild the cache, re-run the frozen I-8 harness and `compare_runs.py` to prove the 192
   frozen answers are unchanged, then run the PF-2 draft questions (`pf2_question_draft.json`) and review every served
   answer against its page as in I-8. Report coverage and precision separately; do not claim improvement before that.

## Precedence for PF-2 (proposal, not implemented)

Policy facts from the Faculty Handbook; interface steps, URLs and system contacts from the newest official ITC or HR
guide; a third-party document never answers a UOS policy or procedure; an official-versus-official conflict is reported,
not merged (C1 is served as two parts, C2 is excluded or marked disputed).

## Storage

Mixed: commit the approved official guides that PF-2 ingests under `data/` with their registry records; keep vendor,
student, Arabic-only, unverified DOCX files and the RAR archive outside the repository.
