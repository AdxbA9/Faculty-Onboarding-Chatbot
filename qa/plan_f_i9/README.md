# Runtime freeze evidence (runtime integration, phase I-9)

The freeze itself is documented in `docs/RUNTIME_FREEZE_I9.md` (identifier `runtime-freeze-i9`, contracts,
reason-code vocabularies, frozen metrics, known limitations, post-freeze change policy) and recorded as D-108 in
`docs/DECISIONS.md`. This folder holds the reproducible evidence and the tools a later comparison needs.

| File | Purpose |
|---|---|
| `make_manifest.py` | writes the freeze manifest: SHA-256 of the integration modules, the seam and interface files, the six frozen specialists and their helpers, the evaluation set and the QA artifacts, the v12 knowledge-base cache files; the git HEAD, branch and subject; Python and key package versions; the evaluation-set size; the test counts passed on the command line; the frozen I-8 metrics. No absolute path. |
| `freeze_manifest.json` | the manifest written at the freeze commit |
| `concurrency_smoke.py` | eight threads ask the ACTIVE seam at once on a fresh runtime state (each of eight representative questions twice): the registry must be built exactly once, every concurrent answer must equal the sequential answer of the same question, the recorded diagnostics must be one per call with no foreign pages, no exception, 0 model calls |

Run from the repository root with the project interpreter:

    python -X utf8 qa/plan_f_i9/make_manifest.py . qa/plan_f_i9/freeze_manifest.json --tests "<gate>,<full>"
    python -X utf8 qa/plan_f_i9/concurrency_smoke.py . 8

Reproducing the frozen evaluation (the I-8 harness on the cached knowledge base, about 16 minutes on CPU):

    python -X utf8 qa/plan_f_i8/master_eval.py . <output dir>
    python -X utf8 qa/plan_f_i8/compare_runs.py qa/plan_f_i8/results_2026-10-30.json <output dir>/i8_results.json
    python -X utf8 qa/plan_f_i8/analyze.py <output dir>/i8_results.json qa/plan_f_i8/review.json
    python -X utf8 qa/plan_f_i8/legacy_check.py .

A comparison after post-freeze work is expected to show the same served or fallback decision, answer text, pages,
routing code, specialist count, handoff outcome and aggregation outcome for every question, and SHADOW identical to the
direct legacy call for all 192; wall-clock timings vary. Any other difference is a behaviour change and needs the
before-and-after record the change policy asks for.

The I-9 rerun of 2026-10-30 (recorded in the STATUS entry and the freeze document) matched the committed I-8 results on
every compared field for all 192 questions in ACTIVE and SHADOW, with 24 of 24 determinism questions stable over four
observations, the fresh-process LEGACY check clean, the concurrency smoke clean, and the app smoke in ACTIVE, SHADOW and
LEGACY passing; the rerun's own results file is not committed (it duplicates the frozen one).
