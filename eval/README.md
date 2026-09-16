# Evaluation Harness

Part 1 was evaluated by hand: 56 questions asked in the browser, 60 screenshots,
each answer graded by a person. The Part 1 report lists that as a limitation.
This directory replaces it with something repeatable.

## Why running and scoring are separate

`run_eval.py` asks the questions and records raw answers. `score.py` grades a
recorded run. They are separate programs on purpose:

- A run is expensive. It loads two transformer models, builds or reads the FAISS
  index, and makes one Groq call per non-deterministic question.
- Scoring is free and the rubric will change as we learn what the model actually
  gets wrong.

Keeping them apart means a single run can be re-scored many times without
spending API calls, and a rubric change can never be confused with a model
change.

## Files

| File | Purpose |
|---|---|
| `golden_set.jsonl` | The questions, with expected answers verified against the handbook PDF |
| `run_eval.py` | Asks every question through `answer_question`, writes `runs/<timestamp>.jsonl` |
| `score.py` | Grades a run and prints the metric tables |
| `runs/` | Raw run outputs. Committed, so any number in the report can be traced back to the run that produced it |

## Usage

```bash
python eval/run_eval.py --tag baseline      # ask all questions, save the run
python eval/score.py                        # score the newest run
python eval/score.py --verbose              # per-case detail
python eval/score.py eval/runs/X.jsonl --csv out.csv
python eval/run_eval.py --limit 3           # smoke test
python eval/run_eval.py --category contact  # one category
```

Set `GROQ_API_KEY` in `.env` before a full run. Without it, only the
deterministic contact / count / date extractor paths produce answers and every
policy question is recorded as the missing-key message. `run_eval.py` prints a
warning in that state, and such a run must not be compared against Part 1.

## Golden set schema

One JSON object per line. `#` comments and blank lines are ignored.

| Field | Meaning |
|---|---|
| `id` | Stable identifier. Joins a run record to its case |
| `category` | Question family, e.g. `contact`, `date`, `count`, `governance` |
| `question` | Asked verbatim |
| `expected_route` | Intent `classify_query` should return. Measures bug 4 |
| `expected_pages` | Physical PDF pages that support the answer. Measures bug 3 |
| `expected_answer` | Reference answer in prose, for the report and the LLM judge |
| `grading` | `must_include`, `any_of`, `exact`, `refusal`, or `judge_only` |
| `must_include` | Required terms. A nested list means "any one of these is acceptable" |
| `must_not_include` | Terms whose presence makes the answer wrong |
| `source` | Exactly one of `qa_docx` (Q&A for ChatBots/UOS_Faculty_Handbook_QA.docx), `table_6-2` (report pilot table; its 3 overlaps with qa_docx are labelled qa_docx), `p65_txt` ("some q for the minichatbot v4.txt"), `reconstructed` (written from the handbook) |
| `part1_verdict` | Part 1 outcome if known: `correct`, `partial`, `incorrect`, or empty |
| `evidence_quote` | The handbook text the expected answer was taken from |
| `notes` | Why the case is written the way it is |

### Page numbers

`expected_pages` are **physical PDF page numbers, 1-based**, matching
`meta["page"]` produced in `handbook_bot/pdf_loader.py`. For this handbook the
printed `Page N` footer happens to equal the physical index, so the two
conventions agree. That is a property of this document, not a general rule.

### Grading modes

- **`must_include`** - every term must appear. All present is fully correct, some
  present is partially correct, none present is incorrect. This is what produces
  the partial-credit column that the Part 1 table has.
- **`any_of`** - at least one term must appear. Used where naming one valid item
  out of many *is* the task, e.g. "Name one standing committee". Requiring all of
  them would mis-score a correct answer as wrong.
- **`exact`** - normalised string equality. Reserved for answers with exactly one
  correct form.
- **`refusal`** - the answer must be the refusal string. Used for out-of-scope
  questions.
- **`judge_only`** - open-ended, no fair string rule exists. Excluded from the
  deterministic tables and left to the LLM judge. It is never guessed at.

### Term matching rules

`score.py` matches terms with three deliberate behaviours:

1. Terms of four characters or fewer are matched on **word boundaries**. Plain
   substring matching would count `25` as present in `2025` and `not` as present
   in `note`, turning wrong answers into passes.
2. Longer terms are matched as substrings, so ordinary prose is not punished for
   inflection or surrounding words.
3. Any term containing six or more digits is also compared digits-only, so
   `5050000` matches `+971 6 5050000` and `(06) 505-0000` alike.

## Metrics

Three tables are printed, and they are kept apart on purpose.

1. **Answer accuracy** - the Part 1 table: fully correct, partially correct,
   incorrect, correct refusals, strict success, weighted success (partial = 0.5).
2. **Route accuracy** - `expected_route` against `QAResult.query_type`. Isolates
   the intent classifier (bug 4) from everything downstream.
3. **Citation accuracy** - whether `QAResult.pages` contains an expected page.
   Where a citation is wrong, the scorer also reports whether the correct page
   *was* retrieved, which separates a citation bug (bug 3) from a retrieval miss.

Part 1 reported a single blended accuracy figure. That number cannot distinguish
"right answer, wrong citation" from "right answer, right citation", and for a
system whose selling point is verifiability that distinction is the point.

## Honesty rules

These are constraints on the harness, not style preferences.

- Everything in `expected_answer`, `expected_pages` and `evidence_quote` was read
  out of `data/UOS Faculty Handbook 25-26.pdf`. Nothing is inferred.
- `judge_only` cases are excluded from the deterministic totals and reported
  separately. A combined figure over graded and ungraded cases would be invented.
- `part1_verdict` is filled in only where the Part 1 outcome for that exact
  question is actually known. Where it is empty, the Part 1 vs Part 2 table
  simply covers fewer cases and says so.
- `source` separates recovered Part 1 questions from questions added in Part 2.
  The Part 1 headline figures apply only to the `part1` subset.

## Status

23 of the 56 Part 1 questions have been recovered and verified so far. The
remaining 33 are not yet written; the harness reports on what exists rather than
padding the set.
