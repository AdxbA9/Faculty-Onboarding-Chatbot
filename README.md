# UOS Faculty Onboarding Chatbot

A retrieval-augmented (RAG) chatbot that answers questions about the
**University of Sharjah Faculty Handbook**. It reads the handbook PDF,
builds a searchable knowledge base, and answers questions with direct
page citations.

Built with `sentence-transformers`, `FAISS`, a cross-encoder reranker,
and a Groq-hosted LLM (Llama-3.3-70B by default). The UI is a modern
chatbot interface built with [NiceGUI](https://nicegui.io/) — single
Python process, no Node, no Docker, no build step.

---

## Features

- **Professional web interface** (PF-10) — NiceGUI page with a
  light-first design system in the University of Sharjah green (dark
  mode on request), a header with plain-language service status, a
  sidebar with the saved conversations of this browser, a first-use
  screen that explains the scope and offers verified example questions,
  answer cards with a Sources row and an expandable "Supporting
  evidence" list, labelled refusal and service notices, a composer with
  Enter to send / Shift+Enter for a new line, and a copy-answer button.
  Answers are rendered as escaped Markdown with filtered links; the
  layout is direction-aware for the coming Arabic phase.
- **Arabic and English (bilingual) support** (PF-11) — ask in Arabic,
  English or a mix of both. An Arabic question is rewritten deterministically
  into the English retrieval question the documents answer (a bilingual
  domain lexicon; one bounded model request only when the lexicon covers
  too little), goes through the unchanged English pipeline and its
  verification, and the verified English answer is translated by one
  model request and checked (numbers, contacts, acronyms, negation,
  structure) before it is shown; when the check fails or no model is
  configured, the Arabic lead-in quotes the verified English text
  verbatim. The interface switches between English and Arabic
  (right-to-left) without losing the conversation; the answer language
  follows the question unless the user chooses otherwise. See
  "Arabic support" below.
- **Voice input in English and Arabic** (PF-12) — a microphone button in
  the composer uses the browser's Web Speech API (Chrome, Edge, Safari;
  not Firefox) to turn one spoken question into text. The transcript is
  appended to the message box for review and editing; nothing is sent
  until you press Send or Enter. The voice language follows the interface
  language unless you choose English or Arabic in the composer footer.
  The browser asks for microphone permission once; the audio is processed
  by the browser vendor's speech service, not by this application, which
  keeps only the text you see. See "Voice input" below.
- **One platform, seven pages** (PF-15) — every page shares the same
  sidebar navigation (Assistant chat, Onboarding checklist, Guided
  procedures, Resources & documents, Conversation history, Guest profile,
  Settings), which becomes a slide-in menu on phones and tablets. The
  chat stays the start page. Settings add a larger text size and reduced
  motion; the guest profile shows what this browser has saved and an
  optional display name. See "Navigation, settings, guest profile and
  history" below.
- **Developer mode** — open the page as `http://localhost:8501/?dev=1`
  (or set `APP_DEVELOPER_MODE=1`) to get a Developer switch in the
  header that reveals a per-answer panel: mode, served path, routing,
  verification, evidence, timings, model calls and knowledge-base
  statistics. Ordinary users never see it.
- **Grounded answers** — every answer is tied to handbook pages; the
  bot refuses to answer when evidence is weak.
- **Hybrid retrieval** — dense FAISS search + lexical overlap +
  intent-aware routing boosts.
- **Cross-encoder reranking** for higher-precision top-k.
- **Table-aware chunking** — paragraph chunks plus row and row-window
  chunks so tables (contacts, calendar, program lists) retrieve
  cleanly.
- **Optional image OCR** — set `ENABLE_OCR=1` and the ingestion pass
  will also OCR embedded images and text-sparse pages using
  `rapidocr-onnxruntime` (pure pip, no system Tesseract). Recovered
  text is indexed with an `image_ocr` chunk type and surfaced in the
  source chips.
- **Deterministic extractors** for contact / date / count questions,
  so those answers are 100% grounded (zero LLM hallucination risk).
- **Answer verification** — a lightweight check rejects answers that
  don't appear in the retrieved context.
- **Embedding cache** — first run embeds once, subsequent runs are
  fast.

---

- **Saved conversations, settings and guest sessions** (PF-13) — every
  conversation is stored on the server in SQLite (standard library, one
  file under `var/`) for the browser that held it, listed in the sidebar,
  reopened from its stored record (never re-asked), renamed or deleted
  with confirmation, and continued after a refresh or a server restart.
  Arabic answers keep their English intermediate and translation state;
  evidence and citations are stored with the answer. Settings (interface
  language, answer language, theme, voice language) persist. Identity is
  a signed HttpOnly session cookie, hashed in the database and checked on
  every query; it is a guest identity, not a university login. Backups,
  integrity checks and versioned migrations come with a small command
  line. See "Saved conversations and settings" below.
- **Onboarding checklist, guided procedures and resource finder** (PF-14)
  — twelve source-grounded onboarding activities with persisted,
  self-reported progress; seven documented procedures shown step by step
  as exact guide text; a bilingual finder for the approved documents,
  departments (contacts as printed) and checked official pages; all three
  connected to each other and to the chat, whose unchanged pipeline
  answers contextual questions. See "Onboarding checklist, guided
  procedures and resource finder" below.

## Folder structure

```
UOS_Handbook_Chatbot/
├── app.py                  # Single entry point — just `python app.py`
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── SETUP.bat               # Windows: one-time setup (environment in %LOCALAPPDATA%\fobv, tested versions)
├── START_CHATBOT.bat       # Windows: start the chatbot
├── constraints.txt         # The library versions the final test run used
├── start.ps1               # Setup + run from PowerShell
├── start.sh                # One-shot setup + run for macOS / Linux
│
├── eval/                   # Golden question set, runners (run_ui_eval.py = the page's own path) and scorer
├── tests/                  # pytest suite
├── qa/                     # QA scripts and recorded results of every phase
├── archive/                # Earlier prototypes (MiniV1-V5) and pre-upgrade backups; not used by the app
│
├── assets/
│   └── logo.png            # UOS logo served as a static asset
│
├── data/
│   └── UOS Faculty Handbook 25-26.pdf
│
├── cache/                  # Auto-generated embeddings cache (gitignored)
│
├── var/                    # Saved conversations (SQLite) + cookie secret (gitignored)
│
├── onboarding/             # PF-14: task/procedure/resource catalog, progress arithmetic, finder, URL policy
│
├── persistence/            # SQLite layer: schema, migrations, store, backup CLI
│   ├── db.py               # Connection settings, versioned migrations, backup/restore
│   ├── store.py            # Owner-scoped persistence service used by the page
│   ├── records.py          # Answer <-> rows (faithful replay)
│   ├── secret.py           # Stable cookie-signing secret
│   └── manage.py           # python -m persistence.manage check|backup|restore|...
│
├── android/                # Android app: a WebView shell around the hosted website (see "Android app")
│
├── ui/                     # UI layer (NiceGUI)
│   ├── __init__.py
│   ├── chat.py             # Page layout + interaction logic
│   ├── onboarding.py       # Onboarding dashboard, procedure navigator, resource finder
│   ├── styles.py           # Theme + CSS + inline SVG icons
│   └── pipeline.py         # Thin wrapper over the backend
│
└── handbook_bot/           # Backend RAG package
    ├── __init__.py
    ├── config.py           # Tunable parameters
    ├── text_utils.py       # Regexes, tokenizer, normalisers
    ├── pdf_loader.py       # Layout-aware PDF extraction + OCR
    ├── chunking.py         # Paragraph + row + row_window + image_ocr
    ├── retrieval.py        # FAISS, lexical scoring, query routing
    ├── extractors.py       # Deterministic contact/count/date answers
    ├── qa.py               # Prompt, LLM call, verification, pipeline
    └── knowledge_base.py   # One-shot bootstrap helper
```

---

## Requirements

- **Python 3.10, 3.11, or 3.12** (3.11 or 3.12 recommended)
- A free [Groq API key](https://console.groq.com/keys)
- Internet connection on first run (to download the embedding and
  reranker models from Hugging Face, about ~500 MB; plus ~100 MB of
  OCR models if `ENABLE_OCR=1`)
- For the test suite: `pytest` and `pytest-asyncio` (both in
  `requirements.txt`); the page tests use NiceGUI's simulated user and
  need no browser. `qa/pf10_ui/browser_qa.py` drives an installed
  Chrome or Edge for real-browser checks.

---

## Quick start — Windows

1. Install **Python 3.11** from https://www.python.org/downloads/ (tick "Add python.exe to PATH").
2. Double-click **`SETUP.bat`**. It creates the Python environment, installs the tested library versions
   (`constraints.txt`) and creates `.env`. The first time this downloads about 2 GB.
3. Open `.env` in Notepad and paste your `GROQ_API_KEY`.
4. Double-click **`START_CHATBOT.bat`**, wait for the "ready" line, then open **http://localhost:8501**.

`.\start.ps1` does steps 2 and 4 from PowerShell.

### "File path too long"

Windows refuses paths longer than 260 characters, and the libraries (torch, transformers, onnxruntime) contain files
that need 145 characters below the environment folder. An environment created inside the project (`.venv`) therefore
fails as soon as the project sits in a deep folder such as `C:\Users\<name>\OneDrive - ...\Desktop\...`.

- `SETUP.bat` avoids this: the environment is always created in `%LOCALAPPDATA%\fobv` (about 32 characters), wherever
  the project folder is. Do not create a `.venv` inside the project on Windows.
- Unzip the project into a short folder such as `C:\fob`. The longest path inside the project itself is under 100
  characters.
- The Android build output (`android\app\build`) has the longest paths in the repository; it is generated, ignored by
  git and never part of the shared zip.

**Manual install** (if you prefer your own commands):

```powershell
py -3.11 -m venv "$env:LOCALAPPDATA\fobv"
& "$env:LOCALAPPDATA\fobv\Scripts\python.exe" -m pip install -r requirements.txt -c constraints.txt
copy .env.example .env
# Open .env in Notepad and paste your GROQ_API_KEY
$env:PLAN_F_ENABLED = "1"; $env:PLAN_F_GUIDE_SCOPES = "1"
& "$env:LOCALAPPDATA\fobv\Scripts\python.exe" -X utf8 app.py
```

---

## Quick start — macOS / Linux

**Option A — helper script**:

```bash
chmod +x start.sh
./start.sh
```

**Option B — manual**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Open .env and paste your GROQ_API_KEY
python app.py
```

Then open **http://localhost:8501**.

---

## Setting the `GROQ_API_KEY`

Get a free key from https://console.groq.com/keys — it looks like
`gsk_xxxxxxxxxxxxxxxxxxx`.

Preferred — edit `.env`:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxx
```

Alternative — export it in the current shell (one session only):

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxx"
```

```bash
# macOS / Linux
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxx"
```

---

## Arabic support

The approved documents are English; the chatbot answers Arabic questions
through its English pipeline (decision record D-122):

1. **Detection** — script-based, tolerant of short and code-switched
   questions (`handbook_bot/arabic_text.py`).
2. **Query transformation** — the normalised Arabic question is rewritten
   into an English retrieval question with the bilingual domain lexicon
   (`handbook_bot/arabic_lexicon.py`, `handbook_bot/arabic_query.py`);
   Latin words, acronyms and numbers pass through unchanged, Arabic-Indic
   digits become Western digits, clause structure is kept. When the
   lexicon covers less than `ARABIC_QUERY_MIN_COVERAGE` of the question
   and a model key is configured, ONE translation request may replace
   the lexicon query (`ARABIC_QUERY_MODEL_FALLBACK`); the record of the
   transformation travels with the answer (Developer panel, "Language").
3. **Answering** — the English query goes through the unchanged
   Coordinator, specialists, retrieval, verification and fallback.
4. **Arabic presentation** — refusals, greetings and service notices are
   fixed Arabic sentences; a real answer is translated by ONE model
   request and accepted only when the deterministic check finds every
   number, telephone, e-mail, URL, acronym and list item preserved and
   no negation dropped (`handbook_bot/arabic_answers.py`); otherwise the
   page shows an Arabic lead-in naming the source and page and the
   verified English text verbatim (`ARABIC_ANSWER_TRANSLATION=0` forces
   this). The page says which of the two it is showing and offers the
   verified English source answer under the translation.

Interface language: the header button switches between English and
العربية (the page opens in Arabic when the browser prefers Arabic, or
with `?lang=ar`). Answer language: "Same as the question" by default,
English or Arabic on request (composer footer). Western digits are kept in
answers by design. `ARABIC_SUPPORT=0` switches the whole layer off.

Evaluation tooling: `qa/pf11_arabic/` (paired question set, retrieval
probe, offline evaluation, bounded live smoke test, bilingual browser QA).

---

## Voice input

Press the microphone (or focus it and press Enter) to dictate one question.
While listening the button pulses red and shows Stop; a strip under the
composer shows what is being recognised; × or Escape cancels. When the
recogniser finishes, the text is added to the message box after anything
already typed, the box gets the focus, and the help line asks you to check
and send. Send and Enter work as usual; Shift+Enter still adds a new line.

- **Languages**: the voice language follows the interface language
  (English → `en-US`, Arabic → `ar-AE`); the composer footer lets you pick
  the other one without restarting. A mixed English/Arabic utterance is
  recognised in the chosen language's service and may need correction.
- **Browsers**: Chrome and Edge (Windows, macOS, Android), Safari (macOS,
  iOS). Firefox has no Web Speech API; the button shows that voice input is
  not available and typing works as before.
- **Privacy**: the browser's speech service (Google for Chrome, Microsoft
  for Edge, Apple for Safari) may process the audio on its servers. The
  application never receives audio, stores none and logs none; it keeps
  only the text in the message box. The page states this on first use.
- **Permission**: the browser asks once per site. If it was refused, allow
  the microphone in the browser's site settings for `127.0.0.1:8501` (the
  app's default address is a secure context; another host needs HTTPS).
- **Errors** (no microphone, no speech heard, service unreachable, time
  limit, language not available) are shown as plain sentences in the
  interface language; the text input is always available.

---

## Saved conversations and settings

Conversations are saved on the server in a small SQLite database
(`var/app.db`, created on first start) and listed in the sidebar, newest
first. Reopening a conversation shows the stored answer exactly as it was
given, with its citations and evidence, marked "Saved answer"; nothing is
asked again. Each conversation can be renamed or deleted (with
confirmation) from its menu; at phone and tablet width the history opens
from the clock icon in the header. The gear icon opens Settings: interface
language, answer language, theme and voice-input language, all remembered
for the next visit, plus "Reset to defaults" and "Delete all
conversations".

- **Who sees what**: a conversation belongs to the browser that held it.
  The browser is recognised by NiceGUI's signed, HttpOnly session cookie
  (14 days); the database stores only a hash of that identity, and every
  query is scoped to it, so another browser or another person on the
  same computer with a different browser profile cannot see or reach it.
  This is a guest identity, not a university account or a verified
  faculty login. Clearing the cookie starts a fresh, empty guest.
- **What is stored**: questions, answers, the answer's pages, evidence
  excerpts, the English intermediate and translation state of Arabic
  answers, timing and path diagnostics, the preferences (languages,
  theme, text size, reduced motion) and the optional display name. Never
  stored: audio, partial voice transcripts, the Groq key, the cookie
  secret or any token.
- **Where**: `APP_DB_PATH` moves the database; `APP_STORAGE_SECRET` (or a
  file at `APP_STORAGE_SECRET_PATH`, default `var/storage_secret.txt`)
  is the cookie-signing secret, generated once and kept across restarts.
  `var/` is ignored by git.
- **Maintenance**: `python -m persistence.manage check | info | backup
  [DEST] | restore SRC --yes | retention DAYS --yes`. Backups use
  SQLite's online backup API (safe while the app runs); a restore keeps
  the replaced file next to the database. Schema upgrades back the file
  up first and never drop data; a file written by a newer version is
  refused rather than downgraded.
- **If the database cannot be written**, the page says so, still
  answers, and marks the answer "Not saved" instead of pretending.

---

## Onboarding checklist, guided procedures and resource finder

Three connected pages for a new faculty member's first weeks, built only
from the approved documents (PF-14, decision record D-125):

- **Onboarding** (`/onboarding`): twelve source-grounded activities in
  four categories, each with its kind (required by the handbook,
  documented activity, optional, guidance), the exact evidence sentence
  and page, a status you set yourself (not started, in progress,
  completed, not applicable), progress figures and the next activities.
  Progress is saved for your browser and survives reloads and restarts.
  Marking a task complete records your own progress; it is not a
  university approval, and the assistant never changes it.
- **Guided procedures** (`/onboarding/procedure/<id>`): seven documented
  procedures (Blackboard access and announcements, attendance reports,
  grade transfer, iThenticate) shown step by step, every step the exact
  text of the approved guide, with conditional and after-the-procedure
  items flagged; Previous / Next remember your place; "Ask about this
  step" opens the chat with the question pre-filled for you to review and
  send.
- **Resources** (`/resources`): the eight approved documents, the
  departments with the contact numbers exactly as the handbook prints
  them (units without a printed contact say so), and four official web
  pages that were opened and checked on the date shown. Search works in
  English and Arabic; a task context lists its resources first.

What the model does here: nothing on these pages. A question asked from a
task or a step goes through the same Coordinator, specialists, retrieval
and verification as any typed question. Data: schema version 2 of the
PF-13 database (`onboarding_tasks`, `task_progress`, `progress_events`,
`procedure_positions`, `conversations.context_task`); a version-1
database upgrades at the first start with a backup first.

## Navigation, settings, guest profile and history

Every page uses one frame (PF-15, decision record D-126): a sidebar with
the brand, **New chat**, the navigation in two groups ("Workspace" and
"Your guest session"), the guest chip and "About this assistant". Below
900 px the sidebar becomes a drawer opened by the menu button in the
header (Escape or a tap outside closes it). The current page is marked
for assistive technology, and every item is an ordinary link, so the
browser's Back button and bookmarks work.

| Page | Address | What it does |
|---|---|---|
| Assistant chat | `/` | the start page; `/?new=1` opens an empty conversation |
| Onboarding checklist | `/onboarding` | the PF-14 checklist |
| Guided procedures | `/procedures` | the seven procedures, with "Continue at step k of n" where you left one |
| Resources & documents | `/resources` | the PF-14 finder |
| Conversation history | `/history` | search titles, open, rename and delete (with confirmation); groups by age |
| Guest profile | `/profile` | what this browser has saved, progress, recent activity, optional display name |
| Settings | `/settings` | languages, theme, text size, reduced motion, reset, delete all conversations |

- **Settings are saved at once** for this browser and applied on every
  page, also after a server restart. Large text enlarges the reading text;
  reduced motion turns off transitions (the device's own reduced-motion
  setting is always respected). The quick settings dialog on the chat
  page remains and links to the full page.
- **The display name** is optional, at most 40 characters, and shown only
  in this browser. Email addresses, long digit sequences (ID or phone
  numbers), line breaks and markup characters are refused. Resetting the
  settings keeps it; "Remove" deletes it.
- **A guest profile is not an account.** It is the PF-13 guest identity
  (a signed, HttpOnly browser cookie). There is no sign-in, and the
  application never asks for a university password, employee number or
  Emirates ID.
- **Times** are shown in your own time zone and language.
- **Data**: schema version 3 adds `text_size`, `reduced_motion` and
  `display_name` to `preferences`; a version-2 database upgrades at the
  first start with a backup first.

---

## Developer mode

Developer diagnostics are reachable only through a developer-facing
path: open `http://localhost:8501/?dev=1` (or start the app with
`APP_DEVELOPER_MODE=1`) and toggle **Developer** in the header to reveal
a collapsible panel under each assistant answer. It shows:

- Query type (contact / date / count / policy / list / yes-no)
- Which path answered (deterministic extractor or LLM)
- Whether any evidence came from OCR'd images
- Number of candidates retrieved, reranked, and kept
- Pages cited, best section
- Retrieval / rerank / generation / total timings (milliseconds)
- Knowledge-base statistics (total chunks, pages, OCR chunks)

Useful for demos — and for understanding the pipeline at a glance.

---

## Image / OCR support

The handbook's tables and prose are parsed as normal text via PyMuPDF.
If the PDF also has screenshots, diagrams, or scanned pages with useful
text, enable OCR:

```
# in .env
ENABLE_OCR=1
```

OCR runs on two passes per page:

1. Every embedded image above ~200×200 px.
2. A full-page raster at 180 DPI if the page's normal text is very
   short (catches mostly-image pages).

OCR uses [`rapidocr-onnxruntime`](https://pypi.org/project/rapidocr-onnxruntime/)
— pure pip, no system dependencies, ONNX models bundled. If the library
is missing or the runtime fails on a page, OCR silently becomes a
no-op; nothing else breaks.

OCR-derived chunks are tagged with `chunk_type="image_ocr"` in metadata
and appear as an `image-OCR evidence` chip under answers that use them.
Developer mode shows the total number of OCR chunks in the index.

**Expect the first indexing pass with OCR to take several minutes.**
Everything is cached to disk after — subsequent runs are as fast as
before.

---

## Architecture

The handbook PDF is parsed with PyMuPDF, which exposes word-level
coordinates — that lets us reconstruct table rows from pages that mix
tables with prose. Each page produces up to four chunk types:
**paragraph windows** for prose, **single rows** for table entries,
**row windows** for lists, and **image_ocr** for text recovered from
embedded images or rasterised pages (when OCR is on). All chunks are
embedded with MPNet and indexed with FAISS. At query time we classify
the question (contact / count / date / list / yes-no / policy), run
both dense and lexical retrieval, apply routing boosts that favour the
chunk type the question expects, rerank the short-list with a
cross-encoder, and then either (a) build the answer directly from the
top row if it's a contact / date / count question, or (b) prompt Groq
Llama-3.3-70B with the top 5 passages and instructions to refuse if
the evidence doesn't support an answer. Every answer includes page
citations verified against the retrieved context. The NiceGUI UI calls
this pipeline directly in the same Python process — no extra services.

---

## Configuration

Every tunable lives in `handbook_bot/config.py`. Common changes:

| Setting | Meaning | Default |
|---|---|---|
| `EMBED_MODEL` | Dense retriever | `all-mpnet-base-v2` |
| `RERANK_MODEL` | Cross-encoder | `ms-marco-MiniLM-L-12-v2` |
| `GROQ_MODEL` | LLM (env-overridable) | `llama-3.3-70b-versatile` |
| `ENABLE_OCR` | OCR embedded images + sparse pages | `False` |
| `OCR_MIN_IMAGE_PIXELS` | Minimum image size to OCR | `40000` |
| `OCR_PAGE_DPI` | DPI for full-page OCR raster | `180` |
| `PARA_CHUNK_SIZE_WORDS` | Paragraph chunk size | `160` |
| `FINAL_K` | Passages used in the prompt | `5` |
| `MIN_RERANK_SCORE` | Below this → refuse | `-1.4` |
| `ARABIC_SUPPORT` | Arabic layer on/off (env) | `True` |
| `ARABIC_QUERY_MODEL_FALLBACK` | One model request may translate a poorly covered Arabic question (env) | `True` |
| `ARABIC_QUERY_MIN_COVERAGE` | Lexicon coverage below which that fallback applies (env) | `0.6` |
| `ARABIC_ANSWER_TRANSLATION` | Translate verified answers with one model request, else quote the English (env) | `True` |
| `LLM_PROCESS_CALL_CAP` | Hard cap on model requests per process, 0 = none (env) | `0` |

Bump `CACHE_VERSION` in `config.py` if you change chunking/ingestion
logic and want to invalidate the embedding cache.

### Runtime modes (set these before an evaluation)

| Mode | Switches (`.env`) | What answers |
|---|---|---|
| LEGACY (application default) | none | one retrieval, reranking, extractor-or-model and Verifier pipeline; no specialist is consulted and no specialist label is shown |
| ACTIVE (acceptance configuration) | `PLAN_F_ENABLED=1`, `PLAN_F_GUIDE_SCOPES=1` | the Coordinator and the four specialists answer first when their deterministic, source-quoted answer passes the findings verification; every other question falls back to the LEGACY pipeline |
| SHADOW (diagnostics) | `PLAN_F_SHADOW=1` | the LEGACY answer is shown; the specialists run beside it for the Developer panel only |

The owner's manual acceptance tests and every QA run since PF-6 used ACTIVE with guide scopes on, so a human
evaluation of the Agentic AI behaviour must set those two switches; with an empty `.env` the application runs in
LEGACY. The Developer panel (`?dev=1`) shows the mode of every answer.

---

## Troubleshooting

**"No handbook PDF was found"**
→ Drop the PDF into the `data/` folder.

**"GROQ_API_KEY is not set" warning**
→ Edit `.env`, paste your Groq key, restart the app. Deterministic
contact / date / count answers still work without the key.

**PowerShell refuses to activate the venv**
→ Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**PowerShell says the script "is not digitally signed"**
→ Run `Unblock-File .\start.ps1` once. Happens when the project was
downloaded from the internet or OneDrive.

**First run is very slow**
→ Expected — the embedding and reranker models are downloading (~500
MB) and the whole PDF is being embedded. Subsequent runs use the
cache. If `ENABLE_OCR=1`, the first run also OCRs images, adding
several minutes.

**Port 8501 is already in use**
→ Close the other app using it, or edit `DEFAULT_PORT` in `app.py`.

**`faiss` install errors on Apple Silicon**
→ `pip install faiss-cpu --no-binary :all:` inside the activated venv.

**Wrong or outdated answer**
→ Delete the `cache/` folder contents; it will regenerate on next
run.

**The bot answers "I do not have this information"**
→ Either the handbook really doesn't cover the topic, or the
retriever couldn't find strong-enough evidence. Try rephrasing with
more specific terms from the handbook.

**OCR models fail to load on first use**
→ They download on first use. If your network blocks it, OCR silently
falls back to "no-op" — the rest of the pipeline works fine. Check
the terminal for a `[OCR]` line explaining the failure.

---

## What changed vs the previous version

- **Branding**: corrected to *UOS Faculty Onboarding Chatbot* with the
  official UOS logo used in the sidebar, landing hero, assistant
  avatar, and browser favicon.
- **Design**: full rewrite to a premium monochrome palette — graphite
  and charcoal in dark mode, off-white and soft gray in light mode —
  with UOS green used only as a subtle accent.
- **Theme toggle**: proper moon/sun icon button (not a switch) wired
  to a runtime-settable `data-theme` attribute, with custom light
  palette (not just inverted colors).
- **Layout**: wider main column (max 860 px), redesigned sidebar with
  brand block, theme/dev controls, and a refined status card. Top bar
  added with live status pill.
- **Composer**: send button now uses a clearly visible arrow icon
  (solid filled button in both themes), with a keyboard hint line
  below.
- **Chat behavior**: Enter-to-send fixed via proper Quasar/NiceGUI
  `keydown.enter.prevent` binding; Shift+Enter still creates a newline.
- **Source chips**: deduplicated and sorted page numbers; cleaner
  styling with a dedicated accent for pages.
- **Developer mode**: expanded from a flat list to four sections —
  Pipeline / Retrieval / Timing / Knowledge base — inside a
  collapsible panel with a rotation-animated chevron.
- **OCR**: new ingestion pass for embedded images and text-sparse
  pages via `rapidocr-onnxruntime`. Off by default; opt-in via
  `ENABLE_OCR=1`. Recovered text is indexed with a dedicated chunk
  type and surfaced in the UI.
- **Code**: UI split into `styles.py` (theme), `chat.py` (layout +
  interaction), and `pipeline.py` (backend bridge). Inline SVG icons
  replace emoji. Single in-memory state store keyed by NiceGUI client
  id.

---

## Known limitations

- OCR quality depends on the quality of the embedded images. Heavily
  stylised or very low-DPI screenshots produce noisy text.
- Arabic answers depend on the translation and checking layer (PF-11);
  an independent fluency review of the Arabic interface is outstanding.
- Conversations and settings belong to one browser's guest identity;
  clearing the browser's site data starts a new, empty guest.
- Phone and tablet layouts were checked by viewport emulation in Chrome,
  not on physical devices.

---

## License / credits

Student project. Uses open models from Hugging Face, FAISS from Meta,
the Groq API for inference, RapidOCR (ONNX) for image text, and
NiceGUI for the interface. The University of Sharjah logo is the
property of the University of Sharjah.

## Conversation upgrade (November 2026)

- **Friendly small talk (English + Arabic).** Greetings, thanks, goodbyes, "who are you", "how are you" and
  "what can you do / I need help" get a warm reply from a rotating pool, so the same sentence is never repeated back
  to back. Typos are tolerated (`thnaks`, `helo`, `شكراااا`). A message that also contains a real question still goes
  to the normal answer pipeline. Code: `handbook_bot/conversation.py`; switch off with `SMALL_TALK=0`.
- **Friendlier "not found" message.** When the documents do not support an answer, the page shows a varied, human
  sentence and tip (in the answer's language) plus two clickable example questions, instead of the fixed
  "I do not have this information." The backend decision is unchanged.
- **Spelling correction.** Clear typos in English words are fixed towards the vocabulary of the loaded documents before
  answering (`aproves confrence` -> `approves conference`), and the answer shows "Showing results for ...". Real English
  words, names and acronyms are never changed (e.g. `paternity` stays `paternity`). Code: `handbook_bot/spelling.py`
  (needs `pyspellchecker`); switch off with `SPELL_CORRECTION=0`.
- **Start the app:** double-click `START_CHATBOT.bat`, then open http://localhost:8501.
- Tests: `tests/test_conversation.py`, `tests/test_spelling.py` (plus the updated UI tests).

## Final pass (November 2026): stability and accuracy

What changed, why, and how it is switched on. `START_CHATBOT.bat`, `start.ps1` and `start.sh` set all of it; a manual
`python app.py` needs the variables in `.env` (they are in `.env.example`).

| Change | Problem it fixes | Setting |
|---|---|---|
| Wait for the rate limit | The Groq free tier allows 8,000 tokens per minute. A second question inside the minute got HTTP 429, the request was repeated after 1-2 seconds, failed again, and the user saw "the language-model service did not respond". The announced `Retry-After` is now honoured (cut to the cap). | `LLM_RETRY_AFTER_MAX` (seconds, default 30) |
| Low reasoning effort | The hidden reasoning of `openai/gpt-oss-120b` counts against the same 8,000 tokens. `low` is enough for answering from supplied context and makes each answer smaller and faster. Sent only to `openai/gpt-oss-*` models. | `LLM_REASONING_EFFORT=low` |
| Refusal rescue | Retrieval hands the model short excerpts. When they pass the evidence threshold and the model still refuses, the ONE retry is spent on a draft that also sees the full text of the best pages (policy questions only: a calendar or directory page holds several tables, and a draft that reads the wrong one cannot be told from a right one). | `REFUSAL_RESCUE=1` |
| Keep the supported sentences | A draft with one sentence the verifier could not ground used to become the refusal sentence. The unsupported sentences are now removed and the rest is verified again; it is served only when that check accepts it. | `REFUSAL_RESCUE=1` |
| Typo correction really on | `pyspellchecker` was missing from the environment, so typo correction was silently off. It is in `constraints.txt` and startup prints a NOTE when it is missing. | - |
| List bullets | The PDF's hollow bullet is extracted as the letter "o" (`o "E" Lecturer ...`). It is shown as a bullet; the stored answer stays the exact quotation. | - |
| Setup | `SETUP.bat` (environment in a short fixed folder, tested versions), see "File path too long". | - |

**Measuring it.** `eval/run_ui_eval.py` asks a question set through the page's own entry point (small talk, typo
correction, Arabic layer, specialists, verification), which `eval/run_eval.py` does not. `eval/extra_set.jsonl` adds
Arabic, typos, small talk, out-of-scope questions and the Blackboard / iThenticate guides.

```
set PLAN_F_ENABLED=1& set PLAN_F_GUIDE_SCOPES=1& set REFUSAL_RESCUE=1& set LLM_REASONING_EFFORT=low
python eval/run_ui_eval.py --tag final_ui --pause 2
python eval/score.py eval/runs/<the new file>.jsonl --verbose
```

`score.py` grades by required words, so a correct paraphrase ("familiarizing themselves" for "familiarize") is
counted as wrong; read the verbose output before quoting a number.

**What was measured on 13 November 2026** (ACTIVE mode, `openai/gpt-oss-120b`, Groq free tier):

- `pytest`: 7,918 tests pass on the final code.
- Golden set (37 graded questions + 5 that must be refused), run after the rate-limit fix and before the refusal
  rescue (`eval/runs/20261113_ui_golden_rate_limit_fix.jsonl`): no "service did not respond" answers, 5 of 5
  out-of-scope questions refused, route 32/32, 19 answers fully correct and 12 partial by word matching. Reading the
  answers: one is wrong (handbook-004 names "excellence, innovation, continuous improvement" instead of teaching,
  research and service), four are false refusals (the vision, two calendar questions, the workload model), several
  "partial" ones are correct paraphrases or incomplete lists.
- Extra set with the rescue on (`eval/runs/20261113_ui_extra_with_rescue.jsonl`): typos, Arabic, small talk, the
  guides and all five out-of-scope questions behave as intended; a long multi-part question and the Arabic annual-leave
  question are still refused.
- The rescue and the sentence trim were checked on the individual failing questions (the vision is now answered from
  page 15; the AI-policy and sabbatical answers are served without their unsupported sentence). They were NOT
  re-measured on the whole golden set: the test runs of that day used up the account's daily quota (200,000 tokens).
  Run the three commands above on a fresh day to get that number.

**Limits of the free Groq tier.** 8,000 tokens per minute and 200,000 tokens per day for the whole account: roughly
one or two questions per minute and a few hundred model-written answers per day, shared by everyone who uses the same
key. A short rate limit is waited out (the answer takes longer); when the daily quota is used up the service asks to
wait minutes or hours, the request is not repeated, the user sees "the language-model service did not respond" at once
and the log says `model quota exhausted`. Contact, calendar-row and quoted-guide answers do not use the model and keep
working. Several concurrent users need a paid Groq tier or one key per deployment.

**Sharing the project.** `.\make_zip.ps1` writes `dist\UOS_Chatbot.zip` without `.env`, saved conversations, git
history, the archive and build output, and with the current embedding cache.

## Using it on a phone / installing it as an app

The website is a responsive, installable web app (PWA). The Android app in `android/` is a thin shell around the same
website (see "Android app" below); there is no separate native user interface to maintain.

- **Layout.** The phone rules are the last blocks of `ui/styles.py` ("PHONES (mobile pass)"): 44 px touch targets, text of at
  least 12 px, 16 px form fields (so iOS does not zoom), tables that scroll sideways inside the answer, safe-area padding.
  `ui/shell.py` (`SHELL_JS`) keeps the message box above the on-screen keyboard and closes the slide-in menu by swipe,
  backdrop tap or Escape.
- **Install.** `ui/pwa.py` serves `/manifest.webmanifest` (standalone display, theme colour `#0B7A52`), `/sw.js` (shows
  `/offline.html` when the server cannot be reached; it never caches pages or answers) and the icons in `assets/icons/`.
  Regenerate the icons after changing `assets/logo.png` with `python qa/mobile_pwa/make_icons.py`.
- **Trying it from a phone.** The server listens on this computer only. To open it from a phone on the same network set
  `APP_HOST=0.0.0.0` in `.env` and browse to `http://<this-computer's-IP>:8501` (the development server has no login, so do
  this on a trusted network only). On iPhone use Share > "Add to Home Screen". Chrome on Android only offers "Install app"
  over HTTPS (or on `localhost`), so a real deployment needs an HTTPS address.

## Android app

`android/` is a standard Android Studio project (Kotlin, one activity, package `app.fob.uosfaculty`, Android 7.0+). It
opens the hosted chatbot full screen in a WebView; the Python server, its pages and its answers are unchanged, and the
answer pipeline still runs on the server, never on the phone.

What the shell adds to the website:

- **Voice input.** Android's WebView has no Web Speech API, so `VoiceBridge.kt` supplies `webkitSpeechRecognition` from
  the device's speech recogniser (English and Arabic, interim results). The page's microphone button and its state machine
  (`ui/voice.py`) work as in Chrome: the transcript lands in the message box and nothing is sent until Send. Microphone
  permission is asked on first use. The bridge is only exposed to the chatbot's own address.
- **Only the chatbot loads inside the app.** Official pages, documents, `mailto:` and `tel:` links open in the app that
  normally handles them.
- **A bundled "cannot be reached" page** with Try again, the system back button, a loading bar, and launcher shortcuts
  (New chat, Onboarding checklist, Resources & documents).
- **The guest session cookie is kept**, so saved conversations and settings survive closing the app. The app has its own
  cookie store: it does not share conversations with Chrome on the same phone.

Build it:

1. Install [Android Studio](https://developer.android.com/studio) (it brings the JDK, the Android SDK and Gradle).
2. Set the chatbot's HTTPS address in `android/gradle.properties`: `FOB_SERVER_URL=https://...` (no trailing slash). A
   release build refuses to start while the placeholder or an `http://` address is there.
3. Open the `android` folder in Android Studio, let it sync, then Run on a phone or emulator, or
   Build > Generate Signed App Bundle / APK for a release (create the signing key there; `android/.gitignore` keeps
   keystores out of the repository).

To try it before the server has an HTTPS address, start the server with `APP_HOST=0.0.0.0` and install a debug build that
points at this computer (debug builds allow `http://`, release builds never do):

```
gradlew installDebug -PFOB_SERVER_URL=http://<this-computer's-IP>:8501
```

Without Android Studio, the command line needs a JDK 17 (`JAVA_HOME`) and the Android SDK (`ANDROID_HOME`, or `sdk.dir`
in `android/local.properties`) with `platforms;android-35` and `build-tools;35.0.0`; then `gradlew assembleDebug` writes
`android/app/build/outputs/apk/debug/app-debug.apk`.

The launcher icon is `assets/icons/icon-maskable-512.png` (copied to `android/app/src/main/res/drawable-nodpi/
ic_launcher_art.png`) and `icon-192.png`; copy them again after regenerating the icons. iPhone users keep using
Share > "Add to Home Screen"; an iOS shell would need a Mac with Xcode.
