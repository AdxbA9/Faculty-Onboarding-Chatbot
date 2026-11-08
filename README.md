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
  sidebar with the conversations of the browser session, a first-use
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

## Folder structure

```
UOS_Handbook_Chatbot/
├── app.py                  # Single entry point — just `python app.py`
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── start.ps1               # One-shot setup + run for Windows
├── start.sh                # One-shot setup + run for macOS / Linux
│
├── assets/
│   └── logo.png            # UOS logo served as a static asset
│
├── data/
│   └── UOS Faculty Handbook 25-26.pdf
│
├── cache/                  # Auto-generated embeddings cache (gitignored)
│
├── ui/                     # UI layer (NiceGUI)
│   ├── __init__.py
│   ├── chat.py             # Page layout + interaction logic
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

## Quick start — Windows (PowerShell)

**Option A — helper script** (recommended):

```powershell
.\start.ps1
```

The first time it creates `.venv`, installs dependencies, and copies
`.env.example` → `.env`. Edit `.env`, paste your `GROQ_API_KEY`, and run
the script a second time to launch the app.

If PowerShell blocks the script, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If it was downloaded from the internet, also run:

```powershell
Unblock-File .\start.ps1
```

**Option B — manual**:

```powershell
.\start.ps1

py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Open .env in Notepad and paste your GROQ_API_KEY
python app.py
```

Then open **http://localhost:8501**.

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
- The bot answers only in English; the UI copy is English-only.
- Chat history is per-browser-tab and held in memory — closing the
  tab clears it.

---

## License / credits

Student project. Uses open models from Hugging Face, FAISS from Meta,
the Groq API for inference, RapidOCR (ONNX) for image text, and
NiceGUI for the interface. The University of Sharjah logo is the
property of the University of Sharjah.
