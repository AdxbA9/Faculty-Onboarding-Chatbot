# Source Inventory

What the retrieval corpus actually contains, how each source is identified,
and how a search can be restricted to part of it. Everything below describes
the repository as it is; nothing is planned coverage.

## Indexed sources

| Source id | Title | File | Type | Version | Status |
|---|---|---|---|---|---|
| `uos_faculty_handbook_2025_26` | UOS Faculty Handbook 2025/26 | `data/UOS Faculty Handbook 25-26.pdf` (272 pages) | pdf | 2025/2026 (from the cover) | verified local file, the only indexed document |

The registry is `knowledge/sources.json`. Paths are repository-relative;
no machine-specific path is stored.

### Not present locally

There is no standalone Blackboard guide, Banner guide, HR handbook, research
guide or institutional guide in the repository, and none is indexed. The
handbook contains policy-level references to Blackboard and Banner (for
example the syllabus upload rule on page 222 and the grade flow from
Blackboard to Banner on page 224) but no step-by-step procedure for either
system. Any future answer about a detailed Blackboard or Banner workflow
must therefore be reported as not found or partial until a verified guide
is added to the registry and indexed. Procedures are never inferred.

The academic calendar (section 1.14, pages 49 to 51) is inside the handbook;
it is not a separate source.

## Source registry (`handbook_bot/sources.py`)

- `SourceRecord`: `source_id`, `title`, `relative_path`, `source_type`,
  `version`, `authority`, `verification_status`, optional `section_map`,
  `notes`. Immutable; validated on construction (repository-relative
  forward-slash path, lowercase id).
- `SourceRegistry`: `get_source(id)` (raises `UnknownSourceError` for an
  unknown id), `find_source_for_path(path)`, `list_sources()`,
  `validate_source_registry()` (file exists, section map loads). Duplicate
  ids are rejected when the registry is built.
- `load_source_registry(path=None)` reads the JSON file; there is no
  module-level instance and importing the module has no side effect.
- `unregistered_source(path)` gives a document that is not in the registry
  a deterministic identity (`unregistered_<file name>`) with no section map,
  so a scope restricted to registered sources never admits it.

Adding a source later means adding one record (and, optionally, its own
section map file); the schema does not change.

### Registry capability versus current runtime ingestion

The registry schema is ready for several source records, but the runtime
still ingests exactly one document: `pdf_loader.find_pdf_file` selects the
first PDF in `data/`, `KnowledgeBase` holds one `pdf_file`, and the cache
is keyed by that one file. Adding a record to `knowledge/sources.json` does
not index anything by itself. Multi-document ingestion is a later change to
the knowledge base and the cache, not to the registry.

### Source identity

Identity is strict and by location: a document is the registered source
only when the file actually opened is the file at the registered
repository-relative location, in any spelling of that path (relative to the
working directory or absolute, either separator, any letter case on
Windows). A file elsewhere with the same name is unregistered. Location is
not content: a different edition saved at the registered path would still
be identified as the handbook, which is why the section map is applied only
when the loaded document's page count and highest page index equal the
map's `page_count`. Otherwise the document keeps its source identity, its
chunks carry no chapter or section labels, and a warning is printed at
start-up. Content fingerprinting (size or hash on the record) is a possible
future hardening and is not implemented.

## Section map (`knowledge/handbook_sections.json`)

Derived from the handbook's own table of contents (PDF pages 4 to 13) at
chapter and level-2 section granularity, and checked against the document:
every chapter heading and all 186 listed section headings were found on
their listed pages. Page numbers are physical 1-based PDF pages; in this
document they equal the printed "Page N" footer, the same convention as
`meta["page"]`.

- 206 records: 4 front-matter ranges (cover, preface, important notice,
  contents), 16 chapters (pages 14 to 272, contiguous), 186 sections.
- Ranges are inclusive. A section that starts on the page where the previous
  one ends shares that boundary page; the validator allows overlap only on
  such a page.
- `SectionMap.resolve(page)` returns the chapter, every section present on
  the page, and one primary section: the first section that starts on the
  page, otherwise the section in progress. Front-matter pages resolve to
  their part with no chapter or section. No page of the handbook is
  unmapped.
- Document quirks are recorded, not repaired: the handbook uses 5.2 and 7.8
  twice, and prints the copyright policy inside chapter 9 as 12.3. Each such
  record carries a note. The chapter of a page always comes from the chapter
  page ranges, never from the section number, so chapter scoping is right
  even for the mis-numbered section, and section scoping is
  chapter-consistent (see Retrieval scope).
- Section numbers are recorded as printed and are not unique identifiers:
  the handbook uses 5.2 and 7.8 twice (adjacent sections of the same
  chapter) and 12.3 twice (chapter 9 and chapter 12).
- 115 of the 272 pages carry more than one section (up to eight on one
  page), so page-level section labels are coarse by nature.

The old `section` key (the loader's heading guess, "Page N" on almost every
page) is preserved unchanged for compatibility and should not be used for
scoping.

## Chunk metadata

`annotate_metadata` (called by `build_knowledge_base` after chunking) adds to
every chunk, without removing any existing key:

| Key | Value |
|---|---|
| `source_id` | registry id, or `unregistered_...` |
| `source_title`, `source_type`, `source_version` | from the source record |
| `chapter`, `chapter_title` | from the page's chapter range, or null |
| `section_no`, `section_title` | the page's primary section, or null |
| `page_section_nos` | every section present on the page, document order (empty when unmapped) |

Existing keys (`page`, `section`, `chunk_type`, `chunk_id`, `text`,
`row_id`, `row_start`, `row_end`, `table_like`) are unchanged, and so is the
chunk text: 13,688 chunks before and after. A retrieved item therefore
carries the source, the exact page, the chunk id, the chapter and section
where mapped, and the verbatim evidence text (`item["chunk"]`).

## Retrieval scope (`handbook_bot/retrieval.py`)

`gather_candidates(..., scope=None)` gained an optional `RetrievalScope`:

- constraints: `source_ids`, `chapters`, `section_numbers`,
  `section_prefixes` (dotted prefix: `"1"` matches `"1.14"`, never `"12"`;
  `"12.1"` matches `"12.1"`, never `"12.10"`), `page_ranges` (inclusive).
  All given constraints must hold; within one, any value matches.
- section constraints are chapter-consistent: a printed section number is
  eligible only when its leading number equals the chunk's resolved
  `chapter` (from the chapter page ranges). The section the handbook prints
  as 12.3 on page 180 keeps that number in the metadata, but neither
  `section_numbers={"12.3"}` nor `section_prefixes={"12"}` admits it; it is
  reachable through `chapters={9}` or a page range. `chapters={9}` combined
  with `section_numbers={"12.3"}` intentionally matches nothing.
- section constraints are page-granular: they operate on page-level
  section metadata and restrict pages associated with a section (every
  section present on a page counts); they do not guarantee text-level
  isolation on a page shared by several sections. A chunk admitted for
  1.14 on page 51 may contain text of 1.15, and a chunk admitted for 12.12
  on page 222 may contain text of 12.10 or 12.11.
- printed section numbers are not unique identifiers: `section_no` is
  metadata as printed, so `section_numbers={"7.8"}` admits both printed
  7.8 sections. Caller guidance: prefer `source_ids` plus `chapters`; use
  `page_ranges` where exact physical isolation is required; never treat
  `section_no` as a primary key.
- dense search is restricted at the index level with a FAISS id selector
  (exact search over the allowed chunks only); an in-scope chunk that ranks
  below the global top-k is still found. If an index cannot take search
  parameters at all, the allowed vectors are scored directly, and only for
  an inner-product index; any other metric raises an error rather than
  returning misleading scores. A genuine search failure propagates.
- lexical search scores only the allowed chunks; routing boosts and the
  reranker then see allowed candidates only.
- a scope that matches no chunk returns an empty list; there is no fallback
  to the whole corpus.
- `scope=None` runs the previous code path unchanged; the candidate order on
  the test corpus is pinned in `tests/test_retrieval_scope.py`.

Nothing in the current pipeline passes a scope yet; the orchestrator,
Router, Synthesis, Verifier and UI are unchanged.

## Cache

`CACHE_VERSION` moved from v11 to v12 because the cached metadata gained the
keys above. Chunk text, the embedding model and the reranker are unchanged.
Every machine rebuilds its cache once on the next start (the handbook is
re-embedded; a few minutes on CPU).
