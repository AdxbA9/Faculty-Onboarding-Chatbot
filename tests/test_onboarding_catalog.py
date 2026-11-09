"""The curated onboarding catalog (post-freeze PF-14): validity, provenance of every quote against the frozen evidence
fixture (and against the corpus cache when it is present), exclusion of held sources, procedure fidelity to the
deterministic guide recovery, link safety, and the eight-versus-nine resolution for Blackboard announcements."""
from __future__ import annotations

import json
import os
import re

import pytest

from onboarding import catalog as C
from onboarding.links import anchor, safe_url

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "pf14_evidence.json")
CACHE = os.path.join(ROOT, "cache", "corpus_c8987b6079b11f39_v13")


@pytest.fixture(scope="module")
def evidence():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)["chunks"]


def _held(source_id: str) -> bool:
    return source_id in C.HELD_SOURCES


def _quote_in_fixture(evidence, source_id: str, page: int, quote: str) -> bool:
    return any(c["source_id"] == source_id and c["page"] == page and quote in c["text"] for c in evidence.values())


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------
def test_every_task_is_complete_and_classified() -> None:
    assert len(C.TASKS) >= 10
    for t in C.TASKS.values():
        assert re.match(r"^[a-z][a-z0-9_]{2,40}$", t.id)
        assert t.category in C.CATEGORIES and t.kind in C.KINDS
        assert t.title_en and t.title_ar and t.description_en and t.description_ar
        assert re.search(r"[\u0600-\u06FF]", t.title_ar) and re.search(r"[\u0600-\u06FF]", t.description_ar)
        assert t.evidence, t.id
        for e in t.evidence:
            assert e.source_id in C.SOURCE_TITLES and not _held(e.source_id) and e.page >= 1 and len(e.quote) >= 20
        for rid in t.resources:
            assert rid in C.RESOURCES, (t.id, rid)
        if t.procedure_id:
            assert t.procedure_id in C.PROCEDURES
    assert len({t.order for t in C.TASKS.values()}) == len(C.TASKS)              # a stable order
    assert {t.category for t in C.TASKS.values()} == set(C.CATEGORIES)            # every specialist domain has tasks


def test_required_tasks_rest_on_handbook_responsibility_or_condition_statements() -> None:
    required = [t for t in C.TASKS.values() if t.kind == "required"]
    assert {t.id for t in required} == {"hr_documents", "ilhe_training"}
    for t in required:
        assert all(e.source_id == C.HANDBOOK for e in t.evidence)


def test_no_task_states_a_deadline_contact_or_approval_the_source_does_not() -> None:
    for t in C.TASKS.values():
        text = " ".join([t.description_en, t.timing_en])
        assert "deadline" not in text.lower()
        assert not re.search(r"\+971|@", text)                                   # contacts live in the resource catalog, as printed
        assert "approved by the university" not in text.lower()


def test_every_procedure_has_ordered_traceable_steps_and_a_specialist() -> None:
    for p in C.PROCEDURES.values():
        assert p.steps and p.source_id in C.SOURCE_TITLES and not _held(p.source_id)
        assert p.specialist in ("teaching", "research", "faculty_services", "institutional")
        assert p.title_en and p.title_ar and p.heading and p.structure_key.startswith(p.source_id)
        for s in p.steps:
            assert s.page in p.pages and isinstance(s.chunk_id, int) and s.text and s.flag in ("", "conditional", "after")


def test_every_resource_is_typed_bilingual_and_only_cites_approved_sources() -> None:
    for r in C.RESOURCES.values():
        assert r.type in ("document", "department", "website") and r.category in C.CATEGORIES
        assert r.title_en and r.title_ar and r.description_en and r.description_ar and r.keywords_en and r.keywords_ar
        if r.source_id:
            assert r.source_id in C.SOURCE_TITLES and not _held(r.source_id)
        if r.type == "document":
            assert r.source_id and r.pages
        if r.type == "website":
            assert r.links and not r.contacts
        for tid in r.related_tasks:
            assert tid in C.TASKS
    assert len([r for r in C.RESOURCES.values() if r.type == "document"]) == 8     # the eight enabled sources, no more


def test_the_held_source_appears_nowhere() -> None:
    blob = json.dumps({"tasks": [t.__dict__ for t in C.TASKS.values()], "procedures": [p.__dict__ for p in C.PROCEDURES.values()],
                       "resources": [r.__dict__ for r in C.RESOURCES.values()]}, default=str, ensure_ascii=False)
    for held in C.HELD_SOURCES:
        assert held not in blob
    assert "Performance Appraisal" not in blob


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_every_quote_is_an_exact_substring_of_an_indexed_chunk_of_its_page(evidence) -> None:
    for t in C.TASKS.values():
        for e in t.evidence:
            assert _quote_in_fixture(evidence, e.source_id, e.page, e.quote), (t.id, e.quote)
    for p in C.PROCEDURES.values():
        for s in p.steps:
            chunk = evidence[str(s.chunk_id)]
            assert chunk["source_id"] == p.source_id and chunk["page"] == s.page and s.text in chunk["text"], (p.id, s.text)
    for r in C.RESOURCES.values():
        for e in r.evidence:
            assert _quote_in_fixture(evidence, e.source_id, e.page, e.quote), (r.id, e.quote)
        for c in r.contacts:
            assert _quote_in_fixture(evidence, c.evidence.source_id, c.evidence.page, c.evidence.quote), (r.id, c.value)
            assert c.value in c.evidence.quote and c.label in c.evidence.quote


def test_contacts_are_printed_numbers_only_and_units_without_one_say_so() -> None:
    for r in C.RESOURCES.values():
        for c in r.contacts:
            assert re.fullmatch(r"(\+971 6 \d{7})( \+971 6 \d{7})?", c.value), c.value
        if r.type == "department" and not r.contacts:
            assert r.contact_note_en and r.contact_note_ar


@pytest.mark.skipif(not os.path.exists(CACHE + "_chunks.json"), reason="the corpus cache is not present in this environment")
def test_every_quote_still_matches_the_live_corpus_cache() -> None:
    chunks = json.load(open(CACHE + "_chunks.json", encoding="utf-8"))
    meta = json.load(open(CACHE + "_meta.json", encoding="utf-8"))
    by_page = {}
    for i, m in enumerate(meta):
        by_page.setdefault((m["source_id"], m["page"]), []).append(i)

    def found(src, page, quote):
        return any(quote in chunks[i] for i in by_page.get((src, page), []))
    for t in C.TASKS.values():
        for e in t.evidence:
            assert found(e.source_id, e.page, e.quote), (t.id, e.quote)
    for p in C.PROCEDURES.values():
        for s in p.steps:
            assert meta[s.chunk_id]["source_id"] == p.source_id and meta[s.chunk_id]["page"] == s.page and s.text in chunks[s.chunk_id], (p.id, s.text)


@pytest.mark.skipif(not os.path.exists(CACHE + "_pages.json"), reason="the corpus cache is not present in this environment")
def test_procedures_match_the_deterministic_guide_recovery() -> None:
    """Each procedure's steps are the items of an ordered list the existing recovery reads from the guide, in order."""
    from handbook_bot import guide_grounding as gg
    from handbook_bot.sources import load_source_registry
    chunks = json.load(open(CACHE + "_chunks.json", encoding="utf-8"))
    meta = json.load(open(CACHE + "_meta.json", encoding="utf-8"))
    pages = {d["source_id"]: d["pages"] for d in json.load(open(CACHE + "_pages.json", encoding="utf-8"))}
    index = gg.GuideIndex(chunks, meta, pages, list(load_source_registry()))
    for p in C.PROCEDURES.values():
        structures = {s.key: s for s in index.structures_of(p.source_id)}
        recovered = []
        for key in p.structure_key.split("+"):
            s = structures[key]
            assert s.list_type == "ordered_procedure", (p.id, key, s.list_type)
            recovered.extend(part.quote for part in s.parts)                     # the lead-in and every item, in print order
        quotes = [s.text for s in p.steps]
        joined = " ".join(recovered)
        assert all(q in joined for q in quotes), (p.id, quotes, recovered)
        positions = [joined.index(q) for q in quotes]
        assert positions == sorted(positions), p.id                             # the catalog keeps the printed order


# ---------------------------------------------------------------------------
# The announcement procedure (eight versus nine)
# ---------------------------------------------------------------------------
def test_the_announcement_procedure_has_eight_printed_items_two_conditional_one_after() -> None:
    p = C.PROCEDURES["bb_announcement"]
    assert len(p.steps) == 8 and p.pages == (9,)
    assert [s.flag for s in p.steps] == ["", "", "", "conditional", "conditional", "", "", "after"]
    assert p.steps[0].text.startswith("From details and actions") and p.steps[6].text.startswith("Click on Post")
    assert all(s.text.startswith("If you want to") for s in p.steps if s.flag == "conditional")


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------
def test_every_catalog_link_is_https_verified_and_labelled() -> None:
    for r in C.RESOURCES.values():
        for link in r.links:
            assert safe_url(link.url) == link.url
            assert link.url.startswith("https://") and ".sharjah.ac.ae" in link.url
            assert link.kind in ("official_website", "document_link") and link.verified_on and link.observed_title and link.label_ar
            assert "outside the approved" in link.note_en or link.kind == "document_link"


@pytest.mark.parametrize("bad", ["javascript:alert(1)", "data:text/html,hi", "file:///C:/x", "ftp://example.org/x", "https://", "https://user:pw@example.org/",
                                 "https://exa mple.org/", "http://localhost/x", "", None, "https://example.org/\x00"])
def test_unsafe_or_malformed_urls_are_refused(bad) -> None:
    assert safe_url(bad) is None
    assert "href" not in anchor(bad or "", "label")


def test_anchor_escapes_text_and_adds_noopener() -> None:
    html = anchor("https://www.sharjah.ac.ae/en/?a=1&b=<x>", "<b>UoS</b> & co")
    assert 'rel="noopener noreferrer"' in html and 'target="_blank"' in html
    assert "&lt;b&gt;UoS&lt;/b&gt; &amp; co" in html and "<b>" not in html.split(">", 1)[1]
    assert 'href="https://www.sharjah.ac.ae/en/?a=1&amp;b=&lt;x&gt;"' in html
