"""Safe rendering, notices and evidence rows of the redesigned chat page (post-freeze PF-10).

Pure functions: no browser, no model, no corpus. ``rendering.answer_html`` must never let HTML from an answer reach the
page, ``presentation.notice_kind`` must label exactly the backend's own no-answer sentences, and
``presentation.evidence_rows`` must report only what the evidence items carry.
"""
from __future__ import annotations

import re

import pytest

from handbook_bot.orchestrator import GREETING_REPLY, LLM_BUDGET_REPLY, LLM_UNAVAILABLE_REPLY, MODEL_UNAVAILABLE_REPLY, NO_KEY_REPLY
from handbook_bot.qa import QAResult, REFUSAL
from ui import presentation as P
from ui import rendering as R


# ---------------------------------------------------------------------------
# answer_html
# ---------------------------------------------------------------------------
def test_raw_html_in_an_answer_is_shown_as_text_not_executed():
    out = R.answer_html('Call <script>alert(1)</script> now <img src=x onerror="alert(2)"> ok')
    assert "<script" not in out and "<img" not in out
    assert re.search(r"<[^>]*onerror", out) is None             # the attribute survives only as visible text
    assert "&lt;script&gt;" in out and "&lt;img" in out
    assert R.plain_text(out) == 'Call <script>alert(1)</script> now <img src=x onerror="alert(2)"> ok'


def test_markdown_structure_survives_escaping():
    out = R.answer_html("Steps:\n\n1. Open the course\n2. Click **Announcements**\n\n| A | B |\n|---|---|\n| 1 | 2 |")
    assert "<ol>" in out and "<li>Open the course</li>" in out and "<strong>Announcements</strong>" in out
    assert "<table>" in out and "<td>2</td>" in out


def test_http_links_open_safely_and_other_schemes_become_plain_text():
    out = R.answer_html("See [the portal](https://my.sharjah.ac.ae) and [this](javascript:alert(1)) and [mail](mailto:hr@sharjah.ac.ae).")
    assert 'href="https://my.sharjah.ac.ae" target="_blank" rel="noopener noreferrer"' in out
    assert 'href="mailto:hr@sharjah.ac.ae" target="_blank" rel="noopener noreferrer"' in out
    assert "javascript:" not in out and "<span>this</span>" in out
    assert out.count("<a ") == 2 and out.count("</a>") == 2 and out.count("</span>") == 1


def test_identifiers_and_quotes_are_preserved():
    text = P.escape_identifiers("Set GROQ_MODEL in .env; it's the model's \"name\".")
    out = R.answer_html(text)
    assert "GROQ_MODEL" in R.plain_text(out) and "it's the model's" in R.plain_text(out) and '"name"' in R.plain_text(out)
    assert "<em>" not in out


def test_empty_answer_renders_nothing():
    assert R.answer_html("") == "" and R.answer_html(None) == ""


# ---------------------------------------------------------------------------
# notice_kind
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentence,kind,label", [
    (REFUSAL, "empty", "No supporting evidence found"),
    (NO_KEY_REPLY, "config", "Language-model service not configured"),
    (LLM_UNAVAILABLE_REPLY, "outage", "Language-model service unavailable"),
    (MODEL_UNAVAILABLE_REPLY, "config", "Language-model configuration problem"),
    (LLM_BUDGET_REPLY, "limit", "Answer limit reached"),
    (P.ERROR_TEXT, "error", "Something went wrong"),
])
def test_each_backend_no_answer_sentence_gets_its_label(sentence, kind, label):
    assert P.notice_kind(sentence) == (kind, label)
    assert P.notice_kind(sentence.rstrip(".")) == (kind, label)
    assert kind in P.NOTICE_HINTS


def test_real_answers_and_the_greeting_are_not_notices():
    assert P.notice_kind(GREETING_REPLY) is None
    assert P.notice_kind("The phone number is +971 6 5050023.") is None
    assert P.notice_kind("I do not have this information about leave, but here is what the handbook says.") is None
    assert P.notice_kind("") is None and P.notice_kind(None) is None


def test_hints_never_state_a_university_fact():
    for hint in P.NOTICE_HINTS.values():
        assert not re.search(r"\+971|page \d|\b(days?|AED|%)\b", hint)


# ---------------------------------------------------------------------------
# evidence_rows
# ---------------------------------------------------------------------------
def _item(text, page, title="UOS Faculty Handbook 2025/26", section="Contact Information", source_id="hb", chunk_type="row"):
    return {"chunk": text, "meta": {"page": page, "source_title": title, "section_title": section, "source_id": source_id, "chunk_type": chunk_type}}


def test_evidence_rows_report_document_page_section_and_text():
    result = QAResult(answer="x", pages=[54], items=[_item("Human Resources | +971 6 5050023 | +971 6 5585200", 54)])
    rows = P.evidence_rows(result)
    assert rows == [{"document": "UOS Faculty Handbook 2025/26", "page": 54, "section": "Contact Information",
                     "excerpt": "Human Resources · +971 6 5050023 · +971 6 5585200", "ocr": False}]


def test_evidence_rows_invent_nothing_and_drop_duplicates():
    result = QAResult(answer="x", items=[_item("same text", 9, title="", section="Page 9"), _item("same text", 9, title="", section="Page 9"),
                                         {"chunk": "", "meta": {"page": 3}}, "not a dict"])
    rows = P.evidence_rows(result)
    assert len(rows) == 1 and rows[0]["document"] is None and rows[0]["section"] is None and rows[0]["page"] == 9


def test_long_evidence_is_shortened_and_ocr_is_flagged():
    result = QAResult(answer="x", items=[_item("word " * 200, 12, chunk_type="image_ocr")])
    rows = P.evidence_rows(result, limit=80)
    assert len(rows[0]["excerpt"]) <= 80 and rows[0]["excerpt"].endswith("...") and rows[0]["ocr"] is True


def test_a_result_without_items_has_no_rows():
    assert P.evidence_rows(QAResult(answer=REFUSAL)) == []
