"""November 2026 final pass: rate-limit back-off, the leftover-bullet tidy and the refusal rescue's page evidence."""
from __future__ import annotations

from types import SimpleNamespace

from handbook_bot import orchestrator, qa
from ui import rendering


class _RateLimit(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("rate limit")
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = SimpleNamespace(headers=headers, status_code=429)


class _ServerError(Exception):
    status_code = 503


def test_rate_limit_waits_for_the_announced_time(monkeypatch):
    monkeypatch.delenv("LLM_RETRY_AFTER_MAX", raising=False)
    assert 12.0 <= qa._backoff_seconds(0, _RateLimit(12)) <= 12.5            # was ignored above 5 s: the repeat failed again
    assert qa._backoff_seconds(0, _RateLimit(120)) <= 30.0                    # never an unbounded wait


def test_retry_after_cap_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_AFTER_MAX", "3")
    assert qa._backoff_seconds(0, _RateLimit(12)) <= 3.0


def test_other_transient_failures_keep_the_short_backoff():
    assert qa._backoff_seconds(0, _ServerError()) == 0.8
    assert qa._backoff_seconds(1, _ServerError()) == 1.6


def test_pdf_bullet_is_shown_as_a_bullet_and_the_stored_text_is_untouched():
    text = 'o "E" Lecturer is mainly devoted to teaching and services.\n\no Ranked full time regular faculty'
    html = rendering.answer_html(text)
    assert "\u2022 &quot;E&quot; Lecturer" in html.replace('"', "&quot;") or '\u2022 "E" Lecturer' in html
    assert "\u2022 Ranked full time" in html and ">o " not in html
    assert text.startswith('o "E"')


def test_ordinary_answers_keep_their_words():
    for text in ("only full-time faculty may apply.", "Office hours are five per week.", "Go to options."):
        assert "\u2022" not in rendering.answer_html(text)

def _meta(page, kind, n, source="doc"):
    return {"page": page, "chunk_type": kind, "chunk_id_on_page": n, "source_id": source}


def test_page_items_rebuild_the_page_from_its_own_paragraphs():
    chunks = ["second paragraph", "first paragraph", "a row", "other document"]
    metadata = [_meta(15, "paragraph", 1), _meta(15, "paragraph", 0), _meta(15, "row", 2), _meta(15, "paragraph", 0, "guide")]
    items = [{"chunk": "a row", "meta": _meta(15, "row", 2)}]
    (page_item,) = orchestrator._page_items(items, chunks, metadata, 3)
    assert page_item["chunk"] == "first paragraph\nsecond paragraph"          # printed order, same document only
    assert page_item["meta"]["chunk_type"] == "page" and page_item["meta"]["page"] == 15
    assert items[0]["meta"]["chunk_type"] == "row"                            # the original item is not modified


def test_page_items_respect_the_page_limit_and_skip_duplicates():
    chunks = ["p1", "p2", "p3"]
    metadata = [_meta(1, "paragraph", 0), _meta(2, "paragraph", 0), _meta(3, "paragraph", 0)]
    items = [{"chunk": "p1", "meta": _meta(1, "paragraph", 0)}, {"chunk": "x", "meta": _meta(2, "row", 0)},
             {"chunk": "y", "meta": _meta(3, "row", 0)}]
    pages = [item["meta"]["page"] for item in orchestrator._page_items(items, chunks, metadata, 2)]
    assert pages == [2]                                                        # page 1 is already whole; page 3 is past the limit


def test_rescue_is_off_by_default_and_limited_to_policy_routes():
    assert orchestrator.REFUSAL_RESCUE is False or isinstance(orchestrator.REFUSAL_RESCUE, bool)
    assert "date" not in orchestrator._RESCUE_ROUTES and "contact" not in orchestrator._RESCUE_ROUTES


def test_unsupported_sentences_are_removed_and_the_rest_reads_as_sentences():
    answer = ("AI tools may be used to generate ideas, but users must verify any facts before using the output. "
              "All AI-generated material must be properly attributed; using it without attribution is academic misconduct. "
              "Staff must not disclose confidential information to AI tools.")
    trimmed = orchestrator._without_claims(answer, ["All AI-generated material must be properly attributed"])
    assert trimmed == ("AI tools may be used to generate ideas, but users must verify any facts before using the output. "
                       "Using it without attribution is academic misconduct. "
                       "Staff must not disclose confidential information to AI tools.")


def test_trim_gives_up_when_a_claim_is_not_found_or_too_little_is_left():
    answer = "Faculty hold five office hours per week. They are spread over two days."
    assert orchestrator._without_claims(answer, ["a sentence that is not in the answer"]) is None
    assert orchestrator._without_claims(answer, ["Faculty hold five office hours per week.", "They are spread over two days."]) is None
    assert orchestrator._without_claims("Short one. " + "x" * 200, ["x" * 200]) is None


def test_an_exhausted_daily_quota_is_reported_at_once(ask):
    from conftest import LEAVE_QUESTION, FakeGroqClient
    client = FakeGroqClient(_RateLimit(1196))                                 # "try again in 19m56s": no repeat can help
    result = ask(LEAVE_QUESTION, client)
    assert client.calls == 1 and result.answer == orchestrator.LLM_UNAVAILABLE_REPLY


def test_a_short_rate_limit_is_waited_out_and_answered(ask):
    from conftest import GROUNDED_LEAVE, LEAVE_QUESTION, FakeGroqClient
    client = FakeGroqClient(_RateLimit(7), GROUNDED_LEAVE)
    assert "thirty days" in ask(LEAVE_QUESTION, client).answer and client.calls == 2
