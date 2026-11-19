"""Small talk and friendly refusal wording (handbook_bot.conversation + the pipeline's small-talk seam)."""
from __future__ import annotations

import pytest

from handbook_bot import conversation as C

SMALL_TALK = [
    ("hi", "greeting"), ("Hello!", "greeting"), ("heyyy", "greeting"), ("helo", "greeting"), ("good morning", "greeting"),
    ("thanks!", "thanks"), ("thank you so much", "thanks"), ("thnaks", "thanks"), ("thx bot", "thanks"), ("ty", "thanks"),
    ("ok thanks", "thanks"), ("bye", "bye"), ("goodbey", "bye"), ("see you later", "bye"),
    ("who are you?", "identity"), ("what can you do?", "help"), ("I need help.", "help"), ("help", "help"),
    ("ok", "ack"), ("cool", "ack"), ("how are you", "how_are_you"),
    ("شكرا", "thanks"), ("شكراً جزيلاً", "thanks"), ("شكراااا", "thanks"), ("يعطيك العافية", "thanks"),
    ("مرحبا", "greeting"), ("السلام عليكم", "greeting"), ("كيف حالك؟", "how_are_you"), ("ساعدني", "help"),
    ("من انت", "identity"), ("مع السلامة", "bye"), ("تمام", "ack"),
]

REAL_QUESTIONS = [
    "hi, how many days of sick leave do I get?", "thanks, and what about paternity leave?", "how do i make a grade column",
    "help me with leave", "who approves travel", "phone number", "school", "ho", "leave", "great job on the gradebook guide",
    "ما هو رقم هاتف الموارد البشرية؟", "مرحبا، ما هو العبء التدريسي؟", "شكرا، كم يوم إجازة مرضية؟",
]


@pytest.mark.parametrize("text,intent", SMALL_TALK)
def test_small_talk_is_recognised_with_typos_in_both_languages(text, intent):
    assert C.detect(text) == intent


@pytest.mark.parametrize("text", REAL_QUESTIONS)
def test_a_real_question_is_never_taken_for_small_talk(text):
    assert C.detect(text) is None


def test_replies_rotate_and_never_repeat_back_to_back():
    for lang in ("en", "ar"):
        for intent in C.REPLIES[lang]:
            seen = [C.reply(intent, lang) for _ in range(len(C.REPLIES[lang][intent]) * 2)]
            assert all(a != b for a, b in zip(seen, seen[1:])) or len(C.REPLIES[lang][intent]) == 1
            assert set(seen) == set(C.REPLIES[lang][intent])


def test_arabic_replies_are_arabic_and_english_replies_are_not():
    assert all(C.is_arabic(r) for pool in C.REPLIES["ar"].values() for r in pool)
    assert not any(C.is_arabic(r) for pool in C.REPLIES["en"].values() for r in pool)


def test_friendly_refusal_is_stable_per_seed_and_varies_across_seeds():
    assert C.friendly_refusal("en", 3) == C.friendly_refusal("en", 3)
    texts = {C.friendly_refusal("en", s)[0] for s in range(12)}
    assert len(texts) == len(C.REFUSALS["en"])
    text, tip = C.friendly_refusal("ar", 0)
    assert C.is_arabic(text) and C.is_arabic(tip)
    assert "I do not have this information" not in " ".join(C.REFUSALS["en"])


def test_the_pipeline_answers_small_talk_without_the_knowledge_base():
    from ui import pipeline
    result, diag = pipeline.ask_with_diagnostics(object(), "thanks!")       # no KB, no model: never touched
    assert result.query_type == "smalltalk" and diag["served"] == "smalltalk" and diag["small_talk"] == "thanks"
    assert result.answer in C.REPLIES["en"]["thanks"] and result.answer_language == "en" and not result.pages
    result, _ = pipeline.ask_with_diagnostics(object(), "شكرا")
    assert result.answer in C.REPLIES["ar"]["thanks"] and result.answer_language == "ar"
    result, _ = pipeline.ask_with_diagnostics(object(), "hello", answer_language="ar")
    assert result.answer in C.REPLIES["ar"]["greeting"]


def test_small_talk_can_be_switched_off(monkeypatch):
    from ui import pipeline
    monkeypatch.setenv("SMALL_TALK", "0")
    assert pipeline._small_talk("thanks") is None
