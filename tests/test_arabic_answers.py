"""Arabic answers and the bilingual layer around the seam (post-freeze PF-11). Offline: the fake knowledge base and the
scripted model client of ``conftest``; no network."""
from __future__ import annotations

import faiss
import pytest

from conftest import FakeEmbedder, FakeGroqClient, FakeReranker, KB_CORPUS
from handbook_bot import arabic_answers as A
from handbook_bot import bilingual, config
from handbook_bot.groq_check import CappedClient, LLMCallCapExceeded
from handbook_bot.knowledge_base import KnowledgeBase
from handbook_bot.orchestrator import GREETING_REPLY, LLM_UNAVAILABLE_REPLY, NO_KEY_REPLY
from handbook_bot.qa import REFUSAL, QAResult
from ui import pipeline

HR_ITEM = {"chunk": "Human Resources +971 6 5050023 +971 6 5585200", "meta": {"page": 54, "source_id": "uos_faculty_handbook_2025_26", "source_title": "UOS Faculty Handbook 2025/26"}}


# ---------------------------------------------------------------------------
# Fixed replies and the frame
# ---------------------------------------------------------------------------
def test_every_backend_reply_has_an_arabic_counterpart_and_real_answers_have_none():
    for english in (REFUSAL, GREETING_REPLY, NO_KEY_REPLY, LLM_UNAVAILABLE_REPLY):
        assert A.localize_reply(english) and A.localize_reply(english) != english
    assert A.localize_reply("The phone number is +971 6 5050023.") is None
    assert A.localize_reply(REFUSAL.rstrip(".")) == A.REPLIES_AR[REFUSAL]


def test_the_frame_keeps_the_english_text_verbatim_and_names_source_and_page():
    framed = A.frame_answer("Human Resources +971 6 5050023 +971 6 5585200", [HR_ITEM], [54])
    assert "> Human Resources +971 6 5050023 +971 6 5585200" in framed
    assert "دليل أعضاء هيئة التدريس 2025/2026" in framed and "صفحة 54" in framed and A.FRAME_NOTE in framed
    multi = A.frame_answer("Line one.\nLine two.", [], [53, 54])
    assert "> Line one.\n> Line two." in multi and "الصفحات 53 و54" in multi and A.DEFAULT_SOURCE_AR in multi


# ---------------------------------------------------------------------------
# Translation check
# ---------------------------------------------------------------------------
GOOD_EN = "Female faculty members are entitled to sixty (60) calendar days of maternity leave with full pay. Additional leave without pay requires VCAA approval."
GOOD_AR = "يحق لعضوات هيئة التدريس ستون (60) يومًا تقويميًا من إجازة الأمومة براتب كامل. وتتطلب الإجازة الإضافية بدون راتب موافقة VCAA."


def test_a_faithful_translation_passes():
    check = A.check_translation(GOOD_EN, GOOD_AR)
    assert check.ok, check.problems


@pytest.mark.parametrize("arabic, problem", [
    ("يحق لعضوات هيئة التدريس ستون يومًا تقويميًا من إجازة الأمومة براتب كامل. وتتطلب الإجازة الإضافية بدون راتب موافقة VCAA.", "numbers_differ"),
    ("يحق لعضوات هيئة التدريس ستون (60) يومًا تقويميًا من إجازة الأمومة براتب كامل. وتتطلب الإجازة الإضافية بدون راتب موافقة نائب المدير.", "acronyms_missing"),
    ("Female faculty members are entitled to sixty (60) calendar days of maternity leave with full pay. Additional leave without pay requires VCAA approval.", "not_arabic"),
    ("", "empty"),
    ("يحق لعضوات هيئة التدريس ستون (60) يومًا. VCAA\nPages: 107", "pages_line_added"),
])
def test_unfaithful_translations_are_rejected_with_a_reason_code(arabic, problem):
    check = A.check_translation(GOOD_EN, arabic)
    assert not check.ok and problem in check.problems


def test_a_dropped_negation_is_rejected_but_arabic_only_negation_idioms_are_tolerated():
    english = "Consultancy involving managerial responsibilities in for-profit organizations is not allowed."
    assert "negation_dropped" in A.check_translation(english, "يُسمح بالاستشارات التي تتضمن مسؤوليات إدارية في المنظمات الربحية.").problems
    assert A.check_translation(english, "لا يُسمح بالاستشارات التي تتضمن مسؤوليات إدارية في المنظمات الربحية.").ok
    assert A.check_translation("Faculty must hold five office hours per week.", "يجب على أعضاء هيئة التدريس عقد خمس (5) ساعات مكتبية أسبوعيًا.").problems == ["numbers_differ"]
    assert A.check_translation("Faculty must hold 5 office hours per week.", "يجب على أعضاء هيئة التدريس عقد ما لا يقل عن 5 ساعات مكتبية أسبوعيًا.").ok


def test_contacts_urls_and_list_structure_must_survive():
    en = "Contact HR:\n- Phone +971 6 5050023\n- Fax +971 6 5585200\n- https://www.sharjah.ac.ae/hr"
    ar_ok = "تواصل مع HR:\n- الهاتف +971 6 5050023\n- الفاكس +971 6 5585200\n- https://www.sharjah.ac.ae/hr"
    assert A.check_translation(en, ar_ok).ok
    assert "phones_differ" in A.check_translation(en, ar_ok.replace("5585200", "5585201")).problems
    assert "list_structure_differs" in A.check_translation(en, "تواصل مع HR: الهاتف +971 6 5050023 والفاكس +971 6 5585200 https://www.sharjah.ac.ae/hr").problems
    assert "urls_differ" in A.check_translation(en, ar_ok.replace("/hr", "/hr2")).problems


def test_arabic_indic_digits_in_a_translation_count_as_the_same_numbers():
    assert A.check_translation("Up to 15 working days.", "حتى ١٥ يوم عمل.").ok


def test_the_translation_request_is_bounded_and_counted_by_the_caller():
    client = FakeGroqClient("نص عربي")
    assert A.translate_answer(client, "Some text") == "نص عربي" and client.calls == 1
    assert "Western digits" in A.TRANSLATION_SYSTEM and "Blackboard" in A.TRANSLATION_SYSTEM
    with pytest.raises(ValueError):
        A.translate_answer(FakeGroqClient(""), "Some text")


# ---------------------------------------------------------------------------
# The bilingual layer through the seam (fake knowledge base)
# ---------------------------------------------------------------------------
def _kb(client=None) -> KnowledgeBase:
    chunks = [row[0] for row in KB_CORPUS]
    metadata = [{"page": row[1], "section": row[2], "chunk_type": row[3], "source_id": "uos_faculty_handbook_2025_26", "source_title": "UOS Faculty Handbook 2025/26"} for row in KB_CORPUS]
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return KnowledgeBase(pdf_file="handbook.pdf", embedder=embedder, reranker=FakeReranker(), groq_client=client, pages=[{"page": i + 1, "text": ""} for i in range(50)],
                         chunks=chunks, metadata=metadata, embeddings=None, index=index)


IT_AR = "ما هو رقم هاتف مركز تكنولوجيا المعلومات؟"
IT_EN = "What is the phone number of the Information Technology Center?"


def test_an_arabic_contact_question_is_answered_from_the_extractor_and_framed_without_a_model(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result, diag = pipeline.ask_with_diagnostics(_kb(None), IT_AR)
    assert result.answer_language == "ar" and result.llm_calls == 0 and result.query_type == "contact"
    assert result.original_answer == "The phone number is +971 6 5050000."
    assert "> The phone number is +971 6 5050000." in result.answer and "صفحة 14" in result.answer
    assert result.translation["status"] == "framed" and result.translation["reason"] == "no_client"
    assert result.query_transform["english_query"] == IT_EN and result.query_transform["method"] == "lexicon"
    assert diag["language"]["detected"] == "ar" and diag["language"]["answer_language"] == "ar" and diag["language"]["translation_status"] == "framed"
    assert "english_query" not in diag["language"]                       # diagnostics carry codes and counts only


def test_an_english_question_takes_exactly_the_pre_pf11_path():
    direct = pipeline.ask(_kb(None), IT_EN)
    result, diag = pipeline.ask_with_diagnostics(_kb(None), IT_EN)
    assert result.answer == direct.answer == "The phone number is +971 6 5050000."
    assert result.answer_language == "en" and result.translation == {} and result.query_transform == {}
    assert "language" not in diag


def test_the_arabic_refusal_and_greeting_keep_the_canonical_sentence_and_flag_the_language(monkeypatch):
    monkeypatch.setattr(config, "ARABIC_QUERY_MODEL_FALLBACK", False)
    result, _ = pipeline.ask_with_diagnostics(_kb(None), "السلام عليكم")
    assert result.answer == GREETING_REPLY and result.answer_language == "ar" and result.translation["status"] == "fixed"
    kb = _kb(FakeGroqClient(REFUSAL))
    result, diag = pipeline.ask_with_diagnostics(kb, "ما هي سياسة الامتحانات التعويضية؟")
    assert result.answer in (REFUSAL, NO_KEY_REPLY) or result.answer_language == "ar"
    assert result.answer_language == "ar" and result.translation["status"] == "fixed"
    assert diag["language"]["translation_llm_calls"] == 0


def test_a_verified_english_answer_is_translated_checked_and_counted():
    client = FakeGroqClient("Faculty members are entitled to thirty days of annual leave each academic year.\nPages: 3",
                            "يحق لأعضاء هيئة التدريس ثلاثون يومًا من الإجازة السنوية في كل عام أكاديمي.")
    result, diag = pipeline.ask_with_diagnostics(_kb(client), "كم يوماً من الإجازة السنوية يستحق أعضاء هيئة التدريس؟")
    assert result.answer == "يحق لأعضاء هيئة التدريس ثلاثون يومًا من الإجازة السنوية في كل عام أكاديمي."
    assert result.original_answer.startswith("Faculty members are entitled to thirty days")
    assert result.translation["status"] == "translated" and result.translation["check"]["ok"]
    assert result.llm_calls == 2 and client.calls == 2                 # synthesis + translation, nothing else
    assert diag["language"]["translation_llm_calls"] == 1 and result.pages == [3]


def test_a_translation_that_fails_the_check_falls_back_to_the_frame_and_keeps_the_english_verbatim():
    result = QAResult(answer="Faculty members are entitled to 30 days of annual leave.", pages=[3], items=[{"chunk": "x", "meta": {"page": 3}}])
    record = bilingual.localize(result, FakeGroqClient("يحق لأعضاء هيئة التدريس إجازة سنوية كل عام."), allow_translation=True)
    assert record["status"] == "framed_after_check" and "numbers_differ" in record["problems"]
    assert "> Faculty members are entitled to 30 days of annual leave." in result.answer and result.llm_calls == 1


def test_a_failing_translation_service_frames_the_answer_and_records_the_redacted_error():
    result = QAResult(answer="Faculty members hold five office hours.", pages=[79])
    record = bilingual.localize(result, FakeGroqClient(RuntimeError("boom gsk_abc12")), allow_translation=True)
    assert record["status"] == "framed_after_error" and "gsk_***" in record["error"] and "abc12" not in record["error"]
    assert result.answer.startswith("وفقًا") and result.original_answer == "Faculty members hold five office hours."


def test_translation_can_be_disabled_and_then_no_request_is_made():
    client = FakeGroqClient("unused")
    result = QAResult(answer="Faculty members hold five office hours.", pages=[79])
    record = bilingual.localize(result, client, allow_translation=False)
    assert record["status"] == "framed" and record["reason"] == "disabled" and client.calls == 0


def test_answer_language_preferences_are_honoured_in_both_directions():
    result, diag = pipeline.ask_with_diagnostics(_kb(None), IT_EN, answer_language="ar")
    assert result.answer_language == "ar" and "+971 6 5050000" in result.answer and diag["language"]["query_method"] == "passthrough"
    result, diag = pipeline.ask_with_diagnostics(_kb(None), IT_AR, answer_language="en")
    assert result.answer_language == "en" and result.answer == "The phone number is +971 6 5050000."
    assert result.query_transform["english_query"] == IT_EN and diag["language"]["translation_status"] == "not_requested"
    assert bilingual.resolve_answer_language("auto", "كيف أستخدم Blackboard؟") == "ar"
    assert bilingual.resolve_answer_language("auto", "How do I use Blackboard?") == "en"


def test_the_layer_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setattr(config, "ARABIC_SUPPORT", False)
    result, diag = pipeline.ask_with_diagnostics(_kb(None), IT_AR)
    assert result.answer_language == "en" and result.query_transform == {} and "language" not in diag


def test_the_process_call_cap_refuses_requests_beyond_the_cap_and_shares_the_counter_across_with_options():
    inner = FakeGroqClient("reply")
    capped = CappedClient(inner, 2)
    capped.chat.completions.create(messages=[{"role": "user", "content": "a"}])
    capped.with_options(timeout=1).chat.completions.create(messages=[{"role": "user", "content": "b"}])
    with pytest.raises(LLMCallCapExceeded):
        capped.chat.completions.create(messages=[{"role": "user", "content": "c"}])
    assert capped.calls_issued == 2 and capped.calls_refused == 1 and inner.calls == 2


def test_a_capped_client_turns_an_exhausted_budget_into_the_outage_notice_not_a_refusal():
    client = CappedClient(FakeGroqClient("Faculty members are entitled to thirty days of annual leave each academic year.\nPages: 3"), 0)
    result, _ = pipeline.ask_with_diagnostics(_kb(client), "How much annual leave are faculty members entitled to?")
    assert result.answer == LLM_UNAVAILABLE_REPLY and client.calls_refused == 1


def test_a_negation_marker_followed_by_punctuation_counts(): # followed_by_punctuation
    """The live smoke case L10: "No; faculty are entitled ..." translated as "لا؛ يحق ..." was rejected by the first check."""
    english = "No; faculty are entitled to sick leave with full pay for up to 15 working days."
    assert A.check_translation(english, "لا؛ يحق لأعضاء هيئة التدريس إجازة مرضية بأجر كامل لمدة تصل إلى 15 يوم عمل.").ok
    assert "negation_dropped" in A.check_translation(english, "يحق لأعضاء هيئة التدريس إجازة مرضية بأجر كامل لمدة تصل إلى 15 يوم عمل.").problems
