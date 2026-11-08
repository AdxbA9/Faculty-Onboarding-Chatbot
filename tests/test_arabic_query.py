"""Arabic text layer and Arabic-to-English query transformation (post-freeze PF-11). Offline; no model, no corpus."""
from __future__ import annotations

import pytest

from handbook_bot import arabic_lexicon
from handbook_bot.arabic_query import is_arabic_greeting, transform_query
from handbook_bot.arabic_text import detect_language, lookup_forms, normalize_arabic, strip_article, western_digits


# ---------------------------------------------------------------------------
# Normalisation and detection
# ---------------------------------------------------------------------------
def test_normalisation_folds_digits_punctuation_diacritics_and_alef_variants():
    assert normalize_arabic("١٢ ساعةً؟") == "12 ساعة?"
    assert normalize_arabic("إجازة أمومة آمنة ٱلعلمي") == "اجازة امومة امنة العلمي"
    assert normalize_arabic("مكتـــب") == "مكتب"                                  # tatweel removed
    assert normalize_arabic("مستشفى") == "مستشفي" and normalize_arabic("سؤال") == "سوال"


def test_ta_marbuta_is_kept_so_library_and_office_stay_distinct():
    assert normalize_arabic("مكتبة") != normalize_arabic("مكتب")
    forms = [f for f, _ in lookup_forms(normalize_arabic("المكتبة"))]
    assert forms[0] == "المكتبة" and "مكتبة" in forms and "مكتب" not in forms    # as written first, then article-free; the ة is never stripped


def test_western_digits_changes_nothing_but_the_digits():
    assert western_digits("يحق لهم ٦٠ يومًا بأجر كامل.") == "يحق لهم 60 يومًا بأجر كامل."


@pytest.mark.parametrize("text, code, primary", [
    ("ما هو العبء التدريسي للمحاضرين؟", "ar", "ar"),
    ("What is the teaching load for lecturers?", "en", "en"),
    ("كيف أستخدم Blackboard لإرسال Announcement؟", "mixed", "ar"),
    ("How do I use بلاك بورد?", "mixed", "en"),
    ("Blackboard: كيف أرسل إعلانًا للطلبة؟", "mixed", "ar"),
    ("12345 ???", "unknown", "unknown"),
    ("hr", "en", "en"),
    ("ر", "ar", "ar"),
])
def test_language_detection_is_script_based_and_tolerant_of_short_or_mixed_text(text, code, primary):
    profile = detect_language(text)
    assert (profile.code, profile.primary) == (code, primary)


def test_article_and_clitic_stripping():
    assert strip_article("الموارد") == ("موارد", None)
    assert strip_article("للمحاضرين") == ("محاضرين", "ل")
    assert strip_article("بالفصل") == ("فصل", "ب")
    assert strip_article("في") == ("في", None)                                     # too short to be an article + word
    forms = [f for f, _ in lookup_forms("للمحاضرين")]
    assert "محاضرين" in forms and "محاضر" in forms


# ---------------------------------------------------------------------------
# Lexicon integrity
# ---------------------------------------------------------------------------
def test_every_lexicon_entry_has_arabic_forms_and_an_english_rendering_or_an_explicit_drop():
    for forms, english, flags in arabic_lexicon.ENTRIES:
        assert forms.strip(), (forms, english)
        assert english or "x" in flags.split(), (forms, english, flags)
        for flag in flags.split():
            assert flag in ("q", "v", "h", "pl", "pt", "x", "neg"), (forms, flag)


def test_lexicon_contains_no_policy_facts():
    """Routing vocabulary only: no telephone numbers, no money amounts, no leave durations."""
    for forms, english, _ in arabic_lexicon.ENTRIES:
        assert "+971" not in english and "AED " not in english and " days" not in english.replace("working days", "").replace("calendar days", "").replace("how many days", "")


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arabic, english", [
    ("ما هو العبء التدريسي للمحاضرين؟", "What is the teaching load for lecturers?"),
    ("ما هو رقم هاتف قسم الموارد البشرية؟", "What is the phone number of the Human Resources Department?"),
    ("ما هو رقم فاكس وحدة الجوازات؟", "What is the fax number of the Passport Unit?"),
    ("من يوافق على سفر المؤتمرات؟", "Who approves conference travel?"),
    ("كم يوماً من إجازة الأمومة تستحقها عضوات هيئة التدريس؟", "How many days of maternity leave are female faculty entitled to?"),
    ("كم عدد الأطروحات التي يمكن لعضو هيئة التدريس الإشراف عليها في الوقت نفسه؟", "How many theses can a faculty member supervise at once?"),
    ("كم عدد الساعات المكتبية التي يجب على أعضاء هيئة التدريس الالتزام بها؟", "How many office hours must faculty hold?"),
    ("هل الحضور إلزامي للطلبة؟", "Is attendance mandatory for students?"),
    ("من يعد عروض التوظيف؟", "Who drafts employment offers?"),
    ("كيف يمكنني الدخول إلى البلاك بورد؟", "How can I log in to Blackboard?"),
    ("ما هي إجراءات التقديم للإجازة؟", "What are the procedures for applying for leave?"),
])
def test_arabic_questions_become_the_english_questions_the_documents_answer(arabic, english):
    result = transform_query(arabic, client=None, allow_model=False)
    assert result.method == "lexicon" and result.language.is_arabic
    assert result.english_query == english
    assert result.unmapped == [] and result.coverage == 1.0


def test_code_switched_questions_keep_their_english_terms():
    result = transform_query("كيف أستخدم Blackboard لإرسال Announcement؟", client=None, allow_model=False)
    assert result.english_query == "How do I use Blackboard to send Announcement?"
    assert result.language.code == "mixed" and result.language.primary == "ar"
    assert transform_query("ما هي إجراءات Faculty Appraisal؟", client=None, allow_model=False).english_query == "What are the procedures of the Faculty Appraisal?"


def test_multipart_questions_stay_multipart():
    result = transform_query("ما هو العبء التدريسي للمحاضرين وما هو رقم هاتف مكتب التسجيل؟", client=None, allow_model=False)
    assert result.english_query == "What is the teaching load for lecturers and what is the phone number of the Registration office?"


def test_arabic_indic_digits_and_negation_are_preserved():
    assert transform_query("هل العبء التدريسي للمحاضرين ١٢ ساعة معتمدة في الفصل؟", client=None, allow_model=False).english_query == \
        "Is teaching load for lecturers 12 credit hours per semester?"
    assert "not" in transform_query("هل صحيح أن أعضاء هيئة التدريس لا يحصلون على إجازة مرضية؟", client=None, allow_model=False).english_query


def test_english_questions_pass_through_untouched():
    result = transform_query("What is the teaching load for lecturers?", client=None)
    assert result.method == "passthrough" and result.english_query == "What is the teaching load for lecturers?" and result.llm_calls == 0


def test_unknown_words_lower_the_coverage_and_are_reported_not_invented():
    result = transform_query("ما هي عاصمة فرنسا؟", client=None, allow_model=False)
    assert result.coverage < 0.5 and set(result.unmapped) == {"عاصمة", "فرنسا"}
    assert "فرنسا" not in result.english_query and "France" not in result.english_query


def test_greetings_are_recognised():
    assert is_arabic_greeting("مرحبا") and is_arabic_greeting("السلام عليكم ورحمة الله") and is_arabic_greeting("صباح الخير!")
    assert not is_arabic_greeting("مرحبا، ما هو العبء التدريسي؟")
    assert transform_query("مرحبا", client=None).is_greeting


class _Reply:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _ScriptedClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def with_options(self, **_):
        return self

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return _Reply(self.reply)


def test_the_model_fallback_is_used_only_below_the_coverage_threshold_and_is_validated():
    client = _ScriptedClient("What is the capital of France?")
    result = transform_query("ما هي عاصمة فرنسا؟", client=client, allow_model=True, min_coverage=0.6)
    assert result.method == "model" and result.llm_calls == 1 and client.calls == 1
    assert result.english_query == "What is the capital of France?" and result.lexicon_query == "What is the?"
    assert client.kwargs["temperature"] == 0.0
    # a well-covered question never spends the call
    client = _ScriptedClient("ignored")
    result = transform_query("ما هو العبء التدريسي للمحاضرين؟", client=client, allow_model=True)
    assert result.method == "lexicon" and client.calls == 0
    # a reply that loses a number is rejected and the lexicon query stands
    client = _ScriptedClient("What is the capital?")
    result = transform_query("ما هي عاصمة فرنسا في ٢٠٢٥؟", client=client, allow_model=True, min_coverage=0.6)
    assert result.method == "lexicon" and result.model_error and "number" in result.model_error
    # an Arabic reply is rejected too
    client = _ScriptedClient("ما هي عاصمة فرنسا؟")
    assert transform_query("ما هي عاصمة فرنسا؟", client=client, allow_model=True).method == "lexicon"


def test_the_fallback_can_be_switched_off():
    client = _ScriptedClient("What is the capital of France?")
    result = transform_query("ما هي عاصمة فرنسا؟", client=client, allow_model=False)
    assert result.method == "lexicon" and client.calls == 0


def test_the_transformation_record_is_json_friendly_and_auditable():
    record = transform_query("ما هو رقم هاتف قسم الموارد البشرية؟", client=None, allow_model=False).to_dict()
    assert record["method"] == "lexicon" and record["language"]["code"] == "ar"
    assert {"ar": "رقم هاتف", "en": "phone number"} in record["mapped"]
    assert {"ar": "قسم الموارد البشرية", "en": "Human Resources Department"} in record["mapped"]
