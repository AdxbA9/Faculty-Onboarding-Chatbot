"""Interface strings and language handling of the chat page (post-freeze PF-11)."""
from __future__ import annotations

import re

from handbook_bot.arabic_text import has_arabic
from ui import chat, i18n, styles

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def test_every_string_exists_in_both_languages_with_the_same_placeholders():
    assert set(i18n.STRINGS) == {"en", "ar"}
    assert set(i18n.STRINGS["en"]) == set(i18n.STRINGS["ar"])
    for key, english in i18n.STRINGS["en"].items():
        arabic = i18n.STRINGS["ar"][key]
        assert english.strip() and arabic.strip(), key
        assert set(_PLACEHOLDER.findall(english)) == set(_PLACEHOLDER.findall(arabic)), key


def test_arabic_strings_are_arabic_and_english_strings_are_not():
    latin_allowed = {"hint_enter", "hint_shift", "lang_switch", "hint_text", "ocr", "you"}
    for key, arabic in i18n.STRINGS["ar"].items():
        assert has_arabic(arabic) or key in latin_allowed, key
    for key, english in i18n.STRINGS["en"].items():
        assert not has_arabic(english) or key == "lang_switch", key
    assert i18n.STRINGS["en"]["lang_switch"] == "العربية" and i18n.STRINGS["ar"]["lang_switch"] == "English"


def test_lookup_falls_back_to_english_and_formats():
    assert i18n.t("ar", "page", page=54) == "صفحة 54"
    assert i18n.t("en", "evidence", count=3) == "Supporting evidence (3)"
    assert i18n.t("xx", "send") == "Send" and i18n.t("ar", "no_such_key") == "no_such_key"


def test_initial_language_comes_from_the_query_then_the_browser_preference_then_english():
    assert i18n.lang_from_request("ar", "en-US,en") == "ar"
    assert i18n.lang_from_request("en", "ar-AE,ar") == "en"
    assert i18n.lang_from_request(None, "ar-AE,ar;q=0.9,en;q=0.8") == "ar"
    assert i18n.lang_from_request(None, "en-GB,en") == "en"
    assert i18n.lang_from_request("fr", None) == "en"
    assert i18n.DIRECTION == {"en": "ltr", "ar": "rtl"}


def test_the_arabic_first_use_screen_mirrors_the_english_one():
    assert [t["id"] for t in i18n.TOPICS_AR] == [t["id"] for t in chat.TOPICS]
    for ar, en in zip(i18n.TOPICS_AR, chat.TOPICS):
        assert len(ar["examples"]) == len(en["examples"]) and ar["icon"] == en["icon"]
        assert has_arabic(ar["title"]) and has_arabic(ar["blurb"])
        for example in ar["examples"]:
            assert has_arabic(example) and example.endswith("؟")
    assert len(chat.EXAMPLE_QUESTIONS_AR) == len(set(chat.EXAMPLE_QUESTIONS_AR)) == len(chat.EXAMPLE_QUESTIONS)


def test_domain_names_exist_in_both_languages_for_the_four_specialists():
    for lang in ("en", "ar"):
        assert set(i18n.DOMAIN_NAMES[lang]) == {"teaching", "research", "faculty_services", "institutional"}
    assert i18n.domain_name("ar", "research") == "البحث والابتكار" and i18n.domain_name("ar", "unknown") == "unknown"


def test_notices_are_localised_without_changing_the_backend_sentence():
    from handbook_bot.arabic_answers import REPLIES_AR
    from handbook_bot.qa import REFUSAL
    en = chat.notice_for(REFUSAL, "en")
    ar = chat.notice_for(REFUSAL, "ar")
    assert en["kind"] == ar["kind"] == "empty" and en["text"] == REFUSAL and ar["text"] == REPLIES_AR[REFUSAL]
    assert has_arabic(ar["label"]) and has_arabic(ar["hint"]) and not has_arabic(en["label"])
    assert chat.notice_for("A real answer.", "ar") is None


def test_the_stylesheet_carries_arabic_typography_and_rtl_rules():
    css = styles.GLOBAL_CSS
    assert "--sans-ar" in css and "Noto Naskh Arabic" in css and 'html[lang="ar"] body' in css
    assert 'html[dir="rtl"]' in css and ".translation-note" in css and "details.original" in css and ".topbar-lang" in css
    assert "globe" in styles.ICONS


def test_the_page_sets_lang_and_dir_and_rebuilds_from_state_on_a_language_switch():
    source = open(chat.__file__, encoding="utf-8").read()
    assert "lang={state[\"lang\"]} dir={i18n.DIRECTION[state[\"lang\"]]}" in source
    assert "def _render_all()" in source and "_on_lang_toggle" in source and 'mark("lang")' in source
    assert 'mark("answer-lang")' in source and "answer_language=answer_language" in source
    assert chat._client_state("pf11-lang-default")["lang"] == "en" and chat._client_state("pf11-lang-default")["answer_lang"] == "auto"
    assert chat._client_state("pf11-lang-ar", "ar")["lang"] == "ar"
