"""Composer and phone-width controls (defects D-09 and D-10 of the 2026-11-07 audit).

The key handler runs in the browser and the layout rule in CSS, so these tests pin the text the page receives. Behaviour in a
real browser is checked separately.
"""
from __future__ import annotations

import re

from ui import chat, styles


def test_plain_enter_sends_and_shift_enter_is_left_alone():
    js = chat.ENTER_SENDS_JS
    assert "e.key === 'Enter'" in js and "!e.shiftKey" in js
    assert js.index("!e.shiftKey") < js.index("preventDefault")


def test_typing_in_an_input_method_does_not_send():
    assert "!e.isComposing" in chat.ENTER_SENDS_JS


def test_the_modifier_nicegui_misreads_is_gone():
    source = open(chat.__file__, encoding="utf-8").read()
    assert "keydown.enter.exact" not in source


def test_new_chat_button_shows_only_at_phone_width():
    css = styles.STYLES if hasattr(styles, "STYLES") else open(styles.__file__, encoding="utf-8").read()
    assert re.search(r"\.topbar-new-chat \{ display: none; \}", css)
    phone = css[css.index("@media (max-width: 540px)"):]
    assert ".sidebar { display: none; }" in phone and ".topbar-new-chat { display: flex; }" in phone


# ---------------------------------------------------------------------------
# Post-freeze PF-10: design system, accessibility and language readiness of the page
# ---------------------------------------------------------------------------
def _css() -> str:
    return styles.GLOBAL_CSS


def _source() -> str:
    return open(chat.__file__, encoding="utf-8").read()


def test_light_theme_is_the_default_and_dark_is_a_choice():
    assert chat._client_state("pf10-theme-check")["theme"] == "light"
    chat._CLIENTS.pop("pf10-theme-check", None)
    assert '[data-theme="dark"]' in _css()


def test_keyboard_focus_is_visible_and_motion_can_be_reduced():
    css = _css()
    assert ":focus-visible { outline: 2px solid var(--brand)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_the_layout_uses_direction_aware_properties_for_the_arabic_phase():
    css = _css()
    assert "border-inline-end" in css and "padding-inline-start" in css and "text-align: start" in css
    assert 'dir="auto"' in _source() and "dir=auto" in _source()


def test_the_page_has_landmarks_a_heading_and_live_regions():
    source = _source()
    assert '<h1 class="topbar-title">' in source
    assert 'ui.element("main")' in source and 'ui.element("aside")' in source and 'ui.element("header")' in source
    assert 'role="log" aria-live="polite"' in source and 'role="status" aria-live="polite"' in source
    assert 'class="skip-link"' in source


def test_sidebar_hides_and_header_new_chat_appears_below_900px_as_well():
    css = _css()
    tablet = css[css.index("@media (max-width: 900px)"):css.index("@media (max-width: 640px)")]
    assert ".sidebar { display: none; }" in tablet and ".topbar-new-chat { display: flex; }" in tablet


def test_no_technical_identifiers_reach_ordinary_users():
    source = _source()
    assert "GROQ_MODEL" not in source and "ENABLE_OCR" not in source and "pdf_name" not in source.split("def service_status")[1].split("def _new_session")[0]
    assert "RAG" not in chat.APP_SUBTITLE


def test_every_example_question_belongs_to_a_topic_and_is_unique():
    assert len(chat.EXAMPLE_QUESTIONS) == len(set(chat.EXAMPLE_QUESTIONS)) >= 4
    assert {t["id"] for t in chat.TOPICS} == {"teaching", "research", "faculty_services", "institutional"}


def test_service_status_is_plain_language(monkeypatch):
    monkeypatch.setitem(chat._KB_STATE, "error", "Could not load the handbook (OSError).")
    assert chat.service_status()["label"] == "Setup needed"
    monkeypatch.setitem(chat._KB_STATE, "error", None)
    monkeypatch.setitem(chat._KB_STATE, "kb", None)
    assert chat.service_status()["label"] == "Starting up"
    monkeypatch.setitem(chat._KB_STATE, "kb", object())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert chat.service_status()["label"] == "Limited service"
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert chat.service_status()["label"] == "Ready"
    for s in ("Setup needed", "Starting up", "Limited service", "Ready"):
        assert "gpt" not in s and "GROQ" not in s


def test_developer_mode_needs_the_query_or_the_environment(monkeypatch):
    class Req:
        def __init__(self, dev):
            self.query_params = {"dev": dev} if dev is not None else {}
    monkeypatch.setattr(chat, "_DEV_ENV", False)
    assert chat.developer_requested(Req(None)) is False
    assert chat.developer_requested(Req("0")) is False
    assert chat.developer_requested(Req("1")) is True and chat.developer_requested(Req("true")) is True
    monkeypatch.setattr(chat, "_DEV_ENV", True)
    assert chat.developer_requested(Req(None)) is True
