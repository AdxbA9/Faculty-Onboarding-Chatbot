"""Regression tests for the final-audit fixes (19 November 2026).

1. Copy answer: the button used an inline ``onclick`` inside a single-quoted attribute whose own code contained single
   quotes, so the browser cut the handler short ("Uncaught SyntaxError: Unexpected end of input") and nothing was ever
   copied. The button now carries no script; the text travels as JSON in a hidden element and one delegated listener
   (``rendering.COPY_JS``) copies it. These tests parse the generated HTML the way a browser does.
2. Disclosure: the page says that model-assisted answers send the question and the retrieved passages to Groq.
3. Android: the distributable project must not carry one installation's private network address.

Pure functions and files: no browser, no model, no corpus.
"""
from __future__ import annotations

import json
import os
import re
from html.parser import HTMLParser

import pytest

from ui import i18n
from ui import rendering as R
from ui import shell

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AWKWARD_ANSWERS = [
    "Human Resources (contact directory): telephone +971 6 5050023.",
    "The dean's office approves it; faculty can't exceed 'two' requests.",
    'The handbook says: "office hours are posted".',
    "</span><script>alert(1)</script> & <b>bold</b> &amp; &lt;tag&gt;",
    "' onclick='alert(1)",
    '"><img src=x onerror=alert(1)>',
    "Line one\nLine two\r\n\tTabbed \\ backslash \\n not a newline",
    "الموارد البشرية: هاتف ‎+971 6 5050023 —   narrow space,   line separator",
    "",
]


class _Toolbar(HTMLParser):
    """Collects what a browser would see: each start tag with its attributes, and the text inside ``.copy-src``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags, self.source, self._in_source, self.label, self._in_label = [], "", False, "", False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        classes = (attrs.get("class") or "").split()
        self._in_source = "copy-src" in classes
        self._in_label = "copy-lbl" in classes

    def handle_endtag(self, tag):
        self._in_source = self._in_label = False

    def handle_data(self, data):
        if self._in_source:
            self.source += data
        if self._in_label:
            self.label += data


def _parse(markup: str) -> _Toolbar:
    parser = _Toolbar()
    parser.feed(markup)
    parser.close()
    return parser


# ---------------------------------------------------------------------------
# 1. Copy answer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer", AWKWARD_ANSWERS)
def test_the_copy_button_round_trips_any_answer_text(answer):
    parsed = _parse(R.copy_button_html(answer, "Copy answer", "Copied", "Could not copy", '<svg viewBox="0 0 1 1"></svg>'))
    assert json.loads(parsed.source) == answer                    # what JSON.parse(source.textContent) gives the handler
    assert parsed.label == "Copy answer"


@pytest.mark.parametrize("answer", AWKWARD_ANSWERS)
def test_the_copy_button_carries_no_script_and_no_stray_attributes(answer):
    parsed = _parse(R.copy_button_html(answer, "Copy answer", "Copied", "Could not copy"))
    assert [tag for tag, _ in parsed.tags] == ["button", "span", "span"]          # the answer adds no element of its own
    button = parsed.tags[0][1]
    assert set(button) == {"type", "class", "data-copy-answer", "aria-label", "data-copied", "data-copy-failed"}
    assert button["data-copied"] == "Copied" and button["data-copy-failed"] == "Could not copy"
    for _, attrs in parsed.tags:
        assert not any(name.startswith("on") for name in attrs)                   # no inline event handler anywhere


def test_labels_with_quotes_stay_inside_their_attributes():
    parsed = _parse(R.copy_button_html("x", 'Copy "this" answer', "It's copied", "Couldn't <copy>"))
    button = parsed.tags[0][1]
    assert button["aria-label"] == 'Copy "this" answer'
    assert button["data-copied"] == "It's copied" and button["data-copy-failed"] == "Couldn't <copy>"
    assert parsed.label == 'Copy "this" answer'


def test_the_copy_script_is_delegated_and_has_a_fallback_for_plain_http():
    js = R.COPY_JS
    assert "button[data-copy-answer]" in js and "addEventListener('click'" in js
    assert "navigator.clipboard" in js and "isSecureContext" in js and "execCommand('copy')" in js
    assert "data-copy-failed" in js                               # a failed copy is reported, not shown as "Copied"
    assert "</script" not in js.lower()                           # it is embedded in a <script> element


def test_the_chat_page_no_longer_builds_an_inline_copy_handler():
    source = open(os.path.join(REPO, "ui", "chat.py"), encoding="utf-8").read()
    assert "clipboard.writeText" not in source and "copy_js" not in source     # the handler and its inline builder are gone
    assert "onclick=\\'" not in source                                         # no handler in a single-quoted attribute
    assert "rendering.copy_button_html(" in source and "rendering.COPY_JS" in source


def test_copy_strings_exist_in_both_languages():
    for lang in ("en", "ar"):
        for key in ("copy", "copied", "copy_failed"):
            assert i18n.STRINGS[lang][key].strip()


# ---------------------------------------------------------------------------
# 2. External-processing disclosure
# ---------------------------------------------------------------------------
def test_the_disclosure_names_the_provider_and_what_is_sent():
    en = i18n.STRINGS["en"]["about_processing"]
    assert "Groq" in en and "external" in en.lower()
    assert "question" in en.lower() and "passages" in en.lower()
    assert "personal or confidential" in en.lower()
    ar = i18n.STRINGS["ar"]["about_processing"]
    assert "Groq" in ar and re.search("[؀-ۿ]", ar)


def test_the_disclosure_promises_nothing_the_application_cannot_verify():
    en = i18n.STRINGS["en"]["about_processing"].lower()
    for phrase in ("not stored", "never stored", "deleted", "anonymous", "encrypted", "private", "secure", "offline", "on this device"):
        assert phrase not in en


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_the_about_block_of_every_page_shows_the_disclosure(lang):
    block = shell.about_html(lang)
    assert shell.esc(i18n.STRINGS[lang]["about"]) in block
    assert shell.esc(i18n.STRINGS[lang]["about_processing"]) in block


def test_the_settings_dialog_shows_the_disclosure_next_to_the_data_note():
    source = open(os.path.join(REPO, "ui", "chat.py"), encoding="utf-8").read()
    line = next(l for l in source.splitlines() if '"settings_data_note"' in l)
    assert '"about_processing"' in line


def test_the_voice_note_still_says_the_audio_goes_to_the_browser_speech_service():
    en = i18n.STRINGS["en"]["voice_privacy"].lower()
    assert "browser" in en and "servers" in en and "offline" not in en


# ---------------------------------------------------------------------------
# 3. Android server address
# ---------------------------------------------------------------------------
def _android(*parts: str) -> str:
    return os.path.join(REPO, "android", *parts)


@pytest.mark.skipif(not os.path.exists(_android("gradle.properties")), reason="the Android project is not part of this checkout")
def test_the_android_project_ships_a_placeholder_not_a_private_address():
    text = open(_android("gradle.properties"), encoding="utf-8").read()
    value = next(l.split("=", 1)[1].strip() for l in text.splitlines() if l.startswith("FOB_SERVER_URL="))
    assert "CHANGE-ME" in value                                   # what build.gradle.kts and MainActivity.kt both test for
    assert value.startswith("https://")
    private = re.compile(r"\.ts\.net|tailscale|\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b|\b192\.168\.\d+\.\d+\b|\b10\.\d+\.\d+\.\d+\b", re.I)
    assert not private.search(value)
    assert not re.search(r"\.ts\.net", text, re.I)                # not in a comment either


@pytest.mark.skipif(not os.path.exists(_android("app", "build.gradle.kts")), reason="the Android project is not part of this checkout")
def test_the_android_build_and_activity_still_recognise_the_placeholder():
    build = open(_android("app", "build.gradle.kts"), encoding="utf-8").read()
    activity = open(_android("app", "src", "main", "java", "app", "fob", "uosfaculty", "MainActivity.kt"), encoding="utf-8").read()
    assert 'contains("CHANGE-ME")' in build and 'contains("CHANGE-ME")' in activity
