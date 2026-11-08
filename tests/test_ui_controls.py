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
