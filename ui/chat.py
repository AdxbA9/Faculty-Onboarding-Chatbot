"""
NiceGUI interface for the UOS Faculty Onboarding Chatbot (post-freeze PF-10 redesign, PF-11 bilingual interface).

Layout:
    +------------------+--------------------------------------------------+
    |  Sidebar (aside) |  Header: h1, plain-language status, language,    |
    |  - brand         |          theme, New chat (phone), Developer      |
    |  - New chat      |  ------------------------------------------------ |
    |  - conversations |  Conversation (role=log): first-use screen or     |
    |    (this session)|  message cards with sources + supporting evidence |
    |  - about/status  |  ------------------------------------------------ |
    |                  |  Composer: textarea, Send, help / answer language |
    +------------------+--------------------------------------------------+

What this module decides: layout, wording of the shell and of the page's own notices (``ui.i18n`` in English and
Arabic), which backend sentence is shown as a labelled notice (``presentation.notice_kind``), how the answer HTML is made
safe (``rendering.answer_html``), and how the three languages of a turn are presented: the interface language
(``state["lang"]``), the question's language (detected by the backend) and the answer language preference
(``state["answer_lang"]``: auto / en / ar, passed to the seam). What it never decides: the answer, its evidence, its
citations, its refusal or its translation; those come from ``ui.pipeline`` unchanged.

Design notes:
- Every nested helper inside ``_build_page`` is defined BEFORE any UI construction runs.
- The whole shell lives under one root element and is rebuilt from the page state when the interface language changes,
  so switching language keeps every conversation and every message (D-122).
- The page ``client`` is captured at build time so handlers that outlive their originating element can still run
  JavaScript without "slot has been deleted" errors.
- Enter sends and Shift+Enter inserts a newline; the decision is made in the browser (``ENTER_SENDS_JS``, D-09).
- The sidebar is hidden under 900 px and the header carries New chat (D-10 kept at 540 px).
- ``<html lang dir>`` follow the interface language; message bubbles and answer blocks use ``dir="auto"`` so Arabic and
  English text each read in their own direction whatever the interface language.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Request
from nicegui import app, run, ui

from handbook_bot import QAResult
from handbook_bot.arabic_answers import localize_reply

from . import i18n, presentation, rendering
from .i18n import t
from .pipeline import groq_is_configured, load_knowledge_base, preflight
from .styles import GLOBAL_CSS, ICONS

_log = logging.getLogger(__name__)


APP_TITLE = "UOS Faculty Onboarding Chatbot"           # browser tab title (unchanged)
ORG_NAME = i18n.STRINGS["en"]["org"]
APP_NAME = i18n.STRINGS["en"]["app_name"]
APP_SUBTITLE = i18n.STRINGS["en"]["app_subtitle"]
ABOUT_TEXT = i18n.STRINGS["en"]["about"]
SESSION_NOTE = i18n.STRINGS["en"]["session_note"]

#: Runs in the browser: plain Enter sends, and any other key, Shift+Enter included, keeps its default action.
ENTER_SENDS_JS = "(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); emit(); } }"

#: The first-use screen: one card per specialist domain with example questions. Every example is checked to be answered
#: from the approved documents without a model call (``qa``: PF-10 example check; PF-11 for the Arabic examples), so each
#: shortcut works even when no model key is configured.
TOPICS = [
    {"id": "teaching", "icon": "teaching", "title": "Teaching & Learning",
     "blurb": "Teaching loads, Blackboard and classroom tools, assessment rules.",
     "examples": ["What is the teaching load for lecturers?", "How do I send an announcement to my students in Blackboard?"]},
    {"id": "research", "icon": "research", "title": "Research & Innovation",
     "blurb": "Research support, conference travel, supervision and publication rules.",
     "examples": ["Who approves conference travel requests?"]},
    {"id": "faculty_services", "icon": "services", "title": "Faculty Services & HR",
     "blurb": "Leave, benefits, appointments and human-resources procedures.",
     "examples": ["How many days of maternity leave are female faculty entitled to?"]},
    {"id": "institutional", "icon": "institutional", "title": "Institutional Navigation",
     "blurb": "Offices, contact numbers, governance and campus services.",
     "examples": ["What is the phone number of the Human Resources Department?", "What is the fax number of the Passport Unit?"]},
]
TOPICS_BY_LANG = {"en": TOPICS, "ar": i18n.TOPICS_AR}
EXAMPLE_QUESTIONS = [q for topic in TOPICS for q in topic["examples"]]
EXAMPLE_QUESTIONS_AR = [q for topic in i18n.TOPICS_AR for q in topic["examples"]]

LANDING_NOTE_KEYS = [("book", "note_grounded"), ("shield", "note_cited"), ("clock", "session_note"), ("globe", "note_arabic")]
LANDING_NOTES = [(icon, i18n.STRINGS["en"][key]) for icon, key in LANDING_NOTE_KEYS]

_LOGO_URL = "/static/logo.png"
_DEV_ENV = os.getenv("APP_DEVELOPER_MODE", "").strip().lower() in ("1", "true", "yes")
_NOTICE_KEYS = {"empty": ("notice_empty", "hint_empty"), "config": ("notice_config", "hint_config"), "outage": ("notice_outage", "hint_outage"),
                "limit": ("notice_limit", "hint_limit"), "error": ("notice_error", "hint_error")}


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_KB_STATE: Dict[str, Any] = {"kb": None, "error": None, "pdf_name": "", "stats": {}}
_CLIENTS: Dict[str, Dict[str, Any]] = {}


def _ensure_kb_loaded() -> None:
    if _KB_STATE["kb"] is not None or _KB_STATE["error"] is not None:
        return
    check = preflight()
    if not check.ok:
        _KB_STATE["error"] = check.message
        return
    try:
        kb = load_knowledge_base(verbose=True)
        _KB_STATE["kb"] = kb
        _KB_STATE["pdf_name"] = Path(kb.pdf_file).name
        _KB_STATE["stats"] = kb.stats or {}
    except Exception as exc:  # pragma: no cover
        _log.exception("knowledge base failed to load")          # the traceback goes to the server log only
        _KB_STATE["error"] = "Could not load the handbook (%s). Check the server log, then reload the app." % type(exc).__name__


def service_status(lang: str = "en") -> Dict[str, str]:
    """The plain-language status the header and the sidebar show (never a model id or a file name)."""
    if _KB_STATE["error"]:
        return {"css": "danger", "label": t(lang, "status_setup"), "title": _KB_STATE["error"]}
    if _KB_STATE["kb"] is None:
        return {"css": "warn", "label": t(lang, "status_starting"), "title": t(lang, "status_starting_title")}
    if not groq_is_configured():
        return {"css": "warn", "label": t(lang, "status_limited"), "title": t(lang, "status_limited_title")}
    return {"css": "ok", "label": t(lang, "status_ready"), "title": t(lang, "status_ready_title")}


# ---------------------------------------------------------------------------
# Per-client state
# ---------------------------------------------------------------------------
def _new_session(lang: str = "en") -> Dict[str, Any]:
    return {"id": uuid.uuid4().hex[:12], "title": "", "messages": [], "created_at": time.time()}


def _client_state(client_id: str, lang: str = "en") -> Dict[str, Any]:
    state = _CLIENTS.get(client_id)
    if state is None:
        s = _new_session(lang)
        state = {"sessions": [s], "active_id": s["id"], "debug": False, "theme": "light", "lang": i18n.normalize_lang(lang), "answer_lang": "auto"}
        _CLIENTS[client_id] = state
    return state


def _active_session(state: Dict[str, Any]) -> Dict[str, Any]:
    for sess in state["sessions"]:
        if sess["id"] == state["active_id"]:
            return sess
    s = _new_session(state.get("lang", "en"))
    state["sessions"].insert(0, s)
    state["active_id"] = s["id"]
    return s


def _safe(callable_, *args, **kwargs) -> None:
    """Run a UI call and swallow the context errors NiceGUI raises when the originating element is already gone."""
    try:
        callable_(*args, **kwargs)
    except Exception:
        pass


def _searching(lang: str) -> str:
    """The processing text: the backend-independent English constant the suite pins, or its Arabic counterpart."""
    return presentation.SEARCHING_TEXT if lang == "en" else t(lang, "searching")


def _title_for(question: str) -> str:
    q = " ".join(question.split())
    return (q[:42].rstrip() + "…") if len(q) > 44 else q


def notice_for(content: str, lang: str) -> Optional[Dict[str, str]]:
    """The labelled notice for a backend sentence in the interface language: kind, label, the sentence to show (the
    Arabic counterpart of the canonical English sentence when the interface is Arabic) and the page's hint."""
    notice = presentation.notice_kind(content)
    if not notice:
        return None
    kind = notice[0]
    label_key, hint_key = _NOTICE_KEYS.get(kind, ("notice_error", "hint_error"))
    label = t(lang, "notice_config_problem") if (kind == "config" and "configuration" in notice[1].lower()) else t(lang, label_key)
    text = content
    if lang == "ar":
        text = localize_reply(content) or (t("ar", "error_text") if content.strip() == presentation.ERROR_TEXT else content)
    return {"kind": kind, "label": label, "text": text, "hint": t(lang, hint_key)}


# ===========================================================================
# Page
# ===========================================================================
def _build_page(developer: bool = False, lang: str = "en") -> None:
    client_id = ui.context.client.id
    page_client = ui.context.client
    state = _client_state(client_id, lang)

    ui.add_head_html(GLOBAL_CSS)
    ui.query("html").props(f'data-theme={state["theme"]} lang={state["lang"]} dir={i18n.DIRECTION[state["lang"]]}')

    refs: Dict[str, Any] = {}          # live elements: root, sessions, chat, scroll, input, send, composer, help, status, theme
    is_busy: Dict[str, bool] = {"v": False}

    def L() -> str:
        return state["lang"]

    # =======================================================================
    # Helpers (defined before any construction)
    # =======================================================================
    def _set_help(text: str) -> None:
        el = refs.get("help")
        if el is not None:
            _safe(setattr, el, "content", html.escape(text) if text else "")

    def _set_live(text: str) -> None:
        el = refs.get("live")
        if el is not None:
            _safe(setattr, el, "content", html.escape(text) if text else "")

    def _status_html() -> str:
        s = service_status(L())
        return ('<div class="status-pill %s" role="status" title="%s"><span class="status-dot" aria-hidden="true"></span>'
                '<span class="status-text">%s</span><span class="sr-only">. %s</span></div>'
                % (s["css"], html.escape(s["title"], quote=True), html.escape(s["label"]), html.escape(s["title"])))

    def _refresh_status() -> None:
        el = refs.get("status")
        if el is not None:
            _safe(setattr, el, "content", _status_html())
        if _KB_STATE["kb"] is not None or _KB_STATE["error"] is not None:
            timer = refs.get("status_timer")
            if timer is not None:
                _safe(setattr, timer, "active", False)
            if not _active_session(state)["messages"]:
                _render_chat()                                   # the first-use screen drops its "starting up" banner

    def _theme_label() -> str:
        return t(L(), "theme_to_light") if state["theme"] == "dark" else t(L(), "theme_to_dark")

    def _on_theme_toggle() -> None:
        state["theme"] = "light" if state["theme"] == "dark" else "dark"
        _safe(ui.query("html").props, f'data-theme={state["theme"]}')
        btn = refs.get("theme")
        if btn is not None:
            _safe(btn.props, 'aria-label="%s"' % _theme_label())

    def _on_lang_toggle() -> None:
        state["lang"] = "ar" if state["lang"] == "en" else "en"
        _safe(ui.query("html").props, f'lang={state["lang"]} dir={i18n.DIRECTION[state["lang"]]}')
        _render_all()                                            # every conversation and message survives the rebuild

    def _on_answer_lang(value: Any) -> None:
        state["answer_lang"] = value if value in i18n.ANSWER_LANGUAGES else "auto"

    def _on_debug_toggle(value: bool) -> None:
        state["debug"] = bool(value)
        _render_chat()

    async def _on_new_chat() -> None:
        current = _active_session(state)
        if not current["messages"]:                              # already on an empty conversation: nothing to add
            _render_chat()
            return
        s = _new_session(L())
        state["sessions"].insert(0, s)
        state["active_id"] = s["id"]
        _render_sessions()
        _render_chat()

    def _render_sessions() -> None:
        container = refs.get("sessions")
        if container is None:
            return
        container.clear()
        with container:
            for sess in state["sessions"]:
                active = sess["id"] == state["active_id"]

                def _make_click(sid: str):
                    async def _click() -> None:
                        state["active_id"] = sid
                        _render_sessions()
                        _render_chat()
                    return _click

                btn = ui.button(on_click=_make_click(sess["id"])) \
                    .props('flat no-caps align=left' + (' aria-current="true"' if active else '')) \
                    .classes("session-item" + (" active" if active else "")).mark("session")
                with btn:
                    ui.html('<span class="session-title" dir="auto">%s</span>' % html.escape(sess["title"] or t(L(), "new_conversation")))

    # ---- first-use screen -------------------------------------------------
    def _render_landing(container) -> None:
        lang = L()
        with container:
            with ui.element("div").classes("landing"):
                ui.html(
                    '<div class="hero">'
                    f'  <div class="hero-logo"><img src="{_LOGO_URL}" alt="{html.escape(t(lang, "logo_alt"), quote=True)}"/></div>'
                    f'  <div class="hero-org">{html.escape(t(lang, "org"))}</div>'
                    f'  <h2>{html.escape(t(lang, "welcome", app=t(lang, "app_name")))}</h2>'
                    f'  <p>{html.escape(t(lang, "app_subtitle"))}</p>'
                    '</div>'
                )
                if _KB_STATE["error"]:
                    ui.html(f'<div class="banner danger" role="alert">{ICONS["alert"]}<div><strong>{html.escape(t(lang, "banner_setup"))}</strong> '
                            f'{html.escape(_KB_STATE["error"])}</div></div>')
                elif _KB_STATE["kb"] is None:
                    ui.html(f'<div class="banner" role="status">{ICONS["clock"]}<div><strong>{html.escape(t(lang, "banner_starting"))}</strong> '
                            f'{html.escape(t(lang, "banner_starting_text"))}</div></div>')
                elif not groq_is_configured():
                    ui.html(f'<div class="banner warn" role="status">{ICONS["info"]}<div><strong>{html.escape(t(lang, "banner_limited"))}</strong> '
                            f'{html.escape(t(lang, "banner_limited_text"))}</div></div>')

                ui.html(f'<h3 class="landing-h">{html.escape(t(lang, "what_to_ask"))}</h3>')
                with ui.element("div").classes("topic-grid"):
                    for topic in TOPICS_BY_LANG.get(lang, TOPICS):
                        with ui.element("section").classes("topic-card").props('aria-label="%s"' % html.escape(topic["title"], quote=True)):
                            ui.html(f'<div class="topic-title"><span class="topic-icon">{ICONS[topic["icon"]]}</span>'
                                    f'{html.escape(topic["title"])}</div>'
                                    f'<div class="topic-blurb">{html.escape(topic["blurb"])}</div>')
                            with ui.element("div").classes("example-list"):
                                for question in topic["examples"]:
                                    def _make_click(q: str):
                                        async def _click() -> None:
                                            # one tick later, so the click handler returns before its button is removed
                                            asyncio.create_task(_submit(q))
                                        return _click
                                    ui.button(question, on_click=_make_click(question), icon="arrow_forward") \
                                        .props('flat no-caps align=left').classes("example-btn").mark("example")

                ui.html('<ul class="landing-notes">' + "".join(
                    f'<li>{ICONS[icon]}<span>{html.escape(t(lang, key))}</span></li>' for icon, key in LANDING_NOTE_KEYS) + '</ul>')

    # ---- message parts ----------------------------------------------------
    def _render_sources(result: QAResult, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        lang = L()
        chips_html: List[str] = []
        domain_ids = [str(s) for s in ((diagnostics or {}).get("routing_final") or ([(diagnostics or {}).get("specialist_id")] if (diagnostics or {}).get("specialist_id") else []))]
        domains = [i18n.domain_name(lang, d) for d in domain_ids] if presentation.served_by_plan_f(diagnostics) else []
        if domains:                                              # the verified specialist path served this answer
            chips_html.append('<span class="chip verified" title="%s">%s%s</span>'
                              % (html.escape(t(lang, "verified_title", domains=", ".join(domains)), quote=True), ICONS["check"],
                                 html.escape(t(lang, "verified_by", domains=", ".join(domains)))))
        for css, label in presentation.source_chips(result):
            if css == "page" and lang == "ar":
                label = label.replace("Page ", t("ar", "page", page="").strip() + " ", 1)
            chips_html.append(f'<span class="chip {css}" dir="auto">{html.escape(label)}</span>')
        if chips_html:
            ui.html('<div class="sources" aria-label="%s"><span class="sources-label">%s</span>' % (html.escape(t(lang, "sources"), quote=True), html.escape(t(lang, "sources")))
                    + "".join(chips_html) + '</div>')
        rows = presentation.evidence_rows(result)
        if rows:
            items = []
            for r in rows:
                meta = []
                if r["document"]:
                    meta.append('<span class="ev-doc">%s</span>' % html.escape(r["document"]))
                if r["page"] is not None:
                    meta.append(html.escape(t(lang, "page", page=r["page"])))
                if r["section"]:
                    meta.append(html.escape(r["section"]))
                if r["ocr"]:
                    meta.append(html.escape(t(lang, "ocr")))
                items.append('<li class="evidence-item"><div class="ev-meta" dir="auto">%s</div><blockquote class="ev-text" dir="auto">%s</blockquote></li>'
                             % (" · ".join(meta) or html.escape(t(lang, "source_document")), html.escape(r["excerpt"])))
            ui.html('<details class="evidence"><summary>%s<span>%s</span></summary><ol class="evidence-list">%s</ol></details>'
                    % (ICONS["chevron"], html.escape(t(lang, "evidence", count=len(items))), "".join(items)))

    def _render_translation(result: QAResult) -> None:
        """The status of an Arabic answer's text: translated and checked, or the English source shown verbatim; and the
        verified English intermediate on request."""
        lang = L()
        status = (result.translation or {}).get("status") if isinstance(result.translation, dict) else None
        if result.answer_language != "ar" or not status or status == "fixed":
            return
        if status == "translated":
            ui.html(f'<div class="translation-note">{ICONS["check"]}<span>{html.escape(t(lang, "translation_checked"))}</span></div>')
        else:
            ui.html(f'<div class="translation-note">{ICONS["info"]}<span>{html.escape(t(lang, "translation_framed"))}</span></div>')
        if result.original_answer and status == "translated":
            ui.html('<details class="original"><summary>%s<span>%s</span></summary><div class="original-body" dir="ltr" lang="en">%s</div></details>'
                    % (ICONS["chevron"], html.escape(t(lang, "show_original")), rendering.answer_html(presentation.escape_identifiers(result.original_answer))))

    def _render_toolbar(text: str) -> None:
        lang = L()
        js_literal = json.dumps(text)
        copy_js = ("(function(btn){ navigator.clipboard.writeText(%s).then(function(){ var l = btn.querySelector('.copy-lbl'); "
                   "if(!l) return; var o = l.innerText; l.innerText = %s; setTimeout(function(){ l.innerText = o; }, 1400); }); })(this);"
                   % (js_literal, json.dumps(t(lang, "copied"))))
        ui.html(f'<div class="msg-toolbar"><button type="button" class="icon-btn" aria-label="{html.escape(t(lang, "copy"), quote=True)}" onclick=\'{copy_js}\'>'
                f'{ICONS["copy"]}<span class="copy-lbl">{html.escape(t(lang, "copy"))}</span></button></div>')

    def _render_debug(msg: Dict[str, Any]) -> None:
        result: QAResult = msg["result"]
        sections = presentation.dev_sections(result, msg.get("diagnostics"), _KB_STATE.get("stats") or {}, msg.get("wall_ms"))

        def _rows(rs) -> str:
            return "".join(f'<div class="dev-row"><span class="k">{html.escape(k)}</span>'
                           f'<span class="v" title="{html.escape(v, quote=True)}" dir="auto">{html.escape(v)}</span></div>' for k, v in rs)

        body = "".join(f'<div class="dev-section">{html.escape(title)}</div>' + _rows(rows) for title, rows in sections if rows)
        ui.html(f'<div class="dev-panel" dir="ltr"><details open><summary>{html.escape(t(L(), "developer_details"))}</summary><div class="dev-body">{body}</div></details></div>')

    def _render_message(msg: Dict[str, Any]) -> None:
        lang = L()
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            with ui.element("div").classes("msg-row user").props('role="article" aria-label="%s"' % html.escape(t(lang, "your_message"), quote=True)):
                ui.html(f'<div class="bubble user" dir="auto">{html.escape(content)}</div>')
                ui.html(f'<div class="avatar user" aria-hidden="true">{html.escape(t(lang, "you"))}</div>')
            return

        result: Optional[QAResult] = msg.get("result")
        with ui.element("div").classes("msg-row assistant").props('role="article" aria-label="%s"' % html.escape(t(lang, "assistant_answer"), quote=True)):
            ui.html(f'<div class="avatar assistant" aria-hidden="true"><img src="{_LOGO_URL}" alt=""/></div>')
            with ui.element("div").classes("assistant-wrap"):
                if content == "__typing__":
                    with ui.element("div").classes("bubble assistant"):
                        ui.html('<div class="typing" role="status" aria-live="polite"><span class="dots" aria-hidden="true"><span></span><span></span><span></span></span>'
                                f'<span>{html.escape(_searching(lang))}</span></div>')
                    return
                notice = notice_for(content, lang)
                if notice:
                    kind = notice["kind"]
                    icon = ICONS["alert"] if kind in ("error", "outage", "config", "limit") else ICONS["info"]
                    ui.html(f'<div class="bubble assistant notice {kind}" role="status" dir="auto"><div class="notice-head">{icon}<span>{html.escape(notice["label"])}</span></div>'
                            f'<p class="notice-text" dir="auto">{html.escape(notice["text"])}</p>'
                            + (f'<p class="notice-hint">{html.escape(notice["hint"])}</p>' if notice["hint"] else "") + '</div>')
                else:
                    answer_lang = getattr(result, "answer_language", "en") if result is not None else "en"
                    with ui.element("div").classes("bubble assistant").props('dir="auto" lang="%s"' % answer_lang):
                        ui.html(rendering.answer_html(presentation.escape_identifiers(content))).classes("answer")
                if result is not None:
                    if not notice:                           # a refusal or a service notice is never dressed with evidence
                        _render_translation(result)
                        _render_sources(result, msg.get("diagnostics"))
                        _render_toolbar(content)
                    if state["debug"]:
                        _render_debug(msg)

    def _render_chat() -> None:
        container = refs.get("chat")
        if container is None:
            return
        try:
            container.clear()
        except Exception:
            return
        sess = _active_session(state)
        if not sess["messages"]:
            _render_landing(container)
            return
        try:
            with container:
                for msg in sess["messages"]:
                    _render_message(msg)
        except Exception:
            return
        _scroll_to_bottom()

    def _scroll_to_bottom() -> None:
        scroll_el = refs.get("scroll")
        if scroll_el is None:
            return
        try:
            page_client.run_javascript(f"setTimeout(() => {{ const e = getElement({scroll_el.id}); if (e) e.scrollTop = e.scrollHeight; }}, 30);")
        except Exception:
            pass

    def _set_busy(busy: bool) -> None:
        is_busy["v"] = busy
        send = refs.get("send")
        composer = refs.get("composer")
        if send is not None:
            _safe(send.props, "disable") if busy else _safe(send.props, remove="disable")
        if composer is not None:
            _safe(composer.classes, add="busy") if busy else _safe(composer.classes, remove="busy")
        _set_live(_searching(L()) if busy else "")

    async def _submit(question: str) -> None:
        question = (question or "").strip()
        if is_busy["v"]:
            _set_help(t(L(), "help_busy"))
            return
        if not question:
            _set_help(t(L(), "help_empty"))
            return
        _set_help("")
        if _KB_STATE["kb"] is None:
            _set_help(_KB_STATE["error"] or t(L(), "help_loading"))
            return

        sess = _active_session(state)
        if not sess["messages"]:
            sess["title"] = _title_for(question)
            _render_sessions()
        sess["messages"].append({"role": "user", "content": question, "result": None})
        sess["messages"].append({"role": "assistant", "content": "__typing__", "result": None})
        try:
            refs["input"].value = ""
        except Exception:
            pass
        _set_busy(True)
        _render_chat()
        try:
            result, diagnostics, wall_ms = await run.io_bound(_call_pipeline, _KB_STATE["kb"], question, state["answer_lang"])
        except Exception as exc:  # pragma: no cover
            result = QAResult(answer=presentation.ERROR_TEXT)
            diagnostics, wall_ms = presentation.error_diagnostics(exc), None
        sess["messages"][-1] = presentation.assistant_message(result, diagnostics, wall_ms)
        _set_busy(False)
        _render_chat()
        _set_live(t(L(), "answer_ready"))

    def _on_keydown_send(_e) -> None:
        val = (refs["input"].value or "").strip()
        if not val:
            _set_help(t(L(), "help_empty"))
            return
        if is_busy["v"]:
            _set_help(t(L(), "help_busy"))
            return
        try:
            refs["input"].value = ""
        except Exception:
            pass
        asyncio.create_task(_submit(val))

    async def _on_send_click() -> None:
        await _submit(refs["input"].value or "")

    # =======================================================================
    # Shell (rebuilt from the page state when the interface language changes)
    # =======================================================================
    def _render_all() -> None:
        root = refs["root"]
        lang = L()
        try:
            root.clear()
        except Exception:
            return
        with root:
            ui.html(f'<a class="skip-link" href="#composer-input">{html.escape(t(lang, "skip"))}</a>')
            with ui.element("div").classes("app-shell"):
                sidebar = ui.element("aside").classes("sidebar").props('aria-label="%s"' % html.escape(t(lang, "conversations"), quote=True))
                main = ui.element("main").classes("main")

            # ---------- Sidebar ------------------------------------------------
            with sidebar:
                ui.html(f'<div class="brand"><div class="brand-logo"><img src="{_LOGO_URL}" alt="{html.escape(t(lang, "logo_alt"), quote=True)}"/></div>'
                        f'<div class="brand-text"><div class="brand-org">{html.escape(t(lang, "org"))}</div>'
                        f'<div class="brand-name">{html.escape(t(lang, "app_name"))}</div></div></div>')
                ui.button(t(lang, "new_chat"), on_click=_on_new_chat, icon="add").props('unelevated no-caps').classes("new-chat-btn").mark("new-chat")
                ui.html(f'<div class="sb-label">{html.escape(t(lang, "conversations"))}</div>')
                refs["sessions"] = ui.element("nav").classes("session-list").props('aria-label="%s"' % html.escape(t(lang, "conversations_aria"), quote=True))
                ui.html(f'<div class="sb-note">{html.escape(t(lang, "session_note"))}</div>')
                with ui.element("div").classes("sb-footer"):
                    ui.html(f'<div class="about"><strong>{html.escape(t(lang, "about_lead"))}</strong> {html.escape(t(lang, "about"))}</div>')

            # ---------- Main ---------------------------------------------------
            with main:
                with ui.element("header").classes("topbar"):
                    ui.html(f'<div class="topbar-left"><div class="topbar-brand" aria-hidden="true"><img src="{_LOGO_URL}" alt=""/></div>'
                            f'<div><h1 class="topbar-title">{html.escape(t(lang, "app_name"))}</h1><div class="topbar-sub">{html.escape(t(lang, "org"))}</div></div></div>').classes("tl")
                    with ui.element("div").classes("topbar-right"):
                        refs["status"] = ui.html(_status_html())
                        if developer:
                            with ui.element("div").classes("dev-toggle"):
                                ui.html(f'<span>{html.escape(t(lang, "developer"))}</span>')
                                ui.switch(value=state["debug"], on_change=lambda e: _on_debug_toggle(e.value)).props('dense aria-label="%s"' % html.escape(t(lang, "developer_aria"), quote=True))
                        # New chat at phone and tablet width, where the sidebar is hidden (D-10).
                        phone_new_btn = ui.button(on_click=_on_new_chat).props('flat dense aria-label="%s"' % html.escape(t(lang, "new_chat"), quote=True)).classes("topbar-btn topbar-new-chat").mark("new-chat-phone")
                        with phone_new_btn:
                            ui.html(ICONS["plus"])
                        # Interface language: the label names the OTHER language, as is conventional for a switch.
                        ui.button(t(lang, "lang_switch"), on_click=_on_lang_toggle).props('flat dense no-caps aria-label="%s" lang="%s"' % (
                            html.escape(t(lang, "lang_switch_aria"), quote=True), "ar" if lang == "en" else "en")).classes("topbar-btn topbar-lang").mark("lang")
                        theme_btn = ui.button(on_click=_on_theme_toggle).props('flat dense aria-label="%s"' % html.escape(_theme_label(), quote=True)).classes("topbar-btn").mark("theme")
                        refs["theme"] = theme_btn
                        with theme_btn:
                            ui.html(ICONS["moon"] + ICONS["sun"])

                chat_scroll = ui.element("div").classes("chat-scroll")
                with chat_scroll:
                    refs["chat"] = ui.element("section").classes("chat-inner").props('aria-label="%s" role="log" aria-live="polite" aria-relevant="additions"'
                                                                                      % html.escape(t(lang, "conversations"), quote=True))
                refs["scroll"] = chat_scroll

                with ui.element("div").classes("composer-wrap").props('aria-label="%s"' % html.escape(t(lang, "composer_aria"), quote=True)):
                    composer = ui.element("div").classes("composer")
                    refs["composer"] = composer
                    with composer:
                        txt = ui.textarea(placeholder=t(lang, "composer_placeholder")) \
                            .props('autogrow borderless dense for=composer-input dir=auto aria-label="%s"' % html.escape(t(lang, "composer_aria"), quote=True)) \
                            .classes("flex-1").mark("composer")
                        refs["input"] = txt
                        send_btn = ui.button(t(lang, "send"), on_click=_on_send_click, icon="arrow_upward") \
                            .props('unelevated dense no-caps aria-label="%s"' % html.escape(t(lang, "send_aria"), quote=True)).classes("send-btn").mark("send")
                        refs["send"] = send_btn
                    with ui.element("div").classes("composer-foot"):
                        refs["help"] = ui.html("").classes("composer-help").props('role="status" aria-live="polite"')
                        with ui.element("div").classes("answer-lang"):
                            ui.html(f'<span class="lbl">{html.escape(t(lang, "answer_language"))}</span>')
                            options = {"auto": t(lang, "answer_auto"), "en": t(lang, "answer_en"), "ar": t(lang, "answer_ar")}
                            ui.select(options, value=state["answer_lang"], on_change=lambda e: _on_answer_lang(e.value)) \
                                .props('dense borderless options-dense aria-label="%s"' % html.escape(t(lang, "answer_language"), quote=True)).mark("answer-lang")
                        ui.html('<div class="composer-hint"><kbd>%s</kbd> %s</div>' % (
                            html.escape(t(lang, "hint_enter")),
                            html.escape(t(lang, "hint_text", shift=t(lang, "hint_shift"), enter=t(lang, "hint_enter"))).replace(
                                html.escape(t(lang, "hint_shift")), "<kbd>%s</kbd>" % html.escape(t(lang, "hint_shift")), 1).replace(
                                "+" + html.escape(t(lang, "hint_enter")), "+<kbd>%s</kbd>" % html.escape(t(lang, "hint_enter")), 1)))
                    refs["live"] = ui.html("").classes("sr-only").props('role="status" aria-live="polite"')

                # Enter sends; Shift+Enter keeps the textarea's newline. Decided in the browser (D-09).
                txt.on("keydown", _on_keydown_send, js_handler=ENTER_SENDS_JS)

        _render_sessions()
        _render_chat()

    refs["root"] = ui.element("div").classes("app-root")
    if _KB_STATE["kb"] is None and _KB_STATE["error"] is None:
        refs["status_timer"] = ui.timer(2.0, _refresh_status)

    # =======================================================================
    # Initial render
    # =======================================================================
    _render_all()


def _call_pipeline(kb, question: str, answer_language: str = "auto"):
    """One turn: the answer, the diagnostics the seam recorded for it and the end-to-end wall time of the call."""
    from .pipeline import ask_with_diagnostics
    started = time.perf_counter()
    result, diagnostics = ask_with_diagnostics(kb, question, answer_language=answer_language)
    return result, diagnostics, (time.perf_counter() - started) * 1000.0


def developer_requested(request: Optional[Request]) -> bool:
    """Developer diagnostics are shown only when the page is opened with ``?dev=1`` (or ``APP_DEVELOPER_MODE=1``)."""
    if _DEV_ENV:
        return True
    try:
        value = (request.query_params.get("dev") if request is not None else "") or ""
    except Exception:
        value = ""
    return value.strip().lower() in ("1", "true", "yes")


def language_requested(request: Optional[Request]) -> str:
    """The initial interface language: ``?lang=ar|en``, else the browser's Accept-Language when it prefers Arabic, else English."""
    try:
        query_lang = request.query_params.get("lang") if request is not None else None
        accept = request.headers.get("accept-language") if request is not None else None
    except Exception:
        query_lang, accept = None, None
    return i18n.lang_from_request(query_lang, accept)


def register_pages() -> None:
    """Register the chat page on the NiceGUI app (called by ``start_app`` and by the page tests)."""
    @ui.page("/")
    def index(request: Request) -> None:
        _build_page(developer=developer_requested(request), lang=language_requested(request))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def start_app(port: int = 8501, host: Optional[str] = None) -> None:
    """Serve the application. Post-freeze PF-8 (D-119): it listens on this computer only (``127.0.0.1``) unless
    ``APP_HOST`` (or ``host``) says otherwise, because the development server has no authentication and the previous
    default announced itself on every network address."""
    host = host or os.getenv("APP_HOST", "").strip() or "127.0.0.1"
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    if assets_dir.is_dir():
        app.add_static_files("/static", str(assets_dir))

    app.on_startup(_ensure_kb_loaded)
    register_pages()

    ui.run(
        host=host,
        port=port,
        title=APP_TITLE,
        dark=None,
        reload=False,
        show=False,
        favicon=(str(assets_dir / "logo.png") if assets_dir.is_dir() else None),
        storage_secret=secrets.token_hex(16),
    )
