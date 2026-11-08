"""
NiceGUI interface for the UOS Faculty Onboarding Chatbot (post-freeze PF-10 redesign).

Layout:
    +------------------+--------------------------------------------------+
    |  Sidebar (aside) |  Header: h1, plain-language status, theme,        |
    |  - brand         |          New chat (phone), Developer (?dev=1)     |
    |  - New chat      |  ------------------------------------------------ |
    |  - conversations |  Conversation (role=log): first-use screen or     |
    |    (this session)|  message cards with sources + supporting evidence |
    |  - about/status  |  ------------------------------------------------ |
    |                  |  Composer: textarea, Send, help/status line        |
    +------------------+--------------------------------------------------+

What this module decides: layout, wording of the shell and of the page's own notices, which backend sentence is shown
as a labelled notice (``presentation.notice_kind``), and how the answer HTML is made safe (``rendering.answer_html``).
What it never decides: the answer, its evidence, its citations or its refusal; those come from ``ui.pipeline`` unchanged.

Design notes:
- Every nested helper inside ``_build_page`` is defined BEFORE any UI construction runs (a def after its call site would
  be an UnboundLocalError, because Python treats every name assigned in the function as local).
- The page ``client`` is captured at build time so handlers that outlive their originating element (an example button
  that disappears when the first answer renders) can still run JavaScript without "slot has been deleted" errors.
- Enter sends and Shift+Enter inserts a newline; the decision is made in the browser (``ENTER_SENDS_JS``, D-09).
- The sidebar is hidden under 900 px and the header carries New chat (D-10 kept at 540 px).
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

from . import presentation, rendering
from .pipeline import groq_is_configured, load_knowledge_base, preflight
from .styles import GLOBAL_CSS, ICONS

_log = logging.getLogger(__name__)


APP_TITLE = "UOS Faculty Onboarding Chatbot"           # browser tab title (unchanged)
ORG_NAME = "University of Sharjah"
APP_NAME = "Faculty Onboarding Assistant"
APP_SUBTITLE = ("Answers for faculty members about university policies, procedures, services and teaching tools, "
                "grounded in approved University of Sharjah documents and cited to the page they come from.")
ABOUT_TEXT = ("Answers come from the UOS Faculty Handbook 2025/26 and approved official guides. "
              "When the documents do not state something, the assistant says so instead of guessing.")
SESSION_NOTE = "Conversations are kept for this browser session only."

#: Runs in the browser: plain Enter sends, and any other key, Shift+Enter included, keeps its default action.
ENTER_SENDS_JS = "(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); emit(); } }"

#: The first-use screen: one card per specialist domain with example questions. Every example is checked to be answered
#: from the approved documents without a model call (``qa``: PF-10 example check), so each shortcut works even when no
#: model key is configured.
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
EXAMPLE_QUESTIONS = [q for topic in TOPICS for q in topic["examples"]]

LANDING_NOTES = [
    ("book", "Grounded in approved documents: the Faculty Handbook 2025/26 and official university guides."),
    ("shield", "Every answer names its source and the printed page, so you can check it."),
    ("clock", SESSION_NOTE),
]

_LOGO_URL = "/static/logo.png"
_DEV_ENV = os.getenv("APP_DEVELOPER_MODE", "").strip().lower() in ("1", "true", "yes")


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


def service_status() -> Dict[str, str]:
    """The plain-language status the header and the sidebar show (never a model id or a file name)."""
    if _KB_STATE["error"]:
        return {"css": "danger", "label": "Setup needed", "title": _KB_STATE["error"]}
    if _KB_STATE["kb"] is None:
        return {"css": "warn", "label": "Starting up", "title": "The document index is loading. First start can take a minute."}
    if not groq_is_configured():
        return {"css": "warn", "label": "Limited service",
                "title": "Contact, date and count lookups work; open questions need the language-model service, which is not configured."}
    return {"css": "ok", "label": "Ready", "title": "Documents indexed and the answer service configured."}


# ---------------------------------------------------------------------------
# Per-client state
# ---------------------------------------------------------------------------
def _new_session() -> Dict[str, Any]:
    return {"id": uuid.uuid4().hex[:12], "title": "New conversation", "messages": [], "created_at": time.time()}


def _client_state(client_id: str) -> Dict[str, Any]:
    state = _CLIENTS.get(client_id)
    if state is None:
        s = _new_session()
        state = {"sessions": [s], "active_id": s["id"], "debug": False, "theme": "light"}
        _CLIENTS[client_id] = state
    return state


def _active_session(state: Dict[str, Any]) -> Dict[str, Any]:
    for sess in state["sessions"]:
        if sess["id"] == state["active_id"]:
            return sess
    s = _new_session()
    state["sessions"].insert(0, s)
    state["active_id"] = s["id"]
    return s


def _safe(callable_, *args, **kwargs) -> None:
    """Run a UI call and swallow the context errors NiceGUI raises when the originating element is already gone."""
    try:
        callable_(*args, **kwargs)
    except Exception:
        pass


def _title_for(question: str) -> str:
    q = " ".join(question.split())
    return (q[:42].rstrip() + "…") if len(q) > 44 else q


# ===========================================================================
# Page
# ===========================================================================
def _build_page(developer: bool = False) -> None:
    client_id = ui.context.client.id
    page_client = ui.context.client
    state = _client_state(client_id)

    ui.add_head_html(GLOBAL_CSS)
    ui.query("html").props(f'data-theme={state["theme"]}')

    refs: Dict[str, Any] = {}          # live elements: sessions, chat, scroll, input, send, composer, help, status, theme
    is_busy: Dict[str, bool] = {"v": False}

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
        s = service_status()
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

    def _on_theme_toggle() -> None:
        state["theme"] = "light" if state["theme"] == "dark" else "dark"
        _safe(ui.query("html").props, f'data-theme={state["theme"]}')
        btn = refs.get("theme")
        if btn is not None:
            _safe(btn.props, 'aria-label="%s"' % _theme_label())

    def _theme_label() -> str:
        return "Switch to light mode" if state["theme"] == "dark" else "Switch to dark mode"

    def _on_debug_toggle(value: bool) -> None:
        state["debug"] = bool(value)
        _render_chat()

    async def _on_new_chat() -> None:
        current = _active_session(state)
        if not current["messages"]:                              # already on an empty conversation: nothing to add
            _render_chat()
            return
        s = _new_session()
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
                    ui.html('<span class="session-title">%s</span>' % html.escape(sess["title"] or "New conversation"))

    # ---- first-use screen -------------------------------------------------
    def _render_landing(container) -> None:
        with container:
            with ui.element("div").classes("landing"):
                ui.html(
                    '<div class="hero">'
                    f'  <div class="hero-logo"><img src="{_LOGO_URL}" alt="University of Sharjah logo"/></div>'
                    f'  <div class="hero-org">{html.escape(ORG_NAME)}</div>'
                    f'  <h2>Welcome to the {html.escape(APP_NAME)}</h2>'
                    f'  <p>{html.escape(APP_SUBTITLE)}</p>'
                    '</div>'
                )
                if _KB_STATE["error"]:
                    ui.html(f'<div class="banner danger" role="alert">{ICONS["alert"]}<div><strong>Setup required.</strong> '
                            f'{html.escape(_KB_STATE["error"])}</div></div>')
                elif _KB_STATE["kb"] is None:
                    ui.html(f'<div class="banner" role="status">{ICONS["clock"]}<div><strong>Starting up.</strong> '
                            'The document index and language models are loading; the first start can take a minute.</div></div>')
                elif not groq_is_configured():
                    ui.html(f'<div class="banner warn" role="status">{ICONS["info"]}<div><strong>Limited service.</strong> '
                            'Contact, date and count lookups work. Open questions need the language-model service, which is not '
                            'configured on this installation.</div></div>')

                ui.html('<h3 class="landing-h">What you can ask</h3>')
                with ui.element("div").classes("topic-grid"):
                    for topic in TOPICS:
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
                                    ui.button(question, on_click=_make_click(question), icon="arrow_forward")                                         .props('flat no-caps align=left').classes("example-btn").mark("example")

                ui.html('<ul class="landing-notes">' + "".join(
                    f'<li>{ICONS[icon]}<span>{html.escape(text)}</span></li>' for icon, text in LANDING_NOTES) + '</ul>')

    # ---- message parts ----------------------------------------------------
    def _render_sources(result: QAResult, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        chips_html: List[str] = []
        domains = presentation.domain_labels(diagnostics)
        if domains:                                              # the verified specialist path served this answer
            chips_html.append('<span class="chip verified" title="Verified from the approved documents by the %s specialist">%s%s</span>'
                              % (html.escape(", ".join(domains), quote=True), ICONS["check"], html.escape("Verified · " + ", ".join(domains))))
        for css, label in presentation.source_chips(result):
            chips_html.append(f'<span class="chip {css}">{html.escape(label)}</span>')
        if chips_html:
            ui.html('<div class="sources" aria-label="Sources"><span class="sources-label">Sources</span>' + "".join(chips_html) + '</div>')
        rows = presentation.evidence_rows(result)
        if rows:
            items = []
            for r in rows:
                meta = []
                if r["document"]:
                    meta.append('<span class="ev-doc">%s</span>' % html.escape(r["document"]))
                if r["page"] is not None:
                    meta.append("Page %d" % r["page"])
                if r["section"]:
                    meta.append(html.escape(r["section"]))
                if r["ocr"]:
                    meta.append("image text (OCR)")
                items.append('<li class="evidence-item"><div class="ev-meta">%s</div><blockquote class="ev-text" dir="auto">%s</blockquote></li>'
                             % (" · ".join(meta) or "Source document", html.escape(r["excerpt"])))
            ui.html('<details class="evidence"><summary>%s<span>Supporting evidence (%d)</span></summary><ol class="evidence-list">%s</ol></details>'
                    % (ICONS["chevron"], len(items), "".join(items)))

    def _render_toolbar(text: str) -> None:
        js_literal = json.dumps(text)
        copy_js = ("(function(btn){ navigator.clipboard.writeText(%s).then(function(){ var l = btn.querySelector('.copy-lbl'); "
                   "if(!l) return; var o = l.innerText; l.innerText = 'Copied'; setTimeout(function(){ l.innerText = o; }, 1400); }); })(this);"
                   % js_literal)
        ui.html(f'<div class="msg-toolbar"><button type="button" class="icon-btn" aria-label="Copy answer" onclick=\'{copy_js}\'>'
                f'{ICONS["copy"]}<span class="copy-lbl">Copy answer</span></button></div>')

    def _render_debug(msg: Dict[str, Any]) -> None:
        result: QAResult = msg["result"]
        sections = presentation.dev_sections(result, msg.get("diagnostics"), _KB_STATE.get("stats") or {}, msg.get("wall_ms"))

        def _rows(rs) -> str:
            return "".join(f'<div class="dev-row"><span class="k">{html.escape(k)}</span>'
                           f'<span class="v" title="{html.escape(v, quote=True)}">{html.escape(v)}</span></div>' for k, v in rs)

        body = "".join(f'<div class="dev-section">{html.escape(title)}</div>' + _rows(rows) for title, rows in sections if rows)
        ui.html(f'<div class="dev-panel"><details open><summary>Developer details</summary><div class="dev-body">{body}</div></details></div>')

    def _render_message(msg: Dict[str, Any]) -> None:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            with ui.element("div").classes("msg-row user").props('role="article" aria-label="Your message"'):
                ui.html(f'<div class="bubble user" dir="auto">{html.escape(content)}</div>')
                ui.html('<div class="avatar user" aria-hidden="true">YOU</div>')
            return

        result: Optional[QAResult] = msg.get("result")
        with ui.element("div").classes("msg-row assistant").props('role="article" aria-label="Assistant answer"'):
            ui.html(f'<div class="avatar assistant" aria-hidden="true"><img src="{_LOGO_URL}" alt=""/></div>')
            with ui.element("div").classes("assistant-wrap"):
                if content == "__typing__":
                    with ui.element("div").classes("bubble assistant"):
                        ui.html('<div class="typing" role="status" aria-live="polite"><span class="dots" aria-hidden="true"><span></span><span></span><span></span></span>'
                                f'<span>{html.escape(presentation.SEARCHING_TEXT)}</span></div>')
                    return
                notice = presentation.notice_kind(content)
                if notice:
                    kind, label = notice
                    icon = ICONS["alert"] if kind in ("error", "outage", "config", "limit") else ICONS["info"]
                    hint = presentation.NOTICE_HINTS.get(kind, "")
                    ui.html(f'<div class="bubble assistant notice {kind}" role="status"><div class="notice-head">{icon}<span>{html.escape(label)}</span></div>'
                            f'<p class="notice-text" dir="auto">{html.escape(content)}</p>'
                            + (f'<p class="notice-hint">{html.escape(hint)}</p>' if hint else "") + '</div>')
                else:
                    with ui.element("div").classes("bubble assistant").props('dir="auto"'):
                        ui.html(rendering.answer_html(presentation.escape_identifiers(content))).classes("answer")
                if result is not None:
                    if not notice:                           # a refusal or a service notice is never dressed with evidence
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
        _set_live(presentation.SEARCHING_TEXT if busy else "")

    async def _submit(question: str) -> None:
        question = (question or "").strip()
        if is_busy["v"]:
            _set_help("Please wait for the current answer to finish.")
            return
        if not question:
            _set_help("Type a question to send.")
            return
        _set_help("")
        if _KB_STATE["kb"] is None:
            _set_help(_KB_STATE["error"] or "The document index is still loading. Please try again in a moment.")
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
            result, diagnostics, wall_ms = await run.io_bound(_call_pipeline, _KB_STATE["kb"], question)
        except Exception as exc:  # pragma: no cover
            result = QAResult(answer=presentation.ERROR_TEXT)
            diagnostics, wall_ms = presentation.error_diagnostics(exc), None
        sess["messages"][-1] = presentation.assistant_message(result, diagnostics, wall_ms)
        _set_busy(False)
        _render_chat()
        _set_live("Answer ready.")

    def _on_keydown_send(_e) -> None:
        val = (refs["input"].value or "").strip()
        if not val:
            _set_help("Type a question to send.")
            return
        if is_busy["v"]:
            _set_help("Please wait for the current answer to finish.")
            return
        try:
            refs["input"].value = ""
        except Exception:
            pass
        asyncio.create_task(_submit(val))

    async def _on_send_click() -> None:
        await _submit(refs["input"].value or "")

    # =======================================================================
    # Shell
    # =======================================================================
    ui.html('<a class="skip-link" href="#composer-input">Skip to the message box</a>')
    with ui.element("div").classes("app-shell"):
        sidebar = ui.element("aside").classes("sidebar").props('aria-label="Conversations"')
        main = ui.element("main").classes("main")

    # ---------- Sidebar ----------------------------------------------------
    with sidebar:
        ui.html(f'<div class="brand"><div class="brand-logo"><img src="{_LOGO_URL}" alt="University of Sharjah logo"/></div>'
                f'<div class="brand-text"><div class="brand-org">{html.escape(ORG_NAME)}</div>'
                f'<div class="brand-name">{html.escape(APP_NAME)}</div></div></div>')
        ui.button("New chat", on_click=_on_new_chat, icon="add").props('unelevated no-caps').classes("new-chat-btn").mark("new-chat")
        ui.html('<div class="sb-label">Conversations</div>')
        refs["sessions"] = ui.element("nav").classes("session-list").props('aria-label="Conversations in this session"')
        ui.html(f'<div class="sb-note">{html.escape(SESSION_NOTE)}</div>')
        with ui.element("div").classes("sb-footer"):
            ui.html(f'<div class="about"><strong>About this assistant.</strong> {html.escape(ABOUT_TEXT)}</div>')

    # ---------- Main -------------------------------------------------------
    with main:
        with ui.element("header").classes("topbar"):
            ui.html(f'<div class="topbar-left"><div class="topbar-brand" aria-hidden="true"><img src="{_LOGO_URL}" alt=""/></div>'
                    f'<div><h1 class="topbar-title">{html.escape(APP_NAME)}</h1><div class="topbar-sub">{html.escape(ORG_NAME)}</div></div></div>').classes("tl")
            with ui.element("div").classes("topbar-right"):
                refs["status"] = ui.html(_status_html())
                if developer:
                    with ui.element("div").classes("dev-toggle"):
                        ui.html('<span>Developer</span>')
                        ui.switch(value=state["debug"], on_change=lambda e: _on_debug_toggle(e.value)).props('dense aria-label="Developer mode"')
                # New chat at phone and tablet width, where the sidebar is hidden (D-10).
                phone_new_btn = ui.button(on_click=_on_new_chat).props('flat dense aria-label="New chat"').classes("topbar-btn topbar-new-chat").mark("new-chat-phone")
                with phone_new_btn:
                    ui.html(ICONS["plus"])
                theme_btn = ui.button(on_click=_on_theme_toggle).props('flat dense aria-label="%s"' % _theme_label()).classes("topbar-btn").mark("theme")
                refs["theme"] = theme_btn
                with theme_btn:
                    ui.html(ICONS["moon"] + ICONS["sun"])

        chat_scroll = ui.element("div").classes("chat-scroll")
        with chat_scroll:
            refs["chat"] = ui.element("section").classes("chat-inner").props('aria-label="Conversation" role="log" aria-live="polite" aria-relevant="additions"')
        refs["scroll"] = chat_scroll

        with ui.element("div").classes("composer-wrap").props('aria-label="Message composer"'):
            composer = ui.element("div").classes("composer")
            refs["composer"] = composer
            with composer:
                txt = ui.textarea(placeholder="Ask about a policy, procedure, office or teaching tool…") \
                    .props('autogrow borderless dense for=composer-input dir=auto aria-label="Message the Onboarding Chatbot"') \
                    .classes("flex-1").mark("composer")
                refs["input"] = txt
                send_btn = ui.button("Send", on_click=_on_send_click, icon="arrow_upward")                     .props('unelevated dense no-caps aria-label="Send message"').classes("send-btn").mark("send")
                refs["send"] = send_btn
            with ui.element("div").classes("composer-foot"):
                refs["help"] = ui.html("").classes("composer-help").props('role="status" aria-live="polite"')
                ui.html('<div class="composer-hint"><kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line</div>')
            refs["live"] = ui.html("").classes("sr-only").props('role="status" aria-live="polite"')

        # Enter sends; Shift+Enter keeps the textarea's newline. Decided in the browser (D-09).
        txt.on("keydown", _on_keydown_send, js_handler=ENTER_SENDS_JS)

    if _KB_STATE["kb"] is None and _KB_STATE["error"] is None:
        refs["status_timer"] = ui.timer(2.0, _refresh_status)

    # =======================================================================
    # Initial render
    # =======================================================================
    _render_sessions()
    _render_chat()


def _call_pipeline(kb, question: str):
    """One turn: the answer, the diagnostics the seam recorded for it and the end-to-end wall time of the call."""
    from .pipeline import ask_with_diagnostics
    started = time.perf_counter()
    result, diagnostics = ask_with_diagnostics(kb, question)
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


def register_pages() -> None:
    """Register the chat page on the NiceGUI app (called by ``start_app`` and by the page tests)."""
    @ui.page("/")
    def index(request: Request) -> None:
        _build_page(developer=developer_requested(request))


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
