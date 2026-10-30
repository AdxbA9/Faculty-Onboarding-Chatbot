"""
NiceGUI interface for the UOS Faculty Onboarding Chatbot.

Layout:
    +--------------+----------------------------------------+
    |  Sidebar     |  Topbar (status pill)                  |
    |  - branding  |  ------------------------------------  |
    |  - new chat  |  Chat scroll area                      |
    |  - history   |                                        |
    |  - footer    |                                        |
    |    (theme,   |                                        |
    |     dev,     |  Sticky composer -------------------   |
    |     status)  |  Enter to send  /  Shift+Enter newline |
    +--------------+----------------------------------------+

Design notes:
- Every nested helper inside ``_build_page`` is defined BEFORE any UI
  construction runs. Python treats any name assigned in a function as a
  local, so defining a def AFTER a call site causes UnboundLocalError.
- The page ``client`` is captured at build time so event handlers that
  outlive their originating DOM element (e.g. an example card that gets
  removed when the chat re-renders) can still call ``client.run_javascript``
  without hitting "slot has been deleted" errors.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from nicegui import app, run, ui

from handbook_bot import QAResult
from handbook_bot.config import ENABLE_OCR, GROQ_MODEL

from . import presentation
from .pipeline import groq_is_configured, load_knowledge_base, preflight
from .styles import GLOBAL_CSS, ICONS

_log = logging.getLogger(__name__)


APP_TITLE = "UOS Faculty Onboarding Chatbot"
APP_TAGLINE = "Handbook Assistant"
APP_SUBTITLE = (
    "RAG-based assistant for University of Sharjah faculty onboarding "
    "and handbook guidance."
)

SAMPLE_PROMPTS = [
    ("Mission",  "What is the University of Sharjah's mission?"),
    ("Contact",  "What is the phone number for the Information Technology Center?"),
    ("Programs", "How many degree programs does UoS offer in Fall 2025/2026?"),
    ("Handbook", "What is the purpose of the Faculty Handbook?"),
]


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_KB_STATE: Dict[str, Any] = {
    "kb": None,
    "error": None,
    "pdf_name": "",
    "stats": {},
}
_CLIENTS: Dict[str, Dict[str, Any]] = {}

_LOGO_URL = "/static/logo.png"


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
        # The full traceback goes to the server log; the page shows only the failure category.
        _log.exception("knowledge base failed to load")
        _KB_STATE["error"] = (
            "Could not load the handbook (%s). Check the server log, then reload the app."
            % type(exc).__name__
        )


# ---------------------------------------------------------------------------
# Per-client state helpers
# ---------------------------------------------------------------------------
def _new_session() -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "title": "New chat",
        "messages": [],
        "created_at": time.time(),
    }


def _client_state(client_id: str) -> Dict[str, Any]:
    state = _CLIENTS.get(client_id)
    if state is None:
        s = _new_session()
        state = {
            "sessions": [s],
            "active_id": s["id"],
            "debug": False,
            "theme": "dark",
        }
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
    """Run a UI-context-sensitive call and swallow context errors.

    NiceGUI raises ``RuntimeError("The parent element this slot belongs to
    has been deleted.")`` when an event handler tries to touch the DOM
    after its originating element was removed (common with
    clear()+re-render patterns). We never want those errors to crash the
    user's interaction, so we just ignore them and move on.
    """
    try:
        callable_(*args, **kwargs)
    except RuntimeError:
        pass
    except Exception:
        pass


# ===========================================================================
# Page
# ===========================================================================
def _build_page() -> None:
    client_id = ui.context.client.id
    page_client = ui.context.client  # captured for use after re-renders
    state = _client_state(client_id)

    ui.add_head_html(GLOBAL_CSS)
    ui.query("html").props(f'data-theme={state["theme"]}')

    session_list_container: Dict[str, Any] = {}
    chat_container: Dict[str, Any] = {}
    composer_input: Dict[str, Any] = {}
    send_button_ref: Dict[str, Any] = {}
    is_busy: Dict[str, bool] = {"v": False}

    # =======================================================================
    # HELPERS - defined up-front so nothing hits an UnboundLocalError
    # during the sidebar/main construction phase.
    # =======================================================================
    def _on_theme_toggle() -> None:
        state["theme"] = "light" if state["theme"] == "dark" else "dark"
        _safe(ui.query("html").props, f'data-theme={state["theme"]}')

    def _on_debug_toggle(value: bool) -> None:
        state["debug"] = bool(value)
        _render_chat()

    def _render_status_block() -> None:
        ok = groq_is_configured() and _KB_STATE["kb"] is not None
        warn = (not groq_is_configured()) or (_KB_STATE["error"] is not None)
        dot = "ok" if ok else ("warn" if warn else "")
        status_label = (
            "Ready"
            if ok else
            ("API key missing" if not groq_is_configured() else "Loading")
        )
        ocr_label = "on" if ENABLE_OCR else "off"
        pdf_label = _KB_STATE["pdf_name"] or "-"

        ui.html(
            f'<div class="status-block">'
            f'  <div class="status-line">'
            f'    <span class="status-dot {dot}"></span>'
            f'    <span class="label">Status</span>'
            f'    <span class="value">{html.escape(status_label)}</span>'
            f'  </div>'
            f'  <div class="status-line">'
            f'    <span class="label">Model</span>'
            f'    <span class="value">{html.escape(GROQ_MODEL)}</span>'
            f'  </div>'
            f'  <div class="status-line">'
            f'    <span class="label">PDF</span>'
            f'    <span class="value">{html.escape(pdf_label)}</span>'
            f'  </div>'
            f'  <div class="status-line">'
            f'    <span class="label">OCR</span>'
            f'    <span class="value">{html.escape(ocr_label)}</span>'
            f'  </div>'
            f'</div>'
        )

    async def _on_new_chat() -> None:
        s = _new_session()
        state["sessions"].insert(0, s)
        state["active_id"] = s["id"]
        _render_sessions()
        _render_chat()

    def _render_sessions() -> None:
        container = session_list_container["el"]
        container.clear()
        with container:
            for sess in state["sessions"]:
                active = sess["id"] == state["active_id"]
                cls = "session-item active" if active else "session-item"
                title_preview = sess["title"] or "New chat"

                def _make_click(sid: str):
                    async def _click() -> None:
                        state["active_id"] = sid
                        _render_sessions()
                        _render_chat()
                    return _click

                ui.html(f'<div>{html.escape(title_preview)}</div>') \
                    .classes(cls) \
                    .on("click", _make_click(sess["id"]))

    def _render_landing(container) -> None:
        with container:
            with ui.element("div").classes("landing"):
                # Accent the word "Onboarding" in UOS green for brand pop
                ui.html(
                    f'<div class="landing-logo"><img src="{_LOGO_URL}" alt="UOS"/></div>'
                    f'<div class="landing-title">'
                    f'  UOS Faculty <span class="accent">Onboarding</span> Chatbot'
                    f'</div>'
                    f'<div class="landing-subtitle">{html.escape(APP_SUBTITLE)}</div>'
                )

                if _KB_STATE["error"]:
                    ui.html(
                        f'<div class="banner danger">'
                        f'<strong>Setup required.</strong> '
                        f'{html.escape(_KB_STATE["error"])}'
                        f'</div>'
                    )
                elif _KB_STATE["kb"] is None:
                    ui.html(
                        '<div class="banner">'
                        '<strong>Warming up.</strong> '
                        'Indexing the handbook and loading models. '
                        'First launch can take a minute.'
                        '</div>'
                    )
                elif not groq_is_configured():
                    ui.html(
                        '<div class="banner">'
                        '<strong>Groq API key missing.</strong> '
                        'Contact, date, and count lookups still work; '
                        'open-ended questions need a key in <code>.env</code>.'
                        '</div>'
                    )

                with ui.element("div").classes("example-grid"):
                    for label, prompt in SAMPLE_PROMPTS:
                        p = prompt
                        lb = label

                        def _make_click(q: str):
                            async def _click() -> None:
                                # Defer the submit one tick so that the
                                # click event handler can return cleanly
                                # BEFORE we clear the landing (which deletes
                                # the element this handler fired from).
                                asyncio.create_task(_submit(q))
                            return _click

                        ui.html(
                            f'<div>'
                            f'  <div class="example-label">{html.escape(lb)}</div>'
                            f'  <div>{html.escape(p)}</div>'
                            f'</div>'
                        ).classes("example-card").on("click", _make_click(p))

    def _render_sources(result: QAResult, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        chips_html: List[str] = []
        domains = presentation.domain_labels(diagnostics)
        if domains:                                       # the verified specialist path served this answer
            chips_html.append(
                '<span class="chip verified" title="Verified from the handbook by the specialist path">'
                + html.escape("Verified \u00b7 " + ", ".join(domains)) + '</span>'
            )
        for css, label in presentation.source_chips(result):
            chips_html.append(f'<span class="chip {css}">{html.escape(label)}</span>')
        if not chips_html:
            return
        ui.html('<div class="source-row">' + "".join(chips_html) + '</div>')

    def _render_toolbar(text: str) -> None:
        js_literal = json.dumps(text)
        copy_js = (
            f"(function(btn){{"
            f"  navigator.clipboard.writeText({js_literal}).then(function(){{"
            f"    var lbl = btn.querySelector('.copy-lbl');"
            f"    if(!lbl) return;"
            f"    var orig = lbl.innerText;"
            f"    lbl.innerText = 'Copied';"
            f"    setTimeout(function(){{ lbl.innerText = orig; }}, 1200);"
            f"  }});"
            f"}})(this);"
        )
        ui.html(
            f'<div class="msg-toolbar">'
            f'  <button class="icon-btn" onclick=\'{copy_js}\'>'
            f'    {ICONS["copy"]}<span class="copy-lbl">Copy</span>'
            f'  </button>'
            f'</div>'
        )

    def _render_debug(msg: Dict[str, Any]) -> None:
        result: QAResult = msg["result"]
        sections = presentation.dev_sections(result, msg.get("diagnostics"), _KB_STATE.get("stats") or {}, msg.get("wall_ms"))

        def _rows(rs) -> str:
            return "".join(
                f'<div class="dev-row"><span class="k">{html.escape(k)}</span>'
                f'<span class="v" title="{html.escape(v)}">{html.escape(v)}</span></div>'
                for k, v in rs
            )

        body = "".join(
            f'<div class="dev-section">{html.escape(title)}</div>' + _rows(rows)
            for title, rows in sections if rows
        )
        ui.html(
            '<div class="dev-panel">'
            '  <details open>'
            '    <summary>Developer details</summary>'
            f'    <div class="dev-body">{body}</div>'
            '  </details>'
            '</div>'
        )

    def _render_message(msg: Dict[str, Any]) -> None:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with ui.element("div").classes("msg-row user"):
                ui.html(
                    f'<div class="bubble user">{html.escape(content)}</div>'
                )
                ui.html('<div class="avatar user">YOU</div>')
            return

        result: Optional[QAResult] = msg.get("result")
        with ui.element("div").classes("msg-row assistant"):
            ui.html(f'<div class="avatar assistant"><img src="{_LOGO_URL}" alt="UOS"/></div>')
            with ui.element("div").classes("assistant-wrap"):
                with ui.element("div").classes("bubble assistant"):
                    if content == "__typing__":
                        ui.html(
                            '<div class="typing" role="status" aria-live="polite">'
                            '<span></span><span></span><span></span>'
                            f'<em class="typing-text">{html.escape(presentation.SEARCHING_TEXT)}</em>'
                            '</div>'
                        )
                    else:
                        ui.markdown(presentation.escape_identifiers(content))

                if result is not None and content != "__typing__":
                    _render_sources(result, msg.get("diagnostics"))
                    _render_toolbar(content)
                    if state["debug"]:
                        _render_debug(msg)

    def _render_chat() -> None:
        container = chat_container.get("el")
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
        """Nudge the chat scroll to the bottom using the captured client.

        Using the page-build-time client reference (not ui.context.client)
        is important: when _render_chat is triggered from a click handler
        whose element has since been removed, ui.context.client raises
        'slot has been deleted'. Going through the captured client avoids
        that context lookup entirely.
        """
        scroll_el = chat_container.get("scroll")
        if scroll_el is None:
            return
        try:
            page_client.run_javascript(
                f"setTimeout(() => {{"
                f"  const e = getElement({scroll_el.id});"
                f"  if (e) e.scrollTop = e.scrollHeight;"
                f"}}, 30);"
            )
        except Exception:
            # Client gone or context torn down - safe to ignore.
            pass

    async def _submit(question: str) -> None:
        question = (question or "").strip()
        if not question or is_busy["v"]:
            return
        if _KB_STATE["kb"] is None:
            _safe(
                ui.notify,
                _KB_STATE["error"] or "The knowledge base is still loading.",
                type="warning",
            )
            return

        sess = _active_session(state)
        if not sess["messages"]:
            sess["title"] = (question[:38] + "...") if len(question) > 40 else question
            _render_sessions()

        sess["messages"].append({"role": "user", "content": question, "result": None})
        sess["messages"].append({
            "role": "assistant",
            "content": "__typing__",
            "result": None,
        })
        try:
            composer_input["el"].value = ""
        except Exception:
            pass
        is_busy["v"] = True
        if "el" in send_button_ref:
            _safe(send_button_ref["el"].props, "disable")
        _render_chat()

        try:
            result, diagnostics, wall_ms = await run.io_bound(
                _call_pipeline, _KB_STATE["kb"], question
            )
        except Exception as exc:  # pragma: no cover
            result = QAResult(answer=presentation.ERROR_TEXT)
            diagnostics, wall_ms = presentation.error_diagnostics(exc), None

        sess["messages"][-1] = presentation.assistant_message(result, diagnostics, wall_ms)
        is_busy["v"] = False
        if "el" in send_button_ref:
            _safe(send_button_ref["el"].props, remove="disable")
        _render_chat()

    def _on_keydown_send(_e) -> None:
        val = (composer_input["el"].value or "").strip()
        if val and not is_busy["v"]:
            try:
                composer_input["el"].value = ""
            except Exception:
                pass
            asyncio.create_task(_submit(val))

    async def _on_send_click() -> None:
        val = composer_input["el"].value or ""
        await _submit(val)

    # =======================================================================
    # ROOT SHELL
    # =======================================================================
    with ui.element("div").classes("app-shell"):
        sidebar = ui.element("div").classes("sidebar")
        main = ui.element("div").classes("main")

    # ---------- Sidebar ----------------------------------------------------
    with sidebar:
        ui.html(
            f'<div class="brand">'
            f'  <div class="brand-logo"><img src="{_LOGO_URL}" alt="UOS"/></div>'
            f'  <div class="brand-text">'
            f'    <div class="brand-title">{html.escape(APP_TITLE)}</div>'
            f'    <div class="brand-subtitle">{html.escape(APP_TAGLINE)}</div>'
            f'  </div>'
            f'</div>'
        )

        new_btn = ui.button(on_click=_on_new_chat) \
            .props("flat no-caps") \
            .classes("new-chat-btn")
        with new_btn:
            ui.html(f'<span class="plus">{ICONS["plus"]}</span>')
            ui.html('<span>New chat</span>')

        ui.html('<div class="sb-label">Recent</div>')
        session_list_container["el"] = ui.element("div").classes("w-full flex flex-col gap-1")

        # Sidebar footer now hosts only the status block (theme toggle and
        # developer-mode switch have been promoted to the topbar for better
        # reach and clearer grouping).
        with ui.element("div").classes("sb-footer"):
            _render_status_block()

    # ---------- Main -------------------------------------------------------
    with main:
        # Topbar: title on the left, theme + dev mode + model pill on the right
        with ui.element("div").classes("topbar"):
            ui.html(
                f'<div class="topbar-left">'
                f'  <span class="topbar-accent"></span>'
                f'  <span class="topbar-title">{html.escape(APP_TITLE)}</span>'
                f'</div>'
            )
            with ui.element("div").classes("topbar-right"):
                # Developer-mode pill (labeled toggle)
                dev_pill_cls = "dev-toggle-pill active" if state["debug"] else "dev-toggle-pill"
                with ui.element("div").classes(dev_pill_cls) as dev_pill:
                    ui.html('<span class="dev-label">Developer mode</span>')

                    def _dev_changed(e, pill=dev_pill):
                        _on_debug_toggle(e.value)
                        # Visually sync the pill's active state
                        if e.value:
                            pill.classes(add="active")
                        else:
                            pill.classes(remove="active")

                    ui.switch(
                        value=state["debug"],
                        on_change=_dev_changed,
                    ).props('dense aria-label="Developer mode"')

                # Theme toggle button (sun/moon icon)
                theme_btn = ui.button(on_click=_on_theme_toggle) \
                    .props('flat dense aria-label="Toggle light or dark mode"') \
                    .classes("topbar-btn")
                theme_btn.tooltip("Toggle light / dark mode")
                with theme_btn:
                    ui.html(ICONS["moon"] + ICONS["sun"])

                # Model status pill
                ui.html(
                    f'<div class="topbar-pill">'
                    f'  <span class="status-dot {"ok" if _KB_STATE["kb"] else "warn"}"></span>'
                    f'  <span>{html.escape(GROQ_MODEL)}</span>'
                    f'</div>'
                )

        chat_scroll = ui.element("div").classes("chat-scroll")
        with chat_scroll:
            chat_inner = ui.element("div").classes("chat-inner")
            chat_container["el"] = chat_inner
            chat_container["scroll"] = chat_scroll

        with ui.element("div").classes("composer-wrap"):
            with ui.element("div").classes("composer"):
                txt = ui.textarea(placeholder="Message the Onboarding Chatbot...") \
                    .props('autogrow outlined=false borderless dense rounded aria-label="Message the Onboarding Chatbot"') \
                    .classes("flex-1")
                composer_input["el"] = txt

                send_btn = ui.button(on_click=_on_send_click) \
                    .props('flat dense no-caps aria-label="Send message"') \
                    .classes("send-btn")
                with send_btn:
                    ui.html(ICONS["send"])
                send_button_ref["el"] = send_btn

            ui.html(
                '<div class="composer-hint">'
                '<kbd>Enter</kbd> to send &nbsp;/&nbsp; '
                '<kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line'
                '</div>'
            )

        # Enter-to-send: `.exact` means only plain Enter fires - Shift+Enter
        # falls through to the textarea's default newline.
        txt.on("keydown.enter.exact.prevent", _on_keydown_send)

    # =======================================================================
    # INITIAL RENDER
    # =======================================================================
    _render_sessions()
    _render_chat()


def _call_pipeline(kb, question: str):
    """One turn: the answer, the diagnostics the seam recorded for it and the end-to-end wall time of the call."""
    from .pipeline import ask_with_diagnostics
    started = time.perf_counter()
    result, diagnostics = ask_with_diagnostics(kb, question)
    return result, diagnostics, (time.perf_counter() - started) * 1000.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def start_app(port: int = 8501) -> None:
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    if assets_dir.is_dir():
        app.add_static_files("/static", str(assets_dir))

    app.on_startup(_ensure_kb_loaded)

    @ui.page("/")
    def index() -> None:
        _build_page()

    ui.run(
        host="0.0.0.0",
        port=port,
        title=APP_TITLE,
        dark=None,
        reload=False,
        show=False,
        favicon=(str(assets_dir / "logo.png") if assets_dir.is_dir() else None),
        storage_secret=secrets.token_hex(16),
    )
