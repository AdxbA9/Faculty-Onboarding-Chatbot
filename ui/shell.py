"""
The shared website shell (post-freeze PF-15, decision record D-126).

Every page, the chat included, is framed the same way:

    .app-shell
      aside.sidebar#app-nav     brand, New chat, primary navigation (seven real links in two groups, the current page
                                marked with ``aria-current="page"``), page-specific content (the chat lists recent
                                conversations), the guest chip linking to the profile, "About this assistant"
      .nav-backdrop             below 900 px the sidebar becomes a drawer over the page; the backdrop closes it
      main.main
        header.topbar           menu button (below 900 px), title, page controls (language, theme, ...)
        content                 the chat log and composer, or ``.page-scroll > .page-main`` for every other page

Navigation items are plain anchors, so the browser's back button, new tabs and direct URLs behave as on any website.
The drawer is opened and closed in the browser (``fobNav``): ``aria-expanded`` is kept in sync, focus moves into the
drawer, Escape and the backdrop close it and focus returns to the menu button.

``Page`` is the frame of the non-chat pages (onboarding, procedures, resources, history, profile, settings): it resolves
the PF-13 identity through ``chat.page_context`` (a visit creates no owner row; the first write does), applies the
saved preferences to ``<html>`` (theme, language and direction, text size, reduced motion) and offers the small helpers
the pages share (notices, preference saving, confirmation dialogs).
"""
from __future__ import annotations

import html
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlencode

from fastapi import Request
from nicegui import run, ui

from . import i18n
from .i18n import t
from .styles import GLOBAL_CSS, ICONS

LOGO_URL = "/static/logo.png"

#: (key, href, label key, icon) in display order, in two groups.
NAV_MAIN: Tuple[Tuple[str, str, str, str], ...] = (
    ("chat", "/", "nav_chat", "chat"),
    ("onboarding", "/onboarding", "nav_onboarding", "list_checks"),
    ("procedures", "/procedures", "nav_procedures", "steps"),
    ("resources", "/resources", "nav_resources", "library"),
    ("history", "/history", "nav_history", "history"),
)
NAV_YOU: Tuple[Tuple[str, str, str, str], ...] = (
    ("profile", "/profile", "nav_profile", "user"),
    ("settings", "/settings", "nav_settings", "sliders"),
)
NAV_KEYS = tuple(item[0] for item in NAV_MAIN + NAV_YOU)

#: The browser half of the shell: the drawer and local-time rendering of ``<time data-local>`` elements.
SHELL_JS = r"""
<script>
(function () {
  if (window.fobNav) return;
  window.fobNav = function (open) {
    var shell = document.querySelector('.app-shell');
    if (!shell) return;
    var want = (open === undefined) ? !shell.classList.contains('nav-open') : !!open;
    shell.classList.toggle('nav-open', want);
    document.querySelectorAll('.nav-toggle').forEach(function (b) { b.setAttribute('aria-expanded', want ? 'true' : 'false'); });
    if (want) {
      setTimeout(function () { var a = document.querySelector('#app-nav .nav-item.active') || document.querySelector('#app-nav .nav-item'); if (a) a.focus(); }, 40);
    }
  };
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    var shell = document.querySelector('.app-shell.nav-open');
    if (!shell) return;
    window.fobNav(false);
    var b = document.querySelector('.topbar .nav-toggle');
    if (b) b.focus();
  });
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  window.fobTimes = function () {
    var lang = (document.documentElement.getAttribute('lang') || 'en') === 'ar' ? 'ar-AE-u-nu-latn' : 'en-GB';
    document.querySelectorAll('time[data-local]:not([data-done])').forEach(function (el) {
      var d = new Date(el.getAttribute('datetime'));
      if (isNaN(d.getTime())) return;
      var now = new Date();
      var sameDay = d.toDateString() === now.toDateString() && el.getAttribute('data-format') !== 'full';
      var opts = sameDay ? {hour: '2-digit', minute: '2-digit'} : {day: 'numeric', month: 'short', year: (d.getFullYear() === now.getFullYear() ? undefined : 'numeric'), hour: '2-digit', minute: '2-digit'};
      try { el.textContent = new Intl.DateTimeFormat(lang, opts).format(d); } catch (err) { el.textContent = pad(d.getHours()) + ':' + pad(d.getMinutes()); }
      try { el.title = new Intl.DateTimeFormat(lang, {dateStyle: 'full', timeStyle: 'short'}).format(d); } catch (err) {}
      el.setAttribute('data-done', '1');
    });
  };
  // PF-15: a UAE phone number inside an answer is shown whole and left-to-right, also inside an Arabic sentence. Display
  // only: the server HTML and the text are unchanged; code is left alone; an element already wrapped is skipped.
  var TEL = /\+971(?:[     \-]?\d){8,9}(?!\d)/g;   // PF-16: model answers separate digits with no-break spaces
  window.fobTel = function () {
    document.querySelectorAll('.answer, .original-body').forEach(function (box) {
      var walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT, {acceptNode: function (n) {
        if (!n.nodeValue || n.nodeValue.indexOf('+971') < 0) return NodeFilter.FILTER_REJECT;
        var p = n.parentElement;
        return (p && p.closest('code, pre, .tel')) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      }});
      var nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach(function (n) {
        var text = n.nodeValue, last = 0, frag = document.createDocumentFragment(), m;
        TEL.lastIndex = 0;
        while ((m = TEL.exec(text))) {
          if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
          var s = document.createElement('span');
          s.className = 'tel'; s.dir = 'ltr'; s.textContent = m[0];
          frag.appendChild(s);
          last = m.index + m[0].length;
        }
        if (!last) return;
        if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
        n.parentNode.replaceChild(frag, n);
      });
    });
  };
  new MutationObserver(function () { window.fobTimes(); window.fobTel(); }).observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener('DOMContentLoaded', function () { window.fobTimes(); window.fobTel(); });
})();
</script>
"""


def esc(text: Any, quote_: bool = False) -> str:
    return html.escape(str(text if text is not None else ""), quote=quote_)


def time_html(iso: Any, css: str = "msg-time", *, full: bool = False) -> str:
    """A ``<time>`` the browser renders in the reader's local time (``full``: always with the date); the server text (UTC)
    is the fallback."""
    text = str(iso or "")
    if len(text) < 16 or "T" not in text:
        return ""
    fallback = text[:16].replace("T", " ") + " UTC"
    return '<time class="%s" data-local%s datetime="%s">%s</time>' % (css, ' data-format="full"' if full else "", esc(text, True), esc(fallback))


def html_props(theme: str, lang: str, text_size: Optional[str], motion: Optional[str]) -> str:
    """The attributes of ``<html>`` that carry the preferences (CSS reads them)."""
    return 'data-theme=%s lang=%s dir=%s data-text=%s data-motion=%s' % (
        "dark" if theme == "dark" else "light", lang, i18n.DIRECTION.get(lang, "ltr"),
        "large" if text_size == "large" else "standard", "reduced" if motion == "on" else "full")


def initial_of(name: Optional[str]) -> str:
    text = (name or "").strip()
    return text[:1].upper() if text else "G"


def nav_html(lang: str, active: str, counts: Optional[Dict[str, int]] = None) -> str:
    """The primary navigation: two labelled groups of links, the current page marked."""
    counts = counts or {}

    def item(key: str, href: str, label: str, icon: str) -> str:
        current = key == active
        count = counts.get(key)
        return '<a class="nav-item%s" href="%s"%s>%s<span>%s</span>%s</a>' % (
            " active" if current else "", href, ' aria-current="page"' if current else "", ICONS[icon], esc(t(lang, label)),
            ('<span class="nav-count" aria-hidden="true">%d</span>' % count) if isinstance(count, int) and count > 0 else "")
    main = "".join(item(*entry) for entry in NAV_MAIN)
    you = "".join(item(*entry) for entry in NAV_YOU)
    return ('<nav class="primary-nav" aria-label="%s"><div class="nav-block"><div class="nav-section">%s</div>%s</div>'
            '<div class="nav-block"><div class="nav-section">%s</div>%s</div></nav>') % (
        esc(t(lang, "nav_aria"), True), esc(t(lang, "nav_section_main")), main, esc(t(lang, "nav_section_you")), you)


def brand_html(lang: str) -> str:
    return ('<a class="brand" href="/" style="text-decoration:none"><div class="brand-logo"><img src="%s" alt="%s"/></div>'
            '<div class="brand-text"><div class="brand-org">%s</div><div class="brand-name">%s</div></div></a>') % (
        LOGO_URL, esc(t(lang, "logo_alt"), True), esc(t(lang, "org")), esc(t(lang, "app_name")))


def drawer_close_html(lang: str) -> str:
    return '<button type="button" class="drawer-close" aria-label="%s" onclick="fobNav(false)">%s</button>' % (esc(t(lang, "nav_close"), True), ICONS["close"])


def nav_toggle_html(lang: str) -> str:
    return '<button type="button" class="nav-toggle" aria-controls="app-nav" aria-expanded="false" aria-label="%s" onclick="fobNav()">%s</button>' % (
        esc(t(lang, "nav_open"), True), ICONS["menu"])


def guest_chip_html(lang: str, display_name: Optional[str]) -> str:
    name = display_name or t(lang, "guest_label")
    return ('<a class="guest-chip" href="/profile" aria-label="%s"><span class="guest-avatar" aria-hidden="true">%s</span>'
            '<span class="guest-text"><span class="guest-name" dir="auto">%s</span><span class="guest-sub">%s</span></span></a>') % (
        esc(t(lang, "guest_chip_aria", name=name), True), esc(initial_of(display_name)), esc(name), esc(t(lang, "guest_badge")))


def about_html(lang: str) -> str:
    return '<details class="about"><summary>%s</summary><div>%s</div></details>' % (esc(t(lang, "about_lead").rstrip(".")), esc(t(lang, "about")))


def same_page_with(request: Request, **params: str) -> str:
    query = {k: v for k, v in request.query_params.items() if k not in params}
    query.update({k: v for k, v in params.items() if v is not None})
    return request.url.path + ("?" + urlencode(query, quote_via=quote) if query else "")


# =====================================================================================================================
# The frame of every page other than the chat
# =====================================================================================================================
class Page:
    """One page build: identity context, the shared shell and the helpers the pages share."""

    def __init__(self, request: Request, active: str) -> None:
        from . import chat                                       # imported here: chat imports this module
        self._chat = chat
        self.ctx = chat.page_context(request)
        self.lang = self.ctx["lang"]
        self.active = active
        self.refs: Dict[str, Any] = {}

    # ---- identity and preferences --------------------------------------------------------------------------------------
    def L(self) -> str:
        return self.lang

    def store(self):
        return self.ctx["store"]

    def owner_id(self) -> Optional[int]:
        return self.ctx.get("owner_id")

    def pref(self, key: str, default: Any = None) -> Any:
        return (self.ctx.get("prefs") or {}).get(key, default)

    def ensure_owner(self) -> Optional[int]:
        """The owner id, creating the row on this first write (never on a visit)."""
        if self.ctx.get("owner_id") is not None:
            return self.ctx["owner_id"]
        store, token = self.ctx.get("store"), self.ctx.get("owner_token")
        if store is None or not token:
            return None
        try:
            self.ctx["owner_id"] = store.owner_for_token(token)
        except Exception as exc:
            self._chat._log.error("creating the owner failed: %s", type(exc).__name__)
            return None
        return self.ctx["owner_id"]

    async def save_pref(self, **changes: Any) -> bool:
        """Store preference changes for this guest; False (and a logged error class) when they could not be stored."""
        owner = self.ensure_owner()
        store = self.store()
        if owner is None or store is None:
            return False
        try:
            saved = await run.io_bound(lambda: store.set_preferences(owner, **changes))
        except Exception as exc:
            self._chat._log.error("saving preferences failed: %s", type(exc).__name__)
            return False
        self.ctx["prefs"] = saved
        return True

    def apply_html(self) -> None:
        self._chat._safe(ui.query("html").props, html_props(self.ctx["theme"], self.lang, self.pref("text_size"), self.pref("reduced_motion")))

    # ---- notices and dialogs ----------------------------------------------------------------------------------------------
    def notify(self, text: str, kind: str = "info") -> None:
        el = self.refs.get("live")
        if el is not None:
            icon = ICONS["check"] if kind == "ok" else (ICONS["alert"] if kind == "warn" else ICONS["info"])
            self._chat._safe(setattr, el, "content", '<div class="page-note %s" role="status">%s<span>%s</span></div>' % (kind, icon, esc(text)))

    def dialog(self, css: str):
        host = self.refs.get("dialogs")
        with host:
            dialog = ui.dialog()
            with dialog:
                card = ui.card().classes("fob-dialog " + css).props('dir=%s lang=%s' % (i18n.DIRECTION[self.lang], self.lang))
        dialog.on("hide", lambda _e: self._chat._safe(dialog.delete))
        return dialog, card

    def close_dialog(self, dialog) -> None:
        self._chat._safe(dialog.close)
        self._chat._safe(dialog.delete)

    def confirm(self, title: str, text: str, ok_label: str, on_ok: Callable[[], Any], *, ok_marker: str, danger: bool = True) -> None:
        dialog, card = self.dialog("confirm-dialog")
        lang = self.lang
        with card:
            ui.html('<h2 class="dialog-title">%s</h2><p class="dialog-text" dir="auto">%s</p>' % (esc(title), esc(text)))
            with ui.element("div").classes("dialog-actions"):
                ui.button(t(lang, "cancel"), on_click=lambda: self.close_dialog(dialog)).props("flat no-caps autofocus").classes("btn-secondary").mark("dialog-cancel")

                async def _ok() -> None:
                    self.close_dialog(dialog)
                    result = on_ok()
                    if hasattr(result, "__await__"):
                        await result
                ui.button(ok_label, on_click=_ok).props("unelevated no-caps" + (" color=negative" if danger else "")).mark(ok_marker)
        dialog.open()

    # ---- the frame ----------------------------------------------------------------------------------------------------------
    def shell(self, page_title: str, *, crumb: Optional[str] = None, nav_counts: Optional[Dict[str, int]] = None):
        """Build the shared shell and return the content container of this page."""
        lang = self.lang
        ui.add_head_html(GLOBAL_CSS)
        ui.add_head_html(SHELL_JS)
        self.apply_html()
        root = ui.element("div").classes("app-root ob-root")
        with root:
            ui.html('<a class="skip-link" href="#page-main">%s</a>' % esc(t(lang, "skip_to_content")))
            with ui.element("div").classes("app-shell").props('id="app-shell"'):
                sidebar = ui.element("aside").classes("sidebar").props('id="app-nav" aria-label="%s"' % esc(t(lang, "nav_aria"), True))
                ui.html('<div class="nav-backdrop" onclick="fobNav(false)"></div>')
                main = ui.element("main").classes("main")
            with sidebar:
                ui.html(drawer_close_html(lang))
                ui.html(brand_html(lang))
                ui.html('<a class="new-chat-link" href="/?new=1">%s<span>%s</span></a>' % (ICONS["plus"], esc(t(lang, "new_chat"))))
                ui.html(nav_html(lang, self.active, nav_counts))
                with ui.element("div").classes("sb-footer"):
                    self.refs["chip"] = ui.html(guest_chip_html(lang, self.pref("display_name")))
                    ui.html(about_html(lang))
            with main:
                with ui.element("header").classes("topbar page-topbar"):
                    ui.html('<div class="topbar-left">%s<div class="topbar-brand" aria-hidden="true"><img src="%s" alt=""/></div>'
                            '<div><div class="topbar-title">%s</div><div class="topbar-crumb">%s</div></div></div>' % (
                                nav_toggle_html(lang), LOGO_URL, esc(page_title), esc(crumb or t(lang, "app_name")))).classes("tl")
                    with ui.element("div").classes("topbar-right"):
                        ui.button(t(lang, "lang_switch"), on_click=self._toggle_lang).props('flat dense no-caps aria-label="%s" lang="%s"' % (
                            esc(t(lang, "lang_switch_aria"), True), "ar" if lang == "en" else "en")).classes("topbar-btn topbar-lang").mark("lang")
                        theme_btn = ui.button(on_click=self._toggle_theme).props('flat dense aria-label="%s"' % esc(
                            t(lang, "theme_to_light" if self.ctx["theme"] == "dark" else "theme_to_dark"), True)).classes("topbar-btn theme-btn").mark("theme")
                        self.refs["theme"] = theme_btn
                        with theme_btn:
                            ui.html(ICONS["moon"] + ICONS["sun"])
                with ui.element("div").classes("page-scroll"):
                    content = ui.element("div").classes("page-main ob-main").props('id="page-main" tabindex="-1"')
                    with content:
                        self.refs["live"] = ui.html("").classes("ob-live").props('aria-live="polite"')
                        if self.ctx.get("storage_notice"):
                            ui.html('<div class="banner warn" role="status">%s<div><strong>%s</strong> %s</div></div>' % (
                                ICONS["info"], esc(t(lang, "storage_unavailable")), esc(t(lang, "ob_storage_unavailable"))))
            self.refs["dialogs"] = ui.element("div").classes("dialog-host")
        return content

    async def _toggle_lang(self) -> None:
        new = "ar" if self.lang == "en" else "en"
        await self.save_pref(interface_lang=new)
        ui.navigate.to(same_page_with(self.ctx["request"], lang=new))

    async def _toggle_theme(self) -> None:
        new = "light" if self.ctx["theme"] == "dark" else "dark"
        self.ctx["theme"] = new
        self._chat._safe(ui.query("html").props, "data-theme=%s" % new)
        btn = self.refs.get("theme")
        if btn is not None:
            self._chat._safe(btn.props, 'aria-label="%s"' % t(self.lang, "theme_to_light" if new == "dark" else "theme_to_dark"))
        await self.save_pref(theme=new)


__all__ = ["LOGO_URL", "NAV_KEYS", "NAV_MAIN", "NAV_YOU", "Page", "SHELL_JS", "about_html", "brand_html", "drawer_close_html", "esc", "guest_chip_html",
           "html_props", "initial_of", "nav_html", "nav_toggle_html", "same_page_with", "time_html"]
