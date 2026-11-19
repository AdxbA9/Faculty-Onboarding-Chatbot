"""
Safe HTML for the chat page (post-freeze PF-10).

The answer text comes from the model or from an extractor. NiceGUI's Markdown element passes raw HTML through
(``<script>`` survives ``prepare_content``), so an answer is rendered here instead: every HTML special character is
escaped FIRST, then Markdown is applied, then every link is checked. Only ``http``, ``https`` and ``mailto`` links are
kept, and they open in a new tab with ``rel="noopener noreferrer"``; any other scheme becomes plain text. Block elements carry ``dir="auto"`` so mixed Arabic/English answers keep each block's own direction (PF-11).
Nothing else of the answer is changed, so lists, numbered steps, tables and code blocks still render.
"""
from __future__ import annotations

import html
import json
import re

from nicegui.elements.markdown import prepare_content

_EXTRAS = "fenced-code-blocks tables"
_ANCHOR = re.compile(r"<a\s+([^>]*?)>", re.I)
_HREF = re.compile(r"""href\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_SAFE_SCHEME = re.compile(r"^(?:https?://|mailto:)", re.I)
_TAG = re.compile(r"<[^>]+>")
_BLOCKQUOTE_MARK = re.compile(r"(?m)^([ \t]{0,3})(>+)")
_BQ = "\x00"
_PDF_BULLET = re.compile("(?m)^([ \\t]*)o[ \\t]+(?=[\"'\u201c\u2018(A-Z])")
#: Block elements get ``dir="auto"`` so an Arabic lead-in and an English quotation each read in their own direction
#: (post-freeze PF-11); the attribute is added only where none is present.
_BLOCK_OPEN = re.compile(r"<(p|li|blockquote|h[1-6]|td|th|pre)((?:\s(?![^>]*\bdir=)[^>]*)?)>", re.I)


def _rewrite_anchor(match: "re.Match[str]") -> str:
    attrs = match.group(1)
    href = _HREF.search(attrs)
    url = (href.group(1) if href and href.group(1) is not None else (href.group(2) if href else "")) or ""
    url = html.unescape(url).strip()
    if not _SAFE_SCHEME.match(url):
        return "<span>"                                   # the closing </a> is rewritten below
    return '<a href="%s" target="_blank" rel="noopener noreferrer">' % html.escape(url, quote=True)


def answer_html(text: str) -> str:
    """The answer as safe HTML: escaped, Markdown-rendered, links filtered."""
    if not text:
        return ""
    # Display only (the stored answer stays the exact quotation): the PDF's hollow list bullet is extracted as the
    # letter "o" (``o "E" Lecturer is mainly devoted ...``); at the start of a line it is shown as a bullet.
    text = _PDF_BULLET.sub(lambda m: m.group(1) + "• ", text)
    # Markdown's blockquote marker is a ">" at the start of a line; it is kept as Markdown, everything else is escaped.
    protected = _BLOCKQUOTE_MARK.sub(lambda m: m.group(1) + _BQ * len(m.group(2)), text)
    rendered = prepare_content(html.escape(protected, quote=False).replace(_BQ, ">"), _EXTRAS)
    out = []
    pos = 0
    open_span = 0
    for m in _ANCHOR.finditer(rendered):
        out.append(rendered[pos:m.start()])
        replacement = _rewrite_anchor(m)
        if replacement == "<span>":
            open_span += 1
        out.append(replacement)
        pos = m.end()
    out.append(rendered[pos:])
    result = _BLOCK_OPEN.sub(r'<\1 dir="auto"\2>', "".join(out))
    if open_span:
        # every </a> that belonged to a dropped anchor becomes </span>; anchors are not nested in Markdown output, so the
        # first `open_span` closing tags after dropped anchors are theirs
        result = _close_dropped(result)
    return result


def _close_dropped(rendered: str) -> str:
    """Pair each ``<span>`` that replaced an unsafe anchor with its ``</a>``, in document order."""
    out = []
    pos = 0
    pending = 0
    for m in re.finditer(r"<a\s[^>]*>|<span>|</a>", rendered):
        tag = m.group(0)
        out.append(rendered[pos:m.start()])
        if tag == "<span>":
            pending += 1
            out.append(tag)
        elif tag == "</a>":
            if pending:
                pending -= 1
                out.append("</span>")
            else:
                out.append(tag)
        else:
            out.append(tag)
        pos = m.end()
    out.append(rendered[pos:])
    return "".join(out)


def plain_text(rendered_html: str) -> str:
    """The visible text of rendered HTML (for titles and copy helpers)."""
    return html.unescape(_TAG.sub("", rendered_html or "")).strip()


# ---------------------------------------------------------------------------
# Copy answer
# ---------------------------------------------------------------------------
#: One delegated click handler for every "Copy answer" button. The button carries no script: the text to copy sits
#: beside it as JSON inside a hidden element, so no answer text is ever written into an HTML attribute or into
#: JavaScript source (an inline ``onclick`` broke on the quotes of its own code and on any apostrophe in an answer).
#: ``navigator.clipboard`` exists only in a secure context (https or localhost); over plain http on a local network
#: (a phone on the same Wi-Fi) the handler falls back to a temporary text box and ``document.execCommand('copy')``.
#: The label reports the real outcome: the "copied" text only after a copy succeeded, the "failed" text otherwise.
COPY_JS = r"""
(function () {
  if (window.fobCopy) return;
  function legacyCopy(text) {
    var box = document.createElement('textarea');
    box.value = text; box.setAttribute('readonly', ''); box.setAttribute('aria-hidden', 'true');
    box.style.position = 'fixed'; box.style.top = '0'; box.style.left = '-9999px'; box.style.opacity = '0';
    document.body.appendChild(box);
    var ok = false;
    try { box.select(); box.setSelectionRange(0, text.length); ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(box);
    return ok;
  }
  function report(btn, ok) {
    var label = btn.querySelector('.copy-lbl');
    var message = btn.getAttribute(ok ? 'data-copied' : 'data-copy-failed');
    if (!label || !message) return;
    if (btn.fobCopyTimer) clearTimeout(btn.fobCopyTimer); else btn.fobCopyLabel = label.textContent;
    label.textContent = message;
    btn.fobCopyTimer = setTimeout(function () { label.textContent = btn.fobCopyLabel; btn.fobCopyTimer = null; }, 1400);
  }
  window.fobCopy = function (btn) {
    var source = btn.parentNode ? btn.parentNode.querySelector('.copy-src') : null;
    var text;
    try { text = JSON.parse(source.textContent); } catch (e) { report(btn, false); return false; }
    if (typeof text !== 'string') { report(btn, false); return false; }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { report(btn, true); }, function () { report(btn, legacyCopy(text)); });
    } else {
      report(btn, legacyCopy(text));
    }
    return true;
  };
  document.addEventListener('click', function (ev) {
    var btn = ev.target && ev.target.closest ? ev.target.closest('button[data-copy-answer]') : null;
    if (btn) window.fobCopy(btn);
  });
})();
"""


def copy_button_html(text: str, label: str, copied: str, failed: str, icon: str = "") -> str:
    """The "Copy answer" button and, beside it, the text it copies (JSON in a hidden element, HTML-escaped). The click
    is handled by :data:`COPY_JS`; nothing here is executable, whatever characters the answer contains."""
    esc = lambda value: html.escape(value, quote=True)                                     # noqa: E731
    return ('<button type="button" class="icon-btn" data-copy-answer aria-label="%s" data-copied="%s" data-copy-failed="%s">'
            '%s<span class="copy-lbl">%s</span></button><span class="copy-src" hidden style="display:none">%s</span>'
            % (esc(label), esc(copied), esc(failed), icon, html.escape(label), html.escape(json.dumps(text), quote=False)))


__all__ = ["COPY_JS", "answer_html", "copy_button_html", "plain_text"]
