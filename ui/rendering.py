"""
Safe HTML for the chat page (post-freeze PF-10).

The answer text comes from the model or from an extractor. NiceGUI's Markdown element passes raw HTML through
(``<script>`` survives ``prepare_content``), so an answer is rendered here instead: every HTML special character is
escaped FIRST, then Markdown is applied, then every link is checked. Only ``http``, ``https`` and ``mailto`` links are
kept, and they open in a new tab with ``rel="noopener noreferrer"``; any other scheme becomes plain text. Nothing else
of the answer is changed, so lists, numbered steps, tables and code blocks still render.
"""
from __future__ import annotations

import html
import re

from nicegui.elements.markdown import prepare_content

_EXTRAS = "fenced-code-blocks tables"
_ANCHOR = re.compile(r"<a\s+([^>]*?)>", re.I)
_HREF = re.compile(r"""href\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_SAFE_SCHEME = re.compile(r"^(?:https?://|mailto:)", re.I)
_TAG = re.compile(r"<[^>]+>")


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
    rendered = prepare_content(html.escape(text, quote=False), _EXTRAS)
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
    result = "".join(out)
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


__all__ = ["answer_html", "plain_text"]
