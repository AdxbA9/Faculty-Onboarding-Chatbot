"""
The URL policy of the onboarding features (post-freeze PF-14).

A link reaches a page only when it comes from the catalog (``catalog.Link``: opened and checked on a recorded date, or
printed inside an approved guide and labelled so) AND passes :func:`safe_url`: an absolute ``https``/``http`` URL with a
host, no credentials, no whitespace or control characters, no ``javascript:``/``data:``/``file:`` scheme. Rendering goes
through :func:`anchor`, which escapes the text and adds ``rel="noopener noreferrer"``. Nothing builds a URL from a title.
"""
from __future__ import annotations

import html
import re
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

ALLOWED_SCHEMES = ("https", "http")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")
_HOST = re.compile(r"^[A-Za-z0-9.-]+(?::\d{1,5})?$")


def safe_url(url: object) -> Optional[str]:
    """The URL when it may be rendered as a link, else None."""
    if not isinstance(url, str) or not url or _CONTROL.search(url) or len(url) > 2048:
        return None
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in ALLOWED_SCHEMES or not parts.netloc or parts.username or parts.password:
        return None
    if not _HOST.match(parts.hostname or "") or "." not in (parts.hostname or ""):
        return None
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path or "/", parts.query, ""))


def anchor(url: str, text: str, *, css: str = "res-link", title: str = "") -> str:
    """An escaped external anchor, or escaped text when the URL is not allowed (never a broken or unsafe href)."""
    safe = safe_url(url)
    label = html.escape(text, quote=False)
    if safe is None:
        return '<span class="%s unavailable">%s</span>' % (css, label)
    return '<a class="%s" href="%s" target="_blank" rel="noopener noreferrer" title="%s">%s</a>' % (
        css, html.escape(safe, quote=True), html.escape(title or safe, quote=True), label)


def host_of(url: str) -> str:
    safe = safe_url(url)
    return urlsplit(safe).hostname or "" if safe else ""


__all__ = ["ALLOWED_SCHEMES", "anchor", "host_of", "safe_url"]
