"""
Installable web app (PWA) support: the web app manifest, the home-screen icons, the theme colour and a minimal service
worker. The website stays one website; this only lets a phone add it to the home screen and open it without browser bars.

    /manifest.webmanifest     name, icons, ``display: standalone``, theme colour #0B7A52, shortcuts
    /sw.js                    service worker (scope "/"): shows /offline.html when a page cannot be loaded; it never caches
                              pages or answers, so what the reader sees always comes from the running application
    /offline.html             a small bilingual "no connection" page
    /static/icons/*           icons generated from assets/logo.png by qa/mobile_pwa/make_icons.py

``HEAD_HTML`` is added to the ``<head>`` of every page next to the stylesheet (``ui/chat.py`` and ``ui/shell.py``);
``register(app)`` adds the three routes (``start_app``). Browsers only offer installation over HTTPS or on ``localhost``.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import Response

APP_NAME = "UOS Faculty Onboarding Chatbot"
SHORT_NAME = "UOS Faculty"
THEME_COLOR = "#0B7A52"
BACKGROUND_COLOR = "#F3F5F4"
ICON_BASE = "/static/icons"
MANIFEST_URL = "/manifest.webmanifest"
SERVICE_WORKER_URL = "/sw.js"
OFFLINE_URL = "/offline.html"
CACHE_NAME = "fob-shell-v1"

#: ``viewport-fit=cover`` lets the page use the whole screen in standalone mode (the CSS keeps clear of the notch and the
#: home indicator); ``interactive-widget=resizes-content`` makes Android resize the page when the keyboard opens.
VIEWPORT = "width=device-width, initial-scale=1, viewport-fit=cover, interactive-widget=resizes-content"


def manifest() -> Dict[str, Any]:
    return {
        "id": "/",
        "name": APP_NAME,
        "short_name": SHORT_NAME,
        "description": "Answers for University of Sharjah faculty from the Faculty Handbook and official guides, in English and Arabic.",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "orientation": "any",
        "dir": "auto",
        "lang": "en",
        "theme_color": THEME_COLOR,
        "background_color": BACKGROUND_COLOR,
        "categories": ["education", "productivity"],
        "icons": [
            {"src": ICON_BASE + "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": ICON_BASE + "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
            {"src": ICON_BASE + "/icon-maskable-192.png", "sizes": "192x192", "type": "image/png", "purpose": "maskable"},
            {"src": ICON_BASE + "/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        ],
        "shortcuts": [
            {"name": "New chat", "url": "/?new=1", "icons": [{"src": ICON_BASE + "/icon-192.png", "sizes": "192x192"}]},
            {"name": "Onboarding checklist", "url": "/onboarding", "icons": [{"src": ICON_BASE + "/icon-192.png", "sizes": "192x192"}]},
            {"name": "Resources & documents", "url": "/resources", "icons": [{"src": ICON_BASE + "/icon-192.png", "sizes": "192x192"}]},
        ],
    }


HEAD_HTML = """
<link rel="manifest" href="%(manifest)s">
<meta name="theme-color" content="%(theme)s">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="%(short)s">
<meta name="application-name" content="%(short)s">
<meta name="format-detection" content="telephone=no">
<link rel="apple-touch-icon" sizes="180x180" href="%(icons)s/apple-touch-icon.png">
<script>
if ('serviceWorker' in navigator && window.isSecureContext) {
  window.addEventListener('load', function () { navigator.serviceWorker.register('%(sw)s', {scope: '/'}).catch(function () {}); });
}
</script>
""" % {"manifest": MANIFEST_URL, "theme": THEME_COLOR, "short": SHORT_NAME, "icons": ICON_BASE, "sw": SERVICE_WORKER_URL}

SERVICE_WORKER_JS = """// UOS Faculty Onboarding Chatbot: service worker. Pages and answers are never cached.
const CACHE = '%(cache)s';
const OFFLINE = '%(offline)s';
self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll([OFFLINE, '%(icons)s/icon-192.png'])).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (event) => {
  if (event.request.mode !== 'navigate') return;                       // everything else goes straight to the network
  event.respondWith(fetch(event.request).catch(() => caches.match(OFFLINE)));
});
""" % {"cache": CACHE_NAME, "offline": OFFLINE_URL, "icons": ICON_BASE}

OFFLINE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="%(theme)s"><title>%(name)s</title>
<style>
body{margin:0;min-height:100vh;min-height:100dvh;display:grid;place-items:center;background:#F3F5F4;color:#17201C;font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Arabic",Tahoma,sans-serif;line-height:1.6}
main{max-width:420px;padding:32px 24px;text-align:center}
img{width:96px;height:96px;border-radius:22px}
h1{font-size:20px;margin:18px 0 6px}p{margin:0 0 10px;font-size:15px;color:#46524C}
button{margin-top:14px;min-height:48px;padding:0 28px;border:0;border-radius:999px;background:%(theme)s;color:#fff;font:inherit;font-weight:600;font-size:16px}
@media (prefers-color-scheme: dark){body{background:#0F1513;color:#E7ECE9}p{color:#A9B5AF}}
</style></head>
<body><main>
<img src="%(icons)s/icon-192.png" alt="">
<h1>The assistant cannot be reached</h1>
<p>Check your connection to the university network, then try again.</p>
<h1 lang="ar" dir="rtl">تعذّر الوصول إلى المساعد</h1>
<p lang="ar" dir="rtl">تحقق من اتصالك بشبكة الجامعة ثم حاول مرة أخرى.</p>
<button type="button" onclick="location.reload()">Try again · إعادة المحاولة</button>
</main></body></html>
""" % {"theme": THEME_COLOR, "name": APP_NAME, "icons": ICON_BASE}


def register(app: Any) -> None:
    """Add the manifest, service worker and offline page to the (NiceGUI / FastAPI) application."""
    @app.get(MANIFEST_URL, include_in_schema=False)
    def _manifest() -> Response:
        return Response(json.dumps(manifest(), ensure_ascii=False), media_type="application/manifest+json", headers={"Cache-Control": "no-cache"})

    @app.get(SERVICE_WORKER_URL, include_in_schema=False)
    def _service_worker() -> Response:
        return Response(SERVICE_WORKER_JS, media_type="text/javascript", headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})

    @app.get(OFFLINE_URL, include_in_schema=False)
    def _offline() -> Response:
        return Response(OFFLINE_HTML, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})


__all__ = ["APP_NAME", "BACKGROUND_COLOR", "HEAD_HTML", "ICON_BASE", "MANIFEST_URL", "OFFLINE_HTML", "OFFLINE_URL", "SERVICE_WORKER_JS", "SERVICE_WORKER_URL",
           "SHORT_NAME", "THEME_COLOR", "VIEWPORT", "manifest", "register"]
