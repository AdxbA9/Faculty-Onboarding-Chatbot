"""The installable web app (mobile pass): manifest, icons, head tags, service worker and the phone rules of the stylesheet."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui import pwa, shell
from ui.styles import GLOBAL_CSS

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _client() -> TestClient:
    app = FastAPI()
    pwa.register(app)
    return TestClient(app)


def test_the_manifest_makes_the_site_installable_in_standalone_mode() -> None:
    response = _client().get(pwa.MANIFEST_URL)
    assert response.status_code == 200 and "manifest+json" in response.headers["content-type"]
    data = json.loads(response.text)
    assert data["display"] == "standalone" and data["start_url"] == "/" and data["scope"] == "/"
    assert data["theme_color"] == "#0B7A52" and data["name"] and len(data["short_name"]) <= 12
    sizes = {(icon["sizes"], icon["purpose"]) for icon in data["icons"]}
    assert {("192x192", "any"), ("512x512", "any"), ("512x512", "maskable")} <= sizes


def test_every_manifest_icon_exists_with_the_declared_size() -> None:
    from PIL import Image
    for icon in pwa.manifest()["icons"]:
        path = ASSETS / icon["src"].replace("/static/", "")
        assert path.is_file(), path
        with Image.open(path) as image:
            assert "%dx%d" % image.size == icon["sizes"]
    with Image.open(ASSETS / "icons" / "apple-touch-icon.png") as image:
        assert image.size == (180, 180) and image.mode == "RGB"          # iOS shows transparency as black


def test_the_head_links_the_manifest_theme_colour_and_touch_icon() -> None:
    head = pwa.HEAD_HTML
    assert 'rel="manifest" href="/manifest.webmanifest"' in head
    assert '<meta name="theme-color" content="#0B7A52">' in head
    assert "apple-touch-icon" in head and "apple-mobile-web-app-capable" in head
    assert "serviceWorker" in head and "isSecureContext" in head
    assert "viewport-fit=cover" in pwa.VIEWPORT and "interactive-widget=resizes-content" in pwa.VIEWPORT
    assert "user-scalable=no" not in pwa.VIEWPORT and "maximum-scale" not in pwa.VIEWPORT      # pinch-zoom stays available


def test_the_service_worker_only_handles_page_loads_and_falls_back_to_the_offline_page() -> None:
    client = _client()
    worker = client.get(pwa.SERVICE_WORKER_URL)
    assert worker.status_code == 200 and "javascript" in worker.headers["content-type"]
    assert "event.request.mode !== 'navigate'" in worker.text and pwa.OFFLINE_URL in worker.text
    offline = client.get(pwa.OFFLINE_URL)
    assert offline.status_code == 200 and 'dir="rtl"' in offline.text and "Try again" in offline.text


def test_the_stylesheet_and_shell_script_carry_the_phone_rules() -> None:
    assert "var(--app-h, 100dvh)" in GLOBAL_CSS and "env(safe-area-inset-bottom" in GLOBAL_CSS
    assert "html.kb-open" in GLOBAL_CSS and "(pointer: coarse)" in GLOBAL_CSS
    assert "window.fobFit" in shell.SHELL_JS and "visualViewport" in shell.SHELL_JS and "kb-open" in shell.SHELL_JS
