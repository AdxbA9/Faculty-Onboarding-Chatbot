"""
The secret that signs the browser session cookie (post-freeze PF-13).

NiceGUI's ``app.storage.browser`` is Starlette's signed session cookie (HttpOnly, SameSite=Lax, 14-day max age). The
guest identity of a browser lives in it, so the signing secret must survive restarts: until PF-13 ``start_app`` drew a new
random secret on every start, which silently gave every browser a new identity after each restart. The secret comes from
``APP_STORAGE_SECRET`` when set, else from ``var/storage_secret.txt`` (created once with 48 random bytes, readable by the
owner only where the platform allows). The file is ignored by git; the value is never logged.
"""
from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path
from typing import Optional

DEFAULT_SECRET_PATH = os.path.join("var", "storage_secret.txt")


def storage_secret(path: Optional[str] = None) -> str:
    env = os.environ.get("APP_STORAGE_SECRET", "").strip()
    if env:
        return env
    target = Path(path or os.environ.get("APP_STORAGE_SECRET_PATH", "").strip() or DEFAULT_SECRET_PATH)
    if target.exists():
        value = target.read_text(encoding="utf-8").strip()
        if value:
            return value
    target.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    target.write_text(value, encoding="utf-8")
    try:
        os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return value


__all__ = ["DEFAULT_SECRET_PATH", "storage_secret"]
