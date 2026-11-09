"""
Bilingual lexical search over the resource catalog (post-freeze PF-14). Deterministic, no index, no model.

A resource matches when a query token (English, lower-cased; Arabic, normalised with the PF-11 helpers so that
hamza, ta marbuta and the definite article do not matter) occurs in its title, keywords or description. Scores favour
titles and keywords over descriptions, and resources tied to the task the user is looking at. A resource with no match
is never shown "to fill the page": an empty query lists the catalog by category instead.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .catalog import CATEGORIES, RESOURCES, Resource, task

try:                                                               # the PF-11 normalisation, when the package is importable
    from handbook_bot.arabic_text import normalize_arabic as _normalize_arabic   # type: ignore
except Exception:                                                  # pragma: no cover - defensive import
    _normalize_arabic = None

_ARABIC = re.compile(r"[؀-ۿ]")
_TOKEN = re.compile(r"[\w؀-ۿ][\w؀-ۿ'’-]*", re.UNICODE)
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
STOP_EN = frozenset("a an the of to for in on at and or how do i my is are what where can".split())
STOP_AR = frozenset("في من على عن الى إلى ما هو هي كيف أين هل و أو".split())


def _norm_ar(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    if _normalize_arabic is not None:
        try:
            text = _normalize_arabic(text)
        except Exception:
            pass
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")
    return text


def normalize(text: str) -> str:
    text = (text or "").strip().lower()
    return _norm_ar(text) if _ARABIC.search(text) else text


def tokens(text: str) -> List[str]:
    out = []
    for tok in _TOKEN.findall(normalize(text)):
        tok = tok.strip("'’-")
        if not tok or tok in STOP_EN or tok in STOP_AR:
            continue
        if _ARABIC.search(tok) and tok.startswith("ال") and len(tok) > 4:
            tok = tok[2:]
        out.append(tok)
    return out


def _fields(resource: Resource) -> Dict[str, str]:
    return {
        "title": normalize(resource.title_en + " " + resource.title_ar),
        "keywords": " | ".join(normalize(k) for k in resource.keywords_en + resource.keywords_ar),
        "description": normalize(resource.description_en + " " + resource.description_ar),
    }


def score(resource: Resource, query_tokens: List[str], task_id: Optional[str] = None) -> int:
    """Title and keyword hits weigh more than description hits; matching more of the query's tokens (coverage) and the
    whole query as a phrase weigh more still; a resource tied to the task in view gets a bonus."""
    f = _fields(resource)
    total = 0
    matched = 0
    for tok in query_tokens:
        hit = False
        if tok in f["title"]:
            total += 3
            hit = True
        if tok in f["keywords"]:
            total += 3
            hit = True
        elif tok in f["description"]:
            total += 1
            hit = True
        matched += 1 if hit else 0
    if matched:
        total += 2 * matched
        phrase = " ".join(query_tokens)
        if len(query_tokens) > 1 and (phrase in f["title"] or phrase in f["keywords"]):
            total += 4
    if task_id and total:
        t = task(task_id)
        if t and (resource.id in t.resources or task_id in resource.related_tasks):
            total += 5
    return total


def search(query: str = "", *, category: Optional[str] = None, task_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Ranked resources: ``[{"resource": Resource, "score": int, "contextual": bool}]``.

    With a query, only matching resources are returned (score > 0). Without one, the catalog is listed (category filter
    and task context applied), contextual resources first.
    """
    cat = category if category in CATEGORIES else None
    q = tokens(query or "")
    context_ids = set()
    context_rank: Dict[str, int] = {}
    if task_id:
        t = task(task_id)
        if t:
            context_rank = {rid: n for n, rid in enumerate(t.resources)}          # the task lists its resources in a deliberate order
            context_ids = set(t.resources) | {r.id for r in RESOURCES.values() if task_id in r.related_tasks}
    out: List[Dict[str, Any]] = []
    for r in RESOURCES.values():
        if cat and r.category != cat:
            continue
        s = score(r, q, task_id) if q else (5 if r.id in context_ids else 1)
        if q and s <= 0:
            continue
        out.append({"resource": r, "score": s, "contextual": r.id in context_ids})
    type_rank = {"document": 0, "department": 1, "website": 2}
    out.sort(key=lambda e: (-e["score"], 0 if e["contextual"] else 1, context_rank.get(e["resource"].id, 99), type_rank.get(e["resource"].type, 9), e["resource"].title_en))
    return out[:max(int(limit), 0)]


__all__ = ["normalize", "score", "search", "tokens"]
