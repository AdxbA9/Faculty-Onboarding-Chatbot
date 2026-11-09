"""
Deterministic progress arithmetic for the onboarding checklist (post-freeze PF-14). Pure functions over the catalog and
the stored progress rows; the store (``persistence.Store``) owns the rows, this module never writes.

Rules:
* a task is APPLICABLE unless its status is ``not_applicable``;
* progress percentage = completed / applicable, rounded to a whole number, 0 when nothing is applicable;
* "next activities" are the first applicable tasks that are not completed, required ones first, then recommended,
  optional and informational, in catalog order;
* any status may move to any other status: completion is the user's statement about their own work, not an approval,
  so it can be undone; a repeated identical update is a no-op (the store writes no second event).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .catalog import CATEGORIES, KINDS, STATUSES, Task, tasks_by_category

KIND_RANK = {kind: n for n, kind in enumerate(KINDS)}
DEFAULT_STATUS = "not_started"


def status_of(progress: Mapping[str, Mapping[str, Any]], task_id: str) -> str:
    row = progress.get(task_id)
    status = row.get("status") if isinstance(row, Mapping) else None
    return status if status in STATUSES else DEFAULT_STATUS


def valid_status(value: object) -> bool:
    return isinstance(value, str) and value in STATUSES


def dashboard(progress: Mapping[str, Mapping[str, Any]], tasks: Optional[Iterable[Task]] = None, next_limit: int = 3) -> Dict[str, Any]:
    """Totals, percentage, per-category breakdown and the next activities, from stored statuses only."""
    items = list(tasks) if tasks is not None else tasks_by_category()
    counts = {s: 0 for s in STATUSES}
    by_category: Dict[str, Dict[str, int]] = {c: {"total": 0, "applicable": 0, "completed": 0, "in_progress": 0} for c in CATEGORIES}
    for t in items:
        status = status_of(progress, t.id)
        counts[status] += 1
        cat = by_category.setdefault(t.category, {"total": 0, "applicable": 0, "completed": 0, "in_progress": 0})
        cat["total"] += 1
        if status != "not_applicable":
            cat["applicable"] += 1
        if status == "completed":
            cat["completed"] += 1
        if status == "in_progress":
            cat["in_progress"] += 1
    total = len(items)
    applicable = total - counts["not_applicable"]
    completed = counts["completed"]
    percent = int(round(100.0 * completed / applicable)) if applicable else 0
    pending = [t for t in items if status_of(progress, t.id) in ("not_started", "in_progress")]
    pending.sort(key=lambda t: (0 if status_of(progress, t.id) == "in_progress" else 1, KIND_RANK.get(t.kind, 9), t.order))
    return {
        "total": total, "applicable": applicable, "completed": completed, "in_progress": counts["in_progress"],
        "not_started": counts["not_started"], "not_applicable": counts["not_applicable"],
        "remaining": max(applicable - completed, 0), "percent": percent,
        "by_category": by_category, "next": [t.id for t in pending[:max(next_limit, 0)]],
        "all_done": applicable > 0 and completed == applicable,
    }


def summarize_events(events: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Audit rows in a stable, JSON-able shape (newest first as the store returns them)."""
    return [{"task_id": e.get("task_id"), "from": e.get("from_status"), "to": e.get("to_status"), "origin": e.get("origin"), "at": e.get("created_at")}
            for e in events]


__all__ = ["DEFAULT_STATUS", "KIND_RANK", "dashboard", "status_of", "summarize_events", "valid_status"]
