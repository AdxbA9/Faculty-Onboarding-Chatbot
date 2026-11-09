"""
Faculty onboarding features (post-freeze PF-14, decision record D-125).

Three connected, deterministic layers over the approved corpus and the PF-13 store:

    catalog.py   the curated task catalog, the guided procedures (every step an exact quote of an indexed chunk of an
                 approved guide) and the resource catalog (approved documents, departments with verified contacts,
                 official links checked individually); every entry carries its provenance
    progress.py  guest-owned task progress and procedure positions on the PF-13 store (schema version 2): status
                 transitions, dashboard figures, audit events; nothing here calls a model
    finder.py    bilingual lexical search over the resource catalog, category filters and task context
    links.py     the only URL policy: scheme allow-list, verified-link records, safe rendering helpers

Agentic AI is NOT inside this package: a contextual question from a task or a step is a question typed into the chat
composer and answered by the unchanged controlled pipeline (``ui.pipeline.ask_with_diagnostics``); the answer never
changes any progress row.
"""
from .catalog import (CATALOG_VERSION, CATEGORIES, KINDS, PROCEDURES, RESOURCES, STATUSES, TASKS, procedure, procedure_for_task, resource, task,
                      tasks_by_category)

__all__ = ["CATALOG_VERSION", "CATEGORIES", "KINDS", "PROCEDURES", "RESOURCES", "STATUSES", "TASKS", "procedure", "procedure_for_task", "resource",
           "task", "tasks_by_category"]
