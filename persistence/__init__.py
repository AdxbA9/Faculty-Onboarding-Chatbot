"""
Application persistence (post-freeze PF-13, decision record D-124).

SQLite through the standard library, one file under ``var/`` (``APP_DB_PATH``), for the user-facing application state:
guest owners, their preferences, conversations, messages and the structured record of every answer with its evidence.
It is NOT the knowledge base: FAISS, the chunks and the registry stay where they are and are never copied here.

    db.py       connection settings, versioned migrations (backup first when data is at risk), integrity check,
                backup and restore through SQLite's online backup API
    store.py    the owner-scoped persistence service the page uses (every query carries the owner)
    records.py  how a verified answer becomes rows and how rows become the ``QAResult`` and diagnostics the page renders
    secret.py   the stable secret that signs the browser session cookie (without it every restart forgets who is who)
    manage.py   command line: init, check, info, backup, restore, retention
"""
from .store import AccessError, Store, StoreError, StoreUnavailable, get_store, reset_store

__all__ = ["AccessError", "Store", "StoreError", "StoreUnavailable", "get_store", "reset_store"]
