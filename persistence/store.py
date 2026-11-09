"""
The owner-scoped persistence service (post-freeze PF-13).

Every operation takes the owner it acts for and every query carries ``owner_id`` in its WHERE clause: a conversation id
from a browser is never authorisation on its own, and a conversation of another owner is simply absent (``None`` or
:class:`AccessError`), so no probe can tell "forbidden" from "nonexistent". Owners are identified by the SHA-256 hash of
the browser's session identity: the database never holds the identity itself.

One connection per operation, each write inside one explicit transaction (``BEGIN IMMEDIATE`` ... ``COMMIT``), so an
interrupted write leaves nothing half-stored and the caller learns of a failure (:class:`StoreUnavailable`) instead of a
silent "saved". Appends are idempotent per ``client_token`` (a repeated callback returns the existing message). All SQL
is parameterised; no user text is ever concatenated into a statement.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional

from . import db as _db
from .records import answer_record, dumps, loads, restore_turn

PREFERENCE_KEYS = ("interface_lang", "answer_lang", "theme", "voice_lang", "text_size", "reduced_motion", "display_name")
#: The allowed values of each choice preference (PF-15: every value is validated before it is stored).
PREFERENCE_VALUES = {"interface_lang": ("en", "ar"), "answer_lang": ("auto", "en", "ar"), "voice_lang": ("auto", "en", "ar"),
                     "theme": ("light", "dark"), "text_size": ("standard", "large"), "reduced_motion": ("off", "on")}
MAX_DISPLAY_NAME = 40
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
_LONG_NUMBER = re.compile(r"\d{6,}")


def clean_display_name(value: Any) -> Optional[str]:
    """The display name to store, or None to remove it. Raises StoreError for anything that is not a short plain name:
    control or bidi-override characters, more than 40 characters, an email address or a long number (the profile asks for
    no identifiers)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise StoreError("invalid display name")
    if _CONTROL_CHARS.search(value.replace("\n", " ").replace("\t", " ")) or "\n" in value or "\r" in value:
        raise StoreError("invalid display name")
    text = " ".join(value.split())
    if not text:
        return None
    if len(text) > MAX_DISPLAY_NAME or "@" in text or _LONG_NUMBER.search(text) or any(ch in text for ch in "<>{}[]\\`"):
        raise StoreError("invalid display name")
    return text


def clean_preference(key: str, value: Any) -> Optional[str]:
    """The stored form of one preference value (None clears it); StoreError for a value outside the allowed set."""
    if value is None or value == "":
        return None
    if key == "display_name":
        return clean_display_name(value)
    allowed = PREFERENCE_VALUES.get(key)
    if allowed is None or str(value) not in allowed:
        raise StoreError("invalid value for %s" % key)
    return str(value)
TASK_STATUSES = ("not_started", "in_progress", "completed", "not_applicable")
MAX_TITLE = 120
MAX_CONTENT = 20000


class StoreError(RuntimeError):
    """Base class of the store's own errors."""


class StoreUnavailable(StoreError):
    """The database could not be read or written (locked beyond the timeout, missing, corrupt, disk full...)."""


class AccessError(StoreError):
    """The record does not exist for this owner."""


class Store:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = _db.db_path(path)
        self._lock = threading.RLock()
        self.init_report = _db.initialize(self.path)

    # ---- plumbing --------------------------------------------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        try:
            return _db.connect(self.path)
        except sqlite3.Error as exc:
            raise StoreUnavailable("database unavailable: %s" % type(exc).__name__) from exc

    def _write(self, fn):
        """Run ``fn(conn)`` inside one immediate transaction; roll back on any error and surface it."""
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(conn)
                    conn.execute("COMMIT")
                    return result
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
            except sqlite3.OperationalError as exc:
                raise StoreUnavailable("database write failed: %s" % str(exc)[:80]) from exc
            except sqlite3.DatabaseError as exc:
                raise StoreUnavailable("database error: %s" % type(exc).__name__) from exc
            finally:
                conn.close()

    def _read(self, fn):
        conn = self._conn()
        try:
            return fn(conn)
        except sqlite3.DatabaseError as exc:
            raise StoreUnavailable("database read failed: %s" % type(exc).__name__) from exc
        finally:
            conn.close()

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(("fob-owner:" + str(token)).encode("utf-8")).hexdigest()

    # ---- owners ----------------------------------------------------------------------------------------------------------
    def find_owner(self, token: str) -> Optional[int]:
        """The owner id for a browser identity token when one exists; None otherwise (a read: no row is created)."""
        if not token or not isinstance(token, str):
            return None
        digest = self.token_hash(token)
        row = self._read(lambda conn: conn.execute("SELECT id FROM owners WHERE token_hash = ?", (digest,)).fetchone())
        return int(row["id"]) if row else None

    def owner_for_token(self, token: str) -> int:
        """The owner id for a browser identity token (created on first sight; ``last_seen_at`` refreshed)."""
        if not token or not isinstance(token, str):
            raise AccessError("no identity")
        digest = self.token_hash(token)
        now = _db.now_iso()

        def op(conn):
            row = conn.execute("SELECT id FROM owners WHERE token_hash = ?", (digest,)).fetchone()
            if row:
                conn.execute("UPDATE owners SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
                return int(row["id"])
            cur = conn.execute("INSERT INTO owners (token_hash, kind, created_at, last_seen_at) VALUES (?, 'guest', ?, ?)", (digest, now, now))
            return int(cur.lastrowid)
        return self._write(op)

    # ---- preferences ----------------------------------------------------------------------------------------------------
    def get_preferences(self, owner_id: int) -> Dict[str, Any]:
        def op(conn):
            row = conn.execute("SELECT interface_lang, answer_lang, theme, voice_lang, text_size, reduced_motion, display_name, extra, updated_at "
                               "FROM preferences WHERE owner_id = ?", (owner_id,)).fetchone()
            if not row:
                return {}
            out = {k: row[k] for k in PREFERENCE_KEYS if row[k] is not None}
            out["updated_at"] = row["updated_at"]
            extra = loads(row["extra"], {})
            out["extra"] = extra if isinstance(extra, dict) else {}
            return out
        return self._read(op)

    def set_preferences(self, owner_id: int, **changes: Any) -> Dict[str, Any]:
        """Update the given preference columns (unknown keys rejected); returns the stored preferences."""
        unknown = [k for k in changes if k not in PREFERENCE_KEYS]
        if unknown:
            raise StoreError("unknown preference(s): %s" % ", ".join(sorted(unknown)))
        cleaned = {key: clean_preference(key, value) for key, value in changes.items()}     # validated before any write
        now = _db.now_iso()

        def op(conn):
            row = conn.execute("SELECT owner_id FROM preferences WHERE owner_id = ?", (owner_id,)).fetchone()
            if not row:
                conn.execute("INSERT INTO preferences (owner_id, updated_at) VALUES (?, ?)", (owner_id, now))
            for key, value in cleaned.items():
                conn.execute("UPDATE preferences SET %s = ?, updated_at = ? WHERE owner_id = ?" % key, (value, now, owner_id))
            return None
        self._write(op)
        return self.get_preferences(owner_id)

    def set_extra(self, owner_id: int, **values: Any) -> Dict[str, Any]:
        """Merge small page-state values (the active conversation, for instance) into the preferences' ``extra`` JSON."""
        now = _db.now_iso()

        def op(conn):
            row = conn.execute("SELECT extra FROM preferences WHERE owner_id = ?", (owner_id,)).fetchone()
            current = loads(row["extra"], {}) if row else {}
            if not isinstance(current, dict):
                current = {}
            for key, value in values.items():
                if value is None:
                    current.pop(str(key), None)
                else:
                    current[str(key)] = value
            if row:
                conn.execute("UPDATE preferences SET extra = ?, updated_at = ? WHERE owner_id = ?", (dumps(current), now, owner_id))
            else:
                conn.execute("INSERT INTO preferences (owner_id, extra, updated_at) VALUES (?, ?, ?)", (owner_id, dumps(current), now))
            return current
        return self._write(op)

    def reset_preferences(self, owner_id: int) -> None:
        """Return every setting to its default. The optional display name is profile data, not a setting: it is kept
        (PF-15); without one the row is removed as before."""
        now = _db.now_iso()

        def op(conn):
            row = conn.execute("SELECT display_name FROM preferences WHERE owner_id = ?", (owner_id,)).fetchone()
            if row and row["display_name"]:
                conn.execute("UPDATE preferences SET interface_lang = NULL, answer_lang = NULL, theme = NULL, voice_lang = NULL, text_size = NULL, "
                             "reduced_motion = NULL, extra = '{}', updated_at = ? WHERE owner_id = ?", (now, owner_id))
            else:
                conn.execute("DELETE FROM preferences WHERE owner_id = ?", (owner_id,))
            return None
        self._write(op)

    def owner_info(self, owner_id: int) -> Dict[str, Any]:
        """When this guest was first saved and last seen (no identifier is returned)."""
        def op(conn):
            row = conn.execute("SELECT kind, created_at, last_seen_at FROM owners WHERE id = ?", (owner_id,)).fetchone()
            return dict(row) if row else {}
        return self._read(op)

    # ---- conversations --------------------------------------------------------------------------------------------------
    def list_conversations(self, owner_id: int, *, limit: int = 200) -> List[Dict[str, Any]]:
        def op(conn):
            rows = conn.execute(
                "SELECT c.id, c.title, c.status, c.interface_lang, c.created_at, c.updated_at, c.context_task, "
                "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c WHERE c.owner_id = ? AND c.status = 'active' ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?",
                (owner_id, int(limit))).fetchall()
            return [dict(r) for r in rows]
        return self._read(op)

    def create_conversation(self, owner_id: int, title: str = "", interface_lang: Optional[str] = None, context_task: Optional[str] = None) -> Dict[str, Any]:
        cid = secrets.token_urlsafe(16)
        now = _db.now_iso()
        title = (title or "")[:MAX_TITLE]
        context_task = str(context_task)[:64] if context_task else None

        def op(conn):
            conn.execute("INSERT INTO conversations (id, owner_id, title, status, interface_lang, created_at, updated_at, context_task) VALUES (?, ?, ?, 'active', ?, ?, ?, ?)",
                         (cid, owner_id, title, interface_lang, now, now, context_task))
            return {"id": cid, "title": title, "status": "active", "interface_lang": interface_lang, "created_at": now, "updated_at": now, "message_count": 0,
                    "context_task": context_task}
        return self._write(op)

    def get_conversation(self, owner_id: int, conversation_id: str) -> Optional[Dict[str, Any]]:
        def op(conn):
            row = conn.execute("SELECT id, title, status, interface_lang, created_at, updated_at, context_task FROM conversations WHERE id = ? AND owner_id = ?",
                               (str(conversation_id), owner_id)).fetchone()
            return dict(row) if row else None
        return self._read(op)

    def rename_conversation(self, owner_id: int, conversation_id: str, title: str) -> Dict[str, Any]:
        title = " ".join((title or "").split())[:MAX_TITLE]
        if not title:
            raise StoreError("a title is required")
        now = _db.now_iso()

        def op(conn):
            cur = conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND owner_id = ?", (title, now, str(conversation_id), owner_id))
            if cur.rowcount != 1:
                raise AccessError("conversation not found")
            return None
        self._write(op)
        return self.get_conversation(owner_id, conversation_id) or {}

    def delete_conversation(self, owner_id: int, conversation_id: str) -> bool:
        """Delete the conversation and, through the cascades, its messages, answers and evidence. True when it existed."""
        def op(conn):
            cur = conn.execute("DELETE FROM conversations WHERE id = ? AND owner_id = ?", (str(conversation_id), owner_id))
            return cur.rowcount == 1
        return bool(self._write(op))

    def delete_all_conversations(self, owner_id: int) -> int:
        """Delete every conversation of ONE owner (cascading to messages, answers and evidence). Returns the count."""
        return int(self._write(lambda conn: conn.execute("DELETE FROM conversations WHERE owner_id = ?", (owner_id,)).rowcount))

    # ---- messages ---------------------------------------------------------------------------------------------------------
    def _owned(self, conn, owner_id: int, conversation_id: str) -> None:
        row = conn.execute("SELECT 1 FROM conversations WHERE id = ? AND owner_id = ?", (str(conversation_id), owner_id)).fetchone()
        if not row:
            raise AccessError("conversation not found")

    def _existing(self, conn, conversation_id: str, client_token: Optional[str]) -> Optional[int]:
        if not client_token:
            return None
        row = conn.execute("SELECT id FROM messages WHERE conversation_id = ? AND client_token = ?", (str(conversation_id), client_token)).fetchone()
        return int(row["id"]) if row else None

    def append_user_message(self, owner_id: int, conversation_id: str, content: str, *, client_token: Optional[str] = None,
                            answer_language: Optional[str] = None) -> int:
        content = str(content or "")[:MAX_CONTENT]
        if not content.strip():
            raise StoreError("empty message")
        now = _db.now_iso()

        def op(conn):
            self._owned(conn, owner_id, conversation_id)
            existing = self._existing(conn, conversation_id, client_token)
            if existing is not None:
                return existing
            seq = int(conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?", (str(conversation_id),)).fetchone()[0])
            cur = conn.execute("INSERT INTO messages (conversation_id, seq, role, content, answer_language, status, client_token, created_at) VALUES (?, ?, 'user', ?, ?, 'complete', ?, ?)",
                               (str(conversation_id), seq, content, answer_language, client_token, now))
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, str(conversation_id)))
            return int(cur.lastrowid)
        return self._write(op)

    def append_assistant_message(self, owner_id: int, conversation_id: str, result: Any, diagnostics: Optional[Dict[str, Any]], wall_ms: Optional[float], *,
                                 client_token: Optional[str] = None, corpus_key: Optional[str] = None) -> int:
        """Store the assistant turn with its answer row and evidence rows in ONE transaction."""
        record = answer_record(result, diagnostics, wall_ms, corpus_key)
        content = record["message"]["content"][:MAX_CONTENT]
        now = _db.now_iso()

        def op(conn):
            self._owned(conn, owner_id, conversation_id)
            existing = self._existing(conn, conversation_id, client_token)
            if existing is not None:
                return existing
            seq = int(conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?", (str(conversation_id),)).fetchone()[0])
            cur = conn.execute("INSERT INTO messages (conversation_id, seq, role, content, answer_language, status, client_token, created_at) VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?)",
                               (str(conversation_id), seq, content, record["message"]["answer_language"], record["message"]["status"], client_token, now))
            mid = int(cur.lastrowid)
            a = record["answer"]
            conn.execute("INSERT INTO answers (message_id, served, outcome, specialist, plan_f_path, query_type, pages, best_section, evidence_preview, used_llm, llm_calls, "
                         "retried, original_answer, translation, query_transform, diagnostics, corpus_key, wall_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (mid, a["served"], a["outcome"], a["specialist"], a["plan_f_path"], a["query_type"], a["pages"], a["best_section"], a["evidence_preview"],
                          a["used_llm"], a["llm_calls"], a["retried"], a["original_answer"], a["translation"], a["query_transform"], a["diagnostics"], a["corpus_key"], a["wall_ms"]))
            conn.executemany("INSERT INTO evidence_items (message_id, position, source_id, page, section, chunk_type, excerpt, meta, rerank_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             [(mid, e["position"], e["source_id"], e["page"], e["section"], e["chunk_type"], e["excerpt"], e["meta"], e["rerank_score"]) for e in record["evidence"]])
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, str(conversation_id)))
            return mid
        return self._write(op)

    def load_messages(self, owner_id: int, conversation_id: str) -> List[Dict[str, Any]]:
        """The conversation's turns in order, as the page's message records (user turns and restored assistant turns)."""
        def op(conn):
            self._owned(conn, owner_id, conversation_id)
            rows = conn.execute("SELECT id, seq, role, content, answer_language, status, created_at FROM messages WHERE conversation_id = ? ORDER BY seq", (str(conversation_id),)).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                message = dict(row)
                if message["role"] == "user":
                    out.append({"role": "user", "content": message["content"], "result": None, "message_id": message["id"], "stored_at": message["created_at"]})
                    continue
                answer_row = conn.execute("SELECT * FROM answers WHERE message_id = ?", (message["id"],)).fetchone()
                evidence = [dict(e) for e in conn.execute("SELECT * FROM evidence_items WHERE message_id = ? ORDER BY position", (message["id"],)).fetchall()]
                result, diagnostics, extras = restore_turn(message, dict(answer_row) if answer_row else None, evidence)
                out.append({"role": "assistant", "content": result.answer, "result": result, "diagnostics": diagnostics, "wall_ms": extras.get("wall_ms"),
                            "message_id": message["id"], "stored": True, "stored_at": extras.get("stored_at"), "corpus_key": extras.get("corpus_key")})
            return out
        return self._read(op)

    def export_conversation(self, owner_id: int, conversation_id: str) -> Dict[str, Any]:
        """Everything stored for one conversation, JSON-able (the owner's own data only)."""
        conversation = self.get_conversation(owner_id, conversation_id)
        if conversation is None:
            raise AccessError("conversation not found")

        def op(conn):
            messages = []
            for m in conn.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq", (str(conversation_id),)).fetchall():
                entry = dict(m)
                answer = conn.execute("SELECT * FROM answers WHERE message_id = ?", (m["id"],)).fetchone()
                if answer:
                    entry["answer"] = dict(answer)
                    entry["evidence"] = [dict(e) for e in conn.execute("SELECT * FROM evidence_items WHERE message_id = ? ORDER BY position", (m["id"],)).fetchall()]
                messages.append(entry)
            return {"conversation": conversation, "messages": messages}
        return self._read(op)

    # ---- onboarding checklist (post-freeze PF-14) --------------------------------------------------------------------------
    def sync_task_catalog(self, tasks: Iterable[Dict[str, Any]], catalog_version: int) -> Dict[str, int]:
        """Mirror the curated catalog into ``onboarding_tasks`` (idempotent). Tasks missing from the catalog are deactivated,
        never deleted: existing progress rows keep their foreign key and are simply not counted."""
        rows = list(tasks)
        now = _db.now_iso()

        def op(conn):
            seen = set()
            inserted = updated = 0
            for t in rows:
                seen.add(t["id"])
                existing = conn.execute("SELECT catalog_version, category, kind, title_en, title_ar, source_id, page, section, procedure_id, sort_order, active "
                                        "FROM onboarding_tasks WHERE id = ?", (t["id"],)).fetchone()
                values = (int(catalog_version), t["category"], t["kind"], t["title_en"], t["title_ar"], t["source_id"], t.get("page"), t.get("section"),
                          t.get("procedure_id"), int(t.get("sort_order") or 0), 1)
                if existing is None:
                    conn.execute("INSERT INTO onboarding_tasks (id, catalog_version, category, kind, title_en, title_ar, source_id, page, section, procedure_id, sort_order, active, updated_at) "
                                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (t["id"],) + values + (now,))
                    inserted += 1
                elif tuple(existing) != values:
                    conn.execute("UPDATE onboarding_tasks SET catalog_version = ?, category = ?, kind = ?, title_en = ?, title_ar = ?, source_id = ?, page = ?, section = ?, "
                                 "procedure_id = ?, sort_order = ?, active = ?, updated_at = ? WHERE id = ?", values + (now, t["id"]))
                    updated += 1
            deactivated = 0
            for row in conn.execute("SELECT id FROM onboarding_tasks WHERE active = 1").fetchall():
                if row["id"] not in seen:
                    conn.execute("UPDATE onboarding_tasks SET active = 0, updated_at = ? WHERE id = ?", (now, row["id"]))
                    deactivated += 1
            return {"inserted": inserted, "updated": updated, "deactivated": deactivated, "active": len(seen)}
        return self._write(op)

    def active_task_ids(self) -> List[str]:
        return self._read(lambda conn: [r["id"] for r in conn.execute("SELECT id FROM onboarding_tasks WHERE active = 1 ORDER BY sort_order, id").fetchall()])

    def task_progress(self, owner_id: int) -> Dict[str, Dict[str, Any]]:
        """``{task_id: {"status", "updated_at", "completed_at"}}`` for this owner (active tasks only; absent = not started)."""
        def op(conn):
            rows = conn.execute("SELECT p.task_id, p.status, p.updated_at, p.completed_at FROM task_progress p JOIN onboarding_tasks t ON t.id = p.task_id "
                                "WHERE p.owner_id = ? AND t.active = 1", (owner_id,)).fetchall()
            return {r["task_id"]: {"status": r["status"], "updated_at": r["updated_at"], "completed_at": r["completed_at"]} for r in rows}
        return self._read(op)

    def set_task_status(self, owner_id: int, task_id: str, status: str, *, origin: str = "user") -> Dict[str, Any]:
        """Set one task's status for one owner, atomically, with an audit event when it changes. A repeated identical
        update changes nothing and writes no event. Unknown or inactive tasks and unknown statuses are refused."""
        if status not in TASK_STATUSES:
            raise StoreError("unknown status")
        if origin != "user":
            raise StoreError("progress changes come from the user only")
        now = _db.now_iso()

        def op(conn):
            task = conn.execute("SELECT id FROM onboarding_tasks WHERE id = ? AND active = 1", (str(task_id),)).fetchone()
            if task is None:
                raise AccessError("task not found")
            row = conn.execute("SELECT status, completed_at FROM task_progress WHERE owner_id = ? AND task_id = ?", (owner_id, str(task_id))).fetchone()
            previous = row["status"] if row else "not_started"
            if row is not None and previous == status:
                return {"task_id": str(task_id), "status": status, "previous": previous, "changed": False, "updated_at": None}
            completed_at = now if status == "completed" else None
            if row is None:
                conn.execute("INSERT INTO task_progress (owner_id, task_id, status, updated_at, completed_at) VALUES (?, ?, ?, ?, ?)",
                             (owner_id, str(task_id), status, now, completed_at))
            else:
                conn.execute("UPDATE task_progress SET status = ?, updated_at = ?, completed_at = ? WHERE owner_id = ? AND task_id = ?",
                             (status, now, completed_at, owner_id, str(task_id)))
            conn.execute("INSERT INTO progress_events (owner_id, task_id, from_status, to_status, origin, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (owner_id, str(task_id), previous if row is not None else None, status, origin, now))
            return {"task_id": str(task_id), "status": status, "previous": previous, "changed": True, "updated_at": now}
        return self._write(op)

    def progress_events(self, owner_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        return self._read(lambda conn: [dict(r) for r in conn.execute(
            "SELECT task_id, from_status, to_status, origin, created_at FROM progress_events WHERE owner_id = ? ORDER BY id DESC LIMIT ?", (owner_id, int(limit))).fetchall()])

    def procedure_position(self, owner_id: int, procedure_id: str) -> int:
        row = self._read(lambda conn: conn.execute("SELECT step_index FROM procedure_positions WHERE owner_id = ? AND procedure_id = ?", (owner_id, str(procedure_id))).fetchone())
        return int(row["step_index"]) if row else 0

    def procedure_positions(self, owner_id: int) -> Dict[str, int]:
        """Every saved viewing position of this owner (procedure id -> step index)."""
        return self._read(lambda conn: {r["procedure_id"]: int(r["step_index"]) for r in conn.execute(
            "SELECT procedure_id, step_index FROM procedure_positions WHERE owner_id = ?", (owner_id,)).fetchall()})

    def set_procedure_position(self, owner_id: int, procedure_id: str, step_index: int) -> None:
        """Where the owner is in a guided procedure (viewing state only; never a completion)."""
        now = _db.now_iso()
        index = max(int(step_index), 0)
        self._write(lambda conn: conn.execute(
            "INSERT INTO procedure_positions (owner_id, procedure_id, step_index, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(owner_id, procedure_id) DO UPDATE SET step_index = excluded.step_index, updated_at = excluded.updated_at",
            (owner_id, str(procedure_id)[:64], index, now)))

    def delete_older_than(self, days: int) -> int:
        """Retention: delete conversations (all owners) not updated for ``days`` days. Returns how many were removed."""
        if days <= 0:
            raise StoreError("days must be positive")
        import datetime as _dt
        cutoff = (_dt.datetime.utcnow() - _dt.timedelta(days=int(days))).strftime("%Y-%m-%dT%H:%M:%SZ")
        return int(self._write(lambda conn: conn.execute("DELETE FROM conversations WHERE updated_at < ?", (cutoff,)).rowcount))

    def counts(self) -> Dict[str, int]:
        return self._read(lambda conn: {t: int(conn.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0]) for t in (
            "owners", "preferences", "conversations", "messages", "answers", "evidence_items", "onboarding_tasks", "task_progress", "progress_events", "procedure_positions")})


_STORE: Optional[Store] = None
_STORE_LOCK = threading.Lock()


def get_store(path: Optional[str] = None) -> Store:
    """The process-wide store (created on first use for ``APP_DB_PATH``)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None or (path and os.path.abspath(path) != _STORE.path):
            _STORE = Store(path)
        return _STORE


def reset_store() -> None:
    global _STORE
    with _STORE_LOCK:
        _STORE = None


__all__ = ["AccessError", "MAX_CONTENT", "MAX_TITLE", "PREFERENCE_KEYS", "Store", "StoreError", "StoreUnavailable", "get_store", "reset_store"]
