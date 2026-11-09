"""
SQLite file, schema versions, migrations, integrity, backup and restore (post-freeze PF-13).

Why SQLite: a single-machine application served on localhost with a handful of concurrent browsers; transactions,
foreign keys and the online backup API are in the standard library; no server, no driver, no container. Its limits are
stated in docs/DECISIONS.md D-124 (one writer at a time, one machine, no row-level permissions): a multi-host deployment
with real accounts is the point at which a server database would replace this module behind the same ``Store`` API.

Connection settings (every connection): ``foreign_keys = ON`` (SQLite leaves it off by default), ``journal_mode = WAL``
(readers do not block the writer, the page never freezes on a write), ``busy_timeout`` (a locked database waits
instead of failing at once), ``synchronous = NORMAL`` (safe with WAL).

Migrations: ``PRAGMA user_version`` holds the schema version; ``MIGRATIONS`` maps each version to the SQL that reaches
it from the previous one, applied inside one transaction each. When a database that already holds data is behind the
code's version, a file backup is taken first (``<db>.before-v<N>.<timestamp>.bak``); a database AHEAD of the code is
refused (``SchemaTooNew``), never downgraded or recreated. Nothing here ever drops user data.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 3
DEFAULT_DB_PATH = os.path.join("var", "app.db")
#: Where a backup taken before a migration goes (next to the database).
BACKUP_SUFFIX = ".bak"


class DatabaseError(RuntimeError):
    """A database problem the caller must report (never swallowed into a fake success)."""


class SchemaTooNew(DatabaseError):
    """The file was written by a newer version of the application."""


class IntegrityProblem(DatabaseError):
    """``PRAGMA integrity_check`` found damage."""


MIGRATIONS: Dict[int, str] = {
    1: """
    CREATE TABLE owners (
        id            INTEGER PRIMARY KEY,
        token_hash    TEXT    NOT NULL UNIQUE,
        kind          TEXT    NOT NULL DEFAULT 'guest' CHECK (kind IN ('guest', 'account')),
        created_at    TEXT    NOT NULL,
        last_seen_at  TEXT    NOT NULL
    );
    CREATE TABLE preferences (
        owner_id        INTEGER PRIMARY KEY REFERENCES owners(id) ON DELETE CASCADE,
        interface_lang  TEXT,
        answer_lang     TEXT,
        theme           TEXT,
        voice_lang      TEXT,
        extra           TEXT    NOT NULL DEFAULT '{}',
        updated_at      TEXT    NOT NULL
    );
    CREATE TABLE conversations (
        id              TEXT    PRIMARY KEY,
        owner_id        INTEGER NOT NULL REFERENCES owners(id) ON DELETE CASCADE,
        title           TEXT    NOT NULL DEFAULT '',
        status          TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
        interface_lang  TEXT,
        created_at      TEXT    NOT NULL,
        updated_at      TEXT    NOT NULL
    );
    CREATE INDEX conversations_owner_updated ON conversations(owner_id, updated_at DESC);
    CREATE TABLE messages (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id  TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        seq              INTEGER NOT NULL,
        role             TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
        content          TEXT    NOT NULL,
        answer_language  TEXT,
        status           TEXT    NOT NULL DEFAULT 'complete' CHECK (status IN ('complete', 'error', 'notice')),
        client_token     TEXT,
        created_at       TEXT    NOT NULL,
        UNIQUE (conversation_id, seq),
        UNIQUE (conversation_id, client_token)
    );
    CREATE INDEX messages_conversation_seq ON messages(conversation_id, seq);
    CREATE TABLE answers (
        message_id        INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
        served            TEXT,
        outcome           TEXT,
        specialist        TEXT,
        plan_f_path       TEXT,
        query_type        TEXT,
        pages             TEXT    NOT NULL DEFAULT '[]',
        best_section      TEXT,
        evidence_preview  TEXT,
        used_llm          INTEGER NOT NULL DEFAULT 0,
        llm_calls         INTEGER NOT NULL DEFAULT 0,
        retried           INTEGER NOT NULL DEFAULT 0,
        original_answer   TEXT,
        translation       TEXT    NOT NULL DEFAULT '{}',
        query_transform   TEXT    NOT NULL DEFAULT '{}',
        diagnostics       TEXT    NOT NULL DEFAULT '{}',
        corpus_key        TEXT,
        wall_ms           REAL
    );
    CREATE TABLE evidence_items (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id    INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        position      INTEGER NOT NULL,
        source_id     TEXT,
        page          INTEGER,
        section       TEXT,
        chunk_type    TEXT,
        excerpt       TEXT    NOT NULL,
        meta          TEXT    NOT NULL DEFAULT '{}',
        rerank_score  REAL,
        UNIQUE (message_id, position)
    );
    """,
    # ---- version 2 (post-freeze PF-14): the onboarding checklist ---------------------------------------------------------
    # The curated task catalog is mirrored into ``onboarding_tasks`` (idempotent sync at start; a task removed from the
    # catalog is deactivated, never deleted, so existing progress keeps its foreign key); progress is guest-owned, one row
    # per (owner, task); every status change is an audit event; a procedure's viewing position is separate from the task
    # status; a conversation may carry the task it was asked about.
    2: """
    CREATE TABLE onboarding_tasks (
        id               TEXT    PRIMARY KEY,
        catalog_version  INTEGER NOT NULL,
        category         TEXT    NOT NULL,
        kind             TEXT    NOT NULL,
        title_en         TEXT    NOT NULL,
        title_ar         TEXT    NOT NULL,
        source_id        TEXT    NOT NULL,
        page             INTEGER,
        section          TEXT,
        procedure_id     TEXT,
        sort_order       INTEGER NOT NULL DEFAULT 0,
        active           INTEGER NOT NULL DEFAULT 1,
        updated_at       TEXT    NOT NULL
    );
    CREATE TABLE task_progress (
        owner_id      INTEGER NOT NULL REFERENCES owners(id) ON DELETE CASCADE,
        task_id       TEXT    NOT NULL REFERENCES onboarding_tasks(id) ON DELETE CASCADE,
        status        TEXT    NOT NULL DEFAULT 'not_started' CHECK (status IN ('not_started', 'in_progress', 'completed', 'not_applicable')),
        updated_at    TEXT    NOT NULL,
        completed_at  TEXT,
        PRIMARY KEY (owner_id, task_id)
    );
    CREATE INDEX task_progress_owner_status ON task_progress(owner_id, status);
    CREATE TABLE progress_events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_id     INTEGER NOT NULL REFERENCES owners(id) ON DELETE CASCADE,
        task_id      TEXT    NOT NULL,
        from_status  TEXT,
        to_status    TEXT    NOT NULL,
        origin       TEXT    NOT NULL DEFAULT 'user',
        created_at   TEXT    NOT NULL
    );
    CREATE INDEX progress_events_owner_time ON progress_events(owner_id, created_at);
    CREATE TABLE procedure_positions (
        owner_id      INTEGER NOT NULL REFERENCES owners(id) ON DELETE CASCADE,
        procedure_id  TEXT    NOT NULL,
        step_index    INTEGER NOT NULL DEFAULT 0,
        updated_at    TEXT    NOT NULL,
        PRIMARY KEY (owner_id, procedure_id)
    );
    ALTER TABLE conversations ADD COLUMN context_task TEXT;
    """,
    # ---- version 3 (post-freeze PF-15): settings page and guest profile ---------------------------------------------------
    # Two accessibility preferences that change the page (text size, reduced motion) and an optional display name shown only
    # to the same browser. Columns, not JSON, so the store can validate them like the other preferences.
    3: """
    ALTER TABLE preferences ADD COLUMN text_size TEXT;
    ALTER TABLE preferences ADD COLUMN reduced_motion TEXT;
    ALTER TABLE preferences ADD COLUMN display_name TEXT;
    """,
}


def db_path(path: Optional[str] = None) -> str:
    """The database file: the argument, else ``APP_DB_PATH``, else ``var/app.db`` under the current directory."""
    return os.path.abspath(path or os.environ.get("APP_DB_PATH", "").strip() or DEFAULT_DB_PATH)


def now_iso() -> str:
    """UTC with milliseconds (``2026-11-09T10:12:13.456Z``): conversations written in the same second still order."""
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + ".%03dZ" % int((now - int(now)) * 1000)


def connect(path: str, *, timeout: float = 5.0) -> sqlite3.Connection:
    """A configured connection (the caller closes it)."""
    conn = sqlite3.connect(path, timeout=timeout, isolation_level=None)   # no implicit transactions: explicit BEGIN/COMMIT only
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = %d" % int(timeout * 1000))
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            pass                                               # read-only media: keep the default journal
        conn.execute("PRAGMA synchronous = NORMAL")
    except BaseException:
        conn.close()                                           # PF-16 (D-127): a damaged file must not leave an open handle behind
        raise
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _has_data(conn: sqlite3.Connection) -> bool:
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
    for table in tables:
        if conn.execute('SELECT 1 FROM "%s" LIMIT 1' % table.replace('"', '""')).fetchone():
            return True
    return False


def file_backup(path: str, label: str) -> str:
    """Copy the database file (with its WAL checkpointed) next to itself. Returns the backup path."""
    dest = "%s.%s.%s%s" % (path, label, time.strftime("%Y%m%d-%H%M%S"), BACKUP_SUFFIX)
    backup(path, dest)
    return dest


def raw_file_copy(path: str, label: str) -> str:
    """Byte-for-byte copy of a database file that SQLite cannot read, with its ``-wal``/``-shm`` files beside it, so that
    nothing is lost even when the online backup API refuses the file (post-freeze PF-16). Returns the copy's path."""
    dest = "%s.%s.%s%s" % (path, label, time.strftime("%Y%m%d-%H%M%S"), BACKUP_SUFFIX)
    shutil.copyfile(path, dest)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(path + suffix):
            shutil.copyfile(path + suffix, dest + suffix)
    return dest


def initialize(path: Optional[str] = None) -> Dict[str, Any]:
    """Create or upgrade the database at ``path``. Returns what happened (created, migrated from/to, backup path).

    Never destroys data: a database with data behind the code's version is backed up to a file first; a database ahead of
    the code raises :class:`SchemaTooNew`; an integrity problem in an existing file raises :class:`IntegrityProblem`.
    """
    target = db_path(path)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    existed = os.path.exists(target) and os.path.getsize(target) > 0
    report: Dict[str, Any] = {"path": target, "existed": existed, "created": not existed, "from_version": None,
                              "to_version": SCHEMA_VERSION, "backup": None, "migrated": []}
    try:
        conn = connect(target)
    except sqlite3.DatabaseError as exc:
        raise IntegrityProblem("not a readable database: %s" % exc) from exc
    try:
        if existed:
            try:
                problem = integrity_problem(conn)
            except sqlite3.DatabaseError as exc:                 # "file is not a database" and friends
                raise IntegrityProblem("not a readable database: %s" % exc) from exc
            if problem:
                raise IntegrityProblem("integrity_check: %s" % problem)
        current = schema_version(conn)
        report["from_version"] = current
        if current > SCHEMA_VERSION:
            raise SchemaTooNew("database schema version %d is newer than this application's %d" % (current, SCHEMA_VERSION))
        if current < SCHEMA_VERSION and existed and _has_data(conn):
            conn.close()
            report["backup"] = file_backup(target, "before-v%d" % SCHEMA_VERSION)
            conn = connect(target)
        for version in range(current + 1, SCHEMA_VERSION + 1):
            conn.execute("BEGIN IMMEDIATE")                       # one transaction per version: all of it or none of it
            try:
                for statement in _statements(MIGRATIONS[version]):
                    conn.execute(statement)
                conn.execute("PRAGMA user_version = %d" % version)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            report["migrated"].append(version)
        conn.execute("PRAGMA foreign_key_check")
    finally:
        conn.close()
    return report


def _statements(sql: str) -> Iterable[str]:
    """The statements of a migration script (``executescript`` would commit the surrounding transaction first)."""
    for part in sql.split(";"):
        text = part.strip()
        if text:
            yield text


def integrity_problem(conn: sqlite3.Connection) -> Optional[str]:
    """None when ``PRAGMA integrity_check`` reports ok, else its first message."""
    row = conn.execute("PRAGMA integrity_check").fetchone()
    text = str(row[0]) if row else "no result"
    return None if text == "ok" else text


def check(path: Optional[str] = None) -> Dict[str, Any]:
    """Integrity and foreign-key check plus row counts, without changing anything."""
    target = db_path(path)
    if not os.path.exists(target):
        return {"path": target, "exists": False}
    conn = connect(target)
    try:
        counts = {}
        for table in ("owners", "preferences", "conversations", "messages", "answers", "evidence_items", "onboarding_tasks", "task_progress", "progress_events", "procedure_positions"):
            try:
                counts[table] = int(conn.execute('SELECT COUNT(*) FROM "%s"' % table).fetchone()[0])
            except sqlite3.DatabaseError:
                counts[table] = None
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
        return {"path": target, "exists": True, "schema_version": schema_version(conn), "integrity": integrity_problem(conn) or "ok",
                "foreign_key_violations": len(fk), "counts": counts, "size_bytes": os.path.getsize(target),
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0]}
    finally:
        conn.close()


def backup(path: str, dest: str) -> str:
    """Consistent copy of the live database through SQLite's online backup API (safe while the app runs)."""
    source = connect(db_path(path))
    try:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
            target.execute("PRAGMA journal_mode = DELETE")        # a single self-contained file
        finally:
            target.close()
    finally:
        source.close()
    return dest


def restore(src: str, path: Optional[str] = None) -> Dict[str, Any]:
    """Replace the database with ``src`` after checking the source's integrity and schema; the current file is kept
    as ``<db>.before-restore.<timestamp>.bak``. Callers must stop the application first (open connections keep the old
    file)."""
    target = db_path(path)
    if not os.path.exists(src):
        raise DatabaseError("no such backup: %s" % src)
    probe = sqlite3.connect(src)
    try:
        try:
            problem = integrity_problem(probe)
        except sqlite3.DatabaseError as exc:
            raise IntegrityProblem("the backup is not a readable database: %s" % exc) from exc
        if problem:
            raise IntegrityProblem("the backup fails integrity_check: %s" % problem)
        version = schema_version(probe)
        if version > SCHEMA_VERSION:
            raise SchemaTooNew("the backup's schema version %d is newer than this application's %d" % (version, SCHEMA_VERSION))
    finally:
        probe.close()
    kept = None
    if os.path.exists(target) and os.path.getsize(target) > 0:
        try:
            kept = file_backup(target, "before-restore")
        except sqlite3.DatabaseError:
            # PF-16 (D-127): a damaged database cannot go through the online backup API, and a damaged database is the
            # main reason to restore. It is still kept, byte for byte (with its journal files), before it is replaced.
            kept = raw_file_copy(target, "before-restore-damaged")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("-wal", "-shm"):
        if os.path.exists(target + suffix):
            os.remove(target + suffix)
    shutil.copyfile(src, target)
    report = initialize(target)                                  # upgrades an older backup to the current schema
    report["restored_from"] = os.path.abspath(src)
    report["previous_kept_as"] = kept
    return report


def list_backups(path: Optional[str] = None) -> List[str]:
    target = db_path(path)
    folder = Path(target).parent
    name = Path(target).name
    return sorted(str(p) for p in folder.glob(name + ".*" + BACKUP_SUFFIX))


__all__ = ["BACKUP_SUFFIX", "DEFAULT_DB_PATH", "DatabaseError", "IntegrityProblem", "MIGRATIONS", "SCHEMA_VERSION", "SchemaTooNew", "backup", "check",
           "connect", "db_path", "file_backup", "initialize", "integrity_problem", "list_backups", "now_iso", "restore", "schema_version"]
