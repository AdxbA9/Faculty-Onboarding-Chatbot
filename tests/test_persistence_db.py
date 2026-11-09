"""The SQLite file layer (post-freeze PF-13): schema creation, versioned migrations that back up first and roll back on
failure, refusal of newer or damaged files, foreign-key cascades, online backup and restore, the integrity check and the
command line. Every test works on a temporary database; nothing here touches ``var/``."""
from __future__ import annotations

import os
import sqlite3

import pytest

from persistence import db, manage
from persistence.store import Store


V1_TABLES = ("owners", "preferences", "conversations", "messages", "answers", "evidence_items")


def _tables(path: str) -> set:
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")}
    finally:
        conn.close()


def _populate(path: str) -> Store:
    store = Store(path)
    owner = store.owner_for_token("browser-one")
    conv = store.create_conversation(owner, "Attendance", "en")
    store.append_user_message(owner, conv["id"], "What is the attendance policy?", client_token="q1")
    return store


# ---------------------------------------------------------------------------
# Creation and connection settings
# ---------------------------------------------------------------------------
def test_initialize_creates_every_table_at_the_current_schema_version(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    report = db.initialize(path)
    assert report["created"] is True and report["migrated"] == list(range(1, db.SCHEMA_VERSION + 1)) and report["to_version"] == db.SCHEMA_VERSION
    assert _tables(path) >= {"owners", "preferences", "conversations", "messages", "answers", "evidence_items"}        # version 1
    assert _tables(path) >= {"onboarding_tasks", "task_progress", "progress_events", "procedure_positions"}            # version 2 (PF-14)
    conn = db.connect(path)
    try:
        assert db.schema_version(conn) == db.SCHEMA_VERSION
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_initialize_creates_the_parent_directory_and_is_idempotent(tmp_path) -> None:
    path = str(tmp_path / "nested" / "deeper" / "app.db")
    db.initialize(path)
    second = db.initialize(path)
    assert second["created"] is False and second["migrated"] == [] and second["from_version"] == db.SCHEMA_VERSION


def test_db_path_prefers_the_argument_then_the_environment_then_the_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("APP_DB_PATH", raising=False)
    assert db.db_path().endswith(os.path.join("var", "app.db"))
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "env.db"))
    assert db.db_path() == str(tmp_path / "env.db")
    assert db.db_path(str(tmp_path / "arg.db")) == str(tmp_path / "arg.db")


# ---------------------------------------------------------------------------
# Refusals: never destroy what is there
# ---------------------------------------------------------------------------
def test_a_database_written_by_a_newer_application_is_refused_not_downgraded(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    _populate(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = %d" % (db.SCHEMA_VERSION + 5))
    conn.commit()
    conn.close()
    with pytest.raises(db.SchemaTooNew):
        db.initialize(path)
    with pytest.raises(db.SchemaTooNew):
        Store(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION + 5     # untouched
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()


def test_a_file_that_is_not_a_database_is_reported_and_left_alone(tmp_path) -> None:
    path = tmp_path / "app.db"
    path.write_bytes(b"definitely not sqlite " * 64)
    before = path.read_bytes()
    with pytest.raises(db.IntegrityProblem):
        db.initialize(str(path))
    assert path.read_bytes() == before
    assert manage.main(["--db", str(path), "check"]) == 2                     # the command line reports it, no traceback


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------
def test_a_migration_backs_up_a_database_with_data_first_and_keeps_the_data(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "app.db")
    _populate(path)
    current = db.SCHEMA_VERSION
    monkeypatch.setattr(db, "SCHEMA_VERSION", current + 1)
    monkeypatch.setitem(db.MIGRATIONS, current + 1, "ALTER TABLE conversations ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;")
    report = db.initialize(path)
    assert report["from_version"] == current and report["migrated"] == [current + 1]
    assert report["backup"] and os.path.exists(report["backup"]) and report["backup"].endswith(db.BACKUP_SUFFIX)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == current + 1
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
        assert "pinned" in [r[1] for r in conn.execute("PRAGMA table_info(conversations)")]
    finally:
        conn.close()
    backup_conn = sqlite3.connect(report["backup"])
    try:
        assert backup_conn.execute("PRAGMA user_version").fetchone()[0] == current  # the backup is the pre-migration state
        assert backup_conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        backup_conn.close()


def test_an_empty_database_migrates_without_a_backup(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "app.db")
    db.initialize(path)
    current = db.SCHEMA_VERSION
    monkeypatch.setattr(db, "SCHEMA_VERSION", current + 1)
    monkeypatch.setitem(db.MIGRATIONS, current + 1, "CREATE TABLE later (id INTEGER PRIMARY KEY);")
    report = db.initialize(path)
    assert report["migrated"] == [current + 1] and report["backup"] is None and "later" in _tables(path)


def test_a_failing_migration_rolls_back_completely(tmp_path, monkeypatch) -> None:
    path = str(tmp_path / "app.db")
    _populate(path)
    current = db.SCHEMA_VERSION
    monkeypatch.setattr(db, "SCHEMA_VERSION", current + 1)
    monkeypatch.setitem(db.MIGRATIONS, current + 1, "CREATE TABLE half_done (id INTEGER PRIMARY KEY); INSERT INTO no_such_table VALUES (1);")
    with pytest.raises(sqlite3.OperationalError):
        db.initialize(path)
    assert "half_done" not in _tables(path)
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == current
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()


def test_migration_scripts_are_split_into_statements_without_losing_any(tmp_path) -> None:
    statements = list(db._statements(db.MIGRATIONS[1]))
    assert len(statements) == 8 and all(s.upper().startswith(("CREATE TABLE", "CREATE INDEX")) for s in statements)
    statements = list(db._statements(db.MIGRATIONS[2]))
    assert len(statements) == 7 and all(s.upper().startswith(("CREATE TABLE", "CREATE INDEX", "ALTER TABLE")) for s in statements)


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------
def test_deleting_an_owner_cascades_through_every_dependent_table(tmp_path) -> None:
    from handbook_bot.qa import QAResult
    path = str(tmp_path / "app.db")
    store = Store(path)
    owner = store.owner_for_token("browser-one")
    store.set_preferences(owner, theme="dark")
    conv = store.create_conversation(owner, "Attendance", "en")
    store.append_user_message(owner, conv["id"], "What is the attendance policy?")
    result = QAResult(answer="Attend 75% of classes (p. 12).", pages=[12], items=[{"chunk": "Students must attend 75%.", "meta": {"page": 12}}])
    store.append_assistant_message(owner, conv["id"], result, {"served": "plan_f"}, 10.0)
    assert {k: v for k, v in store.counts().items() if k in V1_TABLES} == {"owners": 1, "preferences": 1, "conversations": 1, "messages": 2, "answers": 1, "evidence_items": 1}
    conn = db.connect(path)
    try:
        conn.execute("DELETE FROM owners WHERE id = ?", (owner,))
    finally:
        conn.close()
    assert {k: v for k, v in store.counts().items() if k in V1_TABLES} == {"owners": 0, "preferences": 0, "conversations": 0, "messages": 0, "answers": 0, "evidence_items": 0}


def test_an_orphan_row_cannot_be_inserted(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    db.initialize(path)
    conn = db.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO conversations (id, owner_id, created_at, updated_at) VALUES ('c1', 999, 'now', 'now')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO messages (conversation_id, seq, role, content, created_at) VALUES ('ghost', 1, 'user', 'hi', 'now')")
        with pytest.raises(sqlite3.IntegrityError):                                   # the CHECK constraints hold too
            conn.execute("INSERT INTO owners (token_hash, kind, created_at, last_seen_at) VALUES ('h', 'admin', 'now', 'now')")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Backup, restore, check
# ---------------------------------------------------------------------------
def test_backup_restore_round_trip_brings_deleted_data_back_and_keeps_the_replaced_file(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    store = _populate(path)
    owner = store.owner_for_token("browser-one")
    backup_path = db.backup(path, str(tmp_path / "backups" / "app.bak"))
    assert os.path.exists(backup_path)
    probe = sqlite3.connect(backup_path)
    try:
        assert probe.execute("PRAGMA journal_mode").fetchone()[0] == "delete"      # a single self-contained file
        assert probe.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        probe.close()
    for conv in store.list_conversations(owner):
        store.delete_conversation(owner, conv["id"])
    assert store.counts()["messages"] == 0
    report = db.restore(backup_path, path)
    assert report["restored_from"] == os.path.abspath(backup_path)
    assert report["previous_kept_as"] and os.path.exists(report["previous_kept_as"])
    assert Store(path).counts()["messages"] == 1
    assert db.list_backups(path) == [report["previous_kept_as"]]


def test_restore_refuses_a_damaged_or_newer_source(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    _populate(path)
    bad = tmp_path / "bad.bak"
    bad.write_bytes(b"garbage" * 100)
    with pytest.raises(db.IntegrityProblem):
        db.restore(str(bad), path)
    with pytest.raises(db.DatabaseError):
        db.restore(str(tmp_path / "missing.bak"), path)
    assert manage.main(["--db", path, "restore", str(bad), "--yes"]) == 2
    newer = str(tmp_path / "newer.bak")
    db.backup(path, newer)
    conn = sqlite3.connect(newer)
    conn.execute("PRAGMA user_version = 42")
    conn.commit()
    conn.close()
    with pytest.raises(db.SchemaTooNew):
        db.restore(newer, path)
    assert Store(path).counts()["messages"] == 1                                  # the live database was never touched


def test_check_reports_version_integrity_counts_and_journal_mode(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    assert db.check(path) == {"path": os.path.abspath(path), "exists": False}
    _populate(path)
    report = db.check(path)
    assert report["schema_version"] == db.SCHEMA_VERSION and report["integrity"] == "ok" and report["foreign_key_violations"] == 0
    assert report["counts"]["messages"] == 1 and report["journal_mode"] == "wal" and report["size_bytes"] > 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def test_the_command_line_initialises_checks_backs_up_and_guards_restore(tmp_path, capsys) -> None:
    path = str(tmp_path / "app.db")
    assert manage.main(["--db", path, "init"]) == 0
    assert manage.main(["--db", path, "check"]) == 0
    assert manage.main(["--db", path, "info"]) == 0
    dest = str(tmp_path / "backups" / "copy.bak")
    assert manage.main(["--db", path, "backup", dest]) == 0 and os.path.exists(dest)
    assert manage.main(["--db", path, "restore", dest]) == 1                 # without --yes nothing happens
    assert manage.main(["--db", path, "retention", "30"]) == 1
    assert manage.main(["--db", path, "restore", dest, "--yes"]) == 0
    assert manage.main(["--db", path, "retention", "30", "--yes"]) == 0
    out = capsys.readouterr()
    assert "removed_conversations" in out.out and "token" not in out.out.lower()
