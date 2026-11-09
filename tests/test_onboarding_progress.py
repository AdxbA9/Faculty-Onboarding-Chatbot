"""Guest-owned task progress on schema version 2 (post-freeze PF-14): catalog sync, status transitions, audit events,
deterministic dashboard arithmetic (zero applicable tasks included), ownership isolation, duplicate completion,
invalid identifiers, catalog updates that remove a task, procedure positions, and the migration of a version-1
database that already holds conversations. Temporary databases only."""
from __future__ import annotations

import sqlite3

import pytest

from handbook_bot.qa import QAResult
from onboarding import catalog as C
from onboarding.progress import dashboard, status_of, valid_status
from persistence import AccessError, Store, StoreError, db


def catalog_rows():
    return [{"id": t.id, "category": t.category, "kind": t.kind, "title_en": t.title_en, "title_ar": t.title_ar, "source_id": t.evidence[0].source_id,
             "page": t.evidence[0].page, "section": t.evidence[0].section, "procedure_id": t.procedure_id, "sort_order": t.order} for t in C.tasks_by_category()]


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(str(tmp_path / "app.db"))
    s.sync_task_catalog(catalog_rows(), C.CATALOG_VERSION)
    return s


# ---------------------------------------------------------------------------
# Catalog sync
# ---------------------------------------------------------------------------
def test_catalog_sync_is_idempotent_and_deactivates_removed_tasks(store: Store) -> None:
    assert store.sync_task_catalog(catalog_rows(), C.CATALOG_VERSION) == {"inserted": 0, "updated": 0, "deactivated": 0, "active": len(C.TASKS)}
    assert store.active_task_ids() == [t.id for t in C.tasks_by_category()]
    rows = catalog_rows()
    removed = rows.pop()
    rows[0]["title_en"] = "Renamed task"
    report = store.sync_task_catalog(rows, C.CATALOG_VERSION + 1)
    assert report == {"inserted": 0, "updated": len(rows), "deactivated": 1, "active": len(rows)}   # every row carries the new version
    assert removed["id"] not in store.active_task_ids()
    store.sync_task_catalog(catalog_rows(), C.CATALOG_VERSION)                   # the task comes back active, nothing lost
    assert removed["id"] in store.active_task_ids()


def test_progress_on_a_removed_task_is_kept_but_not_counted(store: Store) -> None:
    owner = store.owner_for_token("a")
    rows = catalog_rows()
    removed = rows[-1]["id"]
    store.set_task_status(owner, removed, "completed")
    store.sync_task_catalog(rows[:-1], C.CATALOG_VERSION + 1)
    assert removed not in store.task_progress(owner)                              # not reported for a deactivated task
    conn = db.connect(store.path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM task_progress WHERE task_id = ?", (removed,)).fetchone()[0] == 1   # but never deleted
    finally:
        conn.close()
    with pytest.raises(AccessError):
        store.set_task_status(owner, removed, "in_progress")                     # and no longer writable


# ---------------------------------------------------------------------------
# Status transitions and audit
# ---------------------------------------------------------------------------
def test_status_changes_are_explicit_atomic_and_audited(store: Store) -> None:
    owner = store.owner_for_token("a")
    assert store.task_progress(owner) == {}
    first = store.set_task_status(owner, "bb_access", "in_progress")
    assert first["changed"] is True and first["previous"] == "not_started"
    done = store.set_task_status(owner, "bb_access", "completed")
    assert done["changed"] is True and done["previous"] == "in_progress"
    row = store.task_progress(owner)["bb_access"]
    assert row["status"] == "completed" and row["completed_at"] and row["completed_at"] == row["updated_at"]
    again = store.set_task_status(owner, "bb_access", "completed")             # a duplicate completion
    assert again["changed"] is False and again["updated_at"] is None
    assert store.task_progress(owner)["bb_access"]["completed_at"] == row["completed_at"]
    undo = store.set_task_status(owner, "bb_access", "not_started")           # completion can be withdrawn
    assert undo["changed"] is True and store.task_progress(owner)["bb_access"]["completed_at"] is None
    events = store.progress_events(owner)
    assert [(e["from_status"], e["to_status"]) for e in events] == [("completed", "not_started"), ("in_progress", "completed"), (None, "in_progress")]
    assert all(e["origin"] == "user" for e in events)


def test_invalid_statuses_tasks_and_origins_are_refused(store: Store) -> None:
    owner = store.owner_for_token("a")
    with pytest.raises(StoreError):
        store.set_task_status(owner, "bb_access", "done")
    with pytest.raises(AccessError):
        store.set_task_status(owner, "no_such_task", "completed")
    with pytest.raises(AccessError):
        store.set_task_status(owner, "bb_access'; DROP TABLE task_progress; --", "completed")
    with pytest.raises(StoreError):
        store.set_task_status(owner, "bb_access", "completed", origin="assistant")   # the model never writes progress
    assert store.task_progress(owner) == {} and store.progress_events(owner) == []


def test_progress_is_isolated_between_owners(store: Store) -> None:
    a, b = store.owner_for_token("a"), store.owner_for_token("b")
    store.set_task_status(a, "hr_documents", "completed")
    assert store.task_progress(b) == {} and store.progress_events(b) == []
    store.set_task_status(b, "hr_documents", "in_progress")
    assert store.task_progress(a)["hr_documents"]["status"] == "completed"
    assert store.task_progress(b)["hr_documents"]["status"] == "in_progress"
    conn = db.connect(store.path)
    try:
        conn.execute("DELETE FROM owners WHERE id = ?", (b,))                     # cascades: progress, events, positions
        assert conn.execute("SELECT COUNT(*) FROM task_progress").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM progress_events").fetchone()[0] == 1
    finally:
        conn.close()


def test_concurrent_status_updates_keep_one_row_per_task(store: Store) -> None:
    import threading
    owner = store.owner_for_token("a")
    errors = []

    def worker(n):
        try:
            for i in range(10):
                store.set_task_status(owner, "bb_announcement", ("in_progress", "completed")[(n + i) % 2])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    conn = db.connect(store.path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM task_progress WHERE owner_id = ?", (owner,)).fetchone()[0] == 1
        events = conn.execute("SELECT from_status, to_status FROM progress_events WHERE owner_id = ? ORDER BY id", (owner,)).fetchall()
    finally:
        conn.close()
    assert all(e[0] != e[1] for e in events)                                       # every event is a real change


# ---------------------------------------------------------------------------
# Procedure positions
# ---------------------------------------------------------------------------
def test_procedure_positions_are_viewing_state_only(store: Store) -> None:
    owner = store.owner_for_token("a")
    assert store.procedure_position(owner, "bb_announcement") == 0
    store.set_procedure_position(owner, "bb_announcement", 5)
    store.set_procedure_position(owner, "bb_announcement", 3)
    assert store.procedure_position(owner, "bb_announcement") == 3
    store.set_procedure_position(owner, "bb_announcement", -4)
    assert store.procedure_position(owner, "bb_announcement") == 0
    assert store.task_progress(owner) == {}                                       # viewing never completes anything
    other = store.owner_for_token("b")
    assert store.procedure_position(other, "bb_announcement") == 0


# ---------------------------------------------------------------------------
# Dashboard arithmetic
# ---------------------------------------------------------------------------
def test_dashboard_counts_percentage_breakdown_and_next_activities() -> None:
    tasks = C.tasks_by_category()
    empty = dashboard({}, tasks)
    assert empty["total"] == len(tasks) and empty["applicable"] == len(tasks) and empty["completed"] == 0 and empty["percent"] == 0
    assert empty["remaining"] == len(tasks) and empty["all_done"] is False
    assert empty["next"][0] in ("hr_documents", "ilhe_training")                 # required tasks first
    progress = {"hr_documents": {"status": "completed"}, "ilhe_training": {"status": "in_progress"}, "grade_transfer": {"status": "not_applicable"},
                "bb_attendance": {"status": "not_applicable"}}
    d = dashboard(progress, tasks)
    assert d["applicable"] == len(tasks) - 2 and d["completed"] == 1 and d["in_progress"] == 1 and d["not_applicable"] == 2
    assert d["remaining"] == len(tasks) - 3 and d["percent"] == round(100 / (len(tasks) - 2))
    assert d["next"][0] == "ilhe_training"                                        # an in-progress task comes before untouched ones
    assert d["by_category"]["teaching"]["applicable"] == d["by_category"]["teaching"]["total"] - 2
    assert d["by_category"]["faculty_services"]["completed"] == 1


def test_dashboard_with_zero_applicable_tasks_is_zero_percent_not_an_error() -> None:
    tasks = C.tasks_by_category()
    d = dashboard({t.id: {"status": "not_applicable"} for t in tasks}, tasks)
    assert d["applicable"] == 0 and d["percent"] == 0 and d["remaining"] == 0 and d["next"] == [] and d["all_done"] is False
    assert dashboard({}, []) == {"total": 0, "applicable": 0, "completed": 0, "in_progress": 0, "not_started": 0, "not_applicable": 0, "remaining": 0,
                                 "percent": 0, "by_category": {c: {"total": 0, "applicable": 0, "completed": 0, "in_progress": 0} for c in C.CATEGORIES},
                                 "next": [], "all_done": False}


def test_all_tasks_completed_reads_one_hundred_percent() -> None:
    tasks = C.tasks_by_category()
    d = dashboard({t.id: {"status": "completed"} for t in tasks}, tasks)
    assert d["percent"] == 100 and d["remaining"] == 0 and d["all_done"] is True and d["next"] == []


def test_status_helpers_default_and_validate() -> None:
    assert status_of({}, "bb_access") == "not_started" and status_of({"bb_access": {"status": "bogus"}}, "bb_access") == "not_started"
    assert valid_status("completed") and not valid_status("done") and not valid_status(None)


# ---------------------------------------------------------------------------
# Migration of a version-1 database with conversations
# ---------------------------------------------------------------------------
def _version_one_database(path: str) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("BEGIN")
    for statement in db._statements(db.MIGRATIONS[1]):
        conn.execute(statement)
    conn.execute("PRAGMA user_version = 1")
    conn.execute("COMMIT")
    conn.close()


def test_a_version_one_database_with_conversations_upgrades_with_everything_intact(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    _version_one_database(path)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO owners (id, token_hash, kind, created_at, last_seen_at) VALUES (1, 'h1', 'guest', '2026-11-08T20:00:00Z', '2026-11-08T20:00:00Z')")
    conn.execute("INSERT INTO preferences (owner_id, interface_lang, theme, extra, updated_at) VALUES (1, 'ar', 'dark', '{\"active_conversation\": \"c1\"}', '2026-11-08T20:00:00Z')")
    conn.execute("INSERT INTO conversations (id, owner_id, title, status, interface_lang, created_at, updated_at) VALUES ('c1', 1, 'سياسة الحضور', 'active', 'ar', '2026-11-08T20:00:00Z', '2026-11-08T20:05:00Z')")
    conn.execute("INSERT INTO messages (id, conversation_id, seq, role, content, answer_language, status, client_token, created_at) VALUES (1, 'c1', 1, 'user', 'ما هي سياسة الحضور؟', 'auto', 'complete', 't1', '2026-11-08T20:01:00Z')")
    conn.execute("INSERT INTO messages (id, conversation_id, seq, role, content, answer_language, status, client_token, created_at) VALUES (2, 'c1', 2, 'assistant', 'الإجابة (ص. 12)', 'ar', 'complete', 't1:a', '2026-11-08T20:02:00Z')")
    conn.execute("INSERT INTO answers (message_id, served, pages, original_answer, translation, diagnostics) VALUES (2, 'plan_f', '[12]', 'The answer (p. 12)', '{\"status\": \"translated\"}', '{\"served\": \"plan_f\"}')")
    conn.execute("INSERT INTO evidence_items (message_id, position, source_id, page, excerpt, meta) VALUES (2, 0, 'uos_faculty_handbook_2025_26', 12, 'Students must attend', '{\"page\": 12}')")
    conn.commit()
    conn.close()

    report = db.initialize(path)
    assert report["from_version"] == 1 and report["migrated"] == [2] and report["backup"] and report["backup"].endswith(".bak")
    store = Store(path)
    owner = store.find_owner  # noqa: F841  (the API is unchanged)
    assert store.get_preferences(1) == {"interface_lang": "ar", "theme": "dark", "updated_at": "2026-11-08T20:00:00Z", "extra": {"active_conversation": "c1"}}
    conversations = store.list_conversations(1)
    assert [c["title"] for c in conversations] == ["سياسة الحضور"] and conversations[0]["message_count"] == 2 and conversations[0]["context_task"] is None
    turns = store.load_messages(1, "c1")
    assert turns[0]["content"] == "ما هي سياسة الحضور؟" and turns[1]["result"].answer == "الإجابة (ص. 12)"
    assert turns[1]["result"].original_answer == "The answer (p. 12)" and turns[1]["result"].translation == {"status": "translated"}
    assert turns[1]["result"].items[0]["chunk"] == "Students must attend" and turns[1]["result"].pages == [12]
    # the new tables are empty and usable; the old data is untouched by the catalog sync
    assert store.counts()["task_progress"] == 0 and store.counts()["onboarding_tasks"] == 0
    store.sync_task_catalog(catalog_rows(), C.CATALOG_VERSION)
    store.set_task_status(1, "bb_access", "completed")
    assert store.task_progress(1)["bb_access"]["status"] == "completed"
    conv = store.create_conversation(1, "About Blackboard", "en", context_task="bb_access")
    assert store.get_conversation(1, conv["id"])["context_task"] == "bb_access"
    assert len(store.list_conversations(1)) == 2 and store.load_messages(1, "c1")[1]["result"].answer == "الإجابة (ص. 12)"
    # the pre-migration backup is the version-1 file
    probe = sqlite3.connect(report["backup"])
    try:
        assert probe.execute("PRAGMA user_version").fetchone()[0] == 1
        assert probe.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        assert "task_progress" not in {r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        probe.close()


def test_the_store_opens_a_version_one_file_and_migrates_it_once(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    _version_one_database(path)
    first = Store(path)
    assert first.init_report["migrated"] == [2] and first.init_report["backup"] is None   # empty: no backup needed
    second = Store(path)
    assert second.init_report["migrated"] == [] and db.list_backups(path) == []


def test_a_conversation_context_task_round_trips_and_is_bounded(tmp_path) -> None:
    store = Store(str(tmp_path / "app.db"))
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x", "en", context_task="x" * 200)
    assert len(store.get_conversation(owner, conv["id"])["context_task"]) == 64
    plain = store.create_conversation(owner, "y", "en")
    assert store.get_conversation(owner, plain["id"])["context_task"] is None
    result = QAResult(answer="ok (p. 1)", pages=[1])
    store.append_user_message(owner, conv["id"], "q")
    store.append_assistant_message(owner, conv["id"], result, {}, 1.0)
    assert store.list_conversations(owner)[0]["context_task"] == "x" * 64
