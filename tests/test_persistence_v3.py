"""Schema version 3 (post-freeze PF-15): preference validation, the display name rules, reset keeping the display name,
owner information, procedure positions and the migration of a version-2 database that holds conversations, progress
and preferences. Temporary databases only."""
from __future__ import annotations

import sqlite3

import pytest

from onboarding import catalog as C
from persistence import Store, StoreError, db
from persistence.store import clean_display_name, clean_preference


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(str(tmp_path / "app.db"))


def test_the_schema_is_version_three_with_the_new_preference_columns(store: Store) -> None:
    assert db.SCHEMA_VERSION == 3
    conn = db.connect(store.path)
    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(preferences)")}
    finally:
        conn.close()
    assert {"text_size", "reduced_motion", "display_name"} <= columns
    assert list(db._statements(db.MIGRATIONS[3])) == [
        "ALTER TABLE preferences ADD COLUMN text_size TEXT", "ALTER TABLE preferences ADD COLUMN reduced_motion TEXT", "ALTER TABLE preferences ADD COLUMN display_name TEXT"]


@pytest.mark.parametrize("key,good,bad", [
    ("interface_lang", "ar", "fr"), ("answer_lang", "auto", "de"), ("voice_lang", "en", "xx"), ("theme", "dark", "neon"),
    ("text_size", "large", "huge"), ("reduced_motion", "on", "yes")])
def test_every_choice_preference_accepts_its_values_and_refuses_others(store: Store, key: str, good: str, bad: str) -> None:
    owner = store.owner_for_token("a")
    assert store.set_preferences(owner, **{key: good})[key] == good
    with pytest.raises(StoreError):
        store.set_preferences(owner, **{key: bad})
    assert store.get_preferences(owner)[key] == good                          # a refused value changes nothing
    store.set_preferences(owner, **{key: None})
    assert key not in store.get_preferences(owner)


def test_a_refused_value_in_a_batch_writes_nothing(store: Store) -> None:
    owner = store.owner_for_token("a")
    with pytest.raises(StoreError):
        store.set_preferences(owner, theme="dark", text_size="enormous")
    assert store.get_preferences(owner) == {}


@pytest.mark.parametrize("raw,expected", [("  Dr.   Sara  ", "Dr. Sara"), ("سارة أحمد", "سارة أحمد"), ("", None), ("   ", None), (None, None), ("O'Neil-Smith", "O'Neil-Smith"), ("tab\there", "tab here")])
def test_display_names_are_normalised(raw, expected) -> None:
    assert clean_display_name(raw) == expected


@pytest.mark.parametrize("raw", ["x" * 41, "me@sharjah.ac.ae", "Emirates ID 784198012345671", "phone 0501234567", "<b>bold</b>", "{x}", "[x]", "back`tick",
                                 "two\nlines", "rtl‮override", "zero​width", 42])
def test_display_names_that_are_identifiers_or_markup_are_refused(raw) -> None:
    with pytest.raises(StoreError):
        clean_display_name(raw)
    with pytest.raises(StoreError):
        clean_preference("display_name", raw)


def test_reset_keeps_the_display_name_and_clears_every_setting(store: Store) -> None:
    owner = store.owner_for_token("a")
    store.set_preferences(owner, theme="dark", text_size="large", reduced_motion="on", answer_lang="ar", display_name="Sara")
    store.set_extra(owner, active_conversation="c1")
    store.reset_preferences(owner)
    prefs = store.get_preferences(owner)
    assert prefs["display_name"] == "Sara" and prefs["extra"] == {}
    assert not any(k in prefs for k in ("theme", "text_size", "reduced_motion", "answer_lang", "interface_lang", "voice_lang"))
    other = store.owner_for_token("b")
    store.set_preferences(other, theme="dark")
    store.reset_preferences(other)
    assert store.get_preferences(other) == {}                                   # without a name the row goes, as before


def test_owner_info_and_procedure_positions_are_owner_scoped(store: Store) -> None:
    a, b = store.owner_for_token("a"), store.owner_for_token("b")
    info = store.owner_info(a)
    assert info["kind"] == "guest" and info["created_at"] and info["last_seen_at"] and "token_hash" not in info
    store.set_procedure_position(a, "bb_announcement", 3)
    store.set_procedure_position(a, "bb_access", 1)
    assert store.procedure_positions(a) == {"bb_announcement": 3, "bb_access": 1}
    assert store.procedure_positions(b) == {}
    assert store.owner_info(999) == {}


def _version_two_database(path: str) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("BEGIN")
    for version in (1, 2):
        for statement in db._statements(db.MIGRATIONS[version]):
            conn.execute(statement)
    conn.execute("PRAGMA user_version = 2")
    conn.execute("COMMIT")
    conn.close()


def test_a_version_two_database_with_data_upgrades_to_three_with_everything_intact(tmp_path) -> None:
    path = str(tmp_path / "app.db")
    _version_two_database(path)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO owners (id, token_hash, kind, created_at, last_seen_at) VALUES (1, 'h1', 'guest', '2026-11-09T08:00:00Z', '2026-11-09T09:00:00Z')")
    conn.execute("INSERT INTO preferences (owner_id, interface_lang, theme, extra, updated_at) VALUES (1, 'ar', 'dark', '{\"active_conversation\": \"c1\"}', '2026-11-09T08:00:00Z')")
    conn.execute("INSERT INTO conversations (id, owner_id, title, status, interface_lang, created_at, updated_at, context_task) VALUES ('c1', 1, 'HR', 'active', 'en', '2026-11-09T08:00:00Z', '2026-11-09T08:05:00Z', 'hr_documents')")
    conn.execute("INSERT INTO messages (id, conversation_id, seq, role, content, status, created_at) VALUES (1, 'c1', 1, 'user', 'What is the HR number?', 'complete', '2026-11-09T08:01:00Z')")
    conn.execute("INSERT INTO messages (id, conversation_id, seq, role, content, answer_language, status, created_at) VALUES (2, 'c1', 2, 'assistant', 'Human Resources +971 6 5050023 (p. 54)', 'en', 'complete', '2026-11-09T08:02:00Z')")
    conn.execute("INSERT INTO answers (message_id, served, pages, diagnostics) VALUES (2, 'plan_f', '[54]', '{\"served\": \"plan_f\"}')")
    conn.execute("INSERT INTO evidence_items (message_id, position, source_id, page, excerpt, meta) VALUES (2, 0, 'uos_faculty_handbook_2025_26', 54, 'Human Resources +971 6 5050023', '{\"page\": 54}')")
    conn.execute("INSERT INTO onboarding_tasks (id, catalog_version, category, kind, title_en, title_ar, source_id, page, sort_order, active, updated_at) "
                 "VALUES ('hr_documents', 1, 'faculty_services', 'required', 't', 't', 'uos_faculty_handbook_2025_26', 61, 70, 1, '2026-11-09T08:00:00Z')")
    conn.execute("INSERT INTO task_progress (owner_id, task_id, status, updated_at, completed_at) VALUES (1, 'hr_documents', 'completed', '2026-11-09T08:10:00Z', '2026-11-09T08:10:00Z')")
    conn.execute("INSERT INTO progress_events (owner_id, task_id, from_status, to_status, origin, created_at) VALUES (1, 'hr_documents', NULL, 'completed', 'user', '2026-11-09T08:10:00Z')")
    conn.execute("INSERT INTO procedure_positions (owner_id, procedure_id, step_index, updated_at) VALUES (1, 'bb_announcement', 2, '2026-11-09T08:20:00Z')")
    conn.commit()
    conn.close()

    report = db.initialize(path)
    assert report["from_version"] == 2 and report["migrated"] == [3] and report["backup"] and report["backup"].endswith(".bak")
    store = Store(path)
    assert store.get_preferences(1) == {"interface_lang": "ar", "theme": "dark", "updated_at": "2026-11-09T08:00:00Z", "extra": {"active_conversation": "c1"}}
    turns = store.load_messages(1, "c1")
    assert [m["content"] for m in turns] == ["What is the HR number?", "Human Resources +971 6 5050023 (p. 54)"]
    assert turns[1]["result"].pages == [54] and turns[1]["result"].items[0]["chunk"] == "Human Resources +971 6 5050023"
    assert store.list_conversations(1)[0]["context_task"] == "hr_documents"
    assert store.task_progress(1)["hr_documents"]["status"] == "completed" and len(store.progress_events(1)) == 1
    assert store.procedure_positions(1) == {"bb_announcement": 2}
    store.set_preferences(1, text_size="large", display_name="Sara")             # the new columns work on the upgraded file
    assert store.get_preferences(1)["text_size"] == "large"
    probe = sqlite3.connect(report["backup"])
    try:
        assert probe.execute("PRAGMA user_version").fetchone()[0] == 2           # the backup is the version-2 file
        assert "display_name" not in {r[1] for r in probe.execute("PRAGMA table_info(preferences)")}
    finally:
        probe.close()
    # the catalog sync of a start leaves the upgraded progress alone
    rows = [{"id": t.id, "category": t.category, "kind": t.kind, "title_en": t.title_en, "title_ar": t.title_ar, "source_id": t.evidence[0].source_id,
             "page": t.evidence[0].page, "section": t.evidence[0].section, "procedure_id": t.procedure_id, "sort_order": t.order} for t in C.tasks_by_category()]
    store.sync_task_catalog(rows, C.CATALOG_VERSION)
    assert store.task_progress(1)["hr_documents"]["status"] == "completed"
