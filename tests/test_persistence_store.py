"""The owner-scoped persistence service (post-freeze PF-13): identity hashing, preferences, conversations, messages with
their answer record and evidence, faithful restoration (English, Arabic, translation state, diagnostics), owner isolation,
idempotent appends, concurrency, interrupted writes, malformed rows and an unavailable or locked database. Temporary
databases only."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

import pytest

from handbook_bot.qa import QAResult, REFUSAL
from persistence import AccessError, Store, StoreError, StoreUnavailable, db, get_store, records, reset_store

ARABIC_QUESTION = "ما هي سياسة الحضور؟"
ARABIC_ANSWER = "تنص السياسة على حضور 75% من المحاضرات (ص. 12)."
ENGLISH_ANSWER = "The policy requires attending 75% of classes (p. 12)."
ITEMS = [
    {"chunk": "Students must attend at least 75% of classes.", "rerank_score": 0.91,
     "meta": {"page": 12, "source_id": "uos_faculty_handbook_2025_26", "source_title": "UOS Faculty Handbook 2025/26", "section_title": "Attendance", "chunk_type": "text"}},
    {"chunk": "Absence beyond the limit leads to a failing grade.", "rerank_score": 0.52, "meta": {"page": 13, "source_id": "uos_faculty_handbook_2025_26"}},
]
DIAGNOSTICS = {"mode": "active", "served": "plan_f", "attempted": True, "specialist_id": "teaching", "plan_f_path": "single_finding_passthrough",
               "plan_f_pages": [12, 13], "routing_final": ["teaching"], "plan_f_llm_calls": 0, "question_text": "must not be persisted"}


def arabic_result() -> QAResult:
    return QAResult(answer=ARABIC_ANSWER, pages=[12, 13], best_section="Attendance", evidence="Students must attend...", query_type="policy",
                    items=ITEMS, used_llm=False, llm_calls=0, retried=False, answer_language="ar", original_answer=ENGLISH_ANSWER,
                    translation={"status": "checked", "reasons": [], "method": "deterministic"},
                    query_transform={"english": "What is the attendance policy?", "method": "lexicon", "coverage": 1.0})


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(str(tmp_path / "app.db"))


# ---------------------------------------------------------------------------
# Owners
# ---------------------------------------------------------------------------
def test_the_same_browser_identity_maps_to_the_same_owner_and_only_its_hash_is_stored(store: Store) -> None:
    first = store.owner_for_token("session-id-abc")
    assert store.owner_for_token("session-id-abc") == first
    assert store.owner_for_token("session-id-xyz") != first
    conn = db.connect(store.path)
    try:
        rows = conn.execute("SELECT token_hash, kind FROM owners ORDER BY id").fetchall()
    finally:
        conn.close()
    assert rows[0]["kind"] == "guest"
    assert rows[0]["token_hash"] == hashlib.sha256(b"fob-owner:session-id-abc").hexdigest()
    assert "session-id-abc" not in json.dumps([dict(r) for r in rows])


def test_find_owner_reads_without_creating(store: Store) -> None:
    assert store.find_owner("never-seen") is None and store.counts()["owners"] == 0
    created = store.owner_for_token("never-seen")
    assert store.find_owner("never-seen") == created and store.find_owner("") is None


def test_an_empty_identity_is_refused(store: Store) -> None:
    with pytest.raises(AccessError):
        store.owner_for_token("")
    with pytest.raises(AccessError):
        store.owner_for_token(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------
def test_preferences_start_empty_update_partially_reject_unknown_keys_and_reset(store: Store) -> None:
    owner = store.owner_for_token("a")
    assert store.get_preferences(owner) == {}
    saved = store.set_preferences(owner, interface_lang="ar", theme="dark")
    assert saved["interface_lang"] == "ar" and saved["theme"] == "dark" and "answer_lang" not in saved
    saved = store.set_preferences(owner, answer_lang="en")
    assert saved["interface_lang"] == "ar" and saved["answer_lang"] == "en"           # the earlier choices stay
    store.set_preferences(owner, theme="")
    assert "theme" not in store.get_preferences(owner)                                # empty clears a preference
    with pytest.raises(StoreError):
        store.set_preferences(owner, is_admin="yes")
    store.reset_preferences(owner)
    assert store.get_preferences(owner) == {}


def test_extra_values_merge_and_a_none_removes_one(store: Store) -> None:
    owner = store.owner_for_token("a")
    assert store.set_extra(owner, active_conversation="c1") == {"active_conversation": "c1"}
    assert store.set_extra(owner, sidebar="collapsed") == {"active_conversation": "c1", "sidebar": "collapsed"}
    assert store.set_extra(owner, active_conversation=None) == {"sidebar": "collapsed"}
    assert store.get_preferences(owner)["extra"] == {"sidebar": "collapsed"}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
def test_conversations_list_most_recently_written_first_with_message_counts(store: Store) -> None:
    owner = store.owner_for_token("a")
    first = store.create_conversation(owner, "First", "en")
    second = store.create_conversation(owner, "Second", "en")
    assert [c["title"] for c in store.list_conversations(owner)] == ["Second", "First"]
    conn = db.connect(store.path)                                  # make the first one newer by writing to it
    conn.execute("UPDATE conversations SET updated_at = '2099-01-01T00:00:00Z' WHERE id = ?", (first["id"],))
    conn.close()
    listed = store.list_conversations(owner)
    assert [c["title"] for c in listed] == ["First", "Second"] and listed[0]["message_count"] == 0
    store.append_user_message(owner, second["id"], "hello")
    assert store.list_conversations(owner)[1]["message_count"] == 1
    assert store.get_conversation(owner, first["id"])["interface_lang"] == "en"
    assert len(first["id"]) >= 20 and first["id"] != second["id"]                 # opaque, unpredictable identifiers


def test_rename_normalises_whitespace_limits_length_and_refuses_an_empty_title(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    assert store.rename_conversation(owner, conv["id"], "  Attendance   policy  ")["title"] == "Attendance policy"
    assert len(store.rename_conversation(owner, conv["id"], "y" * 500)["title"]) == 120
    with pytest.raises(StoreError):
        store.rename_conversation(owner, conv["id"], "   ")


def test_delete_removes_the_conversation_and_everything_under_it(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    store.append_user_message(owner, conv["id"], ARABIC_QUESTION)
    store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 12.0)
    assert store.counts()["evidence_items"] == 2
    assert store.delete_conversation(owner, conv["id"]) is True
    assert store.delete_conversation(owner, conv["id"]) is False
    counts = store.counts()
    assert {k: counts[k] for k in ("owners", "preferences", "conversations", "messages", "answers", "evidence_items")} == {
        "owners": 1, "preferences": 0, "conversations": 0, "messages": 0, "answers": 0, "evidence_items": 0}


def test_delete_all_is_scoped_to_one_owner(store: Store) -> None:
    a, b = store.owner_for_token("a"), store.owner_for_token("b")
    store.create_conversation(a, "a1")
    store.create_conversation(a, "a2")
    store.create_conversation(b, "b1")
    assert store.delete_all_conversations(a) == 2
    assert store.list_conversations(a) == [] and [c["title"] for c in store.list_conversations(b)] == ["b1"]


# ---------------------------------------------------------------------------
# Messages and faithful restoration
# ---------------------------------------------------------------------------
def test_an_arabic_turn_is_restored_exactly_with_its_english_source_translation_state_evidence_and_diagnostics(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, ARABIC_QUESTION, "ar")
    store.append_user_message(owner, conv["id"], ARABIC_QUESTION, answer_language="auto", client_token="t1")
    store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 345.6, client_token="t1:a", corpus_key="corpus-v1")
    turns = store.load_messages(owner, conv["id"])
    assert [m["role"] for m in turns] == ["user", "assistant"]
    assert turns[0]["content"] == ARABIC_QUESTION and turns[0]["result"] is None
    restored: QAResult = turns[1]["result"]
    original = arabic_result()
    assert restored.answer == original.answer and restored.pages == original.pages and restored.best_section == original.best_section
    assert restored.evidence == original.evidence and restored.query_type == original.query_type
    assert restored.answer_language == "ar" and restored.original_answer == ENGLISH_ANSWER
    assert restored.translation == original.translation and restored.query_transform == original.query_transform
    assert restored.used_llm is False and restored.llm_calls == 0 and restored.retried is False
    assert len(restored.items) == 2
    assert restored.items[0]["chunk"] == ITEMS[0]["chunk"] and restored.items[0]["meta"] == ITEMS[0]["meta"] and restored.items[0]["rerank_score"] == 0.91
    assert restored.items[1]["meta"]["page"] == 13
    assert turns[1]["content"] == ARABIC_ANSWER and turns[1]["stored"] is True and turns[1]["corpus_key"] == "corpus-v1"
    assert turns[1]["wall_ms"] == 345.6 and turns[1]["stored_at"].endswith("Z") and len(turns[1]["stored_at"]) == 24
    diagnostics = turns[1]["diagnostics"]
    assert diagnostics["served"] == "plan_f" and diagnostics["specialist_id"] == "teaching" and diagnostics["routing_final"] == ["teaching"]
    assert diagnostics["plan_f_pages"] == [12, 13]
    assert "question_text" not in diagnostics                                     # only the allow-listed keys are kept


def test_the_page_renderers_accept_a_restored_turn(store: Store) -> None:
    from ui import presentation
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    store.append_user_message(owner, conv["id"], ARABIC_QUESTION)
    store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 1.0)
    turn = store.load_messages(owner, conv["id"])[1]
    chips = presentation.source_chips(turn["result"])
    assert any("12" in label for _css, label in chips)
    rows = presentation.evidence_rows(turn["result"])
    assert rows and rows[0]["page"] == 12
    assert presentation.served_by_plan_f(turn["diagnostics"]) is True
    sections = presentation.dev_sections(turn["result"], turn["diagnostics"], {}, turn["wall_ms"])
    assert sections


def test_sequence_numbers_are_contiguous_and_a_repeated_client_token_stores_nothing_twice(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    first = store.append_user_message(owner, conv["id"], "q1", client_token="tok-1")
    assert store.append_user_message(owner, conv["id"], "q1 again, same callback", client_token="tok-1") == first
    second = store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 1.0, client_token="tok-1:a")
    assert store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 1.0, client_token="tok-1:a") == second
    store.append_user_message(owner, conv["id"], "q2")
    conn = db.connect(store.path)
    try:
        assert [r[0] for r in conn.execute("SELECT seq FROM messages WHERE conversation_id = ? ORDER BY seq", (conv["id"],))] == [1, 2, 3]
    finally:
        conn.close()
    assert store.counts()["messages"] == 3 and store.counts()["answers"] == 1


def test_an_empty_question_is_refused_and_long_content_is_bounded(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    with pytest.raises(StoreError):
        store.append_user_message(owner, conv["id"], "   ")
    store.append_user_message(owner, conv["id"], "x" * 50000)
    assert len(store.load_messages(owner, conv["id"])[0]["content"]) == 20000


def test_a_refusal_is_stored_as_a_notice_and_restored_without_evidence(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    store.append_user_message(owner, conv["id"], "Who won the world cup?")
    store.append_assistant_message(owner, conv["id"], QAResult(answer=REFUSAL, pages=[], items=ITEMS), {"served": "plan_f", "outcome": "refused"}, 1.0)
    conn = db.connect(store.path)
    try:
        assert conn.execute("SELECT status FROM messages WHERE role = 'assistant'").fetchone()[0] == "notice"
    finally:
        conn.close()
    turn = store.load_messages(owner, conv["id"])[1]
    assert turn["result"].answer == REFUSAL and turn["result"].pages == [] and len(turn["result"].items) == 2


def test_sql_in_user_text_is_stored_literally(store: Store) -> None:
    owner = store.owner_for_token("a")
    hostile = "'); DROP TABLE messages; --"
    conv = store.create_conversation(owner, hostile)
    store.append_user_message(owner, conv["id"], hostile)
    assert store.rename_conversation(owner, conv["id"], hostile)["title"] == hostile
    assert store.load_messages(owner, conv["id"])[0]["content"] == hostile
    assert store.counts()["messages"] == 1


def test_export_contains_only_this_conversation_with_its_rows(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    store.append_user_message(owner, conv["id"], ARABIC_QUESTION)
    store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 1.0)
    other = store.create_conversation(owner, "other")
    store.append_user_message(owner, other["id"], "elsewhere")
    export = store.export_conversation(owner, conv["id"])
    assert export["conversation"]["id"] == conv["id"] and len(export["messages"]) == 2
    assert export["messages"][1]["answer"]["original_answer"] == ENGLISH_ANSWER and len(export["messages"][1]["evidence"]) == 2
    json.dumps(export, ensure_ascii=False)                                        # JSON-able as promised


# ---------------------------------------------------------------------------
# Owner isolation
# ---------------------------------------------------------------------------
def test_another_owner_cannot_see_open_rename_delete_or_write_to_a_conversation(store: Store) -> None:
    a, b = store.owner_for_token("a"), store.owner_for_token("b")
    conv = store.create_conversation(a, "private")
    store.append_user_message(a, conv["id"], ARABIC_QUESTION)
    assert store.list_conversations(b) == []
    assert store.get_conversation(b, conv["id"]) is None
    with pytest.raises(AccessError):
        store.load_messages(b, conv["id"])
    with pytest.raises(AccessError):
        store.rename_conversation(b, conv["id"], "hijacked")
    with pytest.raises(AccessError):
        store.append_user_message(b, conv["id"], "injected")
    with pytest.raises(AccessError):
        store.append_assistant_message(b, conv["id"], arabic_result(), DIAGNOSTICS, 1.0)
    with pytest.raises(AccessError):
        store.export_conversation(b, conv["id"])
    assert store.delete_conversation(b, conv["id"]) is False
    assert store.get_conversation(a, conv["id"])["title"] == "private" and len(store.load_messages(a, conv["id"])) == 1


def test_a_forged_or_missing_identifier_looks_exactly_like_a_foreign_one(store: Store) -> None:
    a = store.owner_for_token("a")
    b = store.owner_for_token("b")
    conv = store.create_conversation(a, "private")
    for probe in ("no-such-id", "", "1 OR 1=1", conv["id"][:-1]):
        assert store.get_conversation(b, probe) is None and store.get_conversation(a, probe) is None
        with pytest.raises(AccessError) as foreign:
            store.load_messages(b, conv["id"])
        with pytest.raises(AccessError) as missing:
            store.load_messages(a, probe)
        assert str(foreign.value) == str(missing.value)                             # no oracle between "forbidden" and "absent"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_a_locked_database_raises_store_unavailable_instead_of_hanging_or_pretending(store: Store, monkeypatch) -> None:
    owner = store.owner_for_token("a")
    locker = sqlite3.connect(store.path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    original = db.connect
    monkeypatch.setattr(db, "connect", lambda path, timeout=5.0: original(path, timeout=0.2))
    try:
        with pytest.raises(StoreUnavailable):
            store.create_conversation(owner, "while locked")
    finally:
        locker.execute("ROLLBACK")
        locker.close()
    monkeypatch.setattr(db, "connect", original)
    assert store.list_conversations(owner) == []                                  # nothing half-written, and the store recovers
    assert store.create_conversation(owner, "after the lock")["title"] == "after the lock"


def test_an_interrupted_answer_save_leaves_no_partial_turn(store: Store, monkeypatch) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    store.append_user_message(owner, conv["id"], "q")
    real = records.answer_record

    def broken(result, diagnostics, wall_ms, corpus_key=None):
        record = real(result, diagnostics, wall_ms, corpus_key)
        record["evidence"][1]["position"] = record["evidence"][0]["position"]     # violates UNIQUE(message_id, position) mid-transaction
        return record
    monkeypatch.setattr("persistence.store.answer_record", broken)
    with pytest.raises(StoreUnavailable):
        store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 1.0)
    assert store.counts()["messages"] == 1 and store.counts()["answers"] == 0 and store.counts()["evidence_items"] == 0
    assert [m["role"] for m in store.load_messages(owner, conv["id"])] == ["user"]


def test_malformed_stored_json_is_tolerated_on_load(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    store.append_user_message(owner, conv["id"], "q")
    store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 1.0)
    conn = db.connect(store.path)
    try:
        conn.execute("UPDATE answers SET translation = 'not json', query_transform = '[1,2', diagnostics = '{\"served\": ', pages = 'oops'")
        conn.execute("UPDATE evidence_items SET meta = '{bad json'")
    finally:
        conn.close()
    turn = store.load_messages(owner, conv["id"])[1]
    assert turn["result"].answer == ARABIC_ANSWER and turn["result"].translation == {} and turn["result"].pages == []
    assert turn["result"].items[0]["meta"]["page"] == 12                             # the columns stand in for the broken JSON
    assert turn["diagnostics"]["served"] == "plan_f"


def test_an_assistant_message_without_an_answer_row_still_restores(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    store.append_user_message(owner, conv["id"], "q")
    mid = store.append_assistant_message(owner, conv["id"], arabic_result(), DIAGNOSTICS, 1.0)
    conn = db.connect(store.path)
    try:
        conn.execute("DELETE FROM evidence_items WHERE message_id = ?", (mid,))
        conn.execute("DELETE FROM answers WHERE message_id = ?", (mid,))
    finally:
        conn.close()
    turn = store.load_messages(owner, conv["id"])[1]
    assert turn["result"].answer == ARABIC_ANSWER and turn["result"].items == [] and turn["result"].answer_language == "ar"


def test_an_unusable_location_fails_at_construction(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(Exception):
        Store(str(blocker / "sub" / "app.db"))


def test_concurrent_appends_from_many_threads_all_land_with_unique_sequence_numbers(store: Store) -> None:
    owner = store.owner_for_token("a")
    conv = store.create_conversation(owner, "x")
    errors = []

    def worker(n: int) -> None:
        try:
            for i in range(5):
                store.append_user_message(owner, conv["id"], "thread %d message %d" % (n, i))
        except Exception as exc:  # pragma: no cover - reported through the assertion below
            errors.append(exc)
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    turns = store.load_messages(owner, conv["id"])
    assert len(turns) == 40
    conn = db.connect(store.path)
    try:
        seqs = [r[0] for r in conn.execute("SELECT seq FROM messages WHERE conversation_id = ? ORDER BY seq", (conv["id"],))]
    finally:
        conn.close()
    assert seqs == list(range(1, 41))


def test_retention_deletes_only_conversations_older_than_the_cutoff(store: Store) -> None:
    owner = store.owner_for_token("a")
    old = store.create_conversation(owner, "old")
    store.create_conversation(owner, "fresh")
    conn = db.connect(store.path)
    conn.execute("UPDATE conversations SET updated_at = '2020-01-01T00:00:00Z' WHERE id = ?", (old["id"],))
    conn.close()
    assert store.delete_older_than(30) == 1
    assert [c["title"] for c in store.list_conversations(owner)] == ["fresh"]
    with pytest.raises(StoreError):
        store.delete_older_than(0)


# ---------------------------------------------------------------------------
# Process-wide store
# ---------------------------------------------------------------------------
def test_get_store_follows_the_configured_path_and_reset_forgets_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "one.db"))
    reset_store()
    first = get_store()
    assert first is get_store() and first.path == str(tmp_path / "one.db")
    other = get_store(str(tmp_path / "two.db"))
    assert other is not first and other.path == str(tmp_path / "two.db")
    reset_store()
    assert get_store() is not other
    reset_store()
