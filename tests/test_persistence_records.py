"""How an answer becomes rows and rows become the ``QAResult`` the page renders (post-freeze PF-13)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from handbook_bot.qa import QAResult, REFUSAL
from persistence import records


def test_jsonable_flattens_what_json_cannot_take() -> None:
    @dataclass
    class Point:
        x: int
        y: float

    value = {"set": {1, 2}, "tuple": (1, "a"), "nan": float("nan"), "inf": float("inf"), "dc": Point(1, 2.5), "nested": [{"k": (3,)}], "obj": object()}
    out = records.jsonable(value)
    assert sorted(out["set"]) == [1, 2] and out["tuple"] == [1, "a"] and out["nan"] is None and out["inf"] is None
    assert out["dc"] == {"x": 1, "y": 2.5} and out["nested"] == [{"k": [3]}] and isinstance(out["obj"], str)
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is part of the project environment
        return
    assert records.jsonable({"score": np.float32(0.5), "n": np.int64(3)}) == {"score": 0.5, "n": 3}


def test_answer_record_keeps_the_evidence_text_and_metadata_and_bounds_the_excerpt() -> None:
    long_chunk = "word " * 2000
    result = QAResult(answer="Yes (p. 3).", pages=[3, True, "x"], items=[{"chunk": long_chunk, "meta": {"page": 3, "source_id": "guide", "chunk_type": "text"}, "rerank_score": math.nan},
                                                                          "not a dict", {"text": "legacy text field", "meta": {}}])
    record = records.answer_record(result, {"served": "plan_f", "specialist_id": "teaching", "secret": "no"}, 12.0, corpus_key="k1")
    assert record["message"] == {"content": "Yes (p. 3).", "answer_language": "en", "status": "complete"}
    assert record["answer"]["pages"] == "[3]" and record["answer"]["corpus_key"] == "k1" and record["answer"]["wall_ms"] == 12.0
    assert record["answer"]["specialist"] == "teaching" and '"secret"' not in record["answer"]["diagnostics"]
    assert len(record["evidence"]) == 2
    assert len(record["evidence"][0]["excerpt"]) == records.EXCERPT_LIMIT and record["evidence"][0]["rerank_score"] is None
    assert record["evidence"][0]["page"] == 3 and record["evidence"][0]["source_id"] == "guide" and record["evidence"][0]["chunk_type"] == "text"
    assert record["evidence"][1]["excerpt"] == "legacy text field" and record["evidence"][1]["position"] == 1


def test_a_refusal_or_service_notice_is_recorded_as_a_notice() -> None:
    assert records.answer_record(QAResult(answer=REFUSAL), {}, 1.0)["message"]["status"] == "notice"
    assert records.answer_record(QAResult(answer="A real answer (p. 1)."), {}, 1.0)["message"]["status"] == "complete"


def test_restore_turn_rebuilds_defaults_when_rows_are_missing_or_partial() -> None:
    result, diagnostics, extras = records.restore_turn({"content": "Saved text", "answer_language": "ar", "created_at": "2026-11-09T10:00:00Z"}, None, [])
    assert result.answer == "Saved text" and result.answer_language == "ar" and result.pages == [] and result.items == []
    assert result.translation == {} and result.query_transform == {} and result.original_answer == ""
    assert diagnostics == {"served": None, "specialist_id": None, "plan_f_path": None, "outcome": None}
    assert extras == {"stored_at": "2026-11-09T10:00:00Z", "corpus_key": None, "wall_ms": None}


def test_restore_turn_orders_evidence_by_position_and_fills_meta_from_columns() -> None:
    rows = [{"position": 1, "excerpt": "second", "meta": "{}", "page": 7, "source_id": "s", "chunk_type": "row", "rerank_score": 0.2},
            {"position": 0, "excerpt": "first", "meta": '{"page": 5, "section_title": "Leave"}', "page": 5, "source_id": None, "chunk_type": None, "rerank_score": None}]
    result, _d, _e = records.restore_turn({"content": "x"}, {"pages": "[5, 7]", "used_llm": 1, "llm_calls": 2}, rows)
    assert [i["chunk"] for i in result.items] == ["first", "second"]
    assert result.items[0]["meta"] == {"page": 5, "section_title": "Leave"} and "rerank_score" not in result.items[0]
    assert result.items[1]["meta"] == {"page": 7, "source_id": "s", "chunk_type": "row"} and result.items[1]["rerank_score"] == 0.2
    assert result.pages == [5, 7] and result.used_llm is True and result.llm_calls == 2
