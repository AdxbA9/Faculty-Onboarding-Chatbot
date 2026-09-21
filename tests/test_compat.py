"""Backwards compatibility: the Part 1 public surface that ui/ and eval/ rely on,
plus the Pages parser the Synthesis agent uses."""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
import time
from pathlib import Path

import pytest

from handbook_bot import QAResult, answer_question
from handbook_bot.qa import (
    REFUSAL,
    legacy_fallback_pages,
    parse_answer_and_pages,
    split_answer_and_claimed_pages,
    verify_answer,
)

from conftest import GROUNDED_LEAVE, LEAVE_QUESTION, FakeGroqClient

REPO = Path(__file__).resolve().parents[1]

PART1_FIELDS = ["answer", "pages", "best_section", "evidence", "query_type", "items", "used_llm",
                "timings", "num_candidates", "num_reranked"]
PART2_FIELDS = ["agent_trace", "llm_calls", "retried", "sub_questions"]


def test_qaresult_keeps_part1_fields_and_adds_part2_fields():
    names = [f.name for f in dataclasses.fields(QAResult)]
    assert names[:len(PART1_FIELDS)] == PART1_FIELDS
    for name in PART2_FIELDS:
        assert name in names


def test_qaresult_part1_constructor_still_works():
    result = QAResult(answer="x")
    assert result.pages == [] and result.query_type == "policy"
    assert result.agent_trace == [] and result.llm_calls == 0
    assert result.retried is False and result.sub_questions == []


def test_qaresult_to_dict_is_json_serialisable(ask):
    result = ask(LEAVE_QUESTION, FakeGroqClient(GROUNDED_LEAVE))
    payload = result.to_dict()
    json.dumps(payload)
    for name in PART1_FIELDS + PART2_FIELDS:
        assert name in payload


def test_answer_question_signature_is_unchanged():
    params = inspect.signature(answer_question).parameters
    assert list(params) == ["question", "embedder", "reranker", "index", "chunks", "metadata", "pages",
                            "groq_client"]
    assert params["groq_client"].default is None


def test_ui_reads_only_fields_that_exist():
    source = (REPO / "ui" / "chat.py").read_text(encoding="utf-8")
    used = set(re.findall(r"result\.([a-zA-Z_]+)", source))
    fields = {f.name for f in dataclasses.fields(QAResult)} | {"to_dict"}
    assert used <= fields


def test_ui_pipeline_calls_the_public_api():
    source = (REPO / "ui" / "pipeline.py").read_text(encoding="utf-8")
    assert "answer_question(" in source
    for kw in ("embedder=", "reranker=", "index=", "chunks=", "metadata=", "pages=", "groq_client="):
        assert kw in source


def test_eval_runner_record_has_part1_and_part2_keys():
    source = (REPO / "eval" / "run_eval.py").read_text(encoding="utf-8")
    for key in ("answer", "pages", "query_type", "used_llm", "llm_calls", "retried", "agent_trace"):
        assert '"%s"' % key in source


# ---------------------------------------------------------------------------
# Pages parser
# ---------------------------------------------------------------------------
ITEMS = [{"chunk": "x", "meta": {"page": p}} for p in (3, 4, 12, 13, 14, 1, 272)]


@pytest.mark.parametrize("raw,expected", [
    ("The dean approves it.\nPages: 12", ("The dean approves it.", [12])),
    ("Answer here. Pages: 12-14.", ("Answer here.", [12, 13, 14])),
    ("See two pages: 3 and 4 for details. Pages: 4", ("See two pages: 3 and 4 for details.", [4])),
    ("I do not have this information.\nPages: none", (REFUSAL, [])),
    ("I do not have this information.\nPages:", (REFUSAL, [])),
    ("Answer (Pages: 5)", ("Answer", [])),                # 5 is not in the context
    ("Answer.\nPages: 1-272", ("Answer.", [1, 272])),     # a span this wide is two numbers
    ("Answer.\nPages: 3, 4 and 12", ("Answer.", [3, 4, 12])),
    ("Answer with no citation at all.", ("Answer with no citation at all.", [])),
])
def test_split_answer_and_claimed_pages(raw, expected):
    assert split_answer_and_claimed_pages(raw, ITEMS) == expected


def test_legacy_parser_keeps_part1_fallback_only_for_compatibility():
    assert parse_answer_and_pages("No citation.", ITEMS)[1] == legacy_fallback_pages(ITEMS) == [1, 3, 4]


@pytest.mark.parametrize("raw", [
    "Pages: " + "1, " * 20000 + "x",
    "pages: " * 5000,
    "Answer" + " " * 50000 + "Pages: 3",
    "Pages: " + "3-" * 20000,
    "(" * 20000 + "Pages: 3",
])
def test_parser_has_no_catastrophic_backtracking(raw):
    started = time.perf_counter()
    split_answer_and_claimed_pages(raw, ITEMS)
    assert time.perf_counter() - started < 1.0


def test_part1_verify_answer_is_reproduced():
    ctx = [{"chunk": "The phone number is 06 505 0000 and email is it@sharjah.ac.ae", "meta": {"page": 1}}]
    assert verify_answer("Call 06 505 0000.", ctx, "contact")
    assert not verify_answer("Call 06 999 9999.", ctx, "contact")
    assert verify_answer(REFUSAL, ctx, "policy")
