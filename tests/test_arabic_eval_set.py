"""The PF-11 multilingual evaluation set is well formed and anchored in the frozen English set (offline)."""
from __future__ import annotations

import importlib.util
import pathlib

from handbook_bot.arabic_text import has_arabic

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AR = _load("pf11_question_set", ROOT / "qa" / "pf11_arabic" / "question_set.py")
I8 = _load("i8_question_set", ROOT / "qa" / "plan_f_i8" / "question_set.py")
FROZEN = {q["id"]: q for q in I8.QUESTIONS}


def test_ids_are_unique_and_every_entry_is_complete():
    ids = [q["id"] for q in AR.ALL]
    assert len(ids) == len(set(ids)) and len(AR.QUESTIONS) >= 40 and len(AR.HOLDOUT) >= 10
    for q in AR.ALL:
        assert has_arabic(q["ar"]) and q["en"] and q["expect"] in ("serve", "fallback", "either"), q["id"]
        assert q["split"] in ("dev", "holdout") and q["source"] and isinstance(q["specialists"], list)
        assert q["pages"] is None or isinstance(q["pages"], list)
        assert q["expect"] != "fallback" or not q["pages"], q["id"]          # a fallback case claims no expected page


def test_paired_questions_keep_the_frozen_english_wording_and_expectation():
    for q in AR.ALL:
        frozen = FROZEN.get(q["pair"])
        if frozen is None:
            assert q["pair"] in ("PF10-EX", "MIX"), q["id"]
            continue
        assert q["en"] == frozen["q"], q["id"]
        assert q["expect"] == frozen["expect"], q["id"]


def test_the_set_covers_every_specialist_and_the_required_categories():
    domains = {q["domain"] for q in AR.ALL}
    assert {"teaching", "research", "faculty_services", "institutional", "cross"} <= domains
    categories = {q["category"] for q in AR.ALL}
    assert {"contact", "numeric", "multipart", "cross_domain", "adversarial", "unsupported", "mixed", "negation", "procedure"} <= categories
    assert sum(1 for q in AR.ALL if q["mixed"]) >= 3 and sum(1 for q in AR.ALL if q["expect"] == "fallback") >= 8


def test_holdout_questions_do_not_repeat_dev_wording():
    dev = {q["ar"] for q in AR.QUESTIONS}
    assert all(q["ar"] not in dev for q in AR.HOLDOUT)
