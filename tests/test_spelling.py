"""Conservative spelling correction (handbook_bot.spelling) and its pipeline seam."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from handbook_bot import spelling
from handbook_bot.qa import QAResult

pytestmark = pytest.mark.skipif(not spelling.available(), reason="pyspellchecker not installed")

CORPUS = [
    "The Vice Chancellor approves conference travel requests. Human Resources telephone number +971 6 5050023.",
    "Faculty receive sick leave with full pay. Maternity leave is sixty days. Teaching load per semester.",
    "To take attendance in Blackboard open the Gradebook. Office hours. Probation period. iThenticate guidelines.",
] * 3


@pytest.fixture(scope="module")
def speller():
    return spelling.Speller(CORPUS)


@pytest.mark.parametrize("typed,expected", [
    ("who aproves confrence travel", "who approves conference travel"),
    ("whats the phone numbr of human resorces", "whats the phone number of human resources"),
    ("how many sick leve days", "how many sick leave days"),
    ("take attendence in blackbord", "take attendance in blackboard"),
    ("maternty leave", "maternity leave"),
    ("Teching load per semster", "Teaching load per semester"),
])
def test_clear_typos_are_corrected_towards_the_documents(speller, typed, expected):
    assert speller.correct(typed)[0] == expected


@pytest.mark.parametrize("text", [
    "is there paternity leave",            # a real English word absent from the documents: never "maternity"
    "check it in iThenticate", "what does ILHE do", "gradebook", "Who approves conference travel requests?",
    "sabbatical", "ما هو رقم هاتف الموارد البشرية",
])
def test_real_words_names_acronyms_and_arabic_are_left_alone(speller, text):
    corrected, changes = speller.correct(text)
    assert corrected == text and changes == {}


def test_the_pipeline_answers_the_corrected_question_and_marks_it(monkeypatch):
    from ui import pipeline
    seen = []

    def core(kb, q):
        seen.append(q)
        return QAResult(answer="ok"), {"served": "legacy"}

    monkeypatch.setattr(pipeline, "_ask_core", core)
    kb = SimpleNamespace(chunks=list(CORPUS), groq_client=None)
    result, diag = pipeline.ask_with_diagnostics(kb, "who aproves confrence travel")
    assert seen == ["who approves conference travel"]
    assert result.spelling["corrected"] == "who approves conference travel" and diag["spelling"] == {"aproves": "approves", "confrence": "conference"}
    result, diag = pipeline.ask_with_diagnostics(kb, "who approves conference travel")
    assert not hasattr(result, "spelling") and "spelling" not in diag


def test_spelling_correction_can_be_switched_off(monkeypatch):
    from ui import pipeline
    monkeypatch.setenv("SPELL_CORRECTION", "0")
    kb = SimpleNamespace(chunks=list(CORPUS))
    assert pipeline._correct_spelling(kb, "confrence") == ("confrence", {})
