"""Conditional exceptions a model answer leaves out (post-freeze PF-16.6, D-130). The PF-16.5 live answer to "Can an assistant
professor apply for promotion before completing five years in rank?" was a flat "No" from p146 although its own evidence held
the p152 exception. The texts below are the printed handbook sentences (pp. 146, 152, 165, 63). Fake model; no network."""
from __future__ import annotations

import pytest

from conftest import FakeGroqClient, make_items
from handbook_bot import exception_guard as X
from handbook_bot import orchestrator
from handbook_bot.agents.types import EvidenceResult
from handbook_bot.qa import REFUSAL

P146 = ("An assistant professor applying for promotion to associate professor should meet the following criteria: A. Have a minimum "
        "of five years of experience as an assistant professor, with at least one year at the University of Sharjah.")
P152 = ("Page 152 B. An assistant or associate professor may apply for promotion to a higher rank four months before completing the "
        "term at the lower rank if other conditions required for promotion to the higher rank have been fulfilled. C. If promotion "
        "procedures are completed before the end of the required term at a lower rank, the promotion to the higher rank shall be "
        "effective at the end of that period.")
P152_ROW = "B. An assistant or associate professor may apply for promotion to a higher rank four months before"   # a printed row: a fragment
P165 = "Faculty members may not act as principal investigators for two projects simultaneously, except for seed projects."
P63 = "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."
EXCEPTION_B = ("An assistant or associate professor may apply for promotion to a higher rank four months before completing the term at "
               "the lower rank if other conditions required for promotion to the higher rank have been fulfilled.")
LOSSY = "Can assistant professor promotion before five years in?"          # the PF-16.5 Arabic rewrite, verbatim
QUESTION = "Can an assistant professor apply for promotion before completing five years in rank?"
FLAT_NO = ("No; the handbook requires a minimum of five years of experience as an assistant professor (including at least one year at "
           "the University of Sharjah) before promotion to associate professor.")
CONDITIONAL = ("Yes, conditionally: the required term is five years, but an assistant professor may apply for promotion four months "
               "before completing the term at the lower rank if the other conditions required for promotion have been fulfilled.")
ITEMS = make_items((P146, 146, 6.9), (P152, 152, 5.3), (P152_ROW, 152, 4.9))


# ---- the guard ---------------------------------------------------------------------------------------------------------
def test_exception_sentences_are_whole_printed_sentences_and_fragments_are_skipped():
    found = X.exception_sentences(ITEMS)
    assert [s for s, _p in found] == [EXCEPTION_B] and found[0][1] == 152


@pytest.mark.parametrize("question", [LOSSY, QUESTION])
def test_a_flat_answer_that_drops_the_printed_exception_is_caught(question):
    assert X.uncovered(question, FLAT_NO, ITEMS) == [(EXCEPTION_B, 152)]


def test_an_answer_that_states_the_exception_is_left_alone():
    assert X.uncovered(QUESTION, CONDITIONAL, ITEMS) == []


def test_restating_the_rule_is_not_stating_its_exception():
    items = make_items((P165, 165))
    question = "Can I be principal investigator on two research projects at the same time?"
    assert X.uncovered(question, "No; faculty members may not act as principal investigators for two projects simultaneously.", items) == [(P165, 165)]
    assert X.uncovered(question, "No, except for seed projects.", items) == []


@pytest.mark.parametrize("question", [
    "How long is the faculty probation period?",                 # not a yes/no, eligibility or timing question
    "What is the phone number of the Human Resources Department?",
])
def test_questions_that_are_not_yes_no_or_eligibility_are_not_checked(question):
    assert X.uncovered(question, "One year.", make_items((P63, 63))) == []


def test_an_unrelated_exception_is_not_relevant():
    assert X.uncovered("Can faculty members apply for the bonus award in consecutive years?", "Yes.", make_items((P165, 165), (P63, 63))) == []


def test_the_note_quotes_the_printed_sentence_and_its_page():
    note = X.note([(EXCEPTION_B, 152)])
    assert note == "The approved document also states (page 152): “%s”" % EXCEPTION_B and EXCEPTION_B in P152


# ---- the legacy orchestrator -------------------------------------------------------------------------------------------
@pytest.fixture
def promotion_evidence(monkeypatch):
    def gather(question, route, **_kw):
        return (EvidenceResult(question, route, [dict(i) for i in ITEMS], 6.9, True, None, False, None), 3, None,
                {"stamped": [], "report": []})
    monkeypatch.setattr(orchestrator, "_gather_evidence", gather)
    monkeypatch.setattr(orchestrator, "answerable_clauses", lambda q: [])


def _guard(result):
    return [(e.decision, e.detail) for e in result.agent_trace if e.agent == "exception_guard"]


def test_the_retry_quotes_the_exception_and_its_answer_replaces_the_flat_one(run, promotion_evidence):
    client = FakeGroqClient(FLAT_NO + "\nPages: 146", CONDITIONAL + "\nPages: 152")
    result = run(QUESTION, client=client)
    assert client.calls == 2 and "four months before completing the term" in client.prompts[1]
    assert result.answer == CONDITIONAL and 152 in result.pages and "also states" not in result.answer
    assert _guard(result)[0][0] == "retry_kept"


def test_an_answer_that_still_drops_it_gets_the_printed_sentence_appended(run, promotion_evidence):
    client = FakeGroqClient(FLAT_NO + "\nPages: 146", FLAT_NO + "\nPages: 146")
    result = run(QUESTION, client=client)
    assert client.calls == 2
    assert result.answer == FLAT_NO + "\n\n" + X.note([(EXCEPTION_B, 152)])
    assert 152 in result.pages and 146 in result.pages
    assert [d for d, _ in _guard(result)] == ["retry_not_kept", "note"]


def test_with_no_retry_left_the_sentence_is_appended_without_another_model_call(run, promotion_evidence):
    # the first draft is rejected by the Verifier (an invented sentence), the existing retry is used for it
    client = FakeGroqClient("Promotions are decided by a lottery.\nPages: 146", FLAT_NO + "\nPages: 146")
    result = run(QUESTION, client=client)
    assert client.calls == 2 and result.answer.startswith(FLAT_NO) and result.answer.endswith("“%s”" % EXCEPTION_B)
    assert [d for d, _ in _guard(result)] == ["note"]


def test_a_question_the_guard_does_not_apply_to_is_untouched(run, promotion_evidence):
    client = FakeGroqClient(FLAT_NO + "\nPages: 146")
    result = run("What criteria must an assistant professor meet for promotion to associate professor", client=client)
    assert client.calls == 1 and result.answer == FLAT_NO and _guard(result) == []


def test_a_refusal_gets_no_note(run, promotion_evidence):
    client = FakeGroqClient(REFUSAL)
    result = run(QUESTION, client=client)
    assert result.answer == REFUSAL and _guard(result) == []
