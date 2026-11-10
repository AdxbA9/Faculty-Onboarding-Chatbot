"""Contact-column guard (post-freeze PF-16, D-127): a model-written telephone answer must not be a fax number, and the
other way round. Synthetic directory pages that copy the printed layouts of handbook pages 53 and 54; no model, no
network."""
from __future__ import annotations

import pytest

from handbook_bot import contact_guard as G
from handbook_bot import orchestrator
from handbook_bot.qa import REFUSAL

#: The header split over three lines, a labelled multi-line entry (telephones "Men:"/"Women:", fax numbers after them,
#: joined by "/"), a complete row, a telephone split over two lines, and an "(External)" value.
PAGES = [
    {"page": 52, "lines": ["Page 52", "Some text before the directory +971 6 5050999"]},
    {"page": 53, "lines": ["Page 53", "Office / College / Department", "Telephone Number", "Fax Number",
                           "Dental Hospital Call Center", "+971 6 5057666 (External) Ext 7666 (Internal)",
                           "College of Engineering", "Men: +971 6 5050901", "Women: +971 6 5050952", "+971 6 5585173 /", "+971 6 5585191"]},
    {"page": 54, "lines": ["Page 54", "Office / College / Department", "Telephone Number", "Fax Number",
                           "Human Resources", "+971 6 5050023", "+971 6 5585200",
                           "Center for Continuing Education and", "Professional Development (CCE)", "+971 6 5057081 / +971 6", "5057087",
                           "+971 6 5057078"]},
    {"page": 14, "text": "a page without lines gives no columns"},
]


def test_every_number_is_assigned_to_its_printed_column():
    table = G.directory_columns(PAGES)
    tel, fax = {"telephone"}, {"fax"}
    expected = {"97165050901": tel, "97165050952": tel, "97165585173": fax, "97165585191": fax, "97165057666": tel,
                "97165050023": tel, "97165585200": fax, "97165057081": tel, "97165057087": tel, "97165057078": fax}
    assert {k: table.get(k) for k in expected} == expected
    assert "97165050999" not in table                       # text before the header is not the directory


@pytest.mark.parametrize("question, answer, wrong", [
    ("What is the telephone number of the College of Engineering?", "The telephone number is +971 6 5585173.", True),
    ("What is the telephone number of the College of Engineering?", "Men: +971 6 5050901, Women: +971 6 5050952.", False),
    ("What is the phone number of Human Resources?", "Telephone +971 6 5050023, fax +971 6 5585200.", False),   # both given: fine
    ("What is the fax number of Human Resources?", "The fax number is +971 6 5050023.", True),
    ("What is the fax number of Human Resources?", "The fax number is +971 6 5585200.", False),
    ("How can I reach Human Resources?", "+971 6 5585200", False),                                # no single column asked
    ("What is the phone number of CCE?", "+971 6 5057081 / +971 6 5057087", False),
    ("What is the phone number of CCE?", "+971 6 5057078", True),
    ("What is the phone number of the Dental Hospital Call Center?", "+971 6 5057666", False),
    ("What is the phone number of HR?", "Call +971 6 5050777.", False),                            # not in the directory: the Verifier's job
])
def test_an_answer_from_the_other_column_is_a_problem(question, answer, wrong):
    assert bool(G.answer_problem(question, answer, PAGES)) is wrong


def test_the_legacy_path_refuses_a_fax_number_given_as_a_telephone(run, monkeypatch):
    # The extractor and (since PF-16.5) the directory reader are switched off so the answer is model-written, as on the
    # live path of the frozen question I14 before PF-16.5 and for any unit neither of them can bind.
    from conftest import FakeGroqClient
    monkeypatch.setattr(orchestrator, "extract_contact_answer", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator.contact_directory, "answer", lambda *a, **k: None)
    pages = [{"page": 14, "lines": ["Page 14", "Office / College / Department", "Telephone Number", "Fax Number",
                                    "Information Technology Center", "+971 6 5050000", "+971 6 5585101"]}]
    question = "What is the telephone number of the Information Technology Center?"
    wrong = run(question, client=FakeGroqClient("The telephone number of the Information Technology Center is +971 6 5585101.\nPages: 14"), pages=pages)
    assert wrong.answer == REFUSAL and any(e.agent == "contact_guard" and e.decision == "answer_refused" for e in wrong.agent_trace)
    right = run(question, client=FakeGroqClient("The telephone number of the Information Technology Center is +971 6 5050000.\nPages: 14"), pages=pages)
    assert "+971 6 5050000" in right.answer and not any(e.agent == "contact_guard" for e in right.agent_trace)
