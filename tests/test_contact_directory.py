"""Contact directory reader (post-freeze PF-16.5, D-128): every number bound to its unit and its column, on the printed lines
of handbook pages 52-55 (fixture copied verbatim from the frozen corpus). No model, no network."""
from __future__ import annotations

import json
import os
import re

import pytest

from handbook_bot import contact_directory as D
from handbook_bot import contact_guard as G
from handbook_bot import orchestrator

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "pf16_5_directory_pages.json")
with open(FIXTURE, encoding="utf-8") as _f:
    PAGES = json.load(_f)["pages"]
PAGE_DIGITS = {p["page"]: re.sub(r"\D", "", " ".join(p["lines"])) for p in PAGES}
NUMBER = re.compile(r"\+971 6 (?:\d{7}|\d{3} \d{4})")


def _entry(name):
    return next(e for e in D.parse(PAGES) if e.name == name)


def test_multi_line_labelled_entries_bind_telephones_and_fax_numbers_to_the_right_columns():
    eng = _entry("College of Engineering")
    assert [(g.label, g.column, g.numbers) for g in eng.groups] == [
        ("Men", "telephone", ["+971 6 5050901"]), ("Women", "telephone", ["+971 6 5050952"]),
        ("", "fax", ["+971 6 5585173", "+971 6 5585191"])]
    housing = _entry("Student Housing")                       # four numbers wrapped over three lines
    assert housing.groups[1].numbers == ["+971 6 5051406", "+971 6 5051401", "+971 6 5051443", "+971 6 5051414"]
    assert housing.groups[2].column == "fax" and housing.groups[2].numbers == ["+971 6 5585487", "+971 6 5585419"]
    assert _entry("College of Graduate Studies and Research").groups[0].numbers == ["+971 6 5050550"]   # name wrapped
    vcaa = _entry("Vice Chancellor for Academic Affairs")    # the telephone printed on the name line
    assert [(g.column, g.numbers) for g in vcaa.groups] == [("telephone", ["+971 6 5053032"]), ("fax", ["+971 6 5053034"])]
    libraries = _entry("Libraries")                           # sub-units, one of them with a wrapped name
    assert [g.label for g in libraries.groups if g.labelled] == ["Medical Colleges Library", "College of Fine Arts & Design Library",
                                                                 "Men’s Library", "Women’s Library"]
    assert libraries.groups[-1].column == "fax" and libraries.groups[-1].numbers == ["+971 6 5585186", "+971 6 5053167"]


@pytest.mark.parametrize("question, must, must_not, page", [
    # the PF-16 live defect: the fax was given as the telephone
    ("What is the telephone number of the College of Engineering?", ["Men: +971 6 5050901", "Women: +971 6 5050952"], ["5585173", "5585191"], 53),
    ("What is the fax number of the College of Engineering?", ["+971 6 5585173", "+971 6 5585191"], ["5050901", "5050952"], 53),
    # the PF-16 live defect: one library's number given as "the Library"
    ("What is the phone number of the Library?", ["Medical Colleges Library: +971 6 5057161", "College of Fine Arts & Design Library: +971 6 5053190",
                                                  "Men’s Library: +971 6 5050245", "Women’s Library: +971 6 5050673"], ["5585186", "5053167"], 54),
    ("What is the phone number of the Men's Library?", ["+971 6 5050245"], ["5050673", "5053190", "5057161"], 54),
    ("What is the phone number of the Women’s Library?", ["+971 6 5050673"], ["5050245"], 54),
    ("What is the fax number of the Dental Hospital Call Center?", ["lists no fax number", "+971 6 5057666"], [], 53),
    ("What is the phone number of the College of Graduate Studies and Research?", ["+971 6 5050550"], ["5050552"], 54),
    ("What is the phone number of the Human Resources Department on page 100?", ["+971 6 5050023"], ["5585200"], 54),
    ("What is the phone number of the Vice Chancellor for Academic Affairs?", ["+971 6 5053032"], ["5053034"], 52),
    ("What is the phone number of CCE?", ["+971 6 5057081", "+971 6 5057087"], ["5057078"], 54),
    ("What is the phone number of the Medical Service Department?", ["Men’s Clinic: +971 6 5050207", "Women’s Clinic: +971 6 5050642"], [], 54),
    ("What is the phone number of the Central Laboratories?", ["+971 6 5053453"], ["5050444"], 54),   # no-break space
])
def test_answers_state_the_asked_column_of_the_asked_unit_with_its_page(question, must, must_not, page):
    text, pages, evidence = D.answer(question, PAGES)
    assert pages[0] == page and all(m in text for m in must) and not any(x in text for x in must_not)
    assert evidence


@pytest.mark.parametrize("question", [
    "How can I reach the Information Technology Center?",       # no unit after "of" / "for": left to the existing path
    "What is the phone number of the Department of Parking?",    # not in the directory
    "How many office hours must I keep?",                         # not a contact question
    "What is the phone number of the College?",                   # a generic word without several named sub-units
])
def test_an_uncertain_or_unknown_unit_answers_nothing(question):
    assert D.answer(question, PAGES) is None


def test_every_stated_number_is_on_the_cited_page_and_in_the_asked_column_for_every_unit():
    columns = G.directory_columns(PAGES)                    # the independent PF-16 column reader
    checked = 0
    for unit in D.units(PAGES):
        for kind, want in (("phone", "telephone"), ("fax", "fax")):
            result = D.answer("What is the %s number of the %s?" % (kind, unit.name), PAGES)
            if result is None:
                continue
            text, pages, _ = result
            for number in NUMBER.findall(text):
                digits = re.sub(r"\D", "", number)
                assert digits in PAGE_DIGITS[pages[0]]
                if "lists no" not in text:
                    assert want in columns.get(digits, set()), (unit.name, kind, number)
                checked += 1
    assert checked >= 90


def test_the_legacy_path_answers_from_the_directory_without_the_model(run, monkeypatch):
    from conftest import FakeGroqClient
    monkeypatch.setattr(orchestrator, "extract_contact_answer", lambda *a, **k: None)   # the single-line binding declines
    pages = [{"page": 14, "lines": ["Page 14", "Office / College / Department", "Telephone Number", "Fax Number",
                                    "Information Technology Center", "+971 6 5050000", "+971 6 5585101"]}]
    client = FakeGroqClient("The telephone number is +971 6 5585101.\nPages: 14")
    result = run("What is the telephone number of the Information Technology Center?", client=client, pages=pages)
    assert "+971 6 5050000" in result.answer and "5585101" not in result.answer and result.pages == [14]
    assert client.calls == 0 and not result.used_llm


@pytest.mark.parametrize("question, must", [
    # PF-16.5 offline finding: a summary sentence ("lists 4 units whose name ...") is not printed text and the unchanged
    # Verifier refused it; every directory answer must survive the real Verifier on the real page lines
    ("What is the phone number of the Library?", ["+971 6 5057161", "+971 6 5053190", "+971 6 5050245", "+971 6 5050673"]),
    ("What is the fax number of the Dental Hospital Call Center?", ["lists no fax number", "+971 6 5057666"]),
    ("What is the phone number of the Medical Service Department?", ["+971 6 5050207", "+971 6 5050642"]),
    ("What is the phone number of the Vice Chancellor for Academic Affairs?", ["+971 6 5053032"]),
])
def test_directory_answers_pass_the_unchanged_verifier(run, question, must):
    page = next(p for p in PAGES if p["page"] == D.answer(question, PAGES)[1][0])
    result = run(question, client=None, pages=[page])
    assert all(m in result.answer for m in must), result.answer
    assert not any(e.agent == "verifier" and e.decision != "accepted" for e in result.agent_trace), [
        (e.decision, e.detail) for e in result.agent_trace if e.agent == "verifier"]


def test_the_directory_lines_become_the_evidence_the_verifier_checks(run):
    # retrieval kept no Engineering cell (the fake corpus has none): the printed lines the answer rests on are its evidence
    page53 = next(p for p in PAGES if p["page"] == 53)
    result = run("What is the telephone number of the College of Engineering?", client=None, pages=[page53])
    assert "Men: +971 6 5050901" in result.answer and "Women: +971 6 5050952" in result.answer
    assert "5585173" not in result.answer and result.pages == [53]
    first = result.items[0]
    assert first["meta"]["page"] == 53 and first["chunk"].startswith("Contact directory:") and "+971 6 5050901" in first["chunk"]
