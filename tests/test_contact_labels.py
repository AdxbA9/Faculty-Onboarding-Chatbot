"""Labelled Plan F directory answers (post-freeze PF-16.6, D-129).

Plan F's Institutional specialist binds the asked fields of a complete directory row from the page's printed column header
and used to render the bare row ("Passport Unit +971 6 5050066 +971 6 5050026"), which does not say which number is the
fax. The row is now rendered with the asked field's label, every label is confirmed by the independent page-line reader
before it is shown, and the evidence item stays the verbatim printed row. Real handbook pages; no model; no network."""
from __future__ import annotations

import copy
import json
import os
import re

import pytest

import test_plan_f_runtime as T
from handbook_bot import contact_directory as D
from handbook_bot import contact_guard as G
from handbook_bot import plan_f_runtime as PF
from handbook_bot.agents.specialists import navigation_cues as N

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "pf16_5_directory_pages.json")
with open(FIXTURE, encoding="utf-8") as _f:
    PAGES = json.load(_f)["pages"]
NUMBER = re.compile(r"\+971 6 (?:\d{7}|\d{3} \d{4})")
LABELLED = re.compile(r"\b(telephone|fax) (\+971 6 (?:\d{7}|\d{3} \d{4}))")


@pytest.fixture(scope="module")
def kb():
    return T.make_kb()


@pytest.fixture(scope="module")
def runtime(kb):
    runtime, build = PF.build_runtime(kb)
    assert build.ok and runtime is not None, build.errors
    return runtime


def _column(number):
    return G.directory_columns(PAGES).get(re.sub(r"\D", "", number), set())


# ---- the renderer -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("row, values, expected", [
    ("Human Resources +971 6 5050023 +971 6 5585200", {"telephone": "+971 6 5050023"},
     "Human Resources (contact directory): telephone +971 6 5050023."),
    ("Passport Unit +971 6 5050066 +971 6 5050026", {"fax": "+971 6 5050026"},
     "Passport Unit (contact directory): fax +971 6 5050026."),
    # both fields asked: in the order asked, each with its label
    ("Passport Unit +971 6 5050066 +971 6 5050026", {"fax": "+971 6 5050026", "telephone": "+971 6 5050066"},
     "Passport Unit (contact directory): fax +971 6 5050026; telephone +971 6 5050066."),
    # two fax numbers printed in one cell
    ("Deanship of Student Affairs +971 6 5050759 +971 6 5585174 / +971 6 5585160", {"fax": "+971 6 5585174 / +971 6 5585160"},
     "Deanship of Student Affairs (contact directory): fax +971 6 5585174 / +971 6 5585160."),
    # narrow no-break and no-break spaces in the printed row and in the value
    ("Central Laboratories +971 6 5053453 +971 6 5050444", {"telephone": "+971 6 5053453"},
     "Central Laboratories (contact directory): telephone +971 6 5053453."),
])
def test_a_bound_row_is_rendered_with_the_asked_fields_labels(row, values, expected):
    assert D.bound_row_answer(row, values) == expected


@pytest.mark.parametrize("row, values", [
    ("Human Resources +971 6 5050023 +971 6 5585200", {"telephone": "+971 6 5050099"}),     # value not printed in the row
    ("Human Resources +971 6 5050023 +971 6 5585200", {"extension": "1234"}),               # no telephone/fax label for it
    ("Human Resources +971 6 5050023 +971 6 5585200", {}),                                  # nothing bound
    ("+971 6 5050023 +971 6 5585200", {"telephone": "+971 6 5050023"}),                     # no printed name
])
def test_a_row_that_cannot_be_labelled_from_its_own_text_is_not_rendered(row, values):
    assert D.bound_row_answer(row, values) is None


# ---- the independent column check ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("values, ok", [
    ({"telephone": "+971 6 5050023"}, True),      # Human Resources telephone
    ({"fax": "+971 6 5585200"}, True),            # Human Resources fax
    ({"telephone": "+971 6 5585200"}, False),     # the fax given as a telephone
    ({"fax": "+971 6 5050066"}, False),           # Passport Unit telephone given as its fax
    ({"fax": "+971 6 5050026"}, True),            # Passport Unit fax
    ({"telephone": "+971 6 5053453"}, True),      # Central Laboratories telephone
    ({"telephone": "+971 6 5050444"}, False),     # Central Laboratories fax given as a telephone
    ({"telephone": "+971 6 5057666"}, True),      # Dental Hospital Call Center (telephone only)
    ({"telephone": "+971 6 5585173"}, False),     # College of Engineering fax given as a telephone (the PF-16 defect)
    ({"fax": "+971 6 5585174 / +971 6 5585160"}, True),
    ({"telephone": "+971 6 1234567"}, False),     # not printed in the directory at all
    ({"extension": "1234"}, False),
])
def test_every_label_is_confirmed_by_the_page_line_reader(values, ok):
    assert D.column_check(values, PAGES)[0] is ok


def test_without_page_lines_the_printed_header_binding_stands_alone_and_says_so():
    ok, note = D.column_check({"telephone": "+971 6 5050023"}, [{"page": 54, "text": ""}])
    assert ok and "unavailable" in note


# ---- the Plan F runtime on the real handbook ----------------------------------------------------------------------------
@pytest.mark.parametrize("question, field, number", [
    ("What is the phone number of the Human Resources Department?", "telephone", "+971 6 5050023"),
    ("What is the fax number of the Human Resources Department?", "fax", "+971 6 5585200"),
    ("What is the fax number of the Passport Unit?", "fax", "+971 6 5050026"),          # frozen I02: the telephone came first
    ("What is the phone number of the Passport Unit?", "telephone", "+971 6 5050066"),
    ("What is the phone number of the Central Laboratories?", "telephone", "+971 6 5053453"),
    ("What is the phone number of the Information Technology Center?", "telephone", "+971 6 5050000"),
    ("What is the fax number of the Finance Department?", "fax", "+971 6 5585183"),
])
def test_plan_f_states_only_the_asked_field_with_its_label_and_keeps_the_printed_row_as_evidence(runtime, question, field, number):
    attempt = runtime.attempt(question)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED, attempt.reason
    result = attempt.result
    assert LABELLED.findall(result.answer) == [(field, number)]
    assert number in result.items[0]["chunk"] and len(NUMBER.findall(result.items[0]["chunk"])) == 2     # the verbatim row
    assert result.pages == [result.items[0]["meta"]["page"]] and 52 <= result.pages[0] <= 55
    labels = [e for e in attempt.trace if e.agent == "contact_labels"]
    assert labels and labels[0].decision == "verified" and "page reader" in labels[0].detail["check"]


def test_no_plan_f_contact_answer_gives_a_number_from_the_other_column(runtime):
    # every office of the Institutional lexicon, asked for its telephone and its fax
    served = 0
    for office in N.OFFICES_BY_ID.values():
        for kind, field in (("phone", "telephone"), ("fax", "fax")):
            attempt = runtime.attempt("What is the %s number of the %s?" % (kind, office.name))
            if attempt.result is None:
                continue
            served += 1
            pairs = LABELLED.findall(attempt.result.answer)
            assert pairs and all(label == field for label, _n in pairs), attempt.result.answer
            assert all(field in _column(n) for _label, n in pairs), attempt.result.answer
            assert len(NUMBER.findall(attempt.result.answer)) == len(pairs)          # no unlabelled number
    assert served >= 30


def test_a_label_the_page_reader_contradicts_is_never_shown(kb):
    # the printed HR line with its columns swapped: the page reader now puts 5050023 in the fax column, the specialist
    # (reading the row chunk under the header) still binds it as the telephone - the two readers disagree
    pages = copy.deepcopy(kb.pages)
    swapped = 0
    for page in pages:
        lines = page.get("lines") or []
        for i in range(len(lines) - 2):           # printed as three lines: name, telephone, fax
            if lines[i].strip() == "Human Resources" and lines[i + 1].strip() == "+971 6 5050023" and lines[i + 2].strip() == "+971 6 5585200":
                lines[i + 1], lines[i + 2] = lines[i + 2], lines[i + 1]
                swapped += 1
    assert swapped == 1 and "fax" in G.directory_columns(pages)["97165050023"]
    runtime, build = PF.build_runtime(kb)
    runtime.directory_pages = pages
    attempt = runtime.attempt("What is the phone number of the Human Resources Department?")
    assert attempt.outcome is PF.PlanFOutcome.REQUIRES_FALLBACK and "contact labels not verified" in attempt.reason
    assert attempt.result is None
    assert any(e.agent == "contact_labels" and e.decision == "rejected" for e in attempt.trace)


def test_the_aggregation_path_labels_its_directory_part_too(runtime):
    question = "How many theses can a faculty member supervise at once and what is the phone number of the Human Resources Department?"
    attempt = runtime.attempt(question)
    assert attempt.outcome is PF.PlanFOutcome.ACCEPTED and attempt.path == PF.PATH_AGGREGATE, attempt.reason
    assert attempt.result.answer.endswith("\n\nHuman Resources (contact directory): telephone +971 6 5050023.")
    assert "Human Resources +971 6 5050023 +971 6 5585200" in [i["chunk"] for i in attempt.result.items]


def test_the_rendering_changes_nothing_but_the_text_of_a_directory_row(runtime):
    # non-contact Plan F answers are rendered verbatim as before
    for question in ("How many theses can a faculty member supervise at once?", "How many credit hours is a full teaching load?"):
        attempt = runtime.attempt(question)
        assert attempt.outcome is PF.PlanFOutcome.ACCEPTED
        assert attempt.result.answer == "\n\n".join(i["chunk"] for i in attempt.result.items)
        assert not any(e.agent == "contact_labels" for e in attempt.trace)
