"""Deterministic extractors state a value only when the printed line binds it (defect D-03 of the 2026-11-07 audit).

An extractor answer skips the model and the Verifier's prose check, so a loose match reads as a certain fact. These tests
use a synthetic directory and calendar with the shapes the real tables print: complete rows, cells split over several
lines, Men / Women sub-rows and labelled sub-entries.
"""
from __future__ import annotations

from handbook_bot.extractors import extract_contact_answer, extract_date_answer, page_row_lines

HEADER = "Office / College / Department Telephone Number Fax Number"
DIRECTORY = [
    HEADER,
    "Payroll Office +971 6 5000001 +971 6 5000002",
    "College of Marine Studies Men: +971 6 5000011",
    "+971 6 5000012 /",
    "Women: +971 6 5000013",
    "+971 6 5000014",
    "College of Fine Crafts +971 6 5000021 +971 6 5000022",
    "College of Pottery +971 6 500 0031",
    "College of Long Printed",
    "+971 6 5000041 +971 6 5000042",
    "Clinic Service +971 6 5000051 /",
    "Records Unit +971 6 5000061 +971 6 5000062",
    "Medical Colleges Library: +971 6",
    "College of Fine Crafts",
    "Libraries Library: +971 6 5000071",
    "Men's Library: +971 6 5000081",
]


def row_item(line, page=9):
    return {"chunk": line, "meta": {"page": page, "chunk_type": "row", "source_id": "doc"}}


def rows(*lines, page=9):
    return [row_item(line, page) for line in lines]


def lookup(lines=DIRECTORY):
    return lambda item: list(lines)


def contact(question, items, lines=DIRECTORY):
    return extract_contact_answer(question, items, lookup(lines))


# ---------------------------------------------------------------------------
# A complete row binds both columns
# ---------------------------------------------------------------------------
def test_complete_row_gives_the_telephone():
    result = contact("What is the phone number for the Payroll Office?", rows(DIRECTORY[1]))
    assert result[0] == "The phone number is +971 6 5000001."
    assert result[1] == [9]


def test_complete_row_gives_the_fax_from_the_fax_column():
    result = contact("What is the fax number for the Payroll Office?", rows(DIRECTORY[1]))
    assert result[0] == "The fax number is +971 6 5000002."


def test_a_request_for_both_gets_both():
    result = contact("What are the phone and fax numbers for the Payroll Office?", rows(DIRECTORY[1]))
    assert result[0] == "The phone number is +971 6 5000001. The fax number is +971 6 5000002."


def test_office_name_comparison_ignores_case_and_apostrophe_style():
    lines = [HEADER, "Dean’s Office +971 6 5000091 +971 6 5000092"]
    result = contact("What is the phone number for the dean's office?", rows(lines[1]), lines)
    assert result[0] == "The phone number is +971 6 5000091."


# ---------------------------------------------------------------------------
# Anything less than a complete row declines, so the model and the Verifier handle it
# ---------------------------------------------------------------------------
def test_one_number_is_never_labelled_a_fax():
    assert contact("What is the fax number for the College of Pottery?", rows(DIRECTORY[7])) is None


def test_one_number_is_not_given_a_column():
    assert contact("What is the phone number for the College of Pottery?", rows(DIRECTORY[7])) is None


def test_men_and_women_entry_declines():
    items = rows(DIRECTORY[2], DIRECTORY[3])
    assert contact("What is the phone number of the College of Marine Studies?", items) is None
    assert contact("College of Marine Studies, men's section: phone and fax please.", items) is None


def test_a_generic_word_the_directory_does_not_print_is_dropped():
    assert contact("What is the phone number of the Payroll Office Department?", rows("Payroll Office +971 6 5000001 +971 6 5000002"))[0] == "The phone number is +971 6 5000001."
    lines = [HEADER, "Records Unit +971 6 5000061 +971 6 5000062"]
    assert contact("What is the phone number of the Records Unit office?", rows(*lines), lines)[0] == "The phone number is +971 6 5000061."


def test_a_printed_name_is_not_shortened():
    lines = [HEADER, "Payroll Office Unit +971 6 5000001 /", "Payroll Office +971 6 5000091 +971 6 5000092"]
    assert contact("What is the phone number of the Payroll Office Unit?", rows(*lines), lines) is None


def test_only_a_generic_word_left_names_nothing():
    assert contact("What is the phone number of the department?", rows(*DIRECTORY)) is None


def test_a_longer_printed_name_is_not_the_asked_office():
    assert contact("What is the phone number for the College of Fine?", rows(DIRECTORY[6])) is None


def test_a_unit_inside_an_office_is_not_the_office():
    question = "What is the phone number of the College of Fine Crafts Library?"
    assert contact(question, rows(DIRECTORY[6], DIRECTORY[14])) is None


def test_a_shared_word_does_not_name_a_library():
    assert contact("What is the phone number of the University Library?", rows(DIRECTORY[14], DIRECTORY[15])) is None


def test_name_on_one_line_and_numbers_on_the_next_declines():
    assert contact("What is the phone number for the College of Long Printed?", rows(DIRECTORY[8], DIRECTORY[9])) is None


def test_a_row_after_a_line_that_runs_on_declines():
    assert contact("What is the phone number for the Records Unit?", rows(DIRECTORY[11])) is None


def test_a_row_before_a_sub_row_declines():
    lines = [HEADER, "Bursar Office +971 6 5000101 +971 6 5000102", "Women: +971 6 5000103"]
    assert contact("What is the phone number for the Bursar Office?", rows(lines[1]), lines) is None


def test_a_row_before_a_broken_number_declines():
    lines = [HEADER, "Bursar Office +971 6 5000101 +971 6 5000102", "5000103 +971 6 5000104"]
    assert contact("What is the phone number for the Bursar Office?", rows(lines[1]), lines) is None


def test_a_page_with_no_column_header_declines():
    lines = DIRECTORY[1:]
    assert contact("What is the phone number for the Payroll Office?", rows(lines[0]), lines) is None


def test_columns_printed_the_other_way_round_decline():
    lines = ["Office Fax Number Telephone Number", DIRECTORY[1]]
    assert contact("What is the phone number for the Payroll Office?", rows(lines[1]), lines) is None


def test_an_office_printed_twice_declines():
    lines = [HEADER, DIRECTORY[1], "Payroll Office +971 6 5000111 +971 6 5000112"]
    assert contact("What is the phone number for the Payroll Office?", rows(lines[1]), lines) is None


def test_a_window_is_read_line_by_line_and_binds_only_a_complete_line():
    window = {"chunk": " | ".join(DIRECTORY[6:8]), "meta": {"page": 9, "chunk_type": "row_window", "source_id": "doc"}}
    assert contact("What is the fax number for the College of Fine Crafts?", [window])[0] == "The fax number is +971 6 5000022."
    assert contact("What is the fax number for the College of Pottery?", [window]) is None


def test_prose_with_a_number_is_left_to_the_model():
    item = {"chunk": "Call the Payroll Office on +971 6 5000001 between nine and five.",
            "meta": {"page": 9, "chunk_type": "paragraph", "source_id": "doc"}}
    assert contact("What is the phone number for the Payroll Office?", [item]) is None


def test_no_page_lookup_declines_a_number():
    assert extract_contact_answer("What is the phone number for the Payroll Office?", rows(DIRECTORY[1])) is None


# ---------------------------------------------------------------------------
# E-mail: the row must name the whole asked target
# ---------------------------------------------------------------------------
def test_email_needs_every_word_of_the_target_on_the_row():
    item = row_item("Payroll Office help desk: payroll@example.ac.ae")
    assert contact("What is the email for the Payroll Office?", [item])[0] == "The email address is payroll@example.ac.ae."
    assert contact("What is the email for the Admissions Office?", [item]) is None


# ---------------------------------------------------------------------------
# Page lines come from the chunk store, in printed order, for one document page
# ---------------------------------------------------------------------------
def test_page_row_lines_keeps_the_order_of_one_document_page():
    chunks = ["a one", "paragraph", "other document", "a two", "other page"]
    metadata = [{"page": 9, "chunk_type": "row", "source_id": "doc"}, {"page": 9, "chunk_type": "paragraph", "source_id": "doc"},
                {"page": 9, "chunk_type": "row", "source_id": "elsewhere"}, {"page": 9, "chunk_type": "row", "source_id": "doc"},
                {"page": 8, "chunk_type": "row", "source_id": "doc"}]
    assert page_row_lines(chunks, metadata, row_item("x")) == ["a one", "a two"]


# ---------------------------------------------------------------------------
# Dates: the row must state the asked event and a date
# ---------------------------------------------------------------------------
def date_rows(*lines):
    return [{"chunk": line, "meta": {"page": 49, "chunk_type": "row"}} for line in lines]


def test_calendar_row_answers_unchanged():
    result = extract_date_answer("When do classes begin?", date_rows("Mon 25 Aug 02 Rabi I Classes begin"))
    assert result == ("Mon 25 Aug 02 Rabi I Classes begin.", [49], "Mon 25 Aug 02 Rabi I Classes begin")


def test_final_exam_calendar_row_answers():
    result = extract_date_answer("When are final exams held?", date_rows("Sat -Tue 06-16 Dec 15 -25 Jumada II Final Exams"))
    assert result[0] == "Sat -Tue 06-16 Dec 15 -25 Jumada II Final Exams."


def test_a_row_without_a_date_is_never_returned():
    lines = ("course within two weeks after grades have been posted to have their final examination answer",)
    assert extract_date_answer("When can students review their final exam paper?", date_rows(*lines)) is None


def test_a_number_is_not_a_date():
    assert extract_date_answer("When are final exams weighted?", date_rows("40-50% allocated for the final exam")) is None


def test_a_date_on_a_row_about_something_else_is_not_the_answer():
    assert extract_date_answer("When do classes begin?", date_rows("Mon 25 Aug 02 Rabi I Orientation day")) is None


def test_the_row_with_the_date_wins_over_a_wordier_row_without_one():
    lines = ("classes begin when the students arrive and classes begin again", "Mon 25 Aug 02 Rabi I Classes begin")
    assert extract_date_answer("When do classes begin?", date_rows(*lines))[0] == "Mon 25 Aug 02 Rabi I Classes begin."


def test_a_row_of_another_term_is_not_the_answer():
    question = "When do classes begin in the summer semester?"
    assert extract_date_answer(question, date_rows("Mon 25 Aug 02 Rabi I Classes begin")) is None


def test_a_row_that_prints_the_asked_term_answers():
    row = "Mon 24 Aug 11 Rabi I Classes begin for Fall 2026-2027"
    assert extract_date_answer("When do classes begin for Fall 2026-2027?", date_rows(row))[0] == row + "."


def test_a_year_the_row_does_not_print_is_not_answered():
    assert extract_date_answer("When do classes begin in 2027?", date_rows("Mon 25 Aug 02 Rabi I Classes begin")) is None


def test_a_relative_term_cannot_be_bound_to_a_row():
    for question in ("When do classes begin next semester?", "When do classes begin this term?"):
        assert extract_date_answer(question, date_rows("Mon 25 Aug 02 Rabi I Classes begin")) is None


# ---------------------------------------------------------------------------
# Through the answer pipeline: a bound row answers phone and fax together without the model
# ---------------------------------------------------------------------------
def test_phone_and_fax_of_one_office_stays_one_question(ask):
    result = ask("What are the phone and fax numbers for the Information Technology Center?")
    assert result.answer == "The phone number is +971 6 5050000. The fax number is +971 6 5585101."
    assert result.llm_calls == 0
