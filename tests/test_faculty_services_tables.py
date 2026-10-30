"""
Faculty Services & HR structured data: the salary table rows, the administrative compensation list and the pages whose
numbers sit close together (Milestone 4, Step 4.11; D-087).
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus

from faculty_services_support import P106, P107, P108, P148, P149, P149_HEAD, P150, ScriptedReranker, corpus_from, make_specialist, registry_and_map, run

SUPPORTED, PARTIAL = FindingStatus.SUPPORTED, FindingStatus.PARTIAL


@pytest.fixture(scope="module")
def rm():
    return registry_and_map()


@pytest.fixture(scope="module")
def sp(rm):
    return make_specialist(corpus_from(rm, [P106, P107, P108]))


@pytest.fixture(scope="module")
def high(rm):
    return make_specialist(corpus_from(rm, [P106, P107, P108]), ScriptedReranker([], default=9.0))


def status(specialist, question):
    return run(specialist, question).status


# ---------------------------------------------------------------------------
# 1. The salary table
# ---------------------------------------------------------------------------
def test_salary_row_shape_and_reader():
    assert F.salary_row("Associate Professor 25,318 – 37,926") == ("associate professor", 25318, 37926)
    assert F.salary_row("Professor 30,576 – 43,860") == ("professor", 30576, 43860)
    assert F.salary_row("Lecturer 14,242 – 23,022") == ("lecturer", 14242, 23022)
    assert F.salary_row("The University of Sharjah offers a competitive basic salary based on faculty rank") is None
    # D-090 (Step 4.12-C3): the header names the columns and is no table row; only a rank row is
    assert F.is_table_row("Instructor 18,744 – 22,744") and not F.is_table_row("Academic Rank Basic Monthly Salary (AED)")
    assert not F.is_table_row("Faculty members are entitled to various benefits as part of their employment package:")


def test_asked_rank_reads_the_rank_not_the_nearest_word():
    assert F.asked_rank("What is the basic salary of an associate professor?") == "associate professor"
    assert F.asked_rank("What is the salary of a full professor?") == "professor"
    assert F.asked_rank("What is the lecturer salary?") == "lecturer"
    assert F.asked_rank("What is the salary scale?") is None


def test_correct_rank_and_range_are_supported(sp, high):
    for s in (sp, high):
        r = run(s, "What is the basic salary of an associate professor?")
        assert r.status is SUPPORTED and any(f.evidence_quote.startswith("Associate Professor 25,318") for f in r.findings)
        assert status(s, "Is the basic monthly salary of a professor 30,576 to 43,860 AED?") is SUPPORTED


def test_wrong_rank_or_another_rows_range_is_partial(sp, high):
    for s in (sp, high):
        assert status(s, "Is the basic monthly salary of a lecturer 30,576 to 43,860 AED?") is PARTIAL
        assert status(s, "Is the basic monthly salary of an assistant professor 25,318 to 37,926 AED?") is PARTIAL
        assert status(s, "Is the basic monthly salary of a professor 14,242 to 23,022 AED?") is PARTIAL


def test_an_unknown_rank_never_inherits_the_nearest_row(sp, high):
    for s in (sp, high):
        r = run(s, "What is the basic salary of a clinical professor?")
        assert r.status is not SUPPORTED


def test_general_salary_scale_overview_is_supported(sp):
    r = run(sp, "What is the salary scale for faculty?")
    assert r.status is SUPPORTED


def test_a_salary_row_never_settles_an_actor_or_obligation_ask(sp, high):
    for s in (sp, high):
        assert status(s, "Must the dean pay a professor 30,576 AED?") is not SUPPORTED
        assert status(s, "Does the Chancellor approve the salary of an associate professor of 25,318 AED?") is not SUPPORTED


# ---------------------------------------------------------------------------
# 2. Administrative compensation: role, amount and reduction in one item
# ---------------------------------------------------------------------------
def test_items_split_the_compensation_list_by_role():
    unit = "o Dean: Monthly compensation of AED 10,000 and a reduction in teaching load of nine credit hours per semester. o Associate Dean: Monthly compensation of AED 5,000 and a reduction in teaching load of three to six credit hours per semester."
    items = F.hr_items(unit)
    assert len(items) == 2 and items[0].startswith("dean:") and items[1].startswith("associate dean:")


def test_amounts_stay_with_their_role(sp, high):
    for s in (sp, high):
        assert status(s, "Is the monthly compensation of a dean AED 10,000?") is SUPPORTED
        assert status(s, "Is the monthly compensation of a chairperson AED 10,000?") is PARTIAL
        assert status(s, "Is the monthly compensation of a chairperson AED 4,000?") is SUPPORTED
        assert status(s, "Is the monthly compensation of a program coordinator AED 4,000?") is PARTIAL
        assert status(s, "Is the monthly compensation of a director of a unit AED 2,000 to AED 3,000?") is SUPPORTED
        assert status(s, "Is the monthly compensation of a director of a unit AED 3,000 to AED 5,000?") is PARTIAL


def test_load_reductions_stay_with_their_role(sp, high):
    for s in (sp, high):
        assert status(s, "Is the teaching load reduction for a dean nine credit hours per semester?") is SUPPORTED
        assert status(s, "Is the teaching load reduction for a chairperson nine credit hours per semester?") is PARTIAL
        assert status(s, "Is the teaching load reduction for an assistant dean three credit hours per semester?") is SUPPORTED


def test_a_hyphenated_period_is_the_spaced_one(sp, high):
    for s in (sp, high):
        assert status(s, "Is the teaching load reduction for a dean nine credit hours per-semester?") is SUPPORTED
        assert status(s, "Is the teaching load reduction for a dean nine credit hours per-year?") is PARTIAL


def test_a_pure_teaching_load_ask_is_not_hr(sp):
    r = run(sp, "What is the teaching load of a dean?")
    assert r.status is FindingStatus.OUT_OF_SCOPE and [h.requested_specialist.value for h in r.handoff_requests] == ["teaching"]


# ---------------------------------------------------------------------------
# 3. Numbers that sit close together on one page
# ---------------------------------------------------------------------------
def test_benefit_amounts_do_not_combine_across_bullets(sp, high):
    for s in (sp, high):
        assert status(s, "Is the furniture allowance 25,000 Dirhams?") is PARTIAL
        assert status(s, "Is the children's education allowance 30,000 Dirhams per child?") is PARTIAL
        assert status(s, "Is health insurance offered for up to three dependent children?") is SUPPORTED
        assert status(s, "Is health insurance offered for up to four dependent children?") is PARTIAL


def test_leave_numbers_do_not_combine_across_sub_sections(sp, high):
    for s in (sp, high):
        assert status(s, "Can unpaid special leave of up to thirty days be granted?") is SUPPORTED
        assert status(s, "Can unpaid special leave of up to sixty days be granted?") is PARTIAL
        assert status(s, "Is Hajj leave up to twenty calendar days?") is SUPPORTED
        assert status(s, "Is Hajj leave up to thirty calendar days?") is PARTIAL


# ---------------------------------------------------------------------------
# D-088 (Step 4.12-C1): the organisational level of the promotion procedures and the content-free heading
# ---------------------------------------------------------------------------
def _high(rm, pages):
    return make_specialist(corpus_from(rm, pages), ScriptedReranker([], default=9.0))


def test_promotion_procedure_levels_are_bound(rm):
    sp = _high(rm, [P148, P149, P150])
    # "what is the ... procedure" is a process request: at most partial, the documented stages quoted (D-085)
    for level in ("department", "college", "university"):
        r = run(sp, "What is the promotion procedure at the %s level?" % level)
        assert r.status is PARTIAL and F.is_process_request("What is the promotion procedure at the %s level?" % level)
        assert any("the complete procedure" in m for m in r.missing)
    # a level-qualified fact is settled only by an item the asked level governs
    assert status(sp, "At the department level, can promotion applications be submitted in the first week of January, April or October?") is SUPPORTED
    assert status(sp, "At the college level, does the College Council form the College Academic Promotions Committee in September of each year?") is SUPPORTED
    assert status(sp, "At the university level, is the University Central Committee for the Promotion of Faculty Members chaired by the Chancellor or a delegate?") is SUPPORTED
    assert status(sp, "At the college level, can promotion applications be submitted in the first week of January, April or October?") is PARTIAL
    assert status(sp, "At the university level, can promotion applications be submitted in the first week of January, April or October?") is PARTIAL
    assert status(sp, "At the department level, does the College Council form the College Academic Promotions Committee in September of each year?") is PARTIAL
    assert status(sp, "At the university level, does the College Council form the College Academic Promotions Committee in September of each year?") is PARTIAL
    assert status(sp, "At the department level, is the University Central Committee for the Promotion of Faculty Members chaired by the Chancellor or a delegate?") is PARTIAL
    assert status(sp, "At the college level, is the University Central Committee for the Promotion of Faculty Members chaired by the Chancellor or a delegate?") is PARTIAL
    # a level is never settled by another level's pages
    assert status(_high(rm, [P148]), "At the college level, does the College Council form the College Academic Promotions Committee in September of each year?") is not SUPPORTED
    assert status(_high(rm, [P150]), "At the department level, can promotion applications be submitted in the first week of January, April or October?") is not SUPPORTED
    assert F._levels_stated("the college dean shall forward the report to the chair of the university central committee") == ["college", "university"]
    assert F._levels_stated("the department chair shall forward the report to the college dean") == ["department", "college"]


def test_a_content_free_heading_settles_no_procedure_ask(rm):
    heading = "7.8 Promotion Procedures at the College Level The promotion procedures at the college level involve the following steps: A."
    assert F.quote_problem(heading, "prose") == "lead-in"
    assert F.quote_problem("The promotion procedures at the college level involve a review by the College Council.", "prose") is None
    for q in ("Are there promotion procedures at the college level?", "What is the promotion procedure at the college level?"):
        r = run(_high(rm, [P149_HEAD]), q)
        assert r.status is not SUPPORTED and all("following steps" not in f.evidence_quote for f in r.findings)


def test_the_chair_of_a_promotion_committee_is_bound(rm):
    sp = _high(rm, [P148, P149, P150])
    assert status(sp, "Is the College Academic Promotions Committee chaired by the Chancellor?") is not SUPPORTED
    assert status(sp, "Is the promotion committee at the department level chaired by the Chancellor?") is not SUPPORTED
    assert status(sp, "Is the College Academic Promotions Committee chaired by the college dean if a full professor?") is SUPPORTED


def test_an_administrative_role_never_settles_its_sibling(sp, high):
    # associate and assistant deans share the role "dean": only the administrative-role axis keeps their figures apart
    for s in (sp, high):
        assert status(s, "Is the teaching load reduction for a dean nine credit hours per semester?") is SUPPORTED
        assert status(s, "Is the teaching load reduction for a dean six credit hours per semester?") is PARTIAL


# ---------------------------------------------------------------------------
# D-090 (Step 4.12-C3): the salary header is schema, never evidence; a data row settles only rank, amounts, currency
# ---------------------------------------------------------------------------
HEADER = "Academic Rank Basic Monthly Salary (AED)"
ROW = "Associate Professor 25,318 – 37,926"


def _grounding(q):
    return next(d for d in F.assess_ask(q).details if d.family == "grounding")


def test_the_salary_header_is_schema_and_never_a_row():
    assert F.is_salary_header(HEADER) and not F.is_salary_header(ROW)
    assert F.is_salary_data_row(ROW) and not F.is_salary_data_row(HEADER)
    assert not F.is_table_row(HEADER) and F.is_table_row(ROW)
    assert F.salary_row(HEADER) is None and F.salary_row(ROW) == ("associate professor", 25318, 37926)


def test_the_salary_header_is_never_quoted_as_evidence():
    assert F.quote_problem(HEADER, "prose") == "salary header"
    assert F.quote_problem(HEADER, "line") == "salary header"
    assert F.quote_problem(ROW, "line") is None


def test_the_salary_header_settles_no_proposition():
    # every word of this ask is on the header: only the header rule keeps it from being evidence
    assert not _grounding("Is the academic rank basic monthly salary in AED?").check(HEADER)
    assert not _grounding("Is the salary scale confidential?").check(HEADER)


def test_a_salary_row_settles_only_its_rank_amounts_and_currency():
    assert _grounding("Is the basic salary of an associate professor 25,318 to 37,926 AED?").check(ROW)
    assert not _grounding("Is an associate professor's salary negotiated individually?").check(ROW)
    assert not _grounding("Is an associate professor's salary based on years of service?").check(ROW)


def test_unsupported_salary_propositions_are_never_supported(sp, high):
    for s in (sp, high):
        for q in ("Is the salary scale based on years of service?", "Is the salary scale reviewed every year?", "Does salary depend on the college?",
                  "Is the salary scale confidential?", "Is academic-rank salary negotiated individually?", "Does salary depend on nationality?",
                  "Does salary depend on department size?", "Is salary automatically increased each semester?", "Is salary determined by teaching load?",
                  "Does a promotion immediately double salary?", "Is a professor's salary based on years of service?",
                  "Is the lecturer salary reviewed every year?", "Is an associate professor's salary negotiated individually?"):
            assert status(s, q) is not SUPPORTED, q
        assert status(s, "Is the basic salary of a professor 30,576 to 43,860 AED?") is SUPPORTED
        assert status(s, "Is the basic salary of an assistant professor 20,142 to 31,674 AED?") is SUPPORTED
        assert status(s, "Is the basic salary of a lecturer 14,242 to 23,022 AED?") is SUPPORTED
        assert status(s, "Is the basic salary of a lecturer 25,318 to 37,926 AED?") is PARTIAL


# ---------------------------------------------------------------------------
# D-091 (Step 4.12-C4): a qualified rank has no salary row; full professor is the one alias
# ---------------------------------------------------------------------------
def test_qualified_ranks_are_read_as_unknown():
    assert F.qualified_rank("Is the basic salary of a senior lecturer 14,242 to 23,022 AED?") == "senior lecturer"
    assert F.qualified_rank("What does a research professor earn?") == "research professor"
    assert F.qualified_rank("Is the salary of an assistant lecturer listed?") == "assistant lecturer"
    assert F.qualified_rank("Is the basic salary of a clinical assistant professor listed?") == "clinical assistant professor"
    assert F.qualified_rank("Is the basic salary of an associate professor 25,318 to 37,926 AED?") is None
    assert F.qualified_rank("Is the basic salary of an assistant professor 20,142 to 31,674 AED?") is None


def test_full_professor_is_the_one_accepted_alias(sp, high):
    assert F.qualified_rank("Is the basic salary of a full professor 30,576 to 43,860 AED?") is None
    assert F.qualified_rank("Is the basic salary of a senior professor 30,576 to 43,860 AED?") == "senior professor"
    for s in (sp, high):
        assert status(s, "Is the basic salary of a full professor 30,576 to 43,860 AED?") is SUPPORTED


def test_a_qualified_rank_never_inherits_the_base_row(sp, high):
    for s in (sp, high):
        for q in ("Is the basic salary of a senior lecturer 14,242 to 23,022 AED?", "Is the basic salary of a senior instructor 18,744 to 22,744 AED?",
                  "Is the basic salary of an adjunct professor 30,576 to 43,860 AED?", "Is the basic salary of a distinguished professor 30,576 to 43,860 AED?",
                  "Is the basic salary of a visiting professor 30,576 to 43,860 AED?", "Is the basic salary of a clinical professor 30,576 to 43,860 AED?",
                  "Is the basic salary of a research professor 30,576 to 43,860 AED?", "Is the basic salary of an assistant lecturer 14,242 to 23,022 AED?",
                  "Is the basic salary of a principal lecturer 14,242 to 23,022 AED?"):
            assert status(s, q) is not SUPPORTED, q
        for q in ("Is the basic salary of a professor 30,576 to 43,860 AED?", "Is the basic salary of an associate professor 25,318 to 37,926 AED?",
                  "Is the basic salary of an assistant professor 20,142 to 31,674 AED?", "Is the basic salary of a lecturer 14,242 to 23,022 AED?",
                  "Is the basic salary of an instructor 18,744 to 22,744 AED?"):
            assert status(s, q) is SUPPORTED, q
