"""
Faculty Services & HR specialist: source scope, ownership, exclusions and the section guard on shared pages
(Milestone 4, Step 4.11; D-085).
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus, SpecialistId
from handbook_bot.sources import unregistered_source

from faculty_services_support import (P106, P107, P122, P124, P255, P257, P263, P266, P268, PAGES, ScriptedReranker, corpus_from,
                                      make_specialist, registry_and_map, run, task)


@pytest.fixture(scope="module")
def rm():
    return registry_and_map()


@pytest.fixture(scope="module")
def corpus(rm):
    return corpus_from(rm, PAGES)


@pytest.fixture(scope="module")
def specialist(corpus):
    return make_specialist(corpus)


# ---------------------------------------------------------------------------
# Ownership is section level, exclusions explicit
# ---------------------------------------------------------------------------
def test_owned_identities_are_sections_never_chapters(rm):
    _source, section_map = rm
    records = {(r.chapter, r.section_no) for r in section_map.sections()}
    for chapter, number, _label in F.SCOPE_SECTIONS:
        assert (chapter, number) in records, (chapter, number)
        assert "." in number                                         # a printed section number, never a chapter


def test_scope_pages_follow_the_real_section_map(rm):
    _source, section_map = rm
    scope = F.compile_scope(section_map)
    assert scope.describe()["page_ranges"] == [[56, 74], [85, 88], [98, 111], [122, 152], [255, 255], [257, 257], [266, 268]]


def test_excluded_sections_are_not_in_scope(rm):
    _source, section_map = rm
    owned = {(c, n) for c, n, _ in F.SCOPE_SECTIONS}
    for chapter, number, _label in F.EXCLUDED_SECTIONS:
        assert (chapter, number) not in owned
    # the directory, the teaching sections, the research sections and the AI guidelines stay out
    for chapter, number in ((1, "1.16"), (3, "3.1"), (3, "3.4"), (3, "3.11"), (5, "5.3"), (5, "5.4"), (6, "6.2"), (6, "6.3"), (16, "16.1"), (16, "16.4"), (16, "16.6")):
        assert (chapter, number) not in owned


def test_shared_sections_are_bounded_and_listed():
    shared = {(c, n) for c, n, _ in F.SHARED_SECTIONS}
    assert shared == {(6, "6.4"), (7, "7.2"), (7, "7.3"), (7, "7.7")}
    assert not shared & {(c, n) for c, n, _ in F.OWNED_SECTIONS}


def test_chapter_two_preamble_is_the_only_extra_page(rm):
    assert F.EXTRA_PAGES == ((56, 56),)
    _source, section_map = rm
    assert 56 not in {p for r in section_map.sections() for p in range(r.start_page, r.end_page + 1) if r.chapter == 2 and r.section_no}
    assert 106 in {p for r in section_map.sections() if (r.chapter, r.section_no) == (4, "4.1") for p in range(r.start_page, r.end_page + 1)}


def test_scope_never_silently_smaller_or_wider(rm):
    _source, section_map = rm
    with pytest.raises(ValueError):
        F.compile_scope(section_map, owned=F.SCOPE_SECTIONS + ((4, "4.99", "no such section"),))
    with pytest.raises(ValueError):
        F.compile_scope(section_map, extra_pages=((79, 79),))     # 3.4 Office Hours is not HR's


# ---------------------------------------------------------------------------
# Approved source only
# ---------------------------------------------------------------------------
def test_unregistered_source_is_refused(corpus):
    from types import SimpleNamespace
    bad = SimpleNamespace(**{**vars(corpus), "source": unregistered_source("other_pdf.pdf")})
    with pytest.raises(ValueError):
        make_specialist(bad)


def test_unapproved_task_scope_is_an_error(specialist):
    t = task("What is the probation period?", source_scope=["some_other_source"])
    result = specialist.run(t)
    assert result.status is FindingStatus.ERROR and "not approved" in result.metadata["error"]


def test_task_for_another_specialist_is_rejected(specialist):
    with pytest.raises(ValueError):
        specialist.run(task("What is the probation period?", specialist_id="teaching"))


# ---------------------------------------------------------------------------
# The section guard on the shared pages named by the Step 4.9 audit
# ---------------------------------------------------------------------------
def _quotes(result):
    return [f.evidence_quote for f in result.findings]


def test_p122_teaching_evaluation_text_never_supports_an_hr_ask(rm):
    sp = make_specialist(corpus_from(rm, [P122]), ScriptedReranker([], default=9.0))
    r = run(sp, "Must new faculty undergo a probationary review within the first year of appointment?")
    assert r.status is FindingStatus.SUPPORTED and all("student evaluations" not in q for q in _quotes(r))
    r = run(sp, "Does the annual evaluation of faculty use student evaluations to assess teaching performance?")
    assert all("student evaluations" not in q and "course objectives" not in q for q in _quotes(r))
    assert r.status is not FindingStatus.SUPPORTED


def test_hr_cued_text_inside_a_non_owned_section_is_never_quoted(rm):
    # a sentence with an HR cue printed under the Teaching heading 6.2 (p122) or the community-service heading 6.5
    # (p124): the guard attributes it to the non-owned section and the specialist never quotes it
    sp = make_specialist(corpus_from(rm, [P122]), ScriptedReranker([], default=9.0))
    r = run(sp, "Are faculty on probation evaluated through student evaluations each semester?")
    assert r.status is not FindingStatus.SUPPORTED and all("student evaluations" not in q for q in _quotes(r))
    sp = make_specialist(corpus_from(rm, [P124]), ScriptedReranker([], default=9.0))
    r = run(sp, "Do faculty with administrative appointments receive a service weight of 30 percent in the evaluation?")
    assert r.status is not FindingStatus.SUPPORTED and all("service weight" not in q for q in _quotes(r))


def test_p124_research_and_community_text_never_supports_an_hr_process_ask(rm):
    sp = make_specialist(corpus_from(rm, [P124]), ScriptedReranker([], default=9.0))
    r = run(sp, "Who conducts the faculty's performance evaluation?")
    assert r.status is FindingStatus.SUPPORTED and any("Department Chair and College Dean" in q for q in _quotes(r))
    for q in _quotes(r):
        assert "patents" not in q and "community projects" not in q and "thesis and dissertation" not in q
    r = run(sp, "Does the faculty evaluation include recognition through awards, grants and patents?")
    assert all("patents" not in q for q in _quotes(r)) and r.status is not FindingStatus.SUPPORTED


def test_p255_and_p257_service_sections_stop_at_their_neighbours(rm):
    sp = make_specialist(corpus_from(rm, [P255, P257]), ScriptedReranker([], default=9.0))
    r = run(sp, "Does the University nursery provide care for the children of employees?")
    for q in _quotes(r):
        assert "Disability Resource" not in q and "Facilities Management" not in q
    r = run(sp, "Do the University Medical Clinics serve faculty housing?")
    for q in _quotes(r):
        assert "swimming pools" not in q and "DRC" not in q


def test_p263_p266_p268_harassment_policy_is_quoted_without_research_or_ai_text(rm):
    sp = make_specialist(corpus_from(rm, [P263, P266, P268]), ScriptedReranker([], default=9.0))
    r = run(sp, "Does the University prohibit sexual harassment?")
    assert r.status is FindingStatus.SUPPORTED
    for q in _quotes(r):
        assert "animals" not in q and "informed consent" not in q and "Artificial Intelligence" not in q
    r = run(sp, "Can I use AI tools for my course materials?")
    assert r.status is FindingStatus.OUT_OF_SCOPE or all("Artificial Intelligence" not in q for q in _quotes(r))


def test_p106_p107_quotes_are_extractive_and_in_scope(corpus, specialist):
    for q in ("Is the furniture allowance AED 30,000?", "Is sick leave with full pay 15 working days?"):
        r = run(specialist, q)
        assert r.status is FindingStatus.SUPPORTED
        for f in r.findings:
            chunk = corpus.chunks[f.metadata["chunk_id"]]
            assert f.evidence_quote in chunk and f.claim == f.evidence_quote
            assert f.source_id == F.APPROVED_SOURCE_ID and f.page in (106, 107)
            assert f.metadata["extractive"] is True


def test_result_metadata_names_the_scope_and_shared_pages(specialist):
    r = run(specialist, "What is the probation period?")
    assert r.metadata["scope"]["source_ids"] == [F.APPROVED_SOURCE_ID]
    assert set(r.metadata["boundary"]["shared_pages"]) >= {122, 124, 255, 257, 266, 268}
    assert r.metadata["boundary"]["unresolved_pages"] == []
    assert r.llm_used is False and r.specialist_id is SpecialistId.FACULTY_SERVICES
