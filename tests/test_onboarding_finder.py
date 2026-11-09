"""The resource finder's deterministic search (post-freeze PF-14): English and Arabic queries, relevance, missing
results, category filters, task context, provenance of what is shown, and the exclusion of held sources."""
from __future__ import annotations

from onboarding import catalog as C
from onboarding.finder import normalize, search, tokens


def ids(results):
    return [r["resource"].id for r in results]


def test_english_queries_find_the_expected_resources_first() -> None:
    assert ids(search("Blackboard help"))[0] == "doc_bb_short_guide"
    assert ids(search("Faculty Handbook"))[0] == "doc_handbook"
    assert ids(search("Human Resources"))[0] == "dept_hr"
    assert ids(search("Research funding"))[0] == "dept_research_funding"
    assert ids(search("Orientation information"))[0] == "dept_ilhe"
    assert "doc_grade_transfer" in ids(search("transfer grades to banner"))[:2]


def test_arabic_queries_find_the_same_resources() -> None:
    assert ids(search("دليل أعضاء هيئة التدريس"))[0] == "doc_handbook"
    assert ids(search("خدمات الموارد البشرية"))[0] == "dept_hr"
    assert ids(search("بلاكبورد"))[0] in ("doc_bb_short_guide", "doc_bb_communication", "doc_bb_attendance", "doc_bb_collaborate", "doc_bb_gradebook")
    assert ids(search("تمويل البحث"))[0] == "dept_research_funding"
    assert "dept_ilhe" in ids(search("التهيئة"))[:2]                            # the article is dropped, the ta marbuta normalised


def test_a_query_without_a_match_returns_nothing_instead_of_fillers() -> None:
    assert search("parking permit") == []
    assert search("xyzzy") == [] and search("قطة") == []


def test_only_matching_resources_are_returned_and_scores_are_positive() -> None:
    results = search("iThenticate")
    assert ids(results)[0] == "doc_ithenticate" and results[0]["score"] > results[-1]["score"] > 0
    assert set(ids(results)) <= {"doc_ithenticate", "dept_itc"}                  # the ITC publishes the guide; nothing else mentions it
    for r in search("research"):
        assert r["score"] > 0


def test_category_filters_restrict_the_listing_and_the_search() -> None:
    for category in C.CATEGORIES:
        listed = search("", category=category)
        assert listed and all(r["resource"].category == category for r in listed)
    assert all(r["resource"].category == "teaching" for r in search("guide", category="teaching"))
    assert search("Human Resources", category="research") == []


def test_task_context_puts_related_resources_first() -> None:
    listed = search("", task_id="bb_announcement")
    assert listed[0]["contextual"] and listed[0]["resource"].id in ("doc_bb_short_guide", "doc_bb_communication")
    searched = search("guide", task_id="grade_transfer")
    assert searched[0]["resource"].id in ("doc_grade_transfer", "doc_bb_gradebook") and searched[0]["contextual"]
    assert not any(r["contextual"] for r in search("", task_id="no_such_task"))


def test_an_empty_query_lists_the_whole_catalog_documents_first() -> None:
    listed = search("", limit=100)
    assert len(listed) == len(C.RESOURCES)
    assert listed[0]["resource"].type == "document"
    assert not any(r["resource"].source_id in C.HELD_SOURCES for r in listed)


def test_tokeniser_drops_stop_words_and_normalises_arabic() -> None:
    assert tokens("How do I send an announcement in Blackboard?") == ["send", "announcement", "blackboard"]
    assert tokens("ما هو رقم هاتف قسم الموارد البشرية؟") == ["رقم", "هاتف", "قسم", "موارد", "بشريه"]
    assert normalize("الإدارة") == "الاداره"


def test_shown_contacts_and_pages_come_from_the_catalog_provenance() -> None:
    for r in search("", limit=100):
        res = r["resource"]
        for c in res.contacts:
            assert c.evidence.page == 54 and c.value in c.evidence.quote
        if res.type == "document":
            assert res.pages[0] >= 1 and res.pages[-1] >= res.pages[0]
