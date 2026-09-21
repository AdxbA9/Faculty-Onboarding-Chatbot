"""Verifier agent: support across ALL evidence items, citation invariants,
rejection reasons and retry feedback. No LLM is involved anywhere here."""
from __future__ import annotations

import pytest

from handbook_bot.agents import verifier
from handbook_bot.agents.types import EvidenceResult, SubQuestion, SynthesisResult, VerifyResult

from conftest import make_items

LEAVE = "Faculty members are entitled to thirty days of annual leave each academic year subject to approval by the dean."
CONTACT = "Information Technology Center | +971 6 5050000 | itcenter@sharjah.ac.ae"
CALENDAR = "Mon | 25 Aug | 02 Rabi I | Classes begin"
PROGRAMS = "UoS offers a total of 149 degree programs: 62 Bachelor's, 60 Master's and 23 PhD programs."
COUNCIL = "The University Council is chaired by the Chancellor and meets twice each semester."


def check(answer, items, *, claimed=(), route="policy", is_retry=False, used_extractor=False,
          sub_questions=()):
    evidence = EvidenceResult("q", route, items, 1.0, True, None, used_extractor, None)
    return verifier.verify("q", SynthesisResult(answer, list(claimed), "test"), [evidence],
                           list(sub_questions), is_retry=is_retry)


def test_returns_a_well_formed_verify_result():
    result = check("Faculty get thirty days of annual leave each academic year.", make_items((LEAVE, 3)))
    assert isinstance(result, VerifyResult)
    assert isinstance(result.accepted, bool)
    assert isinstance(result.pages, list) and all(isinstance(p, int) for p in result.pages)
    assert isinstance(result.unsupported_claims, list)
    assert isinstance(result.uncovered_subquestions, list)
    assert result.retry_feedback is None or isinstance(result.retry_feedback, str)


def test_support_found_in_first_item():
    result = check("Faculty members are entitled to thirty days of annual leave.", make_items((LEAVE, 3), (COUNCIL, 7)))
    assert result.accepted and result.pages == [3]


def test_support_found_only_in_a_later_item():
    """The Part 1 check compared with items[0] only and rejected this answer."""
    items = make_items((COUNCIL, 7, 3.0), (CONTACT, 14, 2.0), (LEAVE, 3, 1.0))
    result = check("Faculty members are entitled to thirty days of annual leave each year.", items)
    assert result.accepted
    assert result.pages == [3]


def test_unsupported_answer_is_rejected_with_targeted_feedback():
    result = check("Faculty receive a free parking permit every semester.", make_items((LEAVE, 3)), claimed=[3])
    assert not result.accepted
    assert result.pages == []
    assert result.unsupported_claims == ["Faculty receive a free parking permit every semester."]
    assert "parking permit" in result.retry_feedback
    assert "Page 3" in result.retry_feedback          # the wrong citation is named
    assert result.retry_feedback != "Try again."


def test_number_absent_from_evidence_rejects_the_claim():
    result = check("UoS offers 150 degree programs.", make_items((PROGRAMS, 15)), claimed=[15], route="count")
    assert not result.accepted
    assert "150" in result.retry_feedback and "does not appear" in result.retry_feedback


def test_number_present_in_evidence_is_accepted():
    result = check("UoS offers 149 degree programs in total.", make_items((COUNCIL, 7), (PROGRAMS, 15)),
                   claimed=[15], route="count")
    assert result.accepted and result.pages == [15]


def test_valid_claimed_citation_is_kept():
    result = check("Faculty get thirty days of annual leave each academic year.", make_items((LEAVE, 3), (COUNCIL, 7)),
                   claimed=[3])
    assert result.accepted and result.pages == [3]


def test_invalid_claimed_citation_is_removed_and_answer_still_accepted():
    result = check("Faculty get thirty days of annual leave each academic year.", make_items((LEAVE, 3), (COUNCIL, 7)),
                   claimed=[7])
    assert result.accepted
    assert result.pages == [3]          # page 7 was claimed but does not support the answer


def test_no_pages_are_invented_when_nothing_supports_the_answer():
    result = check("Faculty receive a free parking permit.", make_items((LEAVE, 3), (COUNCIL, 7), (PROGRAMS, 15)))
    assert not result.accepted
    assert result.pages == []           # never "the first three pages"


def test_pages_only_ever_come_from_supporting_items():
    items = make_items((LEAVE, 3), (COUNCIL, 7), (PROGRAMS, 15), (CONTACT, 14))
    result = check("Faculty get thirty days of annual leave. UoS offers 149 degree programs.", items,
                   claimed=[3, 15, 99])
    assert result.accepted
    assert set(result.pages) == {3, 15}
    assert 99 not in result.pages and 7 not in result.pages and 14 not in result.pages


def test_multiple_evidence_pages_are_cited_claimed_first():
    items = make_items((LEAVE, 3), (PROGRAMS, 15))
    result = check("UoS offers 149 degree programs in total. Faculty get thirty days of annual leave.", items,
                   claimed=[15])
    assert result.accepted
    assert result.pages == [15, 3]


def test_cited_pages_are_capped():
    items = make_items((LEAVE, 3), (PROGRAMS, 15), (CONTACT, 14), (CALENDAR, 49), (COUNCIL, 7))
    answer = ("Faculty get thirty days of annual leave. UoS offers 149 degree programs. "
              "The phone number is +971 6 5050000. Classes begin on Mon 25 Aug. "
              "The University Council is chaired by the Chancellor.")
    result = check(answer, items)
    assert result.accepted
    assert len(result.pages) == verifier.MAX_CITED_PAGES


def test_mixed_answer_with_one_unsupported_sentence_is_rejected():
    result = check("Faculty members are entitled to thirty days of annual leave. "
                   "They also receive a housing allowance of 5000 dirhams.", make_items((LEAVE, 3)), claimed=[3])
    assert not result.accepted
    assert result.unsupported_claims == ["They also receive a housing allowance of 5000 dirhams."]
    assert "5000" in result.retry_feedback


def test_phone_number_claim_is_supported_by_the_row():
    result = check("The phone number is +971 6 5050000.", make_items((LEAVE, 3), (CONTACT, 14)), route="contact")
    assert result.accepted and result.pages == [14]


def test_fabricated_phone_and_email_are_rejected():
    items = make_items((CONTACT, 14))
    assert not check("The phone number is +971 6 5059999.", items, route="contact").accepted
    assert not check("Email helpdesk@sharjah.ac.ae.", items, route="contact").accepted
    assert check("Email itcenter@sharjah.ac.ae.", items, route="contact").accepted


def test_date_claims_check_day_month_and_number():
    items = make_items((CALENDAR, 49), (LEAVE, 3))
    assert check("Classes begin on Monday 25 August 2025.", items, route="date").accepted
    assert not check("Classes begin on Monday 26 August 2025.", items, route="date").accepted
    assert not check("Classes begin on Monday 25 September 2025.", items, route="date").accepted


def test_may_as_a_verb_is_not_a_month():
    result = check("Faculty may take thirty days of annual leave with dean approval.", make_items((LEAVE, 3)))
    assert result.accepted


def test_refusal_is_accepted_and_cites_nothing():
    result = check("I do not have this information.", make_items((LEAVE, 3)), claimed=[3])
    assert result.accepted and result.pages == [] and result.retry_feedback is None


def test_empty_answer_is_rejected():
    result = check("", make_items((LEAVE, 3)))
    assert not result.accepted and result.pages == [] and result.retry_feedback


def test_no_evidence_items_rejects():
    result = check("Faculty get thirty days of annual leave.", [])
    assert not result.accepted and result.pages == []


def test_bare_yes_without_a_supporting_statement_is_rejected():
    result = check("Yes.", make_items((LEAVE, 3)))
    assert not result.accepted
    assert "rule" in result.retry_feedback.lower()


def test_retry_rejection_carries_no_feedback():
    result = check("Faculty receive a free parking permit.", make_items((LEAVE, 3)), is_retry=True)
    assert not result.accepted and result.retry_feedback is None


def test_feedback_is_bounded_for_the_retry_prompt():
    answer = " ".join("Sentence %d is invented and mentions the value %d." % (i, 900 + i) for i in range(12))
    result = check(answer, make_items((LEAVE, 3)))
    assert not result.accepted
    assert len(result.retry_feedback) <= verifier.MAX_FEEDBACK_CHARS


def test_extractor_answer_built_from_a_row_is_accepted():
    result = check("Mon 25 Aug 02 Rabi I Classes begin.", make_items((CALENDAR, 49), (LEAVE, 3)),
                   claimed=[49], route="date", used_extractor=True)
    assert result.accepted and result.pages == [49]


def test_abbreviation_does_not_split_a_sentence():
    claims = verifier.split_claims("The model derives from Decision No. (3) of 2019. Faculty must apply.")
    assert claims == ["The model derives from Decision No. (3) of 2019.", "Faculty must apply."]


def test_uncovered_subquestions_are_reported():
    items = make_items((LEAVE, 3), (PROGRAMS, 15))
    subs = [SubQuestion("How many degree programs are offered?", "count"),
            SubQuestion("How much annual leave do faculty get?", "policy")]
    result = check("UoS offers 149 degree programs in total.", items, sub_questions=subs)
    assert result.accepted
    assert result.uncovered_subquestions == [1]


@pytest.mark.parametrize("text,expected", [
    ("Call +971 6 5050000 or (06) 505-0001.", ["97165050000", "065050001"]),
    ("classes begin 25 August 2025 (02 Rabi I)", []),          # a date, not a phone
])
def test_phone_extraction_ignores_date_fragments(text, expected):
    assert verifier._phones(text) == expected
