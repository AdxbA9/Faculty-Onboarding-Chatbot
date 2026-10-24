"""
Faculty Services & HR statement-level composite claim (Milestone 4, Step 4.12-C11, D-098): the predicate-statement reader
(clause boundaries, secondary predications, elided subjects, anchors, subjects, restrictions and hedges), the safe leading
frames of a question and the statement fallback.
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import claim_model as C
from handbook_bot.agents.specialists import faculty_services as F

AK, RK = C.AnchorKind, C.RestrictionKind

B1 = "Full-time faculty must take their annual leave in the summer, while part-time faculty may take their annual leave in the summer."
PROBATION = "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."
DEAN = ("The appointment of the Dean is subject to the decision of the Board of Trustees, based on the Chancellor’s recommendation, for a period of "
        "three years, renewable only once.")
CHAIR = ("The Chancellor appoints a faculty member preferably holding the rank of professor as department chair upon the recommendation of the Dean, "
         "for a period of three years, renewable normally only once.")
WITNESSES = ("The Faculty Grievance Committee may contact witnesses upon request, ensuring their appearance at the hearing, unless the Committee Chair "
             "determines it to be unreasonable or irrelevant to the issues at hand.")
SENIOR = ("Senior administrative appointments require Chancellor approval, except for Deans, Vice Chancellors, and the Chancellor, whose appointments "
          "are endorsed by the Chancellor and approved by the Board of Trustees and the President.")
CLINICAL = "The appointment of clinical faculty/academic non-faculty is limited to Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law."
NOTICE = "Faculty members intending to resign during probation must provide a six-month notice."
RESIGN = ("Faculty members intending to resign from UoS must submit their resignation to their respective college deans and through the HR portal at "
          "least six months before the effective date and must coincide with the end of an academic semester.")


def frame(question):
    claim = F.read_claim(question)
    assert claim is not None, question
    return F.read_claim_frame(question, claim)


def statements(question, unit):
    return F.read_clause_statements(unit, frame(question))


def kinds(statement):
    return {r.kind for r in statement.restrictions}


# ---------------------------------------------------------------------------
# Clause boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentence,clauses,elided", [
    (B1, ["Full-time faculty must take their annual leave in the summer", "part-time faculty may take their annual leave in the summer."], [False, False]),
    ("The Dean must approve the leave plan, and faculty members may comment on the leave plan.",
     ["The Dean must approve the leave plan", "faculty members may comment on the leave plan."], [False, False]),
    (CLINICAL, [CLINICAL], [False]),
    ("Faculty members must submit the request and must attach the medical report.",
     ["Faculty members must submit the request", "must attach the medical report."], [False, True]),
    ("Exceptions: the Dean may approve them.", ["Exceptions", "the Dean may approve them."], [False, False]),
    ("Faculty may apply in May; the committee meets in June.", ["Faculty may apply in May", "the committee meets in June."], [False, False]),
    ("Faculty and staff may attend the session.", ["Faculty and staff may attend the session."], [False]),
    ("The committee reviews the file and the Dean approves it.", ["The committee reviews the file and the Dean approves it."], [False]),
])
def test_finite_clauses_split_only_where_both_sides_carry_a_verb_group(sentence, clauses, elided):
    spans = F._clause_spans(sentence)
    assert [sentence[a:b].strip(" ,") for a, b, _e in spans] == [c.strip(" ,") for c in clauses]
    assert [e for _a, _b, e in spans] == elided


def test_an_enumeration_is_never_a_clause_break():
    assert len(F._clause_spans("Nominations can be made by department chairs, deans, or the faculty themselves.")) == 1


# ---------------------------------------------------------------------------
# Secondary predications, elided subjects, antecedents
# ---------------------------------------------------------------------------
def test_a_secondary_predication_belongs_to_its_statement_with_its_own_restrictions():
    s = [x for x in statements("Can the probation period be extended to two years?", PROBATION) if x.anchor.word.startswith("extend")]
    assert s and s[0].anchor.secondary
    assert {RK.SPECIAL, RK.GATE} <= kinds(s[0])
    assert "faculty probation period" in s[0].text and "two years" in s[0].text


def test_a_main_predicate_does_not_inherit_the_restrictions_of_its_secondary_predication():
    for s in statements("Is the probation period one year?", PROBATION):
        if not s.anchor.secondary:
            assert RK.SPECIAL not in kinds(s), s.predicate


def test_nursing_is_no_secondary_predication():
    s = statements("Is the appointment of clinical faculty limited to Medicine and Law?", CLINICAL)
    assert s and all(not x.anchor.secondary for x in s) and s[0].predicate.rstrip(".").endswith("Law")


def test_a_participle_before_the_verb_group_stays_in_the_subject():
    s = statements("Is the Department Hiring Committee formed each year?",
                   "The Department Hiring Committee, chaired by the department chair, is formed each year.")
    assert s and not s[0].anchor.secondary and "chaired by" in s[0].subject_text


def test_renewable_only_once_is_a_secondary_predication_of_the_dean_statement():
    s = [x for x in statements("Is the Dean's appointment renewable only once?", DEAN) if x.anchor.word == "renewable"]
    assert s and s[0].anchor.secondary and s[0].subject is None


def test_a_trailing_antecedent_after_a_secondary_predication_restricts_the_main_predicate():
    s = [x for x in statements("May the Faculty Grievance Committee contact witnesses upon request?", WITNESSES) if x.anchor.word == "contact"]
    assert s and RK.EXCEPTION in kinds(s[0])


def test_an_elided_clause_inherits_the_subject_of_the_previous_clause():
    s = [x for x in statements("Must the resignation coincide with the end of an academic semester?", RESIGN) if x.anchor.word == "coincide"]
    assert s and s[0].subject_inherited and s[0].subject_text.startswith("Faculty members intending to resign")


def test_a_sentence_initial_antecedent_conditions_every_later_clause():
    unit = "If the Dean approves, faculty members may carry over five days, and the department must record the decision."
    s = [x for x in statements("Must the department record the decision?", unit) if x.anchor.word == "record"]
    assert s and RK.SUFFICIENT in kinds(s[0])


def test_a_trailing_antecedent_conditions_its_own_clause_only():
    unit = "The Dean must approve the request, and faculty members may appeal if the request is denied."
    s = [x for x in statements("Must the Dean approve the request?", unit) if x.anchor.word == "approve"]
    assert s and RK.SUFFICIENT not in kinds(s[0])


# ---------------------------------------------------------------------------
# Hedges
# ---------------------------------------------------------------------------
def test_a_parenthetical_hedge_is_not_bound():
    s = statements("Must faculty members take leave at the end of the semester?", "Faculty members must take leave at the end of the semester (typically in June).")
    assert s and all(not x.hedges for x in s)


def test_a_hedge_on_a_noun_phrase_modifier_is_not_bound_and_a_hedge_on_the_slot_is():
    s = [x for x in statements("Is the department chair's appointment renewable only once?", CHAIR) if x.anchor.word == "renewable"]
    assert s and s[0].hedges == ("normally",)
    assert all("preferably" not in x.hedges for x in statements("Are department chairs appointed by the Chancellor?", CHAIR))


# ---------------------------------------------------------------------------
# Anchors and subjects
# ---------------------------------------------------------------------------
def test_the_asked_verb_is_a_verb_anchor_and_an_entitlement_word_an_entitlement_anchor():
    assert statements("Must faculty members submit the request?", "Faculty members must submit the request.")[0].anchor.kind is AK.VERB
    s = statements("Are faculty members eligible for the housing allowance?", "Faculty members are entitled to the housing allowance.")
    assert s and s[0].anchor.kind is AK.ENTITLEMENT


def test_the_subject_is_read_from_the_clause_start_not_from_before_an_object_anchor():
    s = statements("Must faculty on probation give six months' notice to resign?", NOTICE)
    assert s and all(x.anchor.word != "resign" for x in s)
    notice = [x for x in s if x.anchor.word == "notice"]
    assert notice and notice[0].anchor.kind is AK.OBJECT
    assert notice[0].subject is not None and notice[0].subject.text.startswith("faculty members intending to resign during probation")


def test_a_noun_shaped_occurrence_inside_the_subject_is_no_anchor():
    assert all(x.anchor.word != "appointment" for x in statements("Is the Dean's appointment renewable only once?", DEAN))


@pytest.mark.parametrize("question,unit,word", [
    ("Must faculty members resign during probation?", NOTICE, "resign"),                                                          # an infinitive
    ("Must faculty members apply for promotion?", "Faculty members applying for promotion must submit a portfolio.", "applying"),     # an -ing form
    ("Must the Dean approve the appointment?", "The approval of the Dean is required for the appointment.", "approval"),            # noun-shaped
])
def test_the_asked_verb_inside_the_subject_and_no_main_verb_is_no_anchor(question, unit, word):
    assert all(x.anchor.word != word for x in statements(question, unit))


def test_a_finite_active_verb_without_an_auxiliary_is_the_main_anchor():
    s = [x for x in statements("Do all senior administrative appointments require Chancellor approval?", SENIOR) if x.anchor.word == "require"]
    assert s and s[0].anchor.kind is AK.VERB and s[0].subject.text == "senior administrative appointments"
    assert s[0].member_exceptions and s[0].member_exceptions[0].startswith("except for Deans")


def test_at_most_three_anchors_per_clause():
    unit = "Faculty members submit the form, submit the report, submit the plan and submit the list."
    assert len(statements("Do faculty members submit the form?", unit)) <= 3


def test_split_binding_subjects_and_modalities_are_read_per_clause():
    s = statements("Must part-time faculty take their annual leave in the summer?", B1)
    assert [(x.subject.text, x.modality) for x in s] == [("full-time faculty", C.Modality.OBLIGATION), ("part-time faculty", C.Modality.PERMISSION)]


def test_statement_subjects_are_read_from_raw_text():
    s = statements("Is the appointment of clinical faculty limited to Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law?", CLINICAL)
    assert s and "faculty/academic" in s[0].subject_text


# ---------------------------------------------------------------------------
# Restrictions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,unit,kind", [
    ("Must all new faculty members complete the annual leave training?",
     "All new faculty members must complete the annual leave training only if they work in laboratories.", RK.NECESSARY),
    ("Is the Dean appointed for three years?", DEAN, RK.GATE),
    ("Is renewal for the second year granted?", "Renewal for the second year is subject to department needs and successful annual evaluations.",
     RK.NECESSARY),
    ("Can additional sick leave be granted without pay?",
     "In special circumstances, up to an additional 30 days of sick leave without pay may be granted.", RK.SPECIAL),
    ("Can additional sick leave be granted without pay?",
     "In special circumstances, up to an additional 30 days of sick leave without pay may be granted.", RK.POSSIBILITY),
    ("Is annual leave approved in January?", "Unless the Dean objects, annual leave is approved in January.", RK.EXCEPTION),
    ("Can the probation period be extended?", PROBATION, RK.GATE),
])
def test_restrictions_are_bound_to_the_predicate(question, unit, kind):
    assert any(kind in kinds(s) for s in statements(question, unit)), [s.restrictions for s in statements(question, unit)]


# ---------------------------------------------------------------------------
# Safe leading frames (claim side only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,stripped", [
    ("According to HR policy, does the Chancellor approve termination for cause?", "does the Chancellor approve termination for cause?"),
    ("Under the HR policy, does the Dean approve termination during probation?", "does the Dean approve termination during probation?"),
    ("At the department level, must the committee meet in September?", "must the committee meet in September?"),
    ("In practice, may faculty members carry over leave?", "may faculty members carry over leave?"),
    ("Per the Employee Handbook, is settling in the UAE supported by HR?", "is settling in the UAE supported by HR?"),
    ("In general, are faculty eligible for the allowance?", "In general, are faculty eligible for the allowance?"),
    ("Typically, are faculty eligible for the allowance?", "Typically, are faculty eligible for the allowance?"),
    ("According to all HR policies, does the Dean approve leave?", "According to all HR policies, does the Dean approve leave?"),
    ("At the department level, the committee meets in September.", "At the department level, the committee meets in September."),
])
def test_only_a_safe_leading_frame_is_stripped(question, stripped):
    assert F.strip_leading_frame(question) == stripped


def test_a_framed_question_is_read_as_a_claim_only_with_the_frame_switch(monkeypatch):
    q = "According to HR policy, does the Chancellor approve termination for cause?"
    assert F.read_claim(q) is not None and F.read_claim_frame(q).frame == "according to hr policy"
    monkeypatch.setattr(F, "_C11_FRAMES", False)
    assert F.read_claim(q) is None


# ---------------------------------------------------------------------------
# Statement fallback
# ---------------------------------------------------------------------------
def test_with_no_anchor_the_clause_with_the_most_claim_content_is_a_fallback_statement():
    s = statements("Is the probation period one year?", PROBATION)
    assert len(s) == 1 and s[0].anchor.kind is AK.FALLBACK
    assert RK.SPECIAL not in kinds(s[0])                    # the fallback predicate is the main part, not the secondary predication


def test_with_no_shared_content_there_is_no_statement():
    assert statements("Is the probation period one year?", "The library opens at eight.") == ()
