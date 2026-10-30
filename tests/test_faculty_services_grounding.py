"""
Faculty Services & HR grounding: roles, concepts, ask shapes, approval direction, category binding, leave-type and
pay-status binding, numeric facts, money ranges and the contract / probation values (Milestone 4, Step 4.11; D-086,
D-087).
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus

from faculty_services_support import P61, P62, P63, P64, P64_APPT, P65, P106, P107, P108, P148, P149, P150, PAGES, ScriptedReranker, corpus_from, make_specialist, registry_and_map, run

SUPPORTED, PARTIAL = FindingStatus.SUPPORTED, FindingStatus.PARTIAL


@pytest.fixture(scope="module")
def rm():
    return registry_and_map()


@pytest.fixture(scope="module")
def sp(rm):
    return make_specialist(corpus_from(rm, PAGES))


@pytest.fixture(scope="module")
def high(rm):
    return make_specialist(corpus_from(rm, PAGES), ScriptedReranker([], default=9.0))


def status(specialist, question):
    return run(specialist, question).status


# ---------------------------------------------------------------------------
# 1. Roles and concepts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,role", [
    ("the Vice Chancellor for Academic Affairs", "vcaa"), ("the VCAA", "vcaa"), ("the Vice Chancellor", "vice_chancellor"), ("the Chancellor", "chancellor"),
    ("the college dean", "dean"), ("the department chair", "chair"), ("HR", "hr"), ("Human Resources", "hr"), ("the Passport Unit", "passport_unit"),
    ("the Services Department", "services_department"), ("the central hiring and evaluation committee", "hiring_committee"),
    ("the Department Academic Promotions Committee", "promotion_committee"), ("the College Council", "college_council"), ("the Board of Trustees", "board_of_trustees"),
    ("academic administrators", "administration"), ("clinical faculty", "clinical_faculty"), ("emeritus faculty", "emeritus_faculty"),
    ("visiting faculty", "visiting_faculty"), ("part-time instructors", "part_time_faculty"), ("faculty members", "faculty"), ("students", "student"),
])
def test_roles_are_distinct(text, role):
    assert F.ASK_GATE.lexicon.roles_in(text) == frozenset({role}), text


def test_authority_roles_never_collapse():
    lex = F.ASK_GATE.lexicon
    assert lex.roles_in("the dean and the chair") == {"dean", "chair"}
    assert lex.roles_in("the Vice Chancellor and the Chancellor") == {"vice_chancellor", "chancellor"}
    assert lex.roles_in("the VCAA and the Vice Chancellor") == {"vcaa", "vice_chancellor"}


@pytest.mark.parametrize("text,anchor", [
    ("sick leave", "sick_leave"), ("maternity leave", "maternity_leave"), ("unpaid special leave", "special_leave"), ("Hajj leave", "hajj_leave"),
    ("vacation days", "vacation_leave"), ("end-of-service", "end_of_service"), ("clinical allowance", "clinical_allowance"), ("furniture allowance", "furniture_allowance"),
    ("children's education allowance", "education_allowance"), ("annual return air ticket", "air_ticket"), ("health insurance", "insurance"), ("housing", "housing"),
    ("basic salary", "salary"), ("monetary benefits", "benefit"), ("duplication of benefits", "duplication"), ("compensation for administrative positions", "admin_compensation"),
    ("annual incentive", "incentive"), ("demise policy", "demise"), ("probation period", "probation"), ("resignation", "resignation"), ("termination", "termination"),
    ("contract renewal", "contract_renewal"), ("employment offer", "contract"), ("confirmation", "confirmation"), ("faculty appointment", "appointment"),
    ("nepotism", "nepotism"), ("Emirati", "emirati"), ("emeritus", "emeritus"), ("dual career", "dual_career"), ("performance evaluation", "evaluation"),
    ("Faculty Information System", "fis"), ("promotion", "promotion"), ("dossier", "dossier"), ("appeal", "appeal"), ("grievance", "grievance"),
    ("disciplinary action", "discipline"), ("professional ethics", "ethics"), ("sexual harassment", "harassment"), ("entry visa", "visa"), ("employee file", "employee_file"),
    ("child care center", "child_care"), ("medical clinics", "clinic"), ("human resources", "hr"), ("associate professor", "rank_associate_professor"),
])
def test_hr_concepts_and_aliases(text, anchor):
    assert anchor in F.concepts(text).anchors, text


def test_generic_words_carry_no_concept():
    for word in sorted(F.GENERIC_TERMS):
        assert not F.concepts(word).vocab, word


def test_relevance_needs_an_anchor_or_two_vocabulary_concepts():
    clause = F.concepts("Is the furniture allowance AED 30,000?")
    assert F.is_relevant(clause, "A one-time furniture allowance of 30,000 Dirhams is provided.")
    assert not F.is_relevant(clause, "The University of Sharjah aims to attract and retain top talent.")


# ---------------------------------------------------------------------------
# 2. Ask shapes and the bogus-relation guard
# ---------------------------------------------------------------------------
def test_number_words_and_trailing_currency_are_normalised_for_the_parser():
    assert F.ask_text("Is maternity leave sixty calendar days with full pay?") == "Is maternity leave 60 calendar days with full pay?"
    assert F.ask_text("Is the salary 43,860 AED?") == "Is the salary 43,860?"
    assert F.ask_text("a definite term contract") == "a definite contract"


def test_copular_number_asks_carry_no_relation():
    for q in ("Is maternity leave sixty calendar days with full pay?", "Is the children's education allowance up to 25,000 Dirhams per child?",
              "Is the monthly compensation of a director of a unit AED 2,000 to AED 3,000?", "Is the probation period one year?"):
        a = F.assess_ask(q)
        assert a.shape == F.ASK_RECOGNISED and a.relation is None, q


def test_verb_asks_keep_their_relation():
    a = F.assess_ask("Does the Chancellor approve termination for cause?")
    assert a.relation is not None and a.relation.actor == "chancellor" and a.relation.action == "approve"
    a = F.assess_ask("Who approves additional maternity leave without pay?")
    assert a.relation is not None and a.relation.party is True


def test_unknown_specific_ask_gets_the_unknown_detail():
    a = F.assess_ask("Which?")
    assert a.shape == F.ASK_UNKNOWN and [d.family for d in a.details] == ["ask_unknown"]


# ---------------------------------------------------------------------------
# 3. Approval direction (D-080, D-081 reused)
# ---------------------------------------------------------------------------
def test_termination_approval_chain_direction(rm):
    sp = make_specialist(corpus_from(rm, [P63, P64]))
    assert status(sp, "Does the Chancellor approve termination of a contract for cause?") is SUPPORTED
    assert status(sp, "Does the Chancellor recommend termination for cause?") is PARTIAL
    assert status(sp, "Does the dean approve termination during the probation period?") is SUPPORTED
    assert status(sp, "Does the department chair approve termination during probation?") is PARTIAL
    assert status(sp, "Does the Registrar approve termination during probation?") is PARTIAL


def test_maternity_approval_names_the_vcaa_not_the_dean(rm):
    sp = make_specialist(corpus_from(rm, [P107]))
    assert status(sp, "Does the dean approve additional maternity leave without pay?") is PARTIAL
    # the object stands more than nine tokens before the passive verb (the shared D-081 patient window): a documented
    # conservative miss, never a wrong actor
    assert status(sp, "Does the Vice Chancellor for Academic Affairs approve additional maternity leave?") is not SUPPORTED


def test_authority_detail_needs_an_approval_verb_and_a_named_authority():
    detail = next(d for d in F.requested_details("Who approves termination for cause?") if d.family == "authority")
    assert detail.check("Termination of faculty contracts for cause requires the approval of the Chancellor.")
    assert not detail.check("Faculty members intending to resign must do so at the end of an academic semester.")
    assert not detail.check("Faculty are entitled to sick leave with full pay for up to 15 working days.")


def test_a_probation_stage_rule_settles_no_unscoped_fact_ask():
    probation = "Faculty members intending to resign during probation must provide a six-month notice."
    assert not F.hr_fact_detail("Must I give six months' notice when I resign?").check(probation)
    assert F.hr_fact_detail("Must I give six months' notice when I resign during probation?").check(probation)
    assert not F.hr_fact_detail("Are new faculty entitled to annual leave?").check(
        "New faculty members who resign during probation are not entitled to benefits such as annual leave and ticket allowances.")
    assert F.hr_fact_detail("Is the probation period one year?").check(
        "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval.")


def test_joining_formalities_keep_their_actors(rm):
    sp = make_specialist(corpus_from(rm, [P61]))
    assert status(sp, "Does HR assist new faculty members in obtaining entry visas through the Passport Unit?") is SUPPORTED
    assert status(sp, "Does the dean arrange entry visas for new faculty?") is PARTIAL


# ---------------------------------------------------------------------------
# 4. Category binding (D-086)
# ---------------------------------------------------------------------------
def test_part_time_never_inherits_a_full_time_fact(rm):
    sp = make_specialist(corpus_from(rm, [P106, P107]))
    assert status(sp, "Do part-time faculty receive health insurance for their spouse?") is PARTIAL
    assert status(sp, "Is maternity leave sixty calendar days with full pay for part-time faculty?") is PARTIAL
    assert status(sp, "Is maternity leave sixty calendar days with full pay for full-time faculty?") is SUPPORTED


def test_new_faculty_rule_does_not_settle_a_confirmed_faculty_ask(rm):
    sp = make_specialist(corpus_from(rm, [P63]))
    assert status(sp, "Are new faculty members who are terminated during probation entitled to return tickets?") is SUPPORTED
    assert status(sp, "Are visiting faculty entitled to return tickets when terminated?") is PARTIAL


def test_emirati_track_reduction_keeps_its_value_and_category(rm):
    sp = make_specialist(corpus_from(rm, [P65]))
    assert status(sp, "Are faculty on the Assistant Professor Emirati Track entitled to a teaching load reduction of 6 credit hours in the first year of appointment?") is SUPPORTED
    assert status(sp, "Are faculty on the Assistant Professor Emirati Track entitled to a teaching load reduction of 9 credit hours in the first year of appointment?") is PARTIAL


# ---------------------------------------------------------------------------
# 5. Leave-type and pay-status binding (D-086)
# ---------------------------------------------------------------------------
def test_leave_types_and_pay_statuses_are_read():
    assert F.leave_types_of("sick leave with full pay") == {"sick"} and F.pay_statuses_of("sick leave with full pay") == {"full"}
    assert F.leave_types_of("additional maternity leave without pay") == {"maternity"} and F.pay_statuses_of("without pay") == {"unpaid"}
    assert F.pay_statuses_of("half pay for an additional 15 working days") == {"half"}
    assert F.leave_types_of("their first Hajj may receive twenty days of paid leave") == {"hajj"}


def test_leave_durations_never_cross_types(rm):
    sp = make_specialist(corpus_from(rm, [P107]))
    assert status(sp, "Am I entitled to 15 days of sick leave with full pay?") is SUPPORTED
    assert status(sp, "Am I entitled to 15 days of maternity leave?") is PARTIAL
    assert status(sp, "Is maternity leave sixty calendar days with full pay?") is SUPPORTED
    assert status(sp, "Am I entitled to 60 days of Hajj leave?") is PARTIAL
    assert status(sp, "Can Muslim faculty receive up to twenty calendar days of paid leave for their first Hajj?") is SUPPORTED
    assert status(sp, "Can faculty receive twenty days of sick leave for Hajj?") is PARTIAL


def test_pay_status_binds_to_its_own_number(rm):
    sp = make_specialist(corpus_from(rm, [P107]))
    assert status(sp, "Is sick leave with half pay 15 working days?") is SUPPORTED
    assert status(sp, "Is sick leave without pay 15 working days?") is PARTIAL
    assert status(sp, "Can up to 30 days of sick leave without pay be granted?") is SUPPORTED
    assert status(sp, "Is maternity leave 30 days with full pay?") is PARTIAL
    assert status(sp, "Can additional maternity leave without pay of up to thirty days be approved?") is SUPPORTED


# ---------------------------------------------------------------------------
# 6. Money, ranges and qualifiers (D-087)
# ---------------------------------------------------------------------------
def test_money_facts_read_amounts_ranges_and_percentages():
    facts = F.money_facts(F._hr_norm("A one-time furniture allowance of 30,000 Dirhams; AED 2,000 to AED 3,000; 10% to 20% of the basic salary; 25,000 Dirhams per child"))
    assert [(f["kind"], f["low"], f["high"]) for f in facts] == [("aed", 30000.0, None), ("aed", 2000.0, 3000.0), ("percent", 10.0, 20.0), ("aed", 25000.0, None)]


def test_amounts_bind_to_their_allowance(rm):
    sp = make_specialist(corpus_from(rm, [P106]))
    assert status(sp, "Is the furniture allowance AED 30,000?") is SUPPORTED
    assert status(sp, "Is the furniture allowance AED 25,000?") is PARTIAL
    assert status(sp, "Is the children's education allowance up to 25,000 Dirhams per child?") is SUPPORTED
    assert status(sp, "Is the education allowance 75,000 Dirhams per child?") is PARTIAL
    assert status(sp, "Is the total children's education allowance 25,000 Dirhams?") is PARTIAL
    # D-098 (Step 4.12-C11): the maximum is stated for "all qualifying children"; an explicit universal over a narrower noun
    # phrase does not state "all children" (a pre-classified conservative loss), while the exact noun phrase stays supported.
    assert status(sp, "Is the maximum children's education allowance 75,000 Dirhams for all children?") is PARTIAL
    assert status(sp, "Is the maximum children's education allowance 75,000 Dirhams for all qualifying children?") is SUPPORTED


def test_one_time_never_matches_annual(rm):
    sp = make_specialist(corpus_from(rm, [P106]))
    assert status(sp, "Is the furniture allowance an annual 30,000 Dirhams?") is PARTIAL
    assert status(sp, "Is the furniture allowance a one-time 30,000 Dirhams?") is SUPPORTED


def test_a_percentage_range_is_one_fact(rm):
    sp = make_specialist(corpus_from(rm, [P106]))
    assert status(sp, "Is the clinical allowance 10 to 20 percent of the basic salary?") is SUPPORTED
    assert status(sp, "Is the clinical allowance 20 percent of the basic salary?") is PARTIAL
    assert status(sp, "Is the clinical allowance up to 20 percent of the basic salary?") is SUPPORTED


# ---------------------------------------------------------------------------
# 7. Contract and probation values never cross (D-087)
# ---------------------------------------------------------------------------
def test_probation_and_extension(rm):
    sp = make_specialist(corpus_from(rm, [P62, P63]))
    assert status(sp, "Is the probation period one year?") is SUPPORTED
    assert status(sp, "Is the probation period two years?") is PARTIAL
    assert status(sp, "Can the probation period be extended to a maximum of two years?") is SUPPORTED
    assert status(sp, "Can the probation period be extended to a maximum of three years?") is PARTIAL


def test_contract_durations_and_renewals(rm):
    sp = make_specialist(corpus_from(rm, [P62, P63]))
    assert status(sp, "Is the initial contract for regular faculty a two-year offer?") is SUPPORTED
    assert status(sp, "May the contract be renewed for three years upon successful annual evaluations?") is SUPPORTED
    assert status(sp, "Is the first renewal of a regular faculty contract for four years?") is PARTIAL
    assert status(sp, "Is the subsequent renewal for four years?") is SUPPORTED
    assert status(sp, "Is the initial contract for regular faculty a four-year offer?") is PARTIAL


def test_notice_and_deduction(rm):
    sp = make_specialist(corpus_from(rm, [P63]))
    # the six-month notice is stated for resignation during probation ("must provide a six-month notice") and in general
    # ("at least six months before the effective date"): the probation sentence settles only an ask that names probation
    assert status(sp, "Must I give six months' notice when I resign during probation?") is SUPPORTED
    assert status(sp, "Must I give three months' notice when I resign during probation?") is PARTIAL
    assert status(sp, "Must I give six months' notice when I resign?") is PARTIAL
    assert status(sp, "Must I give three months' notice when I resign?") is PARTIAL
    assert status(sp, "Can failure to comply result in a 50 percent deduction from the end of service compensation?") is SUPPORTED
    assert status(sp, "Can failure to comply result in a 25 percent deduction from the end of service compensation?") is PARTIAL


def test_offer_acceptance_deadline(rm):
    sp = make_specialist(corpus_from(rm, [P61]))
    assert status(sp, "Is an offer acceptance submitted within 10 days?") is SUPPORTED
    assert status(sp, "Is an offer acceptance submitted within 30 days?") is PARTIAL


# ---------------------------------------------------------------------------
# D-088 (Step 4.12-C1): tranche qualifiers, the fact's subject, the demanded duration anchors
# ---------------------------------------------------------------------------
def test_a_base_ask_is_never_settled_by_a_tranche_number():
    d = F.hr_fact_detail
    assert not d("Is sick leave 10 working days?").check("Faculty receive an additional 10 working days of sick leave.")
    assert d("Is sick leave 10 working days?").check("Faculty receive 10 working days of sick leave.")
    assert d("Is additional sick leave 10 working days?").check("Faculty receive an additional 10 working days of sick leave.")
    assert not d("Is Hajj leave twenty calendar days?").check("Muslim faculty may receive up to twenty calendar days of Hajj leave.")
    assert d("Is Hajj leave up to twenty calendar days?").check("Muslim faculty may receive up to twenty calendar days of Hajj leave.")
    assert not d("Is the notice six months?").check("Terminated faculty members receive a minimum six-month notice.")
    assert d("Is the notice at least six months?").check("Terminated faculty members receive a minimum six-month notice.")
    assert not d("Is maternity leave thirty days?").check("Maternity leave without pay of thirty days may be approved.")
    assert d("Is unpaid maternity leave thirty days?").check("Maternity leave without pay of thirty days may be approved.")
    assert d("Is maternity leave sixty days?").check("Faculty are entitled to sixty days of maternity leave with full pay.")


def test_the_stated_number_must_measure_the_asked_subject():
    probation = "Faculty members intending to resign during probation must provide a six-month notice."
    one_year = "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."
    assert not F.hr_fact_detail("Is the probation period six months?").check(probation)
    assert F.hr_fact_detail("Must I give a six-month notice when I resign during probation?").check(probation)
    assert not F.hr_fact_detail("Is the notice period one year?").check(one_year)
    assert F.hr_fact_detail("Is the probation period one year?").check(one_year)
    assert not F.hr_fact_detail("Is the contract period one year?").check(one_year)


def test_notice_contract_and_appointment_are_demanded_subjects(rm, sp, high):
    assert {"notice", "contract", "appointment"}.isdisjoint(F._FACT_FREE_ANCHORS)
    for s in (sp, high):
        assert status(s, "Is the notice period during probation one year?") is not SUPPORTED
        assert status(s, "Is the contract during probation one year?") is not SUPPORTED
        assert status(s, "Is the appointment during probation one year?") is not SUPPORTED
        assert status(s, "Must I give a six-month notice when I resign during probation?") is SUPPORTED
        assert status(s, "Is a visiting appointment a definite term contract of up to two years?") is SUPPORTED
    # the appointment term is demanded and found: the nominalised "appointment" builds no relation any more
    appointment = "2.9.1.2 Initial Faculty Appointment: The first faculty appointment on the Assistant Professor Emirati Track shall be for a duration of three years."
    assert F.hr_fact_detail("Is the first appointment on the Emirati track for three years?").check(appointment)
    assert not F.hr_fact_detail("Is the first appointment on the Emirati track for two years?").check(appointment)
    for rr in (None, ScriptedReranker([], default=9.0)):
        s = make_specialist(corpus_from(rm, [P64_APPT]), rr)
        assert status(s, "Is the first appointment on the Emirati track for three years?") is SUPPORTED
        assert status(s, "Is the first appointment on the Emirati track for two years?") is PARTIAL


def test_probation_and_notice_matrix(sp, high):
    for s in (sp, high):
        assert status(s, "Is the probation period one year?") is SUPPORTED
        assert status(s, "Can the probation period be extended to a maximum of two years?") is SUPPORTED
        assert status(s, "Must I give a six-month notice when I resign during probation?") is SUPPORTED
        assert status(s, "Is the probation period six months?") is PARTIAL
        assert status(s, "Is the notice period one year?") is PARTIAL
        assert status(s, "Is the notice period two years?") is not SUPPORTED
        assert status(s, "Is the probation period two years?") is PARTIAL


def test_maternity_matrix(sp, high):
    for s in (sp, high):
        assert status(s, "Is maternity leave sixty calendar days with full pay?") is SUPPORTED
        assert status(s, "Is maternity leave sixty calendar days?") is SUPPORTED
        assert status(s, "Is maternity leave thirty days?") is PARTIAL
        assert status(s, "Is maternity leave thirty calendar days?") is PARTIAL
        assert status(s, "Is additional maternity leave without pay up to a maximum of thirty days?") is SUPPORTED
        assert status(s, "Is maternity leave without pay up to thirty additional days?") is SUPPORTED
        assert status(s, "Is Hajj leave sixty calendar days?") is not SUPPORTED
        assert status(s, "Is maternity leave 15 working days?") is not SUPPORTED


def test_sick_leave_matrix(sp, high):
    for s in (sp, high):
        assert status(s, "Is sick leave with full pay 15 working days?") is SUPPORTED
        assert status(s, "Is sick leave with half pay 15 working days?") is SUPPORTED
        assert status(s, "Is additional sick leave without pay up to 30 days?") is PARTIAL   # D-099: only in special circumstances
        assert status(s, "Is sick leave 30 days?") is PARTIAL
        assert status(s, "Is sick leave thirty working days?") is not SUPPORTED
        assert status(s, "Is maternity leave 30 days?") is not SUPPORTED
        assert status(s, "Is Hajj leave 30 days?") is not SUPPORTED


def test_a_nominal_action_or_a_pay_noun_builds_no_relation():
    assert F.assess_ask("Is the first appointment on the Emirati track for three years?").relation is None
    assert F.assess_ask("Is additional maternity leave without pay up to a maximum of thirty days?").relation is None
    assert F.assess_ask("Does the Chancellor approve termination for cause?").relation is not None


def test_hr_verbs_keep_their_relation_and_a_dropped_relation_keeps_its_roles(sp, high):
    assert F._real_verb("chaired") and F._real_verb("formed") and F.assess_ask("Is the first appointment on the Emirati track for three years?").relation is None
    chaired = F.assess_ask("Is the promotion committee at the department level chaired by the Chancellor?")
    assert chaired.relation is not None and chaired.relation.actor == "chancellor"
    bogus = F.assess_ask("Does the Chancellor approve additional maternity leave without pay of up to thirty days?")
    assert bogus.relation is None and "chancellor" in bogus.roles
    for s in (sp, high):
        assert status(s, "Does the Chancellor approve additional maternity leave without pay of up to thirty days?") is not SUPPORTED
        assert status(s, "Does the Dean approve additional maternity leave without pay of up to thirty days?") is not SUPPORTED
        assert status(s, "Does the Vice Chancellor for Academic Affairs approve additional maternity leave without pay of up to thirty days?") is SUPPORTED


# ---------------------------------------------------------------------------
# D-089 (Step 4.12-C2): named roles stay demanded; an HR-built relation binds the actor when the parser builds none
# ---------------------------------------------------------------------------
def test_named_roles_are_preserved_without_a_relation():
    assert F._named_roles("Does the Chancellor form the College Academic Promotions Committee?") >= {"chancellor", "promotion_committee"}
    assert "faculty" not in F._named_roles("Are faculty entitled to sick leave?")
    a = F.assess_ask("Does the Chancellor form the College Academic Promotions Committee?")
    assert "chancellor" in a.roles and a.relation is not None and a.relation.actor == "chancellor" and a.relation.action == "form"
    a = F.assess_ask("Is the dean a full professor?")
    assert "dean" in a.roles and a.relation is None
    rel = F._hr_relation("Does the department chair chair the College Academic Promotions Committee?")
    assert rel is not None and rel.actor == "chair" and rel.action == "chair" and "committee" in rel.obj
    assert F._hr_relation("Is the probation period one year?") is None


def test_form_and_chair_collisions_bind_the_actor_locally():
    college = "In September of each year, the College Council shall form the College Academic Promotions Committee, chaired by the college dean if a full professor."
    grounding = lambda q: next(d for d in F.assess_ask(q).details if d.family == "grounding")
    assert grounding("Does the College Council form the College Academic Promotions Committee?").check(college)
    assert not grounding("Does the Chancellor form the College Academic Promotions Committee?").check(college)
    assert not grounding("Does the college dean form the College Academic Promotions Committee?").check(college)
    assert not grounding("Does the Deans Council form the College Academic Promotions Committee?").check(college)
    outside = "The college dean, in consultation with the department chair, shall select faculty members from outside the department to form a committee of at least three members."
    assert not grounding("Does the department chair form the College Academic Promotions Committee?").check(outside)
    assert not grounding("Does the Chancellor form the College Academic Promotions Committee?").check(outside)
    chaired = "The College Academic Promotions Committee is chaired by the college dean, in consultation with the department chair."
    assert not grounding("Does the department chair chair the College Academic Promotions Committee?").check(chaired)
    assert grounding("Does the college dean chair the College Academic Promotions Committee?").check(chaired)
    # a verb-less ask ("is the dean a full professor": the parser reads "a" as the action) keeps its named role demanded
    assert grounding("Is the dean a full professor?").check("Individuals appointed as Deans must hold the rank of full professor.")
    assert not grounding("Is the dean a full professor?").check("Emeritus status is conferred upon retired faculty at the rank of Professor who have demonstrated exemplary service.")
    # a navigation half names a place, not an actor: the policy half is still answered from the probation text
    assert "hr" not in F._named_roles("What is the probation period and where is the HR office?")
    assert "hr" in F._named_roles("Does HR assist new faculty members in obtaining entry visas?")


def test_actor_matrix_on_the_promotion_pages(rm):
    sp = make_specialist(corpus_from(rm, [P64, P148, P149, P150]), ScriptedReranker([], default=9.0))
    # positives: the real actor, action and object
    assert status(sp, "Does the College Council form the College Academic Promotions Committee?") is SUPPORTED
    assert status(sp, "Is the University Central Committee for the Promotion of Faculty Members chaired by the Chancellor or a delegate?") is SUPPORTED
    assert status(sp, "Does the Chancellor approve termination for cause?") is SUPPORTED
    assert status(sp, "Does the department chair present promotion applications to the Department Academic Promotions Committee?") is SUPPORTED
    assert status(sp, "Does the Department Academic Promotions Committee review the promotion applications?") is SUPPORTED
    # swapped actors, bodies and chairs
    for q in ("Does the Chancellor form the College Academic Promotions Committee?", "Does the department chair form the College Academic Promotions Committee?",
              "Does the college dean form the College Academic Promotions Committee?", "Does the Dean form the Department Academic Promotions Committee?",
              "Does the Deans Council form the College Academic Promotions Committee?", "Does the department chair chair the College Academic Promotions Committee?",
              "Does the Dean chair the University Central Committee for the Promotion of Faculty Members?",
              "Does the Vice Chancellor for Academic Affairs approve termination for cause?",
              "Does the Department Academic Promotions Committee forward the promotion decision to the Board of Trustees?",
              "Does the Chancellor present promotion applications to the Department Academic Promotions Committee?"):
        assert status(sp, q) is not SUPPORTED, q


def test_the_promotion_committees_are_distinct_bodies(rm):
    assert F._bodies_of("the Department Academic Promotions Committee and the University Central Committee for Promotion") == {"department", "university"}
    assert F._bodies_of("the College Academic Promotions Committee") == {"college"}
    sp = make_specialist(corpus_from(rm, [P148, P149, P150]), ScriptedReranker([], default=9.0))
    # positives: the committee in its own place
    assert status(sp, "Is the College Academic Promotions Committee formed by the College Council?") is SUPPORTED
    assert status(sp, "Is the University Central Committee for the Promotion of Faculty Members formed and chaired by the Chancellor or a delegate?") is SUPPORTED
    assert status(sp, "Does the department chair present promotion applications to the Department Academic Promotions Committee?") is SUPPORTED
    # another committee's text never settles the asked one, as object, target or actor
    for q in ("Is the College Academic Promotions Committee formed by the Chancellor?", "Is the Department Academic Promotions Committee chaired by the college dean?",
              "Does the department chair present promotion applications to the College Academic Promotions Committee?",
              "Is the Department Academic Promotions Committee formed by the College Council?",
              "Does the College Academic Promotions Committee present promotion applications to the department chair?"):
        assert status(sp, q) is not SUPPORTED, q


# ---------------------------------------------------------------------------
# D-091 (Step 4.12-C4): sameness and independence claims need stated universality
# ---------------------------------------------------------------------------
def test_sameness_frames_are_detected():
    for q in ("Is the basic salary the same for every rank?", "Is salary uniform across all colleges?", "Is the leave entitlement equal for all faculty?",
              "Do all appointment categories have the same contract duration?", "Is the salary scale identical for all specializations?"):
        assert F.universality_frame(q) == "sameness", q
    assert F.universality_frame("Is the deduction equal to 50 percent?") is None


def test_independence_frames_are_detected():
    for q in ("Is the salary scale independent of specialization?", "Is salary paid regardless of rank?", "Does salary not depend on rank?",
              "Is the allowance irrespective of nationality?"):
        assert F.universality_frame(q) == "independence", q
    assert F.universality_frame("Is the probation period one year?") is None


def test_a_based_on_sentence_settles_no_sameness_or_independence():
    based = "The University of Sharjah offers a competitive basic salary based on faculty rank and specialization, following the salary scale below:"
    assert not F.universality_detail("Is the basic salary the same for every specialization?").check(based)
    assert not F.universality_detail("Is the salary scale independent of specialization?").check(based)
    stated = "This policy applies to all members of the University community, regardless of gender."
    assert F.universality_detail("Does the policy apply regardless of gender?").check(stated)
    assert F.universality_detail("Does the policy apply the same to all members?").check(stated)


def test_universality_claims_are_never_supported_from_topic_overlap(rm, sp, high):
    salary = make_specialist(corpus_from(rm, [P106]), ScriptedReranker([], default=9.0))
    for q in ("Is the salary scale the same for all specializations?", "Is the basic salary the same for every specialization?", "Is the basic salary the same for every rank?",
              "Is the salary scale independent of specialization?", "Do all faculty receive the same salary?", "Is salary uniform across all colleges?"):
        assert status(salary, q) is not SUPPORTED, q
    for s in (sp, high):
        for q in ("Is the probation period the same for all faculty?", "Is the leave entitlement equal for all faculty?",
                  "Do all appointment categories have the same contract duration?", "Do all deans receive the same compensation?"):
            assert status(s, q) is not SUPPORTED, q
        assert status(s, "Is the probation period one year?") is SUPPORTED


# ---------------------------------------------------------------------------
# D-092 (Step 4.12-C5): sameness is bound to the asked subject and comparison
# ---------------------------------------------------------------------------
def test_universal_ask_names_the_compared_subject_and_the_comparison():
    cases = {"Are promotion requirements the same for every rank?": ({"promotion", "requirement"}, {"rank"}),
             "Is the promotion procedure the same at every college?": ({"promotion", "procedure"}, {"college"}),
             "Are the promotion procedures the same at department and college level?": ({"promotion", "procedure"}, {"department", "college", "level"}),
             "Is the promotion committee membership the same for every department?": ({"promotion", "committee", "membership"}, {"department"}),
             "Do all colleges follow the same promotion procedure?": ({"promotion", "procedure"}, {"college"}),
             "Is the evaluation process the same for clinical and regular faculty?": ({"evaluation", "process"}, {"clinical", "regular", "faculty"}),
             "Do all ranks receive equal pay?": ({"pay"}, {"rank"}),
             "Does the policy apply regardless of gender?": ({"policy"}, {"gender"})}
    for q, (subject, comparison) in cases.items():
        ask = F.universal_ask(q)
        assert (set(ask.subject), set(ask.comparison)) == (subject, comparison), q
    assert F.universal_ask("Is the probation period one year?") is None
    assert F.universality_frame("Do all ranks receive equal pay?") == "sameness"
    assert F.universality_frame("Is the deduction equal to 50 percent?") is None


def test_bare_equal_is_no_sameness_evidence():
    for text in ("a rank equal to or higher than the desired rank", "principles of fairness, impartiality, and equal opportunity",
                 "a lump sum equal to 5% of the annual salary"):
        assert F._SAMENESS_EVIDENCE.search(text) is None, text
    for text in ("The allowance is paid equally for all faculty.", "The policy is applied equally to all members.", "The load is equal across all ranks."):
        assert F._SAMENESS_EVIDENCE.search(text) is not None, text
    # a threshold that names the subject and the comparison still states no sameness
    threshold = "Promotion requirements for every rank are equal to or higher than the departmental minimum."
    assert not F.universality_detail("Are promotion requirements equal for every rank?").check(threshold)
    assert not F.universality_detail("Are promotion requirements the same for every rank?").check("At the same time, promotion requirements apply to every rank.")


def test_the_same_field_purpose_or_rank_never_settles_another_property():
    field = "Promotion requirements state that external members must have the same field of specialization as the applicant."
    assert not F.universality_detail("Are promotion requirements the same for every applicant?").check(field)
    rank = "Promotion requirements state that members must hold the same academic rank as the applicant."
    assert not F.universality_detail("Are promotion requirements the same for every rank?").check(rank)
    purpose = "In every department, promotion committee membership excludes members of higher-level committees for the same purpose."
    assert not F.universality_detail("Is the promotion committee membership the same for every department?").check(purpose)
    # the same noun that IS the asked subject settles it
    clinical = "Clinical faculty/academic non-faculty members' evaluation follows the same timeline and process as that of regular faculty."
    assert F.universality_detail("Is the evaluation process the same for clinical and regular faculty?").check(clinical)


def test_the_compared_subject_and_the_comparison_must_both_be_stated():
    assert not F.universality_detail("Are promotion requirements the same for every rank?").check("Requirements are the same for every rank.")
    assert not F.universality_detail("Are promotion requirements the same for every rank?").check("Promotion requirements are the same for every college.")
    assert F.universality_detail("Are promotion requirements the same for every rank?").check("Promotion requirements are the same for every rank.")
    stated = "This policy applies to all members of the University community, regardless of gender."
    assert F.universality_detail("Does the policy apply regardless of gender?").check(stated)
    assert not F.universality_detail("Does the policy apply regardless of nationality?").check(stated)


def test_a_narrow_same_procedure_never_broadens_to_the_asked_scope():
    from faculty_services_support import P150_D
    fallback = P150_D["lines"][1]
    for q in ("Is the promotion procedure the same at every college?", "Are the promotion procedures the same at department and college level?",
              "Do all colleges follow the same promotion procedure?"):
        assert not F.universality_detail(q).check(fallback), q
    ask = "Is the promotion procedure the same at every college?"
    assert not F.universality_detail(ask).check("At every college, the promotion procedure follows the same procedure mentioned above.")
    conditional = "If fewer than three members qualify, the promotion procedure is the same at every college."
    assert not F.universality_detail(ask).check(conditional)
    assert F.universality_detail("If fewer than three members qualify, is the promotion procedure the same at every college?").check(conditional)
    exception = "The promotion evaluation process is the same for clinical and regular faculty, with the exception that external evaluations differ."
    assert not F.universality_detail("Is the promotion evaluation process the same for clinical and regular faculty?").check(exception)
    assert F.universality_detail("Is the promotion evaluation process the same for clinical and regular faculty, except for external evaluations?").check(exception)


def test_semantic_direction_of_universality_claims():
    from faculty_services_support import P148_B, P149_D
    based = "The University of Sharjah offers a competitive basic salary based on faculty rank and specialization."
    assert not F.universality_detail("Is the basic salary the same regardless of specialization?").check(based)
    assert not F.universality_detail("Is the basic salary independent of rank?").check("The basic salary depends on faculty rank.")
    assert not F.universality_detail("Is the basic salary the same for every rank?").check("Professor 30,576 – 43,860")
    assert not F.universality_detail("Are promotion requirements the same for every rank?").check(P149_D["lines"][1])
    assert not F.universality_detail("Is the promotion committee membership the same for every department?").check(P148_B["lines"][1])


def test_r4_sameness_cases_fail_closed_on_the_handbook_lines(rm):
    from faculty_services_support import P69_CLINICAL, P148_B, P149_D, P150_D, RawOverlapReranker
    pages = [P69_CLINICAL, P148_B, P149_D, P150_D]
    for reranker in (RawOverlapReranker(), ScriptedReranker([], default=9.0)):
        s = make_specialist(corpus_from(rm, pages), reranker)
        for q in ("Are promotion requirements the same for every rank?", "Is the promotion procedure the same at every college?",
                  "Are the promotion procedures the same at department and college level?", "Is the promotion committee membership the same for every department?"):
            r = run(s, q)
            assert r.status is not SUPPORTED, q
            assert not r.findings or any("is not stated: the evidence states the topic" in m for m in r.missing), q
        assert run(s, "Is the evaluation process the same for clinical and regular faculty?").status is SUPPORTED


def test_sameness_negative_matrix_across_hr_domains(rm):
    from faculty_services_support import P69_CLINICAL, P148_B, P149_D, P150_D
    s = make_specialist(corpus_from(rm, PAGES + [P69_CLINICAL, P148_B, P149_D, P150_D]), ScriptedReranker([], default=9.0))
    for q in ("Are promotion requirements the same for every rank?", "Are the promotion procedures the same at department and college level?",
              "Is the promotion procedure the same at every college?", "Is the promotion committee membership the same for every department?",
              "Is the salary the same for every rank?", "Is the salary the same for all specializations?", "Is the probation period the same for every faculty category?",
              "Is the leave entitlement the same for all faculty?", "Is the contract duration the same for all appointment types?",
              "Is administrative compensation the same for all roles?", "Are evaluation weights the same for every category?",
              "Are benefits the same for every faculty member?"):
        assert run(s, q).status is not SUPPORTED, q
    assert run(s, "Is the probation period one year?").status is SUPPORTED


def test_a_passive_ask_binds_its_verb_to_the_asked_subject():
    probation = "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."
    visiting = "In exceptional cases, and with Chancellor approval, the visiting contract may be extended for an additional year."
    nearest = "During the probation period, the visiting contract may be extended."
    ask = F.passive_ask("Can the probation period be extended?")
    assert ask is not None and F._passive_bound(probation, ask)
    assert not F._passive_bound(visiting, ask)
    assert not F._passive_bound(nearest, ask)
    assert F._passive_bound(visiting, F.passive_ask("Can the visiting contract be extended?"))
    assert not F._passive_bound(probation, F.passive_ask("Can the probation period be renewed?"))
    assert F.passive_ask("Can additional maternity leave be approved by the Vice Chancellor for Academic Affairs?") is None
    assert F.passive_ask("Is the probation period one year?") is None


def test_passive_asks_never_settle_on_another_subject(high):
    for q in ("Can the housing allowance be extended?", "Can the probation period be renewed?", "Can the furniture allowance be extended?",
              "Can the salary be extended?", "Can the end of service compensation be extended?"):
        assert status(high, q) is not SUPPORTED, q
    r = run(high, "Can the probation period be extended?")
    assert r.status is not SUPPORTED or "extendable" in r.findings[0].evidence_quote
    assert all("visiting contract may be extended" not in f.evidence_quote or "probation period is one year" in f.evidence_quote
               for f in r.findings[:1])
    assert status(high, "Can the contract be renewed?") is SUPPORTED
    assert status(high, "Can the visiting contract be extended?") is SUPPORTED


# ---------------------------------------------------------------------------
# D-093 (Step 4.12-C6): condition polarity and the passive binding through noun relations
# ---------------------------------------------------------------------------
def test_negated_conditions_and_leave_states_are_read_from_the_ask():
    cases = {"Can the probation be extended without Chancellor approval?": [{"approval", "chancellor"}],
             "Can faculty resign during probation without notice?": [{"notice"}],
             "Can a faculty member resign during probation with no notice?": [{"notice"}],
             "Is there no need for Chancellor approval to extend probation?": [{"approval", "chancellor"}],
             "Does termination for cause not require the Chancellor's approval?": [{"approval", "chancellor"}],
             "Is the Chancellor's approval not required for termination for cause?": [{"approval", "chancellor"}],
             "Can the visiting contract be extended even if not approved by the Chancellor?": [{"approved", "chancellor"}]}
    for q, absent in cases.items():
        assert [set(w) for w in F.ask_condition(q).absent] == absent, q
    assert F.ask_condition("Can unpaid special leave be granted when the employee has unused vacation leave?").leave_states == (True,)
    assert F.ask_condition("Can unpaid special leave be granted if the employee has no unused vacation leave?").leave_states == (False,)
    assert F.ask_condition("Can unpaid special leave be granted even if the employee still has vacation leave?").leave_states == (True,)
    assert F.ask_condition("Is the probation period one year?") is None


def test_without_pay_stays_a_pay_status():
    assert F.ask_condition("Can additional sick leave be granted without pay?") is None
    assert F.ask_condition("Is additional maternity leave without full pay?") is None


def test_a_negated_condition_needs_a_stated_absence():
    probation = "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."
    cause = "Termination of faculty contracts for cause requires the approval of the Chancellor, and recommendations for termination must be made with justifiable reasons."
    notice = "Faculty members intending to resign during probation must provide a six-month notice."
    assert not F.condition_detail("Can the probation be extended without Chancellor approval?").check(probation)
    assert not F.condition_detail("Can the probation be extended with no Chancellor approval?").check(probation)
    assert not F.condition_detail("Can a faculty contract be terminated for cause without the Chancellor's approval?").check(cause)
    assert not F.condition_detail("Does termination for cause not require the Chancellor's approval?").check(cause)
    assert not F.condition_detail("Can faculty resign during probation without notice?").check(notice)
    hire = ("The Chancellor holds the jurisdiction to appoint national faculty who fulfil required criteria and professors with a high Scopus "
            "H-index or rare specializations without undergoing the regular review and appointment process.")
    assert F.condition_detail("Can the Chancellor appoint professors without undergoing the regular review?").check(hire)
    assert F.condition_detail("Is Chancellor approval not required to extend probation?").check("Chancellor approval is not required to extend probation.")


def test_the_leave_state_is_bound_with_its_polarity():
    special = ("Unpaid special leave, not exceeding thirty days, may be granted upon approval from appropriate University authorities if the "
               "employee has no unused vacation leave.")
    assert not F.condition_detail("Can unpaid special leave be granted when the employee has unused vacation leave?").check(special)
    assert not F.condition_detail("Can unpaid special leave be granted even if the employee still has vacation leave?").check(special)
    assert F.condition_detail("Can unpaid special leave be granted if the employee has no unused vacation leave?").check(special)


def test_contradicted_conditions_never_support(rm):
    for reranker in (None, ScriptedReranker([], default=9.0)):
        s = make_specialist(corpus_from(rm, PAGES), reranker)
        for q in ("Can the probation be extended without Chancellor approval?", "Can the visiting contract be extended without Chancellor approval?",
                  "Can faculty resign during probation without notice?", "Can a faculty contract be terminated for cause without the Chancellor's approval?",
                  "Is termination during probation possible without the Dean's approval?",
                  "Can unpaid special leave be granted when the employee has unused vacation leave?",
                  "Can a faculty member be promoted without the department committee's recommendation?",
                  "Can a faculty member be appointed without the required approval?", "Can the contract be renewed without successful annual evaluations?",
                  "Can unpaid special leave be granted without approval?"):
            r = run(s, q)
            assert r.status is not SUPPORTED, q
        r = run(s, "Can the probation be extended without Chancellor approval?")
        assert not r.findings or any("stated with the asked polarity is not stated" in m for m in r.missing)
        for q in ("Can the probation be extended with Chancellor approval?", "Can the visiting contract be extended with Chancellor approval?",
                  "Can unpaid special leave be granted if the employee has no unused vacation leave?", "Can additional sick leave be granted without pay?",
                  "Must I give a six-month notice when I resign during probation?"):
            assert status(s, q) is SUPPORTED, q


def test_passive_binding_runs_when_the_parser_reads_a_noun_relation(rm):
    from faculty_services_support import P63_CONFIRMATION
    assert F.assess_ask("Can the probation period be confirmed?").relation is not None
    confirmation = "Employment confirmation includes the probation period when calculating the employee's End of Service Benefit."
    detail = next(d for d in F.assess_ask("Can the probation period be confirmed?").details if d.family == "grounding")
    assert not detail.check(confirmation)
    for reranker in (None, ScriptedReranker([], default=9.0)):
        s = make_specialist(corpus_from(rm, [P63_CONFIRMATION]), reranker)
        for q in ("Can the probation period be confirmed?", "Can probation be confirmed?", "Can the probation period be included?"):
            assert status(s, q) is not SUPPORTED, q


def test_a_noun_form_binds_only_through_its_of_phrase():
    ask = F.passive_ask
    assert F._passive_bound("Termination of faculty contracts for cause requires the approval of the Chancellor.", ask("Can the contract be terminated?"))
    assert F._passive_bound("Renewal of the contract is subject to successful annual evaluations.", ask("Can the contract be renewed?"))
    assert F._passive_bound("UoS may terminate faculty contracts in the event of discontinued academic colleges.", ask("Can the contract be terminated?"))
    assert not F._passive_bound("Extension of the visiting contract requires Chancellor approval.", ask("Can the probation period be extended?"))
    assert not F._passive_bound("The Chancellor may extend the visiting contract.", ask("Can the probation period be extended?"))
    assert not F._passive_bound("New faculty members who are terminated during probation are entitled to return tickets.", ask("Can the probation be terminated?"))
    assert not F._passive_bound("The dean may terminate during probation any appointment.", ask("Can the probation be terminated?"))
    assert not F._passive_bound("Termination during the probation period requires a recommendation.", ask("Can the probation period be terminated?"))
    assert not F._passive_bound("Employment confirmation includes the probation period.", ask("Can the probation period be confirmed?"))
    assert not F._passive_bound("Employment confirmation includes the probation period.", ask("Can probation be included?"))


def test_noun_based_contract_facts_stay_answerable(high):
    for q in ("Can the contract be terminated?", "Can the contract be renewed?", "Can the visiting contract be extended?", "Can the probation period be extended?"):
        assert status(high, q) is SUPPORTED, q
    for q in ("Can probation be confirmed?", "Can probation be included?", "Can the allowance be terminated?", "Can leave be renewed?"):
        assert status(high, q) is not SUPPORTED, q


# ---------------------------------------------------------------------------
# D-094 (Step 4.12-C7): negation scope, absence frames, evidence polarity, leave-state phrasing, verb binding
# ---------------------------------------------------------------------------
SPECIAL_LEAVE = ("Unpaid special leave, not exceeding thirty days, may be granted upon approval from appropriate University authorities if the "
                 "employee has no unused vacation leave.")


def test_a_negator_governs_only_its_own_clause():
    assert not F._absence_stated(SPECIAL_LEAVE, frozenset({"approval"}))
    assert not F.condition_detail("Can unpaid special leave be granted with no approval from University authorities?").check(SPECIAL_LEAVE)
    assert F._absence_stated("professors with rare specializations without undergoing the regular review and appointment process.",
                             frozenset({"regular", "review", "process"}))
    assert F._absence_stated("Chancellor approval is not required for this step.", frozenset({"chancellor", "approval"}))
    assert F._absence_stated("Prior approval is unnecessary for this step.", frozenset({"approval"}))
    assert F._absence_stated("In the absence of approval, the request lapses.", frozenset({"approval"}))
    assert not F._absence_stated("The term may not exceed one year, with approval from the Dean.", frozenset({"approval"}))


def test_new_absence_frames_are_read():
    cases = {"Does termination for cause need no justifiable reasons?": {"justifiable", "reason"},
             "Does the dean's approval need no recommendation?": {"recommendation"},
             "Does termination require no Chancellor approval?": {"approval", "chancellor"},
             "Does resignation during probation require no notice?": {"notice"},
             "Is no Chancellor approval required to extend probation?": {"approval", "chancellor"},
             "Is Chancellor approval unnecessary for termination for cause?": {"approval", "chancellor"},
             "Can probation be extended in the absence of Chancellor approval?": {"approval", "chancellor"},
             "Can termination for cause happen absent the Chancellor's approval?": {"approval", "chancellor"},
             "Can the visiting contract be extended without needing Chancellor approval?": {"approval", "chancellor"}}
    for q, words in cases.items():
        assert set(F.ask_condition(q).absent[0]) == words, q
    assert F.ask_condition("Can faculty be absent from exams?") is None


def test_new_absence_frames_never_settle_a_stated_requirement():
    cause = "Termination of faculty contracts for cause requires the approval of the Chancellor, and recommendations for termination must be made with justifiable reasons."
    notice = "Faculty members intending to resign during probation must provide a six-month notice."
    recommend = "Termination during the probation period requires a recommendation by the relevant academic administrators."
    assert not F.condition_detail("Does termination for cause need no justifiable reasons?").check(cause)
    assert not F.condition_detail("Does termination require no Chancellor approval?").check(cause)
    assert not F.condition_detail("Does resignation during probation require no notice?").check(notice)
    assert not F.condition_detail("Does termination during probation need no recommendation?").check(recommend)
    assert not F.condition_detail("Can probation be extended in the absence of Chancellor approval?").check(
        "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval.")


def test_negated_evidence_never_supports_an_affirmative_ask():
    resign = ("New faculty members who are terminated during probation are entitled to return tickets, while new faculty members who resign "
              "during probation are not entitled to benefits such as annual leave and ticket allowances.")
    assert not F.predicate_detail("Are faculty who resign during probation entitled to annual leave?").check(resign)
    assert not F.predicate_detail("Is a resigning faculty member entitled to return tickets during probation?").check(resign)
    assert not F.predicate_detail("Does the bonus award impact the monthly salary?").check(
        "The Annual Bonus Award is a prestigious recognition. It is important to note that the award shall not impact the monthly salary of faculty members.")
    assert not F.predicate_detail("Is contract renewal based on a single year's performance?").check(
        "Contract renewal is not based on a single year's performance but rather on the faculty member's cumulative contributions.")
    assert not F.predicate_detail("Are committee members part of higher-level committees?").check(
        "Members of this committee should not be part of higher-level committees for the same purpose.")
    # a negating clause of the asked subject rejects the unit even beside an affirmative clause of the same subject
    both = ("Faculty who resign during probation are entitled to return tickets; faculty who resign during probation are not entitled to "
            "annual leave.")
    assert not F.predicate_detail("Are faculty who resign during probation entitled to annual leave?").check(both)
    # an affirmative clause of another subject never stands for the asked one
    assert not F.predicate_detail("Are probationary faculty who resign entitled to benefits?").check(resign)
    # the affirmative clause of the asked subject still settles it; a unit that never states the predicate is not bound by it
    assert F.predicate_detail("Is a terminated faculty member entitled to return tickets?").check(resign)
    assert F.predicate_detail("Is the basic salary based on rank and specialization?").check(
        "The University of Sharjah offers a competitive basic salary based on faculty rank and specialization.")
    assert F.predicate_detail("Is an employee eligible for unpaid special leave when no vacation leave is unused?").check(SPECIAL_LEAVE)


def test_leave_state_phrasings_keep_their_polarity():
    for text in ("when vacation leave is unused", "if vacation leave remains", "when vacation leave is left", "who has remaining vacation leave",
                 "who still has vacation leave", "who has leave left"):
        assert F._leave_states(text) == [True], text
    for text in ("if the employee has no unused vacation leave", "when no vacation leave remains", "once vacation leave is exhausted",
                 "after the employee has used all vacation leave", "when no vacation leave is unused"):
        assert F._leave_states(text) == [False], text
    assert not F.condition_detail("Is an employee eligible for unpaid special leave when vacation leave is unused?").check(SPECIAL_LEAVE)
    assert not F.condition_detail("Can unpaid special leave be granted when vacation leave remains?").check(SPECIAL_LEAVE)
    assert F.condition_detail("Can unpaid special leave be granted when vacation leave is exhausted?").check(SPECIAL_LEAVE)


def test_an_active_non_role_ask_keeps_its_verb():
    ask = F.active_ask("Does employment confirmation extend the probation period?")
    assert ask is not None and set(ask.obj_anchors) == {"probation"}
    assert not F._active_bound("Employment confirmation includes the probation period when calculating the End of Service Benefit.", ask)
    assert F._active_bound("Employment confirmation extends the probation period for new faculty.", ask)
    assert not F._active_bound("The faculty probation period is one year, extendable to a maximum of two years.", F.active_ask("Does the probation period extend the contract?"))
    assert F.active_ask("Does the Chancellor approve termination for cause?") is None


def test_an_is_x_verbed_ask_keeps_its_verb_and_agent():
    ask = F.passive_ask("Is the probation period terminated by the Dean?")
    assert ask is not None and set(ask.agent) == {"dean"}
    assert not F._passive_bound("Termination during the probation period requires approval from the Dean.", ask)
    assert F._passive_bound("The probation period may be terminated by the Dean.", ask)
    assert not F._passive_bound("The probation period may be terminated by the Chancellor.", ask)
    for q in ("Is a terminated faculty member entitled to return tickets?", "Are new faculty terminated during probation entitled to return tickets?",
              "Are new faculty members who are terminated during probation entitled to return tickets?"):
        assert F.passive_ask(q) is None, q
    # an HR-concept subject (bindable) with a later predicate or a relative clause is no passive ask either
    for q in ("Is the contract terminated during probation subject to approval?", "Is the special leave that was approved by the dean?"):
        assert F.passive_ask(q) is None, q
    assert F.passive_ask("Is the contract terminated during probation?") is not None


def test_a_relational_verb_never_stands_for_an_action():
    confirmation = "Employment confirmation includes the probation period when calculating the employee's End of Service Benefit."
    assert not F._active_bound(confirmation, F.active_ask("Does employment confirmation extend the probation period?"))
    assert not F._passive_bound(confirmation, F.passive_ask("Is the employment confirmation extended with the probation?"))


def test_r6_wrong_cases_fail_closed_and_true_facts_stay_answerable(rm):
    from faculty_services_support import P63_CONFIRMATION
    for reranker in (None, ScriptedReranker([], default=9.0)):
        s = make_specialist(corpus_from(rm, PAGES + [P63_CONFIRMATION]), reranker)
        for q in ("Can unpaid special leave be granted with no approval from University authorities?", "Does termination for cause need no justifiable reasons?",
                  "Does resignation during probation require no notice?", "Does termination during probation need no recommendation?",
                  "Does extending probation need no Chancellor approval?", "Is a resigning faculty member entitled to return tickets during probation?",
                  "Are faculty who resign during probation entitled to annual leave?", "Does the bonus award impact the monthly salary?",
                  "Is contract renewal based on a single year's performance?", "Is an employee eligible for unpaid special leave when vacation leave is unused?",
                  "Does employment confirmation extend the probation period?", "Is the employment confirmation extended with the probation?",
                  "Does the probation period extend the contract?", "Is the probation period terminated by the Dean?",
                  # D-102 (Step 4.14-W1): supported until D-102; page 63 states the inclusion only "when calculating the employee's End
                  # of Service Benefit", a "when" condition the question omits
                  "Does confirmation of employment include the probation period?"):
            assert status(s, q) is not SUPPORTED, q
        for q in ("Is a terminated faculty member entitled to return tickets?", "Is the basic salary based on rank and specialization?",
                  "Can unpaid special leave be granted if the employee has no unused vacation leave?",
                  "Is an employee eligible for unpaid special leave when no vacation leave is unused?",
                  "Does confirmation of employment include the probation period when calculating the End of Service Benefit?",     # D-102: the condition asked
                  "Can the contract be renewed?", "Can the contract be terminated?", "Can the probation be extended with Chancellor approval?"):
            assert status(s, q) is SUPPORTED, q


def test_campaign_found_phrasings_are_bound():
    associate = ("If there are no full professors in a particular department, an associate professor may be a committee member. However, "
                 "associate professors are not involved in reviewing applications for full professor rank.")
    assert not F.predicate_detail("Are associate professors involved in reviewing applications for full professor rank?").check(associate)
    assert F._leave_states("if the employee still has some vacation leave") == [True]
    assert F._leave_states("when vacation leave has not been used") == [True]
    assert F._leave_states("when vacation leave has not yet been fully used") == [True]
    hr = "UoS may terminate faculty contracts in the event of discontinued academic colleges, according to UoS HR policies."
    assert not F._passive_bound(hr, F.passive_ask("Is the contract terminated by HR?"))
    assert F._passive_bound("The Chancellor may terminate faculty contracts for cause.", F.passive_ask("Is the contract terminated by the Chancellor?"))


def test_entitlement_words_form_one_family_and_structural_predicates_must_be_stated():
    resign = ("New faculty members who are terminated during probation are entitled to return tickets, while new faculty members who resign "
              "during probation are not entitled to benefits such as annual leave and ticket allowances.")
    assert not F.predicate_detail("Are resigning faculty eligible for ticket allowances during probation?").check(resign)
    college = ("In September of each year, the College Council shall form the College Academic Promotions Committee. The Committee shall "
               "include a full professor from each department in the college.")
    assert not F.predicate_detail("Are department committee members part of the college committee for promotion?").check(college)
    assert F.predicate_detail("Is an employee eligible for unpaid special leave when no vacation leave is unused?").check(SPECIAL_LEAVE)


def test_negation_of_another_object_and_role_subjects_do_not_overblock():
    renewal = ("Contract renewal is not based on a single year's performance but rather on the faculty member's cumulative contributions over "
               "the duration of the current contract.")
    assert F.predicate_detail("Is contract renewal based on cumulative contributions over the contract rather than a single year?").check(renewal)
    assert not F.predicate_detail("Is contract renewal based on a single year's performance?").check(renewal)
    assert F.passive_ask("Is a dean appointed for a period of three years, renewable only once?") is None
    assert F.active_ask("Does the Chancellor extend the visiting contract?") is None


# ---------------------------------------------------------------------------
# D-095 (Step 4.12-C8): negative-ask polarity, entitlement synonyms, independent absence frames, role-subject verbs, labels
# ---------------------------------------------------------------------------
RESIGN = ("New faculty members who are terminated during probation are entitled to return tickets, while new faculty members who resign "
          "during probation are not entitled to benefits such as annual leave and ticket allowances.")
PROBATION = ("The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor "
             "approval. Faculty members intending to resign during probation must provide a six-month notice.")
SALARY = "The University of Sharjah offers a competitive basic salary based on faculty rank and specialization, following the salary scale below:"
COLLEGE = "A. In September of each year, the College Council shall form the College Academic Promotions Committee, chaired by the college dean."
RENEWAL = ("Contract renewal is not based on a single year's performance but rather on the faculty member's cumulative contributions over "
           "the duration of the current contract.")
AWARD = "It is important to note that the award shall not impact the monthly salary of faculty members."
SICK = "In special circumstances, up to an additional 30 days of sick leave without pay may be granted."


def test_a_negative_ask_reads_its_kind_subject_and_negated_words():
    cases = {"Is the probation period not one year?": ("value", {"probation", "period"}, {"one", "year"}),
             "Is the monthly compensation of a dean not AED 10,000?": ("value", {"monthly", "compensation", "dean"}, {"aed", "10", "000"}),
             "Is the College Academic Promotions Committee not formed by the College Council?":
                 ("relation", {"college", "academic", "promotion", "committee"}, {"form", "college", "council"}),
             "Is unpaid leave not deducted from salary?": ("relation", {"unpaid", "leave"}, {"deduct", "salary"}),
             "Is the basic salary not based on rank?": ("predicate", {"basic", "salary"}, {"based", "rank"}),
             "Are terminated faculty not entitled to return tickets?": ("predicate", {"terminat", "faculty"}, {"entitl", "return", "ticket"}),
             "Do resigning faculty not get annual leave?": ("predicate", {"resign", "faculty"}, {"entitl", "annual", "leave"}),
             "Does the award not impact the monthly salary?": ("predicate", {"award"}, {"impact", "monthly", "salary"}),
             "Is there no furniture allowance?": ("existence", set(), {"furniture", "allowance"}),
             "Is there no probation period for new faculty?": ("existence", set(), {"probation", "period", "new", "faculty"}),
             "Is no housing provided to faculty?": ("existence", set(), {"hous", "entitl", "faculty"}),
             "Are there not any sick leave days with full pay?": ("predicate", set(), {"sick", "leave", "day", "full", "pay"}),
             "Do faculty get no annual return air ticket?": ("predicate", {"faculty"}, {"entitl", "annual", "return", "air", "ticket"}),
             "Does the award have no impact on the monthly salary?": ("predicate", {"award"}, {"impact", "monthly", "salary"})}
    for q, (kind, subject, negated) in cases.items():
        ask = F.negative_ask(q)
        assert ask is not None and ask.kind == kind and set(ask.subject) == subject and set(ask.negated) == negated, (q, ask)
    # a rhetorical opener, "whether / or not", a quantity bound, a relative or conditional negation and an absent condition
    for q in ("Isn't the probation period one year?", "Is it unclear whether or not the probation is extended?",
              "Is unpaid special leave granted for not more than thirty days?", "Can faculty who have not completed probation resign?",
              "Is unpaid special leave granted for no more than thirty days?", "Can a faculty member who has no unused vacation leave take unpaid special leave?",
              "Can unpaid special leave be granted if the employee has no unused vacation leave?", "Can the contract be terminated with no notice?",
              "Is my employment confirmed if I receive no notification six months before the contract ends?",
              "Can unpaid special leave be granted when vacation leave has not been used?", "Does termination for cause not require the Chancellor's approval?",
              "Is a security clearance not necessary for an employment offer?"):
        assert F.negative_ask(q) is None, q


def test_a_negative_ask_is_settled_only_by_the_same_negation():
    # affirmative evidence contradicts the negative claim: salary, probation, notice, committee formation, entitlement
    assert not F.negative_detail("Is the basic salary not based on rank?").check(SALARY)
    assert not F.negative_detail("Is the probation period not one year?").check(PROBATION)
    assert not F.negative_detail("Is the notice period for resigning during probation not six months?").check(PROBATION)
    assert not F.negative_detail("Is the College Academic Promotions Committee not formed by the College Council?").check(COLLEGE)
    assert not F.negative_detail("Are terminated faculty not entitled to return tickets?").check(RESIGN)
    # the negation of another subject never stands for the asked one
    assert not F.negative_detail("Are terminated faculty not entitled to ticket allowances?").check(RESIGN)
    # the same negation of the same subject settles it
    assert F.negative_detail("Are faculty who resign during probation not entitled to annual leave?").check(RESIGN)
    assert F.negative_detail("Are faculty who resign during probation not entitled to ticket allowances?").check(RESIGN)
    assert F.negative_detail("Is contract renewal not based on a single year's performance?").check(RENEWAL)
    assert F.negative_detail("Does the award not impact the monthly salary?").check(AWARD)
    assert F.negative_detail("Are resigning faculty not entitled to tickets?").check("Resigning faculty are ineligible for tickets.")
    # "no" as a determiner: an existence or a light verb is bound the same way
    furniture = "Furniture Allowance: A one-time furniture allowance of 30,000 Dirhams is provided, which can be utilized over four years."
    assert not F.negative_detail("Is there no furniture allowance?").check(furniture)
    assert F.negative_detail("Is there no furniture allowance?").check("There is no furniture allowance.")    # D-096: narrower evidence is in the scope tests
    assert not F.negative_detail("Is there no Hajj leave for Muslim faculty?").check(
        "Muslim faculty attending their first Hajj may receive up to twenty calendar days of paid leave.")
    assert not F.negative_detail("Do terminated faculty get no return tickets?").check(RESIGN)
    assert F.negative_detail("Do faculty who resign during probation get no ticket allowances?").check(RESIGN)
    assert F.negative_detail("Does the award have no impact on the monthly salary?").check(AWARD)
    # a negator governs its own scope only, and a negated value stands right after it
    assert not F.negative_detail("Is the basic salary not based on rank?").check(
        "The basic salary is not paid in advance; the basic salary is based on rank.")
    assert not F.negative_detail("Is the probation period not one year?").check(
        "The probation period, if not completed within one year, is reviewed by the Dean.")
    assert F.negative_detail("Is the probation period not one year?").check("The probation period shall not be one year.")
    assert not F.negative_detail("Is the probation period not one year?").check(
        "The probation period does not apply to faculty members for one year.")         # the negator governs "apply", not the value
    assert F.negative_detail("Is the College Academic Promotions Committee not formed by the College Council?").check(
        "The College Academic Promotions Committee shall not be formed by the College Council.")


def test_entitlement_synonyms_keep_their_polarity():
    for q in ("Are faculty who resign during probation qualified for ticket allowances?", "Do faculty who resign during probation get annual leave?",
              "Do faculty who resign during probation receive ticket allowances?", "Are faculty who resign during probation given annual leave?"):
        ask = F.predicate_ask(q)
        assert ask is not None and ask.name == "entitled", q
        assert not F.predicate_detail(q).check(RESIGN), q
    # a negative synonym ask reads the same family: "not receive" / "not eligible" is settled by "not entitled"
    assert F.negative_detail("Do faculty who resign during probation not receive ticket allowances?").check(RESIGN)
    assert F.negative_detail("Are faculty who resign during probation not eligible for annual leave?").check(RESIGN)
    # an affirmative synonym of the stated subject stays answerable; a bare "provided" of another subject settles nothing
    assert F.predicate_detail("Do new faculty terminated during probation receive return tickets?").check(RESIGN)
    assert F.predicate_detail("Is health insurance provided for dependent children under 18?").check(
        "Furniture Allowance: A one-time furniture allowance of 30,000 Dirhams is provided. Health Insurance: Comprehensive health insurance is "
        "offered for faculty members, their spouse, and up to three dependent children under 18 years old.")
    assert F.negative_ask("Are terminated faculty no longer entitled to return tickets?") is not None
    # a synonym followed by "no", inside the ask's own condition or in a relative clause of its subject is no affirmative claim
    assert F.predicate_ask("Do faculty who resign during probation get no ticket allowances?") is None
    assert F.predicate_ask("Is my employment confirmed if I receive no notification six months before the contract ends?") is None
    ask = F.predicate_ask("Are faculty who receive a housing allowance entitled to free housing?")
    assert ask is not None and {"allowance", "hous"} <= set(ask.subject) and set(ask.obj) == {"free", "hous"}
    ask = F.predicate_ask("Are faculty who qualify for housing entitled to a housing allowance?")
    assert ask is not None and set(ask.subject) == {"faculty", "qualify", "hous"} and set(ask.obj) == {"allowance", "hous"}
    ask = F.predicate_ask("Do faculty who get a housing allowance receive free housing?")
    assert ask is not None and ask.mode == "negation" and set(ask.subject) == {"faculty", "allowance", "hous"} and set(ask.obj) == {"free", "hous"}


def test_each_absence_frame_is_read_on_its_own():
    assert [set(w) for w in F.ask_condition("Can additional sick leave without pay be granted without special circumstances?").absent] == [{"special", "circumstance"}]
    assert {frozenset(w) for w in F.ask_condition("Can faculty resign during probation without notice and without approval?").absent} == {
        frozenset({"notice"}), frozenset({"approval"})}
    assert {frozenset(w) for w in F.ask_condition("Can the probation be extended with no notice and without Chancellor approval?").absent} == {
        frozenset({"notice"}), frozenset({"chancellor", "approval"})}
    assert not F.condition_detail("Can additional sick leave without pay be granted without special circumstances?").check(SICK)
    assert F.ask_condition("Can additional sick leave be granted without pay?") is None


def test_a_role_subject_keeps_its_explicit_verb():
    for q, actor, action, obj in (("Does the Chancellor extend probation?", "chancellor", "extend", ("probation",)),
                                  ("Does the Dean renew the contract?", "dean", "renew", ("contract",)),
                                  ("Does HR waive the notice?", "hr", "waive", ("notice",)),
                                  ("Does the Chancellor approve termination for cause?", "chancellor", "approve", ("termination",))):
        rel = F._hr_relation(q)
        assert rel is not None and (rel.actor, rel.action, rel.obj[:1]) == (actor, action, obj), (q, rel)
    assert not F._hr_relation_stated(F._hr_relation("Does the Chancellor extend probation?"), PROBATION)


def test_a_policy_label_is_no_actor():
    for q in ("Does the Chancellor policy extend probation?", "Does the HR policy approve unpaid special leave?", "Do committee guidelines approve promotions?",
              "Does the Dean section approve leave?", "Does the Chancellor policy on leave extend probation?"):
        assert F._hr_relation(q) is None, q
        assert F.label_ask(q) is not None, q
    assert F.label_ask("Does the Chancellor extend probation?") is None
    # a reporting frame leaves the role as the subject of its own verb
    rel = F._hr_relation("Does the Faculty Handbook say HR provides an Employee Handbook?")
    assert rel is not None and rel.actor == "hr"
    ask = F.label_ask("Does the HR policy approve unpaid special leave?")
    assert F._label_bound("The HR policy approves unpaid special leave for staff.", ask)
    assert not F._label_bound("According to UoS HR policies, the Dean approves unpaid leave.", ask)
    assert not F._label_bound("The HR policy approval process is described below.", ask)


def test_r7_wrong_cases_fail_closed_and_true_negatives_stay_answerable(rm):
    from faculty_services_support import P63_CONFIRMATION
    for reranker in (None, ScriptedReranker([], default=9.0)):
        s = make_specialist(corpus_from(rm, PAGES + [P63_CONFIRMATION]), reranker)
        for q in ("Is the basic salary not based on rank?", "Is the probation period not one year?", "Is the furniture allowance not 30,000 Dirhams?",
                  "Is the notice period for resigning during probation not six months?", "Is the College Academic Promotions Committee not formed by the College Council?",
                  "Is a professor's basic salary not 30,576 to 43,860 AED?", "Is the monthly compensation of a dean not AED 10,000?",
                  "Is unpaid leave not deducted from salary?", "Are terminated faculty not entitled to return tickets?", "Are terminated faculty not entitled to ticket allowances?",
                  "Are female faculty not entitled to maternity leave with full pay?", "Are faculty not entitled to sick leave with full pay?",
                  "Are faculty who resign during probation qualified for ticket allowances?", "Do faculty who resign during probation get annual leave?",
                  "Do faculty who resign during probation receive ticket allowances?", "Are resigning probationary faculty given annual leave?",
                  "Can additional sick leave without pay be granted without special circumstances?", "Can faculty resign during probation without notice and without approval?",
                  "Does the Chancellor policy extend probation?", "Does the Chancellor extend probation?", "Does the Dean renew the contract?",
                  "Does the HR policy assist new faculty members in obtaining entry visas?", "Does the HR policy on visas assist new faculty members in obtaining entry visas?",
                  "Is there no probation period for new faculty?", "Is there no furniture allowance?", "Are there no sick leave days with full pay?",
                  "Is there no Hajj leave for Muslim faculty?", "Do terminated faculty get no return tickets?"):
            assert status(s, q) is not SUPPORTED, q
        for q in ("Are faculty who resign during probation not entitled to annual leave?", "Are faculty who resign during probation not entitled to ticket allowances?",
                  "Are faculty who resign during probation not eligible for annual leave?", "Is contract renewal not based on a single year's performance?",
                  "Does the award not impact the monthly salary?", "Can additional sick leave be granted without pay?",
                  "Does the Chancellor approve termination for cause?", "Does the College Council form the College Academic Promotions Committee?",
                  "Does HR assist new faculty members in obtaining entry visas?", "Is the probation period one year?", "Is the basic salary based on rank and specialization?",
                  "Is a terminated faculty member entitled to return tickets?", "Are full-time female faculty eligible for sixty days of maternity leave with full pay?",
                  "Is my employment confirmed if I receive no notification six months before the contract ends?",
                  "Does the award have no impact on the monthly salary?", "Do faculty who resign during probation get no ticket allowances?"):
            assert status(s, q) is SUPPORTED, q
        # D-097 strict generic scope: the evidence grants it to female faculty employed on a full-time basis, narrower than the asked
        # generic "female faculty"; the quote is returned with the scope reason (it was answerable here before Step 4.12-C10).
        r = run(s, "Are female faculty eligible for sixty days of maternity leave with full pay?")
        assert r.status is not SUPPORTED and any("generic population (female faculty)" in m for m in r.missing), r.missing


# ---------------------------------------------------------------------------
# D-096 (Step 4.12-C9): claim compatibility - the semantic invariants of the consolidated contracts
# ---------------------------------------------------------------------------
BONUS = "Faculty members may apply for the bonus award in consecutive years. Parttime and visiting faculty members are not eligible for the bonus award."
RELATIVES = "Employment of a relative in a position where they directly or indirectly manage another relative is not allowed."
ASSOCIATE = "However, associate professors are not involved in reviewing applications for full professor rank."
CONFIRMATION = ("New faculty members must receive written notification of employment confirmation or termination six months before the end of the "
                "contract. If neither occurs, the new faculty member's employment is considered confirmed.")
SUSPENSION = ("If the case is not referred to the Disciplinary Council within one month after the suspension decision, the suspension shall be "
              "revoked, and the Disciplinary Council's decision shall be enforced.")
DEAN_CHAIR = ("In September of each year, the College Council shall form the College Academic Promotions Committee, chaired by the college dean "
              "if a full professor or by another professor within the college if the dean is not a full professor.")
OFFER = "Employment offers are drafted by HR upon receiving the selected candidate's Security Clearance."
AUTHORITY = "The Vice Chancellor for Academic Affairs, in consultation with the Chancellor, exercises authority over the Faculty Employment Policy."


def test_scope_a_narrower_negation_never_settles_a_broader_negative_claim():
    # the negation of the asked population settles the ask; the same ask about all faculty must not stay settled
    for narrow, broad, evidence in (
            ("Are faculty who resign during probation not entitled to ticket allowances?", "Are faculty not entitled to ticket allowances?", RESIGN),
            ("Are faculty who resign during probation not entitled to benefits?", "Are faculty members not entitled to benefits?", RESIGN),
            ("Are visiting faculty members not eligible for the bonus award?", "Are faculty not eligible for the bonus award?", BONUS),
            ("Is employment of a relative who manages another relative not allowed?", "Is the employment of a relative not allowed?", RELATIVES)):
        assert F.negative_detail(narrow).check(evidence), narrow
        assert not F.negative_detail(broad).check(evidence), broad
    # a narrower negated object: a modifier before it or a restriction after it that the ask does not name
    assert F.negative_detail("Is contract renewal not based on a single year's performance?").check(RENEWAL)
    assert not F.negative_detail("Is contract renewal not based on performance?").check(RENEWAL)
    assert F.negative_detail("Are associate professors not involved in reviewing applications for full professor rank?").check(ASSOCIATE)
    assert not F.negative_detail("Are associate professors not involved in reviewing applications?").check(ASSOCIATE)
    assert not F.negative_detail("Is there no furniture allowance?").check("There is no furniture allowance for visiting faculty.")
    assert not F.negative_detail("Is the probation period not one year?").check("The probation period shall not be one year for visiting faculty.")
    # a category of the population (appointment basis, track, rank) narrows it as a relative clause does
    part_time = "Part-time faculty members are not eligible for the bonus award."
    assert F.negative_detail("Are part-time faculty not eligible for the bonus award?").check(part_time)
    assert not F.negative_detail("Are faculty not eligible for the bonus award?").check(part_time)
    assert not F.negative_detail("Are full-time faculty not eligible for the bonus award?").check(part_time)


def test_conditional_what_an_antecedent_says_is_no_stated_fact():
    assert not F.negative_detail("Does the employee have no unused vacation leave?").check(SPECIAL_LEAVE)
    assert not F.negative_detail("Is the case not referred to the Disciplinary Council?").check(SUSPENSION)
    assert F.negative_detail("Is the case not referred to the Disciplinary Council?").check("The case is not referred to the Disciplinary Council.")
    assert not F.assertion_detail("Does the employee have unused vacation leave?").check(SPECIAL_LEAVE)
    # metamorphic: the same negation, unrestricted, settles the ask stated as a fact and never under "if"
    for q, conditional, stated in (
            ("Does the employee have no unused vacation leave?", "If the employee has no unused vacation leave, unpaid special leave may be granted.",
             "The employee has no unused vacation leave."),
            ("Is the case not referred to the Disciplinary Council?", "If the case is not referred to the Disciplinary Council, the suspension shall be revoked.",
             "The case is not referred to the Disciplinary Council."),
            ("Is the faculty member not eligible for the bonus award?",
             "If the faculty member is not eligible for the bonus award, the dean shall inform the faculty member.",
             "The faculty member is not eligible for the bonus award.")):
        assert not F.negative_detail(q).check(conditional), q
        assert F.negative_detail(q).check(stated), q
    # an affirmative claim, whatever its predicate, stated only inside an antecedent is no fact; its consequent is one
    deans = ("If the Deans Council approves a promotion, the Chancellor shall forward the decision to the Board of Trustees for approval. "
             "The Chancellor shall issue the Board of Trustee decisions that approve promotion cases.")
    utilized = "If all sick leave days have been utilized, annual vacation leave may be used."
    for q, conditional, stated in (
            ("Does the Deans Council approve the promotion?", deans, "The Deans Council approves the promotion."),
            ("Have all sick leave days been utilized?", utilized, "All sick leave days have been utilized."),
            ("Is the faculty member eligible for the bonus award?", "If the faculty member is eligible for the bonus award, the dean shall inform the faculty member.",
             "The faculty member is eligible for the bonus award.")):
        assert not F.assertion_detail(q).check(conditional), q
        assert F.assertion_detail(q).check(stated), q
    assert F.assertion_detail("Does the Chancellor forward the decision to the Board of Trustees?").check(deans)
    assert F.assertion_detail("May annual vacation leave be used?").check(utilized)
    assert F.affirmative_claim("Have all sick leave days been utilized?").predicate == frozenset({"utiliz"})
    # the same content asked under the same condition stays answerable
    assert F.condition_detail("Can unpaid special leave be granted if the employee has no unused vacation leave?").check(SPECIAL_LEAVE)
    assert F.condition_detail("Is the suspension revoked if the case is not referred to the Disciplinary Council within one month?").check(SUSPENSION)
    # antecedents: span (to the comma before the main clause), polarity, anaphora
    ants = F._antecedents(SUSPENSION)
    assert len(ants) == 1 and not ants[0].positive and SUSPENSION[ants[0].start:ants[0].end].endswith("suspension decision")
    first, second = CONFIRMATION.split(". ", 1)
    ants = F._antecedents(second, first)
    assert len(ants) == 1 and not ants[0].positive and {"notify", "terminat", "employ"} <= set(ants[0].roots)


def test_negative_existence_forms_are_negative_claims():
    furniture = "Furniture Allowance: A one-time furniture allowance of 30,000 Dirhams is provided, which can be utilized over four years."
    for q in ("Is not any furniture allowance provided?", "Isn't any furniture allowance provided?", "Is no furniture allowance provided?",
              "Is there no furniture allowance?", "Are there no furniture allowances?", "Is there not any furniture allowance?"):
        ask = F.negative_ask(q)
        assert ask is not None and {"furniture", "allowance"} <= set(ask.negated), q
        assert F.predicate_ask(q) is None, q                     # an ask opened by a negation claims no affirmative predicate
        assert not F.negative_detail(q).check(furniture), q
        assert F.negative_detail(q).check("There is no furniture allowance provided."), q


def test_a_compound_or_coordinated_label_is_never_an_actor():
    for q in ("Do the HR rules and regulations arrange travel for new faculty?", "Do the HR rules and procedures assist new faculty with entry visas?",
              "Do HR policy and procedures arrange travel for new faculty?", "Do HR guidelines draft employment offers?",
              "Does the HR policy manual assist new faculty in obtaining entry visas?", "Do the Chancellor rules and regulations extend probation?",
              "Do Dean rules and regulations approve termination during probation?", "Do committee guidelines review promotion applications?"):
        assert F.label_ask(q) is not None, q
        assert F._hr_relation(q) is None, q
    # removing the label restores the role as the actor; a content verb makes a document the natural subject, no label ask
    assert F._hr_relation("Does HR draft employment offers?").actor == "hr"
    assert F.label_ask("Does the Employee Handbook cover HR policies?") is None
    ask = F.label_ask("Do the HR rules and regulations arrange travel for new faculty?")
    assert not F._label_bound("HR arranges travel and hotel accommodations for new faculty members.", ask)
    assert F._label_bound("The HR rules and regulations arrange travel for new faculty members.", ask)


def test_an_explicit_verb_outside_the_inventory_binds_actor_action_and_object():
    rel = F._hr_relation("Does HR draft the Security Clearance?")
    assert (rel.actor, rel.action, rel.obj) == ("hr", "draft", ("security", "clearance"))
    assert not F._hr_relation_stated(rel, OFFER)                 # HR drafts the offer, not the clearance
    assert F._hr_relation_stated(F._hr_relation("Does HR draft employment offers?"), OFFER)
    rel = F._hr_relation("Does the Chancellor exercise authority over the Faculty Employment Policy?")
    assert (rel.actor, rel.action) == ("chancellor", "exercise")
    assert not F._hr_relation_stated(rel, AUTHORITY)             # consulted, not exercising
    assert F._hr_relation_stated(F._hr_relation("Does the Vice Chancellor for Academic Affairs exercise authority over the Faculty Employment Policy?"),
                                 AUTHORITY)
    # an unrelated verb in place of the stated one never stays stated
    assert not F._hr_relation_stated(F._hr_relation("Does HR translate employment offers?"), OFFER)
    # a generic verb's "for" phrase naming no role is what is acted on, not a direction
    rel = F._hr_relation("Does HR compile the required documents for the Employee File?")
    assert (rel.actor, rel.action, rel.target) == ("hr", "compile", ())
    assert F._hr_relation_stated(rel, "HR compiles all required documents and information to maintain in the Employee File.")
    assert not F._hr_relation_stated(F._hr_relation("Does HR verify the required documents for the Employee File?"),
                                     "HR compiles all required documents and information to maintain in the Employee File.")
    # after "do" or a modal the explicit verb is bare and no auxiliary follows it; a "be" makes the ask passive (no explicit verb)
    for q in ("May grievances against Disciplinary Council decisions be directed to the President within two weeks?", "Do Chancellor decisions bind the colleges?",
              "May the Dean be consulted?"):
        st = F.ask_structure(q)
        assert st is None or st.verb not in ("decisions", "within", "consulted"), q
        assert F._hr_relation(q) is None, q
    assert F.ask_structure("Does HR process applications?").verb == "process"
    assert (F.ask_structure("Do the HR forms require approval?").label, F.ask_structure("Do the HR forms require approval?").verb) == ("forms", "require")


def test_a_negated_rule_contradicts_the_affirmative_ask_for_any_predicate():
    for ask, negated, affirmed in (
            ("Is the investigation case rendered void if the faculty member under investigation resigns?",
             "The investigation case is not rendered void if the faculty member under investigation submits their resignation.",
             "The investigation case is rendered void if the faculty member under investigation submits their resignation."),
            ("Does UoS have a statute of limitations for professional misconduct?",
             "UoS has no statute of limitations for professional misconduct and may take action for violations during academic employment.",
             "UoS has a statute of limitations for professional misconduct."),
            ("Is transfer between branches applicable for the Assistant Professor-Emirati Track?",
             "Transfer between branches/campuses is not applicable for Assistant Professor-Emirati Track.",
             "Transfer between branches/campuses is applicable for Assistant Professor-Emirati Track."),
            ("Should termination be the first resort?", "Termination should not be the first resort.", "Termination should be the first resort."),
            ("Will disciplinary action impede criminal prosecution pursued by the accuser?",
             "Additionally, this disciplinary action will not impede or hinder any criminal prosecution pursued by the accuser.",
             "Additionally, this disciplinary action will impede any criminal prosecution pursued by the accuser."),
            ("Can a clinical faculty member transfer to a regular faculty member while employed at UoS?",
             "A clinical faculty member shall not be permitted to transfer to a regular faculty member (teaching and research) while employed at UoS.",
             "A clinical faculty member may transfer to a regular faculty member while employed at UoS.")):
        detail = F.polarity_detail(ask)
        assert detail is not None, ask
        assert not detail.check(negated), ask
        assert detail.check(affirmed), ask
    # the D-094 predicates keep their own detail; a quantity bound negates nothing
    assert F.polarity_detail("Are faculty who resign during probation entitled to annual leave?") is None
    assert F.polarity_detail("Can unpaid special leave be granted?").check(SPECIAL_LEAVE)


def test_relative_and_participle_modifiers_keep_their_polarity():
    cond = F.ask_condition("Are faculty who are not terminated during probation entitled to return tickets?")
    assert [set(w) for w in cond.absent_modifiers] == [{"terminated", "probation"}]
    assert not F.condition_detail("Are faculty who are not terminated during probation entitled to return tickets?").check(RESIGN)
    assert F.condition_detail("Are faculty who are terminated during probation entitled to return tickets?").check(RESIGN)
    notify = "Newly appointed faculty members not meeting UoS standards should be notified as early as possible."
    assert not F.condition_detail("Should newly appointed faculty members meeting UoS standards be notified as early as possible?").check(notify)
    assert F.condition_detail("Should newly appointed faculty members not meeting UoS standards be notified as early as possible?").check(notify)
    # contracted negation, either apostrophe, is a negator in the ask and in the evidence
    improve = "Faculty who doesn\u2019t meet expectations during a contract term will be provided with an opportunity to improve."
    for q in ("Will faculty who do not meet expectations during a contract term be provided with an opportunity to improve?",
              "Will faculty who don't meet expectations during a contract term be provided with an opportunity to improve?",
              "Will faculty who don\u2019t meet expectations during a contract term be provided with an opportunity to improve?"):
        assert [set(w) for w in F.ask_condition(q).absent_modifiers] == [{"meet", "expectation", "contract", "term"}], q
        assert F.condition_detail(q).check(improve) and F.condition_detail(q).check(improve.replace("\u2019", "'")), q
    assert not F.condition_detail("Will faculty who meet expectations during a contract term be provided with an opportunity to improve?").check(improve)
    rank = "Clinical faculty members with no prior appointment in an academic setting shall be appointed at the rank of an assistant clinical professor."
    assert not F.condition_detail("Are clinical faculty members with a prior appointment in an academic setting appointed at the rank of assistant "
                                  "clinical professor?").check(rank)
    # a negated population modifier is no negative claim of the ask; a negated predicate still is
    assert F.negative_ask("Should newly appointed faculty members not meeting UoS standards be notified as early as possible?") is None
    assert F.negative_ask("Are terminated faculty not entitled to return tickets?") is not None


def test_a_clausal_absent_condition_needs_its_absence():
    cases = {"Can the probation period be extended if the Chancellor does not approve?": ({"approve", "chancellor"}, PROBATION),
             "Can probation be extended when the Chancellor has not approved it?": ({"approved", "chancellor"}, PROBATION),
             "Can additional sick leave without pay be granted if there are no special circumstances?": ({"special", "circumstance"}, SICK),
             "Is an employment offer final if the candidate does not submit the signed copy?": (
                 {"candidate", "submit", "signed", "copy"}, "Employment Offers are considered final upon submission of the signed copy by the candidate to HR."),
             "Can the contract be renewed if the annual evaluations are not successful?": (
                 {"annual", "evaluation", "successful"}, "The contract may be renewed for three years upon successful annual evaluations as per UoS policies.")}
    for q, (words, evidence) in cases.items():
        assert [set(w) for w in F.ask_condition(q).absent_clauses] == [words], q
        assert not F.condition_detail(q).check(evidence), q
    # the same condition stated as given is answered by the rule; "unless" states an absence; "when ...?" asks a time
    assert F.condition_detail("Can the probation period be extended if the Chancellor approves?").check(PROBATION)
    assert [set(w) for w in F.ask_condition("Can probation be extended unless the Chancellor approves?").absent_clauses] == [{"approve", "chancellor"}]
    assert F.ask_condition("When is the next payroll date?") is None


def test_the_asked_condition_is_the_evidence_condition():
    assert not F.condition_detail("Is employment confirmed if the faculty member receives a termination notification?").check(CONFIRMATION)
    assert F.condition_detail("Is my employment confirmed if I receive no notification six months before the contract ends?").check(CONFIRMATION)
    assert not F.condition_detail("Is employment confirmed if the probation is extended?").check(CONFIRMATION)
    assert F.condition_detail("Is the suspension revoked if the case is not referred to the Disciplinary Council within one month?").check(SUSPENSION)
    assert not F.condition_detail("Is the suspension revoked if the case is referred to the Disciplinary Council within one month?").check(SUSPENSION)
    # a disjunctive rule: each condition governs its own consequent
    assert F.condition_detail("Is the College Academic Promotions Committee chaired by the college dean if a full professor?").check(DEAN_CHAIR)
    assert not F.condition_detail("Is the College Academic Promotions Committee chaired by the college dean if the dean is not a full professor?").check(
        DEAN_CHAIR)
    # a claim stated without conditions holds under any asked condition; a condition stated as "upon approval" is the asked one
    assert F.condition_detail("Is the probation period one year if the faculty member is visiting?").check(PROBATION)
    assert F.condition_detail("Can unpaid special leave be granted if approved?").check(SPECIAL_LEAVE)


def test_a_qualified_value_binds_its_qualifier():
    d = F.hr_fact_detail
    assert d("Is the probation period one year?").check(PROBATION)
    assert not d("Is the maximum probation period one year?").check(PROBATION)    # the stated maximum is two years
    assert not d("Is the probation period not more than one year?").check(PROBATION)
    notice = "Terminated faculty members receive a minimum six-month notice."
    assert d("Is the minimum notice for terminated faculty six months?").check(notice)
    assert not d("Is the maximum notice for terminated faculty six months?").check(notice)
    assert d("Is unpaid special leave not more than thirty days?").check(SPECIAL_LEAVE)
    assert not d("Is unpaid special leave not less than thirty days?").check(SPECIAL_LEAVE)
    maternity = "Additional maternity leave without pay, up to a maximum of thirty days, may be approved by the Vice Chancellor for Academic Affairs."
    assert d("Is the maximum additional maternity leave without pay thirty days?").check(maternity)
    assert not d("Is the maximum additional maternity leave without pay sixty days?").check(maternity)
    # a limit of another quantity, or of another pay status, in the same item binds nothing
    insurance = "Comprehensive health insurance is offered for faculty members, their spouse, and up to three dependent children under 18 years old."
    assert d("Is health insurance offered for up to three dependent children under 18 years old?").check(insurance)
    sick = "Faculty are entitled to sick leave with full pay for up to 15 working days and half pay for an additional 15 working days per contract year."
    assert d("Is sick leave with half pay up to 15 working days?").check(sick)


def test_r8_semantic_cases_fail_closed_and_true_facts_stay_answerable(rm):
    from faculty_services_support import C9_PAGES, P63_CONFIRMATION
    wrong = ("Are faculty members not entitled to benefits?", "Are faculty not entitled to ticket allowances?", "Is contract renewal not based on performance?",
             "Are faculty not eligible for the bonus award?", "Is the employment of a relative not allowed?", "Is not any furniture allowance provided?",
             "Does the employee have no unused vacation leave?", "Is the case not referred to the Disciplinary Council?",
             "Do the HR rules and regulations arrange travel for new faculty?", "Do the HR rules and procedures assist new faculty with entry visas?",
             "Do HR policy and procedures arrange travel for new faculty?", "Do the Chancellor rules and regulations extend probation?",
             "Do Dean rules and regulations approve termination during probation?", "Does the HR guideline draft employment offers?",
             "Do HR guidelines draft employment offers?", "Does HR draft the Security Clearance?",
             "Does the Chancellor exercise authority over the Faculty Employment Policy?",
             "Is the investigation case rendered void if the faculty member under investigation resigns?",
             "Does UoS have a statute of limitations for professional misconduct?", "Is transfer between branches applicable for the Assistant Professor-Emirati Track?",
             "Should termination be the first resort?", "Will disciplinary action impede criminal prosecution pursued by the accuser?",
             "Can a clinical faculty member transfer to a regular faculty member while employed at UoS?",
             "Are faculty who are not terminated during probation entitled to return tickets?",
             "Are female faculty who are not employed full-time entitled to sixty days of maternity leave with full pay?",
             "Are faculty who are not on the Emirati track entitled to a teaching load reduction of 6 credit hours?",
             "Are faculty who are not engaged in clinical activities eligible for a clinical allowance?",
             "Are Muslim faculty who are not attending their first Hajj eligible for twenty days of paid leave?",
             "Should newly appointed faculty members meeting UoS standards be notified as early as possible?",
             "Are clinical faculty members with a prior appointment in an academic setting appointed at the rank of assistant clinical professor?",
             "Can the probation period be extended if the Chancellor does not approve?", "Can unpaid special leave be granted if University authorities do not approve it?",
             "Can additional sick leave without pay be granted if there are no special circumstances?",
             "Can a faculty member resign during probation if they do not provide notice?",
             "Is an employment offer final if the candidate does not submit the signed copy?", "Can the contract be renewed if the annual evaluations are not successful?",
             "Can probation be extended if there are no exceptional circumstances?",
             "Is employment confirmed if the faculty member receives a termination notification?", "Is the maximum probation period one year?",
             "Does the Deans Council approve the promotion?", "Have all sick leave days been utilized?")
    right = ("Are faculty who resign during probation not entitled to ticket allowances?", "Are visiting faculty members not eligible for the bonus award?",
             "Is contract renewal not based on a single year's performance?", "Does misconduct not require intent?", "Is termination not the first resort?",
             "Is the suspension revoked if the case is not referred to the Disciplinary Council within one month?",
             "Is my employment confirmed if I receive no notification six months before the contract ends?", "Does HR draft employment offers?",
             "Does the Vice Chancellor for Academic Affairs exercise authority over the Faculty Employment Policy?",
             "Does HR arrange travel and hotel accommodations for new faculty members?", "Is the probation period one year?",
             "Can the probation be extended with Chancellor approval?", "Can faculty members apply for the bonus award in consecutive years?")
    for reranker in (None, ScriptedReranker([], default=9.0)):
        s = make_specialist(corpus_from(rm, PAGES + C9_PAGES + [P63_CONFIRMATION]), reranker)
        for q in wrong:
            assert status(s, q) is not SUPPORTED, q
        for q in right:
            assert status(s, q) is SUPPORTED, q
        r = run(s, "Is transfer between branches applicable for the Assistant Professor-Emirati Track?")
        assert any("is not stated: the evidence negates it for the asked subject" in m for m in r.missing)
