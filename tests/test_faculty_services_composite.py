"""
Faculty Services & HR statement-level composite claim (Milestone 4, Step 4.12-C11, D-098): the claim_model force, restriction,
frequency, slot-universal, enumeration and ledger tables; the joint comparison of a claim with ONE predicate statement and
the window vetoes; the first-failure message; the switches; and the R10 / DR2 cases, the conservative losses and the
avoidable-loss guards on the fixture pages under both rankings, with the synthetic classes on crafted pages.
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import claim_model as C
from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus
from handbook_bot.agents.specialists.details import Detail

from faculty_services_support import C11_CORPUS_PAGES, ScriptedReranker, corpus_from, make_specialist, page, registry_and_map, run

SUPPORTED = FindingStatus.SUPPORTED
FO, RK, FC = C.Force, C.RestrictionKind, C.FrequencyClass
COMPATIBLE, CONTRADICTED, UNKNOWN, NA = C.Compatibility.COMPATIBLE, C.Compatibility.CONTRADICTED, C.Compatibility.UNKNOWN, C.Compatibility.NOT_APPLICABLE


def frame(question):
    claim = F.read_claim(question)
    assert claim is not None, question
    return F.read_claim_frame(question, claim)


def composite(question):
    detail = F.claim_detail(question, F.read_claim(question))
    assert detail is not None, question
    return detail


def local(question, unit):
    return composite(question).check(unit)


def window(question, quote):
    return composite(question).check.window(quote)


# ---------------------------------------------------------------------------
# Force
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,force", [
    ("Must faculty members submit the request?", FO.DEONTIC), ("Should faculty members submit the request?", FO.DEONTIC),
    ("May faculty members submit the request?", FO.EXISTENTIAL), ("Are faculty members eligible for the allowance?", FO.GENERIC),
    ("Are faculty members entitled to the allowance?", FO.GENERIC), ("Are all faculty members eligible for the allowance?", FO.UNIVERSAL_QUANTIFIED),
    ("Does each college form a hiring committee?", FO.UNIVERSAL_QUANTIFIED), ("Is the allowance always paid in January?", FO.UNIVERSAL_FREQUENCY),
    ("Is the allowance never paid in January?", FO.UNIVERSAL_FREQUENCY), ("Can all faculty members apply for the award?", FO.UNIVERSAL_QUANTIFIED),
    ("Can the committee always contact witnesses?", FO.UNIVERSAL_FREQUENCY), ("Is the visiting contract extendable?", FO.EXISTENTIAL),
    ("Is the allowance sometimes paid in January?", FO.EXISTENTIAL), ("Is the allowance paid each year?", FO.GENERIC),
])
def test_force_classes_in_their_priority_order(question, force):
    assert frame(question).force is force


# ---------------------------------------------------------------------------
# Force x restriction (the D-098 table, written out)
# ---------------------------------------------------------------------------
_U = {FO.EXISTENTIAL: set(), FO.GENERIC: {RK.NECESSARY, RK.SPECIAL}, FO.DEONTIC: {RK.NECESSARY, RK.SPECIAL},
      FO.UNIVERSAL_QUANTIFIED: {RK.NECESSARY, RK.EXCEPTION, RK.SPECIAL, RK.HEDGE},
      FO.UNIVERSAL_FREQUENCY: {RK.NECESSARY, RK.EXCEPTION, RK.SPECIAL, RK.HEDGE, RK.GATE, RK.SUFFICIENT, RK.TRIGGER, RK.POSSIBILITY}}


@pytest.mark.parametrize("force", list(FO))
@pytest.mark.parametrize("kind", list(RK))
def test_the_force_restriction_table(force, kind):
    assert C.restriction_compatibility(force, [kind]) is (UNKNOWN if kind in _U[force] else COMPATIBLE)


def test_a_concessive_condition_never_restricts():
    assert all(C.restriction_compatibility(f, [RK.CONCESSIVE]) is COMPATIBLE for f in FO)


# ---------------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("asked,predicate,kinds,negative,expected", [
    (FC.UNIVERSAL, "suspension from duty always results in the suspension of salaries", [], False, COMPATIBLE),
    (FC.UNIVERSAL, "suspension from duty results in the suspension of salaries", [], False, UNKNOWN),
    (FC.UNIVERSAL, "suspension from duty always results in the suspension of salaries", [RK.EXCEPTION], False, UNKNOWN),
    (FC.UNIVERSAL, "the committee may always contact witnesses", [RK.POSSIBILITY], False, UNKNOWN),
    (FC.UNIVERSAL, "the allowance is paid in all cases", [RK.CONCESSIVE], False, COMPATIBLE),
    (FC.NEGATIVE_UNIVERSAL, "they will not face disciplinary action", [], True, COMPATIBLE),
    (FC.NEGATIVE_UNIVERSAL, "they will not face disciplinary action", [RK.EXCEPTION], True, UNKNOWN),
    (FC.NEGATIVE_UNIVERSAL, "they face disciplinary action", [], False, UNKNOWN),
    (FC.EXISTENTIAL, "leave may be granted in special circumstances", [RK.SPECIAL, RK.POSSIBILITY], False, COMPATIBLE),
    (FC.EXISTENTIAL, "leave is not granted", [], True, UNKNOWN),
    (FC.DEFAULT, "legal representation is generally not permitted", [], True, COMPATIBLE),
    (FC.DEFAULT, "legal representation is not permitted", [], True, UNKNOWN),
    (FC.PROPORTIONAL, "the committee often meets in May", [], False, COMPATIBLE),
    (FC.PROPORTIONAL, "the committee usually meets in May", [], False, UNKNOWN),
])
def test_frequency_compatibility(asked, predicate, kinds, negative, expected):
    assert C.frequency_compatibility(asked, predicate, kinds, negative) is expected


@pytest.mark.parametrize("text,cls", [
    ("Is it granted in all circumstances?", FC.UNIVERSAL), ("Is it granted without exception?", FC.UNIVERSAL),
    ("Is it automatically renewed?", FC.UNIVERSAL), ("Is it granted under no circumstances?", FC.NEGATIVE_UNIVERSAL),
    ("Is it ever granted?", FC.EXISTENTIAL), ("Is it as a rule granted?", FC.DEFAULT), ("Is it hardly ever granted?", FC.PROPORTIONAL),
    ("Is it granted?", None),
])
def test_the_closed_frequency_lexicon(text, cls):
    assert C.frequency_of(text)[0] is cls


# ---------------------------------------------------------------------------
# Slot universals, enumerations, residue, ledger
# ---------------------------------------------------------------------------
def _np(text):
    return F._c11_noun_phrases(text, lambda w: bool(F._C11_VERB_GROUP.fullmatch(w)))


@pytest.mark.parametrize("asked,stated,expected", [
    ("all dependent children", "all dependent children", COMPATIBLE), ("all dependent children", "up to three dependent children", UNKNOWN),
    ("all children", "all qualifying children", UNKNOWN), ("all qualifying children", "all qualifying children from KG1", COMPATIBLE),
    ("every faculty appointment", "all faculty appointments", COMPATIBLE), ("all appointments", "all three appointments", UNKNOWN),
    ("all appointments", "some appointments", UNKNOWN), ("all committees", "all colleges", UNKNOWN),
])
def test_slot_universal_compatibility(asked, stated, expected):
    a = _np(asked)
    assert a, asked
    assert C.slot_universal_compatibility(a[0], _np(stated), F._same_root) is expected


def test_a_number_or_cap_inside_a_universal_noun_phrase_marks_it_capped():
    # The reader marks the cap; the table never takes a capped universal for an unlimited one, even with the same number asked.
    assert _np("all three dependent children")[0].capped and _np("all 3 appointments")[0].capped
    assert not _np("all dependent children")[0].capped
    assert C.slot_universal_compatibility(_np("all three appointments")[0], _np("all three appointments"), F._same_root) is UNKNOWN


def test_the_shared_noun_phrase_reader_stops_at_the_asked_verb_and_at_determiners():
    words = ["each", "college", "forms", "a", "department"]
    assert C.noun_phrase_after(words, 0, lambda w: w == "forms") == ("college", ())
    assert C.noun_phrase_after(["all", "qualifying", "children", "from", "kg"], 0, lambda w: False) == ("children", ("qualifying",))


def _items(text):
    return C.enumeration_items(text, 0, F._root)


@pytest.mark.parametrize("asked,stated,expected", [
    ("Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law", "Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law",
     COMPATIBLE),
    ("Medicine and Law", "Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law", UNKNOWN),
    ("Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, Law and Engineering",
     "Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law", UNKNOWN),
    ("Law, Medicine", "Medicine and Law", COMPATIBLE), ("Medicine and Engineering", "Medicine and Law", UNKNOWN),
    ("all faculty members", "new faculty members", UNKNOWN), ("faculty members with dependent children and spouses", "faculty members with dependent children",
                                                               UNKNOWN),
])
def test_enumeration_equality_is_one_to_one_and_order_free(asked, stated, expected):
    assert C.enumeration_compatibility(_items(asked), [_items(stated)]) is expected


def test_enumeration_items_keep_modifiers_and_drop_procedural_wrappers():
    assert [i.text for i in _items("Medicine, Dental Medicine, and Law.")] == ["Medicine", "Dental Medicine", "Law"]
    assert [sorted(i.roots) for i in _items("a decision from the Disciplinary Council.")] == [sorted({F._root("disciplinary"), F._root("council")})]
    assert [i.text for i in _items("new faculty members who joined this year")] == ["new faculty members"]


def test_residue_words_and_their_restatement():
    assert C.residue_words("does the university strictly prohibit any form of harassment") == ("strictly",)
    assert C.residue_words("is the leave fully paid every month") == ("fully",)
    assert C.residue_words("is the report submitted annually and always on time") == ()
    assert C.restated("equal", "a rank equivalent to or higher than the rank applied for")
    assert not C.restated("fully", "the leave is paid")


def test_the_ledger_requires_every_operator_to_be_consumed_on_the_statement():
    e = C.LedgerEntry
    assert C.ledger_compatibility((e("frequency", "always", (0, 6), "occasion", "frequency"),), frozenset({"frequency"}), "").compatibility is COMPATIBLE
    r = C.ledger_compatibility((e("frequency", "always", (0, 6), "occasion", "frequency"),), frozenset(), "")
    assert r.compatibility is UNKNOWN and "always" in r.reason
    assert C.ledger_compatibility((e("residue", "strictly", (0, 0), "predicate", "residue"),), frozenset(), "it strictly prohibits").compatibility is COMPATIBLE
    assert C.ledger_compatibility((e("residue", "fully", (0, 0), "predicate", "residue"),), frozenset(), "it is paid").compatibility is UNKNOWN
    assert C.ledger_compatibility((e("comparison", "more than", (0, 0), "value", "static"),), frozenset(), "").compatibility is COMPATIBLE
    assert C.ledger_compatibility((), frozenset(), "").compatibility is NA


@pytest.mark.parametrize("question,ledger", [
    ("Is the housing allowance paid only once?", [("exclusivity", "only", "exclusivity")]),
    ("Is the end-of-service gratuity paid once?", []),
])
def test_a_frequency_once_is_no_condition_in_the_ledger(question, ledger):
    assert [(e.cls, e.text, e.handler) for e in frame(question).ledger] == ledger


@pytest.mark.parametrize("unit", ["The library opens at eight.", "Probation is explained in the handbook."])
def test_a_unit_with_no_statement_never_supports_the_composite(unit):
    assert F.read_clause_statements(unit, frame("Is the probation period one year?")) == ()
    assert local("Is the probation period one year?", unit) is False


def test_the_comparison_runs_every_dimension_on_one_statement_in_the_fixed_order():
    seen = []

    def make(name):
        def check(f, s):
            seen.append(name)
            return C.DimensionResult(name, COMPATIBLE)
        return check

    checks = {name: make(name) for name in C.DIMENSIONS[:-1]}
    f = frame("Must faculty members submit the request?")
    s = F.read_clause_statements("Faculty members must submit the request.", f)[0]
    results = C.compare_claim_to_statement(f, s, checks)
    assert seen == list(C.DIMENSIONS[:-1]) and [r.dimension for r in results] == list(C.DIMENSIONS)
    assert C.first_failure(results) is None
    assert C.first_failure((C.DimensionResult("scope", UNKNOWN, "x"), C.DimensionResult("modality", UNKNOWN, "y"))).dimension == "scope"


# ---------------------------------------------------------------------------
# The composite on statements
# ---------------------------------------------------------------------------
B1 = "Full-time faculty must take their annual leave in the summer, while part-time faculty may take their annual leave in the summer."


@pytest.mark.parametrize("question,ok", [
    ("Must part-time faculty take their annual leave in the summer?", False), ("May part-time faculty take their annual leave in the summer?", True),
    ("Must full-time faculty take their annual leave in the summer?", True),
])
def test_split_binding_never_supports(question, ok):
    assert local(question, B1) is ok


SUSPENSION = "Unless otherwise specified by the Disciplinary Council, suspension from duty also results in the suspension of salaries."


@pytest.mark.parametrize("question,unit,ok", [
    ("Does suspension from duty always result in the suspension of salary?", SUSPENSION, False),
    ("Does suspension from duty always result in the suspension of salary?", "Suspension from duty always results in the suspension of salaries.", True),
    ("Does suspension from duty result in the suspension of salary?", SUSPENSION, False),            # D-099: the "unless" governs it
    ("Unless the Disciplinary Council specifies otherwise, does suspension from duty result in the suspension of salary?", SUSPENSION, True),
    ("Is sick leave without pay granted in all circumstances?", "In special circumstances, additional sick leave without pay may be granted.", False),
    ("Can additional sick leave be granted without pay?", "In special circumstances, additional sick leave without pay may be granted.", True),
])
def test_frequency_is_read_on_the_statement(question, unit, ok):
    assert local(question, unit) is ok


@pytest.mark.parametrize("question,unit,ok", [
    ("Does health insurance cover all dependent children of faculty members?",
     "Comprehensive health insurance is offered for faculty members, their spouse, and up to three dependent children under 18 years old.", False),
    ("Do all senior administrative appointments require Chancellor approval?",
     "Senior administrative appointments require Chancellor approval, except for Deans, Vice Chancellors, and the Chancellor.", False),
    ("Must all faculty appointments be assigned a title and an employee class?",
     "All faculty appointments must be assigned a title and an employee class.", True),
    ("Does each college form a hiring committee at the beginning of each year?",
     "Each college forms a hiring committee at the beginning of each year.", True),
    ("Is the maximum allowance 75,000 Dirhams for all children?", "A maximum of 75,000 Dirhams is provided for all qualifying children.", False),
    ("Is the maximum allowance 75,000 Dirhams for all qualifying children?", "A maximum of 75,000 Dirhams is provided for all qualifying children.", True),
])
def test_universal_noun_phrases_need_an_explicit_unlimited_universal(question, unit, ok):
    assert local(question, unit) is ok


CLINICAL = "The appointment of clinical faculty/academic non-faculty is limited to Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law."


@pytest.mark.parametrize("question,ok", [
    ("Is the appointment of clinical faculty limited to Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law?", True),
    ("Is the appointment of clinical faculty limited to Medicine and Law?", False),
    ("Is the appointment of clinical faculty limited to Medicine, Dental Medicine and Pharmacy?", False),
])
def test_an_exclusive_enumeration_needs_the_same_item_set(question, ok):
    assert local(question, CLINICAL) is ok


@pytest.mark.parametrize("question,quote,veto", [
    ("Is the housing allowance paid only once?", "The housing allowance is paid once, and it may be paid again with the Dean's approval.", True),
    ("Is the housing advance paid only once?", "The housing advance is paid once, and it may be paid a second time with the Dean's approval.", True),
    ("Is the housing allowance paid only once?", "The housing allowance is paid once, and a second payment may be approved by the Dean.", True),
    ("Is the housing allowance paid only once?", "The housing allowance is paid once and cannot be paid again.", False),
    ("Must faculty members submit the annual leave plan only in September?",
     "Faculty members must submit the annual leave plan in September of each year. New faculty members must also submit the annual leave plan within "
     "one month of joining.", True),
    ("Can faculty resign only at the end of an academic semester?",
     "Faculty members must submit their resignation at least six months before the effective date and must coincide with the end of an academic "
     "semester.", False),
    ("Is the allowance paid only with Dean approval?", "The allowance is paid only with Dean approval. The allowance is paid without Dean approval "
                                                       "in emergencies.", True),
])
def test_exclusivity_owns_the_alternative_fillers_of_the_window(question, quote, veto):
    assert (window(question, quote) is CONTRADICTED) is veto


@pytest.mark.parametrize("question,unit,ok", [
    ("Must all new faculty members complete the annual leave training?",
     "All new faculty members must complete the annual leave training only if they work in laboratories.", False),
    ("May new faculty members complete the annual leave training?", "New faculty members may complete the annual leave training only if they work in "
                                                                     "laboratories.", True),
    ("Are department chairs appointed by the Chancellor?",
     "The Chancellor appoints department chairs upon the recommendation of the Dean.", True),
    ("Is the visiting contract extended with Chancellor approval?",
     "In exceptional cases, and with Chancellor approval, the visiting contract may be extended for an additional year.", False),
    ("Can the visiting contract be extended?",
     "In exceptional cases, and with Chancellor approval, the visiting contract may be extended for an additional year.", True),
    ("Is the probation period one year?", "The faculty probation period is one year, extendable under exceptional circumstances.", True),
    ("Is renewal subject to successful evaluations?", "Renewal for the second year is subject to department needs and successful annual evaluations.", True),
    ("Is the allowance always paid in January?", "The allowance is paid in January if the Dean approves.", False),
])
def test_an_unconditional_claim_against_the_restrictions_of_its_statement(question, unit, ok):
    assert local(question, unit) is ok


def test_a_universal_claim_is_vetoed_by_an_exception_in_the_window():
    quote = ("All senior administrative appointments require Chancellor approval. Senior administrative appointments of Deans do not require "
             "Chancellor approval.")
    assert window("Do all senior administrative appointments require Chancellor approval?", quote) is CONTRADICTED
    quote2 = "Annual leave is approved in January. In exceptional cases, the Dean approves annual leave in March."
    assert window("Is annual leave always approved in January?", quote2) is CONTRADICTED


def test_an_exclusive_claim_is_not_stated_with_a_hedge():
    chair = ("The Chancellor appoints a faculty member as department chair upon the recommendation of the Dean, for a period of three years, renewable "
             "normally only once.")
    assert local("Is the department chair's appointment renewable only once?", chair) is False
    assert local("Is the department chair's appointment renewable only once?", chair.replace("normally ", "")) is True


@pytest.mark.parametrize("question,unit,ok", [
    ("Does the University strictly prohibit any form of harassment?", "The University strictly prohibits any form of harassment.", True),
    ("Does the University fully prohibit any form of harassment?", "The University prohibits any form of harassment.", False),
    ("Will individuals who report harassment never face disciplinary action?", "Individuals who report harassment will not face disciplinary action.", True),
])
def test_every_operator_is_consumed_or_the_claim_fails_closed(question, unit, ok):
    assert local(question, unit) is ok


def test_an_explicit_universal_with_a_stated_exception_is_not_stated():
    q = "Do all senior administrative appointments require Chancellor approval?"
    assert local(q, "All senior administrative appointments require Chancellor approval.") is True
    assert local(q, "All senior administrative appointments require Chancellor approval, except for Deans.") is False


def test_an_existential_frequency_tolerates_restrictions():
    assert local("Is additional sick leave without pay sometimes granted?",
                 "In special circumstances, additional sick leave without pay may be granted.") is True


def test_a_copular_attribute_stated_only_in_special_circumstances_is_not_proved():
    # D-099 (class C): a copular claim meets the same force x restriction table as any other claim - no copular bypass. The
    # claim with its special circumstances asked, and the possibility claim, are proved.
    unit = "In special circumstances, up to an additional 30 days of sick leave without pay may be granted."
    assert local("Is additional sick leave without pay up to 30 days?", unit) is False
    assert local("Is additional sick leave without pay granted for up to 30 days?", unit) is False
    assert local("Is additional sick leave without pay up to 30 days in special circumstances?", unit) is True
    assert local("Can up to 30 days of additional sick leave without pay be granted?", unit) is True


def test_a_statement_whose_subject_names_another_role_cannot_support():
    assert local("Does HR assist new faculty with entry visas?", "The Human Resources Department assists new faculty with entry visas.") is True
    assert local("Does HR assist new faculty with entry visas?", "The Dean assists new faculty with entry visas.") is False
    unit = "The Dean must approve the request, and the Chancellor may approve it in exceptional cases."
    assert local("Must the Dean approve the request?", unit) is True
    assert local("Must the Chancellor approve the request?", unit) is False      # split binding across two roles
    assert local("Does the University Medical Clinic offer diagnostic care?",
                 "At UMC, offer a comprehensive range of medical services including diagnostic care.") is True     # no role subject: tolerated


def test_only_the_head_of_a_subject_names_its_role():
    # A role in a prepositional modifier is no subject ("Termination of faculty contracts"); a role name with a preposition
    # stays one span. An actor the ask parse found outside the claim subject (the Chancellor of "Chancellor approval") binds nothing.
    assert F._head_roles("Termination of faculty contracts for cause") == frozenset()
    assert F._head_roles("The Dean of the Faculty of Medicine") == {"dean"}
    assert F._head_roles("The Vice Chancellor for Academic Affairs") == {"vcaa"}
    assert F._head_roles("The Head of the Department") == {"chair"}
    unit = "Termination of faculty contracts for cause requires Chancellor approval."
    assert local("Does termination of contracts for cause require Chancellor approval?", unit) is True
    assert local("Does termination require Chancellor approval?", unit) is True
    assert local("Must the Dean approve the request?", "The Chancellor of the University must approve the request.") is False


@pytest.mark.parametrize("question,unit,ok", [
    # the subject text runs from the clause start: an asked actor named after a leading antecedent or an unsplit clause is tolerated
    ("Does the Chancellor notify the applicant in writing when a promotion is not approved?",
     "If the Central Committee, Deans Council, or Board of Trustees does not approve a promotion, the Chancellor shall notify the applicant in writing.", True),
    ("Must faculty members attest the documents required for the residence visa?",
     "The HR checklist lists the documents required for the residence visa, and faculty members must attest the documents.", True),
    # another role and the asked one nowhere: still not stated of the asked actor
    ("Does the Chancellor notify the applicant in writing?",
     "If the Council does not approve the promotion, the Dean shall notify the applicant in writing.", False),
])
def test_an_asked_actor_named_in_the_subject_text_is_tolerated(question, unit, ok):
    assert local(question, unit) is ok
    if not ok:
        assert "subject" in composite(question).check.explain(unit)


@pytest.mark.parametrize("question,unit,applies", [
    ("Does termination require Chancellor approval?", "The Dean's decision on termination requires Chancellor approval.", False),  # actor in the object
    ("Must the Dean approve the request?", "The Dean must approve the request.", True),
])
def test_the_subject_dimension_binds_only_an_actor_that_is_the_claim_subject(question, unit, applies):
    f = frame(question)
    statements = F.read_clause_statements(unit, f)
    assert statements, unit
    assert (F._composite_checks(f, question)["subject"](f, statements[0]) is not None) is applies


def test_polarity_reuses_the_c9_contracts():
    assert local("Does misconduct not require intent?", "Misconduct does not require intent and shall be investigated.") is True
    assert local("Is contract renewal based on cumulative contributions over the contract rather than a single year?",
                 "Contract renewal is not based on a single year’s performance but rather on the faculty member’s cumulative contributions over "
                 "the duration of the current contract.") is True


def test_the_value_is_read_by_the_c9_fact_reader_on_the_statement():
    assert local("Are female faculty employed full time entitled to sixty days of maternity leave with full pay?",
                 "Female faculty employed on a full-time basis are entitled to sixty calendar days of maternity leave with full pay.") is True


# ---------------------------------------------------------------------------
# Missing message: first failed dimension of the best statement
# ---------------------------------------------------------------------------
def test_the_message_names_the_first_failed_dimension_of_the_best_statement():
    msg = composite("Must part-time faculty take their annual leave in the summer?").check.explain(B1)
    assert "modality" in msg and "part-time faculty may take" in msg
    msg = composite("Is the appointment of clinical faculty limited to Medicine and Law?").check.explain(CLINICAL)
    assert "exclusivity" in msg and "medicine; law" in msg.lower()
    msg = composite("Do all senior administrative appointments require Chancellor approval?").check.explain(
        "Senior administrative appointments require Chancellor approval, except for Deans.")
    assert "scope" in msg
    assert "no statement" in composite("Is the probation period one year?").check.explain("The library opens at eight.")


def test_the_first_failure_follows_the_fixed_dimension_order():
    msg = composite("Must all part-time faculty take their annual leave in the summer?").check.explain(B1)
    assert msg.startswith("the asked claim is not stated in one statement: modality"), msg       # modality before scope


# ---------------------------------------------------------------------------
# Switches: each rule has its own seam
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("switch,question,unit,on", [
    ("_C11_JOINT", "Must part-time faculty take their annual leave in the summer?", B1, False),
    ("_C11_ENUMERATION", "Is the appointment of clinical faculty limited to Medicine and Law?", CLINICAL, False),
    ("_C11_UNCONDITIONAL", "Must all new faculty members complete the annual leave training?",
     "All new faculty members must complete the annual leave training only if they work in laboratories.", False),
    ("_C11_LEDGER", "Does the University fully prohibit any form of harassment?", "The University prohibits any form of harassment.", False),
])
def test_each_switch_turns_its_own_rule_off(monkeypatch, switch, question, unit, on):
    assert local(question, unit) is on
    monkeypatch.setattr(F, switch, False)
    detail = F.claim_detail(question, F.read_claim(question))
    assert detail is None or detail.check(unit) is True


def test_the_slot_universal_switch_leaves_its_operator_to_the_hardened_ledger(monkeypatch):
    # D-099: with the slot reader off, the C11 "static" handler is no longer accepted - the universal stays unconsumed and the
    # claim fails closed; only with the C12 ledger hardening off as well does the switch open it.
    q = "Does health insurance cover all dependent children of faculty members?"
    unit = "Comprehensive health insurance is offered for faculty members, their spouse, and up to three dependent children under 18 years old."
    assert local(q, unit) is False
    monkeypatch.setattr(F, "_C11_SLOT_UNIVERSAL", False)
    assert local(q, unit) is False
    monkeypatch.setattr(F, "_C12_LEDGER", False)
    assert local(q, unit) is True


def test_a_frequency_word_never_disappears_silently(monkeypatch):
    q = "Does suspension from duty always result in the suspension of salary?"
    explicit = "Suspension from duty always results in the suspension of salaries."
    assert local(q, explicit) is True and local(q, SUSPENSION) is False
    monkeypatch.setattr(F, "_C11_FREQUENCY", False)
    assert local(q, SUSPENSION) is False and local(q, explicit) is False      # unread, the ledger keeps it closed
    monkeypatch.setattr(F, "_C11_LEDGER", False)
    assert local(q, SUSPENSION) is False                                     # the universal-frequency force still meets "unless"
    monkeypatch.setattr(F, "_C11_UNCONDITIONAL", False)
    assert local(q, SUSPENSION) is True


@pytest.mark.parametrize("switch,question,quote", [
    ("_C11_ALTERNATIVES", "Is the housing allowance paid only once?", "The housing allowance is paid once, and it may be paid again with the Dean's approval."),
    ("_C11_EXCEPTION_WINDOW", "Is annual leave always approved in January?",
     "Annual leave is approved in January. In exceptional cases, the Dean approves annual leave in March."),
])
def test_each_window_switch_turns_its_own_veto_off(monkeypatch, switch, question, quote):
    assert window(question, quote) is CONTRADICTED
    monkeypatch.setattr(F, switch, False)
    assert window(question, quote) is COMPATIBLE


# ---------------------------------------------------------------------------
# Fixture: R10 / DR2 cases, conservative losses, avoidable-loss guards
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def specialists():
    corpus = corpus_from(registry_and_map(), C11_CORPUS_PAGES)
    return {"raw": make_specialist(corpus), "high": make_specialist(corpus, ScriptedReranker([], default=9.0))}


CATCHES = [
    "Does health insurance cover all dependent children of faculty members?", "Is the appointment of clinical faculty limited to Medicine and Law?",
    "Does suspension from duty always result in the suspension of salary?", "Do all senior administrative appointments require Chancellor approval?",
    "Can the Faculty Grievance Committee always contact witnesses upon request?", "Is sick leave without pay granted in all circumstances?",
    "Does every senior administrative appointment require Chancellor approval?", "Does health insurance cover every dependent child of a faculty member?",
    "Is the appointment of clinical faculty limited to Medicine, Dental Medicine and Pharmacy?", "Is the department chair appointment renewable only once?",
]


@pytest.mark.parametrize("question", CATCHES)
def test_the_r10_and_dr2_wrong_answers_are_partial_with_the_composite_message(specialists, question):
    for ranking in ("raw", "high"):
        r = run(specialists[ranking], question)
        assert r.status is not SUPPORTED, (ranking, question)
        assert any("the asked claim" in m or "vetoes the asked claim" in m for m in r.missing), (ranking, r.missing)


CONSERVATIVE = [
    "Is the end-of-service gratuity always one month's salary per year of service?", "Is the maximum children's education allowance 75,000 Dirhams for all children?",
    "Is the visiting contract extended with Chancellor approval?", "Is the department chair's appointment renewable only once?",
    "Is additional sick leave without pay up to 30 days?",            # D-099: stated only in special circumstances (was a C11 guard)
]


@pytest.mark.parametrize("question", CONSERVATIVE)
def test_the_pre_classified_conservative_losses_are_partial(specialists, question):
    for ranking in ("raw", "high"):
        assert run(specialists[ranking], question).status is not SUPPORTED, (ranking, question)


GUARDS = [
    "Is the appointment of clinical faculty limited to Medicine, Dental Medicine, Pharmacy, Physiotherapy, Nursing, and Law?",
    "Does misconduct not require intent?", "Is the probation period one year?", "Can the probation period be extended to two years?",
    "Is the Dean's appointment renewable only once?",
    "Is the maximum children's education allowance 75,000 Dirhams for all qualifying children?",
    "Does each college form a Department and College Hiring Committee at the beginning of each year?",
    "Must faculty on probation give six months' notice to resign?", "Is the visiting contract extendable subject to Chancellor approval?",
    "Can a visiting contract be extended only with Chancellor approval?", "Can additional sick leave be granted without pay?",
    "Are female faculty employed full time entitled to sixty days of maternity leave with full pay?",
]


@pytest.mark.parametrize("question", GUARDS)
def test_the_avoidable_loss_guards_stay_supported(specialists, question):
    for ranking in ("raw", "high"):
        r = run(specialists[ranking], question)
        assert r.status is SUPPORTED, (ranking, question, r.missing)


def test_the_composite_counts_on_a_unit_only_with_every_other_detail():
    unit = "Faculty members must submit the annual leave request."
    claim = composite("Must faculty members submit the annual leave request?")
    assert F._supporting_only(claim, [Detail("x", "x", lambda u: "request" in u)]).check(unit) is True
    gated = F._supporting_only(claim, [Detail("y", "y", lambda u: False)])
    assert gated.check(unit) is False and gated.check.ungated(unit) is True        # its own reading is kept for the pronoun pair


@pytest.mark.parametrize("question", [
    "Does the University Central Committee forward the promotion decision to the Board of Trustees?",
    "Does the Vice Chancellor for Academic Affairs approve additional maternity leave?", "Have I used up my Hajj leave?",
    "Is the maximum children's education allowance 75,000 Dirhams for all children?",      # a D-098 catch keeps its evidence
])
def test_a_partial_answer_reports_the_same_units_and_quotes(specialists, monkeypatch, question):
    # The composite decides support only: a partial answer keeps the units, quotes and coverage it has without it.
    label = "the asked claim stated in one statement"
    for ranking in ("raw", "high"):
        on = run(specialists[ranking], question)
        monkeypatch.setattr(F, "_C11_JOINT", False)
        off = run(specialists[ranking], question)
        monkeypatch.setattr(F, "_C11_JOINT", True)
        assert on.status is not SUPPORTED and on.findings, (ranking, question)
        assert [(f.evidence_quote, [x for x in f.metadata["details_covered"] if x != label]) for f in on.findings] == [
            (f.evidence_quote, f.metadata["details_covered"]) for f in off.findings], (ranking, question)


# ---------------------------------------------------------------------------
# Synthetic classes and split binding on crafted pages (HR page 107)
# ---------------------------------------------------------------------------
SYNTHETIC = [
    ("Full-time faculty must take their annual leave in the summer, while part-time faculty may take their annual leave in the summer.",
     [("Must part-time faculty take their annual leave in the summer?", False), ("May part-time faculty take their annual leave in the summer?", True),
      ("Must full-time faculty take their annual leave in the summer?", True)]),
    ("The housing allowance is limited to new faculty members.",
     [("Is the housing allowance limited to all faculty members?", False), ("Is the housing allowance limited to faculty members?", False)]),
    ("The housing advance is restricted to faculty members with dependent children.",
     [("Is the housing advance restricted to faculty members with dependent children and spouses?", False),
      ("Is the housing advance restricted to all faculty members?", False)]),
    ("The housing advance is paid once, and it may be paid a second time with the Dean's approval.", [("Is the housing advance paid only once?", False)]),
    ("The housing allowance is paid once, and it may be paid again with the Dean's approval.", [("Is the housing allowance paid only once?", False)]),
    ("The housing allowance is paid once, and a second payment may be approved by the Dean.", [("Is the housing allowance paid only once?", False)]),
    ("Unless the Dean objects, the housing advance is paid in January.", [("Is the housing advance always paid in January?", False)]),
    ("All new faculty members must complete the annual leave training only if they work in laboratories.",
     [("Must all new faculty members complete the annual leave training?", False)]),
    ("Faculty members must submit the annual leave plan in September of each year. New faculty members must also submit the annual leave plan within "
     "one month of joining.", [("Must faculty members submit the annual leave plan only in September?", False)]),
    ("Faculty members must submit the annual leave request, and they may take unpaid leave in the summer.",
     [("Must faculty members take unpaid leave in the summer?", False), ("May faculty members take unpaid leave in the summer?", True),
      ("Must faculty members submit the annual leave request?", True)]),
]


@pytest.mark.parametrize("lines,questions", SYNTHETIC)
def test_the_synthetic_classes_and_split_binding_on_crafted_pages(lines, questions):
    corpus = corpus_from(registry_and_map(), [page(107, lines)])
    for reranker in (None, ScriptedReranker([], default=9.0)):
        sp = make_specialist(corpus, reranker)
        for question, supported in questions:
            assert (run(sp, question).status is SUPPORTED) is supported, question


# ---------------------------------------------------------------------------
# Metamorphic: adding an operator never adds support; a safe frame keeps the status
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base,variant", [
    ("Is the probation period one year?", "Is the probation period always one year?"),
    ("Can the probation period be extended to two years?", "Can the probation period always be extended to two years?"),
    ("Is the Dean's appointment renewable only once?", "Is the Dean's appointment normally renewable only once?"),
    ("Does each college form a Department and College Hiring Committee at the beginning of each year?",
     "Does each college fully form a Department and College Hiring Committee at the beginning of each year?"),
])
def test_an_added_operator_never_adds_support(specialists, base, variant):
    for ranking in ("raw", "high"):
        assert run(specialists[ranking], base).status is SUPPORTED, base
        assert run(specialists[ranking], variant).status is not SUPPORTED, variant


@pytest.mark.parametrize("question", ["Is the probation period one year?", "Must all faculty appointments be assigned a title?",
                                      "Is the Dean's appointment renewable only once?"])
def test_a_safe_leading_frame_reads_the_same_claim(question):
    framed = frame("According to HR policy, " + question[0].lower() + question[1:])
    plain = frame(question)
    assert framed.frame == "according to hr policy" and plain.frame is None
    assert (framed.force, framed.frequency, framed.anchor_roots, [(e.cls, e.text, e.handler) for e in framed.ledger]) == (
        plain.force, plain.frequency, plain.anchor_roots, [(e.cls, e.text, e.handler) for e in plain.ledger])
