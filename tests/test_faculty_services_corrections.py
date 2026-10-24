"""
Faculty Services & HR bounded corrections (Milestone 4, Step 4.14, D-100). The two reproducible defects of Step 4.14-A and
their corrections, each behind its own switch: P1, the asked entity of a claim with no role actor and no object is stated in
the supporting clause ("Is the housing allowance paid once?" against "... receive the housing allowance monthly and the
furniture allowance is paid once"); P2, a bracketed numeral that repeats its number words is dropped before the readers
("two (2) years"). The controls on the same pages, the reader helpers, the switches end to end, and the Step 4.14-A guard
lists: false specific claims never supported, true ones still supported, under both stand-in rankings.

The reconstructed 194-case campaign and its runner live in ``qa/hr_4_14_a/``.
"""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import claim_model as C
from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus

from faculty_services_support import C12_CORPUS_PAGES, ScriptedReranker, corpus_from, make_specialist, page, registry_and_map, run

SUPPORTED = FindingStatus.SUPPORTED
ENTITY_SENTENCE = "Faculty members receive the housing allowance monthly and the furniture allowance is paid once."
ENTITY_PAGE = [page(107, ENTITY_SENTENCE)]


def _specialists(pages):
    corpus = corpus_from(registry_and_map(), pages)
    return {"raw": make_specialist(corpus), "high": make_specialist(corpus, ScriptedReranker([], default=9.0))}


@pytest.fixture(scope="module")
def fixture_specialists():
    return _specialists(C12_CORPUS_PAGES)


@pytest.fixture(scope="module")
def entity_specialists():
    return _specialists(C12_CORPUS_PAGES + ENTITY_PAGE)


def frame_of(question):
    text = F.specific_claim_text(question)
    return F.read_claim_frame(text, F._c12_read_claim(text))


# ---------------------------------------------------------------------------
# P1 (HR-414A-D1): a statement about another entity in the same sentence never supports
# ---------------------------------------------------------------------------
def test_p1_a_statement_about_another_entity_in_the_same_sentence_never_supports(entity_specialists):
    for ranking, sp in entity_specialists.items():
        r = run(sp, "Is the housing allowance paid once?")
        assert r.status is not SUPPORTED, (ranking, r.status, r.missing)
        assert "another subject" in " ".join(r.missing), (ranking, r.missing)


@pytest.mark.parametrize("question,supported", [
    ("Is the furniture allowance paid once?", True),             # the right entity of the same sentence
    ("Is the furniture allowance paid monthly?", False),
    ("Is the shipment allowance paid monthly?", False),
    ("Do faculty members receive the housing allowance monthly?", True),
    ("Does health insurance cover the spouse of faculty members?", True),   # a verb the claim reader does not anchor: no entity veto
])
def test_p1_controls_on_the_crafted_page(entity_specialists, question, supported):
    for ranking, sp in entity_specialists.items():
        r = run(sp, question)
        assert (r.status is SUPPORTED) is supported, (ranking, question, r.status, r.missing)


def test_p1_the_switch_turns_the_entity_binding_off_end_to_end(entity_specialists, monkeypatch):
    monkeypatch.setattr(F, "_C12_ENTITY_SUBJECT", False)
    for ranking, sp in entity_specialists.items():
        assert run(sp, "Is the housing allowance paid once?").status is SUPPORTED, ranking


@pytest.mark.parametrize("question,phrase,roots", [
    ("Is the housing allowance paid once?", "the housing allowance", ("housing", "allowance")),
    ("Is the probation period one year?", "the probation period", ("probation", "period")),
    ("Is the Dean's monthly compensation AED 10,000?", "the dean's monthly compensation aed", ("dean", "monthly", "compensation")),
])
def test_p1_the_asked_entity_is_the_noun_phrase_before_the_anchor_without_determiners_or_currency(question, phrase, roots):
    frame = frame_of(question)
    got_phrase, got_roots = F._c12_asked_entity(frame)
    assert got_phrase == phrase
    assert set(got_roots) == {F._root(w) for w in roots}, got_roots


@pytest.mark.parametrize("question,bound", [
    ("Is the housing allowance paid once?", True),               # no actor, no object: the entity binds
    ("Does HR draft employment offers?", False),                 # a role actor: the D-098 subject reader binds the actor
    ("Do faculty members receive a housing allowance?", False),  # a population subject with a head and an asked object
])
def test_p1_the_subject_override_applies_only_to_a_claim_with_no_actor_and_no_object(question, bound):
    frame = frame_of(question)
    assert ("subject" in F._c12_dimension_checks(frame, F.specific_claim_text(question), {})) is bound


def test_p1_a_single_clause_sentence_is_left_to_the_unit_level_reader():
    frame = frame_of("Is the housing allowance paid once?")
    subject = F._c12_dimension_checks(frame, "Is the housing allowance paid once?", {})["subject"]
    one_clause = F.read_clause_statements("The furniture allowance is paid once.", frame)
    assert one_clause and subject(frame, one_clause[0]) is None
    two_clauses = F.read_clause_statements(ENTITY_SENTENCE, frame)
    other = [s for s in two_clauses if "furniture" in s.text]
    assert other and subject(frame, other[0]).compatibility is C.Compatibility.UNKNOWN


# ---------------------------------------------------------------------------
# P2 (HR-414A-D2): a bracketed numeral is bound like its number word
# ---------------------------------------------------------------------------
def test_p2_a_parenthesised_numeral_is_bound_like_its_number_word(fixture_specialists):
    for ranking, sp in fixture_specialists.items():
        r = run(sp, "Is the probation period two (2) years?")
        assert r.status is not SUPPORTED, (ranking, r.status, r.missing)


@pytest.mark.parametrize("question,supported", [
    ("Is the probation period two years?", False),
    ("Is the probation period 2 years?", False),
    ("Is the probation period 24 months?", False),
    ("Is the faculty probation period one year?", True),
    ("Is the faculty probation period 1 year?", True),
    ("Is the probation period one (1) year?", True),
    ("Is maternity leave with full pay thirty (30) days?", False),
    ("Is the Dean's monthly compensation five thousand (5,000) AED?", False),
])
def test_p2_controls(fixture_specialists, question, supported):
    for ranking, sp in fixture_specialists.items():
        r = run(sp, question)
        assert (r.status is SUPPORTED) is supported, (ranking, question, r.status, r.missing)


def test_p2_the_switch_turns_the_numeral_normalisation_off_end_to_end(fixture_specialists, monkeypatch):
    monkeypatch.setattr(F, "_C12_PAREN_NUMERAL", False)
    for ranking, sp in fixture_specialists.items():
        assert run(sp, "Is the probation period two (2) years?").status is SUPPORTED, ranking


@pytest.mark.parametrize("text,normalised", [
    ("Is the probation period two (2) years?", "Is the probation period two years?"),
    ("Is maternity leave with full pay thirty (30) days?", "Is maternity leave with full pay thirty days?"),
    ("five thousand (5,000) Dirhams", "five thousand Dirhams"),
    ("one thousand five hundred (1,500) Dirhams", "one thousand five hundred Dirhams"),
    ("one hundred and twenty (120) hours", "one hundred and twenty hours"),
    ("twenty-one (21) days", "twenty-one days"),
    ("Thirty (30) Days", "Thirty Days"),
    ("one (2) years", "one (2) years"),                             # a numeral that does not repeat the words is kept
    ("Article (3) six months", "Article (3) six months"),
    ("within the first (1) month", "within the first (1) month"),  # an ordinal is no number word
    ("1,500 (1500) Dirhams", "1,500 (1500) Dirhams"),
    ("the probation period (two years)", "the probation period (two years)"),
])
def test_p2_only_a_numeral_that_repeats_its_number_words_is_dropped(text, normalised):
    assert F._c12_drop_parenthesised_numeral(text) == normalised


@pytest.mark.parametrize("words,value", [
    (["two"], 2), (["thirty"], 30), (["thirty", "thousand"], 30000), (["five", "thousand"], 5000), (["one", "thousand", "five", "hundred"], 1500),
    (["one", "hundred", "and", "twenty"], 120), (["two", "hundred", "thousand"], 200000), ([], None), (["first"], None), (["two", "years"], None),
])
def test_p2_number_words_value(words, value):
    assert F._c12_number_words_value(words) == value


def test_p2_a_none_reading_stays_none_and_the_cache_is_untouched():
    assert F.specific_claim_text("Aren't terminated faculty entitled to two (2) return tickets?") is None
    assert F.is_specific_claim("Is the probation period two (2) years?")


# ---------------------------------------------------------------------------
# Step 4.14-A guard lists on the verbatim fixture pages, both rankings
# ---------------------------------------------------------------------------
NEVER_SUPPORTED = [
    "Does the department chair draft employment offers?", "Does HR appoint department chairs?", "Does the Chancellor appoint the Dean?",
    "Does the Dean approve additional maternity leave without pay?", "Is the Associate Dean's monthly compensation AED 10,000?",
    "Does HR approve employment offers?", "Does HR pay for hotel accommodations for new faculty members?", "Do faculty members receive a furniture allowance every year?",
    "Is the furniture allowance 40,000 Dirhams?", "Is the faculty probation period two years?", "Is maternity leave with full pay ninety calendar days?",
    "Is the children's education allowance 75,000 Dirhams per child?", "Do faculty members receive a housing allowance of 30,000 Dirhams?",
    "Is health insurance offered for only three dependent children?", "Are faculty members entitled to 30 days of sick leave without pay?",
    "Does misconduct require intent?", "Are visiting faculty members eligible for the bonus award?", "Is the grievance hearing open to the public?",
    "Hey there, is the furniture allowance 40,000 Dirhams?", "I heard the resignation notice is three months. Is that right?",
    "Is the probation period two years or one year?", "Is the probation period one year or two years?", "So the furniture allowance is 40,000 Dirhams?",
    "Does the Chancellor exercise authority over the Faculty Employment Policy?", "Is contract renewal based on a single year's performance?",
    "Does the department chair present promotion applications to the College Academic Promotions Committee?",
]


@pytest.mark.parametrize("question", NEVER_SUPPORTED)
def test_false_specific_claims_are_never_supported(fixture_specialists, question):
    for ranking, sp in fixture_specialists.items():
        assert run(sp, question).status is not SUPPORTED, (ranking, question)


STILL_SUPPORTED = [
    "Is the faculty probation period one year?", "Must Deans hold the rank of full professor?", "Does HR draft employment offers?",
    "Is the Dean's monthly compensation AED 10,000?", "Is the one-time furniture allowance 30,000 Dirhams?",
    "Are female faculty employed on a full-time basis entitled to sixty calendar days of maternity leave with full pay?",
    "Is health insurance offered for up to three dependent children under 18 years old?", "Is the Dean's teaching load reduction nine credit hours per semester?",
    "Is the clinical allowance 10% to 20% of the basic salary?", "Does the Faculty Grievance Committee hold the hearing within thirty days of receiving the written grievance?",
    "Does HR arrange hotel accommodations for new faculty members?", "Is the faculty probation period 1 year?",
]


@pytest.mark.parametrize("question", STILL_SUPPORTED)
def test_true_specific_claims_stay_supported(fixture_specialists, question):
    for ranking, sp in fixture_specialists.items():
        r = run(sp, question)
        assert r.status is SUPPORTED, (ranking, question, r.status, r.missing)


def test_p1_a_root_the_sentence_states_nowhere_is_no_subject_mismatch():
    """"Does health insurance cover ...": the claim reader anchors on the object noun, so the entity phrase carries the verb
    "cover"; the page-106 sentence never states "cover", so the subject reader leaves the statement to the other readers."""
    frame = frame_of("Does health insurance cover all children of faculty members?")
    phrase, roots = F._c12_asked_entity(frame)
    assert phrase == "health insurance cover" and F._root("cover") in roots            # the leaked verb, documented
    subject = F._c12_dimension_checks(frame, F.specific_claim_text("Does health insurance cover all children of faculty members?"), {})["subject"]
    sentence = "• Health Insurance: Comprehensive health insurance is offered for faculty members, their spouse, and up to three dependent children under 18 years old."
    statements = F.read_clause_statements(sentence, frame)
    assert statements and all(subject(frame, s) is None for s in statements)


def test_p1_the_sk12_universal_claim_still_fails_on_its_operator_not_on_the_subject(fixture_specialists):
    for ranking, sp in fixture_specialists.items():
        r = run(sp, "Does health insurance cover all children of faculty members?")
        assert r.status is not SUPPORTED and "another subject" not in " ".join(r.missing), (ranking, r.missing)


# ---------------------------------------------------------------------------
# The three D-099 switches no reconstructed case killed (Step 4.14 investigation)
# ---------------------------------------------------------------------------
def test_the_hedge_switch_changes_the_reading_not_the_status(fixture_specialists, monkeypatch):
    """_C12_HEDGE: on, a hedged question is read with the hedge as its frequency operator and fails on that dimension; off, the
    claim cannot be read and the gate fails closed. Both are partial: the mutant is equivalent at the status level."""
    q = "In general, must Deans hold the rank of full professor?"
    text = F.specific_claim_text(q)
    assert F._c12_split_hedge(text)[0] == "In general" and F._c12_read_claim(text) is not None
    for ranking, sp in fixture_specialists.items():
        r = run(sp, q)
        assert r.status is not SUPPORTED and any("frequency" in m for m in r.missing), (ranking, r.missing)
    monkeypatch.setattr(F, "_C12_HEDGE", False)
    assert F._c12_split_hedge(text) == (None, text) and F._c12_read_claim(text) is None
    for ranking, sp in fixture_specialists.items():
        r = run(sp, q)
        assert r.status is not SUPPORTED and any("could not be read" in m for m in r.missing), (ranking, r.missing)


def test_the_leads_switch_is_visible_at_clause_level_only(fixture_specialists, monkeypatch):
    """_C12_LEADS: with the Coordinator's split, on, the lead is re-attached and the one clause fails on its frequency; off, the
    lead clause is not searched and the yes/no clause is supported without its hedge. The task status is partial either way."""
    q, clauses = "In general, must Deans hold the rank of full professor?", ["In general", "must Deans hold the rank of full professor?"]
    for ranking, sp in fixture_specialists.items():
        r = run(sp, q, clauses=clauses)
        assert r.status is not SUPPORTED and any("frequency" in m for m in r.missing) and not any("not searched" in m for m in r.missing), (ranking, r.missing)
    monkeypatch.setattr(F, "_C12_LEADS", False)
    for ranking, sp in fixture_specialists.items():
        r = run(sp, q, clauses=clauses)
        assert r.status is not SUPPORTED and any("not searched" in m for m in r.missing) and not any("frequency" in m for m in r.missing), (ranking, r.missing)


NEGATIVE_PAGE = [page(107, "Faculty members are not entitled to the furniture allowance if they join after the first semester.")]


def test_the_negative_condition_switch_is_killed_by_a_negative_claim_stated_only_under_a_condition(monkeypatch):
    """_C12_NEGATIVE_CONDITION: a negative claim whose only evidence is conditional is partial; with the switch off it is supported."""
    q = "Are faculty members not entitled to the furniture allowance?"
    sps = _specialists(C12_CORPUS_PAGES + NEGATIVE_PAGE)
    for ranking, sp in sps.items():
        r = run(sp, q)
        assert r.status is not SUPPORTED and any("restriction" in m for m in r.missing), (ranking, r.missing)
    monkeypatch.setattr(F, "_C12_NEGATIVE_CONDITION", False)
    for ranking, sp in sps.items():
        assert run(sp, q).status is SUPPORTED, ranking


# ---------------------------------------------------------------------------
# HR-414-N1 (D-101, Step 4.14-N1): a condition that restates the asked entity is still a condition
# ---------------------------------------------------------------------------
N1_SENTENCE = "Faculty members are not entitled to the housing allowance if they receive university housing."
N1_CASES = [   # (question, crafted sentence): supported before D-101 because the condition's only distinguishing word was dropped
    ("Are faculty members not entitled to the housing allowance?", N1_SENTENCE),
    ("Do faculty members receive a housing allowance?", "Faculty members receive a housing allowance if housing is not provided."),                       # affirmative
    ("Are faculty members not entitled to the furniture allowance?", "Faculty members are not entitled to the furniture allowance if the furniture is provided by the university."),
    ("Is the housing allowance paid monthly?", "The housing allowance is paid monthly if the housing allowance is approved by the Dean."),                 # heavy overlap
    ("Is the shipment allowance paid?", "The shipment allowance is paid if the shipment is insured."),                                                    # the same word on both sides
    ("Is the shipment allowance paid in full?", "The shipment allowance is paid in full if the shipment allowance claim is approved."),                  # the entity restated whole
    ("Are visiting faculty members eligible for the bonus award?", "Visiting faculty members are eligible for the bonus award if the award committee nominates visiting faculty."),
]


@pytest.mark.parametrize("question,sentence", N1_CASES)
def test_n1_a_claim_stated_only_under_a_condition_that_restates_its_entity_never_supports(question, sentence):
    for ranking, sp in _specialists(C12_CORPUS_PAGES + [page(107, sentence)]).items():
        r = run(sp, question)
        assert r.status is not SUPPORTED, (ranking, question, r.status)
        assert any("restriction" in m for m in r.missing), (ranking, question, r.missing)


@pytest.mark.parametrize("question,sentence", N1_CASES)
def test_n1_the_own_content_switch_turns_the_correction_off_end_to_end(monkeypatch, question, sentence):
    monkeypatch.setattr(F, "_C12_OWN_CONTENT", False)
    for ranking, sp in _specialists(C12_CORPUS_PAGES + [page(107, sentence)]).items():
        assert run(sp, question).status is SUPPORTED, (ranking, question)


@pytest.mark.parametrize("question,sentence,supported", [
    ("Do faculty members receive a housing allowance if housing is not provided?", "Faculty members receive a housing allowance if housing is not provided.", True),   # the question asks the condition
    ("Is unpaid special leave granted if the employee has no unused vacation leave?", None, True),                                                                  # real page 107
    ("Are faculty members not entitled to the housing allowance if they receive university housing?", N1_SENTENCE, False),                                           # unchanged: the asked role check
    ("Are faculty members entitled to a housing allowance?", "Faculty members are entitled to a housing allowance unless university housing is provided.", False),  # partial before and after
    ("Are visiting faculty members not eligible for the bonus award?", None, True),                                                                                 # unconditional negative, real page
])
def test_n1_asked_conditions_and_unconditional_claims_are_unchanged(question, sentence, supported):
    for ranking, sp in _specialists(C12_CORPUS_PAGES + ([page(107, sentence)] if sentence else [])).items():
        r = run(sp, question)
        assert (r.status is SUPPORTED) is supported, (ranking, question, r.status, r.missing)


def test_n1_the_condition_keeps_its_subject_and_scope_and_only_its_distinguishing_word_is_judged():
    q = "Are faculty members not entitled to the housing allowance?"
    text = F.specific_claim_text(q)
    claim = F._c12_read_claim(text)
    frame = F.read_claim_frame(text, claim)
    (s,) = F.read_clause_statements(N1_SENTENCE, frame)
    assert s.subject_text.strip().lower().startswith("faculty members")           # the condition did not become the subject
    (r,) = s.restrictions
    assert r.kind is C.RestrictionKind.SUFFICIENT and r.text == "if they receive university housing"
    main = F._c11_main_roots(s)
    assert {F._root("housing"), F._root("allowance"), F._root("entitled")} <= set(main) and F._root("university") not in main
    claim_roots = F._roots(text, frozenset())
    assert F._c11_own(r.text, r.roots, claim_roots) is True                      # the D-098 rule alone: half of {hous, university} is the claim's
    assert F._c11_own(r.text, r.roots, claim_roots, main) is False               # D-101: only "university" distinguishes the condition; it is not asked
    assert C.RestrictionKind.SUFFICIENT in F._c11_unasked(s, claim_roots, F.cue_text(text.lower()), claim)


@pytest.mark.parametrize("restriction,main,claim,own", [
    ("if they receive university housing", "Faculty members are not entitled to the housing allowance", "Are faculty members not entitled to the housing allowance?", False),
    ("if housing is not provided", "Faculty members receive a housing allowance", "Do faculty members receive a housing allowance?", False),
    ("if housing is not provided", "Faculty members receive a housing allowance", "Do faculty members receive a housing allowance if housing is not provided?", False),  # asked: exempt through _c12_asked_condition, not through own content
    ("if they join after the first semester", "Faculty members are not entitled to the furniture allowance", "Are faculty members not entitled to the furniture allowance?", False),
    ("if the employee has no unused vacation leave", "Unpaid special leave may be granted upon approval from appropriate University authorities",
     "Is unpaid special leave granted if the employee has no unused vacation leave?", True),                                                                           # the distinguishing words are the asked ones
])
def test_n1_own_content_is_judged_on_the_words_that_distinguish_the_restriction(restriction, main, claim, own):
    # restriction roots as the statement reader keeps them (claim stop words removed); claim and main-clause roots as the
    # dimension readers compute them (no stop list)
    assert F._c11_own(restriction, F._roots(restriction), F._roots(claim, frozenset()), F._roots(main, frozenset())) is own


def test_n1_without_main_clause_roots_the_d098_rule_is_unchanged():
    def own(restriction, claim):
        return F._c11_own(restriction, F._roots(restriction), F._roots(claim, frozenset()))
    assert own("if they receive university housing", "Are faculty members not entitled to the housing allowance?") is True
    assert own("if they join after the first semester", "Are faculty members not entitled to the furniture allowance?") is False


def test_n1_the_correction_never_grants_an_exemption_the_d098_rule_did_not():
    """Removing the main-clause words raises the share of claim words; D-101 must not turn that into a new exemption."""
    restriction, main, claim = "if approved by the Dean and the Chancellor", "The Dean and the Chancellor grant leave", "Is leave granted when approved?"
    args = (restriction, F._roots(restriction), F._roots(claim, frozenset()))
    assert F._c11_own(*args) is False                                            # D-098: 1 of 3 distinguishing words is the claim's
    assert F._c11_own(*args, F._roots(main, frozenset())) is False               # D-101 alone would see 1 of 1; the result stays False


# ---------------------------------------------------------------------------
# HR-R2-W1 (D-102, Step 4.14-W1, Option A): a "when" / "whenever" condition is a condition, as "if" is
# ---------------------------------------------------------------------------
W1_SENTENCE = "Faculty members are not entitled to the housing allowance when they receive university housing."
W1_OMITTED = [   # (question, crafted sentence): the evidence holds only under a "when" condition the question leaves out
    ("Are faculty members not entitled to the housing allowance?", W1_SENTENCE),                                                         # A state, negative
    ("Do faculty members receive a housing allowance?", "Faculty members receive a housing allowance whenever housing is not provided."),  # A state, "whenever"
    ("Must HR pay the housing allowance?", "HR must pay the housing allowance when university housing is not available."),                 # A state, deontic
    ("Is a relocation allowance paid to faculty members?", "A relocation allowance is paid to faculty members when they join the University."),  # B event
    ("Do faculty members receive a housing allowance?", "Faculty members receive a housing allowance when housing is not provided."),     # E positive
]


@pytest.mark.parametrize("question,sentence", W1_OMITTED)
def test_w1_a_claim_stated_only_under_an_omitted_when_condition_never_supports(question, sentence):
    for ranking, sp in _specialists(C12_CORPUS_PAGES + [page(107, sentence)]).items():
        r = run(sp, question)
        assert r.status is not SUPPORTED, (ranking, question, r.status)
        assert any("restriction" in m for m in r.missing), (ranking, question, r.missing)


@pytest.mark.parametrize("question,sentence", W1_OMITTED)
def test_w1_the_when_condition_switch_turns_the_correction_off_end_to_end(monkeypatch, question, sentence):
    monkeypatch.setattr(F, "_C12_WHEN_CONDITION", False)
    for ranking, sp in _specialists(C12_CORPUS_PAGES + [page(107, sentence)]).items():
        assert run(sp, question).status is SUPPORTED, (ranking, question)


W1_UNCHANGED = [   # (question, crafted sentence or None for the fixture pages, supported): the same status with the rule on and off
    ("Do faculty members receive a housing allowance when housing is not provided?", "Faculty members receive a housing allowance when housing is not provided.", True),   # C asked
    ("Is a relocation allowance paid to faculty members when they join the University?", "A relocation allowance is paid to faculty members when they join the University.", True),  # C asked event
    ("May faculty members receive a housing allowance?", "Faculty members may receive a housing allowance when housing is not provided.", True),   # existential: a possibility
    ("Does a faculty member's primary and continuous appointment remain their faculty role?", None, True),          # real page 71: "even when" concedes
    ("Are faculty members entitled to a housing allowance?", "Faculty members are entitled to a housing allowance unless university housing is provided.", False),  # D unless
    ("Are faculty members not entitled to the housing allowance?", N1_SENTENCE, False),                                # D "if" (N1)
    ("Are faculty members not entitled to the housing allowance when they receive university housing?", W1_SENTENCE, False),  # asked negative: the asked-role check
    ("What is the housing allowance policy?", "Faculty members receive a housing allowance when housing is not provided.", False),  # H general question
]


@pytest.mark.parametrize("question,sentence,supported", W1_UNCHANGED)
def test_w1_asked_conditions_possibilities_concessions_and_controls_are_unchanged(monkeypatch, question, sentence, supported):
    sps = _specialists(C12_CORPUS_PAGES + ([page(107, sentence)] if sentence else []))
    for ranking, sp in sps.items():
        assert (run(sp, question).status is SUPPORTED) is supported, (ranking, question)
    monkeypatch.setattr(F, "_C12_WHEN_CONDITION", False)
    for ranking, sp in sps.items():
        assert (run(sp, question).status is SUPPORTED) is supported, (ranking, question, "switch off")


def test_w1_only_a_when_or_whenever_trigger_is_a_when_condition():
    def restriction(question, unit):
        text = F.specific_claim_text(question)
        frame = F.read_claim_frame(text, F._c12_read_claim(text))
        (s,) = F.read_clause_statements(unit, frame)
        return s, [r for r in s.restrictions if r.kind is C.RestrictionKind.TRIGGER]
    s, (r,) = restriction("Are faculty members not entitled to the housing allowance?", W1_SENTENCE)
    assert F._c12_when_condition(r) and r.text == "when they receive university housing"
    s, (r,) = restriction("Does a faculty member's primary and continuous appointment remain their faculty role?",
                          "A faculty member's primary and continuous appointment remains their faculty role, even when holding an administrative title.")
    assert F._c12_when_condition(r)                                               # read as a trigger by the shared reader ...
    text = F.specific_claim_text("Does a faculty member's primary and continuous appointment remain their faculty role?")
    claim = F._c12_read_claim(text)
    assert C.RestrictionKind.TRIGGER not in F._c12_material_conditions(s, claim, F._roots(text, frozenset()), text.lower())  # ... but "even when" concedes


W1_PAIRS = [(q, s) for q, s in W1_OMITTED] + [(q, s) for q, s, _ok in W1_UNCHANGED if s and F.is_specific_claim(q)] + [   # specific claims: the gate decides
    ("Does the basic salary increase from the following month?", "When a faculty member is promoted, the basic salary increases from the following month."),
    ("When a faculty member is promoted, does the basic salary increase from the following month?",
     "When a faculty member is promoted, the basic salary increases from the following month."),
    ("Does confirmation of employment include the probation period?",
     "Employment confirmation includes the probation period when calculating the employee's End of Service Benefit."),
    ("Is the faculty probation period one year?", "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."),
]


@pytest.mark.parametrize("question,unit", W1_PAIRS)
def test_w1_the_rule_never_creates_support(monkeypatch, question, unit):
    """D-102 may only withdraw support: whenever the composite covers the unit with the rule on, it covers it with the rule off, and
    every statement's material conditions with the rule on contain those with the rule off."""
    detail = F._specific_claim_support_gate(question)[2]
    assert detail is not None, question                                          # the property is about specific claims
    on = detail.check(unit)
    text = F.specific_claim_text(question)
    claim = F._c12_read_claim(text)
    frame = F.read_claim_frame(text, claim) if claim is not None else None
    statements = F.read_clause_statements(unit, frame) if frame is not None else ()
    roots, low = F._roots(text, frozenset()), text.lower()
    material_on = [F._c12_material_conditions(s, claim, roots, low) for s in statements]
    monkeypatch.setattr(F, "_C12_WHEN_CONDITION", False)
    off = F._specific_claim_support_gate(question)[2].check(unit)
    material_off = [F._c12_material_conditions(s, claim, roots, low) for s in statements]
    assert (not on) or off, (question, "supported only with the rule on")
    assert all(b <= a for a, b in zip(material_on, material_off)), (question, material_on, material_off)
