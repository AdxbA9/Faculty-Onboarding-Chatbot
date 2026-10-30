"""
Faculty Services & HR structured claim model (Milestone 4, Step 4.12-C10, D-097): the claim_model readers and tables, the
HR claim and statement readers, the modality, scope, exclusivity, condition-role and fail-closed operator details, the
window veto, and the known-failure corpus with its controls on the fixture pages under both rankings.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from handbook_bot.agents.specialists import claim_model as C
from handbook_bot.agents.specialists import faculty_services as F
from handbook_bot.agents.specialists.contracts import FindingStatus

from faculty_services_support import C10_PAGES, PAGES, ScriptedReranker, corpus_from, make_specialist, registry_and_map, run

SUPPORTED = FindingStatus.SUPPORTED
M, Q, X, R, K = C.Modality, C.Quantifier, C.ExclusivityFocus, C.ConditionRole, C.OperatorKind
COMPATIBLE, CONTRADICTED, UNKNOWN, NA = C.Compatibility.COMPATIBLE, C.Compatibility.CONTRADICTED, C.Compatibility.UNKNOWN, C.Compatibility.NOT_APPLICABLE
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = os.path.join(REPO, "handbook_bot", "agents", "specialists", "claim_model.py")


def check(builder, question, unit):
    detail = builder(question, F.read_claim(question))
    assert detail is not None, question
    return detail.check(unit)


def window(builder, question, quote):
    detail = builder(question, F.read_claim(question))
    assert detail is not None, question
    return detail.check.window(quote)


# ---------------------------------------------------------------------------
# 1. The module: standard library only, imported by the HR specialist alone
# ---------------------------------------------------------------------------
def test_claim_model_imports_the_standard_library_only():
    tree = ast.parse(open(MODULE, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.add((node.module or "").split(".")[0])
    assert names <= {"__future__", "re", "dataclasses", "enum", "typing"}, names


def test_claim_model_imports_alone_in_a_fresh_interpreter():
    code = ("import sys, handbook_bot.agents.specialists.claim_model as m; "
            "print(sorted(k for k in sys.modules if k.startswith('handbook_bot.agents')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    for name in ("faculty_services", "grounding", "teaching", "research", "coordinator", "dispatch", "evidence"):
        assert name not in out.stdout, name


def test_only_the_hr_specialist_imports_claim_model():
    users = []
    for root, _dirs, files in os.walk(os.path.join(REPO, "handbook_bot")):
        for name in files:
            if name.endswith(".py") and name != "claim_model.py" and "claim_model" in open(os.path.join(root, name), encoding="utf-8").read():
                users.append(name)
    assert users == ["faculty_services.py"]


# ---------------------------------------------------------------------------
# 2. Readers: question modality, statement modality bound to the asked verb
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,modality", [
    ("Must the committee contact witnesses?", M.OBLIGATION), ("Shall the Dean submit the report?", M.OBLIGATION),
    ("Should faculty attend the induction?", M.RECOMMENDATION), ("May faculty apply for the award?", M.PERMISSION),
    ("Can the contract be renewed?", M.PERMISSION), ("Can failure to comply result in a deduction?", M.POSSIBILITY),
    ("May administrative appointments influence the evaluation?", M.POSSIBILITY), ("Are faculty required to submit the report?", M.OBLIGATION),
    ("Are faculty expected to publish annually?", M.RECOMMENDATION), ("Are faculty encouraged to attend workshops?", M.ENCOURAGEMENT),
    ("Are faculty allowed to transfer?", M.PERMISSION), ("Is a clinical faculty member prohibited from transferring?", M.PROHIBITION),
    ("Do I have to give notice?", M.OBLIGATION), ("Is it mandatory to attend the training?", M.OBLIGATION),
    ("Is approval required for unpaid special leave?", M.NONE), ("Does HR compile the required documents for the Employee File?", M.NONE),
    ("Are faculty entitled to annual leave?", M.NONE), ("Does HR draft employment offers?", M.NONE),
])
def test_question_modality(question, modality):
    assert C.ask_modality(question) is modality


@pytest.mark.parametrize("unit,verb,modality", [
    ("Faculty members may apply for the bonus award in consecutive years.", "apply", M.PERMISSION),
    ("While the number of office hours may vary, part-time faculty are equally encouraged to allocate sufficient time.", "allocate", M.ENCOURAGEMENT),
    ("While the number of office hours may vary, part-time faculty are equally encouraged to allocate sufficient time.", "vary", M.POSSIBILITY),
    ("Faculty members intending to resign during probation must provide a six-month notice.", "notice", M.OBLIGATION),
    ("The Faculty Grievance Committee shall hold a hearing within thirty days.", "hold", M.OBLIGATION),
    ("A clinical faculty member shall not be permitted to transfer to a regular faculty member.", "transfer", M.PROHIBITION),
    ("Faculty members are not required to submit the report.", "submit", M.NONE),
    ("Newly appointed faculty members not meeting UoS standards should be notified as early as possible.", "notified", M.RECOMMENDATION),
    ("The faculty probation period is one year, extendable to a maximum of two years.", "extendable", M.PERMISSION),
    ("The Chancellor holds the jurisdiction to appoint national faculty.", "appoint", M.PERMISSION),
    ("Faculty may not travel without approval.", "travel", M.PROHIBITION),
    ("Emeritus faculty may serve ex officio on committees and teach courses as needed.", "teach", M.PERMISSION),
    ("HR compiles all required documents and information to maintain in the Employee File.", "compiles", M.NONE),
    ("Faculty are expected to publish annually.", "publish", M.RECOMMENDATION),
])
def test_statement_modality_is_bound_to_the_verb(unit, verb, modality):
    assert C.statement_modality(unit, unit.index(verb), verb) is modality


@pytest.mark.parametrize("unit,verb,modality", [
    ("Faculty members shall not be permitted to transfer to another college.", "transfer", M.PROHIBITION),
    ("Faculty members cannot transfer to another college.", "transfer", M.PROHIBITION),
    ("Faculty members must not transfer to another college.", "transfer", M.PROHIBITION),
    ("Faculty members are prohibited from transferring to another college.", "transferring", M.PROHIBITION),
    ("Faculty members need not transfer to another college.", "transfer", M.NONE),
    ("Faculty members do not have to transfer to another college.", "transfer", M.NONE),
    ("Faculty members are not obliged to transfer to another college.", "transfer", M.NONE),
])
def test_negated_deontics_are_prohibitions_or_no_obligation(unit, verb, modality):
    assert C.statement_modality(unit, unit.index(verb), verb) is modality


def test_a_modal_governs_only_its_own_clause_and_reach():
    far = "Faculty may apply for the award, and once the committee has reviewed every file and consulted the department, it notifies them."
    assert C.statement_modality(far, far.index("notifies"), "notifies") is M.NONE                  # beyond the reach and the clause
    cond = "If faculty may apply, the committee notifies the applicants."
    assert C.statement_modality(cond, cond.index("notifies"), "notifies") is M.NONE                # the modal of the antecedent
    assert C.unit_modality("Faculty must attend and may also present.") is M.OBLIGATION               # unbound: the strongest marker


def test_a_requirement_states_an_obligation_of_what_it_requires():
    unit = "Termination of faculty contracts for cause requires the approval of the Chancellor."
    assert C.statement_modality(unit, unit.index("approval"), "approval") is M.OBLIGATION
    for text in ("Termination does not require the approval of the Chancellor.", "Termination requires no approval of the Chancellor."):
        assert C.statement_modality(text, text.index("approval"), "approval") is M.NONE, text
    assert C.ask_modality("Is Chancellor approval required to terminate a contract for cause?") is M.NONE       # a requirement predicate
    assert C.ask_modality("Are faculty members required to submit their resignation through the HR portal?") is M.OBLIGATION
    assert check(F.modality_detail, "Must the Chancellor approve the termination of contracts for cause?", unit)


@pytest.mark.parametrize("word,quantifier", [
    ("all", Q.UNIVERSAL), ("every", Q.UNIVERSAL), ("each", Q.UNIVERSAL), ("any", Q.UNIVERSAL), ("some", Q.EXISTENTIAL),
    ("certain", Q.EXISTENTIAL), ("most", Q.PROPORTIONAL), ("several", Q.PROPORTIONAL), ("the", Q.GENERIC), ("faculty", Q.GENERIC),
])
def test_quantifier_of(word, quantifier):
    assert C.quantifier_of(word) is quantifier


def test_every_operator_names_its_handler():
    ops = C.find_operators("Must all faculty members not apply only once if the Dean approves?")
    handled = {(o.kind, o.handled_by) for o in ops}
    assert (K.MODALITY, "modality") in handled and (K.NEGATION, "negation") in handled
    assert any(o.kind is K.QUANTIFIER and o.handled_by for o in ops) and any(o.kind is K.EXCLUSIVITY and o.handled_by for o in ops)
    assert not C.unhandled(ops)


# ---------------------------------------------------------------------------
# 3. Readers: quantifier slots, exclusivity focus, condition roles, approval gates, operators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Are all faculty members eligible for the award?", [("all", "population")]),
    ("Can every faculty member apply?", [("every", "population")]),
    ("Are some faculty members not eligible?", [("some", "population")]),
    ("Does the College Council form the committee each September?", [("each", "time")]),
    ("Is the committee formed at the beginning of each academic year?", [("each", "time")]),
    ("Is the contract renewed every two years?", [("every", "time")]),
    ("Does HR manage all communication with candidates?", [("all", "object")]),
    ("Does the University prohibit any form of harassment?", [("any", "object")]),
    ("Are most faculty members eligible?", [("most", "proportional")]),
    ("Is sick leave at most 15 days?", []),
    ("How many days of leave are allowed?", []),
    ("Is there any furniture allowance?", []),
])
def test_quantifier_slots(text, expected):
    assert [(w, slot) for w, slot, _h, _s in C.quantifier_phrases(text)] == expected


@pytest.mark.parametrize("text,focus,target", [
    ("Is HR the only unit that assists with entry visas?", X.ACTOR, {"hr"}),
    ("Are only Muslim faculty eligible for Hajj leave?", X.POPULATION, {"muslim", "faculty"}),
    ("Is the probation period only one year?", X.VALUE, {"one", "year"}),
    ("Is the appointment renewable only once?", X.VALUE, {"once"}),
    ("Is the appointment just three years?", X.VALUE, {"three", "years"}),
    ("Can leave be granted only in special circumstances?", X.CONDITION, {"special", "circumstances"}),
    ("Is the probation period only extendable with Chancellor approval?", X.CONDITION, {"extendable", "chancellor", "approval"}),
    ("Can faculty resign only at the end of an academic semester?", X.TIME, {"end", "academic", "semester"}),
    ("Do the scholarships cover only tuition fees?", X.OBJECT, {"tuition", "fees"}),
    ("The suspension period may only be extended by a decision from the Disciplinary Council.", X.ACTOR, {"decision", "disciplinary", "council"}),
])
def test_exclusivity_focus(text, focus, target):
    found, ambiguous = C.read_exclusivity(text)
    assert [(e.focus, set(e.target)) for e in found] == [(focus, target)] and not ambiguous


def test_just_is_exclusive_only_before_a_number():
    assert C.read_exclusivity("Is it just a formality?") == ((), (C.Operator(K.AMBIGUOUS, "just", (6, 10), None),))
    assert C.read_exclusivity("Is the appointment just three years?")[0][0].focus is X.VALUE


@pytest.mark.parametrize("opener,role", [
    ("if", R.SUFFICIENT), ("in case", R.SUFFICIENT), ("when", R.TRIGGER), ("once", R.TRIGGER), ("upon", R.TRIGGER), ("after", R.TRIGGER),
    ("only if", R.NECESSARY), ("only when", R.NECESSARY), ("provided that", R.NECESSARY), ("as long as", R.NECESSARY), ("subject to", R.NECESSARY),
    ("unless", R.EXCEPTION), ("even if", R.CONCESSIVE), ("regardless of", R.CONCESSIVE), ("irrespective of", R.CONCESSIVE), ("whereby", R.UNKNOWN),
])
def test_condition_roles(opener, role):
    assert C.condition_role(opener) is role


@pytest.mark.parametrize("text,gate", [
    ("extendable to a maximum of two years with Chancellor approval", True), ("may be granted upon approval from appropriate University authorities", True),
    ("subject to the Dean's approval", True), ("following the recommendation of the Dean", True),
    ("sick leave with full pay", False), ("faculty with a mentor", False),
])
def test_approval_gates(text, gate):
    assert bool(C.approval_phrases(text)) is gate


@pytest.mark.parametrize("question,unhandled", [
    ("Are most faculty members eligible for the award?", [K.PROPORTIONAL]), ("Do many faculty members apply?", [K.PROPORTIONAL]),
    ("Are several faculty members exempt?", [K.PROPORTIONAL]), ("Is it just a formality?", [K.AMBIGUOUS]),
    ("Is the allowance more than 25,000 Dirhams?", [K.COMPARATIVE]), ("Can the probation period exceed two years?", [K.COMPARATIVE]),
    ("Does the Chancellor forward the decision before the Deans Council approves it?", [K.EVENT_ORDER]),
    ("Is the probation period at most two years?", []), ("Is unpaid leave not more than thirty days?", []), ("Is the award given each year?", []),
    ("Can leave be used after sick leave days are exhausted?", []),
])
def test_unhandled_operators_fail_closed(question, unhandled):
    assert [o.kind for o in C.unhandled(C.find_operators(question))] == unhandled


# ---------------------------------------------------------------------------
# 4. Compatibility tables
# ---------------------------------------------------------------------------
MODAL_EXPECTED = {
    M.OBLIGATION: "CCUUUUX", M.RECOMMENDATION: "CCCUUUX", M.ENCOURAGEMENT: "CCCUUUX",
    M.PERMISSION: "CCCCUCX", M.POSSIBILITY: "CCCCCCU", M.PROHIBITION: "XXXXUUC",
}
STATED_ORDER = [M.OBLIGATION, M.RECOMMENDATION, M.ENCOURAGEMENT, M.PERMISSION, M.POSSIBILITY, M.NONE, M.PROHIBITION]
LETTER = {"C": COMPATIBLE, "U": UNKNOWN, "X": CONTRADICTED}


@pytest.mark.parametrize("asked,stated", [(a, s) for a in MODAL_EXPECTED for s in STATED_ORDER])
def test_modal_matrix(asked, stated):
    assert C.modal_compatibility(asked, stated) is LETTER[MODAL_EXPECTED[asked][STATED_ORDER.index(stated)]]


def test_modal_edges():
    assert C.modal_compatibility(M.NONE, M.PERMISSION) is NA
    assert C.modal_compatibility(M.PERMISSION, None) is COMPATIBLE and C.modal_compatibility(M.OBLIGATION, None) is UNKNOWN


B = C.Breadth


@pytest.mark.parametrize("claimed,stated,breadth,result", [
    (Q.UNIVERSAL, Q.UNIVERSAL, B.SAME_OR_BROADER, COMPATIBLE), (Q.UNIVERSAL, Q.GENERIC, B.SAME_OR_BROADER, UNKNOWN),
    (Q.UNIVERSAL, None, B.SAME_OR_BROADER, UNKNOWN), (Q.UNIVERSAL, Q.UNIVERSAL, B.NARROWER, UNKNOWN),
    (Q.UNIVERSAL, Q.UNIVERSAL, B.UNRELATED, UNKNOWN),
    (Q.GENERIC, Q.GENERIC, B.SAME_OR_BROADER, COMPATIBLE), (Q.GENERIC, None, B.SAME_OR_BROADER, COMPATIBLE),
    (Q.GENERIC, Q.UNIVERSAL, B.SAME_OR_BROADER, COMPATIBLE), (Q.GENERIC, Q.GENERIC, B.NARROWER, UNKNOWN), (Q.GENERIC, Q.GENERIC, B.UNRELATED, UNKNOWN),
    (Q.EXISTENTIAL, Q.GENERIC, B.NARROWER, COMPATIBLE), (Q.EXISTENTIAL, Q.GENERIC, B.SAME_OR_BROADER, COMPATIBLE),
    (Q.EXISTENTIAL, None, B.UNRELATED, UNKNOWN), (Q.DEFINITE, Q.GENERIC, B.NARROWER, NA), (Q.PROPORTIONAL, Q.GENERIC, B.SAME_OR_BROADER, NA),
])
def test_scope_matrix(claimed, stated, breadth, result):
    assert C.scope_compatibility(claimed, stated, breadth) is result


@pytest.mark.parametrize("focus,exclusive,alternative,exact,result", [
    (X.ACTOR, True, False, False, COMPATIBLE), (X.ACTOR, False, False, False, UNKNOWN), (X.ACTOR, True, True, False, CONTRADICTED),
    (X.VALUE, False, False, True, COMPATIBLE), (X.VALUE, False, True, True, CONTRADICTED), (X.VALUE, False, False, False, UNKNOWN),
    (X.POPULATION, False, False, True, UNKNOWN),
])
def test_exclusivity_table(focus, exclusive, alternative, exact, result):
    assert C.exclusivity_compatibility(focus, exclusive, alternative, exact) is result


@pytest.mark.parametrize("asked,stated,gate,result", [
    (R.NECESSARY, R.NECESSARY, False, COMPATIBLE), (R.NECESSARY, None, True, COMPATIBLE), (R.NECESSARY, R.SUFFICIENT, False, UNKNOWN),
    (R.NECESSARY, None, False, UNKNOWN), (R.SUFFICIENT, R.SUFFICIENT, False, COMPATIBLE), (R.SUFFICIENT, R.NECESSARY, False, UNKNOWN),
    (R.SUFFICIENT, R.NECESSARY, True, COMPATIBLE), (R.TRIGGER, R.TRIGGER, False, COMPATIBLE), (R.EXCEPTION, R.SUFFICIENT, False, NA),
    (R.CONCESSIVE, None, False, NA),
])
def test_condition_role_table(asked, stated, gate, result):
    assert C.condition_role_compatibility(asked, stated, gate) is result


def test_combination_is_conservative():
    D = C.DimensionResult
    assert C.combine([D("a", COMPATIBLE), D("b", CONTRADICTED), D("c", UNKNOWN)]) is CONTRADICTED
    assert C.combine([D("a", COMPATIBLE), D("b", UNKNOWN)]) is UNKNOWN
    assert C.combine([D("a", COMPATIBLE), D("b", NA)]) is COMPATIBLE
    assert C.combine([D("a", NA)]) is NA and C.combine([]) is NA


# ---------------------------------------------------------------------------
# 5. The HR claim reader
# ---------------------------------------------------------------------------
def test_the_claim_of_a_modal_universal_or_exclusive_question():
    c = F.read_claim("Must every Dean hold the rank of full professor?")
    assert (c.modality, c.subject.quantifier, c.subject.head, c.action) == (M.OBLIGATION, Q.UNIVERSAL, frozenset({"dean"}), "hold")
    c = F.read_claim("Are all faculty members eligible for the bonus award?")
    assert (c.subject.quantifier, c.subject.head, c.modality) == (Q.UNIVERSAL, frozenset({"faculty", "member"}), M.NONE)
    c = F.read_claim("Is HR the only unit that assists with entry visas?")
    assert [(e.focus, set(e.target)) for e in c.exclusivity] == [(X.ACTOR, {"hr"})]
    c = F.read_claim("Are only Muslim faculty eligible for Hajj leave?")
    assert c.subject.restrictions == frozenset({"muslim"}) and c.exclusivity[0].focus is X.POPULATION


def test_the_claim_reader_keeps_entities_labels_and_conditions_apart():
    assert F.read_claim("Must the Faculty Grievance Committee contact witnesses?").subject.quantifier is Q.DEFINITE    # an entity, no population
    assert F.read_claim("Does the Employee Handbook cover HR policies?").subject in (None,) or F.read_claim(
        "Does the Employee Handbook cover HR policies?").subject.quantifier is Q.DEFINITE
    c = F.read_claim("If I have used all my sick leave days, may annual vacation leave be used?")
    assert not c.subject or c.subject.quantifier is not Q.UNIVERSAL                            # "all" belongs to the condition
    c = F.read_claim("Is the Dean's appointment renewable only once?")
    assert [e.focus for e in c.exclusivity] == [X.VALUE] and not c.conditions                  # "once" is a value, no condition
    c = F.read_claim("Can probation be extended only if the Chancellor approves?")
    assert [x.role for x in c.conditions] == [R.NECESSARY]
    c = F.read_claim("Is the Faculty Grievance Committee required to hold a hearing?")
    assert (c.modality, c.action) == (M.OBLIGATION, "hold")
    c = F.read_claim("Must faculty with administrative assignments undergo an annual evaluation?")
    assert c.action == "undergo"
    assert F.read_claim("What is the probation period?") is None


@pytest.mark.parametrize("question,action", [
    ("Must the Faculty Grievance Committee contact witnesses?", "contact"), ("Must the Faculty Grievance Committee hold a hearing?", "hold"),
    ("Must the Chancellor grant special permission for a clinical faculty appointment in other disciplines?", "grant"),
    ("Can the Chancellor waive the probation period?", "waive"), ("May faculty receive a housing allowance?", "receive"),
    ("Must faculty members on probation give six months notice?", "give"),
    ("Must faculty with administrative assignments undergo an annual evaluation?", "undergo"),
    ("Can failure to comply result in a 50 percent deduction from the end of service compensation?", "result"),
    ("Does termination for cause require the Chancellor's approval?", "require"),
    ("Does the University Central Committee present promotion applications to the department chair?", "present"),
    ("Does neither the Dean nor the Chancellor approve termination during probation?", "approve"),
    ("Can termination during probation happen with no Dean approval?", "happen"),
    ("Can termination for financial emergency happen without a Board of Trustees declaration?", "happen"),
    ("Do faculty who resign during probation receive ticket allowances?", "receive"),
    ("Does the promotion evaluation of service include serving on department, college and university committees?", "include"),
    ("Can a faculty member appeal a warning to the Dean within seven days?", "appeal"),
    ("Does the University strictly prohibit any form of sexual harassment?", "prohibit"),
    ("Must the referees have supervised the applicant's research at the master or doctoral levels?", "supervised"),
    ("Must emeritus candidates have served the University for at least ten years?", "served"),
    ("Can the furniture allowance be used over four years?", "used"), ("Is the Faculty Grievance Committee required to hold a hearing?", "hold"),
    ("Do all appointment categories have the same contract duration?", None), ("Should termination be the first resort?", None),
    ("Must faculty members intending to resign during probation provide six months' notice?", "provide"),
    ("Is the suspension period required to be extended by the Disciplinary Council?", "extended"),
    ("Are new faculty members who are terminated during probation entitled to return tickets?", "entitled"),
    ("Are Emirati nationals who are recent graduates considered on a competitive basis among applicants?", "considered"),
])
def test_the_asked_action(question, action):
    assert F.read_claim(question).action == action


def test_a_question_led_by_its_condition_keeps_its_claim():
    q = "If I resign during probation, must I give six months notice?"
    c = F.read_claim(q)
    assert (c.modality, c.action, [x.role for x in c.conditions]) == (M.OBLIGATION, "give", [R.SUFFICIENT])
    assert not check(F.modality_detail, q, "Faculty members who resign during probation may give six months notice.")
    assert check(F.modality_detail, q, "Faculty members who resign during probation must give six months notice.")
    assert F.read_claim("If I resign, what happens?") is None


@pytest.mark.parametrize("question", ["What is the probation period?", "How many days of sick leave do faculty get?",
                                      "Who approves the extension of probation?", "Tell me about Hajj leave."])
def test_an_open_question_has_no_structured_claim(question):
    assert F.read_claim(question) is None and F.structured_details(question, F.read_claim(question)) == []


def test_a_negative_claim_takes_no_modality_or_scope():
    for q in ("Are all faculty members not eligible for the bonus award?", "Must faculty members not apply for the bonus award?"):
        c = F.read_claim(q)
        assert c.polarity is C.Polarity.NEGATIVE
        assert F.modality_detail(q, c) is None and F.scope_detail(q, c) is None


@pytest.mark.parametrize("question,families", [
    ("Must all faculty members apply for the bonus award?", ["modality", "scope"]),
    ("Must faculty apply only once for the bonus award?", ["modality", "scope", "exclusivity"]),         # a generic population: scope
    ("Are all faculty eligible only for Hajj leave?", ["scope", "exclusivity"]),
    ("Can probation be extended only if the Chancellor approves?", ["modality", "condition_role"]),
    ("Must all faculty members attend only one induction if the Dean approves?", ["modality", "scope", "exclusivity", "condition_role"]),
    ("Are most faculty members eligible for the bonus award?", ["operator"]),
    ("May most faculty members apply for the bonus award?", ["modality", "operator"]),
])
def test_each_dimension_of_a_combined_question_is_checked(question, families):
    assert [d.family for d in F.structured_details(question, F.read_claim(question))] == families


def test_the_subject_runs_up_to_the_verb_and_keeps_its_relative_clause():
    c = F.read_claim("Must faculty members intending to resign during probation provide six months' notice?")
    assert {"resign", "probat"} <= c.subject.restrictions
    c = F.read_claim("Are new faculty members who are terminated during probation entitled to return tickets?")
    assert {"new", "terminat", "probat"} <= c.subject.restrictions


def test_a_question_led_by_an_approval_phrase_keeps_its_claim():
    c = F.read_claim("Subject to Chancellor approval, can the probation period be extended?")
    assert (c.modality, c.action, [x.role for x in c.conditions]) == (M.PERMISSION, "extended", [R.NECESSARY])
    assert c.subject.text == "the probation period" and c.subject.quantifier is Q.DEFINITE


def test_a_subject_cut_inside_a_name_is_an_entity_with_no_scope():
    q = "Do applicants who meet the requirements receive apologies from the Faculty Emiratization Committee?"
    c = F.read_claim(q)                              # the structural parse stops inside "the Faculty | Emiratization Committee"
    assert c.subject.quantifier is Q.DEFINITE and F.scope_detail(q, c) is None


def test_a_population_reads_its_restrictions_not_its_light_words():
    p = F._population("muslim faculty attending their first hajj".split())
    assert (p.head, "muslim" in p.restrictions, "attend" in p.restrictions) == (frozenset({"faculty"}), True, False)
    assert F._population("individuals appointed as deans".split()).restrictions == frozenset()
    assert F._population("all faculty including those with administrative assignments".split()).restrictions == frozenset()
    assert "exclud" in F._population("all faculty excluding visiting faculty".split()).restrictions
    assert F._population("the faculty grievance committee".split()).quantifier is Q.DEFINITE


# ---------------------------------------------------------------------------
# 6. Modality: metamorphic matrix, binding, attributive words
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,unit,ok", [
    ("Must the committee contact witnesses?", "The committee must contact witnesses.", True),
    ("Must the committee contact witnesses?", "The committee shall contact witnesses.", True),
    ("Must the committee contact witnesses?", "The committee should contact witnesses.", True),
    ("Must the committee contact witnesses?", "The committee is encouraged to contact witnesses.", False),
    ("Must the committee contact witnesses?", "The committee may contact witnesses.", False),
    ("Must the committee contact witnesses?", "The committee contacts witnesses.", False),
    ("Must the committee contact witnesses?", "The committee must not contact witnesses.", False),
    ("May the committee contact witnesses?", "The committee must contact witnesses.", True),
    ("May the committee contact witnesses?", "The committee may contact witnesses.", True),
    ("May the committee contact witnesses?", "The committee contacts witnesses.", True),
    ("May the committee contact witnesses?", "The committee may not contact witnesses.", False),
    ("Should the committee contact witnesses?", "The committee is encouraged to contact witnesses.", True),
    ("Should the committee contact witnesses?", "The committee may contact witnesses.", False),
    ("Is the committee prohibited from contacting witnesses?", "The committee must not contact witnesses.", True),
    ("Is the committee prohibited from contacting witnesses?", "The committee may contact witnesses.", False),
])
def test_modality_is_compatible_by_force(question, unit, ok):
    assert check(F.modality_detail, question, unit) is ok


def test_modality_binds_to_the_asked_action_not_any_modal_word():
    q = "Must faculty allocate time for student consultations?"
    assert not check(F.modality_detail, q, "While office hours may vary, faculty are encouraged to allocate time for student consultations.")
    assert not check(F.modality_detail, q, "Faculty must attend the meetings, and faculty are encouraged to allocate time for student consultations.")
    assert check(F.modality_detail, q, "Faculty must allocate time for student consultations.")
    assert not check(F.modality_detail, "Must faculty vary their office hours?", "The number of office hours may vary.")   # epistemic


def test_an_attributive_requirement_asks_no_modality():
    assert F.modality_detail("Does HR compile the required documents for the Employee File?",
                             F.read_claim("Does HR compile the required documents for the Employee File?")) is None
    assert F.modality_detail("Is approval required for unpaid special leave?", F.read_claim("Is approval required for unpaid special leave?")) is None


# ---------------------------------------------------------------------------
# 7. Scope: universal, generic (strict), existential, categories, light words, time quantifiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,unit,ok", [
    ("Are all faculty members eligible for the award?", "All faculty members are eligible for the award.", True),
    ("Are all faculty members eligible for the award?", "Faculty members are eligible for the award.", False),
    ("Are all faculty members eligible for the award?", "Full-time faculty members are eligible for the award.", False),
    ("Are faculty members eligible for the award?", "Faculty members are eligible for the award.", True),
    ("Are faculty members eligible for the award?", "All faculty members are eligible for the award.", True),
    ("Are faculty members eligible for the award?", "Full-time faculty members are eligible for the award.", False),
    ("Are faculty members entitled to Hajj leave?", "Muslim faculty attending their first Hajj may receive up to twenty calendar days of paid leave.", False),
    ("Can Muslim faculty receive paid leave for their first Hajj?",
     "Muslim faculty attending their first Hajj may receive up to twenty calendar days of paid leave.", True),
    ("Are some faculty members eligible for the award?", "Full-time faculty members are eligible for the award.", True),
    ("Must Deans hold the rank of full professor?", "Individuals appointed as Deans must hold the rank of full professor.", True),
    ("Must every Dean hold the rank of full professor?", "Individuals appointed as Deans must hold the rank of full professor.", False),
])
def test_scope_is_compatible_by_breadth(question, unit, ok):
    assert check(F.scope_detail, question, unit) is ok


def test_an_object_or_time_quantifier_is_no_population_quantifier():
    assert check(F.scope_detail, "Must faculty members submit all required documents?", "Faculty members must submit all required documents.")
    assert check(F.scope_detail, "Are faculty members evaluated each academic year?", "Faculty members are evaluated each academic year.")


HAJJ_UNIT = ("The unpaid leave will be deducted from the next month's salary. 4.3.5 Hajj Leave Muslim faculty attending their first Hajj may "
             "receive up to twenty calendar days of paid leave.")


def test_scope_reads_the_sentences_that_name_a_population():
    assert not check(F.scope_detail, "Are faculty entitled to Hajj leave?", HAJJ_UNIT)      # not the heading, not the deduction sentence
    assert check(F.scope_detail, "Can Muslim faculty receive paid leave for their first Hajj?", HAJJ_UNIT)
    pops = F._stated_populations(HAJJ_UNIT, frozenset({"entitl"}), frozenset({"hajj", "leav"}))
    assert [sorted(p.head) for p in pops] == [["faculty"]] and "muslim" in pops[0].restrictions


def test_only_population_claims_have_a_scope():
    for q in ("Does HR draft employment offers?", "Is the probation period one year?", "Does the College Council form the committee each September?",
              "Does HR manage all communication with candidates?", "Are most faculty members eligible for the award?"):
        assert F.scope_detail(q, F.read_claim(q)) is None, q


# ---------------------------------------------------------------------------
# 8. The window veto: bound contradictions and exceptions anywhere in the returned quote
# ---------------------------------------------------------------------------
QUOTE_EXCLUSION = "All faculty members are eligible for the award. Visiting faculty members are not eligible for the award."
QUOTE_UNRELATED = "All faculty members are eligible for the award. The award does not affect the monthly salary."
QUOTE_EXCEPTION = "All faculty members must attend the training. In exceptional cases, the Dean may exempt a faculty member from the training."
QUOTE_CONDITION = "If a faculty member is not eligible for the award, the Dean informs the faculty member. All faculty members are eligible for the award."


def test_a_universal_claim_is_vetoed_by_a_bound_exclusion_or_exception_in_the_window():
    assert window(F.scope_detail, "Are all faculty members eligible for the award?", QUOTE_EXCLUSION) is CONTRADICTED
    assert window(F.scope_detail, "Must all faculty members attend the training?", QUOTE_EXCEPTION) is CONTRADICTED


def test_a_generic_claim_keeps_a_stated_exception_and_an_unrelated_negation_vetoes_nothing():
    assert window(F.scope_detail, "Are faculty members eligible for the award?", QUOTE_EXCLUSION) is COMPATIBLE
    assert window(F.scope_detail, "Must faculty members attend the training?", QUOTE_EXCEPTION) is COMPATIBLE
    assert window(F.scope_detail, "Are all faculty members eligible for the award?", QUOTE_UNRELATED) is COMPATIBLE
    assert window(F.scope_detail, "Are all faculty members eligible for the award?", QUOTE_CONDITION) is COMPATIBLE   # inside an antecedent


def test_a_generic_claim_is_vetoed_by_the_negation_of_the_same_population():
    quote = "Faculty members are eligible for the award. Faculty members are not eligible for the award in their first year."
    assert window(F.scope_detail, "Are faculty members eligible for the award?", quote) is CONTRADICTED


def test_the_window_veto_removes_a_vetoed_detail_from_coverage():
    q = "Are all faculty members eligible for the award?"
    d = F.scope_detail(q, F.read_claim(q))
    key = F._detail_key(d)
    assert F._window_vetoed([d], QUOTE_EXCLUSION, frozenset({key})) == frozenset()
    assert F._window_vetoed([d], QUOTE_UNRELATED, frozenset({key})) == frozenset({key})


def test_a_statement_population_is_the_subject_before_the_predicate():
    unit = "In exceptional cases, full-time faculty members appointed before 2020 may apply for the award."
    p = F._statement_population(unit, unit.index("apply"))
    assert p.head == frozenset({"faculty", "member"}) and "full" in p.restrictions
    assert F._statement_population("The award is granted annually.", 10).head == frozenset()


def test_breadth_compares_the_stated_population_with_the_asked_one():
    claim = F.read_claim("Are faculty members eligible for the award?")
    roots = F._roots("Are faculty members eligible for the award?")
    assert F._breadth(claim, roots, F._population("faculty members".split())) is C.Breadth.SAME_OR_BROADER
    assert F._breadth(claim, roots, F._population("visiting faculty members".split())) is C.Breadth.NARROWER
    assert F._breadth(claim, roots, F._population("the students".split())) is C.Breadth.SAME_OR_BROADER   # no person head
    assert F._breadth(claim, roots, None) is C.Breadth.SAME_OR_BROADER


RESIGN_QUOTE = ("New faculty members who are terminated during probation are entitled to return tickets, while new faculty members who resign "
                "during probation are not entitled to benefits such as annual leave and ticket allowances.")


@pytest.mark.parametrize("question,vetoed", [
    ("If I resign during probation, am I entitled to annual leave?", True),
    ("If a faculty member resigns during probation, do they get annual leave?", True),
    ("Are faculty entitled to annual leave if they resign during probation?", True),
    ("If I am terminated during probation, am I entitled to return tickets?", False),
])
def test_a_negation_for_the_population_a_condition_names_vetoes_the_claim(question, vetoed):
    d = F.condition_role_detail(question, F.read_claim(question))
    assert (d.check.window(RESIGN_QUOTE) is CONTRADICTED) is vetoed


def test_a_relative_clause_belongs_to_the_statement_subject():
    i = RESIGN_QUOTE.index("are not")
    p = F._statement_population(RESIGN_QUOTE, i)
    assert p.head == frozenset({"faculty", "member"}) and {"new", "resign", "probat"} <= p.restrictions
    q = "Are new faculty members who are terminated during probation entitled to return tickets?"
    assert window(F.scope_detail, q, RESIGN_QUOTE) is COMPATIBLE                              # a narrower negation: no veto


def test_a_contracted_negator_is_no_restriction_of_a_statement_subject():
    unit = "Faculty who doesn't meet expectations during a contract term will be provided with an opportunity to improve."
    p = F._statement_population(unit, unit.index("will"))
    assert p.head == frozenset({"faculty"}) and "doesn" not in p.restrictions
    assert check(F.scope_detail, "Will faculty who do not meet expectations during a contract term be provided with an opportunity to improve?", unit)


def test_statements_record_negation_exceptions_and_roles():
    claim = F.read_claim("Are all faculty members eligible for the award?")
    s = F.read_statements(QUOTE_EXCLUSION, claim)
    assert [x.polarity for x in s] == [C.Polarity.AFFIRMATIVE, C.Polarity.NEGATIVE]
    assert s[1].subject.restrictions == frozenset({"visit"}) and s[0].subject.quantifier is Q.UNIVERSAL
    s = F.read_statements(QUOTE_EXCEPTION, F.read_claim("Must all faculty members attend the training?"))
    assert s[1].exceptions and s[0].modality is M.OBLIGATION
    s = F.read_statements("Probation is extendable only if the Chancellor approves.", F.read_claim("Can probation be extended?"))
    assert [a.role for a in s[0].antecedents] == [R.NECESSARY]


# ---------------------------------------------------------------------------
# 9. Exclusivity: all six foci, the value window
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,unit,ok", [
    ("Is the probation period only one year?", "The probation period is one year.", True),
    ("Is the probation period only one year?", "The probation period is only one year.", True),
    ("Is the probation period only one year?", "The probation period is one year, extendable to two years.", False),
    ("Is the appointment renewable only once?", "The appointment is for three years, renewable only once.", True),
    ("Is HR the only unit that assists with visas?", "Only HR assists with visas.", True),
    ("Is HR the only unit that assists with visas?", "HR assists with visas.", False),
    ("Are only Muslim faculty eligible for Hajj leave?", "Hajj leave is limited to Muslim faculty.", True),
    ("Are only Muslim faculty eligible for Hajj leave?", "Muslim faculty may receive Hajj leave.", False),
    ("Can faculty resign only at the end of a semester?", "Resignations must coincide with the end of a semester.", True),
    ("Can faculty resign only at the end of a semester?", "Faculty may resign at the end of a semester.", False),
    ("Can sick leave without pay be granted only in special circumstances?", "Sick leave without pay may be granted only in special circumstances.", True),
    ("Can sick leave without pay be granted only in special circumstances?", "In special circumstances, sick leave without pay may be granted.", False),
    ("Do the scholarships cover only tuition fees?", "The scholarships cover tuition fees only.", True),
    ("Do the scholarships cover only tuition fees?", "The scholarships cover tuition fees.", False),
])
def test_exclusivity_is_stated_on_its_focus(question, unit, ok):
    assert check(F.exclusivity_detail, question, unit) is ok


@pytest.mark.parametrize("text,values", [
    ("one year", {("1", "year")}), ("six months", {("6", "month")}), ("renewable only once", {("once", "")}),
    ("for 3 years, renewable twice", {("3", "year"), ("twice", "")}), ("the award", set()),
])
def test_value_tokens(text, values):
    assert F._value_tokens(text) == frozenset(values)


def test_a_value_exclusivity_is_vetoed_by_an_extension_in_the_window():
    q = "Is the probation period only one year?"
    assert window(F.exclusivity_detail, q, "The probation period is one year. It is extendable to two years with Chancellor approval.") is CONTRADICTED
    assert window(F.exclusivity_detail, q, "The probation period is one year. Leave is granted upon approval.") is COMPATIBLE
    assert window(F.exclusivity_detail, q, "The probation period is one year. Leave may be extended to two years.") is COMPATIBLE   # unbound
    assert window(F.exclusivity_detail, q, "The probation period is one year. The probation period may be extended.") is CONTRADICTED
    assert check(F.exclusivity_detail, q, "The probation period is one year and cannot be extended.")              # a negated extension
    assert check(F.exclusivity_detail, "Is the appointment renewable only once?", "The appointment is renewable once.")
    assert not check(F.exclusivity_detail, "Is the appointment renewable only once?", "The appointment is renewable once, and in some cases twice.")
    assert F.exclusivity_detail("Is the probation period one year?", F.read_claim("Is the probation period one year?")) is None


# ---------------------------------------------------------------------------
# 10. Condition roles and fail-closed operators
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,unit,ok", [
    ("Can probation be extended only if the Chancellor approves?", "Probation is extendable with Chancellor approval.", True),
    ("Can probation be extended only if the Chancellor approves?", "Probation may be extended only if the Chancellor approves.", True),
    ("Can probation be extended only if the Chancellor approves?", "Probation may be extended if the Chancellor approves.", False),
    ("Can probation be extended if the Chancellor approves?", "Probation may be extended if the Chancellor approves.", True),
    ("Can probation be extended if the Chancellor approves?", "Probation may be extended with Chancellor approval.", True),
    ("Can probation be extended if the Chancellor approves?", "Probation may be extended only if the Chancellor approves.", False),
])
def test_condition_roles_of_the_ask_and_the_evidence(question, unit, ok):
    assert check(F.condition_role_detail, question, unit) is ok


def test_an_exception_or_concessive_condition_asks_no_role():
    for q in ("Unless the Chancellor approves, can probation be extended?", "Can probation be extended even if the Dean objects?"):
        assert F.condition_role_detail(q, F.read_claim(q)) is None, q


def test_a_proportional_subject_has_no_scope_and_fails_closed():
    q = "Are most faculty members eligible for the bonus award?"
    c = F.read_claim(q)
    assert c.subject.quantifier is Q.PROPORTIONAL and F.scope_detail(q, c) is None
    d = F.operator_detail(q, c)
    assert d.label == "the asked proportional (most)" and not d.check("All faculty members are eligible for the bonus award.")


def test_a_condition_role_is_checked_whatever_the_polarity_of_its_content():
    q = "Can unpaid special leave be granted only if the employee has no unused vacation leave?"
    assert not check(F.condition_role_detail, q, "Unpaid special leave may be granted upon approval if the employee has no unused vacation leave.")
    assert check(F.condition_role_detail, q, "Unpaid special leave may be granted only if the employee has no unused vacation leave.")


VISITING = ("Visiting Appointment: Involves a definite term contract of up to two years, including a probation year. In exceptional cases, and "
            "with Chancellor approval, the visiting contract may be extended for an additional year.")
PROBATION = "The faculty probation period is one year, extendable to a maximum of two years under exceptional circumstances with Chancellor approval."


def test_a_condition_of_an_entity_subject_is_read_in_a_sentence_about_it():
    q = "Subject to Chancellor approval, can the probation period be extended?"
    assert not check(F.condition_role_detail, q, VISITING) and check(F.condition_role_detail, q, PROBATION)
    q = "Is the probation period only extendable with Chancellor approval?"
    assert not check(F.exclusivity_detail, q, VISITING) and check(F.exclusivity_detail, q, PROBATION)
    q = "Can the probation period be extended only with Chancellor approval?"
    assert check(F.condition_role_detail, q, "The probation period is one year. It may be extended with Chancellor approval.")   # a pronoun


def test_unknown_operators_never_support():
    for q in ("Are most faculty members eligible for the bonus award?", "Is the probation period more than one year?", "Is it just a formality?",
              "Does the Chancellor forward the decision before the Deans Council approves it?"):
        d = F.operator_detail(q, F.read_claim(q))
        assert d is not None and not d.check("Faculty members are eligible for the bonus award."), q
    for q in ("Is the probation period at most two years?", "Is the appointment just three years?", "Can leave be used after sick leave is exhausted?"):
        assert F.operator_detail(q, F.read_claim(q)) is None, q


def test_an_existential_negative_is_settled_by_a_narrower_negation():
    bonus = "Faculty members may apply for the bonus award in consecutive years. Parttime and visiting faculty members are not eligible for the bonus award."
    assert F.negative_detail("Are some faculty members not eligible for the bonus award?").check(bonus)
    assert F.negative_detail("Are certain faculty members not eligible for the bonus award?").check(bonus)
    assert not F.negative_detail("Are faculty members not eligible for the bonus award?").check(bonus)       # M1: a narrower negation settles no generic


# ---------------------------------------------------------------------------
# 11. The known failures and their controls on the fixture pages, both rankings
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def specialists():
    corpus = corpus_from(registry_and_map(), PAGES + C10_PAGES)
    return {"raw": make_specialist(corpus), "high": make_specialist(corpus, ScriptedReranker([], default=9.0))}


KNOWN = [
    ("Must the Faculty Grievance Committee contact witnesses?", "modality"), ("Must emeritus faculty teach courses?", "modality"),
    ("Must faculty members apply for the bonus award?", "modality"),
    ("Must the Chancellor grant special permission for a clinical faculty appointment in other disciplines?", "modality"),
    ("Are all faculty members eligible for the bonus award?", "population"), ("Can every faculty member apply for the bonus award?", "population"),
    ("Are all faculty members entitled to Hajj leave?", "population"), ("Must every Dean hold the rank of full professor?", "population"),
    ("Is HR the only unit that assists with entry visas?", "exclusivity"), ("Is the probation period only one year?", "exclusivity"),
    ("Are only Muslim faculty eligible for Hajj leave?", "exclusivity"),
]
CONTROLS = [
    "May faculty members apply for the bonus award?", "Must the Faculty Grievance Committee hold a hearing?",
    "Can the Chancellor grant special permission for a clinical faculty appointment in other disciplines?", "Must Deans hold the rank of full professor?",
    "Is the Dean's appointment renewable only once?", "Can the suspension period be extended by the Disciplinary Council?",
    "Is the probation period one year?", "Does HR assist new faculty members in obtaining entry visas?",
]


@pytest.mark.parametrize("ranking", ["raw", "high"])
@pytest.mark.parametrize("question,dimension", KNOWN)
def test_the_known_failures_are_partial_with_their_dimension(specialists, ranking, question, dimension):
    r = run(specialists[ranking], question)
    assert r.status is not SUPPORTED, question
    assert any(dimension in m for m in r.missing), r.missing


@pytest.mark.parametrize("ranking", ["raw", "high"])
@pytest.mark.parametrize("question", CONTROLS)
def test_the_paired_controls_stay_supported(specialists, ranking, question):
    assert run(specialists[ranking], question).status is SUPPORTED, question


@pytest.mark.parametrize("ranking", ["raw", "high"])
def test_an_existential_negative_is_supported_by_a_narrower_negation_on_the_fixture(specialists, ranking):
    assert run(specialists[ranking], "Are some faculty members not eligible for the bonus award?").status is SUPPORTED
    assert run(specialists[ranking], "Are faculty members not eligible for the bonus award?").status is not SUPPORTED


CAMPAIGN = [
    ("Are faculty entitled to Hajj leave?", False), ("Can Muslim faculty receive up to twenty calendar days of paid leave for their first Hajj?", True),
    ("If I resign during probation, am I entitled to annual leave?", False), ("If I am terminated during probation, am I entitled to return tickets?", True),
    ("Can unpaid special leave be granted only if the employee has no unused vacation leave?", False),
    ("Must the Chancellor approve the termination of contracts for cause?", True),
]


@pytest.mark.parametrize("ranking", ["raw", "high"])
@pytest.mark.parametrize("question,supported", CAMPAIGN)
def test_campaign_answers_on_the_fixture(specialists, ranking, question, supported):
    assert (run(specialists[ranking], question).status is SUPPORTED) is supported, question
