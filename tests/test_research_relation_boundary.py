"""Research & Innovation Specialist: the semantic relation boundary (Milestone 4, Step 4.5A-D).

Architectural decision: domain specialists stay deterministic evidence specialists. A recognised specific ask is
``supported`` only when deterministic rules PROVE that the answering statement states the asked relation (STRONG):
every role the question names is named there, every asked action is predicated there of the asked object, a stated
purpose or receiver is the asked one, a required thing is what the statement requires, and a yes/no proposition is
contained in the unit. Anything the rules cannot prove is AMBIGUOUS and at most ``partial``: a false negative is
preferable to a false-positive ``supported``. Semantic relation verification belongs to the Milestone 5 Verifier, which
this step does not implement; general overview questions keep their own rule.

Categories: A strong relations, B ambiguous relations, C wrong action, D wrong role, E wrong purpose or beneficiary,
F general controls, G position binding (a party named in the statement in another grammatical position, a negated or
contrasting occurrence, an argument stated with another preposition, another period). Every question runs against pages whose lines are the real handbook wording (the fixtures of
tests/test_research_specialist.py and tests/test_research_corrections.py), with the evidence ranked first by a scripted
reranker and, for C to E, with every chunk of the group ranked first as well. No model, no LLM, no network.
"""
from __future__ import annotations

import re

import pytest

import test_research_specialist as RT
from test_research_corrections import corpora as corr_corpora  # noqa: F401  (module fixture)
from test_research_corrections import corpora2  # noqa: F401  (module fixture)
from test_research_specialist import corpora, registry_and_map  # noqa: F401  (module fixtures)
from handbook_bot.agents.specialists import research as R
from handbook_bot.agents.specialists.contracts import FindingStatus

S, P = FindingStatus.SUPPORTED, FindingStatus.PARTIAL


@pytest.fixture(scope="module")
def corpus_of(corpora, corr_corpora, corpora2):  # noqa: F811
    def get(source, group):
        if source == "rt":
            return corpora(group)
        if source == "rc":
            return corr_corpora(group)
        return corpora2()

    return get


def ranked_run(corpus_of, source, group, question, needle=None):
    """The needle's chunk ranked first (or, without a needle, every chunk of the group)."""
    corpus = corpus_of(source, group)
    rules = [(needle, 9.0)] if needle else []
    sp = RT.make_specialist(corpus, RT.ScriptedReranker(rules, default=-5.0 if needle else 9.0))
    result = RT.run(sp, question)
    RT.assert_traceable(result, corpus)
    return result


def clause(result):
    return result.metadata["clauses"][0]


def never_supported(corpus_of, source, group, question, needle):
    for ranking in (needle, None):
        result = ranked_run(corpus_of, source, group, question, ranking)
        assert result.status is not S, (question, ranking, clause(result), RT.quotes(result))


# ---------------------------------------------------------------------------
# A. STRONG relations: the stating statement proves the asked relation -> SUPPORTED
# ---------------------------------------------------------------------------
STRONG = [
    ("rt", "conference", "Who approves conference travel?", "Vice Chancellor for Academic Affairs approves travel requests"),
    ("rt", "grants", "Who determines the funding ceiling for research grants?", "VCRGS determines the funding ceiling"),
    ("rt", "grants", "Who reviews research grant applications?", "Research committees review research grant applications"),
    ("rt", "consultancy", "Which committee reviews research on human subjects?", "Institutional Review Board (IRB)"),
    ("r2", None, "Who appoints the moderator in an IP dispute?", "shall appoint a moderator"),
    ("rt", "consultancy", "Who approves research involving animals?", "approved by the Institutional Animal Care and Use Committee (IACUC)"),
    ("rt", "grants", "Who approves collaborative research agreements?", "Collaborative research agreements with external organizations"),
    ("rt", "ip", "Who approves transferring tangible research property for commercial purposes?", "For commercial purposes, transferring"),
    ("rt", "grants", "How much petty cash advance can a principal investigator receive?", "entitled to a petty cash advance"),
    ("rt", "supervision", "How many theses can a faculty member supervise at once?", "supervise a maximum of four theses"),
    ("rt", "consultancy", "Is prior approval required for non-compensated consultancy?", "prior approval is required"),
    ("rt", "consultancy", "How many days per week can faculty spend on consultancy?", "may not exceed one day per week"),
    ("rt", "ip", "What percentage of the University's share goes to the VCRGS?", "40% to the VCRGS"),
    ("rt", "ip", "What percentage of net royalty revenue goes to the inventor?", "50% to the inventor(s)"),
    ("rt", "ip", "Who owns intellectual property created by faculty?", "shall own all Intellectual Property Rights"),
    ("rt", "consultancy", "Do researchers need IRB approval before recruiting human participants?", "Researchers must obtain IRB approval"),
    ("rt", "conference", "How much is the per diem for conference travel?", "Per diem rates between AED 1000-1500"),
    ("rt", "conference", "When must conference requests be submitted?", "Requests must be submitted at least one month"),
    ("rc", "grants", "Who approves the closure of a research project?", "The Dean may approve the project's closure"),
    ("rc", "consultancy", "Who approves the use of the University's name in consultancy?", "16.1.8 Use of University Name"),
    ("rc", "grants", "Who approves a petty cash advance?", "entitled to a petty cash advance"),
    ("rc", "ip", "Who owns copyrighted materials?", "Copyrighted materials remain the property of their creators"),
    ("rt", "grants", "What is the maximum research advance for a principal investigator?", "21. The principal investigator may apply"),
    ("r2", None, "Who signs licenses for university-owned intellectual property?", "7. The authority to sign licenses"),
]


@pytest.mark.parametrize("source,group,question,needle", STRONG)
def test_a_strong_relation_is_supported_by_the_stating_statement(corpus_of, source, group, question, needle):
    result = ranked_run(corpus_of, source, group, question, needle)
    assert result.status is S, (question, result.missing, RT.quotes(result))
    c = clause(result)
    assert c["ask_shape"] == R.ASK_RECOGNISED and c["relation"] == R.RELATION_STRONG, (question, c)
    stating = [f for f in result.findings if needle in f.evidence_quote]
    assert stating and stating[0].metadata["relation"] == R.RELATION_STRONG, (question, RT.quotes(result))


# ---------------------------------------------------------------------------
# B. AMBIGUOUS relations: the relation may hold but no deterministic rule proves it -> never SUPPORTED
# ---------------------------------------------------------------------------
AMBIGUOUS = [
    # a pronoun object ("have THEM reviewed")
    ("rt", "grants", "Who reviews the research grant forms?", "have them reviewed by research committees"),
    ("rt", "grants", "Who reviews the research grant application forms?", "have them reviewed by research committees"),
    # a participle attached after a comma or a conjunction: its noun is not provable
    ("rc", "grants", "Who nominates the Editorial Secretary?", "nominated by the Editor"),
    ("rc", "grants", "Who appoints the Editorial Secretary?", "nominated by the Editor"),
    ("rc", "grants", "Who approves the annual research progress report?", "annual progress report for the project, approved by the Dean"),
    # a bare definite anaphor ("reviews THE application", "close THE project") for a modified asked target
    ("rc", "grants", "Who reviews my research grant application?", "The Research Purchasing Committee under the VCRGS"),
    ("rc", "grants", "Who reviews the application for a research grant?", "The Research Purchasing Committee under the VCRGS"),
    ("rc", "grants", "Who approves the research grant application?", "The Research Purchasing Committee under the VCRGS"),
    ("rc", "grants", "Who reviews the application for a research advance?", "The Research Purchasing Committee under the VCRGS"),
    # an approval stated of another sentence's subject
    ("rc", "grants", "Who approves a research advance?", "The Research Purchasing Committee under the VCRGS"),
    ("rc", "grants", "Who approves invoices for a research advance?", "The Research Purchasing Committee under the VCRGS"),
    # the required thing is not what the statement requires (a synonym is not proof)
    ("rt", "consultancy", "Is IRB clearance required before recruiting human participants?", "Researchers must obtain IRB approval"),
    ("r2", None, "Is IRB approval necessary for every study?", "Researchers must obtain written informed consent"),
    ("r2", None, "Is IRB approval necessary for every study?", "Researchers must monitor the progress of the study"),
    ("rc", "grants", "Is a principal investigator mandatory for each research project?", "15. The principal investigator of a project must hold"),
    # an approval inside a list of conditions ("subject to performance evaluation, University policy, Board approval, ...")
    ("r2", None, "Who approves the registration of research groups?", "Research groups are registered for three renewable years"),
    # the head in the complement of another head
    ("r2", None, "Who approves the coverage of research projects?", "The College Dean may approve the coverage of publishing"),
    ("rc", "grants", "Who approves the Editorial Secretary's appointment?", "nominated by the Editor"),
    # an act stated of a list item whose lead-in does not name the asked subject
    ("r2", None, "Who approves research group budgets?", "7. Approving the annual research budget"),
    ("rc", "ip", "Who reviews Intellectual Property disclosures?", "h. The TTO shall inform the Inventor(s)"),
    # a generic head shared by two objects: "the advance" of a petty cash unit never answers for a research advance
    ("rc", "grants", "When can a research advance be closed?", "20. The principal investigator is entitled"),
    # a yes/no proposition the unit does not state ("extra pay" / "sabbatical or financial awards")
    ("rt", "grants", "Is extra pay available for coordinators of research groups?", "i. The University Research Board may approve"),
]


@pytest.mark.parametrize("source,group,question,needle", AMBIGUOUS)
def test_an_ambiguous_relation_is_never_supported(corpus_of, source, group, question, needle):
    result = ranked_run(corpus_of, source, group, question, needle)
    assert result.status is not S, (question, clause(result), RT.quotes(result))
    c = clause(result)
    if result.status is P and c["ask_shape"] == R.ASK_RECOGNISED and c["relation"] == R.RELATION_AMBIGUOUS:
        assert any("Milestone 5 Verifier" in m for m in result.missing), result.missing


# ---------------------------------------------------------------------------
# C. WRONG ACTION: the evidence states another act on the asked object -> never SUPPORTED
# ---------------------------------------------------------------------------
WRONG_ACTION = [
    ("rt", "grants", "Who approves research grant applications?", "Research committees review research grant applications"),
    ("rt", "conference", "Who reviews conference travel requests?", "Vice Chancellor for Academic Affairs approves travel requests"),
    ("rt", "grants", "Who reviews the funding ceiling for research grants?", "VCRGS determines the funding ceiling"),
    ("rt", "ip", "Who signs the invention disclosure form?", "obligated to promptly disclose"),
    ("r2", None, "Who approves the appointment of a moderator in an IP dispute?", "shall appoint a moderator"),
    ("rt", "grants", "Who reviews collaborative research agreements?", "Collaborative research agreements with external organizations"),
    ("rt", "grants", "Who signs collaborative research agreements?", "Collaborative research agreements with external organizations"),
    ("rt", "ip", "Who approves invention disclosures?", "obligated to promptly disclose"),
    ("rt", "grants", "Who decides the funding ceiling for a seed project?", "d. The funding ceiling for each seed project"),
    ("rc", "consultancy", "Who approves consultancy time allocation?", "16.1.7 Time Allocation for Consultancy"),
    ("rc", "grants", "Who approves hosting research groups?", "Research groups report to the VCRGS"),
    ("rt", "grants", "Who approves teaching load reductions for journal editors?", "have their teaching loads reduced"),
    ("rt", "grants", "Who reviews petty cash advances?", "entitled to a petty cash advance"),
    ("rt", "ip", "Who approves the protection of inventions?", "shall actively promote the protection"),
    ("rt", "consultancy", "Who reviews conflicts of interest?", "must disclose any conflict of interest"),
]


@pytest.mark.parametrize("source,group,question,needle", WRONG_ACTION)
def test_a_wrong_action_is_never_supported(corpus_of, source, group, question, needle):
    never_supported(corpus_of, source, group, question, needle)


# ---------------------------------------------------------------------------
# D. WRONG ROLE: the rule is stated for another role -> never SUPPORTED
# ---------------------------------------------------------------------------
WRONG_ROLE = [
    ("rt", "grants", "How much petty cash can a co-investigator receive?", "entitled to a petty cash advance"),
    ("rt", "supervision", "How many theses can a research assistant supervise?", "supervise a maximum of four theses"),
    ("rt", "ip", "Who owns intellectual property created by visiting researchers?", "shall own all Intellectual Property Rights"),
    ("rt", "ip", "What percentage of net royalty revenue goes to the Dean?", "50% to the inventor(s)"),
    ("rt", "conference", "How many conferences can a research assistant attend per year?", "one local and one external university-supported conference"),
    ("rt", "conference", "Who approves conference travel for students?", "Vice Chancellor for Academic Affairs approves travel requests"),
    ("rt", "consultancy", "How many days per week can students spend on consultancy?", "may not exceed one day per week"),
    ("rc", "grants", "Do co-investigators need a Ph.D. degree?", "15. The principal investigator of a project must hold"),
    ("rc", "grants", "What is the research advance limit for co-investigators?", "21. The principal investigator may apply for a research advance"),
    ("rt", "consultancy", "Do inventors have to disclose conflicts of interest?", "must disclose any conflict of interest"),
    ("rc", "ip", "Who owns copyrighted materials created by staff?", "Copyrighted materials remain the property of their creators"),
    ("rc", "conference", "How many conferences can a postdoc attend per year?", "allowed to attend a maximum of one international"),
    ("rt", "grants", "Is the Dean entitled to a petty cash advance?", "entitled to a petty cash advance"),
    ("rt", "grants", "How long is sabbatical leave for the Chief Editor?", "Sabbatical leave for visiting researchers"),
    ("rt", "ip", "What must a research assistant disclose to the TTO?", "obligated to promptly disclose"),
]


@pytest.mark.parametrize("source,group,question,needle", WRONG_ROLE)
def test_a_rule_for_another_role_is_never_supported(corpus_of, source, group, question, needle):
    never_supported(corpus_of, source, group, question, needle)


# ---------------------------------------------------------------------------
# E. WRONG PURPOSE / BENEFICIARY: the rule is stated for another purpose, receiver or beneficiary -> never SUPPORTED
# ---------------------------------------------------------------------------
WRONG_PURPOSE = [
    ("rt", "ip", "Who approves transferring tangible research property for research purposes?", "For commercial purposes, transferring"),
    ("rt", "ip", "Who approves transferring tangible research property for teaching purposes?", "For commercial purposes, transferring"),
    ("rt", "grants", "Who approves collaborative research agreements for commercial purposes?", "Collaborative research agreements with external organizations"),
    ("rt", "ip", "Who owns intellectual property created by faculty for personal purposes?", "shall own all Intellectual Property Rights"),
    ("rt", "ip", "What percentage of net royalty revenue goes to the department?", "3. After recovering IP protection expenses"),
    ("rt", "ip", "What percentage of the University's share goes to the TTO?", "The University's share, in turn"),
    ("r2", None, "What percentage of the University's share goes to the inventor?", "b. 5% to the Inventors"),
    ("rc", "grants", "Does the University allocate a yearly budget for faculty research projects?", "allocates a yearly budget"),
    ("r2", None, "What percentage of external research grants is deducted for the College's account?", "30% of research grants deducted"),
    ("rt", "grants", "Who reviews research grant applications submitted by students?", "Research committees review research grant applications"),
    ("rc", "grants", "What is the funding ceiling for a faculty research project?", "d. The funding ceiling for each project"),
    ("rc", "consultancy", "Who approves the use of the University's name in consultancy for commercial purposes?", "16.1.8 Use of University Name"),
    ("rt", "grants", "What percentage of the project's budget can be spent on conference travel?", "e. Costs for research assistants"),
    ("rt", "grants", "What percentage of the research project's budget can be spent on equipment?", "e. Costs for research assistants"),
    ("rt", "grants", "How long is sabbatical leave for faculty members?", "Sabbatical leave for visiting researchers"),
    # a purpose the statement does not state
    ("rt", "ip", "Who owns intellectual property created by faculty for research purposes?", "shall own all Intellectual Property Rights"),
    # a percentage of another base: 50% of the net royalty revenue is not a share of the University's share
    ("rt", "ip", "What percentage of the University's share goes to the inventor?", "3. After recovering IP protection expenses"),
]


@pytest.mark.parametrize("source,group,question,needle", WRONG_PURPOSE)
def test_a_rule_for_another_purpose_or_beneficiary_is_never_supported(corpus_of, source, group, question, needle):
    never_supported(corpus_of, source, group, question, needle)


# ---------------------------------------------------------------------------
# F. GENERAL controls: overview questions stay supported (the relation contract does not apply)
# ---------------------------------------------------------------------------
GENERAL = [
    ("grants", "What are research groups?", "encourages the establishment of research groups"),
    ("ip", "What does the TTO do?", "to manage all intellectual property assets"),
    ("grants", "Tell me about research grants.", "robust support for research projects"),
    ("ip", "Explain the University's intellectual property policy.", "shall own all Intellectual Property Rights"),
    ("grants", "What are the responsibilities of the Research Funding Department?", "is responsible for pre- and post-award management"),
    ("ip", "What is the role of the Technology Transfer Office?", "to manage all intellectual property assets"),
    ("grants", "Describe the University Research Board.", "overseen by the University Research Board"),
    ("grants", "What are the research priority areas?", "research priority areas"),
    ("ip", "Give me an overview of the IP policy.", "shall own all Intellectual Property Rights"),
    ("grants", "What are research institutes?", "three research institutes"),
    ("grants", "What are the functions of the research committees?", "Research committees review research grant applications"),
    ("consultancy", "Tell me about faculty consultancy.", "recognizes the value of faculty consultancy"),
    ("ip", "What is commercialization at UoS?", "Commercialization is a priority"),
    ("consultancy", "What is the Institutional Review Board?", "An independent committee responsible for reviewing"),
    ("grants", "Tell me about the research institutes.", "three research institutes"),
]


@pytest.mark.parametrize("group,question,needle", GENERAL)
def test_a_general_overview_question_is_still_supported(corpus_of, group, question, needle):
    result = ranked_run(corpus_of, "rt", group, question, needle)
    assert result.status is S, (question, result.missing, RT.quotes(result))
    assert clause(result)["ask_shape"] == R.ASK_GENERAL and clause(result)["relation"] is None
    assert any(needle in q for q in RT.quotes(result))
    assert all("relation" not in f.metadata for f in result.findings)


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------
def test_relation_strength_is_strong_only_for_a_covered_recognised_grounding():
    grounding = [R._grounding_detail([], (), None, None, None, frozenset(), R.RelationFrame(frozenset(), (), None, ()))]
    key = R._details._detail_key(grounding[0])
    assert R._relation_strength(R.ASK_RECOGNISED, grounding, frozenset({key})) == R.RELATION_STRONG
    assert R._relation_strength(R.ASK_RECOGNISED, grounding, frozenset()) == R.RELATION_AMBIGUOUS
    assert R._relation_strength(R.ASK_RECOGNISED, [], frozenset()) == R.RELATION_AMBIGUOUS
    for shape in (R.ASK_GENERAL, R.ASK_UNKNOWN, None):
        assert R._relation_strength(shape, grounding, frozenset({key})) is None
    assert R._relation_downgrade(R.RELATION_AMBIGUOUS) is True
    assert R._relation_downgrade(R.RELATION_STRONG) is False and R._relation_downgrade(None) is False


def test_an_ambiguous_relation_is_reported_with_the_verifier_boundary(corpus_of):
    result = ranked_run(corpus_of, "rt", "grants", "Who reviews the research grant forms?", "have them reviewed by research committees")
    assert result.status is P and clause(result)["relation"] == R.RELATION_AMBIGUOUS
    message = next(m for m in result.missing if "not established unambiguously" in m)
    assert "semantic relation verification belongs to the Milestone 5 Verifier" in message
    assert all(f.metadata["relation"] == R.RELATION_AMBIGUOUS for f in result.findings)


def frame(question):
    assessment = R.assess_ask(question, R.requested_details(question), question)
    return R._relation_frame(question, assessment.details, "yes_no" in assessment.kinds)


def test_the_relation_frame_names_roles_actions_purpose_receivers_and_the_required_thing():
    f = frame("Who approves transferring tangible research property for commercial purposes?")
    assert [a for a, _h in f.actions] == ["approve", "transfer"] and f.purpose == "commercial"
    assert frame("How much petty cash can a co-investigator receive?").roles == frozenset({"co_investigator"})
    assert frame("How many theses can a principal investigator supervise?").roles == frozenset({"principal_investigator"})
    assert frame("What percentage of net royalty revenue goes to the inventor?").recipients == ("inventor",)
    assert frame("Is IRB clearance required before recruiting human participants?").required == "clearance"
    assert frame("Should researchers get Dean approval for every project?").required == "approval"
    assert dict(frame("Who reviews my research grant application?").modifiers)["review"] == ("research", "grant")
    assert dict(frame("Who reviews the application for a research grant?").modifiers)["review"] == ("research", "grant")
    assert frame("Who approves research involving animals?").actions == (("approve", "research"),)
    assert frame("Who reviews my research grant application?").actions == (("review", "application"),)


# ---------------------------------------------------------------------------
# The deterministic building blocks
# ---------------------------------------------------------------------------
def occurrence(text, word):
    m = re.search(r"\b%s\b" % re.escape(word), text)
    assert m, (word, text)
    return m


@pytest.mark.parametrize("text,word,form", [
    # how a QUESTION uses an action word (a statement's own nouns are screened by _statement_noun)
    ("Who approves the research budget?", "approves", "verbal"),
    ("Can I travel to a conference?", "travel", "verbal"),
    ("When is the research budget approved?", "approved", "verbal"),
    ("Who owns IP created by faculty?", "created", "modifier"),
    ("Is prior approval required?", "approval", "nominal"),
    ("Who gives approval for the budget?", "approval", "nominal"),
    ("Who approves conference travel requests?", "travel", "noun"),
    ("Who approves the funding decisions?", "funding", "noun"),
])
def test_an_occurrence_is_verbal_nominal_a_modifier_or_a_noun(text, word, form):
    assert R._occurrence_form(text, occurrence(text, word)) == form


@pytest.mark.parametrize("text,word,expected", [
    ("The advance may be closed at the end of the year.", "closed", "The advance may be"),
    ("Original invoices approved by the Dean are accepted.", "approved", "Original invoices"),
    ("A report for the project, approved by the Dean.", "approved", ""),
    ("The Dean may approve the project's closure if its objectives are met.", "approve", "the project's closure if its objectives are met"),
    ("Agreements require the approval of the VCRGS.", "approval", "Agreements require the"),
])
def test_the_local_target_is_the_noun_an_occurrence_is_predicated_of(text, word, expected):
    assert R._local_target(text, occurrence(text, word)).strip() == expected


@pytest.mark.parametrize("target,phrase,complement", [
    ("travel requests and interprets the policy", {"travel", "requests", "interprets", "policy"}, set()),
    ("the funding ceiling for research grants based on", {"funding", "ceiling"}, {"research", "grants"}),
    ("the project's closure if its objectives", {"closure"}, set()),
    ("licenses or agreements related to the transfer", {"licenses", "agreements"}, set()),
    ("research protocols involving human subjects", {"research", "protocols"}, set()),
    ("in one local and one external university supported conference per year", {"one", "local", "external", "university", "supported", "conference"}, set()),
    ("a moderator (arbiter) to assist", {"moderator"}, set()),
])
def test_the_object_noun_phrase_and_its_complement(target, phrase, complement):
    words, extra = R._object_words(target)
    assert set(words) == phrase and set(extra) == complement


def test_the_asked_head_heads_the_phrase_or_is_in_the_complement_of_a_head_the_question_names():
    assert R._phrase_names("the funding ceiling for research grants", "ceiling", frozenset())
    assert not R._phrase_names("payment procedures for research projects", "project", frozenset({"dean", "approval", "project"}))
    assert R._phrase_names("copyright protection for books, computer software", "software", frozenset({"copyright", "protection", "software"}))
    assert R._bare_definite("the application and makes recommendations") and not R._bare_definite("travel requests")


@pytest.mark.parametrize("roles,statement,unit,ok", [
    ({"principal_investigator"}, "The principal investigator is entitled to an advance.", "", True),
    ({"co_investigator"}, "The principal investigator is entitled to an advance.", "", False),
    ({"principal_investigator"}, "Additionally, they must be full-time employees.",
     "The principal investigator must hold a Ph.D. Additionally, they must be full-time employees.", True),
    ({"inventor"}, "Faculty members must disclose conflicts.", "Inventors are faculty. Faculty members must disclose conflicts.", False),
])
def test_an_asked_role_is_named_in_the_statement_or_by_its_pronoun(roles, statement, unit, ok):
    assert R._roles_stated(frozenset(roles), statement, unit or statement) is ok


@pytest.mark.parametrize("statement,actions,ok", [
    ("Faculty members may apply for grants.", (("apply", None),), True),
    ("Faculty on sabbatical are not allowed to combine teaching with research.", (("teach", None),), False),
    ("Participation without a paper may be permitted with the Chancellor's approval.", (("attend", None),), True),
    ("Participation without a paper may be permitted with the Chancellor's approval.", (("publish", None),), False),
])
def test_a_permission_answers_only_the_action_it_governs(statement, actions, ok):
    assert R._permission_governs(statement, actions) is ok


@pytest.mark.parametrize("recipients,statement,ok", [
    (("inventor",), "3. After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), "
                    "and 50% to the University.", True),
    (("inventor",), "The University's share, in turn, shall be distributed as follows: a. 40% to the VCRGS. b. 5% to the Inventors' College.", False),
    (("vcrgs",), "3. After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), "
                 "and 50% to the University.", False),
    ((), "Anything at all.", True),
])
def test_a_share_receiver_is_named_as_the_receiver_in_the_statement(recipients, statement, ok):
    assert R._recipients_stated(recipients, statement) is ok


def test_a_share_owner_is_stated_with_the_share_or_by_its_list_lead_in():
    unit = ("The University's share, in turn, shall be distributed as follows: a. 40% to the VCRGS to support TTO and research activities "
            "at the University. b. 5% to the Inventors' College.")
    assert R._share_statement_ok("university", unit, unit)
    assert not R._share_statement_ok("inventor", unit, unit)
    item = "40% to the VCRGS to support TTO and research activities at the University."
    assert R._share_statement_ok("university", item, item)                 # a single statement: its lead-in is checked by the bind
    assert R._share_statement_ok(None, item, item)
    labelled = "a. 40% to the VCRGS. b. 5% to the Inventors' College."
    assert not R._share_statement_ok("university", "40% to the VCRGS.", labelled)   # no lead-in in the unit: not proved


@pytest.mark.parametrize("head,statement,ok", [
    ("approval", "Even in cases of non-compensated consultancy, prior approval is required.", True),
    ("approval", "Researchers must obtain IRB approval before recruiting human participants.", True),
    ("approval", "All research involving animals must be approved by the IACUC.", True),
    ("clearance", "Researchers must obtain IRB approval before recruiting human participants.", False),
    ("investigator", "The principal investigator of a project must hold a Ph.D. degree.", False),
    ("approval", "Researchers must obtain written informed consent from all participants.", False),
    (None, "Anything at all.", True),
])
def test_the_required_thing_is_what_the_statement_requires(head, statement, ok):
    assert R._requirement_binds(head, statement) is ok


# ---------------------------------------------------------------------------
# Earlier layers the relation contract now overlaps (kept as defence in depth, pinned so a regression is still visible)
# ---------------------------------------------------------------------------
def test_an_adjectival_participle_is_no_approval_act():
    assert R._approval_targets("The College Dean may cover the cost if the researcher is involved in an approved research project.") == []
    assert R._approval_targets("An appointed Editor-in-Chief leads each series.") == []
    assert R._approval_targets("The research project is approved by the Dean.") == ["The research project is "]


def test_a_stated_purpose_other_than_the_asked_one_is_a_conflict():
    statement = "For commercial purposes, transferring tangible research property requires the approval of the VCRGS."
    assert R._purpose_conflict("research", statement) is True
    assert R._purpose_conflict("commercial", statement) is False
    assert R._purpose_conflict("research", "Agreements require the approval of the VCRGS.") is False
    assert R._purpose_ok("research", "Agreements require the approval of the VCRGS.") is False   # the contract: the asked purpose is stated


# ---------------------------------------------------------------------------
# G. POSITION BINDING (the fresh adversarial check of Step 4.5A-D): a statement that names the asked party proves the
# relation only when the party occupies the asked position (subject, agent, patient, recipient), a negated or
# contrasting occurrence states nothing, and a period is the stated one. Lines mirror the handbook.
# ---------------------------------------------------------------------------
POSITION_PAGES = [
    RT.page(117, "5.4 Participation in Conferences",
            "• Conferences attended at no cost to the University are discretionary and require the same approval process as cost-based conferences.",
            "• Attendance as plenary, keynote, or invited speakers fully sponsored by external organizations is allowed. If not fully sponsored and meeting the requirements, such attendance will count against the travel limits.",
            "• Conferences supported by external research grants are permitted outside the college-allocated budget, provided they do not conflict with primary commitments and follow the same rules.",
            "• Faculty members are allowed to attend a maximum of one international and one local university-supported conference per year, provided it does not conflict with their primary commitments."),
    RT.page(118, "• The University allocates an annual budget for each college to cover conference expenses. Funds should be distributed equitably among faculty.",
            "• University-supported participation without a paper may be permitted with the Chancellor's approval in cases such as:",
            "o Serving as a main member of the organizing committee.", "o Being a keynote speaker."),
    RT.page(119, "o Uncompensated conference travel may be authorized based on its value to the University and is subject to the same review and approval process as compensated conferences.",
            "• Per diem rates between AED 1000-1500 for a maximum of four days, depending on the conference location.",
            "5.5 Policy on Offering Courses in the Summer Semester",
            "1. Faculty members' compensation for teaching in the summer semester shall not exceed two months' basic salary."),
    RT.page(155, "In addition to the above, internal funding includes a range of other grants, such as collaborative research grants, research visits grants, summer stay research grants, extra conference participation grants, researchers' hosting grants, exchange student researchers grant, and grants for graduate research assistants."),
    RT.page(160, "8.6 Research Committees", "The responsibilities of these committees include:",
            "1. Reviewing research grant applications submitted by faculty members, ensuring their eligibility, and providing appropriate recommendations to the Deanship through the deans of colleges.",
            "4. Assessing the colleges' needs for major research equipment and forwarding complete applications to the VCRGS through the College Deans."),
    RT.page(157, "8.4 Research Groups",
            "Research groups serve as the nucleus for potential future research centers and units, subject to recommendations in line with approved regulations and rules set forth by the Research Board upon the Dean's recommendation."),
    RT.page(163, "8.9 Scientific Publishing at the University of Sharjah Journal",
            "The Chancellor assumes the role of the general supervisor of the journal, with an appointed Editor-In-Chief for each series.",
            "Additionally, an Editorial Secretary from among the faculty members, nominated by the Editor-In-Chief and appointed with the Chancellor's approval, supports the journal's operations.",
            "The Editorial Secretary may receive an appropriate financial renumeration or have their teaching loads reduced based on a percentage determined by the Chancellor."),
    RT.page(165, "c. The VCRGS determines the funding ceiling for research grants based on the Board's recommendation and the Chancellor's approval.",
            "d. Colleges provide the VCRGS with names and addresses of potential referees related to the research proposal."),
    RT.page(166, "20. The principal investigator is entitled to a petty cash advance of up to AED 5000, subject to approval by the VCRGS. The advance may be closed at the end of the academic year or carried forward to the next academic year. The Principal Investigator may extend the advance to another period based on the project budget and approved completion period. The petty cash amount may also be substituted by original invoices approved by the Dean until the advance term's closure at the end of the project.",
            "23. In case a reference book required for the research project is not available in the University Library, the following steps are taken:",
            "a. The principal investigator submits a book acquisition application to the VCRGS.",
            "24. The principal investigator must immediately deliver the purchased book to the VCRGS for approval under the University Library Cataloguing System for Scientific Research.",
            "21. The principal investigator may apply for a research advance of up to 10% of the basic research project grant. The Research Purchasing Committee under the VCRGS reviews the application and makes recommendations, subject to approval by the VCRGS.",
            "22. The advance may be raised to a maximum of 30% of the basic research project grant in special cases justified by the principal investigator and approved by the VCRGS upon the Research Purchasing Committee's recommendation."),
    RT.page(168, "29. Reporting Requirements for Research Projects",
            "The principal investigator must provide the VCRGS with an annual progress report for the project, approved by the Dean. Upon project completion, a final report and a special form provided by the VCRGS must be submitted. The Dean may approve the project's closure if its objectives have been achieved. However, if the project period ends, and the final report is not submitted or justified, the Dean may refer the issue to the Board, which may close the project and reclaim any spent funds from the investigator.",
            "30. Payment Procedures for Research Projects",
            "All payment procedures for competitive, collaborative, targeted, seed, and student research projects must be approved by the Dean."),
    RT.page(169, "34. Coverage of Publishing Academic Research in Conferences",
            "The College Dean may approve the coverage of publishing academic research presented at international scientific conferences and gatherings if the researcher is involved in an approved research project as a principal investigator, associate investigator, coordinator, or member of a research group. The publishing cost will be deducted from the research group's budget or the research project's budget, as applicable."),
    RT.page(169, "36. Formation of Research Groups", "Research groups must observe the following:",
            "a. Groups of researchers (at least three) can register a research group under the VCRG with an appropriate name. One researcher, preferably holding the rank of Professor or Associate Professor, should coordinate its functions. Researchers can be from the same college or different colleges, and external researchers may join as associate members without receiving University member privileges."),
    RT.page(170, "i. The University Research Board may approve partial or full sabbatical or financial awards for coordinators of research groups, upon the Chancellor's approval.",
            "37. Hosting Research Groups",
            "Research groups report to the VCRGS but may be hosted by colleges, which should publicize them through various media, including print and electronic formats."),
    RT.page(171, "43. Pricing of Published Books",
            "The prices of books published by the VCRGS are determined by the VCRGS, upon the recommendation of the Head of the Scientific Research Department."),
    RT.page(172, "50. Publication of Conference Proceedings",
            "The University publishes conference and symposium proceedings, with organizers responsible for collecting, editing, formatting, reviewing, and submitting the final document to the VCRGS for publication.",
            "51. Requirements for Publishing Conference Proceedings",
            "Organizers submit requests for publishing proceedings through the college or research group that organized the event."),
    RT.page(173, "56. External Funding for Research Projects",
            "The VCRGS facilitates obtaining external funding from national and international institutions for research projects, with 30% of research grants deducted for the University of Sharjah's account.",
            "57. Allocation of Research Grants",
            "University grants cover faculty members' costs based on a recommended equation, considering any external funding received from the host institution.",
            "60. Visiting Researcher Grants",
            "Faculty members can apply for grants to host visiting researchers, subject to Board or VCRGS approval for research groups."),
    RT.page(174, "Sabbatical leave for visiting researchers should not exceed two months and should not interfere with University commitments."),
    RT.page(177, "9.2 University Responsibilities",
            "The Technology Transfer Office (TTO) under the Office of the Vice Chancellor for Research and Graduate Studies shall undertake the following responsibilities:",
            "a. Establish procedures for the exploitation of Intellectual Property created by UoS Inventors.",
            "d. Reward inventors by ensuring a fair sharing of royalties from inventions assigned to the university."),
    RT.page(178, "9.4.1 The Obligation to Disclose",
            "a. Individuals subject to this Intellectual Property Policy are obligated to promptly disclose to the TTO all Intellectual Property in which the University has an ownership interest under the provisions of the Intellectual Property Policy."),
    RT.page(179, "c. As per this Policy, the TTO is required to consult with the Inventor(s) or other relevant parties to better understand the disclosure's scientific and commercial merit and develop strategies for protection, development, and commercialization."),
    RT.page(180, "6. UoS has a royalty-free non-commercial use of any copyrightable materials developed by staff or students (except for books and materials published outside the university) if they were developed using university facilities, supplies, equipment, or services."),
    RT.page(181, "3. The TTO, in consultation with the inventor(s), shall choose the most suitable method to transfer UoS Intellectual Property that aligns with its mission and benefits the public.",
            "9.9 Distribution of Licensing Revenues",
            "3. After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), and 50% to the University.",
            "The University's share, in turn, shall be distributed as follows: a. 40% to the VCRGS to support TTO and research activities at the University. b. 5% to the Inventors' College(s), if applicable."),
    RT.page(182, "7. In the commercialization of intellectual property owned by more than one inventor, the net royalty income shall be shared equally among all inventors, unless they agree otherwise as specified in the IP Disclosure Form."),
    RT.page(260, "16.1.7 Time Allocation for Consultancy: Full-time faculty members are permitted to spend up to 20% of their time or an equivalent of one working day per week on professional consultancy activities.",
            "16.1.9 Independent Consultancy: Independent consultancy is the sole responsibility of the academic member consultant. The academic member must obtain pre-approval from the College Dean, and in the event the Dean is planning to engage in independent professional consultancy, a written approval from the Vice Chancellor for Academic Affairs is required."),
    RT.page(261, "16.1.11 Scholarly Communications and Consultation: Scholarly communications that may earn academic member compensation are not viewed as consultancy."),
    RT.page(265, "16.3.9 Compliance and Oversight: UoS is committed to complying with all national and international regulations and guidelines pertaining to animal research. The IACUC ensures regular audits, inspections, and reviews of animal research activities to maintain ethical standards and adherence to the policy.",
            "16.4 Human Use in Research"),
    RT.page(266, "Informed Consent: Informed consent is a fundamental aspect of human research. Researchers must obtain written informed consent from all participants before their involvement in the study.",
            "Monitoring and Reporting: Researchers must monitor the progress of the study regularly and promptly report any adverse events or unanticipated problems to the IRB.",
            "16.5 Policy Prohibiting Sexual Harassment",
            "The University prohibits sexual harassment and will investigate every complaint promptly."),
]


@pytest.fixture(scope="module")
def position_corpus(registry_and_map):  # noqa: F811
    return RT.corpus_from(registry_and_map, POSITION_PAGES)


def position_run(corpus, question, needle):
    sp = RT.make_specialist(corpus, RT.ScriptedReranker([(needle, 9.0)], default=-5.0))
    result = RT.run(sp, question)
    RT.assert_traceable(result, corpus)
    return result


WRONG_POSITION = [
    # the asked party is not the subject of the asked predicate
    ("Does the Chancellor determine the funding ceiling for research grants?", "VCRGS determines the funding ceiling"),
    ("Does the VCRGS approve the funding ceiling for research grants?", "VCRGS determines the funding ceiling"),
    ("Is the VCRGS entitled to a petty cash advance of up to AED 5000?", "entitled to a petty cash advance"),
    ("How much petty cash advance can the VCRGS receive?", "entitled to a petty cash advance"),
    ("Does the VCRGS submit the book acquisition application?", "submits a book acquisition application"),
    ("Must the VCRGS deliver the purchased book to the principal investigator?", "deliver the purchased book"),
    ("Does the VCRGS host research groups?", "may be hosted by colleges"),
    ("Does the VCRGS provide colleges with the names and addresses of potential referees?", "Colleges provide the VCRGS"),
    ("Can the Dean reclaim spent funds from the investigator?", "reclaim any spent funds"),
    ("Can visiting researchers apply for grants?", "Faculty members can apply for grants to host visiting researchers"),
    ("Do organizers publish conference proceedings?", "The University publishes conference and symposium proceedings"),
    ("Does the college submit requests for publishing proceedings?", "Organizers submit requests"),
    ("Do deans review research grant applications?", "Reviewing research grant applications"),
    ("Do college deans assess the colleges' needs for major research equipment?", "Assessing the colleges' needs"),
    ("Is the inventor required to consult with the TTO?", "the TTO is required to consult"),
    ("Does the inventor choose the most suitable method to transfer UoS intellectual property?", "shall choose the most suitable method"),
    ("Do inventors establish procedures for the exploitation of intellectual property?", "Establish procedures for the exploitation"),
    ("Do researchers ensure regular audits and inspections of animal research?", "The IACUC ensures regular audits"),
    ("Does the VCRGS recommend the prices of published books?", "upon the recommendation of the Head"),
    ("Does the Head of the Scientific Research Department determine the prices of published books?", "determined by the VCRGS"),
    ("How much of the University's share does the TTO receive?", "40% to the VCRGS to support TTO"),
    ("Can the coordinator approve sabbatical awards for research groups?", "awards for coordinators of research groups"),
    # the asked agent of a passive is another party
    ("Are book prices determined by the Head of the Scientific Research Department?", "determined by the VCRGS"),
    ("Is the Editorial Secretary nominated by the Chancellor?", "nominated by the Editor-In-Chief"),
    ("Is the Editorial Secretary appointed by the Editor-in-Chief?", "nominated by the Editor-In-Chief"),
    ("Is attendance as an invited speaker sponsored by the University allowed?", "sponsored by external organizations"),
    ("Are conferences supported by internal research grants permitted outside the college budget?", "supported by external research grants"),
    ("Does UoS have royalty-free use of copyrightable materials developed by faculty?", "developed by staff or students"),
    # the stated patient contrasts with the asked one
    ("Can compensated conference travel be authorized based on its value to the University?", "Uncompensated conference travel"),
    # a recipient, source or beneficiary stated with the same preposition names another party
    ("Must researchers report adverse events to the IACUC?", "to the IRB"),
    ("Must researchers obtain written informed consent from the IRB?", "from all participants"),
    ("Must the principal investigator deliver the purchased book to the Library?", "deliver the purchased book to the VCRGS"),
    ("Is the conference publishing cost deducted from the college budget?", "deducted from the research group's budget"),
    ("Is net royalty income shared equally between the inventor and the university?", "shared equally among all inventors"),
    ("Is funding allocated equitably among colleges for conference expenses?", "distributed equitably among faculty"),
    ("Is the summer stay research grant for students?", "summer stay research grants"),
    ("Are there grants for undergraduate research assistants?", "grants for graduate research assistants"),
    ("Does the university allocate an annual budget for research groups to cover conference expenses?", "annual budget for each college"),
    # a negated occurrence, a number, a temporal word, "without"
    ("Does UoS have royalty-free commercial use of copyrightable materials developed by students?", "non-commercial use"),
    ("Are conferences attended at the University's cost discretionary?", "at no cost to the University"),
    ("Are scholarly communications that earn compensation viewed as research?", "are not viewed as consultancy"),
    ("Am I allowed to spend 30% of my time on consultancy?", "up to 20% of their time"),
    ("Must researchers obtain written informed consent after the study?", "before their involvement in the study"),
    ("Can the principal investigator extend the advance without the project budget?", "based on the project budget"),
    # the stated object restricts the head with a word the question lacks
    ("May the Editorial Secretary have their research load reduced?", "teaching loads reduced"),
    ("Is the Chancellor the general supervisor of the research groups?", "general supervisor of the journal"),
    ("How many regional conferences can faculty attend each year?", "one international and one local"),
    ("Who reviews publication grant applications?", "Reviewing research grant applications"),
    ("Do university grants cover costs of the host institution?", "cover faculty members' costs"),
    ("How long can the sabbatical leave of a faculty member who hosts visiting researchers be?", "Sabbatical leave for visiting researchers"),
    # another period
    ("How many days per month can faculty spend on consultancy?", "one working day per week"),
    ("How many international conferences can I attend per semester?", "one international and one local"),
    ("What is the per diem per week for international conferences?", "maximum of four days"),
    # a rule conditioned on another role
    ("Does an academic member need written approval from the VCAA for independent consultancy?", "in the event the Dean is planning"),
    # the 1,069-question re-run: a recipient is not the object, "without", a value ask's predicate and complement, a
    # percentage base, a nominal attached to a whole clause, an asked "for" complement, a list lead-in's subject
    ("Does the TTO disclose intellectual property?", "obligated to promptly disclose to the TTO"),
    ("How many conferences per year can faculty attend without university support?", "one international and one local"),
    ("When can the petty cash advance be carried forward?", "substituted by original invoices approved by the Dean"),
    ("What is the minimum number of external researchers in a research group?", "Groups of researchers (at least three)"),
    ("What percentage of total licensing revenue does the VCRGS receive?", "40% to the VCRGS"),
    ("Who approves research group coordinators?", "awards for coordinators of research groups"),
    ("Who approves full sabbatical for research group members?", "awards for coordinators of research groups"),
    ("Who approves the Research Purchasing Committee?", "The Research Purchasing Committee under the VCRGS reviews"),
    ("Who recommends research centers?", "nucleus for potential future research centers"),
    ("Who recommends the basic research project grant?", "upon the Research Purchasing Committee's recommendation"),
    ("Who determines the funding ceiling for publication grants?", "VCRGS determines the funding ceiling"),
    ("Who reviews applications for publication grants?", "Reviewing research grant applications"),
    ("Do faculty members review research grant applications?", "Reviewing research grant applications"),
    ("Do researchers assess the colleges' needs for major research equipment?", "Assessing the colleges' needs"),
]


@pytest.mark.parametrize("question,needle", WRONG_POSITION)
def test_a_party_in_another_position_is_never_supported(position_corpus, question, needle):
    result = position_run(position_corpus, question, needle)
    assert result.status is not S, (question, clause(result), RT.quotes(result))
    c = clause(result)
    if result.findings and c["ask_shape"] == R.ASK_RECOGNISED:
        assert c["relation"] == R.RELATION_AMBIGUOUS, (question, c)


RIGHT_POSITION = [
    ("Does the VCRGS determine the funding ceiling for research grants?", "VCRGS determines the funding ceiling"),
    ("Is the principal investigator entitled to a petty cash advance?", "entitled to a petty cash advance"),
    ("Does the principal investigator submit the book acquisition application to the VCRGS?", "submits a book acquisition application"),
    ("Must the principal investigator deliver the purchased book to the VCRGS?", "deliver the purchased book to the VCRGS"),
    ("Are research groups hosted by colleges?", "may be hosted by colleges"),
    ("Do colleges provide the VCRGS with names of potential referees?", "Colleges provide the VCRGS"),
    ("Can faculty members apply for grants to host visiting researchers?", "Faculty members can apply for grants"),
    ("Are book prices determined by the VCRGS?", "determined by the VCRGS"),
    ("Must researchers report adverse events to the IRB?", "to the IRB"),
    ("Must researchers obtain written informed consent before the study?", "before their involvement in the study"),
    ("Are conferences attended at no cost to the University discretionary?", "at no cost to the University"),
    ("Am I allowed to spend 20% of my time on consultancy?", "up to 20% of their time"),
    ("How many days per week can faculty spend on consultancy?", "one working day per week"),
    ("Is net royalty income shared equally among all inventors?", "shared equally among all inventors"),
    ("Does UoS have royalty-free non-commercial use of materials developed by students?", "non-commercial use"),
    ("Does the University publish conference proceedings?", "The University publishes conference and symposium proceedings"),
    ("Do organizers submit requests for publishing proceedings through the college?", "Organizers submit requests"),
    ("Who approves the funding ceiling for research grants?", "VCRGS determines the funding ceiling"),
    ("Who recommends the funding ceiling for research grants?", "VCRGS determines the funding ceiling"),
    ("What percentage of net royalty revenue goes to the inventors?", "50% to the inventor(s)"),
    ("Is the TTO required to consult with the inventor?", "the TTO is required to consult"),
    ("Is uncompensated conference travel authorized based on its value to the University?", "Uncompensated conference travel"),
]


@pytest.mark.parametrize("question,needle", RIGHT_POSITION)
def test_a_party_in_the_asked_position_is_supported(position_corpus, question, needle):
    result = position_run(position_corpus, question, needle)
    assert result.status is S, (question, result.missing, RT.quotes(result))
    assert clause(result)["relation"] == R.RELATION_STRONG and any(needle in q for q in RT.quotes(result))


@pytest.mark.parametrize("question,subject,roles,generic,predicate,passive", [
    ("Does the VCRGS submit the book acquisition application?", (), {"vcrgs"}, False, "submit", False),
    ("Is my petty cash advance approved by the Dean?", ("petty", "cash", "advance"), set(), False, "approved", True),
    ("Can visiting researchers apply for grants?", (), {"visitor"}, False, "apply", False),
    ("Do researchers ensure regular audits of animal research?", (), set(), True, "ensure", False),
    ("Are conferences attended at no cost to the faculty member require the same approval process?",
     ("conferences", "attended", "cost", "faculty", "member"), set(), False, "require", False),
    ("What is the maximum number of dissertations I can supervise?", (), set(), True, "supervise", False),
    ("Can compensated conference travel be authorized based on its value to the University?", ("compensated", "conference", "travel"), set(), False, "authorized", True),
])
def test_the_ask_structure_reads_subject_predicate_and_voice(question, subject, roles, generic, predicate, passive):
    structure = R._ask_structure(R.cue_text(question), question.split()[0].lower() in ("does", "is", "can", "do", "are", "must"))
    assert structure.subject == subject and structure.subject_roles == frozenset(roles) and structure.generic is generic
    assert structure.predicate == predicate and structure.passive is passive


def test_a_value_ask_with_a_copula_binds_no_subject():
    assert R._ask_structure(R.cue_text("What is the maximum petty cash advance?"), False).predicate is None
    assert R._ask_structure(R.cue_text("How much is the per diem for conference travel?"), False).subject == ()
    assert R._ask_structure(R.cue_text("Who must approve the annual progress report?"), False).predicate is None


@pytest.mark.parametrize("text,stripped", [
    ("16.3 Animal Use in Research All research involving animals must be approved by the IACUC.", "All research involving animals must be approved by the IACUC."),
    ("Journals in SCOPUS database are classified as A if their SJR metric ranks in the top 20%.", "Journals in SCOPUS database are classified as A if their SJR metric ranks in the top 20%."),
    ("The Vice Chancellor for Academic Affairs approves travel requests and interprets the policy.",
     "The Vice Chancellor for Academic Affairs approves travel requests and interprets the policy."),
    ("Visiting Researcher Grants Faculty members can apply for grants to host visiting researchers.", "Faculty members can apply for grants to host visiting researchers."),
])
def test_a_glued_heading_is_stripped_but_a_name_is_not(text, stripped):
    assert R._strip_heading(text) == stripped


def test_a_negated_occurrence_states_nothing_unless_the_question_negates_it_too():
    assert not R._stated_positively("UoS has a royalty-free non-commercial use of any copyrightable materials.", "commercial")
    assert R._stated_positively("UoS has a royalty-free non-commercial use of any copyrightable materials.", "commercial", negated=("commercial",))
    assert not R._stated_positively("Conferences attended at no cost to the University are discretionary.", "cost")
    assert R._stated_positively("Researchers must obtain approval prior to recruiting participants.", "before")   # a temporal synonym
    assert not R._class_stated_positively("pay", "Even in cases of non-compensated consultancy, prior approval is required.")
    assert R._proposition_words("Can the principal investigator extend the advance without the project budget?").count("without") == 1
    assert "30" in R._proposition_words("Am I allowed to spend 30% of my time on consultancy?")


@pytest.mark.parametrize("period,statement,ok", [
    (("week", ""), "Faculty members are permitted to spend one working day per week on consultancy.", True),
    (("month", ""), "Faculty members are permitted to spend one working day per week on consultancy.", False),
    (("year", "two"), "Faculty are allowed to attend one local conference per year.", False),
    (("year", ""), "Faculty are allowed to attend one local conference annually.", True),
    (None, "Anything at all.", True),
])
def test_an_asked_period_is_the_stated_period(period, statement, ok):
    assert R._period_ok(period, statement) is ok
    assert R._ask_period("How many conferences can I attend in two years?") == ("year", "two")
    assert R._ask_period("How much per diem is paid for conference travel?") is None


def test_arguments_are_bound_with_their_preposition_for_a_yes_no_ask():
    statement = "Researchers must monitor the progress of the study and promptly report any adverse events to the IRB."
    assert R._arguments_bound((("to", ("irb",)),), statement, True)
    assert not R._arguments_bound((("to", ("iacuc",)),), statement, True)
    assert not R._arguments_bound((("to", ("participants",)),), "Researchers must disclose conflicts of interest that affect the participants.", True)
    assert R._arguments_bound((("for", ("non", "compensated", "consultancy")),), "Even in cases of non-compensated consultancy, prior approval is required.", True)
    assert not R._arguments_bound((("for", ("students",)),), "Internal funding includes exchange student researchers grants, and grants for graduate research assistants.", True)
    assert R._arguments_bound((("for", ("conference",)),), "Per diem rates for a maximum of four days, depending on the conference location.", False)


def test_a_rule_conditioned_on_another_role_answers_nothing():
    statement = ("The academic member must obtain pre-approval from the College Dean, and in the event the Dean is planning to engage in "
                 "independent professional consultancy, a written approval from the Vice Chancellor for Academic Affairs is required.")
    assert not R._conditioned_role_ok(frozenset({"vcaa"}), statement)
    assert R._conditioned_role_ok(frozenset({"vcaa", "dean"}), statement)
    assert R._conditioned_role_ok(frozenset(), "In case of a dispute arising from the application of this policy, the VCRGS shall appoint a moderator.")
    assert R._conditioned_role_ok(frozenset(), "The College Dean may approve the coverage if the researcher is involved as a principal investigator.")


@pytest.mark.parametrize("text,word,voice", [
    ("The IACUC ensures regular audits of animal research activities.", "ensures", "active"),
    ("All payment procedures must be approved by the Dean.", "approved", "passive"),
    ("Faculty members on sabbatical leave are not allowed to combine teaching with grant applications.", "allowed", "passive"),
    ("Prior approval is required.", "approval", "nominal"),
    ("30% of research grants deducted for the University's account.", "deducted", "modifier"),
])
def test_the_voice_of_a_statement_occurrence(text, word, voice):
    assert R._occurrence_voice(text, occurrence(text, word)) == voice


def test_a_nominal_stated_of_a_clause_binds_its_subject_or_an_object():
    clause = "The University Research Board may approve partial or full sabbatical or financial awards for coordinators of research groups"
    assert R._clause_core_names(clause, "awards") and not R._clause_core_names(clause, "coordinators")
    assert R._clause_core_names("The VCRGS determines the funding ceiling for research grants based on the Board's recommendation", "ceiling")
    assert R._clause_core_names("The principal investigator is entitled to a petty cash advance of up to AED 5000", "advance")
    assert R._clause_core_names("Collaborative research agreements with external organizations", "agreements")
    assert not R._clause_core_names("Research groups serve as the nucleus for potential future research centers and units", "centers")


def test_a_percentage_of_another_base_is_not_an_item_of_a_share():
    unit = ("3. After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), and 50% "
            "to the University. The University's share, in turn, shall be distributed as follows: a. 40% to the VCRGS to support TTO and research "
            "activities at the University.")
    item = "40% to the VCRGS to support TTO and research activities at the University."
    assert R._percent_base("What percentage of total licensing revenue does the VCRGS receive?") == ("total", "licensing", "revenue")
    assert not R._share_statement_ok(None, item, unit, ("total", "licensing", "revenue"))
    assert R._share_statement_ok(None, item, unit, ("university", "share"))
    first = "After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), and 50% to the University."
    assert R._share_statement_ok(None, first, unit, ("net", "royalty", "revenue"))
