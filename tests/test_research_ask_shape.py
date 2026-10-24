"""Research & Innovation Specialist: the completeness root rule (Milestone 4, Step 4.5A-R).

Before this step a clause was complete as soon as every RECOGNISED requested detail was covered, so an ask the
detail families did not recognise requested nothing and topic relevance alone made it ``supported`` (fail-open).
Each substantive segment is now assessed from its question structure:

    general              an overview; topic evidence may answer it
    specific_recognised  a specific ask whose detail family answers its interrogative; the answer, the unmapped words
                         of the ask, the yes/no subject, the approval act and the purpose must be stated together
    specific_unknown     any other specific ask: never ``supported``; at most ``partial``

These tests are about the RULE, not about phrases: unknown asks are exercised with every research chunk ranked first,
overview and recognised asks with the stating unit ranked first, on pages whose lines are the real handbook wording
(tests/test_research_specialist.py and tests/test_research_corrections.py). No model, no LLM, no network.
"""
from __future__ import annotations

import pytest

import test_research_corrections as RC
import test_research_specialist as RT
from test_research_corrections import corpora as corr_corpora  # noqa: F401  (module fixture)
from test_research_corrections import corpora2, registry  # noqa: F401  (module fixtures)
from test_research_specialist import corpora, registry_and_map  # noqa: F401  (module fixtures)
from handbook_bot.agents import coordinator
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import domain_cues
from handbook_bot.agents.specialists import research as R
from handbook_bot.agents.specialists import teaching as T
from handbook_bot.agents.specialists.contracts import FindingStatus, SpecialistId

S, P, N, O = FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND, FindingStatus.OUT_OF_SCOPE


def assess(question):
    return R.assess_ask(question, R.requested_details(question), question)


def everything_first(corpus):
    """Every research chunk ranked first: only the gates decide."""
    return RT.make_specialist(corpus, RT.ScriptedReranker([], default=9.0))


def needle_first(corpus, needle, drop_default=-5.0):
    return RT.make_specialist(corpus, RT.ScriptedReranker([(needle, 9.0)], default=drop_default))


def outcome(result):
    return result.metadata["clauses"][0]


# ---------------------------------------------------------------------------
# The root cause: an empty recognised-detail set never means "complete"
# ---------------------------------------------------------------------------
SPECIFIC_UNKNOWN = [
    "What is the timeframe for submitting research grant applications?", "What is the submission window for research grants?",
    "When exactly do research grant applications close?", "How late can I submit a research grant application?",
    "What is the timeline for research grant applications?", "During which period can I apply for research grants?",
    "When am I supposed to submit the final research report?", "What is the schedule for research grant submissions?",
    "Who notarizes a research proposal?", "What colour is the research grant form?", "Who stamps a research proposal?",
    "What font must a research proposal use?", "Who negotiates licensing agreements?", "What happens to my patent after I leave the university?",
    "Do I lose my research grant when I leave the university?", "What option do I select to request conference funding?",
    "How do I log in to submit a research proposal?", "What equipment do the central laboratories provide for researchers?",
    "Which rank do I need for promotion to professor based on research?", "Why was my research grant rejected?",
    "What cut of licensing revenue do inventors receive?", "How is the research grant ceiling calculated?", "Who pays for my conference visa?",
    "How are research groups dissolved?",
]


@pytest.mark.parametrize("question", SPECIFIC_UNKNOWN)
def test_an_unrecognised_specific_ask_is_classified_specific_unknown(question):
    assessment = assess(question)
    assert assessment.shape == R.ASK_UNKNOWN, (question, assessment)
    assert any(d.family == "ask_unknown" for d in assessment.details)
    assert all(not d.check("The University Research Board approves research grants.") for d in assessment.details if d.family == "ask_unknown")


@pytest.mark.parametrize("question", SPECIFIC_UNKNOWN)
def test_an_unrecognised_specific_ask_is_never_supported_by_topic_relevance(corpora, question):
    corpus = corpora("all")
    result = RT.run(everything_first(corpus), question)
    assert result.status is not S, (question, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    if result.status is P:
        assert any("specific ask of clause 0 is not recognised" in m for m in result.missing) or outcome(result)["ask_shape"] != R.ASK_UNKNOWN
    RT.assert_traceable(result, corpus)


def test_the_unknown_ask_detail_is_never_covered_and_empty_details_are_not_a_blanket_partial():
    # the safety guard itself
    assert R._ask_unknown_check("anything at all") is False
    # a general question with no requested detail keeps an empty detail list and stays answerable
    assessment = assess("What are research groups?")
    assert assessment.shape == R.ASK_GENERAL and not any(d.family.startswith("ask") for d in assessment.details)
    assert assess("Research grants.").shape == R.ASK_GENERAL


# ---------------------------------------------------------------------------
# General overview questions stay supported
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
    ("grants", "What does the Research Funding Department do?", "pre- and post-award management"),
    ("ip", "Give me an overview of the IP policy.", "shall own all Intellectual Property Rights"),
    ("grants", "What are research institutes?", "three research institutes"),
    ("grants", "What are the functions of the research committees?", "Research committees review research grant applications"),
    ("consultancy", "Tell me about faculty consultancy.", "recognizes the value of faculty consultancy"),
    ("ip", "What is commercialization at UoS?", "Commercialization is a priority"),
    ("grants", "Tell me about the research institutes.", "three research institutes"),
    ("consultancy", "What is the Institutional Review Board?", "An independent committee responsible for reviewing"),
]


@pytest.mark.parametrize("group,question,needle", GENERAL)
def test_a_general_overview_question_is_classified_general(group, question, needle):
    assert assess(question).shape == R.ASK_GENERAL, (question, assess(question))


@pytest.mark.parametrize("group,question,needle", GENERAL)
def test_a_general_overview_question_is_still_supported_by_the_describing_unit(corpora, group, question, needle):
    corpus = corpora(group)
    result = RT.supported_with(needle_first(corpus, needle), question, needle)
    assert outcome(result)["ask_shape"] == R.ASK_GENERAL
    RT.assert_traceable(result, corpus)


# ---------------------------------------------------------------------------
# Recognised specific asks stay supported by the unit that states them
# ---------------------------------------------------------------------------
RECOGNISED = [
    ("conference", "What is the per diem for international conferences?", "Per diem rates between AED 1000-1500", "amount or cost"),
    ("conference", "Who approves conference travel requests?", "approves travel requests", "approving authority"),
    ("conference", "How many conferences can a faculty member attend per year?", "one local and one external university-supported conference per year",
     "number or amount"),
    ("conference", "What is the maximum registration fee for a conference?", "registration fees up to $750", "maximum or limit"),
    ("conference", "When must conference requests be submitted?", "at least one month before the conference date", "deadline or date"),
    ("ip", "What percentage of net royalty revenue goes to the inventor?", "50% to the inventor", "percentage or share"),
    ("ip", "Who owns intellectual property created by faculty?", "shall own all Intellectual Property Rights", "ownership or rights holder"),
    ("ip", "When must the TTO decide whether to file a patent application?", "no later than 90 calendar days", "deadline or date"),
    ("ip", "Who decides whether to file a patent application?", "The TTO shall decide whether to file a patent application", "approving authority"),
    ("ip", "Who approves transferring tangible research property for commercial purposes?", "For commercial purposes, transferring tangible research property",
     "approving authority"),
    ("grants", "What is the maximum research advance?", "may apply for a research advance of up to 10%", "maximum or limit"),
    ("grants", "Who determines the funding ceiling for research grants?", "VCRGS determines the funding ceiling", "approving authority"),
    ("grants", "What is the funding ceiling for a seed project?", "funding ceiling for each seed project is AED 5000", "amount or cost"),
    ("grants", "Who approves collaborative research agreements?", "require the approval of the VCRGS", "approving authority"),
    ("grants", "What percentage of the project budget can be spent on research assistants?", "Costs for research assistants are capped", "percentage or share"),
    ("consultancy", "How many days per week can faculty spend on consultancy?", "may not exceed one day per week", "number or amount"),
    ("consultancy", "Is prior approval required for non-compensated consultancy?", "prior approval is required", "an explicit requirement statement about the subject"),
    ("consultancy", "Do I need IRB approval before recruiting human participants?", "must obtain IRB approval before recruiting",
     "an explicit requirement statement about the subject"),
    ("promotion", "Which journals are classified as class A?", "SCOPUS", "publication type"),
    ("promotion", "What research counts for promotion?", "A. Research published", "the Faculty Services & HR subject asked about (promotion)"),
    ("grants", "How long is the term of office for the chief editor?", "two years, renewable once", "duration"),
]


@pytest.mark.parametrize("group,question,needle,label", RECOGNISED)
def test_a_recognised_specific_ask_is_classified_recognised(group, question, needle, label):
    assert assess(question).shape == R.ASK_RECOGNISED, (question, assess(question))


@pytest.mark.parametrize("group,question,needle,label", RECOGNISED)
def test_a_recognised_specific_ask_is_supported_by_the_stating_unit(corpora, group, question, needle, label):
    corpus = corpora(group)
    result = RT.supported_with(needle_first(corpus, needle), question, needle)
    assert any(needle in f.evidence_quote and label in f.metadata["details_covered"] for f in result.findings), \
        [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings]
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("question,label", [
    ("Who is the chief editor?", "the named person"),
    ("Who is the editor-in-chief of the University journal?", "the named person"),
])
def test_a_recognised_ask_whose_answer_is_absent_reports_the_detail_not_an_unknown_ask(corpora2, question, label):
    # recognised (a named person is asked for) is distinct from not understood: the missing line names the detail
    assert assess(question).shape == R.ASK_RECOGNISED
    sp, corpus = RC.ranked2(corpora2, [("nominated by the Editor", 9.0)], default=0.5)
    result = RT.run(sp, question)
    assert result.status is P and any(m.startswith(label) for m in result.missing), result.missing
    assert not any("not recognised" in m for m in result.missing)


# ---------------------------------------------------------------------------
# Wrong object, wrong action, wrong beneficiary or purpose: a recognised ask is grounded in its answer
# ---------------------------------------------------------------------------
WRONG_ACTION_RT = [
    # (corpus group, question, wrong evidence ranked first)
    ("grants", "Who approves sabbatical awards for faculty members?", "i. The University Research Board may approve partial or full sabbatical"),
    ("ip", "Who approves taking samples of tangible research property when leaving the university?", "3. For commercial purposes, transferring"),
    ("ip", "Who approves transferring tangible research property for research purposes?", "3. For commercial purposes, transferring"),
    ("ip", "Who approves transferring tangible research property to another institution for research?", "3. For commercial purposes, transferring"),
    ("ip", "Who decides how a patent is commercialized?", "2. The TTO shall decide whether to file a patent application"),
    ("ip", "Who grants licenses for university intellectual property?", "1. Commercialization is a priority"),
    ("grants", "Who approves research grant applications?", "Research committees review research grant applications"),
    ("grants", "Does the Dean approve research advances?", "21. The principal investigator may apply for a research advance"),
    ("grants", "Can a co-investigator apply for a research advance?", "21. The principal investigator may apply for a research advance"),
    ("conference", "Who approves the conference registration fee?", "The Vice Chancellor for Academic Affairs approves travel requests"),
    # the asked object is a research topic the object table does not list: its head noun must be stated with the approval
    ("grants", "Who approves the research institutes?", "Collaborative research agreements with external organizations require the approval"),
]
WRONG_ACTION_R2 = [
    # (question, wrong evidence ranked first, lines removed)
    ("Who approves extending the petty cash advance to another period?", "20. The principal investigator is entitled", ()),
    ("Who approves field trips for my research project?", "19. The research grant for a project typically", ()),
    ("What percentage of royalties goes to the TTO?", "a. 40% to the VCRGS to support TTO", ()),
    ("Who signs agreements to transfer university patents?", "6. The TTO, in consultation", ("7. The authority to sign",)),
    ("What percentage of revenue from my consulting work goes to the University?", "3. After recovering IP protection expenses", ()),
    ("What percentage of research overhead goes to the college?", "b. 5% to the Inventors", ()),
]


@pytest.mark.parametrize("group,question,wrong", WRONG_ACTION_RT)
def test_a_recognised_ask_is_never_supported_by_another_action_object_beneficiary_or_purpose(corpora, group, question, wrong):
    corpus = corpora(group)
    result = RT.run(needle_first(corpus, wrong, drop_default=0.5), question)
    assert result.status is not S, (question, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("question,wrong,drop", WRONG_ACTION_R2)
def test_a_recognised_ask_is_grounded_in_the_statement_that_answers_it(corpora2, question, wrong, drop):
    sp, corpus = RC.ranked2(corpora2, [(wrong, 9.0)], drop=drop, default=0.5)
    result = RT.run(sp, question)
    assert result.status is not S, (question, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    RT.assert_traceable(result, corpus)


def test_the_international_per_diem_question_never_takes_the_local_rate(corr_corpora):
    sp, corpus = RC.ranked(corr_corpora, "conference", [("Local conferences (e.g., Al Ain", 9.0)], drop=("Per diem rates between",), default=0.5)
    result = RT.run(sp, "How much per diem do I get when I travel to Japan for a conference?")
    assert result.status is not S


@pytest.mark.parametrize("question,words", [
    ("Who approves extending the petty cash advance to another period?", ("action extend", "action approve")),
    ("Who approves field trips for my research project?", ("trip",)),
    ("How much per diem do I get when I travel to Japan for a conference?", ("japan",)),
])
def test_the_grounding_detail_names_the_unmapped_specifics(question, words):
    grounding = next(d for d in assess(question).details if d.family == "ask_grounding")
    assert all(w in grounding.label for w in words), grounding.label


# ---------------------------------------------------------------------------
# Nonsense and unsupported specific properties
# ---------------------------------------------------------------------------
NONSENSE = [
    "Is the seed grant refundable?", "Is conference per diem taxable?", "Who notarizes a research proposal?", "What colour is the research grant form?",
    "Can a research grant pay for a bicycle?", "Where do I fax a patent request?", "Is a research grant transferable to my spouse?",
    "Who stamps a research proposal?", "What password is used for the grant portal?", "Which button approves the research advance?",
    "How many parking spaces does the TTO have?", "Is the IRB open on Fridays?", "Can I pay the conference registration fee with bitcoin?",
    "Who laminates the patent certificate?", "What font must a research proposal use?", "Does the research advance come with a free laptop?",
]


@pytest.mark.parametrize("question", NONSENSE)
def test_a_nonsense_or_unsupported_specific_property_is_never_supported(corpora, corpora2, question):
    for corpus in (corpora("all"), corpora2(())):
        result = RT.run(everything_first(corpus), question)
        assert result.status is not S, (question, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
        RT.assert_traceable(result, corpus)


# ---------------------------------------------------------------------------
# The five confirmed repros that still partly reproduced after Step 4.5A
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,inherited", [
    ("By when must I submit final grades and my conference travel request?", "What is the deadline for my conference travel request?"),
    ("What is the deadline for exam grades and our research grant proposal?", "What is the deadline for our research grant proposal?"),
    ("When must I submit final grades and what about my conference travel request?", "When must my conference travel request be submitted?"),
])
def test_tim03_an_elliptical_research_clause_inherits_the_question_head_ask(registry, question, inherited):
    result = D.run_question(question, registry, executable=(SpecialistId.TEACHING, SpecialistId.RESEARCH))
    research = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
    clause = research.findings.metadata["clauses"][0]
    assert clause["inherited_ask"] == inherited and "deadline or date" in clause["requested_details"]
    assert research.findings.status is not S or "deadline or date" not in clause["uncovered_details"]


def test_tim03_a_bare_phrase_of_a_specific_question_without_a_carried_ask_is_unknown():
    assessment = R.assess_ask("my research grant proposal", R.requested_details("my research grant proposal"),
                              "Who stamps final grades and my research grant proposal?")
    assert assessment.shape == R.ASK_UNKNOWN and assessment.kinds == ("elliptical",)


@pytest.mark.parametrize("question", ["What is the TTO's role and contact details?", "What does the IRB review, and its email?",
                                      "What is the per diem for conferences and the email of the research office?"])
def test_nav05_a_channel_clause_that_asks_a_subject_question_keeps_research(question):
    selected = coordinator.coordinate(question).selected_specialists
    assert SpecialistId.RESEARCH in selected and SpecialistId.INSTITUTIONAL in selected


def test_nav05_research_answers_the_substance_and_never_the_contact(corpora):
    corpus = corpora("ip")
    result = RT.run(needle_first(corpus, "to manage all intellectual property assets"), "What is the TTO's role and contact details?")
    assert result.status is P and any("contact or location details" in m for m in result.missing)
    assert result.handoff_requests and result.handoff_requests[0].requested_specialist is SpecialistId.INSTITUTIONAL


@pytest.mark.parametrize("clause,unit,expected", [
    # no body the specialist can name: a URL of something else never counts (NAV-07)
    ("What does the Research Funding Department do and how do I contact it?",
     "Data are deposited in the national repository at https://repository.example.ae for public access.", False),
    ("How do I contact the TTO?", "The Technology Transfer Office (TTO) can be reached at tto@example.ae.", True),
    ("How do I contact the TTO?", "The IRB can be reached at irb@example.ae.", False),
])
def test_nav07_contact_evidence_must_describe_a_named_body(clause, unit, expected):
    assert R._contact_check(clause)(unit) is expected


def test_saf03_a_promotion_procedure_is_never_supported_by_research_text_mentioning_promotion(corpora):
    corpus = corpora("promotion")
    result = RT.run(needle_first(corpus, "1. Journals in SCOPUS database", drop_default=0.5),
                    "What are the promotion procedures at the department level for research publications?")
    assert result.status is not S
    assert not any("department promotion committee reviews the dossier" in q for q in RT.quotes(result))   # excluded 7.8


# ---------------------------------------------------------------------------
# Compound questions: every intent is a task or a handoff, and no half answers another
# ---------------------------------------------------------------------------
COMPOUND = [
    ("By when must I submit final grades and my conference travel request?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
    ("What is the deadline for exam grades and our research grant proposal?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
    ("What is the TTO's role and contact details?", {SpecialistId.RESEARCH, SpecialistId.INSTITUTIONAL}),
    ("What does the IRB review, and its email?", {SpecialistId.RESEARCH, SpecialistId.INSTITUTIONAL}),
    ("Is the seed grant refundable and how many office hours must I hold?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
    ("What is the timeframe for research grant applications and my teaching load?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
    ("Who owns software I develop and what is my teaching load?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
    ("Hi there, what is the petty cash limit and my teaching load?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
    ("What is the per diem for international conferences and what is my teaching load?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
    ("Who stamps research proposals and how many office hours must I hold?", {SpecialistId.RESEARCH, SpecialistId.TEACHING}),
]


@pytest.mark.parametrize("question,domains", COMPOUND)
def test_a_compound_question_keeps_every_intent(registry, question, domains):
    result = D.run_question(question, registry, executable=(SpecialistId.TEACHING, SpecialistId.RESEARCH))
    tasks = {r.specialist_id for r in result.runs}
    handed = {h.requested_specialist for h in result.handoffs}
    assert domains <= tasks | handed, (question, tasks, handed)
    research = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
    assert research.findings.status is not O, (question, research.findings.missing)
    for clause in research.findings.metadata["clauses"]:
        assert "greeting" not in clause["clause"].lower() and not clause["clause"].lower().startswith("hi")


@pytest.mark.parametrize("question", ["Is the seed grant refundable and how many office hours must I hold?",
                                      "What is the timeframe for research grant applications and my teaching load?",
                                      "Who stamps research proposals and how many office hours must I hold?"])
def test_the_research_half_of_a_compound_with_an_unknown_ask_is_never_supported(registry, question):
    result = D.run_question(question, registry, executable=(SpecialistId.TEACHING, SpecialistId.RESEARCH))
    research = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
    assert research.findings.status is not S


def test_teaching_hands_a_research_intent_off_when_research_is_capped_out():
    question = "What is my teaching load and the editorial board term, how many days of annual leave do I get, and where is the HR office?"
    assert SpecialistId.RESEARCH in T.other_domains("What is my teaching load and the editorial board term")
    decision = coordinator.coordinate(question)
    assert SpecialistId.RESEARCH not in decision.selected_specialists             # the specialist cap of three
    teaching_task = next(t for t in decision.subtasks if t.specialist_id is SpecialistId.TEACHING)
    assert any("editorial board" in c for c in teaching_task.context["focus_clauses"])


# ---------------------------------------------------------------------------
# One Research ownership lexicon for the Coordinator, Teaching and Research
# ---------------------------------------------------------------------------
def test_the_coordinator_research_cues_are_the_shared_lexicon():
    strong = [p.pattern for sid, weight, p in coordinator._DOMAIN_CUES if sid is SpecialistId.RESEARCH and weight == coordinator.STRONG]
    assert strong == list(domain_cues.RESEARCH_OWNERSHIP_PATTERNS)
    assert coordinator._WLAM_CATEGORY is domain_cues.WLAM_CATEGORY and coordinator._DEVELOPMENT_FUNDING is domain_cues.DEVELOPMENT_FUNDING


@pytest.mark.parametrize("question", [
    "Who owns software I develop at the university?", "Who owns the data I collect?", "Who owns creative works I produce?",
    "Can I work as a consultant for a private company?", "May I act as a consultant to industry?", "What is the editorial board term?",
    "What are the page charges?", "What does the disclosure evaluation involve?", "Can I take TRP samples when I leave?",
    "Does the university fund attendance at a scientific symposium?",
])
def test_a_phrase_the_coordinator_gives_to_research_is_accepted_by_research_and_handed_off_by_teaching(question):
    assert domain_cues.research_owned(question)
    assert SpecialistId.RESEARCH in coordinator.coordinate(question).selected_specialists
    assert R.classify_clause(question)[0] is True, question
    assert SpecialistId.RESEARCH in T.other_domains(question)


@pytest.mark.parametrize("question", ["What is the teaching load for research-intensive faculty?",
                                      "What funding is available for professional development workshops?",
                                      "Can I book the conference room?"])
def test_the_shared_lexicon_keeps_the_research_view_exclusions(question):
    assert not domain_cues.research_owned(question)
