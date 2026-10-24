"""Research & Innovation Specialist: bounded corrections after the isolation QA (Milestone 4, Step 4.5A).

Every test reproduces a Step 4.5 finding on pages whose lines are the REAL handbook wording (copied from the audited
pages; headings printed as lines), chunked by the project's chunker and annotated with the real registry record and
section map. Wrong-object evidence is ranked first by a scripted reranker and, where the question needs it, the right
evidence is removed, so the gates alone decide. No model, no LLM, no network.

M4-F6  navigation and contact clauses are Institutional's; a shared clause is partial with a handoff
M4-F7  the modal "may" is never a date; a deadline needs an end point, a window start is not one
M4-F8  a requested value, approver, eligibility or timing must belong to the asked object (and conference locality)
M4-F9  topic relevance never makes a yes/no, ownership, permission, lead-time or focus question supported
M4-F10 a compound Research + Teaching question keeps both intents (Coordinator cues, Teaching handoff cues)
M4-F11 Coordinator reachability for Research-owned phrasings, with over-routing controls
M4-F12 Research vocabulary and classifier morphology
M4-F13 detail false negatives and required-documents list completeness
M4-F14 reranker output length contract
Round 2 (after the adversarial verification of the first round): TIM timing, BIND binding, TOP topical support,
NAV navigation, RT routing and SAF safety residuals, each named at its test.
"""
from __future__ import annotations

import pytest

import test_research_integration as RI
import test_research_specialist as RT
from test_research_specialist import registry_and_map  # noqa: F401  (module fixture)
from handbook_bot.agents import coordinator
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import research as R
from handbook_bot.agents.specialists import teaching as T
from handbook_bot.agents.specialists.contracts import FindingStatus, SpecialistId
from handbook_bot.agents.specialists.registry import SpecialistRegistry

page = RT.page
S, P, N, O, E = FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND, FindingStatus.OUT_OF_SCOPE, FindingStatus.ERROR

# ---------------------------------------------------------------------------
# Real handbook wording (one printed line or one joined item per entry)
# ---------------------------------------------------------------------------
PAGES = [
    page(81,
         "5. Theses/Dissertations Supervision: To ensure effective mentorship and balanced supervision of theses/dissertations:",
         "• A faculty member is allowed to supervise a maximum of four theses/dissertations simultaneously, exception may be granted by the College of Graduate Studies.",
         "• Faculty members are encouraged to advise a maximum of four students, exception may be granted by the College of Graduate Studies."),
    page(117,
         "5.4 Participation in Conferences",
         "The University of Sharjah (UoS) recognizes the immense value of faculty's participation in conferences and seminars organized by scientific and specialized organizations, both at local and international levels.",
         "• Faculty members are allowed to attend a maximum of one international and one local university-supported conference per year, provided it does not conflict with their primary commitments.",
         "• Faculty teaching summer courses are not allowed to travel for conferences during the summer term.",
         "• The Vice Chancellor for Academic Affairs approves travel requests and interprets the policy."),
    page(118,
         "• Participation in conferences is not permitted during the first and last weeks of teaching or during examination periods.",
         "• University-supported participation without a paper may be permitted with the Chancellor's approval in cases such as:",
         "o Serving as a main member of the organizing committee.",
         "o Being a keynote speaker.",
         "• Participants must submit a report to their Department Chair and Dean's Office upon return.",
         "• Faculty are entitled to participate in one local and one external university-supported conference annually, budget permitting."),
    page(119,
         "• Per diem rates between AED 1000-1500 for a maximum of four days, depending on the conference location.",
         "• Conference registration fees up to $750.",
         "• Local conferences (e.g., Al Ain, Abu Dhabi) are covered for transportation and a per diem up to AED 700 for a maximum of three days.",
         "5.4.5 Required Documentation",
         "Requests must be submitted at least one month before the conference date, including:",
         "o Approval from the concerned college.",
         "o A copy of the full paper or abstract.",
         "o Evidence that conference proceedings will be published in a Scopus-indexed outlet or commitment to submit to a Scopus journal within six months.",
         "o A copy of the conference flier with fees stated.",
         "5.5 Policy on Offering Courses in the Summer Semester",
         "Courses offered during the summer semester are primarily limited to first and second-year classes."),
    page(154,
         "8.2 Research Support Units and Departments",
         "8.2.1 Research Funding Department",
         "The Research Funding Department (RFD) is responsible for pre- and post-awards administration of research project's funds."),
    page(155,
         "8.2.3 Technology Transfer Office",
         "The Technology Transfer Office (TTO) is entrusted with managing all intellectual property matters related to research activities and outputs at UoS."),
    page(163,
         "8.9 Scientific Publishing at the University of Sharjah Journal",
         "Additionally, an Editorial Secretary from among the faculty members, nominated by the Editor-In-Chief and appointed with the Chancellor's approval, supports the journal's operations.",
         "The terms of office for the Chief Editor, Editorial Board, and Editorial Secretary are two years, renewable."),
    page(164,
         "11. Announcements for seed projects are made once a year during the period specified by the VCRGS. Researchers may apply for this funding type only once, during their first year at the University."),
    page(165,
         "14. Faculty members may not act as principal investigators for two projects simultaneously, except for seed projects.",
         "15. The principal investigator of a project must hold a Ph.D. degree or an equivalent degree recognized by the University in their specialty. Additionally, they must be full-time employees of the University and not visitors.",
         "18. The research grant for a project covers various aspects of the research, including:",
         "a. Costs for research assistants, capped at a maximum of 30% of the project's budget, which may be raised to 50% in special and justified cases by the Board."),
    page(166,
         "20. The principal investigator is entitled to a petty cash advance of up to AED 5000, subject to approval by the VCRGS. The advance may be closed at the end of the academic year or carried forward to the next academic year.",
         "21. The principal investigator may apply for a research advance of up to 10% of the basic research project grant. The Research Purchasing Committee under the VCRGS reviews the application and makes recommendations, subject to approval by the VCRGS."),
    page(167,
         "28. Funding for Student Research Projects",
         "The University allocates a yearly budget to support distinguished research projects of undergraduate and graduate students."),
    page(168,
         "d. The funding ceiling for each project is AED 5000.",
         "29. Reporting Requirements for Research Projects",
         "The principal investigator must provide the VCRGS with an annual progress report for the project, approved by the Dean.",
         "The Dean may approve the project's closure if its objectives have been achieved. However, if the project period ends, and the final report is not submitted or justified, the Dean may refer the issue to the Board, which may close the project and reclaim any spent funds from the investigator."),
    page(169,
         "a. Groups of researchers (at least three) can register a research group under the VCRG with an appropriate name."),
    page(170,
         "37. Hosting Research Groups",
         "Research groups report to the VCRGS but may be hosted by colleges, which should publicize them through various media, including print and electronic formats."),
    page(175,
         "Chapter 9: Intellectual Properties Management",
         "The University of Sharjah has established the Technology Transfer Office (TTO) to manage all intellectual property assets derived from the research activities and outcomes at the University."),
    page(176,
         "9.1 Intellectual Property Ownership Rights",
         "Unless stated otherwise in this policy, the University of Sharjah shall own all Intellectual Property Rights of inventions or creative works produced by faculty, staff, and students at or under the auspices of the University while making substantial use of its resources.",
         "9.1.2 Intellectual Property Created by Students",
         "a. Students who create intellectual property as a result of their participation in study programs at UoS, such as PhD dissertations, research reports, or graduation projects, shall retain ownership rights unless the creation involves substantial use of UoS property."),
    page(177,
         "a. Establish procedures for the exploitation of Intellectual Property created by UoS Inventors."),
    page(179,
         "e. The decision to seek patent protection is made by the TTO based on the disclosure data submitted by the inventor(s) after objective evaluation of the invention's originality.",
         "h. The TTO shall inform the Inventor(s) of the outcome of its review regarding rights and obligations as soon as practical but no later than 90 calendar days from receipt of a complete Intellectual Property disclosure."),
    page(180,
         "12.3 Copyright Protection Policy",
         "1. This Policy also covers copyright protection for books, computer software, artworks, and other copyrightable materials.",
         "2. Copyrighted materials remain the property of their creators (individually owned) unless otherwise agreed.",
         "9.7 Tangible Research Property",
         "2. Faculty or students seeking to transfer TRP to another institution for research purposes must complete and submit a Materials Transfer Agreement Form (MTA) available from the TTO."),
    page(181,
         "9.9 Distribution of Licensing Revenues",
         "3. After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), and 50% to the University.",
         "The University's share, in turn, shall be distributed as follows: a. 40% to the VCRGS to support TTO and research activities at the University. b. 5% to the Inventors’ College(s), if applicable."),
    page(259,
         "16.1 Faculty Professional Consultancy",
         "The University of Sharjah recognizes the value of faculty consultancy as a means to contribute to the profession and the community."),
    page(260,
         "16.1.7 Time Allocation for Consultancy: Full-time faculty members are permitted to spend up to 20% of their time or an equivalent of one working day per week on professional consultancy activities.",
         "16.1.8 Use of University Name: The use of the University's name in relation to any professional consultancy activity is not permitted unless explicitly approved in writing by the Vice Chancellor for Academic Affairs.",
         "16.1.9 Independent Consultancy: The academic member must obtain pre-approval from the College Dean."),
]
GROUPS = {
    "all": None,
    "supervision": (81,),
    "conference": (117, 118, 119),
    "grants": (81, 154, 155, 163, 164, 165, 166, 167, 168, 169, 170),
    "ip": (155, 175, 176, 177, 179, 180, 181),
    "consultancy": (259, 260),
    "units": (154, 155),
}


@pytest.fixture(scope="module")
def corpora(registry_and_map):  # noqa: F811
    cache = {}

    def get(group, drop=()):
        key = (group, tuple(drop))
        if key not in cache:
            pages = [p for p in PAGES if GROUPS[group] is None or p["page"] in GROUPS[group]]
            pages = [page(p["page"], *[l for l in p["rows"] if not any(n in l for n in drop)]) for p in pages]
            cache[key] = RT.corpus_from(registry_and_map, pages)
        return cache[key]

    return get


def ranked(corpora, group, rules=(), drop=(), default=-5.0):
    """A specialist whose reranker puts ``rules`` first; ``drop`` removes the lines that would answer correctly."""
    corpus = corpora(group, drop)
    reranker = RT.ScriptedReranker(rules, default=default) if rules or default != -5.0 else RT.RawOverlapReranker()
    return RT.make_specialist(corpus, reranker), corpus


def covering(result, label):
    return [f.evidence_quote for f in result.findings if label in f.metadata["details_covered"]]


# ---------------------------------------------------------------------------
# M4-F6 Research <-> Institutional boundary
# ---------------------------------------------------------------------------
PURE_NAVIGATION = [
    "How do I contact the IRB?", "Where do I submit my research grant application?", "Where are the research institutes located?",
    "What is the email address of the TTO?", "Where is the research office?", "Who is the contact person for research grants?",
    "What is the phone number of the Research Funding Department?", "Where can I find the IACUC?", "Which office handles research grants?",
    "Where is the Technology Transfer Office?", "Who should I contact about research ethics?", "Who manages the research portal?",
    "How can I reach the Research Funding Department?", "What is the location of the TTO office?",
]


@pytest.mark.parametrize("question", PURE_NAVIGATION)
def test_pure_navigation_is_out_of_scope_with_an_institutional_handoff(corpora, question):
    # every research chunk is ranked at 9.0: nothing may be quoted for a location or contact request
    sp, _corpus = ranked(corpora, "all", default=9.0)
    result = RT.run(sp, question)
    assert result.status is O and result.findings == [], (question, result.status, [f.evidence_quote for f in result.findings])
    assert result.requested_agents == [SpecialistId.INSTITUTIONAL]
    handoff = result.handoff_requests[0]
    assert handoff.metadata["kind"] == "misrouted" and handoff.task.context["focus_clauses"] == [question]
    assert handoff.task.specialist_id is SpecialistId.INSTITUTIONAL and handoff.task.requested_by == "research"
    assert R.navigation_intent(question) == (True, False)


@pytest.mark.parametrize("question,group,needle", [
    ("What does the Research Funding Department do and how do I contact it?", "grants", "Research Funding Department (RFD) is responsible"),
    ("Who handles research funding and what is their phone number?", "units", "Research Funding Department (RFD) is responsible"),
    ("Where is the Technology Transfer Office and what does it do?", "ip", "entrusted with managing all intellectual property"),
])
def test_shared_navigation_clause_answers_the_substance_and_stays_partial(corpora, question, group, needle):
    # the unit that states the substantive role is ranked first; the contact half can never be supported
    sp, corpus = ranked(corpora, group, [(needle, 9.0)], default=0.5)
    result = RT.run(sp, question)
    assert result.status is P and result.findings and any(needle in q for q in RT.quotes(result)), (result.status, RT.quotes(result))
    assert result.requested_agents[0] is SpecialistId.INSTITUTIONAL and result.handoff_requests[0].metadata["kind"] == "shared"
    assert any("contact or location details are not in the Research evidence" in m for m in result.missing)
    assert all("@" not in q and "+971" not in q for q in RT.quotes(result))       # no contact channel is ever produced
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("question", [
    "Which committee reviews research protocols?", "Who approves conference travel?", "Who handles research grants?",
    "Which body reviews research on animals?", "Who determines the funding ceiling for research grants?",
])
def test_a_policy_action_by_a_named_body_is_not_navigation(question):
    assert R.navigation_intent(question) == (False, True)
    research, other = R.classify_clause(question)
    assert research


def test_a_system_click_procedure_is_not_navigation():
    assert R.navigation_intent("Where do I click to submit a grant in the research portal?")[0] is False
    assert R.navigation_intent("Where do I find the button in MyUOS?", "myuos")[0] is False


@pytest.mark.parametrize("unit,expected", [
    ("Human Resources +971 6 5050023 +971 6 5585200", True), ("Contact research@sharjah.ac.ae for details.", True),
    ("The office is located in Building W8, Room 102.", True),
    ("The Research Funding Department (RFD) is responsible for pre- and post-awards administration.", False),
    ("The Technology Transfer Office (TTO) is entrusted with managing all intellectual property matters.", False),
])
def test_contact_evidence_needs_a_real_channel_or_location(unit, expected):
    assert R._contact_ok(unit) is expected


# ---------------------------------------------------------------------------
# M4-F7 (and M4-F2) timing: the modal "may"; deadline versus window
# ---------------------------------------------------------------------------
MODAL_MAY = [
    "Research groups report to the VCRGS but may be hosted by colleges, which should publicize them.",
    "The Dean may approve the project's closure if its objectives have been achieved.",
    "4. The VCRGS may allocate portions of the University's share of the net royalty income to any other relevant unit.",
    "Researchers may apply for this funding type only once.",
    "The University may establish additional guidelines addressing potential conflicts of interest in licensing.",
    "May the principal investigator extend the advance to another period?",
    "The advance may be raised to a maximum of 30% of the basic research project grant.",
]


@pytest.mark.parametrize("unit", MODAL_MAY)
def test_the_modal_may_is_never_a_date(unit):
    for clause in ("When is the deadline for the final research report?", "When are seed projects announced?", "When must the application be submitted?"):
        timing = next(d for d in R.requested_details(clause) if d.family == "deadline")
        assert timing.value(unit) is False, (clause, unit)


@pytest.mark.parametrize("unit,deadline,timing", [
    ("Applications close on 15 May 2026.", True, True),
    ("Submit the report by May 15.", True, True),
    ("Proposals are due by the end of May.", True, True),
    ("The proposal deadline is 1 May.", True, True),
    ("The call for proposals opens in May 2026.", False, True),
    ("Seed projects are announced at the beginning of each academic year.", False, True),
    ("Applications can be submitted from the beginning of the first semester until the end of April of every academic year.", True, True),
    ("Requests must be submitted at least one month before the conference date.", True, True),
    ("The TTO shall inform the inventors no later than 90 calendar days from receipt of a complete disclosure.", True, True),
    ("Participants must submit a report within two weeks of return.", True, True),
    ("As of Fall 2024, there are 62 active research groups at UoS.", False, False),
])
def test_calendar_may_and_end_points(unit, deadline, timing):
    end = next(d for d in R.requested_details("What is the deadline?") if d.family == "deadline")
    when = next(d for d in R.requested_details("When does it happen?") if d.family == "deadline")
    assert end.value(unit) is deadline and when.value(unit) is timing, unit


def test_modal_may_never_answers_the_final_report_deadline(corpora):
    # both "may" units ranked first; the handbook states no final-report deadline
    sp, corpus = ranked(corpora, "grants", [("may be hosted by colleges", 9.0), ("The Dean may approve the project's closure", 9.0)])
    result = RT.run(sp, "When is the deadline for the final research report?")
    assert result.status is not S and covering(result, "deadline or date") == []
    RT.assert_traceable(result, corpus)


def test_a_window_start_never_answers_a_deadline_but_answers_when_it_opens(registry_and_map):  # noqa: F811
    corpus = RT.corpus_from(registry_and_map, [page(165, "13. Seed projects are announced at the beginning of each academic year.")])
    sp = RT.make_specialist(corpus, RT.ScriptedReranker([], default=9.0))
    deadline = RT.run(sp, "What is the deadline for seed projects?")
    assert deadline.status is not S and covering(deadline, "deadline or date") == []
    opens = RT.run(sp, "When are seed projects announced?")
    assert opens.status is S and any("beginning of each academic year" in q for q in covering(opens, "deadline or date"))


# ---------------------------------------------------------------------------
# M4-F8 value, approver, eligibility and timing bound to the asked object
# ---------------------------------------------------------------------------
WRONG_OBJECT = [
    # (group, question, wrong evidence ranked first, lines removed so the right answer is absent, text never quoted as the detail)
    ("conference", "What is the per diem for international conferences?", "Local conferences (e.g., Al Ain", ("Per diem rates between",), "AED 700"),
    ("conference", "What is the maximum per diem for international conferences?", "Local conferences (e.g., Al Ain", ("Per diem rates between",), "AED 700"),
    ("conference", "What is the per diem for local conferences?", "Per diem rates between", ("Local conferences (e.g., Al Ain",), "AED 1000-1500"),
    ("conference", "What is the registration fee for conferences?", "Per diem rates between", ("registration fees up to",), "AED 1000-1500"),
    ("conference", "What is the per diem for conferences?", "Conference registration fees up to $750.", ("Per diem rates between", "per diem up to AED 700"), "$750"),
    ("conference", "When must conference requests be submitted?", "within six months", ("Requests must be submitted at least",), "six months"),
    ("conference", "What is the penalty if the conference proceedings are not published?", "not allowed to travel for conferences", (), "summer term"),
    ("consultancy", "What percentage of consultancy income goes to the university?", "20% of their time", (), "20% of their time"),
    ("consultancy", "Who approves the use of the University's name in consultancy?", "pre-approval from the College Dean", ("16.1.8 Use of University Name",), "College Dean"),
    ("grants", "Who is eligible to be a principal investigator?", "may apply for a research advance", ("must hold a Ph.D.",), "research advance"),
    ("grants", "What is the maximum research advance amount?", "petty cash advance of up to AED 5000", ("may apply for a research advance",), "AED 5000"),
    ("grants", "What is the maximum funding for student research projects?", "advise a maximum of four students", ("funding ceiling for each project",), "four students"),
    ("grants", "How many research drones may a faculty member own?", "Groups of researchers (at least three)", (), "at least three"),
    ("grants", "What percentage of the budget can be spent on research assistants?", "may apply for a research advance of up to 10%", ("Costs for research assistants",), "10%"),
    ("grants", "Who approves the final research report?", "The Dean may approve the project's closure", (), "project's closure"),
    ("grants", "What are the duties of the editorial secretary?", "The terms of office for the Chief Editor", ("supports the journal's operations",), "two years"),
    ("ip", "What percentage of net royalty revenue goes to the inventor?", "The University's share, in turn", ("50% to the inventor(s)",), "40% to the VCRGS"),
    ("ip", "What is the deadline to disclose an invention?", "no later than 90 calendar days", (), "90 calendar days"),
    ("ip", "Who owns inventions made by faculty?", "Establish procedures for the exploitation", ("shall own all Intellectual Property Rights", "shall retain ownership"), "Establish procedures"),
]


@pytest.mark.parametrize("group,question,wrong,drop,never", WRONG_OBJECT)
def test_a_value_of_another_object_never_supports_the_question(corpora, group, question, wrong, drop, never):
    sp, corpus = ranked(corpora, group, [(wrong, 9.0)], drop=drop, default=0.5)
    result = RT.run(sp, question)
    assert result.status is not S, (question, result.status, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    for f in result.findings:
        if f.metadata["details_covered"] and never in f.evidence_quote:
            # the wrong unit may be quoted as related context, never as the unit that states the detail
            assert all(label.startswith("the requested") for label in f.metadata["details_covered"]), (question, f.metadata["details_covered"])
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("group,question,needle,label", [
    ("conference", "What is the per diem for international conferences?", "AED 1000-1500", "amount or cost"),
    ("conference", "What is the per diem for local conferences?", "AED 700", "amount or cost"),
    ("conference", "How many conferences can I attend each year?", "one international and one local university-supported conference", "number or amount"),
    ("consultancy", "How much time can I spend on consultancy?", "20% of their time", "maximum or limit"),
    ("consultancy", "Who approves the use of the University's name in consultancy?", "Vice Chancellor for Academic Affairs", "approving authority"),
    ("grants", "Who is eligible to be a principal investigator?", "must hold a Ph.D.", "eligibility conditions"),
    ("grants", "How much petty cash can a principal investigator receive?", "AED 5000", "amount or cost"),
    ("grants", "Who approves petty cash advances?", "subject to approval by the VCRGS", "approving authority"),
    ("grants", "What is the maximum research advance?", "research advance of up to 10%", "maximum or limit"),
    ("grants", "What percentage of the budget can be spent on research assistants?", "30% of the project's budget", "percentage or share"),
    ("grants", "How many projects can a principal investigator lead at the same time?", "two projects simultaneously", "number or amount"),
    ("grants", "What are the duties of the editorial secretary?", "supports the journal's operations", "the responsible role"),
    ("ip", "What percentage of net royalty revenue goes to the inventor?", "50% to the inventor(s)", "percentage or share"),
    ("ip", "What percentage of the University's share goes to the college?", "5% to the Inventors", "percentage or share"),
])
def test_the_value_of_the_asked_object_is_supported(corpora, group, question, needle, label):
    sp, corpus = ranked(corpora, group)
    result = RT.run(sp, question)
    assert result.status is S, (question, result.status, result.missing, RT.quotes(result))
    assert any(needle in q for q in covering(result, label)), (needle, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    RT.assert_traceable(result, corpus)


def test_a_lead_time_stated_in_a_lead_in_is_bound_to_the_requests(registry_and_map):  # noqa: F811
    corpus = RT.corpus_from(registry_and_map, [page(119, "5.4.5 Required Documentation",
                                                    "Requests must be submitted at least one month before the conference date, including:",
                                                    "o A copy of the conference flier with fees stated.",
                                                    "o Evidence that conference proceedings will be published in a Scopus-indexed outlet or commitment to submit to a Scopus journal within six months.",
                                                    "5.5 Policy on Offering Courses in the Summer Semester",
                                                    "Courses offered during the summer semester are primarily limited to first and second-year classes.")])
    sp = RT.make_specialist(corpus, RT.ScriptedReranker([], default=9.0))
    for question, label in (("When must conference requests be submitted?", "deadline or date"), ("How early must I submit a conference request?", "how far in advance")):
        result = RT.run(sp, question)
        assert result.status is S, (question, result.status, result.missing)
        assert all("at least one month before the conference date" in q for q in covering(result, label)) and covering(result, label)


def test_a_real_lead_in_whose_first_item_is_unrelated_is_a_bounded_false_negative(corpora):
    """On the real page 119 the lead-in "Requests must be submitted at least one month before the conference date,
    including:" is followed by "Approval from the concerned college.", which shares nothing with a timing question, so
    the shared quote selector never joins the lead-in. The result is partial, never the proceedings' "six months"."""
    sp, corpus = ranked(corpora, "conference")
    result = RT.run(sp, "When must conference requests be submitted?")
    assert result.status is not S and all("six months" not in q for q in covering(result, "deadline or date"))
    RT.assert_traceable(result, corpus)


def test_a_list_item_is_quoted_whole_so_its_later_sentence_answers_a_permission_question(registry_and_map):  # noqa: F811
    """Item 15 opens with the Ph.D. rule and ends with "and not visitors"; the item is one evidence unit, so it is
    quoted whole and its prohibition answers "Can visitors be principal investigators?"."""
    corpus = RT.corpus_from(registry_and_map, [page(165,
        "15. The principal investigator of a project must hold a Ph.D. degree or an equivalent degree recognized by the University in their specialty. Additionally, they must be full-time employees of the University and not visitors.")])
    result = RT.run(RT.make_specialist(corpus, RT.ScriptedReranker([], default=9.0)), "Can visitors be principal investigators?")
    assert result.status is S, (result.status, result.missing)
    quote = covering(result, "a statement permitting or prohibiting it")[0]
    assert quote.startswith("15. The principal investigator") and quote.endswith("and not visitors.")


def test_whole_item_never_adds_a_sibling_or_text_outside_the_allowed_span():
    text = "15. The rule is stated here. Additionally, they must be employees. 16. Another item follows."
    assert R.whole_item(text, "15. The rule is stated here.") == "15. The rule is stated here. Additionally, they must be employees."
    assert R.whole_item(text, "15. The rule is stated here.", allowed=[(0, text.index("Additionally"))]) == "15. The rule is stated here."
    assert R.whole_item(text, "Additionally, they must be employees.") == "Additionally, they must be employees."


def test_a_single_category_rule_never_answers_a_general_question(registry_and_map):  # noqa: F811
    corpus = RT.corpus_from(registry_and_map, [page(173,
        "60. Visiting Researcher Grants Faculty members can apply for grants to host visiting researchers, subject to Board or VCRGS approval for research groups.",
        "14. Faculty members may not act as principal investigators for two projects simultaneously, except for seed projects.")])
    sp = RT.make_specialist(corpus, RT.ScriptedReranker([], default=9.0))
    general = RT.run(sp, "Who approves research grant applications?")
    assert general.status is not S and covering(general, "approving authority") == []
    own = RT.run(sp, "Who approves visiting researcher grants?")
    assert own.status is S and any("visiting researchers" in q for q in covering(own, "approving authority"))
    # a category named only as an exception leaves the rule general
    count = RT.run(sp, "How many projects can a principal investigator lead at the same time?")
    assert any("two projects simultaneously" in q for q in covering(count, "number or amount"))


def test_petty_cash_approver_is_never_the_dean_who_approves_invoices(registry_and_map):  # noqa: F811
    corpus = RT.corpus_from(registry_and_map, [page(166,
        "The Principal Investigator may extend the advance to another period based on the project budget and approved completion period.",
        "The petty cash amount may also be substituted by original invoices approved by the Dean until the advance term's closure at the end of the project.")])
    result = RT.run(RT.make_specialist(corpus, RT.ScriptedReranker([], default=9.0)), "Who approves petty cash advances?")
    assert result.status is not S and covering(result, "approving authority") == []


@pytest.mark.parametrize("question,unit,expected", [
    ("What is the per diem for international conferences?", "Local conferences (e.g., Al Ain, Abu Dhabi) are covered for transportation and a per diem up to AED 700.", False),
    ("What is the per diem for local conferences?", "Per diem rates between AED 1000-1500 for a maximum of four days, depending on the conference location.", False),
    ("What is the per diem for international conferences?", "Per diem rates between AED 1000-1500 for a maximum of four days, depending on the conference location.", True),
    ("What is the registration fee for conferences?", "Per diem rates between AED 1000-1500 for a maximum of four days.", False),
    ("What percentage of consultancy income goes to the university?", "Faculty may spend up to 20% of their time on professional consultancy activities.", False),
    ("What is the maximum research advance amount?", "The principal investigator is entitled to a petty cash advance of up to AED 5000. The advance may be closed at the end of the year.", False),
])
def test_bound_detail_checks_on_a_single_unit(question, unit, expected):
    details = R.requested_details(question)
    value_details = [d for d in details if d.family in ("amount", "percentage", "maximum")]
    assert value_details and all(d.check(unit) is expected for d in value_details), [(d.label, d.check(unit)) for d in value_details]


def test_a_list_item_is_bound_through_its_lead_in():
    clause = "How is the University's share of licensing revenue distributed?"
    details = R.requested_details(clause)
    assert {d.family for d in details} == {"percentage"}
    quote = "The University's share, in turn, shall be distributed as follows: a. 40% to the VCRGS to support TTO and research activities at the University."
    assert R.unit_coverage(details, R.concepts(clause), quote) == frozenset({"percentage"})


def test_a_start_reference_never_binds_a_deadline_to_its_object():
    deadline = next(d for d in R.requested_details("What is the deadline to disclose an invention?") if d.family == "deadline")
    unit = "The TTO shall inform the Inventor(s) of the outcome no later than 90 calendar days from receipt of a complete Intellectual Property disclosure."
    assert deadline.value(unit) is True and deadline.check(unit) is False


# ---------------------------------------------------------------------------
# M4-F9 topic relevance alone never answers the question
# ---------------------------------------------------------------------------
UNANSWERED_FOCUS = [
    ("grants", "Is there a research grant for time travel?"), ("grants", "Can a research grant pay for a new car?"),
    ("grants", "Can I use a research grant to pay for my child's tuition?"), ("grants", "Does the university fund research in Antarctica?"),
    ("grants", "Is there special funding for COVID-19 research?"), ("ip", "Does the TTO handle trademarks?"),
    ("grants", "Are part-time faculty eligible for research grants?"), ("grants", "Is there a grant for buying a 3D printer?"),
    ("ip", "What is the royalty share for music compositions performed on Mars?"), ("grants", "How many research drones may a faculty member own?"),
    ("conference", "Can I attend a conference without presenting a paper?"), ("grants", "Can visitors act as research assistants?"),
]


@pytest.mark.parametrize("group,question", UNANSWERED_FOCUS)
def test_a_topically_related_quote_never_answers_a_specific_focus(corpora, group, question):
    sp, corpus = ranked(corpora, group, default=9.0)                   # every chunk of the area ranked first
    result = RT.run(sp, question)
    assert result.status in (P, N), (question, result.status, RT.quotes(result))
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("clause,terms", [
    ("Is there a research grant for time travel?", ["time", "travel"]), ("Can a research grant pay for a new car?", ["car"]),
    ("Does copyright protection cover computer software?", ["computer", "software"]), ("Does the university fund research in Antarctica?", ["antarctica", "research"]),
    ("Can I attend a conference without presenting a paper?", ["paper"]), ("Are part-time faculty eligible for research grants?", ["part time"]),
    # a slot that names only the research topic carries no focus
    ("What is the per diem for international conferences?", []), ("Is there funding for conference travel?", []),
    ("What are the rules for student research projects?", []), ("How do I apply for a research grant?", []),
    ("Does thesis supervision count in research evaluation?", []), ("What documents are required for a conference request?", []),
])
def test_focus_terms(clause, terms):
    assert R.focus_terms(clause) == terms


@pytest.mark.parametrize("group,question,needle,label,first", [
    ("ip", "Who owns inventions made by faculty?", "shall own all Intellectual Property Rights of inventions", "ownership or rights holder", None),
    ("ip", "Who owns IP created by students?", "students", "ownership or rights holder", None),
    ("ip", "Does copyright protection cover computer software?", "computer software", "the specific focus asked about (computer, software)", None),
    ("ip", "How is the University's share of licensing revenue distributed?", "40% to the VCRGS", "percentage or share", None),
])
def test_ownership_permission_lead_time_and_focus_are_supported_by_the_stating_unit(corpora, group, question, needle, label, first):
    sp, corpus = ranked(corpora, group, [(first, 9.0)] if first else (), default=0.5 if first else -5.0)
    result = RT.run(sp, question)
    assert result.status is S, (question, result.status, result.missing, RT.quotes(result))
    assert any(needle in q for q in covering(result, label)), [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings]
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("group,question,wrong,drop", [
    ("ip", "Who owns inventions made by faculty?", "Establish procedures for the exploitation", ("shall own all", "shall retain ownership")),
    ("ip", "Does copyright protection cover computer software?", "Copyrighted materials remain the property", ("computer software",)),
    ("grants", "Can visitors be principal investigators?", "may not act as principal investigators", ("not visitors",)),
    ("conference", "How early must I submit a conference request?", "within six months", ("Requests must be submitted at least",)),
    ("conference", "Can I attend a conference without presenting a paper?", "not permitted during the first and last weeks", ("without a paper",)),
])
def test_ownership_permission_lead_time_and_focus_are_never_supported_by_topic_alone(corpora, group, question, wrong, drop):
    sp, corpus = ranked(corpora, group, [(wrong, 9.0)], drop=drop, default=0.5)
    result = RT.run(sp, question)
    assert result.status is not S, (question, result.status, RT.quotes(result))
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("clause,families", [
    ("Who owns inventions made by faculty?", {"ownership"}), ("Does the university own the copyright of my book?", {"ownership", "focus"}),
    ("Can visitors be principal investigators?", {"permission", "focus"}), ("Is commercialization allowed?", {"permission"}),
    ("How early must I submit a conference request?", {"advance"}), ("How far in advance must I request conference support?", {"advance"}),
    ("How is the University's share of licensing revenue distributed?", {"percentage"}), ("How are royalties shared with inventors?", {"percentage"}),
])
def test_new_semantic_families(clause, families):
    assert {d.family for d in R.requested_details(clause)} == families


@pytest.mark.parametrize("clause", ["Walk me through applying for a seed grant.", "How do I fill in the invention disclosure form?",
                                    "How do I fill out the conference request form?", "Guide me through the IRB application.",
                                    "How do I complete the Materials Transfer Agreement form?"])
def test_walk_through_and_form_filling_are_process_requests(clause):
    assert R.is_process_request(clause) is True


@pytest.mark.parametrize("group,question", [("grants", "Walk me through applying for a seed grant."), ("ip", "How do I fill in the invention disclosure form?")])
def test_walk_through_and_form_filling_are_never_supported(corpora, group, question):
    sp, _corpus = ranked(corpora, group, default=9.0)
    result = RT.run(sp, question)
    assert result.status in (P, N) and any("complete submission procedure" in m for m in result.missing) or result.status is N


# ---------------------------------------------------------------------------
# M4-F10 compound Research + Teaching questions keep both intents
# ---------------------------------------------------------------------------
COMPOUND = [
    ("What is the per diem for international conferences and what is my teaching load?", "per diem", "teaching load"),
    ("What is the deadline for conference requests and how many office hours must I hold?", "conference requests", "office hours"),
    ("How do I apply for a research grant and what must my syllabus include?", "research grant", "syllabus"),
    ("How is licensing revenue shared and what is the grading policy?", "licensing revenue", "grading"),
    ("Do I need research ethics approval and how do I use Blackboard?", "research ethics", "Blackboard"),
    ("What research counts for promotion and what is the WLAM category for active research faculty?", "promotion", "WLAM"),
    ("How much conference funding can I get and how many days of annual leave do I have?", "conference funding", None),
    ("How do I contact the TTO and what is my teaching workload?", "TTO", "teaching workload"),
    ("How many conferences can I attend per year and what are my office hours?", "conferences", "office hours"),
    ("What is the conference registration fee covered and how do I submit grades?", "registration fee", "grades"),
    ("Who approves petty cash and when must I submit grades?", "petty cash", "grades"),
]


@pytest.mark.parametrize("question,research_part,teaching_part", COMPOUND)
def test_the_coordinator_keeps_the_research_half_of_a_compound_question(question, research_part, teaching_part):
    decision = coordinator.coordinate(question)
    tasks = {t.specialist_id: t for t in decision.subtasks}
    research_or_institutional = [t for sid, t in tasks.items() if sid in (SpecialistId.RESEARCH, SpecialistId.INSTITUTIONAL)]
    assert any(research_part.lower() in " ".join(t.context["focus_clauses"]).lower() for t in research_or_institutional), decision.to_dict()["tasks"]
    if teaching_part is not None:
        assert SpecialistId.TEACHING in tasks and teaching_part.lower() in " ".join(tasks[SpecialistId.TEACHING].context["focus_clauses"]).lower()


@pytest.mark.parametrize("question", [q for q, _r, t in COMPOUND if t is not None and "TTO" not in q])
def test_a_compound_clause_sent_whole_to_teaching_keeps_a_research_handoff(question):
    teaching_cue, other = T.classify_clause(question)
    assert teaching_cue and other is SpecialistId.RESEARCH, (question, teaching_cue, other)


def test_research_hands_every_foreign_intent_of_one_clause_off():
    clause = "Can research reduce my teaching load and how many days of annual leave do I get?"
    assert R.other_domains(clause)[:2] == [SpecialistId.TEACHING, SpecialistId.FACULTY_SERVICES] or \
        set(R.other_domains(clause)) >= {SpecialistId.TEACHING, SpecialistId.FACULTY_SERVICES}


def test_research_multi_handoff_task_ids_are_unique(corpora):
    sp, _corpus = ranked(corpora, "grants")
    clause = "Can research reduce my teaching load and how many days of annual leave do I get?"
    result = RT.run(sp, clause, clauses=[clause])
    targets = [h.requested_specialist for h in result.handoff_requests]
    assert set(targets) >= {SpecialistId.TEACHING, SpecialistId.FACULTY_SERVICES}
    ids = [h.task.task_id for h in result.handoff_requests]
    assert len(ids) == len(set(ids)) and ids[0] == "task-1-handoff-0"


@pytest.fixture(scope="module")
def registry(registry_and_map):  # noqa: F811
    teaching = RI.TI.teaching_for(RI._teaching_corpus())
    research = RT.make_specialist(RT.corpus_from(registry_and_map, PAGES))
    reg = SpecialistRegistry()
    reg.register(teaching)
    reg.register(research)
    return reg


@pytest.mark.parametrize("question", [COMPOUND[0][0], COMPOUND[1][0], COMPOUND[8][0], COMPOUND[9][0]])
def test_dispatch_executes_both_halves_and_executes_no_handoff(registry, question):
    result = D.run_question(question, registry, executable=(SpecialistId.TEACHING, SpecialistId.RESEARCH))
    assert set(result.executed_specialists) >= {SpecialistId.TEACHING, SpecialistId.RESEARCH}
    assert result.llm_calls == 0 and all(r.task.requested_by == "coordinator" for r in result.runs)
    research_run = next(r for r in result.runs if r.specialist_id is SpecialistId.RESEARCH)
    assert [c["outcome"] for c in research_run.findings.metadata["clauses"]] and research_run.findings.status is not O


# ---------------------------------------------------------------------------
# M4-F11 Coordinator reachability (with over-routing controls)
# ---------------------------------------------------------------------------
REACHES_RESEARCH = [
    "What is the per diem for conferences abroad?", "How much is the conference registration fee covered?", "How many conferences can I attend per year?",
    "Who approves conference travel?", "What documents are needed for a conference travel request?", "How far in advance must I request conference support?",
    "Are local conferences covered?", "Can I attend a conference without a paper?", "How long is the chief editor's term?",
    "What is the term of office of the editor-in-chief?", "Who appoints the editorial board?", "How often are seed projects announced?",
    "Who owns inventions made by faculty?", "Who owns inventions made by students?", "Who owns copyright of books written by faculty?",
    "What is a Materials Transfer Agreement?", "Who signs licensing agreements?", "Do I need approval for consulting work?",
    "Can I use university equipment for consulting?", "Can I use the university name in consulting?", "What must be disclosed as a conflict of interest?",
    "What happens if a final project report is not submitted?", "Are publishing costs covered by the university?", "Who is eligible for petty cash?",
    "Who approves petty cash?", "What is the petty cash limit?", "What does the tech-transfer office do?", "What is the per-diem for conferences?",
    "What are the co-investigator responsibilities?", "Is informed consent required for my study?",
]


@pytest.mark.parametrize("question", REACHES_RESEARCH)
def test_research_owned_phrasings_reach_research(question):
    assert SpecialistId.RESEARCH in coordinator.coordinate(question).selected_specialists


@pytest.mark.parametrize("question", [
    "What funding is available for professional development workshops?", "Is there a teaching project for new faculty?", "What is the course project deadline?",
    "Is the student project graded?", "How do I join a video conference?", "Can I book the conference room?", "Where can I publish my course materials?",
    "What is the publication of course materials policy?", "What are the rules on professional ethics?", "What is my teaching workload?",
    "How do I access library journals?", "Who provides IT support?", "How is HR compensation calculated?", "Is there a conference call with my chair?",
])
def test_non_research_phrasings_never_select_research(question):
    assert SpecialistId.RESEARCH not in coordinator.coordinate(question).selected_specialists


@pytest.mark.parametrize("question", ["Where is the Research Institute for Medical and Health Sciences located?", "How do I contact the IRB?"])
def test_location_and_contact_requests_also_select_institutional(question):
    assert SpecialistId.INSTITUTIONAL in coordinator.coordinate(question).selected_specialists


# ---------------------------------------------------------------------------
# M4-F12 vocabulary and morphology
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", [
    "When are seed projects announced?", "How long can a seed project last?", "Can I use university equipment for consulting?",
    "What are the rules for consulting activities?", "What is an MTA?", "Do I need a materials transfer agreement?",
    "What happens if the final research report is late?", "What does the tech-transfer office do?", "What does tech transfer mean?",
    "Who runs technology-transfer?", "Is informed consent required?",
])
def test_research_owned_morphology_is_a_research_clause(clause):
    assert R.classify_clause(clause)[0] is True


@pytest.mark.parametrize("text,anchor", [("What is an MTA?", "mta"), ("Is informed consent required?", "irb"), ("What does the tech-transfer office do?", "technology_transfer"),
                                         ("Can I use equipment for consulting?", "consultancy")])
def test_new_research_anchors(text, anchor):
    assert anchor in R.concepts(text).anchors


def test_the_per_diem_is_its_own_anchor_and_a_registration_line_shares_none():
    # object binding also stops a registration fee answering a per diem question; the anchor is kept as a second guard
    assert "per_diem" in R.concepts("What is the per diem for conferences?").anchors
    assert R.relevance(R.concepts("What is the per diem?"), "Conference registration fees up to $750.")[0] == 0


def test_the_community_service_consultant_role_is_not_consultancy():
    assert "consultancy" not in R.concepts("serving as a consultant or research partner").anchors


# ---------------------------------------------------------------------------
# M4-F13 detail false negatives and list completeness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,family,unit", [
    ("What happens if the final report is not submitted?", "penalty",
     "However, if the project period ends, and the final report is not submitted or justified, the Dean may refer the issue to the Board, which may close the project and reclaim any spent funds from the investigator."),
    ("Who manages intellectual property?", "responsibility",
     "The University of Sharjah has established the Technology Transfer Office (TTO) to manage all intellectual property assets."),
    ("Who decides whether to seek patent protection?", "authority",
     "e. The decision to seek patent protection is made by the TTO based on the disclosure data submitted by the inventor(s)."),
    ("How many conferences can I attend each year?", "quantity",
     "Faculty members are allowed to attend a maximum of one international and one local university-supported conference per year."),
])
def test_bounded_detail_false_negatives_are_fixed(clause, family, unit):
    detail = next(d for d in R.requested_details(clause) if d.family == family)
    assert detail.check(unit) is True


def test_one_item_of_a_long_documents_list_is_partial(registry_and_map):  # noqa: F811
    items = ["o Item %d copy of the supporting letter number %d for the file." % (n, n) for n in range(1, 16)]
    corpus = RT.corpus_from(registry_and_map, [page(119, "5.4.5 Required Documentation",
                                                    "Requests must be submitted at least one month before the conference date, including:", *items,
                                                    "5.5 Policy on Offering Courses in the Summer Semester",
                                                    "Courses offered during the summer semester are primarily limited to first and second-year classes.")])
    result = RT.run(RT.make_specialist(corpus, RT.ScriptedReranker([], default=9.0)), "What documents are required for a conference request?")
    assert result.status is not S
    assert any("complete list of required documents is not quoted" in m for m in result.missing), result.missing


def test_a_short_documents_list_is_quoted_whole_and_supported(corpora):
    sp, corpus = ranked(corpora, "conference")
    result = RT.run(sp, "What documents are required for a conference request?")
    quote = next((q for q in covering(result, "required documents")), "")
    assert result.status is S and "Approval from the concerned college" in quote and "conference flier with fees stated" in quote
    assert "5.5 Policy" not in quote and "summer semester" not in quote                 # the list ends at the next heading
    RT.assert_traceable(result, corpus)


def test_complete_list_never_adds_text_outside_the_chunk_or_the_allowed_span():
    text = "Documents needed, including: o A copy. o Another copy of it. 5.5 Heading Other text here."
    quote, complete = R.complete_list(text, "Documents needed, including: o A copy.", allowed=[(0, text.index("5.5"))])
    assert complete is True and quote == "Documents needed, including: o A copy. o Another copy of it." and quote in text
    quote2, complete2 = R.complete_list(text, "Documents needed, including: o A copy.", allowed=[(0, text.index("o Another"))])
    assert complete2 is False and quote2 == "Documents needed, including: o A copy."
    assert R.complete_list(text, "Other text here.", None) == ("Other text here.", None)


# ---------------------------------------------------------------------------
# M4-F14 reranker output length contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scores", [lambda n: [1.0], lambda n: [1.0] * (n + 2), lambda n: []])
def test_a_reranker_that_returns_the_wrong_number_of_scores_is_an_error(corpora, scores):
    corpus = corpora("grants")

    class Wrong:
        def predict(self, pairs):
            return scores(len(pairs))

    result = RT.run(RT.make_specialist(corpus, Wrong()), "What is the maximum research advance?")
    assert result.status is E and result.findings == []
    assert "reranker returned" in result.metadata["error"] and "Users" not in result.metadata["error"]


def test_a_reranker_that_returns_one_score_per_pair_is_accepted(corpora):
    result = RT.run(RT.make_specialist(corpora("grants")), "What is the maximum research advance?")
    assert result.status in (S, P)


# ===========================================================================
# Step 4.5A round 2: residuals confirmed by the adversarial verification of the first correction round.
# Each block names the verification finding it reproduces (TIM timing, BIND binding, TOP topical support,
# NAV navigation, RT routing, SAF safety). Pages are real printed rows of the handbook, joined per item.
# ===========================================================================
PAGES_R2 = [
    page(154,
         "8.2 Research Support Units and Departments",
         "8.2.1 Research Funding Department",
         "The Research Funding Department (RFD) is responsible for pre- and post-awards administration of research project's funds. It facilitates the procurement of research funding and supports faculty and students in their research projects and groups. The Department ensures compliance with university regulations and bylaws for internal and external research funding and oversees the management of approved research projects, including purchasing and fund reallocation."),
    page(159,
         "8.5 University Research Board",
         "The University Research Board holds several key duties and authorities, including:",
         "1. Developing scientific research policies and proposals at the University.",
         "2. Approving periodic reports on scientific research at the University.",
         "3. Approving decisions on funding research projects for faculty members and graduate students based on recommendations from the College of Graduate Studies and Research."),
    page(160,
         "7. Approving the annual research budget for research grants, research groups, and major research bodies at the University, based on the College's recommendation."),
    page(163,
         "8.9 Scientific Publishing at the University of Sharjah Journal",
         "The University publishes a scientific refereed journal titled \"The University of Sharjah Journal\" through the VCRGS. The journal may be issued in multiple series or as specialized journals. The Chancellor assumes the role of the general supervisor of the journal, with an appointed Editor-In-Chief for each series. The Editorial Board comprises distinguished faculty members representing various specializations in each series, alongside an advisory body consisting of world-class researchers. Additionally, an Editorial Secretary from among the faculty members, nominated by the Editor-In-Chief and appointed with the Chancellor's approval, supports the journal's operations.",
         "The terms of office for the Chief Editor, Editorial Board, and Editorial Secretary are two years, renewable."),
    page(165,
         "13. Applications for all types of research grants can be submitted from the beginning of the first semester until the end of April of every academic year. Seed projects are announced at the beginning of each academic year."),
    page(166,
         "19. The research grant for a project typically does not cover expenses related to participation in conferences, gatherings, seminars, or similar activities, except in limited and exceptional cases where the research is the primary subject matter of the event. Such exceptions require justifiable recommendations by the VCRGS and approval from the Chancellor.",
         "20. The principal investigator is entitled to a petty cash advance of up to AED 5000, subject to approval by the VCRGS. The advance may be closed at the end of the academic year or carried forward to the next academic year. The petty cash amount may also be substituted by original invoices approved by the Dean until the advance term's closure at the end of the project.",
         "25. Computers, laptops, and PCs fall under the responsibility of the University Purchasing Department, which provides them to all currently employed faculty members. Therefore, it is not permitted to purchase computers or their accessories for research projects if they are already available."),
    page(169,
         "34. Coverage of Publishing Academic Research in Conferences",
         "The College Dean may approve the coverage of publishing academic research presented at international scientific conferences and gatherings if the researcher is involved in an approved research project as a principal investigator, associate investigator, coordinator, or member of a research group. The publishing cost will be deducted from the research group's budget or the research project's budget, as applicable. For research projects outside funded projects or research groups, the cost will be covered by the scientific research budget, and the approval of the Chancellor is required if the amount exceeds AED 2,000.",
         "36. Formation of Research Groups",
         "Research groups must observe the following:",
         "c. Research groups are registered for three renewable years, subject to performance evaluation, University policy, Board approval, and the Chancellor's endorsement."),
    page(173,
         "56. External Funding for Research Projects",
         "The VCRGS facilitates obtaining external funding from national and international institutions for research projects, with 30% of research grants deducted for the University of Sharjah’s account.",
         "60. Visiting Researcher Grants",
         "Faculty members can apply for grants to host visiting researchers, subject to Board or VCRGS approval for research groups."),
    page(181,
         "9.8 Commercialization",
         "6. The TTO, in consultation with the inventor(s), holds the authority to make decisions regarding the method of commercializing UoS Intellectual Property. External parties, including legal advisors and patenting agencies, may be consulted to facilitate commercialization.",
         "7. The authority to sign licenses or agreements related to the transfer of university-owned intellectual property remains with the VCRGS and the TTO. Inventor(s) of university-owned intellectual property have no right to negotiate licensing agreements with any external party under any circumstances.",
         "9.9 Distribution of Licensing Revenues",
         "3. After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), and 50% to the University.",
         "The University's share, in turn, shall be distributed as follows:",
         "a. 40% to the VCRGS to support TTO and research activities at the University.",
         "b. 5% to the Inventors’ College(s), if applicable."),
    page(182,
         "5. In cases where inventor(s) of Intellectual Property cannot be clearly or fairly identified, royalties shall be shared between the university and the relevant academic department or research institute.",
         "9.10 Dispute Resolution",
         "1. In case of a dispute arising from the application of this policy or any related procedures, the VCRGS shall appoint a moderator (arbiter) to assist the parties in resolving the dispute."),
    page(266,
         "Informed Consent:",
         "Informed consent is a fundamental aspect of human research. Researchers must obtain written informed consent from all participants before their involvement in the study.",
         "Monitoring and Reporting:",
         "Researchers must monitor the progress of the study regularly and promptly report any adverse events or unanticipated problems to the IRB. The IRB may suspend or terminate the study if necessary to protect participants' rights and welfare.",
         "16.5 Policy Prohibiting Sexual Harassment",
         "The University of Sharjah is dedicated to fostering a safe and respectful environment characterized by mutual trust, dignity, and respect for all members of its community."),
]


@pytest.fixture(scope="module")
def corpora2(registry_and_map):  # noqa: F811
    cache = {}

    def get(drop=()):
        key = tuple(drop)
        if key not in cache:
            pages = [page(p["page"], *[l for l in p["rows"] if not any(n in l for n in drop)]) for p in PAGES_R2]
            cache[key] = RT.corpus_from(registry_and_map, pages)
        return cache[key]

    return get


def ranked2(corpora2, rules=(), drop=(), default=-5.0):
    corpus = corpora2(drop)
    return RT.make_specialist(corpus, RT.ScriptedReranker(rules, default=default)), corpus


# ---------------------------------------------------------------------------
# BIND / TIM: a value, approver or time of another object, sentence or recommender never answers the question
# ---------------------------------------------------------------------------
WRONG_OBJECT_R2 = [
    # (question, wrong evidence ranked first, lines removed so that the right answer is absent)
    ("Who approves conference travel?", "19. The research grant for a project typically", ()),                         # BIND-01
    ("Who approves payment procedures for research projects?", "20. The principal investigator is entitled", ()),      # BIND-01
    ("Who approves the annual research budget?", "7. Approving the annual research budget", ()),                       # BIND-02 (a recommender)
    ("Who approves decisions on funding research projects?", "3. Approving decisions on funding", ()),                 # BIND-02
    ("Who appoints the Editor-in-Chief?", "nominated by the Editor", ()),                                              # BIND-03
    ("Who appoints the moderator in an IP dispute?", "5. In cases where inventor(s)", ("shall appoint a moderator",)),  # BIND-03
    ("What percentage of consultancy income goes to the University?", "50% to the inventor", ()),                      # BIND-04
    ("What percentage of royalties goes to the Inventors' College?", "3. After recovering", ("b. 5% to the Inventors",)),  # BIND-05
    ("Can I buy a laptop with my research grant?", "19. The research grant for a project typically", ("25. Computers, laptops",)),  # BIND-07
    ("Who signs IP licensing agreements?", "6. The TTO, in consultation", ("7. The authority to sign",)),              # BIND-08
    ("Who approves covering the cost of publishing in a journal?", "34. Coverage of Publishing", ()),                  # BIND-10
    ("Who approves coverage of journal publication fees?", "The College Dean may approve the coverage", ()),           # BIND-10
    ("Who is responsible for approving research grants?", "The Research Funding Department (RFD)", ()),                # NAV-09 (the action)
    ("Which office approves research funding?", "The Research Funding Department (RFD)", ()),                          # NAV-09
    ("Who is the chief editor?", "nominated by the Editor", ()),                                                        # RT-08 (a named person)
    ("By when are seed projects announced?", "13. Applications for all types", ()),                                    # TIM-07 (another sentence)
    ("What is the deadline for announcing seed projects?", "13. Applications for all types", ()),                      # TIM-07
    ("When must adverse events be reported to the IRB?", "Informed consent is a fundamental", ()),                     # TIM-05
    ("What is the deadline for IRB approval?", "Informed consent is a fundamental", ()),                               # TIM-05
    ("When must human research be approved by the IRB?", "Informed consent is a fundamental", ()),                     # TIM-05
]


@pytest.mark.parametrize("question,wrong,drop", WRONG_OBJECT_R2)
def test_round2_wrong_object_sentence_or_recommender_never_supports(corpora2, question, wrong, drop):
    sp, corpus = ranked2(corpora2, [(wrong, 9.0)], drop=drop, default=0.5)
    result = RT.run(sp, question)
    assert result.status is not S, (question, result.status, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    assert result.missing
    RT.assert_traceable(result, corpus)


WRONG_OBJECT_R2_SHARED = [
    # the same attacks on the Step 4.5A corpora: orphan list items, the other conference locality, a lead time of another document
    ("grants", "What is the funding ceiling for a faculty research project?", "funding ceiling for each project", ()),         # BIND-06
    ("grants", "What is the funding ceiling for research grants?", "funding ceiling for each project", ()),                    # BIND-06
    ("conference", "What is the per diem for regional conferences?", "Local conferences (e.g., Al Ain", ("Per diem rates between",)),  # BIND-09
    ("conference", "When must the conference flier be submitted?", "within six months", ("Requests must be submitted at least",)),     # TIM-04
    ("conference", "When must I submit the abstract for conference funding?", "within six months", ("Requests must be submitted at least",)),
    ("conference", "What is the deadline for submitting the conference acceptance letter?", "within six months", ("Requests must be submitted at least",)),
    ("conference", "What is the deadline for the conference flier?", "within six months", ("Requests must be submitted at least",)),
    ("conference", "By when must I provide evidence of the conference quality?", "within six months", ("Requests must be submitted at least",)),
    ("conference", "When must conference proceedings be published?", "within six months", ("Requests must be submitted at least",)),
    ("conference", "What is the deadline for publishing conference proceedings?", "within six months", ("Requests must be submitted at least",)),
]


@pytest.mark.parametrize("group,question,wrong,drop", WRONG_OBJECT_R2_SHARED)
def test_round2_orphan_item_locality_and_document_lead_time_never_support(corpora, group, question, wrong, drop):
    sp, corpus = ranked(corpora, group, [(wrong, 9.0)], drop=drop, default=0.5)
    result = RT.run(sp, question)
    assert result.status is not S, (question, result.status, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("question,needle,label", [
    ("Who appoints the moderator in an IP dispute?", "shall appoint a moderator", "approving authority"),                      # BIND-03
    ("Who signs IP licensing agreements?", "7. The authority to sign", "approving authority"),                                  # BIND-08
    ("Who signs licensing agreements for university IP?", "7. The authority to sign", "approving authority"),                   # BIND-08
    ("What percentage of royalties goes to the Inventors' College?", "b. 5% to the Inventors", "percentage or share"),          # BIND-05
    ("What percentage of the University's share goes to the VCRGS?", "a. 40% to the VCRGS", "percentage or share"),             # BIND-11
    ("What is the royalty share for inventors?", "50% to the inventor", "percentage or share"),                                 # RT-08
    ("How much does the University deduct from external research grants?", "30% of research grants deducted", "amount or cost"),  # BIND-14
    ("Who is eligible to host visiting researchers?", "Faculty members can apply for grants to host", "eligibility conditions"),  # BIND-13
    ("What is the deadline for research grant applications?", "13. Applications for all types", "deadline or date"),            # TIM-08
    ("When are seed projects announced?", "13. Applications for all types", "deadline or date"),                                # TIM-07 control
    ("When must the petty cash advance be closed?", "20. The principal investigator is entitled", "deadline or date"),          # TIM-09
    ("What is the deadline for closing the petty cash advance?", "20. The principal investigator is entitled", "deadline or date"),
])
def test_round2_the_stating_unit_supports_the_bound_detail(corpora2, question, needle, label):
    sp, corpus = ranked2(corpora2, [(needle, 9.0)], default=-5.0)
    result = RT.run(sp, question)
    assert result.status is S, (question, result.status, result.missing, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    assert any(needle in q for q in covering(result, label)), (question, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    RT.assert_traceable(result, corpus)


def test_round2_a_participle_attached_after_a_comma_is_an_ambiguous_relation(corpora2):
    # Step 4.5A-D: "an Editorial Secretary from among the faculty members, nominated by the Editor-In-Chief" can attach the
    # nomination to the Secretary or to the faculty members; the deterministic specialist cannot prove which, so the
    # quote stays partial (a documented false negative the Milestone 5 Verifier may upgrade)
    sp, corpus = ranked2(corpora2, [("nominated by the Editor", 9.0)], default=-5.0)
    result = RT.run(sp, "Who nominates the Editorial Secretary?")
    assert result.status is P and result.metadata["clauses"][0]["relation"] == "ambiguous"
    assert any("nominated by the Editor" in q for q in RT.quotes(result))
    RT.assert_traceable(result, corpus)


def test_round2_the_harassment_line_of_a_shared_ethics_page_is_never_quoted(corpora2):
    # SAF-2 on the real p266 layout: the IRB paragraphs are Research's, 16.5 is not
    sp, corpus = ranked2(corpora2, [("strictly", 9.9), ("dedicated to fostering a safe", 9.9)], default=5.0)
    result = RT.run(sp, "What does informed consent require in research and what is the harassment policy?")
    assert result.status is P and not any("fostering a safe" in q for q in RT.quotes(result))
    assert any("Faculty Services & HR subject asked about (harassment)" in m for m in result.missing)
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("unit,expected", [
    # NAV-09: the RFD "oversees the management of approved research projects": a responsibility, but not for approving
    ("The Department ensures compliance with university regulations and bylaws for internal and external research funding and oversees the "
     "management of approved research projects, including purchasing and fund reallocation.", False),
    ("The University Research Board is responsible for approving research grants.", True),
    ("Research grants are approved by the University Research Board, which is responsible for their review.", True),
])
def test_round2_a_responsibility_for_an_action_needs_the_action_itself(unit, expected):
    detail = next(d for d in R.requested_details("Who is responsible for approving research grants?") if d.family == "responsibility")
    assert detail.check(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("7. Approving the annual research budget for research grants, research groups, and major research bodies at the University, based on the "
     "College's recommendation.", False),
    ("The annual research budget is approved by the University Research Board.", True),
])
def test_round2_a_recommending_body_is_never_the_approver(unit, expected):
    detail = next(d for d in R.requested_details("Who approves the annual research budget?") if d.family == "authority")
    assert detail.check(unit) is expected


# ---------------------------------------------------------------------------
# TIM-01 / TIM-02 / TIM-06 / TIM-10: deadline wording, end points and window starts (unit level)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", [
    "What is the closing date for research grant applications?", "What is the last date to apply for research grants?",
    "What is the latest date to submit a seed project proposal?", "What is the final date for seed project applications?",
    "When may I apply for a research grant?", "When could I apply for a research grant?", "When shall the final research report be submitted?",
    "In which month are research grant applications due?", "What month do research grant calls open?",
    "At what point must research grant forms be submitted?", "How soon must the final report be submitted?",
    "What time of year are seed projects announced?", "When were seed projects announced?", "When is the progress report due?",
])
def test_round2_every_timing_interrogative_requests_a_date(clause):
    assert "deadline" in [d.family for d in R.requested_details(clause)], clause


@pytest.mark.parametrize("unit,end", [
    ("Applications must be received on or before 30 April.", True), ("Requests must be submitted 30 days ahead of the conference.", True),
    ("Proposals are due in May.", True), ("The window closes mid-May.", True), ("Applications close in May.", True),
    ("Upon project completion, the final report must be submitted.", True), ("Submit your application by Friday.", True),
    ("Journals can be classified by the committee before submission.", True), ("Applications are accepted until the end of April.", True),
    ("Have been regularly issued for at least three years before the publication of the applicant's article.", False),
    ("Research proposals are presented before the Research Board for approval.", False),
    ("The research budget may be increased by 10% in justified cases.", False),
    ("Seed project applications are accepted from 1 September until further notice.", False),
    ("Seed project applications open at the end of the first semester.", False),
    ("Final research reports are submitted after the end of the project.", False),
])
def test_round2_deadline_end_points(unit, end):
    deadline = next(d for d in R.requested_details("What is the deadline?") if d.family == "deadline")
    assert deadline.value(unit) is end, unit


@pytest.mark.parametrize("clause,subject,action", [
    ("When must the conference flier be submitted?", ("flier",), None),
    ("When must I submit the abstract for conference funding?", ("abstract",), None),
    ("What is the deadline for submitting the conference acceptance letter?", ("acceptance", "letter"), None),
    ("By when are seed projects announced?", (), "announc"),
    ("When must conference proceedings be published?", (), "publish"),
    ("What is the deadline for publishing conference proceedings?", (), "publish"),
    ("When must adverse events be reported to the IRB?", ("adverse", "events"), "report"),
    ("What is the deadline for research grant applications?", (), None),
    ("What is the deadline for seed project applications?", (), None),
    ("When is the final report due?", (), None),
])
def test_round2_timing_subject_and_action(clause, subject, action):
    assert R._timing_subject(clause) == (subject, action)


def test_round2_a_time_binds_only_inside_its_own_sentence():
    item13 = ("13. Applications for all types of research grants can be submitted from the beginning of the first semester until the end of April "
              "of every academic year. Seed projects are announced at the beginning of each academic year.")
    applications = next(d for d in R.requested_details("What is the deadline for research grant applications?") if d.family == "deadline")
    announced = next(d for d in R.requested_details("By when are seed projects announced?") if d.family == "deadline")
    assert applications.check(item13) and not announced.check(item13)


# ---------------------------------------------------------------------------
# TOP-F1 .. TOP-F6: a yes/no property, a pronoun-led slot, an agent noun and no prefix matching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,terms", [
    ("Is the seed grant renewable?", ["renewable"]), ("Is the per diem taxable?", ["taxable"]),
    ("Are conference registration fees refundable?", ["registration", "fees", "refundable"]), ("Is research leave paid?", ["leave", "paid"]),
    ("Can I hire my spouse as a research assistant?", ["spouse"]), ("Can adjunct faculty apply for seed grants?", ["adjunct"]),
    ("What happens to my patent after I leave the university?", ["leave"]), ("Do inventors get royalties after they retire?", ["retire"]),
    ("Can I patent an invention made with ChatGPT?", ["chatgpt"]), ("Is there a research grant for publishers?", ["publishers"]),
    ("Does the research grant cover memberships?", ["memberships"]), ("Can I buy a laptop with my research grant?", ["laptop"]),
    # controls: the topic alone, a verb after a pronoun and a covered slot add no focus
    ("Can I hire a research assistant?", []), ("Is there funding for conference travel?", []), ("Is there funding for researchers?", []),
    ("Does thesis supervision count in research evaluation?", []),
])
def test_round2_yes_no_focus_terms(clause, terms):
    assert R.focus_terms(clause) == terms


@pytest.mark.parametrize("question", [
    "Is the per diem taxable?", "Are conference registration fees refundable?", "Is the seed grant renewable?", "Is research leave paid?",
    "Can I hire my spouse as a research assistant?", "Can adjunct faculty apply for seed grants?", "Can emeritus professors apply for seed grants?",
    "What happens to my patent after I leave the university?", "Do inventors get royalties after they retire?", "Can I patent an invention made with ChatGPT?",
    "Is there a research grant for publishers?", "Does the research grant cover memberships?", "Does the university insure research equipment?",
    "Is the research advance interest-free?", "Are research grants taxable income?", "Does the university offer research grants to undergraduates?",
    "Is business class allowed for conference travel?", "Can lab technicians be co-investigators on research projects?",
    "Does the university pay for membership in professional research societies?", "Is there research funding for administrators?",
])
def test_round2_a_topically_related_quote_never_answers_a_yes_no_property(corpora, question):
    sp, corpus = ranked(corpora, "all", default=9.0)                     # every research chunk ranked first
    result = RT.run(sp, question)
    assert result.status is not S, (question, [(f.evidence_quote, f.metadata["details_covered"]) for f in result.findings])
    RT.assert_traceable(result, corpus)


@pytest.mark.parametrize("text,word,expected", [
    ("Academic members must prioritize", "membership", False), ("pre- and post-awards administration", "administrators", False),
    ("research students", "studentships", False), ("the professor", "professional", False), ("the department approves", "departure", False),
    ("The advance may be closed at the end of the year", "closing", True), ("a licence agreement", "license", True),
    ("Commercialization is a priority", "commercialisation", True), ("the inventors' shares", "inventor", True),
])
def test_round2_word_forms_without_prefix_matching(text, word, expected):
    assert R._has_word(text, word) is expected


# ---------------------------------------------------------------------------
# SAF-1 / SAF-2: click requests and foreign subjects of a shared clause are never supported
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause", [
    "What link do I use to apply for a competitive research grant?", "Which button do I click to apply for a research advance?",
    "Which screen do I use to submit an invention disclosure?", "What do I press to submit my conference travel request?",
    "What are the instructions for submitting an invention disclosure form?", "List the steps needed to request a petty cash advance.",
])
def test_round2_click_screen_and_step_requests_are_process_requests(clause):
    assert R._PROCESS_CUES.search(clause), clause


@pytest.mark.parametrize("question", ["What link do I use to apply for a competitive research grant?", "Which button do I click to apply for a research advance?"])
def test_round2_click_requests_are_never_supported(corpora, question):
    sp, corpus = ranked(corpora, "all", default=9.0)
    result = RT.run(sp, question)
    assert result.status is not S and any("procedure" in m for m in result.missing), (result.status, result.missing)


@pytest.mark.parametrize("question,label", [
    ("What are the laboratory safety rules for research labs?", "the Institutional Navigation subject asked about (laboratory safety)"),
    ("Is informed consent required for human research, and what are the lab safety rules?", "the Institutional Navigation subject asked about (lab safety)"),
    ("What is the policy prohibiting sexual harassment in research groups?", "the Faculty Services & HR subject asked about (sexual harassment)"),
])
def test_round2_a_foreign_subject_of_a_shared_clause_stays_missing(corpora2, question, label):
    sp, corpus = ranked2(corpora2, default=9.0)                           # every chunk ranked first, the IRB paragraphs included
    result = RT.run(sp, question)
    assert result.status in (P, N), (result.status, result.missing)
    if result.status is P:                                                # the research half is answered, the foreign subject is not
        assert any(m.startswith(label) for m in result.missing), result.missing
    assert result.handoff_requests and result.handoff_requests[0].metadata["kind"] == "shared"
    RT.assert_traceable(result, corpus)


def test_round2_a_task_not_issued_by_the_coordinator_is_never_executed(registry):
    # SAF-5: a handoff task placed in a decision stays data; no second round runs through dispatch
    from handbook_bot.agents.specialists.contracts import CoordinatorDecision, SpecialistTask

    forged = SpecialistTask(task_id="task-1-handoff-0", question="What is the per diem for international conferences?", specialist_id="research",
                            domain="research", context={"focus_clauses": ["What is the per diem for international conferences?"]}, requested_by="teaching")
    decision = CoordinatorDecision(selected_specialists=[SpecialistId.RESEARCH], subtasks=[forged], reason="a replayed handoff")
    result = D.dispatch(decision, registry)
    assert [r.state for r in result.runs] == [D.STATE_PENDING] and result.executed_specialists == []
    assert "not issued by the Faculty Onboarding Coordinator" in result.runs[0].reason


# ---------------------------------------------------------------------------
# NAV-12 / RT-01 .. RT-13: routing residuals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", [
    "How early must I apply before the conference?", "What is the limit on the number of conferences per year?",
    "Can I join a conference without submitting a paper?", "Who pays page charges for my paper?", "Who owns software I develop at the university?",
    "Can I travel abroad to a congress to present my work?", "Does the university fund attendance at a scientific symposium?",
])
def test_round2_research_phrasings_reach_research(question):
    assert SpecialistId.RESEARCH in coordinator.coordinate(question).selected_specialists, question


@pytest.mark.parametrize("question", [
    "Can I book the conference room?", "Is there IT support for conference calls?", "Who grants tenure?", "How do I grade a student project report?",
    "What is the IP address of the classroom projector?", "Where is the rec center?", "Do I need to attend a conference call with the chair?",
    "Is there a professional ethics committee for faculty misconduct?",
])
def test_round2_over_routing_controls_never_select_research(question):
    assert SpecialistId.RESEARCH not in coordinator.coordinate(question).selected_specialists, question


def test_round2_teaching_never_hands_a_conference_call_to_research():
    assert T.other_domains("Can I attend a conference call during my office hours?") == []
    assert T.other_domains("Can I attend a conference abroad?") == [SpecialistId.RESEARCH]


@pytest.mark.parametrize("question", ["Where do I submit my research proposal?", "Which office handles research grant applications?",
                                      "Who should I contact about research grants?", "How do I contact the Research Funding Department?"])
def test_round2_teaching_hands_navigation_to_institutional_first(question):
    teaching, other = T.classify_clause(question)
    assert not teaching and other is SpecialistId.INSTITUTIONAL


@pytest.mark.parametrize("question", ["What floor is the research office on?", "How do I find the research office?", "Which building is the TTO in?",
                                      "What is the address of the research office?"])
def test_round2_location_questions_select_institutional_and_are_navigation(question):
    assert SpecialistId.INSTITUTIONAL in coordinator.coordinate(question).selected_specialists
    assert R.navigation_intent(question) == (True, False)


def test_round2_a_channel_half_goes_to_institutional_and_the_subject_half_to_research():
    decision = coordinator.coordinate("What is the per diem for conferences and what is the phone number of the research office?")
    assert decision.selected_specialists[:2] == [SpecialistId.RESEARCH, SpecialistId.INSTITUTIONAL]
    research = next(t for t in decision.subtasks if t.specialist_id is SpecialistId.RESEARCH)
    assert research.context["focus_clauses"][0] == "What is the per diem for conferences"


def test_round2_a_continuation_keeps_its_shared_head_and_a_greeting_is_no_clause():
    assert coordinator._split_clauses("What is my teaching load and the registration fee for local and international conferences?") == [
        "What is my teaching load and the registration fee for local and international conferences"]
    assert coordinator._split_clauses("Good morning, what is the petty cash limit and my teaching load?") == [
        "what is the petty cash limit", "my teaching load"]
    assert coordinator._split_clauses("Hello!") == ["Hello!"]
