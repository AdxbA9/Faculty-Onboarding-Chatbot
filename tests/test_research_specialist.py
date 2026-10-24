"""Research & Innovation Specialist, deterministic version 1.

Everything here runs offline. The corpus is a set of synthetic handbook pages
whose lines mirror the real handbook (real page numbers, real section
headings printed as lines, real sentences where the audit recorded them),
chunked by the project's own chunker and annotated with the real registry
record and the real section map. Ranking is a raw token-overlap reranker or a
scripted reranker with predetermined scores, so a test can put a wrong chunk
first and the right chunk last. Run-based tests use focused corpora (the
pages of one policy area) so they exercise the specialist's gates, not the
hashed stand-in embedder's recall. No model, no LLM, no network.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import faiss
import pytest

from handbook_bot.agents import coordinator
from handbook_bot.agents.specialists import research
from handbook_bot.agents.specialists.base import Specialist, check_findings
from handbook_bot.agents.specialists.contracts import FindingStatus, SpecialistId, SpecialistTask
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.agents.specialists.research import (
    ANCHOR_CONCEPTS,
    APPROVED_SOURCE_ID,
    EXCLUDED_SECTIONS,
    EXTRA_PAGES,
    GENERIC_TERMS,
    OWNED_SECTIONS,
    SCOPE_SECTIONS,
    SHARED_SECTIONS,
    VOCABULARY_CONCEPTS,
    ResearchInnovationSpecialist,
    ResearchResources,
    classify_clause,
    compile_scope,
    concepts,
    detail_covered,
    is_procedure_request,
    is_process_request,
    is_relevant,
    quote_problem,
    requested_details,
    select_quote,
)
from handbook_bot.chunking import build_chunks
from handbook_bot.config import MIN_RERANK_SCORE
from handbook_bot.sources import annotate_metadata, load_section_map, load_source_registry, unregistered_source
from handbook_bot.text_utils import normalize_text, tokenize

from conftest import FakeEmbedder

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixture pages (lines mirror the handbook; headings are printed as lines)
# ---------------------------------------------------------------------------
def page(number, *lines):
    lines = list(lines)
    return {"page": number, "lines": ["Page %d" % number] + lines, "rows": lines,
            "text": normalize_text(" ".join(["Page %d" % number] + lines)), "section_hint": "", "ocr_snippets": []}


RESEARCH_PAGES = [
    page(80,
         "3. Consideration for Evening Courses: Faculty members assigned to teach evening courses should schedule at least one office hour during the evening.",   # 3.4
         "3.5 Responsibility to Research",
         "In addition to their commitment to quality education, faculty members at the University of Sharjah hold a fundamental responsibility to advance knowledge in their respective academic disciplines through research and creative endeavors."),
    page(81,
         "4. Supervision of Academic Activities: Faculty members play an instrumental role in guiding students in their academic pursuits. This responsibility includes supervising dissertations, student research projects, and other academic activities.",
         "5. Theses/Dissertations Supervision: To ensure effective mentorship and balanced supervision of theses/dissertations:",
         "• A faculty member is allowed to supervise a maximum of four theses/dissertations simultaneously, exception may be granted by the College of Graduate Studies."),
    page(82,
         "Faculty members are expected to disseminate their research findings through publications in refereed journals and presentations at conferences.",
         "3.6 Responsibility to Community Service",
         "This may include serving as a consultant or research partner in their respective professional areas, offering expertise as a resource person."),
    page(117,
         "5.4 Participation in Conferences",
         "The University of Sharjah (UoS) recognizes the immense value of faculty's participation in conferences and seminars organized by scientific and specialized organizations.",
         "5.4.1 Policy",
         "The University provides financial support to eligible faculty, subject to the following guidelines and conditions:",
         "• Applicants must be regular faculty members who have passed the probationary period.",
         "• Conferences attended at no cost to the University are discretionary and require the same approval process as cost-based conferences.",
         "• The Vice Chancellor for Academic Affairs approves travel requests and interprets the policy."),
    page(118,
         "5.4.2 Eligibility and Application",
         "• The participant must be the main or joint author of the paper and must be the presenter.",
         "• The paper must bear the name of the University of Sharjah.",
         "• Faculty are entitled to participate in one local and one external university-supported conference per year, provided it does not conflict with their teaching duties."),
    page(119,
         "• Per diem rates between AED 1000-1500 for a maximum of four days, depending on the conference location.",
         "• Conference registration fees up to $750.",
         "• For local conferences, support is up to AED 700 for a maximum of three days.",
         "5.4.5 Required Documentation",
         "Requests must be submitted at least one month before the conference date, including:",
         "• A copy of the conference flier with fees stated.",
         "• The abstract acceptance letter and a copy of the paper.",
         "• Participants must submit a report to their Department Chair and Dean's Office within two weeks of return.",
         "5.5 Policy on Offering Courses in the Summer Semester",
         "1. Faculty members' compensation for teaching in the summer semester shall not exceed two months' basic salary."),
    page(123,
         "Peer and student feedback on teaching is collected each semester.",                          # 6.2 tail
         "6.3 Academic Advising",
         "Faculty participation in academic advising will be evaluated as part of the overall performance.",
         "6.4 Research and Scholarly Activities Evaluation",
         "UoS faculty members are expected to engage in scholarly activities that contribute to their respective disciplines. The evaluation criteria encompass a range of scholarly contributions, including:",
         "• Publications in esteemed journals and conferences, with special recognition for publications in refereed journals of class (A) and (B).",
         "• Recognition through awards, grants, contracts, patents, and peer-reviewed accomplishments."),
    page(124,
         "• Adherence to professional standards and ethics governing research activities.",
         "6.5 Community Service Evaluation",
         "Community service is evaluated through documented contributions to the University and the wider community.",
         "6.6 Evaluation Process and Criteria",
         "The evaluation process is conducted annually by the department chair using the evaluation form.",
         "6.7 Special Issues and Considerations",
         "The research score shall be weighted at 30 percent of the annual evaluation for regular faculty."),
    page(144,
         "7.1 Promotion Criteria",
         "Promotion is based on research, teaching and service as detailed below.",
         "7.2 Research and Scholarship",
         "For promotion purposes, research considered admissible includes:",
         "A. Research published (or accepted for publication) in recognized specialized refereed journals.",
         "B. Books published by recognized academic publishers."),
    page(145,
         "C. Patents registered with recognized patent offices.",
         "7.3 Teaching and Academic Advising",
         "Teaching effectiveness is assessed through student evaluations and peer observation of teaching."),
    page(147,
         "The candidate must have served at least five years as an assistant professor.",              # 7.5 tail
         "7.6 Promotion to the Rank of Full Professor",
         "A. Have a minimum of five years as an associate professor, with at least two years at the University.",
         "7.7 Criteria for Accrediting Periodicals and Journals for Promotion Purposes",
         "Academic periodicals and journals must meet the following conditions:",
         "1. Journals in SCOPUS database are classified as A if their SJR metric ranks in the top 20% of their field."),
    page(148,
         "8. Have been regularly issued for at least three years before the publication of the applicant's paper.",
         "7.8 Promotion Procedures at the Department Level",
         "The department promotion committee reviews the dossier and votes within two weeks of receiving it."),
    page(153,
         "Chapter 8: Scientific Research",
         "Scientific research at the University of Sharjah is overseen by the University Research Board, which is dedicated to promoting outstanding research endeavors.",
         "8.1 Research Institutes",
         "The University of Sharjah houses three research institutes, each led by a director who reports to the Vice Chancellor for Research and Graduate Studies."),
    page(154,
         "8.2 Research Support Units and Departments",
         "8.2.1 Research Funding Department",
         "The Research Funding Department (RFD) is responsible for pre- and post-award management of research grants.",
         "8.2.1.1 Internal Funding",
         "Internal funding covers seed projects and competitive projects funded from the university research budget."),
    page(156,
         "8.3 Research Priority Areas",
         "To address the most pressing challenges faced by the local and global communities, UoS directs its research activities towards research priority areas in line with its strategic plan."),
    page(157,
         "8.4 Research Groups",
         "The University of Sharjah actively encourages the establishment of research groups within various disciplines and fosters collaborations among them.",
         "As of Fall 2024, there are 62 active research groups at UoS, administered and supervised by the Research Groups Department."),
    page(159,
         "8.5 University Research Board",
         "The University Research Board comprises the following members:",
         "1. Vice Chancellor for Research and Graduate Studies (Chair)"),
    page(160,
         "8.6 Research Committees",
         "Research committees review research grant applications submitted by faculty members and forward them to the VCRGS through the College Deans.",
         "8.7 Research Grants",
         "The University of Sharjah provides robust support for research projects proposed by faculty members. These funded projects encompass a range of categories, including seed projects, competitive projects, collaborative research, and targeted research.",
         "Notable external funders include the National Science Foundation, Emirates Foundation, Qatar Foundation, the British Council, the European Union, and Al-Jalilla Foundation."),
    page(161,
         "8.8 Research Regulations",
         "• Research involving animals must be reviewed and approved by the Institutional Animal Care and Use Committee.",
         "• Disclose conflicts of interest in publications and grants.",
         "• Avoid plagiarism, redundant publication, or fabrication/falsification of data."),
    page(163,
         "8.9 Scientific Publishing at the University of Sharjah Journal",
         "The terms of office for the Chief Editor, Editorial Board, and Editorial Secretary are two years, renewable once.",
         "Editors of the University journal have their teaching loads reduced based on a percentage determined by the Chancellor.",
         "8.10 Collaborative Research Agreements",
         "Collaborative research agreements with external organizations require the approval of the VCRGS.",
         "8.11 Scientific Research Procedures and Regulations",
         "1. The University supports specialized, competitive, collaborative, and targeted research projects of high quality."),
    page(164,
         "12. To apply for research grants, the following steps must be taken:",
         "a. Complete the research grants forms and have them reviewed by research committees in the colleges.",
         "b. Forward the application to the VCRGS through the College Dean.",
         "13. Applications for all types of research grants can be submitted from the beginning of the academic year until the end of the second semester."),
    page(165,
         "c. The VCRGS determines the funding ceiling for research grants based on the Board's recommendation and the Chancellor's approval.",
         "d. The funding ceiling for each seed project is AED 5000.",
         "e. Costs for research assistants are capped at a maximum of 30% of the project's budget, which may be raised to 50% in special and justified cases by the Board."),
    page(166,
         "20. The principal investigator is entitled to a petty cash advance of up to AED 5000, subject to settlement within 60 days.",
         "21. The principal investigator may apply for a research advance of up to 10% of the basic research project grant.",
         "22. The advance may be raised to a maximum of 30% of the basic research project grant in justified cases."),
    page(170,
         "i. The University Research Board may approve partial or full sabbatical or financial awards for coordinators of research groups, upon the Chancellor's approval."),
    page(173,
         "58. Sabbatical Leave and Research Grants",
         "Faculty members on sabbatical leave are not allowed to combine teaching with research grant applications to fund their sabbatical.",
         "62. Sabbatical Leave for Visiting Researchers"),
    page(174,
         "Sabbatical leave for visiting researchers should not exceed two months and should not interfere with University commitments."),
    page(175,
         "Chapter 9: Intellectual Properties Management",
         "The University of Sharjah has established the Technology Transfer Office (TTO) to manage all intellectual property assets derived from the research activities and outcomes at the University.",
         "The implementation of this policy falls under the purview of the Office of the Vice Chancellor for Research and Graduate Studies (OVCRGS) and the TTO.",
         "1. Inventor: Refers to an individual or group of individuals responsible for the creation of Intellectual Property, whether through invention, authorship, or any other means.",
         "3. Gross Income: Signifies the total monetary payments received by the University resulting from the transfer of Intellectual Property rights to a third party."),
    page(176,
         "9.1 Intellectual Property Ownership Rights",
         "Unless stated otherwise in this policy, the University of Sharjah shall own all Intellectual Property Rights of inventions or creative works produced by faculty, staff, and students in the course of their employment or study.",
         "b. Intellectual Property developed by faculty or staff with significant support from UoS, including funds, facilities, resources, and space, shall be assigned to the University."),
    page(177,
         "9.2 University Responsibilities",
         "The Technology Transfer Office (TTO) under the Office of the Vice Chancellor for Research and Graduate Studies shall undertake the following responsibilities:",
         "9.3 Responsibilities of UoS Inventors",
         "Faculty, staff, and students have the following responsibilities:"),
    page(178,
         "9.4 Disclosure of Intellectual Property",
         "9.4.1 The Obligation to Disclose",
         "a. Individuals subject to this Intellectual Property Policy are obligated to promptly disclose in writing any and all Intellectual Property to the TTO using the invention disclosure form."),
    page(179,
         "9.5 Patent Protection",
         "1. In administering patent protection, the TTO shall actively promote the protection and utilization of all inventions in which UoS holds ownership rights.",
         "2. The TTO shall decide whether to file a patent application as soon as practical but no later than 90 calendar days from receipt of a complete invention disclosure."),
    page(180,
         "12.3 Copyright Protection Policy",
         "1. This Policy also covers copyright protection for books, computer software, and other copyrightable materials produced at the University.",
         "9.7 Tangible Research Property",
         "3. For commercial purposes, transferring tangible research property to external parties requires the approval of the VCRGS."),
    page(181,
         "9.8 Commercialization",
         "1. Commercialization is a priority for UoS in developing and distributing intellectual property and facilitating technology transfer for the benefit of society.",
         "9.9 Distribution of Licensing Revenues",
         "3. After recovering IP protection expenses, the net royalty revenue shall be distributed as follows: 50% to the inventor(s), and 50% to the University.",
         "The University's share, in turn, shall be distributed as follows: a. 40% to the VCRGS to support TTO and research activities at the University. b. 5% to the Inventors' College."),
    page(182,
         "9.10 Dispute Resolution",
         "1. In case of a dispute arising from the application of this policy, the VCRGS shall appoint a moderator (arbiter) to assist the parties in reaching a settlement."),
    page(259,
         "Chapter 16: Other Policies",
         "16.1 Faculty Professional Consultancy",
         "The University of Sharjah recognizes the value of faculty consultancy as a means to contribute to the profession and the community.",
         "16.1.3 Pre-Approval for Consultancy Activities: All proposed faculty consultancy activities must receive prior review and approval through the appropriate channels, including the Department Chair and the College Dean."),
    page(260,
         "16.1.4 Non-Compensated Consultancy: Even in cases of non-compensated consultancy, prior approval is required.",
         "Consultancy may not exceed one day per week during the semester."),
    page(261,
         "16.2 Conflict of Interest",
         "Faculty members must disclose any conflict of interest arising from consultancy or sponsored research to the Department Chair."),
    page(263,
         "16.3 Animal Use in Research",
         "All research involving animals must be approved by the Institutional Animal Care and Use Committee (IACUC) before it begins."),
    page(265,
         "The IACUC ensures regular audits, inspections, and reviews of animal research activities to maintain ethical standards.",
         "16.4 Human Use in Research",
         "• Institutional Review Board (IRB): An independent committee responsible for reviewing and approving research protocols involving human subjects to ensure ethical and regulatory compliance."),
    page(266,
         "Researchers must obtain IRB approval before recruiting human participants for a study.",
         "16.5 Policy Prohibiting Sexual Harassment",
         "The University prohibits sexual harassment and will investigate every complaint promptly."),
    # outside the Research scope
    page(96, 'o "C" Research Intensive Faculty has exceptionally high research productivity. The teaching load is 6 credit hours per semester.'),
    page(107, "Faculty are entitled to sick leave with full pay for up to 15 working days and half pay for the following 15 days."),
    page(109, "Annual incentives include AED 20,000 for distinction in Research and Scholarly Activities."),
    page(126, "The total score shall be out of 100 %. There are different weights for different faculty designations."),
    page(16, "Emiratization: 25% of the UoS faculty will be Emiratis, contributing on a world stage in research and innovation."),
    page(150, "If the number of College Academic Promotions Committee members holding a rank equal to the applicant's is insufficient, the Dean appoints external members."),
    page(54, "Human Resources +971 6 5050023 +971 6 5585200"),
    page(192, "10.6 Learning Management System (LMS) Policy", "The IT department administers the LMS and provides Blackboard support to faculty members."),
    page(240, "13.8 Laboratory Safety", "All laboratory users, including students and staff, must receive safety training before working in research laboratories."),
]

#: Focused page groups: run-based tests use the pages of one policy area so
#: that the hashed stand-in embedder's recall never decides a test.
GROUPS = {
    "all": None,
    "supervision": (80, 81, 82),
    "conference": (117, 118, 119),
    "evaluation": (123, 124, 126),
    "promotion": (144, 145, 147, 148, 150),
    "grants": (153, 154, 156, 157, 159, 160, 161, 163, 164, 165, 166, 170, 173, 174),
    "ip": (175, 176, 177, 178, 179, 180, 181, 182),
    "consultancy": (259, 260, 261, 263, 265, 266),
}


class RawOverlapReranker:
    """Raw token overlap from the shared tokenizer; strongly negative when there is none."""

    def predict(self, pairs):
        return [float(len(set(tokenize(q)) & set(tokenize(t)))) or -5.0 for q, t in pairs]


class ScriptedReranker:
    """Predetermined scores: the first rule whose substring occurs in the chunk text wins."""

    def __init__(self, rules, default=-5.0):
        self.rules = list(rules)
        self.default = default

    def predict(self, pairs):
        return [float(next((s for needle, s in self.rules if needle in text), self.default)) for _q, text in pairs]


@pytest.fixture(scope="module")
def registry_and_map():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    return registry.get_source(APPROVED_SOURCE_ID), load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))


def corpus_from(registry_and_map, pages):
    source, section_map = registry_and_map
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


@pytest.fixture(scope="module")
def corpora(registry_and_map):
    cache = {}

    def get(group):
        if group not in cache:
            pages = RESEARCH_PAGES if GROUPS[group] is None else [p for p in RESEARCH_PAGES if p["page"] in GROUPS[group]]
            cache[group] = corpus_from(registry_and_map, pages)
        return cache[group]

    return get


@pytest.fixture(scope="module")
def corpus(corpora):
    return corpora("all")


def make_specialist(corpus, reranker=None, **kw):
    res = ResearchResources(embed=lambda t: corpus.embedder.encode([t]), rerank=(reranker or RawOverlapReranker()).predict,
                            index=corpus.index, chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source, section_map=corpus.section_map)
    return ResearchInnovationSpecialist(res, **kw)


@pytest.fixture(scope="module")
def specialist(corpus):
    return make_specialist(corpus)


@pytest.fixture(scope="module")
def specialists(corpora):
    cache = {}

    def get(group):
        if group not in cache:
            cache[group] = make_specialist(corpora(group))
        return cache[group]

    return get


def task(question, *, clauses=None, system=None, intent="policy", task_id="task-1", specialist_id="research", **extra):
    context = {"full_question": question, "focus_clauses": list(clauses) if clauses else [question]}
    return SpecialistTask(task_id=task_id, question=question, specialist_id=specialist_id, domain="research",
                          intent=intent, system=system, context=context, requested_by="coordinator", **extra)


def run(sp, question, **kw):
    t = task(question, **kw)
    return check_findings(sp, t, sp.run(t))


def quotes(result):
    return [f.evidence_quote for f in result.findings]


def assert_traceable(result, corpus):
    for f in result.findings:
        chunk, meta = corpus.chunks[f.metadata["chunk_id"]], corpus.metadata[f.metadata["chunk_id"]]
        assert f.evidence_quote and f.evidence_quote in chunk and f.claim == f.evidence_quote
        assert f.source_id == meta["source_id"] == APPROVED_SOURCE_ID and f.source_title == meta["source_title"]
        assert f.page == meta["page"] and f.metadata["rerank_score"] >= MIN_RERANK_SCORE
        assert f.metadata["section_label_page_level"] is True and f.metadata["extractive"] is True
        assert "page_section_nos" in f.metadata and "boundary_guard" in f.metadata


def supported_with(sp, question, needle, **kw):
    result = run(sp, question, **kw)
    assert result.status is FindingStatus.SUPPORTED, (question, result.status, result.missing, quotes(result))
    assert any(needle in q for q in quotes(result)), (needle, quotes(result))
    return result


# ---------------------------------------------------------------------------
# Import safety, contract, resources
# ---------------------------------------------------------------------------
def test_importing_research_loads_no_model_and_registers_nothing():
    code = ("import sys; import handbook_bot.agents.specialists.research as r; "
            "print('HEAVY', sorted(m for m in ('sentence_transformers','torch','transformers','groq','huggingface_hub') if m in sys.modules)); "
            "import handbook_bot.agents.specialists.registry as reg; "
            "print('REG', [k for k, v in vars(r).items() if isinstance(v, reg.SpecialistRegistry)]); "
            "print('TEACHING', 'handbook_bot.agents.specialists.teaching' in sys.modules)")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.startswith(("HEAVY", "REG", "TEACHING"))]
    assert lines == ["HEAVY []", "REG []", "TEACHING False"]


def test_research_never_imports_teaching_or_an_llm():
    source = (REPO / "handbook_bot" / "agents" / "specialists" / "research.py").read_text(encoding="utf-8")
    assert "from .teaching" not in source and "import teaching" not in source
    imports = [l for l in source.splitlines() if l.startswith(("from ", "import "))]
    assert not any(w in l.lower() for l in imports for w in ("groq", "openai", "requests", "httpx", "urllib", "sentence_transformers", "synthesis", "verifier"))


def test_specialist_implements_the_contract_and_registers(specialist):
    assert isinstance(specialist, Specialist) and specialist.specialist_id is SpecialistId.RESEARCH
    assert specialist.display_name == "Research & Innovation"
    registry = SpecialistRegistry()
    assert registry.register(specialist) is SpecialistId.RESEARCH and registry.resolve("research") is specialist


def test_task_assigned_to_another_specialist_is_rejected(specialist):
    with pytest.raises(ValueError):
        specialist.run(task("What are the research priority areas?", specialist_id="teaching"))
    with pytest.raises(TypeError):
        specialist.run("not a task")


def test_specialist_refuses_an_unregistered_source(corpus):
    res = ResearchResources(embed=lambda t: corpus.embedder.encode([t]), rerank=RawOverlapReranker().predict, index=corpus.index,
                            chunks=corpus.chunks, metadata=corpus.metadata, source=unregistered_source("other.pdf"), section_map=corpus.section_map)
    with pytest.raises(ValueError):
        ResearchInnovationSpecialist(res)


def test_from_knowledge_base_wires_a_loaded_document(corpora):
    corpus = corpora("grants")

    class Emb:
        def encode(self, texts, **kw):
            return corpus.embedder.encode(texts)

    kb = SimpleNamespace(source=corpus.source, stats={"source": {"section_records": 206}}, embedder=Emb(), reranker=RawOverlapReranker(),
                         index=corpus.index, chunks=corpus.chunks, metadata=corpus.metadata)
    sp = ResearchInnovationSpecialist.from_knowledge_base(kb)
    assert run(sp, "What are the research priority areas?").status is FindingStatus.SUPPORTED
    with pytest.raises(ValueError):
        ResearchResources.from_knowledge_base(SimpleNamespace(source=corpus.source, stats={"source": {}}))


# ---------------------------------------------------------------------------
# Scope and section guard
# ---------------------------------------------------------------------------
def test_scope_is_the_approved_page_ranges(registry_and_map, specialist):
    scope = compile_scope(registry_and_map[1])
    assert set(scope.source_ids) == {APPROVED_SOURCE_ID}
    assert list(scope.page_ranges) == [(80, 82), (117, 119), (123, 124), (144, 145), (147, 148), (153, 182), (259, 266)]
    assert specialist.scope.page_ranges == scope.page_ranges and EXTRA_PAGES == ((175, 175),)
    assert len(OWNED_SECTIONS) == 27 and len(SHARED_SECTIONS) == 3 and SCOPE_SECTIONS == OWNED_SECTIONS + SHARED_SECTIONS


@pytest.mark.parametrize("page_no", [80, 82, 117, 119, 123, 124, 144, 148, 153, 163, 175, 180, 182, 259, 266])
def test_owned_pages_are_inside_the_scope(specialist, page_no):
    assert specialist.scope.allows({"source_id": APPROVED_SOURCE_ID, "page": page_no})


@pytest.mark.parametrize("page_no", [79, 96, 107, 109, 112, 116, 120, 122, 125, 143, 146, 149, 150, 152, 183, 240, 250, 258, 267, 268])
def test_excluded_pages_are_outside_the_scope(specialist, page_no):
    assert not specialist.scope.allows({"source_id": APPROVED_SOURCE_ID, "page": page_no})


def test_excluded_sections_are_really_excluded(registry_and_map):
    scope = compile_scope(registry_and_map[1])
    owned = {(c, n) for c, n, _ in SCOPE_SECTIONS}
    for chapter, number, _ in EXCLUDED_SECTIONS:
        assert (chapter, number) not in owned
    for p in (89, 90, 93, 96, 97, 107, 108, 109, 110, 149, 150, 151):
        assert not scope.allows({"source_id": APPROVED_SOURCE_ID, "page": p})


def test_extra_page_175_is_admitted_and_never_widens(registry_and_map):
    section_map = registry_and_map[1]
    assert compile_scope(section_map, extra_pages=()).allows({"source_id": APPROVED_SOURCE_ID, "page": 175}) is False
    assert compile_scope(section_map).allows({"source_id": APPROVED_SOURCE_ID, "page": 175}) is True
    with pytest.raises(ValueError):
        compile_scope(section_map, owned=OWNED_SECTIONS, extra_pages=((124, 124),))   # 6.5 to 6.7 are not owned
    with pytest.raises(ValueError):
        compile_scope(section_map, extra_pages=((96, 96),))                            # WLAM is Teaching's


def test_scope_compilation_fails_loudly(registry_and_map):
    with pytest.raises(ValueError):
        compile_scope(registry_and_map[1], owned=SCOPE_SECTIONS + ((8, "8.12", "missing"),))
    with pytest.raises(ValueError):
        compile_scope(registry_and_map[1], source_id="other_source")


def test_guard_resolves_every_shared_page_of_the_fixture(specialist):
    assert sorted(specialist.guard.shared_pages) == [80, 82, 117, 119, 123, 124, 144, 145, 147, 148, 266]
    assert specialist.guard.unresolved_pages == []
    for p in (163, 175, 180):
        assert p not in specialist.guard.shared_pages                        # owned-only or preamble pages are admitted whole


@pytest.mark.parametrize("group,question,forbidden", [
    ("supervision", "Is consultancy allowed for faculty?", "resource person"),                  # 3.6 on page 82
    ("conference", "How much per diem is paid for conference travel?", "basic salary"),         # 5.5 on page 119
    ("evaluation", "How is research weighted in the annual evaluation?", "30 percent"),         # 6.7 on page 124
    ("promotion", "What research counts for promotion?", "student evaluations"),                # 7.3 on page 145
    ("promotion", "Which journals count for promotion?", "promotion committee"),                # 7.8 on page 148
    ("consultancy", "Who approves research on human subjects?", "sexual harassment"),           # 16.5 on page 266
    ("supervision", "What are my research responsibilities?", "office hour"),                   # 3.4 on page 80
])
def test_excluded_section_text_on_a_shared_page_is_never_evidence(corpora, group, question, forbidden):
    corpus = corpora(group)
    sp = make_specialist(corpus, ScriptedReranker([(forbidden, 9.0)], default=0.5))
    result = run(sp, question)
    assert all(forbidden not in q for q in quotes(result)), quotes(result)
    assert_traceable(result, corpus)


def test_synthetic_owned_excluded_owned_page_is_attributed_by_heading(registry_and_map):
    pages = [page(145, "C. Patents registered with recognized patent offices.",
                  "7.3 Teaching and Academic Advising",
                  "Teaching effectiveness is assessed through student evaluations and peer observation of teaching.")]
    corpus = corpus_from(registry_and_map, pages)
    sp = make_specialist(corpus, ScriptedReranker([], default=9.0))
    result = run(sp, "Do patents count as research for promotion?")
    assert result.status is not FindingStatus.NOT_FOUND
    assert all("student evaluations" not in q for q in quotes(result))
    assert any("Patents registered" in q for q in quotes(result))


def test_page_175_findings_carry_an_honest_boundary_label(specialists):
    result = run(specialists("ip"), "Who manages the technology transfer office?")
    hits = [f for f in result.findings if f.page == 175]
    assert hits, quotes(result)
    for f in hits:
        assert f.metadata["chapter"] == 9 and f.metadata["section_no"] == "9" and f.metadata["page_section_nos"] == ["9"]
        assert f.metadata["boundary_guard"] == "chapter preamble: page carries no section record"


def test_owned_only_shared_pages_keep_page_level_metadata(specialists):
    result = run(specialists("ip"), "What percentage of licensing revenue goes to the inventor?")
    f = next(x for x in result.findings if x.page == 181)
    assert f.metadata["page_section_nos"] == ["9.7", "9.8", "9.9"] and f.metadata["shared_page"] is False
    assert f.metadata["boundary_guard"] == "page carries owned sections only"


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,expected", [
    ("What categories of research grants exist?", (True, None)),
    ("How do I apply for internal funding?", (True, None)),
    ("Do I need ethics approval for a survey?", (True, None)),
    ("Who owns a patent developed at the university?", (True, None)),
    ("What does the Technology Transfer Office do?", (True, None)),
    ("How many theses can I supervise at once?", (True, None)),
    ("Who supervises postgraduate research?", (True, None)),
    ("Can I do paid consultancy?", (True, None)),
    ("What is the policy on research misconduct?", (True, None)),
    ("What are the research priority areas?", (True, None)),
    ("What research output is expected of research-intensive faculty?", (True, SpecialistId.TEACHING)),
    ("Can research reduce my teaching load?", (True, SpecialistId.TEACHING)),
    ("Can I take a sabbatical for research?", (True, SpecialistId.FACULTY_SERVICES)),
    ("What research counts for promotion?", (True, SpecialistId.FACULTY_SERVICES)),
    ("Who handles research grants and what is their phone number?", (True, SpecialistId.INSTITUTIONAL)),
    ("Where is the research office?", (True, SpecialistId.INSTITUTIONAL)),
    ("What is the teaching load for research-intensive faculty?", (False, SpecialistId.TEACHING)),
    ("How many credits does research-intensive faculty teach?", (False, SpecialistId.TEACHING)),
    ("How is thesis supervision calculated?", (False, SpecialistId.TEACHING)),
    ("How many thesis supervision credits count toward workload?", (False, SpecialistId.TEACHING)),
    ("Am I eligible to supervise theses?", (False, SpecialistId.TEACHING)),
    ("How many days of annual leave do I get?", (False, SpecialistId.FACULTY_SERVICES)),
    ("What is the promotion procedure at college level?", (False, SpecialistId.FACULTY_SERVICES)),
    ("Where is the HR department?", (False, SpecialistId.INSTITUTIONAL)),
    ("What is the parking fee?", (False, None)),
    ("Hello there", (False, None)),
])
def test_clause_classification(clause, expected):
    assert classify_clause(clause) == expected


@pytest.mark.parametrize("question", [
    "What is the teaching load for research-intensive faculty?", "How many credits does research-intensive faculty teach?",
    "What is the WLAM category for active research faculty?", "How many thesis supervision credits count toward workload?",
    "Am I eligible to supervise theses?",
])
def test_wlam_and_supervision_credit_clauses_are_teachings_even_when_routed_here(specialist, question):
    result = run(specialist, question)
    assert result.status is FindingStatus.OUT_OF_SCOPE and result.findings == []
    assert result.requested_agents == [SpecialistId.TEACHING]
    assert result.handoff_requests[0].metadata["kind"] == "misrouted" and result.handoff_requests[0].task.context["focus_clauses"] == [question]


@pytest.mark.parametrize("question,target", [
    ("How many days of annual leave do I get?", SpecialistId.FACULTY_SERVICES),
    ("What is the promotion procedure at college level?", SpecialistId.FACULTY_SERVICES),
    ("Where is the HR department?", SpecialistId.INSTITUTIONAL),
    ("What is the phone number of the registrar?", SpecialistId.INSTITUTIONAL),
])
def test_misrouted_clauses_are_out_of_scope_with_a_handoff(specialist, question, target):
    result = run(specialist, question)
    assert result.status is FindingStatus.OUT_OF_SCOPE and result.requested_agents == [target]
    assert result.handoff_requests[0].from_specialist is SpecialistId.RESEARCH and result.handoff_requests[0].task.requested_by == "research"


def test_unowned_clause_is_reported_not_searched(specialist):
    result = run(specialist, "What is the parking fee?")
    assert result.status is FindingStatus.OUT_OF_SCOPE and result.handoff_requests == []
    assert any("no research cue" in m for m in result.missing)


def test_shared_clause_is_answered_and_handed_off(specialists, corpora):
    result = run(specialists("grants"), "Can I take a sabbatical for research?")
    assert result.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIAL) and result.findings
    assert result.requested_agents == [SpecialistId.FACULTY_SERVICES] and result.handoff_requests[0].metadata["kind"] == "shared"
    assert any("sabbatical" in q.lower() for q in quotes(result))
    assert_traceable(result, corpora("grants"))


def test_research_output_of_a_category_is_answered_with_a_teaching_handoff(specialist):
    result = run(specialist, "What research output is expected of research-intensive faculty?")
    assert result.requested_agents == [SpecialistId.TEACHING]
    assert all("credit hours" not in q for q in quotes(result))                 # the WLAM row is outside the scope


def test_full_question_is_never_rescanned(specialists):
    result = run(specialists("grants"), "What are the research priority areas, and how many days of annual leave do I get?",
                 clauses=["What are the research priority areas"])
    assert result.status is FindingStatus.SUPPORTED and result.handoff_requests == []


def test_research_never_calls_the_coordinator(specialists, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("coordinator must not be called by the specialist")

    monkeypatch.setattr(coordinator, "analyze", boom)
    monkeypatch.setattr(coordinator, "coordinate", boom)
    assert run(specialists("grants"), "What are the research priority areas?").status is FindingStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,anchor", [
    ("How do I apply for a research grant?", "grant"), ("Is external funding available?", "funding"), ("Do I need research ethics approval?", "research_ethics"),
    ("Does the IRB review surveys?", "irb"), ("Does the IACUC review animal studies?", "iacuc"), ("Who owns intellectual property?", "ip"),
    ("Can I file a patent?", "patent"), ("What does the TTO do?", "technology_transfer"), ("How are royalties shared?", "commercialisation"),
    ("What must an inventor disclose?", "inventor"), ("What are the research institutes?", "research_institute"), ("How do I join a research group?", "research_group"),
    ("Who chairs the University Research Board?", "research_board"), ("What do research committees do?", "research_committee"),
    ("How are research projects funded?", "research_project"), ("What may a principal investigator claim?", "principal_investigator"),
    ("Which journals count?", "publishing"), ("Can I do consultancy?", "consultancy"), ("What is research misconduct?", "misconduct"),
    ("What are the research priority areas?", "research_priority"), ("Is conference travel funded?", "conference"), ("Can I hire a research assistant?", "research_assistant"),
    ("How many theses can I supervise?", "supervision"), ("What are the research regulations?", "research_policy"),
])
def test_research_anchors(text, anchor):
    assert anchor in concepts(text).anchors and anchor in ANCHOR_CONCEPTS


def test_generic_terms_carry_no_concept():
    for term in sorted(GENERIC_TERMS):
        assert concepts(term).vocab == frozenset(), term
    assert ANCHOR_CONCEPTS <= VOCABULARY_CONCEPTS
    assert {"research", "project", "student", "period", "approval", "fund", "ethics"} <= VOCABULARY_CONCEPTS - ANCHOR_CONCEPTS


@pytest.mark.parametrize("clause,text", [
    ("Is external funding available for research?", "Funding for training is provided by the HR department."),
    ("How are research projects funded?", "Is there a training project for new faculty?"),
    ("Which journals count for research?", "Publish your course materials on Blackboard before the first lecture."),
    ("Is conference travel funded?", "Video conference facilities are available in every classroom."),
    ("What is the research ethics process?", "Professional ethics and disciplinary actions apply to all employees."),
    ("Who supervises postgraduate research?", "Student advising is a core teaching responsibility of every faculty member."),
    ("What are the research priority areas?", "Emiratization: 25% of the UoS faculty will be Emiratis by 2030."),
    ("Which journals count for promotion?", "Library journal circulation is limited to two weeks per loan."),
])
def test_false_positive_controls_are_not_relevant(clause, text):
    assert not is_relevant(concepts(clause), text), (clause, text)


@pytest.mark.parametrize("clause,text", [
    ("What percentage of licensing revenue goes to the inventor?", "the net royalty revenue shall be distributed as follows: 50% to the inventor(s), and 50% to the University."),
    ("How much per diem is paid for conferences?", "Per diem rates between AED 1000-1500 for a maximum of four days, depending on the conference location."),
    ("Who approves conference travel?", "The Vice Chancellor for Academic Affairs approves travel requests and interprets the policy."),
    ("What are the research regulations?", "Faculty members must report potential conflicts at the proposal stage of any research project."),
    ("Is consultancy allowed?", "All proposed faculty consultancy activities must receive prior review and approval."),
    ("Which journals are classified as class A?", "Journals in SCOPUS database are classified as A if their SJR metric ranks in the top 20% of their field."),
])
def test_handbook_evidence_is_relevant(clause, text):
    assert is_relevant(concepts(clause), text)


# ---------------------------------------------------------------------------
# Requested details
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,families", [
    ("How much per diem is paid for conferences?", {"amount"}),
    ("What is the maximum number of theses I can supervise?", {"quantity", "maximum"}),
    ("When must a patent decision be made?", {"deadline"}),
    ("How long is the term of office of the chief editor?", {"duration"}),
    ("How often can I attend a funded conference?", {"frequency"}),
    ("What is the minimum service before applying?", {"minimum"}),
    ("What percentage of licensing revenue goes to the inventor?", {"percentage"}),
    ("Who approves conference travel?", {"authority"}),
    ("Who manages research grants?", {"responsibility"}),
    ("Am I eligible for conference funding?", {"eligibility"}),
    ("What documents must I attach to a conference request?", {"documents"}),
    ("What categories of research grants exist?", {"project_type"}),
    ("Who funds research at the university?", {"funding_source"}),
    ("What is the role of the principal investigator?", {"role"}),
    ("Which journals count for promotion?", {"publication_type"}),
    ("What must an inventor disclose?", {"requirement"}),
    ("What is the funding ceiling for a seed project?", {"amount", "category"}),
    # a party question asks for the party: the value noun in the object is not a requested value
    ("Who determines the funding ceiling for research grants?", {"authority"}),
    ("Which committee reviews research on human subjects?", {"authority"}),
    ("Who approves the per diem and how much is it?", {"authority", "amount"}),
    # a percentage question does not also ask for a money amount
    ("What is the maximum research advance as a percentage of the grant?", {"maximum", "percentage"}),
    ("How many theses can I supervise at once?", {"quantity"}),
    ("What are the research priority areas?", set()),
])
def test_requested_detail_families(clause, families):
    assert {d.family for d in requested_details(clause)} == families


@pytest.mark.parametrize("unit,expected", [
    ("The terms of office are two years, renewable once.", True), ("a maximum of four days", True), ("should not exceed two months", True),
    ("no later than 90 calendar days from receipt", True), ("As of Fall 2024, there are 62 active research groups", False),
    ("see section 8.4 of the regulations", False), ("item 20 of the procedures", False), ("page 163 of the handbook", False),
])
def test_duration_needs_a_number_and_a_time_unit(unit, expected):
    assert research._duration_ok(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("The TTO shall decide no later than 90 calendar days from receipt of a complete invention disclosure.", True),
    ("Requests must be submitted at least one month before the conference date.", True),
    ("Applications can be submitted from the beginning of the academic year until the end of the second semester.", True),
    ("As of Fall 2024, there are 62 active research groups at UoS.", False),
    ("The research regulations were approved in 2019.", False),
    ("The deadline is described in section 8.4.", False),
    ("See item 20 of the research procedures.", False),
])
def test_deadline_needs_a_timing_expression_never_a_bare_year(unit, expected):
    # the timing SHAPE only; binding the timing to the asked object (a grant application) is tested in
    # test_research_corrections.py (Step 4.5A, M4-F8), where the TTO decision unit no longer answers this clause
    deadline = next(d for d in requested_details("When is the grant application deadline?") if d.family == "deadline")
    assert deadline.value(unit) is expected, unit


@pytest.mark.parametrize("unit,expected", [
    ("Applicants must be regular faculty members who have passed the probationary period.", True),
    ("Faculty are entitled to participate in one local and one external conference per year.", True),
    ("Researchers may apply for this funding type only once, during their first year.", True),
    ("The participant must be the main or joint author of the paper.", True),
    ("The University provides financial support to faculty.", False),
    ("Conference participation is encouraged.", False),
])
def test_eligibility_evidence(unit, expected):
    assert research._eligibility_ok(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("A copy of the conference flier with fees stated.", True), ("The abstract acceptance letter and a copy of the paper.", True),
    ("Participants must submit a report to their Department Chair.", True), ("Complete the research grants forms.", True),
    ("Submit an application to the VCRGS.", False), ("Requests must be submitted at least one month before the conference.", False),
])
def test_required_documents_need_a_named_artefact(unit, expected):
    assert research._documents_ok(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("The Vice Chancellor for Academic Affairs approves travel requests.", True),
    ("The VCRGS determines the funding ceiling based on the Board's recommendation.", True),
    ("Research involving animals must be reviewed and approved by the Institutional Animal Care and Use Committee.", True),
    ("The TTO shall decide whether to file a patent application.", True),
    ("Prior approval is required for all consultancy activities.", False),
    ("Approval must be obtained before the conference.", False),
])
def test_authority_needs_a_named_body(unit, expected):
    assert research._authority_ok(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("The funding ceiling for each seed project is AED 5000.", True), ("Conference registration fees up to $750.", True),
    ("support is up to AED 700 for a maximum of three days", True), ("Per diem rates between AED 1000-1500", True),
    ("fee applies to examination 101", False), ("payment of a fixed fee", False),
    ("Conferences attended at no cost to the University are discretionary.", False),   # not an amount of funding
    ("The service is provided free of charge.", False),
])
def test_amounts_need_a_numeric_monetary_value(unit, expected):
    assert research._amount_ok(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("50% to the inventor(s), and 50% to the University", True), ("a research advance of up to 10% of the grant", True),
    ("The research score shall be weighted at 30 percent", True),
    ("Editors have their teaching loads reduced based on a percentage determined by the Chancellor.", False),
    ("The percentage is set annually by the Board.", False),
])
def test_percentages_need_a_numeric_value(unit, expected):
    assert research._percentage_ok(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("a maximum of 30% of the project's budget", True), ("a maximum of four theses", True), ("should not exceed two months", True),
    ("Conference registration fees up to $750.", True), ("support is up to AED 700 for local conferences", True),
    ("a petty cash advance of up to AED 5000", True),
    ("a maximum in 2024", False), ("the maximum is in section 8.4", False), ("maximum item 20", False),
    ("the maximum is set by the Board", False), ("up to the College Dean", False),
])
def test_maximum_needs_a_value_shaped_number(unit, expected):
    assert research._max_ok(unit) is expected


@pytest.mark.parametrize("unit,expected", [
    ("a minimum of AED 500 per project", True), ("at least one month before the conference", True),
    ("a minimum in 2024", False), ("at least section 8.4 applies", False),
])
def test_minimum_needs_a_value_shaped_number(unit, expected):
    assert research._min_ok(unit) is expected


def test_maximum_registration_fee_is_supported_from_a_currency_first_amount(specialists, corpora):
    result = supported_with(specialists("conference"), "What is the maximum conference registration fee covered?", "up to $750")
    assert {"maximum or limit", "amount or cost"} <= set(result.findings[0].metadata["details_covered"])
    assert_traceable(result, corpora("conference"))


def test_quantity_counts_research_nouns():
    clause = "How many research groups are active?"
    quantity = next(d for d in requested_details(clause) if d.family == "quantity")
    assert detail_covered(quantity, concepts(clause), "As of Fall 2024, there are 62 active research groups at UoS.")
    assert not detail_covered(quantity, concepts(clause), "Research groups were established in 2024.")
    theses = "How many theses can I supervise at once?"
    q2 = next(d for d in requested_details(theses) if d.family == "quantity")
    assert detail_covered(q2, concepts(theses), "A faculty member is allowed to supervise a maximum of four theses/dissertations simultaneously.")
    assert not detail_covered(q2, concepts(theses), "A faculty member may supervise theses as listed in item 5.")


def test_details_are_never_combined_across_units():
    clause = "How much is the petty cash advance and when must it be settled?"
    details = requested_details(clause)
    assert {d.family for d in details} == {"amount", "deadline"}
    joined = "The principal investigator is entitled to a petty cash advance of up to AED 5000, subject to settlement within 60 days."
    split = "The principal investigator is entitled to a petty cash advance of up to AED 5000. The advance must be settled within 60 days."
    assert research.unit_coverage(details, concepts(clause), joined) == frozenset({"amount", "deadline"})
    assert research.unit_coverage(details, concepts(clause), split) == frozenset({"amount"})


# ---------------------------------------------------------------------------
# Supported questions from the fixture (real handbook wording)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("group,question,needle,covered", [
    ("ip", "What percentage of licensing revenue goes to the inventor?", "50% to the inventor(s)", "percentage or share"),
    ("conference", "How much per diem is paid for conference travel?", "AED 1000-1500", "amount or cost"),
    ("conference", "Who approves conference travel requests?", "Vice Chancellor for Academic Affairs approves", "approving authority"),
    ("grants", "Who determines the funding ceiling for research grants?", "VCRGS determines the funding ceiling", "approving authority"),
    # "which committee reviews" is an authority question in the approved design ("review" is an authority verb)
    ("consultancy", "Which committee reviews research on human subjects?", "Institutional Review Board (IRB)", "approving authority"),
    ("grants", "What categories of research grants exist?", "seed projects, competitive projects", "research category or project type"),
    ("conference", "Who is eligible for conference funding?", "Faculty are entitled to participate", "eligibility conditions"),
    ("supervision", "How many theses can I supervise at once?", "maximum of four theses", "number or amount"),
    ("grants", "How long is the term of office of the chief editor?", "two years", "duration"),
    ("grants", "Who funds research externally?", "National Science Foundation", "funding source"),
    ("conference", "What documents must I attach to a conference request?", "copy of the conference flier", "required documents"),
    ("promotion", "Which journals are classified as class A?", "SCOPUS", "publication type"),
    # item 21 states the rule ("up to 10%"), item 22 the exception ("raised to a maximum of 30%"); either states a
    # maximum percentage of the grant in one unit, and item 21 shares more of the clause's concepts
    ("grants", "What is the maximum research advance as a percentage of the grant?", "% of the basic research project grant", "maximum or limit"),
    ("grants", "Who determines the funding ceiling for research grants?", "VCRGS determines the funding ceiling", "approving authority"),
    ("grants", "Who manages research grants?", "Research Funding Department (RFD) is responsible", "responsible party or unit"),
    ("ip", "When must the TTO decide on a patent application?", "90 calendar days", "deadline or date"),
    ("grants", "What is the funding ceiling for a seed project?", "AED 5000", "amount or cost"),
])
def test_supported_research_questions(specialists, corpora, group, question, needle, covered):
    result = supported_with(specialists(group), question, needle)
    assert result.missing == [] and result.confidence == 1.0
    assert any(covered in f.metadata["details_covered"] for f in result.findings), [f.metadata["details_covered"] for f in result.findings]
    assert_traceable(result, corpora(group))


@pytest.mark.parametrize("group,question,needle", [
    ("grants", "What are the research priority areas?", "research priority areas"),
    ("grants", "What does the research funding department do?", "pre- and post-award management"),
])
def test_supported_topic_questions_without_requested_details(specialists, corpora, group, question, needle):
    result = supported_with(specialists(group), question, needle)
    assert result.findings[0].metadata["details_covered"] == []
    assert_traceable(result, corpora(group))


# Step 4.5A (M4-F9): ownership, permission and focus questions were answered here as topic questions; they now ask
# for an explicit detail, and the same quotes stay fully supported because they state it.
@pytest.mark.parametrize("group,question,needle,label", [
    ("ip", "Who owns intellectual property created by faculty?", "shall own all Intellectual Property Rights", "ownership or rights holder"),
    ("grants", "Can research groups get sabbatical awards?", "sabbatical or financial awards", "a statement permitting or prohibiting it"),
    ("ip", "Does copyright protection cover software?", "computer software", "the specific focus asked about (software)"),
    # Step 4.5A round 2 (TOP-F1): a yes/no question asks for its property ("priority"); the same quote states it
    ("ip", "Is commercialisation of intellectual property a priority?", "Commercialization is a priority", "the specific focus asked about (priority)"),
    # Step 4.5A-R: "Do I need X?" asks whether X is required; the same quote states the requirement
    ("consultancy", "Do I need approval before consultancy?", "prior review and approval", "an explicit requirement statement about the subject"),
    # Step 4.5A (SAF-3): the HR subject of a shared clause must be stated too; the 7.2 lead-in states it
    ("promotion", "What research counts for promotion?", "A. Research published", "the Faculty Services & HR subject asked about (promotion)"),
])
def test_ownership_permission_and_focus_questions_are_supported_by_their_detail(specialists, corpora, group, question, needle, label):
    result = supported_with(specialists(group), question, needle)
    assert label in result.findings[0].metadata["details_covered"]
    assert_traceable(result, corpora(group))


# ---------------------------------------------------------------------------
# Partial, not_found, out_of_scope, error
# ---------------------------------------------------------------------------
def group_corpus(registry_and_map, group, drop=()):
    """The pages of one policy area with every line containing one of ``drop`` removed: the complete answer is
    absent, not merely ranked low."""
    pages = [p for p in RESEARCH_PAGES if GROUPS[group] is None or p["page"] in GROUPS[group]]
    return corpus_from(registry_and_map, [page(p["page"], *[l for l in p["rows"] if not any(n in l for n in drop)]) for p in pages])


@pytest.mark.parametrize("group,question,rules,missing_label,fabricated,drop", [
    ("conference", "How much conference funding is available?", [("financial support to eligible faculty", 9.0)], "amount or cost", "AED 1000", ()),
    # item 13 (the application window) shares a page with the forms item; it is removed so only related evidence remains
    ("grants", "When is the grant application deadline?", [("research grants forms", 9.0)], "deadline or date", "beginning of the academic year",
     ("Applications for all types of research grants",)),
    ("consultancy", "Who approves consultancy?", [("prior approval is required", 9.0)], "approving authority", "Department Chair and the College Dean", ()),
    ("grants", "Am I eligible for a seed project?", [("seed projects and competitive projects funded", 9.0)], "eligibility conditions", "must be", ()),
    ("grants", "What documents must I attach to a grant application?", [("Research committees review research grant applications", 9.0)], "required documents", "forms", ()),
    ("grants", "How long is a visiting researcher sabbatical?", [("Sabbatical Leave for Visiting Researchers", 9.0), ("sabbatical or financial awards", 8.0)], "duration", "two months", ()),
])
def test_related_but_incomplete_evidence_is_partial_not_supported(registry_and_map, group, question, rules, missing_label, fabricated, drop):
    corpus = group_corpus(registry_and_map, group, drop)
    sp = make_specialist(corpus, ScriptedReranker(rules, default=-5.0))
    result = run(sp, question)
    assert result.status is not FindingStatus.SUPPORTED and result.confidence < 1.0, (result.status, quotes(result))
    if result.status is FindingStatus.PARTIAL:
        assert any(missing_label in m for m in result.missing), result.missing
    assert all(fabricated not in q for q in quotes(result))
    assert_traceable(result, corpus)


def test_top_20_percent_journal_ranking_never_answers_the_royalty_share(corpora):
    sp = make_specialist(corpora("all"), ScriptedReranker([("top 20%", 9.0)], default=-5.0))
    result = run(sp, "What percentage of licensing revenue goes to the inventor?")
    assert result.status is not FindingStatus.SUPPORTED
    assert all("SCOPUS" not in q for q in quotes(result))


def test_evaluation_weight_never_answers_a_research_share_question(corpora):
    sp = make_specialist(corpora("evaluation"), ScriptedReranker([("weighted at 30 percent", 9.0), ("out of 100", 9.0)], default=-5.0))
    result = run(sp, "What percentage of the annual evaluation is research?")
    assert result.status is not FindingStatus.SUPPORTED
    assert all("30 percent" not in q and "100 %" not in q for q in quotes(result))
    assert result.requested_agents == [SpecialistId.FACULTY_SERVICES]


@pytest.mark.parametrize("question", [
    "Which portal do I use to submit a research grant?", "What is the maximum external grant amount?",
    "How do I register a patent with the national patent office?", "What is the exact internal grant deadline date?",
])
def test_not_found_when_the_handbook_has_no_usable_evidence(corpus, question):
    sp = make_specialist(corpus, ScriptedReranker([], default=-5.0))
    result = run(sp, question)
    assert result.status is FindingStatus.NOT_FOUND and result.findings == [] and result.confidence == 0.0
    assert any("clause 0" in m for m in result.missing)


def test_status_semantics_follow_the_contract(specialists):
    sp = specialists("grants")
    supported = run(sp, "What are the research priority areas?")
    assert supported.status is FindingStatus.SUPPORTED and supported.missing == [] and supported.confidence == 1.0
    partial = run(sp, "How do I apply for a research grant?")
    assert partial.status is FindingStatus.PARTIAL and partial.findings and partial.missing and partial.confidence == 0.5
    foreign = run(sp, "How many days of annual leave do I get?")
    assert foreign.status is FindingStatus.OUT_OF_SCOPE and foreign.findings == [] and foreign.confidence == 0.0


@pytest.mark.parametrize("stage", ["embed", "rerank", "index"])
def test_failures_become_error_results_with_redacted_messages(corpus, stage):
    def boom(*a, **k):
        raise RuntimeError(r"failure at C:\Users\PC\secret\index.faiss with gsk_ABCDEFGHIJKLMNOP")

    class Broken:
        ntotal = corpus.index.ntotal

        def search(self, *a, **k):
            boom()

    res = ResearchResources(embed=boom if stage == "embed" else (lambda t: corpus.embedder.encode([t])),
                            rerank=boom if stage == "rerank" else RawOverlapReranker().predict,
                            index=Broken() if stage == "index" else corpus.index,
                            chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source, section_map=corpus.section_map)
    result = run(ResearchInnovationSpecialist(res), "What are the research priority areas?")
    assert result.status is FindingStatus.ERROR and result.findings == []
    assert "Users" not in result.metadata["error"] and "gsk_ABC" not in result.metadata["error"] and "<path>" in result.metadata["error"]


def test_unapproved_source_scope_is_refused_without_retrieval(specialist):
    result = run(specialist, "What are the research priority areas?", source_scope=["grade_transfer_guide"])
    assert result.status is FindingStatus.ERROR and "not approved" in result.metadata["error"]


# ---------------------------------------------------------------------------
# Conference policy (5.4) detail families, independently
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,needle,label", [
    ("Who is eligible for conference support?", "entitled to participate", "eligibility conditions"),
    ("How often can I attend a university-supported conference?", "one local and one external", "frequency"),
    ("What is the per diem for conference travel?", "AED 1000-1500", "amount or cost"),
    ("What is the maximum number of days covered for a conference?", "maximum of four days", "maximum or limit"),
    ("How much is the conference registration fee covered?", "$750", "amount or cost"),
    ("How much support is given for a local conference?", "AED 700", "amount or cost"),
    ("Who approves conference travel?", "Vice Chancellor for Academic Affairs", "approving authority"),
    ("What documentation is required for a conference request?", "flier", "required documents"),
])
def test_conference_policy_details(specialists, corpora, question, needle, label):
    result = supported_with(specialists("conference"), question, needle)
    assert any(label in f.metadata["details_covered"] for f in result.findings)
    assert all("basic salary" not in q for q in quotes(result))
    assert_traceable(result, corpora("conference"))


def test_conference_deadline_stated_in_a_lead_in_is_quoted_with_its_list(specialists, corpora):
    """"Requests must be submitted at least one month before the conference date, including:" introduces the
    document list; the first item is quoted with it, so the deadline is quoted verbatim, never inferred."""
    result = run(specialists("conference"), "When must a conference request be submitted?")
    assert result.status is FindingStatus.SUPPORTED, (result.status, result.missing)
    quote = next(q for q in quotes(result) if "one month before the conference date" in q)
    assert "including: • A copy of the conference flier" in quote                     # the lead-in and its first item only
    assert "abstract acceptance" not in quote
    assert "deadline or date" in result.findings[0].metadata["details_covered"]
    assert_traceable(result, corpora("conference"))


def test_grant_application_window_answers_the_deadline_question(specialists, corpora):
    result = supported_with(specialists("grants"), "When is the grant application deadline?", "until the end of the second semester")
    assert "deadline or date" in result.findings[0].metadata["details_covered"]
    assert_traceable(result, corpora("grants"))


def test_lead_in_join_never_joins_a_heading_or_a_sibling():
    clause = concepts("What research counts for promotion?")
    text = "7.2 Research and Scholarship For promotion purposes, research considered admissible includes: A. Research published in refereed journals. B. Books published by academic publishers."
    got = select_quote(text, "paragraph", clause)
    assert got and got[0].endswith("refereed journals.") and "B. Books" not in got[0]
    heading = "8.7 Research Grants: A. Research published in refereed journals."
    assert select_quote(heading, "paragraph", concepts("What counts as research for promotion?")) is None   # a bare heading is not a lead-in
    sibling = "A. Research published in refereed journals for promotion purposes. B. Books published by academic publishers."
    got = select_quote(sibling, "paragraph", concepts("Do books count as research for promotion?"))
    assert got is None or not ("A. Research" in got[0] and "B. Books" in got[0])     # a sibling item is never joined


def test_summer_compensation_on_page_119_is_masked_by_the_guard(corpora):
    corpus = corpora("conference")
    sp = make_specialist(corpus, ScriptedReranker([("basic salary", 9.0)], default=-5.0))
    result = run(sp, "How much per diem is paid for conference travel?")
    assert all("basic salary" not in q for q in quotes(result))
    assert result.status is FindingStatus.SUPPORTED and any("AED 1000-1500" in q for q in quotes(result))   # the 5.4 text of the same window
    assert_traceable(result, corpus)


# ---------------------------------------------------------------------------
# Supervision, procedure safety, systems
# ---------------------------------------------------------------------------
def test_thesis_cap_is_research_and_credit_is_teaching(specialists):
    sp = specialists("supervision")
    cap = run(sp, "What is the maximum number of dissertations I can supervise?")
    assert cap.status is FindingStatus.SUPPORTED and any("four theses" in q for q in quotes(cap))
    credit = run(sp, "How many thesis supervision credits count toward workload?")
    assert credit.status is FindingStatus.OUT_OF_SCOPE and credit.requested_agents == [SpecialistId.TEACHING]
    projects = run(sp, "How are research projects supervised?")
    assert projects.status in (FindingStatus.SUPPORTED, FindingStatus.PARTIAL, FindingStatus.NOT_FOUND) and projects.handoff_requests == []


@pytest.mark.parametrize("clause,expected", [
    ("Where do I click to submit a grant in the research portal?", (True, "portal")),
    ("Which screen in MyUOS shows my research budget?", (True, "myuos")),
    ("How do I upload the invention disclosure form in the portal?", (True, "portal")),
    ("How do I apply for a research grant?", (False, None)),
    ("What are the research priority areas?", (False, None)),
])
def test_procedure_detector(clause, expected):
    assert is_procedure_request(clause, None) == expected


def test_procedure_detector_uses_the_task_system_hint():
    assert is_procedure_request("Where do I click to submit my proposal?", "portal") == (True, "portal")
    assert is_procedure_request("Where do I click to submit my proposal?", None) == (False, None)


@pytest.mark.parametrize("clause,expected", [
    ("How do I apply for a research grant?", True), ("What is the procedure for consultancy approval?", True), ("How to submit a grant proposal?", True),
    ("What happens after my grant application is submitted?", False), ("Who reviews research grant applications?", False), ("What is the funding ceiling?", False),
])
def test_process_request_detector(clause, expected):
    assert is_process_request(clause) is expected


def test_a_rule_stated_for_another_role_is_an_ambiguous_relation(corpora):
    # Step 4.5A-D: "Individuals subject to this Intellectual Property Policy are obligated to promptly disclose ..." names
    # no inventor; whether the rule is the inventor's is a semantic judgement for the Milestone 5 Verifier, so the
    # deterministic specialist quotes it as partial with an ambiguous relation (a documented false negative)
    corpus = corpora("ip")
    result = run(make_specialist(corpus, ScriptedReranker([("obligated to promptly disclose", 9.0)], default=-5.0)), "What must an inventor disclose?")
    assert result.status is FindingStatus.PARTIAL and result.metadata["clauses"][0]["relation"] == "ambiguous"
    assert any("obligated to promptly disclose" in q for q in quotes(result))
    assert any("Milestone 5 Verifier" in m for m in result.missing)
    assert_traceable(result, corpus)


def test_policy_process_is_partial_and_review_chain_can_be_supported(specialists, corpora):
    corpus = corpora("grants")
    # a real cross-encoder ranks the procedure page first; the token-overlap stand-in prefers item 21 ("may apply
    # for a research advance"), so the ranking is scripted here and the gates alone decide the status
    ranked = make_specialist(corpus, ScriptedReranker([("To apply for research grants", 9.0), ("Research committees review", 8.0)], default=-5.0))
    process = run(ranked, "How do I apply for a research grant?")
    assert process.status is FindingStatus.PARTIAL and any("policy stages only" in m for m in process.missing)
    assert any("research grants forms" in q or "Forward the application" in q for q in quotes(process))
    unranked = run(specialists("grants"), "How do I apply for a research grant?")
    assert unranked.status is FindingStatus.PARTIAL                                   # never promoted, whatever the ranking
    # Step 4.5A-D: the chain is supported by the statement that states it; "have THEM reviewed by research committees"
    # states the review of a pronoun, which the deterministic rules cannot bind, so on its own it stays partial
    chain = run(make_specialist(corpus, ScriptedReranker([("Research committees review", 9.0)], default=-5.0)), "Who reviews research grant applications?")
    assert chain.status is FindingStatus.SUPPORTED and chain.metadata["clauses"][0]["relation"] == "strong"
    assert any("Research committees review" in q for q in quotes(chain))
    assert "approving authority" in chain.findings[0].metadata["details_covered"]
    assert_traceable(chain, corpus)
    pronoun = run(make_specialist(corpus, ScriptedReranker([("have them reviewed by research committees", 9.0)], default=-5.0)),
                  "Who reviews research grant applications?")
    assert pronoun.status is FindingStatus.PARTIAL and pronoun.metadata["clauses"][0]["relation"] == "ambiguous"


@pytest.mark.parametrize("question,system", [
    ("Where do I click to submit a grant in the research portal?", "portal"),
    ("Which screen in MyUOS shows my research budget?", "myuos"),
    ("How do I upload the invention disclosure form in the portal?", "portal"),
])
def test_system_procedures_are_never_supported(specialist, question, system):
    result = run(specialist, question)
    assert result.status in (FindingStatus.NOT_FOUND, FindingStatus.PARTIAL)
    assert any(system in m and "step-by-step procedure" in m for m in result.missing)
    assert any("policy-level references to %s only" % system in l for l in result.limitations)
    assert all(not any(w in q.lower() for w in ("click", "screen", "button")) for q in quotes(result))


# ---------------------------------------------------------------------------
# Quote quality, traceability, determinism, limits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quote,kind,problem", [
    ("8.9 Scientific Publishing at the University of Sharjah Journal", "prose", "heading"),
    ("Academic periodicals and journals must meet the following conditions:", "prose", "lead-in"),
    ("subject to the following guidelines and conditions:", "prose", "lead-in"),
    ("A. Research published (or accepted for publication) in recognized specialized refereed journals.", "prose", None),
    ("Chapter 8: Scientific Research", "prose", "heading"),
    ("Chapter 9: Intellectual Properties Management", "prose", "heading"),
])
def test_quote_quality_rules(quote, kind, problem):
    assert quote_problem(quote, kind) == problem


def test_heading_rows_are_never_findings_even_when_ranked_first(corpora):
    sp = make_specialist(corpora("grants"), ScriptedReranker([("8.7 Research Grants", 9.0), ("8.3 Research Priority Areas", 9.0)], default=0.5))
    result = run(sp, "What categories of research grants exist?")
    assert result.findings
    assert all(q not in ("8.7 Research Grants", "8.3 Research Priority Areas", "Chapter 8: Scientific Research") for q in quotes(result))


def test_lettered_items_are_quoted_whole(specialists):
    """Item A is quoted whole, with the lead-in that says the list is about promotion, and never merged with its
    sibling item B."""
    result = run(specialists("promotion"), "What research counts for promotion?")
    quote = next((q for q in quotes(result) if "A. Research published (or accepted for publication) in recognized specialized refereed journals." in q), None)
    assert quote is not None, quotes(result)
    assert "For promotion purposes, research considered admissible includes: A. Research published" in quote
    assert all("B. Books" not in q or "A. Research" not in q for q in quotes(result))
    assert result.requested_agents == [SpecialistId.FACULTY_SERVICES]                # the promotion process stays HR's


def test_select_quote_is_an_exact_slice_and_respects_allowed_spans():
    text = "Faculty members may not do consultancy. 3.6 Responsibility to Community Service This may include serving as a consultant or research partner."
    cut = text.index("3.6 ")
    clause = concepts("Is consultancy allowed?")
    got = select_quote(text, "paragraph", clause, allowed=[(0, cut)])
    assert got and got[0] == "Faculty members may not do consultancy." and got[0] in text
    assert select_quote(text, "paragraph", clause, allowed=[(cut, len(text))]) is None


def test_row_join_limitation_is_bounded_by_locality(registry_and_map):
    """R-1 analogue: a bullet without terminal punctuation followed by a capitalised sentence forms one unit; the
    locality rule still applies inside it, and a detail present only in the joined bullet is recorded here."""
    pages = [page(166, "o Costs of research assistants and forfeiture of unused funds",
                  "The principal investigator may apply for a research advance of up to 10% of the basic research project grant.")]
    corpus = corpus_from(registry_and_map, pages)
    sp = make_specialist(corpus, ScriptedReranker([], default=9.0))
    result = run(sp, "What is the maximum research advance for a principal investigator?")
    assert result.status is not FindingStatus.NOT_FOUND
    for f in result.findings:
        assert f.evidence_quote in corpus.chunks[f.metadata["chunk_id"]]
        assert "10%" in f.evidence_quote                                     # the detail comes from the sentence, not the bullet
    penalty = run(sp, "What is the penalty for a principal investigator who overspends the research advance?")
    assert penalty.status is not FindingStatus.SUPPORTED or any("forfeiture" in q for q in quotes(penalty))


def test_runs_are_deterministic(specialist):
    for q in ("What percentage of licensing revenue goes to the inventor?", "How do I apply for a research grant?", "Can I take a sabbatical for research?"):
        outs = []
        for _ in range(5):
            d = run(specialist, q).to_dict()
            d.pop("ms", None)
            outs.append(d)
        assert all(o == outs[0] for o in outs), q


def test_multi_clause_task_reports_per_clause_outcomes(specialists):
    result = run(specialists("grants"), "What are the research priority areas and how many days of annual leave do I get?",
                 clauses=["What are the research priority areas", "how many days of annual leave do I get"])
    assert [c["outcome"] for c in result.metadata["clauses"]] == ["supported", "handed_off"]
    assert result.status is FindingStatus.SUPPORTED and result.requested_agents == [SpecialistId.FACULTY_SERVICES]


def test_clause_limit(specialist):
    clauses = ["What are the research priority areas", "who owns intellectual property", "how much is the per diem", "what are the research regulations"]
    result = run(specialist, " and ".join(clauses), clauses=clauses)
    assert result.metadata["unprocessed_clauses"] == clauses[3:] and any("not processed" in m for m in result.missing)


def test_missing_focus_clauses_fall_back_to_the_full_question(specialists):
    t = SpecialistTask(task_id="task-1", question="What are the research priority areas?", specialist_id="research", context={}, requested_by="coordinator")
    result = specialists("grants").run(t)
    assert result.status is FindingStatus.SUPPORTED and result.metadata["clauses"][0]["clause"] == "What are the research priority areas?"


def test_coordinator_task_is_accepted_as_is(specialists, corpora):
    decision = coordinator.coordinate("What percentage of licensing revenue goes to the inventor?")
    t = next(x for x in decision.subtasks if x.specialist_id is SpecialistId.RESEARCH)
    sp = specialists("ip")
    result = check_findings(sp, t, sp.run(t))
    assert result.status is FindingStatus.SUPPORTED and result.task_id == t.task_id
    assert_traceable(result, corpora("ip"))


def test_finding_metadata_is_complete(specialists):
    result = run(specialists("ip"), "What percentage of licensing revenue goes to the inventor?")
    f = result.findings[0]
    for key in ("chunk_id", "chunk_type", "chapter", "section_no", "section_title", "section_label_page_level", "page_section_nos", "shared_page",
                "boundary_guard", "clauses", "route", "rerank_score", "score_semantics", "relevance", "details_covered", "extractive"):
        assert key in f.metadata
    assert result.metadata["boundary"]["extra_pages"] == [[175, 175]] and result.metadata["scope"]
    assert result.llm_used is False and result.metadata["confidence_semantics"].startswith("ordinal")
