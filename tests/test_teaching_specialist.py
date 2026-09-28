"""Teaching & Learning Specialist, deterministic version 1 (corrected).

Everything here runs offline. The corpus is a set of synthetic handbook pages
whose lines mirror the real handbook (real page numbers, real section
headings printed as lines), chunked by the project's own chunker and
annotated with the real registry record and the real section map. Ranking is
either a raw token-overlap reranker (independent of the specialist's concept
normalisation) or a scripted reranker with predetermined scores, so a test
can put a semantically wrong chunk first and a relevant chunk last.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from handbook_bot.agents import coordinator
from handbook_bot.agents.specialists import teaching
from handbook_bot.agents.specialists.base import Specialist, check_findings
from handbook_bot.agents.specialists.contracts import FindingStatus, SpecialistFindings, SpecialistId, SpecialistTask
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.agents.specialists.teaching import (
    ANCHOR_CONCEPTS,
    APPROVED_SOURCE_ID,
    EXCLUDED_SECTIONS,
    GENERIC_TERMS,
    OWNED_SECTIONS,
    VOCABULARY_CONCEPTS,
    TeachingLearningSpecialist,
    TeachingResources,
    calendar_context,
    classify_clause,
    compile_scope,
    concepts,
    is_procedure_request,
    is_relevant,
    is_table_row,
    quote_problem,
    requested_term,
    select_quote,
)
from handbook_bot.chunking import build_chunks
from handbook_bot.config import MIN_RERANK_SCORE
from handbook_bot.sources import SectionMap, SectionRecord, annotate_metadata, load_section_map, load_source_registry, unregistered_source
from handbook_bot.text_utils import normalize_text, tokenize

from conftest import FakeEmbedder

REPO = Path(__file__).resolve().parents[1]
TITLE = "UOS Faculty Handbook 2025/26"


# ---------------------------------------------------------------------------
# Fixture pages (lines mirror the handbook; headings are printed as lines)
# ---------------------------------------------------------------------------
def page(number, *lines):
    lines = list(lines)
    return {"page": number, "lines": ["Page %d" % number] + lines, "rows": lines,
            "text": normalize_text(" ".join(["Page %d" % number] + lines)), "section_hint": "", "ocr_snippets": []}


PAGES = [
    page(49,
         "expectations of the global academic community.",                                # 1.13 tail
         "1.14 Academic Calendar", "Fall Semester 2025/ 2026", "Day Description",
         "Mon 18 Aug 24 Safar Return of Academic Staff", "Mon 25 Aug 02 Rabi I Classes begin",
         "Thu 28 Aug 05 Rabi I Last day for Add/Drop", "Mon - Thu 13-23 Oct 21 Rabi II- 01 Midterm exams",
         "Thu 04 Dec 13 Jumada II Classes end", "Sat -Tue 06-16 Dec 15 -25 Jumada II Final Exams"),
    page(50,
         "Spring Semester 2025/2026", "Day Description", "Thu - Sun 04 Dec - 03 Jan 13 Jumada II - 14 Online admission for Spring semester",
         "Mon 12 Jan 23 Rajab Classes begin", "Thu 15 Jan 26 Rajab Last day for Add/Drop", "Sat - Tue 02 - 12 May 15 -25 Dhi Al- Final exams",
         "Summer Semester 2025/ 2026", "Day Description", "Mon 01 June 15 Dhi Al-Hijja Classes begin", "Tue 02 June 16 Dhi Al-Hijja Last day for Add/Drop"),
    page(51,
         "Thu 16 July 02 Safar Summer classes end", "Sat - Thu 18 - 23 July 04 - 09 Safar Final exams",
         "Mon 24 Aug 11 Rabi I Classes begin for Fall 2026-2027",
         "1.15 Statement on Diversity, Equity, and Inclusion",
         "Diversity and social inclusion are deeply ingrained values within the University of Sharjah. These values are central to our teaching and research, reflecting the mission of the university."),
    page(75,
         "Chapter 3. Responsibilities of the Faculty", "3.1 Teaching Responsibilities",
         "Quality education lies at the core of University of Sharjah's mission statement. Faculty members must approach their teaching assignments with dedication to teaching excellence. The key teaching responsibilities of faculty members include the following:",
         "1. Responsibility for Teaching Excellence: Faculty members must prepare their courses carefully and deliver lectures that meet the stated learning outcomes."),
    page(79,
         "3.4 Responsibility to Office Hours",
         "Faculty members at the University of Sharjah embrace the responsibility to hold regular office hours outside of their scheduled instructional time. This commitment ensures that students have ample opportunities to seek guidance.",
         "1. Full-Time Faculty: Full-time faculty members in academic and career/technical programs are required to schedule a minimum of five office hours per week, spread over a minimum of two days.",
         "2. Part-Time Faculty: Part-time faculty members office hours will be prorated based on their teaching assignments."),
    page(80,
         "3. Consideration for Evening Courses: Faculty members assigned to teach evening courses should schedule at least one office hour during the evening.",
         "By upholding their responsibilities to office hours, faculty members foster a supportive academic environment.",
         "3.5 Responsibility to Research",
         "In addition to their commitment to quality education, faculty members at the University of Sharjah hold a fundamental responsibility to advance knowledge in their respective academic disciplines through research and creative endeavors.",
         "1. Staying Current with Developing Knowledge: Faculty members have teaching responsibilities and a research responsibility to stay abreast of the latest developments in their academic fields."),
    page(83,
         "Faculty members are encouraged to contribute to community service activities that benefit society and the teaching profession.",   # 3.6 tail
         "3.7 Professional Development Responsibilities",
         "These activities include workshops and training on teaching strategies.",                  # antecedent is the heading: not quotable alone
         "As part of their core responsibilities, faculty members at the University of Sharjah are expected to actively participate in professional development activities that support excellence in teaching, research, and service.",
         "Complete training modules on teaching strategies, Blackboard tools, curriculum design, and accreditation standards."),
    page(89,
         "3.11 Workload Allocation Model (WLAM)",
         "The Deans Council approved guidelines for assigning faculty workload, encompassing the teaching load in credit hours and contact hours for full-time faculty."),
    page(91,
         "Assignment by", "Special Assignment 1-9",                                           # WLAM release table fragments
         "Faculty with administrative duties receive a workload release approved by the Vice Chancellor for Academic Affairs."),
    page(96,
         'o "A" Regular Faculty (Assistant, Associate, Full Professor) is devoted to developing and delivering undergraduate and graduate courses, conducting research, and performing service. The baseline teaching load for this category is 12 credit hours per semester.',
         'o "B" Active Research Faculty has high research productivity compared to regular faculty. The teaching load for this category is 9 credit hours per semester.',
         'o "C" Research Intensive Faculty has exceptionally high research productivity. The teaching load is 6 credit hours per semester.',
         'o "D" Teaching Track Faculty are excellent in teaching and are expected to conduct pedagogic research. The teaching load is 15 credit hours per semester.',
         'o "E" Lecturer is mainly devoted to teaching and services. The teaching load for lecturers is 15 credit hours per semester.'),
    page(120,
         "Faculty teaching in the summer semester may not attend conferences or training outside the University during that period."),
    page(192,
         "Personal information, including contact numbers, addresses, email addresses, and other details, stored by any department shall not be shared with external parties without the consent of the information owner.",   # 10.5 tail
         "10.6 Learning Management System (LMS) Policy",
         "The purpose of this Policy is to address important considerations in the use of the Learning Management System (LMS) at the University of Sharjah."),
    page(193,
         "Attendance and Participation: Student attendance for online courses must follow the University policy on attendance.",
         "Course Backups: Instructors are recommended to regularly take backups of their courses in the LMS at the end of each semester.",
         "Course Enrollment: Instructors are enrolled in their LMS courses automatically. They become Course Instructors for the respective LMS courses."),
    page(221,
         "The curricula approval process requires the department council to approve any course modification before it is submitted to the college council.",   # 12.7 tail
         "12.8 Internship Policy:",
         "The University of Sharjah places significant emphasis on developing students' knowledge and skills through internship placements with industry partners.",
         "12.9 Undergraduate Completion Policy:",
         "Undergraduate students must complete all program courses and requirements to graduate."),
    page(222,
         "semesters to receive a bachelor degree.",                                            # 12.9 tail
         "12.10 Graduate Completion Requirements Policy:",
         "Before graduation, graduate students must meet all graduation requirements, which include successfully completing all program courses, the thesis, and/or essays as specified in the curriculum, and obtaining a minimum cumulative GPA of 2.5.",
         "12.11 Academic Progress Policy:",
         "• The minimum residency requirement for all undergraduate students is six regular semesters.",
         "12.12 Examinations Policy:", "12.12.1 Teaching and Evaluation",
         "• Instructors are required to prepare detailed syllabi that outline the course objectives, outcomes, content, teaching methods, evaluation criteria, references, and additional readings. These syllabi shall be distributed to students at the beginning of the semester and uploaded on Blackboard and kept in the course files within the college."),
    page(224,
         "Faculty members shall directly enter the grades electronically into the blackboard which will forward it automatically to the registration system (Banner). The Registration Department shall document and announce the results to students.",
         "Requests for re-marking of the final examination are subject to payment of a fixed fee.",
         "Instructors must correct final examination answer sheets and submit the results, documented in letter grades and percentages, to the Department Chair within forty-eight hours of the examination date."),
    page(227,
         "12.14 Grading System:",
         "All final course grades are evaluated numerically and converted to a point average according to the following grading system:",
         "Grades Percentages Points", "A 90-100 4.00", "B+ 85-89 3.50", "F Below 60 0.00"),
    page(228,
         "12.14.1 Cumulative Grade Point Average (CGPA): The CGPA is calculated based on all grade points earned in the courses studied.",
         "12.15 Student Attendance and Assessment Policies:",
         "1. Students are required to attend all theoretical lectures, laboratory hours, training sessions, research sessions, and examinations for the courses they are enrolled in.",
         "2. A student whose absences exceed 10% of the total class hours may be barred from the final examination.",
         "Late Detection of Cheating: If cheating is detected at a later stage, the offender will still be held accountable, and the case will be referred to the appropriate committee for investigation and determination of the suitable penalty."),
    page(229,
         "Instructors shall hand the syllabus to the department office and post it on Blackboard.",
         "12.18 Class Size Policy: The University of Sharjah prioritizes the efficient delivery of curricular programs.",
         "4. Office hours",
         "12.18.1 Maximum and Minimum Class Size: The enrollment limits for each course are determined by the level of student learning and the instructional method."),
    page(230,
         "Class Type Maximum Enrollment Minimum Enrollment Lecture 70-120 30 Tutorial 30-40 15 Laboratory 20 10",
         "The enrollment limits for each course are determined by various factors, including the instructional method and available classroom capacity."),
    page(232,
         "12.18.5 Direct Study/Independent Study: Directed/Independent Study refers to courses where students receive individual supervision from faculty.",
         "Such courses must have an appropriate syllabus, learning outcomes, teaching/learning methods, and assessment tools.",
         "12.19 Student Code of Honor Student Disciplinary Policy",
         "These rules apply to all students and violations are referred to the disciplinary committee."),
    # outside the Teaching scope
    page(107, "Faculty members are entitled to annual leave as specified in their contracts, subject to approval by the dean."),
    page(161, "Research projects involving human subjects require approval from the Research Ethics Committee before data collection begins."),
    page(54, "Human Resources +971 6 5050023 +971 6 5585200"),
    page(118, "Faculty members may apply for financial support to present papers at international conferences."),
]


class RawOverlapReranker:
    """Raw token overlap from the shared tokenizer, independent of the
    specialist's concept normalisation; strongly negative when there is none."""

    def predict(self, pairs):
        return [float(len(set(tokenize(q)) & set(tokenize(t)))) or -5.0 for q, t in pairs]


class ScriptedReranker:
    """Predetermined scores: the first rule whose substring occurs in the
    chunk text wins; other chunks get ``default``."""

    def __init__(self, rules, default=-5.0):
        self.rules = list(rules)
        self.default = default

    def predict(self, pairs):
        out = []
        for _query, text in pairs:
            score = next((s for needle, s in self.rules if needle in text), self.default)
            out.append(float(score))
        return out


@pytest.fixture(scope="module")
def registry_and_map():
    registry = load_source_registry(str(REPO / "knowledge" / "sources.json"))
    return registry.get_source(APPROVED_SOURCE_ID), load_section_map(str(REPO / "knowledge" / "handbook_sections.json"))


@pytest.fixture(scope="module")
def corpus(registry_and_map):
    source, section_map = registry_and_map
    chunks, metadata = build_chunks(PAGES)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


def make_specialist(corpus, reranker=None):
    res = TeachingResources(embed=lambda t: corpus.embedder.encode([t]), rerank=(reranker or RawOverlapReranker()).predict,
                            index=corpus.index, chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source, section_map=corpus.section_map)
    return TeachingLearningSpecialist(res)


@pytest.fixture(scope="module")
def specialist(corpus):
    return make_specialist(corpus)


def task(question, *, clauses=None, system=None, intent="policy", task_id="task-1", specialist_id="teaching", **extra):
    context = {"full_question": question, "focus_clauses": list(clauses) if clauses else [question]}
    return SpecialistTask(task_id=task_id, question=question, specialist_id=specialist_id, domain="teaching",
                          intent=intent, system=system, context=context, requested_by="coordinator", **extra)


def run(sp, question, **kw):
    t = task(question, **kw)
    return check_findings(sp, t, sp.run(t))


def assert_traceable(result, corpus):
    for f in result.findings:
        chunk, meta = corpus.chunks[f.metadata["chunk_id"]], corpus.metadata[f.metadata["chunk_id"]]
        assert f.evidence_quote and f.evidence_quote in chunk
        assert f.claim == f.evidence_quote
        assert f.source_id == meta["source_id"] == APPROVED_SOURCE_ID and f.source_title == meta["source_title"]
        assert f.page == meta["page"] and f.metadata["rerank_score"] >= MIN_RERANK_SCORE
        assert f.metadata["section_label_page_level"] is True and f.metadata["extractive"] is True


def quotes(result):
    return [f.evidence_quote for f in result.findings]


# ---------------------------------------------------------------------------
# Import safety, contract, registry, resources
# ---------------------------------------------------------------------------
def test_importing_teaching_loads_no_model_and_registers_nothing():
    code = ("import sys; import handbook_bot.agents.specialists.teaching as t; "
            "heavy=[m for m in ('sentence_transformers','torch','transformers','groq','huggingface_hub') if m in sys.modules]; "
            "print(heavy, hasattr(t, 'REGISTRY') or hasattr(t, '_REGISTRY'))")
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("[] False")


def test_specialist_implements_the_contract_and_registers(specialist):
    assert isinstance(specialist, Specialist) and specialist.specialist_id is SpecialistId.TEACHING
    registry = SpecialistRegistry()
    assert registry.register(specialist) is SpecialistId.TEACHING and registry.resolve("teaching") is specialist
    result = run(specialist, "What are my office hour obligations?")
    assert isinstance(result, SpecialistFindings) and result.llm_used is False and result.ms >= 0.0
    assert len({f.finding_id for f in result.findings}) == len(result.findings)
    json.dumps(result.to_dict())


def test_task_assigned_to_another_specialist_is_rejected(specialist):
    with pytest.raises(ValueError):
        specialist.run(task("What are my office hour obligations?", specialist_id="research"))
    assert specialist.run(task("What are my office hour obligations?", specialist_id=None)).status is FindingStatus.SUPPORTED
    with pytest.raises(TypeError):
        specialist.run("not a task")


def test_from_knowledge_base_wires_a_loaded_document(corpus):
    kb = SimpleNamespace(embedder=FakeEmbedder(), reranker=RawOverlapReranker(), index=corpus.index, chunks=corpus.chunks,
                         metadata=corpus.metadata, source=corpus.source, stats={"source": {"section_records": 206}})
    sp = TeachingLearningSpecialist.from_knowledge_base(kb)
    assert sp.scope.page_ranges and sp.resources.embed("office hours").shape == (1, FakeEmbedder.dim)
    assert run(sp, "What are my office hour obligations?").status is FindingStatus.SUPPORTED
    kb.stats = {"source": {"section_records": 0}}
    with pytest.raises(ValueError):
        TeachingLearningSpecialist.from_knowledge_base(kb)
    kb.source = unregistered_source("data/other.pdf")
    with pytest.raises(ValueError):
        TeachingLearningSpecialist.from_knowledge_base(kb)


# ---------------------------------------------------------------------------
# Scope from the real section map
# ---------------------------------------------------------------------------
def test_scope_is_source_plus_page_ranges(registry_and_map, specialist):
    _source, section_map = registry_and_map
    scope = specialist.scope
    assert scope.source_ids == frozenset({APPROVED_SOURCE_ID}) and scope.chapters is None and scope.section_numbers is None
    expected = set()
    for chapter, number, _ in OWNED_SECTIONS:
        records = [r for r in section_map.sections() if r.chapter == chapter and r.section_no == number]
        assert records, (chapter, number)
        expected |= {p for r in records for p in range(r.start_page, r.end_page + 1)}
    assert {p for a, b in scope.page_ranges for p in range(a, b + 1)} == expected


@pytest.mark.parametrize("page_no", [49, 51, 75, 79, 80, 83, 88, 93, 98, 112, 117, 119, 121, 192, 194, 220, 222, 224, 227, 228, 232, 268, 272])
def test_owned_pages_are_inside_the_scope(specialist, page_no):
    assert specialist.scope.allows({"source_id": APPROVED_SOURCE_ID, "page": page_no})


@pytest.mark.parametrize("page_no", [14, 54, 65, 81, 85, 107, 118, 122, 123, 161, 183, 203, 219, 233, 236, 267])
def test_excluded_pages_are_outside_the_scope(specialist, page_no):
    assert not specialist.scope.allows({"source_id": APPROVED_SOURCE_ID, "page": page_no})
    assert not specialist.scope.allows({"source_id": "unregistered_other", "page": 79})


def test_excluded_sections_are_really_excluded(registry_and_map):
    _source, section_map = registry_and_map
    owned = {(c, n) for c, n, _ in OWNED_SECTIONS}
    scope = compile_scope(section_map)
    for chapter, number, _ in EXCLUDED_SECTIONS:
        assert (chapter, number) not in owned
        for r in [r for r in section_map.sections() if r.chapter == chapter and r.section_no == number]:
            interior = [p for p in range(r.start_page + 1, r.end_page) if scope.allows({"source_id": APPROVED_SOURCE_ID, "page": p})]
            assert interior == [], (number, interior)                # only shared boundary pages may remain


def test_scope_compilation_fails_loudly():
    partial = SectionMap(APPROVED_SOURCE_ID, 272, [SectionRecord("chapter", 3, "3", "Responsibilities", 75, 105),
                                                   SectionRecord("section", 3, "3.4", "Responsibility to Office Hours", 79, 80)])
    with pytest.raises(ValueError, match="absent"):
        compile_scope(partial)
    assert compile_scope(partial, owned=[(3, "3.4", "Office hours")]).page_ranges == ((79, 80),)
    with pytest.raises(ValueError):
        compile_scope(SectionMap("other_source_x", 10, [SectionRecord("chapter", 1, "1", "A", 1, 10)]))


# ---------------------------------------------------------------------------
# F-1: section guard on shared pages
# ---------------------------------------------------------------------------
def test_guard_resolves_every_shared_page_of_the_fixture(specialist):
    guard = specialist.guard
    assert guard.unresolved_pages == []
    assert {49, 51, 80, 83, 192, 221, 222} <= set(guard.shared_pages)
    assert 79 not in guard.shared_pages and 96 not in guard.shared_pages          # owned sections only
    assert 232 in guard.shared_pages


def test_guard_attributes_rows_and_paragraphs_by_printed_heading(specialist, corpus):
    for i, m in enumerate(corpus.metadata):
        if m["page"] != 80:
            continue
        allowed = specialist.guard.owned_spans(i, corpus.chunks[i], m)
        text = corpus.chunks[i]
        if m["chunk_type"] == "row":
            assert bool(allowed) == ("Research" not in text and "research" not in text), text
        elif m["chunk_type"] == "paragraph":
            owned_text = " ".join(text[s:e] for s, e in allowed)
            assert "3.5 Responsibility to Research" not in owned_text
            assert "advance knowledge" not in owned_text and "Staying Current" not in owned_text
            if "office hour" in text:
                assert "office hour" in owned_text
    for i, m in enumerate(corpus.metadata):
        if m["page"] == 222 and m["chunk_type"] == "paragraph":
            owned_text = " ".join(corpus.chunks[i][s:e] for s, e in specialist.guard.owned_spans(i, corpus.chunks[i], m))
            assert "graduation requirements" not in owned_text and "residency requirement" not in owned_text
            if "uploaded on Blackboard" in corpus.chunks[i]:
                assert "uploaded on Blackboard" in owned_text


def test_guard_rejects_a_shared_page_whose_heading_is_not_printed(registry_and_map):
    source, section_map = registry_and_map
    pages = [page(80, "By upholding their responsibilities to office hours, faculty members foster a supportive academic environment.",
                  "Faculty members hold a fundamental responsibility to advance knowledge through research.")]      # no "3.5" heading line
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    sp = TeachingLearningSpecialist(TeachingResources(embed=lambda t: embedder.encode([t]), rerank=RawOverlapReranker().predict, index=index,
                                                      chunks=chunks, metadata=metadata, source=source, section_map=section_map))
    assert 80 in sp.guard.unresolved_pages
    assert sp.guard.owned_spans(0, chunks[0], metadata[0]) == []
    result = run(sp, "What are my office hour obligations?")
    assert result.status is FindingStatus.NOT_FOUND and result.metadata["boundary"]["unresolved_pages"] == [80]


@pytest.mark.parametrize("question,forbidden", [
    ("What are the teaching responsibilities of faculty regarding research?", ("advance knowledge", "Staying Current", "3.5 Responsibility")),
    ("What syllabus requirements apply to graduate students?", ("graduation requirements", "12.10")),
    ("Which courses must students complete for the internship policy?", ("internship", "12.8")),
    ("What community service is expected alongside teaching?", ("community service",)),
    ("What personal contact information may instructors share about students?", ("contact numbers",)),
])
def test_excluded_section_text_on_a_shared_page_is_never_evidence(corpus, question, forbidden):
    high = ScriptedReranker([(needle, 9.0) for needle in forbidden] + [("", 1.0)])     # the excluded text outranks everything
    result = run(make_specialist(corpus, high), question)
    for f in result.findings:
        assert not any(n in f.evidence_quote for n in forbidden), f.evidence_quote
        assert (f.metadata["chapter"], f.metadata["section_no"]) not in {(3, "3.5"), (12, "12.8"), (12, "12.10"), (12, "12.11")} or f.metadata["shared_page"]
    assert_traceable(result, corpus)


@pytest.mark.parametrize("question", [
    "What are the graduate completion requirements?",
    "What responsibilities do faculty have to advance knowledge in their discipline?",
    "What are the internship requirements?",
])
def test_questions_without_a_teaching_cue_are_not_searched(specialist, question):
    result = run(specialist, question)
    assert result.status is FindingStatus.OUT_OF_SCOPE and result.findings == [] and result.handoff_requests == []
    assert result.metadata["clauses"][0]["outcome"] == "unowned"


def test_internship_question_with_a_student_cue_finds_nothing_from_the_boundary(specialist):
    result = run(specialist, "What is the internship policy for students?")
    assert result.status is FindingStatus.NOT_FOUND and result.findings == []


# ---------------------------------------------------------------------------
# F-2 and F-6: concepts, relevance, morphology
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ("syllabus", "syllabi"), ("exam", "examinations"), ("examination", "exams"), ("grade", "grading"), ("grades", "graded"),
    ("class", "classes"), ("classroom", "classrooms"), ("policy", "policies"), ("responsibility", "responsibilities"),
    ("assessment", "assessments"), ("course", "courses"), ("teach", "teaching"), ("taught", "teaches"),
    ("advising", "advisor"), ("attendance", "absences"), ("upload", "uploaded"), ("submit", "submission"),
    ("office hours", "office hour"), ("credit hours", "credit hour"), ("Blackboard", "blackboard"), ("LMS", "learning management system"),
])
def test_concept_normalisation_pairs(a, b):
    ca, cb = concepts(a), concepts(b)
    if a in ("policy", "policies"):
        assert ca.vocab == cb.vocab == frozenset()                       # generic: no concept at all
    else:
        assert ca.vocab and ca.vocab == cb.vocab


def test_generic_terms_carry_no_concept():
    for term in sorted(GENERIC_TERMS):
        assert concepts(term).vocab == frozenset(), term
    assert ANCHOR_CONCEPTS <= VOCABULARY_CONCEPTS
    assert {"syllabus", "blackboard", "banner", "office_hours", "teaching_load", "attendance", "lms"} <= ANCHOR_CONCEPTS
    assert {"exam", "grade", "teach", "course", "class"} <= VOCABULARY_CONCEPTS - ANCHOR_CONCEPTS


@pytest.mark.parametrize("clause,text,expected", [
    ("What is the fee for a make-up exam?", "Requests for re-marking are subject to payment of a fixed fee.", False),
    ("What is the fee for a make-up exam?", "Coordinating the preparation of unified exams, particularly the midterm and final exams.", False),
    ("What is the penalty for a late syllabus upload?", "Late detection of cheating will still result in the suitable penalty.", False),
    ("What does the LMS policy require?", "Discretionary trips require the same approval process as cost-based conferences.", False),
    ("Give me the exact Blackboard clicks to create an assignment.", "Special Assignment 1-9", False),
    ("Who do I contact about a Blackboard problem?", "Personal information, including contact numbers, shall not be shared.", False),
    ("What are my office hour obligations?", "Faculty members must hold regular office hours outside their instructional time.", True),
    ("What is my teaching load?", "The baseline teaching load for this category is 12 credit hours per semester.", True),
    ("What are the main teaching responsibilities of faculty?", "Faculty members must approach their teaching assignments with dedication.", False),
    ("What are the main teaching responsibilities of faculty?", "The key teaching responsibilities of faculty members include the following items.", True),
    ("Where must I upload my syllabus?", "These syllabi shall be uploaded on Blackboard.", True),
    ("What grading scale does the handbook specify?", "Grades are converted according to the following grading system:", True),
])
def test_relevance_rule(clause, text, expected):
    assert is_relevant(concepts(clause), text) is expected


@pytest.mark.parametrize("word", ["fee", "policy", "process", "assignment", "contact", "require"])
def test_one_generic_word_never_makes_an_unrelated_chunk_supported(corpus, word):
    question = {"fee": "What is the fee for a make-up exam?", "policy": "What does the LMS policy require?",
                "process": "What is the process for peer observation of teaching?", "assignment": "Give me the exact Blackboard clicks to create an assignment.",
                "contact": "Who do I contact about a Blackboard problem?", "require": "What does the LMS policy require?"}[word]
    distractors = {"fee": "payment of a fixed fee", "policy": "10.5", "process": "approval process requires", "assignment": "Special Assignment",
                   "contact": "contact numbers", "require": "residency requirement"}
    sp = make_specialist(corpus, ScriptedReranker([(distractors[word], 9.0), ("", 0.5)]))
    result = run(sp, question)
    for f in result.findings:
        assert distractors[word] not in f.evidence_quote and f.metadata["relevance"]["anchors"] + f.metadata["relevance"]["vocabulary"] >= 1
        assert f.metadata["relevance"]["anchors"] >= 1 or f.metadata["relevance"]["vocabulary"] >= 2
    assert_traceable(result, corpus)


@pytest.mark.parametrize("question", ["What is the fee for a make-up exam?", "Which Blackboard menu records attendance?",
                                      "How do I set up my grading workflow in MyUOS?", "How do I open the fee payment screen in Banner?"])
def test_realistic_zero_evidence_questions_reach_not_found_or_partial(corpus, question):
    sp = make_specialist(corpus, ScriptedReranker([("payment of a fixed fee", 8.0), ("Late Detection of Cheating", 8.0), ("", 0.5)]))
    result = run(sp, question)
    assert result.status in (FindingStatus.NOT_FOUND, FindingStatus.PARTIAL)
    assert result.status is not FindingStatus.SUPPORTED
    for f in result.findings:
        assert "fixed fee" not in f.evidence_quote and "Cheating" not in f.evidence_quote


def test_high_scoring_wrong_candidate_loses_to_lower_ranked_relevant_evidence(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("12.18.1 Maximum and Minimum Class Size", 9.0), ("uploaded on Blackboard", 0.2), ("", -0.5)]))
    result = run(sp, "Where must I upload my syllabus?")
    assert result.status is FindingStatus.SUPPORTED
    assert result.findings[0].page in (222, 229) and "Blackboard" in result.findings[0].evidence_quote
    assert all("Class Size" not in q for q in quotes(result))


# ---------------------------------------------------------------------------
# F-3: quote quality, row/row_window handling, deduplication
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quote,kind,problem", [
    ("3.1 Teaching Responsibilities", "line", "heading"),
    ("3.1 Teaching Responsibilities", "prose", "heading"),
    ("12.12.1 Teaching and Evaluation", "line", "heading"),
    ("Chapter 3. Responsibilities of the Faculty", "prose", "heading"),
    ("4. Office hours", "line", "heading"),
    ("the following grading system:", "line", "lead-in"),
    ("Instruction for Course Syllabus:", "prose", "lead-in"),
    ("credit hours per semester.", "line", "dangling"),
    ("uploaded on Blackboard and kept in the course files within the college.", "line", "dangling"),
    ("Assignment by", "line", "line fragment"),
    ("Special Assignment 1-9", "line", "line fragment"),
    ("Faculty members are encouraged to communicate their office hours in course syllabi", "line", "line fragment"),
    ("Mon 25 Aug 02 Rabi I Classes begin", "line", None),
    ("Thu 28 Aug 05 Rabi I Last day for Add/Drop", "line", None),
    ("A 90-100 4.00", "line", None),
    ("The teaching load is 6 credit hours per semester.", "prose", None),
    ("1. Students are required to attend all lectures and examinations for the courses they are enrolled in.", "prose", None),
    ("All final course grades are evaluated numerically and converted to a point average according to the following grading system:", "prose", "lead-in"),
    ("The following guidelines outline the faculty's dedication to office hours:", "prose", "lead-in"),
    ("All final course grades are evaluated numerically according to the following grading system: A 90-100 4.00", "prose", None),
    ("Too short.", "prose", "too short"),
])
def test_quote_quality_rules(quote, kind, problem):
    assert quote_problem(quote, kind) == problem


def test_truncated_last_sentence_of_a_paragraph_is_rejected():
    assert quote_problem("Faculty teaching summer courses are not allowed to travel for conferences during the", "prose", last_unit=True) == "truncated"
    assert quote_problem("Faculty teaching summer courses are not allowed to travel for conferences during the", "prose", last_unit=False) is None


@pytest.mark.parametrize("line,expected", [
    ("Mon 25 Aug 02 Rabi I Classes begin", True), ("A 90-100 4.00", True), ("Grades Percentages Points", False),
    ("Fall Semester 2025/ 2026", False), ("Special Assignment 1-9", False), ("credit hours per semester.", False), ("F Below 60 0.00", True),
    ("Comprehensive Exam COM ALL 0 0 0 0.00 NA assigned to CRN in Banner University-Wide 1 Service to University", True),
])
def test_table_row_detection(line, expected):
    assert is_table_row(line) is expected


def test_heading_rows_are_never_findings_even_when_ranked_first(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("3.1 Teaching Responsibilities", 9.0), ("4. Office hours", 9.0), ("", 0.5)]))
    for question in ("What are the main teaching responsibilities of faculty?", "What are my office hour obligations?"):
        result = run(sp, question)
        assert result.status is FindingStatus.SUPPORTED
        assert all(q not in ("3.1 Teaching Responsibilities", "4. Office hours") for q in quotes(result))
        assert all(quote_problem(q, "prose") is None or quote_problem(q, "line") is None for q in quotes(result))


def test_duplicate_row_and_row_window_evidence_yields_one_finding(specialist, corpus):
    result = run(specialist, "When do Fall classes begin?", intent="date")
    assert result.status is FindingStatus.SUPPORTED
    assert quotes(result) == ["Mon 25 Aug 02 Rabi I Classes begin"]
    kinds = {corpus.metadata[i]["chunk_type"] for i, c in enumerate(corpus.chunks) if "Mon 25 Aug 02 Rabi I Classes begin" in c}
    assert {"row", "row_window"} <= kinds                                   # both chunk shapes existed; one finding


def test_bullet_items_are_quoted_whole(specialist):
    result = run(specialist, "What is the faculty teaching load?", clauses=["What is the teaching load for regular faculty?"])
    assert result.status is FindingStatus.SUPPORTED
    item = next(f.evidence_quote for f in result.findings if "12 credit hours" in f.evidence_quote)
    assert item.startswith('o "A" Regular Faculty') and item.endswith("12 credit hours per semester.")
    assert quote_problem(item, "prose") is None


def test_findings_cap_and_ordering(specialist):
    result = run(specialist, "What is the faculty teaching load?")
    assert 1 <= len(result.findings) <= teaching.MAX_FINDINGS_PER_CLAUSE
    scores = [f.metadata["rerank_score"] for f in result.findings]
    assert scores == sorted(scores, reverse=True)                       # no requested detail: reranker order


# ---------------------------------------------------------------------------
# Supported handbook facts (raw-overlap ranking, distractors present)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,page_no,must_contain,intent", [
    ("What are my office hour obligations?", 79, "office hours", "policy"),
    ("What are the main teaching responsibilities of faculty?", 75, "teaching responsibilities", "list"),
    ("What is the faculty teaching load?", 96, "teaching load", "policy"),
    ("When do classes begin?", 49, "Classes begin", "date"),
    ("Where must I upload my syllabus?", 222, "uploaded on Blackboard", "policy"),
    ("What is the student attendance rule?", 228, "attend", "policy"),
    ("What grading scale does the handbook specify?", 227, "grading system", "policy"),
    ("What does the LMS policy require?", 192, "Learning Management System", "policy"),
    ("How do grades entered in Blackboard reach Banner?", 224, "Banner", "policy"),
    ("Which Blackboard training modules must I complete?", 83, "Blackboard tools", "policy"),
    ("Who approves a course modification?", 221, "course modification", "policy"),
])
def test_supported_teaching_questions(specialist, corpus, question, page_no, must_contain, intent):
    result = run(specialist, question, intent=intent)
    assert result.status is FindingStatus.SUPPORTED, (result.summary, result.missing)
    assert result.missing == []
    assert any(f.page == page_no and must_contain.lower() in f.evidence_quote.lower() for f in result.findings), quotes(result)
    assert result.confidence == 1.0 and "not a probability" in result.metadata["confidence_semantics"]
    assert_traceable(result, corpus)


def test_shared_page_findings_carry_the_page_level_caveat(specialist):
    result = run(specialist, "Where must I upload my syllabus?")
    f = next(f for f in result.findings if f.page == 222)
    assert f.metadata["shared_page"] is True and f.metadata["section_label_page_level"] is True
    assert "12.12" in f.metadata["page_section_nos"] and f.metadata["boundary_guard"].startswith("attributed")
    plain = next(f for f in run(specialist, "What is the faculty teaching load?").findings if f.page in (89, 96))
    assert plain.metadata["shared_page"] is False and "3.11" in plain.metadata["page_section_nos"]
    assert plain.metadata["boundary_guard"] == "page carries owned sections only"


# ---------------------------------------------------------------------------
# F-5: calendar context
# ---------------------------------------------------------------------------
def test_calendar_context_from_line_and_header():
    header = ["1.14 Academic Calendar", "Fall Semester 2025/ 2026", "Day Description", "Mon 18 Aug 24 Safar Return of Academic Staff"]
    assert calendar_context(header, "Mon 25 Aug 02 Rabi I Classes begin") == {"term": "fall", "academic_year": "2025/2026", "from": "header"}
    assert calendar_context(header, "Mon 24 Aug 11 Rabi I Classes begin for Fall 2026-2027") == {"term": "fall", "academic_year": "2026/2027", "from": "line"}
    assert calendar_context([], "Thu 16 July 02 Safar Summer classes end") == {"term": "summer", "academic_year": None, "from": "line"}
    assert calendar_context(["Day Description"], "Mon 12 Jan 23 Rajab Classes begin") is None
    assert requested_term("When do Fall classes begin?") == ("fall", None)
    assert requested_term("When do classes begin in Fall 2026-2027?") == ("fall", "2026/2027")
    assert requested_term("When are final exams?") == (None, None)


@pytest.mark.parametrize("question,term,page_no,must_contain", [
    ("When do Fall classes begin?", "fall", 49, "Mon 25 Aug 02 Rabi I Classes begin"),
    ("When does the Spring semester begin?", "spring", 50, "Mon 12 Jan 23 Rajab Classes begin"),
    ("When do Summer classes begin?", "summer", 50, "Mon 01 June 15 Dhi Al-Hijja Classes begin"),
    ("When is the add/drop deadline?", None, 49, "Last day for Add/Drop"),
    ("When are final exams?", None, 49, "Final Exams"),
])
def test_calendar_questions_select_the_requested_term_and_current_year(specialist, question, term, page_no, must_contain):
    result = run(specialist, question, intent="date")
    assert result.status is FindingStatus.SUPPORTED, result.missing
    assert any(f.page == page_no and must_contain in f.evidence_quote for f in result.findings), quotes(result)
    for f in result.findings:
        ctx = f.metadata["calendar_context"]
        assert ctx["academic_year"] == "2025/2026"
        if term:
            assert ctx["term"] == term
        assert "2026-2027" not in f.evidence_quote


def test_wrong_year_calendar_row_never_beats_the_current_year(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("Classes begin for Fall 2026-2027", 9.0), ("Classes begin", 1.0), ("", -5.0)]))
    result = run(sp, "When do Fall classes begin?", intent="date")
    assert quotes(result) == ["Mon 25 Aug 02 Rabi I Classes begin"]
    explicit = run(sp, "When do classes begin in Fall 2026-2027?", intent="date")
    assert quotes(explicit) == ["Mon 24 Aug 11 Rabi I Classes begin for Fall 2026-2027"]
    assert explicit.findings[0].metadata["calendar_context"]["academic_year"] == "2026/2027"


def test_admission_row_is_not_evidence_for_semester_start(specialist):
    result = run(specialist, "When does the Spring semester begin?", intent="date")
    assert all("admission" not in q.lower() for q in quotes(result))


# ---------------------------------------------------------------------------
# F-4: ownership and handoff cues
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,expected", [
    ("What are my office hours by appointment rules?", (True, None)),
    ("My teaching appointment is at 9 AM, what are my office hour obligations?", (True, None)),
    ("Can students get a deadline extension on an assignment?", (True, None)),
    ("What is my contract teaching load?", (True, None)),
    ("What are the benefits of peer observation of teaching?", (True, None)),
    ("Is there funding for teaching innovation?", (True, None)),
    ("Who approves a course modification?", (True, None)),
    ("Who approves changes to a course?", (True, None)),
    ("Who approves teaching workload changes?", (True, None)),
    ("Who approves a new course syllabus?", (True, None)),
    ("How can AI be used in teaching?", (True, None)),
    ("What is the maximum number of students in a class?", (True, None)),
    ("Does my employment appointment affect probation?", (False, SpecialistId.FACULTY_SERVICES)),
    ("How many days of annual leave do I get?", (False, SpecialistId.FACULTY_SERVICES)),
    ("Who approves annual leave?", (False, SpecialistId.FACULTY_SERVICES)),
    ("What is the research ethics approval process?", (False, SpecialistId.RESEARCH)),
    ("How do I apply for a research grant?", (False, SpecialistId.RESEARCH)),
    ("What is the HR phone number?", (False, SpecialistId.INSTITUTIONAL)),
    ("Where is the IT help desk?", (False, SpecialistId.INSTITUTIONAL)),
    ("Who do I contact about Blackboard?", (True, SpecialistId.INSTITUTIONAL)),
    ("Can my teaching load be reduced if I have a funded research project?", (True, SpecialistId.RESEARCH)),
    ("Can I take annual leave during the teaching period?", (True, SpecialistId.FACULTY_SERVICES)),
    ("What are the internship requirements?", (False, None)),
    ("What is the parking fee?", (False, None)),
])
def test_clause_classification(clause, expected):
    assert classify_clause(clause) == expected


@pytest.mark.parametrize("question", [
    "What are my office hours by appointment rules?", "Can students get a deadline extension on an assignment?", "What is my contract teaching load?",
    "What are the benefits of peer observation of teaching?", "Is there funding for teaching innovation?", "Who approves a course modification?",
    "Who approves changes to a course?", "Who approves teaching workload changes?", "Who approves a new course syllabus?",
])
def test_ambiguous_words_create_no_handoff(specialist, question):
    result = run(specialist, question)
    assert result.handoff_requests == [] and result.status is not FindingStatus.OUT_OF_SCOPE


def test_approval_question_is_answered_from_teaching_evidence(specialist):
    result = run(specialist, "Who approves a course modification?")
    assert result.status is FindingStatus.SUPPORTED and result.handoff_requests == []
    assert any("department council" in q for q in quotes(result))


@pytest.mark.parametrize("question,target", [
    ("How many days of annual leave do I get?", SpecialistId.FACULTY_SERVICES),
    ("What is the research ethics approval process?", SpecialistId.RESEARCH),
    ("What is the HR phone number?", SpecialistId.INSTITUTIONAL),
    ("How do I renew my visa and passport?", SpecialistId.FACULTY_SERVICES),
    ("Where is the IT help desk?", SpecialistId.INSTITUTIONAL),
])
def test_misrouted_clauses_are_out_of_scope_with_a_handoff(specialist, question, target):
    result = run(specialist, question)
    assert result.status is FindingStatus.OUT_OF_SCOPE and result.findings == []
    assert result.requested_agents == [target]
    handoff = result.handoff_requests[0]
    assert handoff.from_specialist is SpecialistId.TEACHING and handoff.task.question == question
    assert handoff.task.context["focus_clauses"] == [question] and handoff.task.requested_by == "teaching"
    assert handoff.metadata["kind"] == "misrouted"


def test_shared_clause_is_answered_and_handed_off(specialist, corpus):
    result = run(specialist, "Can my teaching load be reduced if I have a funded research project?")
    assert result.status is FindingStatus.SUPPORTED and result.findings[0].page == 96
    assert result.requested_agents == [SpecialistId.RESEARCH] and result.handoff_requests[0].metadata["kind"] == "shared"
    assert_traceable(result, corpus)


def test_blackboard_contact_question_hands_off_without_using_privacy_text(specialist):
    result = run(specialist, "Who do I contact about a Blackboard problem?")
    assert result.requested_agents == [SpecialistId.INSTITUTIONAL]
    assert all("contact numbers" not in q for q in quotes(result))


def test_full_question_is_never_rescanned(specialist):
    question = "What is my teaching load, and what is the research ethics approval process?"
    result = run(specialist, question, clauses=["What is my teaching load"])
    assert result.status is FindingStatus.SUPPORTED and result.handoff_requests == []


def test_teaching_never_calls_the_coordinator(specialist, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("coordinator must not be called by the specialist")

    monkeypatch.setattr(coordinator, "analyze", boom)
    monkeypatch.setattr(coordinator, "coordinate", boom)
    assert run(specialist, "What is the faculty teaching load?").status is FindingStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Systems: policy versus procedure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,expected", [
    ("Which buttons do I click in Banner to submit grades?", (True, "banner")),
    ("How do grades reach Banner?", (False, "banner")),
    ("How do I create an assignment in Blackboard?", (True, "blackboard")),
    ("What does the handbook say about the LMS?", (False, "blackboard")),
    ("Where exactly in MyUOS do I click to see my teaching schedule?", (True, "myuos")),
    ("How do I create an assignment?", (False, None)),
    ("How do I prepare a syllabus?", (False, None)),
    ("What are my office hours obligations?", (False, None)),
])
def test_procedure_detector(clause, expected):
    assert is_procedure_request(clause, None) == expected


def test_procedure_detector_uses_the_task_system_hint():
    assert is_procedure_request("How do I create an assignment?", "blackboard") == (True, "blackboard")
    assert is_procedure_request("How do I create an assignment?", "unknown_system") == (False, None)


@pytest.mark.parametrize("question,system", [
    ("Give me the exact Blackboard clicks to create an assignment.", "blackboard"),
    ("Which buttons do I click to upload my syllabus in Blackboard?", "blackboard"),
    ("Show me every Banner screen I use to submit final grades.", "banner"),
    ("Where exactly in MyUOS do I click to see my teaching schedule?", "myuos"),
])
def test_system_procedures_are_never_invented(specialist, corpus, question, system):
    result = run(specialist, question)
    assert result.status in (FindingStatus.PARTIAL, FindingStatus.NOT_FOUND)
    assert any(system in m for m in result.missing) and any(system in lim for lim in result.limitations)
    for f in result.findings:
        assert not any(w in f.evidence_quote.lower() for w in ("click", "menu", "button", "screen"))
    assert_traceable(result, corpus)


def test_policy_and_procedure_forms_of_the_same_fact(specialist):
    policy = run(specialist, "How do grades entered in Blackboard reach Banner?", system="banner")
    procedure = run(specialist, "Which screens do I click in Blackboard and Banner to submit my grades?", system="banner")
    assert policy.status is FindingStatus.SUPPORTED and policy.missing == []
    assert procedure.status is FindingStatus.PARTIAL and any("banner" in m or "blackboard" in m for m in procedure.missing)
    assert "Banner" in policy.findings[0].evidence_quote


# ---------------------------------------------------------------------------
# Status semantics, clause handling, failures
# ---------------------------------------------------------------------------
def test_multi_part_task_reports_per_clause_outcomes(specialist, corpus):
    question = "What is my teaching load, when do classes begin, and how do I submit grades in Banner?"
    result = run(specialist, question, clauses=["What is my teaching load", "when do classes begin", "how do I submit grades in Banner"], system="banner")
    assert result.status is FindingStatus.PARTIAL
    assert [r["outcome"] for r in result.metadata["clauses"]] == ["supported", "supported", "partial"]
    assert {f.page for f in result.findings} >= {96, 49}
    assert len(result.missing) == 1 and "clause 2" in result.missing[0]
    assert_traceable(result, corpus)


def test_unowned_clause_makes_a_partial_result(specialist):
    result = run(specialist, "x", clauses=["What is my teaching load", "What are the internship requirements"])
    assert result.status is FindingStatus.PARTIAL and result.handoff_requests == []
    assert [r["outcome"] for r in result.metadata["clauses"]] == ["supported", "unowned"]
    assert any("no teaching cue" in m for m in result.missing)


def test_same_chunk_serving_two_clauses_gives_one_finding(specialist):
    result = run(specialist, "x", clauses=["What is the teaching load for lecturers?", "How many credit hours do lecturers teach?"])
    assert len(quotes(result)) == len(set(quotes(result)))
    shared = [f for f in result.findings if len(f.metadata["clauses"]) == 2]
    assert shared and shared[0].metadata["clauses"] == [0, 1]
    assert any("lecturers is 15 credit hours" in q for q in quotes(result))


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_clause_limit(specialist, count):
    clauses = ["What is my teaching load", "when do classes begin", "what are my office hour obligations",
               "what is the grading system", "what does the LMS policy cover"][:count]
    result = run(specialist, " and ".join(clauses), clauses=clauses)
    processed = min(count, teaching.MAX_CLAUSES)
    assert [r["index"] for r in result.metadata["clauses"]] == list(range(processed))
    assert result.metadata["unprocessed_clauses"] == clauses[teaching.MAX_CLAUSES:]
    assert sum("not processed" in m for m in result.missing) == max(0, count - teaching.MAX_CLAUSES)
    assert result.status is (FindingStatus.PARTIAL if count > teaching.MAX_CLAUSES else FindingStatus.SUPPORTED)


def test_missing_focus_clauses_fall_back_to_the_full_question(specialist):
    plain = SpecialistTask(task_id="task-9", question="What are my office hour obligations?", specialist_id="teaching")
    result = specialist.run(plain)
    assert result.status is FindingStatus.SUPPORTED and result.metadata["clauses"][0]["clause"] == plain.question


@pytest.mark.parametrize("scope", [["unregistered_other_guide"], ["uos_blackboard_guide_2025"], [APPROVED_SOURCE_ID, "unregistered_other_guide"]])
def test_unapproved_source_scope_is_refused_without_retrieval(specialist, scope):
    result = run(specialist, "What are my office hour obligations?", source_scope=scope)
    assert result.status is FindingStatus.ERROR and result.findings == [] and "not approved" in result.metadata["error"]
    assert run(specialist, "What are my office hour obligations?", source_scope=[APPROVED_SOURCE_ID]).status is FindingStatus.SUPPORTED


@pytest.mark.parametrize("stage", ["embed", "index", "rerank"])
def test_failures_become_error_results_with_redacted_messages(corpus, stage):
    def boom(*a, **k):
        raise RuntimeError("failure gsk_ABCDEFGHIJKLMNOP " + "x" * 3000)

    class Broken:
        ntotal = corpus.index.ntotal

        def search(self, *a, **k):
            raise RuntimeError("index failure gsk_ABCDEFGHIJKLMNOP")

    base = dict(embed=lambda t: corpus.embedder.encode([t]), rerank=RawOverlapReranker().predict, index=corpus.index,
                chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source, section_map=corpus.section_map)
    base[stage] = {"embed": boom, "index": Broken(), "rerank": boom}[stage]
    result = run(TeachingLearningSpecialist(TeachingResources(**base)), "What are my office hour obligations?")
    assert result.status is FindingStatus.ERROR and result.findings == [] and result.llm_used is False
    assert "RuntimeError" in result.metadata["error"] and "gsk_ABC" not in result.metadata["error"] and len(result.metadata["error"]) < 200


def test_specialist_refuses_an_unregistered_source(corpus):
    with pytest.raises(ValueError):
        TeachingLearningSpecialist(TeachingResources(embed=lambda t: corpus.embedder.encode([t]), rerank=RawOverlapReranker().predict, index=corpus.index,
                                                     chunks=corpus.chunks, metadata=corpus.metadata, source=unregistered_source("data/other.pdf"), section_map=corpus.section_map))


def test_runs_are_deterministic(specialist):
    for question in ("What is my teaching load?", "When do Fall classes begin?", "Give me the exact Blackboard clicks to create an assignment."):
        outs = []
        for _ in range(5):
            d = specialist.run(task(question)).to_dict()
            d.pop("ms", None)
            outs.append(d)
        assert all(o == outs[0] for o in outs)


# ---------------------------------------------------------------------------
# Coordinator compatibility (no orchestration) and architecture boundary
# ---------------------------------------------------------------------------
def test_coordinator_task_is_accepted_as_is(specialist, corpus):
    decision = coordinator.coordinate("Where must I upload my syllabus to Blackboard?")
    assert decision.selected_specialists == [SpecialistId.TEACHING]
    t = decision.subtasks[0]
    result = check_findings(specialist, t, specialist.run(t))
    assert result.task_id == t.task_id and t.system == "blackboard"
    assert result.status is FindingStatus.SUPPORTED and any(f.page in (222, 229) and "Blackboard" in f.evidence_quote for f in result.findings)
    assert_traceable(result, corpus)


def test_teaching_answers_only_its_own_focus_of_a_cross_domain_question(specialist):
    question = "Can I reduce my teaching load if I have a funded research project, and who approves it?"
    decision = coordinator.coordinate(question)
    mine = next(t for t in decision.subtasks if t.specialist_id is SpecialistId.TEACHING)
    assert mine.question == question and mine.context["focus_clauses"]
    result = check_findings(specialist, mine, specialist.run(mine))
    assert result.status is FindingStatus.SUPPORTED and result.findings[0].page == 96
    assert result.requested_agents == [SpecialistId.RESEARCH]


def test_production_pipeline_is_untouched():
    from handbook_bot import config

    assert config.PLAN_F_ENABLED is False and config.MAX_LLM_CALLS == 2 and config.CACHE_VERSION == "v12"
    for name in ("orchestrator.py", "qa.py", "retrieval.py"):
        text = (REPO / "handbook_bot" / name).read_text(encoding="utf-8")
        assert "specialists" not in text


# ---------------------------------------------------------------------------
# Quote selection unit tests
# ---------------------------------------------------------------------------
def test_quote_selection_is_an_exact_slice_and_respects_allowed_spans():
    text = ("First sentence about payroll. Faculty members must hold regular office hours for students. "
            "3.5 Responsibility to Research Faculty members hold office hours for research students too.")
    clause = concepts("What are my office hour obligations?")
    full = select_quote(text, "paragraph", clause)
    assert full and full[0].startswith("Faculty members must hold regular office hours for students.") and full[0] in text
    cut = text.index("3.5 Responsibility")
    restricted = select_quote(text, "paragraph", clause, allowed=[(0, cut)])
    assert restricted and restricted[0] == "Faculty members must hold regular office hours for students."
    assert select_quote(text, "paragraph", clause, allowed=[(cut, len(text))]) is None or "3.5" in select_quote(text, "paragraph", clause, allowed=[(cut, len(text))])[0]
    assert select_quote(text, "paragraph", concepts("parking fee")) is None


def test_quote_extends_to_a_neighbouring_relevant_unit_only():
    text = "Office hours must be posted for students. Office hours are held weekly. Grants are separate."
    quote = select_quote(text, "paragraph", concepts("office hours for students"))
    assert quote and quote[0] == "Office hours must be posted for students. Office hours are held weekly."
    text2 = "The student must repeat the same course. 12.17 Instructor Responsibilities: Instructors are responsible for developing detailed syllabi that cover the course objectives."
    quote2 = select_quote(text2, "paragraph", concepts("What is the penalty for a late syllabus upload?"))
    assert quote2 and quote2[0].startswith("12.17 Instructor Responsibilities")


def test_row_window_units_are_rows_and_prose_lines_are_not_evidence():
    window = "Mon 25 Aug Classes begin | Thu 28 Aug Last day for Add/Drop | Mon 13 Oct Midterm exams"
    got = select_quote(window, "row_window", concepts("when is the last day for add/drop"))
    assert got and got[0] == "Thu 28 Aug Last day for Add/Drop"
    prose = "faculty members are required to schedule a minimum of five office hours | per week, spread over a minimum of two days."
    assert select_quote(prose, "row_window", concepts("office hours per week")) is None
# ===========================================================================
# Step 3.9C corrections
# ===========================================================================
from handbook_bot.agents.specialists.teaching import CATEGORY_GROUPS, SectionGuard, detail_covered, requested_details  # noqa: E402

EXCL_SENT = "The administrative faculty member coordinates departmental academic advising committees and course teaching schedules for the department."
OWN_SENT = "Faculty members enjoy academic freedom in teaching their courses and in discussing course content with students."


def overlap_corpus(registry_and_map, words_before):
    """Page 88: excluded 3.9 prose, then the owned 3.10 heading, then owned prose."""
    source, section_map = registry_and_map
    lines = [EXCL_SENT] * (words_before // len(EXCL_SENT.split()) + 1) + ["3.10 Academic Freedom and Responsibility"] + [OWN_SENT] * 6
    pages = [page(88, *lines)]
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    sp = TeachingLearningSpecialist(TeachingResources(embed=lambda t: embedder.encode([t]), rerank=RawOverlapReranker().predict, index=index,
                                                      chunks=chunks, metadata=metadata, source=source, section_map=section_map))
    return sp, chunks, metadata, pages[0]["text"]


# ---- F-A: overlap attribution ------------------------------------------------
@pytest.mark.parametrize("words_before", [60, 120, 160, 200, 220, 260])
def test_excluded_text_before_a_heading_inside_the_overlap_never_leaks(registry_and_map, words_before):
    sp, chunks, metadata, _text = overlap_corpus(registry_and_map, words_before)
    assert sp.guard.unresolved_pages == [] and 88 in sp.guard.shared_pages
    paragraphs = [i for i, m in enumerate(metadata) if m["chunk_type"] == "paragraph"]
    assert len(paragraphs) >= 2                                              # the layout really overlaps
    for i in paragraphs:
        for s, e in sp.guard.owned_spans(i, chunks[i], metadata[i]) or []:
            assert "administrative faculty member" not in chunks[i][s:e]
    result = run(sp, "What are the teaching schedules and academic advising duties of faculty?")
    assert all("administrative faculty" not in q for q in quotes(result))
    owned = run(sp, "What academic freedom do faculty have in teaching their courses?")
    assert owned.status is FindingStatus.SUPPORTED and all("academic freedom" in q for q in quotes(owned))


@pytest.mark.parametrize("words_before", [120, 160, 220])
def test_overlapping_windows_attribute_the_same_characters_consistently(registry_and_map, words_before):
    sp, chunks, metadata, text = overlap_corpus(registry_and_map, words_before)
    ownership = {}                                                           # page-text offset -> owned?
    for i, m in enumerate(metadata):
        if m["chunk_type"] != "paragraph":
            continue
        offset = text.find(chunks[i])
        assert offset >= 0
        spans = sp.guard.shared_pages[88].para_spans[i]
        for s, e, owned in spans:
            for pos in range(offset + s, offset + e):
                assert ownership.setdefault(pos, owned) == owned, (i, pos)
    heading = text.index("3.10 Academic Freedom")
    assert not any(owned for pos, owned in ownership.items() if pos < heading)
    assert any(owned for pos, owned in ownership.items() if pos >= heading)


def test_guard_pre_cut_region_uses_the_preceding_section_not_window_state(registry_and_map):
    source, section_map = registry_and_map
    # page 222: 12.9 in progress (excluded), then 12.10 and 12.11 (excluded), then 12.12 (owned)
    filler = "The residency requirement is counted in regular semesters for every undergraduate student enrolled in the college."
    lines = ["semesters to receive a bachelor degree.", "12.10 Graduate Completion Requirements Policy:"] + [filler] * 12 + \
            ["12.11 Academic Progress Policy:"] + [filler] * 12 + ["12.12 Examinations Policy:", "12.12.1 Teaching and Evaluation",
             "Instructors are required to prepare detailed syllabi that outline the course objectives and upload them on Blackboard."] * 3
    chunks, metadata = build_chunks([page(222, *lines)])
    annotate_metadata(metadata, source, section_map)
    guard = SectionGuard(chunks, metadata, section_map, OWNED_SECTIONS, compile_scope(section_map))
    for i, m in enumerate(metadata):
        if m["chunk_type"] == "paragraph":
            owned_text = " ".join(chunks[i][s:e] for s, e in guard.owned_spans(i, chunks[i], m))
            assert "residency requirement" not in owned_text and "bachelor degree" not in owned_text
            if "upload them on Blackboard" in chunks[i]:
                assert "upload them on Blackboard" in owned_text


# ---- F-B: requested details ------------------------------------------------------
@pytest.mark.parametrize("clause,families", [
    ("What is the fee for a make-up exam?", {"fee"}),
    ("What is the penalty for a late syllabus upload?", {"penalty"}),
    ("When is the syllabus upload deadline?", {"deadline"}),
    ("How many office hours must part-time lecturers hold each week?", {"quantity", "part_time", "category"}),
    ("Who approves a change to my office hours?", {"authority"}),
    ("What is the teaching load for research-intensive faculty?", {"category"}),
    ("What is the maximum class size for laboratory sections?", {"maximum"}),
    ("What percentage corresponds to grade A in the grading system?", {"percentage"}),
    ("What are my office hour obligations?", set()),
    ("How do grades entered in Blackboard reach Banner?", set()),
    ("Where must I upload my syllabus?", {"location"}),
    ("What are the most important teaching responsibilities?", set()),
    ("What must the syllabus contain?", {"requirement"}),
    ("How often must I hold office hours?", {"frequency"}),
])
def test_requested_detail_families(clause, families):
    assert {d.family for d in requested_details(clause)} == families


def test_detail_coverage_is_local_to_the_subject():
    clause = concepts("How many office hours must part-time lecturers hold each week?")
    quantity = next(d for d in requested_details("How many office hours must part-time lecturers hold each week?") if d.family == "quantity")
    part_time = next(d for d in requested_details("How many office hours must part-time lecturers hold each week?") if d.family == "part_time")
    assert detail_covered(quantity, clause, "Full-time faculty members are required to schedule a minimum of five office hours per week.")
    assert not detail_covered(part_time, clause, "Full-time faculty members are required to schedule a minimum of five office hours per week.")
    detached = "Part-time staff are welcome. • The library opens for five hours on Fridays."
    assert not detail_covered(quantity, clause, detached)                     # the number is not about office hours
    prorated = "2. Part-Time Faculty: Part-time faculty members' office hours will be prorated based on their teaching assignments."
    assert not detail_covered(quantity, clause, prorated)                     # a list marker is not a quantity
    assert detail_covered(part_time, clause, prorated)
    assert {d.family for d in requested_details("What happens when a student's absences exceed the limit?")} >= {"penalty"}
    assert detail_covered(part_time, clause, "Part-time lecturers hold office hours by arrangement with the department.")


def corpus_from(registry_and_map, pages):
    """A corpus built from the given fixture pages only."""
    source, section_map = registry_and_map
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


def corpus_without(registry_and_map, needles):
    """The fixture corpus with every line containing one of ``needles`` removed:
    the right evidence is absent, not merely ranked low."""
    source, section_map = registry_and_map
    pages = [page(p["page"], *[l for l in p["rows"] if not any(n in l for n in needles)]) for p in PAGES]
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


# (label, question, scripted scores, expected status, missing label, text that must not appear, evidence lines removed)
ADVERSARIAL = [
    ("make-up exam fee", "What is the fee for a make-up exam?", [("payment of a fixed fee", 9.0)], FindingStatus.NOT_FOUND, None, ("fee",), ()),
    ("late syllabus penalty", "What is the penalty for a late syllabus upload?", [("uploaded on Blackboard", 9.0)], FindingStatus.PARTIAL, "penalty or consequence", ("penalt",), ()),
    ("syllabus deadline", "When is the syllabus upload deadline?", [("post it on Blackboard", 9.0)], FindingStatus.PARTIAL, "deadline or date", ("beginning of the semester",), ()),
    ("part-time office hours", "How many office hours must part-time lecturers hold each week?", [("minimum of five office hours", 9.0)], FindingStatus.PARTIAL, "part-time distinction", ("part-time",), ()),
    ("office-hours authority", "Who approves a change to my office hours?", [("hold regular office hours", 9.0), ("at least one office hour", 8.0)], FindingStatus.PARTIAL, "approving authority", ("approv",), ()),
    ("research-intensive load", "What is the teaching load for research-intensive faculty?", [('"A" Regular Faculty', 9.0), ('"E" Lecturer', 8.0)], FindingStatus.PARTIAL, "research intensive faculty", ("Research Intensive",), ('"C" Research Intensive',)),
    ("laboratory maximum", "What is the maximum class size for laboratory sections?", [("enrollment limits for each course", 9.0)], FindingStatus.PARTIAL, "maximum or limit", ("Laboratory 20",), ("Class Type Maximum Enrollment",)),
    ("Banner deadline", "When is the deadline to submit final grades to Banner?", [("forward it automatically to the registration system", 9.0)], FindingStatus.PARTIAL, "deadline or date", ("forty-eight",), ("forty-eight hours",)),
    ("A-grade percentage", "What percentage of grade points is an A?", [("Cumulative Grade Point Average", 9.0)], FindingStatus.PARTIAL, "percentage or range", ("90-100",), ()),
    ("teaching-track workload", "What is the workload for teaching track faculty?", [('"A" Regular Faculty', 9.0), ('"E" Lecturer', 8.0)], FindingStatus.PARTIAL, "teaching track faculty", ("Teaching Track",), ('"D" Teaching Track',)),
]


@pytest.mark.parametrize("label,question,rules,status,missing_label,fabricated,drop", ADVERSARIAL, ids=[a[0] for a in ADVERSARIAL])
def test_related_but_incomplete_evidence_never_gives_full_support(corpus, registry_and_map, label, question, rules, status, missing_label, fabricated, drop):
    corpus = corpus_without(registry_and_map, drop) if drop else corpus
    sp = make_specialist(corpus, ScriptedReranker(rules, default=-5.0))
    result = run(sp, question)
    assert result.status is status, (result.status, result.missing)
    assert result.status is not FindingStatus.SUPPORTED and result.confidence < 1.0
    if missing_label:
        assert any(missing_label in m for m in result.missing), result.missing
    for f in result.findings:
        assert not any(x.lower() in f.evidence_quote.lower() for x in fabricated), f.evidence_quote
    assert_traceable(result, corpus)


@pytest.mark.parametrize("question,must_contain", [
    ("How many office hours must full-time faculty hold?", "five office hours"),
    ("Who must approve a course modification?", "department council"),
    ("What percentage corresponds to grade A in the grading system?", "A 90-100"),
    ("What is the teaching load for Regular Faculty?", '"A" Regular Faculty'),
    ("What is the teaching load for research-intensive faculty?", '"C" Research Intensive Faculty'),
    ("What is the maximum class size for laboratory sections?", "Laboratory 20"),
    ("When must I upload my syllabus?", "beginning of the semester"),
    ("When is the deadline to submit final grades to Banner?", "forty-eight hours"),
    ("What is the minimum number of office hours per week?", "minimum of five"),
    ("What is the penalty when a student is caught cheating late?", "suitable penalty"),
])
def test_fully_answered_questions_stay_supported(specialist, corpus, question, must_contain):
    result = run(specialist, question)
    assert result.status is FindingStatus.SUPPORTED, (question, result.status, result.missing)
    assert any(must_contain in f.evidence_quote for f in result.findings), quotes(result)
    assert result.missing == [] and result.confidence == 1.0
    assert_traceable(result, corpus)


def test_detail_coverage_orders_findings_before_reranker_score(corpus):
    sp = make_specialist(corpus, ScriptedReranker([('"A" Regular Faculty', 9.0), ('"E" Lecturer', 8.0), ('"C" Research Intensive', 0.5)], default=-5.0))
    result = run(sp, "What is the teaching load for research-intensive faculty?")
    assert result.status is FindingStatus.SUPPORTED and '"C" Research Intensive' in result.findings[0].evidence_quote
    assert "research intensive faculty" in " ".join(result.findings[0].metadata["details_covered"])


def test_missing_details_are_reported_in_plain_words(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("minimum of five office hours", 9.0)], default=-5.0))
    result = run(sp, "How many office hours must part-time lecturers hold each week?")
    assert result.status is FindingStatus.PARTIAL
    assert any(m.startswith("part-time distinction not found in the evidence") for m in result.missing)
    assert any("faculty category (lecturers)" in m for m in result.missing)
    report = result.metadata["clauses"][0]
    assert "number or amount" in report["requested_details"] and "part-time distinction" in report["uncovered_details"]
    assert "number or amount" not in report["uncovered_details"]


# ---- F-C: ownership vocabulary -------------------------------------------------
@pytest.mark.parametrize("clause", [
    "Is there a lesson plan template?", "What training is offered to new faculty?", "Are there workshops on assessment design?",
    "How do I use the grade center?", "Can I use Collaborate for online sessions?", "What is my instructional load?",
    "Does the gradebook sync?", "Is SafeAssign enabled?", "Is Turnitin available?", "Who mentors new faculty?",
    "How is thesis supervision counted?", "Is there a teaching award?", "How often must I be available for student consultations?",
    "What are my consultation hours?", "How do I get a section cancelled?", "Can a class section be merged?",
    "What is expected in the first week of classes?", "How is my timetable decided?", "Where is the course timetable published?",
    "How many learners can be in a lab?",
])
def test_coordinator_teaching_phrasings_are_owned(clause):
    assert classify_clause(clause)[0] is True


@pytest.mark.parametrize("clause", [
    "What are the internship requirements?", "What is the research grant deadline?", "When is my salary paid?",
    "How do I renew my visa?", "Where is the IT help desk?", "How do I get a parking permit?",
])
def test_non_teaching_questions_stay_unowned(clause):
    teaching, _other = classify_clause(clause)
    assert teaching is False


# ---- F-D: grade-table rows ---------------------------------------------------------
@pytest.mark.parametrize("line,expected", [
    ("F Below 60 0.00", True), ("D+ 65-69 1.50", True), ("A 90-100 4.00", True), ("C Above 70 2.00", True),
    ("B. The policy applies to all faculty", False), ("A student must attend all lectures", False), ("F fails the course", False),
])
def test_grade_rows_with_textual_ranges(line, expected):
    assert is_table_row(line) is expected
    if expected:
        assert quote_problem(line, "line") is None


# ---- F-E: attend versus attendance ------------------------------------------------
def test_bare_attend_is_not_classroom_attendance():
    assert "attendance" not in concepts("Faculty may attend international conferences with financial support.").anchors
    assert "attendance" in concepts("Students are required to attend all theoretical lectures.").anchors
    assert "attendance" in concepts("A student whose absences exceed 10% may be barred.").anchors
    assert not is_relevant(concepts("What is the student attendance requirement?"), "Faculty may attend international conferences with full financial support.")


def test_conference_attendance_never_supports_an_attendance_question(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("may not attend conferences", 9.0), ("required to attend all theoretical lectures", 0.5)], default=-5.0))
    result = run(sp, "What is the student attendance requirement?")
    assert result.status is FindingStatus.SUPPORTED
    assert all("conferences" not in q for q in quotes(result)) and any("attend all theoretical lectures" in q for q in quotes(result))


# ---- F-F: antecedents ---------------------------------------------------------------
def test_pronoun_initial_sentence_is_quoted_with_its_antecedent(specialist):
    result = run(specialist, "What syllabus requirements apply to directed study courses?")
    assert result.status is FindingStatus.SUPPORTED
    quote = next(q for q in quotes(result) if "Such courses must have" in q)
    assert quote.startswith("12.18.5 Direct Study") and "Such courses must have an appropriate syllabus" in quote


def test_pronoun_initial_sentence_without_a_safe_antecedent_is_rejected(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("These activities include workshops", 9.0)], default=-5.0))
    result = run(sp, "What training on teaching strategies is expected?")
    assert all(not q.startswith("These activities") for q in quotes(result))
    lms = run(make_specialist(corpus), "What is the LMS course enrollment rule?")
    for q in quotes(lms):
        assert not q.startswith("They become")


def test_pronoun_extension_never_crosses_an_excluded_span():
    text = "Research projects need ethics approval. Such courses must have an appropriate syllabus and assessment tools."
    clause = concepts("What syllabus requirements apply to directed study courses?")
    cut = text.index("Such courses")
    assert select_quote(text, "paragraph", clause, allowed=[(cut, len(text))]) is None
    assert select_quote(text, "paragraph", clause) is not None                 # whole chunk allowed: antecedent usable


# ---- F-G: procedural requests ----------------------------------------------------------
def test_tangential_sentence_cannot_make_a_procedure_request_partial(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("teach evening courses should schedule", 9.0)], default=-5.0))
    result = run(sp, "Where exactly in MyUOS do I click to see my teaching schedule?")
    assert result.status is FindingStatus.NOT_FOUND and result.findings == []
    assert any("myuos" in m for m in result.missing)


@pytest.mark.parametrize("question,system,subject", [
    ("Which Blackboard menu records attendance?", "blackboard", "attend"),
    ("Show me every Banner screen I use to submit final grades.", "banner", "grade"),
    ("Which buttons do I click to upload my syllabus in Blackboard?", "blackboard", "syllab"),
])
def test_procedure_request_is_partial_only_with_evidence_about_the_system_or_subject(specialist, question, system, subject):
    result = run(specialist, question)
    assert result.status is FindingStatus.PARTIAL
    assert all(system in q.lower() or subject in q.lower() for q in quotes(result)), quotes(result)


# ---- F-H: invariant --------------------------------------------------------------------
def test_guard_refuses_metadata_whose_chunk_ids_do_not_match_list_order(corpus):
    metadata = [dict(m) for m in corpus.metadata]
    metadata[3]["chunk_id"] = 4
    with pytest.raises(ValueError, match="out of order"):
        SectionGuard(corpus.chunks, metadata, corpus.section_map, OWNED_SECTIONS, compile_scope(corpus.section_map))
    with pytest.raises(ValueError, match="differ in length"):
        SectionGuard(corpus.chunks[:-1], corpus.metadata, corpus.section_map, OWNED_SECTIONS, compile_scope(corpus.section_map))


def test_category_groups_are_documented_and_matched():
    labels = [label for label, _ in CATEGORY_GROUPS]
    assert {"regular faculty", "research intensive faculty", "lecturers", "instructors"} <= set(labels)
    details = requested_details("How many credit hours do lecturers teach?")
    assert [d.label for d in details if d.family == "category"] == ["the requested faculty category (lecturers)"]
# ===========================================================================
# Step 3.9E corrections
# ===========================================================================
from handbook_bot.agents.specialists.teaching import _safe_error, unit_coverage  # noqa: E402


def small_corpus(registry_and_map, pages):
    source, section_map = registry_and_map
    chunks, metadata = build_chunks(pages)
    annotate_metadata(metadata, source, section_map)
    embedder = FakeEmbedder()
    index = faiss.IndexFlatIP(FakeEmbedder.dim)
    index.add(embedder.encode(chunks))
    return SimpleNamespace(chunks=chunks, metadata=metadata, index=index, embedder=embedder, source=source, section_map=section_map)


# ---- F-1: completeness within one local unit ------------------------------------------
def test_page_79_layout_part_time_quantity_is_partial_not_supported(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("Part-Time Faculty: Part-time faculty", 9.0), ("minimum of five office hours", 8.0)], default=-5.0))
    result = run(sp, "How many office hours must part-time faculty hold?")
    assert result.status is FindingStatus.PARTIAL and result.confidence < 1.0
    assert {f.page for f in result.findings} == {79}
    assert any(m.startswith("number or amount") and "part-time" in m.lower() or "not stated together" in m for m in result.missing), result.missing
    report = result.metadata["clauses"][0]
    assert "number or amount" in report["uncovered_details"] or "part-time distinction" in report["uncovered_details"]


def test_same_quote_union_does_not_answer_a_part_time_quantity(registry_and_map):
    pages = [page(79, "3.4 Responsibility to Office Hours",
                  "Part-time faculty members office hours are prorated based on their teaching assignments. Full-time faculty members must hold five office hours per week.")]
    c = small_corpus(registry_and_map, pages)
    result = run(make_specialist(c), "How many office hours must part-time faculty hold?")
    assert result.status is not FindingStatus.SUPPORTED
    clause = concepts("How many office hours must part-time faculty hold?")
    quote = "Part-time faculty members office hours are prorated based on their teaching assignments. Full-time faculty members must hold five office hours per week."
    assert len(unit_coverage(requested_details("How many office hours must part-time faculty hold?"), clause, quote)) < 2
    linked = "Part-time faculty members must hold two office hours per week."
    assert len(unit_coverage(requested_details("How many office hours must part-time faculty hold?"), clause, linked)) == 2


CROSS_FINDING = [
    ("part-time + number", "How many office hours must part-time faculty hold?",
     ["3.4 Responsibility to Office Hours", "Part-time faculty members office hours are prorated based on their teaching assignments.", "• Full-time faculty members must hold five office hours per week."], 79),
    ("category + load", "What is the teaching load for research-intensive faculty?",
     ["3.11 Workload Allocation Model (WLAM)", 'o "C" Research Intensive Faculty has exceptionally high research productivity.', 'o "A" Regular Faculty teaching load is 12 credit hours per semester.'], 89),
    ("grade + percentage", "What percentage corresponds to grade A in the grading system?",
     ["12.14 Grading System:", "Grade A is the highest grade in the grading system.", "Scores of 90-100 percent are considered excellent by the committee."], 227),
    ("authority + approval", "Who approves a change to my office hours?",
     ["3.4 Responsibility to Office Hours", "Changes to office hours must be approved in advance.", "The department chair publishes the office hours timetable."], 79),
    ("minimum + number", "What is the minimum number of office hours per week?",
     ["3.4 Responsibility to Office Hours", "A minimum applies to office hours for all faculty.", "Faculty typically hold five office hours per week."], 79),
    ("maximum + number", "What is the maximum class size for a lecture?",
     ["12.18 Class Size Policy: Class sizes are capped for every course.", "A lecture normally has 70 students according to the schedule."], 229),
    ("deadline + date", "When is the deadline to submit final grades?",
     ["12.12 Examinations Policy:", "Final grades are due by the deadline set by the Department Chair.", "The final examination date is 6 December for all courses."], 224),
    ("fee + amount", "What is the fee for re-marking a final exam?",
     ["12.12 Examinations Policy:", "A fee applies to re-marking of the final examination.", "The amount payable for examination services is AED 100."], 224),
]


@pytest.mark.parametrize("label,question,lines,page_no", CROSS_FINDING, ids=[c[0] for c in CROSS_FINDING])
def test_cross_finding_union_never_gives_full_support(registry_and_map, label, question, lines, page_no):
    c = small_corpus(registry_and_map, [page(page_no, *lines)])
    high = ScriptedReranker([(lines[1][:30], 9.0), (lines[-1][:30], 8.0)], default=9.0)
    result = run(make_specialist(c, high), question)
    assert result.status is not FindingStatus.SUPPORTED, (label, result.status, quotes(result))
    assert result.confidence < 1.0
    if result.findings:
        assert result.missing, label


@pytest.mark.parametrize("question,must_contain", [
    ("What is the teaching load for Regular Faculty?", "12 credit hours"),
    ("What is the teaching load for research-intensive faculty?", "6 credit hours"),
    ("What percentage corresponds to grade A in the grading system?", "A 90-100"),
    ("How many office hours must full-time faculty hold?", "five office hours"),
    ("Who must approve a course modification?", "department council"),
    ("When must final grades be submitted to the Department Chair?", "forty-eight hours"),
    ("How many credit hours do lecturers teach?", "15 credit hours"),
    ("What is the minimum number of office hours per week?", "minimum of five office hours"),
])
def test_same_unit_evidence_stays_supported(specialist, corpus, question, must_contain):
    result = run(specialist, question)
    assert result.status is FindingStatus.SUPPORTED, (question, result.status, result.missing)
    assert any(must_contain in f.evidence_quote for f in result.findings), quotes(result)
    assert_traceable(result, corpus)


# ---- F-2: number semantics ----------------------------------------------------------------
@pytest.mark.parametrize("unit,expected", [
    ("Faculty hold five office hours per week.", True),
    ("Full-time faculty are required to schedule a minimum of five office hours per week.", True),
    ("Office hours: four (4) office hours per week on average.", True),
    ("Office hours are held in room 204.", False),
    ("Office hours are listed on page 79.", False),
    ("Office hours are described in section 3.4.", False),
    ("Office hours apply in the 2025/2026 academic year.", False),
    ("Office hours begin on August 25.", False),
    ("Office hours are item 3 of the list.", False),
    ("The course has 3 credit hours and office hours are posted.", False),
])
def test_quantity_needs_a_number_tied_to_the_counted_subject(unit, expected):
    clause = "How many office hours must faculty hold?"
    quantity = next(d for d in requested_details(clause) if d.family == "quantity")
    assert detail_covered(quantity, concepts(clause), unit) is expected, unit


def test_quantity_subject_can_be_a_generic_count_noun():
    clause = "How many students can be in a laboratory section?"
    quantity = next(d for d in requested_details(clause) if d.family == "quantity")
    assert detail_covered(quantity, concepts(clause), "Laboratory sections are limited to 20 students.")
    assert not detail_covered(quantity, concepts(clause), "Laboratory sections meet in room 204 of the science building.")


def test_frequency_is_a_separate_family():
    assert {d.family for d in requested_details("How often must I hold office hours?")} == {"frequency"}
    freq = requested_details("How often must I hold office hours?")[0]
    assert detail_covered(freq, concepts("How often must I hold office hours?"), "Office hours are held weekly.")
    assert not detail_covered(freq, concepts("How often must I hold office hours?"), "Office hours are held in room 204.")


# ---- F-3: professional development and training ------------------------------------------
def test_development_and_training_are_anchors():
    assert "faculty_development" in concepts("What professional development is expected of faculty?").anchors
    assert "faculty_development" in concepts("Complete training modules on teaching strategies").anchors
    assert concepts("What training must new faculty complete?").anchors & {"faculty_development", "new_faculty"}
    assert "new_faculty" in concepts("New faculty must attend the orientation programme.").anchors
    assert "faculty_development" not in concepts("All staff must complete annual security training.").anchors


@pytest.mark.parametrize("question", ["What professional development is expected of faculty?", "What training must new faculty complete?"])
def test_development_questions_are_answered_from_section_3_7(specialist, question):
    result = run(specialist, question)
    assert result.status is FindingStatus.SUPPORTED, (result.status, result.missing)
    assert any(f.page == 83 for f in result.findings)


@pytest.mark.parametrize("text", ["All staff must complete annual security training.", "Lab safety training is mandatory for laboratory users.",
                                  "Research compliance training is run by the research office.", "Faculty may attend conferences abroad."])
def test_unrelated_training_is_not_faculty_development_evidence(text):
    assert not is_relevant(concepts("What professional development is expected of faculty?"), text)
    assert not is_relevant(concepts("What training must new faculty complete?"), text)


# ---- F-4: numbered pronoun items -----------------------------------------------------------
def test_numbered_pronoun_item_needs_a_lead_in_antecedent():
    clause = concepts("Where must syllabi be uploaded?")
    sibling = "1. Office hours must be posted weekly. 2. These syllabi must be uploaded on Blackboard."
    assert select_quote(sibling, "paragraph", clause) is None
    lead_in = "Instructors must prepare course syllabi: 1. These syllabi must be uploaded on Blackboard."
    got = select_quote(lead_in, "paragraph", clause)
    assert got and got[0].startswith("Instructors must prepare course syllabi:")
    bullet = "Instructors must prepare course syllabi. • These syllabi must be uploaded on Blackboard."
    got = select_quote(bullet, "paragraph", clause)
    assert got and got[0].startswith("Instructors must prepare")
    heading = "12.12 Examinations Policy: 1. These syllabi must be uploaded on Blackboard."
    assert select_quote(heading, "paragraph", clause) is None                # a heading is not an antecedent


def test_numbered_pronoun_item_never_crosses_an_excluded_span():
    clause = concepts("Where must syllabi be uploaded?")
    text = "Instructors must prepare course syllabi: 1. These syllabi must be uploaded on Blackboard."
    cut = text.index("1. These")
    assert select_quote(text, "paragraph", clause, allowed=[(cut, len(text))]) is None


# ---- F-5: requirement family -----------------------------------------------------------------
@pytest.mark.parametrize("clause,expected", [
    ("What must the syllabus contain?", True), ("What are the requirements for office hours?", True),
    ("What must a faculty member provide at the start of a course?", True), ("What information must a course outline contain?", True),
    ("What is the teaching load?", False), ("What are my office hour obligations?", False), ("What does the LMS policy require?", False),
])
def test_requirement_family_detection(clause, expected):
    assert ("requirement" in {d.family for d in requested_details(clause)}) is expected


def test_topic_mention_without_an_obligation_is_not_full_support(corpus):
    sp = make_specialist(corpus, ScriptedReranker([("Syllabi for the current and previous course offerings", 9.0), ("appropriate syllabus, learning outcomes", 0.5)], default=-5.0))
    result = run(sp, "What must the syllabus contain?")
    assert result.status is not FindingStatus.SUPPORTED or all("must" in q or "required" in q or "responsible" in q for q in quotes(result))
    strict = make_specialist(corpus, ScriptedReranker([("Syllabi for the current and previous course offerings", 9.0)], default=-5.0))
    weak = run(strict, "What must the syllabus contain?")
    assert weak.status in (FindingStatus.PARTIAL, FindingStatus.NOT_FOUND)


def test_requirement_questions_with_real_obligations_stay_supported(specialist):
    result = run(specialist, "What must the syllabus contain?")
    assert result.status is FindingStatus.SUPPORTED, (result.status, result.missing)
    assert any(("required to prepare detailed syllabi" in q) or ("must have an appropriate syllabus" in q) for q in quotes(result)), quotes(result)
    hours = run(specialist, "What are the requirements for office hours?")
    assert hours.status is FindingStatus.SUPPORTED and any("required to schedule" in q or "embrace the responsibility" in q for q in quotes(hours))


# ---- F-6: error redaction ----------------------------------------------------------------------
@pytest.mark.parametrize("message,hidden", [
    (r"index failure at C:\Users\PC\secret\path", ("Users", "secret")),
    (r"cannot open C:\project\data\file.faiss", ("project", "file.faiss")),
    ("cannot open /home/user/secret/file", ("home", "secret")),
    ("cannot open /Users/name/private/file", ("private", "name")),
    ("failure gsk_ABCDEFGHIJKLMNOP while reading /var/lib/app/index.faiss", ("gsk_ABC", "/var", "index.faiss")),
])
def test_error_redaction_hides_paths_and_keys(message, hidden):
    text = _safe_error(RuntimeError(message))
    assert text.startswith("RuntimeError:") and "<path>" in text or "gsk_***" in text
    for h in hidden:
        assert h not in text, text
    assert "failure" in text or "cannot open" in text


def test_index_failure_result_hides_the_path(corpus):
    class Broken:
        ntotal = corpus.index.ntotal

        def search(self, *a, **k):
            raise RuntimeError(r"index failure at C:\Users\PC\secret\path gsk_ABCDEFGHIJKLMNOP")

    res = TeachingResources(embed=lambda t: corpus.embedder.encode([t]), rerank=RawOverlapReranker().predict, index=Broken(),
                            chunks=corpus.chunks, metadata=corpus.metadata, source=corpus.source, section_map=corpus.section_map)
    result = run(TeachingLearningSpecialist(res), "What are my office hour obligations?")
    assert result.status is FindingStatus.ERROR
    assert "Users" not in result.metadata["error"] and "gsk_ABC" not in result.metadata["error"] and "<path>" in result.metadata["error"]


# ---------------------------------------------------------------------------
# Step 3.12A (F-1): institutional phrasings that reach Teaching are handed off, never absorbed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,expected", [
    ("Who manages Blackboard support?", (True, SpecialistId.INSTITUTIONAL)),
    ("Who supports Blackboard?", (True, SpecialistId.INSTITUTIONAL)),
    ("Where is the Registrar?", (False, SpecialistId.INSTITUTIONAL)),
    ("Where is the Registrar office?", (False, SpecialistId.INSTITUTIONAL)),
    ("Where can I find the Registrar?", (False, SpecialistId.INSTITUTIONAL)),
    ("Who manages course grading?", (True, None)),
    ("Where are office hours held?", (True, None)),
    ("Who teaches the course?", (True, None)),
])
def test_institutional_navigation_phrasings(clause, expected):
    assert classify_clause(clause) == expected


def test_where_is_the_registrar_is_misrouted_to_institutional(specialist):
    result = run(specialist, "Where is the Registrar?")
    assert result.status is FindingStatus.OUT_OF_SCOPE and result.requested_agents == [SpecialistId.INSTITUTIONAL]


def test_who_manages_a_system_is_never_absorbed_without_a_named_party(registry_and_map):
    unnamed = [page(192, "10.6 Learning Management System (LMS) Policy", "Instructors are enrolled in their Blackboard courses automatically.")]
    sp = make_specialist(corpus_from(registry_and_map, unnamed), ScriptedReranker([], default=9.0))
    result = run(sp, "Who manages Blackboard support?")
    assert result.status is FindingStatus.PARTIAL and result.requested_agents == [SpecialistId.INSTITUTIONAL]
    assert any(m.startswith("responsible party or unit not found") for m in result.missing)
    named = [page(192, "10.6 Learning Management System (LMS) Policy", "The IT department administers the LMS and provides Blackboard support to faculty members.")]
    sp = make_specialist(corpus_from(registry_and_map, named), ScriptedReranker([], default=9.0))
    result = run(sp, "Who manages Blackboard support?")
    assert result.status is FindingStatus.SUPPORTED and result.requested_agents == [SpecialistId.INSTITUTIONAL]
    assert "responsible party or unit" in result.findings[0].metadata["details_covered"]


def test_management_system_is_not_a_responsibility_statement():
    detail = next(d for d in requested_details("Who manages the LMS?") if d.family == "responsibility")
    clause = concepts("Who manages the LMS?")
    assert not detail_covered(detail, clause, "10.6 Learning Management System (LMS) Policy The purpose of this Policy is to address the use of the LMS.")
    assert detail_covered(detail, clause, "The IT department administers the LMS to provide enhanced access to education.")
    assert detail_covered(detail, clause, "Instructors retain responsibility as the primary course instructor of their LMS course.")


def test_compound_clause_given_to_teaching_only_is_not_rescanned(specialist):
    question = "Who manages Blackboard support and what is my teaching load?"
    result = run(specialist, question, clauses=["what is my teaching load"])
    assert result.status is FindingStatus.SUPPORTED and result.handoff_requests == []


# ---------------------------------------------------------------------------
# Step 3.12A (F-2): grading and exam topic anchors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phrase", ["grading policy", "grading rules", "grading scheme", "grading system", "grading criteria"])
def test_grading_phrases_are_the_grading_anchor(phrase):
    c = concepts("What is the %s?" % phrase)
    assert "grading_system" in c.anchors and "grade" in c.vocab


@pytest.mark.parametrize("phrase", ["exam policy", "exam rules", "examination rules", "examination regulations", "exam procedures"])
def test_exam_phrases_are_the_exam_policy_anchor(phrase):
    c = concepts("What are the %s?" % phrase)
    assert "exam_policy" in c.anchors and "exam" in c.vocab


@pytest.mark.parametrize("clause,text", [
    ("What is the grading policy?", "All final course grades are evaluated numerically and converted to a point average."),
    ("What is the grading policy?", "Instructors may establish grading policies for assignments involving AI tools."),
    ("What are the exam rules?", "A. Breach of Exam Rules: If a student breaches exam rules, the instructor will collect the answer sheet."),
    ("What are the examination regulations?", "Students must settle all financial obligations before being allowed to sit for examination."),
])
def test_grading_and_exam_questions_accept_handbook_evidence(clause, text):
    assert is_relevant(concepts(clause), text)


@pytest.mark.parametrize("clause", ["What is the grading policy?", "What are the exam rules?"])
@pytest.mark.parametrize("text", [
    "The research policy requires ethics approval before data collection.",
    "The HR policy on travel rules applies to all staff.",
    "Parking rules are enforced by campus security.",
    "Travel rules require prior approval from the dean.",
    "Conference regulations require a copy of the flier with fees stated.",
])
def test_other_policies_and_rules_are_not_grading_or_exam_evidence(clause, text):
    assert not is_relevant(concepts(clause), text)


@pytest.mark.parametrize("text", ["What is the research policy?", "What are the parking rules?", "What are the travel rules?",
                                  "What are the conference regulations?", "policy rules regulations"])
def test_bare_policy_and_rules_are_not_anchors(text):
    assert concepts(text).anchors == frozenset()
    assert not ({"policy", "rules", "regulations"} & concepts(text).vocab)


GRADING_AND_EXAM_PAGES = [
    page(227, "12.14 Grading System:",
         "All final course grades are evaluated numerically and converted to a point average according to the following grading system:", "A 90-100 4.00"),
    page(228, "12.16 Penalties for Academic Misconduct During Exams:",
         "A. Breach of Exam Rules: If a student breaches exam rules, disregards instructor instructions, or disrupts the required silence during exams intentionally, the instructor will collect the student's answer sheet.",
         "B. Attempted Cheating in an Exam: If a student is found attempting to cheat during an exam, the following penalties will apply in combination:"),
    page(229, "Faculty must follow the travel rules and the HR policy of the University."),
    page(270, "Instructors may establish grading policies for assignments involving AI tools, provided that these policies are stated in the syllabus."),
]


def test_grading_policy_and_exam_rules_are_answered_from_owned_sections(registry_and_map):
    corpus = corpus_from(registry_and_map, GRADING_AND_EXAM_PAGES)
    sp = make_specialist(corpus, ScriptedReranker([], default=9.0))
    grading = run(sp, "What is the grading policy?")
    assert grading.status is FindingStatus.SUPPORTED, (grading.status, grading.missing)
    assert any("grading policies" in q or "final course grades" in q for q in quotes(grading))
    exams = run(sp, "What are the exam rules?")
    assert exams.status is FindingStatus.SUPPORTED, (exams.status, exams.missing)
    assert any(q.startswith("A. Breach of Exam Rules: If a student breaches exam rules") for q in quotes(exams))
    for result in (grading, exams):
        assert all("travel rules" not in q and "HR policy" not in q for q in quotes(result))
        assert_traceable(result, corpus)


# ---------------------------------------------------------------------------
# Step 3.12A (F-3): hyphen-like separators and the e-learning topic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hyphenated,spaced", [("office-hours", "office hours"), ("add/drop", "add drop"), ("add-drop", "add drop"),
                                               ("e-learning", "e learning"), ("part-time", "part time"), ("make-up exam", "make up exam"),
                                               ("re-marking", "re marking")])
def test_hyphen_like_separators_do_not_change_concepts_or_ownership(hyphenated, spaced):
    assert concepts(hyphenated) == concepts(spaced) and concepts(hyphenated).vocab
    assert classify_clause("What is the %s policy?" % hyphenated) == classify_clause("What is the %s policy?" % spaced)
    if hyphenated != "part-time":                                                   # part-time alone is not a teaching cue
        assert classify_clause("What is the %s policy?" % hyphenated)[0] is True


def test_e_learning_is_the_lms_topic():
    assert "lms" in concepts("What is the e-learning policy?").anchors
    assert is_relevant(concepts("What is the e-learning policy?"), "The purpose of this Policy is to address the use of the Learning Management System (LMS).")
    assert "lms" not in concepts("What is the machine learning policy?").anchors


# ---------------------------------------------------------------------------
# Step 3.12A (F-4): minimum, maximum and fee need value-shaped numbers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("unit,expected", [
    ("Full-time faculty must schedule a minimum of five office hours per week.", True),
    ("Laboratory sections are limited to a maximum of 20 students.", True),
    ("The class size must not exceed 40.", True),
    ("Absences must not exceed 10% of the total class hours.", True),
    ("A 90 percent maximum applies to coursework.", True),
    ("Class Type Maximum Enrollment Minimum Enrollment Lecture 70-120 30 Tutorial 30-40 15 Laboratory 20 10", True),
    ("The minimum is described in section 3.4 for office hours.", False),
    ("Class sizes reached a maximum in 2024 for every course.", False),
    ("The maximum is posted in room 204.", False),
    ("A minimum applies to course 101.", False),
    ("The maximum was set on page 12 of the handbook.", False),
])
def test_limit_values_are_tied_to_a_unit_or_a_limit_phrase(unit, expected):
    check = teaching._min_ok if "minimum" in unit.lower() else teaching._max_ok
    assert check(unit) is expected, unit


@pytest.mark.parametrize("unit,expected", [
    ("The re-marking fee is AED 100 per paper.", True),
    ("The cost is 100 AED for each request.", True),
    ("A fee of 75,000 Dirhams applies.", True),
    ("The service is provided free of charge.", True),
    ("Re-marking fees apply to examination 101 for every course.", False),
    ("Requests for re-marking are subject to payment of a fixed fee.", False),
    ("The fee schedule is in section 12.4 of the handbook.", False),
])
def test_fees_need_a_monetary_value(unit, expected):
    assert teaching._fee_ok(unit) is expected, unit


BARE_NUMBER_CASES = [
    ("minimum", "What is the minimum number of office hours per week?",
     page(79, "3.4 Responsibility to Office Hours", "The minimum is described in section 3.4 for office hours."), ("minimum", "number or amount")),
    ("maximum", "What is the maximum class size?", page(229, "12.18 Class Size Policy: Class sizes reached a maximum in 2024 for every course."), ("maximum or limit",)),
    ("fee", "What is the fee for re-marking an exam?", page(224, "12.12 Examinations Policy:", "A re-marking fee applies to examination 101 for every course."), ("fee or cost",)),
]


@pytest.mark.parametrize("label,question,fixture,labels", BARE_NUMBER_CASES, ids=[c[0] for c in BARE_NUMBER_CASES])
def test_bare_numbers_never_satisfy_limit_or_fee_questions(registry_and_map, label, question, fixture, labels):
    sp = make_specialist(corpus_from(registry_and_map, [fixture]), ScriptedReranker([], default=9.0))
    result = run(sp, question)
    assert result.status is FindingStatus.PARTIAL, (result.status, result.missing)
    for missing_label in labels:
        assert any(m.startswith(missing_label + " not found") for m in result.missing), (missing_label, result.missing)


# ---------------------------------------------------------------------------
# Step 3.12A (F-5): "should" satisfies a requirement question by design
# ---------------------------------------------------------------------------
def test_should_satisfies_a_requirement_question_by_design(registry_and_map):
    """The handbook states many obligations with "should"; the quote keeps the original wording."""
    pages = [page(229, "12.17 Instructor Responsibilities:", "The syllabus should include the course learning outcomes and the assessment scheme.")]
    sp = make_specialist(corpus_from(registry_and_map, pages), ScriptedReranker([], default=9.0))
    result = run(sp, "What must the syllabus include?")
    assert result.status is FindingStatus.SUPPORTED
    assert any("should include" in q and "must include" not in q for q in quotes(result))
    assert teaching._OBLIGATION.search("should") and teaching._OBLIGATION.search("must")


# ---------------------------------------------------------------------------
# Step 3.12A (F-6): "What are the <subject> requirements?"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clause,expected", [
    ("What are the office hours requirements?", True), ("What are the office-hours requirements?", True), ("What are the syllabus requirements?", True),
    ("What are the attendance requirements?", True), ("What are the research requirements?", True), ("What are my course file requirements?", True),
    ("What are my office hour obligations?", False), ("What are the most important teaching responsibilities?", False),
])
def test_what_are_the_subject_requirements_is_a_requirement_question(clause, expected):
    assert ("requirement" in {d.family for d in requested_details(clause)}) is expected


def test_requirement_detection_never_changes_ownership():
    assert classify_clause("What are the research requirements?") == (False, SpecialistId.RESEARCH)
    assert classify_clause("What are the internship requirements?") == (False, None)
    assert classify_clause("What are the office-hours requirements?") == (True, None)


def test_subject_requirement_question_needs_an_obligation_statement(registry_and_map):
    weak = make_specialist(corpus_from(registry_and_map, [page(79, "Office hours are listed on the department noticeboard each semester.")]), ScriptedReranker([], default=9.0))
    result = run(weak, "What are the office hours requirements?")
    assert result.status is FindingStatus.PARTIAL and any("requirement statement" in m for m in result.missing)
    strong = make_specialist(corpus_from(registry_and_map, [page(79, "Full-time faculty members are required to schedule a minimum of five office hours per week.")]), ScriptedReranker([], default=9.0))
    assert run(strong, "What are the office-hours requirements?").status is FindingStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Step 3.12A (F-7): lettered list items
# ---------------------------------------------------------------------------
def test_lettered_items_are_list_items():
    clause = concepts("Where must syllabi be uploaded?")
    sibling = "a. Office hours must be posted weekly. b. These syllabi must be uploaded on Blackboard."
    assert select_quote(sibling, "paragraph", clause) is None                      # "a." is a sibling, never an antecedent
    upper = "A. Office hours must be posted weekly. B. These syllabi must be uploaded on Blackboard."
    assert select_quote(upper, "paragraph", clause) is None
    lead_in = "Instructors must prepare course syllabi: a. These syllabi must be uploaded on Blackboard."
    got = select_quote(lead_in, "paragraph", clause)
    assert got and got[0] == lead_in
    heading = "12.17 Instructor Responsibilities: a. These syllabi must be uploaded on Blackboard."
    assert select_quote(heading, "paragraph", clause) is None
    plain = "b. Syllabi must be uploaded on Blackboard before classes begin."
    got = select_quote(plain, "paragraph", clause)
    assert got and got[0] == plain                                                  # a lettered item is not a dangling fragment
    assert quote_problem("Students receive a grade of A. The syllabi must be uploaded on Blackboard.", "prose") is None
    units = teaching._local_units("Rules: A. Breach of Exam Rules: If a student cheats, penalties apply. B. Attempted Cheating: The exam is void.")
    assert [u[:12] for u in units] == ["Rules:", "A. Breach of", "B. Attempted"]


def test_uppercase_lettered_item_is_quoted_whole_with_its_marker(registry_and_map):
    corpus = corpus_from(registry_and_map, GRADING_AND_EXAM_PAGES[1:2])
    result = run(make_specialist(corpus, ScriptedReranker([], default=9.0)), "What are the exam rules?")
    assert result.status is FindingStatus.SUPPORTED
    assert any(q.startswith("A. Breach of Exam Rules: If a student breaches exam rules") and q.endswith("answer sheet.") for q in quotes(result))
    assert all("B. Attempted" not in q for q in quotes(result))                     # the sibling item is not joined
    assert_traceable(result, corpus)


# ---------------------------------------------------------------------------
# Step 3.12A (F-8): "N hours of <subject>"
# ---------------------------------------------------------------------------
def test_n_hours_of_subject_counts_only_for_that_subject(registry_and_map):
    clause = "How many office hours must faculty hold each semester?"
    quantity = next(d for d in requested_details(clause) if d.family == "quantity")
    assert detail_covered(quantity, concepts(clause), "Faculty must hold 48 hours of office hours each semester.")
    assert not detail_covered(quantity, concepts(clause), "Faculty must complete 48 hours of training each semester.")
    assert not detail_covered(quantity, concepts(clause), "Faculty must complete 48 hours of professional development each semester.")
    sp = make_specialist(corpus_from(registry_and_map, [page(79, "3.4 Responsibility to Office Hours", "Faculty must hold 48 hours of office hours each semester.")]), ScriptedReranker([], default=9.0))
    assert run(sp, clause).status is FindingStatus.SUPPORTED
    other = make_specialist(corpus_from(registry_and_map, [page(79, "3.4 Responsibility to Office Hours", "Faculty must complete 48 hours of training each semester.")]), ScriptedReranker([], default=9.0))
    assert run(other, clause).status is not FindingStatus.SUPPORTED
