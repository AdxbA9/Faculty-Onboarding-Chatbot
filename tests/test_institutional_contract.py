"""
Institutional Navigation contract, fail-closed behaviour, determinism and dispatch (Milestone 4, Step 4.16-B1, D-104;
updated in Steps 4.16-B2, 4.16-B3 and 4.16-B4).

The specialist reads every focus clause. Phase B2 answers telephone and fax asks from the contact directory (tested in
``test_institutional_directory.py``); phase B3 answers office responsibilities, structure and documented locations from
statements that bind them (tested in ``test_institutional_offices.py``). Every other ask is still answered with nothing:
a valid ``not_found`` with no finding, one ``missing`` entry per clause and the phase limitation, so no e-mail, URL,
procedure, unbound responsibility or undocumented location is ever output; since phase B4 an HR or Research subject is
``out_of_scope`` with a handoff to its owner (tested in ``test_institutional_integration.py``). Since Step 4.16-B6
(D-105) the default dispatch executes it when it is registered (``EXECUTABLE_SPECIALISTS`` holds all four specialists);
an unregistered Institutional Navigation stays pending.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from handbook_bot.agents.coordinator import coordinate
from handbook_bot.agents.specialists import dispatch as D
from handbook_bot.agents.specialists import institutional as I
from handbook_bot.agents.specialists.base import check_findings
from handbook_bot.agents.specialists.contracts import FindingStatus, SpecialistFindings, SpecialistId
from handbook_bot.agents.specialists.registry import SpecialistRegistry
from handbook_bot.text_utils import EMAIL_PATTERN, PHONE_PATTERN

from institutional_support import RawOverlapReranker, ScriptedReranker, institutional_corpus, make_institutional, run, task

NOT_FOUND = FindingStatus.NOT_FOUND
OVERRIDE = D.EXECUTABLE_SPECIALISTS + (SpecialistId.INSTITUTIONAL,)
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.I)
QUESTIONS = [
    "What is the phone number of Human Resources?",            # channel: answered from the directory since phase B2
    "What is the fax number of the Finance Department?",
    "What is the email of the IT service desk?",               # channel, but no lexicon office and no e-mail: not answered
    "Who is responsible for health and safety on campus?",     # responsibility (phase B3)
    "Which office handles purchasing requests?",
    "Which office handles faculty housing?",
    "Who should I contact about Blackboard?",
    "Who approves a faculty research request?",
    "Where is the Main Library located?",
    "How do I log in to MyUOS?",                               # procedure: no E3 evidence exists
    "How do I submit an IT service request?",
    "What is the organizational structure of the university?",
    "How many colleges does the university have?",
]
#: The two directory questions phase B2 answers.
DIRECTORY_QUESTIONS = QUESTIONS[:2]
#: The questions phase B3 answers from bound statements (Step 4.16-B3), with the status each gets: the Main Library has
#: two printed entries (W16, M16) and the question does not pick one, so it stays partial; no statement names "health
#: and safety" itself as a duty, so the Health and Safety Office's narrower duties are given as candidates (partial).
B3_ANSWERED = {
    "Who is responsible for health and safety on campus?": FindingStatus.PARTIAL,
    "Who should I contact about Blackboard?": FindingStatus.SUPPORTED,
    "Where is the Main Library located?": FindingStatus.PARTIAL,
    "What is the organizational structure of the university?": FindingStatus.SUPPORTED,
    "How many colleges does the university have?": FindingStatus.SUPPORTED,
}
#: The questions whose subject the shared cues give to another specialist (Step 4.16-B4): out_of_scope, handed off.
B4_HANDED_OFF = {
    "Which office handles faculty housing?": SpecialistId.FACULTY_SERVICES,
    "Who approves a faculty research request?": SpecialistId.RESEARCH,
}
#: Every other question is still not answered: no e-mail exists, no statement binds the duty, a procedure.
NOT_ANSWERED = [q for q in QUESTIONS[2:] if q not in B3_ANSWERED and q not in B4_HANDED_OFF]


@pytest.fixture(scope="module")
def sp():
    return make_institutional()


def _comparable(result: SpecialistFindings) -> dict:
    out = result.to_dict()
    out.pop("ms", None)
    return out


# ---------------------------------------------------------------------------
# Identity and resources
# ---------------------------------------------------------------------------
def test_identity(sp):
    assert I.InstitutionalNavigationSpecialist.specialist_id is SpecialistId.INSTITUTIONAL
    assert sp.specialist_id.value == "institutional" and sp.display_name == "Institutional Navigation"


@pytest.mark.parametrize("source", [SimpleNamespace(source_id="another_source", registered=True),
                                    SimpleNamespace(source_id=I.APPROVED_SOURCE_ID, registered=False)])
def test_only_the_registered_handbook_is_accepted(source):
    corpus = institutional_corpus()
    res = I.InstitutionalResources(embed=lambda t: corpus.embedder.encode([t]), rerank=RawOverlapReranker().predict, index=corpus.index,
                                   chunks=corpus.chunks, metadata=corpus.metadata, source=source, section_map=corpus.section_map)
    with pytest.raises(ValueError):
        I.InstitutionalNavigationSpecialist(res)


@pytest.mark.parametrize("kb", [SimpleNamespace(source=None),
                                SimpleNamespace(source=SimpleNamespace(source_id=I.APPROVED_SOURCE_ID, registered=True), stats={"source": {}})])
def test_resources_from_a_knowledge_base_need_the_handbook_and_its_section_map(kb):
    with pytest.raises(ValueError):
        I.InstitutionalResources.from_knowledge_base(kb)


# ---------------------------------------------------------------------------
# Asks the current phase does not answer: valid, fail-closed results
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", NOT_ANSWERED)
def test_every_result_is_a_valid_not_found_with_no_finding(sp, question):
    r = run(sp, question)
    assert isinstance(r, SpecialistFindings) and r.specialist_id is SpecialistId.INSTITUTIONAL and r.task_id == "task-1"
    assert r.status is NOT_FOUND and r.findings == [] and r.handoff_requests == []
    assert r.llm_used is False and r.confidence == 0.0 and r.ms >= 0.0
    assert len(r.missing) == 1 and question in r.missing[0]
    assert any("phase %s" % I.PHASE in limitation for limitation in r.limitations)
    report = r.metadata["clauses"][0]
    assert r.metadata["phase"] == I.PHASE == "C1" and r.metadata["source_id"] == I.APPROVED_SOURCE_ID
    assert report["answered"] is False and report["evidence_tier_reached"] is None and report["clause"] == question


@pytest.mark.parametrize("question", QUESTIONS[2:])
def test_no_contact_detail_or_url_is_ever_output(sp, question):
    text = json.dumps(_comparable(run(sp, question)), ensure_ascii=False)
    assert not PHONE_PATTERN.search(text) and not EMAIL_PATTERN.search(text) and not URL_PATTERN.search(text), text


def test_no_answer_status_is_reachable_for_the_asks_not_answered(sp):
    statuses = {run(sp, q).status for q in NOT_ANSWERED}
    assert statuses == {NOT_FOUND}


@pytest.mark.parametrize("question", sorted(B4_HANDED_OFF))
def test_an_hr_or_research_subject_is_out_of_scope_with_a_handoff_since_phase_b4(sp, question):
    r = run(sp, question)
    assert r.status is FindingStatus.OUT_OF_SCOPE and r.findings == [] and r.missing == [] and r.confidence == 0.0
    assert [h.requested_specialist for h in r.handoff_requests] == [B4_HANDED_OFF[question]]
    h = r.handoff_requests[0]
    assert h.from_specialist is SpecialistId.INSTITUTIONAL and h.task.requested_by == "institutional"
    assert h.task.context == {"full_question": question, "focus_clauses": [question]}
    assert r.metadata["clauses"][0]["outcome"] == "handed_off" and r.metadata["phase"] == "C1"


@pytest.mark.parametrize("question", sorted(B3_ANSWERED))
def test_the_asks_phase_b3_answers_come_from_bound_statements_never_the_directory(sp, question):
    r = run(sp, question)
    assert r.status is B3_ANSWERED[question] and r.findings and r.handoff_requests == [] and r.llm_used is False
    for f in r.findings:
        assert "directory_entry" not in f.metadata and f.page not in (52, 53, 54, 55)
        assert f.claim == f.evidence_quote and f.metadata["evidence_tier"] in ("E2", "stated_fact")
        assert f.metadata["office_id"] is not None and f.metadata["extractive"] is True
    assert r.metadata["phase"] == I.PHASE and r.metadata["clauses"][0]["evidence_tier_reached"] in ("E2", "stated_fact")


def test_the_two_directory_questions_are_answered_from_the_directory_since_phase_b2(sp):
    phone, fax = (run(sp, q) for q in DIRECTORY_QUESTIONS)
    assert phone.status is FindingStatus.SUPPORTED and phone.findings[0].metadata["contact_values"] == {"telephone": "+971 6 5050023"}
    assert fax.status is FindingStatus.SUPPORTED and fax.findings[0].metadata["contact_values"] == {"fax": "+971 6 5585183"}
    for r in (phone, fax):
        assert r.findings[0].page == 54 and not EMAIL_PATTERN.search(json.dumps(_comparable(r))) and not URL_PATTERN.search(json.dumps(_comparable(r)))


def test_the_clause_report_names_only_lexicon_offices(sp):
    r = run(sp, "What is the phone number of Human Resources?", intent="contact")
    assert r.metadata["clauses"][0]["offices"] == ["human_resources"] and r.metadata["clauses"][0]["ask_kind"] == "channel"
    r = run(sp, "Which office handles faculty housing?")
    assert r.metadata["clauses"][0]["offices"] == []


def test_each_focus_clause_is_reported_in_order(sp):
    clauses = ["What is the phone number of Human Resources", "where is the Main Library located"]
    r = run(sp, "What is the phone number of Human Resources and where is the Main Library located?", clauses=clauses)
    assert [c["clause"] for c in r.metadata["clauses"]] == clauses
    assert len(r.missing) == 1 and r.missing[0].startswith("clause 1 (location ask)")       # clause 0 is answered since phase B2


def test_a_task_without_focus_clauses_uses_the_full_question(sp):
    t = task("Where is the Main Library located?")
    t.context = {"full_question": t.question}
    r = check_findings(sp, t, sp.run(t))
    assert [c["clause"] for c in r.metadata["clauses"]] == ["Where is the Main Library located?"]


def test_an_unapproved_source_scope_fails_closed(sp):
    r = run(sp, "What is the phone number of Human Resources?", source_scope=["blackboard_guide_2022"])
    assert r.status is NOT_FOUND and "not approved" in r.missing[0] and "blackboard_guide_2022" in r.missing[0]


def test_task_validation(sp):
    with pytest.raises(TypeError):
        sp.run({"question": "Where is the Main Library located?"})
    with pytest.raises(ValueError):
        sp.run(task("Where is the Main Library located?", specialist_id="teaching"))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", QUESTIONS)
def test_deterministic_across_runs_instances_and_rankings(sp, question):
    first = _comparable(run(sp, question))
    assert _comparable(run(sp, question)) == first
    assert _comparable(run(make_institutional(), question)) == first
    assert _comparable(run(make_institutional(reranker=ScriptedReranker([], default=9.0)), question)) == first


# ---------------------------------------------------------------------------
# Dispatch: the default dispatch runs it since Step 4.16-B6 (D-105); the executable= override still works
# ---------------------------------------------------------------------------
def test_the_default_executable_set_holds_all_four_specialists_since_b6():
    assert D.EXECUTABLE_SPECIALISTS == (SpecialistId.TEACHING, SpecialistId.RESEARCH, SpecialistId.FACULTY_SERVICES,
                                        SpecialistId.INSTITUTIONAL)


def _registry(sp):
    reg = SpecialistRegistry()
    reg.register(sp)
    return reg


def test_the_override_runs_institutional_navigation_through_the_dispatcher(sp):
    out = D.run_question("Where is the Main Library located?", _registry(sp), executable=OVERRIDE)
    assert out.executed_specialists == [SpecialistId.INSTITUTIONAL] and out.llm_calls == 0
    result = out.results[0]
    # since phase B3 the two printed Main Library entries are given; the question picks neither, so the result is partial
    assert result.status is FindingStatus.PARTIAL and [f.metadata["building_code"] for f in result.findings] == ["W16", "M16"]
    assert all(f.metadata["relation"] == "location" and f.page == 199 for f in result.findings)


def test_the_default_dispatch_runs_it_when_registered_and_keeps_it_pending_otherwise(sp):
    out = D.run_question("Where is the Main Library located?", _registry(sp))
    assert out.executed_specialists == [SpecialistId.INSTITUTIONAL] and out.pending_specialists == [] and out.llm_calls == 0
    assert _comparable(out.results[0]) == _comparable(D.run_question("Where is the Main Library located?", _registry(sp),
                                                                     executable=OVERRIDE).results[0])
    unregistered = D.run_question("Where is the Main Library located?", SpecialistRegistry())
    assert unregistered.pending_specialists == [SpecialistId.INSTITUTIONAL] and unregistered.executed_specialists == []
    assert "executable but not registered" in unregistered.runs[0].reason


@pytest.mark.parametrize("question", [
    "What is the phone number of HR and how much annual leave do I get?",
    "Who should I contact about Blackboard?",
    "What is the grading policy and who approves grade changes?",
])
def test_institutional_works_only_on_its_own_coordinator_task(sp, question):
    decision = coordinate(question)
    out = D.dispatch(decision, _registry(sp), executable=OVERRIDE, question=question)
    for run_ in out.runs:
        if run_.specialist_id is SpecialistId.INSTITUTIONAL:
            assert run_.state == D.STATE_EXECUTED
            assert [c["clause"] for c in run_.findings.metadata["clauses"]] == run_.task.context["focus_clauses"]
            assert run_.findings.handoff_requests == []
        else:
            assert run_.state == D.STATE_PENDING                         # the other specialists are not registered here
    assert SpecialistId.INSTITUTIONAL in decision.selected_specialists
