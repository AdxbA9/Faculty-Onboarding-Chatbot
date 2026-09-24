"""Plan F contracts: construction, validation, canonical vocabularies and
serialisation. Pure dataclass logic; no models, no network."""
from __future__ import annotations

import json

import pytest

from handbook_bot.agents.specialists.contracts import (
    ALL_FINDING_STATUSES,
    ALL_SPECIALIST_IDS,
    CoordinatorDecision,
    Finding,
    FindingStatus,
    HandoffRequest,
    SpecialistFindings,
    SpecialistId,
    SpecialistTask,
)

QUOTE = "Faculty members are entitled to thirty days of annual leave each academic year."


def task(**overrides) -> SpecialistTask:
    fields = dict(task_id="t1", question="How much annual leave do faculty get?",
                  specialist_id=SpecialistId.FACULTY_SERVICES, domain="faculty_services")
    fields.update(overrides)
    return SpecialistTask(**fields)


def finding(**overrides) -> Finding:
    fields = dict(finding_id="f1", claim="Faculty get thirty days of annual leave per year.",
                  evidence_quote=QUOTE, source_id="handbook-25-26", source_title="Faculty Handbook 2025/26",
                  page=107, confidence=0.9)
    fields.update(overrides)
    return Finding(**fields)


# ---------------------------------------------------------------------------
# Canonical specialist ids
# ---------------------------------------------------------------------------
def test_canonical_specialist_ids_are_exactly_the_four_plan_f_specialists():
    assert ALL_SPECIALIST_IDS == ("teaching", "research", "faculty_services", "institutional")
    assert [m.value for m in SpecialistId] == list(ALL_SPECIALIST_IDS)


def test_specialist_id_display_names():
    assert SpecialistId.TEACHING.display_name == "Teaching & Learning"
    assert SpecialistId.RESEARCH.display_name == "Research & Innovation"
    assert SpecialistId.FACULTY_SERVICES.display_name == "Faculty Services & HR"
    assert SpecialistId.INSTITUTIONAL.display_name == "Institutional Navigation"


def test_specialist_id_parse_accepts_member_or_string():
    assert SpecialistId.parse("teaching") is SpecialistId.TEACHING
    assert SpecialistId.parse(" Faculty_Services ") is SpecialistId.FACULTY_SERVICES
    assert SpecialistId.parse(SpecialistId.RESEARCH) is SpecialistId.RESEARCH
    assert SpecialistId.TEACHING == "teaching"          # str-enum: comparable with its value


@pytest.mark.parametrize("bad", ["orientation", "blackboard", "banner", "coordinator", "", None, 3])
def test_non_canonical_specialist_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        SpecialistId.parse(bad)


# ---------------------------------------------------------------------------
# Statuses
# ---------------------------------------------------------------------------
def test_finding_statuses_are_explicit():
    assert ALL_FINDING_STATUSES == ("supported", "partial", "not_found", "out_of_scope", "error")
    assert FindingStatus.parse("NOT_FOUND") is FindingStatus.NOT_FOUND


@pytest.mark.parametrize("bad", ["ok", "success", "unknown", "", None])
def test_invalid_status_is_rejected(bad):
    with pytest.raises(ValueError):
        FindingStatus.parse(bad)
    with pytest.raises(ValueError):
        SpecialistFindings(SpecialistId.TEACHING, "t1", bad)


# ---------------------------------------------------------------------------
# SpecialistTask
# ---------------------------------------------------------------------------
def test_specialist_task_construction_and_defaults():
    t = SpecialistTask(task_id="t1", question="How do I upload a lecture to Blackboard?")
    assert t.specialist_id is None
    assert t.source_scope == [] and t.context == {} and t.metadata == {}
    assert t.requested_by == "coordinator"
    assert t.domain is None and t.intent is None and t.level is None and t.system is None


def test_specialist_task_normalises_specialist_id_from_string():
    t = task(specialist_id="teaching", system="blackboard", level="university", intent="procedure")
    assert t.specialist_id is SpecialistId.TEACHING
    assert t.to_dict()["specialist_id"] == "teaching"
    assert t.to_dict()["system"] == "blackboard"


@pytest.mark.parametrize("overrides", [
    {"task_id": ""},
    {"task_id": "   "},
    {"question": ""},
    {"specialist_id": "orientation"},
    {"source_scope": "handbook"},          # a bare string is not a list of sources
    {"source_scope": [1, 2]},
    {"context": ["not", "a", "dict"]},
    {"metadata": None},
    {"requested_by": ""},
    {"domain": 5},
])
def test_specialist_task_rejects_malformed_values(overrides):
    with pytest.raises(ValueError):
        task(**overrides)


# ---------------------------------------------------------------------------
# Finding: claim and evidence_quote stay separate
# ---------------------------------------------------------------------------
def test_finding_construction():
    f = finding()
    assert f.page == 107 and f.confidence == 0.9 and f.metadata == {}
    assert f.source_title == "Faculty Handbook 2025/26"


def test_claim_and_evidence_quote_are_separate_fields():
    f = finding(claim="Annual leave is thirty days.", evidence_quote=QUOTE)
    assert f.claim != f.evidence_quote
    assert f.claim == "Annual leave is thirty days."
    assert f.evidence_quote == QUOTE
    d = f.to_dict()
    assert d["claim"] == f.claim and d["evidence_quote"] == f.evidence_quote
    assert set(d) >= {"claim", "evidence_quote"}
    f.claim = "A different wording of the claim."
    assert f.evidence_quote == QUOTE                    # editing one never touches the other


def test_finding_requires_a_non_empty_quote_and_a_non_empty_claim():
    with pytest.raises(ValueError):
        finding(evidence_quote="")            # a finding without evidence is not grounded
    with pytest.raises(ValueError):
        finding(evidence_quote="   ")
    with pytest.raises(ValueError):
        finding(claim="")


@pytest.mark.parametrize("overrides", [
    {"finding_id": ""},
    {"source_id": ""},
    {"page": 0},
    {"page": -3},
    {"page": True},
    {"page": "107"},
    {"confidence": 1.5},
    {"confidence": -0.1},
    {"confidence": "high"},
    {"evidence_quote": None},
    {"metadata": "x"},
])
def test_finding_rejects_malformed_values(overrides):
    with pytest.raises(ValueError):
        finding(**overrides)


# ---------------------------------------------------------------------------
# SpecialistFindings, one test per status
# ---------------------------------------------------------------------------
def test_supported_findings():
    r = SpecialistFindings(SpecialistId.FACULTY_SERVICES, "t1", FindingStatus.SUPPORTED,
                           findings=[finding()], summary="Thirty days of annual leave.", confidence=0.9)
    assert r.status is FindingStatus.SUPPORTED
    assert len(r.findings) == 1 and r.missing == [] and r.requested_agents == []


def test_partial_findings_name_what_is_missing():
    r = SpecialistFindings("faculty_services", "t1", "partial", findings=[finding()],
                           missing=["application procedure"], confidence=0.6)
    assert r.status is FindingStatus.PARTIAL
    assert r.missing == ["application procedure"]


def test_not_found_findings_carry_no_findings():
    r = SpecialistFindings(SpecialistId.TEACHING, "t1", FindingStatus.NOT_FOUND,
                           limitations=["Blackboard enrolment is not documented in the approved sources."])
    assert r.status is FindingStatus.NOT_FOUND
    assert r.findings == [] and r.limitations


def test_out_of_scope_findings_can_request_a_handoff():
    request = HandoffRequest(SpecialistId.TEACHING, SpecialistId.INSTITUTIONAL, "who approves the change")
    r = SpecialistFindings(SpecialistId.TEACHING, "t1", FindingStatus.OUT_OF_SCOPE, handoff_requests=[request])
    assert r.status is FindingStatus.OUT_OF_SCOPE
    assert r.requested_agents == [SpecialistId.INSTITUTIONAL]
    assert r.to_dict()["requested_agents"] == ["institutional"]


def test_error_findings():
    r = SpecialistFindings(SpecialistId.RESEARCH, "t1", FindingStatus.ERROR,
                           metadata={"error": "TimeoutError: specialist exceeded 8s"})
    assert r.status is FindingStatus.ERROR and r.findings == []


def test_supported_status_requires_at_least_one_finding():
    with pytest.raises(ValueError):
        SpecialistFindings(SpecialistId.TEACHING, "t1", FindingStatus.SUPPORTED, findings=[])


def test_supported_status_means_complete_so_missing_must_be_empty():
    with pytest.raises(ValueError):
        SpecialistFindings(SpecialistId.TEACHING, "t1", FindingStatus.SUPPORTED,
                           findings=[finding()], missing=["application procedure"])


def test_partial_status_requires_at_least_one_finding():
    with pytest.raises(ValueError):
        SpecialistFindings(SpecialistId.TEACHING, "t1", FindingStatus.PARTIAL, missing=["everything"])


def test_finding_ids_must_be_unique_within_a_result():
    with pytest.raises(ValueError):
        SpecialistFindings(SpecialistId.TEACHING, "t1", "supported",
                           findings=[finding(finding_id="f1"), finding(finding_id="f1")])


def test_handoff_requests_must_come_from_the_same_specialist():
    foreign = HandoffRequest(SpecialistId.RESEARCH, SpecialistId.INSTITUTIONAL, "approval chain")
    with pytest.raises(ValueError):
        SpecialistFindings(SpecialistId.TEACHING, "t1", "out_of_scope", handoff_requests=[foreign])


@pytest.mark.parametrize("overrides", [
    {"task_id": ""},
    {"findings": [{"claim": "dict, not Finding"}]},
    {"confidence": 2},
    {"llm_used": "yes"},
    {"ms": -1},
    {"missing": "procedure"},
    {"handoff_requests": ["institutional"]},
])
def test_specialist_findings_rejects_malformed_values(overrides):
    fields = dict(specialist_id=SpecialistId.TEACHING, task_id="t1", status="not_found")
    fields.update(overrides)
    with pytest.raises(ValueError):
        SpecialistFindings(**fields)


# ---------------------------------------------------------------------------
# HandoffRequest
# ---------------------------------------------------------------------------
def test_handoff_request_contract():
    t = task(task_id="t2", specialist_id="institutional", requested_by="teaching")
    h = HandoffRequest("teaching", "institutional", "who approves a load reduction", task=t, priority=1)
    assert h.from_specialist is SpecialistId.TEACHING
    assert h.requested_specialist is SpecialistId.INSTITUTIONAL
    d = h.to_dict()
    assert d["from_specialist"] == "teaching" and d["requested_specialist"] == "institutional"
    assert d["task"]["task_id"] == "t2" and d["priority"] == 1


def test_handoff_request_to_self_is_rejected():
    with pytest.raises(ValueError):
        HandoffRequest(SpecialistId.TEACHING, SpecialistId.TEACHING, "loop")


@pytest.mark.parametrize("kwargs", [
    dict(from_specialist="teaching", requested_specialist="institutional", reason=""),
    dict(from_specialist="teaching", requested_specialist="orientation", reason="x"),
    dict(from_specialist="teaching", requested_specialist="research", reason="x", task="t1"),
    dict(from_specialist="teaching", requested_specialist="research", reason="x", priority="high"),
])
def test_handoff_request_rejects_malformed_values(kwargs):
    with pytest.raises(ValueError):
        HandoffRequest(**kwargs)


# ---------------------------------------------------------------------------
# CoordinatorDecision (contract only)
# ---------------------------------------------------------------------------
def test_coordinator_decision_contract():
    subtasks = [task(task_id="t1", specialist_id="research", question="Is the grant recognised?"),
                task(task_id="t2", specialist_id="teaching", question="Can my load be reduced?")]
    d = CoordinatorDecision(domains=["research", "teaching"], intents=["policy"], level="university",
                            systems=[], selected_specialists=["research", "teaching"], subtasks=subtasks,
                            requires_synthesis=True, confidence=0.8, reason="cross-domain question")
    assert d.selected_specialists == [SpecialistId.RESEARCH, SpecialistId.TEACHING]
    assert d.requires_synthesis is True and d.used_llm is False
    out = d.to_dict()
    assert out["selected_specialists"] == ["research", "teaching"]
    assert [t["task_id"] for t in out["subtasks"]] == ["t1", "t2"]


def test_empty_coordinator_decision_is_valid():
    d = CoordinatorDecision()
    assert d.selected_specialists == [] and d.subtasks == [] and d.confidence == 0.0


def test_coordinator_decision_subtasks_must_target_selected_specialists():
    with pytest.raises(ValueError):
        CoordinatorDecision(selected_specialists=["teaching"],
                            subtasks=[task(task_id="t1", specialist_id="research")])
    with pytest.raises(ValueError):
        CoordinatorDecision(selected_specialists=["teaching"],
                            subtasks=[task(task_id="t1", specialist_id=None)])


@pytest.mark.parametrize("kwargs", [
    dict(selected_specialists=["teaching", "teaching"]),
    dict(selected_specialists=["orientation"]),
    dict(selected_specialists="teaching"),
    dict(subtasks=[task(task_id="t1", specialist_id="teaching"), task(task_id="t1", specialist_id="teaching")],
         selected_specialists=["teaching"]),
    dict(requires_synthesis="no"),
    dict(confidence=7),
    dict(domains="teaching"),
])
def test_coordinator_decision_rejects_malformed_values(kwargs):
    with pytest.raises(ValueError):
        CoordinatorDecision(**kwargs)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def test_every_contract_serialises_to_plain_json():
    t = task(task_id="t1", specialist_id="teaching")
    request = HandoffRequest("teaching", "institutional", "approval", task=task(task_id="t2", specialist_id="institutional"))
    r = SpecialistFindings("teaching", "t1", "partial", findings=[finding()], missing=["approver"],
                           handoff_requests=[request], confidence=0.7, llm_used=True, ms=12.5)
    d = CoordinatorDecision(selected_specialists=["teaching"], subtasks=[t], confidence=1.0)
    for obj in (t, finding(), request, r, d):
        text = json.dumps(obj.to_dict())
        assert isinstance(json.loads(text), dict)
    assert json.loads(json.dumps(r.to_dict()))["status"] == "partial"
    assert json.loads(json.dumps(r.to_dict()))["specialist_id"] == "teaching"
