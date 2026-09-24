"""Plan F specialist base class and registry, exercised with harmless test
doubles that return ``not_found``. No real specialist exists yet."""
from __future__ import annotations

import pytest

from handbook_bot.agents.specialists import registry as registry_module
from handbook_bot.agents.specialists.base import Specialist, check_findings
from handbook_bot.agents.specialists.contracts import (
    FindingStatus,
    SpecialistFindings,
    SpecialistId,
    SpecialistTask,
)
from handbook_bot.agents.specialists.registry import (
    DuplicateSpecialistError,
    SpecialistRegistry,
    UnknownSpecialistError,
)


class _NotFoundStub(Specialist):
    """Test double: answers every task with not_found under its own id."""

    def run(self, task: SpecialistTask) -> SpecialistFindings:
        return SpecialistFindings(self.specialist_id, task.task_id, FindingStatus.NOT_FOUND,
                                  limitations=["test double"])


class TeachingStub(_NotFoundStub):
    specialist_id = SpecialistId.TEACHING


class ResearchStub(_NotFoundStub):
    specialist_id = "research"                      # a string id is normalised at class creation


class InstitutionalStub(_NotFoundStub):
    specialist_id = SpecialistId.INSTITUTIONAL


class FacultyServicesStub(_NotFoundStub):
    specialist_id = SpecialistId.FACULTY_SERVICES


TASK = SpecialistTask(task_id="t1", question="How do I upload a lecture to Blackboard?",
                      specialist_id=SpecialistId.TEACHING)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
def test_specialist_interface_is_enforced():
    class Incomplete(Specialist):
        specialist_id = SpecialistId.TEACHING

    with pytest.raises(TypeError):          # abstract run() not implemented
        Incomplete()


def test_specialist_id_is_validated_when_the_class_is_created():
    with pytest.raises(ValueError):
        class Wrong(_NotFoundStub):
            specialist_id = "orientation"


def test_string_specialist_id_is_normalised_to_the_enum():
    assert ResearchStub.specialist_id is SpecialistId.RESEARCH
    assert ResearchStub().display_name == "Research & Innovation"


def test_run_returns_specialist_findings_for_the_given_task():
    result = TeachingStub().run(TASK)
    assert isinstance(result, SpecialistFindings)
    assert result.specialist_id is SpecialistId.TEACHING
    assert result.task_id == "t1"
    assert result.status is FindingStatus.NOT_FOUND


def test_check_findings_accepts_consistent_results():
    stub = TeachingStub()
    result = stub.run(TASK)
    assert check_findings(stub, TASK, result) is result
    assert check_findings("teaching", TASK, result) is result


def test_check_findings_rejects_wrong_type_specialist_or_task():
    stub = TeachingStub()
    with pytest.raises(TypeError):
        check_findings(stub, TASK, "an answer string")
    foreign = SpecialistFindings(SpecialistId.RESEARCH, "t1", "not_found")
    with pytest.raises(ValueError):
        check_findings(stub, TASK, foreign)
    other_task = SpecialistFindings(SpecialistId.TEACHING, "t99", "not_found")
    with pytest.raises(ValueError):
        check_findings(stub, TASK, other_task)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_register_and_resolve_by_id_or_string():
    reg = SpecialistRegistry()
    stub = TeachingStub()
    assert reg.register(stub) is SpecialistId.TEACHING
    assert reg.resolve(SpecialistId.TEACHING) is stub
    assert reg.resolve("teaching") is stub
    assert reg.is_registered("teaching") and "teaching" in reg and SpecialistId.TEACHING in reg
    assert len(reg) == 1 and list(reg) == [SpecialistId.TEACHING]


def test_unknown_specialist_id_is_rejected_clearly():
    reg = SpecialistRegistry()
    reg.register(TeachingStub())
    with pytest.raises(UnknownSpecialistError) as info:
        reg.resolve("orientation")
    assert "unknown specialist id" in str(info.value)
    assert not reg.is_registered("orientation") and "orientation" not in reg


def test_unregistered_canonical_id_is_rejected_and_never_substituted():
    reg = SpecialistRegistry()
    reg.register(TeachingStub())
    with pytest.raises(UnknownSpecialistError) as info:
        reg.resolve(SpecialistId.RESEARCH)
    assert "research" in str(info.value) and "registered: teaching" in str(info.value)
    assert isinstance(info.value, KeyError)


def test_duplicate_registration_is_rejected_unless_replace_is_explicit():
    reg = SpecialistRegistry()
    first, second = TeachingStub(), TeachingStub()
    reg.register(first)
    with pytest.raises(DuplicateSpecialistError):
        reg.register(second)
    assert reg.resolve("teaching") is first            # the first registration survives
    reg.register(second, replace=True)
    assert reg.resolve("teaching") is second


def test_only_specialist_instances_can_be_registered():
    reg = SpecialistRegistry()
    with pytest.raises(TypeError):
        reg.register("teaching")
    with pytest.raises(TypeError):
        reg.register(TeachingStub)                     # the class, not an instance


def test_subclass_without_an_id_cannot_be_registered():
    class Anonymous(_NotFoundStub):
        pass

    with pytest.raises(ValueError):
        SpecialistRegistry().register(Anonymous())


def test_ids_are_reported_in_canonical_order_regardless_of_registration_order():
    reg = SpecialistRegistry()
    for stub in (InstitutionalStub(), TeachingStub(), FacultyServicesStub(), ResearchStub()):
        reg.register(stub)
    assert reg.ids() == [SpecialistId.TEACHING, SpecialistId.RESEARCH,
                         SpecialistId.FACULTY_SERVICES, SpecialistId.INSTITUTIONAL]
    assert len(reg) == 4


def test_registries_are_independent_and_no_global_instance_exists():
    a, b = SpecialistRegistry(), SpecialistRegistry()
    a.register(TeachingStub())
    assert "teaching" in a and "teaching" not in b
    assert not any(isinstance(v, SpecialistRegistry) for v in vars(registry_module).values())


def test_dispatch_through_registry_needs_no_specialist_specific_code():
    reg = SpecialistRegistry()
    reg.register(TeachingStub())
    reg.register(ResearchStub())
    for task in (SpecialistTask("t1", "q", specialist_id="teaching"),
                 SpecialistTask("t2", "q", specialist_id="research")):
        specialist = reg.resolve(task.specialist_id)
        result = check_findings(specialist, task, specialist.run(task))
        assert result.specialist_id is task.specialist_id and result.task_id == task.task_id
