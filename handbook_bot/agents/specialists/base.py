"""
Specialist base class.

A specialist owns one domain and one decision: whether its scoped evidence
answers the task it was given, expressed as a :class:`SpecialistFindings`.
Every real specialist (Teaching & Learning, Research & Innovation, Faculty
Services & HR, Institutional Navigation) subclasses :class:`Specialist` and
implements :meth:`Specialist.run`.

Rules a specialist must respect (the orchestrator enforces them at the
boundary with :func:`check_findings`):

* it answers the task id it was given, under its own specialist id;
* it never runs another specialist; it may return handoff requests;
* it never returns free-form prose as the result; findings carry the claim
  and the evidence quote separately.

This module holds no retrieval, no prompts and no LLM calls: those arrive
with the specialists themselves in later phases.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import ClassVar, Union

from .contracts import SpecialistFindings, SpecialistId, SpecialistTask

# ---------------------------------------------------------------------------
# Error redaction (one implementation for the dispatcher and every specialist)
# ---------------------------------------------------------------------------
_KEY_SHAPED = re.compile(r"gsk_[A-Za-z0-9]+")
#: A directory name may hold spaces ("C:\Users\John Smith\", "/home/john smith/"): a
#: spaced segment counts only when a separator follows it, or as the final name when it
#: is closed by a quote or carries a file extension (Step 4.5A, SAF-4).
_SEGMENT = r"[^\\/\s'\"<>|]+(?: +[^\\/\s'\"<>|]+)*"
_LAST = r"(?:" + _SEGMENT + r"(?=['\"])|" + _SEGMENT + r"\.[A-Za-z0-9]{1,5}(?!\w)|[^\s'\"<>|\\/]*)"
_PATH_SHAPED = re.compile(
    r"[A-Za-z]:[\\/]+(?:" + _SEGMENT + r"[\\/]+)*" + _LAST +                 # Windows drive paths
    r"|[\\]{2,}[\w.$-]+[\\]+(?:" + _SEGMENT + r"[\\]+)*" + _LAST +          # UNC paths (\\server\share\..., also as an escaped repr)
    r"|(?<![\w:])//[\w.$-]+/+(?:" + _SEGMENT + r"/+)*" + _LAST +            # UNC paths written with forward slashes
    r"|(?<![\w:./])/(?:home|users|tmp|var|etc|opt|mnt|root|srv|usr|c|d|data|app|workspace)/(?:" + _SEGMENT + r"/)*" + _LAST +   # Unix homes and system trees
    r"|(?<![\w:./])/[\w.-]+/(?:" + _SEGMENT + r"/)*[^\s'\"<>|/]+",       # any other absolute path
    re.I)


def redact_error(exc: BaseException) -> str:
    """The exception type and a short message with key-shaped secrets and
    filesystem paths removed. Used by the dispatcher for a raising
    specialist and by every specialist for its own ``error`` results."""
    text = _KEY_SHAPED.sub("gsk_***", str(exc))
    text = _PATH_SHAPED.sub("<path>", text)
    text = re.sub(r"\s+", " ", text)[:160]
    return "%s: %s" % (type(exc).__name__, text) if text else type(exc).__name__


class Specialist(ABC):
    """Interface implemented by every Plan F domain specialist.

    Subclasses set ``specialist_id`` to their canonical id. It is validated
    when the subclass is created, so a typo fails at import time rather than
    at the first question.
    """

    #: Canonical id of the specialist. Must be set by every concrete subclass.
    specialist_id: ClassVar[SpecialistId]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("specialist_id", None)
        if declared is not None:
            cls.specialist_id = SpecialistId.parse(declared)

    @property
    def display_name(self) -> str:
        return self.specialist_id.display_name

    @abstractmethod
    def run(self, task: SpecialistTask) -> SpecialistFindings:
        """Answer ``task`` from this specialist's scoped evidence.

        Must return a :class:`SpecialistFindings` whose ``specialist_id`` is
        this specialist's id and whose ``task_id`` equals ``task.task_id``.
        It may raise: the orchestrator turns exceptions into an ``error``
        status. It must not call another specialist.
        """


def check_findings(
    specialist: Union[Specialist, SpecialistId, str],
    task: SpecialistTask,
    findings: object,
) -> SpecialistFindings:
    """Validate what a specialist returned for ``task``.

    Raises ``TypeError`` when it is not a :class:`SpecialistFindings` and
    ``ValueError`` when it answers a different task or carries a different
    specialist id. Returns the findings unchanged when they are consistent.
    """
    expected = specialist.specialist_id if isinstance(specialist, Specialist) else SpecialistId.parse(specialist)
    if not isinstance(findings, SpecialistFindings):
        raise TypeError("specialist %s returned %s, expected SpecialistFindings"
                        % (expected.value, type(findings).__name__))
    if findings.specialist_id != expected:
        raise ValueError("findings carry specialist id %s, expected %s"
                         % (findings.specialist_id.value, expected.value))
    if findings.task_id != task.task_id:
        raise ValueError("findings answer task %r, expected %r" % (findings.task_id, task.task_id))
    return findings


__all__ = ["Specialist", "check_findings", "redact_error"]
