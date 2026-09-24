"""
Specialist registry.

Maps a canonical :class:`SpecialistId` to the object that implements it. The
orchestrator will dispatch through a registry instead of an if/elif chain, so
adding a specialist later means registering it, not editing the orchestrator.

Behaviour, all deterministic:

* ``register`` accepts only :class:`Specialist` instances with a canonical id;
* registering an id twice raises :class:`DuplicateSpecialistError` unless the
  caller passes ``replace=True`` explicitly;
* ``resolve`` of an unregistered id raises :class:`UnknownSpecialistError`;
  it never substitutes another specialist.

There is no module-level registry instance. Whoever needs one creates it, so
tests cannot leak registrations into each other and import has no side
effects.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Union

from .base import Specialist
from .contracts import SpecialistId


class UnknownSpecialistError(KeyError):
    """Raised when an id is not canonical or nothing is registered for it."""

    def __str__(self) -> str:                      # KeyError quotes its argument; keep the message readable
        return str(self.args[0]) if self.args else ""


class DuplicateSpecialistError(ValueError):
    """Raised when an id is registered a second time without ``replace=True``."""


class SpecialistRegistry:
    """A small, explicit map from specialist id to implementation."""

    def __init__(self) -> None:
        self._specialists: Dict[SpecialistId, Specialist] = {}

    def register(self, specialist: Specialist, *, replace: bool = False) -> SpecialistId:
        """Register ``specialist`` under its own canonical id and return that id."""
        if not isinstance(specialist, Specialist):
            raise TypeError("only Specialist instances can be registered, got %s"
                            % type(specialist).__name__)
        declared = getattr(type(specialist), "specialist_id", None)
        if declared is None:
            raise ValueError("%s does not declare a specialist_id" % type(specialist).__name__)
        specialist_id = SpecialistId.parse(declared)
        if specialist_id in self._specialists and not replace:
            raise DuplicateSpecialistError(
                "specialist %r is already registered (%s); pass replace=True to override"
                % (specialist_id.value, type(self._specialists[specialist_id]).__name__)
            )
        self._specialists[specialist_id] = specialist
        return specialist_id

    def resolve(self, specialist_id: Union[SpecialistId, str]) -> Specialist:
        """Return the implementation for ``specialist_id``; never a substitute."""
        try:
            key = SpecialistId.parse(specialist_id)
        except ValueError as exc:
            raise UnknownSpecialistError(str(exc)) from None
        try:
            return self._specialists[key]
        except KeyError:
            raise UnknownSpecialistError(
                "no specialist registered for %r (registered: %s)"
                % (key.value, ", ".join(s.value for s in self.ids()) or "none")
            ) from None

    def is_registered(self, specialist_id: Union[SpecialistId, str]) -> bool:
        try:
            return SpecialistId.parse(specialist_id) in self._specialists
        except ValueError:
            return False

    def ids(self) -> List[SpecialistId]:
        """Registered ids in canonical declaration order."""
        return [member for member in SpecialistId if member in self._specialists]

    def __contains__(self, specialist_id: object) -> bool:
        return isinstance(specialist_id, (SpecialistId, str)) and self.is_registered(specialist_id)

    def __iter__(self) -> Iterator[SpecialistId]:
        return iter(self.ids())

    def __len__(self) -> int:
        return len(self._specialists)


__all__ = ["DuplicateSpecialistError", "SpecialistRegistry", "UnknownSpecialistError"]
