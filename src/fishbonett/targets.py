"""Semantic targets for observables on represented bath modes."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class BathMode:
    """Address one represented bath mode without relying on tensor-node numbers.

    Parameters are zero based. ``system_site`` selects the system site, ``bath``
    selects one of the independent baths attached to that site in input order,
    and ``mode`` selects a mode in that bath's chosen representation.
    """

    system_site: int
    bath: int = 0
    mode: int = 0

    def __post_init__(self):
        """Require non-negative integer indices for every target coordinate."""
        for name in ("system_site", "bath", "mode"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))


__all__ = ["BathMode"]
