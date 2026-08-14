"""Structural interfaces implemented by numerical Hamiltonian encodings."""

from typing import Protocol, runtime_checkable

__all__ = [
    "MPOHamiltonian",
    "StaticGraphHamiltonian",
    "StaticGateFactory",
    "SwapGateFactory",
    "DisplacementFactory",
    "require_capability",
]


@runtime_checkable
class MPOHamiltonian(Protocol):
    n_sites: int
    phys_dim: int
    system: tuple
    static: bool

    def mpo(self, t=None): ...


@runtime_checkable
class StaticGraphHamiltonian(Protocol):
    dims: list
    edges: list

    def gates(self, dt): ...


@runtime_checkable
class StaticGateFactory(Protocol):
    def gates(self, dt): ...


@runtime_checkable
class SwapGateFactory(Protocol):
    def get_u(self, t, dt, factor=1): ...


@runtime_checkable
class DisplacementFactory(Protocol):
    def displacement_mpo(self, t, delta): ...


def require_capability(encoding, capability, *, engine):
    """Return an encoding or fail with an engine-specific contract error."""
    if not isinstance(encoding, capability):
        raise TypeError(
            f"engine {engine!r} requires encoding capability "
            f"{capability.__name__}; got {type(encoding).__name__}")
    return encoding
