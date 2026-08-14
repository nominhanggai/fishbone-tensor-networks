"""Structural interfaces emitted by Hamiltonian frames.

Frames intentionally do not share one concrete output type: a static graph, an
MPO, a swap-gate factory and a conditional displacement expose different useful
structure.  These runtime-checkable protocols let plan compilers state and verify
the capability they consume without coupling to a particular frame class.
"""
from typing import Protocol, runtime_checkable

__all__ = [
    "MPOHamiltonian", "StaticGraphHamiltonian", "StaticGateFactory",
    "SwapGateFactory", "DisplacementFactory", "require_capability",
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


def require_capability(frame, capability, *, engine):
    """Return ``frame`` or fail early with an engine-specific contract error."""
    if not isinstance(frame, capability):
        raise TypeError(
            f"engine {engine!r} requires frame capability "
            f"{capability.__name__}; got {type(frame).__name__}")
    return frame
