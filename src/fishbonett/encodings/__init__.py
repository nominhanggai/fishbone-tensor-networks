"""Numerical encodings of Hamiltonians and finite-step propagators.

Representations describe the transformed Hamiltonian.  This package translates
that mathematical object into the data structure a tensor-network algorithm
consumes: local terms, Trotter gates, an MPO, or a factorized propagator.  Keeping
this layer separate prevents a representation from depending on TEBD, TDVP, or a
particular tensor-network geometry.
"""

from fishbonett.encodings.capabilities import (
    DisplacementFactory,
    MPOHamiltonian,
    StaticGateFactory,
    StaticGraphHamiltonian,
    SwapGateFactory,
    require_capability,
)

__all__ = [
    "DisplacementFactory",
    "MPOHamiltonian",
    "StaticGateFactory",
    "StaticGraphHamiltonian",
    "SwapGateFactory",
    "require_capability",
]
