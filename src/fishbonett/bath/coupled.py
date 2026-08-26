"""Binding between a bath specification and its system coupling operators."""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from fishbonett.bath.spec import Bath

__all__ = ["CoupledBath", "bind_bath"]


def _operators(value: ArrayLike | Sequence[ArrayLike]) -> tuple[np.ndarray, ...]:
    try:
        is_matrix = np.asarray(value).ndim == 2
    except ValueError:
        is_matrix = False
    values = (
        [value] if is_matrix or not isinstance(value, (list, tuple)) else value
    )
    out = []
    for op in values:
        item = np.array(op, dtype=complex, copy=True)
        item.setflags(write=False)
        out.append(item)
    return tuple(out)


def _same_operators(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(a.shape == b.shape and np.array_equal(a, b)
               for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class CoupledBath:
    """A bath plus the system operators through which it acts.

    The wrapped :attr:`bath` contains environment physics and numerical settings;
    :attr:`operators` belongs to the model.  Multiple operators denote channels
    sharing one set of modes.
    """

    bath: Bath
    operators: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the coupling operators in normalized tuple form."""
        ops = _operators(self.operators)
        if not ops:
            raise ValueError("a coupled bath needs at least one operator")
        shape = ops[0].shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("coupling operators must be square matrices")
        for op in ops:
            if op.shape != shape:
                raise ValueError("all coupling operators must have the same shape")
            if not np.allclose(op, op.conj().T):
                raise ValueError("coupling operators must be Hermitian")
        n_densities = len(self.bath.spectral_densities())
        if n_densities not in (1, len(ops)):
            raise ValueError(
                "Bath.J must be one shared spectral density or one density per "
                f"coupling operator; got {n_densities} densities and {len(ops)} "
                "operators")
        if len(ops) > 1 and self.bath.discretization != "legendre":
            raise ValueError(
                "a multichannel bath must use the 'legendre' discretization: its "
                "Gauss nodes are shared across channels, whereas measure-adapted "
                "TEDOPA nodes are not")
        object.__setattr__(self, "operators", ops)

    @property
    def is_multichannel(self) -> bool:
        """Whether several system operators share this bath's modes."""
        return len(self.operators) > 1

    @property
    def operator(self) -> np.ndarray:
        """Return the sole coupling operator, rejecting multichannel bindings."""
        if self.is_multichannel:
            raise ValueError("this bath has several coupling operators")
        return self.operators[0]

    @property
    def n_modes(self) -> int | None:
        """Configured or resolved number of represented bath modes."""
        return self.bath.n_modes

    @property
    def phys_dim(self) -> int:
        """Local Fock-space dimension of every represented bath mode."""
        return self.bath.phys_dim

    @property
    def domain(self) -> tuple[float, float] | None:
        """Configured or resolved bath-frequency interval."""
        return self.bath.domain

    def resolved(self, t_max: float | None = None) -> CoupledBath:
        """Return a binding whose automatic bath settings cover ``t_max``."""
        bath = self.bath.resolved(t_max)
        if bath is self.bath:
            return self
        # Bath is intentionally a user-editable specification.  Do not cache a
        # resolved copy here: mutating an automatic domain, mode count, or
        # spectral-density callable between runs must not return stale physics.
        return replace(self, bath=bath)


def bind_bath(
    bath: Bath | CoupledBath,
    coupling: ArrayLike | Sequence[ArrayLike] | None = None,
) -> CoupledBath:
    """Return a :class:`CoupledBath` with an explicit coupling."""
    if isinstance(bath, CoupledBath):
        if coupling is not None:
            supplied = _operators(coupling)
            if not _same_operators(bath.operators, supplied):
                raise ValueError(
                    "coupling was supplied both by CoupledBath and separately, "
                    "and the values differ")
        return bath

    if coupling is None:
        raise ValueError(
            "no system-bath coupling operator was supplied; use "
            "bath.bind(operator)"
        )
    return CoupledBath(bath=bath, operators=_operators(coupling))
