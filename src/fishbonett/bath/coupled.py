"""Binding between a bath specification and system coupling operators.

Historically ``Bath.coupling`` and ``SystemBath(coupling=...)`` could both own the
same operator.  The multichannel drivers followed the former and silently ignored
the latter.  ``CoupledBath`` is the explicit model-level object that removes that
ambiguity while allowing ``Bath.coupling`` as a compatibility input.
"""
from dataclasses import dataclass, replace

import numpy as np

from fishbonett.bath.compiled import StarBath, compile_chain, compile_star

__all__ = ["CoupledBath", "bind_bath"]


def _operators(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for op in values:
        item = np.array(op, dtype=complex, copy=True)
        item.setflags(write=False)
        out.append(item)
    return tuple(out)


def _same_operators(left, right):
    if len(left) != len(right):
        return False
    return all(a.shape == b.shape and np.array_equal(a, b)
               for a, b in zip(left, right))


@dataclass(frozen=True)
class CoupledBath:
    """A bath plus the system operators through which it acts.

    The wrapped :attr:`bath` contains environment physics and numerical settings;
    :attr:`operators` belongs to the model.  Multiple operators denote channels
    sharing one set of modes.
    """

    bath: object
    operators: tuple

    def __post_init__(self):
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
    def is_multichannel(self):
        return len(self.operators) > 1

    @property
    def operator(self):
        if self.is_multichannel:
            raise ValueError("this bath has several coupling operators")
        return self.operators[0]

    @property
    def n_modes(self):
        return self.bath.n_modes

    @property
    def phys_dim(self):
        return self.bath.phys_dim

    @property
    def domain(self):
        return self.bath.domain

    def resolved(self, t_max=None):
        bath = self.bath.resolved(t_max)
        return self if bath is self.bath else replace(self, bath=bath)

    def compiled_star(self):
        star = compile_star(self.bath)
        # One spectral density may be shared by several system operators.  The
        # Bath compiler sees one scalar profile; this model-level binding expands
        # it into one identical profile per channel.
        if star.n_channels == 1 and len(self.operators) > 1:
            strengths = np.repeat(star.couplings, len(self.operators), axis=0)
            return StarBath(star.frequencies, strengths, star.phys_dim)
        return star

    def compiled_chain(self):
        if self.is_multichannel:
            raise ValueError("a multichannel shared bath cannot compile to one chain")
        return compile_chain(self.bath)

    def shared_mode_star(self):
        star = self.compiled_star()
        return star.frequencies, star.combine(self.operators)


def bind_bath(bath, coupling=None, *, default_operator=None,
              validate_legacy=False):
    """Return a :class:`CoupledBath` from new or compatibility-style input.

    ``coupling`` is the model-owned value.  If ``validate_legacy`` is true and the
    old ``Bath.coupling`` field is also populated, both must agree.  This turns the
    former silent-ignore behavior into a construction-time error.
    """
    if isinstance(bath, CoupledBath):
        if coupling is not None:
            supplied = _operators(coupling)
            if not _same_operators(bath.operators, supplied):
                raise ValueError(
                    "coupling was supplied both by CoupledBath and separately, "
                    "and the values differ")
        return bath

    legacy = bath.coupling
    chosen = coupling
    if chosen is None:
        chosen = legacy if legacy is not None else default_operator
    if chosen is None:
        raise ValueError(
            "no system-bath coupling operator was supplied; bind the Bath to an "
            "operator or set the compatibility Bath.coupling field")
    if validate_legacy and legacy is not None:
        model_ops, legacy_ops = _operators(chosen), _operators(legacy)
        if not _same_operators(model_ops, legacy_ops):
            raise ValueError(
                "coupling is specified twice with different values: "
                "SystemBath(coupling=...) owns the model coupling; remove "
                "Bath.coupling or make it identical during migration")
    return CoupledBath(bath=bath, operators=_operators(chosen))
