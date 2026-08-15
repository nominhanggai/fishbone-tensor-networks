"""Binding between a bath specification and its system coupling operators.

``CoupledBath`` keeps environment properties in :class:`~fishbonett.bath.Bath`
and associates them with the operators belonging to a physical model.
``Bath.coupling`` remains available as a deprecated compatibility input.
"""
from dataclasses import dataclass, field, replace
import warnings

import numpy as np

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
    _resolved_cache: dict = field(default_factory=dict, init=False, repr=False,
                                  compare=False)

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
        if bath is self.bath:
            return self
        key = None if t_max is None else float(t_max)
        cached = self._resolved_cache.get(key)
        if cached is None:
            cached = replace(self, bath=bath)
            self._resolved_cache[key] = cached
        return cached


def bind_bath(bath, coupling=None, *, default_operator=None,
              validate_legacy=False):
    """Return a :class:`CoupledBath` from explicit or deprecated input.

    ``coupling`` is the model-owned value. If ``validate_legacy`` is true and the
    deprecated ``Bath.coupling`` field is also populated, both must agree.
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
    if legacy is not None:
        warnings.warn(
            "Bath.coupling is deprecated and will be removed in a future major "
            "release; pass CoupledBath objects to Fishbone/TreeFishbone or use "
            "bath.bind(operator)",
            DeprecationWarning, stacklevel=2)
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
