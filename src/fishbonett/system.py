"""The system: the non-bath half of an open-quantum-system model.

:class:`Bath` says what the environment is; :class:`System` says what is coupled to
it.  Together they are the two inputs every model takes.

The package supports an **arbitrary** system throughout -- ``h`` may be any ``(d, d)``
Hermitian matrix, not just a two-level spin, and ``coupling`` any ``(d, d)`` Hermitian
operator, not just ``sigma_z``.  This class is where that promise is checked, once,
so the representations and models do not each re-derive it.

.. rubric:: API

======================  ========================================================
:class:`System`         ``h`` + ``coupling`` + ``initial``, validated
:func:`check_operator`  one ``(d, d)`` Hermitian operator, for callers holding
                        loose arrays rather than a :class:`System`
======================  ========================================================
"""
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["System", "check_operator"]


def check_operator(
    op: ArrayLike, name: str, dim: int | None = None, hermitian: bool = True,
) -> np.ndarray:
    """Validate and return one operator as a complex ``(d, d)`` array.

    Raises :class:`ValueError` naming ``name`` when the operator is not square,
    not of dimension ``dim`` (when given), or not Hermitian (when required).
    The returned copy is read-only so a model's validated Hamiltonian cannot be
    mutated behind its cached representations; make an explicit copy before
    editing it.
    """
    a = np.array(op, dtype=complex, copy=True)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f"{name} must be a non-empty square matrix, got shape {a.shape}"
        )
    if dim is not None and a.shape[0] != dim:
        raise ValueError(f"{name} has shape {a.shape}, expected {(dim, dim)}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    if hermitian and not np.allclose(a, a.conj().T, atol=1e-9):
        raise ValueError(f"{name} must be Hermitian")
    a.setflags(write=False)
    return a


@dataclass
class System:
    """An arbitrary quantum system coupled to a bath.

    Parameters
    ----------
    h : (d, d) array
        The system Hamiltonian.  Any Hermitian matrix; not restricted to two levels.
    coupling : (d, d) array or list of them
        The Hermitian operator(s) through which the bath couples.  A *list* means a
        multichannel bath -- several operators sharing one set of modes, which is a
        different physical situation from several independent baths (see
        :mod:`fishbonett.representations.multichannel`).
    initial : str or (d,) array
        ``"up"`` / ``"down"`` (the first two basis states), ``"ground"`` (the ground
        state of ``h``), or an explicit state vector.  Normalized on use.
    """

    h: ArrayLike
    coupling: ArrayLike | Sequence[ArrayLike]
    initial: str | ArrayLike = "up"

    #: Filled in by ``__post_init__``; the system's Hilbert-space dimension.
    dim: int = field(init=False)

    def __post_init__(self):
        """Validate the Hamiltonian, coupling channels, and initial-state input."""
        self.h = check_operator(self.h, "h")
        self.dim = self.h.shape[0]
        if self.is_multichannel:
            if not self.coupling:
                raise ValueError("coupling must contain at least one operator")
            self.coupling = tuple(
                check_operator(o, f"coupling[{i}]", self.dim)
                for i, o in enumerate(self.coupling)
            )
        else:
            self.coupling = check_operator(self.coupling, "coupling", self.dim)

    @property
    def is_multichannel(self):
        """True when the bath couples through several operators at once."""
        if not isinstance(self.coupling, (list, tuple)):
            return False
        if len(self.coupling) == 0:
            return True
        try:
            return np.asarray(self.coupling).ndim != 2
        except ValueError:
            return True

    def initial_vector(self, initial=None):
        """The initial state as a normalized ``(dim,)`` complex vector.

        ``initial`` overrides the one given at construction, which is what ``run``
        does -- the system is declared once, but each propagation may start from a
        different state.
        """
        d = self.dim
        init = self.initial if initial is None else initial
        if isinstance(init, str):
            if init == "up":
                v = np.zeros(d, complex); v[0] = 1.0; return v
            if init == "down":
                v = np.zeros(d, complex); v[min(1, d - 1)] = 1.0; return v
            if init == "ground":
                w, U = np.linalg.eigh(self.h)
                return U[:, int(np.argmin(w))].astype(complex)
            raise ValueError(
                f"unknown initial state {init!r}; use 'up', 'down', 'ground', or a "
                f"length-{d} vector")
        v = np.asarray(init, complex).reshape(-1)
        if v.shape[0] != d:
            raise ValueError(f"initial state has length {v.shape[0]}, expected the "
                             f"system dimension {d}")
        if not np.all(np.isfinite(v)):
            raise ValueError("initial state must contain only finite values")
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("initial state must have a finite non-zero norm")
        return v / norm

    def observables(self):
        """The default observables: Pauli z/x for a two-level system, else none.

        A general system has no canonical set, so ``run`` returns only the reduced
        density matrix unless observables are named explicitly.
        """
        if self.dim != 2:
            return {}
        from fishbonett.operators import sigma_x, sigma_z
        return {"sz": sigma_z, "sx": sigma_x}
