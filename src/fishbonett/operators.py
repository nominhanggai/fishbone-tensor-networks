"""Spin (Pauli) and bosonic operators, plus entropy helpers.

The single home for the small dense matrices every other module needs.  The
bosonic ladder operators in particular used to be redefined in four places
(``crea``/``anih``/``numb`` in the propagators, ``c_``/``_c`` in the builders);
they are defined **once** here and imported everywhere else.

.. rubric:: What's here

=========================================  ==================================
:data:`sigma_x`, :data:`sigma_y`, ...      Pauli matrices and the 2x2 identity
:func:`annihilate`, :func:`create`         bosonic ladder operators ``b``, ``b^dag``
:func:`number`                             ``b^dag b`` on a truncated Fock space
:func:`temp_factor`                        thermal weight for thermofield doubling
:func:`entang`, :func:`rlogr`              von Neumann entropy from Schmidt values
=========================================  ==================================

All the bosonic operators act on a **truncated** ``dim``-level Fock space, so
``[b, b^dag] != 1`` at the top of the ladder.  Operator-norm identities are
therefore not exact near the truncation edge; convergence must be checked by
raising ``phys_dim`` and watching the dynamics, not by testing such identities.
"""
import numpy as np

__all__ = ["sigma_p", "sigma_m", "sigma_x", "sigma_y", "sigma_z", "sigma_0",
           "sigma_1", "annihilate", "create", "number", "temp_factor",
           "rlogr", "entang"]

# -- Pauli matrices ----------------------------------------------------------
sigma_p = np.float64([[0, 1], [0, 0]])        # raising  S^+
sigma_m = np.float64([[0, 0], [1, 0]])        # lowering S^-
sigma_x = np.float64([[0, 1], [1, 0]])
sigma_y = np.complex64([[0, -1j], [1j, 0]])
sigma_z = np.float64([[1, 0], [0, -1]])
sigma_0 = np.zeros((2, 2))                    # zero matrix
sigma_1 = np.eye(2)                           # identity


def temp_factor(temp, w):
    """Thermal factor ``0.5(1 + coth(beta w / 2))`` (temperature in kelvin)."""
    beta = 1 / (0.6950348009119888 * temp)
    return 0.5 * (1. + 1. / np.tanh(beta * w / 2.))


def rlogr(si):
    """``-s log s`` (a single Schmidt term of the entanglement entropy)."""
    return (-1) * si * np.log(si)


def entang(s):
    """von Neumann entanglement entropy from a list of Schmidt values ``s``."""
    etg = 0.0
    for si in s:
        if si != 0:
            etg += rlogr(si ** 2)
    return etg


def annihilate(dim: int):
    """Bosonic annihilation operator ``b`` on a ``dim``-level Fock truncation.

    ``b|n> = sqrt(n)|n-1>``, i.e. the real upper-bidiagonal matrix with
    ``sqrt(1), sqrt(2), ...`` above the diagonal.
    """
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op


def create(dim: int):
    """Bosonic creation operator ``b^dagger`` -- the transpose of
    :func:`annihilate` (which is real)."""
    return annihilate(dim).T


def number(dim: int):
    """Bosonic number operator ``b^dagger b = diag(0, 1, ..., dim-1)``."""
    return annihilate(dim).T @ annihilate(dim)
