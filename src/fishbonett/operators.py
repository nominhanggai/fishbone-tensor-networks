"""Spin (Pauli) and bosonic operators, plus small utilities.

Formerly part of the catch-all ``fishbonett.stuff`` module, now split into this
operator module and :mod:`fishbonett.spectral_densities`.
"""
import numpy as np

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


def _c(dim: int):
    """Bosonic annihilation operator on a ``dim``-level Fock truncation."""
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op


def _num(dim: int):
    """Bosonic number operator ``a^dagger a`` on a ``dim``-level truncation."""
    return _c(dim).T @ _c(dim)
