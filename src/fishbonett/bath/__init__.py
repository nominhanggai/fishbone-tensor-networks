"""Everything about the bath: what it is, how it is discretized, how it is mapped.

The pipeline runs in one direction -- a continuous spectral density becomes a
star of independent modes, and the star becomes a nearest-neighbour chain::

    Bath(J=..., domain=..., n_modes=...)      # the specification (spec)
        -> star modes (freq_k, V_k)           # discretization (legendre / orthpol)
        -> chain (w_j, k_j)                   # Lanczos map (lanczos / chain)

.. rubric:: What's here

================================  ==============================================
:class:`Bath`                     the bath specification you pass to ``run``
:func:`thermalize`                T-TEDOPA thermalized density from a ``T=0`` one
:func:`get_bath_nn_paras`         spectral density -> chain ``(w_list, k_list)``
:func:`get_coupling`              the same, via polynomial recurrences
:func:`get_vn_squared`            Gauss-Legendre star discretization
:func:`make_orthpol_discretizer`  measure-adapted ORTHPOL star (peaked/IR baths)
:func:`lanczos`                   star -> chain tridiagonalization
================================  ==============================================

Submodules: :mod:`~fishbonett.bath.spec`, :mod:`~fishbonett.bath.chain`,
:mod:`~fishbonett.bath.legendre`, :mod:`~fishbonett.bath.orthpol`,
:mod:`~fishbonett.bath.lanczos`, :mod:`~fishbonett.bath.recurrence`,
:mod:`~fishbonett.bath.auto` (automatic domain / mode count).
"""
from fishbonett.bath.spec import Bath, thermalize
from fishbonett.bath.chain import get_bath_nn_paras, get_coupling
from fishbonett.bath.legendre import get_vn_squared, get_legendre_recursion
from fishbonett.bath.orthpol import (
    get_vn_squared_orthpol, make_orthpol_discretizer,
)
from fishbonett.bath.lanczos import lanczos
from fishbonett.bath.recurrence import recurrenceCoefficients

__all__ = [
    "Bath", "thermalize",
    "get_bath_nn_paras", "get_coupling",
    "get_vn_squared", "get_legendre_recursion",
    "get_vn_squared_orthpol", "make_orthpol_discretizer",
    "lanczos", "recurrenceCoefficients",
]
