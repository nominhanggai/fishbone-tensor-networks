"""Everything about the bath: what it is, how it is mapped, how it is discretized.

A continuous spectral density becomes a finite set of modes, in either of two
equivalent representations::

    Bath(J=..., domain=..., n_modes=...)      # the specification (spec)
        -> chain (w_j, k_j)                   # TEDOPA: J as the OP weight function
        -> star modes (freq_k, V_k)           # "starize": diagonalize the chain

The chain is the primary object -- TEDOPA maps the continuum straight onto it, with
no discretization step in between (:mod:`~fishbonett.bath.chain`).  A *star* of
independent modes is the same bath in the eigenbasis of that chain, which is what
the interaction-picture engines need.  Equivalently, a star can be built directly
as an ``n``-point quadrature of ``J``: against the uniform measure
(:func:`get_vn_squared`, ``discretization="legendre"``) or against ``J`` itself,
matching TEDOPA exactly (:func:`make_tedopa_discretizer`,
``discretization="tedopa"``).

.. rubric:: What's here

================================  ==============================================
:class:`Bath`                     the bath specification you pass to ``run``
:func:`thermalize`                T-TEDOPA thermalized density from a ``T=0`` one
:func:`get_bath_nn_paras`         spectral density -> chain ``(w_list, k_list)``
:func:`get_coupling`              the same, via polynomial recurrences
:func:`get_vn_squared`            Gauss-Legendre (uniform-measure) star
:func:`make_tedopa_discretizer`   measure-adapted TEDOPA star (peaked/IR baths)
:func:`lanczos`                   star -> chain tridiagonalization
================================  ==============================================

Submodules: :mod:`~fishbonett.bath.spec`, :mod:`~fishbonett.bath.chain`,
:mod:`~fishbonett.bath.legendre`, :mod:`~fishbonett.bath.tedopa`,
:mod:`~fishbonett.bath.lanczos`, :mod:`~fishbonett.bath.recurrence`,
:mod:`~fishbonett.bath.auto` (automatic domain / mode count).
"""
from fishbonett.bath.spec import Bath, thermalize
from fishbonett.bath.chain import get_bath_nn_paras, get_coupling
from fishbonett.bath.legendre import get_vn_squared, get_legendre_recursion
from fishbonett.bath.tedopa import (
    get_vn_squared_tedopa, make_tedopa_discretizer,
)
from fishbonett.bath.lanczos import lanczos
from fishbonett.bath.recurrence import recurrenceCoefficients

__all__ = [
    "Bath", "thermalize",
    "get_bath_nn_paras", "get_coupling",
    "get_vn_squared", "get_legendre_recursion",
    "get_vn_squared_tedopa", "make_tedopa_discretizer",
    "lanczos", "recurrenceCoefficients",
]
