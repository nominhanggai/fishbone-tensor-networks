"""Everything about the bath: what it is, how it is mapped, how it is discretized.

A continuous spectral density becomes finite star data and, when requested, its
star-to-chain transform::

    Bath(J=..., domain=..., n_modes=...)      # the specification (spec)
        -> StarBath(freq_k, g_ck, transform)  # finite independent modes
        -> ChainBath(w_j, t_j, c0)            # nearest-neighbour transform

The package can obtain chain coefficients directly from orthogonal-polynomial
recurrences, or obtain a finite star and tridiagonalize it. Conversely,
diagonalizing a finite chain recovers equivalent star data. These are numerical
routes to the same finite bath. A star can be built as an ``n``-point quadrature
of ``J``: against the uniform measure
(:func:`get_vn_squared`, ``discretization="legendre"``) or against ``J`` itself,
matching TEDOPA exactly (:func:`make_tedopa_discretizer`,
``discretization="tedopa"``). Interaction representations always consume the
finite star first and only then optionally apply its star-to-chain transform.

.. rubric:: What's here

=================================  =============================================
:class:`Bath`                      environment physics + resolution settings
:class:`CoupledBath`               explicit model binding to system operator(s)
:class:`ChainBath`                 compiled nearest-neighbour coefficients
:class:`StarBath`                  compiled shared-grid oscillator coefficients
:class:`PolaronBath`               compiled ``J/w^2`` chain + reorganization shift
:func:`thermalize`                 T-TEDOPA thermalized density from a ``T=0`` one
:func:`get_bath_nn_paras`          spectral density -> chain ``(w_list, k_list)``
:func:`get_coupling`               the same, via polynomial recurrences
:func:`get_vn_squared`             Gauss-Legendre (uniform-measure) star
:func:`make_tedopa_discretizer`    measure-adapted TEDOPA star (peaked/IR baths)
:func:`compile_chain`              resolved specification -> ``ChainBath``
:func:`compile_star`               resolved specification -> ``StarBath``
:func:`compile_polaron`            resolved specification -> ``PolaronBath``
:func:`lanczos`                    star -> chain tridiagonalization
=================================  =============================================

Submodules: :mod:`~fishbonett.bath.spec`, :mod:`~fishbonett.bath.chain`,
:mod:`~fishbonett.bath.legendre`, :mod:`~fishbonett.bath.tedopa`,
:mod:`~fishbonett.bath.lanczos`, :mod:`~fishbonett.bath.recurrence`,
:mod:`~fishbonett.bath.auto` (automatic domain / mode count).
"""
from fishbonett.bath.spec import Bath, thermalize
from fishbonett.bath.coupled import CoupledBath, bind_bath
from fishbonett.bath.compiled import (
    StarBath, ChainBath, PolaronBath, compile_star, compile_chain,
    compile_polaron,
)
from fishbonett.bath.conventions import (
    integrated_free_phase, reorganization_energy, star_coupling_squared,
)
from fishbonett.bath.chain import get_bath_nn_paras, get_coupling
from fishbonett.bath.legendre import get_vn_squared, get_legendre_recursion
from fishbonett.bath.tedopa import (
    get_vn_squared_tedopa, make_tedopa_discretizer,
)
from fishbonett.bath.lanczos import lanczos
from fishbonett.bath.recurrence import recurrenceCoefficients

__all__ = [
    "Bath", "CoupledBath", "bind_bath", "thermalize",
    "StarBath", "ChainBath", "PolaronBath", "compile_star", "compile_chain",
    "compile_polaron", "integrated_free_phase", "reorganization_energy",
    "star_coupling_squared",
    "get_bath_nn_paras", "get_coupling",
    "get_vn_squared", "get_legendre_recursion",
    "get_vn_squared_tedopa", "make_tedopa_discretizer",
    "lanczos", "recurrenceCoefficients",
]
