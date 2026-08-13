"""TEDOPA chain mapping: spectral density -> nearest-neighbour chain parameters.

The last step of the bath pipeline.  A spectral density is first *discretized*
into a star of independent modes (:mod:`fishbonett.bath.legendre` or
:mod:`fishbonett.bath.orthpol`), then *Lanczos-mapped* into a chain
(:mod:`fishbonett.bath.lanczos`) in which only neighbouring modes couple -- which
is what gives a matrix-product state something local to exploit.

The split from :mod:`fishbonett.linalg` is by subject: this subpackage turns
physics (a spectral density) into chain parameters, and ``linalg`` manipulates
the tensors those parameters end up in.

.. rubric:: What's here

===============================  ================================================
:func:`get_bath_nn_paras`        ``(w_list, k_list)`` via star -> Lanczos -> chain
:func:`get_coupling`             the same, via orthogonal-polynomial recurrences
===============================  ================================================

Both return ``(w_list, k_list)``: the chain on-site energies ``w_j`` and the
couplings ``k_list = [k0, hop_1, hop_2, ...]``, where ``k0`` is the system-bath
coupling and the rest are mode-mode hoppings.
"""
import numpy as np

from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.lanczos import lanczos
import fishbonett.bath.recurrence as rc

__all__ = ["get_bath_nn_paras", "get_coupling"]


def get_bath_nn_paras(sd, n, domain, discretizer=None):
    """Nearest-neighbour (chain) bath parameters for a spectral density.

    Discretises ``sd`` into ``n`` star modes and Lanczos-maps them to a chain,
    returning ``(w_list, k_list)``: the chain on-site energies and the couplings
    ``[k0, hopping_1, ...]`` where ``k0`` is the system-bath coupling.

    ``discretizer`` is a ``get_vn_squared``-compatible callable ``(sd, n, domain)
    -> (freq, V_squared)``; it defaults to the Gauss-Legendre star
    (:func:`fishbonett.bath.legendre.get_vn_squared`).  Pass a
    measure-adapted ORTHPOL discretizer (see
    :func:`fishbonett.bath.orthpol.make_orthpol_discretizer`) to resolve
    infrared-divergent / sharply peaked spectral densities.
    """
    disc = discretizer if discretizer is not None else get_vn_squared
    star_freq, v_squared = disc(sd, n=n, domain=domain)
    v = np.sqrt(v_squared / np.pi)
    k0 = np.linalg.norm(v)
    tri_mat, _ = lanczos(np.diag(star_freq), v)
    w_list = np.diagonal(tri_mat).copy()          # chain on-site energies
    k_list = np.array([k0] + list(np.diagonal(tri_mat, -1)))
    return w_list, k_list


def get_coupling(sd, n, domain, g=1, ncap=20000, discretizer=None):
    """Chain parameters from orthogonal-polynomial recurrence coefficients.

    The classical TEDOPA route: the recurrence coefficients ``(alpha, beta)`` of
    the polynomials orthogonal under the measure ``sd`` *are* the chain on-site
    energies and hoppings, with no explicit star in between.  Returns
    ``(w_list, k_list)`` in the same layout as :func:`get_bath_nn_paras`.

    ``g`` rescales the frequency axis, ``ncap`` caps the discretization used
    inside the recurrence, and ``discretizer`` selects the quadrature.
    """
    alphaL, betaL = rc.recurrenceCoefficients(
        n - 1, lb=domain[0], rb=domain[1], j=sd, g=g, ncap=ncap,
        discretizer=discretizer,
    )
    w_list = g * np.array(alphaL)
    k_list = g * np.sqrt(np.array(betaL))
    k_list[0] = k_list[0] / g
    return w_list, k_list
