"""TEDOPA chain mapping: spectral density -> nearest-neighbour chain parameters.

TEDOPA maps a continuum bath **directly** onto a chain in which only neighbouring
modes couple -- which is what gives a matrix-product state something local to
exploit.  There is no discretization step in between: the spectral density is used
as the *weight function* of a family of orthogonal polynomials, and that family's
three-term recurrence coefficients ``(alpha_j, beta_j)`` *are* the chain's on-site
energies and hoppings.  That is :func:`get_coupling`.

The same chain can also be reached by way of a star, and
:func:`get_bath_nn_paras` takes that route: build an ``n``-point quadrature of the
density (:mod:`fishbonett.bath.legendre` or :mod:`fishbonett.bath.tedopa`), then
tridiagonalize it with Lanczos (:mod:`fishbonett.bath.lanczos`).  The two agree
because the operations are inverse: diagonalizing ("starizing") an ``n``-site chain
recovers the ``n``-point quadrature of the same measure.  Which route to use is a
numerical choice, not a physical one -- going via a star lets you pick the measure
the modes are placed against, which is what ``discretization=`` selects.

The split from :mod:`fishbonett.linalg` is by subject: this subpackage turns
physics (a spectral density) into chain parameters, and ``linalg`` manipulates
the tensors those parameters end up in.

.. rubric:: What's here

===============================  ================================================
:func:`get_coupling`             ``(w_list, k_list)`` straight from the recurrence
:func:`get_bath_nn_paras`        the same, via a star + Lanczos tridiagonalization
===============================  ================================================

Both return ``(w_list, k_list)``: the chain on-site energies ``w_j`` and the
couplings ``k_list = [k0, hop_1, hop_2, ...]``, where ``k0`` is the system-bath
coupling and the rest are mode-mode hoppings.
"""
import numpy as np

from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.lanczos import lanczos
import fishbonett.bath.recurrence as rc

__all__ = ["get_bath_nn_paras", "get_coupling", "star_transform"]


def get_bath_nn_paras(sd, n, domain, discretizer=None):
    """Nearest-neighbour (chain) bath parameters, via a star.

    Builds an ``n``-point quadrature of ``sd`` and Lanczos-tridiagonalizes it into a
    chain, returning ``(w_list, k_list)``: the chain on-site energies and the
    couplings ``[k0, hopping_1, ...]`` where ``k0`` is the system-bath coupling.
    Equivalent to :func:`get_coupling` (see the module docstring); the reason to
    come this way is that it lets ``discretizer`` choose the measure.

    ``discretizer`` is a ``get_vn_squared``-compatible callable ``(sd, n, domain)
    -> (freq, V_squared)``; it defaults to the Gauss-Legendre star
    (:func:`fishbonett.bath.legendre.get_vn_squared`).  Pass a
    measure-adapted TEDOPA discretizer (see
    :func:`fishbonett.bath.tedopa.make_tedopa_discretizer`) to resolve
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


def star_transform(sd, n, domain, discretizer=None):
    """``(freq, Vn, coefT)``: star frequencies, couplings ``sqrt(V^2/pi)`` and the
    sign-fixed star->chain (Lanczos) transform ``P.T``.

    The sign fixing (by each eigenvector's first component) makes the transform
    deterministic, which matters because the interaction picture rebuilds its
    couplings ``d_j(t) = coefT @ (Vn * exp(-i freq t))`` from it every step.

    It lives here rather than in a representation because it is bath machinery -- the same
    star/Lanczos pair as :func:`get_bath_nn_paras`, returning the transform matrix
    instead of discarding it.  Two representations need it (the chain-geometry star MPO and
    the mode-tree engine), and they had a copy each.
    """
    disc = discretizer if discretizer is not None else get_vn_squared
    freq, v_sq = disc(sd, n, list(domain))
    Vn = np.sqrt(v_sq / np.pi)
    _, P = lanczos(np.diag(freq), Vn)
    sign = np.sign(P[0, :])
    P = P @ np.diag(sign)
    return np.asarray(freq), np.asarray(Vn), np.ascontiguousarray(P.T)


def get_coupling(sd, n, domain, g=1, ncap=20000, discretizer=None):
    """Chain parameters from orthogonal-polynomial recurrence coefficients.

    The classical TEDOPA route: the recurrence coefficients ``(alpha, beta)`` of
    the polynomials orthogonal under the measure ``sd`` *are* the chain on-site
    energies and hoppings, with no explicit star in between.  Returns
    ``(w_list, k_list)`` in the same layout as :func:`get_bath_nn_paras`.

    ``g`` rescales the frequency axis and ``discretizer`` selects the quadrature.

    ``ncap`` is **accepted and ignored**, kept because callers pass it: the
    discretization accuracy is set by ``n``.  It used to cap a fixed-length
    recursion in the py-orthpol implementation that this replaced -- see
    :func:`fishbonett.bath.recurrence.recurrenceCoefficients`, which says the same.
    Tuning it does nothing.
    """
    alphaL, betaL = rc.recurrenceCoefficients(
        n - 1, lb=domain[0], rb=domain[1], j=sd, g=g, ncap=ncap,
        discretizer=discretizer,
    )
    w_list = g * np.array(alphaL)
    k_list = g * np.sqrt(np.array(betaL))
    k_list[0] = k_list[0] / g
    return w_list, k_list
