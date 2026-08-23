#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gauss-Legendre star discretization -- the default way to make a bath finite.

Replaces the continuous spectral density by ``n`` discrete modes.  The nodes are
the Gauss-Legendre quadrature points of the frequency window, obtained as the
eigenvalues of the Jacobi matrix of the Legendre recurrence, and each mode's
coupling is ``V_n^2 = J(w_n) * weight_n``.

The measure here is **uniform** over the window, independent of ``J``.  That is
robust and cheap, but it means a sharply peaked or infrared-divergent ``J`` can
fall between nodes; :mod:`fishbonett.bath.tedopa` adapts the nodes to the
measure instead and should be preferred in those cases.

.. rubric:: API

==============================  =================================================
:func:`get_legendre_recursion`  Legendre recurrence coefficients on a window
:func:`get_vn_squared`          the star: ``(freq, V_squared)`` -- the main entry
:func:`get_approx_func`         the discretized ``J`` as a smooth function, to plot
:func:`get_recursion`           numerical recurrence route (deprecated)
==============================  =================================================
"""

import numpy as np


def get_recursion(n, j, domain, g=1, ncap=20000):  # j=weight function
    """Return recurrence coefficients numerically.

    Deprecated because :func:`get_legendre_recursion` evaluates the Legendre
    coefficients analytically at lower cost.
    """
    import fishbonett.bath.recurrence as rc
    alphaL, sqrt_betaL = rc.recurrenceCoefficients(
        n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
    )
    j = lambda x: j(x) * np.pi
    alphaL = g * np.array(alphaL)
    sqrt_betaL = g * np.sqrt(np.array(sqrt_betaL))
    sqrt_betaL[0] = sqrt_betaL[0] / g
    return alphaL, sqrt_betaL[1:]  # k=sqrt(beta), w=alpha, sqrt_beta[0] is dropped


def get_legendre_recursion(n, domain):
    """Legendre recurrence coefficients ``(alpha, beta)`` on ``domain``.

    Analytic, so this costs nothing: shifting the standard Legendre recurrence to
    ``[l, r]`` gives ``alpha_k = (l+r)/2`` (constant) and
    ``beta_k = (r-l)/2 * k / sqrt(4k^2 - 1)``.  ``alpha`` has length ``n`` and
    ``beta`` length ``n-1``; together they are the Jacobi matrix whose eigenvalues
    are the quadrature nodes.
    """
    if (not isinstance(n, (int, np.integer))
            or isinstance(n, (bool, np.bool_)) or n < 1):
        raise ValueError("n must be a positive integer")
    l = float(domain[0])
    r = float(domain[1])
    if not np.isfinite(l) or not np.isfinite(r) or l >= r:
        raise ValueError("domain must contain two finite values with left < right")
    a = (l + r) / 2
    a = np.repeat(a, n)
    _temp = (r - l) / 2
    k = np.arange(1, n, dtype=float)
    b = _temp * k / np.sqrt(4 * k ** 2 - 1)
    return a, b


def get_vn_squared(j, n: int, domain):
    """Discretize the spectral density ``j`` into ``n`` star modes on ``domain``.

    Diagonalizing the Jacobi matrix of the Legendre recurrence gives the
    quadrature nodes ``freq`` and, from the first component of each eigenvector,
    the weights; the squared coupling of each mode is ``J(w_n) * weight_n``.

    Returns ``(freq, V_squared)``.  This is the reference signature every
    discretizer follows: pass a compatible callable as the ``discretizer``
    argument of :func:`fishbonett.bath.chain.get_bath_nn_paras` to swap in the
    measure-adapted TEDOPA star instead.
    """
    alpha, beta = get_legendre_recursion(n, domain)
    M = np.diag(alpha) + np.diag(beta, -1) + np.diag(beta, 1)
    freq, eig_vec = np.linalg.eigh(M)
    W = (eig_vec[0, :]) ** 2 * (domain[1] - domain[0])
    V_squared = [j(w) * W[n] for n, w in enumerate(freq)]
    return freq, np.array(V_squared)


def get_approx_func(J, n, domain, epsilon):
    """The discretized ``J`` back as a smooth function, for checking the star.

    Replaces each discrete mode by a Lorentzian of width ``epsilon`` and sums
    them, so the result can be plotted against the original ``J``.  A visible
    mismatch means ``n`` is too small or the nodes are in the wrong places (in
    which case try the TEDOPA discretizer).  Diagnostic only -- nothing in the
    propagation path uses it.
    """
    delta = lambda x: 1 / np.pi * epsilon / (epsilon ** 2 + x ** 2)
    w, V_squared = get_vn_squared(J, n, domain)
    j_approx = lambda x: np.sum([vi * delta(x - wi) for wi, vi in zip(w, V_squared)])
    return np.vectorize(j_approx)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from fishbonett.spectral_densities import lorentzian
    drude = lambda x, gam, lam: 2 * lam * gam * x / (x ** 2 + gam ** 2)
    lorentzian1 = lambda w: lorentzian(10, w, 10, 1000) + lorentzian(10, w, 10, 2000)\
    + lorentzian(10, w, 10, 3000) + lorentzian(10, w, 10, 4000)
    J = lorentzian1
    J_approx = get_approx_func(J, 1000, [-1000, 5000], 2)
    print("Get approx func:", J_approx(10))

    x = np.linspace(0, 5000, 1000)
    disc = []
    for xi in x:
        disc += [J_approx(xi)]

    plt.plot(x, J(x), 'r-', label='original')
    plt.plot(x, disc, 'k-', label='approx')
    plt.legend()
    plt.savefig("legendre_discretization.png")
