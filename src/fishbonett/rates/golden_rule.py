"""Fermi golden-rule electron-transfer rates from a discretized spectral density.

A perturbative alternative to propagating the dynamics: when the donor-acceptor
coupling ``c`` is small, the transfer rate follows from the *bath correlation
function* alone, with no tensor network involved.  The rate is the Fourier
transform of the lineshape function ``exp(g(t))``, with

    ``g(t) = -sum_n (V_n^2 / pi w_n^2) [coth(w_n/2kT)(1 - cos w_n t) + i sin w_n t]``

evaluated on the star modes ``(w, V^2)`` that
:func:`fishbonett.bath.legendre.get_vn_squared` returns.

.. rubric:: API

===============================  ================================================
:func:`fgr_rate`                 the golden-rule rate (2nd order in ``c``)
:func:`fgr_rate_by_order`        rate with the donor-acceptor coupling expanded
                                 to a given Taylor order
:func:`fgr_decay_profile`        ``|exp(g(t))|`` itself, to check convergence
:func:`marcus_rate`              the classical Marcus limit, for comparison
===============================  ================================================

Units are whatever ``J``, ``e`` and ``kbT`` share (atomic units in the example
below).  For the multi-state / higher-order corrections see
:mod:`fishbonett.rates.golden_rule_multi`.
"""
import numpy as np
from scipy import integrate
from fishbonett.bath.legendre import get_vn_squared


def fgr_rate(c, e, kbT, _w, _v_sq):
    """Golden-rule transfer rate ``2 c^2 Re int_0^inf dt e^{g(t)} e^{i e t}``.

    Parameters
    ----------
    c : float
        Donor-acceptor electronic coupling.  The rate is second order in it.
    e : float
        Driving force (donor-acceptor energy gap).
    kbT : float
        Temperature in energy units.
    _w, _v_sq : arrays
        Star-mode frequencies and squared couplings, e.g. from
        :func:`fishbonett.bath.legendre.get_vn_squared`.

    The time integral is taken to infinity by ``scipy.integrate.quad``; it
    converges because ``Re g(t)`` decays.  A near-zero mode frequency makes
    ``1/w^2`` blow up, so keep the discretization domain away from ``w = 0``.
    """
    w = np.array(_w)
    v_sq = np.array(_v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))
    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) + 1j * np.sin(t * w)))
    integrand = lambda t: np.real(np.exp(exponent(t)) * np.exp(1j * e * t))
    integral, _ = integrate.quad(integrand, 0, np.inf, limit=5000)
    return 2 * (c ** 2) * integral


def fgr_decay_profile(e, kbT, _w, _v_sq, t):
    """The lineshape decay ``Re e^{g(t)}`` on ``[0, t]``, sampled at 500 points.

    A diagnostic for :func:`fgr_rate`: the rate integral only converges if this
    profile has decayed within the window.  Returns ``(t_grid, values)`` and warns
    when ``t < 5/e``, where the oscillatory factor ``e^{i e t}`` is not yet
    resolved.
    """
    if t < 5 / e:
        print("Warning: t is too small for this energy difference" + f"Recommend t > {5 / e}")
    t = np.linspace(0, t, 500)
    w = np.array(_w)
    v_sq = np.array(_v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))
    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) + 1j * np.sin(t * w)))
    integrand = lambda t: np.real(np.exp(exponent(t)))
    integrand_discrete = np.vectorize(integrand)(t)
    return t, integrand_discrete


def fgr_rate_by_order(c, e, kbT, _w, _v_sq, perturbation, order: int):
    """Golden-rule rate with an extra ``perturbation`` expanded to ``order``.

    Same integral as :func:`fgr_rate` with the additional factor
    ``exp(-i * perturbation * t)`` replaced by its Taylor series truncated at
    ``order``.  Comparing successive orders shows whether treating that term
    perturbatively is justified; if the series has not settled, propagate the
    dynamics instead.
    """
    import math
    def taylor_exp(x, n):
        exp_approx = 1
        for i in range(1, n + 1):
            num = x ** (i)
            denom = math.factorial(i)
            exp_approx += num / denom
        return exp_approx

    p = perturbation
    w = np.array(_w)
    v_sq = np.array(_v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))

    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) + 1j * np.sin(t * w)))
    integrand = lambda t: np.real(np.exp(exponent(t)) * np.exp(1j * (e) * t) * taylor_exp(- 1j * p * t, order))

    integral, _ = integrate.quad(integrand, 0, np.inf, limit=5000, epsrel=5e-25)
    return 2 * (c ** 2) * integral


def marcus_rate(c: float, e: float, kbT: float, reorg_e: float):
    """Classical Marcus rate for reorganization energy ``reorg_e``.

    The high-temperature / classical-bath limit of :func:`fgr_rate`, depending on
    the bath only through ``reorg_e = (1/pi) int J(w)/w dw``.  Useful as a sanity
    check: the two agree when ``kbT`` is large compared with the bath frequencies,
    and diverge when nuclear tunneling matters.
    """
    return 2 * np.pi * c ** 2 / np.sqrt(4 * np.pi * kbT * reorg_e) * np.exp(
        -(reorg_e - e) ** 2 / (4 * kbT * reorg_e))


if __name__ == "__main__":
    """
    Example calculation. Reproduce Figure 2 in dx.doi.org/10.1021/jp400462f | J. Phys. Chem. A 2013, 117, 6196−6204
    """

    import matplotlib.pyplot as plt

    # Lorentzian spectral density parameters. Atomic units.
    reorg_e = 2.39e-2
    Omega = 3.5e-4
    kbT = 9.5e-4
    eta = 1.2e-3
    domain = [0, 5e-3]
    C_DA = 5e-5
    j = lambda w: 0.5 * (4 * reorg_e) * Omega ** 2 * eta * w / ((Omega ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2)

    w, v_sq = get_vn_squared(j, 100, domain)
    print("Discrete Reorganization E", np.sum(v_sq / w / np.pi))

    x = np.linspace(*domain, 1000)
    plt.plot(x, j(x), label='j')
    plt.savefig('spectral_density.png')
    plt.clf()

    coup = 5e-3
    e = np.linspace(0.015, 0.03, 20)
    rate_fgr = np.vectorize(lambda ei: fgr_rate(C_DA, ei - coup, kbT, w, v_sq)
                            )(e)

    order = 5
    rate_fgr_perturbative = np.vectorize(lambda ei: fgr_rate_by_order(C_DA, ei, kbT, w, v_sq, coup, order)
                                         )(e)
    rate_marcus = np.vectorize(lambda ei: marcus_rate(C_DA, ei - coup, kbT, reorg_e))(e)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(e, rate_fgr, 'o-', label='FGR rate')
    ax.plot(e, rate_marcus, 'x', label='Marcus rate')
    ax.plot(e, rate_fgr_perturbative, 'd-', label=f'FGR rate perturbation order {order}')
    ax.legend()
    ax.set_xlabel('E (a.u.)')
    ax.set_ylabel('Rate (a.u.)')
    ax.set_xlim(0.015, 0.03)

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.4)

    fig.savefig('golden_rule_figure_2.png')
