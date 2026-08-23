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
import warnings
from fishbonett.bath.legendre import get_vn_squared


def _validated_star(kbT, frequencies, weights):
    if not np.isfinite(kbT) or kbT <= 0:
        raise ValueError("kbT must be finite and positive")
    w = np.asarray(frequencies, float)
    v_sq = np.asarray(weights, float)
    if w.ndim != 1 or v_sq.shape != w.shape or w.size == 0:
        raise ValueError("frequencies and squared couplings must be non-empty 1D arrays")
    if (np.any(~np.isfinite(w)) or np.any(~np.isfinite(v_sq))
            or np.any(w == 0) or np.any(v_sq < 0)):
        raise ValueError(
            "frequencies must be finite and non-zero and squared couplings "
            "finite and non-negative"
        )
    return w, v_sq


def _finite_window(t_max):
    if t_max is None:
        raise ValueError(
            "a finite discrete star has a quasiperiodic correlation and cannot "
            "be integrated to infinity; provide a convergence-tested t_max"
        )
    if not np.isfinite(t_max) or t_max <= 0:
        raise ValueError("t_max must be finite and positive")
    return float(t_max)


def fgr_rate(c, e, kbT, _w, _v_sq, *, t_max=None):
    """Windowed golden-rule integral ``2 c^2 Re int_0^t_max e^{g(t)+iet} dt``.

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

    A finite star correlation is quasiperiodic and does not decay at infinite
    time, so ``t_max`` is required and must be checked for a stable rate plateau.
    """
    t_max = _finite_window(t_max)
    w, v_sq = _validated_star(kbT, _w, _v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))
    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) + 1j * np.sin(t * w)))
    integrand = lambda t: np.real(np.exp(exponent(t)) * np.exp(1j * e * t))
    integral, _ = integrate.quad(integrand, 0, t_max, limit=5000)
    return 2 * (c ** 2) * integral


def fgr_decay_profile(e, kbT, _w, _v_sq, t):
    """The lineshape magnitude ``|e^{g(t)}|`` on ``[0, t]`` at 500 points.

    A diagnostic for :func:`fgr_rate`: the rate integral only converges if this
    profile has decayed within the window.  Returns ``(t_grid, values)`` and warns
    when ``t < 5/e``, where the oscillatory factor ``e^{i e t}`` is not yet
    resolved.
    """
    if not np.isfinite(t) or t <= 0:
        raise ValueError("t must be finite and positive")
    if e != 0 and t < 5 / abs(e):
        warnings.warn(
            f"t may be too short to resolve the energy difference; try t > {5 / abs(e):g}",
            RuntimeWarning,
            stacklevel=2,
        )
    t = np.linspace(0, t, 500)
    w, v_sq = _validated_star(kbT, _w, _v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))
    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) + 1j * np.sin(t * w)))
    integrand = lambda t: np.abs(np.exp(exponent(t)))
    integrand_discrete = np.vectorize(integrand)(t)
    return t, integrand_discrete


def fgr_rate_by_order(c, e, kbT, _w, _v_sq, perturbation, order: int, *,
                      t_max=None):
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

    if (not isinstance(order, (int, np.integer))
            or isinstance(order, (bool, np.bool_)) or order < 0):
        raise ValueError("order must be a non-negative integer")
    t_max = _finite_window(t_max)
    p = perturbation
    w, v_sq = _validated_star(kbT, _w, _v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))

    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) + 1j * np.sin(t * w)))
    integrand = lambda t: np.real(np.exp(exponent(t)) * np.exp(1j * (e) * t) * taylor_exp(- 1j * p * t, order))

    integral, _ = integrate.quad(
        integrand, 0, t_max, limit=5000, epsrel=1e-10
    )
    return 2 * (c ** 2) * integral


def marcus_rate(c: float, e: float, kbT: float, reorg_e: float):
    """Classical Marcus rate for reorganization energy ``reorg_e``.

    The high-temperature / classical-bath limit of :func:`fgr_rate`, depending on
    the bath only through ``reorg_e = (1/pi) int J(w)/w dw``.  Useful as a sanity
    check: the two agree when ``kbT`` is large compared with the bath frequencies,
    and diverge when nuclear tunneling matters.
    """
    if (not np.isfinite(kbT) or kbT <= 0 or not np.isfinite(reorg_e)
            or reorg_e <= 0):
        raise ValueError("kbT and reorg_e must be finite and positive")
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
    rate_fgr = np.vectorize(lambda ei: fgr_rate(
        C_DA, ei - coup, kbT, w, v_sq, t_max=5000.0)
                            )(e)

    order = 5
    rate_fgr_perturbative = np.vectorize(lambda ei: fgr_rate_by_order(
        C_DA, ei, kbT, w, v_sq, coup, order, t_max=5000.0)
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
