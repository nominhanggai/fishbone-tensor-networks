"""Three-state golden rule: higher-order corrections for a donor and two acceptors.

:mod:`fishbonett.rates.golden_rule` is second order in the electronic coupling,
which is enough for a single donor-acceptor pair.  With a **second acceptor**
the system can hop D -> A1 -> A2 -> A1 -> ..., and those repeated excursions
enter at fourth order and beyond.  This module computes those corrections.

Each correction of order ``k`` is a ``k``-dimensional *time-ordered* integral
``t_max > t_1 > ... > t_k > 0`` of the bath influence functional.  The three
states enter through their reorganization shifts ``s_list = (s_D, s_A1, s_A2)``
-- only the *differences* matter, and each pair contributes a
``cos``/``sin``-weighted term in the exponent.

.. rubric:: Integrators

===========================================  ==================================
:func:`fgr_rate3_correction_order1`          order 1, ``nquad`` (3D)
:func:`fgr_rate3_correction_order2`          order 2, ``nquad`` (4D)
:func:`fgr_rate3_correction_order2_vegas`    order 2, VEGAS Monte Carlo
:func:`fgr_rate3_correction_order_quad`      any order, ``nquad``
:func:`fgr_rate3_correction_order_vegas`     any order, VEGAS Monte Carlo
===========================================  ==================================

**Which to use.** Deterministic ``nquad`` is accurate but its cost grows steeply
with dimension, so it is practical to about order 2-3; the VEGAS variants scale
to higher orders but return a stochastic estimate whose error must be checked by
varying ``nitn``/``neval``.  VEGAS requires the optional ``vegas`` package
(``pip install -e ".[rates]"``), imported lazily so the rest of the module works
without it.

The general-order routines build their nested integration limits as ordinary
Python closures.  They do not evaluate generated source code.
"""
from dataclasses import dataclass

import numpy as np
from scipy import integrate
from fishbonett.bath.legendre import get_vn_squared
import itertools as it


@dataclass(frozen=True)
class MonteCarloEstimate:
    """A VEGAS estimate with its one-standard-deviation uncertainty."""

    mean: float
    sdev: float


def _path_coupling(couplings, order):
    """Electronic prefactor for D->A1, ``order`` A1/A2 hops, then return."""
    c12, c31, c23 = couplings
    return c12 * c23 ** order * (c31 if order % 2 else c12)


def fgr_rate3_correction_order1(c_list, e_list, kbT, _w, s_list, t_max):
    """First correction beyond the golden rule for a donor + two acceptors.

    A 3D time-ordered integral evaluated with ``scipy.integrate.nquad``.

    Parameters
    ----------
    c_list : (c12, c31, c23)
        Electronic couplings between the three states.
    e_list : (e1, e2, e3)
        State energies.
    kbT : float
        Temperature in energy units.
    _w : array
        Star-mode frequencies (see :func:`fishbonett.bath.legendre.get_vn_squared`).
    s_list : (s1, s2, s3)
        Per-state reorganization shifts; only their differences enter.
    t_max : float
        Upper time limit -- take it past the decay of the lineshape (check with
        :func:`fishbonett.rates.golden_rule.fgr_decay_profile`).
    """
    c12, c31, c23 = c_list
    e1, e2, e3 = e_list
    s1, s2, s3 = np.array(s_list)

    s12 = s1 - s2
    s23 = s2 - s3
    s31 = s3 - s1

    w = np.array(_w)
    w_sq = w ** 2

    coth = 1 / np.tanh(w / (2 * kbT))
    const_exponent = -coth * (s12 ** 2 + s23 ** 2 + s31 ** 2) / (2 * w_sq * np.pi)


    prefactor_1 = s12 * s23 / w_sq / np.pi
    prefactor_2 = s12 * s31 / w_sq / np.pi
    prefactor_3 = s23 * s31 / w_sq / np.pi


    exponent = lambda t1, t2, t3: np.sum(
        prefactor_1 * (-coth * np.cos(w * (t1 - t2)) + 1j * np.sin(w * (t1 - t2))) +
        prefactor_2 * (-coth * np.cos(w * (t1 - t3)) + 1j * np.sin(w * (t1 - t3))) +
        prefactor_3 * (-coth * np.cos(w * (t2 - t3)) + 1j * np.sin(w * (t2 - t3)))
        + const_exponent
    )
    integrand = lambda t3, t2: np.real(
        (-1j) ** 3 * (
                np.exp(1j * (e1 - e2) * t_max + 1j * (e2 - e3) * t2 + 1j * (e3 - e1) * t3) *
                np.exp(exponent(t_max, t2, t3))
        )
    )

    def range_t3(t2):
        return [0, t2]

    def range_t2():
        return [0, t_max]

    integral, _ = integrate.nquad(integrand, [range_t3, range_t2], opts={"epsrel": 1e-3})
    return -2 * c12 * c23 * c31 * integral


def fgr_rate3_correction_order2(c_list, e_list, kbT, _w, s_list, t_max):
    """Second correction (one more D->A1->A2 excursion): a 4D ``nquad`` integral.

    Arguments as :func:`fgr_rate3_correction_order1`.  This is about the practical
    limit for deterministic quadrature; for higher orders use
    :func:`fgr_rate3_correction_order2_vegas` or the general-order routines.
    """
    c12, _c31, c23 = c_list
    e1, e2, e3 = e_list
    s1, s2, s3 = np.array(s_list)

    s12 = s1 - s2
    s21 = s2 - s1
    s23 = s2 - s3
    s32 = s3 - s2

    w = np.array(_w)
    w_sq = w ** 2

    coth = 1 / np.tanh(w / (2 * kbT))
    const_exponent = -coth * (s12 ** 2 + s23 ** 2 + s32 ** 2 + s21 ** 2) / (2 * w_sq * np.pi)


    prefactor_12 = s12 * s23 / w_sq / np.pi
    prefactor_13 = s12 * s32 / w_sq / np.pi
    prefactor_14 = s12 * s21 / w_sq / np.pi
    prefactor_23 = s23 * s32 / w_sq / np.pi
    prefactor_24 = s23 * s21 / w_sq / np.pi
    prefactor_34 = s32 * s21 / w_sq / np.pi


    exponent = lambda t1, t2, t3, t4: np.sum(
        prefactor_12 * (-coth * np.cos(w * (t1 - t2)) + 1j * np.sin(w * (t1 - t2))) +
        prefactor_13 * (-coth * np.cos(w * (t1 - t3)) + 1j * np.sin(w * (t1 - t3))) +
        prefactor_14 * (-coth * np.cos(w * (t1 - t4)) + 1j * np.sin(w * (t1 - t4))) +
        prefactor_23 * (-coth * np.cos(w * (t2 - t3)) + 1j * np.sin(w * (t2 - t3))) +
        prefactor_24 * (-coth * np.cos(w * (t2 - t4)) + 1j * np.sin(w * (t2 - t4))) +
        prefactor_34 * (-coth * np.cos(w * (t3 - t4)) + 1j * np.sin(w * (t3 - t4)))
        + const_exponent
    )
    integrand = lambda t4, t3, t2: np.real(
        (-1j) ** 4 * (
                np.exp(1j * (e1 - e2) * t_max + 1j * (e2 - e3) * t2 + 1j * (e3 - e2) * t3 + 1j * (e2 - e1) * t4) *
                np.exp(exponent(t_max, t2, t3, t4))
        )
    )

    def range_t4(t3, t2):
        return [0, t3]

    def range_t3(t2):
        return [0, t2]

    def range_t2():
        return [0, t_max]

    integral, _ = integrate.nquad(integrand, [range_t4, range_t3, range_t2], opts={"epsrel": 1e-4})
    return -2 * c12 ** 2 * c23 ** 2 * integral


def _require_vegas():
    """Import ``vegas``, or explain which extra provides it.

    It is imported inside the functions that need it, not at module scope, so the
    rest of this module works without it.  The bare ``ModuleNotFoundError`` did not
    mention that ``fishbonett[rates]`` exists for exactly this.
    """
    try:
        import vegas
    except ImportError as exc:                     # pragma: no cover - needs the extra absent
        raise ImportError(
            "the VEGAS Monte-Carlo rate integrators need the optional `vegas` "
            "package: pip install 'fishbonett[rates]'.  The deterministic "
            "`..._quad` variants in this module need no extra."
        ) from exc
    return vegas


def fgr_rate3_correction_order2_vegas(c_list, e_list, kbT, _w, s_list, t_max,
                                      nitn=10, neval=1000):
    """:func:`fgr_rate3_correction_order2` by VEGAS Monte Carlo instead of ``nquad``.

    The time-ordered simplex is mapped to the unit cube by the nested
    substitution ``t_{k+1} = y_{k+1} t_k / t_max``, whose Jacobian appears as the
    extra polynomial factor in the integrand; VEGAS then adapts to the
    oscillatory structure over ``nitn`` iterations of ``neval`` samples each.

    Returns :class:`MonteCarloEstimate`, containing both the mean and one-standard-
    deviation uncertainty. Increase ``nitn``/``neval`` and verify stability.
    Requires the optional ``vegas`` package.
    """
    c12, _c31, c23 = c_list
    e1, e2, e3 = e_list
    s1, s2, s3 = np.array(s_list)

    s12 = s1 - s2
    s21 = s2 - s1
    s23 = s2 - s3
    s32 = s3 - s2

    w = np.array(_w)
    w_sq = w ** 2

    coth = 1 / np.tanh(w / (2 * kbT))
    const_exponent = -coth * (s12 ** 2 + s23 ** 2 + s32 ** 2 + s21 ** 2) / (2 * w_sq * np.pi)

    prefactor_12 = s12 * s23 / w_sq / np.pi
    prefactor_13 = s12 * s32 / w_sq / np.pi
    prefactor_14 = s12 * s21 / w_sq / np.pi
    prefactor_23 = s23 * s32 / w_sq / np.pi
    prefactor_24 = s23 * s21 / w_sq / np.pi
    prefactor_34 = s32 * s21 / w_sq / np.pi

    exponent = lambda t1, y2, y3, y4: np.sum(
        prefactor_12 * (-coth * np.cos(w * (t1 - y2)) + 1j * np.sin(w * (t1 - y2))) +
        prefactor_13 * (-coth * np.cos(w * (t1 - y3 * y2 / t1)) + 1j * np.sin(w * (t1 - y3 * y2 / t1))) +
        prefactor_14 * (-coth * np.cos(w * (t1 - y4 * y3 * y2 / t1 ** 2)) + 1j * np.sin(
            w * (t1 - y4 * y3 * y2 / t1 ** 2))) +
        prefactor_23 * (-coth * np.cos(w * (y2 - y3 * y2 / t1)) + 1j * np.sin(w * (y2 - y3 * y2 / t1))) +
        prefactor_24 * (-coth * np.cos(w * (y2 - y4 * y3 * y2 / t1 ** 2)) + 1j * np.sin(
            w * (y2 - y4 * y3 * y2 / t1 ** 2))) +
        prefactor_34 * (-coth * np.cos(w * (y3 * y2 / t1 - y4 * y3 * y2 / t1 ** 2)) + 1j * np.sin(
            w * (y3 * y2 / t1 - y4 * y3 * y2 / t1 ** 2)))
        + const_exponent
    )

    # y = [y2,y3,y4]
    integrand = lambda y: np.real(
        (-1j) ** 4 * (
                np.exp(1j * (e1 - e2) * t_max + 1j * (e2 - e3) * y[0] + 1j * (e3 - e2) * y[1] * y[0] / t_max + 1j * (
                        e2 - e1) * y[2] * y[1] * y[0] / t_max ** 2) *
                np.exp(exponent(t_max, y[0], y[1], y[2]))
        )
        * y[0] / t_max * y[0] * y[1] / t_max ** 2
    )

    vegas = _require_vegas()
    int_interval = [0, t_max]
    integ = vegas.Integrator([int_interval] * 3)

    result = integ(integrand, nitn=nitn, neval=neval)
    scale = -2 * c12 ** 2 * c23 ** 2
    return MonteCarloEstimate(
        mean=float(scale * result.mean),
        sdev=float(abs(scale) * result.sdev),
    )


def fgr_rate3_correction_order_quad(c_list, e_list, kbT, _w, s_list, t1, order):
    """The correction at **arbitrary** ``order``, by nested ``nquad``.

    Generalizes :func:`fgr_rate3_correction_order1` / ``_order2``: the state
    sequence alternates A1 <-> A2 for ``order`` excursions before returning to the
    donor, and the integrand plus its ``order+1`` nested integration limits are
    constructed at call time with closures.

    Cost grows steeply with ``order`` -- past 2 or 3 prefer
    :func:`fgr_rate3_correction_order_vegas`.  ``t1`` is the outermost time limit
    (``t_max`` in the fixed-order routines).
    """
    c = np.asarray(c_list)
    s_values = np.asarray(s_list)
    energy_values = np.asarray(e_list)
    s = {1: s_values[0], 2: s_values[1], 3: s_values[2]}
    E = {1: energy_values[0], 2: energy_values[1], 3: energy_values[2]}
    w = np.array(_w)
    w_sq = w ** 2

    tl = range(1, order + 3)

    sub_list = {1: (1, 2)}
    for i in range(2, order + 2):
        if i % 2 == 0:
            sub_list[i] = (2, 3)
        else:
            sub_list[i] = (3, 2)
    if order % 2 == 0:
        sub_list[order + 2] = (2, 1)
    if order % 2 == 1:
        sub_list[order + 2] = (3, 1)

    delta = {}
    for t in tl:
        k, l = sub_list[t]
        delta[t] = s[k] - s[l]

    coth = 1 / np.tanh(w / (2 * kbT))

    const_exponent = np.sum(-coth * [delta[t] ** 2 for t in tl], axis=0) / (2 * w_sq * np.pi)

    pre = {}
    for m, n in it.combinations(tl, 2):
        pre[(m, n)] = delta[m] * delta[n] / w_sq / np.pi

    def integrand(*reverse_times):
        times = {1: t1}
        for index, value in zip(range(order + 2, 1, -1), reverse_times):
            times[index] = value
        exponent = np.array(const_exponent, complex)
        for (m, n), factor in pre.items():
            delta_t = times[m] - times[n]
            exponent += factor * (
                -coth * np.cos(w * delta_t) + 1j * np.sin(w * delta_t)
            )
        phase = 0.0j
        for index in tl:
            left, right = sub_list[index]
            phase += 1j * times[index] * (E[left] - E[right])
        coefficient = _path_coupling(c, order)
        return coefficient * np.real(
            (-1j) ** (order + 2)
            * np.exp(phase)
            * np.exp(np.sum(exponent))
        )

    ranges = []
    for index in range(order + 2, 1, -1):
        if index == 2:
            ranges.append([0, t1])
        else:
            ranges.append(lambda *outer: [0, outer[0]])

    integral, _ = integrate.nquad(integrand, ranges, opts={"epsrel": 1e-4})

    return -2 * integral


def fgr_rate3_correction_order_vegas(c_list, e_list, kbT, w, s_list, t_max, order,
                                     nitn=10, neval=1000):
    """The correction at **arbitrary** ``order``, by VEGAS Monte Carlo.

    The scalable route: the Monte Carlo cost is set by ``neval``, not by the
    dimension, so this is the one to use past order 2-3.  States are labelled
    ``D``/``A1``/``A2`` and alternate as in
    :func:`fgr_rate3_correction_order_quad`.

    Returns a stochastic estimate -- vary ``nitn``/``neval`` to confirm it has
    converged.  Requires the optional ``vegas`` package.
    """
    c = np.array(c_list)
    e_list = np.array(e_list)
    s_list = np.array(s_list)
    w = np.array(w)
    w_sq = w ** 2

    s = {"D": s_list[0], "A1": s_list[1], "A2": s_list[2]}
    E = {"D": e_list[0], "A1": e_list[1], "A2": e_list[2]}

    sub_list = {0: ("D", "A1")}
    for i in range(1, order + 1):
        if i % 2 == 1:
            sub_list[i] = ("A1", "A2")
        else:
            sub_list[i] = ("A2", "A1")
    if order % 2 == 0:
        sub_list[order + 1] = ("A1", "D")
    if order % 2 == 1:
        sub_list[order + 1] = ("A2", "D")

    delta = {}
    for i in range(order + 2):
        l, r = sub_list[i]
        delta[i] = s[l] - s[r]

    coth = 1 / np.tanh(w / (2 * kbT))
    const_exponent = np.sum(-coth * [delta[i] ** 2 for i in range(order + 2)], axis=0) / (2 * w_sq * np.pi)

    # Generate exponent
    def exponent(*t):
        """
        Args:
            t : a list storing time variables. E.g., for order 3, the list t has three elements

        Returns:
            float
        """
        summand = 0
        for m, n in it.combinations(range(len(t)), 2):
            summand += delta[m] * delta[n] / w_sq / np.pi \
                       * (- coth * np.cos(w * (t[m] - t[n]))
                          + 1j * np.sin(w * (t[m] - t[n]))
                          )

        return np.sum(summand + const_exponent)

    def time_factor(*t):
        f = 1
        for i in range(len(t)):
            k, l = sub_list[i]
            f *= np.exp(1j * t[i] * (E[k] - E[l]))
        return f

    # changing variables

    def y2t(y, beta):
        t = []
        for i, yi in enumerate(y):
            t.append(np.prod(y[:i + 1]) / beta ** i)
        return t

    def t2y_jacobian(y, beta):
        jacobian = 1
        n = len(y)
        for i, yi in enumerate(y[:-1]):
            jacobian *= (yi / beta) ** (n - 1 - i)
        return jacobian

    def integrand(y):
        """

        Args: y (): y_ is the list of y1, y2, ..., y_{n-1} for the n-th order. Note the argument of the functions
        time_factor() and exponent() is t0, t_1, t_2, ..., t_{n-1}.

        Returns: float

        """
        t_ = y2t(y, t_max)  # t1, t2, ..., t_{n-1}
        return np.real(
            (-1j) ** (order + 2)
            * time_factor(t_max, *t_)
            * np.exp(exponent(t_max, *t_))
            * t2y_jacobian(y, t_max)
        )

    vegas = _require_vegas()

    int_interval = [0, t_max]
    integrator = vegas.Integrator([int_interval] * (order + 1))

    result = integrator(integrand, nitn=nitn, neval=neval)
    scale = -2 * _path_coupling(c, order)
    return MonteCarloEstimate(
        mean=float(scale * result.mean),
        sdev=float(abs(scale) * result.sdev),
    )


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
    v = np.sqrt(v_sq)
    print("Discrete Reorganization E", np.sum(v_sq / w / np.pi))

    aa_coupling = 5e-3
    e = np.linspace(0.015, 0.03, 10)
    print(len(e))

    # rate_fgr_perturbative_1 = np.vectorize(lambda ei: fgr_rate_by_order(C_DA, ei, kbT, w, v_sq, aa_coupling, 1)
    #                                        )(e)
    #
    # rate_fgr_perturbative_0 = np.vectorize(lambda ei: fgr_rate_by_order(C_DA, ei, kbT, w, v_sq, aa_coupling, 0)
    #                                        )(e)
    #
    # rate_fgr_perturbative_2 = np.vectorize(lambda ei: fgr_rate_by_order(C_DA, ei, kbT, w, v_sq, aa_coupling, order)
    #                                        )(e)
    #
    # fgr_rate3_correction_2 = rate_fgr_perturbative_0.copy()
    # for n in range(1, order + 1):
    #     fgr_rate3_correction_2 += np.vectorize(
    #         lambda ei: fgr_rate3_correction_by_order([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
    #                                                  1000, n)
    #     )(e)

    #
    from time import time

    start1 = time()
    fgr_rate3_correction_1 = np.vectorize(
        lambda ei: fgr_rate3_correction_order_quad([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
                                                   1000, 0)
    )(e)
    end1 = time()
    print("finished")
    # fgr_rate3_correction_1_mcmc = np.vectorize(
    #         lambda ei: fgr_rate3_correction_by_order_mcmc([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
    #                                                  400, 2, 100000, burn_in=1000)
    #     )(e)

    fgr_rate3_correction_1_vegas = np.vectorize(
        lambda ei: fgr_rate3_correction_order_vegas([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
                                                    1000, 0, nitn=10, neval=1000)
    )(e)

    end2 = time()
    print(f"Quadrature {start1 - end1}; MCMC {end2 - end1}")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(e, fgr_rate3_correction_1, 'd-', label=f'Quadrature {1}')
    ax.plot(e, fgr_rate3_correction_1_vegas, 'd-', label=f'vegas {1}')
    # ax.plot(e, fgr_rate3_correction_1_mcmc, 'd-', label=f'MCMC {1}')

    ax.legend()
    ax.set_xlabel('E (a.u.)')
    ax.set_ylabel('Rate (a.u.)')
    ax.set_xlim(0.015, 0.03)

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.4)

    fig.savefig('golden_rule_figure_3.png')

    print("finished")
