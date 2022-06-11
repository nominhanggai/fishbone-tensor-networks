import numpy as np
from scipy import integrate
from legendre_discretization import get_vn_squared, get_approx_func
from golden_rule_rate import *
import itertools as it
from mcmc_integrator import mcmc_time_ordered

"""
Calculate fermi's golden rule rate for 2-state (2 acceptors) electron transfer reactions, given spectral densities.
"""


def fgr_rate3_correction_order1(c_list, e_list, kbT, _w, s_list, t_max):
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

    print("Correct", const_exponent)

    prefactor_1 = s12 * s23 / w_sq / np.pi
    prefactor_2 = s12 * s31 / w_sq / np.pi
    prefactor_3 = s23 * s31 / w_sq / np.pi

    print("Correct", prefactor_1, prefactor_2, prefactor_3)

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
    return -2 * c12 ** 2 * c23 * integral


def fgr_rate3_correction_order2(c_list, e_list, kbT, _w, s_list, t_max):
    c12, c31, c23 = c_list
    e1, e2, e3 = e_list
    s1, s2, s3 = np.array(s_list)

    s12 = s1 - s2
    s21 = s2 - s1
    s23 = s2 - s3
    s32 = s3 - s2
    s31 = s3 - s1

    w = np.array(_w)
    w_sq = w ** 2

    coth = 1 / np.tanh(w / (2 * kbT))
    const_exponent = -coth * (s12 ** 2 + s23 ** 2 + s32 ** 2 + s21 ** 2) / (2 * w_sq * np.pi)

    print("Correct", const_exponent)

    prefactor_12 = s12 * s23 / w_sq / np.pi
    prefactor_13 = s12 * s32 / w_sq / np.pi
    prefactor_14 = s12 * s21 / w_sq / np.pi
    prefactor_23 = s23 * s32 / w_sq / np.pi
    prefactor_24 = s23 * s21 / w_sq / np.pi
    prefactor_34 = s32 * s21 / w_sq / np.pi

    print("Correct", prefactor_12, prefactor_13, prefactor_14, prefactor_23, prefactor_24, prefactor_34)

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


def fgr_rate3_correction_order2_vegas(c_list, e_list, kbT, _w, s_list, t_max, nitn=10, neval=1000):
    c12, c31, c23 = c_list
    e1, e2, e3 = e_list
    s1, s2, s3 = np.array(s_list)

    s12 = s1 - s2
    s21 = s2 - s1
    s23 = s2 - s3
    s32 = s3 - s2
    s31 = s3 - s1

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
        prefactor_34 * (-coth * np.cos(w * (y3 - y4 * y3 * y2 / t1 ** 2)) + 1j * np.sin(
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

    import vegas
    int_interval = [0, t_max]
    integ = vegas.Integrator([int_interval] * 3)

    result = integ(integrand, nitn=nitn, neval=neval).mean
    return -2 * c12 ** 2 * c23 ** 2 * result


def fgr_rate3_correction_by_order(c_list, e_list, kbT, _w, s_list, t1, order):
    c = c_list
    s = {1: s_list[0], 2: s_list[1], 3: s_list[2]}
    E = {1: e_list[0], 2: e_list[1], 3: e_list[2]}
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

    # Generate exponent
    exponent = "lambda "
    for t in tl:
        exponent += f"t{t},"

    pre = {}
    exponent = exponent[:-1] + ": np.sum("
    for m, n in it.combinations(tl, 2):
        pre[(m, n)] = delta[m] * delta[n] / w_sq / np.pi
        exponent += f"pre[({m},{n})] * (-coth * np.cos(w * (t{m} - t{n})) + 1j * np.sin(w * (t{m} - t{n}))) +"

    exponent = exponent[:-1] + " + const_exponent)"

    time_factor = "np.exp("
    for t in tl:
        k, l = sub_list[t]
        time_factor += f"1j * t{t}*(E[{k}]-E[{l}]) +"

    time_factor = time_factor[:-1] + ")"

    args = ",".join([f"t{t}" for t in tl])
    integrand = "lambda "
    for t in tl[::-1][:-1]:
        integrand += f"t{t},"
    integrand = integrand[:-1] + f": c[0]* c[2]**{order} * c[1] *np.real(\n"
    integrand += f"(-1j) ** {order + 2} * {time_factor} * np.exp(({exponent})({args}))\n"
    integrand += ")"

    # generate integration range
    int_range_str = ""
    int_range_dict = {}
    for t in tl[::-1][:-1]:
        args = ",".join([f"t{i}" for i in range(t - 1, 1, -1)])
        range_func = f"lambda {args}: [0, t{t - 1}]"
        print(f"lambda {args}: [0, t{t - 1}]")
        int_range_dict[t] = eval(f"lambda {args}: [0, t{t - 1}]", {"t1": t1})
        int_range_str += f"int_range_dict[{t}],"

    int_range_str = int_range_str[:-1]

    integrator = f"integrate.nquad({integrand}, [{int_range_str}], opts={{'epsrel': 1e-4}})"

    integral, _ = eval(integrator, {"E": E, "c": c,
                                    "t1": t1, "pre": pre, "coth": coth, "w": w,
                                    "w_sq": w_sq, "const_exponent": const_exponent,
                                    "integrand": integrand, "int_range_dict": int_range_dict,
                                    "integrate": integrate, "np": np, "kbT": kbT})

    return -2 * integral


def fgr_rate3_correction_by_order_mcmc(c_list, e_list, kbT, _w, s_list, t1, order, N, burn_in=1000):
    c = c_list
    s = {1: s_list[0], 2: s_list[1], 3: s_list[2]}
    E = {1: e_list[0], 2: e_list[1], 3: e_list[2]}
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

    # Generate exponent
    exponent = "lambda "
    for t in tl:
        exponent += f"t{t},"

    pre = {}
    exponent = exponent[:-1] + ": np.sum("
    for m, n in it.combinations(tl, 2):
        pre[(m, n)] = delta[m] * delta[n] / w_sq / np.pi
        exponent += f"pre[({m},{n})] * (-coth * np.cos(w * (t{m} - t{n})) + 1j * np.sin(w * (t{m} - t{n}))) +"

    exponent = exponent[:-1] + " + const_exponent)"

    time_factor = "np.exp("
    for t in tl:
        k, l = sub_list[t]
        time_factor += f"1j * t{t}*(E[{k}]-E[{l}]) +"

    time_factor = time_factor[:-1] + ")"

    args = ",".join([f"t{t}" for t in tl])
    integrand = "lambda "
    for t in tl[1:]:
        integrand += f"t{t},"
    integrand = integrand[:-1] + f": c[0]* c[2]**{order} * c[1] *np.real(\n"
    integrand += f"(-1j) ** {order + 2} * {time_factor} * np.exp(({exponent})({args}))\n"
    integrand += ")"

    # generate integration range
    int_range_str = ""
    int_range_dict = {}
    for t in tl[::-1][:-1]:
        args = ",".join([f"t{i}" for i in range(t - 1, 1, -1)])
        range_func = f"lambda {args}: [0, t{t - 1}]"
        print(f"lambda {args}: [0, t{t - 1}]")
        int_range_dict[t] = eval(f"lambda {args}: [0, t{t - 1}]", {"t1": t1})
        int_range_str += f"int_range_dict[{t}],"

    int_range_str = int_range_str[:-1]

    integrator = f"mcmc_time_ordered({integrand}, {order + 1}, [0,{t1}], {N}, burn_in={burn_in})"

    integral, _, _ = eval(integrator, {"E": E, "c": c,
                                       "t1": t1, "pre": pre, "coth": coth, "w": w,
                                       "w_sq": w_sq, "const_exponent": const_exponent,
                                       "integrand": integrand, "int_range_dict": int_range_dict,
                                       "mcmc_time_ordered": mcmc_time_ordered, "np": np, "kbT": kbT})

    return -2 * integral


def fgr_rate3_correction_by_order_vegas(c_list, e_list, kbT, w, s_list, t_max, order, nitn=10, neval=1000):
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
        k, l = sub_list[i]
        delta[i] = s[k] - s[l]

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
        pre = {}
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

    import vegas

    int_interval = [0, t_max]
    integrator = vegas.Integrator([int_interval] * (order + 1))

    integral = integrator(integrand, nitn=nitn, neval=neval).mean
    return -2 * c[0] * c[2] ** order * c[1] * integral


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
        lambda ei: fgr_rate3_correction_by_order([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
                                                 1000, 0)
    )(e)
    end1 = time()
    print("finished")
    # fgr_rate3_correction_1_mcmc = np.vectorize(
    #         lambda ei: fgr_rate3_correction_by_order_mcmc([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
    #                                                  400, 2, 100000, burn_in=1000)
    #     )(e)

    fgr_rate3_correction_1_vegas = np.vectorize(
        lambda ei: fgr_rate3_correction_by_order_vegas([C_DA, C_DA, aa_coupling], [0, -ei, -ei], kbT, w, [-v * 0, v, v],
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
