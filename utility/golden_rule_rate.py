import numpy as np
from scipy import integrate
from legendre_discretization import get_vn_squared, get_approx_func

"""
Calculate fermi's golden rule rate for electron transfer reactions, given a spectral density.
"""


def fgr_rate(c, e, kbT, _w, _v_sq):
    w = np.array(_w)
    v_sq = np.array(_v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))
    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) - 1j * np.sin(t * w)))
    integrand = lambda t: np.real(np.exp(exponent(t)) * np.exp(-1j * e * t))
    integral, _ = integrate.quad(integrand, 0, np.inf, limit=5000)
    return 2 * (c ** 2) * integral


def marcus_rate(c: float, e: float, kbT: float, reorg_e: float):
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

    w, v_sq = get_vn_squared(j, 1000, domain)
    print("Discrete Reorganization E", np.sum(v_sq / w / np.pi))

    x = np.linspace(*domain, 1000)
    plt.plot(x, j(x), label='j')
    plt.savefig('spectral_density.png')
    plt.clf()

    e = np.linspace(0.015, 0.03, 20)
    rate_fgr = np.vectorize(lambda ei: fgr_rate(C_DA, ei, kbT, w, v_sq)
                            )(e)
    rate_marcus = np.vectorize(lambda ei: marcus_rate(C_DA, ei, kbT, reorg_e))(e)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(e, rate_fgr, 'o-', label='FGR rate')
    ax.plot(e, rate_marcus, 'x', label='Marcus rate')
    ax.legend()
    ax.set_xlabel('E (a.u.)')
    ax.set_ylabel('Rate (a.u.)')
    ax.set_xlim(0.015, 0.03)

    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 0.4)

    fig.savefig('Figure_2.png')
