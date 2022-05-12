import numpy as np
from scipy import integrate
from legendre_discretization import get_vn_squared, get_approx_func

"""
Calculate fermi's golden rule rate for electron transfer reactions, given a spectral density.
"""

k = 0.69


def fgr_rate_riemann_sum(c, e, kbT, dt, t, _w, _v_sq):
    w = np.array(_w)
    v_sq = np.array(_v_sq)
    t_grid = np.array([n * dt for n in range(int(t / dt))])
    print("t_grid:", t_grid, len(t_grid))
    print("Calculating w_t_tensor_in_exponent...")
    j_factor = (-v_sq * 4 / np.pi / w ** 2).reshape(len(w), 1)
    print("Finished j factor")
    coth = 1 / np.tanh(w / (2 * kbT)).reshape(len(w), 1)
    print("Finished coth")
    wt_grid = np.outer(w, t_grid)
    summand_1 = coth * (1 - np.cos(wt_grid))
    print("Finished summand 1", summand_1.shape)
    print(wt_grid.shape)
    summand_2 = 1j * np.sin(wt_grid)
    print("Finished summand 2", summand_2.shape, np.sin(wt_grid).shape)
    exponent = np.sum(j_factor * (summand_1 - summand_2), axis=0)
    print("Finished exponent", exponent.shape)
    t_tensor = np.exp(exponent) * np.exp(1j * e * t_grid) * dt
    import matplotlib.pyplot as plt
    plt.plot(t_grid, t_tensor.real, 'o', label='real')
    plt.plot(t_grid, t_tensor.imag, 'o', label='imag')
    plt.legend()
    plt.savefig('w_t_tensor.png')
    real_part = np.sum(t_tensor).real
    return 2 * (c ** 2) * real_part


def fgr_rate(c, e, kbT, _w, _v_sq):
    w = np.array(_w)
    v_sq = np.array(_v_sq)
    j_factor = (-v_sq / np.pi / w ** 2)
    coth = 1 / np.tanh(w / (2 * kbT))
    exponent = lambda t: np.sum(j_factor * (coth * (1 - np.cos(w * t)) - -1j * np.sin(t * w)))
    integrand = lambda t: np.real(np.exp(exponent(t)) * np.exp(1j * e * t))
    integral, _ = integrate.quad(integrand, 0, np.inf, limit=5000)
    return 2 * (c ** 2) * integral


def marcus_rate(c: float, e: float, kbT: float, reorg_e: float):
    return 2 * np.pi * c ** 2 / np.sqrt(4 * np.pi * kbT * reorg_e) * np.exp(
        -(reorg_e - e) ** 2 / (4 * kbT * reorg_e))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Lorentzian spectral density
    reorg_e = 3000
    Omega = 200
    Omega2 = 400
    kbT = 0.69 * 3
    eta = 200
    domain = [0, 3000]
    C_DA = 500
    j = lambda w: 0.5 * (4 * reorg_e) * Omega ** 2 * eta * w / ((Omega ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2)
    # j = lambda w: 0.5 * (4 * reorg_e/2) * Omega ** 2 * eta * w / ((Omega ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2) \
    # + 0.5 * (4 * reorg_e/2) * Omega2 ** 2 * eta * w / ((Omega2 ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2)

    w, v_sq = get_vn_squared(j, 1000, domain)
    print("Discretized Reorganization E", np.sum(v_sq / w / np.pi))

    x = np.linspace(*domain, 1000)
    plt.plot(x, j(x), label='j')
    plt.savefig('j.png')
    plt.clf()

    temp = np.linspace(5, 600, 20)
    e = np.linspace(4167, 4333, 10)
    kb = 0.69

    for ei in e:
        r1_fgr = np.array([fgr_rate(C_DA, ei, kb * Ti, w, v_sq) for Ti in temp])
        r2_fgr = np.array([fgr_rate(np.sqrt(2) * C_DA, ei, kb * Ti, w, v_sq * 2) for Ti in temp])
        plt.plot(temp, r2_fgr / r1_fgr, label=f'$\Delta G$={ei: .0f}')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig('fgr_ratio.png', bbox_inches="tight")
    exit()
    r1_marcus = np.array([marcus_rate(C_DA, ei, kbT, reorg_e) for ei in e])
    r2_marcus = np.array([marcus_rate(np.sqrt(2) * C_DA, ei, kbT, reorg_e * 2) for ei in e])
    plt.plot(e, r1_fgr, label='1 Acceptor FGR')
    plt.plot(e, r2_fgr, label='2 Acceptor FGR')
    plt.plot(e, r1_marcus, label='1 Acceptor Marcus')
    plt.plot(e, r2_marcus, label='2 Acceptor Marcus')
    plt.legend()
    plt.savefig('rate.png')
    plt.clf()

    # Ratio
    plt.plot(e, r2_fgr / r1_fgr, label='Ratio FGR')
    plt.legend()
    plt.savefig('rate_ratio_fgr.png')
    plt.clf()
    plt.plot(e, r2_marcus / r1_marcus, label='Ratio Marcus')
    plt.legend()
    plt.savefig('rate_ratio_marcus.png')
    plt.clf()