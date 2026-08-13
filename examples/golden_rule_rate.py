"""Fermi golden-rule vs Marcus electron-transfer rate from a spectral density.

Discretises a Lorentzian (Brownian) spectral density and compares the numerically
exact Fermi golden-rule rate with the Marcus expression across a range of driving
energies. Reproduces the trend of Fig. 2 of J. Phys. Chem. A 2013, 117, 6196.

Run with:  python examples/golden_rule_rate.py
"""
import numpy as np

from fishbonett.bath.legendre import get_vn_squared
from fishbonett.rates import fgr_rate, marcus_rate


def main():
    reorg_e, Omega, kbT, eta = 2.39e-2, 3.5e-4, 9.5e-4, 1.2e-3
    domain = [0.0, 5e-3]
    c_da, coup = 5e-5, 5e-3
    j = lambda w: 0.5 * (4 * reorg_e) * Omega ** 2 * eta * w \
        / ((Omega ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2)

    w, v_sq = get_vn_squared(j, 100, domain)
    energies = np.linspace(0.015, 0.03, 10)
    rate_fgr = np.array([fgr_rate(c_da, e - coup, kbT, w, v_sq) for e in energies])
    rate_marcus = np.array([marcus_rate(c_da, e - coup, kbT, reorg_e) for e in energies])

    print(f"{'E (a.u.)':>10} {'FGR rate':>14} {'Marcus rate':>14}")
    for e, rf, rm in zip(energies, rate_fgr, rate_marcus):
        print(f"{e:>10.4f} {rf:>14.4e} {rm:>14.4e}")
    return energies, rate_fgr, rate_marcus


if __name__ == "__main__":
    main()
