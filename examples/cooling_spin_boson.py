"""Chain "cooling" spin-boson dynamics.

Demonstrates the finite-temperature "cooling" scheme, in which each bath site
carries a heating operator applied when reading out the reduced density matrix
(:meth:`get_rdm`), emulating a dissipative steady state controlled by the
parameter ``betaOmega``.

Run with:  python examples/cooling_spin_boson.py
"""
import numpy as np

from fishbonett.representations.coolingchain import SystemBathCoolingChain
from fishbonett.operators import sigma_x, sigma_z, temp_factor
from fishbonett.spectral_densities import drude


def main():
    n_boson = 20
    pd = [2] + [12] * n_boson          # system on site 0, then the bath chain

    g = 1000.0
    temp = 226.0
    eth = SystemBathCoolingChain(
        pd, betaOmega=0.2, h_sys=78.5 * sigma_x, coupling=sigma_z,
        sd=lambda w: drude(w, lam=4 * 39.0, gam=78.5) * temp_factor(temp, w),
        domain=[-g, g]).build()

    dt, n_steps, bond_dim, eps = 5e-4, 20, 200, 1e-4
    u_one = eth.get_u(2 * dt)
    u_half = eth.get_u(dt)
    even, odd = list(range(0, n_boson, 2)), list(range(1, n_boson, 2))

    pops = []
    for _ in range(n_steps):
        eth.U = u_half
        for j in odd:
            eth.update_bond(j, bond_dim, eps, swap=0)
        eth.U = u_one
        for j in even:
            eth.update_bond(j, bond_dim, eps, swap=0)
        eth.U = u_half
        for j in odd:
            eth.update_bond(j, bond_dim, eps, swap=0)
        rho = eth.get_rdm()
        pops.append(np.einsum('ij,ji', rho, sigma_z).real)

    pops = np.array(pops)
    print("<sigma_z>(t):", np.round(pops, 4))
    return pops


if __name__ == "__main__":
    main()
