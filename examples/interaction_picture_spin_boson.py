"""Interaction-picture spin-boson dynamics with a discrete multichannel bath.

Propagates the population <sigma_z>(t) of a two-level system linearly coupled to a
discrete set of harmonic modes, in the interaction picture with respect to the
system-bath coupling. The bath is mapped to a chain by a Lanczos tridiagonalisation
and evolved with the unified TEBD engine using leg swaps.

Run with:  python examples/interaction_picture_spin_boson.py
"""
import numpy as np

from fishbonett.backwardSpinBosonMultiChannel import BosonicBath
from fishbonett.mps import BosonicBathMPS
from fishbonett.stuff import sigma_x, sigma_z


def main():
    # A small discrete bath: 3 modes -> 6 after thermofield doubling.
    freq = [10.0, 25.0, 40.0]
    coup = [(5.0, -5.0), (-3.0, 3.0), (2.0, -2.0)]     # (|0>, |1>) mode couplings
    coup_mat = [np.diag(c) for c in coup]
    n_boson = 2 * len(freq)
    pd = [10] * n_boson + [2]                           # boson dims + spin

    eth = BosonicBath(pd, coup_mat=coup_mat, freq=freq, temp=100.0)
    eth.h1e = 130.0 * sigma_x + np.diag([0.0, -200.0])  # system Hamiltonian
    eth.build(n=0)

    etn = BosonicBathMPS(pd)
    etn.B[-1][0, 0, 0] = 1.0                            # start in |0>

    dt, n_steps, chi, eps = 1e-3, 30, 40, 1e-6
    pops = []
    for tn in range(n_steps):
        u1, u2 = eth.get_u(2 * tn * dt, 2 * dt, factor=2)
        etn.U = u1
        for j in range(n_boson - 1, 0, -1):
            etn.update_bond(j, chi, eps, swap=1)
        etn.update_bond(0, chi, eps, swap=0)
        etn.update_bond(0, chi, eps, swap=0)
        etn.U = u2
        for j in range(1, n_boson):
            etn.update_bond(j, chi, eps, swap=1)
        theta = etn.get_theta1(n_boson)
        rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
        pops.append(np.einsum('ij,ji', rho, sigma_z).real)

    pops = np.array(pops)
    print("<sigma_z>(t):", np.round(pops, 4))
    return pops


if __name__ == "__main__":
    main()
