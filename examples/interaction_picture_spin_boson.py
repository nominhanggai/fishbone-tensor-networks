"""Interaction-picture spin-boson dynamics with a discrete multichannel bath.

Propagates the population <sigma_z>(t) of a two-level system linearly coupled to a
discrete set of harmonic modes, in the interaction picture with respect to the
*free bath*.  Two coupling channels share the same modes, so the noise they impose
is cross-correlated (see :mod:`fishbonett.frames.multichannel`).

This is the **low-level** route, and the reason to reach for it is the bath: the
modes are given explicitly rather than sampled from a continuous ``J(omega)``.  For
a continuous spectral density use the high-level interface instead --
``SimpleSysBath(...).run(method="multichannel-ip")``, which builds the same
frame from a :class:`~fishbonett.bath.spec.Bath`.

Run with:  python examples/interaction_picture_spin_boson.py
"""
import numpy as np

from fishbonett.frames.multichannel import SimpleSysBathMultiChannel
from fishbonett.states.mps import SimpleSysBathMPS
from fishbonett.evolve import tebd
from fishbonett.operators import sigma_x, sigma_z


def main():
    # A small discrete bath: 3 modes -> 6 after thermofield doubling.  The
    # constructor mirrors `freq` onto the negative axis and applies the thermal
    # weight, so `pd` needs 2 * len(freq) boson sites.
    freq = [10.0, 25.0, 40.0]
    coup = [(5.0, -5.0), (-3.0, 3.0), (2.0, -2.0)]     # (|0>, |1>) mode couplings
    coup_mat = [np.diag(c) for c in coup]
    n_boson = 2 * len(freq)
    pd = [2] + [10] * n_boson                          # system on site 0, then bath

    eth = SimpleSysBathMultiChannel(
        pd, coup_mat=coup_mat, freq=freq, temp=100.0,
        h_sys=130.0 * sigma_x + np.diag([0.0, -200.0])).build(n=0)

    etn = SimpleSysBathMPS(pd)
    etn.B[0][:] = 0.0
    etn.B[0][0, 0, 0] = 1.0                            # start in |0>

    dt, n_steps, chi, eps = 1e-3, 30, 40, 1e-6
    pops = []
    for tn in range(n_steps):
        # One symmetric swap-network step: the system is walked along the chain so
        # that every mode gets its turn adjacent to it, then walked back.
        tebd.symmetric_swap_step(etn, eth, tn * dt, dt, n_boson, chi, eps)
        theta = etn.get_theta1(0)
        rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
        pops.append(np.einsum('ij,ji', rho, sigma_z).real / np.trace(rho).real)

    pops = np.array(pops)
    print("<sigma_z>(t):", np.round(pops, 4))
    return pops


if __name__ == "__main__":
    main()
