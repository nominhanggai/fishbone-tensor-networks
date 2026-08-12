"""Spin-boson dynamics in the interaction picture with respect to H_SB.

Demonstrates the keyword-constructed :class:`fishbonett.int_pic_hsb_spin_boson.
SpinBosonModel` -- the tidied high-level API in which the model is built from
physical parameters rather than by mutating attributes -- with a symmetric
(half-full-half) Trotter sweep over the boson chain.

Run with:  python examples/hsb_interaction_picture.py
"""
import numpy as np

from fishbonett.int_pic_hsb_spin_boson import SpinBosonModel
from fishbonett.spin_boson_mps import SpinBosonMPS
from fishbonett.stuff import drude


def main():
    n_boson = 6
    pd_boson = [8] * n_boson
    g = 3000.0
    sd = lambda w: 10.0 * drude(w, 1000, 5)             # Drude spectral density
    dt = 0.01 / 20

    eth = SpinBosonModel(v_x=300.0, v_z=0.0, pd_spin=2, pd_boson=pd_boson,
                         boson_domain=[0.0, g], sd=sd, dt=dt)
    etn = SpinBosonMPS(pd_spin=2, pd_boson=pd_boson)
    etn.B[0][0, 0, 0] = 1.0                             # system site first; start in |0>

    bond_dim, eps, n_steps = 200, 1e-4, 30
    pops = []
    for n in range(n_steps):
        u_one, u_half = eth.get_u(n * dt, dt)
        etn.U = u_half
        for j in range(0, n_boson, 2):
            etn.update_bond(j, bond_dim, eps, swap=0)
        etn.U = u_one
        for j in range(1, n_boson, 2):
            etn.update_bond(j, bond_dim, eps, swap=0)
        etn.U = u_half
        for j in range(0, n_boson, 2):
            etn.update_bond(j, bond_dim, eps, swap=0)
        theta = etn.get_theta1(0)
        rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
        pops.append(abs(rho[0, 0]))

    pops = np.array(pops)
    print("donor population rho[0,0](t):", np.round(pops, 4))
    return pops


if __name__ == "__main__":
    main()
