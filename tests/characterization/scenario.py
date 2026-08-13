"""Deterministic small-scale characterization scenarios.

These reproduce a tiny multichannel interaction-picture spin-boson propagation
using whatever ``(SystemBath model, MPS engine)`` classes are passed in.  The same
scenario is therefore runnable against the *pre-refactor* and *post-refactor* code
and the resulting physical observables compared numerically, which is how we
guarantee the "unify + preserve" refactor does not silently change the numerics.

The randomized SVD used inside the engine draws from ``numpy.random``; we seed it
so runs are bitwise-reproducible as long as the engine's control flow (the
sequence of SVD calls) is unchanged -- which is exactly the invariant the
refactor must preserve.
"""
import io
from contextlib import redirect_stdout

import numpy as np


def _discrete_bath():
    """A tiny, fixed discrete multichannel bath (3 input modes -> 6 after doubling)."""
    freq = [10.0, 25.0, 40.0]
    coup1 = [5.0, -3.0, 2.0]   # coupling seen by |0>
    coup2 = [-5.0, 3.0, -2.0]  # coupling seen by |1>
    coup_mat = [np.diag([coup1[i], coup2[i]]) for i in range(len(freq))]
    return freq, coup_mat


def run_multichannel_ic(SystemBath, Mps, sigma_x, sigma_z, num_op, *, lbo=False,
                        phys_dim=4, bond_dim=30, threshold=1e-8, dt=1e-3,
                        num_steps=4, temp=100.0, seed=1234):
    """Run the scenario and return a dict of physical observables.

    Parameters
    ----------
    SystemBath : class
        Multichannel interaction-picture Hamiltonian builder, constructed as
        ``SystemBath(pd, coup_mat=..., freq=..., temp=..., h_sys=...)`` with
        ``.build(n=0)`` and ``.get_u(t, dt, factor=...)``.
    Mps : class
        TEBD engine constructed as ``Mps(pd)`` exposing ``B``, ``U``,
        ``update_bond`` and ``get_theta1``.
    lbo : bool
        Whether ``Mps.update_bond`` takes the extra ``eps_LBO`` argument.
    """
    np.random.seed(seed)
    freq, coup_mat = _discrete_bath()
    n_boson = 2 * len(freq)              # thermofield doubling
    pd = [2] + [phys_dim] * n_boson

    buf = io.StringIO()
    with redirect_stdout(buf):          # the legacy engine is extremely chatty
        eth = SystemBath(pd, coup_mat=coup_mat, freq=freq, temp=temp,
                               h_sys=130.0 * sigma_x + np.diag([0.0, -200.0]))
        eth.build(n=0)

        etn = Mps(pd)
        etn.B[0][0, 1, 0] = 0.0
        etn.B[0][0, 0, 0] = 1.0        # start in |0>

        def ub(j, swap):
            if lbo:
                etn.update_bond(j, bond_dim, threshold, swap, eps_lbo=1e-9)
            else:
                etn.update_bond(j, bond_dim, threshold, swap)

        spin_rho = []
        for tn in range(num_steps):
            U1, U2 = eth.get_u(2 * tn * dt, 2 * dt, factor=2)
            etn.U = U1
            for j in range(n_boson - 1):
                ub(j, 1)
            ub(n_boson - 1, 0)
            ub(n_boson - 1, 0)
            etn.U = U2
            for j in range(n_boson - 2, -1, -1):
                ub(j, 1)
            theta = etn.get_theta1(0)
            spin_rho.append(np.einsum('LiR,LjR->ij', theta, theta.conj()))

        boson_num_final = []
        for i, d in enumerate(pd[1:]):
            theta = etn.get_theta1(i + 1)
            r = np.einsum('LiR,LjR->ij', theta, theta.conj())
            boson_num_final.append(np.einsum('ij,ji', r, num_op(d)).real)

    spin_rho = np.array(spin_rho)                       # (num_steps, 2, 2) complex
    pop_z = np.einsum('tij,ji->t', spin_rho, sigma_z).real
    return {
        "spin_rho": spin_rho,
        "pop_z": pop_z,
        "boson_num_final": np.array(boson_num_final),
    }
