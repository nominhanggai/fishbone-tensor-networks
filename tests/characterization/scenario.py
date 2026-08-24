"""Deterministic multichannel regression scenario."""

import numpy as np


def _discrete_bath():
    """A tiny, fixed discrete multichannel bath (3 input modes -> 6 after doubling)."""
    freq = [10.0, 25.0, 40.0]
    coup1 = [5.0, -3.0, 2.0]   # coupling seen by |0>
    coup2 = [-5.0, 3.0, -2.0]  # coupling seen by |1>
    coup_mat = [np.diag([coup1[i], coup2[i]]) for i in range(len(freq))]
    return freq, coup_mat


def run_multichannel_ic(
    representation_cls,
    mps_cls,
    sigma_x,
    sigma_z,
    num_op,
    *,
    lbo=False,
    phys_dim=4,
    bond_dim=30,
    threshold=1e-8,
    dt=1e-3,
    num_steps=4,
    temp=100.0,
    seed=1234,
):
    """Run the scenario and return a dict of physical observables.

    Parameters
    ----------
    representation_cls : type
        Multichannel interaction-picture representation exposing ``tebd_gates``.
    mps_cls : type
        MPS state class exposing ``B``, ``U``,
        ``update_bond`` and ``get_theta1``.
    lbo : bool
        Whether ``mps_cls.update_bond`` takes the extra ``eps_lbo`` argument.
    """
    np.random.seed(seed)
    freq, coup_mat = _discrete_bath()
    n_boson = 2 * len(freq)              # thermofield doubling
    pd = [2] + [phys_dim] * n_boson

    representation = representation_cls.from_positive_star(
        pd,
        coup_mat=coup_mat,
        freq=freq,
        temp=temp,
        h_sys=130.0 * sigma_x + np.diag([0.0, -200.0]),
    )
    representation.build(n=0)

    state = mps_cls(pd)
    state.B[0][0, 1, 0] = 0.0
    state.B[0][0, 0, 0] = 1.0

    def update_bond(index, swap):
        if lbo:
            state.update_bond(
                index, bond_dim, threshold, swap, eps_lbo=1e-9,
            )
        else:
            state.update_bond(index, bond_dim, threshold, swap)

    spin_rho = []
    for step in range(num_steps):
        first, second = representation.tebd_gates(
            2 * step * dt, 2 * dt, factor=2,
        )
        state.U = first
        for index in range(n_boson - 1):
            update_bond(index, 1)
        update_bond(n_boson - 1, 0)
        update_bond(n_boson - 1, 0)
        state.U = second
        for index in range(n_boson - 2, -1, -1):
            update_bond(index, 1)
        theta = state.get_theta1(0)
        spin_rho.append(np.einsum("LiR,LjR->ij", theta, theta.conj()))

    boson_num_final = []
    for index, dimension in enumerate(pd[1:]):
        theta = state.get_theta1(index + 1)
        rho = np.einsum("LiR,LjR->ij", theta, theta.conj())
        boson_num_final.append(
            np.einsum("ij,ji", rho, num_op(dimension)).real,
        )

    spin_rho = np.array(spin_rho)                       # (num_steps, 2, 2) complex
    pop_z = np.einsum("tij,ji->t", spin_rho, sigma_z).real
    return {
        "spin_rho": spin_rho,
        "pop_z": pop_z,
        "boson_num_final": np.array(boson_num_final),
    }
