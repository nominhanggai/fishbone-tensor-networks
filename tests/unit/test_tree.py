"""Unit tests for the tree tensor-network engine, validated against exact
diagonalization of the star Hamiltonian (interaction- and Schroedinger-picture
<sigma_z> agree because sigma_z commutes with the bath)."""
import numpy as np
import pytest

from fishbonett import Bath
from fishbonett.bath.chain import star_transform
from fishbonett.evolve._modetree_core import _hamiltonian_direct
from fishbonett.evolve.modetree import (run_tree_tebd, build_balanced_tree, build_tree_mpo,
                             tree_depth, hamiltonian_from_mpo, SZ, SX)
from fishbonett.operators import annihilate, create
from fishbonett.representations.interaction import InteractionRepresentation

DOMAIN = (-25.0, 36.0)


def test_tensor_network_rejects_duplicate_edges_before_building_gauges():
    from fishbonett.states.tree import TreeTensorNetwork

    with pytest.raises(ValueError, match="duplicate edge"):
        TreeTensorNetwork([2, 2, 2], [(0, 1), (1, 0)], root=0)


def _Jb(w):
    aw = abs(w)
    if aw < 1e-12:
        return 0.0
    nb = 1.0 / np.expm1(aw)
    j = 0.2 * aw * np.exp(-aw / 5.0)
    return j * (nb + 1.0) if w > 0 else j * nb


def _embed(op, site, dims):
    mats = [np.eye(dm) for dm in dims]
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _exact_sz(n_chain, d, V, ts):
    freq, Vn, _ = star_transform(_Jb, n_chain, DOMAIN)
    dims = [2] + [d] * n_chain
    H = _embed(V * SX, 0, dims)
    for k in range(n_chain):
        H = H + freq[k] * _embed(create(d) @ annihilate(d), 1 + k, dims)
        H = H + Vn[k] * (_embed(SZ, 0, dims) @ _embed(annihilate(d) + create(d), 1 + k, dims))
    E, Uv = np.linalg.eigh(H)
    psi0 = np.zeros(int(np.prod(dims)), dtype=complex)
    psi0[0] = 1.0
    coef = Uv.conj().T @ psi0
    szop = _embed(SZ, 0, dims)
    return np.array([(lambda p: (p.conj() @ (szop @ p)).real)
                     (Uv @ (np.exp(-1j * E * t) * coef)) for t in ts])


def _representation(n_chain, d, V):
    bath = Bath(J=_Jb, domain=DOMAIN, n_modes=n_chain, phys_dim=d)
    return InteractionRepresentation(
        representation="interaction-chain", h_sys=V * SX,
        coupling=SZ, bath=bath).build()


def test_tree_mpo_reproduces_direct_hamiltonian():
    rng = np.random.default_rng(0)
    for n_modes in (1, 2, 3, 4):
        d = 3
        dcoup = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)
        nodes, root, _ = build_balanced_tree(n_modes, d)
        build_tree_mpo(nodes, root, dcoup, 0.7 * SX + 0.5 * 0.4 * SZ, SZ)
        Hmpo = hamiltonian_from_mpo(nodes, root, n_modes, d)
        Hdir = _hamiltonian_direct(dcoup, 0.7, 0.4, d, n_modes)
        assert np.abs(Hmpo - Hdir).max() < 1e-12
        assert np.abs(Hmpo - Hmpo.conj().T).max() < 1e-12


def test_tree_is_shallower_than_chain():
    nodes, root, _ = build_balanced_tree(64, d=3)
    assert tree_depth(nodes, root) < 64        # log-depth, vs chain length 65


def test_tree_tebd_matches_exact():
    n_chain, d, V = 3, 5, 1.0
    t, sz = run_tree_tebd(
        _representation(n_chain, d, V), dt=0.05, nsteps=12, bond_dim=30,
        trunc_eps=1e-12)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.max(np.abs(sz - sz_ex)) < 5e-3


def test_tree_driver_preserves_public_chain_mode_order(monkeypatch):
    """Tree leaf ``mode=k`` must receive interaction coefficient ``k``."""
    from fishbonett.evolve import _modetree_driver as driver

    class DistinctCoefficients:
        h_sys = SX
        coupling = SZ
        dimensions = (2, 3, 3, 3)

        @staticmethod
        def interval_coefficients(_time, _dt):
            return np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])

    captured = []
    build = driver.build_coupling_op

    def record(nodes, root, amplitudes, coupling):
        captured.append(np.array(amplitudes, copy=True))
        return build(nodes, root, amplitudes, coupling)

    monkeypatch.setattr(driver, "build_coupling_op", record)
    driver.run_tree_tebd(
        DistinctCoefficients(), dt=0.01, nsteps=1, bond_dim=8,
        trunc_eps=1e-12,
    )
    np.testing.assert_array_equal(
        captured[0], DistinctCoefficients.interval_coefficients(0.0, 0.01)
    )
