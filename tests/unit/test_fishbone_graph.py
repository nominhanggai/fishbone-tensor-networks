"""Arbitrary electronic graphs on a comb tensor-network state."""
import numpy as np
import pytest
from scipy.linalg import expm

from fishbonett import Fishbone
from fishbonett import Bath


N = np.diag([0.0, 1.0])


def _one_excitation_projection(n_sites):
    columns = []
    for excited in range(n_sites):
        vector = np.array([1.0 + 0j])
        for site in range(n_sites):
            vector = np.kron(vector, [0, 1] if site == excited else [1, 0])
        columns.append(vector)
    return np.stack(columns, axis=1)


def _dense_model(model):
    n_sites = len(model.sites)
    dimension = 2 ** n_sites
    total = np.zeros((dimension, dimension), complex)
    for site, operator in enumerate(model.sites):
        factors = [np.eye(2)] * n_sites
        factors[site] = operator
        term = factors[0]
        for factor in factors[1:]:
            term = np.kron(term, factor)
        total += term
    for (left, right), operator in model.graph_couplings.items():
        shaped = operator.reshape(2, 2, 2, 2)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        factors = [np.eye(2)] * n_sites
                        factors[left] = np.eye(2)[a:a+1].T @ np.eye(2)[c:c+1]
                        factors[right] = np.eye(2)[b:b+1].T @ np.eye(2)[d:d+1]
                        term = factors[0]
                        for factor in factors[1:]:
                            term = np.kron(term, factor)
                        total += shaped[a, b, c, d] * term
    return total


def test_single_excitation_builder_reproduces_input_hamiltonian():
    h = np.array([[1.0, 0.2 + 0.1j, -0.3],
                  [0.2 - 0.1j, 2.0, 0.4j],
                  [-0.3, -0.4j, 1.5]])
    model = Fishbone.from_single_excitation(h, baths={})
    projection = _one_excitation_projection(3)
    represented = projection.conj().T @ _dense_model(model) @ projection
    assert np.allclose(represented, h)


def test_cyclic_graph_tebd_matches_exact_dynamics():
    h = np.array([[0.1, 0.25, -0.12j],
                  [0.25, -0.2, 0.18],
                  [0.12j, 0.18, 0.35]])
    model = Fishbone.from_single_excitation(h, baths={})
    initial = [[0, 1], [1, 0], [1, 0]]
    result = model.run(
        dt=0.002, n_steps=30, initial=initial,
        observables={"population": N}, bond_dim=64, trunc_eps=1e-13)
    psi0 = np.array([1, 0, 0], complex)
    expected = []
    for time in result.t:
        psi = expm(-1j * h * time) @ psi0
        expected.append(np.abs(psi) ** 2)
    assert np.max(np.abs(result.expect["population"] - expected)) < 2e-5


def test_graph_input_validation():
    sites = [np.zeros((2, 2))] * 3
    with pytest.raises(ValueError, match="either backbone or couplings"):
        Fishbone(sites, {}, backbone=[np.zeros((4, 4))] * 2, couplings={})
    with pytest.raises(ValueError, match="i < j"):
        Fishbone(sites, {}, couplings={(1, 0): np.zeros((4, 4))})
    with pytest.raises(ValueError, match="outside"):
        Fishbone(sites, {}, couplings={(0, 3): np.zeros((4, 4))})
    with pytest.raises(ValueError, match="Hermitian"):
        Fishbone(sites, {}, couplings={(0, 1): np.triu(np.ones((4, 4)))})


def test_multisite_interaction_chain_matches_schrodinger_chain():
    h = np.array([[0.2, 0.3], [0.3, -0.1]])
    bath = Bath.vibronic([1.7], [0.08], phys_dim=3)
    coupling = N
    baths = {site: bath.bind(coupling) for site in range(2)}
    initial = [[0, 1], [1, 0]]
    static = Fishbone.from_single_excitation(h, baths=baths).run(
        dt=0.004, n_steps=20, initial=initial,
        observables={"population": N}, bond_dim=100, trunc_eps=1e-12)
    interaction = Fishbone.from_single_excitation(h, baths=baths).run(
        dt=0.004, n_steps=20, initial=initial,
        method="interaction-chain-fishbone-tebd",
        observables={"population": N},
        bond_dim=100, trunc_eps=1e-12)
    assert interaction.method == "interaction-chain-fishbone-tebd"
    assert np.max(np.abs(interaction.expect["population"]
                         - static.expect["population"])) < 2e-4
