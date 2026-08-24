import numpy as np
import pytest

from fishbonett import (
    Bath, Fishbone, GibbsPurification, SimulationCheckpoint, TreeFishbone,
    energy_current_operator,
)
from fishbonett.operators import sigma_x, sigma_y, sigma_z
from fishbonett.states.tree import TreeTensorNetwork


def _partial_trace_ancillas(vector, dimensions):
    n = len(dimensions)
    tensor = vector.reshape(*[d * d for d in dimensions])
    tensor = tensor.reshape(*sum(([d, d] for d in dimensions), []))
    physical = list(range(0, 2 * n, 2))
    ancilla = list(range(1, 2 * n, 2))
    matrix = np.transpose(tensor, physical + ancilla).reshape(
        int(np.prod(dimensions)), int(np.prod(dimensions)))
    return matrix @ matrix.conj().T


def test_gibbs_purification_reduces_to_exact_density_matrix():
    sites = [0.3 * sigma_z, -0.2 * sigma_z]
    bond = 0.4 * (np.kron(sigma_x, sigma_x)
                  + 0.7 * np.kron(sigma_z, sigma_z))
    thermal = GibbsPurification(sites, [bond], temperature=0.8)

    energy, vectors = np.linalg.eigh(thermal.hamiltonian)
    weights = np.exp(-(energy - energy.min()) / 0.8)
    exact = (vectors * (weights / weights.sum())) @ vectors.conj().T
    reduced = _partial_trace_ancillas(
        thermal.vector, thermal.physical_dimensions)
    assert np.allclose(reduced, exact, atol=1e-13)

    lifted = thermal.lift_operator(np.kron(sigma_z, sigma_x), [0, 1])
    pure_expectation = np.vdot(thermal.vector, lifted @ thermal.vector)
    exact_expectation = np.trace(exact @ np.kron(sigma_z, sigma_x))
    assert pure_expectation == pytest.approx(exact_expectation, abs=1e-13)
    assert np.array_equal(
        thermal.lift_site_operator(sigma_z, 1),
        thermal.lift_operator(sigma_z, [1]))


def test_site_indexed_bath_mapping_is_explicit_and_validated():
    bath = Bath(
        J=lambda w: 0.03 * w * np.exp(-w), domain=(0.0, 4.0),
        n_modes=2, phys_dim=3, discretization="tedopa")
    left = bath.bind(sigma_x)
    right = bath.bind(sigma_z)

    chain = Fishbone(
        sites=[sigma_z, sigma_z, sigma_z],
        baths={0: left, 2: right})
    assert chain.baths == [[left], [], [right]]

    tree = TreeFishbone(
        sites=[sigma_z, sigma_z, sigma_z],
        edges=[(0, 1), (1, 2)], baths={0: left, 2: right})
    assert tree.baths[0] == [left]
    assert tree.baths[1] == []
    assert tree.baths[2] == [right]

    with pytest.raises(TypeError, match="integer site indices"):
        Fishbone(sites=[sigma_z], baths={"left": left})
    with pytest.raises(ValueError, match="0 <= site < 1"):
        Fishbone(sites=[sigma_z], baths={1: left})


def test_gibbs_mps_embeds_in_a_tree_with_spectator_arms():
    bond = 0.25 * np.kron(sigma_x, sigma_x)
    thermal = GibbsPurification(
        [0.1 * sigma_z, -0.2 * sigma_z, 0.3 * sigma_z],
        [bond, bond], beta=1.2)
    # System path 0-1-2 with two product-state bath arms.
    state = TreeTensorNetwork(
        [4, 4, 4, 3, 3], [(0, 1), (1, 2), (0, 3), (2, 4)])
    thermal.initialize_tree(state, range(3))
    reduced = state.joint_rdm([0, 1, 2])
    exact = np.outer(thermal.vector, thermal.vector.conj())
    assert np.allclose(reduced, exact, atol=2e-13)
    tensor = thermal.vector.reshape(4, 4, 4)
    for site in range(3):
        local = np.moveaxis(tensor, site, 0).reshape(4, -1)
        expected = local @ local.conj().T
        assert np.allclose(state.rdm(site), expected, atol=2e-13)


def test_energy_current_is_the_continuity_equation_operator():
    h = [0.2 * sigma_z, -0.1 * sigma_z, 0.3 * sigma_z]
    left = 0.4 * (np.kron(sigma_x, sigma_x)
                  + np.kron(sigma_y, sigma_y))
    right = 0.3 * (np.kron(sigma_x, sigma_x)
                   + 0.8 * np.kron(sigma_z, sigma_z))
    current = energy_current_operator(h[1], right, left)
    h_left = (np.kron(left, np.eye(2))
              + np.kron(np.eye(2), np.kron(h[1], np.eye(2))))
    crossing = np.kron(np.eye(2), right)
    expected = 1j * (h_left @ crossing - crossing @ h_left)
    assert np.allclose(current, expected)
    assert np.allclose(current, current.conj().T)

    rng = np.random.default_rng(12)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    full_h = (np.kron(np.kron(h[0], np.eye(2)), np.eye(2))
              + np.kron(np.eye(2), np.kron(h[1], np.eye(2)))
              + np.kron(np.eye(4), h[2])
              + np.kron(left, np.eye(2)) + crossing)
    region = (np.kron(np.kron(h[0], np.eye(2)), np.eye(2))
              + h_left)
    derivative = np.vdot(psi, (1j * (full_h @ region - region @ full_h)) @ psi)
    outgoing = np.vdot(psi, current @ psi)
    assert outgoing == pytest.approx(-derivative, abs=2e-13)


def _checkpoint_model(field=0.2):
    bath = Bath(
        J=lambda w: 0.03 * w * np.exp(-w), domain=(0.0, 4.0),
        n_modes=2, phys_dim=3, discretization="tedopa")
    exchange = 0.15 * np.kron(sigma_x, sigma_x)
    return Fishbone(
        sites=[field * sigma_z, -0.1 * sigma_z],
        backbone=[exchange],
        baths=[bath.bind(sigma_x), bath.bind(sigma_x)])


def test_checkpoint_continuation_matches_one_shot_and_roundtrips(tmp_path):
    observable = {"z": sigma_z}
    model = _checkpoint_model()
    whole = model.run(
        dt=0.02, n_steps=6, bath_horizon=0.12,
        trunc_eps=1e-12, bond_dim=100, observables=observable)
    first = model.run(
        dt=0.02, n_steps=2, bath_horizon=0.12,
        trunc_eps=1e-12, bond_dim=100, observables=observable)
    path = first.checkpoint.save(tmp_path / "state.npz")
    loaded = SimulationCheckpoint.load(path)
    second = model.run(
        dt=0.02, n_steps=4, resume=loaded,
        trunc_eps=1e-12, bond_dim=100, observables=observable)

    joined = np.concatenate([first.expect["z"], second.expect["z"]])
    assert np.allclose(joined, whole.expect["z"], atol=2e-12)
    assert np.allclose(second.rdm[-1], whole.rdm[-1], atol=2e-12)
    assert np.allclose(second.t, [0.06, 0.08, 0.10, 0.12])
    assert second.checkpoint.elapsed == pytest.approx(0.12)
    assert loaded.signature == first.checkpoint.signature
    assert all(np.array_equal(a, b)
               for a, b in zip(loaded.tensors, first.checkpoint.tensors))


def test_checkpoint_rejects_incompatible_or_overlong_continuation():
    first = _checkpoint_model().run(
        dt=0.02, n_steps=2, bath_horizon=0.08,
        trunc_eps=1e-12, bond_dim=100)
    with pytest.raises(ValueError, match="exceeds.*bath_horizon"):
        _checkpoint_model().run(
            dt=0.02, n_steps=3, resume=first.checkpoint,
            trunc_eps=1e-12, bond_dim=100)
    with pytest.raises(ValueError, match="Hamiltonian does not match"):
        _checkpoint_model(field=0.25).run(
            dt=0.02, n_steps=1, resume=first.checkpoint,
            trunc_eps=1e-12, bond_dim=100)
    with pytest.raises(ValueError, match="initial and resume"):
        _checkpoint_model().run(
            dt=0.02, n_steps=1, resume=first.checkpoint, initial="up",
            trunc_eps=1e-12, bond_dim=100)


def test_observation_stride_changes_sampling_not_final_state():
    model = _checkpoint_model()
    dense = model.run(
        dt=0.02, n_steps=5, bath_horizon=0.10, trunc_eps=1e-12,
        bond_dim=100, observables={"z": sigma_z})
    sparse = model.run(
        dt=0.02, n_steps=5, bath_horizon=0.10, trunc_eps=1e-12,
        bond_dim=100, observables={"z": sigma_z}, observe_every=2)
    assert np.allclose(sparse.t, [0.04, 0.08, 0.10])
    assert np.allclose(sparse.expect["z"], dense.expect["z"][[1, 3, 4]])
    assert np.allclose(sparse.rdm[-1], dense.rdm[-1])
