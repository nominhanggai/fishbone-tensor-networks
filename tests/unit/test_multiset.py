"""Multi-set MPS state and coupled-TDVP validation."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from fishbonett import Bath, MultiSetMPS, SystemBath
from fishbonett.evolve.multiset import split_system_mpo
from fishbonett.operators import annihilate, sigma_x, sigma_z
from fishbonett.representations.interaction import InteractionRepresentation


REPRESENTATIONS = (
    "schrodinger-chain",
    "schrodinger-star",
    "interaction-chain",
    "polaron-chain",
    "polaron-star",
)


def _load_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "multiset_holstein.py"
    spec = importlib.util.spec_from_file_location("test_multiset_holstein", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _density(frequency):
    return 0.2 * frequency**3 * np.exp(-frequency / 5.0)


def _bath(n_modes=2):
    return Bath(J=_density, domain=(0.4, 10.0), n_modes=n_modes, phys_dim=4)


def _exact_dynamics(times, n_modes=2):
    h_system = 0.5 * sigma_x + 0.15 * sigma_z
    coupling = sigma_z
    bath = _bath(n_modes)
    representation = InteractionRepresentation(
        representation="interaction-chain",
        h_sys=h_system,
        coupling=coupling,
        bath=bath,
    ).build()
    dimension = bath.phys_dim
    dimensions = (2,) + (dimension,) * n_modes

    def embed(operator, site):
        value = np.ones((1, 1), complex)
        for index, local_dimension in enumerate(dimensions):
            local = operator if index == site else np.eye(local_dimension)
            value = np.kron(value, local)
        return value

    destroy = annihilate(dimension)
    hamiltonian = embed(h_system, 0)
    for mode, (frequency, strength) in enumerate(
        zip(representation.frequencies, representation.star_couplings), start=1
    ):
        hamiltonian += frequency * embed(destroy.conj().T @ destroy, mode)
        hamiltonian += strength * (embed(coupling, 0) @ embed(destroy + destroy.conj().T, mode))
    energies, vectors = np.linalg.eigh(hamiltonian)
    initial = np.zeros(np.prod(dimensions), complex)
    initial[0] = 1.0
    coefficients = vectors.conj().T @ initial
    values = []
    for time in times:
        wavefunction = (vectors @ (np.exp(-1j * energies * time) * coefficients)).reshape(2, -1)
        rho = wavefunction @ wavefunction.conj().T
        values.append(rho)
    return np.asarray(values)


def test_multiset_product_rdm_and_conventional_embedding_agree():
    state = MultiSetMPS.product([3.0, 4.0j], [3, 2])
    expected = np.outer(
        np.array([3.0, 4.0j]) / 5.0,
        np.array([3.0, 4.0j]).conj() / 5.0,
    )
    assert np.allclose(state.system_rdm(), expected)

    full = state.combined_mps()
    split = MultiSetMPS.from_full_mps(full)
    assert np.allclose(split.system_rdm(), expected)
    assert split.peak_bond() == 2  # the exact embedding retains its set blocks


def test_split_system_mpo_shares_the_immutable_operator_tail():
    mpo = [
        np.ones((1, 2, 2, 2), complex),
        np.ones((2, 2, 3, 3), complex),
        np.ones((2, 1, 3, 3), complex),
    ]
    blocks = split_system_mpo(mpo)
    assert blocks[0][0][0] is not mpo[1]
    assert blocks[0][0][1] is mpo[2]
    assert blocks[1][1][1] is mpo[2]


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_every_system_bath_representation_runs_on_multiset_mps(representation):
    model = SystemBath(
        h=0.5 * sigma_x + 0.15 * sigma_z,
        coupling=sigma_z,
        bath=_bath(),
    )
    result = model.run(
        dt=0.01,
        n_steps=10,
        representation=representation,
        state_geometry="multi-set-mps",
        integrator="tdvp2",
        trunc_eps=1e-12,
        bond_dim=30,
        krylov=30,
        tol=1e-11,
        observables={"sz": sigma_z},
    )
    exact = _exact_dynamics(result.t)
    assert result.method == f"{representation}-multi-set-tdvp2"
    assert result.meta["n_sets"] == 2
    assert result.meta["state_geometry"] == "multi-set-mps"
    # Polaron-chain convergence also contains the finite-Fock displacement
    # error; all five representations remain within the directly checked scale.
    assert np.max(np.abs(result.rdm - exact)) < 4e-4
    assert np.allclose(result.expect["sz"], np.einsum("tij,ji->t", result.rdm, sigma_z).real)


def test_multiset_targeted_bath_observable_and_progress_contract():
    from fishbonett import BathMode

    updates = []
    number = np.diag(np.arange(4.0))
    result = SystemBath(
        h=0.5 * sigma_x,
        coupling=sigma_z,
        bath=_bath(),
    ).run(
        dt=0.01,
        n_steps=2,
        method="interaction-chain-multi-set-tdvp2",
        trunc_eps=1e-10,
        bond_dim=20,
        observables={"n0": (number, BathMode(0, 0, 0))},
        progress=updates.append,
    )
    assert result.expect["n0"].shape == (2,)
    assert np.all(result.expect["n0"] >= -1e-12)
    assert len(updates) == 2
    assert set(updates[-1]) == {"step", "n_steps", "t", "bond", "rdm", "state"}
    assert isinstance(updates[-1]["state"], MultiSetMPS)


def test_one_mode_bath_uses_complete_one_site_multiset_update():
    model = SystemBath(
        h=0.5 * sigma_x + 0.15 * sigma_z,
        coupling=sigma_z,
        bath=_bath(n_modes=1),
    )
    result = model.run(
        dt=0.01,
        n_steps=5,
        method="schrodinger-star-multi-set-tdvp2",
        trunc_eps=1e-12,
        bond_dim=20,
        krylov=30,
        tol=1e-11,
    )
    assert np.max(np.abs(result.rdm - _exact_dynamics(result.t, 1))) < 1e-10


def test_holstein_example_compares_the_same_dynamics():
    module = _load_example()
    arrays, summary = module.run(module.PROFILES["smoke"])
    assert np.array_equal(arrays["t"], np.array([0.04, 0.08]))
    assert summary["maximum_population_difference"] < 1e-5
    assert summary["multiset_peak_bond"] <= summary["conventional_peak_bond"]
