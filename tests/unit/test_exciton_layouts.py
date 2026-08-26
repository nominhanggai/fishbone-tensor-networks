"""Cross-validation of excitonic MPS and multi-set tree layouts."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from fishbonett import Bath, ExcitonBath, MultiSetTreeTensorNetwork
from fishbonett.operators import annihilate
from fishbonett.representations.exciton import ExcitonInteractionRepresentation


METHODS = (
    "interaction-chain-system-first-tdvp2",
    "interaction-chain-interleaved-tdvp2",
    "interaction-chain-multi-set-tdvp2",
    "interaction-chain-multi-set-tree-tdvp2",
)


def _load_fmo_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "fmo_state_layouts.py"
    spec = importlib.util.spec_from_file_location("test_fmo_state_layouts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _density(frequency):
    return 0.08 * frequency**2 * np.exp(-frequency) if frequency >= 0 else 0.0


def _model(levels=3, modes=1):
    hamiltonian = np.array(
        [
            [0.2, 0.3, -0.1],
            [0.3, -0.1, 0.17],
            [-0.1, 0.17, 0.4],
        ],
        complex,
    )[:levels, :levels]
    baths = [Bath(J=_density, domain=(0.2, 3.0), n_modes=modes, phys_dim=3) for _ in range(levels)]
    return ExcitonBath(hamiltonian, baths)


def _exact_rdm(model, times):
    representation = ExcitonInteractionRepresentation(
        model.h, model.baths, max(times), layout="system-first"
    )
    local_dimension = representation.branches[0][1].pd_boson[0]
    mode_count = sum(branch.len_boson for _level, branch in representation.branches)
    dimensions = (model.n_levels,) + (local_dimension,) * mode_count

    def embed(operator, site):
        value = np.ones((1, 1), complex)
        for position, dimension in enumerate(dimensions):
            local = operator if position == site else np.eye(dimension)
            value = np.kron(value, local)
        return value

    destroy = annihilate(local_dimension)
    hamiltonian = embed(model.h, 0)
    mode_site = 1
    for level, branch in representation.branches:
        projector = np.zeros((model.n_levels, model.n_levels), complex)
        projector[level, level] = 1.0
        for frequency, coupling in zip(branch.frequencies, branch.star_couplings):
            hamiltonian += frequency * embed(destroy.conj().T @ destroy, mode_site)
            hamiltonian += coupling * (
                embed(projector, 0) @ embed(destroy + destroy.conj().T, mode_site)
            )
            mode_site += 1
    energies, vectors = np.linalg.eigh(hamiltonian)
    initial = np.zeros(int(np.prod(dimensions)), complex)
    initial[0] = 1.0
    coefficients = vectors.conj().T @ initial
    result = []
    for time in times:
        wavefunction = (vectors @ (np.exp(-1j * energies * time) * coefficients)).reshape(
            model.n_levels, -1
        )
        result.append(wavefunction @ wavefunction.conj().T)
    return np.asarray(result)


@pytest.mark.parametrize("method", METHODS)
def test_every_exciton_layout_matches_the_same_dense_finite_bath(method):
    model = _model(levels=3, modes=1)
    result = model.run(
        dt=0.01,
        n_steps=4,
        method=method,
        trunc_eps=1e-12,
        bond_dim=100,
        krylov=30,
        tol=1e-11,
    )
    exact = _exact_rdm(model, result.t)
    assert np.max(np.abs(result.rdm - exact)) < 1e-8
    assert np.allclose(result.expect["population"], np.diagonal(exact, axis1=1, axis2=2))
    assert np.allclose(np.trace(result.rdm, axis1=1, axis2=2), 1.0)


def test_two_mode_bath_tree_and_flat_mps_agree():
    model = _model(levels=2, modes=2)
    common = dict(
        dt=0.01,
        n_steps=3,
        trunc_eps=1e-11,
        bond_dim=100,
        krylov=30,
        tol=1e-11,
    )
    flat = model.run(method="interaction-chain-system-first-tdvp2", **common)
    tree = model.run(method="interaction-chain-multi-set-tree-tdvp2", **common)
    assert np.max(np.abs(tree.rdm - flat.rdm)) < 1e-9
    assert tree.meta["tree_dimensions"] == (1, 1, 3, 3, 3, 3)
    assert len(tree.meta["tree_edges"]) == 5


def test_interleaved_layout_preserves_a_general_one_excitation_initial_state():
    model = _model(levels=3, modes=1)
    initial = np.array([1.0, 2.0j, -0.5], complex)
    initial /= np.linalg.norm(initial)
    result = model.run(
        dt=1e-5,
        n_steps=1,
        method="interaction-chain-interleaved-tdvp2",
        initial=initial,
        trunc_eps=1e-12,
        bond_dim=100,
        tol=1e-12,
    )
    assert np.max(np.abs(result.rdm[0] - np.outer(initial, initial.conj()))) < 1e-5
    assert result.meta["electronic_sites"] == (0, 2, 4)


def test_multiset_tree_progress_exposes_the_actual_state():
    updates = []
    result = _model(levels=2, modes=1).run(
        dt=0.01,
        n_steps=2,
        method="interaction-chain-multi-set-tree-tdvp2",
        trunc_eps=1e-10,
        progress=updates.append,
    )
    assert result.method == "interaction-chain-multi-set-tree-tdvp2"
    assert len(updates) == 2
    assert isinstance(updates[-1]["state"], MultiSetTreeTensorNetwork)
    assert set(updates[-1]) == {"step", "n_steps", "t", "bond", "rdm", "state"}


def test_exciton_model_validates_baths_and_initial_level():
    with pytest.raises(ValueError, match="one entry per electronic level"):
        ExcitonBath(np.eye(2), [Bath(J=_density, domain=(0.2, 3), n_modes=1)])
    model = _model(levels=2)
    with pytest.raises(ValueError, match="initial level"):
        model.initial_vector(2)


def test_fmo_example_smoke_compares_all_mps_layouts():
    example = _load_fmo_example()
    arrays, summary = example.run(
        example.PROFILES["smoke"], ["system-first", "interleaved", "multi-set"]
    )
    assert arrays["t"].tolist() == [0.02]
    assert summary["interleaved"]["maximum_population_difference"] < 1e-7
    assert summary["multi-set"]["maximum_population_difference"] < 1e-7
