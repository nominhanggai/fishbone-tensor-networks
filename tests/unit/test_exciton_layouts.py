"""Cross-validation of excitonic MPS and multi-set tree layouts."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from fishbonett._products import ScaledTreeIdentity
from fishbonett import (
    Bath,
    ExcitonBath,
    MultiSetTreeTensorNetwork,
    SimulationCheckpoint,
)
from fishbonett.models import registry
from fishbonett.operators import annihilate
from fishbonett.representations.exciton import ExcitonInteractionRepresentation


METHODS = tuple(
    name for name, spec in registry.METHODS.items()
    if "exciton-bath" in spec.models
)


def _load_fmo_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "fmo_state_layouts.py"
    spec = importlib.util.spec_from_file_location("test_fmo_state_layouts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fmo_methods_example():
    path = Path(__file__).resolve().parents[2] / "examples" / "fmo_mps_methods.py"
    spec = importlib.util.spec_from_file_location("test_fmo_mps_methods", path)
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
        for frequency, coupling in zip(
            branch.frequencies, branch.star_couplings, strict=True
        ):
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
    options = dict(
        dt=0.01,
        n_steps=4,
        method=method,
        trunc_eps=1e-12,
        bond_dim=100,
        krylov=30,
    )
    if "tdvp" in method:
        options["tol"] = 1e-11
    result = model.run(**options)
    exact = _exact_rdm(model, result.t)
    assert np.max(np.abs(result.rdm - exact)) < 1e-7
    assert np.allclose(
        result.expect["population"],
        np.diagonal(exact, axis1=1, axis2=2),
        atol=1e-7,
    )
    assert np.allclose(np.trace(result.rdm, axis1=1, axis2=2), 1.0)


def test_a1tdvp_qr_completion_is_deterministic_under_occupied_gauge():
    """The adaptive directions must not inherit occupied-column phases."""
    import fishbonett.evolve._tdvp_sweeps as sweeps

    rng = np.random.default_rng(19)
    matrix = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    phases = np.exp(1j * np.array([0.2, -1.1, 2.4]))
    first, triangular = sweeps._partial_full_qr(matrix, 7)
    gauged, gauged_triangular = sweeps._partial_full_qr(
        matrix * phases[None, :], 7
    )
    assert np.allclose(first.conj().T @ first, np.eye(7), atol=1e-13)
    assert np.allclose(first[:, :3] @ triangular, matrix, atol=1e-13)
    assert np.allclose(
        gauged[:, :3] @ gauged_triangular,
        matrix * phases[None, :],
        atol=1e-13,
    )
    assert np.allclose(first[:, 3:], gauged[:, 3:], atol=1e-13)


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


def test_tree_hopping_blocks_use_compact_identity_descriptors():
    model = _model(levels=3, modes=1)
    representation = ExcitonInteractionRepresentation(
        model.h, model.baths, 0.1, layout="system-first"
    )
    operators = representation.multiset_tree_operators(0.05)
    hopping = operators[0][1]
    assert isinstance(hopping, ScaledTreeIdentity)
    assert hopping.coefficient == model.h[0, 1]
    assert hopping.dimensions == representation.tree_dimensions
    assert hopping.edges == representation.tree_edges
    assert isinstance(operators[0][0], list)


@pytest.mark.parametrize("integrator", ["tebd", "trotter-mpo", "tdvp2"])
def test_interleaved_layout_preserves_a_general_one_excitation_initial_state(
    integrator,
):
    model = _model(levels=3, modes=1)
    initial = np.array([1.0, 2.0j, -0.5], complex)
    initial /= np.linalg.norm(initial)
    options = dict(
        dt=1e-5,
        n_steps=1,
        method=f"interaction-chain-interleaved-{integrator}",
        initial=initial,
        trunc_eps=1e-12,
        bond_dim=100,
    )
    if integrator == "tdvp2":
        options["tol"] = 1e-12
    result = model.run(**options)
    assert np.max(np.abs(result.rdm[0] - np.outer(initial, initial.conj()))) < 1e-5
    assert result.meta["electronic_sites"] == (0, 2, 4)


@pytest.mark.parametrize("layout", ["system-first", "interleaved"])
@pytest.mark.parametrize("integrator", ["tdvp1", "a1tdvp"])
def test_fixed_or_adaptive_one_site_tdvp_requires_a_bond_cap(layout, integrator):
    with pytest.raises(ValueError, match="bond_dim must be given"):
        _model(levels=2).run(
            dt=0.01,
            n_steps=1,
            method=f"interaction-chain-{layout}-{integrator}",
        )


@pytest.mark.parametrize("layout", ["system-first", "interleaved"])
@pytest.mark.parametrize(
    "integrator", ["tebd", "trotter-mpo", "tdvp1", "tdvp2", "a1tdvp"]
)
def test_conventional_mps_checkpoint_matches_uninterrupted_run(
    layout, integrator,
):
    model = _model(levels=2, modes=1)
    options = dict(
        dt=0.01,
        bath_horizon=0.04,
        method=f"interaction-chain-{layout}-{integrator}",
        trunc_eps=1e-12,
        bond_dim=20,
    )
    if "tdvp" in integrator:
        options["tol"] = 1e-11
    whole = model.run(n_steps=4, **options)
    first = model.run(n_steps=2, **options)
    continued = model.run(n_steps=2, resume=first.checkpoint, **options)

    assert continued.t.tolist() == pytest.approx([0.03, 0.04])
    assert continued.checkpoint.elapsed == pytest.approx(0.04)
    assert np.max(np.abs(continued.rdm - whole.rdm[2:])) < 1e-12


def test_conventional_mps_checkpoint_roundtrips_without_pickle(tmp_path):
    model = _model(levels=2, modes=1)
    options = dict(
        dt=0.01,
        bath_horizon=0.03,
        method="interaction-chain-interleaved-trotter-mpo",
        trunc_eps=1e-12,
        bond_dim=20,
    )
    first = model.run(n_steps=2, **options)
    loaded = SimulationCheckpoint.load(
        first.checkpoint.save(tmp_path / "exciton-mps.npz")
    )
    continued = model.run(n_steps=1, resume=loaded, **options)
    whole = model.run(n_steps=3, **options)
    assert np.max(np.abs(continued.rdm[-1] - whole.rdm[-1])) < 1e-12


def test_a1tdvp_resume_rejects_a_ceiling_below_the_saved_bond():
    model = _model(levels=2, modes=1)
    options = dict(
        dt=0.01,
        n_steps=2,
        bath_horizon=0.04,
        method="interaction-chain-system-first-a1tdvp",
        trunc_eps=1e-12,
        bond_dim=20,
    )
    first = model.run(initial=0, **options)
    assert first.max_bond[-1] > 1
    with pytest.raises(ValueError, match="exceeds the A1TDVP ceiling"):
        model.run(
            dt=0.01,
            n_steps=1,
            bath_horizon=0.04,
            method=first.method,
            trunc_eps=1e-12,
            bond_dim=1,
            resume=first.checkpoint,
        )


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


def test_fmo_propagator_summary_records_the_resolved_baths(tmp_path):
    example = _load_fmo_methods_example()
    summary = example.run_method(
        example.PROFILES["smoke"], "system-first-trotter-mpo", tmp_path
    )
    assert summary["state_family"] == "conventional-mps"
    assert summary["state_geometry"] == "system-first-mps"
    assert summary["svd_backend"] == "auto"
    assert summary["factorization_backend"] == "adaptive-svd:auto"
    assert summary["bath_modes_per_level"] == [1] * 7

    adaptive = example.run_method(
        example.PROFILES["smoke"], "system-first-a1tdvp", tmp_path
    )
    assert adaptive["svd_backend"] is None
    assert adaptive["factorization_backend"] == "deterministic-qr-completion"

    fixed = example.run_method(
        example.PROFILES["smoke"], "system-first-tdvp1", tmp_path
    )
    assert fixed["svd_backend"] is None
    assert fixed["factorization_backend"] == "reduced-qr"
