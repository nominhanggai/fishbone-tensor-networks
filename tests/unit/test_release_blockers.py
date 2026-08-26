"""Regression tests for scientific contracts required by a public release."""

import inspect
from pathlib import Path

import fishbonett
import numpy as np
import pytest

from fishbonett import Bath
from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.lanczos import block_lanczos
from fishbonett.diabatization import diabatize
from fishbonett.evolve import _tdvp_driver, _tdvp_sweeps
from fishbonett.models.propagate import resolve_time_grid
from fishbonett.models.result import SimulationCheckpoint
from fishbonett.models.simulation import _expect_from_rdm
from fishbonett.operators import sigma_x, sigma_y, sigma_z
from fishbonett.representations.polaron import PolaronRepresentation
from fishbonett.representations.multichannel import (
    MultichannelInteractionRepresentation,
)
from fishbonett.spectral_densities import brownian
from fishbonett.states.mps import SystemBathMPS
from fishbonett.system import System


def test_suite_imports_the_current_checkout():
    expected = Path(__file__).resolve().parents[2] / "src" / "fishbonett"
    imported = Path(fishbonett.__file__).resolve().parent
    assert imported == expected


def test_opt_einsum_is_a_core_dependency():
    metadata = (
        Path(__file__).resolve().parents[2] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    project, optional = metadata.split("[project.optional-dependencies]", 1)
    assert '"opt_einsum>=3.3"' in project
    assert "\nspeed =" not in optional


def test_local_job_artifacts_are_ignored_and_excluded_from_sdists():
    root = Path(__file__).resolve().parents[2]
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    metadata = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "/.codex-jobs/" in ignore
    assert '"/.codex-jobs"' in metadata


def test_complex_multichannel_local_hamiltonian_is_hermitian():
    representation = MultichannelInteractionRepresentation.from_signed_star(
        [2, 3], [sigma_z + sigma_y], [1.0], h_sys=np.zeros((2, 2)),
        representation="interaction-chain",
    ).build()

    hamiltonian, _, _ = representation.two_site_hamiltonians(0.2, 0.1)[0]
    dense = hamiltonian.toarray()
    np.testing.assert_allclose(dense, dense.conjugate().T, atol=1e-14)


def test_brownian_density_has_positive_reference_denominator():
    value = brownian(2.0, lam=1.0, gam=0.1, w0=1.0)
    expected = 2.0 * 0.1 * 2.0 / ((1.0 - 4.0) ** 2 + 0.1**2 * 4.0)
    assert value == pytest.approx(expected)
    assert value > 0.0


def test_ordinary_continuum_bath_reports_reorganization_energy():
    bath = Bath(J=lambda w: w, domain=(0.0, 1.0), n_modes=2)
    assert bath.reorganization_energy() == pytest.approx(1.0 / np.pi)


def test_legendre_discretization_accepts_one_mode():
    frequency, coupling_squared = get_vn_squared(
        lambda w: 2.0, 1, [0.0, 4.0]
    )
    np.testing.assert_allclose(frequency, [2.0])
    np.testing.assert_allclose(coupling_squared, [8.0])


def test_multichannel_auto_mode_count_is_permutation_invariant(monkeypatch):
    def fake_auto_n_modes(density, _domain, _t_max, **_kwargs):
        return int(density(1.0))

    monkeypatch.setattr(
        "fishbonett.bath.auto.auto_n_modes", fake_auto_n_modes
    )
    first = lambda _w: 3.0
    second = lambda _w: 7.0
    a = Bath(J=[first, second], domain=(0.0, 2.0)).resolved(1.0)
    b = Bath(J=[second, first], domain=(0.0, 2.0)).resolved(1.0)
    assert a.n_modes == b.n_modes == 7


def test_mps_tebd_uses_a_relative_singular_value_threshold():
    state = SystemBathMPS([2, 2])
    theta = np.zeros((1, 2, 2, 1), complex)
    theta[0, 0, 0, 0] = 0.1
    theta[0, 1, 1, 0] = 5e-5

    state._split_cpu(theta, 0, 10, 1e-4, None)
    assert len(state.S[1]) == 2


def test_tdvp2_does_not_invent_an_unrequested_bond_cap(monkeypatch):
    received = []

    def fake_sweep(_dt, state, _operator, chi_max, _eps,
                   environments=None, **_kwargs):
        received.append(chi_max)
        return state, environments

    class Representation:
        static = True
        dimensions = (2, 2)

        @staticmethod
        def tdvp_mpo(_time):
            return None

    monkeypatch.setattr(_tdvp_driver, "tdvp2sweep", fake_sweep)
    _tdvp_driver.run_mpo_hamiltonian(
        Representation(), dt=0.1, nsteps=1, sweep="tdvp2",
        bond_dim=None, observe=lambda _state: np.eye(2),
    )
    assert received == [None]


def test_dynamic_tdvp_exposes_only_the_common_truncation_controls():
    driver_parameters = inspect.signature(
        _tdvp_driver.run_mpo_hamiltonian
    ).parameters
    sweep_parameters = inspect.signature(
        _tdvp_sweeps.a1tdvp_sweep
    ).parameters
    assert "trunc_eps" in driver_parameters
    assert "bond_dim" in driver_parameters
    assert "trunc_eps" in sweep_parameters
    assert "bond_dim" in sweep_parameters
    assert {"prec", "Afull", "FRs", "Dlim", "Dplusmax"}.isdisjoint(
        driver_parameters | sweep_parameters
    )


@pytest.mark.parametrize("kwargs", [
    {"eps": np.nan}, {"eps": np.inf}, {"eps": True},
    {"max_bond": 1.5}, {"max_bond": True},
])
def test_truncation_rejects_nonfinite_or_nonintegral_settings(kwargs):
    from fishbonett import Truncation

    with pytest.raises(ValueError):
        Truncation(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"dt": 0.0, "n_steps": 1},
    {"dt": np.nan, "n_steps": 1},
    {"dt": 0.1, "n_steps": 1.5},
    {"dt": 0.1, "n_steps": True},
    {"dt": 0.1, "t_max": 0.25},
])
def test_time_grid_rejects_invalid_or_silently_rounded_input(kwargs):
    with pytest.raises(ValueError):
        resolve_time_grid(**kwargs)


def test_system_rejects_nonfinite_operators_and_zero_initial_state():
    with pytest.raises(ValueError, match="finite"):
        System([[np.nan]], [[1.0]])
    system = System(sigma_x, sigma_z)
    with pytest.raises(ValueError, match="non-zero"):
        system.initial_vector([0.0, 0.0])


def test_list_matrix_is_one_coupling_not_two_channels():
    system = System(sigma_x.tolist(), sigma_z.tolist())
    assert not system.is_multichannel
    assert system.coupling.shape == (2, 2)

    coupled = Bath.vibronic([1.0], [0.1]).bind(sigma_z.tolist())
    assert not coupled.is_multichannel
    assert coupled.operator.shape == (2, 2)


@pytest.mark.parametrize("representation", ["polaron-star", "polaron-chain"])
def test_polaron_uses_discrete_vibronic_modes_and_exact_counterterm(
        representation):
    bath = Bath.vibronic([2.0], [0.25], phys_dim=4).resolved(1.0)
    built = PolaronRepresentation(
        representation=representation, h_sys=sigma_x,
        coupling=sigma_z, bath=bath,
    ).build()
    assert built.reorganization_energy == pytest.approx(0.5)
    assert built.displacements[0] == pytest.approx(0.5)


def test_polaron_rejects_an_ohmic_continuum_touching_zero():
    bath = Bath(
        J=lambda w: max(float(w), 0.0) * np.exp(-max(float(w), 0.0)),
        domain=(0.0, 10.0), n_modes=4, phys_dim=3,
    )
    with pytest.raises(ValueError, match=r"J\(w\)/w\*\*2"):
        PolaronRepresentation(
            representation="polaron-chain", h_sys=sigma_x,
            coupling=sigma_z, bath=bath,
        ).build()


def test_complex_block_lanczos_preserves_orthogonality_and_spectrum():
    rng = np.random.default_rng(4)
    seed, _ = np.linalg.qr(
        rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    )
    hamiltonian = np.diag([0.5, 1.3, 2.7, 4.2]).astype(complex)
    projected, transform = block_lanczos(hamiltonian, seed)
    np.testing.assert_allclose(
        transform.conj().T @ transform, np.eye(4), atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(projected), np.linalg.eigvalsh(hamiltonian),
        atol=1e-12,
    )


def test_degenerate_boys_localization_is_well_defined():
    dipoles = np.zeros((2, 2, 3))
    transform, localized = diabatize(dipoles)
    np.testing.assert_array_equal(transform, np.eye(2))
    np.testing.assert_array_equal(localized, dipoles)


def test_bath_validates_its_declarative_contract_eagerly():
    with pytest.raises(TypeError, match="J"):
        Bath(J=None)
    with pytest.raises(ValueError, match="domain"):
        Bath(J=lambda w: w, domain=(1.0, 0.0))
    with pytest.raises(ValueError, match="discretization"):
        Bath(J=lambda w: w, discretization="unknown")
    with pytest.raises(ValueError, match="equal length"):
        Bath(
            J=lambda w: w, discrete_frequencies=(1.0,),
            discrete_couplings=(),
        )


def test_coupled_bath_resolution_does_not_cache_mutable_specification(monkeypatch):
    bath = Bath(J=lambda w: w, domain=(0.0, 1.0))
    coupled = bath.bind(sigma_z)
    counts = iter((3, 7))
    monkeypatch.setattr(
        "fishbonett.bath.auto.auto_n_modes",
        lambda *_args, **_kwargs: next(counts),
    )
    assert coupled.resolved(1.0).bath.n_modes == 3
    assert coupled.resolved(1.0).bath.n_modes == 7


def test_nonhermitian_observable_expectations_keep_their_complex_part():
    rho = np.array([[0.5, -0.5j], [0.5j, 0.5]])
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]])
    value = _expect_from_rdm([rho], {"lowering": lowering})["lowering"][0]
    assert value == pytest.approx(0.5j)


def test_fishbone_graph_couplings_are_not_dropped_by_local_terms():
    from fishbonett import Fishbone

    exchange = np.zeros((4, 4), complex)
    exchange[1, 2] = 1j
    exchange[2, 1] = -1j
    model = Fishbone(
        sites=[np.zeros((2, 2)), np.zeros((2, 2))],
        baths=[None, None], couplings={(0, 1): exchange},
    )
    terms = model.local_terms()
    np.testing.assert_array_equal(terms.graph_bond[(0, 1)], exchange)
    with pytest.raises(ValueError, match="graph_bond"):
        model.hamiltonians()


@pytest.mark.parametrize("bad_tensor", [
    np.ones((1, 3)),
    np.array([[np.nan, 0.0]]),
])
def test_checkpoint_restore_rejects_corrupt_tensor_data(bad_tensor):
    checkpoint = SimulationCheckpoint(
        tensors=(bad_tensor, np.ones((1, 2))), dims=(2, 2),
        edges=((0, 1),), oc=0, method="test", elapsed=0.0,
        bath_horizon=1.0, signature="signature",
    )
    with pytest.raises(ValueError, match="tensor 0"):
        checkpoint.restore_tree((2, 2), ((0, 1),), "signature")
