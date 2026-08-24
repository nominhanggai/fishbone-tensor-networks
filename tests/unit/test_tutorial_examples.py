"""Contract tests for the profiled scientific tutorials."""

import importlib.util
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = (
    "vibronic_dimer",
    "nonadiabatic_spin_boson",
    "bridge_electron_transfer",
    "two_bath_heat_flow",
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"test_example_{name}", ROOT / "examples" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_profiles_have_fast_default_and_automatic_resolved_runs(name):
    module = _load(name)
    assert tuple(module.PROFILES) == ("smoke", "docs", "reference")
    smoke = module.PROFILES["smoke"]
    docs = module.PROFILES["docs"]
    reference = module.PROFILES["reference"]
    smoke_horizon = getattr(smoke, "t_max", getattr(smoke, "t_max_ps", None))
    smoke_step = getattr(smoke, "dt", getattr(smoke, "dt_ps", None))
    assert round(smoke_horizon / smoke_step) == 4
    assert docs.n_modes is None or docs.n_modes >= 12
    assert reference.n_modes is None


@pytest.mark.parametrize("name", TUTORIALS)
def test_tutorial_pages_are_self_contained(name):
    """A reader must not need to reverse-engineer the corresponding example."""
    page = (ROOT / "docs" / "tutorials" / f"{name}.md").read_text(
        encoding="utf-8"
    )
    programs = re.findall(r"```python\n(.*?)```", page, flags=re.DOTALL)
    assert "Complete runnable" in page
    assert "Common mistakes" in page
    assert programs
    program = max(programs, key=len)
    assert "from fishbonett import" in program
    assert ".run(" in program
    assert "observables=" in program
    assert ".expect[" in program
    assert "matplotlib" in program


def test_vibronic_dimer_smoke_conserves_the_excitation():
    module = _load("vibronic_dimer")
    summary = module.summarize(module.run_profile("smoke"))
    assert summary["normalization_error"] < 1e-10
    assert summary["resolved_modes"] == {8.0: (4, 4)}


def test_nonadiabatic_smoke_population_is_physical():
    module = _load("nonadiabatic_spin_boson")
    suite = module.run_profile("smoke")
    population = suite["results"]["interaction"].expect["population_up"]
    assert np.all(np.asarray(population) >= -1e-10)
    assert np.all(np.asarray(population) <= 1.0 + 1e-10)


def test_electron_transfer_units_and_smoke_populations():
    module = _load("bridge_electron_transfer")
    expected_conversion = 0.1883651567308853
    assert module.CM_TO_RAD_PS == pytest.approx(expected_conversion)
    assert module.BATH_ALPHA == pytest.approx(1.67)
    assert module.BATH_CUTOFF_CM == pytest.approx(600.0)
    reorganization = 0.5 * module.BATH_ALPHA * module.BATH_CUTOFF_CM
    assert reorganization == pytest.approx(501.0)
    assert module.REORGANIZATION_CM == pytest.approx(reorganization)
    assert module.PROFILES["docs"].variants == (("primary", 6, 1e-4),)
    weak_h, weak_coupling = module._case("weak_diagonal")
    noncondon_h, noncondon_coupling = module._case("noncondon")
    assert np.array_equal(weak_h, noncondon_h)
    assert not np.array_equal(weak_coupling, noncondon_coupling)
    propagated_h, propagated_coupling = module.quapi_equivalent_hamiltonian(
        "noncondon"
    )
    expected_renormalization = (
        module.CM_TO_RAD_PS * reorganization
        * (propagated_coupling @ propagated_coupling)
    )
    assert np.allclose(
        propagated_h, noncondon_h + expected_renormalization
    )
    # The QUAPI calculation is performed after diagonalizing M.  Applying the
    # local lambda D^2 term there and rotating back must give lambda M^2; taking
    # only the diagonal of M^2 would depend on the chosen electronic basis.
    eigenvalues, eigenvectors = np.linalg.eigh(propagated_coupling)
    rotated_renormalization = (
        module.CM_TO_RAD_PS * reorganization
        * eigenvectors @ np.diag(eigenvalues ** 2) @ eigenvectors.conj().T
    )
    assert np.allclose(rotated_renormalization, expected_renormalization)
    assert abs(expected_renormalization[0, 1]) > 0.0
    suite = module.run_profile("smoke")
    assert suite["bath"]["alpha"] == pytest.approx(1.67)
    assert suite["bath"]["cutoff_cm"] == pytest.approx(600.0)
    assert suite["bath"]["n_modes"] == 4
    summary = module.summarize(suite)
    assert tuple(summary) == (
        "diagonal_reference", "weak_diagonal", "noncondon"
    )
    for case in summary:
        assert summary[case]["normalization_error"] < 1e-10
        assert 0.0 <= summary[case]["final_acceptor_population"] <= 1.0
        assert np.isnan(summary[case]["effective_lifetime_ps"])


def test_electron_transfer_reference_maps_reproduce_paper_dynamics():
    module = _load("bridge_electron_transfer")
    validation = module.long_validation()
    assert validation["metadata"] == {
        "memory_ps": pytest.approx(0.15),
        "trunc_eps": pytest.approx(1e-4),
        "bath_alpha": pytest.approx(1.67),
        "bath_cutoff_cm": pytest.approx(600.0),
        "temperature_k": pytest.approx(300.0),
        "dt_ps": pytest.approx(0.002),
        "phys_dim": 6,
        "bath_n_modes": 95,
        "bath_domain_cm": pytest.approx(
            (-870.7656455500296, 4312.930476458639)
        ),
        "method": "interaction-chain-trotter-mpo",
    }
    expected_lifetimes = {
        "diagonal_reference": 2.36,
        "noncondon": 2.50,
    }
    for case, expected_lifetime in expected_lifetimes.items():
        result = validation["results"][case]
        summary = validation["summary"][case]
        assert result["populations"].shape == (7500, 3)
        assert summary["population_rmse"] < 0.006
        assert summary["max_population_error"] < 0.012
        assert summary["last_transfer_norm"] < 1.6e-4
        assert summary["heldout_population_error"] < 1e-4
        assert summary["direct_map_trace_error"] < 1e-12
        assert summary["direct_map_minimum_choi_eigenvalue"] > -3e-5
        assert summary["trace_error"] < 2e-11
        assert summary["minimum_eigenvalue"] > -1e-10
        assert summary["lifetime_ps"] == pytest.approx(
            expected_lifetime, rel=0.03
        )
        assert summary["paper_curve_lifetime_ps"] == pytest.approx(
            expected_lifetime, rel=0.01
        )
        convergence = result["memory_convergence"]
        assert convergence["cutoff_ps"][0] == pytest.approx(0.04)
        assert convergence["cutoff_ps"][-1] == pytest.approx(0.15)
        assert convergence["max_population_difference"][-1] == pytest.approx(0.0)
        at_012 = np.flatnonzero(np.isclose(
            convergence["cutoff_ps"], 0.12
        ))[0]
        assert convergence["max_population_difference"][at_012] < 0.002
        assert abs(
            convergence["donor_lifetime_ps"][at_012]
            - convergence["donor_lifetime_ps"][-1]
        ) < 0.02


def test_electron_transfer_tomography_spans_liouville_space():
    module = _load("bridge_electron_transfer")
    states = module.tomography_states(3)
    assert tuple(states) == (
        "d0", "d1", "d2", "r01", "i01", "r02", "i02", "r12", "i12"
    )
    identity_runs = {
        label: np.repeat(
            (state[:, None] @ state.conj()[None, :])[None, :, :], 2, axis=0
        )
        for label, state in states.items()
    }
    maps = module.assemble_dynamical_maps(identity_runs)
    np.testing.assert_allclose(
        maps, np.repeat(np.eye(9)[None, :, :], 2, axis=0), atol=2e-16
    )


def test_heat_flow_smoke_uses_distinct_baths_and_obeys_continuity():
    module = _load("two_bath_heat_flow")
    suite = module.run_profile("smoke")
    case = suite["results"]["nonequilibrium"]["primary"]
    result = case["result"]
    branches = result.meta["bath_branches"]
    assert [(item["system_site"], item["bath"]) for item in branches] == [
        (0, 0), (0, 1),
    ]
    assert result.meta["observable_targets"]["hot_system_mode"] != (
        result.meta["observable_targets"]["cold_system_mode"]
    )
    summary = module.summarize(suite)
    assert summary["nonequilibrium"]["continuity_rms"] < 1e-3
