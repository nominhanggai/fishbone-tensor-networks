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
    propagated_h, propagated_coupling = module.propagation_hamiltonian(
        "noncondon"
    )
    expected_counterterm = (
        module.CM_TO_RAD_PS * reorganization
        * (propagated_coupling @ propagated_coupling)
    )
    assert np.allclose(propagated_h, noncondon_h + expected_counterterm)
    # M^2 is not diagonal for a non-Condon coupling matrix.  Retaining its
    # off-diagonal entries is essential to reproduce the same Hamiltonian.
    assert abs(expected_counterterm[0, 1]) > 0.0
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
