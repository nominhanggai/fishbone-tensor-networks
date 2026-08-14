"""Architectural contracts for prepared single-system simulations."""
from types import SimpleNamespace

import numpy as np
import pytest

from fishbonett.models.propagate import RunCtx
from fishbonett.models.result import Result
from fishbonett.models.simulation import SimulationPlan


def _spec():
    return SimpleNamespace(name="test-plan")


def test_step_plan_exposes_step_measure_and_cost_policies():
    count = [0]
    rho = np.diag([1.0, 0.0])

    def step(_index):
        count[0] += 1

    context = RunCtx(
        dt=0.1, n_steps=3, obs_ops={"population": rho})
    plan = SimulationPlan(
        _spec(), context, step=step, measure_rdm=lambda: rho,
        peak_bond=lambda: count[0] + 1)

    assert plan.is_step_based
    result = plan.run()
    assert count == [3]
    assert np.allclose(result.t, [0.1, 0.2, 0.3])
    assert np.allclose(result.expect["population"], 1.0)
    assert np.array_equal(result.max_bond, [2, 3, 4])


def test_whole_run_plan_wraps_a_native_driver():
    expected = Result(
        t=np.array([0.1]), expect={}, max_bond=np.array([1]),
        rdm=np.eye(2)[None], method="test-plan")
    plan = SimulationPlan(
        _spec(), RunCtx(dt=0.1, n_steps=1), execute=lambda: expected)

    assert not plan.is_step_based
    assert plan.run() is expected


@pytest.mark.parametrize("kwargs", [
    {},
    {"step": lambda _k: None},
    {
        "step": lambda _k: None,
        "measure_rdm": lambda: np.eye(2),
        "peak_bond": lambda: 1,
        "execute": lambda: None,
    },
])
def test_plan_requires_exactly_one_execution_form(kwargs):
    with pytest.raises(ValueError, match="simulation plan"):
        SimulationPlan(_spec(), RunCtx(dt=0.1, n_steps=1), **kwargs)


def test_physical_model_does_not_import_representation_or_evolution_engines():
    import inspect
    from fishbonett.models.system_bath import SystemBath

    source = inspect.getsource(inspect.getmodule(SystemBath))
    assert "fishbonett.evolve" not in source
    assert "fishbonett.representations" not in source
    assert not hasattr(SystemBath, "_DRIVERS")
    assert not hasattr(SystemBath, "_MPO_REPRESENTATIONS")
    assert not hasattr(SystemBath, "_SWAP_REPRESENTATIONS")


def test_representations_own_their_numerical_products():
    from fishbonett.representations.interaction import InteractionRepresentation
    from fishbonett.representations.polaron import PolaronRepresentation
    from fishbonett.representations.schrodinger import (
        LocalTerms, SchrodingerRepresentation,
    )

    assert hasattr(SchrodingerRepresentation, "tdvp_mpo")
    assert hasattr(InteractionRepresentation, "tdvp_mpo")
    assert hasattr(InteractionRepresentation, "tebd_gates")
    assert hasattr(InteractionRepresentation, "trotter_mpo")
    assert hasattr(PolaronRepresentation, "tdvp_mpo")
    assert hasattr(PolaronRepresentation, "tebd_gates")
    terms = LocalTerms(
        dims=[2], edges=[], site=[np.zeros((2, 2))], bond={})
    site_gates, edge_gates = terms.tebd_gates(0.1)
    assert site_gates == [None]
    assert edge_gates == {}


def test_multisite_models_compile_through_simulation_plan(monkeypatch):
    from fishbonett import Bath, Fishbone, TreeFishbone
    from fishbonett.models import simulation
    from fishbonett.operators import sigma_x, sigma_z

    seen = []
    real = simulation.compile_plan

    def record(model, spec, context):
        seen.append((type(model).__name__, spec.engine))
        return real(model, spec, context)

    monkeypatch.setattr(simulation, "compile_plan", record)
    bath = Bath(
        J=lambda w: 0.2 * w * np.exp(-w / 5.0),
        domain=(0.0, 20.0), n_modes=2, phys_dim=3)
    TreeFishbone(
        sites=[sigma_x], edges=[], baths=[bath.bind(sigma_z)]).run(
            dt=0.01, n_steps=1)
    Fishbone(sites=[sigma_x], baths=[bath.bind(sigma_z)]).run(
        dt=0.01, n_steps=1)

    assert seen == [
        ("TreeFishbone", "static-tree-tebd"),
        ("Fishbone", "static-tree-tebd"),
    ]


def test_run_seed_is_reproducible_and_does_not_touch_global_rng():
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    def run(seed):
        model = SystemBath(
            h=0.5 * sigma_x, coupling=sigma_z,
            bath=Bath(
                J=lambda w: 0.2 * w * np.exp(-w / 5.0),
                domain=(0.0, 30.0), n_modes=3, phys_dim=4))
        return model.run(
            dt=0.05, n_steps=3, method="schrodinger-chain-tdvp1", bond_dim=4,
            observables={"sz": sigma_z}, seed=seed)

    np.random.seed(123)
    expected_global = np.random.random(3)
    np.random.seed(123)
    first = run(7)
    second = run(7)
    different = run(8)

    np.testing.assert_array_equal(first.rdm, second.rdm)
    assert not np.array_equal(first.rdm, different.rdm)
    np.testing.assert_array_equal(np.random.random(3), expected_global)
