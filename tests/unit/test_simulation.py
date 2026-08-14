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


def test_physical_model_does_not_import_frame_or_evolution_engines():
    import inspect
    from fishbonett.models.system_bath import SystemBath

    source = inspect.getsource(inspect.getmodule(SystemBath))
    assert "fishbonett.evolve" not in source
    assert "fishbonett.frames" not in source
    assert not hasattr(SystemBath, "_DRIVERS")
    assert not hasattr(SystemBath, "_MPO_FRAMES")
    assert not hasattr(SystemBath, "_SWAP_FRAMES")
