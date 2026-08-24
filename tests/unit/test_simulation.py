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


def test_system_bath_rejects_unknown_engine_options():
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    model = SystemBath(
        h=sigma_x, coupling=sigma_z,
        bath=Bath(
            J=lambda w: 0.2 * w * np.exp(-w / 5.0),
            domain=(0.0, 20.0), n_modes=2, phys_dim=3))
    with pytest.raises(TypeError, match=r"unexpected run option.*trunc_epz"):
        model.run(
            dt=0.02, n_steps=1, method="interaction-chain-tebd",
            trunc_epz=1e-5)
    with pytest.raises(TypeError, match=r"unexpected run option.*initial_bond"):
        model.run(
            dt=0.02, n_steps=1, method="schrodinger-chain-tdvp2",
            initial_bond=4)


@pytest.mark.parametrize("method", [
    "interaction-chain-tree-tebd",
    "interaction-chain-tebd",
    "interaction-chain-trotter-mpo",
    "interaction-chain-tdvp2",
    "polaron-chain-tebd",
])
def test_progress_payload_is_consistent_across_engines(method):
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    model = SystemBath(
        h=sigma_x, coupling=sigma_z,
        bath=Bath.vibronic([1.0, 2.0], [0.1, 0.05], phys_dim=3))
    updates = []
    model.run(
        dt=0.02, n_steps=2, method=method,
        bond_dim=16, progress=updates.append)
    assert [item["step"] for item in updates] == [0, 1]
    assert [item["t"] for item in updates] == pytest.approx([0.02, 0.04])
    assert all(item["bond"] >= 1 for item in updates)
    assert all(set(item) == {
        "step", "n_steps", "t", "bond", "rdm", "state"
    } for item in updates)


@pytest.mark.parametrize("method", [
    "interaction-chain-tree-tebd",
    "interaction-chain-tebd",
    "interaction-chain-trotter-mpo",
    "interaction-chain-tdvp2",
    "interaction-star-tdvp2",
    "schrodinger-chain-tdvp2",
    "schrodinger-star-tdvp2",
    "polaron-chain-tebd",
    "polaron-chain-tdvp2",
    "polaron-star-tdvp2",
])
def test_system_bath_mode_observable_and_sampling_contract(method):
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import annihilate, sigma_x, sigma_z
    from fishbonett.targets import BathMode

    dimension = 3
    destroy = annihilate(dimension)
    number = destroy.conj().T @ destroy
    model = SystemBath(
        h=0.3 * sigma_x, coupling=sigma_z,
        bath=Bath.vibronic(
            [1.0, 2.0], [0.1, 0.05], phys_dim=dimension
        ),
    )
    result = model.run(
        dt=0.01, n_steps=3, method=method, bond_dim=16,
        trunc_eps=1e-8, observe_every=2,
        observables={"mode_0_number": (number, BathMode(0, 0, 0))},
    )
    np.testing.assert_allclose(result.t, [0.02, 0.03])
    assert result.expect["mode_0_number"].shape == (2,)
    assert np.all(result.expect["mode_0_number"] >= -1e-12)
    assert result.meta["observe_every"] == 2


def test_system_bath_rejects_an_unavailable_bath_mode():
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import annihilate, sigma_x, sigma_z
    from fishbonett.targets import BathMode

    destroy = annihilate(3)
    model = SystemBath(
        h=sigma_x, coupling=sigma_z,
        bath=Bath.vibronic([1.0], [0.1], phys_dim=3),
    )
    with pytest.raises(ValueError, match="resolved bath has 1 modes"):
        model.run(
            dt=0.01, n_steps=1, method="interaction-chain-tebd",
            observables={
                "missing": (destroy.conj().T @ destroy, BathMode(0, 0, 1))
            },
        )


def test_mps_bath_measurement_respects_forward_and_reversed_layouts():
    from fishbonett.models.simulation import _measure_mps_bath

    number = np.diag([0.0, 1.0, 2.0])

    def product(local_index, dimension):
        tensor = np.zeros((1, 1, dimension), complex)
        tensor[0, 0, local_index] = 1.0
        return tensor

    tensors = [product(0, 2), product(1, 3), product(2, 3)]
    observables = {"mode_0": (number, 0), "mode_1": (number, 1)}
    forward = _measure_mps_bath(tensors, observables)
    reversed_layout = _measure_mps_bath(
        tensors, observables, reverse_modes=True
    )
    assert forward == {"mode_0": 1.0, "mode_1": 2.0}
    assert reversed_layout == {"mode_0": 2.0, "mode_1": 1.0}


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


def _comb_vibronic_model(n_sites=3, modes=10):
    """A small interaction-chain comb: independent vibronic baths on a backbone."""
    import numpy as np
    from fishbonett import Bath
    from fishbonett.models import Fishbone

    occupied = np.diag([0.0, 1.0])
    cm = 2.0 * np.pi * 2.99792458e-2
    cutoff = 120.0 * cm
    bath = Bath.vibronic(
        [420.0 * cm], [0.04],
        continuum=lambda w: 2.0 * np.pi * 0.05 * max(float(w), 0.0)
                            * np.exp(-max(float(w), 0.0) / cutoff),
        temperature=0.695034 * 300.0 * cm,
        domain=(-6.0 * cutoff, 8.0 * cutoff), n_modes=modes, phys_dim=3,
        discretization="tedopa")
    h = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        h[i, i] = (n_sites - 1 - i) * 250.0 * cm
        if i + 1 < n_sites:
            h[i, i + 1] = h[i + 1, i] = 90.0 * cm
    model = Fishbone.from_single_excitation(
        h, baths={i: bath.bind(occupied) for i in range(n_sites)})
    initial = [np.array([1.0, 0.0], complex) for _ in range(n_sites)]
    initial[0] = np.array([0.0, 1.0], complex)
    return model, initial, occupied


def test_interaction_chain_comb_run_can_be_resumed():
    """Splitting a run at a checkpoint must reproduce the unsplit run exactly.

    This is not a bookkeeping check.  The interaction-picture couplings d_n(t)
    are functions of *absolute* time, so a continuation that restarted the clock
    at zero would keep evolving -- with a different Hamiltonian -- and produce a
    perfectly healthy-looking wrong answer.  The negative control below pins that
    the comparison can actually fail.
    """
    import numpy as np

    model, initial, occupied = _comb_vibronic_model()
    kw = dict(dt=0.002, representation="interaction-chain",
              state_geometry="tree", integrator="tebd",
              observables={"population": occupied},
              trunc_eps=1e-5, bond_dim=64, bath_horizon=0.02)

    whole = model.run(n_steps=10, initial=initial, **kw)
    assert whole.checkpoint is not None, "no checkpoint emitted"

    first = model.run(n_steps=4, initial=initial, **kw)
    second = model.run(n_steps=6, resume=first.checkpoint, **kw)

    end_whole = np.asarray(whole.expect["population"])[-1]
    end_split = np.asarray(second.expect["population"])[-1]
    assert np.array_equal(end_whole, end_split)
    assert np.isclose(second.t[-1], whole.t[-1])
    assert second.t[0] > first.t[-1], "the resumed time grid must continue"

    # negative control: restarting the clock is a *detectable* error, so the
    # equality above is a real constraint rather than a tautology
    restarted = model.run(n_steps=6, initial=initial, **kw)
    assert not np.allclose(
        np.asarray(restarted.expect["population"])[-1], end_split, atol=1e-6)


def test_resumed_comb_rejects_a_different_hamiltonian(tmp_path):
    """A checkpoint may only continue into the model that produced it, and it
    survives a pickle-free round trip through disk."""
    import numpy as np
    from fishbonett.models.result import SimulationCheckpoint

    model, initial, occupied = _comb_vibronic_model()
    kw = dict(dt=0.002, representation="interaction-chain",
              state_geometry="tree", integrator="tebd",
              observables={"population": occupied},
              trunc_eps=1e-5, bond_dim=64, bath_horizon=0.02)
    first = model.run(n_steps=4, initial=initial, **kw)

    reloaded = SimulationCheckpoint.load(first.checkpoint.save(tmp_path / "c.npz"))
    a = model.run(n_steps=3, resume=first.checkpoint, **kw)
    b = model.run(n_steps=3, resume=reloaded, **kw)
    assert np.array_equal(np.asarray(a.expect["population"])[-1],
                          np.asarray(b.expect["population"])[-1])

    cm = 2.0 * np.pi * 2.99792458e-2
    other, other_initial, _ = _comb_vibronic_model()
    other.sites[0] = other.sites[0] + 5.0 * cm * np.diag([0.0, 1.0])
    with pytest.raises(ValueError, match="does not match this resolved model"):
        other.run(n_steps=2, resume=first.checkpoint, **kw)

    # A different bath horizon changes the resolved model.
    with pytest.raises(ValueError, match="bath_horizon"):
        model.run(n_steps=2, resume=first.checkpoint,
                  **{**kw, "bath_horizon": 0.05})


def test_resumed_comb_rejects_changed_resolved_bath_coefficients():
    """Equal topology is insufficient: the finite bath measure must also match."""
    import numpy as np
    from fishbonett import Bath
    from fishbonett.models import Fishbone

    occupied = np.diag([0.0, 1.0])

    def model(frequency):
        bath = Bath.vibronic([frequency], [0.08], phys_dim=3)
        return Fishbone(
            sites=[0.4 * np.array([[0.0, 1.0], [1.0, 0.0]])],
            baths=[bath.bind(occupied)])

    options = dict(
        dt=0.01, n_steps=1,
        method="interaction-chain-fishbone-trotter-mpo",
        bath_horizon=0.02, bond_dim=32, trunc_eps=1e-10,
        observables={"population": occupied})
    first = model(1.0).run(**options)
    with pytest.raises(ValueError, match="does not match this resolved model"):
        model(9.0).run(**{**options, "resume": first.checkpoint})
