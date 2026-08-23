"""Architectural contracts for Bath inputs and model-owned couplings."""
import importlib.util

import numpy as np
import pytest

from fishbonett import Bath, SystemBath
from fishbonett.bath.spec import thermalize
from fishbonett.operators import sigma_x, sigma_z


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _bath(J=_J):
    return Bath(J=J, domain=(0.0, 40.0), n_modes=3, phys_dim=4)


def test_bath_is_the_only_public_discretization_input():
    import fishbonett
    from fishbonett.representations.interaction import InteractionRepresentation
    from fishbonett.representations.polaron import PolaronRepresentation
    from fishbonett.representations.schrodinger import SchrodingerRepresentation

    bath = _bath()
    schrodinger = SchrodingerRepresentation(
        representation="schrodinger-chain", h_sys=sigma_x,
        coupling=sigma_z, bath=bath)
    interaction = InteractionRepresentation(
        representation="interaction-star", h_sys=sigma_x,
        coupling=sigma_z, bath=bath).build()
    polaron_bath = Bath(J=_J, domain=(0.2, 40.0), n_modes=3, phys_dim=4)
    polaron = PolaronRepresentation(
        representation="polaron-chain", h_sys=sigma_x,
        coupling=sigma_z, bath=polaron_bath).build()

    assert len(schrodinger.tdvp_mpo()) == 4
    assert len(interaction.tdvp_mpo(0.0)) == 4
    assert len(polaron.tdvp_mpo()) == 4
    for name in (
        "StarBath", "ChainBath", "PolaronBath",
        "compile_star", "compile_chain", "compile_polaron",
    ):
        assert not hasattr(fishbonett, name)
    assert importlib.util.find_spec("fishbonett.bath.compiled") is None


def test_low_level_representation_requires_a_resolved_bath():
    from fishbonett.representations.interaction import InteractionRepresentation

    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=None, phys_dim=4)
    with pytest.raises(ValueError, match=r"bath\.resolved\(t_max\)"):
        InteractionRepresentation(
            representation="interaction-star", h_sys=sigma_x,
            coupling=sigma_z, bath=bath)


def test_automatic_resolution_is_fresh_per_run_horizon():
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=None, phys_dim=4)
    coupled = bath.bind(sigma_z)

    first = coupled.resolved(0.2)
    second = coupled.resolved(0.2)
    assert first is not second
    assert first.bath.n_modes == second.bath.n_modes
    assert first is not coupled.resolved(0.3)
    assert coupled.bath.n_modes is None


def test_legacy_coupling_emits_a_migration_warning():
    bath = Bath(J=_J, coupling=sigma_z, domain=(0.0, 40.0),
                n_modes=3, phys_dim=4)
    with pytest.warns(DeprecationWarning, match="Bath.coupling is deprecated"):
        coupled = bath.bind()
    np.testing.assert_array_equal(coupled.operator, sigma_z)


def test_model_owned_multichannel_couplings_need_no_bath_duplicate():
    bath = _bath(J=[_J, _J])
    model = SystemBath(
        h=0.5 * sigma_x, coupling=[sigma_z, sigma_x], bath=bath)

    assert model.coupled_bath.is_multichannel
    result = model.run(dt=0.02, n_steps=2, bond_dim=20,
                       observables={"sz": sigma_z})
    assert result.method == "schrodinger-star-tree-tebd"
    assert result.expect["sz"].shape == (2,)
    result_ip = model.run(
        dt=0.02, n_steps=2, method="interaction-chain-tebd",
        bond_dim=20, observables={"sz": sigma_z})
    assert result_ip.method == "interaction-chain-tebd"
    assert result_ip.expect["sz"].shape == (2,)


def test_one_density_can_be_shared_by_several_model_channels():
    from fishbonett.representations.multichannel import (
        MultichannelInteractionRepresentation,
    )

    representation = MultichannelInteractionRepresentation(
        representation="interaction-star", h_sys=0.5 * sigma_x,
        coupling=[sigma_z, sigma_x], bath=_bath()).build()
    assert representation.coup_mat_np.shape == (3, 2, 2)


def test_conflicting_legacy_and_model_couplings_are_rejected():
    bath = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x],
                domain=(0.0, 40.0), n_modes=3, phys_dim=4)

    with pytest.warns(DeprecationWarning, match="Bath.coupling is deprecated"):
        with pytest.raises(ValueError, match="specified twice"):
            SystemBath(h=0.5 * sigma_x,
                       coupling=[sigma_x, sigma_x], bath=bath)


def test_matching_legacy_duplicate_remains_compatible():
    bath = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x],
                domain=(0.0, 40.0), n_modes=3, phys_dim=4)
    with pytest.warns(DeprecationWarning, match="Bath.coupling is deprecated"):
        model = SystemBath(h=0.5 * sigma_x,
                           coupling=[sigma_z, sigma_x], bath=bath)
    assert model.coupled_bath.is_multichannel


def test_thermalized_ohmic_density_has_the_correct_zero_frequency_limit():
    density = thermalize(lambda w: 0.4 * w, beta=2.0)
    assert density(0.0) == pytest.approx(0.2, rel=1e-7)
    assert density(1e-9) == pytest.approx(0.2, rel=1e-7)
    assert density(-1e-9) == pytest.approx(0.2, rel=1e-7)


def test_thermalize_rejects_nonphysical_beta():
    with pytest.raises(ValueError, match="beta must be finite and positive"):
        thermalize(_J, beta=0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_modes": 1.5}, "n_modes"),
        ({"n_modes": 0}, "n_modes"),
        ({"phys_dim": 0}, "phys_dim"),
        ({"temperature": 0.0}, "temperature"),
        ({"beta": -1.0}, "beta"),
        ({"temperature": 1.0, "beta": 1.0}, "temperature or beta"),
    ],
)
def test_bath_rejects_invalid_discretization_sizes_and_temperatures(
        kwargs, message):
    with pytest.raises(ValueError, match=message):
        Bath(J=_J, domain=(0.0, 10.0), **kwargs)
