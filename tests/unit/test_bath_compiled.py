"""Architectural contracts for bath compilation and model-owned couplings."""
import numpy as np
import pytest

from fishbonett import Bath, ChainBath, StarBath, SystemBath
from fishbonett.operators import sigma_x, sigma_z


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _bath(J=_J):
    return Bath(J=J, domain=(0.0, 40.0), n_modes=3, phys_dim=4)


def test_compiled_representations_are_operator_free_and_immutable():
    bath = _bath()
    star = bath.bind(sigma_z).compiled_star()
    chain = bath.bind(sigma_z).compiled_chain()

    assert isinstance(star, StarBath)
    assert star.couplings.shape == (1, 3)
    assert star.chain_transform.shape == (3, 3)
    assert isinstance(chain, ChainBath)
    assert chain.frequencies.shape == (3,)
    assert chain.hoppings.shape == (2,)
    assert not star.frequencies.flags.writeable
    assert not star.couplings.flags.writeable
    assert not chain.frequencies.flags.writeable
    assert not chain.hoppings.flags.writeable
    # System-space operators live on CoupledBath, never in compiled bath data.
    assert not hasattr(star, "operator")
    assert not hasattr(chain, "operator")


def test_bound_bath_caches_each_compiled_representation():
    coupled = _bath().bind(sigma_z)

    assert coupled.compiled_star() is coupled.compiled_star()
    assert coupled.compiled_chain() is coupled.compiled_chain()
    polaron = coupled.compiled_polaron()
    assert polaron is coupled.compiled_polaron()
    assert polaron.chain.n_modes == coupled.n_modes


def test_automatic_resolution_is_cached_per_run_horizon():
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=None, phys_dim=4)
    coupled = bath.bind(sigma_z)

    first = coupled.resolved(0.2)
    assert first is coupled.resolved(0.2)
    assert first is not coupled.resolved(0.3)


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
    star = model.coupled_bath.compiled_star()
    assert star.couplings.shape == (2, 3)
    result = model.run(dt=0.02, n_steps=2, bond_dim=20,
                       observables={"sz": sigma_z})
    assert result.method == "schrodinger-star-tree-tebd"
    assert result.expect["sz"].shape == (2,)
    result_ip = model.run(dt=0.02, n_steps=2, method="interaction-chain-tebd",
                          bond_dim=20, observables={"sz": sigma_z})
    assert result_ip.method == "interaction-chain-tebd"
    assert result_ip.expect["sz"].shape == (2,)


def test_one_density_can_be_shared_by_several_model_channels():
    coupled = _bath().bind([sigma_z, sigma_x])
    star = coupled.compiled_star()

    assert star.couplings.shape == (2, 3)
    np.testing.assert_array_equal(star.couplings[0], star.couplings[1])
    assert star.combine(coupled.operators).shape == (3, 2, 2)


def test_conflicting_legacy_and_model_couplings_are_rejected():
    bath = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x],
                domain=(0.0, 40.0), n_modes=3, phys_dim=4)

    with pytest.raises(ValueError, match="specified twice"):
        SystemBath(h=0.5 * sigma_x,
                   coupling=[sigma_x, sigma_x], bath=bath)


def test_matching_legacy_duplicate_remains_compatible():
    bath = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x],
                domain=(0.0, 40.0), n_modes=3, phys_dim=4)
    model = SystemBath(h=0.5 * sigma_x,
                       coupling=[sigma_z, sigma_x], bath=bath)
    assert model.coupled_bath.is_multichannel
