"""Tests for the integrated truncation policy (``Truncation``: eps + max_bond)."""
import numpy as np
import pytest

from fishbonett import Bath, SystemBath, Truncation
from fishbonett.linalg import DEFAULT_EPS, cap_rank
from fishbonett.operators import sigma_x, sigma_z


def _model(n_modes=3, phys_dim=5):
    bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5.0), domain=(-25.0, 36.0),
                temperature=1.0, n_modes=n_modes, phys_dim=phys_dim)
    return SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)


# -- the policy object -------------------------------------------------------
def test_default_eps_is_1em4_not_something_tighter():
    """The documented default.  A far tighter default is the single most common
    way to waste time here, so it is pinned by a test."""
    assert DEFAULT_EPS == 1e-4
    assert Truncation().eps == 1e-4
    assert Truncation().max_bond is None       # None == unlimited


def test_keep_drops_values_below_relative_eps():
    t = Truncation(eps=1e-4)
    assert t.keep(np.array([1.0, 1e-2, 1e-6])) == 2       # third is below eps
    assert t.keep(np.array([1.0, 1e-2, 1e-3])) == 3       # all above
    # the threshold is *relative* to the largest value, so a rescale changes nothing
    assert t.keep(np.array([1.0, 1e-2, 1e-6]) * 1e8) == 2


def test_max_bond_caps_and_none_is_unlimited():
    s = np.array([1.0, 0.5, 0.25, 0.125])
    assert Truncation(eps=0.0, max_bond=None).keep(s) == 4     # unlimited
    assert Truncation(eps=0.0, max_bond=2).keep(s) == 2        # capped
    assert Truncation(eps=1e-4).keep(np.zeros(5)) == 1         # always keep one


def test_cap_rank_matches_the_method():
    for count, chi in [(7, None), (7, 3), (0, 5), (-4, 2)]:
        assert cap_rank(count, chi) == Truncation(max_bond=chi).cap(count)


def test_resolve_accepts_object_float_or_keywords():
    assert Truncation.resolve(None) == Truncation(eps=DEFAULT_EPS, max_bond=None)
    assert Truncation.resolve(1e-6) == Truncation(eps=1e-6)
    assert Truncation.resolve(None, eps=1e-5, max_bond=50) == Truncation(1e-5, 50)
    obj = Truncation(eps=1e-3, max_bond=7)
    assert Truncation.resolve(obj) is obj


def test_resolve_rejects_contradictory_input():
    with pytest.raises(TypeError):
        Truncation.resolve(Truncation(), eps=1e-3)
    with pytest.raises(TypeError):
        Truncation.resolve("nonsense")


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        Truncation(eps=-1)
    with pytest.raises(ValueError):
        Truncation(max_bond=0)          # 0 would mean "keep nothing"


# -- wiring into run() -------------------------------------------------------
def test_trunc_object_and_loose_keywords_agree():
    """``trunc=Truncation(...)`` must be exactly equivalent to the loose form."""
    kw = dict(dt=0.05, n_steps=3, method="tebd", observables={"sz": sigma_z})
    a = _model().run(trunc=Truncation(eps=1e-5, max_bond=20), **kw)
    b = _model().run(trunc_eps=1e-5, bond_dim=20, **kw)
    np.testing.assert_allclose(a.expect["sz"], b.expect["sz"], rtol=0, atol=0)


def test_run_rejects_both_forms_at_once():
    with pytest.raises(TypeError):
        _model().run(dt=0.05, n_steps=1, method="tebd",
                     trunc=Truncation(eps=1e-5), trunc_eps=1e-5)


def test_unlimited_bond_grows_beyond_a_small_cap():
    """``bond_dim=None`` really is unlimited: the same run under a small cap must
    not exceed it, and the uncapped one must be free to go past it."""
    kw = dict(dt=0.05, n_steps=6, method="tebd", trunc_eps=1e-8,
              observables={"sz": sigma_z})
    capped = _model().run(bond_dim=3, **kw)
    free = _model().run(bond_dim=None, **kw)
    assert capped.max_bond.max() <= 3
    assert free.max_bond.max() > capped.max_bond.max()


def test_fixed_bond_methods_require_an_explicit_cap():
    """1-site TDVP cannot grow a bond, so 'unlimited' is meaningless for it and
    must be rejected with a message naming a usable alternative."""
    with pytest.raises(ValueError, match="fixed bond dimension"):
        _model().run(dt=0.05, n_steps=1, method="mpo-tdvp1", bond_dim=None)
