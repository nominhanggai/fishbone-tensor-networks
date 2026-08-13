"""Tests for the model taxonomy: model -> frame -> propagator.

The registry (:mod:`fishbonett.models.registry`) is the single source of truth for
which methods exist.  These tests pin it to the *dispatch* so the two cannot
drift: the previous hand-maintained tables drifted twice (the ``tree-*`` methods
were labelled ``chain`` although their state is a tree, and the multichannel path
was labelled interaction picture although it builds a static Hamiltonian).
"""
import re

import numpy as np
import pytest

from fishbonett import Bath, SystemBath, Fishbone
from fishbonett.models import TreeFishbone
from fishbonett.models import registry as R
from fishbonett.models.registry import FIXED_BOND_METHODS as _FIXED_BOND_METHODS
from fishbonett.operators import sigma_x, sigma_z


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _bath(**kw):
    kw.setdefault("domain", (-25.0, 36.0))
    kw.setdefault("temperature", 1.0)
    kw.setdefault("n_modes", 3)
    kw.setdefault("phys_dim", 4)
    return Bath(J=_J, **kw)


# -- the taxonomy is self-consistent -----------------------------------------
def test_registry_is_the_only_dispatch_table():
    """There is one table, so there is no seam to drift.

    This used to compare ``registry`` against three dispatch dicts in
    ``models/system_bath.py`` that listed the same method names again -- a test
    whose only job was to check two tables agreed.  ``METHODS`` is now the single
    source: ``Model.frames`` is derived from it, and ``run`` dispatches on
    ``Method.integrator``.  What is left to check is that every declared method is
    actually reachable.
    """
    from fishbonett.models.system_bath import SystemBath as SB

    assert set(R.all_methods()) == set(R.METHODS)
    for name, spec in R.METHODS.items():
        assert spec.frame in R.FRAMES, f"{name} names unknown frame {spec.frame!r}"
        assert spec.models, f"{name} belongs to no model"
        for mk in spec.models:
            assert mk in R.MODELS, f"{name} names unknown model {mk!r}"
            assert name in R.MODELS[mk].methods()
        # every integrator must resolve to a driver that exists
        owners = set(spec.models)
        if owners & {"chain", "star", "mode-tree", "multichannel"}:
            attr = SB._DRIVERS[spec.integrator]
            assert callable(getattr(SB, attr, None)), (
                f"{name}: integrator {spec.integrator!r} -> missing {attr}")


def test_fixed_bond_methods_are_registry_data():
    """It was a private set in ``models/system_bath.py`` that tests had to import
    through the underscore.  Which methods cannot grow a bond is taxonomy."""
    assert R.FIXED_BOND_METHODS == frozenset(
        n for n, s in R.METHODS.items() if s.fixed_bond)
    # the 1-site TDVP variants and the adaptive ones, as before
    assert "mpo-tdvp1" in R.FIXED_BOND_METHODS
    assert "mpo-dtdvp" in R.FIXED_BOND_METHODS
    assert "tebd" not in R.FIXED_BOND_METHODS


def test_every_model_frame_pair_has_at_least_one_method():
    for key, m in R.MODELS.items():
        assert m.frames, f"model {key!r} declares no frames"
        for frame, methods in m.frames.items():
            assert frame in R.FRAMES, f"{key!r} names unknown frame {frame!r}"
            assert methods, f"{key!r}/{frame!r} declares no methods"


def test_every_gap_has_a_reason():
    """An absent model/frame combination must say *why* -- impossible, unwise, or
    merely unimplemented.  Silence is what made "is there a polaron tree?"
    unanswerable before."""
    for key, m in R.MODELS.items():
        for frame, why in m.gaps.items():
            assert frame in R.FRAMES, f"{key!r} gap names unknown frame {frame!r}"
            assert why and why.strip(), f"{key!r}/{frame!r} gap has no reason"
            assert frame not in m.frames, (
                f"{key!r}/{frame!r} is listed both as available and as a gap")


def test_frames_and_gaps_together_cover_every_frame():
    """No model may leave a frame unmentioned: either it works, or there is a
    recorded reason it does not."""
    for key, m in R.MODELS.items():
        mentioned = set(m.frames) | set(m.gaps)
        assert mentioned == set(R.FRAMES), (
            f"model {key!r} says nothing about "
            f"{sorted(set(R.FRAMES) - mentioned)}")


def test_method_frames_is_derived_and_carries_the_two_corrections():
    # a tree state is not a chain
    assert R.METHOD_FRAMES["tree-tdvp"] == ("interaction", "mode-tree")
    assert R.METHOD_FRAMES["tree-tebd"] == ("interaction", "mode-tree")
    # polaron is its own frame, not a Schrodinger sub-case
    assert R.METHOD_FRAMES["polaron"] == ("polaron", "chain")
    assert R.METHOD_FRAMES["polaron-tdvp1"] == ("polaron", "chain")
    # unchanged
    assert R.METHOD_FRAMES["mpo-tdvp1"] == ("schrodinger", "chain")
    assert R.METHOD_FRAMES["mpo-ip-tdvp1"] == ("interaction", "star")
    assert R.METHOD_FRAMES["trotter-mpo"] == ("interaction", "chain")


def test_multichannel_default_path_is_schrodinger_not_interaction():
    """F2: the multichannel model's *default* path routes through TreeFishbone,
    whose shared-mode star puts the bath frequencies **on-site** -- a static
    Hamiltonian, i.e. the Schroedinger picture.  It was previously labelled
    interaction picture.  Assert the label against the built Hamiltonian so a
    future rewire cannot silently contradict it.

    The model now has a genuine interaction-picture path too
    (``multichannel-ip``), which is a *different* method -- the point of this test
    is that the static one is not it."""
    frames = R.MODELS["multichannel"].frames
    assert R.STATIC_TREE_TEBD in frames["schrodinger"]
    assert R.MULTICHANNEL_IP in frames["interaction"]
    assert R.STATIC_TREE_TEBD not in frames.get("interaction", ())

    mc = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x], domain=(0.0, 40.0),
              n_modes=3, phys_dim=4)
    fb = TreeFishbone(sites=[sigma_x], edges=[], baths=[mc])
    _dims, _edges, site_H, _edge_H = fb.hamiltonians(t_max=1.0)
    # the bath nodes (everything past the single system site) carry w_k * n
    bath_on_site = [H for H in site_H[1:] if np.any(H)]
    assert bath_on_site, ("no on-site bath terms: the multichannel star is in the "
                          "interaction picture, so the registry label is wrong")


# -- the interface is uniform across models ----------------------------------
def _run_for(model_key):
    """A minimal runnable instance of each model, plus the method to use."""
    h = 0.5 * sigma_x
    if model_key in ("chain", "star", "mode-tree"):
        return SystemBath(h=h, coupling=sigma_z, bath=_bath()), None
    if model_key == "multichannel":
        mc = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x], domain=(0.0, 40.0),
                  n_modes=3, phys_dim=4)
        return SystemBath(h=h, coupling=[sigma_z, sigma_x], bath=mc), None
    if model_key == "comb":
        return Fishbone(sites=[h, h], baths=[_bath(), None]), None
    if model_key == "site-tree":
        return TreeFishbone(sites=[h], edges=[], baths=[_bath()]), None
    raise AssertionError(model_key)


@pytest.mark.parametrize("model_key", sorted(R.MODELS))
def test_each_model_runs_its_own_methods_and_reports_them(model_key):
    obj, _ = _run_for(model_key)
    default = R.methods_of(model_key, "schrodinger")[0] if model_key == "multichannel" else None
    for method in R.methods_of(model_key):
        kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z})
        if method in _FIXED_BOND_METHODS:  # these require an explicit cap
            kw["bond_dim"] = 12
        if method == default:
            # the multichannel model is selected by the bath, so its Schrodinger
            # path is what you get with no `method` at all
            r = obj.run(**kw)
        else:
            r = obj.run(method=method, **kw)
        assert r.t.shape == (2,)
        assert r.method == method, (
            f"{model_key}/{method} reported method={r.method!r}")


@pytest.mark.parametrize("model_key", ["comb", "site-tree"])
def test_multi_site_models_reject_a_single_system_method(model_key):
    """Asking a multi-site model for `tebd` must name the model that owns it,
    rather than raising a bare TypeError as it did before."""
    obj, _ = _run_for(model_key)
    with pytest.raises(ValueError, match="1D system-bath"):
        obj.run(dt=0.02, n_steps=1, method="tebd")


def test_multichannel_bath_rejects_another_models_method():
    """The multichannel model is chosen by the bath's shape, so `method` can only
    pick among *its* propagators -- say so instead of ignoring it."""
    mc = Bath(J=[_J, _J], coupling=[sigma_z, sigma_x], domain=(0.0, 40.0),
              n_modes=3, phys_dim=4)
    m = SystemBath(h=0.5 * sigma_x, coupling=[sigma_z, sigma_x], bath=mc)
    with pytest.raises(ValueError, match="multichannel"):
        m.run(dt=0.02, n_steps=1, method="tebd")


def test_multi_site_models_report_max_bond():
    """Truncation reporting must match the single-system models -- max_bond used
    to be left as None on the tree path."""
    r = TreeFishbone(sites=[0.5 * sigma_x], edges=[],
                     baths=[_bath()]).run(dt=0.02, n_steps=3)
    assert r.max_bond is not None and r.max_bond.shape == (3,)
    assert np.all(r.max_bond >= 1)


# -- lookups ------------------------------------------------------------------
def test_models_of_reports_every_owner():
    assert R.models_of("tebd") == ("chain",)
    # the static tree engine genuinely serves three models
    assert set(R.models_of(R.STATIC_TREE_TEBD)) == {"comb", "multichannel",
                                                    "site-tree"}
    assert R.models_of("not-a-method") == ()


def test_methods_of_explains_a_gap_instead_of_a_bare_keyerror():
    """Asking for an absent frame must quote the registry's recorded reason, not
    raise a bare KeyError -- the reason text itself is free to be reworded."""
    reason = R.MODELS["star"].gaps["polaron"]
    with pytest.raises(KeyError, match=re.escape(reason)):
        R.methods_of("star", "polaron")


def test_describe_taxonomy_mentions_every_model_and_method():
    text = R.describe_taxonomy()
    for key, m in R.MODELS.items():
        assert m.label in text
        for method in m.methods():
            assert method in text
