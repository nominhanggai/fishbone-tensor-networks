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


def _bath_pos():
    """Zero-temperature bath on a positive domain.

    The cross-method comparison uses this rather than the thermalized signed-domain
    ``_bath`` so that every method sees the same bath with no thermofield branch --
    what is being compared is the propagators, not the discretization.
    """
    return Bath(J=_J, domain=(0.0, 40.0), n_modes=3, phys_dim=4)


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
        assert spec.basis_for() in R.BASES, f"{name} names unknown basis"
        assert spec.geometry in R.GEOMETRIES, f"{name} names unknown geometry"
        # every engine must resolve to a driver that exists
        if set(spec.models) & {"system-bath", "multichannel"}:
            attr = SB._DRIVERS[spec.engine]
            assert callable(getattr(SB, attr, None)), (
                f"{name}: engine {spec.engine!r} -> missing {attr}")


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


def test_every_absent_frame_has_a_reason_available():
    """No model may leave a frame unexplained: either it works, or asking why
    produces an answer.

    Deliberately *not* ``set(frames) | set(gaps) == set(FRAMES)``, which is what this
    checked while every reason was hand-written.  Most are derived now -- the
    (multichannel, polaron) cell exists in no ``gaps`` entry because two constraints
    clash there -- so the invariant is that the reason is obtainable, not that
    somebody typed it."""
    for key, m in R.MODELS.items():
        for frame in R.FRAMES:
            if frame in m.frames:
                continue
            why = R.why_not(key, frame)
            assert why, f"model {key!r} cannot explain why it has no {frame!r} frame"


def test_method_frames_is_a_projection_not_an_identity():
    """``METHOD_FRAMES`` maps a method to ``(frame, model)`` -- which no longer
    identifies it, and that is the point.

    The old taxonomy gave these different *models* (``chain`` / ``star`` /
    ``mode-tree``) so that the pair looked like a key.  Those were a bath basis and a
    state geometry wearing a model's name; collapsing them means the pair is now a
    genuine projection, and the axes that separate the collisions are the two the
    taxonomy gained."""
    sb = ("interaction", "system-bath")
    assert R.METHOD_FRAMES["tree-tdvp2"] == sb
    assert R.METHOD_FRAMES["mpo-ip-tdvp2"] == sb
    # ...same frame, same model, same integrator -- separated only by geometry
    assert R.METHODS["tree-tdvp2"].geometry == "binary-tree"
    assert R.METHODS["mpo-ip-tdvp2"].geometry == "path"

    schro = ("schrodinger", "system-bath")
    assert R.METHOD_FRAMES["mpo-tdvp2"] == schro
    assert R.METHOD_FRAMES["mpo-star-tdvp2"] == schro
    # ...separated only by basis: the one pair in the table that is
    assert R.METHODS["mpo-tdvp2"].basis_for() == "chain"
    assert R.METHODS["mpo-star-tdvp2"].basis_for() == "star"

    # polaron is its own frame, not a Schrodinger sub-case
    assert R.METHOD_FRAMES["polaron"] == ("polaron", "system-bath")
    assert R.METHOD_FRAMES["trotter-mpo"] == sb


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
    if model_key == "system-bath":
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


@pytest.mark.parametrize("model_key", sorted(R.MODELS))
def test_every_method_reports_max_bond(model_key):
    """No method may leave ``max_bond`` unreported.

    Six of them used to, for no reason beyond which driver they happened to go
    through: the 1-site TDVP wrappers returned ``(t, obs)`` while the 2-site ones
    returned ``maxD``, and the mode-tree engine never collected it -- yet
    ``polaron-tdvp1``, the same sweep with the same fixed bond, did report it.  It
    is the same quantity for every method, constant or not.
    """
    obj, _ = _run_for(model_key)
    default = R.methods_of(model_key, "schrodinger")[0] if model_key == "multichannel" else None
    for method in R.methods_of(model_key):
        kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z})
        if method in _FIXED_BOND_METHODS:
            kw["bond_dim"] = 12
        r = obj.run(**kw) if method == default else obj.run(method=method, **kw)
        assert r.max_bond is not None, f"{model_key}/{method} reports no max_bond"
        assert np.shape(r.max_bond) == (2,), f"{model_key}/{method}: one per step"
        assert np.all(np.asarray(r.max_bond) >= 1), f"{model_key}/{method}"


def test_result_shape_contract_matches_its_docstring():
    """``models/result.py`` documents the two Result shapes; check both.

    That docstring is a table of promises -- what ``expect``, ``rdm`` and ``meta``
    hold for a single-system model versus a multi-site one -- and nothing was
    checking any of it.  The shape is what downstream code indexes, so it is a real
    interface, not a comment.
    """
    h, kw = 0.5 * sigma_x, dict(dt=0.02, n_steps=2, trunc_eps=1e-7,
                                observables={"sz": sigma_z})

    single = SystemBath(h=h, coupling=sigma_z, bath=_bath_pos()).run(
        method="tebd", **kw)
    assert single.meta == {}, "a single-system Result carries no n_sites"
    assert np.shape(single.rdm) == (2, 2, 2), "(n_steps, d, d)"
    assert np.shape(single.expect["sz"]) == (2,), "(n_steps,)"

    for n_sites, obj in (
        (1, TreeFishbone(sites=[h], edges=[], baths=[_bath_pos()])),
        (2, Fishbone(sites=[h, h], baths=[_bath_pos(), None])),
        (3, TreeFishbone(sites=[h, h, h], edges=[(0, 1), (1, 2)],
                         baths=[_bath_pos(), None, None])),
    ):
        r = obj.run(**kw)
        assert r.meta == {"n_sites": n_sites}, r.meta
        assert np.shape(r.rdm) == (2, n_sites, 2, 2), "(n_steps, n_sites, d, d)"
        # a bare per-site operator gives one column per site
        assert np.shape(r.expect["sz"]) == (2, n_sites), "(n_steps, n_sites)"


def test_every_method_agrees_on_the_same_physics():
    """The methods are each other's cross-check -- so check them against each other.

    Every ``system-bath`` method is one system plus one bath written a different
    way -- two bases, two geometries, three frames -- and a one-site ``site-tree``
    is the same physics again on the general tree engine.  All of those rewritings
    are exact, not approximations, so every one must land on the same trajectory to
    within its own Trotter and truncation error.

    This is the broadest correctness statement the package can make about itself:
    17 independent code paths agreeing on one number.  A frame that dropped a term,
    a geometry wired to the wrong bath, or a propagator applying gates in the wrong
    order would show up here as a gross disagreement rather than a subtle one.
    """
    h, coup = 0.5 * sigma_x, sigma_z
    dt, n_steps = 0.02, 20

    def run(model_key, method):
        if model_key == "site-tree":
            obj = TreeFishbone(sites=[h], edges=[], baths=[_bath_pos()])
        else:
            obj = SystemBath(h=h, coupling=coup, bath=_bath_pos())
        r = obj.run(dt=dt, n_steps=n_steps, method=method, bond_dim=40,
                    trunc_eps=1e-10, observables={"sz": sigma_z})
        sz = np.asarray(r.expect["sz"])
        return sz.reshape(n_steps, -1)[:, 0] if sz.ndim > 1 else sz

    results = {}
    for key in ("system-bath", "site-tree"):
        for m in R.methods_of(key):
            results[(key, m)] = run(key, m)

    ref = results[("system-bath", "tebd")]
    for (key, m), sz in results.items():
        # mpo-dtdvp is bond-adaptive: its accuracy is set by `prec`, not by dt, so
        # it sits a decade or so off the rest.  Still far from a wrong answer.
        tol = 5e-3 if m == "mpo-dtdvp" else 1e-3
        assert np.abs(sz - ref).max() < tol, (
            f"{key}/{m} disagrees with system-bath/tebd by "
            f"{np.abs(sz - ref).max():.2e} on identical physics")


def test_application_matches_what_the_drivers_actually_do():
    """``Method.application`` records how H's *interaction* graph meets the state's.

    The package had no name for this, which is why three methods each re-derived a
    swap network and nothing said they did.  It is now *derived* from the other axes
    rather than stored, so this test is what keeps the derivation honest: exactly the
    methods deriving ``"swap"`` must be the ones whose driver calls
    ``symmetric_swap_step``.
    """
    import inspect
    from fishbonett.models.system_bath import SystemBath as SB

    assert {s.application for s in R.METHODS.values()} <= set(R.APPLICATIONS)

    declared = {n for n, s in R.METHODS.items() if s.application == "swap"}
    actual = set()
    for name, spec in R.METHODS.items():
        attr = SB._DRIVERS.get(spec.engine)
        if attr is None:
            continue
        if "symmetric_swap_step" in inspect.getsource(getattr(SB, attr)):
            actual.add(name)
    assert declared == actual, (
        f"application='swap' says {sorted(declared)} but the drivers that swap are "
        f"{sorted(actual)}")

    # a swap network is what a *star* basis costs on a *path* state -- which is now
    # sayable directly, rather than being a property of a method's name
    assert all(R.METHODS[n].basis_for() == "star"
               and R.METHODS[n].geometry == "path" for n in declared)

    # and an application is realized *once*: the swap methods share one engine, and
    # differ only in which frame supplies H(t)
    assert len({R.METHODS[n].engine for n in declared}) == 1, (
        "the swap application should have one driver, not one per frame")
    assert set(SB._SWAP_FRAMES) == {
        (R.METHODS[n].frame, R.METHODS[n].models[0]) for n in declared}


def test_run_takes_the_axes_directly():
    """A method name *is* a point in the five-axis space, so both spell the same
    run.  The axes are the structure; the name is the shorthand."""
    kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z}, trunc_eps=1e-7)
    by_axes = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath()).run(
        frame="interaction", geometry="path", integrator="tdvp2", **kw)
    by_name = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath()).run(
        method="mpo-ip-tdvp2", **kw)
    assert by_axes.method == by_name.method == "mpo-ip-tdvp2"
    assert np.array_equal(by_axes.rdm, by_name.rdm)

    # the axis vocabulary is uniform across frames: "tebd" means the same word
    # whether the frame dresses the state or not
    sb = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath())
    assert sb.run(frame="polaron", integrator="tebd", **kw).method == "polaron"
    assert sb.run(frame="interaction", geometry="path", integrator="tebd",
                  **kw).method == "tebd"
    # the basis is *inferable* from the frame, so naming it changes nothing
    assert sb.run(frame="polaron", basis="chain", integrator="tebd",
                  **kw).method == "polaron"


def test_axis_errors_name_the_physics_not_just_the_table():
    """The constraints are physical, so a rejected combination should say why.

    These three used to be hand-written ``gaps`` prose attached to models that no
    longer exist; they are derived from :func:`forced_basis` now, and the error is
    where that derivation becomes visible."""
    kw = dict(dt=0.02, n_steps=2, trunc_eps=1e-7)
    sb = lambda: SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath())

    # the interaction picture has no chain: it rotates out H_B, diagonal only in
    # the star basis
    with pytest.raises(ValueError, match="diagonal only in the star basis"):
        sb().run(frame="interaction", basis="chain", **kw)
    # the polaron displacement has nowhere to localize in a star
    with pytest.raises(ValueError, match="localizes that on c0"):
        sb().run(frame="polaron", basis="star", **kw)
    # a chain's hoppings are long-range on a balanced binary tree
    with pytest.raises(ValueError, match="no mode-mode terms"):
        sb().run(frame="schrodinger", basis="chain", geometry="binary-tree", **kw)

    # under-specified: schrodinger exists in both bases
    with pytest.raises(ValueError, match="ambiguous"):
        sb().run(frame="schrodinger", **kw)
    # ...and the interaction picture in both geometries
    with pytest.raises(ValueError, match="ambiguous"):
        sb().run(frame="interaction", integrator="tdvp2", **kw)
    # a name already fixes all five axes, so mixing spellings is a mistake
    with pytest.raises(ValueError, match="not both"):
        sb().run(method="tebd", frame="polaron", **kw)
    # an axis that is not one
    with pytest.raises(TypeError, match="unknown axis"):
        R.resolve({"system-bath"}, layout="swap")


def test_the_old_model_names_say_what_they_became():
    """``chain``/``star``/``mode-tree`` were a basis and a geometry wearing a
    model's name.  They are gone -- but the error has to teach the replacement,
    because they were the documented spelling."""
    for gone, axis in (("chain", "basis='chain'"), ("star", "basis='star'"),
                       ("mode-tree", "geometry='binary-tree'")):
        with pytest.raises(KeyError, match=re.escape(axis)):
            R.model(gone)


def test_every_method_is_reachable_by_its_axes():
    """Whatever the registry declares must be selectable by its axes.

    Every method must be pinned down by the five together -- if two rows shared all
    five they would be the same run under two names."""
    seen = {}
    for mk in R.MODELS:
        for name in R.methods_of(mk):
            spec = R.METHODS[name]
            axes = dict(model=mk, frame=spec.frame, basis=spec.basis_for(mk),
                        geometry=spec.geometry, integrator=spec.integrator)
            got = R.resolve(set(R.MODELS), **axes)
            assert got.name == name, f"{axes} -> {got.name}"
            key = tuple(sorted(axes.items()))
            assert key not in seen, f"{name} and {seen[key]} share all five axes"
            seen[key] = name


@pytest.mark.parametrize("model_key", ["comb", "site-tree"])
def test_multi_site_models_reject_a_single_system_method(model_key):
    """Asking a multi-site model for `tebd` must name the model that owns it,
    rather than raising a bare TypeError as it did before."""
    obj, _ = _run_for(model_key)
    with pytest.raises(ValueError, match="belongs to system-bath"):
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
    assert R.models_of("tebd") == ("system-bath",)
    # the static tree engine genuinely serves three models
    assert set(R.models_of(R.STATIC_TREE_TEBD)) == {"comb", "multichannel",
                                                    "site-tree"}
    assert R.models_of("not-a-method") == ()


def test_methods_of_explains_a_gap_instead_of_a_bare_keyerror():
    """Asking for an absent frame must quote the registry's recorded reason, not
    raise a bare KeyError -- the reason text itself is free to be reworded."""
    reason = R.MODELS["comb"].gaps["polaron"]
    with pytest.raises(KeyError, match=re.escape(reason)):
        R.methods_of("comb", "polaron")


def test_a_derived_impossibility_explains_itself_too():
    """(multichannel, polaron) exists in no table and in no ``gaps`` entry: the
    frame forces a chain basis, the model forces a star, and nothing satisfies
    both.  That has to be sayable without anyone having written it down."""
    why = R.why_not("multichannel", "polaron")
    assert why and "basis='chain'" in why and "basis='star'" in why
    with pytest.raises(KeyError, match="no basis left"):
        R.methods_of("multichannel", "polaron")


def test_declared_bases_agree_with_the_forced_ones():
    """``Method.basis`` is written out only where it is a real choice; everywhere
    else it comes from the rule.  If a row ever declares one that contradicts the
    rule, the table and the physics have diverged."""
    free = set()
    for name, spec in R.METHODS.items():
        for mk in spec.models:
            need = R.forced_basis(spec.frame, mk)
            if need is None:
                free.add((spec.frame, mk))
                assert spec.basis, (
                    f"{name}: basis is free for ({spec.frame}, {mk}) so the row "
                    f"must declare one")
            else:
                assert not spec.basis or spec.basis == need, (
                    f"{name} declares basis={spec.basis!r} but ({spec.frame}, "
                    f"{mk}) forces {need!r}")
    # ...and there is exactly one such cell, which is why mpo-tdvp2 and
    # mpo-star-tdvp2 are the only pair differing by basis alone
    assert free == {("schrodinger", "system-bath")}


def test_describe_taxonomy_mentions_every_model_and_method():
    text = R.describe_taxonomy()
    for key, m in R.MODELS.items():
        assert m.label in text
        for method in m.methods():
            assert method in text
