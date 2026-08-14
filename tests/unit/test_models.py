"""Tests for the model taxonomy: model -> representation -> propagator.

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
def test_registry_and_plan_compilers_are_the_two_dispatch_boundaries():
    """Taxonomy rows and engine implementations cannot drift.

    This used to compare ``registry`` against three dispatch dicts in
    ``models/system_bath.py`` that listed the same method names again -- a test
    whose only job was to check duplicate tables agreed.  ``METHODS`` is the source
    for method names and axes; ``PLAN_COMPILERS`` maps only its coarser engine keys
    to implementations.  The physical model owns neither mapping.
    """
    from fishbonett.models.system_bath import SystemBath as SB
    from fishbonett.models.simulation import PLAN_COMPILERS

    assert set(R.all_methods()) == set(R.METHODS)
    for name, spec in R.METHODS.items():
        assert spec.representation in R.REPRESENTATIONS, f"{name} names unknown representation {spec.representation!r}"
        assert spec.models, f"{name} belongs to no model"
        for mk in spec.models:
            assert mk in R.MODELS, f"{name} names unknown model {mk!r}"
            assert name in R.MODELS[mk].methods()
        assert spec.representation.count("-") == 1
        assert spec.geometry in R.GEOMETRIES, f"{name} names unknown geometry"
        # every single-system engine must resolve to one plan compiler
        if set(spec.models) & {"system-bath", "multichannel"}:
            assert callable(PLAN_COMPILERS.get(spec.engine)), (
                f"{name}: engine {spec.engine!r} has no plan compiler")

    assert not hasattr(SB, "_DRIVERS")
    assert not hasattr(SB, "_MPO_REPRESENTATIONS")
    assert not hasattr(SB, "_SWAP_REPRESENTATIONS")


def test_fixed_bond_methods_are_registry_data():
    """It was a private set in ``models/system_bath.py`` that tests had to import
    through the underscore.  Which methods cannot grow a bond is taxonomy."""
    assert R.FIXED_BOND_METHODS == frozenset(
        n for n, s in R.METHODS.items() if s.fixed_bond)
    # the 1-site TDVP variants and the adaptive ones, as before
    assert "mpo-tdvp1" in R.FIXED_BOND_METHODS
    assert "mpo-dtdvp" in R.FIXED_BOND_METHODS
    assert "tebd" not in R.FIXED_BOND_METHODS


def test_every_model_representation_pair_has_at_least_one_method():
    for key, m in R.MODELS.items():
        assert m.representations, f"model {key!r} declares no representations"
        for representation, methods in m.representations.items():
            assert representation in R.REPRESENTATIONS, f"{key!r} names unknown representation {representation!r}"
            assert methods, f"{key!r}/{representation!r} declares no methods"


def test_every_gap_has_a_reason():
    """An absent model/representation combination must say *why* -- impossible, unwise, or
    merely unimplemented.  Silence is what made "is there a polaron tree?"
    unanswerable before."""
    for key, m in R.MODELS.items():
        for representation, why in m.gaps.items():
            assert representation in R.REPRESENTATIONS, f"{key!r} gap names unknown representation {representation!r}"
            assert why and why.strip(), f"{key!r}/{representation!r} gap has no reason"
            assert representation not in m.representations, (
                f"{key!r}/{representation!r} is listed both as available and as a gap")


def test_every_absent_representation_has_a_reason_available():
    """No model may leave a representation unexplained: either it works, or asking why
    produces an answer.

    Deliberately *not* ``set(representations) | set(gaps) == set(REPRESENTATIONS)``, which is what this
    checked while every reason was hand-written.  Most are derived now -- the
    (multichannel, polaron) cell exists in no ``gaps`` entry because two constraints
    clash there -- so the invariant is that the reason is obtainable, not that
    somebody typed it."""
    for key, m in R.MODELS.items():
        for representation in R.REPRESENTATIONS:
            if representation in m.representations:
                continue
            why = R.why_not(key, representation)
            assert why, f"model {key!r} cannot explain why it has no {representation!r} representation"


def test_method_representations_is_a_projection_not_an_identity():
    """``METHOD_REPRESENTATIONS`` maps a method to ``(representation, model)`` -- which no longer
    identifies it, and that is the point.

    The old taxonomy gave these different *models* (``chain`` / ``star`` /
    ``mode-tree``) so that the pair looked like a key.  The first two are now part
    of the complete representation name and the third is a state geometry; this means the pair is now a
    genuine projection, and the axes that separate the collisions are the two the
    taxonomy gained."""
    sb = ("interaction-chain", "system-bath")
    assert R.METHOD_REPRESENTATIONS["tree-tdvp2"] == sb
    assert R.METHOD_REPRESENTATIONS["mpo-ip-tdvp2"] == sb
    # ...same representation, same model, same integrator -- separated only by geometry
    assert R.METHODS["tree-tdvp2"].geometry == "binary-tree"
    assert R.METHODS["mpo-ip-tdvp2"].geometry == "path"

    assert R.METHOD_REPRESENTATIONS["mpo-tdvp2"] == ("schrodinger-chain", "system-bath")
    assert R.METHOD_REPRESENTATIONS["mpo-star-tdvp2"] == ("schrodinger-star", "system-bath")

    # polaron is its own representation, not a Schrodinger sub-case
    assert R.METHOD_REPRESENTATIONS["polaron"] == ("polaron-chain", "system-bath")
    assert R.METHOD_REPRESENTATIONS["trotter-mpo"] == sb


def test_multichannel_default_path_is_schrodinger_not_interaction():
    """F2: the multichannel model's *default* path routes through TreeFishbone,
    whose shared-mode star puts the bath frequencies **on-site** -- a static
    Hamiltonian, i.e. the Schroedinger picture.  It was previously labelled
    interaction picture.  Assert the label against the built Hamiltonian so a
    future rewire cannot silently contradict it.

    The model now has a genuine interaction-picture path too
    (``multichannel-ip``), which is a *different* method -- the point of this test
    is that the static one is not it."""
    representations = R.MODELS["multichannel"].representations
    assert R.MULTICHANNEL_STATIC in representations["schrodinger-star"]
    assert R.MULTICHANNEL_IP in representations["interaction-chain"]
    assert R.MULTICHANNEL_IP_STAR in representations["interaction-star"]
    assert R.MULTICHANNEL_STATIC not in representations.get("interaction-star", ())
    assert set(representations) == {
        "schrodinger-star", "interaction-chain", "interaction-star"}

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
        mc = Bath(J=[_J, _J], domain=(0.0, 40.0),
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
    default = R.methods_of(model_key, "schrodinger-star")[0] if model_key == "multichannel" else None
    for method in R.methods_of(model_key):
        kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z})
        if method in _FIXED_BOND_METHODS:  # these require an explicit cap
            kw["bond_dim"] = 12
        if method == default:
            # the multichannel model is selected by its coupling list, so its Schrodinger
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
    default = R.methods_of(model_key, "schrodinger-star")[0] if model_key == "multichannel" else None
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

    Every ``system-bath`` method is one system plus one bath written in one of six
    representations on two geometries, and a one-site ``site-tree``
    is the same physics again on the general tree engine.  All of those rewritings
    are exact, not approximations, so every one must land on the same trajectory to
    within its own Trotter and truncation error.

    This is the broadest correctness statement the package can make about itself:
    17 independent code paths agreeing on one number.  A representation that dropped a term,
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
    from fishbonett.models.simulation import PLAN_COMPILERS

    assert {s.application for s in R.METHODS.values()} <= set(R.APPLICATIONS)

    declared = {n for n, s in R.METHODS.items() if s.application == "swap"}
    actual = set()
    for name, spec in R.METHODS.items():
        compiler = PLAN_COMPILERS.get(spec.engine)
        if compiler is None:
            continue
        if "symmetric_swap_step" in inspect.getsource(compiler):
            actual.add(name)
    assert declared == actual, (
        f"application='swap' says {sorted(declared)} but the drivers that swap are "
        f"{sorted(actual)}")

    # a swap network is what a star *interaction graph* costs on a *path* state.
    # Deliberately keyed on `mode_decoupled`: `tebd` is
    # interaction-chain and still swaps, because it is rotating H_B away that
    # spreads the coupling over every mode, not the choice of modes to write it in.
    assert all(R.REPRESENTATIONS[R.METHODS[n].representation].mode_decoupled
               and R.METHODS[n].geometry == "path" for n in declared)
    assert R.METHODS["tebd"].representation == "interaction-chain"

    # and an application is realized *once*: the swap methods share one engine, and
    # differ only in which representation supplies H(t)
    swap_engines = {R.METHODS[n].engine for n in declared}
    assert len(swap_engines) == 1, (
        "the swap application should have one driver, not one per representation")
    assert "symmetric_swap_step" in inspect.getsource(
        PLAN_COMPILERS[next(iter(swap_engines))])


def test_run_takes_the_axes_directly():
    """A method name *is* a point in the four-axis space, so both spell the same
    run.  The axes are the structure; the name is the shorthand."""
    kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z}, trunc_eps=1e-7)
    by_axes = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath()).run(
        representation="interaction-chain", geometry="path", integrator="tdvp2", **kw)
    by_name = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath()).run(
        method="mpo-ip-tdvp2", **kw)
    assert by_axes.method == by_name.method == "mpo-ip-tdvp2"
    assert np.array_equal(by_axes.rdm, by_name.rdm)

    # the axis vocabulary is uniform across representations: "tebd" means the same word
    # whether the representation dresses the state or not
    sb = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath())
    assert sb.run(representation="polaron-chain", integrator="tebd", **kw).method == "polaron"
    assert sb.run(representation="interaction-chain", geometry="path", integrator="tebd",
                  **kw).method == "tebd"
    # Partial names are intentionally not another public taxonomy.
    with pytest.raises(ValueError, match="no method"):
        sb.run(representation="polaron", integrator="tebd", **kw)


def test_six_complete_representations_are_registered():
    """All six names are complete public choices and all are implemented.

    In particular, ``interaction-chain`` means star discretization, free-star
    interaction transformation, then star-to-chain transformation.  The
    ``polaron-star`` representation retains the per-mode Lang--Firsov
    displacements."""
    grid = {f"{p}-{b}" for p in ("schrodinger", "interaction", "polaron")
            for b in ("chain", "star")}
    assert set(R.REPRESENTATIONS) == grid
    assert set(R.MODELS["system-bath"].representations) == grid
    assert {s.representation for s in R.METHODS.values()} == grid
    for item in R.REPRESENTATIONS.values():
        assert not hasattr(item, "picture")
        assert not hasattr(item, "basis")


def test_interaction_chain_is_what_the_ip_methods_actually_run():
    """The implemented interaction-picture methods hold **chain** modes.

    The star-to-chain transform rotates the star phases into chain modes, so at
    ``t = 0`` the coupling sits entirely on ``c0`` -- the Schroedinger chain
    configuration -- and spreads outward with ``t``.  Star modes would give every
    entry nonzero at ``t = 0``.  These were labelled ``interaction-star`` until this
    was measured."""
    from fishbonett.bath.chain import star_transform
    freq, Vn, coefT = star_transform(_J, 6, (0.0, 40.0))

    d0 = np.abs(coefT @ (Vn * np.exp(-1j * freq * 0.0)))
    assert np.isclose(d0[0], np.linalg.norm(Vn))       # all of it on one site
    assert np.allclose(d0[1:], 0.0)                    # ...and none anywhere else
    assert (np.abs(Vn) > 1e-3).sum() > 1               # the star is *not* like that

    d1 = np.abs(coefT @ (Vn * np.exp(-1j * freq * 0.5)))
    assert (d1 > 1e-3).sum() > 1, "the coupling must spread as t grows"

    for m in ("tebd", "trotter-mpo", "mpo-ip-tdvp1", "mpo-ip-tdvp2",
              "tree-tdvp", "tree-tdvp2", "tree-tebd"):
        assert R.METHODS[m].representation == "interaction-chain", m
    # Multichannel exposes the same distinction explicitly.
    assert R.METHODS[R.MULTICHANNEL_IP].representation == "interaction-chain"
    assert R.METHODS[R.MULTICHANNEL_IP_STAR].representation == "interaction-star"
    assert R.METHODS["mpo-ip-star-tdvp2"].representation == "interaction-star"


def test_the_two_interaction_representations_agree():
    """`interaction-chain` and `interaction-star` are one orthogonal transform
    apart, so they must land on the same trajectory.

    The point of implementing the star one: it reaches the same answer through a
    completely different coupling vector (``V_k e^{-i w_k t}`` rather than its
    rotation back to the chain), so agreement checks the chain route rather than
    restating it.  It is also what catches a wrong site-order convention."""
    h, kw = 0.5 * sigma_x, dict(dt=0.02, n_steps=20, bond_dim=40,
                                trunc_eps=1e-10, observables={"sz": sigma_z})
    def run(m):
        sb = SystemBath(h=h, coupling=sigma_z, bath=_bath_pos())
        return np.asarray(sb.run(method=m, **kw).expect["sz"])
    chain, star = run("mpo-ip-tdvp2"), run("mpo-ip-star-tdvp2")
    assert np.abs(chain - star).max() < 1e-3, np.abs(chain - star).max()


def test_axis_errors_name_what_separates_the_candidates():
    kw = dict(dt=0.02, n_steps=2, trunc_eps=1e-7)
    sb = lambda: SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath())

    # A partial name is not a public representation.
    with pytest.raises(ValueError, match="no method"):
        sb().run(representation="schrodinger", **kw)
    # ...and one representation spanning two geometries is too
    with pytest.raises(ValueError, match="ambiguous"):
        sb().run(representation="interaction-chain", integrator="tdvp2", **kw)
    # An exact representation still needs enough axes to choose its integrator.
    with pytest.raises(ValueError, match="ambiguous"):
        sb().run(representation="polaron-star", **kw)
    # a chain representation on a binary tree: the one geometry constraint left
    with pytest.raises(ValueError, match="no mode-mode terms"):
        sb().run(representation="schrodinger-chain", geometry="binary-tree", **kw)
    # a name already fixes all four axes, so mixing spellings is a mistake
    with pytest.raises(ValueError, match="not both"):
        sb().run(method="tebd", representation="polaron-chain", **kw)
    # Neither a deprecated decomposition nor an application detail is an axis.
    for gone in ("basis", "frame", "layout"):
        with pytest.raises(TypeError, match="unknown axis"):
            R.resolve({"system-bath"}, **{gone: "star"})


def test_the_old_model_names_say_what_they_became():
    """``chain``/``star``/``mode-tree`` were half a representation and a geometry wearing a
    model's name.  They are gone -- but the error has to teach the replacement,
    because they were the documented spelling."""
    for gone, hint in (("chain", "schrodinger-chain"), ("star", "schrodinger-star"),
                       ("mode-tree", "geometry='binary-tree'")):
        with pytest.raises(KeyError, match=re.escape(hint)):
            R.model(gone)


def test_every_method_is_reachable_by_its_axes():
    """Whatever the registry declares must be selectable by its axes.

    Every method must be pinned down by the four together -- if two rows shared all
    four they would be the same run under two names."""
    seen = {}
    for mk in R.MODELS:
        for name in R.methods_of(mk):
            spec = R.METHODS[name]
            axes = dict(model=mk, representation=spec.representation,
                        geometry=spec.geometry, integrator=spec.integrator)
            got = R.resolve(set(R.MODELS), **axes)
            assert got.name == name, f"{axes} -> {got.name}"
            key = tuple(sorted(axes.items()))
            assert key not in seen, f"{name} and {seen[key]} share all four axes"
            seen[key] = name


@pytest.mark.parametrize("model_key", ["comb", "site-tree"])
def test_multi_site_models_reject_a_single_system_method(model_key):
    """Asking a multi-site model for `tebd` must name the model that owns it,
    rather than raising a bare TypeError as it did before."""
    obj, _ = _run_for(model_key)
    with pytest.raises(ValueError, match="belongs to system-bath"):
        obj.run(dt=0.02, n_steps=1, method="tebd")


def test_multichannel_model_rejects_another_models_method():
    """The multichannel model is chosen by the coupling list, so `method` can only
    pick among *its* propagators -- say so instead of ignoring it."""
    mc = Bath(J=[_J, _J], domain=(0.0, 40.0),
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
    # the static tree engine serves both multi-site models
    assert set(R.models_of(R.STATIC_TREE_TEBD)) == {"comb", "site-tree"}
    assert R.models_of("not-a-method") == ()


def test_methods_of_explains_a_gap_instead_of_a_bare_keyerror():
    """Asking for an absent representation must quote the registry's recorded reason, not
    raise a bare KeyError -- the reason text itself is free to be reworded."""
    reason = R.MODELS["comb"].gaps["polaron-chain"]
    with pytest.raises(KeyError, match=re.escape(reason)):
        R.methods_of("comb", "polaron-chain")


def test_one_engine_can_serve_two_representations():
    """``tree-tebd-static`` and ``multichannel-static`` are the same engine on the
    same geometry, split because they are different **representations**.

    ``representations/schrodinger.py`` picks ``star_terms`` exactly when the bath is
    multichannel and chain terms otherwise, so the split was always in the code.
    One row could not carry both Hamiltonians, which is what forced it into the
    table -- and the two now say plainly which representation each
    model's bath is in."""
    static, mc = R.METHODS[R.STATIC_TREE_TEBD], R.METHODS[R.MULTICHANNEL_STATIC]
    assert static.engine == mc.engine == "static-tree-tebd"
    assert static.geometry == mc.geometry == "comb-tree"
    assert (static.representation, mc.representation) == ("schrodinger-chain", "schrodinger-star")
    assert static.models == ("comb", "site-tree") and mc.models == ("multichannel",)


def test_describe_taxonomy_mentions_every_model_and_method():
    text = R.describe_taxonomy()
    for key, m in R.MODELS.items():
        assert m.label in text
        for method in m.methods():
            assert method in text
