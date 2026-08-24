"""Model taxonomy and dispatch tests."""
import re

import numpy as np
import pytest

from fishbonett import Bath, CoupledBath, SystemBath, Fishbone
from fishbonett.models import TreeFishbone
from fishbonett.models import registry as R
from fishbonett.models.registry import BOND_CAP_REQUIRED_METHODS as _BOND_CAP_REQUIRED_METHODS
from fishbonett.operators import sigma_x, sigma_z


def _J(w):
    # Super-Ohmic at the origin so every registered representation, including
    # the Lang--Firsov polaron, is mathematically defined on the same bath.
    return 0.008 * w ** 3 * np.exp(-w / 5.0)


def _bath(**kw):
    kw.setdefault("domain", (-25.0, 36.0))
    kw.setdefault("temperature", 1.0)
    kw.setdefault("n_modes", 3)
    kw.setdefault("phys_dim", 4)
    return Bath(J=_J, **kw)


def _bath_pos():
    """Zero-temperature bath shared by cross-method comparisons."""
    return Bath(J=_J, domain=(0.0, 40.0), n_modes=3, phys_dim=4)


# -- the taxonomy is self-consistent -----------------------------------------
def test_registry_and_plan_compilers_are_the_two_dispatch_boundaries():
    """Every registered single-system engine has a plan compiler."""
    from fishbonett.models.simulation import PLAN_COMPILERS

    assert set(R.all_methods()) == set(R.METHODS)
    for name, spec in R.METHODS.items():
        assert spec.representation in R.REPRESENTATIONS, f"{name} names unknown representation {spec.representation!r}"
        assert spec.models, f"{name} belongs to no model"
        for mk in spec.models:
            assert mk in R.MODELS, f"{name} names unknown model {mk!r}"
            assert name in R.MODELS[mk].methods()
        assert spec.representation.count("-") == 1
        assert spec.state_geometry in R.STATE_GEOMETRIES, (
            f"{name} names unknown state geometry")
        if name == "interaction-chain-fishbone-tebd":
            assert spec.models == ("comb",)
            assert spec.state_geometry == "tree"
        # every single-system engine must resolve to one plan compiler
        if set(spec.models) & {"system-bath", "multichannel"}:
            assert callable(PLAN_COMPILERS.get(spec.engine)), (
                f"{name}: engine {spec.engine!r} has no plan compiler")

def test_state_geometry_vocabulary_is_explicit():
    assert set(R.STATE_GEOMETRIES) == {"mps", "binary-tree", "tree"}
    assert "path" not in R.STATE_GEOMETRIES
    assert "comb-tree" not in R.STATE_GEOMETRIES
    assert all(not hasattr(spec, "geometry") for spec in R.METHODS.values())


def test_high_level_run_signatures_are_uniform():
    import inspect

    signatures = {
        cls.__name__: tuple(inspect.signature(cls.run).parameters)
        for cls in (SystemBath, Fishbone, TreeFishbone)
    }
    assert len(set(signatures.values())) == 1, signatures


def test_bond_cap_requirements_are_registry_data():
    """Bond-cap requirements are part of each method specification."""
    assert R.BOND_CAP_REQUIRED_METHODS == frozenset(
        n for n, s in R.METHODS.items() if s.requires_bond_cap)
    assert "schrodinger-chain-tdvp1" in R.BOND_CAP_REQUIRED_METHODS
    assert "schrodinger-chain-dtdvp" in R.BOND_CAP_REQUIRED_METHODS
    assert "interaction-chain-tebd" not in R.BOND_CAP_REQUIRED_METHODS


def test_every_model_representation_pair_has_at_least_one_method():
    for key, m in R.MODELS.items():
        assert m.representations, f"model {key!r} declares no representations"
        for representation, methods in m.representations.items():
            assert representation in R.REPRESENTATIONS, f"{key!r} names unknown representation {representation!r}"
            assert methods, f"{key!r}/{representation!r} declares no methods"


def test_every_gap_has_a_reason():
    """Every declared model/representation gap has a reason."""
    for key, m in R.MODELS.items():
        for representation, why in m.gaps.items():
            assert representation in R.REPRESENTATIONS, f"{key!r} gap names unknown representation {representation!r}"
            assert why and why.strip(), f"{key!r}/{representation!r} gap has no reason"
            assert representation not in m.representations, (
                f"{key!r}/{representation!r} is listed both as available and as a gap")


def test_every_absent_representation_has_a_reason_available():
    """Every unavailable representation has an explanatory result."""
    for key, m in R.MODELS.items():
        for representation in R.REPRESENTATIONS:
            if representation in m.representations:
                continue
            why = R.why_not(key, representation)
            assert why, f"model {key!r} cannot explain why it has no {representation!r} representation"


def test_why_not_rejects_unknown_taxonomy_values():
    with pytest.raises(KeyError, match="unknown model"):
        R.why_not("not-a-model", "interaction-chain")
    with pytest.raises(KeyError, match="unknown representation"):
        R.why_not("system-bath", "not-a-representation")
    with pytest.raises(KeyError, match="unknown state_geometry"):
        R.why_not("system-bath", state_geometry="not-a-geometry")


def test_method_representations_is_a_lossless_projection():
    """The projection records representation only; model ownership is separate."""
    assert R.METHOD_REPRESENTATIONS["interaction-chain-tree-tebd"] == "interaction-chain"
    assert R.METHOD_REPRESENTATIONS["interaction-chain-tdvp2"] == "interaction-chain"
    # ...same representation, same model, same integrator -- separated only by state geometry
    assert R.METHODS["interaction-chain-tree-tebd"].state_geometry == "binary-tree"
    assert R.METHODS["interaction-chain-tdvp2"].state_geometry == "mps"

    assert R.METHOD_REPRESENTATIONS["schrodinger-chain-tdvp2"] == "schrodinger-chain"
    assert R.METHOD_REPRESENTATIONS["schrodinger-star-tdvp2"] == "schrodinger-star"

    # polaron is its own representation, not a Schrodinger sub-case
    assert R.METHOD_REPRESENTATIONS["polaron-chain-tebd"] == "polaron-chain"
    assert R.METHOD_REPRESENTATIONS["interaction-chain-trotter-mpo"] == "interaction-chain"


def test_multichannel_default_tree_is_schrodinger_not_interaction():
    """The default multichannel tree Hamiltonian is a Schrodinger star."""
    representations = R.MODELS["multichannel"].representations
    assert R.SCHRODINGER_STAR_TREE_TEBD in representations["schrodinger-star"]
    assert R.INTERACTION_CHAIN_TEBD in representations["interaction-chain"]
    assert R.INTERACTION_STAR_TEBD in representations["interaction-star"]
    assert R.SCHRODINGER_STAR_TREE_TEBD not in representations.get("interaction-star", ())
    assert set(representations) == {
        "schrodinger-star", "interaction-chain", "interaction-star"}

    mc = Bath(J=[_J, _J], domain=(0.0, 40.0),
              n_modes=3, phys_dim=4)
    fb = TreeFishbone(
        sites=[sigma_x], edges=[], baths=[mc.bind([sigma_z, sigma_x])])
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
        return Fishbone(
            sites=[h, h], baths=[_bath().bind(sigma_z), None],
        ), None
    if model_key == "site-tree":
        return TreeFishbone(
            sites=[h], edges=[], baths=[_bath().bind(sigma_z)],
        ), None
    raise AssertionError(model_key)


def test_multisite_baths_have_one_normalized_public_shape():
    coupled = _bath().bind(sigma_z)
    fishbone = Fishbone(
        sites=[sigma_x, sigma_x], baths=[coupled, None],
    )
    tree = TreeFishbone(
        sites=[sigma_x, sigma_x], edges=[(0, 1)], baths=[coupled, None],
    )
    for model in (fishbone, tree):
        assert len(model.baths) == 2
        assert isinstance(model.baths[0], list)
        assert isinstance(model.baths[0][0], CoupledBath)
        assert model.baths[1] == []


@pytest.mark.parametrize("model_key", sorted(R.MODELS))
def test_each_model_runs_its_own_methods_and_reports_them(model_key):
    obj, _ = _run_for(model_key)
    default = R.methods_of(model_key, "schrodinger-star")[0] if model_key == "multichannel" else None
    for method in R.methods_of(model_key):
        kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z})
        if method in _BOND_CAP_REQUIRED_METHODS:  # these require an explicit cap
            kw["bond_dim"] = 12
        if method == default:
            # the multichannel model is selected by its coupling list, so its Schrodinger
            # default method is what you get with no `method` at all
            r = obj.run(**kw)
        else:
            r = obj.run(method=method, **kw)
        assert r.t.shape == (2,)
        assert r.method == method, (
            f"{model_key}/{method} reported method={r.method!r}")


@pytest.mark.parametrize("model_key", sorted(R.MODELS))
def test_every_method_reports_max_bond(model_key):
    """Every method reports one maximum bond dimension per recorded step."""
    obj, _ = _run_for(model_key)
    default = R.methods_of(model_key, "schrodinger-star")[0] if model_key == "multichannel" else None
    for method in R.methods_of(model_key):
        kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z})
        if method in _BOND_CAP_REQUIRED_METHODS:
            kw["bond_dim"] = 12
        r = obj.run(**kw) if method == default else obj.run(method=method, **kw)
        assert r.max_bond is not None, f"{model_key}/{method} reports no max_bond"
        assert np.shape(r.max_bond) == (2,), f"{model_key}/{method}: one per step"
        assert np.all(np.asarray(r.max_bond) >= 1), f"{model_key}/{method}"


def test_result_shape_contract_matches_its_docstring():
    """Single- and multi-site results follow their documented shapes."""
    h, kw = 0.5 * sigma_x, dict(dt=0.02, n_steps=2, trunc_eps=1e-7,
                                observables={"sz": sigma_z})

    single = SystemBath(h=h, coupling=sigma_z, bath=_bath_pos()).run(
        method="interaction-chain-tebd", **kw)
    assert "n_sites" not in single.meta
    assert single.meta["method"] == "interaction-chain-tebd"
    assert single.meta["dt"] == kw["dt"]
    assert np.shape(single.rdm) == (2, 2, 2), "(n_steps, d, d)"
    assert np.shape(single.expect["sz"]) == (2,), "(n_steps,)"

    for n_sites, obj in (
        (1, TreeFishbone(
            sites=[h], edges=[], baths=[_bath_pos().bind(sigma_z)],
        )),
        (2, Fishbone(
            sites=[h, h], baths=[_bath_pos().bind(sigma_z), None],
        )),
        (3, TreeFishbone(sites=[h, h, h], edges=[(0, 1), (1, 2)],
                         baths=[_bath_pos().bind(sigma_z), None, None])),
    ):
        r = obj.run(**kw)
        assert r.meta["n_sites"] == n_sites
        assert r.meta["dt"] == kw["dt"]
        assert np.shape(r.rdm) == (2, n_sites, 2, 2), "(n_steps, n_sites, d, d)"
        # a bare per-site operator gives one column per site
        assert np.shape(r.expect["sz"]) == (2, n_sites), "(n_steps, n_sites)"


def test_every_method_agrees_on_the_same_physics():
    """Equivalent representations and geometries agree within numerical error."""
    h, coup = 0.5 * sigma_x, sigma_z
    dt, n_steps = 0.02, 20

    def run(model_key, method):
        if model_key == "site-tree":
            obj = TreeFishbone(
                sites=[h], edges=[], baths=[_bath_pos().bind(sigma_z)],
            )
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

    ref = results[("system-bath", "interaction-chain-tebd")]
    for (key, m), sz in results.items():
        # schrodinger-chain-dtdvp is bond-adaptive: its accuracy is set by `prec`, not by dt, so
        # it sits a decade or so off the rest.  Still far from a wrong answer.
        tol = 5e-3 if m == "schrodinger-chain-dtdvp" else 1e-3
        assert np.abs(sz - ref).max() < tol, (
            f"{key}/{m} disagrees with system-bath/interaction-chain-tebd by "
            f"{np.abs(sz - ref).max():.2e} on identical physics")


def test_swap_engine_matches_what_the_driver_actually_does():
    """Every method routed to the swap engine uses the shared swap driver."""
    import inspect
    from fishbonett.models.simulation import PLAN_COMPILERS

    declared = {n for n, s in R.METHODS.items() if s.engine == "swap-tebd"}
    actual = set()
    for name, spec in R.METHODS.items():
        compiler = PLAN_COMPILERS.get(spec.engine)
        if compiler is None:
            continue
        if "symmetric_swap_step" in inspect.getsource(compiler):
            actual.add(name)
    assert declared == actual, (
        f"swap engine methods {sorted(declared)} differ from drivers that call "
        f"symmetric_swap_step: {sorted(actual)}")

    # Mode-decoupled TEBD interactions on an MPS use the swap engine.
    assert all(R.REPRESENTATIONS[R.METHODS[n].representation].mode_decoupled
               and R.METHODS[n].state_geometry == "mps" for n in declared)
    assert R.METHODS["interaction-chain-tebd"].representation == "interaction-chain"

    # All swap methods share one engine.
    swap_engines = {R.METHODS[n].engine for n in declared}
    assert len(swap_engines) == 1, (
        "swap methods should have one driver, not one per representation")
    assert "symmetric_swap_step" in inspect.getsource(
        PLAN_COMPILERS[next(iter(swap_engines))])


def test_run_takes_the_axes_directly():
    """A method name and its four explicit axes select the same run."""
    kw = dict(dt=0.02, n_steps=2, observables={"sz": sigma_z}, trunc_eps=1e-7)
    by_axes = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath()).run(
        representation="interaction-chain", state_geometry="mps",
        integrator="tdvp2", **kw)
    by_name = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath()).run(
        method="interaction-chain-tdvp2", **kw)
    assert by_axes.method == by_name.method == "interaction-chain-tdvp2"
    assert np.array_equal(by_axes.rdm, by_name.rdm)

    # the axis vocabulary is uniform across representations: "tebd" means the same word
    # whether the representation dresses the state or not
    sb = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath())
    assert sb.run(representation="polaron-chain", integrator="tebd", **kw).method == "polaron-chain-tebd"
    assert sb.run(representation="interaction-chain", state_geometry="mps",
                  integrator="tebd",
                  **kw).method == "interaction-chain-tebd"
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
    """Interaction-chain methods use star-to-chain transformed couplings."""
    from fishbonett.bath.chain import star_transform
    freq, Vn, coefT = star_transform(_J, 6, (0.0, 40.0))

    d0 = np.abs(coefT @ (Vn * np.exp(-1j * freq * 0.0)))
    assert np.isclose(d0[0], np.linalg.norm(Vn))       # all of it on one site
    assert np.allclose(d0[1:], 0.0)                    # ...and none anywhere else
    assert (np.abs(Vn) > 1e-3).sum() > 1               # the star is *not* like that

    d1 = np.abs(coefT @ (Vn * np.exp(-1j * freq * 0.5)))
    assert (d1 > 1e-3).sum() > 1, "the coupling must spread as t grows"

    for m in ("interaction-chain-tebd", "interaction-chain-trotter-mpo",
              "interaction-chain-tdvp1", "interaction-chain-tdvp2",
              "interaction-chain-tree-tebd"):
        assert R.METHODS[m].representation == "interaction-chain", m
    # Multichannel exposes the same distinction explicitly.
    assert R.METHODS[R.INTERACTION_CHAIN_TEBD].representation == "interaction-chain"
    assert R.METHODS[R.INTERACTION_STAR_TEBD].representation == "interaction-star"
    assert R.METHODS["interaction-star-tdvp2"].representation == "interaction-star"


def test_the_two_interaction_representations_agree():
    """Interaction-chain and interaction-star trajectories agree."""
    h, kw = 0.5 * sigma_x, dict(dt=0.02, n_steps=20, bond_dim=40,
                                trunc_eps=1e-10, observables={"sz": sigma_z})
    def run(m):
        sb = SystemBath(h=h, coupling=sigma_z, bath=_bath_pos())
        return np.asarray(sb.run(method=m, **kw).expect["sz"])
    chain, star = run("interaction-chain-tdvp2"), run("interaction-star-tdvp2")
    assert np.abs(chain - star).max() < 1e-3, np.abs(chain - star).max()


def test_axis_errors_name_what_separates_the_candidates():
    kw = dict(dt=0.02, n_steps=2, trunc_eps=1e-7)
    sb = lambda: SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_bath())

    # A partial name is not a public representation.
    with pytest.raises(ValueError, match="no method"):
        sb().run(representation="schrodinger", **kw)
    # ...and one representation spanning two state geometries is too
    with pytest.raises(ValueError, match="ambiguous"):
        sb().run(representation="interaction-chain", **kw)
    # An exact representation still needs enough axes to choose its integrator.
    with pytest.raises(ValueError, match="ambiguous"):
        sb().run(representation="polaron-star", **kw)
    # a chain representation on a binary tree: the one state-geometry constraint left
    with pytest.raises(ValueError, match="without mode-mode terms"):
        sb().run(representation="schrodinger-chain",
                 state_geometry="binary-tree", **kw)
    # a name already fixes all four axes, so mixing spellings is a mistake
    with pytest.raises(ValueError, match="not both"):
        sb().run(method="interaction-chain-tebd", representation="polaron-chain", **kw)
    # Representation and application details are not additional axes.
    for gone in ("basis", "frame", "layout"):
        with pytest.raises(TypeError, match="unknown axis"):
            R.resolve({"system-bath"}, **{gone: "star"})
    with pytest.raises(TypeError):
        sb().run(geometry="path", **kw)
    with pytest.raises(TypeError, match="unknown axis"):
        R.resolve({"system-bath"}, geometry="path")
    for unknown in ("path", "comb-tree"):
        with pytest.raises(ValueError, match="available state geometries"):
            sb().run(state_geometry=unknown, **kw)
        with pytest.raises(ValueError, match="available state geometries"):
            R.resolve({"system-bath"}, state_geometry=unknown)


def test_unknown_model_error_lists_current_models():
    with pytest.raises(KeyError, match="system-bath"):
        R.model("unknown")


def test_every_method_is_reachable_by_its_axes():
    """Whatever the registry declares must be selectable by its axes.

    Every method must be pinned down by the four together -- if two rows shared all
    four they would be the same run under two names."""
    seen = {}
    for mk in R.MODELS:
        for name in R.methods_of(mk):
            spec = R.METHODS[name]
            axes = dict(model=mk, representation=spec.representation,
                        state_geometry=spec.state_geometry,
                        integrator=spec.integrator)
            got = R.resolve(set(R.MODELS), **axes)
            assert got.name == name, f"{axes} -> {got.name}"
            key = tuple(sorted(axes.items()))
            assert key not in seen, f"{name} and {seen[key]} share all four axes"
            seen[key] = name


@pytest.mark.parametrize("model_key", ["comb", "site-tree"])
def test_multi_site_models_reject_a_single_system_method(model_key):
    """A multi-site model reports the owner of a single-system method."""
    obj, _ = _run_for(model_key)
    with pytest.raises(ValueError, match="belongs to system-bath"):
        obj.run(dt=0.02, n_steps=1, method="interaction-chain-tebd")


def test_multichannel_model_rejects_another_models_method():
    """A multichannel model rejects methods owned by another model."""
    mc = Bath(J=[_J, _J], domain=(0.0, 40.0),
              n_modes=3, phys_dim=4)
    m = SystemBath(h=0.5 * sigma_x, coupling=[sigma_z, sigma_x], bath=mc)
    with pytest.raises(ValueError, match="system-bath"):
        m.run(dt=0.02, n_steps=1, method="schrodinger-chain-tdvp2")


def test_multi_site_models_report_max_bond():
    """Multi-site results report maximum bond dimensions."""
    r = TreeFishbone(sites=[0.5 * sigma_x], edges=[],
                     baths=[_bath().bind(sigma_z)]).run(dt=0.02, n_steps=3)
    assert r.max_bond is not None and r.max_bond.shape == (3,)
    assert np.all(r.max_bond >= 1)


# -- lookups ------------------------------------------------------------------
def test_models_of_reports_every_owner():
    assert set(R.models_of("interaction-chain-tebd")) == {
        "system-bath", "multichannel"}
    # the static tree engine serves both multi-site models
    assert set(R.models_of(R.SCHRODINGER_CHAIN_TREE_TEBD)) == {"comb", "site-tree"}
    assert R.models_of("not-a-method") == ()


def test_methods_of_explains_a_gap_instead_of_a_bare_keyerror():
    """An absent representation reports the registry's reason."""
    reason = R.MODELS["comb"].gaps["polaron-chain"]
    with pytest.raises(KeyError, match=re.escape(reason)):
        R.methods_of("comb", "polaron-chain")


def test_one_engine_can_serve_two_representations():
    """One tree engine can consume distinct Schrodinger representations."""
    static = R.METHODS[R.SCHRODINGER_CHAIN_TREE_TEBD]
    mc = R.METHODS[R.SCHRODINGER_STAR_TREE_TEBD]
    assert static.engine == mc.engine == "static-tree-tebd"
    assert static.state_geometry == mc.state_geometry == "tree"
    assert (static.representation, mc.representation) == ("schrodinger-chain", "schrodinger-star")
    assert static.models == ("comb", "site-tree") and mc.models == ("multichannel",)


def test_describe_taxonomy_mentions_every_model_and_method():
    text = R.describe_taxonomy()
    for key, m in R.MODELS.items():
        assert m.label in text
        for method in m.methods():
            assert method in text


#: ``state_geometry`` -> the infix a method name carries.  Two geometries are
#: trees, so "insert tree" is not a rule that can name both.
GEOMETRY_INFIX = {"mps": "", "binary-tree": "tree", "tree": "tree"}

#: Methods whose names need a qualifier to avoid a collision.
#:
#: ``"collision"``  the derived name is already taken by another method.  Both tree
#:                  geometries take the infix "tree", so ``interaction-chain`` +
#:                  ``tebd`` collides with the binary-tree method and the comb one
#:                  is named "fishbone" instead.
#: ``"sibling"``    the derived name is free, but the method shares a model and a
#:                  representation with a ``"collision"`` exception and is named to
#:                  match it, so the two read alike where a user meets them.
NAME_EXCEPTIONS = {
    "interaction-chain-fishbone-tebd":
        ("interaction-chain", "tree", "tebd", "collision"),
    "interaction-chain-fishbone-trotter-mpo":
        ("interaction-chain", "tree", "trotter-mpo", "sibling"),
    # The unqualified name belongs to the binary-tree method.
    "interaction-chain-fishbone-tdvp2":
        ("interaction-chain", "tree", "tdvp2", "sibling"),
}


def test_every_method_name_is_derivable_from_its_axes():
    """Each method name is derivable from its axes and recorded qualifier."""
    assert set(GEOMETRY_INFIX) == {s.state_geometry for s in R.METHODS.values()}, (
        "a new state_geometry needs an infix here and in docs/architecture.md")
    for name, spec in R.METHODS.items():
        if name in NAME_EXCEPTIONS:
            *axes, reason = NAME_EXCEPTIONS[name]
            assert tuple(axes) == (
                spec.representation, spec.state_geometry, spec.integrator), name
            infix = GEOMETRY_INFIX[spec.state_geometry]
            derived = "-".join([spec.representation, infix, spec.integrator])
            taken = derived in R.METHODS and R.METHODS[derived] is not spec
            if reason == "collision":
                assert taken, (
                    f"{name} claims a collision but nothing holds {derived!r} "
                    f"-- rename it to the derived name instead")
            else:
                assert reason == "sibling", f"{name}: unknown reason {reason!r}"
                assert not taken, (
                    f"{name} claims to be named after a sibling, but {derived!r} "
                    f"is taken -- it is really a collision")
                qualifier = name.rsplit(spec.integrator, 1)[0]
                kin = [other for other, value in NAME_EXCEPTIONS.items()
                       if value[-1] == "collision"
                       and other.startswith(qualifier)
                       and value[0] == spec.representation]
                assert kin, (
                    f"{name} is named after a sibling, but no 'collision' "
                    f"exception shares the prefix {qualifier!r}")
                assert set(R.METHODS[name].models) & set(R.METHODS[kin[0]].models), (
                    f"{name} and {kin[0]} do not share a model")
            continue
        infix = GEOMETRY_INFIX[spec.state_geometry]
        parts = [spec.representation] + ([infix] if infix else []) + [spec.integrator]
        assert name == "-".join(parts), name


def test_no_gaps_does_not_mean_every_integrator_exists():
    """Representation availability does not imply every integrator is available."""
    assert R.MODELS["system-bath"].gaps == {}
    have = {(s.representation, s.integrator) for s in R.METHODS.values()}
    assert ("interaction-chain", "trotter-mpo") in have
    for rep in ("schrodinger-chain", "schrodinger-star", "polaron-chain",
                "polaron-star", "interaction-star"):
        assert (rep, "trotter-mpo") not in have, rep
