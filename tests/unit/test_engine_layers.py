"""Dependency and compatibility contracts for the split numerical engines."""
import ast
import inspect


def _imports(module):
    tree = ast.parse(inspect.getsource(module))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


def test_tdvp_facade_exports_snake_case_entry_points():
    from fishbonett.evolve import tdvp
    from fishbonett.evolve import _tdvp_driver as driver
    from fishbonett.evolve import _tdvp_kernels as kernels
    from fishbonett.evolve import _tdvp_sweeps as sweeps

    assert tdvp.apply_h1 is kernels.applyH1
    assert tdvp.apply_h0 is kernels.applyH0
    assert tdvp.apply_h2 is sweeps.applyH2
    assert tdvp.evolve_site_tensor is kernels.evolveAC
    assert tdvp.evolve_bond_tensor is kernels.evolveC
    assert tdvp.update_left_environment is kernels.updateleftenv
    assert tdvp.update_right_environment is kernels.updaterightenv
    for retired in (
        "applyH0", "applyH1", "applyH2", "evolveAC", "evolveC",
        "updateleftenv", "updaterightenv",
    ):
        assert not hasattr(tdvp, retired)
    assert tdvp.expmv_lanczos is kernels.expmv_lanczos
    assert tdvp.tdvp1sweep is sweeps.tdvp1sweep
    assert tdvp.tdvp2sweep is sweeps.tdvp2sweep
    assert tdvp.tdvp1sweep_dynamic is sweeps.tdvp1sweep_dynamic
    assert tdvp.run_mpo_hamiltonian is driver.run_mpo_hamiltonian


def test_tdvp_dependencies_point_from_driver_to_sweep_to_kernel():
    from fishbonett.evolve import _tdvp_driver as driver
    from fishbonett.evolve import _tdvp_kernels as kernels
    from fishbonett.evolve import _tdvp_sweeps as sweeps

    assert not {"fishbonett.evolve._tdvp_sweeps",
                "fishbonett.evolve._tdvp_driver"} & _imports(kernels)
    assert "fishbonett.evolve._tdvp_kernels" in _imports(sweeps)
    assert "fishbonett.evolve._tdvp_driver" not in _imports(sweeps)
    assert "fishbonett.evolve._tdvp_sweeps" in _imports(driver)


def test_modetree_facade_preserves_established_entry_points():
    from fishbonett.evolve import modetree
    from fishbonett.evolve import _modetree_core as core
    from fishbonett.evolve import _modetree_driver as driver
    from fishbonett.evolve import _modetree_sweeps as sweeps

    assert modetree.Node is core.Node
    assert modetree.build_tree_mpo is core.build_tree_mpo
    assert modetree.apply_op_node is sweeps.apply_op_node
    assert modetree.contract_center is sweeps.contractC
    assert not hasattr(modetree, "contractC")
    assert modetree.truncate_from_root is sweeps.truncate_from_root
    assert modetree.run_tree_tebd is driver.run_tree_tebd


def test_modetree_dependencies_point_from_driver_to_sweep_and_core():
    from fishbonett.evolve import _modetree_core as core
    from fishbonett.evolve import _modetree_driver as driver
    from fishbonett.evolve import _modetree_sweeps as sweeps

    assert not {"fishbonett.evolve._modetree_sweeps",
                "fishbonett.evolve._modetree_driver"} & _imports(core)
    assert "fishbonett.evolve._modetree_core" in _imports(sweeps)
    assert "fishbonett.evolve._modetree_driver" not in _imports(sweeps)
    assert "fishbonett.evolve._modetree_core" in _imports(driver)
    assert "fishbonett.evolve._modetree_sweeps" in _imports(driver)


#: ``representations/coolingchain.py`` is exempt because it is not a
#: representation builder at all: ``SystemBathCoolingChain`` *subclasses*
#: ``states.mps.SystemBathMPS``, so it is a state that happens to live in this
#: package.  It is not one of the six entries in ``registry.REPRESENTATIONS``.
#: Relocating it to ``states/`` would remove the exemption; until then it is
#: named here so the exemption is a decision rather than an oversight.
LAYERING_EXEMPT = {"coolingchain"}


def test_transformed_representations_do_not_import_propagation_layers():
    """Representations materialize operators but never advance tensor states.

    Discovered by globbing the package rather than from a hand-written module
    list: a list silently stops covering whatever is added next, and the module
    that actually violates this invariant was missing from the previous one.
    """
    import importlib
    import pathlib

    forbidden_prefixes = ("fishbonett.evolve", "fishbonett.states")
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "fishbonett" / "representations"
    checked = set()
    for path in sorted(root.glob("*.py")):
        if path.stem.startswith("__") or path.stem in LAYERING_EXEMPT:
            continue
        module = importlib.import_module(f"fishbonett.representations.{path.stem}")
        checked.add(path.stem)
        offenders = {n for n in _imports(module) if n.startswith(forbidden_prefixes)}
        assert not offenders, f"{module.__name__} imports {sorted(offenders)}"

    # the sweep must actually have covered the six public representations
    assert {"interaction", "multichannel", "polaron", "schrodinger"} <= checked


def test_layering_exemptions_are_real_modules():
    """An exemption for a module that no longer exists silently weakens the sweep."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "fishbonett" / "representations"
    for stem in LAYERING_EXEMPT:
        assert (root / f"{stem}.py").exists(), stem
