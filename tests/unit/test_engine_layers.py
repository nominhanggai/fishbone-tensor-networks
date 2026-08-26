"""Dependency contracts for the numerical-engine layers."""
import ast
import inspect

import numpy as np


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
    assert tdvp.a1tdvp_sweep is sweeps.a1tdvp_sweep
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


def test_transformed_representations_do_not_import_propagation_layers():
    """Representations materialize operators but never advance tensor states."""
    import importlib
    import pathlib

    forbidden_prefixes = ("fishbonett.evolve", "fishbonett.states")
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "fishbonett" / "representations"
    checked = set()
    for path in sorted(root.glob("*.py")):
        if path.stem.startswith("__"):
            continue
        module = importlib.import_module(f"fishbonett.representations.{path.stem}")
        checked.add(path.stem)
        offenders = {n for n in _imports(module) if n.startswith(forbidden_prefixes)}
        assert not offenders, f"{module.__name__} imports {sorted(offenders)}"

    # the sweep must actually have covered every public representation module
    assert {"interaction", "multichannel", "polaron", "schrodinger"} <= checked


def test_tree_contractions_reuse_shape_compiled_expressions():
    from fishbonett.contract import _cached_expression, _contract_cached

    left = np.arange(6.0).reshape(2, 3)
    right = np.arange(12.0).reshape(3, 4)
    _cached_expression.cache_clear()
    first = _contract_cached(left, [0, 1], right, [1, 2], [0, 2])
    after_first = _cached_expression.cache_info()
    second = _contract_cached(left + 1, [0, 1], right, [1, 2], [0, 2])
    after_second = _cached_expression.cache_info()

    assert np.allclose(first, left @ right)
    assert np.allclose(second, (left + 1) @ right)
    assert after_first.misses == 1
    assert after_second.hits == 1


def test_package_contractions_do_not_bypass_opt_einsum():
    """Numerical package code must use the shared contraction backend."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "fishbonett"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        numpy_aliases = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "numpy"
        }
        direct_aliases = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "numpy"
            for alias in node.names
            if alias.name == "einsum"
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            direct = isinstance(node.func, ast.Name) and node.func.id in direct_aliases
            qualified = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "einsum"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in numpy_aliases
            )
            if direct or qualified:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, "direct numpy.einsum calls: " + ", ".join(offenders)
