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


def test_tdvp_facade_preserves_established_entry_points():
    from fishbonett.evolve import tdvp
    from fishbonett.evolve import _tdvp_driver as driver
    from fishbonett.evolve import _tdvp_kernels as kernels
    from fishbonett.evolve import _tdvp_sweeps as sweeps

    assert tdvp.applyH1 is kernels.applyH1
    assert tdvp.expmv_lanczos is kernels.expmv_lanczos
    assert tdvp.tdvp1sweep is sweeps.tdvp1sweep
    assert tdvp.tdvp2sweep is sweeps.tdvp2sweep
    assert tdvp.tdvp1sweep_dynamic is sweeps.tdvp1sweep_dynamic
    assert tdvp.run_mpo_frame is driver.run_mpo_frame


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
    assert modetree.truncate_from_root is sweeps.truncate_from_root
    assert modetree.run_tree_mpo is driver.run_tree_mpo
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
