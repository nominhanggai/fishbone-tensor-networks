"""Propagation algorithms -- what actually advances a state in time.

These act on the state ansaetze in :mod:`fishbonett.states`.  Two axes decide
which module you want: the **geometry** of the state, and the **integrator**.

.. rubric:: Which module for which geometry

Every propagator here is written for one geometry; none of them is generic.
In particular :func:`run_tdvp1` and its siblings are **1D-chain only** -- they
build a chain MPO over a linear MPS.  For a branching geometry use the tree
module instead; the algorithm is the same idea, the contractions are not.

=========================  ==========================  =========================
geometry                   module                      integrators
=========================  ==========================  =========================
1D chain (MPS)             :mod:`~fishbonett.evolve.tebd`   TEBD (Trotter gates)
1D chain (MPS + MPO)       :mod:`~fishbonett.evolve.tdvp`   TDVP 1-site / 2-site / DTDVP
binary tree (TTN)          :mod:`~fishbonett.evolve.treetdvp`  TDVP 1-site / 2-site, TEBD
arbitrary tree             :class:`fishbonett.states.tree.TreeTEBD`  TEBD
comb / fishbone            :mod:`fishbonett.states.comb`    TEBD
=========================  ==========================  =========================

.. rubric:: Layers

Each module is layered primitive -> sweep -> driver, so you can enter at
whichever level you need:

*primitive*
    one bond or one site: :func:`~fishbonett.evolve.tebd.update_bond`,
    ``tdvp.applyH1``/``applyH0``.
*sweep*
    one pass over the state: :func:`~fishbonett.evolve.tebd.sweep`,
    ``tdvp.tdvp1sweep``, ``treetdvp.tdvp_sweep``.
*whole step / driver*
    one symmetric time step, or a whole simulation:
    :func:`~fishbonett.evolve.tebd.symmetric_swap_step`, :func:`run_tdvp1`, ...

The ``run_*`` drivers are self-contained convenience entry points: they take a
*spectral density* and build their own chain, state and operator.  For ordinary
use go through :meth:`fishbonett.models.system_bath.SystemBath.run`, which handles bath
resolution, initial states and observables; reach for a ``run_*`` driver when you
want one engine in isolation, e.g. for a benchmark.
"""
from fishbonett.evolve.tebd import (
    update_bond, sweep, swap_in, swap_out,
    symmetric_swap_step, symmetric_static_step,
)
from fishbonett.evolve.tdvp import (
    run_tdvp1, run_tdvp2, run_dtdvp, run_ip_tdvp1, run_ip_tdvp2,
    run_star_tdvp1, run_star_tdvp2,
)
from fishbonett.evolve.treetdvp import (
    run_tree_tdvp, run_tree_tdvp2, run_tree_tebd,
)
from fishbonett.evolve.mpo_apply import apply_mpo, compress

__all__ = [
    # TEBD (1D chain)
    "update_bond", "sweep", "swap_in", "swap_out",
    "symmetric_swap_step", "symmetric_static_step",
    # MPO application (1D chain)
    "apply_mpo", "compress",
    # TDVP drivers (1D chain)
    "run_tdvp1", "run_tdvp2", "run_dtdvp", "run_ip_tdvp1", "run_ip_tdvp2",
    "run_star_tdvp1", "run_star_tdvp2",
    # tree drivers
    "run_tree_tdvp", "run_tree_tdvp2", "run_tree_tebd",
]
