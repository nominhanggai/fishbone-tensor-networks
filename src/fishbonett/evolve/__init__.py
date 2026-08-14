"""Propagation algorithms -- what actually advances a state in time.

These act on the state ansaetze in :mod:`fishbonett.states`.  Two axes decide
which module you want: the **geometry** of the state, and the **integrator**.

.. rubric:: Which module for which geometry

Every propagator here is written for one geometry; none of them is generic.
In particular the projector-splitting sweeps are **1D-chain only**: they act on a
linear MPS and MPO. For a branching geometry use the graph-generic tree-operator
module instead.

====================  ==================================  =====================
geometry              module                              integrators
====================  ==================================  =====================
1D chain (MPS)        :mod:`~fishbonett.evolve.tebd`      TEBD (Trotter gates)
1D chain (MPS + MPO)  :mod:`~fishbonett.evolve.tdvp`      TDVP 1/2-site, adaptive
binary tree of modes  :mod:`~fishbonett.evolve.modetree`  TTNO + Schmidt truncation
any tree (incl comb)  :mod:`~fishbonett.evolve.sitetree`  TEBD
====================  ==================================  =====================

The last row used to point into :mod:`fishbonett.states`, which is where that
algorithm lived; it is here now, so *every* gate application is in this package and
``states`` holds only tensors and canonical form.  The comb is not a separate row:
it is a tree, and :class:`fishbonett.models.fishbone.Fishbone` propagates it through
``sitetree`` like any other.

Note the two tree rows are different **models**, not two integrators for one
geometry: ``modetree`` is a balanced binary tree of *bath modes* around one system
site (interaction picture, ``tree-tebd``), while ``sitetree`` is an arbitrary tree
of *system sites* each with its own bath (Schroedinger, ``tree-tebd-static``).  See
:mod:`fishbonett.models.registry`.

.. rubric:: Layers

The public ``tdvp`` and ``modetree`` modules are facades over private
implementation layers. You can enter through the documented public functions at
the level you need:

*primitive*
    one bond or one site: :func:`~fishbonett.evolve.tebd.update_bond`,
    :func:`~fishbonett.evolve.sitetree.apply_edge`, ``tdvp.applyH1``/``applyH0``.
*sweep / graph operation*
    one pass over the state: :func:`~fishbonett.evolve.tebd.sweep`,
    ``tdvp.tdvp1sweep``, or ``modetree.apply_coupling`` followed by
    ``modetree.truncate_from_root``.
*whole step / driver*
    one symmetric time step, or a whole simulation:
    :func:`~fishbonett.evolve.tebd.symmetric_swap_step`,
    :func:`~fishbonett.evolve.sitetree.symmetric_tree_step`,
    :func:`run_mpo_frame`, ...

Every whole step here is **second order** (Strang): each takes half-step gates and
applies them in palindromic order.

The private split enforces dependency direction: kernels know only tensor algebra,
sweeps depend on kernels, and whole-run drivers depend on both.  Neither kernels
nor sweeps resolve a bath or import a Hamiltonian frame.

:func:`run_mpo_frame` takes a :class:`~fishbonett.frames.mpo.MPOFrame` -- the
Hamiltonian, already built -- plus a sweep name, and runs the whole simulation.  It
replaced seven ``run_*`` functions (``run_tdvp1``, ``run_star_tdvp2``,
``run_ip_tdvp1``, ...), one per *(MPO builder, sweep)* pair, which each built their
own Hamiltonian and repeated the same loop.  Building a Hamiltonian is a frame
question, so nothing here imports :mod:`fishbonett.frames` any more.

For ordinary use go through
:meth:`fishbonett.models.system_bath.SystemBath.run`, which handles bath
resolution, initial states and observables; call a driver here directly when you
want one engine in isolation, e.g. for a benchmark.
"""
from fishbonett.evolve.tebd import (
    update_bond, sweep, swap_in, swap_out,
    symmetric_swap_step, symmetric_static_step,
)
from fishbonett.evolve.tdvp import run_mpo_frame
from fishbonett.evolve.modetree import (
    run_tree_tdvp, run_tree_tdvp2, run_tree_tebd,
)
from fishbonett.evolve.mpo_apply import apply_mpo, compress

__all__ = [
    # TEBD (1D chain)
    "update_bond", "sweep", "swap_in", "swap_out",
    "symmetric_swap_step", "symmetric_static_step",
    # MPO application (1D chain)
    "apply_mpo", "compress",
    # TDVP driver (1D chain): one loop, any frame, any sweep
    "run_mpo_frame",
    # tree drivers
    "run_tree_tdvp", "run_tree_tdvp2", "run_tree_tebd",
]
