"""Propagation algorithms -- what actually advances a state in time.

These act on the state ansaetze in :mod:`fishbonett.states`.  Two axes decide
which module you want: the **geometry** of the state, and the **integrator**.

.. rubric:: Which module for which geometry

Every propagator here is written for one geometry; none of them is generic.
In particular :func:`run_tdvp1` and its siblings are **1D-chain only** -- they
build a chain MPO over a linear MPS.  For a branching geometry use the tree
module instead; the algorithm is the same idea, the contractions are not.

====================  ===================================  ====================
geometry              module                               integrators
====================  ===================================  ====================
1D chain (MPS)        :mod:`~fishbonett.evolve.tebd`       TEBD (Trotter gates)
1D chain (MPS + MPO)  :mod:`~fishbonett.evolve.tdvp`       TDVP 1/2-site, DTDVP
binary tree of modes  :mod:`~fishbonett.evolve.treetdvp`   TDVP 1/2-site, TEBD
arbitrary tree        :mod:`~fishbonett.evolve.tebd_tree`  TEBD
comb / fishbone       :mod:`~fishbonett.evolve.tebd_comb`  TEBD
====================  ===================================  ====================

The last two rows used to point into :mod:`fishbonett.states`, which is where those
algorithms lived; they are here now, so *every* gate application is in this package
and ``states`` holds only tensors and canonical form.

Note the two tree rows are different **models**, not two integrators for one
geometry: ``treetdvp`` is a balanced binary tree of *bath modes* around one system
site (interaction picture, ``tree-tebd``), while ``tebd_tree`` is an arbitrary tree
of *system sites* each with its own bath (Schroedinger, ``tree-tebd-static``).  See
:mod:`fishbonett.models.registry`.

.. rubric:: Layers

Each module is layered primitive -> sweep -> driver, so you can enter at
whichever level you need:

*primitive*
    one bond or one site: :func:`~fishbonett.evolve.tebd.update_bond`,
    :func:`~fishbonett.evolve.tebd_tree.apply_edge`, ``tdvp.applyH1``/``applyH0``.
*sweep*
    one pass over the state: :func:`~fishbonett.evolve.tebd.sweep`,
    ``tdvp.tdvp1sweep``, ``treetdvp.tdvp_sweep``.
*whole step / driver*
    one symmetric time step, or a whole simulation:
    :func:`~fishbonett.evolve.tebd.symmetric_swap_step`,
    :func:`~fishbonett.evolve.tebd_tree.symmetric_tree_step`,
    :func:`run_tdvp1`, ...

Every whole step here is **second order** (Strang): each takes half-step gates and
applies them in palindromic order.

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
