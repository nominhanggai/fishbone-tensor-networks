"""Propagation algorithms -- what actually advances a state in time.

These algorithms act on the state ansaetze in :mod:`fishbonett.states`. The
tensor-network geometry and integrator determine the propagation module.

.. rubric:: Propagation modules

Projector-splitting sweeps act on a one-dimensional MPS and MPO. Branching tensor
networks use one of the tree modules.

=======================  ==================================  =====================
tensor-network geometry  module                              integrators
=======================  ==================================  =====================
1D chain (MPS)        :mod:`~fishbonett.evolve.tebd`      TEBD (Trotter gates)
1D chain (MPS + MPO)  :mod:`~fishbonett.evolve.tdvp`      TDVP 1/2-site, adaptive
binary tree of modes  :mod:`~fishbonett.evolve.modetree`  TTNO + Schmidt truncation
any tree (incl comb)  :mod:`~fishbonett.evolve.sitetree`  TEBD
=======================  ==================================  =====================

Note the two tree modules serve different geometries, not two integrators for one
graph: ``modetree`` is a balanced binary tree of *bath modes* around one system
site (interaction representation, ``interaction-chain-tree-tebd``), while
``sitetree`` is an arbitrary tree of *system sites* and their baths
(``schrodinger-chain-tree-tebd`` for the multi-site models).  See
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
    :func:`run_mpo_hamiltonian`, ...

Every whole step here is **second order** (Strang): each takes half-step gates and
applies them in palindromic order.

Private kernels implement tensor algebra, sweeps compose the kernels, and whole-run
drivers compose the sweeps. Kernels and sweeps do not resolve baths or import
Hamiltonian representations.

:func:`run_mpo_hamiltonian` takes a representation exposing ``tdvp_mpo`` -- the
Hamiltonian, already built -- plus a sweep name, and runs the whole simulation.
Building the engine-facing operator is a representation concern, so this module
does not import concrete representation classes.

For ordinary use go through
:meth:`fishbonett.models.system_bath.SystemBath.run`, which handles bath
resolution, initial states and observables; call a driver here directly when you
want one engine in isolation, e.g. for a benchmark.
"""
from fishbonett.evolve.tebd import (
    update_bond, sweep, swap_in, swap_out,
    symmetric_swap_step, symmetric_static_step,
)
from fishbonett.evolve.tdvp import run_mpo_hamiltonian
from fishbonett.evolve.modetree import run_tree_tebd
from fishbonett.evolve.mpo_apply import apply_mpo, compress

__all__ = [
    # TEBD (1D chain)
    "update_bond", "sweep", "swap_in", "swap_out",
    "symmetric_swap_step", "symmetric_static_step",
    # MPO application (1D chain)
    "apply_mpo", "compress",
    # TDVP driver (1D chain): one loop, any representation, any sweep
    "run_mpo_hamiltonian",
    # tree drivers
    "run_tree_tebd",
]
