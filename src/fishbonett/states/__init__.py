"""Tensor-network state containers and canonical-form operations.

The package provides three related state types:

* :class:`~fishbonett.states.network.TensorNetwork` provides topology,
  canonical-form operations and observables for loop-free tensor networks.
* :class:`~fishbonett.states.mps.SystemBathMPS` -- the path case, in Vidal
  (``Gamma-Lambda``) form with singular values on bonds, local-basis-optimization
  projectors, an adaptive SVD search and a GPU path.  This is the hot 1D code.
* :class:`~fishbonett.states.tree.TreeTensorNetwork` stores an arbitrary loop-free
  tree in mixed-canonical form for the comb and site-tree models.

Propagation algorithms live in :mod:`fishbonett.evolve`; physical model assembly
lives in :mod:`fishbonett.models`.

.. rubric:: Storage interface

The MPS and tree containers use different leg orders:

===================  ==========================  ==============================
site (chain 2-5-7)   ``SystemBathMPS.B[i]``      ``TreeTensorNetwork.T[i]``
===================  ==========================  ==============================
0 (end)              ``(1, 2, 1)``               ``(1, 2)``
1 (interior)         ``(1, 5, 1)``               ``(1, 1, 5)``
2 (end)              ``(1, 7, 1)``               ``(1, 7)``
===================  ==========================  ==============================

The MPS puts the physical leg in the middle (``vL, p, vR``); the tree puts it last
(``bonds..., p``). :class:`~fishbonett.states.network.TensorNetwork` accesses both
layouts through three storage hooks:

* ``tensor(i)`` -- the site's tensor as ``(bonds..., phys)``, in ``neighbours``
  order, whatever the storage layout is;
* ``set_tensor(i, value)`` -- write it back;
* ``neighbours(i)`` -- the node ids those bond legs lead to.

The MPS hooks permute axes and omit dummy end bonds; the tree hooks use the stored
layout directly.

Two additional hooks define the canonical-form policy:

``_prepare_for(i)``
    choose ``i`` as the orthogonality centre.  The tree walks its centre there by
    QR; the MPS moves no data, because in ``Gamma-Lambda`` form every site is
    already canonical and the choice is only a change of view.
``_gauged_tensor(i)``
    the tensor a *multi-node* contraction needs at ``i``: the centre tensor at the
    centre, and the isometry pointing at it everywhere else.  For the tree that is
    the stored tensor, so this defaults to ``tensor``; the MPS must override it,
    since its ``tensor`` carries the bond weights and would count them twice on
    every internal bond.

``set_tensor`` is unavailable for the MPS because changing one tensor independently
would make its bond singular values inconsistent. MPS updates therefore use
``update_bond`` to preserve the gauge.
"""
from fishbonett.states.mps import SystemBathMPS
from fishbonett.states.thermal import GibbsPurification
from fishbonett.states.tree import TreeTensorNetwork

__all__ = ["SystemBathMPS", "TreeTensorNetwork", "GibbsPurification"]
