"""``LocalTerms`` -- the interface between the physics and the numerics.

Every frame's job is to turn a model plus its discretized baths into **local terms
on a graph**: one operator per node, one per edge.  Everything downstream consumes
only that:

* Trotter gates are the exponential of each term (:meth:`LocalTerms.gates`);
* an MPO is a finite-state machine over the same graph
  (:mod:`fishbonett.frames.mpo`);
* the state is a tensor network over the same graph
  (:mod:`fishbonett.states`).

Keeping this one shape is what lets a propagator be written once and used on a
chain, a comb or an arbitrary tree -- they differ only in the edge list.  The
geometry is *in the data*, not in the code.

.. note::
   Not every frame fits this container, and that is deliberate rather than an
   oversight.  The interaction picture has **no** on-site bath terms at all and a
   *time-dependent* coupling, so its "terms" are a function of ``t`` rebuilt every
   step; the polaron frame folds the coupling into a displacement on one bond.
   ``LocalTerms`` is the shape of a **static** Hamiltonian, which is why the
   Schroedinger picture is what it serves.

   The time-dependent frames emit a plain list of two-site Hamiltonians instead and
   compile it with :func:`fishbonett.frames.gates.swap_gate_pairs` -- the same split
   of "what the terms are" from "how they become gates", one layer down.  See
   :class:`fishbonett.frames.interaction_picture.SystemBathIP` and
   :class:`fishbonett.frames.polaron.SystemBathPolaron`.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

__all__ = ["LocalTerms"]


@dataclass
class LocalTerms:
    """A static Hamiltonian as one operator per node and one per edge.

    Parameters
    ----------
    dims : list of int
        Physical dimension of each node, indexed by node id.
    edges : list of (int, int)
        The graph's edges.  Must be a tree over ``len(dims)`` nodes -- loop-free is
        what makes an exact canonical form (and hence the whole approach) possible.
    site : list of (d, d) array
        On-site operator per node.  An all-zero entry means "nothing here", which
        :meth:`gates` turns into ``None`` rather than an identity gate, so the
        propagators can skip it.
    bond : dict
        ``{(i, j): operator}`` with the operator shaped ``(d_i*d_j, d_i*d_j)``.
        Stored under one orientation; consumers that need the other transpose it.
    """

    dims: List[int]
    edges: List[Tuple[int, int]]
    site: List[np.ndarray]
    bond: Dict[Tuple[int, int], np.ndarray]

    def __post_init__(self):
        n = len(self.dims)
        if len(self.edges) != n - 1:
            raise ValueError(
                f"edges must form a tree over {n} nodes (expected {n - 1} edges, "
                f"got {len(self.edges)})")
        if len(self.site) != n:
            raise ValueError(f"site has {len(self.site)} entries, expected {n}")

    @property
    def n_nodes(self):
        return len(self.dims)

    def gates(self, dt):
        """Trotter gates ``exp(-i dt H_term)`` for every node and edge.

        Returns ``(site_gates, edge_gates)``.  ``site_gates[i]`` is ``None`` where
        the on-site operator is zero; ``edge_gates[(i, j)]`` is reshaped to
        ``(d_i, d_j, d_i*, d_j*)``, the layout
        :func:`fishbonett.evolve.sitetree.apply_edge` consumes.

        ``dt`` is the gate's own time argument, **not** the step: the symmetric
        (Strang) steps apply every gate twice, so they are handed ``dt/2``.
        """
        site_gates = [expm(-1j * H * dt) if np.any(H) else None for H in self.site]
        edge_gates = {}
        for (a, b), H in self.bond.items():
            da, db = self.dims[a], self.dims[b]
            edge_gates[(a, b)] = expm(-1j * H * dt).reshape(da, db, da, db)
        return site_gates, edge_gates

    def as_tuple(self):
        """``(dims, edges, site, bond)`` -- the historical 4-tuple shape."""
        return self.dims, self.edges, self.site, self.bond
