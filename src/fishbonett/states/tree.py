"""Mixed-canonical tree tensor-network state.

The *state* only: node tensors, their canonical form, and reduced density
matrices.  It knows nothing about baths or representations -- the models that drive it live
in :mod:`fishbonett.models.fishbone`, and the gate application and Trotter step in
:mod:`fishbonett.evolve.sitetree` (the methods here are thin wrappers around it,
as in :mod:`fishbonett.states.mps`).

Node ``i``'s tensor carries legs ``[bond to each neighbour (in ``order[i]``)
..., physical]``, physical last.  A tree has no loops, so there is an exact
mixed-canonical form: a single orthogonality centre moves by QR along the unique
path between any two nodes, and a site's RDM is read straight off the centre.

That leg order is exactly the one
:class:`~fishbonett.states.network.TensorNetwork` asks for, so the three storage
hooks below are the identity and everything structural -- topology, the
orthogonality-centre walk, the RDMs -- is inherited.  What is left here is the
storage itself, the product-state constructor, and the wrappers around
:mod:`fishbonett.evolve.sitetree`.  :class:`fishbonett.states.mps.SystemBathMPS` is
the same network over a path, differing only in that it stores the physical leg in
the middle of a fixed ``(vL, p, vR)`` and keeps its gauge in Vidal form.
"""
import numpy as np

from fishbonett.states.network import TensorNetwork

__all__ = ["TreeTensorNetwork"]


class TreeTensorNetwork(TensorNetwork):
    """Mixed-canonical tree tensor-network state (TTN).

    Named for the ansatz, as :class:`fishbonett.states.mps.SystemBathMPS` is.  It was
    called ``TreeTEBD``, which named an *algorithm* -- and one that now lives in
    :mod:`fishbonett.evolve.sitetree` rather than here.

    The topology, the orthogonality-centre machinery and the observables all live in
    :class:`~fishbonett.states.network.TensorNetwork`; what is left here is the
    storage (``T`` indexed by node, legs already in the base's
    ``(bonds..., phys)`` order) and the product-state constructor.

    Parameters
    ----------
    dims : list[int]
        Physical dimension of each node.
    edges : list[(int, int)]
        Undirected tree edges; must connect all ``len(dims)`` nodes without loops.
    root : int
        Node to root the canonical form at (the orthogonality centre starts here).
    """

    def __init__(self, dims, edges, root=0):
        self.n = len(dims)
        self.dims = list(dims)
        self.root = root
        self.oc = root
        self._build_topology(edges)
        self.order = [list(nb) for nb in self.adj]   # neighbour order == leg order
        # product state: every bond dimension 1, physical leg |0>
        self.T = []
        for i in range(self.n):
            shape = [1] * len(self.order[i]) + [self.dims[i]]
            t = np.zeros(shape, complex)
            t[tuple([0] * len(self.order[i]) + [0])] = 1.0
            self.T.append(t)

    # -- storage: legs are already (bonds..., phys), so these are the identity --
    def tensor(self, i):
        return self.T[i]

    def set_tensor(self, i, value):
        self.T[i] = value

    def neighbours(self, i):
        return self.order[i]

    # -- initial state -------------------------------------------------------
    def set_physical(self, i, vec):
        """Set node ``i``'s physical leg to ``vec``, leaving every bond at
        dimension 1 (a product state)."""
        vec = np.asarray(vec, complex)
        t = np.zeros_like(self.T[i])
        idx = [0] * len(self.order[i])
        for a in range(len(vec)):
            t[tuple(idx + [a])] = vec[a]
        self.T[i] = t

    # -- gate application (algorithm lives in fishbonett.evolve.sitetree) ----
    def apply_edge(self, i, j, U, chi, eps):
        """Apply gate ``U`` (di_out, dj_out, di_in, dj_in) on edge (i, j); the OC
        moves from ``i`` to ``j``.  Grows the shared bond by SVD truncation.

        Thin convenience wrapper: the algorithm lives in
        :func:`fishbonett.evolve.sitetree.apply_edge` (this state object only holds
        the tensors and their canonical form).
        """
        from fishbonett.evolve.sitetree import apply_edge as _apply_edge
        _apply_edge(self, i, j, U, chi, eps)

    def apply_site(self, i, U):
        """Apply a single-site gate ``U`` (``(d, d)``) to node ``i``.

        Thin wrapper around :func:`fishbonett.evolve.sitetree.apply_site`.  
        Single-site gates touch no bond, so they neither grow the state nor disturb
        the canonical form -- no truncation is needed.
        """
        from fishbonett.evolve.sitetree import apply_site as _apply_site
        _apply_site(self, i, U)

    def step(self, site_gates, edge_gates, chi, eps):
        """One 2nd-order (Strang) Trotter step; ``site_gates``/``edge_gates`` are
        the **half-step** gates.

        Thin wrapper around
        :func:`fishbonett.evolve.sitetree.symmetric_tree_step`, which explains why
        a palindromic sweep is needed rather than a plain Euler tour.
        """
        from fishbonett.evolve.sitetree import symmetric_tree_step
        symmetric_tree_step(self, site_gates, edge_gates, chi, eps)

