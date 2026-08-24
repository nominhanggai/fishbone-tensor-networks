"""Common operations for loop-free tensor-network states.

A matrix-product state and a tree tensor network both place one tensor and one
physical leg on each node of a loop-free graph, with one bond per edge. This module
provides topology, mixed-canonical-form operations, orthogonality-centre movement,
reduced density matrices and expectation values for both containers.

.. rubric:: Storage interface

The containers store their legs differently: the MPS uses
``(vL, p, vR)`` with the physical leg in the middle, the tree uses
``(bonds..., p)`` with it last. Subclasses expose either layout through three hooks:

=====================  ======================================================
:meth:`tensor`         the node's tensor, in ``(bonds..., phys)`` order
:meth:`set_tensor`     write it back, undoing whatever permutation was needed
:meth:`neighbours`     the node ids its bond legs correspond to, in leg order
=====================  ======================================================

The common algorithms operate on the normalized ``(bonds..., phys)`` view and do
not depend on the stored axis order.
"""
import numpy as np

from fishbonett.contract import contract

__all__ = ["TensorNetwork"]


class TensorNetwork:
    """Tensors on a loop-free graph, with a single orthogonality centre.

    Subclasses own the storage and supply :meth:`tensor`, :meth:`set_tensor` and
    :meth:`neighbours`; this class supplies the topology, the canonical form and the
    observables.  ``self.n``, ``self.dims``, ``self.root`` and ``self.oc`` are
    expected to exist.
    """

    # -- storage hooks (subclasses override) ----------------------------------
    def tensor(self, i):
        """Node ``i``'s tensor with legs ``(bond per neighbour..., physical)``."""
        raise NotImplementedError

    def set_tensor(self, i, value):
        """Write back a tensor given in ``(bonds..., physical)`` leg order."""
        raise NotImplementedError

    def neighbours(self, i):
        """The node ids of ``i``'s bond legs, in the order the legs appear."""
        raise NotImplementedError

    # -- topology --------------------------------------------------------------
    def _build_topology(self, edges):
        """Adjacency, rooting and the parent/child relation.  ``edges`` must form a
        tree: ``n - 1`` of them, connecting every node."""
        edges = list(edges)
        normalized = set()
        for edge in edges:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError("each edge must contain two node indices")
            a, b = edge
            if (not isinstance(a, (int, np.integer))
                    or isinstance(a, (bool, np.bool_))
                    or not isinstance(b, (int, np.integer))
                    or isinstance(b, (bool, np.bool_))):
                raise TypeError("edge endpoints must be integer node indices")
            a, b = int(a), int(b)
            if a < 0 or b < 0 or a >= self.n or b >= self.n:
                raise ValueError(f"edge {(a, b)} is outside the tensor network")
            if a == b:
                raise ValueError(f"edge {(a, b)} is a self-edge")
            key = tuple(sorted((a, b)))
            if key in normalized:
                raise ValueError(f"duplicate edge {key}")
            normalized.add(key)
        self.adj = [[] for _ in range(self.n)]
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        if len(edges) != self.n - 1:
            raise ValueError("edges must form a tree (n-1 edges, no loops)")
        self._root_tree()

    def _root_tree(self):
        self.parent = [None] * self.n
        self.children = [[] for _ in range(self.n)]
        seen = [False] * self.n
        stack = [self.root]
        seen[self.root] = True
        order_visit = []
        while stack:
            u = stack.pop()
            order_visit.append(u)
            for v in self.adj[u]:
                if not seen[v]:
                    seen[v] = True
                    self.parent[v] = u
                    self.children[u].append(v)
                    stack.append(v)
        if not all(seen):
            raise ValueError("the site graph is not connected")
        self._visit = order_visit

    def _leg(self, i, j):
        """Which of ``i``'s bond legs points at neighbour ``j``."""
        return list(self.neighbours(i)).index(j)

    def path(self, a, b):
        """Node path ``a -> b``.  A tree has exactly one, which is what makes the
        canonical form exact: walk up to the lowest common ancestor and back down."""
        up_a, x = [a], a
        while self.parent[x] is not None:
            x = self.parent[x]; up_a.append(x)
        up_b, y = [b], b
        while self.parent[y] is not None:
            y = self.parent[y]; up_b.append(y)
        set_b = {node: k for k, node in enumerate(up_b)}
        for ka, node in enumerate(up_a):
            if node in set_b:
                kb = set_b[node]; break
        return up_a[:ka] + up_b[:kb + 1][::-1]

    # -- gauge / canonicalisation ----------------------------------------------
    def _qr_toward(self, i, leg):
        """Isometrise ``tensor(i)`` on all legs except ``leg``; return ``R``."""
        A = self.tensor(i)
        nd = A.ndim
        perm = [ax for ax in range(nd) if ax != leg] + [leg]
        Ap = np.transpose(A, perm).reshape(-1, A.shape[leg])
        Q, R = np.linalg.qr(Ap)
        r = R.shape[0]
        Qshape = [A.shape[ax] for ax in range(nd) if ax != leg] + [r]
        Q = Q.reshape(Qshape)
        inv = list(range(nd - 1))
        inv.insert(leg, nd - 1)
        self.set_tensor(i, np.transpose(Q, inv))
        return R

    def _absorb(self, j, i, R):
        """Absorb ``R`` (``r x old``) into ``tensor(j)`` on the leg toward ``i``."""
        lj = self._leg(j, i)
        X = np.tensordot(R, self.tensor(j), axes=([1], [lj]))
        self.set_tensor(j, np.moveaxis(X, 0, lj))

    def move_oc(self, i, j):
        """Move the orthogonality centre from ``i`` to neighbour ``j`` (QR gauge)."""
        R = self._qr_toward(i, self._leg(i, j))
        self._absorb(j, i, R)
        self.oc = j

    def move_oc_to(self, target):
        """Move the orthogonality centre to ``target``, one QR per edge on the way."""
        for nxt in self.path(self.oc, target)[1:]:
            self.move_oc(self.oc, nxt)

    # -- observables ------------------------------------------------------------
    def _prepare_for(self, i):
        """Put the network in a gauge whose orthogonality centre is node ``i``.

        The default is the mixed-canonical one: walk the centre to ``i`` by QR.  A
        subclass in Vidal (``Gamma-Lambda``) form overrides this to move no data,
        because there *every* site is already canonical and picking a centre is
        just a change of view.
        """
        self.move_oc_to(i)

    def _gauged_tensor(self, i):
        """Node ``i``'s tensor in the gauge :meth:`_prepare_for` just established.

        This is what a multi-node contraction must use, and it is **not** always
        :meth:`tensor`.  What the contraction needs at ``i`` is the centre tensor if
        ``i`` is the centre and the *isometry pointing at the centre* otherwise; only
        then do the bonds leaving the subtree close into identities.  In
        mixed-canonical form the stored tensor is already that, so the default just
        returns :meth:`tensor`.  A Vidal-form subclass must override it, since there
        :meth:`tensor` carries the bond weights and would count them twice on every
        internal bond.
        """
        return self.tensor(i)

    def rdm(self, i):
        """Reduced density matrix on node ``i``.

        Once :meth:`_prepare_for` has run, every other tensor is isometric, so
        tracing them out is the identity and the RDM is read off this one tensor.
        """
        self._prepare_for(i)
        A = self.tensor(i)
        phys = A.ndim - 1
        bonds = list(range(phys))
        rho = np.tensordot(A, A.conj(), axes=(bonds, bonds))
        return rho / np.trace(rho)

    def _spanning_subtree(self, sites):
        """Nodes on the minimal subtree connecting ``sites``."""
        sub = set(sites)
        base = sites[0]
        for s in sites[1:]:
            sub.update(self.path(base, s))
        return sub

    def joint_rdm(self, sites):
        """Joint reduced density matrix of ``sites`` (ordered).

        Contracts the double layer over the spanning subtree only.  With the
        orthogonality centre inside it, the tensors outside are isometric, so each
        leaving bond contracts to an identity and the cost stays local rather than
        exponential in the whole network.
        """
        sites = [int(s) for s in sites]
        if len(sites) == 1:
            return self.rdm(sites[0])
        sub = self._spanning_subtree(sites)
        # any node of the subtree is a valid centre; the lowest id is one, and on a
        # path it is the leftmost, which lets an MPS use its stored right-isometries
        # for the whole rest of the subtree.
        self._prepare_for(min(sub))
        counter = [0]

        def new():
            counter[0] += 1
            return counter[0]

        bond_ket, bond_bra, leave = {}, {}, {}
        phys_ket, phys_bra = {}, {}
        operands = []
        for n in sub:
            legs_k, legs_b = [], []
            for m in self.neighbours(n):
                if m in sub:                        # internal bond: ket & bra kept
                    key = frozenset((n, m))
                    if key not in bond_ket:
                        bond_ket[key], bond_bra[key] = new(), new()
                    legs_k.append(bond_ket[key]); legs_b.append(bond_bra[key])
                else:                               # leaving bond: capped (identity)
                    if (n, m) not in leave:
                        leave[(n, m)] = new()
                    legs_k.append(leave[(n, m)]); legs_b.append(leave[(n, m)])
            if n in sites:                          # keep this physical leg open
                phys_ket[n], phys_bra[n] = new(), new()
                legs_k.append(phys_ket[n]); legs_b.append(phys_bra[n])
            else:                                   # trace this physical leg
                pt = new()
                legs_k.append(pt); legs_b.append(pt)
            t = self._gauged_tensor(n)
            operands += [t, legs_k, t.conj(), legs_b]
        out = [phys_ket[s] for s in sites] + [phys_bra[s] for s in sites]
        operands.append(out)
        rho = contract(*operands)
        d = int(np.prod([self.dims[s] for s in sites]))
        rho = rho.reshape(d, d)
        return rho / np.trace(rho)

    def expectation(self, operator, sites):
        """Expectation of ``operator`` on ``sites`` (an int or a list of node ids).

        ``operator`` is ``(D, D)`` with ``D`` the product of those nodes' dimensions
        in the order given.
        """
        sites = [sites] if np.isscalar(sites) or isinstance(sites, (int, np.integer)) \
            else list(sites)
        rho = self.joint_rdm(sites)
        return np.trace(rho @ np.asarray(operator))
