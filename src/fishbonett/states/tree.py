"""Mixed-canonical tree tensor-network state with a TEBD step.

The *state* only: node tensors, their canonical form, gate application and
reduced density matrices.  It knows nothing about baths or frames -- the models
that drive it live in :mod:`fishbonett.models.fishbone`.

Node ``i``'s tensor carries legs ``[bond to each neighbour (in ``order[i]``)
..., physical]``, physical last.  A tree has no loops, so there is an exact
mixed-canonical form: a single orthogonality centre moves by QR along the unique
path between any two nodes, and a site's RDM is read straight off the centre.
"""
import numpy as np

from fishbonett.contract import contract
from fishbonett.linalg import cap_rank

__all__ = ["TreeTEBD"]


def _svd_trunc(mat, chi, eps):
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    k = cap_rank(np.sum(S > eps * (S[0] if S.size else 1.0)), chi)
    return U[:, :k], S[:k], Vh[:k, :]


class TreeTEBD:
    """Mixed-canonical tree TEBD state.

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
        self.adj = [[] for _ in range(self.n)]
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        if len(edges) != self.n - 1:
            raise ValueError("edges must form a tree (n-1 edges, no loops)")
        self.order = [list(nb) for nb in self.adj]     # neighbour order == bond-leg order
        self.root = root
        self._root_tree()
        self.oc = root
        # product state: every bond dim 1, physical = |0>
        self.T = []
        for i in range(self.n):
            shape = [1] * len(self.order[i]) + [self.dims[i]]
            t = np.zeros(shape, complex)
            t[tuple([0] * len(self.order[i]) + [0])] = 1.0
            self.T.append(t)

    # -- topology ------------------------------------------------------------
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
        return self.order[i].index(j)

    def path(self, a, b):
        """Node path a -> b along the tree."""
        # walk parents to root from both, find LCA
        up_a, x = [a], a
        while self.parent[x] is not None:
            x = self.parent[x]; up_a.append(x)
        up_b, y = [b], b
        while self.parent[y] is not None:
            y = self.parent[y]; up_b.append(y)
        set_b = {node: k for k, node in enumerate(up_b)}
        for ka, node in enumerate(up_a):
            if node in set_b:
                lca = node; kb = set_b[node]; break
        return up_a[:ka] + up_b[:kb + 1][::-1]

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

    # -- gauge / canonicalisation --------------------------------------------
    def _qr_toward(self, i, leg):
        """Isometrise ``T[i]`` on all legs except ``leg``; return R (r x dim_leg)."""
        A = self.T[i]
        nd = A.ndim
        perm = [ax for ax in range(nd) if ax != leg] + [leg]
        Ap = np.transpose(A, perm).reshape(-1, A.shape[leg])
        Q, R = np.linalg.qr(Ap)
        r = R.shape[0]
        Qshape = [A.shape[ax] for ax in range(nd) if ax != leg] + [r]
        Q = Q.reshape(Qshape)
        inv = list(range(nd - 1))
        inv.insert(leg, nd - 1)
        self.T[i] = np.transpose(Q, inv)
        return R

    def _absorb(self, j, i, R):
        """Absorb ``R`` (r x old) into ``T[j]`` on the leg toward ``i``."""
        lj = self._leg(j, i)
        X = np.tensordot(R, self.T[j], axes=([1], [lj]))   # [r, (Tj legs != lj)]
        self.T[j] = np.moveaxis(X, 0, lj)

    def move_oc(self, i, j):
        """Move the orthogonality centre from ``i`` to neighbour ``j`` (QR gauge)."""
        R = self._qr_toward(i, self._leg(i, j))
        self._absorb(j, i, R)
        self.oc = j

    def move_oc_to(self, target):
        """Move the orthogonality centre to node ``target``, one QR gauge
        transformation per edge along the tree path."""
        for nxt in self.path(self.oc, target)[1:]:
            self.move_oc(self.oc, nxt)

    # -- two-site gate on an edge (OC at i) ----------------------------------
    def apply_edge(self, i, j, U, chi, eps):
        """Apply gate ``U`` (di_out, dj_out, di_in, dj_in) on edge (i, j); the OC
        moves from ``i`` to ``j``.  Grows the shared bond by SVD truncation."""
        li, lj = self._leg(i, j), self._leg(j, i)
        Ti, Tj = self.T[i], self.T[j]
        ki, kj = len(self.order[i]), len(self.order[j])
        pool = iter("abcdefghijklmnopqrstuvwABCDEFGHIJKLMNOPQRSTUVW")
        shared = next(pool)
        ib = [next(pool) for _ in range(ki)]; ib[li] = shared
        jb = [next(pool) for _ in range(kj)]; jb[lj] = shared
        si, sj = next(pool), next(pool)           # physical in
        so_i, so_j = next(pool), next(pool)       # physical out
        theta_out = ([b for k, b in enumerate(ib) if k != li] + [so_i]
                     + [b for k, b in enumerate(jb) if k != lj] + [so_j])
        sub = (f"{''.join(ib)}{si},{''.join(jb)}{sj},{so_i}{so_j}{si}{sj}"
               f"->{''.join(theta_out)}")
        theta = np.einsum(sub, Ti, Tj, U, optimize=True)
        nleft = (ki - 1) + 1                       # i's other bonds + phys_out
        lsh, rsh = theta.shape[:nleft], theta.shape[nleft:]
        Um, S, Vh = _svd_trunc(theta.reshape(int(np.prod(lsh)), int(np.prod(rsh))),
                               chi, eps)
        k = S.shape[0]
        left = Um.reshape(list(lsh) + [k])         # [i-other-bonds..., phys_i, new]
        right = (S[:, None] * Vh).reshape([k] + list(rsh))  # [new, j-other-bonds..., phys_j]
        self.T[i] = self._reassemble_left(left, i, li)
        self.T[j] = self._reassemble_right(right, j, lj)
        self.oc = j

    def _reassemble_left(self, left, i, li):
        # left legs: [i bonds except li (orig order), phys, new] -> T[i] order
        ki = len(self.order[i])
        newax = left.ndim - 1
        physax = left.ndim - 2
        src = []
        other = [ax for ax in range(ki - 1)]       # 0..ki-2 are the non-li bonds
        oi = iter(other)
        for t in range(ki):
            src.append(newax if t == li else next(oi))
        src.append(physax)
        return np.transpose(left, src)

    def _reassemble_right(self, right, j, lj):
        # right legs: [new, j bonds except lj (orig order), phys] -> T[j] order
        kj = len(self.order[j])
        newax = 0
        physax = right.ndim - 1
        other = list(range(1, kj))                 # 1..kj-1 are the non-lj bonds
        oi = iter(other)
        src = []
        for t in range(kj):
            src.append(newax if t == lj else next(oi))
        src.append(physax)
        return np.transpose(right, src)

    # -- single-site gate (bond-preserving, canonical-form preserving) -------
    def apply_site(self, i, U):
        """Apply a single-site gate ``U`` (``(d, d)``) to node ``i``.

        Single-site gates touch no bond, so this neither grows the state nor
        disturbs the canonical form -- no truncation is needed.
        """
        phys = self.T[i].ndim - 1
        X = np.tensordot(U, self.T[i], axes=([1], [phys]))   # [out, other legs...]
        self.T[i] = np.moveaxis(X, 0, phys)

    # -- one Trotter step ----------------------------------------------------
    def step(self, site_gates, edge_gates, chi, eps):
        """First-order Trotter: all single-site gates, then an Euler tour of the
        two-site edge gates.  ``site_gates[i]`` is a (d,d) unitary or None;
        ``edge_gates[(i,j)]`` a (di,dj,di,dj) unitary (either edge orientation)."""
        for i in range(self.n):
            g = site_gates[i]
            if g is not None:
                self.apply_site(i, g)

        def dfs(node):
            for c in self.children[node]:
                U = edge_gates.get((node, c))
                if U is None:
                    Ur = edge_gates[(c, node)]
                    U = np.transpose(Ur, (1, 0, 3, 2))
                self.apply_edge(node, c, U, chi, eps)   # OC node -> c
                dfs(c)
                self.move_oc(c, node)                    # OC c -> node
        dfs(self.root)

    # -- observables ---------------------------------------------------------
    def rdm(self, i):
        """Reduced density matrix on site ``i`` (moves the OC there)."""
        self.move_oc_to(i)
        A = self.T[i]
        phys = A.ndim - 1
        bonds = list(range(phys))
        rho = np.tensordot(A, A.conj(), axes=(bonds, bonds))  # [phys, phys*]
        return rho / np.trace(rho).real

    def _spanning_subtree(self, sites):
        """Set of nodes on the minimal subtree connecting ``sites``."""
        sub = set(sites)
        base = sites[0]
        for s in sites[1:]:
            sub.update(self.path(base, s))
        return sub

    def joint_rdm(self, sites):
        """Joint reduced density matrix of ``sites`` (ordered), shape
        ``(prod d_s, prod d_s)``.  Contracts the double layer over the spanning
        subtree; with the orthogonality centre inside it, tensors outside are
        isometric so the leaving bonds contract to identity and only the spanning
        subtree is touched (no exponential full contraction)."""
        sites = [int(s) for s in sites]
        if len(sites) == 1:
            return self.rdm(sites[0])
        sub = self._spanning_subtree(sites)
        self.move_oc_to(sites[0])
        counter = [0]

        def new():
            counter[0] += 1
            return counter[0]

        bond_ket, bond_bra, leave = {}, {}, {}
        phys_ket, phys_bra = {}, {}
        operands = []
        for n in sub:
            legs_k, legs_b = [], []
            for m in self.order[n]:
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
            operands += [self.T[n], legs_k, self.T[n].conj(), legs_b]
        out = [phys_ket[s] for s in sites] + [phys_bra[s] for s in sites]
        operands.append(out)
        rho = contract(*operands)
        d = int(np.prod([self.dims[s] for s in sites]))
        rho = rho.reshape(d, d)
        return rho / np.trace(rho).real

    def expectation(self, operator, sites):
        """Expectation of ``operator`` acting on ``sites`` (an int or a list of
        site indices).  ``operator`` is ``(D, D)`` with ``D = prod`` of the site
        dimensions in the given order."""
        sites = [sites] if np.isscalar(sites) or isinstance(sites, (int, np.integer)) \
            else list(sites)
        rho = self.joint_rdm(sites)
        return np.trace(rho @ np.asarray(operator)).real

