"""General tree tensor-network TEBD over an arbitrary loop-free site graph.

Unlike the 1D comb (:mod:`fishbonett.fishbone`), this engine evolves *any* tree
of physical sites: electronic sites wired into an arbitrary tree, each carrying
one or more TEDOPA bath chains.  Because a tree has no loops it admits an exact
mixed-canonical form, so a site's reduced density matrix is read off from the
orthogonality-centre tensor.

The state is a set of node tensors ``T[i]`` with legs ``[bond to each neighbour
(in ``order[i]``) ..., physical]`` (physical leg last).  A first-order Trotter
step applies every single-site gate (bond-preserving, order-free), then walks an
Euler tour of the edges applying each two-site gate once with the orthogonality
centre on the active edge.
"""
import numpy as np
from scipy.linalg import expm

from fishbonett.common import get_bath_nn_paras
from fishbonett.model import _c
from fishbonett.simulate import Result
from fishbonett.stuff import sigma_x, sigma_z

__all__ = ["TreeTEBD", "TreeFishbone"]


def _svd_trunc(mat, chi, eps):
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    k = min(chi, max(1, int(np.sum(S > eps * (S[0] if S.size else 1.0)))))
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


def _bath_ops(d):
    a = _c(d)                       # annihilation
    ad = a.T                        # creation
    return a, ad, a + ad, ad @ a    # a, a^dag, x = a+a^dag, number


class TreeFishbone:
    """Electronic sites wired into an *arbitrary tree*, each with one or more baths.

    Generalises :class:`fishbonett.fishbone_sim.Fishbone` (a 1D chain) to any
    loop-free electronic topology.

    Parameters
    ----------
    sites : list of (d, d) array
        Electronic site Hamiltonians.
    edges : list of (i, j) or (i, j, coupling)
        Electronic-electronic couplings; the pairs must form a tree over the
        sites.  ``coupling`` is a ``(d_i*d_j, d_i*d_j)`` operator (default: none).
    baths : list
        One entry per site: a single :class:`~fishbonett.simulate.Bath`, a list of
        baths, or ``None``.  Each bath carries its own ``coupling`` operator
        (default ``sigma_z``).  Baths may have different domains/discretizations.
    """

    def __init__(self, sites, edges, baths):
        self.sites = [np.asarray(h, complex) for h in sites]
        self.ns = len(self.sites)
        self.de = [h.shape[0] for h in self.sites]
        self.edges = []
        for e in edges:
            if len(e) == 2:
                i, j = e
                C = np.zeros((self.de[i] * self.de[j],) * 2, complex)
            else:
                i, j, C = e
                C = np.asarray(C, complex)
            self.edges.append((int(i), int(j), C))
        if len(self.edges) != self.ns - 1:
            raise ValueError("edges must form a tree over the sites (n_sites-1 edges)")
        self.baths = []
        for entry in baths:
            if entry is None:
                self.baths.append([])
            elif isinstance(entry, (list, tuple)):
                self.baths.append(list(entry))
            else:
                self.baths.append([entry])
        if len(self.baths) != self.ns:
            raise ValueError("baths must have one entry per site")

    def hamiltonians(self):
        """The chain-mapped physical tree: ``(dims, edges, site_H, edge_H)`` where
        ``site_H[node]`` is the on-site Hamiltonian and ``edge_H[(a, b)]`` the
        two-site coupling.  Electronic sites are nodes ``0..n_sites-1``; each bath
        is a chain of nodes hanging off its site."""
        dims = list(self.de)
        edges = [(i, j) for (i, j, _) in self.edges]
        site_H = [self.sites[i].copy() for i in range(self.ns)]
        edge_H = {(i, j): C for (i, j, C) in self.edges}
        node = self.ns
        for i in range(self.ns):
            for bath in self.baths[i]:
                w, k = get_bath_nn_paras(bath.spectral_density(), bath.n_modes,
                                         list(bath.domain), discretizer=bath.discretizer())
                w = np.asarray(w, float); k = np.asarray(k, float)
                d = bath.phys_dim
                a, ad, x, numb = _bath_ops(d)
                cop = np.asarray(bath.coupling if bath.coupling is not None else sigma_z,
                                 complex)
                prev = i
                for m in range(bath.n_modes):
                    dims.append(d)
                    site_H.append(w[m] * numb)
                    edges.append((prev, node))
                    if m == 0:
                        edge_H[(prev, node)] = k[0] * np.kron(cop, x)
                    else:
                        edge_H[(prev, node)] = k[m] * (np.kron(ad, a) + np.kron(a, ad))
                    prev = node
                    node += 1
        return dims, edges, site_H, edge_H

    def _build(self, dt):
        """Physical tree plus the single-site and two-site Trotter gates."""
        dims, edges, site_H, edge_H = self.hamiltonians()
        site_gates = [expm(-1j * H * dt) if np.any(H) else None for H in site_H]
        edge_gates = {}
        for (a_, b_), H in edge_H.items():
            da, db = dims[a_], dims[b_]
            edge_gates[(a_, b_)] = expm(-1j * H * dt).reshape(da, db, da, db)
        return dims, edges, site_gates, edge_gates

    def _initial_vec(self, initial, i):
        de = self.de[i]
        if initial is None or (isinstance(initial, str) and initial == "up"):
            v = np.zeros(de, complex); v[0] = 1.0; return v
        if isinstance(initial, str) and initial == "down":
            v = np.zeros(de, complex); v[min(1, de - 1)] = 1.0; return v
        if isinstance(initial, str) and initial == "ground":
            w, U = np.linalg.eigh(self.sites[i])
            return U[:, int(np.argmin(w))].astype(complex)
        item = initial[i] if isinstance(initial, (list, tuple)) else initial
        v = np.asarray(item, complex)
        return v / np.linalg.norm(v)

    def run(self, *, dt, t_max=None, n_steps=None, bond_dim=100, trunc_eps=1e-10,
            observables=None, initial="up"):
        """Propagate and return a :class:`~fishbonett.simulate.Result` with per-site
        data (``expect[name]`` shape ``(n_steps, n_sites)``; ``rdm`` shape
        ``(n_steps, n_sites, d, d)``)."""
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        if observables is None:
            observables = {"sz": sigma_z, "sx": sigma_x} if all(
                d == 2 for d in self.de) else {}
        dims, edges, site_gates, edge_gates = self._build(dt)
        st = TreeTEBD(dims, edges, root=0)
        for i in range(self.ns):
            st.set_physical(i, self._initial_vec(initial, i))

        rdms = np.empty((n_steps, self.ns), dtype=object)
        for tn in range(n_steps):
            st.step(site_gates, edge_gates, bond_dim, trunc_eps)
            for i in range(self.ns):
                rdms[tn, i] = st.rdm(i)
            st.move_oc_to(0)
        expect = {name: np.array([[np.trace(rdms[tn, i] @ np.asarray(O)).real
                                   for i in range(self.ns)] for tn in range(n_steps)])
                  for name, O in observables.items()}
        rdm = np.array([[rdms[tn, i] for i in range(self.ns)] for tn in range(n_steps)])
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=expect, rdm=rdm, method="treebone",
                      meta={"n_sites": self.ns})
