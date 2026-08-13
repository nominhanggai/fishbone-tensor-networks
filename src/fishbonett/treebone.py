"""General tree tensor-network TEBD over an arbitrary loop-free site graph.

Unlike the 1D comb (:mod:`fishbonett.states.comb`), this engine evolves *any* tree
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

from fishbonett.contract import contract

from fishbonett.common import get_bath_nn_paras
from fishbonett.linalg import cap_rank
from fishbonett.bath.legendre import get_vn_squared
from fishbonett.operators import _c, sigma_x, sigma_z
from fishbonett.simulate import Result

__all__ = ["TreeTEBD", "TreeFishbone"]


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


def _bath_ops(d):
    a = _c(d)                       # annihilation
    ad = a.T                        # creation
    return a, ad, a + ad, ad @ a    # a, a^dag, x = a+a^dag, number


def _parse_observable(spec):
    """Normalise an observable spec to ``(kind, operator, sites)``.

    A bare ``(d, d)`` operator is measured on every matching site (``kind
    "persite"``); ``(operator, i)`` or ``(operator, (i, j, ...))`` targets a
    specific site or a composite of sites (``kind "sites"``)."""
    if isinstance(spec, tuple):
        op, where = spec
        if np.isscalar(where) or isinstance(where, (int, np.integer)):
            sites = [int(where)]
        else:
            sites = [int(s) for s in where]
        return "sites", np.asarray(op), sites
    return "persite", np.asarray(spec), None


class TreeFishbone:
    """Electronic sites wired into an *arbitrary tree*, each with one or more baths.

    Generalises :class:`fishbonett.simulate.Fishbone` (a 1D chain) to any
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

    def hamiltonians(self, t_max=None):
        """The chain-mapped physical tree: ``(dims, edges, site_H, edge_H)`` where
        ``site_H[node]`` is the on-site Hamiltonian and ``edge_H[(a, b)]`` the
        two-site coupling.  Electronic sites are nodes ``0..n_sites-1``; each bath
        is a chain of nodes hanging off its site.  ``t_max`` sizes any bath whose
        ``n_modes`` is automatic (see :meth:`fishbonett.simulate.Bath.resolved`)."""
        dims = list(self.de)
        edges = [(i, j) for (i, j, _) in self.edges]
        site_H = [self.sites[i].copy() for i in range(self.ns)]
        edge_H = {(i, j): C for (i, j, C) in self.edges}
        node = self.ns
        for i in range(self.ns):
            for bath in self.baths[i]:
                bath = bath.resolved(t_max)          # fill automatic domain/n_modes
                d = bath.phys_dim
                a, ad, x, numb = _bath_ops(d)
                if getattr(bath, "is_multichannel", False):
                    node = self._add_multichannel_star(bath, i, node, dims, edges,
                                                       site_H, edge_H, a, ad, x, numb)
                    continue
                w, k = get_bath_nn_paras(bath.spectral_density(), bath.n_modes,
                                         list(bath.domain), discretizer=bath.discretizer())
                w = np.asarray(w, float); k = np.asarray(k, float)
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

    def _add_multichannel_star(self, bath, site, node, dims, edges, site_H, edge_H,
                               a, ad, x, numb):
        """One bath coupled to ``site`` through several operators, as a *shared-mode
        star*: every channel uses the same Gauss-Legendre nodes ``omega_k`` (so the
        channels cross-correlate), and mode ``k`` couples via the combined operator
        ``M_k = sum_c g_{c,k} O_c``, ``g_{c,k} = sqrt(J_c(omega_k) w_k / pi)``."""
        if bath.discretization != "legendre":
            raise ValueError("a multichannel bath must use the 'legendre' "
                             "discretization: its Gauss nodes are shared across "
                             "channels, whereas measure-adapted orthpol nodes are not")
        channels = bath.channels()
        freq = None
        g = []
        for Jc, _op in channels:
            f, v_sq = get_vn_squared(Jc, bath.n_modes, list(bath.domain))
            f = np.asarray(f, float)
            g.append(np.sqrt(np.asarray(v_sq, float) / np.pi))
            if freq is None:
                freq = f
            elif not np.allclose(freq, f):        # nodes are shared, so unreachable
                raise ValueError("multichannel channels do not share the mode grid")
        for k in range(bath.n_modes):
            dims.append(bath.phys_dim)
            site_H.append(freq[k] * numb)
            M = sum(g[c][k] * channels[c][1] for c in range(len(channels)))
            edges.append((site, node))
            edge_H[(site, node)] = np.kron(M, x)    # (site op M) (x) (a + a^dag)
            node += 1
        return node

    def _build(self, dt, t_max=None):
        """Physical tree plus the single-site and two-site Trotter gates."""
        dims, edges, site_H, edge_H = self.hamiltonians(t_max)
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

    def run(self, *, dt, t_max=None, n_steps=None, bond_dim=None, trunc_eps=1e-4,
            observables=None, initial="up"):
        """Propagate and return a :class:`~fishbonett.simulate.Result`.

        ``trunc_eps`` is the accuracy knob (singular values below it are dropped);
        ``bond_dim`` is an optional hard cap, ``None`` meaning **unlimited** -- the
        bond then grows to whatever ``trunc_eps`` requires.

        Each entry of ``observables`` is one of:

        * a bare ``(d, d)`` operator -- measured on **every** matching site;
          ``expect[name]`` is then ``(n_steps, n_sites)`` (NaN where the operator
          dimension does not match a site);
        * ``(operator, i)`` -- the operator on the single site ``i``;
        * ``(operator, (i, j, ...))`` -- a composite operator on those sites
          (``operator`` is ``(D, D)`` with ``D`` = product of the site dimensions
          in that order, e.g. a two-site correlation ``sigma_z (x) sigma_z``).
          For the last two forms ``expect[name]`` is ``(n_steps,)``.

        ``rdm`` holds the single-site reduced density matrices per step."""
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        if observables is None:
            observables = {"sz": sigma_z, "sx": sigma_x} if all(
                d == 2 for d in self.de) else {}
        parsed = [(name, _parse_observable(spec))
                  for name, spec in observables.items()]
        dims, edges, site_gates, edge_gates = self._build(dt, n_steps * dt)
        st = TreeTEBD(dims, edges, root=0)
        for i in range(self.ns):
            st.set_physical(i, self._initial_vec(initial, i))

        expect = {name: (np.full((n_steps, self.ns), np.nan) if kind == "persite"
                         else np.full(n_steps, np.nan))
                  for name, (kind, _O, _s) in parsed}
        rdms = np.empty((n_steps, self.ns), dtype=object)
        for tn in range(n_steps):
            st.step(site_gates, edge_gates, bond_dim, trunc_eps)
            for i in range(self.ns):
                rdms[tn, i] = st.rdm(i)
            for name, (kind, O, sites) in parsed:
                if kind == "persite":
                    for i in range(self.ns):
                        if O.shape == (self.de[i], self.de[i]):
                            expect[name][tn, i] = np.trace(rdms[tn, i] @ O).real
                else:
                    expect[name][tn] = st.expectation(O, sites)
            st.move_oc_to(0)
        if len(set(self.de)) == 1:                        # uniform sites -> dense
            rdm = np.array([[rdms[tn, i] for i in range(self.ns)]
                            for tn in range(n_steps)])
        else:                                             # mixed dims -> object array
            rdm = rdms
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=expect, rdm=rdm, method="treebone",
                      meta={"n_sites": self.ns})
