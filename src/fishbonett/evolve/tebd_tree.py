"""TEBD on a tree tensor network: gate application and whole symmetric steps.

The tree counterpart of :mod:`fishbonett.evolve.tebd`.  It drives a
:class:`~fishbonett.states.tree.TreeTEBD` state -- an arbitrary loop-free tree of
*sites*, each of which may carry its own bath chain -- which is the geometry the
``comb``, ``site-tree`` and ``multichannel`` models use (method
``tree-tebd-static``).

Do not confuse it with :func:`fishbonett.evolve.treetdvp.run_tree_tebd`, which is
also tree TEBD but on a different geometry and in a different frame: there the tree
is a balanced binary tree of *bath modes* hanging off one system site, in the
interaction picture (method ``tree-tebd``).  Here the frame is Schroedinger, so the
gates are time-independent and built once.

.. rubric:: What's here

===========================  ==================================================
:func:`apply_site`           a one-site gate (no bond, so no truncation)
:func:`apply_edge`           a two-site gate on an edge, moving the OC across it
:func:`symmetric_tree_step`  one 2nd-order (Strang) step over the whole tree
===========================  ==================================================

A tree has no loops, so the orthogonality centre moves by QR along the unique path
between two nodes; :mod:`fishbonett.states.tree` owns that machinery and this
module only applies gates through it.
"""
import numpy as np

from fishbonett.linalg import cap_rank

__all__ = ["apply_site", "apply_edge", "symmetric_tree_step", "edge_gate"]


def _svd_trunc(mat, chi, eps):
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    k = cap_rank(np.sum(S > eps * (S[0] if S.size else 1.0)), chi)
    return U[:, :k], S[:k], Vh[:k, :]


def edge_gate(edge_gates, i, j):
    """The gate for edge ``(i, j)``, whichever orientation it is stored under.

    ``edge_gates`` is keyed by one orientation only; the other is the leg-swapped
    transpose, since a gate with legs ``(di, dj, di*, dj*)`` read from ``j``'s side
    is ``(dj, di, dj*, di*)``.
    """
    U = edge_gates.get((i, j))
    if U is None:
        U = np.transpose(edge_gates[(j, i)], (1, 0, 3, 2))
    return U


# -- single-site gate (bond-preserving, canonical-form preserving) -------------
def apply_site(state, i, U):
    """Apply a single-site gate ``U`` (``(d, d)``) to node ``i`` of ``state``.

    Single-site gates touch no bond, so this neither grows the state nor disturbs
    the canonical form -- no truncation is needed.
    """
    phys = state.T[i].ndim - 1
    X = np.tensordot(U, state.T[i], axes=([1], [phys]))   # [out, other legs...]
    state.T[i] = np.moveaxis(X, 0, phys)


# -- two-site gate on an edge (OC at i) ---------------------------------------
def apply_edge(state, i, j, U, chi, eps):
    """Apply gate ``U`` (``di_out, dj_out, di_in, dj_in``) on edge ``(i, j)``.

    The orthogonality centre must be at ``i`` and ends at ``j``.  The shared bond
    grows and is then truncated by SVD, keeping singular values above ``eps``
    (relative to the largest) and at most ``chi`` of them.
    """
    li, lj = state._leg(i, j), state._leg(j, i)
    Ti, Tj = state.T[i], state.T[j]
    ki, kj = len(state.order[i]), len(state.order[j])
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
    state.T[i] = _reassemble_left(state, left, i, li)
    state.T[j] = _reassemble_right(state, right, j, lj)
    state.oc = j


def _reassemble_left(state, left, i, li):
    # left legs: [i bonds except li (orig order), phys, new] -> T[i] order
    ki = len(state.order[i])
    newax = left.ndim - 1
    physax = left.ndim - 2
    src = []
    other = [ax for ax in range(ki - 1)]       # 0..ki-2 are the non-li bonds
    oi = iter(other)
    for t in range(ki):
        src.append(newax if t == li else next(oi))
    src.append(physax)
    return np.transpose(left, src)


def _reassemble_right(state, right, j, lj):
    # right legs: [new, j bonds except lj (orig order), phys] -> T[j] order
    kj = len(state.order[j])
    newax = 0
    physax = right.ndim - 1
    other = list(range(1, kj))                 # 1..kj-1 are the non-lj bonds
    oi = iter(other)
    src = []
    for t in range(kj):
        src.append(newax if t == lj else next(oi))
    src.append(physax)
    return np.transpose(right, src)


# -- whole symmetric step ------------------------------------------------------
def symmetric_tree_step(state, site_gates, edge_gates, chi, eps):
    """One 2nd-order (Strang) Trotter step over the whole tree.

    ``site_gates[i]`` is a ``(d, d)`` unitary or ``None``; ``edge_gates[(i, j)]`` a
    ``(di, dj, di*, dj*)`` unitary under either edge orientation.  **Both must be
    half-step (``dt/2``) gates** -- this applies each of them twice, as
    :func:`fishbonett.evolve.tebd.symmetric_static_step` does on a chain.

    Second order requires the *whole* edge sequence to be palindromic, which is
    subtler on a tree than on a chain.  Two schemes that look right are not:

    * one full gate per edge in a single Euler tour -- what this engine used to do.
      A plain Lie-Trotter product; measured order **1.07**.
    * half a gate on the way *down* the tour and half on the way back *up*.  At a
      branching node the two halves of one edge end up adjacent and merge back into
      a full gate: for children ``A, B, C`` the tour emits
      ``A/2 A/2 B/2 B/2 C/2 C/2 = A B C``, so the palindrome is lost.  It is better
      than the first scheme but still not second order; measured **1.79**.

    So the two passes below are a forward sweep in **pre-order** and a backward
    sweep in **reverse post-order**, which emit ``e_1 ... e_m`` and then
    ``e_m ... e_1``: palindromic for any tree shape, measured order **2.00**.  On a
    path this reduces exactly to the chain's up-then-down sweep; on a star it gives
    ``A/2 B/2 C/2 C/2 B/2 A/2``, where only the middle pair merges.  Both passes
    move the orthogonality centre only between a parent and a child, so no
    long-range :meth:`~fishbonett.states.tree.TreeTEBD.move_oc_to` walks are needed.

    ``tests/unit/test_fishbone_sim.py::test_tree_step_is_second_order_in_dt`` pins
    this by halving ``dt`` on a branching tree -- the shape that tells the three
    schemes apart.
    """
    def all_site_gates():
        for i in range(state.n):
            g = site_gates[i]
            if g is not None:
                apply_site(state, i, g)

    all_site_gates()                                    # first half

    def forward(node):                                  # pre-order: gate on descent
        for c in state.children[node]:
            apply_edge(state, node, c, edge_gate(edge_gates, node, c), chi, eps)
            forward(c)
            state.move_oc(c, node)                      # bare gauge move back up
    forward(state.root)

    def backward(node):                     # reverse post-order: gate on ascent
        for c in reversed(state.children[node]):
            state.move_oc(node, c)                      # bare gauge move down
            backward(c)
            apply_edge(state, c, node, edge_gate(edge_gates, c, node), chi, eps)
    backward(state.root)

    all_site_gates()                                    # second half
