"""The **site-tree** engine: system sites on a tree, each with its own bath.

Named for the model it serves (:mod:`fishbonett.models.registry`).  It drives a
:class:`~fishbonett.states.tree.TreeTensorNetwork` state -- an arbitrary loop-free tree of
*sites*, each of which may carry its own bath chain -- which is the geometry the
``comb``, ``site-tree`` and ``multichannel`` models use.  The first two use
``schrodinger-chain-tree-tebd``; the shared-mode multichannel model uses
``schrodinger-star-tree-tebd``.  Both representations are Schroedinger, so the
gates are time-independent and built once.

.. rubric:: Relation to :mod:`fishbonett.evolve.modetree`

The mode-tree module propagates a binary tree of bath modes around one system in
the interaction representation (``interaction-chain-tree-tebd``). Its internal
nodes have no physical leg and its drivers maintain per-node operators and
environments. The site-tree module propagates physical system and bath sites with
one physical leg per node and supports arbitrary node degree. The two structures
therefore use different state containers and canonical gauges.

.. rubric:: API

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

from fishbonett.evolve._tdvp_kernels import init_right_envs
from fishbonett.evolve._tdvp_sweeps import DEFAULT_BOND_EXPAND, tdvp2sweep
from fishbonett.linalg import cap_rank
from fishbonett._svd import robust_svd as _robust_svd

__all__ = [
    "apply_site", "apply_edge", "symmetric_tree_step", "edge_gate",
    "symmetric_graph_step", "symmetric_branch_swap_step", "apply_branch_mpo",
    "read_branch", "write_branch", "tdvp_branch_step",
]


def _svd_trunc(mat, chi, eps):
    U, S, Vh = _robust_svd(mat, full_matrices=False)
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


# -- an MPO along one branch of the tree ---------------------------------------
def _fuse_mpo_legs(tensor, mpo, leg_prev, leg_next):
    """Contract one MPO tensor into one node tensor, fusing the MPO bonds.

    ``tensor`` is ``(bonds..., phys)``; ``mpo`` is ``(bond_l, bond_r, out, in)``.
    The MPO bond is fused into the node's bond along the path, MPO-index major, the
    same convention :func:`fishbonett.evolve.mpo_apply.apply_mpo` uses on a chain,
    so both ends of an edge agree.  ``leg_prev``/``leg_next`` are the node's leg
    indices toward its path neighbours, or ``None`` at an end of the path, where the
    corresponding MPO bond must be trivial.  Legs the path does not use -- the
    backbone bonds of a comb's electronic site -- keep their position and size.
    """
    n_bonds = tensor.ndim - 1
    merged = np.tensordot(tensor, mpo, axes=([n_bonds], [3]))
    axis_prev, axis_next = n_bonds, n_bonds + 1
    axis_out = n_bonds + 2
    order, shape, trailing = [], [], []
    for leg in range(n_bonds):
        extra = (axis_prev if leg == leg_prev
                 else axis_next if leg == leg_next else None)
        if extra is None:
            order.append(leg)
            shape.append(merged.shape[leg])
        else:
            order.extend((extra, leg))
            shape.append(merged.shape[extra] * merged.shape[leg])
    for axis, name in ((axis_prev, leg_prev), (axis_next, leg_next)):
        if name is None:                     # unused end of the path
            if merged.shape[axis] != 1:
                raise ValueError(
                    "the MPO bond at the end of a branch must be trivial, got "
                    f"{merged.shape[axis]}")
            trailing.append(axis)            # size 1: fold in as a leading axis
    order.append(axis_out)
    shape.append(merged.shape[axis_out])
    return np.transpose(merged, trailing + order).reshape(shape)


def _svd_toward(state, i, j, chi, eps):
    """Truncate the bond ``(i, j)`` with the orthogonality centre at ``i``.

    The truncating counterpart of
    :meth:`~fishbonett.states.network.TensorNetwork._qr_toward`: isometrise ``i`` on
    every leg but the one toward ``j``, keep the leading singular values, and absorb
    the rest into ``j``, which becomes the centre.
    """
    tensor = state.tensor(i)
    leg = state._leg(i, j)
    order = [ax for ax in range(tensor.ndim) if ax != leg] + [leg]
    matrix = np.transpose(tensor, order).reshape(-1, tensor.shape[leg])
    u, singular, vh = _svd_trunc(matrix, chi, eps)
    keep = singular.shape[0]
    kept = [tensor.shape[ax] for ax in range(tensor.ndim) if ax != leg] + [keep]
    inverse = list(range(tensor.ndim - 1))
    inverse.insert(leg, tensor.ndim - 1)
    state.set_tensor(i, np.transpose(u.reshape(kept), inverse))
    state._absorb(j, i, singular[:, None] * vh)
    state.oc = j


def apply_branch_mpo(state, mpo, path, chi, eps):
    """Apply an MPO living on ``path`` to a tree state, then re-compress it.

    ``path`` is ``[system_node, mode_0, ..., mode_N]`` and ``mpo[k]`` acts on
    ``path[k]``.  This is the operator counterpart of
    :func:`symmetric_branch_swap_step`: instead of walking the system's physical
    index down the branch behind a swap network -- two truncating SVDs per bond, and
    the orthogonality centre dragged across every one of them -- the whole interval
    is carried by one low-bond operator.  Each path bond grows by the MPO bond once,
    and a single sweep truncates it back.

    The MPO's outer bonds must be trivial, so only the path's own bonds grow; a
    comb's backbone bonds are untouched, which is what makes the branches
    independent of one another.

    Gauge: the centre is moved to ``path[0]`` on entry, so the rest of the tree is
    isometric toward this branch and the truncation below is variationally
    meaningful.  It is left at ``path[0]`` on exit, ready for the next branch.
    """
    path = list(path)
    if len(path) != len(mpo):
        raise ValueError(
            f"got {len(mpo)} MPO tensors for a path of {len(path)} nodes")
    if len(path) < 2:
        return
    state.move_oc_to(path[0])
    for position, node in enumerate(path):
        leg_prev = (state._leg(node, path[position - 1])
                    if position > 0 else None)
        leg_next = (state._leg(node, path[position + 1])
                    if position + 1 < len(path) else None)
        state.set_tensor(node, _fuse_mpo_legs(
            state.tensor(node), mpo[position], leg_prev, leg_next))
    # outward QR: restores the gauge the fusion destroyed, without truncating
    for position in range(len(path) - 1):
        state._absorb(path[position + 1], path[position],
                      state._qr_toward(path[position],
                                       state._leg(path[position],
                                                  path[position + 1])))
    state.oc = path[-1]
    # inward truncating sweep, ending with the centre back at the system node
    for position in range(len(path) - 1, 0, -1):
        _svd_toward(state, path[position], path[position - 1], chi, eps)


def read_branch(state, path):
    """View one branch as a plain MPS, plus what is needed to write it back.

    The system node's *other* legs -- a comb's backbone bonds -- are folded into
    the MPS's left bond, so the branch becomes ``[(vL, p, vR), ...]`` in
    :mod:`fishbonett.evolve.mpo_apply` order and the chain routines apply
    unchanged.  Nothing about the backbone is altered by that folding; it is
    undone by :func:`write_branch`.

    This adapts the tree branch to the one-dimensional TDVP sweep.
    """
    tensors, layout = [], []
    for position, node in enumerate(path):
        tensor = state.tensor(node)
        phys = tensor.ndim - 1
        leg_next = (state._leg(node, path[position + 1])
                    if position + 1 < len(path) else None)
        leg_prev = (state._leg(node, path[position - 1])
                    if position > 0 else None)
        spectators = [ax for ax in range(phys)
                      if ax not in (leg_prev, leg_next)]
        # (spectators..., prev, phys, next): the path's two ends are the MPS bonds
        # and everything else folds into the left one
        order = (spectators + ([leg_prev] if leg_prev is not None else [])
                 + [phys] + ([leg_next] if leg_next is not None else []))
        sizes = [tensor.shape[ax] for ax in spectators]
        rows = int(np.prod(sizes, dtype=int))
        if leg_prev is not None:
            rows *= tensor.shape[leg_prev]
        cols = tensor.shape[leg_next] if leg_next is not None else 1
        tensors.append(np.transpose(tensor, order).reshape(
            rows, tensor.shape[phys], cols))
        layout.append((node, order, sizes,
                       leg_prev is not None, leg_next is not None))
    return tensors, layout


def write_branch(state, tensors, layout):
    """Undo :func:`read_branch`, restoring each node's stored leg order.

    Truncation resizes the path's own bonds, so only the folded spectator sizes are
    replayed from the layout; the rest come from the tensor as it now is.
    """
    for tensor, (node, order, sizes, has_prev, has_next) in zip(
            tensors, layout):
        rows, physical, cols = tensor.shape
        shape = list(sizes)
        if has_prev:
            shape.append(rows // int(np.prod(sizes, dtype=int)))
        shape.append(physical)
        if has_next:
            shape.append(cols)
        state.set_tensor(node, np.transpose(
            tensor.reshape(shape), np.argsort(order)))


def tdvp_branch_step(state, mpo, path, dt, chi_max, eps,
                     expand=DEFAULT_BOND_EXPAND, **krylov):
    """Advance one bath branch by a two-site TDVP step.

    :func:`apply_branch_mpo` propagates with the interval *propagator*; this one
    propagates with the *generator*, projected onto the two-site tangent space.
    Both are second order and both agree with exact diagonalization.

    Two-site TDVP evolves a two-site block and then splits it with a truncating SVD
    (:func:`~fishbonett.evolve._tdvp_sweeps._split2`), so once the cap binds it
    discards weight. A binding cap therefore requires a convergence check.
    One-site TDVP instead evolves within a fixed-bond manifold and needs the bond
    padded before propagation because it cannot grow.

    ``mpo`` is the branch Hamiltonian along ``path``, sampled at the midpoint of the
    interval by the caller, in the tree's *forward* mode order (see
    :meth:`~fishbonett.representations.interaction.InteractionRepresentation.tdvp_mpo`
    and its ``reverse`` argument).

    Gauge: the centre is moved to ``path[0]`` on entry, which makes the branch
    right-canonical with its centre at site 0 -- exactly the gauge the sweep
    expects -- and makes the rest of the tree isometric toward the branch, so the
    left environment is the identity on the folded backbone bonds.  It is left at
    ``path[0]``, where the sweep also ends.
    """
    path = list(path)
    if len(path) != len(mpo):
        raise ValueError(
            f"got {len(mpo)} MPO tensors for a path of {len(path)} nodes")
    if len(path) < 2:
        return
    state.move_oc_to(path[0])
    blocks, layout = read_branch(state, path)
    # read_branch yields (left, phys, right); the sweeps want (left, right, phys)
    tensors = [np.transpose(block, (0, 2, 1)) for block in blocks]

    environments = init_right_envs(tensors, mpo)
    # The rest of the tree is isometric toward path[0], so tracing it leaves the
    # identity on the folded bond; init_right_envs assumes a width-1 left edge.
    width = tensors[0].shape[0]
    environments[0] = np.eye(width, dtype=complex).reshape(width, 1, width)

    tensors, _environments = tdvp2sweep(
        dt, tensors, mpo, chi_max, eps, environments, expand=expand, **krylov)
    write_branch(state, [np.transpose(t, (0, 2, 1)) for t in tensors], layout)
    state.oc = path[0]


# -- whole symmetric step ------------------------------------------------------
def symmetric_tree_step(state, site_gates, edge_gates, chi, eps):
    """One 2nd-order (Strang) Trotter step over the whole tree.

    ``site_gates[i]`` is a ``(d, d)`` unitary or ``None``; ``edge_gates[(i, j)]`` a
    ``(di, dj, di*, dj*)`` unitary under either edge orientation.  **Both must be
    half-step (``dt/2``) gates** -- this applies each of them twice, as
    :func:`fishbonett.evolve.tebd.symmetric_static_step` does on a chain.

    Second order requires the complete edge sequence to be palindromic. The two
    passes use a forward sweep in **pre-order** and a backward sweep in **reverse
    post-order**, which emit ``e_1 ... e_m`` and then ``e_m ... e_1``. On a
    path this reduces exactly to the chain's up-then-down sweep; on a star it gives
    ``A/2 B/2 C/2 C/2 B/2 A/2``, where only the middle pair merges.  Both passes
    move the orthogonality centre only between a parent and a child, so no
    long-range :meth:`~fishbonett.states.tree.TreeTensorNetwork.move_oc_to` walks are needed.
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


def _swap_gate(dimension):
    """The two-site SWAP tensor in ``(out_l, out_r, in_l, in_r)`` order."""
    eye = np.eye(dimension * dimension, dtype=complex).reshape(
        dimension, dimension, dimension, dimension)
    return np.transpose(eye, (1, 0, 2, 3))


def _crossing_gate(gate):
    """Apply ``gate`` to two logical sites and exchange their positions."""
    return np.transpose(gate, (1, 0, 2, 3))


def symmetric_graph_step(state, graph_gates, system_nodes, chi, eps):
    """Apply a palindromic routed step to an electronic graph.

    ``system_nodes`` must form a path in the tensor tree and have one common
    physical dimension. ``graph_gates[(i, j)]`` are half-step gates indexed by
    *logical* system-site numbers, with ``i < j``. Sparse graphs route each
    requested edge to adjacency and immediately restore the site order. Dense
    graphs use an odd/even all-pairs network. Both routes apply the edge sequence
    and its reverse, so the step is second order and bath branches see the same
    logical system site before and after the graph sweep.
    """
    nodes = list(system_nodes)
    n = len(nodes)
    if n < 2:
        return
    dimension = state.dims[nodes[0]]
    if any(state.dims[node] != dimension for node in nodes):
        raise ValueError(
            "graph swap TEBD requires equal physical dimensions on system sites")
    if any(nodes[k + 1] not in state.adj[nodes[k]] for k in range(n - 1)):
        raise ValueError("system_nodes must be a path in the tensor-network tree")

    logical = list(range(n))
    swap = _swap_gate(dimension)

    edge_sequence = sorted(graph_gates)
    for edge in edge_sequence:
        if (not isinstance(edge, tuple) or len(edge) != 2
                or not 0 <= edge[0] < edge[1] < n):
            raise ValueError(
                f"graph edge {edge!r} must satisfy 0 <= left < right < {n}")
    sparse_work = 2 * sum(
        2 * (right - left - 1) + 1 for left, right in edge_sequence)
    dense_work = n * (n - 1)

    if sparse_work < dense_work:
        def apply_swap(position):
            left_node, right_node = nodes[position], nodes[position + 1]
            state.move_oc_to(left_node)
            apply_edge(state, left_node, right_node, swap, chi, eps)

        def apply_sparse_edge(edge):
            left, right = edge
            # Move the right logical site beside the left one, apply its gate,
            # then undo the swaps. No persistent logical-order bookkeeping is
            # needed because every routed edge restores the path immediately.
            for position in range(right - 1, left, -1):
                apply_swap(position)
            state.move_oc_to(nodes[left])
            apply_edge(
                state, nodes[left], nodes[left + 1], graph_gates[edge], chi, eps)
            for position in range(left + 1, right):
                apply_swap(position)

        for edge in edge_sequence:
            apply_sparse_edge(edge)
        for edge in reversed(edge_sequence):
            apply_sparse_edge(edge)
        return

    def cross(position):
        left_node, right_node = nodes[position], nodes[position + 1]
        left_logical, right_logical = logical[position:position + 2]
        key = tuple(sorted((left_logical, right_logical)))
        gate = graph_gates.get(key)
        if gate is None:
            combined = swap
        else:
            if left_logical > right_logical:
                gate = np.transpose(gate, (1, 0, 3, 2))
            combined = _crossing_gate(gate)
        state.move_oc_to(left_node)
        apply_edge(state, left_node, right_node, combined, chi, eps)
        logical[position], logical[position + 1] = right_logical, left_logical

    layers = [list(range(layer % 2, n - 1, 2)) for layer in range(n)]
    for positions in layers:
        for position in positions:
            cross(position)
    for positions in reversed(layers):
        for position in reversed(positions):
            cross(position)
    if logical != list(range(n)):  # defensive: the bath attachment invariant
        raise RuntimeError("electronic swap network did not restore site order")


def symmetric_branch_swap_step(state, representation, path, t0, dt, chi, eps):
    """Interaction-picture swap sweep down one bath branch and back.

    ``path`` is ``[system_node, mode_0, ..., mode_N]``.  The representation
    supplies integrated first- and second-half gates.  The system's physical
    state is walked through the branch, the outer mode is applied without a
    swap, and the reverse sweep restores it to ``path[0]`` before returning.
    """
    path = list(path)
    n_modes = len(path) - 1
    if n_modes < 1:
        return
    hdt = dt / 2.0
    inward, _ = representation.tebd_gates(
        t0, hdt, include_system=False)
    midpoint, outward = representation.tebd_gates(
        t0 + hdt, hdt, include_system=False)
    for position in range(n_modes - 1):
        state.move_oc_to(path[position])
        apply_edge(state, path[position], path[position + 1],
                   _crossing_gate(inward[position]), chi, eps)
    state.move_oc_to(path[-2])
    apply_edge(state, path[-2], path[-1], inward[-1], chi, eps)
    state.move_oc_to(path[-2])
    apply_edge(state, path[-2], path[-1], midpoint[-1], chi, eps)
    for position in range(n_modes - 2, -1, -1):
        state.move_oc_to(path[position])
        apply_edge(state, path[position], path[position + 1],
                   _crossing_gate(outward[position]), chi, eps)
