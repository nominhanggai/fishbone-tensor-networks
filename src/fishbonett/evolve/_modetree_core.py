"""Balanced mode-tree topology and generic tree-operator construction.

The interaction representation removes the free bath and leaves a sum of modes
coupled to the system. The finite conditional-coupling exponential is represented
as a low-bond tree operator, applied to a balanced binary tree tensor network,
then compressed by canonicalization and relative Schmidt truncation. Time
propagation uses a symmetric second-order split with the system Hamiltonian.
"""
import numpy as np

from fishbonett.operators import annihilate, create, sigma_x, sigma_z

SZ = sigma_z.astype(complex)
SX = sigma_x.astype(complex)


class Node:
    """One node of a rooted tree tensor network.

    Bond legs are ordered ``[parent, children...]`` (the parent leg is absent at
    the root), followed by the physical leg.
    """

    __slots__ = (
        "id", "kind", "parent", "children", "mode", "d", "tensor", "mpo",
        "opmpo",
    )

    def __init__(self, nid, kind, d=1):
        self.id = int(nid)
        self.kind = kind
        self.parent = None
        self.children = []
        self.mode = None
        self.d = int(d)
        self.tensor = None
        self.mpo = None
        self.opmpo = None

    @property
    def neighbours(self):
        return (([self.parent] if self.parent is not None else [])
                + list(self.children))


def _resolve_sys(hsys, cop, init, V, eps):
    h = (np.asarray(hsys, complex) if hsys is not None
         else V * SX + 0.5 * eps * SZ)
    coupling = np.asarray(cop, complex) if cop is not None else SZ.copy()
    if h.ndim != 2 or h.shape[0] != h.shape[1] or coupling.shape != h.shape:
        raise ValueError("hsys and cop must be square matrices of equal shape")
    if init is None:
        state = np.zeros(h.shape[0], complex)
        state[0] = 1.0
    else:
        state = np.asarray(init, complex).reshape(-1)
    if state.shape != (h.shape[0],):
        raise ValueError("initial state dimension does not match hsys")
    return h, coupling, state / np.linalg.norm(state), h.shape[0]


def build_balanced_tree(n_modes, d, d_sys=2):
    """Build a binary tree whose leaves, in order, are the bath modes."""
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    nodes = [Node(0, "spin", d_sys)]
    leaves = [None] * n_modes

    def new_node(kind, physical):
        node = Node(len(nodes), kind, physical)
        nodes.append(node)
        return node.id

    def connect(parent, child):
        nodes[child].parent = parent
        nodes[parent].children.append(child)

    def attach_group(parent, modes):
        if len(modes) == 1:
            leaf = new_node("leaf", d)
            nodes[leaf].mode = modes[0]
            leaves[modes[0]] = leaf
            connect(parent, leaf)
            return
        midpoint = len(modes) // 2
        for group in (modes[:midpoint], modes[midpoint:]):
            if len(group) == 1:
                attach_group(parent, group)
            else:
                internal = new_node("internal", 1)
                connect(parent, internal)
                attach_group(internal, group)

    attach_group(0, list(range(n_modes)))
    return nodes, 0, leaves


def tree_depth(nodes, root):
    """Maximum edge distance from ``root`` to a leaf."""
    def depth(node_id):
        children = nodes[node_id].children
        return 0 if not children else 1 + max(depth(child) for child in children)
    return depth(root)


def init_state(nodes, root, sys_state=None):
    """Install a system-plus-vacuum product state on an existing tree."""
    system = (np.array([1.0, 0.0], complex) if sys_state is None
              else np.asarray(sys_state, complex).reshape(-1))
    system = system / np.linalg.norm(system)
    for node in nodes:
        shape = [1] * len(node.neighbours) + [node.d]
        tensor = np.zeros(shape, complex)
        if node.id == root:
            tensor[(0,) * len(node.neighbours)] = system
        else:
            tensor[(0,) * len(node.neighbours) + (0,)] = 1.0
        node.tensor = tensor
    return nodes


def _local_products(nodes, root, dcoup, hsys, cop):
    """Full local-operator rows for the interaction-picture Hamiltonian."""
    leaves = sorted((node for node in nodes if node.kind == "leaf"),
                    key=lambda node: node.mode)
    if len(dcoup) != len(leaves):
        raise ValueError("one coupling amplitude is required per mode")
    identities = {node.id: np.eye(node.d, dtype=complex) for node in nodes}
    products = []
    system_term = identities.copy()
    system_term[root] = np.asarray(hsys, complex)
    products.append(system_term)
    for leaf, amplitude in zip(leaves, dcoup):
        a = annihilate(leaf.d)
        term = identities.copy()
        term[root] = np.asarray(cop, complex)
        term[leaf.id] = amplitude * a + np.conj(amplitude) * a.conj().T
        products.append(term)
    return products


def build_tree_mpo(nodes, root, dcoup, hsys, cop):
    """Compile a sum of product operators as a tree tensor-network operator."""
    products = _local_products(nodes, root, np.asarray(dcoup, complex), hsys, cop)
    rank = len(products)
    for node in nodes:
        degree = len(node.neighbours)
        operator = np.zeros([rank] * degree + [node.d, node.d], complex)
        for term, product in enumerate(products):
            operator[(term,) * degree] = product[node.id]
        node.mpo = operator
    return nodes


def _contract_mpo(nodes, nid):
    """Contract one operator subtree, leaving its parent bond open if present."""
    subtree = []

    def visit(node_id):
        subtree.append(node_id)
        for child in nodes[node_id].children:
            visit(child)
    visit(nid)
    edge_labels, next_label = {}, 0

    def label_for(a, b):
        nonlocal next_label
        edge = frozenset((a, b))
        if edge not in edge_labels:
            edge_labels[edge] = next_label
            next_label += 1
        return edge_labels[edge]

    physical = [node_id for node_id in subtree
                if nodes[node_id].kind != "internal"]
    physical.sort(key=lambda node_id: (
        0 if node_id == nid and nodes[node_id].kind == "spin" else 1,
        -1 if nodes[node_id].mode is None else nodes[node_id].mode))
    out_labels, in_labels, operands = [], [], []
    for node_id in subtree:
        node = nodes[node_id]
        labels = [label_for(node_id, neighbour) for neighbour in node.neighbours]
        out_label, in_label = next_label, next_label + 1
        next_label += 2
        labels.extend((out_label, in_label))
        operands.extend((node.mpo, labels))
        if node_id in physical:
            out_labels.append(out_label)
            in_labels.append(in_label)
    parent_label = []
    if nodes[nid].parent is not None:
        parent_label = [label_for(nid, nodes[nid].parent)]
    result = np.einsum(*operands, parent_label + out_labels + in_labels,
                       optimize=True)
    leaf_count = sum(nodes[node_id].kind == "leaf" for node_id in subtree)
    return result, leaf_count


def hamiltonian_from_mpo(nodes, root, n_modes, d):
    """Materialize a small tree operator as a dense validation matrix."""
    operator, _ = _contract_mpo(nodes, root)
    physical_nodes = [root] + [
        node.id for node in sorted(
            (node for node in nodes if node.kind == "leaf"),
            key=lambda node: node.mode)
    ]
    dimension = int(np.prod([nodes[node_id].d for node_id in physical_nodes]))
    return np.asarray(operator).reshape(dimension, dimension)


def _embed(op, site, dims):
    factors = [np.eye(dim, dtype=complex) for dim in dims]
    factors[site] = np.asarray(op, complex)
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _hamiltonian_direct(dcoup, V, eps, d, n):
    """Dense interaction-picture Hamiltonian used only for validation."""
    amplitudes = np.asarray(dcoup, complex)
    if len(amplitudes) != int(n):
        raise ValueError("dcoup length differs from n")
    dims = [2] + [int(d)] * int(n)
    hamiltonian = _embed(V * SX + 0.5 * eps * SZ, 0, dims)
    a = annihilate(d)
    for mode, amplitude in enumerate(amplitudes):
        bath = amplitude * a + np.conj(amplitude) * create(d)
        hamiltonian += _embed(SZ, 0, dims) @ _embed(bath, mode + 1, dims)
    return hamiltonian
