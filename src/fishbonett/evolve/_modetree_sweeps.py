"""Graph-generic tensor operations for balanced mode-tree propagation.

The high-level integrator applies the commuting system--bath exponential as a
tree tensor-network operator (TTNO), then restores mixed-canonical form and
truncates every edge by its Schmidt spectrum.
"""
import numpy as np
import scipy.linalg

from fishbonett.operators import annihilate
from fishbonett.evolve._modetree_core import SZ
from fishbonett.linalg import threshold_svd


def _einsum(subscripts, *operands):
    return np.einsum(subscripts, *operands, optimize=True)


def applyH1(tensor, operator, envs):
    """Apply a generic one-node TTNO effective Hamiltonian."""
    degree = tensor.ndim - 1
    if operator.ndim != degree + 2:
        raise ValueError("operator and state node degrees differ")
    next_label = 0
    bra, op_bond, ket = [], [], []
    for _ in range(degree):
        bra.append(next_label); op_bond.append(next_label + 1)
        ket.append(next_label + 2); next_label += 3
    pout, pin = next_label, next_label + 1
    operands = [tensor, ket + [pin], operator, op_bond + [pout, pin]]
    for leg in range(degree):
        operands.extend((envs[leg], [bra[leg], op_bond[leg], ket[leg]]))
    return np.einsum(*operands, bra + [pout], optimize=True)


def applyH0(center, first_env, second_env):
    """Apply the effective operator on one tree edge."""
    return np.einsum(
        "amc,cr,bmr->ab", first_env, center, second_env, optimize=True)


def update_env(tensor, operator, free_leg, envs):
    """Contract a node into the message sent along ``free_leg``."""
    degree = tensor.ndim - 1
    next_label = 0
    bra, op_bond, ket = [], [], []
    for _ in range(degree):
        bra.append(next_label); op_bond.append(next_label + 1)
        ket.append(next_label + 2); next_label += 3
    pout, pin = next_label, next_label + 1
    operands = [tensor.conj(), bra + [pout], operator,
                op_bond + [pout, pin], tensor, ket + [pin]]
    for leg in range(degree):
        if leg != free_leg:
            operands.extend((envs[leg], [bra[leg], op_bond[leg], ket[leg]]))
    output = [bra[free_leg], op_bond[free_leg], ket[free_leg]]
    return np.einsum(*operands, output, optimize=True)


def qr_leg(tensor, leg):
    """Make every axis except ``leg`` isometric and return the residual matrix."""
    axes = [axis for axis in range(tensor.ndim) if axis != leg] + [leg]
    matrix = np.transpose(tensor, axes).reshape(-1, tensor.shape[leg])
    q, r = np.linalg.qr(matrix, mode="reduced")
    rank = q.shape[1]
    shape = [tensor.shape[axis] for axis in axes[:-1]] + [rank]
    q = q.reshape(shape)
    inverse = list(range(tensor.ndim - 1))
    inverse.insert(leg, tensor.ndim - 1)
    return np.transpose(q, inverse), r


def contractC(tensor, center, leg):
    """Absorb ``center[new, old]`` into one tensor bond."""
    merged = np.tensordot(center, tensor, axes=([1], [leg]))
    return np.moveaxis(merged, 0, leg)


def init_envs(nodes, root):
    """Build child-to-parent TTNO environment messages."""
    messages = {}

    def recurse(node_id):
        node = nodes[node_id]
        for child in node.children:
            recurse(child)
        if node.parent is not None:
            envs = {
                1 + index: messages[child]
                for index, child in enumerate(node.children)
            }
            messages[node_id] = update_env(node.tensor, node.mpo, 0, envs)
    recurse(root)
    return messages


def _node_envs(nodes, messages, nid, root):
    node = nodes[nid]
    envs = {}
    if node.parent is not None:
        envs[0] = messages[node.parent]
    offset = 0 if node.parent is None else 1
    for index, child in enumerate(node.children):
        envs[offset + index] = messages[child]
    return envs


def measure_rdm_oc(nodes, root):
    """Reduced density matrix of the root in root-canonical form."""
    tensor = nodes[root].tensor
    bonds = tuple(range(tensor.ndim - 1))
    rho = np.tensordot(tensor, tensor.conj(), axes=(bonds, bonds))
    trace = np.trace(rho)
    if abs(trace) == 0:
        raise ValueError("cannot measure a zero tree state")
    return rho / trace


def measure_node_rdm(nodes, target):
    """Reduced density matrix of any mode-tree node without moving the gauge."""
    if target < 0 or target >= len(nodes):
        raise ValueError("target node is outside the mode tree")

    def message(node_id, recipient):
        node = nodes[node_id]
        neighbours = node.neighbours
        ket = list(range(len(neighbours)))
        bra = list(range(len(neighbours), 2 * len(neighbours)))
        physical = 2 * len(neighbours)
        operands = [node.tensor, ket + [physical],
                    node.tensor.conj(), bra + [physical]]
        for leg, neighbour in enumerate(neighbours):
            if neighbour == recipient:
                continue
            operands.extend((message(neighbour, node_id), [ket[leg], bra[leg]]))
        recipient_leg = neighbours.index(recipient)
        operands.append([ket[recipient_leg], bra[recipient_leg]])
        return np.einsum(*operands, optimize=True)

    node = nodes[target]
    degree = len(node.neighbours)
    ket = list(range(degree))
    bra = list(range(degree, 2 * degree))
    physical_ket, physical_bra = 2 * degree, 2 * degree + 1
    operands = [node.tensor, ket + [physical_ket],
                node.tensor.conj(), bra + [physical_bra]]
    for leg, neighbour in enumerate(node.neighbours):
        operands.extend((message(neighbour, target), [ket[leg], bra[leg]]))
    operands.append([physical_ket, physical_bra])
    rho = np.einsum(*operands, optimize=True)
    trace = np.trace(rho)
    if abs(trace) == 0:
        raise ValueError("cannot measure a zero mode-tree state")
    return rho / trace


def measure_sz_oc(nodes, root):
    return float(np.trace(measure_rdm_oc(nodes, root) @ SZ).real)


def _contract_subtree(nodes, nid):
    """Contract a state subtree, keeping its parent bond and physical legs."""
    subtree = []

    def visit(node_id):
        subtree.append(node_id)
        for child in nodes[node_id].children:
            visit(child)
    visit(nid)
    edge_labels, next_label = {}, 0

    def edge_label(a, b):
        nonlocal next_label
        edge = frozenset((a, b))
        if edge not in edge_labels:
            edge_labels[edge] = next_label
            next_label += 1
        return edge_labels[edge]

    physical_nodes = [node_id for node_id in subtree
                      if nodes[node_id].kind != "internal"]
    physical_nodes.sort(key=lambda node_id: (
        0 if node_id == nid and nodes[node_id].kind == "spin" else 1,
        -1 if nodes[node_id].mode is None else nodes[node_id].mode))
    phys_labels, operands = {}, []
    for node_id in subtree:
        node = nodes[node_id]
        labels = [edge_label(node_id, neighbour) for neighbour in node.neighbours]
        phys_labels[node_id] = next_label
        next_label += 1
        operands.extend((node.tensor, labels + [phys_labels[node_id]]))
    output = []
    if nodes[nid].parent is not None:
        output.append(edge_label(nid, nodes[nid].parent))
    output.extend(phys_labels[node_id] for node_id in physical_nodes)
    return np.einsum(*operands, output, optimize=True)


def measure_sz_tree(nodes, root):
    wavefunction = np.asarray(_contract_subtree(nodes, root))
    flat = wavefunction.reshape(nodes[root].d, -1)
    rho = flat @ flat.conj().T
    norm = float(np.trace(rho).real)
    return float(np.trace(rho @ SZ).real / norm), norm


def _dummy_env():
    return np.ones((1, 1, 1), complex)


def _quad_eig(d):
    a = annihilate(d)
    return np.linalg.eigh(a + a.conj().T)


def build_coupling_op(nodes, root, d_nt, cop):
    """Build the exact commuting-coupling exponential as a TTNO.

    Diagonalizing the system operator gives
    ``sum_k P_k (x) product_j exp(-i mu_k B_j)``.  The eigenvalue label is
    copied along every tree edge, so the TTNO bond is only the system dimension.
    """
    amplitudes = np.asarray(d_nt, complex)
    coupling = np.asarray(cop, complex)
    eigenvalues, eigenvectors = np.linalg.eigh(coupling)
    rank = len(eigenvalues)
    for node in nodes:
        degree = len(node.neighbours)
        operator = np.zeros([rank] * degree + [node.d, node.d], complex)
        for channel, eigenvalue in enumerate(eigenvalues):
            index = (channel,) * degree
            if node.id == root:
                vector = eigenvectors[:, channel]
                local = np.outer(vector, vector.conj())
            elif node.kind == "internal":
                local = np.ones((1, 1), complex)
            else:
                amplitude = amplitudes[node.mode]
                a = annihilate(node.d)
                generator = amplitude * a + np.conj(amplitude) * a.conj().T
                local = scipy.linalg.expm(-1j * eigenvalue * generator)
            operator[index] = local
        node.opmpo = operator
    return nodes


def apply_op_node(tensor, operator):
    """Apply one TTNO node and fuse each operator bond with the state bond."""
    degree = tensor.ndim - 1
    if operator.ndim != degree + 2:
        raise ValueError("state and operator node degrees differ")
    combined = np.tensordot(tensor, operator, axes=([-1], [-1]))
    permutation = []
    for leg in range(degree):
        permutation.extend((leg, degree + leg))
    permutation.append(2 * degree)
    combined = np.transpose(combined, permutation)
    shape = [tensor.shape[leg] * operator.shape[leg] for leg in range(degree)]
    shape.append(operator.shape[-2])
    return np.ascontiguousarray(combined.reshape(shape))


def apply_coupling(nodes):
    for node in nodes:
        node.tensor = apply_op_node(node.tensor, node.opmpo)
    return nodes


def apply_sys(nodes, root, gate):
    tensor = nodes[root].tensor
    transformed = np.tensordot(tensor, np.asarray(gate), axes=([-1], [1]))
    nodes[root].tensor = transformed
    return nodes


def canon_to_root(nodes, root):
    """Canonicalize all subtrees toward ``root`` by post-order QR."""
    def recurse(node_id):
        node = nodes[node_id]
        for child in node.children:
            recurse(child)
        if node.parent is not None:
            node.tensor, residual = qr_leg(node.tensor, 0)
            parent = nodes[node.parent]
            parent_leg = parent.children.index(node_id)
            if parent.parent is not None:
                parent_leg += 1
            parent.tensor = contractC(parent.tensor, residual, parent_leg)
    recurse(root)
    return nodes


def _svd_split(tensor, leg, D, eps):
    axes = [axis for axis in range(tensor.ndim) if axis != leg] + [leg]
    matrix = np.transpose(tensor, axes).reshape(-1, tensor.shape[leg])
    return threshold_svd(matrix, eps, max_rank=D)


def truncate_from_root(nodes, root, D, eps):
    """Schmidt-truncate each edge while walking outward and back to the root."""
    def recurse(parent_id):
        parent = nodes[parent_id]
        for child_index, child_id in enumerate(parent.children):
            leg = child_index + (1 if parent.parent is not None else 0)
            tensor = parent.tensor
            axes = [axis for axis in range(tensor.ndim) if axis != leg] + [leg]
            other_shape = [tensor.shape[axis] for axis in axes[:-1]]
            u, singular, vh = _svd_split(tensor, leg, D, eps)
            rank = len(singular)
            u_tensor = u.reshape(other_shape + [rank])
            inverse = list(range(tensor.ndim - 1))
            inverse.insert(leg, tensor.ndim - 1)
            parent.tensor = np.transpose(u_tensor, inverse)
            transfer = singular[:, None] * vh
            child = nodes[child_id]
            child.tensor = contractC(child.tensor, transfer, 0)
            recurse(child_id)
            # Return the centre to the parent before processing its next child.
            child.tensor, residual = qr_leg(child.tensor, 0)
            parent.tensor = contractC(parent.tensor, residual, leg)
    recurse(root)
    norm = np.linalg.norm(nodes[root].tensor)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("tree truncation produced a zero or non-finite state")
    nodes[root].tensor /= norm
    return nodes
