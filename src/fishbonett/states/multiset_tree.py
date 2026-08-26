r"""Multi-set tree tensor-network states.

Each exact system-basis component owns an independently gauged bath TTN on the
same loop-free graph.  The outer system index is not a tensor-network bond.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from fishbonett.contract import contract
from fishbonett.states.tree import TreeTensorNetwork

__all__ = ["MultiSetTreeTensorNetwork"]


def _copy_tree(tree):
    return deepcopy(tree)


def _tree_edges(tree):
    """Canonical edge list of a tree state."""
    return tuple(
        (left, right) for left in range(tree.n) for right in tree.adj[left] if left < right
    )


def _matrix_element(bra, ket, *, operator=None, site=None):
    """Contract ``<bra|ket>`` or one local tree matrix element."""
    counter = [0]

    def new_index():
        counter[0] += 1
        return counter[0]

    messages = {}
    for node in reversed(ket._visit):
        parent = ket.parent[node]
        ket_bonds = {neighbor: new_index() for neighbor in ket.order[node]}
        bra_bonds = {neighbor: new_index() for neighbor in bra.order[node]}
        physical_ket = new_index()
        physical_bra = new_index() if node == site else physical_ket
        operands = [
            ket.T[node],
            [ket_bonds[n] for n in ket.order[node]] + [physical_ket],
            bra.T[node].conj(),
            [bra_bonds[n] for n in bra.order[node]] + [physical_bra],
        ]
        if node == site:
            operands.extend([operator, [physical_bra, physical_ket]])
        for child in ket.children[node]:
            operands.extend([messages[(child, node)], [ket_bonds[child], bra_bonds[child]]])
        if parent is None:
            operands.append([])
        else:
            operands.append([ket_bonds[parent], bra_bonds[parent]])
        messages[(node, parent)] = contract(*operands)

    return np.asarray(messages[(ket.root, None)]).reshape(-1)[0]


class MultiSetTreeTensorNetwork:
    """One bath :class:`TreeTensorNetwork` per exact system-basis state."""

    def __init__(self, sets):
        sets = list(sets)
        if not all(isinstance(tree, TreeTensorNetwork) for tree in sets):
            raise TypeError("sets must contain TreeTensorNetwork objects")
        self.sets = [_copy_tree(tree) for tree in sets]
        self._validate()

    def _validate(self):
        if not self.sets:
            raise ValueError("a multi-set tree needs at least one system set")
        reference = self.sets[0]
        edges = _tree_edges(reference)
        for index, tree in enumerate(self.sets):
            if tree.dims != reference.dims or _tree_edges(tree) != edges:
                raise ValueError("all component trees must have the same graph and dimensions")
            if tree.root != reference.root:
                raise ValueError("all component trees must use the same root")
            if any(not np.all(np.isfinite(tensor)) for tensor in tree.T):
                raise ValueError(f"sets[{index}] contains non-finite tensor values")
        self.dimensions = tuple(reference.dims)
        self.edges = edges
        self.root = reference.root

    @classmethod
    def product(cls, amplitudes, dimensions, edges, *, root=0):
        """Product vacua weighted by normalized system amplitudes."""
        amplitudes = np.asarray(amplitudes, complex).reshape(-1)
        norm = np.linalg.norm(amplitudes)
        if amplitudes.size == 0 or norm == 0 or not np.isfinite(norm):
            raise ValueError("amplitudes must have a finite nonzero norm")
        amplitudes = amplitudes / norm
        sets = []
        for amplitude in amplitudes:
            tree = TreeTensorNetwork(dimensions, edges, root=root)
            tree.T[root] *= amplitude
            sets.append(tree)
        return cls(sets)

    @property
    def n_sets(self):
        """Number of exact system-basis components."""
        return len(self.sets)

    @property
    def n_sites(self):
        """Number of bath-tree nodes in each component."""
        return len(self.dimensions)

    def copy(self):
        """Independent copy of every component tree."""
        return MultiSetTreeTensorNetwork(self.sets)

    def system_rdm(self):
        """Normalized reduced density matrix of the outer system index."""
        rho = np.empty((self.n_sets, self.n_sets), complex)
        for left in range(self.n_sets):
            for right in range(self.n_sets):
                rho[left, right] = _matrix_element(self.sets[right], self.sets[left])
        trace = np.trace(rho)
        if abs(trace) == 0 or not np.isfinite(trace):
            raise ValueError("cannot measure a zero or non-finite multi-set tree")
        return rho / trace

    def node_expectation(self, operator, site):
        """Normalized expectation of one represented bath-node operator."""
        if site < 0 or site >= self.n_sites:
            raise IndexError("tree node is outside the multi-set state")
        operator = np.asarray(operator, complex)
        dimension = self.dimensions[site]
        if operator.shape != (dimension, dimension):
            raise ValueError(
                f"operator has shape {operator.shape}, expected {(dimension, dimension)}"
            )
        numerator = sum(
            _matrix_element(tree, tree, operator=operator, site=site) for tree in self.sets
        )
        denominator = sum(_matrix_element(tree, tree) for tree in self.sets)
        if abs(denominator) == 0 or not np.isfinite(denominator):
            raise ValueError("cannot measure a zero or non-finite multi-set tree")
        return numerator / denominator

    def peak_bond(self):
        """Largest retained bond in any component TTN."""
        return max(
            (
                tensor.shape[leg]
                for tree in self.sets
                for node, tensor in enumerate(tree.T)
                for leg in range(len(tree.order[node]))
            ),
            default=1,
        )

    def set_peak_bonds(self):
        """Largest retained bond in each component TTN."""
        return tuple(
            max(
                (
                    tensor.shape[leg]
                    for node, tensor in enumerate(tree.T)
                    for leg in range(len(tree.order[node]))
                ),
                default=1,
            )
            for tree in self.sets
        )
