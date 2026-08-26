"""Interaction-chain MPOs for an exciton with independent local baths."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from fishbonett._products import ScaledTreeIdentity
from fishbonett.operators import annihilate, displacement
from fishbonett.representations._mpo import identity_product, product_sum_mpo
from fishbonett.representations.interaction import InteractionRepresentation
from fishbonett.system import check_operator

__all__ = ["ExcitonInteractionRepresentation"]


def _mode_operator(dimension, coefficient):
    destroy = annihilate(dimension)
    return coefficient * destroy + np.conj(coefficient) * destroy.conj().T


def _one_excitation_gate_mpo(hamiltonian, duration):
    """Exact electronic gate on the one-excitation sector, as an MPO."""
    hamiltonian = np.asarray(hamiltonian, complex)
    levels = hamiltonian.shape[0]
    unitary = la.expm(-1j * float(duration) * hamiltonian)
    dimensions = [2] * levels
    number = np.diag([0.0, 1.0]).astype(complex)
    plus = np.array([[0.0, 0.0], [1.0, 0.0]], complex)
    minus = plus.conj().T
    products, coefficients = [], []
    for target in range(levels):
        for source in range(levels):
            row = identity_product(dimensions)
            if target == source:
                row[target] = number
            else:
                row[target] = plus
                row[source] = minus
            products.append(row)
            coefficients.append(unitary[target, source])
    return product_sum_mpo(dimensions, products, coefficients)


def _identity_mpo_tensor(rank, dimension):
    tensor = np.zeros((rank, rank, dimension, dimension), complex)
    identity = np.eye(dimension, dtype=complex)
    for branch in range(rank):
        tensor[branch, branch] = identity
    return tensor


def _system_first_mpo(hamiltonian, branches, time):
    """Low-rank MPO for ``H_s + sum_r P_r sum_k B_rk(t)``."""
    system_dimension = hamiltonian.shape[0]
    count = len(branches)
    done = count
    first = np.zeros((1, count + 1, system_dimension, system_dimension), complex)
    first[0, done] = hamiltonian
    for branch, (level, _representation) in enumerate(branches):
        first[0, branch, level, level] = 1.0
    mpo = [first]
    for branch, (_level, representation) in enumerate(branches):
        coefficients = representation.coefficients(time)
        for dimension, coefficient in zip(representation.pd_boson, coefficients):
            tensor = np.zeros((count + 1, count + 1, dimension, dimension), complex)
            identity = np.eye(dimension, dtype=complex)
            for state in range(count + 1):
                tensor[state, state] = identity
            tensor[branch, done] = _mode_operator(dimension, coefficient)
            mpo.append(tensor)
    mpo[-1] = mpo[-1][:, done : done + 1]
    return mpo


def _interleaved_mpo(hamiltonian, branches, time):
    """Finite-state MPO for local baths and arbitrary excitonic hopping."""
    levels = hamiltonian.shape[0]
    start = 0
    bath_open = 1
    plus_open = bath_open + levels
    minus_open = plus_open + levels
    done = minus_open + levels
    rank = done + 1
    branch_by_level = {level: representation for level, representation in branches}
    number = np.diag([0.0, 1.0]).astype(complex)
    plus = np.array([[0.0, 0.0], [1.0, 0.0]], complex)
    minus = plus.conj().T
    mpo = []

    def propagator(dimension):
        tensor = np.zeros((rank, rank, dimension, dimension), complex)
        identity = np.eye(dimension, dtype=complex)
        for state in range(rank):
            tensor[state, state] = identity
        return tensor

    for level in range(levels):
        electronic = propagator(2)
        electronic[start, done] += hamiltonian[level, level] * number
        if level in branch_by_level:
            electronic[start, bath_open + level] = number
        electronic[start, plus_open + level] = plus
        electronic[start, minus_open + level] = minus
        for earlier in range(level):
            electronic[plus_open + earlier, done] += hamiltonian[earlier, level] * minus
            electronic[minus_open + earlier, done] += hamiltonian[level, earlier] * plus
        mpo.append(electronic)
        representation = branch_by_level.get(level)
        if representation is None:
            continue
        for dimension, coefficient in zip(
            representation.pd_boson, representation.coefficients(time)
        ):
            mode = propagator(dimension)
            mode[bath_open + level, done] = _mode_operator(dimension, coefficient)
            mpo.append(mode)
    mpo[0] = mpo[0][start : start + 1]
    mpo[-1] = mpo[-1][:, done : done + 1]
    return mpo


def _tree_topology(levels, branches):
    """Dummy electronic connector path with one bath-mode branch per level."""
    dimensions = [1] * levels
    edges = [(level, level + 1) for level in range(levels - 1)]
    mode_nodes = {}
    next_node = levels
    for level, representation in branches:
        nodes = list(range(next_node, next_node + representation.len_boson))
        mode_nodes[level] = tuple(nodes)
        path = [level, *nodes]
        edges.extend(zip(path[:-1], path[1:]))
        dimensions.extend(representation.pd_boson)
        next_node += representation.len_boson
    return tuple(dimensions), tuple(edges), mode_nodes


def _tree_adjacency(count, edges):
    adjacency = [[] for _ in range(count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return adjacency


def _identity_ttno(dimensions, edges, coefficient=1.0, root=0):
    return ScaledTreeIdentity(
        complex(coefficient),
        tuple(dimensions),
        tuple(tuple(edge) for edge in edges),
        int(root),
    )


def _local_sum_ttno(dimensions, edges, local_operators, constant=0.0, root=0):
    """Bond-two TTNO for ``constant I + sum_i h_i`` on a tree."""
    count = len(dimensions)
    adjacency = _tree_adjacency(count, edges)
    parent = [None] * count
    children = [[] for _ in range(count)]
    stack = [root]
    seen = {root}
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            parent[neighbor] = node
            children[node].append(neighbor)
            stack.append(neighbor)
    tensors = []
    for node, dimension in enumerate(dimensions):
        degree = len(adjacency[node])
        tensor = np.zeros((2,) * degree + (dimension, dimension), complex)
        identity = np.eye(dimension, dtype=complex)
        local = np.asarray(local_operators.get(node, np.zeros((dimension, dimension))), complex)
        parent_leg = None if parent[node] is None else adjacency[node].index(parent[node])
        child_legs = [adjacency[node].index(child) for child in children[node]]
        for bond_state in np.ndindex(*((2,) * degree)):
            child_count = sum(bond_state[leg] for leg in child_legs)
            if parent_leg is None:
                if child_count == 0:
                    tensor[bond_state] += constant * identity + local
                elif child_count == 1:
                    tensor[bond_state] += identity
                continue
            parent_state = bond_state[parent_leg]
            if parent_state == child_count:
                tensor[bond_state] += identity
            if parent_state == child_count + 1:
                tensor[bond_state] += local
        tensors.append(tensor)
    return tensors


class ExcitonInteractionRepresentation:
    """Resolved independent baths in an interaction-chain layout."""

    static = False

    def __init__(self, hamiltonian, baths, horizon, *, layout="system-first"):
        if layout not in {"system-first", "interleaved"}:
            raise ValueError("layout must be 'system-first' or 'interleaved'")
        self.hamiltonian = check_operator(hamiltonian, "hamiltonian")
        baths = tuple(baths)
        if len(baths) != self.hamiltonian.shape[0]:
            raise ValueError("baths must contain one entry per electronic level")
        self.layout = layout
        self.branches = []
        for level, bath in enumerate(baths):
            if bath is None:
                continue
            resolved = bath.resolved(horizon)
            representation = InteractionRepresentation(
                representation="interaction-chain",
                h_sys=np.zeros((1, 1), complex),
                coupling=np.ones((1, 1), complex),
                bath=resolved,
            ).build()
            self.branches.append((level, representation))
        if not self.branches:
            raise ValueError("at least one resolved bath branch is required")
        dimensions = []
        if layout == "system-first":
            dimensions.append(self.hamiltonian.shape[0])
            for _level, representation in self.branches:
                dimensions.extend(representation.pd_boson)
        else:
            by_level = dict(self.branches)
            for level in range(self.hamiltonian.shape[0]):
                dimensions.append(2)
                if level in by_level:
                    dimensions.extend(by_level[level].pd_boson)
        self.dimensions = tuple(dimensions)
        (
            self.tree_dimensions,
            self.tree_edges,
            self.tree_mode_nodes,
        ) = _tree_topology(self.hamiltonian.shape[0], self.branches)

    def tdvp_mpo(self, time=None):
        """Instantaneous MPO in the selected physical-site ordering."""
        time = 0.0 if time is None else float(time)
        if self.layout == "system-first":
            return _system_first_mpo(self.hamiltonian, self.branches, time)
        return _interleaved_mpo(self.hamiltonian, self.branches, time)

    @property
    def branch_sites(self):
        """MPS positions of each bath-bearing electronic site and its modes."""
        sites = {}
        if self.layout == "system-first":
            position = 1
            for level, representation in self.branches:
                modes = tuple(range(position, position + representation.len_boson))
                sites[level] = (0, modes)
                position += representation.len_boson
            return sites
        by_level = dict(self.branches)
        position = 0
        for level in range(self.hamiltonian.shape[0]):
            electronic = position
            position += 1
            representation = by_level.get(level)
            if representation is None:
                continue
            modes = tuple(range(position, position + representation.len_boson))
            sites[level] = (electronic, modes)
            position += representation.len_boson
        return sites

    def trotter_mpo(self, time, dt):
        """Conditional-displacement MPO for the integrated bath coupling."""
        if self.layout == "system-first":
            levels = self.hamiltonian.shape[0]
            first = np.zeros((1, levels, levels, levels), complex)
            for level in range(levels):
                first[0, level, level, level] = 1.0
            mpo = [first]
            for owner, representation in self.branches:
                for dimension, coefficient in zip(
                    representation.pd_boson,
                    representation.interval_coefficients(time, dt),
                ):
                    tensor = _identity_mpo_tensor(levels, dimension)
                    tensor[owner, owner] = displacement(
                        -1j * np.conj(coefficient), dimension
                    )
                    mpo.append(tensor)
            last = mpo[-1]
            collapsed = np.zeros(
                (levels, 1, last.shape[2], last.shape[3]), complex
            )
            for branch in range(levels):
                collapsed[branch, 0] = last[branch, branch]
            mpo[-1] = collapsed
            return mpo

        branch_by_level = dict(self.branches)
        mpo = []
        vacuum = np.diag([1.0, 0.0]).astype(complex)
        occupied = np.diag([0.0, 1.0]).astype(complex)
        for level in range(self.hamiltonian.shape[0]):
            representation = branch_by_level.get(level)
            if representation is None:
                mpo.append(np.eye(2, dtype=complex).reshape(1, 1, 2, 2))
                continue
            electronic = np.zeros((1, 2, 2, 2), complex)
            electronic[0, 0] = vacuum
            electronic[0, 1] = occupied
            mpo.append(electronic)
            coefficients = representation.interval_coefficients(time, dt)
            for mode, (dimension, coefficient) in enumerate(zip(
                representation.pd_boson, coefficients
            )):
                tensor = _identity_mpo_tensor(2, dimension)
                tensor[1, 1] = displacement(
                    -1j * np.conj(coefficient), dimension
                )
                if mode == len(coefficients) - 1:
                    collapsed = np.zeros((2, 1, dimension, dimension), complex)
                    collapsed[0, 0] = tensor[0, 0]
                    collapsed[1, 0] = tensor[1, 1]
                    tensor = collapsed
                mpo.append(tensor)
        return mpo

    def electronic_mpo(self, duration):
        """MPO for exact electronic evolution over ``duration``."""
        if self.layout == "system-first":
            unitary = la.expm(-1j * float(duration) * self.hamiltonian)
            mpo = [unitary.reshape(1, 1, *unitary.shape)]
            mpo.extend(
                np.eye(dimension, dtype=complex).reshape(
                    1, 1, dimension, dimension
                )
                for dimension in self.dimensions[1:]
            )
            return mpo

        electronic = _one_excitation_gate_mpo(self.hamiltonian, duration)
        branch_by_level = dict(self.branches)
        mpo = []
        for level, tensor in enumerate(electronic):
            mpo.append(tensor)
            representation = branch_by_level.get(level)
            if representation is None:
                continue
            rank = tensor.shape[1]
            mpo.extend(
                _identity_mpo_tensor(rank, dimension)
                for dimension in representation.pd_boson
            )
        return mpo

    def tebd_gates(self, time, dt):
        """Integrated system--mode gates for the system-first swap network."""
        if self.layout != "system-first":
            raise ValueError(
                "interleaved TEBD uses one local swap network per bath branch"
            )
        first_order, reversed_order = [], []
        first_mode = True
        levels = self.hamiltonian.shape[0]
        for owner, representation in self.branches:
            projector = np.zeros((levels, levels), complex)
            projector[owner, owner] = 1.0
            for dimension, coefficient in zip(
                representation.pd_boson,
                representation.interval_coefficients(time, dt),
            ):
                bath = _mode_operator(dimension, coefficient)
                generator = np.kron(projector, bath)
                if first_mode:
                    generator += float(dt) * np.kron(
                        self.hamiltonian, np.eye(dimension)
                    )
                    first_mode = False
                gate = la.expm(-1j * generator).reshape(
                    levels, dimension, levels, dimension
                )
                first_order.append(gate)
                reversed_order.append(np.transpose(gate, (1, 0, 3, 2)))
        return first_order, reversed_order

    def multiset_tree_operators(self, time=None):
        """Bath-only TTNO block matrix for coupled multi-set tree TDVP.

        Scalar identity blocks are returned as compact descriptors so the
        propagator can use cross-overlap environments without materializing a
        general TTNO at every electronic hopping matrix element.
        """
        time = 0.0 if time is None else float(time)
        levels = self.hamiltonian.shape[0]
        blocks = []
        branch_by_level = dict(self.branches)
        for output in range(levels):
            row = []
            for input_ in range(levels):
                coefficient = self.hamiltonian[output, input_]
                if output != input_:
                    row.append(
                        None
                        if coefficient == 0
                        else _identity_ttno(
                            self.tree_dimensions,
                            self.tree_edges,
                            coefficient,
                        )
                    )
                    continue
                local = {}
                representation = branch_by_level.get(output)
                if representation is not None:
                    nodes = self.tree_mode_nodes[output]
                    for node, dimension, value in zip(
                        nodes,
                        representation.pd_boson,
                        representation.coefficients(time),
                    ):
                        local[node] = _mode_operator(dimension, value)
                if local:
                    row.append(
                        _local_sum_ttno(
                            self.tree_dimensions,
                            self.tree_edges,
                            local,
                            constant=coefficient,
                        )
                    )
                else:
                    row.append(
                        _identity_ttno(
                            self.tree_dimensions,
                            self.tree_edges,
                            coefficient,
                        )
                    )
            blocks.append(row)
        return blocks

    @property
    def electronic_sites(self):
        """MPS indices of local electronic sites in the interleaved layout."""
        if self.layout != "interleaved":
            return (0,)
        sites = []
        by_level = dict(self.branches)
        position = 0
        for level in range(self.hamiltonian.shape[0]):
            sites.append(position)
            position += 1
            if level in by_level:
                position += by_level[level].len_boson
        return tuple(sites)


def _interleaved_initial_mps(amplitudes, dimensions, electronic_sites):
    """Exact one-excitation MPS with every oscillator in its vacuum."""
    amplitudes = np.asarray(amplitudes, complex).reshape(-1)
    norm = np.linalg.norm(amplitudes)
    if amplitudes.size == 0 or norm == 0 or not np.isfinite(norm):
        raise ValueError("amplitudes must have a finite nonzero norm")
    amplitudes = amplitudes / norm
    dimensions = tuple(int(value) for value in dimensions)
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("dimensions must contain positive integers")
    electronic_sites = tuple(electronic_sites)
    if len(electronic_sites) != amplitudes.size:
        raise ValueError("electronic_sites must contain one site per amplitude")
    if (
        len(set(electronic_sites)) != len(electronic_sites)
        or any(site < 0 or site >= len(dimensions) for site in electronic_sites)
        or any(dimensions[site] != 2 for site in electronic_sites)
    ):
        raise ValueError("electronic_sites must be distinct two-level MPS sites")
    level_at = {site: level for level, site in enumerate(electronic_sites)}
    tensors = []
    for site, dimension in enumerate(dimensions):
        left = 1 if site == 0 else 2
        right = 1 if site == len(dimensions) - 1 else 2
        tensor = np.zeros((left, right, dimension), complex)
        if site in level_at:
            level = level_at[site]
            if left == 1:
                if right == 1:
                    tensor[0, 0, 1] = amplitudes[level]
                else:
                    tensor[0, 0, 0] = 1.0
                    tensor[0, 1, 1] = amplitudes[level]
            elif right == 1:
                tensor[1, 0, 0] = 1.0
                tensor[0, 0, 1] = amplitudes[level]
            else:
                tensor[0, 0, 0] = 1.0
                tensor[1, 1, 0] = 1.0
                tensor[0, 1, 1] = amplitudes[level]
        else:
            if right == 1 and left > 1:
                tensor[1, 0, 0] = 1.0
            else:
                tensor[0, 0, 0] = 1.0
            if left > 1 and right > 1:
                tensor[1, 1, 0] = 1.0
        tensors.append(tensor)
    return tensors
