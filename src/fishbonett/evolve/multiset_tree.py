"""Coupled two-site TDVP for multi-set tree tensor networks."""

from __future__ import annotations

from itertools import count

import numpy as np

from fishbonett._products import ScaledTreeIdentity
from fishbonett.contract import _contract_cached
from fishbonett.evolve._tdvp_sweeps import DEFAULT_BOND_EXPAND
from fishbonett.evolve._validation import positive_integer, time_steps
from fishbonett.evolve.multiset import _evolve_components, _pairwise_apply
from fishbonett.linalg import Truncation, threshold_svd
from fishbonett.states.multiset_tree import MultiSetTreeTensorNetwork

__all__ = ["multiset_tree_tdvp2_sweep", "run_multiset_tree_hamiltonian"]


def _tree_contract(operands):
    """Contract with a path cached by leg pattern and current bond shapes."""
    return _contract_cached(*operands)


def _is_tree_identity(operator):
    return isinstance(operator, ScaledTreeIdentity)


def _identity_scale(operator):
    return complex(operator.coefficient)


def _validate_operators(state, operators):
    count = state.n_sets
    if len(operators) != count or any(len(row) != count for row in operators):
        raise ValueError("operator block dimensions do not match the number of sets")
    reference = state.sets[0]
    for row in operators:
        for operator in row:
            if operator is None:
                continue
            if _is_tree_identity(operator):
                edges = {
                    tuple(sorted((int(left), int(right))))
                    for left, right in operator.edges
                }
                if tuple(operator.dimensions) != state.dimensions or edges != set(
                    state.edges
                ):
                    raise ValueError("identity TTNO descriptor does not match the state")
                if operator.root != state.root:
                    raise ValueError("identity TTNO descriptor uses a different root")
                if not np.isfinite(_identity_scale(operator)):
                    raise ValueError("identity TTNO coefficient must be finite")
                continue
            if len(operator) != state.n_sites:
                raise ValueError("every TTNO block must match the component tree")
            for node, tensor in enumerate(operator):
                degree = len(reference.order[node])
                expected = state.dimensions[node]
                if tensor.ndim != degree + 2 or tensor.shape[-2:] != (
                    expected,
                    expected,
                ):
                    raise ValueError(
                        f"operator node {node} does not match degree {degree} "
                        f"and physical dimension {expected}"
                    )
                if not np.all(np.isfinite(tensor)):
                    raise ValueError(f"operator node {node} contains non-finite values")
            for left, right in state.edges:
                left_leg = reference.order[left].index(right)
                right_leg = reference.order[right].index(left)
                if operator[left].shape[left_leg] != operator[right].shape[right_leg]:
                    raise ValueError(f"TTNO bond {(left, right)} is incompatible")


def _node_message(bra, ket, operator, node, exclude, messages):
    """Contract one operator subtree toward ``exclude``."""
    neighbors = ket.order[node]
    counter = count(1)
    op_bonds = {neighbor: next(counter) for neighbor in neighbors}
    ket_bonds = {neighbor: next(counter) for neighbor in neighbors}
    bra_bonds = {neighbor: next(counter) for neighbor in neighbors}
    physical_out, physical_in = next(counter), next(counter)
    operands = [
        operator[node],
        [op_bonds[n] for n in neighbors] + [physical_out, physical_in],
        ket.T[node],
        [ket_bonds[n] for n in neighbors] + [physical_in],
        bra.T[node].conj(),
        [bra_bonds[n] for n in neighbors] + [physical_out],
    ]
    for neighbor in neighbors:
        if neighbor == exclude:
            continue
        operands.extend(
            [
                messages[(neighbor, node)],
                [op_bonds[neighbor], ket_bonds[neighbor], bra_bonds[neighbor]],
            ]
        )
    operands.append([op_bonds[exclude], ket_bonds[exclude], bra_bonds[exclude]])
    return _tree_contract(operands)


def _identity_node_message(bra, ket, node, exclude, messages):
    """Cross-overlap message for an identity operator subtree."""
    neighbors = ket.order[node]
    counter = count(1)
    ket_bonds = {neighbor: next(counter) for neighbor in neighbors}
    bra_bonds = {neighbor: next(counter) for neighbor in neighbors}
    physical = next(counter)
    operands = [
        ket.T[node],
        [ket_bonds[n] for n in neighbors] + [physical],
        bra.T[node].conj(),
        [bra_bonds[n] for n in neighbors] + [physical],
    ]
    for neighbor in neighbors:
        if neighbor == exclude:
            continue
        operands.extend(
            [
                messages[(neighbor, node)],
                [ket_bonds[neighbor], bra_bonds[neighbor]],
            ]
        )
    operands.append([ket_bonds[exclude], bra_bonds[exclude]])
    return _tree_contract(operands)


def _initialize_messages(state, operators):
    all_messages = [[None for _ in range(state.n_sets)] for _ in range(state.n_sets)]
    for output in range(state.n_sets):
        for input_ in range(state.n_sets):
            operator = operators[output][input_]
            if operator is None:
                continue
            messages = {}
            identity = _is_tree_identity(operator)

            tree = state.sets[input_]
            for node in reversed(tree._visit):
                parent = tree.parent[node]
                if parent is None:
                    continue
                if identity:
                    messages[(node, parent)] = _identity_node_message(
                        state.sets[output], tree, node, parent, messages
                    )
                else:
                    messages[(node, parent)] = _node_message(
                        state.sets[output],
                        tree,
                        operator,
                        node,
                        parent,
                        messages,
                    )
            for node in tree._visit:
                for child in tree.children[node]:
                    if identity:
                        messages[(node, child)] = _identity_node_message(
                            state.sets[output], tree, node, child, messages
                        )
                    else:
                        messages[(node, child)] = _node_message(
                            state.sets[output],
                            tree,
                            operator,
                            node,
                            child,
                            messages,
                        )
            all_messages[output][input_] = messages
    return all_messages


def _apply_one(value, operator, messages, tree, node):
    neighbors = tree.order[node]
    counter = count(1)
    op_bonds = {neighbor: next(counter) for neighbor in neighbors}
    ket_bonds = {neighbor: next(counter) for neighbor in neighbors}
    bra_bonds = {neighbor: next(counter) for neighbor in neighbors}
    physical_out, physical_in = next(counter), next(counter)
    operands = [
        operator[node],
        [op_bonds[n] for n in neighbors] + [physical_out, physical_in],
        value,
        [ket_bonds[n] for n in neighbors] + [physical_in],
    ]
    for neighbor in neighbors:
        operands.extend(
            [
                messages[(neighbor, node)],
                [op_bonds[neighbor], ket_bonds[neighbor], bra_bonds[neighbor]],
            ]
        )
    operands.append([bra_bonds[n] for n in neighbors] + [physical_out])
    return _tree_contract(operands)


def _apply_identity_one(value, messages, tree, node, coefficient):
    """Apply a scaled identity through cross-set one-site environments."""
    neighbors = tree.order[node]
    counter = count(1)
    ket_bonds = {neighbor: next(counter) for neighbor in neighbors}
    bra_bonds = {neighbor: next(counter) for neighbor in neighbors}
    physical = next(counter)
    operands = [
        value,
        [ket_bonds[n] for n in neighbors] + [physical],
    ]
    for neighbor in neighbors:
        operands.extend(
            [
                messages[(neighbor, node)],
                [ket_bonds[neighbor], bra_bonds[neighbor]],
            ]
        )
    operands.append([bra_bonds[n] for n in neighbors] + [physical])
    return coefficient * _tree_contract(operands)


def _merge_edge(tree, source, destination):
    source_leg = tree.order[source].index(destination)
    destination_leg = tree.order[destination].index(source)
    merged = np.tensordot(
        tree.T[source], tree.T[destination], axes=([source_leg], [destination_leg])
    )
    source_count = len(tree.order[source]) - 1
    destination_count = len(tree.order[destination]) - 1
    # tensordot gives source external bonds, p_source, destination external
    # bonds, p_destination.  Centres keep both physical legs at the end.
    order = (
        list(range(source_count))
        + list(range(source_count + 1, source_count + 1 + destination_count))
        + [source_count, merged.ndim - 1]
    )
    return np.transpose(merged, order)


def _apply_two(value, operator, messages, tree, source, destination):
    source_neighbors = [n for n in tree.order[source] if n != destination]
    destination_neighbors = [n for n in tree.order[destination] if n != source]
    counter = count(1)
    shared_operator = next(counter)
    source_op = {n: next(counter) for n in source_neighbors}
    destination_op = {n: next(counter) for n in destination_neighbors}
    source_ket = {n: next(counter) for n in source_neighbors}
    destination_ket = {n: next(counter) for n in destination_neighbors}
    source_bra = {n: next(counter) for n in source_neighbors}
    destination_bra = {n: next(counter) for n in destination_neighbors}
    source_out, source_in = next(counter), next(counter)
    destination_out, destination_in = next(counter), next(counter)

    source_operator_indices = []
    for neighbor in tree.order[source]:
        source_operator_indices.append(
            shared_operator if neighbor == destination else source_op[neighbor]
        )
    destination_operator_indices = []
    for neighbor in tree.order[destination]:
        destination_operator_indices.append(
            shared_operator if neighbor == source else destination_op[neighbor]
        )
    operands = [
        operator[source],
        source_operator_indices + [source_out, source_in],
        operator[destination],
        destination_operator_indices + [destination_out, destination_in],
        value,
        [source_ket[n] for n in source_neighbors]
        + [destination_ket[n] for n in destination_neighbors]
        + [source_in, destination_in],
    ]
    for neighbor in source_neighbors:
        operands.extend(
            [
                messages[(neighbor, source)],
                [source_op[neighbor], source_ket[neighbor], source_bra[neighbor]],
            ]
        )
    for neighbor in destination_neighbors:
        operands.extend(
            [
                messages[(neighbor, destination)],
                [
                    destination_op[neighbor],
                    destination_ket[neighbor],
                    destination_bra[neighbor],
                ],
            ]
        )
    operands.append(
        [source_bra[n] for n in source_neighbors]
        + [destination_bra[n] for n in destination_neighbors]
        + [source_out, destination_out]
    )
    return _tree_contract(operands)


def _apply_identity_two(
    value,
    messages,
    tree,
    source,
    destination,
    coefficient,
):
    """Apply a scaled identity through cross-set two-site environments."""
    source_neighbors = [n for n in tree.order[source] if n != destination]
    destination_neighbors = [n for n in tree.order[destination] if n != source]
    counter = count(1)
    source_ket = {n: next(counter) for n in source_neighbors}
    destination_ket = {n: next(counter) for n in destination_neighbors}
    source_bra = {n: next(counter) for n in source_neighbors}
    destination_bra = {n: next(counter) for n in destination_neighbors}
    source_physical, destination_physical = next(counter), next(counter)
    operands = [
        value,
        [source_ket[n] for n in source_neighbors]
        + [destination_ket[n] for n in destination_neighbors]
        + [source_physical, destination_physical],
    ]
    for neighbor in source_neighbors:
        operands.extend(
            [
                messages[(neighbor, source)],
                [source_ket[neighbor], source_bra[neighbor]],
            ]
        )
    for neighbor in destination_neighbors:
        operands.extend(
            [
                messages[(neighbor, destination)],
                [destination_ket[neighbor], destination_bra[neighbor]],
            ]
        )
    operands.append(
        [source_bra[n] for n in source_neighbors]
        + [destination_bra[n] for n in destination_neighbors]
        + [source_physical, destination_physical]
    )
    return coefficient * _tree_contract(operands)


def _split_edge(tree, source, destination, center, max_bond, eps, expand):
    source_neighbors = [n for n in tree.order[source] if n != destination]
    destination_neighbors = [n for n in tree.order[destination] if n != source]
    source_count = len(source_neighbors)
    destination_count = len(destination_neighbors)
    source_physical = source_count + destination_count
    destination_physical = source_physical + 1
    order = (
        list(range(source_count))
        + [source_physical]
        + list(range(source_count, source_count + destination_count))
        + [destination_physical]
    )
    matrix_view = np.transpose(center, order)
    left_shape = matrix_view.shape[: source_count + 1]
    right_shape = matrix_view.shape[source_count + 1 :]
    matrix = matrix_view.reshape(int(np.prod(left_shape)), int(np.prod(right_shape)))
    u, singular, vh = threshold_svd(
        matrix,
        eps,
        max_rank=max_bond,
        extra_rank=max(0, int(expand)),
    )
    rank = singular.size
    left = u.reshape(*left_shape, rank)
    right = (singular[:, None] * vh).reshape(rank, *right_shape)

    source_axes = []
    external = iter(range(source_count))
    for neighbor in tree.order[source]:
        source_axes.append(left.ndim - 1 if neighbor == destination else next(external))
    source_axes.append(source_count)
    tree.T[source] = np.transpose(left, source_axes)

    destination_axes = []
    external = iter(range(1, destination_count + 1))
    for neighbor in tree.order[destination]:
        destination_axes.append(0 if neighbor == source else next(external))
    destination_axes.append(destination_count + 1)
    tree.T[destination] = np.transpose(right, destination_axes)
    tree.oc = destination


def _walk(tree):
    crossings = []

    def visit(node):
        for child in tree.children[node]:
            crossings.append((node, child))
            visit(child)
            crossings.append((child, node))

    visit(tree.root)
    return crossings


def multiset_tree_tdvp2_sweep(
    dt,
    state,
    operators,
    max_bond,
    eps,
    *,
    expand=DEFAULT_BOND_EXPAND,
    **krylov,
):
    """Advance all component TTNs by one coupled tree TDVP2 sweep."""
    if not isinstance(state, MultiSetTreeTensorNetwork):
        raise TypeError("state must be a MultiSetTreeTensorNetwork")
    _validate_operators(state, operators)
    for tree in state.sets:
        tree.move_oc_to(state.root)
    messages = _initialize_messages(state, operators)
    crossings = _walk(state.sets[0])
    half = 0.5 * float(dt)
    for crossing_index, (source, destination) in enumerate(crossings):
        centers = [_merge_edge(tree, source, destination) for tree in state.sets]

        def apply_two(
            output, input_, value, *, source=source, destination=destination
        ):
            operator = operators[output][input_]
            if operator is None:
                return None
            if _is_tree_identity(operator):
                return _apply_identity_two(
                    value,
                    messages[output][input_],
                    state.sets[input_],
                    source,
                    destination,
                    _identity_scale(operator),
                )
            return _apply_two(
                value,
                operator,
                messages[output][input_],
                state.sets[input_],
                source,
                destination,
            )

        centers = _evolve_components(centers, _pairwise_apply(apply_two), -1j * half, **krylov)
        for set_index, center in enumerate(centers):
            _split_edge(
                state.sets[set_index],
                source,
                destination,
                center,
                max_bond,
                eps,
                expand,
            )
        for output in range(state.n_sets):
            for input_ in range(state.n_sets):
                operator = operators[output][input_]
                if operator is None:
                    continue
                if _is_tree_identity(operator):
                    messages[output][input_][(source, destination)] = (
                        _identity_node_message(
                            state.sets[output],
                            state.sets[input_],
                            source,
                            destination,
                            messages[output][input_],
                        )
                    )
                else:
                    messages[output][input_][(source, destination)] = _node_message(
                        state.sets[output],
                        state.sets[input_],
                        operator,
                        source,
                        destination,
                        messages[output][input_],
                    )
        next_crossing = (
            crossings[crossing_index + 1] if crossing_index + 1 < len(crossings) else None
        )
        if next_crossing is None or next_crossing == (destination, source):
            continue
        next_node = next_crossing[1]
        parent = state.sets[0].parent[destination]
        # The tree tangent projector subtracts a node projector once for every
        # independent way branches meet there (degree - 1).  Arrival from the
        # parent and departure back to it each carry a half step; moving between
        # two child branches carries the corresponding full backward step.  On
        # a path this reduces exactly to the usual TDVP2 half-step corrections.
        correction = float(dt) if source != parent and next_node != parent else half
        centers = [tree.T[destination] for tree in state.sets]

        def apply_one(output, input_, value, *, destination=destination):
            operator = operators[output][input_]
            if operator is None:
                return None
            if _is_tree_identity(operator):
                return _apply_identity_one(
                    value,
                    messages[output][input_],
                    state.sets[input_],
                    destination,
                    _identity_scale(operator),
                )
            return _apply_one(
                value,
                operator,
                messages[output][input_],
                state.sets[input_],
                destination,
            )

        centers = _evolve_components(centers, _pairwise_apply(apply_one), 1j * correction, **krylov)
        for set_index, center in enumerate(centers):
            state.sets[set_index].T[destination] = center
    return state


def run_multiset_tree_hamiltonian(
    representation,
    *,
    state,
    dt,
    nsteps,
    trunc=None,
    bond_dim=None,
    trunc_eps=None,
    krylov=30,
    tol=1e-7,
    eshift=False,
    bond_expand=None,
    observe=None,
    progress=None,
):
    """Propagate a bath-tree block Hamiltonian with coupled tree TDVP2."""
    if not isinstance(state, MultiSetTreeTensorNetwork):
        raise TypeError("state must be a MultiSetTreeTensorNetwork")
    dt, nsteps = time_steps(dt, nsteps)
    truncation = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
    krylov = positive_integer(krylov, "krylov")
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    if bond_expand is not None and (
        isinstance(bond_expand, (bool, np.bool_))
        or not isinstance(bond_expand, (int, np.integer))
        or bond_expand < 0
    ):
        raise ValueError("bond_expand must be a non-negative integer or None")
    if tuple(representation.tree_dimensions) != state.dimensions:
        raise ValueError("representation and multi-set tree dimensions differ")
    represented_edges = {
        tuple(sorted((int(left), int(right)))) for left, right in representation.tree_edges
    }
    if represented_edges != set(state.edges):
        raise ValueError("representation and multi-set tree edges differ")
    expand = DEFAULT_BOND_EXPAND if bond_expand is None else int(bond_expand)
    measure = (lambda current: current.system_rdm()) if observe is None else observe
    observations, peak_bonds, set_bonds = [], [], []
    options = {"m": krylov, "tol": float(tol), "eshift": eshift}
    for step in range(nsteps):
        operators = representation.multiset_tree_operators((step + 0.5) * dt)
        state = multiset_tree_tdvp2_sweep(
            dt,
            state,
            operators,
            truncation.max_bond,
            truncation.eps,
            expand=expand,
            **options,
        )
        observations.append(measure(state))
        peak_bonds.append(state.peak_bond())
        set_bonds.append(state.set_peak_bonds())
        if progress is not None:
            progress(
                {
                    "step": step,
                    "n_steps": nsteps,
                    "t": (step + 1) * dt,
                    "bond": peak_bonds[-1],
                    "rdm": observations[-1],
                    "state": state,
                }
            )
    return (
        np.arange(1, nsteps + 1, dtype=float) * dt,
        np.asarray(observations),
        np.asarray(peak_bonds, dtype=int),
        np.asarray(set_bonds, dtype=int),
        state,
    )
