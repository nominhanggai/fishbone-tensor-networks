"""Finite-temperature pure-state preparation for short system chains.

The baths use thermofield/T-TEDOPA internally.  This module supplies the
corresponding construction for a finite interacting *system*: purify its Gibbs
density matrix with one inert ancilla per site, group each physical site and its
ancilla into a supersite, and factor the resulting pure state into an exact MPS.

The ancillas are state bookkeeping only.  Hamiltonians, bath couplings and
observables are lifted as ``O_physical (x) I_ancilla``.
"""
from __future__ import annotations

import numpy as np

from fishbonett.contract import _contract_cached
from fishbonett.linalg import full_svd

__all__ = ["GibbsPurification"]


def _validate_square(operator, dimension, name):
    value = np.asarray(operator, complex)
    if value.shape != (dimension, dimension):
        raise ValueError(
            f"{name} has shape {value.shape}, expected {(dimension, dimension)}")
    if not np.allclose(value, value.conj().T, atol=1e-10):
        raise ValueError(f"{name} must be Hermitian")
    return value


def _embed(operator, sites, dimensions):
    """Embed an operator on ordered ``sites`` into a dense product space."""
    sites = tuple(int(site) for site in sites)
    selected = [dimensions[site] for site in sites]
    operator = np.asarray(operator, complex)
    expected = int(np.prod(selected, dtype=int))
    if operator.shape != (expected, expected):
        raise ValueError(
            f"operator on sites {sites} has shape {operator.shape}, expected "
            f"{(expected, expected)}")
    n = len(dimensions)
    operands = [operator.reshape(*(selected + selected)),
                list(sites) + [n + site for site in sites]]
    for site, dimension in enumerate(dimensions):
        if site not in sites:
            operands.extend([np.eye(dimension), [site, n + site]])
    operands.append(list(range(2 * n)))
    return _contract_cached(*operands).reshape(
        int(np.prod(dimensions)), int(np.prod(dimensions)))


def _lift(operator, dimensions):
    """Lift an operator to interleaved ``(physical, ancilla)`` supersites."""
    dimensions = tuple(int(d) for d in dimensions)
    total = int(np.prod(dimensions, dtype=int))
    operator = np.asarray(operator, complex)
    if operator.shape != (total, total):
        raise ValueError(
            f"operator has shape {operator.shape}, expected {(total, total)}")
    n = len(dimensions)
    # labels: physical outputs, ancilla outputs, physical inputs, ancilla inputs
    physical_out = list(range(n))
    ancilla_out = list(range(n, 2 * n))
    physical_in = list(range(2 * n, 3 * n))
    ancilla_in = list(range(3 * n, 4 * n))
    operands = [operator.reshape(*(dimensions + dimensions)),
                physical_out + physical_in]
    for i, dimension in enumerate(dimensions):
        operands.extend([
            np.eye(dimension), [ancilla_out[i], ancilla_in[i]]])
    output = []
    for i in range(n):
        output.extend([physical_out[i], ancilla_out[i]])
    for i in range(n):
        output.extend([physical_in[i], ancilla_in[i]])
    operands.append(output)
    supersite_dimensions = [d * d for d in dimensions]
    return _contract_cached(*operands).reshape(
        int(np.prod(supersite_dimensions)),
        int(np.prod(supersite_dimensions)))


def _factor_mps(vector, dimensions):
    """Exact left-to-right SVD of ``vector`` into ``(left, phys, right)``."""
    dimensions = list(map(int, dimensions))
    work = np.asarray(vector, complex).reshape(dimensions)
    tensors = []
    left = 1
    for dimension in dimensions[:-1]:
        matrix = work.reshape(left * dimension, -1)
        u, singular, vh = full_svd(matrix, full_matrices=False)
        # Exact preparation: remove only numerical zeroes produced by SVD.
        cutoff = np.finfo(float).eps * max(matrix.shape) * singular[0]
        rank = max(1, int(np.count_nonzero(singular > cutoff)))
        tensors.append(u[:, :rank].reshape(left, dimension, rank))
        work = singular[:rank, None] * vh[:rank]
        left = rank
    tensors.append(work.reshape(left, dimensions[-1], 1))
    return tuple(tensors)


class GibbsPurification:
    """Exact Gibbs purification of a finite chain of system sites.

    Parameters
    ----------
    sites
        Physical single-site Hamiltonians.
    backbone
        Physical nearest-neighbour Hamiltonians, one per adjacent pair.
    temperature, beta
        Temperature or inverse temperature in the package's natural units.
        Supply exactly one.

    Notes
    -----
    Exact diagonalization scales exponentially with the number of physical
    sites.  This helper is intended for short systems whose baths, rather than
    the system itself, dominate the tensor-network size.
    """

    def __init__(self, sites, backbone, *, temperature=None, beta=None):
        if (temperature is None) == (beta is None):
            raise ValueError("provide exactly one of temperature or beta")
        if temperature is not None:
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            beta = 1.0 / float(temperature)
        elif beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta = float(beta)
        self.temperature = np.inf if self.beta == 0 else 1.0 / self.beta

        raw_sites = [np.asarray(value, complex) for value in sites]
        self.physical_dimensions = tuple(value.shape[0] for value in raw_sites)
        self.n_sites = len(raw_sites)
        if not self.n_sites:
            raise ValueError("a Gibbs purification needs at least one site")
        self.physical_sites = tuple(
            _validate_square(value, self.physical_dimensions[i], f"sites[{i}]")
            for i, value in enumerate(raw_sites))
        if len(backbone) != self.n_sites - 1:
            raise ValueError("backbone must have n_sites - 1 entries")
        self.physical_backbone = tuple(
            _validate_square(
                value,
                self.physical_dimensions[i] * self.physical_dimensions[i + 1],
                f"backbone[{i}]")
            for i, value in enumerate(backbone))

        hamiltonian = np.zeros(
            (int(np.prod(self.physical_dimensions)),) * 2, complex)
        for i, value in enumerate(self.physical_sites):
            hamiltonian += _embed(value, [i], self.physical_dimensions)
        for i, value in enumerate(self.physical_backbone):
            hamiltonian += _embed(value, [i, i + 1], self.physical_dimensions)
        self.hamiltonian = hamiltonian

        energy, vectors = np.linalg.eigh(hamiltonian)
        weights = np.exp(-self.beta * (energy - energy.min()))
        weights /= weights.sum()
        sqrt_density = (vectors * np.sqrt(weights)) @ vectors.conj().T
        # sqrt(rho)[physical, ancilla], then interleave the site axes.
        tensor = sqrt_density.reshape(
            *(self.physical_dimensions + self.physical_dimensions))
        permutation = []
        for i in range(self.n_sites):
            permutation.extend([i, self.n_sites + i])
        vector = np.transpose(tensor, permutation).reshape(-1)
        self.vector = vector / np.linalg.norm(vector)
        self.dimensions = tuple(d * d for d in self.physical_dimensions)
        self.tensors = _factor_mps(self.vector, self.dimensions)

        self.sites = tuple(
            self.lift_operator(value, [i])
            for i, value in enumerate(self.physical_sites))
        self.backbone = tuple(
            self.lift_operator(value, [i, i + 1])
            for i, value in enumerate(self.physical_backbone))

    def lift_operator(self, operator, sites):
        """Return ``operator (x) I_ancilla`` on the selected supersites."""
        sites = tuple(int(site) for site in sites)
        if len(set(sites)) != len(sites):
            raise ValueError("sites must not contain duplicates")
        if any(site < 0 or site >= self.n_sites for site in sites):
            raise ValueError("site index outside the purified system")
        return _lift(operator, [self.physical_dimensions[site] for site in sites])

    def lift_site_operator(self, operator, site):
        """Lift a physical operator onto one purified system site.

        This constructs the local physical-plus-ancilla matrix.  The site where
        a bath is attached is specified separately by the model's ``baths``
        mapping.
        """
        return self.lift_operator(operator, [site])

    def initialize_tree(self, state, system_nodes):
        """Embed the purified system MPS into a bath-bearing tree state."""
        nodes = tuple(int(node) for node in system_nodes)
        if len(nodes) != self.n_sites:
            raise ValueError(
                f"purification has {self.n_sites} sites, model has {len(nodes)}")
        if tuple(state.dims[node] for node in nodes) != self.dimensions:
            raise ValueError("purification supersite dimensions do not match model")
        for position, node in enumerate(nodes):
            source = self.tensors[position]  # left, physical, right
            neighbours = list(state.neighbours(node))
            shape = []
            source_axis = []
            for neighbour in neighbours:
                if position > 0 and neighbour == nodes[position - 1]:
                    shape.append(source.shape[0]); source_axis.append(0)
                elif position + 1 < self.n_sites and neighbour == nodes[position + 1]:
                    shape.append(source.shape[2]); source_axis.append(2)
                else:
                    shape.append(1); source_axis.append(None)
            target = np.zeros(shape + [source.shape[1]], complex)
            # Remove the dummy MPS end bonds, then order the remaining system
            # bonds as the tree's neighbour legs request.
            compact = source
            axes = [0, 1, 2]
            if position == 0:
                compact = compact[0]
                axes = [1, 2]
            if position == self.n_sites - 1:
                compact = np.take(compact, 0, axis=axes.index(2))
                axes.remove(2)
            ordered_axes = [axes.index(axis) for axis in source_axis
                            if axis is not None] + [axes.index(1)]
            compact = np.transpose(compact, ordered_axes)
            index = tuple(0 if axis is None else slice(None)
                          for axis in source_axis) + (slice(None),)
            target[index] = compact
            state.set_tensor(node, target)
        # The left-to-right SVD leaves its orthogonality centre on the last
        # system tensor.  Record that gauge truth, then move it to the tree root:
        # static-tree TEBD starts its first edge sweep there.
        state.oc = nodes[-1]
        state.move_oc_to(state.root)
