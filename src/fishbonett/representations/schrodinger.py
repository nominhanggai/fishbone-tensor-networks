"""Schrodinger representations: nothing rotated out, ``H`` static and local.

The representation with no transformation.  The chain-mapped Hamiltonian is written down as
it stands -- bath frequencies on their own sites, nearest-neighbour hoppings between
them -- so ``H`` is time-independent and its gates and MPO are built **once**.  The
price is entanglement: nothing has been removed, so the state carries the full
system-bath correlation and the bond dimensions are the largest of any representation.

This module is the representation for *any* topology: one system site with a
chain or star of modes, a comb, or an arbitrary loop-free tree of sites each
with its own bath(s).  It directly materializes the represented Hamiltonian as
an MPO for TDVP or as local terms and TEBD gates for a state tree.

.. rubric:: API

* :class:`SchrodingerRepresentation`: one system and bath as a TDVP MPO.
* :func:`terms`: systems and baths as static :class:`LocalTerms`.
* :func:`chain_terms`: nodes and edges contributed by one bath chain.
* :func:`star_terms`: nodes and edges for a shared-mode multichannel star.

The representation never advances tensor states; :mod:`fishbonett.evolve`
consumes the MPOs and gates built here.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import expm

from fishbonett.bath._coefficients import (
    chain_coefficients, combined_star_operators, require_resolved,
    star_coefficients,
)
from fishbonett.bath.coupled import bind_bath
from fishbonett.operators import annihilate, create, number, sigma_z
from fishbonett.representations._mpo import identity_product, product_sum_mpo
from fishbonett.system import check_operator

__all__ = [
    "SchrodingerRepresentation", "LocalTerms", "terms", "chain_terms",
    "star_terms", "bath_ops",
]


@dataclass
class LocalTerms:
    """A static represented Hamiltonian with one term per node and edge."""

    dims: List[int]
    edges: List[Tuple[int, int]]
    site: List[np.ndarray]
    bond: Dict[Tuple[int, int], np.ndarray]
    graph_bond: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n_nodes = len(self.dims)
        if len(self.edges) != n_nodes - 1:
            raise ValueError(
                f"edges must form a tree over {n_nodes} nodes (expected "
                f"{n_nodes - 1} edges, got {len(self.edges)})")
        if len(self.site) != n_nodes:
            raise ValueError(
                f"site has {len(self.site)} entries, expected {n_nodes}")

    @property
    def n_nodes(self):
        return len(self.dims)

    def tebd_gates(self, dt):
        """Return node and edge gates for a symmetric static TEBD step."""
        site_gates = [
            expm(-1j * hamiltonian * dt) if np.any(hamiltonian) else None
            for hamiltonian in self.site
        ]
        edge_gates = {}
        for (left, right), hamiltonian in self.bond.items():
            d_left, d_right = self.dims[left], self.dims[right]
            edge_gates[(left, right)] = expm(
                -1j * hamiltonian * dt).reshape(
                    d_left, d_right, d_left, d_right)
        return site_gates, edge_gates

    def as_tuple(self):
        """Return ``(dims, edges, site, bond)`` for tuple-based consumers."""
        return self.dims, self.edges, self.site, self.bond


class SchrodingerRepresentation:
    """Static single-system ``schrodinger-chain`` or ``-star`` Hamiltonian."""

    names = frozenset({"schrodinger-chain", "schrodinger-star"})
    static = True

    def __init__(self, *, representation, h_sys, coupling, bath):
        if representation not in self.names:
            raise ValueError(
                "representation must be 'schrodinger-chain' or "
                "'schrodinger-star'")
        self.name = representation
        self.h_sys = check_operator(h_sys, "h_sys")
        self.coupling = check_operator(
            coupling, "coupling", self.h_sys.shape[0])
        self.bath = require_resolved(bath)
        if len(self.bath.spectral_densities()) != 1:
            raise ValueError("a Schrödinger MPO requires one bath channel")
        self.pd_sys = self.h_sys.shape[0]
        self.pd_boson = [self.bath.phys_dim] * self.bath.n_modes
        self.dimensions = (self.pd_sys, *self.pd_boson)
        self._tdvp_mpo = None

    @property
    def n_sites(self):
        return len(self.dimensions)

    def tdvp_mpo(self, _time=None):
        """Return the static Hamiltonian MPO consumed by TDVP."""
        if self._tdvp_mpo is None:
            if self.name == "schrodinger-chain":
                coefficients = chain_coefficients(self.bath)
                self._tdvp_mpo = _chain_mpo(
                    self.h_sys, self.coupling,
                    coefficients.system_coupling,
                    coefficients.frequencies,
                    coefficients.hoppings,
                    self.bath.phys_dim,
                )
            else:
                coefficients = star_coefficients(self.bath)
                self._tdvp_mpo = _star_mpo(
                    self.h_sys, self.coupling,
                    coefficients.frequencies,
                    coefficients.couplings[0],
                    self.bath.phys_dim,
                )
        return self._tdvp_mpo


def _chain_mpo(h_sys, coupling, system_coupling, frequencies, hoppings,
               dimension):
    dimensions = [h_sys.shape[0]] + [dimension] * len(frequencies)
    destroy, create_op, number_op = (
        annihilate(dimension), create(dimension), number(dimension))
    products, coefficients = [], []

    row = identity_product(dimensions)
    row[0] = h_sys
    products.append(row)
    coefficients.append(1.0)

    row = identity_product(dimensions)
    row[0] = coupling
    row[1] = destroy + create_op
    products.append(row)
    coefficients.append(system_coupling)

    for mode, frequency in enumerate(frequencies):
        row = identity_product(dimensions)
        row[mode + 1] = number_op
        products.append(row)
        coefficients.append(frequency)
    for mode, hopping in enumerate(hoppings):
        for left, right in ((create_op, destroy), (destroy, create_op)):
            row = identity_product(dimensions)
            row[mode + 1] = left
            row[mode + 2] = right
            products.append(row)
            coefficients.append(hopping)
    return product_sum_mpo(dimensions, products, coefficients)


def _star_mpo(h_sys, coupling, frequencies, couplings, dimension):
    dimensions = [h_sys.shape[0]] + [dimension] * len(frequencies)
    destroy, create_op, number_op = (
        annihilate(dimension), create(dimension), number(dimension))
    products, coefficients = [], []

    row = identity_product(dimensions)
    row[0] = h_sys
    products.append(row)
    coefficients.append(1.0)
    for mode, (frequency, strength) in enumerate(
            zip(frequencies, couplings)):
        row = identity_product(dimensions)
        row[mode + 1] = number_op
        products.append(row)
        coefficients.append(frequency)

        row = identity_product(dimensions)
        row[0] = coupling
        row[mode + 1] = strength * destroy + np.conj(strength) * create_op
        products.append(row)
        coefficients.append(1.0)
    return product_sum_mpo(dimensions, products, coefficients)


def bath_ops(d):
    """``(a, a_dag, x, n)`` on a ``d``-dimensional boson site, with ``x = a + a^dag``."""
    a = annihilate(d)
    ad = a.T
    return a, ad, a + ad, ad @ a


def chain_terms(bath, site, next_node, dims, edges, site_H, edge_H):
    """Append one bath as a TEDOPA **chain** hanging off ``site``.

    Mode ``m`` gets its frequency ``w[m]`` on its own node; the system-bath coupling
    ``k[0]`` sits on the ``(site, c_0)`` edge and the mode-mode hoppings ``k[m]`` on
    the edges after it.  Returns the next free node id.
    """
    coupled = bind_bath(bath, default_operator=sigma_z)
    coefficients = chain_coefficients(coupled.bath)
    d = coupled.bath.phys_dim
    a, ad, x, numb = bath_ops(d)
    w = coefficients.frequencies
    cop = coupled.operator
    prev = site
    node = next_node
    for m in range(coefficients.n_modes):
        dims.append(d)
        site_H.append(w[m] * numb)
        edges.append((prev, node))
        if m == 0:
            edge_H[(prev, node)] = (
                coefficients.system_coupling * np.kron(cop, x))
        else:
            edge_H[(prev, node)] = coefficients.hoppings[m - 1] * (
                np.kron(ad, a) + np.kron(a, ad))
        prev = node
        node += 1
    return node


def star_terms(bath, site, next_node, dims, edges, site_H, edge_H):
    """Append one **multichannel** bath as a shared-mode star on ``site``.

    Every channel is discretized on the *same* Gauss-Legendre nodes ``omega_k``, so
    the channels are cross-correlated rather than independent, and mode ``k`` couples
    through the combined operator ``M_k = sum_c g_{c,k} O_c`` with
    ``g_{c,k} = sqrt(J_c(omega_k) w_k / pi)``.  Returns the next free node id.
    """
    coupled = bind_bath(bath)
    frequencies, coup_mat = combined_star_operators(
        coupled.bath, coupled.operators)
    dimension = coupled.bath.phys_dim
    _a, _ad, x, numb = bath_ops(dimension)
    node = next_node
    for k in range(len(frequencies)):
        dims.append(dimension)
        site_H.append(frequencies[k] * numb)
        edges.append((site, node))
        # (site op M_k) (x) (a + a^dag)
        edge_H[(site, node)] = np.kron(coup_mat[k], x)
        node += 1
    return node


def terms(sites, edges, baths, t_max=None):
    """The static Hamiltonian of a multi-site model, as :class:`LocalTerms`.

    Parameters
    ----------
    sites : list of (d, d) array
        The system-site Hamiltonians.  Nodes ``0..len(sites)-1``.
    edges : list of (i, j, C)
        System-system couplings, forming a tree over the sites; ``C`` is a
        ``(d_i*d_j, d_i*d_j)`` operator on the pair.
    baths : list
        One entry per site: a list of :class:`~fishbonett.bath.spec.Bath` (possibly
        empty).  Each bath becomes a chain -- or, if multichannel, a star -- of nodes
        hanging off its site.
    t_max : float, optional
        Propagation time, needed only to size a bath whose ``n_modes`` is automatic
        (see :meth:`fishbonett.bath.spec.Bath.resolved`).
    """
    ns = len(sites)
    dims = [np.asarray(h).shape[0] for h in sites]
    edge_list = [(i, j) for (i, j, _) in edges]
    site_H = [np.asarray(sites[i], complex).copy() for i in range(ns)]
    edge_H = {(i, j): C for (i, j, C) in edges}
    node = ns
    for i in range(ns):
        for bath in baths[i]:
            bath = bath.resolved(t_max)          # fill automatic domain / n_modes
            coupled = bind_bath(bath, default_operator=sigma_z)
            build = (star_terms if coupled.is_multichannel
                     else chain_terms)
            node = build(coupled, i, node, dims, edge_list, site_H, edge_H)
    return LocalTerms(dims=dims, edges=edge_list, site=site_H, bond=edge_H)
