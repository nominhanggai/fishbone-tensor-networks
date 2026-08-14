"""Star and chain interaction representations of a harmonic bath.

The construction follows the mathematical order directly:

1. discretize the bath into independent star modes ``a_k``;
2. rotate with respect to ``sum_k omega_k a_k^dag a_k``;
3. retain the star operators for ``interaction-star``, or apply the
   star-to-chain transform for ``interaction-chain``.

Consequently the star coefficients are ``g_k exp(-i omega_k t)`` and the chain
coefficients are ``d_n(t) = sum_k U[n,k] g_k exp(-i omega_k t)``.  Diagonalizing
a finite chain can recover the same star quadrature, but that is a numerical
discretization route rather than the definition of this representation.

The representation directly materializes its Hamiltonian for the supported
tensor-network algorithms through :meth:`tdvp_mpo`, :meth:`trotter_mpo`, and
:meth:`tebd_gates`.  It does not advance a tensor-network state.
"""

import numpy as np
import scipy.linalg as la

from fishbonett.bath._coefficients import require_resolved, star_coefficients
from fishbonett.bath.conventions import integrated_free_phase
from fishbonett.linalg import expm_gate, kron
from fishbonett.operators import annihilate, create
from fishbonett.representations._mpo import identity_product, product_sum_mpo
from fishbonett.system import check_operator

__all__ = ["InteractionRepresentation", "star_edges"]


def star_edges(n_modes):
    """Interaction-graph edges for a mode-decoupled representation."""
    return [(0, mode) for mode in range(1, n_modes + 1)]


def _swap_gate_pairs(hamiltonians, factor=1):
    """Exponentiate interval Hamiltonians in both swap-network leg orders."""
    first, second = [], []
    for hamiltonian, d_boson, d_system in hamiltonians:
        dense = (hamiltonian.toarray()
                 if hasattr(hamiltonian, "toarray") else hamiltonian)
        gate = expm_gate(dense / factor, 1).reshape(
            d_boson, d_system, d_boson, d_system,
        ).transpose(1, 0, 3, 2)
        first.append(gate)
        second.append(np.transpose(gate, (1, 0, 3, 2)))
    return first, second


class InteractionRepresentation:
    """The ``interaction-star`` or ``interaction-chain`` Hamiltonian.

    Parameters
    ----------
    representation
        Exactly ``"interaction-star"`` or ``"interaction-chain"``.
    h_sys, coupling
        Hermitian system Hamiltonian and coupling operator.
    bath
        Resolved bath specification. The representation discretizes it into a
        finite star and applies the star-to-chain transform when requested.
    """

    names = frozenset({"interaction-star", "interaction-chain"})
    static = False

    def __init__(self, *, representation, h_sys, coupling, bath):
        if representation not in self.names:
            raise ValueError(
                "representation must be 'interaction-star' or "
                "'interaction-chain'")
        self.name = representation
        self.h_sys = check_operator(h_sys, "h_sys")
        self.pd_sys = self.h_sys.shape[0]
        self.coupling = check_operator(coupling, "coupling", self.pd_sys)
        self.bath = require_resolved(bath)
        self.pd_boson = [self.bath.phys_dim] * self.bath.n_modes
        self.len_boson = len(self.pd_boson)
        if not self.pd_boson:
            raise ValueError("bath must include at least one mode")
        self.frequencies = None
        self.star_couplings = None
        self.star_to_chain = None

    @property
    def dimensions(self):
        return (self.pd_sys, *self.pd_boson)

    @property
    def n_sites(self):
        return len(self.dimensions)

    def build(self):
        """Prepare the finite star data and optional star-to-chain transform."""
        star = star_coefficients(self.bath)
        if star.n_channels != 1:
            raise ValueError("an interaction representation requires one channel")
        if self.name == "interaction-chain" and star.transform is None:
            raise ValueError(
                "interaction-chain requires a star-to-chain transform")
        self.frequencies = star.frequencies
        self.star_couplings = star.couplings[0]
        self.star_to_chain = star.transform
        return self

    def _express(self, star_values):
        values = np.asarray(star_values, complex)
        if self.name == "interaction-star":
            return values
        return self.star_to_chain @ values

    def coefficients(self, t):
        """Instantaneous coupling coefficients in this representation."""
        phases = np.exp(-1j * self.frequencies * float(t))
        return self._express(self.star_couplings * phases)

    def tdvp_mpo(self, t=None):
        """Return the instantaneous Hamiltonian MPO consumed by TDVP."""
        if self.frequencies is None:
            raise ValueError("build the interaction representation first")
        coefficients = self.coefficients(0.0 if t is None else t)
        dimensions = list(self.dimensions)
        destroy = annihilate(self.pd_boson[0])
        create_op = create(self.pd_boson[0])
        products, values = [], []

        row = identity_product(dimensions)
        row[0] = self.h_sys
        products.append(row)
        values.append(1.0)
        # TDVP stores the bath sites in reverse coefficient order.
        for mode, amplitude in enumerate(coefficients[::-1]):
            row = identity_product(dimensions)
            row[0] = self.coupling
            row[mode + 1] = (
                amplitude * destroy + np.conj(amplitude) * create_op)
            products.append(row)
            values.append(1.0)
        return product_sum_mpo(dimensions, products, values)

    def interval_coefficients(self, t, delta):
        """Couplings integrated over ``[t, t + delta]``."""
        phases = np.array([
            integrated_free_phase(omega, t, delta)
            for omega in self.frequencies
        ])
        return self._express(self.star_couplings * phases)

    def two_site_hamiltonians(self, t, delta, include_system=True):
        """One interval-integrated system–mode Hamiltonian per bath mode.

        Matrices use ``(mode, system)`` ordering.  :meth:`tebd_gates`
        exponentiates and arranges them for the swap network.
        """
        out = []
        for dimension, amplitude in zip(
                self.pd_boson, self.interval_coefficients(t, delta)):
            destroy = annihilate(dimension)
            bath_operator = (
                amplitude * destroy
                + np.conj(amplitude) * destroy.conj().T
            )
            out.append((
                kron(bath_operator, self.coupling),
                dimension,
                self.pd_sys,
            ))
        if include_system:
            dimension = self.pd_boson[0]
            system_term = delta * kron(np.eye(dimension), self.h_sys)
            out[0] = (out[0][0] + system_term, dimension, self.pd_sys)
        return out

    def tebd_gates(self, t, dt, factor=1, include_system=True):
        """Return both leg orderings of the interval's swap-network gates."""
        return _swap_gate_pairs(
            self.two_site_hamiltonians(
                t, dt, include_system=include_system),
            factor,
        )

    def trotter_mpo(self, t, dt):
        """Return the exact conditional-displacement MPO for one interval.

        This is the interaction part of the registered Strang/Trotter step; the
        system Hamiltonian half-steps are applied by the simulation planner.
        """
        eigenvalues, vectors = la.eigh(self.coupling)
        coefficients = self.interval_coefficients(t, dt)
        rank = len(eigenvalues)
        tensors = [np.zeros((1, rank, self.pd_sys, self.pd_sys), complex)]
        for branch in range(rank):
            vector = vectors[:, branch]
            tensors[0][0, branch] = np.outer(vector, vector.conj())

        for mode, coefficient in enumerate(coefficients):
            dimension = self.pd_boson[mode]
            destroy = annihilate(dimension)
            create_op = destroy.conj().T
            right_rank = rank if mode < len(coefficients) - 1 else 1
            tensor = np.zeros(
                (rank, right_rank, dimension, dimension), complex)
            for branch, eigenvalue in enumerate(eigenvalues):
                alpha = -1j * eigenvalue * np.conj(coefficient)
                target = branch if right_rank > 1 else 0
                tensor[branch, target] = la.expm(
                    alpha * create_op - np.conj(alpha) * destroy)
            tensors.append(tensor)
        return tensors
