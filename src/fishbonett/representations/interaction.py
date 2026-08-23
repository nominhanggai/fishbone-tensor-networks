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

from fishbonett.bath._coefficients import require_resolved, star_coefficients
from fishbonett.bath.conventions import integrated_free_phase
from fishbonett.linalg import expm_gate, kron
from fishbonett.operators import annihilate, create, displacement
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

    def tdvp_mpo(self, t=None, reverse=True):
        """Return the instantaneous Hamiltonian MPO consumed by TDVP.

        ``reverse`` chooses which end of the chain carries the *first*
        coefficient.  The 1D drivers store the bath sites in reverse coefficient
        order, so that is the default.  A comb stores each branch forward, in the
        same order as :meth:`interval_coefficients` and :meth:`tebd_gates`, so the
        tree engine passes ``reverse=False``.  Getting it backwards attaches every
        coupling to the wrong mode: the Hamiltonian stays Hermitian and the run
        stays stable, so the error shows up only as wrong dynamics.
        """
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
        for mode, amplitude in enumerate(
                coefficients[::-1] if reverse else coefficients):
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

    def coupling_spectrum(self):
        """Cached ``eigh`` of the coupling operator.

        The coupling does not change during a run, but :meth:`trotter_mpo` is
        called every step, so this is computed once.
        """
        if getattr(self, "_coupling_spectrum", None) is None:
            operator = np.asarray(self.coupling, complex)
            if not np.allclose(operator, operator.conj().T, atol=1e-12):
                raise ValueError(
                    "trotter-mpo needs a Hermitian coupling operator: the "
                    "propagator is built from its eigenbasis, so a non-Hermitian "
                    "one would give a non-unitary step")
            self._coupling_spectrum = np.linalg.eigh(operator)
        return self._coupling_spectrum

    def trotter_mpo(self, t, dt):
        """Return the exact conditional-displacement MPO for one interval.

        This is the interaction part of the registered Strang/Trotter step; the
        system Hamiltonian half-steps are applied by the simulation planner.

        All the coupling terms share one system operator ``O`` and act on distinct
        modes, so they commute and the interval propagator factorizes exactly: in
        each eigenbranch ``lambda`` of ``O`` every mode is displaced by
        ``alpha = -i lambda conj(c_k)``.  That makes the MPO bond exactly
        ``len(eigenvalues)`` and diagonal in the branch index.

        The displacements come from :func:`~fishbonett.operators.displacement`, in
        closed form, rather than one ``expm`` per (mode, branch).

        Notes
        -----
        A single displacement is the exact interval propagator only on an
        untruncated ladder.  On ``phys_dim`` levels the residual is a
        ``lambda**2``-weighted phase of order ``dt**3`` per step, plus a deviation
        confined to the top Fock level.  Because the weight is ``lambda**2``, the
        phase is common to all branches -- and so unobservable -- whenever the
        coupling's eigenvalues share a magnitude (``sigma_z``, for instance); it is
        a relative phase for couplings that do not (a projector). Either way it
        accumulates as ``O(dt**2)``, matching the order of the surrounding Strang
        splitting.
        """
        eigenvalues, vectors = self.coupling_spectrum()
        coefficients = self.interval_coefficients(t, dt)
        rank = len(eigenvalues)
        tensors = [np.zeros((1, rank, self.pd_sys, self.pd_sys), complex)]
        for branch in range(rank):
            vector = vectors[:, branch]
            tensors[0][0, branch] = np.outer(vector, vector.conj())

        # (mode, branch) displacement amplitudes in one array
        alphas = -1j * np.outer(np.conj(coefficients), eigenvalues)
        dimensions = list(self.pd_boson[:len(coefficients)])
        uniform = len(set(dimensions)) == 1
        matrices = (displacement(alphas, dimensions[0]) if uniform else None)

        for mode in range(len(coefficients)):
            dimension = dimensions[mode]
            right_rank = rank if mode < len(coefficients) - 1 else 1
            tensor = np.zeros(
                (rank, right_rank, dimension, dimension), complex)
            block = (matrices[mode] if uniform
                     else displacement(alphas[mode], dimension))
            for branch in range(rank):
                tensor[branch, branch if right_rank > 1 else 0] = block[branch]
            tensors.append(tensor)
        return tensors
