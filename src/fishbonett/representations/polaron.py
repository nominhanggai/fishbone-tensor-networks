"""The ``polaron-star`` and ``polaron-chain`` representations.

The Lang–Firsov transform displaces every star mode by ``g_k / omega_k``.
Keeping those modes gives ``polaron-star``.  Applying the star-to-chain transform
associated with the reweighted measure ``J(omega) / omega**2`` concentrates the
displacement on the first chain mode and gives ``polaron-chain``.

This module owns the transformed Hamiltonian, its TDVP MPO and TEBD gates, the
transformed initial state, and recovery of laboratory observables.
"""

import numpy as np
import scipy.linalg as la

from fishbonett.bath._coefficients import require_resolved, star_coefficients
from fishbonett.bath.lanczos import lanczos
from fishbonett.linalg import expm_gate
from fishbonett.operators import annihilate
from fishbonett.representations._mpo import identity_product, product_sum_mpo
from fishbonett.system import check_operator

__all__ = ["PolaronRepresentation"]


def _coherent(dimension, displacement):
    destroy = annihilate(dimension)
    generator = destroy.conj().T - destroy
    vacuum = np.eye(dimension, dtype=complex)[:, 0]
    return la.expm(displacement * generator) @ vacuum


def _validate_infrared_integrability(density, domain):
    """Reject a continuum whose Lang--Firsov displacement norm diverges.

    The norm is proportional to ``integral J(w)/w**2 dw``.  When the domain
    touches zero, estimate the local power law on each represented side and
    require ``J(w) = O(|w|**s)`` with ``s > 1``.  A gapped domain needs no
    extrapolation.
    """
    left, right = map(float, domain)
    if not left <= 0.0 <= right:
        return
    sides = [sign for sign, extent in ((1.0, right), (-1.0, -left)) if extent > 0]
    for sign in sides:
        extent = right if sign > 0 else -left
        frequency = extent * np.power(10.0, -np.arange(3.0, 9.0))
        values = np.array([float(density(sign * value)) for value in frequency])
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("spectral density must be finite and non-negative")
        active = values > np.finfo(float).tiny
        if np.count_nonzero(active) < 3:
            continue
        slope = np.polyfit(
            np.log(frequency[active]), np.log(values[active]), 1
        )[0]
        if slope <= 1.0 + 1e-3:
            raise ValueError(
                "the polaron representation requires finite integral "
                "J(w)/w**2 near zero; use a gapped or super-Ohmic density"
            )


class PolaronRepresentation:
    """A star or chain Lang–Firsov representation."""

    names = frozenset({"polaron-star", "polaron-chain"})

    def __init__(self, *, representation, h_sys, coupling, bath):
        if representation not in self.names:
            raise ValueError(
                "representation must be 'polaron-star' or 'polaron-chain'")
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
        self.hoppings = None
        self.displacements = None
        self.reorganization_energy = None

    @property
    def static(self):
        return True

    @property
    def dimensions(self):
        return (self.pd_sys, *self.pd_boson)

    @property
    def n_sites(self):
        return len(self.dimensions)

    def build(self):
        """Prepare finite coefficients and diagonalize the coupling operator."""
        densities = self.bath.spectral_densities()
        if len(densities) != 1:
            raise ValueError("the polaron representation requires one channel")
        density = densities[0]
        if self.bath.continuum_present:
            _validate_infrared_integrability(density, self.bath.domain)

        # Build the polaron from the *same finite star Hamiltonian* propagated by
        # the other representations.  This includes explicit vibronic lines and
        # makes the counterterm exactly consistent with the discretized model.
        star = star_coefficients(self.bath)
        star_frequencies = np.asarray(star.frequencies, float)
        star_couplings = np.asarray(star.couplings[0], float)
        active = np.abs(star_couplings) > 100 * np.finfo(float).eps
        if np.any(np.abs(star_frequencies[active]) <= 100 * np.finfo(float).eps):
            raise ValueError(
                "the finite bath contains a coupled zero-frequency mode, so the "
                "polaron displacement diverges"
            )
        star_displacements = np.divide(
            star_couplings, star_frequencies,
            out=np.zeros_like(star_couplings), where=star_frequencies != 0,
        )
        reorganization = float(np.sum(
            np.divide(
                star_couplings**2, star_frequencies,
                out=np.zeros_like(star_couplings), where=star_frequencies != 0,
            )
        ))

        if len(star_frequencies) == 1:
            chain_matrix = np.diag(star_frequencies)
        else:
            chain_matrix, _ = lanczos(
                np.diag(star_frequencies), star_displacements
            )
        chain_frequencies = np.diagonal(chain_matrix)
        chain_hoppings = np.diagonal(chain_matrix, -1)
        chain_displacement = np.linalg.norm(star_displacements)

        if self.name == "polaron-star":
            self.frequencies = np.asarray(star_frequencies, float)
            self.hoppings = np.empty(0, float)
            self.displacements = np.asarray(star_displacements, float)
        else:
            self.frequencies = np.asarray(chain_frequencies, float)
            self.hoppings = np.asarray(chain_hoppings, float)
            self.displacements = np.zeros(self.len_boson, float)
            self.displacements[0] = float(chain_displacement)
        self.reorganization_energy = float(reorganization)
        self.eigenvalues, self.eigenvectors = la.eigh(self.coupling)
        self.system_in_coupling_eigenvectors = (
            self.eigenvectors.conj().T
            @ np.asarray(self.h_sys, complex)
            @ self.eigenvectors
        )
        return self

    def tdvp_mpo(self, _time=None):
        """Return the static polaron Hamiltonian MPO consumed by TDVP."""
        if self.reorganization_energy is None:
            raise ValueError("build the polaron representation first")
        dimensions = list(self.dimensions)
        products, coefficients = [], []

        row = identity_product(dimensions)
        row[0] = self.coupling @ self.coupling
        products.append(row)
        coefficients.append(-self.reorganization_energy)

        transformed_system = self.system_in_coupling_eigenvectors
        for left in range(self.pd_sys):
            for right in range(self.pd_sys):
                coefficient = transformed_system[left, right]
                if abs(coefficient) < 1e-14:
                    continue
                row = identity_product(dimensions)
                row[0] = np.outer(
                    self.eigenvectors[:, left],
                    self.eigenvectors[:, right].conj())
                scale = self.eigenvalues[left] - self.eigenvalues[right]
                for mode in range(self.len_boson):
                    row[mode + 1] = self.displacement_operator(mode, scale)
                products.append(row)
                coefficients.append(coefficient)

        for mode, frequency in enumerate(self.frequencies):
            destroy = annihilate(self.pd_boson[mode])
            row = identity_product(dimensions)
            row[mode + 1] = destroy.conj().T @ destroy
            products.append(row)
            coefficients.append(frequency)

        for mode, hopping in enumerate(self.hoppings):
            left_destroy = annihilate(self.pd_boson[mode])
            right_destroy = annihilate(self.pd_boson[mode + 1])
            for left_operator, right_operator in (
                (left_destroy.conj().T, right_destroy),
                (left_destroy, right_destroy.conj().T),
            ):
                row = identity_product(dimensions)
                row[mode + 1] = left_operator
                row[mode + 2] = right_operator
                products.append(row)
                coefficients.append(hopping)
        return product_sum_mpo(dimensions, products, coefficients)

    def tebd_gates(self, dt):
        """Return nearest-neighbour gates for ``polaron-chain`` TEBD."""
        if self.name != "polaron-chain":
            raise ValueError("local TEBD gates require polaron-chain")
        dimension = self.pd_boson[0]
        if any(value != dimension for value in self.pd_boson):
            raise ValueError(
                "polaron-chain gates require a uniform mode dimension")

        destroy = annihilate(dimension)
        create_op = destroy.conj().T
        number_op = create_op @ destroy
        system_mode = np.zeros(
            (self.pd_sys * dimension,) * 2, complex)
        for left in range(self.pd_sys):
            for right in range(self.pd_sys):
                coefficient = self.system_in_coupling_eigenvectors[left, right]
                if abs(coefficient) < 1e-14:
                    continue
                projector = np.outer(
                    self.eigenvectors[:, left],
                    self.eigenvectors[:, right].conj())
                displacement = self.displacement_operator(
                    0, self.eigenvalues[left] - self.eigenvalues[right])
                system_mode += coefficient * np.kron(
                    projector, displacement)
        system_mode += self.frequencies[0] * np.kron(
            np.eye(self.pd_sys), number_op)
        system_mode -= self.reorganization_energy * np.kron(
            self.coupling @ self.coupling, np.eye(dimension))

        gates = [expm_gate(system_mode, dt).reshape(
            self.pd_sys, dimension, self.pd_sys, dimension)]
        for mode, hopping in enumerate(self.hoppings, start=1):
            left_dimension = self.pd_boson[mode - 1]
            right_dimension = self.pd_boson[mode]
            left_destroy = annihilate(left_dimension)
            right_destroy = annihilate(right_dimension)
            right_number = right_destroy.conj().T @ right_destroy
            local = (
                hopping * (
                    np.kron(left_destroy.conj().T, right_destroy)
                    + np.kron(left_destroy, right_destroy.conj().T)
                )
                + self.frequencies[mode]
                * np.kron(np.eye(left_dimension), right_number)
            )
            gates.append(expm_gate(local, dt).reshape(
                left_dimension, right_dimension,
                left_dimension, right_dimension))
        return gates

    def displacement_operator(self, mode, scale):
        """Local displacement for one transformed bath mode."""
        destroy = annihilate(self.pd_boson[mode])
        return la.expm(
            scale * self.displacements[mode]
            * (destroy.conj().T - destroy))

    def initial_mps(self, psi_sys):
        """Transformed initial state in ``(left, right, physical)`` convention."""
        amplitudes = self.eigenvectors.conj().T @ np.asarray(psi_sys, complex)
        rank = self.pd_sys
        tensors = [np.zeros((1, rank, self.pd_sys), complex)]
        for branch in range(rank):
            tensors[0][0, branch] = (
                amplitudes[branch] * self.eigenvectors[:, branch])
        for mode, dimension in enumerate(self.pd_boson):
            right_rank = rank if mode < self.len_boson - 1 else 1
            tensor = np.zeros((rank, right_rank, dimension), complex)
            for branch, eigenvalue in enumerate(self.eigenvalues):
                target = branch if right_rank > 1 else 0
                tensor[branch, target] = _coherent(
                    dimension, eigenvalue * self.displacements[mode])
            tensors.append(tensor)
        return tensors

    def initial_theta(self, psi_sys):
        """Two-site transformed state used by local ``polaron-chain`` gates."""
        if self.name != "polaron-chain":
            raise ValueError("initial_theta is local only for polaron-chain")
        dimension = self.pd_boson[0]
        amplitudes = self.eigenvectors.conj().T @ np.asarray(psi_sys, complex)
        theta = np.zeros((self.pd_sys, dimension), complex)
        for branch, amplitude in enumerate(amplitudes):
            theta += amplitude * np.outer(
                self.eigenvectors[:, branch],
                _coherent(
                    dimension,
                    self.eigenvalues[branch] * self.displacements[0]))
        return theta.reshape(1, self.pd_sys, dimension, 1)

    def recover_rdm(self, tensors):
        """Recover the laboratory system RDM from a transformed MPS."""
        system_tensor = np.asarray(tensors[0], complex)
        if system_tensor.shape[0] != 1:
            raise ValueError("the system tensor must be the left boundary")
        projected = [
            np.einsum(
                "rs,s->r", system_tensor[0],
                self.eigenvectors[:, branch].conj())
            for branch in range(self.pd_sys)
        ]
        transformed = np.zeros((self.pd_sys, self.pd_sys), complex)
        for left in range(self.pd_sys):
            for right in range(self.pd_sys):
                environment = np.outer(
                    projected[left], projected[right].conj())
                scale = self.eigenvalues[right] - self.eigenvalues[left]
                for mode, tensor in enumerate(tensors[1:]):
                    displacement = self.displacement_operator(mode, scale)
                    environment = np.einsum(
                        "ac,abp,cdq,qp->bd",
                        environment, tensor, tensor.conj(), displacement,
                        optimize=True)
                transformed[left, right] = environment.reshape(-1)[0]
        lab = self.eigenvectors @ transformed @ self.eigenvectors.conj().T
        return lab / np.trace(lab)

    def recover_pair_rdm(self, theta):
        """Fast laboratory RDM recovery for the local chain representation."""
        if self.name != "polaron-chain":
            raise ValueError("pair recovery is local only for polaron-chain")
        rho = np.einsum(
            "LXaR,LYbR->XaYb", theta, np.asarray(theta).conj())
        transformed = np.einsum(
            "Xi,XaYb,Yj->iajb",
            self.eigenvectors.conj(), rho, self.eigenvectors)
        out = np.zeros((self.pd_sys, self.pd_sys), complex)
        for left in range(self.pd_sys):
            for right in range(self.pd_sys):
                displacement = self.displacement_operator(
                    0, self.eigenvalues[right] - self.eigenvalues[left])
                out[left, right] = np.einsum(
                    "ab,ba->", transformed[left, :, right, :], displacement)
        lab = self.eigenvectors @ out @ self.eigenvectors.conj().T
        return lab / np.trace(lab)
