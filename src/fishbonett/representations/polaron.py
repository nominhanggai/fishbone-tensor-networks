"""The ``polaron-star`` and ``polaron-chain`` representations.

The Lang–Firsov transform displaces every star mode by ``g_k / omega_k``.
Keeping those modes gives ``polaron-star``.  Applying the star-to-chain transform
associated with the reweighted measure ``J(omega) / omega**2`` concentrates the
displacement on the first chain mode and gives ``polaron-chain``.

This module owns the transformed Hamiltonian data, transformed initial state, and
recovery of laboratory observables.  MPO and gate construction live in
:mod:`fishbonett.encodings.polaron`.
"""

import numpy as np
import scipy.linalg as la

from fishbonett.bath.chain import star_transform
from fishbonett.bath.conventions import reorganization_energy
from fishbonett.operators import annihilate
from fishbonett.system import check_operator

__all__ = ["PolaronRepresentation"]


def _coherent(dimension, displacement):
    destroy = annihilate(dimension)
    generator = destroy.conj().T - destroy
    vacuum = np.eye(dimension, dtype=complex)[:, 0]
    return la.expm(displacement * generator) @ vacuum


class PolaronRepresentation:
    """A complete star or chain Lang–Firsov representation."""

    names = frozenset({"polaron-star", "polaron-chain"})

    def __init__(self, pd, *, representation, h_sys, coupling, sd=None,
                 domain=None, discretizer=None, compiled_polaron=None):
        if representation not in self.names:
            raise ValueError(
                "representation must be 'polaron-star' or 'polaron-chain'")
        self.name = representation
        self.pd_sys = int(pd[0])
        self.pd_boson = [int(value) for value in pd[1:]]
        self.len_boson = len(self.pd_boson)
        if not self.pd_boson:
            raise ValueError("pd must include at least one bath mode")
        self.h_sys = check_operator(h_sys, "h_sys", self.pd_sys)
        self.coupling = check_operator(coupling, "coupling", self.pd_sys)
        if compiled_polaron is None and (sd is None or domain is None):
            raise ValueError(
                "provide compiled_polaron or both sd and domain")
        self.sd = sd
        self.domain = None if domain is None else tuple(domain)
        self.discretizer = discretizer
        self.compiled_polaron = compiled_polaron
        self.frequencies = None
        self.hoppings = None
        self.displacements = None
        self.reorganization_energy = None

    @property
    def static(self):
        return True

    def build(self):
        """Prepare finite coefficients and diagonalize the coupling operator."""
        if self.compiled_polaron is None:
            def displaced_density(frequency):
                if abs(frequency) < 1e-15:
                    return 0.0
                return self.sd(frequency) / frequency ** 2

            frequencies, displacements, transform = star_transform(
                displaced_density, self.len_boson, self.domain,
                self.discretizer)
            displacements = np.asarray(displacements, float)
            chain_matrix = transform @ np.diag(frequencies) @ transform.T
            reorganization = reorganization_energy(self.sd, self.domain)
            star_frequencies = np.asarray(frequencies, float)
            star_displacements = displacements
            chain_frequencies = np.diagonal(chain_matrix)
            chain_hoppings = np.diagonal(chain_matrix, -1)
            chain_displacement = np.linalg.norm(displacements)
            phys_dim = self.pd_boson[0]
        else:
            compiled = self.compiled_polaron
            star = compiled.star
            chain = compiled.chain
            star_frequencies = star.frequencies
            star_displacements = star.couplings[0]
            chain_frequencies = chain.frequencies
            chain_hoppings = chain.hoppings
            chain_displacement = chain.system_coupling
            reorganization = compiled.reorganization_energy
            phys_dim = star.phys_dim

        if len(star_frequencies) != self.len_boson:
            raise ValueError(
                f"compiled polaron data has {len(star_frequencies)} modes but "
                f"pd describes {self.len_boson}")
        if any(size != phys_dim for size in self.pd_boson):
            raise ValueError(
                "compiled polaron phys_dim does not match the mode dimensions")

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
        return lab / np.trace(lab).real

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
        return lab / np.trace(lab).real
