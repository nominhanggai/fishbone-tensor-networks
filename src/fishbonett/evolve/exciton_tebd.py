"""Swap-network TEBD for conventional exciton-bath MPS layouts."""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from fishbonett.contract import _einsum_cached
from fishbonett.evolve._tdvp_kernels import right_canonicalize
from fishbonett.evolve.tebd import update_bond
from fishbonett.linalg import full_svd
from fishbonett.operators import displacement
from fishbonett.states.mps import SystemBathMPS

__all__ = ["InterleavedExcitonTEBD", "vidal_from_mps"]


def vidal_from_mps(dimensions, tensors):
    """Convert ``(left, right, physical)`` tensors to Vidal-form MPS storage."""
    dimensions = tuple(map(int, dimensions))
    current = right_canonicalize(tensors)
    if tuple(tensor.shape[2] for tensor in current) != dimensions:
        raise ValueError("MPS physical dimensions do not match the layout")
    state = SystemBathMPS(dimensions)
    left_singular = np.ones(1)
    for site in range(len(current) - 1):
        left, right, physical = current[site].shape
        matrix = np.transpose(current[site], (0, 2, 1)).reshape(
            left * physical, right
        )
        u, singular, vh = full_svd(matrix, full_matrices=False)
        cutoff = np.finfo(float).eps * max(matrix.shape) * singular[0]
        rank = max(1, int(np.count_nonzero(singular > cutoff)))
        u, singular, vh = u[:, :rank], singular[:rank], vh[:rank]
        singular /= np.linalg.norm(singular)
        tensor = u.reshape(left, physical, rank)
        tensor = tensor / left_singular[:, None, None]
        state.B[site] = tensor * singular[None, None, :]
        state.S[site] = left_singular
        state.S[site + 1] = singular
        current[site + 1] = _einsum_cached(
            "a,ax,xrp->arp", singular, vh, current[site + 1]
        )
        left_singular = singular
    state.B[-1] = (
        np.transpose(current[-1], (0, 2, 1))
        / left_singular[:, None, None]
    )
    state.S[-1] = left_singular
    return state


def _bare_swap(state, bond, bond_dim, trunc_eps):
    """Exchange adjacent physical sites and retain their original state."""
    left = state.B[bond].shape[1]
    right = state.B[bond + 1].shape[1]
    state.U[bond] = np.eye(left * right, dtype=complex).reshape(
        left, right, left, right
    )
    update_bond(state, bond, bond_dim, trunc_eps, swap=1)


def _apply_local(state, site, gate):
    """Apply a physical unitary without changing the Vidal gauge."""
    state.B[site] = _einsum_cached("ij,ajb->aib", gate, state.B[site])


def _apply_distant_pair(state, left, right, gate, bond_dim, trunc_eps):
    """Bring two MPS sites together, apply ``gate``, and restore the layout."""
    if not 0 <= left < right < state.n:
        raise ValueError("pair sites must satisfy 0 <= left < right < n_sites")
    for bond in range(right - 1, left, -1):
        _bare_swap(state, bond, bond_dim, trunc_eps)
    state.U[left] = gate
    update_bond(state, left, bond_dim, trunc_eps, swap=0)
    for bond in range(left + 1, right):
        _bare_swap(state, bond, bond_dim, trunc_eps)


def _conditional_displacement(coefficient, dimension):
    gate = np.zeros((2, dimension, 2, dimension), complex)
    gate[0, :, 0, :] = np.eye(dimension, dtype=complex)
    gate[1, :, 1, :] = displacement(
        -1j * np.conj(coefficient), dimension
    )
    return gate


class InterleavedExcitonTEBD:
    """Second-order gate propagation on an interleaved conventional MPS.

    Electronic sites are temporarily joined by bare MPS swaps for the hopping
    gates.  Each electronic site then traverses only its own contiguous bath
    branch while the exact interval-integrated conditional displacements are
    applied.  Every temporary permutation is reversed before the next stage.
    """

    def __init__(
        self, representation, state, dt, bond_dim, trunc_eps, *, time_offset=0.0
    ):
        if representation.layout != "interleaved":
            raise ValueError("InterleavedExcitonTEBD needs the interleaved layout")
        self.representation = representation
        self.state = state
        self.dt = float(dt)
        self.bond_dim = bond_dim
        self.trunc_eps = float(trunc_eps)
        self.time_offset = float(time_offset)
        self._electronic_operations = self._compile_electronic_half_step()

    def _compile_electronic_half_step(self):
        # The outer split gives H_e a duration dt/2.  A palindromic product over
        # its local terms therefore uses dt/4 gates in each direction.
        substep = self.dt / 4.0
        hamiltonian = self.representation.hamiltonian
        sites = self.representation.electronic_sites
        number = np.diag([0.0, 1.0]).astype(complex)
        plus = np.array([[0.0, 0.0], [1.0, 0.0]], complex)
        minus = plus.conj().T
        operations = []
        for level, site in enumerate(sites):
            local = hamiltonian[level, level] * number
            operations.append(("local", site, la.expm(-1j * substep * local)))
        for left in range(len(sites)):
            for right in range(left + 1, len(sites)):
                hopping = (
                    hamiltonian[left, right] * np.kron(plus, minus)
                    + hamiltonian[right, left] * np.kron(minus, plus)
                )
                if np.allclose(hopping, 0.0):
                    continue
                gate = la.expm(-1j * substep * hopping).reshape(2, 2, 2, 2)
                operations.append(("pair", sites[left], sites[right], gate))
        return tuple(operations)

    def _electronic_half_step(self):
        operations = self._electronic_operations
        for operation in (*operations, *reversed(operations)):
            if operation[0] == "local":
                _, site, gate = operation
                _apply_local(self.state, site, gate)
            else:
                _, left, right, gate = operation
                _apply_distant_pair(
                    self.state, left, right, gate,
                    self.bond_dim, self.trunc_eps,
                )

    def _coupling_step(self, time):
        branches = dict(self.representation.branches)
        for level, (electronic, modes) in self.representation.branch_sites.items():
            representation = branches[level]
            coefficients = representation.interval_coefficients(time, self.dt)
            if len(modes) != len(coefficients):
                raise ValueError("bath branch sites and coefficients differ")
            for offset, (mode, coefficient) in enumerate(zip(modes, coefficients)):
                bond = electronic + offset
                if mode != bond + 1:
                    raise ValueError("interleaved bath branch is not contiguous")
                self.state.U[bond] = _conditional_displacement(
                    coefficient, self.state.B[bond + 1].shape[1]
                )
                update_bond(
                    self.state, bond, self.bond_dim, self.trunc_eps,
                    swap=int(offset + 1 < len(modes)),
                )
            for bond in range(electronic + len(modes) - 2, electronic - 1, -1):
                _bare_swap(
                    self.state, bond, self.bond_dim, self.trunc_eps
                )

    def step(self, index):
        """Advance one complete Strang step while restoring site ordering."""
        self._electronic_half_step()
        self._coupling_step(self.time_offset + index * self.dt)
        self._electronic_half_step()
