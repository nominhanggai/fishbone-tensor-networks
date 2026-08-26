r"""Multi-set matrix-product states.

A multi-set state expands a system--environment wavefunction in an orthonormal
system basis,

.. math::

   |\Psi\rangle = \sum_a |a\rangle |\psi_a\rangle,

and represents every environmental wavefunction ``psi_a`` by an independent
MPS.  The outer system label is exact; truncation acts only inside the
environmental tensor networks.  This is the ansatz introduced for Holstein
dynamics by Kloss, Reichman, and Tempelaar, Phys. Rev. Lett. 123, 126601
(2019).

Tensors use the TDVP convention ``(left bond, right bond, physical)``.  This
container owns no Hamiltonian or propagation logic; coupled
projector-splitting sweeps live in :mod:`fishbonett.evolve.multiset`.
"""

from __future__ import annotations

import numpy as np

from fishbonett.contract import _einsum_cached
__all__ = ["MultiSetMPS"]


def _copy_mps(tensors):
    return [np.array(tensor, dtype=complex, copy=True) for tensor in tensors]


def _overlap(bra, ket, *, operator=None, site=None):
    """Contract ``<bra|ket>`` or one local matrix element."""
    environment = np.ones((1, 1), complex)
    for index, (bra_tensor, ket_tensor) in enumerate(
        zip(bra, ket, strict=True)
    ):
        if index == site:
            environment = _einsum_cached(
                "ab,arq,bsp,qp->rs",
                environment,
                bra_tensor.conj(),
                ket_tensor,
                operator,
            )
        else:
            environment = _einsum_cached(
                "ab,arp,bsp->rs",
                environment,
                bra_tensor.conj(),
                ket_tensor,
            )
    return environment.reshape(-1)[0]


class MultiSetMPS:
    """One environmental MPS for each state of a finite-dimensional system.

    Parameters
    ----------
    sets
        ``sets[a][i]`` is tensor ``i`` of environmental wavefunction ``a``.
        Different sets may have different bond dimensions, but must have the
        same physical dimensions and number of sites.
    """

    def __init__(self, sets):
        """Store and validate one bath MPS per exact system-basis component."""
        self.sets = [_copy_mps(tensors) for tensors in sets]
        self._validate()

    def _validate(self):
        if not self.sets:
            raise ValueError("a multi-set state needs at least one system set")
        if not self.sets[0]:
            raise ValueError("a multi-set state needs at least one environment site")
        count = len(self.sets[0])
        dimensions = tuple(tensor.shape[2] for tensor in self.sets[0])
        for set_index, tensors in enumerate(self.sets):
            if len(tensors) != count:
                raise ValueError("all multi-set MPSs must have the same length")
            for site, tensor in enumerate(tensors):
                if tensor.ndim != 3:
                    raise ValueError(f"sets[{set_index}][{site}] must have three axes")
                if tensor.shape[2] != dimensions[site]:
                    raise ValueError("all multi-set MPSs must have the same physical dimensions")
                if site == 0 and tensor.shape[0] != 1:
                    raise ValueError("every set must have a left boundary of one")
                if site == count - 1 and tensor.shape[1] != 1:
                    raise ValueError("every set must have a right boundary of one")
                if site and tensors[site - 1].shape[1] != tensor.shape[0]:
                    raise ValueError(f"sets[{set_index}] has incompatible bond {site}")
                if not np.all(np.isfinite(tensor)):
                    raise ValueError(f"sets[{set_index}][{site}] contains non-finite values")
        self.dimensions = dimensions

    @classmethod
    def product(cls, amplitudes, dimensions):
        """Create ``sum_a amplitudes[a] |a> |0 ... 0>``."""
        amplitudes = np.asarray(amplitudes, complex).reshape(-1)
        if amplitudes.size == 0:
            raise ValueError("amplitudes must be non-empty")
        norm = np.linalg.norm(amplitudes)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("amplitudes must have a finite nonzero norm")
        amplitudes = amplitudes / norm
        dimensions = tuple(int(value) for value in dimensions)
        if not dimensions or any(value < 1 for value in dimensions):
            raise ValueError("dimensions must contain positive integers")
        sets = []
        for amplitude in amplitudes:
            tensors = []
            for site, dimension in enumerate(dimensions):
                tensor = np.zeros((1, 1, dimension), complex)
                tensor[0, 0, 0] = amplitude if site == 0 else 1.0
                tensors.append(tensor)
            sets.append(tensors)
        return cls(sets)

    @classmethod
    def from_full_mps(cls, tensors):
        """Split a boundary system site off a full system--bath MPS.

        ``tensors[0]`` carries the system physical leg.  Slicing that leg and
        absorbing its right-bond vector into the first environment tensor gives
        one exact environmental MPS per system basis state.
        """
        tensors = _copy_mps(tensors)
        if len(tensors) < 2:
            raise ValueError("a full MPS needs a system and at least one environment site")
        system = tensors[0]
        if system.ndim != 3 or system.shape[0] != 1:
            raise ValueError("the system tensor must be a three-axis left boundary")
        sets = []
        for state in range(system.shape[2]):
            first = _einsum_cached("r,rsp->sp", system[0, :, state], tensors[1])[
                None, :, :
            ]
            sets.append([first] + _copy_mps(tensors[2:]))
        return cls(sets)

    @property
    def n_sets(self):
        """Number of exact system-basis components."""
        return len(self.sets)

    @property
    def n_sites(self):
        """Number of environment sites in each component."""
        return len(self.sets[0])

    def copy(self):
        """Independent copy of every component tensor."""
        return MultiSetMPS(self.sets)

    def system_rdm(self):
        """Normalized reduced density matrix of the outer system index."""
        rho = np.empty((self.n_sets, self.n_sets), complex)
        for left in range(self.n_sets):
            for right in range(self.n_sets):
                rho[left, right] = _overlap(self.sets[right], self.sets[left])
        trace = np.trace(rho)
        if abs(trace) == 0 or not np.isfinite(trace):
            raise ValueError("cannot measure a zero or non-finite multi-set state")
        return rho / trace

    def site_expectation(self, operator, site):
        """Normalized expectation of an environment-site operator."""
        if site < 0 or site >= self.n_sites:
            raise IndexError("environment site is outside the multi-set MPS")
        operator = np.asarray(operator, complex)
        dimension = self.dimensions[site]
        if operator.shape != (dimension, dimension):
            raise ValueError(
                f"operator has shape {operator.shape}, expected {(dimension, dimension)}"
            )
        numerator = sum(
            _overlap(tensors, tensors, operator=operator, site=site) for tensors in self.sets
        )
        denominator = sum(_overlap(tensors, tensors) for tensors in self.sets)
        return numerator / denominator

    def peak_bond(self):
        """Largest bond in any component MPS."""
        return max(
            (tensor.shape[1] for tensors in self.sets for tensor in tensors),
            default=1,
        )

    def bond_dimensions(self):
        """Per-set MPS bond dimensions, including boundary bonds."""
        return tuple(
            (tensors[0].shape[0],) + tuple(tensor.shape[1] for tensor in tensors)
            for tensors in self.sets
        )

    def combined_mps(self):
        """Embed the multi-set state exactly in a conventional full MPS.

        This conversion is intended for observables and cross-checks.  It forms
        block-diagonal virtual bonds and therefore discards the computational
        advantage of keeping the sets separate during propagation.
        """
        count = self.n_sets
        system = np.zeros((1, count, count), complex)
        for state in range(count):
            system[0, state, state] = 1.0
        out = [system]
        for site in range(self.n_sites):
            left_sizes = [tensors[site].shape[0] for tensors in self.sets]
            right_sizes = [tensors[site].shape[1] for tensors in self.sets]
            physical = self.dimensions[site]
            if site == 0:
                left_total = count
                left_offsets = list(range(count))
            else:
                left_total = sum(left_sizes)
                left_offsets = np.cumsum([0] + left_sizes[:-1]).tolist()
            right_total = 1 if site == self.n_sites - 1 else sum(right_sizes)
            right_offsets = (
                [0] * count if right_total == 1 else np.cumsum([0] + right_sizes[:-1]).tolist()
            )
            tensor = np.zeros((left_total, right_total, physical), complex)
            for state, tensors in enumerate(self.sets):
                value = tensors[site]
                left = left_offsets[state]
                right = right_offsets[state]
                if site == 0:
                    tensor[left, right : right + right_sizes[state], :] = value[0]
                else:
                    tensor[
                        left : left + left_sizes[state],
                        right : right + right_sizes[state],
                        :,
                    ] = value
            out.append(tensor)
        return out
