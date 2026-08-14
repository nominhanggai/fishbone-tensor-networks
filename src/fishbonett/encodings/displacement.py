"""Conditional-displacement encoding of interaction propagators."""

import numpy as np
import scipy.linalg as la

from fishbonett.operators import annihilate

__all__ = ["ConditionalDisplacementEncoder"]


class ConditionalDisplacementEncoder:
    """Encode an interaction representation's finite-step propagator as an MPO."""

    def __init__(self, representation):
        self.representation = representation

    def displacement_mpo(self, t, delta):
        """Return the exact commuting system–bath propagator over one interval."""
        source = self.representation
        eigenvalues, vectors = la.eigh(np.asarray(source.coupling, complex))
        coefficients = source.interval_coefficients(t, delta)
        rank = len(eigenvalues)

        tensors = [np.zeros((1, rank, source.pd_sys, source.pd_sys), complex)]
        for branch in range(rank):
            vector = vectors[:, branch]
            tensors[0][0, branch] = np.outer(vector, vector.conj())

        for index, coefficient in enumerate(coefficients):
            dimension = source.pd_boson[index]
            destroy = annihilate(dimension)
            create = destroy.conj().T
            right_rank = rank if index < len(coefficients) - 1 else 1
            tensor = np.zeros(
                (rank, right_rank, dimension, dimension), complex)
            for branch, eigenvalue in enumerate(eigenvalues):
                alpha = -1j * eigenvalue * np.conj(coefficient)
                target = branch if right_rank > 1 else 0
                tensor[branch, target] = la.expm(
                    alpha * create - np.conj(alpha) * destroy)
            tensors.append(tensor)
        return tensors
