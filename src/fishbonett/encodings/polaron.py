"""Numerical encodings of polaron representations."""

import numpy as np

from fishbonett.encodings.mpo import MPOEncoding, product_sum_mpo
from fishbonett.linalg import expm_gate
from fishbonett.operators import annihilate

__all__ = [
    "PolaronGateEncoder", "encode_polaron_mpo",
    "encode_polaron_chain_gates",
]


class PolaronGateEncoder:
    """Expose local gates for a ``polaron-chain`` representation."""

    def __init__(self, representation):
        self.representation = representation

    def gates(self, dt):
        return encode_polaron_chain_gates(self.representation, dt)


def _identity_products(dimensions):
    return [np.eye(dimension, dtype=complex) for dimension in dimensions]


def encode_polaron_mpo(representation, initial_state):
    """Encode ``polaron-star`` or ``polaron-chain`` as a static MPO."""
    dimensions = [representation.pd_sys] + representation.pd_boson
    products = []
    coefficients = []

    row = _identity_products(dimensions)
    coupling = np.asarray(representation.coupling, complex)
    row[0] = coupling @ coupling
    products.append(row)
    coefficients.append(-representation.reorganization_energy)

    transformed_system = representation.system_in_coupling_eigenvectors
    vectors = representation.eigenvectors
    values = representation.eigenvalues
    for left in range(representation.pd_sys):
        for right in range(representation.pd_sys):
            coefficient = transformed_system[left, right]
            if abs(coefficient) < 1e-14:
                continue
            row = _identity_products(dimensions)
            row[0] = np.outer(
                vectors[:, left], vectors[:, right].conj())
            scale = values[left] - values[right]
            for mode in range(representation.len_boson):
                row[mode + 1] = representation.displacement_operator(
                    mode, scale)
            products.append(row)
            coefficients.append(coefficient)

    for mode, frequency in enumerate(representation.frequencies):
        destroy = annihilate(representation.pd_boson[mode])
        row = _identity_products(dimensions)
        row[mode + 1] = destroy.conj().T @ destroy
        products.append(row)
        coefficients.append(frequency)

    for mode, hopping in enumerate(representation.hoppings):
        left_destroy = annihilate(representation.pd_boson[mode])
        right_destroy = annihilate(representation.pd_boson[mode + 1])
        for left_operator, right_operator in (
            (left_destroy.conj().T, right_destroy),
            (left_destroy, right_destroy.conj().T),
        ):
            row = _identity_products(dimensions)
            row[mode + 1] = left_operator
            row[mode + 2] = right_operator
            products.append(row)
            coefficients.append(hopping)

    mpo = product_sum_mpo(dimensions, products, coefficients)
    state = np.asarray(initial_state, complex)
    state = state / np.linalg.norm(state)
    return MPOEncoding(
        n_sites=len(dimensions),
        phys_dim=representation.pd_boson[0],
        system=(representation.h_sys, representation.coupling, state),
        mpo=lambda _time=None: mpo,
        static=True,
    )


def encode_polaron_chain_gates(representation, dt):
    """Encode ``polaron-chain`` as nearest-neighbour two-site gates."""
    if representation.name != "polaron-chain":
        raise ValueError("local Trotter gates require polaron-chain")
    dimension = representation.pd_boson[0]
    if any(value != dimension for value in representation.pd_boson):
        raise ValueError("polaron-chain gates require a uniform mode dimension")

    destroy = annihilate(dimension)
    create = destroy.conj().T
    number = create @ destroy
    system_mode = np.zeros(
        (representation.pd_sys * dimension,) * 2, complex)
    transformed_system = representation.system_in_coupling_eigenvectors
    for left in range(representation.pd_sys):
        for right in range(representation.pd_sys):
            coefficient = transformed_system[left, right]
            if abs(coefficient) < 1e-14:
                continue
            projector = np.outer(
                representation.eigenvectors[:, left],
                representation.eigenvectors[:, right].conj())
            displacement = representation.displacement_operator(
                0,
                representation.eigenvalues[left]
                - representation.eigenvalues[right],
            )
            system_mode += coefficient * np.kron(projector, displacement)
    coupling = np.asarray(representation.coupling, complex)
    system_mode += representation.frequencies[0] * np.kron(
        np.eye(representation.pd_sys), number)
    system_mode -= representation.reorganization_energy * np.kron(
        coupling @ coupling, np.eye(dimension))

    gates = [expm_gate(system_mode, dt).reshape(
        representation.pd_sys,
        dimension,
        representation.pd_sys,
        dimension,
    )]
    for mode, hopping in enumerate(representation.hoppings, start=1):
        left_dimension = representation.pd_boson[mode - 1]
        right_dimension = representation.pd_boson[mode]
        left_destroy = annihilate(left_dimension)
        right_destroy = annihilate(right_dimension)
        right_number = right_destroy.conj().T @ right_destroy
        local = (
            hopping * (
                np.kron(left_destroy.conj().T, right_destroy)
                + np.kron(left_destroy, right_destroy.conj().T)
            )
            + representation.frequencies[mode]
            * np.kron(np.eye(left_dimension), right_number)
        )
        gates.append(expm_gate(local, dt).reshape(
            left_dimension,
            right_dimension,
            left_dimension,
            right_dimension,
        ))
    return gates
