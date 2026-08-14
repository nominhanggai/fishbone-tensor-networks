"""Matrix-product Hamiltonians assembled from sums of operator products.

This module intentionally uses a general construction instead of a hand-written
finite-state machine.  A Hamiltonian is first expressed as

``H = sum_r c_r O[r, 0] (x) ... (x) O[r, n-1]``

and the product label ``r`` becomes the MPO bond. Exact QR/SVD minimization then
removes redundant auxiliary directions. The construction is transparent, easy to
validate, and applies to every supported representation through one compiler.
"""
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from fishbonett.bath.chain import get_bath_nn_paras, star_transform
from fishbonett.operators import annihilate, create, number, sigma_x, sigma_z

SX = sigma_x.astype(complex)
SZ = sigma_z.astype(complex)

__all__ = [
    "MPOEncoding", "product_sum_mpo", "compress_mpo", "chain_coeffs", "resolve_system",
    "build_chain_mpo", "build_static_star_mpo", "build_star_mpo",
    "encode_schrodinger_chain", "encode_schrodinger_star", "encode_interaction",
    "encode_interaction_chain", "encode_interaction_star", "star_transform",
]


@dataclass(frozen=True)
class MPOEncoding:
    """A prepared, static or time-dependent, matrix-product Hamiltonian."""

    n_sites: int
    phys_dim: int
    system: Tuple
    mpo: Callable
    static: bool


def _identity_products(dims, count):
    return [[np.eye(d, dtype=complex) for d in dims] for _ in range(count)]


def product_sum_mpo(dims: Sequence[int], products, coefficients=None):
    """Compile a sum of full-length tensor products into an MPO.

    Parameters
    ----------
    dims
        Physical dimension at each site.
    products
        Sequence of rows; row ``r`` contains one square local operator per site.
    coefficients
        Optional scalar coefficient per row.  It is absorbed at the first site.

    Notes
    -----
    Each product uses one conserved bond label.  The first tensor creates that
    label, intermediate tensors pass it diagonally, and the last tensor closes
    it.  Summing over the label therefore gives precisely the requested sum.
    """
    dims = tuple(int(d) for d in dims)
    rows = [[np.asarray(op, complex) for op in row] for row in products]
    if not rows:
        raise ValueError("an MPO needs at least one product term")
    if any(len(row) != len(dims) for row in rows):
        raise ValueError("every product must contain one operator per site")
    for row in rows:
        for d, op in zip(dims, row):
            if op.shape != (d, d):
                raise ValueError(
                    f"local operator shape {op.shape} does not match {(d, d)}")
    coeff = (np.ones(len(rows), complex) if coefficients is None
             else np.asarray(coefficients, complex))
    if coeff.shape != (len(rows),):
        raise ValueError("coefficients must have one entry per product")

    rank = len(rows)
    if len(dims) == 1:
        out = sum((coeff[r] * rows[r][0] for r in range(rank)),
                  np.zeros((dims[0], dims[0]), complex))
        return [out.reshape(1, 1, dims[0], dims[0])]

    mpo = [np.stack([coeff[r] * rows[r][0] for r in range(rank)], axis=0)
           .reshape(1, rank, dims[0], dims[0])]
    for site in range(1, len(dims) - 1):
        d = dims[site]
        tensor = np.zeros((rank, rank, d, d), complex)
        for r, row in enumerate(rows):
            tensor[r, r] = row[site]
        mpo.append(tensor)
    mpo.append(np.stack([row[-1] for row in rows], axis=0)
               .reshape(rank, 1, dims[-1], dims[-1]))
    return compress_mpo(mpo)


def compress_mpo(mpo, tolerance=1e-13):
    """Remove exact linear dependencies from an MPO's auxiliary bonds.

    Product-sum compilation starts with one bond label per Hamiltonian term.
    QR followed by a reverse SVD is a representation-independent automaton
    minimization: it retains the span of distinct operator prefixes and suffixes
    rather than relying on a manually encoded state machine.
    """
    out = [np.asarray(tensor, complex).copy() for tensor in mpo]
    if len(out) < 2:
        return out
    for site in range(len(out) - 1):
        left, right, d_out, d_in = out[site].shape
        matrix = np.transpose(out[site], (0, 2, 3, 1)).reshape(
            left * d_out * d_in, right)
        q, residual = np.linalg.qr(matrix, mode="reduced")
        rank = q.shape[1]
        out[site] = np.transpose(
            q.reshape(left, d_out, d_in, rank), (0, 3, 1, 2))
        out[site + 1] = np.einsum(
            "xo,orij->xrij", residual, out[site + 1], optimize=True)
    for site in range(len(out) - 1, 0, -1):
        left, right, d_out, d_in = out[site].shape
        matrix = np.transpose(out[site], (0, 2, 3, 1)).reshape(
            left, d_out * d_in * right)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        scale = singular[0] if singular.size else 1.0
        rank = max(1, int(np.sum(singular > tolerance * scale)))
        u, singular, vh = u[:, :rank], singular[:rank], vh[:rank]
        out[site] = np.transpose(
            vh.reshape(rank, d_out, d_in, right), (0, 3, 1, 2))
        transfer = u * singular[None, :]
        out[site - 1] = np.einsum(
            "loij,ok->lkij", out[site - 1], transfer, optimize=True)
    return out


def chain_coeffs(sd, n_chain, domain, discretizer=None):
    """Return chain frequencies, hoppings, and the system coupling strength."""
    frequencies, couplings = get_bath_nn_paras(
        sd, n_chain, list(domain), discretizer=discretizer)
    return (np.asarray(frequencies, float),
            np.asarray(couplings[1:], float), float(couplings[0]))


def resolve_system(hsys, cop, init, eps_bias=0.0, V=1.0):
    """Validate or construct the system Hamiltonian, coupling, and state."""
    h = (np.asarray(hsys, complex) if hsys is not None
         else 0.5 * eps_bias * SZ + V * SX)
    coupling = np.asarray(cop, complex) if cop is not None else SZ.copy()
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("hsys must be square")
    if coupling.shape != h.shape:
        raise ValueError("cop must have the same shape as hsys")
    if init is None:
        state = np.zeros(h.shape[0], complex)
        state[0] = 1.0
    else:
        state = np.asarray(init, complex).reshape(-1)
    if state.shape != (h.shape[0],):
        raise ValueError("initial state dimension does not match hsys")
    return h, coupling, state / np.linalg.norm(state)


def build_chain_mpo(hsys, cop, c0, eps_chain, t_chain, d):
    """Build the nearest-neighbour chain Hamiltonian as a product-sum MPO."""
    frequencies = np.asarray(eps_chain, float)
    hoppings = np.asarray(t_chain, float)
    if hoppings.shape != (max(0, len(frequencies) - 1),):
        raise ValueError("t_chain must have n_modes - 1 entries")
    dims = [np.asarray(hsys).shape[0]] + [int(d)] * len(frequencies)
    a, adag, n_op = annihilate(d), create(d), number(d)
    rows = []
    coeff = []

    row = _identity_products(dims, 1)[0]
    row[0] = np.asarray(hsys, complex)
    rows.append(row); coeff.append(1.0)

    row = _identity_products(dims, 1)[0]
    row[0] = np.asarray(cop, complex)
    row[1] = a + adag
    rows.append(row); coeff.append(c0)

    for mode, omega in enumerate(frequencies):
        row = _identity_products(dims, 1)[0]
        row[mode + 1] = n_op
        rows.append(row); coeff.append(omega)
    for mode, hop in enumerate(hoppings):
        for left, right in ((adag, a), (a, adag)):
            row = _identity_products(dims, 1)[0]
            row[mode + 1] = left
            row[mode + 2] = right
            rows.append(row); coeff.append(hop)
    return product_sum_mpo(dims, rows, coeff)


def build_static_star_mpo(freq, coup, hsys, cop, d):
    """Build the ``schrodinger-star`` Hamiltonian as an MPO."""
    freq = np.asarray(freq, float)
    coup = np.asarray(coup, complex)
    if coup.shape != freq.shape:
        raise ValueError("freq and coup must have the same shape")
    dims = [np.asarray(hsys).shape[0]] + [int(d)] * len(freq)
    a, adag, n_op = annihilate(d), create(d), number(d)
    rows, coeff = [], []
    row = _identity_products(dims, 1)[0]
    row[0] = np.asarray(hsys, complex)
    rows.append(row); coeff.append(1.0)
    for mode, omega in enumerate(freq):
        row = _identity_products(dims, 1)[0]
        row[mode + 1] = n_op
        rows.append(row); coeff.append(omega)
        row = _identity_products(dims, 1)[0]
        row[0] = np.asarray(cop, complex)
        row[mode + 1] = coup[mode] * a + np.conj(coup[mode]) * adag
        rows.append(row); coeff.append(1.0)
    return product_sum_mpo(dims, rows, coeff)


def build_star_mpo(dcoup, hsys, cop, d):
    """Build the interaction-picture star Hamiltonian at one instant."""
    dcoup = np.asarray(dcoup, complex)
    dims = [np.asarray(hsys).shape[0]] + [int(d)] * len(dcoup)
    a, adag = annihilate(d), create(d)
    rows, coeff = [], []
    row = _identity_products(dims, 1)[0]
    row[0] = np.asarray(hsys, complex)
    rows.append(row); coeff.append(1.0)
    for mode, amplitude in enumerate(dcoup):
        row = _identity_products(dims, 1)[0]
        row[0] = np.asarray(cop, complex)
        row[mode + 1] = amplitude * a + np.conj(amplitude) * adag
        rows.append(row); coeff.append(1.0)
    return product_sum_mpo(dims, rows, coeff)


def encode_interaction(representation, init=None):
    """Encode an engine-independent interaction representation as an MPO."""
    if representation.frequencies is None:
        raise ValueError("build the interaction representation before encoding it")
    system = resolve_system(
        representation.h_sys, representation.coupling, init)

    def at_time(t):
        coefficients = representation.coefficients(t)
        return build_star_mpo(
            coefficients[::-1], system[0], system[1],
            representation.pd_boson[0])

    return MPOEncoding(
        representation.len_boson + 1,
        representation.pd_boson[0],
        system,
        at_time,
        False,
    )


def _star_values(sd, n_chain, domain, discretizer, compiled):
    if compiled is None:
        return star_transform(sd, n_chain, domain, discretizer)
    if compiled.n_channels != 1:
        raise ValueError("this MPO encoding requires one bath channel")
    return (compiled.frequencies, compiled.couplings[0],
            compiled.chain_transform)


def encode_schrodinger_chain(sd=None, domain=None, *, n_chain, d, hsys=None, cop=None,
                    init=None, eps_bias=0.0, V=1.0, discretizer=None,
                    compiled=None):
    system = resolve_system(hsys, cop, init, eps_bias, V)
    if compiled is None:
        freq, hop, c0 = chain_coeffs(sd, n_chain, domain, discretizer)
    else:
        freq, hop, c0 = (compiled.frequencies, compiled.hoppings,
                         compiled.system_coupling)
        n_chain, d = compiled.n_modes, compiled.phys_dim
    mpo = build_chain_mpo(system[0], system[1], c0, freq, hop, d)
    return MPOEncoding(len(mpo), d, system, lambda _t=None: mpo, True)


def encode_schrodinger_star(sd=None, domain=None, *, n_chain, d, hsys=None,
                          cop=None, init=None, eps_bias=0.0, V=1.0,
                          discretizer=None, compiled=None):
    system = resolve_system(hsys, cop, init, eps_bias, V)
    freq, coup, _ = _star_values(sd, n_chain, domain, discretizer, compiled)
    if compiled is not None:
        n_chain, d = compiled.n_modes, compiled.phys_dim
    mpo = build_static_star_mpo(freq, coup, system[0], system[1], d)
    return MPOEncoding(n_chain + 1, d, system, lambda _t=None: mpo, True)


def encode_interaction_chain(sd=None, domain=None, *, n_chain, d, hsys=None,
                       cop=None, init=None, eps_bias=0.0, V=1.0,
                       discretizer=None, compiled=None):
    system = resolve_system(hsys, cop, init, eps_bias, V)
    freq, coup, transform = _star_values(
        sd, n_chain, domain, discretizer, compiled)
    if transform is None:
        raise ValueError("interaction-chain requires a star-to-chain transform")
    if compiled is not None:
        n_chain, d = compiled.n_modes, compiled.phys_dim

    def at_time(t):
        amplitude = transform @ (coup * np.exp(-1j * freq * t))
        return build_star_mpo(amplitude[::-1], system[0], system[1], d)

    return MPOEncoding(n_chain + 1, d, system, at_time, False)


def encode_interaction_star(sd=None, domain=None, *, n_chain, d, hsys=None,
                      cop=None, init=None, eps_bias=0.0, V=1.0,
                      discretizer=None, compiled=None):
    system = resolve_system(hsys, cop, init, eps_bias, V)
    freq, coup, _ = _star_values(sd, n_chain, domain, discretizer, compiled)
    if compiled is not None:
        n_chain, d = compiled.n_modes, compiled.phys_dim

    def at_time(t):
        amplitude = coup * np.exp(-1j * freq * t)
        return build_star_mpo(amplitude[::-1], system[0], system[1], d)

    return MPOEncoding(n_chain + 1, d, system, at_time, False)
