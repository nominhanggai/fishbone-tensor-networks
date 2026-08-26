"""Private algebra for compiling sums of operator products into MPO tensors."""

from typing import Sequence

import numpy as np

from fishbonett.contract import _einsum_cached
from fishbonett.linalg import threshold_svd


def identity_product(dimensions):
    """One identity operator for every physical site."""
    return [np.eye(dimension, dtype=complex) for dimension in dimensions]


def dense_operator_mpo(operator, dimensions, tolerance=1e-13):
    """Factor a dense many-site operator into an MPO by exact TT-SVD.

    Parameters
    ----------
    operator
        Square matrix in the product basis ordered by ``dimensions``.
    dimensions
        Physical dimensions from the left end of the MPO to the right.
    tolerance
        Relative cutoff used only to remove floating-point null singular values.

    Notes
    -----
    This private compiler is intended for small, nonlocal gates whose dense
    matrix is already available.  State compression remains controlled by the
    simulation's separate SVD threshold.
    """
    dimensions = tuple(int(dimension) for dimension in dimensions)
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ValueError("dimensions must contain positive integers")
    total = int(np.prod(dimensions, dtype=int))
    value = np.asarray(operator, complex)
    if value.shape != (total, total):
        raise ValueError(
            f"operator has shape {value.shape}, expected {(total, total)}"
        )
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    # Matrix order is (out_0,...,out_N,in_0,...,in_N).  Interleave the local
    # output/input axes before the tensor-train factorization.
    n_sites = len(dimensions)
    tensor = value.reshape(*dimensions, *dimensions)
    order = [axis for site in range(n_sites) for axis in (site, site + n_sites)]
    carry = np.transpose(tensor, order)
    mpo = []
    left_rank = 1
    for dimension in dimensions[:-1]:
        matrix = carry.reshape(left_rank * dimension * dimension, -1)
        u, singular, vh = threshold_svd(matrix, tolerance)
        rank = singular.size
        mpo.append(np.transpose(
            u.reshape(left_rank, dimension, dimension, rank), (0, 3, 1, 2)
        ))
        carry = (singular[:, None] * vh)
        left_rank = rank
    dimension = dimensions[-1]
    mpo.append(carry.reshape(left_rank, 1, dimension, dimension))
    return mpo


def product_sum_mpo(dimensions: Sequence[int], products, coefficients=None):
    """Compile ``sum_r c_r prod_i O[r, i]`` into a compressed MPO.

    The product index is compressed while sweeping from left to right.  Building
    all diagonal ``(n_terms, n_terms, d, d)`` tensors first would use cubic total
    storage for the local Hamiltonians produced by this package, even though
    their final MPO bond is small.
    """
    dimensions = tuple(int(dimension) for dimension in dimensions)
    rows = [[np.asarray(operator, complex) for operator in row]
            for row in products]
    if not rows:
        raise ValueError("an MPO needs at least one product term")
    if any(len(row) != len(dimensions) for row in rows):
        raise ValueError("every product must contain one operator per site")
    for row in rows:
        for dimension, operator in zip(dimensions, row, strict=True):
            if operator.shape != (dimension, dimension):
                raise ValueError(
                    f"local operator shape {operator.shape} does not match "
                    f"{(dimension, dimension)}")

    values = (np.ones(len(rows), complex) if coefficients is None
              else np.asarray(coefficients, complex))
    if values.shape != (len(rows),):
        raise ValueError("coefficients must have one entry per product")

    rank = len(rows)
    if len(dimensions) == 1:
        operator = sum(
            (values[index] * rows[index][0] for index in range(rank)),
            np.zeros((dimensions[0], dimensions[0]), complex),
        )
        return [operator.reshape(1, 1, dimensions[0], dimensions[0])]

    first = np.stack(
        [values[index] * rows[index][0] for index in range(rank)], axis=-1
    )
    matrix = first.reshape(dimensions[0] ** 2, rank)
    u, singular, vh = threshold_svd(matrix, 1e-13)
    kept = singular.size
    mpo = [u.reshape(1, dimensions[0], dimensions[0], kept).transpose(
        0, 3, 1, 2
    )]
    carry = singular[:, None] * vh
    for site in range(1, len(dimensions) - 1):
        dimension = dimensions[site]
        operators = np.stack([row[site] for row in rows], axis=0)
        tensor = _einsum_cached("ar,rij->arij", carry, operators)
        left = tensor.shape[0]
        matrix = np.transpose(tensor, (0, 2, 3, 1)).reshape(
            left * dimension * dimension, rank
        )
        u, singular, vh = threshold_svd(matrix, 1e-13)
        kept = singular.size
        mpo.append(np.transpose(
            u.reshape(left, dimension, dimension, kept), (0, 3, 1, 2)
        ))
        carry = singular[:, None] * vh
    final_operators = np.stack([row[-1] for row in rows], axis=0)
    final = _einsum_cached("ar,rij->aij", carry, final_operators)
    mpo.append(final[:, None, :, :])
    return compress_mpo(mpo)


def compress_mpo(mpo, tolerance=1e-13):
    """Remove linearly dependent auxiliary directions from an MPO."""
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
        out[site + 1] = _einsum_cached(
            "xo,orij->xrij", residual, out[site + 1])

    for site in range(len(out) - 1, 0, -1):
        left, right, d_out, d_in = out[site].shape
        matrix = np.transpose(out[site], (0, 2, 3, 1)).reshape(
            left, d_out * d_in * right)
        u, singular, vh = threshold_svd(matrix, tolerance)
        rank = singular.size
        out[site] = np.transpose(
            vh.reshape(rank, d_out, d_in, right), (0, 3, 1, 2))
        transfer = u * singular[None, :]
        out[site - 1] = _einsum_cached(
            "loij,ok->lkij", out[site - 1], transfer)
    return out
