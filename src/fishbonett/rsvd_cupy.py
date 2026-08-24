"""CuPy implementation of fishbonett's randomized range-finder SVD.

Importing this optional module requires CuPy and a CUDA-capable runtime.  The
algorithm mirrors :func:`fishbonett.randomized.randomized_svd` while keeping all
arrays on the device.
"""
import cupy as cp

__all__ = ["rsvd"]

# Keep the CPU and GPU crossover policy identical. Exact SVD is faster and
# deterministic for the small matrices below this leading dimension.
EXACT_BELOW = 128


def _normal(rng, shape, dtype):
    if cp.issubdtype(dtype, cp.complexfloating):
        real_dtype = cp.float32 if dtype == cp.complex64 else cp.float64
        values = rng.standard_normal(shape, dtype=real_dtype)
        values = values + 1j * rng.standard_normal(shape, dtype=real_dtype)
        return values.astype(dtype, copy=False)
    return rng.standard_normal(shape, dtype=dtype)


def _range(matrix, width, n_iter, rng):
    omega = _normal(rng, (matrix.shape[1], width), matrix.dtype)
    Q, _ = cp.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(n_iter):
        Z, _ = cp.linalg.qr(matrix.conj().T @ Q, mode="reduced")
        Q, _ = cp.linalg.qr(matrix @ Z, mode="reduced")
    return Q


def rsvd(A, k=6, *, n_iter=2, l=None, seed=None):
    """Return the leading ``k`` singular triplets of a CuPy matrix.

    ``l`` is the total sketch width, and ``seed`` controls a run-local CuPy
    generator.
    """
    matrix = cp.asarray(A)
    if matrix.ndim != 2:
        raise ValueError("rsvd expects a two-dimensional matrix")
    if not cp.issubdtype(matrix.dtype, cp.inexact):
        matrix = matrix.astype(cp.float64)
    limit = min(matrix.shape)
    if not 1 <= int(k) <= limit:
        raise ValueError(f"k must be between 1 and {limit}, got {k!r}")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    k = int(k)
    width = min(limit, int(l) if l is not None else 2 * k)
    if width < k:
        raise ValueError("l must be at least k")
    if width == limit or limit <= EXACT_BELOW:
        U, values, Vh = cp.linalg.svd(matrix, full_matrices=False)
        return U[:, :k], values[:k], Vh[:k]

    rng = cp.random.default_rng(seed)
    if matrix.shape[0] >= matrix.shape[1]:
        Q = _range(matrix, width, n_iter, rng)
        U_small, values, Vh = cp.linalg.svd(
            Q.conj().T @ matrix, full_matrices=False)
        return (Q @ U_small)[:, :k], values[:k], Vh[:k]

    Q = _range(matrix.conj().T, width, n_iter, rng)
    U, values, Vh_small = cp.linalg.svd(matrix @ Q, full_matrices=False)
    return U[:, :k], values[:k], (Vh_small @ Q.conj().T)[:k]
