"""Private CuPy implementation of fishbonett's randomized range-finder SVD.

Importing this optional module requires CuPy and a CUDA-capable runtime.  The
algorithm mirrors :func:`fishbonett.randomized.randomized_svd` while keeping all
arrays on the device.
"""
import cupy as cp

__all__ = ["adaptive_svd", "rsvd"]

# Keep the CPU and GPU crossover policy identical. Exact SVD is faster and
# deterministic for matrices below this smaller-dimension crossover.
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


def _threshold_exact(matrix, eps, max_rank):
    u, values, vh = cp.linalg.svd(matrix, full_matrices=False)
    scale = values[0] if values.size and values[0] > 0 else 1.0
    keep = max(1, int(cp.sum(values > eps * scale).item()))
    if max_rank is not None:
        keep = min(keep, int(max_rank))
    return u[:, :keep], values[:keep], vh[:keep]


def _projected_svd(matrix, width, n_iter, rng):
    if matrix.shape[0] >= matrix.shape[1]:
        basis = _range(matrix, width, n_iter, rng)
        projected = basis.conj().T @ matrix
        small_u, values, vh = cp.linalg.svd(projected, full_matrices=False)
        u = basis @ small_u
        residual_squared = 0.0
        for start in range(0, matrix.shape[1], 32):
            stop = min(matrix.shape[1], start + 32)
            difference = (
                matrix[:, start:stop] - basis @ projected[:, start:stop])
            residual_squared += float(cp.vdot(difference, difference).real)
    else:
        basis = _range(matrix.conj().T, width, n_iter, rng)
        projected = matrix @ basis
        u, values, small_vh = cp.linalg.svd(projected, full_matrices=False)
        vh = small_vh @ basis.conj().T
        residual_squared = 0.0
        basis_h = basis.conj().T
        for start in range(0, matrix.shape[0], 32):
            stop = min(matrix.shape[0], start + 32)
            difference = (
                matrix[start:stop] - projected[start:stop] @ basis_h)
            residual_squared += float(cp.vdot(difference, difference).real)
    residual = float(max(0.0, residual_squared) ** 0.5)
    return u, values, vh, residual


def adaptive_svd(A, *, eps, max_rank=None, backend="auto", initial_rank=16,
                 n_iter=2, oversample=8, certificate_factor=0.25, seed=None):
    """GPU counterpart of the certified threshold-controlled CPU policy."""
    matrix = cp.asarray(A)
    if matrix.ndim != 2:
        raise ValueError("adaptive_svd expects a two-dimensional matrix")
    if not cp.issubdtype(matrix.dtype, cp.inexact):
        matrix = matrix.astype(cp.float64)
    if not 0 <= float(eps) < float("inf"):
        raise ValueError("eps must be finite and non-negative")
    if backend not in {"auto", "exact", "randomized"}:
        raise ValueError("backend must be 'auto', 'exact', or 'randomized'")
    limit = min(matrix.shape)
    if max_rank is not None:
        if int(max_rank) < 1:
            raise ValueError("max_rank must be positive or None")
        max_rank = min(int(max_rank), limit)
    if (backend == "exact" or (backend == "auto" and limit <= EXACT_BELOW)
            or (eps == 0 and max_rank is None)):
        return _threshold_exact(matrix, eps, max_rank)

    rng = cp.random.default_rng(seed)
    trial = min(limit, int(initial_rank))
    if max_rank is not None:
        trial = min(trial, max_rank)
    while True:
        width = min(limit, trial + int(oversample))
        if width == limit:
            return _threshold_exact(matrix, eps, max_rank)
        u, values, vh, residual = _projected_svd(
            matrix, width, int(n_iter), rng)
        scale = float(values[0].item()) if values.size and values[0] > 0 else 1.0
        cutoff = float(eps) * scale
        above = int(cp.sum(values > cutoff).item())
        keep = max(1, above)
        if max_rank is not None:
            keep = min(keep, max_rank)
        cap_is_resolved = (
            max_rank is not None and keep == max_rank and above >= max_rank)
        ambiguous_cutoff = (
            cutoff > 0
            and bool(cp.any(cp.abs(values - cutoff) <= residual).item()))
        certified = (
            cutoff > 0 and residual <= certificate_factor * cutoff
            and above < values.size and not ambiguous_cutoff)
        if cap_is_resolved or certified:
            return u[:, :keep], values[:keep], vh[:keep]
        trial = min(limit, max(trial + 1, 2 * trial, above + 1))
