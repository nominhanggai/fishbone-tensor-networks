"""Randomized low-rank linear algebra used by tensor truncation.

This implementation follows the randomized range finder described by Halko,
Martinsson and Tropp (SIAM Review 53, 2011): sketch, orthogonalize, power-iterate,
and diagonalize the projected matrix.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from fishbonett._svd import robust_svd as _robust_svd

__all__ = ["randomized_svd", "random_seed", "EXACT_BELOW"]

#: Sketching only pays on large blocks.  Below this leading dimension the exact
#: LAPACK SVD is *faster* as well as deterministic, so it is used instead.
#: Measured on this package's truncation shapes (complex, keeping ``rank``):
#:
#: ===========  ====  ========  ===========  =======
#: matrix       keep  exact     randomized   speedup
#: ===========  ====  ========  ===========  =======
#: 48 x 48        12  0.258 ms  0.366 ms     0.70x
#: 160 x 160      20  46.9 ms   13.3 ms      3.51x
#: 320 x 320      40  206 ms    110 ms       1.87x
#: 640 x 640      64  816 ms    305 ms       2.68x
#: ===========  ====  ========  ===========  =======
#:
#: The 48 x 48 row is the regime the test suite runs in, where sketching cost
#: 30% *more* time and gave up reproducibility for it.
EXACT_BELOW = 128


# ``numpy.random`` preserves the historical response to ``np.random.seed``.
# A simulation can override it locally without mutating process-global state.
_ACTIVE_RNG = ContextVar("fishbonett_rng", default=np.random)


@contextmanager
def random_seed(seed=None):
    """Use a run-local random generator while inside the context.

    ``seed=None`` uses the NumPy global generator. An integer
    creates an isolated ``Generator`` so two runs with the same seed are
    reproducible without changing randomness elsewhere in the process.
    """
    rng = np.random if seed is None else np.random.default_rng(seed)
    token = _ACTIVE_RNG.set(rng)
    try:
        yield rng
    finally:
        _ACTIVE_RNG.reset(token)


def _next_seed():
    """Draw a reproducible device-RNG seed from the active run generator."""
    rng = _ACTIVE_RNG.get()
    upper = np.iinfo(np.uint32).max
    if hasattr(rng, "integers"):
        return int(rng.integers(0, upper, dtype=np.uint32))
    return int(rng.randint(0, upper))


def _standard_normal(rng, shape, dtype):
    values = rng.standard_normal(shape)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        values = values + 1j * rng.standard_normal(shape)
    return np.asarray(values, dtype=dtype)


def _range(A, width, n_iter, rng):
    """Orthonormal basis for the leading column space of ``A``."""
    omega = _standard_normal(rng, (A.shape[1], width), A.dtype)
    Q, _ = np.linalg.qr(A @ omega, mode="reduced")
    for _ in range(n_iter):
        Z, _ = np.linalg.qr(A.conj().T @ Q, mode="reduced")
        Q, _ = np.linalg.qr(A @ Z, mode="reduced")
    return Q


def randomized_svd(A, rank, *, n_iter=2, oversample=None, rng=None):
    """Approximate the leading singular triplets of a two-dimensional array.

    Parameters
    ----------
    A
        Real or complex matrix.
    rank
        Number of singular triplets to return.
    n_iter
        Stabilized power iterations for slowly decaying spectra.
    oversample
        Extra sketch vectors. Defaults to ``max(2, rank)`` and is capped by the
        smaller matrix dimension.
    rng
        NumPy-compatible generator. When omitted, uses the active simulation RNG.

    Notes
    -----
    Falls back to the exact SVD when the sketch would span the whole matrix, and
    when the matrix is smaller than :data:`EXACT_BELOW` -- there the exact
    decomposition is the faster one anyway.
    """
    matrix = np.asarray(A)
    if matrix.ndim != 2:
        raise ValueError("randomized_svd expects a two-dimensional matrix")
    if not np.issubdtype(matrix.dtype, np.inexact):
        matrix = matrix.astype(float)
    limit = min(matrix.shape)
    if not 1 <= int(rank) <= limit:
        raise ValueError(f"rank must be between 1 and {limit}, got {rank!r}")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    rank = int(rank)
    extra = max(2, rank) if oversample is None else int(oversample)
    width = min(limit, rank + max(0, extra))
    if width == limit or limit <= EXACT_BELOW:
        U, values, Vh = _robust_svd(matrix, full_matrices=False)
        return U[:, :rank], values[:rank], Vh[:rank]

    active = _ACTIVE_RNG.get() if rng is None else rng
    if matrix.shape[0] >= matrix.shape[1]:
        Q = _range(matrix, width, n_iter, active)
        small = Q.conj().T @ matrix
        U_small, values, Vh = _robust_svd(small, full_matrices=False)
        U = Q @ U_small
        return U[:, :rank], values[:rank], Vh[:rank]

    # Work on A^H when the row space is smaller, then swap the singular vectors.
    Q = _range(matrix.conj().T, width, n_iter, active)
    small = matrix @ Q
    U, values, Vh_small = _robust_svd(small, full_matrices=False)
    Vh = Vh_small @ Q.conj().T
    return U[:, :rank], values[:rank], Vh[:rank]
