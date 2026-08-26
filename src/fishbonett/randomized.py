"""Randomized low-rank linear algebra used by tensor truncation.

This implementation follows the randomized range finder described by Halko,
Martinsson and Tropp (SIAM Review 53, 2011): sketch, orthogonalize, power-iterate,
and diagonalize the projected matrix.  :func:`adaptive_svd` adds the part a
threshold-controlled tensor-network calculation needs: it grows the sketch until
the unresolved Frobenius norm certifies that no omitted singular direction can
exceed the requested relative cutoff.  If that certificate would require nearly
the whole matrix, it uses the exact LAPACK decomposition instead.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib

import numpy as np
from fishbonett._svd import robust_svd as _robust_svd

__all__ = [
    "AdaptiveSVDInfo", "adaptive_svd", "randomized_svd", "random_seed",
    "current_svd_backend", "svd_statistics", "EXACT_BELOW",
]

#: Sketching only pays on large blocks.  Below this smaller matrix dimension the exact
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


# The NumPy global generator makes ``np.random.seed`` effective when no run-local
# generator is active.
_ACTIVE_RNG = ContextVar("fishbonett_rng", default=np.random)
# Decompositions performed while constructing a representation (before a
# SimulationPlan enters its run context) should be reproducible too. A seeded
# run temporarily overrides this value; ``random_seed(None)`` explicitly opts
# into NumPy's global generator.
_ACTIVE_SEED = ContextVar("fishbonett_seed", default=0)
_ACTIVE_STATS = ContextVar("fishbonett_svd_stats", default=None)
_ACTIVE_BACKEND = ContextVar("fishbonett_svd_backend", default="auto")


@dataclass(frozen=True)
class AdaptiveSVDInfo:
    """How one threshold-controlled decomposition was resolved."""

    backend: str
    attempts: int
    trial_rank: int
    retained_rank: int
    residual_norm: float
    cutoff: float
    exact_fallback: bool = False


def _empty_statistics():
    return {
        "exact_calls": 0,
        "randomized_calls": 0,
        "exact_fallbacks": 0,
        "maximum_trial_rank": 0,
        "maximum_retained_rank": 0,
        "maximum_residual_ratio": 0.0,
    }


@contextmanager
def random_seed(seed=None, *, backend="auto"):
    """Use a run-local random generator while inside the context.

    ``seed=None`` uses the NumPy global generator. An integer creates an isolated
    ``Generator`` so two runs with the same seed are reproducible without
    changing randomness elsewhere in the process. ``backend`` scopes the SVD
    policy to the same context.
    """
    if backend not in {"auto", "exact", "randomized"}:
        raise ValueError("backend must be 'auto', 'exact', or 'randomized'")
    rng = np.random if seed is None else np.random.default_rng(seed)
    token = _ACTIVE_RNG.set(rng)
    seed_token = _ACTIVE_SEED.set(None if seed is None else int(seed))
    statistics_token = _ACTIVE_STATS.set(_empty_statistics())
    backend_token = _ACTIVE_BACKEND.set(backend)
    try:
        yield rng
    finally:
        _ACTIVE_BACKEND.reset(backend_token)
        _ACTIVE_STATS.reset(statistics_token)
        _ACTIVE_SEED.reset(seed_token)
        _ACTIVE_RNG.reset(token)


def svd_statistics():
    """Return a copy of the decomposition counters for the active run."""
    statistics = _ACTIVE_STATS.get()
    return _empty_statistics() if statistics is None else dict(statistics)


def current_svd_backend():
    """Backend selected by the active simulation context."""
    return _ACTIVE_BACKEND.get()


def _record(info):
    statistics = _ACTIVE_STATS.get()
    if statistics is None:
        return
    key = "exact_calls" if info.backend == "exact" else "randomized_calls"
    statistics[key] += 1
    statistics["exact_fallbacks"] += int(info.exact_fallback)
    statistics["maximum_trial_rank"] = max(
        statistics["maximum_trial_rank"], int(info.trial_rank))
    statistics["maximum_retained_rank"] = max(
        statistics["maximum_retained_rank"], int(info.retained_rank))
    if info.cutoff > 0:
        statistics["maximum_residual_ratio"] = max(
            statistics["maximum_residual_ratio"],
            float(info.residual_norm / info.cutoff),
        )


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


def _matrix_rng(matrix, rng):
    """Return a sketch generator stable across checkpoint segmentation.

    A simulation with an integer seed must use the same sketch for the same
    decomposition whether its propagation loop is uninterrupted or restarted
    from a checkpoint.  A small deterministic sample of the matrix provides a
    call-local key without hashing the complete tensor.  Outside a seeded run,
    the ordinary active random stream is retained.
    """
    if rng is not None:
        return rng
    seed = _ACTIVE_SEED.get()
    if seed is None:
        return _ACTIVE_RNG.get()
    value = np.asarray(matrix)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    flat = value.reshape(-1)
    if flat.size:
        count = min(64, flat.size)
        indices = np.linspace(0, flat.size - 1, count, dtype=np.int64)
        sample = np.ascontiguousarray(flat[indices])
        digest.update(sample.view(np.uint8))
    local_seed = int.from_bytes(digest.digest(), "little", signed=False)
    return np.random.default_rng(local_seed)


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


def _projected_svd(matrix, width, n_iter, rng):
    """SVD inside a randomized range and its exact projection residual."""
    if matrix.shape[0] >= matrix.shape[1]:
        basis = _range(matrix, width, n_iter, rng)
        projected = basis.conj().T @ matrix
        small_u, values, vh = _robust_svd(projected, full_matrices=False)
        u = basis @ small_u
        residual_squared = 0.0
        for start in range(0, matrix.shape[1], 32):
            stop = min(matrix.shape[1], start + 32)
            difference = (
                matrix[:, start:stop] - basis @ projected[:, start:stop])
            residual_squared += float(np.vdot(difference, difference).real)
    else:
        basis = _range(matrix.conj().T, width, n_iter, rng)
        projected = matrix @ basis
        u, values, small_vh = _robust_svd(projected, full_matrices=False)
        vh = small_vh @ basis.conj().T
        residual_squared = 0.0
        basis_h = basis.conj().T
        for start in range(0, matrix.shape[0], 32):
            stop = min(matrix.shape[0], start + 32)
            difference = (
                matrix[start:stop] - projected[start:stop] @ basis_h)
            residual_squared += float(np.vdot(difference, difference).real)
    residual = np.sqrt(max(0.0, residual_squared))
    return u, values, vh, float(residual)


def _exact_threshold_svd(matrix, eps, max_rank, extra_rank):
    u, values, vh = _robust_svd(matrix, full_matrices=False)
    scale = values[0] if values.size and values[0] > 0 else 1.0
    above = int(np.sum(values > float(eps) * scale))
    keep = max(1, min(values.size, above + extra_rank))
    if max_rank is not None:
        keep = min(keep, max_rank)
    return u[:, :keep], values[:keep], vh[:keep], float(eps) * float(scale)


def adaptive_svd(
    A,
    *,
    eps,
    max_rank=None,
    extra_rank=0,
    backend=None,
    initial_rank=16,
    n_iter=2,
    oversample=8,
    certificate_factor=0.25,
    rng=None,
    return_info=False,
):
    """Threshold-truncate a matrix with adaptive randomized range finding.

    ``eps`` is relative to the leading singular value.  ``max_rank=None`` leaves
    the bond uncapped; otherwise it is a hard upper bound.  ``extra_rank`` keeps
    a requested number of Schmidt directions below the threshold, as used by
    two-site TDVP to permit bond growth.

    The randomized path stops only when the Frobenius norm outside its sampled
    subspace is at most ``certificate_factor * eps * s[0]``.  Since the spectral
    norm is bounded by the Frobenius norm, an omitted direction then cannot cross
    the requested cutoff by more than that conservative margin.  Slowly decaying
    spectra naturally reach the exact fallback instead of receiving an
    uncertified truncation.
    """
    matrix = np.asarray(A)
    if matrix.ndim != 2:
        raise ValueError("adaptive_svd expects a two-dimensional matrix")
    if not np.issubdtype(matrix.dtype, np.inexact):
        matrix = matrix.astype(float)
    if (isinstance(eps, (bool, np.bool_)) or not np.isfinite(eps) or eps < 0):
        raise ValueError("eps must be finite and non-negative")
    if max_rank is not None:
        if (isinstance(max_rank, (bool, np.bool_))
                or not isinstance(max_rank, (int, np.integer))
                or max_rank < 1):
            raise ValueError("max_rank must be a positive integer or None")
        max_rank = int(max_rank)
    if (not isinstance(extra_rank, (int, np.integer)) or extra_rank < 0):
        raise ValueError("extra_rank must be a non-negative integer")
    backend = _ACTIVE_BACKEND.get() if backend is None else backend
    if backend not in {"auto", "exact", "randomized"}:
        raise ValueError("backend must be 'auto', 'exact', or 'randomized'")
    if not isinstance(initial_rank, (int, np.integer)) or initial_rank < 1:
        raise ValueError("initial_rank must be a positive integer")
    if not isinstance(oversample, (int, np.integer)) or oversample < 0:
        raise ValueError("oversample must be a non-negative integer")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    if not 0 < certificate_factor <= 1:
        raise ValueError("certificate_factor must lie in (0, 1]")

    limit = min(matrix.shape)
    if limit < 1:
        raise ValueError("adaptive_svd does not accept an empty matrix")
    if max_rank is not None:
        max_rank = min(max_rank, limit)
    exact_requested = (
        backend == "exact"
        or (backend == "auto" and limit <= EXACT_BELOW)
        or (float(eps) == 0.0 and max_rank is None)
    )
    if exact_requested:
        u, values, vh, cutoff = _exact_threshold_svd(
            matrix, eps, max_rank, int(extra_rank))
        info = AdaptiveSVDInfo(
            "exact", 1, limit, len(values), 0.0, cutoff, False)
        _record(info)
        result = (u, values, vh)
        return (*result, info) if return_info else result

    local_rng = _matrix_rng(matrix, rng)
    trial = min(limit, max(1, int(initial_rank)))
    if max_rank is not None:
        trial = min(trial, max_rank)
    attempts = 0
    while True:
        attempts += 1
        width = min(limit, trial + int(oversample))
        if width == limit:
            u, values, vh, cutoff = _exact_threshold_svd(
                matrix, eps, max_rank, int(extra_rank))
            info = AdaptiveSVDInfo(
                "exact", attempts, limit, len(values), 0.0, cutoff,
                exact_fallback=attempts > 1,
            )
            _record(info)
            result = (u, values, vh)
            return (*result, info) if return_info else result

        u, values, vh, residual = _projected_svd(
            matrix, width, int(n_iter), local_rng)
        scale = values[0] if values.size and values[0] > 0 else 1.0
        cutoff = float(eps) * float(scale)
        above = int(np.sum(values > cutoff))
        desired = max(1, above + int(extra_rank))
        if max_rank is not None:
            desired = min(desired, max_rank)
        cap_is_resolved = (
            max_rank is not None
            and desired == max_rank
            and above + int(extra_rank) >= max_rank
            and width >= max_rank
        )
        boundary_is_sampled = above + int(extra_rank) < values.size
        ambiguous_cutoff = (
            cutoff > 0
            and bool(np.any(np.abs(values - cutoff) <= residual))
        )
        certified = (
            cutoff > 0
            and residual <= float(certificate_factor) * cutoff
            and boundary_is_sampled
            and not ambiguous_cutoff
        )
        if cap_is_resolved or certified:
            u, values, vh = u[:, :desired], values[:desired], vh[:desired]
            info = AdaptiveSVDInfo(
                "randomized", attempts, width, desired, residual, cutoff, False)
            _record(info)
            result = (u, values, vh)
            return (*result, info) if return_info else result

        next_trial = max(trial + 1, 2 * trial, above + int(extra_rank) + 1)
        if max_rank is not None and above + int(extra_rank) >= max_rank:
            next_trial = max_rank
        trial = min(limit, next_trial)
        if trial >= limit:
            # The next iteration takes the exact branch.  Keeping this explicit
            # makes the no-uncertified-result invariant easy to audit.
            continue
