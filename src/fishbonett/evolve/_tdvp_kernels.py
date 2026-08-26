"""Environment contractions and local exponential actions for MPS evolution.

Tensors use ``(left bond, right bond, physical)`` and MPO tensors use
``(left operator bond, right operator bond, physical out, physical in)``.
The exponential action is an Arnoldi process, so the primitive remains correct
for a mildly non-Hermitian effective operator as well as the Hermitian case.

Algorithm references (this is an independent implementation of the published
methods, not a translation of any particular codebase):

* projector-splitting TDVP for matrix product states -- Haegeman, Lubich,
  Oseledets, Vandereycken and Verstraete, Phys. Rev. B 94, 165116 (2016);
* one-site, two-site and bond-adaptive TDVP applied to chain-mapped bosonic
  environments -- Dunnett and Chin, Phys. Rev. B 104, 214302 (2021).
"""
import numpy as np
import scipy.linalg

from fishbonett.operators import sigma_x, sigma_z

SX = sigma_x.astype(complex)
SZ = sigma_z.astype(complex)

_KRY = {"calls": 0, "iters": 0}

# Roundoff guard for declaring a Lanczos vector numerically zero. The modest
# multiple covers accumulation across several inner products without tying the
# decision to a problem-specific absolute scale.
_KRYLOV_BREAKDOWN_ULPS = 64


def krylov_statistics(*, reset=False):
    """Return TDVP Krylov call/iteration counters, optionally resetting them."""
    snapshot = dict(_KRY)
    if reset:
        _KRY.update(calls=0, iters=0)
    return snapshot


def init_mps(n_sites, d, sys_state=None):
    """Return ``|system> (x) |vacuum>`` as a bond-one MPS.

    ``d`` may be one common environment dimension or the dimensions of every
    site after the system.  The latter supports interleaved electronic and
    vibrational sites without changing the TDVP tensor convention.
    """
    if int(n_sites) < 1:
        raise ValueError("n_sites must be positive")
    system = (np.array([1.0, 0.0], complex) if sys_state is None
              else np.asarray(sys_state, complex).reshape(-1))
    norm = np.linalg.norm(system)
    # Some dressed representations deliberately provide a zero placeholder and replace
    # the product state through ``prepare`` before canonicalization.
    if norm > 0:
        system = system / norm
    if np.isscalar(d):
        dimensions = (int(d),) * (int(n_sites) - 1)
    else:
        dimensions = tuple(int(value) for value in d)
        if len(dimensions) != int(n_sites) - 1:
            raise ValueError("d must contain one dimension per non-system site")
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("site dimensions must be positive")
    tensors = [system.reshape(1, 1, -1)]
    for dimension in dimensions:
        tensor = np.zeros((1, 1, dimension), complex)
        tensor[0, 0, 0] = 1.0
        tensors.append(tensor)
    return tensors


def applyH1(center, mpo, left, right):
    """Apply the one-site effective Hamiltonian without forming its matrix."""
    # Contract from the smaller right environment inward.  Explicit pairwise
    # contractions avoid recomputing an einsum path on every Arnoldi iteration.
    stage = np.tensordot(center, right, axes=([1], [2]))
    stage = np.tensordot(mpo, stage, axes=([1, 3], [3, 1]))
    stage = np.tensordot(left, stage, axes=([1, 2], [0, 2]))
    return np.ascontiguousarray(np.transpose(stage, (0, 2, 1)))


def applyH0(center, left, right):
    """Apply the zero-site (bond) effective Hamiltonian."""
    return np.einsum("amc,cr,bmr->ab", left, center, right, optimize=True)


def updateleftenv(tensor, mpo, left):
    """Contract one site into a left environment."""
    return np.einsum(
        "amc,crp,mnqp,abq->bnr", left, tensor, mpo, tensor.conj(),
        optimize=True,
    )


def updaterightenv(tensor, mpo, right):
    """Contract one site into a right environment."""
    return np.einsum(
        "abq,mnqp,crp,bnr->amc", tensor.conj(), mpo, tensor, right,
        optimize=True,
    )


def left_qr(tensor):
    """Split off a left-isometric tensor and a bond-centre matrix."""
    dl, dr, physical = tensor.shape
    matrix = np.transpose(tensor, (0, 2, 1)).reshape(dl * physical, dr)
    isometry, center = np.linalg.qr(matrix, mode="reduced")
    rank = isometry.shape[1]
    return (np.transpose(isometry.reshape(dl, physical, rank), (0, 2, 1)),
            center)


def right_lq(tensor):
    """Split off a bond-centre matrix and a right-isometric tensor."""
    dl, dr, physical = tensor.shape
    isometry_t, center_t = np.linalg.qr(
        tensor.reshape(dl, dr * physical).T, mode="reduced")
    rank = isometry_t.shape[1]
    return center_t.T, isometry_t.T.reshape(rank, dr, physical)


def right_canonicalize(tensors):
    """Copy an MPS and place its orthogonality centre at site zero."""
    state = [np.asarray(tensor, complex).copy() for tensor in tensors]
    for site in range(len(state) - 1, 0, -1):
        center, state[site] = right_lq(state[site])
        state[site - 1] = np.einsum(
            "axp,xr->arp", state[site - 1], center, optimize=True)
    norm = np.linalg.norm(state[0])
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("cannot canonicalize a zero or non-finite state")
    state[0] /= norm
    return state


def init_right_envs(tensors, mpo):
    """Build all right environments for a right-canonical state."""
    if len(tensors) != len(mpo):
        raise ValueError("MPS and MPO lengths differ")
    count = len(tensors)
    environments = [None] * (count + 2)
    boundary = np.ones((1, 1, 1), complex)
    environments[0] = boundary
    environments[count + 1] = boundary
    for site in range(count - 1, -1, -1):
        environments[site + 1] = updaterightenv(
            tensors[site], mpo[site], environments[site + 2])
    return environments


def expmv_lanczos(applyH, tau, v, m=30, tol=1e-7, eshift=False):
    """Compute ``exp(tau H) v`` with a reorthogonalized Arnoldi projection.

    The implementation uses Arnoldi rather than a three-term Lanczos recurrence,
    so it does not assume exact Hermiticity and remains robust to small contraction
    or round-off asymmetries.
    """
    shape = v.shape
    initial = np.asarray(v, complex).reshape(-1)
    scale = np.linalg.norm(initial)
    if scale == 0 or not np.isfinite(scale):
        return np.asarray(v, complex).copy()
    dimension = initial.size
    limit = min(max(1, int(m)), dimension)
    basis = np.zeros((limit + 1, dimension), complex)
    hessenberg = np.zeros((limit + 1, limit), complex)
    basis[0] = initial / scale
    coefficients = np.array([1.0 + 0.0j])
    used = 1

    for column in range(limit):
        residual = np.asarray(
            applyH(basis[column].reshape(shape)), complex).reshape(-1)
        # Two modified Gram--Schmidt passes retain orthogonality when the local
        # effective Hamiltonian has clustered eigenvalues.
        for _ in range(2):
            overlap = basis[:column + 1].conj() @ residual
            hessenberg[:column + 1, column] += overlap
            residual -= overlap @ basis[:column + 1]
        next_norm = np.linalg.norm(residual)
        hessenberg[column + 1, column] = next_norm
        used = column + 1
        projected = hessenberg[:used, :used]
        shift = (np.trace(projected) / used if eshift else 0.0)
        exponential = scipy.linalg.expm(
            tau * (projected - shift * np.eye(used)))
        coefficients = exponential[:, 0] * np.exp(tau * shift)
        estimate = (next_norm * abs(coefficients[-1])
                    if column + 1 < limit else 0.0)
        breakdown = (
            next_norm <= _KRYLOV_BREAKDOWN_ULPS * np.finfo(float).eps * scale
        )
        if breakdown or estimate <= tol * scale or column + 1 == limit:
            break
        basis[column + 1] = residual / next_norm

    _KRY["calls"] += 1
    _KRY["iters"] += used
    return (scale * (coefficients @ basis[:used])).reshape(shape)


def _setbond(tensor, left_dim, right_dim):
    """Embed a tensor in larger bond spaces, preserving existing entries."""
    old_left, old_right, physical = tensor.shape
    out = np.zeros((int(left_dim), int(right_dim), physical), tensor.dtype)
    keep_left = min(old_left, int(left_dim))
    keep_right = min(old_right, int(right_dim))
    out[:keep_left, :keep_right] = tensor[:keep_left, :keep_right]
    return out


def evolveAC(dt, center, mpo, left, right, **kwargs):
    """Evolve a one-site centre forward by ``dt``."""
    target = (left.shape[0], right.shape[0])
    if center.shape[:2] != target:
        center = _setbond(center, *target)
    return expmv_lanczos(
        lambda value: applyH1(value, mpo, left, right),
        -1j * dt, center, **kwargs)


def evolveC(dt, center, left, right, **kwargs):
    """Evolve a zero-site centre backward by ``dt``."""
    return expmv_lanczos(
        lambda value: applyH0(value, left, right),
        1j * dt, center, **kwargs)
