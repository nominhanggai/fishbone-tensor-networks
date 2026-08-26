"""Projector-splitting sweeps for matrix-product states.

The one-site and two-site integrators share one environment convention and one
palindromic traversal.  Adaptive one-site TDVP enlarges the local tangent
spaces with the unused columns of full QR factorizations before taking an
ordinary one-site sweep.  It never exponentiates a two-site centre.
"""
import numpy as np

from fishbonett.contract import _einsum_cached
#: Maximum new bond directions considered in one sweep.  For TDVP2 these are
#: Schmidt directions beyond the threshold rank.  For adaptive one-site TDVP
#: they are full-QR complement directions tested by the tangent-space measure.
DEFAULT_BOND_EXPAND = 2
from fishbonett.evolve._tdvp_kernels import (
    SZ, _setbond, applyH0, applyH1, evolveAC, evolveC, expmv_lanczos,
    init_right_envs, left_qr, right_lq, updateleftenv, updaterightenv,
)
from fishbonett.linalg import threshold_svd


def measure_rdm(center):
    """Reduced density matrix at an MPS orthogonality centre."""
    rho = _einsum_cached("abp,abq->pq", center, center.conj())
    trace = np.trace(rho)
    if abs(trace) == 0:
        raise ValueError("cannot measure a zero state")
    return rho / trace


def measure_sz(center):
    """Expectation of sigma-z at a two-level orthogonality centre."""
    return float(np.trace(measure_rdm(center) @ SZ).real)


def bonddims(tensors):
    """Bond dimensions for tensors stored as ``(left, right, physical)``."""
    return [tensors[0].shape[0]] + [tensor.shape[1] for tensor in tensors]


def tdvp1sweep(dt2, tensors, mpo, environments=None, **krylov):
    """Advance a fixed-bond MPS by one symmetric one-site TDVP step."""
    count = len(tensors)
    if count == 1:
        left = right = np.ones((1, 1, 1), complex)
        tensors[0] = evolveAC(dt2, tensors[0], mpo[0], left, right, **krylov)
        return tensors, [left, left, right]
    half = 0.5 * dt2
    env = init_right_envs(tensors, mpo) if environments is None else environments
    center = tensors[0]

    for site in range(count - 1):
        center = evolveAC(
            half, center, mpo[site], env[site], env[site + 2], **krylov)
        tensors[site], bond = left_qr(center)
        env[site + 1] = updateleftenv(tensors[site], mpo[site], env[site])
        bond = evolveC(half, bond, env[site + 1], env[site + 2], **krylov)
        center = _einsum_cached("ax,xbs->abs", bond, tensors[site + 1])

    center = evolveAC(
        dt2, center, mpo[-1], env[count - 1], env[count + 1], **krylov)

    for site in range(count - 2, -1, -1):
        bond, tensors[site + 1] = right_lq(center)
        env[site + 2] = updaterightenv(
            tensors[site + 1], mpo[site + 1], env[site + 3])
        bond = evolveC(half, bond, env[site + 1], env[site + 2], **krylov)
        center = _einsum_cached("axp,xb->abp", tensors[site], bond)
        center = evolveAC(
            half, center, mpo[site], env[site], env[site + 2], **krylov)
    tensors[0] = center
    return tensors, env


def _merge2(left, right):
    """Join adjacent MPS tensors into ``(outer-left, outer-right, p, q)``."""
    return _einsum_cached("amp,mbq->abpq", left, right)


def applyH2(theta, mpo_left, mpo_right, left_env, right_env):
    """Apply the two-site effective Hamiltonian."""
    stage = np.tensordot(theta, right_env, axes=([1], [2]))
    stage = np.tensordot(mpo_right, stage, axes=([1, 3], [4, 2]))
    stage = np.tensordot(mpo_left, stage, axes=([1, 3], [0, 3]))
    stage = np.tensordot(left_env, stage, axes=([1, 2], [0, 3]))
    return np.ascontiguousarray(np.transpose(stage, (0, 3, 1, 2)))


def _split2(theta, chi_max, eps, ortho, expand=DEFAULT_BOND_EXPAND):
    """SVD a two-site centre and put the centre on the requested side.

    ``expand`` retains that many Schmidt directions beyond the threshold rank.
    These directions allow entanglement to grow from a product state when its
    first-step Schmidt values are below ``eps``. The retained rank is bounded by
    the threshold rank plus ``expand`` and by ``chi_max``.
    """
    dl, dr, d_left, d_right = theta.shape
    matrix = np.transpose(theta, (0, 2, 3, 1)).reshape(
        dl * d_left, d_right * dr)
    u, singular, vh = threshold_svd(
        matrix, eps, max_rank=chi_max, extra_rank=max(0, int(expand)))
    keep = singular.size
    if ortho == "left":
        first = np.transpose(u.reshape(dl, d_left, keep), (0, 2, 1))
        second = np.transpose(
            (singular[:, None] * vh).reshape(keep, d_right, dr),
            (0, 2, 1))
    elif ortho == "right":
        first = np.transpose(
            (u * singular[None, :]).reshape(dl, d_left, keep),
            (0, 2, 1))
        second = np.transpose(vh.reshape(keep, d_right, dr), (0, 2, 1))
    else:
        raise ValueError("ortho must be 'left' or 'right'")
    return first, second


def tdvp2sweep(dt2, tensors, mpo, chi_max, eps, environments=None,
               expand=DEFAULT_BOND_EXPAND, **krylov):
    """Advance an MPS by a symmetric two-site projector-splitting step.

    ``expand`` is the bond-expansion allowance; see :func:`_split2`.
    """
    count = len(tensors)
    if count < 2:
        return tdvp1sweep(dt2, tensors, mpo, environments, **krylov)
    half = 0.5 * dt2
    env = init_right_envs(tensors, mpo) if environments is None else environments

    for site in range(count - 1):
        theta = _merge2(tensors[site], tensors[site + 1])
        theta = expmv_lanczos(
            lambda value, k=site: applyH2(
                value, mpo[k], mpo[k + 1], env[k], env[k + 3]),
            -1j * half, theta, **krylov)
        tensors[site], tensors[site + 1] = _split2(
            theta, chi_max, eps, "left", expand)
        env[site + 1] = updateleftenv(tensors[site], mpo[site], env[site])
        if site < count - 2:
            tensors[site + 1] = expmv_lanczos(
                lambda value, k=site: applyH1(
                    value, mpo[k + 1], env[k + 1], env[k + 3]),
                1j * half, tensors[site + 1], **krylov)

    for site in range(count - 2, -1, -1):
        theta = _merge2(tensors[site], tensors[site + 1])
        theta = expmv_lanczos(
            lambda value, k=site: applyH2(
                value, mpo[k], mpo[k + 1], env[k], env[k + 3]),
            -1j * half, theta, **krylov)
        tensors[site], tensors[site + 1] = _split2(
            theta, chi_max, eps, "right", expand)
        env[site + 2] = updaterightenv(
            tensors[site + 1], mpo[site + 1], env[site + 3])
        if site > 0:
            tensors[site] = expmv_lanczos(
                lambda value, k=site: applyH1(
                    value, mpo[k], env[k], env[k + 2]),
                1j * half, tensors[site], **krylov)
    return tensors, env


def _pad_bonds(tensors, D, noise=1e-10, seed=0):
    """Embed an MPS in a fixed-bond manifold, adding reproducible tiny noise."""
    if D is None:
        return tensors
    physical = [tensor.shape[2] for tensor in tensors]
    total = len(tensors)
    bonds = [1]
    left_space = 1
    right_spaces = [1] * (total + 1)
    product = 1
    for site in range(total - 1, -1, -1):
        product *= physical[site]
        right_spaces[site] = product
    for site in range(total - 1):
        left_space *= physical[site]
        bonds.append(min(int(D), left_space, right_spaces[site + 1]))
    bonds.append(1)
    rng = np.random.default_rng(seed)
    out = []
    for site, tensor in enumerate(tensors):
        expanded = _setbond(tensor, bonds[site], bonds[site + 1])
        if expanded.shape != tensor.shape and noise:
            perturbation = (rng.standard_normal(expanded.shape)
                            + 1j * rng.standard_normal(expanded.shape))
            occupied = np.zeros(expanded.shape, bool)
            occupied[:tensor.shape[0], :tensor.shape[1]] = True
            expanded[~occupied] = noise * perturbation[~occupied]
        out.append(expanded)
    return out


def _partial_full_qr(matrix, target):
    """Return a reproducible partial full-QR basis and the reduced R.

    A reduced QR supplies the occupied columns.  The remaining columns are
    completed from projected coordinate vectors in a deterministic pivot
    order.  Full-QR complements are mathematically non-unique, and accepting
    the LAPACK-specific completion made adaptive one-site TDVP choose different
    tangent directions on otherwise equivalent installations.
    """
    matrix = np.asarray(matrix)
    rows, columns = matrix.shape
    if columns > rows or target < columns or target > rows:
        raise ValueError("requested QR dimensions are incompatible")
    basis, triangular = np.linalg.qr(matrix, mode="reduced")

    # Fix the otherwise arbitrary phase/sign of each occupied QR column while
    # preserving ``basis @ triangular == matrix``.
    diagonal = np.diag(triangular)
    phases = np.ones(columns, dtype=matrix.dtype)
    nonzero = np.abs(diagonal) > np.finfo(matrix.real.dtype).eps
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    basis = basis * phases[None, :]
    triangular = phases.conj()[:, None] * triangular

    eps = np.finfo(matrix.real.dtype).eps
    while basis.shape[1] < target:
        # The squared residual of coordinate vector e_j after projection onto
        # the current basis is 1 - ||basis[j, :]||^2.  Prefer the earliest
        # coordinate within roundoff of the largest residual to make ties
        # reproducible across BLAS/LAPACK implementations.
        residuals = np.maximum(
            0.0, 1.0 - np.sum(np.abs(basis) ** 2, axis=1)
        )
        largest = float(np.max(residuals))
        tie = 256.0 * eps * max(1.0, largest)
        candidates = np.flatnonzero(residuals >= largest - tie)
        pivot = int(candidates[0])
        vector = np.zeros(rows, dtype=matrix.dtype)
        vector[pivot] = 1.0
        # Two projections are inexpensive here and suppress loss of
        # orthogonality when the selected coordinate is almost represented.
        for _ in range(2):
            vector -= basis @ (basis.conj().T @ vector)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 64.0 * eps:
            raise np.linalg.LinAlgError(
                "could not construct a stable deterministic QR complement"
            )
        vector /= norm
        basis = np.column_stack((basis, vector))
    return basis, triangular


def _complete_columns(columns, target):
    """Append requested full-QR complement columns without rotating a basis."""
    current = columns.shape[1]
    if target < current or target > columns.shape[0]:
        raise ValueError("completion dimension is outside the available space")
    if target == current:
        return columns.copy()
    complete, _triangular = _partial_full_qr(columns, target)
    return np.column_stack((columns, complete[:, current:target]))


def _left_completion(isometry, target):
    """Extend a left-isometric tensor with full-QR complement columns."""
    dl, current, physical = isometry.shape
    matrix = np.transpose(isometry, (0, 2, 1)).reshape(
        dl * physical, current
    )
    if target > matrix.shape[0]:
        raise ValueError("left completion exceeds the local Hilbert space")
    basis = _complete_columns(matrix, target)
    return np.transpose(basis.reshape(dl, physical, target), (0, 2, 1))


def _right_completion(tensor, target):
    """Extend a right-isometric tensor with full-QR complement rows."""
    current, dr, physical = tensor.shape
    matrix_t = tensor.reshape(current, dr * physical).T
    if target > matrix_t.shape[0]:
        raise ValueError("right completion exceeds the local Hilbert space")
    basis = _complete_columns(matrix_t, target)
    return basis.T.reshape(target, dr, physical)


def _embed_matrix(matrix, size):
    """Embed a bond centre in the leading block of a square matrix."""
    out = np.zeros((size, size), dtype=matrix.dtype)
    rows = min(size, matrix.shape[0])
    columns = min(size, matrix.shape[1])
    out[:rows, :columns] = matrix[:rows, :columns]
    return out


def _adaptive_bond_targets(tensors, mpo, precision, ceiling, expand):
    """Select full-QR tangent-space dimensions for the next one-site sweep.

    For bond ``i`` the convergence function contains the squared norms of
    ``H(i) A_C(i)``, ``K(i) C(i)``, and ``H(i+1) A_C(i+1)``.  The three
    tensors are formed once in the largest local QR-complement space and
    leading blocks give the values for smaller candidates.  The selected
    dimension is the first one whose relative increment is no larger than
    ``precision``.  QR-complement columns are constructed deterministically by
    :func:`_partial_full_qr`, so this test does not inherit a LAPACK-specific
    direction ordering.
    """
    count = len(tensors)
    right_envs = init_right_envs(tensors, mpo)
    boundary = np.ones((1, 1, 1), complex)
    left_envs = [None] * (count + 1)
    left_envs[0] = boundary
    centers = []
    bond_centers = []
    left_isometries = []
    center = tensors[0]
    for site in range(count - 1):
        centers.append(center)
        left, bond = left_qr(center)
        left_isometries.append(left)
        bond_centers.append(bond)
        left_envs[site + 1] = updateleftenv(
            left, mpo[site], left_envs[site]
        )
        center = _einsum_cached(
            "ax,xbs->abs", bond, tensors[site + 1]
        )
    centers.append(center)

    targets = []
    details = []
    for site in range(count - 1):
        current = tensors[site].shape[1]
        if current != tensors[site + 1].shape[0]:
            raise ValueError("adjacent MPS bond dimensions differ")
        if current > ceiling:
            raise ValueError(
                f"current bond dimension {current} exceeds the A1TDVP "
                f"ceiling {ceiling}"
            )
        local_limit = min(
            int(ceiling),
            current + int(expand),
            centers[site].shape[0] * centers[site].shape[2],
            tensors[site + 1].shape[1] * tensors[site + 1].shape[2],
        )
        if local_limit <= current:
            targets.append(current)
            details.append({
                "bond": site,
                "before": current,
                "after": current,
                "relative_increment": 0.0,
            })
            continue

        left = _left_completion(left_isometries[site], local_limit)
        right = _right_completion(tensors[site + 1], local_limit)
        left_expanded = updateleftenv(
            left, mpo[site], left_envs[site]
        )
        right_expanded = updaterightenv(
            right, mpo[site + 1], right_envs[site + 3]
        )

        center_left = _setbond(
            centers[site], centers[site].shape[0], local_limit
        )
        center_right = _setbond(
            centers[site + 1], local_limit, centers[site + 1].shape[1]
        )
        bond = _embed_matrix(bond_centers[site], local_limit)
        action_left = applyH1(
            center_left, mpo[site], left_envs[site], right_expanded
        )
        action_bond = applyH0(bond, left_expanded, right_expanded)
        action_right = applyH1(
            center_right, mpo[site + 1], left_expanded,
            right_envs[site + 3],
        )

        def convergence(
            size,
            action_left=action_left,
            action_bond=action_bond,
            action_right=action_right,
        ):
            values = (
                action_left[:, :size],
                action_bond[:size, :size],
                action_right[:size],
            )
            return float(sum(np.vdot(value, value).real for value in values))

        selected = local_limit
        selected_increment = 0.0
        value = convergence(current)
        for candidate in range(current, local_limit):
            following = convergence(candidate + 1)
            if value == 0.0:
                increment = 0.0 if following == 0.0 else np.inf
            else:
                increment = max(0.0, following / value - 1.0)
            selected_increment = increment
            if increment <= precision:
                selected = candidate
                break
            value = following
        targets.append(selected)
        details.append({
            "bond": site,
            "before": current,
            "after": selected,
            "relative_increment": float(selected_increment),
        })
    return targets, details


def _expand_right_canonical(tensors, targets):
    """Embed an MPS exactly in the selected right-canonical bond spaces."""
    state = [np.asarray(tensor, complex).copy() for tensor in tensors]
    if len(targets) != len(state) - 1:
        raise ValueError("one target dimension is required per MPS bond")
    for site in range(len(state) - 1, 0, -1):
        tensor = state[site]
        old_left, right, physical = tensor.shape
        target = int(targets[site - 1])
        if target < old_left or target > right * physical:
            raise ValueError("requested bond expansion is locally impossible")
        matrix_t = tensor.reshape(old_left, right * physical).T
        basis, triangular = _partial_full_qr(matrix_t, target)
        state[site] = basis.T.reshape(target, right, physical)
        centre = np.zeros((old_left, target), complex)
        centre[:, :old_left] = triangular.T
        state[site - 1] = _einsum_cached(
            "axp,xb->abp", state[site - 1], centre
        )
    norm = np.linalg.norm(state[0])
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("cannot expand a zero or non-finite MPS")
    state[0] /= norm
    return state


def a1tdvp_sweep(
    dt2,
    tensors,
    mpo,
    *,
    trunc_eps=1e-3,
    bond_dim=50,
    expand=DEFAULT_BOND_EXPAND,
    **krylov,
):
    """Advance adaptive one-site TDVP after a full-QR subspace expansion.

    ``trunc_eps`` is the maximum relative increment in the tangent-space
    convergence function; ``bond_dim`` is a strict memory ceiling and
    ``expand`` limits the number of QR-complement directions tested per sweep.
    The time step itself is an ordinary one-site projector-splitting sweep.
    """
    targets, bonds = _adaptive_bond_targets(
        tensors, mpo, float(trunc_eps), int(bond_dim), int(expand)
    )
    state = _expand_right_canonical(tensors, targets)
    state, env = tdvp1sweep(dt2, state, mpo, None, **krylov)
    diagnostic = {
        "bond_dimensions": bonddims(state),
        "precision": float(trunc_eps),
        "bonds": bonds,
    }
    return state, None, env, diagnostic
