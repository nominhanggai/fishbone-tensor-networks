"""Projector-splitting sweeps for matrix-product states.

The one-site and two-site integrators share one environment convention and one
palindromic traversal.  Dynamic TDVP uses the two-site tangent space as its bond
growth mechanism, then applies the requested adaptive truncation threshold.
"""
import numpy as np

from fishbonett.contract import _einsum_cached
#: Schmidt directions kept beyond what ``trunc_eps`` admits, so a two-site
#: bond can grow.  Two is enough to bootstrap a product state and costs a bond
#: of ``rank + 2``; raise it if a run appears stuck at a small bond.
DEFAULT_BOND_EXPAND = 2
from fishbonett.evolve._tdvp_kernels import (
    SZ, _setbond, applyH1, evolveAC, evolveC, expmv_lanczos,
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


def tdvp1sweep_dynamic(dt2, tensors, mpo, Afull, FRs, *, prec=1e-3,
                       Dlim=50, Dplusmax=None, expand=DEFAULT_BOND_EXPAND,
                       **krylov):
    """Adaptive TDVP using a two-site tangent space and SVD rank selection.

    ``prec`` is this integrator's truncation threshold and ``expand`` is the
    bond-expansion allowance; see :func:`_split2`. A positive allowance lets a
    product-state input grow new bond sectors.
    """
    threshold = float(prec)
    state, env = tdvp2sweep(
        dt2, tensors, mpo, Dlim, threshold, None, expand=expand, **krylov)
    diagnostic = {"bond_dimensions": bonddims(state), "threshold": threshold}
    return state, None, env, diagnostic
