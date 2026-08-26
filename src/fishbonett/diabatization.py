"""Boys localization for multi-state diabatization.

Obtain diabatic electronic couplings from adiabatic state/transition dipole
matrices via Jacobi sweeps that maximize the Boys localization function.

Reference: J. Chem. Phys. 129, 244101 (2008).
"""
import itertools as it

import numpy as np
from numpy.linalg import norm

from fishbonett.contract import _einsum_cached


def boys_func(mat_mu):
    """Boys localization functional of a stack of dipole matrices.

    Parameters
    ----------
    mat_mu : ndarray, shape (dim, dim, 3)
        State (diagonal) and transition (off-diagonal) dipole vectors.
    """
    mat_mu = np.asarray(mat_mu)
    if (mat_mu.ndim != 3 or mat_mu.shape[0] == 0
            or mat_mu.shape[0] != mat_mu.shape[1]
            or not np.all(np.isfinite(mat_mu))):
        raise ValueError(
            "mat_mu must be a finite array with shape (n_states, n_states, n_components)"
        )
    dim = mat_mu.shape[0]
    return sum(norm(mat_mu[i, i, :] - mat_mu[j, j, :]) ** 2
               for i, j in it.combinations(range(dim), 2))


def boys_loc(mat_mu, u_final):
    """One Jacobi sweep maximizing the Boys functional.

    Returns
    -------
    (u_final, mat_mu_after, boys_value, boys_value_0)
        The accumulated rotation, the rotated dipole stack, and the Boys
        functional after and before the sweep.
    """
    dim = mat_mu.shape[0]
    mat_mu_after = mat_mu.copy()
    boys_value_0 = boys_func(mat_mu)
    for i, j in it.combinations(range(dim), 2):
        mu_ij = mat_mu_after[i, j]
        mu_ii = mat_mu_after[i, i]
        mu_jj = mat_mu_after[j, j]
        F = norm(mu_ij) ** 2 - .25 * norm(mu_ii - mu_jj) ** 2
        G = mu_ij @ (mu_ii - mu_jj)
        if np.hypot(F, G) <= np.finfo(float).eps:
            # The pair is already completely degenerate in the localization
            # functional; every angle is equivalent, so leave it unchanged.
            continue
        theta = 0.25 * np.arctan2(G, -F)
        u = np.eye(dim)
        u[i, j] = np.sin(theta)
        u[j, i] = -np.sin(theta)
        u[i, i] = u[j, j] = np.cos(theta)
        mat_mu_after = _einsum_cached('ij,jkX,kl->ilX', u, mat_mu_after, u.T)
        u_final = u @ u_final
    return u_final, mat_mu_after, boys_func(mat_mu_after), boys_value_0


def diabatize(mat_mu, tol=1e-3, max_sweeps=1000):
    """Iterate :func:`boys_loc` to convergence.

    Returns the accumulated rotation ``u`` and the localized dipole stack.
    """
    raw = np.asarray(mat_mu)
    if np.iscomplexobj(raw) and np.any(np.abs(raw.imag) > 1e-14):
        raise ValueError("Boys localization currently requires real dipole matrices")
    mat_mu = np.asarray(raw.real, float)
    boys_func(mat_mu)  # validates shape and finiteness
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    if (not isinstance(max_sweeps, (int, np.integer))
            or isinstance(max_sweeps, (bool, np.bool_)) or max_sweeps < 1):
        raise ValueError("max_sweeps must be a positive integer")
    dim = mat_mu.shape[0]
    u_final = np.eye(dim)
    for _ in range(max_sweeps):
        u_final, mat_mu, boys_value, boys_value_0 = boys_loc(mat_mu, u_final)
        if abs(boys_value - boys_value_0) <= tol:
            return u_final, mat_mu
    raise RuntimeError(
        f"Boys localization did not converge within {max_sweeps} sweeps"
    )


if __name__ == "__main__":
    # Dipole moments (a.u.) for a 2-state model at the LE geometry; the diabatic
    # coupling is the off-diagonal of u H u^T in the localized basis.
    mu_mat_2_config1 = np.array([
        [[-1.1410, -0.9247, -0.6088], [-2.84510477626, -0.715646213116, -0.409709899928]],
        [[-2.84510477626, -0.715646213116, -0.409709899928], [-25.3102, -6.2400, -3.7840]],
    ])
    u, _ = diabatize(mu_mat_2_config1)
    H = np.diag([0, -0.1288])
    print("Diabatic Hamiltonian (a.u.):\n", u @ H @ u.T)
    print("Diabatic Hamiltonian (cm^-1):\n", repr(u @ H @ u.T * 8065.54429))
