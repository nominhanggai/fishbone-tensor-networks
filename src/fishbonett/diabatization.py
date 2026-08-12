"""Boys localization for multi-state diabatization.

Obtain diabatic electronic couplings from adiabatic state/transition dipole
matrices via Jacobi sweeps that maximize the Boys localization function.

Reference: J. Chem. Phys. 129, 244101 (2008).
"""
import itertools as it

import numpy as np
from numpy.linalg import norm


def boys_func(mat_mu):
    """Boys localization functional of a stack of dipole matrices.

    Parameters
    ----------
    mat_mu : ndarray, shape (dim, dim, 3)
        State (diagonal) and transition (off-diagonal) dipole vectors.
    """
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
        theta1 = np.arccos(-F / np.sqrt(F ** 2 + G ** 2))
        theta2 = np.arcsin(G / np.sqrt(F ** 2 + G ** 2))
        t1_l = [theta1 + 2 * k * np.pi for k in range(-2, 3)] + \
               [2 * k * np.pi - theta1 for k in range(-1, 1)]
        t2_l = [theta2 + 2 * k * np.pi for k in range(-2, 3)] + \
               [(2 * k + 1) * np.pi - theta2 for k in range(-1, 1)]
        val_l = [a for a, b in it.product(t1_l, t2_l) if np.abs(a - b) <= 1e-4]
        theta = .25 * val_l[int(np.argmin(np.abs(val_l)))]
        u = np.eye(dim)
        u[i, j] = np.sin(theta)
        u[j, i] = -np.sin(theta)
        u[i, i] = u[j, j] = np.cos(theta)
        mat_mu_after = np.einsum('ij,jkX,kl->ilX', u, mat_mu_after, u.T)
        u_final = u @ u_final
    return u_final, mat_mu_after, boys_func(mat_mu_after), boys_value_0


def diabatize(mat_mu, tol=1e-3, max_sweeps=1000):
    """Iterate :func:`boys_loc` to convergence.

    Returns the accumulated rotation ``u`` and the localized dipole stack.
    """
    dim = mat_mu.shape[0]
    u_final = np.eye(dim)
    boys_value, boys_value_0 = 0.0, 1.0
    for _ in range(max_sweeps):
        if abs(boys_value - boys_value_0) <= tol:
            break
        u_final, mat_mu, boys_value, boys_value_0 = boys_loc(mat_mu, u_final)
    return u_final, mat_mu


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
