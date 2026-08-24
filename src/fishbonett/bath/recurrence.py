r"""Orthogonal-polynomial recurrence coefficients for chain mappings.

For a spectral density :math:`J(\omega)` on a finite frequency domain, the
chain mapping uses the monic three-term recurrence coefficients
:math:`(\alpha_k, \beta_k)` of the positive measure
:math:`d\mu(\omega)=J(\omega)d\omega/\pi`.  The implementation discretizes that
measure and obtains its Jacobi matrix by Lanczos tridiagonalization.
"""
import numpy as np

from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.lanczos import lanczos

__all__ = ["recurrence_coefficients"]


def recurrence_coefficients(spectral_density, n_modes, domain, *,
                            discretizer=None):
    r"""Return recurrence coefficients for ``spectral_density``.

    Parameters
    ----------
    spectral_density : callable
        Non-negative spectral density :math:`J(\omega)`.
    n_modes : int
        Number of recurrence coefficients and resulting chain modes.
    domain : (float, float)
        Finite frequency interval on which the density is represented.
    discretizer : callable, optional
        Callable ``(density, n_modes, domain) -> (nodes, weights)``. The
        Gauss--Legendre star discretization is used by default.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``alpha`` contains the chain on-site energies. ``beta[0]`` is the total
        measure (the squared system-bath coupling), while ``beta[1:]`` contains
        the squared nearest-neighbour hoppings.
    """
    if (not isinstance(n_modes, (int, np.integer))
            or isinstance(n_modes, (bool, np.bool_)) or n_modes < 1):
        raise ValueError("n_modes must be a positive integer")
    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError("domain must contain two finite values")
    if not bounds[0] < bounds[1]:
        raise ValueError("domain must satisfy left < right")

    def measure(frequency):
        return spectral_density(frequency) / np.pi

    disc = get_vn_squared if discretizer is None else discretizer
    nodes, masses = disc(measure, int(n_modes), bounds.tolist())
    nodes = np.asarray(nodes, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if nodes.shape != (n_modes,) or masses.shape != (n_modes,):
        raise ValueError("discretizer must return one node and weight per mode")
    if (not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(masses))
            or np.any(masses < 0) or not np.sum(masses) > 0):
        raise ValueError("discretized measure must be finite and have positive mass")

    tri, _ = lanczos(np.diag(nodes), np.sqrt(masses))
    alpha = np.diagonal(tri).copy()
    beta = np.empty(int(n_modes), dtype=float)
    beta[0] = float(np.sum(masses))
    beta[1:] = np.diagonal(tri, -1) ** 2
    return alpha, beta
