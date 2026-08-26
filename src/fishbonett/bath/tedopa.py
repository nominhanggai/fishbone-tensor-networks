r"""Measure-adapted Gaussian discretization for TEDOPA-style chain mappings.

The implementation follows two standard numerical facts.  First, a positive
continuous measure can be represented accurately by a sufficiently fine
composite Gaussian rule.  Second, Lanczos tridiagonalization of the diagonal
node matrix, started from the square roots of those weights, produces the Jacobi
matrix of the measure.  Its eigenvalues and first eigenvector components are the
nodes and weights of the desired Gaussian rule (Golub--Welsch).

No external ORTHPOL implementation is required.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

__all__ = [
    "rkpw_recurrence", "composite_measure_quad",
    "get_vn_squared_tedopa", "make_tedopa_discretizer",
]


def _density_values(density, nodes):
    try:
        values = np.asarray(density(nodes), float)
        if values.shape == nodes.shape:
            return values
    except (TypeError, ValueError):
        pass
    return np.fromiter((float(density(float(x))) for x in nodes),
                       dtype=float, count=len(nodes))


def _geometric_points(limit, smallest, ratio):
    """Positive interior points from ``smallest`` out to ``limit``."""
    if limit <= 0:
        return []
    values = [min(float(smallest), float(limit))]
    while values[-1] < limit:
        nxt = min(limit, values[-1] * ratio)
        if nxt == values[-1]:
            break
        values.append(nxt)
    return values


def composite_measure_quad(Jb, domain, m_per=60, x0=1e-8, ratio=1.7,
                           extra_breaks=()):
    r"""Approximate ``Jb(w) dw`` by a positive composite Gauss--Legendre rule.

    Intervals are geometrically refined toward zero, which resolves thermalized
    sub-Ohmic densities without forcing the entire frequency window onto an
    extremely fine uniform grid.  Explicit breakpoints can be supplied around
    known sharp spectral features.
    """
    left, right = map(float, domain)
    if not left < right:
        raise ValueError("domain must satisfy left < right")
    if int(m_per) < 2:
        raise ValueError("m_per must be at least 2")
    if x0 <= 0 or ratio <= 1:
        raise ValueError("x0 must be positive and ratio must exceed one")

    points = {left, right, 0.0}
    points.update(_geometric_points(max(right, 0.0), x0, ratio))
    points.update(-x for x in _geometric_points(max(-left, 0.0), x0, ratio))
    for value in extra_breaks:
        value = float(value)
        if left < value < right:
            points.add(value)
            # Add scale-local guards without assuming the feature is centred at
            # a positive frequency.
            scale = max(abs(value), x0)
            for fraction in (0.05, 0.15, 0.4):
                points.add(max(left, value - fraction * scale))
                points.add(min(right, value + fraction * scale))
    cuts = np.array(sorted(x for x in points if left <= x <= right))

    base_x, base_w = np.polynomial.legendre.leggauss(int(m_per))
    node_blocks, weight_blocks = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        if hi <= lo:
            continue
        half = 0.5 * (hi - lo)
        nodes = 0.5 * (hi + lo) + half * base_x
        density = _density_values(Jb, nodes)
        if not np.all(np.isfinite(density)):
            raise ValueError("spectral density returned non-finite values")
        tolerance = 1e-13 * max(1.0, np.max(np.abs(density), initial=0.0))
        if np.any(density < -tolerance):
            raise ValueError("measure-adapted discretization requires J >= 0")
        weights = half * base_w * np.maximum(density, 0.0)
        keep = weights > np.finfo(float).tiny
        node_blocks.append(nodes[keep])
        weight_blocks.append(weights[keep])
    if not node_blocks or not any(len(block) for block in node_blocks):
        raise ValueError("spectral density has zero mass on the domain")
    nodes = np.concatenate(node_blocks)
    weights = np.concatenate(weight_blocks)
    order = np.argsort(nodes)
    return nodes[order], weights[order]


def rkpw_recurrence(n, x, w):
    r"""Jacobi recurrence coefficients of a positive discrete measure.

    This is a fully reorthogonalized symmetric Lanczos process for
    ``diag(x)``.  ``beta[0]`` is the total mass and ``beta[k]`` for ``k > 0`` is
    the square of the preceding Jacobi off-diagonal, matching the recurrence
    convention used by the chain mapper.
    """
    nodes = np.asarray(x, float).reshape(-1)
    weights = np.asarray(w, float).reshape(-1)
    n = int(n)
    if nodes.shape != weights.shape:
        raise ValueError("x and w must have the same shape")
    if n < 1 or n > len(nodes):
        raise ValueError("n must be between one and the number of nodes")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and non-negative")
    mass = float(np.sum(weights))
    if not mass > 0:
        raise ValueError("the measure must have positive mass")

    alpha = np.empty(n, float)
    beta = np.zeros(n, float)
    beta[0] = mass
    basis = np.empty((n, len(nodes)), float)
    current = np.sqrt(weights / mass)
    previous = np.zeros_like(current)
    offdiag = 0.0
    for order in range(n):
        basis[order] = current
        residual = nodes * current - offdiag * previous
        alpha[order] = np.dot(current, residual)
        residual -= alpha[order] * current
        # Two modified-Gram--Schmidt passes keep high-order recurrences stable
        # even when the fine measure spans many decades in weight.
        for _ in range(2):
            projection = basis[:order + 1] @ residual
            residual -= projection @ basis[:order + 1]
        if order == n - 1:
            break
        next_offdiag = np.linalg.norm(residual)
        if next_offdiag <= 32 * np.finfo(float).eps * max(
                1.0, np.max(np.abs(nodes))):
            raise ValueError("measure supports fewer than n orthogonal polynomials")
        beta[order + 1] = next_offdiag ** 2
        previous, current = current, residual / next_offdiag
        offdiag = next_offdiag
    return alpha, beta


def get_vn_squared_tedopa(Jb, n, domain, m_per=60, ratio=1.7,
                          extra_breaks=()):
    r"""Return the ``n``-point Gaussian nodes and weights of ``Jb(w) dw``."""
    nodes, weights = composite_measure_quad(
        Jb, domain, m_per=m_per, ratio=ratio, extra_breaks=extra_breaks)
    diagonal, beta = rkpw_recurrence(n, nodes, weights)
    values, vectors = eigh_tridiagonal(diagonal, np.sqrt(beta[1:]))
    gaussian_weights = beta[0] * np.square(vectors[0])
    return values, gaussian_weights


def make_tedopa_discretizer(m_per=60, ratio=1.7, extra_breaks=()):
    """Create a chain-mapper-compatible measure-adapted discretizer."""
    def discretize(sd, n, domain):
        return get_vn_squared_tedopa(
            sd, n, domain, m_per=m_per, ratio=ratio,
            extra_breaks=extra_breaks)
    return discretize
