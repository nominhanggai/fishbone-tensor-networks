"""Numerical forms of the package's open-system conventions.

The package uses ``hbar = 1`` and

``J(omega) = pi * sum_k |g_k|**2 delta(omega - omega_k)``.

Consequently a quadrature node with weight ``q_k`` has
``g_k**2 = J(omega_k) q_k / pi`` and the reorganization energy is
``lambda = integral J(omega) / (pi * omega) d omega``.  Keeping the small
numerical helpers here prevents representation implementations from silently choosing a
different factor of ``pi`` or a different zero-frequency limit.
"""
import numpy as np
from scipy.integrate import trapezoid

__all__ = [
    "integrated_free_phase", "reorganization_energy", "star_coupling_squared",
]


def star_coupling_squared(density, frequency, quadrature_weight):
    """Squared discrete coupling in fishbonett's spectral-density convention."""
    return np.asarray(density(frequency)) * quadrature_weight / np.pi


def integrated_free_phase(frequency, t, dt):
    """Return ``integral_t**(t+dt) exp(-i frequency s) ds`` stably.

    The midpoint/sinc form has the exact ``dt`` limit at zero frequency and
    avoids the cancellation in ``(exp(-i*w*dt) - 1) / (-i*w)`` for small ``w``.
    Scalars and arrays are both supported.
    """
    frequency = np.asarray(frequency, dtype=float)
    value = (dt * np.exp(-1j * frequency * (t + 0.5 * dt))
             * np.sinc(frequency * dt / (2.0 * np.pi)))
    return value.item() if value.ndim == 0 else value


def reorganization_energy(density, domain, *, points=4001):
    """Numerically evaluate ``integral J(w)/(pi*w) dw`` over ``domain``.

    A removable value at exactly zero is omitted.  Baths for which the integral
    is genuinely infrared divergent are not valid inputs to a polaron mapping.
    """
    lo, hi = domain
    frequency = np.linspace(lo, hi, points)
    values = np.zeros_like(frequency)
    mask = np.abs(frequency) > 1e-12
    values[mask] = np.asarray(
        [density(value) for value in frequency[mask]], dtype=float
    ) / frequency[mask]
    if np.any(~mask):
        probe = max(abs(hi - lo) * 1e-8, 1e-12)
        if lo < 0.0 < hi:
            zero_limit = 0.5 * (
                density(probe) / probe + density(-probe) / (-probe))
        elif hi > 0.0:
            zero_limit = density(probe) / probe
        else:
            zero_limit = density(-probe) / (-probe)
        values[~mask] = zero_limit
    return float(trapezoid(values, frequency) / np.pi)
