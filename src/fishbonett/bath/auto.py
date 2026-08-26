"""Automatic bath discretization defaults.

Two quantities that otherwise have to be guessed can be derived from the spectral
density itself:

* the frequency **domain**, from the reorganization energy
  :math:`\\lambda = \\tfrac{1}{\\pi}\\int_0^\\infty J(\\omega)/\\omega\\,d\\omega` --
  choose the window that captures a target fraction (default 99.9%) of it.  At
  finite temperature the two halves of the thermofield density are truncated
  independently, giving an *asymmetric* window whose thermally-suppressed negative
  edge sits closer to zero;
* the number of **modes**, from the interaction-picture chain couplings
  :math:`d_j(t)`.  These form a light-cone along the chain, so a run of length
  ``t_max`` only excites the first ``j_max`` sites.

Both are used as the defaults of :class:`fishbonett.bath.spec.Bath` when ``domain``
/ ``n_modes`` are left unspecified.
"""
import warnings

import numpy as np

from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.lanczos import lanczos

__all__ = ["reorganization_energy", "auto_domain", "auto_n_modes"]


def _density_value(density, frequency):
    """Evaluate a physical positive-frequency density without hiding failures."""
    try:
        value = float(density(float(frequency)))
    except Exception as exc:
        raise ValueError(
            f"spectral density failed at frequency {frequency:.6g}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"spectral density is non-finite at frequency {frequency:.6g}"
        )
    if value < 0:
        raise ValueError(
            f"spectral density is negative at frequency {frequency:.6g}"
        )
    return value


def _discover_log_bounds(J, *, step=0.05, relative_tail=1e-12,
                         tail_span=4.0, max_abs_log=690.0):
    """Find both tails of ``J(exp(x))`` without assuming frequency units.

    In logarithmic coordinates the reorganization integral is simply
    ``integral J(exp(x)) dx``.  Expanding in both directions from an arbitrary
    origin makes a rescaling of the user's frequency unit a translation rather
    than a change of the fixed search window.
    """
    samples = {0: _density_value(J, 1.0)}
    maximum = samples[0]
    below = {-1: 0, 1: 0}
    active = {-1, 1}
    tail_points = max(1, int(np.ceil(tail_span / step)))
    max_steps = int(max_abs_log / step)
    for index in range(1, max_steps + 1):
        for direction in tuple(active):
            frequency = float(np.exp(direction * index * step))
            value = _density_value(J, frequency)
            samples[direction * index] = value
            maximum = max(maximum, value)
            if maximum > 0 and value <= relative_tail * maximum:
                below[direction] += 1
            else:
                below[direction] = 0
            if below[direction] >= tail_points:
                active.remove(direction)
        if not active:
            break
    if maximum <= 0:
        raise ValueError(
            "spectral density has zero mass; provide a non-zero density or an "
            "explicit discrete bath"
        )
    if active:
        raise ValueError(
            "automatic domain search could not resolve the spectral-density "
            "tail; set Bath.domain explicitly"
        )
    indices = np.arange(min(samples), max(samples) + 1)
    log_frequency = indices * step
    frequency = np.exp(log_frequency)
    values = np.array([
        samples.get(int(index), _density_value(J, value))
        for index, value in zip(indices, frequency, strict=True)
    ])
    return frequency, log_frequency, values


def _reorg_profile(J):
    """Scale-adaptive cumulative reorganization integral ``(frequency, cum)``."""
    frequency, log_frequency, values = _discover_log_bounds(J)
    segments = 0.5 * (values[1:] + values[:-1]) * np.diff(log_frequency)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    return frequency, cumulative


def reorganization_energy(J):
    """Reorganization energy ``lambda = (1/pi) int_0^inf J(w)/w dw`` of ``J``."""
    _, cum = _reorg_profile(J)
    return float(cum[-1] / np.pi)


def _tail_cutoff(density, lam, coverage, frequency):
    """Frequency beyond which the reorganization-energy *tail* of ``density`` drops
    below ``(1 - coverage) * lam``, i.e. the modes past the cutoff carry less than
    ``1 - coverage`` of the reorganization energy ``lam``.

    The tail ``(1/pi) int_w^inf density(w')/w' dw'`` converges at high frequency even
    when the density's *total* reorganization energy diverges at ``w -> 0`` (as each
    thermal branch ``J(w) n_beta`` / ``J(w)(n_beta + 1)`` does), so this is the
    well-defined way to place a high-frequency cutoff on a thermalized branch."""
    w = np.asarray(frequency, float)
    log_w = np.log(w)
    values = np.array([_density_value(density, value) for value in w])
    seg = 0.5 * (values[1:] + values[:-1]) * np.diff(log_w)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]]) / np.pi
    below = np.nonzero(tail <= (1.0 - coverage) * lam)[0]
    return float(w[below[0]]) if below.size else float(w[-1])


def auto_domain(J, coverage=0.999, beta=None):
    """Frequency window capturing ``coverage`` of the reorganization energy.

    At zero temperature (``beta is None``) returns ``(0, w_hi)`` from the ordinary
    reorganization energy of ``J``.  At finite temperature the thermalized
    (thermofield / T-TEDOPA) density lives on **both** frequency halves,
    ``J_beta(+w) = J(w)(n_beta + 1)`` and ``J_beta(-w) = J(w) n_beta``; each half is
    truncated by its *own* reorganization-energy tail, giving an **asymmetric**
    ``(-w_lo, w_hi)`` whose negative edge is closer to zero because the negative
    branch is thermally suppressed."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    if beta is not None and (not np.isfinite(beta) or beta <= 0):
        raise ValueError("beta must be finite and positive")
    frequency, cumulative = _reorg_profile(J)
    lam = float(cumulative[-1] / np.pi)
    if lam <= 0:
        raise ValueError("reorganization energy is non-positive; set `domain` "
                         "explicitly")
    if beta is None:
        return (0.0, _tail_cutoff(J, lam, coverage, frequency))
    def nb(x):
        scaled = beta * x
        return 0.0 if scaled > 700.0 else 1.0 / np.expm1(scaled)
    w_hi = _tail_cutoff(
        lambda x: J(x) * (nb(x) + 1.0), lam, coverage, frequency
    )
    w_lo = _tail_cutoff(
        lambda x: J(x) * nb(x), lam, coverage, frequency
    )
    return (-w_lo, w_hi)


def auto_n_modes(sd, domain, t_max, *, buffer=10, rel_threshold=1e-3, n_t=80,
                 discretizer=None, n_start=128, n_max=1024):
    """Number of bath modes needed to propagate to ``t_max``.

    The interaction-picture chain couplings ``d_j(t)`` form a light-cone: the
    coupling to chain site ``j`` stays negligible until the excitation front
    reaches it, so a run of length ``t_max`` only excites the first ``j_max``
    sites.  Build a large trial chain, track ``max_t |d_j(t)|`` over
    ``t in [0, t_max]``, take the furthest site above ``rel_threshold`` of the peak
    and add ``buffer`` sites of headroom.  Derived in the interaction-picture chain
    gauge, but the count bounds the modes needed in every representation.
    """
    if not np.isfinite(t_max) or t_max < 0:
        raise ValueError("t_max must be finite and non-negative")
    if (not isinstance(n_start, (int, np.integer)) or n_start < 1
            or not isinstance(n_max, (int, np.integer)) or n_max < n_start):
        raise ValueError("n_start and n_max must be positive with n_start <= n_max")
    if buffer < 0 or n_t < 2 or not 0 < rel_threshold < 1:
        raise ValueError(
            "buffer must be non-negative, n_t at least 2, and rel_threshold "
            "strictly between zero and one"
        )
    disc = discretizer if discretizer is not None else get_vn_squared
    n_big = n_start
    while True:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            freq, v_sq = disc(sd, n_big, list(domain))
        freq = np.asarray(freq, float)
        v_sq = np.asarray(v_sq, float)
        if (not np.all(np.isfinite(freq)) or not np.all(np.isfinite(v_sq))
                or np.any(v_sq < 0)):
            raise ValueError(
                "bath discretization produced non-finite frequencies or "
                "negative/non-finite coupling weights"
            )
        Vn = np.sqrt(v_sq / np.pi)
        # A large Gaussian star can become numerically rank deficient only at
        # the far end of its Krylov basis.  Automatic resolution needs the
        # light-cone prefix, not an artificial completion of every trial mode,
        # so retain the stable prefix and enlarge the probe below if the front
        # reaches its edge.  Production chain construction remains strict.
        _, P = lanczos(
            np.diag(freq), Vn, allow_early_termination=True
        )                                                # star -> chain (Lanczos)
        coefT = np.ascontiguousarray(P.T)
        resolved_probe = coefT.shape[0]
        dmax = np.zeros(resolved_probe)
        for t in np.linspace(0.0, t_max, n_t):
            dmax = np.maximum(dmax, np.abs(coefT @ (Vn * np.exp(-1j * freq * t))))
        peak = dmax.max()
        sig = np.where(dmax > rel_threshold * peak)[0] if peak > 0 else np.array([0])
        j_max = int(sig.max()) if len(sig) else 0
        n = j_max + 1 + buffer
        if n < resolved_probe:                         # front is contained in probe
            return int(n)
        if n_big >= n_max:
            warnings.warn(
                f"automatic bath resolution reached n_max={n_max} before the "
                "interaction light cone was contained; set n_modes explicitly "
                "or increase the resolution limit",
                RuntimeWarning,
                stacklevel=2,
            )
            return int(resolved_probe)
        n_big = min(2 * n_big, n_max)
