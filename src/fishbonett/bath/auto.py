"""Automatic bath discretization defaults.

Two quantities that otherwise have to be guessed can be derived from the spectral
density itself:

* the frequency **domain**, from the reorganization energy
  :math:`\\lambda = \\tfrac{1}{\\pi}\\int_0^\\infty J(\\omega)/\\omega\\,d\\omega` --
  choose the window that captures a target fraction (default 99.9%) of it;
* the number of **modes**, from the interaction-picture chain couplings
  :math:`d_j(t)`.  These form a light-cone along the chain, so a run of length
  ``t_max`` only excites the first ``j_max`` sites.

Both are used as the defaults of :class:`fishbonett.simulate.Bath` when ``domain``
/ ``n_modes`` are left unspecified.
"""
import numpy as np

from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.lanczos import lanczos

__all__ = ["reorganization_energy", "auto_domain", "auto_n_modes"]


def _reorg_profile(J, lo=1e-8, hi=1e10, n=4000):
    """Cumulative reorganization-energy integrand on a **geometric** grid --
    ``(w, cum)`` with ``cum[k] = int_0^{w[k]} J(w')/w' dw'``.  A geometric grid
    resolves the low-frequency mass and a slow high-frequency tail (e.g. the
    ``1/w`` tail of a Drude bath) at the same time, which a linear grid cannot."""
    w = np.geomspace(lo, hi, n)
    with np.errstate(over="ignore", under="ignore", divide="ignore",
                     invalid="ignore"):
        f = np.array([float(J(x)) for x in w]) / w
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        cum = np.concatenate([[0.0],
                              np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(w))])
    return w, cum


def reorganization_energy(J):
    """Reorganization energy ``lambda = (1/pi) int_0^inf J(w)/w dw`` of ``J``."""
    _, cum = _reorg_profile(J)
    return float(cum[-1] / np.pi)


def auto_domain(J, coverage=0.999, signed=False):
    """Frequency window capturing ``coverage`` of the reorganization energy.

    Returns ``(0, w_hi)`` for a zero-temperature bath, or ``(-w_hi, w_hi)`` when
    ``signed`` (a thermofield / T-TEDOPA density lives on both frequency halves)."""
    w, cum = _reorg_profile(J)
    total = cum[-1]
    if total <= 0:
        raise ValueError("reorganization energy is non-positive; set `domain` "
                         "explicitly")
    idx = min(int(np.searchsorted(cum, coverage * total)), len(w) - 1)
    w_hi = float(w[idx])
    return (-w_hi, w_hi) if signed else (0.0, w_hi)


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
    disc = discretizer if discretizer is not None else get_vn_squared
    n_big = n_start
    while True:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            freq, v_sq = disc(sd, n_big, list(domain))
        freq = np.asarray(freq, float)
        Vn = np.sqrt(np.abs(np.asarray(v_sq, float)) / np.pi)
        _, P = lanczos(np.diag(freq), Vn)               # star -> chain (Lanczos)
        coefT = np.ascontiguousarray(P.T)
        dmax = np.zeros(n_big)
        for t in np.linspace(0.0, t_max, n_t):
            dmax = np.maximum(dmax, np.abs(coefT @ (Vn * np.exp(-1j * freq * t))))
        peak = dmax.max()
        sig = np.where(dmax > rel_threshold * peak)[0] if peak > 0 else np.array([0])
        j_max = int(sig.max()) if len(sig) else 0
        n = j_max + 1 + buffer
        if n < n_big or n_big >= n_max:               # front is contained in n_big
            return int(min(n, n_max))
        n_big = min(2 * n_big, n_max)
