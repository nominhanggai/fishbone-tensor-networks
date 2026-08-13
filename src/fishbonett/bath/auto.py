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


def _tail_cutoff(density, lam, coverage, lo=1e-8, hi=1e10, n=4000):
    """Frequency beyond which the reorganization-energy *tail* of ``density`` drops
    below ``(1 - coverage) * lam``, i.e. the modes past the cutoff carry less than
    ``1 - coverage`` of the reorganization energy ``lam``.

    The tail ``(1/pi) int_w^inf density(w')/w' dw'`` converges at high frequency even
    when the density's *total* reorganization energy diverges at ``w -> 0`` (as each
    thermal branch ``J(w) n_beta`` / ``J(w)(n_beta + 1)`` does), so this is the
    well-defined way to place a high-frequency cutoff on a thermalized branch."""
    w = np.geomspace(lo, hi, n)
    with np.errstate(over="ignore", under="ignore", divide="ignore",
                     invalid="ignore"):
        f = np.array([float(density(x)) for x in w]) / w
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        seg = 0.5 * (f[1:] + f[:-1]) * np.diff(w)             # per-interval reorg energy
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
    lam = reorganization_energy(J)
    if lam <= 0:
        raise ValueError("reorganization energy is non-positive; set `domain` "
                         "explicitly")
    if beta is None:
        return (0.0, _tail_cutoff(J, lam, coverage))
    nb = lambda x: 1.0 / np.expm1(beta * x)
    w_hi = _tail_cutoff(lambda x: J(x) * (nb(x) + 1.0), lam, coverage)   # J_beta(+w)
    w_lo = _tail_cutoff(lambda x: J(x) * nb(x), lam, coverage)           # J_beta(-w)
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
