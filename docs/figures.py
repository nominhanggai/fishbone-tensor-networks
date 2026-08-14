"""Regenerate the documentation figures.

The figures are **not** committed: they are build artefacts, produced here and
written to ``docs/img/`` (gitignored).  ``docs/conf.py`` calls :func:`build_all`
at the start of every Sphinx build, so a plain ``sphinx-build`` -- locally, in CI
or on Read the Docs -- always renders against freshly computed data.  Run it
directly to refresh them by hand::

    python docs/figures.py

Every figure here illustrates :doc:`bath`: that the automatic bath discretization
(reorganization-energy ``domain`` + light-cone ``n_modes``) reproduces the bath
correlation function, and that degrading either choice breaks it.
"""
from pathlib import Path

import numpy as np

IMG = Path(__file__).resolve().parent / "img"

# Every figure compares a discretized bath against an exact correlation function.
T_MAX = 4.0
_TS = np.linspace(0.0, T_MAX, 400)


def _mpl():
    """Import matplotlib with a headless backend (CI has no display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _c_disc(J, domain, n_modes, ts, beta=None):
    """``C_disc(t) = sum_k g_k^2 e^{-i w_k t}`` for a Gauss-Legendre star.

    ``g_k^2 = J(w_k) w_k / pi`` are the Gauss couplings; at finite temperature the
    thermalized density on a signed domain already carries detailed balance, so the
    same formula applies.
    """
    from fishbonett.bath.legendre import get_vn_squared
    freq, v_sq = get_vn_squared(J, n_modes, list(domain))
    return (np.asarray(v_sq)[None, :] / np.pi
            * np.exp(-1j * np.outer(ts, freq))).sum(axis=1)


def _panel(ax, ts, exact, curves, title):
    """Real/imaginary parts of ``exact`` (lines) with discretizations (markers)."""
    ax.plot(ts, exact.real, "-", color="#4C6EF5", lw=2.4, label=r"exact  Re $C(t)$")
    ax.plot(ts, exact.imag, "-", color="#E8590C", lw=2.4, label=r"exact  Im $C(t)$")
    sl = slice(None, None, max(1, len(ts) // 28))
    ax.plot(ts[sl], curves["auto"].real[sl], "o", ms=5, mfc="none",
            color="#4C6EF5", label="auto (Re)")
    ax.plot(ts[sl], curves["auto"].imag[sl], "s", ms=5, mfc="none",
            color="#E8590C", label="auto (Im)")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("$C(t)$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", representationon=False, fontsize=8, ncol=2)


def _error_inset(ax, ts, exact, curves, loc=(0.44, 0.44, 0.52, 0.36)):
    """Inset: relative error of each discretization against the exact curve."""
    ins = ax.inset_axes(loc)
    scale = abs(exact[0])
    style = {"auto": ("#2B8A3E", "-", "auto"),
             "few": ("#868e96", "--", "too few modes"),
             "narrow": ("#C92A2A", ":", "domain too narrow")}
    for key, (color, ls, label) in style.items():
        if key in curves:
            ins.semilogy(ts, np.abs(curves[key] - exact) / scale, ls,
                         color=color, lw=1.4, label=label)
    ins.set_ylim(1e-4, 2.0)
    ins.set_xlabel("$t$", fontsize=7)
    ins.set_ylabel("rel. error", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.grid(alpha=0.25, which="both")
    ins.legend(fontsize=6, representationon=False, loc="lower right")
    return ins


def bath_correlation(path=None):
    """T = 0 Ohmic bath: the automatic discretization vs two degraded ones."""
    from fishbonett import Bath
    plt = _mpl()
    eta, wc = 0.2, 5.0
    J = lambda w: eta * w * np.exp(-w / wc)
    exact = (eta / np.pi) / (1 / wc + 1j * _TS) ** 2
    bath = Bath(J=J, phys_dim=10).resolved(T_MAX)
    curves = {
        "auto": _c_disc(J, bath.domain, bath.n_modes, _TS),
        "few": _c_disc(J, bath.domain, 20, _TS),
        "narrow": _c_disc(J, (0.0, 10.0), bath.n_modes, _TS),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _panel(ax, _TS, exact, curves,
           rf"Ohmic bath, $T=0$  (auto: domain $\approx$ "
           rf"{tuple(round(float(x), 1) for x in bath.domain)}, "
           rf"{bath.n_modes} modes)")
    _error_inset(ax, _TS, exact, curves)
    fig.tight_layout()
    out = Path(path or IMG / "bath_correlation.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def bath_correlation_finite_t(path=None):
    """Finite temperature: the asymmetric signed (thermofield) domain."""
    from fishbonett import Bath, thermalize
    from scipy.integrate import quad
    plt = _mpl()
    eta, wc, kT = 0.2, 5.0, 1.0
    beta = 1.0 / kT
    J = lambda w: eta * w * np.exp(-w / wc)

    def exact_c(t):                       # detailed-balance correlation function
        re = quad(lambda w: J(w) / np.pi / np.tanh(beta * w / 2) * np.cos(w * t),
                  0, 40 * wc, limit=400)[0]
        im = -quad(lambda w: J(w) / np.pi * np.sin(w * t), 0, 40 * wc, limit=400)[0]
        return re + 1j * im

    exact = np.array([exact_c(t) for t in _TS])
    bath = Bath(J=J, temperature=kT, phys_dim=10).resolved(T_MAX)
    Jb = thermalize(J, beta)
    curves = {
        "auto": _c_disc(Jb, bath.domain, bath.n_modes, _TS),
        "few": _c_disc(Jb, bath.domain, 20, _TS),
        "narrow": _c_disc(Jb, (-1.0, 10.0), bath.n_modes, _TS),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _panel(ax, _TS, exact, curves,
           rf"Ohmic bath, $k_BT={kT}$  (auto signed domain "
           rf"{tuple(round(float(x), 1) for x in bath.domain)}, "
           rf"{bath.n_modes} modes)")
    _error_inset(ax, _TS, exact, curves)
    fig.tight_layout()
    out = Path(path or IMG / "bath_correlation_finiteT.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def bath_structured(path=None):
    """A structured density (Ohmic background + two vibrational peaks)."""
    from fishbonett import Bath
    from fishbonett.bath.legendre import get_vn_squared
    from scipy.integrate import quad
    plt = _mpl()

    def J(w):
        w = np.asarray(w, float)
        out = 0.05 * w * np.exp(-w / 2.5)
        for lam, gam, om in [(0.6, 1.2, 6.0), (0.5, 1.0, 13.0)]:
            out = out + (2 * lam * gam * om ** 2 * w
                         / ((om ** 2 - w ** 2) ** 2 + gam ** 2 * w ** 2))
        return out

    exact = np.array([quad(lambda w: J(w) / np.pi * np.cos(w * t), 0, 60,
                           limit=600)[0]
                      - 1j * quad(lambda w: J(w) / np.pi * np.sin(w * t), 0, 60,
                                  limit=600)[0] for t in _TS])
    bath = Bath(J=J, phys_dim=10).resolved(T_MAX)
    curves = {"auto": _c_disc(J, bath.domain, bath.n_modes, _TS)}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    grid = np.linspace(1e-3, float(bath.domain[1]) * 1.05, 900)
    a1.plot(grid, J(grid), "-", color="#4C6EF5", lw=2.0, label=r"$J(\omega)$")
    freq, v_sq = get_vn_squared(J, bath.n_modes, list(bath.domain))
    a1.plot(freq, J(np.asarray(freq)), "o", ms=3, color="#E8590C",
            label=f"{bath.n_modes} star modes")
    a1.axvline(float(bath.domain[1]), color="#868e96", ls="--", lw=1.2,
               label=rf"auto $\omega_{{hi}}={float(bath.domain[1]):.1f}$")
    a1.set_xlabel(r"$\omega$"); a1.set_ylabel(r"$J(\omega)$")
    a1.set_title("structured spectral density")
    a1.legend(representationon=False, fontsize=8); a1.grid(alpha=0.25)

    _panel(a2, _TS, exact, curves, "correlation function")
    _error_inset(a2, _TS, exact, curves, loc=(0.46, 0.60, 0.50, 0.34))
    fig.tight_layout()
    out = Path(path or IMG / "bath_structured.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


FIGURES = (bath_correlation, bath_correlation_finite_t, bath_structured)


def build_all(force=False):
    """Generate every figure into ``docs/img``.

    Skips a figure whose file already exists unless ``force`` is set, so repeated
    incremental Sphinx builds stay fast.  Failures are reported but never abort the
    documentation build.
    """
    IMG.mkdir(parents=True, exist_ok=True)
    written = []
    for fn in FIGURES:
        target = IMG / {"bath_correlation": "bath_correlation.png",
                        "bath_correlation_finite_t": "bath_correlation_finiteT.png",
                        "bath_structured": "bath_structured.png"}[fn.__name__]
        if target.exists() and not force:
            continue
        try:
            written.append(fn())
        except Exception as exc:                       # pragma: no cover
            print(f"[docs/figures] WARNING: {fn.__name__} failed: {exc}")
    return written


if __name__ == "__main__":
    for p in build_all(force=True):
        print("wrote", p)
