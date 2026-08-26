"""Benchmark: measure-adapted TEDOPA star vs Gauss-Legendre star.

Compares how well each N-mode star discretization reproduces the exact bath
correlation function ``C(t) = int J_beta(w) e^{-i w t} dw`` for a few thermalized
spectral densities. The TEDOPA star (Gauss quadrature of the actual measure
J_beta dw) resolves the infrared and sharp peaks that the uniform-measure
Legendre star misses.

Run with:  python benchmarks/bath_discretization.py
"""
import numpy as np
from scipy.integrate import quad

from fishbonett.bath.tedopa import get_vn_squared_tedopa
from fishbonett.bath.legendre import get_vn_squared

DOMAIN = (-25.0, 36.0)
BETA, WC, ALPHA = 1.0, 5.0, 0.2


def make_Jb(s, peak=None):
    def Jbare(w):
        if peak is not None:  # Lorentzian peak
            eta, w0 = 0.3, peak
            return ALPHA * eta * w / ((w0 ** 2 - w ** 2) ** 2 + eta ** 2 * w ** 2)
        return ALPHA * w ** s * WC ** (1 - s) * np.exp(-w / WC)

    def Jb(w):
        aw = abs(w)
        if aw < 1e-12:
            return 0.0
        nb = 1.0 / np.expm1(BETA * aw)
        j = Jbare(aw)
        return j * (nb + 1.0) if w > 0 else j * nb
    return Jb


def main():
    ts = np.linspace(0, 3.0, 60)

    def Cstar(f, v):
        return np.array([np.sum(v * np.exp(-1j * f * t)) for t in ts])

    def Cexact(Jb, pts):
        out = []
        for t in ts:
            re, _ = quad(
                lambda w, t=t: Jb(w) * np.cos(w * t),
                *DOMAIN, limit=200, points=pts,
            )
            im, _ = quad(
                lambda w, t=t: -Jb(w) * np.sin(w * t),
                *DOMAIN, limit=200, points=pts,
            )
            out.append(re + 1j * im)
        return np.array(out)

    print(f"{'bath':12s} {'sum-rule err':>13s} {'C(t): TEDOPA':>15s} {'Legendre':>12s}")
    for name, s, peak in [("super-Ohmic", 1.0, None), ("sub-Ohmic", 0.5, None),
                          ("Lorentzian", 0.5, 2.0)]:
        Jb = make_Jb(s, peak)
        pts = [0.0] + ([peak, -peak] if peak else [])
        mass, _ = quad(Jb, *DOMAIN, limit=400, points=pts)
        eb = (peak, -peak) if peak else ()

        fo, vo = get_vn_squared_tedopa(Jb, 100, DOMAIN, m_per=100, extra_breaks=eb)
        fl, vl = get_vn_squared(Jb, 100, list(DOMAIN))
        Cex = Cexact(Jb, pts)
        e_orth = np.max(np.abs(Cstar(fo, vo) - Cex))
        e_leg = np.max(np.abs(Cstar(fl, vl) - Cex))
        sr = abs(vo.sum() - mass) / abs(mass)
        print(f"{name:12s} {sr:>13.2e} {e_orth:>15.2e} {e_leg:>12.2e}")


if __name__ == "__main__":
    main()
