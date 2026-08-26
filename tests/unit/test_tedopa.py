"""Unit tests for the measure-adapted TEDOPA bath discretization."""
import numpy as np
from scipy.integrate import trapezoid

from fishbonett.bath.tedopa import (
    rkpw_recurrence, get_vn_squared_tedopa, make_tedopa_discretizer,
)
from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.chain import get_bath_nn_parameters

DOMAIN = (-25.0, 36.0)


def _superohmic_Jb(beta=1.0, wc=5.0, alpha=0.2, s=1.0):
    def Jb(w):
        aw = abs(w)
        if aw < 1e-12:
            return 0.0
        nb = 1.0 / np.expm1(beta * aw)
        j = alpha * aw ** s * wc ** (1 - s) * np.exp(-aw / wc)
        return j * (nb + 1.0) if w > 0 else j * nb
    return Jb


def test_rkpw_matches_analytic_legendre():
    """RKPW recurrence of the constant weight on [-1,1] is the monic Legendre
    recurrence: alpha_k = 0, beta_0 = 2, beta_k = k^2/(4k^2-1)."""
    x, w = np.polynomial.legendre.leggauss(400)  # exact for the low-order coeffs
    alpha, beta = rkpw_recurrence(11, x, w)
    beta_exact = np.array([2.0] + [k * k / (4 * k * k - 1) for k in range(1, 11)])
    np.testing.assert_allclose(alpha, 0.0, atol=1e-10)
    np.testing.assert_allclose(beta, beta_exact, rtol=1e-9, atol=1e-12)


def test_tedopa_sum_rule_and_shape():
    Jb = _superohmic_Jb()
    wg = np.linspace(DOMAIN[0], DOMAIN[1], 20001)
    mass = trapezoid(np.array([Jb(w) for w in wg]), wg)
    f, v = get_vn_squared_tedopa(Jb, 80, DOMAIN, m_per=80)
    assert f.shape == v.shape == (80,)
    assert abs(v.sum() - mass) / abs(mass) < 1e-6


def test_tedopa_beats_legendre_on_correlation():
    """The measure-adapted star reproduces the correlation much more accurately
    than the uniform-measure Legendre star."""
    Jb = _superohmic_Jb()
    wg = np.linspace(DOMAIN[0], DOMAIN[1], 20001)
    Jg = np.array([Jb(w) for w in wg])
    ts = np.linspace(0, 3.0, 15)
    Cex = np.array([trapezoid(Jg * np.exp(-1j * wg * t), wg) for t in ts])

    fo, vo = get_vn_squared_tedopa(Jb, 100, DOMAIN, m_per=100)
    fl, vl = get_vn_squared(Jb, 100, list(DOMAIN))
    Co = np.array([np.sum(vo * np.exp(-1j * fo * t)) for t in ts])
    Cl = np.array([np.sum(vl * np.exp(-1j * fl * t)) for t in ts])

    e_tedopa = np.max(np.abs(Co - Cex))
    e_leg = np.max(np.abs(Cl - Cex))
    assert e_tedopa < 1e-6
    assert e_tedopa < e_leg / 100.0     # at least two orders of magnitude better


def test_tedopa_discretizer_is_dropin_for_chain_mapping():
    Jb = _superohmic_Jb()
    disc = make_tedopa_discretizer(m_per=60)
    w, k = get_bath_nn_parameters(Jb, 30, list(DOMAIN), discretizer=disc)
    assert len(w) == 30 and len(k) == 30
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(k))
    assert k[0] > 0                    # positive system-bath coupling
