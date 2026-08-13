"""Unit tests for the orthpol-free bath discretization / chain mapping."""
import numpy as np
import pytest

from fishbonett.bath.recurrence import recurrenceCoefficients


def test_recurrence_matches_analytic_legendre():
    """Constant J = pi on [-1, 1] gives h^2(x) = 1, whose monic orthogonal
    polynomials are the Legendre polynomials with the exact recurrence
    alpha_k = 0, beta_0 = 2, beta_k = k^2 / (4 k^2 - 1)."""
    alpha, beta = recurrenceCoefficients(10, lb=-1.0, rb=1.0, j=lambda w: np.pi, g=1)
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    beta_exact = np.array([2.0] + [k * k / (4 * k * k - 1) for k in range(1, len(beta))])

    assert alpha.shape == beta.shape == (11,)
    np.testing.assert_allclose(alpha, 0.0, atol=1e-10)
    np.testing.assert_allclose(beta, beta_exact, rtol=1e-9, atol=1e-12)


def test_recurrence_needs_no_orthpol():
    import importlib.util
    assert importlib.util.find_spec("orthpol") is None
    # The call path must succeed regardless.
    alpha, beta = recurrenceCoefficients(4, lb=0.0, rb=10.0,
                                         j=lambda w: w * np.exp(-w / 5.0), g=1)
    assert len(alpha) == len(beta) == 5
    assert np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta))
    assert beta[0] > 0  # zeroth moment (system-bath coupling squared) is positive


def test_continuous_bath_driver_builds_with_the_default_quadrature():
    """The interaction-picture builder chain-maps via the plain Gauss-Legendre
    get_coupling (no measure-adapted discretizer) and diagonalises the result."""
    from fishbonett.frames.interaction_picture import SimpleSysBathIP

    n_boson = 4
    eth = SimpleSysBathIP([2] + [6] * n_boson,
                       h_sys=10.0 * np.array([[0.0, 1.0], [1.0, 0.0]]),
                       coupling=np.diag([1.0, -1.0]),
                       sd=lambda w: 0.5 * w * np.exp(-w / 10.0),
                       domain=[0.0, 50.0], ncap=200).build()

    assert len(eth.w_list) == n_boson
    assert len(eth.k_list) == n_boson
    assert np.all(np.isfinite(eth.w_list))
    assert np.all(np.isfinite(eth.k_list))
    freq, coef = eth.freq, eth.coef
    assert len(freq) == n_boson
    # eigenvectors of the (real symmetric) chain are orthonormal
    np.testing.assert_allclose(coef.T @ coef, np.eye(n_boson), atol=1e-10)
