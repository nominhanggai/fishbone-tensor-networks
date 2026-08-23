"""Discrete molecular vibrations in the public Bath specification."""
import numpy as np

from fishbonett import Bath
from fishbonett.bath._coefficients import chain_coefficients, star_coefficients


def test_huang_rhys_modes_have_expected_couplings_and_reorganization():
    frequency = np.array([1.5, 3.0, 7.5])
    huang_rhys = np.array([0.2, 0.05, 0.01])
    bath = Bath.vibronic(frequency, huang_rhys, phys_dim=4).resolved(2.0)
    star = star_coefficients(bath)
    assert np.allclose(star.frequencies, frequency)
    assert np.allclose(star.couplings[0], frequency * np.sqrt(huang_rhys))
    assert np.isclose(bath.reorganization_energy(),
                      np.sum(frequency * huang_rhys))


def test_thermal_discrete_modes_obey_detailed_balance():
    beta = 0.7
    bath = Bath.vibronic([2.0], [0.25], beta=beta).resolved(1.0)
    star = star_coefficients(bath)
    positive = star.couplings[0, star.frequencies > 0][0] ** 2
    negative = star.couplings[0, star.frequencies < 0][0] ** 2
    assert np.isclose(negative / positive, np.exp(-beta * 2.0))
    # Thermal doubling must not count negative effective frequencies in lambda.
    assert np.isclose(bath.reorganization_energy(), 0.5)


def test_vibronic_star_and_chain_have_same_correlation():
    bath = Bath.vibronic([1.0, 2.5, 4.0], [0.3, 0.1, 0.02]).resolved(3.0)
    star = star_coefficients(bath)
    chain = chain_coefficients(bath)
    h_chain = np.diag(chain.frequencies)
    h_chain += np.diag(chain.hoppings, 1) + np.diag(chain.hoppings, -1)
    values, vectors = np.linalg.eigh(h_chain)
    strengths = chain.system_coupling * vectors[0]
    times = np.linspace(0, 3, 101)
    c_star = np.sum(star.couplings[0][None, :] ** 2
                    * np.exp(-1j * np.outer(times, star.frequencies)), axis=1)
    c_chain = np.sum(strengths[None, :] ** 2
                     * np.exp(-1j * np.outer(times, values)), axis=1)
    assert np.allclose(c_chain, c_star, atol=1e-12)


def test_zero_strength_and_degenerate_vibronic_lines_are_reduced():
    bath = Bath.vibronic(
        [1.0, 2.0, 1.0], [0.1, 0.0, 0.2], phys_dim=4).resolved(1.0)
    star = star_coefficients(bath)
    chain = chain_coefficients(bath)
    assert bath.n_modes == 1
    np.testing.assert_allclose(star.frequencies, [1.0])
    np.testing.assert_allclose(star.couplings[0], [np.sqrt(0.3)])
    assert np.all(np.isfinite(star.transform))
    assert np.all(np.isfinite(chain.frequencies))


def test_correlation_controlled_compression_reports_achieved_error():
    bath = Bath.vibronic(
        np.linspace(1.0, 8.0, 12), np.geomspace(0.2, 1e-4, 12)).resolved(0.3)
    compressed = bath.compressed(0.3, correlation_tol=1e-3)
    times = np.linspace(0, 0.3, 401)
    error = np.max(np.abs(bath.correlation(times) - compressed.correlation(times)))
    error /= abs(bath.correlation([0])[0])
    assert error <= 1e-3 * 1.001
    assert compressed.n_modes <= bath.n_modes
