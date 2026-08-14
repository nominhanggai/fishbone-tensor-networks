"""Scientific sign/normalization conventions shared by every frame."""
import numpy as np

from fishbonett import Bath
from fishbonett.bath.conventions import (
    integrated_free_phase, reorganization_energy, star_coupling_squared,
)


def test_integrated_phase_has_exact_zero_frequency_limit():
    assert integrated_free_phase(0.0, 3.2, 0.125) == 0.125
    frequency = np.array([0.0, 1e-12, 0.7])
    got = integrated_free_phase(frequency, 0.3, 0.2)
    reference = np.array([
        0.2,
        0.2 * np.exp(-1j * 1e-12 * 0.4),
        (np.exp(-1j * 0.7 * 0.5) - np.exp(-1j * 0.7 * 0.3))
        / (-1j * 0.7),
    ])
    np.testing.assert_allclose(got, reference, rtol=1e-13, atol=1e-15)


def test_star_quadrature_uses_j_weight_over_pi():
    assert np.isclose(star_coupling_squared(lambda w: 2.0 * w, 3.0, 0.4),
                      2.4 / np.pi)


def test_reorganization_energy_reference_value():
    # J(w)=2 lambda w/wc exp(-w/wc) gives integral_0^inf J/(pi*w)
    # = 2 lambda/pi in this package's convention.
    lam, cutoff = 0.7, 2.0
    density = lambda w: 2.0 * lam * w / cutoff * np.exp(-w / cutoff)
    got = reorganization_energy(density, (0.0, 30.0), points=20001)
    assert np.isclose(got, 2.0 * lam / np.pi, rtol=1e-5)


def test_interaction_picture_matches_star_equation_at_t_zero():
    density = lambda w: 0.2 * w * np.exp(-w / 5.0)
    star = Bath(
        J=density, domain=(0.0, 30.0), n_modes=4, phys_dim=3
    ).bind(np.diag([1.0, -1.0])).compiled_star()
    # Eq. (3) of Nuomin et al., arXiv:2212.06099: annihilation terms carry
    # g_k exp(-i omega_k t), so at t=0 they are exactly the discrete g_k.
    np.testing.assert_allclose(
        star.interaction_couplings(0.0), star.couplings[0], atol=1e-14)
