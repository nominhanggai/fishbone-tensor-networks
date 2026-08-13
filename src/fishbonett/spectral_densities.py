"""Analytic bath spectral densities ``J(omega)``.

Formerly part of the catch-all ``fishbonett.operators`` module.  Pass any of these (or
your own callable) as the ``J`` of a :class:`fishbonett.bath.spec.Bath`.
"""
import numpy as np


def sd_back(Sk, sk, w, wk):
    """Log-normal ("background") peak of Huang-Rhys weight ``Sk`` centred on ``wk``."""
    return np.pi * Sk / (sk * np.sqrt(2 / np.pi)) * w * \
           np.exp(-np.log(np.abs(w) / wk) ** 2 / (2 * sk ** 2))


def sd_high(gamma_m, Omega_m, g_m, w):
    """Underdamped ("high-frequency") vibrational peak at ``Omega_m``."""
    return 4 * gamma_m * Omega_m * g_m * (Omega_m ** 2 + gamma_m ** 2) * w \
           / ((gamma_m ** 2 + (w + Omega_m) ** 2) * (gamma_m ** 2 + (w - Omega_m) ** 2))


def sd_zero_temp(w):
    """Structured zero-temperature density (three background + three sharp peaks)."""
    gamma = 5.
    Omega_1, Omega_2, Omega_3 = 181, 221, 240
    g1, g2, g3 = 0.0173, 0.0246, 0.0182
    S1, S2, S3 = 0.39, 0.23, 0.23
    s1, s2, s3 = 0.4, 0.25, 0.2
    w1, w2, w3 = 26, 51, 85
    return sd_back(S1, s1, w, w1) + sd_back(S2, s2, w, w2) \
           + sd_back(S3, s3, w, w3) + sd_high(gamma, Omega_1, g1, w) \
           + sd_high(gamma, Omega_2, g2, w) \
           + sd_high(gamma, Omega_3, g3, w)


def sd_back_zero_temp(w):
    """The background (log-normal) part of :func:`sd_zero_temp` only."""
    S1, S2, S3 = 0.39, 0.23, 0.23
    s1, s2, s3 = 0.4, 0.25, 0.2
    w1, w2, w3 = 26, 51, 85
    return sd_back(S1, s1, w, w1) + sd_back(S2, s2, w, w2) + sd_back(S3, s3, w, w3)


sd_zero_temp_prime = sd_back_zero_temp        # historical alias


def lorentzian(eta, w, lambd=5245., omega=77.):
    """Lorentzian (single underdamped mode); parameters from dx.doi.org/10.1021/jp400462f."""
    return 0.5 * lambd * (omega ** 2) * eta * w / ((w ** 2 - omega ** 2) ** 2 + (eta ** 2) * (w ** 2))


def drude1(w, lam, gam=100.):
    """Drude-Lorentz (overdamped) density with a unit-converted ``gam``."""
    gam = gam / 1.8836515673088531
    return 2 * lam * gam * w / (w ** 2 + gam ** 2)


def drude(w, lam, gam=100.):
    """Drude-Lorentz (overdamped) density, reorganization energy ``lam``, cutoff ``gam``."""
    return 2 * lam * gam * w / (w ** 2 + gam ** 2)


def brownian(w, lam, gam, w0=1):
    """Brownian-oscillator density centred on ``w0`` with damping ``gam``."""
    return 2 * lam * gam * w0 ** 2 * w / ((w0 ** 2 - w ** 2) + gam ** 2 * w ** 2)


def natphys(w, lam):
    """Structured super-ohmic density (Nature Physics light-harvesting model)."""
    return lam * np.pi * 0.5 * (
            1000 * w ** 5 * np.exp(- np.sqrt(w / 0.57)) + 4.3 * w ** 5 * np.exp(-np.sqrt(w / 1.9))) / (
                   362880. * (1000. * 0.57 ** 5 + 4.3 * 1.9 ** 5))


def lemmer(w, lam, k, wm):
    """Antisymmetrized Lorentzian pair centred on +/- ``wm`` (Lemmer et al.)."""
    return lam ** 2 * (k / (k ** 2 + (w - wm) ** 2) -
                       k / (k ** 2 + (w + wm) ** 2))
