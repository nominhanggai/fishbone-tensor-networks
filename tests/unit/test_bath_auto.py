"""Tests for automatic bath discretization defaults (domain + n_modes)."""
import numpy as np
import pytest

from fishbonett.bath.auto import (reorganization_energy, auto_domain, auto_n_modes,
                                  _reorg_profile)
from fishbonett.simulate import Bath, SpinBoson
from fishbonett.stuff import sigma_x, sigma_z


def _ohmic(w):
    return 0.2 * w * np.exp(-w / 5.0)               # reorg lambda = 0.2*5/pi


def test_reorganization_energy_ohmic():
    assert abs(reorganization_energy(_ohmic) - 0.2 * 5 / np.pi) < 1e-3


def test_reorganization_energy_drude():
    lam, gam = 0.5, 2.0
    J = lambda w: 2 * lam * gam * w / (w**2 + gam**2)   # heavy 1/w tail
    assert abs(reorganization_energy(J) - lam) < 1e-2


def test_auto_domain_covers_reorg_energy():
    lo, hi = auto_domain(_ohmic, coverage=0.999)
    assert lo == 0.0 and 25.0 < hi < 45.0          # ~34.5 for this bath
    w, cum = _reorg_profile(_ohmic)
    covered = np.interp(hi, w, cum) / cum[-1]
    assert covered >= 0.999 - 1e-3


def test_auto_domain_signed_for_temperature():
    lo, hi = auto_domain(_ohmic, signed=True)
    assert hi > 0 and lo == pytest.approx(-hi)


def test_auto_n_modes_grows_with_tmax():
    dom = (0.0, 35.0)
    n_short, n_long = auto_n_modes(_ohmic, dom, 0.5), auto_n_modes(_ohmic, dom, 4.0)
    assert 2 < n_short < n_long


def test_bath_resolved_fills_and_keeps():
    r = Bath(J=_ohmic, phys_dim=10).resolved(2.0)  # both automatic
    assert r.domain is not None and r.n_modes is not None and r.n_modes > 2
    explicit = Bath(J=_ohmic, domain=(0, 30), n_modes=25, phys_dim=10)
    assert explicit.resolved(2.0) is explicit       # explicit values untouched


def test_auto_n_modes_needs_tmax():
    with pytest.raises(ValueError):
        Bath(J=_ohmic, domain=(0, 30), phys_dim=10).resolved()   # n_modes auto, no t_max


def test_auto_defaults_match_explicit_dynamics():
    """A bath with automatic domain/n_modes reproduces a generously-sized
    explicit bath (short time so the light-cone stays small and cheap)."""
    ref = SpinBoson(sigma_x, sigma_z, Bath(J=_ohmic, domain=(-40, 40),
                                           temperature=1.0, n_modes=40, phys_dim=8))
    rr = ref.run(dt=0.05, t_max=0.5, method="tree-tebd", bond_dim=30,
                 observables={"sz": sigma_z})
    auto = SpinBoson(sigma_x, sigma_z, Bath(J=_ohmic, temperature=1.0, phys_dim=8))
    ra = auto.run(dt=0.05, t_max=0.5, method="tree-tebd", bond_dim=30,
                  observables={"sz": sigma_z})
    assert np.max(np.abs(ra.expect["sz"] - rr.expect["sz"])) < 5e-3
