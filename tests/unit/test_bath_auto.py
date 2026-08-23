"""Tests for automatic bath discretization defaults (domain + n_modes)."""
import numpy as np
import pytest

from fishbonett.bath.auto import (reorganization_energy, auto_domain, auto_n_modes,
                                  _reorg_profile)
from fishbonett.bath.lanczos import lanczos
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z


def _ohmic(w):
    return 0.2 * w * np.exp(-w / 5.0)               # reorg lambda = 0.2*5/pi


def test_reorganization_energy_ohmic():
    assert abs(reorganization_energy(_ohmic) - 0.2 * 5 / np.pi) < 1e-3


def test_reorganization_energy_drude():
    lam, gam = 0.5, 2.0
    J = lambda w: 2 * lam * gam * w / (w**2 + gam**2)   # heavy 1/w tail
    assert abs(reorganization_energy(J) - lam) < 1e-2


@pytest.mark.parametrize("frequency_scale", [1e-10, 1.0, 1e10])
def test_automatic_reorganization_is_independent_of_frequency_units(
        frequency_scale):
    lam = 0.7
    density = lambda w: (
        2 * lam * frequency_scale * w / (w**2 + frequency_scale**2)
    )
    assert reorganization_energy(density) == pytest.approx(lam, rel=2e-6)


def test_auto_domain_covers_reorg_energy():
    lo, hi = auto_domain(_ohmic, coverage=0.999)
    assert lo == 0.0 and 25.0 < hi < 45.0          # ~34.5 for this bath
    w, cum = _reorg_profile(_ohmic)
    covered = np.interp(hi, w, cum) / cum[-1]
    assert covered >= 0.999 - 1e-3


def test_auto_domain_thermal_is_asymmetric():
    """A finite-temperature (thermofield) domain is signed but asymmetric: the
    negative branch J(w) n_beta is thermally suppressed, so its edge sits closer to
    zero than the positive edge, and it widens with temperature."""
    lo, hi = auto_domain(_ohmic, beta=1.0)
    assert lo < 0 < hi and abs(lo) < hi              # asymmetric, negative tighter
    lo_hot, hi_hot = auto_domain(_ohmic, beta=0.5)   # T = 2 > 1
    assert abs(lo_hot) > abs(lo)                     # hotter -> wider negative wing
    assert hi_hot == pytest.approx(hi, rel=1e-3)     # positive edge ~ unchanged


def test_auto_n_modes_grows_with_tmax():
    dom = (0.0, 35.0)
    n_short, n_long = auto_n_modes(_ohmic, dom, 0.5), auto_n_modes(_ohmic, dom, 4.0)
    assert 2 < n_short < n_long


def test_lanczos_can_return_a_stable_prefix_for_light_cone_estimation():
    hamiltonian = np.diag([1.0, 1.0])
    coupling = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="terminated before spanning"):
        lanczos(hamiltonian, coupling)
    projected, transform = lanczos(
        hamiltonian, coupling, allow_early_termination=True
    )
    assert projected.shape == (1, 1)
    assert transform.shape == (2, 1)


def test_auto_modes_handles_the_published_dba_bath():
    """The wide, thermally extended DBA density used to break near the end of
    a large trial Lanczos basis even though its physical light cone was valid."""
    cm_to_rad_ps = 0.1883651567308853
    cutoff = 600.0
    alpha = 1.67

    def density(omega):
        omega_cm = omega / cm_to_rad_ps
        return (
            cm_to_rad_ps * 0.5 * alpha * np.pi * omega_cm
            * np.exp(-omega_cm / cutoff)
        )

    beta = 1.0 / (0.6950348009 * 300.0 * cm_to_rad_ps)
    resolved = Bath(
        J=density, beta=beta, phys_dim=3, discretization="tedopa"
    ).resolved(1.0)
    assert resolved.n_modes > 128
    assert resolved.domain[0] < 0.0 < resolved.domain[1]


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
    ref = SystemBath(sigma_x, sigma_z, Bath(J=_ohmic, domain=(-40, 40),
                                           temperature=1.0, n_modes=40, phys_dim=8))
    rr = ref.run(dt=0.05, t_max=0.5, method="interaction-chain-tree-tebd", bond_dim=30,
                 observables={"sz": sigma_z})
    auto = SystemBath(sigma_x, sigma_z, Bath(J=_ohmic, temperature=1.0, phys_dim=8))
    ra = auto.run(dt=0.05, t_max=0.5, method="interaction-chain-tree-tebd", bond_dim=30,
                  observables={"sz": sigma_z})
    assert np.max(np.abs(ra.expect["sz"] - rr.expect["sz"])) < 5e-3
