"""Smoke tests: each method family builds and runs on a tiny problem."""
import importlib

import numpy as np
import pytest


def test_chain_cooling_gives_normalized_rdm():
    from fishbonett.frames.coolingchain import SystemBathCoolingChain
    from fishbonett.operators import sigma_x, sigma_z

    pd = [2, 6, 6, 6]
    eth = SystemBathCoolingChain(
        pd, betaOmega=0.2, h_sys=10.0 * sigma_x, coupling=sigma_z,
        sd=lambda w: 0.5 * abs(w) * np.exp(-abs(w) / 10.0),
        domain=[-50.0, 50.0], ncap=200).build()
    eth.U = eth.get_u(0.01)
    for j in range(len(pd) - 1):
        eth.update_bond(j, 20, 1e-6, swap=0)
    rho = eth.get_rdm()
    assert np.all(np.isfinite(rho))
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)


@pytest.mark.parametrize("module,cls", [("frames.coolingchain", "SystemBathCoolingChain")])
def test_cooling_shares_the_canonical_engine(module, cls):
    mod = importlib.import_module(f"fishbonett.{module}")
    bases = [b.__name__ for b in getattr(mod, cls).__mro__]
    assert "SystemBathMPS" in bases


def test_polaron_builds_and_gives_normalized_rdm():
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    bath = Bath(J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
                n_modes=6, phys_dim=6)
    r = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=bath).run(
        method="polaron", dt=0.05, n_steps=3, bond_dim=30)
    assert np.all(np.isfinite(r.rdm))
    assert np.allclose([np.trace(rho).real for rho in r.rdm], 1.0, atol=1e-6)


def test_public_api_surface():
    import fishbonett as fb
    for name in ("SystemBathMPS", "TreeTEBD", "SystemBath", "Fishbone",
                 "TreeFishbone", "Bath", "Result", "Truncation",
                 "get_bath_nn_paras", "get_coupling", "lanczos",
                 "sigma_x", "sigma_z", "drude", "lorentzian"):
        assert hasattr(fb, name), name
    # every advertised name must resolve, or `from fishbonett import *` breaks
    missing = [n for n in fb.__all__ if not hasattr(fb, n)]
    assert not missing, missing


def test_removed_comb_engine_is_gone():
    """FishBoneNet/FishBoneH/SystemBath1D/SystemBathSchrodinger were unreachable
    from run() and exercised only by a name check; the comb geometry is covered by
    Fishbone -> TreeTEBD, which is validated against exact diagonalization."""
    import fishbonett as fb
    for name in ("FishBoneNet", "FishBoneH", "SystemBath1D",
                 "SystemBathSchrodinger", "init_ttn"):
        assert not hasattr(fb, name), f"{name} should have been removed"
    for mod in ("fishbonett.states.comb", "fishbonett.frames.schrodinger",
                "fishbonett.evolve.tebd_comb"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
