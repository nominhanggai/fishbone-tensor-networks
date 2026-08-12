"""Smoke tests: each method family builds and runs on a tiny problem."""
import importlib

import numpy as np
import pytest


def test_chain_cooling_gives_normalized_rdm():
    from fishbonett.coolingC_SpinBoson import BosonicBath
    from fishbonett.stuff import sigma_x, sigma_z

    pd = [6, 6, 6, 2]
    eth = BosonicBath(pd, betaOmega=0.2)
    eth.domain = [-50.0, 50.0]
    eth.sd = lambda w: 0.5 * abs(w) * np.exp(-abs(w) / 10.0)
    eth.he_dy = sigma_z
    eth.h1e = 10.0 * sigma_x
    eth.build(g=1, ncap=200)
    eth.U = eth.get_u(0.01)
    for j in range(len(pd) - 1):
        eth.update_bond(j, 20, 1e-6, swap=0)
    rho = eth.get_rdm()
    assert np.all(np.isfinite(rho))
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)


@pytest.mark.parametrize("module", ["coolingC_SpinBoson"])
def test_cooling_shares_the_canonical_engine(module):
    mod = importlib.import_module(f"fishbonett.{module}")
    bases = [b.__name__ for b in mod.BosonicBath.__mro__]
    assert "BosonicBathMPS" in bases


def test_public_api_surface():
    import fishbonett as fb
    for name in ("BosonicBathMPS", "FishBoneNet", "FishBoneH",
                 "get_bath_nn_paras", "get_coupling", "lanczos",
                 "sigma_x", "sigma_z", "drude", "lorentzian"):
        assert hasattr(fb, name), name
