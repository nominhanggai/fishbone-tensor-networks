"""Smoke tests: each method family builds and runs on a tiny problem."""
import importlib

import numpy as np
import pytest


def test_spin_boson_model_builds_and_propagates():
    """The clean interaction-picture-wrt-H_SB model (keyword-constructed)."""
    from fishbonett.int_pic_hsb_spin_boson import SpinBosonModel
    from fishbonett.spin_boson_mps import SpinBosonMPS

    pd_boson = [6, 6, 6]
    eth = SpinBosonModel(v_x=50.0, v_z=0.0, pd_spin=2, pd_boson=pd_boson,
                         boson_domain=[0.0, 100.0],
                         sd=lambda w: 0.5 * w * np.exp(-w / 20.0), dt=1e-3)
    u_one, u_half = eth.get_u(0.0, 1e-3)
    assert len(u_one) == len(pd_boson)          # one gate per bond
    assert all(np.all(np.isfinite(u)) for u in u_one)

    etn = SpinBosonMPS(pd_spin=2, pd_boson=pd_boson)
    etn.B[0][0, 0, 0] = 1.0                       # system site is first here
    etn.U = u_one
    for j in range(len(pd_boson)):
        etn.update_bond(j, 20, 1e-8, swap=0)
    assert all(np.all(np.isfinite(b)) for b in etn.B)


def test_chain_cooling_gives_normalized_rdm():
    from fishbonett.coolingC_SpinBoson import SpinBoson
    from fishbonett.stuff import sigma_x, sigma_z

    pd = [6, 6, 6, 2]
    eth = SpinBoson(pd, betaOmega=0.2)
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
    bases = [b.__name__ for b in mod.SpinBoson.__mro__]
    assert "SpinBosonMPS" in bases


def test_public_api_surface():
    import fishbonett as fb
    for name in ("SpinBosonMPS", "FishBoneNet", "FishBoneH", "SpinBosonModel",
                 "get_bath_nn_paras", "get_coupling", "lanczos",
                 "sigma_x", "sigma_z", "drude", "lorentzian"):
        assert hasattr(fb, name), name
