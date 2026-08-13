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
    for mod in ("fishbonett.states.comb", "fishbonett.evolve.tebd_comb"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
    # frames.schrodinger exists again, but as the *frame* rather than a pair of
    # (model, frame) builder classes -- it emits LocalTerms for any topology.
    sch = importlib.import_module("fishbonett.frames.schrodinger")
    assert hasattr(sch, "terms")
    for gone in ("FishBoneH", "SystemBathSchrodinger"):
        assert not hasattr(sch, gone), f"{gone} should not have come back"


def test_schrodinger_frame_serves_every_topology():
    """The point of Stage 3: one frame implementation, any geometry.

    The multi-site models used to build their static Hamiltonian inline, bypassing
    frames/ entirely, which is why the package could hold a `frames` directory that
    half the models never touched.
    """
    import numpy as np
    from fishbonett import Bath, Fishbone
    from fishbonett.models import TreeFishbone
    from fishbonett.frames.terms import LocalTerms
    from fishbonett.operators import sigma_x, sigma_z

    J = lambda w: 0.2 * w * np.exp(-w / 5.0)
    mk = lambda: Bath(J=J, domain=(0.0, 40.0), n_modes=2, phys_dim=4,
                      coupling=sigma_z)
    C = 0.3 * np.kron(sigma_z, sigma_z)
    h = 0.5 * sigma_z + sigma_x

    comb = Fishbone(sites=[h, h], baths=[mk(), mk()], backbone=[C])
    tree = TreeFishbone(sites=[h, h], edges=[(0, 1, C)], baths=[mk(), mk()])
    a, b = comb.local_terms(), tree.local_terms()

    assert isinstance(a, LocalTerms) and isinstance(b, LocalTerms)
    assert a.dims == b.dims and a.edges == b.edges
    assert all(np.allclose(x, y) for x, y in zip(a.site, b.site))
    assert set(a.bond) == set(b.bond)
    assert all(np.allclose(a.bond[k], b.bond[k]) for k in a.bond)

    # a zero on-site term becomes None, not an identity gate, so the propagators
    # can skip it
    site_gates, edge_gates = a.gates(0.01)
    assert len(site_gates) == a.n_nodes and len(edge_gates) == len(a.edges)
