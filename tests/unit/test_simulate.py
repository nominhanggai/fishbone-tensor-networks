"""Tests for the high-level SpinBoson / Bath / Result interface."""
import numpy as np
import pytest

from fishbonett.simulate import Bath, SpinBoson, Result
from fishbonett.stuff import sigma_x, sigma_z
from fishbonett.tree import _star_transform, anih, crea, SZ, SX

N, D, V = 3, 5, 1.0


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _model(discretization="legendre"):
    bath = Bath(J=_J, domain=(-25.0, 36.0), temperature=1.0, n_modes=N, phys_dim=D,
                discretization=discretization)
    return SpinBoson(h=V * sigma_x, coupling=sigma_z, bath=bath)


def _embed(op, s, dims):
    m = [np.eye(x) for x in dims]
    m[s] = op
    o = m[0]
    for x in m[1:]:
        o = np.kron(o, x)
    return o


def _exact_sz(bath, ts):
    freq, Vn, _ = _star_transform(bath.spectral_density(), N, (-25.0, 36.0))
    dims = [2] + [D] * N
    H = _embed(V * SX, 0, dims)
    for k in range(N):
        H = H + freq[k] * _embed(crea(D) @ anih(D), 1 + k, dims)
        H = H + Vn[k] * (_embed(SZ, 0, dims) @ _embed(anih(D) + crea(D), 1 + k, dims))
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(int(np.prod(dims)), complex); p0[0] = 1
    c = U.conj().T @ p0
    sz = _embed(SZ, 0, dims)
    return np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                     @ (sz @ (U @ (np.exp(-1j * E * t) * c))) for t in ts]).real


def _exact_general(h, O, obs_op, ts, nm, dph, domain, sd, init):
    """Exact evolution of a *general* (ds-level) system coupled to the discretized
    star through operator ``O``; positive-domain (T=0) spectral density ``sd``."""
    freq, Vn, _ = _star_transform(sd, nm, domain)
    ds = h.shape[0]
    dims = [ds] + [dph] * nm
    a = anih(dph)
    H = _embed(np.asarray(h, complex), 0, dims)
    for k in range(nm):
        H = H + freq[k] * _embed(crea(dph) @ a, 1 + k, dims)
        H = H + Vn[k] * (_embed(np.asarray(O, complex), 0, dims)
                         @ _embed(a + crea(dph), 1 + k, dims))
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(int(np.prod(dims)), complex)
    v = np.asarray(init, complex); v = v / np.linalg.norm(v)
    stride = int(np.prod(dims[1:]))                      # |init> (x) |vac...vac>
    for i in range(ds):
        p0[i * stride] = v[i]
    c = U.conj().T @ p0
    ob = _embed(np.asarray(obs_op, complex), 0, dims)
    return np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                     @ (ob @ (U @ (np.exp(-1j * E * t) * c))) for t in ts]).real


@pytest.mark.parametrize("method,step", [("tebd", 2), ("mpo-tdvp1", 2),
                                         ("mpo-tdvp2", 2), ("mpo-ip-tdvp1", 2),
                                         ("mpo-ip-tdvp2", 2), ("tree-tdvp", 1),
                                         ("tree-tdvp2", 1), ("tree-tebd", 1)])
def test_method_matches_exact(method, step):
    model = _model()
    r = model.run(dt=0.05, n_steps=10, method=method, bond_dim=40, trunc_eps=1e-12,
                  observables={"sz": sigma_z})
    assert isinstance(r, Result)
    sz_ex = _exact_sz(model.bath, r.t)
    assert np.max(np.abs(r.expect["sz"] - sz_ex)) < 1e-2


def test_result_carries_observables_and_rdm():
    model = _model()
    r = model.run(dt=0.05, n_steps=8, method="tree-tdvp2", bond_dim=30,
                  observables={"sz": sigma_z, "sx": sigma_x})
    assert set(r.expect) == {"sz", "sx"}
    assert r.rdm.shape == (8, 2, 2)
    # populations are real and normalized
    assert np.allclose([np.trace(rho).real for rho in r.rdm], 1.0, atol=1e-6)


def test_orthpol_discretization_runs():
    model = _model(discretization="orthpol")
    r = model.run(dt=0.05, n_steps=6, method="tree-tdvp", bond_dim=30)
    assert np.all(np.isfinite(r.expect["sz"]))


def test_methods_share_time_grid_and_agree():
    """dt/t_max mean the same physical time for every method family."""
    model = _model()
    methods = ["tebd", "mpo-tdvp1", "mpo-tdvp2", "mpo-ip-tdvp1", "mpo-ip-tdvp2",
               "tree-tdvp", "tree-tdvp2", "tree-tebd"]
    results = {m: model.run(dt=0.05, t_max=0.5, method=m, bond_dim=40,
                            trunc_eps=1e-12, observables={"sz": sigma_z})
               for m in methods}
    ref = results["tebd"]
    assert len(ref.t) == 10 and abs(ref.t[-1] - 0.5) < 1e-12
    for m, r in results.items():
        assert np.allclose(r.t, ref.t)                       # same time grid
        assert abs(r.expect["sz"][-1] - ref.expect["sz"][-1]) < 5e-2  # agree


def test_spinboson_multichannel_routes_to_star():
    """SpinBoson with a multichannel bath (sz AND sx) keeps the spin on its own
    site and matches the tree star engine."""
    from fishbonett.treebone import TreeFishbone

    def Jz(w):
        return 0.2 * w * np.exp(-w / 5.0)

    def Jx(w):
        return 0.1 * w * np.exp(-w / 8.0)

    mc = Bath(J=[Jz, Jx], coupling=[sigma_z, sigma_x], domain=(0.0, 40.0),
              n_modes=3, phys_dim=4)
    h = 0.3 * sigma_z + 0.8 * sigma_x
    r = SpinBoson(h=h, coupling=[sigma_z, sigma_x], bath=mc).run(
        dt=0.02, n_steps=10, bond_dim=40, observables={"sz": sigma_z})
    assert r.expect["sz"].shape == (10,)          # single-system, not per-site
    assert r.rdm.shape == (10, 2, 2)
    fbr = TreeFishbone(sites=[h], edges=[], baths=[mc]).run(
        dt=0.02, n_steps=10, bond_dim=40, observables={"sz": sigma_z})
    assert np.allclose(r.expect["sz"], fbr.expect["sz"][:, 0])


def test_composite_spin_vibration_system():
    """System = spin (x) vibration; bath couples only through the spin.  Validated
    vs exact diagonalization of the discretized star."""
    from fishbonett.models.interaction_picture import SpinBoson as Builder, _c
    dv, nm, dph = 2, 2, 4
    I2, Iv = np.eye(2), np.eye(dv)
    bv = _c(dv); nv = bv.T @ bv
    h_sys = (0.25 * np.kron(sigma_z, Iv) + np.kron(sigma_x, Iv)
             + 1.5 * np.kron(I2, nv) + 0.3 * np.kron(sigma_z, bv + bv.T))
    coup = np.kron(sigma_z, Iv)
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=nm, phys_dim=dph)
    model = SpinBoson(h=h_sys, coupling=coup, bath=bath)
    r = model.run(dt=0.02, n_steps=10, method="tebd", bond_dim=40, trunc_eps=1e-12,
                  observables={"sz": coup}, initial="up")
    assert r.rdm.shape == (10, 2 * dv, 2 * dv)

    builder = Builder([dph] * nm + [2 * dv])
    builder.domain = [0.0, 40.0]; builder.sd = _J
    builder.coupling = coup; builder.h_sys = h_sys
    builder.build(g=1)
    freq = builder.freq
    j0 = builder.k_list[0] * builder.coef[0, :]
    dims = [2 * dv] + [dph] * nm

    def emb(op, s):
        m = [np.eye(x) for x in dims]; m[s] = op
        o = m[0]
        for x in m[1:]:
            o = np.kron(o, x)
        return o

    b = _c(dph)
    H = emb(h_sys, 0)
    for k in range(nm):
        H = H + freq[k] * emb(b.T @ b, 1 + k)
        H = H + j0[k] * (emb(coup, 0) @ emb(b + b.T, 1 + k))
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(int(np.prod(dims)), complex); p0[0] = 1
    c = U.conj().T @ p0
    szf = emb(coup, 0)
    sz_ex = np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                      @ (szf @ (U @ (np.exp(-1j * E * t) * c))) for t in r.t]).real
    assert np.max(np.abs(r.expect["sz"] - sz_ex)) < 1e-3


@pytest.mark.parametrize("method", ["mpo-tdvp1", "mpo-tdvp2", "mpo-ip-tdvp1",
                                    "tree-tdvp", "tree-tebd"])
def test_general_coupling_matches_exact(method):
    """The MPO/tree engines handle a non-sigma_z (sigma_x) coupling, validated vs
    exact diagonalization of the discretized star."""
    sd = lambda w: 0.2 * w * np.exp(-w / 5.0)
    bath = Bath(J=sd, domain=(0.0, 40.0), n_modes=3, phys_dim=5)
    h, O = 0.5 * sigma_z + sigma_x, sigma_x
    r = SpinBoson(h=h, coupling=O, bath=bath).run(
        dt=0.02, n_steps=10, method=method, bond_dim=60, trunc_eps=1e-12,
        observables={"sz": sigma_z})
    ex = _exact_general(h, O, sigma_z, r.t, 3, 5, (0.0, 40.0), sd, [1, 0])
    assert np.max(np.abs(r.expect["sz"] - ex)) < 3e-3


@pytest.mark.parametrize("method", ["mpo-tdvp1", "mpo-tdvp2", "tree-tdvp",
                                    "tree-tebd"])
def test_multilevel_system_matches_exact(method):
    """The MPO/tree engines handle a three-level system, validated vs exact."""
    sd = lambda w: 0.15 * w * np.exp(-w / 6.0)
    bath = Bath(J=sd, domain=(0.0, 30.0), n_modes=3, phys_dim=5)
    a3 = anih(3)
    h = np.diag([0.0, 0.8, 1.7]) + 0.3 * (a3 + a3.T)
    O = a3 + a3.T
    n3 = np.diag([0.0, 1.0, 2.0])
    r = SpinBoson(h=h, coupling=O, bath=bath).run(
        dt=0.02, n_steps=10, method=method, bond_dim=60, trunc_eps=1e-12,
        observables={"n": n3}, initial=[1, 0, 0])
    ex = _exact_general(h, O, n3, r.t, 3, 5, (0.0, 30.0), sd, [1, 0, 0])
    assert np.max(np.abs(r.expect["n"] - ex)) < 3e-3


def test_mpo_rejects_non_hermitian_operators():
    """The MPO/tree engines still require Hermitian h / coupling of matching dim."""
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=N, phys_dim=D)
    with pytest.raises(ValueError):                      # non-Hermitian coupling
        SpinBoson(h=sigma_z, coupling=np.array([[0, 1], [0, 0]], complex),
                  bath=bath).run(dt=0.05, n_steps=2, method="mpo-tdvp1")
    with pytest.raises(ValueError):                      # coupling / h dim mismatch
        SpinBoson(h=np.eye(3), coupling=sigma_z, bath=bath).run(
            dt=0.05, n_steps=2, method="tree-tdvp")
