"""Tests for the high-level SimpleSysBath / Bath / Result interface."""
import numpy as np
import pytest

from fishbonett import Bath, SimpleSysBath, Result
from fishbonett.bath.chain import get_vn_squared
from fishbonett.operators import sigma_x, sigma_z
from fishbonett.evolve.treetdvp import _star_transform, annihilate, create, SZ, SX

N, D, V = 3, 5, 1.0


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _model(discretization="legendre"):
    bath = Bath(J=_J, domain=(-25.0, 36.0), temperature=1.0, n_modes=N, phys_dim=D,
                discretization=discretization)
    return SimpleSysBath(h=V * sigma_x, coupling=sigma_z, bath=bath)


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
        H = H + freq[k] * _embed(create(D) @ annihilate(D), 1 + k, dims)
        H = H + Vn[k] * (_embed(SZ, 0, dims) @ _embed(annihilate(D) + create(D), 1 + k, dims))
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
    a = annihilate(dph)
    H = _embed(np.asarray(h, complex), 0, dims)
    for k in range(nm):
        H = H + freq[k] * _embed(create(dph) @ a, 1 + k, dims)
        H = H + Vn[k] * (_embed(np.asarray(O, complex), 0, dims)
                         @ _embed(a + create(dph), 1 + k, dims))
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


@pytest.mark.parametrize("method,step", [("tebd", 2), ("trotter-mpo", 2),
                                         ("mpo-tdvp1", 2),
                                         ("mpo-tdvp2", 2), ("mpo-ip-tdvp1", 2),
                                         ("mpo-ip-tdvp2", 2),
                                         ("mpo-star-tdvp1", 2),
                                         ("mpo-star-tdvp2", 2),
                                         ("tree-tdvp", 1),
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


def test_tedopa_discretization_runs():
    model = _model(discretization="tedopa")
    r = model.run(dt=0.05, n_steps=6, method="tree-tdvp", bond_dim=30)
    assert np.all(np.isfinite(r.expect["sz"]))


def test_methods_share_time_grid_and_agree():
    """dt/t_max mean the same physical time for every method family."""
    model = _model()
    methods = ["tebd", "trotter-mpo", "mpo-tdvp1", "mpo-tdvp2", "mpo-ip-tdvp1",
               "mpo-ip-tdvp2", "mpo-star-tdvp1", "mpo-star-tdvp2",
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
    """SimpleSysBath with a multichannel bath (sz AND sx) keeps the spin on its own
    site and matches the tree star engine."""
    from fishbonett.models.fishbone import TreeFishbone

    def Jz(w):
        return 0.2 * w * np.exp(-w / 5.0)

    def Jx(w):
        return 0.1 * w * np.exp(-w / 8.0)

    mc = Bath(J=[Jz, Jx], coupling=[sigma_z, sigma_x], domain=(0.0, 40.0),
              n_modes=3, phys_dim=4)
    h = 0.3 * sigma_z + 0.8 * sigma_x
    r = SimpleSysBath(h=h, coupling=[sigma_z, sigma_x], bath=mc).run(
        dt=0.02, n_steps=10, bond_dim=40, observables={"sz": sigma_z})
    assert r.expect["sz"].shape == (10,)          # single-system, not per-site
    assert r.rdm.shape == (10, 2, 2)
    fbr = TreeFishbone(sites=[h], edges=[], baths=[mc]).run(
        dt=0.02, n_steps=10, bond_dim=40, observables={"sz": sigma_z})
    assert np.allclose(r.expect["sz"], fbr.expect["sz"][:, 0])


def test_multichannel_ip_matches_the_static_path_and_exact():
    """The two multichannel frames describe the same shared-mode star, so they must
    agree with each other and with exact diagonalization.  This is the check that
    the interaction-picture builder is wired up with the *same* temperature and
    discretization conventions as the static one -- it uses the T-TEDOPA signed
    density rather than the builder's own kelvin-unit thermofield doubling."""
    ops = [sigma_z, sigma_x]
    nm, d = 3, 6
    sd = lambda w: 0.15 * w * np.exp(-w / 6.0)
    mc = Bath(J=[sd, sd], coupling=ops, domain=(0.0, 30.0), n_modes=nm, phys_dim=d)
    h = 0.5 * sigma_x
    model = SimpleSysBath(h=h, coupling=ops, bath=mc)

    obs = {"sz": sigma_z, "sx": sigma_x}
    kw = dict(dt=0.01, n_steps=20, bond_dim=80, trunc_eps=1e-12, observables=obs)
    r_static = model.run(**kw)
    r_ip = model.run(method="multichannel-ip", **kw)
    assert r_static.method == "tree-tebd-static"
    assert r_ip.method == "multichannel-ip"

    # exact diagonalization of the same shared-mode star
    freq, g = None, []
    for Jc, _op in mc.channels():
        f, v_sq = get_vn_squared(Jc, nm, list(mc.domain))
        g.append(np.sqrt(np.asarray(v_sq) / np.pi))
        freq = np.asarray(f) if freq is None else freq
    dims = [2] + [d] * nm

    def emb(op, s):
        m = [np.eye(x) for x in dims]
        m[s] = op
        out = m[0]
        for x in m[1:]:
            out = np.kron(out, x)
        return out

    b = annihilate(d)
    H = emb(h, 0)
    for k in range(nm):
        M = sum(g[c][k] * ops[c] for c in range(len(ops)))
        H = H + freq[k] * emb(b.conj().T @ b, 1 + k) + emb(M, 0) @ emb(b + b.T, 1 + k)
    E, U = np.linalg.eigh(H)
    v0 = np.zeros(int(np.prod(dims)), complex)
    v0[0] = 1.0                                    # |up> (x) vacuum
    c = U.conj().T @ v0
    for name, O in obs.items():
        ref = np.array([
            np.einsum("ij,ji->", (lambda p: p @ p.conj().T)(
                (U @ (np.exp(-1j * E * t) * c)).reshape(2, -1)), O).real
            for t in r_ip.t])
        assert np.max(np.abs(r_ip.expect[name] - ref)) < 3e-3, f"ip vs exact ({name})"
        assert np.max(np.abs(r_static.expect[name] - ref)) < 3e-3, f"static vs exact ({name})"


def test_multichannel_ip_rejects_a_zero_lanczos_seed():
    """A coupling set whose diagonal vanishes in the working basis gives a zero
    Lanczos seed.  That used to produce a silent NaN chain; it must raise."""
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sd = lambda w: 0.15 * w * np.exp(-w / 6.0)
    mc = Bath(J=[sd, sd], coupling=[sigma_x, sy], domain=(0.0, 30.0),
              n_modes=3, phys_dim=4)
    m = SimpleSysBath(h=0.5 * sigma_z, coupling=[sigma_x, sy], bath=mc)
    with pytest.raises(ValueError, match="seed"):
        m.run(dt=0.01, n_steps=1, method="multichannel-ip", bond_dim=20)


def test_composite_spin_vibration_system():
    """System = spin (x) vibration; bath couples only through the spin.  Validated
    vs exact diagonalization of the discretized star."""
    from fishbonett.frames.interaction_picture import SimpleSysBathIP as Builder, annihilate
    dv, nm, dph = 2, 2, 4
    I2, Iv = np.eye(2), np.eye(dv)
    bv = annihilate(dv); nv = bv.T @ bv
    h_sys = (0.25 * np.kron(sigma_z, Iv) + np.kron(sigma_x, Iv)
             + 1.5 * np.kron(I2, nv) + 0.3 * np.kron(sigma_z, bv + bv.T))
    coup = np.kron(sigma_z, Iv)
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=nm, phys_dim=dph)
    model = SimpleSysBath(h=h_sys, coupling=coup, bath=bath)
    r = model.run(dt=0.02, n_steps=10, method="tebd", bond_dim=40, trunc_eps=1e-12,
                  observables={"sz": coup}, initial="up")
    assert r.rdm.shape == (10, 2 * dv, 2 * dv)

    builder = Builder([2 * dv] + [dph] * nm, h_sys=h_sys, coupling=coup,
                      sd=_J, domain=[0.0, 40.0]).build()
    freq = builder.freq
    j0 = builder.k_list[0] * builder.coef[0, :]
    dims = [2 * dv] + [dph] * nm

    def emb(op, s):
        m = [np.eye(x) for x in dims]; m[s] = op
        o = m[0]
        for x in m[1:]:
            o = np.kron(o, x)
        return o

    b = annihilate(dph)
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
    r = SimpleSysBath(h=h, coupling=O, bath=bath).run(
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
    a3 = annihilate(3)
    h = np.diag([0.0, 0.8, 1.7]) + 0.3 * (a3 + a3.T)
    O = a3 + a3.T
    n3 = np.diag([0.0, 1.0, 2.0])
    r = SimpleSysBath(h=h, coupling=O, bath=bath).run(
        dt=0.02, n_steps=10, method=method, bond_dim=60, trunc_eps=1e-12,
        observables={"n": n3}, initial=[1, 0, 0])
    ex = _exact_general(h, O, n3, r.t, 3, 5, (0.0, 30.0), sd, [1, 0, 0])
    assert np.max(np.abs(r.expect["n"] - ex)) < 3e-3


def test_mpo_rejects_non_hermitian_operators():
    """The MPO/tree engines still require Hermitian h / coupling of matching dim."""
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=N, phys_dim=D)
    with pytest.raises(ValueError):                      # non-Hermitian coupling
        SimpleSysBath(h=sigma_z, coupling=np.array([[0, 1], [0, 0]], complex),
                  bath=bath).run(dt=0.05, n_steps=2, method="mpo-tdvp1")
    with pytest.raises(ValueError):                      # coupling / h dim mismatch
        SimpleSysBath(h=np.eye(3), coupling=sigma_z, bath=bath).run(
            dt=0.05, n_steps=2, method="tree-tdvp")


# -- polaron frame -----------------------------------------------------------
def _polaron_bath(nm=14, d=8):
    """T=0, gapped-domain bath so J(w)/w^2 is integrable (polaron precondition)."""
    return Bath(J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
                n_modes=nm, phys_dim=d)


def test_trotter_mpo_bond_is_number_of_coupling_eigenvalues():
    """The conditional-displacement propagator is a sum of one product operator per
    eigenvalue of the coupling ``O``, so the MPO bond is exactly that count -- 2 for
    sigma_z, 3 for a three-eigenvalue coupling -- independent of the chain length."""
    from fishbonett.frames.interaction_picture import SimpleSysBathIP

    for O, expected in [(sigma_z, 2), (np.diag([1.0, 0.0, -1.0]).astype(complex), 3)]:
        ds = O.shape[0]
        b = SimpleSysBathIP([ds] + [6] * 5, h_sys=np.eye(ds), coupling=O,
                         sd=_J, domain=[0.3, 12.0]).build()
        W = b.displacement_mpo(0.0, 0.05)
        assert len(W) == 6                       # system + 5 modes
        assert W[0].shape == (1, expected, ds, ds)
        assert all(w.shape[0] == expected for w in W[1:])
        assert W[-1].shape[1] == 1               # closed at the right edge


def test_trotter_mpo_matches_tebd_general_coupling():
    """Same frame as ``tebd``, so it must agree for a general (3-level) coupling."""
    O = np.diag([1.0, 0.0, -1.0]).astype(complex)
    h = np.zeros((3, 3), complex)
    h[0, 1] = h[1, 0] = h[1, 2] = h[2, 1] = 0.5
    bath = Bath(J=_J, domain=(0.3, 12.0), n_modes=10, phys_dim=8)
    model = SimpleSysBath(h=h, coupling=O, bath=bath)
    kw = dict(dt=0.05, n_steps=20, bond_dim=40, trunc_eps=1e-4, observables={"O": O})
    assert np.max(np.abs(model.run(method="trotter-mpo", **kw).expect["O"]
                         - model.run(method="tebd", **kw).expect["O"])) < 5e-3


POLARON_METHODS = ["polaron", "polaron-tdvp1", "polaron-tdvp2", "polaron-dtdvp"]


@pytest.mark.parametrize("method", POLARON_METHODS)
def test_polaron_matches_ip_populations_and_coherence(method):
    """Every polaron propagator reproduces the interaction-picture chain for a
    2-level spin-boson: the frame-invariant population <sz> and the *un-dressed*
    coherence <sx> both agree (they differ only by the O(dt^2) Trotter split).
    The TEBD variant uses static gates; the TDVP variants use the polaron MPO."""
    model = SimpleSysBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_polaron_bath(nm=10, d=8))
    kw = dict(dt=0.02, n_steps=25, bond_dim=16, trunc_eps=1e-4,
              observables={"sz": sigma_z, "sx": sigma_x})
    rp = model.run(method=method, **kw)
    ri = model.run(method="tebd", **kw)
    # The two frames describe the same physics, so they agree far better than the
    # O(dt^2) splitting bound: ~1e-5 on the population and ~1e-4 on the un-dressed
    # coherence.  Keep the tolerance tight -- a loose one here once hid a misplaced
    # on-site frequency in the polaron gates.
    assert np.max(np.abs(rp.expect["sz"] - ri.expect["sz"])) < 1e-4   # populations
    assert np.max(np.abs(rp.expect["sx"] - ri.expect["sx"])) < 1e-3   # un-dressed


@pytest.mark.parametrize("method", ["polaron", "polaron-dtdvp"])
def test_polaron_general_coupling_matches_ip(method):
    """The polaron frame handles a general (3-level, three-eigenvalue) coupling O."""
    O = np.diag([1.0, 0.0, -1.0]).astype(complex)
    h = np.zeros((3, 3), complex)
    h[0, 1] = h[1, 0] = h[1, 2] = h[2, 1] = 0.5          # off-diagonal in O's eigenbasis
    model = SimpleSysBath(h=h, coupling=O, bath=_polaron_bath(nm=10, d=8))
    kw = dict(dt=0.02, n_steps=25, bond_dim=16, trunc_eps=1e-4, observables={"O": O})
    rp = model.run(method=method, **kw)
    ri = model.run(method="tebd", **kw)
    assert np.max(np.abs(rp.expect["O"] - ri.expect["O"])) < 2e-2


def test_polaron_runs_at_finite_temperature():
    """The polaron frame handles finite T via T-TEDOPA thermalization."""
    bath = Bath(J=_J, domain=(-12.0, 12.0), temperature=1.0, n_modes=8, phys_dim=6)
    r = SimpleSysBath(h=0.5 * sigma_x, coupling=sigma_z, bath=bath).run(
            method="polaron", dt=0.05, n_steps=2, bond_dim=20)
    assert r.t.shape == (2,)
    assert np.allclose(np.trace(r.rdm[-1]), 1.0, atol=1e-6)


def test_free_chain_gates_put_each_frequency_on_its_own_mode():
    """Free-chain bond ``m`` must carry ``k_list[m]`` hopping *and* ``w_list[m]``
    on ``c_m`` -- its **right** leg, because with the system at site 0 the chain
    runs outward.

    The old layout ran the chain inward, where the new mode was the *left* leg;
    carrying that convention over silently pairs each frequency with the
    neighbouring mode.  Populations stay normalized and traces stay 1, so only a
    structural check catches it.
    """
    import scipy.linalg as sla
    from fishbonett.frames.polaron import SimpleSysBathPolaron
    from fishbonett.operators import annihilate

    nb, d, ds = 4, 5, 2
    b = SimpleSysBathPolaron([ds] + [d] * nb, h_sys=0.5 * sigma_x, coupling=sigma_z,
                          sd=lambda w: 0.3 * w * np.exp(-w / 2.5),
                          domain=[0.3, 12.0]).build()

    dt = 1e-5                       # small dt so i log(U)/dt recovers h faithfully
    a = annihilate(d)
    num, Id = a.conj().T @ a, np.eye(d)
    gates = b.gates(dt)
    for m in range(1, nb):          # bond 0 is the dressed bond, checked elsewhere
        want = (b.k_list[m] * (np.kron(a.conj().T, a) + np.kron(a, a.conj().T))
                + b.w_list[m] * np.kron(Id, num))          # w_m on c_m (right leg)
        got = 1j * sla.logm(gates[m].reshape(d * d, d * d)) / dt
        assert np.allclose(got, want, atol=1e-4), (
            f"bond {m}: gate Hamiltonian does not match "
            f"k={b.k_list[m]:.4f} hopping + w={b.w_list[m]:.4f} on c_{m}")
