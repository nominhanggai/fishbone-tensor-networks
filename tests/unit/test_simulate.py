"""Tests for the high-level SystemBath / Bath / Result interface."""
import numpy as np
import pytest

from fishbonett import Bath, SystemBath, Result
from fishbonett.bath.chain import get_vn_squared, star_transform
from fishbonett.evolve.modetree import SZ, SX
from fishbonett.operators import annihilate, create, sigma_x, sigma_z

N, D, V = 3, 5, 1.0


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _model(discretization="legendre"):
    bath = Bath(J=_J, domain=(-25.0, 36.0), temperature=1.0, n_modes=N, phys_dim=D,
                discretization=discretization)
    return SystemBath(h=V * sigma_x, coupling=sigma_z, bath=bath)


def _embed(op, s, dims):
    m = [np.eye(x) for x in dims]
    m[s] = op
    o = m[0]
    for x in m[1:]:
        o = np.kron(o, x)
    return o


def _exact_sz(bath, ts):
    freq, Vn, _ = star_transform(bath.spectral_density(), N, (-25.0, 36.0))
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
    freq, Vn, _ = star_transform(sd, nm, domain)
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


@pytest.mark.parametrize("method,step", [("interaction-chain-tebd", 2),
                                         ("interaction-chain-trotter-mpo", 2),
                                         ("schrodinger-chain-tdvp1", 2),
                                         ("schrodinger-chain-tdvp2", 2), ("interaction-chain-tdvp1", 2),
                                         ("interaction-chain-tdvp2", 2),
                                         ("schrodinger-star-tdvp1", 2),
                                         ("schrodinger-star-tdvp2", 2),
                                         ("interaction-chain-tree-tebd", 1)])
def test_method_matches_exact(method, step):
    model = _model()
    r = model.run(dt=0.05, n_steps=10, method=method, bond_dim=40, trunc_eps=1e-12,
                  observables={"sz": sigma_z})
    assert isinstance(r, Result)
    sz_ex = _exact_sz(model.bath, r.t)
    assert np.max(np.abs(r.expect["sz"] - sz_ex)) < 1e-2


def test_result_carries_observables_and_rdm():
    model = _model()
    r = model.run(dt=0.05, n_steps=8, method="interaction-chain-tree-tebd", bond_dim=30,
                  observables={"sz": sigma_z, "sx": sigma_x})
    assert set(r.expect) == {"sz", "sx"}
    assert r.rdm.shape == (8, 2, 2)
    # populations are real and normalized
    assert np.allclose([np.trace(rho).real for rho in r.rdm], 1.0, atol=1e-6)


def test_tedopa_discretization_runs():
    model = _model(discretization="tedopa")
    r = model.run(dt=0.05, n_steps=6, method="interaction-chain-tree-tebd", bond_dim=30)
    assert np.all(np.isfinite(r.expect["sz"]))


def test_methods_share_time_grid_and_agree():
    """dt/t_max mean the same physical time for every method family."""
    model = _model()
    methods = ["interaction-chain-tebd", "interaction-chain-trotter-mpo",
               "schrodinger-chain-tdvp1", "schrodinger-chain-tdvp2", "interaction-chain-tdvp1",
               "interaction-chain-tdvp2", "schrodinger-star-tdvp1", "schrodinger-star-tdvp2",
               "interaction-chain-tree-tebd"]
    results = {m: model.run(dt=0.05, t_max=0.5, method=m, bond_dim=40,
                            trunc_eps=1e-12, observables={"sz": sigma_z})
               for m in methods}
    ref = results["interaction-chain-tebd"]
    assert len(ref.t) == 10 and abs(ref.t[-1] - 0.5) < 1e-12
    for m, r in results.items():
        assert np.allclose(r.t, ref.t)                       # same time grid
        assert abs(r.expect["sz"][-1] - ref.expect["sz"][-1]) < 5e-2  # agree


def test_spinboson_multichannel_routes_to_star():
    """SystemBath with a multichannel bath (sz AND sx) keeps the spin on its own
    site and matches the tree star engine."""
    from fishbonett.models.fishbone import TreeFishbone

    def Jz(w):
        return 0.2 * w * np.exp(-w / 5.0)

    def Jx(w):
        return 0.1 * w * np.exp(-w / 8.0)

    mc = Bath(J=[Jz, Jx], domain=(0.0, 40.0),
              n_modes=3, phys_dim=4)
    h = 0.3 * sigma_z + 0.8 * sigma_x
    r = SystemBath(h=h, coupling=[sigma_z, sigma_x], bath=mc).run(
        dt=0.02, n_steps=10, bond_dim=40, observables={"sz": sigma_z})
    assert r.expect["sz"].shape == (10,)          # single-system, not per-site
    assert r.rdm.shape == (10, 2, 2)
    fbr = TreeFishbone(
        sites=[h], edges=[], baths=[mc.bind([sigma_z, sigma_x])]).run(
        dt=0.02, n_steps=10, bond_dim=40, observables={"sz": sigma_z})
    assert np.allclose(r.expect["sz"], fbr.expect["sz"][:, 0])


def test_multichannel_ip_mps_matches_the_static_tree_and_exact():
    """The multichannel representations describe the same shared-mode star, so they must
    agree with exact diagonalization. This is the check that
    the interaction-picture builder is wired up with the *same* temperature and
    discretization conventions as the static one -- it uses the T-TEDOPA signed
    density rather than the builder's own kelvin-unit thermofield doubling."""
    ops = [sigma_z, sigma_x]
    nm, d = 3, 6
    sd = lambda w: 0.15 * w * np.exp(-w / 6.0)
    mc = Bath(
        J=[sd, sd], domain=(0.0, 30.0), n_modes=nm, phys_dim=d)
    h = 0.5 * sigma_x
    model = SystemBath(h=h, coupling=ops, bath=mc)

    obs = {"sz": sigma_z, "sx": sigma_x}
    kw = dict(dt=0.01, n_steps=20, bond_dim=80, trunc_eps=1e-12, observables=obs)
    r_static = model.run(**kw)
    r_ip = model.run(method="interaction-chain-tebd", **kw)
    # Two representations of one shared-mode star: the static schrodinger-star
    # and the interaction picture rotated on to a Lanczos chain. The chain seed is
    # one channel, which is legitimate because the interaction picture leaves no
    # mode-mode terms -- see representations/multichannel.py.
    #
    # The static one is `schrodinger-star-tree-tebd`, not the multi-site models'
    # `schrodinger-chain-tree-tebd`: same engine, but those chain-map their baths
    # statically and this cannot, since the channels share one set of modes.
    assert r_static.method == "schrodinger-star-tree-tebd"
    assert r_ip.method == "interaction-chain-tebd"

    # exact diagonalization of the same shared-mode star
    freq, g = None, []
    for Jc in mc.spectral_densities():
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
    """A zero multichannel Lanczos seed is rejected explicitly."""
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sd = lambda w: 0.15 * w * np.exp(-w / 6.0)
    mc = Bath(J=[sd, sd], domain=(0.0, 30.0),
              n_modes=3, phys_dim=4)
    m = SystemBath(h=0.5 * sigma_z, coupling=[sigma_x, sy], bath=mc)
    with pytest.raises(ValueError, match="seed"):
        m.run(dt=0.01, n_steps=1, method="interaction-chain-tebd", bond_dim=20)


def test_composite_spin_vibration_system():
    """System = spin (x) vibration; bath couples only through the spin.  Validated
    vs exact diagonalization of the discretized star."""
    from fishbonett.representations.interaction import InteractionRepresentation as Builder
    from fishbonett.operators import annihilate
    dv, nm, dph = 2, 2, 4
    I2, Iv = np.eye(2), np.eye(dv)
    bv = annihilate(dv); nv = bv.T @ bv
    h_sys = (0.25 * np.kron(sigma_z, Iv) + np.kron(sigma_x, Iv)
             + 1.5 * np.kron(I2, nv) + 0.3 * np.kron(sigma_z, bv + bv.T))
    coup = np.kron(sigma_z, Iv)
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=nm, phys_dim=dph)
    model = SystemBath(h=h_sys, coupling=coup, bath=bath)
    r = model.run(dt=0.02, n_steps=10, method="interaction-chain-tebd",
                  bond_dim=40, trunc_eps=1e-12,
                  observables={"sz": coup}, initial="up")
    assert r.rdm.shape == (10, 2 * dv, 2 * dv)

    builder = Builder(
        representation="interaction-chain",
        h_sys=h_sys, coupling=coup,
        bath=bath,
    ).build()
    freq = builder.frequencies
    j0 = builder.star_couplings
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


@pytest.mark.parametrize("method", ["schrodinger-chain-tdvp1", "schrodinger-chain-tdvp2", "interaction-chain-tdvp1",
                                    "interaction-chain-tree-tebd"])
def test_general_coupling_matches_exact(method):
    """The MPO/tree engines handle a non-sigma_z (sigma_x) coupling, validated vs
    exact diagonalization of the discretized star."""
    sd = lambda w: 0.2 * w * np.exp(-w / 5.0)
    bath = Bath(J=sd, domain=(0.0, 40.0), n_modes=3, phys_dim=5)
    h, O = 0.5 * sigma_z + sigma_x, sigma_x
    r = SystemBath(h=h, coupling=O, bath=bath).run(
        dt=0.02, n_steps=10, method=method, bond_dim=60, trunc_eps=1e-12,
        observables={"sz": sigma_z})
    ex = _exact_general(h, O, sigma_z, r.t, 3, 5, (0.0, 40.0), sd, [1, 0])
    assert np.max(np.abs(r.expect["sz"] - ex)) < 3e-3


@pytest.mark.parametrize("method", ["schrodinger-chain-tdvp1", "schrodinger-chain-tdvp2",
                                    "interaction-chain-tree-tebd"])
def test_multilevel_system_matches_exact(method):
    """The MPO/tree engines handle a three-level system, validated vs exact."""
    sd = lambda w: 0.15 * w * np.exp(-w / 6.0)
    bath = Bath(J=sd, domain=(0.0, 30.0), n_modes=3, phys_dim=5)
    a3 = annihilate(3)
    h = np.diag([0.0, 0.8, 1.7]) + 0.3 * (a3 + a3.T)
    O = a3 + a3.T
    n3 = np.diag([0.0, 1.0, 2.0])
    r = SystemBath(h=h, coupling=O, bath=bath).run(
        dt=0.02, n_steps=10, method=method, bond_dim=60, trunc_eps=1e-12,
        observables={"n": n3}, initial=[1, 0, 0])
    ex = _exact_general(h, O, n3, r.t, 3, 5, (0.0, 30.0), sd, [1, 0, 0])
    assert np.max(np.abs(r.expect["n"] - ex)) < 3e-3


def test_mpo_rejects_non_hermitian_operators():
    """The MPO/tree engines still require Hermitian h / coupling of matching dim."""
    bath = Bath(J=_J, domain=(0.0, 40.0), n_modes=N, phys_dim=D)
    with pytest.raises(ValueError):                      # non-Hermitian coupling
        SystemBath(h=sigma_z, coupling=np.array([[0, 1], [0, 0]], complex),
                  bath=bath).run(dt=0.05, n_steps=2, method="schrodinger-chain-tdvp1")
    with pytest.raises(ValueError):                      # coupling / h dim mismatch
        SystemBath(h=np.eye(3), coupling=sigma_z, bath=bath).run(
            dt=0.05, n_steps=2, method="interaction-chain-tree-tebd")


# -- polaron representation -----------------------------------------------------------
def _polaron_bath(nm=14, d=8):
    """T=0, gapped-domain bath so J(w)/w^2 is integrable (polaron precondition)."""
    return Bath(J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
                n_modes=nm, phys_dim=d)


def test_swap_network_walks_the_system_out_from_site_0_and_back():
    """Pin the swap-network direction to the site-0 convention.

    The swap gate exchanges its two sites, so the physical *dimensions* move with
    the content. With unequal system and boson dimensions, this makes the system's
    outward path and return to site 0 directly observable.
    """
    from fishbonett import Bath
    from fishbonett.evolve import tebd
    from fishbonett.representations.interaction import InteractionRepresentation
    from fishbonett.states.mps import SystemBathMPS

    d_sys, d_bos, n = 2, 5, 4
    pd = [d_sys] + [d_bos] * n
    bath = Bath(
        J=_J, domain=(0.0, 40.0), n_modes=n,
        phys_dim=d_bos)
    builder = InteractionRepresentation(
        representation="interaction-chain", h_sys=sigma_x,
        coupling=sigma_z, bath=bath).build()

    def sys_site(st):
        dims = [b.shape[1] for b in st.B]
        assert dims.count(d_sys) == 1, dims
        return dims.index(d_sys)

    state = SystemBathMPS(pd)
    u1, _ = builder.tebd_gates(0.0, 0.01)
    state.U = u1
    assert sys_site(state) == 0                       # system starts at site 0
    tebd.swap_out(state, n, 40, 1e-10)
    assert sys_site(state) == n - 1                   # ... walked to the far end
    tebd.update_bond(state, n - 1, 40, 1e-10, swap=0)
    assert sys_site(state) == n - 1                   # swap=0 does not move it
    _, u2 = builder.tebd_gates(0.005, 0.01)
    state.U = u2                                      # sites are now reversed
    tebd.swap_in(state, n, 40, 1e-10)
    assert sys_site(state) == 0                       # ... and back to site 0

    # the whole step must be layout-preserving, or step k+1 sees the wrong sites
    state2 = SystemBathMPS(pd)
    tebd.symmetric_swap_step(state2, builder, 0.0, 0.01, n, 40, 1e-10)
    assert sys_site(state2) == 0


def test_trotter_mpo_bond_is_number_of_coupling_eigenvalues():
    """The conditional-displacement propagator is a sum of one product operator per
    eigenvalue of the coupling ``O``, so the MPO bond is exactly that count -- 2 for
    sigma_z, 3 for a three-eigenvalue coupling -- independent of the chain length."""
    from fishbonett import Bath
    from fishbonett.representations.interaction import InteractionRepresentation

    for O, expected in [(sigma_z, 2), (np.diag([1.0, 0.0, -1.0]).astype(complex), 3)]:
        ds = O.shape[0]
        bath = Bath(
            J=_J, domain=(0.3, 12.0), n_modes=5,
            phys_dim=6)
        b = InteractionRepresentation(
            representation="interaction-chain",
            h_sys=np.eye(ds), coupling=O,
            bath=bath).build()
        W = b.trotter_mpo(0.0, 0.05)
        assert len(W) == 6                       # system + 5 modes
        assert W[0].shape == (1, expected, ds, ds)
        assert all(w.shape[0] == expected for w in W[1:])
        assert W[-1].shape[1] == 1               # closed at the right edge


def test_trotter_mpo_matches_tebd_general_coupling():
    """The two interaction-chain methods agree for a general coupling."""
    O = np.diag([1.0, 0.0, -1.0]).astype(complex)
    h = np.zeros((3, 3), complex)
    h[0, 1] = h[1, 0] = h[1, 2] = h[2, 1] = 0.5
    bath = Bath(J=_J, domain=(0.3, 12.0), n_modes=10, phys_dim=8)
    model = SystemBath(h=h, coupling=O, bath=bath)
    kw = dict(dt=0.05, n_steps=20, bond_dim=40, trunc_eps=1e-4, observables={"O": O})
    assert np.max(np.abs(
        model.run(method="interaction-chain-trotter-mpo", **kw).expect["O"]
        - model.run(method="interaction-chain-tebd", **kw).expect["O"]
    )) < 5e-3


POLARON_METHODS = [
    "polaron-chain-tebd", "polaron-chain-tdvp1", "polaron-chain-tdvp2",
    "polaron-chain-dtdvp",
    # The star polaron displaces every mode instead of localizing on c0.  Same
    # physics, so it must reproduce the interaction picture just as the chain
    # one does -- and until this was added the whole representation had no test.
    "polaron-star-tdvp1", "polaron-star-tdvp2", "polaron-star-dtdvp",
]


@pytest.mark.parametrize("method", POLARON_METHODS)
def test_polaron_matches_ip_populations_and_coherence(method):
    """Every polaron propagator reproduces the interaction-picture chain for a
    2-level spin-boson: the representation-invariant population <sz> and the *un-dressed*
    coherence <sx> both agree (they differ only by the O(dt^2) Trotter split).
    The TEBD variant uses static gates; the TDVP variants use the polaron MPO."""
    model = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=_polaron_bath(nm=10, d=8))
    kw = dict(dt=0.02, n_steps=25, bond_dim=16, trunc_eps=1e-4,
              observables={"sz": sigma_z, "sx": sigma_x})
    rp = model.run(method=method, **kw)
    ri = model.run(method="interaction-chain-tebd", **kw)
    # The two representations describe the same physics, so they agree far better than the
    # O(dt^2) splitting bound: ~1e-5 on the population and ~1e-4 on the un-dressed
    # coherence.  Keep the tolerance tight -- a loose one here once hid a misplaced
    # on-site frequency in the polaron gates.
    assert np.max(np.abs(rp.expect["sz"] - ri.expect["sz"])) < 1e-4   # populations
    assert np.max(np.abs(rp.expect["sx"] - ri.expect["sx"])) < 1e-3   # un-dressed


@pytest.mark.parametrize("method", ["polaron-chain-tebd", "polaron-chain-dtdvp"])
def test_polaron_general_coupling_matches_ip(method):
    """The polaron representation handles a general (3-level, three-eigenvalue) coupling O."""
    O = np.diag([1.0, 0.0, -1.0]).astype(complex)
    h = np.zeros((3, 3), complex)
    h[0, 1] = h[1, 0] = h[1, 2] = h[2, 1] = 0.5          # off-diagonal in O's eigenbasis
    model = SystemBath(h=h, coupling=O, bath=_polaron_bath(nm=10, d=8))
    kw = dict(dt=0.02, n_steps=25, bond_dim=16, trunc_eps=1e-4, observables={"O": O})
    rp = model.run(method=method, **kw)
    ri = model.run(method="interaction-chain-tebd", **kw)
    assert np.max(np.abs(rp.expect["O"] - ri.expect["O"])) < 2e-2


def test_polaron_runs_at_finite_temperature():
    """Finite-T polaron propagation works when its IR norm is finite."""
    super_ohmic = lambda w: 0.02 * w**3 * np.exp(-w / 5.0)
    bath = Bath(
        J=super_ohmic, domain=(-12.0, 12.0), temperature=1.0,
        n_modes=8, phys_dim=6,
    )
    r = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=bath).run(
            method="polaron-chain-tebd", dt=0.05, n_steps=2, bond_dim=20)
    assert r.t.shape == (2,)
    assert np.allclose(np.trace(r.rdm[-1]), 1.0, atol=1e-6)


def test_free_chain_gates_put_each_frequency_on_its_own_mode():
    """Free-chain bond ``m`` must carry ``k_list[m]`` hopping *and* ``w_list[m]``
    on ``c_m`` -- its **right** leg, because with the system at site 0 the chain
    runs outward.

    Assigning ``w_list[m]`` to the left leg shifts each frequency to a neighbouring
    mode without violating normalization, so this test checks the gate structure
    directly.
    """
    import scipy.linalg as sla
    from fishbonett import Bath
    from fishbonett.representations.polaron import PolaronRepresentation
    from fishbonett.operators import annihilate

    nb, d = 4, 5
    bath = Bath(
        J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
        n_modes=nb, phys_dim=d)
    b = PolaronRepresentation(
        representation="polaron-chain", h_sys=0.5 * sigma_x,
        coupling=sigma_z, bath=bath).build()

    dt = 1e-5                       # small dt so i log(U)/dt recovers h faithfully
    a = annihilate(d)
    num, Id = a.conj().T @ a, np.eye(d)
    gates = b.tebd_gates(dt)
    for m in range(1, nb):          # bond 0 is the dressed bond, checked elsewhere
        want = (b.hoppings[m - 1] * (np.kron(a.conj().T, a) + np.kron(a, a.conj().T))
                + b.frequencies[m] * np.kron(Id, num))     # w_m on c_m (right leg)
        got = 1j * sla.logm(gates[m].reshape(d * d, d * d)) / dt
        assert np.allclose(got, want, atol=1e-4), (
            f"bond {m}: gate Hamiltonian does not match "
            f"k={b.hoppings[m - 1]:.4f} hopping + "
            f"w={b.frequencies[m]:.4f} on c_{m}")


def test_two_site_tdvp_bond_grows_at_the_default_threshold():
    """Two-site TDVP grows entanglement at the default truncation threshold.

    A product-state step can create Schmidt values below the threshold. Retaining
    a small expansion space prevents repeated truncation from locking the bond at
    one before those components accumulate.
    """
    from fishbonett.linalg import DEFAULT_EPS

    model = _model()
    r = model.run(dt=0.05, n_steps=10, method="interaction-chain-tdvp2",
                  bond_dim=40, trunc_eps=DEFAULT_EPS,
                  observables={"sz": sigma_z})
    sz_exact = _exact_sz(model.bath, r.t)
    err = np.max(np.abs(r.expect["sz"] - sz_exact))
    assert err < 5e-3, f"tdvp2 at the default trunc_eps is off by {err:.2e}"
    assert int(np.max(r.max_bond)) > 1, "the bond never grew past a product state"

    # ...and it must agree with the two independent propagators of the same H(t)
    for other in ("interaction-chain-trotter-mpo", "interaction-chain-tebd"):
        r2 = _model().run(dt=0.05, n_steps=10, method=other, bond_dim=40,
                          trunc_eps=DEFAULT_EPS, observables={"sz": sigma_z})
        gap = np.max(np.abs(np.asarray(r.expect["sz"])
                            - np.asarray(r2.expect["sz"])))
        assert gap < 5e-3, f"tdvp2 disagrees with {other} by {gap:.2e}"


def test_bond_expansion_allowance_is_bounded():
    """The growth allowance must not inflate the bond without limit.

    It admits a fixed few directions beyond the threshold, so the bond settles
    at (threshold rank + expand) rather than creeping to ``bond_dim``.
    """
    from fishbonett.evolve._tdvp_sweeps import DEFAULT_BOND_EXPAND

    model = _model()
    loose = model.run(dt=0.05, n_steps=10, method="interaction-chain-tdvp2",
                      bond_dim=40, trunc_eps=1e-2,
                      observables={"sz": sigma_z})
    # far below the cap: the allowance is additive, not a floor at bond_dim
    assert int(np.max(loose.max_bond)) <= 4 + 2 * DEFAULT_BOND_EXPAND
    # explicitly disabling it reproduces the old locked-at-1 behaviour
    stuck = _model().run(dt=0.05, n_steps=10, method="interaction-chain-tdvp2",
                         bond_dim=40, trunc_eps=1e-2,
                         observables={"sz": sigma_z}, bond_expand=0)
    assert int(np.max(stuck.max_bond)) == 1


def test_dynamic_tdvp_honours_the_bond_expansion_allowance():
    """Dynamic TDVP passes ``bond_expand`` to its two-site sweep."""
    grown = _model().run(dt=0.05, n_steps=8, method="schrodinger-chain-dtdvp",
                         bond_dim=40, trunc_eps=1e-3,
                         observables={"sz": sigma_z})
    off = _model().run(dt=0.05, n_steps=8, method="schrodinger-chain-dtdvp",
                       bond_dim=40, trunc_eps=1e-3, bond_expand=0,
                       observables={"sz": sigma_z})
    assert int(np.max(grown.max_bond)) > int(np.max(off.max_bond)), (
        "bond_expand does not reach the dtdvp sweep")
