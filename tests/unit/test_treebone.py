"""Tests for the general tree-topology engine (TreeTensorNetwork) and TreeFishbone."""
import numpy as np
import pytest
from scipy.linalg import expm

from fishbonett.states.tree import TreeTensorNetwork
from fishbonett.models.fishbone import TreeFishbone
from fishbonett import Bath, Result
from fishbonett.operators import sigma_x, sigma_z

DOM = (0.0, 40.0)


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _bath(nm, d, op):
    return Bath(J=_J, domain=DOM, n_modes=nm, phys_dim=d, coupling=op)


def _embed(op, s, dims):
    m = [np.eye(x) for x in dims]
    m[s] = op
    o = m[0]
    for x in m[1:]:
        o = np.kron(o, x)
    return o


def _embed2(op4, sa, sb, dims):
    da, db = dims[sa], dims[sb]
    C = np.transpose(op4, (0, 2, 1, 3)).reshape(da * da, db * db)
    U, S, Vh = np.linalg.svd(C, full_matrices=False)
    out = np.zeros((int(np.prod(dims)),) * 2, complex)
    for r in range(int(np.sum(S > 1e-12))):
        L = (U[:, r] * S[r]).reshape(da, da)
        R = Vh[r].reshape(db, db)
        ops = [np.eye(d) for d in dims]
        ops[sa] = L; ops[sb] = R
        o = ops[0]
        for x in ops[1:]:
            o = np.kron(o, x)
        out = out + o
    return out


def _evolve_sz(dims, H, ts, nsites):
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(H.shape[0], complex); p0[0] = 1
    c = U.conj().T @ p0
    out = []
    for i in range(nsites):
        sz = _embed(sigma_z, i, dims)
        out.append(np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                             @ (sz @ (U @ (np.exp(-1j * E * t) * c))) for t in ts]).real)
    return np.array(out).T


def test_treetebd_star_matches_exact():
    """A star (non-1D) of four spins vs exact diagonalization."""
    rng = np.random.default_rng(1)
    n = 4
    dims = [2] * n
    edges = [(0, 1), (0, 2), (0, 3)]
    hs = [0.5 * (lambda A: A + A.conj().T)(rng.standard_normal((2, 2))
          + 1j * rng.standard_normal((2, 2))) for _ in range(n)]
    Je = {e: 0.3 + 0.1 * k for k, e in enumerate(edges)}
    H = sum(_embed(hs[i], i, dims) for i in range(n))
    for (a, b) in edges:
        H = H + Je[(a, b)] * (_embed(sigma_z, a, dims) @ _embed(sigma_z, b, dims))
    dt, ns = 0.02, 12
    st = TreeTensorNetwork(dims, edges, root=0)
    # `step` is a symmetric (Strang) step and applies every gate twice, so it takes
    # **half**-step gates -- same convention as evolve.tebd.symmetric_static_step.
    site_gates = [expm(-1j * hs[i] * dt / 2) for i in range(n)]
    edge_gates = {e: expm(-1j * Je[e] * np.kron(sigma_z, sigma_z) * dt / 2).reshape(2, 2, 2, 2)
                  for e in edges}
    sz = []
    for _ in range(ns):
        st.step(site_gates, edge_gates, 32, 1e-12)
        sz.append([np.trace(st.rdm(i) @ sigma_z).real for i in range(n)])
        st.move_oc_to(0)
    ex = _evolve_sz(dims, H, np.arange(1, ns + 1) * dt, n)
    # A star is the shape where a naive down-and-up Euler tour fails to reach second
    # order, so this geometry is worth holding to a tight bound.
    assert np.max(np.abs(np.array(sz) - ex)) < 1e-4


def _fb_exact(fb, ts):
    dims, edges, site_H, edge_H = fb.hamiltonians()
    tot = int(np.prod(dims))
    H = np.zeros((tot, tot), complex)
    for i, h in enumerate(site_H):
        if np.any(h):
            H += _embed(h, i, dims)
    for (a, b), C in edge_H.items():
        da, db = dims[a], dims[b]
        H += _embed2(C.reshape(da, db, da, db), a, b, dims)
    return _evolve_sz(dims, H, ts, fb.ns)


def test_treefishbone_1d_matches_exact():
    fb = TreeFishbone(
        sites=[0.25 * sigma_z + 0.8 * sigma_x, -0.15 * sigma_z + 0.8 * sigma_x],
        edges=[(0, 1, 0.4 * np.kron(sigma_z, sigma_z))],
        baths=[_bath(2, 4, sigma_z), _bath(2, 4, sigma_z)])
    r = fb.run(dt=0.02, n_steps=12, bond_dim=40, trunc_eps=1e-12,
               observables={"sz": sigma_z})
    assert isinstance(r, Result)
    assert r.expect["sz"].shape == (12, 2)
    assert np.max(np.abs(r.expect["sz"] - _fb_exact(fb, r.t))) < 1e-3


def test_treefishbone_star_matches_exact():
    """Three electronic sites in a Y, each with a bath (genuinely non-1D)."""
    C = 0.3 * np.kron(sigma_z, sigma_z)
    fb = TreeFishbone(
        sites=[0.2 * sigma_z + 0.6 * sigma_x, -0.1 * sigma_z + 0.5 * sigma_x,
               0.15 * sigma_z + 0.5 * sigma_x],
        edges=[(0, 1, C), (0, 2, C)],
        baths=[_bath(2, 2, sigma_z), _bath(2, 2, sigma_x), _bath(2, 2, sigma_z)])
    r = fb.run(dt=0.02, n_steps=10, bond_dim=40, trunc_eps=1e-12,
               observables={"sz": sigma_z})
    assert r.rdm.shape == (10, 3, 2, 2)
    assert np.max(np.abs(r.expect["sz"] - _fb_exact(fb, r.t))) < 5e-3


def test_composite_spin_vibration_separate_sites():
    """Spin (dim 2) and vibration (dim dv) on their OWN sites, bath only on the
    spin.  Validated vs exact; also checks mixed-dimension per-site output."""
    from fishbonett.operators import annihilate
    dv = 3
    bvi = annihilate(dv); nv = bvi.T @ bvi
    h_spin = 0.25 * sigma_z + sigma_x
    h_vib = 1.5 * nv
    C = 0.4 * np.kron(sigma_z, bvi + bvi.T)              # spin-vibration coupling
    fb = TreeFishbone(sites=[h_spin, h_vib], edges=[(0, 1, C)],
                      baths=[_bath(2, 4, sigma_z), None])
    r = fb.run(dt=0.02, n_steps=10, bond_dim=40, trunc_eps=1e-12,
               observables={"sz": sigma_z})
    # per-site RDMs keep their own dimension; sz only defined on the 2-level spin
    assert r.rdm[0, 0].shape == (2, 2) and r.rdm[0, 1].shape == (dv, dv)
    assert np.all(np.isfinite(r.expect["sz"][:, 0]))     # spin site: defined
    assert np.all(np.isnan(r.expect["sz"][:, 1]))        # vib site: sz N/A -> NaN

    dims, edges, site_H, edge_H = fb.hamiltonians()
    tot = int(np.prod(dims))

    def emb(op, s):
        m = [np.eye(x) for x in dims]; m[s] = op
        o = m[0]
        for x in m[1:]:
            o = np.kron(o, x)
        return o

    H = np.zeros((tot, tot), complex)
    for i, h in enumerate(site_H):
        if np.any(h):
            H += emb(h, i)
    for (a, b), Ce in edge_H.items():
        H += _embed2(Ce.reshape(dims[a], dims[b], dims[a], dims[b]), a, b, dims)
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(tot, complex); p0[0] = 1
    c = U.conj().T @ p0
    szs = emb(sigma_z, 0)
    sz_ex = np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                      @ (szs @ (U @ (np.exp(-1j * E * t) * c))) for t in r.t]).real
    assert np.max(np.abs(r.expect["sz"][:, 0] - sz_ex)) < 1e-3


def test_multichannel_single_bath():
    """One bath coupled to a spin through BOTH sigma_z and sigma_x (shared modes),
    with different per-channel spectral densities.  Validated vs exact."""
    def Jz(w):
        return 0.20 * w * np.exp(-w / 5.0)

    def Jx(w):
        return 0.10 * w * np.exp(-w / 8.0)

    mc = Bath(J=[Jz, Jx], coupling=[sigma_z, sigma_x],
              domain=(0.0, 40.0), n_modes=3, phys_dim=4)
    assert mc.is_multichannel
    h = 0.3 * sigma_z + 0.8 * sigma_x
    fb = TreeFishbone(sites=[h], edges=[], baths=[mc])
    r = fb.run(dt=0.02, n_steps=12, bond_dim=40, trunc_eps=1e-12,
               observables={"sz": sigma_z})
    dims, edges, site_H, edge_H = fb.hamiltonians()
    assert dims[0] == 2 and len(dims) == 1 + 3       # spin + 3 SHARED modes (star)
    tot = int(np.prod(dims))

    def emb(op, s):
        m = [np.eye(x) for x in dims]; m[s] = op
        o = m[0]
        for x in m[1:]:
            o = np.kron(o, x)
        return o

    H = np.zeros((tot, tot), complex)
    for i, h_ in enumerate(site_H):
        if np.any(h_):
            H += emb(h_, i)
    for (a, b), Ce in edge_H.items():
        H += _embed2(Ce.reshape(dims[a], dims[b], dims[a], dims[b]), a, b, dims)
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(tot, complex); p0[0] = 1
    c = U.conj().T @ p0
    szs = emb(sigma_z, 0)
    sz_ex = np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                      @ (szs @ (U @ (np.exp(-1j * E * t) * c))) for t in r.t]).real
    assert np.max(np.abs(r.expect["sz"][:, 0] - sz_ex)) < 2e-3


def test_general_observables_single_and_two_site():
    """Observable spec: bare op (per-site), (op, i) single-site, (op, [i, j])
    composite -- validated vs exact including a two-site correlation."""
    C = 0.4 * np.kron(sigma_z, sigma_z)
    fb = TreeFishbone(sites=[0.3 * sigma_z + sigma_x, -0.2 * sigma_z + sigma_x],
                      edges=[(0, 1, C)], baths=[_bath(2, 4, sigma_z), _bath(2, 4, sigma_z)])
    zz = np.kron(sigma_z, sigma_z)
    r = fb.run(dt=0.02, n_steps=10, bond_dim=40, trunc_eps=1e-12,
               observables={"z0": (sigma_z, 0), "zz": (zz, [0, 1]), "sz": sigma_z})
    assert r.expect["z0"].shape == (10,)          # single site
    assert r.expect["zz"].shape == (10,)          # two-site correlation
    assert r.expect["sz"].shape == (10, 2)        # per-site

    dims, edges, site_H, edge_H = fb.hamiltonians()
    tot = int(np.prod(dims))

    def emb(op, s):
        m = [np.eye(x) for x in dims]; m[s] = op
        o = m[0]
        for x in m[1:]:
            o = np.kron(o, x)
        return o

    H = np.zeros((tot, tot), complex)
    for i, h in enumerate(site_H):
        if np.any(h):
            H += emb(h, i)
    for (a, b), Ce in edge_H.items():
        H += _embed2(Ce.reshape(dims[a], dims[b], dims[a], dims[b]), a, b, dims)
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(tot, complex); p0[0] = 1
    c = U.conj().T @ p0
    zzf = emb(sigma_z, 0) @ emb(sigma_z, 1)
    zz_ex = np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                      @ (zzf @ (U @ (np.exp(-1j * E * t) * c))) for t in r.t]).real
    assert np.max(np.abs(r.expect["zz"] - zz_ex)) < 2e-3


def test_multichannel_star_on_spin_of_spin_vibration_tree():
    """A multichannel bath attached to the spin site of a spin+vibration tree
    (the two features combined)."""
    from fishbonett.operators import annihilate
    dv = 3
    bvi = annihilate(dv); nv = bvi.T @ bvi
    mc = Bath(J=[lambda w: 0.2 * w * np.exp(-w / 5.0),
                 lambda w: 0.1 * w * np.exp(-w / 8.0)],
              coupling=[sigma_z, sigma_x], domain=(0.0, 40.0), n_modes=2, phys_dim=4)
    fb = TreeFishbone(sites=[0.25 * sigma_z + sigma_x, 1.5 * nv],
                      edges=[(0, 1, 0.4 * np.kron(sigma_z, bvi + bvi.T))],
                      baths=[mc, None])
    r = fb.run(dt=0.02, n_steps=8, bond_dim=40, trunc_eps=1e-12,
               observables={"sz_spin": (sigma_z, 0)})
    dims, edges, site_H, edge_H = fb.hamiltonians()
    assert dims[0] == 2 and dims[1] == dv           # spin + vibration on own sites
    tot = int(np.prod(dims))

    def emb(op, s):
        m = [np.eye(x) for x in dims]; m[s] = op
        o = m[0]
        for x in m[1:]:
            o = np.kron(o, x)
        return o

    H = np.zeros((tot, tot), complex)
    for i, h in enumerate(site_H):
        if np.any(h):
            H += emb(h, i)
    for (a, b), Ce in edge_H.items():
        H += _embed2(Ce.reshape(dims[a], dims[b], dims[a], dims[b]), a, b, dims)
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(tot, complex); p0[0] = 1
    c = U.conj().T @ p0
    szf = emb(sigma_z, 0)
    sz_ex = np.array([(U @ (np.exp(-1j * E * t) * c)).conj()
                      @ (szf @ (U @ (np.exp(-1j * E * t) * c))) for t in r.t]).real
    assert np.max(np.abs(r.expect["sz_spin"] - sz_ex)) < 2e-3


def test_multichannel_requires_legendre():
    def Jz(w):
        return 0.2 * w * np.exp(-w / 5.0)
    mc = Bath(J=[Jz, Jz], coupling=[sigma_z, sigma_x], domain=(0.0, 40.0),
              n_modes=2, phys_dim=3, discretization="tedopa")
    with pytest.raises(ValueError):
        TreeFishbone(sites=[sigma_z], edges=[], baths=[mc])


def test_non_tree_edges_raise():
    with pytest.raises(ValueError):
        TreeTensorNetwork([2, 2, 2], [(0, 1), (1, 2), (2, 0)])       # a loop
    with pytest.raises(ValueError):
        TreeFishbone(sites=[sigma_z, sigma_z, sigma_z],
                     edges=[(0, 1)],                          # too few edges (not a tree)
                     baths=[None, None, None])
