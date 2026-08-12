"""Tests for the general tree-topology engine (TreeTEBD) and TreeFishbone."""
import numpy as np
import pytest
from scipy.linalg import expm

from fishbonett.treebone import TreeTEBD, TreeFishbone
from fishbonett.simulate import Bath, Result
from fishbonett.stuff import sigma_x, sigma_z

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
    st = TreeTEBD(dims, edges, root=0)
    site_gates = [expm(-1j * hs[i] * dt) for i in range(n)]
    edge_gates = {e: expm(-1j * Je[e] * np.kron(sigma_z, sigma_z) * dt).reshape(2, 2, 2, 2)
                  for e in edges}
    sz = []
    for _ in range(ns):
        st.step(site_gates, edge_gates, 32, 1e-12)
        sz.append([np.trace(st.rdm(i) @ sigma_z).real for i in range(n)])
        st.move_oc_to(0)
    ex = _evolve_sz(dims, H, np.arange(1, ns + 1) * dt, n)
    assert np.max(np.abs(np.array(sz) - ex)) < 1e-3


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


def test_non_tree_edges_raise():
    with pytest.raises(ValueError):
        TreeTEBD([2, 2, 2], [(0, 1), (1, 2), (2, 0)])       # a loop
    with pytest.raises(ValueError):
        TreeFishbone(sites=[sigma_z, sigma_z, sigma_z],
                     edges=[(0, 1)],                          # too few edges (not a tree)
                     baths=[None, None, None])
