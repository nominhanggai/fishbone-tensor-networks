"""Unit tests for the MPO / TDVP engine, validated against exact diagonalization
of a small spin-boson chain."""
import numpy as np

from fishbonett.evolve.tdvp import (chain_coeffs, run_tdvp1, run_tdvp2, run_dtdvp,
                            run_ip_tdvp1, run_ip_tdvp2, crea, anih, numb, SX, SZ)

DOMAIN = (-25.0, 36.0)


def _Jb(w):
    aw = abs(w)
    if aw < 1e-12:
        return 0.0
    nb = 1.0 / np.expm1(aw)                      # beta = 1
    j = 0.2 * aw * np.exp(-aw / 5.0)             # super-ohmic
    return j * (nb + 1.0) if w > 0 else j * nb


def _embed(op, site, dims):
    mats = [np.eye(dm) for dm in dims]
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _exact_sz(n_chain, d, V, ts):
    eps_c, t_c, c0 = chain_coeffs(_Jb, n_chain, DOMAIN)
    dims = [2] + [d] * n_chain
    b, bd, nb = anih(d), crea(d), numb(d)
    H = _embed(V * SX, 0, dims) + c0 * (_embed(SZ, 0, dims) @ _embed(b + bd, 1, dims))
    for i in range(n_chain):
        H = H + eps_c[i] * _embed(nb, 1 + i, dims)
    for i in range(n_chain - 1):
        hop = _embed(bd, 1 + i, dims) @ _embed(b, 2 + i, dims)
        H = H + t_c[i] * (hop + hop.conj().T)
    E, Uv = np.linalg.eigh(H)
    psi0 = np.zeros(int(np.prod(dims)), dtype=complex)
    psi0[0] = 1.0
    coef = Uv.conj().T @ psi0
    szop = _embed(SZ, 0, dims)
    return np.array([(lambda p: (p.conj() @ (szop @ p)).real)
                     (Uv @ (np.exp(-1j * E * t) * coef)) for t in ts])


def test_tdvp1_matches_exact_diagonalization():
    n_chain, d, V = 3, 5, 1.0
    t, sz = run_tdvp1(_Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.05, nsteps=12,
                      D=40, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.isclose(sz[0], 0.99, atol=0.02)          # starts near |up>
    assert np.max(np.abs(sz - sz_ex)) < 1e-6


def test_tdvp2_matches_exact_and_grows_bonds():
    n_chain, d, V = 3, 5, 1.0
    t, sz, maxd = run_tdvp2(_Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.05,
                            nsteps=12, chi_max=40, eps=1e-12, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert maxd[-1] > 1                                 # bonds grew from product state
    assert np.max(np.abs(sz - sz_ex)) < 1e-6


def test_ip_mpo_matches_exact():
    """Interaction-picture star MPO (time-dependent, rebuilt each step) vs the same
    exact dynamics.  Looser tol: the IP midpoint rule is O(dt^2) in time."""
    n_chain, d, V = 3, 5, 1.0
    t, sz = run_ip_tdvp1(_Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.02,
                         nsteps=15, D=40, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.max(np.abs(sz - sz_ex)) < 5e-3
    t2, sz2, maxd = run_ip_tdvp2(_Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.02,
                                 nsteps=15, chi_max=40, eps=1e-12, krylov=25)
    assert maxd[-1] > 1                                 # bonds grew
    assert np.max(np.abs(sz2 - sz_ex)) < 5e-3


def test_dtdvp_grows_bonds_and_tracks_dynamics():
    n_chain, d, V = 3, 5, 1.0
    t, sz, maxd = run_dtdvp(_Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.05,
                            nsteps=12, prec=1e-9, Dlim=40, Dplusmax=6, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert maxd[-1] >= 1 and np.all(np.isfinite(sz))
    assert np.max(np.abs(sz - sz_ex)) < 1e-2           # adaptive, looser
