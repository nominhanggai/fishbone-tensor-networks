"""Unit tests for the tree tensor-network engine, validated against exact
diagonalization of the star Hamiltonian (interaction- and Schroedinger-picture
<sigma_z> agree because sigma_z commutes with the bath)."""
import numpy as np

from fishbonett.tree import (_star_transform, run_tree_tdvp, run_tree_tebd,
                             build_balanced_tree, build_tree_mpo, tree_depth,
                             hamiltonian_from_mpo, _hamiltonian_direct,
                             anih, crea, SZ, SX)

DOMAIN = (-25.0, 36.0)


def _Jb(w):
    aw = abs(w)
    if aw < 1e-12:
        return 0.0
    nb = 1.0 / np.expm1(aw)
    j = 0.2 * aw * np.exp(-aw / 5.0)
    return j * (nb + 1.0) if w > 0 else j * nb


def _embed(op, site, dims):
    mats = [np.eye(dm) for dm in dims]
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _exact_sz(n_chain, d, V, ts):
    freq, Vn, _ = _star_transform(_Jb, n_chain, DOMAIN)
    dims = [2] + [d] * n_chain
    H = _embed(V * SX, 0, dims)
    for k in range(n_chain):
        H = H + freq[k] * _embed(crea(d) @ anih(d), 1 + k, dims)
        H = H + Vn[k] * (_embed(SZ, 0, dims) @ _embed(anih(d) + crea(d), 1 + k, dims))
    E, Uv = np.linalg.eigh(H)
    psi0 = np.zeros(int(np.prod(dims)), dtype=complex)
    psi0[0] = 1.0
    coef = Uv.conj().T @ psi0
    szop = _embed(SZ, 0, dims)
    return np.array([(lambda p: (p.conj() @ (szop @ p)).real)
                     (Uv @ (np.exp(-1j * E * t) * coef)) for t in ts])


def test_tree_mpo_reproduces_direct_hamiltonian():
    rng = np.random.default_rng(0)
    for n_modes in (1, 2, 3, 4):
        d = 3
        dcoup = rng.standard_normal(n_modes) + 1j * rng.standard_normal(n_modes)
        nodes, root, _ = build_balanced_tree(n_modes, d)
        build_tree_mpo(nodes, root, dcoup, 0.7, 0.4)
        Hmpo = hamiltonian_from_mpo(nodes, root, n_modes, d)
        Hdir = _hamiltonian_direct(dcoup, 0.7, 0.4, d, n_modes)
        assert np.abs(Hmpo - Hdir).max() < 1e-12
        assert np.abs(Hmpo - Hmpo.conj().T).max() < 1e-12


def test_tree_is_shallower_than_chain():
    nodes, root, _ = build_balanced_tree(64, d=3)
    assert tree_depth(nodes, root) < 64        # log-depth, vs chain length 65


def test_tree_tdvp_matches_exact():
    n_chain, d, V = 3, 5, 1.0
    t, sz = run_tree_tdvp(_Jb, DOMAIN, V=V, n_chain=n_chain, phys_dim=d, dt=0.05,
                          nsteps=12, D=30, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.isclose(sz[0], 0.995, atol=0.01)
    assert np.max(np.abs(sz - sz_ex)) < 5e-3   # 2nd-order Trotter at dt=0.05


def test_tree_tebd_matches_exact():
    n_chain, d, V = 3, 5, 1.0
    t, sz = run_tree_tebd(_Jb, DOMAIN, V=V, n_chain=n_chain, phys_dim=d, dt=0.05,
                          nsteps=12, D=30, trunc_eps=1e-12)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.max(np.abs(sz - sz_ex)) < 5e-3
