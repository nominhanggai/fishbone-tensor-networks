"""Benchmark: tree tensor-network engine (interaction picture).

Validates the tree-MPO against the direct star Hamiltonian, and the tree-TDVP and
tree-TEBD propagators against exact diagonalization of a small spin-boson star,
and shows the log-depth advantage over a chain.

Run with:  python benchmarks/tree_engine.py
"""
import numpy as np

from fishbonett.evolve.modetree import (_star_transform, run_tree_tdvp, run_tree_tdvp2,
                             run_tree_tebd, build_balanced_tree, build_tree_mpo,
                             tree_depth, hamiltonian_from_mpo, _hamiltonian_direct,
                             _resolve_sys, SZ, SX)
# `anih`/`crea` were imported from modetree, which has never defined them -- this
# benchmark has been failing at import since before the model/frame restructure.
# The operators are annihilate/create in fishbonett.operators; the body below uses
# both spellings, so bind both.
from fishbonett.operators import annihilate, create
anih, crea = annihilate, create


def Jb(w):
    aw = abs(w)
    if aw < 1e-12:
        return 0.0
    nb = 1.0 / np.expm1(aw)
    j = 0.2 * aw * np.exp(-aw / 5.0)
    return j * (nb + 1.0) if w > 0 else j * nb


def embed(op, site, dims):
    mats = [np.eye(dm) for dm in dims]
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def exact_sz(n_chain, d, V, ts):
    freq, Vn, _ = _star_transform(Jb, n_chain, (-25.0, 36.0))
    dims = [2] + [d] * n_chain
    H = embed(V * SX, 0, dims)
    for k in range(n_chain):
        H = H + freq[k] * embed(create(d) @ annihilate(d), 1 + k, dims)
        H = H + Vn[k] * (embed(SZ, 0, dims) @ embed(annihilate(d) + create(d), 1 + k, dims))
    E, Uv = np.linalg.eigh(H)
    psi0 = np.zeros(int(np.prod(dims)), dtype=complex)
    psi0[0] = 1.0
    coef = Uv.conj().T @ psi0
    szop = embed(SZ, 0, dims)
    return np.array([(Uv @ (np.exp(-1j * E * t) * coef)).conj()
                     @ (szop @ (Uv @ (np.exp(-1j * E * t) * coef))) for t in ts]).real


def main():
    print("=== tree-MPO vs direct star Hamiltonian ===")
    rng = np.random.default_rng(0)
    V, eps = 0.7, 0.4
    # build_tree_mpo used to take the two-level scalars (V, eps); it takes the
    # system Hamiltonian and coupling *operators* since they were generalized to
    # any dimension.  _resolve_sys builds exactly the pair those scalars meant,
    # which keeps this comparison against _hamiltonian_direct honest.
    Hs, O, _v, _ds = _resolve_sys(None, None, None, V, eps)
    for n in (1, 2, 3, 4, 5):
        dcoup = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        nodes, root, _ = build_balanced_tree(n, 3)
        build_tree_mpo(nodes, root, dcoup, Hs, O)
        H = hamiltonian_from_mpo(nodes, root, n, 3)
        print(f"  n_modes={n}: max|H_mpo - H_direct|="
              f"{np.abs(H - _hamiltonian_direct(dcoup, V, eps, 3, n)).max():.1e}")

    print("\n=== tree dynamics vs exact diagonalization ===")
    n_chain, d, V, dt, nsteps = 3, 6, 1.0, 0.05, 20
    t, sz_t1 = run_tree_tdvp(Jb, (-25.0, 36.0), V=V, n_chain=n_chain, phys_dim=d,
                             dt=dt, nsteps=nsteps, D=40)
    _, sz_t2 = run_tree_tdvp2(Jb, (-25.0, 36.0), V=V, n_chain=n_chain, phys_dim=d,
                              dt=dt, nsteps=nsteps, D=40, trunc_eps=1e-13)
    _, sz_te = run_tree_tebd(Jb, (-25.0, 36.0), V=V, n_chain=n_chain, phys_dim=d,
                             dt=dt, nsteps=nsteps, D=40, trunc_eps=1e-12)
    sz_ex = exact_sz(n_chain, d, V, t)
    print(f"{'t':>6} {'exact':>10} {'TDVP(1s)':>10} {'TDVP(2s)':>10} {'TEBD':>10}")
    for i in range(0, nsteps, 4):
        print(f"{t[i]:>6.2f} {sz_ex[i]:>+10.5f} {sz_t1[i]:>+10.5f} {sz_t2[i]:>+10.5f} {sz_te[i]:>+10.5f}")
    print(f"max|tree-TDVP(1-site) - exact| = {np.max(np.abs(sz_t1 - sz_ex)):.2e}")
    print(f"max|tree-TDVP(2-site) - exact| = {np.max(np.abs(sz_t2 - sz_ex)):.2e}")
    print(f"max|tree-TEBD        - exact| = {np.max(np.abs(sz_te - sz_ex)):.2e}")

    print("\n=== tree depth vs chain length ===")
    for n in (16, 64, 256, 600):
        nodes, root, _ = build_balanced_tree(n, 3)
        print(f"  n_modes={n:4d}: tree depth={tree_depth(nodes, root):2d}  vs chain length={n + 1}")


if __name__ == "__main__":
    main()
