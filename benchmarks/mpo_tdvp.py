"""Benchmark: MPO / TDVP engine vs exact diagonalization.

Propagates <sigma_z(t)> of a small spin-boson chain with the fixed-bond 1-site
TDVP and the bond-adaptive DTDVP engines, and compares against exact
diagonalization of the full chain Hamiltonian (feasible for a few modes).

Run with:  python benchmarks/mpo_tdvp.py
"""
import numpy as np

from fishbonett.mpo import (chain_coeffs, run_tdvp1, run_tdvp2, run_dtdvp, crea,
                            anih, numb, SX, SZ)

DOMAIN = (-25.0, 36.0)


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
    eps_c, t_c, c0 = chain_coeffs(Jb, n_chain, DOMAIN)
    dims = [2] + [d] * n_chain
    b, bd, nb = anih(d), crea(d), numb(d)
    H = embed(V * SX, 0, dims) + c0 * (embed(SZ, 0, dims) @ embed(b + bd, 1, dims))
    for i in range(n_chain):
        H = H + eps_c[i] * embed(nb, 1 + i, dims)
    for i in range(n_chain - 1):
        hop = embed(bd, 1 + i, dims) @ embed(b, 2 + i, dims)
        H = H + t_c[i] * (hop + hop.conj().T)
    E, Uv = np.linalg.eigh(H)
    psi0 = np.zeros(int(np.prod(dims)), dtype=complex)
    psi0[0] = 1.0
    coef = Uv.conj().T @ psi0
    szop = embed(SZ, 0, dims)
    return np.array([(Uv @ (np.exp(-1j * E * t) * coef)).conj()
                     @ (szop @ (Uv @ (np.exp(-1j * E * t) * coef))) for t in ts]).real


def main():
    n_chain, d, V = 3, 6, 1.0
    t, sz1 = run_tdvp1(Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.05, nsteps=30,
                       D=40, krylov=25)
    _, sz2, md2 = run_tdvp2(Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.05,
                            nsteps=30, chi_max=40, eps=1e-12, krylov=25)
    _, szd, maxd = run_dtdvp(Jb, DOMAIN, V=V, n_chain=n_chain, d=d, dt=0.05,
                             nsteps=30, prec=1e-8, Dlim=40, Dplusmax=6, krylov=25)
    sz_ex = exact_sz(n_chain, d, V, t)
    print(f"{'t':>6} {'exact':>10} {'TDVP1':>10} {'TDVP2':>10} {'DTDVP':>10}")
    for i in range(0, len(t), 5):
        print(f"{t[i]:>6.2f} {sz_ex[i]:>+10.5f} {sz1[i]:>+10.5f} {sz2[i]:>+10.5f} {szd[i]:>+10.5f}")
    print(f"max|TDVP1 - exact| = {np.max(np.abs(sz1 - sz_ex)):.2e}   (fixed bond)")
    print(f"max|TDVP2 - exact| = {np.max(np.abs(sz2 - sz_ex)):.2e}   (two-site, maxD={md2[-1]})")
    print(f"max|DTDVP - exact| = {np.max(np.abs(szd - sz_ex)):.2e}   (adaptive, maxD={maxd[-1]})")


if __name__ == "__main__":
    main()
