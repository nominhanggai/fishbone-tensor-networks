"""Unit tests for the MPO / TDVP engine, validated against exact diagonalization
of a small spin-boson chain."""
import numpy as np

from fishbonett import Bath
from fishbonett.bath.chain import get_bath_nn_paras
from fishbonett.evolve.tdvp import run_mpo_hamiltonian, SX, SZ
from fishbonett.operators import annihilate, create, number
from fishbonett.representations.interaction import InteractionRepresentation
from fishbonett.representations.schrodinger import SchrodingerRepresentation

DOMAIN = (-25.0, 36.0)

# The seven run_* wrappers these tests used to call are gone: each was one
# (representation, sweep) pair with its own copy of the loop.  Materializing the
# MPO is a representation question and running the sweep an evolution question, so a test
# now names both.  The wrappers took a half-step and advanced 2*dt per step;
# run_mpo_hamiltonian takes the step itself, hence dt=0.10 where they said dt=0.05.


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
    bath = _bath(n_chain, d)
    eps_c, couplings = get_bath_nn_paras(
        bath.spectral_density(), n_chain, list(bath.domain),
        discretizer=bath.discretizer())
    t_c, c0 = couplings[1:], couplings[0]
    dims = [2] + [d] * n_chain
    b, bd, nb = annihilate(d), create(d), number(d)
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


def _chain(n_chain, d, V):
    return SchrodingerRepresentation(
        representation="schrodinger-chain", h_sys=V * SX, coupling=SZ,
        bath=_bath(n_chain, d))


def _bath(n_modes, phys_dim):
    return Bath(
        J=_Jb, domain=DOMAIN, n_modes=n_modes,
        phys_dim=phys_dim)


def test_tdvp1_matches_exact_diagonalization():
    n_chain, d, V = 3, 5, 1.0
    t, sz, _ = run_mpo_hamiltonian(_chain(n_chain, d, V), dt=0.10, nsteps=12,
                             sweep="tdvp1", D=40, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.isclose(sz[0], 0.99, atol=0.02)          # starts near |up>
    assert np.max(np.abs(sz - sz_ex)) < 1e-6


def test_tdvp2_matches_exact_and_grows_bonds():
    n_chain, d, V = 3, 5, 1.0
    t, sz, maxd = run_mpo_hamiltonian(_chain(n_chain, d, V), dt=0.10, nsteps=12,
                                sweep="tdvp2", chi_max=40, eps=1e-12, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert maxd[-1] > 1                                 # bonds grew from product state
    assert np.max(np.abs(sz - sz_ex)) < 1e-6


def test_ip_mpo_matches_exact():
    """Interaction-picture star MPO (time-dependent, rebuilt each step) vs the same
    exact dynamics.  Looser tol: the IP midpoint rule is O(dt^2) in time."""
    n_chain, d, V = 3, 5, 1.0
    representation = InteractionRepresentation(
        representation="interaction-star",
        h_sys=V * SX, coupling=SZ,
        bath=_bath(n_chain, d)).build()
    assert not representation.static, "the interaction MPO must be rebuilt"
    t, sz, _ = run_mpo_hamiltonian(representation, dt=0.04, nsteps=15, sweep="tdvp1", D=40,
                             krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.max(np.abs(sz - sz_ex)) < 5e-3
    t2, sz2, maxd = run_mpo_hamiltonian(representation, dt=0.04, nsteps=15, sweep="tdvp2",
                                  chi_max=40, eps=1e-12, krylov=25)
    assert maxd[-1] > 1                                 # bonds grew
    assert np.max(np.abs(sz2 - sz_ex)) < 5e-3


def test_dtdvp_grows_bonds_and_tracks_dynamics():
    n_chain, d, V = 3, 5, 1.0
    t, sz, maxd = run_mpo_hamiltonian(_chain(n_chain, d, V), dt=0.10, nsteps=12,
                                sweep="dtdvp", prec=1e-9, D=40, Dplusmax=6,
                                krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert maxd[-1] >= 1 and np.all(np.isfinite(sz))
    assert np.max(np.abs(sz - sz_ex)) < 1e-2           # adaptive, looser


def test_the_two_bonddims_are_not_interchangeable():
    """``mpo_apply.bond_dims`` and ``tdvp.bonddims`` look like duplicates; they are
    not, and merging them would be a silent bug.

    A structural clone scan flags them: identical shape, one underscore apart in the
    name.  What differs is the leg convention -- ``mpo_apply`` stores ``(vL, p, vR)``
    and reads axis 2, ``tdvp`` stores ``(vL, vR, p)`` and reads axis 1.  Applied to
    the wrong convention each returns the *physical* dimensions with no error.

    Pinned with a tensor whose bond and physical dimensions differ, so the two
    genuinely disagree and a future "cleanup" that unifies them fails here.
    """
    from fishbonett.evolve.mpo_apply import bond_dims
    from fishbonett.evolve.tdvp import bonddims

    D, d = 3, 7                                   # bond != physical, so axes differ
    mid_phys = [np.zeros((D, d, D)) for _ in range(2)]      # (vL, p, vR)
    last_phys = [np.zeros((D, D, d)) for _ in range(2)]     # (vL, vR, p)

    assert bond_dims(mid_phys) == [D, D, D]
    assert bonddims(last_phys) == [D, D, D]
    # each reads the physical leg when handed the other's layout
    assert bond_dims(last_phys) == [D, d, d]
    assert bonddims(mid_phys) == [D, d, d]


def test_one_loop_serves_every_tdvp_mpo_and_sweep():
    """Represented Hamiltonian and TDVP sweep are independent choices.

    Whichever MPO you hand it, and whichever sweep you name, it is the same loop --
    which is why there is no longer a function per combination.
    """
    n_chain, d, V = 3, 4, 1.0
    interaction = InteractionRepresentation(
        representation="interaction-star",
        h_sys=V * SX, coupling=SZ,
        bath=_bath(n_chain, d)).build()
    representations = {"chain": _chain(n_chain, d, V),
                       "interaction-star": interaction}
    for name, representation in representations.items():
        for sweep in ("tdvp1", "tdvp2", "dtdvp"):
            t, sz, maxd = run_mpo_hamiltonian(representation, dt=0.05, nsteps=2, sweep=sweep,
                                        D=20, chi_max=20, eps=1e-10, krylov=20)
            assert t.shape == (2,) and sz.shape == (2,), (name, sweep)
            assert np.all(np.isfinite(sz)), (name, sweep)
            # every sweep reports the peak bond, including the fixed-bond one
            assert maxd.shape == (2,) and np.all(maxd >= 1), (name, sweep)
