"""Unit tests for the MPO / TDVP engine, validated against exact diagonalization
of a small spin-boson chain."""
import numpy as np
import pytest

from fishbonett import Bath
from fishbonett.bath.chain import get_bath_nn_parameters
from fishbonett.evolve.mpo_apply import apply_mpo, product_state
from fishbonett.evolve.tdvp import run_mpo_hamiltonian, SX, SZ
from fishbonett.operators import annihilate, create, number
from fishbonett.representations.interaction import InteractionRepresentation
from fishbonett.representations.schrodinger import SchrodingerRepresentation

DOMAIN = (-25.0, 36.0)

# ``run_mpo_hamiltonian`` receives the complete integration step.


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


def test_apply_mpo_rejects_a_mismatched_site_count():
    """An incomplete MPO must not silently drop the final MPS sites."""
    state = product_state((2, 2))
    identity = np.eye(2, dtype=complex).reshape(1, 1, 2, 2)
    with pytest.raises(ValueError):
        apply_mpo(state, [identity])


def test_dense_operator_mpo_reconstructs_a_nonlocal_gate():
    """The private TT-SVD compiler preserves local output/input ordering."""
    import scipy.linalg as la
    from fishbonett.representations._mpo import dense_operator_mpo

    dimensions = (2, 3, 2)
    rng = np.random.default_rng(11)
    hamiltonian = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    hamiltonian = hamiltonian + hamiltonian.conj().T
    gate = la.expm(-0.07j * hamiltonian)
    mpo = dense_operator_mpo(gate, dimensions)

    rebuilt = np.zeros_like(gate)
    for output in np.ndindex(*dimensions):
        row = np.ravel_multi_index(output, dimensions)
        for input_ in np.ndindex(*dimensions):
            column = np.ravel_multi_index(input_, dimensions)
            transfer = np.array([[1.0 + 0.0j]])
            for core, out_index, in_index in zip(
                mpo, output, input_, strict=True
            ):
                transfer = transfer @ core[:, :, out_index, in_index]
            rebuilt[row, column] = transfer.item()
    np.testing.assert_allclose(rebuilt, gate, atol=2e-13)


def _exact_sz(n_chain, d, V, ts):
    bath = _bath(n_chain, d)
    eps_c, couplings = get_bath_nn_parameters(
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
                             sweep="tdvp1", bond_dim=40, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.isclose(sz[0], 0.99, atol=0.02)          # starts near |up>
    assert np.max(np.abs(sz - sz_ex)) < 1e-6


def test_tdvp2_matches_exact_and_grows_bonds():
    n_chain, d, V = 3, 5, 1.0
    t, sz, maxd = run_mpo_hamiltonian(_chain(n_chain, d, V), dt=0.10, nsteps=12,
                                sweep="tdvp2", bond_dim=40,
                                trunc_eps=1e-12, krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert maxd[-1] > 1                                 # bonds grew from product state
    assert np.max(np.abs(sz - sz_ex)) < 1e-6


def test_ip_mpo_matches_exact():
    """Interaction-chain MPO (time-dependent, rebuilt each step) vs the same
    exact dynamics.  Looser tol: the IP midpoint rule is O(dt^2) in time."""
    n_chain, d, V = 3, 5, 1.0
    representation = InteractionRepresentation(
        representation="interaction-chain",
        h_sys=V * SX, coupling=SZ,
        bath=_bath(n_chain, d)).build()
    assert not representation.static, "the interaction MPO must be rebuilt"
    t, sz, _ = run_mpo_hamiltonian(representation, dt=0.04, nsteps=15,
                             sweep="tdvp1", bond_dim=40,
                             krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert np.max(np.abs(sz - sz_ex)) < 5e-3
    t2, sz2, maxd = run_mpo_hamiltonian(representation, dt=0.04, nsteps=15, sweep="tdvp2",
                                  bond_dim=40, trunc_eps=1e-12, krylov=25)
    assert maxd[-1] > 1                                 # bonds grew
    assert np.max(np.abs(sz2 - sz_ex)) < 5e-3


def test_a1tdvp_grows_bonds_and_tracks_dynamics():
    n_chain, d, V = 3, 5, 1.0
    t, sz, maxd = run_mpo_hamiltonian(_chain(n_chain, d, V), dt=0.10, nsteps=12,
                                sweep="a1tdvp", trunc_eps=1e-9, bond_dim=40,
                                krylov=25)
    sz_ex = _exact_sz(n_chain, d, V, t)
    assert maxd[-1] > 1 and np.all(np.diff(maxd) >= 0)
    assert np.all(np.isfinite(sz))
    assert np.max(np.abs(sz - sz_ex)) < 1e-5

    _t, loose_sz, loose_bonds = run_mpo_hamiltonian(
        _chain(n_chain, d, V),
        dt=0.10,
        nsteps=12,
        sweep="a1tdvp",
        trunc_eps=1e-2,
        bond_dim=40,
        krylov=25,
    )
    assert loose_bonds[-1] <= maxd[-1]
    # A convergence precision of 1e-2 intentionally stops at a much smaller
    # tangent space; it should remain a controlled coarse approximation, not
    # satisfy the strict run's 1e-5 accuracy target.
    assert np.max(np.abs(loose_sz - sz_ex)) < 2e-4


def _dense_mps(tensors):
    """Contract TDVP-order tensors into one physical state vector."""
    dense = np.transpose(tensors[0][0], (1, 0))
    for tensor in tensors[1:]:
        dense = np.tensordot(dense, tensor, axes=([-1], [0]))
        dense = np.moveaxis(dense, -2, -1)
    return np.squeeze(dense, axis=-1).reshape(-1)


def test_partial_full_qr_returns_the_requested_full_basis_columns():
    from fishbonett.evolve._tdvp_sweeps import _partial_full_qr

    rng = np.random.default_rng(29)
    matrix = rng.normal(size=(31, 5)) + 1j * rng.normal(size=(31, 5))
    basis, triangular = _partial_full_qr(matrix, 9)

    np.testing.assert_allclose(
        basis.conj().T @ basis, np.eye(9), atol=2e-13
    )
    np.testing.assert_allclose(
        basis[:, :5] @ triangular, matrix, atol=2e-13
    )
    np.testing.assert_allclose(
        basis[:, :5].conj().T @ basis[:, 5:], 0.0, atol=2e-13
    )


def test_adaptive_qr_expansion_preserves_the_state_and_right_gauge():
    from fishbonett.evolve._tdvp_kernels import right_canonicalize
    from fishbonett.evolve._tdvp_sweeps import _expand_right_canonical

    rng = np.random.default_rng(83)
    dimensions = (2, 3, 3, 2)
    bonds = (1, 2, 2, 2, 1)
    state = [
        rng.normal(size=(bonds[i], bonds[i + 1], dimensions[i]))
        + 1j * rng.normal(size=(bonds[i], bonds[i + 1], dimensions[i]))
        for i in range(len(dimensions))
    ]
    state = right_canonicalize(state)
    before = _dense_mps(state)
    expanded = _expand_right_canonical(state, [2, 3, 2])

    np.testing.assert_allclose(_dense_mps(expanded), before, atol=2e-13)
    assert [tensor.shape[1] for tensor in expanded[:-1]] == [2, 3, 2]
    for tensor in expanded[1:]:
        matrix = tensor.reshape(tensor.shape[0], -1)
        np.testing.assert_allclose(
            matrix @ matrix.conj().T,
            np.eye(tensor.shape[0]),
            atol=2e-13,
        )


def test_a1tdvp_time_step_never_uses_a_two_site_exponential(monkeypatch):
    import fishbonett.evolve._tdvp_sweeps as sweeps

    def forbidden(*_args, **_kwargs):
        raise AssertionError("adaptive one-site TDVP called applyH2")

    monkeypatch.setattr(sweeps, "applyH2", forbidden)
    _t, _sz, maxd = run_mpo_hamiltonian(
        _chain(2, 3, 1.0),
        dt=0.05,
        nsteps=3,
        sweep="a1tdvp",
        trunc_eps=1e-8,
        bond_dim=20,
        krylov=20,
    )
    assert maxd[-1] > 1


def test_a1tdvp_preserves_norm_and_energy_after_each_qr_expansion():
    from fishbonett.evolve._tdvp_kernels import updateleftenv

    representation = _chain(3, 4, 1.0)
    mpo = representation.tdvp_mpo(None)
    energies = []
    norms = []

    def record(info):
        state = info["state"]
        environment = np.ones((1, 1, 1), complex)
        for tensor, operator in zip(state, mpo, strict=True):
            environment = updateleftenv(tensor, operator, environment)
        energies.append(environment.item().real)

        environment = np.ones((1, 1, 1), complex)
        for tensor in state:
            physical = tensor.shape[2]
            identity = np.eye(physical).reshape(1, 1, physical, physical)
            environment = updateleftenv(tensor, identity, environment)
        norms.append(environment.item().real)

    run_mpo_hamiltonian(
        representation,
        dt=0.05,
        nsteps=12,
        sweep="a1tdvp",
        trunc_eps=1e-6,
        bond_dim=40,
        krylov=30,
        tol=1e-11,
        progress=record,
    )
    assert np.ptp(energies) < 1e-10
    assert np.max(np.abs(np.asarray(norms) - 1.0)) < 1e-11


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

    Every represented MPO and sweep is routed through the same driver loop.
    """
    n_chain, d, V = 3, 4, 1.0
    interaction = InteractionRepresentation(
        representation="interaction-chain",
        h_sys=V * SX, coupling=SZ,
        bath=_bath(n_chain, d)).build()
    representations = {"chain": _chain(n_chain, d, V),
                       "interaction-chain": interaction}
    for name, representation in representations.items():
        for sweep in ("tdvp1", "tdvp2", "a1tdvp"):
            t, sz, maxd = run_mpo_hamiltonian(representation, dt=0.05, nsteps=2, sweep=sweep,
                                        bond_dim=20, trunc_eps=1e-10,
                                        krylov=20)
            assert t.shape == (2,) and sz.shape == (2,), (name, sweep)
            assert np.all(np.isfinite(sz)), (name, sweep)
            # every sweep reports the peak bond, including the fixed-bond one
            assert maxd.shape == (2,) and np.all(maxd >= 1), (name, sweep)


def test_displacement_matches_a_matrix_exponential():
    """The closed form must equal ``expm`` on the *truncated* generator.

    ``trotter_mpo`` called ``expm`` once per (mode, coupling eigenvalue), every
    step -- thousands of calls per step on a moderately large comb. The displacement
    factorizes as a phase rotation of one fixed matrix, so a single cached ``eigh``
    per dimension replaces all of them; this pins the two against each other,
    including the boundary cases (alpha = 0, and a magnitude large enough that the
    truncation matters).
    """
    import scipy.linalg as la
    from fishbonett.operators import displacement

    for dim in (2, 3, 6, 10, 20):
        lower, raise_ = annihilate(dim), create(dim)
        for alpha in (0.0, 0.3, -0.7, 0.4j, 0.5 - 0.9j, 2.3 + 1.1j, 1e-9):
            reference = la.expm(alpha * raise_ - np.conj(alpha) * lower)
            assert np.allclose(displacement(alpha, dim), reference, atol=1e-12), (
                f"dim={dim} alpha={alpha}")

    # batched over an array of alphas, and unitary
    alphas = np.array([[0.1 + 0.2j, 0.0], [-1.3, 0.7j]])
    batch = displacement(alphas, 7)
    assert batch.shape == (2, 2, 7, 7)
    for i in range(2):
        for j in range(2):
            assert np.allclose(batch[i, j], la.expm(
                alphas[i, j] * create(7) - np.conj(alphas[i, j]) * annihilate(7)),
                atol=1e-12)
    unitary = displacement(0.8 - 0.3j, 9)
    assert np.allclose(unitary @ unitary.conj().T, np.eye(9), atol=1e-12)


def test_trotter_mpo_is_a_displacement_up_to_a_quadratic_phase():
    """Pin what the conditional-displacement MPO does and does *not* carry.

    Within eigenbranch ``lambda`` of the coupling, the interval propagator is the
    displacement the MPO stores times a phase from the second Magnus term.  That
    term is weighted by ``lambda**2``, so it is:

    * common to every branch -- hence unobservable -- when the eigenvalues share a
      magnitude, as for ``sigma_z`` (+-1).  This is why it went unnoticed.
    * a *relative* phase when they do not, as for a projector (0 and 1), which is
      the coupling the comb models use.

    Either way it is O(dt**3) per step, matching the order of the surrounding
    Strang splitting, so it is a property to record rather than a defect to fix.
    Away from the top Fock level the residual is a pure phase, which is what makes
    that statement checkable at all.
    """
    import scipy.linalg as la
    from fishbonett.operators import sigma_z

    def ordered(rep, mode, lam, t, dt, nsub=400):
        """Time-ordered reference on one mode; H = conj(c) a^dag + c a."""
        dim = rep.pd_boson[mode]
        lower = annihilate(dim)
        raise_ = lower.conj().T
        h = dt / nsub
        out = np.eye(dim, dtype=complex)
        for k in range(nsub):
            c = rep.interval_coefficients(t + k * h, h)[mode] / h
            out = la.expm(-1j * h * lam * (np.conj(c) * raise_ + c * lower)) @ out
        return out

    keep = 8                      # stay clear of the truncation edge
    for coupling in (sigma_z, np.array([[1.0, 0.0], [0.0, 0.0]])):
        bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5.0), domain=(0.0, 40.0),
                    n_modes=3, phys_dim=12).resolved(1.0)
        rep = InteractionRepresentation(
            representation="interaction-chain", h_sys=np.zeros((2, 2), complex),
            coupling=coupling, bath=bath).build()
        values, _ = np.linalg.eigh(np.asarray(coupling, float))

        phases = {}
        for dt in (0.08, 0.04):
            angles = []
            for branch, lam in enumerate(values):
                tensor = rep.trotter_mpo(0.3, dt)[1]
                stored = tensor[branch, branch if tensor.shape[1] > 1 else 0]
                residual = (stored @ ordered(rep, 0, lam, 0.3, dt).conj().T
                            )[:keep, :keep]
                centre = np.mean(np.diag(residual))
                # a pure phase: nothing but a multiple of the identity survives
                assert np.max(np.abs(
                    residual - centre * np.eye(keep))) < 1e-6, "not a pure phase"
                angles.append(np.angle(centre))
            phases[dt] = angles

        relative = [abs(a[1] - a[0]) for a in phases.values()]
        if len(set(np.round(values ** 2, 12))) == 1:
            assert max(relative) < 1e-12, (
                "equal |eigenvalues| must leave no observable relative phase")
        else:
            assert relative[0] > 1e-6, "a projector must show a relative phase"
            # and it must shrink at least as fast as dt**2 per step
            assert relative[0] / relative[1] > 3.5
