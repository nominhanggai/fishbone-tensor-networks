"""Tests for the multi-bath Fishbone interface, validated vs exact diagonalization."""
import numpy as np
import pytest

from fishbonett.models import Fishbone
from fishbonett import Bath, Result
from fishbonett.bath.chain import get_coupling
from fishbonett.operators import annihilate
from fishbonett.operators import sigma_x, sigma_z

DOM = (0.0, 40.0)


def _J(w):
    return 0.2 * w * np.exp(-w / 5.0)


def _bath(nm, dph, op):
    return Bath(J=_J, domain=DOM, n_modes=nm, phys_dim=dph, coupling=op)


def _embed(op, s, dims):
    m = [np.eye(x) for x in dims]
    m[s] = op
    out = m[0]
    for x in m[1:]:
        out = np.kron(out, x)
    return out


def _embed_two(op4, sa, sb, dims):
    da, db = dims[sa], dims[sb]
    full = np.zeros((int(np.prod(dims)),) * 2, complex)
    for a in range(da):
        for b in range(db):
            for a2 in range(da):
                for b2 in range(db):
                    val = op4[a, b, a2, b2]
                    if val != 0:
                        ea = np.zeros((da, da), complex); ea[a, a2] = 1
                        eb = np.zeros((db, db), complex); eb[b, b2] = 1
                        full += val * (_embed(ea, sa, dims) @ _embed(eb, sb, dims))
    return full


def _build_exact(sites, specs, backbone, d):
    """Build and diagonalize the full fishbone Hamiltonian.  ``specs[i]`` is a
    list of ``(n_modes, coupling_op, side in {'L','R'})``.  Returns
    ``(E, U, cc, e_idx, dims)`` for evaluating any observable exactly."""
    nc = len(sites)
    de = [h.shape[0] for h in sites]
    dims, e_idx, slots = [], [], []
    for i in range(nc):
        slot = {}
        for (nm, op, side) in specs[i]:
            if side == "L":
                slot["L"] = (len(dims), nm); dims += [d] * nm
        e_idx.append(len(dims)); dims.append(de[i])
        for (nm, op, side) in specs[i]:
            if side == "R":
                slot["R"] = (len(dims), nm); dims += [d] * nm
        slots.append(slot)
    tot = int(np.prod(dims))
    H = np.zeros((tot, tot), complex)
    c = annihilate(d)
    for i in range(nc):
        for (nm, op, side) in specs[i]:
            w, k = get_coupling(_J, nm, list(DOM), 1.0)
            base, _ = slots[i][side]
            for m in range(nm):
                H += w[m] * _embed(c.T @ c, base + m, dims)
            for m in range(nm - 1):
                H += k[m + 1] * (_embed(c.T, base + m, dims) @ _embed(c, base + m + 1, dims)
                                 + _embed(c, base + m, dims) @ _embed(c.T, base + m + 1, dims))
            H += k[0] * (_embed(c + c.T, base, dims) @ _embed(op, e_idx[i], dims))
        H += _embed(sites[i], e_idx[i], dims)
    for i in range(nc - 1):
        H += _embed_two(backbone[i].reshape(de[i], de[i + 1], de[i], de[i + 1]),
                        e_idx[i], e_idx[i + 1], dims)
    E, U = np.linalg.eigh(H)
    p0 = np.zeros(tot, complex); p0[0] = 1
    cc = U.conj().T @ p0
    return E, U, cc, e_idx, dims


def _expect_t(E, U, cc, op_full, ts):
    """``<op_full>(t)`` under the exact evolution."""
    out = []
    for t in ts:
        psi = U @ (np.exp(-1j * E * t) * cc)
        out.append((psi.conj() @ (op_full @ psi)).real)
    return np.array(out)


def _exact(sites, specs, backbone, d, ts):
    """List of ``<sigma_z on site i>(t)``."""
    E, U, cc, e_idx, dims = _build_exact(sites, specs, backbone, d)
    return [_expect_t(E, U, cc, _embed(sigma_z, e_idx[i], dims), ts)
            for i in range(len(sites))]


def _check(sites, baths, backbone, specs, d, tol, dt=0.02, ns=12):
    """Propagate and compare against exact diagonalization.

    The ``tol`` values are ~3x the errors actually measured at ``dt=0.02`` with the
    second-order step (3e-5 to 1.4e-4).  They used to be 1e-3 to 5e-3, which the
    old first-order step also satisfied -- keeping them loose is what let the wrong
    Trotter order go unnoticed, so they are now tight enough that a regression to
    first order fails here as well as in the convergence test.
    """
    fb = Fishbone(sites=sites, baths=baths, backbone=backbone)
    res = fb.run(dt=dt, n_steps=ns, bond_dim=40, trunc_eps=1e-12,
                 observables={"sz": sigma_z})
    assert isinstance(res, Result)
    assert res.expect["sz"].shape == (ns, len(sites))
    assert res.rdm.shape == (ns, len(sites), 2, 2)
    ex = _exact(sites, specs, backbone or [], d, res.t)
    for i in range(len(sites)):
        assert np.max(np.abs(res.expect["sz"][:, i] - ex[i])) < tol


def test_single_site_one_bath():
    _check([0.5 * sigma_z + sigma_x], [_bath(2, 4, sigma_z)], None,
           [[(2, sigma_z, "L")]], d=4, tol=2e-4)


def test_two_sites_one_bath_backbone():
    _check([0.25 * sigma_z + 0.8 * sigma_x, -0.15 * sigma_z + 0.8 * sigma_x],
           [_bath(2, 4, sigma_z), _bath(2, 4, sigma_z)],
           [0.4 * np.kron(sigma_z, sigma_z)],
           [[(2, sigma_z, "L")], [(2, sigma_z, "L")]], d=4, tol=2e-4)


def test_single_site_two_baths():
    _check([0.2 * sigma_z + 0.7 * sigma_x],
           [(_bath(2, 4, sigma_z), _bath(2, 4, sigma_x))], None,
           [[(2, sigma_z, "L"), (2, sigma_x, "R")]], d=4, tol=4e-4)


def test_two_sites_two_baths_fishbone():
    _check([0.2 * sigma_z + 0.6 * sigma_x, -0.1 * sigma_z + 0.6 * sigma_x],
           [(_bath(2, 2, sigma_z), _bath(2, 2, sigma_x)),
            (_bath(2, 2, sigma_z), _bath(2, 2, sigma_x))],
           [0.3 * np.kron(sigma_z, sigma_z)],
           [[(2, sigma_z, "L"), (2, sigma_x, "R")],
            [(2, sigma_z, "L"), (2, sigma_x, "R")]], d=2, tol=5e-4)


def test_tree_step_is_second_order_in_dt():
    """``schrodinger-chain-tree-tebd`` must be **second** order, like every other method.

    It was first order for a long time (measured 1.07): the step applied one full
    gate per edge in a single Euler tour.  Nothing caught that, because every other
    multi-site test asserts only an upper bound on the error, which a first-order
    step at a small ``dt`` still satisfies.  Halving ``dt`` is what distinguishes
    them -- the error must fall by ~4, not ~2.

    Uses a **branching** tree (two electronic sites, each with its own bath chain,
    joined by a backbone) on purpose: the tempting fix of applying half a gate on
    the way down and half on the way back up an Euler tour is *not* second order
    where the tree branches (measured 1.79), so a path-shaped model would not
    discriminate between the two schemes.
    """
    sites = [0.3 * sigma_z + 0.7 * sigma_x, -0.2 * sigma_z + 0.5 * sigma_x]
    baths = [_bath(2, 3, sigma_z), _bath(2, 3, sigma_z)]
    backbone = [0.4 * np.kron(sigma_x, sigma_x)]
    specs = [[(2, sigma_z, "L")], [(2, sigma_z, "L")]]
    E, U, cc, e_idx, dims = _build_exact(sites, specs, backbone, 3)
    op = _embed(sigma_z, e_idx[0], dims)

    t_end, errs = 0.24, []
    for dt in (0.02, 0.01):
        fb = Fishbone(sites=sites, baths=baths, backbone=backbone)
        res = fb.run(dt=dt, n_steps=int(round(t_end / dt)), bond_dim=200,
                     trunc_eps=1e-13, observables={"z0": (sigma_z, 0)})
        exact = _expect_t(E, U, cc, op, res.t)
        errs.append(np.max(np.abs(res.expect["z0"] - exact)))

    ratio = errs[0] / errs[1]
    assert 3.3 < ratio < 4.7, (
        f"halving dt cut the error by {ratio:.2f}x, so the step is order "
        f"{np.log2(ratio):.2f}; expected ~4x (second order).  Errors: "
        f"dt=0.02 -> {errs[0]:.3e}, dt=0.01 -> {errs[1]:.3e}")


def test_result_shapes_and_normalization():
    fb = Fishbone(sites=[0.5 * sigma_z + sigma_x] * 3,
                  baths=[_bath(2, 4, sigma_z)] * 3,
                  backbone=[0.2 * np.kron(sigma_z, sigma_z)] * 2)
    res = fb.run(dt=0.02, n_steps=5, bond_dim=30)
    assert res.rdm.shape == (5, 3, 2, 2)
    for tn in range(5):
        for cn in range(3):
            assert abs(np.trace(res.rdm[tn, cn]).real - 1.0) < 1e-6


def test_multi_site_observable_vs_exact():
    # As a specialization of the general tree engine, the 1D Fishbone now
    # supports the full observable interface -- including a composite operator
    # across two sites, which the old comb engine could not measure.
    sites = [0.3 * sigma_z + 0.7 * sigma_x, -0.2 * sigma_z + 0.5 * sigma_x]
    baths = [_bath(2, 4, sigma_z), _bath(2, 4, sigma_z)]
    backbone = [0.35 * np.kron(sigma_z, sigma_z)]
    specs = [[(2, sigma_z, "L")], [(2, sigma_z, "L")]]
    zz = np.kron(sigma_z, sigma_z)
    fb = Fishbone(sites=sites, baths=baths, backbone=backbone)
    res = fb.run(dt=0.02, n_steps=12, bond_dim=40, trunc_eps=1e-12,
                 observables={"z0": (sigma_z, 0), "zz": (zz, (0, 1))})
    assert res.expect["z0"].shape == (12,)
    assert res.expect["zz"].shape == (12,)
    E, U, cc, e_idx, dims = _build_exact(sites, specs, backbone, 4)
    ex_z0 = _expect_t(E, U, cc, _embed(sigma_z, e_idx[0], dims), res.t)
    ex_zz = _expect_t(E, U, cc,
                      _embed(sigma_z, e_idx[0], dims) @ _embed(sigma_z, e_idx[1], dims),
                      res.t)
    assert np.max(np.abs(res.expect["z0"] - ex_z0)) < 1e-4
    assert np.max(np.abs(res.expect["zz"] - ex_zz)) < 1.5e-4


def test_per_bath_domains_allowed():
    # As a specialization of the general tree engine, each bath discretizes
    # independently, so baths on different frequency domains are allowed.
    b1 = Bath(J=_J, domain=(0.0, 40.0), n_modes=2, phys_dim=4, coupling=sigma_z)
    b2 = Bath(J=_J, domain=(0.0, 30.0), n_modes=2, phys_dim=4, coupling=sigma_z)
    fb = Fishbone(sites=[sigma_z, sigma_z], baths=[b1, b2],
                  backbone=[np.zeros((4, 4))])
    res = fb.run(dt=0.02, n_steps=4, bond_dim=20)
    assert res.rdm.shape == (4, 2, 2, 2)
    for tn in range(4):
        for cn in range(2):
            assert abs(np.trace(res.rdm[tn, cn]).real - 1.0) < 1e-6


PROJECTOR = np.array([[1.0, 0.0], [0.0, 0.0]])
COMB_MPO = "interaction-chain-fishbone-trotter-mpo"
COMB_TEBD = "interaction-chain-fishbone-tebd"
COMB_TDVP2 = "interaction-chain-fishbone-tdvp2"
#: Every integrator the comb offers.  They solve the same H(t) and must agree.
COMB_METHODS = (COMB_TEBD, COMB_MPO, COMB_TDVP2)


def _comb(npig, nm=3, dph=6, op=PROJECTOR):
    """A comb with a backbone coupled to independent projector baths."""
    return Fishbone(
        sites=[0.5 * sigma_z + sigma_x] * npig,
        baths=[_bath(nm, dph, op)] * npig,
        backbone=[0.4 * np.kron(sigma_z, sigma_z)] * (npig - 1))


@pytest.mark.parametrize("method", [COMB_MPO, COMB_TDVP2])
def test_comb_operator_integrators_match_exact_diagonalization(method):
    """One system site is a system-bath model, so there is an exact answer to hit.

    Run it for both couplings the eigenvalue structure distinguishes: ``sigma_z``
    (equal magnitudes) and a projector (unequal), since only the latter exposes the
    branch-dependent phase of the conditional-displacement propagator.
    """
    from fishbonett.bath.chain import star_transform

    for coupling in (sigma_z, PROJECTOR):
        bath = Bath(J=_J, domain=DOM, n_modes=3, phys_dim=6)
        freq, vn, _ = star_transform(bath.spectral_density(), 3, list(DOM))
        dims = [2] + [6] * 3
        h_sys = 1.0 * sigma_x
        H = _embed(h_sys, 0, dims)
        for k in range(3):
            H = H + freq[k] * _embed(
                annihilate(6).T @ annihilate(6), 1 + k, dims)
            H = H + vn[k] * (_embed(coupling, 0, dims)
                             @ _embed(annihilate(6) + annihilate(6).T, 1 + k, dims))
        energies, vectors = np.linalg.eigh(H)
        psi0 = np.zeros(int(np.prod(dims)), complex)
        psi0[0] = 1.0
        amps = vectors.conj().T @ psi0
        observable = _embed(sigma_z, 0, dims)

        model = Fishbone(sites=[h_sys], baths=[_bath(3, 6, coupling)])
        result = model.run(dt=0.02, n_steps=25, method=method, bond_dim=64,
                           trunc_eps=1e-10, observables={"sz": sigma_z})
        reference = np.array([
            (lambda psi: (psi.conj() @ (observable @ psi)).real)(
                vectors @ (np.exp(-1j * energies * t) * amps))
            for t in result.t])
        got = np.asarray(result.expect["sz"]).reshape(len(result.t), -1)[:, 0]
        assert np.max(np.abs(got - reference)) < 1e-3


def test_every_pair_of_comb_integrators_agrees_to_second_order():
    """All three solve the same H(t), so each pairwise difference vanishes as dt^2.

    This is what checks that a new comb integrator carries the *same physics* rather
    than merely something plausible: a wrong term -- a coupling attached to the
    wrong mode, say, which leaves H Hermitian and the run stable -- shows up as a
    wrong convergence order, not as a larger constant.
    """
    series = {method: [] for method in COMB_METHODS}
    for dt in (0.04, 0.02, 0.01):
        for method in COMB_METHODS:
            result = _comb(3).run(
                dt=dt, n_steps=int(round(0.2 / dt)), method=method, bond_dim=64,
                trunc_eps=1e-10, observables={"sz": sigma_z})
            series[method].append(np.asarray(result.expect["sz"]))
    for index, first in enumerate(COMB_METHODS):
        for second in COMB_METHODS[index + 1:]:
            errors = [float(np.max(np.abs(a - b)))
                      for a, b in zip(series[first], series[second])]
            assert errors[0] < 1e-4, (first, second, errors)
            for coarse, fine in zip(errors, errors[1:]):
                assert 3.4 < coarse / fine < 4.6, (
                    f"{first} vs {second} is not second order: {errors}")


def test_two_site_tdvp_does_not_beat_a_trotter_step_at_a_binding_cap():
    """Pin the measured fact, because the intuition points the other way.

    One-site TDVP works strictly inside a fixed-bond manifold and never truncates,
    so it is tempting to assume a capped two-site TDVP inherits that.  It does not:
    ``tdvp2sweep`` splits every two-site block with a truncating SVD, so once the
    cap binds it discards weight exactly as a Trotter step does.

    Each method is compared against *its own* uncapped limit, which isolates the
    cap's cost.  (Averaging the two methods' uncapped runs does not work -- their
    mutual difference exceeds the cap error, so every capped run sits half that
    distance from the average and the cap becomes invisible.)
    """
    cache = {}

    def run(method, cap):
        if (method, cap) not in cache:
            result = _comb(2, nm=4).run(
                dt=0.02, n_steps=20, method=method, bond_dim=cap,
                trunc_eps=1e-12, observables={"sz": sigma_z})
            cache[method, cap] = np.asarray(result.expect["sz"])
        return cache[method, cap]

    tight = {}
    for method in (COMB_MPO, COMB_TDVP2):
        uncapped = run(method, None)
        errors = [float(np.max(np.abs(run(method, cap) - uncapped)))
                  for cap in (3, 5)]
        assert errors[0] > errors[1], f"{method}: a looser cap must not be worse"
        tight[method] = errors[0]

    ratio = tight[COMB_MPO] / tight[COMB_TDVP2]
    assert 0.5 < ratio < 2.0, (
        "the cap is expected to cost the two integrators about the same; if this "
        f"has changed, re-measure before claiming otherwise (ratio {ratio:.2f})")


def test_unknown_comb_run_options_are_rejected():
    """A misspelled numerical option must not be accepted and ignored."""
    for method in (COMB_TEBD, COMB_MPO, COMB_TDVP2):
        with pytest.raises(TypeError, match=r"unexpected run option.*trunc_epz"):
            _comb(2).run(dt=0.02, n_steps=1, method=method,
                         trunc_epz=1e-8, observables={"sz": sigma_z})


def test_branch_workers_is_not_a_public_run_option():
    """Concurrent writes through a shared system tensor are not a valid engine."""
    with pytest.raises(TypeError, match=r"unexpected run option.*branch_workers"):
        _comb(3).run(dt=0.02, n_steps=1, method=COMB_MPO,
                     branch_workers=4, observables={"sz": sigma_z})


def _two_line_bath(freqs, strengths, dph, op, temperature=None):
    """A vibronic bath of the given discrete lines and no continuum.

    ``n_modes`` must be the thermofield-doubled line count: the resolver refuses a
    smaller one, because at finite temperature each physical line becomes a pair.
    """
    freqs = np.asarray(freqs, float)
    doubled = 2 * len(freqs) if temperature is not None else len(freqs)
    return Bath.vibronic(
        freqs, np.asarray(strengths, float) ** 2 / freqs ** 2,
        continuum=None, temperature=temperature, domain=(-40.0, 40.0),
        n_modes=doubled, phys_dim=dph, discretization="legendre").bind(op)


def test_splitting_a_bath_across_branches_is_exact():
    """Two one-mode baths on a site must equal one two-mode bath on that site.

    The bath Hamiltonian is a sum over independent oscillators, so partitioning
    them between branches is an identity, not an approximation.  That makes this the
    sharpest available check on the per-branch bookkeeping the comb planner does --
    node allocation, ``dims``, the edge list and the continuation signature -- since
    any mistake there shows up as a *physics* difference rather than a crash.
    """
    freqs, strengths = np.array([3.0, 11.0]), np.array([0.5, 0.35])
    together = _two_line_bath(freqs, strengths, 6, PROJECTOR)
    apart = [_two_line_bath(freqs[k:k + 1], strengths[k:k + 1], 6, PROJECTOR)
             for k in range(2)]

    series = []
    for spec in (together, apart):
        model = Fishbone(sites=[1.0 * sigma_x], baths=[spec])
        result = model.run(dt=0.02, n_steps=20, method=COMB_MPO, bond_dim=64,
                           trunc_eps=1e-12, observables={"sz": sigma_z})
        series.append(
            np.asarray(result.expect["sz"]).reshape(len(result.t), -1)[:, 0])
    assert np.max(np.abs(series[0] - series[1])) < 3e-8, (
        "splitting a bath between branches changed the dynamics")


def test_a_site_may_carry_baths_with_different_phys_dim():
    """Each branch keeps its own truncation, so parts can be sized independently.

    This is what makes a split worthwhile beyond the chain-mixing question: modes
    that are barely populated do not need the ``phys_dim`` the strongly driven ones
    do, and before this a site had a single bath and therefore a single value.
    """
    freqs, strengths = np.array([3.0, 11.0]), np.array([0.5, 0.35])
    apart = [_two_line_bath(freqs[:1], strengths[:1], 6, PROJECTOR),
             _two_line_bath(freqs[1:], strengths[1:], 3, PROJECTOR)]
    model = Fishbone(sites=[1.0 * sigma_x], baths=[apart])
    result = model.run(dt=0.02, n_steps=4, method=COMB_MPO, bond_dim=32,
                       trunc_eps=1e-10, observables={"sz": sigma_z})
    # one electronic site plus one mode per branch, at 6 and 3 levels
    dims = sorted(t.shape[-1] for t in result.checkpoint.tensors)
    assert dims == [2, 3, 6], dims


def test_an_unbound_list_of_baths_is_rejected():
    """Several baths require explicit operators; their position has no meaning."""
    raw = Bath(J=_J, domain=DOM, n_modes=2, phys_dim=4)
    with pytest.raises(ValueError, match="bind every coupling operator explicitly"):
        Fishbone._site_baths([raw, raw])


@pytest.mark.parametrize("method", COMB_METHODS)
def test_noncommuting_baths_on_one_site_are_second_order(method):
    """Local bath branches need a symmetric split when their operators differ."""
    from scipy.linalg import expm

    h_sys = 0.3 * sigma_z + 0.7 * sigma_x
    frequencies = (2.0, 3.0)
    huang_rhys = (0.08, 0.05)
    dimension = 4
    baths = [
        Bath.vibronic([frequencies[0]], [huang_rhys[0]],
                      phys_dim=dimension).bind(sigma_z),
        Bath.vibronic([frequencies[1]], [huang_rhys[1]],
                      phys_dim=dimension).bind(sigma_x),
    ]
    destroy = annihilate(dimension)
    number = destroy.T @ destroy
    position = destroy + destroy.T
    identity = np.eye(dimension)
    full = np.kron(np.kron(h_sys, identity), identity)
    full += frequencies[0] * np.kron(np.kron(np.eye(2), number), identity)
    full += frequencies[1] * np.kron(np.kron(np.eye(2), identity), number)
    full += frequencies[0] * np.sqrt(huang_rhys[0]) * np.kron(
        np.kron(sigma_z, position), identity)
    full += frequencies[1] * np.sqrt(huang_rhys[1]) * np.kron(
        np.kron(sigma_x, identity), position)
    initial = np.zeros(2 * dimension ** 2, complex)
    initial[0] = 1.0
    end_time = 0.2
    final = expm(-1j * full * end_time) @ initial
    observable = np.kron(np.kron(sigma_z, identity), identity)
    exact = np.vdot(final, observable @ final).real

    errors = []
    for dt in (0.04, 0.02, 0.01):
        result = Fishbone(sites=[h_sys], baths=[baths]).run(
            dt=dt, n_steps=int(round(end_time / dt)), method=method,
            bond_dim=200, trunc_eps=1e-13, observables={"sz": sigma_z})
        got = np.asarray(result.expect["sz"])[-1, 0]
        errors.append(abs(got - exact))
    assert 3.7 < errors[0] / errors[1] < 4.3, errors
    assert 3.7 < errors[1] / errors[2] < 4.3, errors
