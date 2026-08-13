"""Tests for the multi-bath Fishbone interface, validated vs exact diagonalization."""
import numpy as np
import pytest

from fishbonett.models import Fishbone
from fishbonett import Bath, Result
from fishbonett.frames.schrodinger import FishBoneH
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
            w, k = FishBoneH.get_coupling(nm, _J, list(DOM), 1.0)
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
    """``tree-tebd-static`` must be **second** order, like every other method.

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
