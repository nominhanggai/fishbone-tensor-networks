"""Smoke tests: each method family builds and runs on a tiny problem."""
import importlib

import numpy as np
import pytest


def test_chain_cooling_gives_normalized_rdm():
    from fishbonett.representations.coolingchain import SystemBathCoolingChain
    from fishbonett.operators import sigma_x, sigma_z

    pd = [2, 6, 6, 6]
    eth = SystemBathCoolingChain(
        pd, betaOmega=0.2, h_sys=10.0 * sigma_x, coupling=sigma_z,
        sd=lambda w: 0.5 * abs(w) * np.exp(-abs(w) / 10.0),
        domain=[-50.0, 50.0], ncap=200).build()
    eth.U = eth.get_u(0.01)
    for j in range(len(pd) - 1):
        eth.update_bond(j, 20, 1e-6, swap=0)
    rho = eth.get_rdm()
    assert np.all(np.isfinite(rho))
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)


def test_cooling_gauge_cancels_out_of_the_observable():
    """The correctness criterion for the cooling-chain representation, which had none.

    That representation carries finite temperature in a *non-unitary* ``betaOmega`` gauge
    rather than in extra modes, so the propagated state is not the physical one.
    Its docstring says observables must be read back through the heating operators
    ``exp(2 betaOmega n_i)`` and that "reading the RDM the ordinary way would give
    the wrong answer".  Both halves of that are checkable, and the whole representation had
    exactly one test, asserting only that a trace came out as 1.

    Three things, which together are what makes the scheme a gauge:

    * at ``betaOmega=0`` the correction is the identity, so the two reads agree;
    * at ``betaOmega>0`` they genuinely differ -- otherwise the machinery is inert
      and the docstring's warning is empty;
    * the *corrected* read is the same physical state at every gauge strength.

    Measured separation is about 1e9: invariance holds to ~5e-12 while the gauge
    shifts the naive read by up to 4.5e-3.
    """
    import numpy as np
    from fishbonett.representations.coolingchain import SystemBathCoolingChain
    from fishbonett.operators import sigma_x, sigma_z

    sd = lambda w: 0.5 * abs(w) * np.exp(-abs(w) / 10.0)
    pd = [2, 6, 6, 6]

    def evolve(beta_omega):
        st = SystemBathCoolingChain(
            pd, betaOmega=beta_omega, h_sys=10.0 * sigma_x, coupling=sigma_z,
            sd=sd, domain=[-50.0, 50.0], ncap=200).build()
        st.U = st.get_u(0.01)
        for _ in range(6):
            for j in range(len(pd) - 1):
                st.update_bond(j, 20, 1e-7, swap=0)
        return st.get_rdm(), st.rdm(0)      # through the gauge, and ignoring it

    ref, plain0 = evolve(0.0)
    assert np.allclose(ref, plain0, atol=1e-12), (
        "at betaOmega=0 the heating operators are the identity, so the corrected "
        "and ordinary reads must agree")

    for beta_omega in (0.2, 0.5):
        gauged, plain = evolve(beta_omega)
        assert np.abs(gauged - plain).max() > 1e-4, (
            f"betaOmega={beta_omega} changes nothing; the gauge correction is inert")
        assert np.abs(gauged - ref).max() < 1e-9, (
            f"betaOmega={beta_omega} changes the physical RDM by "
            f"{np.abs(gauged - ref).max():.2e}; the gauge is not cancelling")
        assert abs(np.trace(gauged).real - 1.0) < 1e-10


@pytest.mark.parametrize("module,cls", [("representations.coolingchain", "SystemBathCoolingChain")])
def test_cooling_shares_the_canonical_engine(module, cls):
    mod = importlib.import_module(f"fishbonett.{module}")
    bases = [b.__name__ for b in getattr(mod, cls).__mro__]
    assert "SystemBathMPS" in bases


def test_polaron_builds_and_gives_normalized_rdm():
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    bath = Bath(J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
                n_modes=6, phys_dim=6)
    r = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=bath).run(
        method="polaron-chain-tebd", dt=0.05, n_steps=3, bond_dim=30)
    assert np.all(np.isfinite(r.rdm))
    assert np.allclose([np.trace(rho).real for rho in r.rdm], 1.0, atol=1e-6)


def test_both_contraction_backends_agree():
    """``opt_einsum`` and the NumPy fallback must give the same numbers.

    ``opt_einsum`` is an optional extra, so CI's test job -- which installs only
    ``[test]`` -- exercises the NumPy path, while a dev machine with ``[speed]``
    exercises the other.  If they disagreed beyond rounding, the suite would mean
    different things in the two places.

    Measured across the whole method table they agree exactly except
    ``chain/trotter-mpo``, which differs by 2.2e-16 in the RDM -- under one machine
    epsilon, from a different contraction order.  This pins the primitive so a real
    divergence cannot hide behind that expectation.
    """
    import numpy as np

    opt_einsum = pytest.importorskip("opt_einsum",
                                     reason="only one backend is installed")
    from fishbonett.contract import _numpy_contract

    rng = np.random.default_rng(0)
    # the shape of a real tensor-network kernel: several operands, shared indices
    a = rng.standard_normal((4, 5, 3)) + 1j * rng.standard_normal((4, 5, 3))
    b = rng.standard_normal((3, 6)) + 1j * rng.standard_normal((3, 6))
    c = rng.standard_normal((5, 6, 2)) + 1j * rng.standard_normal((5, 6, 2))

    for subs, ops in [("ijk,kl,jlm->im", (a, b, c)),
                      ("ijk,ijk->", (a, a.conj())),
                      ("ijk,kl->ijl", (a, b))]:
        got = opt_einsum.contract(subs, *ops)
        ref = _numpy_contract(subs, *ops)
        assert np.allclose(got, ref, rtol=0, atol=1e-12), subs
        assert np.abs(np.asarray(got) - np.asarray(ref)).max() < 1e-13, subs


def test_every_module_all_is_accurate():
    """``__all__`` must name things that exist, and not omit a module's own public
    callables.

    The first half is what breaks ``from x import *``; the second is how
    ``modetree_peak_bond`` ended up exported by import but not by ``__all__``, next
    to its two siblings that were.  Both are cheap to check and neither is visible
    in a passing test suite otherwise.
    """
    import importlib
    import pkgutil

    import fishbonett

    missing, undeclared = [], []
    names = [m.name for m in pkgutil.walk_packages(fishbonett.__path__, "fishbonett.")
             if not any(p.startswith("_") for p in m.name.split(".")[1:])]
    for name in sorted(set(names + ["fishbonett"])):
        if name == "fishbonett.rsvd_cupy":          # needs CuPy
            continue
        mod = importlib.import_module(name)
        declared = getattr(mod, "__all__", None)
        if declared is None:
            continue
        missing += [f"{name}.__all__ names {n!r}, which does not exist"
                    for n in declared if not hasattr(mod, n)]
        for n in dir(mod):
            if n.startswith("_") or n in declared:
                continue
            obj = getattr(mod, n)
            if getattr(obj, "__module__", None) == name and callable(obj):
                undeclared.append(f"{name}.{n}")
    assert not missing, "\n".join(missing)
    assert not undeclared, (
        "public callables defined in a module but left out of its __all__: "
        + ", ".join(undeclared))


def test_public_api_surface():
    import fishbonett as fb
    for name in ("SystemBathMPS", "TreeTensorNetwork", "SystemBath", "Fishbone",
                 "TreeFishbone", "Bath", "Result", "Truncation",
                 "get_bath_nn_paras", "get_coupling", "lanczos",
                 "sigma_x", "sigma_z", "drude", "lorentzian"):
        assert hasattr(fb, name), name
    # every advertised name must resolve, or `from fishbonett import *` breaks
    missing = [n for n in fb.__all__ if not hasattr(fb, n)]
    assert not missing, missing


def test_removed_comb_engine_is_gone():
    """FishBoneNet/FishBoneH/SystemBath1D/SystemBathSchrodinger were unreachable
    from run() and exercised only by a name check; the comb geometry is covered by
    Fishbone -> TreeTensorNetwork, which is validated against exact diagonalization."""
    import fishbonett as fb
    for name in ("FishBoneNet", "FishBoneH", "SystemBath1D",
                 "SystemBathSchrodinger", "init_ttn"):
        assert not hasattr(fb, name), f"{name} should have been removed"
    for mod in ("fishbonett.states.comb", "fishbonett.evolve.tebd_comb"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
    # representations.schrodinger exists again, but as the *representation* rather than a pair of
    # (model, representation) builder classes -- it emits LocalTerms for any topology.
    sch = importlib.import_module("fishbonett.representations.schrodinger")
    assert hasattr(sch, "terms")
    for gone in ("FishBoneH", "SystemBathSchrodinger"):
        assert not hasattr(sch, gone), f"{gone} should not have come back"


def test_mps_and_tree_are_one_tensor_network():
    """Both state containers are the same loop-free tensor network.

    They used to share no code: the MPS stores legs ``(vL, p, vR)`` and the tree
    ``(bonds..., p)``, and every method hard-coded axis positions.  That is a
    storage convention, not a difference in the mathematics -- so the topology, the
    canonical-form machinery and the observables now live once in ``TensorNetwork``
    and each container supplies ``tensor`` / ``set_tensor`` / ``neighbours``.
    """
    import numpy as np
    from fishbonett import Bath
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.states.tree import TreeTensorNetwork
    from fishbonett.states.network import TensorNetwork
    from fishbonett.evolve import tebd
    from fishbonett.representations.polaron import PolaronRepresentation
    from fishbonett.operators import sigma_x, sigma_z

    assert issubclass(SystemBathMPS, TensorNetwork)
    assert issubclass(TreeTensorNetwork, TensorNetwork)

    pd = [2] + [8] * 5
    compiled = Bath(
        J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
        n_modes=5, phys_dim=8).bind(sigma_z).compiled_polaron()
    builder = PolaronRepresentation(
        representation="polaron-chain", h_sys=0.5 * sigma_x,
        coupling=sigma_z, compiled_polaron=compiled).build()
    gates = builder.tebd_gates(0.01)
    st = SystemBathMPS(pd)
    for _ in range(3):
        tebd.symmetric_static_step(st, gates, len(pd) - 1, 40, 1e-9)

    # the chain's topology is inferred, not special-cased
    assert st.neighbours(0) == [1]
    assert st.neighbours(3) == [2, 4]
    assert st.neighbours(len(pd) - 1) == [len(pd) - 2]
    assert st.path(0, 4) == [0, 1, 2, 3, 4]

    # tensor() presents (bonds..., phys) whatever the storage order is
    assert st.tensor(3).shape == (st.B[3].shape[0], st.B[3].shape[2], st.B[3].shape[1])

    # and the inherited RDM reproduces the contraction the models used to inline
    for i in (0, 2, len(pd) - 1):
        theta = st.get_theta1(i)
        inline = np.einsum("LiR,LjR->ij", theta, theta.conj())
        inline /= np.trace(inline).real
        assert np.allclose(st.rdm(i), inline, atol=1e-12)
        assert abs(np.trace(st.rdm(i)).real - 1.0) < 1e-12

    # expectation() works on an MPS now; it only existed on the tree before
    assert np.isclose(st.expectation(sigma_z, 0),
                      np.einsum('ij,ji->', st.rdm(0), sigma_z).real)

    # Vidal form means every site is already canonical, so choosing a centre moves
    # no data -- it only records which gauge view the observables read from
    B_before = [b.copy() for b in st.B]
    st._prepare_for(3)
    assert st.oc == 3
    assert all(np.array_equal(a, b) for a, b in zip(B_before, st.B))


def test_mps_joint_rdm_matches_the_dense_state():
    """Multi-site observables on the MPS, against the full state vector.

    The inherited ``joint_rdm`` cannot contract ``tensor(n)`` at every node of the
    subtree the way the mixed-canonical tree does: in Vidal form that tensor carries
    the bond weights, so an internal bond would count ``Lambda`` twice (a ~9e-2 error
    on a nearest-neighbour pair, i.e. not subtle).  The centre goes to the lowest
    node of the subtree and every node above it contributes its bare right-isometry.
    """
    import numpy as np
    from fishbonett import Bath
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.evolve import tebd
    from fishbonett.representations.polaron import PolaronRepresentation
    from fishbonett.operators import sigma_x, sigma_z

    pd = [2, 3, 3, 3, 3]
    compiled = Bath(
        J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
        n_modes=4, phys_dim=3).bind(sigma_z).compiled_polaron()
    builder = PolaronRepresentation(
        representation="polaron-chain", h_sys=0.5 * sigma_x,
        coupling=sigma_z, compiled_polaron=compiled).build()
    gates = builder.tebd_gates(0.05)
    st = SystemBathMPS(pd)
    for _ in range(4):                     # entangle it; a product state proves nothing
        tebd.symmetric_static_step(st, gates, len(pd) - 1, 40, 1e-12)

    # the dense wavefunction: psi = B_0 B_1 ... B_{n-1}, with each R put back
    psi = np.einsum('KI,aIb->aKb', st.R[0],
                    np.tensordot(np.diag(st.S[0]), st.B[0], [1, 0]))
    for i in range(1, len(pd)):
        psi = np.tensordot(psi, np.einsum('KI,aIb->aKb', st.R[i], st.B[i]),
                           [psi.ndim - 1, 0])
    psi = psi[0, ..., 0]
    psi /= np.linalg.norm(psi)

    def exact(sites):
        rest = [k for k in range(len(pd)) if k not in sites]
        d = int(np.prod([pd[k] for k in sites]))
        p = np.transpose(psi, list(sites) + rest).reshape(d, -1)
        rho = p @ p.conj().T
        return rho / np.trace(rho).real

    for sites in ([0, 1], [0, 2], [1, 3], [0, 4], [2, 1], [1, 2, 3], [0, 2, 4]):
        assert np.allclose(st.joint_rdm(sites), exact(sites), atol=1e-12), sites

    # a one-site operator padded to two sites must agree with the one-site value
    assert np.isclose(st.expectation(np.kron(sigma_z, np.eye(3)), [0, 2]),
                      st.expectation(sigma_z, 0))


@pytest.mark.parametrize("model_key,method", [
    ("system-bath", "interaction-chain-tebd"),
    ("system-bath", "interaction-chain-trotter-mpo"),
    ("system-bath", "polaron-chain-tebd"),
    ("site-tree", "schrodinger-chain-tree-tebd"),
    ("system-bath", "interaction-chain-tree-tebd"),
])
def test_gate_methods_are_second_order_in_dt(model_key, method):
    """``evolve``'s claim that every whole step is second order (Strang), measured.

    This is the check that caught ``schrodinger-chain-tree-tebd`` propagating at order 1.07
    while the docs claimed second order, so it is worth having as a test rather
    than as a one-off measurement.

    Only the *gate-based* methods are listed.  The static-representation TDVP ones are
    second order too, but their error at a usable ``dt`` sits at 1e-8..1e-10 --
    below the Krylov and round-off floor -- so a Richardson ratio there measures
    noise, not the method.  Verified separately at larger ``dt``, where they come
    out at order >= 2.
    """
    import numpy as np
    from fishbonett import Bath, SystemBath
    from fishbonett.models import TreeFishbone
    from fishbonett.operators import sigma_x, sigma_z

    J = lambda w: 0.2 * w * np.exp(-w / 5.0)
    mk_bath = lambda: Bath(J=J, domain=(0.0, 40.0), n_modes=3, phys_dim=4)
    h = 0.5 * sigma_x

    def final_rdm(dt):
        if model_key == "site-tree":
            obj = TreeFishbone(sites=[h], edges=[], baths=[mk_bath()])
        else:
            obj = SystemBath(h=h, coupling=sigma_z, bath=mk_bath())
        r = obj.run(dt=dt, n_steps=int(round(0.4 / dt)), method=method,
                    bond_dim=40, trunc_eps=1e-12, observables={"sz": sigma_z})
        rho = np.asarray(r.rdm[-1])
        return rho[0] if rho.dtype == object else rho

    a, b, c = final_rdm(0.05), final_rdm(0.025), final_rdm(0.0125)
    d1, d2 = np.abs(a - b).max(), np.abs(b - c).max()
    order = np.log2(d1 / d2)
    assert order > 1.7, (
        f"{model_key}/{method} converges at order {order:.2f}, not 2 "
        f"(|rho(2dt)-rho(dt)|={d1:.2e}, |rho(dt)-rho(dt/2)|={d2:.2e})")


def test_interaction_representations_materialize_swap_network_gates():
    """Both interaction representation classes own the same gate contract."""
    from fishbonett.representations.interaction import InteractionRepresentation
    from fishbonett.representations.multichannel import MultichannelInteractionRepresentation

    for cls in (InteractionRepresentation, MultichannelInteractionRepresentation):
        assert "tebd_gates" in cls.__dict__


def test_interaction_graph_is_a_star_while_the_state_is_a_path():
    """Why the swap layout exists, asserted rather than described.

    The claim the whole model/representation/layout split rests on is that H's *interaction*
    graph and the state's *tensor-network* graph are different objects.  Here they
    demonstrably are: in the interaction picture every mode couples to the system
    and to nothing else (a star), while the MPS holding the state is a path.  The
    swap network is what reconciles them -- which is why, and only why, these
    methods are marked ``layout="swap"``.
    """
    import numpy as np
    from fishbonett import Bath
    from fishbonett.representations.interaction import (
        InteractionRepresentation, star_edges,
    )
    from fishbonett.models import registry as R
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.operators import sigma_x, sigma_z

    n = 5
    pd = [2] + [4] * n
    compiled = Bath(
        J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
        n_modes=n, phys_dim=4).bind(sigma_z).compiled_star()
    builder = InteractionRepresentation(
        representation="interaction-chain", h_sys=0.5 * sigma_x,
        coupling=sigma_z, compiled_star=compiled).build()

    # one two-site term per star edge, each pairing a mode with the system
    h2 = builder.two_site_hamiltonians(0.0, 0.01)
    edges = star_edges(n)
    assert len(h2) == len(edges) == n
    for (_h, d_boson, d_sys) in h2:
        assert (d_sys, d_boson) == (pd[0], pd[1])   # system x mode, never mode x mode

    # the state is a path, so the two graphs genuinely differ
    state = SystemBathMPS(pd)
    path_edges = {(i, i + 1) for i in range(n)}
    assert set(edges) != path_edges
    assert state.neighbours(3) == [2, 4]            # a path, not a star
    shared = set(edges) & path_edges
    assert shared == {(0, 1)}, "only the nearest mode is adjacent to the system"

    # and that mismatch is exactly what the registry derives: a star *interaction
    # graph* on a path geometry is what a swap network costs.  Note the representation is
    # `interaction-chain` -- the modes are chain modes; it is rotating H_B away that
    # makes the coupling reach all of them, which is what `mode_decoupled` records.
    assert R.METHODS["interaction-chain-tebd"].representation == "interaction-chain"
    assert R.REPRESENTATIONS["interaction-chain"].mode_decoupled
    assert R.METHODS["interaction-chain-tebd"].geometry == "path"
    assert R.METHODS["interaction-chain-tebd"].application == "swap"
    assert R.APPLICATIONS["swap"].startswith("a star realized on a path")


def test_schrodinger_representation_serves_every_topology():
    """The point of Stage 3: one representation implementation, any geometry.

    The multi-site models used to build their static Hamiltonian inline, bypassing
    representations/ entirely, which is why the package could hold a `representations` directory that
    half the models never touched.
    """
    import numpy as np
    from fishbonett import Bath, Fishbone
    from fishbonett.models import TreeFishbone
    from fishbonett.representations.schrodinger import LocalTerms
    from fishbonett.operators import sigma_x, sigma_z

    J = lambda w: 0.2 * w * np.exp(-w / 5.0)
    mk = lambda: Bath(J=J, domain=(0.0, 40.0), n_modes=2, phys_dim=4,
                      coupling=sigma_z)
    C = 0.3 * np.kron(sigma_z, sigma_z)
    h = 0.5 * sigma_z + sigma_x

    comb = Fishbone(sites=[h, h], baths=[mk(), mk()], backbone=[C])
    tree = TreeFishbone(sites=[h, h], edges=[(0, 1, C)], baths=[mk(), mk()])
    a, b = comb.local_terms(), tree.local_terms()

    assert isinstance(a, LocalTerms) and isinstance(b, LocalTerms)
    assert a.dims == b.dims and a.edges == b.edges
    assert all(np.allclose(x, y) for x, y in zip(a.site, b.site))
    assert set(a.bond) == set(b.bond)
    assert all(np.allclose(a.bond[k], b.bond[k]) for k in a.bond)

    # a zero on-site term becomes None, not an identity gate, so the propagators
    # can skip it
    site_gates, edge_gates = a.tebd_gates(0.01)
    assert len(site_gates) == a.n_nodes and len(edge_gates) == len(a.edges)
