"""Smoke tests: each method family builds and runs on a tiny problem."""

import numpy as np
import pytest


def test_polaron_builds_and_gives_normalized_rdm():
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    bath = Bath(J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
                n_modes=6, phys_dim=6)
    r = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=bath).run(
        method="polaron-chain-tebd", dt=0.05, n_steps=3, bond_dim=30)
    assert np.all(np.isfinite(r.rdm))
    assert np.allclose([np.trace(rho).real for rho in r.rdm], 1.0, atol=1e-6)


def test_every_module_all_is_accurate():
    """``__all__`` entries exist and include module-defined public callables."""
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
                 "get_bath_nn_parameters", "get_coupling", "lanczos",
                 "sigma_x", "sigma_z", "drude", "lorentzian"):
        assert hasattr(fb, name), name
    # every advertised name must resolve, or `from fishbonett import *` breaks
    missing = [n for n in fb.__all__ if not hasattr(fb, n)]
    assert not missing, missing


def test_mps_and_tree_are_one_tensor_network():
    """MPS and tree states implement the shared tensor-network interface."""
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
    bath = Bath(
        J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
        n_modes=5, phys_dim=8)
    builder = PolaronRepresentation(
        representation="polaron-chain", h_sys=0.5 * sigma_x,
        coupling=sigma_z, bath=bath).build()
    gates = builder.tebd_gates(0.01)
    st = SystemBathMPS(pd)
    for _ in range(3):
        tebd.symmetric_static_step(st, gates, len(pd) - 1, 40, 1e-9)

    # The MPS exposes its line topology through the common interface.
    assert st.neighbours(0) == [1]
    assert st.neighbours(3) == [2, 4]
    assert st.neighbours(len(pd) - 1) == [len(pd) - 2]
    assert st.path(0, 4) == [0, 1, 2, 3, 4]

    # ``tensor`` presents bond axes followed by the physical axis.
    assert st.tensor(3).shape == (st.B[3].shape[0], st.B[3].shape[2], st.B[3].shape[1])

    # The common RDM implementation agrees with the direct MPS contraction.
    for i in (0, 2, len(pd) - 1):
        theta = st.get_theta1(i)
        inline = np.einsum("LiR,LjR->ij", theta, theta.conj())
        inline /= np.trace(inline).real
        assert np.allclose(st.rdm(i), inline, atol=1e-12)
        assert abs(np.trace(st.rdm(i)).real - 1.0) < 1e-12

    # Expectations use the common RDM implementation.
    assert np.isclose(st.expectation(sigma_z, 0),
                      np.einsum('ij,ji->', st.rdm(0), sigma_z).real)

    # Choosing an orthogonality centre does not alter a Vidal-form state.
    B_before = [b.copy() for b in st.B]
    st._prepare_for(3)
    assert st.oc == 3
    assert all(np.array_equal(a, b) for a, b in zip(B_before, st.B))


def test_mps_joint_rdm_matches_the_dense_state():
    """MPS joint reduced density matrices agree with dense contractions."""
    import numpy as np
    from fishbonett import Bath
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.evolve import tebd
    from fishbonett.representations.polaron import PolaronRepresentation
    from fishbonett.operators import sigma_x, sigma_z

    pd = [2, 3, 3, 3, 3]
    bath = Bath(
        J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
        n_modes=4, phys_dim=3)
    builder = PolaronRepresentation(
        representation="polaron-chain", h_sys=0.5 * sigma_x,
        coupling=sigma_z, bath=bath).build()
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
    """Gate-based whole steps show second-order convergence.

    Static TDVP errors reach the Krylov and round-off floor at these step sizes,
    so this ratio test is restricted to gate-based integrators.
    """
    import numpy as np
    from fishbonett import Bath, SystemBath
    from fishbonett.models import TreeFishbone
    from fishbonett.operators import sigma_x, sigma_z

    J = lambda w: 0.2 * w * np.exp(-w / 5.0)
    lower = 0.2 if method.startswith("polaron-") else 0.0
    mk_bath = lambda: Bath(
        J=J, domain=(lower, 40.0), n_modes=3, phys_dim=4
    )
    h = 0.5 * sigma_x

    def final_rdm(dt):
        if model_key == "site-tree":
            obj = TreeFishbone(
                sites=[h], edges=[], baths=[mk_bath().bind(sigma_z)],
            )
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


def test_interaction_graph_is_a_star_while_the_state_is_an_mps():
    """A star interaction graph on an MPS requires a swap network."""
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
    bath = Bath(
        J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
        n_modes=n, phys_dim=4)
    builder = InteractionRepresentation(
        representation="interaction-chain", h_sys=0.5 * sigma_x,
        coupling=sigma_z, bath=bath).build()

    # one two-site term per star edge, each pairing a mode with the system
    h2 = builder.two_site_hamiltonians(0.0, 0.01)
    edges = star_edges(n)
    assert len(h2) == len(edges) == n
    for (_h, d_boson, d_sys) in h2:
        assert (d_sys, d_boson) == (pd[0], pd[1])   # system x mode, never mode x mode

    # the state is a 1D MPS, so the two graphs genuinely differ
    state = SystemBathMPS(pd)
    path_edges = {(i, i + 1) for i in range(n)}
    assert set(edges) != path_edges
    assert state.neighbours(3) == [2, 4]            # a path, not a star
    shared = set(edges) & path_edges
    assert shared == {(0, 1)}, "only the nearest mode is adjacent to the system"

    # The graph mismatch is handled by the swap engine.
    assert R.METHODS["interaction-chain-tebd"].representation == "interaction-chain"
    assert R.REPRESENTATIONS["interaction-chain"].mode_decoupled
    assert R.METHODS["interaction-chain-tebd"].state_geometry == "mps"
    assert R.METHODS["interaction-chain-tebd"].engine == "swap-tebd"


def test_schrodinger_representation_serves_every_topology():
    """The Schrodinger representation serves comb and general-tree models."""
    import numpy as np
    from fishbonett import Bath, Fishbone
    from fishbonett.models import TreeFishbone
    from fishbonett.representations.schrodinger import LocalTerms
    from fishbonett.operators import sigma_x, sigma_z

    J = lambda w: 0.2 * w * np.exp(-w / 5.0)
    mk = lambda: Bath(
        J=J, domain=(0.0, 40.0), n_modes=2, phys_dim=4).bind(sigma_z)
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
