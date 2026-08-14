"""Smoke tests: each method family builds and runs on a tiny problem."""
import importlib

import numpy as np
import pytest


def test_chain_cooling_gives_normalized_rdm():
    from fishbonett.frames.coolingchain import SystemBathCoolingChain
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


@pytest.mark.parametrize("module,cls", [("frames.coolingchain", "SystemBathCoolingChain")])
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
        method="polaron", dt=0.05, n_steps=3, bond_dim=30)
    assert np.all(np.isfinite(r.rdm))
    assert np.allclose([np.trace(rho).real for rho in r.rdm], 1.0, atol=1e-6)


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
    # frames.schrodinger exists again, but as the *frame* rather than a pair of
    # (model, frame) builder classes -- it emits LocalTerms for any topology.
    sch = importlib.import_module("fishbonett.frames.schrodinger")
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
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.states.tree import TreeTensorNetwork
    from fishbonett.states.network import TensorNetwork
    from fishbonett.evolve import tebd
    from fishbonett.frames.polaron import SystemBathPolaron
    from fishbonett.operators import sigma_x, sigma_z

    assert issubclass(SystemBathMPS, TensorNetwork)
    assert issubclass(TreeTensorNetwork, TensorNetwork)

    pd = [2] + [8] * 5
    builder = SystemBathPolaron(pd, h_sys=0.5 * sigma_x, coupling=sigma_z,
                                sd=lambda w: 0.3 * w * np.exp(-w / 2.5),
                                domain=(0.3, 12.0)).build()
    st = SystemBathMPS(pd)
    for _ in range(3):
        tebd.symmetric_static_step(st, builder.gates(0.01), len(pd) - 1, 40, 1e-9)

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
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.evolve import tebd
    from fishbonett.frames.polaron import SystemBathPolaron
    from fishbonett.operators import sigma_x, sigma_z

    pd = [2, 3, 3, 3, 3]
    builder = SystemBathPolaron(pd, h_sys=0.5 * sigma_x, coupling=sigma_z,
                                sd=lambda w: 0.3 * w * np.exp(-w / 2.5),
                                domain=(0.3, 12.0)).build()
    st = SystemBathMPS(pd)
    for _ in range(4):                     # entangle it; a product state proves nothing
        tebd.symmetric_static_step(st, builder.gates(0.05), len(pd) - 1, 40, 1e-12)

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


def test_swap_network_frames_share_one_get_u():
    """The two swap-network frames differ in ``get_h2`` and in nothing after it.

    Found by an AST scan for duplicate function bodies, which is also how the
    star->chain transform turned up.  Both frames had the same two-line ``get_u``;
    it is the mixin's now, and the contract it states -- supply ``get_h2``, receive
    swap-network gates -- is the whole of what they share.
    """
    from fishbonett.frames.gates import SwapNetworkFrame
    from fishbonett.frames.interaction_picture import SystemBathIP
    from fishbonett.frames.multichannel import SystemBathMultiChannel

    for cls in (SystemBathIP, SystemBathMultiChannel):
        assert issubclass(cls, SwapNetworkFrame)
        assert cls.get_u is SwapNetworkFrame.get_u, f"{cls.__name__} re-declares get_u"
        assert cls.get_h2 is not SwapNetworkFrame.get_h2, (
            f"{cls.__name__} must supply its own get_h2 -- that is the contract")


def test_interaction_graph_is_a_star_while_the_state_is_a_path():
    """Why the swap layout exists, asserted rather than described.

    The claim the whole model/frame/layout split rests on is that H's *interaction*
    graph and the state's *tensor-network* graph are different objects.  Here they
    demonstrably are: in the interaction picture every mode couples to the system
    and to nothing else (a star), while the MPS holding the state is a path.  The
    swap network is what reconciles them -- which is why, and only why, these
    methods are marked ``layout="swap"``.
    """
    import numpy as np
    from fishbonett.frames.gates import star_edges
    from fishbonett.frames.interaction_picture import SystemBathIP
    from fishbonett.models import registry as R
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.operators import sigma_x, sigma_z

    n = 5
    pd = [2] + [4] * n
    builder = SystemBathIP(pd, h_sys=0.5 * sigma_x, coupling=sigma_z,
                           sd=lambda w: 0.3 * w * np.exp(-w / 2.5),
                           domain=(0.3, 12.0)).build()

    # one two-site term per star edge, each pairing a mode with the system
    h2 = builder.get_h2(0.0, 0.01)
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

    # and that mismatch is exactly what the registry records
    assert R.METHODS["tebd"].layout == "swap"
    assert R.LAYOUTS["swap"].startswith("a star realized on a path")


def test_schrodinger_frame_serves_every_topology():
    """The point of Stage 3: one frame implementation, any geometry.

    The multi-site models used to build their static Hamiltonian inline, bypassing
    frames/ entirely, which is why the package could hold a `frames` directory that
    half the models never touched.
    """
    import numpy as np
    from fishbonett import Bath, Fishbone
    from fishbonett.models import TreeFishbone
    from fishbonett.frames.terms import LocalTerms
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
    site_gates, edge_gates = a.gates(0.01)
    assert len(site_gates) == a.n_nodes and len(edge_gates) == len(a.edges)
