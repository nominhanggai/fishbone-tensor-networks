"""Unit tests for the orthpol-free bath discretization / chain mapping."""
import numpy as np
import pytest

from fishbonett.bath.recurrence import recurrence_coefficients
from fishbonett.bath.chain import get_bath_nn_parameters, get_coupling


def test_recurrence_matches_analytic_legendre():
    """Constant J = pi on [-1, 1] gives h^2(x) = 1, whose monic orthogonal
    polynomials are the Legendre polynomials with the exact recurrence
    alpha_k = 0, beta_0 = 2, beta_k = k^2 / (4 k^2 - 1)."""
    alpha, beta = recurrence_coefficients(
        lambda w: np.pi, 11, (-1.0, 1.0)
    )
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    beta_exact = np.array([2.0] + [k * k / (4 * k * k - 1) for k in range(1, len(beta))])

    assert alpha.shape == beta.shape == (11,)
    np.testing.assert_allclose(alpha, 0.0, atol=1e-10)
    np.testing.assert_allclose(beta, beta_exact, rtol=1e-9, atol=1e-12)


def test_recurrence_needs_no_orthpol():
    import importlib.util
    assert importlib.util.find_spec("orthpol") is None
    # The call path must succeed regardless.
    alpha, beta = recurrence_coefficients(
        lambda w: w * np.exp(-w / 5.0), 5, (0.0, 10.0))
    assert len(alpha) == len(beta) == 5
    assert np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta))
    assert beta[0] > 0  # zeroth moment (system-bath coupling squared) is positive


def test_native_recurrence_route_matches_star_lanczos_route():
    density = lambda w: (0.2 + w) * np.exp(-w / 4.0)
    domain = (0.0, 12.0)

    direct_w, direct_k = get_coupling(density, 8, domain)
    star_w, star_k = get_bath_nn_parameters(density, 8, domain)

    np.testing.assert_allclose(direct_w, star_w, atol=1e-12)
    np.testing.assert_allclose(direct_k, star_k, atol=1e-12)


def test_interaction_representation_starts_from_a_finite_star():
    """Star discretization precedes the interaction and chain transformations."""
    from fishbonett import Bath
    from fishbonett.representations.interaction import InteractionRepresentation

    n_boson = 4
    coupling = np.diag([1.0, -1.0])
    bath = Bath(
        J=lambda w: 0.5 * w * np.exp(-w / 10.0),
        domain=(0.0, 50.0), n_modes=n_boson, phys_dim=6,
    )
    eth = InteractionRepresentation(
        representation="interaction-chain",
        h_sys=10.0 * np.array([[0.0, 1.0], [1.0, 0.0]]),
        coupling=coupling, bath=bath).build()

    assert len(eth.frequencies) == n_boson
    assert len(eth.star_couplings) == n_boson
    assert np.all(np.isfinite(eth.frequencies))
    assert np.all(np.isfinite(eth.star_couplings))
    np.testing.assert_allclose(
        eth.star_to_chain @ eth.star_to_chain.T,
        np.eye(n_boson), atol=1e-10)

    star_values = eth.star_couplings * np.exp(-1j * eth.frequencies * 0.37)
    np.testing.assert_allclose(
        eth.coefficients(0.37),
        eth.star_to_chain @ star_values)


def test_star_transform_has_one_implementation():
    """The star->chain transform is representation input, not evolution logic.

    The balanced-tree driver consumes interval coefficients from a representation;
    it neither discretizes the bath nor repeats the star-to-chain transform.
    """
    import ast
    import inspect
    from fishbonett.evolve import _modetree_driver as driver

    assert next(iter(inspect.signature(driver.run_tree_tebd).parameters)) == (
        "representation")
    src = inspect.getsource(driver)
    assert not any(isinstance(n, ast.ImportFrom) and n.module
                   and n.module.startswith("fishbonett.bath")
                   for n in ast.walk(ast.parse(src))), \
        "evolution must not discretize a bath"


def test_shared_mode_star_is_the_one_multichannel_discretization():
    """All multichannel representations must use one shared star grid."""
    from fishbonett import Bath
    from fishbonett.bath.legendre import get_vn_squared
    from fishbonett.operators import sigma_x, sigma_z
    from fishbonett.representations.multichannel import (
        MultichannelInteractionRepresentation,
    )

    Ja = lambda w: 0.2 * w * np.exp(-w / 5.0)
    Jb = lambda w: 0.1 * w * np.exp(-w / 5.0)
    bath = Bath(
        J=[Ja, Jb], domain=(0.0, 30.0), n_modes=4, phys_dim=3)
    representation = MultichannelInteractionRepresentation(
        representation="interaction-chain", h_sys=0.5 * sigma_z,
        coupling=[sigma_z, sigma_x], bath=bath).build()
    freq, coup_mat = representation.freq, representation.coup_mat_np

    # the grid is the channels' shared Gauss-Legendre one
    wa, va = get_vn_squared(Ja, 4, [0.0, 30.0])
    wb, vb = get_vn_squared(Jb, 4, [0.0, 30.0])
    np.testing.assert_allclose(freq, wa)
    np.testing.assert_allclose(wa, wb)          # nodes ignore J, so they coincide

    # mode k couples through the single combined operator sum_c g_{c,k} O_c
    assert len(coup_mat) == 4
    for k in range(4):
        expected = (np.sqrt(va[k] / np.pi) * sigma_z
                    + np.sqrt(vb[k] / np.pi) * sigma_x)
        np.testing.assert_allclose(coup_mat[k], expected, atol=1e-12)
        np.testing.assert_allclose(coup_mat[k], coup_mat[k].conj().T)

    # and the representation reads exactly this, one edge per mode
    from fishbonett.representations.schrodinger import star_terms
    dims, edges, site_H, edge_H = [2], [], [0.5 * sigma_z], {}
    assert star_terms(
        bath.bind([sigma_z, sigma_x]),
        0, 1, dims, edges, site_H, edge_H) == 5
    assert edges == [(0, 1), (0, 2), (0, 3), (0, 4)]

    # measure-adapted nodes are per-density, so they cannot be shared
    with pytest.raises(ValueError, match="legendre"):
        MultichannelInteractionRepresentation(
            representation="interaction-chain", h_sys=0.5 * sigma_z,
            coupling=[sigma_z, sigma_x],
            bath=Bath(
                J=[Ja, Jb], domain=(0.0, 30.0), n_modes=4, phys_dim=3,
                discretization="tedopa"))
