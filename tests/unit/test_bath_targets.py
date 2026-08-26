"""Observable targets for represented bath modes."""

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from fishbonett import Bath, BathMode, Fishbone
from fishbonett.models.fishbone import _parse_observable
from fishbonett.operators import annihilate, sigma_x, sigma_y, sigma_z


def test_targeted_observable_is_not_probed_as_a_ragged_array():
    """An ``(operator, target)`` pair is recognized before NumPy conversion."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kind, operator, sites = _parse_observable(
            (sigma_z, 0), dimensions=(2,), name="z0"
        )
    assert not caught
    assert kind == "sites"
    assert np.array_equal(operator, sigma_z)
    assert sites == [0]

    kind, operator, sites = _parse_observable(
        ((1.0, 0.0), (0.0, -1.0)), dimensions=(2,), name="z"
    )
    assert kind == "persite"
    assert np.array_equal(operator, sigma_z)
    assert sites is None


def _density(scale):
    return lambda omega: scale * omega * np.exp(-omega / 2.0)


def _bath(scale=0.2, *, modes=1, dimension=3, operator=sigma_z):
    return Bath(
        J=_density(scale), domain=(0.0, 6.0), n_modes=modes,
        phys_dim=dimension,
    ).bind(operator)


def _embed_one(operator, node, dimensions):
    factors = [np.eye(dimension) for dimension in dimensions]
    factors[node] = operator
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _dense_hamiltonian(terms):
    dimensions = terms.dims
    size = int(np.prod(dimensions))
    result = np.zeros((size, size), complex)
    for node, operator in enumerate(terms.site):
        result += _embed_one(operator, node, dimensions)
    for (left, right), operator in terms.bond.items():
        shaped = operator.reshape(
            dimensions[left], dimensions[right],
            dimensions[left], dimensions[right],
        )
        for i in range(dimensions[left]):
            for j in range(dimensions[right]):
                for k in range(dimensions[left]):
                    for ell in range(dimensions[right]):
                        value = shaped[i, j, k, ell]
                        if value:
                            left_matrix = np.zeros((dimensions[left],) * 2, complex)
                            right_matrix = np.zeros((dimensions[right],) * 2, complex)
                            left_matrix[i, k] = 1.0
                            right_matrix[j, ell] = 1.0
                            result += value * (
                                _embed_one(left_matrix, left, dimensions)
                                @ _embed_one(right_matrix, right, dimensions)
                            )
    return result


def test_static_bath_and_mixed_observables_match_exact_diagonalization():
    model = Fishbone(
        sites=[0.4 * sigma_z + 0.7 * sigma_x],
        baths=[_bath()],
    )
    mode = BathMode(system_site=0)
    number = annihilate(3).T @ annihilate(3)
    position = annihilate(3) + annihilate(3).T
    mixed = np.kron(sigma_y, position)
    result = model.run(
        dt=0.005, n_steps=10, trunc_eps=1e-13,
        observables={"occupation": (number, mode),
                     "system_mode": (mixed, (0, mode))},
    )

    terms = model.local_terms(result.t[-1])
    hamiltonian = _dense_hamiltonian(terms)
    initial = np.zeros(int(np.prod(terms.dims)), complex)
    initial[0] = 1.0
    resolved_mode = terms.bath_nodes[mode]
    exact_number = _embed_one(number, resolved_mode, terms.dims)
    exact_mixed = (
        _embed_one(sigma_y, 0, terms.dims)
        @ _embed_one(position, resolved_mode, terms.dims)
    )
    expected_number, expected_mixed = [], []
    for time in result.t:
        state = expm(-1j * hamiltonian * time) @ initial
        expected_number.append(state.conj() @ exact_number @ state)
        expected_mixed.append(state.conj() @ exact_mixed @ state)

    np.testing.assert_allclose(result.expect["occupation"], expected_number,
                               atol=2e-6)
    np.testing.assert_allclose(result.expect["system_mode"], expected_mixed,
                               atol=2e-6)
    assert result.meta["observable_targets"] == {
        "occupation": (1,), "system_mode": (0, 1),
    }
    branch = result.meta["bath_branches"][0]
    assert {key: branch[key] for key in branch if key != "system_coupling"} == {
        "system_site": 0,
        "bath": 0,
        "representation": "schrodinger-chain",
        "first_node": 1,
        "n_modes": 1,
        "phys_dim": 3,
    }
    assert branch["system_coupling"] > 0


def test_two_baths_on_one_site_have_distinct_semantic_addresses():
    model = Fishbone(
        sites=[0.2 * sigma_x],
        baths=[[_bath(0.1, dimension=2),
                _bath(0.3, dimension=3, operator=sigma_x)]],
    )
    first = BathMode(0, bath=0, mode=0)
    second = BathMode(0, bath=1, mode=0)
    result = model.run(
        dt=0.01, n_steps=2, trunc_eps=1e-12,
        observables={
            "first": (np.diag([0.0, 1.0]), first),
            "second": (np.diag([0.0, 1.0, 2.0]), second),
        },
    )
    assert result.meta["observable_targets"] == {"first": (1,), "second": (2,)}
    assert [branch["bath"] for branch in result.meta["bath_branches"]] == [0, 1]
    assert [branch["phys_dim"] for branch in result.meta["bath_branches"]] == [2, 3]


def test_interaction_chain_targets_survive_checkpoint_resume():
    model = Fishbone(sites=[0.4 * sigma_x], baths=[_bath(modes=2)])
    target = BathMode(0, mode=1)
    number = annihilate(3).T @ annihilate(3)
    options = dict(
        dt=0.01, method="interaction-chain-fishbone-trotter-mpo",
        trunc_eps=1e-12, bath_horizon=0.04,
        observables={"mode_1": (number, target)},
    )
    first = model.run(n_steps=2, **options)
    continued = model.run(n_steps=2, resume=first.checkpoint, **options)
    complete = model.run(n_steps=4, **options)

    np.testing.assert_allclose(
        np.concatenate([first.expect["mode_1"], continued.expect["mode_1"]]),
        complete.expect["mode_1"], atol=1e-11,
    )
    assert complete.meta["observable_targets"] == {"mode_1": (2,)}
    assert complete.meta["bath_branches"][0]["representation"] == "interaction-chain"
    assert complete.meta["bath_branches"][0]["system_coupling"] is None


@pytest.mark.parametrize(
    "arguments, exception",
    [
        ({"system_site": True}, TypeError),
        ({"system_site": -1}, ValueError),
        ({"system_site": 0, "bath": 0.5}, TypeError),
        ({"system_site": 0, "mode": -1}, ValueError),
    ],
)
def test_bath_mode_rejects_invalid_indices(arguments, exception):
    with pytest.raises(exception):
        BathMode(**arguments)


def test_bath_target_validation_reports_address_and_dimension_errors():
    model = Fishbone(sites=[sigma_x], baths=[_bath(dimension=3)])
    with pytest.raises(ValueError, match="unavailable bath mode"):
        model.run(
            dt=0.01, n_steps=1,
            observables={"bad": (np.eye(3), BathMode(0, bath=1))},
        )
    with pytest.raises(ValueError, match=r"expected \(3, 3\)"):
        model.run(
            dt=0.01, n_steps=1,
            observables={"bad": (np.eye(2), BathMode(0))},
        )
    with pytest.raises(ValueError, match="same site more than once"):
        model.run(
            dt=0.01, n_steps=1,
            observables={"bad": (np.eye(9), (BathMode(0), BathMode(0)))},
        )
    with pytest.raises(TypeError, match="system-site integers or BathMode"):
        model.run(
            dt=0.01, n_steps=1,
            observables={"bad": (np.eye(2), "bath zero")},
        )
