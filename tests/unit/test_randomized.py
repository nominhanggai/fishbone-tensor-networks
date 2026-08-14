"""Randomized-SVD accuracy, shape and run-local reproducibility."""
import numpy as np
import pytest

from fishbonett.randomized import randomized_svd, random_seed


@pytest.mark.parametrize("shape", [(40, 18), (18, 40)])
@pytest.mark.parametrize("complex_input", [False, True])
def test_randomized_svd_recovers_a_known_low_rank_matrix(shape, complex_input):
    rng = np.random.default_rng(12)
    left = rng.standard_normal((shape[0], 5))
    right = rng.standard_normal((5, shape[1]))
    if complex_input:
        left = left + 1j * rng.standard_normal(left.shape)
        right = right + 1j * rng.standard_normal(right.shape)
    matrix = left @ right

    U, values, Vh = randomized_svd(
        matrix, 5, n_iter=2, oversample=4, rng=np.random.default_rng(4))

    assert U.shape == (shape[0], 5)
    assert values.shape == (5,)
    assert Vh.shape == (5, shape[1])
    assert np.linalg.norm(matrix - (U * values) @ Vh) / np.linalg.norm(matrix) < 1e-10


def test_run_local_seed_is_reproducible_and_isolated():
    matrix = np.diag(np.geomspace(1.0, 1e-8, 30))
    with random_seed(7):
        first = randomized_svd(matrix, 4, n_iter=0, oversample=1)
    with random_seed(7):
        second = randomized_svd(matrix, 4, n_iter=0, oversample=1)
    with random_seed(8):
        third = randomized_svd(matrix, 4, n_iter=0, oversample=1)

    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not np.array_equal(first[0], third[0])


def test_invalid_randomized_svd_arguments_are_rejected():
    with pytest.raises(ValueError, match="two-dimensional"):
        randomized_svd(np.ones(3), 1)
    with pytest.raises(ValueError, match="rank"):
        randomized_svd(np.eye(3), 0)
    with pytest.raises(ValueError, match="n_iter"):
        randomized_svd(np.eye(3), 1, n_iter=-1)
