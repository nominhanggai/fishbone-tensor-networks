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
    # Sized off EXACT_BELOW so this keeps exercising the *randomized* path: below
    # that threshold randomized_svd defers to the exact SVD, and every seed then
    # agrees -- which would make the "different seeds differ" half vacuous.
    from fishbonett.randomized import EXACT_BELOW

    matrix = np.diag(np.geomspace(1.0, 1e-8, 2 * EXACT_BELOW))
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


def test_small_matrices_use_the_exact_svd():
    """Below :data:`EXACT_BELOW` the sketch is both slower and non-deterministic,
    so it is skipped.  Measured at 48x48 keeping 12: 0.366 ms randomized against
    0.258 ms exact -- 30% *more* time for a worse answer."""
    from fishbonett.randomized import EXACT_BELOW

    matrix = np.diag(np.geomspace(1.0, 1e-6, EXACT_BELOW // 2))
    # identical results from generators that would otherwise disagree
    a = randomized_svd(matrix, 4, rng=np.random.default_rng(1))
    b = randomized_svd(matrix, 4, rng=np.random.default_rng(999))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    exact = np.linalg.svd(matrix, full_matrices=False)[1][:4]
    assert np.allclose(a[1], exact)


def test_a_run_is_reproducible_by_default():
    """Randomized truncation is an internal optimization; it must not make an
    observable depend on when the run happened.

    Regression test: the golden characterization harness compares with ``==``,
    and before ``seed`` defaulted to 0 it failed against a baseline captured from
    the *same* code -- ~7e-08 in <sz> and ~9e-07 in the RDM between repeats.
    """
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    def run(**kw):
        bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5.0), domain=(0.0, 40.0),
                    n_modes=6, phys_dim=6)
        model = SystemBath(h=0.5 * sigma_x, coupling=sigma_z, bath=bath)
        r = model.run(dt=0.02, n_steps=12, bond_dim=12, trunc_eps=1e-10,
                      observables={"sz": sigma_z},
                      method="schrodinger-chain-tdvp1", **kw)
        return np.asarray(r.expect["sz"]), np.asarray(r.rdm)

    first, first_rdm = run()
    again, again_rdm = run()
    assert np.array_equal(first, again)          # bit-identical, not just close
    assert np.array_equal(first_rdm, again_rdm)

    # an explicit seed reproduces too, and a different one is still valid physics
    assert np.array_equal(run(seed=5)[0], run(seed=5)[0])
    assert np.allclose(run(seed=5)[0], first, atol=1e-5)
