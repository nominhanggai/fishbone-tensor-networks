"""Randomized-SVD accuracy, shape and run-local reproducibility."""
import numpy as np
import pytest

from fishbonett.randomized import (
    adaptive_svd, randomized_svd, random_seed, svd_statistics,
)


def test_robust_svd_retries_with_gesvd_after_nonconvergence(monkeypatch):
    """A failed divide-and-conquer SVD falls back without losing the result."""
    import fishbonett._svd as module

    matrix = np.diag([3.0, 2.0, 1.0])
    original = module.scipy.linalg.svd
    drivers = []

    def fail_gesdd_once(value, **options):
        driver = options["lapack_driver"]
        drivers.append(driver)
        if driver == "gesdd":
            raise np.linalg.LinAlgError("forced nonconvergence")
        return original(value, **options)

    monkeypatch.setattr(module.scipy.linalg, "svd", fail_gesdd_once)
    with pytest.warns(UserWarning, match="retrying with gesvd"):
        u, singular, vh = module.robust_svd(matrix, full_matrices=False)

    assert drivers == ["gesdd", "gesvd"]
    assert np.allclose((u * singular) @ vh, matrix)


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

    assert all(
        np.array_equal(a, b) for a, b in zip(first, second, strict=True)
    )
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
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
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


def _known_spectrum_matrix(seed=11):
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.standard_normal((180, 80)))
    right, _ = np.linalg.qr(rng.standard_normal((160, 80)))
    spectrum = np.r_[np.geomspace(1.0, 1e-3, 30), np.full(50, 1e-8)]
    return (left * spectrum) @ right.T, spectrum


def test_adaptive_svd_certifies_threshold_without_a_rank_cap():
    matrix, spectrum = _known_spectrum_matrix()
    with random_seed(4):
        u, values, vh, info = adaptive_svd(
            matrix, eps=1e-4, return_info=True)
        statistics = svd_statistics()

    assert info.backend == "randomized"
    assert not info.exact_fallback
    assert info.residual_norm <= 0.25 * info.cutoff
    assert len(values) == np.sum(spectrum > 1e-4 * spectrum[0])
    assert np.allclose(values, spectrum[:len(values)], rtol=1e-9, atol=1e-12)
    relative_error = np.linalg.norm(matrix - (u * values) @ vh) / np.linalg.norm(matrix)
    assert relative_error < 1e-7
    assert statistics["randomized_calls"] == 1
    assert statistics["exact_calls"] == 0
    assert statistics["maximum_retained_rank"] == len(values)


def test_adaptive_svd_handles_a_wide_complex_matrix():
    matrix, spectrum = _known_spectrum_matrix(seed=13)
    matrix = matrix.T.astype(complex)
    matrix *= np.exp(1j * np.linspace(0.0, 1.0, matrix.shape[0]))[:, None]
    matrix *= np.exp(-1j * np.linspace(0.0, 0.7, matrix.shape[1]))[None, :]

    u, values, vh, info = adaptive_svd(
        matrix, eps=1e-4, backend="randomized", return_info=True)

    assert info.backend == "randomized"
    assert len(values) == np.sum(spectrum > 1e-4 * spectrum[0])
    assert np.allclose(values, spectrum[:len(values)], rtol=1e-9, atol=1e-12)
    relative_error = np.linalg.norm(matrix - (u * values) @ vh) / np.linalg.norm(matrix)
    assert relative_error < 1e-7


def test_adaptive_svd_falls_back_when_the_threshold_requires_full_rank():
    matrix = np.diag(np.linspace(1.0, 0.5, 2 * 128))
    with random_seed(2):
        _, values, _, info = adaptive_svd(
            matrix, eps=1e-8, initial_rank=8, return_info=True)
        statistics = svd_statistics()

    assert info.backend == "exact"
    assert info.exact_fallback
    assert len(values) == matrix.shape[0]
    assert statistics["exact_calls"] == 1
    assert statistics["exact_fallbacks"] == 1


def test_adaptive_svd_uses_exact_fallback_for_an_ambiguous_cutoff():
    spectrum = np.r_[1.0, np.geomspace(1e-1, 1e-3, 12), 1e-4,
                     np.full(146, 1e-8)]
    matrix = np.diag(spectrum)
    _, values, _, info = adaptive_svd(
        matrix, eps=1e-4, backend="randomized", return_info=True)

    assert info.backend == "exact"
    assert info.exact_fallback
    assert len(values) == 13
    assert values[-1] > 1e-4


def test_adaptive_svd_matrix_key_is_checkpoint_segment_stable():
    matrix, _ = _known_spectrum_matrix(seed=15)
    with random_seed(7):
        first = adaptive_svd(matrix, eps=1e-4)
        # Other decompositions must not advance the sketch used for this matrix.
        adaptive_svd(1.01 * matrix, eps=1e-4)
        second = adaptive_svd(matrix, eps=1e-4)
    with random_seed(7):
        resumed = adaptive_svd(matrix, eps=1e-4)

    assert all(
        np.array_equal(a, b) for a, b in zip(first, second, strict=True)
    )
    assert all(
        np.array_equal(a, b) for a, b in zip(first, resumed, strict=True)
    )


def test_run_can_select_exact_svd_and_reports_decomposition_statistics():
    from fishbonett import Bath, SystemBath
    from fishbonett.operators import sigma_x, sigma_z

    bath = Bath(
        J=lambda w: 0.1 * w * np.exp(-w), domain=(0.0, 4.0),
        n_modes=3, phys_dim=3,
    )
    result = SystemBath(
        h=0.2 * sigma_x, coupling=sigma_z, bath=bath,
    ).run(
        dt=0.02, n_steps=2, method="schrodinger-chain-tdvp2",
        trunc_eps=1e-5, svd_backend="exact",
    )

    assert result.meta["svd_backend"] == "exact"
    assert result.meta["svd"]["exact_calls"] > 0
    assert result.meta["svd"]["randomized_calls"] == 0
