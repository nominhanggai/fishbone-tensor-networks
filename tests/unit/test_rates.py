"""Unit tests for the rate-theory and diabatization tools."""
import numpy as np

from fishbonett.rates import marcus_rate, fgr_rate, transfer_mat
from fishbonett.diabatization import boys_func, diabatize
from fishbonett.bath.legendre import get_vn_squared


def test_vegas_rate_integrators_run_up_to_their_optional_dependency():
    """The two VEGAS integrators were executed by nothing at all.

    ``vegas`` is an optional extra that no test or CI job installs, so both
    functions in ``rates/golden_rule_multi`` that use it had never run -- the same
    situation as ``benchmarks/tree_engine.py``, which turned out to have been broken
    for a long time precisely because nothing ran it.

    This exercises everything up to the import: the argument unpacking, the
    displacement algebra, the coth prefactors and the integrand closures all
    execute, and only the missing package stops them.  That is most of the code and
    it is checkable without the extra.  It also pins the error message, which now
    names the extra instead of just saying "No module named 'vegas'".
    """
    import numpy as np
    import pytest

    from fishbonett.rates import golden_rule_multi as grm

    if _has_vegas():
        pytest.skip("vegas is installed; this checks the absent-dependency path")

    w = np.array([1.0, 2.0, 3.0])
    s_list = [np.full(3, 0.4), np.full(3, 0.1), np.full(3, 0.7)]
    c_list, e_list = [0.05, 0.05, 0.05], [0.0, 0.1, 0.2]

    for fn, extra in ((grm.fgr_rate3_correction_order2_vegas, ()),
                      (grm.fgr_rate3_correction_order_vegas, (2,))):
        with pytest.raises(ImportError, match=r"fishbonett\[rates\]"):
            fn(c_list, e_list, 1.0, w, s_list, 1.0, *extra, nitn=2, neval=50)


def _has_vegas():
    try:
        import vegas  # noqa: F401
    except ImportError:
        return False
    return True


def test_marcus_rate_matches_closed_form():
    r = marcus_rate(c=1.0, e=0.0, kbT=1.0, reorg_e=1.0)
    expected = 2 * np.pi / np.sqrt(4 * np.pi) * np.exp(-1.0 / 4.0)
    assert np.isclose(r, expected)


def test_fgr_rate_is_finite_and_positive():
    j = lambda w: 0.5 * (4 * 2.39e-2) * 3.5e-4 ** 2 * 1.2e-3 * w \
        / ((3.5e-4 ** 2 - w ** 2) ** 2 + 1.2e-3 ** 2 * w ** 2)
    w, v_sq = get_vn_squared(j, 60, [0, 5e-3])
    r = fgr_rate(c=5e-5, e=0.02, kbT=9.5e-4, _w=w, _v_sq=v_sq)
    assert np.isfinite(r) and r > 0.0


def test_diabatize_returns_orthogonal_rotation():
    mu = np.array([
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        [[0.5, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    ])
    u, mat = diabatize(mu, tol=1e-6)
    np.testing.assert_allclose(u @ u.T, np.eye(2), atol=1e-8)
    # localization should not decrease the Boys functional
    assert boys_func(mat) >= boys_func(mu) - 1e-9


def test_transfer_mat_first_tensor_is_first_map():
    rng = np.random.RandomState(0)
    maps = [np.eye(4), rng.rand(4, 4), rng.rand(4, 4)]
    T, T_norm = transfer_mat(maps)
    np.testing.assert_allclose(T[0], maps[0])
    assert len(T) == len(maps) and np.all(np.isfinite(T_norm))
