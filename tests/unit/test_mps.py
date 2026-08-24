"""Direct unit tests for the canonical TEBD engine (fishbonett.states.mps)."""

import numpy as np
import pytest
import scipy.linalg

from fishbonett.contract import contract as einsum
from fishbonett.states.mps import SystemBathMPS


def test_gpu_request_without_cupy_says_so_and_still_works():
    """``gpu=True`` without CuPy silently ran on the CPU.

    The fallback itself is right -- the results are identical, only slower -- but
    silence means the only symptom is an unaccountably slow run.  Same class as the
    VEGAS integrators: an optional extra whose absence changed behaviour without
    saying anything.  It now warns once and still produces the same state.
    """
    from fishbonett.states import mps as mps_mod
    from fishbonett.states.mps import SystemBathMPS

    if mps_mod._CUPY:                      # pragma: no cover - needs a GPU box
        pytest.skip("CuPy is installed; this checks the absent-backend path")

    pd = [2, 4, 4]
    theta = np.zeros((1, 2, 4, 4), dtype=complex)
    theta[0, 0, 0, 0] = 1.0

    cpu = SystemBathMPS(pd)
    cpu.split_truncate_theta(theta.copy(), 0, 8, 1e-10)

    asked = SystemBathMPS(pd)
    with pytest.warns(RuntimeWarning, match="CuPy wheel matching"):
        asked.split_truncate_theta(theta.copy(), 0, 8, 1e-10, gpu=True)

    # the warning is the only difference: the state is the same
    for a, b in zip(cpu.B, asked.B):
        assert np.array_equal(a, b)
    for a, b in zip(cpu.S, asked.S):
        assert np.array_equal(a, b)
def _random_gate(d1, d2, rng, dt=0.05):
    h = rng.rand(d1 * d2, d1 * d2)
    h = h + h.T
    return scipy.linalg.expm(-1j * dt * h).reshape(d1, d2, d1, d2)


@pytest.mark.parametrize("mode,kw", [
    ("plain", {}),
    ("adaptive", {"adaptive": True}),
    ("lbo", {"eps_lbo": 1e-10}),
])
def test_engine_preserves_normalization(mode, kw):
    np.random.seed(0)
    rng = np.random.RandomState(1)
    pd = [4, 4, 2]
    etn = SystemBathMPS(pd)
    etn.B[-1][0, 0, 0] = 1.0
    etn.U = [_random_gate(4, 4, rng), _random_gate(4, 2, rng)]

    for _ in range(3):
        for j in (0, 1):
            etn.update_bond(j, 16, 1e-12, swap=0, **kw)

    # Schmidt spectra stay normalized and finite
    for s in etn.S[1:]:
        assert np.isclose(np.linalg.norm(s), 1.0)
    assert all(np.all(np.isfinite(b)) for b in etn.B)

    # reduced density matrix of the system site has unit trace
    theta = etn.get_theta1(2)
    rho = einsum('LiR,LjR->ij', theta, theta.conj())
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-8)


def test_swap_moves_physical_leg():
    """A swap gate exchanges the two sites' physical dimensions."""
    np.random.seed(0)
    rng = np.random.RandomState(2)
    pd = [5, 2]  # boson(5) + spin(2)
    etn = SystemBathMPS(pd)
    etn.B[-1][0, 0, 0] = 1.0
    etn.U = [_random_gate(5, 2, rng)]
    etn.update_bond(0, 16, 1e-12, swap=1)
    # after the swap the physical dims at the two sites are exchanged
    assert etn.B[0].shape[1] == 2
    assert etn.B[1].shape[1] == 5
