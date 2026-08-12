"""Swap-network TEBD propagation for a matrix-product state.

The state (:class:`fishbonett.states.mps.SpinBosonMPS`) holds the tensors and
their canonical form; this module holds the *algorithm* -- applying the two-site
Trotter gate stored on the state and re-splitting the bond.  Keeping the two
separate is why the state module is called ``states.mps`` and this one
``evolve.tebd``.
"""
from fishbonett.contract import contract as einsum

try:  # optional GPU backend (only used when a state requests gpu=True)
    import cupy as cp
    _CUPY = True
except ImportError:  # pragma: no cover - exercised only with a GPU present
    _CUPY = False

__all__ = ["update_bond"]


def update_bond(state, i, chi_max, eps, swap=0, eps_lbo=None, adaptive=False,
                gpu=False):
    """Apply the two-site gate ``state.U[i]`` at bond ``i`` and re-split ``state``.

    Parameters
    ----------
    state : fishbonett.states.mps.SpinBosonMPS
        The MPS whose bond ``i`` is updated in place.
    swap : {0, 1}
        1 transposes the two physical legs during the gate (moves a distant bath
        mode next to the system site -- the interaction / "backward" picture).
    eps_lbo : float, optional
        Local-basis-optimization threshold; enables LBO and the adaptive search.
    adaptive : bool
        Adaptive bond-dimension search without LBO.  Ignored when ``eps_lbo`` is
        given.  Default is a single truncated SVD at ``chi_max``.
    """
    use_gpu = bool(gpu and _CUPY)
    theta = state.get_theta2(i, gpu=use_gpu)
    u_bond = cp.array(state.U[i]) if use_gpu else state.U[i]
    if swap == 1:
        utheta = einsum('ijkl,PklQ->PjiQ', u_bond, theta)
    elif swap == 0:
        utheta = einsum('ijkl,PklQ->PijQ', u_bond, theta)
    else:
        raise ValueError(f"swap must be 0 or 1, got {swap!r}")
    state.split_truncate_theta(utheta, i, chi_max, eps, eps_lbo=eps_lbo,
                               adaptive=adaptive, gpu=use_gpu)
