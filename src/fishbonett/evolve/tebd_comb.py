"""TEBD on the comb ("fishbone") tensor network: the bond update.

The comb counterpart of :mod:`fishbonett.evolve.tebd`.  It drives the states in
:mod:`fishbonett.states.comb` -- :class:`~fishbonett.states.comb.FishBoneNet`, whose
sites carry up to four legs (chain-left, chain-right, bath-up, bath-down), and the
minimal reference chain :class:`~fishbonett.states.comb.SystemBath1D`.

Only the *gate application* lives here.  Building the two-site wavefunction
(``get_theta2``) and re-splitting it into canonical form
(``split_truncate_theta``) are properties of the state and stay on those classes,
exactly as in :mod:`fishbonett.states.mps`.

.. rubric:: What's here

======================  ======================================================
:func:`update_bond`     one gate on one bond of a ``FishBoneNet``
:func:`update_bond_1d`  the same for the plain-chain ``SystemBath1D``
======================  ======================================================

.. note::
   Neither comb state is reachable from ``run(method=...)`` today -- the comb and
   site-tree models both propagate through
   :class:`fishbonett.states.tree.TreeTEBD` instead (see
   :mod:`fishbonett.evolve.tebd_tree`).  ``FishBoneNet`` is the specialized comb
   engine recorded as a gap in :mod:`fishbonett.models.registry`.
"""
import numpy as np

from fishbonett.contract import contract as einsum

try:  # optional GPU backend
    import cupy as cp
    _CUPY = True
    _mempool = cp.get_default_memory_pool()
except ImportError:  # pragma: no cover - exercised only with a GPU present
    _CUPY = False

__all__ = ["update_bond", "update_bond_1d"]


def update_bond(state, n, i, chi_max, eps):
    """Apply ``state.U[n][i]`` at bond ``i`` of chain ``n`` and re-split.

    The comb's sites have different leg counts depending on where they sit, so the
    contraction pattern is chosen per case rather than being uniform as it is on a
    plain chain:

    * ``n == -1`` -- an *inter-chain* electronic bond, joining chain ``i``'s
      electronic site to chain ``i+1``'s (both carry their own bath legs);
    * ``i == ebL[n] - 1`` -- the electron-bath-to-electron bond;
    * ``i == ebL[n]`` -- the electron-to-vibration bond;
    * otherwise -- a plain two-leg bond inside a bath chain.

    ``state`` supplies ``get_theta2`` and ``split_truncate_theta``; this function
    only contracts the gate in.
    """
    theta = state.get_theta2(n, i)
    max_index_n = state._L[n] - 2
    max_index_main = state._nc - 2
    e_index = state._ebL[n] - 1
    v_index = state._ebL[n]
    if n == -1 and 0 <= i <= max_index_main:
        # {Down part: vL i VD vR; Up part: VL' j vU' vR'}
        # {i j [i*] [j*]} * {vL [i] vD vR, vL' [j] vD vR}
        # {i j   k   l}     {a   b  c  d ,  e   f  g  h}
        utheta = einsum('IJKL, aKcdeLgh->aIcdeJgh', state.U[i][-1], theta)
    elif 0 <= n < state._nc and 0 <= i <= max_index_n:
        if i == e_index:
            # {i j [i*] [j*]} * {vL [i]  [j] vU vD vR}
            # {i j  k   l}      {a   b   e   f  g  h}
            utheta = einsum('IJKL, aKLfgh->aIJfgh', state.U[n][i], theta)
        elif i == v_index:
            # {i j [i*] [j*]} * {vL [i] vU vD,  [j]  vR}
            # {i j  k   l}      {a   b  c  d ,   e   h}
            utheta = einsum('IJKL, aKcdLh->aIcdJh', state.U[n][i], theta)
        else:
            # {i j [i*] [j*]} * {vL [i], [j] vR}
            # {I J  K   L}      {a   b,   e   h}
            utheta = einsum('IJKL,aKLh->aIJh', state.U[n][i], theta)
    else:
        raise ValueError(
            f"bond (n={n}, i={i}) is out of range for a comb with "
            f"{state._nc} chain(s); chain n has bonds 0..{max_index_n} and the "
            f"inter-chain bonds are n=-1, i in 0..{max_index_main}")
    state.split_truncate_theta(utheta, n, i, chi_max, eps)


def update_bond_1d(state, i, chi_max, eps, gpu=False):
    """Apply ``state.U[i]`` at bond ``i`` of a plain chain and re-split.

    One TEBD step on a single bond: contract the two-site wavefunction with the
    gate, then hand it back to ``state.split_truncate_theta``.  ``U[i]`` is held as
    a sparse matrix and reshaped to ``(d1, d2, d1*, d2*)`` on use.
    """
    d1, d2 = state.pd[i], state.pd[i + 1]
    use_gpu = bool(gpu and _CUPY)
    if not use_gpu:
        theta = state.get_theta2(i)
        u_bond = state.U[i].toarray().reshape([d1, d2, d1, d2])
        # i j [i*] [j*], vL [i] [j] vR
        utheta = np.tensordot(u_bond, theta, axes=([2, 3], [1, 2]))
        utheta = np.transpose(utheta, [2, 0, 1, 3])          # vL i j vR
        state.split_truncate_theta(utheta, i, chi_max, eps)
        return
    theta = state.get_theta2(i)
    u_bond = state.U[i].reshape([d1, d2, d1, d2])
    utheta = cp.tensordot(u_bond, theta, axes=([2, 3], [1, 2]))
    del theta, u_bond
    _mempool.free_all_blocks()
    utheta = np.transpose(utheta, [2, 0, 1, 3])              # vL i j vR
    state.split_truncate_theta(utheta, i, chi_max, eps, gpu=True)
    _mempool.free_all_blocks()
