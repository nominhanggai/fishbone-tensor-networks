"""TEBD on an MPS: bond update, sweeps, and whole symmetric steps.

TEBD is Trotter-split gates plus truncation, which works on any loop-free
geometry -- so this is one of several TEBD implementations here, not *the* one.
It is the MPS one: it acts on a :class:`~fishbonett.states.mps.SystemBathMPS`
and indexes gates by bond number, which is what ties it to a linear chain.  The
others carry their own bookkeeping for their own geometry:

- :func:`fishbonett.evolve.modetree.run_tree_tebd` -- a balanced binary tree of
  *modes* (``interaction-chain-tree-tebd``).
- :mod:`fishbonett.evolve.sitetree` -- any loop-free tree of *sites*, which the
  comb and site-tree models use with ``schrodinger-chain-tree-tebd``; the
  multichannel model uses the same engine with ``schrodinger-star-tree-tebd``.

Two sweep patterns here, selected by the representation:

- *swap network* (:func:`symmetric_swap_step`) — interaction picture, where every
  mode couples to the system.  ``swap=1`` walks the system along the chain.
- *static* (:func:`symmetric_static_step`) — polaron representation.  Gates built once,
  nearest-neighbour, no swapping.
"""
from fishbonett.contract import contract as einsum

try:  # optional GPU backend (only used when a state requests gpu=True)
    import cupy as cp
    _CUPY = True
except ImportError:  # pragma: no cover - exercised only with a GPU present
    _CUPY = False

__all__ = ["update_bond", "sweep", "swap_in", "swap_out",
           "symmetric_swap_step", "symmetric_static_step"]


# -- primitive ---------------------------------------------------------------
def update_bond(state, i, chi_max, eps, swap=0, eps_lbo=None, adaptive=False,
                gpu=False):
    """Apply the two-site gate ``state.U[i]`` at bond ``i`` and re-split ``state``.

    The primitive every sweep is built from: contract the two site tensors with
    the gate, then SVD back apart, keeping singular values above ``eps``
    (relative) and at most ``chi_max`` of them.

    Parameters
    ----------
    state : fishbonett.states.mps.SystemBathMPS
        The MPS whose bond ``i`` is updated **in place**.
    i : int
        Bond index; the gate acts on sites ``i`` and ``i+1``.
    chi_max : int or None
        Hard bond-dimension cap; ``None`` means unlimited.
    eps : float
        Relative singular-value threshold.
    swap : {0, 1}
        1 transposes the two physical legs during the gate, so sites ``i`` and
        ``i+1`` come back **exchanged** -- this is the swap network of the
        interaction picture, and applied along ascending bonds it walks the system
        site rightward along the chain (see :func:`swap_out`).  0 leaves the sites
        where they are.  Note the gate's legs must be ordered to match the sites as
        they are *now*, which is why the builders hand out both ``U1`` and its
        leg-transposed twin ``U2``.
    eps_lbo : float, optional
        Local-basis-optimization threshold; enables LBO and the adaptive search.
    adaptive : bool
        Adaptive bond-dimension search without LBO.  Ignored when ``eps_lbo`` is
        given.  Default is a single truncated SVD at ``chi_max``.
    gpu : bool
        Use the CuPy backend if it is installed.
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


# -- sweeps ------------------------------------------------------------------
def sweep(state, bonds, chi_max, eps, swap=0, **kw):
    """Apply the state's current gates along ``bonds``, in the order given.

    ``bonds`` is any iterable of bond indices, so the caller picks the direction;
    remaining keywords go straight to :func:`update_bond`.
    """
    for j in bonds:
        update_bond(state, j, chi_max, eps, swap=swap, **kw)


def swap_out(state, n, chi_max, eps, **kw):
    """Swap-network sweep *outward*: bonds ``0, ..., n-2`` with ``swap=1``.

    ``n`` is the number of bath modes, so the chain has ``n + 1`` sites (the system
    at site 0, modes at ``1..n``) and ``n`` bonds.  Each swapped gate exchanges its
    two sites, so sweeping up the chain carries the system site from site 0 to site
    ``n-1``, meeting a different mode at every bond on the way.

    Note this stops one bond short of the end: the modes met here are the ``n-1`` at
    sites ``1..n-1``, and the outermost mode is reached by the separate ``swap=0``
    application at bond ``n-1`` (see :func:`symmetric_swap_step`).  It is the whole
    step, not this sweep, that pairs the system with every mode.
    """
    sweep(state, range(n - 1), chi_max, eps, swap=1, **kw)


def swap_in(state, n, chi_max, eps, **kw):
    """Swap-network sweep *inward*: bonds ``n-2, ..., 0`` with ``swap=1``.

    The exact reverse of :func:`swap_out`, walking the system site back from
    ``n-1`` to site 0 so the state ends a step in the same layout it started in.
    Because the two sites at each bond are now in the opposite order, this consumes
    the leg-transposed gates (``U2``) rather than the ``U1`` of :func:`swap_out`.
    """
    sweep(state, range(n - 2, -1, -1), chi_max, eps, swap=1, **kw)


# -- whole symmetric steps ---------------------------------------------------
def symmetric_static_step(state, gates, n, chi_max, eps, **kw):
    """One 2nd-order (Strang) step with **time-independent** gates.

    ``gates`` are the half-step (``dt/2``) two-site gates, built once.  Sweeping
    up the chain and straight back down applies them in palindromic order, which
    is what makes the step second order in ``dt``.  This is the polaron representation's
    step: no swapping, no per-step rebuild.
    """
    state.U = gates
    sweep(state, range(n), chi_max, eps, swap=0, **kw)
    sweep(state, range(n - 1, -1, -1), chi_max, eps, swap=0, **kw)


def symmetric_swap_step(state, representation, t0, dt, n, chi_max, eps, **kw):
    """One 2nd-order (Strang) swap-network step over ``[t0, t0+dt]``.

    The interaction step. The representation supplies time-dependent gates
    through ``tebd_gates(t, half_dt)``, so they are rebuilt twice per step --
    once per half-interval.

    The ordering is palindromic: the first half-interval's gates sweep inward,
    the second half-interval's sweep back out, and the two innermost (bond-0)
    applications straddle the midpoint, one from each half.  Reusing the *same*
    half-step gates for both would break time symmetry and drop the step to first
    order in ``dt``.  ``tebd_gates`` returns ``(U1, U2)`` where ``U2`` is the
    leg-transposed variant used by the swapped sweeps, so the two un-swapped
    bond-0 updates must both take a ``U1``.
    """
    hdt = dt / 2.0
    u_in, _ = representation.tebd_gates(t0, hdt)
    u_mid, u_out = representation.tebd_gates(t0 + hdt, hdt)

    state.U = u_in
    swap_out(state, n, chi_max, eps, **kw)
    update_bond(state, n - 1, chi_max, eps, swap=0, **kw)
    state.U = u_mid
    update_bond(state, n - 1, chi_max, eps, swap=0, **kw)
    state.U = u_out
    swap_in(state, n, chi_max, eps, **kw)
