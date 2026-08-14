"""Compiling a time-dependent frame's two-site Hamiltonians into Trotter gates.

The static frames go through :meth:`fishbonett.frames.terms.LocalTerms.gates`, which
exponentiates one operator per node and per edge.  The *time-dependent* frames cannot
use that container -- their coupling is a function of ``t`` and there are no on-site
bath terms at all -- so they emit a plain list of two-site Hamiltonians instead and
compile it here.

One module for both because the compilation is the same for every such frame: what
differs between :class:`~fishbonett.frames.interaction_picture.SystemBathIP` and
:class:`~fishbonett.frames.multichannel.SystemBathMultiChannel` is how ``h`` is
*built* (a scalar ``d_n(t)`` times one coupling operator, versus a matrix-valued
coupling summed over channels), not how it is exponentiated.
"""
import numpy as np
from copy import deepcopy as dcopy

from fishbonett.linalg import expm_gate

__all__ = ["swap_gate_pairs", "star_edges"]


def star_edges(n_modes):
    """The **interaction** graph these frames emit: a star centred on the system.

    ``get_h2`` returns one two-site term per mode, and every one of them pairs that
    mode with the *system* -- there are no mode-mode terms in the interaction
    picture, because the free-bath evolution has been rotated into ``d_n(t)``.  So
    the graph of H is ``0-1, 0-2, ..., 0-N``.

    The state, meanwhile, is a **path** ``0-1-2-...-N``.  Those are different graphs
    for ``N > 2``, and reconciling them is exactly what the swap network does and
    what :data:`fishbonett.models.registry.LAYOUTS` calls ``"swap"``.  Returned as
    data so that relationship can be asserted rather than only described.
    """
    return [(0, k) for k in range(1, n_modes + 1)]


def swap_gate_pairs(h2, factor=1):
    """Two-site Trotter gates ``(U1, U2)`` for the swap-network sweep.

    Parameters
    ----------
    h2 : list of (h, d1, d2)
        Two-site Hamiltonians in chain order, as
        :meth:`~fishbonett.frames.interaction_picture.SystemBathIP.get_h2` returns
        them: ``h`` is a sparse ``(d1*d2, d1*d2)`` matrix in ``(d1 x d2)`` =
        **(boson, system)** basis order, with the interval already folded into it.
    factor : int, optional
        Divides each Hamiltonian before exponentiating, for sub-stepping.

    Returns
    -------
    (U1, U2)
        Both lists of ``(d_sys, d_boson, d_sys*, d_boson*)`` gates.  ``U1`` has the
        **system leg first** -- the site-0 convention this package uses throughout,
        which is why ``h`` is transposed rather than merely reshaped.  ``U2`` is
        ``U1`` with the two physical legs exchanged, which is what the *swapped*
        sweeps of :func:`fishbonett.evolve.tebd.symmetric_swap_step` consume: there
        the gate acts on a pair that the swap network has already transposed.

    Because these frames are time-dependent, the gates are valid only for the
    interval ``h2`` was built for and must be rebuilt each step.
    """
    U1 = dcopy(h2)
    U2 = dcopy(U1)
    for i, (h, d1, d2) in enumerate(h2):
        u = expm_gate(h.toarray() / factor, 1)
        u1 = u.reshape([d1, d2, d1, d2]).transpose([1, 0, 3, 2])
        U1[i] = u1
        U2[i] = np.transpose(u1, [1, 0, 3, 2])
    return U1, U2
