"""Encode interval-integrated two-site Hamiltonians as Trotter gates."""
import numpy as np
from copy import deepcopy as dcopy

from fishbonett.linalg import expm_gate

__all__ = ["swap_gate_pairs", "star_edges", "SwapGateEncoder"]


class SwapGateEncoder:
    """Adapt a representation with ``two_site_hamiltonians`` to swap gates."""

    def __init__(self, representation):
        self.representation = representation

    def get_u(self, t, dt, factor=1, inc_sys=True):
        """Two-site Trotter gates over ``[t, t+dt]`` as ``(U1, U2)``.

        Exponentiates the representation's interval Hamiltonians via
        :func:`swap_gate_pairs`.  :func:`fishbonett.evolve.tebd.symmetric_swap_step`
        calls this twice per step -- once per half-interval -- to stay second order.

        Because these representations are time-dependent, the gates are valid only for the
        interval they were built for and must be rebuilt each step.
        """
        hamiltonians = self.representation.two_site_hamiltonians(
            t, dt, include_system=inc_sys)
        return swap_gate_pairs(hamiltonians, factor)


def star_edges(n_modes):
    """A mode-decoupled interaction graph: a star centred on the system.

    ``two_site_hamiltonians`` returns one two-site term per mode, and every one of them pairs that
    mode with the *system* -- there are no mode-mode terms in the interaction
    picture, because the free-bath evolution has been rotated into ``d_n(t)``.  So
    the graph of H is ``0-1, 0-2, ..., 0-N``.

    The state, meanwhile, is a **path** ``0-1-2-...-N``.  Those are different graphs
    for ``N > 2``, and reconciling them is exactly what the swap network does and
    what :data:`fishbonett.models.registry.APPLICATIONS` calls ``"swap"``.  Returned
    as data so that relationship can be asserted rather than only described.
    """
    return [(0, k) for k in range(1, n_modes + 1)]


def swap_gate_pairs(h2, factor=1):
    """Two-site Trotter gates ``(U1, U2)`` for the swap-network sweep.

    Parameters
    ----------
    h2 : list of (h, d1, d2)
        Two-site Hamiltonians in chain order, as
        :meth:`~fishbonett.representations.interaction.InteractionRepresentation.two_site_hamiltonians` returns
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

    Because these representations are time-dependent, the gates are valid only for the
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
