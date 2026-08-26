"""Apply a matrix-product operator to a matrix-product state and re-compress.

The propagation algorithm for ``method="interaction-chain-trotter-mpo"``: the
interaction representation's system-bath propagator is an exact, low-bond MPO
(:meth:`fishbonett.representations.interaction.InteractionRepresentation.trotter_mpo`), so a
step applies the MPO and then compresses the resulting MPS. State containers live
in :mod:`fishbonett.states`; propagation algorithms live in
:mod:`fishbonett.evolve`.

Tensor conventions
------------------
MPS ``A[i]``: ``(bond_l, phys, bond_r)``.
MPO ``W[i]``: ``(bond_l, bond_r, phys_out, phys_in)``.
"""
import numpy as np

from fishbonett.contract import contract as einsum
from fishbonett.linalg import DEFAULT_EPS, full_svd, threshold_svd

__all__ = ["apply_mpo", "compress", "bond_dims", "total_bond_entropy",
           "product_state"]


def product_state(phys_dims, first=None):
    """Bond-1 product state: ``first`` on site 0 and the vacuum on every other site."""
    A = []
    for k, d in enumerate(phys_dims):
        t = np.zeros((1, d, 1), complex)
        if k == 0 and first is not None:
            t[0, :, 0] = np.asarray(first, complex)
        else:
            t[0, 0, 0] = 1.0
        A.append(t)
    return A


def apply_mpo(A, W):
    """``(W A)[i] = sum_in W[i][wl,wr,out,in] A[i][l,in,r]``, fusing the MPO bond
    into the MPS bond.  The bond dimension is multiplied by the MPO bond, so this
    is normally followed by :func:`compress`."""
    out = []
    for a, w in zip(A, W, strict=True):
        dl, _, dr = a.shape
        wl, wr, phys_out, _ = w.shape
        out.append(einsum('xyij,ajb->xaiyb', w, a).reshape(wl * dl, phys_out, wr * dr))
    return out


def compress(A, chi_max=None, eps=DEFAULT_EPS):
    """Truncate ``A`` in place-ish and return it, orthogonality centre at site 0.

    Sweeps left-to-right with QR (making the state left-canonical, so the discarded
    weight is measured correctly), then right-to-left with a truncated SVD keeping
    singular values above ``eps`` *relative to the largest* on each bond, capped at
    ``chi_max``.  ``chi_max=None`` means no cap -- truncation is then controlled by
    ``eps`` alone.
    """
    n = len(A)
    for i in range(n - 1):                                   # left-canonicalize
        dl, d, dr = A[i].shape
        q, r = np.linalg.qr(A[i].reshape(dl * d, dr))
        A[i] = q.reshape(dl, d, -1)
        A[i + 1] = einsum('xy,ybc->xbc', r, A[i + 1])
    for i in range(n - 1, 0, -1):                            # truncate right-to-left
        dl, d, dr = A[i].shape
        u, s, vh = threshold_svd(
            A[i].reshape(dl, d * dr), eps, max_rank=chi_max)
        keep = s.size
        A[i] = vh.reshape(keep, d, dr)
        A[i - 1] = einsum('abc,cx,x->abx', A[i - 1], u, s)
    A[0] = A[0] / np.linalg.norm(A[0])
    return A


def bond_dims(A):
    """Bond dimensions ``[D_0, ..., D_n]`` of an MPS in **(vL, p, vR)** order.

    .. warning::
       :func:`fishbonett.evolve.tdvp.bonddims` expects the TDVP tensor order with
       the physical leg last and therefore is not interchangeable with this
       function.
    """
    return [A[0].shape[0]] + [a.shape[2] for a in A]


def total_bond_entropy(A):
    """Sum of the von Neumann entropies of every bond.  Assumes the orthogonality
    centre sits at site 0 (as left by :func:`compress`)."""
    total, cur = 0.0, A[0]
    for i in range(len(A) - 1):
        dl, d, dr = cur.shape
        _, s, vh = full_svd(cur.reshape(dl * d, dr), full_matrices=False)
        p = s ** 2
        p = p[p > 1e-14]
        if p.size > 1:
            p = p / p.sum()
            total -= float(np.sum(p * np.log(p)))
        cur = einsum('x,xbc->xbc', s, einsum('xy,ybc->xbc', vh, A[i + 1]))
    return total
