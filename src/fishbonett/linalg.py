"""Tensor-network linear algebra: SVD, gate exponentials, and truncation policy.

Certified adaptive SVD (``threshold_svd``), full-spectrum SVD (``full_svd``),
two-site gate exponentials, identity/Kronecker constructors tolerating ``None``
legs, and :class:`Truncation` (``eps`` + ``max_bond``).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
from scipy.sparse import coo_array, csc_matrix, kron as skron
from scipy.sparse.linalg import expm as sparse_expm

from fishbonett._svd import robust_svd
from fishbonett.randomized import adaptive_svd

#: Default relative singular-value threshold.  ``1e-4`` is the accuracy most
#: calculations actually need; tightening it far below the target accuracy is the
#: most common way to waste time (the cost climbs steeply).
DEFAULT_EPS = 1e-4


def eye(d):
    """Identity operator.

    ``None`` / empty -> ``None``; an ``int`` (or ``str``) ``d`` -> ``np.eye(int(d))``;
    a two-element list ``[m, n]`` -> ``np.eye(m, n)`` (a possibly rectangular
    identity).
    """
    if d is None or (isinstance(d, (list, tuple)) and len(d) == 0):
        return None
    if isinstance(d, (int, np.integer, str)):
        return np.eye(int(d))
    if isinstance(d, (list, tuple, np.ndarray)):
        values = np.asarray(d).reshape(-1)
        if values.size not in (1, 2):
            raise ValueError("identity dimensions must contain one or two values")
        return np.eye(*map(int, values))
    raise TypeError(f"unsupported identity dimension {d!r}")


def kron(a, b):
    """Sparse (CSC) Kronecker product that tolerates absent legs.

    Returns ``None`` if either operand is ``None`` (an absent leg, used while
    assembling tensor-network operators).  A list operand is splatted, so
    ``kron([x, y], z)`` means ``skron(x, y, z)``.

    Returns a sparse ``csc_array``. The package uses the result for scaling,
    addition and conversion through ``.toarray()``.
    """
    if a is None or b is None:
        return None
    args = tuple(a) if type(a) is list else (a,)
    args += tuple(b) if type(b) is list else (b,)
    first, *rest = args
    return skron(coo_array(first), *rest, format='csc')


def threshold_svd(A, eps, max_rank=None, *, extra_rank=0, backend=None,
                  initial_rank=16, return_info=False):
    """Return the Schmidt directions admitted by a relative threshold.

    Large low-rank matrices use adaptively enlarged randomized ranges.  The
    omitted Frobenius norm certifies their stopping point; unresolved or nearly
    full-rank cases fall back to LAPACK.  ``backend`` may be ``"auto"``,
    ``"exact"`` or ``"randomized"``.  The accuracy semantics are identical in
    every case: retain singular values above ``eps * s[0]``, optionally keep
    ``extra_rank`` additional directions, and finally apply ``max_rank``.
    """
    return adaptive_svd(
        A, eps=eps, max_rank=max_rank, extra_rank=extra_rank,
        backend=backend, initial_rank=initial_rank, return_info=return_info,
    )


def full_svd(A, full_matrices=False):
    """Return the complete exact spectrum for operations that require it.

    Entropy evaluation and exact state/operator factorization cannot infer their
    output from a partial spectrum, so they deliberately select the exact branch
    of the common package policy.
    """
    return robust_svd(A, full_matrices=full_matrices)


def svd(A, b=None, full_matrices=False):
    """Compatibility facade for a full or fixed-rank decomposition.

    New truncation code should call :func:`threshold_svd` so the requested error
    is explicit.  ``b=None`` returns the complete exact spectrum; a positive
    ``b`` requests at most that many leading directions through the adaptive
    backend.
    """
    if full_matrices:
        if b is not None:
            raise ValueError("a truncated SVD cannot request full_matrices=True")
        return full_svd(A, full_matrices=True)
    if b is None or b < 0:
        return full_svd(A, full_matrices=False)
    return threshold_svd(A, 0.0, max_rank=int(b))


def cap_rank(count, chi_max=None):
    """Clamp a kept-singular-value count to ``chi_max`` (at least 1).

    ``chi_max=None`` means *unlimited*: the bond dimension is then set purely by
    the singular-value threshold.  This is the primitive behind
    :meth:`Truncation.cap`; engines that have already counted their singular
    values call it directly.
    """
    count = max(1, int(count))
    return count if chi_max is None else min(int(chi_max), count)


@dataclass(frozen=True)
class Truncation:
    """How much of a state to discard at each bond -- accuracy *and* memory in one
    object.

    The two controls are deliberately not interchangeable:

    ``eps``
        the **accuracy** knob (default :data:`DEFAULT_EPS` = ``1e-4``).  After
        each SVD, singular values below ``eps`` *relative to the largest on that
        bond* are discarded.  On its own this already determines the bond
        dimension -- the state grows exactly as much as the physics demands.
    ``max_bond``
        an optional **hard cap**, ``None`` meaning *unlimited*.  Use it when you
        need a guaranteed memory bound and will accept a larger error to get it.

    The intended workflow is to set ``eps`` to the accuracy you need, leave
    ``max_bond`` unset, and watch ``result.max_bond``; introduce a cap only if
    the bond grows beyond what you can afford.

    >>> import numpy as np
    >>> t = Truncation(eps=1e-4)
    >>> t.keep(np.array([1.0, 1e-2, 1e-6]))     # third value is below eps
    2
    >>> Truncation(eps=1e-4, max_bond=1).keep(np.array([1.0, 1e-2, 1e-6]))
    1
    """

    eps: float = DEFAULT_EPS
    max_bond: Optional[int] = None

    def __post_init__(self):
        """Validate and normalize the threshold and optional rank cap."""
        if (isinstance(self.eps, (bool, np.bool_))
                or not isinstance(self.eps, (int, float, np.number))
                or not np.isfinite(self.eps) or self.eps < 0):
            raise ValueError(
                f"eps must be a finite non-negative number, got {self.eps!r}"
            )
        object.__setattr__(self, "eps", float(self.eps))
        if self.max_bond is not None:
            if (isinstance(self.max_bond, (bool, np.bool_))
                    or not isinstance(self.max_bond, (int, np.integer))
                    or self.max_bond < 1):
                raise ValueError(
                    "max_bond must be a positive integer or None (unlimited), "
                    f"got {self.max_bond!r}"
                )
            object.__setattr__(self, "max_bond", int(self.max_bond))

    @classmethod
    def resolve(cls, trunc=None, *, eps=None, max_bond=None):
        """Build a :class:`Truncation` from whatever the caller supplied.

        Accepts an existing :class:`Truncation` (returned unchanged), a bare
        ``float`` (read as ``eps``), or ``None`` plus the loose ``eps`` /
        ``max_bond`` keywords.  This is what lets every entry point take either
        ``trunc=Truncation(...)`` or the separate arguments.
        """
        if isinstance(trunc, cls):
            if eps is not None or max_bond is not None:
                raise TypeError("pass either a Truncation or eps/max_bond, not both")
            return trunc
        if isinstance(trunc, (int, float)) and not isinstance(trunc, bool):
            if eps is not None:
                raise TypeError("pass either a Truncation or eps/max_bond, not both")
            eps = float(trunc)
        elif trunc is not None:
            raise TypeError(f"expected a Truncation, a float or None, got {trunc!r}")
        return cls(eps=DEFAULT_EPS if eps is None else float(eps),
                   max_bond=max_bond)

    def keep(self, s):
        """Number of singular values of ``s`` to keep (at least 1).

        ``s`` is a descending singular-value spectrum.  Values below
        ``eps * s[0]`` are dropped, then the count is capped at ``max_bond``.
        """
        s = np.asarray(s)
        if s.size == 0:
            return 1
        smax = s.flat[0] if s.flat[0] > 0 else s.max(initial=0.0)
        count = int(np.sum(s > self.eps * smax)) if smax > 0 else 1
        return self.cap(count)

    def cap(self, count):
        """Clamp an already-chosen kept-value ``count`` to ``max_bond`` (>= 1)."""
        return cap_rank(count, self.max_bond)

    @property
    def svd_rank(self):
        """The fixed-rank compatibility value corresponding to ``max_bond``."""
        return self.max_bond


def expm_gate(H, dt):
    """Dense two-site propagator ``expm(-i * dt * H)``.

    No imaginary unit is folded into ``dt`` -- a real ``dt`` is real-time
    evolution.  Each local operator carries legs ``(i, i+1, i*, (i+1)*)``.
    """
    return scipy.linalg.expm(-dt * 1j * H)


def expm_gate_sparse(H, dt):
    """Sparse variant of :func:`expm_gate` for large two-site bonds.

    Builds a CSC matrix and uses ``scipy.sparse.linalg.expm``; otherwise
    identical to :func:`expm_gate`.
    """
    H_sparse = csc_matrix(H)
    return sparse_expm(-dt * 1j * H_sparse)
