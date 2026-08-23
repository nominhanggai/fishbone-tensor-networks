"""Lanczos tridiagonalization -- the star-to-chain half of the TEDOPA mapping.

Given the diagonal star Hamiltonian ``diag(freq)`` and the star couplings ``v``,
the Lanczos iteration seeded with ``v`` produces an orthogonal basis in which the
Hamiltonian is **tridiagonal**: on-site energies on the diagonal, mode-mode
hoppings on the off-diagonal.  That tridiagonal form *is* the chain, and it is
what gives a matrix-product state something local to exploit.

Both routines use full reorthogonalization, which matters here: the Krylov
vectors lose orthogonality quickly for the strongly graded weights typical of a
discretized spectral density, and the chain coefficients are sensitive to it.

.. rubric:: API

======================  =========================================================
:func:`lanczos`         single-vector iteration -- one coupling channel
:func:`block_lanczos`   block iteration -- several channels sharing one bath
======================  =========================================================

The algorithm follows algorithms 10.3 and 10.4 of
http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf; the single-vector
implementation started from https://github.com/matenure/FastGCN/blob/master/lanczos.py.
"""
import numpy as np


def lanczos(A, p, breakdown_tol=None):
    """Tridiagonalize ``A`` in the Krylov basis seeded by ``p``.

    Parameters
    ----------
    A : (n, n) array
        The star Hamiltonian, normally ``np.diag(star_freq)``.
    p : (n,) array
        The seed vector -- the star couplings.  The chain is built outward from
        it, so chain site 0 is the mode the system actually couples to.

    Returns
    -------
    Sigma : (n, n) array
        ``Q^T A Q``, tridiagonal: ``diagonal`` is the chain on-site energies and
        the first sub/super-diagonal the mode-mode hoppings.
    Q : (n, n) array
        The orthogonal star -> chain transform.
    """
    A = np.asarray(A)
    q = np.asarray(p).reshape(-1).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]
    if q.shape != (n,):
        raise ValueError(f"p must have shape {(n,)}, got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("the Lanczos seed must be finite and nonzero")
    dtype = np.result_type(A.dtype, q.dtype, float)
    Q = np.zeros((n, n), dtype=dtype)
    Q[:, 0] = q / norm
    beta = 0.0
    if breakdown_tol is None:
        scale = max(float(np.linalg.norm(A, ord=2)), 1.0)
        breakdown_tol = 100 * np.finfo(float).eps * n * scale

    for i in range(n):
        if i == 0:
            q = np.dot(A, Q[:, i])
        else:
            q = np.dot(A, Q[:, i]) - beta * Q[:, i - 1]
        alpha = np.vdot(Q[:, i], q)
        q = q - Q[:, i] * alpha
        # A second projection against the complete basis accumulated so far is
        # inexpensive for bath grids and prevents loss of orthogonality for
        # strongly graded spectral weights.
        basis = Q[:, :i + 1]
        q = q - basis @ (basis.conj().T @ q)
        beta = np.linalg.norm(q)
        if i + 1 == n:
            break
        if not np.isfinite(beta) or beta <= breakdown_tol:
            raise ValueError(
                "Lanczos iteration terminated before spanning the bath modes; "
                "remove zero-coupling modes and combine degenerate frequencies")
        Q[:, i + 1] = q / beta

    Sigma = Q.conj().T @ A @ Q
    Sigma = np.real_if_close(Sigma)
    Q = np.real_if_close(Q)
    return Sigma, Q


def block_lanczos(A, p, ortho_threshold=1e-14):
    """Block Lanczos: tridiagonalize ``A`` in blocks seeded by the columns of ``p``.

    The multichannel counterpart of :func:`lanczos`.  Where a single channel gives
    a scalar chain, ``b`` channels sharing one bath give a **block**-tridiagonal
    chain -- each chain site carries a ``b x b`` on-site block and a ``b x b``
    hopping to its neighbour, which is what keeps the channels cross-correlated
    rather than independent.

    ``p`` is ``(n, b)`` (or ``(b, n)``; it is transposed if needed) and its columns
    must be mutually orthogonal to within ``ortho_threshold``; invalid seeds raise
    :class:`ValueError`, because a non-orthogonal seed produces the wrong chain. Returns
    ``(Sigma, Q)`` as in :func:`lanczos`.
    """
    dtype = np.result_type(np.asarray(A).dtype, np.asarray(p).dtype, float)
    A = np.asarray(A, dtype=dtype)
    q = np.array(p, dtype=dtype, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.all(np.isfinite(A)) or not np.allclose(A, A.conj().T):
        raise ValueError("A must be finite and Hermitian")
    n = A.shape[0]
    if q.ndim != 2 or n not in q.shape:
        raise ValueError("p must be a matrix with one dimension equal to A.shape[0]")
    if q.shape[0] != n:
        q = q.T
    b = q.shape[1]
    if not (n % b == 0 and n >= b >= 1):
        raise ValueError("the block width must be positive and divide A.shape[0]")

    for i, vec in enumerate(q.T):
        norm = np.linalg.norm(vec)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("every block Lanczos seed must be finite and nonzero")
        q[:, i] = vec / norm

    from itertools import combinations

    for pair in combinations(range(b), 2):
        i, j = pair
        overlap = abs(np.vdot(q[:, i], q[:, j]))
        if overlap > ortho_threshold:
            raise ValueError(
                "block Lanczos seed columns must be mutually orthogonal; "
                f"columns {i} and {j} overlap by {overlap:g}")

    # Orthonormalize the complete seed block first.  The columns were checked
    # for pairwise orthogonality above, but QR also makes this robust to their
    # different norms and fixes the complex-valued case.
    first, triangular = np.linalg.qr(q, mode="reduced")
    if np.min(np.abs(np.diag(triangular))) <= ortho_threshold:
        raise ValueError("block Lanczos seed is rank deficient")
    blocks = [first]
    for _ in range(1, n // b):
        candidate = A @ blocks[-1]
        # Two full reorthogonalization passes give a stable block Krylov basis.
        # For Hermitian A the projected matrix is block tridiagonal up to roundoff.
        for _pass in range(2):
            for basis in blocks:
                candidate -= basis @ (basis.conj().T @ candidate)
        next_block, triangular = np.linalg.qr(candidate, mode="reduced")
        if np.min(np.abs(np.diag(triangular))) <= ortho_threshold:
            raise ValueError(
                "block Lanczos iteration terminated before spanning the space"
            )
        blocks.append(next_block)

    Q = np.concatenate(blocks, axis=1)
    Sigma = Q.conj().T @ A @ Q
    Sigma = 0.5 * (Sigma + Sigma.conj().T)
    return Sigma, Q


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    dim = 6
    mat = np.random.rand(dim, dim)
    mat = np.diag(range(3, 9))
    ele = np.diagonal(mat)
    r = range(1, 7)
    v0_2 = np.array([[0.1048284802655984, 0.20965697053119683, 0.31448545079679524,
                      0.4193139310623937, 0.524142421327992, 0.6289709015935905],
                     [-0.1805199289514124, 0.18352487171259757, -0.1414561872388148,
                      -0.6848672161902272, -0.21551190552621724, 0.6758111855223703]])
    v0 = np.array([[0.10482848, 0.20965697, 0.31448545, 0.41931393, 0.52414242,
                    0.6289709]])

    T, Q = block_lanczos(mat, v0_2)
    print(T)
    T, Q = lanczos(mat, v0)
    print(T)
    #
    # print(np.sort(np.linalg.eigvals(T)) - ele
    #      )
    # print(Q.T @ Q - np.eye(Q.shape[0]))
