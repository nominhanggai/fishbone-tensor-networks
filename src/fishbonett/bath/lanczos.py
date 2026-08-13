"""Lanczos tridiagonalization -- the star-to-chain half of the TEDOPA mapping.

Given the diagonal star Hamiltonian ``diag(freq)`` and the star couplings ``v``,
the Lanczos iteration seeded with ``v`` produces an orthogonal basis in which the
Hamiltonian is **tridiagonal**: on-site energies on the diagonal, mode-mode
hoppings on the off-diagonal.  That tridiagonal form *is* the chain, and it is
what gives a matrix-product state something local to exploit.

Both routines use full reorthogonalization, which matters here: the Krylov
vectors lose orthogonality quickly for the strongly graded weights typical of a
discretized spectral density, and the chain coefficients are sensitive to it.

.. rubric:: What's here

======================  =========================================================
:func:`lanczos`         single-vector iteration -- one coupling channel
:func:`block_lanczos`   block iteration -- several channels sharing one bath
======================  =========================================================

The algorithm follows algorithms 10.3 and 10.4 of
http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf; the single-vector
implementation started from https://github.com/matenure/FastGCN/blob/master/lanczos.py.
"""
import numpy as np


def lanczos(A, p):
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
    A = np.array(A)
    q = np.array(p).copy()
    n = A.shape[0]
    Q = np.zeros((n, n + 1))
    Q[:, 0] = q / np.linalg.norm(q)
    # print(Q[:,0])
    alpha = 0
    beta = 0

    for i in range(n):
        if i == 0:
            q = np.dot(A, Q[:, i])
        else:
            q = np.dot(A, Q[:, i]) - beta * Q[:, i - 1]
        alpha = np.dot(q.T, Q[:, i])
        q = q - Q[:, i] * alpha
        q = q - np.dot(Q[:, :i], np.dot(Q[:, :i].T, q))  # full reorthogonalization
        beta = np.linalg.norm(q)
        Q[:, i + 1] = q / beta

    Q = Q[:, :n]

    Sigma = np.dot(Q.T, np.dot(A, Q))
    return Sigma, Q


def block_lanczos(A, p, ortho_threshold=1e-14):
    """Block Lanczos: tridiagonalize ``A`` in blocks seeded by the columns of ``p``.

    The multichannel counterpart of :func:`lanczos`.  Where a single channel gives
    a scalar chain, ``b`` channels sharing one bath give a **block**-tridiagonal
    chain -- each chain site carries a ``b x b`` on-site block and a ``b x b``
    hopping to its neighbour, which is what keeps the channels cross-correlated
    rather than independent.

    ``p`` is ``(n, b)`` (or ``(b, n)``; it is transposed if needed) and its columns
    must be mutually orthogonal to within ``ortho_threshold`` -- this is asserted,
    because a non-orthogonal seed silently produces the wrong chain.  Returns
    ``(Sigma, Q)`` as in :func:`lanczos`.
    """
    A = np.array(A)
    q = np.array(p)
    n = A.shape[0]
    b = list(q.shape)
    b.remove(n)
    b = b[0]
    assert n % b == 0 and n >= b >= 1
    q_shape = q.shape
    if q_shape[0] < q_shape[1]:
        q = q.T

    for i, vec in enumerate(q.T):
        q[:, i] = vec / np.linalg.norm(vec)

    from itertools import combinations

    for pair in combinations(range(b), 2):
        i, j = pair
        print(i, j, q[:, i] @ q[:, j])
        assert abs(q[:, i] @ q[:, j]) <= ortho_threshold

    Q = np.zeros((n, n + 2 * b))
    Q[:, b:2 * b] = q
    beta = np.zeros((b, b))

    for i in range(1, n // b + 1):
        Y = A @ Q[:, i * b:(i + 1) * b]
        alpha = Q[:, i * b:(i + 1) * b].T @ Y
        R = Y - Q[:, i * b:(i + 1) * b] @ alpha - Q[:, (i - 1) * b:i * b] @ beta.T

        q, beta = np.linalg.qr(R)
        print("QR", q[:, 0], R[:, 0])
        # Full Orthogonlaization
        q = q - np.dot(Q[:, b:(i + 1) * b], np.dot(Q[:, b:(i + 1) * b].T, q))
        Q[:, (i + 1) * b:(i + 2) * b] = q

    Q = Q[:, b:n + b]

    Sigma = np.dot(Q.T, np.dot(A, Q))
    # print(Q.T@Q)
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
