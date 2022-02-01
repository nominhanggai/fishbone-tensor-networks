"""
    Tridiagonalization function by the Lanczos iterations.
    The following code is excerpted from
    https://github.com/matenure/FastGCN/blob/master/lanczos.py
    The algorithm is closely following the algorithm 10.3 and 10.4 in
    http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter10.pdf
"""
import numpy as np
from math import fsum


def lanczos(A, p):
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
    A = np.array(A)
    q = np.array(p)
    n = A.shape[0]
    b = list(q.shape)
    b.remove(n)
    b = b[0]
    assert n%b == 0 and b<=n and b >= 1
    q_shape = q.shape
    if q_shape[0] < q_shape[1]:
        q = q.T

    for i, vec in enumerate(q.T):
        q[:, i] = vec / np.linalg.norm(vec)

    from itertools import combinations

    for pair in combinations(range(b), 2):
        i, j = pair
        print(i,j, q[:,i]@q[:,j])
        assert abs(q[:,i]@q[:,j]) <= ortho_threshold


    Q = np.zeros((n, n + 2*b))
    Q[:, b:2*b] = q
    beta = np.zeros((b,b))

    for i in range(1, n//b+1):
        Y = A @ Q[:, i*b:(i+1)*b]
        alpha = Q[:, i*b:(i+1)*b].T @ Y
        R = Y - Q[:, i*b:(i+1)*b] @ alpha - Q[:, (i-1)*b:i*b] @ beta.T

        q, beta = np.linalg.qr(R)
        print("QR", q[:,0], R[:,0])
        # Full Orthogonlaization
        q = q - np.dot(Q[:, b:(i + 1) * b], np.dot(Q[:, b:(i + 1) * b].T, q))
        Q[:, (i + 1) * b:(i + 2) * b] = q

    Q = Q[:, b:n+b]

    Sigma = np.dot(Q.T, np.dot(A, Q))
    # print(Q.T@Q)
    return Sigma, Q


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    dim = 6
    mat = np.random.rand(dim,dim)
    mat = np.diag(range(3,9))
    ele = np.diagonal(mat)
    r =range(1,7)
    v0_2 = np.array([[0.1048284802655984 , 0.20965697053119683, 0.31448545079679524,
       0.4193139310623937 , 0.524142421327992  , 0.6289709015935905],
       [-0.1805199289514124 ,  0.18352487171259757, -0.1414561872388148 ,
       -0.6848672161902272 , -0.21551190552621724,  0.6758111855223703]])
    v0 = np.array([[0.10482848,  0.20965697,  0.31448545,  0.41931393,  0.52414242,
         0.6289709]])

    T, Q = block_lanczos(mat, v0_2)
    print(T)
    T, Q = lanczos(mat, v0)
    print(T)
    #
    # print(np.sort(np.linalg.eigvals(T)) - ele
    #      )
    # print(Q.T @ Q - np.eye(Q.shape[0]))
