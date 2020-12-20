from fishbonett.model import FishBoneH, kron, calc_U, _c
from fishbonett.fishbone import FishBoneNet, init_ttn
import numpy as np
from scipy.linalg import expm, svd


def sigmaz(d=2):
    z = np.zeros([d, d])
    z[0, 0] = -1
    z[1, 1] = 1
    return z


def sigmax(d=2):
   z = np.zeros([d, d])
   z[0, 1] = 1
   z[1, 0] = 1
   return z


a = [9] * 4
b = [2]
c = [5]
pd = np.array([[a, b, c, a], [a, b, c, a], [a, b, c, a]], dtype=object)

eth = FishBoneH(pd)
etn = init_ttn(nc=3, L=4, d1=9, de=2, dv=5)

eth.hv_dy = [_c(*c)+_c(*c).T for i in range(3)]
eth.he_dy = [sigmaz() for i in range(3)]

a = kron(_c(*b), _c(*b).T) + kron(_c(*b).T, _c(*b)) # + kron(_c(*b).T, _c(*b).T) + kron(_c(*b), _c(*b))
eth.h2ee = [a for i in range(2)]
eth.h2ev = [kron(sigmaz(), _c(*c) + _c(*c).T) for i in range(3)]
eth.h1e = [sigmaz() + sigmax() for i in range(3)]
eth.h1v = [_c(*c).T @ _c(*c) for i in range(3)]
eth.build()
etn.U = eth.get_u(dt=0.02)
# print(eth.w_list[0][0])
# print(eth.k_list[0][0])
# print(eth.he_dy)
# print(eth.hv_dy)
# # print(eth.h2ev)
# # print(eth.h2ee)
# print(eth.h1e)
# print(eth.h1v)

for tn in range(2):
    for i in range(0, 1):
        for j in range(2, 6):
            print("ij==", i, j)
            etn.update_bond(i, j, 12, 1e-12)
            print("H.shape", etn.U[i][j].shape)
            print(eth.h1e[0])
            print([x.shape for x in etn.ttnB[0]])
            print([x.shape for x in etn.ttnS[0]])

# c = _c(9)
# hb = 0.0033959 * c.T@c
# hc = 0.50801342 * np.kron(c.T+c, sigmaz())
# he = sigmaz() + sigmax()
# h = np.kron(hb, np.eye(2)) + hc + np.kron(np.eye(9), he)
#
#
# u = expm(-1j * h* 0.02)
# u = u.reshape(9,2,9,2)
# # print("Diff U", u - etn.U[0][3])
# # print("Diff H", h - eth.H[0][3][0])
#
# A = etn.ttnB[0][3]
# A[0,1,0]=0
# A[0,8,0]=1
# B = etn.ttnB[0][4]
# print("A",A)
# print("B",B)
# Sl = etn.ttnS[0][3]
# print("1", A.shape, B.shape)
# AB = np.tensordot(A,B, axes=1)
# print("2", AB.shape)
#
# theta = np.einsum('IJKL, aKLfgh->aIJfgh', u, AB)
# chiL_l, p_l, p_r, chiU_r, chiD_r, chiR_r = theta.shape
# theta = np.reshape(theta, [chiL_l * p_l,
#                            p_r * chiU_r * chiD_r * chiR_r])
# A, S, B = svd(theta, full_matrices=False)
# chivC = min(12, np.sum(S > 1e-12))
# piv = np.argsort(S)[::-1][:chivC]  # keep the largest `chivC` singular values
# A, S, B = A[:, piv], S[piv], B[piv, :]
# print("3", S)
# S = S / np.linalg.norm(S)
# #self.ttnS[n][i + 1] = S
# print("Shape of middle S", np.diag(S).shape)
# A = np.reshape(A, [chiL_l, p_l, chivC])
# # A: vL*i*vR -> vL i vR=chivC
# B = np.reshape(B, [chivC, p_r, chiU_r, chiD_r, chiR_r])
# # B: chivC*j*vU*vD*vR -> vL==chivC j vU vD vR
# print("Shape of left S and A", Sl.shape, A.shape)
# A = np.tensordot(np.diag(Sl ** (-1)), A, axes=[1, 0])
# A = np.tensordot(A, np.diag(S), axes=[2, 0])
# print(A.shape, B.shape)


# u = expm(1j*h).T.conj()@expm(1j*h)
# prod = expm(-1j*h)@expm(1j*h)
# diff = expm(1j*h).T.conj() - expm(-1j*h)
# print(np.linalg.norm(prod)**2)
# for j in range(eetn._nc - 1):
#     print("ij==", j)
#     eetn.update_bond(-1, j, 12, 1e-12)
#     print([x.shape for x in [eetn.ttnB[j][4]]])
#     print([x.shape for x in eetn.ttnS[0]])
