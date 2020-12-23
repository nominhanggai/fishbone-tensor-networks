from model import FishBoneH, kron, _c
from fishbone import init_ttn
import numpy as np
from numpy import exp, tanh

from scipy.integrate import quad as integrate
def coth(x):
    return (exp(2*x)-1)/(exp(2*x)+1)

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


def temp_factor(beta, w):
    return 0.5 * (1. + 1. / tanh(beta * w / 2.))


bath_length = 5
a = [3] * bath_length
b = [2]
c = [4]
pd = np.array([[a, b, c, a], [a, b, c, a]], dtype=object)

eth = FishBoneH(pd)
etn = init_ttn(nc=2, L=bath_length, d1=3, de=2, dv=4)

# electronic couplings
tda = 0.1
e = 1.0

#
# half_my_ome_2 = 1.0
# ome = 1.0
# my = 2 * half_my_ome_2/(ome**2)
#
# spectral density parameters
eta = 1.
ome = 1.0
gamma = 1.0
y0 = 1.
eth.domain = [-10, 10]
beta = .1
# A = 1.
def sd_zero_temp(w):
    return eta * w * gamma ** 4 * y0 ** 2\
           / ((ome ** 2 - w ** 2) ** 2 + 4 * w ** 2 * gamma ** 2)

eth.sd[0, 0] = lambda w: sd_zero_temp(w)*temp_factor(beta,w)


def sd_over_w(w):
    return eta * gamma ** 4 * y0 ** 2 * (1 / 3.1415926) \
           / ((ome ** 2 - w ** 2) ** 2 + 4 * w ** 2 * gamma ** 2)


sd = lambda w: sd_over_w(w)


reorg = integrate(sd, *eth.domain)

eth.hv_dy = [_c(*c) + _c(*c).T for i in range(3)]
eth.he_dy = [sigmaz() for i in range(3)]

a = kron(_c(*b), _c(*b).T) + kron(_c(*b).T, _c(*b)) + kron(_c(*b).T, _c(*b).T) + kron(_c(*b), _c(*b))
eth.h2ee = [a for i in range(2)]
eth.h2ev = [kron(sigmaz(), _c(*c) + _c(*c).T) for i in range(3)]
eth.h1e = [sigmaz() + sigmax() for i in range(3)]
eth.h1v = [_c(*c).T @ _c(*c) for i in range(3)]

eth.build()
print(eth.k_list, eth.w_list)

# print(eth.H[n][i])
# etn.U = eth.get_u(dt=0.02)
#
# for tn in range(2000):
#     # for ni in range(etn._nc - 1):
#     #     print("ni", ni)
#     #     print([x.shape for x in etn.ttnB[0]])
#     #     print([x.shape for x in etn.ttnB[1]])
#     #     etn.update_bond(-1, ni, 10, 1e-5)
#     #     print("ni complete", ni)
#
#     for n in range(0, 2):
#         for j in range(bath_length, etn._L[n] -1):
#             print("nj==", n, j)
#             etn.update_bond(n, j, 10, 1e-5)
#             print("H.shape", etn.U[n][j].shape)
#             print(eth.h1e[0])
#             print([x.shape for x in etn.ttnB[n][bath_length-1:]])
#             print([x.shape for x in etn.ttnS[n][bath_length-1:]])
#             print([x.shape for x in etn.U[n]][bath_length-1:])
