from model import FishBoneH, kron, _c
from fishbone import init_ttn
import numpy as np


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

bath_length = 190
a = [3] * bath_length
b = [2]
c = [4]
pd = np.array([[a, b, c, a], [a, b, c, a], [a, b, c, a]], dtype=object)

eth = FishBoneH(pd)
etn = init_ttn(nc=3, L=bath_length, d1=3, de=2, dv=4)

eth.hv_dy = [_c(*c)+_c(*c).T for i in range(3)]
eth.he_dy = [sigmaz() for i in range(3)]

a = kron(_c(*b), _c(*b).T) + kron(_c(*b).T, _c(*b)) + kron(_c(*b).T, _c(*b).T) + kron(_c(*b), _c(*b))
eth.h2ee = [a for i in range(2)]
eth.h2ev = [kron(sigmaz(), _c(*c) + _c(*c).T) for i in range(3)]
eth.h1e = [sigmaz() + sigmax() for i in range(3)]
eth.h1v = [_c(*c).T @ _c(*c) for i in range(3)]
eth.build()
etn.U = eth.get_u(dt=0.02)


for tn in range(2000):
    for ni in range(etn._nc - 1):
        print("ni", ni)
        print([x.shape for x in etn.ttnB[0]])
        print([x.shape for x in etn.ttnB[1]])
        print([x.shape for x in etn.ttnB[2]])
        etn.update_bond(-1, ni, 120, 1e-10)
        print("ni complete", ni)

    for n in range(0, 3):
        for j in range(0, etn._L[n] -1):
            print("nj==", n, j)
            etn.update_bond(n, j, 120, 1e-10)
            print("H.shape", etn.U[n][j].shape)
            print(eth.h1e[0])
            print([x.shape for x in etn.ttnB[n]])
            print([x.shape for x in etn.ttnS[n]])
            print([x.shape for x in etn.U[n]])