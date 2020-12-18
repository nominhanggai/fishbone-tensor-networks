from fishbonett.model import FishBoneH, kron, calc_U, _c
from fishbonett.fishbone import FishBoneNet, init_ttn
import numpy as np


def sigmaz(d=2):
    z = np.zeros(d)
    z[0, 0] = -1
    z[1, 1] = 1
    return z


a = [9] * 4
b = [2]
c = [5]
pd = np.array([[a, b, c, a], [a, b, c, a], [a, b, c, a]], dtype=object)

eeth = FishBoneH(pd)
eetn = init_ttn(nc=3, L=4, d1=9, de=2, dv=5)


eeth.hv_dy = [_c(*c) for i in range(3)]
eeth.he_dy = [sigmaz() for i in range(3)]
a = kron(_c(*b), _c(*b)) + kron(_c(*b), _c(*b).T) + kron(_c(*b).T, _c(*b)) + kron(_c(*b).T, _c(*b).T)
eeth.h2ee = [a for i in range(2)]
eeth.h2ev = [kron(sigmaz(), _c(*c)+_c(*c).T) for i in range(3)]
eeth.h1e = [sigmaz() for i in range(3)]
eeth.h1v = [_c(*c).T@_c(*c) for i in range(3)]
eetn.ttnH = eeth.get_u(dt=0.02)

for i in range(0, 3):
    for j in range(eetn._L[i] - 1):
        print("ij==", i, j)
        eetn.update_bond(i, j, 12, 1e-12)
        print([x.shape for x in eetn.ttnB[0]])

for j in range(eetn._nc - 1):
    print("ij==", j)
    eetn.update_bond(-1, j, 12, 1e-12)
    print([x.shape for x in [eetn.ttnB[j][4]]])

