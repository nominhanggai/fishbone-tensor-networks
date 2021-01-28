from fishbonett.model import FishBoneH
from fishbonett.fishbone import init
import numpy as np
from fishbonett.stuff import temp_factor, sigma_z, sigma_0




bath_length = 12
a = [5,6,7,8,9,10,11,12,13,14,15,16]
b = [2]
pd = np.array([[a, b, [], a], ], dtype=object)

eth = FishBoneH(pd)
etn = init(pd)


g = 2500
eth.domain = [0, g]
ncap = 20000
lambd = 75
Omega = 150
s = 2
j = lambda x: lambd * ((x / Omega) ** s) * np.exp(-x / Omega)

eth.sd[0, 0] = j
# eth.sd[0, 1] = j


eth.he_dy = [(sigma_0 + sigma_z)/4]*2
eth.h1e = [1000. / 2 * sigma_z]*2
eth.h1v = eth.h1e
eth.hv_dy = [(sigma_0 + sigma_z)/3]*2


eth.build(g)
print(eth._H[0][11])
print(eth._H[0][12])
etn.U = eth.get_u(dt=0.0001)

print(len(etn.ttnS[0]))

p = []

for tn in range(1000):
    for n in range(0, 1):
        for j in range(0, bath_length):
            print("Bond ==", n, j)
            etn.update_bond(n, j, 10, 1e-5)
            print([x.shape for x in etn.ttnB[n][:bath_length+1]])
            print([x.shape for x in etn.ttnS[n][:bath_length+1]])
            print([x.shape for x in etn.U[n]][:bath_length+1])
    be = etn.ttnB[n][bath_length]
    s = etn.ttnS[n][bath_length]
    print(be.shape, s.shape)
    c = np.einsum('Ibcde,IJ->Jbcde', be, np.diag(s))
    c1 = c[0, 0,0,0,0]
    c2 = c[0, 1,0,0,0]
    c3 = c[1, 0,0,0,0]
    c4 = c[1, 1,0,0,0]
    p.append(c1 * c2.conj() + c3 * c4.conj())

print("population", [np.abs(x) for x in p[::4]])
