from model import FishBoneH, kron, _c
from fishbone import init_ttn, init
import numpy as np
from numpy import exp, tanh

bath_length = 5
a = [3]*bath_length
b = [2]
c = [4]
pd = np.array([[a[::-1], b, c, a], [a, b, c, a]], dtype=object)

eth = FishBoneH(pd)
etn = init_ttn(nc=2, L=bath_length, d1=3, de=2, dv=4)
etn2 = init(pd)

for n in range(2):
    print("Old", [x.shape for x in etn.ttnB[n]])
    print("New", [x.shape for x in etn2.ttnB[n]])
    print("Diff", [np.linalg.norm(x - y) for x, y in zip(etn.ttnB[n], etn2.ttnB[n])])

print(etn.ttnB[0][0])
print(etn2.ttnB[0][0])