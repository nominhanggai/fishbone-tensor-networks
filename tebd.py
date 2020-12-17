from fishbonett.model import FishBoneH
from fishbonett.fishbone import FishBoneNet, init_ttn
import numpy as np

a = [9, 9, 9]
b = [3]
c = [5]
pd = np.array([[a, b, c, a], [a, b, c, a]], dtype=object)

eeth = FishBoneH(pd)
eetn = init_ttn(nc=2, L=3, d1=9, de=3, dv=5)

# print("h====", eeth.H[0][7])
eetn.ttnH = eeth.get_u(dt=0.01)
print([x.shape for x in eetn.ttnH[0]])
print(eetn.ttnH[0][0].shape)
# print("ebl", eetn._ebL, eetn._vbL, )
# print("B", eetn.ttnB)
eetn.update_bond(0, 2, 9, 1e-2)




