from fishbonett.model import FishBoneH
from fishbonett.fishbone import FishBoneNet, init_ttn
import numpy as np

a = [3, 3, 3]
b = [2]
pd = np.array([[a, b, b, a], [a, b, b, a]], dtype=object)

eeth = FishBoneH(pd)
eetn = init_ttn(nc=1, L=3, d1=3, de=2, dv=5)

# print("h====", eeth.H[0][7])
eetn.ttnH = eeth.get_u(dt=0.01)
print(eetn.ttnH[0][0].shape)
eetn.update_bond(0, 0, 9, 1e-2)




