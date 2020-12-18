from fishbonett.model import FishBoneH, calc_U, _c
from fishbonett.fishbone import FishBoneNet, init_ttn
import numpy as np
from scipy.linalg import svd
from scipy.linalg import expm

a = [2]*4
b = [3]
c = [5]
pd = np.array([[a, b, c, a], [a, b, c, a], [a, b, c, a]], dtype=object)

eeth = FishBoneH(pd)
eetn = init_ttn(nc=3, L=4, d1=2, de=3, dv=5)

# print("h====", eeth.H[0][7])
eetn.ttnH = eeth.get_u(dt=0.02)
# [print(x.shape) for x in eetn.ttnH[0]]
a = eeth.w_list[0][0][0]
b = eeth.w_list[0][0][1]
k = eeth.k_list[0][0][1]
c = _c(2)
h = np.kron(c.T, c)+ np.kron(c, c.T) #+ np.kron(b*c.T@c, np.eye(2))
print(c.T@c)
h = calc_U(h, 0.1)
print(h)
h = h.reshape(2,2,2,2)

a = np.zeros([1, 2, 1], np.float)
a[0,0,0] = 1
b = np.zeros([1, 2, 1], np.float)
b[0,0,0] = 1
print(h[:,:,0,0])
ab = np.tensordot(a,b, [2,0])
print(ab)
ab = np.einsum('ijkl,kl->ij',h, [[1,0],[0,0]])
#i j i j * L i j R -> i j L R
print(ab.shape)
ab = ab.reshape(2,2)
print("ab*h",ab)
a,s,b = svd(ab,full_matrices=False)
#print(a,s,b)

# print([x.shape for x in eetn.ttnB[0]])
# print("ebl", eetn._ebL, eetn._vbL, )
# print("ttnH", eetn.ttnH[0][8].reshape(4,4))
# print("U", eetn.ttnH[0][8].reshape(4,4)- calc_U(eeth.H[0][8][0], 0.02))
# for i in range(2):
#    print("i===", i)
#    #print("EET", eetn.ttnB[0] )
#    eetn.update_bond(0, 8, 12, 1e-12)
#print(eetn.ttnS[0][4],eetn.ttnS[0][4].shape)

#print("EET", eetn.ttnB[0] )



