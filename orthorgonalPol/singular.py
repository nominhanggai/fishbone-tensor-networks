from fishbonett.stuff import sd_zero_temp
from scipy.integrate import *
from fishbonett.model import _c
# eta = 4.11
# f = lambda w: sd_zero_temp(w)/w
# g=1000
# a = quad(f, 0, g)
# print(a)


from fishbonett.stuff import sigma_z, sigma_x, natphys
f = lambda w: natphys(w, 35)/(w*pi
a = quad(f, 0, 600)
print(a)

from fishbonett.model import _c
import numpy as np
from scipy.linalg import expm


# def U(t):
#     return expm(-1j*h*t)
#
# psi0 = np.kron([1,0], [1]+[0]*(dim-1))
#
# states = [U(0.02*t)@psi0 for t in range(0,50)]
# e1 = np.array([s.conj()@h1@s for s in states])
# e0 = np.array([s.conj()@(h0+hI)@s for s in states])
#
# print([np.abs(x) for x in e1])
