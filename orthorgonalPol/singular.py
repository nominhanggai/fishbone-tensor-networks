from fishbonett.stuff import sd_zero_temp
from scipy.integrate import *
from fishbonett.model import _c
# eta = 4.11
# f = lambda w: sd_zero_temp(w)/w
# g=1000
# a = quad(f, 0, g)
# print(a)
print(_c(4))

from fishbonett.stuff import sigma_z, sigma_x
from fishbonett.model import _c
import numpy as np
from scipy.linalg import expm
# [ 1.39341179e+01 -1.20576843e+01  6.60071326e+01 -9.81435002e+00
#  -2.49009939e+00 -7.44357604e-01 -3.25310997e-01 -1.72662343e-01
#  -1.02837123e-01 -6.62454775e-02 -4.51910227e-02 -3.22108175e-02
#  -2.37702342e-02 -1.80431798e-02 -1.40198460e-02 -1.11103184e-02
#  -8.95400136e-03 -7.32191162e-03 -6.06383403e-03 -5.07841685e-03]
# [1487.07351406   75.43508295   28.86502729  163.08166269  156.13277651
#   152.34496506  151.24396057  150.77905627  150.53514974  150.39065063
#   150.29784526  150.23465417  150.18967011  150.15650329  150.13134394
#   150.1118044   150.09632584  150.0838552   150.07366016  150.06521856]
dim = 1000
#
delta = 10.97373  # 5 × 10−5 au (in energy unit)
e = 3292.119  # 0.015 au (in energy unit)
h0 = np.kron(e * sigma_z / 2 + delta * sigma_x, np.eye(dim))
c = _c(dim)
h1 = np.kron(np.eye(2), 1.39341179e+01 * c.T@c)
hI = .1* 1487.07351406*np.kron(sigma_z, (c.T+c))
h = h0 + h1 + hI

w, e = np.linalg.eig(h)


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
