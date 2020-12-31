from fishbonett.model import FishBoneH, kron, _c
from fishbonett.fishbone import FishBoneNet, init
import numpy as np
from numpy import exp, tanh, pi
from numpy.linalg import norm
from scipy.integrate import quad as integrate
from copy import deepcopy as dcopy

def sigmaz(d=2):
    z = np.zeros([d, d])
    z[0, 0] = 1
    z[1, 1] = -1
    return z


def sigmax(d=2):
    z = np.zeros([d, d])
    z[0, 1] = 1
    z[1, 0] = 1
    return z


def temp_factor(temp, w):
    beta = 1/(0.6950348009119888*temp)
    return 0.5 * (1. + 1. / tanh(beta * w / 2.))

bath_length = 2
a = [8]*bath_length
b = [2]
c = [4]
pd = np.array([[a, b, [], []], ], dtype=object)

eth = FishBoneH(pd)
etn = init(pd)

# electronic couplings
tda = 1.0
e = 1.0
#
# # spectral density parameters
# g=350
# eth.domain = [-g, g]
# S1 = 0.39; S2 = 0.23; S3 = 0.23
# s1 = 0.4; s2 = 0.25; s3 = 0.2
# w1 = 26; w2 = 51; w3 = 85
# temp = 0.0001
# def sd_back(Sk, sk, w, wk):
#     return Sk/(sk*np.sqrt(2/pi)) * w * \
#            np.exp(-np.log(np.abs(w)/wk)**2 / (2*sk**2))
#
# gamma = 5.
# Omega_1 = 181; Omgea_2 = 221; Omgea_3 = 240
# g1 = 0.0173; g2 = 0.0246; g3 = 0.0182
#
# def sd_high(gamma_m, Omega_m, g_m, w):
#     return 4*gamma_m*Omega_m*g_m*(Omega_m**2+gamma_m**2)*w / ((gamma_m**2+(w+Omega_m)**2)*(gamma_m**2+(w-Omega_m)**2))
#
# def sd_zero_temp(w):
#     return sd_back(S1,s1,w, w1)+sd_back(S2,s2,w,w2)+sd_back(S3,s3,w,w3) + \
#            sd_high(gamma, Omega_1, g1, w) + sd_high(gamma, Omgea_2, g2, w) + sd_high(gamma, Omgea_3, g3, w)


# def sd_over_w(w):
#     return eta * gamma ** 4 * y0 ** 2 * (1 / 3.1415926) \
#            / ((ome ** 2 - w ** 2) ** 2 + 4 * w ** 2 * gamma ** 2)
#
#
# sd = lambda w: sd_over_w(w)
# reorg = integrate(sd, *eth.domain)

g = 2500
eth.domain = [0, g]
ncap = 20000
lambd = 75
Omega = 150
s = 2
j = lambda x: lambd * ((x / Omega) ** s) * np.exp(-x / Omega)

eth.sd[0, 0] = j

# eth.hv_dy = [_c(*c) + _c(*c).T for i in range(3)]
eth.he_dy = [(np.eye(2) + sigmaz())/4 for i in range(2)]

# a = kron(_c(*b), _c(*b).T) + kron(_c(*b).T, _c(*b)) #+ kron(_c(*b).T, _c(*b).T) + kron(_c(*b), _c(*b))

# eth.h2ee = [a for i in range(2)]
# eth.h2ev = [kron(sigmaz(), _c(*c) + _c(*c).T) for i in range(2)]
eth.h1e = [1000./ 2 * sigmaz() for i in range(2)]
# eth.h1v = [_c(*c).T @ _c(*c) for i in range(2)]

eth.build(g)



etn.U = eth.get_u(dt=0.0001)
# print([x[0].shape for x in eth.get_h_total(0)])
# a = eth.get_h1(0)[0] + eth.get_h1(0)[1]
# print([x.shape for x in eth.get_h1(0)[0]])
# print(eth.get_h1(0)[1])

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

# a = eth.get_h2(0)
# print(a)
# print(b)
# print(c)