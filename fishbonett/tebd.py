from model import FishBoneH, kron, _c
from fishbone import init
import numpy as np
from numpy import exp, tanh
from numpy.linalg import norm
from scipy.integrate import quad as integrate
def coth(x):
    return (exp(2*x)-1)/(exp(2*x)+1)

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


def temp_factor(beta, w):
    return 0.5 * (1. + 1. / tanh(beta * w / 2.))


bath_length = 5
a = [9,8,7,6,5]
b = [2]
c = [4]
pd = np.array([[a, b, c, a[::-1]], [a, b, c, a[::-1]]], dtype=object)

eth = FishBoneH(pd)
etn = init(pd)

# electronic couplings
tda = 0.1
e = 1.0

#
# half_my_ome_2 = 1.0
# ome = 1.0
# my = 2 * half_my_ome_2/(ome**2)
#
# spectral density parameters
eta = 1.
ome = 1.0
gamma = 1.0
y0 = 1.
eth.domain = [-10, 10]
beta = .1
# A = 1.
def sd_zero_temp(w):
    return eta * w * gamma ** 4 * y0 ** 2\
           / ((ome ** 2 - w ** 2) ** 2 + 4 * w ** 2 * gamma ** 2)

eth.sd[0, 0] = lambda w: sd_zero_temp(w)*temp_factor(beta,w)


def sd_over_w(w):
    return eta * gamma ** 4 * y0 ** 2 * (1 / 3.1415926) \
           / ((ome ** 2 - w ** 2) ** 2 + 4 * w ** 2 * gamma ** 2)


sd = lambda w: sd_over_w(w)
reorg = integrate(sd, *eth.domain)

eth.hv_dy = [_c(*c) + _c(*c).T for i in range(3)]
eth.he_dy = [np.eye(2) for i in range(3)]

a = kron(_c(*b), _c(*b).T) + kron(_c(*b).T, _c(*b)) #+ kron(_c(*b).T, _c(*b).T) + kron(_c(*b), _c(*b))

eth.h2ee = [a for i in range(2)]
eth.h2ev = [kron(sigmaz(), _c(*c) + _c(*c).T) for i in range(3)]
eth.h1e = [e*sigmaz() #+ tda*sigmax()
           for i in range(3)]
eth.h1v = [_c(*c).T @ _c(*c) for i in range(3)]

eth.build()



etn.U = eth.get_u(dt=0.001)
p = []
for tn in range(1000):
    # # for ni in range(etn._nc - 1):
    # #     print("ni", ni)
    # #     print([x.shape for x in etn.ttnB[0]])
    # #     print([x.shape for x in etn.ttnB[1]])
    # #     etn.update_bond(-1, ni, 10, 1e-5)
    # #     print("ni complete", ni)
    #
    for n in range(0, 1):
        for j in range(0, 5):
            print("nj==", n, j)
            etn.update_bond(n, j, 100, 1e-10)

            be = etn.ttnB[n][bath_length]
            s = etn.ttnS[n][bath_length]
            print(be.shape,s.shape)
            c = np.einsum('Ibcde,IJ->Jbcde',be,np.diag(s))
            p.append(c[0,0,0,0,0])
            print([x.shape for x in etn.ttnB[n][:bath_length+1]])
            print([x.shape for x in etn.ttnS[n][:bath_length+1]])
            print([x.shape for x in etn.U[n]][:bath_length+1])
# #
#
# s = etn.ttnS[0][2]
# print("S",s , np.linalg.norm(s))

print(etn.ttnB[0])
p=np.array([p])
print("population", [np.abs(x)**2 for x in p])
#
# h0 = eth.H[0][0][0]
# h1 = eth.H[0][1][0]
# h2 = eth.H[0][2][0]
# h3 = eth.H[0][3][0]
#
# print(eth.k_list)
# print(eth.w_list)
# # k0 = 5.15693988
# k0 = eth.k_list[0][0][-1]
# # w0 = -0.04314381
# w0 = eth.w_list[0][0][-1]
# # k1 = 5.4966608
# k1 = eth.k_list[0][0][-2]
# # w1 = -0.21686089
# w1 = eth.w_list[0][0][-2]
# # w2 = 1.01884703
# k2 = eth.k_list[0][0][-3]
# w2 = eth.w_list[0][0][-3]
# k3 = eth.k_list[0][0][-4]
# w3 = eth.w_list[0][0][-4]
# # k4 = eth.k_list[0][0][-5]
w4 = eth.w_list[0][0][-5]
print(w4)
#
# c9 = _c(9)
# c8 = _c(8)
# c7 = _c(7)
# c6 = _c(6)
c5 = _c(5)
#
#
# h0p = np.kron(w0*c9.T@c9, np.eye(8)) + k0*(np.kron(c9.T,c8) + np.kron(c9,c8.T))
# h1eb,_,_ = eth.get_h1(0)
# h10 = h1eb[0]
#
#
# h1p = np.kron(w1*c8.T@c8, np.eye(7)) + k1*(np.kron(c8.T,c7) + np.kron(c8,c7.T))
# h2p = np.kron(w2*c7.T@c7, np.eye(6)) + k2*(np.kron(c7.T,c6) + np.kron(c7,c6.T))
# h3p = np.kron(w3*c6.T@c6, np.eye(5)) + k3*(np.kron(c6.T,c5) + np.kron(c6,c5.T))
# # h4p = np.kron(w4*c8.T@c5, np.eye(7)) + k4*(np.kron(c8.T,c7) + np.kron(c8,c7.T))
#
# # print(norm(h0p-h0))
# # print(norm(h1p-h1))
# #
# # print(norm(h2p-h2))
#
# print(norm(h3p-h3))
print(c5.T@c5)