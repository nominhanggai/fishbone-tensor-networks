from model import SpinBoson, kron, _c
from fishbone import SpinBoson1D
import numpy as np
from numpy import exp, tanh
from numpy.linalg import norm
from scipy.integrate import quad as integrate
def coth(x):
    return (exp(2*x)-1)/(exp(2*x)+1)

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
a = [2]*bath_length

pd = a + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
etn.B[-1][0,1,0] = 1/np.sqrt(2)
etn.B[-1][0,0,0] = 1./np.sqrt(2)
# electronic couplings
# tda = 100.0
e = 1000.

# spectral density parameters
g = 2500
eth.domain = [0, g]
ncap = 20000
lambd = 75
Omega = 150
s = 2
j = lambda x: lambd * ((x / Omega) ** s) * np.exp(-x / Omega)

eth.sd = j
# eth.sd = j
eth.he_dy = (np.eye(2) + sigmaz(2))/4
# eth.he_dy = np.eye(2)
eth.h1e = 1000/2 *sigmaz() # + 0.5*sigmax()


eth.build(g)
etn.U = eth.get_u(dt=0.0001)


p = []
be = etn.B[-1]
s = etn.S[-1]
c = np.einsum('Ibc,IJ->Jbc', be, np.diag(s))
c1 = c[0, 0, 0]
c2 = c[0, 1, 0]
p.append(c1*c2)

for tn in range(1000):
    print("ni complete", tn)
    for j in range(0, bath_length):
        print("j==", j)
        etn.update_bond(j, 50, 1e-20)
    # print(etn.B[1][:,:,0] )
    # print(etn.S[1])
    # print([x.shape for x in etn.U])
    be = etn.B[-1]
    s = etn.S[-1]
    c = np.einsum('Ibc, IJ->Jbc', be, np.diag(s))
    print(c[:,:,0])
    print(s)
    cc = c.conj()
    c1 = c[0,0,0]
    c2 = c[0,1,0]
    c3 = c[1,0,0]
    c4 = c[1,1,0]
    p.append(c1**2+c3**2)

print("population", [np.abs(x) for x in p])
print(s)
# # print(eth.w_list)
# print(eth.k_list)
# c = _c(10)
# hb = np.kron(449.97993819*c.T@c, np.eye(2))
# hs = np.kron(np.eye(10), eth.h1e)
# hi = 84.62805478*np.kron(c.T+c, eth.he_dy)
# h = hb+hs+hi
# U = (-1j*h*0.001)
# u = U.reshape(10,2,10,2)
# state = np.kron(np.array([1]+[0]*9), np.array([1]+[0])).reshape(10,2)
# print(u.shape,state.shape)
# s = np.einsum('ijkl,kl->ij',u, state)