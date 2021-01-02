from fishbonett.model import SpinBoson, kron, _c
from fishbonett.fishbone import SpinBoson1D
import numpy as np
from numpy import exp, tanh
from numpy.linalg import norm
from scipy.linalg import expm, svd
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


bath_length = 120
a = [8]*bath_length

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
eth.he_dy = (np.eye(2) + sigmaz(2))/4
eth.h1e = 1000/2 *sigmaz()
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
    for j in range(0, bath_length):
        print("j==", j, tn )
        etn.update_bond(j, 10, 1e-5)
    be = etn.B[-1]
    s = etn.S[-1]
    c = np.einsum('Ibc, IJ->Jbc', be, np.diag(s))
    cc = c.conj()
    c1 = c[0,0,0]
    c2 = c[0,1,0]
    c3 = c[1,0,0]
    c4 = c[1,1,0]
    p.append(c1*c2.conj()+c3*c4.conj())

print("population", [np.abs(x) for x in p[::4]])

