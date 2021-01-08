from fishbonett.model import SpinBoson, kron, _c
from fishbonett.fishbone import SpinBoson1D
import numpy as np
from numpy import exp, tanh, pi
from numpy.linalg import norm
from scipy.linalg import expm, svd
from scipy.integrate import quad as integrate


def coth(x):
    return (exp(2 * x) - 1) / (exp(2 * x) + 1)


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
    beta = 1 / (0.6950348009119888 * temp)
    return 0.5 * (1. + 1. / tanh(beta * w / 2.))


bath_length = 120
a = [20] * bath_length

pd = a + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
etn.B[-1][0, 1, 0] = 1 / np.sqrt(2)
etn.B[-1][0, 0, 0] = 1. / np.sqrt(2)

# electronic couplings
e = 1.
eth.he_dy = (np.eye(2) + sigmaz(2)) / 2
eth.h1e = e / 2 * sigmaz()  # + 500*sigmax()

eth.domain = [-350, 350]
S1 = 0.39;
S2 = 0.23;
S3 = 0.23
s1 = 0.4;
s2 = 0.25;
s3 = 0.2
w1 = 26;
w2 = 51;
w3 = 85



def sd_back(Sk, sk, w, wk):
    return Sk / (sk * np.sqrt(2 / pi)) * w * \
           np.exp(-np.log(np.abs(w) / wk) ** 2 / (2 * sk ** 2))


gamma = 5.
Omega_1 = 181;
Omgea_2 = 221;
Omgea_3 = 240
g1 = 0.0173;
g2 = 0.0246;
g3 = 0.0182


def sd_high(gamma_m, Omega_m, g_m, w):
    return 4 * gamma_m * Omega_m * g_m * (Omega_m ** 2 + gamma_m ** 2) * w / (
                (gamma_m ** 2 + (w + Omega_m) ** 2) * (gamma_m ** 2 + (w - Omega_m) ** 2))


def sd_zero_temp(w):
    return sd_back(S1, s1, w, w1) + sd_back(S2, s2, w, w2) + sd_back(S3, s3, w, w3) + \
           sd_high(gamma, Omega_1, g1, w) + sd_high(gamma, Omgea_2, g2, w) + sd_high(gamma, Omgea_3, g3, w)

temp = np.float64(300)
eth.sd = lambda w: sd_zero_temp(w) * temp_factor(temp, w)

eth.build(g=350)
etn.U = eth.get_u(dt=0.00001)
print(eth.w_list)
print(eth.k_list)
exit()

p = []
be = etn.B[-1]
s = etn.S[-1]
c = np.einsum('Ibc,IJ->Jbc', be, np.diag(s))
c1 = c[0, 0, 0]
c2 = c[0, 1, 0]
p.append(c1 * c2)

for tn in range(2000):
    print("ni complete", tn)
    for j in range(0, bath_length):
        print("j==", j)
        etn.update_bond(j, 10, 1e-15)
    be = etn.B[-1]
    s = etn.S[-1]
    c = np.einsum('Ibc, IJ->Jbc', be, np.diag(s))
    cc = c.conj()
    c1 = c[0, 0, 0]
    c2 = c[0, 1, 0]
    c3 = c[1, 0, 0]
    c4 = c[1, 1, 0]
    p.append(c1 * c2.conj() + c3 * c4.conj())
print("population", [np.abs(x) for x in p])
print("population", [np.abs(x) for x in p[::10]])

