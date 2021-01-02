import numpy as np
from scipy.optimize import curve_fit
from fishbonett.fishbone import SpinBoson1D
from fishbonett.model import SpinBoson
from fishbonett.stuff import lorentzian, sigma_x, sigma_z, temp_factor

bath_length = 120
a = [8] * bath_length

pd = a + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.
# electronic couplings
delta = 10.97373  # 5 × 10−5 au (in energy unit)
e = 3292.119  # 0.015 au (in energy unit)

# spectral density parameters
g = 2000
eth.domain = [-g, g]
eta = 4.11  # 1.875e-5 au (in energy unit)
j = lambda w: lorentzian(eta, w) * temp_factor(300,w)

eth.sd = j

# system Hamiltonian: Δσ_x + σ_z * ε/2
eth.he_dy = sigma_z
eth.h1e = e * sigma_z / 2 + delta * sigma_x

eth.build(g,ncap=20000)
print(eth.w_list)

print(eth.k_list)

# etn.U = eth.get_u(dt=0.0001)
#
# p = []
#
# for tn in range(1000):
#     for j in range(0, bath_length):
#         print("j==", j, tn)
#         etn.update_bond(j, 10, 1e-5)
#     be = etn.B[-1]
#     s = etn.S[-1]
#     c = np.einsum('Ibc, IJ->Jbc', be, np.diag(s))
#     cc = c.conj()
#     c1 = c[0, 0, 0]
#     c2 = c[0, 1, 0]
#     c3 = c[1, 0, 0]
#     c4 = c[1, 1, 0]
#     p.append(c1 * c1.conj() + c3 * c3.conj())
#
# pop = [np.abs(x) for x in p]
# print("population", pop)
# time = [0.001 * (i + 1) for i in range(1000)]
#
#
# def func(x, one, b):
#     return one * np.exp(-b * x)
#
#
# para, cov = curve_fit(func, time, pop)
#
# print(para)
