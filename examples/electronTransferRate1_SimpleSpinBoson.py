import numpy as np
from scipy.optimize import curve_fit
from fishbonett.fishbone import SpinBoson1D
from fishbonett.model import SpinBoson
from fishbonett.stuff import lorentzian, sigma_x, sigma_z, temp_factor

bath_length = 20
phys_dim = 40
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.


# spectral density parameters
g = 2000
eth.domain = [-g, g]
eta = 4.11  # 1.875e-5 au (in energy unit)
j = lambda w: 4 * lorentzian(eta, w) * temp_factor(77,w)

eth.sd = j

# electronic couplings
delta = 10.97373  # 5 × 10−5 au (in energy unit)
e = 3292.119  # 0.015 au (in energy unit)
# system Hamiltonian: Δσ_x + σ_z * ε/2
eth.he_dy = sigma_z
eth.h1e = e * sigma_z / 2 + delta * sigma_x

eth.build(g=1., ncap=60000)
print(eth.w_list)
print(eth.k_list)

U_one = eth.get_u(dt=0.001)
U_half = eth.get_u(dt=0.0005)

print(len(etn.B))
# ~ 0.5 ps ~ 0.1T
p = []
# time = 0.03644 T = 3644 steps if the time step is 1e-5
num_steps = 1000
etn.U = U_half
for tn in range(num_steps):
    for j in range(0, bath_length, 2):
        print("j==", j, tn)
        etn.update_bond(j, 1000, 1e-7)
    etn.U = U_one
    for j in range(1, bath_length, 2):
        print("j==", j, tn)
        etn.update_bond(j, 1000, 1e-7)
    etn.U = U_half
    for j in range(0, bath_length, 2):
        print("j==", j, tn)
        etn.update_bond(j, 1000, 1e-7)
    be = etn.B[-1]
    s = etn.S[-1]
    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 0])]

pop = [x for x in p[::2]]
print("population", pop)

# print("population", pop)

# time = [0.0001 * (i + 1) for i in range(num_steps)]
#
#
# def func(x, one, b):
#     return one * np.exp(-b * x)
#
#
# para, cov = curve_fit(func, time, pop)
#
# print(para)
