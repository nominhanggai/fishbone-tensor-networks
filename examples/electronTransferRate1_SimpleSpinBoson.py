import numpy as np
from scipy.optimize import curve_fit
from fishbonett.fishbone import SpinBoson1D
from fishbonett.model import SpinBoson
from fishbonett.stuff import lorentzian, sigma_x, sigma_z, temp_factor

bath_length = 20
phys_dim = 40
a = [np.rint(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
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

eth.build(g,ncap=20000)
print(eth.w_list)
print(eth.k_list)


etn.U = eth.get_u(dt=0.0002)
print(len(etn.B))

p = []
# time = 0.03644 T = 3644 steps if the time step is 1e-5
num_steps = 500
for tn in range(num_steps):
    for j in range(0, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, 1000, 1e-7)
    be = etn.B[-1]
    s = etn.S[-1]
    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p.append(np.abs(rho[0, 0]))

pop = [np.abs(x) for x in p]
print("population", pop)

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
