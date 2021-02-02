import numpy as np
from scipy.optimize import curve_fit
from fishbonett.fishbone import SpinBoson1D
from fishbonett.model import SpinBoson
from fishbonett.stuff import lorentzian, sigma_x, sigma_z, temp_factor

bath_length = 100
phys_dim = 100
a = [int(np.ceil(phys_dim - (phys_dim - 2) * (N/bath_length)**0.2)) for N in range(bath_length)]
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
print("ETH Complete")
etn = SpinBoson1D(pd)
print("ETN Complete")
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.


# spectral density parameters
g = 350
eth.domain = [-g, g]
eta = 4.11  # 1.875e-5 au (in energy unit)
j = lambda w:  4*lorentzian(eta, w) * temp_factor(300,w)

eth.sd = j

# electronic couplings
delta = 10.97373  # 5 × 10−5 au (in energy unit)
e = 3292.119  # 0.015 au (in energy unit)
# system Hamiltonian: Δσ_x + σ_z * ε/2
eth.he_dy = sigma_z
eth.h1e = e * sigma_z / 2 + delta * sigma_x

eth.build(g=1., ncap=20000)
print(eth.w_list)
print(eth.k_list)

U_one = eth.get_u(dt=0.002)
U_half = eth.get_u(dt=0.001)

print(len(etn.B))
# ~ 0.5 ps ~ 0.1T
p = []
occu = []
# time = 0.03644 T = 3644 steps if the time step is 1e-5
num_steps = 200
etn.U = U_half
bond_dim = 200
threshold = 1e-3

for tn in range(num_steps):
    for j in range(0, bath_length, 2):
        print("Step Number:", tn, "Bond", j)
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold)
    etn.U = U_one
    for j in range(1, bath_length, 2):
        print("Step Number:", tn, "Bond", j)
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold)
    etn.U = U_half
    for j in range(0, bath_length, 2):
        print("Step Number:", tn, "Bond", j)
        etn.update_bond(j, bond_dim, threshold)
    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 0])]

    for i in range(bath_length):
        theta = etn.get_theta1(i)
        rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
        occu.append(np.diagonal(rho))

pop = [x for x in p]
print("population", pop)

# print(occu)
