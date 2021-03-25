import numpy as np
from scipy.optimize import curve_fit
from fishbonett.fishbone import SpinBoson1D
from fishbonett.backwardSpinBoson import SpinBoson
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, drude1

bath_length = 10
phys_dim = 20
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
# freq 47.46823245   11.65575389 1286.44557149 1031.29582863  964.64706021
#   908.94178924  849.73101296  784.87865938  714.53820807  639.37568193
#   560.11649672  477.41949141  391.87063916  304.13522693  215.63249981
#   130.38218742   57.2196865     7.39010682  -14.98469809  -17.81550915
#  [70, 283, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 9, 11, 16, 26, 58, 446, -219, -184]

a = [int(x) for x in a]
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
# etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
# etn.B[-1][0, 1, 0] = 0.
# etn.B[-1][0, 0, 0] = 1.


# spectral density parameters
g = 350
eth.domain = [0, g]
temp = 0
j = lambda w: drude1(w, 100)

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

U_one = eth.get_u(dt=0.002, t=0.2)
# U_half = eth.get_u(dt=0.001, t=0.2)

# print(len(etn.B))
# # ~ 0.5 ps ~ 0.1T
# p = []
# occu = []
# # time = 0.03644 T = 3644 steps if the time step is 1e-5
# num_steps = 50
# etn.U = U_half
# bond_dim = 1000
# threshold = 1e-3
#
# for tn in range(num_steps):
#     for j in range(0, bath_length, 2):
#         print("j==", j, tn)
#         etn.update_bond(j, bond_dim, threshold)
#     etn.U = U_one
#     for j in range(1, bath_length, 2):
#         print("j==", j, tn)
#         etn.update_bond(j, bond_dim, threshold)
#     etn.U = U_half
#     for j in range(0, bath_length, 2):
#         print("j==", j, tn)
#         etn.update_bond(j, bond_dim, threshold)
#     theta = etn.get_theta1(bath_length) # c.shape vL i vR
#     rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
#     p = p + [np.abs(rho[0, 0])]
#
#     for i in range(bath_length):
#         theta = etn.get_theta1(i)
#         rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
#         occu.append(np.diagonal(rho))
#
# pop = [x for x in p]
# print("population", pop)

# print(occu)
