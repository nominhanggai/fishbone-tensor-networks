import numpy as np
from fishbonett.int_pic_hsb_spin_boson import SpinBosonModel
from fishbonett.spin_boson_mps import SpinBosonMPS
from fishbonett.fishbone import SpinBoson1D as SpinBosonMPS_ref
from fishbonett.model import SpinBoson as SpinBosonModel_ref
from fishbonett.stuff import drude, temp_factor, sigma_z, sigma_x, entang
from time import time

bath_length = 20
phys_dim = 20
pd_boson = [phys_dim]*bath_length
pd = pd_boson[::-1] + [2]
g = 3000
temp = 0.1
sd = lambda w: 10*drude(w, 1000, 5)#*temp_factor(temp,w)
bond_dim = 100000
threshold = 1e-3
dt = 0.01 / 20
num_steps = 40

eth = SpinBosonModel_ref(pd)
eth.domain = [-0, g]
eth.sd = sd
eth.he_dy = sigma_z
eth.h1e = 300 * sigma_x
eth.build(g=1., ncap=20000)

print("eth.w_list", eth.w_list)
print("eth.k_list", eth.k_list)
w_list_0 = eth.w_list
k_list_0 = eth.k_list

etn = SpinBosonMPS(pd_spin=2, pd_boson=pd_boson)
etn = SpinBosonMPS_ref(pd)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[0][0, 1, 0] = 0
etn.B[0][0, 0, 0] = 1

p = []
entanglement = []

u_one = eth.get_u(dt)
u_half = eth.get_u(dt/2)


print("H, 0", eth.H[-1][0].toarray())
print("H, 0", eth.H[-2][0].toarray())
t = 0.
for n in range(num_steps):
    # eth.get_u(n*dt, dt)
    etn.U = u_half
    for j in range(0, bath_length, 2):
        print("Step Number:", n, "Bond", j)
        print("j==", j, n)
        # etn.update_bond(j, bond_dim, threshold, swap=False)
        etn.update_bond(j, bond_dim, threshold)
    etn.U = u_one
    for j in range(1, bath_length, 2):
        print("Step Number:", n, "Bond", j)
        print("j==", j, n)
        # etn.update_bond(j, bond_dim, threshold, swap=False)
        etn.update_bond(j, bond_dim, threshold)
    etn.U = u_half
    for j in range(0, bath_length, 2):
        print("Step Number:", n, "Bond", j)
        # etn.update_bond(j, bond_dim, threshold, swap=False)
        etn.update_bond(j, bond_dim, threshold)
    # theta = etn.get_theta1(0) # c.shape vL i vR
    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 0])]
    entanglement.append(sum([entang(s) for s in etn.S]))

pop0 = p
entanglement0 = entanglement
# print(t)

#  ##############
#  ##############
eth = SpinBosonModel(v_x=300, v_z=0, pd_spin=2,
                    pd_boson=pd_boson, boson_domain=[0, g],
                    sd=sd, dt=dt)
print("H 1", eth.h_full[0][0].toarray())
print("H 1", eth.h_full[1][0].toarray())
print("eth.w_list", eth.w_list)
print("eth.k_list", eth.k_list)
w_list_1 = eth.w_list
k_list_1 = eth.k_list

etn = SpinBosonMPS(pd_spin=2, pd_boson=pd_boson)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[0][0, 1, 0] = 0
etn.B[0][0, 0, 0] = 1

p = []
entanglement = []
t = 0.
for n in range(num_steps):
    u_one, u_half = eth.get_u(n*dt, dt)
    etn.U = u_half
    for j in range(0, bath_length, 2):
        print("Step Number:", n, "Bond", j)
        print("j==", j, n)
        etn.update_bond(j, bond_dim, threshold, swap=False)
        # etn.update_bond(j, bond_dim, threshold)
    etn.U = u_one
    for j in range(1, bath_length, 2):
        print("Step Number:", n, "Bond", j)
        print("j==", j, n)
        etn.update_bond(j, bond_dim, threshold, swap=False)
        # etn.update_bond(j, bond_dim, threshold)
    etn.U = u_half
    for j in range(0, bath_length, 2):
        print("Step Number:", n, "Bond", j)
        etn.update_bond(j, bond_dim, threshold, swap=False)
        # etn.update_bond(j, bond_dim, threshold)
    theta = etn.get_theta1(0) # c.shape vL i vR
    # theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 0])]
    theta_bosons = [etn.get_theta1(j) for j in range(1, bath_length)]
    rho_boson = [np.einsum('LiR,LjR->ij',  theta, theta.conj()) for theta in theta_bosons]
    entanglement.append(sum([entang(s) for s in etn.S]))
    # eigenvalues = np.linalg.eigvalsh(rho_boson[1])
    # s = -sum(eigenval * np.log(eigenval) for eigenval in eigenvalues if eigenval > 0)
    # entanglement.append(s)
pop1 = p
entanglement1 = entanglement

# exit()
# import numpy as np
# from scipy.optimize import curve_fit
# from fishbonett.backwardSpinBoson import SpinBoson
# from fishbonett.spinBosonMPS import SpinBoson1D
# from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude, entang
# from scipy.linalg import expm
# from time import time

# a = [phys_dim]*bath_length
# print(a)

# pd = a[::-1] + [2]
# eth = SpinBoson(pd)
# etn = SpinBoson1D(pd)
# # set the initial state of the system. It's in the high-energy state |0>:
# # if you doubt this, pls check the definition of sigma_z
# etn.B[-1][0, 1, 0] = 0
# etn.B[-1][0, 0, 0] = 1


# # spectral density parameters
# g = 3000
# eth.domain = [-0, g]
# temp = 0.1
# # j = lambda w: 0.001*drude(w, 1000, 5)*temp_factor(temp,w)
# # j = lambda w: 0.001*drude(w, 1000, 5)*temp_factor(temp,w)
# eth.sd = sd

# eth.he_dy = sigma_z
# eth.h1e = 300*sigma_x

# eth.build(g=1., ncap=20000)
# print(eth.w_list)
# print(eth.k_list)
# k_list_ref = eth.k_list

# # U_one = eth.get_u(dt=0.002, t=0.2)

# # ~ 0.5 ps ~ 0.1T
# p = []

# s_dim = np.empty([0,0])
# dt = dt/2
# t = 0.
# for tn in range(num_steps):
#     U1, U2 = eth.get_u(2*tn*dt, dt,mode='normal')

#     t0 = time()
#     etn.U = U1
#     for j in range(bath_length-1,0,-1):
#         print("j==", j, tn)
#         etn.update_bond(j, bond_dim, threshold, swap=1)

#     etn.update_bond(0, bond_dim, threshold, swap=0)
#     etn.update_bond(0, bond_dim, threshold, swap=0)
#     t1 = time()
#     t = t + t1 - t0

#     U1, U2 = eth.get_u((2*tn+1) * dt, dt, mode='reverse')

#     t0 = time()
#     etn.U = U2
#     for j in range(1, bath_length):
#         print("j==", j, tn)
#         etn.update_bond(j, bond_dim, threshold,swap=1)

#     dim = [len(s) for s in etn.S]
#     s_dim = np.append(s_dim, dim)

#     theta = etn.get_theta1(bath_length) # c.shape vL i vR
#     rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
#     p = p + [np.abs(rho[0, 0])]
#     t1 = time()
#     t = t + t1 - t0

# # t1 = time()
# pop = [x for x in p]
# print("population", pop)

import matplotlib.pyplot as plt
# plt.plot(pop0, label='ref')
# plt.plot(pop1, label='new')
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Spin population plot
axs[0].plot(pop0, label='ref')
axs[0].plot(pop1, label='new')
axs[0].legend()
# Oscillator occupation number plot
axs[1].plot(entanglement0, label='ref')
axs[1].plot(entanglement1, label='new')
axs[1].legend()
# Leave the last plot empty
# axs[1, 1].axis('off')
plt.tight_layout()
plt.savefig('comparison_hsb.png')

# print("k_list_0", k_list_0)
# print("k_list_1", k_list_1)
# print("w_list_0", w_list_0)
# print("w_list_1", w_list_1)
