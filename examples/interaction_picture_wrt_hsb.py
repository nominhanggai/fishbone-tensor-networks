import numpy as np
from fishbonett.int_pic_hsb_spin_boson import SpinBosonModel
from fishbonett import spin_boson_mps
from fishbonett.spin_boson_mps import SpinBosonMPS
from fishbonett.stuff import sigma_x, sigma_z, drude, temp_factor
from time import time

bath_length = 50
phys_dim = 15
pd_boson = [phys_dim]*bath_length

g = 300
temp = 0.1
j = lambda w: 10 * drude(w, 100, 5)*temp_factor(temp,w)

eth = SpinBosonModel(v_x=1, v_z=0, pd_spin=2,
                    pd_boson=pd_boson, boson_domain=[0, 300],
                    sd=j, dt=0.001/4)

print("eth.w_list", eth.w_list)
print("eth.k_list", eth.k_list)
exit()
etn = SpinBosonMPS(pd_spin=2, pd_boson=pd_boson)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0
etn.B[-1][0, 0, 0] = 1

p = []

bond_dim = 100000
threshold = 1e-4
dt = 0.001 / 4
num_steps = 100

s_dim = np.empty([0,0])

t = 0.
for n in range(num_steps):
    U1, U2 = eth.get_u(2*n*dt, dt)

    t0 = time()
    etn.U = U1
    for j in range(bath_length-1,0,-1):
        print("j==", j, n)
        etn.update_bond(j, bond_dim, threshold, swap=1)

    etn.update_bond(0, bond_dim, threshold, swap=0)
    etn.update_bond(0, bond_dim, threshold, swap=0)
    t1 = time()
    t = t + t1 - t0

    U1, U2 = eth.get_u((2*n+1) * dt, dt)

    etn.U = U2
    for j in range(1, bath_length):
        print("j==", j, n)
        etn.update_bond(j, bond_dim, threshold,swap=1)

    dim = [len(s) for s in etn.S]
    s_dim = np.append(s_dim, dim)

    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.real(rho[0, 0])]



pop = [x for x in p]
print("population", pop)
print(t)

import matplotlib.pyplot as plt
plt.plot(pop)
plt.show()
