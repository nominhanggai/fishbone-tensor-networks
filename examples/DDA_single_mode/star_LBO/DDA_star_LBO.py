import numpy as np
from starSpinBosonMultiChannel import SpinBoson
from spinBosonMPS_LBO import SpinBoson1D
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer, drude, _num, sigma_1
from bath_discrete import *
from time import time
from sys import argv

delta23 = 50

bath_length = 101*2
phys_dim = 200
bond_dim = 1000

a = [phys_dim] * bath_length
print(a)
pd = a[::-1] + [3]
freq_num = np.array(w_list)

coup_mat = [np.diag([x, y, z]) for x, y, z in zip(c1_list, c2_list, c3_list)]
reorg1 = sum([(c1_list[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg12 = sum([(c1_list[i]-c2_list[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg2 = sum([(c2_list[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg3 = sum([(c3_list[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
print("Reorg", reorg1, reorg2, reorg3, reorg12)
print(f"Len {len(coup_mat)}")

temp = 300
eth = SpinBoson(pd, coup_mat=coup_mat, freq=freq_num, temp=temp)
etn = SpinBoson1D(pd)

# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.

# spectral density parameters
coupling_mat = np.array([
    [0, 299.99987267649203, 0],
    [299.99987267649203, 0, delta23],
    [0, delta23 , 0]
    ])
eth.h1e = coupling_mat + np.diag([0,1613.1385305,219.47463]) + np.diag([reorg1, reorg2, reorg3])
print("FREQ", eth.freq)

# ~ 0.5 ps ~ 0.1T
p1 = []
p2 = []
threshold = 1e-3
eps_LBO = 1e-13
dt = 0.001 / 4
num_steps = 300

t = 0.
tt0 = time()
U1, U2 = eth.get_u(2 * dt, factor=2)

for tn in range(num_steps):
    t0 = time()
    etn.U = U1
    for j in range(bath_length - 1, 0, -1):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, eps_LBO, swap=1)

    etn.update_bond(0, bond_dim, threshold, eps_LBO, swap=0)
    etn.update_bond(0, bond_dim, threshold, eps_LBO, swap=0)

    etn.U = U2
    for j in range(1, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, eps_LBO, swap=1)

    theta = etn.get_theta1(bath_length)  # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
    pop1 = np.abs(rho[0, 0])
    pop2 = np.abs(rho[1, 1])
    p1 = p1 + [pop1]
    p2 = p2 + [pop2]
    print("population1\n", p1)
    print("population2\n", p2)
    t1 = time()
    t = t + t1 - t0

tt1 = time()
print(tt1 - tt0)
p1 = np.array(p1)
p2 = np.array(p2)
# p1.astype('float32').tofile(f'./output/pop1_{delta23}.dat')
# p2.astype('float32').tofile(f'./output/pop2_{delta23}.dat')