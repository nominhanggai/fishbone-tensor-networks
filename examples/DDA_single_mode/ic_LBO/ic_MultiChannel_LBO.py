from ast import arg
import numpy as np
from fishbonett.backwardSpinBosonMultiChannel import SpinBoson
from spinBosonMPS_LBO import SpinBoson1D
from fishbonett.stuff import sigma_x, _num
from bath_discrete import *
from time import time
from sys import argv

bath_length = 101 * 2
#phys_dim = 100
bond_dim = 2000
num_good_mode = 2 #int(argv[2])
a = [2000]*num_good_mode+ [600]*(bath_length-num_good_mode-100) + [200]*100
print(a)

pd = a[::-1] + [3]
print(pd)
freq_num = np.array(w_list)

coup_mat = [np.diag([x, y, z]) for x, y, z in zip(c1_list, c2_list, c3_list)]
reorg1 = sum([(c1_list[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg2 = sum([(c2_list[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg3 = sum([(c3_list[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
print("Reorg", reorg1, reorg2, reorg3)
print(f"Len {len(coup_mat)}")

temp = 300
eth = SpinBoson(pd, coup_mat=coup_mat, freq=freq_num, temp=temp)
etn = SpinBoson1D(pd)

# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.
# 4339 is the calculeted gap

delta23 = 50 #int(argv[1])
coupling_mat = np.array([
    [0, 299.99987267649203, 0],
    [299.99987267649203, 0, delta23],
    [0, delta23 , 0]
    ])
eth.h1e = coupling_mat + np.diag([0,1613.1385305,219.47463]) + np.diag([reorg1, reorg2, reorg3])
eth.build(n=0)

# ~ 0.5 ps ~ 0.1T
p1 = []
p2 = []
p2 = []

threshold = 1e-3 #float(argv[3])
dt = 0.001/5
num_steps = 450
eps_LBO = 1e-4
gpu = True

t = 0.
tt0 = time()
for tn in range(num_steps):
    U1, U2 = eth.get_u(2*tn*dt, 2*dt, factor=2)

    t0 = time()
    etn.U = U1
    for j in range(bath_length-1,0,-1):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, eps_LBO=eps_LBO, swap=1, gpu=gpu)

    etn.update_bond(0, bond_dim, threshold, eps_LBO=eps_LBO, swap=0, gpu=gpu)
    etn.update_bond(0, bond_dim, threshold, eps_LBO=eps_LBO, swap=0, gpu=gpu)

    t1 = time()
    t = t + t1 - t0

    t0 = time()
#    U1, U2 = eth.get_u((2*tn+1) * dt, 2*dt, factor=2)
    etn.U = U2
    for j in range(1, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, eps_LBO=eps_LBO, swap=1, gpu=gpu)

    # for j in range(0, bath_length):
    #     vib_psi = etn.get_theta1(j)
    #     phys_d = vib_psi.shape[1]
    #     num = _num(phys_d)
    #     num = np.einsum('LiR,ij,LjR', vib_psi, num, vib_psi.conj()).real
    #     print("j==", j, tn, "PHYS_D", phys_d, "NUM", num)

    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())

    pop1 = np.abs(rho[0,0])
    pop2 = np.abs(rho[1,1])
    p1 = p1 + [pop1]
    p2 = p2 + [pop2]
    print("population1\n", p1)
    print("population2\n", p2)
    t1 = time()
    t = t + t1 - t0
tt1 = time()
print(tt1-tt0)

p1 = np.array(p1)
p2 = np.array(p2)
p1.astype('float32').tofile(f'./output/pop1_{delta23}_{num_good_mode}_{threshold}.dat')
p2.astype('float32').tofile(f'./output/pop2_{delta23}_{num_good_mode}_{threshold}.dat')

