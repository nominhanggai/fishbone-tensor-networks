import numpy as np
from backwardSpinBosonMultiChannel import SpinBoson
from spinBosonMPS_LBO import SpinBoson1D
from bath_discrete import *
from time import time

bath_length = int((101 * 2 -2)/4/4)
bond_dim = 2000
dim = 200
a = [dim] * bath_length
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

delta23 = 50
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

threshold = 1e-4
dt = 0.001/4
num_steps = 50
eps_LBO = 1e-50
t = 0.
tt0 = time()
for tn in range(num_steps):
    U1, U2 = eth.get_u(2*tn*dt, 2*dt, factor=2)

    t0 = time()
    etn.U = U1
    for j in range(bath_length-1,0,-1):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, eps_LBO, swap=1, toarray=False)

    etn.update_bond(0, bond_dim, threshold, eps_LBO, swap=0, toarray=False)
    etn.update_bond(0, bond_dim, threshold, eps_LBO, swap=0, toarray=False)
    t1 = time()
    t = t + t1 - t0

    t0 = time()
#    U1, U2 = eth.get_u((2*tn+1) * dt, 2*dt, factor=2)
    etn.U = U2
    for j in range(1, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, eps_LBO, swap=1, toarray=False)
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
# p1.astype('float32').tofile(f'./output/pop1_{delta23}.dat')
# p2.astype('float32').tofile(f'./output/pop2_{delta23}.dat')

