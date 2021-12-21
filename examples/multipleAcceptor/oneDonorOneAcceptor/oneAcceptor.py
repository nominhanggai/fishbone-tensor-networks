import numpy as np
from scipy.optimize import curve_fit
# from fishbonett.starSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.backwardSpinBoson import SpinBoson, SpinBoson1D, calc_U
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer, drude, _num, sigma_1
from scipy.linalg import expm
from time import time
import sys

ene = 0
temp = 0.001
gam = 3900
coupling = 120


bath_length =int(200*5*1.5)
phys_dim = 60
bond_dim = 1000
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

a = [phys_dim]*bath_length
print(a)

pd = a[::-1] + [3]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:

etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.


# spectral density parameters
g = 6000
eth.domain = [-g, g]
j = lambda w: drude(w, lam=3952.11670, gam=gam)* temp_factor(temp,w)
eth.sd = j

eth.he_dy = np.diag([1, -1, -1])/2
eth.h1e =  np.diag([ene, 0, 0]) + np.array([
    [0,        coupling, coupling],
    [coupling, 0,        500],
    [coupling, 500,        0]
    ])

eth.build(g=1., ncap=50000)

print(eth.w_list)
print(eth.k_list)

# 0.5 ps ~ 0.1T
p1 = []
p2 = []
p3 = []

threshold = 1e-3
dt = 0.005/10
num_steps = 100*2 # Due to 2nd order Trotter, actual time is dt*2*num_steps

t = 0.
tt0=time()
for tn in range(num_steps):
    U1, U2 = eth.get_u(2*tn*dt, 2*dt, factor=2)

    t0 = time()
    etn.U = U1
    for j in range(bath_length-1,0,-1):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, swap=1)

    etn.update_bond(0, bond_dim, threshold, swap=0)
    etn.update_bond(0, bond_dim, threshold, swap=0)
    t1 = time()
    t = t + t1 - t0

    t0 = time()
    etn.U = U2
    for j in range(1, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold,swap=1)

    dim = [len(s) for s in etn.S]
    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    pop1 = np.abs(rho[0,0])
    pop2 = np.abs(rho[1,1])
    pop3 = np.abs(rho[2,2])
    p1 = p1 + [pop1]
    p2 = p2 + [pop2]
    p3 = p3 + [pop3]
    t1 = time()
    t = t + t1 - t0
tt1 = time()
print(tt1-tt0)
pop1 = np.array([x.real for x in p1])
pop2 = np.array([x.real for x in p2])
pop3 = np.array([x.real for x in p3])
print("p1\n", p1, "p2\n", p2, "p3\n", p3)
pop1.astype('float32').tofile(f'./output/pop1_sigmaz_{coupling}_{temp}_{ene}_{gam}.dat')
pop2.astype('float32').tofile(f'./output/pop2_sigmaz_{coupling}_{temp}_{ene}_{gam}.dat')
pop3.astype('float32').tofile(f'./output/pop3_sigmaz_{coupling}_{temp}_{ene}_{gam}.dat')