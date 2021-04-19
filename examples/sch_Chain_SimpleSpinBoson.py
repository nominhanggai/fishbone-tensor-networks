from fishbonett.model import SpinBoson
from fishbonett.fishbone import SpinBoson1D
import numpy as np
from numpy import exp, tanh, pi
from fishbonett.stuff import sd_zero_temp, drude1,sigma_z, sigma_x, temp_factor, entang, drude
from time import time

bath_length = 200
phys_dim = 100
bond_dim = 100

a = [phys_dim] * bath_length
a = [np.ceil(phys_dim - (phys_dim-2)*(N/ bath_length)**0.2) for N in range(bath_length)]
a = [int(x) for x in a]

pd = a[::-1] + [2]
print(pd)
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
etn.B[-1][0, 1, 0] = 0
etn.B[-1][0, 0, 0] = 1


eth.he_dy = sigma_z
eth.h1e = 78.53981499999999*sigma_x
g = 10000
eth.domain = [-g, g]
temp = 226.00253972894595
eth.sd = lambda w: drude(w, lam=785.3981499999999/2, gam=19.634953749999998) * temp_factor(temp,w)

eth.build(g=1)
dt = 0.001/8
u_one = eth.get_u(dt=dt)
u_half = eth.get_u(dt=dt/2)
num_steps = 2*4*160

p = []

s_dim = np.empty([0,0])

label = [x for x in range(bath_length)]
label_odd = label[0::2]
label_even = label[1::2]
threshold = 1e-3
t0 = time()
for tn in range(num_steps):
    print("ni complete", tn)
    etn.U = u_half
    for j in label_odd:
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold)
    etn.U = u_one
    for j in label_even:
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold)
    etn.U = u_half
    for j in label_odd:
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, 1e-5)
    dim = [len(s) for s in etn.S]
    s_dim = np.append(s_dim, dim)

    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 0])]
t1 = time()
print("population", [2*np.abs(x)-1 for x in p])
print(t1-t0)

s_dim.astype('float32').tofile('heatmap_chain.dat')