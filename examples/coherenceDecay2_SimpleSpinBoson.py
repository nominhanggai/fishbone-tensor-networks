from fishbonett.model import SpinBoson, kron, _c
from fishbonett.fishbone import SpinBoson1D
import numpy as np
from numpy import exp, tanh, pi
from fishbonett.stuff import sd_zero_temp, drude1,sigma_z, sigma_x, temp_factor
from time import time

bath_length = 50
a = [50] * bath_length

pd = a + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
etn.B[-1][0, 1, 0] = 1/np.sqrt(2)
etn.B[-1][0, 0, 0] = 1/np.sqrt(2)


eth.he_dy = (sigma_z + np.eye(2))/2
eth.h1e = 50*sigma_z + 20 *sigma_x

eth.domain = [-350, 350]
temp = 300.
eth.sd = lambda w: sd_zero_temp(w) * temp_factor(300, w)

eth.build(g=350)
etn.U = eth.get_u(dt=0.001)
p = []

s_dim = np.empty([0,0])

t0 = time()
for tn in range(80):
    print("ni complete", tn)
    for j in range(0, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, 300, 1e-5)

    dim = [len(s) for s in etn.S]
    s_dim = np.append(s_dim, dim)

    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 1])]
t1 = time()
print("population", [np.abs(x) for x in p])
print(t1-t0)

s_dim.astype('float32').tofile('heatmap_chain.dat')