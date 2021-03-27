import numpy as np
from scipy.optimize import curve_fit
from fishbonett.starSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp
from scipy.linalg import expm
bath_length = 50
phys_dim = 50
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

a = [phys_dim]*bath_length
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 1. / np.sqrt(2)
etn.B[-1][0, 0, 0] = 1. / np.sqrt(2)


# spectral density parameters
g = 350
eth.domain = [-g-1, g]
temp = 0.0005
j = lambda w: sd_zero_temp(w)*temp_factor(300,w)

eth.sd = j

eth.he_dy = (np.eye(2) + sigma_z)/2
eth.h1e = 50*sigma_z + 20*sigma_x

eth.build(g=350., ncap=20000)
print(eth.w_list)
print(eth.k_list)

# U_one = eth.get_u(dt=0.002, t=0.2)

# ~ 0.5 ps ~ 0.1T
p = []

bond_dim = 100000
threshold = 1e-5
dt = 0.0005
num_steps = 200

# exit()
for tn in range(num_steps):
    U1, U2 = eth.get_u(2*tn*dt, dt,mode='normal')
    etn.U = U1
    for j in range(bath_length-1,0,-1):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, swap=1)

    etn.update_bond(0, bond_dim, threshold, swap=0)
    etn.update_bond(0, bond_dim, threshold, swap=0)
    U1, U2 = eth.get_u((2*tn+1) * dt, dt, mode='reverse')
    etn.U = U2

    for j in range(1, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold,swap=1)

    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 1])]

pop = [x for x in p]
print("population", pop)

# print(occu)
