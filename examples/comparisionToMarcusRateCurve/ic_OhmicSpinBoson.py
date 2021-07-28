import numpy as np
from fishbonett.backwardSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, entang
from time import time

bath_length = 200
phys_dim = 10
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

a = [phys_dim]*bath_length
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0
etn.B[-1][0, 0, 0] = 1


# spectral density parameters
g = 3
eth.domain = [-g, g]

def ohmic(omega, alpha, omega_c):
    return 1/2 * np.pi * alpha * omega * np.exp(-np.abs(omega)/omega_c)

# omega_c = delta = 1; Reorg E (namely, Gamma) = 2*alpha*omega_c = 2*alpha
# Gamma := 10 * delta = 10; thus, alpha = 5
# kb*T = 10*delta -> delta = 10/kb = 10/0.695034800911
temp = 1/0.6950348009119888
j = lambda w: ohmic(w, alpha=5, omega_c=1)*temp_factor(temp,w)

eth.sd = j

eth.he_dy = sigma_z
# Gamma = 10 * delta = 10; e = a factor * Gamma
Gamma = 10
eth.h1e = 0.*sigma_z + 0.5*sigma_x

eth.build(g=1, ncap=50000)

j,f,_,_ = eth.get_dk(t=0, star=True)

print(repr(j))
print(repr(f))
# exit()

p = []

bond_dim = 100000
threshold = 1e-3
dt = 0.05
num_steps = 200

s_dim = np.empty([0,0])

t = 0.
for tn in range(num_steps):
    U1, U2 = eth.get_u(2*tn*dt, dt,mode='normal')

    t0 = time()
    etn.U = U1
    for j in range(bath_length-1,0,-1):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold, swap=1)

    etn.update_bond(0, bond_dim, threshold, swap=0)
    etn.update_bond(0, bond_dim, threshold, swap=0)
    t1 = time()
    t = t + t1 - t0

    U1, U2 = eth.get_u((2*tn+1) * dt, dt, mode='reverse')

    t0 = time()
    etn.U = U2
    for j in range(1, bath_length):
        print("j==", j, tn)
        etn.update_bond(j, bond_dim, threshold,swap=1)

    dim = [len(s) for s in etn.S]
    s_dim = np.append(s_dim, dim)

    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    p = p + [np.abs(rho[0, 0])]
    t1 = time()
    t = t + t1 - t0

# t1 = time()
pop = [x for x in p]
print("population", pop)
print(t)
# s_dim.astype('float32').tofile('heatmap.dat')
# print(occu)
