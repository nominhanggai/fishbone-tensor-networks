import numpy as np
from scipy.optimize import curve_fit
# from fishbonett.starSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.backwardSpinBoson import SpinBoson, SpinBoson1D, calc_U
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer, drude, _num, sd_back_zero_temp
from scipy.linalg import expm
from time import time

bath_length = 200
phys_dim = 20
bond_dim = 1000
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

a = [phys_dim]*bath_length
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.


# spectral density parameters
g = 500
eth.domain = [-g, g]
temp = 226.00253972894595*0.5*1
# temp = 0.1
j = lambda w: drude(w, lam=4.0*78.53981499999999/2, gam=0.25*4*19.634953749999998)* temp_factor(temp,w)
# j = lambda w: sd_back_zero_temp(w)
eth.domain = [-g,g]
# j = lambda w: 0
eth.sd = j

eth.he_dy = sigma_z + np.diag([1,1])
eth.h1e =  78.53981499999999 * sigma_x + np.diag([78.53981499999999/2 * 2, 0])

eth.build(g=1., ncap=20000)
# print(eth.w_list,eth.k_list)
#
# print(len(eth.w_list))
# exit()

b = np.array([np.abs(eth.get_dk(t=i*0.2/100)) for i in range(101)])
j0, freq, coef, reorg = eth.get_dk(1, star=True)
# indexes = np.abs(freq).argsort()
print(repr(j0))
print(f"Reorg={reorg}\n Freq={repr(freq)}")

# exit()
# bj = np.array(bj)
# print(b.shape)
b.astype('float32').tofile('./output/dk.dat')
# bj.astype('float32').tofile('./output/j0.dat')
# freq.astype('float32').tofile('./output/freq.dat')
# coef.astype('float32').tofile('./output/coef.dat')

# print(repr(freq))
# print(repr(bj))
exit()

print(eth.w_list)
print(eth.k_list)


# U_one = eth.get_u(dt=0.002, t=0.2)

# ~ 0.5 ps ~ 0.1T
p = []


threshold = 1e-3
dt = 0.001
num_steps = 100

s_dim = np.empty([0,0])
num_l = np.empty([0,0])
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
    s_dim = np.append(s_dim, dim)
    print("Length", len(dim))
    theta = etn.get_theta1(bath_length) # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij',  theta, theta.conj())
    sigma_z= np.diag([1,0])

    pop = np.einsum('ij,ji', rho, sigma_z)
    p = p + [pop]
    t1 = time()
    t = t + t1 - t0
    numExp = []
    for i, pd in enumerate(a[::-1]):
        theta = etn.get_theta1(i)
        rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
        numExp.append(np.einsum('ij,ji', rho, _num(pd)).real)
    num_l = np.append(num_l, numExp)
tt1 = time()
print(tt1-tt0)
pop = [x.real for x in p]
print("population", pop)
pop = np.array(pop)

s_dim.astype('float32').tofile('./output/dim.dat')
pop.astype('float32').tofile('./output/pop.dat')
num_l.astype('float32').tofile('./output/num_ic.dat')