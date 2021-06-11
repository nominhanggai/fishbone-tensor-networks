import numpy as np
from fishbonett.backwardSpinBosonMultiChannel import SpinBoson, SpinBoson1D, calc_U
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer, drude, _num, sigma_1
from time import time

bath_length = 100
phys_dim = 10
bond_dim = 1000
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

a = [phys_dim]*bath_length
print(a)
pd = a[::-1] + [2]
coup_num = [ 0.49946583,  1.22546878,  2.12929005,  3.17079925,  4.31472722,
        5.52287705,  6.7514691 ,  7.95221291,  9.07659417, 10.0820814 ,
       10.93803761, 11.62912121, 12.15513924, 12.52792191, 12.76678814,
       12.89416984, 12.93232207, 12.90134297, 12.81828918, 12.69702648,
       12.548487  , 12.38109993, 12.20125642, 12.01373805, 11.82208104,
       11.62887142, 11.4359771 , 11.24472702, 11.0560475 , 10.87056549,
       10.68868601, 10.51065041, 10.3365798 , 10.16650765, 10.00040403,
        9.83819381,  9.67977015,  9.52500458,  9.37375448,  9.22586859,
        9.08119108,  8.93956451,  8.80083198,  8.6648386 ,  8.53143252,
        8.40046565,  8.27179406,  8.14527817,  8.02078282,  7.89817728,
        7.77733504,  7.6581337 ,  7.54045468,  7.42418299,  7.30920696,
        7.19541788,  7.08270974,  6.9709789 ,  6.86012369,  6.75004417,
        6.6406417 ,  6.53181858,  6.42347773,  6.31552224,  6.207855  ,
        6.10037826,  5.99299313,  5.88559916,  5.77809375,  5.67037155,
        5.56232387,  5.45383791,  5.34479597,  5.23507451,  5.12454308,
        5.01306314,  4.90048659,  4.78665411,  4.67139315,  4.55451552,
        4.43581455,  4.31506145,  4.19200098,  4.0663459 ,  3.93776998,
        3.80589901,  3.67029898,  3.53046037,  3.38577684,  3.23551543,
        3.07877416,  2.91441925,  2.74098911,  2.55654012,  2.3583844 ,
        2.14260975,  1.90310816,  1.62930998,  1.29957465,  0.85169405]
coup_mat = [(sigma_z) *x for x in coup_num]
freq_num = [1.70809408e-01, 5.72195160e-01, 1.20192325e+00, 2.05854578e+00,
       3.14035102e+00, 4.44552631e+00, 5.97238801e+00, 7.71960955e+00,
       9.68637975e+00, 1.18724294e+01, 1.42779011e+01, 1.69030973e+01,
       1.97481831e+01, 2.28129291e+01, 2.60965434e+01, 2.95976021e+01,
       3.33140555e+01, 3.72432802e+01, 4.13821486e+01, 4.57270991e+01,
       5.02741984e+01, 5.50191921e+01, 5.99575448e+01, 6.50844707e+01,
       7.03949578e+01, 7.58837854e+01, 8.15455394e+01, 8.73746235e+01,
       9.33652694e+01, 9.95115455e+01, 1.05807365e+02, 1.12246491e+02,
       1.18822549e+02, 1.25529026e+02, 1.32359285e+02, 1.39306565e+02,
       1.46363993e+02, 1.53524588e+02, 1.60781267e+02, 1.68126856e+02,
       1.75554091e+02, 1.83055633e+02, 1.90624066e+02, 1.98251912e+02,
       2.05931633e+02, 2.13655643e+02, 2.21416311e+02, 2.29205971e+02,
       2.37016930e+02, 2.44841472e+02, 2.52671871e+02, 2.60500394e+02,
       2.68319310e+02, 2.76120899e+02, 2.83897457e+02, 2.91641306e+02,
       2.99344801e+02, 3.07000336e+02, 3.14600353e+02, 3.22137349e+02,
       3.29603883e+02, 3.36992584e+02, 3.44296159e+02, 3.51507398e+02,
       3.58619182e+02, 3.65624490e+02, 3.72516409e+02, 3.79288134e+02,
       3.85932982e+02, 3.92444393e+02, 3.98815941e+02, 4.05041335e+02,
       4.11114432e+02, 4.17029237e+02, 4.22779911e+02, 4.28360779e+02,
       4.33766332e+02, 4.38991235e+02, 4.44030331e+02, 4.48878645e+02,
       4.53531393e+02, 4.57983982e+02, 4.62232018e+02, 4.66271308e+02,
       4.70097864e+02, 4.73707911e+02, 4.77097885e+02, 4.80264440e+02,
       4.83204452e+02, 4.85915017e+02, 4.88393461e+02, 4.90637339e+02,
       4.92644434e+02, 4.94412767e+02, 4.95940592e+02, 4.97226402e+02,
       4.98268929e+02, 4.99067148e+02, 4.99620285e+02, 4.99927919e+02]

temp = 100
eth = SpinBoson(pd, coup_mat=coup_mat, freq=freq_num, temp=temp)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.


# spectral density parameters

eth.h1e =78.53981499999999*sigma_x

eth.build(n=0)
# exit()
# print(eth.w_list,eth.k_list)
#
# print(len(eth.w_list))
# exit()

# b = np.array([np.abs(eth.get_dk(t=i*0.2/100)) for i in range(100)])
# print(b.shape)
# bj, freq, coef = eth.get_dk(1, star=True)
# coef = eth.get_dk(1, star=True)
# indexes = np.abs(freq).argsort()
# bj = bj[indexes]
# bj = np.array(bj)
# print(b.shape)
# b.astype('float32').tofile('./DA2/dk.dat')
# bj.astype('float32').tofile('./output/j0.dat')
# freq.astype('float32').tofile('./output/freq.dat')
# coef.astype('float32').tofile('./DA2/coef.dat')

# print(freq)
# print(coef)
# exit()


# U_one = eth.get_u(dt=0.002, t=0.2)

# ~ 0.5 ps ~ 0.1T
p = []


threshold = 1e-3
dt = 0.001
num_steps = 50

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