import numpy as np
from fishbonett.starSpinBosonMultiChannel import SpinBoson
from fishbonett.spinBosonMPS import SpinBoson1D
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer, drude, _num, sigma_1
from examples.multipleAcceptor.electronicParametersAndVibronicCouplingDA import freqMol2_LE, freqMol1_GR, coupMol2_LE, coupMol2_CT, coupMol1_CT, coupMol1_LE, coupMol1_LE_CT
from time import time

bath_length = 162 * 2
phys_dim = 20
bond_dim = 1000
# a = [np.ceil(phys_dim - N * (phys_dim - 2) / bath_length) for N in range(bath_length)]
# a = [int(x) for x in a]
a = [phys_dim] * bath_length
print(a)
pd = a[::-1] + [2]
coup_num_LE = np.array(coupMol2_LE)
coup_num_CT = np.array(coupMol2_LE) + np.array(coupMol1_LE_CT)
freq_num = np.array(freqMol2_LE)

coup_num_LE = coup_num_LE * freq_num / np.sqrt(2)  # + list([1.15*x for x in back_coup])
coup_num_CT = coup_num_CT * freq_num / np.sqrt(2)  # + list([-1.15*x for x in back_coup])

coup_mat = [np.diag([x, y]) for x, y in zip(coup_num_LE, coup_num_CT)]
reorg1 = sum([(coup_num_LE[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg2 = sum([(coup_num_CT[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
reorg12 = sum([(coup_num_LE[i]-coup_num_CT[i]) ** 2 / freq_num[i] for i in range(len(freq_num))])
print(f"Reorg: GR->LE {reorg1}, GR->CT {reorg2}, LE-CT {reorg12}")
print(f"Len {len(coup_mat)}")
exit()
temp = 300
eth = SpinBoson(pd, coup_mat=coup_mat, freq=freq_num, temp=temp)
etn = SpinBoson1D(pd)

# set the initial state of the system. It's in the high-energy state |0>:
# if you doubt this, pls check the definition of sigma_z
etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.

# spectral density parameters

eth.h1e = 134.56223 * sigma_x + np.diag([4339.26283, 0]) + np.diag([reorg1, reorg2])

# eth.build(n=0)
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
# coup_num = np.array([[freq_num[n], coup_num[n]] for n in range(len(freq_num))])
# coup_num.astype('float32').tofile('./DA2/coup.dat')
# spacing = [freq_num[i+1]-freq_num[i] for i in range(161)]
# rms = np.sqrt(sum([x**2 for x in spacing])/161)
# print(rms)
# coef.astype('float32').tofile('./DA2/coef.dat')

# print(freq)
# print(coef)
# exit()


# U_one = eth.get_u(dt=0.002, t=0.2)

# ~ 0.5 ps ~ 0.1T
p = []

threshold = 1e-3
dt = 0.001 / 10
num_steps = 400

s_dim = np.empty([0, 0])
num_l = np.empty([0, 0])
t = 0.
tt0 = time()
for tn in range(num_steps):
    U1, U2 = eth.get_u(2 * tn * dt, 2 * dt, factor=2)

    t0 = time()
    etn.U = U1
    for j in range(bath_length - 1, 0, -1):
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
        etn.update_bond(j, bond_dim, threshold, swap=1)

    dim = [len(s) for s in etn.S]
    s_dim = np.append(s_dim, dim)
    print("Length", len(dim))
    theta = etn.get_theta1(bath_length)  # c.shape vL i vR
    rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
    pop = np.abs(rho[0, 0])
    p = p + [pop]
    t1 = time()
    t = t + t1 - t0
tt1 = time()
print(tt1 - tt0)
pop = [x.real for x in p]
print("population", pop)
pop = np.array(pop)
pop.astype('float32').tofile('./output/pop.dat')
# s_dim.astype('float32').tofile('./output/dim.dat')
#
# num_l.astype('float32').tofile('./output/num_ic.dat')
