import numpy as np
from scipy.optimize import curve_fit
# from fishbonett.starSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.backwardSpinBoson import SpinBoson, SpinBoson1D, calc_U
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer, drude, _num, sigma_1
from scipy.linalg import expm
from time import time

bath_length = 600
phys_dim = 20
bond_dim = 1000
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

coup_num = [5.103, 54.005, 7.611, 59.351, 17.864, 78.386, 1.6, 23.205, 16.388, 16.832, 34.04, 59.306, 135.298, 35.683, 49.282, 83.13, 248.267, 7.459, 31.089, 97.948, 86.104, 194.995, 12.813, 224.426, 84.233, 205.916, 7.479, 31.836, 72.212, 158.4, 153.566, 376.591, 453.119, 58.267, 13.659, 56.343, 88.513, 5.511, 2.825, 6.848, 42.189, 11.75, 19.41, 158.066, 87.294, 122.193, 78.661, 50.207, 99.224, 40.341, 64.046, 130.851, 277.49, 32.089, 2.373, 93.805, 87.105, 125.586, 29.542, 103.785, 43.961, 84.533, 20.919, 82.413, 80.448, 44.96, 4.334, 6.546, 7.769, 6.677, 19.429, 4.819, 10.471, 17.463, 1.474, 17.124, 7.595, 0.796, 52.768, 39.948, 33.722, 1.597, 407.67, 71.316, 81.102, 37.91, 193.652, 16.098, 29.362, 24.827, 15.834, 14.593, 0.328, 407.779, 51.496, 3.874, 10.804, 0.831, 393.909, 145.339, 17.925, 103.809, 9.802, 36.976, 38.245, 2.315, 185.475, 40.804, 60.391, 49.15, 11.133, 63.593, 46.785, 73.749, 23.553, 65.253, 50.642, 30.93, 29.445, 55.732, 33.552, 965.151, 14.19, 10.854, 133.846, 13.823, 8.67, 27.031, 117.27, 414.43, 44.316, 1.795, 91.859, 349.608, 83.052, 13.682, 58.041, 147.072, 99.379, 169.014, 79.748, 2864.631, 10.599, 15.745, 19.275, 0.48, 11.785, 47.961, 44.118, 2.755, 13.626, 20.029, 31.7, 109.328, 19.359, 93.529, 81.067, 17.263, 69.647, 178.795, 64.816, 102.892]
freq_num = [18.047, 21.543, 37.121, 53.402, 60.489, 64.442, 80.308, 90.016, 113.04, 137.839, 144.531, 150.438, 163.548, 201.099, 218.22, 238.922, 281.282, 287.493, 295.624, 298.284, 323.893, 327.729, 377.7, 389.886, 397.564, 408.625, 418.221, 423.49, 454.366, 462.786, 471.654, 486.285, 493.924, 512.441, 527.688, 528.911, 536.044, 556.463, 578.389, 603.31, 605.694, 618.301, 632.331, 641.304, 650.712, 664.59, 674.474, 681.808, 697.056, 716.841, 745.758, 747.015, 748.319, 753.33, 777.977, 787.958, 792.117, 806.072, 811.519, 816.248, 832.124, 837.864, 838.562, 864.571, 872.062, 881.844, 887.426, 892.306, 901.819, 916.532, 920.784, 935.997, 942.429, 946.205, 946.662, 967.232, 968.231, 971.044, 978.33, 999.483, 1003.139, 1007.525, 1030.756, 1034.887, 1042.406, 1063.621, 1078.323, 1082.339, 1116.534, 1121.943, 1132.165, 1143.173, 1154.187, 1165.505, 1170.768, 1181.668, 1182.776, 1185.178, 1198.898, 1207.558, 1213.498, 1221.43, 1227.41, 1230.487, 1241.109, 1243.256, 1248.595, 1286.482, 1297.547, 1302.264, 1319.457, 1324.565, 1339.317, 1343.81, 1355.933, 1360.833, 1379.003, 1385.683, 1388.777, 1425.984, 1444.867, 1451.489, 1467.655, 1480.439, 1490.666, 1515.425, 1516.366, 1521.207, 1536.914, 1561.192, 1564.267, 1608.199, 1630.784, 1634.824, 1640.404, 1665.501, 1677.601, 1686.948, 1688.9, 1717.004, 1755.824, 2001.976, 3163.543, 3164.538, 3184.485, 3191.82, 3192.683, 3193.857, 3196.629, 3197.507, 3202.716, 3204.097, 3204.64, 3206.71, 3206.863, 3214.316, 3218.528, 3219.568, 3219.878, 3223.557, 3234.58, 3235.637]
low_freq_num = 0
freq_num = freq_num[low_freq_num:]
coup_num = coup_num[low_freq_num:]

a = [phys_dim]*bath_length
print(a)

pd = a[::-1] + [2]
eth = SpinBoson(pd)
etn = SpinBoson1D(pd)
# set the initial state of the system. It's in the high-energy state |0>:

etn.B[-1][0, 1, 0] = 0.
etn.B[-1][0, 0, 0] = 1.


# spectral density parameters
g = 3500
eth.domain = [-g, g]
temp = 226.00253972894595*0.5*1
temp = 5.5

def lorentzian (w, wi, delta, ki):
    return np.pi * (ki**2/np.pi/wi) * w * delta/((w-wi)**2 + delta**2)

j = lambda w: sum([lorentzian(w, wi=freq_num[i], delta=107, ki=coup_num[i]) for i in range(len(freq_num))])*temp_factor(temp,w)

j_list = np.array([j(w) for w in range(1, 3000,30)])
print(repr(j_list))

eth.sd = j

eth.he_dy = (sigma_z - sigma_1)/2
eth.h1e =  (4339.26283  - 3968.24779) * (sigma_z - sigma_1)/2 + 120.02659*sigma_x
eth.h1e =  (9678.65315 - 3968.24779) * (sigma_z - sigma_1)/2 + 120.02659*sigma_x

eth.build(g=1., ncap=20000)
# print(eth.w_list,eth.k_list)
#
# print(len(eth.w_list))
# exit()

# b = np.array([np.abs(eth.get_dk(t=i*0.2/100)) for i in range(1)])
# bj, freq, coef = eth.get_dk(1, star=True)
# indexes = np.abs(freq).argsort()
# bj = bj[indexes]
# bj = np.array(bj)
# print(b.shape)
# b.astype('float32').tofile('./DA2/dk.dat')
# bj.astype('float32').tofile('./output/j0.dat')
# freq.astype('float32').tofile('./output/freq.dat')
# coef.astype('float32').tofile('./DA2/coef.dat')
# print(coef.shape)
# print(repr(freq))
# print(repr(bj))


print(eth.w_list)
print(eth.k_list)


# U_one = eth.get_u(dt=0.002, t=0.2)

# ~ 0.5 ps ~ 0.1T
p = []


threshold = 1e-3
dt = 0.001/4
num_steps = 50*4*2

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
    sigma_z= sigma_z

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

s_dim.astype('float32').tofile('./DA2/dim.dat')
pop.astype('float32').tofile('./DA2/pop.dat')
num_l.astype('float32').tofile('./DA2/num_ic.dat')