import numpy as np
from scipy.optimize import curve_fit
# from fishbonett.starSpinBoson import SpinBoson, SpinBoson1D
from fishbonett.backwardSpinBoson import SpinBoson
from fishbonett.spinBosonMPS import SpinBoson1D
from fishbonett.stuff import sigma_x, sigma_z, temp_factor, sd_zero_temp, drude1, lemmer, drude, _num, sigma_1
from scipy.linalg import expm
from time import time

bath_length = 800
phys_dim = 20
bond_dim = 1000
a = [np.ceil(phys_dim - N*(phys_dim -2)/ bath_length) for N in range(bath_length)]
a = [int(x) for x in a]

coup_num = [177.223, 8.465, 11.719, 16.929, 26.246, 82.77, 0.193, 0.016, 234.837, 0.14, 15.341, 94.823, 3.374, 0.656, 5.795, 3.186, 13943.097, 2252.48, 750.593, 1732.399, 11418.118, 381.004, 38293.95, 18687.229, 442.687, 360942.75, 1334.463, 0.077, 996.192, 12968.729, 5148.941, 1750.931, 8454.424, 8478.676, 550.107, 31206.323, 90.224, 10.639, 1.008, 5512.685, 835.419, 434.305, 2919.086, 5746.135, 7757.503, 13468.908, 17416.241, 1827.333, 12869.229, 51170.118, 14636.216, 0.523, 6658.904, 791.706, 1449.708, 7051.045, 2402.582, 45156.231, 7845.263, 73295.076, 27038.924, 29628.411, 134.315, 13011.989, 15.133, 71.779, 251.451, 158.121, 3469.65, 532.906, 306.94, 61.836, 757.089, 10.144, 13.817, 347.199, 432.015, 538.34, 3861.645, 28.362, 3.137, 0.175, 7172.315, 673.279, 2.093, 1553.537, 122470.794, 64468.39, 10527.437, 4233.898, 4.101, 2296.961, 11143.386, 275.016, 54.896, 220.777, 1454.255, 40.143, 31797.525, 18816.788, 15.873, 3735.445, 1989.035, 5043.123, 1361.661, 36.693, 78356.816, 178.087, 91497.233, 20629.616, 51456.947, 247.109, 47.746, 286.794, 6.588, 3076.962, 130.162, 71374.642, 2518.246, 2.564, 17.986, 83677.104, 59585.166, 9.23, 304.775, 6.327, 1111.168, 951.794, 435.778, 22.128, 12713.872, 62.173, 886.015, 578.308, 503.695, 90.227, 276.224, 37.523, 542.228, 5.706, 0.981, 69567.439, 9670.982, 52572.087, 7914.601, 88.533, 52432.237, 1386.161, 204.142, 28486.946, 23717.057, 518.815, 47180.494, 955.222, 2433.019, 45569.09, 26373.037, 377343.648, 107366.828, 51782.594, 603.659, 105238.935]
freq_num = [6.115, 19.75, 42.785, 53.035, 59.049, 62.948, 83.663, 87.539, 120.173, 131.58, 143.899, 148.496, 159.891, 201.192, 215.966, 236.831, 278.94, 282.06, 294.762, 303.878, 326.674, 342.415, 390.459, 396.117, 403.132, 409.609, 429.697, 449.173, 457.858, 469.744, 475.314, 489.934, 495.985, 514.18, 515.125, 533.989, 556.77, 579.688, 581.172, 603.909, 607.718, 624.123, 646.965, 655.423, 659.333, 665.403, 670.972, 688.064, 696.46, 719.703, 744.377, 763.996, 777.71, 780.284, 785.892, 787.852, 793.612, 811.478, 817.532, 831.652, 838.931, 842.258, 859.346, 873.155, 886.345, 888.753, 889.866, 890.658, 905.02, 920.056, 938.208, 940.357, 948.667, 966.807, 977.952, 995.335, 999.164, 1001.667, 1004.16, 1007.414, 1013.639, 1015.056, 1020.89, 1033.982, 1046.905, 1047.962, 1048.283, 1063.799, 1123.949, 1132.104, 1144.712, 1149.057, 1156.337, 1161.604, 1163.757, 1181.27, 1185.69, 1190.34, 1201.471, 1207.704, 1213.014, 1227.291, 1227.998, 1230.663, 1242.207, 1247.526, 1251.137, 1258.572, 1306.564, 1316.516, 1333.095, 1340.534, 1344.534, 1347.979, 1355.751, 1362.395, 1385.265, 1412.373, 1423.885, 1432.681, 1435.431, 1469.188, 1481.761, 1500.239, 1510.151, 1516.427, 1524.746, 1539.361, 1553.56, 1597.505, 1643.892, 1655.17, 1656.462, 1666.857, 1685.082, 1696.391, 1698.825, 1708.27, 1712.974, 1727.554, 1768.309, 1780.63, 3163.355, 3164.762, 3186.09, 3189.269, 3190.201, 3191.791, 3194.869, 3196.625, 3197.522, 3199.217, 3199.423, 3206.869, 3207.606, 3210.26, 3213.09, 3213.323, 3218.568, 3224.412, 3225.867, 3226.448]
low_freq_num = 0
freq_num = freq_num[low_freq_num:]
coup_num = coup_num[low_freq_num:]

# space = [freq_num[i+1] - freq_num[i] for i in range(len(freq_num)-1)]
# print(sum(space)/len(space))
# print(space)
# exit()


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
temp = 0.0001

def lorentzian (w, wi, delta, ki):
    return np.pi * (ki /wi) * w * delta/np.pi/((np.abs(w)-wi)**2 + delta**2)

j = lambda w: sum([lorentzian(w, wi=freq_num[i], delta=107, ki=coup_num[i]) for i in range(len(freq_num))])*temp_factor(temp,w)



# j_list = np.array([j(w) for w in range(1, 3000,30)])
# print(repr(j_list))

eth.sd = j

eth.he_dy = (sigma_z - sigma_1)/2
eth.h1e =  (4339.26283  - 3968.24779) * (sigma_z - sigma_1)/2 + 120.02659*sigma_x
# eth.h1e =  (9678.65315 - 3968.24779) * (sigma_z - sigma_1)/2 + 120.02659*sigma_x

eth.build(g=1., ncap=20000)
# print(eth.w_list,eth.k_list)
#
# print(len(eth.w_list))
# exit()

# b = np.array([np.abs(eth.get_dk(t=i*0.2/100)) for i in range(1)])
# bj, freq, coef, reorg = eth.get_dk(1, star=True)
# print("Reorg Ene", reorg)

# exit()
# indexes = np.abs(freq).argsort()
# bj = bj
# bj = np.array([freq, bj]).T
# print(bj.shape)
# print(b.shape)
# b.astype('float32').tofile('./DA2/dk.dat')
# bj.astype('float32').tofile('./output/bj_300.dat')
# freq.astype('float32').tofile('./output/freq.dat')
# coef.astype('float32').tofile('./DA2/coef.dat')
# print(coef.shape)

# print(repr(bj))
# exit()


print(eth.w_list)
print(eth.k_list)


# U_one = eth.get_u(dt=0.002, t=0.2)

# ~ 0.5 ps ~ 0.1T
p = []


threshold = 1e-3
dt = 0.001/4
num_steps = 50*4

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