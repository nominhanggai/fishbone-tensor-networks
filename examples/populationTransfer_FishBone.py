from fishbonett.model import FishBoneH, kron, _c
from fishbonett.fishbone import init, FishBoneNet
import numpy as np
from numpy import pi
from fishbonett.stuff import sigma_z, sigma_x, sigma_1, temp_factor, sd_zero_temp_prime


def init_special(pd):
    """
    Initialize the SimpleTTN class.
    """

    def g_state(dim):
        tensor = np.zeros(dim)
        tensor[(0,) * len(dim)] = 1.
        return tensor

    nc = len(pd)
    length_chain = [len(sum(chain_n, [])) for chain_n in pd]
    eb_tensor = [
        [g_state([1, d, 1]) for d in pd[:, 0][i]]
        for i in range(nc)]
    e_tensor = [
        [g_state([1, d, 1, 1, 1]) for d in pd[:, 1][i]]
        for i in range(nc)]
    v_tensor = [
        [g_state([1, d, 1]) for d in pd[:, 2][i]]
        for i in range(nc)]
    vb_tensor = [
        [g_state([1, d, 1]) for d in pd[:, 3][i]]
        for i in range(nc)]
    eb_s = [
        [np.ones([1], np.float) for b in chain_n]
        for chain_n in eb_tensor
    ]
    e_s = [
        [np.ones([1], np.float) for b in chain_n]
        for chain_n in e_tensor
    ]
    v_s = [
        [np.ones([1], np.float) for b in chain_n]
        for chain_n in v_tensor
    ]
    vb_s = [
        [np.ones([1], np.float) for b in chain_n]
        for chain_n in vb_tensor
    ]
    # main_s = [[np.ones([1], np.float)] for chain_n in vb_tensor]
    # e_tensor[0][0][0,0,0,0,0] = 1/np.sqrt(2)
    # e_tensor[0][0][0,1,0,0,0] = 1/np.sqrt(2)

    main_s = [[np.ones([2], np.float)] for chain_n in vb_tensor]
    main_s[1][0] = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

    e_tensor[0] = [g_state([1, d, 1, 2, 1]) for d in pd[:, 1][0]]
    e_tensor[1] = [g_state([1, d, 2, 1, 1]) for d in pd[:, 1][1]]
    e_tensor[0][0][0, 0, 0, 0, 0] = 1
    e_tensor[0][0][0, 0, 0, 1, 0] = 0
    e_tensor[0][0][0, 1, 0, 0, 0] = 0
    e_tensor[0][0][0, 1, 0, 1, 0] = 1

    e_tensor[1][0][0, 0, 0, 0, 0] = 0
    e_tensor[1][0][0, 0, 1, 0, 0] = 1
    e_tensor[1][0][0, 1, 0, 0, 0] = 1
    e_tensor[1][0][0, 1, 1, 0, 0] = 0

    vb_s_and_main_s = [vb_s[i] + main_s[i] for i in range(nc)]
    return FishBoneNet(
        (eb_tensor, e_tensor, v_tensor, vb_tensor),
        (eb_s, e_s, v_s, vb_s_and_main_s)
    )


bath_length = 50
a = [15] * bath_length
b = [2]
c = [4]
pd = np.array([[a, b, [], []], [a, b, [], []]], dtype=object)

eth = FishBoneH(pd)
etn = init_special(pd)

'''
Spectral Density Parameters
'''
eth.domain = [-350, 350]
temp = 77.
# set the spectral densities on the two e-b bath chain.
eth.sd[0, 0] = lambda w: sd_zero_temp_prime(w) * temp_factor(temp, w)
eth.sd[1, 0] = lambda w: sd_zero_temp_prime(w) * temp_factor(temp, w)

'''
Hamiltonians that are needed to be assigned
'''
eth.he_dy = [(sigma_1 + sigma_z) / 2] * 2
eth.h1e = [0. * sigma_z] * 2
lam = 69
annih = _c(*b)
a = lam * (kron(annih, annih.T) + kron(annih.T, annih))
eth.h2ee = [a]

'''
Build the system and get evolution operators
'''
eth.build(g=350)
etn.U = eth.get_u(dt=0.000125)

'''
Transformation matrix to turn the density matrix
 to the a basis where H_sys is diagonal.
'''
tran_mat = np.array([
    [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0],
    [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]])

'''
Run the evolution
'''
p = []
for tn in range(1600):
    print("Step Number:", tn)
    for n in range(etn._nc - 1):
        print("Update Electronic Site ", n)
        print([x.shape for x in etn.ttnB[0]])
        print([x.shape for x in etn.ttnB[1]])
        etn.update_bond(-1, n, 10, 1e-15)
        print("Site %s Completes" % n)
    #
    for n in range(0, 2):
        for j in range(0, bath_length):
            print("Update Bond %s on Chain %s" % (j, n))
            etn.update_bond(n, j, 10, 1e-15)
            print([x.shape for x in etn.ttnB[n][:bath_length + 1]])
            print([x.shape for x in etn.ttnS[n][:bath_length + 1]])
            print([x.shape for x in etn.U[n]][:bath_length + 1])
    t = etn.get_theta2(-1, 0)
    c = np.einsum('LIURlJDr,LiURljDr->IJij', t, t.conj())
    # t.shape is {vL i vU vR; VL' j vD' vR'}
    c = c.reshape(4, 4)
    population = tran_mat @ c @ tran_mat.T
    p.append(population[1, 1])

'''
Output the population on state |+D><+D|
'''
print("population", [np.abs(x) for x in p])
print("population", [np.abs(x) for x in p[::4]])
