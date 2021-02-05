from fishbonett.model import FishBoneH, kron, _c
from fishbonett.fishbone import init, FishBoneNet
import numpy as np
from numpy import pi
from fishbonett.stuff import sigma_z, sigma_x, sigma_1, sigma_p, sigma_m, temp_factor, drude1
from opt_einsum import contract as einsum

def init_special(pd):
    """
    Initialize the FishBoneNet class.
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

    main_s = [[np.ones([1], np.float)] for chain_n in vb_tensor]
    #main_s[1][0] = np.array([1.])

    e_tensor[0] = [g_state([1, d, 1, 1, 1]) for d in pd[:, 1][0]]
    #e_tensor[1] = [g_state([1, d, 1, 1, 1]) for d in pd[:, 1][1]]
    e_tensor[0][0][0, 0, 0, 0, 0] = 0
    e_tensor[0][0][0, 1, 0, 0, 0] = 0
    e_tensor[0][0][0, 2, 0, 0, 0] = 1./np.sqrt(2)
    e_tensor[0][0][0, 3, 0, 0, 0] = 1./np.sqrt(2)

    #e_tensor[1][0][0, 0, 0, 0, 0] = 0
    #e_tensor[1][0][0, 1, 0, 0, 0] = 1

    vb_s_and_main_s = [vb_s[i] + main_s[i] for i in range(nc)]
    return FishBoneNet(
        (eb_tensor, e_tensor, v_tensor, vb_tensor),
        (eb_s, e_s, v_s, vb_s_and_main_s)
    )


bath_length = 100
phys_dim = 90
a = [int(np.ceil(phys_dim - (phys_dim - 2) * (N/bath_length)**0.2)) for N in range(bath_length)]
a = a[::-1]
print(a)
# a = [phys_dim] * bath_length
b = [4]
pd = np.array([[a, b, [], a[::-1]],], dtype=object)

eth = FishBoneH(pd)
etn = init_special(pd)

'''
Spectral Density Parameters
'''
g=350
eth.domain = [-g, g]
temp = 300.
reorg = 500.
reorg2 = reorg
# set the spectral densities on the two e-b bath chain.
eth.sd[0, 0] = lambda w: drude1(w, reorg) * temp_factor(temp, w)
eth.sd[0, 1] = lambda w: drude1(w, reorg2) * temp_factor(temp, w)

'''
Hamiltonians that are needed to be assigned
'''
h1e_mat = np.array([[  0,   10,  150,    0],
                    [ 10,  400,    0,  150],
                    [150,    0, 1000,   10],
                    [  0,  150,   10, 1400]])

eth.h1e = [h1e_mat]
eth.h1v = [h1e_mat]

he_dy = np.diag([1.0, 1.0, 0.0, 0.0])
hv_dy = np.diag([0.0, 0.0, 1.0, 1.0])

eth.hv_dy = [hv_dy,]
eth.he_dy = [he_dy,]

'''
Build the system and get evolution operators
'''
eth.build(g, ncap=20000)
print(eth.w_list)
print(eth.k_list)
# exit()
time_step = 0.001
U_one = eth.get_u(dt=time_step)
U_half = eth.get_u(dt=time_step/2.)
etn.U = U_half
'''
Run the evolution
'''
label1 = [(0, x) for x in range(2*bath_length)]
#label2 = [(1, bath_length-1-x) for x in range(bath_length)]
label = label1# + [(-1,0)] + label2
label_odd = label[0::2]
label_even = label[1::2]
p = []
bond_dim = 300
threshold = 1e-2
num_steps = 200

for tn in range(num_steps):
    for idx in label_odd:
        print("Step Number:", tn, "Bond", idx)
        etn.update_bond(*idx, bond_dim, threshold)
    etn.U = U_one
    for idx in label_even:
        print("Step Number:", tn, "Bond", idx)
        etn.update_bond(*idx, bond_dim, threshold)
    etn.U = U_half
    for idx in label_odd:
        print("Step Number:", tn, "Bond", idx)
        etn.update_bond(*idx, bond_dim, threshold)
    t = etn.get_theta1(0, bath_length)
    c = einsum('LiUDR,LjUDR->ij', t, t.conj())
    # t.shape is {vL i vU vR; VL' j vD' vR'}
    c = c.reshape(4, 4)
    p.append(np.diagonal(c))
    p0 = pd[0][0]
    p1 = pd[0][3]
    for i, d in enumerate(p0):
        th = etn.get_theta1(0,i)
        rh = einsum('LiR,LjR->ij', th.conj(), th)
        c = _c(d)
        print("Occu", 0, i, d, np.abs(np.trace(c.T@c@rh)))
    for i, d in enumerate(p1):
        th = etn.get_theta1(0,bath_length+1+i)
        rh = einsum('LiR,LjR->ij', th.conj(), th)
        c = _c(d)
        print("Occu", 1, i, d, np.abs(np.trace(c.T@c@rh)))
'''
Output the population on state |+D><+D|
'''
print("pop1={", ','.join(map(str, [np.abs(x[0]) for x in p])), "};")
print("pop2={", ','.join(map(str, [np.abs(x[1]) for x in p])), "};")
print("pop3={", ','.join(map(str, [np.abs(x[2]) for x in p])), "};")
print("pop4={", ','.join(map(str, [np.abs(x[3]) for x in p])), "};")

