from fishbonett.model import FishBoneH, kron, _c
from fishbonett.fishbone import init, FishBoneNet
import numpy as np
from numpy import pi
from fishbonett.stuff import sz, sx, temp_factor



def init_special(pd):
    """
    Initialize the SimpleTTN class.
    """
    def g_state(dim):
        tensor = np.zeros(dim)
        tensor[(0,)*len(dim)] = 1.
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
    main_s[1][0] = np.array([1/np.sqrt(2),1/np.sqrt(2)])

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


bath_length = 120
a = [20]*bath_length
b = [2]
c = [4]
pd = np.array([[a, b, [], []], [a, b, [], []]], dtype=object)

eth = FishBoneH(pd)
etn = init_special(pd)



'''
Spectral Density Parameters
'''
eth.domain = [-350, 350]
S1 = 0.39; S2 = 0.23; S3 = 0.23
s1 = 0.4; s2 = 0.25; s3 = 0.2
w1 = 26; w2 = 51; w3 = 85
temp = 300.
def sd_back(Sk, sk, w, wk):
    return Sk/(sk*np.sqrt(2/pi)) * w * \
    np.exp(-np.log(np.abs(w)/wk)**2 / (2*sk**2))

gamma = 5.
Omega_1 = 181; Omgea_2 = 221; Omgea_3 = 240
g1 = 0.0173; g2 = 0.0246; g3 = 0.0182


def sd_high(gamma_m, Omega_m, g_m, w):
    return 4*gamma_m*Omega_m*g_m*(Omega_m**2+gamma_m**2)*w\
    / ((gamma_m**2+(w+Omega_m)**2)*(gamma_m**2+(w-Omega_m)**2))


# zero-temperature spectral density
def sd_zero_temp(w):
    return sd_back(S1,s1,w, w1)+sd_back(S2,s2,w,w2)\
    +sd_back(S3,s3,w,w3) + sd_high(gamma, Omega_1, g1, w)\
    + sd_high(gamma, Omgea_2, g2, w) \
    + sd_high(gamma, Omgea_3, g3, w)


# set the spectral densities on the two e-b bath chain.
eth.sd[0, 0] = lambda w: sd_zero_temp(w) * temp_factor(temp,w)
eth.sd[1, 0] = lambda w: sd_zero_temp(w) * temp_factor(temp,w)

'''
Hamiltonians that are needed to be assigned
'''
eth.he_dy = [(np.eye(2) + sz()) / 2 for i in range(2)]

eth.h1e = [0. * sz() for i in range(2)]
lam = 69
annih = _c(*b)
a = lam * (kron(annih, annih.T) + kron(annih.T, annih))
eth.h2ee = [a for i in range(1)]


'''
Build the system and get evolution operators
'''
eth.build(g=350)
etn.U = eth.get_u(dt=0.0001)


p = []
'''
Transformation matrix to turn the density matrix
 to the a basis where H_sys is diagonal.
'''
tran_mat = np.array([[0,-1/np.sqrt(2),1/np.sqrt(2),0],[0,1/np.sqrt(2),1/np.sqrt(2),0],[0,0,0,1],[1,0,0,0]])

'''
Run the evolution
'''
for tn in range(2000):
    print("Step Number:", tn)
    for n in range(etn._nc - 1):
        print("Update Electronic Site ", n)
        print([x.shape for x in etn.ttnB[0]])
        print([x.shape for x in etn.ttnB[1]])
        etn.update_bond(-1, n, 10, 1e-5)
        print("Site %s Completes" % n)
    #
    for n in range(0, 2):
        for j in range(0, bath_length):
            print("Update Bond %s on Chain %s" % (j, n))
            etn.update_bond(n, j, 20, 1e-15)
            print([x.shape for x in etn.ttnB[n][:bath_length+1]])
            print([x.shape for x in etn.ttnS[n][:bath_length+1]])
            print([x.shape for x in etn.U[n]][:bath_length+1])
    t = etn.get_theta2(-1,0)
    c = np.einsum('LIURlJDr,LiURljDr->IJij', t, t.conj())
    # t.shape is {vL i vU vR; VL' j vD' vR'}
    c = c.reshape(4,4)
    population = tran_mat@c@tran_mat.T
    p.append(population[1,1])

'''
Output the population on state |+D><+D|
'''
print("population", [np.abs(x) for x in p])
print("population", [np.abs(x) for x in p[::5]])

