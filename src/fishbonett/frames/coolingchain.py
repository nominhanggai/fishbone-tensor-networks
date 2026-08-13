import numpy as np
import scipy

from copy import deepcopy as dcopy
from fishbonett.contract import contract as einsum
import fishbonett.recurrence_coefficients as rc
# kron/svd and the sparse two-site gate exponential (calc_U) are shared;
# _c is the bosonic annihilation operator from fishbonett.operators.
from fishbonett.linalg import kron, svd, expm_gate_sparse as calc_U
from fishbonett.operators import temp_factor, _c

from fishbonett.mps import BosonicBathMPS


class BosonicBathCoolingChain(BosonicBathMPS):
    """Cooling-chain builder: system + harmonic bath, dissipative cooling ansatz.

    Extends the 1D :class:`~fishbonett.states.mps.BosonicBathMPS` engine with a
    ``betaOmega`` cooling gauge: each bath mode carries a heating operator so the
    chain is progressively cooled, and :meth:`get_rdm` reads the system reduced
    density matrix through those operators.  Historically named ``SpinBoson``
    (aliases kept).
    """

    def __init__(self, pd, betaOmega=2.):
        super().__init__(pd)
        def g_state(dim):
            tensor = np.zeros(dim)
            tensor[(0,) * len(dim)] = 1.
            return tensor

        self.pd_sys = pd[-1]
        self.pd_boson = pd[0:-1]
        self.B = [g_state([1, d, 1]) for d in pd]
        self.S = [np.ones([1]) for _ in pd]
        self.U = [np.zeros(0) for _ in pd[1:]]
        self.H = [np.zeros(0) for _ in pd[1:]]
        self.betaOmega = betaOmega

        self.pd_sys = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = lambda x: np.heaviside(x, 1) / 1. * np.exp(-x / 1)
        self.domain = [0, 1]
        self.he_dy = np.eye(self.pd_sys)
        self.h1e = np.eye(self.pd_sys)
        self.k_list = []
        self.w_lsit = []
        self.H = []
        self.coef= []
        self.freq = []
        self.phase = lambda lam, t, delta: (np.exp(-1j*lam*(t+delta)) - np.exp(-1j*lam*t))/(-1j*lam)
        self.phase_func = lambda lam, t: np.exp(-1j * lam * (t))



    def get_rdm(self):
        theta = self.get_theta1(0)
        rho = einsum('PiQ,ij,PjL->QL', theta, self.heating_op[0], theta.conj())
        for i in range(1, self.len_boson):
            rho = einsum('PQ, PiK, ij, QjL->KL', rho, self.B[i], self.heating_op[i], self.B[i].conj())
            rho = rho/einsum('KK', rho)
        rho = einsum('PQ,PiL,QjL->ij', rho, self.B[-1], self.B[-1].conj())
        return rho

    def get_coupling(self, n, j, domain, g, ncap=20000):
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        self.domain = domain
        return w_list, k_list

    def build_coupling(self, g, ncap):
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g, ncap)

    def build(self, g, ncap=20000):
        self.build_coupling(g, ncap)
        hee = self.get_h2()
        self.H = hee
        self.heating_op = [scipy.linalg.expm(2 * self.betaOmega # * np.sign(freq[i])
                                             * _c(d).T @ _c(d)) for i, d in
                           enumerate(self.pd_boson)]
        self.heating_op = [op / np.linalg.norm(op) for op in self.heating_op]

    def get_h1(self):
        w_list = self.w_list[::-1]
        h1 = []
        for i, w in enumerate(w_list):
            c = _c(self.pd_boson[i])
            h1.append(w * c.T @ c)
        h1.append(self.h1e)
        return h1


    def get_h2(self):
        h1 = self.get_h1()
        k_list = self.k_list[::-1]
        k0 = k_list[-1]
        k_list = k_list = k_list[0:-1]
        h2 = []
        for i, k in enumerate(k_list):
            d1 = self.pd_boson[i]
            d2 = self.pd_boson[i + 1]
            c1 = _c(d1)
            c2 = _c(d2)
            coup = k * (kron(c1.T, c2) + kron(c1, c2.T))
            site = kron(h1[i], np.eye(d2))
            h2.append((coup + site, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_sys
        annih = np.exp(self.betaOmega)  # *np.sign(self.freq[::-1][i]))
        creat = np.exp(-1 * self.betaOmega)  # *np.sign(self.freq[::-1][i]))
        c0 = _c(d1)
        coup = k0 * kron(annih*c0 + creat*c0.T, self.he_dy)
        site = kron(h1[-2], np.eye(d2)) + kron(np.eye(d1), h1[-1])
        h20 = coup + site
        h2.append((h20, d1, d2))
        return h2


    def get_u(self, dt):
        U = [0]*len(self.H)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h, dt)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # u = u.reshape([r0, s0, r1, s1])
            U[i] = u.toarray().reshape(r0,s0,r1,s1)
        return U


#: ``BosonicBathCoolingChain`` was historically named ``SpinBoson``; both aliases kept.
BosonicBath = BosonicBathCoolingChain
SpinBoson = BosonicBathCoolingChain  # deprecated


if __name__ == '__main__':
    from fishbonett.spectral_densities import drude
    from fishbonett.operators import entang, sigma_z, sigma_x
    from time import time

    bath_length = 200
    phys_dim = 20
    threshold = 1e-4
    coup = 4.0
    bond_dim = 1000
    tmp = 2.0
    bath_freq = 1.0

    pd = [phys_dim] * bath_length + [2]
    bo = 0.2
    etn = BosonicBath(pd=pd, betaOmega=bo)
    g = 500 + bath_freq * 500
    etn.domain = [-g, g]
    temp = 226.00253972894595 * 0.5 * tmp

    j = lambda w: drude(w, lam=coup * 78.53981499999999 / 2, gam=bath_freq * 4 * 19.634953749999998) * temp_factor(temp,w)
    etn.sd = j
    etn.he_dy = sigma_z
    etn.h1e = (78.53981499999999) * sigma_x

    etn.build(g=1, ncap=20000)

    dt = 0.001 / int(np.ceil(bath_freq)) / 10
    num_steps = 100 * int(np.ceil(bath_freq)) * 2

    p = []
    s_dim = np.empty([0,0])
    s_ent = np.empty([0,0])



    u_one = etn.get_u(2 * dt)
    u_half = etn.get_u(dt)

    label = [x for x in range(bath_length)]
    label_odd = label[0::2]
    label_even = label[1::2]
    for tn in range(num_steps):

        t0 = time()
        etn.U = u_half
        for j in label_odd:
            etn.update_bond(j, bond_dim, threshold, swap=0)

        etn.U = u_one
        for j in label_even:
            etn.update_bond(j, bond_dim, threshold, swap=0)
        etn.U = u_half
        for j in label_odd:
            etn.update_bond(j, bond_dim, threshold, swap=0)

        theta = etn.get_theta1(bath_length)  # c.shape vL i vR
        rho = etn.get_rdm()
        # rho = np.einsum('LiR,LjR->ij', theta, theta.conj())
        pop = np.einsum('ij,ji', rho, sigma_z)
        p = p + [pop]

        dim = [len(s) for s in etn.S]
        ent = [entang(s) for s in etn.S]
        s_dim = np.append(s_dim, dim)
        s_ent = np.append(s_ent, ent)

    pop = [x.real for x in p]
    print("population", pop)

    p = np.array(p)
    # p.astype('float32').tofile(f'./output/pop_cooling_BO{bo}.dat')
    # s_dim.astype('float32').tofile(f'./output/sDim_cooling_BO{bo}.dat')
    # s_ent.astype('float32').tofile(f'./output/entropy_cooling_BO{bo}.dat')


