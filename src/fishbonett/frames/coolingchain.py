"""Cooling-chain frame: a finite-temperature bath as a *gauged* zero-temperature one.

An alternative to thermofield doubling for finite temperature.  Instead of
mirroring the spectral density onto a signed frequency axis (what
:func:`fishbonett.bath.thermalize` does), each chain mode is reweighted by a
``betaOmega`` gauge: the annihilation and creation halves of the system-bath
coupling are scaled by ``e^{+betaOmega}`` and ``e^{-betaOmega}``, so the chain is
progressively "cooled" along its length while thermal weight is carried by the
non-unitary gauge instead of by extra modes.

The gauge makes the propagation **non-unitary**, so the state's norm is not the
physical one: observables must be read through the matching heating operators
``exp(2 betaOmega n_i)``, which is what :meth:`SystemBathCoolingChain.get_rdm`
does (renormalizing as it contracts).  Reading the RDM the ordinary way would give
the wrong answer.

Unlike the other frames this class *is* the state -- it subclasses
:class:`~fishbonett.states.mps.SystemBathMPS` rather than building gates for a
separate state object.  It is exploratory rather than part of the ``method=``
dispatch; see :class:`~fishbonett.frames.interaction_picture.SystemBathIP` for
the maintained finite-temperature route.
"""
import numpy as np
import scipy

from fishbonett.contract import contract as einsum
import fishbonett.bath.recurrence as rc
from fishbonett.linalg import kron, expm_gate_sparse as calc_U
from fishbonett.operators import temp_factor, annihilate

from fishbonett.states.mps import SystemBathMPS


class SystemBathCoolingChain(SystemBathMPS):
    """Cooling-chain builder: system + harmonic bath, dissipative cooling ansatz.

    Extends the 1D :class:`~fishbonett.states.mps.SystemBathMPS` engine with a
    ``betaOmega`` cooling gauge: each bath mode carries a heating operator so the
    chain is progressively cooled, and :meth:`get_rdm` reads the system reduced
    density matrix through those operators.
    """

    def __init__(self, pd, betaOmega=2.):
        super().__init__(pd)
        def g_state(dim):
            tensor = np.zeros(dim)
            tensor[(0,) * len(dim)] = 1.
            return tensor

        self.pd_sys = pd[0]
        self.pd_boson = pd[1:]
        self.B = [g_state([1, d, 1]) for d in pd]
        self.S = [np.ones([1]) for _ in pd]
        self.U = [np.zeros(0) for _ in pd[1:]]
        self.H = [np.zeros(0) for _ in pd[1:]]
        self.betaOmega = betaOmega

        self.pd_sys = pd[0]
        self.pd_boson = pd[1:]
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
        """System reduced density matrix, read through the heating operators.

        The cooling gauge is non-unitary, so the plain contraction of the MPS
        would not give the physical RDM.  This contracts the bath sites against
        ``exp(2 betaOmega n_i)`` instead, renormalizing at each site to keep the
        result finite, and traces out the bath.
        """
        # contract outward from the system (site 0) through the boson chain
        theta = self.get_theta1(0)    # system site
        rho = einsum('PiQ,PjL->iQjL', theta, theta.conj())
        # trace through each boson site with its heating operator
        for i in range(self.len_boson):
            bi = self.B[i + 1]        # boson site i+1 in the MPS
            rho = einsum('iQjL,QkR,kl,LlS->iRjS', rho, bi, self.heating_op[i], bi.conj())
            rho = rho / einsum('iRiR', rho)
        rho = einsum('iRjR->ij', rho)
        return rho

    def get_coupling(self, n, j, domain, g, ncap=20000):
        """Chain parameters ``(w_list, k_list)`` for ``n`` modes of density ``j``,
        from orthogonal-polynomial recurrence coefficients.  ``k_list[0]`` is the
        system-bath coupling and the rest are mode-mode hoppings."""
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        self.domain = domain
        return w_list, k_list

    def build_coupling(self, g, ncap):
        """Chain-map ``self.sd`` over ``self.domain`` and store ``w_list``/``k_list``."""
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g, ncap)

    def build(self, g, ncap=20000):
        """Chain-map the bath, assemble the two-site Hamiltonians, and build the
        normalized heating operators ``exp(2 betaOmega n_i)`` used by
        :meth:`get_rdm`.  Call before :meth:`get_u`."""
        self.build_coupling(g, ncap)
        hee = self.get_h2()
        self.H = hee
        self.heating_op = [scipy.linalg.expm(2 * self.betaOmega # * np.sign(freq[i])
                                             * annihilate(d).T @ annihilate(d)) for i, d in
                           enumerate(self.pd_boson)]
        self.heating_op = [op / np.linalg.norm(op) for op in self.heating_op]

    def get_h1(self):
        """On-site terms: system first, then bath modes in chain order."""
        h1 = [self.h1e]
        for i in range(self.len_boson):
            c = annihilate(self.pd_boson[i])
            h1.append(self.w_list[i] * c.T @ c)
        return h1

    def get_h2(self):
        """Two-site Hamiltonians ``[(h, d1, d2), ...]`` along the chain.

        Bond 0 (system to c0) carries the cooling gauge: the coupling is
        ``k0 (e^{betaOmega} b + e^{-betaOmega} b^dag) (x) he_dy``.
        Remaining bonds are free-chain mode-mode hoppings.
        """
        h1 = self.get_h1()
        k0 = self.k_list[0]
        h2 = []
        d1 = self.pd_sys
        d2 = self.pd_boson[0]
        annih = np.exp(self.betaOmega)
        creat = np.exp(-1 * self.betaOmega)
        c0 = annihilate(d2)
        coup = k0 * kron(self.he_dy, annih*c0 + creat*c0.T)
        site = kron(h1[0], np.eye(d2)) + kron(np.eye(d1), h1[1])
        h2.append((coup + site, d1, d2))
        for i in range(1, self.len_boson):
            d1 = self.pd_boson[i - 1]
            d2 = self.pd_boson[i]
            c1 = annihilate(d1)
            c2 = annihilate(d2)
            coup = self.k_list[i] * (kron(c1.T, c2) + kron(c1, c2.T))
            site = kron(h1[i], np.eye(d2))
            h2.append((coup + site, d1, d2))
        return h2


    def get_u(self, dt):
        """Two-site gates ``exp(-i dt h)`` with legs ``(d1, d2, d1*, d2*)``.

        The Hamiltonian is time-independent in this frame, so unlike the
        interaction-picture builders these gates are built **once**.  They are not
        unitary -- see the module docstring.
        """
        U = [0]*len(self.H)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h, dt)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # u = u.reshape([r0, s0, r1, s1])
            U[i] = u.toarray().reshape(r0,s0,r1,s1)
        return U




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
    etn = SystemBath(pd=pd, betaOmega=bo)
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


