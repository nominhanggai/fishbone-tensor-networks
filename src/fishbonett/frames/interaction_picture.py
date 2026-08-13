"""Interaction-picture builder for an arbitrary system coupled to a harmonic
bath, driving the swap-network TEBD engine (:mod:`fishbonett.states.mps`).

The system is any Hermitian ``h_sys`` with any coupling operator (not just a
two-level spin -- the "spin-boson" name is historical); only the bath is
constrained to be harmonic/bosonic.  Works in the interaction picture with
respect to the system-bath coupling: the
chain-mapped bath is diagonalized and each mode carries a time-dependent coupling
``d_j(t)``, so the two-site Trotter gates are rebuilt each step.  This is the
builder behind ``BosonicBath.run(method="tebd")``.
"""
import numpy as np
from numpy import exp
from copy import deepcopy as dcopy

import fishbonett.recurrence_coefficients as rc
from fishbonett.contract import contract as einsum
# eye/svd are re-exported for back-compat; the class below uses kron and the
# dense gate exponential (_expm_gate).
from fishbonett.linalg import eye, kron, svd, expm_gate as _expm_gate
from fishbonett.operators import temp_factor, _c
from fishbonett.spectral_densities import drude


class BosonicBathIP:
    """Interaction-picture builder: arbitrary system + harmonic bath, TEBD engine.

    Diagonalizes the chain-mapped bath and rebuilds the time-dependent two-site
    Trotter gates every step (:meth:`get_u`), working in the interaction picture
    with respect to the system-bath coupling.  Set :attr:`coupling` (the coupling
    operator) and :attr:`h_sys` (the system Hamiltonian), call :meth:`build`, then
    step with :meth:`get_u`.  Historically named ``SpinBoson`` (aliases kept).
    """

    def __init__(self, pd):
        self.pd_sys = pd[-1]
        self.pd_boson = pd[0:-1]
        self.len_boson = len(self.pd_boson)
        self.sd = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
        self.domain = [0, 1]
        self.coupling = np.eye(self.pd_sys)
        self.h_sys = np.eye(self.pd_sys)
        self.k_list = []
        self.w_list = []
        self.H = []
        self.coef= []
        self.freq = []
        self.phase = lambda lam, t, delta: (np.exp(-1j*lam*(t+delta)) - np.exp(-1j*lam*t))/(-1j*lam)
        self.phase_func = lambda lam, t: np.exp(-1j * lam * (t))
        # self.phase = lambda lam, t, delta: np.exp(-1j * lam * (t+delta/2)) * delta

    def get_coupling(self, n, j, domain, g, ncap=20000, discretizer=None):
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap,
            discretizer=discretizer,
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        _, _, self.h_squared = rc._j_to_hsquared(func=j, lb=domain[0], rb=domain[1], g=g)
        self.domain = domain
        return w_list, k_list

    def build_coupling(self, g, ncap, discretizer=None):
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g,
                                                     ncap, discretizer=discretizer)

    def diag(self):
        w= self.w_list
        k = self.k_list
        self.coup = np.diag(w) + np.diag(k[1:], 1) + np.diag(k[1:], -1)
        freq, coef = np.linalg.eigh(self.coup)
        sign = np.sign(coef[0,:])
        coef = coef.dot(np.diag(sign))
        return freq, coef

    def get_h2(self, t, delta, inc_sys=True):
        freq = self.freq
        coef = self.coef
        e = self.phase
        k0 = self.k_list[0]
        j0 = k0 * coef[0,:] # interaction strength in the diagonal representation
        phase_factor = np.array([e(w, t, delta) for w in freq])
        perm = np.abs(j0).argsort()
        shuffle = coef.T#[perm]
        d_nt = [einsum('k,k,k', j0, shuffle[:,n], phase_factor) for n in range(len(freq))]
        # print(f'd_nt{d_nt}')
        d_nt = d_nt[::-1]
        h2 = []
        # ul = _expm_gate(self.h_sys, -t)
        # coupling = ul @ self.coupling @ (ul.T.conj())
        coupling = self.coupling
        for i, k in enumerate(d_nt):
            d1 = self.pd_boson[i]
            d2 = self.pd_sys
            c1 = _c(d1)
            kc = k.conjugate()
            coup = kron(k*c1 + kc* c1.T, coupling)
            h2.append((coup, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_sys
        site = delta*kron(np.eye(d1), self.h_sys)
        if inc_sys is True:
            h2[-1] = (h2[-1][0] + site, d1, d2)
        else:
            h2[-1] = (h2[-1][0], d1, d2)
        return h2

    def build(self, g, ncap=20000, discretizer=None):
        self.build_coupling(g, ncap, discretizer=discretizer)
        self.freq, self.coef = self.diag()
        # self.pn_list = self.poly()
        # hee = self.get_h2(t)
        # print("Hamiltonian Over")
        # self.H = hee

    def get_u(self, t, dt, mode='normal', factor=1, inc_sys=True):
        self.H = self.get_h2(t, dt, inc_sys)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = _expm_gate(h.toarray()/factor, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # print(u)
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1,0,3,2])
            U1[i] = u1
            U2[i] = u2
        return U1, U2


#: ``BosonicBathIP`` was historically named ``SpinBoson``; both aliases kept.
BosonicBath = BosonicBathIP
SpinBoson = BosonicBathIP  # deprecated
