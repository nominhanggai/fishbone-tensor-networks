"""Interaction-picture builder for an arbitrary system coupled to a harmonic
bath, driving the swap-network TEBD engine (:mod:`fishbonett.states.mps`).

The system is any Hermitian ``h_sys`` with any coupling operator (not just a
two-level spin -- the "spin-boson" name is historical); only the bath is
constrained to be harmonic/bosonic.

Works in the interaction picture with respect to the **free bath** Hamiltonian:
the chain-mapped bath is diagonalized into its star modes, whose free evolution is
absorbed into time-dependent couplings ``d_n(t)`` (:meth:`SystemBathIP.mode_couplings`).
What remains is ``H_sb(t) = A_s (x) sum_n [d_n b_n + h.c.]`` plus the system term,
so the gates are rebuilt each step.  Two propagators use this frame:

* ``SystemBath.run(method="tebd")`` -- two-site Trotter gates (:meth:`get_u`)
  applied by the swap network;
* ``SystemBath.run(method="trotter-mpo")`` -- the same propagator written exactly
  as one low-bond conditional-displacement MPO (:meth:`displacement_mpo`).
"""
import numpy as np
import scipy.linalg as la
from numpy import exp
from copy import deepcopy as dcopy

import fishbonett.bath.recurrence as rc
from fishbonett.contract import contract as einsum
from fishbonett.linalg import kron, expm_gate as _expm_gate
from fishbonett.operators import annihilate


class SystemBathIP:
    """Interaction-picture builder: arbitrary system + harmonic bath.

    Diagonalizes the chain-mapped bath into its star modes and absorbs their free
    evolution into time-dependent couplings ``d_n(t)``
    (:meth:`mode_couplings`) -- the interaction picture with respect to the **free
    bath** Hamiltonian.  Set :attr:`coupling` (the system operator ``A_s``) and
    :attr:`h_sys`, call :meth:`build`, then take either

    * :meth:`get_u` -- two-site Trotter gates for the swap-network TEBD sweep, or
    * :meth:`displacement_mpo` -- the same propagator as one exact low-bond MPO.

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
        """Chain parameters ``(w_list, k_list)`` for ``n`` modes of density ``j``.

        Uses orthogonal-polynomial recurrence coefficients: ``w_list`` are the
        chain on-site energies and ``k_list = [k0, hop_1, ...]`` the couplings,
        ``k0`` being the system-bath one.  Also records ``self.domain`` and the
        ``h_squared`` weights.  ``g`` rescales the frequency axis; ``discretizer``
        selects the quadrature (``None`` = Gauss-Legendre).
        """
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
        """Chain-map ``self.sd`` over ``self.domain`` into ``w_list``/``k_list``
        (one chain site per entry of ``pd_boson``)."""
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g,
                                                     ncap, discretizer=discretizer)

    def diag(self):
        """Diagonalize the chain back into its star: ``(freq, coef)``.

        The interaction picture needs the *star* modes, because it is their free
        evolution ``e^{-i w_k t}`` that is rotated out.  Eigen-decomposes the
        tridiagonal chain matrix and fixes each eigenvector's sign by its first
        component, so the transform is deterministic.
        """
        w= self.w_list
        k = self.k_list
        self.coup = np.diag(w) + np.diag(k[1:], 1) + np.diag(k[1:], -1)
        freq, coef = np.linalg.eigh(self.coup)
        sign = np.sign(coef[0,:])
        coef = coef.dot(np.diag(sign))
        return freq, coef

    def mode_couplings(self, t, delta):
        """Time-integrated interaction-picture coupling ``d_n`` of each chain mode,
        in chain-site order (site ``i`` of ``pd_boson``).

        ``d_n(t) = int_t^{t+delta} dt' sum_k j0_k U_kn e^{-i w_k t'}`` -- the free-bath
        phase of every star mode, rotated back to the chain basis.  Note ``delta`` is
        already folded in, so the propagator generated by these couplings needs no
        further factor of ``dt``.
        """
        j0 = self.k_list[0] * self.coef[0, :]     # star couplings in the diagonal basis
        phase_factor = np.array([self.phase(w, t, delta) for w in self.freq])
        shuffle = self.coef.T
        d_nt = [einsum('k,k,k', j0, shuffle[:, n], phase_factor)
                for n in range(len(self.freq))]
        return np.array(d_nt[::-1])               # chain-site order

    def displacement_mpo(self, t, delta):
        """Conditional-displacement MPO of the system-bath propagator over
        ``[t, t+delta]``; sites ordered ``[system, mode_0, ..., mode_{N-1}]``.

        In the interaction picture the coupling is ``H_sb(t) = A_s (x) B(t)`` with
        ``B(t) = sum_n [d_n b_n + d_n* b_n^dag]``.  Every term ``A_s (x) X_n``
        commutes with every other (distinct modes commute and ``A_s`` is a common
        factor), so the multimode propagator factorizes **exactly**:

            exp(-i A_s (x) B) = prod_n exp(-i A_s (x) X_n)

        Diagonalizing ``A_s = sum_a a P_a``, each factor becomes a displacement of
        that mode conditioned on the system eigenvalue, giving

            U_sb = sum_a P_a (x) (x)_n D_n(alpha_{a,n}),   alpha_{a,n} = -i a d_n*

        which is an MPO whose bond dimension is the number of *distinct* eigenvalues
        of ``A_s`` (2 for a ``sigma_z`` spin-boson).  Tensors are
        ``(bond_l, bond_r, phys_out, phys_in)``.
        """
        lam, V = la.eigh(np.asarray(self.coupling, complex))
        d_n = self.mode_couplings(t, delta)
        r, ds, n = len(lam), self.pd_sys, self.len_boson

        W = [np.zeros((1, r, ds, ds), complex)]
        for a in range(r):                        # projectors P_a on the system site
            W[0][0, a] = np.outer(V[:, a], V[:, a].conj())
        for i in range(n):
            d = self.pd_boson[i]
            b = annihilate(d); bd = b.conj().T
            wr = r if i < n - 1 else 1            # the branch label is carried through
            Wi = np.zeros((r, wr, d, d), complex)
            for a in range(r):
                alpha = -1j * lam[a] * np.conj(d_n[i])
                Wi[a, a if i < n - 1 else 0] = la.expm(alpha * bd - np.conj(alpha) * b)
            W.append(Wi)
        return W

    def get_h2(self, t, delta, inc_sys=True):
        """Two-site coupling Hamiltonians over ``[t, t+delta]``, in chain order.

        Returns ``[(h, d_boson, d_sys), ...]`` with
        ``h = (d_n b + d_n* b^dag) (x) A_s`` for each mode, ``d_n`` from
        :meth:`mode_couplings`.  With ``inc_sys`` the system term
        ``delta * h_sys`` is added to the last (system-adjacent) bond.  ``delta``
        is already integrated into ``d_n``, so no extra factor of ``dt`` is needed.
        """
        d_nt = self.mode_couplings(t, delta)
        h2 = []
        # ul = _expm_gate(self.h_sys, -t)
        # coupling = ul @ self.coupling @ (ul.T.conj())
        coupling = self.coupling
        for i, k in enumerate(d_nt):
            d1 = self.pd_boson[i]
            d2 = self.pd_sys
            c1 = annihilate(d1)
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
        """Chain-map the bath and diagonalize it into star modes.

        Must be called after setting :attr:`sd`, :attr:`domain`,
        :attr:`coupling` and :attr:`h_sys`, and before :meth:`get_u` or
        :meth:`displacement_mpo`.
        """
        self.build_coupling(g, ncap, discretizer=discretizer)
        self.freq, self.coef = self.diag()
        # self.pn_list = self.poly()
        # hee = self.get_h2(t)
        # print("Hamiltonian Over")
        # self.H = hee

    def get_u(self, t, dt, mode='normal', factor=1, inc_sys=True):
        """Two-site Trotter gates over ``[t, t+dt]`` as ``(U1, U2)``.

        Exponentiates each two-site Hamiltonian from :meth:`get_h2`.  ``U1`` has
        legs ``(d1, d2, d1*, d2*)`` and ``U2`` is its leg-transposed variant,
        which the *swapped* sweeps of the swap network consume -- see
        :func:`fishbonett.evolve.tebd.symmetric_swap_step`, which calls this twice
        per step (once per half-interval) to stay second order.

        Because the frame is time-dependent, the gates are valid only for the
        interval they were built for.
        """
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


