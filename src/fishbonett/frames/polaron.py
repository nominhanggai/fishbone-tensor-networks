"""Polaron (Lang-Firsov) frame builder: an arbitrary system coupled to a harmonic
bath, driving the swap-free nearest-neighbour TEBD engine
(:mod:`fishbonett.states.mps`).

The polaron transform ``U_p = exp(O (x) sum_k (g_k/w_k)(a_k^dag - a_k))`` (``O`` the
Hermitian system-bath coupling) removes the static system-bath coupling and folds
it into a displacement of the bath.  Seeding the chain from the *reweighted*
spectral density ``J(w)/w^2`` makes the first chain mode ``c0`` the polaron
collective mode, so the dressed system term is a single two-site gate on the
``(c0, system)`` bond and the rest of the chain is free (nearest-neighbour
hopping) -- a plain Trotter sweep, no swap network.

Concretely, diagonalizing the coupling and writing ``h`` for the system
Hamiltonian, the polaron-frame Hamiltonian on the ``(c0, system)`` bond is::

    O  = sum_i lam_i |i><i|
    H~ = sum_ij <i|h|j> |i><j| (x) D_c0((lam_i - lam_j) kappa0)
         + w0 * n_c0 (x) I  -  E_reorg * I (x) O^2

i.e. each off-diagonal (in ``O``'s eigenbasis) block of ``h`` is dressed by a
displacement of the ``c0`` mode; diagonal blocks are undressed.  The physical
(Franck-Condon) bath-vacuum initial state maps to a **displaced** coherent state
on ``c0`` (:meth:`initial_theta`); diagonal observables (in ``O``'s eigenbasis)
are frame-invariant while coherences must be un-dressed (:meth:`undress_rdm`).

Applicability: **zero temperature**, uniform boson ``phys_dim``, and
``kappa0^2 = (1/pi) int J(w)/w^2 dw`` finite (a gapped or super-ohmic bath).
"""
import numpy as np
import scipy.linalg as la

from fishbonett.bath.chain import get_bath_nn_paras
from fishbonett.linalg import expm_gate
from fishbonett.operators import annihilate


def _coherent(d, alpha):
    """``D(alpha)|0> = exp(alpha (a^dag - a))|0>`` -- a real-displacement coherent state."""
    a = annihilate(d)
    return la.expm(alpha * (a.conj().T - a)) @ np.eye(d)[:, 0]


class SystemBathPolaron:
    """Polaron-frame gate builder for a general system ``O``-coupled to a harmonic bath.

    Set :attr:`coupling` (the Hermitian coupling operator ``O``), :attr:`h_sys`,
    :attr:`sd` (spectral density ``J``) and :attr:`domain`; call :meth:`build`,
    then :meth:`gates` for the static two-site Trotter gates.  :attr:`kappa0` is
    the polaron displacement; :meth:`initial_theta` prepares the displaced
    ``(c0, system)`` initial tensor and :meth:`undress_rdm` recovers the lab-frame
    system reduced density matrix.  Zero temperature only.
    """

    def __init__(self, pd):
        self.pd_sys = pd[-1]
        self.pd_boson = list(pd[0:-1])
        self.len_boson = len(self.pd_boson)
        self.coupling = np.eye(self.pd_sys, dtype=complex)
        self.h_sys = np.eye(self.pd_sys, dtype=complex)
        self.sd = lambda w: np.heaviside(w, 1.0) * np.exp(-w)
        self.domain = [0.3, 1.0]
        self.w_list = []
        self.k_list = []
        self.kappa0 = 0.0
        self.e_reorg = 0.0

    def build(self, discretizer=None):
        """Chain-map the polaron-adapted density ``J(w)/w^2`` and cache ``O``'s
        eigendecomposition and the reorganization energy."""
        sd_pol = lambda w: self.sd(w) / w ** 2
        self.w_list, self.k_list = get_bath_nn_paras(
            sd_pol, self.len_boson, self.domain, discretizer=discretizer)
        self.kappa0 = float(self.k_list[0])
        self.e_reorg = self._reorg_energy()
        self._evals, self._evecs = la.eigh(np.asarray(self.coupling, complex))

    def _reorg_energy(self):
        """``E_reorg = (1/pi) int_domain J(w)/w dw`` (the ``O^2`` on-site shift)."""
        lo, hi = self.domain
        w = np.linspace(lo, hi, 4001)
        return float(np.trapezoid(self.sd(w) / w, w) / np.pi)

    # -- static two-site gates; chain reversed so c0 is adjacent to the system ----
    def gates(self, dt):
        """Static gate list ``U[0..n-1]``; bond ``n-1`` is the dressed ``(c0, system)``
        gate, bonds ``0..n-2`` the free-chain hoppings.  Reshaped ``(d1,d2,d1*,d2*)``."""
        n = self.len_boson
        U = [None] * n
        # free-chain bonds: bond n-1-m connects (c_m, c_{m-1}); c_m on-site lives here
        for m in range(1, n):
            i = n - 1 - m
            dm, dmm = self.pd_boson[m], self.pd_boson[m - 1]
            a1, a2 = annihilate(dm), annihilate(dmm)
            num1 = a1.conj().T @ a1
            h = (self.k_list[m] * (np.kron(a1.conj().T, a2) + np.kron(a1, a2.conj().T))
                 + self.w_list[m] * np.kron(num1, np.eye(dmm)))
            U[i] = expm_gate(h, dt).reshape([dm, dmm, dm, dmm])
        # dressed (c0, system) bond at index n-1
        U[n - 1] = expm_gate(self._h_sysbond(), dt).reshape(
            [self.pd_boson[0], self.pd_sys, self.pd_boson[0], self.pd_sys])
        return U

    def _h_sysbond(self):
        d0, ds = self.pd_boson[0], self.pd_sys
        a0 = annihilate(d0); num0 = a0.conj().T @ a0
        lam, V = self._evals, self._evecs
        heig = V.conj().T @ np.asarray(self.h_sys, complex) @ V   # h in O-eigenbasis
        gen = a0.conj().T - a0
        h_sb = np.zeros((d0 * ds, d0 * ds), complex)
        for i in range(ds):
            for j in range(ds):
                if abs(heig[i, j]) < 1e-14:
                    continue
                D = la.expm((lam[i] - lam[j]) * self.kappa0 * gen)   # displace c0
                proj = np.outer(V[:, i], V[:, j].conj())             # |i><j|, comp. basis
                h_sb += heig[i, j] * np.kron(D, proj)
        O = np.asarray(self.coupling, complex)
        h_sb += self.w_list[0] * np.kron(num0, np.eye(ds))           # c0 on-site
        h_sb += -self.e_reorg * np.kron(np.eye(d0), O @ O)           # reorganization
        return h_sb

    # -- MPO form (for the TDVP drivers) -----------------------------------------
    def mpo(self):
        """Finite-state-machine MPO of the polaron Hamiltonian, sites ordered
        ``[system, c0, c1, ...]`` (so ``c0`` is adjacent to the system).

        Unlike the interaction picture, the polaron ``H~`` is **time-independent**,
        so it has a plain MPO and can be propagated with TDVP.  The system->c0 bond
        basis is ``{done, (i,j) dressed pairs..., start}``: each pair carries
        ``h_eig[i,j] |i><j|`` on the system and the displacement
        ``D((lam_i - lam_j) kappa0)`` on ``c0``; the remaining chain is the standard
        free nearest-neighbour MPO.  Tensors are ``(bond_l, bond_r, d, d)``.
        """
        d = self.pd_boson[0]
        if any(x != d for x in self.pd_boson):
            raise ValueError("the polaron MPO requires a uniform boson phys_dim")
        a = annihilate(d); ad = a.conj().T; num = ad @ a; Id = np.eye(d)
        gen = ad - a
        lam, V = self._evals, self._evecs
        heig = V.conj().T @ np.asarray(self.h_sys, complex) @ V
        ds = self.pd_sys
        pairs = [(i, j) for i in range(ds) for j in range(ds)
                 if abs(heig[i, j]) > 1e-14]
        P, Nb = len(pairs), self.len_boson
        O = np.asarray(self.coupling, complex)
        eps_chain, t_chain = self.w_list, self.k_list[1:]

        M = []
        Ms = np.zeros((1, P + 2, ds, ds), complex)              # system site
        Ms[0, 0] = -self.e_reorg * (O @ O)                      # reorganization shift
        for p, (i, j) in enumerate(pairs):
            Ms[0, 1 + p] = heig[i, j] * np.outer(V[:, i], V[:, j].conj())
        Ms[0, P + 1] = np.eye(ds)
        M.append(Ms)

        M0 = np.zeros((P + 2, 4, d, d), complex)                # c0 (polaron mode)
        M0[0, 0] = Id
        for p, (i, j) in enumerate(pairs):                      # dressing
            M0[1 + p, 0] = la.expm((lam[i] - lam[j]) * self.kappa0 * gen)
        M0[P + 1, 0] = eps_chain[0] * num
        if Nb > 1:
            M0[P + 1, 1] = t_chain[0] * a
            M0[P + 1, 2] = t_chain[0] * ad
        M0[P + 1, 3] = Id
        M.append(M0)

        for i in range(1, Nb - 1):                              # free bulk chain
            Mi = np.zeros((4, 4, d, d), complex)
            Mi[0, 0] = Id; Mi[1, 0] = ad; Mi[2, 0] = a
            Mi[3, 0] = eps_chain[i] * num
            Mi[3, 1] = t_chain[i] * a; Mi[3, 2] = t_chain[i] * ad; Mi[3, 3] = Id
            M.append(Mi)
        if Nb > 1:
            Mn = np.zeros((4, 1, d, d), complex)
            Mn[0, 0] = Id; Mn[1, 0] = ad; Mn[2, 0] = a
            Mn[3, 0] = eps_chain[Nb - 1] * num
            M.append(Mn)
        return M

    def initial_mps_pair(self, psi_sys):
        """``(A_sys, A_c0)`` for the MPO site order, in the TDVP tensor convention
        ``(bond_l, bond_r, phys)``.  ``A_c0`` is right-canonical and carries the
        polaron displacement; the bond is 1 when ``psi_sys`` is an ``O``-eigenstate
        and grows only as far as the displaced branches require."""
        d0, ds = self.pd_boson[0], self.pd_sys
        theta = self.initial_theta(psi_sys).reshape(d0, ds).T      # (ds, d0)
        U, S, Vh = la.svd(theta, full_matrices=False)
        keep = max(1, int(np.sum(S > 1e-12 * S[0])))
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
        A_sys = np.ascontiguousarray((U * S).reshape(1, ds, keep).transpose(0, 2, 1))
        A_c0 = np.ascontiguousarray(Vh.reshape(keep, 1, d0))
        return A_sys, A_c0

    # -- initial state and lab-frame observable recovery --------------------------
    def initial_theta(self, psi_sys):
        """``(c0, system)`` two-site tensor ``[1, d0, ds, 1]`` for the physical
        state ``psi_sys (x) |vac>`` mapped into the polaron frame:
        ``sum_i c_i |coherent(lam_i kappa0)>_c0 (x) |i>``."""
        d0, ds = self.pd_boson[0], self.pd_sys
        c = self._evecs.conj().T @ np.asarray(psi_sys, complex)
        theta = np.zeros((d0, ds), complex)
        for i in range(ds):
            if abs(c[i]) < 1e-14:
                continue
            theta += c[i] * np.outer(_coherent(d0, self._evals[i] * self.kappa0),
                                     self._evecs[:, i])
        return theta.reshape(1, d0, ds, 1)

    def undress_rdm_tdvp(self, A_sys, A_c0):
        """Lab-frame system RDM from the TDVP-convention ``(system, c0)`` tensors
        ``(bond_l, bond_r, phys)``; wraps :meth:`undress_rdm`."""
        theta = np.einsum('lms,mrx->lxsr', A_sys, A_c0)     # -> [L, d0, ds, R]
        return self.undress_rdm(theta)

    def undress_rdm(self, theta2):
        """Lab-frame system RDM from the ``(c0, system)`` two-site wavefunction
        ``theta2 [L, d0, ds, R]``.

        Since ``U_p`` is diagonal in ``O``'s eigenbasis::

            U_p         = sum_i |i><i| (x) D(lam_i)
            rho_lab[i,j] = <i| Tr_B[ U_p^dag rho~ U_p ] |j>
                         = Tr_B[ D(-lam_i) rho~_ij D(lam_j) ]
                         = Tr_B[ rho~_ij D(lam_j - lam_i) ]

        using cyclicity and ``D(lam_j) D(-lam_i) = D(lam_j - lam_i)`` (exact -- both
        share the generator ``c0^dag - c0``).  Diagonal elements get ``D(0) = 1`` and
        are therefore frame-invariant; coherences pick up the Franck-Condon factor.
        Only the ``(c0, system)`` block is needed because ``U_p`` displaces the
        ``c0`` mode alone, so tracing out the rest of the chain (the ``L``/``R``
        bonds) commutes with the un-dressing.
        """
        d0, ds = self.pd_boson[0], self.pd_sys
        rho2 = np.einsum('LaXR,LbYR->aXbY', theta2, theta2.conj())     # [c0o,so,c0i,si]
        V, lam, a0 = self._evecs, self._evals, annihilate(d0)
        gen = a0.conj().T - a0
        rho2e = np.einsum('Xi,aXbY,Yj->aibj', V.conj(), rho2, V)        # system legs -> eig
        M = np.zeros((ds, ds), complex)
        for i in range(ds):
            for j in range(ds):
                D = la.expm((lam[j] - lam[i]) * self.kappa0 * gen)
                M[i, j] = np.einsum('ab,ba->', rho2e[:, i, :, j], D)
        rho_lab = V @ M @ V.conj().T
        return rho_lab / np.trace(rho_lab).real
