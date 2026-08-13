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

Applicability: uniform boson ``phys_dim`` and
``kappa0^2 = (1/pi) int J(w)/w^2 dw`` finite (a gapped or super-ohmic bath).
Finite temperature works via T-TEDOPA: set ``bath.temperature`` and the spectral
density is thermalized onto a signed frequency axis before chain mapping, exactly
as in the interaction picture.
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

    Everything the frame needs is given at construction; :meth:`build` then does the
    chain mapping (the one expensive step) and :meth:`gates` returns the static
    two-site Trotter gates.  :attr:`kappa0` is the polaron displacement;
    :meth:`initial_theta` prepares the displaced ``(c0, system)`` initial tensor and
    :meth:`undress_rdm` recovers the lab-frame system reduced density matrix.

    Parameters
    ----------
    pd : sequence of int
        Physical dimensions ``[d_sys, d_boson, ...]`` -- the system on site 0, one
        entry per chain mode after it.
    h_sys : (d, d) array
        System Hamiltonian.
    coupling : (d, d) array
        The Hermitian system-bath coupling ``O``.
    sd : callable
        Spectral density ``J(w)``, already thermalized if the bath is at finite
        temperature (see :func:`fishbonett.bath.spec.thermalize`).
    domain : (float, float)
        Frequency window to chain-map over.
    discretizer : callable, optional
        Quadrature for the star discretization; ``None`` is Gauss-Legendre.
    """

    def __init__(self, pd, *, h_sys, coupling, sd, domain, discretizer=None):
        self.pd_sys = pd[0]
        self.pd_boson = list(pd[1:])
        self.len_boson = len(self.pd_boson)
        if self.len_boson == 0:
            raise ValueError("pd must contain at least one boson mode after the "
                             "system dimension")
        self.h_sys = np.asarray(h_sys, complex)
        self.coupling = np.asarray(coupling, complex)
        for name, op in (("h_sys", self.h_sys), ("coupling", self.coupling)):
            if op.shape != (self.pd_sys, self.pd_sys):
                raise ValueError(f"{name} has shape {op.shape}, expected "
                                 f"{(self.pd_sys, self.pd_sys)} to match pd[0]")
        self.sd = sd
        self.domain = list(domain)
        self.discretizer = discretizer
        self.w_list = []
        self.k_list = []
        self.kappa0 = 0.0
        self.e_reorg = 0.0

    def build(self):
        """Chain-map the polaron-adapted density ``J(w)/w^2`` and cache ``O``'s
        eigendecomposition and the reorganization energy.  Call before
        :meth:`gates`."""
        sd_pol = lambda w: self.sd(w) / w ** 2
        self.w_list, self.k_list = get_bath_nn_paras(
            sd_pol, self.len_boson, self.domain, discretizer=self.discretizer)
        self.kappa0 = float(self.k_list[0])
        self.e_reorg = self._reorg_energy()
        self._evals, self._evecs = la.eigh(self.coupling)
        return self

    def _reorg_energy(self):
        """``E_reorg = (1/pi) int_domain J(w)/w dw`` (the ``O^2`` on-site shift)."""
        lo, hi = self.domain
        w = np.linspace(lo, hi, 4001)
        sd_vec = np.vectorize(self.sd)
        mask = np.abs(w) > 1e-12
        jw = np.zeros_like(w)
        jw[mask] = sd_vec(w[mask]) / w[mask]
        return float(np.trapezoid(jw, w) / np.pi)

    # -- static two-site gates; system at site 0, c0 at site 1, then c1, c2, ... ----
    def gates(self, dt):
        """Static gate list ``U[0..n-1]``; bond 0 is the dressed ``(system, c0)``
        gate, bonds ``1..n-1`` the free-chain hoppings.  Reshaped ``(d1,d2,d1*,d2*)``."""
        n = self.len_boson
        U = [None] * n
        # dressed (system, c0) bond at index 0
        U[0] = expm_gate(self._h_sysbond(), dt).reshape(
            [self.pd_sys, self.pd_boson[0], self.pd_sys, self.pd_boson[0]])
        # free-chain bonds: bond m connects (c_{m-1}, c_m).  c_0's on-site term is
        # already in the dressed bond above, so bond m carries c_m's -- the *right*
        # leg -- which keeps every mode's frequency on that same mode.
        for m in range(1, n):
            dm_prev, dm = self.pd_boson[m - 1], self.pd_boson[m]
            a1, a2 = annihilate(dm_prev), annihilate(dm)
            num2 = a2.conj().T @ a2
            h = (self.k_list[m] * (np.kron(a1.conj().T, a2) + np.kron(a1, a2.conj().T))
                 + self.w_list[m] * np.kron(np.eye(dm_prev), num2))
            U[m] = expm_gate(h, dt).reshape([dm_prev, dm, dm_prev, dm])
        return U

    def _h_sysbond(self):
        """Two-site Hamiltonian on the (system, c0) bond, order ``(ds, d0)``."""
        d0, ds = self.pd_boson[0], self.pd_sys
        a0 = annihilate(d0); num0 = a0.conj().T @ a0
        lam, V = self._evals, self._evecs
        heig = V.conj().T @ np.asarray(self.h_sys, complex) @ V   # h in O-eigenbasis
        gen = a0.conj().T - a0
        h_sb = np.zeros((ds * d0, ds * d0), complex)
        for i in range(ds):
            for j in range(ds):
                if abs(heig[i, j]) < 1e-14:
                    continue
                D = la.expm((lam[i] - lam[j]) * self.kappa0 * gen)   # displace c0
                proj = np.outer(V[:, i], V[:, j].conj())             # |i><j|, comp. basis
                h_sb += heig[i, j] * np.kron(proj, D)
        O = np.asarray(self.coupling, complex)
        h_sb += self.w_list[0] * np.kron(np.eye(ds), num0)           # c0 on-site
        h_sb += -self.e_reorg * np.kron(O @ O, np.eye(d0))           # reorganization
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
        """``(A_sys, A_c0)`` for the MPO site order ``[system, c0, ...]``, in the
        TDVP tensor convention ``(bond_l, bond_r, phys)``.  ``A_c0`` is
        right-canonical and carries the polaron displacement."""
        d0, ds = self.pd_boson[0], self.pd_sys
        theta = self.initial_theta(psi_sys).reshape(ds, d0)        # (ds, d0)
        U, S, Vh = la.svd(theta, full_matrices=False)
        keep = max(1, int(np.sum(S > 1e-12 * S[0])))
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
        A_sys = np.ascontiguousarray((U * S).reshape(1, ds, keep).transpose(0, 2, 1))
        A_c0 = np.ascontiguousarray(Vh.reshape(keep, 1, d0))
        return A_sys, A_c0

    # -- initial state and lab-frame observable recovery --------------------------
    def initial_theta(self, psi_sys):
        """``(system, c0)`` two-site tensor ``[1, ds, d0, 1]`` for the physical
        state ``psi_sys (x) |vac>`` mapped into the polaron frame:
        ``sum_i c_i |i> (x) |coherent(lam_i kappa0)>_c0``."""
        d0, ds = self.pd_boson[0], self.pd_sys
        c = self._evecs.conj().T @ np.asarray(psi_sys, complex)
        theta = np.zeros((ds, d0), complex)
        for i in range(ds):
            if abs(c[i]) < 1e-14:
                continue
            theta += c[i] * np.outer(self._evecs[:, i],
                                     _coherent(d0, self._evals[i] * self.kappa0))
        return theta.reshape(1, ds, d0, 1)

    def undress_rdm_tdvp(self, A_sys, A_c0):
        """Lab-frame system RDM from the TDVP-convention ``(system, c0)`` tensors
        ``(bond_l, bond_r, phys)``; wraps :meth:`undress_rdm`."""
        theta = np.einsum('lms,mrx->lsxr', A_sys, A_c0)     # -> [L, ds, d0, R]
        return self.undress_rdm(theta)

    def undress_rdm(self, theta2):
        """Lab-frame system RDM from the ``(system, c0)`` two-site wavefunction
        ``theta2 [L, ds, d0, R]``.

        Since ``U_p`` is diagonal in ``O``'s eigenbasis::

            U_p         = sum_i |i><i| (x) D(lam_i)
            rho_lab[i,j] = <i| Tr_B[ U_p^dag rho~ U_p ] |j>
                         = Tr_B[ D(-lam_i) rho~_ij D(lam_j) ]
                         = Tr_B[ rho~_ij D(lam_j - lam_i) ]

        Diagonal elements get ``D(0) = 1`` and are frame-invariant; coherences
        pick up the Franck-Condon factor.  Only the ``(system, c0)`` block is needed
        because ``U_p`` displaces ``c0`` alone.
        """
        d0, ds = self.pd_boson[0], self.pd_sys
        rho2 = np.einsum('LXaR,LYbR->XaYb', theta2, theta2.conj())     # [so,c0o,si,c0i]
        V, lam, a0 = self._evecs, self._evals, annihilate(d0)
        gen = a0.conj().T - a0
        rho2e = np.einsum('Xi,XaYb,Yj->iajb', V.conj(), rho2, V)        # system legs -> eig
        M = np.zeros((ds, ds), complex)
        for i in range(ds):
            for j in range(ds):
                D = la.expm((lam[j] - lam[i]) * self.kappa0 * gen)
                M[i, j] = np.einsum('ab,ba->', rho2e[i, :, j, :], D)
        rho_lab = V @ M @ V.conj().T
        return rho_lab / np.trace(rho_lab).real
