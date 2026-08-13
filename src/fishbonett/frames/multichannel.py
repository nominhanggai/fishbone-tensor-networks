"""Multichannel interaction-picture builder: one bath, several coupling operators.

The other frames assume the bath couples to the system through a *single*
operator.  Here several operators ``A_c`` share the **same** modes::

    H_sb = sum_k (sum_c A_c g_k^(c)) (b_k + b_k^dagger)

which is genuinely different from several independent baths: one set of modes
drives every channel, so the noises they impose are **cross-correlated**.
Physically that is the difference between a molecule whose electronic gap and
inter-site coupling are modulated by the *same* vibrations and by unrelated ones.

Two consequences shape the code below:

* the coupling is **matrix-valued** -- each mode carries a matrix ``A(d_n(t))``
  rather than a scalar times one operator -- so the star-to-chain map is a
  :func:`~fishbonett.bath.lanczos.block_lanczos` seeded with all channels at
  once (:meth:`SystemBathMultiChannel.build`);
* finite temperature is folded in through
  :func:`~fishbonett.operators.temp_factor` on a signed frequency axis, which is
  why the frequency array is mirrored in the constructor.

Selected by the *bath*, not by a ``method`` name: give
:class:`~fishbonett.bath.spec.Bath` a list of ``coupling`` operators.  See
:doc:`/methods/interaction/multichannel`.
"""
from copy import deepcopy as dcopy

import numpy as np
from numpy import exp

from fishbonett.contract import contract as einsum
from fishbonett.bath.lanczos import lanczos
from fishbonett.linalg import kron, expm_gate as calc_U
from fishbonett.operators import temp_factor, annihilate







class SystemBathMultiChannel:
    """Multichannel interaction-picture builder: system + harmonic bath, >=2 channels.

    Generalizes :class:`~fishbonett.frames.interaction_picture.SystemBathIP` to a
    matrix-valued coupling -- several coupling channels ``A_k`` share one bath (any
    Hermitian system, not just a spin), with the finite-temperature thermofield
    doubling folded in via ``temp_factor``.
    """

    def __init__(self, pd, coup_mat, freq, temp, H_add=None):
        if H_add is None:
            self.H_add = []
        else:
            self.H_add = H_add
        self.pd_sys = pd[0]
        self.pd_boson = pd[1:]
        self.len_boson = len(self.pd_boson)
        self.sd = [lambda x: np.heaviside(x, 1) / 1. * exp(-x / 100)] * self.pd_sys
        self.domain = [0, 1]
        self.he_dy = np.eye(self.pd_sys)
        self.h1e = np.eye(self.pd_sys)
        self.temp = temp
        freq = np.array(freq)
        self.freq = np.concatenate((-freq, freq))
        coup_mat = np.concatenate((coup_mat, coup_mat))
        self.coup_mat = [mat * np.sqrt(np.abs(temp_factor(temp, self.freq[n]))) for n, mat in enumerate(coup_mat)]
        self.size = self.coup_mat[0].shape[0]
        self.coup_mat_np = np.array(self.coup_mat)
        #  ↑ A list of coupling matrices A_k. H_i = \sum_k A_k \otimes (a+a^\dagger)
        self.H = []
        self.coef = []
        self.phase = lambda lam, t, delta: (np.exp(-1j * lam * (t + delta)) - np.exp(-1j * lam * t)) / (-1j * lam)
        self.phase_func = lambda lam, t: np.exp(-1j * lam * (t))

    def get_h2(self, t, delta, inc_sys=True):
        """Two-site coupling Hamiltonians over ``[t, t+delta]``, in chain order.

        Returns ``[(h, d_boson, d_sys), ...]``: for each chain mode, the
        matrix-valued interaction-picture coupling summed over channels,
        ``kron(b, D_n) + kron(b^dag, D_n*)`` with ``D_n`` the channel-weighted
        coupling matrix.  With ``inc_sys`` the system term ``delta * h1e`` is
        added to the site nearest the system.  Any extra explicit modes in
        ``H_add`` are appended.
        """
        freq = self.freq
        coef = self.coef
        e = self.phase
        mat_list = self.coup_mat_np
        phase_factor = np.array([e(w, t, delta) for w in freq])
        d_nt_mat = [einsum('kst,k,k', mat_list, coef[:, n], phase_factor) for n in range(len(freq))]
        h2 = []
        for i, k in enumerate(d_nt_mat[:self.len_boson]):
            d1 = self.pd_boson[i]
            d2 = self.pd_sys
            c1 = annihilate(d1)
            kc = k.conjugate()
            coup = kron(c1, k) + kron(c1.T, kc)
            h2.append((coup, d1, d2))
        d1 = self.pd_boson[0]
        d2 = self.pd_sys
        site = delta * kron(np.eye(d1), self.h1e)
        if inc_sys is True:
            h2[0] = (h2[0][0] + site, d1, d2)
        else:
            pass
        for hi in self.H_add:
            hs, hb, w = hi
            ds, db = hs.shape[0], hb.shape[0]
            c = annihilate(db)
            coup = kron(hb, hs) + w * kron(c.T@c, np.eye(ds))
            h2.append((coup, db, ds))
        return h2

    def build(self, n):
        """Chain-map the shared star, seeded by channel ``n``'s couplings.

        Lanczos-tridiagonalizes the star Hamiltonian ``diag(freq)`` and stores the
        star -> chain transform in ``self.coef`` and the chain frequencies in
        ``self.chain_freq``.  Call before :meth:`get_u`.
        """
        def tri_diag(self, n):
            v0 = [mat[n, n] for mat in self.coup_mat_np]
            h = np.diag(self.freq)
            tri_mat, coef = lanczos(h, v0)
            return tri_mat, coef

        # self.build_coupling(g, ncap)
        chain_freq, Q = tri_diag(self, n)
        res = np.diagonal(Q.T @ Q - np.eye(Q.shape[0]))
        # print(repr(Q[:,0])) ## Should be parallel to one of the coup_mat vectors
        self.coef = Q
        self.chain_freq = np.diagonal(chain_freq)

    def get_u(self, t, dt, mode='normal', factor=1, inc_sys=True):
        """Two-site Trotter gates over ``[t, t+dt]`` as ``(U1, U2)``.

        Exponentiates each two-site Hamiltonian from :meth:`get_h2`.  ``U1`` has
        legs ``(d1, d2, d1*, d2*)``; ``U2`` is the leg-transposed variant the
        *swapped* sweeps consume.  ``factor`` divides the Hamiltonian (for
        sub-stepping).  Because the frame is time-dependent, this must be called
        afresh each step.
        """
        self.H = self.get_h2(t, dt, inc_sys)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h.toarray() / factor, 1)
            # h is in (d1 x d2) basis; transpose to (d2, d1, d2, d1) = (sys, boson, ...)
            u1 = u.reshape([d1, d2, d1, d2]).transpose([1, 0, 3, 2])
            u2 = np.transpose(u1, [1, 0, 3, 2])
            U1[i] = u1
            U2[i] = u2
        return U1, U2


