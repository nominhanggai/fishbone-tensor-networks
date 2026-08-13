from copy import deepcopy as dcopy

import numpy as np
from numpy import exp

from fishbonett.contract import contract as einsum
from fishbonett.lanczos import lanczos
# calc_U here is the dense gate exponential; eye/svd re-exported for back-compat.
from fishbonett.linalg import eye, kron, svd, expm_gate as calc_U
from fishbonett.operators import temp_factor, _c





from fishbonett.mps import BosonicBathMPS as BosonicBath1D  # engine unified into fishbonett.mps


class BosonicBathMultiChannel:
    """Multichannel interaction-picture builder: system + harmonic bath, >=2 channels.

    Generalizes :class:`~fishbonett.frames.interaction_picture.BosonicBathIP` to a
    matrix-valued coupling -- several coupling channels ``A_k`` share one bath (any
    Hermitian system, not just a spin), with the finite-temperature thermofield
    doubling folded in via ``temp_factor``.  Historically named ``SpinBoson``
    (aliases kept).
    """

    def __init__(self, pd, coup_mat, freq, temp, H_add=None):
        if H_add is None:
            self.H_add = []
        else:
            self.H_add = H_add
        self.pd_sys = pd[-1]
        self.pd_boson = pd[0:-1]
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
        freq = self.freq
        coef = self.coef
        e = self.phase
        mat_list = self.coup_mat_np
        phase_factor = np.array([e(w, t, delta) for w in freq])
        d_nt_mat = [einsum('kst,k,k', mat_list, coef[:, n], phase_factor) for n in range(len(freq))]
        h2 = []
        for i, k in enumerate(d_nt_mat[:self.len_boson]):
            d1 = self.pd_boson[::-1][i]
            d2 = self.pd_sys
            c1 = _c(d1)
            kc = k.conjugate()
            coup = kron(c1, k) + kron(c1.T, kc)
            h2.append((coup, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_sys
        site = delta * kron(np.eye(d1), self.h1e)
        if inc_sys is True:
            h2[0] = (h2[0][0] + site, d1, d2)
        else:
            pass
        for hi in self.H_add:
            hs, hb, w = hi
            ds, db = hs.shape[0], hb.shape[0]
            c = _c(db)
            coup = kron(hb, hs) + w * kron(c.T@c, np.eye(ds))
            h2.append((coup, db, ds))
        return h2[::-1]

    def build(self, n):
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
        self.H = self.get_h2(t, dt, inc_sys)
        U1 = dcopy(self.H)
        U2 = dcopy(U1)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h.toarray() / factor, 1)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            u1 = u.reshape([r0, s0, r1, s1])
            u2 = np.transpose(u1, [1, 0, 3, 2])
            U1[i] = u1
            U2[i] = u2
        return U1, U2


#: ``BosonicBathMultiChannel`` was historically named ``SpinBoson``; both aliases kept.
BosonicBath = BosonicBathMultiChannel
SpinBoson = BosonicBathMultiChannel  # deprecated
