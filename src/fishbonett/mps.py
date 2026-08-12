"""Canonical time-evolving block-decimation (TEBD) engine for spin-boson chains.

This single :class:`SpinBosonMPS` engine replaces the ~20 near-identical
``SpinBoson1D`` copies that used to live inside the individual driver modules.  It
is the feature-superset of those copies and supports every truncation scheme they
implemented, selected per :meth:`update_bond` call:

* **swap** -- ``swap=1`` transposes the two physical legs during the gate, so a
  distant bath mode can be evolved next to the system site (the interaction /
  "backward" picture);
* **single truncated SVD** at ``chi_max`` (the default), or an **adaptive**
  bond-dimension search (``adaptive=True``) that grows the trial rank until the
  truncation error is resolved;
* optional **local basis optimization (LBO)** -- pass ``eps_lbo`` to project each
  boson site onto its optimal reduced basis before the SVD (this implies the
  adaptive search);
* optional **GPU** execution through CuPy (``gpu=True``), used only if CuPy is
  importable.

With ``eps_lbo=None`` and ``adaptive=False`` the update reduces exactly to the
plain single-SVD scheme, so results are unchanged from the historical engines.
"""
import numpy as np
import scipy.linalg
from opt_einsum import contract as einsum

from fishbonett.common import svd

try:  # optional GPU backend
    import cupy as cp
    from fishbonett.rsvd_cupy import rsvd as _curdsvd

    _CUPY = True
    _mempool = cp.get_default_memory_pool()

    def _cusvd(a, b, full_matrices=False):
        dim = min(a.shape[0], a.shape[1])
        b = min(b, dim)
        return _curdsvd(a, b, True, n_iter=2, l=2 * b)
except ImportError:  # pragma: no cover - exercised only with a GPU present
    _CUPY = False


class SpinBosonMPS:
    """Matrix-product state of a boson chain terminated by a system (spin) site.

    Parameters
    ----------
    pd : sequence of int
        Physical dimensions ``[d_boson_0, ..., d_boson_{L-1}, d_spin]``; the
        system site is last.
    svd_expansion_factor : float, optional
        Growth factor for the adaptive trial bond dimension (default 1.5).
    """

    def __init__(self, pd, svd_expansion_factor=1.5):
        self.pd_spin = pd[-1]
        self.pd_boson = pd[0:-1]
        self.pre_factor = svd_expansion_factor
        self.B = [self._ground(d) for d in pd]
        self.S = [np.ones([1], float) for _ in pd]
        self.U = [np.zeros(0) for _ in pd[1:]]
        # Optimal-basis projectors for LBO; identity means "no optimization".
        self.R = [np.eye(d) for d in pd]

    @staticmethod
    def _ground(d):
        tensor = np.zeros([1, d, 1], dtype=np.complex128)
        tensor[0, 0, 0] = 1.0
        return tensor

    # -- wavefunction accessors ------------------------------------------------
    def get_theta1(self, i, gpu=False):
        if gpu and _CUPY:
            proj = cp.tensordot(cp.diag(self.S[i]), cp.array(self.B[i]), [1, 0])
            return einsum('KI,aIb->aKb', cp.array(self.R[i]), proj)
        proj = np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])
        return einsum('KI,aIb->aKb', self.R[i], proj)

    def get_theta2(self, i, gpu=False):
        j = i + 1
        if gpu and _CUPY:
            return einsum('aIb,LJ,bJc->aILc', self.get_theta1(i, gpu),
                          cp.array(self.R[j]), cp.array(self.B[j]))
        return einsum('aIb,LJ,bJc->aILc', self.get_theta1(i), self.R[j], self.B[j])

    # -- core update -----------------------------------------------------------
    def update_bond(self, i, chi_max, eps, swap=0, eps_lbo=None, adaptive=False,
                    gpu=False):
        """Apply the two-site gate ``self.U[i]`` at bond ``i`` and re-split.

        Parameters
        ----------
        swap : {0, 1}
            1 transposes the two physical legs during the gate.
        eps_lbo : float, optional
            Local-basis-optimization threshold; enables LBO (and the adaptive
            search) when not ``None``.
        adaptive : bool
            Use the adaptive bond-dimension search without LBO.  Ignored when
            ``eps_lbo`` is given.  Default is a single truncated SVD at
            ``chi_max``.
        """
        use_gpu = bool(gpu and _CUPY)
        theta = self.get_theta2(i, gpu=use_gpu)
        u_bond = cp.array(self.U[i]) if use_gpu else self.U[i]
        if swap == 1:
            utheta = einsum('ijkl,PklQ->PjiQ', u_bond, theta)
        elif swap == 0:
            utheta = einsum('ijkl,PklQ->PijQ', u_bond, theta)
        else:
            raise ValueError(f"swap must be 0 or 1, got {swap!r}")
        self.split_truncate_theta(utheta, i, chi_max, eps, eps_lbo=eps_lbo,
                                  adaptive=adaptive, gpu=use_gpu)

    def split_truncate_theta(self, theta, i, chi_max, eps, eps_lbo=None,
                             adaptive=False, gpu=False):
        if gpu and _CUPY:
            self._split_gpu(theta, i, chi_max, eps, eps_lbo)
        elif eps_lbo is None and not adaptive:
            self._split_plain(theta, i, chi_max, eps)
        else:
            self._split_adaptive(theta, i, chi_max, eps, eps_lbo)

    # -- single truncated SVD at chi_max, no LBO -------------------------------
    def _split_plain(self, theta, i, chi_max, eps):
        chi_ll, phys_l, phys_r, chi_rr = theta.shape
        theta = np.reshape(theta, [chi_ll * phys_l, phys_r * chi_rr])
        A, S, B = svd(theta, chi_max, full_matrices=False)
        chivC = min(chi_max, int(np.sum(S > eps)))
        self._store_split(A, S, B, i, chi_ll, phys_l, phys_r, chi_rr, chivC)
        self.R[i] = np.eye(phys_l)
        self.R[i + 1] = np.eye(phys_r)

    # -- adaptive bond dimension, optional LBO ---------------------------------
    def _split_adaptive(self, theta, i, chi_max, eps, eps_lbo):
        if eps_lbo is not None:
            # Local basis optimization: project each site onto its optimal basis.
            w_A, v_A = scipy.linalg.eigh(einsum('aIJb,aKJb->IK', theta, theta.conj()))
            n_A = max(10, int(np.sum(w_A > eps_lbo)))
            self.R[i] = v_A[:, np.argsort(w_A)[::-1][:n_A]]
            w_B, v_B = scipy.linalg.eigh(einsum('aIJb, aIKb->JK', theta, theta.conj()))
            n_B = max(10, int(np.sum(w_B > eps_lbo)))
            self.R[i + 1] = v_B[:, np.argsort(w_B)[::-1][:n_B]]
            theta = einsum('KI,LJ,aIJb->aKLb', self.R[i].T.conj(),
                           self.R[i + 1].T.conj(), theta)

        chi_ll, phys_l, phys_r, chi_rr = theta.shape
        theta = np.reshape(theta, [chi_ll * phys_l, phys_r * chi_rr])

        chi_try = int(self.pre_factor * len(self.S[i + 1])) + 10
        A, S, B = svd(theta, chi_try, full_matrices=False)
        chivC = min(chi_max, int(np.sum(S > eps)), chi_try)
        while chivC == chi_try and chi_try < min(*theta.shape):
            chi_try = int(round(self.pre_factor * chi_try))
            A, S, B = svd(theta, chi_try, full_matrices=False)
            chivC = min(chi_max, int(np.sum(S > eps)), chi_try)

        self._store_split(A, S, B, i, chi_ll, phys_l, phys_r, chi_rr, chivC)
        if eps_lbo is None:
            # keep the trivial projectors synced with the current physical dims
            self.R[i] = np.eye(phys_l)
            self.R[i + 1] = np.eye(phys_r)

    def _store_split(self, A, S, B, i, chi_ll, phys_l, phys_r, chi_rr, chivC):
        piv = np.argsort(S)[::-1][:chivC]
        A, S, B = A[:, piv], S[piv], B[piv, :]
        S = S / np.linalg.norm(S)
        A = np.reshape(A, [chi_ll, phys_l, chivC])
        B = np.reshape(B, [chivC, phys_r, chi_rr])
        A = np.tensordot(np.diag(self.S[i] ** (-1)), A, [1, 0])
        A = np.tensordot(A, np.diag(S), [2, 0])
        self.S[i + 1] = S
        self.B[i] = A
        self.B[i + 1] = B

    # -- GPU path (adaptive; optional LBO) -------------------------------------
    def _split_gpu(self, theta, i, chi_max, eps, eps_lbo):  # pragma: no cover
        if eps_lbo is not None:
            w_A, v_A = cp.linalg.eigh(einsum('aIJb,aKJb->IK', theta, theta.conj()))
            n_A = max(10, int(cp.sum(w_A > eps_lbo)))
            R1 = v_A[:, cp.argsort(w_A)[::-1][:n_A]]
            self.R[i] = R1.get()
            w_B, v_B = cp.linalg.eigh(einsum('aIJb, aIKb->JK', theta, theta.conj()))
            n_B = max(10, int(cp.sum(w_B > eps_lbo)))
            R2 = v_B[:, cp.argsort(w_B)[::-1][:n_B]]
            self.R[i + 1] = R2.get()
            theta = einsum('KI,LJ,aIJb->aKLb', R1.T.conj(), R2.T.conj(), theta)
            del R1, R2

        chi_ll, phys_l, phys_r, chi_rr = theta.shape
        theta = cp.reshape(theta, [chi_ll * phys_l, phys_r * chi_rr])
        _mempool.free_all_blocks()

        chi_try = int(self.pre_factor * len(self.S[i + 1])) + 10
        A, S, B = _cusvd(theta, chi_try, full_matrices=False)
        chivC = min(chi_max, int(cp.sum(S > eps).item()), chi_try)
        while chivC == chi_try and chi_try < min(*theta.shape):
            chi_try = int(round(self.pre_factor * chi_try))
            A, S, B = _cusvd(theta, chi_try, full_matrices=False)
            chivC = min(chi_max, int(cp.sum(S > eps).item()), chi_try)
        del theta
        _mempool.free_all_blocks()

        piv = cp.argsort(S)[::-1][:chivC]
        A, S, B = A[:, piv], S[piv], B[piv, :]
        S = S / cp.linalg.norm(S)
        A = cp.reshape(A, [chi_ll, phys_l, chivC])
        B = cp.reshape(B, [chivC, phys_r, chi_rr])
        A = cp.tensordot(cp.diag(self.S[i] ** (-1)), A, [1, 0])
        A = cp.tensordot(A, cp.diag(S), [2, 0])
        self.S[i + 1] = S.get()
        self.B[i] = A.get()
        self.B[i + 1] = B.get()
        if eps_lbo is None:
            self.R[i] = np.eye(phys_l)
            self.R[i + 1] = np.eye(phys_r)
        _mempool.free_all_blocks()


# Backwards-compatible alias for the historical class name.
SpinBoson1D = SpinBosonMPS
