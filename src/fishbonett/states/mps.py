"""Matrix-product-state ansatz for a spin-boson chain (the *state*, separate from
the propagation algorithms in :mod:`fishbonett.evolve`).

:class:`SystemBathMPS` holds the chain tensors ``B``/``S`` and their canonical
form, the per-bond gate store ``U``, and the local-basis-optimization projectors
``R``.  It provides the wavefunction accessors (:meth:`get_theta1`,
:meth:`get_theta2`) and the canonical-form primitive :meth:`split_truncate_theta`
(threshold-controlled SVD, optional LBO, optional CuPy GPU). The
swap-network TEBD sweep that applies the gates lives in
:func:`fishbonett.evolve.tebd.update_bond`; :meth:`update_bond` is a thin wrapper
around it. Truncation behavior is selected per :meth:`update_bond` call:

* **swap** -- ``swap=1`` transposes the two physical legs during the gate, so the
  two sites come back exchanged.  Sweeping outward from site 0 therefore walks the
  system site along the chain, giving it a turn adjacent to every mode: the swap
  network the interaction picture needs, since there every mode couples to the
  system and the interaction is not nearest-neighbour;
* a certified adaptive SVD that grows its trial rank until the truncation error
  is resolved, with exact fallback for unresolved spectra;
* optional **local basis optimization (LBO)** -- pass ``eps_lbo`` to project each
  boson site onto its optimal reduced basis before the SVD;
* optional **GPU** execution through CuPy (``gpu=True``), used only if CuPy is
  importable.

The high-level run's ``svd_backend`` selects automatic, exact, or randomized
resolution for every eligible split.
"""
import warnings

import numpy as np
import scipy.linalg
from fishbonett.contract import contract as einsum

from fishbonett.linalg import threshold_svd
from fishbonett.randomized import _next_seed, current_svd_backend
from fishbonett.states.network import TensorNetwork

# LBO is an auxiliary compression before the bond SVD. Retaining a modest floor
# avoids collapsing a weakly occupied oscillator basis before occupations have
# had time to develop; slicing naturally caps it at the physical dimension.
MIN_LBO_DIMENSION = 10

try:  # optional GPU backend
    import cupy as cp
    from fishbonett._rsvd_cupy import adaptive_svd as _cu_threshold_svd

    _CUPY = True
    _mempool = cp.get_default_memory_pool()

except ImportError:  # pragma: no cover - exercised only with a GPU present
    _CUPY = False


class SystemBathMPS(TensorNetwork):
    """Matrix-product state of a system (spin) site followed by a boson chain.

    This is the ``mps`` state geometry: a linear
    :class:`~fishbonett.states.network.TensorNetwork`. It inherits the topology
    and observables (``rdm``, ``joint_rdm``, ``expectation``) from that base.
    Two things make it its own class rather than a bare ``TreeTensorNetwork``:

    * **leg order.**  Storage is ``(vL, p, vR)`` -- physical in the middle -- which
      the whole 1D TEBD/TDVP stack and every representation's gate layout assume.  The base
      wants ``(bonds..., phys)``, so :meth:`tensor` and :meth:`set_tensor` permute
      between the two.  That permutation is the entire difference; see
      :mod:`fishbonett.states`.
    * **gauge.**  This is Vidal (``Gamma-Lambda``) form: singular values live on the
      bonds in ``S`` and *every* site is already canonical, so
      :meth:`_prepare_for` picks a centre without moving any data, where the
      mixed-canonical tree has to sweep QRs.  The price is that the bond weights sit
      inside :meth:`tensor`, so multi-node contractions go through
      :meth:`_gauged_tensor` instead.  ``R`` additionally carries the
      local-basis-optimization projectors.

    Parameters
    ----------
    pd : sequence of int
        Physical dimensions ``[d_spin, d_boson_0, ..., d_boson_{L-1}]``; the
        system site is first (site 0).
    svd_expansion_factor : float, optional
        Growth factor for the adaptive trial bond dimension (default 1.5).
    """

    def __init__(self, pd, svd_expansion_factor=1.5):
        """Create a ground-product Vidal MPS with the requested dimensions."""
        self.pd_sys = pd[0]
        self.pd_boson = pd[1:]
        self.pre_factor = svd_expansion_factor
        self.B = [self._ground(d) for d in pd]
        self.S = [np.ones([1], float) for _ in pd]
        self.U = [np.zeros(0) for _ in pd[1:]]
        # Optimal-basis projectors for LBO; identity means "no optimization".
        self.R = [np.eye(d) for d in pd]
        # the graph: a path 0 -- 1 -- ... -- n-1, rooted at the system site
        self.n = len(pd)
        self.dims = list(pd)
        self.root = 0
        self.oc = 0
        self._build_topology([(i, i + 1) for i in range(self.n - 1)])

    @staticmethod
    def _ground(d):
        tensor = np.zeros([1, d, 1], dtype=np.complex128)
        tensor[0, 0, 0] = 1.0
        return tensor

    # -- TensorNetwork storage hooks -------------------------------------------
    def neighbours(self, i):
        """``i``'s bond legs in leg order: left neighbour then right neighbour.

        Matches the ``(vL, p, vR)`` storage, so leg ``k`` of :meth:`tensor` is the
        bond to ``neighbours(i)[k]`` -- which is what the base relies on.
        """
        return ([i - 1] if i > 0 else []) + ([i + 1] if i < self.n - 1 else [])

    def tensor(self, i):
        """Node ``i`` as ``(bonds..., phys)``, with ``S`` and ``R`` contracted in.

        The base's observables want the *physical* state around the site, which in
        Vidal form is :meth:`get_theta1`, not the bare right-canonical ``B[i]``.
        End sites keep their dummy dimension-1 bond, so the leg count still matches
        :meth:`neighbours`.
        """
        theta = self.get_theta1(i)                     # (vL, p, vR)
        t = np.moveaxis(theta, 1, -1)                  # (vL, vR, p)
        if i == 0:
            t = t[0]                                   # drop the dummy left bond
        if i == self.n - 1:
            t = t[..., 0, :]                           # drop the dummy right bond
        return t

    def set_tensor(self, i, value):
        """Reject single-tensor writes that would invalidate the Vidal gauge."""
        raise NotImplementedError(
            "SystemBathMPS is in Vidal form: writing a single tensor back would "
            "leave S inconsistent.  Use update_bond / split_truncate_theta, which "
            "maintain the gauge.")

    def _prepare_for(self, i):
        """Choose ``i`` as the orthogonality centre.  Moves **no data**: in Vidal
        form every site is already canonical, so unlike the mixed-canonical tree
        this is a change of view rather than a sweep of QRs."""
        self.oc = i

    def _gauged_tensor(self, i):
        """``i``'s tensor as a multi-site contraction needs it, given the centre.

        At the centre that is :meth:`tensor` (``= Lambda_i B_i``, the full local
        state).  To its right it is the bare right-canonical ``B[i]`` -- taking
        :meth:`tensor` there instead would contract ``Lambda`` a second time on
        every internal bond, since the centre tensor already carries it.
        """
        if i == self.oc:
            return self.tensor(i)
        if i < self.oc:
            # would need the left-isometric Lambda_i Gamma_i, which this Vidal
            # storage can only form as B[i] Lambda_{i+1}^-1.  joint_rdm always puts
            # the centre at the lowest node of the subtree, so it never asks.
            raise NotImplementedError(
                f"no left-isometric form stored for site {i} < oc {self.oc}")
        t = einsum('KI,aIb->aKb', self.R[i], self.B[i])   # (vL, p, vR)
        t = np.moveaxis(t, 1, -1)                        # (vL, vR, p)
        if i == self.n - 1:
            t = t[..., 0, :]                             # drop the dummy right bond
        return t

    # -- wavefunction accessors ------------------------------------------------
    def get_theta1(self, i, gpu=False):
        """One-site wavefunction at site ``i``, legs ``(bond_l, phys, bond_r)``.

        Contracts the stored singular values ``S[i]`` and the local basis ``R[i]``
        back into the right-canonical tensor ``B[i]``, so the result is the
        physical state around that site (the orthogonality centre).  Trace it
        against its conjugate to get the site reduced density matrix.
        """
        if gpu and _CUPY:
            proj = cp.tensordot(cp.diag(self.S[i]), cp.array(self.B[i]), [1, 0])
            return einsum('KI,aIb->aKb', cp.array(self.R[i]), proj)
        proj = np.tensordot(np.diag(self.S[i]), self.B[i], [1, 0])
        return einsum('KI,aIb->aKb', self.R[i], proj)

    def get_theta2(self, i, gpu=False):
        """Two-site wavefunction on bond ``i``, legs ``(bond_l, phys_i, phys_i+1,
        bond_r)`` -- the object a two-site gate acts on before
        :meth:`split_truncate_theta` puts it back."""
        j = i + 1
        if gpu and _CUPY:
            return einsum('aIb,LJ,bJc->aILc', self.get_theta1(i, gpu),
                          cp.array(self.R[j]), cp.array(self.B[j]))
        return einsum('aIb,LJ,bJc->aILc', self.get_theta1(i), self.R[j], self.B[j])

    # -- TEBD step (algorithm lives in fishbonett.evolve.tebd) -----------------
    def update_bond(self, i, chi_max, eps, swap=0, eps_lbo=None, gpu=False):
        """Apply the two-site gate ``self.U[i]`` at bond ``i`` and re-split.

        Thin convenience wrapper: the swap-network TEBD sweep logic lives in
        :func:`fishbonett.evolve.tebd.update_bond` (this state object only holds
        the tensors and their canonical form).  ``swap=1`` transposes the two
        physical legs during the gate; ``eps_lbo`` enables local basis
        optimization before the common threshold-controlled split.
        """
        from fishbonett.evolve.tebd import update_bond as _update_bond
        _update_bond(self, i, chi_max, eps, swap=swap, eps_lbo=eps_lbo,
                     gpu=gpu)

    # -- canonical-form primitives (state operations) --------------------------
    def split_truncate_theta(self, theta, i, chi_max, eps, eps_lbo=None,
                             gpu=False):
        """Split a two-site ``theta`` back into two sites, truncating bond ``i``.

        Restores the canonical form in place: SVD the two-site tensor, keep
        singular values above ``eps`` (relative) and at most ``chi_max`` of them
        (``None`` = unlimited), and write back ``B[i]``, ``B[i+1]`` and ``S[i+1]``.

        ``eps_lbo`` additionally compresses the local boson basis ``R`` before
        the common threshold-controlled bond decomposition.

        ``gpu=True`` without CuPy installed falls back to the CPU path -- the
        results are the same, only slower -- but says so once, because otherwise
        the only symptom is a run that is unaccountably slow.
        """
        if gpu and not _CUPY:
            warnings.warn(
                "gpu=True but CuPy is not importable; falling back to the CPU "
                "path. Install the CuPy wheel matching your CUDA platform to "
                "enable GPU truncation.",
                RuntimeWarning, stacklevel=2)
        if gpu and _CUPY:
            self._split_gpu(theta, i, chi_max, eps, eps_lbo)
        else:
            self._split_cpu(theta, i, chi_max, eps, eps_lbo)

    # -- CPU threshold decomposition, with optional LBO ------------------------
    def _split_cpu(self, theta, i, chi_max, eps, eps_lbo):
        if eps_lbo is not None:
            # Local basis optimization: project each site onto its optimal basis.
            w_A, v_A = scipy.linalg.eigh(einsum('aIJb,aKJb->IK', theta, theta.conj()))
            n_A = max(MIN_LBO_DIMENSION, int(np.sum(w_A > eps_lbo)))
            self.R[i] = v_A[:, np.argsort(w_A)[::-1][:n_A]]
            w_B, v_B = scipy.linalg.eigh(einsum('aIJb, aIKb->JK', theta, theta.conj()))
            n_B = max(MIN_LBO_DIMENSION, int(np.sum(w_B > eps_lbo)))
            self.R[i + 1] = v_B[:, np.argsort(w_B)[::-1][:n_B]]
            theta = einsum('KI,LJ,aIJb->aKLb', self.R[i].T.conj(),
                           self.R[i + 1].T.conj(), theta)

        chi_ll, phys_l, phys_r, chi_rr = theta.shape
        theta = np.reshape(theta, [chi_ll * phys_l, phys_r * chi_rr])

        initial_rank = int(self.pre_factor * len(self.S[i + 1])) + 10
        A, S, B = threshold_svd(
            theta, eps, max_rank=chi_max, initial_rank=initial_rank)
        chivC = S.size

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
            n_A = max(MIN_LBO_DIMENSION, int(cp.sum(w_A > eps_lbo)))
            R1 = v_A[:, cp.argsort(w_A)[::-1][:n_A]]
            self.R[i] = R1.get()
            w_B, v_B = cp.linalg.eigh(einsum('aIJb, aIKb->JK', theta, theta.conj()))
            n_B = max(MIN_LBO_DIMENSION, int(cp.sum(w_B > eps_lbo)))
            R2 = v_B[:, cp.argsort(w_B)[::-1][:n_B]]
            self.R[i + 1] = R2.get()
            theta = einsum('KI,LJ,aIJb->aKLb', R1.T.conj(), R2.T.conj(), theta)
            del R1, R2

        chi_ll, phys_l, phys_r, chi_rr = theta.shape
        theta = cp.reshape(theta, [chi_ll * phys_l, phys_r * chi_rr])
        _mempool.free_all_blocks()

        A, S, B = _cu_threshold_svd(
            theta, eps=eps, max_rank=chi_max,
            backend=current_svd_backend(),
            initial_rank=int(self.pre_factor * len(self.S[i + 1])) + 10,
            seed=_next_seed())
        chivC = S.size
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
