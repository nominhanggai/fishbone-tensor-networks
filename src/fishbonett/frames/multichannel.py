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
  rather than a scalar times one operator.  In this frame that costs nothing:
  there are no mode-mode terms to tridiagonalize, so the star-to-chain map is a
  free choice of orthogonal basis and :meth:`SystemBathMultiChannel.build` takes a
  plain single-vector :func:`~fishbonett.bath.lanczos.lanczos` seeded by one
  channel.  See that method for why the seed does not change the physics.
* finite temperature needs the negative half of the frequency axis, either by
  mirroring the star explicitly (:meth:`SystemBathMultiChannel.__init__`) or by
  folding the weight into the spectral density beforehand
  (:meth:`SystemBathMultiChannel.from_signed_star`, what ``run`` uses).

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

    def __init__(self, pd, coup_mat, freq, temp, h_sys=None, H_add=None):
        """Build from a **positive**-frequency star at temperature ``temp``.

        ``freq`` is mirrored to ``(-freq, freq)`` and each coupling matrix scaled by
        ``sqrt(|temp_factor(temp, w)|)``: explicit thermofield doubling, so the star
        handed in must be the bare ``T = 0`` discretization.  ``temp`` is in
        **kelvin** with frequencies in cm^-1 (:func:`~fishbonett.operators.temp_factor`
        carries Boltzmann's constant), which is *not* the natural-units convention
        :class:`~fishbonett.bath.spec.Bath` uses -- see
        :meth:`from_signed_star` for the entry point that avoids the question.

        Parameters
        ----------
        pd : sequence of int
            ``[d_sys, d_boson, ...]``; ``len(pd) - 1`` sites hold the doubled star,
            so it should be ``2 * len(freq)`` to keep the whole bath.
        coup_mat : sequence of (d, d) array
            One system-space coupling matrix per positive-frequency mode.
        freq : (n,) array
            Positive star frequencies.
        temp : float
            Temperature in kelvin.
        h_sys : (d, d) array, optional
            System Hamiltonian; defaults to zero (a free bath).
        H_add : list, optional
            Extra explicit ``(h_sys_op, h_bath_op, w)`` modes appended to the chain.
        """
        freq = np.array(freq)
        self.temp = temp
        signed = np.concatenate((-freq, freq))
        self._setup(pd,
                    [mat * np.sqrt(np.abs(temp_factor(temp, w)))
                     for mat, w in zip(np.concatenate((coup_mat, coup_mat)), signed)],
                    signed, h_sys, H_add)

    @classmethod
    def from_signed_star(cls, pd, coup_mat, freq, h_sys=None, H_add=None):
        """Build from an already thermofield-doubled (**signed**) star.

        The T-TEDOPA route: temperature is folded into the *spectral density* on a
        signed frequency axis (:func:`~fishbonett.bath.spec.thermalize`), so the
        discretization already carries ``sqrt`` of the thermal weight and there is
        nothing left to double -- ``freq`` may contain negative entries and is used
        as given.  This is the convention the rest of the package uses, and it needs
        no temperature units, so it is what ``run(method="multichannel-ip")`` calls.
        ``len(pd) - 1`` must equal ``len(freq)``: in the interaction picture the
        chain is a *basis rotation* of the star, so dropping sites drops bath.
        """
        self = cls.__new__(cls)
        self.temp = None
        self._setup(pd, list(coup_mat), np.asarray(freq, float), h_sys, H_add)
        return self

    def _setup(self, pd, coup_mat, freq, h_sys, H_add):
        """Shared state for both constructors: ``freq``/``coup_mat`` are final."""
        self.H_add = [] if H_add is None else H_add
        self.pd_sys = pd[0]
        self.pd_boson = pd[1:]
        self.len_boson = len(self.pd_boson)
        self.h_sys = (np.zeros((self.pd_sys, self.pd_sys), complex) if h_sys is None
                      else np.asarray(h_sys, complex))
        if self.h_sys.shape != (self.pd_sys, self.pd_sys):
            raise ValueError(f"h_sys has shape {self.h_sys.shape}, expected "
                             f"{(self.pd_sys, self.pd_sys)} to match pd[0]")
        for mat in coup_mat:
            if np.asarray(mat).shape != (self.pd_sys, self.pd_sys):
                raise ValueError(
                    f"every coup_mat entry must be {(self.pd_sys, self.pd_sys)} to "
                    f"match pd[0]; got {np.asarray(mat).shape}")
        self.freq = np.asarray(freq)
        self.coup_mat = list(coup_mat)
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
        coupling matrix.  With ``inc_sys`` the system term ``delta * h_sys`` is
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
        site = delta * kron(np.eye(d1), self.h_sys)
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

    def build(self, n=0):
        """Rotate the shared star into a chain basis, seeded by channel ``n``.

        Lanczos-tridiagonalizes the star Hamiltonian ``diag(freq)`` with the seed
        vector ``[A_k[n, n]]_k`` -- basis state ``n``'s coupling profile -- and
        stores the star -> chain transform in ``self.coef`` and the chain
        frequencies in ``self.chain_freq``.  Call before :meth:`get_u`.

        **The seed does not change the physics.**  In the interaction picture there
        are no mode-mode terms: the bath enters only through the phases
        ``e(w_k, t, dt)``, and :meth:`get_h2` reassembles site ``m``'s coupling as
        ``sum_k A_k Q[k, m] e(w_k, t, dt)``.  Substituting ``b_k = sum_m Q[k, m] c_m``
        into ``sum_k A_k e_k (x) (b_k + b_k^dag)`` reproduces that exactly for *any*
        orthogonal ``Q``, so the seed only decides how the bath is spread over the
        sites -- i.e. the entanglement and the Fock truncation per site, not the
        answer.  What the seed must not be is **zero**: a coupling whose diagonal
        vanishes in the working basis (e.g. channels ``sigma_x`` and ``sigma_y``
        only) gives ``v0 = 0`` and a meaningless chain, so that is rejected.

        This also means ``len(pd) - 1`` must cover ``len(freq)``.  Keeping fewer
        sites is not a chain truncation as it would be in the Schrodinger picture
        (where distant modes are weakly coupled through the hoppings) -- here it
        simply discards bath modes.
        """
        # `diag(freq)` is real symmetric, so the Krylov basis is real: take the
        # real part of the seed explicitly rather than letting numpy discard the
        # imaginary one silently.  Only the seed's *direction* matters (see above),
        # so this is a choice of basis, not an approximation.
        v0 = np.real(np.array([mat[n, n] for mat in self.coup_mat_np]))
        if not np.any(np.abs(v0) > 1e-14):
            raise ValueError(
                f"the Lanczos seed from channel n={n} is zero: every coupling "
                f"matrix has a vanishing (real part of its) ({n}, {n}) element, so "
                f"no chain can be built from it.  Pick an `n` for which the "
                f"couplings have a nonzero diagonal, or work in a basis where they "
                f"do.")
        chain_freq, Q = lanczos(np.diag(self.freq), v0)
        self.coef = Q
        self.chain_freq = np.diagonal(chain_freq)
        return self

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


