"""Schroedinger-picture Hamiltonian builders -- nothing rotated out, ``H`` static.

The bare chain-mapped Hamiltonian, in two geometries:

===============================  ================================================
:class:`SystemBathSchrodinger`   one system + one bath, a 1D chain
:class:`FishBoneH`               the **comb**: several electronic sites, each with
                                 an electronic and a vibrational bath chain
===============================  ================================================

Because nothing is transformed away, ``H`` is **time-independent** and strictly
nearest-neighbour.  Two things follow, and they are the reason this frame exists:
its MPO is built **once** (so TDVP conserves energy and pays no per-step rebuild),
and its Trotter gates are likewise built once.  The price is entanglement -- the
state carries the full system-bath correlation, so bond dimensions are the largest
of any frame.  See :doc:`/methods/schrodinger/chain` and :doc:`/methods/index`.

``FishBoneH`` describes the comb geometry evolved by
:class:`fishbonett.states.comb.FishBoneNet`: per chain ``n`` it holds the
electron-bath chain ``eb``, the electronic site ``e``, the vibrational site ``v``
and the vibration-bath chain ``vb``, plus the electronic couplings ``h2ee``
between neighbouring chains.  For an arbitrary (non-comb) tree of sites use
:class:`fishbonett.treebone.TreeFishbone` instead.

.. note::
   This module was called ``frames.hamiltonian``.  It is named for its *frame*
   now, like every other module here (``interaction_picture``, ``polaron``,
   ``multichannel``, ``coolingchain``) -- "hamiltonian" described every one of
   them equally and so distinguished nothing.
"""
import numpy as np
import sys
from numpy import exp
from copy import deepcopy as dcopy

import fishbonett.bath.recurrence as rc

from fishbonett.linalg import eye, kron, expm_gate_sparse as calc_U
from fishbonett.operators import annihilate


def _to_list(x):
    """
    Converts x to [x] if x is a np.ndarray. If x is None,
    convert x(=None) to []. If x is already a list of a
    np.ndarray return x itself. Else if x is not a list of
    just one np.ndarray, raise TypeError.
    :param x: an np.array or a list of one np.ndarray
    :type x:
    :return:
    :rtype:
    """
    if x is None:
        return []
    elif x is list:
        return x
    else:
        return [x]


class FishBoneH:
    """Schroedinger-picture Hamiltonian of the **comb** (fishbone) geometry.

    Describes ``nc`` parallel chains; chain ``n`` runs

        ``eb (electron-bath modes) -- e (electronic site) -- v (vibrational
        site) -- vb (vibration-bath modes)``

    and neighbouring chains are coupled electronically through :attr:`h2ee`.  The
    state that carries it is :class:`fishbonett.states.comb.FishBoneNet`.

    Set the terms as attributes (each is a list, one entry per chain), call
    :meth:`build` to chain-map the baths, then :meth:`get_u` for the two-site
    gates:

    ==============  ===========================================================
    :attr:`h1e`     on-site electronic Hamiltonian
    :attr:`h1v`     on-site vibrational Hamiltonian
    :attr:`h2ee`    electronic coupling between neighbouring chains
    :attr:`h2ev`    electron-vibration coupling within a chain
    :attr:`he_dy`   electronic operator the electron bath couples to
    :attr:`hv_dy`   vibrational operator the vibration bath couples to
    ==============  ===========================================================

    ``h1e``/``h1v`` are normalized through :func:`_to_list`, so a bare array and a
    one-element list are both accepted.

    For an arbitrary tree of sites (rather than this fixed comb) use
    :class:`fishbonett.treebone.TreeFishbone`.
    """

    @property
    def H(self):
        """The assembled two-site Hamiltonians, populated by :meth:`build`."""
        return self._H

    @property
    def _sd(self):
        return self.sd

    @property
    def h1e(self):
        """On-site electronic Hamiltonian, one entry per chain."""
        return [_to_list(x) for x in self._h1e]

    @h1e.setter
    def h1e(self, m):
        self._h1e = m

    @property
    def h1v(self):
        """On-site vibrational Hamiltonian, one entry per chain."""
        return [_to_list(x) for x in self._h1v]

    @h1v.setter
    def h1v(self, m):
        self._h1v = m

    @property
    def h2ee(self):
        """Electronic coupling between neighbouring chains (``nc - 1`` entries)."""
        return self._h2ee

    @h2ee.setter
    def h2ee(self, m):
        self._h2ee = m

    @property
    def h2ev(self):
        """Electron-vibration coupling within each chain."""
        return self._h2ev

    @h2ev.setter
    def h2ev(self, m):
        self._h2ev = m

    @property
    def he_dy(self):
        """Electronic operator the electron bath couples to (one per chain)."""
        return self._he_dy

    @he_dy.setter
    def he_dy(self, m):
        self._he_dy = m

    @property
    def hv_dy(self):
        """Vibrational operator the vibration bath couples to (one per chain)."""
        return self._hv_dy

    @hv_dy.setter
    def hv_dy(self, m):
        self._hv_dy = m

    def __init__(self, pd: np.ndarray, ):
        """
        TODO
        :type pd: nd.ndarray
        :param pd: is a list.
         pD[0] contains physical dimensions of eb, ev, vb on the first chain,
         pD[1] contains physical dimensions of eb, ev, vb on the second chain,
         etc.
        """
        self._pd = pd
        self._nc = len(pd)  # an int
        # pD is a np.ndarray.
        self._ebL = [len(x) for x in self._pd[:, 0]]
        # pD[:,0] is the first column of the array, the eb column
        self._eL = [len(x) for x in self._pd[:, 1]]
        self._vL = [len(x) for x in self._pd[:, 2]]
        self._evL = [x + y for x, y in zip(self._eL, self._vL)]
        # pD[:,2] is the third column of the array, the ev column
        self._vbL = [len(x) for x in pd[:, 3]]
        # pD[:,3] is the fourth column of the array, the vb column
        self._L = [sum(x) for x in zip(self._ebL, self._evL, self._vbL)]
        self._ebD = self._pd[:, 0]
        self._eD = self._pd[:, 1]
        self._vD = self._pd[:, 2]
        self._vbD = self._pd[:, 3]
        # PLEASE NOTE THE SHAPE of pd and nd.array structure.
        # pd = nd.array([
        # [eb0, ev0, vb0], [eb1, ev1, vb1], [eb2, ev2, vb2]
        # ])
        # | eb0 ev0 vb0 |
        # | eb1 ev1 vb1 |
        # | eb2 ev2 vb2 | is the same as the structure depicted in SimpleTTS class.

        self.sd = np.empty([self._nc, 2], dtype=object)
        self.domain = []
        # TODO two lists. w is frequency, k is coupling.
        #  Get them from the function `get_coupling`

        self.w_list = [[[None] * self._ebL[n], [None] * self._vbL[n]] for n in range(self._nc)]
        self.k_list = [[[None] * self._ebL[n], [None] * self._vbL[n]] for n in range(self._nc)]

        # initialize spectral densities.
        for n in range(self._nc):
            if self._ebL[n] > 0:
                self.sd[n, 0] = lambda x: 0  # np.heaviside(x, 1) / 1. * exp(-x / 1)
            elif self._ebL[n] == 0:
                self.sd[n, 0] = None
            if self._vbL[n] > 0:
                self.sd[n, 1] = lambda x: 0  #np.heaviside(x, 1) / 1. * exp(-x / 1)
            elif self._vbL[n] == 0:
                self.sd[n, 1] = None
            else:
                raise SystemError  # TODO tell users what happens.
        # TODO Must have p-leg dims for e and v. Use [] if v not existent.

        # Assign the matrices below according to self.pd
        self._H = []  # list -> all bond Hamiltonians.
        # _H = [ [Heb00, Heb01, ..., Hev0, Hvb00, Hvb01, ..., Hvb0N, Hee0],
        #        [Heb10, Heb11, ..., Hev1, Hvb00, Hvb01, ..., Hvb0N, Hee1],
        #        [Heb00, Heb01, ..., Hev1, Hvb00, Hvb01, ..., Hvb0N, None]
        #      ] in the case of 3 chains.
        self._h1e = [eye(d) for d in self._eD]
        # list -> single Hamiltonian on e site. None as a placeholder if the p-leg is [].
        self._h1v = [eye(d) for d in self._vD]
        # list -> single Hamiltonian on v site. None as a placeholder if the p-leg is [].
        self._h2ee = [kron(eye(m), eye(n)) for (m, n) in zip(self._eD[:-1], self._eD[1:])]
        # list -> coupling Hamiltonian on e and e
        self._h2ev = [kron(eye(m), eye(n)) for (m, n) in
                      zip(self._eD, self._vD)]  # list -> coupling Hamiltonian on e and v
        self._he_dy = [eye(d) for d in self._eD]  # list -> e dynamic variables coupled to eb
        self._hv_dy = [eye(d) for d in self._vD]  # list -> v dynamic variables coupled to vb

    @classmethod
    def get_coupling(self, n, j, domain, g, ncap=20000, discretizer=None):
        """Chain parameters ``(w_list, k_list)`` for ``n`` modes of density ``j``,
        from orthogonal-polynomial recurrence coefficients.  ``k_list[0]`` is the
        system-bath coupling, the rest are mode-mode hoppings."""
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap,
            discretizer=discretizer,
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        return w_list, k_list

    def build_coupling(self, g, ncap=20000, discretizer=None):
        """Chain-map every bath of every chain into ``w_list``/``k_list``.

        Index ``[n][0]`` is chain ``n``'s electron bath and ``[n][1]`` its
        vibration bath; an absent bath (length 0) gets empty lists.
        """
        number_of_chains = self._nc
        for n in range(number_of_chains):
            len_of_eb = self._ebL[n]
            len_of_vb = self._vbL[n]
            if len_of_eb != 0:
                self.w_list[n][0], self.k_list[n][0] = \
                    self.get_coupling(len_of_eb, self.sd[n, 0], self.domain, g,
                                      ncap, discretizer=discretizer)
            else:
                self.w_list[n][0], self.k_list[n][0] = [], []
            if len_of_vb != 0:
                self.w_list[n][1], self.k_list[n][1] = \
                    self.get_coupling(len_of_vb, self.sd[n, 1], self.domain, g,
                                      ncap, discretizer=discretizer)
            else:
                self.w_list[n][1], self.k_list[n][1] = [], []

    def get_h1(self, n, c=None) -> tuple:
        """On-site terms of chain ``n`` as ``(h1eb, h1ev_list, h1vb)``.

        ``h1eb`` and ``h1vb`` are the ``w_i n_i`` terms of the electron-bath and
        vibration-bath modes; ``h1ev_list`` holds the electronic and (if present)
        vibrational site Hamiltonians.  ``c`` is unused.

        :param n: which chain (``0 <= n < nc``)
        :type n: int
        :return: ``(h1eb, h1ev_list, h1vb)``
        :rtype: tuple
        """
        if 0 <= n < self._nc:
            """
            Generates h1eb
            """
            w_list = self.w_list[n][0]
            pd = self._pd[n, 0]
            # Physical dimensions of sites -> on eb of the nth chain.
            # h1eb: EB Hamiltonian list

            h1eb = [None] * len(w_list)
            w_list = w_list[::-1]
            for i, w in enumerate(w_list):
                c = annihilate(pd[i])
                h1eb[i] = w * c.T @ c
            # If w_list = [], so as pd = [],then h1eb becomes []

            """
            Generates h1vb
            """
            w_list = self.w_list[n][1]
            pd = self._pd[n, 3]
            # n -> the nth chain, 0 -> the 3rd element -> w_list for vb.
            h1vb = [None] * len(w_list)  # VB Hamiltonian list on the chain n
            for i, w in enumerate(w_list):
                c = annihilate(pd[i])
                h1vb[i] = w * c.T @ c
            # EV single Hamiltonian list on the chain n
            if self._vD[n] != []:
                h1ev_list = self.h1e[n] + self.h1v[n]
            else:
                h1ev_list = self.h1e[n]
            return h1eb, h1ev_list, h1vb
        else:
            raise ValueError

    def get_h_total(self, n):
        """All two-site Hamiltonians of chain ``n``, as ``[(h, d1, d2), ...]``.

        ``n == -1`` is special: it returns the **inter-chain** electronic bonds
        instead, ``h2ee[i]`` plus the one-body term of electronic site ``i``, with
        the last site's one-body term folded into the final bond (see the comment
        there -- the bookkeeping is easy to get wrong by one site).  Requires
        ``nc > 1``.

        For ``0 <= n <= nc-1`` the bonds run along the chain: electron-bath
        hoppings, the bath-to-electron coupling through :attr:`he_dy`, the
        electron-vibration coupling, and the vibration-bath chain.
        """
        if n == -1 and self._nc > 1:
            e = self._h1e.copy()
            for i, d in enumerate(self._eD[1:]):
                e[i] = kron(e[i], eye(d))
            e[-1] = kron(eye(self._eD[-1]), e[-1])

            ee = self.h2ee
            # Bond i couples electronic sites i and i+1; e[i] = h1e_i (x) I places
            # site i's one-body Hamiltonian on the left of bond i, so sites
            # 0..nc-2 each appear once here and the last site is added on the
            # final bond below via e[-1] = I (x) h1e_{nc-1}.  (Indexing e[n]/ee[n]
            # with the constant n == -1 would drop site 0 and double the last.)
            h_total_ee = [(e[i] + ee[i], self._eD[i][0], self._eD[i + 1][0]) for i in range(self._nc - 1)]
            h_total_ee[-1] = (h_total_ee[-1][0] + e[-1], self._eD[-2][0], self._eD[-1][0])
            return h_total_ee
        elif n == -1 and self._nc == 1:
            raise SystemError

        if 0 <= n <= self._nc - 1:
            h1eb, h1ev, h1vb = self.get_h1(n)
            # Start to generate ev Hamiltonian lists
            pd_eb = self._pd[n, 0]  # pd_eb is a list
            kL = self.k_list[n][0][::-1]
            # kL is a list of k's (coupling constants). Index 0 indicates eb
            if len(kL) != 0 and len(pd_eb) != 0:
                k0, kn = kL[-1], kL[0:-1]
                h2eb = []
                for i, k in enumerate(kn):
                    r0, r1 = pd_eb[i], pd_eb[i + 1]
                    c1 = annihilate(r0);
                    c2 = annihilate(r1)
                    h1 = h1eb[i]
                    h2 = kron(h1, eye(r1)) + k * (kron(c1.T, c2) + kron(c1, c2.T))
                    h2eb.append((h2, r0, r1))
                # The following requires that we must have a e site.
                c0 = annihilate(pd_eb[-1])
                pd_e = self._pd[n, 1][0]  # pd_e is a number
                # TODO: add an condition to determine if the dimensions match.
                h2eb0 = kron(h1eb[-1], np.eye(pd_e)) + k0 * kron((c0 + c0.T), self.he_dy[n])
                h2eb.append((h2eb0, pd_eb[-1], pd_e))
            else:
                h2eb = []

            pd_vb = self._pd[n, 3]  # 3 indicates the vb list
            kL = self.k_list[n][1]
            # kL is a list of k's (coupling constants) 0 indicates eb
            if len(kL) != 0 and len(pd_vb) != 0:
                k0, kn = kL[0], kL[1:]
                c0 = annihilate(pd_vb[0])
                pd_e = self._pd[n, 1][0]

                if self._pd[n, 2] != []:
                    # This condition statement is related to the third
                    # condition statement below. Please also see it.
                    # This statement overlaps the
                    pd_v = self._pd[n, 2][0]  # pd_v is a number
                else:
                    pd_v = pd_e
                pd_vb1 = h1vb[0].shape[0]
                assert pd_vb1 == pd_vb[0]
                h2vb0 = k0 * kron(self._hv_dy[n], c0 + c0.T) + \
                        kron(self._h1v[n], eye(pd_vb1))
                if len(kn) == 0:
                    # Single vb mode: its on-site frequency has no inter-bath
                    # bond to live on, so place it on the E-vb0 bond here.
                    h2vb0 = h2vb0 + kron(eye(pd_v), h1vb[0])
                h2vb = [(h2vb0, pd_v, pd_vb1)]
                for i, k in enumerate(kn):
                    r0, r1 = pd_vb[i], pd_vb[i + 1]
                    c0 = annihilate(r0);
                    c1 = annihilate(r1)
                    h_site1 = kron(h1vb[i], eye(r1))
                    if i == len(kn) - 1:
                        # Last inter-bath bond: also carry the farthest vb mode's
                        # on-site frequency, which is otherwise never applied
                        # (h1vb[i] only covers modes 0..vbL-2).
                        h_site1 = h_site1 + kron(eye(r0), h1vb[i + 1])
                    h_coup = k * (kron(c0.T, c1) + kron(c0, c1.T))
                    h2 = h_site1 + h_coup
                    # h2.shape is (m*n, m*n)
                    h2vb.append((h2, r0, r1))
            else:
                h2vb = []

            h2ev = []
            if self._vbD[n] != [] and self._vD[n] != []:
                h2_ev = self._h2ev[n]
                r0 = self._eD[n][0]
                r1 = self._vD[n][0]
                h2_ev = h2_ev + kron(self._h1e[n], eye(r1))
                h2ev.append((h2_ev, r0, r1))
            if self._vbD[n] == [] and self._vD[n] != []:
                h2_ev = self.h2ev[n]
                r0 = self._eD[n][0]
                r1 = self._vD[n][0]
                h2_ev = h2_ev + kron(self._h1e[n], eye(r1)) + kron(eye(r0), self._h1v[n])
                h2ev.append((h2_ev, r0, r1))
            if self._vbD[n] != [] and self._vD[n] == []:
                # A special case, where the v site is overlapped with the e site.
                # b-b-b--E(V)-b-b-b-b
                # In this case, the dynamical operator of V becomes the second dynamical
                # operator of E. This second dynamical operator of the E site serves as
                # the operator belonging to the E site that couples with the right-hand-side bath.
                # One need set the 1-site Hamiltonian h1v identical to h1e.
                return h2eb + h2ev + h2vb
            elif h2eb != []:
                he = self._h1e[n]
                d_of_e = he.shape[0]
                # The e-site one-body Hamiltonian is added on the eb bond only
                # when there is no backbone (nc == 1); for nc > 1 every site
                # energy is carried once by the backbone bond (see
                # get_h_total(-1)), so adding it here too would double count it.
                if self._nc == 1:
                    h2_eb0 = h2eb[-1][0] + kron(eye(self._ebD[n][-1]), he)
                else:
                    h2_eb0 = h2eb[-1][0]
                h2eb[-1] = (h2_eb0, self._ebD[n][-1], d_of_e)
            return h2eb + h2ev + h2vb
        else:
            raise ValueError

    def build(self, g, ncap=20000, discretizer=None):
        """Chain-map every bath and assemble :attr:`H`.

        Call after setting the ``h1e``/``h1v``/``h2ee``/``h2ev``/``he_dy``/
        ``hv_dy`` terms and the spectral densities, and before :meth:`get_u`.
        With more than one chain the inter-chain electronic bonds
        (``get_h_total(-1)``) are appended to each chain's bond list.
        """
        self.build_coupling(g, ncap, discretizer=discretizer)
        H = []
        for n in range(self._nc):
            h = self.get_h_total(n)
            H.append(h)
        if self._nc > 1:
            h2_ee = self.get_h_total(-1)
            for n in range(self._nc - 1):
                H[n].append(h2_ee[n])
        self._H = H

    def get_u(self, dt):
        """Two-site gates ``exp(-i dt h)`` for every bond, shaped like :attr:`H`.

        Each gate has legs ``(d1, d2, d1*, d2*)``.  ``H`` is time-independent in
        this frame, so these are built **once** and reused for every step.
        """
        U = dcopy(self.H)
        for i, r in enumerate(self.H):
            for j, s in enumerate(r):
                h = self.H[i][j][0]
                u = calc_U(h, dt).toarray()
                r0 = r1 = self.H[i][j][1]  # physical dimension for site A
                s0 = s1 = self.H[i][j][2]  # physical dimension for site B
                u = u.reshape([r0, s0, r1, s1])
                U[i][j] = u
        return U


class SystemBathSchrodinger:
    """Schroedinger-picture chain Hamiltonian builder: arbitrary system + harmonic bath.

    Builds the explicit (time-independent) chain Hamiltonian and its two-site
    Trotter gates directly, without moving to the interaction picture (any
    Hermitian system, not just a spin -- the "spin-boson" name is historical).
    Exported publicly as :class:`fishbonett.SystemBathSchrodinger`.
    """

    def __init__(self, pd):
        self.pd_sys = pd[0]
        self.pd_boson = pd[1:]
        self.sd = lambda x: np.heaviside(x, 1) / 1. * exp(-x / 1)
        self.domain = [0, 1]
        self.he_dy = np.eye(self.pd_sys)
        self.h1e = np.eye(self.pd_sys)
        self.k_list = []
        self.w_lsit = []
        self.H = []

    def get_coupling(self, n, j, domain, g, ncap=20000):
        """Chain parameters ``(w_list, k_list)`` for ``n`` modes of density ``j``,
        from orthogonal-polynomial recurrence coefficients.  ``k_list[0]`` is the
        system-bath coupling, the rest are mode-mode hoppings."""
        alphaL, betaL = rc.recurrenceCoefficients(
            n - 1, lb=domain[0], rb=domain[1], j=j, g=g, ncap=ncap
        )
        w_list = g * np.array(alphaL)
        k_list = g * np.sqrt(np.array(betaL))
        k_list[0] = k_list[0] / g
        return w_list, k_list

    def build_coupling(self, g, ncap):
        """Chain-map ``self.sd`` over ``self.domain`` into ``w_list``/``k_list``."""
        n = len(self.pd_boson)
        self.w_list, self.k_list = self.get_coupling(n, self.sd, self.domain, g, ncap)

    def get_h1(self):
        """On-site terms in chain order: ``w_i n_i`` per bath mode, then the
        system Hamiltonian ``h1e`` last."""
        w_list = self.w_list[::-1]
        h1 = []
        for i, w in enumerate(w_list):
            c = annihilate(self.pd_boson[i])
            h1.append(w * c.T @ c)
        h1.append(self.h1e)
        return h1

    def get_h2(self):
        """Two-site Hamiltonians ``[(h, d1, d2), ...]``, chain order, system last.

        Each mode-mode bond carries the hopping ``k (b_i^dag b_{i+1} + h.c.)``
        plus the left site's on-site term; the final bond carries the system-bath
        coupling ``k0 (b + b^dag) (x) he_dy`` plus both remaining on-site terms.
        Use :meth:`get_h2_only` for the couplings without the on-site parts.
        """
        h1 = self.get_h1()
        k_list = self.k_list[::-1]
        k0 = k_list[-1]
        k_list = k_list = k_list[0:-1]
        h2 = []
        for i, k in enumerate(k_list):
            d1 = self.pd_boson[i]
            d2 = self.pd_boson[i + 1]
            c1 = annihilate(d1)
            c2 = annihilate(d2)
            coup = k * (kron(c1.T, c2) + kron(c1, c2.T))
            site = kron(h1[i], np.eye(d2))
            h2.append((coup + site, d1, d2))
        d1 = self.pd_boson[-1]
        d2 = self.pd_sys
        c0 = annihilate(d1)
        coup = k0 * kron(c0 + c0.T, self.he_dy)
        site = kron(h1[-2], np.eye(d2)) + kron(np.eye(d1), h1[-1])
        h20 = coup + site
        h2.append((h20, d1, d2))
        return h2

    def get_h2_only(self):
        """The bond couplings alone -- :meth:`get_h2` without the on-site terms.

        Returns bare matrices rather than ``(h, d1, d2)`` tuples.  Useful when the
        on-site parts are applied separately (e.g. in a split where the free
        evolution is treated exactly).
        """
        k_list = self.k_list[::-1]
        k0 = k_list[-1]
        k_list = k_list[0:-1]
        h2 = []
        for i, k in enumerate(k_list):
            d1 = self.pd_boson[i]
            d2 = self.pd_boson[i + 1]
            c1 = annihilate(d1)
            c2 = annihilate(d2)
            coup = k * (kron(c1.T, c2) + kron(c1, c2.T))
            h2.append(coup)
        d1 = self.pd_boson[-1]
        d2 = self.pd_sys
        c0 = annihilate(d1)
        coup = k0 * kron(c0 + c0.T, self.he_dy)
        h20 = coup
        h2.append(h20)
        return h2

    def build(self, g, ncap=20000):
        """Chain-map the bath.  Call before :meth:`get_u`."""
        self.build_coupling(g, ncap)


    def get_u(self, dt):
        """Two-site gates ``exp(-i dt h)`` for every bond (sparse matrices).

        ``H`` is time-independent in this frame, so these are built **once** and
        reused for every step -- the practical advantage of the Schroedinger
        picture over the interaction picture.
        """
        self.H = self.get_h2()
        U = [0]*len(self.H)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = calc_U(h, dt)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # u = u.reshape([r0, s0, r1, s1])
            U[i] = u
        return U

def exponential(h_d1_d2, dt):
    """``exp(-i dt h)`` for one ``(h, d1, d2)`` bond entry, as a sparse matrix.

    A convenience for exponentiating a single bond outside the whole-chain
    :meth:`FishBoneH.get_u`; the physical dimensions are carried along only so
    that the caller can reshape the result to ``(d1, d2, d1*, d2*)``.
    """
    h, d1, d2 = h_d1_d2
    u = calc_U(h, dt)
    return u



if __name__ == "__main__":
    a = [3, 3, 3]
    b = [2]
    pd = np.array([[a, b, b, a], [a, b, b, a]], dtype=object)
    tri = FishBoneH(pd)
    # tri.H
