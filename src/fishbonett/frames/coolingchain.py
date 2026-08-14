"""Cooling-chain frame: a finite-temperature bath as a *gauged* zero-temperature one.

An alternative to thermofield doubling for finite temperature.  Instead of
mirroring the spectral density onto a signed frequency axis (what
:func:`fishbonett.bath.thermalize` does), each chain mode is reweighted by a
``betaOmega`` gauge: the annihilation and creation halves of the system-bath
coupling are scaled by ``e^{+betaOmega}`` and ``e^{-betaOmega}``, so the chain is
progressively "cooled" along its length while thermal weight is carried by the
non-unitary gauge instead of by extra modes.

The gauge makes the propagation **non-unitary**, so the state's norm is not the
physical one: observables must be read through the matching heating operators
``exp(2 betaOmega n_i)``, which is what :meth:`SystemBathCoolingChain.get_rdm`
does (renormalizing as it contracts).  Reading the RDM the ordinary way would give
the wrong answer.

Unlike the other frames this class *is* the state -- it subclasses
:class:`~fishbonett.states.mps.SystemBathMPS` rather than building gates for a
separate state object.  It is exploratory rather than part of the ``method=``
dispatch; see :class:`~fishbonett.frames.interaction_picture.SystemBathIP` for
the maintained finite-temperature route.
"""
import numpy as np
import scipy

from fishbonett.contract import contract as einsum
from fishbonett.bath.chain import get_coupling
from fishbonett.system import check_operator
from fishbonett.linalg import kron, expm_gate_sparse
from fishbonett.operators import temp_factor, annihilate

from fishbonett.states.mps import SystemBathMPS


class SystemBathCoolingChain(SystemBathMPS):
    """Cooling-chain builder: system + harmonic bath, dissipative cooling ansatz.

    Extends the 1D :class:`~fishbonett.states.mps.SystemBathMPS` engine with a
    ``betaOmega`` cooling gauge: each bath mode carries a heating operator so the
    chain is progressively cooled, and :meth:`get_rdm` reads the system reduced
    density matrix through those operators.  Everything the frame needs is given at
    construction; :meth:`build` then does the chain mapping.

    Parameters
    ----------
    pd : sequence of int
        Physical dimensions ``[d_sys, d_boson, ...]`` -- the system on site 0, one
        entry per chain mode after it.
    h_sys : (d, d) array
        System Hamiltonian.
    coupling : (d, d) array
        The Hermitian system-bath coupling.
    sd : callable
        Spectral density ``J(w)``.  Pass the **bare** ``T = 0`` density: the gauge,
        not the density, is what carries the thermal weight in this frame.
    domain : (float, float)
        Frequency window to chain-map over.
    betaOmega : float, optional
        The cooling gauge strength.
    g : float, optional
        Frequency-axis rescaling passed to the recurrence coefficients.
    ncap : int, optional
        Accepted and ignored -- the chain mapping's accuracy is set by the number
        of modes.  Kept because callers pass it; see
        :func:`fishbonett.bath.chain.get_coupling`.
    discretizer : callable, optional
        Quadrature for the star discretization; ``None`` is Gauss-Legendre.  This
        frame could not accept one until the chain mapping was shared with
        :func:`fishbonett.bath.chain.get_coupling` -- its private copy had dropped
        the argument.
    """

    def __init__(self, pd, *, h_sys, coupling, sd, domain, betaOmega=2.,
                 g=1.0, ncap=20000, discretizer=None):
        super().__init__(pd)
        self.len_boson = len(self.pd_boson)
        if self.len_boson == 0:
            raise ValueError("pd must contain at least one boson mode after the "
                             "system dimension")
        self.h_sys = check_operator(h_sys, "h_sys", self.pd_sys)
        self.coupling = check_operator(coupling, "coupling", self.pd_sys)
        self.sd = sd
        self.domain = list(domain)
        self.betaOmega = betaOmega
        self.g = g
        self.ncap = ncap
        self.discretizer = discretizer
        self.k_list = []
        self.w_list = []
        self.H = []


    def get_rdm(self):
        """System reduced density matrix, read through the heating operators.

        The cooling gauge is non-unitary, so the plain contraction of the MPS
        would not give the physical RDM.  This contracts the bath sites against
        ``exp(2 betaOmega n_i)`` instead, renormalizing at each site to keep the
        result finite, and traces out the bath.
        """
        # contract outward from the system (site 0) through the boson chain
        theta = self.get_theta1(0)    # system site
        rho = einsum('PiQ,PjL->iQjL', theta, theta.conj())
        # trace through each boson site with its heating operator
        for i in range(self.len_boson):
            bi = self.B[i + 1]        # boson site i+1 in the MPS
            rho = einsum('iQjL,QkR,kl,LlS->iRjS', rho, bi, self.heating_op[i], bi.conj())
            rho = rho / einsum('iRiR', rho)
        rho = einsum('iRjR->ij', rho)
        return rho

    def build_coupling(self):
        """Chain-map ``self.sd`` over ``self.domain`` and store ``w_list``/``k_list``.

        Uses :func:`fishbonett.bath.chain.get_coupling`.  This class used to carry
        its own copy, which had silently lost the ``discretizer`` argument, so a
        measure-adapted (TEDOPA) star was unreachable from this frame even though
        every other frame supported one.
        """
        n = len(self.pd_boson)
        self.w_list, self.k_list = get_coupling(
            self.sd, n, self.domain, self.g, self.ncap,
            discretizer=self.discretizer)

    def build(self):
        """Chain-map the bath, assemble the two-site Hamiltonians, and build the
        normalized heating operators ``exp(2 betaOmega n_i)`` used by
        :meth:`get_rdm`.  The one expensive step; call before :meth:`get_u`."""
        self.build_coupling()
        self.H = self.get_h2()
        self.heating_op = [scipy.linalg.expm(2 * self.betaOmega # * np.sign(freq[i])
                                             * annihilate(d).T @ annihilate(d)) for i, d in
                           enumerate(self.pd_boson)]
        self.heating_op = [op / np.linalg.norm(op) for op in self.heating_op]
        return self

    def get_h1(self):
        """On-site terms: system first, then bath modes in chain order."""
        h1 = [self.h_sys]
        for i in range(self.len_boson):
            c = annihilate(self.pd_boson[i])
            h1.append(self.w_list[i] * c.T @ c)
        return h1

    def get_h2(self):
        """Two-site Hamiltonians ``[(h, d1, d2), ...]`` along the chain.

        Bond 0 (system to c0) carries the cooling gauge: the coupling is
        ``k0 (e^{betaOmega} b + e^{-betaOmega} b^dag) (x) coupling``, plus the
        system's and ``c0``'s on-site terms.  Bond ``i`` then carries the
        ``c_{i-1}``-``c_i`` hopping plus ``c_i``'s on-site term -- the **right**
        leg, so each mode's frequency is placed exactly once.
        """
        h1 = self.get_h1()          # [h_sys, w_0 n_0, w_1 n_1, ...]
        k0 = self.k_list[0]
        h2 = []
        d1 = self.pd_sys
        d2 = self.pd_boson[0]
        annih = np.exp(self.betaOmega)
        creat = np.exp(-1 * self.betaOmega)
        c0 = annihilate(d2)
        coup = k0 * kron(self.coupling, annih*c0 + creat*c0.T)
        site = kron(h1[0], np.eye(d2)) + kron(np.eye(d1), h1[1])
        h2.append((coup + site, d1, d2))
        for i in range(1, self.len_boson):
            d1 = self.pd_boson[i - 1]
            d2 = self.pd_boson[i]
            c1 = annihilate(d1)
            c2 = annihilate(d2)
            coup = self.k_list[i] * (kron(c1.T, c2) + kron(c1, c2.T))
            site = kron(np.eye(d1), h1[i + 1])      # c_i's on-site term
            h2.append((coup + site, d1, d2))
        return h2


    def get_u(self, dt):
        """Two-site gates ``exp(-i dt h)`` with legs ``(d1, d2, d1*, d2*)``.

        The Hamiltonian is time-independent in this frame, so unlike the
        interaction-picture builders these gates are built **once**.  They are not
        unitary -- see the module docstring.
        """
        U = [0]*len(self.H)
        for i, h_d1_d2 in enumerate(self.H):
            h, d1, d2 = h_d1_d2
            u = expm_gate_sparse(h, dt)
            r0 = r1 = d1  # physical dimension for site A
            s0 = s1 = d2  # physical dimension for site B
            # u = u.reshape([r0, s0, r1, s1])
            U[i] = u.toarray().reshape(r0,s0,r1,s1)
        return U
