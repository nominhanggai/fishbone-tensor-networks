"""Star and chain interaction representations of a harmonic bath.

The construction follows the mathematical order directly:

1. discretize the bath into independent star modes ``a_k``;
2. rotate with respect to ``sum_k omega_k a_k^dag a_k``;
3. retain the star operators for ``interaction-star``, or apply the
   star-to-chain transform for ``interaction-chain``.

Consequently the star coefficients are ``g_k exp(-i omega_k t)`` and the chain
coefficients are ``d_n(t) = sum_k U[n,k] g_k exp(-i omega_k t)``.  Diagonalizing
a finite chain can recover the same star quadrature, but that is a numerical
discretization route rather than the definition of this representation.

This module contains no TEBD, TDVP, MPO, or tensor-network state logic.  Numerical
encodings live in :mod:`fishbonett.encodings`.
"""

import numpy as np

from fishbonett.bath.chain import star_transform
from fishbonett.bath.compiled import StarBath
from fishbonett.bath.conventions import integrated_free_phase
from fishbonett.linalg import kron
from fishbonett.operators import annihilate
from fishbonett.system import check_operator

__all__ = ["InteractionRepresentation"]


class InteractionRepresentation:
    """The ``interaction-star`` or ``interaction-chain`` Hamiltonian.

    Parameters
    ----------
    pd
        ``[d_system, d_mode, ...]``.
    representation
        Exactly ``"interaction-star"`` or ``"interaction-chain"``.
    h_sys, coupling
        Hermitian system Hamiltonian and coupling operator.
    compiled_star
        Preferred input: a finite star discretization and its optional
        star-to-chain transform.  The ``sd``/``domain`` route remains available to
        low-level research scripts.
    """

    names = frozenset({"interaction-star", "interaction-chain"})

    def __init__(self, pd, *, representation, h_sys, coupling, sd=None,
                 domain=None, discretizer=None, compiled_star=None):
        if representation not in self.names:
            raise ValueError(
                "representation must be 'interaction-star' or "
                "'interaction-chain'")
        self.name = representation
        self.pd_sys = int(pd[0])
        self.pd_boson = [int(value) for value in pd[1:]]
        self.len_boson = len(self.pd_boson)
        if not self.pd_boson:
            raise ValueError("pd must include at least one bath mode")
        self.h_sys = check_operator(h_sys, "h_sys", self.pd_sys)
        self.coupling = check_operator(coupling, "coupling", self.pd_sys)
        if compiled_star is None and (sd is None or domain is None):
            raise ValueError(
                "provide compiled_star or both sd and domain")
        self.sd = sd
        self.domain = None if domain is None else tuple(domain)
        self.discretizer = discretizer
        self.compiled_star = compiled_star
        self.frequencies = None
        self.star_couplings = None
        self.star_to_chain = None

    @property
    def static(self):
        return False

    def build(self):
        """Prepare the finite star data and optional star-to-chain transform."""
        if self.compiled_star is None:
            frequencies, couplings, transform = star_transform(
                self.sd, self.len_boson, self.domain, self.discretizer)
            star = StarBath(
                frequencies, np.asarray(couplings)[None, :],
                self.pd_boson[0], transform)
        else:
            star = self.compiled_star
        if star.n_channels != 1:
            raise ValueError("an interaction representation requires one channel")
        if star.n_modes != self.len_boson:
            raise ValueError(
                f"compiled star has {star.n_modes} modes but pd describes "
                f"{self.len_boson}")
        if any(size != star.phys_dim for size in self.pd_boson):
            raise ValueError(
                "compiled star phys_dim does not match the mode dimensions")
        if self.name == "interaction-chain" and star.chain_transform is None:
            raise ValueError(
                "interaction-chain requires a star-to-chain transform")
        self.frequencies = star.frequencies
        self.star_couplings = star.couplings[0]
        self.star_to_chain = star.chain_transform
        return self

    def _express(self, star_values):
        values = np.asarray(star_values, complex)
        if self.name == "interaction-star":
            return values
        return self.star_to_chain @ values

    def coefficients(self, t):
        """Instantaneous coupling coefficients in this representation."""
        phases = np.exp(-1j * self.frequencies * float(t))
        return self._express(self.star_couplings * phases)

    def interval_coefficients(self, t, delta):
        """Couplings integrated over ``[t, t + delta]``."""
        phases = np.array([
            integrated_free_phase(omega, t, delta)
            for omega in self.frequencies
        ])
        return self._express(self.star_couplings * phases)

    def two_site_hamiltonians(self, t, delta, include_system=True):
        """One interval-integrated system–mode Hamiltonian per bath mode.

        Matrices use ``(mode, system)`` ordering.  They are representation data;
        :mod:`fishbonett.encodings.gates` decides whether and how to exponentiate
        or arrange them on a tensor-network state.
        """
        out = []
        for dimension, amplitude in zip(
                self.pd_boson, self.interval_coefficients(t, delta)):
            destroy = annihilate(dimension)
            bath_operator = (
                amplitude * destroy
                + np.conj(amplitude) * destroy.conj().T
            )
            out.append((
                kron(bath_operator, self.coupling),
                dimension,
                self.pd_sys,
            ))
        if include_system:
            dimension = self.pd_boson[0]
            system_term = delta * kron(np.eye(dimension), self.h_sys)
            out[0] = (out[0][0] + system_term, dimension, self.pd_sys)
        return out
