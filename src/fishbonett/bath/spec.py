"""The :class:`Bath` specification -- what a bath *is*, before any propagation.

A :class:`Bath` is a declarative description: a spectral density, a frequency
window, how finely to discretize it, how big each mode's Fock space is, and which
system operator(s) it couples to.  It does no tensor work itself; it is the input
that :class:`fishbonett.models.system_bath.SystemBath` turns into chain parameters.

It lives here rather than in :mod:`fishbonett.models` because it is bath
physics, not simulation machinery: everything it knows about (thermalization,
the discretization scheme, the automatic domain and mode count) is the subject of
this subpackage.

.. rubric:: What's here

======================  ==========================================================
:class:`Bath`           the bath specification (spectral density + discretization)
:func:`thermalize`      T-TEDOPA thermalized density from a ``T=0`` one
======================  ==========================================================

Two conveniences are worth knowing about: leaving ``domain`` unset picks the
window covering 99.9% of the reorganization energy, and leaving ``n_modes`` unset
picks the mode count from the light cone of the propagation time.  Both are
resolved by :meth:`Bath.resolved`, which ``run()`` calls for you.
"""
from dataclasses import dataclass, replace

import numpy as np

from fishbonett.bath.orthpol import make_orthpol_discretizer

__all__ = ["Bath", "thermalize"]


def thermalize(J, beta):
    """T-TEDOPA thermalized spectral density ``J_beta`` (positive on both halves)
    from a zero-temperature ``J(w>0)``.

    Finite temperature is folded into an *effective* zero-temperature density on
    a **signed** frequency axis: ``J_beta(w) = J(|w|) (n_beta(|w|) + 1)`` for
    ``w > 0`` and ``J(|w|) n_beta(|w|)`` for ``w < 0``, with
    ``n_beta = 1/(e^{beta|w|} - 1)``.  The negative half then carries the
    stimulated-emission weight, so a thermal bath can be propagated by the same
    zero-temperature machinery.
    """
    def Jb(w):
        aw = abs(w)
        if aw < 1e-12:
            return 0.0
        nb = 1.0 / np.expm1(beta * aw)
        j = float(J(aw))
        return j * (nb + 1.0) if w > 0 else j * nb
    return Jb


@dataclass
class Bath:
    """A bosonic bath specified by its spectral density and discretization.

    Parameters
    ----------
    J : callable
        Spectral density ``J(w)``.  If ``temperature`` (or ``beta``) is given and
        ``thermalized`` is False, ``J`` is treated as the zero-temperature density
        and thermalized internally.
    domain : (float, float), optional
        Signed bath frequency window.  If omitted, it is chosen automatically as
        the window covering 99.9% of the reorganization energy
        ``lambda = (1/pi) int J(w)/w dw`` (signed when a temperature is set).
    n_modes : int, optional
        Number of discretized modes.  If omitted, it is chosen automatically from
        the light-cone of the interaction-picture chain couplings ``d_j(t)`` for
        the propagation time (so it depends on ``t_max``); see
        :func:`fishbonett.bath.auto.auto_n_modes`.
    phys_dim : int
        The local boson Hilbert-space dimension of each mode.
    temperature, beta : float, optional
        Temperature (or inverse temperature) for thermalization.
    thermalized : bool
        Set True if ``J`` is already the thermalized density.
    discretization : {'legendre', 'orthpol'}
        Bath discretization: uniform-measure Gauss-Legendre star, or the
        measure-adapted ORTHPOL star (resolves IR-divergent / sharply peaked baths).
    extra_breaks, m_per : ORTHPOL quadrature options.
    coupling : (d, d) array, or list of (d, d) arrays
        System operator(s) this bath couples to.  A single operator is an ordinary
        bath.  A **list** of operators makes this a *multichannel single bath*: the
        one bath couples through every operator on shared modes (distinct from
        several independent baths -- the channels cross-correlate).  For a
        multichannel bath ``J`` is either one spectral density (shared) or a list of
        the same length as ``coupling`` (one per channel), and the discretization
        must be ``'legendre'`` (shared Gauss nodes).  Defaults to ``sigma_z``.
    """
    J: object
    domain: tuple = None
    n_modes: int = None
    phys_dim: int = 20
    temperature: float = None
    beta: float = None
    thermalized: bool = False
    discretization: str = "legendre"
    extra_breaks: tuple = ()
    m_per: int = 60
    coupling: object = None

    def _thermalized(self, Jfunc):
        if self.thermalized or (self.temperature is None and self.beta is None):
            return Jfunc
        b = self.beta if self.beta is not None else 1.0 / self.temperature
        return thermalize(Jfunc, b)

    def spectral_density(self):
        """The (thermalized, if applicable) spectral density this bath propagates
        with.  For a multichannel bath this is the *first* channel's density."""
        J0 = self.J[0] if isinstance(self.J, (list, tuple)) else self.J
        return self._thermalized(J0)

    @property
    def is_multichannel(self):
        """True when the bath couples through several operators (``coupling`` is a
        list) -- a single bath with cross-correlated channels, distinct from
        several independent baths."""
        return isinstance(self.coupling, (list, tuple))

    def channels(self):
        """``[(thermalized_J_c, operator_c), ...]`` for a multichannel bath.

        The channels share the same mode grid (same ``domain``/``n_modes``/
        ``discretization``); ``J`` may be one spectral density (shared by all
        channels) or a list of the same length as ``coupling``."""
        ops = list(self.coupling)
        Js = self.J if isinstance(self.J, (list, tuple)) else [self.J] * len(ops)
        if len(Js) != len(ops):
            raise ValueError("a multichannel Bath needs `J` and `coupling` of the "
                             "same length (one spectral density per channel)")
        return [(self._thermalized(Jc), np.asarray(op, complex))
                for Jc, op in zip(Js, ops)]

    def discretizer(self):
        """The star-discretization callable this bath's ``discretization`` selects
        (``None`` means the default Gauss-Legendre star)."""
        if self.discretization == "orthpol":
            return make_orthpol_discretizer(m_per=self.m_per,
                                            extra_breaks=self.extra_breaks)
        if self.discretization == "legendre":
            return None
        raise ValueError(f"unknown discretization {self.discretization!r}")

    def _auto_domain(self):
        from fishbonett.bath.auto import auto_domain
        beta = self.beta if self.beta is not None else (
            1.0 / self.temperature if self.temperature is not None else None)
        Js = self.J if isinstance(self.J, (list, tuple)) else [self.J]
        doms = [auto_domain(Jc, beta=beta) for Jc in Js]          # cover every channel
        return (min(d[0] for d in doms), max(d[1] for d in doms))

    def resolved(self, t_max=None):
        """A copy with automatic ``domain`` / ``n_modes`` filled in.

        ``domain`` (if unset) becomes the window covering 99.9% of the
        reorganization energy; ``n_modes`` (if unset) the light-cone extent of the
        interaction-picture chain couplings up to ``t_max``.  Returns ``self`` when
        both are already given.  Called by ``run`` with the propagation time."""
        domain = self.domain if self.domain is not None else self._auto_domain()
        n_modes = self.n_modes
        if n_modes is None:
            if t_max is None:
                raise ValueError("Bath.n_modes is automatic and needs the "
                                 "propagation time; call from run() (which supplies "
                                 "t_max) or set n_modes explicitly")
            from fishbonett.bath.auto import auto_n_modes
            n_modes = auto_n_modes(self.spectral_density(), domain, t_max,
                                   discretizer=self.discretizer())
        if domain is self.domain and n_modes == self.n_modes:
            return self
        return replace(self, domain=tuple(domain), n_modes=int(n_modes))
