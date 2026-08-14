"""The :class:`Bath` specification -- what a bath *is*, before any propagation.

A :class:`Bath` is a declarative description: a spectral density, a frequency
window, how finely to discretize it, and how big each mode's Fock space is.  It
does no tensor work itself. :meth:`Bath.bind` associates it with model-owned
system operators, while Hamiltonian representations discretize it into the
finite star or chain coefficients they require.

It lives here rather than in :mod:`fishbonett.models` because it is bath
physics, not simulation machinery: everything it knows about (thermalization,
the discretization scheme, the automatic domain and mode count) is the subject of
this subpackage. Representations discretize a resolved specification into the
finite coefficients required by their Hamiltonian.

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

from fishbonett.bath.tedopa import make_tedopa_discretizer

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
    discretization : {'legendre', 'tedopa'}
        Bath discretization: uniform-measure Gauss-Legendre star, or the
        measure-adapted TEDOPA star (resolves IR-divergent / sharply peaked baths).
    extra_breaks, m_per : TEDOPA quadrature options.
    coupling : (d, d) array, or list of (d, d) arrays
        Deprecated compatibility field for the Fishbone API, where each bath historically
        carried its edge operator.  New code should bind operators explicitly with
        :meth:`Bath.bind`; ``SystemBath(coupling=...)`` is authoritative for the
        single-system API. Using it emits ``DeprecationWarning`` and a conflicting
        duplicate is rejected. A list denotes channels sharing one mode grid and
        therefore requires ``'legendre'``.
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
        return self.spectral_densities()[0]

    def spectral_densities(self):
        """All thermalized channel densities, independent of system operators.

        A scalar ``J`` is one density and a sequence is one density per channel.
        Keeping this operation independent of ``coupling`` is the first half of
        separating bath physics from the model that couples to it.
        """
        densities = self.J if isinstance(self.J, (list, tuple)) else (self.J,)
        if not densities:
            raise ValueError("Bath.J must contain at least one spectral density")
        return tuple(self._thermalized(density) for density in densities)

    @property
    def is_multichannel(self):
        """Compatibility view of whether ``Bath.coupling`` contains a list.

        Prefer ``bath.bind(operators).is_multichannel``; channel topology belongs
        to the coupled model, not the environment specification alone.
        """
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

    def bind(self, coupling=None, *, default_operator=None,
             validate_legacy=False):
        """Bind this bath to model-owned coupling operator(s).

        New code should keep the operator outside the bath specification and use
        this explicit binding.  ``Bath.coupling`` remains temporarily accepted for
        the Fishbone API and emits ``DeprecationWarning``; it is checked when
        ``validate_legacy`` is requested.
        """
        from fishbonett.bath.coupled import bind_bath
        return bind_bath(self, coupling, default_operator=default_operator,
                         validate_legacy=validate_legacy)

    def discretizer(self):
        """The star-discretization callable this bath's ``discretization`` selects
        (``None`` means the default Gauss-Legendre star)."""
        if self.discretization == "tedopa":
            return make_tedopa_discretizer(m_per=self.m_per,
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
