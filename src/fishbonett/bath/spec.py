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

.. rubric:: API

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
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")

    def Jb(w):
        aw = abs(w)
        if aw == 0:
            # The Bose factor diverges at the origin while an Ohmic density
            # vanishes.  Evaluate their finite product at a scale where expm1
            # is accurate.  Averaging the two one-sided formulas removes the
            # vanishing spontaneous-emission term.
            probe = np.sqrt(np.finfo(float).eps) / max(beta, 1.0)
            return float(J(probe)) * (
                1.0 / np.expm1(beta * probe) + 0.5)
        argument = beta * aw
        nb = 0.0 if argument > 700.0 else 1.0 / np.expm1(argument)
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
        Deprecated compatibility field. Bind operators explicitly with
        :meth:`Bath.bind`, or pass ``SystemBath(coupling=...)`` for the
        single-system API. Using this field emits ``DeprecationWarning``; a
        conflicting duplicate is rejected. A list denotes channels sharing one
        mode grid and therefore requires ``'legendre'``.
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
    discrete_frequencies: tuple = ()
    discrete_couplings: tuple = ()
    # A normally constructed Bath describes a continuum.  Discrete-only
    # constructors set this flag to False explicitly.
    continuum_present: bool = True
    physical_reorganization: float = None
    compression_error: float = None
    uncompressed_modes: int = None

    def __post_init__(self):
        densities = self.J if isinstance(self.J, (list, tuple)) else (self.J,)
        if not densities or not all(callable(density) for density in densities):
            raise TypeError("J must be a callable or a non-empty sequence of callables")
        if self.domain is not None:
            if len(self.domain) != 2:
                raise ValueError("domain must contain exactly (lower, upper)")
            lower, upper = map(float, self.domain)
            invalid_width = (
                lower >= upper if self.continuum_present else lower > upper
            )
            if (not np.isfinite(lower) or not np.isfinite(upper)
                    or invalid_width):
                relation = "lower < upper" if self.continuum_present else "lower <= upper"
                raise ValueError(f"domain must be finite with {relation}")
            self.domain = (lower, upper)
        if (not isinstance(self.phys_dim, (int, np.integer))
                or isinstance(self.phys_dim, (bool, np.bool_))
                or self.phys_dim < 1):
            raise ValueError("phys_dim must be a positive integer")
        self.phys_dim = int(self.phys_dim)
        if self.n_modes is not None:
            if (not isinstance(self.n_modes, (int, np.integer))
                    or isinstance(self.n_modes, (bool, np.bool_))
                    or self.n_modes < 1):
                raise ValueError("n_modes must be a positive integer or None")
            self.n_modes = int(self.n_modes)
        if self.temperature is not None and self.beta is not None:
            raise ValueError("provide temperature or beta, not both")
        for name, value in (("temperature", self.temperature),
                            ("beta", self.beta)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.thermalized, (bool, np.bool_)):
            raise TypeError("thermalized must be a boolean")
        if not isinstance(self.continuum_present, (bool, np.bool_)):
            raise TypeError("continuum_present must be a boolean")
        if self.discretization not in {"legendre", "tedopa"}:
            raise ValueError(
                "discretization must be either 'legendre' or 'tedopa'"
            )
        if (not isinstance(self.m_per, (int, np.integer))
                or isinstance(self.m_per, (bool, np.bool_)) or self.m_per < 2):
            raise ValueError("m_per must be an integer of at least 2")
        self.m_per = int(self.m_per)
        breaks = tuple(float(value) for value in self.extra_breaks)
        if not all(np.isfinite(value) for value in breaks):
            raise ValueError("extra_breaks must contain only finite values")
        self.extra_breaks = breaks

        frequencies = np.asarray(self.discrete_frequencies, float)
        couplings = np.asarray(self.discrete_couplings, float)
        if frequencies.ndim != 1 or couplings.ndim != 1:
            raise ValueError("discrete frequencies and couplings must be one-dimensional")
        if frequencies.shape != couplings.shape:
            raise ValueError("discrete frequencies and couplings must have equal length")
        invalid_frequency = (
            frequencies == 0 if self.thermalized else frequencies <= 0
        )
        if (np.any(~np.isfinite(frequencies)) or np.any(invalid_frequency)
                or np.any(~np.isfinite(couplings))):
            frequency_rule = (
                "finite and non-zero" if self.thermalized
                else "finite and positive"
            )
            raise ValueError(
                f"discrete frequencies must be {frequency_rule} and couplings finite"
            )
        self.discrete_frequencies = tuple(map(float, frequencies))
        self.discrete_couplings = tuple(map(float, couplings))

        for name in ("physical_reorganization", "compression_error"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.uncompressed_modes is not None:
            if (not isinstance(self.uncompressed_modes, (int, np.integer))
                    or isinstance(self.uncompressed_modes, (bool, np.bool_))
                    or self.uncompressed_modes < 1):
                raise ValueError("uncompressed_modes must be a positive integer")
            self.uncompressed_modes = int(self.uncompressed_modes)

    @classmethod
    def vibronic(cls, frequencies, huang_rhys, *, continuum=None,
                 temperature=None, beta=None, phys_dim=20, domain=None,
                 n_modes=None, discretization="tedopa", **kwargs):
        """A bath of resolved molecular vibrations and an optional continuum.

        Frequencies are positive angular frequencies in the package's working
        units.  With ``J = pi sum |g_k|^2 delta(w-w_k)``, a Huang--Rhys factor
        gives ``g_k = w_k sqrt(S_k)`` and contributes ``w_k S_k`` to the
        physical reorganization energy.  At finite temperature the discrete
        modes are thermofield doubled internally; callers still supply only the
        positive physical frequencies.
        """
        frequencies = np.asarray(frequencies, float)
        huang_rhys = np.asarray(huang_rhys, float)
        if (frequencies.ndim != 1 or huang_rhys.shape != frequencies.shape
                or frequencies.size == 0):
            raise ValueError(
                "frequencies and huang_rhys must be non-empty 1D arrays of equal length")
        if np.any(frequencies <= 0):
            raise ValueError("vibronic frequencies must be positive")
        if np.any(huang_rhys < 0):
            raise ValueError("Huang-Rhys factors must be non-negative")
        if temperature is not None and beta is not None:
            raise ValueError("provide temperature or beta, not both")
        # Zero-strength modes do not belong to the bath measure.  Degenerate
        # lines couple only through one bright linear combination, so combine
        # their Huang--Rhys factors before the star-to-chain Lanczos mapping.
        # Leaving either case in the grid creates a rank-deficient Krylov space.
        active = huang_rhys > 0
        frequencies = frequencies[active]
        huang_rhys = huang_rhys[active]
        if frequencies.size:
            unique, inverse = np.unique(frequencies, return_inverse=True)
            combined = np.zeros(len(unique), float)
            np.add.at(combined, inverse, huang_rhys)
            frequencies, huang_rhys = unique, combined
        elif continuum is None:
            raise ValueError(
                "a vibronic bath needs a positive Huang-Rhys factor or a continuum")

        strengths = frequencies * np.sqrt(huang_rhys)
        zero = lambda _w: 0.0
        density = continuum if continuum is not None else zero
        physical_reorganization = float(np.sum(frequencies * huang_rhys))
        if continuum is not None and domain is not None:
            from fishbonett.bath.conventions import reorganization_energy
            lo, hi = domain
            if hi > 0:
                physical_reorganization += reorganization_energy(
                    continuum, (max(0.0, lo), hi))
        elif continuum is not None:
            physical_reorganization = None
        if continuum is None and n_modes is not None:
            if (not isinstance(n_modes, (int, np.integer))
                    or isinstance(n_modes, (bool, np.bool_))):
                raise ValueError("n_modes must be a positive integer or None")
            thermalized = bool(kwargs.get("thermalized", False))
            represented = len(frequencies) * (
                2 if (temperature is not None or beta is not None)
                and not thermalized else 1)
            if int(n_modes) != represented:
                raise ValueError(
                    "without a continuum, n_modes must equal the number of "
                    f"represented vibronic modes ({represented})")
        return cls(
            J=density, domain=domain, n_modes=n_modes, phys_dim=phys_dim,
            temperature=temperature, beta=beta,
            discretization=discretization,
            discrete_frequencies=tuple(frequencies),
            discrete_couplings=tuple(strengths),
            continuum_present=continuum is not None,
            physical_reorganization=physical_reorganization,
            **kwargs,
        )

    def _discrete_star_data(self):
        """Effective zero-temperature star modes for the discrete component."""
        frequency = np.asarray(self.discrete_frequencies, float)
        coupling = np.asarray(self.discrete_couplings, float)
        if not len(frequency):
            return frequency, coupling
        if self.thermalized or (self.temperature is None and self.beta is None):
            return frequency, coupling
        beta = self.beta if self.beta is not None else 1.0 / self.temperature
        occupation = 1.0 / np.expm1(beta * frequency)
        return (
            np.concatenate((frequency, -frequency)),
            np.concatenate((coupling * np.sqrt(occupation + 1.0),
                            coupling * np.sqrt(occupation))),
        )

    def reorganization_energy(self):
        """Physical reorganization energy, using positive frequencies only."""
        if self.physical_reorganization is not None:
            return float(self.physical_reorganization)
        from fishbonett.bath.conventions import reorganization_energy
        if self.domain is None:
            raise ValueError("resolve the bath or provide a domain first")
        lo, hi = self.domain
        discrete = 0.0
        frequency = np.asarray(self.discrete_frequencies, float)
        coupling = np.asarray(self.discrete_couplings, float)
        positive = frequency > 0
        if np.any(positive):
            discrete = float(np.sum(coupling[positive] ** 2 / frequency[positive]))
        continuum = (reorganization_energy(self.J, (max(0.0, lo), hi))
                     if self.continuum_present and hi > 0 else 0.0)
        return discrete + continuum

    def correlation(self, times):
        """Correlation function of the resolved finite star representation."""
        from fishbonett.bath._coefficients import star_coefficients
        star = star_coefficients(self)
        times = np.asarray(times, float)
        return np.sum(
            star.couplings[0][None, :] ** 2
            * np.exp(-1j * np.outer(times, star.frequencies)), axis=1)

    def compressed(self, t_max, correlation_tol=1e-3, *, samples=401,
                   max_modes=None):
        """Compress a resolved vibronic measure by correlation-controlled quadrature.

        The first ``m`` Lanczos coefficients define the ``m``-node Gaussian
        quadrature of the complete finite thermal measure.  The smallest ``m``
        whose maximum absolute correlation error, normalized by ``C(0)``, is at
        most ``correlation_tol`` on ``[0, t_max]`` is returned.
        """
        if t_max < 0:
            raise ValueError("t_max must be non-negative")
        if correlation_tol <= 0:
            raise ValueError("correlation_tol must be positive")
        if samples < 2:
            raise ValueError("samples must be at least 2")
        if max_modes is not None and max_modes < 1:
            raise ValueError("max_modes must be positive")
        from fishbonett.bath._coefficients import star_coefficients
        from fishbonett.bath.lanczos import lanczos
        bath = self.resolved(t_max)
        star = star_coefficients(bath)
        if star.n_channels != 1:
            raise ValueError("correlation compression requires one bath channel")
        frequency = star.frequencies
        coupling = star.couplings[0]
        tri, _ = lanczos(np.diag(frequency), coupling)
        times = np.linspace(0.0, float(t_max), int(samples))
        exact = np.sum(coupling[None, :] ** 2
                       * np.exp(-1j * np.outer(times, frequency)), axis=1)
        scale = max(abs(exact[0]), np.finfo(float).tiny)
        limit = min(len(frequency), max_modes or len(frequency))
        selected = None
        for count in range(1, limit + 1):
            values, vectors = np.linalg.eigh(tri[:count, :count])
            strengths = np.linalg.norm(coupling) * vectors[0, :]
            approx = np.sum(strengths[None, :] ** 2
                            * np.exp(-1j * np.outer(times, values)), axis=1)
            error = float(np.max(np.abs(approx - exact)) / scale)
            if error <= correlation_tol:
                selected = (values, np.abs(strengths), error)
                break
        if selected is None:
            raise ValueError(
                f"no quadrature with at most {limit} modes reaches correlation_tol="
                f"{correlation_tol:g}")
        values, strengths, error = selected
        zero = lambda _w: 0.0
        compressed = replace(
            bath, J=zero, domain=(float(values.min()), float(values.max())),
            n_modes=len(values), temperature=None, beta=None, thermalized=True,
            discretization="legendre", discrete_frequencies=tuple(values),
            discrete_couplings=tuple(strengths), continuum_present=False,
            compression_error=error, uncompressed_modes=len(frequency))
        return compressed

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
        discrete_frequency, _ = self._discrete_star_data()
        if self.domain is not None:
            domain = self.domain
        elif self.continuum_present:
            domain = self._auto_domain()
        elif len(discrete_frequency):
            domain = (float(np.min(discrete_frequency)),
                      float(np.max(discrete_frequency)))
        else:
            domain = self._auto_domain()
        n_modes = self.n_modes
        if n_modes is None:
            if not self.continuum_present and len(discrete_frequency):
                n_modes = len(discrete_frequency)
            elif t_max is None:
                raise ValueError("Bath.n_modes is automatic and needs the "
                                 "propagation time; call from run() (which supplies "
                                 "t_max) or set n_modes explicitly")
            else:
                from fishbonett.bath.auto import auto_n_modes
                continuum_modes = max(
                    auto_n_modes(
                        density, domain, t_max,
                        discretizer=self.discretizer(),
                    )
                    for density in self.spectral_densities()
                )
                n_modes = len(discrete_frequency) + continuum_modes
        if n_modes < len(discrete_frequency):
            raise ValueError(
                "n_modes cannot be smaller than the thermally represented discrete modes")
        if (self.continuum_present and len(discrete_frequency)
                and n_modes == len(discrete_frequency)):
            raise ValueError(
                "n_modes must leave at least one represented mode for the "
                "continuum in addition to the discrete vibronic modes"
            )
        if domain is self.domain and n_modes == self.n_modes:
            return self
        return replace(self, domain=tuple(domain), n_modes=int(n_modes))
