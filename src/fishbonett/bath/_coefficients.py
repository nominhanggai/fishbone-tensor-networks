"""Private finite coefficients obtained by discretizing a :class:`Bath`.

These containers are implementation details shared by Hamiltonian
representations.  Users pass ``Bath`` objects; representations decide which
finite coefficients they require.
"""
from dataclasses import dataclass

import numpy as np

from fishbonett.bath.chain import get_bath_nn_parameters, star_transform
from fishbonett.bath.legendre import get_vn_squared
from fishbonett.bath.lanczos import lanczos


def require_resolved(bath):
    """Validate that automatic bath resolution has already been performed."""
    if bath.domain is None or bath.n_modes is None:
        raise ValueError(
            "a representation needs a resolved Bath; call "
            "bath.resolved(t_max) first")
    return bath


def _readonly(value, dtype):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class _StarCoefficients:
    frequencies: np.ndarray
    couplings: np.ndarray
    transform: np.ndarray | None

    def __post_init__(self):
        frequencies = _readonly(self.frequencies, float)
        couplings = _readonly(self.couplings, float)
        if frequencies.ndim != 1:
            raise ValueError("star frequencies must be one-dimensional")
        if couplings.ndim != 2 or couplings.shape[1] != len(frequencies):
            raise ValueError(
                "star couplings must have shape (n_channels, n_modes)")
        transform = self.transform
        if transform is not None:
            transform = _readonly(transform, float)
            if transform.shape != (len(frequencies), len(frequencies)):
                raise ValueError("star-to-chain transform must span all modes")
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "couplings", couplings)
        object.__setattr__(self, "transform", transform)

    @property
    def n_modes(self):
        return len(self.frequencies)

    @property
    def n_channels(self):
        return self.couplings.shape[0]


@dataclass(frozen=True)
class _ChainCoefficients:
    frequencies: np.ndarray
    hoppings: np.ndarray
    system_coupling: float

    def __post_init__(self):
        frequencies = _readonly(self.frequencies, float)
        hoppings = _readonly(self.hoppings, float)
        if frequencies.ndim != 1:
            raise ValueError("chain frequencies must be one-dimensional")
        if hoppings.shape != (max(len(frequencies) - 1, 0),):
            raise ValueError("a chain needs N frequencies and N-1 hoppings")
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "hoppings", hoppings)
        object.__setattr__(self, "system_coupling", float(self.system_coupling))

    @property
    def n_modes(self):
        return len(self.frequencies)


def star_coefficients(bath):
    """Discretize a resolved bath into private independent-mode data."""
    bath = require_resolved(bath)
    discrete_frequency, discrete_coupling = bath._discrete_star_data()
    if len(discrete_frequency):
        continuum_modes = bath.n_modes - len(discrete_frequency)
        frequency = np.asarray(discrete_frequency, float)
        coupling = np.asarray(discrete_coupling, float)
        if continuum_modes:
            if not bath.continuum_present:
                raise ValueError("resolved mode count exceeds the discrete vibronic modes")
            continuum_frequency, continuum_strength, _ = star_transform(
                bath.spectral_density(), continuum_modes, bath.domain,
                bath.discretizer())
            frequency = np.concatenate((frequency, continuum_frequency))
            coupling = np.concatenate((coupling, continuum_strength))
        if len(frequency) == 1:
            transform = np.ones((1, 1))
        else:
            tri, vectors = lanczos(np.diag(frequency), coupling)
            sign = np.sign(vectors[0, :])
            sign[sign == 0] = 1.0
            transform = np.ascontiguousarray((vectors @ np.diag(sign)).T)
        return _StarCoefficients(
            frequency, coupling[None, :], transform)

    densities = bath.spectral_densities()
    if len(densities) == 1:
        frequencies, strengths, transform = star_transform(
            densities[0], bath.n_modes, bath.domain, bath.discretizer())
        return _StarCoefficients(
            frequencies, np.asarray(strengths)[None, :], transform)

    if bath.discretization != "legendre":
        raise ValueError(
            "a multichannel bath must use the 'legendre' discretization: its "
            "Gauss nodes are shared across channels, whereas measure-adapted "
            "TEDOPA nodes are not")
    frequencies = None
    strengths = []
    for density in densities:
        nodes, weights = get_vn_squared(
            density, bath.n_modes, list(bath.domain))
        nodes = np.asarray(nodes, float)
        if frequencies is None:
            frequencies = nodes
        elif not np.allclose(frequencies, nodes):
            raise ValueError("multichannel densities do not share the mode grid")
        strengths.append(np.sqrt(np.asarray(weights, float) / np.pi))
    return _StarCoefficients(frequencies, np.asarray(strengths), None)


def combined_star_operators(bath, operators):
    """Discretize shared channels and combine them with system operators."""
    star = star_coefficients(bath)
    ops = tuple(operators)
    strengths = star.couplings
    if star.n_channels == 1 and len(ops) > 1:
        strengths = np.repeat(strengths, len(ops), axis=0)
    if len(strengths) != len(ops):
        raise ValueError(
            "the number of spectral densities must match the coupling operators")
    combined = np.einsum(
        "ck,cij->kij", strengths, np.asarray(ops), optimize=True)
    return star.frequencies, combined


def chain_coefficients(bath):
    """Map a resolved single-channel bath to private chain coefficients."""
    bath = require_resolved(bath)
    discrete_frequency, _ = bath._discrete_star_data()
    if len(discrete_frequency):
        star = star_coefficients(bath)
        coupling = star.couplings[0]
        if star.n_modes == 1:
            return _ChainCoefficients(
                star.frequencies.copy(), np.empty(0), abs(coupling[0]))
        tri, _ = lanczos(np.diag(star.frequencies), coupling)
        return _ChainCoefficients(
            np.diagonal(tri).copy(), np.diagonal(tri, -1).copy(),
            np.linalg.norm(coupling))
    densities = bath.spectral_densities()
    if len(densities) != 1:
        raise ValueError(
            "a shared multichannel bath has no single scalar chain mapping")
    frequencies, couplings = get_bath_nn_parameters(
        densities[0], bath.n_modes, list(bath.domain),
        discretizer=bath.discretizer())
    couplings = np.asarray(couplings, float)
    return _ChainCoefficients(
        frequencies, couplings[1:], couplings[0])
