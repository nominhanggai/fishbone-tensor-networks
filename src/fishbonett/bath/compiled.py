"""Compiled finite representations of a continuous bosonic bath.

``Bath`` is user input: a spectral density plus resolution settings.  Tensor
network representations should not repeatedly interpret that input themselves, so this
module owns the boundary where a resolved specification becomes numerical data.

The compiled objects deliberately contain no system operator.  Bath coefficients
and a system--bath coupling are different pieces of the physical problem; combining
them is the responsibility of :class:`fishbonett.bath.coupled.CoupledBath`.
"""
from dataclasses import dataclass

import numpy as np

from fishbonett.bath.chain import get_bath_nn_paras, star_transform
from fishbonett.bath.conventions import reorganization_energy
from fishbonett.bath.legendre import get_vn_squared

__all__ = [
    "StarBath", "ChainBath", "PolaronBath", "compile_star", "compile_chain",
    "compile_polaron",
]


def _readonly(value, dtype):
    """Return an owned, read-only ndarray for immutable compiled coefficients."""
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class StarBath:
    """Independent oscillator modes on a common frequency grid.

    ``couplings`` has shape ``(n_channels, n_modes)``.  It contains scalar mode
    strengths only; :meth:`combine` binds those strengths to system operators.
    For a single channel, ``chain_transform`` is the orthogonal star-to-chain
    transform used by the ``interaction-chain`` representation.
    """

    frequencies: np.ndarray
    couplings: np.ndarray
    phys_dim: int
    chain_transform: np.ndarray | None = None

    def __post_init__(self):
        freq = _readonly(self.frequencies, float)
        coup = _readonly(self.couplings, float)
        if freq.ndim != 1:
            raise ValueError("star frequencies must be one-dimensional")
        if coup.ndim != 2 or coup.shape[1] != len(freq):
            raise ValueError(
                "star couplings must have shape (n_channels, n_modes)")
        transform = self.chain_transform
        if transform is not None:
            transform = _readonly(transform, float)
            if transform.shape != (len(freq), len(freq)):
                raise ValueError("chain_transform must be square over the modes")
        object.__setattr__(self, "frequencies", freq)
        object.__setattr__(self, "couplings", coup)
        object.__setattr__(self, "chain_transform", transform)

    @property
    def n_modes(self):
        return len(self.frequencies)

    @property
    def n_channels(self):
        return self.couplings.shape[0]

    def combine(self, operators):
        """Return one system-space coupling matrix per star mode.

        Mode ``k`` carries ``sum_c g[c,k] O[c]``.  Keeping this operation here
        makes the shared-grid invariant explicit and gives every representation exactly the
        same multichannel discretization.
        """
        ops = np.asarray(tuple(operators), complex)
        if ops.ndim != 3 or ops.shape[0] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} coupling operators with a common "
                "matrix shape; got array shape " + repr(ops.shape))
        if ops.shape[1] != ops.shape[2]:
            raise ValueError("coupling operators must be square")
        return np.einsum("ck,cij->kij", self.couplings, ops)

    def interaction_couplings(self, t, *, representation="interaction-star"):
        """Single-channel interaction-picture coefficients at time ``t``.

        The result is in natural mode order.  State-layout reversal, where needed,
        belongs to the representation or geometry rather than the bath compiler.
        """
        if self.n_channels != 1:
            raise ValueError("scalar interaction couplings require one channel")
        values = self.couplings[0] * np.exp(-1j * self.frequencies * t)
        if representation == "interaction-star":
            return values
        if representation == "interaction-chain":
            if self.chain_transform is None:
                raise ValueError("this star has no chain transform")
            return self.chain_transform @ values
        raise ValueError(
            "representation must be 'interaction-star' or 'interaction-chain'")


@dataclass(frozen=True)
class ChainBath:
    """Nearest-neighbour chain coefficients for one bath channel."""

    frequencies: np.ndarray
    hoppings: np.ndarray
    system_coupling: float
    phys_dim: int

    def __post_init__(self):
        freq = _readonly(self.frequencies, float)
        hopping = _readonly(self.hoppings, float)
        if freq.ndim != 1 or hopping.shape != (max(len(freq) - 1, 0),):
            raise ValueError("a chain needs N frequencies and N-1 hoppings")
        object.__setattr__(self, "frequencies", freq)
        object.__setattr__(self, "hoppings", hopping)
        object.__setattr__(self, "system_coupling", float(self.system_coupling))

    @property
    def n_modes(self):
        return len(self.frequencies)


@dataclass(frozen=True)
class PolaronBath:
    """Finite star and chain data for the polaron representations."""

    star: StarBath
    chain: ChainBath
    reorganization_energy: float

    def __post_init__(self):
        object.__setattr__(self, "reorganization_energy",
                           float(self.reorganization_energy))


def _require_resolved(bath):
    if bath.domain is None or bath.n_modes is None:
        raise ValueError(
            "compile a resolved Bath: call bath.resolved(t_max) first")


def compile_star(bath):
    """Compile a resolved ``Bath`` into a :class:`StarBath`."""
    _require_resolved(bath)
    densities = bath.spectral_densities()
    if len(densities) == 1:
        freq, strength, transform = star_transform(
            densities[0], bath.n_modes, bath.domain, bath.discretizer())
        return StarBath(freq, np.asarray(strength)[None, :], bath.phys_dim,
                        transform)

    if bath.discretization != "legendre":
        raise ValueError(
            "a multichannel bath must use the 'legendre' discretization: its "
            "Gauss nodes are shared across channels, whereas measure-adapted "
            "TEDOPA nodes are not")
    frequencies = None
    strengths = []
    for density in densities:
        freq, v_sq = get_vn_squared(density, bath.n_modes, list(bath.domain))
        freq = np.asarray(freq, float)
        if frequencies is None:
            frequencies = freq
        elif not np.allclose(frequencies, freq):
            raise ValueError("multichannel densities do not share the mode grid")
        strengths.append(np.sqrt(np.asarray(v_sq, float) / np.pi))
    return StarBath(frequencies, np.asarray(strengths), bath.phys_dim)


def compile_chain(bath):
    """Compile a resolved, single-channel ``Bath`` into a :class:`ChainBath`."""
    _require_resolved(bath)
    densities = bath.spectral_densities()
    if len(densities) != 1:
        raise ValueError(
            "a shared multichannel bath has no single scalar chain mapping")
    onsite, coupling = get_bath_nn_paras(
        densities[0], bath.n_modes, list(bath.domain),
        discretizer=bath.discretizer())
    coupling = np.asarray(coupling, float)
    return ChainBath(onsite, coupling[1:], coupling[0], bath.phys_dim)


def compile_polaron(bath):
    """Compile the reweighted bath required by the Lang--Firsov representation."""
    _require_resolved(bath)
    densities = bath.spectral_densities()
    if len(densities) != 1:
        raise ValueError("the polaron representation requires one bath channel")
    density = densities[0]

    def displaced_density(frequency):
        if abs(frequency) < 1e-15:
            return 0.0
        return density(frequency) / frequency ** 2

    frequencies, displacements, transform = star_transform(
        displaced_density, bath.n_modes, list(bath.domain),
        bath.discretizer())
    displacements = np.asarray(displacements, float)
    star = StarBath(
        frequencies, displacements[None, :], bath.phys_dim, transform)
    chain_matrix = transform @ np.diag(frequencies) @ transform.T
    chain = ChainBath(
        np.diagonal(chain_matrix), np.diagonal(chain_matrix, -1),
        np.linalg.norm(displacements), bath.phys_dim)
    return PolaronBath(
        star=star,
        chain=chain,
        reorganization_energy=reorganization_energy(density, bath.domain),
    )
