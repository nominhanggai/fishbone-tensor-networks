"""Pauli matrices, bosonic ladder operators, and entropy helpers.

Bosonic operators (``annihilate``, ``create``, ``number``) act on a truncated
``dim``-level Fock space, so ``[b, b^dag] != 1`` at the top of the ladder.
Check convergence by raising ``phys_dim``, not by testing commutator identities.
"""
import numpy as np

from fishbonett.contract import _einsum_cached

__all__ = ["sigma_p", "sigma_m", "sigma_x", "sigma_y", "sigma_z", "sigma_0",
           "sigma_1", "annihilate", "create", "number", "temp_factor",
           "rlogr", "entang", "energy_current_operator", "displacement"]

# -- Pauli matrices ----------------------------------------------------------
sigma_p = np.float64([[0, 1], [0, 0]])        # raising  S^+
sigma_m = np.float64([[0, 0], [1, 0]])        # lowering S^-
sigma_x = np.float64([[0, 1], [1, 0]])
sigma_y = np.complex64([[0, -1j], [1j, 0]])
sigma_z = np.float64([[1, 0], [0, -1]])
sigma_0 = np.zeros((2, 2))                    # zero matrix
sigma_1 = np.eye(2)                           # identity


def temp_factor(temp, w):
    """Thermal factor ``0.5(1 + coth(beta w / 2))`` (temperature in kelvin)."""
    beta = 1 / (0.6950348009119888 * temp)
    return 0.5 * (1. + 1. / np.tanh(beta * w / 2.))


def rlogr(si):
    """``-s log s`` (a single Schmidt term of the entanglement entropy)."""
    return (-1) * si * np.log(si)


def entang(s):
    """von Neumann entanglement entropy from a list of Schmidt values ``s``."""
    etg = 0.0
    for si in s:
        if si != 0:
            etg += rlogr(si ** 2)
    return etg


def annihilate(dim: int):
    """Bosonic annihilation operator ``b`` on a ``dim``-level Fock truncation.

    ``b|n> = sqrt(n)|n-1>``, i.e. the real upper-bidiagonal matrix with
    ``sqrt(1), sqrt(2), ...`` above the diagonal.
    """
    op = np.zeros((dim, dim))
    for i in range(dim - 1):
        op[i, i + 1] = np.sqrt(i + 1)
    return op


def create(dim: int):
    """Bosonic creation operator ``b^dagger`` -- the transpose of
    :func:`annihilate` (which is real)."""
    return annihilate(dim).T


def number(dim: int):
    """Bosonic number operator ``b^dagger b = diag(0, 1, ..., dim-1)``."""
    return annihilate(dim).T @ annihilate(dim)


#: Cached ``(vectors, eigenvalues)`` of ``i(b^dagger - b)`` per truncation, for
#: :func:`displacement`.  One small ``eigh`` per dimension for the whole process.
_DISPLACEMENT_BASIS = {}


def _displacement_basis(dim: int):
    """Eigendecomposition of the Hermitian ``K = i(b^dagger - b)``."""
    basis = _DISPLACEMENT_BASIS.get(dim)
    if basis is None:
        generator = 1j * (create(dim) - annihilate(dim))
        eigenvalues, vectors = np.linalg.eigh(generator)
        basis = _DISPLACEMENT_BASIS[dim] = (vectors, eigenvalues)
    return basis


def displacement(alpha, dim: int):
    """``exp(alpha b^dagger - conj(alpha) b)`` on a ``dim``-level truncation.

    ``alpha`` may be a scalar or an array; the result has shape
    ``(*np.shape(alpha), dim, dim)``.

    Computed in closed form rather than by a matrix exponential per ``alpha``.
    Writing ``alpha = r e^{i phi}`` and using ``e^{i phi n} b^dagger e^{-i phi n} =
    e^{i phi} b^dagger``,

    .. math::

        \\alpha b^\\dagger - \\alpha^* b
            = r\\, e^{i\\phi n} (b^\\dagger - b) e^{-i\\phi n},

    so with the *fixed* Hermitian ``K = i(b^dagger - b) = V diag(mu) V^dagger``
    (one cached ``eigh`` per ``dim``)

    .. math::

        D(\\alpha) = P(\\phi)\\, V e^{-i r \\mu} V^\\dagger P(\\phi)^\\dagger ,
        \\qquad P(\\phi) = \\mathrm{diag}(e^{i \\phi n}).

    Exact on the truncated ladder, because it exponentiates the *truncated*
    generator -- the same operator ``expm`` would be handed.  Note that a single
    displacement is not the exact propagator of a truncated oscillator, since
    ``[b, b^dagger] = I - dim |dim-1><dim-1|``; that discrepancy lives at the top
    Fock level and is controlled by ``phys_dim``, not by this routine.

    Notes
    -----
    Replaces a per-``alpha`` ``scipy.linalg.expm``, which dominated the cost of
    building :meth:`~fishbonett.representations.interaction.InteractionRepresentation.trotter_mpo`
    (one call per mode per coupling eigenvalue, every step).
    """
    alpha = np.asarray(alpha, dtype=complex)
    vectors, eigenvalues = _displacement_basis(dim)
    radius, phi = np.abs(alpha), np.angle(alpha)
    # V exp(-i r mu) V^dagger, batched over alpha
    inner = _einsum_cached(
        "am,...m,bm->...ab", vectors,
        np.exp(-1j * radius[..., None] * eigenvalues), vectors.conj())
    ladder = np.arange(dim)
    rotation = np.exp(1j * phi[..., None] * ladder)        # diag of P(phi)
    return rotation[..., :, None] * inner * rotation.conj()[..., None, :]


def energy_current_operator(onsite, right_bond, left_bond=None):
    """Local energy current through the bond to the right of one site.

    For a nearest-neighbour Hamiltonian with site term ``h_i`` and bonds
    ``V_(i-1,i)`` / ``V_(i,i+1)``, the energy leaving the region ending at
    site ``i`` is

    ``j_i = 1j * [V_(i-1,i) + h_i, V_(i,i+1)]``.

    The returned operator acts on ``(i, i+1)`` when ``left_bond`` is omitted
    (the left boundary), otherwise on ``(i-1, i, i+1)``.  This definition is
    derived directly from the continuity equation and fixes the current sign:
    positive means flow from left to right.
    """
    onsite = np.asarray(onsite, complex)
    right_bond = np.asarray(right_bond, complex)
    if onsite.ndim != 2 or onsite.shape[0] != onsite.shape[1]:
        raise ValueError("onsite must be a square matrix")
    d_mid = onsite.shape[0]
    if right_bond.ndim != 2 or right_bond.shape[0] != right_bond.shape[1]:
        raise ValueError("right_bond must be a square matrix")
    if right_bond.shape[0] % d_mid:
        raise ValueError("right_bond dimension is incompatible with onsite")
    d_right = right_bond.shape[0] // d_mid
    if left_bond is None:
        stored = np.kron(onsite, np.eye(d_right))
        current = 1j * (stored @ right_bond - right_bond @ stored)
    else:
        left_bond = np.asarray(left_bond, complex)
        if left_bond.ndim != 2 or left_bond.shape[0] != left_bond.shape[1]:
            raise ValueError("left_bond must be a square matrix")
        if left_bond.shape[0] % d_mid:
            raise ValueError("left_bond dimension is incompatible with onsite")
        d_left = left_bond.shape[0] // d_mid
        stored = (np.kron(left_bond, np.eye(d_right))
                  + np.kron(np.eye(d_left), np.kron(onsite, np.eye(d_right))))
        crossing = np.kron(np.eye(d_left), right_bond)
        current = 1j * (stored @ crossing - crossing @ stored)
    # Suppress roundoff anti-Hermitian components without concealing bad input.
    return 0.5 * (current + current.conj().T)
