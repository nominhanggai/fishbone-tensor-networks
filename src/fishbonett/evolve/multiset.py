"""Coupled TDVP for multi-set matrix-product states.

For ``|Psi> = sum_a |a>|psi_a>`` the Hamiltonian is a matrix of
environmental operators ``H[a, b]``.  Every operator block is an MPO, and the
Schrodinger equation is

``i d|psi_a>/dt = sum_b H[a, b] |psi_b>``.

The system basis is orthonormal, so the tangent projector is a direct sum of
the tangent projectors of the component MPSs.  A local TDVP update therefore
evolves all component centre tensors together under a block effective
Hamiltonian, then splits each component independently.  This module implements
that coupled two-site projector-splitting sweep without translating code from
another package.
"""

from __future__ import annotations

import numpy as np

from fishbonett.contract import _einsum_cached
from fishbonett.evolve._tdvp_kernels import expmv_lanczos
from fishbonett.evolve._tdvp_sweeps import (
    DEFAULT_BOND_EXPAND,
    _merge2,
    _split2,
)
from fishbonett.evolve._validation import positive_integer, time_steps
from fishbonett.linalg import Truncation
from fishbonett.states.multiset import MultiSetMPS

__all__ = [
    "split_system_mpo",
    "multiset_tdvp2_sweep",
    "run_multiset_mpo_hamiltonian",
]


def split_system_mpo(mpo, system_dimension=None):
    """Remove a left-boundary system site and return ``H[a][b]`` bath MPOs.

    Parameters
    ----------
    mpo
        Full MPO in ``(left, right, physical_out, physical_in)`` order, with
        the finite-dimensional system at the left boundary.
    system_dimension
        Optional dimension check.  The dimension is inferred from ``mpo[0]``
        when omitted.
    """
    mpo = [np.asarray(tensor, complex) for tensor in mpo]
    if len(mpo) < 2:
        raise ValueError("a multi-set MPO needs a system and at least one bath site")
    for site, tensor in enumerate(mpo):
        if tensor.ndim != 4:
            raise ValueError(f"mpo[{site}] must have four axes")
        if tensor.shape[2] != tensor.shape[3]:
            raise ValueError(f"mpo[{site}] has unequal physical dimensions")
        if site and tensor.shape[0] != mpo[site - 1].shape[1]:
            raise ValueError(f"the MPO bond before site {site} is incompatible")
    if mpo[-1].shape[1] != 1:
        raise ValueError("the MPO must have a right boundary of one")
    first = mpo[0]
    if first.shape[0] != 1:
        raise ValueError("the system MPO tensor must be a four-axis left boundary")
    dimension = first.shape[2]
    if system_dimension is not None and int(system_dimension) != dimension:
        raise ValueError(f"system MPO dimension is {dimension}, expected {system_dimension}")
    second = mpo[1]
    blocks = []
    for output in range(dimension):
        row = []
        for input_ in range(dimension):
            boundary = _einsum_cached(
                "r,rsij->sij",
                first[0, :, output, input_],
                second,
            )[None, :, :, :]
            if not np.any(boundary):
                row.append(None)
            else:
                # Every block has its own contracted left boundary, while the
                # remaining MPO tensors are immutable during a sweep. Sharing
                # that tail avoids N**2 copies for an N-level multi-set model.
                row.append([boundary, *mpo[2:]])
        blocks.append(row)
    return blocks


def _update_left(ket, operator, left, bra):
    return _einsum_cached(
        "amc,crp,mnqp,abq->bnr",
        left,
        ket,
        operator,
        bra.conj(),
    )


def _update_right(ket, operator, right, bra):
    return _einsum_cached(
        "abq,mnqp,crp,bnr->amc",
        bra.conj(),
        operator,
        ket,
        right,
    )


def _apply_h1(ket, operator, left, right):
    stage = np.tensordot(ket, right, axes=([1], [2]))
    stage = np.tensordot(operator, stage, axes=([1, 3], [3, 1]))
    stage = np.tensordot(left, stage, axes=([1, 2], [0, 2]))
    return np.ascontiguousarray(np.transpose(stage, (0, 2, 1)))


def _apply_h2(ket, left_operator, right_operator, left, right):
    stage = np.tensordot(ket, right, axes=([1], [2]))
    stage = np.tensordot(right_operator, stage, axes=([1, 3], [4, 2]))
    stage = np.tensordot(left_operator, stage, axes=([1, 3], [0, 3]))
    stage = np.tensordot(left, stage, axes=([1, 2], [0, 3]))
    return np.ascontiguousarray(np.transpose(stage, (0, 3, 1, 2)))


def _init_environments(sets, operators):
    count = len(sets)
    sites = len(sets[0])
    environments = [[None for _ in range(count)] for _ in range(count)]
    for output in range(count):
        for input_ in range(count):
            if operators[output][input_] is None:
                continue
            values = [None] * (sites + 2)
            boundary = np.ones((1, 1, 1), complex)
            values[0] = boundary
            values[sites + 1] = boundary
            for site in range(sites - 1, -1, -1):
                values[site + 1] = _update_right(
                    sets[input_][site],
                    operators[output][input_][site],
                    values[site + 2],
                    sets[output][site],
                )
            environments[output][input_] = values
    return environments


def _pack(values):
    shapes = tuple(value.shape for value in values)
    sizes = tuple(value.size for value in values)
    vector = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    return vector, shapes, sizes


def _unpack(vector, shapes, sizes):
    out = []
    start = 0
    for shape, size in zip(shapes, sizes, strict=True):
        out.append(np.asarray(vector[start : start + size]).reshape(shape))
        start += size
    return out


def _evolve_components(values, apply_block, tau, **krylov):
    vector, shapes, sizes = _pack(values)

    def apply(packed):
        components = _unpack(packed, shapes, sizes)
        result = [np.zeros(shape, complex) for shape in shapes]
        for output in range(len(components)):
            for input_ in range(len(components)):
                value = apply_block(output, input_, components[input_])
                if value is not None:
                    result[output] += value
        return np.concatenate([value.reshape(-1) for value in result])

    evolved = expmv_lanczos(apply, tau, vector, **krylov)
    return _unpack(evolved, shapes, sizes)


def multiset_tdvp2_sweep(
    dt,
    state,
    operators,
    chi_max,
    eps,
    environments=None,
    *,
    expand=DEFAULT_BOND_EXPAND,
    **krylov,
):
    """Advance a :class:`MultiSetMPS` by one coupled two-site TDVP sweep."""
    if not isinstance(state, MultiSetMPS):
        raise TypeError("state must be a MultiSetMPS")
    count = state.n_sets
    sites = state.n_sites
    if len(operators) != count or any(len(row) != count for row in operators):
        raise ValueError("operator block dimensions do not match the number of sets")
    for row in operators:
        for operator in row:
            if operator is None:
                continue
            if len(operator) != sites:
                raise ValueError("every operator block must match the MPS length")
            for site, tensor in enumerate(operator):
                expected = state.dimensions[site]
                if tensor.ndim != 4 or tensor.shape[2:] != (expected, expected):
                    raise ValueError(
                        f"operator site {site} does not match physical dimension {expected}"
                    )
                if site and tensor.shape[0] != operator[site - 1].shape[1]:
                    raise ValueError(f"operator bond before site {site} is incompatible")
                if not np.all(np.isfinite(tensor)):
                    raise ValueError(f"operator site {site} contains non-finite values")
            if operator[0].shape[0] != 1 or operator[-1].shape[1] != 1:
                raise ValueError("every operator block must have unit boundary bonds")
    half = 0.5 * float(dt)
    tensors = state.sets
    env = _init_environments(tensors, operators) if environments is None else environments

    # With one bath site the multi-set manifold spans the complete finite
    # state space, so no projector splitting is needed.  Evolving the coupled
    # one-site centres also keeps the public TDVP2 method useful for a bath
    # discretization that resolves to a single mode.
    if sites == 1:
        centers = [values[0] for values in tensors]

        def apply_one_site(output, input_, value):
            if operators[output][input_] is None:
                return None
            return _apply_h1(
                value,
                operators[output][input_][0],
                env[output][input_][0],
                env[output][input_][2],
            )

        centers = _evolve_components(centers, apply_one_site, -1j * float(dt), **krylov)
        for set_index, center in enumerate(centers):
            tensors[set_index][0] = center
        return state, env

    for site in range(sites - 1):
        centers = [_merge2(values[site], values[site + 1]) for values in tensors]

        def apply(output, input_, value, position=site):
            if operators[output][input_] is None:
                return None
            return _apply_h2(
                value,
                operators[output][input_][position],
                operators[output][input_][position + 1],
                env[output][input_][position],
                env[output][input_][position + 3],
            )

        centers = _evolve_components(centers, apply, -1j * half, **krylov)
        for set_index in range(count):
            tensors[set_index][site], tensors[set_index][site + 1] = _split2(
                centers[set_index], chi_max, eps, "left", expand
            )
        for output in range(count):
            for input_ in range(count):
                if operators[output][input_] is None:
                    continue
                env[output][input_][site + 1] = _update_left(
                    tensors[input_][site],
                    operators[output][input_][site],
                    env[output][input_][site],
                    tensors[output][site],
                )
        if site < sites - 2:
            centers = [values[site + 1] for values in tensors]

            def apply_one(output, input_, value, position=site + 1):
                if operators[output][input_] is None:
                    return None
                return _apply_h1(
                    value,
                    operators[output][input_][position],
                    env[output][input_][position],
                    env[output][input_][position + 2],
                )

            centers = _evolve_components(centers, apply_one, 1j * half, **krylov)
            for set_index, center in enumerate(centers):
                tensors[set_index][site + 1] = center

    for site in range(sites - 2, -1, -1):
        centers = [_merge2(values[site], values[site + 1]) for values in tensors]

        def apply(output, input_, value, position=site):
            if operators[output][input_] is None:
                return None
            return _apply_h2(
                value,
                operators[output][input_][position],
                operators[output][input_][position + 1],
                env[output][input_][position],
                env[output][input_][position + 3],
            )

        centers = _evolve_components(centers, apply, -1j * half, **krylov)
        for set_index in range(count):
            tensors[set_index][site], tensors[set_index][site + 1] = _split2(
                centers[set_index], chi_max, eps, "right", expand
            )
        for output in range(count):
            for input_ in range(count):
                if operators[output][input_] is None:
                    continue
                env[output][input_][site + 2] = _update_right(
                    tensors[input_][site + 1],
                    operators[output][input_][site + 1],
                    env[output][input_][site + 3],
                    tensors[output][site + 1],
                )
        if site > 0:
            centers = [values[site] for values in tensors]

            def apply_one(output, input_, value, position=site):
                if operators[output][input_] is None:
                    return None
                return _apply_h1(
                    value,
                    operators[output][input_][position],
                    env[output][input_][position],
                    env[output][input_][position + 2],
                )

            centers = _evolve_components(centers, apply_one, 1j * half, **krylov)
            for set_index, center in enumerate(centers):
                tensors[set_index][site] = center
    return state, env


def run_multiset_mpo_hamiltonian(
    representation,
    *,
    state,
    dt,
    nsteps,
    trunc=None,
    bond_dim=None,
    trunc_eps=None,
    krylov=30,
    tol=1e-7,
    eshift=False,
    bond_expand=None,
    observe=None,
    progress=None,
):
    """Propagate a representation with coupled two-site multi-set TDVP.

    The representation must expose the same ``tdvp_mpo(t)`` contract as the
    conventional MPS TDVP engine.  Its boundary system leg is converted to
    operator blocks at each time midpoint.
    """
    if not isinstance(state, MultiSetMPS):
        raise TypeError("state must be a MultiSetMPS")
    dt, nsteps = time_steps(dt, nsteps)
    truncation = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
    bond_dim, trunc_eps = truncation.max_bond, truncation.eps
    krylov = positive_integer(krylov, "krylov")
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    if bond_expand is not None and (
        isinstance(bond_expand, (bool, np.bool_))
        or not isinstance(bond_expand, (int, np.integer))
        or bond_expand < 0
    ):
        raise ValueError("bond_expand must be a non-negative integer or None")
    dimensions = tuple(representation.dimensions)
    expected = (state.n_sets, *state.dimensions)
    if dimensions != expected:
        raise ValueError(
            f"representation dimensions are {dimensions}, but the multi-set "
            f"state represents {expected}"
        )
    expand = DEFAULT_BOND_EXPAND if bond_expand is None else int(bond_expand)
    krylov_options = {"m": krylov, "tol": float(tol), "eshift": eshift}
    measure = (lambda current: current.system_rdm()) if observe is None else observe
    operator = None
    if representation.static:
        operator = split_system_mpo(representation.tdvp_mpo(None), state.n_sets)
    environments = None
    observations = []
    peak_bonds = []
    set_bonds = []
    for step in range(nsteps):
        if not representation.static:
            operator = split_system_mpo(representation.tdvp_mpo((step + 0.5) * dt), state.n_sets)
            environments = None
        state, environments = multiset_tdvp2_sweep(
            dt,
            state,
            operator,
            bond_dim,
            trunc_eps,
            environments,
            expand=expand,
            **krylov_options,
        )
        observations.append(measure(state))
        peak_bonds.append(state.peak_bond())
        set_bonds.append(tuple(max(values) for values in state.bond_dimensions()))
        if progress is not None:
            progress(
                {
                    "step": step,
                    "n_steps": nsteps,
                    "t": (step + 1) * dt,
                    "bond": peak_bonds[-1],
                    "rdm": observations[-1],
                    "state": state,
                }
            )
    times = np.arange(1, nsteps + 1, dtype=float) * dt
    return (
        times,
        np.asarray(observations),
        np.asarray(peak_bonds, dtype=int),
        np.asarray(set_bonds, dtype=int),
        state,
    )
