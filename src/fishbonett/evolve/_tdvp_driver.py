"""Whole-run orchestration for prepared MPO Hamiltonians."""
import numpy as np

from fishbonett.linalg import Truncation
from fishbonett.evolve._tdvp_kernels import init_mps, right_canonicalize
from fishbonett.evolve._tdvp_sweeps import (
    DEFAULT_BOND_EXPAND, _pad_bonds, bonddims, measure_sz, tdvp1sweep,
    a1tdvp_sweep, tdvp2sweep,
)
from fishbonett.evolve._validation import (
    positive_integer, time_steps,
)


def run_mpo_hamiltonian(representation, *, dt, nsteps, sweep, initial=None,
                        trunc=None, bond_dim=None, trunc_eps=None,
                        krylov=30, observe=None, prepare=None,
                        canonicalize=True, tol=1e-7,
                        eshift=False, verbose=False, seed=0,
                        initial_bond=None, progress=None,
                        bond_expand=None, state=None, time_offset=0.0,
                        return_state=False):
    """Propagate an MPS using a representation's ``tdvp_mpo`` Hamiltonian.

    Time-dependent Hamiltonians are sampled at each step midpoint.  Static ones keep
    their environments between steps; changing an MPO invalidates those cached
    contractions automatically.

    ``state`` accepts TDVP-order tensors from a preceding segment and is mutually
    exclusive with ``initial`` and ``prepare``. ``time_offset`` keeps a
    time-dependent representation on its absolute clock. Set ``return_state`` to
    return those final tensors as a fourth result; the default three-result tuple
    remains ``(times, observations, peak_bonds)``.

    For ``tdvp2``, ``trunc_eps`` is the relative SVD threshold. For ``a1tdvp`` it
    is the relative convergence precision used to select full-QR tangent-space
    extensions before each one-site sweep. ``bond_expand`` limits the extra
    Schmidt directions retained by ``tdvp2`` or the QR-complement directions
    tested per bond by ``a1tdvp``.
    """
    dt, nsteps = time_steps(dt, nsteps)
    if sweep not in {"tdvp1", "tdvp2", "a1tdvp"}:
        raise ValueError(
            f"unknown sweep {sweep!r}; expected 'tdvp1', 'tdvp2' or 'a1tdvp'")
    truncation = Truncation.resolve(
        trunc, eps=trunc_eps, max_bond=bond_dim
    )
    bond_dim, trunc_eps = truncation.max_bond, truncation.eps
    initial_bond = positive_integer(
        initial_bond, "initial_bond", allow_none=True
    )
    krylov = positive_integer(krylov, "krylov")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if bond_expand is not None:
        if (isinstance(bond_expand, (bool, np.bool_))
                or not isinstance(bond_expand, (int, np.integer))
                or bond_expand < 0):
            raise ValueError("bond_expand must be a non-negative integer or None")
    if not np.isfinite(time_offset) or time_offset < 0:
        raise ValueError("time_offset must be finite and non-negative")
    time_offset = float(time_offset)
    dimensions = tuple(representation.dimensions)
    if len(dimensions) < 2:
        raise ValueError("TDVP system-bath propagation needs at least two sites")
    measure = observe if observe is not None else lambda state: measure_sz(state[0])
    if state is None:
        if initial is None:
            initial = np.zeros(dimensions[0], complex)
            initial[0] = 1.0
        state = init_mps(len(dimensions), dimensions[1:], initial)
        if prepare is not None:
            state = prepare(state)
    else:
        if initial is not None or prepare is not None:
            raise ValueError("state cannot be combined with initial or prepare")
        state = [np.asarray(tensor, complex).copy() for tensor in state]
        if len(state) != len(dimensions) or tuple(
            tensor.shape[2] for tensor in state
        ) != dimensions:
            raise ValueError("state physical dimensions do not match representation")

    adaptive_cap = bond_dim
    if sweep == "a1tdvp" and adaptive_cap is None:
        raise ValueError("a1tdvp requires a finite bond-dimension ceiling")
    if sweep == "tdvp1":
        fixed_cap = bond_dim if initial_bond is None else initial_bond
        if bond_dim is not None:
            fixed_cap = min(int(fixed_cap), int(bond_dim))
        state = right_canonicalize(_pad_bonds(state, fixed_cap, seed=seed))
    elif canonicalize:
        state = right_canonicalize(state)

    operator = representation.tdvp_mpo(None) if representation.static else None
    environments = None
    observations, peak_bonds = [], []
    krylov_options = {"m": krylov, "tol": tol, "eshift": eshift}
    # The allowance controls distinct bond-growth mechanisms: extra Schmidt
    # directions for TDVP2 and full-QR complement directions for A1TDVP.
    expand = DEFAULT_BOND_EXPAND if bond_expand is None else int(bond_expand)
    for step in range(nsteps):
        if not representation.static:
            operator = representation.tdvp_mpo(
                time_offset + (step + 0.5) * dt
            )
            environments = None
        if sweep == "tdvp1":
            state, environments = tdvp1sweep(
                dt, state, operator, environments, **krylov_options)
        elif sweep == "tdvp2":
            state, environments = tdvp2sweep(
                dt, state, operator, adaptive_cap, trunc_eps, environments,
                expand=expand, **krylov_options)
        else:
            state, _full, environments, _diagnostic = a1tdvp_sweep(
                dt,
                state,
                operator,
                trunc_eps=trunc_eps,
                bond_dim=adaptive_cap,
                expand=expand,
                **krylov_options)
        observations.append(measure(state))
        peak_bonds.append(max(bonddims(state)))
        if progress is not None:
            progress({"step": step, "n_steps": nsteps,
                      "t": time_offset + dt * (step + 1), "bond": peak_bonds[-1],
                      "rdm": observations[-1], "state": state})
        if verbose:
            print(
                f"  {sweep} {step + 1}/{nsteps} "
                f"t={time_offset + dt * (step + 1):.6g} "
                f"maxD={peak_bonds[-1]}",
                flush=True,
            )
    times = time_offset + np.arange(1, nsteps + 1, dtype=float) * dt
    result = times, np.asarray(observations), np.asarray(peak_bonds, dtype=int)
    if return_state:
        return (*result, state)
    return result
