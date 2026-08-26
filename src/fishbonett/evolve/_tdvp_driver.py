"""Whole-run orchestration for prepared MPO Hamiltonians."""
import numpy as np

from fishbonett.linalg import Truncation
from fishbonett.evolve._tdvp_kernels import init_mps, right_canonicalize
from fishbonett.evolve._tdvp_sweeps import (
    DEFAULT_BOND_EXPAND, _pad_bonds, bonddims, measure_sz, tdvp1sweep,
    tdvp1sweep_dynamic, tdvp2sweep,
)
from fishbonett.evolve._validation import (
    nonnegative_finite, positive_integer, time_steps,
)


def run_mpo_hamiltonian(representation, *, dt, nsteps, sweep, initial=None,
                        trunc=None, bond_dim=None, trunc_eps=None,
                        krylov=30, observe=None, prepare=None,
                        canonicalize=True, prec=None, tol=1e-7,
                        eshift=False, verbose=False, seed=0,
                        initial_bond=None, progress=None,
                        bond_expand=None):
    """Propagate an MPS using a representation's ``tdvp_mpo`` Hamiltonian.

    Time-dependent Hamiltonians are sampled at each step midpoint.  Static ones keep
    their environments between steps; changing an MPO invalidates those cached
    contractions automatically.
    """
    dt, nsteps = time_steps(dt, nsteps)
    if sweep not in {"tdvp1", "tdvp2", "dtdvp"}:
        raise ValueError(
            f"unknown sweep {sweep!r}; expected 'tdvp1', 'tdvp2' or 'dtdvp'")
    truncation = Truncation.resolve(
        trunc, eps=trunc_eps, max_bond=bond_dim
    )
    bond_dim, trunc_eps = truncation.max_bond, truncation.eps
    initial_bond = positive_integer(
        initial_bond, "initial_bond", allow_none=True
    )
    krylov = positive_integer(krylov, "krylov")
    if sweep == "dtdvp":
        prec = trunc_eps if prec is None else nonnegative_finite(prec, "prec")
    elif prec is not None:
        raise TypeError("prec is only used by the dtdvp sweep; use eps otherwise")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if bond_expand is not None:
        if (isinstance(bond_expand, (bool, np.bool_))
                or not isinstance(bond_expand, (int, np.integer))
                or bond_expand < 0):
            raise ValueError("bond_expand must be a non-negative integer or None")
    dimensions = tuple(representation.dimensions)
    if len(dimensions) < 2:
        raise ValueError("TDVP system-bath propagation needs at least two sites")
    if initial is None:
        initial = np.zeros(dimensions[0], complex)
        initial[0] = 1.0
    measure = observe if observe is not None else lambda state: measure_sz(state[0])
    state = init_mps(len(dimensions), dimensions[1:], initial)
    if prepare is not None:
        state = prepare(state)

    adaptive_cap = bond_dim
    if sweep == "dtdvp" and adaptive_cap is None:
        raise ValueError("dtdvp requires a finite bond-dimension ceiling")
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
    # Both two-site sweeps use the same rank-expansion allowance.
    expand = DEFAULT_BOND_EXPAND if bond_expand is None else int(bond_expand)
    for step in range(nsteps):
        if not representation.static:
            operator = representation.tdvp_mpo((step + 0.5) * dt)
            environments = None
        if sweep == "tdvp1":
            state, environments = tdvp1sweep(
                dt, state, operator, environments, **krylov_options)
        elif sweep == "tdvp2":
            state, environments = tdvp2sweep(
                dt, state, operator, adaptive_cap, trunc_eps, environments,
                expand=expand, **krylov_options)
        else:
            state, _full, environments, _diagnostic = tdvp1sweep_dynamic(
                dt, state, operator, None, environments, prec=prec,
                Dlim=adaptive_cap, expand=expand,
                **krylov_options)
        observations.append(measure(state))
        peak_bonds.append(max(bonddims(state)))
        if progress is not None:
            progress({"step": step, "n_steps": nsteps,
                      "t": dt * (step + 1), "bond": peak_bonds[-1],
                      "rdm": observations[-1], "state": state})
        if verbose:
            print(
                f"  {sweep} {step + 1}/{nsteps} "
                f"t={dt * (step + 1):.6g} maxD={peak_bonds[-1]}",
                flush=True,
            )
    times = np.arange(1, nsteps + 1, dtype=float) * dt
    return times, np.asarray(observations), np.asarray(peak_bonds, dtype=int)
