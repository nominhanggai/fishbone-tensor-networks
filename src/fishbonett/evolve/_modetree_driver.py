"""Whole-run drivers for the balanced mode-tree TTNO propagator."""
import numpy as np
import scipy.linalg

from fishbonett.linalg import DEFAULT_EPS, Truncation
from fishbonett.evolve._modetree_core import (
    _resolve_sys, build_balanced_tree, init_state,
)
from fishbonett.evolve._modetree_sweeps import (
    apply_coupling, apply_sys, build_coupling_op, canon_to_root,
    measure_sz_oc, truncate_from_root,
)
from fishbonett.evolve._validation import (
    nonnegative_finite, positive_integer, time_steps,
)


def _peak_bond(nodes):
    return max((node.tensor.shape[0] for node in nodes
                if node.parent is not None), default=1)


def _run_ttno(representation, *, dt=0.05, nsteps=120, D=20,
              trunc_eps=DEFAULT_EPS, observe=None, track_bond=False,
              init=None, seed=0, progress=None):
    """Strang propagation of an interaction representation on a mode tree."""
    dt, nsteps = time_steps(dt, nsteps)
    D = positive_integer(D, "D", allow_none=True)
    trunc_eps = nonnegative_finite(trunc_eps, "trunc_eps")
    h_system, coupling, initial, d_system = _resolve_sys(
        representation.h_sys, representation.coupling, init, 1.0, 0.0)
    dimensions = tuple(representation.dimensions)
    n_chain = len(dimensions) - 1
    if n_chain < 1 or len(set(dimensions[1:])) != 1:
        raise ValueError(
            "the balanced mode tree requires uniform bath-site dimensions")
    phys_dim = dimensions[1]
    nodes, root, _leaves = build_balanced_tree(n_chain, phys_dim, d_system)
    init_state(nodes, root, initial)
    half_system = scipy.linalg.expm(-0.5j * dt * h_system)
    measure = observe if observe is not None else measure_sz_oc

    values, peak = [], []
    for step in range(nsteps):
        time = step * dt
        # ``build_balanced_tree`` labels leaves in forward chain-mode order, and
        # ``build_coupling_op`` indexes amplitudes by that public ``node.mode``.
        # Keep the same mode-k -> coefficient-k convention as the MPS engines.
        amplitudes = representation.interval_coefficients(time, dt)
        apply_sys(nodes, root, half_system)
        build_coupling_op(nodes, root, amplitudes, coupling)
        apply_coupling(nodes)
        canon_to_root(nodes, root)
        truncate_from_root(nodes, root, D, trunc_eps)
        apply_sys(nodes, root, half_system)
        values.append(measure(nodes, root))
        bond = _peak_bond(nodes)
        if track_bond:
            peak.append(bond)
        if progress is not None:
            progress({
                "step": step, "n_steps": nsteps,
                "t": (step + 1) * dt, "bond": bond,
                "rdm": values[-1], "state": nodes,
            })
    times = np.arange(1, nsteps + 1, dtype=float) * dt
    result = (times, np.asarray(values))
    if track_bond:
        result += (np.asarray(peak, dtype=int),)
    return result


def run_tree_tebd(
    representation, *, trunc=None, bond_dim=None, trunc_eps=None, **kwargs,
):
    """Propagate with the second-order conditional-coupling TTNO step.

    Truncation uses the same ``Truncation`` / ``bond_dim`` / ``trunc_eps``
    vocabulary as the high-level model API.
    """
    policy = Truncation.resolve(
        trunc, eps=trunc_eps, max_bond=bond_dim
    )
    return _run_ttno(
        representation, D=policy.max_bond, trunc_eps=policy.eps, **kwargs
    )
