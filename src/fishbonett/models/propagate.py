"""The propagation call: what every ``run`` has in common.

A ``run`` is one *(model, representation, integrator)* combination
(:class:`fishbonett.models.registry.Method`) applied to one set of run
parameters.  :class:`RunCtx` is that second half -- the arguments that are the
same whichever combination was picked -- so a driver takes ``(spec, ctx)`` and
nothing else, and dispatch can be a table lookup rather than a chain of ``if``
statements over method names.
"""
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from fishbonett.models.result import Result

__all__ = ["RunCtx", "propagate",
           "mps_peak_bond", "tree_peak_bond", "modetree_peak_bond"]


@dataclass
class RunCtx:
    """Everything a driver needs that does not depend on which method ran.

    Parameters
    ----------
    dt, n_steps
        Step and count.  ``t_max`` is resolved to ``n_steps`` before this is built.
    bond_dim, trunc_eps
        The resolved :class:`~fishbonett.linalg.Truncation`, split into the two
        numbers the engines take.  ``bond_dim=None`` means unlimited.
    obs_ops
        ``{name: operator}``, already defaulted from the system when the caller
        passed nothing.
    initial
        The initial-state spec, still in user form (``"up"``, a vector, ...);
        each model resolves it, since what "up" means depends on the model.
    krylov
        Krylov dimension for the TDVP exponentials.  Ignored by the TEBD drivers.
    seed
        Run-local random seed for randomized truncation and the tiny subspace
        seeds needed by fixed-bond TDVP.  It never mutates NumPy's
        process-global random state.  Defaults to ``0`` so that a run is
        **reproducible**: randomized truncation is an internal optimization and
        must not make an observable depend on when it was run.  Pass ``None``
        to draw from NumPy's global generator instead.
    kw
        Engine-specific extras passed through from ``run(**engine_kw)``.
    """

    dt: float
    n_steps: int
    bond_dim: Optional[int] = None
    trunc_eps: float = 1e-4
    obs_ops: Mapping[str, Any] = field(default_factory=dict)
    initial: Any = "up"
    krylov: int = 25
    seed: Optional[int] = 0
    kw: Mapping[str, Any] = field(default_factory=dict)
    resume: Any = None
    bath_horizon: Optional[float] = None
    observe_every: int = 1
    #: Optional ``callable(info)`` invoked after **every** step with a dict of
    #: ``step``, ``n_steps``, ``t``, ``bond`` and ``state``.  Separate from
    #: ``observe_every``, which controls what is *recorded* into the Result:
    #: this controls what is *reported* while a long run is still going, so a
    #: multi-hour propagation is not silent between observations.
    progress: Any = None

    @property
    def t_max(self):
        """Physical time the run covers -- what sizes an automatic bath."""
        return self.n_steps * self.dt

    @property
    def elapsed(self):
        """Time already represented by a continuation checkpoint."""
        return 0.0 if self.resume is None else float(self.resume.elapsed)


def mps_peak_bond(state):
    """Peak bond of an MPS in Vidal form -- the widest ``S``."""
    return max((len(s) for s in state.S), default=1)


def tree_peak_bond(state):
    """Peak bond of a tree state -- the widest bond leg over every node."""
    return max((t.shape[leg] for t in state.T for leg in range(t.ndim - 1)),
               default=1)


def modetree_peak_bond(nodes):
    """Peak bond of the mode-tree engine's own node list.

    Its legs are ``[parent_bond, child_bonds..., physical]`` with the physical one
    present only on ``'spin'`` and ``'leaf'`` nodes -- internal nodes carry none
    (which is one of the reasons that engine keeps its own container).  A dummy
    dimension-1 parent leg at the root is harmless to a maximum.
    """
    best = 1
    for n in nodes:
        t = getattr(n, "tensor", None)
        if t is None:
            continue
        n_bonds = t.ndim - (1 if n.kind in ("spin", "leaf") else 0)
        for leg in range(n_bonds):
            best = max(best, t.shape[leg])
    return best


def propagate(spec, ctx, *, step, rdm, peak_bond, expect_from_rdm):
    """Run ``ctx.n_steps`` steps and assemble the :class:`Result`.

    Method-specific operations are supplied as callbacks:

    ``step(k)``
        Advance one ``dt`` from step index ``k``.
    ``rdm()``
        Return the system reduced density matrix in the lab representation after
        that step. A dressed representation such as polaron must transform the
        observable back to the lab representation.
    ``peak_bond()``
        Return the widest current state bond, as implemented by
        :func:`mps_peak_bond` / :func:`tree_peak_bond`.

    Every method reports ``max_bond``; it is constant for fixed-bond methods.
    """
    rdms, max_bond = [], []
    for k in range(ctx.n_steps):
        step(k)
        rdms.append(rdm())
        max_bond.append(peak_bond())
        if ctx.progress is not None:
            ctx.progress({"step": k, "n_steps": ctx.n_steps,
                          "t": (k + 1) * ctx.dt, "bond": max_bond[-1],
                          "rdm": rdms[-1], "state": None})
    return Result(t=np.arange(1, ctx.n_steps + 1) * ctx.dt,
                  expect=expect_from_rdm(rdms, ctx.obs_ops),
                  max_bond=np.array(max_bond),
                  rdm=np.asarray(rdms),
                  method=spec.name)
