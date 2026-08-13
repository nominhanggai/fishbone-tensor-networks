"""The propagation call: what every ``run`` has in common.

A ``run`` is one *(model, frame, integrator)* combination
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

__all__ = ["RunCtx", "propagate", "mps_peak_bond", "tree_peak_bond"]


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
    kw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def t_max(self):
        """Physical time the run covers -- what sizes an automatic bath."""
        return self.n_steps * self.dt


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

    The loop every single-system method has in common.  It used to be written out
    once per method -- five times in ``models/system_bath.py`` alone -- which is why
    they disagreed about what to report: some collected the peak bond, some did not,
    and two used differently-named helpers to compute it.

    What genuinely varies is passed in, and each is a *layer's* business:

    ``step(k)``
        advance one ``dt``, from step index ``k``.  The **integrator**.
    ``rdm()``
        the system reduced density matrix **in the lab frame** after that step.  The
        **frame**, because a frame that dresses the state (polaron) has to undress
        the observable, and one that does not simply reads it off.
    ``peak_bond()``
        the widest bond right now.  The **state**, hence
        :func:`mps_peak_bond` / :func:`tree_peak_bond`.

    Every method reports ``max_bond``, including the fixed-bond ones where it is
    constant: it is the same quantity, and "not reported" used to mean nothing more
    than "this driver's author did not collect it".
    """
    rdms, max_bond = [], []
    for k in range(ctx.n_steps):
        step(k)
        rdms.append(rdm())
        max_bond.append(peak_bond())
    return Result(t=np.arange(1, ctx.n_steps + 1) * ctx.dt,
                  expect=expect_from_rdm(rdms, ctx.obs_ops),
                  max_bond=np.array(max_bond),
                  rdm=np.asarray(rdms),
                  method=spec.name)
