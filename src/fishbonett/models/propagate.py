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

__all__ = ["RunCtx"]


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
