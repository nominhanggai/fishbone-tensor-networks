"""Single-excitation electronic systems with independent harmonic baths."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from fishbonett.bath.spec import Bath
from fishbonett.linalg import Truncation
from fishbonett.models import registry
from fishbonett.models.propagate import (
    RunCtx,
    _resolve_continuation,
    _resolve_sampling_options,
    resolve_time_grid,
)
from fishbonett.system import check_operator

__all__ = ["ExcitonBath"]


def _bath_entries(baths, count):
    if isinstance(baths, Mapping):
        entries = [None] * count
        for level, bath in baths.items():
            if isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer)):
                raise TypeError("bath mapping keys must be integer electronic levels")
            level = int(level)
            if level < 0 or level >= count:
                raise ValueError(f"bath level {level} is outside 0 <= level < {count}")
            entries[level] = bath
        return entries
    entries = list(baths)
    if len(entries) != count:
        raise ValueError("baths must contain one entry per electronic level")
    return entries


class ExcitonBath:
    r"""An :math:`N`-level single-excitation Hamiltonian with local baths.

    ``baths[i]`` couples to the population projector ``|i><i|``.  The model
    keeps that physical statement independent of whether the state stores all
    electronic levels at one site, one local two-level site per level, or one
    bath tensor network per electronic basis state.

    Parameters
    ----------
    h
        Hermitian ``(N, N)`` electronic Hamiltonian in the site basis.
    baths
        One :class:`~fishbonett.bath.spec.Bath` or ``None`` per electronic
        level.  A mapping may omit levels without a bath.
    """

    _MODEL = "exciton-bath"

    def __init__(self, h, baths):
        """Validate the site Hamiltonian and align one bath entry per level."""
        self.h = check_operator(h, "h")
        self.n_levels = self.h.shape[0]
        self.baths = _bath_entries(baths, self.n_levels)
        for level, bath in enumerate(self.baths):
            if bath is not None and not isinstance(bath, Bath):
                raise TypeError(f"baths[{level}] must be a Bath or None")
        if not any(bath is not None for bath in self.baths):
            raise ValueError("at least one electronic level must have a bath")

    def initial_vector(self, initial=None):
        """Normalize an initial site-basis vector or basis-state index."""
        if initial is None:
            initial = 0
        if not isinstance(initial, (bool, np.bool_)) and isinstance(initial, (int, np.integer)):
            level = int(initial)
            if level < 0 or level >= self.n_levels:
                raise ValueError(f"initial level {level} is outside 0 <= level < {self.n_levels}")
            vector = np.zeros(self.n_levels, complex)
            vector[level] = 1.0
            return vector
        vector = np.asarray(initial, complex).reshape(-1)
        if vector.shape != (self.n_levels,):
            raise ValueError(f"initial must have shape {(self.n_levels,)} or be a level index")
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("initial must have a finite nonzero norm")
        return vector / norm

    def run(
        self,
        *,
        dt,
        t_max=None,
        n_steps=None,
        method=None,
        model=None,
        representation=None,
        state_geometry=None,
        integrator=None,
        trunc=None,
        bond_dim=None,
        trunc_eps=None,
        observables=None,
        initial=None,
        krylov=25,
        seed=0,
        resume=None,
        bath_horizon=None,
        progress=None,
        observe_every=1,
        svd_backend="auto",
        **engine_kw,
    ):
        """Propagate an excitonic model with an explicitly selected layout.

        The conventional ``system-first-mps`` and ``interleaved-mps`` layouts
        support TEBD, Trotter-MPO, TDVP1, TDVP2, and dTDVP. The
        ``multi-set-mps`` and ``multi-set-tree`` layouts support TDVP2. Every
        result includes ``expect["population"]`` with shape
        ``(recorded_times, n_levels)`` in addition to requested system
        observables. Conventional-MPS results include a checkpoint that can be
        resumed within the original ``bath_horizon``.
        """
        axes = {
            "model": model,
            "representation": representation,
            "state_geometry": state_geometry,
            "integrator": integrator,
        }
        axes_given = any(value is not None for value in axes.values())
        if method is None and not axes_given:
            method = "interaction-chain-system-first-tdvp2"
        spec = registry.resolve(
            {self._MODEL},
            method=None if method is None else method.lower().replace("_", "-"),
            **axes,
        )
        allowed = set()
        if spec.engine in {
            "exciton-mpo-tdvp", "multiset-tdvp", "multiset-tree-tdvp"
        }:
            allowed.update({"tol", "eshift"})
            if spec.integrator == "tdvp2":
                allowed.add("bond_expand")
            elif spec.integrator == "dtdvp":
                allowed.update({"prec", "bond_expand"})
        unknown = set(engine_kw) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unexpected run option(s) for {spec.name}: {names}")
        dt, n_steps = resolve_time_grid(dt, t_max=t_max, n_steps=n_steps)
        truncation = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
        if truncation.max_bond is None and spec.requires_bond_cap:
            reason = (
                "uses a fixed one-site TDVP manifold"
                if spec.integrator == "tdvp1"
                else "grows bonds adaptively but requires a finite memory ceiling"
            )
            raise ValueError(
                f"method {spec.name!r} {reason}, so bond_dim must be given "
                "explicitly"
            )
        observe_every, bath_horizon = _resolve_sampling_options(observe_every, bath_horizon)
        bath_horizon = _resolve_continuation(
            resume=resume,
            initial=initial,
            method=spec.name,
            dt=dt,
            n_steps=n_steps,
            bath_horizon=bath_horizon,
            supports_resume=spec.state_geometry in {
                "system-first-mps", "interleaved-mps"
            },
        )
        if observables is None:
            observables = {}
        if not hasattr(observables, "items"):
            raise TypeError("observables must be a mapping from names to operators")
        normalized = {}
        for name, operator in observables.items():
            value = np.asarray(operator, complex)
            if value.shape != self.h.shape or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"observable {name!r} must have shape {self.h.shape} and be finite"
                )
            normalized[name] = value
        context = RunCtx(
            dt=dt,
            n_steps=n_steps,
            bond_dim=truncation.max_bond,
            trunc_eps=truncation.eps,
            obs_ops=normalized,
            initial=self.initial_vector(initial),
            krylov=krylov,
            seed=seed,
            svd_backend=svd_backend,
            resume=resume,
            bath_horizon=bath_horizon,
            observe_every=observe_every,
            progress=progress,
            kw=engine_kw,
        )
        from fishbonett.models.simulation import compile_plan

        return compile_plan(self, spec, context).run()
