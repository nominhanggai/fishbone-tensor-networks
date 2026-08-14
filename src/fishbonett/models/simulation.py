"""Compile a resolved method into an executable simulation plan.

The model layer resolves user-facing choices to a
:class:`~fishbonett.models.registry.Method`.  This module owns the next boundary:
it lowers that method and the physical problem into a prepared frame, state,
integrator and measurement policy.  The resulting :class:`SimulationPlan` is the
only object :class:`~fishbonett.models.system_bath.SystemBath` has to execute.

Keeping this orchestration outside the physical model prevents the model from
becoming a second frame and integrator registry.  The taxonomy says *which* engine
implements a method; :data:`PLAN_COMPILERS` says how to prepare that engine.
"""
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np
import scipy.linalg as _la

from fishbonett.evolve import modetree as _tree
from fishbonett.evolve import tdvp as _mpo
from fishbonett.evolve import tebd as _tebd
from fishbonett.frames import mpo as _frames_mpo
from fishbonett.frames.capabilities import (
    DisplacementFactory, MPOHamiltonian, StaticGateFactory,
    StaticGraphHamiltonian, SwapGateFactory, require_capability,
)
from fishbonett.models.propagate import (
    RunCtx, modetree_peak_bond, mps_peak_bond, propagate, tree_peak_bond,
)
from fishbonett.models.result import Result
from fishbonett.randomized import random_seed

__all__ = ["SimulationPlan", "PLAN_COMPILERS", "compile_plan"]


@dataclass(frozen=True)
class SimulationPlan:
    """A fully prepared simulation, separated from its physical model.

    A step-based engine exposes the three policies common to propagation:
    ``step`` advances the state, ``measure_rdm`` returns the lab-frame reduced
    density matrix, and ``peak_bond`` reports the state cost.  Engines whose native
    API owns a full sweep loop use ``execute`` instead.  Exactly one of these two
    forms is present.

    Parameters
    ----------
    spec
        The resolved method-registry row.
    context
        Run parameters independent of the selected method.
    step, measure_rdm, peak_bond
        Prepared policies for a step-based engine.
    execute
        Prepared whole-run driver for an engine that owns its loop.
    """

    spec: object
    context: RunCtx
    step: Optional[Callable[[int], None]] = None
    measure_rdm: Optional[Callable[[], np.ndarray]] = None
    peak_bond: Optional[Callable[[], int]] = None
    execute: Optional[Callable[[], Result]] = None

    def __post_init__(self):
        policies = (self.step, self.measure_rdm, self.peak_bond)
        has_step_plan = all(policy is not None for policy in policies)
        if any(policy is not None for policy in policies) and not has_step_plan:
            raise ValueError(
                "a step-based simulation plan needs step, measure_rdm and "
                "peak_bond together")
        if has_step_plan == (self.execute is not None):
            raise ValueError(
                "a simulation plan needs exactly one of step policies or execute")

    @property
    def is_step_based(self):
        """Whether :meth:`run` uses the shared step/measure/collect loop."""
        return self.step is not None

    def run(self):
        """Execute the prepared plan and return a uniform :class:`Result`."""
        with random_seed(self.context.seed):
            if self.execute is not None:
                return self.execute()
            return propagate(
                self.spec, self.context, step=self.step, rdm=self.measure_rdm,
                peak_bond=self.peak_bond, expect_from_rdm=_expect_from_rdm)


def _expect_from_rdm(rdms, obs_ops: Mapping[str, np.ndarray]):
    rdms = np.asarray(rdms)
    return {name: np.einsum("tij,ji->t", rdms, np.asarray(operator)).real
            for name, operator in obs_ops.items()}


def _check_single_channel(model):
    if model.system.is_multichannel:
        raise ValueError(
            "this method takes a single coupling operator, but the system was "
            "given a list of them.  A multichannel model is selected by the "
            "SystemBath coupling list and has its own propagators; see "
            "fishbonett.models.registry.")


def _undressed_mpo_frame(model, maker, bath, context, compiled):
    frame = maker(
        n_chain=bath.n_modes,
        d=bath.phys_dim, hsys=model.h, cop=model.coupling,
        init=model.system.initial_vector(context.initial),
        compiled=compiled)
    hooks = dict(
        observe=lambda tensors: _mpo.measure_rdm(tensors[0]),
        prec=context.kw.get("prec", 1e-4),
        tol=context.kw.get("tol", 1e-7),
        eshift=context.kw.get("eshift", False),
    )
    return frame, hooks


def _polaron_builder(model, context):
    from fishbonett.frames.polaron import SystemBathPolaron

    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(context.t_max)
    bath = coupled.bath
    n_modes, d_sys = bath.n_modes, model.h.shape[0]
    phys_dims = [d_sys] + [bath.phys_dim] * n_modes
    builder = SystemBathPolaron(
        phys_dims, h_sys=model.h, coupling=model.coupling,
        compiled_polaron=coupled.compiled_polaron()).build()
    return builder, bath, n_modes, phys_dims


def _compile_mpo_frame(model, spec, context, bath, coupled):
    """Prepare the one MPO frame selected by ``spec.frame``.

    This is intentionally a frame compiler, not another method registry: several
    registry rows differing only in integrator share the same prepared frame.
    """
    if spec.frame == "schrodinger-chain":
        compiled = coupled.compiled_chain()
        return _undressed_mpo_frame(
            model, _frames_mpo.chain_mpo_frame, bath, context, compiled)
    if spec.frame == "schrodinger-star":
        compiled = coupled.compiled_star()
        return _undressed_mpo_frame(
            model, _frames_mpo.static_star_mpo_frame, bath, context, compiled)
    if spec.frame == "interaction-chain":
        compiled = coupled.compiled_star()
        return _undressed_mpo_frame(
            model, _frames_mpo.ip_chain_mpo_frame, bath, context, compiled)
    if spec.frame == "interaction-star":
        compiled = coupled.compiled_star()
        return _undressed_mpo_frame(
            model, _frames_mpo.ip_star_mpo_frame, bath, context, compiled)
    if spec.frame == "polaron-chain":
        builder, _bath, _n_modes, _phys_dims = _polaron_builder(model, context)
        mpo = builder.mpo()

        def prepare(tensors):
            tensors[0], tensors[1] = builder.initial_mps_pair(
                model.system.initial_vector(context.initial))
            return tensors

        frame = _frames_mpo.MPOFrame(
            n_sites=len(mpo), phys_dim=bath.phys_dim,
            system=(model.h, model.coupling,
                    np.zeros(model.h.shape[0], complex)),
            mpo=lambda t=None: mpo, static=True)
        hooks = dict(
            prepare=prepare, canonicalize=False,
            observe=lambda tensors: builder.undress_rdm_tdvp(
                tensors[0], tensors[1]),
            prec=context.kw.get("prec", context.trunc_eps),
            # Padding every bond to a large memory cap is especially wasteful in
            # the displaced frame.  Six seed states converge the prepared
            # coherent block; users can request a larger fixed manifold explicitly.
            initial_bond=context.kw.get(
                "initial_bond", min(context.bond_dim or 6, 6)),
        )
        return frame, hooks
    raise ValueError(f"engine 'mpo-tdvp' has no compiler for frame {spec.frame!r}")


def _compile_mpo_plan(model, spec, context):
    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(context.t_max)
    bath = coupled.bath
    frame, hooks = _compile_mpo_frame(model, spec, context, bath, coupled)
    require_capability(frame, MPOHamiltonian, engine=spec.engine)

    def execute():
        times, rdms, max_bond = _mpo.run_mpo_frame(
            frame, dt=context.dt, nsteps=context.n_steps, sweep=spec.driver,
            D=context.bond_dim, chi_max=context.bond_dim,
            eps=context.trunc_eps, krylov=context.krylov,
            seed=context.seed,
            Dplusmax=context.kw.get("Dplusmax", 4), **hooks)
        return Result(
            t=times, expect=_expect_from_rdm(rdms, context.obs_ops),
            max_bond=max_bond, rdm=np.asarray(rdms), method=spec.name)

    return SimulationPlan(spec, context, execute=execute)


def _compile_modetree_plan(model, spec, context):
    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(context.t_max)
    bath = coupled.bath

    def execute():
        max_bond = []

        def observe(nodes, root):
            max_bond.append(modetree_peak_bond(nodes))
            return _tree.measure_rdm_oc(nodes, root)

        common = dict(
            hsys=model.h, cop=model.coupling,
            init=model.system.initial_vector(context.initial),
            n_chain=bath.n_modes, phys_dim=bath.phys_dim, dt=context.dt,
            nsteps=context.n_steps, D=context.bond_dim,
            discretizer=bath.discretizer(),
            compiled=coupled.compiled_star(),
            observe=observe, seed=context.seed, **context.kw)
        density = domain = None
        if spec.driver == "run_tree_tdvp":
            times, rdms = _tree.run_tree_mpo(
                density, domain, sweep="tdvp1", **common)
        elif spec.driver == "run_tree_tdvp2":
            times, rdms = _tree.run_tree_mpo(
                density, domain, sweep="tdvp2",
                trunc_eps=context.trunc_eps, **common)
        elif spec.driver == "run_tree_tebd":
            times, rdms = _tree.run_tree_tebd(
                density, domain, trunc_eps=context.trunc_eps, **common)
        else:
            raise ValueError(f"unknown mode-tree driver {spec.driver!r}")
        return Result(
            t=times, expect=_expect_from_rdm(rdms, context.obs_ops),
            max_bond=np.array(max_bond), rdm=np.asarray(rdms),
            method=spec.name)

    return SimulationPlan(spec, context, execute=execute)


def _compile_static_tree_plan(model, spec, context):
    """Compile Schroedinger-picture TEBD for every site-tree topology.

    ``TreeFishbone`` and its linear ``Fishbone`` specialization use this directly.
    The one-site multichannel model is lowered to the same tree representation and
    its leading site axis is removed again from the public result.
    """
    from fishbonett.models.fishbone import (
        Fishbone, TreeFishbone, _parse_observable,
    )
    from fishbonett.states.tree import TreeTensorNetwork

    single_system = not isinstance(model, (Fishbone, TreeFishbone))
    if single_system:
        tree = TreeFishbone(
            sites=[model.h], edges=[], baths=[model.coupled_bath])
        initial = [model.system.initial_vector(context.initial)]
    else:
        tree = model._tree() if isinstance(model, Fishbone) else model
        initial = context.initial

    terms = tree.local_terms(context.t_max)
    require_capability(terms, StaticGraphHamiltonian, engine=spec.engine)
    site_gates, edge_gates = terms.gates(context.dt / 2.0)
    parsed = [(name, _parse_observable(value))
              for name, value in context.obs_ops.items()]

    def execute():
        state = TreeTensorNetwork(terms.dims, terms.edges, root=0)
        for site in range(tree.ns):
            state.set_physical(site, tree._initial_vec(initial, site))

        expect = {
            name: (np.full((context.n_steps, tree.ns), np.nan)
                   if kind == "persite"
                   else np.full(context.n_steps, np.nan))
            for name, (kind, _operator, _sites) in parsed
        }
        rdms = np.empty((context.n_steps, tree.ns), dtype=object)
        max_bond = np.empty(context.n_steps, dtype=int)
        for step in range(context.n_steps):
            state.step(
                site_gates, edge_gates, context.bond_dim, context.trunc_eps)
            for site in range(tree.ns):
                rdms[step, site] = state.rdm(site)
            for name, (kind, operator, sites) in parsed:
                if kind == "persite":
                    for site in range(tree.ns):
                        if operator.shape == (tree.de[site], tree.de[site]):
                            expect[name][step, site] = np.trace(
                                rdms[step, site] @ operator).real
                else:
                    expect[name][step] = state.expectation(operator, sites)
            max_bond[step] = tree_peak_bond(state)
            state.move_oc_to(0)

        if len(set(tree.de)) == 1:
            rdm = np.array([
                [rdms[step, site] for site in range(tree.ns)]
                for step in range(context.n_steps)
            ])
        else:
            rdm = rdms
        result = Result(
            t=np.arange(1, context.n_steps + 1) * context.dt,
            expect=expect, rdm=rdm, max_bond=max_bond, method=spec.name,
            meta={"n_sites": tree.ns},
        )
        if not single_system:
            return result
        return Result(
            t=result.t,
            expect={name: values[:, 0] for name, values in result.expect.items()},
            rdm=result.rdm[:, 0], max_bond=result.max_bond,
            method=spec.name,
        )

    return SimulationPlan(spec, context, execute=execute)


def _interaction_frame(model, coupled):
    from fishbonett.frames.interaction_picture import SystemBathIP

    bath = coupled.bath
    phys_dims = [model.h.shape[0]] + [bath.phys_dim] * bath.n_modes
    chain = coupled.compiled_chain()
    builder = SystemBathIP(
        phys_dims, h_sys=model.h, coupling=model.coupling,
        compiled_chain=chain).build()
    return builder, phys_dims


def _multichannel_interaction_frame(model, coupled):
    from fishbonett.frames.multichannel import SystemBathMultiChannel

    bath = coupled.bath
    star = coupled.compiled_star()
    coupling_matrix = star.combine(coupled.operators)
    phys_dims = [model.h.shape[0]] + [star.phys_dim] * star.n_modes
    builder = SystemBathMultiChannel.from_signed_star(
        phys_dims, coupling_matrix, star.frequencies,
        h_sys=model.h).build(n=0)
    return builder, phys_dims


def _compile_swap_plan(model, spec, context):
    from fishbonett.states.mps import SystemBathMPS

    coupled = model.coupled_bath.resolved(context.t_max)
    bath = coupled.bath
    if "multichannel" in spec.models:
        builder, phys_dims = _multichannel_interaction_frame(model, coupled)
    else:
        _check_single_channel(model)
        builder, phys_dims = _interaction_frame(model, coupled)
    state = SystemBathMPS(phys_dims)
    require_capability(builder, SwapGateFactory, engine=spec.engine)
    psi0 = model.system.initial_vector(context.initial)
    state.B[0][:] = 0.0
    for index, amplitude in enumerate(psi0):
        state.B[0][0, index, 0] = amplitude

    return SimulationPlan(
        spec, context,
        step=lambda k: _tebd.symmetric_swap_step(
            state, builder, k * context.dt, context.dt, bath.n_modes,
            context.bond_dim, context.trunc_eps),
        measure_rdm=lambda: state.rdm(0),
        peak_bond=lambda: mps_peak_bond(state),
    )


def _compile_displacement_plan(model, spec, context):
    from fishbonett.evolve.mpo_apply import (
        apply_mpo, bond_dims, compress, product_state,
    )

    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(context.t_max)
    bath = coupled.bath
    builder, phys_dims = _interaction_frame(model, coupled)
    require_capability(builder, DisplacementFactory, engine=spec.engine)
    tensors = product_state(
        phys_dims, model.system.initial_vector(context.initial))
    u_half = _la.expm(-0.5j * context.dt * np.asarray(model.h, complex))

    def step(index):
        nonlocal tensors
        tensors[0] = np.einsum(
            "ij,ajb->aib", u_half, tensors[0])
        tensors = compress(
            apply_mpo(
                tensors,
                builder.displacement_mpo(index * context.dt, context.dt)),
            context.bond_dim, context.trunc_eps)
        tensors[0] = np.einsum(
            "ij,ajb->aib", u_half, tensors[0])

    def measure_rdm():
        rho = np.einsum(
            "lsr,ltr->st", tensors[0], tensors[0].conj())
        return rho / np.trace(rho).real

    return SimulationPlan(
        spec, context, step=step, measure_rdm=measure_rdm,
        peak_bond=lambda: max(bond_dims(tensors)))


def _compile_polaron_plan(model, spec, context):
    from fishbonett.states.mps import SystemBathMPS

    builder, _bath, n_modes, phys_dims = _polaron_builder(model, context)
    require_capability(builder, StaticGateFactory, engine=spec.engine)
    state = SystemBathMPS(phys_dims)
    psi0 = model.system.initial_vector(context.initial)
    state.split_truncate_theta(
        builder.initial_theta(psi0), 0, context.bond_dim, 1e-14)
    gates = builder.gates(context.dt / 2.0)
    return SimulationPlan(
        spec, context,
        step=lambda _index: _tebd.symmetric_static_step(
            state, gates, n_modes, context.bond_dim, context.trunc_eps),
        measure_rdm=lambda: builder.undress_rdm(state.get_theta2(0)),
        peak_bond=lambda: mps_peak_bond(state),
    )


#: Engine key -> plan compiler.  Method names occur only in the taxonomy; this
#: table is the implementation boundary for its coarser engine categories.
PLAN_COMPILERS = {
    "mpo-tdvp": _compile_mpo_plan,
    "modetree": _compile_modetree_plan,
    "swap-tebd": _compile_swap_plan,
    "displacement-mpo": _compile_displacement_plan,
    "polaron-tebd": _compile_polaron_plan,
    "static-tree-tebd": _compile_static_tree_plan,
}


def compile_plan(model, spec, context):
    """Lower a model and resolved registry row to a :class:`SimulationPlan`."""
    try:
        compiler = PLAN_COMPILERS[spec.engine]
    except KeyError:
        raise ValueError(
            f"no simulation-plan compiler for engine {spec.engine!r}") from None
    return compiler(model, spec, context)
