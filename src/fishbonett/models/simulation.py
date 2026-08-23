"""Compile a resolved method into an executable simulation plan.

The model layer resolves user-facing choices to a
:class:`~fishbonett.models.registry.Method`.  This module owns the next boundary:
it lowers that method and the physical problem into a prepared representation, state,
integrator and measurement policy.  The resulting :class:`SimulationPlan` is the
only object :class:`~fishbonett.models.system_bath.SystemBath` has to execute.

Keeping this orchestration outside the physical model prevents the model from
becoming a second representation and integrator registry.  The taxonomy says *which* engine
implements a method; :data:`PLAN_COMPILERS` says how to prepare that engine.
"""
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np
import scipy.linalg as _la

from fishbonett.evolve import modetree as _tree
from fishbonett.evolve import tdvp as _mpo
from fishbonett.evolve import tebd as _tebd
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
    ``step`` advances the state, ``measure_rdm`` returns the laboratory reduced
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


def _tdvp_hooks(context):
    """Driver options shared by every representation supplying a TDVP MPO."""
    hooks = dict(
        observe=lambda tensors: _mpo.measure_rdm(tensors[0]),
        prec=context.kw.get("prec", 1e-4),
        tol=context.kw.get("tol", 1e-7),
        eshift=context.kw.get("eshift", False),
    )
    return hooks


def _schrodinger_representation(model, coupled, name):
    from fishbonett.representations.schrodinger import (
        SchrodingerRepresentation,
    )

    _check_single_channel(model)
    return SchrodingerRepresentation(
        representation=name, h_sys=model.h, coupling=model.coupling,
        bath=coupled.bath)


def _polaron_representation(model, context, name):
    from fishbonett.representations.polaron import PolaronRepresentation

    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(context.t_max)
    bath = coupled.bath
    n_modes, d_sys = bath.n_modes, model.h.shape[0]
    phys_dims = [d_sys] + [bath.phys_dim] * n_modes
    representation = PolaronRepresentation(
        representation=name, h_sys=model.h, coupling=model.coupling,
        bath=bath).build()
    return representation, bath, n_modes, phys_dims


def _compile_tdvp_representation(model, spec, context, coupled):
    """Build the representation object supplying this method's ``tdvp_mpo``."""
    if spec.representation in {"schrodinger-chain", "schrodinger-star"}:
        representation = _schrodinger_representation(
            model, coupled, spec.representation)
        return representation, _tdvp_hooks(context)
    if spec.representation in {"interaction-chain", "interaction-star"}:
        representation, _phys_dims = _interaction_representation(
            model, coupled, spec.representation)
        return representation, _tdvp_hooks(context)
    if spec.representation in {"polaron-chain", "polaron-star"}:
        transformed, _bath, _n_modes, _phys_dims = _polaron_representation(
            model, context, spec.representation)
        initial = model.system.initial_vector(context.initial)

        def prepare(tensors):
            return transformed.initial_mps(initial)

        hooks = dict(
            prepare=prepare,
            observe=transformed.recover_rdm,
            prec=context.kw.get("prec", context.trunc_eps),
            # Padding every bond to a large memory cap is especially wasteful in
            # the displaced representation.  Six seed states converge the prepared
            # coherent block; users can request a larger fixed manifold explicitly.
            initial_bond=context.kw.get(
                "initial_bond", min(context.bond_dim or 6, 6)),
        )
        return transformed, hooks
    raise ValueError(f"engine 'mpo-tdvp' has no compiler for representation {spec.representation!r}")


def _compile_mpo_plan(model, spec, context):
    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(context.t_max)
    representation, hooks = _compile_tdvp_representation(
        model, spec, context, coupled)
    initial = model.system.initial_vector(context.initial)

    def execute():
        times, rdms, max_bond = _mpo.run_mpo_hamiltonian(
            representation, initial=initial,
            dt=context.dt, nsteps=context.n_steps, sweep=spec.driver,
            D=context.bond_dim, chi_max=context.bond_dim,
            eps=context.trunc_eps, krylov=context.krylov,
            seed=context.seed, progress=context.progress,
            bond_expand=context.kw.get("bond_expand"),
            Dplusmax=context.kw.get("Dplusmax", 4), **hooks)
        return Result(
            t=times, expect=_expect_from_rdm(rdms, context.obs_ops),
            max_bond=max_bond, rdm=np.asarray(rdms), method=spec.name)

    return SimulationPlan(spec, context, execute=execute)


def _compile_modetree_plan(model, spec, context):
    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(context.t_max)
    representation, _phys_dims = _interaction_representation(
        model, coupled, spec.representation)

    def execute():
        max_bond = []

        def observe(nodes, root):
            max_bond.append(modetree_peak_bond(nodes))
            return _tree.measure_rdm_oc(nodes, root)

        common = dict(
            init=model.system.initial_vector(context.initial),
            dt=context.dt, nsteps=context.n_steps, D=context.bond_dim,
            observe=observe, seed=context.seed, progress=context.progress,
            **context.kw)
        if spec.driver == "run_tree_tdvp":
            times, rdms = _tree.run_tree_mpo(
                representation, sweep="tdvp1", **common)
        elif spec.driver == "run_tree_tdvp2":
            times, rdms = _tree.run_tree_mpo(
                representation, sweep="tdvp2",
                trunc_eps=context.trunc_eps, **common)
        elif spec.driver == "run_tree_tebd":
            times, rdms = _tree.run_tree_tebd(
                representation, trunc_eps=context.trunc_eps, **common)
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

    horizon = context.bath_horizon or context.t_max
    terms = tree.local_terms(horizon)
    graph_couplings = (model.graph_couplings
                       if isinstance(model, Fishbone) else None)
    if graph_couplings is not None:
        terms.graph_bond.update(graph_couplings)
    gate_time = context.dt / 4.0 if graph_couplings is not None else context.dt / 2.0
    site_gates, edge_gates = terms.tebd_gates(gate_time)
    if graph_couplings is not None:
        from scipy.linalg import expm
        graph_gates = {
            edge: expm(-0.5j * context.dt * value).reshape(
                tree.de[edge[0]], tree.de[edge[1]],
                tree.de[edge[0]], tree.de[edge[1]])
            for edge, value in graph_couplings.items()
        }
    parsed = [(name, _parse_observable(value))
              for name, value in context.obs_ops.items()]

    def execute():
        if context.resume is not None:
            state = context.resume.restore(terms)
        else:
            state = TreeTensorNetwork(terms.dims, terms.edges, root=0)
            if hasattr(initial, "initialize_tree"):
                initial.initialize_tree(state, range(tree.ns))
            else:
                for site in range(tree.ns):
                    state.set_physical(site, tree._initial_vec(initial, site))

        record_steps = [step for step in range(context.n_steps)
                        if (step + 1) % context.observe_every == 0
                        or step + 1 == context.n_steps]
        record_step_set = set(record_steps)
        n_records = len(record_steps)
        expect = {
            name: (np.full((n_records, tree.ns), np.nan)
                   if kind == "persite"
                   else np.full(n_records, np.nan))
            for name, (kind, _operator, _sites) in parsed
        }
        rdms = np.empty((n_records, tree.ns), dtype=object)
        max_bond = np.empty(n_records, dtype=int)
        record = 0
        for step in range(context.n_steps):
            if graph_couplings is None:
                state.step(
                    site_gates, edge_gates, context.bond_dim, context.trunc_eps)
            else:
                from fishbonett.evolve.sitetree import symmetric_graph_step
                state.step(site_gates, edge_gates,
                           context.bond_dim, context.trunc_eps)
                symmetric_graph_step(
                    state, graph_gates, range(tree.ns),
                    context.bond_dim, context.trunc_eps)
                state.step(site_gates, edge_gates,
                           context.bond_dim, context.trunc_eps)
            if context.progress is not None:
                context.progress({
                    "step": step, "n_steps": context.n_steps,
                    "t": context.elapsed + (step + 1) * context.dt,
                    "bond": tree_peak_bond(state), "state": state,
                })
            if step not in record_step_set:
                continue
            for site in range(tree.ns):
                rdms[record, site] = state.rdm(site)
            for name, (kind, operator, sites) in parsed:
                if kind == "persite":
                    for site in range(tree.ns):
                        if operator.shape == (tree.de[site], tree.de[site]):
                            expect[name][record, site] = np.trace(
                                rdms[record, site] @ operator).real
                else:
                    expect[name][record] = state.expectation(operator, sites)
            max_bond[record] = tree_peak_bond(state)
            state.move_oc_to(0)
            record += 1

        if len(set(tree.de)) == 1:
            rdm = np.array([
                [rdms[step, site] for site in range(tree.ns)]
                for step in range(n_records)
            ])
        else:
            rdm = rdms
        elapsed = context.elapsed
        from fishbonett.models.result import SimulationCheckpoint
        checkpoint = SimulationCheckpoint.from_state(
            state, terms, method=spec.name,
            elapsed=elapsed + context.n_steps * context.dt,
            bath_horizon=horizon)
        result = Result(
            t=elapsed + (np.asarray(record_steps) + 1) * context.dt,
            expect=expect, rdm=rdm, max_bond=max_bond, method=spec.name,
            meta={"n_sites": tree.ns}, checkpoint=checkpoint,
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


def _interaction_representation(model, coupled, name):
    from fishbonett.representations.interaction import InteractionRepresentation

    bath = coupled.bath
    phys_dims = [model.h.shape[0]] + [bath.phys_dim] * bath.n_modes
    representation = InteractionRepresentation(
        representation=name, h_sys=model.h,
        coupling=model.coupling,
        bath=bath).build()
    return representation, phys_dims


def _multichannel_interaction_representation(model, coupled, name):
    from fishbonett.representations.multichannel import MultichannelInteractionRepresentation

    bath = coupled.bath
    phys_dims = [model.h.shape[0]] + [bath.phys_dim] * bath.n_modes
    representation = MultichannelInteractionRepresentation(
        representation=name, h_sys=model.h, coupling=coupled.operators,
        bath=bath).build(n=0)
    return representation, phys_dims


def _compile_swap_plan(model, spec, context):
    from fishbonett.states.mps import SystemBathMPS

    coupled = model.coupled_bath.resolved(context.t_max)
    bath = coupled.bath
    if model.coupled_bath.is_multichannel:
        representation, phys_dims = _multichannel_interaction_representation(
            model, coupled, spec.representation)
    else:
        _check_single_channel(model)
        representation, phys_dims = _interaction_representation(
            model, coupled, spec.representation)
    state = SystemBathMPS(phys_dims)
    psi0 = model.system.initial_vector(context.initial)
    state.B[0][:] = 0.0
    for index, amplitude in enumerate(psi0):
        state.B[0][0, index, 0] = amplitude

    return SimulationPlan(
        spec, context,
        step=lambda k: _tebd.symmetric_swap_step(
            state, representation, k * context.dt, context.dt, bath.n_modes,
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
    representation, phys_dims = _interaction_representation(
        model, coupled, spec.representation)
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
                representation.trotter_mpo(
                    index * context.dt, context.dt)),
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

    representation, _bath, n_modes, phys_dims = _polaron_representation(
        model, context, spec.representation)
    state = SystemBathMPS(phys_dims)
    psi0 = model.system.initial_vector(context.initial)
    state.split_truncate_theta(
        representation.initial_theta(psi0), 0, context.bond_dim, 1e-14)
    gates = representation.tebd_gates(context.dt / 2.0)
    return SimulationPlan(
        spec, context,
        step=lambda _index: _tebd.symmetric_static_step(
            state, gates, n_modes, context.bond_dim, context.trunc_eps),
        measure_rdm=lambda: representation.recover_pair_rdm(
            state.get_theta2(0)),
        peak_bond=lambda: mps_peak_bond(state),
    )


def _compile_interaction_fishbone_plan(model, spec, context):
    """Independent interaction-chain baths on an electronic comb."""
    from scipy.linalg import expm
    from fishbonett.models.fishbone import _parse_observable
    from fishbonett.models.result import SimulationCheckpoint, plan_signature
    from fishbonett.representations.interaction import InteractionRepresentation
    from fishbonett.states.tree import TreeTensorNetwork
    from fishbonett.evolve.sitetree import (
        apply_branch_mpo, apply_site,
        symmetric_branch_swap_step, symmetric_graph_step, tdvp_branch_step,
    )

    horizon = context.bath_horizon or context.t_max
    system_count = model.nc
    dims = list(model.de)
    edges = [(site, site + 1) for site in range(system_count - 1)]
    branches = []
    representation_cache = {}
    # Material for the continuation signature: a checkpoint may only be resumed
    # into the same resolved Hamiltonian, and here that means the same electronic
    # terms *and* the same per-site bath resolution.
    signature_arrays = [np.asarray(value) for value in model.sites]
    signature_scalars = [f"horizon={horizon!r}"]
    next_node = system_count
    for site, entry in enumerate(model.baths):
        coupled_baths = model._site_baths(entry)
        if coupled_baths is None:
            continue
        if not isinstance(coupled_baths, list):
            coupled_baths = [coupled_baths]
        # One branch per independent bath. Operators on the same system site need
        # not commute, so their propagators are composed symmetrically below.
        for index, coupled in enumerate(coupled_baths):
            bath = coupled.bath.resolved(horizon)
            cache_key = (
                id(coupled.bath), horizon, model.de[site],
                np.asarray(coupled.operator).tobytes())
            representation = representation_cache.get(cache_key)
            if representation is None:
                representation = InteractionRepresentation(
                    representation="interaction-chain",
                    h_sys=np.zeros_like(model.sites[site]),
                    coupling=coupled.operator, bath=bath).build()
                representation_cache[cache_key] = representation
            nodes = list(range(next_node, next_node + bath.n_modes))
            path = [site, *nodes]
            edges.extend(zip(path[:-1], path[1:]))
            dims.extend([bath.phys_dim] * bath.n_modes)
            branches.append((path, representation))
            signature_arrays.append(np.asarray(coupled.operator))
            signature_arrays.extend((
                np.asarray(representation.frequencies),
                np.asarray(representation.star_couplings),
                np.asarray(representation.star_to_chain),
            ))
            label = f"site={site}"
            if len(coupled_baths) > 1:
                label += f" branch={index}"
            signature_scalars.append(
                f"{label} representation={representation.name} "
                f"modes={bath.n_modes} d={bath.phys_dim}")
            next_node += bath.n_modes

    site_quarter = [expm(-0.25j * context.dt * value)
                    for value in model.sites]
    graph = model.graph_couplings
    if graph is None:
        graph = {(site, site + 1): model.backbone[site]
                 for site in range(system_count - 1)
                 if np.any(model.backbone[site])}
    graph_quarter = {
        edge: expm(-0.25j * context.dt * value).reshape(
            model.de[edge[0]], model.de[edge[1]],
            model.de[edge[0]], model.de[edge[1]])
        for edge, value in graph.items()
    }
    for edge, value in sorted(graph.items()):
        signature_scalars.append(f"edge={tuple(map(int, edge))}")
        signature_arrays.append(np.asarray(value))
    signature = plan_signature(dims, edges, signature_arrays, signature_scalars)

    if context.resume is not None:
        # The bath was discretized for a horizon; resuming against a different one
        # would silently continue into a different environment.
        if not np.isclose(context.resume.bath_horizon, horizon):
            raise ValueError(
                f"checkpoint was produced with bath_horizon="
                f"{context.resume.bath_horizon!r} but this run resolves "
                f"{horizon!r}; pass bath_horizon= to keep them equal")
        state = context.resume.restore_tree(dims, edges, signature)
    else:
        state = TreeTensorNetwork(dims, edges, root=0)
        helper = model._tree()
        for site in range(system_count):
            state.set_physical(site, helper._initial_vec(context.initial, site))

    def branch_step(path, representation, when, delta, operator_cache):
        """Advance one bath branch over ``[when, when + delta]``.

        The comb integrators solve the same H(t) and differ only in how a branch's
        terms reach the state. ``tebd`` uses a swap network, ``trotter-mpo``
        applies one conditional-displacement operator, and ``tdvp2`` evolves
        with the midpoint generator.
        """
        if spec.integrator == "trotter-mpo":
            key = (id(representation), float(when), float(delta), "propagator")
            operator = operator_cache.get(key)
            if operator is None:
                operator = representation.trotter_mpo(when, delta)
                operator_cache[key] = operator
            apply_branch_mpo(
                state, operator, path,
                context.bond_dim, context.trunc_eps)
        elif spec.integrator == "tdvp2":
            # TDVP needs the generator, sampled at the interval midpoint, and in
            # the tree's forward mode order rather than the 1D drivers' reversed one
            key = (id(representation), float(when + 0.5 * delta), "generator")
            operator = operator_cache.get(key)
            if operator is None:
                operator = representation.tdvp_mpo(
                    when + 0.5 * delta, reverse=False)
                operator_cache[key] = operator
            tdvp_branch_step(
                state, operator,
                path, delta, context.bond_dim, context.trunc_eps,
                **{"m": context.krylov})
        else:
            symmetric_branch_swap_step(
                state, representation, path, when, delta,
                context.bond_dim, context.trunc_eps)

    def advance_branches(when):
        # Branches on different system sites commute. Branches sharing a site
        # need not: compose each such group as A(dt/2) B(dt/2) ... Z(dt) ...
        # B(dt/2) A(dt/2), with the second half evaluated on the second time
        # interval. This restores second-order convergence for arbitrary local
        # coupling operators.
        grouped = {}
        for branch in branches:
            grouped.setdefault(branch[0][0], []).append(branch)
        operator_cache = {}
        for group in grouped.values():
            if len(group) == 1:
                path, representation = group[0]
                branch_step(path, representation, when, context.dt,
                            operator_cache)
                continue
            half = 0.5 * context.dt
            for path, representation in group[:-1]:
                branch_step(path, representation, when, half, operator_cache)
            path, representation = group[-1]
            branch_step(path, representation, when, context.dt, operator_cache)
            for path, representation in reversed(group[:-1]):
                branch_step(path, representation, when + half, half,
                            operator_cache)

    def electronic_half_step():
        for site, gate in enumerate(site_quarter):
            apply_site(state, site, gate)
        if graph_quarter:
            symmetric_graph_step(
                state, graph_quarter, range(system_count),
                context.bond_dim, context.trunc_eps)
        for site, gate in enumerate(site_quarter):
            apply_site(state, site, gate)

    parsed = [(name, _parse_observable(value))
              for name, value in context.obs_ops.items()]

    def execute():
        # `elapsed` is physics here, not bookkeeping: the interaction-picture
        # couplings d_n(t) are functions of absolute time, so a continuation that
        # restarted the clock would quietly evolve a different Hamiltonian.
        elapsed = context.elapsed
        record_steps = [step for step in range(context.n_steps)
                        if (step + 1) % context.observe_every == 0
                        or step + 1 == context.n_steps]
        record_set = set(record_steps)
        n_records = len(record_steps)
        expect = {
            name: (np.full((n_records, system_count), np.nan)
                   if kind == "persite" else np.full(n_records, np.nan))
            for name, (kind, _operator, _sites) in parsed}
        rdms = np.empty((n_records, system_count), dtype=object)
        max_bond = np.empty(n_records, dtype=int)
        record = 0
        for step in range(context.n_steps):
            electronic_half_step()
            advance_branches(elapsed + step * context.dt)
            electronic_half_step()
            if context.progress is not None:
                context.progress({
                    "step": step, "n_steps": context.n_steps,
                    "t": elapsed + (step + 1) * context.dt,
                    "bond": tree_peak_bond(state), "state": state,
                })
            if step not in record_set:
                continue
            for site in range(system_count):
                rdms[record, site] = state.rdm(site)
            for name, (kind, operator, sites) in parsed:
                if kind == "persite":
                    for site in range(system_count):
                        if operator.shape == (model.de[site], model.de[site]):
                            expect[name][record, site] = np.trace(
                                rdms[record, site] @ operator).real
                else:
                    expect[name][record] = state.expectation(operator, sites)
            max_bond[record] = tree_peak_bond(state)
            record += 1
        rdm = (np.array([[rdms[t, site] for site in range(system_count)]
                         for t in range(n_records)])
               if len(set(model.de)) == 1 else rdms)
        checkpoint = SimulationCheckpoint.from_tree(
            state, dims, edges, signature=signature, method=spec.name,
            elapsed=elapsed + context.n_steps * context.dt,
            bath_horizon=horizon)
        return Result(
            t=elapsed + (np.asarray(record_steps) + 1) * context.dt,
            expect=expect, rdm=rdm, max_bond=max_bond, method=spec.name,
            meta={"n_sites": system_count, "representation": "interaction-chain"},
            checkpoint=checkpoint)

    return SimulationPlan(spec, context, execute=execute)


#: Engine key -> plan compiler.  Method names occur only in the taxonomy; this
#: table is the implementation boundary for its coarser engine categories.
PLAN_COMPILERS = {
    "mpo-tdvp": _compile_mpo_plan,
    "modetree": _compile_modetree_plan,
    "swap-tebd": _compile_swap_plan,
    "displacement-mpo": _compile_displacement_plan,
    "polaron-tebd": _compile_polaron_plan,
    "static-tree-tebd": _compile_static_tree_plan,
    "interaction-fishbone": _compile_interaction_fishbone_plan,
}


def compile_plan(model, spec, context):
    """Lower a model and resolved registry row to a :class:`SimulationPlan`."""
    try:
        compiler = PLAN_COMPILERS[spec.engine]
    except KeyError:
        raise ValueError(
            f"no simulation-plan compiler for engine {spec.engine!r}") from None
    return compiler(model, spec, context)
