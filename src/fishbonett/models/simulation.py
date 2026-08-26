"""Compile a resolved method into an executable simulation plan.

The model layer resolves user-facing choices to a
:class:`~fishbonett.models.registry.MethodSpec`.  This module owns the next boundary:
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
from fishbonett.evolve import multiset as _multiset
from fishbonett.evolve import tdvp as _mpo
from fishbonett.evolve import tebd as _tebd
from fishbonett.models.propagate import (
    RunCtx, modetree_peak_bond, mps_peak_bond, propagate, tree_peak_bond,
)
from fishbonett.models.result import Result
from fishbonett.randomized import random_seed, svd_statistics
from fishbonett.targets import BathMode

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
    measure_expect: Optional[Callable[[], Mapping[str, complex]]] = None
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
        if self.measure_expect is not None and not has_step_plan:
            raise ValueError("measure_expect is available only on step-based plans")

    @property
    def is_step_based(self):
        """Whether :meth:`run` uses the shared step/measure/collect loop."""
        return self.step is not None

    def run(self):
        """Execute the prepared plan and return a uniform :class:`Result`."""
        with random_seed(
                self.context.seed, backend=self.context.svd_backend):
            if self.execute is not None:
                result = self.execute()
            else:
                result = propagate(
                    self.spec, self.context, step=self.step,
                    rdm=self.measure_rdm, peak_bond=self.peak_bond,
                    expect_from_rdm=_expect_from_rdm,
                    measure_expect=self.measure_expect,
                )
            decomposition_statistics = svd_statistics()
        metadata = {
            "method": self.spec.name,
            "representation": getattr(self.spec, "representation", ""),
            "state_geometry": getattr(self.spec, "state_geometry", ""),
            "integrator": getattr(self.spec, "integrator", ""),
            "dt": self.context.dt,
            "n_steps": self.context.n_steps,
            "observe_every": self.context.observe_every,
            "bath_horizon": self.context.bath_horizon,
            "trunc_eps": self.context.trunc_eps,
            "max_bond_cap": self.context.bond_dim,
            "krylov": self.context.krylov,
            "seed": self.context.seed,
            "svd_backend": self.context.svd_backend,
            "svd": decomposition_statistics,
        }
        metadata.update(result.meta)
        result.meta = metadata
        if (self.context.observe_every > 1
                and len(result.t) == self.context.n_steps):
            indices = np.array([
                step for step in range(self.context.n_steps)
                if (step + 1) % self.context.observe_every == 0
                or step + 1 == self.context.n_steps
            ], dtype=int)
            result.t = result.t[indices]
            result.expect = {
                name: np.asarray(values)[indices]
                for name, values in result.expect.items()
            }
            if result.max_bond is not None:
                result.max_bond = np.asarray(result.max_bond)[indices]
            if result.rdm is not None:
                result.rdm = np.asarray(result.rdm)[indices]
        return result


def _expect_from_rdm(rdms, obs_ops: Mapping[str, np.ndarray]):
    rdms = np.asarray(rdms)
    return {
        name: np.real_if_close(
            np.einsum("tij,ji->t", rdms, np.asarray(operator))
        )
        for name, operator in obs_ops.items()
        if not isinstance(operator, tuple)
    }


def _bath_observables(obs_ops, bath):
    """Validate and return ``name -> (operator, represented mode)``."""
    selected = {}
    for name, value in obs_ops.items():
        if not isinstance(value, tuple):
            continue
        operator, target = value
        if not isinstance(target, BathMode):
            raise TypeError(f"observable {name!r} has an invalid target")
        if target.mode >= bath.n_modes:
            raise ValueError(
                f"observable {name!r} targets bath mode {target.mode}, but the "
                f"resolved bath has {bath.n_modes} modes"
            )
        operator = np.asarray(operator, complex)
        expected = (bath.phys_dim, bath.phys_dim)
        if operator.shape != expected:
            raise ValueError(
                f"observable {name!r} has shape {operator.shape}, expected "
                f"{expected} for a bath mode"
            )
        selected[name] = (operator, target.mode)
    return selected


def _mps_site_rdm(tensors, site, *, physical_middle=False):
    """One-site RDM for ``(left, right, physical)`` MPS tensors."""
    if physical_middle:
        tensors = [np.moveaxis(tensor, 1, -1) for tensor in tensors]
    left = np.ones((1, 1), complex)
    for tensor in tensors[:site]:
        left = np.einsum(
            "ab,arp,bsp->rs", left, tensor, tensor.conj(), optimize=True
        )
    right = np.ones((1, 1), complex)
    for tensor in reversed(tensors[site + 1:]):
        right = np.einsum(
            "arp,bsp,rs->ab", tensor, tensor.conj(), right, optimize=True
        )
    tensor = tensors[site]
    rho = np.einsum(
        "ab,arp,bsq,rs->pq", left, tensor, tensor.conj(), right,
        optimize=True,
    )
    return rho / np.trace(rho)


def _measure_mps_bath(
    tensors, observables, *, reverse_modes=False, physical_middle=False,
):
    values = {}
    for name, (operator, mode) in observables.items():
        site = len(tensors) - mode - 1 if reverse_modes else mode + 1
        values[name] = np.trace(
            _mps_site_rdm(
                tensors, site, physical_middle=physical_middle
            ) @ operator
        )
    return values


def _mps_product_matrix_element(tensors, operators):
    """Contract a product of local operators against TDVP-order MPS tensors."""
    environment = np.ones((1, 1), complex)
    for site, tensor in enumerate(tensors):
        operator = operators.get(site)
        if operator is None:
            operator = np.eye(tensor.shape[2], dtype=complex)
        environment = np.einsum(
            "ab,arp,bsq,qp->rs",
            environment,
            tensor,
            tensor.conj(),
            operator,
            optimize=True,
        )
    return environment.reshape(-1)[0]


def _interleaved_exciton_rdm(tensors, electronic_sites):
    """One-excitation electronic RDM from local two-level MPS sites."""
    sites = tuple(electronic_sites)
    count = len(sites)
    number = np.diag([0.0, 1.0]).astype(complex)
    plus = np.array([[0.0, 0.0], [1.0, 0.0]], complex)
    minus = plus.conj().T
    norm = _mps_product_matrix_element(tensors, {})
    if abs(norm) == 0 or not np.isfinite(norm):
        raise ValueError("cannot measure a zero or non-finite interleaved MPS")
    rho = np.zeros((count, count), complex)
    for left, site in enumerate(sites):
        rho[left, left] = _mps_product_matrix_element(
            tensors, {site: number}
        ) / norm
        for right in range(left + 1, count):
            value = _mps_product_matrix_element(
                tensors,
                {site: minus, sites[right]: plus},
            ) / norm
            rho[left, right] = value
            rho[right, left] = np.conj(value)
    return rho / np.trace(rho)


def _check_single_channel(model):
    if model.system.is_multichannel:
        raise ValueError(
            "this method takes a single coupling operator, but the system was "
            "given a list of them.  A multichannel model is selected by the "
            "SystemBath coupling list and has its own propagators; see "
            "fishbonett.models.registry.")


def _tdvp_hooks(context, driver):
    """Driver options shared by every representation supplying a TDVP MPO."""
    hooks = dict(
        observe=lambda tensors: _mpo.measure_rdm(tensors[0]),
        tol=context.kw.get("tol", 1e-7),
        eshift=context.kw.get("eshift", False),
    )
    if driver == "dtdvp":
        hooks["prec"] = context.kw.get("prec", context.trunc_eps)
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
    coupled = model.coupled_bath.resolved(
        context.bath_horizon or context.t_max
    )
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
        return representation, _tdvp_hooks(context, spec.driver)
    if spec.representation == "interaction-chain":
        representation, _phys_dims = _interaction_representation(
            model, coupled, spec.representation)
        return representation, _tdvp_hooks(context, spec.driver)
    if spec.representation in {"polaron-chain", "polaron-star"}:
        transformed, _bath, _n_modes, _phys_dims = _polaron_representation(
            model, context, spec.representation)
        initial = model.system.initial_vector(context.initial)

        def prepare(tensors):
            return transformed.initial_mps(initial)

        hooks = dict(
            prepare=prepare,
            observe=transformed.recover_rdm,
        )
        if spec.driver == "dtdvp":
            hooks["prec"] = context.kw.get("prec", context.trunc_eps)
        if spec.driver == "tdvp1":
            # One-site TDVP evolves on the requested fixed-bond manifold.
            hooks["initial_bond"] = context.bond_dim
        return transformed, hooks
    raise ValueError(f"engine 'mpo-tdvp' has no compiler for representation {spec.representation!r}")


def _compile_mpo_plan(model, spec, context):
    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(
        context.bath_horizon or context.t_max
    )
    representation, hooks = _compile_tdvp_representation(
        model, spec, context, coupled)
    initial = model.system.initial_vector(context.initial)
    bath_observables = _bath_observables(context.obs_ops, coupled.bath)
    base_observe = hooks.pop("observe")

    def execute():
        targeted = {name: [] for name in bath_observables}

        def observe(tensors):
            measured = _measure_mps_bath(
                tensors, bath_observables,
                reverse_modes=spec.representation.startswith("interaction-"),
            )
            for name, value in measured.items():
                targeted[name].append(value)
            return base_observe(tensors)

        times, rdms, max_bond = _mpo.run_mpo_hamiltonian(
            representation, initial=initial,
            dt=context.dt, nsteps=context.n_steps, sweep=spec.driver,
            bond_dim=context.bond_dim, trunc_eps=context.trunc_eps,
            krylov=context.krylov,
            seed=context.seed, progress=context.progress,
            bond_expand=context.kw.get("bond_expand"),
            observe=observe, **hooks)
        expectations = _expect_from_rdm(rdms, context.obs_ops)
        expectations.update({
            name: np.real_if_close(np.asarray(values))
            for name, values in targeted.items()
        })
        return Result(
            t=times, expect=expectations,
            max_bond=max_bond, rdm=np.asarray(rdms), method=spec.name)

    return SimulationPlan(spec, context, execute=execute)


def _compile_multiset_plan(model, spec, context):
    """Compile coupled two-site TDVP on independent bath MPS components."""
    from fishbonett.models.exciton import ExcitonBath

    if isinstance(model, ExcitonBath):
        return _compile_exciton_multiset_plan(model, spec, context)

    from fishbonett.states.multiset import MultiSetMPS

    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(
        context.bath_horizon or context.t_max
    )
    initial = model.system.initial_vector(context.initial)
    if spec.representation in {"schrodinger-chain", "schrodinger-star"}:
        representation = _schrodinger_representation(
            model, coupled, spec.representation
        )
        state = MultiSetMPS.product(initial, representation.dimensions[1:])
        recover = lambda current: current.system_rdm()
    elif spec.representation == "interaction-chain":
        representation, dimensions = _interaction_representation(
            model, coupled, spec.representation
        )
        state = MultiSetMPS.product(initial, dimensions[1:])
        recover = lambda current: current.system_rdm()
    elif spec.representation in {"polaron-chain", "polaron-star"}:
        representation, _bath, _n_modes, _dimensions = _polaron_representation(
            model, context, spec.representation
        )
        state = MultiSetMPS.from_full_mps(
            representation.initial_mps(initial)
        )
        recover = lambda current: representation.recover_rdm(
            current.combined_mps()
        )
    else:
        raise ValueError(
            "engine 'multiset-tdvp' has no compiler for representation "
            f"{spec.representation!r}"
        )

    bath_observables = _bath_observables(context.obs_ops, coupled.bath)

    def execute():
        targeted = {name: [] for name in bath_observables}

        def observe(current):
            for name, (operator, mode) in bath_observables.items():
                site = (
                    current.n_sites - mode - 1
                    if spec.representation == "interaction-chain" else mode
                )
                targeted[name].append(
                    current.site_expectation(operator, site)
                )
            return recover(current)

        times, rdms, max_bond, set_bonds, final_state = (
            _multiset.run_multiset_mpo_hamiltonian(
                representation,
                state=state,
                dt=context.dt,
                nsteps=context.n_steps,
                bond_dim=context.bond_dim,
                trunc_eps=context.trunc_eps,
                krylov=context.krylov,
                tol=context.kw.get("tol", 1e-7),
                eshift=context.kw.get("eshift", False),
                bond_expand=context.kw.get("bond_expand"),
                observe=observe,
                progress=context.progress,
            )
        )
        expectations = _expect_from_rdm(rdms, context.obs_ops)
        expectations.update({
            name: np.real_if_close(np.asarray(values))
            for name, values in targeted.items()
        })
        return Result(
            t=times,
            expect=expectations,
            max_bond=max_bond,
            rdm=rdms,
            method=spec.name,
            meta={
                "n_sets": final_state.n_sets,
                "final_set_bonds": set_bonds[-1].tolist(),
            },
        )

    return SimulationPlan(spec, context, execute=execute)


def _compile_exciton_multiset_plan(model, spec, context):
    """Flatten independent exciton baths into each multi-set component MPS."""
    from fishbonett.representations.exciton import ExcitonInteractionRepresentation
    from fishbonett.states.multiset import MultiSetMPS

    representation = ExcitonInteractionRepresentation(
        model.h,
        model.baths,
        context.bath_horizon or context.t_max,
        layout="system-first",
    )
    state = MultiSetMPS.product(context.initial, representation.dimensions[1:])

    def execute():
        times, rdms, max_bond, set_bonds, final_state = (
            _multiset.run_multiset_mpo_hamiltonian(
                representation,
                state=state,
                dt=context.dt,
                nsteps=context.n_steps,
                bond_dim=context.bond_dim,
                trunc_eps=context.trunc_eps,
                krylov=context.krylov,
                tol=context.kw.get("tol", 1e-7),
                eshift=context.kw.get("eshift", False),
                bond_expand=context.kw.get("bond_expand"),
                progress=context.progress,
            )
        )
        expectations = _expect_from_rdm(rdms, context.obs_ops)
        expectations["population"] = np.real_if_close(
            np.diagonal(rdms, axis1=1, axis2=2)
        )
        return Result(
            t=times,
            expect=expectations,
            max_bond=max_bond,
            rdm=rdms,
            method=spec.name,
            meta={
                "n_sets": final_state.n_sets,
                "final_set_bonds": set_bonds[-1].tolist(),
                "layout": "system-first baths inside each set MPS",
                "bath_branches": tuple(
                    {
                        "electronic_level": level,
                        "n_modes": branch.len_boson,
                        "phys_dim": branch.pd_boson[0],
                    }
                    for level, branch in representation.branches
                ),
            },
        )

    return SimulationPlan(spec, context, execute=execute)


def _compile_exciton_mpo_plan(model, spec, context):
    """Compile either requested conventional-MPS ordering for an exciton."""
    from fishbonett.representations.exciton import (
        ExcitonInteractionRepresentation,
        _interleaved_initial_mps,
    )

    layout = {
        "system-first-mps": "system-first",
        "interleaved-mps": "interleaved",
    }[spec.state_geometry]
    representation = ExcitonInteractionRepresentation(
        model.h,
        model.baths,
        context.bath_horizon or context.t_max,
        layout=layout,
    )
    if layout == "system-first":
        initial = context.initial
        observe = lambda tensors: _mpo.measure_rdm(tensors[0])
        prepare = None
    else:
        initial = np.array([1.0, 0.0], complex)
        electronic_sites = representation.electronic_sites

        def prepare(_tensors):
            return _interleaved_initial_mps(
                context.initial, representation.dimensions, electronic_sites
            )

        observe = lambda tensors: _interleaved_exciton_rdm(
            tensors, electronic_sites
        )

    def execute():
        times, rdms, max_bond = _mpo.run_mpo_hamiltonian(
            representation,
            initial=initial,
            prepare=prepare,
            observe=observe,
            dt=context.dt,
            nsteps=context.n_steps,
            sweep="tdvp2",
            bond_dim=context.bond_dim,
            trunc_eps=context.trunc_eps,
            krylov=context.krylov,
            tol=context.kw.get("tol", 1e-7),
            eshift=context.kw.get("eshift", False),
            bond_expand=context.kw.get("bond_expand"),
            seed=context.seed,
            progress=context.progress,
        )
        expectations = _expect_from_rdm(rdms, context.obs_ops)
        expectations["population"] = np.real_if_close(
            np.diagonal(rdms, axis1=1, axis2=2)
        )
        return Result(
            t=times,
            expect=expectations,
            max_bond=max_bond,
            rdm=rdms,
            method=spec.name,
            meta={
                "layout": layout,
                "electronic_sites": representation.electronic_sites,
                "bath_branches": tuple(
                    {
                        "electronic_level": level,
                        "n_modes": branch.len_boson,
                        "phys_dim": branch.pd_boson[0],
                    }
                    for level, branch in representation.branches
                ),
            },
        )

    return SimulationPlan(spec, context, execute=execute)


def _compile_exciton_multiset_tree_plan(model, spec, context):
    """Compile one independently truncated bath TTN per excitonic state."""
    from fishbonett.evolve.multiset_tree import run_multiset_tree_hamiltonian
    from fishbonett.representations.exciton import ExcitonInteractionRepresentation
    from fishbonett.states.multiset_tree import MultiSetTreeTensorNetwork

    representation = ExcitonInteractionRepresentation(
        model.h,
        model.baths,
        context.bath_horizon or context.t_max,
        layout="system-first",
    )
    state = MultiSetTreeTensorNetwork.product(
        context.initial,
        representation.tree_dimensions,
        representation.tree_edges,
    )

    def execute():
        times, rdms, max_bond, set_bonds, final_state = (
            run_multiset_tree_hamiltonian(
                representation,
                state=state,
                dt=context.dt,
                nsteps=context.n_steps,
                bond_dim=context.bond_dim,
                trunc_eps=context.trunc_eps,
                krylov=context.krylov,
                tol=context.kw.get("tol", 1e-7),
                eshift=context.kw.get("eshift", False),
                bond_expand=context.kw.get("bond_expand"),
                progress=context.progress,
            )
        )
        expectations = _expect_from_rdm(rdms, context.obs_ops)
        expectations["population"] = np.real_if_close(
            np.diagonal(rdms, axis1=1, axis2=2)
        )
        return Result(
            t=times,
            expect=expectations,
            max_bond=max_bond,
            rdm=rdms,
            method=spec.name,
            meta={
                "n_sets": final_state.n_sets,
                "final_set_bonds": set_bonds[-1].tolist(),
                "layout": "bath TTN inside each excitonic set",
                "tree_dimensions": representation.tree_dimensions,
                "tree_edges": representation.tree_edges,
                "bath_branches": tuple(
                    {
                        "electronic_level": level,
                        "nodes": representation.tree_mode_nodes[level],
                        "n_modes": branch.len_boson,
                        "phys_dim": branch.pd_boson[0],
                    }
                    for level, branch in representation.branches
                ),
            },
        )

    return SimulationPlan(spec, context, execute=execute)


def _compile_modetree_plan(model, spec, context):
    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(
        context.bath_horizon or context.t_max
    )
    representation, _phys_dims = _interaction_representation(
        model, coupled, spec.representation)
    bath_observables = _bath_observables(context.obs_ops, coupled.bath)

    def execute():
        max_bond = []
        targeted = {name: [] for name in bath_observables}

        def observe(nodes, root):
            max_bond.append(modetree_peak_bond(nodes))
            mode_nodes = {
                node.mode: node.id for node in nodes if node.mode is not None
            }
            for name, (operator, mode) in bath_observables.items():
                rho = _tree.measure_node_rdm(nodes, mode_nodes[mode])
                targeted[name].append(np.trace(rho @ operator))
            return _tree.measure_rdm_oc(nodes, root)

        common = dict(
            init=model.system.initial_vector(context.initial),
            dt=context.dt, nsteps=context.n_steps,
            observe=observe, seed=context.seed, progress=context.progress,
            **context.kw)
        if spec.driver == "run_tree_tebd":
            times, rdms = _tree.run_tree_tebd(
                representation, bond_dim=context.bond_dim,
                trunc_eps=context.trunc_eps, **common)
        else:
            raise ValueError(f"unknown mode-tree driver {spec.driver!r}")
        expectations = _expect_from_rdm(rdms, context.obs_ops)
        expectations.update({
            name: np.real_if_close(np.asarray(values))
            for name, values in targeted.items()
        })
        return Result(
            t=times, expect=expectations,
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
        _resolve_observable_target,
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
    terms = model.local_terms(horizon) if isinstance(model, Fishbone) else tree.local_terms(horizon)
    graph_couplings = (model.graph_couplings
                       if isinstance(model, Fishbone) else None)
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
    parsed = []
    observable_targets = {}
    for name, value in context.obs_ops.items():
        parsed_value = _resolve_observable_target(
            _parse_observable(value, tree.de, name),
            terms.dims, terms.bath_nodes, name,
        )
        parsed.append((name, parsed_value))
        kind, operator, nodes = parsed_value
        observable_targets[name] = tuple(
            site for site in range(tree.ns)
            if operator.shape == (tree.de[site], tree.de[site])
        ) if kind == "persite" else tuple(nodes)

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
            name: (np.full((n_records, tree.ns), np.nan, dtype=complex)
                   if kind == "persite"
                   else np.full(n_records, np.nan, dtype=complex))
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
                    "bond": tree_peak_bond(state), "rdm": None,
                    "state": state,
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
                                rdms[record, site] @ operator)
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
            expect={name: np.real_if_close(values)
                    for name, values in expect.items()},
            rdm=rdm, max_bond=max_bond, method=spec.name,
            meta={
                "n_sites": tree.ns,
                "bath_branches": tuple(dict(item) for item in terms.bath_branches),
                "observable_targets": observable_targets,
            }, checkpoint=checkpoint,
        )
        if not single_system:
            return result
        return Result(
            t=result.t,
            expect={
                name: values[:, 0] if np.ndim(values) == 2 else values
                for name, values in result.expect.items()
            },
            rdm=result.rdm[:, 0], max_bond=result.max_bond,
            method=spec.name, meta=result.meta, checkpoint=result.checkpoint,
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

    coupled = model.coupled_bath.resolved(
        context.bath_horizon or context.t_max
    )
    bath = coupled.bath
    if model.coupled_bath.is_multichannel:
        representation, phys_dims = _multichannel_interaction_representation(
            model, coupled, spec.representation)
    else:
        _check_single_channel(model)
        representation, phys_dims = _interaction_representation(
            model, coupled, spec.representation)
    state = SystemBathMPS(phys_dims)
    bath_observables = _bath_observables(context.obs_ops, bath)
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
        measure_expect=lambda: {
            name: state.expectation(operator, mode + 1)
            for name, (operator, mode) in bath_observables.items()
        },
    )


def _compile_displacement_plan(model, spec, context):
    from fishbonett.evolve.mpo_apply import (
        apply_mpo, bond_dims, compress, product_state,
    )

    _check_single_channel(model)
    coupled = model.coupled_bath.resolved(
        context.bath_horizon or context.t_max
    )
    representation, phys_dims = _interaction_representation(
        model, coupled, spec.representation)
    bath_observables = _bath_observables(context.obs_ops, coupled.bath)
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
        return rho / np.trace(rho)

    return SimulationPlan(
        spec, context, step=step, measure_rdm=measure_rdm,
        peak_bond=lambda: max(bond_dims(tensors)),
        measure_expect=lambda: _measure_mps_bath(
            tensors, bath_observables, physical_middle=True
        ))


def _compile_polaron_plan(model, spec, context):
    from fishbonett.states.mps import SystemBathMPS

    representation, _bath, n_modes, phys_dims = _polaron_representation(
        model, context, spec.representation)
    bath_observables = _bath_observables(context.obs_ops, _bath)
    state = SystemBathMPS(phys_dims)
    psi0 = model.system.initial_vector(context.initial)
    # Factorize the analytically prepared pair without applying the propagation
    # truncation threshold; this cutoff removes only floating-point null values.
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
        measure_expect=lambda: {
            name: state.expectation(operator, mode + 1)
            for name, (operator, mode) in bath_observables.items()
        },
    )


def _kron_all(factors):
    """Kronecker product of a non-empty sequence, in the given site order."""
    out = np.asarray(factors[0], complex)
    for factor in factors[1:]:
        out = np.kron(out, np.asarray(factor, complex))
    return out


def _polaron_dressed_system_operator(
    operator, sites, representations, all_system_dims,
):
    """Dress a multi-site laboratory operator and return ``(matrix, nodes)``.

    The returned node list interleaves every electronic site with the first mode
    of its independent polaron chain.  Sites without a bath remain bare.
    """
    system_dims = [all_system_dims[site] for site in sites]
    total = int(np.prod(system_dims, dtype=int))
    operator = np.asarray(operator, complex)
    if operator.shape != (total, total):
        raise ValueError(
            f"system observable has shape {operator.shape}, expected "
            f"{(total, total)}"
        )
    basis = _kron_all([
        (representations[site][0].eigenvectors
         if site in representations
         else np.eye(all_system_dims[site], dtype=complex))
        for site in sites
    ])
    transformed = basis.conj().T @ operator @ basis
    output_dims = []
    nodes = []
    for site in sites:
        item = representations.get(site)
        if item is None:
            output_dims.append(all_system_dims[site])
            nodes.append(site)
        else:
            representation, mode_node = item
            output_dims.extend((representation.pd_sys,
                                representation.pd_boson[0]))
            nodes.extend((site, mode_node))
    dimension = int(np.prod(output_dims, dtype=int))
    dressed = np.zeros((dimension, dimension), complex)
    for row in np.ndindex(*system_dims):
        row_flat = np.ravel_multi_index(row, system_dims)
        for column in np.ndindex(*system_dims):
            column_flat = np.ravel_multi_index(column, system_dims)
            coefficient = transformed[row_flat, column_flat]
            if abs(coefficient) < 1e-14:
                continue
            factors = []
            for position, site in enumerate(sites):
                left, right = row[position], column[position]
                item = representations.get(site)
                if item is None:
                    transition = np.zeros(
                        (all_system_dims[site],) * 2, complex
                    )
                    transition[left, right] = 1.0
                    factors.append(transition)
                else:
                    representation, _mode_node = item
                    factors.extend((
                        np.outer(
                            representation.eigenvectors[:, left],
                            representation.eigenvectors[:, right].conj(),
                        ),
                        representation.displacement_operator(
                            0,
                            representation.eigenvalues[left]
                            - representation.eigenvalues[right],
                        ),
                    ))
            dressed += coefficient * _kron_all(factors)
    return dressed, nodes


def _polaron_graph_gate_mpo(
    coupling, left, right, representations, system_dims, intermediate_dims,
    duration,
):
    """Compile one independently dressed electronic coupling to a path MPO."""
    from fishbonett.representations._mpo import dense_operator_mpo

    rep_left = representations.get(left)
    rep_right = representations.get(right)
    eig_left = (rep_left[0].eigenvectors if rep_left is not None
                else np.eye(system_dims[left], dtype=complex))
    eig_right = (rep_right[0].eigenvectors if rep_right is not None
                 else np.eye(system_dims[right], dtype=complex))
    values_left = (rep_left[0].eigenvalues if rep_left is not None
                   else np.zeros(system_dims[left]))
    values_right = (rep_right[0].eigenvalues if rep_right is not None
                    else np.zeros(system_dims[right]))
    transformed = np.kron(eig_left, eig_right).conj().T
    transformed = transformed @ np.asarray(coupling, complex)
    transformed = transformed @ np.kron(eig_left, eig_right)

    endpoint_dims = []
    if rep_left is not None:
        endpoint_dims.append(rep_left[0].pd_boson[0])
    endpoint_dims.append(system_dims[left])
    left_core_count = len(endpoint_dims)
    endpoint_dims.append(system_dims[right])
    if rep_right is not None:
        endpoint_dims.append(rep_right[0].pd_boson[0])
    endpoint_hamiltonian = np.zeros(
        (int(np.prod(endpoint_dims, dtype=int)),) * 2, complex
    )
    pair_dims = (system_dims[left], system_dims[right])
    for left_state in np.ndindex(*pair_dims):
        left_flat = np.ravel_multi_index(left_state, pair_dims)
        for right_state in np.ndindex(*pair_dims):
            right_flat = np.ravel_multi_index(right_state, pair_dims)
            coefficient = transformed[left_flat, right_flat]
            if abs(coefficient) < 1e-14:
                continue
            factors = []
            if rep_left is not None:
                representation = rep_left[0]
                factors.append(representation.displacement_operator(
                    0, values_left[left_state[0]] - values_left[right_state[0]]
                ))
            factors.append(np.outer(
                eig_left[:, left_state[0]], eig_left[:, right_state[0]].conj()
            ))
            factors.append(np.outer(
                eig_right[:, left_state[1]], eig_right[:, right_state[1]].conj()
            ))
            if rep_right is not None:
                representation = rep_right[0]
                factors.append(representation.displacement_operator(
                    0, values_right[left_state[1]] - values_right[right_state[1]]
                ))
            endpoint_hamiltonian += coefficient * _kron_all(factors)

    gate = _la.expm(-1j * duration * endpoint_hamiltonian)
    endpoint_mpo = dense_operator_mpo(gate, endpoint_dims)
    if not intermediate_dims:
        return endpoint_mpo
    bond = endpoint_mpo[left_core_count - 1].shape[1]
    identities = []
    for dimension in intermediate_dims:
        core = np.zeros((bond, bond, dimension, dimension), complex)
        identity = np.eye(dimension, dtype=complex)
        for index in range(bond):
            core[index, index] = identity
        identities.append(core)
    return [
        *endpoint_mpo[:left_core_count],
        *identities,
        *endpoint_mpo[left_core_count:],
    ]


def _compile_polaron_fishbone_plan(model, spec, context):
    """Independent Lang--Firsov bath transformations on a Fishbone model.

    Each bath is mapped with the displacement-weighted chain transformation.
    The local displacement then lives on its first chain mode.  Electronic
    couplings are transformed by both endpoint displacements and applied as
    exact path MPO gates; the state remains the model's tree tensor network.
    """
    from fishbonett.evolve.sitetree import apply_branch_mpo, apply_edge
    from fishbonett.models.fishbone import _parse_observable
    from fishbonett.models.result import SimulationCheckpoint, plan_signature
    from fishbonett.representations.polaron import PolaronRepresentation
    from fishbonett.states.tree import TreeTensorNetwork

    horizon = context.bath_horizon or context.t_max
    system_count = model.nc
    dims = list(model.de)
    edges = [(site, site + 1) for site in range(system_count - 1)]
    representations = {}
    bath_nodes = {}
    bath_branches = []
    signature_arrays = [np.asarray(value) for value in model.sites]
    signature_scalars = [f"horizon={horizon!r}"]
    next_node = system_count
    for site, entry in enumerate(model.baths):
        coupled_baths = model._site_baths(entry)
        if not coupled_baths:
            continue
        if not isinstance(coupled_baths, list):
            coupled_baths = [coupled_baths]
        if len(coupled_baths) != 1:
            raise ValueError(
                "polaron-chain on a Fishbone currently supports at most one "
                "independent bath per system site"
            )
        coupled = coupled_baths[0]
        bath = coupled.bath.resolved(horizon)
        representation = PolaronRepresentation(
            representation="polaron-chain", h_sys=model.sites[site],
            coupling=coupled.operator, bath=bath,
        ).build()
        nodes = list(range(next_node, next_node + bath.n_modes))
        path = [site, *nodes]
        edges.extend(zip(path[:-1], path[1:]))
        dims.extend([bath.phys_dim] * bath.n_modes)
        representations[site] = (representation, nodes[0])
        for mode, node in enumerate(nodes):
            bath_nodes[BathMode(site, 0, mode)] = node
        bath_branches.append({
            "system_site": site,
            "bath": 0,
            "representation": "polaron-chain",
            "first_node": nodes[0],
            "n_modes": bath.n_modes,
            "phys_dim": bath.phys_dim,
            "system_coupling": None,
        })
        signature_arrays.extend((
            np.asarray(coupled.operator),
            np.asarray(representation.frequencies),
            np.asarray(representation.hoppings),
            np.asarray(representation.displacements),
        ))
        signature_scalars.append(
            f"site={site} representation=polaron-chain "
            f"modes={bath.n_modes} d={bath.phys_dim}"
        )
        next_node += bath.n_modes

    site_gates = [None] * len(dims)
    edge_gates = {
        (site, site + 1): np.eye(
            model.de[site] * model.de[site + 1], dtype=complex
        ).reshape(
            model.de[site], model.de[site + 1],
            model.de[site], model.de[site + 1],
        )
        for site in range(system_count - 1)
    }
    for site, value in enumerate(model.sites):
        item = representations.get(site)
        if item is None:
            site_gates[site] = _la.expm(-0.25j * context.dt * value)
            continue
        representation, first_node = item
        gates = representation.tebd_gates(context.dt / 4.0)
        edge_gates[(site, first_node)] = gates[0]
        for offset, gate in enumerate(gates[1:]):
            edge_gates[(first_node + offset, first_node + offset + 1)] = gate

    graph = model.graph_couplings
    if graph is None:
        graph = {
            (site, site + 1): model.backbone[site]
            for site in range(system_count - 1)
            if np.any(model.backbone[site])
        }
    graph_mpos = []
    for (left, right), value in sorted(graph.items()):
        if left > right:
            left, right = right, left
        path = []
        if left in representations:
            path.append(representations[left][1])
        path.extend(range(left, right + 1))
        if right in representations:
            path.append(representations[right][1])
        mpo = _polaron_graph_gate_mpo(
            value, left, right, representations, model.de,
            model.de[left + 1:right], context.dt / 2.0,
        )
        if len(mpo) != len(path):
            raise RuntimeError("dressed graph MPO does not match its tree path")
        graph_mpos.append((path, mpo))
        signature_scalars.append(f"edge={(left, right)}")
        signature_arrays.append(np.asarray(value))
    signature = plan_signature(dims, edges, signature_arrays, signature_scalars)

    if context.resume is not None:
        if not np.isclose(context.resume.bath_horizon, horizon):
            raise ValueError(
                f"checkpoint was produced with bath_horizon="
                f"{context.resume.bath_horizon!r} but this run resolves "
                f"{horizon!r}; pass bath_horizon= to keep them equal"
            )
        state = context.resume.restore_tree(dims, edges, signature)
    else:
        state = TreeTensorNetwork(dims, edges, root=0)
        helper = model._tree()
        for site in range(system_count):
            state.set_physical(site, helper._initial_vec(context.initial, site))
        # U_P |psi_sys>|0> is a product of commuting conditional displacements,
        # one on each independent system--bath pair.
        for site, (representation, first_node) in representations.items():
            state.move_oc_to(site)
            apply_edge(
                state, site, first_node, representation.initial_pair_gate(),
                context.bond_dim, 1e-14,
            )

    parsed = []
    observable_targets = {}
    for name, value in context.obs_ops.items():
        kind, operator, targets = _parse_observable(value, model.de, name)
        if kind == "persite":
            parsed.append((name, kind, operator, None, None))
            observable_targets[name] = tuple(
                site for site in range(system_count)
                if operator.shape == (model.de[site], model.de[site])
            )
            continue
        if any(isinstance(target, BathMode) for target in targets):
            if not all(isinstance(target, BathMode) for target in targets):
                raise ValueError(
                    "mixed system and represented-bath composite observables "
                    "are not implemented for polaron-chain"
                )
            nodes = []
            for target in targets:
                try:
                    nodes.append(bath_nodes[target])
                except KeyError as exc:
                    raise ValueError(
                        f"{name} targets unavailable bath mode {target}"
                    ) from exc
            expected = int(np.prod([dims[node] for node in nodes]))
            if operator.shape != (expected, expected):
                raise ValueError(
                    f"{name} has shape {operator.shape}, expected "
                    f"{(expected, expected)} for resolved tensor nodes {nodes}"
                )
            parsed.append((name, "represented", operator, nodes, None))
            observable_targets[name] = tuple(nodes)
            continue
        if len(targets) == 1:
            site = targets[0]
            parsed.append((name, "lab-local", operator, [site], None))
            observable_targets[name] = (site,)
            continue
        dressed, nodes = _polaron_dressed_system_operator(
            operator, targets, representations, model.de
        )
        parsed.append((name, "dressed", dressed, nodes, None))
        observable_targets[name] = tuple(nodes)

    def laboratory_rdm(site):
        item = representations.get(site)
        if item is None:
            return state.rdm(site)
        representation, first_node = item
        return representation.recover_joint_rdm(
            state.joint_rdm([site, first_node])
        )

    def execute():
        elapsed = context.elapsed
        record_steps = [
            step for step in range(context.n_steps)
            if (step + 1) % context.observe_every == 0
            or step + 1 == context.n_steps
        ]
        record_set = set(record_steps)
        n_records = len(record_steps)
        expect = {}
        for name, kind, _operator, _nodes, _unused in parsed:
            expect[name] = (
                np.full((n_records, system_count), np.nan, dtype=complex)
                if kind == "persite" else np.full(n_records, np.nan, dtype=complex)
            )
        rdms = np.empty((n_records, system_count), dtype=object)
        max_bond = np.empty(n_records, dtype=int)
        record = 0
        for step in range(context.n_steps):
            state.step(
                site_gates, edge_gates,
                context.bond_dim, context.trunc_eps,
            )
            for path, mpo in graph_mpos:
                apply_branch_mpo(
                    state, mpo, path,
                    context.bond_dim, context.trunc_eps,
                )
            for path, mpo in reversed(graph_mpos):
                apply_branch_mpo(
                    state, mpo, path,
                    context.bond_dim, context.trunc_eps,
                )
            state.step(
                site_gates, edge_gates,
                context.bond_dim, context.trunc_eps,
            )
            if context.progress is not None:
                context.progress({
                    "step": step,
                    "n_steps": context.n_steps,
                    "t": elapsed + (step + 1) * context.dt,
                    "bond": tree_peak_bond(state),
                    "rdm": None,
                    "state": state,
                })
            if step not in record_set:
                continue
            for site in range(system_count):
                rdms[record, site] = laboratory_rdm(site)
            for name, kind, operator, nodes, _unused in parsed:
                if kind == "persite":
                    for site in range(system_count):
                        if operator.shape == (model.de[site], model.de[site]):
                            expect[name][record, site] = np.trace(
                                rdms[record, site] @ operator
                            )
                elif kind == "lab-local":
                    expect[name][record] = np.trace(
                        rdms[record, nodes[0]] @ operator
                    )
                else:
                    expect[name][record] = state.expectation(operator, nodes)
            max_bond[record] = tree_peak_bond(state)
            record += 1
        rdm = (
            np.array([[rdms[t, site] for site in range(system_count)]
                      for t in range(n_records)])
            if len(set(model.de)) == 1 else rdms
        )
        checkpoint = SimulationCheckpoint.from_tree(
            state, dims, edges, signature=signature, method=spec.name,
            elapsed=elapsed + context.n_steps * context.dt,
            bath_horizon=horizon,
        )
        return Result(
            t=elapsed + (np.asarray(record_steps) + 1) * context.dt,
            expect={name: np.real_if_close(values)
                    for name, values in expect.items()},
            rdm=rdm,
            max_bond=max_bond,
            method=spec.name,
            meta={
                "n_sites": system_count,
                "representation": "polaron-chain",
                "bath_branches": tuple(dict(item) for item in bath_branches),
                "observable_targets": observable_targets,
            },
            checkpoint=checkpoint,
        )

    return SimulationPlan(spec, context, execute=execute)


def _compile_interaction_fishbone_plan(model, spec, context):
    """Independent interaction-chain baths on an electronic comb."""
    from scipy.linalg import expm
    from fishbonett.models.fishbone import (
        _parse_observable, _resolve_observable_target,
    )
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
    bath_nodes = {}
    bath_branches = []
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
            for mode, node in enumerate(nodes):
                bath_nodes[BathMode(site, index, mode)] = node
            bath_branches.append({
                "system_site": site,
                "bath": index,
                "representation": "interaction-chain",
                "first_node": nodes[0],
                "n_modes": bath.n_modes,
                "phys_dim": bath.phys_dim,
                "system_coupling": None,
            })
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

    parsed = []
    observable_targets = {}
    for name, value in context.obs_ops.items():
        parsed_value = _resolve_observable_target(
            _parse_observable(value, model.de, name),
            dims, bath_nodes, name,
        )
        parsed.append((name, parsed_value))
        kind, operator, nodes = parsed_value
        observable_targets[name] = tuple(
            site for site in range(system_count)
            if operator.shape == (model.de[site], model.de[site])
        ) if kind == "persite" else tuple(nodes)

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
            name: (np.full((n_records, system_count), np.nan, dtype=complex)
                   if kind == "persite"
                   else np.full(n_records, np.nan, dtype=complex))
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
                    "bond": tree_peak_bond(state), "rdm": None,
                    "state": state,
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
                                rdms[record, site] @ operator)
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
            expect={name: np.real_if_close(values)
                    for name, values in expect.items()},
            rdm=rdm, max_bond=max_bond, method=spec.name,
            meta={
                "n_sites": system_count,
                "representation": "interaction-chain",
                "bath_branches": tuple(dict(item) for item in bath_branches),
                "observable_targets": observable_targets,
            },
            checkpoint=checkpoint)

    return SimulationPlan(spec, context, execute=execute)


#: Engine key -> plan compiler.  Method names occur only in the taxonomy; this
#: table is the implementation boundary for its coarser engine categories.
PLAN_COMPILERS = {
    "mpo-tdvp": _compile_mpo_plan,
    "exciton-mpo-tdvp": _compile_exciton_mpo_plan,
    "multiset-tdvp": _compile_multiset_plan,
    "multiset-tree-tdvp": _compile_exciton_multiset_tree_plan,
    "modetree": _compile_modetree_plan,
    "swap-tebd": _compile_swap_plan,
    "displacement-mpo": _compile_displacement_plan,
    "polaron-tebd": _compile_polaron_plan,
    "polaron-fishbone": _compile_polaron_fishbone_plan,
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
