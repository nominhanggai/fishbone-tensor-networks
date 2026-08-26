"""The single-system models: one system site coupled to one bath.

This class supports two models:

* ``system-bath`` -- one system, one bath, one coupling operator. All five
  representations on conventional or multi-set MPSs, plus the binary-tree
  interaction-chain method;
* ``multichannel`` -- several couplings on shared modes, selected automatically
  by giving ``SystemBath(coupling=...)`` a *list* of operators.

See :mod:`fishbonett.models.registry` for supported representation,
state-geometry, and integrator combinations.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from numpy.typing import ArrayLike

from fishbonett.bath.coupled import CoupledBath, bind_bath
from fishbonett.bath.spec import Bath
from fishbonett.linalg import Truncation
from fishbonett.system import System
from fishbonett.models.propagate import (
    RunCtx, _resolve_continuation, _resolve_sampling_options,
    resolve_time_grid,
)
from fishbonett.models import registry
from fishbonett.models.registry import (
    BOND_CAP_REQUIRED_METHODS, methods_of, unknown_method_error,
)
from fishbonett.models.result import Result, SimulationCheckpoint
from fishbonett.targets import BathMode

__all__ = ["SystemBath"]

#: Default when no method axis is supplied. Multichannel baths select their own
#: compatible default after the coupling topology is known.
_DEFAULT_METHOD = "interaction-chain-tree-tebd"


def _bond_growing_siblings(method):
    """Methods in the same ``(representation, model)`` as ``method`` that grow their own
    bond dimension -- what to suggest when a fixed-bond method is asked for
    ``bond_dim=None``."""
    spec = registry.METHODS.get(method.lower().replace("_", "-"))
    if spec is None:
        return []
    siblings = {
        name
        for model_key in spec.models
        for name in methods_of(model_key, spec.representation)
        if name not in BOND_CAP_REQUIRED_METHODS
    }
    return sorted(siblings)


def _normalize_observables(observables, system_dim):
    """Validate single-system observables without discarding bath targets."""
    from fishbonett.models.fishbone import _parse_observable

    normalized = {}
    for name, value in observables.items():
        kind, operator, targets = _parse_observable(
            value, [system_dim], f"observables[{name!r}]"
        )
        if kind == "persite":
            normalized[name] = operator
            continue
        if len(targets) != 1:
            raise ValueError(
                "SystemBath observables target one system or bath mode at a time"
            )
        target = targets[0]
        if isinstance(target, BathMode):
            if target.system_site != 0 or target.bath != 0:
                raise ValueError(
                    "SystemBath has only system_site=0 and bath=0"
                )
            normalized[name] = (operator, target)
        elif target == 0:
            normalized[name] = operator
        else:
            raise ValueError("SystemBath has only system site 0")
    return normalized


class SystemBath:
    """**One** system site coupled to **one** bath -- a *single-system* model.

    Only the topology is fixed: there is no site graph and no per-site baths.  The
    system itself is unrestricted -- ``h`` is any ``(d, d)`` Hermitian matrix and the
    coupling ``O`` any ``(d, d)`` Hermitian operator, and every method here supports
    an arbitrary system dimension, a general coupling and an arbitrary initial
    state.  For several system sites, each with its own bath(s), use the *multi-site*
    models :class:`~fishbonett.models.fishbone.Fishbone` (sites on a 1D backbone) or
    :class:`~fishbonett.models.fishbone.TreeFishbone` (sites in any loop-free tree).

    This one class covers the ``system-bath`` and ``multichannel`` models.  The
    Hamiltonian ``representation`` and the ``state_geometry`` are other axes of
    ``run``, not separate models.

    When the system has *distinct* internal degrees of freedom (e.g. a spin **and**
    a vibration), prefer to keep each on its own site with ``TreeFishbone`` (a spin
    site and a vibration site joined by an edge, with the bath on the spin) --
    putting ``spin (x) vibration`` on a single ``d = 2*d_vib`` site here works but
    defeats the MPS advantage.  Passing a list as ``SystemBath.coupling`` routes
    through the shared-mode multichannel model.

    Parameters
    ----------
    h : (d, d) array
        System Hamiltonian.
    coupling : (d, d) array, or list of (d, d) arrays
        System operator(s) coupling to the bath (a list for a multichannel bath).
    bath : Bath
    """

    def __init__(
        self,
        h: ArrayLike,
        coupling: ArrayLike | Sequence[ArrayLike],
        bath: Bath | CoupledBath,
    ) -> None:
        """Validate the system and bind its operator or channels to one bath."""
        # `System` validates h and the coupling(s) once -- square, matching
        # dimension, Hermitian -- and keeps a multichannel coupling as a *list*
        # rather than collapsing it into a 3-D array.
        self.system = System(h, coupling)
        # mirrored as plain attributes: the validated forms, which the representation
        # builders take as `h_sys=` / `coupling=`
        self.h = self.system.h
        self.coupling = self.system.coupling
        # Every driver reads this binding for channel topology and star couplings.
        self.coupled_bath = bind_bath(bath, self.coupling)
        self.bath = self.coupled_bath.bath

    # -- public API ----------------------------------------------------------
    def run(
        self,
        *,
        dt: float,
        t_max: float | None = None,
        n_steps: int | None = None,
        method: str | None = None,
        model: str | None = None,
        representation: str | None = None,
        state_geometry: str | None = None,
        integrator: str | None = None,
        trunc: Truncation | float | None = None,
        bond_dim: int | None = None,
        trunc_eps: float | None = None,
        observables: Mapping[str, object] | None = None,
        initial: str | ArrayLike | None = None,
        krylov: int = 25,
        seed: int | None = 0,
        resume: SimulationCheckpoint | None = None,
        bath_horizon: float | None = None,
        progress: Callable[[dict[str, object]], None] | None = None,
        observe_every: int = 1,
        svd_backend: str = "auto",
        **engine_kw: object,
    ) -> Result:
        """Propagate and return a :class:`Result`.

        .. rubric:: Method selection

        Select a run with the four public axes::

            sb.run(dt=..., t_max=..., representation="interaction-chain",
                   state_geometry="mps",
                   integrator="tdvp2")

        ``model`` specifies what is coupled to what. ``representation`` specifies
        how ``H`` is written: ``"schrodinger-chain"``,
        ``"schrodinger-star"``, ``"interaction-chain"``,
        ``"polaron-chain"``, or ``"polaron-star"``. ``state_geometry`` is the
        tensor-network geometry (``"mps"``, ``"multi-set-mps"``,
        ``"binary-tree"``, or ``"tree"``), and
        ``integrator`` how a step is taken (``"tebd"``, ``"tdvp1"``, ``"tdvp2"``,
        ``"a1tdvp"``, ``"trotter-mpo"``).  Omit an axis and it is inferred when only
        one combination fits; if several do, the error lists them.

        Representation values use the full names above; partial values such as
        ``representation="polaron"`` are rejected. For ``interaction-chain`` the
        discretized star bath is put in
        the interaction picture with respect to its free Hamiltonian and the
        resulting time-dependent coupling is then transformed star-to-chain.

        A method name provides a shorthand for the same selection::

            sb.run(dt=..., t_max=..., method="interaction-chain-tdvp2")

        ``"interaction-chain-tdvp2"`` selects ``(interaction-chain, mps, tdvp2)``.
        Do not combine ``method=`` with individual axis arguments.
        ``describe_taxonomy()`` prints the available combinations.

        With no method or axes, a single-channel model defaults to
        ``"interaction-chain-tree-tebd"``. A multichannel model defaults to its
        registry-defined method. Pass a method explicitly in published scripts
        so the numerical representation remains visible in the input.

        Supported models:

        * **system-bath** -- 1 system + 1 bath + 1 coupling operator, in all five
          representations.  *schrodinger-chain*
          (``schrodinger-chain-tdvp1 | schrodinger-chain-tdvp2 |
          schrodinger-chain-a1tdvp``) and *schrodinger-star*
          (``schrodinger-star-tdvp1 | schrodinger-star-tdvp2 |
          schrodinger-star-a1tdvp``) -- static, so the
          MPO is built once and TDVP conserves energy, at the cost of the largest
          bond dimensions.  *interaction-chain*
          (``interaction-chain-tebd``, ``interaction-chain-trotter-mpo``,
          ``interaction-chain-tdvp1/tdvp2/a1tdvp`` on a 1D MPS;
          ``interaction-chain-tree-tebd`` on a
          balanced binary tree, which keeps the high-bond region ``O(log N)`` edges
          deep instead of ``O(N)``) -- low entanglement, gates rebuilt each step;
          for the single-channel model the mode terms commute, which makes
          ``interaction-chain-trotter-mpo``'s exact factorization possible.
          *polaron-chain* (``polaron-chain-tebd``,
          ``polaron-chain-tdvp1/tdvp2/a1tdvp``) -- static *and* low-entanglement; needs
          ``int J/w^2`` finite (gapped or super-ohmic).  The corresponding star
          polaron representation uses ``polaron-star-tdvp1/2/a1tdvp``.  Finite
          temperature works via T-TEDOPA thermalization.
          Every one of these five representations also supports coupled
          two-site TDVP on ``state_geometry="multi-set-mps"``.  That ansatz
          stores one independently truncated bath MPS per system-basis state.
        * **multichannel** -- one bath through several couplings on shared modes.
          Selected by giving ``SystemBath(coupling=...)`` a *list* of coupling
          operators.

        For several system sites use :class:`~fishbonett.models.fishbone.Fishbone` (1D
        backbone) or :class:`~fishbonett.models.fishbone.TreeFishbone` (any tree).

        **Truncation.**  Accuracy and memory are one setting, expressed either as
        a :class:`~fishbonett.linalg.Truncation` or as the two loose keywords::

            model.run(..., trunc=Truncation(eps=1e-5, max_bond=200))
            model.run(..., trunc_eps=1e-5, bond_dim=200)     # equivalent

        ``trunc_eps`` (default ``1e-4``) is the accuracy knob. SVD-based methods
        discard singular values below that relative threshold. For ``a1tdvp``
        it is the relative convergence precision for the full-QR
        tangent-space expansion.
        ``bond_dim`` is an *optional* safety cap; the default ``None`` means
        **unlimited**, i.e. an SVD-based bond grows to whatever ``trunc_eps``
        requires (``result.max_bond`` reports what was actually used). One-site TDVP
        methods evolve on a fixed bond manifold and therefore
        require ``bond_dim``. A1TDVP can grow its manifold, but also
        requires ``bond_dim`` as a finite memory ceiling.

        ``svd_backend="auto"`` uses certified adaptive randomized truncation on
        sufficiently large matrices and exact LAPACK otherwise. ``"exact"`` and
        ``"randomized"`` select either policy explicitly; randomized requests
        retain exact safety fallbacks when the residual cannot be certified.

        ``observables`` maps a name to either a ``(d, d)`` operator on the
        system or ``(operator, BathMode(0, 0, mode))`` on a represented bath
        mode. ``result.expect[name]`` has one value per recorded time. The
        default measures ``sigma_z``/``sigma_x`` for a two-level system (and
        nothing for a larger system). ``result.rdm`` is the system reduced
        density matrix. ``observe_every`` controls recording without changing
        integration; ``bath_horizon`` controls automatic bath resolution.
        """
        dt, n_steps = resolve_time_grid(dt, t_max=t_max, n_steps=n_steps)
        trunc = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
        observe_every, bath_horizon = _resolve_sampling_options(
            observe_every, bath_horizon
        )
        bond_dim, trunc_eps = trunc.max_bond, trunc.eps
        # a general system has no canonical observables, so it gets the RDM only
        obs_ops = observables if observables is not None else self.system.observables()
        if not hasattr(obs_ops, "items"):
            raise TypeError("observables must be a mapping from names to operators")
        obs_ops = _normalize_observables(obs_ops, self.system.dim)

        axis_kw = dict(
            model=model, representation=representation,
            state_geometry=state_geometry, integrator=integrator)
        axes = any(v is not None for v in axis_kw.values())
        multichannel = self.coupled_bath.is_multichannel
        if multichannel:
            # A coupling list selects the multichannel model.
            mine = {"multichannel"}
            own = methods_of("multichannel")
            if method is not None:
                requested = method.lower().replace("_", "-")
                if requested not in own:
                    raise unknown_method_error(requested, "multichannel")
        else:
            mine = {"system-bath"}
        if method is None and not axes:
            method = own[0] if multichannel else _DEFAULT_METHOD
        # One lookup, either spelling: the object selects the model and the
        # registry says which representation/state-geometry/integrator combination and
        # engine to use.  The planner then prepares the representation, state,
        # integrator and measurement policy.
        spec = registry.resolve(
            mine, method=None if method is None else method.lower().replace("_", "-"),
            **axis_kw)
        if not set(spec.models) & mine:
            raise unknown_method_error(spec.name)
        allowed_engine_options = set()
        if spec.engine in {"mpo-tdvp", "multiset-tdvp"}:
            allowed_engine_options.update({"tol", "eshift"})
            if spec.integrator == "tdvp2":
                allowed_engine_options.add("bond_expand")
            elif spec.integrator == "a1tdvp":
                allowed_engine_options.add("bond_expand")
        unknown_options = set(engine_kw) - allowed_engine_options
        if unknown_options:
            names = ", ".join(sorted(unknown_options))
            raise TypeError(
                f"unexpected run option(s) for {spec.name}: {names}")
        if bond_dim is None and spec.requires_bond_cap:
            alternatives = _bond_growing_siblings(spec.name) or [
                "interaction-chain-tebd"]
            reason = (
                "uses a fixed one-site TDVP manifold"
                if spec.integrator == "tdvp1"
                else "grows bonds adaptively but requires a finite memory ceiling"
            )
            raise ValueError(
                f"method {spec.name!r} {reason}, so bond_dim must be given "
                "explicitly. To let trunc_eps choose an uncapped bond instead, "
                "use: " + ", ".join(alternatives)
            )
        bath_horizon = _resolve_continuation(
            resume=resume, initial=initial, method=spec.name, dt=dt,
            n_steps=n_steps, bath_horizon=bath_horizon,
            supports_resume=spec.engine == "static-tree-tebd",
        )
        ctx = RunCtx(dt=dt, n_steps=n_steps, bond_dim=bond_dim,
                     trunc_eps=trunc_eps, obs_ops=obs_ops, initial=initial,
                     krylov=krylov, seed=seed, svd_backend=svd_backend,
                     kw=engine_kw, resume=resume,
                     bath_horizon=bath_horizon, observe_every=observe_every,
                     progress=progress)
        # Local import keeps the physical model independent of every concrete
        # representation and evolution engine until a run is actually compiled.
        from fishbonett.models.simulation import compile_plan
        return compile_plan(self, spec, ctx).run()
