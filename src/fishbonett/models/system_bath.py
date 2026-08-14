"""The single-system models: one system site coupled to one bath.

Two models share this one class:

* ``system-bath`` -- one system, one bath, one coupling operator.  All implemented representations
  and both single-system geometries;
* ``multichannel`` -- several couplings on shared modes, selected automatically
  by giving ``SystemBath(coupling=...)`` a *list* of operators.

What used to be three further "models" here -- ``chain``, ``star`` and
``mode-tree`` -- were never topologies.  ``chain`` and ``star`` belong in the
complete representation name (``schrodinger-chain``, ``interaction-star``, ...),
and ``mode-tree`` is a state **geometry**.  See
:mod:`fishbonett.models.registry`, which is the authority on which combination
exists and how it is dispatched.
"""
from fishbonett.linalg import Truncation
from fishbonett.system import System
from fishbonett.bath.coupled import bind_bath
from fishbonett.models.propagate import RunCtx
from fishbonett.models import registry
from fishbonett.models.registry import (
    FIXED_BOND_METHODS, methods_of, unknown_method_error,
)

__all__ = ["SystemBath"]

#: ``run``'s default.  Used to tell "the user asked for a method" apart from
#: "the user left it alone", which matters for the multichannel model where
#: ``method`` has nothing to choose.
_DEFAULT_METHOD = "interaction-chain-tree-tdvp2"


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
        if name not in FIXED_BOND_METHODS
    }
    return sorted(siblings)


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
    Hamiltonian ``representation`` and the state's ``geometry`` are other axes of
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

    def __init__(self, h, coupling, bath):
        # `System` validates h and the coupling(s) once -- square, matching
        # dimension, Hermitian -- and keeps a multichannel coupling as a *list*
        # rather than collapsing it into a 3-D array.
        self.system = System(h, coupling)
        # mirrored as plain attributes: the validated forms, which the representation
        # builders take as `h_sys=` / `coupling=`
        self.h = self.system.h
        self.coupling = self.system.coupling
        # One authoritative binding.  ``Bath.coupling`` is accepted only as a
        # compatibility duplicate and must agree; every driver below reads this
        # object for channel topology and combined star couplings.
        self.coupled_bath = bind_bath(
            bath, self.coupling, validate_legacy=True)
        self.bath = self.coupled_bath.bath

    # -- public API ----------------------------------------------------------
    def run(self, *, dt, t_max=None, n_steps=None, method=None,
            model=None, representation=None, geometry=None, integrator=None,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial="up", krylov=25, seed=None, **engine_kw):
        """Propagate and return a :class:`Result`.

        .. rubric:: Two spellings, one lookup

        A run is four independent choices, and they can be given **as themselves**::

            sb.run(dt=..., t_max=..., representation="interaction-star", geometry="path",
                   integrator="tdvp2")

        ``model`` is what is coupled to what, and ``representation`` is one complete
        choice for how ``H`` is written: ``"schrodinger-chain"``,
        ``"schrodinger-star"``, ``"interaction-chain"``, ``"interaction-star"``,
        ``"polaron-chain"``, or ``"polaron-star"``.  ``geometry`` is the graph the
        state lives on (``"path"``/``"binary-tree"``), and
        ``integrator`` how a step is taken (``"tebd"``, ``"tdvp1"``, ``"tdvp2"``,
        ``"dtdvp"``, ``"trotter-mpo"``).  Omit an axis and it is inferred when only
        one combination fits; if several do, the error lists them.

        Representation names are deliberately exact: partial names such as
        ``representation="polaron"`` are rejected.  This keeps construction order
        explicit.  For ``interaction-chain`` the discretized star bath is put in
        the interaction picture with respect to its free Hamiltonian and the
        resulting time-dependent coupling is then transformed star-to-chain.

        The named combinations still work and are often shorter::

            sb.run(dt=..., t_max=..., method="interaction-chain-tdvp2")

        because ``"interaction-chain-tdvp2"`` fixes ``(interaction-chain, path,
        tdvp2)`` and this object supplies the ``system-bath`` model.  Every method
        name starts with its complete representation.  Give one spelling or the
        other, not both.
        ``describe_taxonomy()`` prints the table; :mod:`fishbonett.models.registry`
        is its source.

        Two models live on this class:

        * **system-bath** -- 1 system + 1 bath + 1 coupling operator, in all six
          representations.  *schrodinger-chain*
          (``schrodinger-chain-tdvp1 | schrodinger-chain-tdvp2 |
          schrodinger-chain-dtdvp``) and *schrodinger-star*
          (``schrodinger-star-tdvp1 | schrodinger-star-tdvp2``) -- static, so the
          MPO is built once and TDVP conserves energy, at the cost of the largest
          bond dimensions.  *interaction-chain*
          (``interaction-chain-tebd``, ``interaction-chain-trotter-mpo``,
          ``interaction-chain-tdvp1/2`` on a path;
          ``interaction-chain-tree-tdvp1/2 | interaction-chain-tree-tebd`` on a
          balanced binary tree, which keeps the high-bond region ``O(log N)`` edges
          deep instead of ``O(N)``) -- low entanglement, gates rebuilt each step;
          all coupling terms commute here, which is what makes
          ``interaction-chain-trotter-mpo``'s exact factorization possible.
          *polaron-chain* (``polaron-chain-tebd``,
          ``polaron-chain-tdvp1/tdvp2/dtdvp``) -- static *and* low-entanglement; needs
          ``int J/w^2`` finite (gapped or super-ohmic).  The corresponding star
          representations use ``interaction-star-tdvp1/2`` and
          ``polaron-star-tdvp1/2/dtdvp``.  Finite temperature works via T-TEDOPA
          thermalization.
        * **multichannel** -- one bath through several couplings on shared modes.
          Selected by giving ``SystemBath(coupling=...)`` a *list* of coupling
          operators, **not** by a ``method`` name.  The deprecated
          ``Bath.coupling`` field may contain the same list for compatibility, but
          emits a warning and a conflicting duplicate is rejected.

        For several system sites use :class:`~fishbonett.models.fishbone.Fishbone` (1D
        backbone) or :class:`~fishbonett.models.fishbone.TreeFishbone` (any tree).

        **Truncation.**  Accuracy and memory are one setting, expressed either as
        a :class:`~fishbonett.linalg.Truncation` or as the two loose keywords::

            model.run(..., trunc=Truncation(eps=1e-5, max_bond=200))
            model.run(..., trunc_eps=1e-5, bond_dim=200)     # equivalent

        ``trunc_eps`` (default ``1e-4``) is the accuracy knob: singular values
        below it are discarded, so it alone decides the bond dimension.
        ``bond_dim`` is an *optional* safety cap; the default ``None`` means
        **unlimited**, i.e. the bond grows to whatever ``trunc_eps`` requires
        (``result.max_bond`` reports what was actually used).  Fixed-bond methods
        (``schrodinger-chain-tdvp1``, ``interaction-chain-tdvp1``,
        ``polaron-chain-tdvp1``, ``schrodinger-chain-dtdvp``) cannot grow their
        own bonds and therefore *require* an explicit cap.

        ``observables`` maps a name to a ``(d, d)`` operator on the (single) system;
        ``result.expect[name]`` is then that expectation over time, shape
        ``(n_steps,)``.  The default measures ``sigma_z``/``sigma_x`` for a
        two-level system (and nothing for a larger system -- pass ``observables``).
        ``result.rdm`` is the system reduced density matrix per step.
        """
        stale_axes = [name for name in ("frame", "basis") if name in engine_kw]
        if stale_axes:
            joined = " and ".join(repr(name) for name in stale_axes)
            raise TypeError(
                f"{joined} is not a public run axis; use one exact "
                "representation= value instead")
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        trunc = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
        bond_dim, trunc_eps = trunc.max_bond, trunc.eps
        # a general system has no canonical observables, so it gets the RDM only
        obs_ops = observables if observables is not None else self.system.observables()

        axis_kw = dict(model=model, representation=representation, geometry=geometry,
                       integrator=integrator)
        axes = any(v is not None for v in axis_kw.values())
        multichannel = self.coupled_bath.is_multichannel
        if multichannel:
            # This model is selected by the model coupling's shape, so `method` can only
            # pick among its propagators -- say so rather than ignoring it.
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
        # registry says which representation/geometry/integrator combination and
        # engine to use.  The planner then prepares the representation, state,
        # integrator and measurement policy.
        spec = registry.resolve(
            mine, method=None if method is None else method.lower().replace("_", "-"),
            **axis_kw)
        if not set(spec.models) & mine:
            raise unknown_method_error(spec.name)
        if bond_dim is None and spec.fixed_bond:
            alternatives = _bond_growing_siblings(spec.name) or [
                "interaction-chain-tebd"]
            raise ValueError(
                f"method {spec.name!r} has a fixed bond dimension and cannot grow "
                "it from a product state, so bond_dim must be given explicitly "
                "(bond_dim=None means 'unlimited', which is only meaningful for "
                "the truncation-driven methods).  To let trunc_eps choose the bond "
                "instead, use a bond-growing method of the same representation: "
                f"{', '.join(alternatives)}")
        ctx = RunCtx(dt=dt, n_steps=n_steps, bond_dim=bond_dim,
                     trunc_eps=trunc_eps, obs_ops=obs_ops, initial=initial,
                     krylov=krylov, seed=seed, kw=engine_kw)
        # Local import keeps the physical model independent of every concrete
        # representation and evolution engine until a run is actually compiled.
        from fishbonett.models.simulation import compile_plan
        return compile_plan(self, spec, ctx).run()
