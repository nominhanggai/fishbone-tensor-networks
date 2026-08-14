"""The single-system models: one system site coupled to one bath.

Two models share this one class:

* ``system-bath`` -- one system, one bath, one coupling operator.  All four frames
  and both single-system geometries;
* ``multichannel`` -- several couplings on shared modes, selected automatically
  by giving the :class:`~fishbonett.bath.spec.Bath` a *list* of couplings.

What used to be three further "models" here -- ``chain``, ``star`` and
``mode-tree`` -- were never topologies.  ``chain``/``star`` are half of a **frame**
(the mode basis, which goes with the picture: ``schrodinger-chain``,
``interaction-star``, ...), and ``mode-tree`` is a state **geometry**.  See
:mod:`fishbonett.models.registry`, which is the authority on which combination
exists and how it is dispatched.
"""
import numpy as np
import scipy.linalg as _la

from fishbonett.linalg import Truncation
from fishbonett.system import System
from fishbonett.evolve import tdvp as _mpo
from fishbonett.evolve import tebd as _tebd
from fishbonett.frames import mpo as _frames_mpo
from fishbonett.evolve import modetree as _tree
from fishbonett.models.propagate import (RunCtx, modetree_peak_bond,
                                         mps_peak_bond, propagate)
from fishbonett.models.result import Result
from fishbonett.models import registry
from fishbonett.models.registry import (
    FIXED_BOND_METHODS, METHOD_FRAMES, methods_of, unknown_method_error,
)

__all__ = ["SystemBath"]

#: ``run``'s default.  Used to tell "the user asked for a method" apart from
#: "the user left it alone", which matters for the multichannel model where
#: ``method`` has nothing to choose.
_DEFAULT_METHOD = "tree-tdvp2"


def _bond_growing_siblings(method):
    """Methods in the same ``(frame, model)`` as ``method`` that grow their own
    bond dimension -- what to suggest when a fixed-bond method is asked for
    ``bond_dim=None``."""
    key = METHOD_FRAMES.get(method.lower().replace("_", "-"))
    if key is None:
        return []
    frame, model_key = key
    return [n for n in methods_of(model_key, frame)
            if n not in FIXED_BOND_METHODS]


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
    bath's *representation* -- its mode basis, which travels with the picture as
    part of the ``frame``, and the state's ``geometry`` -- are other axes of
    ``run``, not separate models.

    When the system has *distinct* internal degrees of freedom (e.g. a spin **and**
    a vibration), prefer to keep each on its own site with ``TreeFishbone`` (a spin
    site and a vibration site joined by an edge, with the bath on the spin) --
    putting ``spin (x) vibration`` on a single ``d = 2*d_vib`` site here works but
    defeats the MPS advantage.  Passing a multichannel :class:`Bath` (``coupling``
    a list) routes through the tree so the spin stays on its own site.

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
        # mirrored as plain attributes: the validated forms, which the frame
        # builders take as `h_sys=` / `coupling=`
        self.h = self.system.h
        self.coupling = self.system.coupling
        self.bath = bath

    # -- public API ----------------------------------------------------------
    def run(self, *, dt, t_max=None, n_steps=None, method=None,
            model=None, frame=None, geometry=None, integrator=None,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial="up", krylov=25, **engine_kw):
        """Propagate and return a :class:`Result`.

        .. rubric:: Two spellings, one lookup

        A run is four independent choices, and they can be given **as themselves**::

            sb.run(dt=..., t_max=..., frame="interaction-star", geometry="path",
                   integrator="tdvp2")

        ``model`` is what is coupled to what, ``frame`` how ``H`` is written down --
        a picture *and* a mode basis: ``"schrodinger-chain"``,
        ``"schrodinger-star"``, ``"interaction-star"``, ``"polaron-chain"`` --
        ``geometry`` the graph the state lives on (``"path"``/``"binary-tree"``), and
        ``integrator`` how a step is taken (``"tebd"``, ``"tdvp1"``, ``"tdvp2"``,
        ``"dtdvp"``, ``"trotter-mpo"``).  Omit an axis and it is inferred when only
        one combination fits; if several do, the error lists them.

        The frame carries the basis because the two are one choice, and because that
        makes the impossible pairs **unnameable**: there is no ``interaction-chain``
        (the interaction picture rotates out ``H_B``, diagonal only in the star
        basis) and no ``polaron-star`` (the displacement has to localize on ``c0``).
        A bare picture works where it is unambiguous -- ``frame="polaron"`` resolves,
        ``frame="schrodinger"`` names two frames and says so.

        The named combinations still work and are often shorter::

            sb.run(dt=..., t_max=..., method="mpo-ip-tdvp2")   # the same run

        because ``"mpo-ip-tdvp2"`` *is* ``(system-bath, interaction-star, path,
        tdvp2)``.  Give one spelling or the other, not both.
        ``describe_taxonomy()`` prints the table; :mod:`fishbonett.models.registry`
        is its source.

        Two models live on this class:

        * **system-bath** -- 1 system + 1 bath + 1 coupling operator, in all four
          frames.  *schrodinger-chain* (``mpo-tdvp1 | mpo-tdvp2 | mpo-dtdvp``) and
          *schrodinger-star* (``mpo-star-tdvp1 | mpo-star-tdvp2``) -- static, so the
          MPO is built once and TDVP conserves energy, at the cost of the largest
          bond dimensions.  *interaction-star* (``tebd``, ``trotter-mpo``,
          ``mpo-ip-tdvp1/2`` on a path; ``tree-tdvp | tree-tdvp2 | tree-tebd`` on a
          balanced binary tree, which keeps the high-bond region ``O(log N)`` edges
          deep instead of ``O(N)``) -- low entanglement, gates rebuilt each step;
          all coupling terms commute here, which is what makes ``trotter-mpo``'s
          exact factorization possible.  *polaron-chain* (``polaron``,
          ``polaron-tdvp1/tdvp2/dtdvp``) -- static *and* low-entanglement; needs
          ``int J/w^2`` finite (gapped or super-ohmic).  Finite temperature works
          via T-TEDOPA thermalization.
        * **multichannel** -- one bath through several couplings on shared modes.
          Selected by giving :class:`~fishbonett.bath.spec.Bath` a *list* of
          coupling operators, **not** by a ``method`` name; ``method`` is then
          ignored.

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
        (``mpo-tdvp1``, ``mpo-ip-tdvp1``, ``tree-tdvp``, ``polaron-tdvp1``,
        ``mpo-dtdvp``) cannot grow their own bonds and therefore *require* an
        explicit cap.

        ``observables`` maps a name to a ``(d, d)`` operator on the (single) system;
        ``result.expect[name]`` is then that expectation over time, shape
        ``(n_steps,)``.  The default measures ``sigma_z``/``sigma_x`` for a
        two-level system (and nothing for a larger system -- pass ``observables``).
        ``result.rdm`` is the system reduced density matrix per step.
        """
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        trunc = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
        bond_dim, trunc_eps = trunc.max_bond, trunc.eps
        # a general system has no canonical observables, so it gets the RDM only
        obs_ops = observables if observables is not None else self.system.observables()

        axis_kw = dict(model=model, frame=frame, geometry=geometry,
                       integrator=integrator)
        axes = any(v is not None for v in axis_kw.values())
        multichannel = getattr(self.bath, "is_multichannel", False)
        if multichannel:
            # This model is selected by the *bath's shape*, so `method` can only
            # pick among its propagators -- say so rather than ignoring it.
            mine = {"multichannel"}
            own = methods_of("multichannel")
            if method is not None and method.lower().replace("_", "-") not in own:
                raise ValueError(
                    f"this Bath is multichannel (a list of couplings), which "
                    f"selects the 'multichannel' model; it does not have "
                    f"method={method!r}.  Its propagators are {', '.join(own)} "
                    f"(or drop the method argument for the default).  To use the "
                    f"single-channel models, pass a single coupling operator.")
        else:
            mine = {"system-bath"}
        if method is None and not axes:
            method = own[0] if multichannel else _DEFAULT_METHOD
        # one lookup, either spelling: the registry says which combination of the
        # four axes this is and which engine realizes it; the driver table says
        # where that engine lives on this class.
        spec = registry.resolve(
            mine, method=None if method is None else method.lower().replace("_", "-"),
            **axis_kw)
        if not set(spec.models) & mine:
            raise unknown_method_error(spec.name)
        if bond_dim is None and spec.fixed_bond:
            alternatives = _bond_growing_siblings(spec.name) or ["tebd"]
            raise ValueError(
                f"method {spec.name!r} has a fixed bond dimension and cannot grow "
                "it from a product state, so bond_dim must be given explicitly "
                "(bond_dim=None means 'unlimited', which is only meaningful for "
                "the truncation-driven methods).  To let trunc_eps choose the bond "
                "instead, use a bond-growing method of the same frame: "
                f"{', '.join(alternatives)}")
        ctx = RunCtx(dt=dt, n_steps=n_steps, bond_dim=bond_dim,
                     trunc_eps=trunc_eps, obs_ops=obs_ops, initial=initial,
                     krylov=krylov, kw=engine_kw)
        return getattr(self, self._DRIVERS[spec.engine])(spec, ctx)

    #: ``registry.Method.engine`` -> the driver on this class that realizes it.
    #: Keyed on the engine, not on the ``integrator`` axis: ``tdvp2`` is one
    #: integrator but two engines reach it (the MPO sweep on a path, the mode-tree
    #: sweep on a binary tree), and conversely one engine serves several
    #: integrators.
    _DRIVERS = {
        "mpo-tdvp": "_run_mpo",
        "modetree": "_run_tree",
        "swap-tebd": "_run_swap_tebd",
        "displacement-mpo": "_run_trotter_mpo",
        "polaron-tebd": "_run_polaron",
        "static-tree-tebd": "_run_multichannel",
    }

    # -- dispatchers ---------------------------------------------------------
    def _expect_from_rdm(self, rdms, obs_ops):
        rdms = np.asarray(rdms)
        return {name: np.einsum("tij,ji->t", rdms, np.asarray(O)).real
                for name, O in obs_ops.items()}

    def _check_system(self):
        """Reject a multichannel coupling on a single-channel engine.

        Shape and Hermiticity are already guaranteed by
        :class:`fishbonett.system.System` at construction; what remains is that the
        single-channel engines take one operator, not a list.
        """
        if self.system.is_multichannel:
            raise ValueError(
                "this method takes a single coupling operator, but the system was "
                "given a list of them.  A multichannel bath is selected by the "
                "Bath's shape and has its own propagators; see "
                "fishbonett.models.registry.")

    #: which MPO each frame implies, and how it needs to be driven.  This is the
    #: whole point of the taxonomy: the frame decides what H looks like, the
    #: integrator (``spec.driver``) is an independent choice, and the geometry a
    #: third.  Each returns ``(MPOFrame, hooks)`` -- the hooks being whatever that
    #: frame does to the state on the way in and to the observable on the way out.
    #:
    #: A flat dict keyed on the frame, because a frame is a picture *and* a basis.
    #: It used to be keyed on ``(frame, model)`` with ``chain``/``star`` in the
    #: model slot, which is how the taxonomy came to call a basis a model.  The
    #: builders in :mod:`fishbonett.frames.mpo` were already named this way.
    _MPO_FRAMES = {
        "schrodinger-chain": "_chain_mpo_frame",
        "schrodinger-star": "_static_star_mpo_frame",
        "interaction-chain": "_ip_chain_mpo_frame",
        "interaction-star": "_ip_star_mpo_frame",
        "polaron-chain": "_polaron_mpo_frame",
    }

    def _undressed_mpo_frame(self, maker, b, ctx):
        """A frame that does nothing to the state: read the RDM straight off it."""
        frame = maker(b.spectral_density(), b.domain, n_chain=b.n_modes,
                      d=b.phys_dim, hsys=self.h, cop=self.coupling,
                      init=self._initial_state(ctx.initial),
                      discretizer=b.discretizer())
        return frame, dict(observe=lambda A: _mpo.measure_rdm(A[0]),
                           prec=ctx.kw.get("prec", 1e-4),
                           tol=ctx.kw.get("tol", 1e-7),
                           eshift=ctx.kw.get("eshift", False))

    def _chain_mpo_frame(self, b, ctx):
        return self._undressed_mpo_frame(_frames_mpo.chain_mpo_frame, b, ctx)

    def _static_star_mpo_frame(self, b, ctx):
        return self._undressed_mpo_frame(_frames_mpo.static_star_mpo_frame, b, ctx)

    def _ip_chain_mpo_frame(self, b, ctx):
        return self._undressed_mpo_frame(_frames_mpo.ip_chain_mpo_frame, b, ctx)

    def _ip_star_mpo_frame(self, b, ctx):
        return self._undressed_mpo_frame(_frames_mpo.ip_star_mpo_frame, b, ctx)

    def _polaron_mpo_frame(self, b, ctx):
        """The polaron ``H~`` as an MPO, plus the dressing that makes it a frame.

        ``H~`` is time-independent, so it has a plain MPO and the ordinary sweeps
        apply.  What is polaron-specific is not the loop but the two hooks: the bath
        vacuum is a *displaced* coherent state on ``c0``, and the RDM read off that
        pair has to be un-dressed to get back to the lab frame.

        Two settings stay per-frame because they are genuine differences: ``prec``
        defaults to ``trunc_eps`` rather than ``1e-4``, and the two-site sweep must
        **not** be re-canonicalized -- that would re-gauge the displaced block this
        frame just installed.
        """
        builder, _b, _n, _pd = self._polaron_builder(ctx)
        M = builder.mpo()

        def prepare(A):
            A[0], A[1] = builder.initial_mps_pair(self._initial_state(ctx.initial))
            return A

        frame = _frames_mpo.MPOFrame(
            n_sites=len(M), phys_dim=b.phys_dim,
            system=(self.h, self.coupling, np.zeros(self.h.shape[0], complex)),
            mpo=lambda t=None: M, static=True)
        return frame, dict(prepare=prepare, canonicalize=False,
                           observe=lambda A: builder.undress_rdm_tdvp(A[0], A[1]),
                           prec=ctx.kw.get("prec", ctx.trunc_eps))

    def _run_mpo(self, spec, ctx):
        """TDVP on an MPO -- one loop for all seven (frame, sweep) combinations.

        Each of these used to be a whole-run wrapper in
        :mod:`fishbonett.evolve.tdvp` that built its own Hamiltonian and ran its own
        loop.  The Hamiltonian is a frame question, so it comes from
        :mod:`fishbonett.frames.mpo` now; what is left is the loop, which never
        differed.
        """
        self._check_system()
        b = self.bath.resolved(ctx.t_max)
        make = getattr(self, self._MPO_FRAMES[spec.frame])
        frame, hooks = make(b, ctx)
        t, rdms, maxb = _mpo.run_mpo_frame(
            frame, dt=ctx.dt, nsteps=ctx.n_steps, sweep=spec.driver,
            D=ctx.bond_dim, chi_max=ctx.bond_dim, eps=ctx.trunc_eps,
            krylov=ctx.krylov, Dplusmax=ctx.kw.get("Dplusmax", 4), **hooks)
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=maxb, rdm=np.asarray(rdms), method=spec.name)

    def _run_tree(self, spec, ctx):
        self._check_system()
        b = self.bath.resolved(ctx.t_max)
        # The mode-tree engine still owns its loop (its MPO is per-node with an
        # environment per edge, so it is not the chain driver of _run_mpo), but it
        # calls `observe` once per step -- enough to collect the peak bond it never
        # reported, so these methods stop being the only ones with max_bond=None.
        max_bond = []

        def observe(nodes, root):
            max_bond.append(modetree_peak_bond(nodes))
            return _tree.measure_rdm_oc(nodes, root)

        common = dict(hsys=self.h, cop=self.coupling,
                      init=self._initial_state(ctx.initial),
                      n_chain=b.n_modes, phys_dim=b.phys_dim, dt=ctx.dt,
                      nsteps=ctx.n_steps, D=ctx.bond_dim,
                      discretizer=b.discretizer(),
                      observe=observe, **ctx.kw)
        sd, dom = b.spectral_density(), b.domain
        # `krylov` is not passed: the tree sweeps never took one.  Both wrappers
        # accepted it and neither forwarded it to tdvp_sweep / tdvp2_sweep, so it
        # was threaded through the hot path and dropped -- the same dead parameter
        # as `mode` on get_u.
        if spec.driver == "run_tree_tdvp":
            t, rdms = _tree.run_tree_mpo(sd, dom, sweep="tdvp1", **common)
        elif spec.driver == "run_tree_tdvp2":
            t, rdms = _tree.run_tree_mpo(sd, dom, sweep="tdvp2",
                                         trunc_eps=ctx.trunc_eps, **common)
        else:
            t, rdms = _tree.run_tree_tebd(sd, dom, trunc_eps=ctx.trunc_eps,
                                          **common)
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method=spec.name)

    def _run_multichannel(self, spec, ctx):
        """One bath coupled to the system through several operators: a shared-mode
        star attached to the (single) system site.  Built on the tree engine so the
        system stays on its own site."""
        from fishbonett.models.fishbone import TreeFishbone
        fb = TreeFishbone(sites=[self.h], edges=[], baths=[self.bath])
        r = fb.run(dt=ctx.dt, n_steps=ctx.n_steps, bond_dim=ctx.bond_dim,
                   trunc_eps=ctx.trunc_eps, observables=ctx.obs_ops,
                   initial=[self._initial_state(ctx.initial)])
        # collapse the single-site axis: this is a single-system model, so the
        # Result should have the single-system shape (see models/result.py)
        expect = {name: r.expect[name][:, 0] for name in r.expect}
        rdm = np.array([r.rdm[k, 0] for k in range(ctx.n_steps)])
        return Result(t=r.t, expect=expect, rdm=rdm, max_bond=r.max_bond,
                      method=spec.name)

    def _set_system_site(self, state, initial):
        """Put the initial system state on site 0 of a product-state MPS.

        The bath sites are already vacuum, so this only writes the system tensor.
        Both swap-network drivers used to inline the same three lines.
        """
        psi0 = self._initial_state(initial)
        state.B[0][:] = 0.0
        for a in range(len(psi0)):
            state.B[0][0, a, 0] = psi0[a]
        return state

    def _initial_state(self, initial):
        """Initial system state as a ``d_sys``-vector.

        Thin wrapper over :meth:`fishbonett.system.System.initial_vector`, which is
        where ``"up"``/``"down"``/``"ground"`` and explicit vectors are interpreted.
        """
        return self.system.initial_vector(initial)

    #: The ``application="swap"`` frames: which builder supplies ``H(t)`` for a
    #: given *(frame, model)*.  Everything downstream of the builder is the
    #: application's business, not the frame's, which is why there is one driver
    #: below and not one per frame.
    _SWAP_FRAMES = {
        ("interaction-chain", "system-bath"): "_ip_frame",
        ("interaction-star", "multichannel"): "_multichannel_swap_frame",
    }

    def _ip_frame(self, b, ctx):
        """One system, one bath, scalar coupling: the plain interaction picture.

        Built here once and used by **both** of its integrators -- the swap-network
        TEBD and the conditional-displacement MPO.  That is the frame/integrator
        split working: same ``H(t)``, two ways of applying it.
        """
        from fishbonett.frames.interaction_picture import SystemBathIP
        pd = [self.h.shape[0]] + [b.phys_dim] * b.n_modes
        return SystemBathIP(pd, h_sys=self.h, coupling=self.coupling,
                            sd=b.spectral_density(), domain=b.domain,
                            ncap=ctx.kw.get("ncap", 20000),
                            discretizer=b.discretizer()).build(), pd

    def _multichannel_swap_frame(self, b, ctx):
        """The shared-mode star in the **interaction picture**: the free-bath
        evolution is rotated out, so the modes carry no on-site frequency and the
        matrix-valued coupling becomes time-dependent.

        Cross-checks the static (Schrodinger) multichannel path: same physics,
        different frame, so the two must agree -- which is only meaningful because
        both take their star from the same
        :meth:`~fishbonett.bath.spec.Bath.shared_mode_star`.  Temperature comes in
        the same T-TEDOPA way as every other method (the thermalized density on a
        signed domain) rather than through the explicit thermofield doubling of
        :meth:`~fishbonett.frames.multichannel.SystemBathMultiChannel.__init__`,
        which uses a different unit convention for ``temp``.
        """
        from fishbonett.frames.multichannel import SystemBathMultiChannel
        freq, coup_mat = b.shared_mode_star()
        pd = [self.h.shape[0]] + [b.phys_dim] * b.n_modes
        return SystemBathMultiChannel.from_signed_star(
            pd, coup_mat, freq, h_sys=self.h).build(n=0), pd

    def _run_swap_tebd(self, spec, ctx):
        """TEBD under the ``swap`` application -- one driver for every frame that
        needs it.

        The interaction picture couples *every* mode to the system, so its
        interaction graph is a **star** while the state is a **path**.  The swap
        network is what reconciles the two: each step walks the system site out past
        every mode and back, applying its gate on the way.  That is a property of
        the *application*, not of the frame -- so the frame only has to supply
        ``H(t)``, and the two methods that need this (``tebd`` and
        ``multichannel-ip``, the two whose ``Method.application`` derives to
        ``"swap"``) share everything after the builder.

        One symmetric (Strang) step per iteration, so each advances the user's
        physical ``dt`` -- matching the tree and MPO drivers.
        """
        from fishbonett.states.mps import SystemBathMPS
        b = self.bath.resolved(ctx.t_max)
        make = getattr(self, self._SWAP_FRAMES[(spec.frame, spec.models[0])])
        builder, pd = make(b, ctx)
        state = SystemBathMPS(pd)               # the MPS being evolved
        self._set_system_site(state, ctx.initial)
        return propagate(
            spec, ctx,
            step=lambda k: _tebd.symmetric_swap_step(
                state, builder, k * ctx.dt, ctx.dt, b.n_modes, ctx.bond_dim,
                ctx.trunc_eps),
            rdm=lambda: state.rdm(0),          # inherited from TensorNetwork
            peak_bond=lambda: mps_peak_bond(state),
            expect_from_rdm=self._expect_from_rdm)

    def _run_trotter_mpo(self, spec, ctx):
        """Interaction picture propagated by the exact conditional-displacement MPO.

        Same frame and same physics as ``method="tebd"``, but the whole system-bath
        propagator is applied as one low-bond MPO instead of being Trotterized into
        two-site gates and shuttled with a swap network: no swaps, no ``d x d``
        bosonic gates, and the multimode factorization is *exact* (see
        :meth:`~fishbonett.frames.interaction_picture.SystemBathIP.displacement_mpo`).
        The system term is Strang-split around it, so the step is second order.

        It shares :meth:`_ip_frame` with ``tebd``: identical ``H(t)``, a different
        way of applying it.  Nothing about the frame knows which."""
        from fishbonett.evolve.mpo_apply import (apply_mpo, compress, bond_dims,
                                                 product_state)
        self._check_system()
        b = self.bath.resolved(ctx.t_max)
        builder, pd = self._ip_frame(b, ctx)

        # sites are [system, mode_0, ..., mode_{n-1}] for the MPO
        A = product_state(pd, self._initial_state(ctx.initial))
        u_half = _la.expm(-0.5j * ctx.dt * np.asarray(self.h, complex))

        def step(k):
            # Strang: half a system step, the exact displacement MPO, half again
            nonlocal A
            A[0] = np.einsum('ij,ajb->aib', u_half, A[0])
            A = compress(
                apply_mpo(A, builder.displacement_mpo(k * ctx.dt, ctx.dt)),
                ctx.bond_dim, ctx.trunc_eps)
            A[0] = np.einsum('ij,ajb->aib', u_half, A[0])

        def rdm():
            rho = np.einsum('lsr,ltr->st', A[0], A[0].conj())
            return rho / np.trace(rho).real

        return propagate(spec, ctx, step=step, rdm=rdm,
                         peak_bond=lambda: max(bond_dims(A)),
                         expect_from_rdm=self._expect_from_rdm)

    def _polaron_builder(self, ctx):
        """Shared polaron setup: validate, resolve the bath and build the frame.
        Returns ``(builder, resolved_bath, n_modes, pd)``."""
        from fishbonett.frames.polaron import SystemBathPolaron
        self._check_system()
        b = self.bath.resolved(ctx.t_max)
        n, d_sys = b.n_modes, self.h.shape[0]
        pd = [d_sys] + [b.phys_dim] * n
        builder = SystemBathPolaron(pd, h_sys=self.h, coupling=self.coupling,
                                    sd=b.spectral_density(), domain=b.domain,
                                    discretizer=b.discretizer()).build()
        return builder, b, n, pd

    def _run_polaron(self, spec, ctx):
        """Polaron-frame chain: the static system-bath coupling is absorbed into a
        displacement of the first (reweighted-``J/w^2``) chain mode ``c0``, leaving
        a free chain plus a dressed ``(c0, system)`` gate.  Plain nearest-neighbour
        Trotter (no swap network); the physical bath vacuum is a displaced coherent
        state on ``c0``; lab-frame observables are recovered by un-dressing.
        See :mod:`fishbonett.frames.polaron`."""
        from fishbonett.states.mps import SystemBathMPS
        builder, b, n, pd = self._polaron_builder(ctx)
        state = SystemBathMPS(pd)               # boson sites default to vacuum
        psi0 = self._initial_state(ctx.initial)
        # displaced (system, c0) initial block at bond 0; other boson sites stay vacuum
        state.split_truncate_theta(builder.initial_theta(psi0), 0, ctx.bond_dim,
                                   1e-14)

        # static frame, so the gates are built once; symmetric Strang per step
        gates = builder.gates(ctx.dt / 2.0)
        return propagate(
            spec, ctx,
            step=lambda _k: _tebd.symmetric_static_step(
                state, gates, n, ctx.bond_dim, ctx.trunc_eps),
            # the frame dressed the state, so the frame undresses the observable
            rdm=lambda: builder.undress_rdm(state.get_theta2(0)),
            peak_bond=lambda: mps_peak_bond(state),
            expect_from_rdm=self._expect_from_rdm)

