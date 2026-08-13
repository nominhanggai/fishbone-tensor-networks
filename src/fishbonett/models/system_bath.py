"""The single-system models: one system site coupled to one bath.

Four models share this one class, distinguished by how the bath is represented
and selected through ``run(method=...)``:

* ``chain`` -- modes chain-mapped into 1D (all three frames);
* ``star`` -- no chain mapping (Schrodinger and interaction pictures);
* ``mode-tree`` -- modes on a balanced binary tree (interaction picture);
* ``multichannel`` -- several couplings on shared modes, selected automatically
  by giving the :class:`~fishbonett.bath.spec.Bath` a *list* of couplings
  (Schrodinger and interaction pictures).

:mod:`fishbonett.models.registry` is the authority on which method belongs to
which model and frame.
"""
import numpy as np
import scipy.linalg as _la

from fishbonett.linalg import Truncation
from fishbonett.system import System
from fishbonett.evolve import tdvp as _mpo
from fishbonett.evolve import tebd as _tebd
from fishbonett.frames import mpo as _frames_mpo
from fishbonett.evolve import modetree as _tree
from fishbonett.models.propagate import RunCtx
from fishbonett.models.result import Result
from fishbonett.models.registry import (
    FIXED_BOND_METHODS, METHOD_FRAMES, METHODS, methods_of,
    unknown_method_error,
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

    This one class covers four models -- ``chain``, ``star``, ``mode-tree`` and
    ``multichannel`` -- because they differ only in how the *bath* is represented,
    which ``run(method=...)`` selects.

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
    def run(self, *, dt, t_max=None, n_steps=None, method=_DEFAULT_METHOD,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial="up", krylov=25, **engine_kw):
        """Propagate and return a :class:`Result`.

        ``method`` picks a **model** and a **frame** at once -- the four
        single-system models live on this class, and each admits its own frames
        (see :mod:`fishbonett.models.registry`; ``describe_taxonomy()`` prints the
        table):

        * **chain** -- 1 system + 1 bath, modes chain-mapped to 1D.  The only
          model with all three frames.  *Schroedinger*: ``mpo-tdvp1 | mpo-tdvp2 |
          mpo-dtdvp`` -- static, so the MPO is built once and TDVP conserves
          energy, at the cost of the largest bond dimensions.  *interaction*:
          ``tebd``, ``trotter-mpo`` -- low entanglement, gates rebuilt each step;
          all coupling terms commute here, which is what makes ``trotter-mpo``'s
          exact factorization possible.  *polaron*: ``polaron``,
          ``polaron-tdvp1/tdvp2/dtdvp`` -- static *and* low-entanglement; needs
          ``int J/w^2`` finite (gapped or super-ohmic).  Finite temperature works
          via T-TEDOPA thermalization.
        * **star** -- no chain mapping; every mode couples straight to the system,
          so there are no mode-mode terms but no locality either.
          *interaction*: ``mpo-ip-tdvp1 | mpo-ip-tdvp2``.
        * **mode-tree** -- the same chain-mapped modes placed on a balanced binary
          tree, keeping the high-bond region ``O(log N)`` edges deep instead of
          ``O(N)``.  *interaction*: ``tree-tdvp | tree-tdvp2 | tree-tebd``.
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
        if bond_dim is None and method.lower().replace("_", "-") in FIXED_BOND_METHODS:
            alternatives = _bond_growing_siblings(method) or ["tebd"]
            raise ValueError(
                f"method {method!r} has a fixed bond dimension and cannot grow it "
                "from a product state, so bond_dim must be given explicitly "
                "(bond_dim=None means 'unlimited', which is only meaningful for "
                "the truncation-driven methods).  To let trunc_eps choose the bond "
                "instead, use a bond-growing method of the same frame: "
                f"{', '.join(alternatives)}")
        # a general system has no canonical observables, so it gets the RDM only
        obs_ops = observables if observables is not None else self.system.observables()
        m = method.lower().replace("_", "-")
        if getattr(self.bath, "is_multichannel", False):
            # The multichannel model is selected by the bath's shape, so `method`
            # can only pick among *its* propagators -- say so rather than ignoring it.
            own = methods_of("multichannel")
            if m not in own and method != _DEFAULT_METHOD:
                raise ValueError(
                    f"this Bath is multichannel (a list of couplings), which "
                    f"selects the 'multichannel' model; it does not have "
                    f"method={method!r}.  Its propagators are {', '.join(own)} "
                    f"(or drop the method argument for the default).  To use the "
                    f"single-channel models, pass a single coupling operator.")
            m = m if m in own else own[0]        # no method given -> its default
            mine = {"multichannel"}
        else:
            mine = {"chain", "star", "mode-tree"}
        # one lookup instead of a chain of `if`s: the registry says which driver
        # realizes this (model, frame, integrator), and the driver table says
        # where that driver lives on this class.
        spec = METHODS.get(m)
        if spec is None or not set(spec.models) & mine:
            raise unknown_method_error(m)
        ctx = RunCtx(dt=dt, n_steps=n_steps, bond_dim=bond_dim,
                     trunc_eps=trunc_eps, obs_ops=obs_ops, initial=initial,
                     krylov=krylov, kw=engine_kw)
        return getattr(self, self._DRIVERS[spec.integrator])(spec, ctx)

    #: ``registry.Method.integrator`` -> the driver on this class that realizes it.
    _DRIVERS = {
        "chain-mpo-tdvp": "_run_mpo",
        "modetree": "_run_tree",
        "swap-tebd": "_run_tebd",
        "displacement-mpo": "_run_trotter_mpo",
        "polaron-tebd": "_run_polaron",
        "polaron-tdvp": "_run_polaron_tdvp",
        "static-tree-tebd": "_run_multichannel",
        "multichannel-swap-tebd": "_run_multichannel_ip",
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

    #: which MPO the *(frame, model)* pair implies.  This is the whole point of the
    #: taxonomy: the frame decides what H looks like, the model decides on what
    #: geometry, and the integrator (``spec.driver``) is an independent choice.
    _MPO_FRAMES = {
        ("schrodinger", "chain"): "chain_mpo_frame",
        ("schrodinger", "star"): "static_star_mpo_frame",
        ("interaction", "star"): "ip_star_mpo_frame",
    }

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
        make = getattr(_frames_mpo, self._MPO_FRAMES[(spec.frame, spec.models[0])])
        frame = make(b.spectral_density(), b.domain, n_chain=b.n_modes,
                     d=b.phys_dim, hsys=self.h, cop=self.coupling,
                     init=self._initial_state(ctx.initial),
                     discretizer=b.discretizer())
        t, rdms, maxb = _mpo.run_mpo_frame(
            frame, dt=ctx.dt, nsteps=ctx.n_steps, sweep=spec.driver,
            D=ctx.bond_dim, chi_max=ctx.bond_dim, eps=ctx.trunc_eps,
            krylov=ctx.krylov, observe=_mpo.measure_rdm,
            prec=ctx.kw.get("prec", 1e-4), Dplusmax=ctx.kw.get("Dplusmax", 4),
            tol=ctx.kw.get("tol", 1e-7), eshift=ctx.kw.get("eshift", False))
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=maxb, rdm=np.asarray(rdms), method=spec.name)

    def _run_tree(self, spec, ctx):
        self._check_system()
        b = self.bath.resolved(ctx.t_max)
        common = dict(hsys=self.h, cop=self.coupling,
                      init=self._initial_state(ctx.initial),
                      n_chain=b.n_modes, phys_dim=b.phys_dim, dt=ctx.dt,
                      nsteps=ctx.n_steps, D=ctx.bond_dim,
                      discretizer=b.discretizer(),
                      observe=_tree.measure_rdm_oc, **ctx.kw)
        sd, dom = b.spectral_density(), b.domain
        if spec.driver == "run_tree_tdvp":
            t, rdms = _tree.run_tree_tdvp(sd, dom, krylov=ctx.krylov, **common)
        elif spec.driver == "run_tree_tdvp2":
            t, rdms = _tree.run_tree_tdvp2(sd, dom, trunc_eps=ctx.trunc_eps,
                                           krylov=ctx.krylov, **common)
        else:
            t, rdms = _tree.run_tree_tebd(sd, dom, trunc_eps=ctx.trunc_eps,
                                          **common)
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      rdm=np.asarray(rdms), method=spec.name)

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

    def _run_multichannel_ip(self, spec, ctx):
        """The same shared-mode star in the **interaction picture**: the free-bath
        evolution is rotated out, so the modes carry no on-site frequency and the
        matrix-valued coupling becomes time-dependent.

        Cross-checks the static (Schrodinger) path above: same physics, different
        frame, so the two must agree.  Temperature comes in the same T-TEDOPA way as
        every other method -- the thermalized density on a signed domain -- rather
        than through the explicit thermofield doubling of
        :meth:`~fishbonett.frames.multichannel.SystemBathMultiChannel.__init__`,
        which uses a different unit convention for ``temp``."""
        from fishbonett.frames.multichannel import SystemBathMultiChannel
        from fishbonett.states.mps import SystemBathMPS
        b = self.bath.resolved(ctx.t_max)
        # the same shared-mode star the Schroedinger frame builds, from the same
        # Bath method -- the two paths must discretize identically to cross-check
        freq, coup_mat = b.shared_mode_star()
        d_sys = self.h.shape[0]
        pd = [d_sys] + [b.phys_dim] * b.n_modes
        builder = SystemBathMultiChannel.from_signed_star(
            pd, coup_mat, freq, h_sys=self.h).build(n=0)

        state = SystemBathMPS(pd)
        psi0 = self._initial_state(ctx.initial)
        state.B[0][:] = 0.0
        for a in range(d_sys):
            state.B[0][0, a, 0] = psi0[a]

        rdms, max_bond = [], []
        for step in range(ctx.n_steps):
            _tebd.symmetric_swap_step(state, builder, step * ctx.dt, ctx.dt,
                                      b.n_modes, ctx.bond_dim, ctx.trunc_eps)
            rdms.append(state.rdm(0))     # inherited from TensorNetwork
            max_bond.append(max((len(s) for s in state.S), default=1))
        t = np.arange(1, ctx.n_steps + 1) * ctx.dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method=spec.name)

    def _initial_state(self, initial):
        """Initial system state as a ``d_sys``-vector.

        Thin wrapper over :meth:`fishbonett.system.System.initial_vector`, which is
        where ``"up"``/``"down"``/``"ground"`` and explicit vectors are interpreted.
        """
        return self.system.initial_vector(initial)

    def _run_tebd(self, spec, ctx):
        from fishbonett.frames.interaction_picture import SystemBathIP as _IPBuilder
        from fishbonett.states.mps import SystemBathMPS
        b = self.bath.resolved(ctx.t_max)
        n = b.n_modes
        d_sys = self.h.shape[0]
        pd = [d_sys] + [b.phys_dim] * n
        builder = _IPBuilder(pd, h_sys=self.h, coupling=self.coupling,
                             sd=b.spectral_density(), domain=b.domain,
                             ncap=ctx.kw.get("ncap", 20000),
                             discretizer=b.discretizer()).build()

        state = SystemBathMPS(pd)               # the MPS being evolved
        psi0 = self._initial_state(ctx.initial)
        state.B[0][:] = 0.0
        for a in range(d_sys):
            state.B[0][0, a, 0] = psi0[a]

        # One symmetric (Strang) swap-network step per iteration, so each advances
        # the user's physical dt -- matching the tree/mpo drivers.
        rdms, max_bond = [], []
        for step in range(ctx.n_steps):
            _tebd.symmetric_swap_step(state, builder, step * ctx.dt, ctx.dt, n,
                                      ctx.bond_dim, ctx.trunc_eps)
            rdms.append(state.rdm(0))     # inherited from TensorNetwork
            max_bond.append(max((len(s) for s in state.S), default=1))
        t = np.arange(1, ctx.n_steps + 1) * ctx.dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method=spec.name)

    def _run_trotter_mpo(self, spec, ctx):
        """Interaction picture propagated by the exact conditional-displacement MPO.

        Same frame and same physics as ``method="tebd"``, but the whole system-bath
        propagator is applied as one low-bond MPO instead of being Trotterized into
        two-site gates and shuttled with a swap network: no swaps, no ``d x d``
        bosonic gates, and the multimode factorization is *exact* (see
        :meth:`~fishbonett.frames.interaction_picture.SystemBathIP.displacement_mpo`).
        The system term is Strang-split around it, so the step is second order."""
        from fishbonett.frames.interaction_picture import SystemBathIP
        from fishbonett.evolve.mpo_apply import (apply_mpo, compress, bond_dims,
                                                 product_state)
        self._check_system()
        b = self.bath.resolved(ctx.t_max)
        n, d_sys = b.n_modes, self.h.shape[0]
        builder = SystemBathIP([d_sys] + [b.phys_dim] * n, h_sys=self.h,
                               coupling=self.coupling, sd=b.spectral_density(),
                               domain=b.domain, ncap=ctx.kw.get("ncap", 20000),
                               discretizer=b.discretizer()).build()

        # sites are [system, mode_0, ..., mode_{n-1}] for the MPO
        A = product_state([d_sys] + [b.phys_dim] * n,
                          self._initial_state(ctx.initial))
        u_half = _la.expm(-0.5j * ctx.dt * np.asarray(self.h, complex))
        rdms, max_bond = [], []
        for step in range(ctx.n_steps):
            A[0] = np.einsum('ij,ajb->aib', u_half, A[0])        # half system step
            A = compress(
                apply_mpo(A, builder.displacement_mpo(step * ctx.dt, ctx.dt)),
                ctx.bond_dim, ctx.trunc_eps)
            A[0] = np.einsum('ij,ajb->aib', u_half, A[0])
            rho = np.einsum('lsr,ltr->st', A[0], A[0].conj())
            rdms.append(rho / np.trace(rho).real)
            max_bond.append(max(bond_dims(A)))
        t = np.arange(1, ctx.n_steps + 1) * ctx.dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method=spec.name)

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

    def _run_polaron_tdvp(self, spec, ctx):
        """Polaron frame propagated with TDVP.  Because ``H~`` is time-independent
        it has a plain MPO (:meth:`~fishbonett.frames.polaron.SystemBathPolaron.mpo`),
        so the 1-site / 2-site / bond-adaptive sweeps all apply.  1-site TDVP never
        forms a two-site block, which avoids the ``O(d^4)`` boson-boson gates of the
        polaron TEBD sweep."""
        from fishbonett.evolve.tdvp import (init_mps, tdvp1sweep, tdvp2sweep,
                                            tdvp1sweep_dynamic, bonddims,
                                            _pad_bonds, right_canonicalize)
        builder, b, n, pd = self._polaron_builder(ctx)
        variant = spec.driver
        M = builder.mpo()
        A = init_mps(len(M), b.phys_dim, np.zeros(self.h.shape[0], complex))
        A[0], A[1] = builder.initial_mps_pair(self._initial_state(ctx.initial))
        if variant == "tdvp1":
            # 1-site TDVP conserves the bond dimension, so it cannot grow out of a
            # product state: pad to the requested bond first (as run_tdvp1 does).
            A = right_canonicalize(_pad_bonds(A, ctx.bond_dim))
        env = Afull = FRs = None
        rdms, max_bond = [], []
        for _ in range(ctx.n_steps):
            if variant == "tdvp1":
                A, env = tdvp1sweep(ctx.dt, A, M, env, m=ctx.krylov)
            elif variant == "tdvp2":
                A, env = tdvp2sweep(ctx.dt, A, M, ctx.bond_dim, ctx.trunc_eps,
                                    env, m=ctx.krylov)
            else:
                A, Afull, FRs, _ = tdvp1sweep_dynamic(
                    ctx.dt, A, M, Afull, FRs,
                    prec=ctx.kw.get("prec", ctx.trunc_eps),
                    Dlim=ctx.bond_dim, Dplusmax=ctx.kw.get("Dplusmax", 4),
                    m=ctx.krylov)
            rdms.append(builder.undress_rdm_tdvp(A[0], A[1]))   # lab frame
            max_bond.append(max(bonddims(A)))
        t = np.arange(1, ctx.n_steps + 1) * ctx.dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method=spec.name)

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

        # static; symmetric Strang per step
        gates = builder.gates(ctx.dt / 2.0)
        rdms, max_bond = [], []
        for step in range(ctx.n_steps):
            _tebd.symmetric_static_step(state, gates, n, ctx.bond_dim,
                                        ctx.trunc_eps)
            rho = builder.undress_rdm(state.get_theta2(0))   # lab-frame RDM
            rdms.append(rho)
            max_bond.append(max((len(s) for s in state.S), default=1))
        t = np.arange(1, ctx.n_steps + 1) * ctx.dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, ctx.obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method=spec.name)

