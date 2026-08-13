"""The single-system models: one system site coupled to one bath.

Four models share this one class, distinguished by how the bath is represented
and selected through ``run(method=...)``:

* ``chain`` -- modes chain-mapped into 1D (all three frames);
* ``star`` -- no chain mapping (interaction picture);
* ``mode-tree`` -- modes on a balanced binary tree (interaction picture);
* ``multichannel`` -- several couplings on shared modes, selected automatically
  by giving the :class:`~fishbonett.bath.spec.Bath` a *list* of couplings.

:mod:`fishbonett.models.registry` is the authority on which method belongs to
which model and frame.
"""
import numpy as np
import scipy.linalg as _la

from fishbonett.operators import sigma_x, sigma_z
from fishbonett.linalg import Truncation
from fishbonett.evolve import tdvp as _mpo
from fishbonett.evolve import tebd as _tebd
from fishbonett.evolve import treetdvp as _tree
from fishbonett.models.result import Result
from fishbonett.models.registry import (
    METHOD_FRAMES, methods_of, unknown_method_error,
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
            if n not in _FIXED_BOND_METHODS]

_MPO_METHODS = {"mpo-tdvp1": "run_tdvp1",
                "mpo-tdvp2": "run_tdvp2",
                "mpo-dtdvp": "run_dtdvp",
                "mpo-ip-tdvp1": "run_ip_tdvp1",
                "mpo-ip-tdvp2": "run_ip_tdvp2"}
#: Methods whose bond dimension is fixed up front (1-site TDVP variants cannot grow
#: a bond, and the adaptive DTDVP needs a finite ceiling), so ``bond_dim=None``
#: ("unlimited") is not meaningful for them.
_FIXED_BOND_METHODS = frozenset({
    "mpo-tdvp1", "mpo-ip-tdvp1", "polaron-tdvp1",
    "mpo-dtdvp", "polaron-dtdvp", "tree-tdvp",
})

#: Polaron-frame TDVP variants.  The polaron ``H~`` is time-independent, so it has
#: a plain MPO and can drive the standard 1-site / 2-site / bond-adaptive sweeps.
_POLARON_TDVP_METHODS = {"polaron-tdvp1": "tdvp1",
                         "polaron-tdvp2": "tdvp2",
                         "polaron-dtdvp": "dtdvp"}
_TREE_METHODS = {"tree-tdvp": "run_tree_tdvp",
                 "tree-tdvp2": "run_tree_tdvp2", "tree-tebd": "run_tree_tebd"}


class SystemBath:
    """A system coupled to a :class:`Bath`.

    ``h`` may be any ``(d, d)`` Hermitian Hamiltonian (not only two-level) and the
    coupling ``O`` any ``(d, d)`` Hermitian operator (not only ``sigma_z``); *every*
    method (``tebd``, the MPO and the tree engines) supports an arbitrary system
    dimension, a general coupling and an arbitrary initial state.  When the system
    has *distinct* internal degrees of freedom (e.g. a spin **and** a vibration),
    prefer to keep each on its own site with
    :class:`~fishbonett.models.fishbone.TreeFishbone` (a spin site and a vibration site
    joined by an edge, with the bath on the spin) -- putting ``spin (x) vibration``
    on a single ``d = 2*d_vib`` site here works but defeats the MPS advantage.
    Passing a multichannel :class:`Bath` (``coupling`` a list) routes through the
    tree so the spin stays on its own site.

    Parameters
    ----------
    h : (d, d) array
        System Hamiltonian.
    coupling : (d, d) array, or list of (d, d) arrays
        System operator(s) coupling to the bath (a list for a multichannel bath).
    bath : Bath
    """

    def __init__(self, h, coupling, bath):
        self.h = np.asarray(h, complex)
        self.coupling = np.asarray(coupling, complex)
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
        if bond_dim is None and method.lower().replace("_", "-") in _FIXED_BOND_METHODS:
            alternatives = _bond_growing_siblings(method) or ["tebd"]
            raise ValueError(
                f"method {method!r} has a fixed bond dimension and cannot grow it "
                "from a product state, so bond_dim must be given explicitly "
                "(bond_dim=None means 'unlimited', which is only meaningful for "
                "the truncation-driven methods).  To let trunc_eps choose the bond "
                "instead, use a bond-growing method of the same frame: "
                f"{', '.join(alternatives)}")
        if observables is not None:
            obs_ops = observables
        elif self.h.shape[0] == 2:
            obs_ops = {"sz": sigma_z, "sx": sigma_x}
        else:
            obs_ops = {}                    # general system: return the RDM only
        m = method.lower().replace("_", "-")
        if getattr(self.bath, "is_multichannel", False):
            # The multichannel model is selected by the bath's shape, so there is
            # nothing for `method` to choose.  Say so rather than ignoring it.
            if m not in methods_of("multichannel") and method != _DEFAULT_METHOD:
                raise ValueError(
                    f"this Bath is multichannel (a list of couplings), which "
                    f"selects the 'multichannel' model -- it has one propagator "
                    f"and does not take method={method!r}.  Drop the method "
                    f"argument, or pass a single coupling operator to use the "
                    f"single-channel models.")
            return self._run_multichannel(dt, n_steps, bond_dim, trunc_eps,
                                          obs_ops, initial)
        if m == "tebd":
            return self._run_tebd(dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                  initial, engine_kw)
        if m == "trotter-mpo":
            return self._run_trotter_mpo(dt, n_steps, bond_dim, trunc_eps,
                                         obs_ops, initial, engine_kw)
        if m == "polaron":
            return self._run_polaron(dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                     initial, engine_kw)
        if m in _POLARON_TDVP_METHODS:
            return self._run_polaron_tdvp(m, dt, n_steps, bond_dim, trunc_eps,
                                          obs_ops, initial, krylov, engine_kw)
        if m in _MPO_METHODS:
            return self._run_mpo(m, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                 initial, krylov, engine_kw)
        if m in _TREE_METHODS:
            return self._run_tree(m, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                  initial, krylov, engine_kw)
        raise unknown_method_error(m)

    # -- dispatchers ---------------------------------------------------------
    def _expect_from_rdm(self, rdms, obs_ops):
        rdms = np.asarray(rdms)
        return {name: np.einsum("tij,ji->t", rdms, np.asarray(O)).real
                for name, O in obs_ops.items()}

    def _check_system(self):
        """Validate that ``h`` and ``coupling`` are square Hermitian operators of
        matching dimension.  The MPO/tree engines accept a general system: any
        Hermitian ``h`` and coupling ``O`` (the interaction-picture gates
        diagonalize ``O``), not only a two-level ``sigma_z`` spin-boson model."""
        h, O = self.h, self.coupling
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("h must be a square matrix")
        if O.shape != h.shape:
            raise ValueError(f"coupling shape {O.shape} does not match the system "
                             f"dimension {h.shape}")
        if not np.allclose(h, h.conj().T, atol=1e-9):
            raise ValueError("h must be Hermitian")
        if not np.allclose(O, O.conj().T, atol=1e-9):
            raise ValueError("the system-bath coupling must be Hermitian")

    def _run_mpo(self, m, dt, n_steps, bond_dim, trunc_eps, obs_ops, initial,
                 krylov, kw):
        self._check_system()
        b = self.bath.resolved(n_steps * dt)
        # The MPO drivers take a half-step and advance 2*dt of physical time per
        # sweep; pass dt/2 so one sweep advances the user's physical dt (matching
        # the tree/tebd drivers, so every method reaches the same t_max).
        common = dict(hsys=self.h, cop=self.coupling,
                      init=self._initial_state(initial),
                      n_chain=b.n_modes, d=b.phys_dim,
                      dt=dt / 2.0, nsteps=n_steps, krylov=krylov,
                      discretizer=b.discretizer(), observe=_mpo.measure_rdm, **kw)
        sd, dom = b.spectral_density(), b.domain
        driver = _MPO_METHODS[m]
        maxb = None
        if driver == "run_tdvp1":
            t, rdms = _mpo.run_tdvp1(sd, dom, D=bond_dim, **common)
        elif driver == "run_ip_tdvp1":
            t, rdms = _mpo.run_ip_tdvp1(sd, dom, D=bond_dim, **common)
        elif driver == "run_tdvp2":
            t, rdms, maxb = _mpo.run_tdvp2(sd, dom, chi_max=bond_dim,
                                           eps=trunc_eps, **common)
        elif driver == "run_ip_tdvp2":
            t, rdms, maxb = _mpo.run_ip_tdvp2(sd, dom, chi_max=bond_dim,
                                              eps=trunc_eps, **common)
        else:
            t, rdms, maxb = _mpo.run_dtdvp(sd, dom, Dlim=bond_dim, **common)
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      max_bond=maxb, rdm=np.asarray(rdms), method=m)

    def _run_tree(self, m, dt, n_steps, bond_dim, trunc_eps, obs_ops, initial,
                  krylov, kw):
        self._check_system()
        b = self.bath.resolved(n_steps * dt)
        common = dict(hsys=self.h, cop=self.coupling,
                      init=self._initial_state(initial),
                      n_chain=b.n_modes, phys_dim=b.phys_dim, dt=dt,
                      nsteps=n_steps, D=bond_dim, discretizer=b.discretizer(),
                      observe=_tree.measure_rdm_oc, **kw)
        sd, dom = b.spectral_density(), b.domain
        if _TREE_METHODS[m] == "run_tree_tdvp":
            t, rdms = _tree.run_tree_tdvp(sd, dom, krylov=krylov, **common)
        elif _TREE_METHODS[m] == "run_tree_tdvp2":
            t, rdms = _tree.run_tree_tdvp2(sd, dom, trunc_eps=trunc_eps,
                                           krylov=krylov, **common)
        else:
            t, rdms = _tree.run_tree_tebd(sd, dom, trunc_eps=trunc_eps, **common)
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      rdm=np.asarray(rdms), method=m)

    def _run_multichannel(self, dt, n_steps, bond_dim, trunc_eps, obs_ops, initial):
        """One bath coupled to the system through several operators: a shared-mode
        star attached to the (single) system site.  Built on the tree engine so the
        system stays on its own site."""
        from fishbonett.models.fishbone import TreeFishbone
        fb = TreeFishbone(sites=[self.h], edges=[], baths=[self.bath])
        r = fb.run(dt=dt, n_steps=n_steps, bond_dim=bond_dim, trunc_eps=trunc_eps,
                   observables=obs_ops, initial=[self._initial_state(initial)])
        # collapse the single-site axis: this is a single-system model, so the
        # Result should have the single-system shape (see models/result.py)
        expect = {name: r.expect[name][:, 0] for name in r.expect}
        rdm = np.array([r.rdm[k, 0] for k in range(n_steps)])
        return Result(t=r.t, expect=expect, rdm=rdm, max_bond=r.max_bond,
                      method=methods_of("multichannel")[0])

    def _initial_state(self, initial):
        """Initial system state as a ``d_sys``-vector.  ``"up"``/``"down"`` are the
        first two basis states, ``"ground"`` the ground state of ``h``; a vector
        (e.g. an explicit spin (x) vibration state) is accepted and normalized."""
        d = self.h.shape[0]
        if isinstance(initial, str):
            if initial == "up":
                v = np.zeros(d, complex); v[0] = 1.0; return v
            if initial == "down":
                v = np.zeros(d, complex); v[min(1, d - 1)] = 1.0; return v
            if initial == "ground":
                w, U = np.linalg.eigh(self.h)
                return U[:, int(np.argmin(w))].astype(complex)
            raise ValueError(f"unknown initial state {initial!r}")
        v = np.asarray(initial, complex).reshape(-1)
        if v.shape[0] != d:
            raise ValueError(f"initial state has length {v.shape[0]}, expected "
                             f"the system dimension {d}")
        return v / np.linalg.norm(v)

    def _run_tebd(self, dt, n_steps, bond_dim, trunc_eps, obs_ops, initial, kw):
        from fishbonett.frames.interaction_picture import SystemBathIP as _IPBuilder
        from fishbonett.states.mps import SystemBathMPS
        b = self.bath.resolved(n_steps * dt)
        n = b.n_modes
        d_sys = self.h.shape[0]
        pd = [d_sys] + [b.phys_dim] * n
        builder = _IPBuilder(pd)               # interaction-picture gate builder
        builder.domain = list(b.domain)
        builder.sd = b.spectral_density()
        builder.coupling = self.coupling
        builder.h_sys = self.h
        builder.build(g=1, ncap=kw.get("ncap", 20000), discretizer=b.discretizer())

        state = SystemBathMPS(pd)               # the MPS being evolved
        psi0 = self._initial_state(initial)
        state.B[0][:] = 0.0
        for a in range(d_sys):
            state.B[0][0, a, 0] = psi0[a]

        # One symmetric (Strang) swap-network step per iteration, so each advances
        # the user's physical dt -- matching the tree/mpo drivers.
        rdms, max_bond = [], []
        for step in range(n_steps):
            _tebd.symmetric_swap_step(state, builder, step * dt, dt, n,
                                      bond_dim, trunc_eps)
            theta = state.get_theta1(0)
            rho = np.einsum("LiR,LjR->ij", theta, theta.conj())
            rdms.append(rho / np.trace(rho).real)
            max_bond.append(max((len(s) for s in state.S), default=1))
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms), method="tebd")

    def _run_trotter_mpo(self, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                         initial, kw):
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
        b = self.bath.resolved(n_steps * dt)
        n, d_sys = b.n_modes, self.h.shape[0]
        builder = SystemBathIP([d_sys] + [b.phys_dim] * n)
        builder.domain = list(b.domain)
        builder.sd = b.spectral_density()
        builder.coupling = self.coupling
        builder.h_sys = self.h
        builder.build(g=1, ncap=kw.get("ncap", 20000), discretizer=b.discretizer())

        # sites are [system, mode_0, ..., mode_{n-1}] for the MPO
        A = product_state([d_sys] + [b.phys_dim] * n, self._initial_state(initial))
        u_half = _la.expm(-0.5j * dt * np.asarray(self.h, complex))
        rdms, max_bond = [], []
        for step in range(n_steps):
            A[0] = np.einsum('ij,ajb->aib', u_half, A[0])        # half system step
            A = compress(apply_mpo(A, builder.displacement_mpo(step * dt, dt)),
                         bond_dim, trunc_eps)
            A[0] = np.einsum('ij,ajb->aib', u_half, A[0])
            rho = np.einsum('lsr,ltr->st', A[0], A[0].conj())
            rdms.append(rho / np.trace(rho).real)
            max_bond.append(max(bond_dims(A)))
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method="trotter-mpo")

    def _polaron_builder(self, dt, n_steps):
        """Shared polaron setup: validate, resolve the bath and build the frame.
        Returns ``(builder, resolved_bath, n_modes, pd)``."""
        from fishbonett.frames.polaron import SystemBathPolaron
        self._check_system()
        b = self.bath.resolved(n_steps * dt)
        n, d_sys = b.n_modes, self.h.shape[0]
        pd = [d_sys] + [b.phys_dim] * n
        builder = SystemBathPolaron(pd)
        builder.domain = list(b.domain)
        builder.sd = b.spectral_density()
        builder.coupling = self.coupling
        builder.h_sys = self.h
        builder.build(discretizer=b.discretizer())
        return builder, b, n, pd

    def _run_polaron_tdvp(self, m, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                          initial, krylov, kw):
        """Polaron frame propagated with TDVP.  Because ``H~`` is time-independent
        it has a plain MPO (:meth:`~fishbonett.frames.polaron.SystemBathPolaron.mpo`),
        so the 1-site / 2-site / bond-adaptive sweeps all apply.  1-site TDVP never
        forms a two-site block, which avoids the ``O(d^4)`` boson-boson gates of the
        polaron TEBD sweep."""
        from fishbonett.evolve.tdvp import (init_mps, tdvp1sweep, tdvp2sweep,
                                            tdvp1sweep_dynamic, bonddims,
                                            _pad_bonds, right_canonicalize)
        builder, b, n, pd = self._polaron_builder(dt, n_steps)
        variant = _POLARON_TDVP_METHODS[m]
        M = builder.mpo()
        A = init_mps(len(M), b.phys_dim, np.zeros(self.h.shape[0], complex))
        A[0], A[1] = builder.initial_mps_pair(self._initial_state(initial))
        if variant == "tdvp1":
            # 1-site TDVP conserves the bond dimension, so it cannot grow out of a
            # product state: pad to the requested bond first (as run_tdvp1 does).
            A = right_canonicalize(_pad_bonds(A, bond_dim))
        env = Afull = FRs = None
        rdms, max_bond = [], []
        for _ in range(n_steps):
            if variant == "tdvp1":
                A, env = tdvp1sweep(dt, A, M, env, m=krylov)
            elif variant == "tdvp2":
                A, env = tdvp2sweep(dt, A, M, bond_dim, trunc_eps, env, m=krylov)
            else:
                A, Afull, FRs, _ = tdvp1sweep_dynamic(
                    dt, A, M, Afull, FRs, prec=kw.get("prec", trunc_eps),
                    Dlim=bond_dim, Dplusmax=kw.get("Dplusmax", 4), m=krylov)
            rdms.append(builder.undress_rdm_tdvp(A[0], A[1]))   # lab frame
            max_bond.append(max(bonddims(A)))
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms), method=m)

    def _run_polaron(self, dt, n_steps, bond_dim, trunc_eps, obs_ops, initial, kw):
        """Polaron-frame chain: the static system-bath coupling is absorbed into a
        displacement of the first (reweighted-``J/w^2``) chain mode ``c0``, leaving
        a free chain plus a dressed ``(c0, system)`` gate.  Plain nearest-neighbour
        Trotter (no swap network); the physical bath vacuum is a displaced coherent
        state on ``c0``; lab-frame observables are recovered by un-dressing.
        See :mod:`fishbonett.frames.polaron`."""
        from fishbonett.states.mps import SystemBathMPS
        builder, b, n, pd = self._polaron_builder(dt, n_steps)
        state = SystemBathMPS(pd)               # boson sites default to vacuum
        psi0 = self._initial_state(initial)
        # displaced (system, c0) initial block at bond 0; other boson sites stay vacuum
        state.split_truncate_theta(builder.initial_theta(psi0), 0, bond_dim,
                                   1e-14)

        gates = builder.gates(dt / 2.0)          # static; symmetric Strang per step
        rdms, max_bond = [], []
        for step in range(n_steps):
            _tebd.symmetric_static_step(state, gates, n, bond_dim, trunc_eps)
            rho = builder.undress_rdm(state.get_theta2(0))   # lab-frame RDM
            rdms.append(rho)
            max_bond.append(max((len(s) for s in state.S), default=1))
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms),
                      method="polaron")

