"""User-friendly high-level interface for spin-boson dynamics.

Wraps the low-level engines (TEBD, MPO/TDVP, tree) behind a small set of classes,
so a simulation is specified declaratively and run with a single call instead of
by hand-writing a TEBD sweep loop::

    import numpy as np
    from fishbonett.simulate import Bath, SpinBoson
    from fishbonett.stuff import sigma_x, sigma_z

    bath = Bath(J=lambda w: 0.5 * w * np.exp(-w / 5),
                domain=(-25, 36), temperature=1.0,
                n_modes=40, phys_dim=20, discretization='orthpol')
    model = SpinBoson(h=0.5 * eps * sigma_z + V * sigma_x, coupling=sigma_z, bath=bath)
    result = model.run(dt=0.01, t_max=4.0, method='tree-tdvp2', bond_dim=200,
                       observables={'sz': sigma_z, 'sx': sigma_x})

    result.t                 # time grid
    result.expect['sz']      # <sigma_z>(t)
    result.max_bond          # peak bond dimension per step (adaptive methods)
"""
from dataclasses import dataclass, field, replace

import numpy as np

from fishbonett.stuff import sigma_x, sigma_z
from fishbonett.bath.orthpol import make_orthpol_discretizer
from fishbonett import mpo as _mpo
from fishbonett import tree as _tree

__all__ = ["Bath", "SpinBoson", "Fishbone", "Result", "thermalize"]

_MPO_METHODS = {"tdvp1": "run_tdvp1", "mpo-tdvp1": "run_tdvp1",
                "tdvp2": "run_tdvp2", "mpo-tdvp2": "run_tdvp2",
                "dtdvp": "run_dtdvp", "mpo-dtdvp": "run_dtdvp",
                "mpo-ip-tdvp1": "run_ip_tdvp1", "ip-tdvp1": "run_ip_tdvp1",
                "mpo-ip-tdvp2": "run_ip_tdvp2", "ip-tdvp2": "run_ip_tdvp2"}
_TREE_METHODS = {"tree-tdvp": "run_tree_tdvp", "tree-tdvp1": "run_tree_tdvp",
                 "tree-tdvp2": "run_tree_tdvp2", "tree-tebd": "run_tree_tebd"}


def thermalize(J, beta):
    """T-TEDOPA thermalized spectral density ``J_beta`` (positive on both halves)
    from a zero-temperature ``J(w>0)``."""
    def Jb(w):
        aw = abs(w)
        if aw < 1e-12:
            return 0.0
        nb = 1.0 / np.expm1(beta * aw)
        j = float(J(aw))
        return j * (nb + 1.0) if w > 0 else j * nb
    return Jb


@dataclass
class Bath:
    """A bosonic bath specified by its spectral density and discretization.

    Parameters
    ----------
    J : callable
        Spectral density ``J(w)``.  If ``temperature`` (or ``beta``) is given and
        ``thermalized`` is False, ``J`` is treated as the zero-temperature density
        and thermalized internally.
    domain : (float, float), optional
        Signed bath frequency window.  If omitted, it is chosen automatically as
        the window covering 99.9% of the reorganization energy
        ``lambda = (1/pi) int J(w)/w dw`` (signed when a temperature is set).
    n_modes : int, optional
        Number of discretized modes.  If omitted, it is chosen automatically from
        the light-cone of the interaction-picture chain couplings ``d_j(t)`` for
        the propagation time (so it depends on ``t_max``); see
        :func:`fishbonett.bath.auto.auto_n_modes`.
    phys_dim : int
        The local boson Hilbert-space dimension of each mode.
    temperature, beta : float, optional
        Temperature (or inverse temperature) for thermalization.
    thermalized : bool
        Set True if ``J`` is already the thermalized density.
    discretization : {'legendre', 'orthpol'}
        Bath discretization: uniform-measure Gauss-Legendre star, or the
        measure-adapted ORTHPOL star (resolves IR-divergent / sharply peaked baths).
    extra_breaks, m_per : ORTHPOL quadrature options.
    coupling : (d, d) array, or list of (d, d) arrays
        System operator(s) this bath couples to.  A single operator is an ordinary
        bath.  A **list** of operators makes this a *multichannel single bath*: the
        one bath couples through every operator on shared modes (distinct from
        several independent baths -- the channels cross-correlate).  For a
        multichannel bath ``J`` is either one spectral density (shared) or a list of
        the same length as ``coupling`` (one per channel), and the discretization
        must be ``'legendre'`` (shared Gauss nodes).  Defaults to ``sigma_z``.
    """
    J: object
    domain: tuple = None
    n_modes: int = None
    phys_dim: int = 20
    temperature: float = None
    beta: float = None
    thermalized: bool = False
    discretization: str = "legendre"
    extra_breaks: tuple = ()
    m_per: int = 60
    coupling: object = None

    def _thermalized(self, Jfunc):
        if self.thermalized or (self.temperature is None and self.beta is None):
            return Jfunc
        b = self.beta if self.beta is not None else 1.0 / self.temperature
        return thermalize(Jfunc, b)

    def spectral_density(self):
        J0 = self.J[0] if isinstance(self.J, (list, tuple)) else self.J
        return self._thermalized(J0)

    @property
    def is_multichannel(self):
        """True when the bath couples through several operators (``coupling`` is a
        list) -- a single bath with cross-correlated channels, distinct from
        several independent baths."""
        return isinstance(self.coupling, (list, tuple))

    def channels(self):
        """``[(thermalized_J_c, operator_c), ...]`` for a multichannel bath.

        The channels share the same mode grid (same ``domain``/``n_modes``/
        ``discretization``); ``J`` may be one spectral density (shared by all
        channels) or a list of the same length as ``coupling``."""
        ops = list(self.coupling)
        Js = self.J if isinstance(self.J, (list, tuple)) else [self.J] * len(ops)
        if len(Js) != len(ops):
            raise ValueError("a multichannel Bath needs `J` and `coupling` of the "
                             "same length (one spectral density per channel)")
        return [(self._thermalized(Jc), np.asarray(op, complex))
                for Jc, op in zip(Js, ops)]

    def discretizer(self):
        if self.discretization == "orthpol":
            return make_orthpol_discretizer(m_per=self.m_per,
                                            extra_breaks=self.extra_breaks)
        if self.discretization == "legendre":
            return None
        raise ValueError(f"unknown discretization {self.discretization!r}")

    def _auto_domain(self):
        from fishbonett.bath.auto import auto_domain
        beta = self.beta if self.beta is not None else (
            1.0 / self.temperature if self.temperature is not None else None)
        Js = self.J if isinstance(self.J, (list, tuple)) else [self.J]
        doms = [auto_domain(Jc, beta=beta) for Jc in Js]          # cover every channel
        return (min(d[0] for d in doms), max(d[1] for d in doms))

    def resolved(self, t_max=None):
        """A copy with automatic ``domain`` / ``n_modes`` filled in.

        ``domain`` (if unset) becomes the window covering 99.9% of the
        reorganization energy; ``n_modes`` (if unset) the light-cone extent of the
        interaction-picture chain couplings up to ``t_max``.  Returns ``self`` when
        both are already given.  Called by ``run`` with the propagation time."""
        domain = self.domain if self.domain is not None else self._auto_domain()
        n_modes = self.n_modes
        if n_modes is None:
            if t_max is None:
                raise ValueError("Bath.n_modes is automatic and needs the "
                                 "propagation time; call from run() (which supplies "
                                 "t_max) or set n_modes explicitly")
            from fishbonett.bath.auto import auto_n_modes
            n_modes = auto_n_modes(self.spectral_density(), domain, t_max,
                                   discretizer=self.discretizer())
        if domain is self.domain and n_modes == self.n_modes:
            return self
        return replace(self, domain=tuple(domain), n_modes=int(n_modes))


@dataclass
class Result:
    """Result of a propagation."""
    t: np.ndarray
    expect: dict                      # observable name -> array over time
    max_bond: np.ndarray = None       # peak bond dimension per step (adaptive)
    rdm: np.ndarray = None            # spin reduced density matrix per step (T,2,2)
    method: str = ""
    meta: dict = field(default_factory=dict)


def _decompose_h(h):
    """Decompose ``h`` assuming ``h = (eps/2) sigma_z + V sigma_x``."""
    h = np.asarray(h, complex)
    return float((h[0, 0] - h[1, 1]).real), float(h[0, 1].real)


class SpinBoson:
    """A system coupled to a :class:`Bath`.

    ``h`` may be any ``(d, d)`` Hermitian Hamiltonian (not only two-level) and the
    coupling ``O`` any ``(d, d)`` Hermitian operator (not only ``sigma_z``); *every*
    method (``tebd``, the MPO and the tree engines) supports an arbitrary system
    dimension, a general coupling and an arbitrary initial state.  When the system
    has *distinct* internal degrees of freedom (e.g. a spin **and** a vibration),
    prefer to keep each on its own site with
    :class:`~fishbonett.treebone.TreeFishbone` (a spin site and a vibration site
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
    def run(self, *, dt, t_max=None, n_steps=None, method="tree-tdvp2",
            bond_dim=200, trunc_eps=1e-10, observables=None, initial="up",
            krylov=25, **engine_kw):
        """Propagate and return a :class:`Result`.

        ``method`` is one of ``'tebd'`` (interaction-picture swap network),
        ``'mpo-tdvp1' | 'mpo-tdvp2' | 'mpo-dtdvp'`` (Schroedinger-picture MPO),
        ``'mpo-ip-tdvp1' | 'mpo-ip-tdvp2'`` (interaction-picture star MPO), or
        ``'tree-tdvp' | 'tree-tdvp2' | 'tree-tebd'`` (interaction-picture tree).
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
        if observables is not None:
            obs_ops = observables
        elif self.h.shape[0] == 2:
            obs_ops = {"sz": sigma_z, "sx": sigma_x}
        else:
            obs_ops = {}                    # general system: return the RDM only
        if getattr(self.bath, "is_multichannel", False):
            return self._run_multichannel(dt, n_steps, bond_dim, trunc_eps,
                                          obs_ops, initial)
        m = method.lower().replace("_", "-")
        if m == "tebd":
            return self._run_tebd(dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                  initial, engine_kw)
        if m in _MPO_METHODS:
            return self._run_mpo(m, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                 initial, krylov, engine_kw)
        if m in _TREE_METHODS:
            return self._run_tree(m, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                  initial, krylov, engine_kw)
        raise ValueError(f"unknown method {method!r}; choose from tebd, "
                         "mpo-tdvp1/tdvp2/dtdvp, mpo-ip-tdvp1/tdvp2, "
                         "tree-tdvp/tdvp2/tebd")

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
        from fishbonett.treebone import TreeFishbone
        fb = TreeFishbone(sites=[self.h], edges=[], baths=[self.bath])
        r = fb.run(dt=dt, n_steps=n_steps, bond_dim=bond_dim, trunc_eps=trunc_eps,
                   observables=obs_ops, initial=[self._initial_state(initial)])
        expect = {name: r.expect[name][:, 0] for name in r.expect}
        rdm = np.array([r.rdm[k, 0] for k in range(n_steps)])
        return Result(t=r.t, expect=expect, rdm=rdm, method="multichannel")

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
        from fishbonett.models.interaction_picture import SpinBoson as _IPBuilder
        from fishbonett.states.mps import SpinBosonMPS
        b = self.bath.resolved(n_steps * dt)
        n = b.n_modes
        d_sys = self.h.shape[0]
        pd = [b.phys_dim] * n + [d_sys]
        builder = _IPBuilder(pd)               # interaction-picture gate builder
        builder.domain = list(b.domain)
        builder.sd = b.spectral_density()
        builder.coupling = self.coupling
        builder.h_sys = self.h
        builder.build(g=1, ncap=kw.get("ncap", 20000), discretizer=b.discretizer())

        state = SpinBosonMPS(pd)               # the MPS being evolved
        psi0 = self._initial_state(initial)
        state.B[-1][:] = 0.0
        for a in range(d_sys):
            state.B[-1][0, a, 0] = psi0[a]

        # Each iteration is a symmetric forward/backward pair over hdt = dt/2, so
        # it advances the user's physical dt (matching the tree/mpo drivers).
        hdt = dt / 2.0
        rdms, max_bond = [], []
        for step in range(n_steps):
            t0 = 2 * step * hdt
            u_fwd, _ = builder.get_u(t0, hdt, mode="normal")
            state.U = u_fwd
            for j in range(n - 1, 0, -1):
                state.update_bond(j, bond_dim, trunc_eps, swap=1)
            state.update_bond(0, bond_dim, trunc_eps, swap=0)
            state.update_bond(0, bond_dim, trunc_eps, swap=0)
            _, u_bwd = builder.get_u(t0 + hdt, hdt, mode="reverse")
            state.U = u_bwd
            for j in range(1, n):
                state.update_bond(j, bond_dim, trunc_eps, swap=1)
            theta = state.get_theta1(n)
            rho = np.einsum("LiR,LjR->ij", theta, theta.conj())
            rdms.append(rho / np.trace(rho).real)
            max_bond.append(max((len(s) for s in state.S), default=1))
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      max_bond=np.array(max_bond), rdm=np.asarray(rdms), method="tebd")


# -- 1D fishbone: a specialization of TreeFishbone to a linear backbone -------
class Fishbone:
    """A 1D chain of electronic sites, each coupled to one or two baths.

    A convenience specialization of :class:`~fishbonett.treebone.TreeFishbone`
    (which handles *any* loop-free electronic topology) to a **linear** backbone:
    site ``i`` is joined to site ``i+1`` by ``backbone[i]``.  Each ``baths`` entry
    is a single :class:`Bath` (one bath -- may be multichannel), a ``(left, right)``
    pair (two baths per site -- the fishbone), or ``None``.  A left bath defaults
    to a ``sigma_z`` coupling and a right bath to ``sigma_x`` when the :class:`Bath`
    itself sets none.  ``run`` and the returned :class:`Result` are exactly those
    of :meth:`fishbonett.treebone.TreeFishbone.run`.
    """

    def __init__(self, sites, baths, backbone=None):
        self.sites = [np.asarray(h, complex) for h in sites]
        self.nc = len(self.sites)
        self.de = [h.shape[0] for h in self.sites]
        if len(baths) != self.nc:
            raise ValueError("baths must have one entry per site")
        self.baths = list(baths)
        if backbone is None:
            backbone = [np.zeros((self.de[i] * self.de[i + 1],) * 2, complex)
                        for i in range(self.nc - 1)]
        if len(backbone) != max(self.nc - 1, 0):
            raise ValueError("backbone must have n_sites - 1 entries")
        self.backbone = [np.asarray(b, complex) for b in backbone]

    @staticmethod
    def _site_baths(entry):
        """Map a per-site bath spec to the TreeFishbone form, defaulting a
        left/right pair's couplings to sigma_z / sigma_x when unset."""
        if entry is None:
            return None
        if isinstance(entry, (tuple, list)):
            out = []
            for pos, b in enumerate(entry):
                if b is None:
                    continue
                if b.coupling is None:
                    b = replace(b, coupling=(sigma_z if pos == 0 else sigma_x))
                out.append(b)
            return out
        return entry

    def _tree(self):
        from fishbonett.treebone import TreeFishbone
        edges = [(i, i + 1, self.backbone[i]) for i in range(self.nc - 1)]
        return TreeFishbone(sites=self.sites, edges=edges,
                            baths=[self._site_baths(b) for b in self.baths])

    def run(self, **kwargs):
        """Propagate the 1D fishbone (delegates to the general tree engine).  See
        :meth:`fishbonett.treebone.TreeFishbone.run` for the arguments, the
        observable spec and the per-site :class:`Result` layout."""
        return self._tree().run(**kwargs)
