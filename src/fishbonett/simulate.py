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
from dataclasses import dataclass, field

import numpy as np

from fishbonett.stuff import sigma_x, sigma_z
from fishbonett.bath.orthpol import make_orthpol_discretizer
from fishbonett.models.hamiltonian import FishBoneH
from fishbonett.states.comb import FishBoneNet
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
    domain : (float, float)
        Signed bath frequency window.
    n_modes, phys_dim : int
        Number of discretized modes and the local boson Hilbert-space dimension.
    temperature, beta : float, optional
        Temperature (or inverse temperature) for thermalization.
    thermalized : bool
        Set True if ``J`` is already the thermalized density.
    discretization : {'legendre', 'orthpol'}
        Bath discretization: uniform-measure Gauss-Legendre star, or the
        measure-adapted ORTHPOL star (resolves IR-divergent / sharply peaked baths).
    extra_breaks, m_per : ORTHPOL quadrature options.
    coupling : (d, d) array, optional
        System operator this bath couples to.  Only used by the multi-bath
        :class:`Fishbone` / :class:`~fishbonett.treebone.TreeFishbone` interfaces
        (a single :class:`SpinBoson` takes its coupling operator directly);
        defaults to ``sigma_z`` there when unset.
    """
    J: object
    domain: tuple
    n_modes: int = 40
    phys_dim: int = 20
    temperature: float = None
    beta: float = None
    thermalized: bool = False
    discretization: str = "legendre"
    extra_breaks: tuple = ()
    m_per: int = 60
    coupling: object = None

    def spectral_density(self):
        if self.thermalized or (self.temperature is None and self.beta is None):
            return self.J
        b = self.beta if self.beta is not None else 1.0 / self.temperature
        return thermalize(self.J, b)

    def discretizer(self):
        if self.discretization == "orthpol":
            return make_orthpol_discretizer(m_per=self.m_per,
                                            extra_breaks=self.extra_breaks)
        if self.discretization == "legendre":
            return None
        raise ValueError(f"unknown discretization {self.discretization!r}")


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

    The system need not be two-level: ``h`` may be any ``(d, d)`` Hamiltonian, so
    a composite system (e.g. a spin tensored with a vibrational mode) is described
    by giving the full ``h`` and a ``coupling`` that acts on the whole space (for
    a bath that sees only the spin, ``coupling = sigma_z (x) I_vib``).  Arbitrary
    system dimensions and initial states are supported by ``method='tebd'``; the
    MPO/tree methods still assume a two-level ``sigma_z``-coupled system.

    Parameters
    ----------
    h : (d, d) array
        System Hamiltonian.
    coupling : (d, d) array
        System operator coupling to the bath.
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
        ``observables`` maps names to (2, 2) operators; the default measures
        ``sigma_z`` and ``sigma_x``.
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
        m = method.lower().replace("_", "-")
        if m == "tebd":
            return self._run_tebd(dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                  initial, engine_kw)
        if m in _MPO_METHODS:
            return self._run_mpo(m, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                 krylov, engine_kw)
        if m in _TREE_METHODS:
            return self._run_tree(m, dt, n_steps, bond_dim, trunc_eps, obs_ops,
                                  krylov, engine_kw)
        raise ValueError(f"unknown method {method!r}; choose from tebd, "
                         "mpo-tdvp1/tdvp2/dtdvp, mpo-ip-tdvp1/tdvp2, "
                         "tree-tdvp/tdvp2/tebd")

    # -- dispatchers ---------------------------------------------------------
    def _expect_from_rdm(self, rdms, obs_ops):
        rdms = np.asarray(rdms)
        return {name: np.einsum("tij,ji->t", rdms, np.asarray(O)).real
                for name, O in obs_ops.items()}

    def _require_standard(self):
        if self.h.shape[0] != 2:
            raise ValueError("the MPO/tree methods assume a two-level system; use "
                             "method='tebd' for a general system dimension "
                             "(e.g. spin (x) vibration)")
        if not np.allclose(self.coupling, sigma_z, atol=1e-9):
            raise ValueError("the MPO/tree methods assume a sigma_z system-bath "
                             "coupling; use method='tebd' for a general coupling")
        return _decompose_h(self.h)

    def _run_mpo(self, m, dt, n_steps, bond_dim, trunc_eps, obs_ops, krylov, kw):
        eps, V = self._require_standard()
        b = self.bath
        # The MPO drivers take a half-step and advance 2*dt of physical time per
        # sweep; pass dt/2 so one sweep advances the user's physical dt (matching
        # the tree/tebd drivers, so every method reaches the same t_max).
        common = dict(eps_bias=eps, V=V, n_chain=b.n_modes, d=b.phys_dim,
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

    def _run_tree(self, m, dt, n_steps, bond_dim, trunc_eps, obs_ops, krylov, kw):
        eps, V = self._require_standard()
        b = self.bath
        common = dict(V=V, eps=eps, n_chain=b.n_modes, phys_dim=b.phys_dim, dt=dt,
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
        from fishbonett.models.backward import SpinBoson as _BackwardBuilder
        from fishbonett.states.mps import SpinBosonMPS
        b = self.bath
        n = b.n_modes
        d_sys = self.h.shape[0]
        pd = [b.phys_dim] * n + [d_sys]
        builder = _BackwardBuilder(pd)         # interaction-picture gate builder
        builder.domain = list(b.domain)
        builder.sd = b.spectral_density()
        builder.he_dy = self.coupling
        builder.h1e = self.h
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


# -- multi-bath fishbone (1D chain of electronic sites, each with baths) ------
def _as_pair(entry):
    """Normalise a per-site bath spec to (left_bath, right_bath)."""
    if entry is None:
        return (None, None)
    if isinstance(entry, (tuple, list)):
        if len(entry) == 1:
            return (entry[0], None)
        if len(entry) == 2:
            return (entry[0], entry[1])
        raise ValueError("a site takes at most two baths (left, right)")
    return (entry, None)


class Fishbone:
    """A 1D chain of electronic sites, each coupled to one or two baths.

    Parameters
    ----------
    sites : list of (d, d) array
        Electronic site Hamiltonians, one per site.
    baths : list
        One entry per site: a single :class:`Bath` (a left/``eb`` bath), a
        ``(left, right)`` pair of baths (two baths per site), or ``None``.  Each
        :class:`Bath` may carry its own ``coupling`` operator (default
        ``sigma_z`` for a left bath, ``sigma_x`` for a right bath).
    backbone : list of (d*d, d*d) array, optional
        Nearest-neighbour couplings between electronic sites ``i`` and ``i+1``
        (length ``n_sites - 1``).  Default: uncoupled sites.

    Notes
    -----
    All baths must share the same frequency ``domain`` (a limitation of the
    underlying comb Hamiltonian).  For an arbitrary (non-1D) electronic topology
    use :class:`fishbonett.treebone.TreeFishbone`.
    """

    def __init__(self, sites, baths, backbone=None):
        self.sites = [np.asarray(h, complex) for h in sites]
        self.nc = len(self.sites)
        if len(baths) != self.nc:
            raise ValueError("baths must have one entry per site")
        self.baths = [_as_pair(b) for b in baths]
        self.de = [h.shape[0] for h in self.sites]
        if backbone is None:
            backbone = [np.zeros((self.de[i] * self.de[i + 1],) * 2, complex)
                        for i in range(self.nc - 1)]
        if len(backbone) != max(self.nc - 1, 0):
            raise ValueError("backbone must have n_sites - 1 entries")
        self.backbone = [np.asarray(b, complex) for b in backbone]
        self._check_domain()

    def _check_domain(self):
        doms = []
        for lb, rb in self.baths:
            for b in (lb, rb):
                if b is not None:
                    doms.append(tuple(b.domain))
        if doms and len(set(doms)) != 1:
            raise ValueError("all baths must share the same frequency domain")
        self.domain = list(doms[0]) if doms else [0.0, 1.0]

    def _build_pd(self):
        pd = np.empty([self.nc, 4], dtype=object)
        for n, ((lb, rb), de) in enumerate(zip(self.baths, self.de)):
            pd[n, 0] = [lb.phys_dim] * lb.n_modes if lb is not None else []
            pd[n, 1] = [de]
            pd[n, 2] = []
            pd[n, 3] = [rb.phys_dim] * rb.n_modes if rb is not None else []
        return pd

    def _discretizer(self):
        discs = set()
        for lb, rb in self.baths:
            for b in (lb, rb):
                if b is not None:
                    discs.add(b.discretization)
        if len(discs) > 1:
            raise ValueError("all baths must use the same discretization")
        for lb, rb in self.baths:
            for b in (lb, rb):
                if b is not None:
                    return b.discretizer()
        return None

    def _build_h(self, g, ncap):
        pd = self._build_pd()
        ham = FishBoneH(pd)
        ham.domain = self.domain
        he_dy, hv_dy, h1e, h1v = [], [], [], []
        for n, ((lb, rb), de, hsite) in enumerate(
                zip(self.baths, self.de, self.sites)):
            two_bath = rb is not None
            if lb is not None:
                ham.sd[n, 0] = lb.spectral_density()
                lc = lb.coupling if lb.coupling is not None else sigma_z
                he_dy.append(np.asarray(lc, complex))
            else:
                he_dy.append(np.eye(de))
            if rb is not None:
                ham.sd[n, 1] = rb.spectral_density()
                rc = rb.coupling if rb.coupling is not None else sigma_x
                hv_dy.append(np.asarray(rc, complex))
            else:
                hv_dy.append(np.eye(de))
            # Route the site energy: two-bath sites carry it on h1v (the vb bond;
            # the comb builder's two-bath branch skips h1e), one-bath sites on h1e
            # (the eb bond for a single site, or the backbone bond for a chain).
            if two_bath:
                h1e.append(np.zeros((de, de), complex))
                h1v.append(hsite)
            else:
                h1e.append(hsite)
                h1v.append(np.zeros((de, de), complex))
        ham.he_dy = he_dy
        ham.hv_dy = hv_dy
        ham.h1e = h1e
        ham.h1v = h1v
        if self.nc > 1:
            ham.h2ee = list(self.backbone)
        ham.build(g=g, ncap=ncap, discretizer=self._discretizer())
        return ham

    def _init_net(self, pd, initial):
        def g_state(dim):
            t = np.zeros(dim, complex)
            t[(0,) * len(dim)] = 1.0
            return t

        nc = self.nc
        eb_t = [[g_state([1, d, 1]) for d in pd[i, 0]] for i in range(nc)]
        e_t = [[g_state([1, d, 1, 1, 1]) for d in pd[i, 1]] for i in range(nc)]
        v_t = [[g_state([1, d, 1]) for d in pd[i, 2]] for i in range(nc)]
        vb_t = [[g_state([1, d, 1]) for d in pd[i, 3]] for i in range(nc)]
        eb_s = [[np.ones(1) for _ in ch] for ch in eb_t]
        e_s = [[np.ones(1) for _ in ch] for ch in e_t]
        v_s = [[np.ones(1) for _ in ch] for ch in v_t]
        vb_s = [[np.ones(1) for _ in ch] for ch in vb_t]
        main_s = [[np.ones(1)] for _ in range(nc)]
        vb_and_main = [vb_s[i] + main_s[i] for i in range(nc)]
        for i in range(nc):
            vec = self._initial_vec(initial, i)
            e_t[i][0][:] = 0.0
            for a in range(len(vec)):
                e_t[i][0][0, a, 0, 0, 0] = vec[a]
        return FishBoneNet((eb_t, e_t, v_t, vb_t), (eb_s, e_s, v_s, vb_and_main))

    def _initial_vec(self, initial, i):
        de = self.de[i]
        if initial is None or (isinstance(initial, str) and initial == "up"):
            v = np.zeros(de, complex); v[0] = 1.0
            return v
        if isinstance(initial, str) and initial == "down":
            v = np.zeros(de, complex); v[min(1, de - 1)] = 1.0
            return v
        if isinstance(initial, str) and initial == "ground":
            w, U = np.linalg.eigh(self.sites[i])
            return U[:, int(np.argmin(w))].astype(complex)
        item = initial[i] if isinstance(initial, (list, tuple)) else initial
        v = np.asarray(item, complex)
        return v / np.linalg.norm(v)

    def run(self, *, dt, t_max=None, n_steps=None, bond_dim=100, trunc_eps=1e-10,
            observables=None, initial="up", g=1.0, ncap=20000):
        """Propagate the fishbone and return a :class:`Result` with per-site data
        (``expect[name]`` shape ``(n_steps, n_sites)``; ``rdm`` shape
        ``(n_steps, n_sites, d, d)``)."""
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        if observables is None:
            observables = {"sz": sigma_z, "sx": sigma_x} if all(
                d == 2 for d in self.de) else {}
        ham = self._build_h(g, ncap)
        pd = self._build_pd()
        state = self._init_net(pd, initial)
        state.U = ham.get_u(dt=dt)

        n_bond = [ham._L[cn] - 1 for cn in range(self.nc)]
        e_index = [ham._ebL[cn] for cn in range(self.nc)]
        rdms = np.empty((n_steps, self.nc), dtype=object)
        for step in range(n_steps):
            if self.nc > 1:
                for i in range(self.nc - 1):
                    state.update_bond(-1, i, bond_dim, trunc_eps)
            for cn in range(self.nc):
                for j in range(n_bond[cn]):
                    state.update_bond(cn, j, bond_dim, trunc_eps)
            for cn in range(self.nc):
                th = state.get_theta1(cn, e_index[cn])
                rho = np.einsum("LiUDR,LjUDR->ij", th, th.conj())
                rdms[step, cn] = rho / np.trace(rho).real

        expect = {}
        for name, O in observables.items():
            O = np.asarray(O)
            expect[name] = np.array(
                [[np.trace(rdms[tn, cn] @ O).real for cn in range(self.nc)]
                 for tn in range(n_steps)])
        rdm = np.array([[rdms[tn, cn] for cn in range(self.nc)]
                        for tn in range(n_steps)])
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=expect, rdm=rdm, method="fishbone",
                      meta={"n_sites": self.nc})
