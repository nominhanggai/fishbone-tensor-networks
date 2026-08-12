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
from fishbonett.orthpol_discretization import make_orthpol_discretizer
from fishbonett import mpo as _mpo
from fishbonett import tree as _tree

__all__ = ["Bath", "SpinBoson", "Result", "thermalize"]

_MPO_METHODS = {"tdvp1": "run_tdvp1", "mpo-tdvp1": "run_tdvp1",
                "tdvp2": "run_tdvp2", "mpo-tdvp2": "run_tdvp2",
                "dtdvp": "run_dtdvp", "mpo-dtdvp": "run_dtdvp"}
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
        :class:`~fishbonett.fishbone_sim.Fishbone` interface (a single
        :class:`SpinBoson` takes its coupling operator directly); defaults to
        ``sigma_z`` there when unset.
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
    """A two-level system coupled to a :class:`Bath`.

    Parameters
    ----------
    h : (2, 2) array
        System Hamiltonian.
    coupling : (2, 2) array
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
        ``'mpo-tdvp1' | 'mpo-tdvp2' | 'mpo-dtdvp'`` (Schroedinger-picture MPO), or
        ``'tree-tdvp' | 'tree-tdvp2' | 'tree-tebd'`` (interaction-picture tree).
        ``observables`` maps names to (2, 2) operators; the default measures
        ``sigma_z`` and ``sigma_x``.
        """
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        obs_ops = observables if observables is not None else {
            "sz": sigma_z, "sx": sigma_x}
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
                         "mpo-tdvp1/tdvp2/dtdvp, tree-tdvp/tdvp2/tebd")

    # -- dispatchers ---------------------------------------------------------
    def _expect_from_rdm(self, rdms, obs_ops):
        rdms = np.asarray(rdms)
        return {name: np.einsum("tij,ji->t", rdms, np.asarray(O)).real
                for name, O in obs_ops.items()}

    def _require_standard(self):
        if not np.allclose(self.coupling, sigma_z, atol=1e-9):
            raise ValueError("the MPO/tree methods assume a sigma_z system-bath "
                             "coupling; use method='tebd' for a general coupling")
        return _decompose_h(self.h)

    def _run_mpo(self, m, dt, n_steps, bond_dim, trunc_eps, obs_ops, krylov, kw):
        eps, V = self._require_standard()
        b = self.bath
        common = dict(eps_bias=eps, V=V, n_chain=b.n_modes, d=b.phys_dim, dt=dt,
                      nsteps=n_steps, krylov=krylov, discretizer=b.discretizer(),
                      observe=_mpo.measure_rdm, **kw)
        sd, dom = b.spectral_density(), b.domain
        maxb = None
        if _MPO_METHODS[m] == "run_tdvp1":
            t, rdms = _mpo.run_tdvp1(sd, dom, D=bond_dim, **common)
        elif _MPO_METHODS[m] == "run_tdvp2":
            t, rdms, maxb = _mpo.run_tdvp2(sd, dom, chi_max=bond_dim,
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

    def _initial_spin(self, initial):
        if isinstance(initial, str):
            if initial == "up":
                return np.array([1.0, 0.0], dtype=complex)
            if initial == "down":
                return np.array([0.0, 1.0], dtype=complex)
            if initial == "ground":
                w, U = np.linalg.eigh(self.h)
                return U[:, int(np.argmin(w))].astype(complex)
            raise ValueError(f"unknown initial state {initial!r}")
        v = np.asarray(initial, complex)
        return v / np.linalg.norm(v)

    def _run_tebd(self, dt, n_steps, bond_dim, trunc_eps, obs_ops, initial, kw):
        from fishbonett.backwardSpinBoson import SpinBoson as _SB
        from fishbonett.mps import SpinBosonMPS
        b = self.bath
        n = b.n_modes
        pd = [b.phys_dim] * n + [2]
        eth = _SB(pd)
        eth.domain = list(b.domain)
        eth.sd = b.spectral_density()
        eth.he_dy = self.coupling
        eth.h1e = self.h
        eth.build(g=1, ncap=kw.get("ncap", 20000), discretizer=b.discretizer())
        etn = SpinBosonMPS(pd)
        g = self._initial_spin(initial)
        etn.B[-1][:] = 0.0
        etn.B[-1][0, 0, 0], etn.B[-1][0, 1, 0] = g[0], g[1]

        rdms, maxb = [], []
        for tn in range(n_steps):
            t0 = 2 * tn * dt
            u1, _ = eth.get_u(t0, dt, mode="normal")
            etn.U = u1
            for j in range(n - 1, 0, -1):
                etn.update_bond(j, bond_dim, trunc_eps, swap=1)
            etn.update_bond(0, bond_dim, trunc_eps, swap=0)
            etn.update_bond(0, bond_dim, trunc_eps, swap=0)
            _, u2 = eth.get_u(t0 + dt, dt, mode="reverse")
            etn.U = u2
            for j in range(1, n):
                etn.update_bond(j, bond_dim, trunc_eps, swap=1)
            theta = etn.get_theta1(n)
            rho = np.einsum("LiR,LjR->ij", theta, theta.conj())
            rdms.append(rho / np.trace(rho).real)
            maxb.append(max((len(s) for s in etn.S), default=1))
        t = np.arange(1, n_steps + 1) * 2 * dt
        return Result(t=t, expect=self._expect_from_rdm(rdms, obs_ops),
                      max_bond=np.array(maxb), rdm=np.asarray(rdms), method="tebd")
