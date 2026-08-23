"""The multi-site models: several system sites, each carrying its own bath(s).

* ``site-tree`` (:class:`TreeFishbone`) -- sites wired into any loop-free tree;
* ``comb`` (:class:`Fishbone`) -- the same with a linear backbone, the fishbone.

Both are Schroedinger-picture, and the Hamiltonian itself is the representation's business,
not the model's: :meth:`TreeFishbone.local_terms` asks
:func:`fishbonett.representations.schrodinger.terms` to turn the sites and baths into a
:class:`~fishbonett.representations.schrodinger.LocalTerms` graph -- bath frequencies on their own
nodes, couplings on the edges -- and the model only decides the topology and drives
the propagation.  The state it evolves is
:class:`fishbonett.states.tree.TreeTensorNetwork`, stepped by
:mod:`fishbonett.evolve.sitetree`.

Do not confuse the ``site-tree`` *model* with the ``binary-tree`` tensor-network
*state geometry*, where
a single system's bath modes are placed on a tree; see
:mod:`fishbonett.models.registry`.
"""
from collections.abc import Mapping

import numpy as np

from fishbonett.representations.schrodinger import terms as schrodinger_terms
from fishbonett.bath.coupled import CoupledBath, bind_bath
from fishbonett.linalg import Truncation
from fishbonett.operators import sigma_x, sigma_z
from fishbonett.models.propagate import RunCtx
from fishbonett.models.registry import (
    SCHRODINGER_CHAIN_TREE_TEBD, method_spec, methods_of,
    unknown_method_error, resolve,
)

__all__ = ["TreeFishbone", "Fishbone", "SCHRODINGER_CHAIN_TREE_TEBD"]


def _site_entries(baths, n_sites):
    """Return one bath specification per system site.

    Sequences retain the original positional interface.  A mapping makes sparse
    attachments explicit: omitted sites have no bath.
    """
    if isinstance(baths, Mapping):
        entries = [None] * n_sites
        for site, entry in baths.items():
            if (not isinstance(site, (int, np.integer))
                    or isinstance(site, (bool, np.bool_))):
                raise TypeError("bath mapping keys must be integer site indices")
            site = int(site)
            if site < 0 or site >= n_sites:
                raise ValueError(
                    f"bath mapping site {site} is outside the valid range "
                    f"0 <= site < {n_sites}")
            entries[site] = entry
        return entries

    entries = list(baths)
    if len(entries) != n_sites:
        raise ValueError("baths must have one entry per site")
    return entries


def _parse_observable(spec):
    """Normalise an observable spec to ``(kind, operator, sites)``.

    A bare ``(d, d)`` operator is measured on every matching site (``kind
    "persite"``); ``(operator, i)`` or ``(operator, (i, j, ...))`` targets a
    specific site or a composite of sites (``kind "sites"``)."""
    if isinstance(spec, tuple):
        op, where = spec
        if np.isscalar(where) or isinstance(where, (int, np.integer)):
            sites = [int(where)]
        else:
            sites = [int(s) for s in where]
        return "sites", np.asarray(op), sites
    return "persite", np.asarray(spec), None


class TreeFishbone:
    """Electronic sites wired into an *arbitrary tree*, each with one or more baths.

    Generalises :class:`fishbonett.models.fishbone.Fishbone` (a 1D chain) to any
    loop-free electronic topology.

    Parameters
    ----------
    sites : list of (d, d) array
        Electronic site Hamiltonians.
    edges : list of (i, j) or (i, j, C)
        Electronic-electronic couplings; the pairs must form a tree over the
        sites.  ``C`` is a ``(d_i*d_j, d_i*d_j)`` operator (default: none).
    baths : sequence or mapping
        A sequence with one entry per site, or a mapping from system-site index
        to bath entry.  Each entry is a single
        :class:`~fishbonett.bath.spec.Bath`, a list of baths, or ``None``.
        Missing mapping keys mean no bath on that site.  Prefer
        :class:`~fishbonett.bath.coupled.CoupledBath` entries made with
        ``bath.bind(operator)``.  A bare bath is bound to ``sigma_z`` for
        compatibility.  Baths may have different settings.
    """

    def __init__(self, sites, edges, baths):
        self.sites = [np.asarray(h, complex) for h in sites]
        self.ns = len(self.sites)
        self.de = [h.shape[0] for h in self.sites]
        self.edges = []
        for e in edges:
            if len(e) == 2:
                i, j = e
                C = np.zeros((self.de[i] * self.de[j],) * 2, complex)
            else:
                i, j, C = e
                C = np.asarray(C, complex)
            self.edges.append((int(i), int(j), C))
        if len(self.edges) != self.ns - 1:
            raise ValueError("edges must form a tree over the sites (n_sites-1 edges)")
        self.baths = []
        for entry in _site_entries(baths, self.ns):
            if entry is None:
                self.baths.append([])
            elif isinstance(entry, (list, tuple)):
                self.baths.append([
                    item if isinstance(item, CoupledBath)
                    else bind_bath(item, default_operator=sigma_z)
                    for item in entry if item is not None
                ])
            else:
                self.baths.append([
                    entry if isinstance(entry, CoupledBath)
                    else bind_bath(entry, default_operator=sigma_z)
                ])

    def local_terms(self, t_max=None):
        """The static Hamiltonian as a
        :class:`~fishbonett.representations.schrodinger.LocalTerms` graph.

        Delegates to :func:`fishbonett.representations.schrodinger.terms` -- these models are
        Schroedinger-picture, and the representation owns how a bath becomes nodes and edges.
        ``t_max`` sizes any bath whose ``n_modes`` is automatic.
        """
        return schrodinger_terms(self.sites, self.edges, self.baths, t_max)

    def hamiltonians(self, t_max=None):
        """``(dims, edges, site_H, edge_H)`` -- :meth:`local_terms` as a 4-tuple.

        Kept because it reads well in tests and reference implementations that want
        the raw arrays; :meth:`local_terms` is the structured form.
        """
        return self.local_terms(t_max).as_tuple()

    def _build(self, dt, t_max=None):
        """Physical tree plus the single-site and two-site Trotter gates.

        ``dt`` is the gate's own time argument, **not** the step: the symmetric step
        (:func:`fishbonett.evolve.sitetree.symmetric_tree_step`) applies every gate
        twice, so ``run`` passes half its step here -- the same convention as
        :func:`fishbonett.evolve.tebd.symmetric_static_step`.
        """
        lt = self.local_terms(t_max)
        site_gates, edge_gates = lt.tebd_gates(dt)
        return lt.dims, lt.edges, site_gates, edge_gates

    def _initial_vec(self, initial, i):
        de = self.de[i]
        if initial is None or (isinstance(initial, str) and initial == "up"):
            v = np.zeros(de, complex); v[0] = 1.0; return v
        if isinstance(initial, str) and initial == "down":
            v = np.zeros(de, complex); v[min(1, de - 1)] = 1.0; return v
        if isinstance(initial, str) and initial == "ground":
            w, U = np.linalg.eigh(self.sites[i])
            return U[:, int(np.argmin(w))].astype(complex)
        item = initial[i] if isinstance(initial, (list, tuple)) else initial
        v = np.asarray(item, complex)
        return v / np.linalg.norm(v)

    #: The model this class realizes.  ``Fishbone`` overrides it with ``"comb"``.
    _MODEL = "site-tree"

    def run(self, *, dt, t_max=None, n_steps=None,
            method=SCHRODINGER_CHAIN_TREE_TEBD,
            representation=None, state_geometry=None, integrator=None,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial=None, seed=None, resume=None, bath_horizon=None,
            observe_every=1):
        """Propagate and return a :class:`~fishbonett.models.result.Result`.

        ``method`` exists for symmetry with
        :meth:`fishbonett.models.system_bath.SystemBath.run`; the multi-site models have a
        single propagator, :data:`SCHRODINGER_CHAIN_TREE_TEBD`, so it is the only
        accepted value.  Asking for a single-system
        method here raises with a message saying which model owns it.  The
        representation gaps are recorded in :mod:`fishbonett.models.registry`.

        The step is second order in ``dt``
        (:func:`fishbonett.evolve.sitetree.symmetric_tree_step`), so halving ``dt``
        cuts the error by about four.

        Truncation is one setting, given either as a
        :class:`~fishbonett.linalg.Truncation` (``trunc=``) or as the loose
        ``trunc_eps`` / ``bond_dim`` keywords.  ``trunc_eps`` (default ``1e-4``)
        is the accuracy knob -- singular values below it are dropped -- and
        ``bond_dim`` is an optional hard cap, ``None`` meaning **unlimited**, so
        the bond grows to whatever ``trunc_eps`` requires.  ``result.max_bond``
        reports the peak bond actually used, per step.

        Each entry of ``observables`` is one of:

        * a bare ``(d, d)`` operator -- measured on **every** matching site;
          ``expect[name]`` is then ``(n_steps, n_sites)`` (NaN where the operator
          dimension does not match a site);
        * ``(operator, i)`` -- the operator on the single site ``i``;
        * ``(operator, (i, j, ...))`` -- a composite operator on those sites
          (``operator`` is ``(D, D)`` with ``D`` = product of the site dimensions
          in that order, e.g. a two-site correlation ``sigma_z (x) sigma_z``).
          For the last two forms ``expect[name]`` is ``(n_steps,)``.

        ``rdm`` holds the single-site reduced density matrices at each recorded
        time. ``observe_every`` records every Nth integration step (and always the
        final one) without changing the TEBD step. ``bath_horizon`` fixes the time
        used for automatic bath resolution; make it cover all continuation
        segments. A returned ``result.checkpoint`` resumes through ``resume=`` and
        is rejected if the resolved Hamiltonian changes."""
        axes_given = any(value is not None for value in
                         (representation, state_geometry, integrator))
        if axes_given:
            if method != SCHRODINGER_CHAIN_TREE_TEBD:
                raise ValueError("give either method= or representation/state_geometry/integrator")
            spec = resolve(
                {self._MODEL}, representation=representation,
                state_geometry=state_geometry, integrator=integrator)
            m = spec.name
        else:
            m = method.lower().replace("_", "-")
            if m not in methods_of(self._MODEL):
                raise unknown_method_error(m, self._MODEL)
            spec = method_spec(m, self._MODEL)
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        trunc = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
        if int(observe_every) != observe_every or observe_every < 1:
            raise ValueError("observe_every must be a positive integer")
        if resume is not None:
            from fishbonett.models.result import SimulationCheckpoint
            if not isinstance(resume, SimulationCheckpoint):
                raise TypeError("resume must be a SimulationCheckpoint")
            if initial is not None:
                raise ValueError("initial and resume cannot be supplied together")
            if resume.method != m:
                raise ValueError(
                    f"checkpoint method is {resume.method!r}, requested {m!r}")
            if bath_horizon is None:
                bath_horizon = resume.bath_horizon
            elif not np.isclose(bath_horizon, resume.bath_horizon):
                raise ValueError("bath_horizon cannot change when resuming")
            if resume.elapsed + n_steps * dt > bath_horizon + 1e-12:
                raise ValueError(
                    "continuation exceeds the checkpoint bath_horizon; rerun the "
                    "initial segment with a horizon covering the complete time")
        elif bath_horizon is None:
            bath_horizon = n_steps * dt
        elif bath_horizon + 1e-12 < n_steps * dt:
            raise ValueError("bath_horizon must cover the requested propagation")
        bond_dim, trunc_eps = trunc.max_bond, trunc.eps
        if observables is None:
            observables = {"sz": sigma_z, "sx": sigma_x} if all(
                d == 2 for d in self.de) else {}
        context = RunCtx(
            dt=dt, n_steps=n_steps, bond_dim=bond_dim,
            trunc_eps=trunc_eps, obs_ops=observables, initial=initial,
            seed=seed, resume=resume, bath_horizon=bath_horizon,
            observe_every=int(observe_every),
        )
        from fishbonett.models.simulation import compile_plan
        return compile_plan(self, spec, context).run()

# -- 1D fishbone: a specialization of TreeFishbone to a linear backbone -------
class Fishbone:
    """Electronic sites with independent baths on a comb tensor network.

    A convenience specialization of :class:`~fishbonett.models.fishbone.TreeFishbone`
    (which handles *any* loop-free electronic topology) to a **linear** tensor
    backbone. The electronic Hamiltonian may use either nearest-neighbour
    ``backbone`` terms or an arbitrary ``couplings={(i, j): operator}`` graph;
    the latter is applied by a reversible swap network. ``baths`` may be a
    sequence with one entry per site or a mapping from site index to entry;
    omitted mapping keys mean no bath.  Each entry is a single :class:`Bath`
    (one bath -- possibly multichannel), a ``(left, right)`` pair (two baths per
    site -- the fishbone), or ``None``.  Prefer explicit ``bath.bind(operator)``
    values.  For compatibility, a bare left bath defaults to ``sigma_z`` and a
    bare right bath to ``sigma_x``.  ``run`` and the returned :class:`Result` are
    exactly those of :meth:`fishbonett.models.fishbone.TreeFishbone.run`.
    """

    #: The model this class realizes -- ``comb`` rather than ``site-tree``, so
    #: error messages name the class the user actually reached for, even though
    #: the two share an engine.
    _MODEL = "comb"

    def __init__(self, sites, baths, backbone=None, *, couplings=None):
        self.sites = [np.asarray(h, complex) for h in sites]
        self.nc = len(self.sites)
        self.de = [h.shape[0] for h in self.sites]
        self.baths = _site_entries(baths, self.nc)
        if couplings is not None and backbone is not None:
            raise ValueError("provide either backbone or couplings, not both")
        self.graph_couplings = None
        if couplings is not None:
            if not isinstance(couplings, Mapping):
                raise TypeError("couplings must be a mapping {(i, j): operator}")
            parsed = {}
            for edge, operator in couplings.items():
                if (not isinstance(edge, tuple) or len(edge) != 2
                        or not all(isinstance(x, (int, np.integer))
                                   and not isinstance(x, (bool, np.bool_))
                                   for x in edge)):
                    raise TypeError("coupling keys must be integer pairs (i, j)")
                i, j = map(int, edge)
                if i >= j:
                    raise ValueError("coupling keys must use the canonical order i < j")
                if i < 0 or j >= self.nc:
                    raise ValueError(f"coupling edge {(i, j)} is outside the system")
                value = np.asarray(operator, complex)
                expected = self.de[i] * self.de[j]
                if value.shape != (expected, expected):
                    raise ValueError(
                        f"coupling {(i, j)} has shape {value.shape}, expected "
                        f"{(expected, expected)}")
                if not np.allclose(value, value.conj().T):
                    raise ValueError(f"coupling {(i, j)} must be Hermitian")
                parsed[(i, j)] = value
            if len(set(self.de)) > 1:
                raise ValueError(
                    "arbitrary-graph Fishbone currently requires equal system-site "
                    "dimensions for its swap network")
            self.graph_couplings = parsed
        if backbone is None:
            backbone = [np.zeros((self.de[i] * self.de[i + 1],) * 2, complex)
                        for i in range(self.nc - 1)]
        if len(backbone) != max(self.nc - 1, 0):
            raise ValueError("backbone must have n_sites - 1 entries")
        self.backbone = [np.asarray(b, complex) for b in backbone]

    @classmethod
    def from_single_excitation(cls, exciton_hamiltonian, *, baths):
        """Build local two-level sites from a one-excitation Hamiltonian.

        ``H[i, i]`` becomes the energy of ``|1>`` on site ``i`` and
        ``H[i, j]`` becomes excitation hopping between sites ``i`` and ``j``.
        The resulting many-site Hamiltonian conserves excitation number and its
        projection onto the one-excitation sector is exactly the supplied matrix.
        """
        hamiltonian = np.asarray(exciton_hamiltonian, complex)
        if (hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]
                or hamiltonian.shape[0] == 0):
            raise ValueError("exciton_hamiltonian must be a non-empty square matrix")
        if not np.allclose(hamiltonian, hamiltonian.conj().T):
            raise ValueError("exciton_hamiltonian must be Hermitian")
        occupied = np.diag([0.0, 1.0]).astype(complex)
        sites = [hamiltonian[i, i].real * occupied
                 for i in range(len(hamiltonian))]
        couplings = {}
        for i in range(len(hamiltonian)):
            for j in range(i + 1, len(hamiltonian)):
                value = hamiltonian[i, j]
                if abs(value) == 0:
                    continue
                term = np.zeros((4, 4), complex)
                term[2, 1] = value
                term[1, 2] = np.conj(value)
                couplings[(i, j)] = term
        return cls(sites=sites, baths=baths, couplings=couplings)

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
                out.append(b if isinstance(b, CoupledBath) else bind_bath(
                    b, default_operator=(sigma_z if pos == 0 else sigma_x)))
            return out
        return (entry if isinstance(entry, CoupledBath)
                else bind_bath(entry, default_operator=sigma_z))

    def _tree(self):
        edges = [(i, i + 1, self.backbone[i]) for i in range(self.nc - 1)]
        return TreeFishbone(sites=self.sites, edges=edges,
                            baths=[self._site_baths(b) for b in self.baths])

    def local_terms(self, t_max=None):
        """The static Hamiltonian as
        :class:`~fishbonett.representations.schrodinger.LocalTerms`.

        Same as :meth:`fishbonett.models.fishbone.TreeFishbone.local_terms`, with the
        linear backbone expanded into the equivalent edge list.
        """
        return self._tree().local_terms(t_max)

    def hamiltonians(self, t_max=None):
        """``(dims, edges, site_H, edge_H)`` -- :meth:`local_terms` as a 4-tuple."""
        return self.local_terms(t_max).as_tuple()

    def run(self, *, dt, t_max=None, n_steps=None,
            method=SCHRODINGER_CHAIN_TREE_TEBD,
            representation=None, state_geometry=None, integrator=None,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial=None, seed=None, resume=None, bath_horizon=None,
            observe_every=1):
        """Propagate the 1D fishbone through the shared simulation planner. See
        :meth:`fishbonett.models.fishbone.TreeFishbone.run` for the arguments, the
        observable spec and the per-site :class:`Result` layout.

        ``method`` is validated against the ``comb`` model before delegating, so
        an unsupported one names ``comb`` rather than ``site-tree``."""
        axes_given = any(value is not None for value in
                         (representation, state_geometry, integrator))
        if axes_given:
            if method != SCHRODINGER_CHAIN_TREE_TEBD:
                raise ValueError(
                    "give either method= or representation/state_geometry/integrator")
            spec = resolve(
                {self._MODEL}, representation=representation,
                state_geometry=state_geometry, integrator=integrator)
            m = spec.name
        else:
            m = method.lower().replace("_", "-")
            if m not in methods_of(self._MODEL):
                raise unknown_method_error(m, self._MODEL)
            spec = method_spec(m, self._MODEL)
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        trunc = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
        if int(observe_every) != observe_every or observe_every < 1:
            raise ValueError("observe_every must be a positive integer")
        if resume is not None:
            from fishbonett.models.result import SimulationCheckpoint
            if not isinstance(resume, SimulationCheckpoint):
                raise TypeError("resume must be a SimulationCheckpoint")
            if initial is not None:
                raise ValueError("initial and resume cannot be supplied together")
            if resume.method != m:
                raise ValueError(
                    f"checkpoint method is {resume.method!r}, requested {m!r}")
            if bath_horizon is None:
                bath_horizon = resume.bath_horizon
            elif not np.isclose(bath_horizon, resume.bath_horizon):
                raise ValueError("bath_horizon cannot change when resuming")
            if resume.elapsed + n_steps * dt > bath_horizon + 1e-12:
                raise ValueError(
                    "continuation exceeds the checkpoint bath_horizon; rerun the "
                    "initial segment with a horizon covering the complete time")
        elif bath_horizon is None:
            bath_horizon = n_steps * dt
        elif bath_horizon + 1e-12 < n_steps * dt:
            raise ValueError("bath_horizon must cover the requested propagation")
        if observables is None:
            observables = ({"sz": sigma_z, "sx": sigma_x}
                           if all(d == 2 for d in self.de) else {})
        context = RunCtx(
            dt=dt, n_steps=n_steps, bond_dim=trunc.max_bond,
            trunc_eps=trunc.eps, obs_ops=observables, initial=initial,
            seed=seed, resume=resume, bath_horizon=bath_horizon,
            observe_every=int(observe_every),
        )
        from fishbonett.models.simulation import compile_plan
        return compile_plan(self, spec, context).run()
