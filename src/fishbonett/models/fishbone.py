"""The multi-site models: several system sites, each carrying its own bath(s).

* ``site-tree`` (:class:`TreeFishbone`) -- sites wired into any loop-free tree;
* ``comb`` (:class:`Fishbone`) -- the same with a linear backbone, the fishbone.

The model only defines the physical topology. Static methods ask
:meth:`TreeFishbone.local_terms` for a Schrödinger representation, while the
interaction-chain comb methods are compiled separately. In either case the
Hamiltonian is the representation's business, not the model's:
:meth:`TreeFishbone.local_terms` asks
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
from fishbonett.bath.coupled import CoupledBath
from fishbonett.linalg import Truncation
from fishbonett.operators import sigma_x, sigma_z
from fishbonett.models.propagate import (
    RunCtx, _resolve_continuation, _resolve_sampling_options,
    resolve_time_grid,
)
from fishbonett.models.registry import SCHRODINGER_CHAIN_TREE_TEBD, resolve
from fishbonett.system import check_operator
from fishbonett.targets import BathMode

__all__ = [
    "TreeFishbone", "Fishbone", "BathMode", "SCHRODINGER_CHAIN_TREE_TEBD",
]


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


def _parse_observable(spec, dimensions=None, name="observable"):
    """Normalise an observable spec to ``(kind, operator, sites)``.

    A bare ``(d, d)`` operator is measured on every matching site (``kind
    "persite"``); ``(operator, i)`` or ``(operator, (i, j, ...))`` targets a
    specific site or a composite of sites (``kind "sites"``)."""
    if isinstance(spec, tuple):
        # A tuple-of-tuples is a perfectly ordinary matrix.  Only interpret a
        # two-tuple as a targeted observable when its entries are not scalar
        # matrix rows. Checking the row structure first avoids asking NumPy to
        # coerce ``(operator, target)`` as a ragged array.
        square_rows = bool(spec) and all(
            isinstance(row, (tuple, list, np.ndarray))
            and np.ndim(row) == 1
            and len(row) == len(spec)
            for row in spec
        )
        whole = np.asarray(spec) if square_rows else None
        if whole is not None and whole.ndim == 2 and whole.shape[0] == whole.shape[1]:
            kind, operator, sites = "persite", whole, None
            return _validate_observable_target(
                kind, operator, sites, dimensions, name
            )
        if len(spec) != 2:
            raise ValueError(
                "a targeted observable must be (operator, site) or "
                "(operator, sites)"
            )
        op, where = spec
        if isinstance(where, BathMode) or (
                isinstance(where, (int, np.integer))
                and not isinstance(where, (bool, np.bool_))):
            sites = [where]
        else:
            try:
                sites = list(where)
            except TypeError as exc:
                raise TypeError(
                    "observable targets must be system-site integers or BathMode "
                    "objects"
                ) from exc
        kind, operator = "sites", np.asarray(op)
        return _validate_observable_target(
            kind, operator, sites, dimensions, name
        )
    return _validate_observable_target(
        "persite", np.asarray(spec), None, dimensions, name
    )


def _validate_observable_target(kind, operator, sites, dimensions, name):
    """Validate one parsed observable against the electronic site dimensions."""
    operator = np.asarray(operator, complex)
    if (operator.ndim != 2 or operator.shape[0] == 0
            or operator.shape[0] != operator.shape[1]
            or not np.all(np.isfinite(operator))):
        raise ValueError(f"{name} must contain a finite square operator")
    if dimensions is None:
        return kind, operator, sites
    if kind == "persite":
        if not any(operator.shape == (d, d) for d in dimensions):
            raise ValueError(
                f"{name} has shape {operator.shape}, which matches no system site"
            )
        return kind, operator, sites
    if not sites:
        raise ValueError(f"{name} must target at least one site")
    normalized = []
    for target in sites:
        if isinstance(target, BathMode):
            normalized.append(target)
        elif (isinstance(target, (int, np.integer))
              and not isinstance(target, (bool, np.bool_))):
            normalized.append(int(target))
        else:
            raise TypeError(
                f"{name} targets must be system-site integers or BathMode objects"
            )
    sites = normalized
    if len(set(sites)) != len(sites):
        raise ValueError(f"{name} cannot target the same site more than once")
    system_sites = [
        target.system_site if isinstance(target, BathMode) else target
        for target in sites
    ]
    if any(site < 0 or site >= len(dimensions) for site in system_sites):
        raise ValueError(f"{name} targets a site outside the system")
    if any(isinstance(target, BathMode) for target in sites):
        return kind, operator, sites
    expected = int(np.prod([dimensions[site] for site in sites]))
    if operator.shape != (expected, expected):
        raise ValueError(
            f"{name} has shape {operator.shape}, expected {(expected, expected)} "
            f"for sites {sites}"
        )
    return kind, operator, sites


def _resolve_observable_target(parsed, dimensions, bath_nodes, name):
    """Resolve semantic targets to tensor nodes and validate the full shape."""
    kind, operator, targets = parsed
    if kind == "persite":
        return parsed
    nodes = []
    for target in targets:
        if isinstance(target, BathMode):
            try:
                node = bath_nodes[target]
            except KeyError as exc:
                raise ValueError(
                    f"{name} targets unavailable bath mode {target}"
                ) from exc
        else:
            node = target
        nodes.append(node)
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"{name} resolves to the same tensor node more than once")
    expected = int(np.prod([dimensions[node] for node in nodes]))
    if operator.shape != (expected, expected):
        raise ValueError(
            f"{name} has shape {operator.shape}, expected {(expected, expected)} "
            f"for resolved tensor nodes {nodes}"
        )
    return kind, operator, nodes


def _bind_site_entry(entry):
    """Normalize one site's explicitly bound bath entry."""
    if entry is None:
        return []
    values = [item for item in entry if item is not None] if isinstance(
        entry, (list, tuple)) else [entry]
    out = []
    for item in values:
        if not isinstance(item, CoupledBath):
            raise ValueError(
                "every multi-site bath must bind its coupling operator, for "
                "example bath.bind(operator)"
            )
        out.append(item)
    return out


def _reject_engine_options(options):
    """Multi-site planners currently expose no method-specific run options."""
    if options:
        names = ", ".join(sorted(map(str, options)))
        raise TypeError(f"unexpected run option(s): {names}")


def _run_multisite_model(
    physical_model, *, dt, t_max, n_steps, method, model,
    representation, state_geometry, integrator, trunc, bond_dim,
    trunc_eps, observables, initial, krylov, seed, resume, bath_horizon,
    progress, observe_every, svd_backend, engine_kw,
):
    """Shared public-run prelude for ``Fishbone`` and ``TreeFishbone``."""
    axis_kw = {
        "model": model,
        "representation": representation,
        "state_geometry": state_geometry,
        "integrator": integrator,
    }
    axes_given = any(value is not None for value in axis_kw.values())
    if method is None and not axes_given:
        method = SCHRODINGER_CHAIN_TREE_TEBD
    normalized_method = (
        None if method is None else method.lower().replace("_", "-")
    )
    spec = resolve(
        {physical_model._MODEL}, method=normalized_method, **axis_kw
    )
    _reject_engine_options(engine_kw)
    dt, n_steps = resolve_time_grid(dt, t_max=t_max, n_steps=n_steps)
    trunc = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
    observe_every, bath_horizon = _resolve_sampling_options(
        observe_every, bath_horizon
    )
    bath_horizon = _resolve_continuation(
        resume=resume, initial=initial, method=spec.name, dt=dt,
        n_steps=n_steps, bath_horizon=bath_horizon, supports_resume=True,
    )
    if observables is None:
        observables = ({"sz": sigma_z, "sx": sigma_x}
                       if all(d == 2 for d in physical_model.de) else {})
    context = RunCtx(
        dt=dt, n_steps=n_steps, bond_dim=trunc.max_bond,
        trunc_eps=trunc.eps, obs_ops=observables, initial=initial,
        krylov=krylov, seed=seed, svd_backend=svd_backend, resume=resume,
        bath_horizon=bath_horizon, observe_every=observe_every,
        progress=progress, kw=engine_kw,
    )
    from fishbonett.models.simulation import compile_plan
    return compile_plan(physical_model, spec, context).run()


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
        :class:`~fishbonett.bath.coupled.CoupledBath`, a list of coupled baths,
        or ``None``. Missing mapping keys mean no bath on that site. Create each
        coupled bath with ``bath.bind(operator)``. Baths may have different
        settings.
    """

    def __init__(self, sites, edges, baths):
        self.sites = [check_operator(h, f"sites[{i}]")
                      for i, h in enumerate(sites)]
        self.ns = len(self.sites)
        if self.ns == 0:
            raise ValueError("sites must contain at least one Hamiltonian")
        self.de = [h.shape[0] for h in self.sites]
        self.edges = []
        seen = set()
        for edge_index, e in enumerate(edges):
            if not isinstance(e, (tuple, list)) or len(e) not in (2, 3):
                raise ValueError(
                    "each edge must be (i, j) or (i, j, coupling)"
                )
            i, j = e[:2]
            if (not isinstance(i, (int, np.integer))
                    or isinstance(i, (bool, np.bool_))
                    or not isinstance(j, (int, np.integer))
                    or isinstance(j, (bool, np.bool_))):
                raise TypeError("edge endpoints must be integer site indices")
            i, j = int(i), int(j)
            if i < 0 or j < 0 or i >= self.ns or j >= self.ns:
                raise ValueError(f"edge {(i, j)} is outside the system")
            if i == j:
                raise ValueError(f"edge {(i, j)} is a self-edge")
            key = tuple(sorted((i, j)))
            if key in seen:
                raise ValueError(f"duplicate edge {key}")
            seen.add(key)
            if len(e) == 2:
                C = np.zeros((self.de[i] * self.de[j],) * 2, complex)
            else:
                C = e[2]
            C = check_operator(
                C, f"edges[{edge_index}].coupling", self.de[i] * self.de[j]
            )
            self.edges.append((i, j, C))
        if len(self.edges) != self.ns - 1:
            raise ValueError("edges must form a tree over the sites (n_sites-1 edges)")
        adjacency = [set() for _ in range(self.ns)]
        for i, j, _ in self.edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        reached = {0}
        frontier = [0]
        while frontier:
            frontier.extend(adjacency[frontier.pop()] - reached)
            reached.update(frontier)
        if len(reached) != self.ns:
            raise ValueError("edges must form one connected tree over the sites")
        self.baths = []
        for entry in _site_entries(baths, self.ns):
            self.baths.append(_bind_site_entry(entry))

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
        v = np.asarray(item, complex).reshape(-1)
        if v.shape != (de,):
            raise ValueError(
                f"initial state for site {i} has length {v.size}, expected {de}"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError(f"initial state for site {i} must be finite")
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(
                f"initial state for site {i} must have a finite non-zero norm"
            )
        return v / norm

    #: The model this class realizes.  ``Fishbone`` overrides it with ``"comb"``.
    _MODEL = "site-tree"

    def run(self, *, dt, t_max=None, n_steps=None,
            method=None, model=None, representation=None,
            state_geometry=None, integrator=None,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial=None, krylov=25, seed=0, resume=None, bath_horizon=None,
            progress=None, observe_every=1, svd_backend="auto", **engine_kw):
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
        * ``(operator, BathMode(...))`` -- an operator on a represented bath
          mode. ``BathMode`` may also be mixed with system-site integers in a
          composite target. Its ``mode`` index refers to the selected chain or
          star representation; operators are not transformed automatically.

        ``rdm`` holds the single-site reduced density matrices at each recorded
        time. ``observe_every`` records every Nth integration step (and always the
        final one) without changing the TEBD step. ``bath_horizon`` fixes the time
        used for automatic bath resolution; make it cover all continuation
        segments. A returned ``result.checkpoint`` resumes through ``resume=`` and
        is rejected if the resolved Hamiltonian changes.

        ``svd_backend`` is ``"auto"`` (exact small-block SVD plus certified
        adaptive randomized truncation), ``"exact"`` or ``"randomized"``. The
        randomized request retains exact fallbacks when its residual cannot be
        certified."""
        return _run_multisite_model(
            self, dt=dt, t_max=t_max, n_steps=n_steps, method=method,
            model=model, representation=representation,
            state_geometry=state_geometry, integrator=integrator, trunc=trunc,
            bond_dim=bond_dim, trunc_eps=trunc_eps,
            observables=observables, initial=initial, krylov=krylov, seed=seed,
            resume=resume, bath_horizon=bath_horizon, progress=progress,
            observe_every=observe_every, svd_backend=svd_backend,
            engine_kw=engine_kw,
        )

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
    (one bath -- possibly multichannel), a list of independent baths, or ``None``.
    Bind every bath explicitly with ``bath.bind(operator)``. ``run`` and the
    returned :class:`Result` are
    exactly those of :meth:`fishbonett.models.fishbone.TreeFishbone.run`.
    """

    #: The model this class realizes -- ``comb`` rather than ``site-tree``, so
    #: error messages name the class the user actually reached for. Both models
    #: use the site-tree propagation path.
    _MODEL = "comb"

    def __init__(self, sites, baths, backbone=None, *, couplings=None):
        self.sites = [check_operator(h, f"sites[{i}]")
                      for i, h in enumerate(sites)]
        self.nc = len(self.sites)
        if self.nc == 0:
            raise ValueError("sites must contain at least one Hamiltonian")
        self.de = [h.shape[0] for h in self.sites]
        self.baths = [
            _bind_site_entry(entry) for entry in _site_entries(baths, self.nc)
        ]
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
                expected = self.de[i] * self.de[j]
                value = check_operator(
                    operator, f"coupling {(i, j)}", expected
                )
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
        self.backbone = [
            check_operator(
                value, f"backbone[{i}]", self.de[i] * self.de[i + 1]
            )
            for i, value in enumerate(backbone)
        ]

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
        """Map a per-site bath specification to explicit coupled baths."""
        if not entry:
            return None
        if isinstance(entry, (list, tuple)) and all(
                isinstance(item, CoupledBath) for item in entry):
            return list(entry)
        return _bind_site_entry(entry)

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
        terms = self._tree().local_terms(t_max)
        if self.graph_couplings:
            terms.graph_bond.update(self.graph_couplings)
        return terms

    def hamiltonians(self, t_max=None):
        """``(dims, edges, site_H, edge_H)`` -- :meth:`local_terms` as a 4-tuple."""
        return self.local_terms(t_max).as_tuple()

    def run(self, *, dt, t_max=None, n_steps=None,
            method=None, model=None, representation=None,
            state_geometry=None, integrator=None,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial=None, krylov=25, seed=0, resume=None, bath_horizon=None,
            progress=None, observe_every=1, svd_backend="auto", **engine_kw):
        """Propagate the 1D fishbone through the shared simulation planner. See
        :meth:`fishbonett.models.fishbone.TreeFishbone.run` for the arguments, the
        observable spec and the per-site :class:`Result` layout.

        ``method`` is validated against the ``comb`` model before delegating, so
        an unsupported one names ``comb`` rather than ``site-tree``."""
        return _run_multisite_model(
            self, dt=dt, t_max=t_max, n_steps=n_steps, method=method,
            model=model, representation=representation,
            state_geometry=state_geometry, integrator=integrator, trunc=trunc,
            bond_dim=bond_dim, trunc_eps=trunc_eps,
            observables=observables, initial=initial, krylov=krylov, seed=seed,
            resume=resume, bath_horizon=bath_horizon, progress=progress,
            observe_every=observe_every, svd_backend=svd_backend,
            engine_kw=engine_kw,
        )
