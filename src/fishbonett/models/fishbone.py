"""The multi-site models: several system sites, each carrying its own bath(s).

* ``site-tree`` (:class:`TreeFishbone`) -- sites wired into any loop-free tree;
* ``comb`` (:class:`Fishbone`) -- the same with a linear backbone, the fishbone.

Both are Schroedinger-picture, and the Hamiltonian itself is the frame's business,
not the model's: :meth:`TreeFishbone.local_terms` asks
:func:`fishbonett.frames.schrodinger.terms` to turn the sites and baths into a
:class:`~fishbonett.frames.terms.LocalTerms` graph -- bath frequencies on their own
nodes, couplings on the edges -- and the model only decides the topology and drives
the propagation.  The state it evolves is
:class:`fishbonett.states.tree.TreeTensorNetwork`, stepped by
:mod:`fishbonett.evolve.sitetree`.

Do not confuse the ``site-tree`` *model* with the ``binary-tree`` *geometry*, where
a single system's bath modes are placed on a tree; see
:mod:`fishbonett.models.registry`.
"""
from dataclasses import replace

import numpy as np

from fishbonett.frames.schrodinger import terms as schrodinger_terms
from fishbonett.bath.coupled import CoupledBath
from fishbonett.linalg import Truncation
from fishbonett.operators import sigma_x, sigma_z
from fishbonett.states.tree import TreeTensorNetwork
from fishbonett.models.propagate import tree_peak_bond
from fishbonett.models.result import Result
from fishbonett.models.registry import (
    STATIC_TREE_TEBD, methods_of, unknown_method_error,
)

__all__ = ["TreeFishbone", "Fishbone", "STATIC_TREE_TEBD"]


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
    baths : list
        One entry per site: a single :class:`~fishbonett.bath.spec.Bath`, a list of
        baths, or ``None``.  Each bath carries its own ``coupling`` operator
        (default ``sigma_z``).  Baths may have different domains/discretizations.
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
        for entry in baths:
            if entry is None:
                self.baths.append([])
            elif isinstance(entry, (list, tuple)):
                self.baths.append(list(entry))
            else:
                self.baths.append([entry])
        if len(self.baths) != self.ns:
            raise ValueError("baths must have one entry per site")

    def local_terms(self, t_max=None):
        """The static Hamiltonian as a
        :class:`~fishbonett.frames.terms.LocalTerms` graph.

        Delegates to :func:`fishbonett.frames.schrodinger.terms` -- these models are
        Schroedinger-picture, and the frame owns how a bath becomes nodes and edges.
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
        site_gates, edge_gates = lt.gates(dt)
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

    def run(self, *, dt, t_max=None, n_steps=None, method=STATIC_TREE_TEBD,
            trunc=None, bond_dim=None, trunc_eps=None, observables=None,
            initial="up"):
        """Propagate and return a :class:`~fishbonett.models.result.Result`.

        ``method`` exists for symmetry with
        :meth:`fishbonett.models.system_bath.SystemBath.run`; the multi-site models have a
        single propagator, :data:`STATIC_TREE_TEBD` (Schroedinger-picture tree
        TEBD), so it is the only accepted value.  Asking for a single-system
        method here raises with a message saying which model owns it.  The
        frame gaps are recorded in :mod:`fishbonett.models.registry`.

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

        ``rdm`` holds the single-site reduced density matrices per step."""
        m = method.lower().replace("_", "-")
        if m not in methods_of(self._MODEL):
            raise unknown_method_error(m, self._MODEL)
        if n_steps is None:
            if t_max is None:
                raise ValueError("provide either t_max or n_steps")
            n_steps = int(round(t_max / dt))
        trunc = Truncation.resolve(trunc, eps=trunc_eps, max_bond=bond_dim)
        bond_dim, trunc_eps = trunc.max_bond, trunc.eps
        if observables is None:
            observables = {"sz": sigma_z, "sx": sigma_x} if all(
                d == 2 for d in self.de) else {}
        parsed = [(name, _parse_observable(spec))
                  for name, spec in observables.items()]
        # half-step gates: the symmetric step applies each of them twice
        dims, edges, site_gates, edge_gates = self._build(dt / 2.0, n_steps * dt)
        st = TreeTensorNetwork(dims, edges, root=0)
        for i in range(self.ns):
            st.set_physical(i, self._initial_vec(initial, i))

        expect = {name: (np.full((n_steps, self.ns), np.nan) if kind == "persite"
                         else np.full(n_steps, np.nan))
                  for name, (kind, _O, _s) in parsed}
        rdms = np.empty((n_steps, self.ns), dtype=object)
        max_bond = np.empty(n_steps, dtype=int)
        for tn in range(n_steps):
            st.step(site_gates, edge_gates, bond_dim, trunc_eps)
            for i in range(self.ns):
                rdms[tn, i] = st.rdm(i)
            for name, (kind, O, sites) in parsed:
                if kind == "persite":
                    for i in range(self.ns):
                        if O.shape == (self.de[i], self.de[i]):
                            expect[name][tn, i] = np.trace(rdms[tn, i] @ O).real
                else:
                    expect[name][tn] = st.expectation(O, sites)
            # peak bond over every edge of the tree, so the truncation reporting
            # matches what the single-system models give
            max_bond[tn] = tree_peak_bond(st)
            st.move_oc_to(0)
        if len(set(self.de)) == 1:                        # uniform sites -> dense
            rdm = np.array([[rdms[tn, i] for i in range(self.ns)]
                            for tn in range(n_steps)])
        else:                                             # mixed dims -> object array
            rdm = rdms
        t = np.arange(1, n_steps + 1) * dt
        return Result(t=t, expect=expect, rdm=rdm, max_bond=max_bond, method=m,
                      meta={"n_sites": self.ns})

# -- 1D fishbone: a specialization of TreeFishbone to a linear backbone -------
class Fishbone:
    """A 1D chain of electronic sites, each coupled to one or two baths.

    A convenience specialization of :class:`~fishbonett.models.fishbone.TreeFishbone`
    (which handles *any* loop-free electronic topology) to a **linear** backbone:
    site ``i`` is joined to site ``i+1`` by ``backbone[i]``.  Each ``baths`` entry
    is a single :class:`Bath` (one bath -- may be multichannel), a ``(left, right)``
    pair (two baths per site -- the fishbone), or ``None``.  A left bath defaults
    to a ``sigma_z`` coupling and a right bath to ``sigma_x`` when the :class:`Bath`
    itself sets none.  ``run`` and the returned :class:`Result` are exactly those
    of :meth:`fishbonett.models.fishbone.TreeFishbone.run`.
    """

    #: The model this class realizes -- ``comb`` rather than ``site-tree``, so
    #: error messages name the class the user actually reached for, even though
    #: the two share an engine.
    _MODEL = "comb"

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
                if isinstance(b, CoupledBath):
                    out.append(b)
                    continue
                if b.coupling is None:
                    b = replace(b, coupling=(sigma_z if pos == 0 else sigma_x))
                out.append(b)
            return out
        return entry

    def _tree(self):
        edges = [(i, i + 1, self.backbone[i]) for i in range(self.nc - 1)]
        return TreeFishbone(sites=self.sites, edges=edges,
                            baths=[self._site_baths(b) for b in self.baths])

    def local_terms(self, t_max=None):
        """The static Hamiltonian as :class:`~fishbonett.frames.terms.LocalTerms`.

        Same as :meth:`fishbonett.models.fishbone.TreeFishbone.local_terms`, with the
        linear backbone expanded into the equivalent edge list.
        """
        return self._tree().local_terms(t_max)

    def hamiltonians(self, t_max=None):
        """``(dims, edges, site_H, edge_H)`` -- :meth:`local_terms` as a 4-tuple."""
        return self.local_terms(t_max).as_tuple()

    def run(self, *, method=STATIC_TREE_TEBD, **kwargs):
        """Propagate the 1D fishbone (delegates to the general tree engine).  See
        :meth:`fishbonett.models.fishbone.TreeFishbone.run` for the arguments, the
        observable spec and the per-site :class:`Result` layout.

        ``method`` is validated against the ``comb`` model before delegating, so
        an unsupported one names ``comb`` rather than ``site-tree``."""
        m = method.lower().replace("_", "-")
        if m not in methods_of(self._MODEL):
            raise unknown_method_error(m, self._MODEL)
        return self._tree().run(method=m, **kwargs)
