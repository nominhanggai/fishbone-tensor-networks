"""The taxonomy: the four axes a run is made of, and which combinations exist.

A run is **four independent choices**, and the point of this module is that they
are independent -- which they were not when three of them were mashed into one
string called the "model"::

    model     what is coupled to what      system-bath | multichannel | comb | site-tree
    representation  how H is written         schrodinger-chain | schrodinger-star
                                             | interaction-chain | interaction-star
                                             | polaron-chain | polaron-star
    geometry  the graph the state lives on path | binary-tree | comb-tree
    integrator  how a step is taken        tebd | tdvp1 | tdvp2 | dtdvp | trotter-mpo

.. rubric:: Six complete representations

Each representation name is complete.  There is no secondary public taxonomy to
combine with it.  All six describe valid Hamiltonians; availability for a model
is recorded by :attr:`Model.gaps`.

* ``interaction-chain`` **exists**, and is what this package actually runs for
  ``interaction-chain-*`` methods.  ``H_B`` is
  obtained by taking the interaction representation of the discretized star bath
  and then applying its star-to-chain transformation.  Its coupling ``|d_n(t)|``
  starts as ``(|V|, 0, ..., 0)`` and spreads outward with ``t``.
* ``polaron-star`` **exists** too: the textbook Lang-Firsov transform is *defined*
  per star mode, ``prod_k D_k(g_k sigma_z / w_k)``.  It is the chain version that
  uses the ``J/w^2`` transformation to localize that displacement on ``c0``.

The star-to-chain transform relates each star/chain pair without creating another
user-facing axis.  The physics is identical while tensor-network costs may differ.

``geometry`` stays a separate axis because it genuinely is one:
``interaction-chain-tdvp1`` and ``interaction-chain-tree-tdvp1`` are the *same
representation*, laid on a path and on a balanced binary tree, which is why they
produce identical numbers rather than merely close ones.
``mode-tree`` used to be listed as a model for that difference; it was never a
model.

:data:`METHODS` is therefore the single source of truth for the taxonomy and
method selection: each row carries its four axes plus the engine that realizes it.
After that lookup, :mod:`fishbonett.models.simulation` compiles the engine into one
prepared plan; the physical model contains no private dispatch tables.

.. note::
   The name ``fishbonett.models`` was used once before, for what is now
   :mod:`fishbonett.representations`.  If you are reading commits from before
   that rename, ``models/`` there means the Hamiltonian builders, not this.

Propagator-level gaps, finer than this table records: the Schroedinger chain could
be driven by TEBD gates but is not (only MPO/TDVP is wired); the
conditional-displacement MPO of ``interaction-chain-trotter-mpo`` exists only in
the interaction representation, because outside it the coupling does not commute
with the free-bath term.
"""
from dataclasses import dataclass, field
from typing import Mapping, Tuple

__all__ = ["Model", "Representation", "Method", "MODELS", "REPRESENTATIONS", "METHODS",
           "GEOMETRIES", "APPLICATIONS", "FIXED_BOND_METHODS",
           "why_not", "models_of", "representations_of",
           "methods_of", "all_methods", "model", "method_spec", "resolve",
           "combinations", "METHOD_REPRESENTATIONS",
           "methods_by_representation", "representation_label", "describe_taxonomy",
           "unknown_method_error", "SCHRODINGER_CHAIN_TREE_TEBD",
           "SCHRODINGER_STAR_TREE_TEBD", "INTERACTION_CHAIN_TEBD",
           "INTERACTION_STAR_TEBD"]


# -- representations ------------------------------------------------------------------
@dataclass(frozen=True)
class Representation:
    """One complete mathematical representation of the Hamiltonian."""
    key: str
    label: str
    blurb: str
    static: bool
    #: Whether the represented Hamiltonian contains no mode-mode terms.
    mode_decoupled: bool


REPRESENTATIONS = {
    "schrodinger-chain": Representation(
        "schrodinger-chain", "Schrodinger chain representation",
        "Nothing rotated out, bath chain-mapped.  H is time-independent and its MPO "
        "is built once, so TDVP conserves energy -- but the state carries the full "
        "system-bath correlation, giving the largest bond dimensions.  The chain's "
        "nearest-neighbour hoppings are what an MPS is good at, and the system "
        "touches only c0.",
        static=True, mode_decoupled=False),
    "schrodinger-star": Representation(
        "schrodinger-star", "Schrodinger star representation",
        "Nothing rotated out, no chain mapping: every mode couples straight to the "
        "system.  No mode-mode terms, but no locality for the MPS to exploit "
        "either.  Static, so still one MPO built once.",
        static=True, mode_decoupled=True),
    "interaction-chain": Representation(
        "interaction-chain", "interaction chain representation",
        "A finite star is put in the interaction representation with respect to "
        "its free Hamiltonian, then its time-dependent coupling is transformed "
        "star-to-chain.  What is lost is locality, not existence -- the coupling "
        "d_n(t) starts concentrated on c0 "
        "at t=0 and spreads outward, so this is 'no longer a chain' in the only "
        "sense that matters to an MPS.  Entanglement is much smaller than the "
        "Schrodinger representation's, but H is time-dependent, so gates/MPOs are "
        "rebuilt "
        "every step.  All the coupling terms commute here, which is what makes the "
        "exact conditional-displacement propagator possible.",
        static=False, mode_decoupled=True),
    "interaction-star": Representation(
        "interaction-star", "interaction star representation",
        "The same rotation as interaction-chain, left in the star modes instead of "
        "being rotated back: the coupling of mode k is simply V_k e^{-i w_k t}.  "
        "Reaches the same trajectory through a completely different coupling "
        "vector, so it is an independent check on the chain route rather than a "
        "restatement of it.  Which is cheaper is not settled -- the guess that the "
        "chain wins (its coupling starts on c0 and spreads) is not what measuring "
        "shows.  The multichannel model exposes both forms by applying one common "
        "orthogonal transform to its matrix-valued star couplings.",
        static=False, mode_decoupled=True),
    "polaron-chain": Representation(
        "polaron-chain", "polaron chain representation",
        "The static part of the coupling is absorbed into a bath displacement, "
        "leaving a free chain plus a dressed tunneling term.  Static like the "
        "Schrodinger representation *and* low-entanglement like the interaction "
        "representation; needs int J/w^2 finite.  Populations are "
        "representation-invariant, "
        "coherences must be un-dressed.  The J/w^2 reweighting is what localizes "
        "the displacement on c0.",
        static=True, mode_decoupled=False),
    "polaron-star": Representation(
        "polaron-star", "polaron star representation",
        "The textbook Lang-Firsov transform, which is *defined* per star mode: "
        "prod_k D_k(g_k sigma_z / w_k).  Perfectly well defined -- it is the chain "
        "version that uses the J/w^2 transformation to localize the collective "
        "displacement onto c0.",
        static=True, mode_decoupled=True),
}

#: The graph the state's tensors live on.  Independent of the representation:
#: the same interaction-chain Hamiltonian runs on a path
#: (``interaction-chain-tdvp1``) and a balanced tree
#: (``interaction-chain-tree-tdvp1``), which is why ``mode-tree`` was never a
#: model.
GEOMETRIES = {
    "path": "an MPS: system at site 0, modes 1..N in a line.",
    "binary-tree": "a balanced binary TTN with the system at the root, keeping the "
                   "high-bond region O(log N) edges deep instead of O(N).",
    "comb-tree": "a tree of system sites, each carrying its own bath chain(s) as a "
                 "branch -- the comb / fishbone geometry.",
}


#: Why the binary tree needs a representation with no mode-mode terms
#: (:attr:`Representation.mode_decoupled`).  Not an oversight: the tree is worth
#: having *because* nothing
#: couples mode to mode, so every mode hangs off the system independently and the
#: only question is how deep the bonds are.
#:
#: Note this is about the *representation*, not the structure --
#: ``interaction-chain`` qualifies (it rotates ``H_B`` away entirely) and is
#: what the ``tree-*`` methods use.
_NO_MODE_MODE = (
    "the balanced binary tree pays off only when there are no mode-mode terms.  "
    "This representation keeps the chain hoppings, which are nearest-neighbour "
    "on a path but "
    "long-range on that tree (only half of the chain-adjacent pairs are "
    "tree-adjacent; the rest span up to 2*log2(N) edges -- measured: 10 edges at "
    "N=32), so the MPO bond grows and the tree loses to the plain path.  Reordering "
    "the leaves to make the chain local just turns the tree back into a path.")


# -- models ------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """A physical setup -- **only** the topology, now that the mode structure lives in
    the representation and the state graph in ``geometry``.

    ``gaps`` maps an absent representation key to the reason it is absent. All six
    names describe valid Hamiltonians; gaps record model-specific implementation
    work, not impossible combinations.
    """
    key: str
    label: str
    blurb: str
    cls: str                              # the class a user instantiates
    gaps: Mapping[str, str] = field(default_factory=dict)
    #: ``"method"`` -- chosen by ``run(method=...)``; ``"coupling"`` -- chosen
    #: automatically from a list of model coupling operators (multichannel).
    selected_by: str = "method"

    @property
    def representations(self):
        """``{representation key: method names}``, in taxonomy order."""
        out = {}
        for representation_key in REPRESENTATIONS:
            names = tuple(name for name, spec in METHODS.items()
                          if spec.representation == representation_key and self.key in spec.models)
            if names:
                out[representation_key] = names
        return out

    def methods(self):
        """Every method name this model offers, across all its representations."""
        return tuple(m for fr in self.representations.values() for m in fr)


# -- methods: the one dispatch table -----------------------------------------
@dataclass(frozen=True)
class Method:
    """One realizable representation/geometry/integrator combination.

    This is the single source of truth for what exists and which engine realizes it.
    ``models`` lists the compatible physical models because one method can serve
    several topologies -- static tree TEBD runs the comb and the site-tree, while
    interaction-chain TEBD serves scalar and multichannel system-bath models.
    """
    name: str
    representation: str
    models: Tuple[str, ...]
    #: which plan-compiler group in :mod:`fishbonett.models.simulation` realizes
    #: it.  This is an implementation key, not an axis.
    engine: str
    #: the entry point in :mod:`fishbonett.evolve`, where there is a single one
    driver: str = ""
    #: 1-site TDVP cannot grow a bond and adaptive tangent expansion needs a ceiling, so
    #: ``bond_dim=None`` ("unlimited") is not meaningful for these.
    fixed_bond: bool = False
    #: The integrator **axis** -- ``"tebd"``, ``"tdvp1"``, ``"tdvp2"``, ``"dtdvp"``,
    #: ``"trotter-mpo"``.  Unique within a ``(model, representation, geometry)``, which is
    #: what lets ``run`` be called by the axes instead of by a name that mashes
    #: them together.
    integrator: str = ""
    #: The state graph -- see :data:`GEOMETRIES`.
    geometry: str = "path"

    @property
    def application(self):
        """How H's **interaction graph** is realized on the state's graph.

        Derived, not declared: it is a *consequence* of the other axes.  A representation can
        make ``H`` non-local relative to the state -- the interaction transformation couples
        every mode to the system, a star, while the state is a path -- and this
        records what pays for that.  See :data:`APPLICATIONS`.

        Keys on :attr:`Representation.mode_decoupled`, **not** on the structure:
        ``interaction-chain-tebd`` still needs a swap network, because it is rotating
        out ``H_B`` that spreads the coupling over every mode, not the choice of
        modes to write it in.
        """
        if self.integrator != "tebd" or self.geometry == "binary-tree":
            return "operator"
        if self.geometry == "path" and REPRESENTATIONS[self.representation].mode_decoupled:
            return "swap"
        return "local"


#: The values :attr:`Method.application` takes.
#:
#: This is the axis the package had no name for, which is why three methods each
#: re-derived a swap network and nothing recorded that they did.  It is not the same
#: question as the representation: the *representation* decides which terms exist, the application
#: decides what it costs to apply them to a state whose graph is shaped differently.
APPLICATIONS = {
    "local": "interaction edges are state edges -- gates apply in place",
    "swap": "a star realized on a path: the system site is walked past every mode "
            "and back each step, so a step costs O(N) swaps on top of the gates",
    "operator": "no gate layout -- a single low-bond operator (an MPO on a path, "
                "a bond-K tree operator on a tree) carries the long-range terms "
                "natively, so the interaction graph never has to match the state's",
}


def _canonical_method_name(representation, integrator, geometry="path"):
    """Derive a method name from its representation and algorithm.

    Path methods use ``<representation>-<integrator>``.  A non-path tensor tree
    inserts ``tree`` so methods that share a representation and integrator remain
    unambiguous.  Keeping this rule here prevents a registry label from drifting
    away from the tuple it denotes.
    """
    geometry_tag = "" if geometry == "path" else "tree-"
    return f"{representation}-{geometry_tag}{integrator}"


def _m(representation, models, engine, driver="", fixed_bond=False,
       integrator="", geometry="path"):
    integrator = integrator or driver
    name = _canonical_method_name(representation, integrator, geometry)
    return Method(name, representation, models, engine, driver, fixed_bond,
                  integrator, geometry)


# Public method constants for model defaults and programmatic callers.
SCHRODINGER_CHAIN_TREE_TEBD = _canonical_method_name(
    "schrodinger-chain", "tebd", "comb-tree")
SCHRODINGER_STAR_TREE_TEBD = _canonical_method_name(
    "schrodinger-star", "tebd", "comb-tree")
INTERACTION_CHAIN_TEBD = _canonical_method_name(
    "interaction-chain", "tebd")
INTERACTION_STAR_TEBD = _canonical_method_name(
    "interaction-star", "tebd")


# Removed spellings from before method names stated their full representation.
# They are error-message hints only; they are deliberately not accepted aliases.
_RENAMED_METHODS = {
    "mpo-tdvp1": "schrodinger-chain-tdvp1",
    "mpo-tdvp2": "schrodinger-chain-tdvp2",
    "mpo-dtdvp": "schrodinger-chain-dtdvp",
    "mpo-star-tdvp1": "schrodinger-star-tdvp1",
    "mpo-star-tdvp2": "schrodinger-star-tdvp2",
    "tebd": "interaction-chain-tebd",
    "trotter-mpo": "interaction-chain-trotter-mpo",
    "mpo-ip-tdvp1": "interaction-chain-tdvp1",
    "mpo-ip-tdvp2": "interaction-chain-tdvp2",
    "mpo-ip-star-tdvp1": "interaction-star-tdvp1",
    "mpo-ip-star-tdvp2": "interaction-star-tdvp2",
    "tree-tdvp": "interaction-chain-tree-tdvp1",
    "tree-tdvp2": "interaction-chain-tree-tdvp2",
    "tree-tebd": "interaction-chain-tree-tebd",
    "polaron": "polaron-chain-tebd",
    "polaron-tdvp1": "polaron-chain-tdvp1",
    "polaron-tdvp2": "polaron-chain-tdvp2",
    "polaron-dtdvp": "polaron-chain-dtdvp",
    "tree-tebd-static": "schrodinger-chain-tree-tebd",
    "multichannel-static": "schrodinger-star-tree-tebd",
    "multichannel-ip": "interaction-chain-tebd",
    "multichannel-ip-star": "interaction-star-tebd",
}


_SB = ("system-bath",)

#: Every method, declared once.  Order matters: :attr:`Model.representations` reads method
#: order off this table, and ``methods_of(model, representation)[0]`` is the default a
#: bath-selected model falls back to.
_METHOD_ROWS = [
    # -- system-bath, static Schroedinger representations --------------------
    _m("schrodinger-chain", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("schrodinger-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("schrodinger-chain", _SB, "mpo-tdvp",
       "dtdvp", fixed_bond=True),
    _m("schrodinger-star", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("schrodinger-star", _SB, "mpo-tdvp", "tdvp2"),
    # -- system-bath, interaction transformation, on a path --------------------------
    # `interaction-chain`, not `-star`: the star-to-chain transform rotates the phases
    # back into the chain modes, so d_n(0) = (|V|, 0, ..., 0) -- the system on c0
    # alone -- and spreads outward with t.  Measured, not assumed.
    _m("interaction-chain", ("system-bath", "multichannel"), "swap-tebd",
       integrator="tebd"),
    _m("interaction-chain", _SB, "displacement-mpo",
       integrator="trotter-mpo"),
    _m("interaction-chain", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("interaction-chain", _SB, "mpo-tdvp", "tdvp2"),
    # -- ...the same rotation left in the star modes --------------------------
    _m("interaction-star", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("interaction-star", _SB, "mpo-tdvp", "tdvp2"),
    # -- ...and the chain representation on a balanced binary tree ---------------------
    _m("interaction-chain", _SB, "modetree",
       "run_tree_tdvp", integrator="tdvp1",
       geometry="binary-tree"),
    _m("interaction-chain", _SB, "modetree",
       "run_tree_tdvp2", integrator="tdvp2", geometry="binary-tree"),
    _m("interaction-chain", _SB, "modetree",
       "run_tree_tebd", integrator="tebd", geometry="binary-tree"),
    # -- system-bath, polaron representation -------------------------------------------
    _m("polaron-chain", _SB, "polaron-tebd", integrator="tebd"),
    _m("polaron-chain", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("polaron-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("polaron-chain", _SB, "mpo-tdvp",
       "dtdvp", fixed_bond=True),
    _m("polaron-star", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("polaron-star", _SB, "mpo-tdvp", "tdvp2"),
    _m("polaron-star", _SB, "mpo-tdvp",
       "dtdvp", fixed_bond=True),
    # -- the static tree engine: one engine, two representations, three topologies -----
    _m("schrodinger-chain", ("comb", "site-tree"),
       "static-tree-tebd", integrator="tebd", geometry="comb-tree"),
    _m("schrodinger-star", ("multichannel",),
       "static-tree-tebd", integrator="tebd", geometry="comb-tree"),
    _m("interaction-star", ("multichannel",),
       "swap-tebd", integrator="tebd"),
]

_method_names = [spec.name for spec in _METHOD_ROWS]
_duplicate_method_names = {
    name for name in _method_names if _method_names.count(name) > 1
}
if _duplicate_method_names:
    raise RuntimeError(
        "method rows have duplicate canonical names: "
        + ", ".join(sorted(_duplicate_method_names)))
METHODS = {spec.name: spec for spec in _METHOD_ROWS}

#: Derived from :data:`METHODS` -- was a hand-maintained set in
#: ``models/system_bath.py`` that the tests had to import privately.
FIXED_BOND_METHODS = frozenset(n for n, s in METHODS.items() if s.fixed_bond)


#: The multi-site models are wired for one representation only.  Their baths are chain-mapped
#: per site, and nothing rotates a transformation out per site.
_MULTISITE = "not implemented for multi-site models."

#: Static scalar-chain mappings do not directly preserve several cross-correlated
#: coupling operators.  Interaction-chain is different: after rotating out the
#: free bath, any orthogonal star-to-chain transform remains exact and retains the
#: full matrix-valued coupling at every transformed mode.
_NO_STATIC_CHAIN = (
    "not implemented for cross-correlated shared channels: a scalar static chain "
    "cannot localize several independent coupling operators at once.")

_NO_MULTICHANNEL_POLARON = (
    "not implemented for several coupling operators: the current Lang-Firsov "
    "representation requires one Hermitian generator.")

MODELS = {
    "system-bath": Model(
        key="system-bath", label="system-bath",
        blurb="One system site coupled to one bath through one coupling operator.  "
              "The most developed model: all six representations, both single-system "
              "geometries and the whole integrator family.  What used to be three "
              "separate 'models' (chain, star, mode-tree) is this one model in the "
              "schrodinger-chain, schrodinger-star and interaction-chain representations, the "
              "last on two geometries.",
        cls="SystemBath"),
    "multichannel": Model(
        key="multichannel", label="multichannel system-bath",
        blurb="One system site, one bath, coupled through *several* operators "
              "that share the same modes -- so the channels are cross-correlated, "
              "unlike independent baths.  Selected by giving SystemBath a list "
              "of coupling operators, not by a method name.",
        cls="SystemBath",
        gaps={"schrodinger-chain": _NO_STATIC_CHAIN,
              "polaron-chain": _NO_MULTICHANNEL_POLARON,
              "polaron-star": _NO_MULTICHANNEL_POLARON},
        selected_by="coupling"),
    "comb": Model(
        key="comb", label="fishbone / comb",
        blurb="Several system sites on a 1D backbone, each carrying one or two "
              "baths -- the fishbone.  A specialization of ``site-tree`` to a "
              "linear backbone.",
        cls="Fishbone",
        gaps={
            "schrodinger-star": _MULTISITE,
            "interaction-chain": _MULTISITE,
            "interaction-star": _MULTISITE,
            "polaron-chain": _MULTISITE,
            "polaron-star": _MULTISITE,
        }),
    "site-tree": Model(
        key="site-tree", label="tree of sites",
        blurb="Several system sites wired into any loop-free tree, each carrying "
              "zero or more baths.  The most general geometry.  Distinct from a "
              "``binary-tree`` geometry, where it is a *single* system's bath modes "
              "that form the tree.",
        cls="TreeFishbone",
        gaps={
            "schrodinger-star": _MULTISITE,
            "interaction-chain": _MULTISITE,
            "interaction-star": _MULTISITE,
            "polaron-chain": _MULTISITE,
            "polaron-star": _MULTISITE,
        }),
}


# -- constraints -------------------------------------------------------------
def why_not(model_key, representation=None, *, geometry=None):
    """Why a combination is unavailable, or ``None`` if it exists.

    Every one of the six representations is a real representation, so nothing here says "impossible".
    What it reports is per-model work nobody has done (:attr:`Model.gaps`, which
    says *why* it would or would not pay), plus the one genuine constraint: a
    balanced binary tree needs a representation with no mode-mode terms.
    """
    if geometry == "binary-tree" and representation is not None \
            and representation in REPRESENTATIONS and not REPRESENTATIONS[representation].mode_decoupled:
        return _NO_MODE_MODE
    if geometry is not None and model_key in ("comb", "site-tree") \
            and geometry != "comb-tree":
        return (f"{MODELS[model_key].label} puts each site's bath on its own "
                f"branch, so its state graph is always 'comb-tree'.")
    if representation is not None:
        m = MODELS.get(model_key)
        if m is not None and representation not in m.representations:
            return m.gaps.get(representation)
    return None


def method_spec(name, model_key=None):
    """The :class:`Method` for ``name``, or a :class:`ValueError` naming what is."""
    spec = METHODS.get(name)
    if spec is None or (model_key is not None and model_key not in spec.models):
        raise unknown_method_error(name, model_key)
    return spec


def combinations(model_keys):
    """``[(model, representation, geometry, integrator, method)]`` for these models."""
    keys = set(model_keys)
    return [(mk, s.representation, s.geometry, s.integrator, s.name)
            for s in METHODS.values() for mk in s.models if mk in keys]


#: The axes :func:`resolve` filters on, in the order they appear in a combination
#: tuple.  Named once so the filter, the error message and the table agree.
_AXES = ("model", "representation", "geometry", "integrator")


def resolve(model_keys, *, method=None, **axes):
    """The :class:`Method` selected by either spelling.

    ``method=`` names a representation/geometry/integrator combination; the
    physical object supplies the compatible model.  The four axes can also be
    given directly.  Method names begin with their exact representation, so
    ``"interaction-chain-tdvp2"`` selects ``(interaction-chain, path, tdvp2)``
    for a system-bath model.  This is one lookup either way, and mixing the two
    spellings is rejected rather than silently resolved.

    Representation names are explicit: use ``interaction-chain`` rather than a
    partial name such as ``interaction``.
    """
    unknown = set(axes) - set(_AXES)
    if unknown:
        raise TypeError(f"unknown axis {sorted(unknown)}; the axes are "
                        f"{', '.join(_AXES)}")
    given = {k: v for k, v in axes.items() if v is not None}
    avail = combinations(model_keys)
    if method is not None:
        if not given:
            return method_spec(method)
        raise ValueError(
            "give either method= or the axes (model / representation / geometry / "
            "integrator), not both -- a method name already fixes the "
            "representation, geometry and integrator, while the object selects "
            "a compatible model.")

    def matches(c):
        for k, v in given.items():
            if c[_AXES.index(k)] != v:
                return False
        return True

    hit = [c for c in avail if matches(c)]
    asked = ", ".join(f"{k}={given[k]!r}" for k in _AXES if k in given)
    if not hit:
        # An unnameable representation or a recorded gap has a reason; say it rather than
        # printing the table and leaving the reader to spot what is missing.
        for mk in sorted(set(model_keys)):
            why = why_not(given.get("model", mk), given.get("representation"),
                          geometry=given.get("geometry"))
            if why:
                raise ValueError(f"no method for {asked}: {why}")
        raise ValueError(
            f"no method for {asked}.  Available here:\n" + _combo_table(avail))
    if len({c[-1] for c in hit}) > 1:
        raise ValueError(
            f"{asked} is ambiguous -- it matches "
            f"{', '.join(sorted({c[-1] for c in hit}))}.  Add the missing axis:\n"
            + _combo_table(hit))
    return METHODS[hit[0][-1]]


def _combo_table(combos):
    rows = sorted(set(combos))
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(_AXES))]
    return "\n".join(
        "    " + "  ".join(f"{a}={str(c[i]):<{widths[i]}}"
                           for i, a in enumerate(_AXES))
        + f"  (method={c[-1]!r})" for c in rows)


# -- lookups -----------------------------------------------------------------
def model(key):
    """The :class:`Model` for ``key``, with a helpful error if it is unknown."""
    try:
        return MODELS[key]
    except KeyError:
        extra = ""
        if key in ("chain", "star"):
            extra = (f"  ({key!r} is half of a *representation*, not a model -- the representations "
                     f"are {', '.join(k for k in REPRESENTATIONS if k.endswith(key))}.)")
        elif key == "mode-tree":
            extra = ("  ('mode-tree' was a state *geometry*, not a model -- use "
                     "model='system-bath', geometry='binary-tree'.)")
        raise KeyError(f"unknown model {key!r}; available: "
                       f"{', '.join(sorted(MODELS))}.{extra}") from None


def models_of(method):
    """Every model key offering ``method``.

    A tuple rather than a single key because
    ``schrodinger-chain-tree-tebd`` genuinely serves both multi-site models --
    the same propagator, different topologies.
    """
    return tuple(k for k, m in MODELS.items() if method in m.methods())


def representations_of(model_key):
    """Representation keys available for a model, in taxonomy order."""
    m = model(model_key)
    return tuple(k for k in REPRESENTATIONS if k in m.representations)


def methods_of(model_key, representation=None):
    """Methods for a model, optionally narrowed to one representation."""
    m = model(model_key)
    if representation is None:
        return m.methods()
    if representation not in m.representations:
        why = why_not(model_key, representation)
        raise KeyError(
            f"model {model_key!r} has no {representation!r} representation"
            + (f": {why}" if why else "")
            + f"  (available: {', '.join(representations_of(model_key))})")
    return tuple(m.representations[representation])


def all_methods():
    """Every method name in the taxonomy."""
    return tuple(sorted({m for mo in MODELS.values() for m in mo.methods()}))


#: Lossless ``method -> representation`` projection.  A method can serve more
#: than one model; use :func:`models_of` for that independent relationship.
METHOD_REPRESENTATIONS = {
    method: spec.representation for method, spec in METHODS.items()
}


def methods_by_representation():
    """``{(representation, model): [methods]}`` -- the taxonomy as a flat grouping."""
    out = {}
    for method, spec in sorted(METHODS.items()):
        for model_key in spec.models:
            out.setdefault((spec.representation, model_key), []).append(method)
    return out


def representation_label(key):
    """Human-readable name of a ``(representation, model)`` pair or a bare representation key."""
    if isinstance(key, tuple):
        representation_key, model_key = key
        return f"{REPRESENTATIONS[representation_key].label} / {MODELS[model_key].label}"
    return REPRESENTATIONS[key].label


def describe_taxonomy():
    """The whole taxonomy as readable text -- what ``run`` prints when asked for
    an unknown method, and what the docs table is generated from."""
    lines = []
    representation_width = max(map(len, REPRESENTATIONS))
    method_width = max(map(len, METHODS))
    for key, m in MODELS.items():
        picked = ("" if m.selected_by == "method" else
                  f"  (selected by {m.selected_by})")
        lines.append(f"{m.label}  [{key}] via {m.cls}{picked}")
        for representation_key in representations_of(key):
            for name in m.representations[representation_key]:
                s = METHODS[name]
                lines.append(f"    {representation_key:<{representation_width}} "
                             f"{name:<{method_width}} "
                             f"geometry={s.geometry:<11} "
                             f"integrator={s.integrator}")
        for representation_key, why in m.gaps.items():
            lines.append(
                f"    {representation_key:<{representation_width}} "
                f"{'-- absent':<{method_width}} {why}")
    return "\n".join(lines)


def unknown_method_error(method, model_key=None):
    """A :class:`ValueError` naming what *is* available.

    When ``model_key`` is given and ``method`` belongs to a different model, say
    so explicitly -- the common mistake is asking a multi-site model for a
    single-system method.
    """
    replacement = _RENAMED_METHODS.get(method)
    if replacement is not None:
        return ValueError(
            f"method {method!r} was renamed to {replacement!r} so its exact "
            "Hamiltonian representation is explicit")

    owners = models_of(method)
    if owners and model_key is not None and model_key not in owners:
        return ValueError(
            f"method {method!r} belongs to "
            f"{' / '.join(MODELS[o].label for o in owners)}, not to "
            f"{MODELS[model_key].label}.  "
            f"{MODELS[model_key].label} offers: "
            f"{', '.join(methods_of(model_key))}.")
    if model_key is not None:
        return ValueError(
            f"unknown method {method!r} for {MODELS[model_key].label}.  "
            f"Available: {', '.join(methods_of(model_key))}.")
    return ValueError(f"unknown method {method!r}.  Available, by model:\n"
                      f"{describe_taxonomy()}")
