"""Registry of supported model, representation, state, and integrator combinations.

A run is specified by four axes::

    model     what is coupled to what      system-bath | multichannel | exciton-bath
                                             | comb | site-tree
    representation  how H is written         schrodinger-chain | schrodinger-star
                                             | interaction-chain
                                             | polaron-chain | polaron-star
    state_geometry  tensor-network state        mps | system-first-mps
                                             | interleaved-mps | multi-set-mps
                                             | multi-set-tree | binary-tree | tree
    integrator  how a step is taken        tebd | tdvp1 | tdvp2 | dtdvp | trotter-mpo

Availability of each Hamiltonian representation for a model is recorded by
:attr:`Model.gaps`.

* ``interaction-chain`` takes the interaction representation of the discretized
  star bath and then applies the star-to-chain transformation. Its coupling
  ``|d_n(t)|`` starts as ``(|V|, 0, ..., 0)`` and spreads outward with ``t``.
* ``polaron-star`` applies the Lang-Firsov transformation per star mode,
  ``prod_k D_k(g_k sigma_z / w_k)``. The chain form uses the ``J/w^2``
  transformation to localize the displacement on ``c0``.

The star-to-chain transform relates each star/chain pair. The physics is
identical while tensor-network costs may differ.

``interaction-chain-tdvp1`` and ``interaction-chain-tree-tebd`` use the same
representation on an MPS and a binary tree tensor network, respectively.

:data:`METHODS` records each supported combination and its implementation engine.
:mod:`fishbonett.models.simulation` compiles a selected row into a prepared plan.

The Schrödinger-chain representation uses static MPO/TDVP propagation. The
conditional-displacement MPO of ``interaction-chain-trotter-mpo`` exists only in
the interaction representation, because outside it the coupling does not commute
with the free-bath term.
"""
from dataclasses import dataclass, field
from typing import Mapping, Tuple

__all__ = ["Model", "RepresentationSpec", "MethodSpec", "MODELS", "REPRESENTATIONS", "METHODS",
           "STATE_GEOMETRIES", "BOND_CAP_REQUIRED_METHODS",
           "why_not", "models_of", "representations_of",
           "methods_of", "all_methods", "model", "method_spec", "resolve",
           "combinations", "METHOD_REPRESENTATIONS",
           "methods_by_representation", "representation_label", "describe_taxonomy",
           "unknown_method_error", "SCHRODINGER_CHAIN_TREE_TEBD",
           "SCHRODINGER_STAR_TREE_TEBD", "INTERACTION_CHAIN_TEBD"]


# -- representations ------------------------------------------------------------------
@dataclass(frozen=True)
class RepresentationSpec:
    """One mathematical representation of the Hamiltonian."""
    key: str
    label: str
    blurb: str
    static: bool
    #: Whether the represented Hamiltonian contains no mode-mode terms.
    mode_decoupled: bool


REPRESENTATIONS = {
    "schrodinger-chain": RepresentationSpec(
        "schrodinger-chain", "Schrodinger chain representation",
        "The finite bath is mapped from star to chain in the Schrodinger picture. "
        "The Hamiltonian is static and nearest-neighbour on an MPS, with the "
        "system coupled only to the first chain mode.",
        static=True, mode_decoupled=False),
    "schrodinger-star": RepresentationSpec(
        "schrodinger-star", "Schrodinger star representation",
        "The finite star remains in the Schrodinger picture. Every bath mode "
        "couples directly to the system, the modes do not couple to one another, "
        "and the Hamiltonian is static.",
        static=True, mode_decoupled=True),
    "interaction-chain": RepresentationSpec(
        "interaction-chain", "interaction-picture chain representation",
        "The bath is discretized as a finite star and put in the interaction "
        "picture with respect to the free star bath. The resulting time-dependent "
        "star coupling is transformed to chain coordinates. The coefficients "
        "d_n(t) start on the first chain coordinate and spread with time; no "
        "mode-mode Hamiltonian terms remain.",
        static=False, mode_decoupled=True),
    "polaron-chain": RepresentationSpec(
        "polaron-chain", "polaron chain representation",
        "A Lang-Firsov transformation absorbs the diagonal coupling into a chain "
        "displacement and dresses the system tunnelling. The representation is "
        "static and requires a finite integral of J(w)/w^2. Physical "
        "coherences require the inverse observable transformation.",
        static=True, mode_decoupled=False),
    "polaron-star": RepresentationSpec(
        "polaron-star", "polaron star representation",
        "A Lang-Firsov transformation displaces each finite-star mode by its "
        "coupling divided by its frequency. The representation is static and "
        "requires a finite integral of J(w)/w^2.",
        static=True, mode_decoupled=True),
}

#: The tensor-network geometry of the state. Independent of the representation:
#: the same interaction-chain Hamiltonian runs on a 1D MPS
#: (``interaction-chain-tdvp1``) and a balanced tree
#: (``interaction-chain-tree-tebd``).
STATE_GEOMETRIES = {
    "mps": "a one-dimensional matrix product state, with the system at site 0",
    "system-first-mps": (
        "one multilevel electronic site followed by grouped bath modes"
    ),
    "interleaved-mps": (
        "local electronic sites interleaved with their grouped bath modes"
    ),
    "multi-set-mps": (
        "one independent environment MPS per exact system-basis state"
    ),
    "multi-set-tree": (
        "one independent environment tree tensor network per system-basis state"
    ),
    "binary-tree": "a balanced binary tree tensor network with the system at the root",
    "tree": "a tree tensor network determined by the multi-site model",
}


#: A binary-tree tensor network requires a represented Hamiltonian without
#: mode-mode terms. ``interaction-chain`` qualifies because the free bath has
#: been removed in the interaction picture.
_NO_MODE_MODE = (
    "the binary-tree geometry requires a representation without mode-mode terms")


# -- models ------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """A physical setup containing only its coupling topology.

    ``gaps`` maps an absent representation key to the reason it is absent. The
    names describe public Hamiltonian representations; gaps record
    model-specific implementation work, not impossible combinations.
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
class MethodSpec:
    """One realizable representation/state-geometry/integrator combination.

    ``models`` lists the compatible physical models. One method can serve several
    topologies: static tree TEBD runs the comb and site-tree, while
    interaction-chain TEBD serves scalar and multichannel system-bath models.
    """
    name: str
    representation: str
    models: Tuple[str, ...]
    #: which plan-compiler group in :mod:`fishbonett.models.simulation` realizes
    #: it.  This is an implementation key, not an axis.
    engine: str
    #: Entry point in :mod:`fishbonett.evolve`.
    driver: str = ""
    #: 1-site TDVP cannot grow a bond and adaptive tangent expansion needs a ceiling, so
    #: ``bond_dim=None`` ("unlimited") is not meaningful for these.
    requires_bond_cap: bool = False
    #: The integrator **axis** -- ``"tebd"``, ``"tdvp1"``, ``"tdvp2"``, ``"dtdvp"``,
    #: ``"trotter-mpo"``. Unique within a
    #: ``(model, representation, state_geometry)``, which is
    #: what lets ``run`` be called by the axes instead of by a name that mashes
    #: them together.
    integrator: str = ""
    #: The tensor-network state graph -- see :data:`STATE_GEOMETRIES`.
    state_geometry: str = "mps"

def _canonical_method_name(representation, integrator, state_geometry="mps"):
    """Derive a method name from its representation and algorithm.

    Conventional MPS methods use ``<representation>-<integrator>``. Other
    layouts insert the tag recorded below so methods that share a representation
    and integrator remain unambiguous.
    """
    tags = {
        "mps": "",
        "system-first-mps": "system-first-",
        "interleaved-mps": "interleaved-",
        "multi-set-mps": "multi-set-",
        "multi-set-tree": "multi-set-tree-",
        "binary-tree": "tree-",
        "tree": "tree-",
    }
    try:
        state_geometry_tag = tags[state_geometry]
    except KeyError:
        raise ValueError(
            f"unknown state geometry {state_geometry!r}") from None
    return f"{representation}-{state_geometry_tag}{integrator}"


def _m(representation, models, engine, driver="", requires_bond_cap=False,
       integrator="", state_geometry="mps", qualifier=None):
    integrator = integrator or driver
    if qualifier is None:
        name = _canonical_method_name(representation, integrator, state_geometry)
    else:
        name = f"{representation}-{qualifier}-{integrator}"
    return MethodSpec(name, representation, models, engine, driver, requires_bond_cap,
                  integrator, state_geometry)


# Public method constants for model defaults and programmatic callers.
SCHRODINGER_CHAIN_TREE_TEBD = _canonical_method_name(
    "schrodinger-chain", "tebd", "tree")
SCHRODINGER_STAR_TREE_TEBD = _canonical_method_name(
    "schrodinger-star", "tebd", "tree")
INTERACTION_CHAIN_TEBD = _canonical_method_name(
    "interaction-chain", "tebd")


_SB = ("system-bath",)

#: Every method, declared once.  Order matters: :attr:`Model.representations` reads method
#: order off this table, and ``methods_of(model, representation)[0]`` is the default a
#: bath-selected model falls back to.
_METHOD_ROWS = [
    # -- system-bath, static Schroedinger representations --------------------
    _m("schrodinger-chain", _SB, "mpo-tdvp",
       "tdvp1", requires_bond_cap=True),
    _m("schrodinger-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("schrodinger-chain", _SB, "mpo-tdvp",
       "dtdvp", requires_bond_cap=True),
    _m("schrodinger-star", _SB, "mpo-tdvp",
       "tdvp1", requires_bond_cap=True),
    _m("schrodinger-star", _SB, "mpo-tdvp", "tdvp2"),
    _m("schrodinger-chain", _SB, "multiset-tdvp", "tdvp2",
       state_geometry="multi-set-mps"),
    _m("schrodinger-star", _SB, "multiset-tdvp", "tdvp2",
       state_geometry="multi-set-mps"),
    # -- system-bath, interaction transformation, on a 1D MPS -------------------------
    # `interaction-chain`, not `-star`: the star-to-chain transform rotates the phases
    # back into the chain modes, so d_n(0) = (|V|, 0, ..., 0) -- the system on c0
    # alone -- and spreads outward with t.
    _m("interaction-chain", ("system-bath", "multichannel"), "swap-tebd",
       integrator="tebd"),
    _m("interaction-chain", _SB, "displacement-mpo",
       integrator="trotter-mpo"),
    _m("interaction-chain", _SB, "mpo-tdvp",
       "tdvp1", requires_bond_cap=True),
    _m("interaction-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("interaction-chain", (*_SB, "exciton-bath"), "multiset-tdvp", "tdvp2",
       state_geometry="multi-set-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mps-tebd",
       integrator="tebd", state_geometry="system-first-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mps-trotter",
       integrator="trotter-mpo", state_geometry="system-first-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mpo-tdvp", "tdvp1",
       requires_bond_cap=True, state_geometry="system-first-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mpo-tdvp", "tdvp2",
       state_geometry="system-first-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mpo-tdvp", "dtdvp",
       requires_bond_cap=True, state_geometry="system-first-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mps-tebd",
       integrator="tebd", state_geometry="interleaved-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mps-trotter",
       integrator="trotter-mpo", state_geometry="interleaved-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mpo-tdvp", "tdvp1",
       requires_bond_cap=True, state_geometry="interleaved-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mpo-tdvp", "tdvp2",
       state_geometry="interleaved-mps"),
    _m("interaction-chain", ("exciton-bath",), "exciton-mpo-tdvp", "dtdvp",
       requires_bond_cap=True, state_geometry="interleaved-mps"),
    _m("interaction-chain", ("exciton-bath",), "multiset-tree-tdvp", "tdvp2",
       state_geometry="multi-set-tree"),
    # -- ...and the chain representation on a balanced binary tree ---------------------
    _m("interaction-chain", _SB, "modetree",
       "run_tree_tebd", integrator="tebd", state_geometry="binary-tree"),
    # -- system-bath, polaron representation -------------------------------------------
    _m("polaron-chain", _SB, "polaron-tebd", integrator="tebd"),
    _m("polaron-chain", _SB, "mpo-tdvp",
       "tdvp1", requires_bond_cap=True),
    _m("polaron-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("polaron-chain", _SB, "mpo-tdvp",
       "dtdvp", requires_bond_cap=True),
    _m("polaron-star", _SB, "mpo-tdvp",
       "tdvp1", requires_bond_cap=True),
    _m("polaron-star", _SB, "mpo-tdvp", "tdvp2"),
    _m("polaron-star", _SB, "mpo-tdvp",
       "dtdvp", requires_bond_cap=True),
    _m("polaron-chain", _SB, "multiset-tdvp", "tdvp2",
       state_geometry="multi-set-mps"),
    _m("polaron-star", _SB, "multiset-tdvp", "tdvp2",
       state_geometry="multi-set-mps"),
    # -- the static tree engine: one engine, two representations, three topologies -----
    _m("schrodinger-chain", ("comb", "site-tree"),
       "static-tree-tebd", integrator="tebd", state_geometry="tree"),
    _m("polaron-chain", ("comb",), "polaron-fishbone",
       integrator="tebd", state_geometry="tree"),
    _m("interaction-chain", ("comb",), "interaction-fishbone",
       integrator="tebd", state_geometry="tree", qualifier="fishbone"),
    # The same per-branch H(t), carried by one conditional-displacement operator
    # instead of a swap network. Branches on different sites commute; several
    # operators sharing one site are composed symmetrically by the planner.
    _m("interaction-chain", ("comb",), "interaction-fishbone",
       integrator="trotter-mpo", state_geometry="tree", qualifier="fishbone"),
    # ...and the same H(t) again, propagated with the *generator* projected onto
    # the two-site tangent space rather than with the interval propagator.
    _m("interaction-chain", ("comb",), "interaction-fishbone",
       integrator="tdvp2", state_geometry="tree", qualifier="fishbone"),
    _m("schrodinger-star", ("multichannel",),
       "static-tree-tebd", integrator="tebd", state_geometry="tree"),
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

#: Methods whose integrator requires a finite bond-dimension ceiling.
BOND_CAP_REQUIRED_METHODS = frozenset(
    name for name, spec in METHODS.items() if spec.requires_bond_cap
)


#: This multi-site representation has no compiler for the selected model.
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

_EXCITON_INTERACTION_ONLY = (
    "not implemented for this exciton layout; independent local baths are "
    "currently assembled in the interaction-chain representation."
)

MODELS = {
    "system-bath": Model(
        key="system-bath", label="system-bath",
        blurb="One system site coupled to one bath through one coupling operator. "
              "All five representations support conventional and multi-set MPSs; "
              "interaction-chain also supports a binary-tree tensor network. "
              "Available integrators depend on the representation and state geometry.",
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
    "exciton-bath": Model(
        key="exciton-bath", label="single-excitation system with local baths",
        blurb="One N-level electronic Hamiltonian with an independent bath on "
              "each site population. It supports a multilevel system-first MPS, "
              "local electronic sites interleaved with their baths, and a "
              "multi-set MPS or bath tree.",
        cls="ExcitonBath",
        gaps={
            "schrodinger-chain": _EXCITON_INTERACTION_ONLY,
            "schrodinger-star": _EXCITON_INTERACTION_ONLY,
            "polaron-chain": _EXCITON_INTERACTION_ONLY,
            "polaron-star": _EXCITON_INTERACTION_ONLY,
        }),
    "comb": Model(
        key="comb", label="fishbone / comb",
        blurb="Several system sites on a 1D backbone, each carrying one or two "
              "baths -- the fishbone.  A specialization of ``site-tree`` to a "
              "linear backbone.",
        cls="Fishbone",
        gaps={
            "schrodinger-star": _MULTISITE,
            "polaron-star": _MULTISITE,
        }),
    "site-tree": Model(
        key="site-tree", label="tree of sites",
        blurb="Several system sites wired into any loop-free tree, each carrying "
              "zero or more baths. The most general tensor-network geometry. "
              "Distinct from a ``binary-tree`` state geometry, where it is a "
              "*single* system's bath modes "
              "that form the tree.",
        cls="TreeFishbone",
        gaps={
            "schrodinger-star": _MULTISITE,
            "interaction-chain": _MULTISITE,
            "polaron-chain": _MULTISITE,
            "polaron-star": _MULTISITE,
        }),
}


# -- constraints -------------------------------------------------------------
def why_not(model_key, representation=None, *, state_geometry=None):
    """Why a combination is unavailable, or ``None`` if it exists.

    Reports unimplemented model combinations (:attr:`Model.gaps`) and the
    structural constraint that a balanced binary tree needs a representation
    with no mode-mode terms.
    """
    selected_model = model(model_key)
    if representation is not None and representation not in REPRESENTATIONS:
        raise KeyError(
            f"unknown representation {representation!r}; available: "
            f"{', '.join(REPRESENTATIONS)}"
        )
    known_geometries = {spec.state_geometry for spec in METHODS.values()}
    if state_geometry is not None and state_geometry not in known_geometries:
        raise KeyError(
            f"unknown state_geometry {state_geometry!r}; available: "
            f"{', '.join(sorted(known_geometries))}"
        )
    if (state_geometry == "binary-tree" and representation is not None
            and representation in REPRESENTATIONS
            and not REPRESENTATIONS[representation].mode_decoupled):
        return _NO_MODE_MODE
    if state_geometry is not None and model_key in ("comb", "site-tree") \
            and state_geometry != "tree":
        return (f"{MODELS[model_key].label} puts each site's bath on its own "
                f"branch, so its state_geometry is always 'tree'.")
    if representation is not None:
        if representation not in selected_model.representations:
            return selected_model.gaps.get(representation)
    return None


def method_spec(name, model_key=None):
    """The :class:`MethodSpec` for ``name``, or an explanatory error."""
    spec = METHODS.get(name)
    if spec is None or (model_key is not None and model_key not in spec.models):
        raise unknown_method_error(name, model_key)
    return spec


def combinations(model_keys):
    """Return ``(model, representation, state_geometry, integrator, method)`` rows."""
    keys = set(model_keys)
    return [(mk, s.representation, s.state_geometry, s.integrator, s.name)
            for s in METHODS.values() for mk in s.models if mk in keys]


#: The axes :func:`resolve` filters on, in the order they appear in a combination
#: tuple.  Named once so the filter, the error message and the table agree.
_AXES = ("model", "representation", "state_geometry", "integrator")


def resolve(model_keys, *, method=None, **axes):
    """The :class:`MethodSpec` selected by either spelling.

    ``method=`` names a representation/state-geometry/integrator combination; the
    physical object supplies the compatible model.  The four axes can also be
    given directly.  Method names begin with their exact representation, so
    ``"interaction-chain-tdvp2"`` selects ``(interaction-chain, mps, tdvp2)``
    for a system-bath model. Mixing ``method=`` with individual axes is an
    error.

    Representation names are explicit: use ``interaction-chain`` rather than a
    partial name such as ``interaction``.
    """
    model_keys = set(model_keys)
    if not model_keys:
        raise ValueError("model_keys must contain at least one model")
    unknown_models = model_keys - set(MODELS)
    if unknown_models:
        raise ValueError(f"unknown model key(s): {', '.join(sorted(unknown_models))}")
    unknown = set(axes) - set(_AXES)
    if unknown:
        raise TypeError(f"unknown axis {sorted(unknown)}; the axes are "
                        f"{', '.join(_AXES)}")
    given = {k: v for k, v in axes.items() if v is not None}
    representation = given.get("representation")
    if representation is not None and representation not in REPRESENTATIONS:
        raise ValueError(
            f"no method for representation={representation!r}; available "
            f"representations: {', '.join(REPRESENTATIONS)}"
        )
    known_geometries = {spec.state_geometry for spec in METHODS.values()}
    requested_geometry = given.get("state_geometry")
    if requested_geometry is not None and requested_geometry not in known_geometries:
        raise ValueError(
            f"no method for state_geometry={requested_geometry!r}; available "
            f"state geometries: {', '.join(sorted(known_geometries))}"
        )
    avail = combinations(model_keys)
    if method is not None:
        if not given:
            spec = method_spec(method)
            if not model_keys.intersection(spec.models):
                requested_for = ", ".join(sorted(model_keys))
                raise ValueError(
                    f"method {method!r} is not available for {requested_for}; "
                    f"it belongs to {', '.join(spec.models)}"
                )
            return spec
        raise ValueError(
            "give either method= or representation/state_geometry/integrator, "
            "not both")

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
            why = why_not(
                given.get("model", mk), given.get("representation"),
                state_geometry=given.get("state_geometry"))
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
        raise KeyError(f"unknown model {key!r}; available: "
                       f"{', '.join(sorted(MODELS))}") from None


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
                             f"state_geometry={s.state_geometry:<11} "
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
