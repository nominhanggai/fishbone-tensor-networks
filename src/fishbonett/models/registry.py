"""Registry of supported model, representation, state, and integrator combinations.

A run is specified by four axes::

    model     what is coupled to what      system-bath | multichannel | comb | site-tree
    representation  how H is written         schrodinger-chain | schrodinger-star
                                             | interaction-chain | interaction-star
                                             | polaron-chain | polaron-star
    state_geometry  tensor-network state        mps | binary-tree | tree
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

At the propagator level, the Schroedinger chain currently supports MPO/TDVP. The
conditional-displacement MPO of ``interaction-chain-trotter-mpo`` exists only in
the interaction representation, because outside it the coupling does not commute
with the free-bath term.
"""
from dataclasses import dataclass, field
from typing import Mapping, Tuple

__all__ = ["Model", "RepresentationSpec", "MethodSpec", "MODELS", "REPRESENTATIONS", "METHODS",
           "STATE_GEOMETRIES", "APPLICATIONS", "BOND_CAP_REQUIRED_METHODS",
           "why_not", "models_of", "representations_of",
           "methods_of", "all_methods", "model", "method_spec", "resolve",
           "combinations", "METHOD_REPRESENTATIONS",
           "methods_by_representation", "representation_label", "describe_taxonomy",
           "unknown_method_error", "SCHRODINGER_CHAIN_TREE_TEBD",
           "SCHRODINGER_STAR_TREE_TEBD", "INTERACTION_CHAIN_TEBD",
           "INTERACTION_STAR_TEBD"]


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
        "Nothing rotated out, bath chain-mapped.  H is time-independent and its MPO "
        "is built once, so TDVP conserves energy -- but the state carries the full "
        "system-bath correlation, giving the largest bond dimensions.  The chain's "
        "nearest-neighbour hoppings are what an MPS is good at, and the system "
        "touches only c0.",
        static=True, mode_decoupled=False),
    "schrodinger-star": RepresentationSpec(
        "schrodinger-star", "Schrodinger star representation",
        "Nothing rotated out, no chain mapping: every mode couples straight to the "
        "system.  No mode-mode terms, but no locality for the MPS to exploit "
        "either.  Static, so still one MPO built once.",
        static=True, mode_decoupled=True),
    "interaction-chain": RepresentationSpec(
        "interaction-chain", "interaction chain representation",
        "A finite star is put in the interaction representation with respect to "
        "its free Hamiltonian, then its time-dependent coupling is transformed "
        "star-to-chain.  What is lost is locality, not existence -- the coupling "
        "d_n(t) starts concentrated on c0 "
        "at t=0 and spreads outward, so this is 'no longer a chain' in the only "
        "sense that matters to an MPS.  Entanglement is much smaller than the "
        "Schrodinger representation's, but H is time-dependent, so gates/MPOs are "
        "rebuilt "
        "every step.  For a single coupling channel the mode terms commute, which "
        "makes the conditional-displacement propagator possible.",
        static=False, mode_decoupled=True),
    "interaction-star": RepresentationSpec(
        "interaction-star", "interaction star representation",
        "The same rotation as interaction-chain, left in the star modes instead of "
        "being rotated back: the coupling of mode k is simply V_k e^{-i w_k t}.  "
        "It is unitarily equivalent to interaction-chain. Their tensor-network "
        "costs can differ because the time-dependent coupling vectors differ. The "
        "multichannel model exposes both forms using one common orthogonal transform "
        "for its matrix-valued star couplings.",
        static=False, mode_decoupled=True),
    "polaron-chain": RepresentationSpec(
        "polaron-chain", "polaron chain representation",
        "The static part of the coupling is absorbed into a bath displacement, "
        "leaving a free chain plus a dressed tunneling term.  Static like the "
        "Schrodinger representation *and* low-entanglement like the interaction "
        "representation; needs int J/w^2 finite.  Populations are "
        "representation-invariant, "
        "coherences must be un-dressed.  The J/w^2 reweighting is what localizes "
        "the displacement on c0.",
        static=True, mode_decoupled=False),
    "polaron-star": RepresentationSpec(
        "polaron-star", "polaron star representation",
        "The textbook Lang-Firsov transform, which is *defined* per star mode: "
        "prod_k D_k(g_k sigma_z / w_k).  Perfectly well defined -- it is the chain "
        "version that uses the J/w^2 transformation to localize the collective "
        "displacement onto c0.",
        static=True, mode_decoupled=True),
}

#: The tensor-network geometry of the state. Independent of the representation:
#: the same interaction-chain Hamiltonian runs on a 1D MPS
#: (``interaction-chain-tdvp1``) and a balanced tree
#: (``interaction-chain-tree-tebd``).
STATE_GEOMETRIES = {
    "mps": "a 1D MPS: system at site 0, modes 1..N in a line.",
    "binary-tree": "a binary tree tensor network with the system at the root, "
                   "keeping the high-bond region O(log N) edges deep instead of O(N).",
    "tree": "a general tree tensor network whose model determines whether the "
            "physical graph is a comb, star, or arbitrary loop-free tree.",
}


#: Why the binary tree needs a representation with no mode-mode terms
#: (:attr:`RepresentationSpec.mode_decoupled`). The tree is useful because nothing
#: couples mode to mode, so every mode hangs off the system independently and the
#: only question is how deep the bonds are.
#:
#: Note this is about the *representation*, not the structure --
#: ``interaction-chain`` qualifies (it rotates ``H_B`` away entirely) and is
#: what the ``tree-*`` methods use.
_NO_MODE_MODE = (
    "the balanced binary tree pays off only when there are no mode-mode terms.  "
    "This representation keeps the chain hoppings, which are nearest-neighbour "
    "on a 1D MPS but "
    "long-range on that tree (only half of the chain-adjacent pairs are "
    "tree-adjacent and the rest can span logarithmically many edges), so the MPO "
    "bond grows. Reordering "
    "the leaves to make the chain local just turns the state back into an MPS.")


# -- models ------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """A physical setup containing only its coupling topology.

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
    #: the entry point in :mod:`fishbonett.evolve`, where there is a single one
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

    @property
    def application(self):
        """How H's **interaction graph** is realized on the state's graph.

        Derived, not declared: it is a *consequence* of the other axes.  A representation can
        make ``H`` non-local relative to the state -- the interaction transformation couples
        every mode to the system, a star, while the state is a 1D MPS -- and this
        records what pays for that.  See :data:`APPLICATIONS`.

        Keys on :attr:`RepresentationSpec.mode_decoupled`, **not** on the structure:
        ``interaction-chain-tebd`` still needs a swap network, because it is rotating
        out ``H_B`` that spreads the coupling over every mode, not the choice of
        modes to write it in.
        """
        if self.integrator != "tebd" or self.state_geometry == "binary-tree":
            return "operator"
        if (self.state_geometry == "mps"
                and REPRESENTATIONS[self.representation].mode_decoupled):
            return "swap"
        return "local"


#: The values :attr:`MethodSpec.application` takes.
#:
#: This is derived from the representation, state geometry, and integrator. The
#: representation decides which terms exist; the application records how those
#: terms act on the selected tensor-network graph.
APPLICATIONS = {
    "local": "interaction edges are state edges -- gates apply in place",
    "swap": "a star realized on a 1D MPS: the system site is walked past every mode "
            "and back each step, so a step costs O(N) swaps on top of the gates",
    "operator": "no gate layout -- a single low-bond operator (an MPO on a 1D MPS, "
                "a bond-K tree operator on a tree) carries the long-range terms "
                "natively, so the interaction graph never has to match the state's",
}


def _canonical_method_name(representation, integrator, state_geometry="mps"):
    """Derive a method name from its representation and algorithm.

    MPS methods use ``<representation>-<integrator>``. A tree tensor network
    inserts ``tree`` so methods that share a representation and integrator remain
    unambiguous.  Keeping this rule here prevents a registry label from drifting
    away from the tuple it denotes.
    """
    state_geometry_tag = "" if state_geometry == "mps" else "tree-"
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
    "tree-tdvp": "interaction-chain-tree-tebd",
    "tree-tdvp2": "interaction-chain-tree-tebd",
    "tree-tebd": "interaction-chain-tree-tebd",
    "interaction-chain-tree-tdvp1": "interaction-chain-tree-tebd",
    "interaction-chain-tree-tdvp2": "interaction-chain-tree-tebd",
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
       "tdvp1", requires_bond_cap=True),
    _m("schrodinger-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("schrodinger-chain", _SB, "mpo-tdvp",
       "dtdvp", requires_bond_cap=True),
    _m("schrodinger-star", _SB, "mpo-tdvp",
       "tdvp1", requires_bond_cap=True),
    _m("schrodinger-star", _SB, "mpo-tdvp", "tdvp2"),
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
    # -- ...the same rotation left in the star modes --------------------------
    _m("interaction-star", _SB, "mpo-tdvp",
       "tdvp1", requires_bond_cap=True),
    _m("interaction-star", _SB, "mpo-tdvp", "tdvp2"),
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
    # -- the static tree engine: one engine, two representations, three topologies -----
    _m("schrodinger-chain", ("comb", "site-tree"),
       "static-tree-tebd", integrator="tebd", state_geometry="tree"),
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

#: Methods whose integrator requires a finite bond-dimension ceiling.
BOND_CAP_REQUIRED_METHODS = frozenset(
    name for name, spec in METHODS.items() if spec.requires_bond_cap
)


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
        blurb="One system site coupled to one bath through one coupling operator. "
              "It supports all six representations, both single-system tensor-network "
              "geometries and the full integrator family. The schrodinger-chain, "
              "schrodinger-star and interaction-chain representations describe the "
              "Hamiltonian, while interaction-chain supports two state geometries.",
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
            "interaction-star": _MULTISITE,
            "polaron-chain": _MULTISITE,
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
            "interaction-star": _MULTISITE,
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
    for a system-bath model.  This is one lookup either way, and mixing the two
    spellings is rejected rather than silently resolved.

    Representation names are explicit: use ``interaction-chain`` rather than a
    partial name such as ``interaction``.
    """
    model_keys = set(model_keys)
    if not model_keys:
        raise ValueError("model_keys must contain at least one model")
    unknown_models = model_keys - set(MODELS)
    if unknown_models:
        raise ValueError(f"unknown model key(s): {', '.join(sorted(unknown_models))}")
    if "geometry" in axes:
        raise TypeError(
            "'geometry' is no longer a public axis; use state_geometry")
    renamed_state_geometries = {"path": "mps", "comb-tree": "tree"}
    state_geometry = axes.get("state_geometry")
    if state_geometry in renamed_state_geometries:
        replacement = renamed_state_geometries[state_geometry]
        raise ValueError(
            f"state_geometry={state_geometry!r} was renamed; use "
            f"state_geometry={replacement!r}")
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
        extra = ""
        if key in ("chain", "star"):
            extra = (f"  ({key!r} is half of a *representation*, not a model -- the representations "
                     f"are {', '.join(k for k in REPRESENTATIONS if k.endswith(key))}.)")
        elif key == "mode-tree":
            extra = ("  ('mode-tree' was a tensor-network state geometry, not a "
                     "model -- use model='system-bath', "
                     "state_geometry='binary-tree'.)")
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
