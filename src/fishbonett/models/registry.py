"""The model taxonomy: which frames and propagators each physical setup admits.

A **model** is the physical setup -- how many system sites there are, how they are
wired to each other, and how the bath is represented.  It is the outermost of
three nested choices:

    model  ->  frame  ->  propagator
    what is coupled     how H is        how a step
    to what             written down    is taken

The nesting is physical, not bookkeeping.  A model fixes the state geometry, and
that plus the frame decides which propagators are even applicable: a static ``H``
admits TDVP on an MPO built once, a time-dependent one must rebuild every step,
and only the interaction picture makes all the coupling terms commute (which is
what lets ``trotter-mpo`` write the propagator in closed form).

This module is the single source of truth for that taxonomy, and -- since
:data:`METHODS` carries each combination's ``integrator`` -- for the **dispatch**
as well.  It answers four questions that were previously spread across a flat
table, several dispatch dicts, and prose in half a dozen doc pages:

* what models exist, and what does each one *mean* physically;
* for a model, which frames are available and which methods realize them;
* for a model/frame that is **absent**, why -- impossible, unwise, or merely
  unimplemented (:attr:`Model.gaps`);
* for a method, **how to run it** -- which driver realizes it, which entry point
  in :mod:`fishbonett.evolve` it uses, and whether its bond is fixed.

That last one used to live in four module-level dicts in
:mod:`fishbonett.models.system_bath`, which restated the method names this table
already had; ``run`` was then a chain of ``if``s over those names.  Both are gone:
:attr:`Model.frames` is *derived* from :data:`METHODS`, and ``run`` is a lookup.
Adding a ``(model, frame, integrator)`` combination is a row here, not a branch.

.. note::
   The name ``fishbonett.models`` was used once before, for what is now
   :mod:`fishbonett.frames`.  If you are reading commits from before that
   rename, ``models/`` there means the Hamiltonian builders, not this.

Propagator-level gaps, which are finer than this table records: the
Schroedinger chain could be driven by TEBD gates but is not (only MPO/TDVP is
wired); the conditional-displacement MPO of ``trotter-mpo`` exists only in the
interaction picture, because outside it the coupling does not commute with the
free-bath term.
"""
from dataclasses import dataclass, field
from typing import Mapping, Tuple

__all__ = ["Model", "Frame", "Method", "MODELS", "FRAMES", "METHODS",
           "FIXED_BOND_METHODS", "models_of", "frames_of",
           "methods_of", "all_methods", "model", "method_spec", "METHOD_FRAMES",
           "methods_by_frame", "frame_label", "describe_taxonomy",
           "unknown_method_error", "STATIC_TREE_TEBD"]


# -- frames ------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """One way of writing the Hamiltonian down."""
    key: str
    label: str
    blurb: str
    static: bool          # is H time-independent?  (decides TDVP applicability)


FRAMES = {
    "schrodinger": Frame(
        "schrodinger", "Schrodinger picture",
        "Nothing rotated out.  H is time-independent and its MPO is built once, "
        "so TDVP conserves energy -- but the state carries the full system-bath "
        "correlation, giving the largest bond dimensions.",
        static=True),
    "interaction": Frame(
        "interaction", "interaction picture",
        "The free-bath evolution is rotated out, leaving only the coupling.  "
        "Entanglement is much smaller, but H is time-dependent so gates/MPOs are "
        "rebuilt every step.  All the coupling terms commute here, which is what "
        "makes the exact conditional-displacement propagator possible.",
        static=False),
    "polaron": Frame(
        "polaron", "polaron frame",
        "The static part of the coupling is absorbed into a bath displacement, "
        "leaving a free chain plus a dressed tunneling term.  Static like the "
        "Schroedinger picture *and* low-entanglement like the interaction "
        "picture; needs int J/w^2 finite.  Populations are frame-invariant, "
        "coherences must be un-dressed.",
        static=True),
}


# -- models ------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """A physical setup, and the frames/propagators it admits.

    ``gaps`` maps an *absent* frame key to the reason it is absent -- so "why is
    there no polaron tree?" has one authoritative answer instead of none.

    ``frames`` is **derived** from :data:`METHODS` rather than stored.  It used to
    be a field, which meant the set of methods was written down twice (here, and in
    the dispatch dicts that said how to call them) and a test existed purely to
    check the two agreed.  One table now, so they cannot disagree.
    """
    key: str
    label: str
    blurb: str
    cls: str                              # the class a user instantiates
    geometry: str                         # the state ansatz this implies
    gaps: Mapping[str, str] = field(default_factory=dict)
    #: ``"method"`` -- chosen by ``run(method=...)``; ``"bath"`` -- chosen
    #: automatically from the bath's shape (the multichannel case).
    selected_by: str = "method"

    @property
    def frames(self):
        """``{frame key: method names}``, in taxonomy order."""
        out = {}
        for frame_key in FRAMES:
            names = tuple(name for name, spec in METHODS.items()
                          if spec.frame == frame_key and self.key in spec.models)
            if names:
                out[frame_key] = names
        return out

    def methods(self):
        """Every method name this model offers, across all its frames."""
        return tuple(m for fr in self.frames.values() for m in fr)


# -- methods: the one dispatch table -----------------------------------------
@dataclass(frozen=True)
class Method:
    """One realizable *(model, frame, integrator)* combination.

    This is the single source of truth: what exists **and** how it is dispatched.
    ``models`` is a tuple because one propagator can serve several topologies --
    the static tree TEBD runs the comb, the site-tree and the multichannel star.
    """
    name: str
    frame: str
    models: Tuple[str, ...]
    #: which driver in the model layer realizes it
    integrator: str
    #: the entry point in :mod:`fishbonett.evolve`, where there is a single one
    driver: str = ""
    #: 1-site TDVP cannot grow a bond and adaptive DTDVP needs a ceiling, so
    #: ``bond_dim=None`` ("unlimited") is not meaningful for these.
    fixed_bond: bool = False


#: Why the mode-tree has only the interaction picture.  Not an oversight: the
#: tree is worth having *because* the interaction picture leaves no mode-mode
#: terms, so every mode hangs off the system independently and the only question
#: is how deep the bonds are.  The static and polaron frames both keep the free
#: chain, whose hoppings couple mode ``i`` to ``i+1`` -- and on a balanced tree
#: with leaves in chain order only half of those pairs are tree-adjacent, the
#: rest spanning up to ``2 log2(N)`` edges (measured: 10 edges at N=32).  Putting
#: them back means carrying ladder operators across those paths, so the MPO bond
#: grows and the tree costs more than the chain it was meant to beat.  Reordering
#: the leaves to make the chain local just turns the tree back into a chain.
_NO_MODE_MODE = (
    "the tree pays off only when there are no mode-mode terms, which is what the "
    "interaction picture buys.  This frame keeps the free chain, whose "
    "nearest-neighbour hoppings are long-range on a balanced tree (only half of "
    "the chain-adjacent pairs are tree-adjacent; the rest span up to 2*log2(N) "
    "edges), so the MPO bond grows and the tree loses to the plain chain.  Use "
    "the 'chain' model instead.")

#: The one propagator the multi-site models have: Schroedinger-picture tree TEBD.
#: Named to distinguish it from the interaction-picture ``tree-tebd`` of the
#: ``mode-tree`` model, which is a different engine on a different geometry.
STATIC_TREE_TEBD = "tree-tebd-static"
_TREE_STATIC = STATIC_TREE_TEBD       # shorthand used in the table below

#: The multichannel model's interaction-picture propagator: a swap-network TEBD
#: sweep against the matrix-valued time-dependent coupling.  Named rather than
#: shared with ``tebd`` because the builder is a different class (the coupling is
#: a matrix per mode, not a scalar times one operator).
MULTICHANNEL_IP = "multichannel-ip"


def _m(name, frame, models, integrator, driver="", fixed_bond=False):
    return Method(name, frame, models, integrator, driver, fixed_bond)


#: Every method, declared once.  Order matters: :attr:`Model.frames` reads method
#: order off this table, and ``methods_of(model, frame)[0]`` is the default a
#: bath-selected model falls back to.
METHODS = {s.name: s for s in [
    # -- chain -------------------------------------------------------------
    _m("mpo-tdvp1", "schrodinger", ("chain",), "chain-mpo-tdvp",
       "run_tdvp1", fixed_bond=True),
    _m("mpo-tdvp2", "schrodinger", ("chain",), "chain-mpo-tdvp", "run_tdvp2"),
    _m("mpo-dtdvp", "schrodinger", ("chain",), "chain-mpo-tdvp",
       "run_dtdvp", fixed_bond=True),
    _m("tebd", "interaction", ("chain",), "swap-tebd"),
    _m("trotter-mpo", "interaction", ("chain",), "displacement-mpo"),
    _m("polaron", "polaron", ("chain",), "polaron-tebd"),
    _m("polaron-tdvp1", "polaron", ("chain",), "polaron-tdvp",
       "tdvp1", fixed_bond=True),
    _m("polaron-tdvp2", "polaron", ("chain",), "polaron-tdvp", "tdvp2"),
    _m("polaron-dtdvp", "polaron", ("chain",), "polaron-tdvp",
       "dtdvp", fixed_bond=True),
    # -- star --------------------------------------------------------------
    _m("mpo-star-tdvp1", "schrodinger", ("star",), "chain-mpo-tdvp",
       "run_star_tdvp1", fixed_bond=True),
    _m("mpo-star-tdvp2", "schrodinger", ("star",), "chain-mpo-tdvp",
       "run_star_tdvp2"),
    _m("mpo-ip-tdvp1", "interaction", ("star",), "chain-mpo-tdvp",
       "run_ip_tdvp1", fixed_bond=True),
    _m("mpo-ip-tdvp2", "interaction", ("star",), "chain-mpo-tdvp",
       "run_ip_tdvp2"),
    # -- mode-tree ---------------------------------------------------------
    _m("tree-tdvp", "interaction", ("mode-tree",), "modetree",
       "run_tree_tdvp", fixed_bond=True),
    _m("tree-tdvp2", "interaction", ("mode-tree",), "modetree",
       "run_tree_tdvp2"),
    _m("tree-tebd", "interaction", ("mode-tree",), "modetree",
       "run_tree_tebd"),
    # -- the static tree engine: one propagator, three topologies ----------
    _m(STATIC_TREE_TEBD, "schrodinger", ("multichannel", "comb", "site-tree"),
       "static-tree-tebd"),
    _m(MULTICHANNEL_IP, "interaction", ("multichannel",), "multichannel-swap-tebd"),
]}

#: Derived from :data:`METHODS` -- was a hand-maintained set in
#: ``models/system_bath.py`` that the tests had to import privately.
FIXED_BOND_METHODS = frozenset(n for n, s in METHODS.items() if s.fixed_bond)


def method_spec(name, model_key=None):
    """The :class:`Method` for ``name``, or a :class:`ValueError` naming what is."""
    spec = METHODS.get(name)
    if spec is None or (model_key is not None and model_key not in spec.models):
        raise unknown_method_error(name, model_key)
    return spec


MODELS = {
    "chain": Model(
        key="chain", label="1D system-bath",
        blurb="One system site coupled to one bath, the bath chain-mapped into a "
              "1D chain of effective modes.  The most developed model: all three "
              "frames and the whole propagator family.",
        cls="SystemBath", geometry="MPS (system at site 0, modes 1..N)"),
    "star": Model(
        key="star", label="star system-bath",
        blurb="One system site, one bath, *no* chain mapping: every discretized "
              "mode couples straight to the system.  No mode-mode terms, but no "
              "locality for the MPS to exploit either.",
        cls="SystemBath", geometry="MPS over a linearized star",
        gaps={
            "polaron": "rejected, not merely unimplemented: the polaron "
                       "displacement acts on the collective mode, which in a "
                       "star is spread over *every* mode at once.  On a chain "
                       "the J/w^2 mapping localizes it on c_0; a star has no "
                       "such site, so the dressing entangles the system with "
                       "all modes simultaneously.  Use the 'chain' model.",
        }),
    "mode-tree": Model(
        key="mode-tree", label="tree system-bath",
        blurb="One system site, one bath: the same chain-mapped modes as "
              "``chain``, but placed on a balanced binary tree with the system at "
              "the root.  Keeps the high-bond region O(log N) edges deep instead "
              "of O(N).  Distinct from ``site-tree``, where it is the *system "
              "sites* that form a tree.",
        cls="SystemBath", geometry="balanced binary TTN, system at the root",
        gaps={
            "schrodinger": _NO_MODE_MODE,
            "polaron": _NO_MODE_MODE,
        }),
    "multichannel": Model(
        key="multichannel", label="multichannel system-bath",
        blurb="One system site, one bath, coupled through *several* operators "
              "that share the same modes -- so the channels are cross-correlated, "
              "unlike independent baths.  Selected by giving the Bath a list of "
              "couplings, not by a method name.",
        cls="SystemBath", geometry="shared-mode star on a one-site tree",
        gaps={
            "polaron": "rejected for the same reason as the 'star' model: the "
                       "channels share one set of star modes, and the polaron "
                       "displacement acts on a collective mode that is spread over "
                       "all of them.  There is no single site to localize it on, so "
                       "the dressing entangles the system with every mode at once.",
        },
        selected_by="bath"),
    "comb": Model(
        key="comb", label="fishbone / comb",
        blurb="Several system sites on a 1D backbone, each carrying one or two "
              "baths -- the fishbone.  A specialization of ``site-tree`` to a "
              "linear backbone.",
        cls="Fishbone", geometry="comb / tree TTN",
        gaps={
            "interaction": "not implemented for multi-site models.",
            "polaron": "not implemented for multi-site models.",
        }),
    "site-tree": Model(
        key="site-tree", label="tree of sites",
        blurb="Several system sites wired into any loop-free tree, each carrying "
              "zero or more baths.  The most general geometry.  Distinct from "
              "``mode-tree``, where a single system's *bath modes* form the tree.",
        cls="TreeFishbone", geometry="arbitrary loop-free TTN",
        gaps={
            "interaction": "not implemented for multi-site models.",
            "polaron": "not implemented for multi-site models.",
        }),
}


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

    A tuple rather than a single key because the static tree engine
    (``tree-tebd-static``) genuinely serves both multi-site models and the
    multichannel one -- the same propagator, different topologies.
    """
    return tuple(k for k, m in sorted(MODELS.items()) if method in m.methods())


def frames_of(model_key):
    """Frame keys available for a model, in taxonomy order."""
    m = model(model_key)
    return tuple(k for k in FRAMES if k in m.frames)


def methods_of(model_key, frame=None):
    """Methods for a model, optionally narrowed to one frame."""
    m = model(model_key)
    if frame is None:
        return m.methods()
    if frame not in m.frames:
        why = m.gaps.get(frame)
        raise KeyError(
            f"model {model_key!r} has no {frame!r} frame"
            + (f": {why}" if why else "")
            + f"  (available: {', '.join(frames_of(model_key))})")
    return tuple(m.frames[frame])


def all_methods():
    """Every method name in the taxonomy."""
    return tuple(sorted({m for mo in MODELS.values() for m in mo.methods()}))


#: Derived ``method -> (frame, model)``.  Replaces the hand-maintained table
#: that preceded this module; note the two corrections it carries:
#:
#: * the ``tree-*`` methods are ``("interaction", "mode-tree")``, not
#:   ``("interaction", "chain")`` -- their state is a tree, not a chain;
#: * ``polaron`` is its own frame, not ``("schrodinger", "polaron-chain")``.
#:   It shares the Schroedinger picture's *staticness* without being it.
METHOD_FRAMES = {
    method: (frame_key, model_key)
    for model_key, mo in MODELS.items()
    for frame_key, methods in mo.frames.items()
    for method in methods
}


def methods_by_frame():
    """``{(frame, model): [methods]}`` -- the taxonomy as a flat grouping."""
    out = {}
    for method, key in sorted(METHOD_FRAMES.items()):
        out.setdefault(key, []).append(method)
    return out


def frame_label(key):
    """Human-readable name of a ``(frame, model)`` pair or a bare frame key."""
    if isinstance(key, tuple):
        frame_key, model_key = key
        return f"{FRAMES[frame_key].label} / {MODELS[model_key].label}"
    return FRAMES[key].label


def describe_taxonomy():
    """The whole taxonomy as readable text -- what ``run`` prints when asked for
    an unknown method, and what the docs table is generated from."""
    lines = []
    for key, m in MODELS.items():
        picked = "" if m.selected_by == "method" else "  (selected by the bath)"
        lines.append(f"{m.label}  [{key}] via {m.cls}{picked}")
        for frame_key in frames_of(key):
            names = ", ".join(m.frames[frame_key])
            lines.append(f"    {FRAMES[frame_key].label:24s} {names}")
        for frame_key, why in m.gaps.items():
            lines.append(f"    {FRAMES[frame_key].label:24s} -- absent: {why}")
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
