"""The taxonomy: the five axes a run is made of, and which combinations exist.

A run is **five independent choices**, and the point of this module is that they
are independent -- which they were not when three of them were mashed into one
string called the "model"::

    model     what is coupled to what      system-bath | multichannel | comb | site-tree
    frame     which unitary is rotated out schrodinger | interaction | polaron
    basis     the bath mode basis          chain | star
    geometry  the graph the state lives on path | binary-tree | comb-tree
    integrator  how a step is taken        tebd | tdvp1 | tdvp2 | dtdvp | trotter-mpo

The axis this module used to be missing is ``basis``, and its absence is what made
the old table wrong rather than merely coarse.  ``model="chain"`` with
``frame="interaction"`` was **not a chain**:
:meth:`~fishbonett.frames.interaction_picture.SystemBathIP.build` chain-maps the bath
and then calls ``diag()``, which diagonalizes that chain straight back into its star,
because it is the *star* modes whose free evolution ``e^{-i w_k t}`` the interaction
picture rotates out.  The chain was built and discarded.  Likewise ``mode-tree`` was
never a model: it is the same one-system/one-bath problem in the same star basis as
``mpo-ip-tdvp1`` -- they call the same
:func:`~fishbonett.bath.chain.star_transform` -- differing only in ``geometry``,
which is why the two produce *identical* numbers rather than merely close ones.

.. rubric:: The frame picks the basis

That is the rule the old table encoded as three separate paragraphs of prose, and
it is physics rather than bookkeeping (see :func:`forced_basis`):

* **interaction** forces ``star`` -- ``H_B = sum_k w_k b_k^dag b_k`` is diagonal
  only there, so there is no chain left to speak of;
* **polaron** forces ``chain`` -- the displacement has to localize on ``c0``, and a
  star has no such site;
* **multichannel** forces ``star`` -- the channels share one set of modes;
* **comb** / **site-tree** force ``chain`` -- one chain mapping per bath;
* **(schrodinger, system-bath)** is the one cell where the basis is a genuinely
  free choice -- which is exactly why ``mpo-tdvp2`` (chain) and ``mpo-star-tdvp2``
  (star) are the only pair in the whole table that differ by basis alone.

:data:`METHODS` is therefore the single source of truth for both the taxonomy and
the **dispatch**: each row carries its five axes plus the engine that realizes it,
so ``run`` is a lookup rather than a chain of ``if``\\ s, and adding a combination
is a row here rather than a branch there.

.. note::
   The name ``fishbonett.models`` was used once before, for what is now
   :mod:`fishbonett.frames`.  If you are reading commits from before that rename,
   ``models/`` there means the Hamiltonian builders, not this.

Propagator-level gaps, finer than this table records: the Schroedinger chain could
be driven by TEBD gates but is not (only MPO/TDVP is wired); the
conditional-displacement MPO of ``trotter-mpo`` exists only in the interaction
picture, because outside it the coupling does not commute with the free-bath term.
"""
from dataclasses import dataclass, field
from typing import Mapping, Tuple

__all__ = ["Model", "Frame", "Method", "MODELS", "FRAMES", "METHODS",
           "BASES", "GEOMETRIES", "APPLICATIONS", "FIXED_BOND_METHODS",
           "forced_basis", "why_not", "models_of", "frames_of",
           "methods_of", "all_methods", "model", "method_spec", "resolve",
           "combinations", "METHOD_FRAMES",
           "methods_by_frame", "frame_label", "describe_taxonomy",
           "unknown_method_error", "STATIC_TREE_TEBD", "MULTICHANNEL_IP"]


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
        "correlation, giving the largest bond dimensions.  The only frame that "
        "leaves the basis free, since it is the only one that rotates out nothing.",
        static=True),
    "interaction": Frame(
        "interaction", "interaction picture",
        "The free-bath evolution is rotated out, leaving only the coupling.  "
        "Entanglement is much smaller, but H is time-dependent so gates/MPOs are "
        "rebuilt every step.  All the coupling terms commute here, which is what "
        "makes the exact conditional-displacement propagator possible.  Forces the "
        "star basis: it is the star modes whose free evolution is being rotated out.",
        static=False),
    "polaron": Frame(
        "polaron", "polaron frame",
        "The static part of the coupling is absorbed into a bath displacement, "
        "leaving a free chain plus a dressed tunneling term.  Static like the "
        "Schroedinger picture *and* low-entanglement like the interaction "
        "picture; needs int J/w^2 finite.  Populations are frame-invariant, "
        "coherences must be un-dressed.  Forces the chain basis: the displacement "
        "has to localize on c0.",
        static=True),
}


# -- the bath mode basis ------------------------------------------------------
#: The two bases the bath modes can be written in.  An orthogonal (Lanczos)
#: transform relates them, so this is a change of *representation*, not of physics
#: -- which is why it is an axis of its own and not a "model".
BASES = {
    "chain": "TEDOPA chain: nearest-neighbour hoppings, so H is local for an MPS "
             "and the system touches only c0.  The free bath is tridiagonal, not "
             "diagonal.",
    "star": "every mode couples straight to the system and nothing couples to "
            "anything else.  The free bath is diagonal (which is what lets the "
            "interaction picture rotate it out) but H is non-local on a path.",
}

#: The graph the state's tensors live on.  Independent of the basis: the same star
#: Hamiltonian runs on a path (``mpo-ip-tdvp1``) and on a balanced binary tree
#: (``tree-tdvp``), which is the whole reason ``mode-tree`` was never a model.
GEOMETRIES = {
    "path": "an MPS: system at site 0, modes 1..N in a line.",
    "binary-tree": "a balanced binary TTN with the system at the root, keeping the "
                   "high-bond region O(log N) edges deep instead of O(N).",
    "comb-tree": "a tree of system sites, each carrying its own bath chain(s) as a "
                 "branch -- the comb / fishbone geometry.",
}


#: The basis each *frame* forces, or absent where it forces none.
_FRAME_BASIS = {
    "interaction": "star",     # H_B is diagonal only in the star basis
    "polaron": "chain",        # the displacement must localize on c0
}

#: The basis each *model* forces, or absent where it forces none.
_MODEL_BASIS = {
    "multichannel": "star",    # the channels share one set of modes
    "comb": "chain",           # one chain mapping per bath
    "site-tree": "chain",
}


def forced_basis(frame, model_key):
    """The basis a ``(frame, model)`` **forces**, or ``None`` where it is free.

    This one function replaces three hand-written ``gaps`` paragraphs, because the
    combinations they described as "rejected" are now simply unsatisfiable
    constraints -- and the reason is stated once, here, instead of once per model.

    The frame and the model can each force a basis, and where they force
    *different* ones the combination does not exist at all -- which is how
    ``(multichannel, polaron)`` is ruled out without anybody writing that down.
    :func:`why_not` reports the clash; this returns the frame's choice, since a
    caller reaching here has already got a combination that exists.
    """
    return _FRAME_BASIS.get(frame) or _MODEL_BASIS.get(model_key)


#: Why each forced basis is forced -- the physics, for error messages.
_BASIS_REASON = {
    ("interaction", "chain"):
        "the interaction picture rotates out H_B = sum_k w_k b_k^dag b_k, which is "
        "diagonal only in the star basis, so no chain survives it.  (SystemBathIP "
        "does chain-map the bath -- and then calls diag() to turn it straight back "
        "into a star.)",
    ("polaron", "star"):
        "the polaron displacement acts on the collective mode.  The J/w^2 chain "
        "mapping localizes that on c0; a star has no such site, so the dressing "
        "would entangle the system with every mode at once.",
}

#: Why the binary tree needs the star basis.  Not an oversight: the tree is worth
#: having *because* there are no mode-mode terms, so every mode hangs off the system
#: independently and the only question is how deep the bonds are.
_NO_MODE_MODE = (
    "the balanced binary tree pays off only when there are no mode-mode terms, "
    "which is what the star basis gives.  A chain's nearest-neighbour hoppings are "
    "long-range on that tree (only half of the chain-adjacent pairs are "
    "tree-adjacent; the rest span up to 2*log2(N) edges -- measured: 10 edges at "
    "N=32), so the MPO bond grows and the tree loses to the plain path.  Reordering "
    "the leaves to make the chain local just turns the tree back into a path.")


# -- models ------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """A physical setup -- **only** the topology, now that basis and geometry are
    axes of their own.

    ``gaps`` maps an absent frame key to the reason it is absent.  It is much
    shorter than it used to be, because the entries that said "rejected because the
    polaron displacement has nowhere to localize" are now *derived* from
    :func:`forced_basis` rather than written out per model.  What is left is
    genuinely-unimplemented work.
    """
    key: str
    label: str
    blurb: str
    cls: str                              # the class a user instantiates
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
    """One realizable combination of the five axes.

    This is the single source of truth: what exists **and** how it is dispatched.
    ``models`` is a tuple because one engine can serve several topologies -- the
    static tree TEBD runs the comb, the site-tree and the multichannel star.
    """
    name: str
    frame: str
    models: Tuple[str, ...]
    #: which driver group in the model layer realizes it (``_DRIVERS`` in
    #: :mod:`fishbonett.models.system_bath` keys on this).  Not an axis.
    engine: str
    #: the entry point in :mod:`fishbonett.evolve`, where there is a single one
    driver: str = ""
    #: 1-site TDVP cannot grow a bond and adaptive DTDVP needs a ceiling, so
    #: ``bond_dim=None`` ("unlimited") is not meaningful for these.
    fixed_bond: bool = False
    #: The integrator **axis** -- ``"tebd"``, ``"tdvp1"``, ``"tdvp2"``, ``"dtdvp"``,
    #: ``"trotter-mpo"``.  Unique within a ``(model, frame, basis, geometry)``,
    #: which is what lets ``run`` be called by the axes instead of by a name that
    #: mashes them together.
    integrator: str = ""
    #: The state graph -- see :data:`GEOMETRIES`.
    geometry: str = "path"
    #: The bath mode basis, declared **only where it is a free choice**; elsewhere
    #: it follows from :func:`forced_basis`, and writing it out again would be a
    #: second place for it to be wrong.  In practice that means only the
    #: ``(schrodinger, system-bath)`` rows carry one.
    basis: str = ""

    def basis_for(self, model_key=None):
        """The bath basis this method uses for one of its models."""
        return self.basis or forced_basis(self.frame,
                                          model_key or self.models[0])

    @property
    def application(self):
        """How H's **interaction graph** is realized on the state's graph.

        Derived, not declared: it is a *consequence* of the other axes.  A frame can
        make ``H`` non-local relative to the state -- the interaction picture couples
        every mode to the system, a star, while the state is a path -- and this
        records what pays for that.  See :data:`APPLICATIONS`.
        """
        if self.integrator != "tebd" or self.geometry == "binary-tree":
            return "operator"
        if self.geometry == "path" and self.basis_for() == "star":
            return "swap"
        return "local"


#: The values :attr:`Method.application` takes.
#:
#: This is the axis the package had no name for, which is why three methods each
#: re-derived a swap network and nothing recorded that they did.  It is not the same
#: question as the frame: the *frame* decides which terms exist, the application
#: decides what it costs to apply them to a state whose graph is shaped differently.
APPLICATIONS = {
    "local": "interaction edges are state edges -- gates apply in place",
    "swap": "a star realized on a path: the system site is walked past every mode "
            "and back each step, so a step costs O(N) swaps on top of the gates",
    "operator": "no gate layout -- a single low-bond operator (an MPO on a path, "
                "a bond-K tree operator on a tree) carries the long-range terms "
                "natively, so the interaction graph never has to match the state's",
}


#: The one propagator the multi-site models have: Schroedinger-picture tree TEBD.
#: Named to distinguish it from the interaction-picture ``tree-tebd`` of the
#: binary-tree geometry, which is a different engine on a different graph.
STATIC_TREE_TEBD = "tree-tebd-static"

#: The multichannel model's interaction-picture propagator: a swap-network TEBD
#: sweep against the matrix-valued time-dependent coupling.  Named rather than
#: shared with ``tebd`` because the builder is a different class (the coupling is
#: a matrix per mode, not a scalar times one operator).
MULTICHANNEL_IP = "multichannel-ip"


def _m(name, frame, models, engine, driver="", fixed_bond=False, integrator="",
       geometry="path", basis=""):
    return Method(name, frame, models, engine, driver, fixed_bond,
                  integrator or driver, geometry, basis)


_SB = ("system-bath",)

#: Every method, declared once.  Order matters: :attr:`Model.frames` reads method
#: order off this table, and ``methods_of(model, frame)[0]`` is the default a
#: bath-selected model falls back to.
METHODS = {s.name: s for s in [
    # -- system-bath, Schroedinger: the one cell where the basis is a choice ---
    _m("mpo-tdvp1", "schrodinger", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True, basis="chain"),
    _m("mpo-tdvp2", "schrodinger", _SB, "mpo-tdvp", "tdvp2", basis="chain"),
    _m("mpo-dtdvp", "schrodinger", _SB, "mpo-tdvp",
       "dtdvp", fixed_bond=True, basis="chain"),
    _m("mpo-star-tdvp1", "schrodinger", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True, basis="star"),
    _m("mpo-star-tdvp2", "schrodinger", _SB, "mpo-tdvp", "tdvp2", basis="star"),
    # -- system-bath, interaction picture (basis=star, forced) ----------------
    _m("tebd", "interaction", _SB, "swap-tebd", integrator="tebd"),
    _m("trotter-mpo", "interaction", _SB, "displacement-mpo",
       integrator="trotter-mpo"),
    _m("mpo-ip-tdvp1", "interaction", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("mpo-ip-tdvp2", "interaction", _SB, "mpo-tdvp", "tdvp2"),
    # -- system-bath, interaction picture, on a balanced binary tree ----------
    _m("tree-tdvp", "interaction", _SB, "modetree",
       "run_tree_tdvp", fixed_bond=True, integrator="tdvp1",
       geometry="binary-tree"),
    _m("tree-tdvp2", "interaction", _SB, "modetree",
       "run_tree_tdvp2", integrator="tdvp2", geometry="binary-tree"),
    _m("tree-tebd", "interaction", _SB, "modetree",
       "run_tree_tebd", integrator="tebd", geometry="binary-tree"),
    # -- system-bath, polaron frame (basis=chain, forced) ---------------------
    _m("polaron", "polaron", _SB, "polaron-tebd", integrator="tebd"),
    _m("polaron-tdvp1", "polaron", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("polaron-tdvp2", "polaron", _SB, "mpo-tdvp", "tdvp2"),
    _m("polaron-dtdvp", "polaron", _SB, "mpo-tdvp",
       "dtdvp", fixed_bond=True),
    # -- the static tree engine: one propagator, three topologies -------------
    _m(STATIC_TREE_TEBD, "schrodinger", ("multichannel", "comb", "site-tree"),
       "static-tree-tebd", integrator="tebd", geometry="comb-tree"),
    _m(MULTICHANNEL_IP, "interaction", ("multichannel",),
       "swap-tebd", integrator="tebd"),
]}

#: Derived from :data:`METHODS` -- was a hand-maintained set in
#: ``models/system_bath.py`` that the tests had to import privately.
FIXED_BOND_METHODS = frozenset(n for n, s in METHODS.items() if s.fixed_bond)


MODELS = {
    "system-bath": Model(
        key="system-bath", label="system-bath",
        blurb="One system site coupled to one bath through one coupling operator.  "
              "The most developed model: all three frames, both bases, both "
              "single-system geometries and the whole integrator family.  What used "
              "to be three separate 'models' (chain, star, mode-tree) is this one "
              "model at (basis=chain, path), (basis=star, path) and (star, "
              "binary-tree).",
        cls="SystemBath"),
    "multichannel": Model(
        key="multichannel", label="multichannel system-bath",
        blurb="One system site, one bath, coupled through *several* operators "
              "that share the same modes -- so the channels are cross-correlated, "
              "unlike independent baths.  Selected by giving the Bath a list of "
              "couplings, not by a method name.",
        cls="SystemBath",
        selected_by="bath"),
    "comb": Model(
        key="comb", label="fishbone / comb",
        blurb="Several system sites on a 1D backbone, each carrying one or two "
              "baths -- the fishbone.  A specialization of ``site-tree`` to a "
              "linear backbone.",
        cls="Fishbone",
        gaps={
            "interaction": "not implemented for multi-site models.",
            "polaron": "not implemented for multi-site models.",
        }),
    "site-tree": Model(
        key="site-tree", label="tree of sites",
        blurb="Several system sites wired into any loop-free tree, each carrying "
              "zero or more baths.  The most general geometry.  Distinct from a "
              "``binary-tree`` geometry, where it is a *single* system's bath modes "
              "that form the tree.",
        cls="TreeFishbone",
        gaps={
            "interaction": "not implemented for multi-site models.",
            "polaron": "not implemented for multi-site models.",
        }),
}


# -- constraints -------------------------------------------------------------
def why_not(model_key, frame=None, *, basis=None, geometry=None):
    """Why a combination is unavailable, or ``None`` if it exists.

    Answers with the *physics* where a constraint rules the combination out, and
    with the recorded gap where it is merely unimplemented -- so "impossible" and
    "nobody wrote it yet" stay distinguishable, which was the original point of
    ``Model.gaps`` and is now mostly derived instead of written down.
    """
    if frame is not None:
        by_frame = _FRAME_BASIS.get(frame)
        by_model = _MODEL_BASIS.get(model_key)
        # The frame and the model each force a basis, and they disagree: the
        # combination is impossible, and neither one alone explains why.
        if by_frame and by_model and by_frame != by_model:
            return (f"the {FRAMES[frame].label} forces basis={by_frame!r} while the "
                    f"{MODELS[model_key].label} model forces basis={by_model!r}, so "
                    f"there is no basis left to write H in.  "
                    + _BASIS_REASON[(frame, by_model)])
        need = by_frame or by_model
        if basis is not None and need is not None and basis != need:
            reason = _BASIS_REASON.get((frame, basis))
            if reason is None:                      # forced by the model, not the frame
                reason = (f"the {MODELS[model_key].label} model forces "
                          f"basis={need!r} ({BASES[need].split(':')[0]}).")
            return reason
        if geometry == "binary-tree" and need == "chain":
            return _NO_MODE_MODE
    if basis == "chain" and geometry == "binary-tree":
        return _NO_MODE_MODE
    if geometry is not None and model_key in ("comb", "site-tree") \
            and geometry != "comb-tree":
        return (f"{MODELS[model_key].label} puts each site's bath on its own "
                f"branch, so its state graph is always 'comb-tree'.")
    if frame is not None:
        m = MODELS.get(model_key)
        if m is not None and frame not in m.frames:
            return m.gaps.get(frame)
    return None


def method_spec(name, model_key=None):
    """The :class:`Method` for ``name``, or a :class:`ValueError` naming what is."""
    spec = METHODS.get(name)
    if spec is None or (model_key is not None and model_key not in spec.models):
        raise unknown_method_error(name, model_key)
    return spec


def combinations(model_keys):
    """``[(model, frame, basis, geometry, integrator, method)]`` for these models."""
    keys = set(model_keys)
    return [(mk, s.frame, s.basis_for(mk), s.geometry, s.integrator, s.name)
            for s in METHODS.values() for mk in s.models if mk in keys]


#: The axes :func:`resolve` filters on, in the order they appear in a combination
#: tuple.  Named once so the filter, the error message and the table agree.
_AXES = ("model", "frame", "basis", "geometry", "integrator")


def resolve(model_keys, *, method=None, **axes):
    """The :class:`Method` selected by either spelling.

    ``method=`` names a combination; the five axes give it directly.  The axes are
    the real structure -- ``"mpo-ip-tdvp2"`` *is* ``(system-bath, interaction, star,
    path, tdvp2)`` -- so this is one lookup either way, and mixing the two spellings
    is rejected rather than silently resolved.
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
            "give either method= or the axes (model / frame / basis / geometry / "
            "integrator), not both -- a method name already fixes all five.")
    hit = [c for c in avail
           if all(c[_AXES.index(k)] == v for k, v in given.items())]
    asked = ", ".join(f"{k}={given[k]!r}" for k in _AXES if k in given)
    if not hit:
        # A constraint violation has a physical reason; say it rather than
        # printing the table and leaving the reader to spot what is missing.
        for mk in sorted(set(model_keys)):
            why = why_not(given.get("model", mk), given.get("frame"),
                          basis=given.get("basis"),
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
        if key in BASES:
            extra = (f"  ({key!r} is a bath *basis*, not a model -- use "
                     f"model='system-bath', basis={key!r}.)")
        elif key == "mode-tree":
            extra = ("  ('mode-tree' was a state *geometry*, not a model -- use "
                     "model='system-bath', geometry='binary-tree'.)")
        raise KeyError(f"unknown model {key!r}; available: "
                       f"{', '.join(sorted(MODELS))}.{extra}") from None


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
        why = why_not(model_key, frame)
        raise KeyError(
            f"model {model_key!r} has no {frame!r} frame"
            + (f": {why}" if why else "")
            + f"  (available: {', '.join(frames_of(model_key))})")
    return tuple(m.frames[frame])


def all_methods():
    """Every method name in the taxonomy."""
    return tuple(sorted({m for mo in MODELS.values() for m in mo.methods()}))


#: Derived ``method -> (frame, model)``.  Kept because it is the natural key for
#: "which siblings share this method's frame", but note it is now a *projection*:
#: two methods can share a ``(frame, model)`` and still differ, in basis
#: (``mpo-tdvp2`` vs ``mpo-star-tdvp2``) or geometry (``mpo-ip-tdvp2`` vs
#: ``tree-tdvp2``).  That is the point -- those used to be different "models".
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
            for name in m.frames[frame_key]:
                s = METHODS[name]
                axes = (f"basis={s.basis_for(key):<5} "
                        f"geometry={s.geometry:<11} "
                        f"integrator={s.integrator}")
                lines.append(f"    {FRAMES[frame_key].label:20s} "
                             f"{name:<17} {axes}")
        for frame_key, why in m.gaps.items():
            lines.append(f"    {FRAMES[frame_key].label:20s} -- absent: {why}")
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
