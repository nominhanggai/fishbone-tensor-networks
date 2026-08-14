"""The taxonomy: the four axes a run is made of, and which combinations exist.

A run is **four independent choices**, and the point of this module is that they
are independent -- which they were not when three of them were mashed into one
string called the "model"::

    model     what is coupled to what      system-bath | multichannel | comb | site-tree
    frame     how H is written down        schrodinger-chain | schrodinger-star
                                           | interaction-star | polaron-chain
    geometry  the graph the state lives on path | binary-tree | comb-tree
    integrator  how a step is taken        tebd | tdvp1 | tdvp2 | dtdvp | trotter-mpo

.. rubric:: A frame is a picture *and* a mode basis

Both halves are a choice of how to write ``H`` down, so they are one axis and the
frame names carry both.  All ``3 x 2`` combinations are real frames -- what is
absent is *unimplemented*, recorded in :attr:`Model.gaps`, not impossible:

* ``interaction-chain`` **exists**, and is what this package actually runs for
  ``tebd`` / ``trotter-mpo`` / ``mpo-ip-tdvp*`` / ``tree-*``.  ``H_B`` is
  tridiagonal rather than diagonal in the chain basis, but still quadratic, so
  rotating it out is well defined: each chain mode evolves into a superposition of
  chain modes.  Measured, the coupling ``|d_n(t)|`` it feeds the propagator starts
  as ``(|V|, 0, ..., 0)`` -- the system touching ``c0`` alone, which is the chain --
  and spreads outward with ``t``.  Locality is what is lost, not existence.
* ``polaron-star`` **exists** too: the textbook Lang-Firsov transform is *defined*
  per star mode, ``prod_k D_k(g_k sigma_z / w_k)``.  It is the chain version that
  needs the ``J/w^2`` trick.  It is simply not implemented, because displacing every
  mode conditionally entangles the system with all of them at once.

``schrodinger-chain`` / ``schrodinger-star`` and ``interaction-chain`` /
``interaction-star`` are each one picture in two bases.  The bases are related by an
orthogonal (Lanczos) transform, so the physics is identical and only the MPS cost
differs -- which is the whole reason the basis is a representation choice rather
than a model.

``geometry`` stays a separate axis because it genuinely is one: ``mpo-ip-tdvp1`` and
``tree-tdvp`` are the *same frame*, laid on a path and on a balanced binary tree,
which is why they produce identical numbers rather than merely close ones.
``mode-tree`` used to be listed as a model for that difference; it was never a
model.

:data:`METHODS` is therefore the single source of truth for both the taxonomy and
the **dispatch**: each row carries its four axes plus the engine that realizes it,
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
           "GEOMETRIES", "APPLICATIONS", "FIXED_BOND_METHODS",
           "why_not", "models_of", "frames_of",
           "methods_of", "all_methods", "model", "method_spec", "resolve",
           "combinations", "METHOD_FRAMES", "pictures_of",
           "methods_by_frame", "frame_label", "describe_taxonomy",
           "unknown_method_error", "STATIC_TREE_TEBD", "MULTICHANNEL_IP",
           "MULTICHANNEL_STATIC"]


# -- frames ------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """One way of writing the Hamiltonian down: a picture **and** a mode basis.

    Both are choices about how ``H`` is written rather than about what is being
    modelled or how it is stepped, so they are one axis.  All ``3 x 2`` of them are
    real frames; which ones this package *implements* is a separate question, and
    the answer lives in :attr:`Model.gaps`.
    """
    key: str
    label: str
    blurb: str
    static: bool          # is H time-independent?  (decides TDVP applicability)
    #: which unitary is rotated out -- ``schrodinger`` / ``interaction`` / ``polaron``
    picture: str = ""
    #: which modes H is written in -- ``chain`` (Lanczos/TEDOPA, nearest-neighbour
    #: hoppings) or ``star`` (the raw discretization, no mode-mode terms)
    basis: str = ""

    @property
    def diagonal_bath(self):
        """Does the free bath contribute **no two-mode terms** in this frame?

        The one predicate the taxonomy turns on, because it has *both* of the
        consequences that matter, and they pull against each other -- **you cannot
        have a local bath and a local coupling at once**:

        * yes ⇒ nothing couples mode to mode, so the system couples to *every* mode
          and the interaction graph is a star.  On a ``path`` state realized with
          gates that costs a swap network (:attr:`Method.application`).
        * no ⇒ the chain hoppings survive, which are nearest-neighbour on a path
          (good) but long-range on a balanced binary tree (see ``_NO_MODE_MODE``).

        Rotating out ``H_B`` removes the mode-mode terms whatever basis you write
        them in -- which is why this keys on the *picture* as well as the basis, and
        why ``interaction-chain`` swaps despite being a chain.
        """
        return self.picture == "interaction" or self.basis == "star"


FRAMES = {
    "schrodinger-chain": Frame(
        "schrodinger-chain", "Schrodinger picture, chain",
        "Nothing rotated out, bath chain-mapped.  H is time-independent and its MPO "
        "is built once, so TDVP conserves energy -- but the state carries the full "
        "system-bath correlation, giving the largest bond dimensions.  The chain's "
        "nearest-neighbour hoppings are what an MPS is good at, and the system "
        "touches only c0.",
        static=True, picture="schrodinger", basis="chain"),
    "schrodinger-star": Frame(
        "schrodinger-star", "Schrodinger picture, star",
        "Nothing rotated out, no chain mapping: every mode couples straight to the "
        "system.  No mode-mode terms, but no locality for the MPS to exploit "
        "either.  Static, so still one MPO built once.",
        static=True, picture="schrodinger", basis="star"),
    "interaction-chain": Frame(
        "interaction-chain", "interaction picture, chain",
        "The free-bath evolution is rotated out, leaving only the coupling, and the "
        "result is expressed back in the chain modes.  H_B is tridiagonal rather "
        "than diagonal here, but still quadratic, so the rotation is well defined: "
        "each chain mode evolves into a superposition of chain modes.  What is lost "
        "is locality, not existence -- the coupling d_n(t) starts concentrated on c0 "
        "at t=0 and spreads outward, so this is 'no longer a chain' in the only "
        "sense that matters to an MPS.  Entanglement is much smaller than the "
        "Schroedinger picture's, but H is time-dependent so gates/MPOs are rebuilt "
        "every step.  All the coupling terms commute here, which is what makes the "
        "exact conditional-displacement propagator possible.",
        static=False, picture="interaction", basis="chain"),
    "interaction-star": Frame(
        "interaction-star", "interaction picture, star",
        "The same rotation as interaction-chain, left in the star modes instead of "
        "being rotated back: the coupling of mode k is simply V_k e^{-i w_k t}.  "
        "Reaches the same trajectory through a completely different coupling "
        "vector, so it is an independent check on the chain route rather than a "
        "restatement of it.  Which is cheaper is not settled -- the guess that the "
        "chain wins (its coupling starts on c0 and spreads) is not what measuring "
        "shows.  Also the only frame available to the multichannel model, whose "
        "shared modes cannot be chain-mapped at all.",
        static=False, picture="interaction", basis="star"),
    "polaron-chain": Frame(
        "polaron-chain", "polaron frame, chain",
        "The static part of the coupling is absorbed into a bath displacement, "
        "leaving a free chain plus a dressed tunneling term.  Static like the "
        "Schroedinger picture *and* low-entanglement like the interaction "
        "picture; needs int J/w^2 finite.  Populations are frame-invariant, "
        "coherences must be un-dressed.  The J/w^2 reweighting is what localizes "
        "the displacement on c0.",
        static=True, picture="polaron", basis="chain"),
    "polaron-star": Frame(
        "polaron-star", "polaron frame, star",
        "The textbook Lang-Firsov transform, which is *defined* per star mode: "
        "prod_k D_k(g_k sigma_z / w_k).  Perfectly well defined -- it is the chain "
        "version that needs the J/w^2 trick to localize onto a single site.  Not "
        "implemented here because it is not useful on an MPS: displacing every mode "
        "conditionally on the system entangles the system with all of them at once, "
        "which is exactly what the chain version avoids.",
        static=True, picture="polaron", basis="star"),
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


#: Why the binary tree needs a frame with no mode-mode terms (:attr:`Frame.
#: diagonal_bath`).  Not an oversight: the tree is worth having *because* nothing
#: couples mode to mode, so every mode hangs off the system independently and the
#: only question is how deep the bonds are.
#:
#: Note this is about the *frame*, not the basis -- ``interaction-chain`` qualifies
#: (it rotates ``H_B`` away entirely) and is what the ``tree-*`` methods use.
_NO_MODE_MODE = (
    "the balanced binary tree pays off only when there are no mode-mode terms.  "
    "This frame keeps the chain hoppings, which are nearest-neighbour on a path but "
    "long-range on that tree (only half of the chain-adjacent pairs are "
    "tree-adjacent; the rest span up to 2*log2(N) edges -- measured: 10 edges at "
    "N=32), so the MPO bond grows and the tree loses to the plain path.  Reordering "
    "the leaves to make the chain local just turns the tree back into a path.")


# -- models ------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """A physical setup -- **only** the topology, now that the mode basis lives in
    the frame and the state graph in ``geometry``.

    ``gaps`` maps an absent frame key to the reason it is absent.  The entries that
    used to say "rejected because the polaron displacement has nowhere to localize"
    are gone: there is no ``polaron-star`` frame to be absent from, so that whole
    class of gap stopped existing rather than needing a reason (see
    :data:`NOT_FRAMES`).  What is left is per-model work nobody has done.
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
    """One realizable combination of the four axes.

    This is the single source of truth: what exists **and** how it is dispatched.
    ``models`` is a tuple because one engine can serve several topologies -- the
    static tree TEBD runs the comb and the site-tree.
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
    #: ``"trotter-mpo"``.  Unique within a ``(model, frame, geometry)``, which is
    #: what lets ``run`` be called by the axes instead of by a name that mashes
    #: them together.
    integrator: str = ""
    #: The state graph -- see :data:`GEOMETRIES`.
    geometry: str = "path"

    @property
    def basis(self):
        """The bath mode basis, which the frame carries."""
        return FRAMES[self.frame].basis

    @property
    def picture(self):
        """Which unitary is rotated out, which the frame carries."""
        return FRAMES[self.frame].picture

    @property
    def application(self):
        """How H's **interaction graph** is realized on the state's graph.

        Derived, not declared: it is a *consequence* of the other axes.  A frame can
        make ``H`` non-local relative to the state -- the interaction picture couples
        every mode to the system, a star, while the state is a path -- and this
        records what pays for that.  See :data:`APPLICATIONS`.

        Keys on :attr:`Frame.diagonal_bath`, **not** on the basis: ``tebd`` is
        ``interaction-chain`` and still needs a swap network, because it is rotating
        out ``H_B`` that spreads the coupling over every mode, not the choice of
        modes to write it in.
        """
        if self.integrator != "tebd" or self.geometry == "binary-tree":
            return "operator"
        if self.geometry == "path" and FRAMES[self.frame].diagonal_bath:
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


#: The one propagator the multi-site models have: Schroedinger-picture tree TEBD on
#: their chain-mapped baths.  Named to distinguish it from the interaction-picture
#: ``tree-tebd`` of the binary-tree geometry, a different engine on a different graph.
STATIC_TREE_TEBD = "tree-tebd-static"

#: The multichannel model's Schroedinger-picture propagator.  The *same engine* as
#: :data:`STATIC_TREE_TEBD`, but a separate row because it is a different **frame**:
#: the shared-mode star cannot be chain-mapped, so this is ``schrodinger-star``
#: where the multi-site models are ``schrodinger-chain``.  See
#: ``frames/schrodinger.py``, which picks ``star_terms`` exactly when the bath is
#: multichannel -- the split was always in the code, just not in the table.
MULTICHANNEL_STATIC = "multichannel-static"

#: The multichannel model's interaction-picture propagator: a swap-network TEBD
#: sweep against the matrix-valued time-dependent coupling.  Named rather than
#: shared with ``tebd`` because the builder is a different class (the coupling is
#: a matrix per mode, not a scalar times one operator).
MULTICHANNEL_IP = "multichannel-ip"


def _m(name, frame, models, engine, driver="", fixed_bond=False, integrator="",
       geometry="path"):
    return Method(name, frame, models, engine, driver, fixed_bond,
                  integrator or driver, geometry)


_SB = ("system-bath",)

#: Every method, declared once.  Order matters: :attr:`Model.frames` reads method
#: order off this table, and ``methods_of(model, frame)[0]`` is the default a
#: bath-selected model falls back to.
METHODS = {s.name: s for s in [
    # -- system-bath, Schroedinger picture: the one picture with both bases ----
    _m("mpo-tdvp1", "schrodinger-chain", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("mpo-tdvp2", "schrodinger-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("mpo-dtdvp", "schrodinger-chain", _SB, "mpo-tdvp",
       "dtdvp", fixed_bond=True),
    _m("mpo-star-tdvp1", "schrodinger-star", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("mpo-star-tdvp2", "schrodinger-star", _SB, "mpo-tdvp", "tdvp2"),
    # -- system-bath, interaction picture, on a path --------------------------
    # `interaction-chain`, not `-star`: `mode_couplings` rotates the star phases
    # back into the chain modes, so d_n(0) = (|V|, 0, ..., 0) -- the system on c0
    # alone -- and spreads outward with t.  Measured, not assumed.
    _m("tebd", "interaction-chain", _SB, "swap-tebd", integrator="tebd"),
    _m("trotter-mpo", "interaction-chain", _SB, "displacement-mpo",
       integrator="trotter-mpo"),
    _m("mpo-ip-tdvp1", "interaction-chain", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("mpo-ip-tdvp2", "interaction-chain", _SB, "mpo-tdvp", "tdvp2"),
    # -- ...the same rotation left in the star modes --------------------------
    _m("mpo-ip-star-tdvp1", "interaction-star", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("mpo-ip-star-tdvp2", "interaction-star", _SB, "mpo-tdvp", "tdvp2"),
    # -- ...and the chain frame on a balanced binary tree ---------------------
    _m("tree-tdvp", "interaction-chain", _SB, "modetree",
       "run_tree_tdvp", fixed_bond=True, integrator="tdvp1",
       geometry="binary-tree"),
    _m("tree-tdvp2", "interaction-chain", _SB, "modetree",
       "run_tree_tdvp2", integrator="tdvp2", geometry="binary-tree"),
    _m("tree-tebd", "interaction-chain", _SB, "modetree",
       "run_tree_tebd", integrator="tebd", geometry="binary-tree"),
    # -- system-bath, polaron frame -------------------------------------------
    _m("polaron", "polaron-chain", _SB, "polaron-tebd", integrator="tebd"),
    _m("polaron-tdvp1", "polaron-chain", _SB, "mpo-tdvp",
       "tdvp1", fixed_bond=True),
    _m("polaron-tdvp2", "polaron-chain", _SB, "mpo-tdvp", "tdvp2"),
    _m("polaron-dtdvp", "polaron-chain", _SB, "mpo-tdvp",
       "dtdvp", fixed_bond=True),
    # -- the static tree engine: one engine, two frames, three topologies -----
    _m(STATIC_TREE_TEBD, "schrodinger-chain", ("comb", "site-tree"),
       "static-tree-tebd", integrator="tebd", geometry="comb-tree"),
    _m(MULTICHANNEL_STATIC, "schrodinger-star", ("multichannel",),
       "static-tree-tebd", integrator="tebd", geometry="comb-tree"),
    _m(MULTICHANNEL_IP, "interaction-star", ("multichannel",),
       "swap-tebd", integrator="tebd"),
]}

#: Derived from :data:`METHODS` -- was a hand-maintained set in
#: ``models/system_bath.py`` that the tests had to import privately.
FIXED_BOND_METHODS = frozenset(n for n, s in METHODS.items() if s.fixed_bond)


#: The multi-site models are wired for one frame only.  Their baths are chain-mapped
#: per site, and nothing rotates a picture out per site.
_MULTISITE = "not implemented for multi-site models."

#: The multichannel model's baths cannot be chain-mapped at all: the channels share
#: one set of modes, and a Lanczos chain exists per coupling operator, not per
#: cross-correlated set of them.  So all three chain frames are out for a reason
#: that is about the *model*, not the picture.
_NO_CHAIN = ("the channels share one set of modes, so there is no chain mapping to "
             "make: Lanczos gives a chain per coupling operator, and these are "
             "cross-correlated.  The shared-mode star is the only representation.")

#: Why ``polaron-star`` is unimplemented.  A real frame -- the textbook Lang-Firsov
#: transform is written this way -- but not a useful one here, which is a different
#: statement from the one this registry used to make about it.
_POLARON_STAR = (
    "possible, not implemented.  The Lang-Firsov displacement is defined per star "
    "mode (prod_k D_k(g_k sigma_z / w_k)) -- it is the chain version that needs the "
    "J/w^2 reweighting to localize on c0 -- so this is the *more* standard way to "
    "write the transform, not a degenerate one.  Nobody has wired it; the guess "
    "that dressing every mode would cost more than dressing c0 is untested, and "
    "the same guess for interaction-star turned out not to hold.  Use "
    "'polaron-chain'.")

MODELS = {
    "system-bath": Model(
        key="system-bath", label="system-bath",
        blurb="One system site coupled to one bath through one coupling operator.  "
              "The most developed model: five of the six frames, both single-system "
              "geometries and the whole integrator family.  What used to be three "
              "separate 'models' (chain, star, mode-tree) is this one model in the "
              "schrodinger-chain, schrodinger-star and interaction-chain frames, the "
              "last on two geometries.",
        cls="SystemBath",
        gaps={"polaron-star": _POLARON_STAR}),
    "multichannel": Model(
        key="multichannel", label="multichannel system-bath",
        blurb="One system site, one bath, coupled through *several* operators "
              "that share the same modes -- so the channels are cross-correlated, "
              "unlike independent baths.  Selected by giving the Bath a list of "
              "couplings, not by a method name.",
        cls="SystemBath",
        gaps={"schrodinger-chain": _NO_CHAIN, "interaction-chain": _NO_CHAIN,
              "polaron-chain": _NO_CHAIN, "polaron-star": _POLARON_STAR},
        selected_by="bath"),
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
def why_not(model_key, frame=None, *, geometry=None):
    """Why a combination is unavailable, or ``None`` if it exists.

    Every one of the six frames is a real frame, so nothing here says "impossible".
    What it reports is per-model work nobody has done (:attr:`Model.gaps`, which
    says *why* it would or would not pay), plus the one genuine constraint: a
    balanced binary tree needs a frame with no mode-mode terms.
    """
    if geometry == "binary-tree" and frame is not None \
            and frame in FRAMES and not FRAMES[frame].diagonal_bath:
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


def pictures_of(picture):
    """Every frame key with this picture -- always both bases, since all six exist.

    Lets ``frame="polaron"`` still mean something.  :func:`resolve` narrows the
    result to frames that actually have methods, so a bare picture resolves whenever
    only one of its two bases is implemented (``polaron`` -> ``polaron-chain``) and
    is reported as ambiguous when both are (``schrodinger``, ``interaction``).
    """
    return tuple(k for k, f in FRAMES.items() if f.picture == picture)


def method_spec(name, model_key=None):
    """The :class:`Method` for ``name``, or a :class:`ValueError` naming what is."""
    spec = METHODS.get(name)
    if spec is None or (model_key is not None and model_key not in spec.models):
        raise unknown_method_error(name, model_key)
    return spec


def combinations(model_keys):
    """``[(model, frame, geometry, integrator, method)]`` for these models."""
    keys = set(model_keys)
    return [(mk, s.frame, s.geometry, s.integrator, s.name)
            for s in METHODS.values() for mk in s.models if mk in keys]


#: The axes :func:`resolve` filters on, in the order they appear in a combination
#: tuple.  Named once so the filter, the error message and the table agree.
_AXES = ("model", "frame", "geometry", "integrator")


def resolve(model_keys, *, method=None, **axes):
    """The :class:`Method` selected by either spelling.

    ``method=`` names a combination; the four axes give it directly.  The axes are
    the real structure -- ``"mpo-ip-tdvp2"`` *is* ``(system-bath, interaction-star,
    path, tdvp2)`` -- so this is one lookup either way, and mixing the two spellings
    is rejected rather than silently resolved.

    ``frame=`` also accepts a bare *picture*: ``"polaron"`` names one frame and
    resolves, ``"schrodinger"`` names two and is reported as ambiguous.
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
            "give either method= or the axes (model / frame / geometry / "
            "integrator), not both -- a method name already fixes all four.")

    # a bare picture stands for its frames; if that is one frame it just resolves
    frames = None
    if "frame" in given and given["frame"] not in FRAMES:
        by_picture = pictures_of(given["frame"])
        if by_picture:
            frames = set(by_picture)

    def matches(c):
        for k, v in given.items():
            if k == "frame" and frames is not None:
                if c[1] not in frames:
                    return False
            elif c[_AXES.index(k)] != v:
                return False
        return True

    hit = [c for c in avail if matches(c)]
    asked = ", ".join(f"{k}={given[k]!r}" for k in _AXES if k in given)
    if not hit:
        # An unnameable frame or a recorded gap has a reason; say it rather than
        # printing the table and leaving the reader to spot what is missing.
        for mk in sorted(set(model_keys)):
            why = why_not(given.get("model", mk), given.get("frame"),
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
            extra = (f"  ({key!r} is half of a *frame*, not a model -- the frames "
                     f"are {', '.join(k for k in FRAMES if k.endswith(key))}.)")
        elif key == "mode-tree":
            extra = ("  ('mode-tree' was a state *geometry*, not a model -- use "
                     "model='system-bath', geometry='binary-tree'.)")
        raise KeyError(f"unknown model {key!r}; available: "
                       f"{', '.join(sorted(MODELS))}.{extra}") from None


def models_of(method):
    """Every model key offering ``method``.

    A tuple rather than a single key because the static tree engine
    (``tree-tebd-static``) genuinely serves both multi-site models -- the same
    propagator, different topologies.
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
#: "which siblings share this method's frame", but note it is a *projection*: two
#: methods can share a ``(frame, model)`` and still differ in geometry
#: (``mpo-ip-tdvp2`` vs ``tree-tdvp2``, both ``interaction-star``).  That is the
#: point -- those used to be called different "models".
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
                lines.append(f"    {frame_key:<18} {name:<19} "
                             f"geometry={s.geometry:<11} "
                             f"integrator={s.integrator}")
        for frame_key, why in m.gaps.items():
            lines.append(f"    {frame_key:<18} -- absent: {why}")
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
