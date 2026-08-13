# API reference

Start from `from fishbonett import Bath, SystemBath, Truncation`.

A calculation is three **nested** choices, one subpackage each: **model** (what is
coupled to what — `models`), **frame** (how the Hamiltonian is written down —
`frames`), and **propagator** (how a step is taken — `evolve`).  The model fixes
the state geometry (`states`) and constrains the other two;
{py:mod}`fishbonett.models.registry` records which combinations exist and why the
rest do not.  `bath` turns a spectral density into the chain parameters every
model starts from, and `linalg`/`operators` hold the shared numerics.  The system
sits at **site 0** throughout, with the bath modes following, nearest first.

## Models — the physical setups

The six models and the taxonomy that relates them to frames and propagators.  A
model's two inputs are a `Bath` (the environment, below) and a
{py:class}`~fishbonett.system.System` — any Hermitian `h`, any Hermitian coupling
and an initial state, validated once so that no frame re-derives it.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.models.registry
   fishbonett.models.system_bath
   fishbonett.models.fishbone
   fishbonett.system
   fishbonett.models.result
```

## Geometry: the state ansätze

A chain is a tree, so both containers are the same object: tensors on a loop-free
graph.  `TensorNetwork` holds everything that follows from loop-freeness — topology,
the orthogonality centre, and the reduced density matrices read off it — and asks
each container for `tensor` / `set_tensor` / `neighbours` so it never has to know
whether the physical leg is stored in the middle (`vL, p, vR`) or last.  What stays
per-container is the gauge policy and the gate splitting: `SystemBathMPS` is in
Vidal form with LBO and a GPU path, `TreeTensorNetwork` is mixed-canonical.  These
hold tensors only; the models that drive them are above.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.states.network
   fishbonett.states.mps
   fishbonett.states.tree
```

## Propagators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.evolve.tebd
   fishbonett.evolve.sitetree
   fishbonett.evolve.mpo_apply
   fishbonett.evolve.tdvp
   fishbonett.evolve.modetree
```

## Frames

One Hamiltonian, several representations — see {doc}`/methods/index` for which
propagator suits which frame.  A frame's output is always
{py:class}`~fishbonett.frames.terms.LocalTerms`: one operator per node and one per
edge, static or a function of `t`.  That is the single interface between the physics
and the numerics — it is what lets one propagator serve every geometry, since it
describes the terms without saying what the graph looks like.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.frames.terms
   fishbonett.frames.gates
   fishbonett.frames.schrodinger
   fishbonett.frames.interaction_picture
   fishbonett.frames.polaron
   fishbonett.frames.multichannel
   fishbonett.frames.coolingchain
   fishbonett.frames.mpo
```

## The bath: specification, discretization, chain mapping

`Bath` (the specification) lives in {py:mod}`fishbonett.bath.spec`; the rest of
the subpackage turns it into chain parameters, in that order.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.bath.spec
   fishbonett.bath.chain
   fishbonett.bath.legendre
   fishbonett.bath.tedopa
   fishbonett.bath.lanczos
   fishbonett.bath.recurrence
   fishbonett.bath.auto
```

## Operators, linear algebra and spectral densities

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.operators
   fishbonett.linalg
   fishbonett.spectral_densities
```

## Rate theory and diabatization

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.rates
   fishbonett.diabatization
```

