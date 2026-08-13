# API reference

```{admonition} At a glance
:class: tip
- **Start at** {py:class}`~fishbonett.bath.spec.Bath` (what the bath is),
  {py:class}`~fishbonett.simulate.SystemBath` (what to propagate) and
  {py:class}`~fishbonett.linalg.Truncation` (how accurately) — all three are
  re-exported at the top level as `fishbonett.*`.
- **Package layout by subject** — `bath` turns a spectral density into chain
  parameters; `frames` writes the Hamiltonian in a chosen representation;
  `states` holds tensors; `evolve` holds propagation algorithms; `linalg` and
  `operators` hold the shared numerics.
- **Each module's docstring opens with its own "What's here" table**, so the
  generated pages below are browsable without reading the source.
```

## High-level interface

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.simulate
   fishbonett.treebone
```

## State ansätze and propagators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.states.mps
   fishbonett.states.comb
   fishbonett.evolve.tebd
   fishbonett.evolve.mpo_apply
   fishbonett.evolve.tdvp
   fishbonett.evolve.treetdvp
```

## Frames

One Hamiltonian, several representations — see {doc}`/methods/index` for the
frame taxonomy and which propagator suits which frame.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.frames.schrodinger
   fishbonett.frames.interaction_picture
   fishbonett.frames.polaron
   fishbonett.frames.multichannel
   fishbonett.frames.coolingchain
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
   fishbonett.bath.orthpol
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

