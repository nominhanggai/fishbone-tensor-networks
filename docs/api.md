# API reference

Start from `from fishbonett import Bath, SystemBath, Truncation`.  The package
is split by subject: `bath` = spectral density → chain parameters, `frames` =
Hamiltonian in a chosen representation, `states` = tensor network ansatz,
`evolve` = propagation algorithms, `linalg`/`operators` = shared numerics.

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

