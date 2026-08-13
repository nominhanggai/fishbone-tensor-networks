# API reference

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

   fishbonett.frames.hamiltonian
   fishbonett.frames.interaction_picture
   fishbonett.frames.polaron
   fishbonett.frames.multichannel
   fishbonett.frames.coolingchain
```

## Bath discretization and chain mapping

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.common
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

