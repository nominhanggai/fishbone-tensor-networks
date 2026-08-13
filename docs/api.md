# API reference

Start from `from fishbonett import Bath, SystemBath, Truncation`.

A calculation is three independent choices, one subpackage each: **frame** (what
the Hamiltonian looks like — `frames`), **geometry** (the shape of the tensor
network — `states`), and **propagator** (how a step is taken — `evolve`).
`simulate` wires them together, `bath` turns a spectral density into the chain
parameters they all start from, and `linalg`/`operators` hold the shared
numerics.  The system sits at **site 0** throughout, with the bath modes
following, nearest first.

## High-level interface

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.simulate
```

## Geometry: the state ansätze

The 1D chain, the comb and the general tree are separate implementations rather
than a class hierarchy — each is optimized for its own geometry, and they share
the truncation policy instead of a base class.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.states.mps
   fishbonett.states.comb
   fishbonett.states.tree
```

## Propagators

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.evolve.tebd
   fishbonett.evolve.mpo_apply
   fishbonett.evolve.tdvp
   fishbonett.evolve.treetdvp
```

## Frames

One Hamiltonian, several representations — see {doc}`/methods/index` for which
propagator suits which frame.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

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

