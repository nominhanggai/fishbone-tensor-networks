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

The six models and the taxonomy that relates them to frames and propagators.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.models.registry
   fishbonett.models.system_bath
   fishbonett.models.fishbone
   fishbonett.models.result
```

## Geometry: the state ansätze

The 1D chain, the comb and the general tree are separate implementations rather
than a class hierarchy — each is optimized for its own geometry, and they share
the truncation policy instead of a base class.  These hold tensors only; the
models that drive them are above.

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

