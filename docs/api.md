# API reference

## State engines and models

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.mps
   fishbonett.fishbone
   fishbonett.model
   fishbonett.int_pic_hsb_spin_boson
```

## Bath discretization and chain mapping

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.common
   fishbonett.legendre_discretization
   fishbonett.lanczos
   fishbonett.recurrence_coefficients
```

## Operators and spectral densities

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.stuff
```

## Rate theory and diabatization

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.rates
   fishbonett.diabatization
```

## Interaction-picture, star and cooling drivers

The following modules provide the Hamiltonian / propagator builders for the
individual method families; they all share the {py:class}`fishbonett.mps.SpinBosonMPS`
engine.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.backwardSpinBoson
   fishbonett.backwardSpinBosonMultiChannel
   fishbonett.starSpinBoson
   fishbonett.coolingC_SpinBoson
   fishbonett.chainSpinBosonDiscrete
```
