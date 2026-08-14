# API reference

The high-level API separates four public choices:

```text
model -> representation -> state_geometry -> integrator
```

```python
result = sb.run(
    dt=0.02,
    t_max=2.0,
    representation="interaction-star",
    state_geometry="mps",
    integrator="tdvp2",
)
```

Each representation name is exact and complete:
`schrodinger-chain`, `schrodinger-star`, `interaction-chain`,
`interaction-star`, `polaron-chain`, or `polaron-star`.

## Models and planning

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.models.registry
   fishbonett.models.system_bath
   fishbonett.models.fishbone
   fishbonett.models.simulation
   fishbonett.models.propagate
   fishbonett.models.result
   fishbonett.system
```

## Representations

Representations contain mathematical Hamiltonian data and transformations and
materialize the TDVP MPOs, Trotter MPOs, or TEBD gates supported by that
Hamiltonian. They do not advance tensor-network states.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.representations.schrodinger
   fishbonett.representations.interaction
   fishbonett.representations.polaron
   fishbonett.representations.multichannel
   fishbonett.representations.coolingchain
```

## Tensor-network states

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.states.network
   fishbonett.states.mps
   fishbonett.states.tree
```

## Propagation algorithms

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

## Bath discretization and mapping

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.bath.spec
   fishbonett.bath.coupled
   fishbonett.bath.chain
   fishbonett.bath.legendre
   fishbonett.bath.tedopa
   fishbonett.bath.lanczos
   fishbonett.bath.recurrence
   fishbonett.bath.auto
   fishbonett.bath.conventions
```

## Operators, linear algebra, rates, and diabatization

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.operators
   fishbonett.linalg
   fishbonett.randomized
   fishbonett.spectral_densities
   fishbonett.rates
   fishbonett.diabatization
```
