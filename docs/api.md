# API reference

The high-level API separates four public choices:

```text
model -> representation -> state_geometry -> integrator
```

```python
result = sb.run(
    dt=0.02,
    t_max=2.0,
    representation="interaction-chain",
    state_geometry="mps",
    integrator="tdvp2",
)
```

The supported Hamiltonian representations are `schrodinger-chain`,
`schrodinger-star`, `interaction-chain`, `polaron-chain`, or `polaron-star`.

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
   fishbonett.targets
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
```

## Tensor-network states

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   fishbonett.states.network
   fishbonett.states.mps
   fishbonett.states.tree
   fishbonett.states.thermal
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
   fishbonett.contract
   fishbonett.linalg
   fishbonett.randomized
   fishbonett.spectral_densities
   fishbonett.rates.golden_rule
   fishbonett.rates.golden_rule_multi
   fishbonett.rates.mcmc
   fishbonett.rates.transfer_tensor
   fishbonett.diabatization
```

## Package tours

These package pages explain how the public layers fit together.

```{eval-rst}
.. automodule:: fishbonett
   :no-members:

.. automodule:: fishbonett.models
   :no-members:

.. automodule:: fishbonett.bath
   :no-members:

.. automodule:: fishbonett.representations
   :no-members:

.. automodule:: fishbonett.states
   :no-members:

.. automodule:: fishbonett.evolve
   :no-members:

.. automodule:: fishbonett.rates
   :no-members:
```
