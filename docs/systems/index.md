# Building models

Every model class takes a {py:class}`~fishbonett.bath.spec.Bath` and a system
Hamiltonian, and returns a {py:class}`~fishbonett.simulate.Result`.  Pick by
geometry:

| you want to model | use | page |
|-------------------|-----|------|
| one system (spin or `d`-level) coupled to a bath | {py:class}`~fishbonett.simulate.SystemBath` | {doc}`spin_boson` |
| a 1D chain of electronic sites, each with bath(s) | {py:class}`~fishbonett.simulate.Fishbone` | {doc}`fishbone` |
| electronic sites in any loop-free tree, each with bath(s) | {py:class}`~fishbonett.states.tree.TreeFishbone` | {doc}`fishbone` |
| a system with internal structure (e.g. spin **+** vibration) | `TreeFishbone` (one site per DOF) | {doc}`composite_multichannel` |
| one bath coupled through several operators (cross-correlated) | multichannel `Bath` | {doc}`composite_multichannel` |
| composite / correlation observables across sites | the observable spec | {doc}`observables` |

The bath itself — its spectral density, discretization, temperature and chain
mapping — is described separately in {doc}`../bath`.

```{toctree}
:maxdepth: 1
:hidden:

spin_boson
fishbone
composite_multichannel
observables
```
