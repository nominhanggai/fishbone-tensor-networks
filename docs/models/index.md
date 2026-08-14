# Models

A model defines the physical topology: what is coupled to what. How the
Hamiltonian is written is selected separately by `representation`, and the
tensor-network geometry is selected by `state_geometry`.

| model | class | physical setup |
|---|---|---|
| `system-bath` | {py:class}`~fishbonett.models.system_bath.SystemBath` | one system, one bath, one coupling operator |
| `multichannel` | {py:class}`~fishbonett.models.system_bath.SystemBath` | one system, shared bath modes, several coupling operators |
| `comb` | {py:class}`~fishbonett.models.fishbone.Fishbone` | several system sites on a line, baths per site |
| `site-tree` | {py:class}`~fishbonett.models.fishbone.TreeFishbone` | several system sites on any loop-free graph, baths per site |

`chain`, `star`, and `mode-tree` are not models. Chain and star occur only inside
complete representation names; a balanced mode tree is a binary tree tensor-network
geometry.

The registry records every implemented combination and explains unavailable
ones:

```python
from fishbonett.models.registry import describe_taxonomy

print(describe_taxonomy())
```

## Selection

For a single coupling operator, `SystemBath` selects `system-bath`. Passing a
list of coupling operators selects `multichannel`, because the operators act on
the same finite bath modes and their noise is cross-correlated.

```python
single = SystemBath(h=H, coupling=O, bath=bath)
shared = SystemBath(h=H, coupling=[O1, O2], bath=multichannel_bath)
```

Use `method=` or the four public axes to choose propagation. See
{doc}`/methods/index` for all six representation names and their methods.

```{toctree}
:maxdepth: 1

spin_boson
fishbone
composite_multichannel
observables
```
