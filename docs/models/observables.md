# Observables on system sites and represented bath modes

On the fishbone engines each entry of `observables` is one of four forms. Mix
them freely in a single `run`:

```python
import numpy as np

from fishbonett import BathMode
from fishbonett.operators import annihilate, sigma_z

a = annihilate(bath_dimension)
first_left_mode = BathMode(system_site=0, bath=0, mode=0)

res = fb.run(dt=0.02, t_max=1.0, trunc_eps=1e-4, observables={
    "sz":   sigma_z,                              # every matching system site
    "sz2":  (sigma_z, 2),                         # only system site 2
    "zz13": (np.kron(sigma_z, sigma_z), (1, 3)),  # two-system-site correlation
    "nL0":  (a.T @ a, first_left_mode),            # one represented bath mode
})
res.expect["sz"]    # (n_steps, n_sites)
res.expect["sz2"]   # (n_steps,)
res.expect["zz13"]  # (n_steps,)
res.expect["nL0"]   # (n_steps,)
```

- A bare `(d, d)` operator is measured on every system site whose dimension
  matches. `expect[name]` has shape `(n_steps, n_sites)`, with `NaN` where the
  operator dimension does not match a site.
- `(operator, i)` targets system site `i`.
- `(operator, (i, j, ...))` targets a composite of system sites. The operator
  dimension is the product of the target dimensions in the given order.
- `(operator, BathMode(...))` targets a represented bath coordinate. A
  `BathMode` can also appear beside system-site integers or other bath modes in
  a composite target. For example,
  `(np.kron(sigma_y, a + a.T), (0, first_left_mode))` measures a system--bath
  correlation.

System-site integers use the order passed to `Fishbone` or `TreeFishbone`.

## Bath addresses and representations

`BathMode(system_site, bath, mode)` is stable against internal tensor-node
numbering:

- `system_site` is the site to which the bath is attached;
- `bath` is zero based among the non-`None` baths attached to that site, in input
  order;
- `mode` is zero based in the selected Hamiltonian representation. It is a chain
  coordinate for a chain representation and a star coordinate for a star
  representation.

The supplied operator acts on that represented coordinate. Changing the
representation does not transform the operator automatically. A chain-mode
occupation and a star-mode occupation are therefore different observables even
when they originate from the same continuous spectral density.

The resolved layout is recorded for diagnostics and reproducible
postprocessing:

```python
res.meta["bath_branches"]
# ({"system_site": 0, "bath": 0, "representation": "schrodinger-chain",
#   "first_node": 4, "n_modes": 20, "phys_dim": 10,
#   "system_coupling": ...}, ...)

res.meta["observable_targets"]["nL0"]  # resolved tensor nodes, for example (4,)
```

Invalid site, bath, or mode indices and incompatible operator dimensions are
rejected while the simulation is compiled, before propagation starts.

## How composite operators are evaluated

A composite expectation value needs the joint reduced density matrix of the
requested nodes. The engine contracts only the minimal subtree spanning those
nodes. With the orthogonality centre inside that subtree, tensors outside it are
isometric and their outgoing bonds contract to identities. This is exposed on
the low-level state as
{py:meth}`TensorNetwork.joint_rdm <fishbonett.states.network.TensorNetwork.joint_rdm>`
and
{py:meth}`TensorNetwork.expectation <fishbonett.states.network.TensorNetwork.expectation>`.

The same contraction works for the 1D
{py:class}`~fishbonett.models.fishbone.Fishbone` and arbitrary
{py:class}`~fishbonett.models.fishbone.TreeFishbone` tensor-network geometries.

## Per-site reduced density matrices

Independently of `observables`, `result.rdm` contains the single-site reduced
density matrix of every system site at each recorded time:

```python
res.rdm.shape  # (n_steps, n_sites, d, d) for uniform system dimension d
res.rdm[t, i]  # d x d reduced density matrix of system site i
```

When system-site dimensions differ, `result.rdm` is an object array of matrices.
