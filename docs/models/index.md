# Models

A **model** is the physical setup: how many system sites there are, how they are
wired to each other, and how the bath is represented.  It is the first of three
nested choices — the model decides which **frames** are available, and the frame
decides which **propagators** apply ({doc}`../methods/index`).

Six models, three classes.  The first four are one class, because they differ
only in how the bath is represented, which `run(method=...)` selects:

| model | what it is | class | frames |
|---|---|---|---|
| `chain` | one system, one bath, modes chain-mapped into 1D | {py:class}`~fishbonett.models.system_bath.SystemBath` | Schrödinger, interaction, polaron |
| `star` | one system, one bath, **no** chain mapping | {py:class}`~fishbonett.models.system_bath.SystemBath` | interaction |
| `mode-tree` | one system, one bath, modes on a balanced binary tree | {py:class}`~fishbonett.models.system_bath.SystemBath` | interaction |
| `multichannel` | one system, one bath through several couplings on shared modes | {py:class}`~fishbonett.models.system_bath.SystemBath` | Schrödinger |
| `comb` | several sites on a 1D backbone, baths per site — the fishbone | {py:class}`~fishbonett.models.fishbone.Fishbone` | Schrödinger |
| `site-tree` | several sites in any loop-free tree, baths per site | {py:class}`~fishbonett.models.fishbone.TreeFishbone` | Schrödinger |

`chain` is the developed one: all three frames and the whole propagator family.
The rest have one frame each today; the registry records *why* each missing
combination is missing, so you can tell "impossible" from "nobody wrote it yet":

```python
from fishbonett.models.registry import describe_taxonomy
print(describe_taxonomy())
```

```{admonition} Two different trees
:class: warning
`mode-tree` and `site-tree` are unrelated.  In `mode-tree` a **single** system's
bath modes are arranged on a tree, to keep the high-bond region $O(\log N)$ edges
deep — the tree is an efficiency trick inside a one-system model.  In `site-tree`
it is the **system sites themselves** that form a tree, each with its own bath —
a genuinely different physical model.
```

## Picking one

- One system coupled to one bath → {py:class}`~fishbonett.models.system_bath.SystemBath`, and
  let `method=` choose how the bath is represented.  Start with `tebd` or
  `tree-tdvp2`.  See {doc}`spin_boson`.
- Several electronic sites, each with a bath → {py:class}`~fishbonett.models.fishbone.Fishbone`
  for a 1D chain of sites, {py:class}`~fishbonett.models.fishbone.TreeFishbone` for any
  other topology.  See {doc}`fishbone`.
- A system with internal structure (spin **and** vibration) → give each degree of
  freedom its own site in a `TreeFishbone`, rather than fattening them onto one.
  See {doc}`composite_multichannel`.
- One bath acting through several operators → pass a *list* of couplings to the
  {py:class}`~fishbonett.bath.spec.Bath`; that selects `multichannel`
  automatically, and `method=` then has nothing to choose.  See
  {doc}`composite_multichannel`.

Every model returns a {py:class}`~fishbonett.models.result.Result` with the same fields.
The shapes differ where the physics differs: single-system models give
`expect[name]` as `(n_steps,)`, multi-site models add a site axis.  See
{doc}`observables` for the observable spec, and {doc}`../bath` for the bath
itself.

```{toctree}
:maxdepth: 1
:hidden:

spin_boson
fishbone
composite_multichannel
observables
```
