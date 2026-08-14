# Models

A **model** is the physical setup: how many system sites there are and how they are
wired to each other.  Only that.  How the *bath* is represented is two separate
axes — the mode `basis` and the state `geometry` — because those are choices of
representation rather than of physics ({doc}`../methods/index`).

Four models, three classes:

| model | what it is | class |
|---|---|---|
| `system-bath` | one system, one bath, one coupling operator | {py:class}`~fishbonett.models.system_bath.SystemBath` |
| `multichannel` | one system, one bath through several couplings on shared modes | {py:class}`~fishbonett.models.system_bath.SystemBath` |
| `comb` | several sites on a 1D backbone, baths per site — the fishbone | {py:class}`~fishbonett.models.fishbone.Fishbone` |
| `site-tree` | several sites in any loop-free tree, baths per site | {py:class}`~fishbonett.models.fishbone.TreeFishbone` |

```{admonition} `chain`, `star` and `mode-tree` were never models
:class: warning
They used to be listed here as three of six models.  The first two name a bath
**basis** and the third a state **geometry**, and all three are the same
one-system/one-bath problem — so they are now axes of `run`, not rows in this table.
`run(model="star")` raises, and says what to write instead.

The tell was in the code all along: `model="chain"` with `frame="interaction"`
chain-maps the bath and then throws the chain away, because the interaction picture
needs the *star* modes.  And `mode-tree` shares its Hamiltonian with
`mpo-ip-tdvp1` — same basis, same frame, different geometry — which is why the two
agree to machine precision.
```

```{admonition} Two different trees
:class: warning
The `site-tree` **model** and the `binary-tree` **geometry** are unrelated.  In the
`binary-tree` geometry a single system's bath modes are arranged on a tree, to keep
the high-bond region $O(\log N)$ edges deep — an efficiency trick inside a
one-system model.  In `site-tree` it is the **system sites themselves** that form a
tree, each with its own bath — a genuinely different physical setup.
```

Every model, the frames and axes it admits, and why each absent combination is
absent — generated from {py:mod}`fishbonett.models.registry` when these docs are
built, so it is whatever the code actually offers:

```{literalinclude} ../_generated/taxonomy.txt
:language: text
```

## Picking one

- One system coupled to one bath → {py:class}`~fishbonett.models.system_bath.SystemBath`, and
  let `method=` (or `frame=`/`basis=`/`geometry=`) choose how the bath is
  represented.  Start with `tebd` or `tree-tdvp2`.  See {doc}`spin_boson`.
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
