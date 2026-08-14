# Getting started

```{admonition} Quick start
:class: tip
`from fishbonett import Bath, SystemBath, Truncation` — build a bath, attach it
to a system, call `model.run(...)`.  See {doc}`models/index` for the model
classes, {doc}`methods/index` for propagation methods, and {doc}`bath` for the
bath itself.
```

## Installation

```bash
pip install -e .                 # from a checkout
pip install -e ".[gpu]"          # optional CuPy GPU truncation
pip install -e ".[rates]"        # optional vegas Monte-Carlo integrator
pip install -e ".[test,docs]"    # development
```

Core dependencies are `numpy`, `scipy` and `opt_einsum`; Python ≥ 3.10 is required.

### Why `opt_einsum` is a core dependency

The tree and MPO engines contract many-operand tensor networks in their inner
loop. `opt_einsum` evaluates such a contraction as a sequence of pairwise
`tensordot`/BLAS calls, whereas `numpy.einsum` — even with `optimize="greedy"`
and a pre-computed path — still falls back to its unvectorized `c_einsum` C loop
for the actual multi-operand contraction. On the tree engine the difference is
about **100×** (measured: 0.51 s vs 55 s for the same `tree-tdvp2` step), so
`opt_einsum` is a hard requirement rather than a convenience: `numpy`'s
`optimize=` option chooses a good contraction *order* but cannot match the
per-contraction throughput.

## A first simulation

The high-level interface ({py:mod}`fishbonett.models`) propagates the population
$\langle\sigma_z\rangle(t)$ of a two-level system coupled to a bath with a single
call. Declare the bath and the system, then `run`:

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5),   # spectral density J(w)
            domain=(-25, 36), temperature=1.0,       # T-TEDOPA thermalization
            n_modes=40, phys_dim=20,
            discretization="tedopa")                # or the default "legendre"
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)

result = model.run(dt=0.05, t_max=4.0, method="tree-tdvp2", bond_dim=200,
                   observables={"sz": sigma_z})

result.t                 # time grid
result.expect["sz"]      # <sigma_z>(t)
result.max_bond          # peak bond dimension per step
```

`method` selects the engine — `"tebd"`, `"trotter-mpo"`, `"polaron"` (and its
TDVP variants), `"mpo-tdvp1" | "mpo-tdvp2" | "mpo-dtdvp"`,
`"mpo-ip-tdvp1" | "mpo-ip-tdvp2"`, or `"tree-tdvp" | "tree-tdvp2" | "tree-tebd"`.
Every method uses the same `dt`/`t_max` and returns the same
{py:class}`~fishbonett.models.result.Result`, so switching engines is a one-word change.
See {doc}`methods/index` for the theory and an example behind each one.

### Accuracy: `dt`, `trunc_eps`, `bond_dim`

Three knobs control accuracy:

- **`dt`** — the time step. Every method here is second order, so halving `dt`
  cuts the time-discretization error roughly 4×.
- **`trunc_eps`** (default `1e-4`) — the truncation threshold. Singular values
  below it are discarded, so this alone sets how large the bond dimension grows.
- **`bond_dim`** (default `None`, meaning **unlimited**) — an optional hard cap on
  the bond dimension, for when memory rather than accuracy is the binding
  constraint. `result.max_bond` reports what was actually used.

The recommended workflow: pick `trunc_eps` for the accuracy you need, leave
`bond_dim` unset, and watch `result.max_bond`. Then confirm convergence by halving
`dt` and tightening `trunc_eps` one notch each and checking the answer moves less
than you care about. (A few methods have a *fixed* bond dimension and therefore
require an explicit `bond_dim` — see {doc}`methods/index`.)

## The fishbone geometry

A fishbone is a set of electronic sites, each coupled to one bath (a comb) or two
baths — one on each side of the site (the fishbone).
{py:class}`~fishbonett.models.fishbone.Fishbone` is the 1D-chain specialization (a linear
backbone) of the general tree engine
{py:class}`~fishbonett.models.fishbone.TreeFishbone`, to which it delegates; both return
per-site data. For a non-chain topology, use ``TreeFishbone`` with an edge list.
The 1D chain is declared the same way as the single-site system:

```python
from fishbonett.models import Fishbone

def bath(op):                                        # one bath, coupling operator op
    return Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(0, 40),
                n_modes=20, phys_dim=10).bind(op)

fb = Fishbone(sites=[0.5 * sigma_z + sigma_x] * 3,           # 3 electronic sites
              baths=[(bath(sigma_z), bath(sigma_x))] * 3,    # two baths per site
              backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2)  # nearest-neighbour
res = fb.run(dt=0.02, t_max=2.0, bond_dim=100, trunc_eps=1e-4,
             observables={"sz": sigma_z})
res.expect["sz"]         # (n_steps, n_sites): sigma_z measured on each site
res.rdm                  # (n_steps, n_sites, d, d): reduced density matrix per site
```

{doc}`models/index` covers the full model vocabulary — arbitrary tree topologies,
composite (spin + vibration) systems, multichannel baths and multi-site
observables — and {doc}`bath` covers bath discretization and finite temperature.

## Picking a method by its axes

`method=` names a combination; you can also give the combination itself, which is
usually clearer and is what the taxonomy is actually made of:

```python
res = sb.run(dt=0.02, t_max=2.0, frame="polaron-chain",
             integrator="tebd")          # == method="polaron"
```

A run is four independent choices:

| axis | values | what it is |
|---|---|---|
| `model` | `system-bath`, `multichannel`, `comb`, `site-tree` | what is coupled to what |
| `frame` | a picture × a basis — six of them, below | how `H` is written down |
| `geometry` | `path`, `binary-tree`, `comb-tree` | the graph the state lives on |
| `integrator` | `tebd`, `tdvp1`, `tdvp2`, `dtdvp`, `trotter-mpo` | how a step is taken |

A **frame** is a picture *and* a mode basis, because both are choices about how `H`
is written down. All $3\times2$ are real frames:

| picture | `chain` | `star` |
|---|---|---|
| **Schrödinger** | `schrodinger-chain` | `schrodinger-star` |
| **interaction** | `interaction-chain` | `interaction-star` |
| **polaron** | `polaron-chain` | `polaron-star` *(not implemented)* |

The two bases are one orthogonal (Lanczos) transform apart, so they are the same
physics at different cost — which is why the basis is a representation choice and
not a model. A bare picture works where it names one frame:
`frame="interaction-chain"` is explicit, and `frame="schrodinger"` names two and
says so.

Every model, the frames it admits, and the reason each absent combination is absent —
generated from {py:mod}`fishbonett.models.registry` when these docs are built, so it
is whatever the code actually offers:

```{literalinclude} _generated/taxonomy.txt
:language: text
```

## Low-level engines

For finer control the underlying engines are available directly.  The high-level
path first compiles `Bath` into immutable star or chain coefficients and gives
those to the frame.  Legacy low-level builders also accept a spectral density and
domain directly, as this polaron example does:

```python
import numpy as np
from fishbonett.frames.polaron import SystemBathPolaron
from fishbonett.states.mps import SystemBathMPS
from fishbonett.evolve import tebd
from fishbonett.operators import sigma_x, sigma_z

pd = [2] + [10] * 8                        # system on site 0, then the bath chain
builder = SystemBathPolaron(
    pd, h_sys=0.5 * sigma_x, coupling=sigma_z,
    sd=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0)).build()

state = SystemBathMPS(pd)
gates = builder.gates(0.02 / 2)            # static frame: gates built once...
tebd.symmetric_static_step(state, gates, len(pd) - 1, chi_max=60, eps=1e-4)
rho = state.rdm(0)                         # inherited from TensorNetwork
```

`symmetric_static_step` applies each gate twice, so it takes **half**-step gates —
the convention every second-order step here uses.  The same construction applies to
{py:class}`~fishbonett.frames.interaction_picture.SystemBathIP` (whose gates are
time-dependent, so they are rebuilt each step by
{py:func}`~fishbonett.evolve.tebd.symmetric_swap_step`) and to
{py:class}`~fishbonett.frames.multichannel.SystemBathMultiChannel`.  For a
multi-site model the state is a {py:class}`~fishbonett.states.tree.TreeTensorNetwork` driven
by {py:mod}`fishbonett.evolve.sitetree` instead.

The high-level interface above is a thin wrapper over exactly this loop, and
additionally resolves the automatic `domain` / `n_modes`, prepares the initial
state, and — in time-dependent frames — rebuilds the gates each step.

See the [`examples/`](https://github.com/nominhanggai/fishbone-tensor-networks/tree/main/examples)
directory for runnable scripts — start with `friendly_interface.py`, which also
covers the interaction picture, the cooling scheme, and rate theory.
