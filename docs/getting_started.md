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
pip install -e ".[gpu-cuda12]"   # or gpu-cuda11, matching your CUDA runtime
pip install -e ".[rates]"        # optional vegas Monte-Carlo integrator
pip install -e ".[test,docs]"    # development
```

Core dependencies are `numpy`, `scipy`, and `opt_einsum`; Python ≥ 3.10 is
required. `opt_einsum` supplies reusable contraction paths for the MPS and tree
tensor-network engines.

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

result = model.run(dt=0.05, t_max=4.0, method="interaction-chain-tree-tebd", bond_dim=200,
                   observables={"sz": sigma_z})

result.t                 # time grid
result.expect["sz"]      # <sigma_z>(t)
result.max_bond          # peak bond dimension per step
```

Every `method` name begins with its Hamiltonian representation, followed by the
integrator; tree tensor-network methods also include `tree`. Examples include
`interaction-chain-tebd`, `interaction-chain-trotter-mpo`,
`polaron-chain-tdvp2`, `schrodinger-chain-tdvp2`, and
`interaction-chain-tree-tebd`. Every method uses the same `dt`/`t_max` and returns the same
{py:class}`~fishbonett.models.result.Result`, so switching engines is a one-word change.
See {doc}`methods/index` for the theory and an example behind each one.

### Accuracy: `dt`, `trunc_eps`, `bond_dim`

Three knobs control accuracy:

- **`dt`** — the time step. Every method here is second order, so halving `dt`
  cuts the time-discretization error roughly 4×.
- **`trunc_eps`** (default `1e-4`) — the relative truncation threshold for
  SVD-based methods and the tangent-space convergence precision for `a1tdvp`.
  This is the main accuracy control on adaptive bond growth.
- **`bond_dim`** (default `None`, meaning **unlimited**) — an optional hard cap on
  the bond dimension, for when memory rather than accuracy is the binding
  constraint. `result.max_bond` reports what was actually used.

The recommended workflow: pick `trunc_eps` for the accuracy you need, leave
`bond_dim` unset, and watch `result.max_bond`. Then confirm convergence by halving
`dt` and tightening `trunc_eps` one notch each and checking the answer moves less
than you care about. (A few methods have a *fixed* bond dimension and therefore
require an explicit `bond_dim` — see {doc}`methods/index`.)

For `tdvp2`, the split keeps up to `bond_expand` (default `2`) Schmidt
directions beyond the threshold rank. This lets an initially product MPS
accumulate the entangling component generated during one step; with
`bond_expand=0`, a loose threshold can leave the state locked at bond one.

`a1tdvp` uses a different mechanism. Before each one-site TDVP sweep it completes
the left and right canonical bases with full QR factorizations, evaluates the
three local effective-Hamiltonian norms associated with each bond, and keeps the
smallest extension whose relative contribution is below `trunc_eps`.
`bond_expand` limits how many QR-complement directions are tested during one
sweep. Because this method grows a one-site tangent space rather than splitting
a two-site centre, `bond_dim` is required as a memory ceiling.

### SVD backend and reproducibility

`svd_backend="auto"` is the default. Small blocks use the exact LAPACK SVD.
Larger blocks use an adaptively enlarged randomized range only when its omitted
Frobenius norm certifies that no unresolved singular direction can exceed the
requested `trunc_eps`. A slowly decaying spectrum falls back to LAPACK instead of
accepting an uncertified truncation. Use `svd_backend="exact"` for a reference
calculation or `svd_backend="randomized"` to request randomized range finding on
smaller eligible blocks; the same residual check and exact fallback still apply.

Randomized truncation draws random numbers. `run(seed=...)` defaults to `0`,
which makes the sketches reproducible on the same numerical backend. Their seeds
are keyed to the matrix being decomposed, so splitting an otherwise identical run
across checkpoints does not change its sketches. The generator is run-local and
never touches NumPy's global random state.

Pass `seed=None` to draw from NumPy's global generator instead. This permits
run-to-run variation from randomized truncation and is generally unsuitable for
convergence comparisons.

In `auto` mode, blocks whose smaller dimension is at most 128 use exact SVD. The
`result.meta["svd"]` counters report exact calls, randomized calls, exact
fallbacks, maximum trial and retained ranks, and the largest certified residual
ratio encountered during propagation.

## The fishbone tensor-network geometry

A fishbone is a set of electronic sites, each coupled to one bath (a comb) or two
baths — one on each side of the site (the fishbone).
{py:class}`~fishbonett.models.fishbone.Fishbone` is the 1D-chain specialization (a linear
backbone) of the tree engine
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
res = sb.run(dt=0.02, t_max=2.0, representation="polaron-chain",
             integrator="tebd")          # == method="polaron-chain-tebd"
```

A run is four independent choices:

| axis | values | what it is |
|---|---|---|
| `model` | `system-bath`, `multichannel`, `exciton-bath`, `comb`, `site-tree` | what is coupled to what |
| `representation` | five exact names, below | how `H` is written down |
| `state_geometry` | `mps`, `system-first-mps`, `interleaved-mps`, `multi-set-mps`, `multi-set-tree`, `binary-tree`, `tree` | tensor-network factorization and site ordering |
| `integrator` | `tebd`, `tdvp1`, `tdvp2`, `a1tdvp`, `trotter-mpo` | how a step is taken |

The general `tree` state supports several model-specific tensor-network
geometries: a comb for `Fishbone`, an arbitrary loop-free tree for
`TreeFishbone`, and a star for the static multichannel model.

Supported Hamiltonian representations are:

| transformation | chain representation | star representation |
|---|---|---|
| **Schrödinger** | `schrodinger-chain` | `schrodinger-star` |
| **interaction** | `interaction-chain` | — |
| **polaron** | `polaron-chain` | `polaron-star` |

The Schrödinger and polaron star/chain pairs are related by an orthogonal
transform. The interaction representation always includes the final
star-to-chain transformation. Partial names such as
`representation="schrodinger"` are rejected; use an exact name.

Every model, the representations it admits, and the reason each absent combination is absent —
generated from {py:mod}`fishbonett.models.registry` when these docs are built, so it
is whatever the code actually offers:

```{literalinclude} _generated/taxonomy.txt
:language: text
```

## Low-level state and evolution API

For finer control the underlying layers are available directly. The high-level
API compiles `Bath`, constructs a representation, asks it for the numerical
product needed by the integrator, and then propagates:

```python
import numpy as np
from fishbonett import Bath
from fishbonett.representations.polaron import PolaronRepresentation
from fishbonett.states.mps import SystemBathMPS
from fishbonett.evolve import tebd
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(
    J=lambda w: 0.3 * w * np.exp(-w / 2.5),
    domain=(0.3, 12.0), n_modes=8, phys_dim=10,
)
representation = PolaronRepresentation(
    representation="polaron-chain",
    h_sys=0.5 * sigma_x, coupling=sigma_z,
    bath=bath,
).build()
pd = list(representation.dimensions)        # system on site 0, then the bath chain

state = SystemBathMPS(pd)
gates = representation.tebd_gates(0.02 / 2)
tebd.symmetric_static_step(state, gates, len(pd) - 1, chi_max=60, eps=1e-4)
rho = state.rdm(0)                         # inherited from TensorNetwork
```

`symmetric_static_step` applies each gate twice, so it takes **half**-step gates —
the convention every second-order step here uses. The interaction and
multichannel representations expose `tebd_gates(t, dt)` directly; their
time-dependent gates are rebuilt each step. For a
multi-site model the state is a {py:class}`~fishbonett.states.tree.TreeTensorNetwork` driven
by {py:mod}`fishbonett.evolve.sitetree` instead.

The high-level interface resolves the automatic `domain` / `n_modes`, prepares
the initial state, runs this loop, and — in time-dependent representations —
rebuilds the gates each step.

See the [`examples/`](https://github.com/nominhanggai/fishbone-tensor-networks/tree/main/examples)
directory for runnable scripts. Start with `friendly_interface.py` for compact
single-system and multi-site examples, then use the paper-backed tutorials for
complete scientific calculations.
