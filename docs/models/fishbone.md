# Fishbone models and tensor-network layout

A **fishbone** is a set of electronic sites, each carrying one or more independent
baths. The electronic sites need not form a chain: the
general model {py:class}`~fishbonett.models.fishbone.TreeFishbone` wires them into
*any* loop-free tree, and the common 1D chain
{py:class}`~fishbonett.models.fishbone.Fishbone` is a convenience specialization of it.

## The general model: `TreeFishbone`

Give `TreeFishbone` a list of electronic site Hamiltonians, an **edge list**
describing the (loop-free) couplings between them, and the baths attached to
their site indices.
For example a central site coupled to three others (a star), each with its own
bath:

```python
import numpy as np
from fishbonett.models.fishbone import TreeFishbone
from fishbonett import Bath
from fishbonett.operators import sigma_x, sigma_z

def bath(op):
    return Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(0, 40),
                n_modes=20, phys_dim=10).bind(op)

C = 0.3 * np.kron(sigma_z, sigma_z)                 # electronic-electronic coupling
fb = TreeFishbone(
    sites=[0.2 * sigma_z + sigma_x] * 4,
    edges=[(0, 1, C), (0, 2, C), (0, 3, C)],        # site 0 at the centre
    baths=[bath(sigma_z) for _ in range(4)])
res = fb.run(dt=0.02, t_max=1.0, bond_dim=80, observables={"sz": sigma_z})
res.expect["sz"]     # (n_steps, 4): <sz> on each of the 4 sites vs time
```

Each `edges` entry is `(i, j)` or `(i, j, coupling)`, where `coupling` is a
`(d_i·d_j, d_i·d_j)` operator on the joined pair; the pairs must form a tree
(`n_sites − 1` edges).  Each `baths` entry is a single `Bath`, a list of baths
(several on one site), or `None`.  Baths may even use different domains and
discretizations.

For sparse attachments, `baths` may instead be a mapping such as
`{0: left_bath, 3: right_bath}`. The keys are system-site indices and omitted
sites have no bath. The positional sequence form remains supported.

## The 1D specialization: `Fishbone`

The 1D chain is the most common case, so it has a convenience class that takes a
linear **backbone** instead of an edge list and uses the same propagation path, so
it has the identical observable interface and {py:class}`~fishbonett.models.result.Result`
layout:

```python
from fishbonett.models import Fishbone

fb = Fishbone(sites=[0.5 * sigma_z + sigma_x] * 3,           # 3 electronic sites
              baths=[(bath(sigma_z), bath(sigma_x))] * 3,    # two baths per site
              backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2)  # site i <-> i+1
# sigma_z is measured on *every* electronic site:
res = fb.run(dt=0.02, t_max=2.0, bond_dim=100, trunc_eps=1e-4,
             observables={"sz": sigma_z})
res.expect["sz"]        # (n_steps, n_sites); [t, i] = <sz> on site i at step t
res.rdm                 # (n_steps, n_sites, d, d)
```

A `baths` entry may be one `bath.bind(operator)` object, a list of explicitly
bound baths, or `None`. Every bath requires an explicit operator.

A mapping makes endpoint attachments readable without counting `None` values:

```python
fb = Fishbone(
    sites=sites,
    backbone=backbone,
    baths={
        0: left_bath.bind(left_coupling),
        3: right_bath.bind(right_coupling),
    },
)
```

The mapping keys attach these baths to system sites 0 and 3. `bind` specifies the
system operator through which each bath couples.

Bath modes can be measured without counting internal tensor nodes. For example,
the first represented coordinate of the second bath attached to site 0 is
`BathMode(system_site=0, bath=1, mode=0)`. See {doc}`observables` for bath-mode
and mixed system--bath observables and for the representation-dependent meaning
of `mode`.

### Cyclic and long-range electronic graphs

The tensor state remains a comb even when the physical electronic Hamiltonian
contains rings. Supply canonical `(i, j)` keys with `i < j` instead of a linear
`backbone`:

```python
fb = Fishbone(
    sites=sites,
    couplings={(0, 1): coupling01, (1, 2): coupling12,
               (0, 2): coupling02},
    baths={0: bath0, 1: bath1, 2: bath2},
)
```

`backbone` and `couplings` are mutually exclusive. For a Frenkel Hamiltonian,
{py:meth}`~fishbonett.models.fishbone.Fishbone.from_single_excitation` performs
the mapping to local two-level sites and hopping operators directly. Long-range
gates use an all-pairs swap network and restore the logical site order before
bath evolution.

Independent baths also support the interaction-chain representation:

```python
result = fb.run(
    dt=0.002, t_max=0.1,
    representation="interaction-chain",
    state_geometry="tree",
    integrator="tebd",
)
```

Each bath is independently transformed from star modes to the interaction-picture
chain. A reversible branch sweep brings the system site next to every represented
mode without changing which bath belongs to which electronic site.

An infrared-finite bath can instead use an independent polaron transform on
each coupled site. For example, a super-Ohmic density with a positive infrared
cutoff has a finite full-polaron displacement norm:

```python
polaron_bath = Bath(
    J=lambda w: 0.03 * w**3 * np.exp(-w / 5),
    domain=(0.05, 40), n_modes=20, phys_dim=10,
)
polaron_fb = Fishbone(
    sites=[0.5 * sigma_z + sigma_x] * 3,
    baths=[polaron_bath.bind(sigma_z) for _ in range(3)],
    backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2,
)
result = polaron_fb.run(
    dt=0.002, t_max=0.1,
    representation="polaron-chain",
    state_geometry="tree",
    integrator="tebd",
)
```

This route transforms the initial state and laboratory electronic observables
automatically. It supports one independent bath per system site; several baths
on the same site and the multi-site polaron-star representation are not yet
implemented. The spectral density must have finite
$\int J(\omega)/\omega^2\,d\omega$.

The interaction-chain comb offers three integrators, which solve the same H(t) and
differ only in how a branch's terms reach the state:

| `integrator` | how a branch is advanced | when to use it |
|---|---|---|
| `tebd` | the swap sweep above: the electronic index walks down the branch and back, truncating at every bond twice | the default; the only choice for a bath whose coupling operator is not Hermitian |
| `trotter-mpo` | one conditional-displacement operator per branch, so each bond grows once and a single sweep truncates it back | compare with `tebd` on a short converged run when operator application dominates |
| `tdvp2` | two-site TDVP along each branch: propagates with the *generator*, projected onto the two-site tangent space | when a generator-based integrator is wanted for its own sake -- **not** for a capped bond, see below |

### Several independent baths on one site

A comb site may carry more than one independent bath, passed as a list. The bath
Hamiltonian is then the sum of the branch Hamiltonians. Each branch keeps its own
`phys_dim`, so parts of different character can be sized independently. If two
branches use noncommuting operators on the same site, their evolution is composed
with a palindromic second-order split.

```{warning}
Every bath in a multi-bath list must be bound explicitly, for example
`[low_frequency.bind(op), vibrations.bind(op)]`. The package rejects an unbound
list because list position does not define a physical coupling operator.
```

```{note}
Splitting a spectral density can improve local Fock-space control, but it may also
increase the number and width of branch bonds. Compare merged and split forms by
converging the same observable; a smaller peak bond alone does not determine the
total contraction cost.
```

`trotter-mpo` needs the per-site coupling operators to be Hermitian (it is built
from their eigenbasis). Operators on different sites commute; noncommuting
operators on the same site are handled by the symmetric branch composition. See
{doc}`/methods/interaction/trotter_mpo` for the truncated-ladder caveat.

All three integrators are second order in the time step. Validate a calculation by
halving `dt` and tightening the SVD threshold until the observable is stable.

```{warning}
One-site TDVP works inside a fixed-bond manifold and never truncates. Two-site
TDVP instead splits each evolved two-site block with a truncating SVD, so a binding
`bond_dim` can discard physical weight just as it can in a Trotter calculation.
Converge the cap independently of the integrator choice.
```

## Thermal preparation and continuation

`GibbsPurification` prepares an exact finite-temperature state for a short,
interacting system backbone and lifts its operators onto physical-plus-ancilla
supersites.

Long static-tree runs can return a `result.checkpoint` and resume with
`run(resume=result.checkpoint, ...)`. Set `bath_horizon` on the first segment to
the complete intended time so automatic bath resolution contains the full light
cone. A checkpoint validates the resolved Hamiltonian and rejects changed bath
temperatures, coefficients, topology, or method. `observe_every` reduces how often
RDMs and expensive composite observables are contracted without changing the TEBD
integration step.

## Cost and truncation

The default for both classes is `schrodinger-chain-tree-tebd`, a **second-order**
(Strang) Trotter step: halving `dt` cuts the error by about four, as with every
other split method here
({doc}`/getting_started`).  On a tree that takes more care than on a chain — the
edge gates must be applied in a palindromic order over the whole tree, not just down
and back up each branch — so the step applies each half-step gate twice.  See
{py:func}`fishbonett.evolve.sitetree.symmetric_tree_step`.

```{note}
Both classes use {py:mod}`fishbonett.evolve.sitetree` with a
{py:class}`~fishbonett.states.tree.TreeTensorNetwork` state. An interior backbone
site with two baths is a high-degree (degree-4) tree tensor, so
its cost scales with the **square** of its bond dimensions.  Retaining singular
values far below the physical entanglement — an over-tight `trunc_eps` — then
inflates those bonds for no accuracy gain (e.g. a backbone bond of true rank 3 held
at `1e-10` can carry 15 values and run ~30× slower). This is the tensor-network geometry where
the default `trunc_eps=1e-4` matters most: set it to the accuracy you actually
need and let `result.max_bond` tell you what that costs.
```

See {doc}`observables` for measuring per-site, single-site and multi-site
(correlation) observables on a fishbone, and {doc}`composite_multichannel` for
giving a site internal structure or a cross-correlated multichannel bath.
