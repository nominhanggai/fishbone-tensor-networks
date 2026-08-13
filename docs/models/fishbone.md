# Fishbone geometries

A **fishbone** is a set of electronic sites, each carrying its own bath (or two
baths — one on each side).  The electronic sites need not form a chain: the
general engine {py:class}`~fishbonett.models.fishbone.TreeFishbone` wires them into
*any* loop-free tree, and the common 1D chain
{py:class}`~fishbonett.models.fishbone.Fishbone` is a convenience specialization of it.

## The general engine: `TreeFishbone`

Give `TreeFishbone` a list of electronic site Hamiltonians, an **edge list**
describing the (loop-free) couplings between them, and one bath entry per site.
For example a central site coupled to three others (a star), each with its own
bath:

```python
import numpy as np
from fishbonett.models.fishbone import TreeFishbone
from fishbonett import Bath
from fishbonett.operators import sigma_x, sigma_z

def bath(op):
    return Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(0, 40),
                n_modes=20, phys_dim=10, coupling=op)

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

## The 1D specialization: `Fishbone`

The 1D chain is the most common case, so it has a convenience class that takes a
linear **backbone** instead of an edge list and delegates to the same engine — so
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

A `baths` entry may be a single `Bath` (one bath), a `(left, right)` pair (two
baths per site — the fishbone), or `None`.  A left bath defaults to a `sigma_z`
coupling and a right bath to `sigma_x` when the `Bath` sets none.

## Cost and truncation

Both classes propagate with `tree-tebd-static`, a **second-order** (Strang) Trotter
step: halving `dt` cuts the error ~4×, as with every other method here
({doc}`/getting_started`).  On a tree that takes more care than on a chain — the
edge gates must be applied in a palindromic order over the whole tree, not just down
and back up each branch — so the step applies each half-step gate twice.  See
{py:func}`fishbonett.evolve.tebd_tree.symmetric_tree_step`.

```{note}
Both classes run on the one general tree-TEBD engine.  On a 1D chain this *is* the
comb algorithm and costs the same at equal truncation.  The one thing to watch: an
interior backbone site with two baths is a high-degree (degree-4) tree tensor, so
its cost scales with the **square** of its bond dimensions.  Retaining singular
values far below the physical entanglement — an over-tight `trunc_eps` — then
inflates those bonds for no accuracy gain (e.g. a backbone bond of true rank 3 held
at `1e-10` can carry 15 values and run ~30× slower).  This is the geometry where
the default `trunc_eps=1e-4` matters most: set it to the accuracy you actually
need and let `result.max_bond` tell you what that costs.
```

If you need the highly-optimized comb tensor network directly, the low-level
builders {py:class}`~fishbonett.frames.schrodinger.FishBoneH` and
{py:class}`~fishbonett.states.comb.FishBoneNet` remain available.

See {doc}`observables` for measuring per-site, single-site and multi-site
(correlation) observables on a fishbone, and {doc}`composite_multichannel` for
giving a site internal structure or a cross-correlated multichannel bath.
