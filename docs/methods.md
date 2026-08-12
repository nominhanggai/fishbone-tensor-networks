# Methods

Every propagation method is selected by the ``method`` argument of
{py:meth}`SpinBoson.run <fishbonett.simulate.SpinBoson.run>`.  The argument
picks three orthogonal choices at once — the **state ansatz** (matrix-product
state / tree tensor network), the **picture** (Schrödinger or interaction), and
the **integrator** (TEBD, or 1-/2-site / bond-adaptive TDVP):

| ``method``        | picture      | state / integrator                 | bond growth        |
|-------------------|--------------|------------------------------------|--------------------|
| ``tebd``          | interaction  | MPS, swap-network TEBD             | SVD truncation     |
| ``mpo-tdvp1``     | Schrödinger  | chain MPO, 1-site TDVP            | fixed              |
| ``mpo-tdvp2``     | Schrödinger  | chain MPO, 2-site TDVP            | SVD truncation     |
| ``mpo-dtdvp``     | Schrödinger  | chain MPO, bond-adaptive DTDVP   | precision threshold|
| ``mpo-ip-tdvp1``  | interaction  | star MPO, 1-site TDVP            | fixed              |
| ``mpo-ip-tdvp2``  | interaction  | star MPO, 2-site TDVP            | SVD truncation     |
| ``tree-tdvp``     | interaction  | binary-tree TTN, 1-site TDVP    | fixed              |
| ``tree-tdvp2``    | interaction  | binary-tree TTN, 2-site TDVP    | SVD truncation     |
| ``tree-tebd``     | interaction  | binary-tree TTN, TEBD           | SVD truncation     |

All methods take the same ``dt`` / ``t_max`` and return the same
{py:class}`~fishbonett.simulate.Result`, so switching engines is a one-word
change and the results are directly comparable.

## One example per method

The `tebd` engine supports an arbitrary system Hamiltonian and system–bath
coupling; the MPO and tree engines assume a `sigma_z` coupling and decompose
`h = (eps/2) sigma_z + V sigma_x`.

```python
import numpy as np
from fishbonett.simulate import Bath, SpinBoson
from fishbonett.stuff import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SpinBoson(h=sigma_x, coupling=sigma_z, bath=bath)

for method in ["tebd",
               "mpo-tdvp1", "mpo-tdvp2", "mpo-dtdvp",
               "mpo-ip-tdvp1", "mpo-ip-tdvp2",
               "tree-tdvp", "tree-tdvp2", "tree-tebd"]:
    r = model.run(dt=0.02, t_max=2.0, method=method, bond_dim=100,
                  observables={"sz": sigma_z})
    print(f"{method:14s} <sz>(t_end) = {r.expect['sz'][-1]:+.4f}")
```

### Notes per method

- **`tebd`** — interaction picture with respect to the system–bath coupling; the
  bath chain is swept with leg swaps.  The only method that accepts a general
  (non-`sigma_z`) coupling and an arbitrary initial spin state (`initial=...`).
- **`mpo-tdvp1`** — Schrödinger-picture finite-state-machine MPO of the TEDOPA
  chain; second-order symmetric 1-site sweep at fixed bond dimension.  The most
  accurate at a given bond dimension.
- **`mpo-tdvp2`** — the same MPO with a 2-site sweep that grows the bond
  dimension by SVD truncation (to `bond_dim` / singular values above
  `trunc_eps`).  Use when you do not know the required bond dimension ahead of
  time.  `result.max_bond` reports the peak bond per step.
- **`mpo-dtdvp`** — bond-adaptive 1-site DTDVP; each bond grows only as far as a
  local precision threshold requires (capped at `bond_dim`).
- **`mpo-ip-tdvp1` / `mpo-ip-tdvp2`** — interaction-picture *star* MPO: the bath
  modes carry time-dependent couplings `d_j(t)` and the MPO is rebuilt at each
  step midpoint (no on-site frequency or inter-mode hopping).  1-site (fixed
  bond) and 2-site (growing) variants.
- **`tree-tdvp` / `tree-tdvp2` / `tree-tebd`** — interaction-picture balanced
  binary tree with the spin at the root, so the high-entanglement region is
  `O(log n)` edges deep instead of `O(n)`.  1-site TDVP, 2-site TDVP and TEBD
  variants.

## Reading the result

```python
r = model.run(dt=0.02, t_max=2.0, method="mpo-tdvp2", bond_dim=100,
              observables={"sz": sigma_z, "sx": sigma_x})
r.t                  # time grid (shape (n_steps,))
r.expect["sz"]       # <sigma_z>(t)
r.rdm                # spin reduced density matrix per step, (n_steps, 2, 2)
r.max_bond           # peak bond dimension per step (adaptive methods)
```

## Fishbone geometries

A **fishbone** is a chain of electronic sites, each carrying its own bath (or two
baths — one on each side).  Use {py:class}`~fishbonett.fishbone_sim.Fishbone`:

```python
from fishbonett.fishbone_sim import Fishbone

def bath(op):
    return Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(0, 40),
                n_modes=20, phys_dim=10, coupling=op)

fb = Fishbone(sites=[0.5 * sigma_z + sigma_x] * 3,           # 3 electronic sites
              baths=[(bath(sigma_z), bath(sigma_x))] * 3,    # two baths per site
              backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2)
res = fb.run(dt=0.02, t_max=2.0, bond_dim=100, observables={"sz": sigma_z})
res.expect["sz"]     # shape (n_steps, n_sites)
```

The topology need not be 1D.  {py:class}`~fishbonett.treebone.TreeFishbone` wires
the electronic sites into *any* loop-free tree via an edge list — for example a
central site coupled to three others (a star), each with its own bath:

```python
from fishbonett.treebone import TreeFishbone

C = 0.3 * np.kron(sigma_z, sigma_z)
fb = TreeFishbone(
    sites=[0.2 * sigma_z + sigma_x] * 4,
    edges=[(0, 1, C), (0, 2, C), (0, 3, C)],       # site 0 at the centre
    baths=[bath(sigma_z) for _ in range(4)])
res = fb.run(dt=0.02, t_max=1.0, bond_dim=80, observables={"sz": sigma_z})
res.expect["sz"]     # shape (n_steps, 4)
```

## Low-level drivers

The high-level interface wraps the low-level drivers, which can be called
directly for finer control — e.g. the Schrödinger-picture chain TDVP:

```python
from fishbonett.mpo import run_tdvp1, run_ip_tdvp1

t, sz = run_tdvp1(bath.spectral_density(), (-25, 36), V=1.0,
                  n_chain=40, d=20, dt=0.025, nsteps=80, D=100)
t, sz = run_ip_tdvp1(bath.spectral_density(), (-25, 36), V=1.0,
                     n_chain=40, d=20, dt=0.025, nsteps=80, D=100)
```
