# Propagation methods

Every propagation method is selected by the `method` argument of
{py:meth}`BosonicBath.run <fishbonett.simulate.BosonicBath.run>`.  A method name
encodes three orthogonal choices:

1. **State ansatz** — a matrix-product state (MPS), a matrix-product operator
   (MPO) of the TEDOPA chain, or a tree tensor network (TTN).
2. **Picture** — the Schrödinger picture, or an *interaction picture* that
   removes the bath's free evolution so the residual entanglement is small and
   spin-mediated.
3. **Integrator** — time-evolving block decimation (TEBD, a Trotter splitting),
   or the time-dependent variational principle (TDVP) in 1-site, 2-site or
   bond-adaptive form.

| ``method``        | picture      | state / integrator                 | bond growth        | page |
|-------------------|--------------|------------------------------------|--------------------|------|
| ``tebd``          | interaction  | MPS, swap-network TEBD             | SVD truncation     | {doc}`tebd` |
| ``polaron``       | polaron      | MPS chain, static-gate TEBD        | SVD truncation     | {doc}`polaron` |
| ``polaron-tdvp1`` | polaron      | polaron MPO, 1-site TDVP          | fixed              | {doc}`polaron` |
| ``polaron-tdvp2`` | polaron      | polaron MPO, 2-site TDVP          | SVD truncation     | {doc}`polaron` |
| ``polaron-dtdvp`` | polaron      | polaron MPO, bond-adaptive DTDVP  | precision threshold| {doc}`polaron` |
| ``mpo-tdvp1``     | Schrödinger  | chain MPO, 1-site TDVP            | fixed              | {doc}`chain_mpo` |
| ``mpo-tdvp2``     | Schrödinger  | chain MPO, 2-site TDVP            | SVD truncation     | {doc}`chain_mpo` |
| ``mpo-dtdvp``     | Schrödinger  | chain MPO, bond-adaptive DTDVP   | precision threshold| {doc}`chain_mpo` |
| ``mpo-ip-tdvp1``  | interaction  | star MPO, 1-site TDVP            | fixed              | {doc}`star_mpo` |
| ``mpo-ip-tdvp2``  | interaction  | star MPO, 2-site TDVP            | SVD truncation     | {doc}`star_mpo` |
| ``tree-tdvp``     | interaction  | binary-tree TTN, 1-site TDVP    | fixed              | {doc}`tree` |
| ``tree-tdvp2``    | interaction  | binary-tree TTN, 2-site TDVP    | SVD truncation     | {doc}`tree` |
| ``tree-tebd``     | interaction  | binary-tree TTN, TEBD           | SVD truncation     | {doc}`tree` |

All methods take the same `dt` / `t_max` and return the same
{py:class}`~fishbonett.simulate.Result`, so switching engines is a one-word
change and the results are directly comparable.  The example below runs the same
spin-boson model through all nine and prints the final population:

```python
import numpy as np
from fishbonett.simulate import Bath, BosonicBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = BosonicBath(h=sigma_x, coupling=sigma_z, bath=bath)

for method in ["tebd",
               "mpo-tdvp1", "mpo-tdvp2", "mpo-dtdvp",
               "mpo-ip-tdvp1", "mpo-ip-tdvp2",
               "tree-tdvp", "tree-tdvp2", "tree-tebd"]:
    r = model.run(dt=0.02, t_max=2.0, method=method, bond_dim=100,
                  observables={"sz": sigma_z})
    print(f"{method:14s} <sz>(t_end) = {r.expect['sz'][-1]:+.4f}")
```

## Choosing a method

- **Just want a number, fast, and accurate?** Use `mpo-tdvp1` — a
  Schrödinger-picture chain MPO with a fixed bond dimension is usually the most
  accurate per unit cost, provided you know a large-enough `bond_dim`.
- **Don't know the required bond dimension?** Use a 2-site or adaptive variant
  (`mpo-tdvp2`, `mpo-dtdvp`, `tree-tdvp2`, `tree-tebd`); they grow the bonds from
  a product state and report the peak bond in `result.max_bond`.
- **Non-`sigma_z` coupling, a non-two-level system, or a custom initial state?**
  Every engine supports these — a Hermitian `h` of any dimension, a Hermitian
  coupling `O`, and any `initial` state (see {doc}`../systems/spin_boson`).
- **Long chains where entanglement piles up in the middle?** The `tree-*`
  methods keep the high-bond region `O(log n)` edges deep instead of `O(n)`
  (see {doc}`tree`).
- **Strong system–bath coupling at zero temperature?** The `polaron` method works
  in the polaron frame — it folds the static reorganization into a bath
  displacement, so the MPS carries less entanglement (smaller `bond_dim`) than the
  interaction-picture chain. It needs `T=0` and a gapped/super-ohmic bath
  (see {doc}`polaron`).

## Shared conventions

**Time stepping.** Give either `t_max` (the engine takes `round(t_max/dt)`
steps) or `n_steps` directly, together with the step `dt`.  The returned
`result.t` is the time grid.

**Truncation.** `bond_dim` caps the bond dimension; `trunc_eps` discards singular
values below `trunc_eps` (relative to the largest).  Fixed-bond methods
(`*-tdvp1`) ignore `trunc_eps`; growing methods honour both.  For the tree
methods on heavily-entangled multi-bath geometries an over-tight `trunc_eps`
inflates cost — see {doc}`../systems/fishbone`.

**Reading the result.** Every method returns a
{py:class}`~fishbonett.simulate.Result`:

```python
r = model.run(dt=0.02, t_max=2.0, method="mpo-tdvp2", bond_dim=100,
              observables={"sz": sigma_z, "sx": sigma_x})
r.t                  # time grid, shape (n_steps,)
r.expect["sz"]       # <sigma_z>(t)
r.rdm                # spin reduced density matrix per step, (n_steps, 2, 2)
r.max_bond           # peak bond dimension per step (adaptive methods)
```

```{toctree}
:maxdepth: 1
:hidden:

tebd
polaron
chain_mpo
star_mpo
tree
```
