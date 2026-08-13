# Propagation methods

Every method here solves the **same problem**: a system coupled to a harmonic bath
that has been chain-mapped into a 1D chain of effective modes ({doc}`../bath`).
They differ only in *how* they represent and propagate that chain, and they are all
selected by the `method` argument of
{py:meth}`BosonicBath.run <fishbonett.simulate.BosonicBath.run>`.

A method name encodes three orthogonal choices:

1. **Frame** — the picture the model is written in. The *Schrödinger* picture uses
   the bare Hamiltonian; the *interaction* picture rotates out the free-bath
   evolution, so the only entanglement generator left is the system–bath coupling;
   the *polaron* frame goes further and absorbs the static part of that coupling
   into a displacement of the bath, leaving only the dressed tunneling.
   The frame is what sets **how much entanglement the state carries**.
2. **State ansatz** — a matrix-product state (MPS) for a chain, or a tree tensor
   network (TTN) when the geometry branches.
3. **Integrator** — a Trotter splitting (TEBD, or an exact MPO propagator), or the
   time-dependent variational principle (TDVP) in 1-site, 2-site or bond-adaptive
   form. The integrator sets **the cost per step and the error in `dt`**.

Frame and integrator are largely independent: `tebd` and `trotter-mpo` share a
frame but differ in integrator, while `polaron` and `polaron-dtdvp` share a frame
and differ likewise.

| ``method``        | picture      | state / integrator                 | bond growth        | page |
|-------------------|--------------|------------------------------------|--------------------|------|
| ``tebd``          | interaction  | MPS, swap-network TEBD             | SVD truncation     | {doc}`tebd` |
| ``trotter-mpo``   | interaction  | MPS, exact conditional-displacement MPO | SVD truncation | {doc}`trotter_mpo` |
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
change and the results are directly comparable — which also makes cross-checking
one method against another the easiest way to validate a calculation.  The example
below runs the same spin-boson model through several and prints the final
population:

```python
import numpy as np
from fishbonett.simulate import Bath, BosonicBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = BosonicBath(h=sigma_x, coupling=sigma_z, bath=bath)

for method in ["tebd", "trotter-mpo",
               "mpo-tdvp1", "mpo-tdvp2", "mpo-dtdvp",
               "mpo-ip-tdvp1", "mpo-ip-tdvp2",
               "tree-tdvp", "tree-tdvp2", "tree-tebd"]:
    r = model.run(dt=0.02, t_max=2.0, method=method, bond_dim=100,
                  observables={"sz": sigma_z})
    print(f"{method:14s} <sz>(t_end) = {r.expect['sz'][-1]:+.4f}")
```

(The polaron methods are omitted from that loop only because they need `T=0` and a
gapped bath — see {doc}`polaron`.)

## Choosing a method

- **Just starting out?** Use the default `tree-tdvp2`, or `tebd` — both grow their
  own bonds, so you only choose `dt` and `trunc_eps`.
- **Just want a number, fast, and accurate?** Use `mpo-tdvp1` — a
  Schrödinger-picture chain MPO with a fixed bond dimension is usually the most
  accurate per unit cost, provided you know a large-enough `bond_dim`.
- **Don't know the required bond dimension?** Use a 2-site or adaptive variant
  (`mpo-tdvp2`, `mpo-dtdvp`, `tree-tdvp2`, `tree-tebd`); they grow the bonds from
  a product state and report the peak bond in `result.max_bond`.
- **Want the interaction picture but cheaper per step?** Use `trotter-mpo` instead
  of `tebd`: identical frame and physics, but the propagator is applied as one
  exact low-bond MPO with no swap network (~1.6× faster at equal bond dimension).
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

**Truncation — `trunc_eps` is the knob, `bond_dim` is the safety net.**
A tensor-network state is only as good as what you throw away, and the package
gives you two independent controls:

- **`trunc_eps`** (default `1e-4`) is the *accuracy* control: after each SVD,
  singular values below this threshold are discarded. This alone determines the
  bond dimension — the state grows exactly as much as the physics demands.
- **`bond_dim`** (default `None` = **unlimited**) is an optional *hard cap*, for
  when you need a guaranteed memory bound and are willing to accept a larger
  error to get it. `result.max_bond` always reports what was actually used.

The usual workflow is to set `trunc_eps` to the accuracy you need, leave
`bond_dim` unset, and watch `result.max_bond`; introduce a cap only if the bond
grows beyond what you can afford. Tightening `trunc_eps` far below your target
accuracy is the most common way to waste time here — `1e-4` is a sensible default,
`1e-6` is already demanding, and the cost climbs steeply.

Two exceptions to be aware of:

- **Fixed-bond methods** (`mpo-tdvp1`, `mpo-ip-tdvp1`, `tree-tdvp`,
  `polaron-tdvp1`) cannot grow a bond at all, and the bond-adaptive `mpo-dtdvp` /
  `polaron-dtdvp` need a ceiling to grow towards. These **require** an explicit
  `bond_dim` and raise a clear error without one.
- For the tree methods on heavily-entangled multi-bath geometries an over-tight
  `trunc_eps` inflates cost sharply — see {doc}`../systems/fishbone`.

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
trotter_mpo
polaron
chain_mpo
star_mpo
tree
```
