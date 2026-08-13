# Propagation methods

All methods are selected by the `method` argument of
{py:meth}`SimpleSysBath.run <fishbonett.models.system_bath.SimpleSysBath.run>` and return the same
{py:class}`~fishbonett.models.result.Result`, so you can switch methods by changing one
string.  If you don't know which to pick, start with `tree-tdvp2` or `tebd` —
both grow their own bonds.

A method name picks two things at once: a **model** (what is coupled to what —
{doc}`../models/index`) and a **frame** (how the Hamiltonian is written down).
What remains is the **propagator**: a Trotter splitting (TEBD, or an exact MPO
propagator) or TDVP in 1-site, 2-site or bond-adaptive form, which sets the cost
per step and the error in `dt`.

The three are nested, not free: the model fixes the state geometry, and that plus
the frame decides which propagators apply at all.  Two structural properties do
the constraining:

- *Is $H$ time-dependent?*  A frame that rotates out the free bath makes $H$
  time-dependent.  TDVP wants a static MPO (built once, energy conserved); in a
  time-dependent frame the MPO must be rebuilt each step.
- *Which terms commute?*  The interaction picture makes all system–bath coupling
  terms commute, because the bath part has been rotated away.  That is what makes
  the exact conditional-displacement MPO ({doc}`/methods/interaction/trotter_mpo`)
  possible — the propagator factorizes without Trotter error.

## Model × frame

| model | Schrödinger ($H$ static) | interaction ($H(t)$) | polaron (static, low entanglement) |
|---|---|---|---|
| **`chain`** 1 system + 1 bath, 1D | {doc}`schrodinger/chain` — `mpo-tdvp1/tdvp2/dtdvp` | {doc}`interaction/tebd`, {doc}`interaction/trotter_mpo` | {doc}`schrodinger/polaron_chain` — `polaron`, `polaron-tdvp1/tdvp2/dtdvp` |
| **`star`** no chain mapping | {doc}`schrodinger/star_mpo` — `mpo-star-tdvp1/tdvp2` | {doc}`interaction/star_mpo` — `mpo-ip-tdvp1/tdvp2` | ✗ no site to localize the displacement on |
| **`mode-tree`** modes on a binary tree | ✗ chain hoppings are long-range on the tree | {doc}`interaction/tree` — `tree-tdvp`, `tree-tdvp2`, `tree-tebd` | ✗ same reason |
| **`multichannel`** several couplings, shared modes | {doc}`interaction/multichannel` — `tree-tebd-static` | {doc}`interaction/multichannel` — `multichannel-ip` | ✗ same reason as `star` |
| **`comb`** / **`site-tree`** several sites + baths | `tree-tebd-static` ({doc}`/models/fishbone`) | not implemented | not implemented |

Reading it: the `chain` model is the developed one and the only one with all three
frames.  The ✗ cells are **not** the same kind of blank as "not implemented" — they
are cells where the combination has been considered and rejected, and
{py:mod}`fishbonett.models.registry` records the reason for every one:

```python
from fishbonett.models.registry import describe_taxonomy
print(describe_taxonomy())        # every model, frame, and gap with its reason
```

```{note}
The `multichannel` model's **default** path is Schrödinger, not interaction picture:
with no `method` it routes through the tree engine, whose shared-mode star carries
the bath frequencies on-site.  Its interaction-picture path is a separate method,
`multichannel-ip`.  (The model itself is still selected by the *bath* — passing a
list of couplings — so `method` only chooses between those two.)
```

## The frames in detail

### Schrödinger picture — static $H$ (chain, star, multichannel)

The bare Hamiltonian, nothing rotated out. $H$ is time-independent, so its MPO is
built **once** — TDVP with exact energy conservation and no per-step rebuild error.

| ``method``          | integrator                       | bond growth         | page |
|---------------------|----------------------------------|---------------------|------|
| ``mpo-tdvp1``       | chain MPO, 1-site TDVP           | fixed               | {doc}`/methods/schrodinger/chain` |
| ``mpo-tdvp2``       | chain MPO, 2-site TDVP           | SVD truncation      | {doc}`/methods/schrodinger/chain` |
| ``mpo-dtdvp``       | chain MPO, bond-adaptive DTDVP   | precision threshold | {doc}`/methods/schrodinger/chain` |
| ``mpo-star-tdvp1``  | static **star** MPO, 1-site TDVP | fixed               | {doc}`/methods/schrodinger/star_mpo` |
| ``mpo-star-tdvp2``  | static **star** MPO, 2-site TDVP | SVD truncation      | {doc}`/methods/schrodinger/star_mpo` |
| ``tree-tebd-static``| tree TEBD, static gates          | SVD truncation      | {doc}`/methods/interaction/multichannel`, {doc}`/models/fishbone` |

The cost of this frame is entanglement: nothing has been removed, so the state
carries the full system–bath correlation and the bond dimension is the largest of
the three frames for a given accuracy.  In exchange it is the most accurate per
step — the static star methods agree with exact diagonalization to $\sim10^{-10}$,
better than any time-dependent frame can, because there is nothing to rebuild.

### Interaction picture — time-dependent $H(t)$ (chain, star, mode-tree, multichannel)

The free-bath evolution is rotated out, leaving only the system–bath coupling,
$H_{sb}(t) = A_s \otimes \sum_n [d_n(t) b_n + \mathrm{h.c.}]$. Entanglement is now
purely *system-mediated* and much smaller — but $H$ is time-dependent, so every
propagator here rebuilds its gates or its MPO each step.

| ``method``           | integrator                                   | bond growth    | page |
|----------------------|----------------------------------------------|----------------|------|
| ``tebd``             | MPS, swap-network Trotter gates              | SVD truncation | {doc}`/methods/interaction/tebd` |
| ``trotter-mpo``      | MPS, **exact** conditional-displacement MPO  | SVD truncation | {doc}`/methods/interaction/trotter_mpo` |
| ``mpo-ip-tdvp1``     | star MPO, 1-site TDVP (rebuilt at midpoint)  | fixed          | {doc}`/methods/interaction/star_mpo` |
| ``mpo-ip-tdvp2``     | star MPO, 2-site TDVP (rebuilt at midpoint)  | SVD truncation | {doc}`/methods/interaction/star_mpo` |
| ``tree-tdvp``        | binary-tree TTN, 1-site TDVP                 | fixed          | {doc}`/methods/interaction/tree` |
| ``tree-tdvp2``       | binary-tree TTN, 2-site TDVP                 | SVD truncation | {doc}`/methods/interaction/tree` |
| ``tree-tebd``        | binary-tree TTN, TEBD                        | SVD truncation | {doc}`/methods/interaction/tree` |
| ``multichannel-ip``  | MPS, swap-network gates, matrix-valued coupling | SVD truncation | {doc}`/methods/interaction/multichannel` |

Because the free-bath term has been rotated away, all coupling terms
$A_s\otimes X_n$ commute.  The multimode propagator therefore factorizes exactly
into a conditional displacement (`trotter-mpo`) with no Trotter error between
modes — something the other frames cannot do.

### Schrödinger picture / polaron chain — static $\tilde H$

The polaron transform additionally absorbs the *static* part of the coupling into a
displacement of the bath, leaving a free chain plus a dressed tunneling term. The
result is **time-independent** — which is why it belongs to the Schrödinger picture
rather than to a frame of its own — so it supports both a static MPO (TDVP) and
static Trotter gates built once, while carrying interaction-picture-like
entanglement.

| ``method``        | integrator                             | bond growth         | page |
|-------------------|----------------------------------------|---------------------|------|
| ``polaron``       | MPS chain, **static** Trotter gates    | SVD truncation      | {doc}`/methods/schrodinger/polaron_chain` |
| ``polaron-tdvp1`` | polaron MPO, 1-site TDVP               | fixed               | {doc}`/methods/schrodinger/polaron_chain` |
| ``polaron-tdvp2`` | polaron MPO, 2-site TDVP               | SVD truncation      | {doc}`/methods/schrodinger/polaron_chain` |
| ``polaron-dtdvp`` | polaron MPO, bond-adaptive DTDVP       | precision threshold | {doc}`/methods/schrodinger/polaron_chain` |

The restriction is on the spectral density, not the temperature: the bath must
have $\int J(\omega)/\omega^2\,d\omega$ finite (gapped or super-ohmic).  Finite
temperature works through T-TEDOPA thermalization of the spectral density, the
same way the interaction-picture chain handles it.

## Which propagator suits which frame

| propagator | Schrödinger (static) | interaction picture (time-dep.) | polaron (static) |
|---|---|---|---|
| **Trotter gates (TEBD)** | ✅ `tree-tebd-static` on the tree engine; the 1D chain could use it too but is only wired for MPO/TDVP | ✅ `tebd`, `multichannel-ip`; gates rebuilt each step, and a swap network is needed because *every* mode couples to the system | ✅ `polaron`; gates are **static**, built once and reused |
| **Exact conditional-displacement MPO** | ❌ the coupling does not commute with the free-bath term, which is still present | ✅ `trotter-mpo` — the only frame where the factorization is exact | ❌ the dressed tunneling does not commute with the free-chain hopping |
| **TDVP (1-site / 2-site / adaptive)** | ✅ `mpo-*`, `mpo-star-*` — MPO built once, energy conserved | ⚠️ `mpo-ip-*`, `tree-*` — works, but the MPO must be rebuilt at each step midpoint and energy is no longer conserved | ✅ `polaron-*` — MPO built once |

Reading the table by column: **TDVP** wants a static picture (Schrödinger chain or
polaron chain); **Trotter gates** work anywhere but are cheapest where the gates are
static (polaron) or the coupling is local; and the **exact conditional-displacement
MPO** exists only in the interaction picture, because only there do all the coupling
terms commute.

All methods take the same `dt` / `t_max` and return the same
{py:class}`~fishbonett.models.result.Result`, so switching engines is a one-word
change and the results are directly comparable — which also makes cross-checking
one method against another the easiest way to validate a calculation.  The example
below runs the same spin-boson model through several and prints the final
population:

```python
import numpy as np
from fishbonett import Bath, SimpleSysBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SimpleSysBath(h=sigma_x, coupling=sigma_z, bath=bath)

for method in ["tebd", "trotter-mpo",
               "mpo-tdvp1", "mpo-tdvp2", "mpo-dtdvp",
               "mpo-ip-tdvp1", "mpo-ip-tdvp2",
               "tree-tdvp", "tree-tdvp2", "tree-tebd"]:
    r = model.run(dt=0.02, t_max=2.0, method=method, bond_dim=100,
                  observables={"sz": sigma_z})
    print(f"{method:14s} <sz>(t_end) = {r.expect['sz'][-1]:+.4f}")
```

(The polaron methods need a gapped or super-ohmic bath —
$\int J(\omega)/\omega^2\,d\omega$ finite — see
{doc}`/methods/schrodinger/polaron_chain`.)

## Choosing a method

Choose the **frame first** — it decides how hard the problem is to represent — then
the propagator within it.

**Step 1: pick the frame.**

- Is $\int J(\omega)/\omega^2$ divergent (e.g. strictly ohmic)? → the polaron
  frame is unavailable; use the **interaction picture** or the **Schrödinger
  picture**.
- Is the coupling **strong**, so that static reorganization dominates? → the
  **polaron** frame carries less entanglement (peak bond entropy 7.6 vs 16,
  bond 21 vs 30 at strong coupling).  It handles finite temperature the same
  way the other frames do, via T-TEDOPA thermalization.
- Otherwise the **interaction picture** is the general-purpose choice: much less
  entanglement than the Schrödinger picture, no restrictions.
- Use the **Schrödinger picture** when you want TDVP's exact energy conservation,
  or a static MPO built once, and the bond dimension is affordable.

**Step 2: pick the propagator within that frame.**

- **Don't know the required bond dimension?** Take a bond-growing method — any
  `*-tdvp2`, `tree-tebd`, `tebd`, `trotter-mpo` or `polaron`. They grow from a
  product state and report the peak in `result.max_bond`.
- **Know a good `bond_dim` and want maximum accuracy per unit cost?** Take a
  1-site TDVP (`mpo-tdvp1`, `polaron-tdvp1`) — no truncation error at all, just a
  projection error.
- **Want adaptive bonds at 1-site cost?** `mpo-dtdvp` / `polaron-dtdvp`.
- **In the interaction picture and want the cheapest step?** `trotter-mpo` rather
  than `tebd`: identical frame and physics, but the propagator is applied as one
  exact low-bond MPO with no swap network (~1.6× faster at equal bond dimension).
- **In the polaron frame?** `polaron-dtdvp` is usually best — 1-site TDVP avoids
  the $O(d^4)$ boson–boson gates that the static-gate TEBD sweep has to form.

**Step 3: the special cases.**

- **Long chains where entanglement piles up in the middle?** The `tree-*`
  methods keep the high-bond region `O(log n)` edges deep instead of `O(n)`
  (see {doc}`/methods/interaction/tree`).
- **Several coupling channels sharing one set of modes?** That is chosen by the
  bath, not by `method` — see {doc}`/methods/interaction/multichannel`.
- **Non-`sigma_z` coupling, a non-two-level system, or a custom initial state?**
  Every engine supports these — a Hermitian `h` of any dimension, a Hermitian
  coupling `O`, and any `initial` state (see {doc}`../models/spin_boson`).

**Just starting out?** Take the default `tree-tdvp2`, or `tebd` — both grow their
own bonds, so the only things you have to choose are `dt` and `trunc_eps`.

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
  `trunc_eps` inflates cost sharply — see {doc}`../models/fishbone`.

**Reading the result.** Every method returns a
{py:class}`~fishbonett.models.result.Result`:

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

schrodinger/chain
schrodinger/star_mpo
schrodinger/polaron_chain
interaction/tebd
interaction/trotter_mpo
interaction/tree
interaction/star_mpo
interaction/multichannel
```
