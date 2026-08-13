# Propagation methods

```{admonition} At a glance
:class: tip
- **Provides** — the `method=` argument of
  {py:meth}`SystemBath.run <fishbonett.simulate.SystemBath.run>`; 14 names in 5
  frames. The frame taxonomy table below is the map.
- **Don't know which?** — the default `tree-tdvp2`, or `tebd`. Both grow their own
  bonds, so you only pick `dt` and `trunc_eps`.
- **Truncation** — `trunc_eps` (accuracy, default `1e-4`) and `bond_dim`
  (optional hard cap, `None` = unlimited); or pass one
  {py:class}`~fishbonett.linalg.Truncation`. Fixed-bond methods (`mpo-tdvp1`,
  `mpo-ip-tdvp1`, `tree-tdvp`, `polaron-tdvp1`, `mpo-dtdvp`) **require** a cap.
- **Same interface everywhere** — every method takes the same `dt`/`t_max` and
  returns the same {py:class}`~fishbonett.simulate.Result`, so cross-checking one
  method against another is the easiest way to validate a calculation.
- **Not chosen by `method`** — a multichannel bath routes automatically; see
  {doc}`/methods/interaction/multichannel`.
```

Every method here solves the **same problem**: a system coupled to a harmonic bath
that has been chain-mapped into a 1D chain of effective modes ({doc}`../bath`).
They differ only in *how* they represent and propagate that chain, and they are all
selected by the `method` argument of
{py:meth}`SystemBath.run <fishbonett.simulate.SystemBath.run>`.

A method name encodes three choices:

1. **Frame** — the picture the model is written in, which fixes *what the
   Hamiltonian looks like* and therefore **how much entanglement the state carries**.
2. **State ansatz** — a matrix-product state (MPS) for a chain, or a tree tensor
   network (TTN) when the geometry branches.
3. **Integrator** — a Trotter splitting (TEBD, or an exact MPO propagator), or the
   time-dependent variational principle (TDVP) in 1-site, 2-site or bond-adaptive
   form.  This sets **the cost per step and the error in `dt`**.

These are **not** independent. The frame decides which integrators are even
available, because it decides two structural properties of the Hamiltonian:

- **Is it time-dependent?** A frame that rotates something out pays for it with
  explicit time dependence. TDVP wants a *static* MPO — it is built once, energy is
  conserved, and the projector-splitting error analysis assumes a fixed $H$. In a
  time-dependent frame the MPO must be rebuilt every step at the step midpoint,
  which still works to second order but forfeits energy conservation and adds
  per-step cost.
- **Which terms commute?** Trotter-type propagators are only cheap when the pieces
  they split are local *and* their non-commutation is mild. Occasionally a frame
  makes a whole family of terms commute exactly — and then the propagator can be
  written in closed form instead of split at all, which is precisely what
  {doc}`/methods/interaction/trotter_mpo` exploits.

## The frame taxonomy

A frame is therefore a **pair**: the *picture* (is $H$ time-dependent?) and the
*bath representation* (how are the modes wired?). Both halves constrain what works.

| picture \\ representation | **chain** (TEDOPA, nearest-neighbour) | **star** (no chain mapping) | **multichannel** (shared modes, several couplings) |
|---|---|---|---|
| **Schrödinger** ($H$ static) | {doc}`schrodinger/chain` — `mpo-tdvp1/tdvp2/dtdvp` | *coherent but not provided* | — |
| — *polaron chain* (static after Lang–Firsov) | {doc}`schrodinger/polaron_chain` — `polaron`, `polaron-tdvp1/tdvp2/dtdvp` | ✗ entanglement-catastrophic in an MPS | — |
| **interaction** ($H(t)$) | {doc}`interaction/tebd`, {doc}`interaction/trotter_mpo`, {doc}`interaction/tree` | {doc}`interaction/star_mpo` — `mpo-ip-tdvp1/tdvp2` | {doc}`interaction/multichannel` |

Reading the rows and columns:

- The **picture** (row) decides the *integrator*. Static rows host TDVP with a
  once-built MPO and exact energy conservation; the time-dependent row must rebuild
  gates/MPOs each step.
- The **representation** (column) decides the *ansatz*. A chain is
  nearest-neighbour, so an MPS has locality to exploit; a star has none, but also no
  mode–mode terms; multichannel shares modes across couplings, so the channels are
  cross-correlated and the system must keep its own site.
- The **polaron chain** is listed under Schrödinger because that is what it *is*
  after the transform: time-independent. It is the only representation that combines
  a static Hamiltonian with low entanglement.

## The frames in detail

### Schrödinger picture / chain — static $H$

The bare chain-mapped Hamiltonian, nothing rotated out. $H$ is time-independent
and strictly nearest-neighbour, so it has a small exact MPO built **once**. This is
the natural home for TDVP.

| ``method``    | integrator                       | bond growth         | page |
|---------------|----------------------------------|---------------------|------|
| ``mpo-tdvp1`` | chain MPO, 1-site TDVP           | fixed               | {doc}`/methods/schrodinger/chain` |
| ``mpo-tdvp2`` | chain MPO, 2-site TDVP           | SVD truncation      | {doc}`/methods/schrodinger/chain` |
| ``mpo-dtdvp`` | chain MPO, bond-adaptive DTDVP   | precision threshold | {doc}`/methods/schrodinger/chain` |

The cost of this frame is entanglement: nothing has been removed, so the state
carries the full system–bath correlation and the bond dimension is the largest of
the three frames for a given accuracy.

### Interaction picture — time-dependent $H(t)$ (chain, star, multichannel)

The free-bath evolution is rotated out, leaving only the system–bath coupling,
$H_{sb}(t) = A_s \otimes \sum_n [d_n(t) b_n + \mathrm{h.c.}]$. Entanglement is now
purely *system-mediated* and much smaller — but $H$ is time-dependent, so every
propagator here rebuilds its gates or its MPO each step.

| ``method``       | integrator                                   | bond growth    | page |
|------------------|----------------------------------------------|----------------|------|
| ``tebd``         | MPS, swap-network Trotter gates              | SVD truncation | {doc}`/methods/interaction/tebd` |
| ``trotter-mpo``  | MPS, **exact** conditional-displacement MPO  | SVD truncation | {doc}`/methods/interaction/trotter_mpo` |
| ``mpo-ip-tdvp1`` | star MPO, 1-site TDVP (rebuilt at midpoint)  | fixed          | {doc}`/methods/interaction/star_mpo` |
| ``mpo-ip-tdvp2`` | star MPO, 2-site TDVP (rebuilt at midpoint)  | SVD truncation | {doc}`/methods/interaction/star_mpo` |
| ``tree-tdvp``    | binary-tree TTN, 1-site TDVP                 | fixed          | {doc}`/methods/interaction/tree` |
| ``tree-tdvp2``   | binary-tree TTN, 2-site TDVP                 | SVD truncation | {doc}`/methods/interaction/tree` |
| ``tree-tebd``    | binary-tree TTN, TEBD                        | SVD truncation | {doc}`/methods/interaction/tree` |

This frame has a property the other two lack: **all the coupling terms
$A_s\otimes X_n$ commute with one another**, because the bath term that would
spoil it has been rotated away. That is what makes `trotter-mpo` possible — the
multimode propagator factorizes exactly into a conditional displacement, with no
splitting error at all. It is unique to this frame.

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

The restriction is physical rather than algorithmic: the transform needs $T=0$ and
a bath with $\int J(\omega)/\omega^2\,d\omega$ finite (gapped or super-ohmic).

## Which propagator suits which frame

| propagator | Schrödinger (static) | interaction picture (time-dep.) | polaron (static) |
|---|---|---|---|
| **Trotter gates (TEBD)** | possible — the chain is nearest-neighbour — but not currently provided | ✅ `tebd`; gates rebuilt each step, and a swap network is needed because *every* mode couples to the system | ✅ `polaron`; gates are **static**, built once and reused |
| **Exact conditional-displacement MPO** | ❌ the coupling does not commute with the free-bath term, which is still present | ✅ `trotter-mpo` — the only frame where the factorization is exact | ❌ the dressed tunneling does not commute with the free-chain hopping |
| **TDVP (1-site / 2-site / adaptive)** | ✅ `mpo-*` — MPO built once, energy conserved | ⚠️ `mpo-ip-*`, `tree-*` — works, but the MPO must be rebuilt at each step midpoint and energy is no longer conserved | ✅ `polaron-*` — MPO built once |

Reading the table by column: **TDVP** wants a static picture (Schrödinger chain or
polaron chain); **Trotter gates** work anywhere but are cheapest where the gates are
static (polaron) or the coupling is local; and the **exact conditional-displacement
MPO** exists only in the interaction picture, because only there do all the coupling
terms commute.

All methods take the same `dt` / `t_max` and return the same
{py:class}`~fishbonett.simulate.Result`, so switching engines is a one-word
change and the results are directly comparable — which also makes cross-checking
one method against another the easiest way to validate a calculation.  The example
below runs the same spin-boson model through several and prints the final
population:

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)

for method in ["tebd", "trotter-mpo",
               "mpo-tdvp1", "mpo-tdvp2", "mpo-dtdvp",
               "mpo-ip-tdvp1", "mpo-ip-tdvp2",
               "tree-tdvp", "tree-tdvp2", "tree-tebd"]:
    r = model.run(dt=0.02, t_max=2.0, method=method, bond_dim=100,
                  observables={"sz": sigma_z})
    print(f"{method:14s} <sz>(t_end) = {r.expect['sz'][-1]:+.4f}")
```

(The polaron methods are omitted from that loop only because they need `T=0` and a
gapped bath — see {doc}`/methods/schrodinger/polaron_chain`.)

## Choosing a method

Choose the **frame first** — it decides how hard the problem is to represent — then
the propagator within it.

**Step 1: pick the frame.**

- Is the bath at **finite temperature**, or is $\int J(\omega)/\omega^2$ divergent
  (e.g. strictly ohmic)? → the polaron frame is unavailable; use the
  **interaction picture** (it handles a thermalized/signed domain natively) or the
  **Schrödinger picture**.
- Is the coupling **strong**, so that static reorganization dominates? → the
  **polaron** frame carries markedly less entanglement (measured: peak bond
  entropy 7.6 vs 16, bond 21 vs 30 at strong coupling), if its $T=0$ requirement
  is met.
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
  coupling `O`, and any `initial` state (see {doc}`../systems/spin_boson`).

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

schrodinger/chain
schrodinger/polaron_chain
interaction/tebd
interaction/trotter_mpo
interaction/tree
interaction/star_mpo
interaction/multichannel
```
