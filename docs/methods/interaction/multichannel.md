# Multichannel — one bath, several couplings

`multichannel-ip` (interaction picture) and `tree-tebd-static` (Schrödinger
picture, the default).  The model is selected by the **bath**, not by `method`:
pass a *list* of `coupling` operators to
{py:class}`~fishbonett.bath.spec.Bath` and the multichannel engine is used
automatically; `method` then picks the frame.  Requires
`discretization="legendre"` (the channels share Gauss nodes).  For independent
baths (no cross-correlation) use
{py:class}`~fishbonett.models.fishbone.TreeFishbone` with one `Bath` per site instead.

The other models assume the bath couples to the system through a *single*
operator $O$. A multichannel bath couples through **several** operators
$A_1, A_2, \dots$ that share the *same* modes:

$$
H_{sb} = \sum_k \big(\textstyle\sum_c A_c\, g^{(c)}_k\big)(b_k + b_k^\dagger),
\qquad g^{(c)}_k = \sqrt{J_c(\omega_k)\,w_k/\pi}.
$$

This is genuinely different from several independent baths. Independent baths have
independent noise; here one set of modes drives every channel, so the channels are
**cross-correlated** — the fluctuations they impose on the system are not
statistically independent. Physically this is the difference between a molecule
whose electronic gap and inter-site coupling are modulated by the *same* vibrations
versus by unrelated ones.

Note the shared nodes $\omega_k$ are what couples the channels, so they must come
from a measure that does not depend on which channel you ask — hence the
Gauss–Legendre requirement.  The measure-adapted `"tedopa"` nodes are placed
against each $J_c$ separately and so would not be shared; see {doc}`/bath`.

## The two frames

|  `method` | frame | bath modes carry | cost |
|---|---|---|---|
| `tree-tebd-static` (default) | Schrödinger | $\omega_k\,n_k$ on-site | $N$ sites |
| `multichannel-ip` | interaction | nothing (rotated out) | $N$ sites, time-dependent gates |

**Schrödinger (default).** The star is built explicitly, with each mode carrying
its frequency on-site and coupling to the system through $M_k=\sum_c g^{(c)}_k A_c$.
Because the system must not be absorbed into a bath site, the run is routed through
the **tree** engine so the system keeps its own site with the shared-mode star
attached to it — see {doc}`/models/composite_multichannel`.

**Interaction picture.** The free-bath evolution is rotated out, so the modes carry
no on-site term and the coupling becomes matrix-valued *and* time-dependent: mode
$n$ carries $A^{(n)}(t)=\sum_k A_k\,Q_{kn}\,e(\omega_k,t,\delta)$ rather than a
scalar times one operator.  Gates are rebuilt each step and applied with a swap
network, as in {doc}`/methods/interaction/tebd`.

The two agree, which is the useful thing about having both — measured against exact
diagonalization of the same shared-mode star (3 modes, $d=8$, two channels
$\sigma_z,\sigma_x$):

| method | $\langle\sigma_z\rangle$ vs exact | $\langle\sigma_x\rangle$ vs exact |
|---|---|---|
| `tree-tebd-static` | $5.4\times10^{-5}$ | $4.5\times10^{-5}$ |
| `multichannel-ip` | $6.4\times10^{-6}$ | $7.3\times10^{-6}$ |

The interaction picture is the more accurate of the two at equal $\Delta t$ because
the free-bath evolution is exact in it rather than Trotterized; its peak bond is
correspondingly a little larger (11 vs 7 here), since it carries the coupling in the
bonds instead of on the sites.

### The chain basis is a free choice here

In the interaction picture there are no mode–mode terms at all — the bath enters
only through the phases $e(\omega_k,t,\delta)$. The "chain" that
{py:meth}`~fishbonett.frames.multichannel.SystemBathMultiChannel.build` produces
is therefore just an orthogonal change of basis $b_k=\sum_n Q_{kn}c_n$, and

$$
\sum_k A_k\,e_k \otimes (b_k + b_k^\dagger)
 = \sum_n \Big(\sum_k A_k\,Q_{kn}\,e_k\Big)\otimes(c_n + c_n^\dagger)
$$

holds for **any** orthogonal $Q$. So `build(n=...)`'s Lanczos seed — basis state
$n$'s coupling profile $[A_k[n,n]]_k$ — changes only how the bath is spread across
sites (the entanglement and the per-site Fock truncation), not the answer. A plain
single-vector Lanczos is enough; no block Lanczos is needed.

Two consequences:

- The seed must be **nonzero**. A coupling set whose diagonal vanishes in the
  working basis (say channels $\sigma_x$ and $\sigma_y$ only) gives a zero seed and
  no chain; this raises rather than producing a silent `NaN`.
- All $N$ sites must be kept. Dropping sites here is not a chain truncation as it
  would be in the Schrödinger picture — with no hoppings, distant sites are not
  weakly coupled, so dropping them simply discards bath modes.

### Finite temperature

Both paths take it the same T-TEDOPA way as every other method: set `temperature`
or `beta` on the `Bath` and the density is thermalized onto a signed frequency axis
before discretization (see {doc}`/bath`).

{py:class}`~fishbonett.frames.multichannel.SystemBathMultiChannel` also has an
older constructor that does the thermofield doubling itself, from a bare $T=0$ star
plus a `temp` argument. Prefer the T-TEDOPA route unless you are handing it an
explicit discrete mode list: that `temp` is in **kelvin** with frequencies in
cm⁻¹, which is *not* the natural-units convention `Bath` uses, so mixing the two
silently rescales the temperature.
{py:meth}`~fishbonett.frames.multichannel.SystemBathMultiChannel.from_signed_star`
is the entry point that avoids the question, and is what `run` calls.

## Example

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

Jz = lambda w: 0.2 * w * np.exp(-w / 5)
Jx = lambda w: 0.1 * w * np.exp(-w / 8)

# one bath, two channels -> a list of J and a list of coupling operators
bath = Bath(J=[Jz, Jx], coupling=[sigma_z, sigma_x],
            domain=(0.0, 40.0), n_modes=20, phys_dim=8)

model = SystemBath(h=0.3 * sigma_z + 0.8 * sigma_x,
                      coupling=[sigma_z, sigma_x], bath=bath)

r = model.run(dt=0.02, t_max=2.0, observables={"sz": sigma_z})   # Schrodinger
r_ip = model.run(dt=0.02, t_max=2.0, method="multichannel-ip",
                 observables={"sz": sigma_z})                    # interaction
```

## Notes

- Passing a *single* operator gives an ordinary bath; passing a *list* is what makes
  it multichannel.  `method` selects the frame among the model's own propagators;
  asking for another model's method (e.g. `tebd`) raises and says so.
- There is no polaron frame here, for the same reason there is none for the `star`
  model: the polaron displacement acts on a collective mode spread over every
  shared mode at once, with no single site to localize it on.  See
  {py:mod}`fishbonett.models.registry`.
- For the builder see {py:mod}`fishbonett.frames.multichannel`.
- For several *independent* baths on one site (or on several sites), use
  {py:class}`~fishbonett.models.fishbone.TreeFishbone` with one `Bath` per bath instead —
  see {doc}`/models/fishbone`.
