# Interaction picture · multichannel — one bath, several couplings

Selected by the bath, not by `method`: pass a *list* of `coupling` operators to
{py:class}`~fishbonett.bath.spec.Bath` and the multichannel engine is used
automatically.  Requires `discretization="legendre"` (the channels share Gauss
nodes).  For independent baths (no cross-correlation) use
{py:class}`~fishbonett.models.fishbone.TreeFishbone` with one `Bath` per site instead.

The other representations assume the bath couples to the system through a *single*
operator $O$. A multichannel bath couples through **several** operators
$A_1, A_2, \dots$ that share the *same* modes:

$$
H_{sb} = \sum_k \big(\textstyle\sum_c A_c\, g^{(c)}_k\big)(b_k + b_k^\dagger).
$$

This is genuinely different from several independent baths. Independent baths have
independent noise; here one set of modes drives every channel, so the channels are
**cross-correlated** — the fluctuations they impose on the system are not
statistically independent. Physically this is the difference between a molecule
whose electronic gap and inter-site coupling are modulated by the *same* vibrations
versus by unrelated ones.

## Structure

After the interaction-picture transformation the coupling is matrix-valued: each
mode carries a matrix $A(d_n(t))$ rather than a scalar times $O$. The chain mapping
is done by a block Lanczos procedure seeded with all channels at once, and the
finite-temperature thermofield doubling is folded in through `temp_factor`.

Because the system must not be absorbed into a bath site, the run is routed through
the **tree** engine so the system keeps its own site with the shared-mode star
attached to it — see {doc}`/models/composite_multichannel`.

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

r = SystemBath(h=0.3 * sigma_z + 0.8 * sigma_x,
                coupling=[sigma_z, sigma_x], bath=bath).run(
        dt=0.02, t_max=2.0, observables={"sz": sigma_z})
```

## Notes

- Passing a *single* operator gives an ordinary bath; passing a *list* is what makes
  it multichannel. The `method` argument is ignored for multichannel baths — the
  tree route is used regardless.
- For the builder see {py:mod}`fishbonett.frames.multichannel`.
- For several *independent* baths on one site (or on several sites), use
  {py:class}`~fishbonett.models.fishbone.TreeFishbone` with one `Bath` per bath instead —
  see {doc}`/models/fishbone`.
