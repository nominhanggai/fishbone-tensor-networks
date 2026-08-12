# Composite systems and multichannel baths

The "system" need not be a bare two-level spin, and a single bath may couple
through more than one operator.  Both cases are handled by keeping every degree of
freedom on **its own site** — fattening them onto one tensor defeats the
tensor-network advantage.

## Composite systems: spin + vibration

A vibrational mode is just another *system site* (with no bath of its own) coupled
to the spin; the bath attaches to the spin.  Build it as a two-site
{py:class}`~fishbonett.treebone.TreeFishbone`:

```python
import numpy as np
from fishbonett.treebone import TreeFishbone
from fishbonett.simulate import Bath
from fishbonett.operators import sigma_x, sigma_z

b = np.diag(np.sqrt(np.arange(1, 4)), 1)           # vibration annihilation (dv=4)
h_spin = 0.25 * sigma_z + sigma_x
h_vib = 1.5 * (b.T @ b)
spin_vib = 0.4 * np.kron(sigma_z, b + b.T)         # coupling on the (spin, vib) edge

fb = TreeFishbone(sites=[h_spin, h_vib], edges=[(0, 1, spin_vib)],
                  baths=[Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(0, 40),
                              n_modes=20, phys_dim=8, coupling=sigma_z), None])
res = fb.run(dt=0.02, t_max=1.0, bond_dim=80)
res.rdm[0, 0]      # 2x2 reduced density matrix of the spin site
res.rdm[0, 1]      # dv x dv reduced density matrix of the vibration site
```

Because the spin and the vibration live on separate sites, the bath only ever
"sees" the spin — the vibration is entangled with the bath *indirectly*, through
the spin, exactly as the physics dictates.  The bath can equally well be attached
to the spin site of such a tree while the spin also carries the multichannel star
below; the two features compose.

```{tip}
`BosonicBath` will also let you put `spin ⊗ vibration` on a single `d = 2·d_vib`
site, but that grows the local Hilbert space and defeats the MPS advantage.  Keep
each DOF on its own site with `TreeFishbone`.
```

## Multichannel single bath

One bath coupled to the system through **several** operators (e.g. `sigma_z` *and*
`sigma_x`) is distinct from two independent baths: the channels share the same
modes and therefore **cross-correlate**.  Give the {py:class}`~fishbonett.simulate.Bath`
a list of couplings and (optionally) a list of per-channel spectral densities:

```python
from fishbonett.simulate import BosonicBath

mc = Bath(J=[lambda w: 0.2 * w * np.exp(-w / 5),   # sigma_z channel
             lambda w: 0.1 * w * np.exp(-w / 8)],  # sigma_x channel (different J)
          coupling=[sigma_z, sigma_x], domain=(0, 40), n_modes=30, phys_dim=8)

res = BosonicBath(h=sigma_x, coupling=[sigma_z, sigma_x], bath=mc).run(
        dt=0.02, t_max=2.0, bond_dim=100, observables={"sz": sigma_z})
```

Internally the bath becomes a **shared-mode star** attached to the spin site: all
channels use the same Gauss–Legendre nodes $\omega_k$, and mode $k$ couples
through the combined operator

$$
M_k = \sum_c g_{c,k}\, O_c, \qquad g_{c,k} = \sqrt{J_c(\omega_k)\, w_k / \pi},
$$

so the channels genuinely cross-correlate rather than acting as independent baths.
Passing a multichannel `Bath` to `BosonicBath` routes it through
{py:class}`~fishbonett.treebone.TreeFishbone` so the spin stays on its own site.

```{note}
A multichannel bath must use the `'legendre'` discretization — the Gauss nodes are
shared across channels, whereas the measure-adapted ORTHPOL nodes are per-density
and would not line up.
```

If `J` is a single callable it is shared by every channel; if it is a list it must
have one entry per coupling operator.
