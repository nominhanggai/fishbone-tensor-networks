# `mpo-ip-tdvp1` / `mpo-ip-tdvp2` — interaction-picture star MPO + TDVP

These methods evolve the bath in its **star** geometry — every discretized mode
coupled directly to the spin, with no chain mapping — in the interaction picture,
using a time-dependent MPO integrated by TDVP.

## Theory

Discretizing $J(\omega)$ on a Gauss grid gives a *star* Hamiltonian

$$
H = \tfrac{\epsilon}{2}\sigma_z + V\sigma_x
    + \sum_j \omega_j\, a_j^\dagger a_j
    + \sigma_z \sum_j g_j\,(a_j + a_j^\dagger),
$$

with mode frequencies $\omega_j$ and couplings $g_j$ read straight off the
discretization (no Lanczos tridiagonalization).  Moving to the interaction
picture with respect to the free bath $\sum_j \omega_j a_j^\dagger a_j$ removes
the on-site frequencies and the inter-mode structure entirely: what remains is a
spin coupled to each mode through a **time-dependent** coupling

$$
d_j(t) = \sum_k (\text{star}\!\to\!\text{normal})_{jk}\; g_k\, e^{-i\omega_k t},
$$

so at any instant the Hamiltonian is just
$\tfrac{\epsilon}{2}\sigma_z + V\sigma_x + \sigma_z\sum_j d_j(t)(a_j + a_j^\dagger)$.
This is a **bond-2 star MPO** (the spin threads a single rank-2 string to all
modes); it is rebuilt at each step's **midpoint** $t_{\mathrm{mid}}$ and the state
is advanced with a TDVP sweep.  Because there is no free bath evolution left to
resolve, the accumulated entanglement is small.

Two variants:

- **`mpo-ip-tdvp1`** — 1-site TDVP at a **fixed** bond dimension `bond_dim`.
- **`mpo-ip-tdvp2`** — 2-site TDVP, **growing** the bond from the product state
  by SVD truncation (to `bond_dim` / `trunc_eps`); `result.max_bond` reports the
  peak bond.

The star MPO engine shares {py:mod}`fishbonett.evolve.tdvp` with the
Schrödinger-picture chain methods (re-exported as {py:mod}`fishbonett.mpo`).

## Example

```python
import numpy as np
from fishbonett.simulate import Bath, SpinBoson
from fishbonett.stuff import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SpinBoson(h=sigma_x, coupling=sigma_z, bath=bath)

r = model.run(dt=0.02, t_max=2.0, method="mpo-ip-tdvp1", bond_dim=80,
              observables={"sz": sigma_z})
r.expect["sz"]

r2 = model.run(dt=0.02, t_max=2.0, method="mpo-ip-tdvp2", bond_dim=120,
               trunc_eps=1e-9, observables={"sz": sigma_z})
r2.max_bond
```

## Low-level driver

```python
from fishbonett.mpo import run_ip_tdvp1

t, sz = run_ip_tdvp1(bath.spectral_density(), (-25, 36), V=1.0,
                     n_chain=40, d=20, dt=0.025, nsteps=80, D=100)
```

## Notes

- Because the MPO is rebuilt every step, these methods have a slightly higher
  per-step overhead than the fixed chain MPO, but often reach a given accuracy at
  a **smaller bond dimension** thanks to the interaction picture.
- Like the chain MPO methods, they assume a two-level, `sigma_z`-coupled system;
  use `tebd` for a general coupling (see {doc}`tebd`).
