# Interaction picture · star — MPO + TDVP

`mpo-ip-tdvp1` (fixed bond) and `mpo-ip-tdvp2` (adaptive) run TDVP on a star-geometry
MPO — no chain mapping, every mode coupled directly to the system.  The MPO is
rebuilt at each step's midpoint (so no energy conservation, unlike the static
Schrödinger frame).  Less entanglement than a chain, but no locality for the MPS.

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

```{note}
That bond-2 structure is the same fact exploited by {doc}`/methods/interaction/trotter_mpo`: because the
coupling is a single product $\sigma_z \otimes \sum_j d_j (a_j + a_j^\dagger)$ and
all its mode terms commute, the operator connecting the system to *every* mode
needs only a rank-2 string — one channel per eigenvalue of the coupling operator.
Here it is used to write the **Hamiltonian** as an MPO for TDVP; there it is used
to write the **propagator** as an MPO for a Trotter step.
```

### Star or chain?

The star and the chain describe the same bath — the chain is just the Lanczos
tridiagonalization of the star — but they distribute *entanglement* very
differently, and that is the whole basis for choosing between them.

- In the **chain**, the system touches only mode $b_0$, and correlations propagate
  outward at a finite speed (a light cone). This locality is what makes an MPS
  ordering natural: a cut far down the chain has seen little of the dynamics.
- In the **star**, every mode couples to the system directly. There is no notion of
  distance between modes, so an MPS ordering of the star has no locality to exploit
  — a single cut must carry the correlations between the system and *all* modes on
  the far side at once.

This is why TEDOPA chain-maps in the first place, and why the star geometry here is
paired with a small fixed/adaptive bond and an interaction picture that keeps the
accumulated correlation small: the star pays off when the residual entanglement is
low enough that the *absence* of the chain's mode–mode terms (and their Trotter
error) is the dominant saving.

Two variants:

- **`mpo-ip-tdvp1`** — 1-site TDVP at a **fixed** bond dimension `bond_dim`
  (required — a 1-site sweep cannot grow a bond; see {doc}`/methods/schrodinger/chain`).
- **`mpo-ip-tdvp2`** — 2-site TDVP, **growing** the bond from the product state
  by SVD truncation (`trunc_eps`, optionally capped by `bond_dim`);
  `result.max_bond` reports the peak bond.

The star MPO engine shares {py:mod}`fishbonett.evolve.tdvp` with the
Schrödinger-picture chain methods (re-exported as {py:mod}`fishbonett.evolve.tdvp`).

## Example

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)

r = model.run(dt=0.02, t_max=2.0, method="mpo-ip-tdvp1", bond_dim=80,
              observables={"sz": sigma_z})
r.expect["sz"]

r2 = model.run(dt=0.02, t_max=2.0, method="mpo-ip-tdvp2", bond_dim=120,
               trunc_eps=1e-4, observables={"sz": sigma_z})
r2.max_bond
```

## Low-level driver

```python
from fishbonett.evolve.tdvp import run_ip_tdvp1

t, sz = run_ip_tdvp1(bath.spectral_density(), (-25, 36), V=1.0,
                     n_chain=40, d=20, dt=0.025, nsteps=80, D=100)
```

## Notes

- Because the MPO is rebuilt every step, these methods have a slightly higher
  per-step overhead than the fixed chain MPO, but often reach a given accuracy at
  a **smaller bond dimension** thanks to the interaction picture.
- `h` and the coupling `O` are carried as matrices, so a general Hermitian system
  and coupling of any dimension work (the illustration uses `sigma_z`); see
  {doc}`/models/spin_boson`.
