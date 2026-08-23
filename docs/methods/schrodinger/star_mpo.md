# Schrödinger-star representation — MPO + TDVP

`schrodinger-star-tdvp1` (fixed bond) and `schrodinger-star-tdvp2` (adaptive) run TDVP on a
**static** star-representation MPO — no chain mapping, every mode coupled directly to the
system, and nothing rotated out.  Because $H$ is time-independent the MPO is built
**once**, which makes these the most accurate methods in the package at a given
`dt`: there is no per-step rebuild error at all.

## Theory

A finite star discretization gives

$$
H = H_{\mathrm{sys}} + \sum_k \omega_k\, a_k^\dagger a_k
    + O \otimes \sum_k g_k\,(a_k + a_k^\dagger),
$$

with the mode frequencies $\omega_k$ and couplings $g_k$ read straight off the
discretization.  Unlike {doc}`/methods/interaction/star_mpo`, the free-bath term
$\sum_k \omega_k a_k^\dagger a_k$ is kept, so $H$ is static and the MPO is assembled
once and reused for every step.

### The MPO needs a third channel

Written as a compressed product-sum MPO over sites $[\,\text{system}, a_0, \dots,
a_{N-1}]$, the interaction-picture star MPO needs bond 2: one channel carrying $O$
rightward to meet each mode's $(a_k+a_k^\dagger)$, and one carrying the finished
terms.  Keeping the free bath adds terms that touch **no** system operator —
$\omega_k a_k^\dagger a_k$ standing alone — and those cannot ride either existing
channel.  They need a third, `START`, which passes the identity rightward until it
emits an on-site $\omega_k n_k$ and closes:

```
        CARRY : O has been placed, waiting for a mode
        DONE  : the term is complete, identity to the right edge
        START : nothing placed yet, identity to the right

system  ->  CARRY: O        DONE: H_sys       START: 1
mode k  ->  CARRY->CARRY: 1     CARRY->DONE: (a+a^dag)
            START->START: 1     START->DONE: w_k n_k
            DONE ->DONE : 1
```

so the bond profile is $[1, 3, 3, \dots, 3, 1]$ — one larger than the
interaction-picture star, and still independent of $N$.  The last mode closes to
bond 1 and must emit **no** identity from `START` or `CARRY`, or the operator would
contain a term with the coupling left dangling. This construction is the
Hermitian MPO returned by `SchrodingerRepresentation.tdvp_mpo`.

### Star or chain?

Same trade-off as in the interaction picture, and worth restating because in this
representation it cuts the other way.

- The **chain** gives the MPS locality to exploit: the system touches only $b_0$ and
  correlations spread outward at a finite speed.  The price is $N$ mode–mode
  hoppings.
- The **star** has no notion of distance between modes, so an MPS ordering of it has
  no locality: a single cut must carry the correlation between the system and *every*
  mode on the far side.  The saving is that there are no mode–mode terms at all.

In the interaction picture the residual entanglement is small enough that the star
often wins.  Here it is not: nothing has been rotated out, so the state carries the
full system–bath correlation *and* the star interaction graph gives no locality to help
represent it.  Expect the bond dimension to grow faster than for
{doc}`/methods/schrodinger/chain` on the same problem.

The static star is also useful as an independent numerical check: it shares no
Hamiltonian-MPO construction with the chain representation beyond the TDVP sweep.
Agreement after independently converging both calculations is therefore a useful
cross-validation.

Two variants:

- **`schrodinger-star-tdvp1`** — 1-site TDVP at a **fixed** bond dimension `bond_dim`
  (required — a 1-site sweep cannot grow a bond; see
  {doc}`/methods/schrodinger/chain`).
- **`schrodinger-star-tdvp2`** — 2-site TDVP, **growing** the bond from the product state by
  SVD truncation (`trunc_eps`, optionally capped by `bond_dim`); `result.max_bond`
  reports the peak bond.

## Example

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=20, phys_dim=12)
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)

r = model.run(dt=0.02, t_max=2.0, method="schrodinger-star-tdvp1", bond_dim=80,
              observables={"sz": sigma_z})
r.expect["sz"]

r2 = model.run(dt=0.02, t_max=2.0, method="schrodinger-star-tdvp2", bond_dim=120,
               trunc_eps=1e-4, observables={"sz": sigma_z})
r2.max_bond
```

## Notes

- Prefer {doc}`/methods/schrodinger/chain` for production runs in this representation — the
  chain's locality usually beats the star's lack of mode–mode terms once nothing has
  been rotated out.  Reach for the static star when you want a reference answer or
  an independent check.
- Because the MPO is static, TDVP conserves energy here (up to truncation), which
  the interaction-picture star cannot offer.
- `h` and the coupling `O` are carried as matrices, so a general Hermitian system
  and coupling of any dimension work; see {doc}`/models/spin_boson`.
- `polaron-star` is a separate static representation implemented through the
  generic TDVP MPO; see {doc}`polaron_chain`.
