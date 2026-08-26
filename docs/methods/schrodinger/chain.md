# Schrödinger-chain representation — MPO + TDVP

`schrodinger-chain-tdvp1` (fixed bond), `schrodinger-chain-tdvp2` (grows by SVD), `schrodinger-chain-a1tdvp`
(bond-adaptive).  Nothing is rotated out, so $H$ is static and the MPO is built
once — TDVP conserves energy.  The cost: the state carries the full system–bath
correlation, so bond dimensions are larger than in the other representations.  For lower
entanglement at the same cost, see the
{doc}`polaron chain </methods/polaron>`.

These three methods integrate the **Schrödinger equation** for the full TEDOPA
chain, with the Hamiltonian represented exactly as a matrix-product operator and
the state evolved by the time-dependent variational principle (TDVP).  They are
the accuracy workhorses for a two-level, `sigma_z`-coupled system.

## Theory

The bath is chain-mapped (see {doc}`/bath`) and the spin-boson chain

$$
H = \tfrac{\epsilon}{2}\sigma_z + V\sigma_x
    + c_0\,\sigma_z (b_0 + b_0^\dagger)
    + \sum_i \epsilon_i\, b_i^\dagger b_i
    + \sum_i t_i\,(b_i^\dagger b_{i+1} + \text{h.c.})
$$

is compiled from a sum of local operator products: the system sits on site 0 and
the bath chain on sites $1\ldots N$. The product label first becomes the MPO bond;
an exact QR/SVD minimization then removes linearly dependent operator prefixes and
suffixes, recovering a small auxiliary bond without a hand-written state machine.
The system block carries `h` and the coupling operator $O$ as matrices, so `h` may
be any Hermitian operator of any dimension and the coupling any Hermitian $O$
(shown here for the two-level $\sigma_z$ case).

### The variational principle

An MPS of fixed bond dimension $D$ is not a linear subspace — it is a curved
*manifold* $\mathcal{M}_D$ inside Hilbert space. The Dirac–Frenkel variational
principle asks for the trajectory on that manifold that best matches the true
Schrödinger flow: minimize $\lVert |\dot\psi\rangle + i H|\psi\rangle \rVert$ over
all tangent vectors, which gives

$$
|\dot\psi\rangle = -i\,\hat{P}_{T_\psi\mathcal{M}_D}\,H\,|\psi\rangle,
$$

with $\hat{P}_{T_\psi\mathcal{M}}$ the orthogonal projector onto the tangent space
at the current state. Two properties follow directly and are the reason TDVP is
attractive here:

- the norm and (for time-independent $H$) the **energy are conserved exactly**,
  because the projector is orthogonal and Hermitian;
- there is **no Trotter angle** — $H$ is treated as one object, so nothing is
  split and no commutator error is introduced. The only error is the *projection*
  error, i.e. the part of $H|\psi\rangle$ that points off the manifold.

The tangent-space projector decomposes into a sum of local terms with alternating
signs — one per site minus one per bond — and integrating each in turn is the
**projector-splitting** integrator: sweep left to right evolving each site tensor
forward by $\Delta t/2$ with the local effective Hamiltonian, evolving the bond
tensor *backward* between sites, then sweep back. The backward bond evolution is
not a trick; it is the negative term in the projector. A symmetric forward/backward
sweep makes the step second order in $\Delta t$.

Each local update is an exponential $e^{-i H_{\text{eff}} \tau}$ of a small
effective Hamiltonian built by contracting the MPO with the environment blocks;
it is applied by a Lanczos/Krylov expansion whose dimension is set by `krylov`.

### The three variants

They differ only in how the bond dimension is handled:

- **`schrodinger-chain-tdvp1`** — 1-site TDVP at a **fixed** bond dimension.  The state never
  leaves $\mathcal{M}_D$, so norm and energy are conserved to machine precision and
  at a well-chosen `bond_dim` this is the most accurate method per unit cost.  The
  catch is structural: a 1-site update **cannot change the bond dimension**, so
  $D$ must be large enough from the very first step.  Starting from a product
  state (all bonds 1) the state would be frozen, so the engine pads the bonds up
  to `bond_dim` with small noise first — which is why `bond_dim` is *required*
  here and `bond_dim=None` (unlimited) is rejected.
- **`schrodinger-chain-tdvp2`** — 2-site TDVP.  Each update acts on two neighbouring sites at
  once and re-splits them with an SVD, so the bond **grows on its own** to
  whatever `trunc_eps` demands (capped by `bond_dim` if you set one).  Growth
  needs the split to keep a little more than the threshold admits — see
  `bond_expand` in {doc}`../../getting_started` — because the entanglement a
  single step creates is itself of order `dt`.  This is the
  method to use when you do not know the required bond dimension in advance; the
  price is an $O(d^2)$ larger local problem and a truncation error that 1-site
  TDVP does not have.  `result.max_bond` reports the peak bond per step.
- **`schrodinger-chain-a1tdvp`** — adaptive one-site TDVP. Before each time
  step, full QR factorizations add candidate basis vectors orthogonal to the
  current left- and right-canonical spaces. For bond $i$, the method evaluates

  $$
  f(D_i)=\lVert H(i)A_C(i)\rVert^2
       +\lVert K(i)C(i)\rVert^2
       +\lVert H(i+1)A_C(i+1)\rVert^2
  $$

  in the enlarged spaces and selects the smallest dimension for which adding
  one more direction changes $f$ by at most `trunc_eps`. The subsequent time
  step contains only one-site and zero-site exponential actions; no two-site
  centre is evolved or split. `bond_expand` limits the candidate directions
  considered in one sweep, and `bond_dim` is the required memory ceiling. This
  is an independent implementation of the adaptive one-site construction of
  [Dunnett and Chin, *Phys. Rev. B* **104**, 214302
  (2021)](https://doi.org/10.1103/PhysRevB.104.214302).

The implementation follows the tangent-space projector splitting described by
Haegeman *et al.*, *Phys. Rev. B* **94**, 165116 (2016). Local exponential
actions use a fully reorthogonalized Arnoldi projection.

## Examples

Fixed bond, most accurate at a given `bond_dim`:

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)

r = model.run(dt=0.02, t_max=2.0, method="schrodinger-chain-tdvp1", bond_dim=60,
              observables={"sz": sigma_z})
r.expect["sz"]
```

Growing / adaptive bond, when the required `bond_dim` is unknown:

```python
r2 = model.run(dt=0.02, t_max=2.0, method="schrodinger-chain-tdvp2", trunc_eps=1e-4,
               observables={"sz": sigma_z})
r2.max_bond            # peak bond dimension reached at each step

r3 = model.run(dt=0.02, t_max=2.0, method="schrodinger-chain-a1tdvp", bond_dim=200,
               trunc_eps=1e-5,
               observables={"sz": sigma_z})
r3.max_bond
```

`schrodinger-chain-tdvp2` needs no `bond_dim` — `trunc_eps` decides how far the bond grows.
`schrodinger-chain-a1tdvp` is bond-adaptive but still needs a ceiling, so
`bond_dim` is required. Here `trunc_eps` controls the relative tangent-space
convergence test, not an SVD truncation.

## Low-level driver

The high-level interface constructs the representation and hands it to
{py:func}`fishbonett.evolve.tdvp.run_mpo_hamiltonian`. Both halves can be called
directly for finer control: `tdvp_mpo()` supplies `H`, and the sweep says how a
step is taken.

```python
from fishbonett.evolve.tdvp import run_mpo_hamiltonian
from fishbonett.representations.schrodinger import SchrodingerRepresentation

rep = SchrodingerRepresentation(
    representation="schrodinger-chain",
    h_sys=sigma_x,
    coupling=sigma_z,
    bath=bath,
)
t, sz, maxd = run_mpo_hamiltonian(
    rep, dt=0.05, nsteps=80, sweep="tdvp1", bond_dim=100)
```

`sweep` is `"tdvp1"`, `"tdvp2"` or `"a1tdvp"`; `dt` is the time advanced per step.

## Notes

- `krylov` sets the Lanczos dimension of the local exponential propagator.
- TDVP conserves the norm and (for `schrodinger-chain-tdvp1`) energy well; a too-small fixed
  `bond_dim` shows up as a projection error, not as a blow-up.
- The chain coefficients come from the package's own discretization, so the
  measure-adapted TEDOPA star (`discretization="tedopa"`) is available here too.
