# `trotter-mpo` — the exact conditional-displacement propagator

`trotter-mpo` propagates the **same interaction-picture model as {doc}`tebd`**, but
instead of Trotterizing the system–bath coupling into two-site gates and shuttling
the system along the chain with a swap network, it writes the *entire* system–bath
propagator as a single matrix-product operator whose bond dimension is the number
of distinct eigenvalues of the coupling operator — **2** for a $\sigma_z$
spin–boson, independent of chain length.

The key observation is that the multimode propagator **factorizes exactly**: there
is no inter-mode Trotter error to make in the first place.

## Theory

### The interaction-picture coupling

After the chain mapping and the star diagonalization ({doc}`../bath`), the
interaction picture with respect to the *free bath* leaves a coupling that is a
single product of a system operator and a bath operator,

$$
H_{sb}(t) = A_s \otimes B(t),
\qquad
B(t) = \sum_n \big[d_n(t)\, b_n + d_n^{*}(t)\, b_n^{\dagger}\big],
$$

where $A_s$ is the (Hermitian) system coupling operator and the $d_n(t)$ are the
time-integrated couplings computed by
{py:meth}`~fishbonett.frames.interaction_picture.BosonicBathIP.mode_couplings`
(they already contain $\int_t^{t+\Delta t}$, so no extra factor of $\Delta t$
appears below).

### Why the modes factorize exactly

Write $H_{sb} = \sum_n A_s \otimes X_n$ with
$X_n = d_n b_n + d_n^{*} b_n^{\dagger}$. Different modes commute,
$[X_n, X_m] = 0$ for $n \neq m$, and $A_s$ is a common factor, so

$$
[A_s\otimes X_n,\; A_s\otimes X_m] = A_s^2 \otimes [X_n, X_m] = 0 .
$$

**Every term commutes with every other term.** Splitting the exponential over
modes is therefore not an approximation — it is an identity:

$$
\exp\!\big(-i\,A_s\otimes B\big) \;=\; \prod_n \exp\!\big(-i\,A_s\otimes X_n\big).
$$

This is why the swap-network TEBD sweep is also exact in the mode sector: its
two-site gates are these same commuting factors. The difference is purely *how*
they are applied.

### From commuting factors to conditional displacements

Diagonalize the system coupling, $A_s = \sum_a a\,P_a$ with $P_a$ the projector
onto its $a$-eigenspace. Since each factor depends on the system only through
$A_s$,

$$
\exp\!\big(-i\,A_s\otimes X_n\big) = \sum_a P_a \otimes \exp\!\big(-i\,a\,X_n\big).
$$

The remaining exponential is linear in $b_n, b_n^\dagger$, i.e. exactly a
**displacement operator** $D_n(\alpha) = \exp(\alpha b_n^{\dagger} - \alpha^{*} b_n)$:
matching coefficients in
$-i a (d_n b_n + d_n^{*} b_n^{\dagger}) = \alpha b_n^\dagger - \alpha^* b_n$ gives

$$
\boxed{\;\alpha_{a,n} = -i\,a\,d_n^{*}(t)\;}
$$

Collecting the modes (they commute, so the product is a plain tensor product):

$$
U_{sb}(t,\Delta t) \;=\; \sum_a P_a \otimes \bigotimes_n D_n\!\big(-i\,a\,d_n^{*}(t)\big).
$$

**The propagator is a sum of $r$ product operators**, where $r$ is the number of
*distinct* eigenvalues of $A_s$. For the spin–boson model $A_s = \sigma_z$ has
$a = \pm 1$, so it is a sum of just two:

$$
U_{sb} = |{\uparrow}\rangle\langle{\uparrow}| \otimes \bigotimes_n D_n(-i\,d_n^{*})
       + |{\downarrow}\rangle\langle{\downarrow}| \otimes \bigotimes_n D_n(+i\,d_n^{*}).
$$

Physically it is a *conditional displacement*: the bath is kicked one way or the
other depending on the system state — the entangling operation, written in closed
form.

### The MPO and its bond dimension

A sum of $r$ product operators is an MPO of bond dimension $r$: the bond carries
only the label $a$ of which branch is active. Sites are ordered
`[system, mode_0, ..., mode_{N-1}]` and the tensors are

$$
W^{\text{sys}}_{1,a} = P_a, \qquad
W^{(n)}_{a,a'} = \delta_{a a'}\, D_n(\alpha_{a,n}),
$$

with the last mode contracting the bond down to 1. So

$$
\chi_U = \#\{\text{distinct eigenvalues of } A_s\} \;=\; 2 \ \ (\sigma_z),
$$

**independent of the number of bath modes**. Built by
{py:meth}`~fishbonett.frames.interaction_picture.BosonicBathIP.displacement_mpo`.

### The full step

$U_{sb}$ handles the coupling; the system Hamiltonian $h_{\mathrm{sys}}$ does not
commute with it and is Strang-split around it, so a step is second order in
$\Delta t$:

$$
U(\Delta t) = e^{-i h_{\mathrm{sys}}\Delta t/2}\;
              U_{sb}(t,\Delta t)\;
              e^{-i h_{\mathrm{sys}}\Delta t/2} + \mathcal{O}(\Delta t^3).
$$

Applying the MPO multiplies the MPS bond by $\chi_U$; a QR + truncated-SVD sweep
({py:func}`fishbonett.evolve.mpo_apply.compress`) brings it back down to whatever
`trunc_eps` requires.

## What this buys

Compared with the swap network of {doc}`tebd`, per step:

- **no swap gates** — the system never travels along the chain;
- **no $d\times d$ bosonic two-site gates** — each mode sees only a $d\times d$
  *single-site* displacement;
- **no inter-mode Trotter error** — that factorization is exact (both methods
  share this, but here it is manifest);
- the entangling operation is applied **all at once** rather than mode by mode.

Measured against the swap network at equal bond dimension it is roughly **1.6×
faster** per step. Accuracy is comparable: both are second order, and the two agree
with exact diagonalization at the same rate (the swap network has a slightly
smaller prefactor).

```{note}
The bond dimension of the *state* is unchanged — this is the same frame and the
same physics as `tebd`, so it carries the same entanglement. `trotter-mpo` changes
the cost of applying the propagator, not the cost of representing the state.  For a
method that lowers the state entanglement itself, see {doc}`polaron`.
```

## General systems

Nothing above assumed a two-level system:

- **A general coupling operator** `O` — any $(d,d)$ Hermitian `coupling`; the MPO
  bond is the number of its distinct eigenvalues (3 for a three-level $O$, and so
  on), still independent of the chain length.
- **A non-two-level system** — `h` may be any $(d,d)$ Hamiltonian.
- **An arbitrary initial state** via `initial=`.
- **Finite temperature** works exactly as for `tebd` (thermofield / signed
  `domain`), since the frame is identical.

## Example

```python
import numpy as np
from fishbonett.simulate import Bath, BosonicBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = BosonicBath(h=sigma_x, coupling=sigma_z, bath=bath)

r = model.run(dt=0.02, t_max=2.0, method="trotter-mpo", trunc_eps=1e-4,
              observables={"sz": sigma_z})
r.expect["sz"]     # <sigma_z>(t) -- identical physics to method="tebd"
r.max_bond         # peak bond dimension of the state
```

## Notes

- The propagator MPO is rebuilt each step because $d_n(t)$ is time-dependent;
  building it is $O(N)$ single-mode `expm`s of size `phys_dim`, negligible next to
  the state update.
- Accepts the same truncation controls as every other method: `trunc_eps` sets the
  accuracy and `bond_dim` is an optional cap (default `None` = unlimited).
- For the builder see
  {py:meth}`~fishbonett.frames.interaction_picture.BosonicBathIP.displacement_mpo`;
  for the application/compression algorithm see
  {py:mod}`fishbonett.evolve.mpo_apply`.
