# Interaction-chain representation — conditional-displacement MPO

`method="interaction-chain-trotter-mpo"` writes the full system–bath propagator as one exact,
low-bond conditional-displacement MPO — no Trotter splitting between modes, no
swap network.  This only works in the interaction picture, where all coupling
terms commute. It represents the same interaction-picture Hamiltonian as
{doc}`/methods/interaction/tebd`.

`interaction-chain-trotter-mpo` propagates the **same interaction representation as {doc}`/methods/interaction/tebd`**, but
instead of Trotterizing the system–bath coupling into two-site gates and shuttling
the system along the chain with a swap network, it writes the *entire* system–bath
propagator as a single matrix-product operator whose bond dimension is the number
of distinct eigenvalues of the coupling operator — **2** for a $\sigma_z$
spin–boson, independent of chain length.

The key observation is that the multimode propagator **factorizes exactly**: there
is no inter-mode Trotter error to make in the first place.

## Theory

### The interaction-picture coupling

First discretize the bath as a finite star and take the interaction
representation with respect to its free Hamiltonian. Then transform the
time-dependent star coupling into chain modes. The result is a single product of
a system operator and a bath operator,

$$
H_{sb}(t) = A_s \otimes B(t),
\qquad
B(t) = \sum_n \big[d_n(t)\, b_n + d_n^{*}(t)\, b_n^{\dagger}\big],
$$

where $A_s$ is the (Hermitian) system coupling operator and the $d_n(t)$ are the
time-integrated couplings computed by
{py:meth}`~fishbonett.representations.interaction.InteractionRepresentation.interval_coefficients`
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
{py:meth}`~fishbonett.representations.interaction.InteractionRepresentation.trotter_mpo`.

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

## Computational properties

Compared with the swap network of {doc}`/methods/interaction/tebd`, per step:

- **no swap gates** — the system never travels along the chain;
- **no $d\times d$ bosonic two-site gates** — each mode sees only a $d\times d$
  *single-site* displacement;
- **no inter-mode Trotter error** — that factorization is exact (both methods
  share this, but here it is manifest);
- the entangling operation is applied **all at once** rather than mode by mode.

Both algorithms are second order. Their relative cost depends on the local Fock
dimension, bath length, state bond dimensions, and contraction backend, so compare
them on a short convergence run for the model of interest.

```{note}
The bond dimension of the *state* is unchanged — this is the same representation and the
same physics as `interaction-chain-tebd`, so it carries the same entanglement.
`interaction-chain-trotter-mpo` changes
the cost of applying the propagator, not the cost of representing the state.  For a
method that lowers the state entanglement itself, see {doc}`/methods/schrodinger/polaron_chain`.
```

### Exactness on a truncated ladder

The factorization above is an identity for a *harmonic* mode. On `phys_dim` Fock
levels it is not quite: $[b, b^{\dagger}] = I - d\,|d{-}1\rangle\langle d{-}1|$, so
composing displacements over an interval is not the truncated model's exact
propagator. The residual has two parts:

- a deviation confined to the **top Fock level**, controlled by raising `phys_dim`
  like every other truncation error here;
- a **phase weighted by $a^2$** — the second Magnus term, $[H(s), H(s')]$ being a
  c-number times $A_s^2$.

The phase is $\mathcal{O}(\Delta t^3)$ per step, so it accumulates as
$\mathcal{O}(\Delta t^2)$: the same order as the Strang splitting around it, which
is why it does not degrade the method's order. Its *observability* depends on the
coupling. If $A_s$'s eigenvalues share a magnitude — $\sigma_z$, with $a = \pm 1$
and $a^2 = 1$ — the phase is common to both branches and cancels as a global phase.
For a coupling whose eigenvalues do not, such as the occupation projector the comb
models use ($a \in \{0, 1\}$), it is a *relative* phase between branches and is
physical.

## On a comb: one operator per bath branch

`method="interaction-chain-fishbone-trotter-mpo"` is the same construction applied
to the {doc}`/models/fishbone` comb, where every electronic site carries its own
bath chain. It is the operator counterpart of
`interaction-chain-fishbone-tebd`.

Baths attached to different system sites commute, so their propagators factorize
over sites — the same argument as over modes, one level up:

$$
U_{\text{bath}}(t, \Delta t) = \prod_{p} \Big[ \sum_a P_a^{(p)} \otimes
   \bigotimes_n D^{(p)}_n\big(-i\,a\,d^{(p)*}_n(t)\big) \Big].
$$

Several baths may share one system site. Their coupling operators need not commute;
the implementation therefore uses a palindromic second-order composition within
that site's branch group. The electronic Hamiltonian — site energies and the
backbone couplings — is Strang-split around the complete bath step, exactly as
$h_{\mathrm{sys}}$ is on a chain.

Because the operator's outer bonds are trivial, applying a branch's MPO grows only
that branch's own bonds, by $\chi_U$ once, and leaves the backbone untouched. One
sweep truncates them back. The swap network instead walks the electronic index down
the branch and back, truncating at every bond twice.

```{warning}
Changing the branch integrator does not remove physical state entanglement.
Converge `trunc_eps`, monitor `result.max_bond`, and treat runtime differences as
model- and hardware-dependent.
```

## General systems

Nothing above assumed a two-level system:

- **A general coupling operator** `O` — any $(d,d)$ Hermitian `coupling`; the MPO
  bond is the number of its distinct eigenvalues (3 for a three-level $O$, and so
  on), still independent of the chain length.
- **A non-two-level system** — `h` may be any $(d,d)$ Hamiltonian.
- **An arbitrary initial state** via `initial=`.
- **Finite temperature** works exactly as for `interaction-chain-tebd`
  (thermofield / signed
  `domain`), since the representation is identical.

## Example

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)
model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)

r = model.run(dt=0.02, t_max=2.0, method="interaction-chain-trotter-mpo", trunc_eps=1e-4,
              observables={"sz": sigma_z})
r.expect["sz"]     # <sigma_z>(t) -- identical physics to interaction-chain-tebd
r.max_bond         # peak bond dimension of the state
```

## Notes

- The propagator MPO is rebuilt each step because $d_n(t)$ is time-dependent, but
  building it costs no matrix exponentials. Writing $\alpha = r e^{i\phi}$ and
  using $e^{i\phi n} b^\dagger e^{-i\phi n} = e^{i\phi} b^\dagger$,

  $$
  \alpha b^\dagger - \alpha^{*} b = r\, e^{i\phi n}\,(b^\dagger - b)\,e^{-i\phi n},
  $$

  so every displacement is a phase rotation of one *fixed* matrix and a single
  cached eigendecomposition per `phys_dim` serves the whole run
  ({py:func}`fishbonett.operators.displacement`).
- Accepts the same truncation controls as every other method: `trunc_eps` sets the
  accuracy and `bond_dim` is an optional cap (default `None` = unlimited).
- For the builder see
  {py:meth}`~fishbonett.representations.interaction.InteractionRepresentation.trotter_mpo`;
  for the application/compression algorithm see
  {py:mod}`fishbonett.evolve.mpo_apply`.
