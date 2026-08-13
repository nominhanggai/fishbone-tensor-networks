# `polaron` / `polaron-tdvp*` — the polaron frame

```{admonition} Frame: polaron (static $\tilde H$) — the only frame that gets both
:class: tip
The polaron transform is the best of both worlds structurally: like the
{doc}`Schrödinger picture <chain_mpo>` the result is **time-independent**, so the
gates can be built **once** and a static MPO drives the full TDVP family; and like
the {doc}`interaction picture <tebd>` it removes a large part of the system–bath
correlation, so the state carries little entanglement.  What it costs is
generality — the transform needs $T=0$ and $\int J(\omega)/\omega^2\,d\omega$
finite.  It does *not* admit the exact conditional-displacement propagator of
{doc}`trotter_mpo`, because the dressed tunneling does not commute with the
free-chain hopping.  See {doc}`index` for the compatibility table.
```

The `polaron` method propagates the model in the **polaron (Lang–Firsov) frame**:
the static system–bath coupling is absorbed into a displacement of the bath, so the
system correlates with the bath only through the *dressed tunneling* on a single
chain bond. Because that static reorganization no longer lives in the MPS bonds, the
polaron chain typically carries **less entanglement** than the interaction-picture
(`tebd`) chain in the strong-coupling regime.

## Theory

For a system coupled to a harmonic bath through a Hermitian operator $O$,

$$
H = H_{\mathrm{sys}} + O\otimes\sum_k g_k (a_k + a_k^\dagger) + \sum_k \omega_k a_k^\dagger a_k,
$$

apply the polaron transform $U_p = \exp\!\big(O\otimes\Lambda\big)$ with
$\Lambda = \sum_k (g_k/\omega_k)(a_k^\dagger - a_k)$. This removes the static
coupling and leaves a free bath plus a dressed system term. Diagonalising
$O=\sum_i \lambda_i\,|i\rangle\langle i|$,

$$
\tilde H = \sum_{ij} \langle i|H_{\mathrm{sys}}|j\rangle\, |i\rangle\langle j|
           \otimes D\big((\lambda_i-\lambda_j)\big)
         + \sum_k \omega_k a_k^\dagger a_k - E_{\mathrm{reorg}}\,O^2,
$$

i.e. each off-diagonal (in $O$'s eigenbasis) block of $H_{\mathrm{sys}}$ is **dressed**
by a bath displacement $D$ of magnitude $(\lambda_i-\lambda_j)$; diagonal blocks are
undressed. The collective mode $\Lambda$ is a **single** bath mode, so seeding the
TEDOPA chain from the **reweighted spectral density** $J(\omega)/\omega^2$ makes the
first chain mode $c_0$ *be* that mode: the dressed term is then a two-site gate on the
$(c_0,\text{system})$ bond and the rest of the chain is free (nearest-neighbour
hopping) — a plain Trotter sweep with **no swap network**. The displacement scale is
$\kappa_0=k_0$, the first chain coupling of the $J/\omega^2$ mapping.

Two consequences to keep in mind:

- **Initial state.** The physical (Franck–Condon) bath vacuum maps to a **displaced**
  coherent state on $c_0$: $U_p|\psi\rangle\otimes|\mathrm{vac}\rangle
  = \sum_i c_i\,|i\rangle\otimes|\text{coherent}(\lambda_i\kappa_0)\rangle$. Skipping
  this displacement solves a *different* physical problem.
- **Observables.** Diagonal observables (populations in $O$'s eigenbasis, e.g.
  $\langle\sigma_z\rangle$) are frame-invariant; coherences are dressed and must be
  un-dressed (next section). `run` returns lab-frame `expect` and `rdm`.

### Recovering lab-frame observables

Because $U_p$ is diagonal in $O$'s eigenbasis,
$U_p=\sum_i |i\rangle\langle i|\otimes D(\lambda_i)$ with
$D(\lambda)=e^{\lambda\kappa_0 (c_0^\dagger-c_0)}$, the lab-frame system reduced
density matrix follows from the polaron-frame state $\tilde\rho$ as

$$
\rho^{\mathrm{lab}}_{ij}
 = \big\langle i\big|\operatorname{Tr}_B\!\big[U_p^\dagger\,\tilde\rho\,U_p\big]\big|j\big\rangle
 = \operatorname{Tr}_B\!\big[D(-\lambda_i)\,\tilde\rho_{ij}\,D(\lambda_j)\big]
 = \operatorname{Tr}_B\!\big[\tilde\rho_{ij}\,D(\lambda_j-\lambda_i)\big],
$$

where $\tilde\rho_{ij}=\langle i|\tilde\rho|j\rangle$ is a bath operator. The last
step uses cyclicity of the trace together with
$D(\lambda_j)D(-\lambda_i)=D(\lambda_j-\lambda_i)$, which is *exact* here (both
displacements share the generator $c_0^\dagger-c_0$, so they compose with no
residual phase).

Two things make this cheap and exact:

- For $i=j$ the factor is $D(0)=\mathbb{1}$, so **populations are unchanged** —
  the frame-invariance quoted above. Off-diagonal elements pick up
  $D(\lambda_j-\lambda_i)$, whose expectation is the Franck–Condon factor. For a
  two-level $O=\sigma_z$ ($\lambda=\pm1$) this is the familiar
  $\langle\sigma_x\rangle_{\mathrm{lab}}=\langle\sigma_+B^2+\mathrm{h.c.}\rangle$
  with $B=D(2\kappa_0)$.
- $U_p$ displaces **only** the collective mode $\Lambda$, and the $J/\omega^2$
  chain mapping makes $c_0\propto\sum_k(g_k/\omega_k)b_k$ *be* that mode (it is the
  Lanczos seed), so every other chain mode is orthogonal to it and untouched.
  Tracing them out therefore commutes with the un-dressing, and only the
  **$(c_0,\text{system})$ two-site** block of the MPS is needed — the coherence is
  stored in the $c_0$–system entanglement and is read back out by the trace above.

This is implemented as `undress_rdm` in {py:mod}`fishbonett.frames.polaron`.

## Propagators: TEBD or TDVP

Unlike the interaction picture, the polaron $\tilde H$ is **time-independent**, so it
has a plain MPO and can drive the full TDVP family as well as Trotter TEBD:

| `method` | integrator | bond growth |
|---|---|---|
| `polaron` | static two-site Trotter gates (TEBD) | SVD truncation (`trunc_eps`) |
| `polaron-tdvp1` | 1-site TDVP on the polaron MPO | **fixed** — padded to `bond_dim` |
| `polaron-tdvp2` | 2-site TDVP | SVD truncation |
| `polaron-dtdvp` | bond-adaptive 1-site TDVP | precision threshold (`prec`) |

`polaron-dtdvp` is usually the best of the four: 1-site TDVP never forms a two-site
block, so it avoids the $O(d^4)$ boson–boson gates that make the TEBD sweep
expensive, and the adaptive bond search finds the smallest representation. Measured
on a moderate-coupling model (all four agree with the interaction-picture chain to
$\sim10^{-3}$):

| method | time | peak bond |
|---|---|---|
| `polaron` (TEBD) | 2.3 s | 6 |
| `polaron-tdvp1` | 23.1 s | 20 (fixed) |
| `polaron-tdvp2` | 5.3 s | 6 |
| **`polaron-dtdvp`** | **1.5 s** | **4** |

Note `polaron-tdvp1` conserves the bond dimension, so it cannot grow out of a
product state; it is padded to `bond_dim` up front, which is why it is the most
expensive here. Prefer `polaron-dtdvp` unless you specifically need a fixed bond.

## General systems

Like the other engines, the polaron builder accepts a general system:

- **A general coupling operator** `O` — any $(d,d)$ Hermitian `coupling`; the frame
  is built from its eigenvalues (a two-level $\sigma_z$ is the special case).
- **A non-two-level system** — `h` may be any $(d,d)$ Hamiltonian.
- **An arbitrary initial state** via `initial=` (`"up"`, `"down"`, `"ground"`, or a
  length-$d$ vector).

## Example

```python
import numpy as np
from fishbonett.simulate import Bath, BosonicBath
from fishbonett.operators import sigma_x, sigma_z

# T=0, gapped bath so int J/w^2 is finite (the polaron precondition)
bath = Bath(J=lambda w: 0.3 * w * np.exp(-w / 2.5), domain=(0.3, 12.0),
            n_modes=24, phys_dim=14)

model = BosonicBath(h=0.5 * sigma_x, coupling=sigma_z, bath=bath)

r = model.run(dt=0.02, t_max=8.0, method="polaron", bond_dim=80,
              observables={"sz": sigma_z, "sx": sigma_x})
r.expect["sz"]        # <sigma_z>(t), lab frame
r.expect["sx"]        # <sigma_x>(t), un-dressed to the lab frame
r.max_bond            # peak bond dimension per step (small in the polaron frame)
```

## Notes

- **Applicability.** Zero temperature only (pass no `temperature`), and
  $\kappa_0^2=\tfrac1\pi\!\int J(\omega)/\omega^2\,d\omega$ must be finite — a gapped
  or super-ohmic bath. Strict ohmic is the log-divergent orthogonality-catastrophe
  edge; a finite-temperature polaron is a planned extension.
- The polaron frame is most advantageous when the **static reorganization** dominates
  (strong coupling): it folds that correlation into the $c_0$ displacement, so the MPS
  bonds carry only the dressed-tunneling entanglement.
- Cost per step is `O(n_modes)` nearest-neighbour two-site updates (no swaps); its
  free-chain bonds act on two boson sites, so the per-gate cost grows with `phys_dim`
  faster than the interaction-picture (system-adjacent) gates.
- **Fock truncation.** The displacement $D$ used in the dressed gate, the initial
  coherent state and the un-dressing is exponentiated on the *truncated*
  `phys_dim`-dimensional ladder, so it is not exactly unitary near the top of the
  Fock space. Since the displacement scale is $\kappa_0$, choose
  `phys_dim` $\gg\kappa_0^2$ (the mean occupation of the displaced $c_0$); the
  returned RDM is trace-normalized, which hides — rather than fixes — a too-small
  `phys_dim`. Converge in `phys_dim` as you would in `bond_dim`.
- For the builder see {py:mod}`fishbonett.frames.polaron`, and for the canonical
  MPS/TEBD state see {py:class}`fishbonett.states.mps.BosonicBathMPS`.
