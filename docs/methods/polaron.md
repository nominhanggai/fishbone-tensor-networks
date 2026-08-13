# `polaron` — polaron-frame MPS TEBD

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
  $\langle\sigma_z\rangle$) are frame-invariant. Coherences are dressed and are
  **un-dressed** back to the lab frame from the $(c_0,\text{system})$ two-site RDM;
  `run` returns lab-frame `expect` and `rdm`.

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
- For the builder see {py:mod}`fishbonett.frames.polaron`, and for the canonical
  MPS/TEBD state see {py:class}`fishbonett.states.mps.BosonicBathMPS`.
