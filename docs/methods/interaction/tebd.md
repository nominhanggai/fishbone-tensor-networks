# Interaction picture · chain — swap-network TEBD

`method="tebd"` runs swap-network TEBD on an MPS in the interaction picture.  The
free-bath evolution is rotated out so entanglement is small, but the gates are
time-dependent and must be rebuilt every step.  For the same physics without the
swap network, see {doc}`/methods/interaction/trotter_mpo` (~1.6× faster).

## Theory

Start from the chain-mapped spin-boson Hamiltonian, in which the continuous bath
has been replaced by a one-dimensional chain of effective modes (see
{doc}`/bath`):

$$
H = H_{\mathrm{sys}} + \underbrace{c_0\, O\,(b_0 + b_0^\dagger)}_{\text{system–bath}}
    + \sum_i \epsilon_i\, b_i^\dagger b_i
    + \sum_i t_i\,(b_i^\dagger b_{i+1} + \text{h.c.}),
$$

where $O$ is the system coupling operator (any Hermitian operator, not just
$\sigma_z$).

### The interaction picture

The engine works in the **interaction picture with respect to the free bath**.
Diagonalizing the chain gives star modes $\omega_k$; moving into their rotating
frame, $b \to b\,e^{-i\omega t}$, removes the bath term entirely and leaves only
the system–bath coupling, now carrying the accumulated bath phase:

$$
H_{sb}(t) = O \otimes \sum_n \big[d_n(t)\, b_n + d_n^{*}(t)\, b_n^{\dagger}\big],
\qquad
d_n(t) = \int_t^{t+\Delta t}\!\!dt'\; \sum_k j_k\,U_{kn}\,e^{-i\omega_k t'} .
$$

Two things are bought by this. The free-bath evolution is now exact — no Trotter
error for the largest energy scale in the problem — and the *only* remaining
entanglement generator is the single operator $O$. The price is that the gates are
time-dependent and must be rebuilt every step. Because $d_n$ already contains the
time integral over the step, it carries the factor of $\Delta t$.

### The swap network

In this frame **every** mode couples directly to the system, so the interaction is
not nearest-neighbour on the chain. TEBD needs adjacency, so the system site is
walked along the chain: a nearest-neighbour gate is applied with `swap=1`, which
both applies the two-site gate and transposes the physical legs, moving the system
one site over. The system starts at site 0, so it is walked **outward** to the far
end and then back **in** to site 0, visiting every mode once in each direction and
leaving the state in the layout the next step expects.

The sweep is arranged **palindromically in time** — the first half-interval's
gates on the way in, the second half-interval's on the way out, with the two
innermost updates straddling the midpoint — which makes the step **second order**
in $\Delta t$ (a Strang splitting). The bond dimension is controlled by SVD
truncation: `trunc_eps` sets the accuracy and `bond_dim` is an optional cap.

```{note}
Splitting the coupling over modes is **exact**, not an approximation: the terms
$O\otimes X_n$ all commute with one another. The only splitting error in a step is
between $h_{\mathrm{sys}}$ and the coupling.  {doc}`/methods/interaction/trotter_mpo` exploits the same
commutation to apply the whole propagator as one bond-2 MPO, avoiding the swaps
altogether — same frame, same physics, cheaper per step.
```

Keeping the system on its own MPS site (rather than absorbing bath modes into it)
is what makes the ansatz efficient, and it is why composite systems should be
built as trees rather than fattened onto one site — see
{doc}`/models/composite_multichannel`.

## General systems

Every `SystemBath` engine accepts a general system — a Hermitian `h` of any
dimension `d`, a Hermitian coupling operator `O` of the same dimension, and an
arbitrary initial state:

- **A general coupling operator** `O` — pass any `(d, d)` Hermitian `coupling`.
- **A non-two-level system** — `h` may be any `(d, d)` Hamiltonian.
- **An arbitrary initial state** via `initial=`: `"up"` (default), `"down"`,
  `"ground"` (the ground state of `h`), or an explicit length-`d` state vector.

`tebd` reaches these on a single MPS via leg swaps; the MPO and tree engines carry
`h` and `O` as matrices in their finite-state-machine operators (their
interaction-picture gates diagonalize `O`).  Only a *multichannel* bath — one bath
coupled through several operators at once — is handled elsewhere (it routes through
the tree so the system stays on its own site; see
{doc}`/models/composite_multichannel`).

## Example

A biased two-level system with a **transverse** ($\sigma_x$) coupling, started
from the ground state of its Hamiltonian:

```python
import numpy as np
from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)

model = SystemBath(h=0.5 * sigma_z + sigma_x,   # biased two-level system
                  coupling=sigma_x,            # transverse coupling (not sigma_z)
                  bath=bath)

r = model.run(dt=0.02, t_max=2.0, method="tebd", bond_dim=100,
              initial="ground",                # start in the ground state of h
              observables={"sz": sigma_z, "sx": sigma_x})
r.expect["sz"]        # <sigma_z>(t)
r.rdm                 # (n_steps, 2, 2) reduced density matrix of the system
```

A three-level system driven by a `sigma_z`-like diagonal coupling works the same
way — just pass a `(3, 3)` `h` and `coupling` and a length-3 `initial` vector.

## Notes

- TEBD is a general propagation algorithm — it is not specific to the interaction
  picture.  (The polaron frame also uses TEBD gates.)  In this package `method="tebd"`
  runs TEBD *in the interaction picture*, where the free-bath evolution is rotated
  out and the gates are time-dependent.  A signed `domain` for finite temperature
  comes from T-TEDOPA thermalization of the spectral density; see {doc}`/bath`.
- Cost per step is $O(N)$ two-site updates in each direction ($2N$ SVDs); the bond
  dimension is set by the physical system–bath entanglement, controlled by
  `trunc_eps` and optionally capped by `bond_dim`.
- The step is second order in `dt`: halving `dt` cuts the error ~4×.
- For the underlying builder see {py:mod}`fishbonett.frames.interaction_picture`, and for the
  canonical MPS/TEBD state see {py:class}`fishbonett.states.mps.SystemBathMPS`.
