# `tebd` — interaction-picture MPS TEBD

The `tebd` method is an interaction-picture, swap-network time-evolving block
decimation on a matrix-product state.  Like every engine in the package it accepts
a general (non-`sigma_z`) system–bath coupling, a system of arbitrary dimension,
and an arbitrary initial system state; `tebd` reaches these through leg swaps on a
single MPS.

## Theory

Start from the chain-mapped spin-boson Hamiltonian, in which the continuous bath
has been replaced by a one-dimensional chain of effective modes (see
{doc}`../bath`):

$$
H = H_{\mathrm{sys}} + \underbrace{c_0\, O\,(b_0 + b_0^\dagger)}_{\text{system–bath}}
    + \sum_i \epsilon_i\, b_i^\dagger b_i
    + \sum_i t_i\,(b_i^\dagger b_{i+1} + \text{h.c.}),
$$

where $O$ is the system coupling operator (any Hermitian operator, not just
$\sigma_z$).  The engine works in the **interaction picture with respect to the
system–bath coupling**: the free chain evolution is folded into time-dependent
gates, leaving the system correlated with the bath only through $O$.  Because the
entanglement is generated locally at the system–bath bond, a distant chain mode
can be brought next to the system site with a sequence of nearest-neighbour
**leg swaps**, the two-site Trotter gate applied, and the mode swapped back — the
"swap network".  Each sweep is a first-order Trotter step; the bond dimension is
controlled by SVD truncation to `bond_dim` / `trunc_eps`.

Keeping the system on its own MPS site (rather than absorbing bath modes into it)
is what makes the ansatz efficient, and it is why composite systems should be
built as trees rather than fattened onto one site — see
{doc}`../systems/composite_multichannel`.

## General systems

Every `SpinBoson` engine accepts a general system — a Hermitian `h` of any
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
{doc}`../systems/composite_multichannel`).

## Example

A biased two-level system with a **transverse** ($\sigma_x$) coupling, started
from the ground state of its Hamiltonian:

```python
import numpy as np
from fishbonett.simulate import Bath, SpinBoson
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5), domain=(-25, 36),
            temperature=1.0, n_modes=40, phys_dim=20)

model = SpinBoson(h=0.5 * sigma_z + sigma_x,   # biased two-level system
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

- `tebd` is interaction-picture, so it needs no separate bath-frequency gauge and
  handles a signed `domain` (thermofield / T-TEDOPA) directly; see {doc}`../bath`.
- Cost per step is `O(n_modes)` two-site updates; the bond dimension is set by the
  physical system–bath entanglement and capped at `bond_dim`.
- For the underlying builder see {py:mod}`fishbonett.models.interaction_picture`, and for the
  canonical MPS/TEBD state see {py:class}`fishbonett.states.mps.SpinBosonMPS`.
