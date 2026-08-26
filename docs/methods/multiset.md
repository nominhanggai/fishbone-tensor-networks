# Multi-set MPS [coupled two-site TDVP]

The multi-set ansatz is useful when a finite-dimensional system labels
environmental wavepackets with substantially different structures. In a chosen
orthonormal system basis,

$$
|\Psi(t)\rangle=\sum_{a=1}^{d_S}|a\rangle|\psi_a(t)\rangle,
$$

each $|\psi_a\rangle$ is represented by its own bath MPS. Its squared norm is
the population of system state $a$. Off-diagonal system coherences are the
cross overlaps $\langle\psi_b|\psi_a\rangle$.

This state construction was introduced for strong-coupling Holstein dynamics by
[Kloss, Reichman, and Tempelaar](https://doi.org/10.1103/PhysRevLett.123.126601).
Their calculation associates one vibrational MPS with each electronic site. The
implementation here writes an independently derived coupled two-site TDVP sweep
for the same variational manifold.

## Coupled equations

Writing the represented Hamiltonian as a matrix of bath operators,

$$
H=\sum_{ab}|a\rangle\langle b|\otimes H_{ab}^{B},
$$

gives

$$
i\frac{d}{dt}|\psi_a\rangle
=\sum_b H_{ab}^{B}|\psi_b\rangle.
$$

Every $H_{ab}^{B}$ is an MPO obtained by slicing the system tensor from the
ordinary full system--bath MPO. During a two-site sweep, all component centre
tensors are evolved together under this block effective Hamiltonian. Each
component is then split and truncated independently. Consequently:

- the system basis label is never truncated;
- `trunc_eps` controls each environmental MPS;
- `result.max_bond` is the largest retained bond in any component;
- computational work grows at least with the number of system states, and the
  block couplings may make it quadratic in that number.

## Running it

`SystemBath` supports the multi-set MPS with all five single-channel
Hamiltonian representations:

```python
import numpy as np

from fishbonett import Bath, SystemBath
from fishbonett.operators import sigma_x, sigma_z

bath = Bath(
    J=lambda omega: 0.2 * omega**3 * np.exp(-omega / 5.0),
    domain=(0.4, 12.0),
    n_modes=24,
    phys_dim=8,
)
model = SystemBath(
    h=0.5 * sigma_x + 0.15 * sigma_z,
    coupling=sigma_z,
    bath=bath,
)

result = model.run(
    dt=0.02,
    t_max=2.0,
    representation="interaction-chain",
    state_geometry="multi-set-mps",
    integrator="tdvp2",
    trunc_eps=1e-4,
    bond_dim=None,
    observables={"sz": sigma_z},
)
```

The equivalent method name is
`interaction-chain-multi-set-tdvp2`. Replace only `representation` to compare
Schrödinger chain or star, interaction chain, and polaron chain or star on the
same resolved finite bath.

## Holstein comparison

`examples/multiset_holstein.py` constructs
the local-mode Hamiltonian studied in the original paper and propagates the
same MPO in two ways: one bath MPS per electronic state, and one conventional
MPS containing the electronic site and every oscillator. Run

```bash
python examples/multiset_holstein.py --profile quick
```

to write both population matrices, root-mean-square displacements, and bond
histories to `multiset_holstein.npz`. The adjacent JSON file reports wall times,
peak bonds, and the maximum population difference. The `paper-scale` profile
uses 31 electronic sites, but is intended as a long calculation whose Fock
dimension, time step, threshold, chain length, and boundaries still require
independent convergence checks.

## When it helps

The ansatz can reduce the bond needed inside any one bath wavepacket when the
ordinary MPS spends much of its bond resolving correlations between a small
system and several distinct environmental responses. A smaller reported bond
does not by itself mean a cheaper calculation: there are $d_S$ component MPSs
and $d_S^2$ Hamiltonian blocks. Compare wall time, component bonds, and physical
observables with a conventional MPS calculation.

The number of sets is the full system-basis dimension. It is therefore natural
for an $N$-level impurity or a single-excitation electronic model, but becomes
exponential if a many-site tensor-product system is expanded without a symmetry
restriction. The present high-level implementation is limited to the
single-system `SystemBath` model. A multi-set tree tensor network would require
coupled tangent-space sweeps for that tree manifold; relabeling the existing
tree TEBD engine would not implement the method.

## Convergence checks

Compare at least:

1. `dt` and `dt/2`;
2. `trunc_eps` and a tighter value;
3. the local Fock dimension;
4. multi-set and conventional MPS dynamics on their common time interval.

For polaron representations, also check the frequency-domain dependence of the
finite displacement and the inverse transformation used for laboratory-frame
observables. See {doc}`/tutorials/convergence` for the common bath-resolution
checks.
