# Exciton with independent local baths

`ExcitonBath` represents a single excitation shared by (N) electronic levels,
with one independent harmonic bath coupled to each site population:

$$
H_S=\sum_{ij}h_{ij}|i\rangle\langle j|,
\qquad
H_{SB}=\sum_i |i\rangle\langle i|\otimes B_i.
$$

The (N\times N) matrix `h` contains site energies and electronic couplings.
`baths[i]` defines (B_i); different entries may use different spectral
densities, temperatures, Fock dimensions, and mode counts.

```python
import numpy as np

from fishbonett import Bath, ExcitonBath

h = np.array([
    [0.30, -0.12, 0.02],
    [-0.12, 0.10, 0.08],
    [0.02, 0.08, 0.00],
])

def density(omega):
    return 0.08 * omega**2 * np.exp(-omega / 2.0)

baths = [
    Bath(J=density, domain=(0.2, 8.0), n_modes=12, phys_dim=6)
    for _ in range(3)
]
model = ExcitonBath(h, baths)
```

## State layouts and propagators

Every method below uses the same `interaction-chain` Hamiltonian. The two
conventional MPS layouts offer five propagators; the multi-set layouts currently
use two-site TDVP.

| `state_geometry` | site layout | available integrators |
|---|---|---|
| `system-first-mps` | one (N)-level site, then the modes of bath 1, bath 2, … | `tebd`, `trotter-mpo`, `tdvp1`, `tdvp2`, `dtdvp` |
| `interleaved-mps` | electronic site 1, its modes, electronic site 2, its modes, … | `tebd`, `trotter-mpo`, `tdvp1`, `tdvp2`, `dtdvp` |
| `multi-set-mps` | one bath MPS for every electronic basis state | `tdvp2` |
| `multi-set-tree` | one branched bath TTN for every electronic basis state | `tdvp2` |

For the conventional layouts, replace the final component of the method name.
For example, the system-first family is
`interaction-chain-system-first-tebd`,
`interaction-chain-system-first-trotter-mpo`,
`interaction-chain-system-first-tdvp1`,
`interaction-chain-system-first-tdvp2`, and
`interaction-chain-system-first-dtdvp`. The interleaved family follows the same
pattern.

For example:

```python
result = model.run(
    dt=0.02,
    t_max=1.0,
    method="interaction-chain-multi-set-tdvp2",
    initial=0,                 # excitation initially on electronic level 0
    trunc_eps=1e-4,
    bond_dim=None,
)

populations = result.expect["population"]  # shape (n_records, 3)
electronic_rdm = result.rdm                 # shape (n_records, 3, 3)
```

The system-first layout keeps the single-excitation restriction directly: its
first physical leg has dimension (N). The interleaved layout uses a local
two-level occupation site for each electronic level. Its MPO conserves total
excitation number, and the initial-state builder places the MPS in the
one-excitation sector. `result.rdm[i,j]` is reconstructed from

$$
\rho_{ij}=\langle \sigma_j^+\sigma_i^-\rangle.
$$

The multi-set tree removes the electronic physical sites from each component
tree. Dimension-one connector nodes retain a low-degree backbone, and every
bath chain branches from the connector associated with its electronic level.
The electronic index remains outside the TTNs and is evolved by the coupled
tree-TDVP equations.

The propagators realize the interaction Hamiltonian differently:

- TEBD applies two-site system--mode gates. Reversible swaps bring each coupled
  pair together and restore the original MPS ordering after the gate sequence.
- Trotter-MPO applies an electronic half-step, the interval-integrated
  conditional-displacement MPO, and a second electronic half-step.
- TDVP1 evolves on a fixed-bond manifold, TDVP2 grows bonds through two-site
  sweeps, and dTDVP expands the one-site tangent space adaptively. TDVP1 and
  dTDVP therefore require an explicit `bond_dim` ceiling.

All ten conventional-MPS methods return a checkpoint. Resolve the bath for the
complete intended horizon, then continue a shorter segment without restarting
the time-dependent interaction coefficients:

```python
first = model.run(
    dt=0.02,
    n_steps=25,
    bath_horizon=1.0,
    method="interaction-chain-interleaved-tdvp2",
    trunc_eps=1e-4,
)
second = model.run(
    dt=0.02,
    n_steps=25,
    resume=first.checkpoint,
    method=first.method,
    trunc_eps=1e-4,
)
```

## Choosing a layout

`system-first-mps` has one compact electronic site and an MPO bond controlled by
the number of independent baths. `interleaved-mps` places each coupling beside
its bath, but electronic hopping crosses the intervening mode blocks. The two
multi-set layouts truncate each conditional bath wavepacket separately; their
Hamiltonian action contains up to (N^2) electronic blocks, so a smaller
reported tensor bond does not guarantee a shorter run.

Compare layouts on the same finite bath before increasing the time horizon. A
useful comparison records populations, coherences, wall time, and retained bonds.
The seven-site example does this for an FMO Hamiltonian:

```bash
python examples/fmo_state_layouts.py --profile smoke
python examples/fmo_state_layouts.py --profile smoke --layouts system-first interleaved multi-set multi-set-tree
```

The multi-set tree has the largest per-step contraction cost for a dense
electronic Hamiltonian. Use its smoke calculation to verify the layout first,
then increase `n_modes`, the horizon, and the local Fock dimension separately.

## Convergence

For each chosen layout, compare:

1. `dt` and `dt/2`;
2. `trunc_eps` and a tighter threshold;
3. the Fock dimension of every bath;
4. the bath mode count and frequency domain;
5. at least two propagators and two state layouts over a common time interval.

See {doc}`/tutorials/convergence` for bath correlation and light-cone checks.
