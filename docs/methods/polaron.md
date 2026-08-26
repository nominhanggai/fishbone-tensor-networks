# Polaron-star and polaron-chain

The package implements both static Lang--Firsov representations.

| representation | methods |
|---|---|
| `polaron-chain` | `polaron-chain-tebd`, `polaron-chain-tdvp1`, `polaron-chain-tdvp2`, `polaron-chain-a1tdvp` |
| `polaron-star` | `polaron-star-tdvp1`, `polaron-star-tdvp2`, `polaron-star-a1tdvp` |
| independent-bath `Fishbone` | `polaron-chain-tree-tebd` |

## Theory

For

$$
H=H_S+O\otimes\sum_k g_k(a_k+a_k^\dagger)
  +\sum_k\omega_ka_k^\dagger a_k,
$$

define

$$
U_p=\exp\!\left[O\otimes\sum_k
\frac{g_k}{\omega_k}(a_k^\dagger-a_k)\right].
$$

The transformed Hamiltonian contains a free bath, a reorganization shift, and a
system Hamiltonian dressed by conditional displacements. `polaron-star` keeps
the individual displacements $g_k/\omega_k$. For `polaron-chain`, the package
constructs the finite star for $J(\omega)/\omega^2$ and applies its star-to-chain
transform. The collective displacement is then localized on $c_0$, while the
free bath becomes nearest-neighbour.

Both require finite
$\int J(\omega)/\omega^2\,d\omega$ over the selected domain.

## State and observables

The physical product state becomes a conditional coherent state after the
transformation. The representation prepares that transformed state explicitly.
Physical coherences also require undoing the displacement during measurement;
the high-level API performs this automatically and returns the physical system
RDM for both star and chain.

## Numerical products

- `polaron-chain` supplies local nearest-neighbour gates for TEBD.
- Both representations supply a static TDVP MPO for one-site, two-site, and
  dynamically adaptive TDVP. The adaptive method uses the full-QR one-site
  expansion and convergence test described in {doc}`schrodinger/chain`; it is
  not a two-site SVD under a different representation name.
- On `Fishbone`, `polaron-chain-tree-tebd` applies one independent
  Lang--Firsov transformation per coupled system site. Local dressed terms stay
  on the system--first-mode edge. A dressed electronic coupling is applied as a
  path MPO spanning its two endpoints and their first chain modes.

```python
chain = model.run(
    dt=0.02, t_max=4.0, method="polaron-chain-tdvp2",
    bond_dim=80, trunc_eps=1e-5,
)

star = model.run(
    dt=0.02, t_max=4.0, method="polaron-star-tdvp2",
    bond_dim=80, trunc_eps=1e-5,
)
```

For a multi-site single-excitation model, bind one independent bath to each
coupled site and select all four axes explicitly:

```python
import numpy as np

from fishbonett import Bath, Fishbone

exciton_hamiltonian = np.array([[0.3, 0.1], [0.1, 0.0]])
occupation = np.diag([0.0, 1.0])

def make_bath():
    return Bath(
        J=lambda w: 0.03 * w**3 * np.exp(-w / 4),
        domain=(0.05, 30),
        n_modes=24,
        phys_dim=8,
    )

fishbone = Fishbone.from_single_excitation(
    exciton_hamiltonian,
    baths={site: make_bath().bind(occupation)
           for site in range(len(exciton_hamiltonian))},
)
result = fishbone.run(
    dt=0.01,
    t_max=2.0,
    representation="polaron-chain",
    state_geometry="tree",
    integrator="tebd",
    trunc_eps=1e-4,
    bond_dim=None,
)
```

The multi-site path currently accepts at most one independent bath per system
site. `polaron-star` is not yet available for `Fishbone`. Both restrictions are
reported by method resolution or plan construction. Gapless Ohmic spectra remain
invalid for a full Lang--Firsov transform because their displacement norm
diverges.

Each dressed electronic edge gate is exponentiated on its two system endpoints
and their first chain modes, then factored into a path MPO. Its dense compilation
cost therefore grows with the product of those four local dimensions. This is
usually modest for two-level sites, but it makes Fock-dimension convergence a
memory consideration as well as an accuracy check.

Converge both the tensor bond and the local Fock dimension. A coherent
displacement with mean occupation comparable to the Fock cutoff can appear
normalized while still giving inaccurate observables.
