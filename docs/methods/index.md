# Representations and propagation methods

A method selects one point in four public axes:

```text
model -> representation -> state_geometry -> integrator
```

Use `method="..."` as a shorthand, or pass the axes directly. Every call returns
the same {py:class}`~fishbonett.models.result.Result` contract.

Method names are representation-explicit. A 1D MPS method is named
`<representation>-<integrator>`; a tree tensor-network method inserts `tree`, as in
`interaction-chain-tree-tebd`. Thus `polaron-chain-tdvp2` states both the
polaron-chain representation and the TDVP2 integrator.

## The six representations

| representation | Hamiltonian structure | time dependence |
|---|---|---|
| `schrodinger-star` | independent star modes coupled directly to the system | static |
| `schrodinger-chain` | nearest-neighbour chain with the system coupled to $c_0$ | static |
| `interaction-star` | free-star evolution absorbed into $g_k e^{-i\omega_k t}$ | time dependent |
| `interaction-chain` | interaction-star coupling transformed star-to-chain | time dependent |
| `polaron-star` | per-star-mode Lang--Firsov displacement | static |
| `polaron-chain` | reweighted star-to-chain transform localizes displacement on $c_0$ | static |

### Interaction construction order

The interaction representations start from a finite star discretization:

$$
H_B=\sum_k\omega_k a_k^\dagger a_k,
\qquad c_k(t)=g_k e^{-i\omega_k t}.
$$

`interaction-star` retains the $a_k$. `interaction-chain` then applies
$b_n=\sum_k U_{nk}a_k$, giving

$$
d_n(t)=\sum_k U_{nk}g_k e^{-i\omega_k t}.
$$

At $t=0$, $d_n(0)$ is localized on $c_0$ for the usual Lanczos transform and
then spreads through the chain modes. Diagonalizing a finite chain is an
alternative way to generate equivalent star data.

### Polaron star and chain

The Lang--Firsov transformation is naturally defined mode by mode in the star:

$$
U_p=\exp\!\left[O\otimes\sum_k
\frac{g_k}{\omega_k}(a_k^\dagger-a_k)\right].
$$

`polaron-star` retains those individual displacements. `polaron-chain` applies a
star-to-chain transform for the reweighted measure $J(\omega)/\omega^2$, which
localizes the collective displacement on the first chain mode. Both describe the
same transformed Hamiltonian and both are implemented.

## System-bath methods

| representation | tensor-network geometry | methods | details |
|---|---|---|---|
| `schrodinger-chain` | 1D MPS | `schrodinger-chain-tdvp1`, `schrodinger-chain-tdvp2`, `schrodinger-chain-dtdvp` | {doc}`schrodinger/chain` |
| `schrodinger-star` | 1D MPS | `schrodinger-star-tdvp1`, `schrodinger-star-tdvp2` | {doc}`schrodinger/star_mpo` |
| `interaction-chain` | 1D MPS | `interaction-chain-tebd`, `interaction-chain-trotter-mpo`, `interaction-chain-tdvp1`, `interaction-chain-tdvp2` | {doc}`interaction/tebd`, {doc}`interaction/trotter_mpo`, {doc}`interaction/star_mpo` |
| `interaction-chain` | binary tree tensor network | `interaction-chain-tree-tebd` | {doc}`interaction/tree` |
| `interaction-star` | 1D MPS | `interaction-star-tdvp1`, `interaction-star-tdvp2` | {doc}`interaction/star_mpo` |
| `polaron-chain` | 1D MPS | `polaron-chain-tebd`, `polaron-chain-tdvp1`, `polaron-chain-tdvp2`, `polaron-chain-dtdvp` | {doc}`polaron` |
| `polaron-star` | 1D MPS | `polaron-star-tdvp1`, `polaron-star-tdvp2`, `polaron-star-dtdvp` | {doc}`polaron` |

The two `interaction-chain` rows use the same Hamiltonian on different tensor
graphs. The representation supplies the requested numerical product, while the
integrator determines how that product advances the tensor state.

## Other models

| model | representation | tensor-network geometry | method |
|---|---|---|---|
| `multichannel` | `schrodinger-star` | star tensor network (`tree`) | `schrodinger-star-tree-tebd` |
| `multichannel` | `interaction-chain` | 1D MPS (`mps`) | `interaction-chain-tebd` |
| `multichannel` | `interaction-star` | 1D MPS (`mps`) | `interaction-star-tebd` |
| `comb` | `schrodinger-chain` | comb tensor network (`tree`) | `schrodinger-chain-tree-tebd` |
| `comb` | `interaction-chain` | comb tensor network (`tree`) | `interaction-chain-fishbone-tebd`, `interaction-chain-fishbone-trotter-mpo`, `interaction-chain-fishbone-tdvp2` |
| `site-tree` | `schrodinger-chain` | arbitrary tree tensor network (`tree`) | `schrodinger-chain-tree-tebd` |

For multichannel interaction propagation, `interaction-star-tebd` retains the
shared star modes and `interaction-chain-tebd` applies a common orthogonal
star-to-chain transform to the matrix-valued mode couplings.

## Choosing an integrator

- Start with a bond-growing method: `interaction-chain-tree-tebd`,
  `interaction-chain-tebd`, or a two-site TDVP
  method.
- One-site TDVP methods require an explicit `bond_dim`; they cannot grow a bond
  from a product state.
- Time-dependent interaction representations rebuild their numerical operator at
  every step midpoint.
- The conditional-displacement method `interaction-chain-trotter-mpo` is
  available for a single coupling channel because its mode terms commute after
  the free bath has been removed.
- Polaron methods require finite
  $\int J(\omega)/\omega^2\,d\omega$ and careful convergence in local Fock
  dimension.

```python
result = model.run(
    dt=0.02,
    t_max=2.0,
    representation="interaction-chain",
    state_geometry="mps",
    integrator="tdvp2",
    bond_dim=100,
)

# Equivalent shorthand:
result = model.run(
    dt=0.02, t_max=2.0, method="interaction-chain-tdvp2", bond_dim=100
)
```

See {doc}`/architecture` for the representation/evolution boundary and
{py:func}`fishbonett.models.registry.describe_taxonomy` for the complete runtime
table.

```{toctree}
:hidden:

schrodinger/chain
schrodinger/star_mpo
polaron
interaction/tebd
interaction/trotter_mpo
interaction/star_mpo
interaction/tree
interaction/multichannel
```
