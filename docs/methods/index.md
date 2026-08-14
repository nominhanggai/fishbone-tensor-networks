# Representations and propagation methods

A method selects one point in four public axes:

```text
model -> representation -> geometry -> integrator
```

Use `method="..."` as a shorthand, or pass the axes directly. Every call returns
the same {py:class}`~fishbonett.models.result.Result` contract.

Method names are representation-explicit. A path method is named
`<representation>-<integrator>`; a non-path tensor tree inserts `tree`, as in
`interaction-chain-tree-tdvp2`. Thus `polaron-chain-tdvp2` states both the
polaron-chain representation and the TDVP2 integrator.

Older engine-first labels are rejected with a precise replacement rather than
accepted as silent aliases. For example, `polaron-tdvp2` reports that its new name
is `polaron-chain-tdvp2`.

## The six representations

Each name below is complete. There is no second public category to combine with
it.

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
then spreads through the chain modes. Diagonalizing a finite chain can generate
equivalent star data, but that is a discretization technique rather than the
definition of the interaction representation.

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

| representation | geometry | methods | details |
|---|---|---|---|
| `schrodinger-chain` | path | `schrodinger-chain-tdvp1`, `schrodinger-chain-tdvp2`, `schrodinger-chain-dtdvp` | {doc}`schrodinger/chain` |
| `schrodinger-star` | path | `schrodinger-star-tdvp1`, `schrodinger-star-tdvp2` | {doc}`schrodinger/star_mpo` |
| `interaction-chain` | path | `interaction-chain-tebd`, `interaction-chain-trotter-mpo`, `interaction-chain-tdvp1`, `interaction-chain-tdvp2` | {doc}`interaction/tebd`, {doc}`interaction/trotter_mpo`, {doc}`interaction/star_mpo` |
| `interaction-chain` | binary tree | `interaction-chain-tree-tdvp1`, `interaction-chain-tree-tdvp2`, `interaction-chain-tree-tebd` | {doc}`interaction/tree` |
| `interaction-star` | path | `interaction-star-tdvp1`, `interaction-star-tdvp2` | {doc}`interaction/star_mpo` |
| `polaron-chain` | path | `polaron-chain-tebd`, `polaron-chain-tdvp1`, `polaron-chain-tdvp2`, `polaron-chain-dtdvp` | {doc}`schrodinger/polaron_chain` |
| `polaron-star` | path | `polaron-star-tdvp1`, `polaron-star-tdvp2`, `polaron-star-dtdvp` | {doc}`schrodinger/polaron_chain` |

The two `interaction-chain` rows use the same Hamiltonian on different tensor
graphs. The representation supplies the requested numerical product, while the
integrator determines how that product advances the tensor state.

## Other models

| model | representation | method |
|---|---|---|
| `multichannel` | `schrodinger-star` | `schrodinger-star-tree-tebd` |
| `multichannel` | `interaction-chain` | `interaction-chain-tebd` |
| `multichannel` | `interaction-star` | `interaction-star-tebd` |
| `comb`, `site-tree` | `schrodinger-chain` | `schrodinger-chain-tree-tebd` |

For multichannel interaction propagation, `interaction-star-tebd` retains the
shared star modes and `interaction-chain-tebd` applies a common orthogonal
star-to-chain transform to the matrix-valued mode couplings.

## Choosing an integrator

- Start with a bond-growing method: `interaction-chain-tree-tdvp2`,
  `interaction-chain-tebd`, or a two-site TDVP
  method.
- One-site TDVP methods require an explicit `bond_dim`; they cannot grow a bond
  from a product state.
- Time-dependent interaction representations rebuild their numerical operator at
  every step midpoint.
- The conditional-displacement method `interaction-chain-trotter-mpo` is
  available because the
  mode coupling terms commute after the free bath has been removed.
- Polaron methods require finite
  $\int J(\omega)/\omega^2\,d\omega$ and careful convergence in local Fock
  dimension.

```python
result = model.run(
    dt=0.02,
    t_max=2.0,
    representation="interaction-chain",
    geometry="path",
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
schrodinger/polaron_chain
interaction/tebd
interaction/trotter_mpo
interaction/star_mpo
interaction/tree
interaction/multichannel
```
