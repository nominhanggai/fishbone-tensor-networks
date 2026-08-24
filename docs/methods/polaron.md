# Polaron-star and polaron-chain

The package implements both static Lang--Firsov representations.

| representation | methods |
|---|---|
| `polaron-chain` | `polaron-chain-tebd`, `polaron-chain-tdvp1`, `polaron-chain-tdvp2`, `polaron-chain-dtdvp` |
| `polaron-star` | `polaron-star-tdvp1`, `polaron-star-tdvp2`, `polaron-star-dtdvp` |

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
Laboratory coherences also require undoing the displacement during measurement;
the high-level API performs this automatically and returns a laboratory-system
RDM for both star and chain.

## Numerical products

- `polaron-chain` supplies local nearest-neighbour gates for TEBD.
- Both representations supply a static TDVP MPO for one-site, two-site, and
  dynamically adaptive TDVP.

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

Converge both the tensor bond and the local Fock dimension. A coherent
displacement with mean occupation comparable to the Fock cutoff can appear
normalized while still giving inaccurate observables.
