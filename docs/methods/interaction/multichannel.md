# Multichannel shared-mode bath

The `multichannel` model couples one finite bath to several system operators:

$$
H_{SB}=\sum_k\left(\sum_c g_{ck}O_c\right)(a_k+a_k^\dagger).
$$

Because every channel acts on the same modes, the fluctuations are
cross-correlated. Passing a list to `SystemBath(coupling=...)` selects this model.

## Representations

| method | representation | description |
|---|---|---|
| `schrodinger-star-tree-tebd` | `schrodinger-star` | static shared star, selected by default |
| `interaction-chain-tebd` | `interaction-chain` | free-star interaction transformation, then a common star-to-chain rotation |

The interaction couplings are matrix valued. If $A_k=\sum_cg_{ck}O_c$, then

$$
A_k(t)=A_ke^{-i\omega_kt}.
$$

A common orthogonal transform $Q$ then gives

$$
D_n(t)=\sum_kQ_{nk}A_ke^{-i\omega_kt}
$$

in `interaction-chain`. The transform changes the finite mode coordinates but
does not separate the shared channels into independent baths.

## Example

```python
model = SystemBath(
    h=H,
    coupling=[O_gap, O_transfer],
    bath=Bath(J=[J_gap, J_transfer], domain=(0.0, 30.0),
              n_modes=30, phys_dim=12),
)

static = model.run(dt=0.02, t_max=2.0)
ip_chain = model.run(dt=0.02, t_max=2.0, method="interaction-chain-tebd")
```

For a thermofield-discretized signed frequency grid, temperature is already
included in the finite star data and the interaction constructors do not apply a
second thermal factor.
