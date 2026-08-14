# Interaction-chain — swap-network TEBD

`method="interaction-chain-tebd"` propagates `interaction-chain` on a path MPS. The represented
Hamiltonian couples the system to every chain mode, so a swap network moves the
system site past the modes and back during each symmetric step.

## Theory

The construction is:

1. discretize the bath as finite star modes $(\omega_k,g_k,a_k)$;
2. take the interaction representation with respect to
   $\sum_k\omega_k a_k^\dagger a_k$;
3. transform the resulting coupling from star to chain modes.

The coefficient on chain mode $n$ is

$$
d_n(t)=\sum_kU_{nk}g_ke^{-i\omega_kt}.
$$

The free bath is absent from $H_I(t)$, so there are no mode--mode terms. The
Hamiltonian interaction graph is a star centered on the system even though the
state tensors are stored on a path. That graph mismatch, rather than the
definition of the representation, is why this method uses swaps.

## Algorithm

Each second-order step builds interval-integrated two-site Hamiltonians,
materializes their gates, sweeps the system outward, applies the far gate without a swap,
and sweeps back with the reversed gate ordering. `trunc_eps` controls each bond
split and `bond_dim` is an optional cap.

```python
result = model.run(
    dt=0.02,
    t_max=2.0,
    method="interaction-chain-tebd",
    trunc_eps=1e-5,
    bond_dim=100,
)
```

At the low level, the representation materializes its own gates:

```python
from fishbonett.representations.interaction import InteractionRepresentation

rep = InteractionRepresentation(
    representation="interaction-chain",
    h_sys=H,
    coupling=O,
    compiled_star=compiled_star,
).build()
forward, swapped = rep.tebd_gates(t=0.0, dt=0.01)
```

The same representation supplies `tdvp_mpo(t)` for TDVP or
`trotter_mpo(t, dt)` for the exact conditional-displacement step.
