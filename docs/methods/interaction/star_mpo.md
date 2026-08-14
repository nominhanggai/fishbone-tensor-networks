# Interaction representations — MPO + TDVP

The MPO/TDVP path supports both interaction representations:

| representation | methods |
|---|---|
| `interaction-chain` | `mpo-ip-tdvp1`, `mpo-ip-tdvp2` |
| `interaction-star` | `mpo-ip-star-tdvp1`, `mpo-ip-star-tdvp2` |

Both use a time-dependent Hamiltonian MPO rebuilt at the step midpoint. The
representation supplies coefficients; TDVP is a later, independent choice.

## Construction

Discretize the bath into star modes:

$$
H=H_S+\sum_k\omega_k a_k^\dagger a_k
 +O\otimes\sum_k g_k(a_k+a_k^\dagger).
$$

Taking the interaction representation with respect to the free star bath gives

$$
H_I(t)=H_S+O\otimes\sum_k
\left[g_ke^{-i\omega_kt}a_k+g_ke^{i\omega_kt}a_k^\dagger\right].
$$

This is `interaction-star`. Applying the star-to-chain transform
$b_n=\sum_kU_{nk}a_k$ afterwards gives `interaction-chain` with

$$
d_n(t)=\sum_kU_{nk}g_ke^{-i\omega_kt}.
$$

The chain version therefore does not mean that a chain Hamiltonian must first be
diagonalized as a conceptual step. Finite-chain diagonalization is merely one
available route to equivalent finite star data.

## Integrators

- The `tdvp1` variants use one-site TDVP and require an explicit `bond_dim`.
- The `tdvp2` variants use two-site TDVP and grow bonds according to
  `trunc_eps`, optionally capped by `bond_dim`.
- Because $H_I(t)$ is time dependent, the encoded MPO and its environments are
  refreshed every step.

## Example

```python
chain = model.run(
    dt=0.02, t_max=2.0, method="mpo-ip-tdvp2",
    bond_dim=100, trunc_eps=1e-5,
)

star = model.run(
    dt=0.02, t_max=2.0, method="mpo-ip-star-tdvp2",
    bond_dim=100, trunc_eps=1e-5,
)
```

The two trajectories should converge to the same laboratory observables. Their
tensor-network cost can differ because their time-dependent coupling vectors are
distributed differently.

## Low-level separation

```python
from fishbonett.encodings.mpo import encode_interaction
from fishbonett.evolve.tdvp import run_mpo_hamiltonian
from fishbonett.representations.interaction import InteractionRepresentation

rep = InteractionRepresentation(
    [2] + [20] * 40,
    representation="interaction-star",
    h_sys=H,
    coupling=O,
    compiled_star=compiled_star,
).build()

mpo = encode_interaction(rep, initial_state)
t, rdm, max_bond = run_mpo_hamiltonian(
    mpo, dt=0.02, nsteps=100, sweep="tdvp2", D=100
)
```

The first object is the mathematical representation; the second is its MPO
encoding; the last call chooses TDVP.
