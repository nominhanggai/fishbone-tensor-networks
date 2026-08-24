# Interaction representations — MPO + TDVP

The 1D MPS with MPO/TDVP supports both interaction representations:

| representation | methods |
|---|---|
| `interaction-chain` | `interaction-chain-tdvp1`, `interaction-chain-tdvp2` |
| `interaction-star` | `interaction-star-tdvp1`, `interaction-star-tdvp2` |

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

Finite-chain diagonalization is one available route to equivalent finite star
data; the interaction construction itself begins with the star modes above.

## Integrators

- The `tdvp1` variants use one-site TDVP and require an explicit `bond_dim`.
- The `tdvp2` variants use two-site TDVP and grow bonds according to
  `trunc_eps`, optionally capped by `bond_dim`.
- Because $H_I(t)$ is time dependent, the MPO and its environments are
  refreshed every step.

## Example

```python
chain = model.run(
    dt=0.02, t_max=2.0, method="interaction-chain-tdvp2",
    bond_dim=100, trunc_eps=1e-5,
)

star = model.run(
    dt=0.02, t_max=2.0, method="interaction-star-tdvp2",
    bond_dim=100, trunc_eps=1e-5,
)
```

The two trajectories should converge to the same laboratory observables. Their
tensor-network cost can differ because their time-dependent coupling vectors are
distributed differently.

## Low-level interface

```python
from fishbonett.evolve.tdvp import run_mpo_hamiltonian
from fishbonett.representations.interaction import InteractionRepresentation

rep = InteractionRepresentation(
    representation="interaction-star",
    h_sys=H,
    coupling=O,
    bath=bath,
).build()

t, rdm, max_bond = run_mpo_hamiltonian(
    rep, initial=initial_state,
    dt=0.02, nsteps=100, sweep="tdvp2", bond_dim=100
)
```

The representation supplies `tdvp_mpo(t)` when the driver requests the
time-dependent Hamiltonian; the last call chooses the TDVP sweep.
