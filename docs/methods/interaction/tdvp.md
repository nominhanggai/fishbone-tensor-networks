# Interaction-chain [TDVP]

The 1D MPS supports `interaction-chain-tdvp1` and
`interaction-chain-tdvp2`. Both rebuild a time-dependent Hamiltonian MPO at
the step midpoint. The representation supplies that MPO; TDVP is an independent
choice of integrator.

## Construction

First discretize the bath into star modes:

$$
H=H_S+\sum_k\omega_k a_k^\dagger a_k
 +O\otimes\sum_k g_k(a_k+a_k^\dagger).
$$

Taking the interaction representation with respect to the free star bath gives
coefficients $g_k e^{-i\omega_k t}$. The star-to-chain transform
$b_n=\sum_kU_{nk}a_k$ is then applied, producing

$$
d_n(t)=\sum_kU_{nk}g_ke^{-i\omega_kt}.
$$

Finite-chain diagonalization is one route to equivalent finite star data; the
conceptual order remains star discretization, interaction transformation, then
star-to-chain transformation.

## Integrators

- `tdvp1` uses one-site TDVP and requires an explicit `bond_dim`.
- `tdvp2` uses two-site TDVP and grows bonds according to `trunc_eps`, optionally
  capped by `bond_dim`.
- Because the represented Hamiltonian is time dependent, its MPO and
  environments are refreshed every step.

## Example

```python
result = model.run(
    dt=0.02,
    t_max=2.0,
    method="interaction-chain-tdvp2",
    bond_dim=100,
    trunc_eps=1e-5,
)
```

## Low-level interface

```python
from fishbonett.evolve.tdvp import run_mpo_hamiltonian
from fishbonett.representations.interaction import InteractionRepresentation

representation = InteractionRepresentation(
    representation="interaction-chain",
    h_sys=H,
    coupling=O,
    bath=bath,
).build()

t, rdm, max_bond = run_mpo_hamiltonian(
    representation,
    initial=initial_state,
    dt=0.02,
    nsteps=100,
    sweep="tdvp2",
    bond_dim=100,
)
```

The representation supplies `tdvp_mpo(t)` when the driver requests the
Hamiltonian; the final call chooses the TDVP sweep.
