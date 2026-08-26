# Architecture

The public decomposition is:

```text
model -> representation -> tensor-network geometry -> integrator
```

- A **model** defines the physical topology and system operators.
- A **representation** defines the mathematical Hamiltonian and materializes its
  supported numerical products: a TDVP MPO, Trotter MPO, or TEBD gates.
- A **tensor-network geometry** is the state ansatz: 1D MPS, binary tree tensor
  network, or a tree tensor network.
- An **integrator** advances the represented operator and state.

These are the four public selection axes.

## Dependency direction

```text
Bath + System / site graph
        |
        v
CoupledBath                         physical system--bath association
        |
        v
representation                     bath discretization + transformation of H
                                   + numerical products
        |
        v
SimulationPlan                     prepared state + propagation + measurement
        |
        v
tensor state + integrator          TEBD / TDVP on a 1D MPS or tree tensor network
        |
        v
Result                             times, RDMs, expectations, bond diagnostics
```

Dependencies point downward. In particular, the five public representation
builders do not import TEBD, TDVP, MPO drivers, or tensor-network state classes.
The planner requests the numerical product required by the resolved integrator
directly from the representation.

## Bath discretization and the interaction representation

The finite star is the starting point for the interaction construction:

1. discretize the continuous bath into independent star modes
   $(\omega_k,g_k,a_k)$;
2. take the interaction representation with respect to
   $H_B=\sum_k\omega_k a_k^\dagger a_k$;
3. apply the star-to-chain transform
   $b_n=\sum_k U_{nk}a_k$ to obtain `interaction-chain`.

Thus

$$
c_k(t)=g_k e^{-i\omega_k t},\qquad
d_n(t)=\sum_k U_{nk}g_k e^{-i\omega_k t}.
$$

Diagonalizing a finite chain is one numerical way to obtain equivalent finite
star data. It is not the conceptual definition of `interaction-chain`; the
defining order is star discretization, interaction transformation, then
star-to-chain transformation.

## Ownership rules

- {py:class}`~fishbonett.bath.spec.Bath` owns environment physics and numerical
  resolution: spectral densities, temperature, domain, mode count, and local
  Fock dimension.
- {py:class}`~fishbonett.bath.coupled.CoupledBath` binds that environment to one
  or more system-space operators. Multiple operators are channels sharing the
  same modes.
- `fishbonett.representations` accepts a resolved `Bath` and owns its finite
  star or chain coefficients, Hamiltonian transformations, time-dependent
  coefficients, transformed initial states, recovery of laboratory observables,
  and the `tdvp_mpo`, `trotter_mpo`, and `tebd_gates` products supported by each
  representation. Private coefficient containers remain internal.
- {py:class}`~fishbonett.models.simulation.SimulationPlan` owns orchestration for
  one resolved method.
- `fishbonett.evolve` advances tensors and does not discretize a bath or select a
  representation.

`SystemBath(coupling=...)` owns the operator for a single-system model. Multi-site
models take explicit `bath.bind(operator)` values at the attached sites.

## Numerical products of a representation

One mathematical representation can support several propagation algorithms. For
example, `interaction-chain` supplies time-dependent Hamiltonian tensors through
`tdvp_mpo(t)`, interval gates through `tebd_gates(t, dt)`, and its
conditional-displacement propagator through `trotter_mpo(t, dt)`. These are
products of the same represented Hamiltonian.

Representations build operators but do not advance tensor states. Evolution
engines consume those products without discretizing baths or selecting a
representation.

## Dispatch boundary

{py:data}`fishbonett.models.registry.METHODS` is the single table that maps a
public method to its compatible models, representation, tensor-network geometry,
integrator and
internal engine key. {py:data}`fishbonett.models.simulation.PLAN_COMPILERS` maps
only those engine keys to preparation code. Physical model classes do not maintain
duplicate method tables.

Method names are `<representation>-<infix>-<integrator>`, where the infix is
fixed by the tensor-network geometry:

| `state_geometry` | infix | example |
|---|---|---|
| `mps` | *(none)* | `polaron-chain-tdvp2` |
| `system-first-mps` | `system-first` | `interaction-chain-system-first-tdvp2` |
| `interleaved-mps` | `interleaved` | `interaction-chain-interleaved-tdvp2` |
| `multi-set-mps` | `multi-set` | `polaron-chain-multi-set-tdvp2` |
| `multi-set-tree` | `multi-set-tree` | `interaction-chain-multi-set-tree-tdvp2` |
| `binary-tree` | `tree` | `interaction-chain-tree-tebd` |
| `tree` | `tree` | `schrodinger-chain-tree-tebd` |

The system-first and interleaved MPS layouts also support `tebd`,
`trotter-mpo`, `tdvp1`, and `dtdvp` by replacing `tdvp2` in the example name.

The `binary-tree` and `tree` rows use `tree` in ordinary method names. The binary-tree method
`interaction-chain-tree-tebd` and the comb method would otherwise collide, so
the interaction-chain comb family uses the `fishbone` infix:
`interaction-chain-fishbone-tebd`,
`interaction-chain-fishbone-trotter-mpo`, and
`interaction-chain-fishbone-tdvp2`.

All high-level model classes execute through `SimulationPlan`. The plan scopes
the `svd_backend` policy and `run(seed=...)` to one propagation without changing
NumPy's global random generator. Adaptive randomized truncation keys each sketch
to its matrix, making the random choice independent of checkpoint segmentation;
the returned result records decomposition and fallback counts in `meta["svd"]`.
