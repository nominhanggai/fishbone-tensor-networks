# Architecture

The public decomposition is:

```text
model -> representation -> tensor-network geometry -> integrator
```

- A **model** defines the physical topology and system operators.
- A **representation** defines the mathematical Hamiltonian and materializes its
  supported numerical products: a TDVP MPO, Trotter MPO, or TEBD gates.
- A **tensor-network geometry** is the state ansatz: 1D MPS, binary tree tensor
  network, or a general tree tensor network.
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

Dependencies point downward. In particular, the six public representation
builders do not import TEBD, TDVP, MPO drivers, or tensor-network state classes.
The planner requests the numerical product required by the resolved integrator
directly from the representation.

## Bath discretization and the interaction representation

The finite star is the starting point for the interaction construction:

1. discretize the continuous bath into independent star modes
   $(\omega_k,g_k,a_k)$;
2. take the interaction representation with respect to
   $H_B=\sum_k\omega_k a_k^\dagger a_k$;
3. retain $a_k$ for `interaction-star`, or apply the star-to-chain transform
   $b_n=\sum_k U_{nk}a_k$ for `interaction-chain`.

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

`Bath.coupling` is deprecated. Use `SystemBath(coupling=...)` for single-system
models and `bath.bind(operator)` for multi-site models.

## Numerical products of a representation

One mathematical representation can support several propagation algorithms. For
example, `interaction-chain` supplies time-dependent Hamiltonian tensors through
`tdvp_mpo(t)`, interval gates through `tebd_gates(t, dt)`, and its exact
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
| `binary-tree` | `tree` | `interaction-chain-tree-tdvp2` |
| `tree` | `tree` | `schrodinger-chain-tree-tebd` |

Both tree geometries take the same infix, so a name is **not** in general
derivable from the axes: `binary-tree` is one system's bath modes on a balanced
tree, `tree` is the comb / general site tree, and the axes do not distinguish
them in the name. Where that collides, the name breaks the tie:

- `interaction-chain-tree-tebd` is the `binary-tree` method, so the `comb`
  method with the same representation and integrator is named
  **`interaction-chain-fishbone-tebd`**. This is the only such exception.

`tests/unit/test_models.py::test_every_method_name_is_derivable_from_its_axes`
pins both the rule and that single exception, so a second collision has to be
named deliberately rather than resolved ad hoc.

All high-level model classes execute through `SimulationPlan`. `run(seed=...)`
also scopes randomized linear algebra to that plan without changing NumPy's
global random generator.
