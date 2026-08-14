# Architecture

The public decomposition is:

```text
model -> representation -> encoding -> state geometry -> integrator
```

- A **model** defines the physical topology and system operators.
- A **representation** defines the mathematical Hamiltonian.
- An **encoding** turns that Hamiltonian into local terms, gates, an MPO, or a
  factorized propagator.
- A **state geometry** is the tensor graph: path, balanced mode tree, or comb.
- An **integrator** advances the encoded operator and state.

Only `model`, `representation`, `geometry`, and `integrator` are public selection
axes. Encoding is an implementation boundary chosen by the resolved method.

## Dependency direction

```text
Bath + System / site graph
        |
        v
CoupledBath                         physical system--bath association
        |
        v
StarBath / ChainBath / PolaronBath  immutable finite bath data
        |
        v
representation                     mathematical transformation of H
        |
        v
encoding                           LocalTerms / MPO / gates / factorized U
        |
        v
SimulationPlan                     prepared state + propagation + measurement
        |
        v
tensor state + integrator          TEBD / TDVP on a path or tree
        |
        v
Result                             times, RDMs, expectations, bond diagnostics
```

Dependencies point downward. In particular, the six public representation
builders do not import TEBD, TDVP, MPO drivers, or tensor-network state classes.
The adapters in `fishbonett.encodings` may consume representation data, and the
planner may combine an encoding with an integrator. The exploratory
`SystemBathCoolingChain` predates this boundary and remains a stateful,
low-level compatibility utility outside `method=` dispatch.

## Bath compilation and the interaction representation

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
- {py:class}`~fishbonett.bath.compiled.StarBath` and
  {py:class}`~fishbonett.bath.compiled.ChainBath` own operator-free finite
  coefficients. {py:class}`~fishbonett.bath.compiled.PolaronBath` contains both
  star and chain data for the reweighted $J(\omega)/\omega^2$ measure plus the
  reorganization energy.
- `fishbonett.representations` owns Hamiltonian transformations, time-dependent
  coefficients, transformed initial states, and recovery of laboratory
  observables.
- `fishbonett.encodings` owns engine-facing forms: MPOs, local terms, swap gates,
  static polaron gates, and conditional-displacement propagators.
- {py:class}`~fishbonett.models.simulation.SimulationPlan` owns orchestration for
  one resolved method.
- `fishbonett.evolve` advances tensors and does not discretize a bath or select a
  representation.

`Bath.coupling` remains a deprecated compatibility input. `SystemBath(coupling=...)`
is authoritative, and multi-site models should receive `bath.bind(operator)`.

## Why encodings are separate

One mathematical representation can support several propagation algorithms. For
example, `interaction-chain` can be encoded as swap-network gates, a
conditional-displacement MPO, a time-dependent Hamiltonian MPO, or a tree
operator. None of those choices changes the represented Hamiltonian.

The separation is checked structurally through protocols in
{py:mod}`fishbonett.encodings.capabilities`. A plan fails during compilation when
an engine receives an incompatible encoding.

## Dispatch boundary

{py:data}`fishbonett.models.registry.METHODS` is the single table that maps a
public method to `(model, representation, geometry, integrator)` and an internal
engine key. {py:data}`fishbonett.models.simulation.PLAN_COMPILERS` maps only those
engine keys to preparation code. Physical model classes do not maintain duplicate
method tables.

All high-level model classes execute through `SimulationPlan`. `run(seed=...)`
also scopes randomized linear algebra to that plan without changing NumPy's
global random generator.
